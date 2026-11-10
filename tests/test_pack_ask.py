"""memory/ask.py: the /pack-ask HTTP gate, the job queue, the worker subprocess, routing in execute, the local model
request, and the spoken form. Synthetic records, a fake Ollama, no devices or network."""

import datetime as dt
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
import types
from email.message import Message
from itertools import pairwise
from pathlib import Path
from unittest.mock import ANY, create_autospec, patch
from unittest.mock import call as mock_call

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import speech_fake
from liberational import config as cfg
from liberational import memory_log, segment_log
from liberational.assistant import voice
from liberational.assistant.voice_auth import Conversation
from liberational.memory import ask as p
from liberational.memory import history_qa as h
from liberational.memory import recall, research
from liberational.people import registry
from voices import write_voices

# ask._owner_http reads DASH_PASSWORD from the module that defines the handler's class (dashboard/server.py in
# production). Setting it here makes a remote request without a session fall through to Handler._authed.
DASH_PASSWORD = "synthetic-test-password"
QUESTION = {"question": "Explain gravity", "mode": "local"}
AGENT_URL = "http://127.0.0.1:9/v1/chat/completions"


class Handler:
    """The parts of the dashboard's request handler that ask.handle_http uses (ask.DashboardHandler)."""

    def __init__(self, body=None, headers=None, address="127.0.0.1", authenticated=True, command="POST", path=""):
        raw = body if isinstance(body, bytes) else json.dumps(QUESTION if body is None else body).encode()
        self.path = "/pack-ask/jobs" + path
        self.command = command
        self.client_address = (address, 12345)
        self.headers = Message()
        defaults = {"Host": "pack.test:8765", "Content-Type": "application/json", "Content-Length": str(len(raw))}
        for key, value in {**defaults, **(headers or {})}.items():
            self.headers[key] = value
        self.rfile = io.BytesIO(raw)
        self.replies = []
        self.authenticated = authenticated

    def _json(self, value, status=200):
        self.replies.append((status, value))

    def _authed(self):
        if not self.authenticated:
            self._json({"error": "Unauthorized"}, 401)
        return self.authenticated


def call(**kwargs):
    """(status, body) of the last reply handle_http gave."""
    handler = Handler(**kwargs)
    assert p.handle_http(handler)
    return handler.replies[-1]


def runner(**kwargs):
    return create_autospec(p.ask, **kwargs)


def worker_output(output):
    return patch.object(p.subprocess, "run", autospec=True, return_value=types.SimpleNamespace(stdout=output))


@pytest.fixture
def jobs(monkeypatch):
    manager = p.JobManager(runner(return_value={"status": "ok", "answer": "synthetic"}))
    monkeypatch.setattr(p, "JOBS", manager)
    yield manager
    # A queued question's worker writes its answer to history; finish before the test's stores close.
    manager.queue.join()


@pytest.fixture
def agent():
    with cfg.override({"agent": {"url": AGENT_URL}}):
        yield


@pytest.fixture
def panel():
    with patch(
        "liberational.assistant.ask_friends.ask", autospec=True, return_value={"status": "ok", "answer": "Panel"}
    ) as friends:
        yield friends


class FakeOllama:
    """urllib.request.build_opener's stand-in: records each request and answers with one chat reply."""

    def __init__(self):
        self.requests = []
        self.contents: list[str] = []  # the replies to give, in order, before the synthetic one
        self.thinking = ""  # a thinking model's reasoning, which Ollama returns apart from the answer

    def __call__(self, *handlers):
        return self

    def open(self, request, timeout):
        self.requests.append(request)
        content = "Synthetic answer"
        if self.contents:
            content = self.contents.pop(0)
        elif "short_answer" in json.loads(request.data)["messages"][0]["content"]:
            content = json.dumps({"short_answer": content, "explanation": ""})
        message = {"content": content, **({"thinking": self.thinking} if self.thinking else {})}
        timings = {"load_duration": 500_000_000, "prompt_eval_duration": 250_000_000, "total_duration": 2_000_000_000}
        if json.loads(request.data).get("stream"):  # one JSON object a line, five characters at a time
            pieces = [content[i : i + 5] for i in range(0, len(content), 5)]
            lines = [json.dumps({"message": {"content": piece}, "done": False}) for piece in pieces]
            lines.append(json.dumps({"message": {"content": ""}, "done": True, **timings}))
            return io.BytesIO(("\n".join(lines) + "\n").encode())
        return io.BytesIO(json.dumps({"message": message, **timings}).encode())

    def last(self):
        return json.loads(self.requests[-1].data)


@pytest.fixture
def ollama(monkeypatch):
    fake = FakeOllama()
    monkeypatch.setattr(p.urllib.request, "build_opener", fake)
    with cfg.override({"models": {"ollama": {"llm": "synthetic"}}}):  # no OpenClaw config: the fallback model
        yield fake


# The HTTP gate


@pytest.mark.parametrize(
    "request_kwargs",
    [
        pytest.param({"headers": {"Origin": "http://pack.test:8765"}, "address": "192.0.2.1"}, id="same-origin-http"),
        pytest.param({"headers": {"Origin": "https://pack.test:8765"}, "address": "192.0.2.1"}, id="same-origin-https"),
        pytest.param({"address": "127.0.0.1", "authenticated": False}, id="loopback-ipv4"),
        pytest.param({"address": "::1", "authenticated": False}, id="loopback-ipv6"),
    ],
)
def test_post_queues_the_question(jobs, request_kwargs):
    code, ack = call(**request_kwargs)
    assert (code, ack["status"], ack["question"]) == (202, "queued", "Explain gravity")


def body(**fields):
    return {"body": {**QUESTION, **fields}}


@pytest.mark.parametrize(
    "request_kwargs, code",
    [
        *[
            pytest.param({"headers": {"Origin": origin}}, 403, id="origin-" + origin)
            for origin in ("https://evil.test", "null", "https://pack.test:8765/other", "https://user@pack.test:8765")
        ],
        pytest.param({"headers": {"Sec-Fetch-Site": "cross-site"}}, 403, id="sec-fetch-cross-site"),
        pytest.param({"address": "192.0.2.1", "authenticated": False}, 401, id="remote-without-session"),
        pytest.param({"headers": {"X-Forwarded-For": "192.0.2.1"}, "authenticated": False}, 401, id="forwarded"),
        pytest.param({"headers": {"Content-Type": "text/plain"}}, 415, id="not-json"),
        pytest.param({"body": b"{broken"}, 400, id="broken-json"),
        *[pytest.param({"body": value}, 400, id=f"body-{value!r}") for value in ([], "text", 42)],
        *[
            pytest.param({"headers": {"Content-Length": length}}, 400, id="length-" + length)
            for length in ("0", "-1", "8193", "invalid")
        ],
        pytest.param({"body": {}}, 400, id="no-fields"),
        pytest.param(body(mode="cloud"), 400, id="mode-cloud"),
        pytest.param(body(question=[]), 400, id="question-list"),
        pytest.param(body(question=" "), 400, id="question-blank"),
        pytest.param(body(question="x" * 1201), 400, id="question-too-long"),
        *[
            pytest.param(body(scope=scope), 400, id=f"scope-{scope!r}")
            for scope in (None, "", "auto", "video", 1, [], {})
        ],
        pytest.param(body(scope="general", day="2026-10-20"), 400, id="general-with-day"),
        *[
            pytest.param(body(day=day), 400, id=f"day-{day!r}")
            for day in ("2026-02-30", "2026-1-1", "../../etc", 1, {}, "2026-10-20T00:00")
        ],
    ],
)
def test_post_is_refused_without_queueing(jobs, request_kwargs, code):
    assert call(**request_kwargs)[0] == code
    assert jobs.jobs == {}


def test_poll_unknown_id_is_not_found(jobs):
    assert call(command="GET", path="/" + "a" * 32)[0] == 404


@pytest.mark.parametrize(
    "fields, day, scope",
    [
        pytest.param({"day": "2026-10-20"}, "2026-10-20", "history", id="review-day"),
        pytest.param({"scope": "general"}, None, "general", id="general"),
    ],
)
def test_day_and_scope_reach_the_worker_subprocess(monkeypatch, fields, day, scope):
    output = {"status": "ok", "answer": "Synthetic answer"}
    manager = p.JobManager()
    monkeypatch.setattr(p, "JOBS", manager)
    with worker_output(json.dumps(output)) as run:
        code, ack = call(**body(**fields))
        manager.queue.join()
    assert (code, ack["day"], ack["scope"]) == (202, day, scope)
    status, job = call(command="GET", path="/" + ack["id"])
    assert (status, job["status"], job["day"], job["result"]) == (200, "completed", day, output)
    payload = json.loads(run.call_args.kwargs["input"])
    assert (payload["question"], payload["day"], payload["scope"]) == ("Explain gravity", day, scope)


# The job queue


def test_job_goes_queued_running_completed_and_get_returns_a_copy():
    entered, release = threading.Event(), threading.Event()

    def slow(*args):
        entered.set()
        assert release.wait(2), "test release missing"
        return {"status": "ok", "answer": "synthetic"}

    manager = p.JobManager(slow)
    try:
        job = manager.submit("Explain gravity", "local")
        assert job["status"] == "queued"
        assert entered.wait(2)
        assert manager.jobs[job["id"]]["status"] == "running"
    finally:
        release.set()
    manager.queue.join()
    result = manager.get(job["id"]) or {}
    assert result["status"] == "completed"
    result["result"]["answer"] = "mutated"
    assert manager.jobs[job["id"]]["result"]["answer"] == "synthetic"


@pytest.mark.parametrize(
    "outcome, error",
    [
        *[
            pytest.param({"status": status, "answer": "not completed"}, "not completed", id=status)
            for status in ("error", "timeout", "blocked_sensitive", "busy", "input_too_long", "synthesis_failed")
        ],
        pytest.param(RuntimeError("synthetic"), "Question processing failed (RuntimeError).", id="raises"),
    ],
)
def test_unsuccessful_runs_fail_the_job(outcome, error):
    manager = p.JobManager(runner(side_effect=[outcome]))
    job = manager.submit("Explain gravity", "local")
    manager.queue.join()
    result = manager.jobs[job["id"]]
    assert (result["status"], result["error"]) == ("failed", error)
    assert result["finished_at"] is not None


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param(p.ask, id="ask"),
        pytest.param(p.execute, id="execute"),
        pytest.param(lambda *a, **k: p.JobManager(runner()).submit(*a, **k), id="submit"),
    ],
)
@pytest.mark.parametrize(
    "mode, options",
    [
        pytest.param("cloud", {}, id="mode-cloud"),
        pytest.param("local", {"scope": "auto"}, id="scope-auto"),
        pytest.param("local", {"scope": None}, id="scope-none"),
        pytest.param("local", {"scope": "general", "day": "2026-10-20"}, id="general-with-day"),
    ],
)
def test_entry_points_reject_bad_mode_scope_and_day(entry, mode, options):
    with pytest.raises(ValueError):
        entry("Explain gravity", mode, **options)


# The worker subprocess


def test_worker_timeout_is_a_timeout_without_history_wording():
    with patch.object(p.subprocess, "run", autospec=True, side_effect=subprocess.TimeoutExpired("synthetic", 1)):
        result = p.ask("Explain gravity", scope="general")
    assert (result["status"], result["scope"]) == ("timeout", "general")
    assert "interval" not in result["answer"]


@pytest.mark.parametrize("stdout", ["not-json", "[]", "{}"])
def test_worker_bad_output_is_an_error(stdout):
    with worker_output(stdout):
        assert p.ask("hello")["status"] == "error"


def test_worker_and_cli_forward_general_scope(monkeypatch):
    monkeypatch.setattr(p.sys, "argv", ["pack-ask", "--worker"])
    monkeypatch.setattr(p.sys, "stdin", io.StringIO(json.dumps({**QUESTION, "scope": "general"})))
    with patch.object(p, "execute", autospec=True, return_value={"answer": "ok"}) as execute:
        p.main()
    execute.assert_called_once_with("Explain gravity", "local", None, day=None, scope="general")
    monkeypatch.setattr(p.sys, "argv", ["pack-ask", "--scope", "general", "--mode", "friends"])
    monkeypatch.setattr(p.sys, "stdin", io.StringIO("Explain gravity"))
    with patch.object(p, "ask", autospec=True, return_value={"answer": "ok"}) as ask:
        p.main()
    ask.assert_called_once_with("Explain gravity", "friends", str(p.STATE), scope="general")


# The local model request


def test_local_settings_reads_the_openclaw_primary_and_its_context(tmp_path):
    config = tmp_path / "config.json"
    defaults = {
        "model": {"primary": "ollama/synthetic-primary"},
        "models": {"ollama/synthetic-primary": {"params": {"num_ctx": 98304}}},
    }
    providers = {"ollama": {"models": [{"id": "synthetic-primary", "contextWindow": 65536}]}}
    config.write_text(json.dumps({"agents": {"defaults": defaults}, "models": {"providers": providers}}))
    assert p.local_settings(config) == ("synthetic-primary", 98304)


def ask_settings(model="", num_ctx=0):
    return cfg.override({"ask": {"model": model, "num_ctx": num_ctx}, "models": {"ollama": {"llm": "synthetic"}}})


@pytest.mark.parametrize(
    "model, num_ctx, expected",
    [
        pytest.param("", 0, ("synthetic", 4096), id="defaults-share-the-voice-model"),
        pytest.param("big", 0, ("big", 16384), id="own-model-own-context"),
        pytest.param("synthetic", 0, ("synthetic", 4096), id="shared-model-named-keeps-shared-context"),
        pytest.param("big", 8192, ("big", 8192), id="own-model-chosen-context"),
        pytest.param("", 16384, ("synthetic", 16384), id="shared-model-chosen-context"),
    ],
)
def test_the_ask_settings_choose_the_model_and_its_context(model, num_ctx, expected):
    with ask_settings(model, num_ctx):
        assert p.local_settings() == expected


def test_ask_model_wins_over_an_openclaw_default_that_is_not_local(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"agents": {"defaults": {"model": {"primary": "cloud/synthetic"}}}}))
    with ask_settings("big"):
        assert p.local_settings(config) == ("big", 16384)
    with pytest.raises(ValueError, match="not a local Ollama model"):
        p.local_settings(config)


@pytest.mark.parametrize("num_ctx", [-1, 2048])
def test_an_ask_context_too_small_is_refused(num_ctx):
    with ask_settings("big", num_ctx), pytest.raises(ValueError):
        p.local_settings()


@pytest.mark.parametrize(
    "model, num_ctx, sent",
    [
        pytest.param("", 0, ("synthetic", 4096, 1024, -1), id="shared-load-stays-loaded"),
        pytest.param("synthetic", 0, ("synthetic", 4096, 1024, -1), id="shared-load-named"),
        pytest.param("big", 0, ("big", 16384, 1400, "30m"), id="own-model-let-go-when-idle"),
        pytest.param("", 8192, ("synthetic", 8192, 1400, "30m"), id="own-context-let-go-when-idle"),
    ],
)
def test_the_answer_call_uses_the_ask_settings(ollama, model, num_ctx, sent):
    with ask_settings(model, num_ctx):
        p.local_answer("hello", "{}")
    payload = ollama.last()
    options = payload["options"]
    assert (payload["model"], options["num_ctx"], options["num_predict"], payload["keep_alive"]) == sent


@pytest.mark.parametrize(
    "model, num_ctx, budget",
    [
        pytest.param("", 0, 4416, id="default"),  # (4096 - 1024 - 1600) * 3
        pytest.param("", 16384, 40152, id="grows-with-context"),  # (16384 - 1400 - 1600) * 3
        pytest.param("big", 65536, 187608, id="65536"),  # (65536 - 1400 - 1600) * 3
        pytest.param("big", 131072, 192000, id="capped"),
    ],
)
def test_the_evidence_budget_grows_with_the_ask_context(model, num_ctx, budget):
    with ask_settings(model, num_ctx):
        assert p.evidence_budget() == budget


def test_a_thinking_answer_gets_room_to_think_and_keeps_the_reasoning_apart(ollama):
    ollama.thinking = "Line T1 says tomatoes."
    trace = {}
    with cfg.override({"ask": {"model": "thinker", "num_ctx": 16384, "think": True}}):
        assert p.local_answer("what did Dana plant", "{}", trace=trace) == ("Synthetic answer", "ollama/thinker")
        assert p.evidence_budget() == 27969  # (16384 - 16384 // 3 - 1600) * 3
    payload = ollama.last()
    assert (payload["think"], payload["options"]["num_predict"]) == (True, 5461)
    assert trace == {"thinking": "Line T1 says tomatoes.", "first_token_s": 0.75, "model_s": 2.0}


@pytest.mark.parametrize(
    "think, expected",
    [
        (False, {"num_ctx": 4096, "num_predict": 1024, "temperature": 0}),
        (
            True,
            {"num_ctx": 4096, "num_predict": 1365, "temperature": 0},
        ),
    ],
)
def test_instant_ask_sampling_stays_greedy_with_or_without_thinking(ollama, think, expected):
    with cfg.override({"ask": {"think": think}}):
        p.local_answer("hello", "{}")
    assert ollama.last()["options"] == expected


@pytest.mark.parametrize(
    "think, expected",
    [(False, {"temperature": 0}), (True, {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0})],
)
def test_research_thinking_payload_uses_sampling(ollama, think, expected):
    with cfg.override({"research": {"think": think, "model": "thinker"}}), research.deep():
        p.local_answer("hello", "{}")
    sampling = {k: v for k, v in ollama.last()["options"].items() if k in ("temperature", "top_p", "top_k", "min_p")}
    assert sampling == expected


def test_a_retried_answer_adds_up_its_trace(ollama):
    ollama.contents = [json.dumps({"short_answer": "Yes. No.", "explanation": ""})]
    trace = {}
    p.local_answer("was the door open", "{}", trace=trace)
    assert ollama.last()["think"] is False
    assert trace == {"first_token_s": 1.5, "model_s": 4.0, "thinking": ""}


def test_asks_own_ollama_gets_the_answer_and_keeps_it_loaded(ollama):
    with cfg.override({"ask": {"model": "synthetic", "num_ctx": 65536, "ollama_url": "127.0.0.1:11435"}}):
        p.local_answer("hello", "{}")
    assert ollama.requests[-1].full_url == "http://127.0.0.1:11435/api/chat"
    assert (ollama.last()["options"]["num_ctx"], ollama.last()["keep_alive"]) == (65536, -1)


def test_cpu_expert_answer_keeps_attention_on_the_dedicated_gpu(ollama):
    with cfg.override(
        {"ask": {"model": "expert", "num_ctx": 32768, "ollama_url": "localhost:11435", "cpu_expert_layers": 48}}
    ):
        p.local_answer("hello", "{}")
    assert ollama.requests[-1].full_url == "http://localhost:11435/api/chat"
    assert ollama.last()["options"] == {
        "num_ctx": 32768,
        "num_predict": 1400,
        "temperature": 0,
        "num_gpu": 999,
        "num_batch": 128,
    }


def test_research_cpu_experts_use_its_explicit_dedicated_server(ollama):
    with (
        cfg.override(
            {
                "ask": {"cpu_expert_layers": 48},
                "research": {"model": "expert-thinker", "num_ctx": 16384, "ollama_url": "localhost:11436"},
            }
        ),
        cfg.override(research.settings()),
    ):
        p.local_answer("hello", "{}")
    assert ollama.requests[-1].full_url == "http://localhost:11436/api/chat"
    assert ollama.last()["options"]["num_gpu"] == 999
    assert ollama.last()["options"]["num_batch"] == 128
    assert ollama.last()["model"] == "expert-thinker"


@pytest.mark.parametrize("layers", [0, 48])
def test_cpu_expert_profile_leaves_the_shared_voice_load_unchanged(ollama, layers):
    with cfg.override({"ask": {"cpu_expert_layers": layers}}):
        p.local_answer("hello", "{}")
    assert ollama.last()["options"] == {"num_ctx": 4096, "num_predict": 1024, "temperature": 0}


@pytest.mark.parametrize("server", ["", "http://localhost:11434", "localhost:11434/"])
def test_cpu_expert_answer_refuses_the_shared_server_before_a_request(ollama, server):
    with (
        cfg.override(
            {
                "ollama": {"url": "http://localhost:11434"},
                "ask": {"model": "expert", "ollama_url": server, "cpu_expert_layers": 48},
            }
        ),
        pytest.raises(ValueError, match="CPU expert.*dedicated"),
    ):
        p.local_answer("hello", "{}")
    assert ollama.requests == []


@pytest.mark.parametrize("think, seconds", [(False, 180), (True, 270)])  # 270: 80 s of rounds, 170 s, and 20
def test_a_thinking_job_gets_the_rounds_and_a_whole_thinking_answer(think, seconds):
    with cfg.override({"ask": {"think": think}}):
        assert p.deadline("local") == seconds
        assert p.deadline("friends") == 360


def test_an_openclaw_context_keeps_the_old_evidence_cap(tmp_path, monkeypatch):
    config = {"agents": {"defaults": {"model": {"primary": "ollama/synthetic-primary"}}}}
    monkeypatch.setattr(p.cfg, "openclaw_config", lambda: config)  # no contextWindow: 65536
    with ask_settings():
        assert p.local_settings() == ("synthetic-primary", 65536)
        assert p.evidence_budget() == 16000


@pytest.mark.parametrize(
    "short",
    [
        "Dr. Alice was recognized.",
        "Alice arrived at 3 p.m. yesterday.",
        "3 visits. [C1]",
        "2.5 minutes were observed.",
        "3 observed visits.",
        "2 or more recorded exchanges.",
        "12 minutes of sampled presence.",
        "14:30 was the last recorded sighting.",
        "Yes, the parcel is visible.",
        "No, the recorded statement rejects the proposal.",
        "Unknown from the available evidence.",
        "Alice and Dana were recognized.",
        "A blue parcel is on the table.",
        "The duration is unknown because coverage is incomplete.",
    ],
)
def test_structured_answer_keeps_specific_lead_and_evidence(short):
    detail = "Recorded observation [C1]. Coverage is partial; no continuous duration is established."
    assert p.render_answer(json.dumps({"short_answer": short, "explanation": detail})) == short + "\n\n" + detail


@pytest.mark.parametrize(
    "raw",
    [
        json.dumps({"short_answer": "First.Second.", "explanation": ""}),
        json.dumps({"short_answer": "Yes.” No.", "explanation": ""}),
        json.dumps({"short_answer": "# Answer", "explanation": ""}),
        "not JSON",
        "[]",
        '{"short_answer": 3, "explanation": "x"}',
        json.dumps({"short_answer": "First. Second.", "explanation": ""}),
        json.dumps({"short_answer": "word " * 26, "explanation": ""}),
    ],
)
def test_malformed_answer_does_not_promote_an_explanation_number(raw):
    with pytest.raises(ValueError):
        p.render_answer(raw)


@pytest.mark.parametrize(
    "short",
    [
        "The last utterance was 'Yeah.' from Speaker 6 at 13:33:42.",
        "Dana shouted 'Watch out!' near the door.",
        "Alice asked 'Is it raining?' before leaving.",
        'Alice said "Stop. Wait here." and left.',
        "Dana’s last line was “Okay!” at 14:02.",
        "Bob’s reply was ‘Sure.’ at noon.",
    ],
)
def test_quoted_line_ending_in_punctuation_is_one_sentence(short):
    detail = "Transcript T1."
    assert p.render_answer(json.dumps({"short_answer": short, "explanation": detail})) == short + "\n\n" + detail


@pytest.mark.parametrize(
    "short",
    [
        "Alice said 'Yeah.' Then Dana left.",
        "Dana's dog barked. Then it stopped.",
    ],
)
def test_second_sentence_after_a_quote_or_apostrophe_is_still_rejected(short):
    with pytest.raises(ValueError, match="one short sentence"):
        p.render_answer(json.dumps({"short_answer": short, "explanation": ""}))


def test_spelled_leading_quantity_is_numeric_without_mining_explanation():
    assert p.render_answer(
        json.dumps({"short_answer": "Three parcels were recorded.", "explanation": "One frame [C1]."})
    ).startswith("3 parcels")
    assert p.render_answer(
        json.dumps({"short_answer": "Unknown count.", "explanation": "Three sampled frames are not three entries."})
    ).startswith("Unknown count.")


def test_local_answer_posts_one_bounded_chat_request(ollama):
    assert p.local_answer("hello", "{}") == ("Synthetic answer", "ollama/synthetic")
    assert ollama.requests[-1].full_url == cfg.ollama_url() + "/api/chat"
    payload = ollama.last()
    assert {k: payload[k] for k in ("model", "keep_alive", "truncate", "think")} == {
        "model": "synthetic",
        "keep_alive": -1,
        "truncate": False,
        "think": False,
    }
    # Ollama 0.40.10 never answers a request with shift false: it hung every Ask.
    assert "shift" not in payload
    # The small fallback model at the context its other users keep it loaded with, so it isn't loaded twice.
    assert (payload["options"]["num_ctx"], payload["options"]["num_predict"]) == (4096, 1024)


@pytest.fixture
def pacific(monkeypatch):
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def prompt(ollama):
    return json.loads(ollama.last()["messages"][-1]["content"])


def test_history_prompt_gives_the_searched_window_beside_now(ollama, pacific):
    # 2026-11-04 13:07:05.4 to 13:37:05.4 PST, asked at 13:37:05.4
    query = {"question": "What was said in the last 30 minutes?", "start": 1793826425.4, "end": 1793828225.4}
    found = {"is_history": True, "answer": None, "evidence": EVIDENCE, "coverage": {}, "warnings": [], "query": query}
    with cfg.override({"recall": {"enabled": False}}), patch.object(h, "collect", autospec=True, return_value=found):
        result = p.execute("What was said in the last 30 minutes?", now=1793828225.4)
    assert result["status"] == "ok"
    content = prompt(ollama)
    assert content["window"] == {"start": "2026-11-04T13:07:05-08:00", "end": "2026-11-04T13:37:05-08:00"}
    assert list(content) == ["evidence", "window", "now", "question"]
    assert "window" in ollama.last()["messages"][0]["content"]


@pytest.mark.parametrize(
    "query",
    [
        pytest.param({}, id="no-query"),
        pytest.param({"start": None, "end": 1793828225.4}, id="open-start"),
    ],
)
def test_history_prompt_without_a_bounded_window_has_no_window(ollama, query):
    found = {"is_history": True, "answer": None, "evidence": EVIDENCE, "coverage": {}, "warnings": [], "query": query}
    with cfg.override({"recall": {"enabled": False}}), patch.object(h, "collect", autospec=True, return_value=found):
        p.execute("What was said yesterday?")
    assert "window" not in prompt(ollama)


def test_an_answer_that_breaks_the_format_is_asked_for_once_more(ollama):
    two = json.dumps({"short_answer": "You talked about lunch. Then the garden.", "explanation": "Lines [T1]."})
    one = json.dumps({"short_answer": "You talked about lunch and the garden.", "explanation": "Lines [T1]."})
    ollama.contents = [two, one]
    assert p.local_answer("what did we talk about today", "{}") == (
        "You talked about lunch and the garden.\n\nLines [T1].",
        "ollama/synthetic",
    )
    assert len(ollama.requests) == 2
    *_, said, told = ollama.last()["messages"]
    assert said == {"role": "assistant", "content": two}
    assert told["role"] == "user" and "must lead with one short sentence" in told["content"]


def test_an_answer_that_breaks_the_format_twice_fails(ollama):
    two = json.dumps({"short_answer": "Yes. No.", "explanation": ""})
    ollama.contents = [two, two]
    with pytest.raises(ValueError, match="one short sentence"):
        p.local_answer("was the door open", "{}")
    assert len(ollama.requests) == 2


def test_a_general_answer_is_never_asked_for_again(ollama):
    ollama.contents = ["First. Second."]
    assert p.local_answer("write a haiku", None) == ("First. Second.", "ollama/synthetic")
    assert len(ollama.requests) == 1


@pytest.mark.parametrize("num_ctx", [4096, 8192, 16384, 65536])
def test_the_evidence_and_the_reply_fit_the_context(num_ctx):
    evidence, reply = p.budgets(num_ctx)
    assert evidence // 3 + 1600 + reply <= num_ctx  # about 3 characters a token; the full prompt and question need room
    assert 2400 <= evidence <= 16000 and reply <= 1400


def test_general_questions_get_a_general_prompt(ollama):
    p.local_answer("Write a function", None)
    system, message = ollama.last()["messages"]
    assert message == {"role": "user", "content": "Write a function"}
    assert "Cite evidence IDs" not in system["content"]


def test_history_questions_get_structured_evidence_and_clock(ollama):
    p.local_answer("Write a function", "{}", FIXED)
    system, message = ollama.last()["messages"]
    assert json.loads(message["content"]) == {
        "evidence": {},
        "now": FIXED.astimezone().isoformat(),
        "question": "Write a function",
    }
    assert "Cite evidence IDs" in system["content"]


FIXED = dt.datetime(2026, 10, 23, 12, tzinfo=dt.UTC)


@pytest.mark.parametrize("now", [FIXED, FIXED.timestamp()], ids=["datetime", "timestamp"])
def test_answer_prompt_states_the_given_now(ollama, now):
    p.local_answer("what happened today", "{}", now)
    assert json.loads(ollama.last()["messages"][1]["content"])["now"] == FIXED.astimezone().isoformat()


# Routing in execute


@pytest.fixture
def no_records():
    with patch.object(
        h, "collect", autospec=True, side_effect=AssertionError("General questions must not read records")
    ):
        yield


def test_general_local_answer_reads_no_records(no_records):
    with patch.object(p, "local_answer", autospec=True, return_value=("General answer", "synthetic")) as local:
        result = p.execute("What happened yesterday?", scope="general")
    local.assert_called_once_with("What happened yesterday?", None)
    assert (result["status"], result["answer"], result["scope"], result["is_history"]) == (
        "ok",
        "General answer",
        "general",
        False,
    )
    assert (result["evidence"], result["coverage"], result["warnings"], result["query"]) == ([], {}, [], {})


@pytest.mark.parametrize("mode, subset", [("friends", "pair"), ("everyone", "all")])
def test_general_panel_gets_no_records(no_records, agent, panel, mode, subset):
    result = p.execute("What happened yesterday?", mode, scope="general")
    panel.assert_called_once_with("What happened yesterday?", context="", subset=subset)
    assert (result["status"], result["answer"], result["evidence"]) == ("ok", "Panel", [])


def test_general_friends_without_an_agent_falls_back_to_local(no_records, panel):
    with (
        cfg.override({"agent": {"url": ""}}),
        patch.object(p, "local_answer", autospec=True, return_value=("General fallback", "synthetic")) as local,
    ):
        result = p.execute("Explain gravity", "friends", scope="general")
    assert (result["scope"], result["mode"], result["answer"]) == ("general", "local", "General fallback")
    assert len(result["warnings"]) == 1
    local.assert_called_once_with("Explain gravity", None)
    panel.assert_not_called()


def test_general_model_failure_does_not_claim_retrieved_evidence():
    with patch.object(p, "local_answer", autospec=True, side_effect=RuntimeError("unavailable")):
        result = p.execute("Explain gravity", scope="general")
    assert (result["status"], result["error"]) == ("error", "RuntimeError")
    assert "evidence" not in result["answer"].lower()


@pytest.mark.parametrize("mode", p.MODES)
def test_recorded_answer_survives_every_mode_without_the_model(agent, panel, tmp_path, mode):
    answer = "2 observed transitions (not a verified outage total)."
    history = {"is_history": True, "answer": answer, "evidence": [], "coverage": {"partial": True}}
    with (
        patch.object(h, "collect", autospec=True, return_value=history) as collect,
        patch.object(p, "local_answer", autospec=True, side_effect=AssertionError("no model for a recorded answer")),
    ):
        result = p.execute("how many times did the lights go off", mode, state_dir=tmp_path, day="2026-10-20")
    collect.assert_called_once_with(
        "how many times did the lights go off", state_dir=tmp_path, now=None, day="2026-10-20"
    )
    assert (result["answer"], result["status"]) == (answer, "partial")


def test_evidence_context_keeps_what_fits_and_says_so():
    history = {"evidence": [{"id": "T1", "text": "x" * 5000}, {"id": "T2", "text": "small"}]}
    context, partial = p.evidence_context(history, 1000)
    assert partial
    assert json.loads(context)["input_selection"] == {"included": 1, "available": 2, "partial": True}


def test_evidence_context_keeps_the_video_method():
    row = {"id": "C1", "text": "A person", "source": "video", "method": "sampled recorded frames"}
    context, partial = p.evidence_context({"evidence": [dict(row, ts=1.0, link="/review")]}, 4000)
    assert json.loads(context)["evidence"] == [row]
    assert not partial


def test_execute_is_partial_when_the_model_saw_only_part_of_the_evidence():
    with (
        patch.object(
            h, "collect", autospec=True, return_value={"is_history": True, "evidence": [{"text": "x" * 20000}]}
        ),
        patch.object(p, "local_answer", autospec=True, return_value=("answer", "synthetic")),
    ):
        result = p.execute("What was said yesterday")
    assert result["status"] == "partial"
    assert any("input budget" in w for w in result["warnings"])


def test_panel_failure_is_not_masked_by_truncated_evidence(agent, panel):
    panel.return_value = {"status": "error", "answer": "Panel failed"}
    history = {"is_history": True, "evidence": [{"text": "x" * 20000}]}
    with patch.object(h, "collect", autospec=True, return_value=history):
        assert p.execute("What was said yesterday", "friends")["status"] == "error"


# The recall loop: on by default, off with recall.enabled or zero rounds, the single pass on failure


EVIDENCE = [{"id": "T1", "ts": 1.0, "text": "Dana booked the venue.", "source": "transcript"}]
SINGLE_PASS = [{"id": "T1", "text": "Dana booked the venue.", "source": "transcript"}]  # what the model is given
ROUNDS = [{"round": 1, "decision": "search", "tool": "journal", "found": 1, "new": ["J2"]}]


def history(**fields):
    query = {"question": "what did Dana say yesterday", "start": 0.0, "end": 86400.0}
    base = {"is_history": True, "answer": None, "evidence": EVIDENCE, "coverage": {}, "warnings": [], "query": query}
    return base | fields


@pytest.fixture
def run_recall(agent, panel):
    def run(found, settings, loop_error=None, mode="local"):
        loop_result = {
            "evidence": EVIDENCE + [{"id": "J2", "text": "Dana planned the party."}],
            "rounds": ROUNDS,
            "warnings": ["Journal entries are summaries."],
            "context": "LOOP CONTEXT",
            "truncated": False,
        }
        with (
            cfg.override({"recall": settings}),
            patch.object(h, "collect", autospec=True, return_value=found),
            patch.object(recall, "loop", autospec=True, return_value=loop_result, side_effect=loop_error) as loop,
            patch.object(p, "local_answer", autospec=True, return_value=("answer", "ollama/test")) as local,
        ):
            return p.execute("what did Dana say yesterday", mode, now=1000.0), loop, local

    return run


@pytest.mark.parametrize(
    "settings, ran",
    [
        pytest.param({}, True, id="on-by-default"),
        pytest.param({"enabled": False, "max_rounds": 3}, False, id="switched-off"),
        pytest.param({"max_rounds": 0}, False, id="zero-rounds"),
    ],
)
def test_recall_loop_switch(run_recall, settings, ran):
    result, _, _ = run_recall(history(), settings)
    assert ("rounds" in result) is ran


def test_single_pass_gives_the_model_the_collected_evidence(run_recall):
    _, _, local = run_recall(history(), {"max_rounds": 0})
    assert json.loads(local.call_args.args[1])["evidence"] == SINGLE_PASS
    assert local.call_args.kwargs["now"] == 1000.0  # the fixed now reaches the answer prompt


def test_loop_answers_from_its_own_context(run_recall):
    result, loop, local = run_recall(history(), {"max_rounds": 2})
    assert loop.call_args.args[1:] == (history(), p.STATE, 1000.0)
    assert local.call_args.args[1] == "LOOP CONTEXT"
    assert result["rounds"] == ROUNDS
    assert [e["id"] for e in result["evidence"]] == ["T1", "J2"]
    assert "Journal entries are summaries." in result["warnings"]
    assert (result["status"], result["answer"]) == ("ok", "answer")


def test_loop_failure_falls_back_to_the_single_pass(run_recall):
    result, _, local = run_recall(history(), {"max_rounds": 2}, loop_error=RuntimeError("boom"))
    assert json.loads(local.call_args.args[1])["evidence"] == SINGLE_PASS
    assert any("search rounds failed (RuntimeError)" in w for w in result["warnings"])
    assert "rounds" not in result
    assert result["answer"] == "answer"


@pytest.mark.parametrize(
    "found, mode",
    [
        pytest.param(history(answer="2 observed on-to-off transitions."), "local", id="recorded-answer"),
        pytest.param(history(), "friends", id="panel"),
        pytest.param({"is_history": False}, "local", id="not-history"),
        pytest.param(history(query={}, answer="Use one explicit YYYY-MM-DD date."), "local", id="unparsed"),
    ],
)
def test_loop_is_skipped(run_recall, found, mode):
    result, _, _ = run_recall(found, {"max_rounds": 2}, mode=mode)
    assert "rounds" not in result


@pytest.mark.parametrize(
    "rounds, answer", [pytest.param(0, h.NO_EVIDENCE, id="off"), pytest.param(2, "answer", id="on")]
)
def test_no_matching_evidence_searches_only_when_the_loop_is_on(run_recall, rounds, answer):
    result, _, _ = run_recall(history(evidence=[], answer=h.NO_EVIDENCE), {"max_rounds": rounds})
    assert result["answer"] == answer


def test_loop_end_to_end_with_a_journal(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    now = dt.datetime(2026, 10, 23, 12, tzinfo=h.TZ).timestamp()
    line = {"ts_start": now - 86400, "speaker": "Dana", "text": "The venue is booked."}
    speech_fake.fake().write([line])
    text = "Dana talked about the venue and the cake."
    dana = str(registry.at(state).ensure("Dana", "test"))
    record = {"ts_start": now - 86400 - 60, "ts_end": now - 86400 + 120, "text": text, "level": 180}
    memory_log.at(state).append(memory_log.JOURNAL, record, subjects=[dana])
    replies = [
        {"decision": "search", "tool": "journal", "args": {"text": "cake", "person": "Dana"}},
        {"decision": "answer"},
    ]

    def post(payload):  # the recall model's two calls: rank the hits, then decide
        if payload["format"] == recall.RANK_SCHEMA:
            return {"message": {"content": '{"ids": ["1"]}'}}
        return {"message": {"content": json.dumps(replies.pop(0))}}

    with (
        cfg.override({"recall": {"max_rounds": 3}}),
        patch.object(recall, "_post", post),
        patch.object(recall, "settings", autospec=True, return_value=("test-llm", 16384)),
        patch.object(
            p, "local_answer", autospec=True, return_value=("Dana booked the venue [T2].", "ollama/test")
        ) as local,
    ):
        result = p.execute("what did Dana say about the venue yesterday", state_dir=state, now=now)
    assert [e["id"] for e in result["evidence"]] == ["J1", "T2"]
    assert result["rounds"][0]["new"] == []  # the journal entry collect already found
    assert result["rounds"][1] == {"round": 2, "decision": "answer"}
    assert replies == []
    assert "The venue is booked." in local.call_args.args[1]


def test_person_question_flags_an_evidence_cap_as_partial(tmp_path, monkeypatch):
    now = dt.datetime(2026, 10, 23, 12, tzinfo=h.TZ).timestamp()
    lines = [("Sam", "I wanted to work on painting the fence."), ("Sam", "I also wanted to plant flowers.")]
    rows = [{"ts_start": now - 86400 + i, "speaker": who, "text": text} for i, (who, text) in enumerate(lines)]
    speech_fake.fake().write(rows)
    monkeypatch.setattr(h, "MAX_EVIDENCE", 1)
    result = h.collect("what all did Sam tell me she wanted to work on yesterday", tmp_path, now)
    assert result["coverage"]["truncated"]
    assert any("partial" in warning for warning in result["warnings"])


# The spoken form


@pytest.mark.parametrize(
    "answer, kept",
    [
        pytest.param("**2 observed transitions** [C1], not a verified outage total.", "not a verified outage total"),
        pytest.param("Dana booked the venue [J1][T2].", "Dana booked the venue"),
    ],
)
def test_voice_strips_citations_and_markup_but_keeps_the_words(answer, kept):
    out = p.spoken({"answer": answer})
    assert not re.search(r"\[[CJT]\d+\]|[*#`]", out)
    assert kept in out


@pytest.mark.parametrize(
    "answer, said",
    [
        pytest.param("Dana booked the venue [J1][T2].", "Dana booked the venue.", id="before-a-period"),
        pytest.param("Two transitions [C1], not verified.", "Two transitions, not verified.", id="before-a-comma"),
        pytest.param("Was it Sam [T1]?", "Was it Sam?", id="before-a-question-mark"),
    ],
)
def test_voice_leaves_no_space_where_a_citation_was(answer, said):
    assert p.spoken({"answer": answer}) == said


MORE = " Ask in the dashboard to see the full answer and sources."


@pytest.mark.parametrize(
    "answer, maximum, said",
    [
        pytest.param("aaaa bbbb. cccc dddd eeee", 12, "aaaa bbbb." + MORE, id="after-a-period"),
        pytest.param("aaaa bbbb cccc dddd", 10, "aaaa bbbb." + MORE, id="mid-sentence"),
        pytest.param("aaaa, bbbb cccc dddd", 7, "aaaa." + MORE, id="after-a-comma"),
        pytest.param("Is it on? yes no maybe", 12, "Is it on?" + MORE, id="after-a-question"),
    ],
)
def test_voice_clips_at_a_word_with_one_sentence_end(answer, maximum, said):
    assert p.spoken({"answer": answer}, maximum=maximum) == said


PARTIAL = "This answer is based on partial recorded evidence. "


@pytest.mark.parametrize(
    "result",
    [
        pytest.param({"answer": "A finding. " * 80 + "This is partial.", "status": "partial"}, id="status-clipped"),
        pytest.param({"answer": "Sam wanted to paint.", "coverage": {"partial": True}}, id="coverage-partial"),
        pytest.param({"answer": "Sam wanted to paint.", "coverage": {"truncated": True}}, id="truncated"),
    ],
)
def test_voice_says_partial_when_evidence_was_cut(result):
    assert p.spoken(result, maximum=200).startswith(PARTIAL)  # before the clip, so it is never cut off


@pytest.mark.parametrize(
    "warning", ["Coverage is incomplete.", "Captions were truncated upstream.", "A partial transcript exists."]
)
def test_voice_never_says_partial_from_a_warning_alone(warning):
    """The dashboard flags partial only when evidence was cut, so voice does too."""
    result = {"answer": "Sam wanted to paint.", "warnings": [warning], "coverage": {"partial": False}}
    assert p.spoken(result) == "Sam wanted to paint."


# Voice routing into ask (assistant/voice.handle; these belong with the voice tests)


@pytest.fixture
def owner_voice(tmp_path, monkeypatch):
    write_voices(tmp_path, {1: "Alice"})  # the owner's voice, by an owner tag
    monkeypatch.setattr(voice, "STATE_DIR", tmp_path)
    monkeypatch.setattr(voice, "CONVERSATION", Conversation(Path("/nonexistent-synthetic-verifier")))
    with cfg.override({"owner": {"name": "Alice"}}):
        yield


OWNER = {"speaker_id": 1, "speaker_sim": 0.9}


@pytest.mark.parametrize(
    "question",
    [
        "who was here last night",
        "who was here around 3 pm",
        "what did Dana say around 3",
        "who was here an hour ago",
        "what did Dana say between 2 and 4",
        "when was Alex last on camera 3",
        "how many times did the lights go off",
        "what all did Sam tell me she wanted to work on yesterday",
    ],
)
def test_spoken_history_questions_use_the_local_backend(owner_voice, monkeypatch, question):
    monkeypatch.setattr(voice, "optional_module", lambda name: None)
    monkeypatch.setattr(voice, "owner_answer", lambda *a, **k: pytest.fail("gateway must not be used"))
    speaker = create_autospec(voice.Speaker, instance=True)
    reply = {"status": "ok", "answer": "Synthetic historical answer", "coverage": {"partial": True}}
    with patch.object(p, "ask", autospec=True, return_value=reply) as ask:
        said = voice.handle(question, speaker, OWNER)
    ask.assert_called_once_with(question, mode="local", on_short=ANY)
    assert said == PARTIAL + "Synthetic historical answer"
    speaker.speak.assert_called_once_with(said)


@pytest.mark.parametrize(
    "prefix, mode",
    [
        ("Ask friends", "friends"),
        ("Ask your friends", "friends"),
        ("Ask everyone", "everyone"),
        ("Ask all friends", "everyone"),
        ("Ask all your friends", "everyone"),
    ],
)
def test_spoken_panel_prefix_picks_the_subset(owner_voice, monkeypatch, prefix, mode):
    monkeypatch.setattr(voice, "optional_module", lambda name: p)
    with patch.object(p, "ask", autospec=True, return_value={"status": "ok", "answer": "Synthetic answer"}) as ask:
        assert voice.handle(prefix + " why stars shine", entry=OWNER) == "Synthetic answer"
    ask.assert_called_once_with("why stars shine", mode=mode)


@pytest.mark.parametrize("mode", ["local", "friends", "everyone"])
def test_presence_answers_bypass_models_in_every_mode(tmp_path, monkeypatch, agent, mode, vision):
    at = dt.datetime(2026, 10, 20, 12, tzinfo=h.TZ).timestamp()
    vision.add(segment_log.WALKBY_COVERAGE, ts_start=at - 3600, ts_end=at, cam="cam0", frames=7200)
    attributes = {"direction": "left_to_right", "crossings": 1, "tracks": 1, "followed": True}
    vision.add(
        segment_log.WALKBY_SPANS, ts_start=at - 60, ts_end=at - 60, cam="cam0", kind="crossing", attributes=attributes
    )
    monkeypatch.setattr(
        p, "local_answer", create_autospec(p.local_answer, side_effect=AssertionError("no model counting"))
    )
    monkeypatch.setattr(recall, "loop", create_autospec(recall.loop, side_effect=AssertionError("no recall counting")))
    with (
        cfg.override({"presence": {"enabled": True}}),
        patch("liberational.assistant.ask_friends.ask", autospec=True, side_effect=AssertionError("no panel counting")),
    ):
        result = p.execute("how many times did someone walk by camera0 today?", mode, tmp_path, now=at + 1)
    assert result["model"] == "local-history-evidence"
    assert result["paths"] == ["presence"]
    assert result["answer"].startswith("People walked past the camera once.")
    assert "walked past the camera once" in p.spoken(result)
    assert "crossings" not in result


@pytest.mark.parametrize("mode", ["local", "friends", "everyone"])
def test_presence_read_failure_never_falls_back_to_models(tmp_path, monkeypatch, agent, mode, vision):
    monkeypatch.setattr(
        p, "local_answer", create_autospec(p.local_answer, side_effect=AssertionError("no model counting"))
    )
    monkeypatch.setattr(recall, "loop", create_autospec(recall.loop, side_effect=AssertionError("no recall counting")))

    def unreachable(stream: str, start: float, end: float) -> list:
        raise ConnectionRefusedError("broker down")

    monkeypatch.setattr(vision, "range", unreachable)
    with (
        cfg.override({"presence": {"enabled": True}}),
        patch("liberational.assistant.ask_friends.ask", autospec=True, side_effect=AssertionError("no panel counting")),
    ):
        result = p.execute("how many times did someone come into view today?", mode, tmp_path)
    assert result["status"] == "error"
    assert result["model"] == "local-history-evidence"
    assert result["paths"] == ["presence"]
    assert result["answer"] == "History source could not be read safely."


@pytest.mark.parametrize("mode", ["friends", "everyone"])
def test_presence_without_agent_does_not_claim_model_fallback(tmp_path, mode):
    with cfg.override({"presence": {"enabled": True}, "agent": {"url": ""}}):
        result = p.execute("how many times did someone come into view today?", mode, tmp_path)
    assert result["model"] == "local-history-evidence"
    assert not any("local model answered" in warning for warning in result["warnings"])
    assert result["mode"] == mode


@pytest.mark.parametrize("mode", ["local", "friends", "everyone"])
@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("clock", ["3", "3pm", "15:30"])
def test_presence_unsupported_time_refuses_without_panel_only_when_enabled(
    tmp_path, agent, panel, mode, enabled, clock
):
    with cfg.override({"presence": {"enabled": enabled}}):
        result = p.execute(f"how many times did someone come into view before {clock} today?", mode, tmp_path)
    if enabled:
        panel.assert_not_called()
        assert result["paths"] == ["presence"]
        assert result["model"] == "local-history-evidence"
        assert result["answer"].startswith("Unsupported time phrase;")
    else:
        if mode == "local":
            panel.assert_not_called()
            assert result["answer"].startswith("Unsupported time phrase;")
        else:
            panel.assert_called_once()
        assert "presence" not in result.get("paths", [])


def history_call(path="", **kwargs):
    handler = Handler(command=kwargs.pop("command", "GET"), **kwargs)
    handler.path = "/pack-ask/history" + path
    assert p.handle_http(handler)
    return handler, handler.replies[-1]


def test_history_http_reopens_completed_exchange_after_job_manager_restart(memory, monkeypatch):
    from liberational.memory import ask_history

    history = ask_history.at(p.STATE)
    ident = history.start("Earlier question", surface="review", day="2023-11-14")
    history.finish(
        ident,
        {"status": "ok", "answer": "Saved answer", "evidence": []},
        started_at=100,
        finished_at=101,
        status="completed",
    )
    monkeypatch.setattr(p, "JOBS", p.JobManager())
    _, (status, body) = history_call("?surface=review&day=2023-11-14")
    assert status == 200
    assert [entry["id"] for entry in body["items"]] == [ident]
    _, (status, body) = history_call("/" + ident)
    assert status == 200
    assert body["result"]["answer"] == "Saved answer"
    status, body = call(command="GET", path="/" + ident)
    assert status == 200
    assert body["status"] == "completed"
    assert body["result"]["answer"] == "Saved answer"


def test_history_delete_consumes_body_and_prevents_running_job_resurrection(memory, monkeypatch):
    from liberational.memory import ask_history

    entered, release = threading.Event(), threading.Event()

    def blocked(question, mode):
        entered.set()
        assert release.wait(5)
        return {"status": "ok", "answer": "Late answer", "evidence": []}

    manager = p.JobManager(blocked)
    monkeypatch.setattr(p, "JOBS", manager)
    job = manager.submit("Question", "local")
    try:
        assert entered.wait(5)
        handler, (status, body) = history_call("/" + job["id"], command="DELETE", body={})
        assert (status, body) == (200, {"deleted": 1})
        assert handler.rfile.read() == b""
    finally:
        release.set()
        manager.queue.join()
    assert manager.get(job["id"]) is None
    assert ask_history.at(p.STATE).get(job["id"]) is None
    assert call(command="GET", path="/" + job["id"])[0] == 404


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"address": "192.0.2.1", "authenticated": False}, 401),
        ({"command": "DELETE", "headers": {"Origin": "https://elsewhere.test"}}, 403),
        ({"command": "DELETE", "headers": {"Content-Type": "text/plain"}}, 415),
    ],
)
def test_history_http_rejects_unauthorized_reads_and_cross_origin_deletes(memory, kwargs, expected):
    _, (status, _) = history_call(**kwargs)
    assert status == expected


def test_history_filters_delete_only_matching_review_day(memory, jobs):
    from liberational.memory import ask_history

    history = ask_history.at(p.STATE)
    removed = history.start("Review question", surface="review", day="2023-11-14")
    retained = history.start("Live question", surface="live")
    _, response = history_call("?surface=review&day=2023-11-14", command="DELETE", body={})
    assert response == (200, {"deleted": 1})
    assert history.get(removed) is None
    assert history.get(retained) is not None


def test_saved_interrupted_job_finishes_polling_with_failure(memory, monkeypatch):
    from liberational.memory import ask_history

    ident = ask_history.at(p.STATE).start("Interrupted question")
    monkeypatch.setattr(p, "JOBS", p.JobManager())
    status, body = call(command="GET", path="/" + ident)
    assert status == 200
    assert body["status"] == "failed"
    assert "interrupted" in body["error"]


def test_completed_job_poll_respects_source_purge_instead_of_cached_answer(memory, monkeypatch):
    from liberational.memory import ask_history

    source_id = memory.append(memory_log.JOURNAL, {"ts_start": 100, "text": "Source words"})
    manager = p.JobManager(
        runner(
            return_value={
                "status": "ok",
                "answer": "Source words [J1]",
                "evidence": [{"id": "J1", "label": "J1", "interpretation": source_id, "ts": 100, "source": "journal"}],
            }
        )
    )
    monkeypatch.setattr(p, "JOBS", manager)
    job = manager.submit("What happened?", "local")
    manager.queue.join()
    assert call(command="GET", path="/" + job["id"])[1]["result"]["answer"] == "Source words [J1]"
    memory.purge([source_id], "test source removal")
    status, restored = call(command="GET", path="/" + job["id"])
    assert status == 200
    assert restored["result"] is None
    assert restored["answer_removed"]
    assert restored["status"] == "failed"
    saved = ask_history.at(p.STATE).get(job["id"])
    assert saved is not None
    assert saved["result"] is None


def test_completion_write_failure_keeps_live_answer_with_storage_warning(memory, monkeypatch):
    from liberational.memory import ask_history

    with patch.object(ask_history.History, "finish", autospec=True, side_effect=OSError("temporary write failure")):
        manager = p.JobManager(runner(return_value={"status": "ok", "answer": "Computed answer", "evidence": []}))
        monkeypatch.setattr(p, "JOBS", manager)
        job = manager.submit("Explain gravity", "local", scope="general")
        manager.queue.join()
    status, polled = call(command="GET", path="/" + job["id"])
    assert status == 200
    assert polled["status"] == "completed"
    assert polled["result"]["answer"] == "Computed answer"
    assert "could not be saved" in " ".join(polled["result"]["warnings"])
    saved = ask_history.at(p.STATE).get(job["id"])
    assert saved is not None
    assert saved["result"] is None


def test_completion_write_failure_does_not_bypass_later_root_or_source_purge(memory, monkeypatch):
    from liberational.memory import ask_history

    revisions = {"journal.entry": 0}
    monkeypatch.setattr(ask_history.History, "_revisions", lambda self: dict(revisions))
    source_id = memory.append(memory_log.JOURNAL, {"ts_start": 100, "text": "Private words"})
    result = {
        "status": "ok",
        "answer": "Private words [J1]",
        "evidence": [
            {"id": "J1", "interpretation": source_id, "ts": 100, "source": "journal"},
        ],
    }
    with patch.object(ask_history.History, "finish", autospec=True, side_effect=OSError("temporary write failure")):
        manager = p.JobManager(runner(return_value=result))
        monkeypatch.setattr(p, "JOBS", manager)
        job = manager.submit("What happened?", "local")
        manager.queue.join()
    assert call(command="GET", path="/" + job["id"])[1]["result"]["answer"] == "Private words [J1]"
    memory.purge([source_id], "test source removal")
    revisions["journal.entry"] = 1
    status, polled = call(command="GET", path="/" + job["id"])
    assert status == 200
    assert polled["status"] == "failed"
    assert polled["result"] is None
    ask_history.at(p.STATE).delete(job["id"])
    assert call(command="GET", path="/" + job["id"])[0] == 404


def test_unsaved_evidence_answer_is_hidden_when_deletion_checkpoint_is_unavailable(memory, monkeypatch):
    from liberational.memory import ask_history

    result = {"status": "ok", "answer": "Unverified words", "evidence": [{"source": "transcript", "ts": 100}]}
    with patch.object(ask_history.History, "finish", autospec=True, side_effect=OSError("write failure")):
        manager = p.JobManager(runner(return_value=result))
        monkeypatch.setattr(p, "JOBS", manager)
        job = manager.submit("What happened?", "local")
        manager.queue.join()
    status, polled = call(command="GET", path="/" + job["id"])
    assert status == 200
    assert polled["status"] == "failed"
    assert polled["result"] is None


# Streaming the instant answer


@pytest.mark.parametrize(
    "raw, expected",
    [
        ('{"short_answer":"A blue parcel."', "A blue parcel."),
        (' \n{ "short_answer" : "A blue parcel.",', "A blue parcel."),
        ('{"short_answer":"A blue parcel.","explanation":"still writing', "A blue parcel."),
        ('{"short_answer":"Three parcels.","explanation":""}', "3 parcels."),
        ('{"short_answer":"  A   blue parcel.  "}', "A blue parcel."),
        (
            '{"short_answer":"Dr. Alice recorded 3.5 parcels at 2 p.m. [T1]"}',
            "Dr. Alice recorded 3.5 parcels at 2 p.m. [T1]",
        ),
        (json.dumps({"short_answer": " ".join(["Parcel"] * 25)}), " ".join(["Parcel"] * 25)),
    ],
)
def test_completed_short_requires_only_a_complete_valid_first_field(raw, expected):
    assert p.completed_short(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "A blue parcel.",
        '"A blue parcel."',
        '[{"short_answer":"A blue parcel."}]',
        '```json\n{"short_answer":"A blue parcel."}',
        '{"nested":{"short_answer":"A blue parcel."}}',
        '{"explanation":"Earlier field.","short_answer":"A blue parcel."}',
        '{"text":"short_answer", "short_answer":"A blue parcel."}',
        '{"short_answer":null}',
        '{"short_answer":3}',
        '{"short_answer":{"text":"A blue parcel."}}',
        '{"short_answer":"A blue parcel.',
        '{"short_answer":"A blue \\',
        '{"short_answer":"A blue \\u12',
        '{"short_answer":"A blue \\q parcel."}',
        '{"short_answer":""}',
        '{"short_answer":"   "}',
        '{"short_answer":"First. Second."}',
        '{"short_answer":"Yes.” No."}',
        '{"short_answer":"# Answer"}',
        '{"short_answer":"- A parcel."}',
        json.dumps({"short_answer": " ".join(["Parcel"] * 26)}),
    ],
)
def test_completed_short_never_commits_partial_nested_or_invalid_answers(raw):
    assert p.completed_short(raw) is None


@settings(max_examples=50, deadline=None)
@given(
    st.text(alphabet=list('abcXYZ012"\\') + ["é", "中", "😀", "\n", "\t", "\r", "\b", "\f"], max_size=12),
    st.booleans(),
)
def test_completed_short_roundtrips_json_escapes_and_never_emits_a_truncated_prefix(fragment, ascii_only):
    short = "A " + fragment + " parcel."
    encoded = '{"short_answer":' + json.dumps(short, ensure_ascii=ascii_only)
    assert p.completed_short(encoded) == " ".join(short.split())
    assert all(p.completed_short(encoded[:cut]) is None for cut in range(len(encoded)))


def test_on_short_alone_streams_and_emits_once_before_explanation_finishes(ollama, monkeypatch):
    short = "A blue parcel."
    ollama.contents = [json.dumps({"short_answer": short, "explanation": "Observed beside the door [C1]."})]
    seen = []
    original_open = ollama.open

    class WatchedStream(io.BytesIO):
        def __iter__(self):
            content = ""
            while line := self.readline():
                content += json.loads(line).get("message", {}).get("content", "")
                if '"explanation"' in content:
                    assert seen == [short], "the completed short waited for explanation generation"
                yield line

    def opened(request, timeout):
        return WatchedStream(original_open(request, timeout).getvalue())

    monkeypatch.setattr(ollama, "open", create_autospec(ollama.open, side_effect=opened))
    answer, _ = p.local_answer("What was beside the door?", "{}", on_short=seen.append)
    assert ollama.last()["stream"] is True
    assert seen == [short]
    assert answer == short + "\n\nObserved beside the door [C1]."


def test_on_short_does_not_repeat_a_committed_answer_during_repair(ollama):
    ollama.contents = [
        '{"short_answer":"Three parcels.","explanation":7}',
        '{"short_answer":"Three parcels.","explanation":"Three sampled parcels [C1]."}',
    ]
    seen = []
    answer, _ = p.local_answer("How many parcels were recorded?", "{}", on_short=seen.append)
    assert len(ollama.requests) == 2
    assert seen == ["3 parcels."]
    assert answer.startswith("3 parcels.")


def test_on_short_rejects_a_final_answer_that_changes_the_committed_short(ollama):
    ollama.contents = [
        '{"short_answer":"Three parcels.","explanation":7}',
        '{"short_answer":"Four parcels.","explanation":"A changed count."}',
    ]
    seen = []
    with pytest.raises(ValueError, match="(?i)short|changed|commit"):
        p.local_answer("How many parcels were recorded?", "{}", on_short=seen.append)
    assert seen == ["3 parcels."]


def test_a_streamed_answer_is_passed_on_as_it_grows_and_ends_whole(ollama, monkeypatch):
    monkeypatch.setattr(p, "STREAM_EVERY_S", 0)
    reply = json.dumps({"short_answer": "Dana wants tomatoes.", "explanation": "She said so at lunch [T1]."})
    ollama.contents = [reply]
    seen: list[str] = []
    answer, _ = p.local_answer("what does Dana want", "{}", on_text=seen.append)
    assert ollama.last()["stream"] is True
    assert seen[-1] == reply
    assert all(b.startswith(a) for a, b in pairwise(seen))
    assert answer.startswith("Dana wants tomatoes.")


def test_an_answer_told_to_yield_part_way_is_stopped_and_asked_again(ollama, monkeypatch):
    monkeypatch.setattr(p, "STREAM_EVERY_S", 0)
    reply = json.dumps({"short_answer": "Dana wants tomatoes.", "explanation": "She said so at lunch [T1]."})
    ollama.contents = [reply, reply]
    told = [True]  # an instant question arrives during the first call, then is answered
    monkeypatch.setattr(p, "YIELD", lambda: bool(told) and told.pop())
    answer, _ = p.local_answer("what does Dana want", "{}")
    assert [json.loads(r.data)["stream"] for r in ollama.requests] == [True, True]
    assert answer.startswith("Dana wants tomatoes.")


def test_an_answer_asks_for_no_more_than_the_reply_cap(ollama, monkeypatch):
    monkeypatch.setattr(p, "MAX_REPLY_TOKENS", 100)
    p.local_answer("hello", "{}")
    assert ollama.last()["options"]["num_predict"] == 100


def test_an_unstreamed_answer_is_asked_for_whole(ollama):
    p.local_answer("hello", "{}")
    assert ollama.last()["stream"] is False


@given(st.text(max_size=80), st.text(max_size=80), st.integers(0, 400))
def test_the_answer_shown_while_writing_is_always_the_start_of_the_final_one(short, more, cut):
    raw = json.dumps({"short_answer": short, "explanation": more})
    final = p.partial_answer(raw)
    assert final == short + ("\n\n" + more if more else "")
    shown = p.partial_answer(raw[: min(cut, len(raw))])
    assert (final + "\n\n").startswith(shown) or final.startswith(shown)


def test_a_general_answer_is_shown_as_written():
    assert p.partial_answer("Gravity bends", history=False) == "Gravity bends"


def test_the_partial_file_holds_the_answer_so_far(tmp_path):
    path = tmp_path / "ask" / "partial" / "j1.txt"
    write = p.partial_writer(path, history=True)
    assert write is not None
    write('{"short_answer": "Dana wants tom')
    assert path.read_text() == "Dana wants tom"
    assert p.partial_writer(None, history=True) is None


def test_a_running_job_shows_its_partial_answer_and_the_file_goes_when_it_ends(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "STATE", tmp_path)
    started, release = threading.Event(), threading.Event()
    seen: dict[str, object] = {}

    def fake_result(question, mode, state_dir=None, day=None, scope="history", partial=None):
        seen["partial"] = partial
        assert partial is not None
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_text("Dana wants tom")
        started.set()
        release.wait(5)
        return {"status": "ok", "answer": "Dana wants tomatoes.", "evidence": []}

    monkeypatch.setattr(p, "_ask_result", fake_result)
    monkeypatch.setattr(p, "_history_start", lambda *a, **k: None)
    manager = p.JobManager()
    manager.runner = fake_result
    job = manager.submit("what does Dana want", "local")
    assert started.wait(5)
    running = manager.get(job["id"])
    assert running is not None and running["partial"] == "Dana wants tom"
    release.set()
    manager.queue.join()
    assert seen["partial"] == p.partial_path(job["id"])
    assert not p.partial_path(job["id"]).exists()
    finished = manager.get(job["id"])
    assert finished is not None and "partial" not in finished


def test_the_worker_subprocess_is_told_where_to_stream(monkeypatch, tmp_path):
    with worker_output(json.dumps({"status": "ok", "answer": "x"})) as run:
        p._ask_result("Explain gravity", "local", tmp_path, scope="general", partial=tmp_path / "j1.txt")
    assert json.loads(run.call_args.kwargs["input"])["partial"] == str(tmp_path / "j1.txt")


def test_camera_context_keeps_native_identity_and_its_name():
    history = {"evidence": [{"id": "C1", "camera": "cam0", "source": "camera", "text": "A violet mug."}]}
    with cfg.override({"vision": {"camera_names": ["cam0: Front camera"]}}):
        context, partial = p.evidence_context(history, 4000)
    assert not partial
    assert json.loads(context)["evidence"] == [
        {"id": "C1", "camera": "cam0", "camera_name": "Front camera", "source": "camera", "text": "A violet mug."}
    ]


def test_named_camera_none_does_not_attribute_transcript_rows_to_that_camera():
    history = {
        "evidence": [
            {"id": "C1", "camera": "None", "source": "camera", "text": "A violet mug."},
            {"id": "T2", "source": "transcript", "text": "The parcel arrived."},
        ]
    }
    with cfg.override({"vision": {"camera_names": ["None: Rear camera"]}}):
        context, partial = p.evidence_context(history, 4000)
    assert not partial
    camera, transcript = json.loads(context)["evidence"]
    assert camera["camera_name"] == "Rear camera"
    assert "camera" not in transcript and "camera_name" not in transcript


def test_history_evidence_reaches_the_model_as_structured_records(ollama):
    evidence = {"evidence": [{"id": "C1", "text": 'A card says "hello" beside a mug.', "camera": "study"}]}
    p.local_answer("What was beside the mug?", json.dumps(evidence), FIXED)
    user = json.loads(ollama.last()["messages"][1]["content"])
    assert user["evidence"] == evidence
    assert user["question"] == "What was beside the mug?"
    assert user["now"] == FIXED.astimezone().isoformat()


def test_history_system_and_evidence_prefix_stay_stable_when_question_time_changes(ollama):
    context = json.dumps({"evidence": [{"id": "C1", "text": "A violet mug on the desk."}]})
    p.local_answer("What was on the desk?", context, FIXED)
    first = ollama.last()["messages"]
    p.local_answer("What was on the desk?", context, FIXED + dt.timedelta(days=1))
    second = ollama.last()["messages"]
    assert first[0] == second[0]
    assert "A violet mug on the desk." in os.path.commonprefix([first[1]["content"], second[1]["content"]])
    assert json.loads(first[1]["content"])["now"] != json.loads(second[1]["content"])["now"]


def test_evidence_prefix_is_reusable_when_only_coverage_changes():
    evidence = [{"id": "T1", "text": "The parcel arrived beside the violet mug.", "source": "transcript"}]
    first, _ = p.evidence_context({"evidence": evidence, "coverage": {"window": "earlier"}}, 4000)
    second, _ = p.evidence_context({"evidence": evidence, "coverage": {"window": "later"}}, 4000)
    assert "The parcel arrived beside the violet mug." in os.path.commonprefix([first, second])
    assert json.loads(first)["coverage"] == {"window": "earlier"}
    assert json.loads(second)["coverage"] == {"window": "later"}


@pytest.mark.parametrize("context", ["Recorded text without JSON.", '"A quoted record."', "null", "42"])
def test_legacy_text_and_scalar_history_context_are_not_reinterpreted(ollama, context):
    p.local_answer("What happened?", context, FIXED)
    assert json.loads(ollama.last()["messages"][1]["content"])["evidence"] == context


@pytest.fixture
def short_worker(monkeypatch):
    """A real bounded child and worker protocol, with model execution replaced by a pipe-controlled fixture."""
    read_fd, write_fd = os.pipe()
    children = []
    original = subprocess.Popen
    program = """
import os, sys
from liberational.memory import ask
gate = int(sys.argv.pop(1))
def execute(*args, **kwargs):
    kwargs['on_short']({'answer': '3 visits.', 'status': 'partial', 'coverage': {'partial': True}})
    os.read(gate, 1)
    return {'answer': '3 visits.\\n\\nThe evidence is partial.', 'status': 'partial'}
ask.execute = execute
ask.main()
"""

    def start(command, **kwargs):
        child = original([sys.executable, "-c", program, str(read_fd), "--worker"], pass_fds=(read_fd,), **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(p.subprocess, "Popen", start)
    yield types.SimpleNamespace(children=children, release=lambda: os.write(write_fd, b"x"))
    os.close(read_fd)
    os.close(write_fd)


def test_short_worker_delivers_before_final_output_on_the_callers_thread(short_worker):
    seen = []
    caller = threading.get_ident()

    def ready(result):
        assert short_worker.children[0].poll() is None
        seen.append((threading.get_ident(), result))
        short_worker.release()

    output = p._spoken_worker(QUESTION, 10, ready)
    assert seen == [(caller, {"answer": "3 visits.", "status": "partial", "coverage": {"partial": True}})]
    assert json.loads(output)["answer"] == "3 visits.\n\nThe evidence is partial."


def test_short_worker_deadline_still_kills_the_child_during_speech(short_worker):
    entered = threading.Event()

    def blocked_speech(result):
        entered.set()
        assert short_worker.children[0].wait(timeout=5) != 0

    with pytest.raises(subprocess.TimeoutExpired):
        p._spoken_worker(QUESTION, 1, blocked_speech)
    assert entered.is_set()
    assert short_worker.children[0].poll() is not None


def test_short_speech_failure_kills_and_waits_for_the_worker(short_worker):
    def failed_speech(result):
        raise RuntimeError("speaker disconnected")

    with pytest.raises(RuntimeError, match="speaker disconnected"):
        p._spoken_worker(QUESTION, 10, failed_speech)
    assert short_worker.children[0].poll() is not None


def test_short_result_file_is_private_atomic_and_immutable(tmp_path):
    path = tmp_path / "short.json"
    write = p._short_writer(path)
    write({"answer": "3 visits."})
    write({"answer": "4 visits."})
    assert json.loads(path.read_text()) == {"answer": "3 visits."}
    assert path.stat().st_mode & 0o777 == 0o600
    assert not path.with_suffix(".tmp").exists()


def test_voice_speaks_short_with_coverage_before_the_explanation_is_ready(owner_voice, monkeypatch):
    monkeypatch.setattr(voice, "optional_module", lambda name: None)
    speaker = create_autospec(voice.Speaker, instance=True)
    short = {"answer": "3 visits.", "status": "partial", "coverage": {"partial": True}}
    prefix = "This answer is based on partial recorded evidence. 3 visits."

    def answer(question, **kwargs):
        kwargs["on_short"](short)
        speaker.speak.assert_called_once_with(prefix)
        kwargs["on_short"](short)
        return {**short, "answer": "3 visits.\n\nThe transcript supports those visits."}

    monkeypatch.setattr(p, "ask", answer)
    reply = voice.handle("how many times did the lights go off", speaker, OWNER)
    assert speaker.speak.call_args_list == [mock_call(prefix), mock_call("The transcript supports those visits.")]
    assert reply == prefix + " The transcript supports those visits."


def test_voice_does_not_replace_an_early_answer_when_the_explanation_fails(owner_voice, monkeypatch):
    monkeypatch.setattr(voice, "optional_module", lambda name: None)
    speaker = create_autospec(voice.Speaker, instance=True)

    def answer(question, **kwargs):
        kwargs["on_short"]({"answer": "3 visits.", "status": "ok", "coverage": {}})
        return {"status": "error", "answer": "The model could not complete the answer."}

    monkeypatch.setattr(p, "ask", answer)
    reply = voice.handle("how many times did the lights go off", speaker, OWNER)
    assert speaker.speak.call_args_list == [mock_call("3 visits."), mock_call("I could not complete the explanation.")]
    assert reply == "3 visits. I could not complete the explanation."


def test_failed_early_tts_does_not_attempt_a_second_reply(owner_voice, monkeypatch):
    monkeypatch.setattr(voice, "optional_module", lambda name: None)
    speaker = create_autospec(voice.Speaker, instance=True)
    speaker.speak.side_effect = RuntimeError("speaker disconnected")

    def answer(question, **kwargs):
        kwargs["on_short"]({"answer": "3 visits.", "status": "ok", "coverage": {}})
        pytest.fail("failed TTS must cancel the answer worker")

    monkeypatch.setattr(p, "ask", answer)
    with pytest.raises(RuntimeError, match="speaker disconnected"):
        voice.handle("how many times did the lights go off", speaker, OWNER)
    speaker.speak.assert_called_once_with("3 visits.")


def test_pronoun_note_precedes_both_early_speech_and_the_final_answer(monkeypatch, tmp_path):
    resolved = types.SimpleNamespace(
        question="What did Dana say yesterday?", note="By she, I mean Dana.", said="she", meant="Dana"
    )
    monkeypatch.setattr(p, "_pronoun", lambda *a: resolved)
    monkeypatch.setattr(p, "_history_start", lambda *a, **kw: None)
    monkeypatch.setattr(p, "_history_finish", lambda ident, result, *a: (dict(result), False))
    seen = []

    def worker(question, mode, state, day, scope, *, on_short):
        assert question == "What did Dana say yesterday?"
        short = {"answer": "Tea was requested.", "status": "ok", "coverage": {}}
        on_short(short)
        return {**short, "answer": "Tea was requested.\n\nThe transcript records that request."}

    monkeypatch.setattr(p, "_ask_result", worker)
    final = p.ask("What did she say yesterday?", state_dir=tmp_path, on_short=seen.append)
    assert seen[0]["answer"] == "By she, I mean Dana. Tea was requested."
    assert final["answer"] == "By she, I mean Dana. Tea was requested.\n\nThe transcript records that request."


@pytest.mark.parametrize(
    "coverage,status", [({}, "ok"), ({"partial": True}, "partial"), ({"truncated": True}, "partial")]
)
def test_short_event_carries_known_coverage_before_model_completion(tmp_path, monkeypatch, coverage, status):
    monkeypatch.setattr(h, "collect", lambda *a, **kw: {"is_history": True, "coverage": coverage, "evidence": []})
    seen = []

    def model(question, context, *, on_short):
        on_short("3 visits.")
        assert seen == [
            {
                "answer": "3 visits.",
                "status": status,
                "coverage": {
                    "partial": bool(coverage.get("partial")),
                    "truncated": bool(coverage.get("truncated")),
                },
            }
        ]
        return "3 visits.\n\nThe evidence supports those visits.", "synthetic"

    monkeypatch.setattr(p, "local_answer", model)
    final = p.execute("Who was here yesterday?", state_dir=tmp_path, on_short=seen.append)
    assert final["status"] == status


# Following up on a research report (the Inbox's "Ask a follow-up")


def finished_report(tmp_path, answer="Dana came by twice [T1]."):
    store = research.Store(tmp_path)
    job = store.submit("Who came by this week?")
    claimed = store.claim("worker-1")
    assert claimed is not None and claimed["id"] == job["id"]
    assert store.finish(job["id"], "worker-1", {"status": "ok", "answer": answer, "evidence": []})
    return job["id"]


def test_a_followup_reads_the_finished_report_by_id(tmp_path):
    ident = finished_report(tmp_path)
    assert p.followup_report({"report": ident}, tmp_path) == {
        "question": "Who came by this week?",
        "answer": "Dana came by twice [T1].",
    }


def test_a_followup_reads_at_most_the_first_part_of_a_long_answer(tmp_path):
    ident = finished_report(tmp_path, answer="x" * (p.MAX_EARLIER + 50))
    assert len(p.followup_report({"report": ident}, tmp_path)["answer"]) == p.MAX_EARLIER


@pytest.mark.parametrize(
    "followup",
    [
        pytest.param({"report": "0" * 32}, id="no-such-report"),
        pytest.param({"report": "../../etc"}, id="not-an-id"),
        pytest.param({"report": 7}, id="not-a-string"),
        pytest.param("0" * 32, id="not-an-object"),
    ],
)
def test_a_followup_on_anything_but_a_finished_report_is_refused(tmp_path, followup):
    with pytest.raises(ValueError, match="report"):
        p.followup_report(followup, tmp_path)


def test_a_followup_on_a_report_still_running_is_refused(tmp_path):
    store = research.Store(tmp_path)
    job = store.submit("Who came by this week?")
    with pytest.raises(ValueError, match="isn't finished"):
        p.followup_report({"report": job["id"]}, tmp_path)


def test_a_followup_job_hands_the_report_to_its_run_once():
    seen = []

    def run(question, mode, **options):
        seen.append(options.get("followup"))
        return {"status": "ok", "answer": "synthetic"}

    manager = p.JobManager(run)
    earlier = {"question": "Who came by this week?", "answer": "Dana, twice."}
    manager.submit("And when?", "local", followup=earlier)
    manager.submit("Explain gravity", "local")
    manager.queue.join()
    assert seen == [earlier, None]
    assert manager.followups == {}


@pytest.mark.parametrize(
    "mode, scope", [pytest.param("friends", "history", id="friends"), pytest.param("local", "general", id="general")]
)
def test_a_followup_asks_only_my_recordings(mode, scope):
    manager = p.JobManager(runner(return_value={"status": "ok", "answer": "synthetic"}))
    with pytest.raises(ValueError, match="My recordings"):
        manager.submit("And when?", mode, scope=scope, followup={"question": "q", "answer": "a"})
    assert manager.jobs == {}


def test_a_post_following_up_on_a_missing_report_is_refused_without_queueing(jobs):
    code, reply = call(body={**QUESTION, "followup": {"report": "0" * 32}})
    assert code == 400 and "report" in reply["error"]
    assert jobs.jobs == {}


def test_a_followup_answer_reads_the_report_before_the_evidence_and_skips_the_ready_answer(ollama):
    found = {
        "is_history": True,
        "answer": "Ready answer",
        "evidence": EVIDENCE,
        "coverage": {},
        "warnings": [],
        "query": {},
    }
    earlier = {"question": "Who came by this week?", "answer": "Dana, twice."}
    with cfg.override({"recall": {"enabled": False}}), patch.object(h, "collect", autospec=True, return_value=found):
        result = p.execute("And when did she leave?", followup=earlier)
    assert result["model"] != "local-history-evidence"
    content = prompt(ollama)
    assert list(content)[:2] == ["earlier_report", "evidence"]
    assert content["earlier_report"] == earlier


def test_the_worker_reads_a_followup_with_a_long_report_whole(monkeypatch):
    earlier = {"question": "q" * p.MAX_QUESTION, "answer": "a" * p.MAX_EARLIER}
    monkeypatch.setattr(p.sys, "argv", ["pack-ask", "--worker"])
    monkeypatch.setattr(
        p.sys, "stdin", io.StringIO(json.dumps({**QUESTION, "question": "q" * 1200, "followup": earlier}))
    )
    with patch.object(p, "execute", autospec=True, return_value={"answer": "ok"}) as execute:
        p.main()
    assert execute.call_args.kwargs["followup"] == earlier
