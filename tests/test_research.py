"""Research jobs for the Inbox: the durable queue, where research may run, yielding to instant Ask, and
the routes."""

import http.client
import json
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path
from unittest.mock import create_autospec

import pytest
import requests

from liberational import config as cfg
from liberational import ollama_health, voice_turn
from liberational.memory import ask, recall, research

T = 1_800_000_000.0
OWN = {"ask": {"ollama_url": "http://127.0.0.1:11435"}, "ollama": {"url": "http://127.0.0.1:11434"}}


@pytest.fixture
def store(tmp_path):
    return research.Store(tmp_path)


def finished(store, result=None, now=T):
    job = store.submit("what did Dana say about the garden", now=now)
    claimed = store.claim("w1", now=now + 1)
    assert claimed is not None and claimed["id"] == job["id"]
    store.finish(
        job["id"], "w1", result or {"status": "ok", "answer": "Dana wants tomatoes [T1].", "evidence": []}, now=now + 31
    )
    return job["id"]


def test_a_job_goes_from_queued_to_running_to_a_report(store):
    job = store.submit("  what did   Dana say about the garden ", now=T)
    assert (job["question"], job["status"], job["read"]) == ("what did Dana say about the garden", "queued", False)
    running = store.claim("w1", now=T + 1)
    assert (running["status"], running["attempts"]) == ("running", 1)
    result = {
        "status": "ok",
        "answer": "Dana wants tomatoes [T1].",
        "model": "ollama/qwen3:4b-instruct",
        "evidence": [{"id": "T1", "source": "transcript"}, {"id": "C2", "source": "camera"}],
        "rounds": [{"round": 1, "decision": "search", "tool": "transcript", "args": {"words": "garden"}, "found": 4}],
    }
    assert store.finish(job["id"], "w1", result, now=T + 31) is True
    done = store.get(job["id"])
    assert (done["status"], done["took_s"], done["read"]) == ("done", 30.0, False)
    assert done["result"]["evidence"] == [{"id": "T1", "source": "transcript"}]  # the sources the answer cites
    assert done["result"]["looked_at"]["sources"] == {"transcript": 1, "camera": 1}
    assert done["result"]["looked_at"]["searches"] == [{"tool": "transcript", "args": {"words": "garden"}, "found": 4}]


def test_reports_are_marked_read_and_unread_and_listed_without_their_report(store):
    ident = finished(store)
    assert store.mark(ident, True) is True
    assert [(j["id"], j["read"], "result" in j) for j in store.jobs()] == [(ident, True, False)]
    assert store.jobs()[0]["answer_status"] == "ok"  # how it came out, for the list
    store.mark(ident, False)
    assert store.get(ident)["read"] is False


def test_a_failed_answer_is_a_failed_job_with_its_message(store):
    ident = finished(store, {"status": "error", "answer": "The local model could not complete the answer."})
    job = store.get(ident)
    assert (job["status"], job["error"]) == ("failed", "The local model could not complete the answer.")


def test_a_job_a_stopped_worker_left_is_retried_once_then_fails(store):
    job = store.submit("why", now=T)
    store.claim("w1", now=T)
    assert store.claim("w2", now=T + 60) is None  # w1's heartbeat is still fresh
    again = store.claim("w2", now=T + research.STALE_S + 1)
    assert (again["worker"], again["attempts"]) == ("w2", 2)
    assert store.claim("w3", now=T + 2 * research.STALE_S + 2) is None
    assert store.get(job["id"])["status"] == "failed"


def test_a_heartbeat_keeps_a_long_job_its_own(store):
    store.submit("why", now=T)
    job = store.claim("w1", now=T)
    store.beat(job["id"], "w1", now=T + 100)
    assert store.claim("w2", now=T + 200) is None


def test_a_report_for_a_job_deleted_while_it_ran_is_dropped(store):
    job = store.submit("why", now=T)
    store.claim("w1", now=T)
    assert store.delete(job["id"]) is True
    assert store.finish(job["id"], "w1", {"status": "ok", "answer": "x"}) is False
    assert store.jobs() == []


def test_jobs_run_oldest_first(store):
    first = store.submit("first", now=T)
    store.submit("second", now=T + 1)
    assert store.claim("w1", now=T + 2)["id"] == first["id"]


def test_the_queue_holds_at_most_ten_waiting(store):
    for n in range(research.MAX_QUEUED):
        store.submit(f"question {n}", now=T + n)
    with pytest.raises(ValueError, match="already waiting"):
        store.submit("one more", now=T + 20)


@pytest.mark.parametrize(
    ("question", "message"),
    [("", "Ask a question"), ("  ", "Ask a question"), (None, "Ask a question"), ("x" * 1201, "under 1200")],
)
def test_submit_refuses(store, question, message):
    with pytest.raises(ValueError, match=message):
        store.submit(question)
    assert store.jobs() == []


@pytest.mark.parametrize(
    ("values", "available"),
    [
        pytest.param(OWN, True, id="asks-own-server"),
        pytest.param({"ask": {"ollama_url": ""}}, False, id="no-own-server"),
        pytest.param(
            {"ask": {"ollama_url": "http://127.0.0.1:11434/"}, "ollama": {"url": "http://127.0.0.1:11434"}},
            False,
            id="voice-loops-server",
        ),
        pytest.param({**OWN, "research": {"enabled": False}}, False, id="turned-off"),
        pytest.param(
            {"ask": {"ollama_url": ""}, "research": {"ollama_url": "http://gpu-box:11434"}},
            True,
            id="own-research-server",
        ),
    ],
)
def test_research_runs_only_away_from_the_voice_loops_ollama(values, available):
    with cfg.override(values):
        assert (research.unavailable() is None) is available


def test_research_uses_asks_model_unless_it_names_its_own():
    asks = {"ask": {"model": "qwen3:4b-instruct", "num_ctx": 65536, "ollama_url": "http://127.0.0.1:11435"}}
    with cfg.override(asks):
        assert research.settings()["ask"] == {
            "model": "qwen3:4b-instruct",
            "num_ctx": 65536,
            "ollama_url": "http://127.0.0.1:11435",
            "think": False,
        }
    with cfg.override({**asks, "research": {"model": "qwen3:14b", "num_ctx": 16384, "think": True, "max_rounds": 5}}):
        s = research.settings()
    assert (s["ask"]["model"], s["ask"]["num_ctx"], s["ask"]["think"], s["recall"]["max_rounds"]) == (
        "qwen3:14b",
        16384,
        True,
        5,
    )


def test_deep_raises_the_limits_and_sets_the_gate_only_inside(tmp_path):
    with cfg.override(OWN), research.deep():
        inside = (
            ask.ANSWER_TIMEOUT_S,
            recall.LOOP_BUDGET_S,
            cfg.get("recall.max_rounds"),
            ollama_health.gate is not None,
        )
    assert inside == (research.ANSWER_TIMEOUT_S, research.LOOP_BUDGET_S, 8, True)
    assert (ask.ANSWER_TIMEOUT_S, recall.LOOP_BUDGET_S, ollama_health.gate) == (90.0, 80, None)


def test_deep_restores_answer_sampling_after_the_job(monkeypatch):
    previous = {"temperature": 0.2}
    monkeypatch.setattr(ask, "ANSWER_SAMPLING", previous)
    with cfg.override({"research": {"think": True}}), research.deep():
        assert ask.ANSWER_SAMPLING == {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0}
    assert ask.ANSWER_SAMPLING is previous


def test_deep_restores_answer_sampling_when_the_job_raises(monkeypatch):
    previous = {"temperature": 0.2}
    monkeypatch.setattr(ask, "ANSWER_SAMPLING", previous)
    with (
        pytest.raises(RuntimeError, match="synthetic failed job"),
        cfg.override({"research": {"think": True}}),
        research.deep(),
    ):
        assert ask.ANSWER_SAMPLING == {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0}
        raise RuntimeError("synthetic failed job")
    assert ask.ANSWER_SAMPLING is previous


def test_research_shared_planner_waits_for_voice_and_restores_the_previous_gate(monkeypatch):
    events = []

    def original(model):
        events.append("original:" + model)

    saved = create_autospec(original, side_effect=original)
    monkeypatch.setattr(ollama_health, "gate", saved)
    monkeypatch.setattr(
        research,
        "yield_to_instant",
        create_autospec(research.yield_to_instant, side_effect=lambda: events.append("instant") or 0),
    )
    monkeypatch.setattr(
        voice_turn, "wait_quiet", create_autospec(voice_turn.wait_quiet, side_effect=lambda: events.append("quiet"))
    )
    with cfg.override({**OWN, "models": {"ollama": {"llm": "small-shared"}}}), research.deep():
        assert ollama_health.gate is not None
        ollama_health.gate("small-shared")
        ollama_health.gate("answer-thinking")
    assert ollama_health.gate is saved
    ollama_health.gate("after")
    assert events == ["instant", "quiet", "instant", "original:after"]


def test_research_caps_its_answers_tokens_and_instant_ask_does_not():
    # Instant Ask keeps its third-context thinking allowance. Research reserves half for reasoning and final JSON.
    assert (ask.reply_tokens(16384, True), ask.reply_tokens(4096, False)) == (5461, 1024)
    with cfg.override(OWN), research.deep():
        assert (ask.reply_tokens(16384, True), ask.reply_tokens(4096, False)) == (8192, 1024)
        assert ask.reply_tokens(16384, False) == 1400
        assert ask.reply_tokens(32768, True) == 8192
    assert ask.reply_tokens(16384, True) == 5461


def test_thinking_research_reserves_reply_space_before_budgeting_its_evidence():
    with (
        cfg.override({"ask": {"model": "thinker", "num_ctx": 16384}, "research": {"think": True}}),
        research.deep(),
    ):
        assert ask.evidence_budget() == 19776  # (16384 - 8192 reply - 1600 prompt) * 3
        assert ask.budgets(16384, cap=20000, think=True) == (19776, 8192)


def test_thinking_research_keeps_the_fixed_prompt_and_minimum_evidence_inside_a_small_context():
    with cfg.override({"research": {"think": True}}), research.deep():
        # 4096 total - 1600 fixed prompt - 800 minimum evidence tokens = 1696 reply tokens.
        assert ask.budgets(4096, think=True) == (2400, 1696)
        assert ask.reply_tokens(4096, True) == 1696


@pytest.mark.parametrize("planner", ["", "explicit-planner"])
def test_research_uses_a_non_thinking_planner_without_reloading_the_voice_context(planner):
    overrides = {
        **OWN,
        "ask": {**OWN["ask"], "model": "answer-instruct", "num_ctx": 32768},
        "research": {"model": "answer-thinking", "num_ctx": 16384, "think": True},
        "recall": {"model": planner},
        "models": {"ollama": {"llm": "small-shared"}},
    }
    with cfg.override(overrides), research.deep():
        assert ask.local_settings() == ("answer-thinking", 16384)
        assert recall.settings() == ((planner, 16384) if planner else ("small-shared", 4096))
        assert recall.url(recall.settings()[0]) == "http://127.0.0.1:11434"


def test_research_waits_while_an_instant_question_runs(tmp_path):
    sleeps = []
    with research.instant(tmp_path):
        markers = list(research.busy_dir(tmp_path).iterdir())

        def sleep(s):
            sleeps.append(s)
            if len(sleeps) == 3:
                markers[0].unlink()

        assert research.yield_to_instant(tmp_path, sleep=sleep) == 1.5
    assert research.yield_to_instant(tmp_path, sleep=sleeps.append) == 0.0


def test_a_marker_left_by_a_dead_process_is_ignored(tmp_path):
    import os

    folder = research.busy_dir(tmp_path)
    folder.mkdir(parents=True)
    old = folder / "1-left"
    old.write_text("x")
    an_hour_ago = time.time() - 3600
    os.utime(old, (an_hour_ago, an_hour_ago))
    assert research.instant_running(tmp_path) is False


def test_a_fresh_marker_left_by_a_process_that_exited_is_ignored(tmp_path):
    done = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True)
    folder = research.busy_dir(tmp_path)
    folder.mkdir(parents=True)
    (folder / f"{done.stdout.strip()}-left").write_text("x")
    assert research.instant_running(tmp_path) is False


def test_an_instant_question_holds_research_from_queued_until_answered():
    entered, release = threading.Event(), threading.Event()

    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(2), "test release missing"
        return {"status": "ok", "answer": "synthetic"}

    state = Path(ask.STATE)
    manager = ask.JobManager(slow)
    try:
        manager.submit("Explain gravity", "local")
        assert entered.wait(2)
        manager.submit("Explain tides", "local")  # waits behind the first
        held = len(list(research.busy_dir(state).iterdir()))
    finally:
        release.set()
    manager.queue.join()
    assert held == 2
    assert research.instant_running(state) is False


def test_the_gate_runs_before_every_wrapped_ollama_call(monkeypatch):
    seen = []
    monkeypatch.setattr(ollama_health, "gate", seen.append)
    with ollama_health.call("qwen3:4b-instruct", ollama_health.Health()):
        pass
    assert seen == ["qwen3:4b-instruct"]


def test_run_asks_with_researchs_settings(tmp_path, monkeypatch):
    seen = {}

    def execute(question, mode, state, day=None, scope="history"):
        seen.update(
            question=question, mode=mode, day=day, rounds=cfg.get("recall.max_rounds"), url=cfg.get("ask.ollama_url")
        )
        return {"status": "ok", "answer": "x", "evidence": []}

    monkeypatch.setattr(ask, "execute", execute)
    with cfg.override(OWN):
        research.run({"question": "why", "day": "2026-10-22"}, tmp_path)
    assert seen == {
        "question": "why",
        "mode": "local",
        "day": "2026-10-22",
        "rounds": 8,
        "url": "http://127.0.0.1:11435",
    }


@pytest.fixture
def warm_requests(model_server):
    seen = []

    def load(sent):
        seen.append((sent.url, sent.payload, sent.timeout, ask.local_settings(), ollama_health.gate))
        return {"done": True}

    model_server.reply = load
    return seen


@pytest.mark.parametrize("fails", [False, True])
@pytest.mark.parametrize("research_model", ["background-thinker", "interactive"])
def test_a_different_research_load_restores_normal_ask_after_its_overrides_exit(
    tmp_path, monkeypatch, warm_requests, fails, research_model
):
    result = {"status": "ok", "answer": "x", "evidence": []}

    def execute(*args, **kwargs):
        assert ask.local_settings() == (research_model, 16384)
        if fails:
            raise ValueError("answer failed")
        return result

    monkeypatch.setattr(ask, "execute", execute)
    with cfg.override(
        {
            "ask": {"model": "interactive", "num_ctx": 32768, "ollama_url": "http://127.0.0.1:11435"},
            "research": {"model": research_model, "num_ctx": 16384},
        }
    ):
        if fails:
            with pytest.raises(ValueError, match="answer failed"):
                research.run({"question": "why"}, tmp_path)
        else:
            assert research.run({"question": "why"}, tmp_path) is result
    assert warm_requests == [
        (
            "http://127.0.0.1:11435/api/generate",
            {"model": "interactive", "stream": False, "keep_alive": -1, "options": {"num_ctx": 32768}},
            120,
            ("interactive", 32768),
            None,
        )
    ]


@pytest.mark.parametrize(
    "values",
    [
        {"research": {"model": "interactive", "num_ctx": 32768}},
        {"research": {"model": "background-thinker", "ollama_url": "http://127.0.0.1:11436"}},
        {"ask": {"ollama_url": "http://127.0.0.1:11434"}, "research": {"model": "background-thinker"}},
        {"ask": {"ollama_url": "http://192.0.2.1:11435"}, "research": {"model": "background-thinker"}},
    ],
    ids=["same-load", "different-server", "voice-server", "remote-server"],
)
def test_research_does_not_warm_unchanged_separate_or_voice_loads(tmp_path, monkeypatch, warm_requests, values):
    monkeypatch.setattr(ask, "execute", lambda *args, **kwargs: {"status": "ok"})
    with (
        cfg.override(
            {
                "ask": {"model": "interactive", "num_ctx": 32768, "ollama_url": "http://127.0.0.1:11435"},
                "ollama": {"url": "http://127.0.0.1:11434"},
            }
        ),
        cfg.override(values),
    ):
        assert research.run({"question": "why"}, tmp_path) == {"status": "ok"}
    assert warm_requests == []


@pytest.mark.parametrize("fails", [False, True])
def test_failed_ask_restore_never_hides_the_research_result_or_exception(tmp_path, monkeypatch, caplog, fails):
    def execute(*args, **kwargs):
        if fails:
            raise ValueError("answer failed")
        return {"status": "ok", "answer": "finished"}

    def post(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(ask, "execute", execute)
    monkeypatch.setattr(requests, "post", post)
    with cfg.override(
        {
            "ask": {"model": "interactive", "num_ctx": 32768, "ollama_url": "http://127.0.0.1:11435"},
            "research": {"model": "background-thinker", "num_ctx": 16384},
        }
    ):
        if fails:
            with pytest.raises(ValueError, match="answer failed"):
                research.run({"question": "why"}, tmp_path)
        else:
            assert research.run({"question": "why"}, tmp_path) == {"status": "ok", "answer": "finished"}
    assert "restore" in caplog.text.lower()


def test_the_worker_runs_a_queued_job_to_its_report(tmp_path, monkeypatch):
    store = research.Store(tmp_path)
    job = store.submit("why", now=T)
    stop = threading.Event()

    def run(j, state=None):
        stop.set()
        return {"status": "ok", "answer": f"Because, for {j['question']}.", "evidence": []}

    monkeypatch.setattr(research, "run", run)
    with cfg.override(OWN):
        research.work(stop, tmp_path)
    done = store.get(job["id"])
    assert done is not None
    assert (done["status"], done["result"]["answer"]) == ("done", "Because, for why.")


def test_a_job_that_raises_is_a_failed_report_and_the_worker_carries_on(tmp_path, monkeypatch):
    store = research.Store(tmp_path)
    job = store.submit("why", now=T)
    stop = threading.Event()

    def run(j, state=None):
        stop.set()
        raise RuntimeError("model gone")

    monkeypatch.setattr(research, "run", run)
    with cfg.override(OWN):
        research.work(stop, tmp_path)
    failed = store.get(job["id"])
    assert failed is not None
    assert failed["error"] == "Research failed (RuntimeError)."


@pytest.fixture
def http_(serve, monkeypatch, tmp_path):
    from liberational.dashboard import server

    monkeypatch.setattr(server, "DASH_PASSWORD", "")
    monkeypatch.setattr(research, "root", lambda state=None: tmp_path / "research")
    netloc = urllib.parse.urlsplit(serve(server.H)).netloc

    def send(method, path, body=None, origin=None):
        conn = http.client.HTTPConnection(netloc, timeout=5)
        headers = {"Content-Type": "application/json", **({"Origin": origin} if origin else {})}
        conn.request(method, path, body=json.dumps(body) if body is not None else None, headers=headers)
        response = conn.getresponse()
        value = response.status, json.loads(response.read())
        conn.close()
        return value

    return send


def test_the_inbox_routes_queue_list_read_and_delete(http_):
    with cfg.override(OWN):
        code, job = http_("POST", "/research/jobs", {"question": "what did Dana say about the garden"})
        assert (code, job["status"]) == (202, "queued")
        code, listing = http_("GET", "/research/jobs")
        assert (code, listing["available"], [j["id"] for j in listing["jobs"]]) == (200, True, [job["id"]])
        assert http_("POST", f"/research/jobs/{job['id']}/read", {"read": True}) == (200, {"ok": True, "read": True})
        assert http_("GET", f"/research/jobs/{job['id']}")[1]["read"] is True
        assert http_("POST", f"/research/jobs/{job['id']}/delete", {}) == (200, {"ok": True})
        assert http_("GET", f"/research/jobs/{job['id']}")[0] == 404


def test_the_inbox_refuses_research_where_it_cant_run_and_other_sites(http_):
    with cfg.override({"ask": {"ollama_url": ""}}):
        code, reply = http_("POST", "/research/jobs", {"question": "why"})
        assert (code, "ask-ollama" in reply["error"]) == (409, True)
        assert http_("GET", "/research/jobs")[1]["available"] is False
    with cfg.override(OWN):
        assert http_("POST", "/research/jobs", {"question": "why"}, origin="http://elsewhere.example")[0] == 403


def test_the_list_says_a_partial_answer_and_none_before_one(store):
    partial = finished(store, {"status": "partial", "answer": "Some of it.", "evidence": []})
    waiting = store.submit("who came by", now=T + 100)["id"]
    assert {j["id"]: j["answer_status"] for j in store.jobs()} == {partial: "partial", waiting: None}
