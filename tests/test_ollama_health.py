import socket
import sys
import time
import urllib.error
import urllib.request
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import requests

from liberational import config as cfg
from liberational import ollama_health as oh
from liberational.see import vlm
from liberational.vision import objects

URL = "http://127.0.0.1:11434"
REFUSED = requests.ConnectionError("Connection refused")
MODELS = {
    "models": {
        "ollama": {
            "llm": "qwen3:4b-instruct",
            "big_llm": "qwen3:4b-instruct",
            "vlm": "qwen3-vl:4b-instruct",
            "embedding": "emb:4b",
        }
    }
}


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def lines():
    return []


@pytest.fixture
def health(clock, lines):
    with cfg.override({"ollama": {"url": URL, "retry_s": 30, "log_every_s": 60}}):
        yield oh.Health(clock, lines.append)


def attempt(health, clock, at, error=None, model="m"):
    """One call at time `at`; whether it reached Ollama (the body ran)."""
    clock.t = 1000.0 + at
    ran = []
    try:
        with oh.call(model, health):
            ran.append(True)
            if error is not None:
                raise error
    except oh.Unavailable:
        pass
    return bool(ran)


def not_found(model):
    r = requests.Response()
    r.status_code = 404
    r._content = f'{{"error":"model \\"{model}\\" not found, try pulling it first"}}'.encode()
    return requests.HTTPError(response=r)


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(REFUSED, id="requests"),
        pytest.param(urllib.error.URLError(ConnectionRefusedError(111, "refused")), id="urllib"),
        pytest.param(ConnectionRefusedError(111, "refused"), id="socket"),
    ],
)
def test_a_refused_connection_logs_the_fix_once_and_raises_down(health, error, lines):
    with pytest.raises(oh.Down), oh.call("m", health):
        raise error
    assert lines == [f"Ollama isn't running at {URL}: {oh.START}. Trying again every 30 s."]


def test_while_down_calls_fail_fast_and_ollama_is_tried_every_30_s(health, clock):
    assert attempt(health, clock, 0, REFUSED)
    assert not attempt(health, clock, 1)
    assert not attempt(health, clock, 29)
    assert attempt(health, clock, 30, REFUSED)  # the retry
    assert not attempt(health, clock, 31)
    assert attempt(health, clock, 61, REFUSED)


def test_a_reminder_is_logged_at_most_once_a_minute(health, clock, lines):
    attempt(health, clock, 0, REFUSED)
    attempt(health, clock, 30, REFUSED)
    assert len(lines) == 1
    attempt(health, clock, 60, REFUSED)
    attempt(health, clock, 90, REFUSED)
    assert lines[1:] == [f"Ollama isn't running at {URL} (down 60 s): {oh.START}"]
    attempt(health, clock, 120, REFUSED)
    assert lines[2] == f"Ollama isn't running at {URL} (down 2 min): {oh.START}"


def test_recovery_logs_one_back_line_and_calls_go_through_again(health, clock, lines):
    attempt(health, clock, 0, REFUSED)
    assert attempt(health, clock, 120)
    assert lines[-1] == f"Ollama is back at {URL} (down 2 min)."
    assert attempt(health, clock, 121)
    assert len(lines) == 2


def test_a_missing_model_names_it_and_the_pull_command_and_backs_off_only_that_model(health, clock, lines):
    assert attempt(health, clock, 0, not_found("emb:4b"), model="emb:4b")
    assert lines == ["Ollama has no model emb:4b: pull it with `ollama pull emb:4b`."]
    assert not attempt(health, clock, 5, model="emb:4b")
    assert attempt(health, clock, 5, model="qwen3:4b-instruct")  # other models still work
    assert attempt(health, clock, 30, model="emb:4b")
    assert lines[-1] == "Ollama has emb:4b now."


def test_a_404_while_down_means_ollama_is_back_for_the_other_models(health, clock, lines):
    """A restarted Ollama without one model must not keep the installed models blocked."""
    attempt(health, clock, 0, REFUSED)
    assert attempt(health, clock, 30, not_found("emb:4b"), model="emb:4b")  # the retry: Ollama answers, 404
    assert attempt(health, clock, 31, model="qwen3:4b-instruct")
    assert lines[1:3] == [
        f"Ollama is back at {URL} (down 30 s).",
        "Ollama has no model emb:4b: pull it with `ollama pull emb:4b`.",
    ]


def test_a_404_for_a_call_with_no_model_is_not_a_missing_model(health, lines):
    """A call that names no model (like /api/ps) gets a 404 that must not log "pull it with `ollama pull `"."""
    with pytest.raises(requests.HTTPError), oh.call("", health):
        raise not_found("")
    assert lines == []


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(requests.Timeout("read timed out"), id="timeout"),
        pytest.param(urllib.error.URLError(TimeoutError("timed out")), id="urllib-timeout"),
        pytest.param(requests.HTTPError(response=requests.Response()), id="http-500"),
        pytest.param(ValueError("bad json"), id="other"),
    ],
)
def test_other_errors_pass_through_without_a_line_or_a_backoff(health, clock, lines, error):
    with pytest.raises(type(error)), oh.call("m", health):
        raise error
    assert lines == []
    assert attempt(health, clock, 1)


def test_the_caption_vlm_logs_one_line_and_stops_posting_every_frame(capsys, model_server):
    """Avoid logging one `vlm error` per second while Ollama is down."""
    frame = np.zeros((32, 32, 3), np.uint8)
    backend = vlm.OllamaVLM("qwen3-vl:4b-instruct")
    model_server.reply = REFUSED
    with cfg.override({"ollama": {"url": URL}}):
        assert backend.caption(frame) == ""
        assert backend.caption(frame) == ""
        assert backend.caption(frame) == ""
    assert len(model_server.sent) == 1
    err = capsys.readouterr().err
    assert err.count("Ollama isn't running") == 1
    assert "vlm error" not in err


class Tags:
    def __init__(self, names: list[str]) -> None:
        self.names = names

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return {"models": [{"name": n, "model": n, "digest": "d"} for n in self.names]}


def test_status_names_the_configured_models_ollama_lacks():
    with cfg.override({"ollama": {"url": URL}, **MODELS}):
        s = oh.status(lambda url, timeout: Tags(["qwen3:4b-instruct", "qwen3-vl:4b-instruct"]))
    assert s == {
        "up": True,
        "missing": ["emb:4b"],
        "message": "Model missing: emb:4b (pull it with `ollama pull emb:4b`)",
        "error": None,
    }


def test_status_checks_the_model_an_environment_override_names(monkeypatch):
    """Callers read models.ollama.<role> with its PACK_ override, so status must too."""
    monkeypatch.setenv("PACK_MODELS_OLLAMA_EMBEDDING", "other-emb:1b")
    with cfg.override({"ollama": {"url": URL}, **MODELS}):
        s = oh.status(lambda url, timeout: Tags(["qwen3:4b-instruct", "qwen3-vl:4b-instruct", "emb:4b"]))
    assert s["missing"] == ["other-emb:1b"]


def test_status_with_every_model_has_no_message():
    with cfg.override({"ollama": {"url": URL}, **MODELS}):
        s = oh.status(lambda url, timeout: Tags(["qwen3:4b-instruct", "qwen3-vl:4b-instruct", "emb:4b"]))
    assert (s["up"], s["missing"], s["message"]) == (True, [], None)


def test_status_says_ollama_isnt_running():
    def refused(url, timeout):
        raise REFUSED

    with cfg.override({"ollama": {"url": URL}, **MODELS}):
        s = oh.status(refused)
    assert (s["up"], s["message"]) == (
        False,
        f"Ollama isn't running at {URL}: {oh.START}",
    )
    # The pack runs Ollama as one of its own services; a system unit it never installs is not the fix.
    assert "pack-services start ollama" in oh.START


def test_status_counts_a_slow_answer_as_running():
    # A busy Ollama took over 1.5 s to list its models 13% of the time, and that read as "isn't running".
    def slow(url, timeout):
        raise requests.exceptions.ReadTimeout("read timed out")

    with cfg.override({"ollama": {"url": URL}, **MODELS}):
        s = oh.status(slow)
    assert (s["up"], s["missing"], s["message"]) == (True, [], None)


class Ps:
    """Ollama's /api/ps answer, as urlopen gives it."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return b'{"models": []}'


@pytest.fixture
def polled(monkeypatch):
    """The dashboard's Stats on a clock of its own, with Ollama answering (until `up` is cleared) and every /api/tags
    read counted."""
    from liberational.dashboard import machine_stats

    clock = {"now": 1_800_000_000.0, "up": True, "tags": 0}

    def urlopen(url, timeout):
        if not clock["up"]:
            raise urllib.error.URLError(ConnectionRefusedError(111, "refused"))
        return Ps()

    def tags(url, timeout):
        clock["tags"] += 1
        if not clock["up"]:
            raise REFUSED
        return Tags(["qwen3:4b-instruct", "qwen3-vl:4b-instruct", "emb:4b"])

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(oh, "_get", tags)
    monkeypatch.setattr(time, "time", lambda: clock["now"])
    with cfg.override({"ollama": {"url": URL}, **MODELS}):
        yield machine_stats.Stats(), clock


def test_stats_polled_every_few_seconds_ask_ollama_for_its_models_once_in_five_minutes(polled):
    # /api/tags took about a second on the Linux machine when asked on every /stats poll (12 a minute).
    stats, clock = polled
    for _ in range(140):  # 140 polls 2.1 s apart: 294 s, under the 300 s the answer stands
        assert stats.ollama_ps()["up"] is True
        clock["now"] += 2.1
    assert clock["tags"] == 1
    clock["now"] = 1_800_000_300.0
    stats.ollama_ps()
    assert clock["tags"] == 2


def test_stats_say_ollama_is_down_at_once_and_back_at_once(polled):
    stats, clock = polled
    stats.ollama_ps()
    clock["up"], clock["now"] = False, clock["now"] + 3
    assert stats.ollama_ps()["up"] is False  # not the cached answer from 3 s before
    clock["up"], clock["now"] = True, clock["now"] + 3
    assert stats.ollama_ps()["up"] is True
    assert clock["tags"] == 2


def test_the_dashboard_stats_carry_what_to_tell_the_owner():
    from liberational.dashboard import machine_stats

    def refused(*a, **k):
        raise urllib.error.URLError(ConnectionRefusedError(111, "refused"))

    with (
        cfg.override({"ollama": {"url": URL}, **MODELS}),
        patch.object(urllib.request, "urlopen", side_effect=refused),
        patch.object(oh, "_get", side_effect=refused),
    ):
        o = machine_stats.Stats().ollama_ps()
    assert (o["up"], o["missing"], o["message"]) == (
        False,
        [],
        f"Ollama isn't running at {URL}: {oh.START}",
    )


# ---- every Ollama caller goes through the helper: against a closed port, two calls make one connection ----

CONNECTS: list[object] = []  # [the closed port's address, then one entry per connection attempt to it]


def _count_connects(event: str, args: tuple[object, ...]) -> None:
    if event == "socket.connect" and CONNECTS and args[1] == CONNECTS[0]:
        CONNECTS.append(args[1])


sys.addaudithook(_count_connects)


@pytest.fixture
def closed_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        address = s.getsockname()
    CONNECTS[:] = [address]
    with cfg.override({"ollama": {"url": f"http://127.0.0.1:{address[1]}"}}):
        yield
    CONNECTS.clear()


def _voice():
    from liberational.assistant import voice

    voice.answer("what time is it")


def _whisper_names():
    from liberational.people import whisper_names

    assert whisper_names.polish("Dana is here.") == "Dana is here."  # the line unpolished, as before


def _profiles():
    from liberational.people import profiles

    profiles.ask_llm("Dana", ["I like tea."])


def _facts():
    from liberational.memory import facts

    facts._ask("system", "user", {"type": "object"})


def _phone_bridge():
    from liberational.phone import bridge

    bridge.ask_llm([{"role": "user", "content": "hi"}], system="Be brief.")


def tried(call):
    try:
        call()
    except oh.Unavailable:
        pass


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(_voice, id="voice"),
        pytest.param(_whisper_names, id="whisper-names"),
        pytest.param(_profiles, id="profiles"),
        pytest.param(_facts, id="facts"),
        pytest.param(_phone_bridge, id="phone-bridge"),
    ],
)
def test_a_down_ollama_is_logged_once_and_not_tried_again_at_once(closed_port, capsys, call):
    tried(call)
    tried(call)
    assert len(CONNECTS) - 1 == 1
    assert capsys.readouterr().err.count("Ollama isn't running") == 1


class Reply:
    """A fake requests.Response for one Ollama /api/chat or /api/generate call."""

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"message": {"content": "A desk."}, "response": ""}


def test_every_vision_model_call_asks_for_the_same_context_so_ollama_never_reloads_it(model_server):
    """A warm-up with no num_ctx loaded the VLM at its 262k context (25.6 GB), then the first caption
    reloaded it at 4096."""
    frame = np.zeros((32, 32, 3), np.uint8)
    backend = vlm.OllamaVLM("qwen3-vl:4b-instruct")
    model_server.says("A desk.")
    with cfg.override({"ollama": {"url": URL}}):
        backend.warm()
        backend.caption(frame)
        backend.ground(frame)
    sent = [(s.path, s.payload.get("options", {}).get("num_ctx")) for s in model_server.sent]
    assert sent == [("/api/generate", 4096), ("/api/chat", 4096), ("/api/chat", 4096)]


@pytest.mark.parametrize(
    ("method", "args"),
    [("warm", ()), ("caption", (np.zeros((32, 32, 3), np.uint8),)), ("ground", (np.zeros((32, 32, 3), np.uint8),))],
)
def test_background_vision_waits_before_sending_inference(monkeypatch, model_server, method, args):
    from liberational import voice_turn

    events = []
    monkeypatch.setattr(voice_turn, "wait_quiet", lambda: events.append("quiet"))

    def infer(sent):
        events.append("inference")
        return {"message": {"content": "A desk."}, "response": ""}

    model_server.reply = infer
    getattr(vlm.OllamaVLM("test-vlm"), method)(*args)
    assert events == ["quiet", "inference"]


def _background_whisper():
    from liberational.people import whisper_names

    whisper_names.polish("Dana is here.")


@pytest.mark.parametrize("call", [_facts, _profiles, _background_whisper])
def test_background_text_callers_wait_before_sending_inference(monkeypatch, model_server, call):
    from liberational import voice_turn

    events = []
    monkeypatch.setattr(voice_turn, "wait_quiet", lambda: events.append("quiet"))

    def infer(sent):
        events.append("inference")
        return {"message": {"content": "[]"}}

    model_server.reply = infer
    call()
    assert events == ["quiet", "inference"]


@pytest.mark.parametrize("background", [True, False])
def test_inprocess_caption_gate_can_be_bypassed_by_already_gated_server(monkeypatch, background):
    import threading
    from types import SimpleNamespace

    from liberational import voice_turn

    events = []
    monkeypatch.setattr(voice_turn, "wait_quiet", lambda: events.append("quiet"))

    def generate(*args, **kwargs):
        events.append("inference")
        return SimpleNamespace(text="A desk.")

    backend: Any = vlm.MlxVLM.__new__(vlm.MlxVLM)
    backend.background = background
    backend.thread = threading.get_ident()
    backend.px = 448
    backend.model = backend.processor = None
    backend._prompt = lambda: "Describe this image."
    backend._generate = generate
    assert backend.caption(np.zeros((32, 32, 3), np.uint8)) == "A desk."
    assert events == (["quiet", "inference"] if background else ["inference"])


def test_server_caption_waits_before_queueing_http_request(monkeypatch):
    from liberational import voice_turn

    events = []
    monkeypatch.setattr(voice_turn, "wait_quiet", lambda: events.append("quiet"))

    class ServerReply(Reply):
        def json(self):
            return {"caption": "A desk."}

    def post(*args, **kwargs):
        events.append("inference")
        return ServerReply()

    backend = vlm.MlxServerVLM.__new__(vlm.MlxServerVLM)
    backend.url, backend.timeout = "http://127.0.0.1:9999", 1
    with patch.object(vlm.requests, "post", autospec=True, side_effect=post):
        assert backend.caption(np.zeros((32, 32, 3), np.uint8)) == "A desk."
    assert events == ["quiet", "inference"]


def test_a_grounding_reply_that_ran_out_of_tokens_is_truncated_not_an_answer(model_server):
    backend = vlm.OllamaVLM("test-vlm")
    model_server.says('[{"label": "tv", "bbox_2d": [5', done_reason="length")
    with cfg.override({"ollama": {"url": URL}}), pytest.raises(objects.GroundingTruncated):
        backend.ground(np.zeros((32, 32, 3), np.uint8))


def test_grounding_asks_ollama_for_at_most_the_objects_the_prompt_asks_for(model_server):
    backend = vlm.OllamaVLM("test-vlm")
    model_server.says("A desk.")
    with cfg.override({"ollama": {"url": URL}}):
        backend.ground(np.zeros((32, 32, 3), np.uint8))
    payload = model_server.sent[0].payload
    assert payload["format"]["type"] == "array" and payload["format"]["maxItems"] == 8
    assert "up to 8" in payload["messages"][0]["content"]


def test_a_missing_threads_tag_says_to_make_it_not_pull_it():
    assert oh.pull("qwen3:4b-instruct-t2") == (
        "make it with `mise run models` (it's qwen3:4b-instruct with fewer CPU threads)"
    )
    assert oh.pull("qwen3:4b-instruct") == "pull it with `ollama pull qwen3:4b-instruct`"
