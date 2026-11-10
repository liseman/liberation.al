"""The shared model-server client (liberational/ollama.py): what it sends, and what each failure becomes."""

import io
import json
import urllib.error
import urllib.request
from unittest.mock import create_autospec

import pytest

from liberational import config as cfg
from liberational import ollama, ollama_health, voice_turn


@pytest.fixture
def server(model_server):
    model_server.says("hello")
    return model_server


def http_error(code, body=b""):
    return urllib.error.HTTPError("http://localhost/api/chat", code, "status", {}, io.BytesIO(body))  # ty: ignore[invalid-argument-type]  # a dict stands in for the headers


def test_chat_sends_the_callers_settings_and_only_the_optional_ones_it_gave(server):
    reply = ollama.chat("small", [{"role": "user", "content": "hi"}], num_ctx=4096, num_predict=40, timeout=8)

    assert reply["message"]["content"] == "hello"
    sent = server.sent[0]
    assert (sent.url, sent.timeout) == (cfg.ollama_url() + "/api/chat", 8)
    assert sent.payload == {
        "model": "small",
        "stream": False,
        "keep_alive": -1,
        "options": {"num_predict": 40, "num_ctx": 4096},
        "messages": [{"role": "user", "content": "hi"}],
    }


def test_chat_sends_schema_think_truncate_and_extra_options_when_given(server):
    ollama.chat(
        "small",
        [],
        num_ctx=2048,
        temperature=0,
        options={"top_p": 0.9},
        keep_alive="5m",
        schema={"type": "object"},
        think=False,
        truncate=False,
        timeout=1,
        url="http://127.0.0.1:1",
    )

    url, payload = server.sent[0].url, server.sent[0].payload
    assert url == "http://127.0.0.1:1/api/chat"
    assert payload["options"] == {"temperature": 0, "top_p": 0.9, "num_ctx": 2048}
    assert (payload["format"], payload["think"], payload["truncate"]) == ({"type": "object"}, False, False)
    assert payload["keep_alive"] == "5m"


def test_the_connection_uses_no_proxy_and_refuses_a_redirect(monkeypatch):
    handlers = []

    class Opener:
        def open(self, request, timeout):
            raise urllib.error.URLError(ConnectionRefusedError())

    monkeypatch.setattr(urllib.request, "build_opener", lambda *h: handlers.extend(h) or Opener())
    with pytest.raises(ollama_health.Down):
        ollama.chat("small", [], num_ctx=4096, timeout=1)

    proxy, redirect = handlers
    assert isinstance(proxy, urllib.request.ProxyHandler) and proxy.proxies == {}
    with pytest.raises(ValueError, match="Redirect refused"):
        redirect.redirect_request(None, None, 302, "Found", {}, "http://example.com/")


def test_only_a_background_call_waits_for_the_voice_turn(server, monkeypatch):
    wait = create_autospec(voice_turn.wait_quiet)
    monkeypatch.setattr(voice_turn, "wait_quiet", wait)

    ollama.chat("small", [], num_ctx=4096, timeout=1)
    wait.assert_not_called()

    ollama.chat("small", [], num_ctx=4096, timeout=1, background=True)
    wait.assert_called_once_with()


@pytest.mark.parametrize("url", ["http://192.0.2.7:11434", "https://localhost:11434", "http://example.com"])
def test_a_local_call_refuses_an_endpoint_on_another_machine_before_sending(server, url):
    with pytest.raises(ValueError, match="isn't on this machine"):
        ollama.chat("small", [], num_ctx=4096, timeout=1, url=url, local=True)
    assert server.sent == []


@pytest.mark.parametrize("url", ["http://localhost:11434", "http://127.0.0.1:11435", "http://[::1]:11434"])
def test_a_local_call_goes_to_this_machine(server, url):
    ollama.chat("small", [], num_ctx=4096, timeout=1, url=url, local=True)
    assert server.sent[0].url == url + "/api/chat"


def test_a_busy_server_is_unavailable_and_logged_once(server, caplog):
    server.reply = http_error(503)
    with pytest.raises(ollama_health.Busy):
        ollama.chat("small", [], num_ctx=4096, timeout=1)
    assert issubclass(ollama_health.Busy, ollama_health.Unavailable)
    assert [r.levelname for r in caplog.records if "busy" in r.getMessage()] == ["WARNING"]


def test_a_missing_model_and_a_stopped_server_come_back_as_health_errors(server):
    server.reply = http_error(404)
    with pytest.raises(ollama_health.ModelMissing):
        ollama.chat("absent", [], num_ctx=4096, timeout=1)

    server.reply = urllib.error.URLError(ConnectionRefusedError())
    with pytest.raises(ollama_health.Down):
        ollama.chat("small", [], num_ctx=4096, timeout=1)


def test_another_http_error_passes_through_with_its_body_readable(server):
    server.reply = http_error(400, b'{"error": "bad request"}')
    with pytest.raises(urllib.error.HTTPError) as raised:
        ollama.chat("small", [], num_ctx=4096, timeout=1)
    assert raised.value.read() == b'{"error": "bad request"}'


def test_an_oversized_reply_is_refused(server):
    server.reply = b" " * (ollama.MAX_REPLY_BYTES + 1)
    with pytest.raises(ValueError, match="too large"):
        ollama.chat("small", [], num_ctx=4096, timeout=1)


def test_a_reader_gets_the_open_response_of_a_streamed_call(server):
    server.reply = b'{"n": 1}\n{"n": 2}\n'

    lines = ollama.chat("small", [], num_ctx=4096, timeout=1, read=lambda r: [json.loads(x) for x in r])

    assert lines == [{"n": 1}, {"n": 2}]
    assert server.sent[0].payload["stream"] is True


@pytest.mark.parametrize(
    ("kwargs", "options", "keep_alive"),
    [
        ({"num_ctx": 4096}, {"num_ctx": 4096}, -1),
        ({"keep_alive": 0}, {}, 0),
    ],
    ids=["load-at-a-context", "unload"],
)
def test_load_asks_generate_with_no_prompt(server, kwargs, options, keep_alive):
    server.reply = b'{"done": true}'

    ollama.load("vlm", **kwargs)

    sent = server.sent[0]
    assert (sent.url, sent.timeout) == (cfg.ollama_url() + "/api/generate", 120)
    assert sent.payload == {"model": "vlm", "stream": False, "keep_alive": keep_alive, "options": options}


@pytest.mark.parametrize(
    ("content", "said"),
    [("  Yes.\n", "Yes."), ("<think>hmm\nwell</think>\n\nYes.", "Yes."), ("", "")],
)
def test_text_is_the_reply_without_its_thinking(content, said):
    assert ollama.text({"message": {"content": content}}) == said


def test_text_raises_for_a_reply_with_no_message():
    with pytest.raises(KeyError):
        ollama.text({"error": "boom"})
