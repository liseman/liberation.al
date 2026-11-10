"""With no agent configured (the default), features that would use it say so and never make a request."""

import urllib.request
from unittest.mock import create_autospec, patch

import pytest
import requests

from liberational import config as cfg
from liberational.assistant import ask_friends, voice
from liberational.assistant.voice_auth import Grant
from liberational.dashboard import server
from liberational.memory import ask, history_qa
from liberational.people import identity
from liberational.phone import bridge


@pytest.fixture(autouse=True)
def requests_made(monkeypatch):
    """Every network request something tried. Refused, and recorded even when the caller swallows the error."""
    assert not cfg.agent_configured()
    calls = []

    def refuse(*a, **k):
        calls.append(a)
        raise AssertionError("made a network request with no agent configured")

    monkeypatch.setattr(requests, "post", refuse)
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", refuse)
    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    return calls


def test_dashboard_lookup_reports_missing_agent():
    assert server.public_lookup("Dana") == {"error": "No agent is configured (see [agent] in settings)."}


def test_ask_friends_says_it_needs_an_agent(monkeypatch):
    monkeypatch.setattr(ask_friends, "configured_inventory", lambda: pytest.fail("looked for models"))
    out = ask_friends.ask("What should I cook tonight?")
    assert out["status"] == "no_agent"
    assert "needs an agent" in out["answer"]
    with pytest.raises(ValueError, match="no agent configured"):
        ask_friends.cloud_request("some/model", [])


def test_phone_bridge_skips_the_agent(requests_made):
    with pytest.raises(RuntimeError, match="no agent configured"):
        bridge.phone_agent_reply("CA1", "What's the weather?")
    bridge.warm_phone_session("CA1")  # swallows errors, so the record below is what shows it made no request
    assert requests_made == []


def owner_grant(request):
    return Grant(request, "gateway", False, False, needs_password=False, needs_confirmation=False, owner=True)


def test_voice_owner_request_answers_without_calling_out():
    with pytest.raises(voice.NoAgent):
        voice.owner_answer("book me a table", "s1", grant=owner_grant("book me a table"))


def test_introducing_someone_needs_no_agent():
    intro = {"reply": "Got it, Katie.", "record": None}
    with (
        patch.object(voice.CONVERSATION, "accept", autospec=True, return_value=owner_grant("that's Katie.")),
        patch.object(identity, "collar_intro", autospec=True, return_value=intro) as collar_intro,
    ):
        assert voice.handle("Hey Bob, that's Katie.", None, {"speaker": "sam"}) == "Got it, Katie."
    collar_intro.assert_called_once_with("Katie")


def test_friends_mode_falls_back_to_local():
    with (
        patch.object(history_qa, "collect", autospec=True, return_value={"warnings": [], "evidence": []}),
        patch.object(ask, "local_answer", autospec=True, return_value=("Pasta.", "ollama/x")) as local,
    ):
        out = ask.execute("What should I cook tonight?", mode="friends")
    assert local.call_args.args[0] == "What should I cook tonight?"
    assert (out["mode"], out["answer"]) == ("local", "Pasta.")
    assert any("No agent is set up" in w for w in out["warnings"])


@pytest.mark.parametrize(
    ("addressed", "spoken"),
    [
        pytest.param(False, None, id="a-question-to-someone-else-stays-quiet"),
        pytest.param(True, voice.NO_AGENT_REPLY, id="hey-bob-hears-why"),
    ],
)
def test_a_failed_answer_is_spoken_only_to_a_line_that_called_the_assistant(monkeypatch, addressed, spoken):
    """With no agent, every question the owner asked someone else in the room got "I can't do that without an
    agent" spoken over them."""
    request = "How many times has that happened?"
    grant = Grant(request, "gateway", False, False, False, False, owner=True, addressed=addressed)
    speaker = create_autospec(voice.Speaker, instance=True)
    monkeypatch.setattr(voice, "log_reply", lambda *a, **k: None)
    with (
        patch.object(voice.CONVERSATION, "accept", autospec=True, return_value=grant),
        patch.object(voice.CONVERSATION, "spoken_reply", autospec=True),
    ):
        assert voice.handle(request, speaker, {"speaker": "sam"}) == spoken
    assert speaker.speak.call_count == (1 if spoken else 0)
