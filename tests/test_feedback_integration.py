"""No live model or secrets: the friends follow-up, and what never reaches the friends panel."""

import time
from unittest.mock import create_autospec

import pytest

from liberational.assistant import voice
from liberational.assistant.voice_auth import Conversation, Grant
from liberational.memory import ask as memory_ask


@pytest.fixture(autouse=True)
def panel(tmp_path, monkeypatch):
    """The friends panel (memory.ask.ask) as an autospecced stand-in that returns an "ok" answer. Any other optional
    module is missing."""
    ask = create_autospec(memory_ask.ask, return_value={"answer": "Opinion", "status": "ok"})
    monkeypatch.setattr(memory_ask, "ask", ask)
    modules = {"liberational.memory.ask": memory_ask}
    monkeypatch.setattr(voice, "optional_module", modules.get)
    return ask


def say(monkeypatch, text, accepted=None):
    """voice.handle(text), with voice_auth answering `accepted`: a Grant, a str (a refusal), or by default an owner's
    grant for exactly this request."""
    owner = Grant(text, "gateway", False, False, needs_password=False, needs_confirmation=False, owner=True)
    conversation = create_autospec(Conversation, instance=True)
    conversation.pending = None
    conversation.accept.return_value = accepted or owner
    monkeypatch.setattr(voice, "CONVERSATION", conversation)
    return voice.handle(text)


def test_ask_your_friends_alone_follows_up_the_last_question(panel, monkeypatch):
    monkeypatch.setattr(voice, "LAST_QUESTION", (time.time(), "Why does ice float?"))
    say(monkeypatch, "Ask your friends")
    panel.assert_called_once_with("Why does ice float?", mode="friends")


def test_a_refusal_clears_the_last_question(monkeypatch):
    monkeypatch.setattr(voice, "LAST_QUESTION", (time.time(), "Why does ice float?"))
    assert say(monkeypatch, "Read private data", "Please say your password") == "Please say your password"
    assert voice.LAST_QUESTION is None


def test_a_sensitive_request_never_reaches_the_panel(panel, monkeypatch):
    text = "Ask your friends about private data"
    say(monkeypatch, text, Grant(text, "gateway", True, False, needs_password=True, needs_confirmation=False))
    panel.assert_not_called()
