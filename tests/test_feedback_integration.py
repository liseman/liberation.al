"""No live model or secrets: the old words for asking outside models get a plain spoken line, and never reach Ask."""

from unittest.mock import create_autospec

import pytest

from liberational.assistant import voice
from liberational.assistant.voice_auth import Conversation, Grant
from liberational.memory import ask as memory_ask


@pytest.fixture(autouse=True)
def panel(tmp_path, monkeypatch):
    """Ask (memory.ask.ask) as an autospecced stand-in that returns an "ok" answer. Any other optional module is
    missing."""
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


@pytest.mark.parametrize(
    "text",
    ["Ask your friends why ice floats", "Ask your friends", "Please ask everyone what to cook", "ask all friends"],
)
def test_asking_outside_models_is_answered_plainly_and_asks_nothing(panel, monkeypatch, text):
    assert say(monkeypatch, text) == voice.NO_OUTSIDE_MODELS
    panel.assert_not_called()
