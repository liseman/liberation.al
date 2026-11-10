from typing import Any

import pytest

from liberational import speech_log
from liberational.assistant import voice


class FakeLog:
    """What voice.py asks of the speech log on Continuum, recorded."""

    def __init__(self) -> None:
        self.turns: list[str] = []
        self.appended: list[tuple[str, dict[str, Any]]] = []

    def set_voice_turn(self, state: str) -> None:
        self.turns.append(state)

    def append(self, stream: str, line: dict[str, Any]) -> str:
        self.appended.append((stream, line))
        return "id"


class FakeSpeaker:
    def __init__(self, log: FakeLog) -> None:
        self.log = log
        self.turn_while_speaking: str | None = None

    def speak(self, text: str) -> None:
        self.turn_while_speaking = self.log.turns[-1]


@pytest.fixture
def on_continuum(monkeypatch: pytest.MonkeyPatch) -> FakeLog:
    log = FakeLog()
    monkeypatch.setattr(voice, "CONTINUUM", log)
    monkeypatch.setattr(voice, "_TURN", "Idle")
    monkeypatch.setattr(voice, "status", lambda *a, **k: None)
    monkeypatch.setattr(voice, "canonical_entry", lambda e: dict(e))
    monkeypatch.setattr(voice, "owner_entry", lambda e: False)
    return log


def test_a_reply_thinks_then_speaks_then_goes_idle_and_is_committed(
    on_continuum: FakeLog, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(voice.CONVERSATION, "accept", lambda *a: "It's ten past nine.")
    speaker = FakeSpeaker(on_continuum)
    heard = {"id": "0190c8e0-0000-7000-8000-000000000009", "ts_start": 5.0, "text": "hey bob what time is it"}
    reply = voice.handle(heard["text"], speaker, heard)  # ty: ignore[invalid-argument-type]
    assert reply == "It's ten past nine."
    assert on_continuum.turns == ["Listening", "Thinking", "Speaking", "Idle"]
    assert speaker.turn_while_speaking == "Speaking"
    ((stream, line),) = on_continuum.appended
    assert stream == speech_log.REPLIES
    assert line["text"] == "It's ten past nine."
    assert line["reply_to"] == heard["id"], "a reply names the utterance it answers"
    assert "assistant" not in line


def test_a_line_that_isnt_for_the_assistant_leaves_the_turn_alone(
    on_continuum: FakeLog, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(voice.CONVERSATION, "accept", lambda *a: None)
    assert voice.handle("just talking", None, {"text": "just talking"}) is None
    assert on_continuum.turns == []
    assert on_continuum.appended == []


def test_status_lines_stay_in_the_log_on_continuum(on_continuum: FakeLog) -> None:
    voice._append({"ts_start": 1.0, "text": "voice ready", "status": True})
    assert on_continuum.appended == []


def test_without_continuum_the_turn_is_never_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(voice, "CONTINUUM", None)
    voice._turn("Thinking")  # nothing to call: no error, nothing recorded
