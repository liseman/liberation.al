"""Barge-in waits for Bob's first audio: a caller's "hello?" into the silence before it isn't talking over Bob."""

import asyncio
import base64
import json
import threading

import numpy as np
import pytest

bridge = pytest.importorskip("liberational.phone.bridge")


def loud_frame() -> str:
    pcm = (np.sin(np.arange(160) / 3) * 12000).astype(np.int16)
    return base64.b64encode(bridge.ulaw_encode(pcm)).decode()


@pytest.mark.parametrize("sent", [False, True])
def test_barge_in_only_after_bob_has_sounded(sent: bool) -> None:
    call = bridge.Call(object())
    call.state = "speaking"
    call.sent_audio = sent

    async def feed() -> None:
        for _ in range(bridge.BARGE_FRAMES + 2):
            await call.on_media(loud_frame())

    asyncio.run(feed())
    assert call.interrupted is sent


class Marks:
    """The media-stream socket: answers each mark at once, as Twilio does once the audio before it has played."""

    def __init__(self, call: bridge.Call) -> None:
        self.call = call

    async def send_text(self, text: str) -> None:
        if '"mark"' in text:
            self.call.mark_event.set()


def test_mission_greeting_is_audible_before_model_result(monkeypatch: pytest.MonkeyPatch, model_server) -> None:
    call = bridge.Call(object())
    call.verified_owner = True
    call.say = "Arrange dinner."
    heard = threading.Event()
    messages = []
    pcm = np.zeros(160, np.int16)
    monkeypatch.setattr(bridge, "FIXED_AUDIO", {"It's Bob.": pcm, "When works?": pcm})

    def model_request(sent):
        assert heard.wait(2), "mission generation blocked the initial greeting"
        return {"message": {"content": "When works?"}}

    class Socket:
        async def send_text(self, text: str) -> None:
            event = json.loads(text)
            messages.append(event)
            if event["event"] == "media":
                heard.set()
            if event["event"] == "mark":
                call.mark_event.set()

    call.ws = Socket()
    model_server.reply = model_request
    asyncio.run(call.greet())
    assert [entry["text"] for entry in call.transcript] == ["It's Bob.", "When works?"]
    assert len([message for message in messages if message["event"] == "media"]) == 2
    assert call.state == "listening"


def test_cached_clip_stops_sending_when_caller_interrupts(monkeypatch: pytest.MonkeyPatch) -> None:
    call = bridge.Call(object())
    events = []

    class Socket:
        async def send_text(self, text: str) -> None:
            event = json.loads(text)
            events.append(event)
            if event["event"] == "media":
                call.interrupted = True
            if event["event"] == "mark":
                call.mark_event.set()

    call.ws = Socket()
    monkeypatch.setattr(bridge, "FIXED_AUDIO", {"It's Bob.": np.zeros(160 * 5, np.int16)})
    asyncio.run(call.speak("It's Bob."))
    assert [event["event"] for event in events] == ["media", "clear", "mark"]


def test_cached_farewell_never_waits_for_synthesis(monkeypatch: pytest.MonkeyPatch) -> None:
    call = bridge.Call(object())
    call.ws = Marks(call)
    monkeypatch.setattr(bridge, "FIXED_AUDIO", {"Okay, bye!": np.zeros(160, np.int16)})

    def no_synthesis(text, on_chunk):
        pytest.fail("cached farewell tried to synthesize")

    monkeypatch.setattr(bridge, "synthesize_stream", no_synthesis)
    asyncio.run(call.speak("Okay, bye!"))
    assert len(call.bot_segments) == 1


def test_speech_that_started_while_thinking_outlives_a_direct_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    # A caller who goes on after a pause while a direct reply (a password answer, "You're already through.") is
    # made: speak() used to clear what was buffered, so the rest of what they said was lost.
    call = bridge.Call(object())
    call.ws = Marks(call)
    monkeypatch.setattr(bridge, "synthesize_stream", lambda text, on_chunk: on_chunk(np.zeros(160, np.int16)))
    heard: list[int] = []

    async def handle_utterance(frames: list[np.ndarray]) -> None:
        heard.append(len(frames))

    monkeypatch.setattr(call, "handle_utterance", handle_utterance)
    quiet = base64.b64encode(bridge.ulaw_encode(np.zeros(160, np.int16))).decode()

    async def scenario() -> None:
        call.state = "thinking"
        for _ in range(30):
            await call.on_media(loud_frame())
        await call.speak("You're already through.")
        for _ in range(bridge.END_SILENCE_FRAMES):
            await call.on_media(quiet)
        await asyncio.sleep(0)

    asyncio.run(scenario())
    assert len(heard) == 1 and heard[0] >= 30  # the speech from before the reply, then the pause that ended it
