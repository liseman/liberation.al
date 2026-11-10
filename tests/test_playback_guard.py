import json
import sys
import threading
from contextlib import contextmanager
from itertools import repeat
from types import SimpleNamespace

import numpy as np
import pytest

from liberational.assistant import voice
from liberational.assistant.voice_auth import Conversation
from liberational.speech import playback_guard as g
from liberational.speech import tts, voice_clone

NOW = 100.0


@pytest.fixture
def windows(tmp_path, monkeypatch):
    """The playback windows file, in a temp dir, with the guard's clock stopped at NOW."""
    path = tmp_path / "windows.json"
    monkeypatch.setattr(g, "PATH", path)
    monkeypatch.setattr(g, "time", SimpleNamespace(time=lambda: NOW))
    return path


def test_a_playing_window_covers_the_clip_and_its_lead_in(windows):
    with g.playback_window(4):
        assert g.overlaps(99, 101)
        assert g.overlaps(102, 104)


@pytest.mark.parametrize(
    "start, end, hit",
    [
        pytest.param(100.5, 100.6, True, id="in-the-tail"),
        pytest.param(99, 100, True, id="delayed-asr-of-earlier-audio"),
        pytest.param(101, 102, False, id="after-the-tail"),
    ],
)
def test_a_finished_window_keeps_a_short_tail(windows, start, end, hit):
    with g.playback_window(4):
        pass
    assert g.overlaps(start, end) is hit


def test_a_failed_playback_still_finishes_its_window(windows):
    with pytest.raises(RuntimeError), g.playback_window(4):
        raise RuntimeError
    assert json.loads(windows.read_text())[0]["end"] == 100.6


def test_a_corrupt_windows_file_fails_closed(windows):
    windows.write_text("{")
    assert g.overlaps(100, 101)
    with pytest.raises(ValueError), g.playback_window(1):
        pytest.fail("must not play")


@pytest.mark.parametrize(
    "start, end",
    [
        pytest.param("abc", None, id="non-numeric-start"),
        pytest.param(float("nan"), None, id="nan-start"),
        pytest.param(100, float("nan"), id="nan-end"),
        pytest.param(float("inf"), None, id="infinite-start"),
    ],
)
def test_a_bad_timestamp_fails_closed(windows, start, end):
    with g.playback_window(4):
        pass
    assert g.overlaps(start, end)


# Conversation's own tests live with voice_auth; this one pins its use of the playback guard.
@pytest.mark.parametrize(
    "entry",
    [
        pytest.param({"ts_start": 99, "ts_end": 105}, id="overlaps-playback"),
        pytest.param({"source": "tts"}, id="tagged-as-tts"),
    ],
)
def test_playback_cannot_confirm_a_pending_request(windows, tmp_path, entry):
    # The owner is never left pending, so a recognized guest holds the pending request.
    c = Conversation(tmp_path / "auth", clock=lambda: 105)
    c.accept("Hey Bob, restart service", {"speaker": "guest"}, False, lambda _: False)
    assert c.pending is not None
    windows.write_text(json.dumps([{"start": 100, "end": 110}]))
    assert c.accept("yes confirm", {"speaker": "guest", **entry}, False, lambda _: False) is None
    assert c.pending is not None


@pytest.fixture
def guarded_while_playing(windows):
    """A stand-in player that records whether the guard covered the moment it played."""
    seen = []

    def play(*args, **kwargs):
        seen.append(g.overlaps(NOW))

    return play, seen


def test_afplay_runs_inside_a_playback_window(monkeypatch, guarded_while_playing):
    play, seen = guarded_while_playing
    monkeypatch.setattr(voice_clone.sys, "platform", "darwin")
    monkeypatch.setattr("subprocess.run", play)
    voice_clone.play_audio(np.zeros(2400), 24000)
    assert seen == [True]


def test_sounddevice_plays_inside_a_playback_window(monkeypatch, guarded_while_playing):
    play, seen = guarded_while_playing
    monkeypatch.setattr(voice_clone.sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(play=play, wait=play))
    voice_clone.play_audio(np.zeros(2400), 24000)
    assert seen == [True, True]


def test_voice_plays_the_first_chunk_before_synthesis_finishes(monkeypatch, windows):
    first_played = threading.Event()
    played = []

    class Output:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def write(self, audio):
            assert g.overlaps(NOW)
            played.append(audio.copy())
            first_played.set()

        def stop(self):
            assert g.overlaps(NOW)

    def chunks(text):
        yield np.full(2400, 0.25, np.float32), 24000
        assert first_played.wait(2), "the first chunk waited for all synthesis to finish"
        yield np.full(2400, 0.5, np.float32), 24000

    monkeypatch.setattr(voice_clone.sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(OutputStream=Output))
    engine = object.__new__(tts.TTS)
    monkeypatch.setattr(engine, "generate", chunks)
    speaker = object.__new__(voice.Speaker)
    speaker.tts = engine
    speaker.speak("A first sentence, then the rest.")
    np.testing.assert_array_equal(np.concatenate(played).reshape(-1), np.repeat([0.25, 0.5], 2400))


def test_streaming_output_failure_stops_a_producer_with_a_full_queue(monkeypatch, windows):
    closed = threading.Event()

    def fail(audio):
        raise RuntimeError("speaker disconnected")

    @contextmanager
    def output(**kwargs):
        try:
            yield SimpleNamespace(write=fail)
        finally:
            assert g.overlaps(NOW + 2), "device drain outlasted the closed write guard"
            closed.set()

    monkeypatch.setattr(voice_clone.sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(OutputStream=output))
    with pytest.raises(RuntimeError, match="speaker disconnected"):
        voice_clone.play_chunks(repeat((np.ones(240, np.float32), 24000), 20))
    assert closed.is_set()
    assert json.loads(windows.read_text())[0]["end"] == NOW + g.TAIL


def test_synthesis_failure_before_audio_does_not_open_a_player(monkeypatch):
    def chunks():
        raise RuntimeError("synthesis failed")
        yield  # pragma: no cover

    monkeypatch.setattr(voice_clone.sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace())
    with pytest.raises(RuntimeError, match="synthesis failed"):
        voice_clone.play_chunks(chunks())


def test_writer_failure_closes_the_model_iterator_on_its_own_thread(monkeypatch, windows):
    model_lock = threading.Lock()
    finalized = []

    def chunks():
        with model_lock:
            try:
                yield from repeat((np.ones(240, np.float32), 24000), 20)
            finally:
                finalized.append(threading.get_ident())

    def fail(audio):
        raise RuntimeError("speaker disconnected")

    @contextmanager
    def output(**kwargs):
        yield SimpleNamespace(write=fail)

    monkeypatch.setattr(voice_clone.sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(OutputStream=output))
    with pytest.raises(RuntimeError, match="speaker disconnected") as caught:
        voice_clone.play_chunks(chunks())
    assert caught.value.__traceback__ is not None
    assert finalized == [threading.get_ident()]
    assert not model_lock.locked()


def test_macos_chunked_reply_keeps_the_fresh_player_path(monkeypatch):
    played = []
    monkeypatch.setattr(voice_clone.sys, "platform", "darwin")
    monkeypatch.setattr(voice_clone, "play_audio", lambda a, sr: played.append((a, sr)))
    voice_clone.play_chunks([(np.full(240, 0.25, np.float32), 24000), (np.full(240, 0.5, np.float32), 24000)])
    assert len(played) == 1 and played[0][1] == 24000
    np.testing.assert_array_equal(played[0][0], np.repeat([0.25, 0.5], 240))


def test_model_output_buffer_reuse_does_not_change_queued_speech(monkeypatch, windows):
    reused = threading.Event()
    played = []

    def write(audio):
        assert reused.wait(2)
        played.append(audio.copy())

    @contextmanager
    def output(**kwargs):
        yield SimpleNamespace(write=write)

    def chunks():
        buffer = np.full(240, 0.25, np.float32)
        yield buffer, 24000
        buffer[:] = 0.5
        reused.set()
        yield buffer, 24000

    monkeypatch.setattr(voice_clone.sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(OutputStream=output))
    voice_clone.play_chunks(chunks())
    np.testing.assert_array_equal(np.concatenate(played).reshape(-1), np.repeat([0.25, 0.5], 240))
