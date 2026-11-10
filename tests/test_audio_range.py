"""Room audio and line clips from the microphones' streams on the broker: Review's audio under recorded
video (/audio/range, /audio/spans) and a line's clip (/clip), read through the speech log."""

import io
import json
import time
import urllib.error
import urllib.request

import numpy as np
import pytest
import soundfile as sf

from liberational.dashboard import server
from speech_fake import fake

SR = 16_000
T0 = 1_800_000_000.0  # when the kept audio starts (seconds)
NS = 1_000_000_000


def tone(seconds: float, amplitude: float = 0.3) -> np.ndarray:
    return (amplitude * np.sin(2 * np.pi * 440 * np.arange(int(seconds * SR)) / SR)).astype(np.float32)


def decoded(wav: bytes) -> tuple[np.ndarray, int]:
    data, sr = sf.read(io.BytesIO(wav), dtype="float32")
    return data, sr


@pytest.fixture
def kept():
    """Ten seconds of a tone on the kitchen mic from T0, and nothing on the hall mic."""
    fake().put_audio("kitchen.audio", int(T0 * NS), tone(10.0))
    fake().mics = {"kitchen.audio": None, "hall.audio": None}


def test_cuts_the_stretch_between_two_times(kept):
    wav, info = server.audio_range(T0 + 2, T0 + 5)
    data, sr = decoded(wav)
    assert (info["stream"], sr) == ("kitchen.audio", SR)
    assert len(data) == 3 * SR and np.count_nonzero(data) > 0.9 * len(data)


def test_silence_fills_where_nothing_was_kept(kept):
    data, _ = decoded(server.audio_range(T0 - 2, T0 + 1)[0])
    assert len(data) == 3 * SR
    assert not np.any(data[: 2 * SR - 10]), "before the audio starts: silence, so the clip still lines up"
    assert np.count_nonzero(data[2 * SR + 10 :]) > 0.9 * SR


def test_the_microphone_of_the_newest_line_is_read_first(kept):
    fake().put_audio("hall.audio", int(T0 * NS), tone(10.0, amplitude=0.05))
    fake().say({"ts_start": T0, "text": "hi", "audio": {"stream": "hall.audio", "start_ns": int(T0 * NS)}})
    assert server.audio_range(T0 + 1, T0 + 2)[1]["stream"] == "hall.audio"


@pytest.mark.parametrize(
    ("start", "end"),
    [
        pytest.param(T0 + 5, T0 + 5, id="empty"),
        pytest.param(T0 + 5, T0 + 1, id="backwards"),
        pytest.param("x", T0 + 1, id="not-a-time"),
    ],
)
def test_refuses_a_bad_range(kept, start, end):
    with pytest.raises(ValueError):
        server.audio_range(start, end)


def test_no_microphone_on_the_broker_is_not_found():
    with pytest.raises(FileNotFoundError):
        server.audio_range(T0, T0 + 1)


def test_quiet_room_audio_comes_out_at_a_normal_level():
    """A mic that records speech near -55 dBFS played as silence. The served audio is brought up."""
    fake().put_audio("kitchen.audio", int(T0 * NS), tone(4.0, amplitude=10 ** (-55 / 20)))
    fake().mics = {"kitchen.audio": None}
    data, _ = decoded(server.audio_range(T0, T0 + 4)[0])
    assert 20 * np.log10(np.sqrt(np.mean(data**2))) > -35


def test_the_timeline_draws_each_microphones_kept_audio(serve, monkeypatch):
    monkeypatch.setattr(server, "DASH_PASSWORD", "")
    day = time.strftime("%Y-%m-%d", time.localtime(T0))
    fake().put_audio("kitchen.audio", int(T0 * NS), tone(10.0))
    fake().mics = {"kitchen.audio": None}
    with urllib.request.urlopen(serve(server.H) + f"/audio/spans?day={day}", timeout=5) as r:
        spans = json.loads(r.read())["spans"]
    assert spans == [{"device": "kitchen", "start": T0, "end": T0 + 10}]


@pytest.fixture
def pacific(monkeypatch):
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


@pytest.mark.parametrize("day,next_day", [("2026-03-08", "2026-03-09"), ("2026-11-01", "2026-11-02")])
def test_audio_day_bounds_keep_the_last_hour_and_exclude_the_next_day_across_dst(pacific, day, next_day):
    late = time.mktime(time.strptime(f"{day} 23:30", "%Y-%m-%d %H:%M"))
    after = time.mktime(time.strptime(f"{next_day} 00:30", "%Y-%m-%d %H:%M"))
    fake().put_audio("kitchen.audio", int(late * NS), tone(1.0))
    fake().put_audio("kitchen.audio", int(after * NS), tone(1.0))
    fake().mics = {"kitchen.audio": None}
    assert server.audio_spans(day) == [{"device": "kitchen", "start": late, "end": late + 1}]


def test_a_lines_clip_plays_its_own_stretch_and_says_when_nothing_was_kept(kept, serve, monkeypatch):
    monkeypatch.setattr(server, "DASH_PASSWORD", "")
    at = int((T0 + 3) * NS)
    fake().say({"ts_start": T0 + 3, "ts_end": T0 + 4, "device": "kitchen", "text": "hi",
                "audio": {"stream": "kitchen.audio", "start_ns": at, "end_ns": at + NS}})  # fmt: skip
    later = int((T0 + 60) * NS)  # past what was kept: still in the ring, or purged
    fake().say({"ts_start": T0 + 60, "ts_end": T0 + 61, "device": "kitchen", "text": "new",
                "audio": {"stream": "kitchen.audio", "start_ns": later, "end_ns": later + NS}})  # fmt: skip
    base = serve(server.H)
    with urllib.request.urlopen(base + f"/clip?line=kitchen@{T0 + 3:.3f}", timeout=5) as r:
        data, sr = decoded(r.read())
    assert len(data) == pytest.approx((1 + 2 * server.PAD_S) * sr, abs=sr // 100)
    with pytest.raises(urllib.error.HTTPError) as no_audio:
        urllib.request.urlopen(base + f"/clip?line=kitchen@{T0 + 60:.3f}", timeout=5)
    assert no_audio.value.code == 404
