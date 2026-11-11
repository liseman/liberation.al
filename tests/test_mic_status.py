"""The Live tab's microphone status and live levels on Continuum: from the listen worker and each
microphone stream's freshness on the broker, not v1's ambient.status.json."""

import json
import time
import urllib.request

import pytest

from liberational.dashboard import context, live_status, mic_levels
from speech_fake import fake


@pytest.fixture(autouse=True)
def fresh_cache(monkeypatch):
    monkeypatch.setitem(live_status._mic_cache, "value", None)


@pytest.mark.parametrize(
    ("listening", "age_s", "healthy"),
    [
        pytest.param(True, 1.0, True, id="listening-and-fresh"),
        pytest.param(True, 60.0, False, id="audio-stopped"),
        pytest.param(False, 1.0, False, id="listen-not-running"),
        pytest.param(True, None, False, id="no-audio-yet"),
    ],
)
def test_a_microphone_is_heard_while_listen_runs_and_its_audio_is_fresh(monkeypatch, listening, age_s, healthy):
    monkeypatch.setattr(live_status, "_listening", lambda: listening)
    fake().mics = {"wave3.audio": None if age_s is None else time.time_ns() - int(age_s * 1e9)}
    status = live_status.mic_status()
    assert status["devices"] == {"wave3": {"name": "wave3", "healthy": healthy}}
    assert ("ts" in status) is healthy  # the Live tab reads "listening" from a fresh ts


def test_a_microphones_auto_gain_is_shown_with_it(monkeypatch, tmp_path):
    from liberational.audio import gain

    monkeypatch.setattr(context, "STATE", tmp_path)
    monkeypatch.setattr(live_status, "_listening", lambda: True)
    fake().mics = {"wave3.audio": time.time_ns()}
    mics = {"alsa_input.usb-Acme_Mic-00.mono-fallback": {"name": "wave3", "auto": True, "kind": "hardware", "db": 30.0}}
    (tmp_path / gain.STATUS).write_text(json.dumps({"ts": time.time(), "mics": mics}))
    status = live_status.mic_status()
    assert status["devices"]["wave3"]["gain"] == {"auto": True, "kind": "hardware", "db": 30.0}


def test_the_gain_status_is_read_from_the_state_and_not_kept_across_a_change(monkeypatch, tmp_path):
    """Two state directories inside the 2 s window are two answers: the gain file is read from `context.STATE`, and
    the cached answer is keyed on it."""
    from liberational.audio import gain

    monkeypatch.setattr(live_status, "_listening", lambda: True)
    fake().mics = {"wave3.audio": time.time_ns()}
    mics = {"usb-Acme_Mic": {"name": "wave3", "auto": True, "kind": "hardware", "db": 30.0}}
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / gain.STATUS).write_text(json.dumps({"ts": time.time(), "mics": mics}))
    monkeypatch.setattr(context, "STATE", first)
    assert live_status.mic_status()["devices"]["wave3"]["gain"] == {"auto": True, "kind": "hardware", "db": 30.0}
    monkeypatch.setattr(context, "STATE", second)  # no gain status here
    assert "gain" not in live_status.mic_status()["devices"]["wave3"]


def test_the_route_serves_the_status(serve_dashboard, monkeypatch):
    monkeypatch.setattr(live_status, "_listening", lambda: True)
    fake().mics = {"wave3.audio": time.time_ns()}
    with urllib.request.urlopen(serve_dashboard() + "/ambient", timeout=5) as r:
        body = json.loads(r.read())
    assert body["devices"]["wave3"]["healthy"] is True and body["ts"] > 0


@pytest.mark.parametrize(
    ("shape", "stride", "frames"),
    [
        pytest.param("PCM_F32LEx1", 16 + 480 * 4, 480, id="wave3"),
        pytest.param("PCM_S16LEx2", 16 + 160 * 4, 160, id="stereo-int16"),
        pytest.param("PCM_S16LEx2", 16 + 3, None, id="stride-not-whole-frames"),
        pytest.param("RGB8", 64, None, id="not-pcm"),
    ],
)
def test_a_rings_layout_comes_from_its_shape_and_stride(shape, stride, frames):
    fmt = mic_levels.layout(shape, stride)
    assert (fmt.frames if fmt else None) == frames
