"""What Transcriber.handle drops before ASR: utterances the Silero speech gate finds no speech in, and anything that
overlaps our own playback."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from liberational import config as cfg
from liberational import model_runtime
from liberational.audio import sensitivity
from liberational.listen import gate


def gate_hearing(monkeypatch, seconds):
    """A real SpeechGate whose Silero model reports `seconds` of speech in any clip."""
    threshold = sensitivity.tuning(sensitivity.default()).gate_threshold  # the gate a default mic is checked at
    g = gate.SpeechGate(threshold, min_speech_s=0.25, model="/nonexistent")
    monkeypatch.setattr(g, "speech_s", lambda audio: seconds)
    return g


@pytest.mark.parametrize("seconds, keep", [(0.1, False), (0.25, True), (2.0, True)])
def test_gate_keeps_clips_with_at_least_the_minimum_of_speech(monkeypatch, seconds, keep):
    g = gate_hearing(monkeypatch, seconds)
    assert g(np.zeros(16000, np.float32)) is keep
    assert g.last_speech_s == seconds


def test_gate_off_switch():
    with cfg.override({"audio": {"speech_gate": False}}):
        assert gate.speech_gate() is None


def test_gate_takes_its_threshold_from_the_sensitivity_and_its_minimum_from_config(monkeypatch, tmp_path):
    monkeypatch.setattr(cfg, "local_config", lambda: tmp_path / "pack.toml")
    (tmp_path / "pack.toml").write_text("[audio]\nsensitivity = 40\n")
    with cfg.override({"audio": {"speech_gate": True, "speech_gate_min_s": 0.3}}):
        g = gate.speech_gate()
    assert g is not None
    assert (g.threshold, g.min_speech_s) == (pytest.approx(0.46), 0.3)  # 0.6 - 0.0035 * 40


# The voice password is never written down as said: only a line that matches it reads "[password]".
VOICE = 4  # the voice the stand-in speaker database matches every line to


def framed_levels(levels):
    """One constant-RMS 20 ms frame per supplied level, independent of the VAD's estimator."""
    return np.repeat(10 ** (np.asarray(levels, dtype=np.float32) / 20), 320)


@pytest.mark.parametrize("background", [-60.0, -45.0, -30.0])
def test_steady_amplified_background_stops_holding_the_vad_open(background):
    vad = gate.VAD(margin_db=6.4, min_dbfs=-75)
    found = list(vad.feed(framed_levels([background] * 1250)))  # 25 seconds of a steady room
    assert not vad.in_speech
    assert vad.floor == pytest.approx(background, abs=1)
    assert all((end - start) / 16000 < 6 for start, end, _audio in found)


def test_background_learning_preserves_sustained_modulated_speech_and_closes_on_its_pause():
    vad = gate.VAD(margin_db=6.4, min_dbfs=-75)
    list(vad.feed(framed_levels([-50] * 250)))
    # Eight seconds of syllables: 100 ms near -25 dBFS, 100 ms near -36 dBFS, without a sentence pause.
    voice = framed_levels(([-25] * 5 + [-36] * 5) * 40)
    assert list(vad.feed(voice)) == []
    assert vad.in_speech
    ((start, end, audio),) = list(vad.feed(framed_levels([-50] * 75)))
    assert start / 16000 == pytest.approx(4.7, abs=0.1)
    assert end / 16000 == pytest.approx(13.3, abs=0.15)
    assert len(audio) / 16000 > 8


def test_background_gain_shift_does_not_make_the_estimator_relearn_old_levels():
    vad = gate.VAD(margin_db=6.4, min_dbfs=-75)
    list(vad.feed(framed_levels([-55] * 250)))
    vad.shift(20)
    assert list(vad.feed(framed_levels([-35] * 100))) == []
    assert not vad.in_speech
    assert vad.floor == pytest.approx(-35, abs=1)


def test_lazy_silero_load_is_reported_only_on_first_use(tmp_path, monkeypatch):
    model = tmp_path / "speech-gate.onnx"
    model.touch()
    detector = object()
    fake = SimpleNamespace(
        VadModelConfig=lambda: SimpleNamespace(silero_vad=SimpleNamespace()),
        VoiceActivityDetector=lambda config, buffer_size_in_seconds: detector,
    )
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake)
    speech_gate = gate.SpeechGate(model=model)
    with cfg.override({"paths": {"state": str(tmp_path)}}), model_runtime.reporting("listen"):
        (report,) = (tmp_path / "model_loads").glob("*.json")
        assert json.loads(report.read_text())["models"] == []
        assert speech_gate.detector() is detector
        assert speech_gate.detector() is detector
        (loaded,) = json.loads(report.read_text())["models"]
        assert (loaded["key"], loaded["name"], loaded["purpose"], loaded["device"]) == (
            "models.files.vad_silero",
            "speech-gate.onnx",
            "speech detection",
            "CPU",
        )
