"""Microphone sensitivity: one 0 to 100 setting per mic, applied to the energy VAD and the speech gate live."""

import tomllib

import numpy as np
import pytest

from liberational import config as cfg
from liberational.audio import sensitivity

SR = 16_000  # the rate the listen worker hands the VAD
from liberational.listen import gate


@pytest.fixture
def pack_toml(tmp_path, monkeypatch):
    path = tmp_path / "pack.toml"
    monkeypatch.setattr(cfg, "local_config", lambda: path)
    return path


@pytest.mark.parametrize(
    ("value", "margin", "floor", "gate"),
    [(0, 12.0, -40.0, 0.6), (70, 6.4, -75.0, 0.355), (100, 4.0, -90.0, 0.25), (150, 4.0, -90.0, 0.25)],
)
def test_one_setting_moves_the_vad_and_the_gate_together(value, margin, floor, gate):
    t = sensitivity.tuning(value)
    assert (t.margin_db, t.min_dbfs, t.gate_threshold) == (pytest.approx(margin), floor, pytest.approx(gate))


def test_a_mic_follows_the_default_until_it_has_its_own(pack_toml):
    assert sensitivity.value("Mic") == 70
    sensitivity.save("Mic", 85)
    assert (sensitivity.value("Mic"), sensitivity.value("Other")) == (85, 70)
    pack_toml.write_text(pack_toml.read_text() + "\n[owner]\nname = 'Sam'\n")  # an edit by hand is read too
    sensitivity.save("Mic", None)
    saved = tomllib.loads(pack_toml.read_text())
    assert "audio" not in saved
    assert saved["owner"] == {"name": "Sam"}
    assert sensitivity.value("Mic") == 70  # back on the default at once, not the setting the process started with


def test_the_default_is_read_from_pack_toml_as_it_is_now(pack_toml):
    pack_toml.write_text("[audio]\nsensitivity = 50\n")
    assert sensitivity.value("Mic") == 50
    pack_toml.write_text("")
    assert sensitivity.value("Mic") == 70


def test_a_setting_out_of_range_is_refused(pack_toml):
    pack_toml.write_text("[audio.mic_sensitivity]\nMic = 140\n")
    with pytest.raises(ValueError, match="0 to 100"):
        sensitivity.value("Mic")


def tone(dbfs, seconds):
    t = np.arange(int(seconds * SR)) / SR
    return (10 ** (dbfs / 20) * np.sqrt(2) * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def speech_at(dbfs):
    """A second of quiet room, a second of 'speech' at dbfs, and a second of quiet again."""
    return np.concatenate([tone(-90, 1), tone(dbfs, 1), tone(-90, 1)])


def test_speech_at_this_rooms_level_opens_at_the_default_and_not_at_the_old_values():
    old, new = gate.VAD(margin_db=9, min_dbfs=-45), gate.VAD()
    new.tune(sensitivity.tuning(70).margin_db, sensitivity.tuning(70).min_dbfs)
    assert list(old.feed(speech_at(-55))) == []
    assert len(list(new.feed(speech_at(-55)))) == 1


def test_a_stretch_just_under_the_threshold_is_noted_as_a_near_miss():
    vad = gate.VAD(margin_db=9, min_dbfs=-45)
    assert list(vad.feed(speech_at(-48))) == []
    assert list(vad.near_misses) == [(pytest.approx(-48.0, abs=0.5), -45.0)]


def test_room_noise_near_the_floor_is_not_a_near_miss():
    rng = np.random.default_rng(1)
    vad = gate.VAD(margin_db=6.4, min_dbfs=-75)
    noise = np.concatenate([tone(-88 + rng.uniform(-1.5, 1.5), 0.02) for _ in range(3000)])  # a minute, wobbling
    assert list(vad.feed(noise)) == []
    assert list(vad.near_misses) == []
