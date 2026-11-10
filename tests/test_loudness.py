"""audio/loudness.py: quiet room audio comes up to a normal level for playback, with a limiter."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from liberational.audio import loudness

SR = 16000
CEILING = int(round(10 ** (-1 / 20) * 32768))  # -1 dBFS as an int16 magnitude


def dbfs_rms(x: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(np.square(x.astype(np.float64) / 32768))))


def quiet_room(seconds: float = 10.0, speech_dbfs: float = -55.0, slam_at: float | None = None) -> np.ndarray:
    """Like the Linux machine's Wave 3: a -80 dBFS floor, speech-like bursts at speech_dbfs (RMS) half the time, and
    maybe one -2 dBFS door slam."""
    rng = np.random.default_rng(7)
    n = int(seconds * SR)
    x = rng.normal(0, 10 ** (-80 / 20), n)
    t = np.arange(n) / SR
    talking = (t % 2) < 1  # one second on, one off
    voice = np.sin(2 * np.pi * 180 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
    voice *= 10 ** (speech_dbfs / 20) / np.sqrt(np.mean(np.square(voice)))
    x[talking] += voice[talking]
    if slam_at is not None:
        i = int(slam_at * SR)
        x[i : i + 400] += 10 ** (-2 / 20) * np.sign(np.sin(np.arange(400)))
    return np.round(np.clip(x, -1, 1 - 1 / 32768) * 32768).astype(np.int16)


def speech_dbfs(x: np.ndarray, seconds: float, skip: tuple[float, float] | None = None) -> float:
    """RMS of the speech seconds only (the first half of each two-second period), leaving out skip."""
    t = np.arange(len(x)) / SR
    keep = ((t % 2) > 0.1) & ((t % 2) < 0.9)
    if skip:
        keep &= (t < skip[0]) | (t > skip[1])
    return dbfs_rms(x[keep])


def test_brings_quiet_speech_up_to_the_target():
    out = loudness.normalize(quiet_room(speech_dbfs=-55), SR)
    assert abs(speech_dbfs(out, 10) - loudness.TARGET_DBFS) < 2


def test_a_door_slam_neither_sets_the_level_nor_clips():
    x = quiet_room(speech_dbfs=-50, slam_at=4.5)
    out = loudness.normalize(x, SR)
    assert np.abs(out).max() <= CEILING
    # Speech away from the slam still lands near the target: the slam was limited, not used to set the gain.
    assert abs(speech_dbfs(out, 10, skip=(4.3, 4.8)) - loudness.TARGET_DBFS) < 2


def test_a_long_loud_stretch_neither_sets_the_level_nor_passes_the_ceiling():
    """A whole second far louder than the speech (a vacuum cleaner): longer than the limiter's reach of a few frames,
    and a tenth of the clip, too much for a mean to ignore."""
    x = quiet_room(speech_dbfs=-55).astype(np.float64)
    i = 3 * SR
    x[i : i + SR] += 10 ** (-10 / 20) * 32768 * np.sin(2 * np.pi * 300 * np.arange(SR) / SR)
    out = loudness.normalize(np.round(np.clip(x, -32768, 32767)).astype(np.int16), SR)
    assert np.abs(out[i : i + SR].astype(np.int32)).max() <= CEILING
    assert np.abs(out[i : i + SR].astype(np.int32)).max() > CEILING * 0.5  # limited, not silenced
    assert abs(speech_dbfs(out, 10, skip=(2.9, 4.1)) - loudness.TARGET_DBFS) < 2


def test_never_turns_up_more_than_the_max_gain():
    x = quiet_room(speech_dbfs=-75)
    assert loudness.gain_for(x, SR) == np.float64(10 ** (loudness.MAX_GAIN_DB / 20))


def test_leaves_audio_that_is_already_loud_enough_alone():
    t = np.arange(2 * SR) / SR
    x = np.round(0.3 * np.sin(2 * np.pi * 440 * t) * 32768).astype(np.int16)  # about -13 dBFS RMS
    assert np.array_equal(loudness.normalize(x, SR), x)


def test_digital_silence_stays_silent():
    assert not loudness.normalize(np.zeros(SR, dtype=np.int16), SR).any()


def test_an_empty_clip_stays_empty():
    assert loudness.normalize(np.zeros(0, dtype=np.int16), SR).size == 0


samples = arrays(np.int16, st.integers(0, 3 * SR), elements=st.integers(-32768, 32767))


@settings(max_examples=60, deadline=None)
@given(samples)
def test_keeps_every_sample_in_place_and_under_the_ceiling(x):
    out = loudness.normalize(x, SR)
    assert out.dtype == np.int16 and len(out) == len(x)
    assert int(np.abs(out.astype(np.int32)).max(initial=0)) <= CEILING
    # A sample never changes sign (Review lines this up with video, and a flipped sample would be a different sound).
    assert (out.astype(np.int64) * x.astype(np.int64) >= 0).all()


@settings(max_examples=60, deadline=None)
@given(arrays(np.int16, st.integers(1, 3 * SR), elements=st.integers(-250, 250)))
def test_quiet_audio_is_only_ever_turned_up_by_the_one_gain(x):
    """Under +-250 (about -42 dBFS), even the largest gain stays under the ceiling, so no limiting: every sample is the
    input times the clip's one gain, at least 1, rounded."""
    gain = loudness.gain_for(x, SR)
    out = loudness.normalize(x, SR).astype(np.float64)
    assert 1 <= gain <= 10 ** (loudness.MAX_GAIN_DB / 20)
    # normalize deliberately multiplies in float32: gain conversion and multiplication each round once.
    expected = x.astype(np.float64) * gain
    precision = 2 * np.finfo(np.float32).eps * np.abs(expected)
    assert np.all(np.abs(out - expected) <= 0.5 + precision)
