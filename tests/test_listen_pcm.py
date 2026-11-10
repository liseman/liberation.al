import struct

import numpy as np
import pytest

from liberational.listen.pcm import Continuity, Format, Resampler


def record(host_ns: int, first: int, body: bytes) -> bytes:
    return struct.pack("<QQ", host_ns, first) + body


def test_a_wave3_record_is_48_khz_from_its_stride_and_block() -> None:
    fmt = Format.of("PCM_S24LE3x1", 16 + 480 * 3, block_ms=10)
    assert (fmt.rate_hz, fmt.frames, fmt.channels, fmt.stride) == (48_000, 480, 1, 1_456)


@pytest.mark.parametrize(
    ("shape", "stride", "block_ms", "match"),
    [
        pytest.param("WAV", 100, 10, "not a PCM shape", id="shape"),
        pytest.param("PCM_S16LEx2", 16 + 3, 10, "doesn't fit", id="partial-frame"),
        pytest.param("PCM_S16LEx1", 16 + 2 * 7, 3, "whole rate", id="rate"),
    ],
)
def test_layouts_that_dont_fit_are_refused(shape: str, stride: int, block_ms: int, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Format.of(shape, stride, block_ms)


@pytest.mark.parametrize(
    ("shape", "frames_bytes", "expected"),
    [
        # Two frames of two channels; channel 1 holds the values under test.
        pytest.param("PCM_S16LEx2", struct.pack("<4h", 0, 16384, 0, -32768), [0.5, -1.0], id="s16"),
        pytest.param("PCM_S32LEx2", struct.pack("<4i", 0, 1 << 30, 0, -(1 << 31)), [0.5, -1.0], id="s32"),
        pytest.param("PCM_F32LEx2", struct.pack("<4f", 0, 0.25, 0, -0.75), [0.25, -0.75], id="f32"),
        # 0x400000 is half of full scale; 0x800000 is the most negative 24-bit value.
        pytest.param(
            "PCM_S24LE3x2", bytes([0, 0, 0, 0, 0, 0x40, 0, 0, 0, 0, 0, 0x80]), [0.5, -1.0], id="s24-sign-extended"
        ),
    ],
)
def test_one_channel_decodes_to_floats_with_the_stamp(shape: str, frames_bytes: bytes, expected: list[float]) -> None:
    fmt = Format.of(shape, 16 + len(frames_bytes), block_ms=1)  # 2 frames per 1 ms record: a 2 kHz toy rate
    host_ns, first, x = fmt.decode(record(123_456, 7, frames_bytes), channel=1)
    assert (host_ns, first) == (123_456, 7)
    assert x.dtype == np.float32
    assert x.tolist() == pytest.approx(expected)


def test_a_record_of_the_wrong_size_or_channel_is_refused() -> None:
    fmt = Format.of("PCM_S16LEx1", 16 + 4, block_ms=1)
    with pytest.raises(ValueError, match="bytes"):
        fmt.decode(b"\0" * 19)
    with pytest.raises(ValueError, match="channel 1 of 1"):
        fmt.decode(b"\0" * 20, channel=1)


def test_continuity_accepts_the_next_block_and_names_each_break() -> None:
    fmt = Format.of("PCM_S16LEx1", 16 + 480 * 2, block_ms=10)  # 48 kHz, 10 ms blocks
    c = Continuity(fmt)
    assert c.check(1_000_000_000, 0) is None, "the first record has nothing to follow"
    assert c.check(1_010_000_000, 480) is None
    assert c.check(1_020_400_000, 960) is None, "0.4 ms of clock correction is within tolerance"
    assert "records lost" in (c.check(1_030_400_000, 1_440, lost=2) or "")
    assert "sample 2400 after 1440" in (c.check(1_050_400_000, 2_400) or "")
    assert "ms off" in (c.check(9_000_000_000, 2_880) or ""), "a stamp far from the block clock (a torn record)"


def test_resampling_48_khz_keeps_the_tone_and_its_phase_across_blocks() -> None:
    t = np.arange(48_000) / 48_000
    tone = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    r = Resampler(48_000)
    out = np.concatenate([r(tone[i : i + 480]) for i in range(0, len(tone), 480)])
    assert len(out) == pytest.approx(16_000, abs=1)
    # The anti-alias filter delays everything by half its length (a linear-phase FIR), 1.7 ms at 48 kHz.
    delay_s = (Resampler.TAPS - 1) / 2 / 48_000
    expected = np.sin(2 * np.pi * 440 * (np.arange(len(out)) / 16_000 - delay_s))
    settled = Resampler.TAPS  # past the filter's start, where it hasn't seen a whole window yet
    assert np.max(np.abs(out - expected)[settled:]) < 0.02, "block edges add no discontinuity"


def test_16_khz_passes_through() -> None:
    x = np.arange(5, dtype=np.float32)
    assert Resampler(16_000)(x) is x


def test_resampling_to_16k_filters_out_what_would_fold_into_the_speech_band() -> None:
    # Without a low-pass, 48 kHz audio's 8 to 24 kHz folds into 0 to 8 kHz: the Wave 3's hiss (45% of its energy)
    # landed in the speech band (2026-10-30). A 12 kHz tone would alias to 4 kHz; a 1 kHz tone is speech.
    from liberational.listen.pcm import Resampler

    t = np.arange(48_000) / 48_000

    def through(freq: float) -> float:
        r = Resampler(48_000)
        x = np.sin(2 * np.pi * freq * t).astype(np.float32)
        y = np.concatenate([r(x[i : i + 480]) for i in range(0, len(x), 480)])
        return float(np.sqrt(np.mean(y[1_000:] ** 2)) * np.sqrt(2))  # amplitude, past the filter's start

    assert through(1_000) == pytest.approx(1.0, abs=0.05)
    assert through(12_000) < 0.01, "folded into the speech band"
