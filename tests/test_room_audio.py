"""room_audio: a ring batch from a flushed chunk split back into its records, the layout read from the batch itself."""

import numpy as np
import pytest

from liberational import room_audio
from liberational.listen.pcm import STAMP


def batch(kind: str, channels: int, frames: int, records: int, dtype: str) -> bytes:
    """`records` ring records of `frames` frames, each stamped with its first sample's index, samples counting up."""
    out = b""
    for r in range(records):
        samples = (np.arange(frames * channels) + r * frames * channels).astype(dtype)
        out += STAMP.pack(1_000 + r, r * frames) + samples.tobytes()
    return out


@pytest.mark.parametrize(
    ("shape", "dtype", "channels", "frames", "rate"),
    [
        pytest.param("PCM_F32LEx1", "<f4", 1, 480, 48_000, id="wave3-48k-10ms"),
        pytest.param("PCM_S16LEx2", "<i2", 2, 160, 16_000, id="stereo-16k-10ms"),
    ],
)
def test_a_batch_splits_into_its_records_with_the_stride_and_rate_it_was_captured_at(
    shape, dtype, channels, frames, rate
):
    b = batch(shape, channels, frames, 3, dtype)
    got = room_audio.split_batch(shape, b, span_ns=30_000_000)  # three 10 ms records
    assert got is not None
    fmt, recs = got
    assert (fmt.frames, fmt.channels, fmt.rate_hz, len(recs)) == (frames, channels, rate, 3)
    assert recs[1] == b[len(b) // 3 : 2 * len(b) // 3]


@pytest.mark.parametrize(
    ("shape", "data"),
    [
        pytest.param("RGB8", b"\0" * 64, id="not-pcm"),
        pytest.param("PCM_S16LEx2", b"\0" * 7, id="shorter-than-a-stamp"),
        pytest.param("PCM_S16LEx2", b"\0" * (STAMP.size + 3), id="not-whole-frames"),
    ],
)
def test_bytes_that_arent_whole_records_are_refused(shape, data):
    assert room_audio.split_batch(shape, data, span_ns=10_000_000) is None


@pytest.mark.parametrize("span_ns", [1, 100_000_000])
def test_explicit_archive_format_uses_true_rate_and_stride_despite_jitter(span_ns):
    data = batch("F32LE", 1, 120, 3, "<f4")
    got = room_audio.split_batch("PCM_F32LEx1", data, span_ns, room_audio.PcmFormat(12_000, 1, 496))
    assert got is not None
    fmt, records = got
    assert (fmt.rate_hz, fmt.frames, len(records)) == (12_000, 120, 3)
    assert records[1] == data[496:992]


@pytest.mark.parametrize(
    "format",
    [room_audio.PcmFormat(0, 1, 496), room_audio.PcmFormat(12_000, 2, 496), room_audio.PcmFormat(12_000, 1, 17)],
)
def test_invalid_explicit_archive_format_does_not_fall_back_to_guessing(format):
    data = batch("F32LE", 1, 120, 3, "<f4")
    assert room_audio.split_batch("PCM_F32LEx1", data, 30_000_000, format) is None
