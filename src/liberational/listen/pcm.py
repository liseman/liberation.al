"""PCM ring records from capture (docs/CONTINUUM.md, "Rings"), as the audio worker reads them.

A microphone's ring holds one record per capture block: a 16-byte stamp (the block's capture time on the host
monotonic clock, then the index of its first sample, both u64 little-endian), then the block's interleaved frames in
the device's native format, every channel. The stream's shape names the format: `PCM_S16LEx1`, `PCM_S24LE3x2`,
`PCM_S32LEx4`, or `PCM_F32LEx6`.

What this module adds on top:

- `Format.decode`: one channel of a record as float32 in [-1, 1].
- `Continuity`: whether a record follows on from the last one. A gap (a lost record, an overrun, a restarted
  capture) or a record whose stamp doesn't fit is reported, so the reader can close an utterance at the gap rather
  than splice two times together. The stamp check also catches a record torn by a stalled writer after a forced
  ring handover (the one case where a reader can validate mixed bytes).
- `Resampler`: v1's stateful linear resample to 16 kHz, which keeps its phase across blocks.
- `HostClock`: host monotonic nanoseconds to UTC, for a stamp taken on this machine.
"""

import re
import struct
import time
from dataclasses import dataclass

import numpy as np

STAMP = struct.Struct("<QQ")
SR = 16_000

_SHAPE = re.compile(r"PCM_(S16LE|S24LE3|S32LE|F32LE)x(\d+)")
_BYTES = {"S16LE": 2, "S24LE3": 3, "S32LE": 4, "F32LE": 4}


@dataclass(frozen=True)
class Format:
    """A ring stream's record layout."""

    kind: str
    channels: int
    rate_hz: int
    frames: int

    @property
    def bytes_per_sample(self) -> int:
        return _BYTES[self.kind]

    @property
    def stride(self) -> int:
        return STAMP.size + self.frames * self.channels * self.bytes_per_sample

    @classmethod
    def of(cls, shape: str, stride: int, block_ms: int) -> Format:
        """The layout of a stream with this shape name and ring stride, captured in `block_ms` blocks."""
        m = _SHAPE.fullmatch(shape)
        if m is None:
            msg = f"not a PCM shape: {shape!r}"
            raise ValueError(msg)
        kind, channels = m.group(1), int(m.group(2))
        frame = channels * _BYTES[kind]
        if channels < 1 or stride <= STAMP.size or (stride - STAMP.size) % frame:
            msg = f"a stride of {stride} doesn't fit {shape} records"
            raise ValueError(msg)
        frames = (stride - STAMP.size) // frame
        if block_ms <= 0 or (frames * 1000) % block_ms:
            msg = f"{frames} frames per record don't make a whole rate at {block_ms} ms blocks"
            raise ValueError(msg)
        return cls(kind, channels, frames * 1000 // block_ms, frames)

    def decode(self, record: bytes, channel: int = 0) -> tuple[int, int, np.ndarray]:
        """A record's capture time (host monotonic ns), first sample index, and `channel` as float32 in [-1, 1]."""
        if len(record) != self.stride:
            msg = f"a record of {len(record)} bytes, not {self.stride}"
            raise ValueError(msg)
        if not 0 <= channel < self.channels:
            msg = f"channel {channel} of {self.channels}"
            raise ValueError(msg)
        host_ns, first = STAMP.unpack_from(record)
        body = memoryview(record)[STAMP.size :]
        if self.kind == "S16LE":
            x = np.frombuffer(body, "<i2").reshape(-1, self.channels)[:, channel].astype(np.float32) / 32768.0
        elif self.kind == "S32LE":
            x = (np.frombuffer(body, "<i4").reshape(-1, self.channels)[:, channel] / 2147483648.0).astype(np.float32)
        elif self.kind == "F32LE":
            x = np.frombuffer(body, "<f4").reshape(-1, self.channels)[:, channel].copy()
        else:  # S24LE3: three bytes per sample, sign-extended through the top byte
            b = np.frombuffer(body, np.uint8).reshape(-1, self.channels, 3)[:, channel, :].astype(np.int32)
            v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
            x = (np.where(v >= 1 << 23, v - (1 << 24), v) / 8388608.0).astype(np.float32)
        return host_ns, first, x


@dataclass
class Continuity:
    """Checks each record follows on from the last: its first sample is the last's plus one block, and its stamp is
    about one block period later. `check` returns None when it does, else why not (the reader treats it as a gap)."""

    fmt: Format
    # How far a stamp may sit from where the last one and the sample count put it. Capture fits the clock
    # continuously; a few milliseconds covers its corrections, while a torn or foreign stamp is far off.
    tolerance_ns: int = 5_000_000
    _last: tuple[int, int] | None = None

    def check(self, host_ns: int, first: int, lost: int = 0) -> str | None:
        last, self._last = self._last, (host_ns, first)
        if last is None:
            return None
        if lost:
            return f"{lost} records lost"
        last_ns, last_first = last
        if first != last_first + self.fmt.frames:
            return f"sample {first} after {last_first} (a block is {self.fmt.frames})"
        expected = last_ns + self.fmt.frames * 1_000_000_000 // self.fmt.rate_hz
        if abs(host_ns - expected) > self.tolerance_ns:
            return f"a stamp {(host_ns - expected) / 1e6:+.1f} ms off the block clock"
        return None


class Resampler:
    """Native rate to 16 kHz, stateful, so the filter and the phase carry across blocks.

    A low-pass filter (CUTOFF_HZ, an FIR run with its state kept between blocks) comes first, then v1's linear
    interpolation (`_resample_stream`). v1 never needed the filter: PipeWire handed it 16 kHz already. Capture on
    Continuum records the microphone's native rate, and without the filter everything between 8 and 24 kHz of a
    48 kHz stream folds back into the speech band: the Wave 3's hiss (45% of its energy) did, so the
    speech gate heard noise and ASR heard no words."""

    TAPS = 161
    CUTOFF_HZ = 7_000.0  # under 16 kHz's Nyquist (8 kHz), with room for the filter's transition

    def __init__(self, rate_hz: int, to_hz: int = SR) -> None:
        from scipy import signal

        self.ratio = rate_hz / to_hz
        self.phase = 0.0
        self.last = 0.0
        self.taps = signal.firwin(self.TAPS, min(self.CUTOFF_HZ, 0.45 * to_hz), fs=rate_hz) if rate_hz > to_hz else None
        self.zi = np.zeros(self.TAPS - 1) if self.taps is not None else None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.ratio == 1.0:
            return x
        if self.taps is not None:
            from scipy import signal

            x, self.zi = signal.lfilter(self.taps, 1.0, x, zi=self.zi)
            x = np.asarray(x, np.float32)
        xs = np.concatenate([[self.last], x])
        pos = np.arange(self.phase, len(x), self.ratio)
        y = np.interp(pos + 1, np.arange(len(xs)), xs).astype(np.float32)
        self.phase = (pos[-1] + self.ratio - len(x)) if len(pos) else self.phase - len(x)
        self.last = float(x[-1])
        return y

    def reset(self) -> None:
        """Forget the last block (after a gap: don't interpolate or filter across it)."""
        self.phase, self.last = 0.0, 0.0
        if self.zi is not None:
            self.zi = np.zeros(self.TAPS - 1)


class HostClock:
    """Host monotonic nanoseconds (what capture stamps records with: `CLOCK_MONOTONIC` on Linux, mach time on macOS,
    the same clocks as Python's `time.monotonic_ns`) to UTC nanoseconds. The offset is measured again every
    `refresh_s`, so a stepped wall clock is followed."""

    def __init__(self, refresh_s: float = 60.0) -> None:
        self.refresh_ns = int(refresh_s * 1e9)
        self._measured_at = -self.refresh_ns
        self._offset = 0

    def utc_ns(self, host_ns: int) -> int:
        now = time.monotonic_ns()
        if now - self._measured_at >= self.refresh_ns:
            # The narrowest of three brackets around a wall reading.
            best = None
            for _ in range(3):
                a = time.monotonic_ns()
                wall = time.time_ns()
                b = time.monotonic_ns()
                if best is None or b - a < best[0]:
                    best = (b - a, wall - (a + b) // 2)
            assert best is not None
            self._offset, self._measured_at = best[1], now
        return host_ns + self._offset
