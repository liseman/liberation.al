"""Loudness for playback (#336): bring quiet room audio up to a normal listening level, with a limiter so a loud
transient can't clip. Only what's served is changed; the archive stays as recorded.

Some microphones record speech very quietly (devbox's Wave 3: speech peaks near -55 dBFS, chunks average -45 dB),
so Review and the line clips played near silence. A single peak normalization doesn't help, since one door slam at
-2 dBFS sets the peak. Instead:

1. The speech level is the median power of the 20 ms frames that stand clearly above the room's floor (ACTIVE_DB
   over its 10th percentile frame). The median, not the mean, so a loud stretch (a door slam, a vacuum cleaner) under
   half of the sound doesn't set it. With no such frames (steady sound, no pauses) it's the median of all frames.
2. The whole clip gets one gain that brings that level to TARGET_DBFS, at most MAX_GAIN_DB, and never below 1 (a clip
   that's already loud enough is left alone). One gain, not a moving one, so the room's quiet and loud parts keep
   their relation.
3. A limiter then turns down only the frames whose peak would pass CEILING_DBFS, smoothed across neighbors so it
   doesn't click, then a final clip catches anything left.

Pure numpy, no ffmpeg, so it runs the same on any machine and keeps every sample in its place (Review lines audio up
with video by sample count).
"""

import numpy as np

TARGET_DBFS = -20.0  # speech RMS after the gain: a comfortable level on laptop speakers
MAX_GAIN_DB = 40.0  # enough for -60 dBFS speech; more mostly raises hiss
CEILING_DBFS = -1.0  # no sample goes past this
FRAME_S = 0.02
ACTIVE_DB = 12.0  # a frame this far over the floor counts as speech (or other sound worth hearing)
MIN_ACTIVE = 0.02  # fewer active frames than this share, and the level comes from all frames
SILENT_RMS = 10 ** (-90 / 20)  # quieter than this is digital silence: nothing to bring up
LIMIT_SPREAD = 2  # frames on each side a limiter cut reaches, so it eases in and out (about 40 ms)


def _db(x: float) -> float:
    return float(10 ** (x / 20))


def gain_for(data: np.ndarray, sr: int) -> float:
    """The one gain normalize applies before limiting, for int16 or float samples in [-1, 1)."""
    x = _as_float(data)
    frame = max(1, int(round(sr * FRAME_S)))
    n = len(x) // frame
    if n == 0:
        return 1.0
    power = np.mean(np.square(x[: n * frame].reshape(n, frame)), axis=1, dtype=np.float64)
    active = power[power > np.percentile(power, 10) * _db(ACTIVE_DB) ** 2]
    pool = active if len(active) >= max(1, MIN_ACTIVE * n) else power
    level = float(np.sqrt(np.median(pool)))
    if level < SILENT_RMS:
        return 1.0
    return float(min(_db(MAX_GAIN_DB), max(1.0, _db(TARGET_DBFS) / level)))


def normalize(data: np.ndarray, sr: int) -> np.ndarray:
    """int16 mono samples brought up to a normal speech level and limited below CEILING_DBFS. Same length; a
    sample's sign never changes; nothing gets quieter except where the limiter has to."""
    if data.size == 0:
        return data.astype(np.int16, copy=True)
    x = _as_float(data)
    x *= np.float32(gain_for(data, sr))
    ceiling = _db(CEILING_DBFS)
    frame = max(1, int(round(sr * FRAME_S)))
    n = -(-len(x) // frame)  # frames, the last one maybe short
    padded = np.zeros(n * frame, dtype=np.float32)
    padded[: len(x)] = np.abs(x)
    peaks = padded.reshape(n, frame).max(axis=1)
    cut = np.minimum(1.0, ceiling / np.maximum(peaks, 1e-12))
    if (cut < 1).any():
        # Each frame takes the deepest cut within LIMIT_SPREAD frames, so the turn-down starts before a transient
        # and lasts past it, then the cut moves smoothly from frame to frame.
        wide = np.pad(cut, LIMIT_SPREAD, constant_values=1.0)
        cut = np.min(np.lib.stride_tricks.sliding_window_view(wide, 2 * LIMIT_SPREAD + 1), axis=1)
        centers = (np.arange(n) + 0.5) * frame
        x *= np.interp(np.arange(len(x), dtype=np.float32), centers, cut).astype(np.float32)
    x = np.clip(x, -ceiling, ceiling)
    return np.round(x * 32768).clip(-32768, 32767).astype(np.int16)


def _as_float(data: np.ndarray) -> np.ndarray:
    """float32, to keep a 330 s clip at 48 kHz to tens of MB rather than hundreds."""
    if data.dtype == np.int16:
        return data.astype(np.float32) / np.float32(32768)
    return data.astype(np.float32)
