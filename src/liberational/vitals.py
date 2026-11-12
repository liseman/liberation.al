"""Heart rate from the colour of skin in a close camera's picture: classical rPPG, no model, NumPy only.

Every heartbeat pushes a little more blood into the capillaries of the face, which absorbs a little more green
light (about 520 to 560 nm) and dips the reflected colour by well under one percent. Remote photoplethysmography
(rPPG) is finding that dip in the average colour of a patch of skin. This module is the whole signal path, in
plain NumPy, so it can be run on a live camera's frames and on a recorded stretch alike (`tools/measure_vitals.py`):

- `face_regions` puts three skin patches (forehead and both cheeks) where the person detector's head joints are:
  halpe26 joints 0 to 4 (nose, both eyes, both ears). The same region every frame, on a face at least
  `MIN_FACE_PX` across.
- `region_means` averages each patch's R, G, and B per frame, cancelling sensor noise over the patch's pixels.
- `Trace` keeps one tracked person's colour trace over a sliding window of about `WINDOW_S` seconds, and
  `Trace.reading` turns it into a `Reading`: POS for the rate, CHROM as the agreement check, an in-band SNR from
  the spectrum, and a confidence from those plus how much the face moved and whether the picture has any colour
  left at all (an infrared night mode's monochrome picture says nothing about blood).

The two methods are the published classical ones, implemented from their papers rather than copied from a
toolbox (rPPG-Toolbox is RAIL-licensed and pyVHR is GPL-3.0):

- POS, Plane-Orthogonal-to-Skin (Wang, den Brinker, Stuijk, and de Haan, IEEE TBME 2017): each ~1.6 s block of the
  temporally normalized RGB trace is projected onto the plane orthogonal to the `[1, 1, 1]` intensity axis
  (`S1 = G - B`, `S2 = G + B - 2R`), the two are combined by the ratio of their standard deviations (which cancels
  the intensity change a moving or dimming light makes), and the blocks overlap-add into one pulse signal.
- CHROM (de Haan and Jeanne, IEEE TBME 2013): two orthogonal chrominance signals `Xs = 3R - 2G` and
  `Ys = 1.5R + G - 1.5B`, band-passed, and combined as `Xf - (std(Xf) / std(Yf)) * Yf`.

Both are run on a window that is detrended (its own linear drift removed) and band-passed to
`BAND_HZ` (0.7 to 3 Hz, 42 to 180 beats a minute). The rate is the strongest line in that band's spectrum, with a
parabolic refinement between bins; the confidence is the product of four scores, so any one of them going bad
vetoes the reading (the multiplication is the point: a still face with no pulse in the band must not read).

`Reading.bpm` is NaN unless the reading is trusted (`confidence >= MIN_CONFIDENCE`): unknown is not zero, and a
number that isn't trustworthy is never handed out.

What this method can't do, which the confidence is there to keep quiet about: H.264 quantization (above about
cq 21), a face under about 140 px across, fewer than 20 frames a second, infrared night mode, head turns and
talking, and heavy compression's chroma blocks all destroy the signal. Room cameras at several metres and a
walking camera are out of reach entirely.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

BAND_HZ = (0.7, 3.0)  # 42 to 180 beats a minute, the band the pulse has to be in
WINDOW_S = 12.0  # how much of the trace one reading looks at: long enough for a line 5 bpm apart to stand out
MIN_FPS = 20.0  # below this the band is too close to the frame rate to trust, so nothing is read
MIN_FACE_PX = 140  # a face this many pixels across or bigger: below it, too few skin pixels to average the noise out
# The bar a reading must reach to be a number at all. Calibrated on synthetic traces (a known rate under noise and
# a slow light drift): every one whose rate came out more than 7 bpm wrong scored under 0.3, every one within
# 1 bpm of the truth scored over 0.5, and traces with no pulse at all scored under 0.5 in 79 of 80 draws.
MIN_CONFIDENCE = 0.5
MIN_COVERAGE = 0.7  # the share of the window's frames a reading must have seen
MIN_SPAN_S = 0.6 * WINDOW_S  # the window must hold at least this much time before a reading is attempted
MONO_CHROMA = 1.5  # mean |R - G| + |G - B| (0 to 255) under this: a monochrome picture (an infrared night mode)
PEAK_HZ = 0.15  # half-width of the peak in the spectrum, and of its harmonic, that counts as the pulse
FACE_SIDE = 2.3  # the face's side in interocular distances (pupils about 63 mm apart on a 145 mm wide face)
FOREHEAD_UP = 0.55  # the forehead patch sits this many interocular distances above the eyes
CHEEK_SIDE = 0.75  # the cheek patches sit this many interocular distances either side of the face's midline
CHEEK_DOWN = 0.15  # and this far below the nose
PATCH = 0.7  # a patch's side, in interocular distances
MIN_PATCH_PX = 8  # a patch clipped to less than this across is left out
MIN_JOINT_CONF = 0.3  # a head joint placed with less confidence than this isn't used
MOTION_FULL = 0.35  # face widths a second: at or above this the face is moving too much for the still score to help
SNR_LOW_DB, SNR_HIGH_DB = 8.0, 18.0  # in-band peak-to-floor SNR mapped to a score of 0 and 1
BAND_LOW_DB, BAND_HIGH_DB = -3.0, 2.5  # and how much of the band the pulse holds, mapping to a score of 0 and 1
AGREE_BPM = 8.0  # POS and CHROM this far apart disagree: the agreement score falls to 0
COVERAGE_LOW, COVERAGE_HIGH = 0.5, 0.9  # the share of frames mapped to a seen score of 0 and 1
BPM_MIN, BPM_MAX = 42.0, 180.0


@dataclass(frozen=True)
class FaceRegions:
    """Where to read a face's skin: the box to crop out of the frame, and the skin patches inside that crop."""

    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in the frame's pixels
    patches: tuple[tuple[int, int, int, int], ...]  # the patches, in the crop's own pixels
    side: float  # the face's side in the frame's pixels
    centre: tuple[float, float]  # its centre, in the frame's pixels


def face_regions(joints: NDArray[np.floating], width: int, height: int) -> FaceRegions | None:
    """The skin patches of one pose's face (K x 3 joints: x and y in the frame's pixels, confidence), or None when
    the nose or an eye isn't placed, the face is edge-on, or the patches would be too small to average noise over.

    `width` and `height` are the frame's size, to keep every patch inside it. A face further away than
    `MIN_FACE_PX` isn't refused here: `Trace` compares the median size over a window, so a person leaning back
    mid-window is a window without a reading rather than a gap in the trace."""
    head = np.asarray(joints, np.float64)[:3]  # nose and both eyes; the ears are often unplaced on a front view
    if head.shape != (3, 3) or (head[:, 2] < MIN_JOINT_CONF).any():
        return None
    (_, nose_y, _), (lx, ly, _), (rx, ry, _) = head[0], head[1], head[2]
    spread = math.hypot(rx - lx, ry - ly)  # the interocular distance
    if spread < MIN_PATCH_PX / PATCH / 2:
        return None
    mid_x, mid_y = (lx + rx) / 2, (ly + ry) / 2
    side = PATCH * spread
    half = side / 2
    wanted = [
        (mid_x, mid_y - FOREHEAD_UP * spread),  # the forehead, above the eyes and inside the hairline
        (mid_x - CHEEK_SIDE * spread, nose_y + CHEEK_DOWN * spread),  # the cheeks, beside and just under the nose
        (mid_x + CHEEK_SIDE * spread, nose_y + CHEEK_DOWN * spread),
    ]
    patches: list[tuple[int, int, int, int]] = []
    for cx, cy in wanted:
        x1, y1 = max(0, round(cx - half)), max(0, round(cy - half))
        x2, y2 = min(width, round(cx + half)), min(height, round(cy + half))
        if x2 - x1 >= MIN_PATCH_PX and y2 - y1 >= MIN_PATCH_PX:
            patches.append((x1, y1, x2, y2))
    if len(patches) < 2:  # one patch alone is a face mostly out of frame
        return None
    x1 = min(p[0] for p in patches)
    y1 = min(p[1] for p in patches)
    x2 = max(p[2] for p in patches)
    y2 = max(p[3] for p in patches)
    inside = tuple((p[0] - x1, p[1] - y1, p[2] - x1, p[3] - y1) for p in patches)
    return FaceRegions((x1, y1, x2, y2), inside, FACE_SIDE * spread, (mid_x, mid_y))


def region_means(pixels: NDArray[np.uint8], patches: Sequence[tuple[int, int, int, int]]) -> NDArray[np.float64]:
    """Each patch's mean R, G, and B (0 to 255) over `pixels` (H x W x 3, BGR as the vision backends hand it over),
    the patches given in that array's own pixels. A patch's pixels average out the sensor's noise; the patches
    average to one colour, so a little hair over a cheek costs less than its share of the forehead."""
    means = np.array([pixels[y1:y2, x1:x2].reshape(-1, 3).mean(axis=0) for x1, y1, x2, y2 in patches], dtype=np.float64)
    return means.mean(axis=0)[::-1].copy()  # BGR as the backends give it, returned as R, G, B


def detrend(trace: NDArray[np.floating]) -> NDArray[np.float64]:
    """A trace with its mean and its straight-line drift removed: a light dimming over the window isn't a pulse."""
    x = np.asarray(trace, np.float64)
    if len(x) < 2:
        return x - x.mean() if len(x) else x
    steps = np.arange(len(x), dtype=np.float64)
    slope = np.polyfit(steps, x, 1)[0]
    return x - (x.mean() + slope * (steps - steps.mean()))


def band_pass(trace: NDArray[np.floating], fps: float, band: tuple[float, float] = BAND_HZ) -> NDArray[np.float64]:
    """`trace` detrended and kept to `band` in the frequency domain (a raised-cosine taper over one bin at each
    edge, so a line just inside the band isn't rung by a hard cut). With fewer than four samples, the detrended
    trace."""
    x = detrend(trace)
    n = len(x)
    if n < 4 or fps <= 0 or n < 2:
        return x
    freqs = np.fft.rfftfreq(n, 1.0 / fps)
    spectrum = np.fft.rfft(x)
    low, high = band
    step = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    rise = np.clip((freqs - low) / step + 0.5, 0.0, 1.0)
    fall = np.clip((high - freqs) / step + 0.5, 0.0, 1.0)
    return np.fft.irfft(spectrum * np.minimum(rise, fall), n)


def spectrum(trace: NDArray[np.floating], fps: float, pad: int = 4) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The trace's (frequencies in Hz, power) with a Hann window and `pad` times zero-padding, so a 12 s window's
    lines land about 1 bpm apart instead of 5."""
    x = np.asarray(trace, np.float64)
    n = len(x)
    if n < 4 or fps <= 0:
        return np.zeros(0), np.zeros(0)
    size = int(2 ** math.ceil(math.log2(n * pad)))
    power = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(n), size)) ** 2
    return np.fft.rfftfreq(size, 1.0 / fps), power


def peak_hz(freqs: NDArray[np.float64], power: NDArray[np.float64], band: tuple[float, float] = BAND_HZ) -> float:
    """The strongest line's frequency in `band`, refined between bins by a parabola through its peak and its two
    neighbours; NaN when the band holds nothing."""
    inside = (freqs >= band[0]) & (freqs <= band[1])
    if not inside.any():
        return float("nan")
    masked = np.where(inside, power, 0.0)
    k = int(np.argmax(masked))
    if not masked[k] > 0:
        return float("nan")
    if 0 < k < len(masked) - 1:
        left, middle, right = np.log(masked[k - 1] + 1e-30), np.log(masked[k] + 1e-30), np.log(masked[k + 1] + 1e-30)
        curvature = left - 2 * middle + right
        if curvature < 0:
            offset = 0.5 * (left - right) / curvature
        else:
            offset = 0.0
        step = freqs[1] - freqs[0] if len(freqs) > 1 else 0.0
        return float(freqs[k] + np.clip(offset, -0.5, 0.5) * step)
    return float(freqs[k])


def rate(trace: NDArray[np.floating], fps: float, band: tuple[float, float] = BAND_HZ) -> float:
    """The beats a minute `trace`'s spectrum peaks at, within `band`; NaN when it has no line worth measuring."""
    freqs, power = spectrum(trace, fps)
    hz = peak_hz(freqs, power, band)
    if math.isnan(hz):
        return float("nan")
    return float(np.clip(hz * 60.0, BPM_MIN, BPM_MAX))


def _pulse_bins(freqs: NDArray[np.float64], peak: float, band: tuple[float, float]) -> NDArray[np.bool_]:
    """The spectrum's bins within `PEAK_HZ` of `peak` or of its harmonic, inside `band`: the pulse's own bins."""
    inside = (freqs >= band[0]) & (freqs <= band[1])
    pulse = (np.abs(freqs - peak) <= PEAK_HZ) | ((2 * peak <= band[1]) & (np.abs(freqs - 2 * peak) <= PEAK_HZ))
    return pulse & inside


def snr_db(trace: NDArray[np.floating], fps: float, peak: float, band: tuple[float, float] = BAND_HZ) -> float:
    """The in-band spectral SNR of the pulse, in decibels: its power (within `PEAK_HZ` of `peak` and of its
    harmonic) over the mean power of one of the band's other bins. A clean periodic line reads tens of dB; noise
    with one lucky bin reads a few, which is why `band_db` weighs in too."""
    if math.isnan(peak) or peak <= 0:
        return 0.0
    freqs, power = spectrum(trace, fps)
    if not len(freqs):
        return 0.0
    pulse = _pulse_bins(freqs, peak, band)
    floor = ~pulse & (freqs >= band[0]) & (freqs <= band[1])
    if not pulse.any() or not floor.any():
        return 0.0
    rest = float(power[floor].mean())
    if rest <= 0:
        return 0.0
    return float(max(-60.0, 10.0 * math.log10(float(power[pulse].sum()) / rest)))


def band_db(trace: NDArray[np.floating], fps: float, peak: float, band: tuple[float, float] = BAND_HZ) -> float:
    """How much of the heart band's power the pulse holds, in decibels: the pulse's bins' power over every other
    bin of the band. It falls as the band fills with noise, which is what makes it the honest second opinion on
    `snr_db`: a peak a bin or two wide stands out of a mean floor even when the band is mostly noise."""
    if math.isnan(peak) or peak <= 0:
        return 0.0
    freqs, power = spectrum(trace, fps)
    if not len(freqs):
        return 0.0
    pulse = _pulse_bins(freqs, peak, band)
    floor = ~pulse & (freqs >= band[0]) & (freqs <= band[1])
    if not pulse.any() or not floor.any():
        return -60.0
    rest = float(power[floor].sum())
    if rest <= 0:
        return 60.0
    return float(max(-60.0, min(60.0, 10.0 * math.log10(float(power[pulse].sum()) / rest))))


def pos(rgb: NDArray[np.floating], fps: float, block_s: float = 1.6) -> NDArray[np.float64]:
    """POS's pulse signal from a T x 3 trace of R, G, and B, overlap-added over ~1.6 s blocks (Wang et al., 2017)."""
    trace = np.asarray(rgb, np.float64)
    n = len(trace)
    block = max(2, min(n, int(round(block_s * fps))))
    out = np.zeros(n)
    step = max(1, block // 2)
    for start in range(0, max(1, n - block + 1), step):
        window = trace[start : start + block]
        if len(window) < 2:
            continue
        mean = window.mean(axis=0)
        if (mean <= 1e-9).any():  # a black frame says nothing
            continue
        normalized = (window / mean).T - 1.0  # temporal normalization: the light's own change divided out
        first = normalized[1] - normalized[2]
        second = normalized[1] + normalized[2] - 2.0 * normalized[0]
        spread = float(second.std())
        alpha = float(first.std()) / spread if spread > 1e-12 else 0.0
        pulse = first + alpha * second
        out[start : start + block] += pulse - pulse.mean()
    return out


def chrom(rgb: NDArray[np.floating], fps: float) -> NDArray[np.float64]:
    """CHROM's pulse signal from a T x 3 trace of R, G, and B (de Haan and Jeanne, 2013)."""
    trace = np.asarray(rgb, np.float64)
    if len(trace) < 2:
        return np.zeros(len(trace))
    xs = 3.0 * trace[:, 0] - 2.0 * trace[:, 1]
    ys = 1.5 * trace[:, 0] + trace[:, 1] - 1.5 * trace[:, 2]
    xf, yf = band_pass(xs, fps), band_pass(ys, fps)
    spread = float(yf.std())
    alpha = float(xf.std()) / spread if spread > 1e-12 else 0.0
    return xf - alpha * yf


@dataclass(frozen=True)
class Reading:
    """One window's measurement: `bpm` is NaN unless the reading is trusted, and the rest is why."""

    bpm: float
    confidence: float
    snr_db: float
    pos_bpm: float  # what POS alone said, for the record
    chrom_bpm: float  # and what CHROM said
    coverage: float  # the share of the window's frames the trace had
    fps: float  # the rate its samples came at
    face_px: int  # the face's side, median over the window
    motion: float  # how fast its centre moved, face widths a second
    mono: bool  # a picture with no colour: an infrared night mode

    @property
    def trusted(self) -> bool:
        """Whether this reading is a number that can be shown: confident, and with a rate to show."""
        return self.confidence >= MIN_CONFIDENCE and not math.isnan(self.bpm)


class Trace:
    """One tracked person's colour trace on one camera, over a sliding window, and the reading it supports."""

    def __init__(self, window_s: float = WINDOW_S, min_face_px: int = MIN_FACE_PX) -> None:
        self.window_s, self.min_face_px = window_s, min_face_px
        # (time, R, G, B, the face's side in pixels, how fast its centre moved in face widths a second)
        self._samples: deque[tuple[int, float, float, float, float, float]] = deque()

    def add(self, ts_ns: int, rgb: NDArray[np.floating], face_px: float, motion: float = 0.0) -> None:
        """One frame's sample, in time order. Samples older than the window are dropped."""
        r, g, b = (float(v) for v in rgb[:3])
        self._samples.append((int(ts_ns), r, g, b, float(face_px), float(motion)))
        oldest = int(ts_ns) - int(self.window_s * 1e9)
        while self._samples and self._samples[0][0] < oldest:
            self._samples.popleft()

    def reset(self) -> None:
        """Frames are missing or the track ended: start the trace again."""
        self._samples.clear()

    @property
    def first_ns(self) -> int:
        """The window's first sample's capture time; 0 while it holds none."""
        return self._samples[0][0] if self._samples else 0

    @property
    def last_ns(self) -> int:
        """And its last's."""
        return self._samples[-1][0] if self._samples else 0

    def __len__(self) -> int:
        return len(self._samples)

    def reading(self) -> Reading | None:
        """The window's reading, or None while it doesn't hold enough of a trace to try.

        A window whose samples are too few, too slow, too mono, or too small-faced still gets a Reading: it says
        why with a confidence of 0 and a NaN rate, so a reader can tell "not enough yet" from "measured, nothing
        trustworthy"."""
        samples = list(self._samples)
        if len(samples) < 8:
            return None
        times = np.array([s[0] for s in samples], np.int64)
        span = (times[-1] - times[0]) / 1e9
        if span < MIN_SPAN_S:
            return None
        rgb = np.array([s[1:4] for s in samples], np.float64)
        sides = np.array([s[4] for s in samples], np.float64)
        steps = np.diff(times) / 1e9
        step = float(np.median(steps[steps > 0])) if (steps > 0).any() else 0.0
        fps = 1.0 / step if step > 0 else 0.0
        coverage = float(np.clip(len(samples) / (round(span / step) + 1), 0.0, 1.0)) if step > 0 else 0.0
        face_px = int(np.median(sides))
        mono = bool(np.mean(np.abs(rgb[:, 0] - rgb[:, 1]) + np.abs(rgb[:, 1] - rgb[:, 2])) < MONO_CHROMA)
        motion = float(np.median([s[5] for s in samples])) if len(samples) > 2 else 0.0
        nothing = Reading(
            bpm=float("nan"),
            confidence=0.0,
            snr_db=0.0,
            pos_bpm=float("nan"),
            chrom_bpm=float("nan"),
            coverage=coverage,
            fps=fps,
            face_px=face_px,
            motion=motion,
            mono=mono,
        )
        if mono or fps < MIN_FPS or face_px < self.min_face_px or coverage < MIN_COVERAGE:
            return nothing
        pulse = band_pass(pos(rgb, fps), fps)
        check = chrom(rgb, fps)
        pos_bpm, chrom_bpm = rate(pulse, fps), rate(check, fps)
        if math.isnan(pos_bpm) or math.isnan(chrom_bpm):
            return nothing
        freqs, power = spectrum(pulse, fps)
        hz = peak_hz(freqs, power)
        snr = snr_db(pulse, fps, hz)
        confidence = (
            _score(snr, SNR_LOW_DB, SNR_HIGH_DB)
            * _score(band_db(pulse, fps, hz), BAND_LOW_DB, BAND_HIGH_DB)
            * _score(abs(pos_bpm - chrom_bpm), AGREE_BPM, 0.0)
            * _score(motion, MOTION_FULL, 0.0)
            * _score(coverage, COVERAGE_LOW, COVERAGE_HIGH)
        )
        trusted = confidence >= MIN_CONFIDENCE
        return Reading(
            bpm=pos_bpm if trusted else float("nan"),
            confidence=confidence,
            snr_db=snr,
            pos_bpm=pos_bpm,
            chrom_bpm=chrom_bpm,
            coverage=coverage,
            fps=fps,
            face_px=face_px,
            motion=motion,
            mono=mono,
        )


def _score(value: float, low: float, high: float) -> float:
    """`value` mapped to 0 at `low` and 1 at `high`, linearly, held at both ends (either way round)."""
    if high == low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))
