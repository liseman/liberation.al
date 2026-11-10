"""How good a face shot is, for picking a person's main photo and the shots on their contact card.

Each sighting gets five factors, each from 0 to 1:

  det      detector confidence: 0 at MIN_DET (0.5, the lowest the backfill keeps), 1 at 0.9 and above
  size     the face's shorter side in video pixels: 0 at 24 px (a face that small is a blur), 1 at 112 px and above
  frontal  how squarely the face looks at the camera, from the detector's five landmarks (frontal() below)
  sharp    Laplacian variance of the face resized to 64 x 64 grey (sharpness() below): v / (v + 150), so 150 reads
           as 0.5. Size is normalized first, so a big face isn't sharp just for having more edges
  sim      similarity to the person's face rows: 0 at the 0.45 match threshold, 1 at 0.75 and above

The score is their weighted geometric mean, so one bad factor sinks a shot however good the rest are: a sharp,
confident profile view still loses to a slightly softer face looking at the camera. Weights (they sum to 1):
frontal 0.3, size 0.25, sharp 0.2, det 0.1, sim 0.15. Frontalness and size carry the most because they are what makes a
contact photo recognizable; confidence and similarity are already high for any named sighting.

A sighting whose frontal or sharp hasn't been measured yet (until the backfill's catch-up reaches
it) counts each unmeasured factor as 0.5 (UNMEASURED): a measured face looking at the camera beats it, and it beats
a measured profile. The catch-up measures each person's best-ranked unmeasured shots first, so the ones that could
be chosen are measured before they are shown.

The score is stored with each sighting (the quality column, vision/sightings.py) and kept current wherever its
inputs change: when the backfill stores or measures a sighting, and when a rematch changes its similarity.
"""

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

MIN_DET = 0.5  # the backfill's sightings and the face photos use the same floor
WEIGHTS = {"frontal": 0.3, "size": 0.25, "sharp": 0.2, "det": 0.1, "sim": 0.15}
SHARP_HALF = 150.0
SHARP_PX = 64
UNMEASURED = 0.5  # the factor an unmeasured frontalness or sharpness counts as


def _ramp(x: float, lo: float, hi: float) -> float:
    """0 at or below lo, 1 at or above hi, linear between. NaN reads as 0."""
    if not math.isfinite(x):
        return 0.0
    return min(1.0, max(0.0, (x - lo) / (hi - lo)))


def frontal(kps: ArrayLike | None) -> float:
    """1 for a face looking straight at the camera, falling to 0 for a profile or a face tipped far up or down.

    kps are the detector's five landmarks as (x, y): left eye, right eye, nose, left mouth corner, right mouth corner
    (InsightFace's order, "left" as seen in the image). Turning the head (yaw) slides the nose sideways from the
    middle of the eyes; at half the eye distance the face is in profile. Tipping it (pitch) moves the nose up or
    down between the eyes and the mouth; straight on it sits a little over halfway down. Eyes that are swapped or on
    top of each other mean the landmarks are wrong, and the shot scores 0."""
    if kps is None:
        return 0.0
    p = np.asarray(kps, np.float64).reshape(-1, 2)
    if p.shape != (5, 2) or not np.isfinite(p).all():
        return 0.0
    le, re, nose, lm, rm = p
    eye_dx = re[0] - le[0]
    eye_mid, mouth_mid = (le + re) / 2, (lm + rm) / 2
    drop = mouth_mid[1] - eye_mid[1]
    if eye_dx <= 1e-6 or drop <= 1e-6:
        return 0.0
    yaw = abs(nose[0] - eye_mid[0]) / eye_dx  # 0 straight on, about 0.5 in profile
    pitch = abs((nose[1] - eye_mid[1]) / drop - 0.55)  # 0 straight on
    return (1 - _ramp(yaw, 0.05, 0.5)) * (1 - _ramp(pitch, 0.1, 0.45))


def sharpness(grey: ArrayLike) -> float:
    """Laplacian variance of a grey face crop resized to SHARP_PX square: higher is sharper (raw, not 0 to 1)."""
    import cv2

    a = np.asarray(grey)
    if a.ndim == 3:
        a = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    if a.size == 0 or min(a.shape[:2]) < 2:
        return 0.0
    small = cv2.resize(a.astype(np.uint8), (SHARP_PX, SHARP_PX), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(small, cv2.CV_64F).var())


def crop_sharpness(frame: np.ndarray, box: Sequence[float]) -> float:
    """sharpness() of the box (x1, y1, x2, y2 in frame pixels) cut out of a BGR frame, clipped to the frame."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (round(float(v)) for v in box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return 0.0
    return sharpness(frame[y1:y2, x1:x2])


def factors(
    det: float, box: Sequence[float], sim: float | None, frontal: float | None, sharp: float | None
) -> dict[str, float | None]:
    """Each factor from 0 to 1 (None for one not measured yet). box is x1, y1, x2, y2 in video pixels."""
    x1, y1, x2, y2 = (float(v) for v in box)
    return {
        "det": _ramp(float(det), MIN_DET, 0.9),
        "size": _ramp(min(x2 - x1, y2 - y1), 24, 112),
        "sim": _ramp(float(sim if sim is not None else 0), 0.45, 0.75),
        "frontal": None if frontal is None else min(1.0, max(0.0, float(frontal))),
        "sharp": None if sharp is None else max(0.0, float(sharp)) / (max(0.0, float(sharp)) + SHARP_HALF),
    }


def score(
    det: float | None, box: Sequence[float], sim: float | None, frontal: float | None = None, sharp: float | None = None
) -> float:
    """The shot's quality from 0 to 1: the weighted geometric mean of its factors. A detection below MIN_DET scores
    0, so it is never chosen. A factor not measured yet counts as UNMEASURED."""
    if det is None or not math.isfinite(float(det)) or float(det) < MIN_DET or any(v is None for v in box):
        return 0.0
    f = {k: UNMEASURED if v is None else v for k, v in factors(det, box, sim, frontal, sharp).items()}
    if any(v <= 0 for v in f.values()):
        return 0.0
    return math.exp(sum(WEIGHTS[k] * math.log(v) for k, v in f.items()))


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two boxes (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    w, h = min(ax2, bx2) - max(ax1, bx1), min(ay2, by2) - max(ay1, by1)
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0
