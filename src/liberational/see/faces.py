"""The face models and rules see-actions' head faces use (`see/head_faces.py`).

- `Detector`: finds the faces in a BGR image, each with its box, confidence, five landmarks, and unit embedding.
  `InsightFaceDetector` is SCRFD detection and AdaFace recognition, run on a head crop.
- `Matcher`: names an embedding. `GalleryMatcher` reads `faces/db.npz` and the unknown clusters (`vision/faces.py`): the
  nearest enrolled face over 0.45 is a match, and an unnamed face joins or starts an unknown cluster ("Face N").
- `readable`: whether a look reads as a face well enough to learn from, name, or show as a photo.
- `VisitBuilder` is the visit rule: one person's matches on one camera, closed once 10 s pass
  with no match.

The whole-frame face worker and its IoU face tracker are gone: see-actions' pose tracks follow people now.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

UNKNOWN_EVERY_S = 3.0  # how often an unnamed track's cluster is looked up (a lookup writes a file)
VISIT_GAP_S = 10.0  # a visit ends when its person isn't matched on its camera for this long

NS = 1_000_000_000

# A look that hardly reads as a face is set aside: a head turned away (rear three-quarter) or a blur, which
# the detector can still box. On the development install (6,116 looks with signals), AdaFace's raw
# feature norm put one household member turned away at 6.5 to 8.6 and clear faces at 19 to 21; under 10 set aside
# 0.7% of the owner's looks at the desk, 6% of that member's (a camera that sees them almost only in profile, which
# stays), and 35% of the unnamed ones, the turned-away looks that started new face groups. The pose's nose
# confidence under 0.4 catches the blurs. Landmark frontalness can't be the gate: it's 0 for every profile.
NORM_MIN = 10.0
NOSE_MIN = 0.4


def readable(face_norm: float | None, nose: float) -> bool:
    """Whether a look reads as a face well enough to learn from or name (see NORM_MIN). A recognizer that gives no
    norm can't say otherwise, so its looks pass."""
    return nose >= NOSE_MIN and (face_norm is None or face_norm >= NORM_MIN)


@dataclass(frozen=True)
class Face:
    """One face a detector found: its box in frame pixels (x1, y1, x2, y2), confidence, the five landmarks (eyes,
    nose, mouth corners, as (x, y)), its unit embedding, and the embedding's raw norm before it was made a unit
    vector (AdaFace's quality signal; None when the recognizer doesn't give one)."""

    bbox: tuple[float, float, float, float]
    det: float
    kps: np.ndarray | None
    emb: np.ndarray
    norm: float | None = None


class Detector(Protocol):
    """Finds the faces in a BGR frame."""

    def detect(self, frame: np.ndarray) -> list[Face]: ...


class Matcher(Protocol):
    """Names an embedding: the nearest enrolled person and its similarity (None under the threshold), and for an
    unnamed one, its unknown cluster's label."""

    def match(self, emb: np.ndarray) -> tuple[str | None, float]: ...

    def unknown(self, emb: np.ndarray) -> str: ...


class InsightFaceDetector:
    """SCRFD detection and AdaFace recognition on ONNX Runtime, on CoreML, CUDA, or the CPU, whichever is there."""

    def detect(self, frame: np.ndarray) -> list[Face]:
        from liberational.vision import faces

        return [
            Face(
                bbox=(float(f.bbox[0]), float(f.bbox[1]), float(f.bbox[2]), float(f.bbox[3])),
                det=float(f.det_score),
                kps=None if f.kps is None else np.asarray(f.kps, np.float64),
                emb=np.asarray(f.normed_embedding, np.float32),
                norm=None if f.get("adaface_norm") is None else float(f["adaface_norm"]),
            )
            for f in faces.app().get(frame)
        ]


class GalleryMatcher:
    """The enrolled names: `faces/db.npz` (reread when it changes) and the unknown clusters, through `vision/faces.py`."""

    def match(self, emb: np.ndarray) -> tuple[str | None, float]:
        from liberational.vision import faces

        return faces.match(emb)

    def unknown(self, emb: np.ndarray) -> str:
        from liberational.vision import faces

        return f"Face {faces.group_unknown(emb)}"

    def learn(self, looks: Sequence[np.ndarray], names: Sequence[str | None], source: str) -> int:
        """Teach the named gallery from a track that ended (`faces.learn_from_track`). Rows added."""
        from liberational.vision import faces

        return faces.learn_from_track(looks, names, source)


VISIT_SHOTS = 3  # the best faces a visit keeps as photos, so hiding one leaves the next


@dataclass
class _Open:
    start: float
    end: float
    best_sim: float
    samples: int
    shots: list[dict[str, Any]] = field(default_factory=list)


class VisitBuilder:
    """`face.visit` records from `face.match` records: one named person's matches on one
    camera, joined while each follows the last within `gap_s`. A visit is put out once it's closed: a later match comes
    after the gap, or `close_before` passes its end plus the gap. Its `shots` are the VISIT_SHOTS best-quality faces
    fed with one (recorded detections: {detection, ts, bbox, quality, sim}), best first: the photos People picks from."""

    def __init__(self, gap_s: float = VISIT_GAP_S) -> None:
        self.gap_s = gap_s
        self.open: dict[tuple[str, str], _Open] = {}

    def feed(self, match: dict[str, Any], shot: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """A match (by time, per camera), with its face as a `shot` when its detection is recorded; returns any visit it
        closed."""
        name, ts = match.get("name"), float(match["ts_start"])
        if not name:
            return []
        key, sim = (name, match["cam"]), float(match.get("sim") or 0.0)
        v = self.open.get(key)
        if v is not None and ts - v.end <= self.gap_s:
            v.end, v.best_sim, v.samples = max(v.end, ts), max(v.best_sim, sim), v.samples + 1
            if shot is not None:
                v.shots = sorted([*v.shots, shot], key=lambda s: -s["quality"])[:VISIT_SHOTS]
            return []
        closed = [self._record(key, v)] if v is not None else []
        self.open[key] = _Open(ts, ts, sim, 1, [shot] if shot is not None else [])
        return closed

    def close_before(self, now: float) -> list[dict[str, Any]]:
        """Closes every visit whose last match is more than the gap before `now`."""
        done = [k for k, v in self.open.items() if now - v.end > self.gap_s]
        return [self._record(k, self.open.pop(k)) for k in done]

    @staticmethod
    def _record(key: tuple[str, str], v: _Open) -> dict[str, Any]:
        name, cam = key
        return {
            "ts_start": v.start,
            "ts_end": v.end,
            "name": name,
            "cam": cam,
            "best_sim": round(v.best_sim, 4),
            "samples": v.samples,
            **({"shots": v.shots} if v.shots else {}),
        }
