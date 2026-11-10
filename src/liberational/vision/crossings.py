"""Tracking and counting-line geometry.

A camera's counting line (`[crossings.lines]`, else the vertical centre line) gives `crossings_of` the times a
track's feet got past it, after hysteresis and enough track evidence. A crossing is not a count of distinct people.
Walk-bys are qualified by see/walk_bys.py. see-actions owns detector inference and uses this module's geometry for
its tracks.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from liberational import config as cfg

COCO_PERSON = 0
COCO_ANIMALS: dict[int, str] = {
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
}
DEFAULT_LINE = dict(kind="vertical", position=0.5)
# Expire before association: an old nearby box must not resurrect an ended track or a cached crossing.
# 4: a same-direction window no longer merges two tracks that were on screen together.
# 5: a restart inside max_gap is not a second person (#1239).


# ------------------------------------------------------------------ configuration


def settings() -> dict[str, Any]:
    """Track evidence and counting-line settings used by movement and animal consumers."""

    def g(key: str, default: Any) -> Any:
        return cfg.get("crossings." + key, default)

    return dict(
        min_track_frames=max(2, int(g("min_track_frames", 3))),
        min_track_seconds=max(0.0, float(g("min_track_seconds", 0.6))),
        hysteresis=min(0.2, max(0.0, float(g("hysteresis", 0.03)))),
        max_gap_seconds=max(0.2, float(g("max_gap_seconds", 1.5))),
        lines=dict(g("lines", {}) or {}),
    )


def line_for(camera: str, conf: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], bool]:
    """The configured line for a camera, else the default vertical centre line. Returns (line, is_default).

    A line is {"kind": "vertical", "position": 0..1} (x as a share of the frame width), {"kind": "horizontal",
    "position": 0..1} (y as a share of the frame height) or {"kind": "segment", "p1": [x, y], "p2": [x, y]} in 0..1
    frame coordinates. [crossings.lines.cam1] in pack.toml sets one per camera.
    """
    conf = conf or settings()
    raw = (conf.get("lines") or {}).get(camera)
    if isinstance(raw, dict):
        try:
            return normalize_line(raw), False
        except ValueError, KeyError, TypeError:
            pass
    return dict(DEFAULT_LINE), True


def normalize_line(raw: Mapping[str, Any]) -> dict[str, Any]:
    kind = raw.get("kind", "vertical")
    if kind in ("vertical", "horizontal"):
        position = float(raw["position"])
        if not 0.0 < position < 1.0:
            raise ValueError("line position must be inside the frame")
        return dict(kind=kind, position=position)
    if kind == "segment":
        p1, p2 = [float(v) for v in raw["p1"]], [float(v) for v in raw["p2"]]
        if len(p1) != 2 or len(p2) != 2 or any(not 0.0 <= v <= 1.0 for v in p1 + p2) or p1 == p2:
            raise ValueError("segment endpoints must be distinct points inside the frame")
        return dict(kind="segment", p1=p1, p2=p2)
    raise ValueError("unknown line kind")


def side_value(line: Mapping[str, Any], x: float, y: float) -> float:
    """Signed distance of a normalized point from the line (positive on one side, negative on the other).

    Vertical: x - position (positive = right). Horizontal: y - position (positive = below). Segment: cross product
    with the p1->p2 direction over its length.
    """
    if line["kind"] == "vertical":
        return x - line["position"]
    if line["kind"] == "horizontal":
        return y - line["position"]
    (x1, y1), (x2, y2) = line["p1"], line["p2"]
    length = math.hypot(x2 - x1, y2 - y1)
    return ((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)) / length


def direction_names(line: Mapping[str, Any]) -> tuple[str, str]:
    """(name for negative->positive, name for positive->negative), in image terms."""
    if line["kind"] == "vertical":
        return "left_to_right", "right_to_left"
    if line["kind"] == "horizontal":
        return "top_to_bottom", "bottom_to_top"
    return "side_a_to_side_b", "side_b_to_side_a"


def line_description(line: Mapping[str, Any], is_default: bool) -> str:
    if line["kind"] == "vertical":
        base = f"vertical line at {line['position'] * 100:.0f}% of the frame width"
    elif line["kind"] == "horizontal":
        base = f"horizontal line at {line['position'] * 100:.0f}% of the frame height"
    else:
        base = "configured line segment"
    prefix = "default " if is_default else "configured "
    suffix = " (no scene calibration)" if is_default else ""
    return prefix + base + suffix + ", both directions counted"


# RF-DETR's pretrained COCO head retains sparse category IDs; callers keep the existing internal IDs.
ANIMAL_SCORE = 0.35  # Animal candidates retain their own floor independently of the person calibration.
RF_CLASS_IDS: dict[int, int] = {COCO_PERSON: 1, **{class_id: class_id + 2 for class_id in COCO_ANIMALS}}
# Screens a person can be shown on (#2468): COCO's tv and laptop, by internal ID (sparse ID in the RF-DETR head). Kept
# apart from RF_CLASS_IDS, whose classes other than person are animals. A phone is left out: one held to a real face
# would take the face for a screen's.
COCO_SCREENS: dict[int, str] = {62: "tv", 63: "laptop"}
RF_SCREEN_IDS: dict[int, int] = {62: 72, 63: 73}
SCREEN_SCORE = 0.4  # a screen candidate at least this sure is a screen


def detector_input(frame: np.ndarray, size: int) -> NDArray[np.float32]:
    """BGR pixels to RF-DETR's RGB float input: direct bilinear resize, then ImageNet normalization.

    Resize float pixels rather than uint8 so OpenCV's half-pixel bilinear math does not round interpolated values.
    This matches the export's tensor resize with antialias=False; there is no letterbox or padding. The live GPU
    path (see/gpu_pose.py) is checked against this.
    """
    import cv2

    rgb = frame[:, :, ::-1].astype(np.float32) / 255.0
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    return np.ascontiguousarray(((resized - mean) / std).transpose(2, 0, 1)[None])


# ------------------------------------------------------------------ tracking


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def anchor(box: Sequence[float]) -> tuple[float, float]:
    """The foot point, bottom centre: it moves across the floor even when the torso is partly hidden."""
    return ((box[0] + box[2]) / 2, box[3])


def _appearance(value: Sequence[float] | NDArray[np.float64] | None) -> NDArray[np.float64] | None:
    """Missing or unusable descriptors cannot supply identity evidence."""
    if value is None:
        return None
    try:
        vector = np.asarray(value, dtype=np.float64)
    except TypeError, ValueError:
        return None
    if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
        return None
    norm = float(np.linalg.norm(vector))
    return vector / norm if math.isfinite(norm) and norm > 0 else None


class Track:
    def __init__(
        self, tid: int, box: Sequence[float], ts: float, appearance: NDArray[np.float64] | None = None
    ) -> None:
        self.id = tid
        self.box: tuple[float, ...] = tuple(box[:4])
        self.velocity = np.zeros(4)
        self.appearance = appearance
        self.first_ts = self.last_ts = ts
        self.hits = 1
        self.points = [(ts, anchor(box))]
        self.tentative = False  # started from a low-score detection, and no confident one has matched it yet

    def predicted(self, ts: float) -> tuple[float, ...]:
        dt = ts - self.last_ts
        return tuple(float(self.box[i] + self.velocity[i] * dt) for i in range(4))

    def update(self, box: Sequence[float], ts: float, appearance: NDArray[np.float64] | None = None) -> None:
        dt = max(ts - self.last_ts, 1e-6)
        self.velocity = 0.6 * self.velocity + 0.4 * (np.array(box[:4]) - np.array(self.box)) / dt
        self.box, self.last_ts = tuple(box[:4]), ts
        self.hits += 1
        self.points.append((ts, anchor(box)))
        if appearance is not None:
            if self.appearance is None:
                self.appearance = appearance
            elif self.appearance.shape == appearance.shape and float(self.appearance @ appearance) >= 0.65:
                # Avoid replacing a person's evidence with an occluder's crop. Low-score rescues never
                # provide a descriptor here, and even confident matches update slowly.
                self.appearance = _appearance(0.9 * self.appearance + 0.1 * appearance)


class Tracker:
    """Multi-object tracker in the ByteTrack manner: confident detections are matched to constant-velocity
    predictions first (IoU, with a centre-distance fallback for fast movers), the leftover tracks then get a second
    chance with low-score detections, and only confident unmatched detections start tracks. A track ends after
    max_gap seconds unseen; update() and flush() return the tracks that ended so the caller can count them.

    With `tentative`, an unmatched low-score detection starts a tentative track instead of being dropped (#1676): a
    person often scores under `high` for the first frames they're seen, so a track born on their first confident
    frame starts where they already were, and a line crossed in those frames was never seen crossed. A tentative
    track is matched like any other, so its frames are its history from its first one, but it isn't in `assigned`
    (its boxes stay unconfirmed) until a confident detection matches it, and one that ends never confirmed is
    dropped, not returned, as is one not confirmed within `tentative_s` of its start (so a faint box that never
    scores `high`, a static false box, holds back what a caller can close for at most that long).
    With optional image descriptors, motion-gated global assignment also uses normalized appearance similarity;
    it adds no model or GPU work and keeps low-confidence observations from replacing identity evidence."""

    def __init__(
        self,
        max_gap: float = 1.5,
        high: float = 0.5,
        match_iou: float = 0.2,
        max_center: float = 0.12,
        *,
        tentative: bool = False,
        tentative_s: float = 3.0,
    ) -> None:
        self.max_gap, self.high, self.match_iou, self.max_center = max_gap, high, match_iou, max_center
        self.tentative, self.tentative_s = tentative, tentative_s
        self.active: list[Track] = []
        self.assigned: dict[int, int] = {}  # the last update's detections (by index) and the track each went to
        self.next_id = 1

    def _associate(
        self,
        tracks: Sequence[Track],
        dets: Sequence[Sequence[float]],
        ts: float,
        min_iou: float,
        appearances: Sequence[NDArray[np.float64] | None] | None = None,
    ) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        pairs = []
        for ti, tr in enumerate(tracks):
            pred = tr.predicted(ts)
            pc = ((pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2)
            for di, d in enumerate(dets):
                overlap = iou(pred, d)
                dist = math.hypot(pc[0] - (d[0] + d[2]) / 2, pc[1] - (d[1] + d[3]) / 2)
                if overlap >= min_iou or dist <= self.max_center:
                    cost = -overlap + dist
                    feature = appearances[di] if appearances is not None else None
                    if feature is not None and tr.appearance is not None and feature.shape == tr.appearance.shape:
                        difference = float(np.clip(1 - tr.appearance @ feature, 0, 2))
                        if difference > 0.65:
                            continue
                        cost += 2 * difference
                    pairs.append((cost, ti, di))
        if appearances is not None:
            return self._global_pairs(pairs, len(tracks), len(dets))
        pairs.sort()
        used_t: set[int] = set()
        used_d: set[int] = set()
        out: list[tuple[int, int]] = []
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            out.append((ti, di))
        return out, used_t, used_d

    @staticmethod
    def _global_pairs(
        pairs: Sequence[tuple[float, int, int]], tracks: int, detections: int
    ) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        """One globally consistent assignment, with an unmatched option for every track."""
        if not pairs:
            return [], set(), set()
        from scipy.optimize import linear_sum_assignment

        # Dedicated unmatched columns allow uncertain candidates to lose to a new track rather
        # than forcing an identity assignment. Motion-forbidden pairs cost more than any dummy.
        costs = np.full((tracks, detections + tracks), 1.0)
        costs[:, :detections] = 4.0
        for cost, ti, di in pairs:
            costs[ti, di] = cost
        rows, columns = linear_sum_assignment(costs)
        matches = [
            (int(ti), int(di)) for ti, di in zip(rows, columns, strict=True) if di < detections and costs[ti, di] < 1
        ]
        return matches, {ti for ti, _ in matches}, {di for _, di in matches}

    def update(
        self,
        dets: Sequence[Sequence[float]],
        ts: float,
        *,
        appearances: Sequence[Sequence[float] | None] | None = None,
    ) -> list[Track]:
        """Match detections, exposing their current IDs through `assigned`, and return ended tracks.

        Optional appearance vectors align with detections and are normalized internally. Missing, malformed,
        or incompatible descriptors fall back to motion alone. Supplying descriptors enables global matching;
        legacy callers without them keep their original greedy matching. Appearance never defeats the motion
        gate, starts a low-confidence track, or extends a track beyond `max_gap`.
        """
        if appearances is not None and len(appearances) != len(dets):
            raise ValueError("appearance vectors must align with detections")
        features = [_appearance(value) for value in appearances] if appearances is not None else None
        ended: list[Track] = []
        active: list[Track] = []
        for track in self.active:
            # Float capture timestamps can round either endpoint outward. Allow only their ULPs,
            # rather than a relative tolerance that would grow the configured gap at large epochs.
            rounding = math.ulp(ts) + math.ulp(track.last_ts)
            gone = ts - track.last_ts > self.max_gap + rounding
            gone = gone or (track.tentative and ts - track.first_ts > self.tentative_s)
            if gone and not track.tentative:
                ended.append(track)
            elif not gone:
                active.append(track)
        self.active = active
        hi = [i for i, d in enumerate(dets) if d[4] >= self.high]
        lo = [i for i, d in enumerate(dets) if d[4] < self.high]
        high, low = [dets[i] for i in hi], [dets[i] for i in lo]
        high_features = [features[i] for i in hi] if features is not None else None
        low_features = [features[i] for i in lo] if features is not None else None
        self.assigned = {}
        pairs, used_t, used_h = self._associate(self.active, high, ts, self.match_iou, high_features)
        for ti, di in pairs:
            self.active[ti].update(high[di], ts, high_features[di] if high_features is not None else None)
            self.active[ti].tentative = False
            self.assigned[hi[di]] = self.active[ti].id
        rest = [t for i, t in enumerate(self.active) if i not in used_t]
        pairs2, _, used_l = self._associate(rest, low, ts, 0.4, low_features)
        for ti, di in pairs2:
            rest[ti].update(low[di], ts)
            if not rest[ti].tentative:
                self.assigned[lo[di]] = rest[ti].id
        if self.tentative:
            for di, d in enumerate(low):
                if di not in used_l:
                    track = Track(self.next_id, d, ts)
                    track.tentative = True
                    self.active.append(track)
                    self.next_id += 1
        for di, d in enumerate(high):
            if di not in used_h:
                self.active.append(Track(self.next_id, d, ts, high_features[di] if high_features is not None else None))
                self.assigned[hi[di]] = self.next_id
                self.next_id += 1
        return ended

    def flush(self) -> list[Track]:
        ended, self.active = [t for t in self.active if not t.tentative], []
        return ended


# ------------------------------------------------------------------ crossings


def crossings_of(
    track: Track, line: Mapping[str, Any], hysteresis: float, min_frames: int, min_seconds: float
) -> list[tuple[float, int]]:
    """Crossings of one track: [(ts, direction)] with direction 0 = negative->positive side, 1 = the reverse.

    Hysteresis: the anchor must get at least hysteresis (normalized distance) past the line on the new side before a
    crossing counts; jitter inside the band changes nothing. A track needs min_frames detections over at least
    min_seconds, so one-frame false positives never cross.
    """
    if track.hits < min_frames or (track.last_ts - track.first_ts) < min_seconds:
        return []
    events, state = [], 0
    for ts, (x, y) in track.points:
        value = side_value(line, x, y)
        side = 1 if value > hysteresis else -1 if value < -hysteresis else 0
        if side == 0:
            continue
        if state and side != state:
            events.append((ts, 0 if side > 0 else 1))
        state = side
    return events


def histogram(timestamps: Iterable[float], start: float, end: float, bins: int = 24) -> list[dict[str, float]]:
    """Counts per equal-width time bin over [start, end)."""
    if end <= start:
        return []
    width = (end - start) / bins
    counts = [0] * bins
    for ts in timestamps:
        if start <= ts < end:
            counts[min(bins - 1, int((ts - start) / width))] += 1
    return [dict(start=start + i * width, end=start + (i + 1) * width, count=c) for i, c in enumerate(counts)]
