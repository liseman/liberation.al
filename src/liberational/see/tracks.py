"""Bounded, source-backed movement fragments from camera-local body tracks (#176).

This is pure geometry and evidence bookkeeping, not identity recognition. Foot anchors match crossings.anchor.
Every evaluated frame used for a visible interval, a gap, or an edge boundary remains a causal parent, including
empty frames and motion evidence. Independent fragments overlap only by two source-backed samples. Consumers
union supported intervals; they must not turn the entire header span into dwell time or join across gaps.

Entry and exit are conservative image-edge candidates, never physical doorway claims. Missing motion, missing
track IDs, sampling gaps, and shutdown censor boundaries. A short occlusion can retain the input track ID while
its unobserved duration remains a gap. No wall clock, model, broker, or runtime state is accessed here.
"""

import copy
import math
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Literal

from liberational.vision.crossings import anchor

VERSION = "movement-1"
DEFAULTS: dict[str, float | int] = {
    "sample_gap": 1.0,
    "max_gap": 1.5,
    "motion_gap": 1.0,
    "motion_shift": 0.03,
    "edge_margin": 0.08,
    "min_displacement": 0.03,
    "edge_min_s": 0.2,
    "max_duration_s": 10.0,
    "min_samples": 3,
    "max_points": 32,
    "max_parents": 64,
    "max_tracks": 64,
    "max_motion": 32,
}


@dataclass(frozen=True)
class Output:
    """A derived record and all source payloads supporting it, ready for acknowledged ingestion."""

    kind: Literal["track", "coverage"]
    record: dict[str, Any]
    parents: tuple[str, ...]
    # The body and motion stretches a track was made from, ("body" | "motion", first header ns, last header ns), which
    # its worker can publish as span dependencies in place of the per-frame parents (#1611). None when they aren't all
    # known (state saved before records kept their header ns, #1660).
    spans: tuple[tuple[str, int, int], ...] | None = None


def _finite(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _transform(value: object) -> list[float] | None:
    """Validate MotionPipeline's canonical nested 3x3 affine matrix; accept legacy flat six/nine values too."""
    if not isinstance(value, list):
        return None
    if len(value) == 3 and all(isinstance(row, list) and len(row) == 3 for row in value):
        flat = [item for row in value for item in row]
    elif len(value) in {6, 9}:
        flat = value
    else:
        return None
    if not all(_finite(item) for item in flat):
        return None
    result = [float(item) for item in flat]
    if len(result) == 9 and any(abs(result[i] - ideal) > 1e-6 for i, ideal in ((6, 0), (7, 0), (8, 1))):
        return None  # Perspective changes cannot be qualified by this affine movement reader.
    return result if len(result) == 9 else [*result, 0.0, 0.0, 1.0]


def _compose(a: list[float], b: list[float]) -> list[float]:
    return [sum(a[row * 3 + k] * b[k * 3 + col] for k in range(3)) for row in range(3) for col in range(3)]


def _camera_bounds(evidence: list[dict[str, Any]]) -> tuple[float, float] | None:
    """Conservative displacement range across every enclosing measured step, including inverse pans.

    Body and motion timestamps need not match. Extra bracketing steps can only increase this range. A step
    shared by adjacent body brackets is composed once, while every use still retains its source parent.
    """
    if any(e["motion"] == "unknown" for e in evidence):
        return None
    transform = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    corners = ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0))
    extrema = [[x, x, y, y] for x, y in corners]
    seen: set[str] = set()
    for evaluation in evidence:
        for ident, step in zip(evaluation["step_ids"], evaluation["steps"], strict=True):
            if ident in seen:
                continue
            seen.add(ident)
            transform = _compose(step, transform)
            for (x, y), bounds in zip(corners, extrema, strict=True):
                px = transform[0] * x + transform[1] * y + transform[2]
                py = transform[3] * x + transform[4] * y + transform[5]
                bounds[0], bounds[1] = min(bounds[0], px), max(bounds[1], px)
                bounds[2], bounds[3] = min(bounds[2], py), max(bounds[3], py)
    return max(b[1] - b[0] for b in extrema), max(b[3] - b[2] for b in extrema)


def _ns(rec: dict[str, Any]) -> int | None:
    """A record's header start ns, when the reader kept it (vision_log keep_ns)."""
    ns = rec.get("_start_ns")
    return ns if type(ns) is int else None


def _parents(evidence: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(pid for e in evidence for pid in [e["id"], *e["motion_ids"]]))


def _spans(evidence: list[dict[str, Any]]) -> tuple[tuple[str, int, int], ...] | None:
    """The body and motion stretches behind `evidence` (_parents' records), in their headers' ns, first to last. None
    when a stretch is unknown, as in evidence restored from a snapshot that kept no ns: float seconds don't round-trip
    to the header ns, so a stretch from them can miss its own first or last record (#1660)."""
    times = [t for e in evidence if type(t := e.get("ns")) is int]
    if len(times) != len(evidence):
        return None
    spans = [("body", min(times), max(times))]
    motion = []
    for e in evidence:
        if e["motion_ids"]:
            span = e.get("motion_span_ns")
            if span is None:
                return None
            motion.append(span)
    if motion:
        spans.append(("motion", min(lo for lo, _ in motion), max(hi for _, hi in motion)))
    return tuple(spans)


def _intervals(evidence: list[dict[str, Any]], gap: float, track: str | None = None) -> tuple[list, list]:
    supported: list[list[float]] = []
    gaps: list[list[float]] = []
    for a, b in pairwise(evidence):
        if b["ts"] <= a["ts"]:
            continue
        good = (
            a["valid"]
            and b["valid"]
            and a["motion"] == b["motion"] == "stable"
            and b["ts"] - a["ts"] <= gap
            and (track is None or a["present"] and b["present"])
        )
        target = supported if good else gaps
        if target and target[-1][1] == a["ts"]:
            target[-1][1] = b["ts"]
        else:
            target.append([a["ts"], b["ts"]])
    return supported, gaps


def _motion_status(evidence: list[dict[str, Any]], threshold: float) -> str:
    values = {e["motion"] for e in evidence}
    bounds = _camera_bounds(evidence)
    if "moved" in values or bounds is not None and max(bounds) > threshold:
        return "moved"
    return "unknown" if bounds is None else "stable"


class MovementPipeline:
    """One camera's bounded state machine. Snapshots and their dependency IDs are persisted together by its caller.

    Config bounds are algorithm parameters, not machine settings. A snapshot from another version, camera, or
    configuration is rejected so its owner can replay raw inputs rather than silently applying incompatible state.
    """

    def __init__(self, cam: str, *, conf: dict[str, Any] | None = None, state: dict[str, Any] | None = None) -> None:
        self.cam = cam
        self.conf = {**DEFAULTS, **(conf or {})}
        for key, value in self.conf.items():
            if key not in DEFAULTS or not _finite(value) or value <= 0:
                raise ValueError(f"invalid movement parameter: {key}")
        for key, minimum, maximum in (
            ("min_samples", 3, 16),
            ("max_points", 3, 32),
            ("max_parents", 8, 64),
            ("max_tracks", 1, 64),
            ("max_motion", 1, 32),
        ):
            value = self.conf[key]
            if int(value) != value or not minimum <= value <= maximum:
                raise ValueError(f"invalid movement bound: {key}")
        self._state: dict[str, Any] = {
            "version": VERSION,
            "cam": cam,
            "conf": self.conf,
            "motion": [],
            "last": None,
            "tracks": {},
            "coverage": [],
        }
        if state is not None:
            if state.get("version") != VERSION or state.get("cam") != cam or state.get("conf") != self.conf:
                raise ValueError("incompatible movement snapshot; replay raw inputs")
            self._state = copy.deepcopy(state)
            if (
                len(self._state["tracks"]) > self.conf["max_tracks"]
                or len(self._state["motion"]) > self.conf["max_motion"]
            ):
                raise ValueError("movement snapshot exceeds state bounds")
            for track in self._state["tracks"].values():
                if (
                    len(track["samples"]) > self.conf["max_points"]
                    or len(_parents(track["evidence"])) > self.conf["max_parents"]
                ):
                    raise ValueError("movement snapshot exceeds fragment bounds")

    def snapshot(self) -> dict[str, Any]:
        """A detached, JSON-safe snapshot. Its owner must commit it before acknowledging the source cursor."""
        return copy.deepcopy(self._state)

    def state_parents(self) -> tuple[str, ...]:
        """Exact source IDs represented by this snapshot, including fresh motion and last-frame context."""
        evidence = list(self._state["coverage"])
        if self._state["last"] is not None:
            evidence.append(self._state["last"])
        for track in self._state["tracks"].values():
            evidence.extend(track["evidence"])
        parents = list(_parents(evidence))
        for motion in self._state["motion"]:
            parents.extend(motion["parents"])
        return tuple(dict.fromkeys(parents))

    def feed_motion(self, rec: dict[str, Any]) -> list[Output]:
        """Cache bounded, timestamped camera evidence; future steps may bracket an earlier body frame but cannot fill a missing motion chain."""
        if not self._input(rec):
            return []
        motions = self._state["motion"]
        if any(m["id"] == rec["id"] for m in motions):
            return []
        value = rec.get("motion")
        value = value if isinstance(value, dict) else {}
        transform = _transform(value.get("transform"))
        status = "unknown"
        if value.get("status") == "measured" and not value.get("lost") and transform is not None:
            shifted = max(
                abs(float(transform[i]) - ideal) for i, ideal in ((0, 1), (1, 0), (2, 0), (3, 0), (4, 1), (5, 0))
            )
            status = "moved" if shifted > self.conf["motion_shift"] else "stable"
        reference = value.get("reference_ts")
        current = value.get("current_ts", rec["ts_start"])
        if (
            not _finite(reference)
            or reference > rec["ts_start"]
            or not _finite(current)
            or abs(current - rec["ts_start"]) > 1e-6
        ):
            status = "unknown"
            reference = None
        epoch = value.get("epoch")
        if epoch is not None and (not isinstance(epoch, int | str) or isinstance(epoch, bool)):
            status = "unknown"
            epoch = None
        before = [m for m in motions if m["ts"] <= rec["ts_start"]]
        previous = max(before, key=lambda m: m["ts"]) if before else None
        parents = [rec["id"]]
        own = _ns(rec)
        if previous is not None:
            parents.append(previous["id"])
            if previous["epoch"] != epoch:
                status = "unknown"
        motions.append(
            {
                "id": rec["id"],
                "ts": float(rec["ts_start"]),
                "epoch": epoch,
                "state": status,
                "reference": reference,
                "matrix": transform,
                "parents": parents,
                # Its own and its parents' first header ns, exact: None when either came without its ns.
                "ns": own,
                "lo_ns": own if previous is None else previous.get("ns"),
            }
        )
        motions.sort(key=lambda m: (m["ts"], m["id"]))
        limit = min(int(self.conf["max_motion"]), int(self.conf["max_parents"]) - 4)
        del motions[: max(0, len(motions) - limit)]
        return []

    def _input(self, rec: dict[str, Any]) -> bool:
        return (
            rec.get("cam") == self.cam
            and isinstance(rec.get("id"), str)
            and bool(rec["id"])
            and _finite(rec.get("ts_start"))
        )

    def _evaluate(self, rec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[float]]]:
        ts = float(rec["ts_start"])
        available = self._state["motion"]
        last = self._state["last"]
        lo = last["ts"] if last is not None else ts
        first = next(
            (
                index
                for index, motion in enumerate(available)
                if motion["reference"] is not None
                and motion["reference"] <= lo + 1e-6
                and motion["ts"] >= lo - 1e-6
                and motion["ts"] - lo <= self.conf["motion_gap"]
            ),
            None,
        )
        selected = []
        if first is not None:
            for motion in available[first:]:
                selected.append(motion)
                if motion["ts"] >= ts - 1e-6:
                    break
        covered = lo
        valid_chain = bool(selected)
        status = "stable"
        steps: list[list[float]] = []
        step_ids = []
        used = []
        for index, motion in enumerate(selected):
            used.extend(motion["parents"])
            reference = motion["reference"]
            if (
                reference is None
                or motion["matrix"] is None
                or motion["ts"] - reference > self.conf["motion_gap"]
                or index
                and abs(reference - covered) > 1e-6
            ):
                valid_chain = False
            # Completion order need not be capture order. Recheck epochs on the
            # selected chronological chain, including its bracketing predecessor.
            if index and motion["epoch"] != selected[index - 1]["epoch"]:
                valid_chain = False
            if motion["state"] == "unknown":
                valid_chain = False
            if motion["state"] == "moved":
                status = "moved"
            if motion["matrix"] is not None:
                steps.append(motion["matrix"])
                step_ids.append(motion["id"])
            covered = motion["ts"]
        if covered < ts - 1e-6:
            valid_chain = False
        if not valid_chain:
            status = "moved" if status == "moved" else "unknown"
            steps, step_ids = [], []
        used = list(dict.fromkeys(used))
        if len(used) >= self.conf["max_parents"]:
            status, steps, step_ids, used = "unknown", [], [], []
        # The motion records `used` names span its chain's predecessors to its last step, in header ns. None when one
        # came without its ns, as from an older snapshot.
        motion_span_ns = (
            [min(m["lo_ns"] for m in selected), max(m["ns"] for m in selected)]
            if used and all(m.get("lo_ns") is not None and m.get("ns") is not None for m in selected)
            else None
        )
        boxes, ids = rec.get("boxes"), rec.get("track_ids")
        valid = not rec.get("error") and isinstance(boxes, list) and isinstance(ids, list) and len(boxes) == len(ids)
        found: dict[str, list[float]] = {}
        if valid:
            counts = Counter(tid for tid in ids if isinstance(tid, str))
            if len(boxes) > self.conf["max_tracks"]:
                valid = False
            for box, tid in zip(boxes, ids):
                if (
                    not isinstance(tid, str)
                    or not tid
                    or counts[tid] != 1
                    or not isinstance(box, list)
                    or len(box) < 5
                    or not all(_finite(v) for v in box[:5])
                    or not 0 <= box[0] < box[2] <= 1
                    or not 0 <= box[1] < box[3] <= 1
                    or not 0 < box[4] <= 1
                ):
                    valid = False
                    continue
                if len(found) < self.conf["max_tracks"]:
                    found[tid] = [float(v) for v in box[:5]]
        return {
            "ts": ts,
            "id": rec["id"],
            "motion": status,
            "motion_ids": used,
            "motion_span_ns": motion_span_ns,
            "ns": _ns(rec),
            "steps": steps,
            "step_ids": step_ids,
            "valid": bool(valid),
            "seen": list(found),
            "boxes": list(found.values()),
        }, found

    def _track_evaluation(self, evaluation: dict[str, Any], tid: str, boundary: list[float]) -> dict[str, Any]:
        value = {key: item for key, item in evaluation.items() if key not in {"seen", "boxes"}}
        value["present"] = tid in evaluation["seen"]
        if not value["present"]:
            margin = float(self.conf["edge_margin"])
            x1, y1, x2, y2 = boundary[:4]
            regions = {
                "left": [0, y1 - margin, x2 + margin, y2 + margin],
                "right": [x1 - margin, y1 - margin, 1, y2 + margin],
                "top": [x1 - margin, 0, x2 + margin, y2 + margin],
                "bottom": [x1 - margin, y1 - margin, x2 + margin, 1],
            }
            value["clear"] = {
                edge: not any(
                    min(region[2], box[2]) > max(region[0], box[0]) and min(region[3], box[3]) > max(region[1], box[1])
                    for box in evaluation["boxes"]
                )
                for edge, region in regions.items()
            }
        return value

    def feed_body(self, rec: dict[str, Any]) -> list[Output]:
        """Consume a successful body snapshot or explicit uncertainty without creating absence from missing inputs."""
        if rec.get("cam") != self.cam:
            return []
        if not self._input(rec):
            return self.flush("unreadable_input")
        last = self._state["last"]
        if last is not None and last["id"] == rec["id"]:
            return []
        out: list[Output] = []
        if last is not None and rec["ts_start"] < last["ts"]:
            out += self.flush("timestamp_order")
            last = None
        evaluation, found = self._evaluate(rec)
        ts = evaluation["ts"]
        coverage = self._state["coverage"]
        if coverage and self._full(coverage, evaluation):
            # Publish every active candidate before asserting this coverage window. Keep
            # young tracks intact so later samples can still establish their entry.
            out.extend(self._track(tid, False, "coverage") for tid in self._state["tracks"])
            out.append(self._coverage(False, "fragment"))
            coverage = self._state["coverage"]
        coverage_evaluation = {key: value for key, value in evaluation.items() if key not in {"seen", "boxes"}}
        if coverage and len(_parents([*coverage, coverage_evaluation])) > self.conf["max_parents"]:
            coverage.clear()  # No source budget for this bridge: the interval remains uncovered.
        coverage.append(coverage_evaluation)
        tracks = self._state["tracks"]
        for tid in list(tracks):
            track = tracks[tid]
            if tid in found and ts - track["samples"][-1]["ts"] > self.conf["max_gap"]:
                out.append(self._track(tid, True, "observation_gap"))
                del tracks[tid]
        for tid, box in found.items():
            if tid not in tracks:
                if len(tracks) >= self.conf["max_tracks"]:
                    # New tracks get their first sample below, after all admissions.
                    # Evict sampled history, preferring tracks absent from this frame.
                    oldest = min(
                        (key for key in tracks if tracks[key]["samples"]),
                        key=lambda key: (key in found, tracks[key]["samples"][-1]["ts"]),
                    )
                    out.append(self._track(oldest, True, "capacity"))
                    del tracks[oldest]
                prior = last is not None and 0 <= ts - last["ts"] <= self.conf["sample_gap"]
                tracks[tid] = {
                    "samples": [],
                    "evidence": [self._track_evaluation(last, tid, box)] if prior else [],
                    "entry_reason": "appearance" if prior else "startup" if last is None else "observation_gap",
                }
        for tid in list(tracks):
            track = tracks[tid]
            boundary = found.get(tid) or track["samples"][-1]["box"]
            track_evaluation = self._track_evaluation(evaluation, tid, boundary)
            if (
                track["samples"]
                and track["evidence"]
                and (
                    self._full(track["evidence"], track_evaluation) or len(track["samples"]) >= self.conf["max_points"]
                )
            ):
                out.append(self._track(tid, False, "fragment"))
                self._carry(track, len(_parents([track_evaluation])))
            track["evidence"].append(track_evaluation)
            if tid in found:
                box = found[tid]
                x, y = anchor(box)
                track["samples"].append({"ts": ts, "id": rec["id"], "point": [ts, x, y], "box": box})
            elif track["samples"] and ts - track["samples"][-1]["ts"] >= self.conf["max_gap"]:
                out.append(self._track(tid, True, "absence"))
                del tracks[tid]
        self._state["last"] = evaluation
        return out

    def _full(self, evidence: list[dict[str, Any]], next_: dict[str, Any]) -> bool:
        return (
            len(_parents([*evidence, next_])) > self.conf["max_parents"]
            or next_["ts"] - evidence[0]["ts"] > self.conf["max_duration_s"]
        )

    def _carry(self, track: dict[str, Any], reserve: int) -> None:
        track["samples"] = track["samples"][-2:]
        lo = track["samples"][0]["ts"]
        # Include missed evaluations after the last positive sample too: removing a trailing absence would
        # falsely bridge the occlusion when the next positive arrives after this rollover.
        evidence = [e for e in track["evidence"] if e["ts"] >= lo]
        if len(_parents(evidence)) > self.conf["max_parents"] - reserve:
            track["samples"] = track["samples"][-1:]
            evidence = [copy.deepcopy(e) for e in evidence if e["id"] == track["samples"][0]["id"]]
            # The bounded fragment cannot carry the whole context. Retain the actual anchor source, but
            # censor continuity rather than publishing an interval from discarded evaluations.
            for evaluation in evidence:
                evaluation.update(
                    valid=False, motion="unknown", motion_ids=[], motion_span_ns=None, steps=[], step_ids=[]
                )
        track["evidence"] = evidence
        track["entry_reason"] = "continuation"

    def _edges(self, box: list[float]) -> list[str]:
        distances = {"left": box[0], "right": 1 - box[2], "top": box[1], "bottom": 1 - box[3]}
        return [edge for edge, distance in distances.items() if distance <= self.conf["edge_margin"]]

    def _edge_travel(self, samples: list[dict[str, Any]], edge: str, inward: bool, bounds: tuple[float, float]) -> bool:
        if len(samples) < self.conf["min_samples"] or samples[-1]["ts"] - samples[0]["ts"] < self.conf["edge_min_s"]:
            return False
        axis = 1 if edge in {"left", "right"} else 2
        sign = 1 if edge in {"left", "top"} else -1
        if not inward:
            sign *= -1
        values = [sample["point"][axis] * sign for sample in samples]
        return values[-1] - values[0] >= self.conf["min_displacement"] + bounds[axis - 1] and all(
            b >= a for a, b in pairwise(values)
        )

    def _boundary(self, tid: str, track: dict[str, Any], *, entry: bool, closure: str) -> dict[str, Any]:
        # min_samples is a floor, not a fixed window. Grow from the actual boundary
        # until duration and personal displacement qualify with continuous evidence.
        result = {"status": "unknown", "reason": track["entry_reason"] if entry else closure}
        for size in range(int(self.conf["min_samples"]), len(track["samples"]) + 1):
            result = self._boundary_window(tid, track, entry=entry, closure=closure, size=size)
            if result["status"] == "supported":
                return result
        return result

    def _boundary_window(
        self, tid: str, track: dict[str, Any], *, entry: bool, closure: str, size: int
    ) -> dict[str, Any]:
        samples, evidence = track["samples"], track["evidence"]
        unknown = {"status": "unknown", "reason": track["entry_reason"] if entry else closure}
        if (
            not samples
            or entry
            and track["entry_reason"] in {"startup", "continuation"}
            or not entry
            and closure != "absence"
        ):
            return unknown
        selected = samples[:size] if entry else samples[-size:]
        boundary = samples[0] if entry else samples[-1]
        context = (
            [e for e in evidence if e["ts"] <= selected[-1]["ts"]]
            if entry
            else [e for e in evidence if e["ts"] >= selected[0]["ts"]]
        )
        if (
            not context
            or not all(e["valid"] for e in context)
            or _motion_status(context, float(self.conf["motion_shift"])) != "stable"
        ):
            return unknown
        bounds = _camera_bounds(context)
        if bounds is None:
            return unknown
        candidates = [edge for edge in self._edges(boundary["box"]) if self._edge_travel(selected, edge, entry, bounds)]
        if len(candidates) != 1:
            return unknown
        edge = candidates[0]
        if any(b["ts"] - a["ts"] > self.conf["sample_gap"] for a, b in pairwise(context)):
            return unknown
        outside = (
            [e for e in context if e["ts"] < boundary["ts"]]
            if entry
            else [e for e in context if e["ts"] > boundary["ts"]]
        )
        if not outside or any(e["present"] or not e.get("clear", {}).get(edge, False) for e in outside):
            return unknown
        # Intervening missed detections cannot establish sustained edge travel.
        lo, hi = selected[0]["ts"], selected[-1]["ts"]
        if any(not e["present"] for e in context if lo <= e["ts"] <= hi):
            return unknown
        interval = [outside[-1]["ts"], boundary["ts"]] if entry else [boundary["ts"], outside[-1]["ts"]]
        return {"status": "supported", "edge": edge, "time": interval, "reason": "edge_motion_and_evaluated_absence"}

    def _track(self, tid: str, finished: bool, closure: str) -> Output:
        track = self._state["tracks"][tid]
        evidence, samples = track["evidence"], track["samples"]
        supported, gaps = _intervals(evidence, float(self.conf["sample_gap"]), tid)
        camera = _motion_status(evidence, float(self.conf["motion_shift"]))
        points = [s["point"] for s in samples]
        direction = "unknown"
        path_gaps = any(hi > points[0][0] and lo < points[-1][0] for lo, hi in gaps) if points else True
        if camera == "stable" and supported and not path_gaps and len(points) >= 2:
            dx, dy = points[-1][1] - points[0][1], points[-1][2] - points[0][2]
            excursion = max(math.hypot(p[1] - points[0][1], p[2] - points[0][2]) for p in points)
            relevant = [e for e in evidence if points[0][0] <= e["ts"] <= points[-1][0]]
            bounds = _camera_bounds(relevant)
            if bounds is not None:
                bx, by = bounds
                if excursion + math.hypot(bx, by) < self.conf["min_displacement"]:
                    direction = "stationary"
                elif max(abs(dx) - bx, abs(dy) - by) >= self.conf["min_displacement"]:
                    direction = (
                        ("right" if dx > 0 else "left")
                        if abs(dx) - bx >= abs(dy) - by
                        else ("down" if dy > 0 else "up")
                    )
        record = {
            "ts_start": evidence[0]["ts"],
            "ts_end": evidence[-1]["ts"],
            "cam": self.cam,
            "track_id": tid,
            "version": VERSION,
            "points": points,
            "supported": supported,
            "gaps": gaps,
            "entry": self._boundary(tid, track, entry=True, closure=closure),
            "exit": self._boundary(tid, track, entry=False, closure=closure),
            "direction": direction,
            "camera_motion": camera,
            "finished": finished,
            "closure": closure,
        }
        return Output("track", record, _parents(evidence), _spans(evidence))

    def _coverage(self, finished: bool, closure: str) -> Output:
        evidence = self._state["coverage"]
        supported, gaps = _intervals(evidence, float(self.conf["sample_gap"]))
        camera = _motion_status(evidence, float(self.conf["motion_shift"]))
        record = {
            "ts_start": evidence[0]["ts"],
            "ts_end": evidence[-1]["ts"],
            "cam": self.cam,
            "version": VERSION,
            "frames": len(evidence),
            "supported": supported,
            "gaps": gaps,
            "tracking": "complete"
            if all(e["valid"] for e in evidence) and camera == "stable" and not gaps
            else "partial",
            "camera_motion": camera,
            "finished": finished,
            "closure": closure,
        }
        self._state["coverage"] = [] if finished else evidence[-1:]
        return Output("coverage", record, _parents(evidence))

    def flush(self, reason: str = "shutdown") -> list[Output]:
        """Censor every open boundary; no wall-clock shutdown is evidence that someone left."""
        out = [self._track(tid, True, reason) for tid in self._state["tracks"] if self._state["tracks"][tid]["samples"]]
        if self._state["coverage"]:
            out.append(self._coverage(True, reason))
        self._state["tracks"] = {}
        self._state["last"] = None
        return out
