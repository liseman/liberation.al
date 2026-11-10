"""Shared, deterministic movement answers from durable camera-local track fragments.

Only supported image-boundary edges count. Track birth, disappearance, query boundaries, missing coverage,
and camera motion do not become entries or exits. Counts on partial coverage are observed lower bounds,
never unique people or room-wide totals. Reads include bounded neighbor context for pairwise co-arrivals;
no permanent day cache can hide late fragments or a purge. Source IDs remain evidence links, not identities.
"""

import bisect
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from liberational import segment_log
from liberational.domain.kernel import check_span

Span = tuple[float, float]
DIRECTIONS = frozenset({"left", "right", "up", "down", "stationary", "unknown"})
EDGES = frozenset({"left", "right", "top", "bottom"})
METHOD = "camera-local tracked movement; supported image-boundary edges"


def _number(value: Any) -> float:
    if type(value) is float and math.isfinite(value):  # nearly every timestamp: skip the checks below
        return value
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError("movement timestamps must be finite numbers")
    return float(value)


def _span(value: Any) -> Span:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError("movement interval needs two endpoints")
    lo, hi = _number(value[0]), _number(value[1])
    if lo > hi:
        raise ValueError("movement interval is reversed")
    return lo, hi


def _spans(record: Mapping[str, Any], key: str) -> list[Span]:
    value = record.get(key)
    if not isinstance(value, list | tuple):
        raise ValueError(f"movement {key} intervals unavailable")
    bounds = _span([record.get("ts_start"), record.get("ts_end")])
    result = [_span(item) for item in value]
    if any(not _inside(interval, bounds) for interval in result):
        raise ValueError(f"movement {key} extends outside its record")
    return result


def _inside(interval: Span, bounds: Span) -> bool:
    # Header nanoseconds and JSON seconds can round at different floating endpoints.
    lo_slack = math.ulp(interval[0]) + math.ulp(bounds[0])
    hi_slack = math.ulp(interval[1]) + math.ulp(bounds[1])
    return interval[0] >= bounds[0] - lo_slack and interval[1] <= bounds[1] + hi_slack


def _union(spans: Iterable[Span]) -> list[Span]:
    result: list[Span] = []
    for lo, hi in sorted(spans):
        if hi <= lo:
            continue
        if result and lo <= result[-1][1]:
            result[-1] = result[-1][0], max(hi, result[-1][1])
        else:
            result.append((lo, hi))
    return result


def _subtract(spans: Iterable[Span], holes: Iterable[Span]) -> list[Span]:
    cuts = _union(holes)
    result = []
    for lo, hi in _union(spans):
        cursor = lo
        for a, b in cuts:
            if b <= cursor or a >= hi:
                continue
            if a > cursor:
                result.append((cursor, a))
            cursor = max(cursor, b)
        if cursor < hi:
            result.append((cursor, hi))
    return result


def _intersect(left: Iterable[Span], right: Iterable[Span]) -> list[Span]:
    a, b = _union(left), _union(right)
    i = j = 0
    result = []
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            result.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return result


def _near(
    spans: Sequence[Span], starts: Sequence[float], ends: Sequence[float], lo: float, hi: float
) -> Sequence[Span]:
    """The spans of a sorted, non-overlapping list that touch [lo, hi]: a superset of those the interval helpers act
    on there, so passing the slice gives the same answer as the whole list, without walking a day of spans per track."""
    return spans[bisect.bisect_left(ends, lo) : bisect.bisect_right(starts, hi)]


class _Index:
    """A day's supported spans and barriers, arranged once so each track's checks are lookups."""

    def __init__(self, supported: Sequence[Span], barriers: Sequence[Span]) -> None:
        self.supported = list(supported)  # _subtract's output: sorted and non-overlapping
        self.sup_starts = [lo for lo, _ in self.supported]
        self.sup_ends = [hi for _, hi in self.supported]
        # Positive-width barriers merged: what _subtract cuts with (it drops zero-width ones), and, as closed spans,
        # what any barrier overlaps. Zero-width barriers are kept apart as points.
        self.cuts = _union(barriers)
        self.cut_starts = [lo for lo, _ in self.cuts]
        self.cut_ends = [hi for _, hi in self.cuts]
        self.points = sorted(lo for lo, hi in barriers if lo == hi)

    def own(self, observed: Sequence[Span], gaps: Sequence[Span], span: Span) -> list[Span]:
        """_subtract(_intersect(observed, supported), [*barriers, *gaps]) for a track inside `span`."""
        lo, hi = span
        seen = _intersect(observed, _near(self.supported, self.sup_starts, self.sup_ends, lo, hi))
        return _subtract(seen, [*_near(self.cuts, self.cut_starts, self.cut_ends, lo, hi), *gaps])

    def covered(self, span: Span, points: Sequence[float]) -> bool:
        """_covered(span, supported, [*barriers, *points as spans])."""
        lo, hi = span
        i = bisect.bisect_right(self.cut_starts, hi) - 1
        if i >= 0 and self.cuts[i][1] >= lo:
            return False
        j = bisect.bisect_left(self.points, lo)
        if j < len(self.points) and self.points[j] <= hi:
            return False
        if any(lo <= p <= hi for p in points):
            return False
        k = bisect.bisect_right(self.sup_starts, lo) - 1
        return k >= 0 and hi <= self.supported[k][1]


def _touches(span: Span, start: float, end: float) -> bool:
    lo, hi = span
    return start <= lo < end if lo == hi else lo < end and hi >= start


def _covered(span: Span, supported: Sequence[Span], barriers: Sequence[Span]) -> bool:
    lo, hi = span
    if any(a <= hi and b >= lo for a, b in barriers):
        return False
    return any(a <= lo and hi <= b for a, b in supported)


def _unknown(span: Span, reason: str, source: str | None, start: float, end: float) -> dict[str, Any] | None:
    if not _touches(span, start, end):
        return None
    return {
        "start": max(start, span[0]),
        "end": min(end, span[1]),
        "reasons": [reason],
        "source_ids": [source] if source else [],
    }


def _merge_unknown(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Union unavailable durations without losing zero-width continuity barriers."""
    spans: list[dict[str, Any]] = []
    points: dict[float, dict[str, Any]] = {}
    for item in sorted(items, key=lambda value: (value["start"], value["end"])):
        if item["start"] == item["end"]:
            last = points.setdefault(item["start"], {**item, "reasons": [], "source_ids": []})
        elif spans and item["start"] <= spans[-1]["end"]:
            last = spans[-1]
            last["end"] = max(last["end"], item["end"])
        else:
            last = {**item, "reasons": [], "source_ids": []}
            spans.append(last)
        last["reasons"] = sorted(set(last["reasons"]) | set(item["reasons"]))
        last["source_ids"] = sorted(set(last["source_ids"]) | set(item["source_ids"]))
    return sorted([*spans, *points.values()], key=lambda value: (value["start"], value["end"]))


def _identity(record: Mapping[str, Any]) -> tuple[str, str, Span]:
    cam, source = record.get("cam"), record.get("id")
    if not isinstance(cam, str) or not cam or not isinstance(source, str) or not source:
        raise ValueError("movement record needs camera and source ID")
    return cam, source, _span([record.get("ts_start"), record.get("ts_end")])


def _camera(
    camera: str,
    records: Sequence[dict[str, Any]],
    coverage: Sequence[dict[str, Any]],
    start: float,
    end: float,
    group_window: float,
) -> dict[str, Any]:
    support: list[Span] = []
    complete: list[Span] = []
    barriers: list[Span] = []
    unknown: list[dict[str, Any]] = []
    errors: list[str] = []
    coverage_ids: set[str] = set()

    def uncertain(span: Span, reason: str, source: str | None = None) -> None:
        value = _unknown(span, reason, source, start, end)
        if value is not None:
            unknown.append(value)

    for record in coverage:
        source = str(record.get("id", ""))
        try:
            _, source, span = _identity(record)
            observed, gaps = _spans(record, "supported"), _spans(record, "gaps")
            tracking, motion = record.get("tracking"), record.get("camera_motion")
            frames = record.get("frames")
            if tracking not in {"complete", "partial"} or motion not in {"stable", "moved", "unknown"}:
                raise ValueError("movement coverage status unavailable")
            if isinstance(frames, bool) or not isinstance(frames, int) or frames < 0:
                raise ValueError("movement coverage frames unavailable")
            if motion != "stable" or not frames:
                barriers.append(span)
                uncertain(span, "camera-motion-" + str(motion) if motion != "stable" else "missing-tracking", source)
            else:
                support.extend(observed)
                coverage_ids.add(source)
                if tracking == "complete":
                    complete.extend(observed)
                else:
                    uncertain(span, "partial-tracking", source)
            barriers.extend(gaps)
            for gap in gaps:
                uncertain(gap, "tracking-gap", source)
        except TypeError, ValueError:
            errors.append(f"Invalid movement coverage: {source or 'missing source ID'}")
    supported = _subtract(support, barriers)
    full_support = _subtract(complete, barriers)
    clipped = _intersect(full_support, [(start, end)])
    available = bool(_intersect(supported, [(start, end)]))
    index = _Index(supported, barriers)
    for gap in _subtract([(start, end)], clipped):
        uncertain(gap, "unsupported-tracking")

    tracks: dict[str, dict[str, Any]] = {}
    all_events: dict[tuple[Any, ...], dict[str, Any]] = {}
    uncertain_events: list[dict[str, Any]] = []
    for record in records:
        source = str(record.get("id", ""))
        try:
            _, source, span = _identity(record)
            track_id = record.get("track_id")
            if not isinstance(track_id, str) or not track_id:
                raise ValueError("movement track ID unavailable")
            observed, gaps = _spans(record, "supported"), _spans(record, "gaps")
            motion = record.get("camera_motion")
            if motion not in {"stable", "moved", "unknown"}:
                raise ValueError("movement camera motion unavailable")
            mine = index.own(observed, gaps, span) if motion == "stable" else []
            boundary_absence: list[Span] = []
            edge_points = [gap[0] for gap in gaps if gap[0] == gap[1]]
            if _touches(span, start, end):
                track = tracks.setdefault(
                    track_id, {"track_id": track_id, "source_ids": set(), "supported": [], "directions": []}
                )
                track["source_ids"].add(source)
                track["supported"].extend(_intersect(mine, [(start, end)]))
                direction = record.get("direction", "unknown")
                if direction not in DIRECTIONS:
                    raise ValueError("movement direction unavailable")
                track["directions"].append(
                    {
                        "direction": direction if motion == "stable" else "unknown",
                        "start": max(start, span[0]),
                        "end": min(end, span[1]),
                        "source_ids": [source],
                    }
                )
                if motion != "stable":
                    uncertain(span, "camera-motion-" + str(motion), source)
            for kind in ("entry", "exit"):
                edge = record.get(kind)
                if edge is None:
                    continue
                if not isinstance(edge, dict):
                    raise ValueError("movement edge unavailable")
                if edge.get("status") == "supported" or edge.get("time") is not None:
                    at = _span(edge.get("time"))
                else:
                    # Startup, continuation, and shutdown can censor an edge without a time estimate.
                    # Keep that uncertainty at its fragment boundary without losing the opposite edge.
                    boundary = span[0] if kind == "entry" else span[1]
                    at = (boundary, boundary)
                if not _inside(at, span):
                    raise ValueError("movement edge extends outside its record")
                event = {
                    "track_id": track_id,
                    "kind": kind,
                    "edge": edge.get("edge"),
                    "time": list(at),
                    "source_ids": [source],
                }
                established = (
                    edge.get("status") == "supported"
                    and edge.get("edge") in EDGES
                    and motion == "stable"
                    # Positive dwell gaps include the evaluated absence before entry and after exit.
                    # The producer's supported edge uses that absence, while dwell must exclude it.
                    # Camera coverage holes, motion, and zero-width continuity barriers still veto edges.
                    and index.covered(at, edge_points)
                )
                if established:
                    boundary_absence.append(at)
                    key = (track_id, kind, edge["edge"], *at)
                    if key in all_events:
                        all_events[key]["source_ids"] = sorted(set(all_events[key]["source_ids"]) | {source})
                    else:
                        all_events[key] = event
                elif _touches(at, start, end):
                    reason = str(edge.get("reason") or "unsupported-movement-edge")
                    uncertain_events.append({**event, "reason": reason})
                    uncertain(at, reason, source)
            if _touches(span, start, end):
                for gap in gaps:
                    # Only validated boundary absence explains a non-visible interval. Interior
                    # occlusions and all continuity barriers remain unknown, and add no dwell.
                    unexplained = [gap] if gap[0] == gap[1] else _subtract([gap], boundary_absence)
                    for interval in unexplained:
                        uncertain(interval, "track-occlusion", source)
        except TypeError, ValueError:
            errors.append(f"Invalid movement track: {source or 'missing source ID'}")

    def belongs(event: Mapping[str, Any]) -> bool:
        lo, hi = event["time"]
        return start <= lo and hi < end

    events = sorted(all_events.values(), key=lambda event: (*event["time"], event["track_id"], event["kind"]))
    for event in events:
        if _touches(tuple(event["time"]), start, end) and not belongs(event):
            uncertain_events.append({**event, "reason": "query-boundary"})
            uncertain(tuple(event["time"]), "query-boundary", event["source_ids"][0])
    entries = [event for event in events if event["kind"] == "entry" and belongs(event)]
    exits = [event for event in events if event["kind"] == "exit" and belongs(event)]
    neighbor_entries = [event for event in events if event["kind"] == "entry"]
    pairs = []
    for i, first in enumerate(neighbor_entries):
        for second in neighbor_entries[i + 1 :]:
            if second["time"][0] - first["time"][1] > group_window:
                break
            separation = max(abs(first["time"][0] - second["time"][1]), abs(first["time"][1] - second["time"][0]))
            if (
                first["track_id"] != second["track_id"]
                and separation <= group_window
                and (belongs(first) or belongs(second))
            ):
                pairs.append(
                    {
                        "track_ids": [first["track_id"], second["track_id"]],
                        "times": [first["time"], second["time"]],
                        "source_ids": sorted(set(first["source_ids"]) | set(second["source_ids"])),
                    }
                )
    output_tracks = []
    for track in sorted(tracks.values(), key=lambda track: track["track_id"]):
        intervals = _union(track.pop("supported"))
        output_tracks.append(
            {
                **track,
                "source_ids": sorted(track["source_ids"]),
                "supported_intervals": [list(span) for span in intervals],
                "supported_s": sum(hi - lo for lo, hi in intervals),
            }
        )
    unknown = _merge_unknown(unknown)
    available = available and not errors
    partial = bool(unknown or uncertain_events or errors or not available)
    return {
        "available": available,
        "entries": entries,
        "exits": exits,
        "entry_count": len(entries) if available else None,
        "exit_count": len(exits) if available else None,
        "tracks": output_tracks,
        "coarrivals": pairs,
        "supported_intervals": [list(span) for span in clipped],
        "supported_s": sum(hi - lo for lo, hi in clipped),
        "unknown": unknown,
        "unknown_seconds": sum(item["end"] - item["start"] for item in unknown),
        "barriers": [u for u in unknown if u["start"] == u["end"]],
        "uncertain_events": uncertain_events,
        "source_ids": sorted(coverage_ids),
        "partial": partial,
        "errors": errors,
    }


def collect(
    state: str | Path,
    *,
    start: float,
    end: float,
    cameras: Sequence[str] = (),
    group_window: float = 3.0,
) -> dict[str, Any]:
    """Per-camera supported movement, clipped to [start,end), with bounded co-arrival context."""
    check_span(start, end)
    if start >= end:
        raise ValueError("movement window must have positive duration")
    window = _number(group_window)
    if window < 0:
        raise ValueError("movement group window must be nonnegative")
    result: dict[str, Any] = {
        "method": METHOD,
        "window": {"start": start, "end": end},
        "cameras": {},
        "available": False,
        "partial": True,
        "errors": [],
    }
    try:
        log = segment_log.at(state)
        lo, hi = max(0.0, start - window), end + window
        # The producer appends every contributing track before its coverage; appends never change what's already
        # there, so reading coverage first can't pair old empty tracks with newly appended complete coverage.
        coverage = log.range(segment_log.MOVEMENT_COVERAGE, lo, hi)
        tracks = log.range(segment_log.MOVEMENT_TRACKS, lo, hi)
    except (OSError, TimeoutError, RuntimeError, ValueError) as error:
        result["errors"] = [type(error).__name__]
        return result
    names = sorted(
        set(cameras) or {r["cam"] for r in [*tracks, *coverage] if isinstance(r.get("cam"), str) and r["cam"]}
    )
    for camera in names:
        data = _camera(
            camera,
            [r for r in tracks if r.get("cam") == camera],
            [r for r in coverage if r.get("cam") == camera],
            start,
            end,
            window,
        )
        result["cameras"][camera] = data
        result["errors"].extend(data["errors"])
    result["available"] = any(data["available"] for data in result["cameras"].values())
    result["partial"] = not names or any(data["partial"] for data in result["cameras"].values())
    return result


def answer_text(result: Mapping[str, Any]) -> str:
    """Explain supported observations without turning missing coverage into zero or merging cameras."""
    cameras = result.get("cameras") or {}
    if not cameras or not result.get("available"):
        return "Tracked movement evidence is unavailable; I cannot establish entry or exit counts for this window."
    lines = []
    for camera, data in cameras.items():
        if not data["available"]:
            lines.append(f"{camera}: tracked movement coverage is unavailable.")
            continue
        qualifier = "At least " if data["partial"] else ""
        lines.append(
            f"{qualifier}{data['entry_count']} supported entries and {data['exit_count']} supported exits on {camera}."
        )
        if data["partial"]:
            lines.append("Missing coverage or uncertain boundaries prevent a complete count.")
    lines.append(
        "These are camera-local image-boundary observations, not room entries or unique people; cameras are reported separately."
    )
    return "\n".join(lines)
