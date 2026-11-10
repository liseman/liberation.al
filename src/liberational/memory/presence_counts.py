"""One per-camera count of the times someone came into view, for answers, speech, chips, and summaries.

Callers gate it with enabled(). The episodes are see-zones' `presence.span` rows (memory/presence_episodes.py, from
see-actions' tracked frames), and what was watched is see-actions' `action.coverage`: time it didn't analyze is
unknown, never "nobody". Counts are episode starts in the half-open window, never a headcount or a sum across
cameras, and never who: a track is camera-local.

Both reads are a window of the segment store, so there is no projection to keep warm and no cache: any window is
answered directly (`ready` is always true).
"""

from __future__ import annotations

import math
import pathlib
import time
from collections.abc import Mapping, Sequence
from typing import Any

from liberational import config as cfg
from liberational import segment_log
from liberational.memory import movement_counts

RULES = 4  # what summaries fingerprint: 4 is episodes from see-actions' tracked frames
COVERED_GAP_S = 1.0  # coverage rows this close are one watched stretch (see-actions writes them back to back)


def enabled() -> bool:
    """The plain product switch; reject ambiguous non-boolean configuration."""
    value = cfg.get("presence.enabled")
    if not isinstance(value, bool):
        raise ValueError("presence.enabled must be a boolean")
    return value


def settings() -> dict[str, float]:
    """The episode rules (memory/presence_episodes.py): an unwatched gap and an absence, in seconds."""
    result = {}
    for name, key in (("gap", "presence.gap_s"), ("absence", "presence.absence_s")):
        value = cfg.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
        result[name] = float(value)
    return result


def day_bounds(day: str) -> tuple[float, float]:
    """[start, end) of a local calendar day 'YYYY-MM-DD' as epoch seconds (DST days are 23 or 25 hours)."""
    t = time.strptime(day, "%Y-%m-%d")
    return time.mktime(t), time.mktime((t.tm_year, t.tm_mon, t.tm_mday + 1, 0, 0, 0, 0, 0, -1))


def ready(start: float, end: float, cameras: Sequence[str] = ()) -> bool:
    """Whether collect answers this window quickly: always, now that it's two segment reads."""
    return True


class Unavailable(RuntimeError):
    """The segment store could not be read (collect's answer said so)."""

    def __init__(self, message: str, *, pending: bool = False) -> None:
        super().__init__(message)
        self.pending = pending


def _watched(rows: Sequence[Mapping[str, Any]], start: float, end: float) -> list[tuple[float, float]]:
    """see-actions' coverage spans clipped to [start, end) and joined where they meet."""
    out: list[tuple[float, float]] = []
    for r in sorted(rows, key=lambda r: r["ts_start"]):
        lo, hi = max(start, r["ts_start"]), min(end, r["ts_end"])
        if hi <= lo:
            continue
        if out and lo - out[-1][1] <= COVERED_GAP_S:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def _unknown(watched: Sequence[tuple[float, float]], start: float, end: float) -> list[dict[str, Any]]:
    out, at = [], start
    for lo, hi in watched:
        if lo > at:
            out.append({"start": at, "end": lo, "reasons": ["not-analyzed"]})
        at = max(at, hi)
    if at < end:
        out.append({"start": at, "end": end, "reasons": ["not-analyzed"]})
    return out


def collect(
    state: str | pathlib.Path,
    *,
    start: float,
    end: float,
    cameras: Sequence[str] = (),
) -> dict[str, Any]:
    """Each camera's episodes in [start, end) (those overlapping it, counted where they start inside it), the window's
    time it wasn't watched, and the movement answer beside them. A camera see-actions never analyzed in the window
    isn't listed: nothing about it is known."""
    if not (math.isfinite(start) and math.isfinite(end)) or start >= end:
        raise ValueError("presence window must have positive duration")
    try:
        log = segment_log.at(state)
        spans = log.range(segment_log.PRESENCE_SPANS, start, end)
        covered = log.range(segment_log.ACTION_COVERAGE, start, end)
    except Exception as error:  # an unreadable store answers "can't say", never "nobody"; a file error may pass
        failed = {"start": start, "end": end, "cameras": {}, "partial": True, "error": str(error)}
        return {**failed, "retryable": isinstance(error, OSError)}
    wanted = set(cameras)
    names = sorted({str(r["cam"]) for r in covered} | {str(s["cam"]) for s in spans})
    out: dict[str, Any] = {"start": start, "end": end, "cameras": {}, "partial": False, "diagnostics": []}
    for cam in names:
        if wanted and cam not in wanted:
            continue
        watched = _watched([r for r in covered if r["cam"] == cam], start, end)
        unknown = _unknown(watched, start, end)
        episodes = [
            {
                "start": s["ts_start"],
                "end": s["ts_end"],
                "left_censored": s["left_censored"],
                "right_censored": s["right_censored"],
                "tracks": s["tracks"],
                "max_people": s["max_people"],
                "counted": start <= s["ts_start"] < end,
            }
            for s in sorted(spans, key=lambda s: s["ts_start"])
            if s["cam"] == cam
        ]
        starts = [e["start"] for e in episodes if e["counted"]]
        censored = sum(1 for e in episodes if e["counted"] and e["left_censored"])
        out["cameras"][cam] = {
            "count": len(starts),
            "starts": starts,
            "episodes": episodes,
            "unknown": unknown,
            "unknown_seconds": sum(u["end"] - u["start"] for u in unknown),
            "censored_starts": censored,
            "partial": bool(unknown or censored),
        }
    out["partial"] = not out["cameras"] or any(c["partial"] for c in out["cameras"].values())
    return {**out, "movement": movement_counts.collect(state, start=start, end=end, cameras=cameras)}


def sampled_answer_text(result: Mapping[str, Any]) -> str:
    """How many separate times someone was in view of each camera, in plain words: never a count of different people
    or of a named person, never a zero where a camera wasn't watching."""
    if result.get("pending"):
        return "I'm still loading the camera history, so I can't say yet how many times someone came into view."
    if result.get("error"):
        return "I can't read the camera history right now, so I can't say how many times someone came into view."
    cameras = result.get("cameras") or {}
    if not cameras:
        return "No camera was watching then, so I can't say how many times someone came into view."
    one = len(cameras) == 1
    lines = []
    for camera, values in cameras.items():
        count = values["count"]
        where = "the camera" if one else camera
        if not count and values["unknown_seconds"] >= result["end"] - result["start"]:
            lines.append(f"I can't tell how many times someone was in view of {where}: it wasn't watching.")
            continue
        times = {1: "once", 2: "twice"}.get(count, f"{count} separate times")
        line = f"Someone was in view of {where} {times}." if count else f"Nobody was seen in view of {where}."
        if values["partial"]:
            line += " It wasn't watching the whole time, so there may have been more."
        lines.append(line)
    lines.append("That counts times someone was in view, not different people, and it can't say who.")
    return " ".join(lines)


def answer_text(
    result: Mapping[str, Any],
    crossed: Mapping[str, Any] | None = None,
    tz: Any = None,
    elapsed_s: float | None = None,
    so_far: bool = False,
) -> str:
    """Ask's and voice's walk-by answer: how many times people walked past and came into the picture, in the
    words of memory/walked_past.py (from `crossed`, a crossing_counts.from_spans of the window, and the
    result's movement), then how many separate times someone was in view."""
    from liberational.memory.walked_past import walked_past_text

    window = float(result.get("end", 0)) - float(result.get("start", 0))
    walked = walked_past_text(
        crossed, result.get("movement") or {}, window if elapsed_s is None else elapsed_s, tz, so_far=so_far
    )
    sampled = sampled_answer_text(result)
    return f"{walked} {sampled}" if walked else sampled
