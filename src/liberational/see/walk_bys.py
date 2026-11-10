"""Walk-bys as they happen, for the activity worker's "Walked past" spans: the same rules as
memory/crossing_counts.py, which counts them over a window after the fact.

A `crossing.event` is the moment one track's foot point got past the line; someone at the line makes one every few
seconds. So the activity worker no longer turns each crossing into a span. It feeds this pipeline each body frame
(each track's foot point, kept compactly: its first, last, nearest-left, and furthest-right point, each with the
body record it came from) and each crossing, and the pipeline decides a chain once it has ended: once every track in it has been unseen for FINAL_S,
longer than crossing_counts.RETRACK_S, so no later track can still continue it. A chain that qualifies (it crossed,
and was seen clearly on both sides of the line, or its tracks weren't followed) is one walk-by, at its first
crossing, naming every crossing record in it and the body records its kept points came from (a broken track's
first half can be what saw the far side, and its frames have no crossing of their own); one that doesn't (someone hovering at the line) is no span. The body
worker commits a track's crossings as the track ends, so a decision comes about FINAL_S after the person leaves.

Times are source times (the body frames'), so a replay decides the same. State is plain bounded JSON for the
activity worker's checkpoint.
"""

import copy
from collections.abc import Mapping
from typing import Any

from liberational.memory import crossing_counts as cc
from liberational.vision import crossings as X

VERSION = "walk-bys-1"

FINAL_S = 10.0  # a chain whose tracks have all been unseen this long is decided
KEEP_S = 120.0  # a track unseen this long, with no crossing waiting on it, is forgotten
MAX_TRACKS = 256  # tracks kept per camera
MAX_PENDING = 1024  # crossings waiting for their chain to end, per camera
_KEYS = {"cam", "line", "tracks", "pending", "latest"}


def _point(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(v, int | float) and not isinstance(v, bool) for v in value[:2])
        and isinstance(value[2], str)
    )


class WalkByPipeline:
    """One camera's walk-bys: feed body frames and crossings in source order; `snapshot()` and `state=` round-trip."""

    def __init__(
        self, cam: str, *, line: Mapping[str, Any] | None = None, state: Mapping[str, Any] | None = None
    ) -> None:
        if state is not None:
            _validate(state, cam)
            self._state: dict[str, Any] = copy.deepcopy(dict(state))
            return
        drawn = dict(line) if line is not None else X.line_for(cam, X.settings())[0]
        self._state = {"cam": cam, "line": drawn, "tracks": {}, "pending": [], "latest": None}

    @property
    def line(self) -> dict[str, Any]:
        return self._state["line"]

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self._state)

    def state_parents(self) -> tuple[str, ...]:
        """The crossing records waiting and the body records behind the kept points, so the checkpoint goes with any
        of them."""
        ids = {str(e["id"]) for e in self._state["pending"] if e.get("id")}
        ids.update(p[2] for row in self._state["tracks"].values() for p in row if p[2])
        return tuple(sorted(ids))

    def feed_body(self, rec: Mapping[str, Any]) -> list[dict[str, Any]]:
        """One `body.detection` frame: each tracked box's foot point. Returns the walk-bys this decides."""
        ts = float(rec["ts_start"])
        source = str(rec.get("id") or "")
        self._advance(ts)
        tracks = self._state["tracks"]
        for box, tid in zip(rec.get("boxes") or [], rec.get("track_ids") or [], strict=False):
            if not isinstance(tid, str) or not tid or not isinstance(box, list | tuple) or len(box) < 4:
                continue
            v = X.side_value(self.line, *X.anchor(box))
            row = tracks.get(tid)
            if row is None:
                if len(tracks) >= MAX_TRACKS:
                    continue  # a frame this crowded is past what this follows; its crossings count unverified
                tracks[tid] = [[ts, v, source] for _ in range(4)]  # first, nearest left, furthest right, last
                continue
            if v < row[1][1]:
                row[1] = [ts, v, source]
            if v > row[2][1]:
                row[2] = [ts, v, source]
            row[3] = [ts, v, source]
        return self._settle(ts)

    def feed_crossing(self, rec: Mapping[str, Any]) -> list[dict[str, Any]]:
        """One `crossing.event`: it waits for its chain to end. Returns the walk-bys this decides."""
        event = {k: rec.get(k) for k in ("id", "ts_start", "direction", "track_id")}
        event["ts_start"] = float(event["ts_start"])  # ty: ignore[invalid-argument-type]
        pending = self._state["pending"]
        if any(e["id"] == event["id"] for e in pending if event["id"]):
            return []  # a replay of one already waiting
        pending.append(event)
        out = []
        if len(pending) > MAX_PENDING:  # never drop one: decide the oldest chains early instead
            out = self._settle(None, force=pending[0]["ts_start"])
        return out + self._settle(event["ts_start"])

    def flush(self) -> list[dict[str, Any]]:
        """Decide every waiting chain (the worker is stopping without a checkpoint to resume from)."""
        return self._settle(None, force=float("inf"))

    def _advance(self, ts: float) -> None:
        """Move source time on, and forget tracks unseen for KEEP_S that no waiting crossing needs. This runs on every
        frame, crossings or not, so ordinary traffic never fills MAX_TRACKS."""
        state = self._state
        if state["latest"] is None or ts > state["latest"]:
            state["latest"] = ts
        latest = state["latest"]
        waiting = {str(e.get("track_id")) for e in state["pending"]}
        state["tracks"] = {
            tid: row for tid, row in state["tracks"].items() if latest - row[3][0] <= KEEP_S or tid in waiting
        }

    def _settle(self, ts: float | None, force: float | None = None) -> list[dict[str, Any]]:
        state = self._state
        if ts is not None:
            self._advance(ts)
        latest = state["latest"]
        if not state["pending"] or (latest is None and force is None):
            return []
        rows = state["tracks"]
        seen = {tid: [p[:2] for p in sorted(row, key=lambda p: p[0])] for tid, row in rows.items()}
        out, decided = [], set()
        for chain in cc.chains(state["pending"], seen):
            if not chain["events"]:
                continue
            last = max([p[0] for p in chain["points"]] + [e["ts_start"] for e in chain["events"]])
            due = (force is not None and chain["events"][0]["ts_start"] <= force) or (
                latest is not None and latest - last >= FINAL_S
            )
            if not due:
                continue
            decided.update(id(e) for e in chain["events"])
            if cc.qualifies(chain, self.line):
                sources = {p[2] for tid in chain["tracks"] for p in rows.get(tid, ()) if p[2]}
                out.append({**cc.walk(chain), "sources": sorted(sources)})
        state["pending"] = [e for e in state["pending"] if id(e) not in decided]
        if latest is not None:
            self._advance(latest)
        return sorted(out, key=lambda w: w["ts"])


def _validate(state: Mapping[str, Any], cam: str) -> None:
    """Reject anything this version could not have produced, before it replaces live state."""
    if not isinstance(state, Mapping) or set(state) != _KEYS or state["cam"] != cam:
        raise ValueError("invalid walk-by state")
    X.normalize_line(state["line"])
    latest = state["latest"]
    if latest is not None and not (isinstance(latest, int | float) and not isinstance(latest, bool)):
        raise ValueError("invalid walk-by time")
    tracks, pending = state["tracks"], state["pending"]
    if not isinstance(tracks, Mapping) or len(tracks) > MAX_TRACKS + MAX_PENDING:
        raise ValueError("invalid walk-by tracks")
    for tid, row in tracks.items():
        if not isinstance(tid, str) or not isinstance(row, list) or len(row) != 4 or not all(map(_point, row)):
            raise ValueError("invalid walk-by track")
    if not isinstance(pending, list) or len(pending) > MAX_PENDING:
        raise ValueError("invalid walk-by crossings")
    for e in pending:
        if not isinstance(e, Mapping) or set(e) != {"id", "ts_start", "direction", "track_id"}:
            raise ValueError("invalid walk-by crossing")
        if not isinstance(e["ts_start"], int | float) or isinstance(e["ts_start"], bool):
            raise ValueError("invalid walk-by crossing")
