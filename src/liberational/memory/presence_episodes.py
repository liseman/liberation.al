"""Times someone was in view of a camera: episodes from see-actions' analyzed frames (see/tracked_frames.py).

An episode starts at the first analyzed frame with anyone tracked and ends at the last one before nobody was seen for
`absence_s` (10 s) of analyzed frames. A gap of more than `gap_s` (30 s) with no analyzed frames at all ends it too,
as censored: nobody was watching, so it may have gone on. An episode that starts right after such a gap (or at the
first frame ever analyzed) is left-censored: someone may have been there before anyone looked. Counts are of
episodes, never of people: a track id is camera- and run-local, never an identity.

Pure bookkeeping over frames, like see/zones.py: see/zones_worker.py feeds it and writes `presence.span` rows, and
memory/presence_counts.py reads them with see-actions' coverage.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

ABSENCE_S = 10.0
GAP_S = 30.0
MAX_TRACKS = 256  # distinct track ids remembered per open episode


class Episodes:
    """One camera's open episode. `snapshot()` and `state=` round-trip it for a checkpoint."""

    def __init__(
        self,
        cam: str,
        *,
        absence_s: float = ABSENCE_S,
        gap_s: float = GAP_S,
        state: Mapping[str, Any] | None = None,
    ) -> None:
        self.cam, self.absence_s, self.gap_s = cam, absence_s, gap_s
        self.last: float | None = None  # the last analyzed frame's time
        self.open: dict[str, Any] | None = None  # start, seen (last frame with anyone), frames, max_people, tracks
        if state is not None:
            self.last = state.get("last")
            self.open = copy.deepcopy(state.get("open"))

    def snapshot(self) -> dict[str, Any]:
        return {"last": self.last, "open": copy.deepcopy(self.open)}

    def _close(self, right_censored: bool) -> dict[str, Any]:
        e = self.open
        assert e is not None
        self.open = None
        return {
            "cam": self.cam,
            "ts_start": e["start"],
            "ts_end": e["seen"],
            "frames": e["frames"],
            "max_people": e["max_people"],
            "tracks": len(e["tracks"]),
            "left_censored": e["left_censored"],
            "right_censored": right_censored,
        }

    def frame(self, ts: float, track_ids: Sequence[str]) -> list[dict[str, Any]]:
        """One analyzed frame and who was tracked in it; returns the episodes it ended."""
        out: list[dict[str, Any]] = []
        if self.last is not None and ts <= self.last:
            return out  # older or repeated: already past it
        unwatched = self.last is None or ts - self.last > self.gap_s
        if unwatched and self.open is not None:
            out.append(self._close(right_censored=True))
        if self.open is not None and ts - self.open["seen"] >= self.absence_s:
            out.append(self._close(right_censored=False))  # nobody seen for absence_s of watched time
        if track_ids:
            if self.open is None:
                self.open = {
                    "start": ts,
                    "seen": ts,
                    "frames": 0,
                    "max_people": 0,
                    "tracks": [],
                    "left_censored": unwatched,
                }
            e = self.open
            e["seen"], e["frames"] = ts, e["frames"] + 1
            e["max_people"] = max(e["max_people"], len(track_ids))
            for t in track_ids:
                if t not in e["tracks"] and len(e["tracks"]) < MAX_TRACKS:
                    e["tracks"].append(t)
        self.last = ts
        return out
