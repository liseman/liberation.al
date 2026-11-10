"""Who was seen while a voice spoke, and who was heard while a face was in view (#225): a suggestion for an unnamed
face from the named voices heard while it was in view, and for an unnamed voice from the named faces in view while it
spoke. A suggestion is only that: "Who's this?" shows it with its evidence, and the owner accepts it with the same
owner tag a manual naming writes. Nothing here asserts who anyone is.

Counting rules (see the constants):

- Together means within WINDOW_S of a line: a face seen from WINDOW_S before a line starts to WINDOW_S after it ends.
- Co-occurrences within MOMENT_S of each other are one moment, so one long conversation counts once, not per line.
- A suggestion needs MIN_MOMENTS moments, and SHARE of all the moments that name anyone.
- For a face, the owner's voice never counts: the cameras are on a rig the owner wears, so the owner is heard nearly
  all the time and almost never seen. Nor does a named voice whose own named face was in view at that moment: that
  speech is explained by the face already known.
"""

import bisect
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

WINDOW_S = 3.0
MOMENT_S = 60.0
MIN_MOMENTS = 2
SHARE = 2 / 3
MOMENTS_SHOWN = 5  # the newest moments a suggestion carries as its evidence

Pair = tuple[float, str]  # (when, the name seen or heard then)


def spans(times: Iterable[float]) -> list[tuple[float, float]]:
    """Times grouped into moments, as (first, last): a time within MOMENT_S of a moment's last time joins it. Oldest
    first."""
    out: list[list[float]] = []
    for t in sorted(times):
        if out and t - out[-1][1] <= MOMENT_S:
            out[-1][1] = t
        else:
            out.append([t, t])
    return [(a, b) for a, b in out]


def moments(times: Iterable[float]) -> list[float]:
    """When each moment started (`spans`), oldest first."""
    return [a for a, _ in spans(times)]


def suggest(pairs: Iterable[Pair]) -> dict[str, Any] | None:
    """The name the co-occurrences point to, or None: {"name", "moments" (how many), "last" (when the newest moment
    ended), "times" (when the newest few moments started, newest first), "others" ({name: moments} for the rest)}."""
    by: dict[str, list[float]] = {}
    for t, name in pairs:
        by.setdefault(name, []).append(t)
    counted = {name: spans(ts) for name, ts in by.items()}
    if not counted:
        return None
    name, best = max(counted.items(), key=lambda kv: (len(kv[1]), kv[1][-1][1]))
    total = sum(len(m) for m in counted.values())
    if len(best) < MIN_MOMENTS or len(best) < SHARE * total:
        return None
    return {
        "name": name,
        "moments": len(best),
        "last": best[-1][1],
        "times": [a for a, _ in best[::-1][:MOMENTS_SHOWN]],
        "others": {n: len(m) for n, m in counted.items() if n != name},
    }


class Heard:
    """Who was heard while a face was in view, set up once for many faces: `Heard(lines, seen, owner)(face_times)`."""

    def __init__(
        self, lines: Sequence[tuple[float, float, str]], seen: Sequence[Pair], owner: Iterable[str] = ()
    ) -> None:
        self.lines = lines
        self.seen = seen
        self.mine = {o.casefold() for o in owner}
        self.starts = [s for s, _, _ in lines]
        self.seen_at = [t for t, _ in seen]
        self.longest = max((e - s for s, e, _ in lines), default=0.0)
        self.lo_edge = np.asarray(self.starts, dtype=np.float64) - WINDOW_S
        ends = np.asarray([e for _, e, _ in lines], dtype=np.float64)
        self.hi_edge = (np.maximum.accumulate(ends) if len(ends) else ends) + WINDOW_S

    def __call__(self, face_times: Iterable[float]) -> list[Pair]:
        out: list[Pair] = []
        times = np.asarray(list(face_times), dtype=np.float64)
        if not len(self.lines) or not len(times):
            return out
        # Only the sightings within reach of some line: a face is seen far more often than anyone named speaks.
        k = np.searchsorted(self.lo_edge, times, side="right") - 1  # the last line starting (with the window) by t
        near = (k >= 0) & (self.hi_edge[np.clip(k, 0, None)] >= times)
        for t in times[near].tolist():
            lo = bisect.bisect_left(self.starts, t - WINDOW_S - self.longest)
            hi = bisect.bisect_right(self.starts, t + WINDOW_S)
            heard = {name for s, e, name in self.lines[lo:hi] if s - WINDOW_S <= t <= e + WINDOW_S}
            if not heard:
                continue
            a = bisect.bisect_left(self.seen_at, t - WINDOW_S)
            b = bisect.bisect_right(self.seen_at, t + WINDOW_S)
            faces_then = {name.casefold() for _, name in self.seen[a:b]}
            out.extend((t, n) for n in heard if n.casefold() not in self.mine and n.casefold() not in faces_then)
        return out


def seen_with(spans: Iterable[tuple[float, float]], seen: Sequence[Pair]) -> list[Pair]:
    """For a voice's lines, (start, end), the named faces in view while it spoke: (line start, the face's name) for
    each named sighting within WINDOW_S of the line. `seen` is sorted by time."""
    seen_at = [t for t, _ in seen]
    out: list[Pair] = []
    for s, e in spans:
        a, b = bisect.bisect_left(seen_at, s - WINDOW_S), bisect.bisect_right(seen_at, e + WINDOW_S)
        out.extend((s, name) for name in {name for _, name in seen[a:b]})
    return out
