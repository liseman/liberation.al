"""Who see-actions' faces saw, read through the registry: when each person was last seen on each camera, from
the `face.visit` records in the segment store. It replaces v1's `detection_identity` projection (a SQLite rematch of
every detection against the gallery, refreshed by a dashboard worker) for the People list's last seen and "when was X
last seen" questions.

A visit names a person as the face worker matched them, and is read as the person who went by that name when the
visit happened (`Registry.known_as(name, at=)`, the rule the other face readers use): a renamed person shows under
the new name, and a forgotten one drops out (nobody is called their name as of then). A visit's time is its end, the
last match of the run.

Every call reads the visits in its window from the segment store and resolves them through one fresh registry
snapshot; a window with an end clamps to it.
"""

import time
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from liberational import segment_log
from liberational.people import registry

#: How far back the newest face matches are read over the visits: a visit is published once its run ends (10 s
#: without a match), so someone still in view has no visit for their current run, only matches.
TAIL_S = 60.0


def last_seen(
    state: str | Path | None = None,
    *,
    start: float = 0.0,
    end: float | None = None,
    cameras: Sequence[str] | None = None,
) -> dict[str, Any]:
    """The newest visit of each person on each camera within [start, end) (seconds; `end` None for now), on `cameras`
    when given, newest first: `{"matches": [{"name", "cam", "ts", "sim"}], "pending": False}`, the shape v1's
    projection answered with (it was pending while it rematched; this never is). `name` is the person's current name."""
    until = end if end is not None else time.time() + 1.0
    log = segment_log.at(state)
    visits = log.range(segment_log.FACE_VISITS, start, until)
    matches = log.range(segment_log.FACE_MATCHES, max(start, until - TAIL_S), until)
    reg = registry.at(state) if registry.log_exists(state) else None
    person_names: dict[str, str] = {}
    wanted = set(cameras or ())
    newest: dict[tuple[str, str], dict[str, Any]] = {}
    names: dict[tuple[str, float], str | None] = {}

    def seen(name: object, cam: object, ts: float, sim: float | None) -> None:
        if not isinstance(name, str) or not name or not cam or (wanted and cam not in wanted):
            return
        if end is not None:
            ts = min(ts, end)
        if ts < start:
            return
        key = (name, ts)
        if key not in names:
            if reg is None:
                names[key] = name
            else:
                pid = reg.known_as(name, at=ts, refresh=False)
                names[key] = person_names.get(pid) if pid is not None else None
        current = names[key]
        if current is None:
            return  # forgotten since (or a name the registry never had)
        at = (current.casefold(), str(cam))
        if at not in newest or ts > newest[at]["ts"]:
            newest[at] = {"name": current, "cam": str(cam), "ts": ts, "sim": sim if sim is not None else 0.0}

    # Keep one identity snapshot across all rows, with no broker probe per distinct visit/name.
    with reg.lock if reg is not None else nullcontext():
        if reg is not None:
            person_names = {p.id: p.display_name for p in reg.people()}
        for v in visits:
            seen(v.get("name"), v.get("cam"), float(v.get("ts_end") or v["ts_start"]), _number(v.get("best_sim")))
        # Someone still in view has no visit for their current run yet: the newest matches of the
        # last TAIL_S say so. They remain a live bounded read, never retained past that tail.
        for m in matches:
            if m.get("name"):
                seen(m["name"], m.get("cam"), float(m["ts_start"]), _number(m.get("sim")))
    return {"matches": sorted(newest.values(), key=lambda r: -r["ts"]), "pending": False, "cursor": 0, "rejected": 0}


def _number(value: Any) -> float | None:
    """A record's similarity as a float, or None for anything that isn't a number."""
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None
