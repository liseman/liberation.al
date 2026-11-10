"""Who a person's zone visit was (#1497): "how many times do I go to the bathroom at night" counts only the owner's.

A zone visit (see/zones.py) is a see-actions track inside a drawn zone, with no identity of its own. see-actions
matches faces on the same tracks (see/head_faces.py), so each `face.match` names its `pose.track` id (#1846): the
matches on a visit's track, within LOOK_S before it started or after it ended, say who it was. The track's person is
the name most of its matches had, when at least MIN_SHARE of them agree; two people on one track (a handoff of the
track) leave it unknown rather than guessed.

Animals' visits are never anyone's. This reads at answer time, so forgetting a person (or their faces) changes the
answer the next time it is asked.
"""

import collections
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from liberational import segment_log

LOOK_S = 120.0  # a face this long before or after the visit, on the same track, still says who it was
MIN_SHARE = 0.75  # this share of a track's matched faces must agree on one name


def track_name(log: Any, cam: str, track: str, start: float, end: float) -> str | None:
    """The name the faces matched on this track agree on, around [start, end), or None."""
    votes: collections.Counter[str] = collections.Counter(
        str(m["name"])
        for m in log.range(segment_log.FACE_MATCHES, start - LOOK_S, end + LOOK_S)
        if m.get("cam") == cam and str(m.get("track_id")) == track and isinstance(m.get("name"), str) and m["name"]
    )
    if not votes:
        return None
    name, n = votes.most_common(1)[0]
    return name if n >= MIN_SHARE * sum(votes.values()) else None


def people(state: Path, visits: Iterable[Mapping[str, Any]], person_of: Callable[[str], Any]) -> dict[str, Any]:
    """Visit id -> the PersonId its track's faces name (`person_of` maps a name to one), or None when they don't."""
    log = segment_log.at(state)
    out: dict[str, Any] = {}
    names: dict[tuple[str, str], str | None] = {}
    for v in visits:
        if v.get("subject_kind") == "animal" or v.get("label") not in (None, "person"):
            out[str(v["id"])] = None
            continue
        cam, track = str(v.get("cam")), str(v.get("track_id"))
        key = (cam, track)
        if key not in names:
            names[key] = track_name(log, cam, track, float(v["ts_start"]), float(v.get("ts_end") or v["ts_start"]))
        name = names[key]
        out[str(v["id"])] = person_of(name) if name else None
    return out
