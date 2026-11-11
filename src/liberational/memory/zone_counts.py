"""How many times something visited a zone: "how many times did the cat use the litter box today".

memory/event_counts.py routes a count question here when it names one of the zones drawn on a camera's view
(see/zones.py) and no detected event has that name. The answer counts the segment store's `zone.visit` candidates
that entered the zone in the window and stayed at least MIN_DWELL_S, over the `zone.coverage` that says where visits
could have been seen:
- No coverage of that zone in the window is "can't count", never zero; partial coverage says how much it covers.
- Visits by the same kind of subject in the same zone less than MERGE_S apart are one visit: a track that splits
  while the cat turns around in the box is still one use of it.
- A visit that ended in a gap (the camera stopped, the zones changed) with less than MIN_DWELL_S seen may have been
  longer; it is reported apart, not counted and not dropped silently.
- A question naming a species the visits have ("the cat") counts that species' visits; one naming a pet ("did May
  use the litter box") counts the visits of May's species that are May by memory/subject_activity.py's attribution
  (labels, recognition, or "probably" the only named pet of the species); otherwise every subject's, split by what
  it was. A question naming a person ("how many times did I go to the bathroom") counts that person's visits: a
  visit's track is theirs when the faces seen on it name them (memory/zone_people.py). Visits whose track no
  face names are reported apart, never counted as theirs or dropped silently.
These are detected visits, not confirmed ones, and the answer says that.
"""

import datetime as dt
import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import numpy as np

from liberational import logs, segment_log
from liberational.see import zones

LOG = logs.get("memory-zone-counts")

MIN_DWELL_S = 10.0  # shorter is passing by: walking past the litter box isn't using it
MERGE_S = 30.0  # same zone, same kind of subject, closer than this: one visit split by its track
GAP_ENDINGS = frozenset({"censored", "zones_changed"})
# The animals the detector knows (COCO's), so a question about the cat reads only animal coverage even before any
# cat visit exists: with no animal coverage that's "can't count", never a zero borrowed from people's coverage.
ANIMALS = frozenset({"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"})
NAMES_TTL_S = 60.0
_NAMES: dict[str, tuple[float, list[str]]] = {}


def names(state: Path) -> list[str]:
    """The names of every zone on every camera now, read at most once a minute.

    A read that fails is logged and not cached: an empty list means no zone is named, and a busy store must not say
    that for the next minute.
    """
    key = str(state)
    hit = _NAMES.get(key)
    if hit is not None and time.monotonic() - hit[0] < NAMES_TTL_S:
        return hit[1]
    try:
        found = sorted({z.name for d in zones.current(state).values() for z in zones.counted(d.zones)})
    except Exception as error:
        LOG.warning("the zones' names could not be read: %r", error)
        return []
    _NAMES[key] = (time.monotonic(), found)
    return found


def _matching(state: Path, asked: str, same: Callable[[str | None, str], bool]) -> dict[str, str]:
    """Zone id -> camera, for each zone now called `asked` (a renamed zone's earlier visits are its own)."""
    return {z.id: cam for cam, d in zones.current(state).items() for z in zones.counted(d.zones) if same(z.name, asked)}


def _clock(ts: float, tz: dt.tzinfo | None) -> str:
    return dt.datetime.fromtimestamp(ts, tz).strftime("%H:%M")


def _duration(seconds: float) -> str:
    minutes = round(seconds / 60)
    if minutes < 1:
        return f"{round(seconds)} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes} min" if hours else f"{minutes} min"


def _union(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def merged(visits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Visits of one zone by one kind of subject, joined when one starts within MERGE_S of the last one's end.

    A joined visit is the first part's dict with `ts_end`, `dwell_s`, and `ended` recomputed over the parts, so its
    dwell is this merge's, not the producer's."""
    out: list[dict[str, Any]] = []
    for v in sorted(visits, key=lambda v: v["ts_start"]):
        last = out[-1] if out else None
        if last and last["label"] == v["label"] and v["ts_start"] - last["ts_end"] < MERGE_S:
            last["ts_end"] = max(last["ts_end"], v["ts_end"])
            last["dwell_s"] = last["ts_end"] - last["ts_start"]
            last["ended"] = v["ended"]
            last["parts"].append(v["id"])
        else:
            out.append({**v, "parts": [v["id"]]})
    return out


def _theirs(
    state: Path, visits: list[dict[str, Any]], person: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(the visits whose track the faces on it say was `person`, people's visits no face named): other people's, and
    animals', are left out of both."""
    from liberational.memory import zone_people
    from liberational.people import registry

    reg = registry.at(state)
    whose = zone_people.people(state, visits, reg.person_id)
    people_visits = [v for v in visits if v.get("label") in (None, "person")]
    mine = [v for v in people_visits if whose.get(str(v["id"])) == person]
    unknown = [v for v in people_visits if whose.get(str(v["id"])) is None]
    return mine, unknown


def _by_zone(visits: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for v in visits:
        groups.setdefault((str(v["zone"]), str(v.get("label"))), []).append(v)
    return list(groups.values())


def answer(
    asked: str,
    query: Mapping[str, Any],
    state: Path,
    stamp: Callable[[float], str],
    tz: dt.tzinfo | None,
    *,
    same: Callable[[str | None, str], bool],
    person: Any = None,
) -> dict[str, Any]:
    """{answer, evidence, coverage, warnings, count, dwell_s} for how many times the zone called `asked` was visited
    in query's window; `dwell_s` is the counted visits' time inside the window. `same` is
    event_counts' name matching, so a zone is named the way an event is."""
    start, end, cameras = float(query["start"]), float(query["end"]), list(query.get("cameras") or [])
    wanted = _matching(state, asked, same)
    if cameras:
        wanted = {z: cam for z, cam in wanted.items() if cam in cameras}
    coverage: dict[str, Any] = {"zones": sorted(wanted), "covered": [], "available": True, "min_dwell_s": MIN_DWELL_S}
    if not wanted:
        return dict(
            answer=f"No camera has a zone called {asked} now, so I can't count visits to it.",
            evidence=[],
            coverage={**coverage, "available": False},
            warnings=["No matching zone."],
        )
    log = segment_log.at(state)
    visits = [
        v
        for v in log.range(segment_log.ZONE_VISITS, start, end)
        if v.get("zone") in wanted and start <= float(v["ts_start"]) < end
    ]
    # The species or pet the reading picked (memory/question.py): "the cat" counts that species' visits, a pet's
    # name the visits attributed to it.
    subjects = query.get("subjects") or {}
    species = subjects.get("species")
    whose = None
    if (pet := subjects.get("pet")) is not None and species is not None:
        from liberational.memory import subject_activity

        whose = subject_activity.attribute(state, species, pet)
    narrowed = person is not None and whose is None and species is None
    kinds = {"animal"} if species else {"person"} if narrowed else {"person", "animal"}
    if species:
        visits = [v for v in visits if v.get("label") == species]
    if whose is not None:
        from liberational.memory import subject_activity

        visits = [v for v in visits if subject_activity.keeps(whose, v.get("track_id"))]
    unknown_whose: list[dict[str, Any]] = []
    if narrowed:
        visits, unknown_whose = _theirs(state, visits, person)
    looked: set[str] = set()

    def checked(columns: Mapping[str, Any]) -> np.ndarray:
        """The coverage rows that looked for one of `kinds` in one of the zones asked about."""
        keep = np.array(
            [
                kind in kinds and not wanted.keys().isdisjoint(json.loads(zones))
                for kind, zones in zip(columns["subject_kind"], columns["zones"], strict=True)
            ],
            dtype=bool,
        )
        looked.update(kind for kind, kept in zip(columns["subject_kind"], keep.tolist(), strict=True) if kept)
        return keep

    found = log.covered(segment_log.ZONE_COVERAGE, start, end, where=(("subject_kind", "zones"), checked))
    unlooked = sorted(kinds - looked)
    covered = _union([span for cam in found.values() for span in cam.spans])
    seconds = sum(b - a for a, b in covered)
    coverage["covered"] = [{"start": a, "end": b} for a, b in covered]
    coverage["seconds"] = seconds
    where = " and ".join(sorted(set(wanted.values())))
    if not covered:
        return dict(
            answer=f"No footage of {asked} has been checked for visits in that time yet, so I can't count them.",
            evidence=[],
            coverage={**coverage, "count_available": False},
            warnings=["Nothing covers the zone in the window: this is not a count of zero."],
        )
    joined = sorted((m for g in _by_zone(visits) for m in merged(g)), key=lambda v: v["ts_start"])
    counted = [v for v in joined if v["dwell_s"] >= MIN_DWELL_S]
    cut = [v for v in joined if v["dwell_s"] < MIN_DWELL_S and v.get("ended") in GAP_ENDINGS]
    who = (
        whose.who
        if whose is not None
        else species or (str(query.get("person") or "you") if narrowed else "anyone or anything")
    )
    if counted:
        times = ", ".join(_clock(float(v["ts_start"]), tz) for v in counted)
        by = sorted({str(v.get("label")) for v in counted})
        text = (
            f"{asked}: {len(counted)} visit{'s' if len(counted) != 1 else ''} of at least {MIN_DWELL_S:g} s by {who} "
            f"({times})"
        )
        if not species and not narrowed and len(by) > 1:
            split = ", ".join(f"{sum(1 for v in counted if v.get('label') == b)} by a {b}" for b in by)
            text += f": {split}"
        elif not species and not narrowed:
            text += f", all by a {by[0]}"
        text += "."
    else:
        text = f"{asked}: no visit of at least {MIN_DWELL_S:g} s by {who} in the checked footage."
    window = end - start
    if seconds < window - 1:
        text += f" Footage on {where} was checked for {_duration(seconds)} of that time."
    if unlooked and len(unlooked) < len(kinds):
        text += f" No {' or '.join(k + 's' for k in unlooked)} were looked for in it, so they aren't counted."
    if cut:
        text += f" {len(cut)} shorter visit{'s' if len(cut) != 1 else ''} ended in a gap and may have been longer."
    if whose is not None and whose.note:
        text += f" {whose.note}"
    if unknown_whose:
        long = [v for g in _by_zone(unknown_whose) for v in merged(g) if v["dwell_s"] >= MIN_DWELL_S]
        if long:
            n = len(long)
            text += (
                f" {n} more visit{'s' if n != 1 else ''} by someone no face named on the camera "
                f"({', '.join(_clock(float(v['ts_start']), tz) for v in long)}) "
                f"{'are' if n != 1 else 'is'} not counted."
            )
    text += " These are detected visits, not confirmed ones."
    evidence = [
        dict(
            ts=float(v["ts_start"]),
            camera=v.get("cam"),
            source="camera",
            quote=f"{v.get('zone_name') or asked}: a {v.get('label')} for {round(v['dwell_s'])} s, "
            f"{_clock(float(v['ts_start']), tz)} to {_clock(float(v['ts_end']), tz)}",
            link="/review#"
            + urlencode(dict(day=stamp(float(v["ts_start"]))[:10], t=float(v["ts_start"]), cam=v.get("cam") or "")),
            method="ZoneVisit: a tracked subject inside an owner-drawn zone, an unconfirmed candidate",
            stream="zone.visit",
            event=v["id"],
        )
        for v in counted
    ]
    dwell = sum(max(0.0, min(float(v["ts_end"]), end) - float(v["ts_start"])) for v in counted)
    return dict(
        answer=text,
        evidence=evidence,
        coverage={**coverage, "count_available": True},
        warnings=[],
        count=len(counted),
        dwell_s=dwell,
    )
