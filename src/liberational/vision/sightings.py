"""Who was seen, and when: the faces see-actions found on its tracked heads (see/head_faces.py), read from the
segment store:

  face.match  one look at a tracked head: its box (as a share of the frame), detector score, quality, its pose.track
              id, and who it is by the face database (its name and similarity), or which unknown face group
  face.visit  a run of one person's matches on one camera, closed after a gap

A sighting here is a face.match row: {id, ts, cam, bbox_norm, det, quality, track_id, name, sim, cluster}. Its name
is the people registry's answer for the sighting first (`person_of`: the owner's word, or the identity worker's
claim), the match's name when the registry has none, and nobody when the owner withdrew it. The owner's corrections
(People's "Not <name>" and "Assign to") are registry claims on the sighting (`correct`). The read helpers return plain
dicts for the dashboard, history Q&A, and the timeline; an id is the match's id (a string).
Readers that scan every sighting (shots, the best shots, unnamed clips) look back SHOT_DAYS.

The voice helpers at the end (hand_labeled, candidate, decide) are the voice pass's, kept here for its callers.
"""

import bisect
import datetime as dt
import itertools
import json
import pathlib
import re
import time
import uuid
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from typing import Any

import numpy as np

from liberational import segment_log, vision_log
from liberational.domain.identity import Basis
from liberational.intervals import interval_groups
from liberational.people import registry
from liberational.see import subjects
from liberational.see.faces import VISIT_GAP_S, VISIT_SHOTS, readable

FACE_THRESH = 0.45  # same as vision/faces.py THRESH (live matching); kept here so readers don't import InsightFace
AUTO = re.compile(r"Speaker \d+", re.IGNORECASE)  # in any capitals, as names compare
SHOT_DAYS = 30  # how far back the readers that scan every sighting look
SPREAD_GAP_S = 60.0  # a second shot from the same day must be this far from the others on its camera
SPREAD_POOL = 500  # how many of the best shots best_spread chooses among
SHOT_NORMS = "shot_norms.json"  # in the state directory: tools/score_shot_crops.py's norms of older shots
NORMS_TTL_S = 60.0  # how long the shots' face.signal norms are reused before they're read again

type State = str | pathlib.Path | None


def _now() -> float:
    return time.time()


def _same(a: object, b: str) -> bool:
    """Names compare without case."""
    return isinstance(a, str) and a.casefold() == b.casefold()


def faces(start: float, end: float, cam: str | None = None, state: State = None) -> list[dict[str, Any]]:
    """Every face see-actions matched in [start, end), on `cam` when given, oldest first, each with who it is now:
    [{id, ts, cam, bbox_norm, det, quality, frontal, sharp, track_id, name, sim, cluster}]."""
    # A camera's faces are read from its slot alone: filtering after reading every camera's cost the whole read.
    found = segment_log.at(state).range(segment_log.FACE_MATCHES, start, end, None if cam is None else [cam])
    return _face_rows(found, {r["id"]: r for r in found}, state) if found else []


def faces_many(start: float, end: float, cams: Iterable[str], state: State = None) -> list[dict[str, Any]]:
    """Every matched face in [start, end) on any camera in `cams`, oldest first, named as `faces()` names them."""
    slots = sorted({cam for cam in cams if cam})
    if not slots:
        return []
    found = segment_log.at(state).range(segment_log.FACE_MATCHES, start, end, slots)
    return _face_rows(found, {r["id"]: r for r in found}, state) if found else []


class _WindowIndex:
    """Sorted starts and prefix maximum ends retain nested spans with half-open overlap."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        # The legacy float/id order can reverse distinct nanoseconds that round to one float. Index exact starts,
        # then restore legacy order within each selected window before joining matches.
        self.records = sorted(
            records, key=lambda r: r["_start_ns"] if "_start_ns" in r else vision_log.to_ns(r["ts_start"])
        )
        self.starts = [r["_start_ns"] if "_start_ns" in r else vision_log.to_ns(r["ts_start"]) for r in self.records]
        self.ends = [
            max(a + 1, r["_end_ns"] if "_end_ns" in r else vision_log.to_ns(r["ts_end"]))
            for a, r in zip(self.starts, self.records)
        ]
        self.max_ends = list(itertools.accumulate(self.ends, max))

    def windows(self, windows: Sequence[tuple[float, float]]) -> Iterator[list[dict[str, Any]]]:
        for a, b in windows:
            lo, hi = vision_log.to_ns(a), vision_log.to_ns(b)
            stop = bisect.bisect_left(self.starts, hi)
            first = bisect.bisect_right(self.max_ends, lo, 0, stop)
            yield sorted(
                [self.records[i] for i in range(first, stop) if self.ends[i] > lo],
                key=lambda r: (r["ts_start"], r.get("id", "")),
            )


class FaceBatch:
    """One request's face snapshot, shared by guesses and co-occurrence: one segment read, sliced into the windows
    each caller asks for. Names and window-local joins remain request-owned."""

    def __init__(self, start: float, end: float, state: State = None) -> None:
        self.start, self.end, self.state = start, end, state
        self.found: _WindowIndex | None = None
        self.matched: _WindowIndex | None = None
        self.who: _Who | None = None
        if end <= start or end - start > (SHOT_DAYS + 1) * 86400:
            return
        found = segment_log.at(state).range(segment_log.FACE_MATCHES, start, end)
        self.found = self.matched = _WindowIndex(found)
        if found:
            self.who = _Who(registry.at(state))

    def faces(
        self, windows: Sequence[tuple[float, float]], *, newest_first: bool = False
    ) -> list[dict[str, Any]] | None:
        if self.found is None or self.matched is None or any(a < self.start or b > self.end for a, b in windows):
            return None
        return [
            row
            for detections, decisions in zip(self.found.windows(windows), self.matched.windows(windows))
            for row in sorted(
                _face_rows(detections, {m.get("detection"): m for m in decisions}, self.state, who=self.who)
                if detections
                else [],
                key=lambda r: (r["ts"], r["id"]),
                reverse=newest_first,
            )
        ]


def bounded_faces(
    start: float,
    end: float,
    state: State = None,
    *,
    windows: Sequence[tuple[float, float]] | None = None,
    newest_first: bool = False,
) -> list[dict[str, Any]] | None:
    """Try one bounded snapshot; callers retain their existing windowed reads on None."""
    if end <= start:
        return []
    return FaceBatch(start, end, state).faces(
        windows if windows is not None else [(start, end)], newest_first=newest_first
    )


def _face_rows(
    found: list[dict[str, Any]], matches: dict[Any, dict[str, Any]], state: State, *, who: _Who | None = None
) -> list[dict[str, Any]]:
    """Project one set of detections and matches through the same current identity rules for both read paths."""
    from liberational.vision import face_rejects

    who = who or _Who(registry.at(state))
    rejected = face_rejects.rejected()
    out = []
    for r in found:
        m = matches.get(r["id"]) or {}
        if r["id"] in rejected and m.get("cluster"):  # taken out of its face group by the owner
            m = {**m, "cluster": None}
        name = who.detection(r["id"], r["ts_start"], m)
        out.append(
            {
                "id": r["id"],
                "ts": r["ts_start"],
                "cam": r.get("cam"),
                "bbox_norm": r.get("bbox"),
                "det": r.get("det"),
                "quality": r.get("quality"),
                "frontal": r.get("frontal"),
                "sharp": r.get("sharp"),
                "track_id": r.get("track_id"),
                "name": name,
                "sim": 1.0 if who.decided.get(r["id"]) else m.get("sim"),
                "cluster": m.get("cluster"),
            }
        )
    return out


class _Who:
    """Who each face is, in the order agreed with the identity worker: the registry's answer for the detection
    (an owner correction, or a decision the identity worker claimed), nobody when the owner withdrew it, the match's
    name as the person who had it when it matched, by their name now (renames followed; nobody once that person is
    forgotten; a name the registry never had kept as it is), the person the match's face group was named since
    ("Face 3", named by an introduction), or nobody. Every registry call reads what's new in its log first, so a
    read asks once which detections anything was said about (only those are looked up one by one), and looks each
    person and group up once."""

    def __init__(self, reg: registry.Registry) -> None:
        self.reg = reg
        self.decided: dict[str, bool] = {}  # detection id -> the registry decided it (the owner's word, or a claim)
        self.fresh = False  # the registry's people were read once this read; later lookups answer from that
        self.groups: dict[str, str | None] = {}
        self.names: dict[Any, str | None] = {}  # person id -> their name now
        self.said: set[str] | None = None  # the detections anything was said about, read on first use

    def _name_of(self, pid: Any) -> str | None:
        if pid is None:
            return None
        if pid not in self.names:
            person = self.reg.person(pid)
            self.names[pid] = person.display_name if person is not None else None
        return self.names[pid]

    def detection(self, sid: str, ts: float, match: Mapping[str, Any]) -> str | None:
        if self.said is None:
            self.said = {s.removeprefix("detection:") for s in self.reg.observed("detection:")}
        if sid in self.said:
            decision = self.reg.person_of(registry.detection(sid))
            if decision.person is not None:
                self.decided[sid] = True
                return self._name_of(decision.person)
            if decision.retractions:
                return None
        name = match.get("name")
        if isinstance(name, str) and name:
            pid = self.reg.known_as(name, at=ts, refresh=not self.fresh)  # who was called that when it matched
            self.fresh = True
            if pid is not None:
                return self._name_of(pid)
            return None if self.reg.forgotten_at(name, ts, refresh=False) else name
        group = subjects.FACE_GROUP.fullmatch(str(match.get("cluster") or ""))
        if group is None:
            return None
        key = group.group(1)
        if key not in self.groups:
            self.groups[key] = self._name_of(self.reg.person_of(registry.face_group(int(key))).person)
        return self.groups[key]


ID_LAG_S = 60.0  # a worker makes a record's id at most this long after the frame it's about


def _id_time(sid: str) -> float | None:
    """When a uuid7 payload id was made (seconds): the face worker makes each as it commits, just after its frame.
    None for an id that isn't a uuid7."""
    try:
        u = uuid.UUID(sid)
    except ValueError:
        return None
    return (u.int >> 80) / 1000 if u.version == 7 else None


def _windows(times: Sequence[float]) -> list[tuple[float, float]]:
    """The stretches to read for faces whose ids were made at `times`: [t - ID_LAG_S, t + 1] each, overlaps joined."""
    out: list[list[float]] = []
    for t in sorted(times):
        if out and t - ID_LAG_S <= out[-1][1]:
            out[-1][1] = t + 1
        else:
            out.append([t - ID_LAG_S, t + 1])
    return [(a, b) for a, b in out]


def _found(sid: str, state: State, around: float | None = None) -> dict[str, Any] | None:
    """Detection `sid` as faces() has it (named by who it is now), or None. Looked for near `around` when given, else
    just before its id's time, else over the last SHOT_DAYS."""
    made = _id_time(sid)
    if around is not None:
        since, until = around - 1, around + 1
    elif made is not None:
        since, until = made - ID_LAG_S, made + 1
    else:
        until = _now() + 1
        since = until - SHOT_DAYS * 86400
    return next((f for f in faces(since, until, state=state) if f["id"] == sid), None)


def correct(ids: Sequence[str], name: str | None, state: State = None) -> list[dict[str, Any]]:
    """The owner says sightings `ids` are `name` (None: nobody they know), as registry claims on each detection (the
    owner's word outranks the identity worker's). Saying someone else, or nobody, also withdraws the name it had
    without the owner (its match's, or its face group's), so it stays off that person even if the one named now is
    deleted later. Returns what each was, for uncorrect(): [{id, was}] (was: the name the registry gave it before, None
    for none; "not": the person whose negative its face became)."""
    from liberational.vision import faces as face_db

    reg = registry.at(state)
    prior: list[dict[str, Any]] = []
    denied: dict[str, str] = {}  # sighting -> the person it had and isn't: their negative in the matcher
    for sid in dict.fromkeys(ids):
        subject = registry.detection(sid)
        decision = reg.person_of(subject)
        person = reg.person(decision.person) if decision.person is not None else None
        prior.append({"id": sid, "was": person.display_name if person is not None else None})
        # Who it is without the owner's word (its match, or the person its face group was named), to withdraw
        matched = (_found(sid, state) or {}).get("name") if decision.person is None else None
        had = person.display_name if person is not None else matched
        if had and (name is None or registry.norm(had) != registry.norm(name)):
            denied[sid] = str(had)
        if matched and (name is None or registry.norm(matched) != registry.norm(name)):
            # A withdrawal needs a claim to withdraw: the owner says who the match said, then takes it back.
            reg.claim(subject, matched, Basis.OWNER_TAG, "dashboard")
            reg.retract(subject, "dashboard")
        elif name is None:
            reg.retract(subject, "dashboard")
        if name is not None:
            reg.claim(subject, name, Basis.OWNER_TAG, "dashboard")
    if name is not None and prior:
        # The owner's word teaches the gallery too: the faces new to the person become their rows, and a
        # "not them" said of these faces before is taken back, so it can't hold the new rows' name off them
        for sid in dict.fromkeys(ids):
            face_db.remove_not(name, f"shot:{sid}")
        taught = _embedding_rows(list(dict.fromkeys(ids)), state)
        rows = face_db.add_correction_rows(
            name,
            [np.asarray(r["vector"], np.float32) for r in taught.values()],
            [_example(sid, r, state) for sid, r in taught.items()],
        )
        if rows:
            prior[0].update(rows=rows, rows_name=name)
    if denied:
        # "Not <person>" teaches too: each face becomes one of their negatives, so the next look of it isn't
        # named them (a run of near-identical faces adds one; see faces.add_not)
        found = _vector_ids(list(denied), state)
        for p in prior:
            sid = p["id"]
            if sid in found and face_db.add_not(denied[sid], found[sid], f"shot:{sid}"):
                p["not"] = denied[sid]
    return prior


def _vectors(ids: Sequence[str], state: State) -> list[np.ndarray]:
    """The face embeddings see-actions recorded for detections `ids`, in their order (those it still has)."""
    found = _vector_ids(ids, state)
    return [found[sid] for sid in ids if sid in found]


def _vector_ids(ids: Sequence[str], state: State) -> dict[str, np.ndarray]:
    """The face embeddings see-actions recorded for detections `ids`, by id (those it still has)."""
    return {sid: np.asarray(r["vector"], np.float32) for sid, r in _embedding_rows(ids, state).items()}


def _embedding_rows(ids: Sequence[str], state: State) -> dict[str, dict[str, Any]]:
    """The face embedding rows see-actions recorded for detections `ids`, by id in their order (those it still has)."""
    times = [t for t in (_id_time(sid) for sid in ids) if t is not None]
    found: dict[str, dict[str, Any]] = {}
    for a, b in _windows(times):
        for r in segment_log.at(state).range(segment_log.FACE_EMBEDDINGS, a, b, None):
            if r.get("match_id") in ids and r.get("vector") is not None:
                found[str(r["match_id"])] = r
    return {sid: found[sid] for sid in ids if sid in found}


def _example(sid: str, row: Mapping[str, Any], state: State) -> Any:
    """Where a corrected sighting's face row came from, for Training images: its camera, its time, and the crop
    see-actions saved of it (`state/shots`, any revision: the owner is looking at this sighting now)."""
    from liberational import config as cfg
    from liberational.people import face_shots
    from liberational.vision.face_examples import CORRECTION, Example

    jpg = None
    try:
        home = pathlib.Path(state) if state is not None else cfg.path("paths.state")
        saved = sorted(face_shots.directory(home, sid).glob(f"{sid}-*.jpg"), key=lambda f: f.stat().st_mtime)
        jpg = saved[-1].read_bytes() if saved else None
    except OSError:
        jpg = None
    cam = row.get("cam")
    at = row.get("ts_start")
    return Example(
        CORRECTION,
        cam=str(cam) if isinstance(cam, str) and cam else None,
        at=float(at) if isinstance(at, int | float) else _id_time(sid),
        ref=f"shot:{sid}",
        jpg=jpg,
    )


def uncorrect(prior: Sequence[Mapping[str, Any]], state: State = None) -> None:
    """Undo correct(): each sighting the registry named before is named so again; one it named nobody goes back to
    the face worker's match (everything the owner said about it is released)."""
    from liberational.vision import faces as face_db

    reg = registry.at(state)
    for p in prior:
        if p.get("rows"):  # the rows the correction added to the gallery
            for row in p["rows"]:
                face_db.remove_row(str(p["rows_name"]), row["vec"], row["dropped"])
        if p.get("not"):  # the negative it added
            face_db.remove_not(str(p["not"]), f"shot:{p['id']}")
        subject = registry.detection(p["id"])
        if p.get("was"):
            reg.claim(subject, p["was"], Basis.OWNER_TAG, "dashboard")
        else:
            reg.release(subject, "dashboard")


# ---- visits ----
def _visits(since: float, until: float, state: State) -> list[dict[str, Any]]:
    """face.visit records overlapping [since, until], as rows: {name, cam, start, end, best_sim, samples}, with
    the ones the registry has changed a face of since rebuilt (_reconciled)."""
    rows = [
        {
            "name": v.get("name"),
            "cam": v.get("cam"),
            "start": v["ts_start"],
            "end": v["ts_end"],
            "best_sim": v.get("best_sim"),
            "samples": v.get("samples"),
            "shots": v.get("shots") or [],
            "id": v.get("id"),
        }
        for v in segment_log.at(state).range(segment_log.FACE_VISITS, since, until + 0.001)
        if v.get("name")
    ]
    return _reconciled(rows, since, until, state)


def _reconciled(rows: list[dict[str, Any]], since: float, until: float, state: State) -> list[dict[str, Any]]:
    """`rows` with every visit the registry changed a face of rebuilt from its faces as they're named now. A visit is
    committed once, under its match's name, so a correction (Not X, a move) or a face group named since would
    otherwise leave it under the old name. The faces the registry said anything about are found by the time in their
    ids; on each camera the stretch around the changed ones is read again and split into visits by name."""
    if not registry.log_exists(state):  # nobody has said anything about a face yet
        return rows
    ids = [s.removeprefix("detection:") for s in registry.at(state).observed("detection:")]
    times = {i: t for i in ids if (t := _id_time(i)) is not None and since - ID_LAG_S <= t <= until + ID_LAG_S}
    if not times:
        return rows
    said: list[dict[str, Any]] = []
    for a, b in _windows(list(times.values())):
        said += [f for f in faces(a, b, state=state) if f["id"] in times]
    spans: dict[str, list[tuple[float, float]]] = {}
    for f in said:
        cam = str(f["cam"])
        cover = next((v for v in rows if v["cam"] == cam and v["start"] <= f["ts"] <= v["end"]), None)
        if cover is None and not f["name"] or cover is not None and _same(f["name"], cover["name"]):
            continue  # nobody, with no visit to undo, or still the visit's person
        spans.setdefault(cam, []).append((cover["start"], cover["end"]) if cover else (f["ts"], f["ts"]))
    out = list(rows)
    for cam, found in spans.items():
        for group in interval_groups(found, 0.0, start=lambda s: s[0], end=lambda s: s[1]):
            a, b = min(s[0] for s in group), max(s[1] for s in group)
            stale = [v for v in out if v["cam"] == cam and v["start"] <= b and v["end"] >= a]
            a, b = min([a, *(v["start"] for v in stale)]), max([b, *(v["end"] for v in stale)])
            out = [v for v in out if not any(v is s for s in stale)] + _rebuilt(cam, a, b, state)
    return out


def _best_faces(found: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The VISIT_SHOTS best-quality faces with a box among `found`, best first, as a visit's shots ({detection, ts,
    bbox, quality, sim})."""
    boxed = sorted((f for f in found if f.get("bbox_norm")), key=lambda f: (-float(f.get("quality") or 0.0), -f["ts"]))
    return [
        {
            "detection": f["id"],
            "ts": f["ts"],
            "bbox": f["bbox_norm"],
            "quality": float(f.get("quality") or 0.0),
            "sim": f.get("sim"),
        }
        for f in boxed[:VISIT_SHOTS]
    ]


def _rebuilt(cam: str, start: float, end: float, state: State) -> list[dict[str, Any]]:
    """The visits on `cam` from start to end made again from its faces as they're named now: each person's run of
    faces no more than VISIT_GAP_S apart."""
    by_name: dict[str, list[dict[str, Any]]] = {}
    for f in faces(start, end + 0.001, cam, state):
        if f["name"]:
            by_name.setdefault(str(f["name"]).casefold(), []).append(f)
    out = []
    for seen in by_name.values():
        for run in interval_groups(seen, VISIT_GAP_S, start=lambda f: f["ts"], end=lambda f: f["ts"]):
            out.append(
                {
                    "name": run[-1]["name"],
                    "cam": cam,
                    "start": run[0]["ts"],
                    "end": run[-1]["ts"],
                    "best_sim": max((f["sim"] or 0.0 for f in run), default=None),
                    "samples": len(run),
                    "shots": _best_faces(run),
                    "payload_ids": [f["id"] for f in run],
                }
            )
    return out


def visits(name: str, since: float = 0.0, until: float | None = None, state: State = None) -> list[dict[str, Any]]:
    """Visits of `name` (case-insensitive) overlapping [since, until], newest first: [{name, cam, start, end, best_sim,
    samples}]."""
    until = _now() if until is None else until
    found = [v for v in _visits(since, until, state) if _same(v["name"], name)]
    return sorted(found, key=lambda v: -v["end"])


def visits_between(since: float = 0.0, until: float | None = None, state: State = None) -> list[dict[str, Any]]:
    """Every visit overlapping [since, until], oldest first: [{name, cam, start, end, best_sim, samples}]. The
    timeline (memory/timeline.py) reads them all to build encounters."""
    until = _now() if until is None else until
    return sorted(_visits(since, until, state), key=lambda v: v["start"])


def last_seen(name: str, state: State = None) -> float | None:
    """When `name` was last seen, or None."""
    return max((v["end"] for v in _visits(0.0, _now(), state) if _same(v["name"], name)), default=None)


def last_seen_where(state: State = None) -> dict[str, tuple[float, str]]:
    """{lowercased name: (last time seen, the camera)}."""
    out: dict[str, tuple[float, str]] = {}
    for v in _visits(0.0, _now(), state):
        key = str(v["name"]).lower()
        if key not in out or v["end"] > out[key][0]:  # "Alice" and "alice" are one person
            out[key] = (v["end"], str(v["cam"]))
    return out


def who_at(ts: float, window: float = 30.0, state: State = None) -> list[dict[str, Any]]:
    """Who was seen within +-window seconds of ts: [{name, cam, start, end, best_sim}], one per person and camera, the
    best match first."""
    runs: dict[tuple[str, str], dict[str, Any]] = {}
    for v in _visits(ts - window, ts + window, state):
        key = (str(v["name"]), str(v["cam"]))
        r = runs.setdefault(key, {"name": v["name"], "cam": v["cam"], "start": v["start"], "end": v["end"]})
        r["start"], r["end"] = min(r["start"], v["start"]), max(r["end"], v["end"])
        r["best_sim"] = max(r.get("best_sim") or 0.0, v["best_sim"] or 0.0)
        ids = r.setdefault("payload_ids", [])
        source_ids = [v["id"]] if v.get("id") is not None else v.get("payload_ids", [])
        for payload_id in source_ids:
            if payload_id is not None and payload_id not in ids:
                ids.append(payload_id)
    return sorted(runs.values(), key=lambda r: -(r.get("best_sim") or 0.0))


def recent_visits(name: str, limit: int = 4, state: State = None) -> list[dict[str, Any]]:
    """The newest visits of `name` (case-insensitive), newest first: [{cam, start, end}]."""
    return [{"cam": v["cam"], "start": v["start"], "end": v["end"]} for v in visits(name, state=state)[:limit]]


def visit_ids(name: str, cam: str, start: float, end: float, state: State = None) -> list[str]:
    """Ids of `name`'s sightings (case-insensitive) on `cam` from start to end: the shots one visit is made of."""
    return [f["id"] for f in faces(start, end + 0.001, cam, state) if _same(f["name"], name)]


# ---- shots ----
def boxes(cam: str, start: float, end: float, state: State = None) -> list[dict[str, Any]]:
    """Faces found on `cam` between start and end (for drawing over the player while scrubbing), oldest first:
    [{ts, bbox_norm: [x1, y1, x2, y2] as shares of the frame, name, sim}]. name is None for a face nobody is named
    for."""
    return [
        {"ts": f["ts"], "bbox_norm": f["bbox_norm"], "name": f["name"], "sim": f["sim"]}
        for f in faces(start, end + 0.001, cam, state)
        if f["bbox_norm"]
    ][:2000]


def _named(name: str, state: State, since: float | None = None, until: float | None = None) -> list[dict[str, Any]]:
    """`name`'s faces (case-insensitive) with a box over [since, until], oldest first: every face in that stretch."""
    until = _now() if until is None else until
    since = until - SHOT_DAYS * 86400 if since is None else since
    return [f for f in faces(since, until + 0.001, state=state) if _same(f["name"], name) and f["bbox_norm"]]


_OLD_SHOTS: dict[tuple[str, str, str], list[dict[str, Any]]] = {}  # (state, visit id, name): its best faces


def _old_visit_shots(v: Mapping[str, Any], name: str, state: State) -> list[dict[str, Any]]:
    """The best faces of a visit committed before visits carried their shots, read from its stretch once per process
    (a stored visit's faces don't change; one the registry changed since is rebuilt with its shots, and has no id)."""
    key = (str(state), str(v.get("id")), name.casefold())
    if v.get("id") is not None and key in _OLD_SHOTS:
        return _OLD_SHOTS[key]
    shots = _best_faces([f for f in faces(v["start"], v["end"] + 0.001, v["cam"], state) if _same(f["name"], name)])
    if v.get("id") is not None:
        _OLD_SHOTS[key] = shots
    return shots


def _visit_shots(name: str, state: State) -> list[dict[str, Any]]:
    """`name`'s photos over the last SHOT_DAYS, oldest first: each visit's best faces (its `shots`, chosen by the face
    worker; for a visit from before visits had them, the best of its faces, read for its own stretch). A few per visit
    rather than every face: a month of faces is millions of records, its visits a few thousand."""
    until = _now()
    out = []
    for v in _visits(until - SHOT_DAYS * 86400, until, state):
        if not _same(v["name"], name):
            continue
        shots = v.get("shots") or _old_visit_shots(v, name, state)
        out += [
            {
                "id": s["detection"],
                "ts": s["ts"],
                "cam": v["cam"],
                "bbox_norm": s["bbox"],
                "quality": s.get("quality"),
                "name": v["name"],
                "sim": s.get("sim", v.get("best_sim")),
            }
            for s in shots
            if s.get("bbox")
        ]
    return sorted(out, key=lambda f: f["ts"])


def shots(name: str, before: float | None = None, limit: int = 60, state: State = None) -> dict[str, Any]:
    """`name`'s photos (case-insensitive), newest first: each visit's best face (_visit_shots), at most one per camera
    per second. One page of `limit` older than `before`: {total, shots: [{id, ts, cam, sim}],
    next}; pass `next` as `before` for the page after, None at the end. Pages break on whole seconds."""
    best_of: dict[tuple[str, int], dict[str, Any]] = {}
    for f in _visit_shots(name, state):
        key = (str(f["cam"]), int(f["ts"]))
        if key not in best_of or (f["sim"] or 0.0) > (best_of[key]["sim"] or 0.0):
            best_of[key] = f
    thinned = sorted(best_of.values(), key=lambda f: (-int(f["ts"]), f["cam"]))
    before = _now() + 1 if before is None else float(before)
    page = [f for f in thinned if f["ts"] < before][: int(limit)]
    return {
        "total": len(thinned),
        "shots": [{"id": f["id"], "ts": f["ts"], "cam": f["cam"], "sim": f["sim"]} for f in page],
        "next": int(page[-1]["ts"]) if len(page) == int(limit) else None,
    }


def shot(sid: str, state: State = None, around: float | None = None) -> dict[str, Any] | None:
    """One sighting for cropping: {id, ts, cam, bbox_norm}, or None. `around` (a time near it, its ts when known)
    narrows the look; without it, the time in its id does."""
    f = _found(sid, state, around)
    return None if f is None else {"id": f["id"], "ts": f["ts"], "cam": f["cam"], "bbox_norm": f["bbox_norm"]}


def _score(f: Mapping[str, Any]) -> float:
    return round(float(f.get("quality") or 0.0), 4)


_signal_norms: dict[str, tuple[float, dict[str, float]]] = {}
_crop_norms: dict[str, tuple[float, dict[str, float]]] = {}


def _state_dir(state: State) -> pathlib.Path:
    from liberational import config as cfg

    return pathlib.Path(state) if state is not None else cfg.path("paths.state")


def norms(state: State = None) -> dict[str, float]:
    """How much each shot reads as a face, by its id: AdaFace's raw norm, 0 for one that doesn't (a back of
    the head, a blur). A look with a face.signal has its own (0 when it fails see.faces.readable); an older shot has
    the norm tools/score_shot_crops.py found on its saved crop, 0 when no face was found there; a shot with neither
    has none. Signals are reread every NORMS_TTL_S, and the crop scores whenever their file changes."""
    key = str(_state_dir(state))
    now = _now()
    held = _signal_norms.get(key)
    if held is None or now - held[0] >= NORMS_TTL_S:
        live: dict[str, float] = {}
        for r in segment_log.at(state).range(segment_log.FACE_SIGNALS, now - SHOT_DAYS * 86400, now + 1):
            norm, head = r.get("norm"), r.get("head") or [0.0]
            live[str(r["match_id"])] = float(norm or 0.0) if readable(norm, float(head[0])) else 0.0
        held = _signal_norms[key] = (now, live)
    path = _state_dir(state) / SHOT_NORMS
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        mtime = -1.0
    crop = _crop_norms.get(key)
    if crop is None or crop[0] != mtime:
        try:
            read = {str(k): float(v) for k, v in json.loads(path.read_text()).items()} if mtime >= 0 else {}
        except OSError, ValueError:
            read = {}
        crop = _crop_norms[key] = (mtime, read)
    return {**crop[1], **held[1]}


def _order(r: Mapping[str, Any]) -> tuple[bool, float, float, float, str]:
    """Shots with a norm first, most face-like first; then those with none, by quality; newest first on a tie."""
    return (r["norm"] is None, -(r["norm"] or 0.0), -r["score"], -r["ts"], r["id"])


def ranked(
    name: str, state: State = None, cam: str | None = None, start: float | None = None, end: float | None = None
) -> list[dict[str, Any]]:
    """`name`'s shots (case-insensitive), best first, optionally on one camera between start and end:
    [{id, ts, cam, score, norm}]. Best is the most face-like by `norms`, then the shots with no norm by quality (score).
    Shots that score 0 (a profile, too small) or whose norm is 0 (a back of the head) are left out."""
    whole = cam is None and start is None and end is None
    found = _visit_shots(name, state) if whole else [f for f in _named(name, state, start, end) if f["cam"] == cam]
    by_id = norms(state)
    rows = [
        {"id": f["id"], "ts": f["ts"], "cam": f["cam"], "score": _score(f), "norm": by_id.get(str(f["id"]))}
        for f in found
        if _score(f) > 0
    ]
    return sorted((r for r in rows if r["norm"] is None or r["norm"] > 0), key=_order)


def best(
    name: str,
    state: State = None,
    cam: str | None = None,
    start: float | None = None,
    end: float | None = None,
    skip: Collection[str] = (),
) -> dict[str, Any] | None:
    """The best-scoring shot of `name`, leaving out the ids in `skip` (hidden from their card), or None."""
    return next((r for r in ranked(name, state, cam, start, end) if r["id"] not in skip), None)


def ranked_shot(name: str, sid: str, state: State = None) -> dict[str, Any] | None:
    """Sighting sid as _ranked gives it, when it is still `name`'s and scores above 0 (a main photo they chose), else
    None. Looked up by itself (by the time in its id), so any of their faces can be their main photo."""
    f = _found(sid, state)
    if f is None or not _same(f["name"], name) or not f["bbox_norm"] or _score(f) <= 0:
        return None
    return {"id": f["id"], "ts": f["ts"], "cam": f["cam"], "score": _score(f)}


def best_spread(
    name: str, limit: int = 6, state: State = None, skip: Collection[str] = (), *, ready: Collection[str] | None = None
) -> list[dict[str, Any]]:
    """Up to `limit` of the best shots of `name`, spread over different days: among their SPREAD_POOL best, the best
    of each local day first (best days first), then, when there are fewer days than `limit`, the next best shots at
    least SPREAD_GAP_S from every one already picked on the same camera. Best first. Ids in `skip` are left out."""
    pool = list(
        itertools.islice(
            (r for r in ranked(name, state) if r["id"] not in skip and (ready is None or r["id"] in ready)), SPREAD_POOL
        )
    )
    picked: list[dict[str, Any]] = []
    days: set[str] = set()
    for r in pool:
        day = dt.datetime.fromtimestamp(r["ts"]).date().isoformat()
        if day not in days:
            days.add(day)
            picked.append(r)
    picked = picked[:limit]
    for r in pool:
        if len(picked) >= limit:
            break
        if all(p["id"] != r["id"] and (p["cam"] != r["cam"] or abs(p["ts"] - r["ts"]) >= SPREAD_GAP_S) for p in picked):
            picked.append(r)
    return sorted(picked, key=_order)


def unnamed_clips(
    gap: float, limit: int = 12, state: State = None, *, rows: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """Recent stretches of faces nobody is named for, newest first: over the last SHOT_DAYS, each camera's run of
    faces with gaps up to `gap`, kept when every face in it is unnamed, then split by face group (the face worker's
    "Face N"), so two strangers at the door are two clips. [{id: "<cam>@<start .2f>" (with "#2", "#3" for a face that
    starts in the same frame), cam, start, end, ids, thumb}], thumb the id of its best shot by quality, None when every
    one scores 0. `rows` may supply the complete SHOT_DAYS projection, including named faces; None reads it,
    while [] is known empty."""
    now = _now()
    by_cam: dict[str, list[dict[str, Any]]] = {}
    for f in faces(now - SHOT_DAYS * 86400, now + 1, state=state) if rows is None else rows:
        if f["bbox_norm"]:
            by_cam.setdefault(str(f["cam"]), []).append(f)
    clips: list[dict[str, Any]] = []
    for cam, seen in by_cam.items():
        for group in interval_groups(seen, gap, start=lambda f: f["ts"], end=lambda f: f["ts"]):
            if any(f["name"] for f in group):
                continue
            groups: dict[str, list[dict[str, Any]]] = {}
            for f in group:
                groups.setdefault(str(f["cluster"] or f["track_id"] or f["id"]), []).append(f)
            seen_ids: set[str] = set()
            for face in groups.values():
                top = max(face, key=_score)
                cid, k = f"{cam}@{face[0]['ts']:.2f}", 1
                while cid in seen_ids:
                    k += 1
                    cid = f"{cam}@{face[0]['ts']:.2f}#{k}"
                seen_ids.add(cid)
                clips.append(
                    {
                        "id": cid,
                        "cam": cam,
                        "start": face[0]["ts"],
                        "end": face[-1]["ts"],
                        "ids": [f["id"] for f in face],
                        "thumb": top["id"] if _score(top) > 0 else None,
                    }
                )
    clips.sort(key=lambda k: -k["end"])
    return clips[:limit]


def status(state: State = None) -> dict[str, int]:
    """Counts for logs and the dashboard, over the last day: faces, named faces, and visits."""
    now = _now()
    found = faces(now - 86400, now + 1, state=state)
    return {
        "sightings": len(found),
        "named": sum(1 for f in found if f["name"]),
        "visits": len(_visits(now - 86400, now, state)),
    }


# ---- voices ----
def hand_labeled(e: dict[str, Any]) -> bool:
    """Lines a person labeled, as people/transcript_view.py reads them: "unlabeled", multi_speaker, or a speaker with
    no similarity (an assignment). The voice pass never touches them."""
    return bool(e.get("unlabeled") or e.get("multi_speaker")) or (
        e.get("speaker_sim") is None and e.get("speaker") not in (None, "", "unknown")
    )


def candidate(e: dict[str, Any]) -> bool:
    """Lines worth re-scoring: unknown or an auto cluster ("Speaker N"), never hand-labeled."""
    if hand_labeled(e):
        return False
    lab = (e.get("speaker") or "unknown").strip()
    return e.get("speaker_id") is None or lab.casefold() == "unknown" or bool(AUTO.fullmatch(lab))
