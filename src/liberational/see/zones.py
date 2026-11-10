"""Named zones on a camera's view (#1316): the litter box, the bathroom door, the food bowl.

The owner draws a zone as a polygon on one camera's picture, in normalized frame coordinates (0 to 1, x right, y
down). A camera's zones are saved together as one `zone.defined` record in the knowledge store (taught state, kept
across a catalog reset: #1755), the camera's whole current set, so the newest record per camera is what holds now and
the older ones are the audit trail: who changed what, and when. A zone keeps its `id` through renames and reshapes,
so its visits count as the same zone across edits.

Saving takes the id of the set it edited (`base`); a save over a set someone else changed since is refused, so two
open editors can't silently overwrite each other.
"""

import copy
import math
import secrets
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from liberational import knowledge
from liberational.vision.crossings import anchor

MAX_ZONES = 16  # per camera
MAX_POINTS = 32  # per polygon
MAX_NAME = 60
MIN_AREA = 1e-4  # a zone covers at least this share of the frame (a 1% by 1% square)
COLLECTION = "zone.defined"  # in the knowledge store, keyed by camera; the record's id is the version's

Point = tuple[float, float]


@dataclass(frozen=True)
class Zone:
    """One named polygon on a camera's view. A zone of kind "screen" marks a TV or monitor the detector misses (#2468):
    nobody is counted in it, and the people it shows are kept out of everything (see/screens.py)."""

    id: str
    name: str
    polygon: tuple[Point, ...]
    kind: str = "area"

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "name": self.name, "polygon": [list(p) for p in self.polygon]}
        if self.kind != "area":
            out["kind"] = self.kind
        return out


ZONE_KINDS = ("area", "screen")


def screen_polygons(definition: Definition | None) -> list[tuple[Point, ...]]:
    """A camera's zones of kind "screen" (#2468)."""
    return [z.polygon for z in definition.zones if z.kind == "screen"] if definition is not None else []


def counted(zones: Sequence[Zone]) -> list[Zone]:
    """The zones people are counted in: all but screens."""
    return [z for z in zones if z.kind != "screen"]


def area(polygon: Sequence[Point]) -> float:
    """The polygon's area (shoelace), as a share of the frame."""
    n = len(polygon)
    twice = sum(polygon[i][0] * polygon[(i + 1) % n][1] - polygon[(i + 1) % n][0] * polygon[i][1] for i in range(n))
    return abs(twice) / 2


def _cross(o: Point, a: Point, b: Point) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _segments_cross(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    """Proper crossing of two segments (touching at an end doesn't count)."""
    d1, d2 = _cross(q1, q2, p1), _cross(q1, q2, p2)
    d3, d4 = _cross(p1, p2, q1), _cross(p1, p2, q2)
    return d1 * d2 < 0 and d3 * d4 < 0


def simple(polygon: Sequence[Point]) -> bool:
    """True when no two edges that don't share a corner cross (no figure-eight)."""
    n = len(polygon)
    edges = [(polygon[i], polygon[(i + 1) % n]) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue  # neighbours share a corner
            if _segments_cross(*edges[i], *edges[j]):
                return False
    return True


def contains(polygon: Sequence[Point], x: float, y: float) -> bool:
    """True when (x, y) is inside the polygon (even-odd ray casting; a point on an edge may land either way)."""
    inside = False
    n = len(polygon)
    for i in range(n):
        (x1, y1), (x2, y2) = polygon[i], polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def new_id() -> str:
    return "zone-" + secrets.token_hex(6)


def _point(value: object) -> Point:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError("each corner is [x, y]")
    x, y = value
    if any(isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v) for v in (x, y)):
        raise ValueError("corners are numbers")
    if not (0 <= x <= 1 and 0 <= y <= 1):
        raise ValueError("corners are shares of the frame, 0 to 1")
    return (round(float(x), 5), round(float(y), 5))


def parse_zone(value: object) -> Zone:
    """A zone from a request or a stored record, checked: a name, and a simple polygon of 3 to MAX_POINTS corners
    covering at least MIN_AREA. A zone without an id gets a new one."""
    if not isinstance(value, Mapping):
        raise ValueError("each zone is an object")
    name = value.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("each zone needs a name")
    name = " ".join(name.split())
    if len(name) > MAX_NAME:
        raise ValueError(f"zone names are at most {MAX_NAME} characters")
    corners = value.get("polygon")
    if not isinstance(corners, list) or not 3 <= len(corners) <= MAX_POINTS:
        raise ValueError(f"a zone has 3 to {MAX_POINTS} corners")
    polygon = tuple(_point(c) for c in corners)
    if not simple(polygon):  # first: a figure-eight's halves cancel in the area
        raise ValueError(f"zone {name!r} crosses itself")
    if area(polygon) < MIN_AREA:
        raise ValueError(f"zone {name!r} is too small to see anything in")
    ident = value.get("id")
    if ident is None:
        ident = new_id()
    elif not isinstance(ident, str) or not ident.startswith("zone-") or not 6 <= len(ident) <= 40:
        raise ValueError("a zone id is one this camera gave out")
    kind = value.get("kind", "area")
    if kind not in ZONE_KINDS:
        raise ValueError(f"a zone's kind is one of {', '.join(ZONE_KINDS)}")
    return Zone(ident, name, polygon, kind)


def parse_zones(values: object) -> tuple[Zone, ...]:
    """A camera's whole set: at most MAX_ZONES, names and ids unique (names ignoring case)."""
    if not isinstance(values, list):
        raise ValueError("zones is a list")
    if len(values) > MAX_ZONES:
        raise ValueError(f"a camera has at most {MAX_ZONES} zones")
    zones = tuple(parse_zone(v) for v in values)
    if len({z.name.casefold() for z in zones}) != len(zones):
        raise ValueError("two zones on one camera can't share a name")
    if len({z.id for z in zones}) != len(zones):
        raise ValueError("two zones share an id")
    return zones


@dataclass(frozen=True)
class Definition:
    """A camera's zone set as one `zone.defined` record saved it."""

    id: str  # the record's id
    cam: str
    zones: tuple[Zone, ...]
    at: float

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "cam": self.cam, "zones": [z.as_dict() for z in self.zones], "at": self.at}


def _store(state: Path) -> knowledge.Store:
    from liberational.people import registry

    return knowledge.at(registry.knowledge_path(state))


def _definition(record: knowledge.Record) -> Definition | None:
    try:
        cam = str(record.value["cam"])
        return Definition(record.id, cam, parse_zones(record.value.get("zones")), record.start_ns / 1e9)
    except KeyError, ValueError:
        return None  # a record this version can't read holds no zones


def history(state: Path, cam: str | None = None) -> list[Definition]:
    """Every saved zone set (on `cam` when given), oldest first: the audit trail (a few records per camera). A state
    with no knowledge store has none, and reading doesn't create one."""
    from liberational.people import registry

    if not registry.knowledge_path(state).exists():
        return []
    rows = _store(state).query(COLLECTION, key=cam)
    found = [d for r in rows if (d := _definition(r.record)) is not None and (cam is None or d.cam == cam)]
    return sorted(found, key=lambda d: d.at)


def current(state: Path) -> dict[str, Definition]:
    """Each camera's newest zone set (a camera whose newest set is empty has no zones)."""
    latest: dict[str, Definition] = {}
    for d in history(state):
        latest[d.cam] = d
    return latest


class StaleBase(Exception):
    """The set being saved was edited from an older version than the camera's current one."""


def define(state: Path, cam: str, zones: Sequence[Zone], base: str | None, now: float | None = None) -> Definition:
    """Save `cam`'s whole zone set as a new version, edited from version `base` (None for a camera with none yet)."""
    if not cam or len(cam) > 64:
        raise ValueError("a camera name is required")
    at = time.time() if now is None else now
    was = current(state).get(cam)
    if (was.id if was else None) != base:
        raise StaleBase("these zones changed since you opened them; reload and try again")
    if was is not None and at <= was.at:
        at = was.at + 0.001  # versions stay in order even with a clock step back
    value = {"cam": cam, "zones": [z.as_dict() for z in zones], "source": "owner"}
    at_ns = round(at * 1e9)
    _store(state).put([knowledge.Record(str(uuid.uuid7()), COLLECTION, cam, at_ns, at_ns, time.time_ns(), value)])
    saved = current(state).get(cam)
    if saved is None:
        raise RuntimeError("the zones were saved but can't be read back")
    return saved


# Visits: who was in which zone, and for how long (#1316). Pure bookkeeping over a camera's detection frames, like
# see/tracks.py: no broker, clock, or model here; see/zones_worker.py feeds it and writes what it returns.
#
# A subject is a tracked box: a person from see-actions' `pose.track` (#1825), or an animal (#1315), keyed by
# its track id (opaque, camera- and run-local, so never an identity). Its position is the bottom centre of its box
# (feet or paws, crossings.anchor), so a cat in the litter box is in it while its head is over the edge.
#
# - A visit starts at the first frame a subject's position is inside a zone. `entry_seen` says whether that subject
#   was seen outside the zone first; a subject first seen already inside may have been there before anyone looked.
# - Seen outside for EXIT_S, it has left: the visit ends at the last frame it was inside (`ended: "left"`).
# - Not detected at all (hidden by the zone itself, like a hooded litter box, or by someone in front), the visit
#   stays open for ABSENT_S; seen inside again, it continues, and seen outside, it ended at its last inside frame.
#   Never seen again, the visit ends at its last inside frame with an unknown exit (`ended: "lost"`).
# - Dwell is the visit's span from first to last inside frame; time hidden inside counts, since the subject was not
#   seen leaving. A count needing a minimum dwell (memory/zone_counts.py) reads `dwell_s` with `ended`.
# - Coverage is what lets a count say "none" rather than "unknown": stretches of evaluated frames, per camera, kind
#   of subject, and zone set. A frame whose detector failed, or with an untracked box inside a zone (a subject
#   nobody can follow), is not coverage. More than MAX_GAP_S without coverage ends every open visit
#   (`ended: "censored"`) and the stretch. A coverage record is only published up to the start of the earliest open
#   visit, so a reader never sees coverage over a visit that hasn't been written yet.
# - Saving a camera's zones ends its open visits (`ended: "zones_changed"`) and its coverage stretch; both kinds of
#   record name the `zone.defined` version they were read against.
#
# Visits are unconfirmed candidates: track ids split and swap, and a hidden subject may have walked off unseen.

VISIT_VERSION = "zone-visits-1"
EXIT_S = 1.5  # seen outside this long: left (one or two jittery frames on the edge don't end a visit)
ABSENT_S = 120.0  # not detected this long: lost (a cat in a hooded box for a couple of minutes is still in it)
MAX_GAP_S = 10.0  # no coverage this long: the camera was off or the detector stopped
COVERAGE_S = 60.0  # a coverage stretch is published at most this long after it started (when no visit holds it)
MAX_TRACKS = 128  # subjects remembered per camera and kind
MAX_VISITS = 256  # open visits per camera and kind
MAX_PARENT_FRAMES = 30  # frames kept as a record's parents (the first, the last, and a spread between)
KINDS = ("person", "animal")


@dataclass(frozen=True)
class Output:
    """A derived record and the source payloads behind it, for the worker to write."""

    kind: str  # "visit" or "coverage"
    record: dict[str, Any]
    parents: tuple[str, ...]


def _sample(frames: list[str], new: str) -> list[str]:
    """Bounded frame parents: keep the first, the newest, and every other one as the list fills."""
    if frames and frames[-1] == new:
        return frames
    frames = [*frames, new]
    if len(frames) > MAX_PARENT_FRAMES:
        frames = [frames[0], *frames[1:-1:2], frames[-1]]
    return frames


class ZonePipeline:
    """One camera's visits and coverage, for both kinds of subject. `snapshot()` and `state=` round-trip it."""

    def __init__(self, cam: str, state: Mapping[str, Any] | None = None) -> None:
        self.cam = cam
        self.definition: Definition | None = None
        # Per kind: the open coverage stretch, open visits, and the subjects seen recently.
        self.kinds: dict[str, dict[str, Any]] = {kind: self._fresh() for kind in KINDS}
        if state is not None:
            self._restore(state)

    @staticmethod
    def _fresh() -> dict[str, Any]:
        # stretch: {start, last, frames, parents} of evaluated frames; visits: "track|zone" -> visit;
        # tracks: track -> {seen (last ts), outside: [zone ids it was last seen outside of]}
        return {"stretch": None, "visits": {}, "tracks": {}}

    # Snapshots, for the worker's checkpoints.

    def snapshot(self) -> dict[str, Any]:
        return {
            "definition": self.definition.as_dict() if self.definition else None,
            "kinds": copy.deepcopy(self.kinds),
        }

    def _restore(self, state: Mapping[str, Any]) -> None:
        d = state.get("definition")
        if d is not None:
            if not isinstance(d, Mapping) or d.get("cam") != self.cam:
                raise ValueError("zone snapshot has another camera's zones")
            self.definition = Definition(str(d["id"]), self.cam, parse_zones(d.get("zones")), float(d["at"]))
        kinds = state.get("kinds")
        if not isinstance(kinds, Mapping) or set(kinds) != set(KINDS):
            raise ValueError("zone snapshot kinds are malformed")
        for kind in KINDS:
            k = kinds[kind]
            if (
                not isinstance(k, Mapping)
                or not isinstance(k.get("visits"), dict)
                or not isinstance(k.get("tracks"), dict)
            ):
                raise ValueError("zone snapshot state is malformed")
            if len(k["visits"]) > MAX_VISITS or len(k["tracks"]) > MAX_TRACKS:
                raise ValueError("zone snapshot bound exceeded")
        self.kinds = copy.deepcopy(dict(kinds))

    # Records.

    def _visit_output(self, kind: str, v: Mapping[str, Any], ended: str) -> Output:
        assert self.definition is not None
        exit_seen = ended == "left"
        record = {
            "cam": self.cam,
            "zone": v["zone"],
            "zone_name": v["zone_name"],
            "definition": self.definition.id,
            "subject_kind": kind,
            "track_id": v["track"],
            "label": v["label"],
            "dwell_s": round(v["last_inside"] - v["entered"], 3),
            "frames": v["frames"],
            "entry_seen": v["entry_seen"],
            "exit_seen": exit_seen,
            "ended": ended,
            "candidate": True,
            "interpretation": {"producer": "zone-visits", "version": VISIT_VERSION, "anchor": "box_bottom_centre"},
            "ts_start": v["entered"],
            "ts_end": v["last_inside"],
        }
        return Output("visit", record, (self.definition.id, *v["parents"]))

    def _coverage_output(self, kind: str, start: float, end: float, stretch: Mapping[str, Any]) -> Output:
        assert self.definition is not None
        record = {
            "cam": self.cam,
            "subject_kind": kind,
            "definition": self.definition.id,
            "zones": [z.id for z in counted(self.definition.zones)],
            "frames": stretch["frames"],
            "ts_start": start,
            "ts_end": end,
        }
        return Output("coverage", record, (self.definition.id, *stretch["parents"]))

    def _close_all(self, kind: str, ended: str) -> list[Output]:
        """End every open visit and the coverage stretch (a gap, a zone change, or a reset)."""
        k = self.kinds[kind]
        out = [self._visit_output(kind, v, ended) for v in sorted(k["visits"].values(), key=lambda v: v["entered"])]
        stretch = k["stretch"]
        if stretch and stretch["last"] > stretch["start"]:
            # Half-open [start, end): the stretch ends just after its last frame, so that frame is inside it.
            out.append(self._coverage_output(kind, stretch["start"], stretch["last"] + 1e-3, stretch))
        self.kinds[kind] = self._fresh()
        return out

    def _flush_coverage(self, kind: str, now: float) -> list[Output]:
        """Publish the stretch up to the earliest open visit, once it's COVERAGE_S long."""
        k = self.kinds[kind]
        stretch = k["stretch"]
        if not stretch or now - stretch["start"] < COVERAGE_S:
            return []
        frontier = min([stretch["last"] + 1e-3, *(v["entered"] for v in k["visits"].values())])
        if frontier - stretch["start"] < COVERAGE_S:
            return []
        out = [self._coverage_output(kind, stretch["start"], frontier, stretch)]
        k["stretch"] = {"start": frontier, "last": stretch["last"], "frames": 0, "parents": stretch["parents"][-1:]}
        return out

    # Feeding.

    def define(self, definition: Definition) -> list[Output]:
        """A new zone set for this camera: what was open under the old one ends."""
        out: list[Output] = []
        if self.definition is not None:
            for kind in KINDS:
                out.extend(self._close_all(kind, "zones_changed"))
        self.definition = definition
        return out

    def frame(self, kind: str, frame: Mapping[str, Any], labels: Sequence[str | None]) -> list[Output]:
        """One detection frame for `kind` subjects: `id`, `ts_start`, `boxes`, `track_ids`, and whether it was
        evaluated (`ok`). `labels` names each box's subject ("person", "cat")."""
        if kind not in KINDS:
            raise ValueError(f"unknown subject kind {kind!r}")
        if self.definition is None or not counted(self.definition.zones):
            return []
        k = self.kinds[kind]
        ts, fid = float(frame["ts_start"]), str(frame["id"])
        out: list[Output] = []
        stretch = k["stretch"]
        if stretch and ts - stretch["last"] > MAX_GAP_S:
            out.extend(self._close_all(kind, "censored"))
            k, stretch = self.kinds[kind], None  # _close_all put fresh state in place
        if stretch and ts <= stretch["last"]:
            return out  # an older or repeated frame: the stretch already went past it
        boxes, tracks = frame.get("boxes") or [], frame.get("track_ids")
        if not frame.get("ok", True):
            return out
        if tracks is None or len(tracks) != len(boxes):
            tracks = [None] * len(boxes)
        zones = counted(self.definition.zones)
        positions = [anchor(b) for b in boxes]
        if any(
            t is None and any(contains(z.polygon, *p) for z in zones) for t, p in zip(tracks, positions, strict=True)
        ):
            return out  # someone untracked is in a zone: this frame can't say who was where
        # The frame is coverage.
        if stretch is None:
            fresh: dict[str, Any] = {"start": ts, "last": ts, "frames": 0, "parents": []}
            stretch = k["stretch"] = fresh
        stretch["last"], stretch["frames"] = ts, stretch["frames"] + 1
        stretch["parents"] = _sample(stretch["parents"], fid)
        seen: dict[str, tuple[tuple[float, float], str | None]] = {}
        for track, position, label in zip(tracks, positions, labels, strict=False):
            if track is not None:
                seen[str(track)] = (position, label)
        for track, (position, label) in seen.items():
            remembered = k["tracks"].get(track) or {"outside": []}
            outside = set(remembered["outside"])
            for z in zones:
                key = f"{track}|{z.id}"
                visit = k["visits"].get(key)
                if contains(z.polygon, *position):
                    if visit is None:
                        if len(k["visits"]) >= MAX_VISITS:
                            out.extend(self._close_all(kind, "censored"))
                            return out
                        opened: dict[str, Any] = {
                            "track": track,
                            "zone": z.id,
                            "zone_name": z.name,
                            "label": label,
                            "entered": ts,
                            "last_inside": ts,
                            "frames": 0,
                            "entry_seen": z.id in outside,
                            "outside_since": None,
                            "parents": [],
                        }
                        visit = k["visits"][key] = opened
                    visit["last_inside"], visit["frames"] = ts, visit["frames"] + 1
                    visit["outside_since"] = None
                    visit["parents"] = _sample(visit["parents"], fid)
                    outside.discard(z.id)
                else:
                    outside.add(z.id)
                    if visit is not None:
                        if visit["outside_since"] is None:
                            visit["outside_since"] = ts
                            visit["parents"] = _sample(visit["parents"], fid)
                        if (
                            ts - visit["outside_since"] >= EXIT_S
                            or visit["outside_since"] - visit["last_inside"] > EXIT_S
                        ):
                            out.append(self._visit_output(kind, k["visits"].pop(key), "left"))
            k["tracks"][track] = {"seen": ts, "outside": sorted(outside)}
        # Subjects not in this frame: a visit stays open while they're hidden, up to ABSENT_S.
        for key in [key for key, v in k["visits"].items() if v["track"] not in seen]:
            if ts - k["visits"][key]["last_inside"] > ABSENT_S:
                out.append(self._visit_output(kind, k["visits"].pop(key), "lost"))
        stale = [t for t, r in k["tracks"].items() if ts - r["seen"] > ABSENT_S]
        for t in stale:
            del k["tracks"][t]
        while len(k["tracks"]) > MAX_TRACKS:
            del k["tracks"][min(k["tracks"], key=lambda t: k["tracks"][t]["seen"])]
        out.extend(self._flush_coverage(kind, ts))
        return out
