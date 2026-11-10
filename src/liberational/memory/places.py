"""Places and visits (part of spatial memory): the data model under maps and floorplans.

A **place** is an indoor place, such as a home or an office. The machine makes one from a phone walkthrough
(`pack-walkthrough`) or from fixed cameras that see the same space (`pack-spatial-cameras`). A
**visit** is a span of time the rig spent in one place, derived from the place signals (the Wi-Fi access points it
hears, GPS at the door, visual relocalization against the place's keyframes).

Two layers, following docs/DOMAIN-MODEL.md:

- **What the machine made is an interpretation.** A place's machine view is a chain of revisions under
  `state/places/<place id>/`: each revision is a `<revision id>.json` written once (its producer, the visits it was
  derived from, the signals that identify the place, which map and plan files it uses, the map's floor plane, the
  map-to-plan transform, and the fixed cameras registered in that frame), plus the files it adds: `<revision id>.map`
  (the point cloud, in the format its producer names; the current writers use binary PLY) and `<revision id>.plan`
  (floorplan DSL text, a few hundred bytes). A re-derivation writes a new revision and never edits an old one; a
  revision that doesn't change the map or the plan points at the earlier file. Plans are kept forever (they are
  small). Maps are
  replaced, not accumulated: the newest `MAPS_KEPT` map files stay, and so does each still-active producer's newest
  map, counting only a map that revision wrote itself. A producer is still active when it wrote one of the last
  `PRODUCER_WINDOW` revisions (tier 1 fuses each new revision into its own last map). Older maps are pruned, and the
  revision records stay as the audit trail. Visits are rows in the `place.visit` segment stream (`segment_log.py`).
- **A revision's splat** is a 3D Gaussian splat trained on that revision's map, cameras, and their recorded
  frames (spatial/splats.py), in the map's frame: `<revision id>.gs.ply` (the standard 3DGS PLY, the master) and
  `<revision id>.spz` (the compact copy the 3D view loads), with `<revision id>.gs.from` saying what it was trained
  from ("walk" or "cameras"). It's written once, after its revision, and isn't a
  revision of its own (a revision is a reconstruction; a splat is a rendering of one). The newest `SPLATS_KEPT`
  revisions' splats stay and older ones are pruned. A splat fits every revision of its map's `frame` (`Revision.frame`), so the 3D view can show
  an earlier revision's splat while the frame holds.
- **What the owner taught is in the knowledge store** (`knowledge.sqlite3`), and supersedes the machine view
  without deleting it, the way zone sets and identity corrections do. `place.named` holds the owner's name for a
  place (newest per place wins; an empty name goes back to the placeholder). `place.plan` holds the owner's
  corrected plan (newest per place wins over every machine plan until the owner withdraws it with an empty plan;
  `Place.machine_newer` says when a machine plan came after it). Every correction stays in the store (`plan_history`),
  so an earlier one can be put back (`restore_plan`). An owner may also bring a plan of their own, registered into a
  revision's frame (spatial/plan_import.py); a revision with no plan file draws in its floor's frame
  (`plan_frame`), shared by the revisions of the same map (`same_plan_frame`).

**Merged places**: two places that turn out to be one home (a whole-home walkthrough passes through both) are
merged by tier 1 (`fuse.join`): one place's newest revision brings in the other's map and cameras, and the other is
retired (`retire`): its last revision names the place it went into (`merged_into`), so no reader lists it again. The
other's owner plan correction is carried into the joined place's plan (`correct_plan` with source "merge"), moved into
its frame; the joined place's own correction holds where both have rooms.

Until the owner names a place it reads as a placeholder, "Place 1", "Place 2", ..., numbered by when each place was
first made. A naming record keeps a copy of the place's signals, so a place detector can recognize a named place
again after its machine record is gone.

Storage across a reset: `state/places/` and the segment store are machine-made, so a catalog reset deletes them
(`reset_catalog.targets`); the owner's names and plan corrections are in the knowledge store and stay. Keyframes
live with the video and follow the 30-day video window.

This module is the schema and the read and write helpers. Its writers are `pack-walkthrough` (spatial/walkthrough.py:
a phone video's cloud, floor, and plan), tier 1's fusion after each (spatial/fuse.py: the place's fused
map), and `pack-spatial-cameras`
(spatial/cameras.py), which makes a place of each group of fixed cameras that see the same space and registers
each camera in it: its pose in the map's frame, with the map's floor plane and the plan's frame, so `camera_to_plan`
can put a pixel of that camera on the plan.

**Places in the journal**: `places_for_span` is the one call the journal's pages make for a
span: the places visited in it and the places a registered fixed camera only observed, each with the rooms seen
there. The pages' Places line keeps the visited ones (`PlaceRef.visited`). Rooms are the owner's drawn area zones
(`see/zones.py`) that a person was in, until a floorplan and a pose track in the place's frame can say which plan
room the rig was in. A zone visit on a fixed camera registered to a place (a revision's `cameras`, written by
spatial/cameras.py) is filed under that camera's place, wherever the rig was; any other is filed under the
place the rig was in at the time. Rooms seen with no place known come back under a ref with no place, so a day
without place visits still lists its rooms.
"""

from __future__ import annotations

import contextlib
import functools
import json
import math
import os
import re
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from liberational import knowledge, locking
from liberational import segment_log as S

PLACES_DIR = "places"  # under paths.state
NAMED = "place.named"  # owner names, in the knowledge store, keyed by place id
PLAN = "place.plan"  # owner plan corrections, in the knowledge store, keyed by place id
MAPS_KEPT = 2  # map files kept per place, newest first; older ones are pruned
SPLATS_KEPT = 2  # revisions per place whose splat files are kept, newest first
#: A splat's files beside its revision: the 3DGS master, the compact web copy, and a note of what it was trained from.
SPLAT_FILES = ("gs.ply", "spz", "gs.from")
# A producer's newest map is kept whatever its age only while that producer wrote one of the place's last this many
# revisions: a retired producer version's last map then goes too (about ten days of daily runs). Tier 1 is no
# exception: 20 fixed-camera snapshots in a row that don't fuse (about 20 days, as an unfused day writes only its
# snapshot) prune the fused map, and the next fusion founds the place's map afresh in a new frame, without the fused
# cameras' registrations. After weeks of not registering, starting again is the better bet.
PRODUCER_WINDOW = 20
MERGED = "places merge v1"  # the producer of a merged place's last revision (`retire`)
MAX_NAME = 80
PLACEHOLDER = "Place {n}"
_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")

Signals = dict[str, list[str]]  # what identifies a place, by kind: {"wifi": [BSSIDs], "camera": [names], ...}


@dataclass(frozen=True)
class Background:
    """What a registered camera sees with nobody in the way, cell by cell of its frame: each cell's surface's
    height above the floor, as a fraction of the camera's own height above it (so the map's levelling and metric scale
    leave it as it is), row by row from the top left, None where the camera's point map had nothing sure. A cell of
    the floor is about 0; one of a couch's back, for a camera 1.5 m up, about 0.6. Cells are looked up by fractions of
    the camera's own frames, as foot points are given (a manifest lens is already allowed for)."""

    cols: int
    rows: int
    heights: tuple[float | None, ...]

    def at(self, u: float, v: float) -> float | None:
        """The height fraction of the cell holding (u, v), fractions of the frame; None outside it or where unknown."""
        if not (0.0 <= u < 1.0 and 0.0 <= v < 1.0):
            return None
        return self.heights[int(v * self.rows) * self.cols + int(u * self.cols)]

    def as_dict(self) -> dict[str, Any]:
        return {
            "cols": self.cols,
            "rows": self.rows,
            "heights": [None if h is None else round(h, 3) for h in self.heights],
        }

    @staticmethod
    def from_dict(value: Mapping[str, Any]) -> Background:
        cols, rows = int(value["cols"]), int(value["rows"])
        heights = value["heights"]
        if not isinstance(heights, list) or len(heights) != cols * rows:
            raise ValueError("a background is cols x rows heights")
        return Background(cols, rows, tuple(None if h is None else float(h) for h in heights))


@dataclass(frozen=True)
class CameraPose:
    """A fixed camera registered in its place's frame: where it sits and how it looks, in the map's metres."""

    pose: tuple[float, ...]  # camera to place, a 4x4 matrix row by row (16 numbers); the camera looks along its +z
    focal: float  # focal length in pixels of the frame it was registered from
    width: int  # that frame's size in pixels
    height: int
    at: float = 0.0  # when it was registered (UTC epoch seconds); a revision that carries it over keeps this time
    # A wide lens's distortion, undone in the frame the pose was registered from (spatial/lens.py's
    # `Lens.as_tuple`: fx, fy, cx, cy as fractions of the frame, then lambda); None for a pinhole. A pixel of the
    # camera's own frames is undistorted with it before it's cast as a ray (`floor_point`).
    lens: tuple[float, ...] | None = None
    # What the camera sees behind people: spatial/footprints.py's `at_edge` reads it. None for a camera
    # registered alone or by MASt3R's alignment alone.
    background: Background | None = None

    def as_dict(self) -> dict[str, Any]:
        out = {"pose": list(self.pose), "focal": self.focal, "width": self.width, "height": self.height, "at": self.at}
        if self.lens is not None:
            out["lens"] = list(self.lens)
        if self.background is not None:
            out["background"] = self.background.as_dict()
        return out

    @staticmethod
    def from_dict(value: Mapping[str, Any]) -> CameraPose:
        pose = tuple(float(x) for x in value["pose"])
        if len(pose) != 16:
            raise ValueError("a camera pose is 16 numbers")
        lens = _numbers(value.get("lens"), 5, "a camera's lens")
        background = Background.from_dict(value["background"]) if value.get("background") else None
        return CameraPose(
            pose,
            float(value["focal"]),
            int(value["width"]),
            int(value["height"]),
            float(value.get("at", 0.0)),
            lens,
            background,
        )


@dataclass(frozen=True)
class Revision:
    """One machine view of a place, as its `<id>.json` saved it."""

    id: str
    place: str
    at: float  # when it was made (UTC epoch seconds)
    producer: str  # what made it: the model or job and its version
    sources: tuple[str, ...]  # the visit ids it was derived from
    signals: Signals = field(default_factory=dict)
    map: str | None = None  # the map file it uses (in the place's directory), or None before the first map
    map_format: str | None = None  # how that file is encoded, as its producer named it
    plan: str | None = None  # the plan file it uses, or None before the first plan
    cameras: dict[str, CameraPose] = field(default_factory=dict)  # fixed cameras registered in this place's frame
    # The map's floor: (nx, ny, nz, d) with a point's height above the floor n . p + d, in metres. It goes
    # with the map: a new map without one has none.
    floor: tuple[float, ...] | None = None  # 4 numbers
    # Map to plan: (plan x, plan y, height above the floor) = this 3x4 matrix, row by row, times (x, y, z, 1): the
    # frame the plan's DSL is drawn in (spatial/floorplan.py's `Floorplan.transform`). It goes with the plan.
    plan_transform: tuple[float, ...] | None = None
    # The map's frame: the id of the revision whose map first stood in it. Tier 1 carries it over while it
    # fuses into the same map without moving it, so a splat made for one revision of that frame still fits a later
    # one's map. None for a revision written before frames were recorded.
    frame: str | None = None
    # The place this one was merged into (`retire`): set only on a merged place's last revision, which has no
    # map, plan, cameras, or signals of its own. A place merged away isn't listed (`place_ids`).
    merged_into: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "place": self.place,
            "at": self.at,
            "producer": self.producer,
            "sources": list(self.sources),
            "signals": {k: list(v) for k, v in self.signals.items()},
            "map": self.map,
            "map_format": self.map_format,
            "plan": self.plan,
            "cameras": {k: v.as_dict() for k, v in self.cameras.items()},
            "floor": None if self.floor is None else list(self.floor),
            "plan_transform": None if self.plan_transform is None else list(self.plan_transform),
            "frame": self.frame,
            **({"merged_into": self.merged_into} if self.merged_into is not None else {}),
        }

    @staticmethod
    def from_dict(value: Mapping[str, Any]) -> Revision:
        signals = value.get("signals") or {}
        if not isinstance(signals, dict):
            raise ValueError("signals must be an object")
        cameras = value.get("cameras") or {}
        if not isinstance(cameras, dict):
            raise ValueError("cameras must be an object")
        return Revision(
            id=_place_id(value["id"]),
            place=_place_id(value["place"]),
            at=float(value["at"]),
            producer=str(value["producer"]),
            sources=tuple(str(s) for s in value.get("sources") or ()),
            signals={str(k): [str(x) for x in v] for k, v in signals.items()},
            map=_file_or_none(value.get("map")),
            map_format=None if value.get("map_format") is None else str(value["map_format"]),
            plan=_file_or_none(value.get("plan")),
            cameras={str(k): CameraPose.from_dict(v) for k, v in cameras.items()},
            floor=_numbers(value.get("floor"), 4, "a floor plane"),
            plan_transform=_numbers(value.get("plan_transform"), 12, "a plan transform"),
            frame=None if value.get("frame") is None else _place_id(value["frame"]),
            merged_into=None if value.get("merged_into") is None else _place_id(value["merged_into"]),
        )


@dataclass(frozen=True)
class Visit:
    """A span of time spent in one place: a `place.visit` row."""

    id: str
    place: str
    start: float
    end: float
    producer: str
    basis: str  # what decided it: "wifi", "gps", "relocalization", ...
    slot: str  # what wrote it: the rig, or a fixed camera

    @staticmethod
    def from_record(record: Mapping[str, Any]) -> Visit:
        return Visit(
            id=str(record["id"]),
            place=str(record["place"]),
            start=float(record["ts_start"]),
            end=float(record["ts_end"]),
            producer=str(record["producer"]),
            basis=str(record["basis"]),
            slot=str(record["cam"]),
        )


@dataclass(frozen=True)
class Place:
    """A place as readers see it: its newest machine revision with the owner's name and plan over it."""

    id: str
    name: str
    named: bool  # the owner named it (otherwise `name` is the placeholder)
    revision: Revision  # the newest machine revision
    plan: str | None  # the plan that holds: the owner's correction if there is one, else the machine's
    plan_source: str | None  # "owner", "machine", or None with no plan yet
    machine_newer: bool  # a machine plan was made after the owner's correction that holds
    # The revision whose frame the holding plan is drawn in (its `plan_transform`): the newest revision for a
    # machine plan, the one current when the owner saved a correction for theirs; None with no plan. Positions are
    # drawn on the plan by reading them in this frame (`spatial.positions.positions(frame_revision=)`).
    plan_revision: str | None = None
    # The owner's plan as drawn on its page, when that isn't the plan frame (`correct_plan`'s `view`); None otherwise
    view: PageView | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "named": self.named,
            "revision": self.revision.as_dict(),
            "plan": self.plan,
            "plan_source": self.plan_source,
            "machine_newer": self.machine_newer,
            "plan_revision": self.plan_revision,
        }


def _numbers(value: object, n: int, what: str) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != n:
        raise ValueError(f"{what} is {n} numbers")
    return tuple(float(x) for x in value)


def _place_id(value: object) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"not a place or revision id: {value!r}")
    return value


def _file_or_none(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f-]{36}\.(map|plan)", value):
        raise ValueError(f"not a place file: {value!r}")
    return value


def root(state: Path) -> Path:
    """Where place revisions, maps, and plans live: `paths.state/places`."""
    return Path(state) / PLACES_DIR


def _write(path: Path, data: bytes) -> None:
    """Write `path` whole, so a reader sees all of it or none of it."""
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def writing(state: Path) -> contextlib.AbstractContextManager[None]:
    """Held while a job writes places: `pack-spatial-cameras` from its first place write through its fusion,
    and `pack-walkthrough` from its revision through its join, one at a time. A nightly run that seated a room into
    a place while a join retired it would make that place's newest revision a live one again, and a revision it wrote
    mid-join would be overwritten by the join's. The lock is `state/places.lock` (locking.py)."""
    return locking.locked(root(state))


def write_whole(path: Path, data: bytes) -> None:
    """`path` written whole (`_write`), for a machine state file beside the places."""
    _write(path, data)


def revisions(state: Path, place: str) -> list[Revision]:
    """Every machine revision of `place`, oldest first. An unreadable revision file is skipped."""
    folder = root(state) / _place_id(place)
    found: list[Revision] = []
    for path in folder.glob("*.json") if folder.is_dir() else ():
        try:
            rev = Revision.from_dict(json.loads(path.read_text()))
        except OSError, ValueError, KeyError, TypeError:
            continue  # a foreign or unreadable file holds no revision
        if rev.place == place and path.stem == rev.id:
            found.append(rev)
    return sorted(found, key=lambda r: (r.at, r.id))


def latest(state: Path, place: str) -> Revision | None:
    got = revisions(state, place)
    return got[-1] if got else None


def place_ids(state: Path) -> list[str]:
    """Every place with at least one revision, except those merged into another (`retire`)."""
    base = root(state)
    if not base.is_dir():
        return []
    return sorted(
        p.name
        for p in base.iterdir()
        if p.is_dir() and _ID.fullmatch(p.name) and (rev := latest(state, p.name)) and rev.merged_into is None
    )


def merged_into(state: Path, place: str) -> str | None:
    """The place `place` now is: itself, or the place it was merged into (following a chain of merges); None for no
    such place."""
    seen = set()
    while (rev := latest(state, place)) is not None and rev.merged_into is not None and place not in seen:
        seen.add(place)
        place = rev.merged_into
    return place if latest(state, place) is not None else None


def retire(state: Path, place: str, into: str, now: float | None = None) -> Revision:
    """Write `place`'s last revision: merged into `into`, whose newest revision has brought in its map and
    cameras. It keeps nothing of its own, so no reader lists it, no camera is registered in it, and no producer writes
    into it again; its earlier revisions stay as the record. The owner's name and plan for it stay in the knowledge
    store."""
    place, into = _place_id(place), _place_id(into)
    if place == into:
        raise ValueError("a place isn't merged into itself")
    was, target = latest(state, place), latest(state, into)
    if was is None or target is None:
        raise KeyError(f"no place {place if was is None else into}")
    at = time.time() if now is None else now
    if at <= was.at:
        at = was.at + 0.001
    rev = Revision(str(uuid.uuid7()), place, at, MERGED, (target.id,), merged_into=into)
    _write(root(state) / place / f"{rev.id}.json", json.dumps(rev.as_dict(), sort_keys=True).encode())
    return rev


def camera_places(state: Path) -> dict[str, str]:
    """Each registered fixed camera with the place it sits in: of the places whose newest revision has it, the one
    that registered it most recently (`CameraPose.at`; on a tie, such as two registrations from before `at` was
    recorded, the more recently revised place). A later revision of a place that only carries a camera over
    doesn't take it back from where it was registered since (a camera that moved). A camera that leaves every
    group (moved out of sight, or no frame) stays registered where it was last seen."""
    newest: dict[str, tuple[float, float, str]] = {}
    for place in place_ids(state):
        rev = latest(state, place)
        for cam, pose in (rev.cameras if rev is not None else {}).items():
            if rev is not None and (cam not in newest or (pose.at, rev.at) > newest[cam][:2]):
                newest[cam] = (pose.at, rev.at, place)
    return {cam: place for cam, (_, _, place) in newest.items()}


MAX_FLOOR_M = 15.0  # a floor hit farther from its camera than this is a pixel near the horizon, not a place in the room
# A ray meeting the floor at less than this (degrees below its plane) lands too far for its error: from 1.7 m up at
# 7.8 degrees, a 4 degree floor tilt moves the hit between 8 and 26 m. On Place 1 every point outside the plan came
# from rays 6.8 to 11 degrees down, and every point inside from 13.3 or more.
MIN_RAY_DEG = 12.0
_MIN_RAY_SIN = math.sin(math.radians(MIN_RAY_DEG))


def undistorted(cam: CameraPose, u: float, v: float) -> tuple[float, float]:
    """A pixel of the camera's own frames (fractions, as `floor_point` takes them) where it is in the undistorted frame
    the pose was registered from: as it is for a pinhole camera, its wide lens undone for one with `lens`."""
    if cam.lens is None:
        return u, v
    from liberational.spatial import lens

    uu, vv = lens.undistort_points(lens.Lens.from_tuple(cam.lens), u, v)
    return float(uu), float(vv)


def floor_point(
    cam: CameraPose, floor: Sequence[float] | None, u: float, v: float, height: float = 0.0
) -> tuple[float, float, float] | None:
    """Where a camera's pixel ray meets the floor, in the map's frame (metres): `u` and `v` as camera_to_plan takes
    them. A pixel of something `height` above the floor (a person's heels and ankles, spatial/footprints.py's FOOT_M)
    is where its ray meets the plane that high, given as the floor point under it: cast to the floor itself, a foot
    point lands too far by its height over the ray's slope (0.2 m at 4.5 m from a camera 1.5 m up). None
    without a floor; when the ray doesn't reach it (at or above the horizon, a camera at or under the floor); when it
    meets the floor less than MIN_RAY_DEG below its plane, where a degree of floor tilt moves the hit metres;
    or when it lands more than MAX_FLOOR_M away. The pixel's ray is a pinhole's with its principal point at the
    frame's centre, as MASt3R fits it, after a wide lens's distortion is undone (`CameraPose.lens`)."""
    if floor is None:
        return None
    u, v = undistorted(cam, u, v)
    m = cam.pose
    d = ((u - 0.5) * cam.width / cam.focal, (v - 0.5) * cam.height / cam.focal, 1.0)
    origin = (m[3], m[7], m[11])
    ray = tuple(m[4 * r] * d[0] + m[4 * r + 1] * d[1] + m[4 * r + 2] * d[2] for r in range(3))
    nx, ny, nz, off = floor
    up = nx * origin[0] + ny * origin[1] + nz * origin[2] + off  # the camera's height
    rate = nx * ray[0] + ny * ray[1] + nz * ray[2]
    length = (ray[0] ** 2 + ray[1] ** 2 + ray[2] ** 2) ** 0.5
    unit = (nx * nx + ny * ny + nz * nz) ** 0.5  # a floor given with a normal not of length 1 still measures angles
    if rate >= -1e-9 or up <= 0 or -rate / (length * unit) < _MIN_RAY_SIN:
        return None  # level with or rising off the floor, too shallow to trust, or a camera at or under the floor
    above = height * unit  # the plane's offset in the floor's own units
    if up <= above:
        return None  # the camera is no higher than the point
    s = (above - up) / rate
    if s * length > MAX_FLOOR_M:
        return None
    drop = above / (unit * unit)  # straight down onto the floor
    return (
        origin[0] + s * ray[0] - drop * nx,
        origin[1] + s * ray[1] - drop * ny,
        origin[2] + s * ray[2] - drop * nz,
    )


def camera_to_plan(state: Path, camera: str, u: float, v: float) -> tuple[str, float, float] | None:
    """Where a pixel of a registered fixed camera lands on its place's floor, in the plan's frame: (place, x, y) in
    metres, the frame the plan's DSL is drawn in, or the floor's own frame (`floor_frame`) while the place has no plan.
    `u` and `v` are the pixel's position as fractions of the frame's width and height (0 to 1, from the top left, the
    frame as see-decode delivers it), so any resolution works; a foot point is the bottom middle of a person's box.
    None when the camera isn't registered, its place has no floor yet, or the pixel's ray doesn't reach the floor (it
    looks at or above the horizon). For many pixels, read the revision once and call `project`."""
    place = camera_places(state).get(camera)
    rev = latest(state, place) if place is not None else None
    hit = project(rev, camera, u, v) if rev is not None else None
    return None if hit is None or rev is None else (rev.place, *hit)


def floor_frame(floor: Sequence[float]) -> tuple[float, ...] | None:
    """A map-to-floor transform for a place with no plan (12 numbers, as `Revision.plan_transform`): x is the map's x
    laid onto the floor (its y when x is within 60 degrees of the floor's up), y is up cross x, and the third row is
    the height above the floor. A camera registered alone never sees a room enclosed, so its place gets no plan, yet
    its people stand on a known floor. None for a floor with no normal."""
    nx, ny, nz, off = (float(f) for f in floor)
    unit = math.sqrt(nx * nx + ny * ny + nz * nz)
    if unit < 1e-12:
        return None
    n = (nx / unit, ny / unit, nz / unit)
    axis = (1.0, 0.0, 0.0) if abs(n[0]) < 0.5 else (0.0, 1.0, 0.0)
    dot = sum(a * b for a, b in zip(axis, n, strict=True))
    along = tuple(a - dot * b for a, b in zip(axis, n, strict=True))
    length = math.sqrt(sum(a * a for a in along))
    x = tuple(a / length for a in along)
    y = (n[1] * x[2] - n[2] * x[1], n[2] * x[0] - n[0] * x[2], n[0] * x[1] - n[1] * x[0])
    return (*x, 0.0, *y, 0.0, *n, off / unit)


@functools.lru_cache(maxsize=64)
def _floor_frame_of(floor: tuple[float, ...]) -> tuple[float, ...] | None:
    """`floor_frame`, worked out once per floor: a positions read projects tens of thousands of points through a
    revision with no plan file of its own."""
    return floor_frame(floor)


def project(rev: Revision, camera: str, u: float, v: float, height: float = 0.0) -> tuple[float, float] | None:
    """`camera_to_plan` against one revision: the pixel's floor point as plan (x, y) in metres, or in the floor's own
    frame (`floor_frame`) when the revision has no plan, or None. `height` as `floor_point` takes it."""
    if camera not in rev.cameras:
        return None
    return project_pose(rev, rev.cameras[camera], u, v, height)


def project_pose(rev: Revision, cam: CameraPose, u: float, v: float, height: float = 0.0) -> tuple[float, float] | None:
    """`project` for a camera pose given in `rev`'s map frame rather than registered in it: a rig camera at one moment
    (spatial/rig.py, #5), whose pose comes from its rig's. The same floor and plan geometry as a fixed camera's."""
    if rev.floor is None:
        return None
    tr = rev.plan_transform or _floor_frame_of(tuple(rev.floor))
    found = floor_point(cam, rev.floor, u, v, height)
    if found is None or tr is None:
        return None
    hit = (*found, 1.0)
    x = sum(tr[i] * hit[i] for i in range(4))
    y = sum(tr[4 + i] * hit[i] for i in range(4))
    return round(x, 3), round(y, 3)


def revise(
    state: Path,
    place: str | None,
    *,
    producer: str,
    sources: Iterable[str] = (),
    signals: Mapping[str, Sequence[str]] | None = None,
    map_bytes: bytes | None = None,
    map_format: str | None = None,
    plan: str | None = None,
    cameras: Mapping[str, CameraPose] | None = None,
    floor: Sequence[float] | None = None,
    plan_transform: Sequence[float] | None = None,
    now: float | None = None,
    frame: str | None = None,
) -> Revision:
    """Write a new machine revision of `place` (a new place when None) and return it.

    What isn't given carries over from the place's previous revision: its signals, its map with its floor, its plan
    with its transform, and its registered cameras. A floor or transform given replaces the old one, with or without a
    new map or plan; a new map with no floor given has none, and a new plan with no transform given has none. A new
    map without a plan of its own has no plan (nor transform): a plan is drawn in its map's frame. A new map stands in
    a `frame` of its own (its revision's id) unless one is given: tier 1 gives the frame it fused into. The map
    and plan files are written before the revision that names them, so a reader never finds a revision whose files
    aren't there yet.
    """
    if not producer:
        raise ValueError("a revision names its producer")
    if map_bytes is not None and not map_format:
        raise ValueError("a map names its format")
    place = str(uuid.uuid7()) if place is None else _place_id(place)
    was = latest(state, place)
    at = time.time() if now is None else now
    if was is not None and at <= was.at:
        at = was.at + 0.001  # revisions stay in order even with a clock step back
    rev_id = str(uuid.uuid7())
    folder = root(state) / place
    folder.mkdir(parents=True, exist_ok=True)
    map_file, fmt = (was.map, was.map_format) if was is not None else (None, None)
    if map_bytes is not None:
        map_file, fmt = f"{rev_id}.map", map_format
        _write(folder / map_file, map_bytes)
    # A plan is drawn in its map's frame: a new map without a plan of its own has none (an old plan would sit in the
    # wrong frame, and so would every point placed on it).
    plan_file = was.plan if was is not None and map_bytes is None else None
    if plan is not None:
        plan_file = f"{rev_id}.plan"
        _write(folder / plan_file, plan.encode())
    kept_signals = dict(was.signals) if was is not None else {}
    if signals is not None:
        kept_signals = {str(k): [str(x) for x in v] for k, v in signals.items()}
    kept_cameras = dict(was.cameras) if was is not None else {}
    if cameras is not None:
        kept_cameras = dict(cameras)
    kept_floor = (
        _numbers(_listed(floor), 4, "a floor plane")
        if floor is not None or map_bytes is not None or was is None
        else was.floor
    )
    kept_transform = (
        _numbers(_listed(plan_transform), 12, "a plan transform")
        if plan_transform is not None or plan is not None or map_bytes is not None or was is None
        else was.plan_transform
    )
    # A new map stands in its own frame unless its writer says which it fused into; one that keeps its map keeps its frame.
    kept_frame = (frame or rev_id) if map_bytes is not None else (frame or (was.frame if was is not None else None))
    rev = Revision(
        rev_id,
        place,
        at,
        producer,
        tuple(sources),
        kept_signals,
        map_file,
        fmt,
        plan_file,
        kept_cameras,
        kept_floor,
        kept_transform,
        kept_frame,
    )
    _write(folder / f"{rev_id}.json", json.dumps(rev.as_dict(), sort_keys=True).encode())
    if map_bytes is not None:
        _prune_maps(state, place)
    return rev


def _listed(value: Sequence[float] | None) -> list[float] | None:
    return None if value is None else [float(x) for x in value]


def _prune_maps(state: Path, place: str) -> None:
    """Keep the newest MAPS_KEPT map files (uuid7 names sort by time), and each producer's newest map whatever its
    age: a producer that builds on its own last map (tier 1's fused map) must not lose it because other
    producers wrote newer ones meanwhile, say a fixed-camera snapshot that didn't fuse and the next day's after it.
    Only producers among the last PRODUCER_WINDOW revisions count, so a retired producer's map isn't kept forever."""
    folder = root(state) / place
    maps = sorted(folder.glob("*.map"), key=lambda p: p.name)
    chain = revisions(state, place)
    # A producer not seen lately (a retired version) lets its last map go.
    active = {rev.producer for rev in chain[-PRODUCER_WINDOW:]}
    newest_by_producer: dict[str, str] = {}
    for rev in chain:  # oldest first, so a later revision's map replaces an earlier one's
        if rev.producer in active and rev.map is not None and rev.map.removesuffix(".map") == rev.id:
            newest_by_producer[rev.producer] = rev.map
    keep = {p.name for p in maps[-MAPS_KEPT:]} | set(newest_by_producer.values())
    for old in maps:
        if old.name not in keep:
            old.unlink(missing_ok=True)


def read_map(state: Path, rev: Revision) -> bytes | None:
    """The map file `rev` uses, or None when it has none or the file was pruned."""
    if rev.map is None:
        return None
    try:
        return (root(state) / rev.place / rev.map).read_bytes()
    except FileNotFoundError:
        return None


def splat_path(state: Path, rev: Revision, kind: str) -> Path:
    """Where `rev`'s splat file of `kind` (one of SPLAT_FILES) lives, whether or not it's there."""
    if kind not in SPLAT_FILES:
        raise ValueError(f"a splat file is one of {', '.join(SPLAT_FILES)}")
    return root(state) / rev.place / f"{rev.id}.{kind}"


def has_splat(state: Path, rev: Revision) -> bool:
    """Whether `rev` has a splat: its compact copy, the file the 3D view loads, is there."""
    return splat_path(state, rev, "spz").is_file()


def splat_source(state: Path, rev: Revision) -> str | None:
    """What `rev`'s splat was trained from, as `write_splat` was told ("walk" or "cameras"); None when it wasn't
    told, or there's no splat."""
    try:
        return splat_path(state, rev, "gs.from").read_text().strip() or None
    except FileNotFoundError:
        return None


def write_splat(state: Path, rev: Revision, master: bytes, web: bytes, source: str | None = None) -> None:
    """Keep `rev`'s splat: the 3DGS PLY `master` and its SPZ copy `web`, each written whole, the copy last (it's
    what says a splat is there), with `source` (what it was trained from: "walk" or "cameras") noted beside them. A
    revision's splat is written once: the dashboard serves it as never changing, so a second one for the same
    revision is refused. Older revisions' splats past SPLATS_KEPT are pruned."""
    if has_splat(state, rev):
        raise ValueError(f"revision {rev.id} already has a splat")
    _write(splat_path(state, rev, "gs.ply"), master)
    if source is not None:
        _write(splat_path(state, rev, "gs.from"), source.encode())
    _write(splat_path(state, rev, "spz"), web)
    _prune_splats(state, rev.place)


def _prune_splats(state: Path, place: str) -> None:
    """Keep the splat files of the newest SPLATS_KEPT revisions that have one (uuid7 names sort by time)."""
    folder = root(state) / place
    ids = sorted({p.name.split(".", 1)[0] for kind in SPLAT_FILES for p in folder.glob(f"*.{kind}")})
    for old in ids[:-SPLATS_KEPT]:
        for kind in SPLAT_FILES:
            (folder / f"{old}.{kind}").unlink(missing_ok=True)


def machine_plan(state: Path, rev: Revision) -> str | None:
    """The plan text `rev` uses, or None before the first plan."""
    if rev.plan is None:
        return None
    try:
        return (root(state) / rev.place / rev.plan).read_text()
    except FileNotFoundError:
        return None


# The owner's names and plan corrections: taught state in the knowledge store.


def _store(state: Path) -> knowledge.Store:
    from liberational.people import registry

    return knowledge.at(registry.knowledge_path(state))


def _owned(state: Path, collection: str, place: str | None = None) -> dict[str, knowledge.Record]:
    """The newest record per place in an owner collection. Reading doesn't create a knowledge store."""
    from liberational.people import registry

    if not registry.knowledge_path(state).exists():
        return {}
    newest: dict[str, knowledge.Record] = {}
    for row in _store(state).query(collection, key=place):
        r = row.record
        if r.key is not None and (r.key not in newest or r.start_ns >= newest[r.key].start_ns):
            newest[r.key] = r
    return newest


def _put(state: Path, collection: str, place: str, value: dict[str, Any], now: float | None) -> None:
    at = time.time() if now is None else now
    was = _owned(state, collection, place).get(place)
    at_ns = round(at * 1e9)
    if was is not None and at_ns <= was.start_ns:
        at_ns = was.start_ns + 1  # the newest record stays the one written last
    record = knowledge.Record(str(uuid.uuid7()), collection, place, at_ns, at_ns, time.time_ns(), value)
    _store(state).put([record])


def name(state: Path, place: str, text: str, now: float | None = None, source: str = "owner") -> None:
    """The owner names `place` (an empty name goes back to the placeholder). Keeps a copy of the place's signals.
    `source` says who wrote it: "merge" for a name carried over from a place merged into this one."""
    place = _place_id(place)
    rev = latest(state, place)
    if rev is None:
        raise KeyError(f"no place {place}")
    text = " ".join(text.split())
    if len(text) > MAX_NAME:
        raise ValueError(f"a place name is at most {MAX_NAME} characters")
    value = {"place": place, "name": text, "signals": rev.signals, "source": source}
    _put(state, NAMED, place, value, now)


def correct_plan(
    state: Path,
    place: str,
    plan: str,
    now: float | None = None,
    base: str | None = None,
    source: str = "owner",
    view: Mapping[str, Any] | None = None,
) -> None:
    """The owner's corrected plan for `place`, drawn in the frame of revision `base` (the revision whose plan the owner
    edited; the newest when not given). An empty plan withdraws the correction, so the machine's plan holds again.
    `source` says who wrote it: "merge" for one that carries a merged place's correction into this one.

    `base` is stored and is the plan's `plan_revision` from then on: an owner who corrects their own plan again after
    a newer revision moved the frame is still editing text in the old frame, and labelling it with the newest would
    put positions off it.

    `view` is how the owner drew it, when that isn't the plan frame (an imported plan): `{"plan": <the DSL as
    drawn on its page>, "turn": <radians>, "shift": [x, y], "mirror": <bool>}`, the move from the page into the
    plan frame (spatial/plan_import.py `Move`). The Places tab draws the page and turns positions onto it
    (`page_view`); `plan` stays in the plan frame for every other reader."""
    place = _place_id(place)
    chain = revisions(state, place)
    if not chain:
        raise KeyError(f"no place {place}")
    if base is not None and base not in {r.id for r in chain}:
        raise ValueError(f"{base} is not a revision of {place}")
    value: dict[str, Any] = {"place": place, "plan": plan, "base": base or chain[-1].id, "source": source}
    if view is not None:
        if page_view(view) is None:
            raise ValueError("a plan's view is its page's plan text, turn, shift, and mirror")
        value["view"] = dict(view)
    _put(state, PLAN, place, value, now)


@dataclass(frozen=True)
class PageView:
    """An owner plan as drawn on its page (`correct_plan`'s `view`): the page's DSL, and the move from the page into
    the plan frame, p = R(turn) F q + shift, F mirroring y when `mirror`."""

    plan: str
    turn: float
    shift: tuple[float, float]
    mirror: bool

    def to_page(self, x: float, y: float) -> tuple[float, float]:
        """A plan-frame point on the page."""
        c, s = math.cos(self.turn), math.sin(self.turn)
        dx, dy = x - self.shift[0], y - self.shift[1]
        px, py = c * dx + s * dy, -s * dx + c * dy  # R(-turn)
        return (px, -py) if self.mirror else (px, py)

    def as_dict(self) -> dict[str, Any]:
        return {"plan": self.plan, "turn": self.turn, "shift": list(self.shift), "mirror": self.mirror}


def page_view(value: object) -> PageView | None:
    """A stored `view` read back, or None when there's none or it isn't one."""
    if not isinstance(value, Mapping):
        return None
    plan, turn, shift, mirror = value.get("plan"), value.get("turn"), value.get("shift"), value.get("mirror")
    if not isinstance(plan, str) or not isinstance(turn, int | float) or not isinstance(mirror, bool):
        return None
    if not isinstance(shift, list | tuple) or len(shift) != 2 or not all(isinstance(v, int | float) for v in shift):
        return None
    return PageView(plan, float(turn), (float(shift[0]), float(shift[1])), mirror)


def plan_history(state: Path, place: str) -> list[knowledge.Record]:
    """Every owner plan record of `place` (`correct_plan`'s), oldest first. The store only appends: a newer plan
    supersedes an older one without removing it, so each earlier plan can be read back or restored (`restore_plan`)."""
    from liberational.people import registry

    place = _place_id(place)
    if not registry.knowledge_path(state).exists():
        return []
    found = [row.record for row in _store(state).query(PLAN, key=place)]
    return sorted(found, key=lambda r: (r.start_ns, r.recorded_ns))


def restore_plan(state: Path, place: str, record: str, now: float | None = None) -> None:
    """Make an earlier owner plan of `place` hold again: its plan, frame (`base`), and source, written as the newest
    record, so the plans in between stay in the history too."""
    was = next((r for r in plan_history(state, place) if r.id == record), None)
    if was is None:
        raise KeyError(f"no plan record {record} for {place}")
    _put(state, PLAN, _place_id(place), dict(was.value), now)


def same_plan_frame(a: Revision, b: Revision) -> bool:
    """Whether two revisions of a place draw their plans in one frame: the same plan file, or, for revisions with no
    plan file, the same map frame (`Revision.frame`), floor, and plan transform, as tier 1's fused revisions of one map
    are, whose plan frame is their floor's (`floor_frame`). An owner plan drawn in a revision with no plan file of its
    own (an imported one) holds for every revision in its frame this way."""
    if a.id == b.id:
        return True
    if a.plan is not None or b.plan is not None:
        return a.plan == b.plan
    return a.frame is not None and a.frame == b.frame and a.floor == b.floor and a.plan_transform == b.plan_transform


def plan_frame(rev: Revision) -> tuple[float, ...] | None:
    """`rev`'s map-to-plan transform (12 numbers): its plan's, else its floor's own frame (`floor_frame`); None with
    neither."""
    return rev.plan_transform or (floor_frame(rev.floor) if rev.floor is not None else None)


def _placeholders(firsts: Mapping[str, float]) -> dict[str, str]:
    order = sorted(firsts, key=lambda p: (firsts[p], p))
    return {p: PLACEHOLDER.format(n=i) for i, p in enumerate(order, start=1)}


def places(state: Path) -> list[Place]:
    """Every place, in the order it was first made, with the owner's name and plan over the machine's."""
    chains = {p: revisions(state, p) for p in place_ids(state)}
    chains = {p: c for p, c in chains.items() if c}
    placeholders = _placeholders({p: c[0].at for p, c in chains.items()})
    names = _owned(state, NAMED)
    plans = _owned(state, PLAN)
    out: list[Place] = []
    for place, chain in sorted(chains.items(), key=lambda kv: (kv[1][0].at, kv[0])):
        rev = chain[-1]
        named = names.get(place)
        given = str(named.value.get("name") or "") if named is not None else ""
        corrected = plans.get(place)
        owner_plan = str(corrected.value.get("plan") or "") if corrected is not None else ""
        machine = machine_plan(state, rev)
        if owner_plan and corrected is not None:
            plan, source = owner_plan, "owner"
            # a machine plan newer than the correction: the one the newest revision has, made by a revision after it.
            # A joined map that derives no plan has none, whatever earlier revisions made.
            made = next((r for r in chain if rev.plan is not None and r.id == rev.plan.removesuffix(".plan")), None)
            machine_newer = machine is not None and made is not None and made.at * 1e9 > corrected.start_ns
            # the frame the owner's plan is drawn in: the revision it was saved against, or for a correction saved
            # without one, the revision current when it was saved
            stored = corrected.value.get("base")
            then = [r for r in chain if r.at * 1e9 <= corrected.start_ns]
            frame = stored if stored in {r.id for r in chain} else (then[-1] if then else chain[0]).id
        else:
            plan, source, machine_newer = machine, ("machine" if machine is not None else None), False
            frame = rev.id if machine is not None else None
        view = page_view(corrected.value.get("view")) if source == "owner" and corrected is not None else None
        out.append(
            Place(place, given or placeholders[place], bool(given), rev, plan, source, machine_newer, frame, view)
        )
    return out


def get(state: Path, place: str) -> Place | None:
    return next((p for p in places(state) if p.id == place), None)


def visits(state: Path, start: float, end: float, place: str | None = None) -> list[Visit]:
    """The visits that overlap [start, end), in time order, in `place` when given."""
    found = [Visit.from_record(r) for r in S.at(state).range(S.PLACE_VISITS, start, end)]
    return sorted((v for v in found if place is None or v.place == place), key=lambda v: (v.start, v.id))


# The journal's view: the places in a span, and the rooms seen in each.


@dataclass(frozen=True)
class PlaceRef:
    """A place visited in a span, as the journal's pages show it."""

    place: str | None  # the place id, or None for rooms seen while no place visit is known
    name: str  # the owner's name, else the placeholder ("Place 1"); "" with no place
    named: bool  # the owner named it
    start: float  # its first and last time in the span (epoch seconds), clipped to the span
    end: float
    rooms: tuple[str, ...]  # rooms seen there in the span, in first-seen order
    # The owner was there: a place visit (the rig) puts them there. False for a place only observed, where a fixed
    # camera registered to it saw someone (a housemate at home while the owner was out); the pages' Places line
    # leaves those out. A ref with no place stands for wherever the rig was, so it counts as visited.
    visited: bool = True


def room_kinds(state: Path) -> dict[str, str]:
    """Each zone id the owner ever drew, with its kind ("area" or "screen")."""
    from liberational.see import zones

    return {z.id: z.kind for d in zones.history(state) for z in d.zones}


def _rooms(state: Path, start: float, end: float) -> list[tuple[float, float, str, str]]:
    """(start, end, room, camera) for each person's visit to a drawn area zone that overlaps [start, end), in time
    order."""
    kinds = room_kinds(state)
    found = []
    for r in S.at(state).range(S.ZONE_VISITS, start, end):
        if r.get("subject_kind") != "person" or kinds.get(str(r.get("zone")), "area") != "area":
            continue
        name = " ".join(str(r.get("zone_name") or "").split())
        if name:
            found.append(
                (max(float(r["ts_start"]), start), min(float(r["ts_end"]), end), name, str(r.get("cam") or ""))
            )
    return sorted(found)


def places_for_span(state: Path, start: float, end: float) -> list[PlaceRef]:
    """The places visited or observed in [start, end), in first-seen order, each with the rooms seen there; [] when
    nothing is known. A place a registered fixed camera saw someone in, with no visit of the owner's, is observed
    only (`visited` False). Rooms seen outside every place visit come last, under a ref with no place."""
    if end <= start:
        return []
    spans: dict[str | None, list[float]] = {}

    def seen(place: str | None, lo: float, hi: float) -> None:
        span = spans.setdefault(place, [lo, hi])
        span[0], span[1] = min(span[0], lo), max(span[1], hi)

    at = visits(state, start, end)
    for v in at:
        seen(v.place, max(v.start, start), min(v.end, end))
    visited = {v.place for v in at}
    rooms: dict[str | None, list[str]] = {}
    cams = camera_places(state)
    for lo, hi, room, cam in _rooms(state, start, end):
        # A fixed camera registered to a place saw the room there, wherever the rig was. Otherwise it's the
        # place the rig was in when the room was seen: the first visit that overlaps it.
        registered = cams.get(cam)
        place = registered or next((v.place for v in at if v.start <= hi and v.end >= lo), None)
        if place is None or registered is not None:
            seen(place, lo, hi)
        if room not in rooms.setdefault(place, []):
            rooms[place].append(room)
    known = {p.id: p for p in places(state)} if any(k is not None for k in spans) else {}
    out: list[PlaceRef] = []
    for place, (lo, hi) in sorted(spans.items(), key=lambda kv: (kv[0] is None, kv[1][0])):
        p = known.get(place) if place is not None else None
        name = p.name if p is not None else ""
        out.append(
            PlaceRef(
                place,
                name,
                p.named if p is not None else False,
                lo,
                hi,
                tuple(rooms.get(place, ())),
                visited=place is None or place in visited,
            )
        )
    return out
