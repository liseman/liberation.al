"""The position layer: where people, pets, and the rig wearer were, in one stream every source writes and one
reader every view uses (the Places tab's paths, a place's last activity).

Placed on write, read as stored. A record (`Position`, stored in `segment_log.PLACED`) is a subject at a time in a
place: who or what (`entity`), the place, the room, plan coordinates (`x`, `y`, in the plan frame of the place
revision `revision` names), how sure, and the source. Every source writes that shape through `write`:
- **fixed-camera**: footprints (spatial/footprints.py) are recorded raw, the foot point as the camera saw it
  (`Sample`, `segment_log.POSITIONS`: `camera`, `u`, `v`, about one per track a second, for people and animals
  alike; a person's `track` is "<camera>:<pose track number>", an animal's "<camera>:<run>:<species>:<n>",
  `animal_track`). The placer (`advance`, run after each recording) puts them on the plan, names them, joins and
  fuses the cameras' tracks (`derive`, everything below), and writes the result.
  A camera is read at the spot the owner gave it on its place's plan when it has one (spatial/spots.py: the plan's
  own frame, no map needed, and the spot outranks a map's registration), else through its registration in the map.
- **zone** (the coarse fallback): the placer writes each person or animal visit to a drawn area zone
  (`zone.visit`) as a room-only record over the visit's span, with no x and y.
- **rig-pose** (once there is a rig pose): the rig's writer gives `place`, `x`, `y`, and the revision whose plan
  frame they are in, and calls `write` itself.

A read (`positions`) decodes stored records and nothing else: it never projects, names tracks, or joins paths, so
its cost is the rows of the window, thinned as they are scanned for a long one, and it stops between days when its
caller says the client has gone (`Cancelled`). Before this, every read placed every raw sample of its window, and a
week of a busy place passed the client's deadline.

What placing on write costs: a record keeps the coordinates and the identity it was given when it was placed.
- **Frames.** Records are placed in the frame of the plan the place draws (`Place.plan_revision`), through the
  cameras' registration then, and say so (`revision`). When a later registration puts people somewhere else (a
  camera registered again with another pose, or the plan moved to a new frame), the placer notices on its next
  run (`_basis`: where a few fixed pixels of each camera land on the plan, against where they landed when the
  place was last placed) and places that place's raw footprints again from scratch (`place_again`), so a path that
  spans the registration is one line in one frame. Until that run, a read for another frame of the place moves
  the points by the similarity between the two frames (`_Places.carried`, for two revisions of one map frame),
  and marks a record it can't move `in_frame` False. A place is placed again into a new generation of its slot:
  the store hides rows appended into a forgotten range, so a generation is a new slot name, not a rewrite.
- **Identity.** A track is named from the faces and labels known when it was placed (at most a day after it was
  seen). A person named since then shows under their name (names are looked up on read), but a track nobody could
  name then stays unnamed until the place is placed again (`place_again`, `reset`).
- **Chunks.** The placer works through time in CHUNK_S pieces, each read with MARGIN_S of context either side, so
  a track that spans a chunk's end is named and joined the same on both sides when its faces and partner are
  within the margin; when they aren't, its path changes id there and a view draws two lines end to end.

The rest of this docstring is how `derive` places and names the raw footprints.

`entity` is an id ("person:<PersonId>", "pet:<pet id>", "face:<N>" for an unnamed face group, "rig"), or None until
someone is identified; `track` ("<camera>:<track id>") still groups an unknown's samples into a path. A reader names
an entity when it reads, as speaker tokens are, so renaming someone rewrites nothing stored.

Who a camera track is, is worked out when it's placed (`identify`): a person's track by the faces see-actions matched on it (the name at least LOOK_S around the read that MIN_SHARE of
them agree on, as zone visits are named in memory/zone_people.py), or, for a track with no named face, by the
unnamed face group ("Face 3") MIN_SHARE of its faces were put in: the same unnamed person's tracks share
"face:3", on every camera. Naming the group names its faces (vision/sightings.py), so its tracks become that
person's when next read. Group ids are never reused, and a recenter keeps a group's id; a group folded or merged
into another (vision/faces.py) is read as the group it went into (segment_log maps the old number on every
read). An
animal's by the owner's label of that track, else by pet recognition where it's switched on
(people/pet_recognition.py). A track nothing names that way takes the entity of the one track of its kind another
camera of its place saw at the same time (`co_observed`): the only one in view then, or the only one beside it
on the plan, as when a person walks away from one camera while another sees their face. One entity's tracks on two
cameras that were seen together close by on the plan are one path (`joined`, `Position.path`), with one fused point a
second where both cameras saw them (see FUSE_M); tracks of one entity further apart stay their own paths, so a path
never zig-zags between where two cameras put two people both named the same.

An animal is up on furniture often enough that its box's bottom cast to the floor can land beyond a wall: a pet on a
cabinet against the far wall of its room, put on the floor, is a pet behind that wall, which the placer used to give
no point at all. Its footprint is put at the furniture instead (`_Places.at_furniture`): the farthest point along the
camera's sight line toward that floor point that the camera still sees by the plan, AT_FURNITURE_M short of where
the line leaves the room, so it is in the room the pet is seen in, at the furniture, short of the wall. That is a
guess to a metre or so, so the record says so (`approximate`), and a view may mark it apart from a point on the
floor. A person's foot there isn't furniture's: it is pose's, and `_deferred` puts it where another camera saw them.

A person's foot on an occluding edge (a couch's back) is put where another camera saw them that second
(`_deferred`). A track whose rows went to one other track that way is that person drawn twice, one line over the other,
though `joined` never joined them: it measures the two at the edge track's own far points. So it's drawn on that
track's path, and the two take one entity (`_deferred_joins`, DEFER_SHARE). Over 7.7 h of Place 1,
named entity-seconds drawn on two paths at once went from 972 of 17,139 (5.7%) to 195 of 18,245
(1.1%). With every name hidden, of the 20 such joins between two tracks faces named, 19 were one person; the other
wasn't drawn joined, and no path held two people faces named apart.

An unnamed face group is no one in particular, so it yields to a name (`_ungrouped`, `_named_apart`). A track
whose faces only made a group takes the one person whose faces named the tracks it partners with on other cameras
(`_partners`, either way), and a group is never named apart from anyone but a person the owner said it isn't
(people/face_nots.py): owner marks win. Living-room looks down on people from behind,
so a named person's track there often gets a fresh group from a few looks at the back of their head while cam0 sees
their face. Over 10 h, of the rows where the person on cam0's ray was named apart from cam0's track (504),
475 were a person against a group; of 8 such pairs checked in the frames, 7 were plainly one person and 1 was unclear.
With named people's names hidden, their partners named them back rightly for 36 tracks and wrongly for 2 (one pair, a
track named by 240 faces against one named by a single face). It renamed 4 group tracks, and cam0's uncertain rows went
from 1,159 to 651.

Retention: positions are dense (a tracked person or pet is a row a second), so past the 30-day window the nightly
retention pass (retention.py) thins each track to one sample per 10 s and its last, kept like the other condensed
streams.
"""

from __future__ import annotations

import bisect
import collections
import json
import math
import statistics
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from liberational import json_state, logs
from liberational import segment_log as S
from liberational.memory import places
from liberational.people import face_nots, pets
from liberational.spatial import cameras, floorplan, footprints, lens, plan_scene, poses, spots

LOG = logs.get("spatial-positions")

FIXED_CAMERA, RIG_POSE, ZONE = "fixed-camera", "rig-pose", "zone"
RIG = "rig"  # the rig wearer's entity, and the rig's slot
FACE = "face"  # an unnamed face group's entity: "face:3" for "Face 3"
LOOK_S = 120.0  # a face matched this long before or after the read, on the track, still says who it was
MIN_SHARE = 0.75  # this share of a track's named faces must agree on one name
# A track no face or label names takes the entity of a track of its kind another camera of its place saw at the same
# time, in either of two ways, each needing CO_MIN_S seconds seen together:
# - by time: it was the only such track in view, for CO_SHARE of the unnamed track's seconds;
# - by where: over their seconds both were put on the plan, it was the only such track within JOIN_M (median
#   distance) and no other was within RIVAL_M. Over 7.2 h of tracks, placed through reprojection
#   registration, the two named 45 more tracks (23 by time alone); hiding named people's entities, the joins put back
#   29 of 29 rightly (28 of 29 at 0.75 m and 1.25 m). Before reprojection registration, plan distance couldn't tell two
# people apart.
CO_MIN_S = 5
CO_SHARE = 0.8
JOIN_M = 0.9
RIVAL_M = 1.4
# One entity's tracks on two cameras of a place are one path when they were seen together for CO_MIN_S seconds
# at a median plan distance of FUSE_M or less (80 of 88 such pairs in one day's tracks, whose median was 0.58 m). In
# each second both cameras placed it, the path's point is their mean weighted by confidence times the squared sine of
# each camera's ray to the floor: a steep ray's point moves less for a pixel's error than a grazing one's. Over those
# seconds the fused point's step-to-step wobble (second difference) was 0.105 m against 0.155 m for one camera alone,
# and the step where the path went from one camera to both was 0.22 m (0.34 m with equal weights). A camera whose foot
# point sits on an occluding edge (`footprints.at_edge`) is left out of a second another camera's isn't.
FUSE_M = 1.4
# A track that ended and one that started on the same camera soon after, where it could have walked to, are one
# person's track broken in two (`handoffs`): the second started within HANDOFF_S of the first's end, within
# HANDOFF_M plus WALK_MS a second of where it ended, and neither had another such partner. Over 9 h of tracks,
# of 99 such pairs whose tracks faces named, 96 named one person; 73 more tracks would be named so. Between
# two cameras it didn't hold: cam0 and living-room see one room, so a track ending on one rarely starts on the other,
# and of 7 such named pairs 2 agreed. A handoff names an unnamed track and joins the two into one path. People's
# tracks only: an animal's handoffs weren't measured.
HANDOFF_S = 10.0
HANDOFF_M = 0.6
WALK_MS = 1.0
# A person's foot on an occluding edge is hidden behind it and guessed by pose, or just beyond it, so they
# stood somewhere on the floor the surface at that pixel hides along the camera's ray: from where the ray meets the
# surface's height to the foot's point. Which way they were from the camera is still right. When the other camera
# saw several people that second (`_deferred`), the one whose point is within RAY_M of that stretch of floor, with
# every other more than RAY_GAP_M further, is the one. Over 10 h, on cam0's feet at an edge in seconds
# living-room saw two or more people whose named one matched cam0's name (69 rows, 25 tracks, names hidden), it picked
# 58 rightly and none wrongly (1.0 m picked 62, but 5 wrongly); on cam0's feet that were clear, with a made-up stretch
# from 0.5 m short of the point to 1 m past it, 270 of 273 rightly and none wrongly. On the rows names or partners
# already settled it never picked another, and the track's own clear points nearby never disagreed with it.
RAY_M = 0.7
RAY_GAP_M = 0.5
# A track whose foot rows were put on another camera's track's points (`_deferred`) is that person drawn twice, one
# line over the other, so it's drawn on that track's path (`_deferred_joins`): when CO_MIN_S or more of its
# rows went to one track and DEFER_SHARE of all that went anywhere did, or (a short track) DEFER_SHARE of all its rows
# did, and the two tracks' clear points, in the seconds both had one, are within FUSE_M (median). Over
# 7.7 h of Place 1 (see the module docstring), 250 of the 268 tracks with a row put elsewhere put every one on one
# track, and 260 put DEFER_SHARE or more there.
DEFER_SHARE = 0.75
# Such a join takes a track of the camera already on that path that shares up to DEFER_OVERLAP_S seconds with it: the
# tracker often starts a person's next track a second or two before it lets go of the last.
DEFER_OVERLAP_S = 2
# A subject whose floor point its camera's spot can't see by the plan (a pet up on a cabinet, cast to the floor beyond
# the wall) is put where its sight line leaves the room it is seen in (`_Places.at_furniture`), AT_FURNITURE_M short
# of that wall: the furniture stands inside the room, short of the wall, and a room's wall centre line is 0.15 m off
# its inner face (a 0.3 m thick wall), so this lands inside the room rather than on its wall. It is a guess to a metre
# or so, so the point says it is approximate.
AT_FURNITURE_M = 0.3
# A bisection's steps for where a sight line leaves the room: 20 of them are a millionth of the line's length.
FURNITURE_STEPS = 20


@dataclass(frozen=True)
class Sample:
    """One stored observation, exactly as written."""

    id: str
    t: float
    end: float
    source: str
    confidence: float
    entity: str | None
    track: str | None
    camera: str | None
    u: float | None
    v: float | None
    place: str | None
    room: str | None
    x: float | None
    y: float | None
    revision: str | None


@dataclass(frozen=True)
class Position:
    """Where an entity was: the record every position source writes (`write`) and every view reads (`positions`),
    placed on its place's plan. `name` is looked up on read and `in_frame` worked out then; `u` and `v` are only on a
    row fresh from `derive`, and aren't stored."""

    id: str
    t: float
    end: float  # t for a sample; a zone visit's end for a zone row
    entity: str | None
    name: str | None  # the entity's current name; None when unknown or unnamed
    track: str | None
    source: str
    confidence: float
    place: str | None
    room: str | None  # the room's name: the plan's, or a zone row's drawn zone
    room_id: str | None  # the plan's room id (its zone's, spatial/plan_scene.py); None for a zone row or with no plan
    x: float | None  # plan metres, in the frame the plan is drawn in; None when only the place or room is known
    y: float | None
    camera: str | None
    u: float | None
    v: float | None
    revision: str | None  # the place revision whose plan frame x and y are in
    # x and y are in the frame asked for (`positions(frame_revision=)`); always True when none was asked for
    in_frame: bool = True
    # The path it's drawn on: its track's, or a joined path's id (its tracks, "+"-joined) when one entity's tracks on
    # several cameras were fused into one (`joined`); a fused point's track and camera are None
    path: str | None = None
    # A person's foot on an occluding edge (`footprints.at_edge`) that no other camera's point settled: hidden
    # behind it and guessed by pose, or just beyond it, so it may be metres off. A view may leave it out of a path.
    uncertain: bool = False
    # An animal its camera's spot can't see on the floor by the plan, placed at the furniture in the room it is seen
    # in instead (`_Places.at_furniture`): a pet up on a cabinet or a shelf. Where it is to a metre or so, not where
    # it stood, so a view may mark it apart from a point on the floor.
    approximate: bool = False


def animal_track(track: str | None) -> str | None:
    """The animal's track id ("<run>:<species>:<n>", people/pets.py's TRACK) of a position's track, or None for a
    person's track or none."""
    rest = (track or "").partition(":")[2]
    return rest if pets.TRACK.fullmatch(rest) else None


def _slot(row: Mapping[str, Any]) -> str:
    return S.slot(str(row.get("camera") or (RIG if row.get("source") == RIG_POSE else row.get("source") or "unknown")))


def write_samples(state: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Append raw camera observations, for the placer (`advance`) to put on the plan (the fields: `t`, `source`, `confidence`, and the optional
    `end`, `id`, `entity`, `track`, `camera`, `u`, `v`, `place`, `room`, `x`, `y`, `revision`). Nothing to write is
    fine."""
    if not rows:
        return
    by_slot: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        t = float(r["t"])
        record = {
            **{k: r.get(k) for k in ("entity", "track", "camera", "u", "v", "place", "room", "x", "y", "revision")},
            "id": str(r.get("id") or uuid.uuid7()),
            "ts_start": t,
            "ts_end": max(t, float(r.get("end", t))),
            "source": str(r["source"]),
            "confidence": float(r["confidence"]),
        }
        by_slot.setdefault(_slot(r), []).append(record)
    writers = S.Writers(S.root(state))
    try:
        for slot, records in by_slot.items():
            writers.append(S.POSITIONS, slot, records)
        writers.sync()
    finally:
        writers.close()


def _sample(r: Mapping[str, Any]) -> Sample:
    return Sample(
        id=str(r["id"]),
        t=float(r["ts_start"]),
        end=float(r["ts_end"]),
        source=str(r["source"]),
        confidence=float(r["confidence"]),
        **{k: r.get(k) for k in ("entity", "track", "camera", "u", "v", "place", "room", "x", "y", "revision")},
    )


def samples(state: Path, start: float, end: float, cameras: Sequence[str] | None = None) -> list[Sample]:
    """Every stored observation overlapping [start, end), exactly as written, in time order (`cameras` keeps those
    cameras' slots): the raw rows, for a job that projects them itself (the walkable floor)."""
    # The store reads only these slots' files: a filter applied after reading every slot saved nothing.
    slots = sorted({S.slot(c) for c in cameras}) if cameras is not None else None
    if slots == []:
        return []
    return [_sample(r) for r in S.at(state).range(S.POSITIONS, start, end, slots)]


def thin[T](items: Iterable[T], key: Callable[[T], object], at: Callable[[T], float], every: float) -> list[T]:
    """Items thinned to a path: of each `key`'s items, the first in every `every` seconds and the very last (so a
    path keeps its end), in time order. Retention thins stored rows past the window with it; `positions(every=)`
    thins a long read with it."""
    ordered = sorted(items, key=at)
    seen: set[tuple[object, int]] = set()
    last: dict[object, int] = {}
    keep: set[int] = set()
    for i, item in enumerate(ordered):
        k = key(item)
        last[k] = i
        bucket = (k, int(at(item) // every))
        if bucket not in seen:
            seen.add(bucket)
            keep.add(i)
    keep.update(last.values())
    return [item for i, item in enumerate(ordered) if i in keep]


def path_key(track: str | None, entity: str | None, source: str | None, camera: str | None, row_id: str) -> object:
    """What one path is: a track (or with none an entity) from one source and camera. A row with neither is its own
    path, so two anonymous untracked people are never merged into one."""
    return (track or entity or f"row:{row_id}", source, camera)


Plan = list[tuple[str, str, tuple[tuple[float, float], ...]]]


def _plan_matrix(transform: Sequence[float]) -> np.ndarray:
    """A revision's 12-number map-to-plan transform as a 4x4."""
    out = np.eye(4)
    out[:3] = np.asarray(transform, dtype=np.float64).reshape(3, 4)
    return out


def plan_move(src: Sequence[float], dst: Sequence[float]) -> floorplan.Similarity | None:
    """The move from one plan frame to another of the same map, on the floor: a point of the plan drawn by map-to-plan
    `src` back to the map, and onto the plan drawn there by `dst`. None when that mirrors the plan (it never should:
    every frame here is right-handed)."""
    full = _plan_matrix(dst) @ np.linalg.inv(_plan_matrix(src))
    corners = np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]]) @ full.T
    o, ex, ey = corners[0, :2], corners[1, :2] - corners[0, :2], corners[2, :2] - corners[0, :2]
    if ex[0] * ey[1] - ex[1] * ey[0] <= 0:
        return None
    scale = (float(np.linalg.norm(ex)) + float(np.linalg.norm(ey))) / 2
    return floorplan.Similarity(scale, math.atan2(float(ex[1]), float(ex[0])), (float(o[0]), float(o[1])))


class _Places:
    """One read of the places for a whole call: each camera's place and current revision, the plan rooms in each
    revision's frame, the rooms points fell in, and the cameras' pose series and held intrinsics, all read once."""

    def __init__(self, state: Path, aims: Mapping[str, spots.Aim] | None = None) -> None:
        self.state = state
        self.cams = places.camera_places(state)
        # The cameras the owner gave a spot on a plan (spatial/spots.py): each is read at its spot, in the revision
        # its plan is drawn in (`revisions` lays the spots over that revision), whatever a map registers
        self.spots = spots.read(state, aims)  # `aims`: a dry run's, in place of the stored ones
        self._all: dict[str, list[places.Revision]] = {}
        self.revs: dict[str, places.Revision | None] = {
            p: next(iter(self.revisions(p)), None) for p in set(self.cams.values())
        }
        self._plans: dict[str, Plan] = {}
        self._rooms: dict[tuple[str, float, float], tuple[str, str] | None] = {}
        self._by_id: dict[tuple[str, str], places.Revision | None] = {}
        self._backgrounds: dict[tuple[str, str], places.Background | None] = {}
        self._poses: dict[tuple[str, str], places.CameraPose | None] = {}  # `camera`'s, by revision and camera
        self.edges: dict[tuple[str, str, float, float], bool] = {}  # `_at_edge`'s answers, by pixel to a thousandth
        self._moves: dict[tuple[str, str], floorplan.Similarity | None] = {}
        # Each revision's cameras' pose series (spatial/poses.py), by (place, map frame, cameras)
        self._series: dict[tuple[str, str | None, tuple[str, ...]], Mapping[str, poses.Series]] = {}
        self._held: dict[str, lens.Held] | None = None  # every camera's held intrinsics, read when first needed
        # A camera's pose at a moment (`camera_at`) and a revision holding it (`view`), by revision, camera, and the
        # pose's 16 numbers: one per still stretch, not per row
        self._at: dict[tuple[str, str, tuple[float, ...]], places.CameraPose] = {}
        self._views: dict[tuple[str, str, tuple[float, ...]], places.Revision] = {}
        self._sights: dict[str, Callable[[tuple[float, float], tuple[float, float]], bool] | None] = {}
        self._furniture: dict[tuple[str, float, float], tuple[float, float] | None] = {}  # `at_furniture`'s, by pixel
        # The footprints left unplaced because their camera can't see their floor point by the plan and nothing came
        # of putting them at the furniture (`place`), as (camera, track, time): a set, since a read places a row
        # more than once
        self.hidden: set[tuple[str, str | None, float]] = set()

    def revision(self, place: str, rev_id: str | None) -> places.Revision | None:
        """A place's revision by id (a rig row names the one whose frame its x and y are in)."""
        if rev_id is None:
            return None
        if (place, rev_id) not in self._by_id:
            self._by_id[(place, rev_id)] = next((r for r in self.revisions(place) if r.id == rev_id), None)
        return self._by_id[(place, rev_id)]

    def revisions(self, place: str) -> list[places.Revision]:
        """A place's revisions, newest first."""
        if place not in self._all:
            try:
                found = places.revisions(self.state, place)
            except ValueError:
                found = []
            found = [spots.overlay(r, self.spots) for r in found]
            self._all[place] = sorted(found, key=lambda r: r.at, reverse=True)
        return self._all[place]

    def spot_revision(self, camera: str) -> places.Revision | None:
        """The revision a camera with an owner spot is read through: the one its plan is drawn in, with the spots
        laid over it. None for a camera with no spot."""
        spot = self.spots.get(camera)
        return self.revision(spot.place, spot.revision) if spot is not None else None

    def background(self, place: str, camera: str) -> places.Background | None:
        """What a camera sees with nobody in the way, by the place's current registration, with the cells it doesn't
        know taken from earlier registrations of the same view (`footprints.filled`)."""
        if (place, camera) not in self._backgrounds:
            now = self.revs.get(place)
            held = now.cameras.get(camera) if now is not None else None
            found = held.background if held is not None else None
            if found is not None and now is not None and None in found.heights:
                earlier = (r.cameras[camera].background for r in self.revisions(place) if camera in r.cameras)
                found = footprints.filled(found, (b for b in earlier if b is not None and b is not found))
            self._backgrounds[(place, camera)] = found
        return self._backgrounds[(place, camera)]

    def camera(self, rev: places.Revision, camera: str | None) -> places.CameraPose | None:
        """`rev`'s pose of a camera, with the current revision's background for it (`background`, filled from earlier
        ones) when `rev` is the current one or has none: a plan the owner corrected stays in the frame of an older
        revision, maybe from before backgrounds were kept, and a background (each pixel's surface height as a
        share of the camera's) doesn't depend on the map frame. An older revision's own background stays its own.
        Made once per revision and camera: every sample of a read asks for it."""
        key = (rev.id, camera or "")
        if key in self._poses:
            return self._poses[key]
        cam = rev.cameras.get(camera) if camera else None
        now = self.revs.get(rev.place)
        if cam is not None and camera is not None and (cam.background is None or now is None or rev.id == now.id):
            held = self.background(rev.place, camera)
            cam = replace(cam, background=held) if held is not None else cam
        self._poses[key] = cam
        return cam

    def series(self, rev: places.Revision) -> Mapping[str, poses.Series]:
        """The pose series (`poses.series`) of `rev`'s fixed cameras in its place and map frame, read once."""
        key = (rev.place, rev.frame, tuple(sorted(rev.cameras)))
        if key not in self._series:
            self._series[key] = poses.series(self.state, rev.place, rev.frame, key[2])
        return self._series[key]

    def _pose_at(self, rev: places.Revision, camera: str, t: float) -> tuple[float, ...] | None:
        """`camera`'s pose at `t` in `rev`'s map frame as 16 numbers, row by row, or None with no series that
        gives one, or for a camera read at its owner's spot."""
        if camera in self.spots:
            return None  # the owner's spot is where it is, at every time
        found = self.series(rev).get(camera)
        m = found.at(t) if found is not None else None
        return None if m is None else tuple(float(x) for x in np.asarray(m, dtype=np.float64).reshape(16))

    def camera_at(self, rev: places.Revision, camera: str | None, t: float) -> places.CameraPose | None:
        """`rev`'s pose of a camera (`camera`, background and all) as it was at `t`: its pose the one its series
        gives at `t` (`series`), with its held focal and lens (spatial/lens.py `held`, `cameras.with_held`); `rev`'s
        own pose when it has no series or none there. Made once per still stretch, not per row."""
        cam = self.camera(rev, camera)
        if cam is None or camera is None:
            return cam
        flat = self._pose_at(rev, camera, t)
        if flat is None:
            return cam
        key = (rev.id, camera, flat)
        if key not in self._at:
            if self._held is None:
                self._held = lens.held(self.state)
            self._at[key] = cameras.with_held(camera, replace(cam, pose=flat), self._held.get(camera))
        return self._at[key]

    def view(self, rev: places.Revision, camera: str, t: float) -> places.Revision:
        """`rev` with `camera` as it was at `t` (`camera_at`), to project its pixels through; `rev` itself when its
        own pose is the one."""
        flat = self._pose_at(rev, camera, t)
        if flat is None or camera not in rev.cameras:
            return rev
        key = (rev.id, camera, flat)
        if key not in self._views:
            cam = self.camera_at(rev, camera, t)
            self._views[key] = rev if cam is None else replace(rev, cameras={**rev.cameras, camera: cam})
        return self._views[key]

    def plan(self, rev: places.Revision) -> Plan:
        """The rooms of the plan drawn in `rev`'s frame. Plan frames aren't stable across revisions: each map is a new
        frame, and each floorplan is rotated and moved to start at 0,0. So it's the owner's correction when `rev` uses
        the very machine plan the owner corrected (the same plan file, so the same frame; for a plan drawn in a revision
        with no plan file, the same map frame, `places.same_plan_frame`), else `rev`'s own machine plan. A place
        with an owner's plan has no other: a revision that isn't in its frame has no rooms, never a machine plan's."""
        if rev.id not in self._plans:
            text: str | None = None
            held = places.get(self.state, rev.place)
            owned = held is not None and held.plan_source == "owner"
            if held is not None and owned and held.plan_revision is not None:
                corrected = self.revision(rev.place, held.plan_revision)
                if corrected is not None and places.same_plan_frame(corrected, rev):
                    text = held.plan
            if text is None and not owned:
                text = places.machine_plan(self.state, rev)
            self._plans[rev.id] = plan_scene.rooms_of(text or "")
        return self._plans[rev.id]

    def room(self, rev: places.Revision, x: float, y: float) -> tuple[str, str] | None:
        """(room id, name) at plan (x, y) in `rev`'s frame, cached to the centimetre."""
        key = (rev.id, round(x, 2), round(y, 2))
        if key not in self._rooms:
            self._rooms[key] = floorplan.room_at(self.plan(rev), x, y)
        return self._rooms[key]

    def carried(self, now: places.Revision, frame: places.Revision) -> floorplan.Similarity | None:
        """The move from `now`'s plan frame (its floor's own frame without a plan, `places.floor_frame`) onto
        `frame`'s, two revisions of one place whose maps stand in one frame (`Revision.frame`): an owner plan stays in
        the frame it was drawn in while later revisions of the same map draw theirs elsewhere. None when the two maps
        are in different frames, or either has no plan frame: nothing relates them."""
        key = (now.id, frame.id)
        if key not in self._moves:
            src, dst = places.plan_frame(now), places.plan_frame(frame)
            same = now.id == frame.id or (now.frame is not None and now.frame == frame.frame)
            self._moves[key] = plan_move(src, dst) if same and src is not None and dst is not None else None
        return self._moves[key]

    def sight(self, place: str) -> Callable[[tuple[float, float], tuple[float, float]], bool] | None:
        """Whether a camera at a spot on `place`'s owner plan sees a floor point by the plan (`plan_scene.sight`), read
        once; None for a place with no owner plan that reads."""
        if place not in self._sights:
            held = places.get(self.state, place)
            found = None
            if held is not None and held.plan_source == "owner" and held.plan:
                try:
                    found = plan_scene.sight(plan_scene.load(held.plan))
                except plan_scene.SceneError:
                    found = None
            self._sights[place] = found
        return self._sights[place]

    def seen(self, camera: str, hit: tuple[float, float]) -> bool:
        """Whether `camera`, read at its owner spot, can see plan point `hit` (in its spot's frame) by the plan. A
        floor point behind a solid wall from the spot isn't where the subject stood: it is up on furniture (a pet
        on a cabinet, cast to the floor as if it stood there, lands metres beyond the wall, and `at_furniture` puts
        it at the furniture instead), or the pose is off."""
        spot = self.spots.get(camera)
        sees = self.sight(spot.place) if spot is not None else None
        return sees is None or spot is None or sees(spot.at, hit)

    def at_furniture(self, camera: str, hit: tuple[float, float]) -> tuple[float, float] | None:
        """Where a subject up on furniture is put when the floor point its pixel casts to (`hit`, in the camera's
        spot's frame) is one its spot can't see: the farthest point along the sight line from the spot toward `hit`
        that the spot still sees by the plan, AT_FURNITURE_M back from where the line leaves the room, so the point
        is in the room the subject is seen in and short of its wall, where the furniture stands. None when it can't
        be put there at all: no spot on a plan, no plan to ask, or a line the spot can't see from its start."""
        key = (camera, round(hit[0], 3), round(hit[1], 3))  # a pet standing still is the same pixel
        if key not in self._furniture:
            self._furniture[key] = self._at_furniture(camera, hit)
        return self._furniture[key]

    def _at_furniture(self, camera: str, hit: tuple[float, float]) -> tuple[float, float] | None:
        """`at_furniture`, worked out once per pixel: a bisection for where the sight line leaves the room. The
        points of a line a spot sees are a prefix of it (the solid wall it crosses hides everything past), so a
        bisection finds that end."""
        spot = self.spots.get(camera)
        sees = self.sight(spot.place) if spot is not None else None
        if spot is None or sees is None:
            return None
        (ax, ay), (bx, by) = spot.at, hit
        run = math.dist((ax, ay), (bx, by))
        if run <= AT_FURNITURE_M:
            return None  # the room's wall is on the camera: nothing between it and the furniture
        near, far = 0.0, 1.0  # of the way from the spot to the floor point
        for _ in range(FURNITURE_STEPS):
            mid = (near + far) / 2
            if sees(spot.at, (ax + mid * (bx - ax), ay + mid * (by - ay))):
                near = mid
            else:
                far = mid
        back = max(near * run - AT_FURNITURE_M, 0.0) / run  # short of the wall, where the furniture stands
        return round(ax + back * (bx - ax), 3), round(ay + back * (by - ay), 3)

    def place(self, sample: Sample, frame: places.Revision | None = None) -> Placed:
        """Where a sample is. With no `frame`: a row that gives its own place (a rig's) keeps its point, in the frame of
        the revision it names, and a camera's row is projected through the camera's current registration. With
        `frame` (a revision of the sample's place), a camera's row is projected through that revision's own pose of
        the camera, so it lands in that revision's plan frame; a row that can't be (a rig row from another frame, or
        a camera that revision doesn't register and that can't be moved onto it, `carried`) keeps its own frame and
        comes back `in_frame` False. Either way, a camera with a pose series in that revision's map frame is projected
        through its pose at the row's time (`view`) in place of the revision's.

        A camera with an owner spot is checked against the plan: a point its spot can't see is behind a wall from it.
        An animal's is put at the furniture in the room it is seen in (`at_furniture`) and comes back `approximate`;
        a person's, and an animal's that can't be put there, gets no point (`hidden`)."""
        if sample.source != FIXED_CAMERA or sample.camera is None or sample.place is not None:
            rev = self.revision(sample.place, sample.revision) if sample.place is not None else None
            found = (
                self.room(rev, sample.x, sample.y) if rev and sample.x is not None and sample.y is not None else None
            )
            # Same frame means the same plan file (`places.same_plan_frame`): equal transform values from two plans don't
            # make one frame.
            same = frame is None or (rev is not None and places.same_plan_frame(rev, frame))
            room = sample.room or (found[1] if found else None)
            return Placed(sample.place, room, found[0] if found else None, sample.x, sample.y, sample.revision, same)
        place = self.cams.get(sample.camera)
        spotted = self.spot_revision(sample.camera)  # an owner's spot outranks the current registration
        rev = spotted or (self.revs.get(place) if place is not None else None)
        in_frame = True
        move = None  # from the current frame onto `frame`'s, for a camera `frame` doesn't register
        if frame is not None and frame.place == place:
            if sample.camera in frame.cameras and (spotted is None or spotted.id == frame.id):
                rev = frame
            elif rev is not None and (move := self.carried(rev, frame)) is None:
                in_frame = False  # that revision doesn't register this camera: placed in the current frame
        elif frame is not None:
            in_frame = False
        if rev is None or sample.u is None or sample.v is None:
            return Placed(place, None, None, None, None, None, in_frame)
        # A person's footprint is their heels or ankles, a few centimetres up; an animal's, its box's bottom
        height = 0.0 if animal_track(sample.track) is not None else footprints.FOOT_M
        hit = places.project(self.view(rev, sample.camera, sample.t), sample.camera, sample.u, sample.v, height)
        approximate = False
        if hit is not None and spotted is not None and rev.id == spotted.id and not self.seen(sample.camera, hit):
            # Behind a wall from the camera: not on the floor where it was seen. An animal is up on furniture there
            # (a pet on a cabinet, its box's bottom cast to the floor beyond the wall), so it is put at the furniture
            # in the room it is seen in, and the point says it is approximate. People's feet are pose's, and a
            # person's point there is the pose's, not furniture's (`_at_edge`, `_deferred`).
            put = self.at_furniture(sample.camera, hit) if animal_track(sample.track) is not None else None
            if put is None:
                self.hidden.add((sample.camera, sample.track, sample.t))  # nothing to put it on
                hit = None
            else:
                hit, approximate = put, True
        if move is not None and frame is not None:
            rev = frame
            if hit is not None:
                x, y = move.point(*hit)
                hit = (round(x, 3), round(y, 3))
        if hit is None:
            return Placed(place, None, None, None, None, rev.id, in_frame)
        found = self.room(rev, *hit)
        return Placed(
            place, found[1] if found else None, found[0] if found else None, *hit, rev.id, in_frame, approximate
        )


@dataclass(frozen=True)
class Placed:
    place: str | None
    room: str | None
    room_id: str | None
    x: float | None
    y: float | None
    revision: str | None
    in_frame: bool
    # Its camera's spot can't see its floor point by the plan and it was put at the furniture in the room it is seen
    # in instead (`_Places.at_furniture`): a pet up on a cabinet. Where it is to a metre or so, not where it stood.
    approximate: bool = False


def _names(state: Path, entities: Iterable[str | None]) -> dict[str, str | None]:
    """Each entity id's current name: a person's through the registry, a pet's through its profile, the rig's the
    owner's."""
    from liberational import config as cfg
    from liberational.people import pets, registry

    wanted = {e for e in entities if e}
    out: dict[str, str | None] = {}
    reg = registry.at(state) if any(e.startswith("person:") for e in wanted) else None
    person_names: dict[str, str | None] = {}
    if reg is not None:
        # One registry snapshot for this read: avoids one refresh per entity.
        person_names = {str(person.id): person.display_name for person in reg.people()}
    pet_names: dict[str, str] = {}
    if any(e.startswith("pet:") for e in wanted):
        pet_names = {str(p["id"]): str(p["name"]) for p in pets.at(state).listing()}
    for e in wanted:
        kind, _, key = e.partition(":")
        if kind == "person" and reg is not None:
            out[e] = person_names.get(key)
        elif kind == "pet":
            out[e] = pet_names.get(key)
        elif e == RIG:
            out[e] = cfg.owner_name() or None
        elif kind == FACE:
            out[e] = f"Face {key}"
        else:
            out[e] = None
    return out


def _zone_rows(state: Path, start: float, end: float, where: _Places) -> list[Position]:
    """Person and animal visits to drawn area zones as room-only positions: the coarse fallback, never stored. Their
    `track` is the same see-actions track a camera's footprints carry ("<camera>:<track id>"), so one person's room
    spans and placed points fall in one path. `entity` is None here (a zone visit's `label` is the detector's class,
    "person" or "cat", not who it was): `positions` names the track as it names a camera's."""
    kinds = places.room_kinds(state)
    visits = places.visits(state, start, end)
    out = []
    for r in S.at(state).range(S.ZONE_VISITS, start, end):
        if r.get("subject_kind") not in ("person", "animal") or kinds.get(str(r.get("zone")), "area") != "area":
            continue
        room = " ".join(str(r.get("zone_name") or "").split())
        if not room:
            continue
        lo, hi, cam = float(r["ts_start"]), float(r["ts_end"]), str(r.get("cam") or "")
        place = where.cams.get(cam) or next((v.place for v in visits if v.start <= hi and v.end >= lo), None)
        out.append(
            Position(
                id=str(r["id"]),
                t=lo,
                end=hi,
                entity=None,
                name=None,
                track=f"{cam}:{r.get('track_id')}" if r.get("track_id") else None,
                source=ZONE,
                confidence=1.0,
                place=place,
                room=room,
                room_id=None,
                x=None,
                y=None,
                camera=cam or None,
                u=None,
                v=None,
                revision=None,
            )
        )
    return out


def _agreed(names: Sequence[str]) -> str | None:
    """The name at least MIN_SHARE of `names` agree on, or None (none, or two people on one track)."""
    if not names:
        return None
    name, n = collections.Counter(names).most_common(1)[0]
    return name if n >= MIN_SHARE * len(names) else None


def identify(state: Path, start: float, end: float, tracks: Iterable[str]) -> dict[str, str]:
    """Who each camera track ("<camera>:<track id>") was over [start, end), by its entity id, for the tracks someone
    can be named on (see the module docstring); the rest are left out. Recognition failing names no pets, never
    fails the read."""
    from liberational.people import pet_recognition, registry
    from liberational.see import subjects
    from liberational.vision import sightings

    people: dict[str, set[str]] = {}
    animals: dict[str, set[str]] = {}
    for track in set(tracks):
        cam, _, tid = track.partition(":")
        if cam and tid:
            (animals if animal_track(track) else people).setdefault(cam, set()).add(tid)
    out: dict[str, str] = {}
    reg = registry.at(state) if people else None
    if people:
        votes: dict[str, list[str]] = {}
        groups: dict[str, list[str]] = {}  # the unnamed face groups ("Face 3" -> "3") the other faces were put in
        # Read faces once, then keep only the tracks this request needs.
        for f in sightings.faces_many(start - LOOK_S, end + LOOK_S, people.keys(), state=state):
            cam = str(f.get("cam") or "")
            tid = str(f.get("track_id") or "")
            if not cam or not tid or tid not in people.get(cam, set()):
                continue
            track = f"{cam}:{tid}"
            name, cluster = f.get("name"), f.get("cluster")
            if isinstance(name, str) and name.strip():
                votes.setdefault(track, []).append(name)
            elif isinstance(cluster, str) and (group := subjects.FACE_GROUP.fullmatch(cluster)):
                groups.setdefault(track, []).append(group.group(1))
        people_by_name: dict[str, str | None] = {}
        for track in votes.keys() | groups.keys():
            if track in votes:  # named faces decide, as before: a group never outvotes them
                name = _agreed(votes[track])
                if not name:
                    continue
                if name not in people_by_name:
                    pid = reg.person_id(name) if reg is not None else None
                    people_by_name[name] = f"person:{pid}" if pid is not None else None
                if people_by_name[name] is not None:
                    out[track] = str(people_by_name[name])
            elif (group := _agreed(groups[track])) is not None:
                out[track] = f"{FACE}:{group}"
    if animals:
        listing = pets.at(state).listing()
        known = {str(p["id"]) for p in listing}
        labels = pet_recognition.labels(state)
        unlabeled = {cam: {tid for tid in wanted if tid not in labels} for cam, wanted in animals.items()}
        try:
            found = pet_recognition.resolve_many(state, start, end, unlabeled, listing)
        except Exception as error:  # recognition is extra: without it the paths still show, unnamed
            LOG.warning("pet recognition for positions failed: %s", error)
            found = {}
        for cam, wanted in animals.items():
            for tid in wanted:
                track = f"{cam}:{tid}"
                label = labels[tid].get("label") if tid in labels else (found.get(track) or {}).get("label")
                if label in known:
                    out[track] = f"pet:{label}"
    return out


def _kind(track: str) -> str:
    """What a track can be joined with: a person's with a person's, an animal's with one of its species."""
    animal = animal_track(track)
    return "person" if animal is None else f"animal:{animal.split(':')[1]}"


def _partners(
    stored: Iterable[Sample], cams: Mapping[str, str], points: Mapping[str, tuple[float, float]] | None = None
) -> tuple[dict[str, str], dict[str, str]]:
    """Each camera track's partner on another camera of its place, whoever either is (see CO_MIN_S): the one track of
    its kind it was seen with (by time), else the one beside it on the plan with no other near (by where), as a
    track's partner and, apart, each track's stored entity. `points` as `co_observed` takes them."""
    seconds: dict[str, dict[int, tuple[float, float] | None]] = {}
    where: dict[str, tuple[str, str, str]] = {}  # track: (place, camera, kind)
    stored_as: dict[str, str] = {}
    for s in stored:
        place = cams.get(s.camera) if s.source == FIXED_CAMERA and s.track and s.camera else None
        if place is None or s.track is None or s.camera is None:
            continue
        at = seconds.setdefault(s.track, {})
        if at.get(int(s.t)) is None:
            at[int(s.t)] = (points or {}).get(s.id)
        where[s.track] = (place, s.camera, _kind(s.track))
        if s.entity:
            stored_as.setdefault(s.track, s.entity)
    seen: dict[tuple[str, str, int], set[str]] = {}  # (place, kind, second): the tracks in view
    for track, secs in seconds.items():
        place, _, kind = where[track]
        for sec in secs:
            seen.setdefault((place, kind, sec), set()).add(track)
    out: dict[str, str] = {}
    for track, secs in seconds.items():
        place, camera, kind = where[track]
        together: collections.Counter[str] = collections.Counter()
        apart: dict[str, list[float]] = {}  # the other track's distance in each second both were placed
        for sec, xy in secs.items():
            for other in seen[(place, kind, sec)]:
                if where[other][1] == camera:
                    continue
                together[other] += 1
                there = seconds[other][sec]
                if xy is not None and there is not None:
                    apart.setdefault(other, []).append(math.dist(xy, there))
        if len(together) == 1:  # by time: the only one in view
            ((other, n),) = together.items()
            if n >= CO_MIN_S and n >= CO_SHARE * len(secs):
                out[track] = other
                continue
        near = {o: statistics.median(d) for o, d in apart.items() if together[o] >= CO_MIN_S and len(d) >= CO_MIN_S}
        close = [o for o, d in near.items() if d <= JOIN_M]
        if len(close) == 1 and all(d > RIVAL_M for o, d in near.items() if o != close[0]):
            out[track] = close[0]  # by where: the only one beside it
    return out, stored_as


def _named_by_partners(
    partner: Mapping[str, str], known: Mapping[str, str], stored_as: Mapping[str, str]
) -> dict[str, str]:
    """The entities `co_observed` gives: an unnamed track takes its partner's, never one found so."""
    named = {**stored_as, **known}
    return {t: named[p] for t, p in partner.items() if t not in named and p in named}


def _named_apart(a: str | None, b: str | None, nots: Mapping[int, set[str]] | None = None) -> bool:
    """Whether two entities are two different people or animals. An unnamed face group is no one in particular (see
    the module docstring), so it's never apart from anyone, nor is an unnamed track, except from a person the owner
    said the group isn't (`nots`, people/face_nots.py: each group's person ids)."""
    if not a or not b or a == b:
        return False
    for group, other in ((a, b), (b, a)):
        if group.startswith(f"{FACE}:"):
            kind, _, pid = other.partition(":")
            number = group.partition(":")[2]
            return kind == "person" and number.isdigit() and pid in (nots or {}).get(int(number), set())
    return True


def _ungrouped(
    partner: Mapping[str, str], known: Mapping[str, str], nots: Mapping[int, set[str]] | None = None
) -> dict[str, str]:
    """The person each track named only by an unnamed face group is (see the module docstring): the one person whose
    faces named the tracks it partners with (`_partners`, either way), unless the owner said the group isn't them
    (`nots`). `known` is who each track was (`identify`)."""
    near: dict[str, set[str]] = {}
    for a, b in partner.items():
        near.setdefault(a, set()).add(b)
        near.setdefault(b, set()).add(a)
    out: dict[str, str] = {}
    for track, entity in known.items():
        if not entity.startswith(f"{FACE}:"):
            continue
        people = {known[o] for o in near.get(track, ()) if known.get(o, "").startswith("person:")}
        if len(people) == 1 and not _named_apart(entity, person := people.pop(), nots):
            out[track] = person
    return out


def co_observed(
    stored: Iterable[Sample],
    known: Mapping[str, str],
    cams: Mapping[str, str],
    points: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, str]:
    """Entities for the camera tracks no face or label names, from a track of their kind another camera of their place
    saw at the same time (see CO_MIN_S): {track: entity}. `known` is who each track was (`identify`); a track's entity
    found here never names another, so it can't spread from track to track. `cams` is each camera's place, and
    `points` each sample's plan point by its id (the join by where uses them; without them, only the join by time)."""
    partner, stored_as = _partners(stored, cams, points)
    return _named_by_partners(partner, known, stored_as)


def pairs(partner: Mapping[str, str]) -> list[tuple[str, str]]:
    """The tracks that are each other's partner (`_partners`), once each: one person seen by two cameras, named or
    not, which `joined` makes one path (so a foot on an occluding edge defers to the other camera's)."""
    return sorted((a, b) for a, b in partner.items() if a < b and partner.get(b) == a)


def joined(
    stored: Iterable[Sample],
    placed: Mapping[str, Placed],
    links: Iterable[tuple[str, str]] = (),
    near: Iterable[tuple[str, str]] = (),
    nots: Mapping[int, set[str]] | None = None,
) -> dict[str, str]:
    """The tracks fused into one path (see FUSE_M), each to its path's id (the sorted tracks, "+"-joined). `placed` is
    where the camera rows were put on the plan, in one frame per place (positions' join by where); `links` are pairs of
    tracks to join as well, after one entity's tracks seen together: a track broken in two (`handoffs`), or two
    cameras' tracks that are each other's partner (`pairs`), named or not. `near` are pairs joined last whatever their
    distance (a track whose rows were put on another's, `_deferred_joins`), and never into a path holding someone
    named apart from them (`nots` as `_named_apart` takes them)."""
    by: dict[str, dict[int, tuple[float, float]]] = {}
    entity: dict[str, str | None] = {}
    info: dict[str, tuple[str | None, str | None, str | None]] = {}  # track: (place, camera, entity)
    seconds: dict[str, set[int]] = {}  # every second each camera track was in view, for the guard below
    camera: dict[str, str | None] = {}
    for s in stored:
        if s.source != FIXED_CAMERA or not s.track:
            continue
        seconds.setdefault(s.track, set()).add(int(s.t))
        camera[s.track] = s.camera
        entity.setdefault(s.track, s.entity)
        p = placed.get(s.id)
        if p is None or p.x is None or p.y is None:
            continue
        by.setdefault(s.track, {}).setdefault(int(s.t), (p.x, p.y))
        info[s.track] = (p.place, s.camera, s.entity)
    at: dict[tuple[str | None, str, int], list[str]] = {}
    for track, secs in by.items():
        place_of, _, entity_of = info[track]
        if entity_of is None:
            continue  # one entity's tracks seen together; an unnamed track joins only by `links`
        for sec in secs:
            at.setdefault((place_of, entity_of, sec), []).append(track)
    shared: collections.Counter[tuple[str, str]] = collections.Counter()
    for tracks in at.values():
        for i, a in enumerate(tracks):
            for b in tracks[i + 1 :]:
                if info[a][1] != info[b][1]:
                    shared[(a, b) if a < b else (b, a)] += 1
    parent = {t: t for t in seconds}

    def root(t: str) -> str:
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    gaps = sorted(
        (statistics.median(math.dist(by[a][sec], by[b][sec]) for sec in by[a].keys() & by[b].keys()), a, b)
        for (a, b), n in shared.items()
        if n >= CO_MIN_S
    )
    members = {t: [t] for t in seconds}

    def apart(a: str, b: str) -> float:
        """Their median distance over the seconds both were placed; 0 when there were none (a track broken in two)."""
        both = by.get(a, {}).keys() & by.get(b, {}).keys()
        return statistics.median(math.dist(by[a][sec], by[b][sec]) for sec in both) if both else 0.0

    # The closest seen together first, then the links, then the tracks put on another's
    for gap, a, b, overlap in [
        *((gap, a, b, 0) for gap, a, b in gaps),
        *((apart(a, b), a, b, 0) for a, b in links if a in seconds and b in seconds),
        *((0.0, a, b, DEFER_OVERLAP_S) for a, b in near if a in seconds and b in seconds),
    ]:
        ra, rb = root(a), root(b)
        if gap > FUSE_M or ra == rb:
            continue
        # Never two tracks of one camera at the same time on one path: two people both named the same, say
        clash = (
            camera[x] == camera[y] and len(seconds[x] & seconds[y]) > overlap for x in members[ra] for y in members[rb]
        )
        if any(clash) or any(_named_apart(entity[x], entity[y], nots) for x in members[ra] for y in members[rb]):
            continue
        parent[ra] = rb
        members[rb] += members.pop(ra)
    return {t: "+".join(sorted(g)) for g in members.values() if len(g) > 1 for t in g}


@dataclass(frozen=True)
class _Span:
    """A person's camera track from its first row to its last, for `handoffs`."""

    place: str
    camera: str
    first: Sample
    last: Sample


def handoffs(
    spans: Mapping[str, _Span], point: Callable[[Sample], tuple[float, float] | None]
) -> list[tuple[str, str]]:
    """People's tracks broken in two on one camera: (a, b) where b started within HANDOFF_S after a ended, on
    the same camera, its first point within HANDOFF_M plus WALK_MS a second of a's last, and neither has another such
    partner. `point` puts a row on the plan (None when it can't be); only the ends of tracks close enough in time are
    put there."""
    lines: dict[tuple[str, str], list[str]] = {}
    for track, sp in spans.items():
        lines.setdefault((sp.place, sp.camera), []).append(track)
    after: dict[str, list[str]] = {}
    before: dict[str, list[str]] = {}
    for tracks in lines.values():
        tracks.sort(key=lambda t: spans[t].first.t)
        starts = [spans[t].first.t for t in tracks]
        for a in tracks:
            end = spans[a].last
            for b in tracks[bisect.bisect_left(starts, end.t) : bisect.bisect_right(starts, end.t + HANDOFF_S)]:
                if b == a:
                    continue
                here, there = point(end), point(spans[b].first)
                gap = spans[b].first.t - end.t
                if here is not None and there is not None and math.dist(here, there) <= HANDOFF_M + WALK_MS * gap:
                    after.setdefault(a, []).append(b)
                    before.setdefault(b, []).append(a)
    return sorted((a, bs[0]) for a, bs in after.items() if len(bs) == 1 and len(before[bs[0]]) == 1)


def _steepness(where: _Places, s: Sample, at: Placed) -> float:
    """The squared sine of the angle at which the camera's ray for `s` meets the floor, in the revision `at` is in,
    from the camera's pose at the row's time (`_Places.camera_at`), as it was placed."""
    rev = where.revision(at.place, at.revision) if at.place is not None else None
    cam = where.camera_at(rev, s.camera, s.t) if rev is not None and s.camera else None
    if rev is None or cam is None or rev.floor is None or s.u is None or s.v is None:
        return 0.0
    hit = places.floor_point(cam, rev.floor, s.u, s.v)
    if hit is None:
        return 0.0
    m, (nx, ny, nz, off) = cam.pose, rev.floor
    origin = (m[3], m[7], m[11])
    up = (nx * origin[0] + ny * origin[1] + nz * origin[2] + off) / math.sqrt(nx * nx + ny * ny + nz * nz)
    return (up / max(math.dist(hit, origin), 1e-9)) ** 2


def _at_edge(where: _Places, s: Sample, at: Placed) -> bool:
    """Whether the camera saw `s`'s foot point on an occluding edge (`footprints.at_edge`). It's judged through the
    place's current registration of the camera, whatever frame `at` is in: EDGE_M is metres, and a plan the owner
    corrected may be in an older revision's frame, at another scale or with no background. A cell of the
    background no registration saw (`_Places.background`) can't say the foot is clear: it's judged as if furniture
    (`footprints.FURNITURE_M`) stood there, so it defers to a clear camera or is uncertain. False for a camera
    with no background at all, no floor, or a foot outside the frame. A pixel whose background couldn't hide EDGE_M of
    floor however far off it lies is answered without casting its ray (most pixels see the floor)."""
    now = where.revs.get(at.place) if at.place is not None else None
    rev = now if now is not None and s.camera in now.cameras else None
    if rev is None and at.place is not None:
        rev = where.revision(at.place, at.revision)
    cam = where.camera(rev, s.camera) if rev is not None else None
    if rev is None or cam is None or rev.floor is None or cam.background is None or s.u is None or s.v is None:
        return False
    if not (0.0 <= s.u < 1.0 and 0.0 <= s.v < 1.0):
        return False
    seen = cam.background.at(s.u, s.v)
    m, (nx, ny, nz, off) = cam.pose, rev.floor
    up = (nx * m[3] + ny * m[7] + nz * m[11] + off) / math.sqrt(nx * nx + ny * ny + nz * nz)
    foot = footprints.FOOT_M
    surface = footprints.FURNITURE_M if seen is None else seen * up
    if up <= foot or max(surface - foot, 0.0) * places.MAX_FLOOR_M / (up - foot) < footprints.EDGE_M:
        return False
    key = (rev.id, str(s.camera), round(s.u, 3), round(s.v, 3))  # someone standing still is the same pixel
    if key not in where.edges:
        unseen = footprints.FURNITURE_M if seen is None else None
        where.edges[key] = footprints.at_edge(cam, rev.floor, s.u, s.v, unseen) is True
    return where.edges[key]


def _stretch(
    where: _Places, s: Sample, at: Placed
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None:
    """The floor the surface at `s`'s foot pixel hides along the camera's ray, on the plan of the revision `at` is in
    (see RAY_M): (the point under the camera, where the hidden floor starts, the foot's point), from the camera's
    pose at the row's time (`_Places.camera_at`), as it was placed. None when unknown."""
    rev = where.revision(at.place, at.revision) if at.place is not None else None
    cam = where.camera_at(rev, s.camera, s.t) if rev is not None else None
    if rev is None or cam is None or rev.floor is None or rev.plan_transform is None or at.x is None or at.y is None:
        return None
    seen = cam.background.at(s.u, s.v) if cam.background is not None and s.u is not None and s.v is not None else None
    if seen is None:
        return None
    m, (nx, ny, nz, off) = cam.pose, rev.floor
    unit = nx * nx + ny * ny + nz * nz
    height = nx * m[3] + ny * m[7] + nz * m[11] + off  # in the floor's own units
    up = height / math.sqrt(unit)
    if up <= footprints.FOOT_M:
        return None
    under = (m[3] - height / unit * nx, m[7] - height / unit * ny, m[11] - height / unit * nz, 1.0)
    tr = rev.plan_transform
    cx, cy = (sum(tr[i] * under[i] for i in range(4)), sum(tr[4 + i] * under[i] for i in range(4)))
    # Along the floor, the ray at the surface's height is (up - surface) / (up - FOOT_M) of the way to the foot's point
    k = min(max((up - seen * up) / (up - footprints.FOOT_M), 0.0), 1.0)
    return (cx, cy), (cx + k * (at.x - cx), cy + k * (at.y - cy)), (at.x, at.y)


def _off_stretch(
    q: tuple[float, float], stretch: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
) -> float:
    """How far plan point `q` is from a `_stretch` of hidden floor: off the ray, and short of or past the stretch."""
    (cx, cy), (ax, ay), (bx, by) = stretch
    run = math.hypot(bx - cx, by - cy)
    if run <= 0:
        return math.dist(q, (bx, by))
    ux, uy = (bx - cx) / run, (by - cy) / run
    rx, ry = q[0] - cx, q[1] - cy
    along, across = rx * ux + ry * uy, abs(ry * ux - rx * uy)
    outside = max(math.hypot(ax - cx, ay - cy) - along, along - run, 0.0)
    return math.hypot(across, outside)


def _deferred(
    stored: Sequence[Sample],
    place: Callable[[Sample], Placed],
    edge: Mapping[str, bool],
    partner: Mapping[str, str],
    where: _Places,
    nots: Mapping[int, set[str]] | None = None,
) -> dict[str, tuple[Sample, Sample]]:
    """A person's camera row whose foot is on an occluding edge (`edge`, by row id) put where another camera of the
    place saw the same person that second, when exactly one did with its foot clear: a track of the same
    entity, or the row's track's partner (`_partners`, either way) when the two aren't named apart. Else, the one of
    those cameras' clear rows that second on the floor the edge hides along the row's ray (see RAY_M), when it isn't
    named apart from the row and no other row of the row's camera picked it too. {row id: (the row at that camera's
    point, kept as its own (place, x, y, revision), the other camera's row it was put at)}."""
    clear: dict[tuple[str, int], list[tuple[Sample, Placed]]] = {}
    for s in stored:
        p = place(s)
        if s.id in edge and not edge[s.id] and p.place is not None and p.x is not None:
            clear.setdefault((p.place, int(s.t)), []).append((s, p))

    def same(a: Sample, b: Sample) -> bool:
        if a.entity and a.entity == b.entity:
            return True
        if _named_apart(a.entity, b.entity, nots):
            return False
        return partner.get(a.track or "") == b.track or partner.get(b.track or "") == a.track

    out: dict[str, tuple[Sample, Sample]] = {}
    by_ray: dict[tuple[str | None, int, str], list[str]] = {}  # (camera, second, the row picked): the rows picking it
    for s in stored:
        if not edge.get(s.id):
            continue
        p = place(s)
        others = [(o, q) for o, q in clear.get((str(p.place), int(s.t)), []) if o.camera != s.camera]
        there = [(o, q) for o, q in others if same(s, o)]
        if len(there) == 1:
            o, q = there[0]
            out[s.id] = (replace(s, place=q.place, room=None, x=q.x, y=q.y, revision=q.revision), o)
            continue
        stretch = _stretch(where, s, p)
        if stretch is None:
            continue
        near = sorted(
            ((_off_stretch((float(q.x or 0.0), float(q.y or 0.0)), stretch), o.id, o, q) for o, q in others),
            key=lambda d: d[:2],
        )
        if not near or near[0][0] > RAY_M or (len(near) > 1 and near[1][0] <= near[0][0] + RAY_GAP_M):
            continue
        _, _, o, q = near[0]
        if _named_apart(s.entity, o.entity, nots) or q.revision != p.revision:
            continue  # named apart, or placed in another frame
        out[s.id] = (replace(s, place=q.place, room=None, x=q.x, y=q.y, revision=q.revision), o)
        by_ray.setdefault((s.camera, int(s.t), o.id), []).append(s.id)
    for rows in by_ray.values():
        if len(rows) > 1:  # two people behind the edge on one ray: it can't say which
            for i in rows:
                del out[i]
    return out


def _deferred_joins(
    moved: Mapping[str, tuple[Sample, Sample]],
    stored: Iterable[Sample],
    edge: Mapping[str, bool],
    place: Callable[[Sample], Placed],
) -> list[tuple[str, str]]:
    """The tracks to draw on the path of the track their rows were put on (`_deferred`, see DEFER_SHARE): (the track,
    the one its rows went to). `edge` and `place` are `_deferred`'s; the two tracks' clear points (`edge` False) in the
    seconds both had one say whether they could be one person."""
    stored = list(stored)
    rows = collections.Counter(s.track for s in stored if s.track)
    went: dict[str, collections.Counter[str]] = {}
    for row, there in moved.values():
        if row.track and there.track:
            went.setdefault(row.track, collections.Counter())[there.track] += 1
    best = {}
    for track, to in went.items():
        other, n = to.most_common(1)[0]
        if n >= DEFER_SHARE * (to.total() if n >= CO_MIN_S else rows[track]):
            best[track] = other
    wanted = set(best) | set(best.values())
    clear: dict[str, dict[int, tuple[float, float]]] = {}
    for s in stored:
        if s.track in wanted and edge.get(s.id) is False:
            p = place(s)
            if p.x is not None and p.y is not None:
                clear.setdefault(s.track, {}).setdefault(int(s.t), (p.x, p.y))
    out = []
    for track, other in sorted(best.items()):
        mine, theirs = clear.get(track, {}), clear.get(other, {})
        both = mine.keys() & theirs.keys()
        if not both or statistics.median(math.dist(mine[t], theirs[t]) for t in both) <= FUSE_M:
            out.append((track, other))
    return out


def _path_entities(paths: Mapping[str, str], entity: Mapping[str, str | None]) -> dict[str, str | None]:
    """The entity each joined path's tracks are drawn as, by track, where a path holds more than one (the tracks of
    `_deferred_joins`): its one person or pet, else its most seen face group, else none."""
    groups: dict[str, list[str]] = {}
    for track, path in paths.items():
        groups.setdefault(path, []).append(track)
    out: dict[str, str | None] = {}
    for tracks in groups.values():
        held = [entity.get(t) for t in tracks]
        if len(set(held)) < 2:
            continue
        named = {e for e in held if e and not e.startswith(f"{FACE}:")}
        faces = collections.Counter(e for e in held if e and e.startswith(f"{FACE}:"))
        one = named.pop() if len(named) == 1 else (faces.most_common(1)[0][0] if faces and not named else None)
        if one is not None or not named:
            out |= {t: one for t in tracks}
    return out


def _fuse(
    stored: Sequence[Sample], paths: Mapping[str, str], place: Callable[[Sample], Placed], where: _Places
) -> tuple[list[Sample], dict[str, str], set[str], set[str]]:
    """`stored` with each joined path's rows in a second two or more of its cameras placed replaced by one fused row
    (see FUSE_M), each fused row's path by its id, the fused rows whose every camera's foot was on an edge, and those
    whose mean is of a camera's point at the furniture (`_Places.at_furniture`, approximate)."""
    at: dict[tuple[str, int], list[Sample]] = {}
    for s in stored:
        if s.track is not None and s.track in paths:
            at.setdefault((paths[s.track], int(s.t)), []).append(s)
    gone: set[str] = set()
    made: list[Sample] = []
    fused: dict[str, str] = {}
    doubtful: set[str] = set()  # the fused rows whose every camera's foot was on an edge
    rough: set[str] = set()  # the fused rows a camera's point at the furniture went into
    for (path, _), rows in at.items():
        if len({r.camera for r in rows}) < 2:
            continue
        here = [(r, p) for r in rows if (p := place(r)).x is not None and p.y is not None and p.in_frame]
        if len({r.camera for r, _ in here}) < 2:
            continue
        # A foot on an occluding edge is hidden behind it and guessed, or just beyond it: the camera can't tell.
        # Defer to the cameras whose foot isn't on one; when every camera's is, none is preferred.
        clear = [(r, p) for r, p in here if not _at_edge(where, r, p)]
        used = clear or here
        weights = [r.confidence * _steepness(where, r, p) for r, p in used]
        if sum(weights) <= 0:
            weights = [1.0] * len(used)
        total = sum(weights)
        x = sum(w * float(p.x or 0.0) for w, (_, p) in zip(weights, used)) / total
        y = sum(w * float(p.y or 0.0) for w, (_, p) in zip(weights, used)) / total
        first = min(here, key=lambda rp: (rp[0].t, rp[0].id))
        row = Sample(
            id=f"fused:{first[0].id}",
            t=first[0].t,
            end=max(r.end for r, _ in here),
            source=FIXED_CAMERA,
            confidence=max(r.confidence for r, _ in here),
            entity=first[0].entity,
            track=None,
            camera=None,
            u=None,
            v=None,
            place=first[1].place,
            room=None,
            x=round(x, 3),
            y=round(y, 3),
            revision=first[1].revision,
        )
        gone.update(r.id for r, _ in here)
        made.append(row)
        fused[row.id] = path
        if not clear:
            doubtful.add(row.id)
        if any(p.approximate for _, p in used):
            rough.add(row.id)
    return [s for s in stored if s.id not in gone] + made, fused, doubtful, rough


def derive(
    state: Path,
    start: float,
    end: float,
    place: str | None = None,
    entity: str | None = None,
    every: float | None = None,
    frame_revision: str | None = None,
    hidden: set[tuple[str, str | None, float]] | None = None,
) -> list[Position]:
    """The raw footprints of [start, end) placed, named, joined, and fused, with the zone fallback, in time order
    (then by id) across every path: what the placer writes to the layer (`advance`). Its cost grows with every raw sample of the window, so no view calls
    it: views read `positions`. `place` and `entity` keep only that place's or that entity's rows, and are applied
    before anything is placed (a place reads only its cameras' rows, and names only their tracks). `every` thins to
    one row per path per that many seconds, plus each path's last.

    `x` and `y` are in the plan frame of `revision`. With `frame_revision` (and `place`), every camera row is
    projected from its raw `u`, `v` through that revision's poses, so it lands in that plan's frame whatever night
    it was seen, and a row that can't be comes back `in_frame` False.

    A fixed camera's row is placed through the camera's pose at the row's time when the camera has a pose series in
    the map frame of the revision it's projected through (spatial/poses.py: its still stretches, smoothed, with its
    held focal and lens), so after a camera is moved or re-aimed its older rows land where it was then. Without a
    series (or one that gives no pose), it's placed through that revision's registration of the camera, whatever the
    row's age, so its older rows land as if seen from where it is now.

    An animal's footprint its camera's spot can't see on the floor by the plan (`_Places.seen`: a pet up on a
    cabinet, cast to the floor beyond a wall) is put at the furniture in the room it is seen in (`_Places.at_furniture`)
    and comes back `approximate`. A person's, and an animal's it can't put anywhere at all, gets no point; `hidden`
    collects which, as (camera, track, time)."""
    where = _Places(state)
    frame: places.Revision | None = None
    if frame_revision is not None:
        if place is None:
            raise ValueError("frame_revision needs the place it's a revision of")
        frame = where.revision(place, frame_revision)
        if frame is None:
            raise ValueError(f"{place} has no revision {frame_revision}")
    slots = None
    if place is not None:
        slots = [c for c, p in where.cams.items() if p == place] + [RIG]
    stored = samples(state, start, end, slots)
    zones = [z for z in _zone_rows(state, start, end, where) if place is None or z.place == place]
    who = identify(state, start, end, {r.track for r in [*stored, *zones] if r.track and r.entity is None})
    # The camera rows the join by where compares, on the plan through each camera's current registration (one frame
    # for all of a place's cameras): those in a second another camera of the place also had someone in view.
    cams_at: dict[tuple[str | None, int], set[str]] = {}
    for s in stored:
        if s.source == FIXED_CAMERA and s.camera and s.track:
            cams_at.setdefault((where.cams.get(s.camera), int(s.t)), set()).add(s.camera)
    placed = {
        s.id: where.place(s)
        for s in stored
        if s.source == FIXED_CAMERA and s.camera and s.track and len(cams_at[(where.cams.get(s.camera), int(s.t))]) > 1
    }
    points = {i: (p.x, p.y) for i, p in placed.items() if p.x is not None and p.y is not None}
    partner, stored_as = _partners(stored, where.cams, points)
    nots = face_nots.read(state) if any(e.startswith(f"{FACE}:") for e in who.values()) else {}
    who |= _ungrouped(partner, who, nots)
    who |= _named_by_partners(partner, who, stored_as)
    # Each person's camera track from its first row to its last, for the tracks broken in two (`handoffs`)
    spans: dict[str, _Span] = {}
    for s in sorted(stored, key=lambda r: r.t):
        place_of = where.cams.get(s.camera) if s.source == FIXED_CAMERA and s.camera and s.track else None
        if place_of is None or s.track is None or s.camera is None or animal_track(s.track) is not None:
            continue
        sp = spans.get(s.track)
        spans[s.track] = _Span(place_of, s.camera, s, s) if sp is None else replace(sp, last=s)

    def point(r: Sample) -> tuple[float, float] | None:
        p = placed.get(r.id) or where.place(r)
        return (p.x, p.y) if p.x is not None and p.y is not None else None

    handed_pairs = handoffs(spans, point)
    handed: dict[str, str] = {}
    for a, b in handed_pairs:  # an entity found so names the other track, never a third
        if a in who and b not in who:
            handed[b] = who[a]
        elif b in who and a not in who:
            handed[a] = who[b]
    who |= handed
    stored = [replace(s, entity=who.get(s.track or "")) if s.entity is None else s for s in stored]
    zones = [replace(z, entity=who.get(z.track or "")) if z.entity is None else z for z in zones]
    stored = [s for s in stored if entity is None or s.entity == entity]
    same = [(a, b) for a, b in handed_pairs if who.get(a) is not None and who.get(a) == who.get(b)]
    links = same + [(a, b) for a, b in pairs(partner) if who.get(a) == who.get(b)]
    paths = joined(stored, placed, links, nots=nots)

    seen_at: dict[str, Placed] = {}  # each row's placement in the frame asked for, made once

    def placement(s: Sample) -> Placed:
        cached = placed.get(s.id) if frame is None and s.place is None else None  # a moved row keeps its own point
        found = seen_at.get(s.id) or cached or where.place(s, frame)
        seen_at[s.id] = found
        return found

    def settle(
        rows: list[Sample], paths: Mapping[str, str]
    ) -> tuple[list[Sample], dict[str, str], set[str], set[str], dict[str, bool], dict[str, tuple[Sample, Sample]]]:
        """`rows` with each joined path fused (`_fuse`), and each person's foot on an occluding edge that no fusion
        settled put where another camera saw them that second (`_deferred`): the rows, `_fuse`'s paths, its doubtful
        and approximate rows, which rows were on an edge, and which were put elsewhere (not yet moved)."""
        rows, fused, doubtful, rough = _fuse(rows, paths, placement, where)
        edge = {
            s.id: _at_edge(where, s, found)
            for s in rows
            if s.source == FIXED_CAMERA
            and s.track
            and animal_track(s.track) is None
            and s.place is None
            and (found := placement(s)).x is not None
        }
        return rows, fused, doubtful, rough, edge, _deferred(rows, placement, edge, partner, where, nots)

    settled, fused, doubtful, rough, edge, moved = settle(stored, paths)
    # A track whose rows were put on another's is that person drawn twice: it joins that track's path, the two take one
    # entity, and the seconds both saw are fused again
    if near := _deferred_joins(moved, settled, edge, placement):
        paths = joined(stored, placed, links, near, nots)
        drawn_as = _path_entities(paths, {s.track: s.entity for s in stored if s.track})
        stored = [replace(s, entity=drawn_as[s.track]) if s.track in drawn_as else s for s in stored]
        zones = [replace(z, entity=drawn_as[z.track]) if z.track in drawn_as else z for z in zones]
        for i in fused:
            seen_at.pop(i, None)  # a fused row of the first pass may be fused again from other rows
        settled, fused, doubtful, rough, edge, moved = settle(stored, paths)
    stored = [moved[s.id][0] if s.id in moved else s for s in settled]
    for i in moved:
        seen_at.pop(i, None)  # placed again at the other camera's point
    if entity is not None:  # a track drawn as another's entity (`_path_entities`) is that entity's now
        stored = [s for s in stored if s.entity == entity]

    def path_of(s: Sample) -> object:
        return fused.get(s.id) or paths.get(s.track or "") or path_key(s.track, s.entity, s.source, s.camera, s.id)

    if every:
        stored = thin(stored, path_of, lambda s: s.t, every)
    names = _names(state, (r.entity for r in [*stored, *zones]))
    out: list[Position] = []
    for s in stored:
        found = placement(s)
        uncertain = s.id in doubtful or (edge.get(s.id, False) and s.id not in moved)
        if place is not None and found.place != place:
            continue
        out.append(
            Position(
                id=s.id,
                t=s.t,
                end=s.end,
                entity=s.entity,
                name=names.get(s.entity) if s.entity else None,
                track=s.track,
                source=s.source,
                confidence=s.confidence,
                place=found.place,
                room=found.room,
                room_id=found.room_id,
                x=found.x,
                y=found.y,
                camera=s.camera,
                u=s.u,
                v=s.v,
                revision=found.revision,
                in_frame=found.in_frame,
                path=fused.get(s.id) or paths.get(s.track or "") or s.track,
                uncertain=uncertain,
                approximate=found.approximate or s.id in rough,
            )
        )
    out += [
        replace(z, name=names.get(z.entity) if z.entity else None, path=paths.get(z.track or "") or z.track)
        for z in zones
        if entity is None or z.entity == entity
    ]
    if hidden is not None:
        hidden.update(where.hidden)
    return sorted(out, key=lambda p: (p.t, p.id))


# The layer itself: stored records, the placer that writes the cameras' and the zones', and the read.

CAMERAS_WRITER, RIG_WRITER = "cameras", "rig"  # the placer (fixed-camera and zone rows), and the rig's own writer
LAYER_MARKS = "positions.json"  # under the places folder: how far the placer has placed, and each place's generation
# The placer's piece of time, on the clock (so a rerun cuts where the last run did), and the context it reads either
# side of one to name and join the tracks that cross its ends. A day of a busy place is a few seconds of placing, so
# a piece is short enough to hold in memory and long enough that few tracks cross an end.
CHUNK_S = 6 * 3600.0
MARGIN_S = 600.0
BACKFILL_S = 30 * 86400.0  # how far back the first run places: the raw footprints' history window
# Each write also goes, thinned to one record per path per this long (and each path's last), to a second stream
# (`segment_log.PLACED_PATHS`). A read thinning at least this much (the page's Week: 15 s) scans that one: a tenth of
# the rows, since decoding rows is what a read costs. A day or less reads every record.
COARSE_S = 10.0
LAYER_STREAMS = (S.PLACED, S.PLACED_PATHS)
READ_S = 86400.0  # a read scans this much at a time, and asks between scans whether anyone still wants it
# A place is placed again when its cameras' registration now puts people somewhere else than when it was last
# placed: when, for any of its cameras, the median move of the plan points REFERENCE pixels land on (`_basis`) is
# over REPLACE_M, or a pixel that landed on the floor no longer does. The median, so a pixel whose ray grazes the
# floor (a tenth of a degree moves it metres) doesn't decide. Paths are good to about half a metre between two
# cameras (see FUSE_M), so a quarter of a metre is a move worth a night's re-placing and a nightly refit's few
# centimetres isn't.
REPLACE_M = 0.25
REFERENCE = tuple((u, v) for v in (0.7, 0.85, 1.0) for u in (0.25, 0.5, 0.75))  # (u, v), the frame's lower part
_ALWAYS = 4102444800.0  # 2100-01-01: past any record, for a forget of everything a slot holds


class Cancelled(Exception):
    """A read stopped because its caller said nobody is waiting for it any more."""


def layer_slot(place: str, source: str, generation: int = 0) -> str:
    """The store slot a place's records from `source` are in: one per writer, so a read of a place opens only that
    place's files. The placer's (fixed-camera and zone) carries the place's generation (`place_again`); the rig's
    doesn't, since nothing can make its records again."""
    if source == RIG_POSE:
        return S.slot(f"{place}.{RIG_WRITER}")
    return S.slot(f"{place}.{CAMERAS_WRITER}.{generation}")


def _marks(state: Path) -> dict[str, Any]:
    try:
        found = json.loads((places.root(state) / LAYER_MARKS).read_text())
    except FileNotFoundError, ValueError:
        return {}
    return found if isinstance(found, dict) else {}


def _generation(marks: Mapping[str, Any], place: str) -> int:
    """How many times `place` has been placed again (`place_again`): its placer slot's generation."""
    return int((marks.get("generations") or {}).get(place, 0))


def _save_marks(state: Path, marks: Mapping[str, Any]) -> None:
    path = places.root(state) / LAYER_MARKS
    json_state.write(path, marks, indent=1, sort_keys=True)


def _line(r: Mapping[str, Any]) -> str:
    """What one drawn line is, for thinning: a record's path, else its track, else its entity; a zone record (a
    visit's whole span) or one with none of those is its own."""
    if r["source"] == ZONE:
        return f"row:{r['id']}"
    return r["path"] or r["track"] or r["entity"] or f"row:{r['id']}"


Seqs = dict[str, dict[str, int]]  # each layer stream's slots' next row numbers


def write(state: Path, rows: Sequence[Position], resume: Mapping[str, Mapping[str, int]] | None = None) -> Seqs:
    """Append records to the layer: what every position source calls. A record with no place can't be read by place
    or drawn, so it's left out. Each call's records also go thinned to the paths stream (see COARSE_S), so a source
    writes a stretch of time at once, not a record at a time (a lone record is always its path's last, and kept).
    `resume` is the placer's: each of its slots' next row number as its last finished piece left it, so rows a piece
    wrote before it failed are dropped when it's written again. Returns each stream's slots' next row numbers."""
    marks = _marks(state)
    by_slot: dict[str, list[dict[str, Any]]] = {}
    for p in rows:
        if p.place is None:
            continue
        by_slot.setdefault(layer_slot(p.place, p.source, _generation(marks, p.place)), []).append(
            {
                "id": p.id.removeprefix("fused:"),  # a fused row stands in for the camera rows it was made from
                "ts_start": p.t,
                "ts_end": max(p.t, p.end),
                "source": p.source,
                "confidence": p.confidence,
                "entity": p.entity,
                "track": p.track,
                "path": p.path,
                "place": p.place,
                "room": p.room,
                "room_id": p.room_id,
                "x": p.x,
                "y": p.y,
                "revision": p.revision,
                "camera": p.camera,
                "uncertain": p.uncertain,
                "approximate": p.approximate,
            }
        )
    if not by_slot:
        return {s: dict(slots) for s, slots in (resume or {}).items()}
    writers = S.Writers(S.root(state), resume)
    try:
        for slot, records in by_slot.items():
            writers.append(S.PLACED, slot, records)
            writers.append(S.PLACED_PATHS, slot, thin(records, _line, lambda r: r["ts_start"], COARSE_S))
        return {s: slots for s, slots in writers.sync().items() if s in LAYER_STREAMS}
    finally:
        writers.close()


def _piece(
    state: Path, held: places.Place, lo: float, hi: float, behind: collections.Counter[tuple[str, str]] | None = None
) -> list[Position]:
    """One place's records for [lo, hi): its raw footprints and zone visits placed in the frame its plan is drawn in
    (`derive`), read with MARGIN_S of context either side. `behind` counts the footprints of [lo, hi) left unplaced
    behind a wall from their camera (`_Places.seen`, and nothing came of putting them at the furniture), by camera and
    kind ("person" or "animal")."""
    hidden: set[tuple[str, str | None, float]] = set()
    try:
        rows = derive(
            state, lo - MARGIN_S, hi + MARGIN_S, place=held.id, frame_revision=held.plan_revision, hidden=hidden
        )
    except ValueError as e:  # the plan names a revision the place no longer has: its current frame, then
        LOG.warning("placing %s in its current frame: %s", held.id, e)
        rows = derive(state, lo - MARGIN_S, hi + MARGIN_S, place=held.id, hidden=hidden)
    if behind is not None:
        behind.update((camera, _kind_of(track)) for camera, track, t in hidden if lo <= t < hi)
    return [r for r in rows if lo <= r.t < hi]


def _kind_of(track: str | None) -> str:
    """A footprint's subject: "animal" for an animal's track, else "person"."""
    return "animal" if animal_track(track) is not None else "person"


def behind_walls(counts: Mapping[tuple[str, str], int]) -> str:
    """Footprints left unplaced behind a wall and nowhere to put them (an animal's are put at the furniture, so only one
    with no furniture point is counted), by camera, in words: "cam0 (people 3, animals 4)"."""
    cams = sorted({c for c, _ in counts})
    return "; ".join(
        f"{c} (people {counts.get((c, 'person'), 0)}, animals {counts.get((c, 'animal'), 0)})" for c in cams
    )


Basis = dict[str, list[list[float] | None]]  # each camera's REFERENCE pixels' plan points; None for no floor point


def _basis(where: _Places, held: places.Place, at: float) -> Basis:
    """Where `held`'s cameras put people at time `at`: the plan point each REFERENCE pixel of each camera lands on,
    placed exactly as the placer places a person's footprint (`_Places.place`, in the frame the place's plan is drawn
    in). It moves when a registration changes a camera's pose, its lens, the floor, or the plan's frame, and not when
    a new revision only carries the same registration over. `where` is one read of the places, shared by every place
    the caller asks about: the spots and the owner plans are the same read for each of them."""
    frame = where.revision(held.id, held.plan_revision) if held.plan_revision is not None else None
    out: Basis = {}
    for camera in sorted(c for c, p in where.cams.items() if p == held.id):
        points: list[list[float] | None] = []
        for u, v in REFERENCE:
            sample = Sample(
                "", at, at, FIXED_CAMERA, 1.0, None, f"{camera}:0", camera, u, v, None, None, None, None, None
            )
            found = where.place(sample, frame)
            points.append(None if found.x is None or found.y is None or not found.in_frame else [found.x, found.y])
        out[camera] = points
    return out


def _moved(was: Basis, now: Basis) -> bool:
    """Whether a registration moved where a place's cameras put people since `was` (see REPLACE_M). A camera only one
    of the two has changes nothing already placed: a new camera has no records yet, and a camera gone has no new pose
    to place its records through."""
    for camera in was.keys() & now.keys():
        pairs = list(zip(was[camera], now[camera]))
        if any((a is None) != (b is None) for a, b in pairs):
            return True
        moves = [math.dist(a, b) for a, b in pairs if a is not None and b is not None]
        if moves and statistics.median(moves) > REPLACE_M:
            return True
    return False


def place_again(state: Path, place: str, upto: float) -> int:
    """Have the placer place `place` again from its raw footprints: its records so far are forgotten, its slot moves
    to a new generation (the store hides rows appended into a forgotten range), and it is marked BACKFILL_S behind
    `upto`, which the next `advance` makes up before anything else. Other places, and the rig's records, are left as
    they are. Returns the place's new generation."""
    from continuum_py import forget_segments

    marks = _marks(state)
    generation = _generation(marks, place)
    root = S.root(state)
    old = layer_slot(place, FIXED_CAMERA, generation)
    streams = [s for s in (*LAYER_STREAMS, *map(S.condensed, LAYER_STREAMS)) if (root / s / old).is_dir()]
    if streams:
        forget_segments(root, 0, S.ns(_ALWAYS), streams, [old], "reset")
    marks["generations"] = {**(marks.get("generations") or {}), place: generation + 1}
    marks["behind"] = {**(marks.get("behind") or {}), place: upto - BACKFILL_S}
    _save_marks(state, marks)
    return generation + 1


def advance(state: Path, upto: float, chunk_s: float = CHUNK_S) -> dict[str, int]:
    """Place what was recorded since the last run, up to `upto` (the time the raw footprints are recorded to), and
    write it to the layer: every place's fixed-camera and zone records, a piece of `chunk_s` at a time, oldest first.
    The first run goes BACKFILL_S back. It runs after a place's cameras may have been registered again, so it first
    checks each place's registration against the one its records were placed through (`_basis`, `_moved`), and a
    place whose cameras now put people somewhere else is placed again from scratch (`place_again`) before the new time
    is placed. The mark moves after each piece is written, so a run that stops part way picks up at the piece it was
    on, and that piece's rows aren't kept twice (`write`'s `resume`). Returns how many records and pieces it wrote
    and how many places it placed again.

    Only time after the mark is placed: a footprint or a zone visit recorded later for a time before it (a camera
    that caught up after an outage, a visit that began before the mark and closed after) isn't in the layer until
    its place is placed again."""
    counts = {"records": 0, "pieces": 0, "replaced": 0, "behind_walls": 0}
    behind_wall: collections.Counter[tuple[str, str]] = collections.Counter()
    held = {p.id: p for p in places.places(state)}
    where = _Places(state)  # one read of the places, for the spots and every place's basis
    lacking = {c: s.missing for c, s in where.spots.items() if s.missing}
    if lacking:
        LOG.info(
            "cameras with a spot on a plan that can't place anyone yet: %s",
            "; ".join(f"{c} ({', '.join(m)})" for c, m in sorted(lacking.items())),
        )
    if not places.camera_places(state):
        LOG.info("no camera has a spot on a plan or a pose in a map: no footprint is placed until one does")
    marks = _marks(state)
    first = marks.get("placed_to") is None
    was: dict[str, Basis] = marks.get("basis") or {}
    now = {place: _basis(where, p, upto) for place, p in held.items()}
    basis: dict[str, Basis] = {}
    for place in sorted(now):
        if not first and place in was and _moved(was[place], now[place]):
            LOG.info("placing %s again: its cameras' registration moved where people land on the plan", place)
            place_again(state, place, upto)
            counts["replaced"] += 1
            basis[place] = now[place]
        else:
            # What its records were placed through stays the measure (with any camera new since), so refits of a few
            # centimetres a night add up to a move, where a basis renewed each night would never see one.
            basis[place] = {**now[place], **was.get(place, {})} if not first else now[place]
    marks = _marks(state)  # read again: `place_again` above writes the marks it leaves a place's redo at
    # Once a place is marked behind (or there is nothing to redo), its records and its basis agree: a run that stops
    # part way goes on from the mark, without finding the place moved again.
    marks["basis"] = basis
    _save_marks(state, marks)
    seq: Seqs = {s: {str(k): int(v) for k, v in slots.items()} for s, slots in (marks.get("seq") or {}).items()}
    placed_to = float(marks["placed_to"]) if marks.get("placed_to") is not None else upto - BACKFILL_S

    def piece(only: places.Place | None, at: float, end: float) -> float:
        """Place one piece from `at` (of `only`, a place being placed again, or of every place that isn't behind),
        write it, and move the mark; returns where the piece ended, always past `at`."""
        nonlocal seq, marks
        hi = min(end, (math.floor(at / chunk_s) + 1) * chunk_s)
        behind = marks.get("behind") or {}
        todo = [only] if only is not None else [p for p in held.values() if p.id not in behind]
        rows = [r for p in todo for r in _piece(state, p, at, hi, behind_wall)]
        seq = write(state, rows, resume=seq)
        marks = {**marks, "seq": seq}
        if only is None:
            marks["placed_to"] = hi
        elif hi < end:
            marks["behind"] = {**behind, only.id: hi}
        else:
            marks["behind"] = {k: v for k, v in behind.items() if k != only.id}
        _save_marks(state, marks)
        counts["records"] += len(rows)
        counts["pieces"] += 1
        return hi

    for place, start in sorted((marks.get("behind") or {}).items()):
        at = float(start)
        if place not in held or at >= placed_to:  # a place since removed, or nothing placed yet to make up
            marks["behind"] = {k: v for k, v in (marks.get("behind") or {}).items() if k != place}
            _save_marks(state, marks)
            continue
        while at < placed_to:
            at = piece(held[place], at, placed_to)
    at = placed_to
    while at < upto:
        at = piece(None, at, upto)
    counts["behind_walls"] = sum(behind_wall.values())
    if behind_wall:
        LOG.info("left unplaced, behind a wall from their camera's spot: %s", behind_walls(behind_wall))
    return counts


def dry_run(state: Path, cams: Sequence[str], now: float, hours: float) -> list[dict[str, Any]]:
    """What a run would place, changing nothing: for each of `cams`, the footprints its pose tracks hold over the
    last `hours` (read as `footprints.record` would record them), how many land on its place's floor, how many of
    those in a room of the plan, how many a camera with a spot left unplaced because its spot can't see their floor
    point by the plan and it couldn't be put at the furniture either (`behind_wall`, people and animals apart,
    `_Places.seen` and `_Places.at_furniture`; None for a camera with no spot), the aim it was read through (fitted
    now when it has none stored, and not kept), and what the camera lacks. One row per camera, in order.

    `behind_wall` is the check that a camera's positions are at the plan's scale: rooms tile a home, so a point a
    pose puts metres too far is still in a room, but in one behind a wall. People stand on the floor, so many of a
    camera's people there means its pose or the plan is off; an animal there is mostly one up on furniture, and is
    now put at the furniture rather than left out, so its count is the ones it couldn't place at all."""
    aims = spots.fit(state, now, save=False)
    where = _Places(state, aims)
    feet = footprints.collect(state, now - hours * 3600.0, now - footprints.FLUSH_S, list(cams))
    out = []
    for camera in cams:
        rows = placed = roomed = people = 0
        spot = where.spots.get(camera)
        for f in feet:
            if f.camera != camera:
                continue
            rows += 1
            track = f"{camera}:{f.track}"
            sample = Sample("", f.t, f.t, FIXED_CAMERA, f.confidence, None, track, camera, f.u, f.v, *[None] * 5)
            found = where.place(sample)
            placed += found.x is not None
            roomed += found.room is not None
            people += found.x is not None and _kind_of(track) == "person"
        behind: collections.Counter[str] = collections.Counter(_kind_of(t) for c, t, _ in where.hidden if c == camera)
        out.append(
            {
                "camera": camera,
                "place": where.cams.get(camera),
                "by": "spot" if spot is not None else ("map" if camera in where.cams else None),
                "footprints": rows,
                "placed": placed,
                "in_a_room": roomed,
                "people_placed": people,
                "behind_wall": {"person": behind["person"], "animal": behind["animal"]}
                if spot is not None and where.sight(spot.place) is not None
                else None,
                "missing": list(spot.missing) if spot is not None else [],
                "aim": spot.aim if spot is not None else None,
            }
        )
    return out


def placing(state: Path, place: str) -> list[dict[str, Any]]:
    """The cameras that sit in `place`, by name, and whether each can place the people it sees: `by` is "spot" for a
    camera read at the owner's spot on the plan and "map" for one seen in the place's map, and `missing` is what it
    lacks (empty when it places): what a spot lacks for a pose (spatial/spots.py's `Spot.missing`). For a reader to
    say what's missing per camera instead of showing nobody."""
    found = spots.read(state)
    here = {c for c, at in places.camera_places(state).items() if at == place}
    out = []
    for camera in sorted(here):
        spot = found.get(camera)
        if spot is not None and spot.place == place:
            out.append({"camera": camera, "by": "spot", "missing": list(spot.missing)})
        elif spot is None:  # a camera with a spot on another place's plan is that place's
            out.append({"camera": camera, "by": "map", "missing": []})
    return out


def reset(state: Path) -> int:
    """Have the placer place every place again from the raw footprints (after a pose series, a plan, or a label
    changed what the stored records should say; a registration that moves a camera is noticed by `advance` itself):
    `place_again` for each place. The rig's records stay: nothing can make them again. Returns how many places."""
    marks = _marks(state)
    upto = float(marks["placed_to"]) if marks.get("placed_to") is not None else 0.0
    ids = [p.id for p in places.places(state)]
    for place in ids:
        place_again(state, place, upto)
    return len(ids)


def _scan(state: Path, lo: float, hi: float, slots: Sequence[str] | None, paths: bool = False) -> list[dict[str, Any]]:
    """The layer's rows whose span reaches into [lo, hi), as the store's columns: every record's stream (the thinned
    paths' with `paths`) and its condensed history's (retention.py), for `slots` (every slot when None)."""
    from continuum_py import read_segments

    root = S.root(state)
    out = []
    held = S.PLACED_PATHS if paths else S.PLACED
    for stream in (held, S.condensed(held)):
        if (root / stream).is_dir():
            cams = None if slots is None else list(slots)
            out.append(read_segments(root, stream, S.PLACED_SCHEMA, S.ns(lo), S.ns(hi), cams, overlapping=True))
    return out


def _place_slots(state: Path, place: str) -> list[str]:
    return [layer_slot(place, FIXED_CAMERA, _generation(_marks(state), place)), layer_slot(place, RIG_POSE)]


def last_seen(state: Path, ids: Iterable[str], now: float, lookbacks: Sequence[float]) -> dict[str, float]:
    """When anyone was last placed in each of the places `ids`, by the layer's newest record there. Read newest
    first over the widening `lookbacks` (seconds before `now`), each place only until it's found: a place seen in the
    last hour costs an hour's read, and one not seen in the longest is left out."""
    found: dict[str, float] = {}
    upto = now
    for back in lookbacks:
        lo = now - back
        for place in ids:
            if place in found:
                continue
            scans = _scan(state, lo, upto, _place_slots(state, place), paths=True)  # a path's last is always kept
            ends = [got["end_ns"] for got in scans if len(got["end_ns"])]
            if ends:
                found[place] = min(max(float(np.max(e)) for e in ends) / S.NS, now)
        upto = lo
    return found


_AS_STORED = {"camera": "seen_by", "revision": "place_revision"}  # the store keeps those two names for its own
_TEXT = ("id", "entity", "track", "source", "place", "room", "room_id", "path")


def _columns(got: Mapping[str, Any]) -> dict[str, list[Any]]:
    """One scan's columns as lists, by `Position`'s field names: a read goes over every row of them, and a list is the
    fastest thing to go over."""
    cols: dict[str, list[Any]] = {name: got[name] for name in _TEXT}
    cols |= {name: got[column] for name, column in _AS_STORED.items()}
    cols["t"] = (np.asarray(got["ts_ns"], dtype=np.int64) / S.NS).tolist()
    cols["end"] = (np.asarray(got["end_ns"], dtype=np.int64) / S.NS).tolist()
    for name in ("x", "y", "confidence", "uncertain", "approximate"):
        cols[name] = np.asarray(got[name]).tolist()
    return cols


def _record(cols: Mapping[str, list[Any]], i: int) -> dict[str, Any]:
    """Row `i` of a scan as `Position`'s fields (all but its name and its raw u and v)."""
    row = {name: values[i] for name, values in cols.items()}
    row["id"] = str(row["id"])
    row["x"] = None if math.isnan(row["x"]) else row["x"]
    row["y"] = None if math.isnan(row["y"]) else row["y"]
    row["uncertain"] = bool(row["uncertain"])
    row["approximate"] = bool(row["approximate"])
    return row


def _to_frame(
    where: _Places, frame: places.Revision, row: dict[str, Any], moves: dict[str | None, floorplan.Similarity | bool]
) -> dict[str, Any]:
    """A stored record as `frame` (a revision of its place) draws it: as it is when it was placed in that plan frame,
    moved by the similarity between the two frames when it was placed in another (`_Places.carried`), and `in_frame`
    False when nothing relates them. A room-only record has no point to move."""
    if row["x"] is None or row["y"] is None or row["revision"] == frame.id:
        return row
    key = row["revision"]
    if key not in moves:
        rev = where.revision(frame.place, key)
        if rev is None:
            moves[key] = False
        elif places.same_plan_frame(rev, frame):
            moves[key] = True
        else:
            moves[key] = where.carried(rev, frame) or False
    move = moves[key]
    if move is True:
        return row
    if move is False:
        return {**row, "in_frame": False}
    x, y = move.point(row["x"], row["y"])
    x, y = round(x, 3), round(y, 3)
    room = where.room(frame, x, y)
    moved = {"x": x, "y": y, "revision": frame.id, "room_id": room[0] if room else None}
    return {**row, **moved, "room": room[1] if room else None}


def positions(
    state: Path,
    start: float,
    end: float,
    place: str | None = None,
    entity: str | None = None,
    every: float | None = None,
    frame_revision: str | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> list[Position]:
    """Where entities were in [start, end), in time order (then by id): the layer's records as they were stored,
    named as people and pets are named now. `place` reads only that place's records and `entity` keeps only that
    entity's. `every` thins a long read as it scans: one record per path per that many seconds, plus each path's
    last (zone records, each a visit's whole span, are all kept), so a week of a busy place builds the few thousand
    records a view draws, not one per second; from COARSE_S up it scans the stream thinned that much when written. `cancelled` is asked before each READ_S of the window is scanned; when
    it says yes the read raises `Cancelled`.

    For a view: group records into paths by `path`, else `entity`, and skip those whose `x` is None when drawing
    (zone records, records of a place with no plan yet, points above the horizon). `x` and `y` are in the plan frame
    of `revision`. To draw on a plan, pass its frame as `frame_revision` (`places.Place.plan_revision`, with
    `place`): a record placed in another frame of the place is moved onto it (`_to_frame`), and one that can't be
    comes back `in_frame` False (skip or mark it)."""
    where: _Places | None = None
    frame: places.Revision | None = None
    if frame_revision is not None:
        if place is None:
            raise ValueError("frame_revision needs the place it's a revision of")
        where = _Places(state)
        frame = where.revision(place, frame_revision)
        if frame is None:
            raise ValueError(f"{place} has no revision {frame_revision}")
    slots = _place_slots(state, place) if place is not None else None
    kept: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()  # (path, the `every` it's in) already kept
    newest: dict[str, float] = {}  # each path's last record's time so far
    tail: dict[str, dict[str, Any] | None] = {}  # each path's last record, when thinning left it out
    at = start
    while at < end:
        if cancelled is not None and cancelled():
            raise Cancelled
        hi = min(end, at + READ_S)
        for got in _scan(state, at, hi, slots, paths=bool(every and every >= COARSE_S)):
            cols = _columns(got)
            times, sources, entities = cols["t"], cols["source"], cols["entity"]
            last: dict[str, int] = {}  # this scan's rows that are their path's last so far; -1 once kept
            for i, t in enumerate(times):
                # A span that began before this scan's piece of the window is the earlier piece's (or, begun before
                # the window, the first's): each record is taken once.
                if not at <= max(t, start) < hi or (entity is not None and entities[i] != entity):
                    continue
                if not every or sources[i] == ZONE:
                    kept.append(_record(cols, i))
                    continue
                key = cols["path"][i] or cols["track"][i] or entities[i] or f"row:{cols['id'][i]}"
                bucket = (key, int(t // every))
                fresh = bucket not in seen
                if fresh:
                    seen.add(bucket)
                    kept.append(_record(cols, i))
                if t >= newest.get(key, -math.inf):
                    newest[key] = t
                    last[key] = -1 if fresh else i
            for key, i in last.items():
                tail[key] = None if i < 0 else _record(cols, i)
        at = hi
    kept += [r for r in tail.values() if r is not None]
    if frame is not None and where is not None:
        moves: dict[str | None, floorplan.Similarity | bool] = {}
        kept = [_to_frame(where, frame, r, moves) for r in kept]
    names = _names(state, {r["entity"] for r in kept})
    out = [Position(**r, name=names.get(r["entity"]) if r["entity"] else None, u=None, v=None) for r in kept]
    return sorted(out, key=lambda p: (p.t, p.id))


# The job: `pack-positions`, hourly (services/services.toml) and by hand.

RUN_LOCK = "positions.run"  # under the places folder: one run at a time
# A dry run flags a camera with more of its people than 1 - this behind a wall from its spot (`dry_run`).
# On the Linux machine four cameras whose spots and tilts held put 2 to 8% of their people there; one read through
# the wrong focal put 73% of all its points there.
VIEW_SHARE = 0.9
#: What a camera lacks before it can place anyone, in words (`placing`'s `missing`).
LACKS = {
    "height": "its spot has no height",
    "facing": "its spot has no facing (yaw)",
    "lens": "it has no held lens and its spot has no fov",
    "tilt": "too few people seen walking to work out its tilt",
    "check": "people come out the wrong height from its spot: its height, lens, or tilt is off",
}


def run(state: Path, fixed: Sequence[str], now: float, first_s: float = footprints.FIRST_S) -> dict[str, int]:
    """One run of the job: fit the tilt of each camera with a spot that has none (spatial/spots.py), record `fixed`'s
    raw footprints since the last run (`footprints.record`; `first_s` back for a camera never recorded), and place
    them (`advance`). Returns `advance`'s counts with how many footprints were recorded."""
    with places.writing(state):
        spots.fit(state, now)
    recorded = footprints.record(state, fixed, now, first_s)
    return {**advance(state, now - footprints.FLUSH_S), "recorded": recorded}


def _print_dry_run(state: Path, fixed: Sequence[str], hours: float) -> None:
    print(f"dry run over the last {hours:g} h: nothing is recorded, placed, or kept")
    by = {"spot": "its spot on the plan", "map": "its pose in the map", None: "nothing: no spot and no pose"}
    for row in dry_run(state, fixed, time.time(), hours):
        aim = row["aim"]
        tilt = (
            ""
            if aim is None
            else f"; tilted {aim.pitch:.1f} deg down, rolled {aim.roll:.1f}, from {aim.people} people walking"
            + ("" if aim.tall is None else f" who come out {aim.tall:.2f} m tall")
        )
        lacks = "".join(f"; {LACKS[m]}" for m in row["missing"])
        view = ""
        if row["behind_wall"] is not None:
            people, animals = row["behind_wall"]["person"], row["behind_wall"]["animal"]
            view = f"; left unplaced behind a wall from its spot: people {people:,}, animals {animals:,}"
            if people > (1 - VIEW_SHARE) * (row["people_placed"] + people):
                view += (
                    " (too many people: its height, tilt, lens, or spot is off, or the plan is drawn at another scale)"
                )
        print(
            f"  {row['camera']}: {row['footprints']:,} footprints, {row['placed']:,} placed on the floor, "
            f"{row['in_a_room']:,} in a room of the plan{view}; read through {by[row['by']]}{tilt}{lacks}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    """`pack-positions`: places the people and pets the fixed cameras saw on each place's plan. A camera is read at
    the spot the owner gave it on the plan, else through its pose in the place's map; one with neither places
    nobody, and the run says what each camera lacks. It runs every hour, and by hand:

      pack-positions --dry-run   # what a run would place per camera over the last day; changes nothing
      pack-positions             # record and place what was seen since the last run
      pack-positions --days 7    # a camera never recorded is read 7 days back (the most), not one
      pack-positions --reset     # forget the placed records and place the last 30 days again

    `--reset` is for after a plan's rooms or a name changed what stored records should say; a spot or a registration
    that moves where a camera puts people needs none, since a run notices and places that place again. A busy place
    is about 20 s of placing for each day."""
    import argparse
    import fcntl

    from liberational import config as cfg

    parser = argparse.ArgumentParser(
        prog="pack-positions", description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--reset", action="store_true", help="forget the placed records and place everything again")
    parser.add_argument("--dry-run", action="store_true", help="print what a run would place; change nothing")
    parser.add_argument("--hours", type=float, default=24.0, help="how far back a dry run reads (default 24)")
    parser.add_argument(
        "--days", type=float, default=footprints.FIRST_S / 86400, help="how far back a camera's first recording reaches"
    )
    args = parser.parse_args(argv)
    logs.setup()
    state = cfg.path("paths.state")
    fixed = cameras.fixed_cameras(cfg.devices_file())
    if args.dry_run:
        return _print_dry_run(state, fixed, args.hours)
    places.root(state).mkdir(parents=True, exist_ok=True)
    with (places.root(state) / RUN_LOCK).open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            LOG.info("another run is placing positions now; leaving it to finish")
            return
        if args.reset:
            LOG.info("placed records of %d places forgotten; placing them again", reset(state))
        began = time.monotonic()
        did = run(state, fixed, time.time(), args.days * 86400)
        for place in places.places(state):
            for note in placing(state, place.id):
                if note["missing"]:
                    LOG.info("%s places nobody yet: %s", note["camera"], "; ".join(LACKS[m] for m in note["missing"]))
        LOG.info(
            "recorded %d raw footprints; placed %d records in %d pieces%s%s, %.0f s",
            did["recorded"],
            did["records"],
            did["pieces"],
            f", {did['replaced']} places placed again after a camera moved" if did["replaced"] else "",
            f", {did['behind_walls']} footprints left unplaced behind walls" if did["behind_walls"] else "",
            time.monotonic() - began,
        )


if __name__ == "__main__":
    main()
