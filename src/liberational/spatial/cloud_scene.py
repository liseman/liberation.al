"""One scene of a place's cameras: each camera's own point cloud, put where its camera is (#445).

spatial/camera_clouds.py gives every camera a cloud of its own view. Here those clouds are **placed**, never blended:
the scene is a list of clouds with a transform each, and every point stays its camera's. Three things decide where a
cloud goes, in this order.

**1. The owner's spot** (`spots_of`). A camera's spot on the owner's plan says where it is and which way it faces.
The cloud is level already (its own floor is z = 0, its camera on the z axis looking along +y), so placing it is a
turn about the vertical and a shift: the camera lands on its spot, facing its facing, the cloud's floor on the plan's.
The scene's frame is the plan's metres, right-handed with z up: the plan's own frame when that is right-handed, else
the plan's with y turned over (a page is drawn x east, y south). A camera with no spot, or a spot with no facing, is
not in the scene.

**2. A pair of cameras that see the same space** (`Match`, `confident`, `solve`). When a camera's cloud is made, its
picture is compared with the other cameras' pictures by MASt3R two-view matching (spatial/pointmap_runner.py), the
nearest by their spots first, at most `MAX_OTHERS`. A match that is confident says where one camera stands as seen
from the other. Confident is measured, not judged: on six cameras on the Linux machine,
fourteen pairs of different space had no pixel MASt3R was sure of in the other's frame and at most 10 percent matched;
the one pair that shares a room had 7 and 63 percent sure, 19 and 20 percent matched, and its two directions agreed to
6 degrees and 0.2 m over 2.8 m. So a match counts when each direction is sure of `MIN_SURE` of the pixels, the
matcher finds `MIN_MATCH` both ways, and the two directions agree within `AGREE`. A confident pair pulls its two
cameras towards what the pictures say, each staying near its own spot: the spots are the reference, and a match only
adjusts cameras against each other (`solve` weighs a match `EDGE` times a spot, so two cameras share the correction
and the scene stays on the plan). A pair that matched in daylight often doesn't at night: while neither camera has
moved and neither spot has changed, the last confident measurement is the one used, and a look that isn't confident
doesn't replace it. A correction is bounded (`BOUND`): a match that would move a camera farther says
that one of the two spots is off, is shown, and moves nothing. Nothing here writes a spot or the plan: corrections
live in `pairs.json`, and one made from another picture or another spot is not used. Only cameras standing on their
own spots are tightened this way: a followed camera is placed by the plan's walls.

**3. A camera that was turned or carried is followed** (`record`, `align`, `Follow`). Cameras move: one is re-aimed
on its shelf, one rides on a backpack. Any of them may, at any time, several at once; none is taken to be fixed. A
camera's `anchor` is the picture it was last placed with. Each new picture is compared with it, and one that
registers within `STILL` hasn't moved. Otherwise the camera is placed again, without asking, from the first of
these that answers:

- *its pictures*: the new picture registers with the anchor farther than `STILL` (it overlaps its old view), or
  with another camera's picture, and the two-view measurement says where it stands now;
- *its own cloud* (spatial/cloud_align.py `locate`): the cloud's walls against the plan's, with no limit on the turn
  or the distance: where it stood facing another way, or anywhere its walls fit. `locate` says what makes an answer
  confident.

The answer is the camera's **working pose**, kept beside its cloud (`follow.json`) and used in place of the spot;
the cloud is then squared and fitted from it as any cloud is. The owner's spot and plan are never written. The same
is done for a camera whose pictures didn't change but whose walls line up with the plan only far from its spot's
facing (a spot set facing the wrong way). A view that can't be placed with confidence (no wall in view, walls that
fit nowhere or in two places) is not drawn: the camera is "lost", says why, and keeps the working pose it had for
when its old view comes back; every new cloud is tried again. The owner can put a followed or lost camera back on
the plan's spot (`use_spot`), which holds until its view next changes, and take that back (`follow_again`).

*While a camera is being carried* no cloud of its own is made: one is made only from a view that has held still
with nobody in it (spatial/surroundings_clouds.py). Meanwhile spatial/camera_track.py tracks it from its live frames
(#516) and keeps its pose and its path beside its cloud (`tracked`). The scene shows the camera where the track has
it and says how long ago the track was held to the plan's walls; the keyframes the walls held go into the place's
map as the camera's own clouds do. When the camera rests and gets a cloud, that cloud is placed from where the track left it
("tracking", then squared and fitted as any cloud is), and the track's files go.

**Square to the house** (`align`, spatial/cloud_align.py). Between 1 and 2: a cloud's own walls say how its camera
really faces, better than a facing set by hand. A camera placed from its spot is turned, and where its walls have
clear counterparts slid, so its walls run along the plan's (bounded; a rigid move). A camera squared this way holds:
the plan is the reference, so a pair's measurement no longer moves it, only a partner that couldn't be squared, and
between two squared cameras the measurement is a check that says how far off it is ("checked"). The squaring is
kept beside the cloud and used only while it is of the same cloud, spot, and plan.

**One map, not a cloud a camera** (spatial/cloud_map.py, #546). What the scene draws is the place's map: every
placed cloud is offered to it (`align`), agrees with what it holds or is refused, and is added up to a density, so
a surface two shots saw is there once. The rows say what the map took of each camera's latest shot.

What a scene is good for: reliefs standing in the right places, not a whole room. Clouds that overlap meet with a seam
(the pair above is right to tens of centimetres), and cameras that share no space are only as right as their spots.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import shutil
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from liberational import json_state, logs
from liberational.memory import places
from liberational.spatial import camera_clouds, cloud_align, cloud_map, cloud_merge, ply

LOG = logs.get("camera-clouds")

MAX_OTHERS = 5  # the other cameras one picture is compared with, the nearest by their spots first
MIN_SURE = 0.03  # of the pixels MASt3R must be sure of in both frames, each way; different space gives 0
MIN_MATCH = 0.15  # of the pixels the matcher must find, one way or the other; different space gave at most 0.10
AGREE = (
    8.0,
    0.25,
    0.10,
    0.10,
)  # the two directions agree within: degrees, metres or this share of the distance, log scale
STILL = (0.3, 4.0)  # metres and degrees: a camera's own pictures this close are one position
BOUND = (0.75, 20.0)  # metres and degrees a match may move a camera from its spot
EDGE = 10.0  # how much a confident match weighs against a spot
FIRM = 10.0  # and how much a camera squared to the plan weighs against a match
TURN_M = 1.5  # metres a radian of facing counts as, when a shift and a turn are weighed together
TINY = (0.01, 0.5)  # metres and degrees: a correction under both isn't one
PAIRS = "pairs.json"


@dataclass(frozen=True)
class At:
    """Where a camera stands on the floor of the scene: x and y in metres, and the bearing it faces (radians,
    clockwise from +y seen from above, so its forward is (sin, cos))."""

    x: float
    y: float
    bearing: float

    def step(self, right: float, ahead: float, turn: float) -> At:
        """The position `right` and `ahead` of this one in its own axes, turned `turn` more."""
        s, c = math.sin(self.bearing), math.cos(self.bearing)
        return At(self.x + right * c + ahead * s, self.y - right * s + ahead * c, self.bearing + turn)

    def seen_from(self, other: At) -> tuple[float, float, float]:
        """This position in `other`'s axes: to its right, ahead of it, and turned from it (-pi to pi)."""
        dx, dy = self.x - other.x, self.y - other.y
        s, c = math.sin(other.bearing), math.cos(other.bearing)
        return dx * c - dy * s, dx * s + dy * c, wrap(self.bearing - other.bearing)


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class SpotAt:
    """A camera's owner spot, as the scene reads it."""

    camera: str
    place: str
    at: At | None  # None for a spot with no facing
    plan: tuple[float, float]  # the spot in the plan's own coordinates
    mirrored: bool  # the plan's frame is right-handed (the scene's own); else its y is turned over
    basis: str  # changes when the owner moves or turns the spot


def scene_at(x: float, y: float, yaw: float, mirrored: bool) -> At:
    """A spot (plan x and y, facing in degrees clockwise from up the page) in the scene's frame."""
    r = math.radians(yaw)
    ahead = (math.sin(r), -math.cos(r))  # on the plan
    if not mirrored:
        return At(x, -y, math.atan2(ahead[0], -ahead[1]))
    return At(x, y, math.atan2(ahead[0], ahead[1]))


def spots_of(state: Path) -> dict[str, SpotAt]:
    """Every camera's spot on an owner plan that holds (spatial/spots.py `read`), by camera."""
    from liberational.spatial import spots

    out = {}
    for camera, spot in spots.read(state).items():
        at = None if spot.yaw is None else scene_at(spot.at[0], spot.at[1], spot.yaw, spot.mirrored)
        basis = f"{spot.place}:{spot.at[0]:.3f},{spot.at[1]:.3f},{spot.yaw},{int(spot.mirrored)}"
        out[camera] = SpotAt(camera, spot.place, at, (spot.at[0], spot.at[1]), spot.mirrored, basis)
    return out


#: What the runner says of two pictures (spatial/pointmap_runner.py `pair`).
Match = Mapping[str, Any]


def confident(match: Match | None) -> str | None:
    """Why a match can't be used, or None when it can: "none" (no match was made), "apart" (the pictures share too
    little: different space), or "unsure" (they share enough, but the two directions disagree)."""
    if not match:
        return "none"
    if "rotation" not in match or min(match["sure"]) < MIN_SURE or max(match["match"]) < MIN_MATCH:
        return "apart"
    turn, shift, scale = match["disagree"]
    far = float(np.linalg.norm(match["shift"]))
    if turn > AGREE[0] or shift > max(AGREE[1], AGREE[2] * far) or scale > AGREE[3]:
        return "unsure"
    return None


def relative(pose: Sequence[float], match: Match) -> tuple[float, float, float] | None:
    """Where the second camera of `match` stands on the floor as seen from the first, whose cloud has `pose` (its
    camera in its level frame): metres to its right, metres ahead, and radians turned. None
    for a camera looking so nearly straight down or up that it has no facing."""
    level = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    turn = level[:3, :3] @ np.asarray(match["rotation"], dtype=np.float64).reshape(3, 3)
    at = level[:3, :3] @ np.asarray(match["shift"], dtype=np.float64) + level[:3, 3]
    ahead = turn[:2, 2]
    if float(np.linalg.norm(ahead)) < 0.2:
        return None
    return float(at[0]), float(at[1]), math.atan2(float(ahead[0]), float(ahead[1]))


def solve(
    spots: Mapping[str, At],
    edges: Sequence[tuple[str, str, tuple[float, float, float]]],
    firm: Sequence[str] = (),
) -> dict[str, At]:
    """Where each camera stands, given its spot and what the confident pairs measured (`edges`: for (a, b, rel), b
    as seen from a). Each camera is held to its spot and each pair to its measurement, a measurement `EDGE` times
    as hard, so a pair's two cameras share the correction between them and nothing floats off the plan. A camera in
    `firm` (one squared to the plan) is held `FIRM` times as hard as a measurement: its partner takes the
    correction. With no edges it is the spots."""
    names = sorted(spots)
    if not edges:
        return dict(spots)
    from scipy.optimize import least_squares

    index = {name: i for i, name in enumerate(names)}
    start = np.array([[spots[n].x, spots[n].y, spots[n].bearing] for n in names]).ravel()
    hold = np.array([EDGE * FIRM if n in firm else 1.0 for n in names])

    def residuals(flat: np.ndarray) -> np.ndarray:
        q = flat.reshape(-1, 3)
        out = [
            (hold[:, None] * (q[:, :2] - start.reshape(-1, 3)[:, :2])).ravel(),
            hold * TURN_M * (q[:, 2] - start.reshape(-1, 3)[:, 2]),
        ]
        for a, b, (right, ahead, turn) in edges:
            pa, pb = At(*q[index[a]]), At(*q[index[b]])
            got = pb.seen_from(pa)
            out.append(EDGE * np.array([got[0] - right, got[1] - ahead, TURN_M * wrap(got[2] - turn)]))
        return np.concatenate(out)

    found = least_squares(residuals, start).x.reshape(-1, 3)
    return {name: At(float(found[i, 0]), float(found[i, 1]), float(found[i, 2])) for name, i in index.items()}


def moved_by(a: At, b: At) -> tuple[float, float]:
    """How far apart two positions are: metres, and degrees of facing."""
    return math.hypot(a.x - b.x, a.y - b.y), abs(math.degrees(wrap(a.bearing - b.bearing)))


# ---- what is kept beside the clouds


def _pair_key(a: str, b: str) -> str:
    return "|".join(sorted((a, b)))


def pairs(store: camera_clouds.Store) -> dict[str, dict[str, Any]]:
    got = json_state.read(store.folder / PAIRS, {})
    return got if isinstance(got, dict) else {}


def _camera_file(store: camera_clouds.Store, camera: str, name: str) -> Path:
    return store.folder / camera_clouds.slug(camera) / name


def anchor(store: camera_clouds.Store, camera: str) -> dict[str, Any] | None:
    """`camera`'s anchor: the picture it was placed with at its spot (`anchor.png`), with that spot's `basis` and
    the `pose` of the cloud made from it. None before it has one."""
    got = json_state.read(_camera_file(store, camera, "anchor.json"))
    return got if isinstance(got, dict) and _camera_file(store, camera, "anchor.png").is_file() else None


def moved(store: camera_clouds.Store, camera: str) -> dict[str, Any] | None:
    """What `camera`'s pictures said when one last stopped matching its anchor (`record`), or None."""
    got = json_state.read(_camera_file(store, camera, "moved.json"))
    return got if isinstance(got, dict) else None


def unsettled(store: camera_clouds.Store, camera: str, cloud: str, basis: str | None = None) -> dict[str, Any] | None:
    """What the pictures said (`moved`) when `cloud` is of a view its camera's anchor isn't: the camera moved, and
    hasn't been placed again with this cloud (a placing makes the cloud's picture the anchor). With `basis`, only
    while the anchor is of that spot."""
    held, was = anchor(store, camera), moved(store, camera)
    if was is None or held is None or was.get("cloud") != cloud:
        return None
    return was if basis is None or held.get("basis") == basis else None


TRACK = "track.json"  # a moving camera's track (spatial/camera_track.py `Track.write`), beside its cloud
TRACK_FOLDER = "track"  # where a track kept its keyframes' clouds before the place had a map: removed with it
TRACK_FRESH_S = 180.0  # a cloud whose picture was taken this long after a track's last pose is placed from it


def tracked(store: camera_clouds.Store, camera: str) -> dict[str, Any] | None:
    """`camera`'s track as spatial/camera_track.py last wrote it, or None."""
    got = json_state.read(_camera_file(store, camera, TRACK))
    return got if isinstance(got, dict) and isinstance(got.get("xyb"), list) and len(got["xyb"]) == 3 else None


def drop_track(store: camera_clouds.Store, camera: str) -> None:
    """Forget `camera`'s track."""
    _camera_file(store, camera, TRACK).unlink(missing_ok=True)
    shutil.rmtree(_camera_file(store, camera, TRACK_FOLDER), ignore_errors=True)


def back_where_it_was(store: camera_clouds.Store, camera: str) -> None:
    """`camera`'s picture shows it where its anchor was taken: a track that has ended says nothing more (it was
    put back, or what was tracked wasn't the camera moving), and goes."""
    was = tracked(store, camera)
    if was is not None and was.get("state") != "tracking":
        drop_track(store, camera)


def _track_of(store: camera_clouds.Store, cloud: camera_clouds.Cloud, basis: str) -> dict[str, Any] | None:
    """The track that says where `cloud`'s camera is now: one beside the spot `basis` that is still running, or
    that ended after the cloud's picture was taken (the camera has moved since)."""
    was = tracked(store, cloud.camera)
    if was is None or was.get("basis") != basis:
        return None
    return was if was.get("state") == "tracking" or float(was.get("at") or 0.0) > cloud.at else None


def standing(store: camera_clouds.Store, camera: str, spots: Mapping[str, SpotAt], rooms: Sequence[Ring]) -> At | None:
    """Where `camera` stands in the scene by its own cloud, or None when it isn't placed (no cloud, no spot with a
    facing, or a view that couldn't be placed)."""
    cloud, spot = store.current(camera), spots.get(camera)
    if cloud is None or spot is None or spot.at is None:
        return None
    stood = _stood(store, camera, cloud, spot, cloud_align.plan_key(rooms))
    settled = stood.told is not None and stood.told.cloud == cloud.id
    if settled and stood.told is not None and stood.told.lost:
        return None
    if not settled and unsettled(store, camera, cloud.id, spot.basis) is not None:
        return None
    return stood.at


def resting(
    store: camera_clouds.Store, camera: str, spots: Mapping[str, SpotAt], rooms: Sequence[Ring]
) -> tuple[At, np.ndarray, Callable[[], np.ndarray | None]] | None:
    """What a track of `camera` begins on (spatial/camera_track.py): where it stands (`standing`), its camera in
    the scene as its cloud has it (4 x 4, carrying the cloud's fitted size), and a reader of the picture that cloud
    was made from (None when it isn't kept). None when the camera isn't placed."""
    at = standing(store, camera, spots, rooms)
    cloud, spot = store.current(camera), spots.get(camera)
    if at is None or cloud is None or spot is None:
        return None
    stood = _stood(store, camera, cloud, spot, cloud_align.plan_key(rooms))
    across, upward = stood.sized or (1.0, 1.0)
    lie = _transform(stood.at) @ np.diag([across, across, upward, 1.0])
    picture = store.picture(cloud)

    def read() -> np.ndarray | None:
        import cv2

        return cv2.imread(str(picture)) if picture.is_file() else None

    return at, lie @ np.asarray(cloud.pose, dtype=np.float64).reshape(4, 4), read


FOLLOW = "follow.json"
#: Why a view can't be placed (`Follow.lost`): spatial/cloud_align.py `Located.state`'s, and "pending" in a scene
#: read before the camera was placed again.
LOST = ("no_walls", "no_plan", "no_match", "ambiguous", "pending")


@dataclass(frozen=True)
class Follow:
    """Where a camera that left its spot is taken to stand: its working pose, kept beside the owner's spot."""

    basis: str  # the owner's spot it is beside (`SpotAt.basis`): it holds only while the spot is that one
    at: tuple[float, float, float] | None  # x, y, and bearing in the scene; None: the spot itself
    #: what placed it: "tracking" (where its track left it), "pictures", "turned" (where it stood, facing another
    #: way), or "walls"
    by: str = ""
    cloud: str = ""  # the cloud it was last worked out with
    since: float = 0.0  # when that cloud's picture was taken
    lost: str = ""  # why that cloud's view can't be placed (one of `LOST`), when it can't: `at` is the last good pose
    pinned: bool = False  # the owner put it back on the plan's spot
    score: float = 0.0  # spatial/cloud_align.py `Located.score`, when its cloud placed it


def following(store: camera_clouds.Store, camera: str, basis: str) -> Follow | None:
    """`camera`'s working pose, when it has one beside the spot `basis`."""
    return _follow_of(json_state.read(_camera_file(store, camera, FOLLOW)), basis)


def _follow_of(raw: object, basis: str) -> Follow | None:
    try:
        if not isinstance(raw, dict) or raw.get("basis") != basis:
            return None
        at = None if raw.get("at") is None else (float(raw["at"][0]), float(raw["at"][1]), float(raw["at"][2]))
        return Follow(basis, at, str(raw.get("by", "")), str(raw.get("cloud", "")), float(raw.get("since", 0.0)),
                      str(raw.get("lost", "")), bool(raw.get("pinned", False)), float(raw.get("score", 0.0)))  # fmt: skip
    except KeyError, TypeError, ValueError, IndexError:
        return None


def _follow(store: camera_clouds.Store, camera: str, told: Follow) -> Follow:
    json_state.write(_camera_file(store, camera, FOLLOW), asdict(told))
    return told


def _working(told: Follow | None, spot: SpotAt) -> tuple[At, str]:
    """Where a camera is taken to stand before its cloud is squared and fitted, and what a squaring from there is
    of: the spot and its basis, or the working pose."""
    assert spot.at is not None
    if told is None or told.at is None:
        return spot.at, spot.basis
    x, y, bearing = told.at
    return At(x, y, bearing), f"{spot.basis}>{x:.3f},{y:.3f},{bearing:.4f}"


@dataclass(frozen=True)
class _Stood:
    told: Follow | None
    basis: str  # what a squaring from where it stands is of (`_working`)
    at: At  # where the camera stands: its spot or working pose, squared and fitted where that holds
    sized: tuple[float, float] | None  # a fitted cloud's size across and upwards
    firm: bool  # placed by the plan's walls
    square: cloud_align.Square | None
    fit: cloud_align.Fit | None


def _stood(store: camera_clouds.Store, camera: str, cloud: camera_clouds.Cloud, spot: SpotAt, plan: str) -> _Stood:
    """Where a camera with a spot and a facing stands by itself: its spot, or its working pose, moved by its
    cloud's squaring and fit while they are of this cloud, pose, and plan."""
    told = following(store, camera, spot.basis)
    base, basis = _working(told, spot)
    have = cloud_align.kept(_camera_file(store, camera, cloud_align.ALIGN).parent)
    if have is None or not have.holds(cloud.id, basis, plan):
        return _Stood(told, basis, base, None, False, None, None)
    fitted = have.fit
    if fitted is not None and fitted.state == "fitted":  # the rooms' walls say more than a slide does
        at = At(base.x + fitted.dx, base.y + fitted.dy, base.bearing + math.radians(fitted.turn_deg))
        return _Stood(told, basis, at, (fitted.scale, fitted.up), True, have.square, fitted)
    if have.square.state == "squared":
        return _Stood(told, basis, base.step(*have.square.rel), None, True, have.square, fitted)
    return _Stood(told, basis, base, None, False, have.square, fitted)


def _set_anchor(store: camera_clouds.Store, cloud: camera_clouds.Cloud, basis: str) -> None:
    shutil.copyfile(store.picture(cloud), _camera_file(store, cloud.camera, "anchor.png"))
    record = {"basis": basis, "at": cloud.at, "pose": list(cloud.pose)}
    json_state.write(_camera_file(store, cloud.camera, "anchor.json"), record)
    _camera_file(store, cloud.camera, "moved.json").unlink(missing_ok=True)


def compare(store: camera_clouds.Store, camera: str, spots: Mapping[str, SpotAt]) -> list[tuple[str, Path]]:
    """The pictures a new picture of `camera` is compared with: its anchor first ("anchor"), then the other cameras
    of its place that have a spot and a cloud's picture, the nearest by their spots first, at most `MAX_OTHERS`.
    None for a camera with no spot: it isn't placed, so there is nothing to tighten or follow."""
    mine = spots.get(camera)
    if mine is None or mine.at is None:
        return []
    out = []
    held = anchor(store, camera)
    if held is not None and held.get("basis") == mine.basis:
        out.append(("anchor", _camera_file(store, camera, "anchor.png")))
    near = []
    for other in store.cameras():
        spot = spots.get(other)
        cloud = store.current(other)
        if other == camera or spot is None or spot.at is None or spot.place != mine.place or cloud is None:
            continue
        if store.picture(cloud).is_file():
            near.append((math.hypot(spot.at.x - mine.at.x, spot.at.y - mine.at.y), other, store.picture(cloud)))
    return out + [(other, path) for _, other, path in sorted(near)[:MAX_OTHERS]]


def stayed(store: camera_clouds.Store, camera: str, matches: Sequence[Match | None]) -> bool:
    """Whether a new picture of `camera` shows it where its anchor was taken: `matches` holds what the picture
    matched of the anchor (the first of `compare`'s). False with no anchor or no confident match."""
    held = anchor(store, camera)
    against = matches[0] if matches else None
    if held is None or confident(against):
        return False
    rel = relative(held["pose"], against or {})
    return rel is not None and math.hypot(rel[0], rel[1]) <= STILL[0] and abs(math.degrees(rel[2])) <= STILL[1]


def relevel(store: camera_clouds.Store, camera: str) -> bool:
    """Stand an older cloud up by its walls (spatial/camera_clouds.py `Store.relevel`), carrying over what the scene
    keeps of it: its anchor's pose, when the anchor is its picture, and that it had moved. Whether it was stood up."""
    was = store.current(camera)
    now = store.relevel(camera)
    if was is None or now is None:
        return False
    held = json_state.read(_camera_file(store, camera, "anchor.json"))
    if isinstance(held, dict) and held.get("at") == was.at:
        json_state.write(_camera_file(store, camera, "anchor.json"), {**held, "pose": list(now.pose)})
    for name in ("moved.json", FOLLOW):
        gone = json_state.read(_camera_file(store, camera, name))
        if isinstance(gone, dict) and gone.get("cloud") == was.id:
            json_state.write(_camera_file(store, camera, name), {**gone, "cloud": now.id})
    LOG.info("%s: its cloud had no floor and stood as if the camera were upright; stood up by its walls", camera)
    return True


def plan_rooms(state: Path, place: str, mirrored: bool) -> tuple[tuple[tuple[float, float], ...], ...]:
    """The rooms of `place`'s plan, when the owner drew it, in the scene's frame; none otherwise."""
    held = places.get(state, place)
    return rooms_of(held.plan if held is not None and held.plan_source == "owner" else None, mirrored)


def _place_again(
    store: camera_clouds.Store,
    cloud: camera_clouds.Cloud,
    spots: Mapping[str, SpotAt],
    was: Mapping[str, Any],
    found: Sequence[cloud_align.Line],
    ring: Sequence[Ring],
    points: np.ndarray,
) -> Follow:
    """Place a camera whose view changed (`was`: what its pictures said) with its new `cloud` (the module
    docstring's 3): its new working pose, or that the view can't be placed, kept beside the cloud."""
    camera, spot = cloud.camera, spots[cloud.camera]
    told = following(store, camera, spot.basis)
    base, _ = _working(told, spot)
    pose, by, score, why = None, "", 0.0, ""
    track = tracked(store, camera)
    if track is not None and track.get("basis") == spot.basis and abs(cloud.at - float(track["at"])) <= TRACK_FRESH_S:
        # It was tracked to here: its walls may still turn it where it stands, and otherwise the track's word holds.
        left = At(*(float(v) for v in track["xyb"]))
        located = cloud_align.locate(found, cloud_align.edges_of(ring), ring, points, (left.x, left.y), anywhere=False)
        near = located.state == "placed" and moved_by(At(located.x, located.y, located.bearing), left)[1] <= BOUND[1]
        pose, by = (At(located.x, located.y, located.bearing) if near else left), "tracking"
    elif was.get("state") == "followed":
        pose, by = base.step(*was["rel"]), "pictures"
    elif was.get("state") == "against":
        other = str(was.get("other"))
        theirs, there = store.current(other), spots.get(other)
        if theirs is not None and theirs.id == was.get("other_cloud") and there is not None and there.at is not None:
            partner = _stood(store, other, theirs, there, cloud_align.plan_key(ring))
            if partner.told is None or not (partner.told.lost and partner.told.cloud == theirs.id):
                pose, by = partner.at.step(*was["rel"]), "pictures"
    if pose is None:
        located = cloud_align.locate(found, cloud_align.edges_of(ring), ring, points, near=(base.x, base.y))
        if located.state == "placed":
            pose, by, score = At(located.x, located.y, located.bearing), located.by, located.score
        why = located.state
    if pose is None:
        LOG.info("%s: its view changed and can't be placed (%s); it keeps its last pose, undrawn", camera, why)
        kept = Follow(spot.basis, None, "", "", 0.0) if told is None else told
        return _follow(store, camera, replace(kept, cloud=cloud.id, since=cloud.at, lost=why))
    shift, turn = moved_by(pose, base)
    LOG.info("%s: its view changed; placed again by %s: turned %.0f degrees, moved %.2f m", camera, by, turn, shift)
    if store.picture(cloud).is_file():
        _set_anchor(store, cloud, spot.basis)  # this picture is the one it is placed with now
    drop_track(store, camera)  # it is an ordinary camera again
    return _follow(
        store, camera, Follow(spot.basis, (pose.x, pose.y, wrap(pose.bearing)), by, cloud.id, cloud.at, score=score)
    )


def shot_of(
    shot: str,
    cloud: camera_clouds.Cloud,
    stand: cloud_merge.Stand,
    up: float,
    held: bool,
    score: float,
    points: np.ndarray,
    colours: np.ndarray | None,
    planes: Sequence[cloud_align.Plane],
    fit: cloud_align.Fit | None,
    lie: np.ndarray | None = None,
) -> cloud_map.Offer:
    """A levelled cloud standing at `stand`, as a shot to offer a place's map (spatial/cloud_map.py). With `lie`
    (the cloud's frame in the scene, 4 x 4), it is offered exactly there: a tracked keyframe, whose camera's tilt
    the track knows better than the keyframe's own cloud does."""
    fill = np.full((len(points), 3), 160, dtype=np.uint8) if colours is None else colours
    pose = np.asarray(cloud.pose, dtype=np.float64).reshape(4, 4)
    return cloud_map.Offer(shot, cloud.id, cloud.camera, cloud.at, held, score, points, fill,
                           cloud_merge.surfaces(planes), stand, up, pose, cloud.focal, cloud.width, cloud.height, fit,
                           lie)  # fmt: skip


def _offer(
    charted: Path,
    ring: Sequence[Ring],
    cloud: camera_clouds.Cloud,
    basis: str,
    base: At,
    placed: cloud_align.Align,
    points: np.ndarray,
    colours: np.ndarray | None,
    planes: Sequence[cloud_align.Plane],
) -> None:
    """Offer a camera's placed cloud to its place's map: held where its fit puts it, or, when the plan's walls
    don't hold it, loose where it stands (the map may bring it into agreement with what it holds)."""
    fitted = placed.fit
    if fitted is not None and fitted.state == "fitted":
        stand = cloud_merge.Stand(base.x + fitted.dx, base.y + fitted.dy, base.bearing + math.radians(fitted.turn_deg),
                                  fitted.scale)  # fmt: skip
        shot = shot_of(
            f"{cloud.id}@{basis}", cloud, stand, fitted.up, True, fitted.share, points, colours, planes, fitted
        )
    else:
        at = base.step(*placed.square.rel) if placed.square.state == "squared" else base
        shot = shot_of(f"{cloud.id}@{basis}", cloud, cloud_merge.Stand(at.x, at.y, at.bearing), 1.0, False, 0.0,
                       points, colours, planes, None)  # fmt: skip
    try:
        cloud_map.offer(charted, ring, shot)
    except (OSError, ValueError) as e:  # the camera is placed; only the map misses its shot
        LOG.warning("%s: its shot wasn't offered to the map: %s", cloud.camera, e)


def align(
    store: camera_clouds.Store,
    camera: str,
    spots: Mapping[str, SpotAt],
    rooms: Callable[[str, bool], Sequence[Ring]],
    heights: Mapping[str, float] | None = None,
) -> bool:
    """Bring `camera`'s place on the plan up to date (the module docstring): a camera whose view changed is placed
    again (`_place_again`), and its cloud is squared and fitted to the plan from where it stands, when it has a
    cloud, a spot with a facing, and no squaring that is of this cloud, pose, and plan (`rooms`: a place's plan
    rooms, `plan_rooms` with the state bound; `heights`: the fixed cameras' measured heights, spatial/rig.py).
    Whether anything was made. Writes only beside the cloud."""
    cloud, spot = store.current(camera), spots.get(camera)
    folder = _camera_file(store, camera, cloud_align.ALIGN).parent
    if cloud is None or spot is None or spot.at is None:
        return False
    ring = rooms(spot.place, spot.mirrored)
    plan = cloud_align.plan_key(ring)
    told = following(store, camera, spot.basis)
    was = unsettled(store, camera, cloud.id, spot.basis)
    due = was is not None and (told is None or told.cloud != cloud.id)
    base, basis = _working(told, spot)
    have = cloud_align.kept(folder)
    gone = tracked(store, camera)
    if gone is not None and not due and gone.get("state") != "tracking" and float(gone.get("at") or 0.0) <= cloud.at:
        drop_track(store, camera)  # a track that ended before this cloud, which shows the camera where it was
    charted = cloud_map.folder(store.folder, spot.place)
    lost = told is not None and bool(told.lost) and told.cloud == cloud.id
    if not due and have is not None and have.holds(cloud.id, basis, plan):
        if lost or cloud_map.answered(cloud_map.record(charted), f"{cloud.id}@{basis}") is not None:
            return False
        points, colours, _ = ply.read(store.file(cloud))  # placed before the place had a map: it is offered now
        _offer(charted, ring, cloud, basis, base, have, points, colours, cloud_align.planes(points))
        return True
    points, colours, _ = ply.read(store.file(cloud))
    flat_surfaces = cloud_align.planes(points)
    walls = cloud_align.lines(flat_surfaces)
    if was is not None and due:
        told = _place_again(store, cloud, spots, was, walls, ring, points)
        base, basis = _working(told, spot)
    measured = (heights or {}).get(camera)
    height = (measured, cloud.pose[11]) if measured and cloud.levelled == "floor" else None  # only a real floor counts

    cloud_id = cloud.id

    def made(at: At, of: str) -> cloud_align.Align:
        return cloud_align.make(cloud_id, points, (at.x, at.y, at.bearing), of, ring, height, walls)

    found = made(base, basis)
    tried = told is not None and (told.pinned or told.cloud == cloud.id)
    if found.square.state == "beyond" and not tried:
        # Its pictures didn't change, but its walls run along the plan's only far from its facing: look where it
        # stands for the facing its walls fit (a spot set facing the wrong way, or a turn the pictures missed).
        located = cloud_align.locate(walls, cloud_align.edges_of(ring), ring, points, (base.x, base.y), anywhere=False)
        if located.state == "placed":
            at = (located.x, located.y, wrap(located.bearing))
            told = _follow(store, camera, Follow(spot.basis, at, "turned", cloud.id, cloud.at, score=located.score))
            base, basis = _working(told, spot)
            found = made(base, basis)
            LOG.info("%s: its walls fit the plan %.0f degrees from its facing; it is placed by them", camera,
                     moved_by(base, spot.at)[1])  # fmt: skip
    cloud_align.keep(folder, found)
    if not (told is not None and told.lost and told.cloud == cloud.id):  # a view that isn't placed isn't mapped
        _offer(charted, ring, cloud, basis, base, found, points, colours, flat_surfaces)
    fitted = found.fit or cloud_align.Fit("no_walls")
    LOG.info("%s: squared to the plan: %s, turned %.1f degrees; fitted to its rooms: %s, resized %.3f, slid %.2f m "
             "(%d of %d walls matched)", camera, found.square.state, found.square.turn_deg, fitted.state, fitted.scale,
             math.hypot(fitted.dx, fitted.dy), len(fitted.walls), fitted.seen)  # fmt: skip
    return True


_UPDATING = threading.Lock()


def update(state: Path, store: camera_clouds.Store, place: str | None = None) -> int:
    """Bring every stored cloud up to date with no picture and no model: an older cloud with no floor is stood up by
    its walls (`relevel`), a camera whose view changed is placed again, and each camera's squaring and fit to its
    plan is made where it is missing or of another cloud, pose, or plan (`align`). With `place`, not the cameras with a spot on another place's plan. How many were
    changed. It is cheap when nothing is stale, so the 3D view asks for it on every scene: clouds stand level and
    fitted without Learn surroundings being on. One at a time in a process; writes only beside the clouds."""
    from liberational.spatial import rig

    with _UPDATING:
        spots = spots_of(state)
        heights = rig.load().heights()
        held: dict[tuple[str, bool], Sequence[Ring]] = {}

        def rooms(where: str, mirrored: bool) -> Sequence[Ring]:
            if (where, mirrored) not in held:
                held[where, mirrored] = plan_rooms(state, where, mirrored)
            return held[where, mirrored]

        changed = 0
        for camera in store.cameras():
            spot = spots.get(camera)
            if place is not None and spot is not None and spot.place != place:
                continue
            try:
                changed += relevel(store, camera)
                changed += align(store, camera, spots, rooms, heights)
            except (OSError, ValueError, KeyError, TypeError) as e:  # one camera's cloud must not stop the others
                LOG.warning("%s: its cloud wasn't brought up to date: %s", camera, e)
        for where, mirrored in {(s.place, s.mirrored) for s in spots.values() if place in (None, s.place)}:
            try:  # a map kept before its floor was counted is counted once
                cloud_map.count_floor(cloud_map.folder(store.folder, where), rooms(where, mirrored))
            except (OSError, ValueError) as e:
                LOG.warning("the map's floor wasn't counted: %s", e)
        return changed


def record(
    store: camera_clouds.Store,
    cloud: camera_clouds.Cloud,
    matches: Mapping[str, Match | None],
    spots: Mapping[str, SpotAt],
) -> None:
    """Take in what `cloud`'s picture matched (`matches`, by `compare`'s labels): whether its camera moved from its
    anchor and where to (the module docstring's 3), and each pair's measurement for `scene` to use (2)."""
    camera = cloud.camera
    mine = spots.get(camera)
    if mine is None or mine.at is None or not store.picture(cloud).is_file():
        return
    held = anchor(store, camera)
    if held is None or held.get("basis") != mine.basis:
        _set_anchor(store, cloud, mine.basis)  # first placed at this spot with this picture
    else:
        against = matches.get("anchor")
        rel = None if confident(against) else relative(held["pose"], against or {})
        if stayed(store, camera, [against]):
            _set_anchor(store, cloud, mine.basis)  # the same position: the newer picture is the one to match next
            back_where_it_was(store, camera)
        elif rel is not None:
            was = {"state": "followed", "cloud": cloud.id, "rel": list(rel), "at": cloud.at}
            json_state.write(_camera_file(store, camera, "moved.json"), was)
            LOG.info(
                "%s: moved %.2f m and %.0f degrees; its cloud follows",
                camera,
                math.hypot(*rel[:2]),
                math.degrees(rel[2]),
            )
        else:
            was = {"state": "unplaced", "cloud": cloud.id, "at": cloud.at}
            for other, match in matches.items():
                theirs = store.current(other) if other != "anchor" else None
                if theirs is None or confident(match):
                    continue
                # the runner measured the other camera as seen from this one: turn it round
                back = relative(cloud.pose, match or {})
                if back is not None:
                    here = At(0.0, 0.0, 0.0).seen_from(At(back[0], back[1], back[2]))
                    was = {"state": "against", "cloud": cloud.id, "other": other, "other_cloud": theirs.id,
                           "rel": list(here), "at": cloud.at}  # fmt: skip
                    break
            json_state.write(_camera_file(store, camera, "moved.json"), was)
            LOG.info("%s: its view no longer matches its spot's (%s)", camera, was["state"])
    kept = pairs(store)
    for other, match in matches.items():
        theirs = store.current(other) if other != "anchor" else None
        spot = spots.get(other)
        if theirs is None or spot is None or match is None:
            continue
        why = confident(match)
        rel = None if why else relative(cloud.pose, match)
        why = why or (None if rel is not None else "unsure")
        key = _pair_key(camera, other)
        looked = {
            "why": why,
            "match": list(match["match"]),
            "sure": list(match["sure"]),
            "disagree": match.get("disagree"),
            "at": cloud.at,
        }
        was = kept.get(key)
        held_bases = was is not None and {was.get("a"): was.get("a_basis"), was.get("b"): was.get("b_basis")} == {
            camera: mine.basis,
            other: spot.basis,
        }
        if why is not None and was is not None and was.get("rel") is not None and held_bases:
            # Not confident this time (a night picture, say): the cameras haven't moved, so the last confident
            # measurement of the pair still says how they sit. It's kept, with what this look came to.
            kept[key] = {**was, "latest": looked}
            continue
        kept[key] = {
            "a": camera,
            "b": other,
            "a_cloud": cloud.id,
            "b_cloud": theirs.id,
            "a_basis": mine.basis,
            "b_basis": spot.basis,
            "rel": None if rel is None else list(rel),
            "scale": match.get("scale"),
            **looked,
        }
    json_state.write(store.folder / PAIRS, kept)


def use_spot(store: camera_clouds.Store, camera: str, basis: str) -> bool:
    """The owner puts `camera` back on the plan's spot (`basis`): the way back from a placement that guessed wrong,
    or from a view that couldn't be placed. Its picture now is the one it is placed with there, so the spot holds
    until its view next changes. The working pose it had is set aside for `follow_again`. Whether there was a cloud
    to put back. Writes only beside the cloud."""
    cloud = store.current(camera)
    if cloud is None:
        return False
    was = following(store, camera, basis)
    aside = None if was is None or was.at is None or was.lost or was.pinned else asdict(was)
    record = asdict(Follow(basis, None, "", cloud.id, cloud.at, pinned=True)) | {"before": aside}
    json_state.write(_camera_file(store, camera, FOLLOW), record)
    if store.picture(cloud).is_file():
        _set_anchor(store, cloud, basis)
    return True


def follow_again(store: camera_clouds.Store, camera: str, basis: str) -> bool:
    """Take `use_spot` back: `camera` stands at the working pose it had, or, when it had none (its view couldn't be
    placed), is placed again by its cloud at the next `align`, as one whose view changed is. Whether it had been
    put on the spot."""
    raw = json_state.read(_camera_file(store, camera, FOLLOW))
    told, cloud = _follow_of(raw, basis), store.current(camera)
    if told is None or not told.pinned or cloud is None or not isinstance(raw, dict):
        return False
    before = _follow_of(raw.get("before"), basis)
    if before is not None:
        _follow(store, camera, replace(before, cloud=cloud.id))
        return True
    _camera_file(store, camera, FOLLOW).unlink(missing_ok=True)
    again = {"state": "unplaced", "cloud": cloud.id, "at": cloud.at}
    json_state.write(_camera_file(store, camera, "moved.json"), again)
    return True


# ---- the scene


@dataclass(frozen=True)
class Placed:
    """One camera in a place's scene, or why it isn't in it."""

    camera: str
    cloud: str  # its cloud's id
    points: int
    #: "spot" (from its spot alone), "adjusted" (from its spot, corrected against `against`), "following" (its view
    #: changed and it was placed again: it stands at its working pose, `follow`), or, not in the scene, "no_spot",
    #: "no_facing", "lost" (its view changed and can't be placed: `follow` says why).
    how: str
    size: str  # "approximate": the model's own metres
    transform: tuple[float, ...] | None = None  # its cloud's level frame to the scene, 4 x 4 row by row
    pose: tuple[float, ...] | None = None  # its camera in the scene
    against: tuple[str, ...] = ()
    shift_m: float = 0.0  # how far from its spot it is placed
    turn_deg: float = 0.0
    at: float = 0.0  # when its cloud's picture was taken: the age of where it is drawn
    #: For a camera with a working pose or a view that can't be placed: `by` (what placed it: "pictures", "turned",
    #: "walls"), `since` (when), `lost` (why its view can't be placed, one of `LOST`, else ""), and `pinned` (the
    #: owner put it back on the plan's spot). None for a camera that never left its spot.
    follow: Mapping[str, Any] | None = None
    #: While the camera is being tracked as it moves, or has just stopped (spatial/camera_track.py): `state`
    #: ("tracking" or "ended"), `since` and `at` (when the track began and when it last placed the camera), `held`
    #: (when a keyframe was last held to the plan's walls, or None), `pose` (the camera in the scene now, 4 x 4),
    #: `went_m` (how far its path ran), `from_m` and `from_deg` (how far it is from where its cloud has it), and
    #: `added` and `added_at` (how many points this move has put into the place's map, and when the last went in).
    #: None otherwise.
    track: Mapping[str, Any] | None = None
    #: What the place's map (spatial/cloud_map.py) holds of the camera, once its cloud was offered: `points` (of the
    #: map's, from this camera's shots), `state` of its latest shot ("added" or "refused"), `why` ("apart": as a
    #: whole it stands somewhere else than the map's shots; "full"), `added` (points of that shot that went in),
    #: `left_out` (points of it that better-placed shots looked through, which were not added), `off_m` (how far
    #: a shot the plan's walls don't hold stayed from the map's surfaces when it couldn't be lined up with them),
    #: and `moved_m`, `turned_deg`, `resized` (how such a shot was moved when it could: then the camera is drawn
    #: where that put it). None before it was offered.
    map: Mapping[str, Any] | None = None
    level: str = "none"  # what its cloud was levelled by: "floor", "walls", or "none"
    tilt_deg: float = 0.0  # the camera's tilt (down) and roll, by that
    roll_deg: float = 0.0
    #: Its squaring to the plan (spatial/cloud_align.py `Square`, as a dict), when one holds; `squared` when it is
    #: what placed it.
    square: Mapping[str, Any] | None = None
    squared: bool = False
    #: Its cloud's fit to the plan's rooms (spatial/cloud_align.py `Fit`, as a dict), when one holds; `fitted` when
    #: it is what placed and sized the cloud, and its floor and matched walls can be drawn flat.
    fit: Mapping[str, Any] | None = None
    fitted: bool = False

    @property
    def placed(self) -> bool:
        return self.transform is not None


@dataclass(frozen=True)
class Pair:
    """What two cameras' pictures said of each other."""

    a: str
    b: str
    #: "tightened", "checked" (both cameras are squared to the plan: the measurement moves neither, and `off_m`
    #: and `off_deg` say how far the two are from it), "apart" (different space), "unsure", "disagrees" (beyond the
    #: bound), "moved" (one of the two moved since), or "stale" (measured at another spot)
    state: str
    match: tuple[float, float] = (0.0, 0.0)
    sure: tuple[float, float] = (0.0, 0.0)
    disagree: tuple[float, float, float] | None = None
    off_m: float = 0.0  # how far the measurement is from what the spots say
    off_deg: float = 0.0
    measured: float | None = None  # when the measurement in use was made
    latest: str | None = None  # why a newer look wasn't used in its place ("apart", "unsure"), when one wasn't


@dataclass(frozen=True)
class Scene:
    place: str
    cameras: tuple[Placed, ...]
    pairs: tuple[Pair, ...] = ()
    mirrored: bool = False
    rooms: tuple[tuple[tuple[float, float], ...], ...] = ()  # the plan's rooms, in the scene's frame
    id: str = field(default="", compare=False)  # changes when what is drawn does
    #: How much of the plan's floor the place's map covers (spatial/cloud_map.py): `floor_m2` of `plan_m2`. None
    #: while the place has no map of this plan.
    covered: Mapping[str, float] | None = None


def _transform(at: At) -> np.ndarray:
    s, c = math.sin(at.bearing), math.cos(at.bearing)
    out = np.eye(4)
    out[:3, :3] = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    out[:2, 3] = (at.x, at.y)
    return out


#: A room's outline on the floor of the scene, as (x, y) corners.
Ring = Sequence[tuple[float, float]]


def rooms_of(plan: str | None, mirrored: bool) -> tuple[tuple[tuple[float, float], ...], ...]:
    """The rooms of an owner's plan as outlines in the scene's frame; none for a plan that doesn't read."""
    from liberational.spatial import plan_scene

    try:
        found = plan_scene.rooms(plan_scene.load(plan)) if plan else []
    except plan_scene.SceneError:
        return ()
    return tuple(tuple((float(x), float(y) if mirrored else -float(y)) for x, y in outline) for *_, outline in found)


def scene(store: camera_clouds.Store, place: str, spots: Mapping[str, SpotAt], rooms: Sequence[Ring] = ()) -> Scene:
    """`place`'s scene as it stands now: every camera of the place that has a cloud, placed (the module docstring) or
    with why it isn't. `rooms` are the plan's, for the view to leave out what lies outside them, and for a squaring
    to be of. Reads only; nothing is written."""
    clouds = {c: cloud for c in store.cameras() if (cloud := store.current(c)) is not None}
    mine = {c: clouds[c] for c in clouds if spots.get(c) is None or spots[c].place == place}
    plan = cloud_align.plan_key(rooms)
    squares: dict[str, cloud_align.Square] = {}  # each camera's squaring that still holds
    fits: dict[str, cloud_align.Fit] = {}  # and its fit to the rooms
    sized: dict[str, tuple[float, float]] = {}  # the cameras placed by their fit: resized across and upwards
    poses: dict[str, Follow] = {}  # the cameras with a working pose, or a view that can't be placed
    shots: dict[str, str] = {}  # each camera's cloud as a shot offered to the place's map
    firm: set[str] = set()
    start: dict[str, At] = {}  # where each placed camera stands by itself
    how: dict[str, str] = {}
    lost: dict[str, str] = {}  # why a camera's view can't be placed
    for camera, cloud in mine.items():
        spot = spots.get(camera)
        if spot is None or spot.place != place:
            how[camera] = "no_spot"
            continue
        if spot.at is None:
            how[camera] = "no_facing"
            continue
        stood = _stood(store, camera, cloud, spot, plan)
        shots[camera] = f"{cloud.id}@{stood.basis}"
        if stood.told is not None:
            poses[camera] = stood.told
        settled = stood.told is not None and stood.told.cloud == cloud.id
        if settled and stood.told is not None and stood.told.lost:
            how[camera], lost[camera] = "lost", stood.told.lost
        elif not settled and unsettled(store, camera, cloud.id, spot.basis) is not None:
            how[camera], lost[camera] = "lost", "pending"  # its view changed, and `align` hasn't placed it yet
        else:
            if stood.square is not None:
                squares[camera] = stood.square
            if stood.fit is not None:
                fits[camera] = stood.fit
            start[camera] = stood.at
            how[camera] = "following" if stood.told is not None and stood.told.at is not None else "spot"
            if stood.sized is not None:
                sized[camera] = stood.sized
            if stood.firm:
                firm.add(camera)

    # The confident pairs between cameras placed from their own spots, measured on the pictures and spots that hold.
    told: list[Pair] = []
    edges: list[tuple[str, str, tuple[float, float, float]]] = []
    for row in pairs(store).values():
        a, b = row.get("a"), row.get("b")
        if a not in mine or b not in mine:
            continue
        held = (
            spots.get(a) is not None
            and spots.get(b) is not None
            and row.get("a_basis") == spots[a].basis
            and row.get("b_basis") == spots[b].basis
        )
        latest = row.get("latest") if isinstance(row.get("latest"), dict) else None
        numbers: dict[str, Any] = {
            "match": tuple(row.get("match") or (0.0, 0.0)),
            "sure": tuple(row.get("sure") or (0.0, 0.0)),
            "disagree": tuple(row["disagree"]) if row.get("disagree") else None,
            "measured": row.get("at"),
            "latest": None if latest is None else str(latest.get("why")),
        }
        if not held:
            told.append(Pair(a, b, "stale", **numbers))
        elif row.get("why") or row.get("rel") is None:
            told.append(Pair(a, b, row.get("why") or "stale", **numbers))
        elif how.get(a) != "spot" or how.get(b) != "spot":
            told.append(Pair(a, b, "moved", **numbers))  # one of the two moved: what they measured no longer holds
        else:
            rel = (float(row["rel"][0]), float(row["rel"][1]), float(row["rel"][2]))
            said = start[b].seen_from(start[a])
            off = (math.hypot(said[0] - rel[0], said[1] - rel[1]), abs(math.degrees(wrap(said[2] - rel[2]))))
            if a in firm and b in firm:  # both squared to the plan: the plan wins, and the pictures are a check
                told.append(Pair(a, b, "checked", off_m=off[0], off_deg=off[1], **numbers))
                continue
            told.append(Pair(a, b, "tightened", off_m=off[0], off_deg=off[1], **numbers))
            edges.append((a, b, rel))
    # A match that would move a camera past the bound is dropped, the farthest off first, and says so.
    spotted = {c: at for c, at in start.items() if how[c] == "spot"}
    while True:
        solved = solve(spotted, edges, sorted(firm))
        over = [
            c
            for c in spotted
            if any(m > limit for m, limit in zip(moved_by(solved[c], spotted[c]), BOUND, strict=True))
        ]
        if not over or not edges:
            break
        worst = max(
            (e for e in edges if e[0] in over or e[1] in over),
            key=lambda e: next(p.off_m + TURN_M * math.radians(p.off_deg) for p in told if (p.a, p.b) == (e[0], e[1])),
        )
        edges.remove(worst)
        told = [replace(p, state="disagrees") if (p.a, p.b) == (worst[0], worst[1]) else p for p in told]
    at = {**start, **solved}
    partners: dict[str, list[str]] = {}
    for a, b, _ in edges:
        partners.setdefault(a, []).append(b)
        partners.setdefault(b, []).append(a)
    # What the place's map says of each camera's cloud. A cloud the plan's walls don't hold that the map brought
    # into agreement with what it holds stands where that put it.
    charted = cloud_map.record(cloud_map.folder(store.folder, place))
    if charted is not None and charted.get("plan") != plan:
        charted = None
    mapped: dict[str, dict[str, Any]] = {}
    for camera, shot in shots.items():
        said = cloud_map.answered(charted, shot)
        if said is None:
            continue
        total = sum(
            int(row.get("points", 0)) for row in (charted or {}).get("shots", ()) if row.get("camera") == camera
        )
        mapped[camera] = {
            "state": said.get("state"),
            "why": said.get("why", ""),
            "added": int(said.get("points", 0)),
            "points": total,
            "off_m": said.get("off_m", 0.0),
            "moved_m": said.get("moved_m", 0.0),
            "turned_deg": said.get("turned_deg", 0.0),
            "resized": said.get("resized", 1.0),
            "left_out": int(said.get("left_out", 0)),
        }
        stand = said.get("stand")
        if camera in at and not said.get("held") and stand and (said.get("moved_m") or said.get("turned_deg")):
            at[camera] = At(float(stand[0]), float(stand[1]), float(stand[2]))
            sized[camera] = (float(stand[3]), float(stand[3]))

    out = []
    for camera in sorted(mine):
        cloud, spot = mine[camera], spots.get(camera)
        tilt, roll = cloud_align.tilt_roll(np.asarray(cloud.pose, dtype=np.float64).reshape(4, 4))
        told_square = squares.get(camera)
        about: dict[str, Any] = {
            "at": cloud.at,
            "level": cloud.levelled,
            "tilt_deg": round(tilt, 1),
            "roll_deg": round(roll, 1),
            "square": None if told_square is None else asdict(told_square),
            "squared": camera in firm,
            "fit": None if camera not in fits else asdict(fits[camera]),
            "fitted": camera in sized,
        }
        moving = None if spot is None or spot.at is None else _track_of(store, cloud, spot.basis)
        if moving is not None and camera in at:
            now = At(*(float(v) for v in moving["xyb"]))
            went = moving.get("path") or ()
            from_m, from_deg = moved_by(now, at[camera])
            about["track"] = {
                "state": moving.get("state"),
                "since": moving.get("since"),
                "at": moving.get("at"),
                "held": moving.get("held"),
                "pose": moving.get("pose"),
                "went_m": round(sum(math.hypot(b[1] - a[1], b[2] - a[2]) for a, b in itertools.pairwise(went)), 2),
                "from_m": round(from_m, 2),
                "from_deg": round(from_deg, 1),
                "added": int(moving.get("added") or 0),
                "added_at": moving.get("added_at"),
            }
        if camera in mapped:
            about["map"] = mapped[camera]
        if camera in poses or camera in lost:
            kept_pose = poses.get(camera)
            about["follow"] = {
                "by": "" if kept_pose is None else kept_pose.by,
                "since": cloud.at if kept_pose is None else kept_pose.since,
                "lost": lost.get(camera, ""),
                "pinned": kept_pose is not None and kept_pose.pinned,
            }
        if camera not in at:
            out.append(Placed(camera, cloud.id, cloud.points, how[camera], cloud.size, **about))
            continue
        across, upward = sized.get(camera, (1.0, 1.0))
        transform = _transform(at[camera]) @ np.diag([across, across, upward, 1.0])
        shift, turn = moved_by(at[camera], spot.at) if spot is not None and spot.at is not None else (0.0, 0.0)
        kind = how[camera]
        if kind == "spot" and camera in partners and (shift >= TINY[0] or turn >= TINY[1]):
            kind = "adjusted"
        out.append(
            Placed(
                camera,
                cloud.id,
                cloud.points,
                kind,
                cloud.size,
                tuple(float(v) for v in transform.ravel()),
                tuple(float(v) for v in (transform @ np.asarray(cloud.pose).reshape(4, 4)).ravel()),
                tuple(sorted(partners.get(camera, ()))) if kind == "adjusted" else (),
                round(shift, 3),
                round(turn, 1),
                **about,
            )
        )
    mirrored = next((spots[c].mirrored for c in mine if spots.get(c) is not None), False)
    kept = tuple(tuple((float(x), float(y)) for x, y in ring) for ring in rooms)
    revision = None if charted is None else charted.get("rev")
    drawn = json.dumps([[[p.camera, p.cloud, p.transform] for p in out if p.placed], kept, revision])
    covered = None
    if charted is not None and charted.get("plan_m2"):
        covered = {"floor_m2": float(charted.get("floor_m2", 0.0)), "plan_m2": float(charted["plan_m2"])}
    return Scene(
        place, tuple(out), tuple(told), mirrored, kept, hashlib.sha256(drawn.encode()).hexdigest()[:32], covered
    )


def _fit_of(placed: Placed) -> cloud_align.Fit | None:
    """The fit a placed camera's cloud is drawn flat by, when it is placed by one."""
    if not placed.fitted or placed.fit is None:
        return None
    return cloud_align.fit_of(placed.fit)


def points(store: camera_clouds.Store, found: Scene, flat: bool = True) -> bytes:
    """What the 3D view draws of the scene, as one PLY: the place's map (spatial/cloud_map.py: every well-placed
    shot added up, each surface once), and beside it the cloud of any placed camera the map hasn't been offered
    yet. Each point has its camera's place in `found.cameras` as `view`, so the view can show, hide, and tint a
    camera's points, and `outside` 1 for a point more than `cloud_align.MARGIN_M` outside every room of the plan,
    which the view leaves out unless asked (a relief's depth is the model's estimate, and can run through a wall).
    With `flat`, the floor and matched walls of fitted clouds are drawn on the plan's (spatial/cloud_align.py
    `flatten`); without, every point is where its cloud has it. A cloud the map refused is not drawn: it would be
    a second copy of what the map holds. Nothing is changed: this is the drawing, made when asked for."""
    xyz, rgb, view = [], [], []
    held = cloud_map.read(cloud_map.folder(store.folder, found.place))
    if held is not None and held.meta.get("plan") == cloud_align.plan_key(found.rooms) and len(held.xyz):
        whose = cloud_map.cameras_of(held, [p.camera for p in found.cameras])
        known = whose != 65535  # a camera that is no longer in the place isn't drawn
        xyz.append((held.flat if flat else held.xyz)[known].astype(np.float64))
        rgb.append(held.rgb[known])
        view.append(whose[known])
    for index, placed in enumerate(found.cameras):
        if placed.transform is None or placed.map is not None:
            continue
        cloud = store.current(placed.camera)
        if cloud is None or cloud.id != placed.cloud:
            continue
        own, colours, _ = ply.read(store.file(cloud))
        move = np.asarray(placed.transform, dtype=np.float64).reshape(4, 4)
        fitted = _fit_of(placed) if flat else None
        xyz.append(own @ move[:3, :3].T + move[:3, 3] if fitted is None else cloud_align.flatten(own, move, fitted)[0])
        rgb.append(colours if colours is not None else np.full((len(own), 3), 160, dtype=np.uint8))
        view.append(np.full(len(own), index, dtype=np.uint16))
    none = np.zeros(0, dtype=np.uint16)
    if not xyz:
        return ply.dumps(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8), {"view": none, "outside": none})
    every = np.concatenate(xyz)
    beyond = cloud_align.outside(every[:, :2], found.rooms).astype(np.uint16)
    return ply.dumps(every, np.concatenate(rgb), {"view": np.concatenate(view), "outside": beyond})


def _kept_drawing(folder: Path, path: Path, others: str, make: Callable[[], bytes]) -> Path:
    """`path`, written by `make` on the first asking; the drawings matching `others` that aren't `keep`'s go."""
    if not path.is_file():
        folder.mkdir(parents=True, exist_ok=True)
        places.write_whole(path, make())
        mine = path.name.rsplit(".", 2)[0]  # the two drawings of one scene, flat and not, are kept side by side
        for old in folder.glob(others):
            if old.name.rsplit(".", 2)[0] != mine:
                old.unlink(missing_ok=True)
    return path


def drawn(store: camera_clouds.Store, found: Scene, flat: bool = True) -> Path:
    """The file of `points` for `found`, made on the first asking and kept until the scene is another (`found.id`)."""
    folder = store.folder / "scenes"
    name = f"{found.place}.{found.id}.{'flat' if flat else 'raw'}.ply"
    return _kept_drawing(folder, folder / name, f"{found.place}.*.ply", lambda: points(store, found, flat))


def own_pose(placed: Placed, cloud: camera_clouds.Cloud) -> np.ndarray:
    """A fitted camera's pose in its own cloud's frame, resized with the cloud (`own_points`)."""
    size = np.asarray(placed.transform, dtype=np.float64).reshape(4, 4)[:3, :3]
    across, upward = float(np.linalg.norm(size[:2, 0])), float(size[2, 2])
    return np.diag([across, across, upward, 1.0]) @ np.asarray(cloud.pose, dtype=np.float64).reshape(4, 4)


def own_points(store: camera_clouds.Store, placed: Placed, cloud: camera_clouds.Cloud) -> bytes:
    """One fitted camera's cloud for its own view, drawn flat: fitted to the room and flattened as in the scene,
    then brought back to the cloud's own frame (its camera on the z axis, looking along +y), so the view of one
    camera shows the same flat floor and walls."""
    own, colours, _ = ply.read(store.file(cloud))
    move = np.asarray(placed.transform, dtype=np.float64).reshape(4, 4)
    fitted = _fit_of(placed)
    scene = own @ move[:3, :3].T + move[:3, 3] if fitted is None else cloud_align.flatten(own, move, fitted)[0]
    rigid = move @ np.linalg.inv(own_pose(placed, cloud) @ np.linalg.inv(np.asarray(cloud.pose).reshape(4, 4)))
    back = np.linalg.inv(rigid)
    fill = colours if colours is not None else np.full((len(own), 3), 160, dtype=np.uint8)
    return ply.dumps(scene @ back[:3, :3].T + back[:3, 3], fill)


def own_key(placed: Placed) -> str:
    """Changes when a fitted camera's own flat drawing does: with its cloud or its fit."""
    return hashlib.sha256(json.dumps([placed.cloud, placed.transform, placed.fit]).encode()).hexdigest()[:32]


def drawn_own(store: camera_clouds.Store, placed: Placed, cloud: camera_clouds.Cloud) -> Path:
    """The file of `own_points`, kept until the camera's cloud or its fit is another."""
    folder = store.folder / "scenes"
    name = f"own-{camera_clouds.slug(placed.camera)}.{own_key(placed)}.flat.ply"
    pattern = f"own-{camera_clouds.slug(placed.camera)}.*.ply"
    return _kept_drawing(folder, folder / name, pattern, lambda: own_points(store, placed, cloud))


def as_dict(found: Scene) -> dict[str, Any]:
    return {"place": found.place, "id": found.id, "cameras": [asdict(p) | {"placed": p.placed} for p in found.cameras],
            "pairs": [asdict(p) for p in found.pairs], "covered": found.covered}  # fmt: skip


Spots = Callable[[], Mapping[str, SpotAt]]
