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
live in `pairs.json`, and one made from another picture or another spot is not used.

**3. A camera that was moved** (`record`). A camera's `anchor` is the picture it was first placed with at its spot.
Each new picture is compared with it. One that registers within `STILL` hasn't moved. One that registers farther
has: its cloud follows by what the pictures measured, and the new spot is suggested to the owner (`Placed.suggest`;
dashboard/places.py saves it as a new plan version when they accept, never before). One that doesn't register is
looking at other space: it is tried against the other cameras' pictures, and placed from the first that registers.
If none does, the camera leaves the scene, saying it was moved and isn't placed, until its spot is set again (which
makes a new anchor) or a picture registers. A moved camera's cloud is never left at its old spot.

**Square to the house** (`align`, spatial/cloud_align.py). Between 1 and 2: a cloud's own walls say how its camera
really faces, better than a facing set by hand. A camera placed from its spot is turned, and where its walls have
clear counterparts slid, so its walls run along the plan's (bounded; a rigid move). A camera squared this way holds:
the plan is the reference, so a pair's measurement no longer moves it, only a partner that couldn't be squared, and
between two squared cameras the measurement is a check that says how far off it is ("checked"). The squaring is
kept beside the cloud and used only while it is of the same cloud, spot, and plan.

What a scene is good for: reliefs standing in the right places, not a whole room. Clouds that overlap meet with a seam
(the pair above is right to tens of centimetres), and cameras that share no space are only as right as their spots.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from liberational import json_state, logs
from liberational.memory import places
from liberational.spatial import camera_clouds, cloud_align, ply

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


def plan_spot(at: At, mirrored: bool) -> tuple[float, float, float]:
    """`scene_at` undone: a scene position as a spot's plan x, y, and facing (degrees, 0 to 360)."""
    y = at.y if mirrored else -at.y
    yaw = math.degrees(math.atan2(math.sin(at.bearing), -math.cos(at.bearing) if mirrored else math.cos(at.bearing)))
    return at.x, y, yaw % 360.0


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
    """What became of `camera` when its picture last stopped matching its anchor (`record`), or None."""
    got = json_state.read(_camera_file(store, camera, "moved.json"))
    return got if isinstance(got, dict) else None


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
    gone = moved(store, camera)
    if gone is not None and gone.get("cloud") == was.id:
        json_state.write(_camera_file(store, camera, "moved.json"), {**gone, "cloud": now.id})
    LOG.info("%s: its cloud had no floor and stood as if the camera were upright; stood up by its walls", camera)
    return True


def plan_rooms(state: Path, place: str, mirrored: bool) -> tuple[tuple[tuple[float, float], ...], ...]:
    """The rooms of `place`'s plan, when the owner drew it, in the scene's frame; none otherwise."""
    held = places.get(state, place)
    return rooms_of(held.plan if held is not None and held.plan_source == "owner" else None, mirrored)


def align(
    store: camera_clouds.Store, camera: str, spots: Mapping[str, SpotAt], rooms: Callable[[str, bool], Sequence[Ring]]
) -> bool:
    """Bring `camera`'s squaring to the plan up to date (the module docstring): made when it has a cloud, a spot with
    a facing, and none that is of this cloud, spot, and plan (`rooms`: a place's plan rooms, `plan_rooms` with the
    state bound). Whether one was made. Writes only beside the cloud."""
    cloud, spot = store.current(camera), spots.get(camera)
    folder = _camera_file(store, camera, cloud_align.ALIGN).parent
    if cloud is None or spot is None or spot.at is None:
        return False
    ring = rooms(spot.place, spot.mirrored)
    have = cloud_align.kept(folder)
    if have is not None and have.holds(cloud.id, spot.basis, cloud_align.plan_key(ring)):
        return False
    points, _, _ = ply.read(store.file(cloud))
    found = cloud_align.make(cloud.id, points, (spot.at.x, spot.at.y, spot.at.bearing), spot.basis, ring)
    cloud_align.keep(folder, found)
    LOG.info("%s: squared to the plan: %s, turned %.1f degrees, moved %.2f m (%d walls, %d with a counterpart)", camera,
             found.square.state, found.square.turn_deg, math.hypot(found.square.right_m, found.square.ahead_m),
             found.square.walls, found.square.slid_on)  # fmt: skip
    return True


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


KEPT = ("anchor.json", "anchor.png", "moved.json")  # what `accept` sets aside, for `take_back` to put back


def accept(store: camera_clouds.Store, camera: str, basis: str) -> None:
    """The owner set `camera`'s spot (to `basis`) from a suggestion: its picture now is the one it is placed with.
    What it was placed with before is set aside, so the owner can take the change back (`take_back`)."""
    cloud = store.current(camera)
    if cloud is None or not store.picture(cloud).is_file():
        return
    for name in KEPT:
        was, aside = _camera_file(store, camera, name), _camera_file(store, camera, f"before.{name}")
        aside.unlink(missing_ok=True)
        if was.is_file():
            shutil.copyfile(was, aside)
    _set_anchor(store, cloud, basis)


def take_back(store: camera_clouds.Store, camera: str) -> bool:
    """Undo `accept` for `camera`: the picture it was placed with before, and that it had moved from it, hold
    again, so with its old spot back it is a moved camera once more, not one left at its old place. Whether there
    was anything to put back."""
    asides = {name: _camera_file(store, camera, f"before.{name}") for name in KEPT}
    if not asides["anchor.json"].is_file():
        return False
    for name, aside in asides.items():
        now = _camera_file(store, camera, name)
        now.unlink(missing_ok=True)
        if aside.is_file():
            aside.replace(now)
    return True


# ---- the scene


@dataclass(frozen=True)
class Placed:
    """One camera in a place's scene, or why it isn't in it."""

    camera: str
    cloud: str  # its cloud's id
    points: int
    #: "spot" (from its spot alone), "adjusted" (from its spot, corrected against `against`), "moved" (it moved and
    #: its cloud followed), "moved_against" (it moved, and is placed from `against`'s picture), or, not in the scene,
    #: "no_spot", "no_facing", "moved_unplaced".
    how: str
    size: str  # "approximate": the model's own metres
    transform: tuple[float, ...] | None = None  # its cloud's level frame to the scene, 4 x 4 row by row
    pose: tuple[float, ...] | None = None  # its camera in the scene
    against: tuple[str, ...] = ()
    shift_m: float = 0.0  # how far from its spot it is placed
    turn_deg: float = 0.0
    suggest: tuple[float, float, float] | None = None  # the spot to offer: plan x, y, and facing in degrees
    at: float = 0.0  # when its cloud's picture was taken
    level: str = "none"  # what its cloud was levelled by: "floor", "walls", or "none"
    tilt_deg: float = 0.0  # the camera's tilt (down) and roll, by that
    roll_deg: float = 0.0
    #: Its squaring to the plan (spatial/cloud_align.py `Square`, as a dict), when one holds; `squared` when it is
    #: what placed it.
    square: Mapping[str, Any] | None = None
    squared: bool = False

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


def _transform(at: At) -> np.ndarray:
    s, c = math.sin(at.bearing), math.cos(at.bearing)
    out = np.eye(4)
    out[:3, :3] = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    out[:2, 3] = (at.x, at.y)
    return out


#: A room's outline on the floor of the scene, as (x, y) corners.
Ring = Sequence[tuple[float, float]]
MARGIN_M = 0.3  # a point this far outside every room of the plan is still drawn


def rooms_of(plan: str | None, mirrored: bool) -> tuple[tuple[tuple[float, float], ...], ...]:
    """The rooms of an owner's plan as outlines in the scene's frame; none for a plan that doesn't read."""
    from liberational.spatial import plan_scene

    try:
        found = plan_scene.rooms(plan_scene.load(plan)) if plan else []
    except plan_scene.SceneError:
        return ()
    return tuple(tuple((float(x), float(y) if mirrored else -float(y)) for x, y in outline) for *_, outline in found)


def outside(xy: np.ndarray, rooms: Sequence[Ring], margin: float = MARGIN_M) -> np.ndarray:
    """Which of the points `xy` (N x 2) lie outside every one of `rooms` by more than `margin`. With no rooms, none:
    there is nothing to be outside of."""
    if not len(rooms) or not len(xy):
        return np.zeros(len(xy), dtype=bool)
    inside = np.zeros(len(xy), dtype=bool)
    x, y = xy[:, 0], xy[:, 1]
    for ring in rooms:
        corners = np.asarray(ring, dtype=np.float64)
        within = np.zeros(len(xy), dtype=bool)
        for (x1, y1), (x2, y2) in zip(corners, np.roll(corners, -1, axis=0), strict=True):
            crosses = (y1 > y) != (y2 > y)
            with np.errstate(divide="ignore", invalid="ignore"):
                within ^= crosses & (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1)
        inside |= within
    out = np.flatnonzero(~inside)
    near = np.zeros(len(out), dtype=bool)
    for ring in rooms:  # a point just past a wall still belongs to the room
        corners = np.asarray(ring, dtype=np.float64)
        for a, b in zip(corners, np.roll(corners, -1, axis=0), strict=True):
            edge = b - a
            along = np.clip(((xy[out] - a) @ edge) / max(float(edge @ edge), 1e-12), 0.0, 1.0)
            near |= np.linalg.norm(xy[out] - (a + along[:, None] * edge), axis=1) <= margin
    inside[out[near]] = True
    return ~inside


def scene(store: camera_clouds.Store, place: str, spots: Mapping[str, SpotAt], rooms: Sequence[Ring] = ()) -> Scene:
    """`place`'s scene as it stands now: every camera of the place that has a cloud, placed (the module docstring) or
    with why it isn't. `rooms` are the plan's, for the view to leave out what lies outside them, and for a squaring
    to be of. Reads only; nothing is written."""
    clouds = {c: cloud for c in store.cameras() if (cloud := store.current(c)) is not None}
    mine = {c: clouds[c] for c in clouds if spots.get(c) is None or spots[c].place == place}
    plan = cloud_align.plan_key(rooms)
    squares: dict[str, cloud_align.Square] = {}  # each camera's squaring that still holds
    for camera, cloud in mine.items():
        spot = spots.get(camera)
        kept_align = cloud_align.kept(_camera_file(store, camera, cloud_align.ALIGN).parent)
        if spot is not None and kept_align is not None and kept_align.holds(cloud.id, spot.basis, plan):
            squares[camera] = kept_align.square
    firm: set[str] = set()
    start: dict[str, At] = {}  # the cameras placed from their own spots
    how: dict[str, str] = {}
    later: dict[str, dict[str, Any]] = {}  # the ones placed from another camera
    for camera, cloud in mine.items():
        spot = spots.get(camera)
        if spot is None or spot.place != place:
            how[camera] = "no_spot"
        elif spot.at is None:
            how[camera] = "no_facing"
        else:
            held, was = anchor(store, camera), moved(store, camera)
            current = (
                was is not None
                and held is not None
                and held.get("basis") == spot.basis
                and was.get("cloud") == cloud.id
            )
            if not current or was is None:
                start[camera], how[camera] = spot.at, "spot"
                if camera in squares and squares[camera].state == "squared":
                    start[camera] = spot.at.step(*squares[camera].rel)
                    firm.add(camera)
            elif was["state"] == "followed":
                start[camera], how[camera] = spot.at.step(*was["rel"]), "moved"
            elif was["state"] == "against":
                later[camera] = was
            else:
                how[camera] = "moved_unplaced"

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
    for camera, was in later.items():  # placed from another camera's picture: after that camera is placed
        other = was.get("other")
        if other in at and other in mine and mine[other].id == was.get("other_cloud"):
            at[camera], how[camera] = at[other].step(*was["rel"]), "moved_against"
            partners[camera] = [other]
        else:
            how[camera] = "moved_unplaced"

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
        }
        if camera not in at:
            out.append(Placed(camera, cloud.id, cloud.points, how[camera], cloud.size, **about))
            continue
        transform = _transform(at[camera])
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
                tuple(sorted(partners.get(camera, ()))) if kind in ("adjusted", "moved_against") else (),
                round(shift, 3),
                round(turn, 1),
                plan_spot(at[camera], spot.mirrored)
                if spot is not None and kind in ("moved", "moved_against")
                else None,
                **about,
            )
        )
    mirrored = next((spots[c].mirrored for c in mine if spots.get(c) is not None), False)
    kept = tuple(tuple((float(x), float(y)) for x, y in ring) for ring in rooms)
    drawn = json.dumps([[[p.camera, p.cloud, p.transform] for p in out if p.placed], kept])
    return Scene(place, tuple(out), tuple(told), mirrored, kept, hashlib.sha256(drawn.encode()).hexdigest()[:32])


def points(store: camera_clouds.Store, found: Scene) -> bytes:
    """The scene's placed clouds as one PLY for the 3D view: every cloud's points moved by its transform, each with
    its camera's place in `found.cameras` as `view`, so the view can show, hide, and tint a camera's points, and
    with `outside` 1 for a point more than `MARGIN_M` outside every room of the plan, which the view leaves out
    unless asked (a relief's depth is the model's estimate, and can run through a wall). The clouds themselves aren't
    touched: this is the drawing, made when asked for."""
    xyz, rgb, view = [], [], []
    for index, placed in enumerate(found.cameras):
        if placed.transform is None:
            continue
        cloud = store.current(placed.camera)
        if cloud is None or cloud.id != placed.cloud:
            continue
        own, colours, _ = ply.read(store.file(cloud))
        move = np.asarray(placed.transform, dtype=np.float64).reshape(4, 4)
        xyz.append(own @ move[:3, :3].T + move[:3, 3])
        rgb.append(colours if colours is not None else np.full((len(own), 3), 160, dtype=np.uint8))
        view.append(np.full(len(own), index, dtype=np.uint16))
    none = np.zeros(0, dtype=np.uint16)
    if not xyz:
        return ply.dumps(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8), {"view": none, "outside": none})
    every = np.concatenate(xyz)
    beyond = outside(every[:, :2], found.rooms).astype(np.uint16)
    return ply.dumps(every, np.concatenate(rgb), {"view": np.concatenate(view), "outside": beyond})


def drawn(store: camera_clouds.Store, found: Scene) -> Path:
    """The file of `points` for `found`, made on the first asking and kept until the scene is another (`found.id`)."""
    folder = store.folder / "scenes"
    path = folder / f"{found.place}.{found.id}.ply"
    if not path.is_file():
        folder.mkdir(parents=True, exist_ok=True)
        places.write_whole(path, points(store, found))
        for old in folder.glob(f"{found.place}.*.ply"):
            if old != path:
                old.unlink(missing_ok=True)
    return path


def as_dict(found: Scene) -> dict[str, Any]:
    return {"place": found.place, "id": found.id, "cameras": [asdict(p) | {"placed": p.placed} for p in found.cameras],
            "pairs": [asdict(p) for p in found.pairs]}  # fmt: skip


Spots = Callable[[], Mapping[str, SpotAt]]
