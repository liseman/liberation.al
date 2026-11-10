"""What a learned map needs before people can be placed on it: its floor, and its fixed cameras (#260).

A map learned from the cameras (spatial/surroundings_map.py) is a point cloud in a frame of its own: the first
frame's camera is the origin, so the map isn't level, and it has no floor and registers no camera. A foot point is
placed where its camera's ray meets the floor (`places.floor_point`), so this module gives the place's revision the
two things that takes, from the map itself:

- **The floor** (`floor`). Up is found from the map's own surfaces: the direction most of its flat patches face,
  starting from the views' rough up (cameras hang more or less upright), since a single camera pitched down 30
  degrees is 30 degrees wrong about up. The floor is then the lowest layer of the surfaces that face up
  (`cameras.floor_along`), and it must lie under every view (a camera is never under the floor). The map file is not
  rewritten: its frame never moves, and the revision's floor plane is the levelling, as every reader already takes
  it (`places.floor_frame`).
- **The fixed cameras** (`fixed`). A camera that isn't on a rig in the device manifest and whose views all sit at
  one pose (one still stretch by spatial/poses.py's measure of a move, `poses.stretches`) is a fixed camera: it's
  registered in the revision at its views' smoothed pose, with the focal of the pictures the map was built from
  and its held lens. A camera on a rig, or whose views sit at several poses, is moving: its foot points need the
  camera's pose at each moment, which nothing gives yet (#7), so it's left unregistered and its foot points stay
  unplaced, with the reason (`moving`).

**What it writes** (`advance`): one revision of the place with no map of its own (producer `PRODUCER`), carrying the
floor and the cameras, which later learned revisions carry over (their frame is the same). It is found once per map
and then left alone: a new map's floor replaces the held one only when it moved by more than `FLOOR_M` at the
views or tilted by more than `FLOOR_DEG`, and a camera's pose only when it moved by more than `CAMERA_M` or
`CAMERA_DEG`, so a map that grows without changing shape writes nothing and nothing downstream (the plan line-up,
the placer's check of where cameras put people) is set off. What it found per place, and why a camera isn't
registered, is in `state/places/<place>/positions.learned` (`status`) for readers to say.

**A camera with an owner's spot** on the plan is placed through the spot, not through this registration
(spatial/spots.py). It's registered here all the same: the plan line-up uses a camera's place in the map against its
spot. Once the map is lined up with the plan, the two poses are compared and the difference is logged: within
`spots.REFINE_M` and `spots.REFINE_DEG` it's noted, past them it's a warning, and either way the spot holds.

`advance` is called at the end of every map step and by the positions job (`pack-positions`), never on a loop of
its own. It takes `places.writing` around its write, and calls the plan line-up after it sets a floor.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from liberational import config as cfg
from liberational import logs
from liberational.memory import places
from liberational.spatial import cameras, lens, ply, poses, spots, surroundings_map

LOG = logs.get("surroundings-positions")

PRODUCER = "surroundings positions v1"
STATUS = "positions.learned"  # in the place's folder
UP_POINTS = 60_000  # the map's points read for its up
UP_CONE_DEG = (50.0, 20.0)  # flat patches within this of the up so far vote for the next, wide then narrow
MIN_ABOVE_M = 0.15  # every view sits at least this far above the floor
FLOOR_M, FLOOR_DEG = 0.05, 2.0  # a floor found again replaces the held one only past these
CAMERA_M, CAMERA_DEG = 0.1, 2.0  # and a camera's pose only past these
FOCAL = 0.02  # or when its focal changed by more than this share
REASONS = ("moving", "floor")  # why a camera of a learned map places nobody (`status`'s `unplaced`)


def rough_up(views: Sequence[surroundings_map.View]) -> tuple[float, float, float]:
    """The views' rough up in the map's frame: the mean of their cameras' up axes (`cameras.up_direction`)."""
    return cameras.up_direction({v.id: v.pose for v in views}, {v.id: (v.width, v.height) for v in views})


def up_of(points: np.ndarray, start: Sequence[float]) -> tuple[float, float, float]:
    """The map's up: the direction its flat patches face, found from `start` (see the module docstring). Each
    point's patch is its 16 nearest neighbours; a patch counts when it's flat and its normal is within the cone of
    the up so far, first UP_CONE_DEG[0] then UP_CONE_DEG[1]. `start` comes back when too few patches say."""
    from scipy.spatial import KDTree

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) > UP_POINTS:
        pts = pts[np.random.default_rng(0).choice(len(pts), UP_POINTS, replace=False)]
    up = np.asarray(start, dtype=np.float64)
    up = up / np.linalg.norm(up)
    if len(pts) < 100:
        return (float(up[0]), float(up[1]), float(up[2]))
    _, idx = KDTree(pts).query(pts, k=16)
    near = pts[idx] - pts[idx].mean(axis=1, keepdims=True)
    values, vectors = np.linalg.eigh(np.einsum("nki,nkj->nij", near, near))
    normals = vectors[:, :, 0][values[:, 0] < 0.05 * values[:, 1]]
    for cone in UP_CONE_DEG:
        along = normals @ up
        facing = normals[np.abs(along) > math.cos(math.radians(cone))]
        if len(facing) < 50:
            break
        # a ceiling's normal counts with the floor's: each is turned to face up before they're averaged
        mean = (facing * np.sign(facing @ up)[:, None]).mean(axis=0)
        if float(np.linalg.norm(mean)) < 1e-6:
            break
        up = mean / np.linalg.norm(mean)
    return (float(up[0]), float(up[1]), float(up[2]))


def floor(points: np.ndarray, views: Sequence[surroundings_map.View]) -> tuple[float, float, float, float] | None:
    """The map's floor plane (nx, ny, nz, d: a point's height above it is n . p + d), or None when the map has no
    layer of flat surfaces to be one, or the one found doesn't lie MIN_ABOVE_M under every view."""
    if not views:
        return None
    found = cameras.floor_along(points, up_of(points, rough_up(views)))
    if found is None:
        return None
    n, d = np.asarray(found[:3]), found[3]
    above = [float(n @ np.asarray(v.pose, dtype=np.float64).reshape(4, 4)[:3, 3] + d) for v in views]
    return found if min(above) >= MIN_ABOVE_M else None


def _tilt(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> float:
    cos = float(np.dot(a[:3], b[:3])) / (float(np.linalg.norm(a[:3])) * float(np.linalg.norm(b[:3])))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def floor_moved(was: Sequence[float], now: Sequence[float], views: Sequence[surroundings_map.View]) -> bool:
    """Whether a floor found again is another floor than the held one: tilted by more than FLOOR_DEG, or more than
    FLOOR_M higher or lower under any view."""
    if _tilt(was, now) > FLOOR_DEG:
        return True

    def above(plane: Sequence[float], at: np.ndarray) -> float:
        n = np.asarray(plane[:3], dtype=np.float64)
        return float(n @ at + plane[3]) / float(np.linalg.norm(n))

    spots_at = [np.asarray(v.pose, dtype=np.float64).reshape(4, 4)[:3, 3] for v in views]
    return any(abs(above(was, p) - above(now, p)) > FLOOR_M for p in spots_at)


def fixed(
    views: Sequence[surroundings_map.View],
    place: str,
    on_a_mount: Sequence[str],
    held: Mapping[str, lens.Held],
    now: float,
) -> tuple[dict[str, places.CameraPose], dict[str, str]]:
    """The fixed cameras among `views`' cameras, each as the revision keeps it, and the others with why they aren't
    one ("moving"). `on_a_mount` are the cameras the device manifest has on no rig (`cameras.fixed_cameras`)."""
    by_camera: dict[str, list[surroundings_map.View]] = {}
    for v in views:
        by_camera.setdefault(v.camera, []).append(v)
    registered: dict[str, places.CameraPose] = {}
    unplaced: dict[str, str] = {}
    for camera, seen in sorted(by_camera.items()):
        observed = [
            poses.Observation(v.t, camera, "walk", place, None, np.asarray(v.pose, dtype=np.float64).reshape(4, 4), 1.0)
            for v in seen
        ]
        still = poses.stretches(observed)
        pose = poses.smoothed(still[0]) if len(still) == 1 else None
        focals = [v.focal for v in seen if v.focal]
        if camera not in on_a_mount or pose is None or not focals:
            unplaced[camera] = "moving"
            continue
        newest = seen[-1]
        focal = float(np.median(focals)) * newest.width
        flat = tuple(float(x) for x in pose.ravel())
        found = places.CameraPose(flat, focal, newest.width, newest.height, now)
        registered[camera] = cameras.with_held(camera, found, held.get(camera))
    return registered, unplaced


def _turn(a: places.CameraPose, b: places.CameraPose) -> tuple[float, float]:
    """How far apart two poses of a camera are: metres, and degrees between the ways they look."""
    m, n = (np.asarray(p.pose, dtype=np.float64).reshape(4, 4) for p in (a, b))
    return float(np.linalg.norm(m[:3, 3] - n[:3, 3])), _tilt(m[:3, 2], n[:3, 2])


def settled(
    was: Mapping[str, places.CameraPose], found: Mapping[str, places.CameraPose]
) -> dict[str, places.CameraPose]:
    """`found` with each camera that hasn't moved past CAMERA_M and CAMERA_DEG, and whose focal is within FOCAL of
    what it was, kept exactly as `was` has it."""
    out = {}
    for camera, pose in found.items():
        held = was.get(camera)
        if held is not None:
            metres, degrees = _turn(held, pose)
            same_optics = (held.width, held.height) == (pose.width, pose.height) and held.lens == pose.lens
            if (
                metres <= CAMERA_M
                and degrees <= CAMERA_DEG
                and same_optics
                and abs(pose.focal / held.focal - 1) <= FOCAL
            ):
                out[camera] = held
                continue
        out[camera] = pose
    return out


def _path(state: Path, place: str) -> Path:
    return places.root(state) / place / STATUS


def status(state: Path, place: str) -> dict[str, Any] | None:
    """What was found for `place`'s learned map when it was last looked at, or None for a place with none:
    `{"map": <the map file looked at>, "floor": bool, "cameras": [registered fixed cameras], "unplaced": {camera:
    reason}, "at": when}`. A registered camera places nobody while the map has no floor (`unplaced`'s "floor")."""
    try:
        found = json.loads(_path(state, place).read_text())
    except OSError, ValueError:
        return None
    return found if isinstance(found, dict) else None


def compare_spots(state: Path, rev: places.Revision) -> None:
    """Log how each camera's pose in the map compares with its owner's spot, once the map is lined up with the plan
    (see the module docstring). The spot holds either way."""
    transform = rev.plan_transform
    if transform is None:
        return
    to_plan = np.eye(4)
    to_plan[:3] = np.asarray(transform, dtype=np.float64).reshape(3, 4)
    for camera, spot in spots.read(state).items():
        mapped = rev.cameras.get(camera)
        if spot.place != rev.place or spot.pose is None or mapped is None:
            continue
        on_plan = to_plan @ np.asarray(mapped.pose, dtype=np.float64).reshape(4, 4)
        metres, degrees = _turn(spot.pose, replace(mapped, pose=tuple(float(x) for x in on_plan.ravel())))
        within = metres <= spots.REFINE_M and degrees <= spots.REFINE_DEG
        LOG.log(
            logging.INFO if within else logging.WARNING,
            "%s: the map puts it %.2f m and %.1f degrees from its owner's spot (%s); the spot holds",
            camera,
            metres,
            degrees,
            "within what a map may refine" if within else "more than a map may refine: check the spot or the map",
        )


def _advance(state: Path, place: str, on_a_mount: Sequence[str], now: float) -> dict[str, Any] | None:
    rev = places.latest(state, place)
    if rev is None or rev.map is None or rev.map_format != "ply" or rev.frame is None:
        return status(state, place)
    prior = status(state, place)
    if prior is not None and prior.get("map") == rev.map:
        return prior  # this map was looked at already
    views = surroundings_map.views(state, place)
    if not views:
        return prior  # not a learned map
    try:
        points = ply.read(places.root(state) / place / rev.map)[0]
    except OSError, ply.PlyError:
        return prior  # pruned since the revision was read
    found = floor(points, views)
    plane = rev.floor
    if found is not None and (plane is None or floor_moved(plane, found, views)):
        plane = found
    registered, unplaced = fixed(views, place, on_a_mount, lens.held(state), now)
    # A registered camera whose kept frames have all been pushed out of the store since stays where it was
    registered = {**{c: p for c, p in rev.cameras.items() if c not in unplaced}, **settled(rev.cameras, registered)}
    if plane is None:
        unplaced |= dict.fromkeys(registered, "floor")
    wrote = False
    with places.writing(state):
        newest = places.latest(state, place)
        if newest is None or newest.map != rev.map:
            return prior  # a map step wrote meanwhile: the next call looks at that map
        if plane != newest.floor or registered != newest.cameras:
            places.revise(state, place, producer=PRODUCER, sources=(rev.id,), floor=plane, cameras=registered, now=now)
            wrote = True
        whole = {
            "map": rev.map,
            "floor": plane is not None,
            "cameras": sorted(registered),
            "unplaced": dict(sorted(unplaced.items())),
            "at": now,
        }
        places.write_whole(_path(state, place), json.dumps(whole, sort_keys=True).encode())
    LOG.info(
        "%s: %s; fixed cameras %s; unplaced %s%s",
        place,
        "floor found" if plane is not None else "no floor in the map yet",
        ", ".join(sorted(registered)) or "none",
        ", ".join(f"{c} ({why})" for c, why in sorted(unplaced.items())) or "none",
        "" if wrote else "; nothing changed",
    )
    return whole | {"looked": True, "wrote": wrote}


def advance(state: Path, now: float | None = None, manifest: Path | None = None) -> dict[str, dict[str, Any]]:
    """Give every learned map its floor and fixed cameras (see the module docstring), and return each place's
    `status`. A place whose map was looked at already costs one small read. After a floor or a camera is written, the
    plan line-up is run (it needs the floor); for each map newly looked at, the cameras' poses are compared with their
    owners' spots."""
    from liberational.spatial import plan_lineup

    at = time.time() if now is None else now
    on_a_mount = cameras.fixed_cameras(manifest or cfg.devices_file())
    out: dict[str, dict[str, Any]] = {}
    looked: list[str] = []
    wrote = False
    for place in places.place_ids(state):
        found = _advance(state, place, on_a_mount, at)
        if found is not None:
            wrote = wrote or bool(found.pop("wrote", False))
            if found.pop("looked", False):
                looked.append(place)
            out[place] = found
    if wrote:
        try:
            plan_lineup.advance(state, at)
        except (OSError, ValueError) as error:
            LOG.warning("couldn't line the maps up with their plans: %s", error)
    for place in looked:
        rev = places.latest(state, place)
        if rev is not None:
            compare_spots(state, rev)
    return out
