"""A camera's pose from the spot the owner gave it on the plan (part of spatial memory).

The owner's plan of a place can hold each fixed camera's spot: where it is, with its height, its facing, and its
field of view when the owner gave them (spatial/plan_scene.py holds them; spatial/plan_import.py reads them). A spot is taught state, and it is enough to place the people that
camera sees on the plan's floor with no map at all: where the camera is, how high, and which way it faces come from
the spot, its focal and lens are the held ones (spatial/lens.py `held`; the spot's `fov` when none is held), and the
one thing a spot doesn't say, how far the camera is tilted, is worked out from the people it sees (`aim`).

**The pose** (`read`, `Spot.pose`) is in the plan's own frame: x and y the plan's metres, and the third axis the
height above the floor, so the floor is z = 0 (`FLOOR`) and the map-to-plan transform is the identity (`PLAN`). A
spot's yaw is a bearing on the plan (clockwise from up the page, so the direction is (sin yaw, -cos yaw) in the
plan's x and y). Which side the camera's right hand is on depends on whether the plan's frame is the page as drawn
(x east, y south) or its mirror image (a frame that is right-handed seen from above, as a map's floor frame is):
`Spot.mirrored`, from the map the plan was drawn on when it has one, else from the owner's page (`PageView.mirror`).

**The tilt** (`aim`). A person walking is upright: a vertical line as tall at the far end of the room as at the
near end. Two things follow for a camera that sees them, and the tilt (pitch and roll) is the one that fits both
best (`up_from_people`, `misfit`). The ray to their feet and the ray to the top of their head span a plane that
holds the world's up, so up lies in every walker's plane. And with the camera tilted as it really is, each walker
comes out the same height wherever they stood, so the heights' spread is least at the true tilt (as a share of their
median, which doesn't depend on how high the camera is). Nobody's height is assumed. A roll is fitted only when
walkers cross enough of the view to tell one (`ROLL_SPAN`): from one strip of it, a roll fits the noise. Only people walking count
(`walking`): someone sitting isn't a vertical line, and sits in one spot for an hour. The fit is then checked against
the spot's measured height: with the camera that high and that tilted, the people it sees must come out a person's
height (`Aim.tall`, within `TALL`), and a fit that doesn't is refused, since the height, the lens, or the tilt is
then wrong and every point would be off.

Read only, on four live cameras on the Linux machine with 52 to 409 walking rows each: the heights' spread had one
clear minimum per camera (1 to 8% of the median, against 10 to 50% a further 15 degrees off), within 2 degrees of
where the planes alone put the pitch on the two cameras whose walkers crossed enough of the view to say, and the
people came out 1.4 to 1.6 m to the top of the head. Fitted from everyone whose head, hip, and feet were in a line
instead, seated people put one camera 79 degrees down.

An aim is machine-made (`places/aims.json`, deleted by a places reset) and fitted once per camera, from up to
`AIM_LOOKBACK_S` of its pose tracks, then kept until the spot or the held lens changes. Nothing fits it in a loop.

**What a camera lacks** (`Spot.missing`), for readers to say instead of placing a guess: `height` (the spot has
none), `facing` (no yaw), `lens` (no held focal and no `fov`), `tilt` (no aim yet: too few people seen walking),
or `check` (the aim's people come out the wrong height). A camera with nothing missing has a pose.

**The owner's spot outranks a map.** The placer places a camera with a spot through the spot, whatever a map's
registration says of it (`overlay`). A map may later refine the pose only within `REFINE_M` and `REFINE_DEG` of the
spot, and whoever applies that logs it (`within_bounds`); past the bound the spot holds and the disagreement is
logged.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from liberational import json_state, logs
from liberational.memory import places
from liberational.spatial import cameras, footprints, lens, plan_import

LOG = logs.get("spatial-spots")

AIMS = "aims.json"  # under the places folder: each camera's fitted aim
REFINED = "refined.json"  # under the places folder: each camera's plan check of its spot, and any refinement in bound
FLOOR = (0.0, 0.0, 1.0, 0.0)  # the plan's floor in the plan's own frame: height is the third axis
PLAN = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)  # and its map-to-plan transform: the identity
MISSING = ("height", "facing", "lens", "tilt", "check")

AIM_LOOKBACK_S = 3 * 86400.0  # how far back an aim reads a camera's pose tracks
AIM_ENOUGH = 1500  # walking people's rows that are plenty: the read stops going back there
AIM_MIN = 40  # and the fewest a fit is made from
STRAIGHT_DEG = 20.0  # an upright person's head, hip, and feet are a line in the picture, bent no more than this
MIN_SPAN = 0.08  # of the frame from head to feet: a smaller person's line is too short to trust
LEGS = (0.7, 1.8)  # hip to feet over head to hip, in the picture, for someone on their feet
STEP = (0.015, 0.2)  # of the frame their feet moved since the last look: walking, and not a jump to another person
PITCH = (-10.0, 85.0)  # degrees below level the search covers
ROLL = 30.0  # and degrees of roll either way
ROLL_COST = 0.01  # the misfit a roll of ROLL must win back
ROLL_SPAN = 0.3  # of the frame's width the walkers' feet must span (5th to 95th percentile) for a roll to be told
TALL = (1.0, 2.2)  # metres: the median height people come out at must be in this for the aim to hold
HEAD, HIP = 17, 19  # halpe26
REFINE_M, REFINE_DEG = 0.3, 5.0  # how far a map may move a camera from its owner's spot


@dataclass(frozen=True)
class Aim:
    """A camera's fitted tilt: the world's up in the camera's axes (x right, y down, z forward), the size of the
    frames it was fitted from, how many walking people's rows went in, the median height they come out at with the
    spot's height (None with no height), and what it was fitted against (`basis`, so a changed spot or lens fits
    again)."""

    up: tuple[float, float, float]
    width: int
    height: int
    people: int
    tall: float | None
    at: float
    basis: str

    @property
    def pitch(self) -> float:
        """Degrees the camera looks below level."""
        return math.degrees(math.asin(max(-1.0, min(1.0, -self.up[2]))))

    @property
    def roll(self) -> float:
        """Degrees it's rolled clockwise as seen from behind."""
        return math.degrees(math.atan2(-self.up[0], -self.up[1]))


@dataclass(frozen=True)
class Spot:
    """One camera's owner spot in a place, and the pose it gives (see the module docstring)."""

    camera: str
    place: str
    revision: str  # the revision whose plan frame the spot is in (`Place.plan_revision`)
    at: tuple[float, float]
    height: float | None
    yaw: float | None
    fov: float | None
    mirrored: bool  # the plan's frame is the mirror image of the page (right-handed seen from above)
    pose: places.CameraPose | None  # camera to plan (x, y, height above the floor); None while something is missing
    missing: tuple[str, ...]
    aim: Aim | None = None
    refined: places.CameraPose | None = None  # a map's within-bound refinement of `pose`, when `pose` still matches


def aims(state: Path) -> dict[str, Aim]:
    """Every camera's fitted aim; {} before any."""
    out: dict[str, Aim] = {}
    for camera, row in (json_state.read(places.root(state) / AIMS, {}) or {}).items():
        try:
            x, y, z = (float(v) for v in row["up"])
            tall = None if row["tall"] is None else float(row["tall"])
            out[str(camera)] = Aim(
                (x, y, z),
                int(row["width"]),
                int(row["height"]),
                int(row["people"]),
                tall,
                float(row["at"]),
                str(row["basis"]),
            )
        except KeyError, TypeError, ValueError:
            continue  # a row from another version: fitted again
    return out


def save_aims(state: Path, found: Mapping[str, Aim]) -> None:
    folder = places.root(state)
    folder.mkdir(parents=True, exist_ok=True)
    json_state.write(folder / AIMS, {c: asdict(a) for c, a in found.items()}, indent=1, sort_keys=True)


def refined_poses(state: Path) -> dict[str, dict[str, Any]]:
    """Every camera's stored within-bound spot refinement (`REFINED`); {} before any."""
    raw = json_state.read(places.root(state) / REFINED, {}) or {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for camera, row in raw.items():
        try:
            spot_p = tuple(float(v) for v in row["spot"])
            pose_p = tuple(float(v) for v in row["pose"])
            if len(spot_p) == 16 and len(pose_p) == 16:
                out[str(camera)] = {
                    "spot": spot_p,
                    "pose": pose_p,
                    "at": float(row["at"]),
                    "basis": row.get("basis") if isinstance(row.get("basis"), str) else None,
                    "check": row.get("check") if isinstance(row.get("check"), dict) else None,
                }
        except KeyError, TypeError, ValueError:
            continue
    return out


def save_refined(
    state: Path,
    camera: str,
    spot_pose: Sequence[float],
    refined_pose: Sequence[float] | None,
    now: float,
    basis: str | None = None,
    check: Mapping[str, Any] | None = None,
) -> None:
    """Save or clear `camera`'s refined pose against `spot_pose`, refusing anything outside `within_bounds`. With a
    `check` (`spot_check.SpotCheck.as_dict`) and the `basis` it was made on, the row is kept even when nothing was
    refined, so later views of the camera reuse the check instead of searching again."""
    folder = places.root(state)
    stored = {
        c: {k: (list(v) if k in ("spot", "pose") else v) for k, v in r.items() if v is not None}
        for c, r in refined_poses(state).items()
    }
    pose = list(spot_pose) if refined_pose is None else list(refined_pose)
    if check is None and np.allclose(spot_pose, pose, atol=1e-6):
        if camera not in stored:
            return
        del stored[camera]
    else:
        base = places.CameraPose(tuple(float(x) for x in spot_pose), 1.0, 1, 1)
        cand = places.CameraPose(tuple(float(x) for x in pose), 1.0, 1, 1)
        if not within_bounds(base, cand):
            return
        row: dict[str, Any] = {
            "spot": [float(x) for x in spot_pose],
            "pose": [float(x) for x in pose],
            "at": float(now),
        }
        if check is not None:
            row["check"] = dict(check)
            if basis is not None:
                row["basis"] = basis
        stored[camera] = row
    folder.mkdir(parents=True, exist_ok=True)
    json_state.write(folder / REFINED, stored, indent=1, sort_keys=True)


def _intrinsics(camera: str, fov: float | None, held: lens.Held | None, width: int, height: int) -> places.CameraPose:
    """A camera's rays, as a pose at the origin: its held focal and lens (`cameras.with_held`), else the spot's `fov`
    as the focal, with the camera's configured lens when it has one (`lens.as_registered`: the `fov` is then the
    undone frame's, as the map takes it). The caller has checked one of the two is there."""
    blank = places.CameraPose(tuple(float(x) for x in np.eye(4).ravel()), float(width), width, height)
    if held is not None:
        return cameras.with_held(camera, blank, held)
    focal = width / (2 * math.tan(math.radians(fov or 90.0) / 2))
    return replace(blank, focal=focal, lens=lens.as_registered(camera))


def _basis(spot: plan_import.Camera, held: lens.Held | None) -> str:
    """What an aim depends on besides the pictures: the spot's height and `fov`, the camera's configured lens when it
    has one, and the held focal and lens."""
    configured = lens.as_registered(spot.id)
    own = "" if configured is None else "/".join(f"{v:.5f}" for v in configured)
    lensed = "" if held is None else f"{held.focal:.5f}/{held.division:.5f}"
    return f"{spot.height}|{spot.fov}|{lensed}" + (f"|{own}" if own else "")


def rays(cam: places.CameraPose, uv: np.ndarray) -> np.ndarray:
    """(N, 3) unit rays in the camera's axes for (N, 2) pixels given as fractions of its frame, a wide lens undone."""
    out = np.empty((len(uv), 3))
    for i, pixel in enumerate(np.asarray(uv, dtype=np.float64).reshape(-1, 2).tolist()):
        uu, vv = places.undistorted(cam, pixel[0], pixel[1])
        out[i] = ((uu - 0.5) * cam.width / cam.focal, (vv - 0.5) * cam.height / cam.focal, 1.0)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def up_at(pitch: float, roll: float) -> np.ndarray:
    """The world's up in the axes of a camera looking `pitch` degrees below level and rolled `roll` degrees
    clockwise as seen from behind."""
    p, r = math.radians(pitch), math.radians(roll)
    return np.array([-math.cos(p) * math.sin(r), -math.cos(p) * math.cos(r), -math.sin(p)])


def heights(up: Sequence[float] | np.ndarray, above: float, feet: np.ndarray, heads: np.ndarray) -> np.ndarray:
    """How tall each person comes out (metres) from a camera `above` the floor with that `up`: their feet where
    the foot ray meets the floor (at footprints.FOOT_M), their head where the head ray passes nearest the vertical
    over them. Rays that don't reach the floor are left out."""
    g = np.asarray(up, dtype=np.float64)
    drop = feet @ g
    ok = drop < -1e-6
    foot = feet[ok] * ((footprints.FOOT_M - above) / drop[ok])[:, None]
    across = np.cross(g, heads[ok])
    length = (across * across).sum(axis=1)
    fine = length > 1e-9
    tall = -(np.cross(foot[fine], heads[ok][fine]) * across[fine]).sum(axis=1) / length[fine]
    return tall + footprints.FOOT_M


def misfit(up: np.ndarray, feet: np.ndarray, heads: np.ndarray, normals: np.ndarray) -> float:
    """How badly a camera's `up` fits its walking people (see the module docstring): the spread of the heights
    they come out at, as a share of their median, plus the median angle (radians) between `up` and each person's
    plane. Infinite when a tenth of them have feet that don't reach the floor, or their median height isn't above
    it."""
    tall = heights(up, 1.0, feet, heads)  # the spread as a share doesn't depend on how high the camera is
    if len(tall) < 0.9 * len(feet):
        return math.inf
    middle = float(np.median(tall))
    if middle <= 0:
        return math.inf
    spread = float(np.median(np.abs(tall - middle))) / middle
    return spread + float(np.median(np.abs(normals @ up)))


def up_from_people(feet: np.ndarray, heads: np.ndarray, roll: bool = True) -> tuple[float, float, float] | None:
    """The world's up in the camera's axes from walking people's foot and head rays ((N, 3) unit rays each): the
    pitch and roll that fit them best (`misfit`), searched coarsely over every pitch from PITCH[0] to PITCH[1] and
    roll within ROLL, then finely about the best. A roll costs a little (ROLL_COST at ROLL's end), and none is
    searched for without `roll` (walkers in one part of the view can't tell one: `aim`). None with fewer than
    AIM_MIN people."""
    normals = np.cross(feet, heads)
    size = np.linalg.norm(normals, axis=1)
    keep = size > 1e-6
    normals, feet, heads = normals[keep] / size[keep, None], feet[keep], heads[keep]
    if len(normals) < AIM_MIN:
        return None

    def cost(pitch: float, roll: float) -> float:
        return misfit(up_at(pitch, roll), feet, heads, normals) + ROLL_COST * (roll / ROLL) ** 2

    best = (math.inf, 0.0, 0.0)
    for step, reach in ((5.0, None), (1.0, 5.0), (0.2, 1.0)):
        _, at_pitch, at_roll = best
        pitches = (
            np.arange(PITCH[0], PITCH[1] + step, step)
            if reach is None
            else at_pitch + np.arange(-reach, reach + step / 2, step)
        )
        rolls = (
            np.arange(-ROLL, ROLL + step, step)
            if reach is None
            else at_roll + np.arange(-reach, reach + step / 2, step)
        )
        rolls = rolls if roll else np.zeros(1)
        for pitch in pitches:
            for tried in rolls:
                found = cost(float(pitch), float(tried))
                if found < best[0]:
                    best = (found, float(pitch), float(tried))
    if not math.isfinite(best[0]):
        return None
    x, y, z = (float(v) for v in up_at(best[1], best[2]))
    return (x, y, z)


def walking(rows: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """The foot and head pixels ((N, 2) each, fractions of the frame) of the people walking upright in one camera's
    pose.track columns, at most one per track a second: heels (else ankles), the top of the head, and the hip all
    placed; head, hip, and feet in a line bent no more than STRAIGHT_DEG and at least MIN_SPAN of the frame long,
    the legs LEGS times the trunk (a seated person's are foreshortened); and the feet STEP of the frame from where
    they were a second or so before (someone sitting for an hour is thousands of rows of one slouch)."""
    feet: list[tuple[float, float]] = []
    heads: list[tuple[float, float]] = []
    last: dict[int, tuple[float, float, float]] = {}
    count = len(rows["ts_ns"])
    for i in sorted(range(count), key=lambda i: int(rows["ts_ns"][i])):
        t, track = int(rows["ts_ns"][i]) / footprints.NS, int(rows["track_id"][i])
        before = last.get(track)
        if before is not None and t - before[0] < footprints.SAMPLE_S:
            continue
        joints = [int(j) for j in rows["joints"][i]]
        foot = footprints.foot_point([float(b) for b in rows["box"][i]], joints)
        if foot is None or len(joints) < 26 * 3:
            continue
        last[track] = (t, foot[0], foot[1])
        if before is None or t - before[0] > 3 * footprints.SAMPLE_S:
            continue
        if not STEP[0] <= math.hypot(foot[0] - before[1], foot[1] - before[2]) <= STEP[1]:
            continue
        scale = footprints.SCALE
        head, hip = ((joints[3 * j] / scale, joints[3 * j + 1] / scale, joints[3 * j + 2] / scale) for j in (HEAD, HIP))
        if min(head[2], hip[2]) < footprints.JOINT_SURE or head[0] < 0 or hip[0] < 0:
            continue
        trunk, legs = (hip[0] - head[0], hip[1] - head[1]), (foot[0] - hip[0], foot[1] - hip[1])
        a, b = math.hypot(*trunk), math.hypot(*legs)
        if a < 1e-6 or b < 1e-6 or math.hypot(foot[0] - head[0], foot[1] - head[1]) < MIN_SPAN:
            continue
        straight = (trunk[0] * legs[0] + trunk[1] * legs[1]) / (a * b) >= math.cos(math.radians(STRAIGHT_DEG))
        if not straight or not LEGS[0] <= b / a <= LEGS[1]:
            continue
        feet.append((foot[0], foot[1]))
        heads.append((head[0], head[1]))
    return np.asarray(feet, dtype=np.float64).reshape(-1, 2), np.asarray(heads, dtype=np.float64).reshape(-1, 2)


def people(state: Path, camera: str, now: float) -> tuple[np.ndarray, np.ndarray]:
    """`camera`'s walking people (`walking`) from its pose tracks, read an hour at a time back from `now` until
    there are AIM_ENOUGH or AIM_LOOKBACK_S has been read."""
    from continuum_py import read_segments

    from liberational import segment_log
    from liberational.see.actions import POSE_SCHEMA
    from liberational.see.actions_worker import POSES

    root = segment_log.root(state)
    feet, heads = [np.zeros((0, 2))], [np.zeros((0, 2))]
    upto, n = now, 0
    while upto > now - AIM_LOOKBACK_S and n < AIM_ENOUGH:
        lo = upto - footprints.HOUR_S
        got = read_segments(root, POSES, POSE_SCHEMA, round(lo * footprints.NS), round(upto * footprints.NS), [camera])
        f, h = walking(got)
        feet.append(f)
        heads.append(h)
        n += len(f)
        upto = lo
    return np.concatenate(feet), np.concatenate(heads)


def aim(
    camera: str,
    spot: plan_import.Camera,
    held: lens.Held | None,
    size: tuple[int, int],
    feet: np.ndarray,
    heads: np.ndarray,
    now: float,
) -> Aim | None:
    """`camera`'s aim from its walking people's foot and head pixels (`walking`) in frames of `size` (width,
    height); None with too few of them to fit."""
    cam = _intrinsics(camera, spot.fov, held, *size)
    foot_rays, head_rays = rays(cam, feet), rays(cam, heads)
    across = np.percentile(feet[:, 0], [5, 95]) if len(feet) else np.zeros(2)
    up = up_from_people(foot_rays, head_rays, roll=float(across[1] - across[0]) >= ROLL_SPAN)
    if up is None:
        return None
    tall: float | None = None
    if spot.height is not None:
        found = heights(up, spot.height, foot_rays, head_rays)
        tall = round(float(np.median(found)), 3) if len(found) else None
    return Aim(up, size[0], size[1], len(feet), tall, now, _basis(spot, held))


def pose(spot: plan_import.Camera, found: Aim, mirrored: bool) -> tuple[float, ...] | None:
    """The camera-to-plan pose (16 numbers, row by row) of a spot with a height and a yaw, tilted as `found` says:
    the plan's x, y, and height of a point given in the camera's axes. None for a camera looking straight up or
    down, whose facing no yaw says."""
    if spot.height is None or spot.yaw is None:
        return None
    g = np.asarray(found.up, dtype=np.float64)
    ahead = np.array([0.0, 0.0, 1.0]) - g[2] * g  # the camera's forward, laid level
    if float(np.linalg.norm(ahead)) < 1e-3:
        return None
    ahead /= np.linalg.norm(ahead)
    right = np.cross(ahead, g)  # the camera's axes are right-handed: right is forward cross up
    r = math.radians(spot.yaw)
    d = (math.sin(r), -math.cos(r))
    side = (d[1], -d[0]) if mirrored else (-d[1], d[0])  # the facing's right hand on the plan
    out = np.eye(4)
    out[0, :3] = d[0] * ahead + side[0] * right
    out[1, :3] = d[1] * ahead + side[1] * right
    out[2, :3] = g
    out[:3, 3] = (spot.at[0], spot.at[1], spot.height)
    return tuple(float(x) for x in out.ravel())


def _mirrored(held: places.Place, base: places.Revision | None) -> bool:
    """Whether the frame `held`'s plan is in is the page's mirror image: by the map it was drawn on when its
    revision has one (a map's frame is right-handed, so a transform that keeps its handedness draws a mirror image),
    else by the owner's page."""
    transform = places.plan_frame(base) if base is not None else None
    if transform is not None:
        return float(np.linalg.det(np.asarray(transform, dtype=np.float64).reshape(3, 4)[:, :3])) > 0
    return held.view is not None and held.view.mirror


def read(state: Path, found: Mapping[str, Aim] | None = None) -> dict[str, Spot]:
    """Every camera with a spot on an owner plan that holds, by camera: its place, the spot, and its pose in the
    plan's frame or what it lacks for one. `found` are the aims to use in place of the stored ones (a dry run's). A
    camera with spots on two places' plans is read at the plan saved last."""
    fitted = aims(state) if found is None else found
    stored_refined = refined_poses(state) if found is None else {}
    intrinsics = lens.held(state)
    out: dict[str, Spot] = {}
    for held in places.places(state):
        if held.plan_source != "owner" or not held.plan or held.plan_revision is None:
            continue
        chain = {r.id: r for r in places.revisions(state, held.id)}
        mirrored = _mirrored(held, chain.get(held.plan_revision))
        for spot in plan_import.spots(held.plan):
            own = intrinsics.get(spot.id)
            aimed = fitted.get(spot.id)
            if aimed is not None and aimed.basis != _basis(spot, own):
                aimed = None  # fitted against another height or lens
            missing = [
                *(["height"] if spot.height is None else []),
                *(["facing"] if spot.yaw is None else []),
                *(["lens"] if own is None and spot.fov is None else []),
            ]
            if "lens" not in missing and aimed is None:
                missing.append("tilt")
            elif aimed is not None and aimed.tall is not None and not TALL[0] <= aimed.tall <= TALL[1]:
                missing.append("check")
            matrix = pose(spot, aimed, mirrored) if aimed is not None and not missing else None
            if aimed is not None and not missing and matrix is None:
                missing.append("facing")  # it looks straight down: a yaw says nothing of which way is up the frame
            posed = None
            refined = None
            if matrix is not None and aimed is not None:
                shape = _intrinsics(spot.id, spot.fov, own, aimed.width, aimed.height)
                posed = replace(shape, pose=matrix, at=aimed.at)
                row = stored_refined.get(spot.id)
                if (
                    row is not None
                    and np.allclose(row["spot"], matrix, atol=1e-6)
                    and not np.allclose(row["pose"], matrix, atol=1e-6)
                ):
                    cand = replace(posed, pose=row["pose"], at=row["at"])
                    if within_bounds(posed, cand):
                        refined = cand
            out[spot.id] = Spot(
                spot.id,
                held.id,
                held.plan_revision,
                spot.at,
                spot.height,
                spot.yaw,
                spot.fov,
                mirrored,
                posed,
                tuple(missing),
                aimed,
                refined,
            )
    return out


#: A camera's frame size (width, height) at a time, or None with no picture of it.
Sizes = Callable[[str, float], tuple[int, int] | None]


def frame_size(camera: str, t: float) -> tuple[int, int] | None:
    """The size of `camera`'s recorded frames near `t` (the latest picture within a few minutes before it)."""
    for back in (0.0, 30.0, 120.0, 600.0):
        picture = cameras.recorded(camera, t - back)
        if picture is not None:
            return int(picture.shape[1]), int(picture.shape[0])
    return None


def fit(state: Path, now: float, sizes: Sizes | None = None, save: bool = True) -> dict[str, Aim]:
    """Fit an aim for each camera with an owner spot that lacks only its tilt (`read`), from its walking people up
    to `now`, and keep it (unless `save` is False: a dry run). Returns every aim that holds afterwards. A camera with
    too few people walking in view is left for a later run, and said so."""
    stored = aims(state)
    intrinsics = lens.held(state)
    plans = {
        c.id: c
        for held in places.places(state)
        if held.plan_source == "owner" and held.plan
        for c in plan_import.spots(held.plan)
    }
    for camera, spot in read(state, stored).items():
        if "tilt" not in spot.missing:
            continue
        size = (sizes or frame_size)(camera, now)
        if size is None:
            LOG.info("%s: no recorded picture to size its frames by, so its tilt isn't fitted yet", camera)
            continue
        feet, heads = people(state, camera, now)
        found = aim(camera, plans[camera], intrinsics.get(camera), size, feet, heads, now)
        if found is None:
            LOG.info("%s: %d people seen walking, too few to work out its tilt (%d needed)", camera, len(feet), AIM_MIN)
            continue
        LOG.info(
            "%s: tilted %.1f degrees down, rolled %.1f, from %d people walking; they come out %s m tall",
            camera,
            found.pitch,
            found.roll,
            found.people,
            "?" if found.tall is None else f"{found.tall:.2f}",
        )
        stored[camera] = found
    if save:
        save_aims(state, stored)
    return stored


def overlay(rev: places.Revision, found: Mapping[str, Spot]) -> places.Revision:
    """`rev` as the placer reads it: when it's the revision an owner plan with spots is drawn in, each posed spot's
    camera is in it at the spot's pose (or its within-bound refinement `Spot.refined`), over any registration a map
    gave that camera, and a revision with no floor of its own takes the plan's (`FLOOR`, `PLAN`). The pose is moved
    from the plan's frame into the revision's map frame through its map-to-plan transform, so the usual geometry
    (`places.project`) lands it back on the plan."""
    mine = {c: s for c, s in found.items() if s.place == rev.place and s.revision == rev.id}
    if not mine:
        return rev
    transform = places.plan_frame(rev)
    floor, plan = (rev.floor, rev.plan_transform) if transform is not None else (FLOOR, PLAN)
    to_plan = np.eye(4)
    to_plan[:3] = np.asarray(transform or PLAN, dtype=np.float64).reshape(3, 4)
    to_map = np.linalg.inv(to_plan)
    held = {c: p for c, p in rev.cameras.items() if c not in mine}  # a spot outranks a map's registration
    for camera, spot in mine.items():
        chosen = spot.refined or spot.pose
        if chosen is not None:
            moved = to_map @ np.asarray(chosen.pose, dtype=np.float64).reshape(4, 4)
            held[camera] = replace(chosen, pose=tuple(float(x) for x in moved.ravel()))
    return replace(rev, cameras=held, floor=floor, plan_transform=plan)


def within_bounds(spot: places.CameraPose, other: places.CameraPose) -> bool:
    """Whether `other` (a map's pose of the camera, in the same frame) is within what a map may refine an owner's
    spot by: REFINE_M and REFINE_DEG."""
    a = np.asarray(spot.pose, dtype=np.float64).reshape(4, 4)
    b = np.asarray(other.pose, dtype=np.float64).reshape(4, 4)
    cos = (float(np.trace(a[:3, :3].T @ b[:3, :3])) - 1.0) / 2.0
    turned = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])) <= REFINE_M and turned <= REFINE_DEG
