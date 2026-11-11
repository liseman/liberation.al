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

**The picture's straight edges** (`edges_in`, `up_from_edges`). A camera's frame holds the walls it looks at: a
wall's floor line, the line its top makes, and the corners it shares with other walls are straight in the picture.
The plan says which headings its walls run at (`headings`), so with the spot's facing each of those heading vectors,
and the vertical, is known in the camera's axes: a straight edge's plane (through the camera) must hold one of them,
which is what a candidate tilt and focal are scored by (each edge's miss of the nearest direction, cut off at
`EDGE_TOL` degrees so furniture, shadows, and people pay no more than a wall's edge does). The focal is fitted with
the tilt because the vanishing points' places depend on both: on the wide-lens camera that raised this, a plumb-line
focal put the pitch at 6 to 13 degrees below level where the walkers put it at 17.8, and across focals of 0.22 to
0.45 of the width the two cues never agreed. A picture whose edges are `EDGE_MIN` agreeing across `EDGE_SPAN` of the
frame's width is evidence enough to fit from.

**Which cue wins.** The walkers' tilt is what holds where the picture's edges can't be used (too few, too
concentrated, or nothing on the plan's walls), and it is the check on the tilt the edges fit: the aim is the edges'
tilt at a focal where the walkers, fitted on their own there, agree with it within `AGREE_DEG` (their roll only
counts when they cross `ROLL_SPAN` of the view, so one strip of it can't outvote the picture) and the people come out
a person's height (`TALL`). Focals between the lens's own and the one the edges fit are tried for that agreement,
which is how a lens whose focal is off is found; the aim then keeps that focal (`Aim.focal`) and the pose casts its
rays with it. No focal agreeing means the edges aren't a fit to be trusted: the walkers' tilt stands, and the aim
says the edges didn't confirm it (`Aim.edges`, `Aim.confirmed`; `edge_note` says it in a dry run).

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
registration says of it (`overlay`). A refinement of the spot within `REFINE_M` and `REFINE_DEG` (`within_bounds`)
that an earlier version's map step stored (`REFINED`, `refined_poses`) is still read; nothing makes new ones at
present, and an owner's change to the spot drops it.
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
# The picture's straight edges (see the module docstring).
EDGE_PIXELS = 320  # the width a picture is read at to find its straight edges
CANNY = (40, 120)  # a pixel's gradient (0 to 255) that starts an edge of them, and that keeps one going
EDGE_SHORT = 0.08  # of the frame's diagonal: a shorter straight edge is clutter, not a wall
EDGE_DUP = 0.02  # of the diagonal: a straight edge this near a longer one on the same bearing is the same edge
EDGE_DUP_DEG = 2.0  # degrees apart a longer edge's bearing has to be to be another edge
EDGE_MIN = 8  # straight edges that agree with the plan's directions enough to fit a tilt (and a focal) from
EDGE_SPAN = 0.35  # of the frame's width the agreeing edges must span (5th to 95th percentile of their middles)
EDGE_TOL = 1.5  # degrees a straight edge may miss the plan direction it lies along and still agree
HEADINGS = 3  # the plan's wall headings kept, the longest wall runs first (the vertical is always a direction too)
HEADING_DEG = 5.0  # walls within this many degrees of each other are one heading
HEADING_M = 2.0  # metres of wall a heading needs to count
FLOOR_STEP = 0.5  # metres between the wall-floor points a picture is scored against
FLOOR_NEAR_M = 0.4  # metres: closer than this to the spot, the floor is under the camera, not in its view
FLOOR_FAR_M = 12.0  # metres: walls farther than this from the spot are too far to say where their floor lines are
WALL_NEAR = 0.03  # of the frame a wall's floor line's point may land from a straight edge and still be seen there
FOCAL_RANGE = (0.55, 1.95)  # how far either way a picture's edges may put the focal from the lens's, as its share
FOCAL_STEPS = 12  # focals tried across that range before the search refines one
AGREE_DEG = 2.5  # degrees between the edges' tilt and the walkers' at one focal for the edges to confirm the tilt
HEAD, HIP = 17, 19  # halpe26
REFINE_M, REFINE_DEG = 0.3, 5.0  # how far a map may move a camera from its owner's spot


@dataclass(frozen=True)
class Aim:
    """A camera's fitted tilt: the world's up in the camera's axes (x right, y down, z forward), the size of the
    frames it was fitted from, how many walking people's rows went in, the median height they come out at with the
    spot's height (None with no height), and what it was fitted against (`basis`, so a changed spot or lens fits
    again).

    `edges`, `edge_pitch`, and `edge_roll` are what the camera's picture's straight edges said (`up_from_edges`): how
    many of them agreed with the plan's wall directions and the vertical, and the tilt they put the camera at. `focal`
    is the focal (pixels of that frame) the tilt was fitted at when the edges fitted it, which the pose then casts its
    rays with (None: the held lens's own); and `confirmed` says the walkers, fitted on their own at that focal, agree
    with the tilt the aim holds. An aim fitted only from walkers has no edges and isn't confirmed."""

    up: tuple[float, float, float]
    width: int
    height: int
    people: int
    tall: float | None
    at: float
    basis: str
    edges: int = 0
    edge_pitch: float | None = None
    edge_roll: float | None = None
    focal: float | None = None
    confirmed: bool = False

    @property
    def pitch(self) -> float:
        """Degrees the camera looks below level."""
        return _pitch_roll(self.up)[0]

    @property
    def roll(self) -> float:
        """Degrees it's rolled clockwise as seen from behind."""
        return _pitch_roll(self.up)[1]


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
                int(row.get("edges") or 0),
                # a row from before the picture's edges were read has neither, and holds as it was
                None if row.get("edge_pitch") is None else float(row["edge_pitch"]),
                None if row.get("edge_roll") is None else float(row["edge_roll"]),
                None if row.get("focal") is None else float(row["focal"]),
                bool(row.get("confirmed") or False),
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


def _intrinsics(camera: str, fov: float | None, held: lens.Held | None, width: int, height: int) -> places.CameraPose:
    """A camera's rays, as a pose at the origin: its held focal and lens (`cameras.with_held`) when they were measured
    through the lens its frames are read through (`lens.through`), else the spot's `fov` as the focal, else the
    configured lens's own (`lens.undone_focal`); with the camera's configured lens when it has one
    (`lens.as_registered`: the `fov` is then the undone frame's, as the map takes it). The caller has checked one of
    the three is there (`_lacks_lens`)."""
    blank = places.CameraPose(tuple(float(x) for x in np.eye(4).ravel()), float(width), width, height)
    configured = lens.as_registered(camera)
    if lens.through(held, configured) is not None:
        return cameras.with_held(camera, blank, held)
    if fov is None and configured is not None:
        focal = lens.undone_focal(lens.Lens.from_tuple(configured)) * width
    else:
        focal = width / (2 * math.tan(math.radians(fov or 90.0) / 2))
    return replace(blank, focal=focal, lens=configured)


def _lacks_lens(spot: plan_import.Camera, held: lens.Held | None) -> bool:
    """Whether a camera has nothing to give its rays a focal: no held focal of the frames it is read through, no `fov`
    on its spot, and no configured lens."""
    configured = lens.as_registered(spot.id)
    return lens.through(held, configured) is None and spot.fov is None and configured is None


def _basis(spot: plan_import.Camera, held: lens.Held | None) -> str:
    """What an aim depends on besides the pictures: the spot's height and `fov`, the camera's configured lens when it
    has one, and the held focal and lens that apply to its frames (`lens.through`)."""
    configured = lens.as_registered(spot.id)
    held = lens.through(held, configured)
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


def _pitch_roll(up: Sequence[float] | np.ndarray) -> tuple[float, float]:
    """Degrees an `up` in a camera's axes is pitched below level (`up_at`'s pitch) and rolled clockwise seen from behind
    (`up_at`'s roll): the inverse of `up_at`."""
    g = np.asarray(up, dtype=np.float64)
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -float(g[2])))))
    return pitch, math.degrees(math.atan2(-float(g[0]), -float(g[1])))


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


# The picture's straight edges: the cameras' frames hold the walls around them, and a wall's floor line, its top
# line, and the corners between walls are straight (see the module docstring).


@dataclass(frozen=True)
class Room:
    """What a place's plan says that a camera's picture can be fitted against: the directions its walls run at and the
    vertical (`world`, (K, 3) in the plan's frame, the third axis the height above the floor), the points along the
    wall-floor lines its spot should see (`floor`, (M, 3) plan metres at the floor), and where its spot puts the
    eye."""

    world: np.ndarray
    floor: np.ndarray
    eye: tuple[float, float, float]


@dataclass(frozen=True)
class Edges:
    """What the straight edges of a camera's picture say about its tilt and focal, fitted against the plan's wall
    directions and the vertical (`up_from_edges`)."""

    pitch: float
    roll: float
    focal: float  # pixels, of the frame they were fitted from
    agreed: int  # the straight edges that lie along a plan direction
    spread: float  # of the frame's width the agreeing edges' middles span (5th to 95th percentile)
    misfit: float  # degrees: the median miss of the agreeing edges

    @property
    def usable(self) -> bool:
        """Whether the picture's edges are enough evidence to fit a tilt with: EDGE_MIN of them, across EDGE_SPAN of
        the frame's width."""
        return self.agreed >= EDGE_MIN and self.spread >= EDGE_SPAN


def headings(text: str) -> tuple[float, ...]:
    """The headings of a plan's walls (degrees clockwise from up the page, as a spot's yaw is), the longest wall runs
    first, up to HEADINGS of them: walls within HEADING_DEG of each other are one heading, and each needs HEADING_M
    metres of wall. () for a plan that has no rooms."""
    try:
        plan = plan_import.parse(text)
    except plan_import.PlanError:
        return ()
    walls = plan_import.walls(plan)
    runs = np.hypot(*(walls.b - walls.a).T)
    along = (walls.b - walls.a)[runs > 0] / runs[runs > 0, None]
    if not len(along):
        return ()
    bearing = np.degrees(np.arctan2(along[:, 0], -along[:, 1])) % 180.0
    groups: list[list[tuple[float, float]]] = []  # (bearing, metres of wall) per heading
    for way, length in zip(bearing.tolist(), runs[runs > 0].tolist(), strict=True):
        for group in groups:
            if _apart(way, group[0][0]) <= HEADING_DEG:
                group.append((way, length))
                break
        else:
            groups.append([(way, length)])
    kept = []
    for group in groups:
        total = sum(length for _, length in group)
        if total < HEADING_M:
            continue
        turns = np.radians([way for way, _ in group]) * 2.0  # bearings circle every 180 degrees
        middle = math.degrees(math.atan2(np.sin(turns).mean(), np.cos(turns).mean()) / 2.0) % 180.0
        kept.append((middle, total))
    kept.sort(key=lambda pair: -pair[1])
    return tuple(round(way, 3) for way, _ in kept[:HEADINGS])


def _apart(a: float, b: float) -> float:
    """How many degrees apart two bearings on a circle of 180 degrees are."""
    gap = abs(a - b) % 180.0
    return min(gap, 180.0 - gap)


def edges_in(picture: Any) -> np.ndarray:
    """The straight edges in a picture of a camera's frame, as (N, 4) fractions of the frame (u1, v1, u2, v2), the
    longest first: read EDGE_PIXELS wide, its Canny edges' straight runs, each at least EDGE_SHORT of the frame's
    diagonal long, with a run on the same bearing as a longer one and within EDGE_DUP of the diagonal of it dropped as
    that edge again. (0, 4) with no picture (`cameras.recorded` has none for a camera whose frames weren't kept)."""
    import cv2

    if picture is None:
        return np.zeros((0, 4))
    img = np.asarray(picture)
    if img.size == 0 or img.ndim not in (2, 3):
        return np.zeros((0, 4))
    if img.ndim == 3:
        grey = cv2.cvtColor(np.ascontiguousarray(img, dtype=np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        grey = np.ascontiguousarray(img, dtype=np.uint8)
    h, w = grey.shape[:2]
    scale = min(1.0, EDGE_PIXELS / float(max(w, 1)))
    if scale < 1.0:
        grey = cv2.resize(grey, (max(8, round(w * scale)), max(8, round(h * scale))), interpolation=cv2.INTER_AREA)
    gh, gw = grey.shape[:2]
    diagonal = math.hypot(gw, gh)
    found = cv2.HoughLinesP(
        cv2.Canny(cv2.GaussianBlur(grey, (5, 5), 0), CANNY[0], CANNY[1]),
        1,
        math.pi / 360.0,
        threshold=max(10, round(0.05 * min(gw, gh))),
        minLineLength=max(8, round(EDGE_SHORT * diagonal)),
        maxLineGap=max(2, round(0.01 * diagonal)),
    )
    if found is None:
        return np.zeros((0, 4))
    runs = np.asarray(found, dtype=np.float64).reshape(-1, 4)
    length = np.hypot(runs[:, 2] - runs[:, 0], runs[:, 3] - runs[:, 1])
    kept: list[int] = []
    for i in np.argsort(-length).tolist():
        a, b = runs[i, :2], runs[i, 2:]
        same = False
        for j in kept:
            c, d = runs[j, :2], runs[j, 2:]
            middle = math.hypot(*((a + b) / 2.0 - (c + d) / 2.0))
            if middle <= EDGE_DUP * diagonal and _apart(_bearing(a, b), _bearing(c, d)) <= EDGE_DUP_DEG:
                same = True
                break
        if not same:
            kept.append(i)
    out = runs[sorted(kept, key=lambda i: -length[i])]
    out[:, [0, 2]] /= gw
    out[:, [1, 3]] /= gh
    return out


def _bearing(a: np.ndarray, b: np.ndarray) -> float:
    """Degrees from the horizontal of the line from `a` to `b`, on a circle of 180 degrees."""
    return math.degrees(math.atan2(float(b[1] - a[1]), float(b[0] - a[0]))) % 180.0


def _world(headings: Sequence[float]) -> np.ndarray:
    """The plan's directions a picture's straight edges may lie along, (K, 3) in the plan's frame (the third axis the
    height above the floor): its wall `headings` along the floor, and the vertical."""
    out = np.zeros((len(headings) + 1, 3))
    for i, heading in enumerate(headings):
        r = math.radians(heading)
        out[i] = (math.sin(r), -math.cos(r), 0.0)
    out[-1] = (0.0, 0.0, 1.0)
    return out


def room(text: str, spot: plan_import.Camera) -> Room:
    """What `text`'s plan says a spot's picture can be fitted against (`Room`): the directions its walls run at
    (`headings`), and the points along the floor lines of the walls the spot should see, every FLOOR_STEP metres:
    FLOOR_NEAR_M to FLOOR_FAR_M from it, in front of its facing, and in its sight (`plan_import.in_sight`, so a wall
    behind another isn't counted). A spot with no height or yaw sees no floor lines, and a plan that doesn't read has
    neither."""
    found = headings(text)
    eye = (spot.at[0], spot.at[1], float(spot.height or 0.0))
    if not found or spot.height is None or spot.yaw is None:
        return Room(_world(found), np.zeros((0, 3)), eye)
    try:
        plan = plan_import.parse(text)
    except plan_import.PlanError:
        return Room(_world(found), np.zeros((0, 3)), eye)
    walls = plan_import.walls(plan)
    runs = []
    for a, b in zip(walls.a, walls.b, strict=True):
        t = np.linspace(0.0, 1.0, max(2, round(float(np.hypot(*(b - a))) / FLOOR_STEP) + 1))
        runs.append(a[None, :] * (1.0 - t)[:, None] + b[None, :] * t[:, None])
    if not runs:
        return Room(_world(found), np.zeros((0, 3)), eye)
    q = np.concatenate(runs)
    r = math.radians(spot.yaw)
    ahead = (q - np.asarray(spot.at)) @ np.array([math.sin(r), -math.cos(r)])
    far = np.hypot(*(q - np.asarray(spot.at)).T)
    seen = plan_import.in_sight(plan, plan_import.sight_walls(plan), spot.at, q)
    kept = (far >= FLOOR_NEAR_M) & (far <= FLOOR_FAR_M) & (ahead > 0.0) & seen
    return Room(_world(found), np.column_stack([q[kept], np.zeros(int(kept.sum()))]), eye)


def _edge_normals(shape: places.CameraPose, edges: np.ndarray) -> np.ndarray:
    """(N, 3) unit normals of the planes through the camera that hold each of `edges` ((N, 4) fractions of the frame):
    a world line seen along the edge lies in that plane, so the plane holds the line's world direction."""
    across = np.cross(rays(shape, edges[:, :2]), rays(shape, edges[:, 2:]))
    return across / np.maximum(np.linalg.norm(across, axis=1), 1e-12)[:, None]


def _edge_missed(normals: np.ndarray, turn: np.ndarray, world: np.ndarray) -> np.ndarray:
    """(N,) degrees each of `normals`' planes misses the nearest of `world`'s directions by, the directions seen from a
    camera `turn` turns: 0 when the plane holds one of them."""
    off = np.abs(normals @ (world @ turn).T)  # (N, K): the sine of each miss
    return np.degrees(np.arcsin(np.clip(off, 0.0, 1.0))).min(axis=1)


def _near(points: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """(M,) each of `points`' ((M, 2) fractions of the frame) distance to the nearest of `edges` ((N, 4) fractions)."""
    starts, ends = edges[None, :, :2], edges[None, :, 2:]
    along = ends - starts
    reach = ((points[:, None, :] - starts) * along).sum(axis=-1) / np.maximum((along * along).sum(axis=-1), 1e-12)
    nearest = starts + np.clip(reach, 0.0, 1.0)[..., None] * along
    return np.hypot(*(points[:, None, :] - nearest).transpose(2, 0, 1)).min(axis=1)


def _wall_missed(shape: places.CameraPose, turn: np.ndarray, room: Room, edges: np.ndarray) -> float:
    """How badly the wall-floor lines a spot should see (`Room.floor`) fit the picture's straight edges: the mean,
    over their points, of how far its tilt puts each from the nearest edge, squared and cut off at WALL_NEAR (so a
    couch in front of a wall pays no more than a wall does), as a share of WALL_NEAR squared; a point the tilt puts
    behind the camera or out of the frame pays the most, so a tilt that loses the walls it should see is no fit. 0
    for a spot that should see no floor lines."""
    if not len(room.floor):
        return 0.0
    camera = (room.floor - np.asarray(room.eye, dtype=np.float64)) @ turn  # plan metres to the camera's axes
    depth = camera[:, 2]
    near = np.full(len(camera), WALL_NEAR)  # what a point behind the camera or out of the frame pays
    in_front = depth > 0.05
    u = 0.5 + shape.focal * camera[in_front, 0] / depth[in_front] / shape.width
    v = 0.5 + shape.focal * camera[in_front, 1] / depth[in_front] / shape.height
    inside = (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
    if inside.any():
        seen = _near(np.column_stack([u[inside], v[inside]]), edges)
        near[np.flatnonzero(in_front)[inside]] = np.minimum(seen, WALL_NEAR)
    return float(np.mean((near / WALL_NEAR) ** 2))


def up_from_edges(
    shape: places.CameraPose,
    edges: np.ndarray,
    yaw: float,
    mirrored: bool,
    room: Room,
    focal: float | None = None,
    start: tuple[float, float] | None = None,
) -> Edges | None:
    """The tilt that fits the straight edges of `shape`'s frames best, and the focal when `focal` is None. Every edge
    is scored by how near one of `room.world`'s directions (the plan's wall headings and the vertical, in the plan's
    frame) its plane holds, from a camera facing `yaw` with `mirrored`'s handedness, EDGE_TOL and further counting the
    same (so clutter pays no more than a wall's edge does), each edge weighing how long it is, since a long edge's
    bearing is surer than a short one's; and `_wall_missed` scores the wall-floor lines the spot should see, whose
    places in the frame say where the focal and the pitch are, which their directions alone can't (a room seen
    straight on leaves a pitch and focal that fit its edges equally well). Searched coarsely over every pitch from
    PITCH[0] to PITCH[1], every roll within ROLL, and a focal from FOCAL_RANGE of `shape`'s, then finely about the
    best; or about `start`, a pitch and roll to refine, and at `focal` alone. None with too few straight edges to
    try."""
    length = np.hypot(edges[:, 2] - edges[:, 0], edges[:, 3] - edges[:, 1]) if len(edges) else np.zeros(0)
    edges = np.asarray(edges, dtype=np.float64).reshape(-1, 4)[length >= EDGE_SHORT]
    length = length[length >= EDGE_SHORT]
    if len(edges) < EDGE_MIN or not len(room.world):
        return None
    best = (math.inf, *(start or (30.0, 0.0)), float(focal if focal is not None else shape.focal))
    focals = np.geomspace(*FOCAL_RANGE, FOCAL_STEPS) * shape.focal if focal is None else np.array([float(focal)])
    stages = (
        ((1.0, 5.0), (0.2, 1.0), (0.05, 0.2))
        if start is not None
        else ((5.0, None), (1.0, 5.0), (0.2, 1.0), (0.05, 0.2))
    )
    for step, reach in stages:
        _, at_pitch, at_roll, _ = best
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
        for tried in focals:
            shaped = replace(shape, focal=float(tried))
            normals = _edge_normals(shaped, edges)
            for pitch in pitches:
                for roll in rolls:
                    turn = _turn(yaw, mirrored, up_at(float(pitch), float(roll)))
                    if turn is None:
                        continue
                    missed = _edge_missed(normals, turn, room.world)
                    cost = float(np.average(np.minimum(missed, EDGE_TOL) ** 2, weights=length)) / EDGE_TOL**2
                    cost += _wall_missed(shaped, turn, room, edges)
                    if cost < best[0]:
                        best = (cost, float(pitch), float(roll), float(tried))
        focals = best[3] * np.array([0.94, 1.0, 1.06, 0.97, 1.03]) if focal is None else focals
    turn = _turn(yaw, mirrored, up_at(best[1], best[2]))
    if not math.isfinite(best[0]) or turn is None:
        return None
    missed = _edge_missed(_edge_normals(replace(shape, focal=best[3]), edges), turn, room.world)
    agreed = missed <= EDGE_TOL
    middles = np.column_stack([(edges[:, 0] + edges[:, 2]) / 2.0, (edges[:, 1] + edges[:, 3]) / 2.0])[agreed]
    spread = float(np.ptp(np.percentile(middles[:, 0], [5, 95]))) if len(middles) else 0.0
    return Edges(
        best[1],
        best[2],
        best[3],
        int(agreed.sum()),
        spread,
        round(float(np.median(missed[agreed])), 3) if agreed.any() else float(EDGE_TOL),
    )


def _focals(held: float, edges: float) -> tuple[float, ...]:
    """The focals to fit a tilt with from a picture's edges and from its walkers: the edges' own, and four between it
    and the lens's, so the two cues can meet somewhere other than at either one's own."""
    if abs(edges - held) < 0.02 * held:
        return (float(edges),)
    return tuple(float(f) for f in np.linspace(held, edges, 5))


def edge_note(aim: Aim) -> str:
    """What a camera's picture's straight edges said about its tilt, after the aim a run fitted (`Aim`), for a reader
    to print: that they confirmed it, that they say another tilt (so the walkers' stands), or that there were too few
    of them to check it."""
    if aim.confirmed:
        return f"; its picture's {aim.edges} straight edges confirmed that tilt"
    if aim.edge_pitch is None or not aim.edges:
        return "; no straight edges in its picture to check its tilt against the plan's walls"
    said = f"its picture's {aim.edges} straight edges say {aim.edge_pitch:.1f} deg down and {aim.edge_roll:.1f} rolled"
    if aim.edges < EDGE_MIN:
        return f"; {said}, too few of them to fit its tilt with"
    return f"; {said}, which its walkers never matched, so their tilt stands"


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

    root = segment_log.root(state)
    feet, heads = [np.zeros((0, 2))], [np.zeros((0, 2))]
    upto, n = now, 0
    while upto > now - AIM_LOOKBACK_S and n < AIM_ENOUGH:
        lo = upto - footprints.HOUR_S
        got = read_segments(
            root, segment_log.POSE_TRACKS, POSE_SCHEMA, round(lo * footprints.NS), round(upto * footprints.NS), [camera]
        )
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
    edges: np.ndarray | None = None,
    room: Room | None = None,
    mirrored: bool = False,
) -> Aim | None:
    """`camera`'s aim from its walking people's foot and head pixels (`walking`) in frames of `size` (width, height),
    and, when a picture of it gave them, its straight `edges` (`edges_in`) scored against `room` (the plan's wall
    headings, the vertical, and the wall-floor lines the spot should see; see the module docstring): the edges' tilt
    and focal hold when the walkers at that focal agree with it, else the walkers' tilt stands. None with too few
    people to fit."""
    shape = _intrinsics(camera, spot.fov, held, *size)
    foot_rays, head_rays = rays(shape, feet), rays(shape, heads)
    across = np.percentile(feet[:, 0], [5, 95]) if len(feet) else np.zeros(2)
    span = float(across[1] - across[0]) if len(feet) else 0.0
    roll_said = span >= ROLL_SPAN  # walkers in one strip of the view can't tell a roll
    up = up_from_people(foot_rays, head_rays, roll=roll_said)
    if up is None:
        return None
    focal, confirmed = float(shape.focal), False
    said = None
    if room is not None and spot.yaw is not None and edges is not None and len(edges):
        said = up_from_edges(shape, edges, spot.yaw, mirrored, room)
        agreed: tuple[float, float, Edges] | None = None  # (how far the two cues are apart, focal, the edges' fit)
        if said is not None and said.usable:
            start = (said.pitch, said.roll)
            for tried in _focals(shape.focal, said.focal):
                shaped = replace(shape, focal=tried)
                seen = up_from_edges(shape, edges, spot.yaw, mirrored, room, focal=tried, start=start)
                walked = up_from_people(rays(shaped, feet), rays(shaped, heads), roll=roll_said)
                if seen is None or walked is None:
                    continue
                if spot.height is not None:  # the edges' tilt there must leave the people a person's height
                    came = heights(up_at(seen.pitch, seen.roll), spot.height, rays(shaped, feet), rays(shaped, heads))
                    if not len(came) or not TALL[0] <= float(np.median(came)) <= TALL[1]:
                        continue
                pitched, rolled = _pitch_roll(walked)
                apart = abs(pitched - seen.pitch) + (abs(rolled - seen.roll) if roll_said else 0.0)
                if apart <= AGREE_DEG * (2.0 if roll_said else 1.0) and (
                    agreed is None or abs(tried - said.focal) < abs(agreed[1] - said.focal)
                ):
                    agreed = (apart, tried, seen)  # the focal nearest the edges' own that the walkers agree with
        if agreed is not None:
            _, focal, said = agreed
            tilted = up_at(said.pitch, said.roll)
            up = (float(tilted[0]), float(tilted[1]), float(tilted[2]))
            confirmed = True
    tall: float | None = None
    if spot.height is not None:
        cast = rays(replace(shape, focal=focal), feet), rays(replace(shape, focal=focal), heads)
        found = heights(up, spot.height, *cast)
        tall = round(float(np.median(found)), 3) if len(found) else None
    return Aim(
        up,
        size[0],
        size[1],
        len(feet),
        tall,
        now,
        _basis(spot, held),
        said.agreed if said is not None else 0,
        None if said is None else said.pitch,
        None if said is None else said.roll,
        focal if confirmed else None,
        confirmed,
    )


def _turn(yaw: float, mirrored: bool, up: np.ndarray) -> np.ndarray | None:
    """The rotation (3, 3) from the camera's axes (x right, y down, z forward) into the plan's (x and y the plan's
    metres, the third the height above the floor) of a spot facing `yaw` with `up` in its axes; None for a camera
    looking straight up or down, whose facing no yaw says. The plan's frame is `mirrored`'s mirror image of the
    camera's handedness (see the module docstring)."""
    g = np.asarray(up, dtype=np.float64)
    ahead = np.array([0.0, 0.0, 1.0]) - g[2] * g  # the camera's forward, laid level
    size = float(np.linalg.norm(ahead))
    if size < 1e-3:
        return None
    ahead = ahead / size
    right = np.cross(ahead, g)  # the camera's axes are right-handed: right is forward cross up
    r = math.radians(yaw)
    d = (math.sin(r), -math.cos(r))
    side = (d[1], -d[0]) if mirrored else (-d[1], d[0])  # the facing's right hand on the plan
    out = np.empty((3, 3))
    out[0] = d[0] * ahead + side[0] * right
    out[1] = d[1] * ahead + side[1] * right
    out[2] = g
    return out


def pose(spot: plan_import.Camera, found: Aim, mirrored: bool) -> tuple[float, ...] | None:
    """The camera-to-plan pose (16 numbers, row by row) of a spot with a height and a yaw, tilted as `found` says:
    the plan's x, y, and height of a point given in the camera's axes. None for a camera looking straight up or
    down, whose facing no yaw says."""
    if spot.height is None or spot.yaw is None:
        return None
    turn = _turn(spot.yaw, mirrored, np.asarray(found.up, dtype=np.float64))
    if turn is None:
        return None
    out = np.eye(4)
    out[:3, :3] = turn
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
                *(["lens"] if _lacks_lens(spot, own) else []),
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
                if aimed.focal is not None:  # the focal its picture's edges fitted: the rays are cast at that one
                    shape = replace(shape, focal=aimed.focal)
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
    """Fit an aim for each camera with an owner spot that lacks only its tilt (`read`), from its walking people and
    its recorded picture's straight edges up to `now`, and keep it (unless `save` is False: a dry run). Returns every
    aim that holds afterwards. A camera with too few people walking in view is left for a later run, and said so."""
    stored = aims(state)
    intrinsics = lens.held(state)
    plans: dict[str, plan_import.Camera] = {}
    texts: dict[str, str] = {}  # each camera's plan, which its picture's edges are scored against
    for held in places.places(state):
        if held.plan_source != "owner" or not held.plan:
            continue
        for c in plan_import.spots(held.plan):
            plans[c.id] = c
            texts[c.id] = held.plan
    for camera, spot in read(state, stored).items():
        if "tilt" not in spot.missing:
            continue
        size = (sizes or frame_size)(camera, now)
        if size is None:
            LOG.info("%s: no recorded picture to size its frames by, so its tilt isn't fitted yet", camera)
            continue
        feet, heads = people(state, camera, now)
        found = aim(
            camera,
            plans[camera],
            intrinsics.get(camera),
            size,
            feet,
            heads,
            now,
            edges=edges_in(cameras.recorded(camera, now)),
            room=room(texts[camera], plans[camera]),
            mirrored=spot.mirrored,
        )
        if found is None:
            LOG.info("%s: %d people seen walking, too few to work out its tilt (%d needed)", camera, len(feet), AIM_MIN)
            continue
        LOG.info(
            "%s: tilted %.1f degrees down, rolled %.1f, from %d people walking; they come out %s m tall%s",
            camera,
            found.pitch,
            found.roll,
            found.people,
            "?" if found.tall is None else f"{found.tall:.2f}",
            edge_note(found),
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
