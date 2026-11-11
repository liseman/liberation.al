"""Standing a camera's cloud upright, square to the house, and fitted to its rooms (#466, #471).

A cloud placed by its camera's spot alone is only as straight as that spot's facing and the cloud's own level. The
cloud holds better evidence of both: its floor is horizontal, its walls are vertical, and its walls run along the
walls of the owner's plan. Two rigid moves use that, and neither stretches or bends anything.

**The flat surfaces** (`planes`). The big planes of a cloud, found one at a time: the plane with the most points
within `TOL_M` of it, refitted to those points, taken out, and again, until none holds `MIN_SHARE` of the points or
`MOST` are found. A plane counts only when its points spread over it (`MIN_SPREAD_M`): a strip gives no direction.
The search then goes on for tall standing planes alone, down to `WALL_SHARE`: a camera hung high and looking down fills
its picture with floor and table tops, and the one wall at its edge held 0.013 of one live cloud's points, under the
bar for a plane, so that cloud was never squared or fitted and a pair's measurement pulled it off its wall.

**Level** (`stand`). A cloud whose floor was found (spatial/camera_clouds.py `floor`) is level already: its tilt and
roll are its floor's. One with no floor in view was taken to be upright, which a tilted camera isn't. Its walls say
which way is up: up runs along every wall, so two walls that meet give it outright and one wall gives the lean
across itself (the lean along it stays as it was). The cloud is turned so that up is up, its lowest points on the
floor. With no wall either, it stays as it was, and says so.

**Square** (`square`). The levelled cloud's walls, where its camera's spot puts them, are compared with the plan's
walls near the camera. *Turn:* the turn about the vertical, through the camera, that best lays the cloud's walls
along the plan's (each wall votes with its points and its length, each plan wall with its length), the smallest such turn when a
room's right angles offer several; a turn past `TURN_DEG` from the facing the owner set is not made (a camera that
far round was moved, which spatial/cloud_scene.py follows by its pictures). *Slide:* each cloud wall that has one
clear counterpart on the plan (parallel, beside it, within `CLEAR_M`, and no other plan wall it could be) asks to
be moved onto it; the slide that suits them best, nearer walls counting more (a cloud's far depth is its least sure),
is made when it is within `SLIDE_M`. A cloud is the model's own estimate of size, so two facing walls can ask for
different slides: they share it, and neither is stretched to fit.

**Fit to the room** (`fit`). The plan's rooms have walls of known length, which makes the plan the measurement of
a cloud's size. Each wall of the squared cloud is matched to the plan wall its camera sees there: the first plan wall
on the sight lines from the camera's spot to the cloud wall (so never one behind another), when it runs the same way
(`FACING_DEG`) and the cloud wall is within `SIZE` of its distance. Of several cloud planes facing one plan wall, the
farthest is the wall (in pieces, where it bends) and the nearer ones are fronts of what stands against it. The cloud
is resized about its camera and slid, by least squares, so its matched walls land on the plan's; one wall gives the
size alone, walls on both sides the slide too. A fit past `SIZE` or `FIT_SLIDE_M`, or with no wall matched, is not
made, and says why. Where the cloud has a floor and its camera a measured height, it is sized upwards by the height,
across by the walls: a floor stays flat and a wall upright under that, and near and far each keep their own measure.

**Flat walls and floor** (`flatten`). In a fitted cloud, the points of its floor are drawn on the floor and the points
of each matched wall on that wall of the plan, where they were along it and up it. A wall drawn so is the plan's wall
with what the camera saw painted on it; where the camera saw nothing there is a hole. Everything else, the fronts of
furniture included, stays the points it is.

Registering overlapping clouds to each other from their points and colours (coloured ICP) was tried on the Linux
machine and left out: one pair of six overlaps at all, and there it moved a cloud 0.3 to 0.4 m and tipped it 4 to 5
degrees off level, farther from what the two pictures measured (0.26 m to 0.54 m).

Levelling is part of making a cloud (spatial/camera_clouds.py `make`, and `relevel` for one made before this). What
comes of squaring is an `Align`: a step of the camera on the floor (to its right, ahead, and a turn). It is kept
beside the cloud (`keep`, `kept`), and holds only for the cloud, the spot, and the plan it was made from:
spatial/cloud_scene.py places a cloud without it when any of those is another. Nothing here writes a spot or the plan.

**What neither move takes out** (`Square.bend_deg`). A cloud is one picture's depth estimate, and its walls aren't
quite square to each other or to its floor. Measured on six cameras on the Linux machine: two clouds that had no
floor and were stood as if their cameras were upright were 11.7 and 12.5 degrees off level, their walls leaning 7 to
15 degrees; by their walls they come upright, the walls within 0 to 3 degrees. After that, walls of one cloud still
lean up to 5 degrees and run 3 to 8 degrees off each other about the vertical, and a wall's points lie 4 to 9 cm
(RMS) off its own flat plane. Squaring lays the walls with the most points along the plan and leaves the rest out
by that much; the view says so. Four slides of five asked for more than `SLIDE_M`, since a cloud's far walls are off
by its size (a tenth at 6 m), which a slide can't mend.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from liberational import json_state
from liberational.memory import places

TOL_M = 0.03  # a point this near a plane is on it
TRIES = 300
SAMPLE = 20000
MIN_SHARE = 0.03  # of the sampled points a plane must hold
WALL_SHARE = 0.01  # and a standing plane (within `WALL_DEG` of vertical): a wall at a picture's edge holds few points
TALL_M = 1.0  # and it must stand this tall: a console's front is low, and a few centimetres off the wall behind it
MOST = 8  # planes looked for in one cloud
MOST_TRIES = 20  # planes taken out of one cloud, kept or not, before the search stops
MIN_SPREAD_M = 0.2  # a plane's points must spread at least this much (standard deviation) along both its axes
WALL_DEG = 35.0  # a plane this near vertical, in a cloud not yet levelled, is taken for a wall: cameras tilt to 28
UPRIGHT_DEG = 6.0  # and in a levelled one
TWO_WALLS = 0.15  # a second wall direction counts when it holds this share of what the first holds
TURN_DEG = 15.0  # the most a cloud is turned from the facing the owner set
SLIDE_M = 0.5  # and the most it is slid
CLEAR_M = 1.0  # a plan wall this near a cloud wall may be its counterpart; two of them and it has none
PARALLEL_DEG = 3.0  # walls within this are parallel
NEAR_M = 10.0  # the plan's walls this near the camera are the ones its cloud is squared to
VOTE_DEG = 1.5  # how sharply a wall votes for a turn
ALIGN = "align.json"
MADE = 2  # 2: walls that hold few of a cloud's points are looked for (`WALL_SHARE`)


@dataclass(frozen=True)
class Plane:
    """One flat surface of a cloud: the points p with `normal` @ p = `offset`."""

    normal: np.ndarray
    offset: float
    points: int  # of the sample that lie on it
    share: float  # of the sample
    centre: np.ndarray
    spread: tuple[float, float]  # the standard deviation of its points along its longer and shorter axes (m)
    rms: float  # of its points' distances from it (m)
    on: np.ndarray = field(repr=False, compare=False)  # its points


def _fit(points: np.ndarray) -> tuple[np.ndarray, float, tuple[float, float]]:
    centre = points.mean(axis=0)
    _, s, vt = np.linalg.svd(points - centre, full_matrices=False)
    spread = s / math.sqrt(max(len(points), 1))
    return vt[2], float(vt[2] @ centre), (float(spread[0]), float(spread[1]))


def planes(points: np.ndarray, seed: int = 0) -> list[Plane]:
    """The big flat surfaces of a cloud (N x 3), the largest first (the module docstring). The same cloud gives the
    same planes: the search is seeded."""
    rng = np.random.default_rng(seed)
    if len(points) < 500:
        return []
    sample = points[rng.choice(len(points), min(len(points), SAMPLE), replace=False)].astype(np.float64)
    total, left, out = len(sample), sample, []
    standing = math.sin(math.radians(WALL_DEG))
    for _ in range(MOST_TRIES):
        if len(left) < WALL_SHARE * total:
            break
        quick = left[rng.choice(len(left), min(len(left), 4000), replace=False)]
        picks = quick[rng.integers(0, len(quick), (TRIES, 3))]
        normals = np.cross(picks[:, 1] - picks[:, 0], picks[:, 2] - picks[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        good = lengths > 1e-9
        normals = normals[good] / lengths[good, None]
        offsets = (normals * picks[good, 0]).sum(axis=1)
        if not len(normals):
            break
        counts = (np.abs(quick @ normals.T - offsets) < TOL_M).sum(axis=0)
        best = int(np.argmax(counts))
        normal, offset = normals[best], float(offsets[best])
        spread = (0.0, 0.0)
        for _ in range(3):  # three points' plane is only as good as those three: refit to the points on it
            on = np.abs(left @ normal - offset) < TOL_M
            if on.sum() < 3:
                break
            normal, offset, spread = _fit(left[on])
        on = np.abs(left @ normal - offset) < TOL_M
        if on.sum() < WALL_SHARE * total:
            break
        held = left[on]
        # Under the bar for a plane, or with `MOST` found, only a tall standing plane still counts.
        only_walls = on.sum() < MIN_SHARE * total or len(out) >= MOST
        wall = abs(float(normal[2])) < standing and float(np.ptp(held[:, 2])) >= TALL_M
        if spread[1] >= MIN_SPREAD_M and (not only_walls or wall):
            rms = float(np.sqrt(np.mean((held @ normal - offset) ** 2)))
            out.append(
                Plane(normal, offset, int(on.sum()), float(on.sum()) / total, held.mean(axis=0), spread, rms, held)
            )
        left = left[np.abs(left @ normal - offset) >= 2 * TOL_M]
    return out


def walls(found: Sequence[Plane], within_deg: float, up: np.ndarray | None = None) -> list[Plane]:
    """Those of `found` within `within_deg` of vertical, each with its normal pointing at the origin's side."""
    up = np.array([0.0, 0.0, 1.0]) if up is None else up
    out = []
    for plane in found:
        if abs(float(plane.normal @ up)) < math.sin(math.radians(within_deg)):
            flip = plane.offset > 0  # the origin is on the side the normal points away from
            out.append(plane if not flip else Plane(-plane.normal, -plane.offset, plane.points, plane.share,
                                                    plane.centre, plane.spread, plane.rms, plane.on))  # fmt: skip
    return out


def up_from_walls(found: Sequence[Plane], guess: np.ndarray) -> np.ndarray | None:
    """Which way is up by the walls among `found`, near `guess` (the module docstring's Level). None with no wall, or
    when the walls put up farther than `WALL_DEG` from the guess."""
    stand = walls(found, WALL_DEG, guess)
    if not stand:
        return None
    normals = np.stack([p.normal for p in stand])
    weights = np.array([float(p.points) for p in stand])
    held = (normals * weights[:, None]).T @ normals  # what the walls' normals span
    values, vectors = np.linalg.eigh(held)  # ascending
    if values[1] >= TWO_WALLS * values[2]:  # two directions of wall: up is what runs along both
        up = vectors[:, 0]
    else:  # one: take the lean across it out of the guess
        across = vectors[:, 2]
        up = guess - float(guess @ across) * across
    up = up / np.linalg.norm(up)
    if up @ guess < 0:
        up = -up
    return up if up @ guess >= math.cos(math.radians(WALL_DEG)) else None


def tilt_roll(pose: np.ndarray) -> tuple[float, float]:
    """A camera's tilt (degrees it looks down) and roll (degrees its right side drops) from its pose in a level
    frame (camera to cloud; the camera's axes are x right, y down, z ahead)."""
    turn = pose[:3, :3]
    return (
        float(np.degrees(np.arcsin(np.clip(-turn[2, 2], -1.0, 1.0)))),
        float(np.degrees(np.arcsin(np.clip(-turn[2, 0], -1.0, 1.0)))),
    )


def stand(points: np.ndarray, pose: np.ndarray, found: Sequence[Plane] | None = None) -> np.ndarray | None:
    """The move (4 x 4) that stands a cloud with no floor upright by its walls (the module docstring's Level), or
    None when it has no wall to go by. `points` are in the cloud's own frame (z up as far as it knew, its camera at
    `pose`); `found` are its planes, looked for here when not given. The camera stays on the z axis, looking along
    +y, and the cloud's lowest points stand in for the floor."""
    guess = np.array([0.0, 0.0, 1.0])
    up = up_from_walls(planes(points) if found is None else found, guess)
    if up is None:
        return None
    ahead = pose[:3, 2] - float(pose[:3, 2] @ up) * up  # the camera's forward, laid on the floor
    if np.linalg.norm(ahead) < 1e-6:
        ahead = np.array([1.0, 0.0, 0.0]) - up[0] * up
    y = ahead / np.linalg.norm(ahead)
    turn = np.stack([np.cross(y, up), y, up])
    eye = pose[:3, 3]
    low = float(np.percentile((points - eye) @ up, 2)) if len(points) else 0.0
    matrix = np.eye(4)
    matrix[:3, :3] = turn
    matrix[:3, 3] = -turn @ eye + np.array([0.0, 0.0, -low])
    return matrix


def apply(matrix: Sequence[float] | np.ndarray, points: np.ndarray) -> np.ndarray:
    move = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    return points @ move[:3, :3].T + move[:3, 3]


# ---- squaring to the plan

#: A wall on the floor of the scene: its two ends.
Edge = tuple[tuple[float, float], tuple[float, float]]


def edges_of(rooms: Sequence[Sequence[tuple[float, float]]]) -> list[Edge]:
    """The walls of a plan's rooms (their outlines, in the scene's frame)."""
    out: list[Edge] = []
    for ring in rooms:
        corners = [(float(x), float(y)) for x, y in ring]
        for a, b in zip(corners, corners[1:] + corners[:1], strict=True):
            if math.hypot(b[0] - a[0], b[1] - a[1]) > 0.2:
                out.append((a, b))
    return out


@dataclass(frozen=True)
class Square:
    """How a cloud was squared to the plan."""

    #: "squared", "no_walls" (no wall in the cloud), "no_plan" (no plan wall near the camera), or "beyond" (its walls
    #: line up only past the bound: left alone)
    state: str
    turn_deg: float = 0.0  # clockwise seen from above; for "beyond", the turn it would have taken
    right_m: float = 0.0  # the slide, in the camera's own axes as its spot has them
    ahead_m: float = 0.0
    walls: int = 0  # the cloud's walls that voted for the turn
    slid_on: int = 0  # and those that had a clear counterpart on the plan
    slide: str = ""  # why it wasn't slid, when it wasn't: "no_counterpart" or "beyond"
    off_deg: float = 0.0  # how far its walls were from the plan's before, on average (degrees)
    #: How far the cloud's walls are out with each other (degrees): the most any of them still leans, or runs off the
    #: plan's walls, once the cloud is level and squared. No rigid move takes this out: it is the depth estimate
    #: bending the room.
    bend_deg: float = 0.0

    @property
    def rel(self) -> tuple[float, float, float]:
        """The squaring as a step of the camera: to its right, ahead, and a turn (radians)."""
        return (self.right_m, self.ahead_m, math.radians(self.turn_deg)) if self.state == "squared" else (0.0, 0.0, 0.0)


def _fold(angle: np.ndarray | float, period: float) -> Any:
    """An angle difference folded into (-period/2, period/2]."""
    return (np.asarray(angle) + period / 2) % period - period / 2


@dataclass(frozen=True)
class Line:
    """A cloud's wall on the floor, in the frame it is given in: normal . p = offset, between `lo` and `hi` along it."""

    normal: tuple[float, float]  # pointing away from the camera
    offset: float
    lo: float  # along (-normal_y, normal_x)
    hi: float
    weight: float
    distance: float  # from the camera
    lean: float = 0.0  # degrees it is off vertical


def lines(found: Sequence[Plane], eye: np.ndarray | None = None) -> list[Line]:
    """The walls among a levelled cloud's planes, as lines on its floor."""
    eye = np.zeros(2) if eye is None else eye
    out = []
    for plane in walls(found, UPRIGHT_DEG):
        flat = plane.normal[:2] / np.linalg.norm(plane.normal[:2])
        if flat @ (plane.centre[:2] - eye) < 0:
            flat = -flat
        along = np.array([-flat[1], flat[0]])
        reach = plane.on[:, :2] @ along
        lo, hi = np.percentile(reach, [3, 97])
        offset = float(np.median(plane.on[:, :2] @ flat))
        if hi - lo < 0.5 or np.ptp(plane.on[:, 2]) < 0.5:
            continue  # too short or too low to be a wall
        distance = abs(offset - float(flat @ eye))
        lean = float(np.degrees(np.arcsin(min(abs(float(plane.normal[2])), 1.0))))
        out.append(
            Line((float(flat[0]), float(flat[1])), offset, float(lo), float(hi), float(plane.points), distance, lean)
        )
    return out


def _moved(line: Line, x: float, y: float, bearing: float) -> Line:
    """`line` (in a cloud's level frame) where a camera at x, y facing `bearing` puts it in the scene."""
    s, c = math.sin(bearing), math.cos(bearing)
    nx, ny = line.normal
    normal = (c * nx + s * ny, -s * nx + c * ny)
    shift = normal[0] * x + normal[1] * y
    along = (-normal[1], normal[0])
    slip = along[0] * x + along[1] * y
    return Line(normal, line.offset + shift, line.lo + slip, line.hi + slip, line.weight, line.distance, line.lean)


def square(found: Sequence[Line], x: float, y: float, bearing: float, plan: Sequence[Edge]) -> Square:
    """Square a levelled cloud's walls `found` (in its own frame, the camera at the origin) to the plan's walls
    `plan`, for a camera the owner put at x, y facing `bearing` in the scene (the module docstring's Square)."""
    if not found:
        return Square("no_walls")
    near = []
    for a, b in plan:
        edge = np.array([b[0] - a[0], b[1] - a[1]])
        t = float(np.clip(((x - a[0]) * edge[0] + (y - a[1]) * edge[1]) / float(edge @ edge), 0.0, 1.0))
        if math.hypot(a[0] + t * edge[0] - x, a[1] + t * edge[1] - y) <= NEAR_M:
            near.append((a, b))
    if not near:
        return Square("no_plan", walls=len(found))
    plan_angle = np.array([math.atan2(b[0] - a[0], b[1] - a[1]) for a, b in near])  # a wall's direction as a bearing
    plan_length = np.array([math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in near])
    # A wall's direction in the scene is its direction in the cloud plus the camera's bearing; lines have no sense.
    wall_angle = np.array([math.atan2(-line.normal[1], line.normal[0]) for line in found]) + bearing
    # A wall's say grows with its points and with its length: a cupboard's front can hold more points than the wall
    # behind it, and a metre of it says less about which way the room runs than four metres of wall.
    weight = np.array([line.weight * (line.hi - line.lo) for line in found])
    turns = np.radians(np.arange(-45.0, 45.0 + 1e-9, 0.25))
    sigma = math.radians(VOTE_DEG)
    apart = _fold(wall_angle[None, :, None] + turns[:, None, None] - plan_angle[None, None, :], math.pi)
    votes = (np.exp(-(apart**2) / (2 * sigma**2)) * plan_length[None, None, :]).sum(axis=2) @ weight
    peaks = [
        i
        for i in range(len(turns))
        if votes[i] >= votes[max(i - 1, 0)] and votes[i] >= votes[min(i + 1, len(turns) - 1)]
    ]
    strong = [i for i in peaks if votes[i] >= 0.6 * votes.max()]
    pick = min(strong, key=lambda i: abs(turns[i]))
    turn = float(turns[pick])
    # refine on the walls that vote for it: each one's own angle to its nearest plan wall
    apart = _fold(wall_angle[:, None] + turn - plan_angle[None, :], math.pi)
    nearest = np.abs(apart).argmin(axis=1)
    off = apart[np.arange(len(found)), nearest]
    voting = np.abs(off) < 3 * sigma
    if not voting.any():
        return Square("no_plan", walls=len(found))
    turn -= float(np.average(off[voting], weights=weight[voting]))
    before = _fold(wall_angle[:, None] - plan_angle[None, :], math.pi)
    off_before = float(np.degrees(np.average(np.abs(before).min(axis=1)[voting], weights=weight[voting])))
    after = np.abs(_fold(wall_angle[:, None] + turn - plan_angle[None, :], math.pi)).min(axis=1)
    bend = round(max(float(np.degrees(after.max())), max(line.lean for line in found)), 1)
    if abs(math.degrees(turn)) > TURN_DEG:
        return Square("beyond", round(math.degrees(turn), 2), walls=int(voting.sum()), off_deg=round(off_before, 2),
                      bend_deg=bend)  # fmt: skip

    # The slide: every cloud wall with one clear counterpart on the plan asks to be moved onto it.
    asks, rows, heavy = [], [], []
    for line in found:
        placed = _moved(line, x, y, bearing + turn)
        normal = np.array(placed.normal)
        along = np.array([-normal[1], normal[0]])
        gaps = []
        for a, b in near:
            ends = np.array([a, b])
            direction = ends[1] - ends[0]
            if abs(
                float(_fold(math.atan2(direction[1], direction[0]) - math.atan2(along[1], along[0]), math.pi))
            ) > math.radians(PARALLEL_DEG):
                continue
            reach = sorted(ends @ along)
            shared = min(reach[1], placed.hi) - max(reach[0], placed.lo)
            if shared < 0.5 * min(reach[1] - reach[0], placed.hi - placed.lo):
                continue  # not beside it
            gap = float(ends.mean(axis=0) @ normal) - placed.offset
            if abs(gap) <= CLEAR_M:
                gaps.append(gap)
        if gaps and max(gaps) - min(gaps) < 0.05:  # one plan wall (two rooms can share its line)
            asks.append(float(np.mean(gaps)))
            rows.append(normal)
            heavy.append(line.weight / max(line.distance, 0.5) ** 2)
    right = ahead = 0.0
    why = "no_counterpart"
    if asks:
        root = np.sqrt(np.array(heavy))[:, None]
        slide, *_ = np.linalg.lstsq(np.array(rows) * root, np.array(asks) * root[:, 0], rcond=1e-3)
        if float(np.linalg.norm(slide)) <= SLIDE_M:
            s, c = math.sin(bearing), math.cos(bearing)  # a step is taken in the camera's axes before it turns
            right, ahead, why = float(slide[0] * c - slide[1] * s), float(slide[0] * s + slide[1] * c), ""
        else:
            why = "beyond"
    return Square("squared", round(math.degrees(turn), 2), round(right, 3), round(ahead, 3), int(voting.sum()),
                  len(asks) if not why else 0, why, round(off_before, 2), bend)  # fmt: skip


# ---- fitting to the room

SIZE = (0.8, 1.25)  # how far a cloud may be resized: the model's metres were 0.96 to 1.06 of the plan's on live walls
FIT_SLIDE_M = 0.5  # and how far it may be slid, once resized
FACING_DEG = 10.0  # a cloud wall within this of a plan wall's direction may be that wall (walls bend up to 8)
RAYS = 5  # sight lines from the camera to a cloud wall, to find the plan wall it sees there
OFF_M = 0.3  # a matched wall left this far from its plan wall by the fit is not that wall
SAME_WALL_M = (
    0.3  # cloud planes this near the farthest one matched to a plan wall are that wall, bent; nearer ones are furniture
)
HOLD = 0.05  # how hard the slide is held to nothing, as a share of the walls' weight
FRONT = 0.3  # how much a plane standing in the room, short of the plan's walls, counts against `Fit.share`


@dataclass(frozen=True)
class Matched:
    """A wall of a cloud and the plan wall it was matched to."""

    a: tuple[float, float]  # the plan wall's ends, in the scene
    b: tuple[float, float]
    normal: tuple[float, float]  # the cloud wall, as a line in the cloud's own frame: normal . p = offset
    offset: float
    lo: float  # its reach along (-normal_y, normal_x)
    hi: float
    off_deg: float  # how far the cloud wall runs off the plan wall, squared
    before_m: float  # how far it stood from the plan wall before the fit (positive: beyond it)
    after_m: float  # and after


@dataclass(frozen=True)
class Fit:
    """How a squared cloud was fitted to the rooms of the plan."""

    #: "fitted"; "no_walls" (no wall in the cloud); "no_match" (no wall of it could be matched to a plan wall);
    #: "beyond" (the fit asks for a resize or a slide past the bounds: left alone, and `scale`, `dx`, `dy` say what it
    #: would have taken)
    state: str
    scale: float = 1.0  # the cloud's size across the floor, against the model's own metres
    up: float = 1.0  # and its size upwards (the camera's measured height over the cloud's, when known; else `scale`)
    dx: float = 0.0  # the slide of the camera from its spot, in the scene (m)
    dy: float = 0.0
    turn_deg: float = 0.0  # the squaring turn it was fitted with
    walls: tuple[Matched, ...] = ()
    seen: int = 0  # the walls in the cloud
    #: How well the plan explains the cloud's standing planes from where the camera is taken to be (`locate`): of
    #: their points, the share on walls matched to the plan and left on them by the fit. A plane that is neither
    #: (beyond a wall, or by a wall and running another way) counts in full against it; one standing in the room,
    #: short of any wall (furniture), counts `FRONT` as much: a room has things in it, but a view of nothing else
    #: says little.
    share: float = 0.0


def _spans(plan: Sequence[Edge]) -> tuple[np.ndarray, np.ndarray]:
    """The plan's walls as arrays: where each starts, and its run to its other end."""
    ends = np.asarray(plan, dtype=np.float64).reshape(-1, 2, 2)
    return ends[:, 0], ends[:, 1] - ends[:, 0]


def _first_hit(
    origin: np.ndarray, toward: np.ndarray, starts: np.ndarray, runs: np.ndarray
) -> tuple[int, float] | None:
    """The plan wall (`_spans`) a sight line from `origin` through `toward` meets first, and how far along the line
    it meets it (1.0 is at `toward`), or None."""
    ray = toward - origin
    cross = ray[0] * runs[:, 1] - ray[1] * runs[:, 0]
    gap = starts - origin
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (gap[:, 0] * runs[:, 1] - gap[:, 1] * runs[:, 0]) / cross  # along the ray
        u = (gap[:, 0] * ray[1] - gap[:, 1] * ray[0]) / cross  # along the wall
    met = (np.abs(cross) >= 1e-12) & (t > 1e-6) & (u >= -1e-9) & (u <= 1 + 1e-9)
    if not met.any():
        return None
    first = int(np.argmin(np.where(met, t, np.inf)))
    return first, float(t[first])


def fit(
    found: Sequence[Line],
    x: float,
    y: float,
    bearing: float,
    plan: Sequence[Edge],
    turn_deg: float = 0.0,
    height: tuple[float, float] | None = None,
) -> Fit:
    """Fit a levelled cloud to the plan's rooms: `found` are its walls (its own frame, the camera at the origin),
    for a camera the owner put at x, y facing `bearing`, squared by `turn_deg`. Each cloud wall is matched to the
    plan wall the camera sees where that wall is: the first plan wall on the sight lines from the camera to it, when
    that wall runs the same way. The plan's walls are the measurement: with the cloud resized by s about its camera
    and slid by t, a wall d metres from the camera along its plan wall's normal m lands s d + m . t from the
    camera's spot, and should land where the plan wall is. s and t are the least-squares answer, t held lightly to
    nothing, so one wall gives the size alone and walls on both sides give size and slide. `height` is the camera's
    measured height and its height in the cloud: the cloud is sized upwards by their ratio, so near things, sized
    by the floor under the camera, and far walls each get their own measure."""
    if not found:
        return Fit("no_walls")
    facing = bearing + math.radians(turn_deg)
    eye = np.array([x, y])
    starts, runs = _spans(plan)
    rows, hit = [], []
    fronts = 0.0  # the weight of the planes that stand in the room, short of any wall of the plan
    for line in found:
        placed = _moved(line, x, y, facing)
        normal = np.array(placed.normal)
        along = np.array([-normal[1], normal[0]])
        marks = np.linspace(placed.lo, placed.hi, RAYS + 2)[1:-1]
        targets = [normal * placed.offset + along * mark for mark in marks]
        hits = [_first_hit(eye, target, starts, runs) for target in targets]
        # How many of its sight lines reach the plane well before any wall: a front of furniture, whichever way it
        # runs, is in the room, and is no wall out of place.
        short = sum(
            got is not None and (got[1] - 1.0) * float(np.linalg.norm(target - eye)) > SAME_WALL_M
            for target, got in zip(targets, hits, strict=True)
        )
        row = None
        seen = [got[0] for got in hits if got is not None]
        which = max(set(seen), key=seen.count) if seen else -1
        if seen and seen.count(which) * 2 > RAYS:  # else the sight lines don't agree on a wall: an opening, a corner
            a, b = plan[which]
            edge = np.array(b) - np.array(a)
            wall_normal = np.array([edge[1], -edge[0]]) / float(np.linalg.norm(edge))
            if wall_normal @ (np.array(a) - eye) < 0:
                wall_normal = -wall_normal
            off = float(np.degrees(np.arccos(np.clip(wall_normal @ normal, -1.0, 1.0))))
            # the cloud wall's distance from the camera along the plan wall's normal, at the middle of its reach
            middle = normal * placed.offset + along * float(np.mean(marks)) - eye
            d, g = float(middle @ wall_normal), float((np.array(a) - eye) @ wall_normal)
            if off <= FACING_DEG and d > 0.3 and SIZE[0] <= g / d <= SIZE[1]:
                row = (line, (a, b), wall_normal, d, g, off)
        if row is None:
            fronts += line.weight if short * 2 > RAYS else 0.0
            continue
        rows.append(row)
        hit.append(which)
    # Several planes of a cloud can face one plan wall: the wall itself, in pieces where it bends, and the fronts of
    # what stands against it. The wall is the farthest; a cupboard's front is not laid on it.
    farthest = {which: max(row[3] for row, w in zip(rows, hit, strict=True) if w == which) for which in set(hit)}
    fronts += sum(row[0].weight for row, which in zip(rows, hit, strict=True) if row[3] < farthest[which] - SAME_WALL_M)
    rows = [row for row, which in zip(rows, hit, strict=True) if row[3] >= farthest[which] - SAME_WALL_M]
    if not rows:
        return Fit("no_match", turn_deg=turn_deg, seen=len(found))
    scale, slide = 1.0, np.zeros(2)
    while rows:
        weight = np.array([row[0].weight for row in rows])
        root = np.sqrt(weight)
        hold = math.sqrt(HOLD * float(weight.sum()))
        lhs = np.array([[row[3], *row[2]] for row in rows]) * root[:, None]
        rhs = np.array([row[4] for row in rows]) * root
        lhs = np.vstack([lhs, [[0.0, hold, 0.0], [0.0, 0.0, hold]]])
        answer, *_ = np.linalg.lstsq(lhs, np.append(rhs, [0.0, 0.0]), rcond=None)
        scale, slide = float(answer[0]), answer[1:]
        left = np.array([scale * row[3] + float(row[2] @ slide) - row[4] for row in rows])
        worst = int(np.abs(left).argmax())
        if abs(left[worst]) <= OFF_M or len(rows) == 1:
            break
        rows.pop(worst)  # not the wall it was taken for
    matched = tuple(
        Matched(a, b, line.normal, line.offset, line.lo, line.hi, round(off, 2), round(d - g, 3), round(left_m, 3))
        for (line, (a, b), m, d, g, off), left_m in zip(
            rows, (scale * row[3] + float(row[2] @ slide) - row[4] for row in rows), strict=True
        )
    )
    up = scale
    if height is not None and height[1] > 0 and 1 / SIZE[1] <= height[0] / height[1] <= SIZE[1]:
        up = height[0] / height[1]
    state = "fitted" if SIZE[0] <= scale <= SIZE[1] and float(np.linalg.norm(slide)) <= FIT_SLIDE_M else "beyond"
    on = sum(row[0].weight for row in rows)
    off_the_plan = sum(line.weight for line in found) - on - fronts
    share = on / max(on + off_the_plan + FRONT * fronts, 1e-9)
    return Fit(state, round(scale, 4), round(up, 4), round(float(slide[0]), 3), round(float(slide[1]), 3), turn_deg,
               matched, len(found), round(float(share), 3))  # fmt: skip


FLOOR_BAND_M = 0.05  # a point this near the cloud's floor is the floor's
WALL_BAND_M = 0.12  # and one this near a matched wall is that wall's: walls bow up to 7 cm
POINT, FLOOR, WALL = 0, 1, 2  # what a drawn point is


def flatten(own: np.ndarray, move: np.ndarray, found: Fit) -> tuple[np.ndarray, np.ndarray]:
    """A fitted cloud's points as they are drawn flat: `own` (its own frame) moved into the scene by `move` (4 x 4),
    with the points of its floor laid on the floor (z = 0) and the points of each matched wall laid on that wall of
    the plan, each where it was along the wall and up it. Returns the scene points, and for each what it is (`POINT`,
    `FLOOR`, `WALL`). Nothing is added: where the camera saw no wall there is none. Everything else stays as it is."""
    scene = own @ move[:3, :3].T + move[:3, 3]
    kind = np.full(len(own), POINT, dtype=np.uint8)
    if found.state != "fitted":
        return scene, kind
    floor = np.abs(own[:, 2]) < FLOOR_BAND_M
    kind[floor] = FLOOR
    scene[floor, 2] = 0.0
    for wall in found.walls:
        normal = np.array(wall.normal)
        along = np.array([-normal[1], normal[0]])
        reach = own[:, :2] @ along
        past = 0.1 + 0.06 * (wall.hi - wall.lo)  # a wall's reach is measured short of its last few points
        on = (
            (np.abs(own[:, :2] @ normal - wall.offset) < WALL_BAND_M)
            & (reach > wall.lo - past)
            & (reach < wall.hi + past)
            & (kind == POINT)
        )
        a, edge = np.array(wall.a), np.array(wall.b) - np.array(wall.a)
        length = float(np.linalg.norm(edge))
        run = edge / length
        where = np.clip((scene[on, :2] - a) @ run, 0.0, length)  # along the plan wall, and not past its ends
        scene[on, :2] = a + where[:, None] * run
        kind[on] = WALL
    return scene, kind


# ---- finding where a camera stands

MARGIN_M = 0.3  # a point this far outside every room of the plan still belongs to the room
MIN_SCORE = 0.5  # of a cloud's walls (by their points) the plan must explain, less the share of points outside it
RIVAL = 0.15  # another answer scoring within this of the best is as good: the view fits two places
SAME = (0.75, 20.0)  # metres and degrees: two answers this near each other are one
CROSS_DEG = 30.0  # walls this far from parallel pin a camera down in both directions
TOP = 5  # the cloud's walls, by their points, that candidates are made from
GRID = (0.2, 2.0)  # metres and degrees: candidates nearer each other than this are tried once
SAMPLE_XY = 1500  # points of the cloud checked against the plan's rooms


def outside(xy: np.ndarray, rooms: Sequence[Sequence[tuple[float, float]]], margin: float = MARGIN_M) -> np.ndarray:
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


@dataclass(frozen=True)
class Located:
    """Where a cloud's own walls put its camera on the plan, or why they can't say."""

    #: "placed"; "no_walls" (no wall in the cloud); "no_plan" (the plan has no walls); "no_match" (its walls fit the
    #: plan nowhere well enough); "ambiguous" (they fit two places, or two facings, about as well)
    state: str
    x: float = 0.0
    y: float = 0.0
    bearing: float = 0.0
    #: "turned" (where it stood, facing another way) or "walls" (somewhere else, by its walls alone)
    by: str = ""
    score: float = 0.0  # of the answer: `Fit.share` less the share of the cloud's points outside the plan's rooms
    rival: float = 0.0  # and of the best answer somewhere else
    walls: int = 0  # the cloud's walls matched to the plan's
    scale: float = 1.0  # the cloud's size against the plan's, by the fit that placed it


@dataclass(frozen=True)
class _Try:
    score: float
    x: float
    y: float
    bearing: float
    walls: int
    crossed: bool  # its matched walls run in two directions
    scale: float = 1.0

    def near(self, other: _Try) -> bool:
        turn = abs((self.bearing - other.bearing + math.pi) % (2 * math.pi) - math.pi)
        return math.hypot(self.x - other.x, self.y - other.y) <= SAME[0] and math.degrees(turn) <= SAME[1]


def _try(
    found: Sequence[Line],
    x: float,
    y: float,
    bearing: float,
    plan: Sequence[Edge],
    rooms: Sequence[Sequence[tuple[float, float]]],
    sample: np.ndarray,
) -> _Try | None:
    """How well the plan explains a cloud whose camera is taken to stand at x, y facing `bearing`: squared and
    fitted from there as any cloud is, then scored. None when it can't be fitted from there."""
    squared = square(found, x, y, bearing, plan)
    turn = squared.turn_deg if squared.state == "squared" else 0.0
    fitted = fit(found, x, y, bearing, plan, turn)
    if fitted.state != "fitted" or fitted.share < MIN_SCORE:
        return None
    facing = bearing + math.radians(turn)
    at = (x + fitted.dx, y + fitted.dy)
    s, c = math.sin(facing), math.cos(facing)
    own = sample * fitted.scale
    placed = np.c_[own[:, 0] * c + own[:, 1] * s + at[0], -own[:, 0] * s + own[:, 1] * c + at[1]]
    beyond = float(outside(placed, rooms).mean()) if len(placed) else 0.0
    normals = np.array([wall.normal for wall in fitted.walls])
    cross = np.abs(normals[:, None, 0] * normals[None, :, 1] - normals[:, None, 1] * normals[None, :, 0])
    crossed = bool((cross > math.sin(math.radians(CROSS_DEG))).any())
    return _Try(fitted.share - beyond, at[0], at[1], facing, len(fitted.walls), crossed, fitted.scale)


def _bearings(found: Sequence[Line], plan: Sequence[Edge]) -> list[float]:
    """The facings that lay one of the cloud's walls along one of the plan's: every way round a wall can run."""
    runs = {round(math.degrees(math.atan2(b[0] - a[0], b[1] - a[1])) % 180.0 / GRID[1]) for a, b in plan}
    out: dict[int, float] = {}
    for line in found:
        wall = math.degrees(math.atan2(-line.normal[1], line.normal[0]))
        for run in runs:
            for flip in (0.0, 180.0):
                bearing = (run * GRID[1] - wall + flip) % 360.0
                out.setdefault(round(bearing / GRID[1]) % round(360.0 / GRID[1]), math.radians(bearing))
    return list(out.values())


def _corners(
    found: Sequence[Line], bearing: float, plan: Sequence[Edge], rooms: Sequence[Sequence[tuple[float, float]]]
) -> np.ndarray:
    """Where a camera facing `bearing` could stand (N x 2): every spot inside a room that puts two of the cloud's
    walls that cross, each at its own distance, on two walls of the plan that run their way and reach that far."""
    starts, runs = _spans(plan)
    lengths = np.linalg.norm(runs, axis=1)
    along = runs / lengths[:, None]
    placed = [_moved(line, 0.0, 0.0, bearing) for line in found]
    out = []
    for i, first in enumerate(placed):
        for second in placed[i + 1 :]:
            n1, n2 = np.array(first.normal), np.array(second.normal)
            if abs(n1[0] * n2[1] - n1[1] * n2[0]) <= math.sin(math.radians(CROSS_DEG)):
                continue
            inverse = np.linalg.inv(np.stack([n1, n2]))
            sides = []
            for line, normal in ((first, n1), (second, n2)):
                runs_its_way = np.flatnonzero(np.abs(along @ normal) < math.sin(math.radians(FACING_DEG)))
                sides.append((runs_its_way, starts[runs_its_way] @ normal - line.offset))
            (p, want1), (q, want2) = sides
            if not len(p) or not len(q):
                continue
            eyes = np.stack(np.meshgrid(want1, want2, indexing="ij"), axis=-1).reshape(-1, 2) @ inverse.T
            which = np.stack(np.meshgrid(p, q, indexing="ij"), axis=-1).reshape(-1, 2)
            reaches = np.ones(len(eyes), dtype=bool)
            for line, normal, wall in ((first, n1, which[:, 0]), (second, n2, which[:, 1])):
                middle = eyes + normal * line.offset + np.array([-normal[1], normal[0]]) * (line.lo + line.hi) / 2
                run = ((middle - starts[wall]) * along[wall]).sum(axis=1)
                reaches &= (run > -0.5) & (run < lengths[wall] + 0.5)
            eyes = eyes[reaches]
            out.append(eyes[~outside(eyes, rooms, 0.0)])
    return np.concatenate(out) if out else np.zeros((0, 2))


def _pick(tries: Sequence[_Try], need_crossed: bool) -> tuple[_Try | None, float]:
    """The best of `tries` that counts, and the score of the best one that isn't the same answer."""
    counted = [t for t in tries if t.score >= MIN_SCORE and (t.crossed or not need_crossed)]
    if not counted:
        return None, 0.0
    best = max(counted, key=lambda t: t.score)
    return best, max((t.score for t in counted if not t.near(best)), default=0.0)


def locate(
    found: Sequence[Line],
    plan: Sequence[Edge],
    rooms: Sequence[Sequence[tuple[float, float]]],
    points: np.ndarray,
    near: tuple[float, float] | None = None,
    anywhere: bool = True,
) -> Located:
    """Where on the plan a levelled cloud's camera stands, by the cloud alone: `found` are its walls (its own frame,
    the camera at the origin), `points` the cloud (N x 3), `plan` and `rooms` the plan's walls and rooms. There is no
    limit to how far it may have turned or gone.

    Every facing that lays a wall of the cloud along a wall of the plan is tried, at two kinds of place: where the
    camera stood (`near`: a camera that was re-aimed), and, with `anywhere`, wherever two crossing walls of the
    cloud land on two walls of the plan at their distances (one that was carried). Each try is squared and fitted as
    any cloud is (`square`, `fit`) and scored: the share of the cloud's walls, by their points, that the plan's
    walls account for from there (`Fit.share`), less the share of the cloud's points that would lie outside every
    room.

    **The test of confidence.** An answer counts when it is fitted and scores `MIN_SCORE`; one found anywhere must
    also hold walls in two directions (`CROSS_DEG`), since one wall says how far the camera is from it and nothing
    about where along it. The camera is taken to have stayed where it stood when an answer there scores within
    `RIVAL` of the best anywhere: a camera that still fits its place was turned, not carried. And the answer must
    stand alone: when another, more than `SAME` away, scores within `RIVAL` of it, the view fits two places (a
    corner like other corners, one wall like other walls), and it is "ambiguous": nothing is placed."""
    if not found:
        return Located("no_walls")
    if not plan:
        return Located("no_plan")
    rng = np.random.default_rng(0)
    xy = np.asarray(points, dtype=np.float64)[:, :2]
    sample = xy[rng.choice(len(xy), min(len(xy), SAMPLE_XY), replace=False)] if len(xy) else xy
    top = sorted(found, key=lambda line: -line.weight)[:TOP]
    bearings = _bearings(top, plan)
    here: list[_Try] = []
    if near is not None:
        here = [t for b in bearings if (t := _try(found, near[0], near[1], b, plan, rooms, sample)) is not None]
    there: list[_Try] = []
    if anywhere:
        seen: set[tuple[int, int, int]] = set()
        for bearing in bearings:
            for x, y in _corners(top, bearing, plan, rooms):
                key = (round(x / GRID[0]), round(y / GRID[0]), round(math.degrees(bearing) / GRID[1]))
                if key in seen:
                    continue
                seen.add(key)
                tried = _try(found, float(x), float(y), bearing, plan, rooms, sample)
                if tried is not None:
                    there.append(tried)
    stayed, _ = _pick(here, False)
    carried, _ = _pick(there, True)
    if stayed is not None and (carried is None or stayed.score >= carried.score - RIVAL):
        best, rival = _pick(here, False)
        by = "turned"
    elif carried is not None:
        best, rival = _pick([*there, *(t for t in here if t.crossed)], True)
        by = "walls"
    else:
        return Located("no_match")
    assert best is not None
    told = {"x": best.x, "y": best.y, "bearing": best.bearing, "by": by, "score": round(best.score, 3),
            "rival": round(rival, 3), "walls": best.walls, "scale": best.scale}  # fmt: skip
    return Located("ambiguous" if rival >= best.score - RIVAL else "placed", **told)


def hold(
    found: Sequence[Line],
    x: float,
    y: float,
    bearing: float,
    plan: Sequence[Edge],
    rooms: Sequence[Sequence[tuple[float, float]]],
    points: np.ndarray,
) -> Located:
    """Hold a cloud whose camera is already known to stand near x, y facing `bearing` (a tracked camera's keyframe)
    to the plan's walls: squared and fitted from there within the bounds any cloud is (`TURN_DEG`, `FIT_SLIDE_M`,
    `SIZE`), and counted by `locate`'s score. "placed" with where the walls put it, or "no_walls", "no_plan", or
    "no_match": then the camera stands where it was taken to. Nothing else is tried: the track says where it is,
    and the walls only take out what it has drifted."""
    if not found:
        return Located("no_walls")
    if not plan:
        return Located("no_plan")
    rng = np.random.default_rng(0)
    xy = np.asarray(points, dtype=np.float64)[:, :2]
    sample = xy[rng.choice(len(xy), min(len(xy), SAMPLE_XY), replace=False)] if len(xy) else xy
    tried = _try(found, x, y, bearing, plan, rooms, sample)
    if tried is None or tried.score < MIN_SCORE:
        return Located("no_match")
    return Located("placed", tried.x, tried.y, tried.bearing, "held", round(tried.score, 3), 0.0, tried.walls,
                   tried.scale)  # fmt: skip


# ---- what is kept


@dataclass(frozen=True)
class Align:
    """A cloud's squaring and fit to the plan, and what they were made from."""

    cloud: str  # the cloud's id
    square: Square
    basis: str  # the spot it was squared from (spatial/cloud_scene.py `SpotAt.basis`)
    plan: str  # and the plan's walls (`plan_key`)
    fit: Fit | None = None  # None in a record kept before clouds were fitted: it is made again
    made: int = 0  # which search for walls made it (`MADE`): one made by an earlier search is made again

    def holds(self, cloud: str, basis: str, plan: str) -> bool:
        """Whether it is still of this cloud, pose, and plan, and made the way one is made now."""
        same = (self.cloud, self.basis, self.plan, self.made) == (cloud, basis, plan, MADE)
        return self.fit is not None and same


def plan_key(rooms: Sequence[Sequence[tuple[float, float]]]) -> str:
    """Changes when the plan's walls do."""
    flat = json.dumps([[[round(float(x), 3), round(float(y), 3)] for x, y in ring] for ring in rooms])
    return hashlib.sha256(flat.encode()).hexdigest()[:16]


def make(
    cloud: str,
    points: np.ndarray,
    at: tuple[float, float, float],
    basis: str,
    rooms: Sequence[Sequence[tuple[float, float]]],
    height: tuple[float, float] | None = None,
    found: Sequence[Line] | None = None,
) -> Align:
    """A levelled cloud's `Align`: `points` (its own frame, its camera on the z axis looking along +y) squared to the
    plan's `rooms` and fitted to them, for a camera taken to stand at `at` (x, y, and bearing in the scene). `height`
    is `fit`'s; `found` are the cloud's walls, looked for here when not given."""
    found = lines(planes(points)) if found is None else found
    walls = edges_of(rooms)
    squared = square(found, at[0], at[1], at[2], walls)
    turn = squared.turn_deg if squared.state == "squared" else 0.0
    return Align(cloud, squared, basis, plan_key(rooms), fit(found, at[0], at[1], at[2], walls, turn, height), MADE)


def fit_of(raw: Mapping[str, Any]) -> Fit:
    """A `Fit` from its dict (`asdict`, or read back from JSON). Raises KeyError, TypeError, or ValueError."""
    walls = tuple(
        Matched(
            (float(w["a"][0]), float(w["a"][1])),
            (float(w["b"][0]), float(w["b"][1])),
            (float(w["normal"][0]), float(w["normal"][1])),
            float(w["offset"]),
            float(w["lo"]),
            float(w["hi"]),
            float(w["off_deg"]),
            float(w["before_m"]),
            float(w["after_m"]),
        )
        for w in raw["walls"]
    )
    return Fit(
        str(raw["state"]),
        float(raw["scale"]),
        float(raw["up"]),
        float(raw["dx"]),
        float(raw["dy"]),
        float(raw["turn_deg"]),
        walls,
        int(raw["seen"]),
        float(raw.get("share", 0.0)),
    )


def keep(folder: Path, found: Align) -> None:
    """Keep a camera's `Align` beside its cloud (`folder` is the camera's own)."""
    places.write_whole(folder / ALIGN, (json.dumps(asdict(found)) + "\n").encode())


def kept(folder: Path) -> Align | None:
    """The `Align` kept in a camera's folder, if one reads."""
    raw = json_state.read(folder / ALIGN)
    if not isinstance(raw, dict):
        return None
    try:
        fitted = None if raw.get("fit") is None else fit_of(raw["fit"])
        made = int(raw.get("made", 0))
        return Align(str(raw["cloud"]), Square(**raw["square"]), str(raw["basis"]), str(raw["plan"]), fitted, made)
    except KeyError, TypeError, ValueError:
        return None
