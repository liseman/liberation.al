"""Standing a camera's cloud upright, and square to the house (#466).

A cloud placed by its camera's spot alone is only as straight as that spot's facing and the cloud's own level. The
cloud holds better evidence of both: its floor is horizontal, its walls are vertical, and its walls run along the
walls of the owner's plan. Two rigid moves use that, and neither stretches or bends anything.

**The flat surfaces** (`planes`). The big planes of a cloud, found one at a time: the plane with the most points
within `TOL_M` of it, refitted to those points, taken out, and again, until none holds `MIN_SHARE` of the points or
`MOST` are found. A plane counts only when its points spread over it (`MIN_SPREAD_M`): a strip gives no direction.

**Level** (`stand`). A cloud whose floor was found (spatial/camera_clouds.py `floor`) is level already: its tilt and
roll are its floor's. One with no floor in view was taken to be upright, which a tilted camera isn't. Its walls say
which way is up: up runs along every wall, so two walls that meet give it outright and one wall gives the lean
across itself (the lean along it stays as it was). The cloud is turned so that up is up, its lowest points on the
floor. With no wall either, it stays as it was, and says so.

**Square** (`square`). The levelled cloud's walls, where its camera's spot puts them, are compared with the plan's
walls near the camera. *Turn:* the turn about the vertical, through the camera, that best lays the cloud's walls
along the plan's (each wall votes with its points, each plan wall with its length), the smallest such turn when a
room's right angles offer several; a turn past `TURN_DEG` from the facing the owner set is not made (a camera that
far round was moved, which spatial/cloud_scene.py follows by its pictures). *Slide:* each cloud wall that has one
clear counterpart on the plan (parallel, beside it, within `CLEAR_M`, and no other plan wall it could be) asks to
be moved onto it; the slide that suits them best, nearer walls counting more (a cloud's far depth is its least sure),
is made when it is within `SLIDE_M`. A cloud is the model's own estimate of size, so two facing walls can ask for
different slides: they share it, and neither is stretched to fit.

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
from collections.abc import Sequence
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
MOST = 8  # planes looked for in one cloud
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
    while len(out) < MOST and len(left) >= MIN_SHARE * total:
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
        if on.sum() < MIN_SHARE * total:
            break
        held = left[on]
        if spread[1] >= MIN_SPREAD_M:
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
    weight = np.array([line.weight for line in found])
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


# ---- what is kept


@dataclass(frozen=True)
class Align:
    """A cloud's squaring to the plan, and what it was made from."""

    cloud: str  # the cloud's id
    square: Square
    basis: str  # the spot it was squared from (spatial/cloud_scene.py `SpotAt.basis`)
    plan: str  # and the plan's walls (`plan_key`)

    def holds(self, cloud: str, basis: str, plan: str) -> bool:
        """Whether it is still of this cloud, spot, and plan."""
        return (self.cloud, self.basis, self.plan) == (cloud, basis, plan)


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
) -> Align:
    """A levelled cloud's `Align`: `points` (its own frame, its camera on the z axis looking along +y) squared to the
    plan's `rooms`, for a camera the owner put at `at` (x, y, and bearing in the scene)."""
    squared = square(lines(planes(points)), at[0], at[1], at[2], edges_of(rooms))
    return Align(cloud, squared, basis, plan_key(rooms))


def keep(folder: Path, found: Align) -> None:
    """Keep a camera's `Align` beside its cloud (`folder` is the camera's own)."""
    places.write_whole(folder / ALIGN, (json.dumps(asdict(found)) + "\n").encode())


def kept(folder: Path) -> Align | None:
    """The `Align` kept in a camera's folder, if one reads."""
    raw = json_state.read(folder / ALIGN)
    if not isinstance(raw, dict):
        return None
    try:
        return Align(str(raw["cloud"]), Square(**raw["square"]), str(raw["basis"]), str(raw["plan"]))
    except KeyError, TypeError, ValueError:
        return None
