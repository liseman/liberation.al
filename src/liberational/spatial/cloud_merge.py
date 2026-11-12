"""How a cloud is brought into agreement with others, and what one shot saw (#546). spatial/cloud_map.py uses both
to build a place's map. Nothing here changes a cloud or writes anything.

**Agreement** (`agree`). A cloud the plan's walls don't hold stands at its camera's spot, at the model's own size,
and can be a metre out. It is brought into agreement with clouds that are held over the surfaces they share: the
same wall and the same raised flat surface (a table top, a counter) are found in both (`Surfaces`, read off the
cloud's planes; `Placed`, where a stand puts them in the scene), and its remaining turn, slide, and size are solved
from them, plane to plane: a wall pair says the turn and the slide across the wall, a table top's height says the
size, and where it stands says the slide along the wall. The answer is taken only within `TURN_DEG`, `SLIDE_M`, and
spatial/cloud_align.py's `SIZE`, and only when the surfaces it shared to begin with then lie within `CLOSE_M` of
their counterparts; otherwise the two are "apart". Points are never matched to points: registering overlapping
clouds from their points and colours was tried (spatial/cloud_align.py's docstring) and moved clouds off the plan.

Surfaces are paired by being near each other, which is a guess: a cupboard's front and the wall behind it are near
each other too. So this is used only to place a cloud nothing else places. Clouds the plan's walls hold are never
compared this way (two cameras either side of one wall would "disagree"); for them, what counts is what was seen.

**What a shot saw** (`View`, `through`). A shot saw as far as its points in each direction. A point nearer than
that, in its picture, is one it looked straight through: there was nothing there when it was taken. What lies
beyond its points, or outside its picture, it says nothing about.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from liberational.spatial import cloud_align

REACH_M = 1.5  # surfaces of two clouds this near each other may be the same surface
TURN_DEG = 30.0  # the most a loose camera is turned to agree with a held one
SLIDE_M = 1.5  # and the most it is slid
CLOSE_M = 0.15  # matched surfaces left farther apart than this don't agree
BEND_DEG = 15.0  # walls of one room run this far off each other in two clouds (a cloud's own walls bend up to 13)
MINOR = 0.1  # a shared surface holding less than this of their weight may be out: a small plane is a poor witness
MIN_SHARED = 0.75  # of the weight of a cloud's surfaces near the others' that must lie close to them
RAISED_M = 0.15  # a flat surface this far above the floor is a table top or a counter, not the floor
LEVEL_DEG = 10.0  # a plane within this of level is flat
HOLD_SLIDE = 0.02  # how hard the slide and the size are held to nothing, as a share of the surfaces' weight
GRID = 64  # cells across and down a picture, for how far a shot saw in each direction
BEHIND = (0.05, 0.2)  # a point nearer than what a shot saw there by this share and this many metres was looked through


@dataclass(frozen=True)
class Flat:
    """A raised flat surface of a cloud, in the cloud's own frame: where its middle is, and how high."""

    x: float
    y: float
    z: float
    weight: float


@dataclass(frozen=True)
class Surfaces:
    """What a levelled cloud has to agree with another by: its walls, as lines on its floor, and its raised flats."""

    walls: tuple[cloud_align.Line, ...] = ()
    flats: tuple[Flat, ...] = ()


def surfaces(planes: Sequence[cloud_align.Plane]) -> Surfaces:
    """The `Surfaces` among a levelled cloud's planes (spatial/cloud_align.py `planes`)."""
    flats = []
    for plane in planes:
        level = abs(float(plane.normal[2])) >= math.cos(math.radians(LEVEL_DEG))
        height = float(np.median(plane.on[:, 2]))
        if level and height >= RAISED_M:
            flats.append(Flat(float(plane.centre[0]), float(plane.centre[1]), height, float(plane.points)))
    return Surfaces(tuple(cloud_align.lines(planes)), tuple(flats))


@dataclass(frozen=True)
class Stand:
    """Where a camera's cloud stands in the scene: its camera on the floor, facing `bearing`, the cloud resized."""

    x: float
    y: float
    bearing: float
    scale: float = 1.0

    def floor(self, xy: np.ndarray) -> np.ndarray:
        """Points of the cloud's own floor (N x 2), in the scene."""
        s, c = math.sin(self.bearing), math.cos(self.bearing)
        own = np.asarray(xy, dtype=np.float64) * self.scale
        return np.c_[own[:, 0] * c + own[:, 1] * s + self.x, -own[:, 0] * s + own[:, 1] * c + self.y]


@dataclass(frozen=True)
class Wall:
    """A wall of a cloud on the scene's floor."""

    centre: tuple[float, float]
    normal: tuple[float, float]
    half: float  # half its reach
    weight: float


@dataclass(frozen=True)
class Placed:
    """A cloud's surfaces where it stands in the scene: its walls, and its raised flats as (x, y, z, weight)."""

    walls: tuple[Wall, ...] = ()
    flats: tuple[tuple[float, float, float, float], ...] = ()

    def __add__(self, other: Placed) -> Placed:
        return Placed(self.walls + other.walls, self.flats + other.flats)


def placed(found: Surfaces, at: Stand) -> Placed:
    """A cloud's `Surfaces` where `at` puts them."""
    s, c = math.sin(at.bearing), math.cos(at.bearing)
    walls = []
    for line in found.walls:
        normal = (c * line.normal[0] + s * line.normal[1], -s * line.normal[0] + c * line.normal[1])
        along = np.array([-line.normal[1], line.normal[0]])
        middle = at.floor((np.array(line.normal) * line.offset + along * (line.lo + line.hi) / 2)[None])[0]
        walls.append(
            Wall((float(middle[0]), float(middle[1])), normal, at.scale * (line.hi - line.lo) / 2, line.weight)
        )
    flats = []
    for flat in found.flats:
        x, y = at.floor(np.array([[flat.x, flat.y]]))[0]
        flats.append((float(x), float(y), flat.z * at.scale, flat.weight))
    return Placed(tuple(walls), tuple(flats))


def _gap(wall: Wall, other: Wall, within_deg: float) -> float | None:
    """How far `wall`'s middle stands from `other`, across it, when the two run the same way (within `within_deg`)
    and `wall` is beside `other`; None when they don't or it isn't."""
    normal = np.array(other.normal)
    if abs(float(np.array(wall.normal) @ normal)) < math.cos(math.radians(within_deg)):
        return None
    apart = np.array(wall.centre) - np.array(other.centre)
    if abs(float(apart @ np.array([-normal[1], normal[0]]))) > other.half + wall.half:
        return None
    return abs(float(apart @ normal))


def _shared(
    mine: Placed, theirs: Placed, reach: float, within_deg: float
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Which of `mine` have a counterpart among `theirs` within `reach`: for each such wall of mine its place in
    `mine.walls` and its counterpart's in `theirs.walls`, and the same for the raised flats."""
    walls, to_walls, flats, to_flats = [], [], [], []
    for i, wall in enumerate(mine.walls):
        gaps = [(gap, j) for j, other in enumerate(theirs.walls) if (gap := _gap(wall, other, within_deg)) is not None]
        if gaps and min(gaps)[0] <= reach:
            walls.append(i)
            to_walls.append(min(gaps)[1])
    for i, flat in enumerate(mine.flats):
        gaps = [
            (math.hypot(other[0] - flat[0], other[1] - flat[1]), j)
            for j, other in enumerate(theirs.flats)
            if abs(other[2] - flat[2]) <= 2 * CLOSE_M
        ]
        if gaps and min(gaps)[0] <= reach:
            flats.append(i)
            to_flats.append(min(gaps)[1])
    return walls, to_walls, flats, to_flats


@dataclass(frozen=True)
class Agreed:
    """Whether a cloud agrees with what is already there, and how a loose one was brought to."""

    state: str  # "agreed" or "apart"
    at: Stand | None = None  # where it stands, when agreed
    moved_m: float = 0.0  # how far its camera moved, and turned, and how it was resized, to agree
    turned_deg: float = 0.0
    resized: float = 1.0
    walls: int = 0  # the walls and the raised flats the two share
    flats: int = 0
    off_m: float = 0.0  # how far apart the shared surfaces are left


def _judge(here: Placed, theirs: Placed, shared: tuple[list[int], list[int], list[int], list[int]]) -> Agreed:
    """Whether the surfaces of `here` paired with `theirs` in `shared` (`_shared`) lie on their counterparts: within
    `CLOSE_M`, by at least `MIN_SHARED` of their weight, and with none that holds `MINOR` of it left out (a side
    wall a metre off is a second wall, however well the rest fits). What is reported is how far out the ones that
    don't are."""
    walls, to_walls, flats, to_flats = shared
    rows = []  # each shared surface's weight, and how far it is from its counterpart
    for i, j in zip(walls, to_walls, strict=True):
        gap = _gap(here.walls[i], theirs.walls[j], BEND_DEG)
        rows.append((here.walls[i].weight, REACH_M if gap is None else gap))
    for i, j in zip(flats, to_flats, strict=True):
        mine_at, other = here.flats[i], theirs.flats[j]
        rows.append(
            (mine_at[3], max(math.hypot(other[0] - mine_at[0], other[1] - mine_at[1]), abs(other[2] - mine_at[2])))
        )
    near = sum(weight for weight, _ in rows)
    close = sum(weight for weight, off in rows if off <= CLOSE_M)
    out = [off for weight, off in rows if off > CLOSE_M]
    stray = any(weight >= MINOR * near for weight, off in rows if off > CLOSE_M)
    agreed = close >= MIN_SHARED * near and not stray
    off = max(out, default=max(off for _, off in rows))
    return Agreed("agreed" if agreed else "apart", walls=len(walls), flats=len(flats), off_m=round(float(off), 3))


def agree(loose: Surfaces, at: Stand, theirs: Placed) -> Agreed | None:
    """Bring a cloud the plan doesn't hold (`loose`, standing at `at`) into agreement with `theirs` over the
    surfaces they share (the module docstring). None when they share none: then nothing is known of it from them,
    and it is left as it is."""
    turn, slide, size = 0.0, np.zeros(2), 1.0
    eye = np.array([at.x, at.y])
    before = placed(loose, at)
    first = _shared(before, theirs, REACH_M, TURN_DEG)  # what it shares with them as it stands: all of it must agree
    for reach in (REACH_M, REACH_M / 2, REACH_M / 4, REACH_M / 4):
        now = Stand(at.x + slide[0], at.y + slide[1], at.bearing + turn, at.scale * size)
        walls, to_walls, flats, to_flats = _shared(placed(loose, now), theirs, reach, TURN_DEG)
        if not walls and not flats:
            return None if reach == REACH_M else Agreed("apart", off_m=REACH_M)
        pairs = [(before.walls[i], theirs.walls[j]) for i, j in zip(walls, to_walls, strict=True)]
        tops = [(before.flats[i], theirs.flats[j]) for i, j in zip(flats, to_flats, strict=True)]
        # The turn: what lays its walls along theirs (a wall has no sense, so the smaller angle either way).
        if pairs:
            angles = [
                (math.atan2(b.normal[1], b.normal[0]) - math.atan2(a.normal[1], a.normal[0]) + math.pi / 2) % math.pi
                - math.pi / 2
                for a, b in pairs
            ]
            turn = -float(np.average(angles, weights=[a.weight * a.half for a, _ in pairs]))  # bearings run clockwise
        # Then the size and the slide, least squares: each wall's middle on its counterpart, each table top's middle
        # on its counterpart's and as high, the slide and the size held lightly to none.
        s, c = math.sin(turn), math.cos(turn)
        spin = np.array([[c, s], [-s, c]])
        rows, wants = [], []
        total = sum(a.weight for a, _ in pairs) + sum(a[3] for a, _ in tops)
        for a, b in pairs:
            arm, normal = spin @ (np.array(a.centre) - eye), np.array(b.normal)
            w = math.sqrt(a.weight)
            rows.append(w * np.array([float(normal @ arm), normal[0], normal[1]]))
            wants.append(w * float(normal @ (np.array(b.centre) - eye)))
        for a, b in tops:
            arm = spin @ (np.array(a[:2]) - eye)
            w = math.sqrt(a[3])
            rows += [w * np.array([arm[0], 1.0, 0.0]), w * np.array([arm[1], 0.0, 1.0]), w * np.array([a[2], 0, 0])]
            wants += [w * (b[0] - eye[0]), w * (b[1] - eye[1]), w * b[2]]
        hold = math.sqrt(HOLD_SLIDE * total)
        rows += [np.array([hold, 0.0, 0.0]), np.array([0.0, hold, 0.0]), np.array([0.0, 0.0, hold])]
        wants += [hold, 0.0, 0.0]
        answer, *_ = np.linalg.lstsq(np.array(rows), np.array(wants), rcond=None)
        size, slide = float(answer[0]), answer[1:]
    now = Stand(at.x + float(slide[0]), at.y + float(slide[1]), at.bearing + turn, at.scale * size)
    found = _judge(placed(loose, now), theirs, first) if first[0] or first[2] else None
    within = (
        abs(math.degrees(turn)) <= TURN_DEG
        and float(np.linalg.norm(slide)) <= SLIDE_M
        and cloud_align.SIZE[0] <= now.scale <= cloud_align.SIZE[1]
    )
    if found is None or found.state != "agreed" or not within:
        return Agreed("apart", walls=len(pairs), flats=len(tops), off_m=REACH_M if found is None else found.off_m)
    return Agreed("agreed", now, round(float(np.linalg.norm(slide)), 3), round(abs(math.degrees(turn)), 1),
                  round(size, 4), found.walls, found.flats, found.off_m)  # fmt: skip


@dataclass(frozen=True)
class View:
    """What one shot saw: where its camera was, and how far it saw in each direction."""

    eye: np.ndarray  # the scene to the shot's camera frame (x right, y down, z forward), 4 x 4
    focal: float  # of the picture, in its pixels
    width: int
    height: int
    far: np.ndarray  # GRID x GRID, flat: how far it saw in each cell of its picture (0 where it saw nothing)


def _cells(eye: np.ndarray, focal: float, width: int, height: int, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scene points in a shot's picture: the cell of the picture each falls in (-1 outside it), and its depth."""
    own = xyz @ eye[:3, :3].T + eye[:3, 3]
    depth = own[:, 2]
    ahead = depth > 0.05
    safe = np.where(ahead, depth, 1.0)
    col = np.floor((focal * own[:, 0] / safe + width / 2) * GRID / max(width, 1)).astype(int)
    row = np.floor((focal * own[:, 1] / safe + height / 2) * GRID / max(height, 1)).astype(int)
    inside = ahead & (col >= 0) & (col < GRID) & (row >= 0) & (row < GRID)
    return np.where(inside, row * GRID + col, -1), depth


def view_of(xyz: np.ndarray, eye: np.ndarray, focal: float, width: int, height: int) -> View:
    """The `View` of a shot whose points stand at `xyz` in the scene: in each cell of its picture, the far end
    (nine tenths of the way through them) of the points it has there."""
    cells, depths = _cells(eye, focal, width, height, xyz)
    far = np.zeros(GRID * GRID)
    seen = cells >= 0
    order = np.lexsort((depths[seen], cells[seen]))
    cell, depth = cells[seen][order], depths[seen][order]
    if len(cell):
        first = np.flatnonzero(np.r_[True, cell[1:] != cell[:-1]])
        count = np.diff(np.r_[first, len(cell)])
        far[cell[first]] = depth[first + np.minimum((count * 0.9).astype(int), count - 1)]
    return View(eye, focal, width, height, far)


def through(view: View, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Which of the scene points `xyz` a shot looked straight through (the module docstring's What a shot saw):
    they lie in its picture, in a direction it saw something, and nearer than what it saw there by more than
    `BEHIND`. And which of them lie in its view at all, in front of or at what it saw."""
    where, depth = _cells(view.eye, view.focal, view.width, view.height, xyz)
    reach = np.where(where >= 0, view.far[np.maximum(where, 0)], 0.0)
    seen = (where >= 0) & (reach > 0)
    return seen & (depth < reach * (1 - BEHIND[0]) - BEHIND[1]), seen & (depth <= reach * (1 + BEHIND[0]) + BEHIND[1])
