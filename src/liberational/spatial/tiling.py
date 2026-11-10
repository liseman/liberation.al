"""Reconstruction output to a plan whose rooms tile (part of spatial memory).

A stored plan (a Pascal scene, spatial/plan_scene.py) has its walls on the edges its rooms share, so the rooms must
tile the footprint exactly on shared wall centrelines, with each door on the wall between the two spaces either side
of it. Reconstruction gives rooms as it saw them: a room model's polygons (RoomFormer, SpatialLM) are
inner faces with a wall's width of gap between neighbours, edges a little skewed, small overlaps, and rooms only
partly seen; the classical baseline (spatial/floorplan.py) is already on centrelines but can leave gaps. `tile` turns
either into rooms that tile:

1. **Square up.** The rooms' dominant edge direction (length-weighted, modulo 90 degrees) becomes the axes, and an
   edge within SQUARE_DEG of an axis is made exactly axis-aligned. An edge further off stays as it is and is drawn
   as steps on the grid below.
2. **Shared centrelines.** Edge coordinates within Settings.snap of that line's first, along one axis, are one wall
   line. When the rooms are inner faces (`faces`), a line with rooms on both sides goes to the middle of their faces,
   and a line with a room on one side only is an exterior face, moved out by half the exterior wall.
3. **Tile.** The wall lines make a grid, and each grid cell goes to the room whose snapped polygon covers its centre;
   where two do (an overlap), to the one whose squared polygon, before that snap, covers more of the cell. A room
   left in pieces keeps its largest; the other pieces, and rooms under MIN_ROOM_M2, go to the neighbour they share
   the most wall with. A sliver that shares a wall with none is outside; a leftover piece that shares none stays a gap.
4. **Gaps.** A gap the rooms enclose (not reaching outside) of up to Settings.absorb_m2 goes to the neighbour it shares
   the most wall with, when it has one; a larger one, or one with no neighbour, is a part nobody has seen (a closed
   door, an unscanned room). It is kept as an area named "Not seen yet" (`Tiled.unexplored`), which `to_scene` writes
   among the scene's own records for the Places tab to draw.
5. **Openings.** Each detected door or window goes on the nearest wall between two spaces within Settings.reach,
   named by those two spaces ("exterior" first when one is outside), and is clamped to that wall: its width to the
   wall's length less CORNER_M clear at each end, and its centre slid inside that, so no opening overflows its wall
   or sits in a corner. One that fits no wall, or opens into a part not seen yet (there is no floor there to open
   onto), is dropped and said so (`Tiled.notes`).

Everything is in the squared-up frame; `Tiled.angle` is the rotation from the input's (radians, anticlockwise).
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage

from liberational.spatial import floorplan, plan_scene

SQUARE_DEG = 10.0  # an edge this close to an axis is made exactly axis-aligned
MIN_ROOM_M2 = 1.0  # a room smaller than this after tiling is a sliver, given to a neighbour
CORNER_M = 0.1  # wall left at each side of an opening, so it never sits in a corner
MIN_OPENING_M = 0.5  # an opening narrower than this after clamping is no opening
EXTERIOR = "exterior"
UNSEEN = "Not seen yet"
OUTSIDE, EMPTY = -1, -2  # cell owners besides rooms: past the footprint, or a gap inside it

Point = tuple[float, float]


@dataclass(frozen=True)
class Room:
    id: str
    polygon: tuple[Point, ...]
    name: str = ""
    kind: str = "other"


@dataclass(frozen=True)
class Opening:
    kind: str  # "door" or "window"
    at: Point
    width: float
    between: tuple[str, str] | None = None  # as detected, if known; the converter names the wall's two spaces


@dataclass(frozen=True)
class Settings:
    faces: bool = True  # the rooms are inner faces (a room model's); False: already on centrelines (the baseline)
    exterior_wall: float = 0.2
    partition_wall: float = 0.1
    snap: float = 0.3  # edge coordinates this close along one axis are one wall line
    absorb_m2: float = 1.5  # an enclosed gap up to this big goes to a neighbour; larger is a part not seen yet
    reach: float = 0.5  # how far a detected opening may be from the wall it goes on


@dataclass
class Tiled:
    rooms: list[Room]
    openings: list[Opening]
    unexplored: list[tuple[Point, ...]]  # areas inside the footprint nobody has seen, emitted as voids
    angle: float = 0.0  # radians the input was turned to square it up
    notes: list[str] = field(default_factory=list)  # what was dropped or merged, for the log


def _area(poly: Sequence[Point]) -> float:
    return 0.5 * sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(poly, [*poly[1:], poly[0]], strict=True))


def _ccw(poly: Sequence[Point]) -> list[Point]:
    pts = [(float(x), float(y)) for x, y in poly]
    return pts if _area(pts) >= 0 else pts[::-1]


def _turn(p: Point, angle: float) -> Point:
    c, s = math.cos(angle), math.sin(angle)
    return (c * p[0] - s * p[1], s * p[0] + c * p[1])


def dominant_angle(rooms: Iterable[Room]) -> float:
    """The rooms' main edge direction modulo 90 degrees, in [-pi/4, pi/4): the length-weighted mean of 4 x each
    edge's angle on the circle, so edges at 0 and 90 degrees agree."""
    sx = sy = 0.0
    for r in rooms:
        pts = r.polygon
        for a, b in zip(pts, [*pts[1:], pts[0]], strict=True):
            dx, dy = b[0] - a[0], b[1] - a[1]
            length = math.hypot(dx, dy)
            theta = math.atan2(dy, dx)
            sx += length * math.cos(4 * theta)
            sy += length * math.sin(4 * theta)
    return math.atan2(sy, sx) / 4 if (sx or sy) else 0.0


def _square(poly: Sequence[Point]) -> list[Point]:
    """`poly` with each edge within SQUARE_DEG of an axis made exactly axis-aligned (its two ends moved to their
    mean coordinate across it)."""
    pts = [list(p) for p in poly]
    n = len(pts)
    tol = math.radians(SQUARE_DEG)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        theta = math.atan2(b[1] - a[1], b[0] - a[0]) % math.pi
        if min(theta, math.pi - theta) <= tol:  # nearly horizontal
            a[1] = b[1] = (a[1] + b[1]) / 2
        elif abs(theta - math.pi / 2) <= tol:  # nearly vertical
            a[0] = b[0] = (a[0] + b[0]) / 2
    return [(p[0], p[1]) for p in pts]


@dataclass
class _Edge:
    room: int
    vertical: bool
    coord: float
    lo: float
    hi: float
    side: int  # +1: the room is on the edge's + side (larger x or y), -1: on its - side


def _edges(index: int, poly: Sequence[Point]) -> list[_Edge]:
    out = []
    for a, b in zip(poly, [*poly[1:], poly[0]], strict=True):
        if a[0] == b[0] and a[1] != b[1]:  # vertical; a CCW room's interior is left of its direction
            out.append(_Edge(index, True, a[0], min(a[1], b[1]), max(a[1], b[1]), -1 if b[1] > a[1] else 1))
        elif a[1] == b[1] and a[0] != b[0]:
            out.append(_Edge(index, False, a[1], min(a[0], b[0]), max(a[0], b[0]), 1 if b[0] > a[0] else -1))
    return out


def _lines(edges: Sequence[_Edge], s: Settings) -> list[tuple[float, float, float]]:
    """(low, high, value) for each wall line along one axis: edge coordinates within `snap` of the line's first are
    one line; its value is the middle of its faces when rooms are on both sides, else (an exterior face) half an
    exterior wall out, or the edges' length-weighted mean when the rooms are already centrelines."""
    out: list[tuple[float, float, float]] = []
    group: list[_Edge] = []

    def close() -> None:
        if not group:
            return
        lo, hi = min(e.coord for e in group), max(e.coord for e in group)
        weight = sum(e.hi - e.lo for e in group) or 1.0
        mean = sum(e.coord * (e.hi - e.lo) for e in group) / weight
        sides = {e.side for e in group}
        if not s.faces:
            value = mean
        elif sides == {-1, 1}:
            # Faces of one wall from its two sides: a room on the + side has its face on the line's high side.
            faces_minus = [e.coord for e in group if e.side == -1]
            faces_plus = [e.coord for e in group if e.side == 1]
            value = (max(faces_minus) + min(faces_plus)) / 2
        else:
            value = mean - s.exterior_wall / 2 * group[0].side
        out.append((lo, hi, value))

    for e in sorted(edges, key=lambda e: e.coord):
        if group and e.coord - group[0].coord > s.snap:
            close()
            group = []
        group.append(e)
    close()
    return out


def _snap(value: float, lines: Sequence[tuple[float, float, float]]) -> float:
    for lo, hi, v in lines:
        if lo - 1e-9 <= value <= hi + 1e-9:
            return v
    return value


def _inside(x: float, y: float, poly: Sequence[Point]) -> bool:
    inside = False
    for (ax, ay), (bx, by) in zip(poly, [*poly[1:], poly[0]], strict=True):
        if (ay > y) != (by > y) and x < ax + (y - ay) * (bx - ax) / (by - ay):
            inside = not inside
    return inside


def _share(poly: Sequence[Point], x0: float, x1: float, y0: float, y1: float) -> float:
    """How much of the cell [x0, x1] x [y0, y1] the polygon covers, sampled on a 4 x 4 grid."""
    hits = sum(
        _inside(x0 + (i + 0.5) * (x1 - x0) / 4, y0 + (j + 0.5) * (y1 - y0) / 4, poly)
        for i in range(4)
        for j in range(4)
    )
    return hits / 16


def _neighbour(owner: np.ndarray, mask: np.ndarray, xs: Sequence[float], ys: Sequence[float]) -> int | None:
    """The room sharing the most wall length with the cells in `mask`, or None."""
    shared: dict[int, float] = {}
    nx, ny = owner.shape
    for i, j in zip(*np.nonzero(mask), strict=True):
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < nx and 0 <= b < ny and not mask[a, b] and owner[a, b] >= 0:
                length = (ys[j + 1] - ys[j]) if di else (xs[i + 1] - xs[i])
                shared[int(owner[a, b])] = shared.get(int(owner[a, b]), 0.0) + length
    return max(shared, key=lambda k: shared[k]) if shared else None


def _cells_area(mask: np.ndarray, xs: Sequence[float], ys: Sequence[float]) -> float:
    w = np.diff(np.asarray(xs))[:, None]
    h = np.diff(np.asarray(ys))[None, :]
    return float((mask * w * h).sum())


Run = tuple[bool, float, float, float, int, int]  # vertical, coord, lo, hi, owner below or left, owner above or right


def _wall_runs(owner: np.ndarray, xs: Sequence[float], ys: Sequence[float]) -> list[Run]:
    """Every wall between two different owners as a run along one grid line: (vertical, coord, lo, hi, a, b) with a
    on the low side. A run ends where either side's owner changes, as a wall does."""
    nx, ny = owner.shape
    padded = np.full((nx + 2, ny + 2), OUTSIDE)
    padded[1:-1, 1:-1] = owner
    runs: list[Run] = []

    def scan(vertical: bool, coord: float, pairs: Sequence[tuple[int, int]], cuts: Sequence[float]) -> None:
        start: int | None = None
        for k in range(len(pairs) + 1):
            here = pairs[k] if k < len(pairs) else None
            wall = here is not None and here[0] != here[1] and (here[0] >= 0 or here[1] >= 0)
            if start is not None and (not wall or here != pairs[start]):
                a, b = pairs[start]
                runs.append((vertical, coord, cuts[start], cuts[k], a, b))
                start = None
            if wall and start is None:
                start = k

    for i in range(nx + 1):  # x = xs[i], between columns i - 1 and i
        scan(True, xs[i], [(int(padded[i, j + 1]), int(padded[i + 1, j + 1])) for j in range(ny)], ys)
    for j in range(ny + 1):  # y = ys[j], between rows j - 1 and j
        scan(False, ys[j], [(int(padded[i + 1, j]), int(padded[i + 1, j + 1])) for i in range(nx)], xs)
    return runs


def tile(rooms: Sequence[Room], openings: Sequence[Opening] = (), settings: Settings | None = None) -> Tiled:
    """Rooms that tile, on shared centrelines, with the openings on their walls (see the module docstring)."""
    s = settings or Settings()
    notes: list[str] = []
    angle = -dominant_angle(rooms)
    polys = [_square(_ccw([_turn(p, angle) for p in r.polygon])) for r in rooms]
    edges = [e for i, p in enumerate(polys) for e in _edges(i, p)]
    vlines = _lines([e for e in edges if e.vertical], s)
    hlines = _lines([e for e in edges if not e.vertical], s)
    snapped = [[(_snap(x, vlines), _snap(y, hlines)) for x, y in p] for p in polys]
    xs = sorted({round(x, 4) for p in snapped for x, _ in p})
    ys = sorted({round(y, 4) for p in snapped for _, y in p})
    nx, ny = len(xs) - 1, len(ys) - 1
    owner = np.full((max(nx, 0), max(ny, 0)), EMPTY, dtype=np.int64)
    for i in range(nx):
        for j in range(ny):
            cx, cy = (xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2
            covering = [k for k, p in enumerate(snapped) if _inside(cx, cy, p)]
            if len(covering) == 1:
                owner[i, j] = covering[0]
            elif covering:  # an overlap: the room that saw more of this cell
                owner[i, j] = max(covering, key=lambda k: _share(polys[k], xs[i], xs[i + 1], ys[j], ys[j + 1]))
    # Outside: the empty cells that reach the grid's edge.
    empty = owner == EMPTY
    labels, _ = ndimage.label(np.pad(empty, 1, constant_values=True))
    owner[(labels[1:-1, 1:-1] == labels[0, 0]) & empty] = OUTSIDE
    # A room in pieces keeps its largest; the rest, and rooms too small to be rooms, go to a neighbour.
    for k, r in enumerate(rooms):
        parts, count = ndimage.label(owner == k)
        if count > 1:
            sizes = [_cells_area(parts == n, xs, ys) for n in range(1, count + 1)]
            keep = 1 + int(np.argmax(sizes))
            for n in range(1, count + 1):
                if n != keep:
                    piece = parts == n
                    owner[piece] = EMPTY
                    near = _neighbour(owner, piece, xs, ys)
                    owner[piece] = near if near is not None else EMPTY
        if 0 < _cells_area(owner == k, xs, ys) < MIN_ROOM_M2:
            piece = owner == k
            owner[piece] = EMPTY
            near = _neighbour(owner, piece, xs, ys)
            owner[piece] = near if near is not None else OUTSIDE
            notes.append(f"room {r.id} was a sliver under {MIN_ROOM_M2} m2, given to a neighbour")
    # Gaps the rooms enclose: small ones to a neighbour, larger ones are parts not seen yet.
    unexplored: list[tuple[Point, ...]] = []
    gaps, count = ndimage.label(owner == EMPTY)
    for n in range(1, count + 1):
        gap = gaps == n
        near = _neighbour(owner, gap, xs, ys)
        if _cells_area(gap, xs, ys) <= s.absorb_m2 and near is not None:
            owner[gap] = near
        else:
            unexplored.append(tuple((xs[i], ys[j]) for i, j in floorplan.cell_outline(gap)))
    out_rooms = []
    for k, r in enumerate(rooms):
        cells = owner == k
        if not cells.any():
            notes.append(f"room {r.id} left nothing after tiling")
            continue
        corners = floorplan.cell_outline(cells)
        out_rooms.append(Room(r.id, tuple((xs[i], ys[j]) for i, j in corners), r.name, r.kind))
    runs = _wall_runs(owner, xs, ys)
    out_openings = []
    names = {k: r.id for k, r in enumerate(rooms)}
    for o in openings:
        placed = _place(o, _turn(o.at, angle), runs, names, s)
        if isinstance(placed, str):
            notes.append(placed)
        else:
            out_openings.append(placed)
    return Tiled(out_rooms, out_openings, unexplored, angle, notes)


def _place(
    o: Opening,
    at: Point,
    runs: Sequence[Run],
    names: Mapping[int, str],
    s: Settings,
) -> Opening | str:
    """The opening on its nearest wall between two spaces, clamped to fit it; or why it fits none."""
    best = None
    for vertical, coord, lo, hi, a, b in runs:
        if a < 0 and b < 0:
            continue
        across, along = (at[0], at[1]) if vertical else (at[1], at[0])
        gap = abs(across - coord) + max(0.0, lo - along, along - hi)
        if gap <= s.reach and (best is None or gap < best[0]):
            best = (gap, vertical, coord, lo, hi, a, b)
    if best is None:
        return f"{o.kind} at {o.at[0]:.2f},{o.at[1]:.2f} is on no wall within {s.reach} m: dropped"
    _, vertical, coord, lo, hi, a, b = best
    if EMPTY in (a, b):
        # there is no floor in a part not seen yet, so nothing may open into one
        return f"{o.kind} at {o.at[0]:.2f},{o.at[1]:.2f} opens into a part not seen yet: dropped"
    width = min(o.width, hi - lo - 2 * CORNER_M)
    if width < MIN_OPENING_M:
        return f"{o.kind} at {o.at[0]:.2f},{o.at[1]:.2f}: its wall is {hi - lo:.2f} m, too short for one: dropped"
    along = at[1] if vertical else at[0]
    along = min(max(along, lo + CORNER_M + width / 2), hi - CORNER_M - width / 2)
    point = (coord, along) if vertical else (along, coord)
    sides = [EXTERIOR if k < 0 else names[k] for k in (a, b)]
    between = (sides[0], sides[1]) if sides[0] == EXTERIOR or sides[1] != EXTERIOR else (sides[1], sides[0])
    return Opening(o.kind, (round(point[0], 3), round(point[1], 3)), round(width, 3), between)


def from_plan(plan: floorplan.Floorplan) -> tuple[list[Room], list[Opening]]:
    """The classical baseline's rooms and openings, as `tile` takes them (with Settings(faces=False): they are
    already on centrelines)."""
    rooms = [Room(r.id, r.polygon, f"Room {i}") for i, r in enumerate(plan.rooms, start=1)]
    openings = [Opening(o.kind, o.at, o.width, o.between) for o in plan.openings]
    return rooms, openings


def to_scene(t: Tiled, title: str, s: Settings | None = None) -> str:
    """The plan as a Pascal scene (spatial/plan_scene.py), the format a place's plan is stored in: the rooms as named
    zones, a wall for each stretch of room edge (exterior where one room has it, a partition where two do), the
    openings on the walls they were placed on, and the not-seen-yet areas among the scene's own records. An opening
    that ended up on no wall is left out, with a note."""
    s = s or Settings()
    walls = plan_scene.cut_walls([r.polygon for r in t.rooms], s.exterior_wall, s.partition_wall)
    openings = []
    for o in t.openings:
        if plan_scene.on_a_wall(walls, o.at):
            openings.append((o.kind, o.at, o.width))
        else:
            t.notes.append(f"{o.kind} at {o.at[0]:.2f},{o.at[1]:.2f} is on no wall of the tiled rooms: dropped")
    records: dict[str, Any] = {"title": title}
    if t.unexplored:
        records["unexplored"] = {
            f"u{i}": {"name": UNSEEN, "poly": [[round(x, 3), round(y, 3)] for x, y in area]}
            for i, area in enumerate(t.unexplored, start=1)
        }
    scene = plan_scene.build(
        walls=walls,
        openings=openings,
        rooms=[(r.id, r.name or r.id, r.kind, r.polygon) for r in t.rooms],
        records=records,
    )
    return plan_scene.dump(scene)
