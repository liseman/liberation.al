"""Classical floorplan baseline: a point cloud in metres to room polygons and openings, written as floorplan DSL.

Part of spatial memory. The learned plan models (RoomFormer, CAGE) have to beat this. Pure numpy
and scipy, no model, no daemon. The steps:

1. **Floor plane.** RANSAC over planes whose normal is within 25 degrees of `Settings.up` (the cloud's rough up
   direction: (0, 0, 1) for a z-up cloud, (0, -1, 0) for a camera-frame one). Each candidate is refitted by least
   squares to the points near it. Of those with at least 30% of the best-supported one's inliers and within
   `PARALLEL_DEG` (5 degrees) of it, the one lowest under the cloud's centre is the floor. A caller can pass a known
   floor (`floor=`) instead of finding one.
2. **Density image.** Points `Settings.band` above the floor (1.0 to 1.9 m: above furniture, below door heads)
   are counted into a top-down image of `Settings.cell` (5 cm) cells, after a rotation that lines the walls up with
   the axes. That Manhattan angle is the one whose x and y histograms are sharpest: the wall band's, when it holds
   at least `MIN_BAND_FOR_ANGLE` points, and otherwise the band plus the walked feet. `walkable` points (where people
   walked, weighted by `walk_weights`) add an outline that stands in for walls the cloud lacks.
3. **Walls and rooms.** Each row and column of the image's dense cells gives wall runs; nearby runs join into one
   axis-aligned wall line at their length-weighted middle, and the walked outline is a wall where the cloud has
   none. Gaps along a line up to `Settings.max_opening`, and wall ends short of a perpendicular wall by up to that
   much, are bridged so rooms close. The free space between the bridged walls splits into connected regions: one
   touching the border is outside, and an enclosed region of at least `Settings.min_room_area` is a room (a smaller
   one stays outside). Wall lines then form a grid; a cell goes to the room with the most of it, when that is at
   least 30% of the cell, and each room's polygon is the outline of its cells, so every corner lies on two wall
   lines. A line with outside on one side and rooms on the other, for most of its length, is the inner face of an
   exterior wall, which a scan from inside only sees, so it moves out by half `Settings.exterior_wall` to the
   centreline the DSL uses.
4. **Openings.** A bridged gap at least `Settings.min_opening` wide is a door between two rooms. On an exterior
   wall it is a window if the scan has wall below it (`Settings.low_band`), else a door.

Known limits of this baseline: walls must be roughly Manhattan (angled walls snap to the nearest axis); an opening
wider than `max_opening` in an exterior wall lets a room leak outside and vanish; a wall line that is exterior
along part of its length and a partition along the rest moves wholesale by its majority; a door whose edge is
less than `min_wall` from a corner reads as running into the corner (its centre off by up to `min_wall` / 2); and
room kinds and names aren't known (every room is kind `other`).
"""

import dataclasses
import math
import re
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from scipy import ndimage


@dataclasses.dataclass(frozen=True)
class Settings:
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)  # the cloud's rough up direction
    cell: float = 0.05  # metres per density-image cell
    band: tuple[float, float] = (1.0, 1.9)  # wall slice, metres above the floor
    low_band: tuple[float, float] = (0.2, 0.8)  # below a window's sill: wall here means a window, not a door
    exterior_wall: float = 0.2  # metres; the DSL's coordinates are wall centrelines
    partition_wall: float = 0.1
    min_wall: float = 0.4  # shortest wall run kept, metres
    min_opening: float = 0.5  # narrowest gap reported as a door or window
    max_opening: float = 2.5  # widest gap bridged as an opening rather than open space
    min_room_area: float = 1.0  # square metres
    floor_tolerance: float = 0.03  # metres from the floor plane that count as floor
    seed: int = 0


@dataclasses.dataclass(frozen=True)
class Room:
    id: str
    polygon: tuple[tuple[float, float], ...]  # plan metres, counter-clockwise, wall centrelines


@dataclasses.dataclass(frozen=True)
class Opening:
    kind: str  # "door" or "window"
    between: tuple[str, str]  # room ids, or "exterior" first
    at: tuple[float, float]  # plan metres, on the wall
    width: float


@dataclasses.dataclass(frozen=True)
class Floorplan:
    rooms: tuple[Room, ...]
    openings: tuple[Opening, ...]
    # Cloud to plan: (x, y, height above the floor) = transform @ (px, py, pz, 1). Ties the plan to the cloud's frame
    # (the doc's proposed anchor transform).
    transform: np.ndarray


@dataclasses.dataclass
class _Line:
    vertical: bool  # True: x = coord (a wall running along y); False: y = coord
    coord: float  # metres in the rotated frame
    spans: list[tuple[float, float]]  # wall extents along the line, metres, sorted, disjoint
    shift: float = 0.0  # added to coord on output (exterior inner face to centreline)


@dataclasses.dataclass(frozen=True)
class _Gap:
    line: int
    lo: float
    hi: float


EXTERIOR = -1


def _unit(v: Sequence[float] | np.ndarray) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    return a / np.linalg.norm(a)


FOOTPRINT_CANDIDATES = 32  # RANSAC's best-supported planes compared by the area their inliers cover
FOOTPRINT_CELL = 0.1  # metres
PARALLEL_DEG = 5.0  # a RANSAC plane from three noisy points can be a few degrees off; the refit settles it


def floor_plane(
    points: np.ndarray, up: Sequence[float], tolerance: float, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    """(n, d) of the floor, with n a unit normal pointing up and n . p = d on it. Raises ValueError when no plane
    near horizontal has support."""
    upv = _unit(up)
    sample = points[rng.choice(len(points), min(len(points), 20_000), replace=False)]
    planes: list[tuple[np.ndarray, float, int]] = []
    remaining = sample
    for _ in range(3):  # floor, ceiling, and one spare (a table top, a bed)
        found = _horizontal_ransac(remaining, upv, tolerance, rng)
        if found is None:
            break
        n, d, inliers = found
        planes.append((n, d, int(inliers.sum())))
        remaining = remaining[~inliers]
        if len(remaining) < 100:
            break
    if not planes:
        raise ValueError("no floor: no plane within 25 degrees of horizontal has enough points")
    # Three noisy points close together can span a plane tilted several degrees off the floor (or the ceiling) that
    # still cuts through enough of it to win, so each candidate is refitted to its inliers before any is compared: a
    # floor whose raw plane was 6 degrees off was once ruled not parallel to the ceiling and dropped.
    planes = [(*_refit(points, n, d, tolerance, upv), count) for n, d, count in planes]
    # The best-supported plane is the floor or the ceiling. A runner-up only competes when it is parallel to it
    # (within PARALLEL_DEG): a tilted plane cutting through the walls would otherwise win by being "lowest". The lowest
    # is the one lowest under the cloud's centre, along up: a plane's offset d is its height at the origin, which can
    # be metres from the room, where a few degrees of tilt moves it by a metre (a 4 degree slice through the
    # walls, 12 m from the origin, read 0.6 m lower than the floor and was taken for it).
    centre = sample.mean(axis=0)
    best_n, _, most = max(planes, key=lambda plane: plane[2])
    parallel = [
        (n, d) for n, d, count in planes if count >= 0.3 * most and n @ best_n >= math.cos(math.radians(PARALLEL_DEG))
    ]
    normal, d = min(parallel, key=lambda plane: (plane[1] - plane[0] @ centre) / (plane[0] @ upv))
    return normal, d


def _refit(
    points: np.ndarray, normal: np.ndarray, d: float, tolerance: float, upv: np.ndarray
) -> tuple[np.ndarray, float]:
    """A plane refitted to the points near it a few times (a least-squares fit, the band narrowing), so a RANSAC plane
    a few degrees off settles onto the surface it cut."""
    for widen in (2.0, 2.0, 1.0, 1.0, 1.0):
        near = points[np.abs(points @ normal - d) < tolerance * widen]
        if len(near) < 3:
            break
        centroid = near.mean(axis=0)
        normal = np.linalg.svd(near - centroid, full_matrices=False)[2][-1]
        normal = normal if normal @ upv > 0 else -normal
        d = float(normal @ centroid)
    return normal, d


def _horizontal_ransac(
    points: np.ndarray, up: np.ndarray, tolerance: float, rng: np.random.Generator
) -> tuple[np.ndarray, float, np.ndarray] | None:
    if len(points) < 50:
        return None
    a, b, c = (points[rng.integers(0, len(points), 2048)] for _ in range(3))
    normals = np.cross(b - a, c - a)
    length = np.linalg.norm(normals, axis=1)
    keep = length > 1e-9
    normals, a = normals[keep] / length[keep, None], a[keep]
    normals *= np.where(normals @ up < 0, -1.0, 1.0)[:, None]
    keep = normals @ up >= math.cos(math.radians(25))
    normals, a = normals[keep], a[keep]
    if not len(normals):
        return None
    offsets = np.einsum("ij,ij->i", normals, a)
    counts = (np.abs(points @ normals.T - offsets) < tolerance).sum(axis=0)
    if counts.max() < 50:
        return None
    # Of the best-supported candidates, the one whose inliers cover the most area: a floor or ceiling covers the
    # room, while a plane tilted a few degrees through it meets the floor in a strip and the walls in lines, and on a
    # sparsely seen floor can gather more raw inliers than the floor itself.
    best, widest = int(np.argmax(counts)), -1
    for i in np.argsort(counts)[::-1][:FOOTPRINT_CANDIDATES]:
        if counts[i] < max(50, 0.3 * counts.max()):
            break
        cells = _footprint(points[np.abs(points @ normals[i] - offsets[i]) < tolerance], normals[i])
        if cells > widest:
            best, widest = int(i), cells
    return normals[best], float(offsets[best]), np.abs(points @ normals[best] - offsets[best]) < tolerance


def _footprint(points: np.ndarray, normal: np.ndarray) -> int:
    """How many FOOTPRINT_CELL cells of the plane with `normal` the points cover."""
    e1 = _unit(np.cross(normal, [1.0, 0.0, 0.0] if abs(normal[0]) < 0.9 else [0.0, 1.0, 0.0]))
    e2 = np.cross(normal, e1)
    cells = np.floor(np.stack([points @ e1, points @ e2], axis=1) / FOOTPRINT_CELL).astype(np.int64)
    return len(np.unique(cells, axis=0))


def manhattan_angle(xy: np.ndarray, cell: float) -> float:
    """The rotation in [-pi/4, pi/4) that lines the walls in xy up with the axes: the one whose x and y histograms
    are sharpest (largest sum of squared bin counts). Of the four equivalent quarter turns it takes the smallest,
    so a cloud already square to its axes keeps its orientation."""

    def sharpness(theta: float) -> float:
        c, s = math.cos(theta), math.sin(theta)
        total = 0.0
        for w in (xy[:, 0] * c + xy[:, 1] * s, -xy[:, 0] * s + xy[:, 1] * c):
            counts = np.bincount(((w - w.min()) / cell).astype(np.int64))
            total += float(np.sum(counts.astype(np.float64) ** 2))
        return total

    coarse = max(np.radians(np.arange(0.0, 90.0, 1.0)), key=sharpness)
    fine = max(coarse + np.radians(np.arange(-1.0, 1.0001, 0.05)), key=sharpness)
    return float((fine + math.pi / 4) % (math.pi / 2) - math.pi / 4)


def _runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int, int]]:
    """(row, start, end) of each run of True along the rows of mask at least min_len long; end is exclusive."""
    padded = np.zeros((mask.shape[0], mask.shape[1] + 2), dtype=np.int8)
    padded[:, 1:-1] = mask
    step = np.diff(padded, axis=1)
    starts, ends = np.nonzero(step == 1), np.nonzero(step == -1)
    return [(int(r), int(s), int(e)) for r, s, e in zip(starts[0], starts[1], ends[1], strict=True) if e - s >= min_len]


def _merge_spans(spans: list[tuple[float, float]], join: float) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for lo, hi in sorted(spans):
        if merged and lo - merged[-1][1] <= join:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def _lines(mask: np.ndarray, origin: tuple[float, float], s: Settings, vertical: bool) -> list[_Line]:
    """Wall lines along one axis: runs of wall cells, grouped by nearby rows (columns when vertical)."""
    grid = mask.T if vertical else mask  # rows of `grid` run along the line
    across0, along0 = (origin[0], origin[1]) if vertical else (origin[1], origin[0])
    runs = sorted(_runs(grid, max(2, round(s.min_wall / s.cell))))
    groups: list[list[tuple[int, int, int]]] = []
    for run in runs:
        if groups and run[0] - groups[-1][-1][0] <= round(0.2 / s.cell):
            groups[-1].append(run)
        else:
            groups.append([run])
    lines = []
    for group in groups:
        weights = np.array([e - b for _, b, e in group], dtype=np.float64)
        rows = np.array([r for r, _, _ in group], dtype=np.float64)
        coord = across0 + (float(np.average(rows, weights=weights)) + 0.5) * s.cell
        spans = [(along0 + b * s.cell, along0 + e * s.cell) for _, b, e in group]
        lines.append(_Line(vertical, coord, _merge_spans(spans, 3 * s.cell)))
    # Lines of one wall that the row grouping kept apart (the runs didn't share rows) join when they are close.
    lines.sort(key=lambda line: line.coord)
    joined: list[_Line] = []
    for line in lines:
        if joined and line.coord - joined[-1].coord < 0.25:
            last = joined[-1]
            length = sum(hi - lo for lo, hi in line.spans)
            last_length = sum(hi - lo for lo, hi in last.spans)
            last.coord = (last.coord * last_length + line.coord * length) / (last_length + length)
            last.spans = _merge_spans(last.spans + line.spans, 3 * s.cell)
        else:
            joined.append(line)
    return joined


def _bridge(lines: list[_Line], s: Settings) -> list[_Gap]:
    """Close each line's gaps and its ends short of a perpendicular wall, up to max_opening, in place; returns the
    closed gaps at least min_opening wide."""
    gaps: list[_Gap] = []
    for i, line in enumerate(lines):
        bridged: list[tuple[float, float]] = []
        for lo, hi in line.spans:
            if bridged and lo - bridged[-1][1] <= s.max_opening:
                if lo - bridged[-1][1] >= s.min_opening:
                    gaps.append(_Gap(i, bridged[-1][1], lo))
                bridged[-1] = (bridged[-1][0], hi)
            else:
                bridged.append((lo, hi))
        line.spans = bridged
    reach = 0.2  # how far past its ends a perpendicular wall must run to stop this one
    for i, line in enumerate(lines):
        others = [o for o in lines if o.vertical != line.vertical]
        ends = []
        for lo, hi in line.spans:
            covering = [o.coord for o in others if any(a - reach <= line.coord <= b + reach for a, b in o.spans)]
            below = [c for c in covering if lo - s.max_opening <= c <= lo + reach]
            above = [c for c in covering if hi - reach <= c <= hi + s.max_opening]
            new_lo = max(below) if below else lo
            new_hi = min(above) if above else hi
            if lo - new_lo >= s.min_opening:
                gaps.append(_Gap(i, new_lo, lo))
            if new_hi - hi >= s.min_opening:
                gaps.append(_Gap(i, hi, new_hi))
            ends.append((min(lo, new_lo), max(hi, new_hi)))
        line.spans = _merge_spans(ends, 0.0)
    return gaps


def _to_cells(value: float, origin: float, cell: float) -> int:
    return int(math.floor((value - origin) / cell))


def _rasterize(lines: list[_Line], shape: tuple[int, int], origin: tuple[float, float], cell: float) -> np.ndarray:
    walls = np.zeros(shape, dtype=bool)  # rows are y, columns x
    for line in lines:
        across = _to_cells(line.coord, origin[0] if line.vertical else origin[1], cell)
        for lo, hi in line.spans:
            a = max(0, _to_cells(lo, origin[1] if line.vertical else origin[0], cell) - 1)
            b = _to_cells(hi, origin[1] if line.vertical else origin[0], cell) + 2
            if line.vertical:
                walls[a:b, max(0, across - 1) : across + 2] = True
            else:
                walls[max(0, across - 1) : across + 2, a:b] = True
    return walls


def _regions(walls: np.ndarray, min_cells: int) -> np.ndarray:
    """Label map: 0 on walls, EXTERIOR outside, 1.. for rooms (free regions not touching the border)."""
    labels, count = ndimage.label(~walls)
    border = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))) - {0}
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    out = np.zeros_like(labels)
    out[~walls] = EXTERIOR
    room = 0
    for label in range(1, count + 1):
        if label not in border and sizes[label] >= min_cells:
            room += 1
            out[labels == label] = room
    return out


def _label_at(labels: np.ndarray, x: float, y: float, origin: tuple[float, float], cell: float) -> int:
    col, row = _to_cells(x, origin[0], cell), _to_cells(y, origin[1], cell)
    if not (0 <= row < labels.shape[0] and 0 <= col < labels.shape[1]):
        return EXTERIOR
    return int(labels[row, col])


def _point(line: _Line, along: float, offset: float = 0.0) -> tuple[float, float]:
    """(x, y) at `along` on line, `offset` across it."""
    across = line.coord + offset
    return (across, along) if line.vertical else (along, across)


def _exterior_shifts(lines: list[_Line], labels: np.ndarray, origin: tuple[float, float], s: Settings) -> None:
    """Move each line with outside on one side and rooms on the other out by half an exterior wall."""
    for line in lines:
        sides: dict[float, list[int]] = {-0.3: [], 0.3: []}
        for lo, hi in line.spans:
            for along in np.arange(lo, hi, 0.1):
                for offset, seen in sides.items():
                    seen.append(_label_at(labels, *_point(line, float(along), offset), origin, s.cell))
        outside = {offset: np.mean(np.array(seen) == EXTERIOR) for offset, seen in sides.items() if seen}
        rooms = {offset: np.mean(np.array(seen) > 0) for offset, seen in sides.items() if seen}
        for offset, other in ((-0.3, 0.3), (0.3, -0.3)):
            if outside.get(offset, 0.0) >= 0.5 and rooms.get(other, 0.0) >= 0.5:
                line.shift = math.copysign(s.exterior_wall / 2, offset)


def cell_outline(cells: np.ndarray) -> list[tuple[int, int]]:
    """The outer boundary of a set of grid cells (cells[k, l]: x index k, y index l) as grid corners,
    counter-clockwise, collinear corners dropped."""
    nx, ny = cells.shape
    edges: dict[tuple[int, int], list[tuple[int, int]]] = {}

    def filled(k: int, j: int) -> bool:
        return 0 <= k < nx and 0 <= j < ny and bool(cells[k, j])

    for k, j in zip(*np.nonzero(cells), strict=True):
        k, j = int(k), int(j)
        if not filled(k, j - 1):
            edges.setdefault((k, j), []).append((k + 1, j))
        if not filled(k + 1, j):
            edges.setdefault((k + 1, j), []).append((k + 1, j + 1))
        if not filled(k, j + 1):
            edges.setdefault((k + 1, j + 1), []).append((k, j + 1))
        if not filled(k - 1, j):
            edges.setdefault((k, j + 1), []).append((k, j))
    loops: list[list[tuple[int, int]]] = []
    while edges:
        start = next(iter(edges))
        loop, here = [start], start
        while True:
            nexts = edges[here]
            there = nexts.pop()
            if not nexts:
                del edges[here]
            if there == start:
                break
            loop.append(there)
            here = there
        loops.append(loop)

    def area(loop: list[tuple[int, int]]) -> float:
        return 0.5 * sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(loop, loop[1:] + loop[:1], strict=True))

    loop = max(loops, key=area)
    corners = []
    for i, here in enumerate(loop):
        before, after = loop[i - 1], loop[(i + 1) % len(loop)]
        if (here[0] - before[0]) * (after[1] - here[1]) != (here[1] - before[1]) * (after[0] - here[0]):
            corners.append(here)
    return corners


def _opening_kind(floor_xyz: np.ndarray, line: _Line, gap: _Gap, s: Settings) -> str:
    """ "window" when the scan has wall below the gap (a sill), else "door"."""
    x, y, z = floor_xyz[:, 0], floor_xyz[:, 1], floor_xyz[:, 2]
    across, along = (x, y) if line.vertical else (y, x)
    low = (z >= s.low_band[0]) & (z <= s.low_band[1]) & (np.abs(across - line.coord) <= 0.15)
    lo, hi = gap.lo + s.cell, gap.hi - s.cell
    inside = along[low & (along >= lo) & (along <= hi)]
    bins = max(1, int((hi - lo) / s.cell))
    covered = np.count_nonzero(np.bincount(((inside - lo) / s.cell).astype(np.int64), minlength=bins)[:bins])
    return "window" if covered >= 0.5 * bins else "door"


WALK_PAD_M = 0.35  # how far past the outermost feet the walkable floor reaches: people keep off the walls
WALK_CLOSE_M = 0.6  # gaps in the walked area this narrow (between two paths) are walkable too
WALK_MIN_FEET = 2  # a cell needs this many feet in it to count: one is as likely a misdetection as a step
WALK_BLOCK_M = 0.5  # the walked area is squared off to blocks this size, so its outline runs along the axes
WALK_MIN_SHARE = 0.02  # a walked area needs this share of all the feet (and WALK_MIN_REGION of them) to be floor
WALK_MIN_REGION = 50
MIN_BAND_FOR_ANGLE = 500  # wall-band points enough to set the plan's angle; fewer, and the feet help set it


def _feet(points: np.ndarray, weights: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """The walked points (and their weights) in cells holding at least WALK_MIN_FEET footprints: a lone foot far from
    the rest (a false detection, a foot near the horizon projected metres away) is noise, not floor. A point's weight
    is how many footprints it stands for (spatial/footprints.py reads them as cells of the frame), so a cell is
    counted by its footprints, not its points."""
    if not len(points):
        return points, weights
    cells = np.floor(points[:, :2] / cell).astype(np.int64)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    keep = np.bincount(inverse, weights=weights)[inverse] >= WALK_MIN_FEET
    return points[keep], weights[keep]


def _walked(
    points: np.ndarray,
    weights: np.ndarray,
    origin: tuple[float, float],
    shape: tuple[int, int],
    cell: float,
    min_area: float,
) -> np.ndarray:
    """The walkable floor as a mask of the density image: cells people's feet were in, grown by WALK_PAD_M, with
    gaps up to WALK_CLOSE_M and enclosed holes (furniture they walked around) filled, areas smaller than `min_area`
    (square metres) or holding fewer feet than WALK_MIN_SHARE of them (or WALK_MIN_REGION) dropped, then squared off to WALK_BLOCK_M blocks (a block half covered is in), so its outline
    is axis-aligned runs the wall lines can follow."""
    mask = np.zeros(shape, dtype=bool)
    rows = ((points[:, 1] - origin[1]) / cell).astype(np.int64)
    cols = ((points[:, 0] - origin[0]) / cell).astype(np.int64)
    inside = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
    mask[rows[inside], cols[inside]] = True

    def square(half_m: float) -> np.ndarray:  # square, not round: a room's corners are corners
        r = max(1, round(half_m / cell))
        return np.ones((2 * r + 1, 2 * r + 1), dtype=bool)

    mask = ndimage.binary_closing(mask, structure=square(WALK_CLOSE_M / 2))
    mask = ndimage.binary_dilation(mask, structure=square(WALK_PAD_M))
    mask = ndimage.binary_fill_holes(mask)
    labels, count = ndimage.label(mask)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    feet = np.bincount(labels[rows[inside], cols[inside]], weights=weights[inside], minlength=count + 1)
    enough = max(WALK_MIN_REGION, WALK_MIN_SHARE * float(weights[inside].sum()))
    keep = np.flatnonzero((sizes >= min_area / cell**2) & (feet >= enough))
    mask = np.isin(labels, keep[keep > 0])
    block = max(1, round(WALK_BLOCK_M / cell))
    h, w = (shape[0] + block - 1) // block, (shape[1] + block - 1) // block
    padded = np.zeros((h * block, w * block), dtype=np.float64)
    padded[: shape[0], : shape[1]] = mask
    share = padded.reshape(h, block, w, block).mean(axis=(1, 3))
    covered = share >= 0.5
    # The grid sits wherever the image starts, so a room's corner block is covered by the product of its two edges'
    # shares: two edges each 70% in leave the corner 49% in, a notch. A block at least a fifth covered with a block
    # in beside it on both axes is that corner, so it's in too.
    beside = np.pad(covered, 1)
    across = beside[1:-1, :-2] | beside[1:-1, 2:]
    along = beside[:-2, 1:-1] | beside[2:, 1:-1]
    covered |= (share >= 0.2) & across & along
    return np.kron(covered, np.ones((block, block), dtype=bool))[: shape[0], : shape[1]].astype(bool)


def floorplan(
    points: np.ndarray,
    settings: Settings | None = None,
    *,
    floor: Sequence[float] | None = None,
    walkable: np.ndarray | None = None,
    walk_weights: np.ndarray | None = None,
) -> Floorplan:
    """Rooms and openings recovered from an (N, 3) point cloud in metres. Raises ValueError when there is no floor
    or no wall to work from.

    `floor` is a known floor plane (nx, ny, nz, d: a point's height above it is n . p + d, as memory/places.py keeps
    it), used instead of finding one. `walkable` is an (M, 3) array of points on the floor where people walked, in
    the cloud's frame (spatial/footprints.py): the outline of the floor they cover stands in for walls the cloud
    doesn't have, so a fixed camera's cloud, which sees mostly floor and furniture, still closes into rooms.
    `walk_weights` is how many footprints each walkable point stands for (one each when not given)."""
    s = settings or Settings()
    rng = np.random.default_rng(s.seed)
    points = np.asarray(points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    walk = np.zeros((0, 3)) if walkable is None else np.asarray(walkable, dtype=np.float64).reshape(-1, 3)
    weight = np.ones(len(walk)) if walk_weights is None else np.asarray(walk_weights, dtype=np.float64).reshape(-1)
    if len(weight) != len(walk):
        raise ValueError(f"{len(weight)} walk weights for {len(walk)} walkable points")
    finite = np.isfinite(walk).all(axis=1) & np.isfinite(weight)
    walk, weight = walk[finite], weight[finite]
    if len(points) < 100:
        raise ValueError(f"too few points for a floorplan ({len(points)})")
    if floor is not None:
        n = np.asarray(floor[:3], dtype=np.float64)
        length = float(np.linalg.norm(n))
        normal, offset = n / length, -float(floor[3]) / length
    else:
        normal, offset = floor_plane(points, s.up, s.floor_tolerance, rng)
    # The plan's x is the cloud's first axis not close to up, laid on the floor, so a cloud square to its axes
    # comes out in its own orientation.
    helper = next(axis for axis in np.eye(3) if abs(axis @ normal) < 0.9)
    e1 = _unit(helper - (helper @ normal) * normal)
    e2 = np.cross(normal, e1)
    flat = np.stack([points @ e1, points @ e2, points @ normal - offset], axis=1)
    band = flat[(flat[:, 2] >= s.band[0]) & (flat[:, 2] <= s.band[1])]
    walked, weight = _feet(np.stack([walk @ e1, walk @ e2, np.zeros(len(walk))], axis=1), weight, s.cell)
    if len(band) < 100 and weight.sum() < 100:
        raise ValueError(f"too few points in the wall band {s.band} m above the floor ({len(band)})")
    # The walls set the plan's angle when the band has enough of them; the feet's spread only when it doesn't. The
    # feet are sampled as points, not by weight: with footprints read as cells of the frame (spatial/footprints.py),
    # where people stood still counts once, so the angle follows the walked area's shape rather than its busiest spots.
    both = band if len(band) >= MIN_BAND_FOR_ANGLE else np.concatenate([band, walked])
    sample = both[rng.choice(len(both), min(len(both), 100_000), replace=False)]
    theta = manhattan_angle(sample[:, :2], s.cell)
    c, si = math.cos(theta), math.sin(theta)
    rotate = np.array([[c, si, 0.0], [-si, c, 0.0], [0.0, 0.0, 1.0]])
    flat = flat @ rotate.T
    band = band @ rotate.T
    walked = walked @ rotate.T
    extent = np.concatenate([band, walked])

    pad = s.max_opening + 1.0
    origin = (float(extent[:, 0].min()) - pad, float(extent[:, 1].min()) - pad)
    shape = (
        int((extent[:, 1].max() + pad - origin[1]) / s.cell) + 1,
        int((extent[:, 0].max() + pad - origin[0]) / s.cell) + 1,
    )
    # Where people walked: its outline is a wall for the lines below, wherever the cloud saw none.
    outline = np.zeros(shape, dtype=bool)
    if len(walked):
        floor_mask = _walked(walked, weight, origin, shape, s.cell, s.min_room_area)
        outline = floor_mask & ~ndimage.binary_erosion(floor_mask)
    density = np.zeros(shape, dtype=np.int64)
    np.add.at(
        density,
        (((band[:, 1] - origin[1]) / s.cell).astype(np.int64), ((band[:, 0] - origin[0]) / s.cell).astype(np.int64)),
        1,
    )
    # A wall face that sits on a cell boundary splits its points between two cells, and each can fall under the
    # threshold, leaving holes that read as openings. So each axis looks for its runs in the density summed over
    # three cells across that axis (rows for walls along x, columns for walls along y).
    lines = []
    for vertical in (False, True):
        across = ndimage.convolve1d(density, np.ones(3, dtype=np.int64), axis=1 if vertical else 0, mode="constant")
        occupied = across[across > 0]
        level = 0.25 * float(np.percentile(occupied, 90)) if len(occupied) else 0.0
        mask = across >= max(3.0, level)
        # A sparse stretch leaves one-cell breaks along a wall, which would cut its runs below min_wall and drop
        # them; close breaks up to 15 cm along the wall's own direction first.
        along = np.ones((5, 1) if vertical else (1, 5), dtype=bool)
        mask = ndimage.binary_closing(mask, structure=along) | outline
        lines += _lines(mask, origin, s, vertical=vertical)
    if not lines:
        raise ValueError("no wall lines in the wall band")
    gaps = _bridge(lines, s)
    labels = _regions(_rasterize(lines, shape, origin, s.cell), round(s.min_room_area / s.cell**2))
    _exterior_shifts(lines, labels, origin, s)

    xs = sorted({line.coord for line in lines if line.vertical})
    ys = sorted({line.coord for line in lines if not line.vertical})
    shift = {(line.vertical, line.coord): line.shift for line in lines}
    out_x = [x + shift[(True, x)] for x in xs]
    out_y = [y + shift[(False, y)] for y in ys]
    owner = np.zeros((max(0, len(xs) - 1), max(0, len(ys) - 1)), dtype=np.int64)
    for k in range(len(xs) - 1):
        cols = slice(_to_cells(xs[k], origin[0], s.cell), _to_cells(xs[k + 1], origin[0], s.cell) + 1)
        for j in range(len(ys) - 1):
            rows = slice(_to_cells(ys[j], origin[1], s.cell), _to_cells(ys[j + 1], origin[1], s.cell) + 1)
            block = labels[rows, cols]
            counts = np.bincount(block[block > 0].ravel(), minlength=1)
            if len(counts) > 1 and counts.max() >= 0.3 * block.size:
                owner[k, j] = int(np.argmax(counts))

    polygons: dict[int, list[tuple[float, float]]] = {}
    for label in range(1, int(labels.max()) + 1):
        cells = owner == label
        if cells.any():
            corners = cell_outline(cells)
            polygons[label] = [(out_x[k], out_y[j]) for k, j in corners]
    polygons = {
        label: poly for label, poly in polygons.items() if abs(_shoelace(poly)) >= s.min_room_area and len(poly) >= 4
    }
    if not polygons:
        raise ValueError("the walls enclose no room")
    min_x = min(x for poly in polygons.values() for x, _ in poly)
    min_y = min(y for poly in polygons.values() for _, y in poly)
    order = sorted(polygons, key=lambda label: (min(y for _, y in polygons[label]), min(x for x, _ in polygons[label])))
    ids = {label: f"r{i}" for i, label in enumerate(order, start=1)}
    rooms = tuple(
        Room(ids[label], tuple((round(x - min_x, 3), round(y - min_y, 3)) for x, y in polygons[label]))
        for label in order
    )

    openings = []
    for gap in gaps:
        line = lines[gap.line]
        mid = (gap.lo + gap.hi) / 2
        sides = [_label_at(labels, *_point(line, mid, offset), origin, s.cell) for offset in (-0.3, 0.3)]
        named = [ids.get(side, "exterior") if side != 0 else "" for side in sides]
        if "" in named or named[0] == named[1]:
            continue
        x, y = _point(line, mid, line.shift)
        at = (round(x - min_x, 3), round(y - min_y, 3))
        width = round((gap.hi - gap.lo) / 0.05) * 0.05
        if "exterior" in named:
            room = named[0] if named[1] == "exterior" else named[1]
            openings.append(Opening(_opening_kind(flat, line, gap, s), ("exterior", room), at, width))
        else:
            first, second = sorted(named)
            openings.append(Opening("door", (first, second), at, width))

    to_flat = np.vstack([e1, e2, normal])
    transform = np.zeros((3, 4))
    transform[:, :3] = rotate @ to_flat
    transform[:, 3] = -(rotate @ np.array([0.0, 0.0, offset])) - np.array([min_x, min_y, 0.0])
    return Floorplan(rooms, tuple(sorted(openings, key=lambda o: (o.kind, o.between, o.at))), transform)


def _shoelace(poly: Sequence[tuple[float, float]]) -> float:
    return 0.5 * sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(poly, [*poly[1:], poly[0]], strict=True))


def _num(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def to_dsl(plan: Floorplan, title: str = "Scan", s: Settings | None = None) -> str:
    """The plan in the houses floorplan library's line DSL. A room whose polygon is an axis-aligned rectangle is
    written `rect x,y WxH`, any other `poly`; openings are placed by `at x,y` on their wall."""
    s = s or Settings()
    title = title.replace('"', "'")
    lines = [f'plan "{title}" walls {_num(s.exterior_wall)}/{_num(s.partition_wall)}', ""]
    for i, room in enumerate(plan.rooms, start=1):
        xs = sorted({x for x, _ in room.polygon})
        ys = sorted({y for _, y in room.polygon})
        name = f'"Room {i}"'
        if len(room.polygon) == 4 and len(xs) == 2 and len(ys) == 2:
            shape = f"rect {_num(xs[0])},{_num(ys[0])} {_num(xs[1] - xs[0])}x{_num(ys[1] - ys[0])}"
        else:
            shape = "poly " + " ".join(f"{_num(x)},{_num(y)}" for x, y in room.polygon)
        lines.append(f"room {room.id} {name} other {shape}")
    if plan.openings:
        lines.append("")
    for opening in plan.openings:
        a, b = opening.between
        x, y = opening.at
        lines.append(f"{opening.kind} {a}>{b} at {_num(x)},{_num(y)} w{_num(opening.width)}")
    return "\n".join(lines) + "\n"


_ROOM = re.compile(r'^\s*room\s+(\S+)\s+"([^"]*)"\s+\S+\s+(rect|poly)\s+(.*)$')
_NUM = r"(-?\d+(?:\.\d+)?)"


def rooms_from_dsl(text: str) -> list[tuple[str, str, tuple[tuple[float, float], ...]]]:
    """Each room in a plan's DSL (`to_dsl`'s, or an owner's edit of it) as (id, name, polygon in plan metres), in order.
    Only `rect x,y WxH` and `poly x,y x,y ...` rooms are read; any other line, or a room it can't read, is skipped."""
    out = []
    for line in text.splitlines():
        m = _ROOM.match(line)
        if not m:
            continue
        room_id, name, shape, rest = m.groups()
        if shape == "rect":
            r = re.match(rf"{_NUM},{_NUM}\s+{_NUM}x{_NUM}", rest.strip())
            if not r:
                continue
            x, y, w, h = (float(g) for g in r.groups())
            out.append((room_id, name, ((x, y), (x + w, y), (x + w, y + h), (x, y + h))))
        else:
            pts = tuple((float(a), float(b)) for a, b in re.findall(rf"{_NUM},{_NUM}", rest))
            if len(pts) >= 3:
                out.append((room_id, name, pts))
    return out


def room_at(
    rooms: Sequence[tuple[str, str, Sequence[tuple[float, float]]]], x: float, y: float
) -> tuple[str, str] | None:
    """The (id, name) of the first room whose polygon holds plan point (x, y), or None (on a wall line counts as
    inside)."""
    for room_id, name, poly in rooms:
        inside = False
        for (ax, ay), (bx, by) in zip(poly, [*poly[1:], poly[0]], strict=True):
            if (ay > y) != (by > y) and x < ax + (y - ay) * (bx - ax) / (by - ay):
                inside = not inside
        on_edge = any(
            min(ax, bx) - 1e-9 <= x <= max(ax, bx) + 1e-9
            and min(ay, by) - 1e-9 <= y <= max(ay, by) + 1e-9
            and abs((bx - ax) * (y - ay) - (by - ay) * (x - ax)) < 1e-9
            for (ax, ay), (bx, by) in zip(poly, [*poly[1:], poly[0]], strict=True)
        )
        if inside or on_edge:
            return room_id, name
    return None


# Carrying an owner's plan into another plan frame: when two places are joined, the merged-away place's
# corrected plan is moved into the joined place's plan frame, statement by statement.

_TOKEN = re.compile(r'"(?:[^"\\]|\\.)*"|\S+')
_PAIR = re.compile(rf"^{_NUM},{_NUM}$")
_SIZE = re.compile(rf"^{_NUM}x{_NUM}$")
# The compass on a plan's page: x grows east, y grows south.
_SIDES = {"east": (1.0, 0.0), "south": (0.0, 1.0), "west": (-1.0, 0.0), "north": (0.0, -1.0)}
_SPACES = ("room", "outdoor", "void")
_OPENINGS = ("door", "window", "cased")
_VERTICAL = ("stairs", "lift", "ramp", "vertical")
_HEADER = ("plan", "walls", "north", "level", "x-anchor", "x-revision", "x-history")  # the target's own holds
_UNMOVABLE = ("grid", "layout")  # a track grid and its picture can't be turned


@dataclasses.dataclass(frozen=True)
class Similarity:
    """A plan-to-plan move: points turned by `turn` (radians, from +x toward +y, so clockwise on the page), scaled by
    `scale`, then shifted by `shift`."""

    scale: float
    turn: float
    shift: tuple[float, float]

    def point(self, x: float, y: float) -> tuple[float, float]:
        c, s = math.cos(self.turn), math.sin(self.turn)
        return (self.scale * (c * x - s * y) + self.shift[0], self.scale * (s * x + c * y) + self.shift[1])

    @property
    def quarters(self) -> int | None:
        """How many quarter turns `turn` is, or None when it isn't a whole number of them."""
        q = self.turn / (math.pi / 2)
        return round(q) % 4 if abs(q - round(q)) < 1e-6 else None

    def side(self, name: str) -> str | None:
        """A wall side (north, south, east, west) after the turn; None when the turn isn't a quarter."""
        if self.quarters is None or name not in _SIDES:
            return None
        x, y = _SIDES[name]
        c, s = math.cos(self.turn), math.sin(self.turn)
        turned = (c * x - s * y, s * x + c * y)
        return max(_SIDES, key=lambda k: _SIDES[k][0] * turned[0] + _SIDES[k][1] * turned[1])

    def snapped(self, about: tuple[float, float], within_deg: float) -> Similarity:
        """The move with its turn rounded to the nearest quarter, when that's within `within_deg`, keeping where `about`
        lands; else as it is."""
        quarter = round(self.turn / (math.pi / 2)) * (math.pi / 2)
        if abs(math.degrees(self.turn - quarter)) > within_deg:
            return self
        there = self.point(*about)
        c, s = math.cos(quarter), math.sin(quarter)
        x, y = about
        shift = (there[0] - self.scale * (c * x - s * y), there[1] - self.scale * (s * x + c * y))
        return Similarity(self.scale, quarter, shift)


@dataclasses.dataclass(frozen=True)
class Carried:
    """A plan's statements moved into another frame (`carry`)."""

    lines: tuple[str, ...]  # the moved statements, their ids renamed
    renamed: dict[str, str]  # each id the plan defined, and its name in the moved lines
    kept_as_comments: int  # statements that couldn't be moved, kept as `#` comments so nothing taught is lost


def _pt(x: float, y: float) -> str:
    return f"{_num(x)},{_num(y)}"


def _ids(lines: Sequence[str]) -> set[str]:
    """The ids a plan's statements define: its spaces, vertical elements, unexplored areas, objects, and the `id:` of
    openings and fixtures."""
    found: set[str] = set()
    for line in lines:
        words = _TOKEN.findall(line)
        if not words:
            continue
        head = words[0]
        if head in (*_SPACES, "stairs", "lift", "ramp", "vertical", "x-unexplored", "x-object") and len(words) > 1:
            found.add(words[1])
        found |= {w[3:] for w in words if w.startswith("id:")}
    return found - {"exterior"}


def plan_ids(text: str) -> set[str]:
    """The ids a plan's DSL defines (`_ids`)."""
    return _ids(text.splitlines())


def _ref(word: str, rename: Mapping[str, str], move: Similarity) -> str | None:
    """A space reference (`id` or `id.side`) renamed and its side turned; None when the side can't be turned."""
    space, _, side = word.partition(".")
    space = rename.get(space, space)
    if not side:
        return space
    turned = move.side(side)
    return None if turned is None else f"{space}.{turned}"


def _poly(points: Sequence[tuple[float, float]], move: Similarity) -> list[str]:
    return ["poly", *(_pt(*move.point(x, y)) for x, y in points)]


def _scaled(key: str, head: str, before: Sequence[str]) -> bool:
    """Whether a `<key><number>` word is a length: an opening's width or position, an arc's radius (just after its
    end point), or a fixture's depth."""
    if key == "r":
        return len(before) >= 2 and before[-2] == "arc"
    if key == "depth:":
        return head == "fixture"
    return head in _OPENINGS


def _box(words: list[str], move: Similarity) -> tuple[list[str], int] | None:
    """A fixture's `at x,y size wxh` box (x, y its top-left corner) moved, put first, and how many words it takes: still
    a box, by its new top-left corner, under a quarter turn; else its four corners as a `poly`. The words as they are,
    and 0, when there's no box; None for a size with no corner."""
    at = next((i for i in range(len(words) - 1) if words[i] == "at" and _PAIR.match(words[i + 1])), None)
    size = next((i for i in range(len(words) - 1) if words[i] == "size" and _SIZE.match(words[i + 1])), None)
    if size is None:
        return words, 0
    if at is None:
        return None
    x, y = (float(v) for v in words[at + 1].split(","))
    w, h = (float(v) for v in words[size + 1].split("x"))
    corners = [move.point(px, py) for px, py in ((x, y), (x + w, y), (x + w, y + h), (x, y + h))]
    rest = [word for i, word in enumerate(words) if i not in (at, at + 1, size, size + 1)]
    if move.quarters is None:
        box = ["poly", *(_pt(*c) for c in corners)]
    else:
        lo = (min(c[0] for c in corners), min(c[1] for c in corners))
        hi = (max(c[0] for c in corners), max(c[1] for c in corners))
        box = ["at", _pt(*lo), "size", f"{_num(hi[0] - lo[0])}x{_num(hi[1] - lo[1])}"]
    return [*box, *rest], len(box)


def _move_words(words: list[str], move: Similarity, rename: Mapping[str, str]) -> list[str] | None:
    """One statement's words moved; None when something in it can't be."""
    head = words[0]
    if head == "fixture" and len(words) > 2:
        boxed = _box(words[2:], move)
        if boxed is None:
            return None
        rest, n = boxed
        moved = _move_words([head, words[1], *rest[n:]], move, rename) if n else None
        if moved is None:
            return None if n else _each_word(words, move, rename)
        spot = next((i + 1 for i, w in enumerate(moved) if w.startswith("in:")), 2)
        return [*moved[:spot], *rest[:n], *moved[spot:]]
    return _each_word(words, move, rename)


def _each_word(words: list[str], move: Similarity, rename: Mapping[str, str]) -> list[str] | None:
    """`_move_words` word by word: positions, lengths, turns, and references."""
    head = words[0]
    out: list[str] = []
    i = 0
    while i < len(words):
        w = words[i]
        nxt = words[i + 1] if i + 1 < len(words) else ""
        if w == "rect" and _PAIR.match(nxt) and i + 2 < len(words) and _SIZE.match(words[i + 2]):
            x, y = (float(v) for v in nxt.split(","))
            sw, sh = (float(v) for v in words[i + 2].split("x"))
            corners = [move.point(px, py) for px, py in ((x, y), (x + sw, y), (x + sw, y + sh), (x, y + sh))]
            if move.quarters is None:
                out += ["poly", *(_pt(*c) for c in corners)]
            else:
                lo = (min(c[0] for c in corners), min(c[1] for c in corners))
                hi = (max(c[0] for c in corners), max(c[1] for c in corners))
                out += ["rect", _pt(*lo), f"{_num(hi[0] - lo[0])}x{_num(hi[1] - lo[1])}"]
            i += 3
            continue
        if w in ("at", "near") and _PAIR.match(nxt):
            out += [w, _pt(*move.point(*(float(v) for v in nxt.split(","))))]
            i += 2
            continue
        if w.startswith(("at:", "near:")) and _PAIR.match(w.partition(":")[2]):
            key, _, pair = w.partition(":")
            out.append(f"{key}:{_pt(*move.point(*(float(v) for v in pair.split(','))))}")
            i += 1
            continue
        if _PAIR.match(w):  # a poly's corner, or an arc's end
            out.append(_pt(*move.point(*(float(v) for v in w.split(",")))))
            i += 1
            continue
        length = re.fullmatch(r"(w|r|@-|@|depth:)(\d+(?:\.\d+)?)", w)
        if length and _scaled(length.group(1), head, out):
            out.append(f"{length.group(1)}{_num(float(length.group(2)) * move.scale)}")
            i += 1
            continue
        if w in ("thickness",) and re.fullmatch(_NUM, nxt):
            out += [w, _num(float(nxt) * move.scale)]
            i += 2
            continue
        if w in ("yaw",) and re.fullmatch(_NUM, nxt):  # degrees clockwise on the page, as the turn is
            out += [w, _num((float(nxt) + math.degrees(move.turn)) % 360)]
            i += 2
            continue
        if w.startswith("up:") and re.fullmatch(_NUM, w[3:]):
            out.append(f"up:{_num((float(w[3:]) + math.degrees(move.turn)) % 360)}")
            i += 1
            continue
        for key in ("in:", "swing:", "id:", "on:"):
            if w.startswith(key):
                ref = _ref(w[len(key) :], rename, move)
                if ref is None:
                    return None
                out.append(key + ref)
                break
        else:
            out.append(w)
        i += 1
    return out


def carry(text: str, move: Similarity, prefix: str, title: str = "") -> Carried:
    """An owner's plan (DSL text) moved into another plan's frame by `move`, ready to be added to that plan: every id
    it defines renamed `<prefix>_<id>` (so it can't clash), every position and length moved, wall sides and bearings
    turned. The header (plan, walls, north, level) and the frame's own `x-anchor` and revisions are left out, since
    the target plan's hold. A statement that can't be moved (a track grid or its layout picture; a wall side, or a box
    given by its size, under a turn that isn't a quarter) is kept as a comment naming `title`, so the owner still sees
    what they taught."""
    source = text.splitlines()
    rename = {i: f"{prefix}_{i}" for i in sorted(_ids(source))}
    out: list[str] = []
    commented = 0
    unmovable = False  # inside a layout picture's indented rows
    for line in source:
        words = _TOKEN.findall(line)
        indented = line[:1].isspace()
        if not words or words[0].startswith("#"):
            if words:
                out.append(line.strip())
            unmovable = False if not indented else unmovable
            continue
        head = words[0]
        if not indented:
            unmovable = head in _UNMOVABLE
        if head in _HEADER and not indented:
            continue
        moved = None if unmovable else _statement(words, move, rename)
        if moved is None:
            out.append(f"# from {title or prefix}: {line.strip()}")
            commented += 1
        else:
            out.append(("  " if indented else "") + " ".join(moved))
    return Carried(tuple(out), rename, commented)


def _statement(words: list[str], move: Similarity, rename: Mapping[str, str]) -> list[str] | None:
    head = words[0]
    if head in (*_SPACES, *_VERTICAL, "x-unexplored", "x-object") and len(words) > 1:
        words = [head, rename.get(words[1], words[1]), *words[2:]]
    elif head in _OPENINGS and len(words) > 1:
        between = words[1]
        if ">" in between:
            a, _, b = between.partition(">")
            words = [head, f"{rename.get(a, a)}>{rename.get(b, b)}", *words[2:]]
        else:
            ref = _ref(between, rename, move)
            if ref is None:
                return None
            words = [head, ref, *words[2:]]
    elif head == "x-source" and len(words) > 1:
        kind, _, ident = words[1].partition(":")
        words = [head, f"{kind}:{rename.get(ident, ident)}", *words[2:]]
    elif head not in ("fixture", "x-camera", "at"):
        return None  # a statement this doesn't know how to move
    return _move_words(words, move, rename)


def without(text: str, ids: Iterable[str], commented: str | None = None) -> str:
    """A plan's DSL with the spaces `ids` taken out, and every statement that names one of them (an opening into it, a
    fixture in it, its provenance) with them; with `commented`, each is kept as a `# <commented>: ...` comment
    instead, as `carry` keeps what it can't move."""
    gone = set(ids)
    kept = []
    for line in text.splitlines():
        words = _TOKEN.findall(line)
        named: set[str] = set()
        if words and words[0] in (*_SPACES, "x-unexplored") and len(words) > 1:
            named.add(words[1])
        if words and words[0] in _OPENINGS and len(words) > 1:
            named |= {part.partition(".")[0] for part in words[1].split(">")}
        if words and words[0] == "x-source" and len(words) > 1:
            named.add(words[1].partition(":")[2])
        named |= {w.partition(":")[2].partition(".")[0] for w in words if w.startswith(("in:", "on:", "swing:"))}
        if not named & gone:
            kept.append(line)
        elif commented is not None:
            kept.append(f"# {commented}: {line.strip()}")
    return "\n".join(kept) + ("\n" if text.endswith("\n") else "")


def covered(
    poly: Sequence[tuple[float, float]], by: Sequence[tuple[str, str, Sequence[tuple[float, float]]]], step: float = 0.1
) -> float:
    """The share of `poly`'s area (plan metres) that lies in the rooms `by` (`rooms_from_dsl`'s), sampled every
    `step` metres; 0 for a polygon with no sample inside it."""
    xs, ys = [x for x, _ in poly], [y for _, y in poly]
    own = [("p", "", poly)]
    inside = hit = 0
    for x in np.arange(min(xs) + step / 2, max(xs), step):
        for y in np.arange(min(ys) + step / 2, max(ys), step):
            if room_at(own, float(x), float(y)) is None:
                continue
            inside += 1
            hit += room_at(by, float(x), float(y)) is not None
    return hit / inside if inside else 0.0
