"""An owner-supplied floorplan, registered into a place's plan frame (part of spatial memory).

An owner may have an accurate plan of their home from elsewhere, such as a phone LiDAR scan app's export redrawn in
the floorplan DSL. It becomes the place's owner plan (memory/places.py `correct_plan`), so positions name its rooms,
and it is ground truth to measure the map and the cameras' registrations against. `tools/import_owner_plan.py` runs
it against a live place; this module is the pure part: no I/O.

**Reading** (`parse`): the subset of the DSL a plan of one storey needs, as the floorplan library's docs give it:
`plan "<title>" walls <exterior>/<partition>`, `room <id> "<name>" <kind> rect x,y WxH | poly x,y ...`, openings
(`door`, `window`, `cased`: `<a>><b> at x,y w<width> ...`), fixtures (`fixture <kind> in:<room> at x,y size WxD ...`),
and owner-measured camera spots (`x-camera <id> at x,y [height m] [yaw deg] [fov deg]`, as the Places tab's DSL
extension spells them). Comments and `level` lines are dropped (a comment may hold the home's address, which has no
place in owner state); any other statement is counted in `Parsed.skipped`. Coordinates are wall centrelines in
metres, with x growing east and y growing south on the page.

**Registering** (`register`): a rigid move of the plan onto the map's plan frame: a turn, a shift, and the mirror the
page needs when the map's frame is right-handed seen from above (the DSL's y grows south), both tried. Scale is not
fitted: both are metric, so a free scale would only soak up error. It is measured by a fit that frees it (`Fit.scale`)
for the report instead (`_free_scale`). The steps:

1. **Coarse.** For each mirror and each whole degree, the map's wall band (points 1.0 to 1.9 m above the floor) is
   turned into the plan's frame and counted into a grid of COARSE_CELL cells, and its cross-correlation with a score
   that falls off with the distance from the nearest plan wall (FFT) gives every shift at once. Each camera the owner
   said sees a room (`seen`) must land in that room, so only shifts where all of them do count; with two or more
   owner-measured camera spots (`anchors`) instead, the turn and shift come from the spots (`_procrustes`). A spot its
   camera lands over ANCHOR_OUTLIER from after the fine fit is that camera's registration gone wrong: it is left out
   and the fit run again (`Fit.dropped`), while three or more spots remain.
2. **Fine.** Trimmed ICP: each band point paired with the nearest point on a plan wall's nearer face (`faces`: half
   its thickness from the centreline, as a scan sees it; pairing with centrelines pulled the fit by centimetres where
   walls differ in thickness), pairs farther than the trim (TRIMS, shrinking, PASSES at each) dropped, and the
   weighted rigid fit solved again. Anchors join as pairs whose weights add up to `anchor_share` of the whole, so
   measured spots hold the fit while the walls settle what they leave free.

The report (`Fit`): the band points' distances to the nearest wall's face (median and 90th percentile of those
within REPORT_NEAR of its centreline, so a couch or the yard doesn't count as error), how many lie within NEAR_FACE of
a face, which room each camera lands in, and each anchor's residual.

**Writing** (`to_dsl`): the plan moved into the target frame, every room a `poly` (a turn that isn't a quarter makes
a rectangle a polygon), openings and fixtures moved with it, a camera spot's yaw turned, and the title replaced.

**A camera's view** (`view_fit`): its own points in the map (the part of it the camera sees) moved to its measured spot and
turned about it, scored by how many it could see from there on the plan (in a room, with no wall but a door or
opening between, `in_sight`; a share inside any room let a view pass through the house's inner walls) and on its
walls. The best turn is the view's own evidence of which way the camera looks, beside its registered yaw; the
same fit against the view reflected across its line of sight (`mirrored`) says whether its video is mirrored, which
makes its reconstruction a reflection no turn undoes. Fixed cameras mostly look down at floor and furniture, so the
evidence is often broad (`View.spread`). On Place 1 it agreed with the registration within 2 degrees for
one camera and showed a 4 m misregistration as a view that fit no better at the measured spot, so it is a check on a
registration, not a replacement for one.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import re
from collections.abc import Mapping, Sequence

import numpy as np
from scipy import signal

from liberational.spatial import floorplan

BAND = (1.0, 1.9)  # wall slice, metres above the floor (floorplan.Settings.band)
COARSE_CELL = 0.1  # metres per cell of the coarse search's grids
COARSE_SIGMA = 0.15  # metres: how fast the coarse score falls off from a wall
COARSE_STEP_DEG = 1.0
MARGIN = 2.0  # metres around the plan's walls in the coarse score grid
MAX_BAND_POINTS = 40_000  # band points the fit uses, evenly sampled
TRIMS = (0.5, 0.3, 0.2, 0.15, 0.1)  # ICP: pair distance kept, metres, shrinking
PASSES = 5  # ICP passes at each trim
REPORT_NEAR = 0.5  # metres: band points this close to a wall centreline are wall points in the residual stats
NEAR_FACE = 0.1  # metres from a wall's face that count as on it
# metres: a camera this far from its measured spot after the fit is registered wrong, not measured wrong (the owner's
# tape is good to centimetres), so its spot is left out and the fit run again, while three or more remain
ANCHOR_OUTLIER = 1.0

_TOKEN = re.compile(r'"(?:[^"\\]|\\.)*"|\S+')
_NUM = r"-?\d+(?:\.\d+)?"
_PAIR = re.compile(rf"^({_NUM}),({_NUM})$")
_SIZE = re.compile(rf"^({_NUM})x({_NUM})$")
_OPENINGS = ("door", "window", "cased")

Point = tuple[float, float]


@dataclasses.dataclass(frozen=True)
class Room:
    id: str
    name: str
    kind: str
    polygon: tuple[Point, ...]


@dataclasses.dataclass(frozen=True)
class Opening:
    kind: str  # door, window, or cased
    between: str  # "a>b" ("exterior>a" for one in an outside wall)
    at: Point
    width: float
    rest: tuple[str, ...]  # the words after the width (entrance, swing:<room>, ...), as written


@dataclasses.dataclass(frozen=True)
class Fixture:
    kind: str
    room: str | None  # its in:<room>
    corners: tuple[Point, ...]  # its box (`at` top-left, `size`) as four corners, or its `poly`
    rest: tuple[str, ...]  # the other words (a label), as written


@dataclasses.dataclass(frozen=True)
class Camera:
    """An owner-measured camera spot (`x-camera`)."""

    id: str
    at: Point
    height: float | None = None  # metres above the floor
    yaw: float | None = None  # degrees clockwise from north (up the page), which way it looks
    fov: float | None = None


@dataclasses.dataclass(frozen=True)
class Parsed:
    title: str
    walls: tuple[float, float]  # exterior and partition thickness, metres
    rooms: tuple[Room, ...]
    openings: tuple[Opening, ...]
    fixtures: tuple[Fixture, ...]
    cameras: tuple[Camera, ...]
    skipped: int  # statements this reader doesn't know, left out


class PlanError(ValueError):
    """A plan this reader can't use, with the line it stopped at."""


def _pair(word: str, line: int) -> Point:
    m = _PAIR.match(word)
    if not m:
        raise PlanError(f"line {line}: expected x,y, got {word!r}")
    return float(m.group(1)), float(m.group(2))


def _box(at: Point, size: str, line: int) -> tuple[Point, ...]:
    m = _SIZE.match(size)
    if not m:
        raise PlanError(f"line {line}: expected WxH, got {size!r}")
    x, y = at
    w, h = float(m.group(1)), float(m.group(2))
    return ((x, y), (x + w, y), (x + w, y + h), (x, y + h))


def _options(words: Sequence[str], line: int, keys: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    i = 0
    while i < len(words):
        if words[i] not in keys or i + 1 >= len(words):
            raise PlanError(f"line {line}: unexpected {words[i]!r}")
        out[words[i]] = words[i + 1]
        i += 2
    return out


def parse(text: str) -> Parsed:
    """The plan in `text` (floorplan DSL). Raises PlanError for a statement it knows but can't read, or a plan with no
    rooms."""
    title, walls = "", (0.2, 0.1)
    rooms: list[Room] = []
    openings: list[Opening] = []
    fixtures: list[Fixture] = []
    cameras: list[Camera] = []
    skipped = 0
    for n, raw in enumerate(text.splitlines(), start=1):
        words = _TOKEN.findall(raw)
        cut = next((i for i, w in enumerate(words) if w.startswith("#")), len(words))
        words = words[:cut]  # a comment, whole-line or trailing
        if not words:
            continue
        head = words[0]
        if raw[:1].isspace():
            skipped += 1  # an indented row belongs to a statement this reader skips (a layout picture)
        elif head == "plan":
            title = words[1].strip('"') if len(words) > 1 else ""
            if "walls" in words:
                ext, _, part = words[words.index("walls") + 1].partition("/")
                walls = (float(ext), float(part or ext))
        elif head == "level":
            continue
        elif head == "room":
            if len(words) < 5:
                raise PlanError(f'line {n}: room <id> "<name>" <kind> rect|poly ...')
            rid, name, kind, shape = words[1], words[2].strip('"'), words[3], words[4]
            if shape == "rect" and len(words) >= 7:
                poly = _box(_pair(words[5], n), words[6], n)
            elif shape == "poly":
                poly = tuple(_pair(w, n) for w in words[5:] if _PAIR.match(w))
            else:
                raise PlanError(f"line {n}: a room is rect x,y WxH or poly x,y ...")
            if len(poly) < 3:
                raise PlanError(f"line {n}: room {rid} has fewer than three corners")
            rooms.append(Room(rid, name, kind, poly))
        elif head in _OPENINGS:
            if len(words) < 5 or words[2] != "at" or not re.fullmatch(rf"w{_NUM}", words[4]):
                raise PlanError(f"line {n}: {head} <a>><b> at x,y w<width> ...")
            openings.append(Opening(head, words[1], _pair(words[3], n), float(words[4][1:]), tuple(words[5:])))
        elif head == "fixture":
            room = next((w[3:] for w in words if w.startswith("in:")), None)
            rest = [w for w in words[2:] if not w.startswith("in:")]
            if "poly" in rest:
                i = rest.index("poly")
                corners = tuple(_pair(w, n) for w in rest[i + 1 :] if _PAIR.match(w))
                rest = rest[:i] + [w for w in rest[i + 1 :] if not _PAIR.match(w)]
            elif "at" in rest and "size" in rest:
                i, j = rest.index("at"), rest.index("size")
                corners = _box(_pair(rest[i + 1], n), rest[j + 1], n)
                rest = [w for k, w in enumerate(rest) if k not in (i, i + 1, j, j + 1)]
            else:
                skipped += 1
                continue
            fixtures.append(Fixture(words[1], room, corners, tuple(rest)))
        elif head == "x-camera":
            if len(words) < 2:
                raise PlanError(f"line {n}: x-camera <id> at x,y ...")
            o = _options(words[2:], n, ("at", "height", "yaw", "fov"))
            if "at" not in o:
                raise PlanError(f"line {n}: x-camera {words[1]} has no at x,y")
            num = {k: float(o[k]) for k in ("height", "yaw", "fov") if k in o}
            cameras.append(Camera(words[1], _pair(o["at"], n), num.get("height"), num.get("yaw"), num.get("fov")))
        else:
            skipped += 1
    if not rooms:
        raise PlanError("the plan has no rooms")
    return Parsed(title, walls, tuple(rooms), tuple(openings), tuple(fixtures), tuple(cameras), skipped)


def spots(text: str) -> tuple[Camera, ...]:
    """The owner-measured camera spots (`x-camera`) in a plan's DSL, whatever else it holds: a reader that wants only
    the spots takes them from a plan `parse` would refuse (one with no rooms yet). A spot line it can't read is left
    out; the later of two lines for one camera wins."""
    found: dict[str, Camera] = {}
    for n, raw in enumerate(text.splitlines(), start=1):
        words = _TOKEN.findall(raw)
        words = words[: next((i for i, w in enumerate(words) if w.startswith("#")), len(words))]
        if len(words) < 2 or words[0] != "x-camera" or raw[:1].isspace():
            continue
        try:
            o = _options(words[2:], n, ("at", "height", "yaw", "fov"))
            num = {k: float(o[k]) for k in ("height", "yaw", "fov") if k in o}
            found[words[1]] = Camera(words[1], _pair(o["at"], n), num.get("height"), num.get("yaw"), num.get("fov"))
        except KeyError, ValueError:  # PlanError is a ValueError
            continue
    return tuple(found.values())


# Walls: each room's polygon edges, a wall shared by two rooms counted once.


@dataclasses.dataclass(frozen=True)
class Walls:
    a: np.ndarray  # (S, 2) segment starts, plan metres
    b: np.ndarray  # (S, 2) segment ends
    half: np.ndarray  # (S,) half each wall's thickness: exterior or partition


def walls(plan: Parsed) -> Walls:
    """The plan's wall centrelines as segments: every room edge, with an edge two rooms share (or overlap along) cut
    into pieces so each stretch is one wall, exterior where only one room has it."""
    edges: list[tuple[Point, Point]] = []
    for room in plan.rooms:
        poly = room.polygon
        edges += [(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))]
    pieces: dict[tuple[Point, Point], int] = {}
    for a, b in edges:
        # cut the edge at every corner of another edge that lies on it, so shared stretches line up
        ax, ay = a
        dx, dy = b[0] - ax, b[1] - ay
        length2 = dx * dx + dy * dy
        if length2 < 1e-12:
            continue
        cuts = {0.0, 1.0}
        for c, d in edges:
            for px, py in (c, d):
                t = ((px - ax) * dx + (py - ay) * dy) / length2
                if 1e-9 < t < 1 - 1e-9 and abs((px - ax) * dy - (py - ay) * dx) / math.sqrt(length2) < 1e-6:
                    cuts.add(t)
        ts = sorted(cuts)
        for t0, t1 in itertools.pairwise(ts):
            p = (round(ax + t0 * dx, 6), round(ay + t0 * dy, 6))
            q = (round(ax + t1 * dx, 6), round(ay + t1 * dy, 6))
            key = (min(p, q), max(p, q))
            pieces[key] = pieces.get(key, 0) + 1
    ext, part = plan.walls
    keys = sorted(pieces)
    a = np.array([k[0] for k in keys], dtype=np.float64).reshape(-1, 2)
    b = np.array([k[1] for k in keys], dtype=np.float64).reshape(-1, 2)
    half = np.array([(ext if pieces[k] == 1 else part) / 2 for k in keys], dtype=np.float64)
    return Walls(a, b, half)


def nearest(points: np.ndarray, w: Walls) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each (N, 2) point: its distance to the nearest wall centreline, that nearest point, and the wall's index."""
    best_d = np.full(len(points), np.inf)
    best_q = np.zeros_like(points)
    best_i = np.zeros(len(points), dtype=np.int64)
    for i in range(len(w.a)):
        a, b = w.a[i], w.b[i]
        ab = b - a
        t = np.clip(((points - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
        q = a + t[:, None] * ab
        d = np.hypot(*(points - q).T)
        closer = d < best_d
        best_d[closer], best_q[closer], best_i[closer] = d[closer], q[closer], i
    return best_d, best_q, best_i


def faces(points: np.ndarray, w: Walls) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each (N, 2) point: its distance to the nearer face of the nearest wall (half the wall's thickness either
    side of its centreline: a scan sees faces, and an outside wall only its inner one), that point of the face, and
    its distance to the centreline."""
    d, q, i = nearest(points, w)
    unit = (points - q) / np.maximum(d, 1e-9)[:, None]
    return np.abs(d - w.half[i]), q + unit * w.half[i][:, None], d


# The move: plan page coordinates q to the target frame p = R(turn) F q + shift, F mirroring y when `mirror`.


@dataclasses.dataclass(frozen=True)
class Move:
    turn: float  # radians, counterclockwise in the target frame
    shift: Point
    mirror: bool  # the plan's y flips (its page's y grows south; a right-handed frame's grows north)

    @property
    def matrix(self) -> np.ndarray:
        c, s = math.cos(self.turn), math.sin(self.turn)
        f = -1.0 if self.mirror else 1.0
        return np.array([[c, -s * f], [s, c * f]])

    def points(self, q: np.ndarray) -> np.ndarray:
        return np.asarray(q, dtype=np.float64) @ self.matrix.T + np.asarray(self.shift)

    def point(self, x: float, y: float) -> Point:
        p = self.points(np.array([[x, y]]))[0]
        return float(p[0]), float(p[1])

    def inverse(self, p: np.ndarray) -> np.ndarray:
        """Target-frame points back onto the plan's page."""
        return (np.asarray(p, dtype=np.float64) - np.asarray(self.shift)) @ self.matrix  # matrix is orthogonal

    def yaw(self, degrees: float) -> float:
        """A page bearing (degrees clockwise from up the page) moved: the same direction, as a bearing on the moved
        plan's page."""
        r = math.radians(degrees)
        dx, dy = self.points(np.array([[math.sin(r), -math.cos(r)]]))[0] - np.asarray(self.shift)
        return math.degrees(math.atan2(dx, -dy)) % 360


def _procrustes(q: np.ndarray, p: np.ndarray, weights: np.ndarray, mirror: bool) -> Move:
    """The weighted rigid move (with the mirror given) that best takes plan points q onto target points p."""
    f = np.array([1.0, -1.0 if mirror else 1.0])
    qf = q * f
    w = weights / weights.sum()
    qc, pc = w @ qf, w @ p
    a, b = qf - qc, p - pc
    # 2D Kabsch: the angle maximising sum w (R a) . b
    sxx = float(w @ (a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]))
    sxy = float(w @ (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]))
    turn = math.atan2(sxy, sxx)
    c, s = math.cos(turn), math.sin(turn)
    shift = pc - np.array([c * qc[0] - s * qc[1], s * qc[0] + c * qc[1]])
    return Move(turn, (float(shift[0]), float(shift[1])), mirror)


def _free_scale(w: Walls, pts: np.ndarray, move: Move) -> float:
    """The scale a similarity fit would take, for the report: trimmed ICP from `move` with the scale left free
    (Umeyama). Pairs to the nearest wall point slide along the walls, so the scale of one set of pairs leans to 1; it
    takes passes with the pairs found again, as the turn and shift do."""
    scale, mv = 1.0, move
    for trim in [t for t in TRIMS[1:] for _ in range(PASSES)]:
        d, q, _ = faces(mv.inverse(pts) / scale, w)
        keep = d < trim
        if keep.sum() < 3:
            return math.nan
        p = pts[keep]
        turned = _procrustes(q[keep], p, np.ones(len(p)), move.mirror)  # the turn doesn't depend on the scale
        a = (q[keep] - q[keep].mean(0)) @ turned.matrix.T  # R F (q - mean)
        b = p - p.mean(0)
        var = float((a * a).sum())
        if var <= 0:
            return math.nan
        scale = float((a * b).sum()) / var
        shift = p.mean(0) - scale * (turned.matrix @ q[keep].mean(0))
        mv = Move(turned.turn, (float(shift[0]), float(shift[1])), move.mirror)
    return scale


@dataclasses.dataclass(frozen=True)
class CameraFit:
    camera: str
    at: Point  # where it lands on the plan's page
    room: str | None  # the plan room id it lands in
    expected: str | None  # the room the owner said it sees
    anchor_error: float | None  # metres from its measured spot, when there is one


@dataclasses.dataclass(frozen=True)
class Fit:
    move: Move  # plan page to the target frame
    band_points: int  # wall band points the fit used
    median: float  # metres from a wall's face, over band points within REPORT_NEAR of a centreline
    p90: float
    near_face: int  # band points within NEAR_FACE of a wall's face
    near_share: float  # near_face over the band points within REPORT_NEAR of some room's bounding box
    scale: float  # what a fit with a free scale would scale the plan by (1.0 is metric agreement); not applied
    cameras: tuple[CameraFit, ...]
    seeded_by: str  # "anchors", "rooms", "walls", or "start"
    dropped: tuple[str, ...] = ()  # measured spots left out: their cameras landed over ANCHOR_OUTLIER from them

    @property
    def outside(self) -> list[str]:
        """Cameras that landed outside the room the owner said they see."""
        return [c.camera for c in self.cameras if c.expected is not None and c.room != c.expected]


def _room_of(plan: Parsed, x: float, y: float) -> str | None:
    found = floorplan.room_at([(r.id, r.name, r.polygon) for r in plan.rooms], x, y)
    return found[0] if found else None


def _sample(points: np.ndarray, n: int) -> np.ndarray:
    if len(points) <= n:
        return points
    return points[np.linspace(0, len(points) - 1, n).astype(np.int64)]


def _raster_rooms(plan: Parsed, origin: np.ndarray, shape: tuple[int, int], cell: float) -> dict[str, np.ndarray]:
    xs = origin[0] + (np.arange(shape[0]) + 0.5) * cell
    ys = origin[1] + (np.arange(shape[1]) + 0.5) * cell
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    out = {}
    for room in plan.rooms:
        poly = np.array(room.polygon)
        inside = np.zeros(shape, dtype=bool)
        for (ax, ay), (bx, by) in zip(poly, np.roll(poly, -1, axis=0), strict=True):
            crosses = (ay > gy) != (by > gy)
            with np.errstate(divide="ignore", invalid="ignore"):
                xcross = ax + (gy - ay) * (bx - ax) / (by - ay)
            inside ^= crosses & (gx < xcross)
        out[room.id] = inside
    return out


def _coarse(
    plan: Parsed,
    w: Walls,
    band: np.ndarray,
    cams: Mapping[str, Point],
    seen: Mapping[str, str],
    mirrors: Sequence[bool],
) -> tuple[Move, str]:
    """The best turn, shift, and mirror on the coarse grid: band against the walls' score, keeping only shifts that put
    every camera in `seen` in its room when any are given."""
    cell = COARSE_CELL
    corners = np.vstack([w.a, w.b])
    o_w = corners.min(0) - MARGIN
    size = np.ceil((corners.max(0) + MARGIN - o_w) / cell).astype(np.int64)
    shape_w = (int(size[0]), int(size[1]))
    xs = o_w[0] + (np.arange(shape_w[0]) + 0.5) * cell
    ys = o_w[1] + (np.arange(shape_w[1]) + 0.5) * cell
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    d, _, _ = nearest(np.c_[gx.ravel(), gy.ravel()], w)
    score = np.exp(-((d.reshape(shape_w) / COARSE_SIGMA) ** 2))
    rooms = _raster_rooms(plan, o_w, shape_w, cell)
    hinted = [(c, cams[c], seen[c]) for c in seen if c in cams and seen[c] in rooms]
    best: tuple[float, Move] | None = None
    for mirror in mirrors:
        f = np.array([1.0, -1.0 if mirror else 1.0])
        for deg in np.arange(0.0, 360.0, COARSE_STEP_DEG):
            th = math.radians(float(deg))
            c, s = math.cos(th), math.sin(th)
            back = np.array([[c, s], [-s, c]])  # R(-turn)
            q0 = (band @ back.T) * f
            o_a = q0.min(0)
            idx = np.floor((q0 - o_a) / cell).astype(np.int64)
            shape_a = (int(idx[:, 0].max()) + 1, int(idx[:, 1].max()) + 1)
            counts = np.zeros(shape_a)
            np.add.at(counts, (idx[:, 0], idx[:, 1]), 1.0)
            corr = signal.fftconvolve(score, counts[::-1, ::-1], mode="full")
            # corr[u, v] scores shift index (m, n) = (u - (Ka - 1), v - (La - 1)): band cell (k, l) onto score cell
            # (k + m, l + n)
            valid = np.ones_like(corr, dtype=bool)
            for _, at, room in hinted:
                k = np.floor(((np.array(at) @ back.T) * f - o_a) / cell).astype(np.int64)
                u = np.arange(corr.shape[0]) - (shape_a[0] - 1) + k[0]
                v = np.arange(corr.shape[1]) - (shape_a[1] - 1) + k[1]
                ok_u = (u >= 0) & (u < shape_w[0])
                ok_v = (v >= 0) & (v < shape_w[1])
                hit = np.zeros_like(valid)
                hit[np.ix_(ok_u, ok_v)] = rooms[room][np.ix_(u[ok_u], v[ok_v])]
                valid &= hit
            if not valid.any():
                continue
            masked = np.where(valid, corr, -np.inf)
            u, v = np.unravel_index(int(np.argmax(masked)), masked.shape)
            value = float(masked[u, v])
            if best is not None and value <= best[0]:
                continue
            m, n = u - (shape_a[0] - 1), v - (shape_a[1] - 1)
            # band point q0 lands at q0 + sh on the page, sh = o_w - o_a + (m, n) cell (cell corners line up)
            sh = o_w - o_a + np.array([m, n]) * cell
            # page q = F R(-turn) p + sh, so p = R(turn) F (q - sh): shift = -R(turn) F sh
            shift = -(np.array([[c, -s], [s, c]]) @ (f * sh))
            best = (value, Move(th, (float(shift[0]), float(shift[1])), mirror))
    if best is None:
        raise PlanError("no turn and shift puts every camera in the room it sees")
    return best[1], ("rooms" if hinted else "walls")


def register(
    plan: Parsed,
    band: np.ndarray,
    cams: Mapping[str, Point],
    seen: Mapping[str, str] | None = None,
    anchors: Mapping[str, Point] | None = None,
    anchor_share: float = 0.5,
    mirror: bool | None = None,
    start: Move | None = None,
) -> Fit:
    """`plan` registered onto a map's wall band (`band`: (N, 2) points in the target frame, 1.0 to 1.9 m above the
    floor). `cams` are the registered cameras' spots in that frame; `seen` names the plan room each camera sees, and
    `anchors` the owner-measured spot of each on the plan's page (`x-camera`), which hold the fit (`anchor_share` of
    the weight; 1 for the spots alone, with the walls only measured). A spot whose camera lands over ANCHOR_OUTLIER
    from it is left out (`Fit.dropped`). `mirror` fixes whether the plan's page is mirrored into the frame (True for a
    right-handed frame seen from above, as every map here is: the page's y grows south); None tries both. `start`, when
    given, is where the fine fit starts instead of the coarse search or the spots (a map already near the plan, as a
    walk placed in the plan's own frame is). See the module docstring."""
    if not 0.0 <= anchor_share <= 1.0:
        raise ValueError("anchor_share is 0 to 1")
    seen = dict(seen or {})
    anchors = {c: a for c, a in (anchors or {}).items() if c in cams}
    w = walls(plan)
    pts = _sample(np.asarray(band, dtype=np.float64).reshape(-1, 2), MAX_BAND_POINTS)
    if len(pts) < 10:
        raise PlanError("the map has too few wall band points to register against")
    names = sorted(anchors)
    dropped: list[str] = []
    while True:
        move, seeded = (
            (start, "start") if start is not None else _seed(plan, w, pts, cams, seen, anchors, names, mirror)
        )
        move = _fine(w, pts, move, [anchors[c] for c in names], [cams[c] for c in names], anchor_share, seeded)
        errors = {c: math.dist(move.inverse(np.array([cams[c]]))[0], anchors[c]) for c in names}
        worst = max(errors, key=lambda c: errors[c], default=None)
        # a spot far off says that camera's registration is wrong; it can't be told with two or fewer
        if worst is None or len(names) <= 2 or errors[worst] <= ANCHOR_OUTLIER:
            break
        names.remove(worst)
        dropped.append(worst)
    face, _, centre = faces(move.inverse(pts), w)
    wall = centre < REPORT_NEAR
    near = int((face <= NEAR_FACE).sum())
    outline = np.zeros(len(pts), dtype=bool)
    on_page = move.inverse(pts)
    for room in plan.rooms:
        lo = np.min(room.polygon, axis=0) - REPORT_NEAR
        hi = np.max(room.polygon, axis=0) + REPORT_NEAR
        outline |= np.all((on_page >= lo) & (on_page <= hi), axis=1)
    fits = []
    for cam, at in sorted(cams.items()):
        x, y = (float(v) for v in move.inverse(np.array([at]))[0])
        anchored = anchors.get(cam)
        err = None if anchored is None else math.hypot(x - anchored[0], y - anchored[1])
        fits.append(CameraFit(cam, (x, y), _room_of(plan, x, y), seen.get(cam), err))
    return Fit(
        move,
        len(pts),
        float(np.median(face[wall])) if wall.any() else math.nan,
        float(np.percentile(face[wall], 90)) if wall.any() else math.nan,
        near,
        near / max(int(outline.sum()), 1),
        _free_scale(w, pts, move),
        tuple(fits),
        seeded,
        tuple(dropped),
    )


def _seed(
    plan: Parsed,
    w: Walls,
    pts: np.ndarray,
    cams: Mapping[str, Point],
    seen: Mapping[str, str],
    anchors: Mapping[str, Point],
    names: Sequence[str],
    mirror: bool | None,
) -> tuple[Move, str]:
    """Where the fine fit starts: the measured spots `names` when there are two or more, else the coarse search, with
    the room each other measured spot is in as the room its camera sees."""
    mirrors = (False, True) if mirror is None else (mirror,)
    if len(names) < 2:
        hints = dict(seen)
        for c in names:
            room = _room_of(plan, *anchors[c])
            if room is not None:
                hints.setdefault(c, room)
        return _coarse(plan, w, pts, cams, hints, mirrors)
    page = np.array([anchors[c] for c in names])
    there = np.array([cams[c] for c in names])
    tries = [_procrustes(page, there, np.ones(len(names)), m) for m in mirrors]
    # two spots fit either mirror exactly, so the walls break the tie: the mean face distance, capped
    best = min(
        tries,
        key=lambda mv: (
            round(float(np.sum((mv.points(page) - there) ** 2)), 4),
            float(np.mean(np.minimum(faces(mv.inverse(pts), w)[0], REPORT_NEAR))),
        ),
    )
    return best, "anchors"


def _fine(
    w: Walls,
    pts: np.ndarray,
    move: Move,
    page: Sequence[Point],
    there: Sequence[Point],
    anchor_share: float,
    seeded: str,
) -> Move:
    """Trimmed ICP from `move`: band points paired with the nearer wall face, and the measured spots `page` paired with
    their cameras' spots `there`, weighted to `anchor_share` of the whole. At a share of 1 the spots alone hold it."""
    if seeded == "anchors" and anchor_share >= 1.0:
        return move
    spots_q = np.asarray(page, dtype=np.float64).reshape(-1, 2)
    spots_p = np.asarray(there, dtype=np.float64).reshape(-1, 2)
    for trim in [t for t in TRIMS for _ in range(PASSES)]:
        d, q, _ = faces(move.inverse(pts), w)
        keep = d < trim
        q_all, p_all = q[keep], pts[keep]
        weights = np.ones(len(q_all))
        if len(spots_q):
            each = anchor_share / max(1.0 - anchor_share, 1e-9) * max(len(weights), 1) / len(spots_q)
            q_all, p_all = np.vstack([q_all, spots_q]), np.vstack([p_all, spots_p])
            weights = np.r_[weights, np.full(len(spots_q), each)]
        if len(weights) < 3 or weights.sum() <= 0:
            break
        move = _procrustes(q_all, p_all, weights, move.mirror)
    return move


def wall_error(plan: Parsed, move: Move, band: np.ndarray) -> tuple[float, float, int]:
    """How a map's wall band (in the frame `move` puts the plan in) lies on the plan's walls: the median and 90th
    percentile distance from a wall's face of the band points within REPORT_NEAR of its centreline, and how many there
    are."""
    face, _, centre = faces(move.inverse(np.asarray(band, dtype=np.float64).reshape(-1, 2)), walls(plan))
    wall = face[centre < REPORT_NEAR]
    if not len(wall):
        return math.nan, math.nan, 0
    return float(np.median(wall)), float(np.percentile(wall, 90)), len(wall)


def _n(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def _pt(p: Point) -> str:
    return f"{_n(p[0])},{_n(p[1])}"


def _poly(points: Sequence[Point], move: Move) -> list[Point]:
    """The corners moved, in the same winding on the page as they had (a mirror reverses it)."""
    out = [move.point(*p) for p in points]
    return out[::-1] if move.mirror else out


def to_dsl(plan: Parsed, move: Move, title: str, cameras: Sequence[Camera] = ()) -> str:
    """The plan in the target frame, as DSL: rooms, openings, fixtures, and camera spots (`plan.cameras` and any
    `cameras` given, the later winning for one id), titled `title`. A door's `hinge:` is left out: which end of its wall
    is the start can change under a mirror."""
    title = title.replace('"', "'")
    ext, part = plan.walls
    lines = [f'plan "{title}" walls {_n(ext)}/{_n(part)}', ""]
    for r in plan.rooms:
        name = r.name.replace('"', "'")
        lines.append(f'room {r.id} "{name}" {r.kind} poly ' + " ".join(_pt(p) for p in _poly(r.polygon, move)))
    if plan.openings:
        lines.append("")
    for o in plan.openings:
        rest = [w for w in o.rest if not w.startswith("hinge:")]
        lines.append(" ".join([o.kind, o.between, "at", _pt(move.point(*o.at)), f"w{_n(o.width)}", *rest]))
    if plan.fixtures:
        lines.append("")
    for f in plan.fixtures:
        where = [f"in:{f.room}"] if f.room else []
        corners = " ".join(_pt(p) for p in _poly(f.corners, move))
        lines.append(" ".join(["fixture", f.kind, *where, "poly", corners, *f.rest]))
    spots = {c.id: c for c in (*plan.cameras, *cameras)}
    if spots:
        lines.append("")
    for c in spots.values():
        words = ["x-camera", c.id, "at", _pt(move.point(*c.at))]
        if c.height is not None:
            words += ["height", _n(c.height)]
        if c.yaw is not None:
            words += ["yaw", _n(round(move.yaw(c.yaw), 1) % 360)]
        if c.fov is not None:
            words += ["fov", _n(c.fov)]
        lines.append(" ".join(words))
    return "\n".join(lines) + "\n"


# The map side: its wall band and its cameras, in a revision's plan frame.


def in_frame(points: np.ndarray, transform: Sequence[float]) -> np.ndarray:
    """(N, 3) map points as (plan x, plan y, height above the floor) through a 12-number map-to-plan transform
    (`places.plan_frame`)."""
    t = np.asarray(transform, dtype=np.float64).reshape(3, 4)
    return np.asarray(points, dtype=np.float64).reshape(-1, 3) @ t[:, :3].T + t[:, 3]


def band(points: np.ndarray, transform: Sequence[float]) -> np.ndarray:
    """The (N, 2) plan points of the map's wall band (BAND above the floor)."""
    xyh = in_frame(points, transform)
    return xyh[(xyh[:, 2] >= BAND[0]) & (xyh[:, 2] <= BAND[1]), :2]


def camera_spots(poses: Mapping[str, Sequence[float]], transform: Sequence[float]) -> dict[str, tuple[float, ...]]:
    """Each camera's (plan x, plan y, height above the floor) from its 4x4 camera-to-map pose (16 numbers)."""
    out = {}
    for cam, pose in poses.items():
        m = np.asarray(pose, dtype=np.float64).reshape(4, 4)
        x, y, h = in_frame(m[:3, 3], transform)[0]
        out[cam] = (float(x), float(y), float(h))
    return out


def page_is_mirrored(poses: Mapping[str, Sequence[float]], transform: Sequence[float]) -> bool | None:
    """Whether the plan's page (y growing south) is the frame's mirror image: True when the frame, with its height
    axis, is right-handed. A camera's pose is a right-handed rotation, so its handedness in the frame says the frame's;
    None with no camera to tell by."""
    t = np.asarray(transform, dtype=np.float64).reshape(3, 4)[:, :3]
    for pose in poses.values():
        r = np.asarray(pose, dtype=np.float64).reshape(4, 4)[:3, :3]
        return bool(np.linalg.det(t @ r) > 0)
    return None


# A camera's view against the plan: which way it looks, and whether it sits where its registration says.

VIEW_POINTS = 4000  # a camera's own points the yaw search uses, evenly sampled
VIEW_HEIGHTS = (-0.2, 2.4)  # metres above the floor: the camera's own points that count (no ceiling, no stray depth)
VIEW_STEP_DEG = 1.0
WALL_WEIGHT = 0.25  # how much the wall band's fit counts beside the share inside the rooms, to break ties
PLATEAU = 0.02  # turns scoring within this of the best are equally good; the facing is their middle


def look_bearing(move: Move, look: Point) -> float:
    """A direction in the target frame as a bearing on the plan's page: degrees clockwise from up the page."""
    dx, dy = np.asarray(look, dtype=np.float64) @ move.matrix  # onto the page: M^T d
    return math.degrees(math.atan2(dx, -dy)) % 360


@dataclasses.dataclass(frozen=True)
class View:
    """How a camera's own points (its room in the map) lie on the plan, turned about it at its measured spot."""

    turn: float  # radians, counterclockwise in the target frame: the middle of the turns its view fits best at `at`
    spread: float  # degrees: half the width of those turns (how sure the facing is)
    inside: float  # share of its points it could see from there on the plan at that turn (`in_sight`)
    inside_now: float  # the same where its registration has it, unturned
    wall: float  # share of its wall band points within NEAR_FACE of a wall's face at that turn
    wall_now: float
    rooms: tuple[tuple[str, float], ...]  # plan rooms its points land in at that turn, by share, most first

    @property
    def score(self) -> float:
        return self.inside + WALL_WEIGHT * self.wall

    @property
    def score_now(self) -> float:
        return self.inside_now + WALL_WEIGHT * self.wall_now


def _turned(points: np.ndarray, centre: Point, at: Point, turn: float) -> np.ndarray:
    c, s = math.cos(turn), math.sin(turn)
    off = np.asarray(points, dtype=np.float64) - np.asarray(centre)
    return off @ np.array([[c, -s], [s, c]]).T + np.asarray(at)


def inside(plan: Parsed, q: np.ndarray) -> np.ndarray:
    """Which (N, 2) page points lie in some room of the plan."""
    x, y = q[:, 0], q[:, 1]
    out = np.zeros(len(q), dtype=bool)
    for room in plan.rooms:
        poly = np.asarray(room.polygon, dtype=np.float64)
        hit = np.zeros(len(q), dtype=bool)
        for (ax, ay), (bx, by) in zip(poly, np.roll(poly, -1, axis=0), strict=True):
            crosses = (ay > y) != (by > y)
            with np.errstate(divide="ignore", invalid="ignore"):
                hit ^= crosses & (x < ax + (y - ay) * (bx - ax) / (by - ay))
        out |= hit
    return out


SIGHT_SLACK = 0.1  # metres short of a point its line of sight is checked to: a wall's own points sit on its face


def sight_walls(plan: Parsed) -> Walls:
    """The plan's walls with a gap at each door and cased opening (which can be seen through; a window can't, for
    this: what's beyond it is outside every room anyway)."""
    w = walls(plan)
    a, b = list(w.a), list(w.b)
    half = list(w.half)
    for o in plan.openings:
        if o.kind == "window":
            continue
        at = np.asarray(o.at, dtype=np.float64)
        for i in range(len(a)):
            ab = b[i] - a[i]
            length = float(np.linalg.norm(ab))
            if length < 1e-9:
                continue
            t = float((at - a[i]) @ ab) / length
            off = abs(float(ab[0] * (at - a[i])[1] - ab[1] * (at - a[i])[0])) / length
            if off > 0.02 or not -0.02 <= t <= length + 0.02:
                continue
            unit = ab / length
            lo, hi = max(t - o.width / 2, 0.0), min(t + o.width / 2, length)
            pieces = [(a[i], a[i] + unit * lo), (a[i] + unit * hi, b[i])]
            pieces = [(p, q) for p, q in pieces if float(np.linalg.norm(q - p)) > 1e-6]
            a[i : i + 1], b[i : i + 1], half[i : i + 1] = (
                [p for p, _ in pieces],
                [q for _, q in pieces],
                [half[i]] * len(pieces),
            )
            break
    return Walls(np.array(a).reshape(-1, 2), np.array(b).reshape(-1, 2), np.array(half))


def in_sight(plan: Parsed, sight: Walls, eye: Point, q: np.ndarray) -> np.ndarray:
    """Which (N, 2) page points a camera at `eye` could see: in some room, and with no wall (`sight_walls`) between
    (checked to SIGHT_SLACK short of the point)."""
    q = np.asarray(q, dtype=np.float64).reshape(-1, 2)
    e = np.asarray(eye, dtype=np.float64)
    d = q - e
    length = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)
    end = e + d * (np.maximum(length - SIGHT_SLACK, 0.0) / length)[:, None]
    seen = inside(plan, q)

    def side(p: np.ndarray, r: np.ndarray, x: np.ndarray) -> np.ndarray:
        return (r[..., 0] - p[..., 0]) * (x[..., 1] - p[..., 1]) - (r[..., 1] - p[..., 1]) * (x[..., 0] - p[..., 0])

    for wa, wb in zip(sight.a, sight.b, strict=True):
        s1, s2 = side(wa, wb, e[None, :].repeat(len(q), 0)), side(wa, wb, end)
        s3, s4 = side(e[None, :], end, wa[None, :]), side(e[None, :], end, wb[None, :])
        seen &= ~(((s1 > 0) != (s2 > 0)) & ((s3 > 0) != (s4 > 0)))
    return seen


def view_fit(plan: Parsed, move: Move, points: np.ndarray, centre: Point, at: Point) -> View:
    """A camera's own points (`points`: (N, 3) plan x, y, and height above the floor, in the target frame), seen from
    `centre` (its registered spot there), moved to `at` (its measured spot, through `move`) and turned about it by
    each whole degree. A turn scores the share of its points it could see from there on the plan (`in_sight`: in a
    room, and with no wall between, so a wrong turn throws the view through one) plus WALL_WEIGHT times the share of its wall band on a wall's face; the facing is the middle of
    the turns within PLATEAU of the best. The view's own evidence of which way the camera looks, independent of its
    registered yaw, and of whether it sits where its registration says (`score_now` against `score`)."""
    w = walls(plan)
    sight = sight_walls(plan)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pts = _sample(pts[(pts[:, 2] >= VIEW_HEIGHTS[0]) & (pts[:, 2] <= VIEW_HEIGHTS[1])], VIEW_POINTS)
    if not len(pts):
        return View(0.0, 180.0, math.nan, math.nan, math.nan, math.nan, ())
    in_band = (pts[:, 2] >= BAND[0]) & (pts[:, 2] <= BAND[1])
    xy = pts[:, :2]

    def shares(turn: float, where: Point) -> tuple[float, float]:
        q = move.inverse(_turned(xy, centre, where, turn))
        wall = float(np.mean(faces(q[in_band], w)[0] <= NEAR_FACE)) if in_band.any() else 0.0
        eye = move.inverse(np.array([where]))[0]
        return float(np.mean(in_sight(plan, sight, (float(eye[0]), float(eye[1])), q))), wall

    degrees = np.arange(0.0, 360.0, VIEW_STEP_DEG)
    found = [shares(math.radians(float(d)), at) for d in degrees]
    scores = np.array([i + WALL_WEIGHT * wl for i, wl in found])
    best = degrees[scores >= scores.max() - PLATEAU]
    # the middle of the best turns, on the circle, and how far the farthest of them is from it
    angles = np.radians(best)
    middle = math.atan2(float(np.mean(np.sin(angles))), float(np.mean(np.cos(angles))))
    spread = float(np.max(np.abs((best - math.degrees(middle) + 180.0) % 360.0 - 180.0)))
    at_best = shares(middle, at)
    on_page = move.inverse(_turned(xy, centre, at, middle))
    counts: dict[str, int] = {}
    for x, y in on_page:
        room = _room_of(plan, float(x), float(y))
        if room is not None:
            counts[room] = counts.get(room, 0) + 1
    rooms = tuple(sorted(((r, n / len(pts)) for r, n in counts.items()), key=lambda rn: -rn[1]))
    now = shares(0.0, centre)
    return View(middle, spread, at_best[0], now[0], at_best[1], now[1], rooms)


def mirrored(points: np.ndarray, centre: Point, look: Point) -> np.ndarray:
    """(N, 3) plan-frame points reflected across the vertical plane through `centre` along `look`, heights kept: the
    view a camera whose video is mirrored left to right would have made, as it really is. A mirrored video makes its
    reconstruction a reflection, which no turn undoes, so a mirrored camera's view fits the plan better reflected."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3).copy()
    d = np.asarray(look, dtype=np.float64)
    d = d / max(float(np.linalg.norm(d)), 1e-12)
    off = pts[:, :2] - np.asarray(centre)
    along = off @ d
    pts[:, :2] = np.asarray(centre) + 2 * along[:, None] * d - off
    return pts
