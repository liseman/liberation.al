"""Lining a learned map up with the owner's plan of its place (part of spatial memory).

A map learned from the cameras (spatial/surroundings_map.py) stands in a frame of its own: its first frame's camera
is the origin, and its scale is the depth model's. The owner's plan of the place is taught state, drawn in metres in
a frame no map made. This module finds the move between them on the floor (a turn, a shift, and a scale), so the 3D
view, the plan, and the positions share one frame.

**What it writes.** When a line-up is found, one revision of the place with no map of its own (producer
`PRODUCER`), whose `plan_transform` takes a map point to (plan x, plan y, height above the floor in the plan's
metres), and the owner's plan saved again, text unchanged, as drawn in that revision's frame (`correct_plan` with
source `SOURCE`). From then on every reader that asks for the plan's frame (`places.same_plan_frame`) gets the
owner's plan for the learned revisions that carry the transform over. It never writes a floor or a camera: the
floor comes from the map's own surfaces, and is read from the newest revision. What it found, or why it found
nothing, is in `state/places/<place>/plan.lineup` (`status`).

**Found once, then checked.** `advance` is called after a map step and after a floor is set. It does nothing unless a
place's map, floor, cameras, or owner plan changed since it last looked.

- *Not lined up yet*: a fit is tried when the map could cover enough of the plan (below), and again only after the
  map's wall cells grew by GROWTH since the last try, so a map that stays as it is is never fitted twice.
- *Lined up*: the grown map is measured on the plan through the transform it has (`measure`), which is one pass over
  the points with no search. While it still sits on the walls (`holds`), nothing is written. When it has left them,
  one fit is tried on the grown map: a fit that passes replaces the line-up, and otherwise the line-up stays and the
  status says `drifted`, with another fit only after the map grew by GROWTH again.

**Evidence**, strongest first:

1. *The cameras' spots.* The owner's plan can hold each fixed camera's measured spot (`x-camera <name> at x,y`). A
   camera registered in the map (the newest revision's `cameras`) should land on its spot, so two or more spots at
   least MIN_SPREAD apart give the turn, the shift, and the scale outright, and the walls only refine them. Every
   spot must end within ANCHOR_M of its camera.
2. *The walls.* The map's wall band (points 1.0 to 1.9 m above the floor, `plan_import.band`) against the plan's
   walls. The turn is one of the four quarter turns that square the map's walls to the plan's
   (`floorplan.manhattan_angle`); for each, and each scale in SCALES, the cross-correlation of the band with a score
   that falls off from the plan's walls gives every shift at once (as `plan_import`'s coarse search), and the best
   few are refined by trimmed ICP with the scale free. A single camera spot rules out the candidates that put its
   camera elsewhere.

**Enough of the plan to tell.** A wall fit is kept only when the band lies on at least MIN_COVER of the plan's wall
length (a corner of one room fits many corners), when it scores MARGIN times better than the best different fit
(a square room fits four ways, and so does one room of two alike), and when the band then sits on the walls
(MAX_MEDIAN, MIN_NEAR). Otherwise the place stays unaligned, with the reason (`REASONS`).

Limits: the walls must be roughly square to each other (as the floorplan baseline assumes), the plan is never
mirrored (a plan frame and a map's floor frame are both right-handed), and a line-up whose map has drifted and
can't be fitted again is flagged, not withdrawn.

`advance` takes `places.writing` itself around its writes: call it outside that lock.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal
from scipy.spatial import KDTree

from liberational import logs
from liberational.memory import places
from liberational.spatial import floorplan, ply
from liberational.spatial import plan_import as pi

LOG = logs.get("spatial")

PRODUCER = "plan lineup v1"
SOURCE = "lineup"  # who saved the owner's plan again in the lined-up frame (`places.correct_plan`'s source)
STATUS = "plan.lineup"  # in the place's folder
ALIGNED, UNALIGNED = "aligned", "unaligned"
#: Why a place is unaligned: nothing learned yet, the map has no floor yet, the plan has no room this reads, the map
#: covers too little of the plan, more than one fit is as good, or the best fit doesn't sit on the walls. `drifted`
#: goes with an aligned place whose grown map has left the plan. `not_tried` is a place `advance` hasn't looked at.
REASONS = ("no_map", "no_floor", "plan_unreadable", "too_little", "ambiguous", "poor_fit", "drifted", "not_tried")

SCALES = (0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25)  # map metres per plan metre, the coarse search's
MIN_SCALE, MAX_SCALE = 0.7, 1.4  # a fit outside these isn't a metric map of this plan
COARSE_CELL = 0.1  # metres per cell of the coarse search's grids
COARSE_SIGMA = 0.2  # metres: how fast the score falls off from a wall
MARGIN_M = 2.0  # metres around the plan's walls in the coarse score grid
PEAKS = 2  # shifts kept for each turn and scale
PEAK_APART = 1.0  # metres between two kept shifts
REFINED = 6  # coarse candidates refined
TRIMS = (0.5, 0.3, 0.2, 0.15, 0.1)  # ICP: pair distance kept, metres, shrinking
PASSES = 4  # ICP passes at each trim
MAX_POINTS = 20_000  # band points a fit or a check uses, evenly sampled
MIN_BAND = 300  # band points a wall fit needs
COVER_CELL = 0.25  # metres: a stretch of wall this long is covered by a band point this near
MIN_COVER = 0.3  # the share of the plan's wall length the band must lie on
MARGIN = 1.15  # how much better the best fit scores than the best different one
SAME_TURN_DEG, SAME_SHIFT_M = 10.0, 0.5  # two fits this close are one fit
MAX_MEDIAN = 0.1  # metres from a wall's face, the median of the band points near a wall
MIN_NEAR = 0.5  # the share of the band points over the plan within NEAR_M of a wall's face
NEAR_M = 0.2
CHECK_MEDIAN, CHECK_NEAR = 0.15, 0.4  # the same two for a line-up being checked: looser, so it isn't refitted on noise
MIN_WALL = 200  # band points near a wall that a check or a spots-only fit needs before the walls say anything
KEPT_NEAR = 0.75  # and the share near a face must stay this much of what it was when the line-up was found
GROWTH = 1.2  # a fit is tried again when the map's wall cells grew this much
MIN_SPREAD = 1.5  # metres between the two farthest camera spots that give a line-up by themselves
ANCHOR_M = 1.0  # a camera farther than this from its measured spot contradicts the fit
ANCHOR_SHARE = 0.5  # the spots' share of the weight when walls refine a fit they seeded

Point = tuple[float, float]


def _turned(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


def to_map(move: floorplan.Similarity, q: np.ndarray) -> np.ndarray:
    """(N, 2) plan points on the map's floor: scaled, turned, and shifted by `move`."""
    return move.scale * (np.asarray(q, dtype=np.float64).reshape(-1, 2) @ _turned(move.turn).T) + np.asarray(move.shift)


def to_plan(move: floorplan.Similarity, p: np.ndarray) -> np.ndarray:
    """(N, 2) points of the map's floor on the plan: `to_map` undone."""
    back = _turned(-move.turn)
    return ((np.asarray(p, dtype=np.float64).reshape(-1, 2) - np.asarray(move.shift)) @ back.T) / move.scale


def transform(floor: Sequence[float], move: floorplan.Similarity) -> tuple[float, ...] | None:
    """The map-to-plan transform (12 numbers, `Revision.plan_transform`) of a map with `floor` whose floor frame
    (`places.floor_frame`) `move` takes the plan onto: plan x, plan y, and the height above the floor in the plan's
    metres. None for a floor with no normal."""
    frame = places.floor_frame(floor)
    if frame is None:
        return None
    f = np.asarray(frame, dtype=np.float64).reshape(3, 4)
    back = _turned(-move.turn) / move.scale
    out = np.zeros((3, 4))
    out[:2, :3] = back @ f[:2, :3]
    out[:2, 3] = back @ (f[:2, 3] - np.asarray(move.shift))
    out[2] = f[2] / move.scale
    return tuple(float(x) for x in out.ravel())


def read_plan(text: str) -> pi.Parsed | None:
    """The owner's plan as the fit takes it: its rooms, wall thicknesses, and measured camera spots. A plan with a
    statement the full reader refuses still gives its rooms (as positions read them, `floorplan.rooms_from_dsl`);
    None for a plan with no room."""
    try:
        return pi.parse(text)
    except pi.PlanError:
        rooms = tuple(pi.Room(rid, name, "other", poly) for rid, name, poly in floorplan.rooms_from_dsl(text))
        return pi.Parsed("", (0.2, 0.1), rooms, (), (), (), 0) if rooms else None


def wall_samples(w: pi.Walls, step: float) -> np.ndarray:
    """(N, 2) points along the plan's wall centrelines, `step` apart."""
    out = []
    for a, b in zip(w.a, w.b, strict=True):
        n = max(int(math.ceil(float(np.hypot(*(b - a))) / step)), 1)
        t = (np.arange(n) + 0.5) / n
        out.append(a + t[:, None] * (b - a))
    return np.vstack(out) if out else np.zeros((0, 2))


def wall_cells(points: np.ndarray) -> int:
    """How many COVER_CELL cells of the floor hold wall band points: what the map has of walls, however dense."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return len(np.unique(np.floor(pts / COVER_CELL).astype(np.int64), axis=0)) if len(pts) else 0


def _sample(points: np.ndarray, n: int) -> np.ndarray:
    if len(points) <= n:
        return points
    return points[np.linspace(0, len(points) - 1, n).astype(np.int64)]


def _similarity(q: np.ndarray, p: np.ndarray, weights: np.ndarray) -> floorplan.Similarity | None:
    """The weighted turn, scale, and shift that best take plan points q onto map points p; None when q is one point."""
    w = weights / weights.sum()
    qc, pc = w @ q, w @ p
    a, b = q - qc, p - pc
    sxx = float(w @ (a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]))
    sxy = float(w @ (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]))
    var = float(w @ (a[:, 0] ** 2 + a[:, 1] ** 2))
    if var <= 1e-12 or math.hypot(sxx, sxy) <= 1e-12:
        return None
    turn = math.atan2(sxy, sxx)
    scale = math.hypot(sxx, sxy) / var
    shift = pc - scale * (_turned(turn) @ qc)
    return floorplan.Similarity(scale, turn, (float(shift[0]), float(shift[1])))


@dataclasses.dataclass(frozen=True)
class Measure:
    """How a map's wall band lies on the plan through a move or a transform."""

    points: int  # band points within `plan_import.REPORT_NEAR` of a wall's centreline
    median: float  # metres from a wall's face, over those; nan with none
    p90: float
    near: float  # the share of the band points over the plan within NEAR_M of a wall's face
    cover: float  # the share of the plan's wall length with a band point on it
    score: float  # the band's mean closeness to a wall's face, 0 to 1: what two fits are compared by
    anchors: dict[str, float]  # each camera's distance from its measured spot, metres

    def as_dict(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in out.items()}


def measure(w: pi.Walls, on_plan: np.ndarray, anchors: Mapping[str, tuple[Point, Point]] | None = None) -> Measure:
    """`on_plan` ((N, 2) wall band points in the plan's frame) against the plan's walls. `anchors` are cameras as
    (measured spot, where the camera lands), both on the plan."""
    off = {c: math.dist(spot, at) for c, (spot, at) in (anchors or {}).items()}
    q = np.asarray(on_plan, dtype=np.float64).reshape(-1, 2)
    if not len(q) or not len(w.a):
        return Measure(0, math.nan, math.nan, 0.0, 0.0, 0.0, off)
    face, _, centre = pi.faces(q, w)
    wall = centre < pi.REPORT_NEAR
    corners = np.vstack([w.a, w.b])
    over = np.all((q >= corners.min(0) - pi.REPORT_NEAR) & (q <= corners.max(0) + pi.REPORT_NEAR), axis=1)
    near = float(((face <= NEAR_M) & over).sum()) / max(int(over.sum()), 1)
    along = wall_samples(w, COVER_CELL)
    reached, _ = KDTree(q).query(along, distance_upper_bound=COVER_CELL)
    return Measure(
        int(wall.sum()),
        float(np.median(face[wall])) if wall.any() else math.nan,
        float(np.percentile(face[wall], 90)) if wall.any() else math.nan,
        near,
        float(np.isfinite(reached).mean()) if len(along) else 0.0,
        float(np.exp(-((face / COARSE_SIGMA) ** 2)).mean()),
        off,
    )


def _coarse(w: pi.Walls, pts: np.ndarray) -> list[tuple[float, floorplan.Similarity]]:
    """Candidate moves from the walls alone, best first: for each quarter turn that squares the band to the plan's
    walls and each scale, the PEAKS best shifts."""
    cell = COARSE_CELL
    corners = np.vstack([w.a, w.b])
    o_w = corners.min(0) - MARGIN_M
    size = np.ceil((corners.max(0) + MARGIN_M - o_w) / cell).astype(np.int64)
    shape_w = (int(size[0]), int(size[1]))
    xs = o_w[0] + (np.arange(shape_w[0]) + 0.5) * cell
    ys = o_w[1] + (np.arange(shape_w[1]) + 0.5) * cell
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    d, _, _ = pi.nearest(np.c_[gx.ravel(), gy.ravel()], w)
    score = np.exp(-((d.reshape(shape_w) / COARSE_SIGMA) ** 2))
    # R(-angle) squares a set of walls to the axes, so the plan turned by (map's - plan's) is square to the band
    base = floorplan.manhattan_angle(pts, 0.05) - floorplan.manhattan_angle(wall_samples(w, 0.05), 0.05)
    apart = max(int(round(PEAK_APART / cell)), 1)
    found: list[tuple[float, floorplan.Similarity]] = []
    for quarter in range(4):
        turn = base + quarter * math.pi / 2
        back = _turned(-turn)
        for scale in SCALES:
            q0 = (pts @ back.T) / scale
            o_a = q0.min(0)
            idx = np.floor((q0 - o_a) / cell).astype(np.int64)
            shape_a = (int(idx[:, 0].max()) + 1, int(idx[:, 1].max()) + 1)
            counts = np.zeros(shape_a)
            np.add.at(counts, (idx[:, 0], idx[:, 1]), 1.0)
            # corr[u, v] scores band cell (k, l) laid onto score cell (k + u - (Ka - 1), l + v - (La - 1))
            corr = signal.fftconvolve(score, counts[::-1, ::-1], mode="full")
            for _ in range(PEAKS):
                u, v = (int(i) for i in np.unravel_index(int(np.argmax(corr)), corr.shape))
                value = float(corr[u, v])
                if not value > 0:
                    break
                sh = o_w - o_a + np.array([u - (shape_a[0] - 1), v - (shape_a[1] - 1)]) * cell
                # a band point is q0 + sh on the plan, so on the map the plan is scale * R(turn) (q - sh)
                shift = -scale * (_turned(turn) @ sh)
                found.append((value / len(pts), floorplan.Similarity(scale, turn, (float(shift[0]), float(shift[1])))))
                corr[max(u - apart, 0) : u + apart + 1, max(v - apart, 0) : v + apart + 1] = -np.inf
    return sorted(found, key=lambda f: -f[0])


def _refine(
    w: pi.Walls,
    pts: np.ndarray,
    move: floorplan.Similarity,
    spots: np.ndarray,
    cams: np.ndarray,
    share: float,
) -> floorplan.Similarity:
    """Trimmed ICP from `move` with the scale free: each band point paired with the nearer face of the nearest plan
    wall, and each measured spot (`spots`, on the plan) with its camera (`cams`, on the map) at `share` of the
    weight."""
    for trim in [t for t in TRIMS for _ in range(PASSES)]:
        face, at, _ = pi.faces(to_plan(move, pts), w)
        keep = face < trim
        q, p = at[keep], pts[keep]
        weights = np.ones(len(q))
        if len(spots) and share > 0:
            each = share / max(1.0 - share, 1e-9) * max(len(weights), 1) / len(spots)
            q, p = np.vstack([q, spots]), np.vstack([p, cams])
            weights = np.r_[weights, np.full(len(spots), each)]
        if len(weights) < 3:
            break
        again = _similarity(q, p, weights)
        if again is None:
            break
        move = again
    return move


@dataclasses.dataclass(frozen=True)
class Fit:
    """A line-up found: the plan onto the map's floor frame, and how well it holds."""

    move: floorplan.Similarity
    by: str  # "anchors" (the cameras' measured spots seeded it) or "walls"
    measure: Measure
    margin: float  # its score over the best different fit's; inf with none to compare

    def as_dict(self) -> dict[str, Any]:
        return {
            "by": self.by,
            "scale": self.move.scale,
            "turn": self.move.turn,
            "shift": list(self.move.shift),
            "margin": None if math.isinf(self.margin) else self.margin,
            **self.measure.as_dict(),
        }


def _landed(move: floorplan.Similarity, anchors: Mapping[str, tuple[Point, Point]]) -> dict[str, tuple[Point, Point]]:
    """Each anchor as (its measured spot, where its camera lands on the plan through `move`)."""
    out = {}
    for cam, (spot, at) in anchors.items():
        x, y = to_plan(move, np.array([at]))[0]
        out[cam] = (spot, (float(x), float(y)))
    return out


def _same(a: floorplan.Similarity, b: floorplan.Similarity, centre: np.ndarray) -> bool:
    turn = abs((a.turn - b.turn + math.pi) % (2 * math.pi) - math.pi)
    apart = float(np.hypot(*(to_map(a, centre[None])[0] - to_map(b, centre[None])[0])))
    return math.degrees(turn) < SAME_TURN_DEG and apart < SAME_SHIFT_M * max(a.scale, b.scale)


def fit(plan: pi.Parsed, band: np.ndarray, anchors: Mapping[str, tuple[Point, Point]] | None = None) -> Fit | None:
    """The best line-up of `plan` with a map's wall band (`band`: (N, 2) points in the map's floor frame), or None
    when there is nothing to fit from. `anchors` are the cameras both have: (the owner's measured spot on the plan,
    the camera's spot on the map's floor). Whether the fit is good enough is `verdict`'s to say."""
    anchors = dict(anchors or {})
    w = pi.walls(plan)
    pts = _sample(np.asarray(band, dtype=np.float64).reshape(-1, 2), MAX_POINTS)
    spots = np.array([a[0] for a in anchors.values()], dtype=np.float64).reshape(-1, 2)
    cams = np.array([a[1] for a in anchors.values()], dtype=np.float64).reshape(-1, 2)
    spread = max((math.dist(a, b) for a in spots.tolist() for b in spots.tolist()), default=0.0)
    if len(anchors) >= 2 and spread >= MIN_SPREAD:
        move = _similarity(spots, cams, np.ones(len(spots)))
        if move is not None:
            if len(pts) >= MIN_BAND:
                move = _refine(w, pts, move, spots, cams, ANCHOR_SHARE)
            return Fit(move, "anchors", measure(w, to_plan(move, pts), _landed(move, anchors)), math.inf)
    if len(pts) < MIN_BAND:
        return None
    centre = np.vstack([w.a, w.b]).mean(0)
    kept: list[Fit] = []
    refused: list[Fit] = []
    for _, start in _coarse(w, pts)[:REFINED]:
        move = _refine(w, pts, start, spots, cams, 0.0)
        found = Fit(move, "walls", measure(w, to_plan(move, pts), _landed(move, anchors)), math.inf)
        if any(off > ANCHOR_M for off in found.measure.anchors.values()):
            refused.append(found)  # a camera's measured spot says this isn't where the map is
            continue
        twin = next((i for i, k in enumerate(kept) if _same(k.move, move, centre)), None)
        if twin is None:
            kept.append(found)
        elif found.measure.score > kept[twin].measure.score:
            kept[twin] = found
    if not kept:
        # every fit the walls offer puts a camera off its spot: the best of them, for `verdict` to refuse
        return max(refused, key=lambda f: f.measure.score) if refused else None
    kept.sort(key=lambda f: -f.measure.score)
    best = kept[0]
    if len(kept) > 1:
        best = dataclasses.replace(best, margin=best.measure.score / max(kept[1].measure.score, 1e-9))
    return best


def verdict(found: Fit | None) -> str | None:
    """Why `found` isn't kept (one of REASONS), or None for a line-up to keep."""
    if found is None:
        return "too_little"
    m = found.measure
    if not MIN_SCALE <= found.move.scale <= MAX_SCALE or any(off > ANCHOR_M for off in m.anchors.values()):
        return "poor_fit"
    if found.by == "anchors":
        return None  # the owner measured where the cameras are; the walls are reported, not asked
    if m.cover < MIN_COVER:
        return "too_little"
    if found.margin < MARGIN:
        return "ambiguous"
    if math.isnan(m.median) or m.median > MAX_MEDIAN or m.near < MIN_NEAR:
        return "poor_fit"
    return None


def holds(m: Measure, near_was: float | None = None) -> bool:
    """Whether a line-up being checked still holds: its cameras on their spots, and the band on the walls once it has
    enough of them to say. `near_was` is the share of the band that was on the walls when the line-up was found: a
    map that slid along its walls keeps half its band on them, so the share must not fall below KEPT_NEAR of that."""
    if any(off > ANCHOR_M for off in m.anchors.values()):
        return False
    if m.points < MIN_WALL:
        return True
    return m.median <= CHECK_MEDIAN and m.near >= max(CHECK_NEAR, KEPT_NEAR * (near_was or 0.0))


# The stored side: one place's line-up, found or checked against its newest revision.


def _path(state: Path, place: str) -> Path:
    return places.root(state) / place / STATUS


def _stored(state: Path, place: str) -> dict[str, Any] | None:
    try:
        found = json.loads(_path(state, place).read_text())
    except OSError, ValueError:
        return None
    return found if isinstance(found, dict) and found.get("state") in (ALIGNED, UNALIGNED) else None


def status(state: Path, place: str) -> dict[str, Any] | None:
    """How `place`'s map stands with its owner's plan, for a reader to say: `state` ("aligned" or "unaligned"),
    `reason` (one of REASONS, or None), `by` ("anchors", "walls", or "owner" for a plan the owner drew in the map's own
    frame), `revision` (the one it was worked out against), `at`, and the numbers of the `fit` and of the last
    `check`. None for a place with no owner's plan, or one whose map has a plan of the machine's."""
    held = places.get(state, place)
    if held is None or held.plan_source != "owner" or not held.plan or held.revision.plan is not None:
        return None
    rev = held.revision
    early = "no_map" if rev.map is None else ("no_floor" if rev.floor is None else None)
    stored = _stored(state, place)
    if early is not None or stored is None:
        return {"state": UNALIGNED, "reason": early or "not_tried", "by": None, "revision": rev.id, "at": None}
    return {k: v for k, v in stored.items() if k not in ("key", "tried")}


def _digest(*parts: object) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _anchors(plan: pi.Parsed, rev: places.Revision, frame: Sequence[float]) -> dict[str, tuple[Point, Point]]:
    """The cameras the plan has a measured spot for and `rev` registers: (the spot, the camera's point in `frame`)."""
    spots = pi.camera_spots({c: p.pose for c, p in rev.cameras.items()}, frame)
    return {c.id: (c.at, (spots[c.id][0], spots[c.id][1])) for c in plan.cameras if c.id in spots}


def _points(state: Path, rev: places.Revision) -> np.ndarray | None:
    if rev.map is None or rev.map_format != "ply":
        return None
    try:
        return ply.read(places.root(state) / rev.place / rev.map)[0]
    except OSError, ply.PlyError:
        return None  # pruned since the revision was read, or not a cloud


def _advance(state: Path, held: places.Place, now: float) -> dict[str, Any] | None:
    place, rev, text = held.id, held.revision, held.plan or ""
    if rev.plan is not None or rev.map is None or rev.floor is None:
        return status(state, place)  # a machine plan's frame, or nothing to line up yet
    floor_frame = places.floor_frame(rev.floor)
    plan = read_plan(text)
    basis = _digest(rev.floor, text, {c: p.pose for c, p in sorted(rev.cameras.items())})
    key = _digest(rev.map, basis)
    prior = _stored(state, place)
    if prior is not None and prior.get("key") == key:
        return status(state, place)

    def save(found: dict[str, Any]) -> dict[str, Any] | None:
        tried = found.pop("tried", None) or (prior or {}).get("tried")
        whole = {"by": None, "fit": None, "check": None, **found, "revision": rev.id, "at": now, "key": key}
        places.write_whole(_path(state, place), json.dumps({**whole, "tried": tried}, sort_keys=True).encode())
        return status(state, place)

    points = _points(state, rev)
    if points is None:
        return status(state, place)
    if plan is None or floor_frame is None:
        return save({"state": UNALIGNED, "reason": "plan_unreadable" if plan is None else "no_floor"})
    w = pi.walls(plan)
    chain = {r.id: r for r in places.revisions(state, place)}
    base = chain.get(held.plan_revision or "")
    drifted: dict[str, Any] | None = None
    if base is not None and places.same_plan_frame(base, rev):
        # The owner's plan is drawn in this map's frame already: by an earlier line-up, or by the owner's own hand
        if base.producer != PRODUCER:
            return save({"state": ALIGNED, "reason": None, "by": "owner"})
        through = places.plan_frame(rev) or floor_frame
        checked = measure(w, _sample(pi.band(points, through), MAX_POINTS), _anchors(plan, rev, through))
        was = {"state": ALIGNED, "by": (prior or {}).get("by"), "fit": (prior or {}).get("fit")}
        near_was = (was["fit"] or {}).get("near") if (was["fit"] or {}).get("points", 0) >= MIN_WALL else None
        if holds(checked, near_was):
            return save({**was, "reason": None, "check": checked.as_dict()})
        drifted = {**was, "reason": "drifted", "check": checked.as_dict()}
        LOG.warning(
            "%s: the map has left its plan (median %.2f m from a wall, %.0f%% of its band on one)",
            held.name, checked.median, 100 * checked.near,
        )  # fmt: skip

    band = pi.band(points, floor_frame)
    anchors = _anchors(plan, rev, floor_frame)
    cells = wall_cells(band)
    tried = (prior or {}).get("tried") or {}
    again = {"cells": cells, "basis": basis}
    grown = tried.get("basis") != basis or cells >= float(tried.get("cells") or 0) * GROWTH
    newly_drifted = drifted is not None and (prior or {}).get("reason") != "drifted"
    if not grown and not newly_drifted:
        # nothing new to tell by since the last fit: what it said stands
        return save(drifted or {k: (prior or {}).get(k) for k in ("state", "reason", "by", "fit", "check")})
    length = float(np.hypot(*(w.b - w.a).T).sum())
    if len(anchors) < 2 and (len(band) < MIN_BAND or cells * COVER_CELL < 0.7 * MIN_COVER * length):
        return save(drifted or {"state": UNALIGNED, "reason": "too_little"})
    found = fit(plan, band, anchors)
    why = verdict(found)
    numbers = found.as_dict() if found is not None else None
    if why is not None or found is None:
        return save({**(drifted or {"state": UNALIGNED, "reason": why, "fit": numbers}), "tried": again})
    lined = transform(rev.floor, found.move)
    with places.writing(state):
        newest = places.latest(state, place)
        if newest is None or newest.frame != rev.frame or newest.floor != rev.floor or newest.map != rev.map:
            return status(state, place)  # the place moved on meanwhile: the next call works on what is there
        made = places.revise(state, place, producer=PRODUCER, sources=(rev.id,), plan_transform=lined, now=now)
        view = held.view.as_dict() if held.view is not None else None
        places.correct_plan(state, place, text, now=now, base=made.id, source=SOURCE, view=view)
    LOG.info(
        "%s: lined up with its plan by its %s (scale %.3f, %.0f%% of the walls, median %.2f m)",
        held.name, found.by, found.move.scale, 100 * found.measure.cover, found.measure.median,
    )  # fmt: skip
    return save({"state": ALIGNED, "reason": None, "by": found.by, "fit": numbers, "tried": again})


def advance(state: Path, now: float | None = None) -> dict[str, dict[str, Any]]:
    """Find or check the line-up of every place the owner has a plan of, and return each one's `status`. Call it after
    a map step and after a floor is set, outside `places.writing`: a place whose map, floor, cameras, and plan are as
    they were when it last looked costs one small read."""
    at = time.time() if now is None else now
    out: dict[str, dict[str, Any]] = {}
    for held in places.places(state):
        if held.plan_source != "owner" or not held.plan:
            continue
        found = _advance(state, held, at)
        if found is not None:
            out[held.id] = found
    return out
