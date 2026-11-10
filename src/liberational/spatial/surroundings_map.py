"""`pack-surroundings-map`: one step of refining a place's map from the frames kept for it (#258). docs/SPATIAL.md
has the design. The sampler (spatial/surroundings_worker.py) starts a step when kept frames are waiting; it can be
run by hand too. A step with nothing waiting does nothing, and nothing runs it on a schedule.

A step takes up to `MAX_NEW` waiting frames (kept, and not yet in a place), oldest first, and:

1. **Pairs.** Each waiting frame is paired with the `NEIGHBOURS` frames it's most alike by descriptor, among the
   frames already placed and the other waiting ones, and MASt3R measures those pairs (spatial/mast3r_runner.py's
   pairs mode, in its own runtime): each gives one frame's pose in the other's, in MASt3R's own scale, with how many
   matches agree with it. A camera with no held intrinsics is also paired with itself, which gives its focal from
   its own point map; that first focal is kept from then on (`State.focals`), so its frames share one.
2. **Depth.** Each new frame gets Depth Anything 3's metric depth (spatial/depth_runner.py, in gsplat's runtime),
   kept beside it as a 16-bit PNG in millimetres. Depth is what makes the map metric: a pair's translation is scaled
   by the frames' metric depth over the pair's own depth (`pair_scale`).
3. **Poses.** Frames are placed outward from the ones that already have a pose (`grow`), each through its pair with
   the most agreeing matches, never fewer than `MIN_INLIERS`. With no place yet, the oldest waiting frame founds one
   at the origin. Waiting frames that pair with each other but with no place found a new place when there are at
   least `FOUND_MIN` of them. A frame no pair places is parked (below).
4. **The splat continues.** The trainer (spatial/splat_runner.py) starts from the place's current splat (`resume`),
   adds each new frame's depth points as new Gaussians, and trains `ROUND_STEPS` steps on the old and new views
   (`FOUND_STEPS` for a place's first), each against its depth, the poses refined when there are at least
   `REFINE_VIEWS` views. Before training it fits each new frame's pose to the current splat and draws the frame
   from it. On seven views of one place with the newest held out, 1,500 steps resumed from the place's earlier splat
   took 23 s and scored 12.1 dB on the held-out view, against 38 s and 9.9 dB for 3,000 steps from the map's points
   (one place, one view: a first reading, not a benchmark).
5. **The second gate.** A new frame the current splat already covered and drew well (`well_drawn`) added nothing:
   it's dropped from the kept frames. The rest are in the place from now on, at their refined poses.
6. **A revision.** The place's point cloud (the old one and the new frames' points, thinned to a `VOXEL_M` grid) and
   the trained splat are written as one new revision (memory/places.py), in the place's frame.

**A camera with a spot on the owner's plan needs no pair.** A fixed camera whose spot gives a pose
(spatial/spots.py: where it is, its measured height, its facing, and the tilt fitted from people walking) has its
kept frames placed at that pose, in the plan's own frame: x and y the plan's metres, z the height above its floor.
Its depth is lifted through the pose and checked against the plan (`spot_fit`): the floor it shows must come out at
the plan's, and its depth is scaled to put it there, since the camera's height is tape-measured and the depth
model's metres aren't. A place with such a view stands in its plan's frame: its revisions carry the plan's floor
(`spots.FLOOR`), the identity as their plan transform (`spots.PLAN`), and the cameras at their spots, and the
owner's plan is saved as drawn in that frame, so the line-up (spatial/plan_lineup.py) finds it lined up and searches
for nothing, and there's no floor to find in the map (spatial/surroundings_positions.py). A spot view's pose is held
through training (`fixed`). This is what lets a house whose fixed cameras overlap nowhere have one map: on the Linux
machine four such cameras kept a frame each, pairing placed none of the other three from the first, and the map
could not grow. A map learned in a frame of its own before a spot view arrives is started again in the plan's frame.

A camera with no spot, a spot that lacks something (its facing, say), and a camera that moves keep the pairing
path: they join a place through a view that overlaps a placed one, a spot view included, and are parked with the
reason until then (`due`, `PARKED`).

What the step knows is in `state/surroundings/map.json` (`State`): each camera's focal, each placed frame's place,
pose, or that it's parked, the frames to drop, and each place's rounds and when it last improved. The map step is that file's
only writer, as the sampler is the kept frames' only writer: the sampler drops the frames listed in `drop`.

While a step runs it says what it's doing in `state/surroundings/progress.json` (`doing`: the phase, how many frames
it took, and in training the place and its new views), and the trainer writes its step count beside it
(`training.json`). `summary` gives both as `step` only while a step holds the lock, so what a step that died left
behind is never shown.

A camera's pictures go to the models as a pinhole's (`optics`): a lens in the device manifest, or the lens of its
held intrinsics, is undone first, and its overlays are painted over and left out of the losses.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from liberational import config as cfg
from liberational import json_state, locking, logs, nvml
from liberational.memory import places
from liberational.spatial import cameras, lens, ply, sog, splats, spot_check, spots
from liberational.spatial import surroundings_frames as frames

LOG = logs.get("surroundings-map")

PRODUCER = "surroundings v1"
SOURCE = "surroundings"  # who saved the owner's plan again as drawn in a learned map's frame (`_train`)
STATE = "map.json"  # under the kept frames' folder
PROGRESS = "progress.json"  # what a running step is doing (`doing`), beside it
TRAINING = "training.json"  # the trainer's own step count while it trains, beside it
PHASES = ("pairing", "depth", "training", "writing")
MAX_NEW = 12  # waiting frames a step takes
NEIGHBOURS = 3  # frames a waiting frame is paired with
MIN_ALIKE = 0.1  # a frame less alike than this isn't worth a pair
MIN_INLIERS = 150  # matches that must agree with a pair's pose for it to place a frame
PARK_S = 86400.0  # how long a frame nothing places stays parked before it's dropped
SPOT_FLOOR_M = 0.05  # a spot view's floor this near the plan's (z = 0) is left as the depth model gave it
SPOT_SCALE = (0.75, 1.35)  # how far a spot view's depth may be scaled to bring its floor to the plan's
SPOT_WALL_POINTS = 200  # band points near a wall a spot view needs before its wall fit says anything...
SPOT_WALL_NEAR = 0.1  # ...and the share of its band on a wall under which the log warns
#: Why a frame is parked, for a page to say (`summary`'s `parked_cameras`): it overlaps nothing in a map (`overlap`);
#: its camera's spot on the plan lacks something (spots.MISSING); the plan's frame is left-handed (`frame`); or the
#: view placed from the spot doesn't sit on the plan's floor (`floor`).
PARKED = ("overlap", "height", "facing", "lens", "tilt", "check", "frame", "floor")
FOUND_MIN = 3  # waiting frames that pair only with each other found a new place when there are this many
TRAIN_LONG = 640  # the long side a frame is trained at, px
STRIDE = 4  # every this many pixels of a training frame is lifted to a point
MAX_DEPTH_M = 12.0  # a depth beyond this isn't lifted: the model's far depths are its least sure
VOXEL_M = 0.03  # the map's point cloud keeps one point per cube this size
MAX_MAP_POINTS = 1_000_000
FOUND_STEPS = 3000
ROUND_STEPS = 1500
MAX_GAUSSIANS = 400_000
REFINE_VIEWS = 3  # with fewer views a pose and the scene trade off freely, so the poses are held
MIN_DIVISION = 0.01  # a held lens division under this is a pinhole: nothing to undo
WELL_COVERED = 0.97  # the share of a new frame the current splat must already cover...
WELL_DRAWN_DB = 21.0  # ...and how well it must draw it, for the frame to add nothing (`well_drawn`)
NICE = 10
DEPTH_UNIT = 0.001  # a depth PNG's count, in metres


@dataclass
class State:
    """What the map step knows between steps (`STATE`)."""

    focals: dict[str, float] = field(default_factory=dict)  # each camera's focal over its frame's width
    frames: dict[str, dict[str, Any]] = field(default_factory=dict)  # by kept frame id: place and pose, or parked
    drop: list[str] = field(default_factory=list)  # kept frames for the sampler to drop
    places: dict[str, dict[str, Any]] = field(default_factory=dict)  # by place id: frame, rounds, views, improved
    last: dict[str, Any] = field(default_factory=dict)  # the last step: when, and what it did

    @staticmethod
    def read(folder: Path) -> State:
        raw = json_state.read(folder / STATE, {}) or {}
        return State(
            focals={str(k): float(v) for k, v in (raw.get("focals") or {}).items()},
            frames={str(k): dict(v) for k, v in (raw.get("frames") or {}).items()},
            drop=[str(x) for x in raw.get("drop") or []],
            places={str(k): dict(v) for k, v in (raw.get("places") or {}).items()},
            last=dict(raw.get("last") or {}),
        )

    def write(self, folder: Path) -> None:
        json_state.write(
            folder / STATE,
            {"focals": self.focals, "frames": self.frames, "drop": self.drop, "places": self.places, "last": self.last},
        )

    def placed(self, place: str | None = None) -> dict[str, np.ndarray]:
        """The poses (camera to world, 4 x 4) of the frames in `place`, or in any place."""
        return {
            k: np.asarray(v["pose"], dtype=np.float64).reshape(4, 4)
            for k, v in self.frames.items()
            if v.get("pose") is not None and (place is None or v.get("place") == place)
        }


def waiting(kept: Sequence[frames.Kept], state: State) -> list[frames.Kept]:
    """The kept frames with no place, oldest first: not placed, and not to be dropped. Some may be parked (`due`)."""
    out = [f for f in kept if state.frames.get(f.id, {}).get("pose") is None and f.id not in state.drop]
    return sorted(out, key=lambda f: f.t)


def expired(kept: Sequence[frames.Kept], state: State, now: float) -> list[frames.Kept]:
    """The parked frames that have been parked for PARK_S: nothing placed them in a day, and what they show has
    likely changed since."""
    out = []
    for f in waiting(kept, state):
        entry = state.frames.get(f.id) or {}
        if entry.get("parked") is not None and now - float(entry.get("since", now)) >= PARK_S:
            out.append(f)
    return out


def bases(spotted: Mapping[str, spots.Spot]) -> dict[str, str]:
    """What each camera with a spot has to be placed by, as a word that changes when the owner changes the spot or a
    tilt is fitted: "spot" when the spot gives a pose, else "spot:" and what it lacks. A camera with no spot isn't
    listed (`NO_SPOT`)."""
    return {c: "spot" if usable(sp) else "spot:" + ",".join(sp.missing or ("frame",)) for c, sp in spotted.items()}


NO_SPOT = "none"


def usable(spot: spots.Spot | None) -> bool:
    """Whether a spot gives a pose a view can be placed at: nothing missing, and a right-handed plan frame (a splat
    can't be drawn through a mirror; a plan read as the page drawn is left-handed, `Spot.mirrored`)."""
    if spot is None or spot.pose is None:
        return False
    turn = np.asarray(spot.pose.pose, dtype=np.float64).reshape(4, 4)[:3, :3]
    return float(np.linalg.det(turn)) > 0


def due(
    kept: Sequence[frames.Kept], state: State, now: float, basis: Mapping[str, str] | None = None
) -> list[frames.Kept]:
    """The waiting frames a step has something to do for, oldest first: the ones no step has tried, the parked ones
    the map has gained a view since (it may now reach them), the parked ones whose camera's spot has changed since
    (`basis`, from `bases`: the owner gave it a facing, or its tilt was fitted), and the parked ones that have
    expired (to drop them).

    A frame nothing places is **parked**: it stays kept, noted with how many views the maps had (`parked`), when it
    was first parked (`since`), what its camera had to be placed by (`basis`), and why it wasn't (`why`, one of
    PARKED). A step for it would find what the last one found, so none starts until a map has more views than
    that, or its camera's basis changes. A camera that looks at nothing the map shows and has no spot keeps one
    parked frame, renewed once a day, and costs nothing between. On the Linux machine a second fixed camera's
    frame, unplaceable from the first's, started a 0-second step every minute and was dropped after three; its
    camera then never kept a frame again."""
    views = len(state.placed())
    old = {f.id for f in expired(kept, state, now)}
    out = []
    for f in waiting(kept, state):
        entry = state.frames.get(f.id) or {}
        parked = entry.get("parked")
        moved = basis is not None and entry.get("basis") != basis.get(f.camera, NO_SPOT)
        if parked is None or views > int(parked) or moved or f.id in old:
            out.append(f)
    return out


@dataclass(frozen=True)
class SpotFit:
    """How a view placed from its camera's spot sits on the plan: the floor its depth implies (metres above the
    plan's, before any fix; None when it shows no floor), what its depth was multiplied by to bring that floor to
    the plan's, how its wall planes, lines, and wireframe sit against the plan (`spot_check`), any refinement applied
    within `spots.within_bounds` (`refined`), any stronger offset noticed past the bound (`suggestion`, while the
    owner's spot holds), and why a view is refused (`floor`), or None."""

    floor: float | None
    scale: float
    wall_points: int
    wall_median: float | None
    wall_near: float | None
    why: str | None
    score: float | None = None
    at_spot: float | None = None
    vp: float | None = None
    free_space: float | None = None
    baseboard: float | None = None
    refined: tuple[float, ...] | None = None
    shift_m: float = 0.0
    turn_deg: float = 0.0
    suggestion: dict[str, float] | None = None
    check: spot_check.SpotCheck | None = None  # the check this fit used, to keep for the camera's later views
    basis: str | None = None  # what `check` was made on (`_check_basis`)
    reused: bool = False  # whether `check` was a kept one (no search ran for this view)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "floor": None if self.floor is None else round(self.floor, 3),
            "scale": round(self.scale, 3),
            "wall_points": self.wall_points,
            "wall_median": None if self.wall_median is None else round(self.wall_median, 3),
            "wall_near": None if self.wall_near is None else round(self.wall_near, 3),
            "why": self.why,
        }
        if self.score is not None:
            out["score"] = round(self.score, 3)
            out["at_spot"] = None if self.at_spot is None else round(self.at_spot, 3)
            out["vp"] = None if self.vp is None else round(self.vp, 3)
            out["free_space"] = None if self.free_space is None else round(self.free_space, 3)
            out["baseboard"] = None if self.baseboard is None else round(self.baseboard, 3)
            out["refined"] = self.refined is not None
            out["reused"] = self.reused
            if self.refined is not None:
                out["shift_m"] = round(self.shift_m, 3)
                out["turn_deg"] = round(self.turn_deg, 2)
            if self.suggestion is not None:
                out["suggestion"] = dict(self.suggestion)
        return out


FLOOR_FLAT_DEG = 15.0  # a surface whose normal is this near straight up is level
FLOOR_BIN_M = 0.04  # the height bins a view's level surfaces are counted in
FLOOR_MIN = 50  # level points the fullest bin needs to be a floor


def level_floor(points: np.ndarray, height: float) -> float | None:
    """The height (z) of the floor in a view's `points`, lifted through a pose whose z is up, for a camera `height`
    above the plan's floor: the level surface with the most points among those below half the camera's height (a
    table is above that, and the floor is the largest thing under it). None when the view shows too little level
    surface there. Surface normals come from each point's 16 nearest neighbours.

    Not the lowest level layer (`cameras.floor_along`, right for a whole map): one fixed camera's depth has strays at
    its depth edges, and on the Linux machine 50 of them 1.2 m under the floor read as one camera's floor."""
    from scipy.spatial import KDTree

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 100:
        return None
    _, idx = KDTree(pts).query(pts, k=16)
    near = pts[idx] - pts[idx].mean(axis=1, keepdims=True)
    values, vectors = np.linalg.eigh(np.einsum("nki,nkj->nij", near, near))
    level = (np.abs(vectors[:, 2, 0]) > math.cos(math.radians(FLOOR_FLAT_DEG))) & (values[:, 0] < 0.05 * values[:, 1])
    z = pts[level & (pts[:, 2] < height / 2), 2]
    if len(z) < FLOOR_MIN:
        return None
    counts, edges = np.histogram(z, np.arange(z.min(), z.max() + 2 * FLOOR_BIN_M, FLOOR_BIN_M))
    best = int(np.argmax(counts))
    if counts[best] < FLOOR_MIN:
        return None
    middle = (edges[best] + edges[best + 1]) / 2
    return float(np.median(z[np.abs(z - middle) < 1.5 * FLOOR_BIN_M]))


def spot_fit(
    depth: np.ndarray,
    focal: float,
    pose: np.ndarray,
    plan: Any,
    image: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    refine: bool = True,
    geom: spot_check.PlanGeometry | None = None,
    known: spot_check.SpotCheck | None = None,
) -> SpotFit:
    """Check a view at its spot's `pose` (camera to plan: x, y, and height above the floor) against `plan`, refine it
    within `spots.within_bounds` when warranted, and report any stronger offset past the bound (`spot_check`).

    The camera's height is tape-measured, and the depth model's metres are only about right, so the floor says how
    far off they are: lifted through the pose, the view's floor (`level_floor`) should be the plan's, z = 0. When it's
    more than SPOT_FLOOR_M off, the depth is scaled about the camera to put it there; a scale outside SPOT_SCALE is no
    depth error, so the view is refused (`floor`).

    When `plan` is a parsed owner plan (`plan_import.Parsed`), `spot_check.verify` scores vanishing points, floor
    flatness, solid-wall free space, projected baseboards and jambs, and oriented wall planes, applying a refinement
    only inside `(spots.REFINE_M, spots.REFINE_DEG)` and keeping the owner's spot untouched with a `suggestion` beyond
    it. `geom` is `plan`'s `spot_check.prepare_plan`, when the caller has it already; `known` is a check kept for
    this camera at this same spot and plan (`_check_basis`), used as it is instead of searching again."""
    from liberational.spatial import plan_import, plan_lineup

    c2w = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    blank = np.zeros((*depth.shape, 3), np.uint8)
    points, _ = lift(depth, blank, focal, c2w, stride=STRIDE)
    centre, height = c2w[:3, 3], float(c2w[2, 3])
    level, scale = level_floor(points, height), 1.0
    if level is not None and abs(level) > SPOT_FLOOR_M and height - level > 0.1:
        scale = height / (height - level)
    if not SPOT_SCALE[0] <= scale <= SPOT_SCALE[1]:
        return SpotFit(level, scale, 0, None, None, "floor")
    if isinstance(plan, plan_import.Parsed):
        if known is not None:
            checked = known
        else:
            geom = geom if geom is not None else spot_check.prepare_plan(plan)
            vf = spot_check.prepare_view(image, depth, mask, focal, stride=STRIDE)
            checked = spot_check.verify(vf, geom, c2w, refine=refine)
        if checked.refined:
            ref_c2w = np.asarray(checked.pose, dtype=np.float64).reshape(4, 4)
            ref_pts, _ = lift(depth, blank, focal, ref_c2w, stride=STRIDE)
            ref_level = level_floor(ref_pts, float(ref_c2w[2, 3]))
            ref_scale = 1.0
            if ref_level is not None and abs(ref_level) > SPOT_FLOOR_M and float(ref_c2w[2, 3]) - ref_level > 0.1:
                ref_scale = float(ref_c2w[2, 3]) / (float(ref_c2w[2, 3]) - ref_level)
            if SPOT_SCALE[0] <= ref_scale <= SPOT_SCALE[1]:
                level, scale = ref_level, ref_scale
        return SpotFit(
            floor=level,
            scale=scale,
            wall_points=checked.chosen.wall_points,
            wall_median=checked.chosen.wall_median,
            wall_near=checked.chosen.wall_near,
            why=None,
            score=checked.chosen.total,
            at_spot=checked.at_spot.total,
            vp=checked.chosen.vp,
            free_space=checked.chosen.free_space,
            baseboard=checked.chosen.baseboard,
            refined=checked.pose if checked.refined else None,
            shift_m=checked.shift_m,
            turn_deg=checked.turn_deg,
            suggestion=checked.suggestion,
            check=checked,
            reused=known is not None,
        )
    scaled = centre + scale * (points.astype(np.float64) - centre)
    m = plan_lineup.measure(plan, plan_import.band(scaled, spots.PLAN)) if plan is not None else None
    if m is None or not m.points:
        return SpotFit(level, scale, 0, None, None, None)
    return SpotFit(level, scale, m.points, float(m.median), float(m.near), None)


@dataclass(frozen=True)
class Optics:
    """How a camera's frames become the pinhole pictures the models take: the lens to undo first (None for none),
    the overlays to paint over, and the focal of the picture that leaves, as a share of its width (None while it
    isn't known)."""

    focal: float | None
    undo: lens.Lens | None = None
    masks: tuple[tuple[float, float, float, float], ...] = ()


def optics(
    configured: lens.Lens | None,
    masks: Sequence[tuple[float, float, float, float]],
    held: lens.Held | None,
    learned: float | None,
) -> Optics:
    """A camera's optics, from the best it has. A lens in the device manifest (`configured`) is undone, and the
    focal of the undone picture is the held one, else the one learned from MASt3R. With no such lens, held
    intrinsics (`lens.Held`, fitted once and kept as taught state) give the focal and, when they hold a division,
    the lens to undo: the undone picture's focal is then the held focal times `lens.out_scale`. With neither, the
    focal is the learned one, and nothing is undone."""
    covers = tuple(masks)
    if configured is not None:
        return Optics(held.focal if held is not None else learned, configured, covers)
    if held is None:
        return Optics(learned, None, covers)
    if abs(held.division) < MIN_DIVISION:
        return Optics(held.focal, None, covers)
    # The held lens on a nominal 1 x 1 frame: fy over the height is the focal over the width times the aspect, which
    # the picture's own size gives when it's undone (`_prepare`), so it's set there.
    return Optics(held.focal, lens.Lens(1, 1, (held.focal, held.focal, 0.0, 0.0), held.division), covers)


def held_lens(undo: lens.Lens, width: int, height: int) -> lens.Lens:
    """`undo` for a `width` x `height` frame: a configured lens as it is, and a held one (nominal, centred, `optics`)
    with its focal in that frame's pixels."""
    if (undo.width, undo.height) != (1, 1):
        return undo
    focal = undo.k[0] * width
    return lens.Lens(width, height, (focal, focal, width / 2 - 0.5, height / 2 - 0.5), undo.division)


def pairs_for(
    new: Sequence[str], placed: Sequence[str], ids: Sequence[str], descriptors: np.ndarray, most: int = NEIGHBOURS
) -> list[tuple[str, str]]:
    """The pairs to measure: each of `new` with the `most` frames it's most alike, among `placed` and the other new
    ones, that are at least MIN_ALIKE alike, as (the other frame, the new one); each pair once, whichever way round.
    `descriptors` has a unit row per id in `ids`."""
    row = {k: i for i, k in enumerate(ids)}
    pool = [k for k in [*placed, *new] if k in row]
    seen: set[frozenset[str]] = set()
    out: list[tuple[str, str]] = []
    for frame in new:
        if frame not in row:
            continue
        others = [k for k in pool if k != frame]
        alike = descriptors[[row[k] for k in others]] @ descriptors[row[frame]] if others else np.zeros(0)
        for j in np.argsort(-alike)[:most]:
            if alike[j] < MIN_ALIKE or frozenset((others[j], frame)) in seen:
                continue
            seen.add(frozenset((others[j], frame)))
            out.append((others[j], frame))
    return out


def pair_scale(depths: Sequence[float], metric_a: float, metric_b: float) -> float | None:
    """What a pair's translation is multiplied by to be in metres: the frames' median metric depths (`metric_a`,
    `metric_b`) over the pair's own median depths of them (`depths`, both in the pair's scale), their geometric
    mean. None when a depth is missing or not positive."""
    ratios = [m / d for m, d in zip((metric_a, metric_b), depths, strict=True) if m > 0 and d > 0]
    if len(ratios) != 2:
        return None
    return float(math.sqrt(ratios[0] * ratios[1]))


@dataclass(frozen=True)
class Edge:
    """A measured pair: `b`'s pose in `a`'s frame, in metres, and how many matches agree with it."""

    a: str
    b: str
    b_in_a: np.ndarray
    inliers: int


def edges(pairs: Sequence[Mapping[str, Any]], metric: Mapping[str, float]) -> list[Edge]:
    """The pairs of MASt3R's answer that give a pose, each scaled to metres by the frames' metric depths (`metric`,
    by frame id). A pair of a frame with itself, a pair with no pose, and one with no scale are left out."""
    out = []
    for p in pairs:
        a, b = str(p["a"]), str(p["b"])
        if a == b or p.get("forward") is None or a not in metric or b not in metric:
            continue
        scale = pair_scale(p["depths"], metric[a], metric[b])
        if scale is None:
            continue
        pose = np.asarray(p["forward"], dtype=np.float64).reshape(4, 4).copy()
        pose[:3, 3] *= scale
        out.append(Edge(a, b, pose, int(min(p["inliers"]))))
    return out


def grow(poses: Mapping[str, np.ndarray], found: Sequence[Edge], least: int = MIN_INLIERS) -> dict[str, np.ndarray]:
    """Place frames outward from the ones that have a pose (`poses`, camera to world): again and again, the edge with
    the most inliers (at least `least`) between a placed frame and an unplaced one places it. Returns the poses of
    the frames it placed."""
    have = dict(poses)
    new: dict[str, np.ndarray] = {}
    usable = sorted((e for e in found if e.inliers >= least), key=lambda e: -e.inliers)
    while True:
        for e in usable:
            if e.a in have and e.b not in have:
                new[e.b] = have[e.b] = have[e.a] @ e.b_in_a
                break
            if e.b in have and e.a not in have:
                new[e.a] = have[e.a] = have[e.b] @ np.linalg.inv(e.b_in_a)
                break
        else:
            return new


def lift(
    depth: np.ndarray, image: np.ndarray, focal: float, pose: np.ndarray, stride: int = STRIDE
) -> tuple[np.ndarray, np.ndarray]:
    """A frame's depth (H x W, metres along the view axis) as points in the map's frame, every `stride` pixels, with
    their colours from `image` (H x W x 3, RGB): the pixels with a depth above zero and at most MAX_DEPTH_M. The
    camera looks along its +z with its principal point at the frame's centre."""
    h, w = depth.shape
    v, u = np.mgrid[stride // 2 : h : stride, stride // 2 : w : stride]
    z = depth[v, u]
    keep = (z > 0) & (z <= MAX_DEPTH_M)
    u, v, z = u[keep], v[keep], z[keep]
    local = np.stack([(u + 0.5 - w / 2) * z / focal, (v + 0.5 - h / 2) * z / focal, z], axis=1)
    c2w = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    return (local @ c2w[:3, :3].T + c2w[:3, 3]).astype(np.float32), np.asarray(image)[v, u].astype(np.uint8)


def thin(
    points: np.ndarray, colors: np.ndarray, voxel: float = VOXEL_M, most: int = MAX_MAP_POINTS
) -> tuple[np.ndarray, np.ndarray]:
    """One point per `voxel` cube (the first given, so points already in the map stay), and at most `most` of them,
    picked with a fixed seed."""
    if not len(points):
        return points, colors
    cells = np.floor(np.asarray(points, dtype=np.float64) / voxel).astype(np.int64)
    _, first = np.unique(cells, axis=0, return_index=True)
    first.sort()
    if len(first) > most:
        first = np.sort(np.random.default_rng(0).choice(first, most, replace=False))
    return points[first], colors[first]


def well_drawn(before: Mapping[str, Any]) -> bool:
    """Whether the splat a step started from already covered a new frame and drew it well (the trainer's `before`
    for it, read after the frame's pose is fitted to the splat): the second gate. Such a frame adds nothing.

    Measured on one place of two fixed cameras on the Linux machine, four views in: eight frames taken a minute or
    ten after a kept one, put past the first gate by hand. Five were covered 0.99 and drew at 21.6 to 24.7 dB: copies,
    dropped. Three stayed: one drew at 21.4 dB but was only 0.74 covered, and two drew at 18.6 and 11.7 dB (the light
    had moved on). A step earlier, the same view in changed light drew at 20.1 dB with 0.99 covered, and a frame from
    the second camera, a new viewpoint, at 11.4 dB with 0.67 covered. Read at their poses as registered, before the
    fit, the five copies drew at 17.8 to 19.2 dB: a few centimetres off is a few pixels off, which reads as change."""
    psnr = before.get("psnr")
    return float(before.get("covered") or 0.0) >= WELL_COVERED and psnr is not None and float(psnr) >= WELL_DRAWN_DB


def adopt(state: Path) -> str | None:
    """The place a first learned map belongs to: the one listed place with no map (an owner's plan kept across a
    reset), when there's exactly one; else None, for a new place."""
    empty = [p.id for p in places.places(state) if p.revision.map is None]
    return empty[0] if len(empty) == 1 else None


def train_size(width: int, height: int) -> tuple[int, int]:
    """A frame's size for training: its long side TRAIN_LONG (or its own, when smaller), both sides even."""
    scale = min(1.0, TRAIN_LONG / max(width, height))
    return max(2, round(width * scale / 2) * 2), max(2, round(height * scale / 2) * 2)


def _prepare(store: frames.Store, frame: frames.Kept, folder: Path, optic: Optics) -> Path:
    """`frame`'s picture as the models take it, at its training size, as a PNG in `folder`: its overlays painted over
    and its lens undone (`optic`), with a `.mask.png` beside it when some of it is then covered (overlays, the
    border an undone lens leaves, and the people the sampler marked in it, `frames.Store.covered_path`)."""
    import cv2

    image = cv2.imread(str(store.path(frame)))
    if image is None:
        raise RuntimeError(f"the kept frame {frame.id} doesn't read")
    undo, _ = _prepare_focal(frame, optic)
    height, width = image.shape[:2]
    image, covered = lens.prepare(image, undo, optic.masks)
    people = cv2.imread(str(store.covered_path(frame)), cv2.IMREAD_GRAYSCALE)
    if people is not None and people.shape == (height, width):
        if undo is not None:  # moved as the picture was
            mx, my = lens.undistort_maps(undo, width, height)
            people = cv2.remap(people, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        covered = covered | (people > 127)
    size = train_size(frame.width, frame.height)
    out = folder / f"{frame.id}.png"
    cv2.imwrite(str(out), cv2.resize(image, size, interpolation=cv2.INTER_AREA))
    if covered.any():
        cv2.imwrite(
            str(_mask_path(out)), cv2.resize(covered.astype(np.uint8) * 255, size, interpolation=cv2.INTER_NEAREST)
        )
    return out


def _mask_path(image: Path) -> Path:
    return image.with_suffix(".mask.png")


def _depth_path(store: frames.Store, frame_id: str) -> Path:
    return store.folder / f"{frame_id}.depth.png"


def _depths(
    store: frames.Store, todo: Sequence[frames.Kept], pictures: Pictures, runtime: Path, gpu: str | None, folder: Path
) -> None:
    """Give each of `todo` its depth PNG (16-bit millimetres, at its training size, 0 where its picture is covered)
    beside its kept frame, from Depth Anything 3's metric model. Raises RuntimeError when the model's code or weights
    are missing or it fails."""
    import cv2

    if not todo:
        return
    source, checkpoint = runtime / splats.DEPTH_SOURCE, splats.depth_checkpoint()
    if not source.is_dir():
        raise RuntimeError(f"Depth Anything 3's code isn't in {source.parent}; run `mise run gsplat`")
    if not checkpoint.is_file():
        raise RuntimeError(f"Depth Anything 3's weights aren't at {checkpoint}; run `mise run models`")
    outs = [folder / f"{f.id}.depth.npy" for f in todo]
    asked = [
        {"image": str(pictures.image(f.id)), "focal": pictures.focal(f.id), "out": str(out)}
        for f, out in zip(todo, outs, strict=True)
    ]
    request = folder / "depth.json"
    request.write_text(json.dumps({"source": str(source), "checkpoint": str(checkpoint), "views": asked}))
    splats.run_depth(runtime, request, gpu)
    for f, out in zip(todo, outs, strict=True):
        counts = np.clip(np.round(np.load(out) / DEPTH_UNIT), 0, 65535).astype(np.uint16)
        mask = pictures.mask(f.id)
        covered = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE) if mask is not None else None
        if covered is not None:
            counts[covered > 127] = 0
        cv2.imwrite(str(_depth_path(store, f.id)), counts)


def _read_rgb(path: Path) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"the frame {path} doesn't read")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _read_mask(path: Path | None) -> np.ndarray | None:
    """A prepared picture's covered pixels (`_mask_path`, 255 where covered), or None without one."""
    import cv2

    return None if path is None else cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def _read_depth(path: Path) -> np.ndarray:
    import cv2

    counts = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if counts is None:
        raise RuntimeError(f"the depth image {path} doesn't read")
    return counts.astype(np.float32) * DEPTH_UNIT


def _metric(depth: np.ndarray) -> float:
    """A depth map's median over the pixels that have one."""
    seen = depth[depth > 0]
    return float(np.median(seen)) if seen.size else 0.0


def forget(state: State, kept: Sequence[frames.Kept]) -> None:
    """Drop what `state` holds about frames that are no longer kept (the sampler's store pushed them out), and the
    drops the sampler has carried out."""
    have = {f.id for f in kept}
    state.frames = {k: v for k, v in state.frames.items() if k in have}
    state.drop = [k for k in state.drop if k in have]


class Pictures:
    """A step's prepared pictures (`_prepare`), made as they're asked for, with each one's focal in its pixels."""

    def __init__(self, store: frames.Store, folder: Path, optic: dict[str, Optics]) -> None:
        self.store, self.folder, self.optic = store, folder, optic
        self.by_id = {f.id: f for f in store.kept}
        self._image: dict[str, Path] = {}
        self._factor: dict[str, float] = {}  # a picture's focal in pixels over its camera's focal share

    def image(self, frame_id: str) -> Path:
        if frame_id not in self._image:
            frame = self.by_id[frame_id]
            optic = self.optic[frame.camera]
            self._image[frame_id] = _prepare(self.store, frame, self.folder, optic)
            self._factor[frame_id] = _prepare_focal(frame, optic)[1]
        return self._image[frame_id]

    def focal(self, frame_id: str) -> float | None:
        self.image(frame_id)
        share = self.optic[self.by_id[frame_id].camera].focal
        return None if share is None else share * self._factor[frame_id]

    def mask(self, frame_id: str) -> Path | None:
        path = _mask_path(self.image(frame_id))
        return path if path.is_file() else None

    def size(self, frame_id: str) -> tuple[int, int]:
        frame = self.by_id[frame_id]
        return train_size(frame.width, frame.height)


def _prepare_focal(frame: frames.Kept, optic: Optics) -> tuple[lens.Lens | None, float]:
    """The lens `_prepare` undoes for `frame`, and what its camera's focal share is multiplied by to be the prepared
    picture's focal in pixels: the picture's width, times `lens.out_scale` when a held lens was undone."""
    undo = held_lens(optic.undo, frame.width, frame.height) if optic.undo is not None else None
    held_one = optic.undo is not None and (optic.undo.width, optic.undo.height) == (1, 1)
    scale = lens.out_scale(undo) if undo is not None and held_one else 1.0
    return undo, scale * train_size(frame.width, frame.height)[0]


def doing(folder: Path, phase: str | None, **more: Any) -> None:
    """Say what the running step is doing (one of PHASES, with what's known of it), or with None that it's done.
    Only for a page to show: a failure to write it never stops a step."""
    try:
        if phase is None:
            (folder / PROGRESS).unlink(missing_ok=True)
            (folder / TRAINING).unlink(missing_ok=True)
        else:
            json_state.write(folder / PROGRESS, {"phase": phase, "t": time.time(), **more})
    except OSError as error:
        LOG.debug("couldn't write the step's progress: %s", error)


def step(
    state_dir: Path,
    now: float | None = None,
    held: Mapping[str, lens.Held] | None = None,
    spotted: Mapping[str, spots.Spot] | None = None,
    plans: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
    """One step (the module docstring); what it did, or None when no frame is waiting. `held` are the cameras' held
    intrinsics, `spotted` their spots on the owners' plans (`spots.read`), and `plans` each place's owner plan text,
    all read from `state_dir` by default. Raises RuntimeError when a runtime or a model is missing or fails; the
    state file is then as it was."""
    with locking.locked(frames.root(state_dir) / "map"):  # one step at a time
        try:
            return _step(
                state_dir,
                time.time() if now is None else now,
                lens.held(state_dir) if held is None else held,
                spots.read(state_dir) if spotted is None else spotted,
                plans,
            )
        finally:
            doing(frames.root(state_dir), None)


def _plan_text(state_dir: Path, place: str, plans: Mapping[str, str] | None) -> str | None:
    """`place`'s owner plan as saved (a scene's text, spatial/plan_scene.py), or None when it has none."""
    if plans is not None:
        return plans.get(place)
    held = places.get(state_dir, place)
    return held.plan if held is not None and held.plan_source == "owner" and held.plan else None


@dataclass
class _PlanAt:
    """A place's owner plan for one step: its scene as parsed (`plan_lineup.read_plan`; None when it doesn't read),
    and its `spot_check.prepare_plan` geometry, made the first time a view of it needs a search."""

    parsed: Any
    _geom: spot_check.PlanGeometry | None = None

    def geom(self) -> spot_check.PlanGeometry | None:
        from liberational.spatial import plan_import

        if self._geom is None and isinstance(self.parsed, plan_import.Parsed):
            self._geom = spot_check.prepare_plan(self.parsed)
        return self._geom


def _plan_shape(plan: Any) -> list[Any]:
    """What a spot check reads of a parsed plan (`plan_import.Parsed`, read from the place's scene): its wall
    thicknesses, its rooms' outlines, and its openings. Nothing else the scene keeps (its title, its records, other
    cameras' spots) changes the check."""
    from liberational.spatial import plan_import

    if not isinstance(plan, plan_import.Parsed):
        return []
    return [
        [round(float(v), 6) for v in plan.walls],
        [[r.id, [[round(float(x), 6), round(float(y), 6)] for x, y in r.polygon]] for r in plan.rooms],
        sorted(
            [o.kind, round(float(o.at[0]), 6), round(float(o.at[1]), 6), round(float(o.width), 6)]
            for o in plan.openings
        ),
    ]


def _check_basis(plan: Any, camera: str, held: lens.Held | None, focal: float, size: Sequence[int]) -> str:
    """What a camera's kept spot check depends on besides its spot's pose: the plan's walls, rooms, and openings
    (`_plan_shape`, from its scene), the camera's configured lens and held intrinsics, the focal of its undone picture
    (a share of its width), and its size, with `spot_check.CHECK_VERSION`. A kept check made on another basis is
    searched again."""
    import hashlib

    held_part = None if held is None else [round(held.focal, 6), round(held.division, 6)]
    raw = json.dumps(
        [
            spot_check.CHECK_VERSION,
            _plan_shape(plan),
            lens.as_registered(camera),
            held_part,
            round(focal / max(size[0], 1), 5),
            list(size),
        ]
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _park(state: State, frame: frames.Kept, now: float, basis: str, why: str) -> None:
    """Park `frame` (`due`)."""
    since = (state.frames.get(frame.id) or {}).get("since", now)
    state.frames[frame.id] = {"parked": len(state.placed()), "since": since, "basis": basis, "why": why}


def _step(
    state_dir: Path,
    now: float,
    held: Mapping[str, lens.Held],
    spotted: Mapping[str, spots.Spot],
    plans: Mapping[str, str] | None,
) -> dict[str, Any] | None:
    store = frames.Store(frames.root(state_dir))
    state = State.read(store.folder)
    forget(state, store.kept)
    for f in expired(store.kept, state, now):
        LOG.info("a frame of %s was parked for a day with nothing to place it by: dropped", f.camera)
        state.drop.append(f.id)
        state.frames.pop(f.id, None)
    basis = bases(spotted)
    by_id = {f.id: f for f in store.kept}
    # A view placed from a spot stands where the spot says. When the owner moves the spot, or the tilt is fitted
    # again, the view is placed again from where the spot is now.
    for k, entry in list(state.frames.items()):
        spot = spotted.get(by_id[k].camera) if k in by_id else None
        was_spot = entry.get("spot_pose") or entry.get("pose")
        if entry.get("spot") and was_spot is not None and (not usable(spot) or not _at(was_spot, spot)):
            LOG.info("%s: its spot on the plan changed; its view is placed again", by_id[k].camera)
            del state.frames[k]
    todo = due(store.kept, state, now, basis)[:MAX_NEW]
    if not todo:
        state.write(store.folder)
        return None
    manifest = lens.manifest()

    def fov_focal(camera: str) -> float | None:
        """A camera's focal as a share of its width from its spot's field of view, when nothing is held for it."""
        spot = spotted.get(camera)
        if spot is None or spot.pose is None or camera in held:
            return None
        return spot.pose.focal / spot.pose.width

    optic = {
        camera: optics(
            *lens.for_camera(manifest, camera), held.get(camera), state.focals.get(camera) or fov_focal(camera)
        )
        for camera in {f.camera for f in store.kept}
    }
    at_spot = {f.id: spotted[f.camera] for f in todo if usable(spotted.get(f.camera)) and optic[f.camera].focal}
    pairing = [f for f in todo if f.id not in at_spot]
    placed_ids = [k for k in state.placed() if k in by_id]
    # A frame that pairs is paired with the placed frames and with this step's spot views, which are about to be.
    measure = pairs_for([f.id for f in pairing], [*placed_ids, *at_spot], [f.id for f in store.kept], store.descriptors)
    unknown = {f.camera: f.id for f in reversed(pairing) if optic[f.camera].focal is None}
    measure += [(k, k) for k in unknown.values()]  # a camera's own focal, from a frame paired with itself
    gpu = cameras.pick_gpu(int(cfg.get("spatial.gpu")), nvml.gpus())
    mast3r, gsplat = cfg.path("spatial.runtime"), cfg.path("spatial.splat_runtime")
    began = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="surroundings-map-") as tmp:
        folder = Path(tmp)
        pictures = Pictures(store, folder, optic)
        involved = sorted({k for pair in measure for k in pair} | {f.id for f in todo})
        answer: dict[str, Any] = {"pairs": []}
        if measure:
            doing(store.folder, "pairing", frames=len(todo))
            images = {k: pictures.image(k) for k in sorted({k for pair in measure for k in pair})}
            answer = cameras.run_mast3r(mast3r, images, folder / "mast3r", gpu, pairs=measure, nice=NICE)
        for p in answer["pairs"]:
            for name, focal, size in zip((p["a"], p["b"]), p.get("focals") or (), p.get("sizes") or (), strict=False):
                camera = by_id[name].camera
                if optic[camera].focal is None and focal and size[0]:
                    state.focals[camera] = float(focal) / float(size[0])
                    optic[camera] = Optics(state.focals[camera], optic[camera].undo, optic[camera].masks)
                    LOG.info("%s: its focal is %.3f of its picture's width, from MASt3R", camera, state.focals[camera])
        ready = [f for f in todo if optic[f.camera].focal is not None]
        doing(store.folder, "depth", frames=len(todo))
        _depths(store, [f for f in ready if not _depth_path(store, f.id).is_file()], pictures, gsplat, gpu, folder)

        # The views placed from their cameras' spots: no pairing, only a check against the plan.
        did: dict[str, Any] = {"at": now, "taken": len(todo), "places": {}, "spots": {}}
        spot_poses: dict[str, dict[str, np.ndarray]] = {}  # place -> frame id -> pose
        spot_fits: dict[str, SpotFit] = {}  # frame id -> fit
        refused: dict[str, str] = {}  # frame id -> why
        plan_at: dict[str, _PlanAt] = {}  # place -> its plan, read once a step
        # Each camera's kept spot check (`spots.REFINED`): its spot's pose, the basis, and the check. A camera's views
        # all stand at one pose, so the search runs once a spot and plan, and every later view reuses it.
        kept_checks: dict[str, tuple[tuple[float, ...], str, spot_check.SpotCheck]] = {}
        for camera, row in spots.refined_poses(state_dir).items():
            kept = spot_check.SpotCheck.from_dict(row["check"]) if row.get("check") else None
            if kept is not None and row.get("basis"):
                kept_checks[camera] = (row["spot"], row["basis"], kept)
        from liberational.spatial import plan_lineup

        for f in todo:
            spot = at_spot.get(f.id)
            if spot is None or spot.pose is None or not _depth_path(store, f.id).is_file():
                continue
            pose = np.asarray(spot.pose.pose, dtype=np.float64).reshape(4, 4)
            focal = pictures.focal(f.id)
            assert focal is not None
            if spot.place not in plan_at:
                text = _plan_text(state_dir, spot.place, plans)
                plan_at[spot.place] = _PlanAt(plan_lineup.read_plan(text) if text else None)
            plan_here = plan_at[spot.place]
            depth_here = _read_depth(_depth_path(store, f.id))
            check_basis = _check_basis(plan_here.parsed, f.camera, held.get(f.camera), focal, depth_here.shape[::-1])
            kept_check = kept_checks.get(f.camera)
            known = (
                kept_check[2]
                if kept_check is not None
                and kept_check[1] == check_basis
                and np.allclose(kept_check[0], spot.pose.pose, atol=1e-6)
                else None
            )
            fit = spot_fit(
                depth_here,
                focal,
                pose,
                plan_here.parsed,
                image=_read_rgb(pictures.image(f.id)) if known is None else None,
                mask=_read_mask(pictures.mask(f.id)) if known is None else None,
                geom=plan_here.geom() if known is None else None,
                known=known,
            )
            if fit.check is not None:
                kept_checks[f.camera] = (spot.pose.pose, check_basis, fit.check)
                fit = replace(fit, basis=check_basis)
            spot_fits[f.id] = fit
            did["spots"][f.camera] = fit.as_dict()
            LOG.info("%s: its view at its spot on the plan: %s", f.camera, json.dumps(fit.as_dict()))
            if fit.refined is not None and not fit.reused:
                LOG.info(
                    "%s: refined within spot bounds by %.2f m and %.1f deg (score %.3f -> %.3f)",
                    f.camera,
                    fit.shift_m,
                    fit.turn_deg,
                    fit.at_spot or 0.0,
                    fit.score or 0.0,
                )
            if fit.suggestion is not None and fit.reused:
                pass  # warned when the check was made; the Places page keeps saying it
            elif fit.suggestion is not None:
                LOG.warning(
                    "%s: its view fits the plan better %.2f m and %.1f deg from its spot (score %.3f vs %.3f); "
                    "the owner's spot holds",
                    f.camera,
                    fit.suggestion["shift_m"],
                    fit.suggestion["turn_deg"],
                    fit.suggestion["score"],
                    fit.at_spot or 0.0,
                )
            elif fit.wall_points >= SPOT_WALL_POINTS and (fit.wall_near or 0.0) < SPOT_WALL_NEAR:
                LOG.warning(
                    "%s: only %.0f%% of its view's wall band is on a wall of the plan (median %.2f m off): its spot, "
                    "facing, or tilt may be off; the spot holds",
                    f.camera,
                    100 * (fit.wall_near or 0.0),
                    fit.wall_median or 0.0,
                )
            if fit.why is not None:
                refused[f.id] = fit.why
                continue
            if fit.scale != 1.0:
                _scale_depth(_depth_path(store, f.id), fit.scale)
            chosen_pose = np.asarray(fit.refined, dtype=np.float64).reshape(4, 4) if fit.refined is not None else pose
            spot_poses.setdefault(spot.place, {})[f.id] = chosen_pose
        # A place with a view at a spot stands in its plan's frame. A map learned in a frame of its own before (its
        # first camera the origin) can't take such a view: it's started again in the plan's, from the frames kept.
        for place in spot_poses:
            if place in state.places and not state.places[place].get("plan_frame"):
                LOG.info("%s: its map is started again in its plan's frame, from its cameras' spots", place)
                del state.places[place]
                for k in [k for k, entry in state.frames.items() if entry.get("place") == place]:
                    del state.frames[k]

        depth = {k: _read_depth(_depth_path(store, k)) for k in involved if _depth_path(store, k).is_file()}
        found = edges(answer["pairs"], {k: _metric(d) for k, d in depth.items()})
        # Place what joins a place (one that exists, or one a spot view founds now), then let what's left found new
        # ones.
        joined: dict[str, tuple[str, np.ndarray]] = {}  # frame id -> (place, pose)
        for place in dict.fromkeys([*state.places, *spot_poses]):
            anchored = {**state.placed(place), **spot_poses.get(place, {})}
            for k, pose in grow(anchored, found).items():
                joined.setdefault(k, (place, pose))
        groups: list[dict[str, np.ndarray]] = []
        for f in ready:
            if f.id in joined or f.id in at_spot or any(f.id in g for g in groups):
                continue
            free = [e for e in found if not {e.a, e.b} & (joined.keys() | at_spot.keys())]
            group = {f.id: np.eye(4), **grow({f.id: np.eye(4)}, free)}
            if (not state.places and not spot_poses and not groups) or len(group) >= FOUND_MIN:
                groups.append(group)
        targets: list[tuple[str | None, dict[str, np.ndarray]]] = []
        for place in dict.fromkeys([*state.places, *spot_poses]):
            mine = {**spot_poses.get(place, {}), **{k: pose for k, (p, pose) in joined.items() if p == place}}
            if mine:
                targets.append((place, mine))
        targets += [(None, g) for g in groups]
        for place, new_poses in targets:
            at = {k: at_spot[k] for k in new_poses if k in at_spot}
            fits_here = {k: spot_fits[k] for k in new_poses if k in spot_fits}
            result = _train(
                state_dir, store, state, place, new_poses, pictures, gsplat, gpu, folder, now, at, fits_here
            )
            did["places"][result["place"]] = result
        taken = {k for _, g in targets for k in g}
        for f in todo:
            if f.id in taken:
                continue
            spot = spotted.get(f.camera)
            why = refused.get(f.id) or (
                "overlap" if spot is None or usable(spot) else (spot.missing[0] if spot.missing else "frame")
            )
            _park(state, f, now, basis.get(f.camera, NO_SPOT), why)
        did["unplaced"] = len([f for f in todo if f.id not in taken])
    did["seconds"] = round(time.monotonic() - began, 1)
    state.last = did
    state.write(store.folder)
    return did


def _at(pose: Sequence[float], spot: spots.Spot | None) -> bool:
    """Whether a view's stored `pose` is still where its camera's `spot` puts it."""
    if spot is None or spot.pose is None:
        return False
    return bool(
        np.allclose(np.asarray(pose, dtype=np.float64), np.asarray(spot.pose.pose, dtype=np.float64), atol=1e-6)
    )


def _scale_depth(path: Path, scale: float) -> None:
    """Multiply a depth PNG's metres by `scale`, in place."""
    import cv2

    counts = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if counts is None:
        raise RuntimeError(f"the depth image {path} doesn't read")
    cv2.imwrite(str(path), np.clip(np.round(counts.astype(np.float64) * scale), 0, 65535).astype(np.uint16))


def _train(
    state_dir: Path,
    store: frames.Store,
    state: State,
    place: str | None,
    new_poses: Mapping[str, np.ndarray],
    pictures: Pictures,
    runtime: Path,
    gpu: str | None,
    folder: Path,
    now: float,
    at_spot: Mapping[str, spots.Spot] | None = None,
    spot_fits: Mapping[str, SpotFit] | None = None,
) -> dict[str, Any]:
    """Continue `place`'s splat (or found a place's, when None) with the frames in `new_poses`, write the revision,
    and note the frames in `state`; what it did. `at_spot` are the new frames placed from their cameras' spots, by
    frame: the place then stands in its owner plan's frame (`spots.FLOOR`, `spots.PLAN`), its revision registers
    those cameras at their spots (or their within-bound refinements), and those views' poses are held, not refined."""
    at_spot = at_spot or {}
    spot_fits = spot_fits or {}
    known = state.places.get(place) if place is not None else None
    in_plan = bool(at_spot) or bool((known or {}).get("plan_frame"))
    held_views = set(at_spot) | {k for k, entry in state.frames.items() if entry.get("spot")}
    by_id = {f.id: f for f in store.kept}
    old_poses = state.placed(place) if place is not None else {}
    old_poses = {k: v for k, v in old_poses.items() if k in by_id and _depth_path(store, k).is_file()}
    order = [*old_poses, *new_poses]
    prior_entries = {k: dict(state.frames[k]) for k in order if k in state.frames}
    poses = {**old_poses, **new_poses}
    views = []
    for k in order:
        mask = pictures.mask(k)
        views.append(
            {
                "image": str(pictures.image(k)),
                "pose": [float(x) for x in poses[k].ravel()],
                "focal": pictures.focal(k),
                "width": pictures.size(k)[0],
                "height": pictures.size(k)[1],
                "depth": str(_depth_path(store, k)),
                **({} if mask is None else {"mask": str(mask)}),
                **({"fixed": True} if k in held_views else {}),
            }
        )
    points, colors = [], []
    for k in new_poses:
        focal = pictures.focal(k)
        assert focal is not None  # a frame is placed only once its camera's focal is known
        p, c = lift(_read_depth(_depth_path(store, k)), _read_rgb(pictures.image(k)), focal, poses[k])
        points.append(p)
        colors.append(c)
    new_points, new_colors = thin(np.concatenate(points), np.concatenate(colors))
    # What the place has in the frame this step adds to: nothing for a place being founded, or started again.
    chain = places.revisions(state_dir, place) if place is not None and known is not None else []
    chain = [r for r in chain if r.frame == known.get("frame")] if known is not None else []
    was = chain[-1] if chain else None
    with_splat = [r for r in chain if places.has_splat(state_dir, r)]
    resume = places.splat_path(state_dir, with_splat[-1], "gs.ply") if with_splat else None
    out = folder / f"train-{len(order)}-{next(iter(new_poses))}"
    out.mkdir()
    steps = ROUND_STEPS if resume is not None else FOUND_STEPS
    (store.folder / TRAINING).unlink(missing_ok=True)  # the training before this one's
    doing(store.folder, "training", frames=len(new_poses), place=place, views=len(new_poses), steps=steps)
    np.savez(out / "start.npz", points=new_points, colors=new_colors)
    job = out / "job.json"
    job.write_text(
        json.dumps(
            {
                "views": views,
                "init": str(out / "start.npz"),
                "steps": steps,
                "progress": str(store.folder / TRAINING),
                "cap": MAX_GAUSSIANS,
                "sh_degree": splats.SH_DEGREE,
                "out": str(out),
                "reg": 0.0,
                "bound": True,
                "pose_opt": len(order) >= REFINE_VIEWS,
                "depth_unit": DEPTH_UNIT,
                **(
                    {}
                    if resume is None
                    else {"resume": str(resume), "pose_start": 0, "score": list(range(len(old_poses), len(order)))}
                ),
            }
        )
    )
    result = splats.run_runner(runtime, job, gpu)
    doing(store.folder, "writing", frames=len(new_poses), place=place, views=len(new_poses))
    master = (out / "splat.ply").read_bytes()
    rows = ply.vertices(master)
    if rows is None:
        raise RuntimeError("the splat trainer wrote a PLY this reader can't take")
    before = {order[int(b["view"])]: b for b in result.get("before") or []}
    dropped = [k for k in new_poses if k in before and well_drawn(before[k])]
    added = [k for k in new_poses if k not in dropped]
    refined = result.get("poses") or [None] * len(order)
    # The cloud: the place's old points, then the points of the new frames that stay.
    cloud_p = np.concatenate(
        [p for k, p in zip(new_poses, points, strict=True) if k in added] or [np.zeros((0, 3), np.float32)]
    )
    cloud_c = np.concatenate(
        [c for k, c in zip(new_poses, colors, strict=True) if k in added] or [np.zeros((0, 3), np.uint8)]
    )
    if was is not None and was.map is not None:
        old_p, old_c, _ = ply.read(places.root(state_dir) / was.place / was.map)
        cloud_p = np.concatenate([old_p.astype(np.float32), cloud_p])
        cloud_c = np.concatenate([old_c if old_c is not None else np.full((len(old_p), 3), 128, np.uint8), cloud_c])
    cloud_p, cloud_c = thin(cloud_p, cloud_c)
    registered: dict[str, places.CameraPose] = {}
    for k, spot in at_spot.items():
        if spot.pose is None or k not in added:
            continue
        fit = spot_fits.get(k)
        registered[by_id[k].camera] = (
            replace(spot.pose, pose=fit.refined, at=now) if fit is not None and fit.refined is not None else spot.pose
        )
    with places.writing(state_dir):
        target = place if place is not None else adopt(state_dir)
        rev = places.revise(
            state_dir,
            target,
            producer=PRODUCER,
            map_bytes=ply.dumps(cloud_p, cloud_c),
            map_format="ply",
            now=now,
            frame=known.get("frame") if known is not None else None,
            # The frame doesn't move, so what was set in it holds: the floor and the plan's transform. A place in
            # its plan's frame has the plan's own floor and no transform to find.
            floor=spots.FLOOR if in_plan else (was.floor if was is not None else None),
            plan_transform=spots.PLAN if in_plan else (was.plan_transform if was is not None else None),
            # The cameras carry over by themselves in a frame that holds; in a new one the old frame's don't.
            cameras={**(was.cameras if was is not None else {}), **registered} if in_plan or was is None else None,
        )
        places.write_splat(state_dir, rev, master, sog.encode(rows), source="surroundings")
        if in_plan and known is None:
            # The owner's plan is drawn in this frame: say so, as a line-up would (`plan_lineup`), so every reader
            # of the plan's frame takes this map's revisions, and the line-up has nothing to search for.
            owned = places.get(state_dir, rev.place)
            if owned is not None and owned.plan and owned.plan_source == "owner":
                view = owned.view.as_dict() if owned.view is not None else None
                places.correct_plan(state_dir, rev.place, owned.plan, now=now, base=rev.id, source=SOURCE, view=view)
        for k, spot in at_spot.items():
            if spot.pose is not None and k in added:
                fit = spot_fits.get(k)
                spots.save_refined(
                    state_dir,
                    by_id[k].camera,
                    spot.pose.pose,
                    fit.refined if fit is not None else None,
                    now,
                    basis=fit.basis if fit is not None else None,
                    check=fit.check.as_dict() if fit is not None and fit.check is not None else None,
                )
    for i, k in enumerate(order):
        if k in dropped:
            state.drop.append(k)
            state.frames.pop(k, None)
            continue
        pose = refined[i] if refined[i] is not None else [float(x) for x in poses[k].ravel()]
        if k in held_views:
            pose = [float(x) for x in poses[k].ravel()]  # held to its spot or within-bound refinement
        focal = pictures.focal(k)
        share = None if focal is None else focal / pictures.size(k)[0]
        state.frames[k] = {"place": rev.place, "pose": pose, "focal": share, "size": list(pictures.size(k))}
        if k in held_views:
            state.frames[k]["spot"] = True
            prior = prior_entries.get(k) or {}
            spot_here = at_spot.get(k)
            if spot_here is not None and spot_here.pose is not None:
                state.frames[k]["spot_pose"] = [float(x) for x in spot_here.pose.pose]
            elif prior.get("spot_pose") is not None:
                state.frames[k]["spot_pose"] = list(prior["spot_pose"])
            fit_here = spot_fits.get(k)
            suggestion = fit_here.suggestion if fit_here is not None else prior.get("suggestion")
            if suggestion is not None:
                state.frames[k]["suggestion"] = dict(suggestion)
    entry = state.places.setdefault(rev.place, {"frame": rev.frame, "rounds": 0})
    if in_plan:
        entry["plan_frame"] = True
    entry["rounds"] = int(entry.get("rounds", 0)) + 1
    entry["views"] = len(order) - len(dropped)
    entry["revision"] = rev.id
    if added:
        entry["improved"] = now
    did = {
        "place": rev.place,
        "revision": rev.id,
        "founded": resume is None,
        "views": entry["views"],
        "added": len(added),
        "already_drawn": len(dropped),
        "before": [before[k] for k in new_poses if k in before],
        "gaussians": result.get("gaussians"),
        "train_psnr": result.get("train_psnr"),
        "pose_shift": result.get("pose_shift"),
        "train_seconds": result.get("seconds"),
        "peak_vram_gib": result.get("peak_vram_gib"),
        "points": len(cloud_p),
    }
    LOG.info("map: %s", json.dumps(did))
    return did


@dataclass(frozen=True)
class View:
    """A kept frame placed in a place: which camera took it and when, its pose (camera to world, 16 numbers row by
    row, in the place's frame, the camera looking along its +z), and its picture as the models saw it (its lens
    undone when the camera has one): its focal as a share of its width, and its size."""

    id: str
    camera: str
    t: float
    pose: tuple[float, ...]
    focal: float | None
    width: int
    height: int


def views(state_dir: Path, place: str) -> list[View]:
    """The views `place`'s map is built from, oldest first; none for a place with no learned map."""
    store = frames.Store(frames.root(state_dir))
    state = State.read(store.folder)
    out = []
    for f in sorted(store.kept, key=lambda f: f.t):
        entry = state.frames.get(f.id) or {}
        if entry.get("place") != place or entry.get("pose") is None:
            continue
        width, height = entry.get("size") or train_size(f.width, f.height)
        focal = entry.get("focal")
        out.append(
            View(
                f.id,
                f.camera,
                f.t,
                tuple(float(x) for x in entry["pose"]),
                None if focal is None else float(focal),
                int(width),
                int(height),
            )
        )
    return out


def _read_spots(state_dir: Path) -> Mapping[str, spots.Spot] | None:
    try:
        return spots.read(state_dir)
    except (OSError, ValueError, KeyError) as error:
        LOG.warning("couldn't read the cameras' spots: %s", error)
        return None


def spot_bases(state_dir: Path) -> dict[str, str] | None:
    """`bases` of the spots as they read now, or None when they can't be read (then no parked frame is taken for
    changed: a store that doesn't read must not start a step a minute)."""
    found = _read_spots(state_dir)
    return None if found is None else bases(found)


def in_plan_frame(state_dir: Path, place: str) -> bool:
    """Whether `place`'s learned map stands in its owner plan's own frame (it has a view placed from a camera's
    spot): its floor is the plan's and its cameras are at their spots, so there's no floor to find in it and no
    camera to register from its views."""
    return bool(State.read(frames.root(state_dir)).places.get(place, {}).get("plan_frame"))


def running(state_dir: Path) -> bool:
    """Whether a map step is running now (it holds the step's lock)."""
    import fcntl

    lock = locking.lock_path(frames.root(state_dir) / "map")
    try:
        with lock.open("a") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except OSError:
        return False
    return False


def _count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def progress(folder: Path) -> dict[str, Any]:
    """What the running step last said it's doing (`doing`): its `phase` (one of PHASES, or None when it hasn't
    said), `frames` (the waiting frames it took), `place` (the place being trained; None for one being founded, and
    before training), `views` (the new views going into it), and in training `step` of `steps`, each None when
    unknown. Read whole or not at all: a file that isn't one of ours says nothing."""
    said = json_state.read(folder / PROGRESS, {}) or {}
    said = said if isinstance(said, dict) else {}
    phase = said.get("phase") if said.get("phase") in PHASES else None
    place = said.get("place")
    out: dict[str, Any] = {
        "phase": phase,
        "frames": _count(said.get("frames")),
        "place": place if isinstance(place, str) else None,
        "views": _count(said.get("views")),
        "step": None,
        "steps": None,
    }
    if phase == "training":
        trained = json_state.read(folder / TRAINING, {}) or {}
        trained = trained if isinstance(trained, dict) else {}
        out["steps"] = _count(trained.get("steps")) or _count(said.get("steps"))
        out["step"] = _count(trained.get("step"))
    return out


def summary(
    state_dir: Path, now: float | None = None, spotted: Mapping[str, spots.Spot] | None = None
) -> dict[str, Any]:
    """What a page says about learning surroundings: whether it's on and which cameras are sampled now, how many
    frames are kept, how many of them a step is due for (`waiting`) and how many are parked until the map grows
    (`parked`, `due`; `parked_cameras` says why for each camera with one, a word of PARKED), any spot whose view fits
    the plan better past `spots.within_bounds` while the owner's spot holds (`spot_suggestions`), whether a step is
    running and what it's doing (`step`: `progress`, None while none runs), when the last step ended (`last_step`, Unix
    seconds, or None), each sampled camera's recent looks by outcome and when a frame of it was last kept (`cameras`,
    from the sampler's status), each learned place's newest revision, views, rounds, and when a step last added a view
    to it (Unix seconds, or None), and each camera's newest kept frame (`frames`, by camera: the frame's id), for a
    page's thumbnail."""
    from liberational.spatial import surroundings

    folder = frames.root(state_dir)
    rows = [row for row in json_state.read(folder / "index.json", []) or [] if isinstance(row, dict)]
    index = []  # the kept frames as the index lists them, without loading their descriptors
    for row in rows:
        try:
            index.append(frames.Kept(**{k: row[k] for k in frames.FIELDS}))
        except KeyError, TypeError:
            continue
    state = State.read(folder)
    switch = surroundings.read()
    current_spots = _read_spots(state_dir) if spotted is None else spotted
    basis = None if current_spots is None else bases(current_spots)
    todo = due(index, state, time.time() if now is None else now, basis)
    due_ids = {f.id for f in todo}
    parked = {
        f.camera: str((state.frames.get(f.id) or {}).get("why") or "overlap")
        for f in waiting(index, state)
        if f.id not in due_ids
    }
    is_running = running(state_dir)
    sampling = list(switch.cameras) if switch.on else []
    sampler = json_state.read(folder / "status.json", {}) or {}
    looked = sampler.get("cameras") if isinstance(sampler, dict) else None
    looked = looked if isinstance(looked, dict) else {}
    began, took = state.last.get("at"), state.last.get("seconds")
    newest: dict[str, frames.Kept] = {}
    newest_placed: dict[str, frames.Kept] = {}
    for kept in index:
        if kept.camera not in newest or kept.t > newest[kept.camera].t:
            newest[kept.camera] = kept
        if (state.frames.get(kept.id) or {}).get("pose") is not None and (
            kept.camera not in newest_placed or kept.t > newest_placed[kept.camera].t
        ):
            newest_placed[kept.camera] = kept
    suggestions: dict[str, dict[str, float]] = {}
    if current_spots is not None:
        for camera, kept in sorted(newest_placed.items()):
            entry = state.frames.get(kept.id) or {}
            sug = entry.get("suggestion")
            was_spot = entry.get("spot_pose") or entry.get("pose")
            if (
                entry.get("spot")
                and isinstance(sug, dict)
                and was_spot is not None
                and _at(was_spot, current_spots.get(camera))
            ):
                shift, turn = float(sug.get("shift_m") or 0.0), float(sug.get("turn_deg") or 0.0)
                suggestions[camera] = {
                    "shift_m": round(shift, 2),
                    "turn_deg": round(turn, 1),
                    # which of the spot's bounds the better fit is past (`spots.within_bounds`)
                    "far_shift": bool(sug.get("far_shift", shift > spots.REFINE_M)),
                    "far_turn": bool(sug.get("far_turn", turn > spots.REFINE_DEG)),
                }
    return {
        "on": switch.on,
        "sampling": sampling,
        "kept": len(index),
        "waiting": len(todo),
        "parked": len(waiting(index, state)) - len(todo),
        "parked_cameras": dict(sorted(parked.items())),
        "spot_suggestions": suggestions,
        "running": is_running,
        "step": progress(folder) if is_running else None,
        "last_step": float(began) + float(took or 0.0) if isinstance(began, (int, float)) else None,
        "cameras": {
            camera: {
                "recent": {str(k): n for k, n in ((looked[camera].get("recent") or {}).items()) if _count(n)},
                "last_kept": looked[camera].get("last_kept"),
            }
            for camera in sampling
            if isinstance(looked.get(camera), dict)
        },
        "places": {
            place: {
                "revision": entry.get("revision"),
                "views": int(entry.get("views") or 0),
                "rounds": int(entry.get("rounds") or 0),
                "improved": entry.get("improved"),
            }
            for place, entry in state.places.items()
        },
        "frames": {camera: kept.id for camera, kept in sorted(newest.items())},
    }


def line_up(state_dir: Path) -> None:
    """Give each grown map its floor and fixed cameras (spatial/surroundings_positions.py), then find or check each
    place's line-up with its owner's plan (spatial/plan_lineup.py), now that a step may have grown its map. It costs
    one small read for a place that didn't change, and a failure is logged, not raised: the step itself is done."""
    from liberational.spatial import plan_lineup, surroundings_positions

    try:
        # the grown map's floor and fixed cameras first (#260): the line-up needs the floor
        surroundings_positions.advance(state_dir)
        found = plan_lineup.advance(state_dir)
    except (OSError, ValueError) as error:
        LOG.warning("couldn't line the maps up with their plans: %s", error)
        return
    for place, status in found.items():
        LOG.info(
            "plan line-up of %s: %s%s", place, status["state"], f" ({status['reason']})" if status["reason"] else ""
        )


def main(argv: Sequence[str] | None = None) -> None:
    """`pack-surroundings-map`: one step of refining the places' maps from the frames kept for them. Needs CUDA,
    MASt3R's runtime (`mise run mast3r`), and gsplat's with Depth Anything 3 (`mise run gsplat`)."""
    parser = argparse.ArgumentParser(prog="pack-surroundings-map", description=main.__doc__)
    parser.parse_args(argv)
    logs.setup()
    if sys.platform == "darwin":
        LOG.info("the map step needs CUDA; skipping on macOS")
        return
    try:
        did = step(cfg.path("paths.state"))
    except RuntimeError as error:
        LOG.error("no map step: %s", error)
        raise SystemExit(1) from None
    if did is None:
        LOG.info("no kept frame is waiting for a place")
    else:
        LOG.info("step: %s", json.dumps({k: v for k, v in did.items() if k != "places"}))
    line_up(cfg.path("paths.state"))


if __name__ == "__main__":
    main()
