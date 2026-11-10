"""Tier 1 refinement (part of spatial memory): each new revision of a place is fused into the
place's map.

A place's revisions come from different reconstructions: a phone walkthrough (`pack-walkthrough`), the fixed
cameras' daily snapshot (`pack-spatial-cameras`), and later the rig. Each MASt3R run has its own frame, scale,
and coverage. MASt3R's scale is not metric: on Place 1's fixed cameras people measured 0.68 of its units tall. `refine`
brings the newest one into the place's fused map and writes the result as a new revision:

1. **Level.** The new cloud is turned so its floor (the revision's floor plane, else one fitted) is z = 0 with +z
   up. The fused map is kept in this levelled frame, so its floor is (0, 0, 1, 0).
   **Metric scale.** The owner's measured camera heights (each fixed camera's `mount.height_m` in `config/devices.toml`, floor to lens) fix each
   cloud's scale (`metric`): the median, over its cameras with a known height, of measured height over height above
   the cloud's floor. The new cloud is scaled by it, and so is the fused map when it is off by more than RESCALE_TOL,
   so registration meets two metric clouds. A cloud with no known camera (a walkthrough) keeps its scale. Walking
   people's median height is logged beside it as a check (STATURE_M at the right scale): they measure short while a
   camera's pitch is off.
2. **First map.** With no fused map yet, the levelled cloud, thinned to `VOXEL` (3 cm), is the map.
3. **Register.** Otherwise the new cloud is registered against the fused map: a coarse search over yaw (every
   `YAW_STEP` degrees) and floor-plane shift, by cross-correlating top-down wall images (`COARSE_CELL` cells, the
   wall band only), then point-to-point ICP with scale, turning only about z (Umeyama on trimmed nearest neighbours,
   with no tilt: both clouds are levelled), on top of each cloud's own MASt3R alignment. The registration is kept when
   at least `MIN_FITNESS` of the new cloud over the map's footprint lies within `INLIER_M` of the map (so a whole-home
   walkthrough registers onto a first map of one room) and at least `MIN_INLIERS` points do; otherwise nothing is fused
   and the place keeps the new revision as it stands. A cloud its caller already placed another way (`fuse_in`'s
   `seated`: a walkthrough the fixed cameras seat) is only polished by ICP against the map around there, and its seat
   stands when that doesn't hold.
4. **Fuse and re-derive.** The registered cloud is merged into the map voxel by voxel (`merge`): each `VOXEL` cell
   keeps its mean position and colour and counts how many fusions saw it and how many missed it while it was in
   view, so things that come and go (a person, a moved chair) fade out after a couple of visits; the counts ride in
   the map's PLY. The fixed cameras keep the poses their own MASt3R run gave them, moved with the cloud: nothing
   shifts them toward each other by the people they see. A second in which each of two cameras sees one person is
   most often two people, each hidden from the other camera (on Place 1, half of such seconds, checked by eye in the
   frames), and a seated person's feet are on a footstool, not the floor; shifting the cameras onto those
   points moved Place 1's two cameras 1.1 m apart at floor landmarks both see. A fused map whose own floor has drifted
   more than `RELEVEL_DEG` off level is turned level again,
   with its cameras (`relevel`). The floorplan is derived again from the fused cloud (spatial/floorplan.py,
   through `cameras.layout`, on the levelled floor at z = 0). The new revision carries the fused map, its plan and
   transform, the floor, the place's signals, and the registered fixed cameras moved into the fused frame (the new
   revision's own, else the fused map's last ones), kept rigid: the registration's scale moves a camera but doesn't
   scale its axes. Where people walked (the fused cameras' footprints) goes in beside the cloud. The new revision keeps
   the fused map's `frame` (memory/places.py) while the map isn't rescaled or turned level, so a splat made for an
   earlier revision of that frame still fits it; otherwise it starts a frame of its own.

**Joining places**: `join` merges other places into one, each other place's fused map moved by a transform
its caller found (`pack-walkthrough`'s `join_places`: a walk MASt3R seats both in), as a tier 1 revision of the
place they go into; the others are retired (memory/places.py).

A merged-away place's owner plan correction is carried into the joined plan (`_carry_plans`): its frame is related to
the joined map (one `frame`, else the cameras both revisions registered, `frame_between`), its statements moved into
the frame of the plan that holds there and turned to the nearest quarter turn (rooms of one home share their walls'
directions), and saved as a correction with source "merge". The joined place's own owner rooms hold where both have
rooms; a carried owner room replaces machine rooms it covers.

**Seating a room's cameras**: after a join, the joined place's rooms come back from the daily fixed-camera run
as separate groups (their cameras never match each other), and the coarse wall search can't find one room in a whole
home. `seat` fuses such a group's snapshot in from where its cameras are registered in the map (`by_cameras`), with
ICP to polish it, refused when ICP moves them more than SEAT_SHIFT_M or SEAT_TURN_DEG off those poses. A camera whose
seat is refused night after night has moved; `release` takes it out of the place so it can be registered on its own.

**LiDAR walks hold, and a map can be rebuilt from them**: a Stray Scanner walk's cloud is fused `walked`,
and each voxel counts the walks that saw it (`Fused.walked`, a `walked` count in the map's PLY). A fixed camera's
MASt3R snapshot is several centimetres softer, so it never fades a walked voxel (`merge`), and its points go in only
where no walk saw or next to the walk's voxels (`gate`); otherwise its walls a few centimetres off would thicken the
walk's into two, night by night. `rebuild` makes a place's map again from its walks and tonight's snapshots alone,
in the frame its owner plan is drawn in, with each fixed camera standing on the spot the owner measured
(tools/rebuild_place.py runs it, dry run first). After that a camera's pose is a time series, not pinned: each `seat`
records its cameras' poses as observations (spatial/poses.py), and the owner's measured spots
(`measured_spots`) are evidence of where a camera was, read with them, not a pin. While the owner's plan is drawn on a fused map's frame with no plan file, tier 1 derives no
machine plan for it (`_revise`), so the frame stays the owner plan's.

`refine` runs after each revision a producer writes (the daily fixed-camera run, a walkthrough's first map) and for
every place in the daily run (a walkthrough into a place already there is fused without a revision of its own first,
`fuse_in`), so a revision missed on the way is fused the next day. A revision tier 1 wrote itself is
never fused again. The run is niced with its producer.
"""

from __future__ import annotations

import itertools
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from liberational import logs
from liberational.memory import places
from liberational.spatial import cameras, floorplan, footprints, ply, rig
from liberational.spatial import poses as poses_mod

LOG = logs.get("spatial-fuse")

PRODUCER = "tier1 fuse v1"
VOXEL = 0.03  # metres: the fused map's resolution
REG_VOXEL = 0.05  # metres: the clouds are thinned to this for registration
REG_POINTS = 60_000  # at most this many points of each cloud take part in registration
COARSE_CELL = 0.10  # metres: the top-down wall images' cells
WALL_BAND = (0.3, 2.0)  # metres above the floor that are wall in the top-down images
YAW_STEP = 5  # degrees between the coarse search's yaws
ICP_ITERS = 40
ICP_REACH = (0.40, 0.20, 0.10, 0.05)  # metres: the nearest-neighbour cut-off, tightening over the iterations
SCALE_RANGE = (0.8, 1.25)  # a registration's scale must be within this (both clouds are put at metric scale first)
INLIER_M = 0.05  # metres: a new point this close to the map after registration counts as fitting it
# The registration is kept when at least MIN_FITNESS of the new cloud's points over the map's footprint (the map's
# top-down cells, grown by one) fit it, and at least MIN_INLIERS (of the thinned cloud) do. Counting only the overlap
# lets a whole-home walkthrough register onto a first map of one room; the count keeps a sliver of overlap from passing.
MIN_FITNESS = 0.35
MIN_INLIERS = 1000
RELEVEL_DEG = 0.3  # a fused map whose own floor is tilted more than this off z is turned level again
#: A candidate floor with more than this share of the map more than UNDER_FLOOR_M below it is no floor (`relevel`).
UNDER_FLOOR_SHARE, UNDER_FLOOR_M = 0.05, 0.05
COLUMN = 0.10  # metres: the top-down column a voxel is in view by (see `merge`)
# Checking the scale by people:
STATURE_M = 1.65  # metres from a walking adult's heels to their head joint (halpe26's top of the head): a check
CALIBRATE_S = 2 * 86400.0  # how far back the fixed cameras' people are read
MIN_WALKERS = 200  # walking people's heights the check needs
RESCALE_TOL = 0.05  # a fused map whose people's scale is off by more than this fraction is rescaled
STEP = 0.03  # a person whose feet moved this far across the frame (a fraction) in a sample or two is walking


def voxelize(points: np.ndarray, colors: np.ndarray | None, size: float) -> tuple[np.ndarray, np.ndarray | None]:
    """One point per occupied `size` cube: the mean position (and colour) of the points in it."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return pts, None if colors is None else np.zeros((0, 3), np.uint8)
    keys = _keys(pts, size)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    sums = np.zeros((len(counts), 3))
    np.add.at(sums, inverse, pts)
    out = sums / counts[:, None]
    if colors is None:
        return out, None
    csum = np.zeros((len(counts), 3))
    np.add.at(csum, inverse, np.asarray(colors, dtype=np.float64).reshape(-1, 3))
    return out, np.clip(np.round(csum / counts[:, None]), 0, 255).astype(np.uint8)


def apply(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """`points` (N, 3) through a 4x4 transform."""
    return points @ transform[:3, :3].T + transform[:3, 3]


def leveller(floor: Sequence[float]) -> np.ndarray:
    """The rigid 4x4 transform that puts the floor (nx, ny, nz, d), a point's height above it n . p + d, at z = 0
    with +z up."""
    n = np.asarray(floor[:3], dtype=np.float64)
    norm = float(np.linalg.norm(n))
    n, d = n / norm, float(floor[3]) / norm
    z = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(n, z), float(n @ z)
    if np.linalg.norm(v) < 1e-12:
        rot = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rot = np.eye(3) + vx + vx @ vx * (1 / (1 + c))
    out = np.eye(4)
    out[:3, :3] = rot
    out[2, 3] = d  # after the turn, a point's z is n . p; its height is that plus d
    return out


def umeyama(src: np.ndarray, dst: np.ndarray, scale: bool = True) -> np.ndarray:
    """The 4x4 similarity that best maps `src` onto `dst` (least squares), turning only about z: a yaw, a uniform
    scale, and a 3D shift. Both clouds are levelled (floor at z = 0, up +z) before they're registered, so a tilt could
    only come from noise, and fused maps build on themselves: a free rotation let each fusion tilt the map a little
    further, and pulled each new, level cloud along with it (1.05, 1.36, then 1.6 degrees on Place 1)."""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    a, b = src - mu_s, dst - mu_d
    # The yaw maximizing sum(b . R a) over rotations about z: atan2 of the xy cross-covariance's antisymmetric part
    # over its trace.
    sxx, sxy = float(b[:, 0] @ a[:, 0]), float(b[:, 0] @ a[:, 1])
    syx, syy = float(b[:, 1] @ a[:, 0]), float(b[:, 1] @ a[:, 1])
    yaw = math.atan2(syx - sxy, sxx + syy)
    c, s_ = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
    turned = a @ rot.T
    s = float((b * turned).sum() / (a**2).sum()) if scale else 1.0
    s = min(max(s, SCALE_RANGE[0]), SCALE_RANGE[1])
    out = np.eye(4)
    out[:3, :3] = s * rot
    out[:3, 3] = mu_d - s * rot @ mu_s
    return out


def _thin(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pts, _ = voxelize(points, None, REG_VOXEL)
    if len(pts) > REG_POINTS:
        pts = pts[rng.choice(len(pts), REG_POINTS, replace=False)]
    return pts


def _walls(points: np.ndarray, origin: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    band = points[(points[:, 2] >= WALL_BAND[0]) & (points[:, 2] <= WALL_BAND[1])]
    cells = np.floor((band[:, :2] - origin) / COARSE_CELL).astype(np.int64)
    ok = (cells[:, 0] >= 0) & (cells[:, 1] >= 0) & (cells[:, 0] < shape[0]) & (cells[:, 1] < shape[1])
    image = np.zeros(shape)
    np.add.at(image, (cells[ok, 0], cells[ok, 1]), 1.0)
    return np.minimum(image, 1.0)


def coarse(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    """The yaw and floor-plane shift that best overlay the source's walls on the target's (both levelled), as a 4x4
    transform, and its overlap score (0 to 1)."""
    lo = np.minimum(source[:, :2].min(axis=0), target[:, :2].min(axis=0))
    span = np.maximum(source[:, :2].max(axis=0), target[:, :2].max(axis=0)) - lo
    size = int(2 ** math.ceil(math.log2(max(16, 2 * float(span.max()) / COARSE_CELL + 8))))
    centre = target[:, :2].mean(axis=0)
    origin = centre - size * COARSE_CELL / 2
    tgt = _walls(target, origin, (size, size))
    ft = np.fft.rfft2(tgt)
    src_centre = source[:, :2].mean(axis=0)
    best = (np.eye(4), -1.0)
    for deg in range(0, 360, YAW_STEP):
        a = math.radians(deg)
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        xy = (source[:, :2] - src_centre) @ rot.T + centre
        moved = np.c_[xy, source[:, 2]]
        img = _walls(moved, origin, (size, size))
        if img.sum() == 0 or tgt.sum() == 0:
            continue
        corr = np.fft.irfft2(ft * np.conj(np.fft.rfft2(img)), s=tgt.shape)
        k = np.unravel_index(int(np.argmax(corr)), corr.shape)
        score = float(corr[k]) / math.sqrt(float(img.sum()) * float(tgt.sum()))
        if score > best[1]:
            shift = np.array([k[0] if k[0] < size // 2 else k[0] - size, k[1] if k[1] < size // 2 else k[1] - size])
            out = np.eye(4)
            out[:2, :2] = rot
            out[:2, 3] = centre - rot @ src_centre + shift * COARSE_CELL
            best = (out, score)
    return best


@dataclass(frozen=True)
class Registration:
    transform: np.ndarray  # new cloud (levelled) to the fused map's frame, a 4x4 similarity
    fitness: float  # the share of the new cloud over the map's footprint within INLIER_M of the map
    inliers: int  # how many points (of the thinned new cloud) are within INLIER_M of the map
    rmse: float  # metres, over those
    coarse: float  # the coarse search's overlap score

    @property
    def ok(self) -> bool:
        return self.fitness >= MIN_FITNESS and self.inliers >= MIN_INLIERS


def icp(source: np.ndarray, target: np.ndarray, start: np.ndarray) -> np.ndarray:
    """Point-to-point ICP with scale from `start`: Umeyama on the nearest neighbours within a tightening reach."""
    from scipy.spatial import KDTree

    tree = KDTree(target)
    transform = start.copy()
    for i in range(ICP_ITERS):
        reach = ICP_REACH[min(i * len(ICP_REACH) // ICP_ITERS, len(ICP_REACH) - 1)]
        moved = apply(transform, source)
        dist, idx = tree.query(moved, distance_upper_bound=reach)
        near = np.isfinite(dist)
        if near.sum() < 30:
            break
        step = umeyama(moved[near], target[idx[near]])
        transform = step @ transform
        if np.allclose(step, np.eye(4), atol=1e-6):
            break
    return transform


def register(source: np.ndarray, target: np.ndarray, seed: int = 0, start: np.ndarray | None = None) -> Registration:
    """`source` registered onto `target` (both levelled clouds): coarse yaw and shift, then ICP with scale. A `start`
    given (`seat`) replaces the coarse search, whose score is then NaN."""
    from scipy.spatial import KDTree

    rng = np.random.default_rng(seed)
    src, tgt = _thin(source, rng), _thin(target, rng)
    start, score = coarse(src, tgt) if start is None else (start, math.nan)
    transform = icp(src, tgt, start)
    moved = apply(transform, src)
    dist, _ = KDTree(tgt).query(moved, distance_upper_bound=INLIER_M)
    near = np.isfinite(dist)
    # Only the part of the new cloud over the map's footprint can fit it: a visit that saw more than the map holds
    # isn't penalised for the rest.
    footprint = {
        (x + dx, y + dy)
        for x, y in map(tuple, np.floor(tgt[:, :2] / COARSE_CELL).astype(np.int64).tolist())
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
    }
    cells = np.floor(moved[:, :2] / COARSE_CELL).astype(np.int64)
    over = np.fromiter(((int(x), int(y)) in footprint for x, y in cells), bool, count=len(cells))
    fitness = float(near[over].mean()) if over.any() else 0.0
    rmse = float(np.sqrt(np.mean(dist[near] ** 2))) if near.any() else math.inf
    return Registration(transform, fitness, int(near.sum()), rmse, score)


def _cloud(state: Path, rev: places.Revision) -> ply.Cloud | None:
    if rev.map is None:
        return None
    path = places.root(state) / rev.place / rev.map
    if not path.exists():
        return None
    return ply.read(path)


def fit(state: Path, fused_rev: places.Revision, levelled: np.ndarray) -> Registration | None:
    """How a levelled cloud (floor at z = 0, metres) registers on a place's fused map, without writing anything:
    `fuse_in`'s own registration (`register`, kept when `Registration.ok`: MIN_FITNESS of the cloud over the map's
    footprint within INLIER_M of it, and MIN_INLIERS points), for choosing which place a walk belongs to
    (walkthrough.py `match_place`). It leaves out `fuse_in`'s metric rescale, which the fusion that follows applies.
    None when the map can't be read."""
    old = _cloud(state, fused_rev)
    if old is None:
        return None
    return register(np.asarray(levelled, dtype=np.float64).reshape(-1, 3), _fused(old).points)


def _floor(points: np.ndarray, rev: places.Revision) -> tuple[float, ...] | None:
    """The revision's floor plane, else one fitted to the cloud (up taken as MASt3R's -y), else None."""
    if rev.floor is not None:
        return rev.floor
    try:
        n, d = floorplan.floor_plane(points, (0.0, -1.0, 0.0), 0.03, np.random.default_rng(0))
    except ValueError:
        return None
    return (*(float(x) for x in n), -float(d))


def _moved(poses: Mapping[str, places.CameraPose], transform: np.ndarray) -> dict[str, places.CameraPose]:
    """The poses through `transform`, kept rigid: a registration's scale moves a camera's position but not its axes
    (the stored 3x3 stays a rotation, so its up axis and any later registration read it as one)."""
    scale = float(np.cbrt(np.linalg.det(transform[:3, :3])))
    out = {}
    for cam, pose in poses.items():
        m = transform @ np.asarray(pose.pose, dtype=np.float64).reshape(4, 4)
        m[:3, :3] /= scale
        out[cam] = replace(pose, pose=tuple(float(x) for x in m.ravel()))
    return out


@dataclass(frozen=True)
class Fused:
    """A fused map: one point per VOXEL cube, its colour, how many fusions saw it and missed it, and how many of those
    were LiDAR walks (`walked`; None for none)."""

    points: np.ndarray
    colors: np.ndarray | None
    seen: np.ndarray
    missed: np.ndarray
    dropped: int = 0  # voxels this fusion faded out
    walked: np.ndarray | None = None

    @property
    def walks(self) -> np.ndarray:
        """How many LiDAR walks saw each voxel."""
        return np.zeros(len(self.points), np.int64) if self.walked is None else self.walked

    def counts(self) -> dict[str, np.ndarray]:
        """The per-voxel counts as the map's PLY keeps them."""
        out = {"seen": self.seen, "missed": self.missed}
        if self.walked is not None and self.walked.any():
            out["walked"] = self.walked
        return out


def _column(keys: np.ndarray) -> np.ndarray:
    """A top-down column's id, for a set test."""
    return keys[:, 0] * (1 << 32) + keys[:, 1]


def _cell(keys: np.ndarray) -> np.ndarray:
    """A voxel's id, for a set test (each key within 2**20 cells of the origin, 31 km at VOXEL)."""
    return keys[:, 0] * (1 << 42) + keys[:, 1] * (1 << 21) + keys[:, 2]


#: Cells' edges are moved down by this share of a cell (3 micrometres at VOXEL) when points are binned. A voxel's
#: position is the running mean of its points, and a mean of points on a cell's lower edge can round a hair below it
#: (1.65 m, 55 cells exactly, averaged back as 1.6499999999999997 m), putting the voxel in the cell next door on the
#: next fusion; so can the float32 the map is stored in. Snapping keeps it in its own cell.
SNAP = 1e-4


def _keys(points: np.ndarray, size: float) -> np.ndarray:
    """Each point's voxel: the cell it's in, with cells' edges moved down by SNAP of a cell."""
    return np.floor(points / size + SNAP).astype(np.int64)


def merge(old: Fused, new_points: np.ndarray, new_colors: np.ndarray | None, walked: bool = False) -> Fused:
    """`new_points` (registered, in the map's frame) fused into `old`, voxel by voxel; `walked` for a LiDAR walk's
    cloud (a Stray Scanner export's).

    A voxel the new cloud has is seen once more, its position and colour averaged in by how often it was seen. A
    voxel of the map with no point of the new cloud within one cell (in it or in one of its 26 neighbours, since
    MASt3R's noise of about a voxel puts a surface that's still there in the next cell over as often as not), but
    whose top-down COLUMN the new cloud covers (it was in view: the new visit saw the floor or something else there),
    is missed once more (a neighbour's point only spares it the miss: it isn't averaged in or counted as seen). A
    voxel missed at least twice, and more often than seen, is dropped: a person on the couch, a chair since moved, a
    bag on the floor fade out of the map after a couple of visits that see the place without them. A voxel outside
    what a visit saw is left alone, so rooms only a walkthrough covered stay while the fixed cameras watch one room
    each day. A column test can't tell a surface hidden behind something nearer from one that's gone, so a floor
    under a table can fade too; the plan's floor comes from the plane fit, which doesn't need every patch.

    A voxel a LiDAR walk saw (`Fused.walked`) is missed only by another walk: a fixed camera's MASt3R snapshot is
    softer than the walk by several centimetres, so missing the walk's walls by that much would fade the sharp map
    into the soft one night by night. Changes there come in with the next walk.
    """
    new_pts, new_cols = voxelize(new_points, new_colors, VOXEL)
    o = len(old.points)
    uniq, inverse = np.unique(np.r_[_keys(old.points, VOXEL), _keys(new_pts, VOXEL)], axis=0, return_inverse=True)
    inverse = inverse.ravel()
    io, inn = inverse[:o], inverse[o:]
    m = len(uniq)
    seen = np.zeros(m)
    np.add.at(seen, io, old.seen)
    weight = seen.copy()
    pos = np.zeros((m, 3))
    np.add.at(pos, io, old.points * old.seen[:, None])
    np.add.at(seen, inn, 1.0)
    np.add.at(pos, inn, new_pts)
    pos /= np.maximum(seen, 1)[:, None]
    colors = None
    if old.colors is not None and new_cols is not None:
        csum = np.zeros((m, 3))
        np.add.at(csum, io, old.colors.astype(np.float64) * old.seen[:, None])
        np.add.at(csum, inn, new_cols.astype(np.float64))
        colors = np.clip(np.round(csum / np.maximum(seen, 1)[:, None]), 0, 255).astype(np.uint8)
    missed = np.zeros(m)
    np.add.at(missed, io, old.missed)
    walks = np.zeros(m)
    np.add.at(walks, io, old.walks)
    can_miss = np.ones(m, bool) if walked else walks == 0
    if walked:
        np.add.at(walks, inn, 1.0)
    in_old = weight > 0
    view = np.unique(_keys(new_pts[:, :2], COLUMN), axis=0)
    cols_of = _keys(pos[:, :2], COLUMN)
    in_view = np.isin(_column(cols_of), _column(view))
    new_cells = np.unique(_cell(_keys(new_pts, VOXEL)))
    near = np.zeros(m, bool)
    for step in itertools.product((-1, 0, 1), repeat=3):
        near |= np.isin(_cell(uniq + np.array(step)), new_cells)
    missed[in_old & ~near & in_view & can_miss] += 1
    keep = ~((missed >= 2) & (missed > seen))
    return Fused(
        pos[keep],
        colors[keep] if colors is not None else None,
        seen[keep].astype(np.int64),
        missed[keep].astype(np.int64),
        int((~keep).sum()),
        walks[keep].astype(np.int64),
    )


#: metres: where a LiDAR walk saw a top-down COLUMN, a camera cloud's points there are fused only this near the walk's
#: voxels (`gate`): two voxels, past MASt3R's surface noise of about one.
GATE_M = 0.06


def gate(fused: Fused, points: np.ndarray) -> np.ndarray:
    """Which of a camera cloud's `points` (placed in the map) to fuse into `fused`: all of them where no LiDAR
    walk saw (no walked voxel in their top-down COLUMN), and those within GATE_M of a walked voxel where one did. A
    walk is sharper than a fixed camera's MASt3R snapshot, so where both saw, the walk holds; a snapshot's wall a few
    centimetres off the walk's would otherwise thicken it into two."""
    from scipy.spatial import KDTree

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    walked = fused.points[fused.walks > 0]
    keep = np.ones(len(pts), bool)
    if not len(walked) or not len(pts):
        return keep
    covered = np.isin(_column(_keys(pts[:, :2], COLUMN)), np.unique(_column(_keys(walked[:, :2], COLUMN))))
    if covered.any():
        d, _ = KDTree(walked).query(pts[covered], distance_upper_bound=GATE_M)
        keep[covered] = np.isfinite(d)
    return keep


def _fused(cloud: ply.Cloud) -> Fused:
    """A fused map as read back (a map from before the counts were kept counts every voxel as seen once)."""
    pts, cols, counts = cloud
    ones, zeros = np.ones(len(pts), np.int64), np.zeros(len(pts), np.int64)
    return Fused(pts, cols, counts.get("seen", ones), counts.get("missed", zeros), walked=counts.get("walked"))


def relevel(
    fused: Fused, cams: Mapping[str, places.CameraPose], place: str = ""
) -> tuple[Fused, dict[str, places.CameraPose]]:
    """The fused map and its cameras turned so the map's own floor is level again (z = 0, up +z), when it has drifted
    more than RELEVEL_DEG off; else as they are. Fusions register only in yaw, but a map that drifted before
    that (Place 1: 1.59 degrees), or one whose first cloud was levelled by a poor floor, is put right here, with its
    camera poses moved by the same rigid turn so they stay registered in it."""
    try:
        normal, d = floorplan.floor_plane(fused.points, (0.0, 0.0, 1.0), 0.03, np.random.default_rng(0))
    except ValueError:
        return fused, dict(cams)
    tilt = math.degrees(math.acos(min(1.0, abs(float(normal[2])))))
    if tilt <= RELEVEL_DEG:
        return fused, dict(cams)
    # A floor has nothing under it. A plane with much of the map below it is a table or dresser top, or a slice through
    # the walls: one camera's view of a room can hold more furniture top than floor, and turning the map onto
    # such a plane would tip a correctly levelled map over.
    below = float(np.mean(fused.points @ normal - d < -UNDER_FLOOR_M))
    if below > UNDER_FLOOR_SHARE:
        LOG.info(
            "%s: kept the floor: the plane %.2f degrees off level has %.0f%% of the map under it",
            place,
            tilt,
            100 * below,
        )
        return fused, dict(cams)
    turn = leveller((*(float(x) for x in normal), -float(d)))
    LOG.info("%s: the fused map's floor was %.2f degrees off level; turned level", place, tilt)
    return replace(fused, points=apply(turn, fused.points)), _moved(cams, turn)


def walking(seen: Sequence[footprints.Sighting]) -> list[footprints.Sighting]:
    """The sightings of a person walking upright (`Sighting.upright`: feet, hips, and head placed, legs long enough):
    the same track a sample or two later with its feet moved at least STEP across the frame."""
    last: dict[tuple[str, int], footprints.Sighting] = {}
    out = []
    for s in sorted(seen, key=lambda s: s.t):
        before = last.get((s.camera, s.track))
        last[(s.camera, s.track)] = s
        if before is None or s.head is None or not s.upright:
            continue
        if (
            footprints.SAMPLE_S <= s.t - before.t <= 2 * footprints.SAMPLE_S
            and math.hypot(s.u - before.u, s.v - before.v) > STEP
        ):
            out.append(s)
    return out


LEVEL_FLOOR = (0.0, 0.0, 1.0, 0.0)  # a levelled cloud's floor


def stature(cam: places.CameraPose, s: footprints.Sighting) -> float | None:
    """How tall a sighted person stands in a levelled cloud's units (floor at z = 0): the height of the point on
    their head's ray nearest the vertical through their feet' floor point. None without a head or a floor point."""
    feet = places.floor_point(cam, LEVEL_FLOOR, s.u, s.v, footprints.FOOT_M)
    if feet is None or s.head is None:
        return None
    m = np.asarray(cam.pose, dtype=np.float64).reshape(4, 4)
    hu, hv = places.undistorted(cam, *s.head)  # a wide lens undone, as floor_point does the feet
    d = np.array([(hu - 0.5) * cam.width / cam.focal, (hv - 0.5) * cam.height / cam.focal, 1.0])
    ray, origin = m[:3, :3] @ d, m[:3, 3]
    w = origin - np.asarray(feet)
    a, b = float(ray @ ray), float(ray[2])
    if a - b * b < 1e-12:
        return None  # a ray straight down
    t = (b * float(w[2]) - float(ray @ w)) / (a - b * b)
    return float(origin[2] + t * ray[2])


def people(seen: Sequence[footprints.Sighting], cams: Mapping[str, places.CameraPose]) -> tuple[float, int] | None:
    """Walking people's median height in a levelled cloud's units and how many were measured; None for too few."""
    heights = [h for s in walking(seen) if s.camera in cams and (h := stature(cams[s.camera], s)) is not None]
    heights = [h for h in heights if h > 0]
    return (float(np.median(heights)), len(heights)) if len(heights) >= MIN_WALKERS else None


def metric(
    cams: Mapping[str, places.CameraPose],
    what: str,
    tolerance: float = 0.0,
    seen: Sequence[footprints.Sighting] = (),
) -> np.ndarray:
    """The 4x4 scale about the origin that puts `cams` (in a levelled cloud's frame, floor at z = 0) at the heights
    the owner measured (each fixed camera's `mount.height_m`, spatial/rig.py): the median of measured over found height across the cameras
    with one. The identity with no such camera, or when the scale is within `tolerance` of 1. Logs each camera's own
    ratio, and walking people's height at the scale (`seen`), as checks."""
    measured = rig.load().heights()
    ratios = {
        c: float(measured[c]) / float(pose.pose[11])
        for c, pose in cams.items()
        if c in measured and float(measured[c]) > 0 and float(pose.pose[11]) > 0
    }
    if not ratios:
        return np.eye(4)
    k = float(np.median(list(ratios.values())))
    if abs(k - 1) <= tolerance:
        return np.eye(4)
    check = people(seen, cams)
    LOG.info(
        "%s: scaled by %.3f from camera heights (%s)%s",
        what,
        k,
        ", ".join(f"{c} {r:.3f}" for c, r in sorted(ratios.items())),
        f"; walking people then measure {k * check[0]:.2f} m ({check[1]})" if check else "",
    )
    out = np.eye(4)
    out[:3, :3] *= k
    return out


def _sightings(state: Path, cams: Sequence[str], now: float) -> list[footprints.Sighting]:
    return footprints.sightings(state, now - CALIBRATE_S, now - footprints.FLUSH_S, cams) if cams else []


def refine(state: Path, place: str, now: float | None = None) -> places.Revision | None:
    """Fuse `place`'s newest revision into its map and write the result as a new revision; None when there is
    nothing to fuse (the newest is tier 1's own, or has no map) or the new cloud doesn't register."""
    revs = places.revisions(state, place)
    if not revs or revs[-1].producer == PRODUCER:
        return None
    newest = revs[-1]
    loaded = _cloud(state, newest)
    if loaded is None:
        return None
    points, colors, _ = loaded
    floor = _floor(points, newest)
    if floor is None:
        LOG.warning("%s: the newest map has no floor to level it by; not fused", place)
        return None
    from liberational.spatial import walkthrough  # walkthrough fuses through this module

    walked = newest.producer == walkthrough.PRODUCER_STRAY
    return fuse_in(state, place, points, colors, floor, newest.cameras, [newest.id], walked=walked, now=now)


def fuse_in(
    state: Path,
    place: str,
    points: np.ndarray,
    colors: np.ndarray | None,
    floor: Sequence[float],
    cams: Mapping[str, places.CameraPose],
    sources: Sequence[str],
    *,
    seated: np.ndarray | None = None,
    signals: Mapping[str, Sequence[str]] | None = None,
    walked: bool = False,
    now: float | None = None,
) -> places.Revision | None:
    """A cloud (`points` in its own frame, with its `floor` and the cameras `cams` registered in it) fused into
    `place`'s map as a tier 1 revision: `refine`'s steps for a cloud that is a revision of the place (`sources` names
    it), or one that isn't written yet (a walkthrough into a place already there). `seated` is where the
    cloud's levelled frame stands in the fused map's frame when its caller found it another way (the fixed cameras
    both see, `pack-walkthrough`'s seats): ICP polishes it against the map around there (SEAT_PAD_M), kept when it
    registers within SEAT_SHIFT_M and SEAT_TURN_DEG of the seat, else the seat stands as found (a map too soft for a
    sharper cloud to fit, like a whole home's fixed-camera snapshots beside a LiDAR walk). `signals` replace the
    place's when given. `walked` marks a LiDAR walk's cloud (`merge`). The fused map is rescaled to its people's
    heights and turned level when its floor has drifted, each of which gives the revision a frame of its own, except
    while the owner's plan is drawn on the map (`owner_drawn`): then the frame holds. None when the cloud can't be
    put at a metric scale or doesn't register."""
    level = leveller(floor)
    fused_rev = fused_revision(state, place)
    at = time.time() if now is None else now
    seen = _sightings(state, sorted({*cams, *(fused_rev.cameras if fused_rev else {})}), at)
    scale = metric(_moved(cams, level), f"{place}'s new map", seen=seen)
    if not cameras.SCALE_BOUNDS[0] <= scale[0, 0] <= cameras.SCALE_BOUNDS[1]:
        # A scale this far from 1 is a wrong floor or a broken cloud, not MASt3R's scale (8.3 on a kitchen)
        LOG.warning("%s: the new map would be scaled by %.2f to its cameras' heights; not fused", place, scale[0, 0])
        return None
    level = scale @ level
    new_pts = apply(level, np.asarray(points, dtype=np.float64).reshape(-1, 3))
    sources = list(sources)
    frame: str | None = None  # a new frame (the new revision's own) unless the fused map's is kept
    if fused_rev is None:
        transform = level
        pts, cols = voxelize(new_pts, colors, VOXEL)
        ones = np.ones(len(pts), np.int64)
        fused = Fused(pts, cols, ones, np.zeros(len(pts), np.int64), walked=ones if walked else None)
        moved = _moved(cams, transform)
        LOG.info("%s: the first map (%d points)", place, len(pts))
    else:
        old = _cloud(state, fused_rev)
        assert old is not None
        # Only the cameras the newest revision registered itself are in its frame: one carried over from an earlier
        # revision (places.revise carries cameras when none are given) is the fused map's own, already in place.
        fresh = {c: p for c, p in cams.items() if fused_rev.cameras.get(c) != p}
        owners = owner_drawn(state, place, fused_rev)
        rescale = np.eye(4) if owners else metric(fused_rev.cameras, f"{place}'s fused map", RESCALE_TOL, seen)
        before = _fused(old)
        before = replace(before, points=apply(rescale, before.points))
        if seated is None:
            reg = register(new_pts, before.points)
            if not reg.ok:
                LOG.warning(
                    "%s: the new map didn't register (fitness %.2f over the map, needs %.2f; %d inliers, needs %d; "
                    "coarse %.2f); not fused",
                    place,
                    reg.fitness,
                    MIN_FITNESS,
                    reg.inliers,
                    MIN_INLIERS,
                    reg.coarse,
                )
                return None
            placed = reg.transform
            how = f"fitness {reg.fitness:.2f}, {reg.inliers} inliers, rmse {reg.rmse:.3f} m"
        else:
            placed, how = _polished(new_pts, before.points, rescale @ seated)
        transform = placed @ level
        fused = merge(before, apply(placed, new_pts), colors, walked)
        moved = {**_moved(fused_rev.cameras, rescale), **_moved(fresh, transform)}
        sources.append(fused_rev.id)
        # The new cloud is registered into the fused map's frame; rescaling the map moves it out.
        frame = fused_rev.frame if np.allclose(rescale, np.eye(4)) else None
        LOG.info(
            "%s: fused (%s, scale %.3f; %d points, %d faded)",
            place,
            how,
            float(np.cbrt(np.linalg.det(placed[:3, :3]))),
            len(fused.points),
            fused.dropped,
        )
    levelled = fused
    if not (fused_rev is not None and owner_drawn(state, place, fused_rev)):
        levelled, moved = relevel(fused, moved, place)
    if levelled is not fused:
        frame = None  # turned level: a frame of its own
    return _revise(state, place, levelled, moved, sources, frame, at, now, signals)


def _polished(new_pts: np.ndarray, map_pts: np.ndarray, start: np.ndarray) -> tuple[np.ndarray, str]:
    """`start` (a levelled cloud's seat in a map, `fuse_in`'s `seated`) polished by ICP against the map around where
    it puts the cloud, when that registers within SEAT_SHIFT_M and SEAT_TURN_DEG of it; else `start`. With what
    happened, for the log."""
    placed = apply(start, new_pts)
    lo, hi = placed.min(axis=0) - SEAT_PAD_M, placed.max(axis=0) + SEAT_PAD_M
    around = map_pts[np.all((map_pts >= lo) & (map_pts <= hi), axis=1)]
    if len(around) == 0:
        return start, "seated by its cameras; the map has nothing there to polish it against"
    reg = register(new_pts, around, start=start)
    centre = new_pts.mean(axis=0)
    shift = float(np.linalg.norm(apply(reg.transform, centre[None]) - apply(start, centre[None])))
    turn = abs(math.degrees(_yaw(reg.transform[:3, :3] @ np.linalg.inv(start[:3, :3]))))
    if reg.ok and shift <= SEAT_SHIFT_M and turn <= SEAT_TURN_DEG:
        return reg.transform, (
            f"seated by its cameras, polished: fitness {reg.fitness:.2f}, {reg.inliers} inliers, "
            f"{shift:.2f} m and {turn:.1f} degrees off the seat"
        )
    return start, (
        f"seated by its cameras; the map around it didn't polish it (fitness {reg.fitness:.2f}, needs "
        f"{MIN_FITNESS:.2f}; {reg.inliers} inliers; {shift:.2f} m and {turn:.1f} degrees off the seat), so the seat "
        "stands"
    )


def _revise(
    state: Path,
    place: str,
    fused: Fused,
    cams: Mapping[str, places.CameraPose],
    sources: Sequence[str],
    frame: str | None,
    at: float,
    now: float | None,
    signals: Mapping[str, Sequence[str]] | None = None,
    plan_frame: Sequence[float] | None = None,
) -> places.Revision:
    """Tier 1's revision of `place`: the fused map with its counts, its plan derived again, the floor at z = 0, and
    `cams` registered in it; `signals` replace the place's when given. `plan_frame` is the map-to-plan transform it
    keeps when no plan is derived (`join`, for an owner plan carried in). No plan is derived while the owner's plan
    is drawn on this map's frame with no plan file (an imported one): a machine plan file would make the
    revision a plan frame of its own (`places.same_plan_frame`), so positions and the 3D view would stop drawing on
    the owner's plan, which holds over it anyway."""
    known = places.get(state, place)
    plan_text, transform = None, plan_frame
    if not _owner_frame(state, known, frame):
        # Where people walked: the fused cameras' recorded footprints, cast onto the fused floor through their
        # poses in this frame, close rooms whose walls the clouds don't hold.
        feet = footprints.walked(state, sorted(cams), at) if cams else []  # counted cells of each frame
        walked = (lambda floor: footprints.project(feet, cams, floor)) if feet else None
        # The fused map stands on its floor at z = 0 (levelled), so that is its floor: a fit to the cloud can take a
        # flat furniture top that outnumbers the floor, as `relevel` refuses to (Place 2's sat 0.88 m up).
        name = (known.name if known else "") or "Place"
        found = cameras.layout(fused.points, (0.0, 0.0, 1.0), name, walked, floor=LEVEL_FLOOR)
        if found.plan is not None:
            plan_text, transform = found.plan, found.transform
    return places.revise(
        state,
        place,
        producer=PRODUCER,
        sources=sources,
        signals=signals,
        map_bytes=ply.dumps(fused.points, fused.colors, fused.counts()),
        map_format="ply",
        plan=plan_text,
        plan_transform=transform,
        floor=LEVEL_FLOOR,
        cameras=cams,
        now=now,
        frame=frame,
    )


def _owner_frame(state: Path, known: places.Place | None, frame: str | None) -> bool:
    """Whether the owner's plan holds for `known` drawn on the map frame `frame` with no plan file (`_revise`)."""
    if known is None or known.plan_source != "owner" or frame is None or not known.plan_revision:
        return False
    drawn = next((r for r in places.revisions(state, known.id) if r.id == known.plan_revision), None)
    return drawn is not None and drawn.plan is None and drawn.frame == frame and drawn.plan_transform is None


def owner_drawn(state: Path, place: str, fused_rev: places.Revision) -> bool:
    """Whether the owner's plan is drawn on `place`'s fused map (`_owner_frame`, on `fused_rev`'s frame). That frame
    is then the owner's, and no fusion moves the map out of it (#215): the map isn't rescaled to its people's heights
    or turned level, either of which starts a frame of its own, where a machine plan would be derived and the owner's
    rooms, drawn in the frame left behind, would no longer lie on the newest map, its cameras, or its positions. A
    floor half a degree off under a plan the owner drew is the smaller wrong."""
    return _owner_frame(state, places.get(state, place), fused_rev.frame)


#: How far (metres) a fixed camera may stand from where it stood on the owner's frame for `onto_owner_plan` to take
#: the newest map as still being that map: a levelling turn of a degree or a rescale of a few percent moves a camera
#: centimetres, and anything past this is another registration, which no relabelling puts right.
BACK_M = 0.25


def _frame_hops(chain: Sequence[places.Revision], owned: str, fused_rev: places.Revision) -> list[float] | None:
    """How far the fused map's frame moved between the owner's (`owned`) and `fused_rev`'s: for each change of frame
    along the tier 1 revisions after the last one in the owner's frame, the farthest a fixed camera stands (metres)
    from where the revision before had it. A frame moves only at the revision that starts it, so that is where it's
    measured: later revisions in a frame add nothing, though a nightly seat re-poses the cameras in it by decimetres
    of its own (#227). Each hop's figure includes what that fusion did to its own cameras, so the sum bounds the move
    from above. None when a hop has no fixed camera on both sides."""
    at = max((i for i, r in enumerate(chain) if r.frame == owned), default=None)
    if at is None:
        return None
    hops: list[float] = []
    before = chain[at]
    for rev in chain[at + 1 :]:
        if rev.producer != PRODUCER or rev.frame is None:
            continue  # a snapshot or a walk's own revision: not the fused map
        if rev.frame != before.frame:
            shared = [c for c in set(before.cameras) & set(rev.cameras) if not c.startswith("walk-")]
            if not shared:
                return None
            hops.append(max(float(np.linalg.norm(_centre(rev, c) - _centre(before, c))) for c in shared))
        before = rev
        if rev.id == fused_rev.id:
            break
    return hops


def _centre(rev: places.Revision, cam: str) -> np.ndarray:
    return np.asarray(rev.cameras[cam].pose, dtype=np.float64).reshape(4, 4)[:3, 3]


def onto_owner_plan(
    state: Path, place: str, now: float | None = None, write: bool = True
) -> tuple[places.Revision | None, str]:
    """The repair for a place whose map a fusion moved off its owner's plan before `owner_drawn` held the frame
    (#215): the newest fused map written again as a tier 1 revision in the frame the owner's plan is drawn in, with
    its cameras and signals as they are and no machine plan, so the owner's plan holds on the newest map again
    (`places.same_plan_frame`), unchanged. Everything a walk added to the map since stays. The map isn't moved back:
    what moved it was a small levelling turn or rescale, so across each change of frame since the owner's the fixed
    cameras must have moved no more than `BACK_M` in all (`_frame_hops`), and a map that moved farther is refused.
    Fusions that kept a frame (a nightly seat, which re-poses its cameras) don't count: the newest cameras are
    written as they are. `write` False only says what it would do. Returns the new revision (None when nothing was written) and
    what it found."""
    known = places.get(state, place)
    fused_rev = fused_revision(state, place)
    if known is None or fused_rev is None:
        return None, "no such place, or it has no fused map"
    if known.plan_source != "owner" or not known.plan or not known.plan_revision:
        return None, "the place has no owner plan to restore"
    if _owner_frame(state, known, fused_rev.frame):
        return None, "the owner's plan is already drawn on the newest map: nothing to do"
    drawn = next((r for r in places.revisions(state, place) if r.id == known.plan_revision), None)
    if drawn is None or drawn.frame is None or drawn.plan is not None or drawn.plan_transform is not None:
        return (
            None,
            "the owner's plan isn't drawn on a map frame (it corrects a machine plan), so there is no frame to restore",
        )
    hops = _frame_hops(places.revisions(state, place), drawn.frame, fused_rev)
    if hops is None:
        return None, "no fixed camera is on both sides of a change of the map's frame, so the move can't be checked"
    moved = sum(hops)
    found = (
        f"the map changed frame {len(hops)} time{'s' if len(hops) != 1 else ''} since the owner's, moving its fixed "
        f"cameras {moved:.3f} m at most"
    )
    if moved > BACK_M:
        return None, f"{found}, past the {BACK_M} m a levelling turn or rescale moves them: not restored"
    if not write:
        return None, f"{found}: the newest map would be written again in the owner's frame"
    old = _cloud(state, fused_rev)
    if old is None:
        return None, "the newest fused map's file is gone"
    at = time.time() if now is None else now
    newest = places.latest(state, place) or fused_rev
    rev = _revise(state, place, _fused(old), fused_rev.cameras, [fused_rev.id], drawn.frame, at, now, newest.signals)
    LOG.info("%s: its newest map written again in the owner plan's frame (%s)", place, found)
    return rev, f"{found}: written as revision {rev.id}"


def refine_all(state: Path, now: float | None = None) -> list[places.Revision]:
    """`refine` for every place: the daily catch-up for a revision missed on the way."""
    out = []
    for place in places.place_ids(state):
        try:
            rev = refine(state, place, now)
        except (ValueError, np.linalg.LinAlgError) as e:
            LOG.warning("%s: tier 1 failed: %s", place, e)
            continue
        if rev is not None:
            out.append(rev)
    return out


def fused_revision(state: Path, place: str) -> places.Revision | None:
    """`place`'s fused map's revision: its newest tier 1 revision whose map is still there; None before one."""
    return next(
        (r for r in reversed(places.revisions(state, place)) if r.producer == PRODUCER and _cloud(state, r)), None
    )


def join(
    state: Path,
    into: str,
    others: Mapping[str, np.ndarray],
    now: float | None = None,
) -> places.Revision:
    """Merge the places `others` into `into`: each other place's fused map, moved into `into`'s fused map's
    frame by its 4x4 transform in `others` (from the other place's frame to `into`'s), fused into `into`'s map voxel by
    voxel (`merge`). The new revision is tier 1's, so later fusions build on it: the merged map with its counts, its
    plan derived again, the floor at z = 0, every place's cameras (each other place's moved rigidly, as `refine` moves
    them), and every place's signals joined. It keeps `into`'s frame, since nothing in it moved. Each other place is
    then retired into `into` (`places.retire`). The owner's name for `into` holds; when `into` has none and the other
    places' names agree on one, it takes that name. Raises KeyError when a place has no fused map."""
    base = fused_revision(state, into)
    if base is None:
        raise KeyError(f"{into} has no fused map to merge into")
    old = _cloud(state, base)
    assert old is not None
    fused = _fused(old)
    sources, cams = [base.id], dict(base.cameras)
    signals = {k: list(v) for k, v in base.signals.items()}
    # The owner's plans, read before any place is retired (a retired place isn't listed)
    held = {p.id: p for p in places.places(state)}
    carried: list[_Carry] = []
    for place, transform in others.items():
        rev = fused_revision(state, place)
        cloud = _cloud(state, rev) if rev is not None else None
        if rev is None or cloud is None:
            raise KeyError(f"{place} has no fused map to merge")
        if (owned := _owner_plan(state, held.get(place), rev, transform)) is not None:
            carried.append(owned)
        fused = merge(fused, apply(transform, cloud[0]), cloud[1])
        cams.update(_moved(rev.cameras, transform))
        for key, values in rev.signals.items():
            have = signals.setdefault(key, [])
            have += [v for v in values if v not in have]
        sources.append(rev.id)
    at = time.time() if now is None else now
    target = held.get(into)
    owner_frame = target is not None and target.plan_source == "owner"
    frame = None if owner_frame or not carried else _plan_frame(carried)
    rev = _revise(state, into, fused, cams, sources, base.frame, at, now, signals, frame)
    if carried and target is not None:
        _carry_plans(state, target, rev, carried)
    named = {p.id: p.name for p in places.places(state) if p.named}
    if into not in named:
        given = {named[p] for p in others if p in named}
        if len(given) == 1:
            places.name(state, into, given.pop(), now=now, source="merge")
    for place in others:
        places.retire(state, place, into, now)
    LOG.info(
        "%s: merged %s into it (%d points, %d cameras)", into, ", ".join(sorted(others)), len(fused.points), len(cams)
    )
    return rev


#: A carried owner plan's turn is rounded to the nearest quarter turn when it is within this (degrees) of one: rooms of
#: one home share their walls' directions, and every plan is drawn square to its walls, so what is left over is the
#: join's own error (the walk's seats agree within 15 degrees).
PLAN_SNAP_DEG = 10.0
#: Two rooms clash when one covers at least this share of the other (`_clash`): a carried room that clashes with the
#: joined place's own owner rooms is left out, kept as comments (the owner's rooms of the place joined into hold),
#: and a machine room that clashes with carried owner rooms is dropped (taught beats made; an owner who drew one room
#: where the machine had a bigger one corrected that whole room).
COVERED_SHARE = 0.5


def _clash(
    poly: Sequence[tuple[float, float]], rooms: Sequence[tuple[str, str, Sequence[tuple[float, float]]]]
) -> bool:
    """Whether `poly` clashes with any of `rooms` (COVERED_SHARE)."""
    return floorplan.covered(poly, rooms) >= COVERED_SHARE or any(
        floorplan.covered(other, [("p", "", poly)]) >= COVERED_SHARE for _, _, other in rooms
    )


#: The plan frame of a levelled map (floor at z = 0): plan (x, y, height) = map (x, y, z).
LEVEL_PLAN = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def _plan_matrix(transform: Sequence[float]) -> np.ndarray:
    """A revision's 12-number map-to-plan transform as a 4x4."""
    out = np.eye(4)
    out[:3] = np.asarray(transform, dtype=np.float64).reshape(3, 4)
    return out


def frame_between(src: places.Revision, dst: places.Revision) -> np.ndarray | None:
    """The 4x4 that takes a point of `src`'s map into `dst`'s, for two revisions of one place: the identity when they
    stand in one frame (`Revision.frame`), else the turn about the floor's normal, scale, and shift that put the
    cameras both registered where `dst` has them (`by_cameras`, after levelling each map by its floor). None without
    a camera in both, or a floor in each."""
    if src.id == dst.id or (src.frame is not None and src.frame == dst.frame):
        return np.eye(4)
    shared = sorted(set(src.cameras) & set(dst.cameras))
    if not shared or src.floor is None or dst.floor is None:
        return None
    lift, land = leveller(src.floor), leveller(dst.floor)
    found = by_cameras(
        _moved({c: src.cameras[c] for c in shared}, lift),
        _moved({c: dst.cameras[c] for c in shared}, land),
        bounds=None,
    )
    return None if found is None else np.linalg.inv(land) @ found @ lift


def plan_move(src: Sequence[float], to_map: np.ndarray, dst: Sequence[float]) -> floorplan.Similarity | None:
    """The move from one plan frame to another on the floor: a point of the plan drawn by map-to-plan `src` back to
    its map, through the 4x4 `to_map` into the other map, and onto the plan drawn there by `dst`. None when that
    mirrors the plan (it never should: every frame here is right-handed)."""
    full = _plan_matrix(dst) @ to_map @ np.linalg.inv(_plan_matrix(src))
    o, ex, ey = apply(full, np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    ex, ey = ex[:2] - o[:2], ey[:2] - o[:2]
    if ex[0] * ey[1] - ex[1] * ey[0] <= 0:
        return None
    scale = (float(np.linalg.norm(ex)) + float(np.linalg.norm(ey))) / 2
    return floorplan.Similarity(scale, math.atan2(float(ex[1]), float(ex[0])), (float(o[0]), float(o[1])))


@dataclass(frozen=True)
class _Carry:
    """A merged-away place's owner plan, on its way into the joined place (`join`)."""

    name: str  # the place's name, as the owner sees it
    plan: str  # the owner's plan text
    plan_frame: tuple[float, ...] | None  # the map-to-plan transform it is drawn by
    to_joined: np.ndarray | None  # that map into the joined map, or None when its frame can't be related


def _owner_plan(state: Path, held: places.Place | None, fused: places.Revision, transform: np.ndarray) -> _Carry | None:
    """A place's owner plan correction, with how its frame goes into the joined map (`transform` from its fused map
    there); None when it has none."""
    if held is None or held.plan_source != "owner" or not held.plan:
        return None
    drawn = next((r for r in places.revisions(state, held.id) if r.id == held.plan_revision), None)
    into_fused = frame_between(drawn, fused) if drawn is not None else None
    # An owner plan drawn on a map with no plan file of its own (an imported one) is in that map's floor frame
    # (`places.plan_frame`), not in no frame
    return _Carry(
        held.name,
        held.plan,
        places.plan_frame(drawn) if drawn is not None else None,
        None if into_fused is None else transform @ into_fused,
    )


def _plan_frame(carried: Sequence[_Carry]) -> tuple[float, ...] | None:
    """A plan frame for a joined map that may get no plan of its own: the first placeable carried plan's own frame
    in the joined map, turned as that plan is (so its walls stay square to the page) and shifted so its rooms start at
    the origin, as a derived plan's do. Metres of the joined map, whatever scale the carried plan was drawn at."""
    for c in carried:
        if c.plan_frame is None or c.to_joined is None:
            continue
        move = plan_move(c.plan_frame, c.to_joined, LEVEL_PLAN)  # the carried plan onto the joined map's floor
        if move is None:
            continue
        cos, sin = math.cos(-move.turn), math.sin(-move.turn)
        upright = floorplan.Similarity(1.0, -move.turn, (0.0, 0.0))
        corners = [upright.point(*move.point(x, y)) for _, _, poly in floorplan.rooms_from_dsl(c.plan) for x, y in poly]
        if corners:
            lo = (min(x for x, _ in corners), min(y for _, y in corners))
            return (cos, -sin, 0.0, -lo[0], sin, cos, 0.0, -lo[1], 0.0, 0.0, 1.0, 0.0)
    return None


def _carry_plans(state: Path, target: places.Place, joined: places.Revision, carried: Sequence[_Carry]) -> None:
    """The merged places' owner plans added to the joined place's plan, as a correction of it (source "merge"):
    each moved into the frame of the plan that holds there (the joined place's own correction's, else the joined
    revision's), turned to the nearest quarter turn (PLAN_SNAP_DEG), its ids renamed `m<n>_<id>` with n the first that
    clashes with no id the plan has. Where the joined place has its own owner rooms, a carried room that clashes with
    them (`_clash`) is left out as comments; where it has only the machine's, a machine room that clashes with the
    carried rooms is dropped. What can't be moved stays in as comments too (`floorplan.carry`)."""
    # Read again: the owner may have saved an edit while the join ran, and this correction replaces what holds
    target = places.get(state, target.id) or target
    chain = {r.id: r for r in places.revisions(state, target.id)}
    owner = target.plan_source == "owner"
    frame = chain.get(target.plan_revision or "") if owner else joined
    if frame is None:
        frame = joined
    base = (target.plan or "") if owner else (places.machine_plan(state, joined) or "")
    to_frame = frame_between(joined, frame)
    frame_plan = places.plan_frame(frame)  # an owner plan drawn on a plan-less map is in its floor's frame
    mine = floorplan.rooms_from_dsl(base)
    # Ids already taken: a place joined before holds the m<n>_ ids its carried plans got, and the floorplan library
    # keeps only the last of two rooms with one id
    taken = floorplan.plan_ids(base)
    pieces: list[str] = []
    theirs: list[tuple[str, str, tuple[tuple[float, float], ...]]] = []
    for c in carried:
        move = None
        if c.plan_frame is not None and c.to_joined is not None and to_frame is not None and frame_plan is not None:
            move = plan_move(c.plan_frame, to_frame @ c.to_joined, frame_plan)
        corners = [p for _, _, poly in floorplan.rooms_from_dsl(c.plan) for p in poly]
        if move is None:
            LOG.warning("%s: %s's plan can't be placed in its frame; kept in its plan as comments", target.id, c.name)
            pieces.append(f"# {c.name}, merged into this place; its plan couldn't be placed here")
            pieces += [f"# from {c.name}: {line.strip()}" for line in c.plan.splitlines() if line.strip()]
            continue
        if corners:
            centre = (sum(x for x, _ in corners) / len(corners), sum(y for _, y in corners) / len(corners))
            move = move.snapped(centre, PLAN_SNAP_DEG)
        own = floorplan.plan_ids(c.plan)
        n = 1
        while any(f"m{n}_{i}" in taken for i in own):
            n += 1
        got = floorplan.carry(c.plan, move, f"m{n}", c.name)
        taken |= set(got.renamed.values())
        text = "\n".join(got.lines) + "\n"
        rooms = floorplan.rooms_from_dsl(text)
        left_out = [rid for rid, _, poly in rooms if owner and _clash(poly, mine)]
        # Kept as comments, as what can't be moved is: the owner drew them, and the joined place's own rooms hold
        text = floorplan.without(text, left_out, f"from {c.name}, where a room of this place is")
        theirs += [r for r in rooms if r[0] not in left_out]
        pieces += [f"# {c.name}, merged into this place", text.rstrip("\n")]
        LOG.info(
            "%s: carried %s's plan into it (%d rooms, %d left out where its own are, %d statements kept as comments; "
            "turned %.1f degrees, scale %.3f)",
            target.id,
            c.name,
            len(rooms) - len(left_out),
            len(left_out),
            got.kept_as_comments,
            math.degrees(move.turn),
            move.scale,
        )
    if not owner:
        made = [rid for rid, _, poly in mine if _clash(poly, theirs)]
        base = floorplan.without(base, made)
        if made:
            LOG.info("%s: %d machine rooms left out under the carried owner rooms", target.id, len(made))
    text = "\n\n".join(part for part in (base.rstrip("\n"), *pieces) if part) + "\n"
    # An owner plan shown as drawn on its page keeps its page, with the carried rooms drawn on it too
    view = None
    if owner and target.view is not None:
        added = "\n".join(pieces)
        page = target.view.plan.rstrip("\n") + "\n"
        drawn = _on_page(added, target.view) if added.strip() else ""
        view = {**target.view.as_dict(), "plan": page + ("\n" + drawn if drawn else "")}
    # Saved just after the joined revision, so it doesn't read as older than the machine plan it was drawn over
    places.correct_plan(state, target.id, text, now=joined.at + 0.001, base=frame.id, source="merge", view=view)


def _on_page(text: str, view: places.PageView) -> str:
    """Plan-frame statements (rooms, openings, fixtures, and camera spots: the subset spatial/plan_import.py reads)
    drawn on an owner plan's page (`places.PageView`, whose move takes the page into the plan frame), without a
    header; "" when they hold no room."""
    from liberational.spatial import plan_import  # the owner plan reader; only a carry onto a page needs it

    try:
        parsed = plan_import.parse(text)
    except plan_import.PlanError:
        return ""
    # The page-to-frame move is p = R(turn) F q + shift; its inverse q = F R(-turn) (p - shift) is a move of the same
    # kind: with the mirror, F R(-turn) = R(turn) F; without it, R(-turn)
    turn = view.turn if view.mirror else -view.turn
    c, s = math.cos(turn), math.sin(turn)
    f = -1.0 if view.mirror else 1.0
    sx, sy = view.shift
    shift = (-(c * sx - s * f * sy), -(s * sx + c * f * sy))
    lines = plan_import.to_dsl(parsed, plan_import.Move(turn, shift, view.mirror), "").splitlines()
    return "\n".join(lines[1:]).strip("\n") + "\n"


#: A group seated by its cameras' own poses (`seat`) is refused when ICP then moves one of them more than this far
#: (metres), or turns the group more than SEAT_TURN_DEG, from where those poses put it: a camera that stayed put is
#: placed by its own pose far closer than that, so a seat this far off is a camera that moved or a snapshot that's wrong.
SEAT_SHIFT_M = 0.5
SEAT_TURN_DEG = 10.0
#: A seat registers its snapshot against the map within this (metres) of where its cameras put it: `register` thins
#: each cloud to REG_POINTS, and a whole home's map thinned that far leaves one room too few points to fit.
#: More than SEAT_SHIFT_M, so the map covers any seat that can hold.
SEAT_PAD_M = 1.0


def _yaw(rotation: np.ndarray) -> float:
    """The turn about z (radians) nearest a 3x3 rotation, as `umeyama` finds it."""
    return math.atan2(float(rotation[1, 0] - rotation[0, 1]), float(rotation[0, 0] + rotation[1, 1]))


def by_cameras(
    new: Mapping[str, places.CameraPose],
    old: Mapping[str, places.CameraPose],
    bounds: tuple[float, float] | None = SCALE_RANGE,
) -> np.ndarray | None:
    """The 4x4 similarity, turning only about z, that puts the cameras where `new` has them (in a levelled cloud) onto
    where `old` has them (in the fused map): the mean of each camera's own turn, the median of its heights' ratio as
    the scale, and the shift that brings their mean position over. None with no camera in both, or for a scale
    outside `bounds` (any scale with None)."""
    both = sorted(set(new) & set(old))
    if not both:
        return None
    a = {c: np.asarray(new[c].pose, dtype=np.float64).reshape(4, 4) for c in both}
    b = {c: np.asarray(old[c].pose, dtype=np.float64).reshape(4, 4) for c in both}
    turns = [_yaw(b[c][:3, :3] @ a[c][:3, :3].T) for c in both]
    yaw = math.atan2(sum(math.sin(t) for t in turns), sum(math.cos(t) for t in turns))
    ratios = [b[c][2, 3] / a[c][2, 3] for c in both if a[c][2, 3] > 0 and b[c][2, 3] > 0]
    scale = float(np.median(ratios)) if ratios else 1.0
    if bounds is not None and not bounds[0] <= scale <= bounds[1]:
        return None
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    src = np.mean([a[k][:3, 3] for k in both], axis=0)
    dst = np.mean([b[k][:3, 3] for k in both], axis=0)
    out = np.eye(4)
    out[:3, :3] = scale * rot
    out[:3, 3] = dst - scale * rot @ src
    return out


def _backgrounds(
    poses: Mapping[str, places.CameraPose], held: Mapping[str, places.CameraPose], place: str
) -> dict[str, places.CameraPose]:
    """`poses` with each camera's background as `footprints.kept` keeps it against the one `held` has: a seated
    camera is where it was, so tonight's snapshot fills in its background and never leaves it sparser."""
    out = {}
    for cam, pose in poses.items():
        was = held[cam].background if cam in held else None
        background = footprints.kept(pose.background, was)
        if was is not None and background is was:
            LOG.info(
                "%s: %s keeps its background (%d of %d cells known; tonight's %s)",
                place,
                cam,
                footprints.known(was),
                len(was.heights),
                f"{footprints.known(pose.background)}" if pose.background is not None else "none",
            )
        out[cam] = replace(pose, background=background)
    return out


@dataclass(frozen=True)
class SeatFit:
    """A cloud seated in a map by its cameras (`seat_fit`): where their poses put it (`start`), and ICP's polish of
    that against the map around there, with how far the polish moved the cameras (metres, the farthest) and turned
    the cloud (degrees). `reg` is None when the map has nothing there."""

    start: np.ndarray
    reg: Registration | None
    shift: float = 0.0
    turn: float = 0.0

    @property
    def held(self) -> bool:
        """ICP registered it within SEAT_SHIFT_M and SEAT_TURN_DEG of where its cameras put it."""
        return self.reg is not None and self.reg.ok and self.shift <= SEAT_SHIFT_M and self.turn <= SEAT_TURN_DEG


def seat_fit(
    new_pts: np.ndarray,
    here: Mapping[str, places.CameraPose],
    there: Mapping[str, places.CameraPose],
    map_pts: np.ndarray,
) -> SeatFit | None:
    """A levelled, metric cloud (`new_pts`, with its cameras posed in it as `here`) seated in a map (`map_pts`) where
    the map has those cameras (`there`): the start from their poses (`by_cameras`), polished by ICP against the map
    within SEAT_PAD_M of where the start puts the cloud. None when they share no camera or the scale is off."""
    start = by_cameras(here, there)
    if start is None:
        return None
    placed = apply(start, new_pts)
    lo, hi = placed.min(axis=0) - SEAT_PAD_M, placed.max(axis=0) + SEAT_PAD_M
    around = map_pts[np.all((map_pts >= lo) & (map_pts <= hi), axis=1)]
    if len(around) == 0:
        return SeatFit(start, None)
    reg = register(new_pts, around, start=start)
    centres = np.array([np.asarray(here[c].pose, dtype=np.float64).reshape(4, 4)[:3, 3] for c in here if c in there])
    shift = float(np.linalg.norm(apply(reg.transform, centres) - apply(start, centres), axis=1).max())
    turn = abs(math.degrees(_yaw(reg.transform[:3, :3] @ np.linalg.inv(start[:3, :3]))))
    return SeatFit(start, reg, shift, turn)


def seat(
    state: Path,
    place: str,
    points: np.ndarray,
    colors: np.ndarray | None,
    floor: Sequence[float],
    poses: Mapping[str, places.CameraPose],
    now: float | None = None,
) -> places.Revision | None:
    """A snapshot of some of `place`'s own cameras (a cloud in their run's frame, its `floor`, and their `poses` in it)
    fused into the place's map by where those cameras are registered in it, as a tier 1 revision. A joined
    place holds rooms whose cameras never match each other, so each night brings it one group a room, and the
    coarse wall search can't find one room in a home: the cameras' registered poses give the start instead
    (`by_cameras`), and ICP polishes it against the map around there (SEAT_PAD_M). Each seated camera's background is
    tonight's filled from the one it had, never a sparser one (`footprints.kept`). Each seated camera gets
    tonight's pose as the seat placed it, and the seat records those poses as `seat` observations (spatial/poses.py,
    quality the fitness): the owner's measured spots (`measured_spots`) are evidence of where a camera was,
    read with them, not a pin. Only the snapshot's points a LiDAR walk didn't already see better go in (`gate`). Every
    other camera of the place keeps its pose, the place keeps its signals with the group's cameras among them, and its
    frame holds unless the map is rescaled or turned level, as in `refine` (never while the owner's plan is drawn on
    it, `owner_drawn`). A run that seats several groups (one a room)
    writes one revision for each, each built on the one before. None, with nothing written, when the place has no fused
    map, the cameras can't place the cloud, the cloud doesn't register, or ICP moves it off where its cameras put it
    (SEAT_SHIFT_M, SEAT_TURN_DEG)."""
    fused_rev = fused_revision(state, place)
    if fused_rev is None:
        return None
    group = ", ".join(sorted(poses))
    at = time.time() if now is None else now
    level = leveller(floor)
    seen = _sightings(state, sorted({*poses, *fused_rev.cameras}), at)
    scale = metric(_moved(poses, level), f"{place}'s snapshot of {group}", seen=seen)
    if not cameras.SCALE_BOUNDS[0] <= scale[0, 0] <= cameras.SCALE_BOUNDS[1]:
        LOG.warning("%s: %s would be scaled by %.2f to their heights; not fused", place, group, scale[0, 0])
        return None
    level = scale @ level
    new_pts = apply(level, np.asarray(points, dtype=np.float64).reshape(-1, 3))
    here = _moved(poses, level)
    owners = owner_drawn(state, place, fused_rev)
    rescale = np.eye(4) if owners else metric(fused_rev.cameras, f"{place}'s fused map", RESCALE_TOL, seen)
    there = _moved(fused_rev.cameras, rescale)
    old = _cloud(state, fused_rev)
    assert old is not None
    before = _fused(old)
    before = replace(before, points=apply(rescale, before.points))
    fit = seat_fit(new_pts, here, there, before.points)
    if fit is None:
        LOG.warning("%s: %s aren't registered in its map at a scale near theirs; not fused", place, group)
        return None
    if fit.reg is None:
        LOG.warning("%s: its map has nothing where %s's cameras put them; not fused", place, group)
        return None
    reg, shift, turn = fit.reg, fit.shift, fit.turn
    if not fit.held:
        LOG.warning(
            "%s: %s didn't seat in its map (fitness %.2f, needs %.2f; %d inliers, needs %d; ICP moved them %.2f m and "
            "%.1f degrees off their poses, at most %.1f and %.0f); not fused, the place keeps its map",
            place,
            group,
            reg.fitness,
            MIN_FITNESS,
            reg.inliers,
            MIN_INLIERS,
            shift,
            turn,
            SEAT_SHIFT_M,
            SEAT_TURN_DEG,
        )
        return None
    placed = apply(reg.transform, new_pts)
    kept = gate(before, placed)
    fused = merge(before, placed[kept], colors[kept] if colors is not None else None)
    cams = {**there, **_moved(_backgrounds(poses, there, place), reg.transform @ level)}
    frame = fused_rev.frame if np.allclose(rescale, np.eye(4)) else None
    LOG.info(
        "%s: seated %s (fitness %.2f, %d inliers, rmse %.3f m, %.2f m and %.1f degrees off their poses; %d points, "
        "%d faded, %d of the snapshot's left out where a walk saw)",
        place,
        group,
        reg.fitness,
        reg.inliers,
        reg.rmse,
        shift,
        turn,
        len(fused.points),
        fused.dropped,
        int((~kept).sum()),
    )
    levelled = fused
    if not owners:
        levelled, cams = relevel(fused, cams, place)
    if levelled is not fused:
        frame = None
    newest = places.latest(state, place)
    signals = {k: list(v) for k, v in (newest or fused_rev).signals.items()}
    have = signals.setdefault("camera", [])
    have += [c for c in sorted(poses) if c not in have]
    rev = _revise(state, place, levelled, cams, [fused_rev.id], frame, at, now, signals)
    poses_mod.record(
        state,
        [
            poses_mod.Observation(
                t=at,
                camera=c,
                source="seat",
                place=place,
                frame=rev.frame,
                pose=np.asarray(rev.cameras[c].pose, dtype=np.float64).reshape(4, 4),
                quality=reg.fitness,
            )
            for c in sorted(poses)
            if c in rev.cameras
        ],
    )
    return rev


def repose(
    state: Path, place: str, poses: Mapping[str, places.CameraPose], now: float | None = None
) -> places.Revision | None:
    """`place` with `poses` replacing those of its cameras, as a tier 1 revision of its fused map (unchanged) with its
    plan derived again from the cameras' footprints. The frame holds. The repair for a seat that wrote a camera off
    its own cloud: the map kept the room where it was, so the camera goes back to the pose that fits it. None
    when the place has no fused map or one of `poses`' cameras isn't the place's."""
    fused_rev = fused_revision(state, place)
    old = _cloud(state, fused_rev) if fused_rev is not None else None
    if fused_rev is None or old is None:
        return None
    newest = places.latest(state, place) or fused_rev
    if not set(poses) <= set(newest.cameras):
        return None
    cams = {**newest.cameras, **poses}
    at = time.time() if now is None else now
    return _revise(state, place, _fused(old), cams, [fused_rev.id], fused_rev.frame, at, now, newest.signals)


def restore(state: Path, place: str, revision: str, now: float | None = None) -> places.Revision | None:
    """`place` back on its tier 1 revision `revision`: a new tier 1 revision of that revision's map (its file must still
    be there), with its cameras and signals and its plan derived again. The frame holds. The repair for a revision that
    took the place's map away from its cameras (a walk into the place that didn't register stood as the place's
    map, with none of its fixed cameras). None when `revision` isn't one of `place`'s tier 1 revisions with its map."""
    held = next((r for r in places.revisions(state, place) if r.id == revision and r.producer == PRODUCER), None)
    old = _cloud(state, held) if held is not None else None
    if held is None or old is None:
        return None
    at = time.time() if now is None else now
    return _revise(state, place, _fused(old), held.cameras, [held.id], held.frame, at, now, held.signals)


#: The room a camera sees, for `move_room`: the map's voxels inside its frame, within this (metres, top-down) of it and
#: nearer to it than to any other camera of the place.
ROOM_M = 6.0


def room_of(points: np.ndarray, cam: str, cams: Mapping[str, places.CameraPose]) -> np.ndarray:
    """Which of `points` (a levelled map's) are camera `cam`'s room (ROOM_M): in its view, and nearer it top-down than
    any other of `cams`."""
    from liberational.spatial import splats  # splats loads the splat runtime's helpers; only this needs it

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    where = {c: np.asarray(p.pose, dtype=np.float64).reshape(4, 4)[:2, 3] for c, p in cams.items()}
    own = np.linalg.norm(pts[:, :2] - where[cam], axis=1)
    out = (own <= ROOM_M) & splats.in_view(pts, {cam: cams[cam]})
    for other, xy in where.items():
        if other != cam:
            out &= own <= np.linalg.norm(pts[:, :2] - xy, axis=1)
    return out


def move_room(
    state: Path, place: str, camera: str, transform: np.ndarray, now: float | None = None
) -> tuple[places.Revision, int] | None:
    """`place` with `camera` and its room moved by `transform` (a rigid 4x4 in the fused map's frame, turning only about
    z): the camera's pose, and the map's voxels of its room (`room_of`), so the map doesn't hold the room twice once
    the right one is fused in. Where a moved voxel lands on one already there, the one there stays. Every other camera
    keeps its pose, and the place keeps its signals and frame. The repair for a room joined in the wrong spot (a
    LiDAR walk's seats of the camera disagreed with its registration by 79 degrees). Returns the revision and how
    many of the room's voxels are in the map where they moved to; None when the place has no fused map or doesn't register `camera`."""
    fused_rev = fused_revision(state, place)
    old = _cloud(state, fused_rev) if fused_rev is not None else None
    newest = places.latest(state, place) or fused_rev
    if fused_rev is None or old is None or newest is None or camera not in newest.cameras:
        return None
    fused = _fused(old)
    room = room_of(fused.points, camera, newest.cameras)
    stays = ~room
    pts = np.r_[fused.points[stays], apply(transform, fused.points[room])]
    order = np.r_[np.flatnonzero(stays), np.flatnonzero(room)]
    # np.unique keeps each voxel's first point: the unmoved ones come first, so they win where a moved one lands (and
    # of two moved ones that land in one voxel, as a turn off the grid's axes can put them, the first stays)
    _, first = np.unique(_keys(pts, VOXEL), axis=0, return_index=True)
    keep = np.sort(first)
    count = int((keep >= int(stays.sum())).sum())
    moved = Fused(
        pts[keep],
        fused.colors[order[keep]] if fused.colors is not None else None,
        fused.seen[order[keep]],
        fused.missed[order[keep]],
        walked=fused.walked[order[keep]] if fused.walked is not None else None,
    )
    cams = {**newest.cameras, **_moved({camera: newest.cameras[camera]}, transform)}
    at = time.time() if now is None else now
    LOG.info("%s: moved %s and %d voxels of its room", place, camera, count)
    rev = _revise(state, place, moved, cams, [fused_rev.id], fused_rev.frame, at, now, newest.signals)
    return rev, count


def release(state: Path, place: str, gone: Sequence[str], now: float | None = None) -> places.Revision | None:
    """`place` without the cameras `gone`, as a tier 1 revision of its fused map (unchanged) with its plan derived
    again from the other cameras' footprints: a camera moved out of the place to a room no other camera sees
    (`cameras.write_places`, after its seat was refused MOVED_NIGHTS running) is registered on its own, and its old
    pose here would cast its new room's footprints into this one. The frame holds. None when the place has no fused
    map."""
    fused_rev = fused_revision(state, place)
    old = _cloud(state, fused_rev) if fused_rev is not None else None
    if fused_rev is None or old is None:
        return None
    out = set(gone)
    newest = places.latest(state, place) or fused_rev
    cams = {c: p for c, p in newest.cameras.items() if c not in out}
    signals = {k: list(v) for k, v in newest.signals.items()}
    signals["camera"] = [c for c in signals.get("camera", []) if c not in out]
    at = time.time() if now is None else now
    return _revise(state, place, _fused(old), cams, [fused_rev.id], fused_rev.frame, at, now, signals)


# Rebuilding a place's map from its sharpest sources.


@dataclass(frozen=True)
class Piece:
    """A cloud a rebuild is made of (`rebuild`): its points and colours in its own frame, its floor there, and its
    cameras posed in it."""

    points: np.ndarray
    colors: np.ndarray | None
    floor: Sequence[float]
    cameras: Mapping[str, places.CameraPose]


@dataclass(frozen=True)
class GroupSeat:
    """How a fixed-camera group's cloud went into a rebuild."""

    cameras: tuple[str, ...]
    fitness: float | None  # ICP's against the walks around where its cameras' spots put it; None where they're empty
    polished: bool  # ICP's polish held (`SeatFit.held`), so the cloud sits where it fits the walks; else by the spots
    shift: float  # metres ICP moved its cameras from their spots, the farthest
    turn: float  # degrees ICP turned it
    kept: int  # its points fused
    gated: int  # its points left out where a walk saw (`gate`) or outside the place (`inside`)
    why: str = ""  # why it isn't in the rebuild, when it isn't
    scale: float = 1.0  # ICP's scale of the cloud over the spots' (1 when it didn't hold)


@dataclass(frozen=True)
class Rebuilt:
    """A place's map rebuilt (`rebuild`): the map with its counts, every camera posed in it, and how each group went."""

    fused: Fused
    cameras: dict[str, places.CameraPose]
    seats: tuple[GroupSeat, ...]
    outside: int  # the walks' points left out outside the place (`inside`)


def levelled(piece: Piece, what: str) -> tuple[np.ndarray, dict[str, places.CameraPose], np.ndarray] | None:
    """A snapshot's points and cameras levelled and put at a metric scale by its cameras' heights (`metric`), as `seat`
    puts them, with that 4x4; None when the scale is out of cameras.SCALE_BOUNDS."""
    level = leveller(piece.floor)
    scale = metric(_moved(piece.cameras, level), what)
    if not cameras.SCALE_BOUNDS[0] <= scale[0, 0] <= cameras.SCALE_BOUNDS[1]:
        return None
    level = scale @ level
    return apply(level, np.asarray(piece.points, dtype=np.float64).reshape(-1, 3)), _moved(piece.cameras, level), level


def rebuild(
    walks: Sequence[tuple[Piece, np.ndarray]],
    groups: Sequence[Piece],
    spots: Mapping[str, places.CameraPose],
    inside: Callable[[np.ndarray], np.ndarray] | None = None,
    held: Mapping[str, places.CameraPose] | None = None,
) -> Rebuilt:
    """A place's map made again from its sharpest sources only, in the frame `spots` are in: the LiDAR walks,
    each moved by its 4x4 (its map into this frame), then each fixed-camera group's snapshot (`levelled`) placed where
    the owner measured its cameras (`spots`: each fixed camera's pose there, the owner's spot and height with its
    registered or given facing, `by_cameras`) and polished by ICP against the walks around there when that holds
    (`seat_fit`). Its points go in where no walk saw and next to the walks' where one did (`gate`). Points `inside`
    says are outside the place (outside every room of the owner's plan, say) are left out.

    The cameras stand on their spots: each keeps tonight's turn, tilt, and lens from its snapshot as placed, at its own
    spot exactly (`on_spots`), and the fuller of its background and the one `held` had (`_backgrounds`). The cloud goes
    where it fits the walks, which can be a few decimetres from where the spots put it: a single camera's MASt3R cloud
    has the camera off its own surfaces by that much (on Place 1, the dining room's by 0.27 m), and the owner's tape
    holds for where the camera is. A camera of `spots` no group has stands at its spot. The walks' looks come along,
    moved. Nothing is written (`write_rebuild` writes it). A group whose cameras have no spot, or whose scale is off,
    is left out (`GroupSeat.why`)."""
    fused: Fused | None = None
    cams: dict[str, places.CameraPose] = {}
    outside = 0
    for piece, transform in walks:
        pts = apply(transform, np.asarray(piece.points, dtype=np.float64).reshape(-1, 3))
        cols = piece.colors
        if inside is not None:
            ok = inside(pts)
            outside += int((~ok).sum())
            pts, cols = pts[ok], (cols[ok] if cols is not None else None)
        if fused is None:
            vp, vc = voxelize(pts, cols, VOXEL)
            ones = np.ones(len(vp), np.int64)
            fused = Fused(vp, vc, ones, np.zeros(len(vp), np.int64), walked=ones)
        else:
            fused = merge(fused, pts, cols, walked=True)
        cams.update(_moved(piece.cameras, transform))
    if fused is None:
        raise ValueError("a rebuild needs a walk")
    seats = []
    for group in groups:
        names = tuple(sorted(group.cameras))
        if not set(names) <= set(spots):
            seats.append(GroupSeat(names, None, False, 0.0, 0.0, 0, 0, "a camera with no measured spot"))
            continue
        got = levelled(group, f"the rebuild's snapshot of {', '.join(names)}")
        if got is None:
            seats.append(GroupSeat(names, None, False, 0.0, 0.0, 0, 0, "its scale to its heights is out of bounds"))
            continue
        new_pts, here, level = got
        fit = seat_fit(new_pts, here, {c: spots[c] for c in names}, fused.points)
        if fit is None:
            seats.append(GroupSeat(names, None, False, 0.0, 0.0, 0, 0, "its scale against the spots is off"))
            continue
        placed = fit.reg.transform if fit.held and fit.reg is not None else fit.start
        pts = apply(placed, new_pts)
        keep = gate(fused, pts)
        if inside is not None:
            keep &= inside(pts)
        cols = group.colors[keep] if group.colors is not None else None
        fused = merge(fused, pts[keep], cols)
        moved = _moved(_backgrounds(group.cameras, held or {}, "rebuild"), placed @ level)
        cams.update(on_spots(moved, {c: _xyz(spots[c]) for c in names}))
        scale = 1.0
        if fit.held and fit.reg is not None:
            scale = float(np.cbrt(np.linalg.det(fit.reg.transform[:3, :3]) / np.linalg.det(fit.start[:3, :3])))
        seats.append(
            GroupSeat(
                names,
                fit.reg.fitness if fit.reg is not None else None,
                fit.held,
                fit.shift,
                fit.turn,
                int(keep.sum()),
                int((~keep).sum()),
                scale=scale,
            )
        )
    for cam, pose in spots.items():  # a camera with no snapshot this time stands at its spot
        cams.setdefault(cam, pose)
    return Rebuilt(fused, cams, tuple(seats), outside)


def _xyz(pose: places.CameraPose) -> tuple[float, float, float]:
    m = np.asarray(pose.pose, dtype=np.float64).reshape(4, 4)
    return float(m[0, 3]), float(m[1, 3]), float(m[2, 3])


def measured_spots(state: Path, place: str) -> dict[str, tuple[float, float, float]]:
    """The owner's measured camera spots in `place`'s fused map: the `x-camera` statements with a height in
    its owner plan, when that plan holds drawn on the fused map's frame with no plan file (`_owner_frame`); {}
    otherwise."""
    from liberational.spatial import plan_import  # the owner plan reader

    known = places.get(state, place)
    fused_rev = fused_revision(state, place)
    if known is None or fused_rev is None or not known.plan or not _owner_frame(state, known, fused_rev.frame):
        return {}
    drawn = next((r for r in places.revisions(state, place) if r.id == known.plan_revision), None)
    frame = places.plan_frame(drawn) if drawn is not None else None
    try:
        plan = plan_import.parse(known.plan)
    except plan_import.PlanError:
        return {}
    if frame is None:
        return {}
    t = np.asarray(frame, dtype=np.float64).reshape(3, 4)
    out = {}
    for mark in plan.cameras:
        if mark.height is not None:
            p = t[:, :3].T @ (np.array([mark.at[0], mark.at[1], mark.height]) - t[:, 3])
            out[mark.id] = (float(p[0]), float(p[1]), float(p[2]))
    return out


#: A LiDAR walk is placed by the owner's plan (`by_plan`) only when its wall band fits the plan this well: a median
#: this near (metres) a wall's face, with PLAN_NEAR_SHARE of it within plan_import.NEAR_FACE of one, and PLAN_INSIDE
#: of its floor inside the plan's rooms. Place 1's walk fitted at 3.6 cm and 79%; a walk turned or shifted into the
#: wrong room puts walls across rooms and floor outside them.
PLAN_MEDIAN_M = 0.06
PLAN_NEAR_SHARE = 0.6
PLAN_INSIDE = 0.9


def by_plan(state: Path, place: str, points: np.ndarray) -> tuple[np.ndarray, str] | None:
    """Where a LiDAR walk's levelled cloud (`points`, floor at z = 0, metres) stands in `place`'s fused map by the
    owner's plan: its wall band registered to the plan (plan_import.register, every turn and shift), when the
    plan holds drawn on the fused map's frame and the fit passes PLAN_MEDIAN_M, PLAN_NEAR_SHARE, and PLAN_INSIDE. The
    4x4 from the levelled cloud into the fused map, with what it found for the log; None otherwise."""
    from liberational.spatial import plan_import  # the owner plan reader

    known = places.get(state, place)
    fused_rev = fused_revision(state, place)
    if known is None or fused_rev is None or not known.plan or not _owner_frame(state, known, fused_rev.frame):
        return None
    drawn = next((r for r in places.revisions(state, place) if r.id == known.plan_revision), None)
    frame = places.plan_frame(drawn) if drawn is not None else None
    if frame is None:
        return None
    try:
        plan = plan_import.parse(known.plan)
        fit = plan_import.register(plan, plan_import.band(points, LEVEL_PLAN), {}, mirror=False)
    except plan_import.PlanError:
        return None
    floor = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    floor = floor[np.abs(floor[:, 2]) < 0.05][:, :2]
    inside = float(plan_import.inside(plan, fit.move.inverse(floor)).mean()) if len(floor) else 0.0
    found = (
        f"fit to the owner plan: walls median {fit.median * 100:.1f} cm, {fit.near_share:.0%} within "
        f"{plan_import.NEAR_FACE * 100:.0f} cm of a wall, {inside:.0%} of its floor inside the rooms"
    )
    if not (fit.median <= PLAN_MEDIAN_M and fit.near_share >= PLAN_NEAR_SHARE and inside >= PLAN_INSIDE):
        LOG.info("%s: the walk doesn't fit the owner plan well enough (%s)", place, found)
        return None
    return np.linalg.inv(_plan_matrix(frame)) @ plan_import.undo(LEVEL_PLAN, fit.move), found


def on_spots(
    cams: Mapping[str, places.CameraPose], at: Mapping[str, tuple[float, float, float]]
) -> dict[str, places.CameraPose]:
    """`cams` with each camera that has a spot in `at` (map metres) moved there, its turn and tilt kept."""
    out = dict(cams)
    for cam, xyz in at.items():
        if cam in out:
            m = np.asarray(out[cam].pose, dtype=np.float64).reshape(4, 4)
            m[:3, 3] = xyz
            out[cam] = replace(out[cam], pose=tuple(float(x) for x in m.ravel()))
    return out


def write_rebuild(
    state: Path,
    place: str,
    rebuilt: Rebuilt,
    sources: Sequence[str],
    now: float | None = None,
) -> places.Revision:
    """`rebuilt` written as a tier 1 revision of `place` in its fused map's frame (the rebuild was made in it), with
    its newest revision's signals. An owner plan drawn in that frame is saved again on the new revision, as it was
    (its text and page view), so positions are read through the rebuilt cameras' poses (they read the plan's
    revision's, `places.Place.plan_revision`); the earlier record stays in its history. Raises KeyError when the
    place has no fused map."""
    fused_rev = fused_revision(state, place)
    if fused_rev is None:
        raise KeyError(f"{place} has no fused map to rebuild")
    newest = places.latest(state, place) or fused_rev
    at = time.time() if now is None else now
    known = places.get(state, place)
    rev = _revise(state, place, rebuilt.fused, rebuilt.cameras, sources, fused_rev.frame, at, now, newest.signals)
    if known is not None and known.plan and _owner_frame(state, known, fused_rev.frame):
        view = known.view.as_dict() if known.view is not None else None
        history = places.plan_history(state, place)
        source = str(history[-1].value.get("source") or "owner") if history else "owner"
        places.correct_plan(state, place, known.plan, now=rev.at + 0.001, base=rev.id, source=source, view=view)
    LOG.info(
        "%s: rebuilt from %d sources (%d points, %d walked; %d cameras)",
        place,
        len(sources),
        len(rebuilt.fused.points),
        int((rebuilt.fused.walks > 0).sum()),
        len(rebuilt.cameras),
    )
    return rev
