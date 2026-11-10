"""Fixed cameras registered by reprojection, each with its own lens (#3069, part of the spatial memory goal #2794).

MASt3R's global alignment (spatial/mast3r_runner.py) takes every camera as a pinhole. A wide lens isn't one: Place
1's living-room camera, a portrait frame from ceiling to floor with a strong barrel, was fitted as a 51 degree pinhole,
and the alignment bent the other camera to agree with it: cam0's focal came out 0.72 to 0.74 of its width against
0.66 on its spec sheet and from GeoCalib, the two cameras' heights over their floor disagreed with the measured ones by
up to 12% between them, and one person seen by both landed 0.6 to 1.2 m apart across the room. The network itself
doesn't model the barrel either: its own point map of that frame is a pinhole's to within 2% from centre to edge.

So a group is registered here instead, from what MASt3R does get right, its matches and each view's depth:

1. **Reference.** One camera's frame is the group's. Its matched pixels are lifted to 3D along a pinhole's rays (its
   focal is fitted below), at the depth MASt3R's pair inference gave them.
2. **Fit.** Every other camera's pose in the reference's frame, its focal, and a one-parameter division lens
   (spatial/lens.py's model: a point seen at normalized (x, y) is really at (x, y) / (1 + lambda r^2)) are fitted, with
   the reference's focal, so the reference's lifted points land on the other camera's matched pixels (least squares
   with a soft loss, from MASt3R's own relative rotation). A lens that undoes the frame past spatial/lens.py's
   MIN_DENOMINATOR is refused and that camera fitted as a pinhole. On Place 1 the fit put cam0 at 0.661 of its width
   and living-room's barrel at lambda -0.33; without the lens the same fit gives cam0 0.734, MASt3R's own error.
   Each camera takes its turn as the reference, and the one whose fits leave the smallest reprojection error wins
   (a barrel lens as the reference would be lifted as a pinhole).
3. **Cloud.** The reference's point map along its fitted rays, and each other camera's (from the same pair inference)
   re-cast along its own fitted rays at the distance MASt3R gave it, all in the reference's frame.

4. **Background.** Each camera's point map is kept on its pixel grid (`Registered.maps`), and `background` turns it
   into what the camera sees with nobody in the way, cell by cell (places.Background): what spatial/footprints.py's
   `at_edge` reads to mark a foot point on an occluding edge, a couch's back, where pose guesses hidden feet (#3070).
   A fitted camera's map is from its own side of its pair with the reference (its own point map), scaled to the
   reference's by their matches (`_own_scale`): its map from the reference's side is sure only where the two views
   overlap. On Place 1 on 2026-11-08, with living-room as the reference, that side was sure on 3% of cam0's pixels
   and its own on 83%, and cam0's background knew 98 of its 2,304 cells (#3220); where both were sure they agreed
   within 2 cm. Without that side, or too few matches sure on it, the reference's side is used.

A camera with no pair to the reference, or a fit that fails, leaves the group to MASt3R's alignment, as before.
`floor` then finds the group's floor from the reference camera's points (the lowest flat layer: each camera's own
floor layer stands a little apart from the others', and the lowest layer of them all mixed them, tilting it 3 to 7
degrees), tilted, where the owner measured two or more of the cameras' heights, so the cameras' heights over it keep
the measured ratios (a degree of floor tilt moves a far floor point of a low camera 0.3 m). On Place 1 the reference's
floor alone kept them within 2.5% before the tilt.

On Place 1 (42 sightings of one person by both cameras, checked by eye in the frames; two of one day's frame pairs; with
footprints.FOOT_M), the same person lands 0.17 m apart near the cameras on both frame pairs (MASt3R's alignment: 0.29
and 0.72), and 0.37 and 0.27 m apart across the room (0.57 and 1.21).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from liberational import logs
from liberational.memory import places

LOG = logs.get("spatial-cameras")

MIN_CONF = 1.5  # as mast3r_runner.py: a point map's pixel this sure takes part in the cloud
MATCH_SHARE = 0.8  # the surest share of a pair's matches the fit uses
MIN_PAIR_MATCHES = 200  # a pair with fewer can't register its camera
LOSS_PX = 3.0  # the soft loss's scale: matches this far off (pixels at MASt3R's size) start to count less
MAX_RESIDUAL_PX = 8.0
#: A held camera's fit is checked each night against a free one (`misfit`): the held lens is called into question when
#: the free fit leaves the matches at least this many pixels closer, and this many times closer
REFIT_PX = 1.0
REFIT_RATIO = 1.5
HELD_EPS = 1e-6  # how far a held focal (pixels) or lens may move in a fit: not at all, but least_squares needs room  # a fit whose median reprojection error is worse than this isn't kept
HEIGHT_TILT_DEG = 3.0  # the most the floor is tilted to the measured heights


PAIR_FIELDS = ("own", "own_conf", "other", "other_conf", "xy", "xy_other", "match_conf")  # mast3r_runner's names


@dataclass(frozen=True)
class Pair:
    """One pair's inference as the runner saves it, from the reference's side: the reference's point map in its own
    frame and its confidences, the other camera's point map in the reference's frame and its confidences (both H x W
    at MASt3R's size), and their matches (pixel x, y in each, and each match's confidence)."""

    own: np.ndarray
    own_conf: np.ndarray
    other: np.ndarray
    other_conf: np.ndarray
    xy: np.ndarray
    xy_other: np.ndarray
    match_conf: np.ndarray


@dataclass(frozen=True)
class Fit:
    """A camera fitted to the reference: `rotation` and `shift` take a point from the reference's frame to the
    camera's (x = R p + t), its `focal` in pixels at MASt3R's size, its lens `division`, the reference's focal, and the
    median reprojection error in pixels."""

    rotation: np.ndarray
    shift: np.ndarray
    focal: float
    division: float
    ref_focal: float
    residual: float


def _centred(xy: np.ndarray, size: Sequence[int]) -> np.ndarray:
    """Pixel indices (x, y) as offsets of their centres from the frame's centre."""
    return np.asarray(xy, dtype=np.float64) + 0.5 - np.array([size[0] / 2, size[1] / 2])


def distort(x: np.ndarray, y: np.ndarray, division: float) -> tuple[np.ndarray, np.ndarray]:
    """Normalized pinhole coordinates to where a division lens shows them (spatial/lens.py's `_distort`)."""
    r = np.hypot(x, y)
    q = division * r
    with np.errstate(divide="ignore", invalid="ignore"):
        rd = np.where(np.abs(q) > 1e-12, (1 - np.sqrt(np.maximum(1 - 4 * q * r, 0.0))) / (2 * q), r)
        s = np.where(r > 1e-12, rd / r, 1.0)
    return x * s, y * s


def undistort(x: np.ndarray, y: np.ndarray, division: float) -> tuple[np.ndarray, np.ndarray]:
    """Where a division lens's normalized seen coordinates really point."""
    s = 1 / (1 + division * (x * x + y * y))
    return x * s, y * s


def _rotation(rotvec: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_rotvec(rotvec).as_matrix()


def _rotvec(matrix: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(matrix).as_rotvec()


def corner_denominator(focal: float, size: Sequence[int], division: float) -> float:
    """1 + lambda r^2 at the frame's corner (spatial/lens.py's `denominator` for a centred lens)."""
    r2 = (size[0] / 2) ** 2 / focal**2 + (size[1] / 2) ** 2 / focal**2
    return 1 + division * r2


def _shift(points: np.ndarray, rotation: np.ndarray, rays: np.ndarray) -> np.ndarray:
    """The shift t that best puts each rotated point R p + t on its ray (least squares on ray x (R p + t) = 0, which
    is linear in t): the fit's start."""
    q = points @ rotation.T

    def skew(v: np.ndarray) -> np.ndarray:
        z = np.zeros(len(v))
        return np.stack([np.c_[z, -v[:, 2], v[:, 1]], np.c_[v[:, 2], z, -v[:, 0]], np.c_[-v[:, 1], v[:, 0], z]], axis=1)

    m = skew(rays)
    a = m.reshape(-1, 3)
    b = -np.einsum("nij,nj->ni", m, q).ravel()
    return np.linalg.lstsq(a, b, rcond=None)[0]


def fit(
    pair: Pair,
    ref_size: Sequence[int],
    size: Sequence[int],
    start: np.ndarray,
    focals: tuple[float, float],
    lens: bool = True,
    held: tuple[float, float] | None = None,
    ref_held: float | None = None,
) -> Fit | None:
    """The other camera of `pair` fitted to the reference (see the module's step 2), from `start`, its rotation from
    the reference's frame as MASt3R had it, and `focals`, the reference's and the camera's as MASt3R had them (pixels
    at its size). `held` fixes the camera's focal (pixels at MASt3R's size) and lens division, and `ref_held` the
    reference's focal, so only the pose is fitted (#3289: a camera's intrinsics are its own and don't change night to
    night). None for too few matches or a fit worse than MAX_RESIDUAL_PX."""
    from scipy.optimize import least_squares

    from liberational.spatial import lens as lens_model

    if len(pair.match_conf) < MIN_PAIR_MATCHES:
        return None
    keep = pair.match_conf >= np.quantile(pair.match_conf, 1 - MATCH_SHARE)
    xy, xy_other = pair.xy[keep], pair.xy_other[keep]
    depth = pair.own[xy[:, 1].astype(int), xy[:, 0].astype(int), 2]
    ok = np.isfinite(depth) & (depth > 0)
    ref_px, px, depth = _centred(xy[ok], ref_size), _centred(xy_other[ok], size), depth[ok]
    if len(depth) < MIN_PAIR_MATCHES:
        return None

    def residual(p: np.ndarray) -> np.ndarray:
        pts = np.c_[ref_px / p[8], np.ones(len(ref_px))] * depth[:, None]
        c = pts @ _rotation(p[:3]).T + p[3:6]
        z = np.maximum(c[:, 2], 1e-6)
        x, y = distort(c[:, 0] / z, c[:, 1] / z, p[7])
        return (np.c_[p[6] * x, p[6] * y] - px).ravel()

    w, w_ref = float(size[0]), float(ref_size[0])
    lo = [-np.inf] * 6 + [0.25 * w, -0.6 if lens else -1e-9, 0.25 * w_ref]
    hi = [np.inf] * 6 + [4.0 * w, 0.3 if lens else 1e-9, 4.0 * w_ref]
    ref_f, cam_f = (min(max(f, 0.3 * w0), 3.9 * w0) for f, w0 in ((focals[0], w_ref), (focals[1], w)))
    division = 0.0
    if held is not None:  # least_squares needs every bound open, so a held value gets a sliver either side
        cam_f, division = held
        lo[6:8], hi[6:8] = [cam_f - HELD_EPS, division - HELD_EPS], [cam_f + HELD_EPS, division + HELD_EPS]
    if ref_held is not None:
        ref_f = ref_held
        lo[8], hi[8] = ref_f - HELD_EPS, ref_f + HELD_EPS
    pts = np.c_[ref_px / ref_f, np.ones(len(ref_px))] * depth[:, None]
    t0 = _shift(pts, start, np.c_[px / cam_f, np.ones(len(px))])
    x0 = np.r_[_rotvec(start), t0, cam_f, division, ref_f]
    best = least_squares(residual, x0, bounds=(lo, hi), loss="soft_l1", f_scale=LOSS_PX, max_nfev=3000)
    p = best.x
    if held is None and lens and corner_denominator(p[6], size, p[7]) < lens_model.MIN_DENOMINATOR:
        LOG.info("a lens of lambda %.2f undoes the frame's corners too far; fitted as a pinhole", p[7])
        return fit(pair, ref_size, size, start, focals, lens=False, ref_held=ref_held)
    err = float(np.median(np.hypot(*best.fun.reshape(-1, 2).T)))
    if err > MAX_RESIDUAL_PX:
        return None
    return Fit(_rotation(p[:3]), p[3:6].copy(), float(p[6]), float(p[7]), float(p[8]), err)


@dataclass(frozen=True)
class Registered:
    """A group registered by reprojection: each camera's pose (4x4, camera to the reference's frame), its focal at
    MASt3R's size, and its lens's division (0 for the reference); the cloud and its colours; the reference."""

    poses: dict[str, np.ndarray]
    focals: dict[str, float]
    divisions: dict[str, float]
    points: np.ndarray
    colors: np.ndarray
    reference: str
    residuals: dict[str, float]
    own: int  # the cloud's first `own` points are the reference camera's
    # Each camera's point map in the reference's frame (H x W x 3 at MASt3R's size, NaN where MASt3R wasn't sure):
    # what it sees with nobody in the way, for `background`
    maps: dict[str, np.ndarray]


def _cast(xy: np.ndarray, size: Sequence[int], focal: float, division: float) -> np.ndarray:
    """Rays (z = 1) of pixels, a division lens undone."""
    c = _centred(xy, size) / focal
    x, y = undistort(c[:, 0], c[:, 1], division)
    return np.c_[x, y, np.ones(len(x))]


def _grid(shape: tuple[int, ...]) -> np.ndarray:
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    return np.c_[xx.ravel(), yy.ravel()]


def register(
    cameras: Sequence[str],
    pairs: Mapping[tuple[str, str], Pair],
    sizes: Mapping[str, Sequence[int]],
    images: Mapping[str, np.ndarray],
    mast3r_poses: Mapping[str, np.ndarray],
    mast3r_focals: Mapping[str, float],
    pinhole: Sequence[str] = (),
    covered: Mapping[str, np.ndarray | None] | None = None,
    held: Mapping[str, tuple[float, float]] | None = None,
) -> Registered | None:
    """The group registered by reprojection (see the module's docstring), or None when no camera has a pair with every
    other one that fits. `pairs[a, b]` is the pair from a's side; `mast3r_poses` are MASt3R's (camera to world), which
    with its `mast3r_focals` give each fit its start. The `pinhole` cameras get no lens (their frames were undone already).
    A camera's `covered` pixels (H x W at MASt3R's size: its overlays and the people in its frame, #3085) are left out of
    the cloud and its point map, so its background is unknown there rather than a person. `held` gives a camera's own
    intrinsics (#3289): its focal as a share of its frame's width and its lens division, which every fit keeps, so only
    poses are fitted; a camera held with a lens isn't a reference (a reference is lifted as a pinhole)."""
    kept = {c: (f * float(sizes[c][0]), d) for c, (f, d) in (held or {}).items() if c in sizes}
    found: list[tuple[float, str, dict[str, Fit]]] = []
    for ref in cameras:
        if abs(kept.get(ref, (0.0, 0.0))[1]) > 1e-9 and any(abs(kept.get(c, (0.0, 0.0))[1]) <= 1e-9 for c in cameras):
            continue
        ref_held = kept[ref][0] if ref in kept else None
        fits: dict[str, Fit] = {}
        for cam in cameras:
            if cam == ref:
                continue
            pair = pairs.get((ref, cam))
            if pair is None:
                break
            start = (np.linalg.inv(np.asarray(mast3r_poses[cam])) @ np.asarray(mast3r_poses[ref]))[:3, :3]
            got = fit(
                pair,
                sizes[ref],
                sizes[cam],
                start,
                (mast3r_focals[ref], mast3r_focals[cam]),
                cam not in pinhole,
                held=kept.get(cam),
                ref_held=ref_held,
            )
            if got is None:
                break
            fits[cam] = got
        else:
            worst = max(f.residual for f in fits.values())
            found.append((worst, ref, fits))
    if not found:
        return None
    _, ref, fits = min(found, key=lambda x: x[0])
    for cam in sorted(set(fits) & set(kept)):
        start = (np.linalg.inv(np.asarray(mast3r_poses[cam])) @ np.asarray(mast3r_poses[ref]))[:3, :3]
        free = fit(
            pairs[ref, cam], sizes[ref], sizes[cam], start, (mast3r_focals[ref], mast3r_focals[cam]), cam not in pinhole
        )
        if free is not None and misfit(fits[cam].residual, free.residual):
            LOG.warning(
                "%s: its held intrinsics leave its matches %.1f px off where a free fit leaves %.1f (focal %.3f of its "
                "width, lens %+.3f): if it says so night after night, fit them again (tools/fit_intrinsics.py)",
                cam,
                fits[cam].residual,
                free.residual,
                free.focal / float(sizes[cam][0]),
                free.division,
            )
    ref_focal = kept[ref][0] if ref in kept else float(np.median([f.ref_focal for f in fits.values()]))
    first = pairs[ref, next(iter(fits))]
    poses = {ref: np.eye(4)}
    focals = {ref: ref_focal}
    divisions = {ref: 0.0}
    grid = _grid(first.own.shape)
    hidden = covered or {}
    keep = (first.own_conf.ravel() > MIN_CONF) & ~_mask(hidden.get(ref), first.own.shape)
    ref_rays = _cast(grid, sizes[ref], ref_focal, 0.0)
    points = [ref_rays[keep] * first.own[..., 2].ravel()[keep, None]]
    colors = [np.asarray(images[ref]).reshape(-1, 3)[keep]]
    maps = {ref: _as_map(first.own.shape, keep, points[0])}
    base = _depth(first)
    for cam, f in fits.items():
        pair = pairs[ref, cam]
        k = base / _depth(pair)  # this pair's depth in the first pair's scale
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = f.rotation.T, -f.rotation.T @ (f.shift * k)
        poses[cam], focals[cam], divisions[cam] = pose, f.focal, f.division
        # The other camera's points: MASt3R's depth for each of its pixels (its point map in the reference's frame,
        # turned into the camera's), re-cast along the camera's fitted rays.
        net_focal = _focal_of(pair.own, sizes[ref])
        other = pair.other.reshape(-1, 3).astype(np.float64) * k
        other[:, :2] *= net_focal / f.ref_focal  # the network's reference rays to the fitted reference's
        in_cam = other @ f.rotation.T + f.shift * k
        distance = np.where(in_cam[:, 2] > 0, np.linalg.norm(in_cam, axis=1), 0.0)
        mask = hidden.get(cam)
        ok, cam_pts = _recast(distance, pair.other_conf, mask, sizes[cam], f, pose)
        points.append(cam_pts)
        colors.append(np.asarray(images[cam]).reshape(-1, 3)[ok])
        maps[cam] = _as_map(pair.other.shape, ok, cam_pts)
        # Its background from its own side of the pair, which is sure wherever it sees, where the reference's side is
        # sure only where the two views overlap (#3220). The cloud keeps the reference's side: the map is mostly built
        # from nights this camera was the reference, and with its whole view cast through tonight's lens fit, Place
        # 1's cam0 seated at fitness 0.36 (0.35 needed) and 0.21 m off its cameras' poses.
        mine = pairs.get((cam, ref))
        scale = _own_scale(pair, mine, f, sizes[ref]) if mine is not None else None
        if mine is not None and scale is not None:
            distance = np.linalg.norm(mine.own.reshape(-1, 3).astype(np.float64), axis=1) * scale * k
            sure, own_pts = _recast(distance, mine.own_conf, mask, sizes[cam], f, pose)
            maps[cam] = _as_map(mine.own.shape, sure, own_pts)
    LOG.info(
        "registered by reprojection on %s: %s",
        ref,
        "; ".join(
            [f"{ref} focal {focals[ref] / sizes[ref][0]:.3f}"]
            + [
                f"{c} focal {focals[c] / sizes[c][0]:.3f} of its width, lens {divisions[c]:+.3f}, {f.residual:.1f} px off"
                for c, f in fits.items()
            ]
        ),
    )
    return Registered(
        poses,
        focals,
        divisions,
        np.concatenate(points).astype(np.float32),
        np.concatenate(colors).astype(np.uint8),
        ref,
        {c: f.residual for c, f in fits.items()},
        len(points[0]),
        maps,
    )


def misfit(held: float, free: float) -> bool:
    """Whether a held camera's reprojection error (`held`, pixels) is clearly worse than a free fit's (REFIT_PX,
    REFIT_RATIO)."""
    return held >= free + REFIT_PX and held >= REFIT_RATIO * free


def _mask(covered: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    """A camera's covered pixels, flat, for a point map of `shape`; none covered without a mask or for one of another
    size."""
    if covered is None or covered.shape != shape[:2]:
        return np.zeros(shape[0] * shape[1], bool)
    return np.asarray(covered, bool).ravel()


def _recast(
    distance: np.ndarray, conf: np.ndarray, covered: np.ndarray | None, size: Sequence[int], f: Fit, pose: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """A fitted camera's sure pixels (`conf` over MIN_CONF, not `covered`, at a `distance` over 0, flat) and their
    points in the reference's frame (its `pose`): each pixel re-cast along the camera's fitted rays at that distance.
    At its distance, not its depth along the axis: the network cast these pixels along a pinhole's rays, and at a
    barrel's edges those part from the true ones by tens of degrees, where depth along the axis differs from distance
    and distance hardly does (re-cast by depth, living-room's floor put it 1.77 times cam0's height over it against
    1.51 measured; by distance 1.47 to 1.50)."""
    ok = (conf.ravel() > MIN_CONF) & np.isfinite(distance) & (distance > 0) & ~_mask(covered, conf.shape)
    rays = _cast(_grid(conf.shape)[ok], size, f.focal, f.division)
    cam_pts = rays / np.linalg.norm(rays, axis=1, keepdims=True) * distance[ok][:, None]
    return ok, cam_pts @ pose[:3, :3].T + pose[:3, 3]


def _own_scale(pair: Pair, mine: Pair, f: Fit, ref_size: Sequence[int]) -> float | None:
    """The factor that puts a fitted camera's own-side point map (`mine.own`, its pair inference with the reference
    from its own side, in that inference's scale) into `pair`'s scale: the median, over `pair`'s matches, of how far
    from the camera the reference's matched pixel lands as `fit` lifts it, over that pixel's own-side distance. None
    with fewer than MIN_PAIR_MATCHES matches sure on both sides."""
    xy, at = pair.xy.astype(int), pair.xy_other.astype(int)
    depth = pair.own[xy[:, 1], xy[:, 0], 2].astype(np.float64)
    lifted = np.c_[_centred(xy, ref_size) / f.ref_focal, np.ones(len(xy))] * depth[:, None]
    theirs = np.linalg.norm(lifted @ f.rotation.T + f.shift, axis=1)
    own = np.linalg.norm(mine.own[at[:, 1], at[:, 0]].astype(np.float64), axis=1)
    ok = (mine.own_conf[at[:, 1], at[:, 0]] > MIN_CONF) & (depth > 0) & (own > 0) & np.isfinite(theirs / own)
    if ok.sum() < MIN_PAIR_MATCHES:
        return None
    return float(np.median(theirs[ok] / own[ok]))


def _as_map(shape: tuple[int, ...], keep: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Points of a camera's sure pixels back on its pixel grid (H x W x 3), NaN elsewhere."""
    out = np.full((shape[0] * shape[1], 3), np.nan, np.float32)
    out[keep] = points
    return out.reshape(shape[0], shape[1], 3)


BACKGROUND_CELL = 8  # pixels at MASt3R's size a background cell spans each way (64 x 36 cells of a 512 x 288 frame)


def background(
    points: np.ndarray,
    pose: np.ndarray,
    floor: Sequence[float],
    undo: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
) -> places.Background | None:
    """A camera's background (places.Background, #3070) from its point map (`Registered.maps`): each BACKGROUND_CELL
    cell's median surface height above `floor` over the camera's own height above it. `undo` maps fractions of the
    camera's frames to fractions of the point map's frame, for a camera whose frames were undistorted before MASt3R
    (a manifest lens): each cell then takes the map's cell its centre lands in. None when the camera isn't above the
    floor."""
    n = np.asarray(floor[:3], dtype=np.float64)
    unit = float(np.linalg.norm(n))
    n, d = n / unit, float(floor[3]) / unit
    up = float(n @ np.asarray(pose)[:3, 3]) + d
    if up <= 0:
        return None
    h, w = points.shape[:2]
    rows, cols = max(1, h // BACKGROUND_CELL), max(1, w // BACKGROUND_CELL)
    height = (points.reshape(-1, 3).astype(np.float64) @ n + d).reshape(h, w) / up
    cells = height[: rows * BACKGROUND_CELL, : cols * BACKGROUND_CELL]
    cells = cells.reshape(rows, BACKGROUND_CELL, cols, BACKGROUND_CELL).transpose(0, 2, 1, 3).reshape(rows, cols, -1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # an all-NaN cell's median is NaN: unknown
        med = np.nanmedian(cells, axis=2)
    if undo is not None:
        cu, cv = np.meshgrid((np.arange(cols) + 0.5) / cols, (np.arange(rows) + 0.5) / rows)
        mu, mv = undo(cu.ravel(), cv.ravel())
        inside = (mu >= 0) & (mu < 1) & (mv >= 0) & (mv < 1)
        picked = np.full(rows * cols, np.nan)
        picked[inside] = med[(mv[inside] * rows).astype(int), (mu[inside] * cols).astype(int)]
        med = picked.reshape(rows, cols)
    return places.Background(cols, rows, tuple(None if not np.isfinite(x) else round(float(x), 3) for x in med.ravel()))


def _depth(pair: Pair) -> float:
    """The reference's median depth over its sure pixels in one pair's inference: each pair has its own scale."""
    return float(np.median(pair.own[..., 2][pair.own_conf > MIN_CONF]))


def _focal_of(points: np.ndarray, size: Sequence[int]) -> float:
    """The pinhole focal (pixels) a point map's own rays imply: pixel offset over normalized offset, the median."""
    c = _centred(_grid(points.shape), size)
    p = points.reshape(-1, 3).astype(np.float64)
    ok = p[:, 2] > 0
    r_px = np.hypot(c[ok, 0], c[ok, 1])
    r_n = np.hypot(p[ok, 0], p[ok, 1]) / p[ok, 2]
    far = r_px > 0.2 * min(size)
    return float(np.median(r_px[far] / r_n[far]))


def camera_pose(pose: np.ndarray, focal: float, division: float, size: Sequence[int], at: float) -> places.CameraPose:
    """A registered camera as a place revision keeps it: a pinhole with no lens, else its lens as spatial/lens.py's
    tuple, centred, and its focal the undistorted frame's (`lens.out_scale` of the fitted one), as floor_point casts
    rays through it."""
    from liberational.spatial import lens as lens_model

    w, h = int(size[0]), int(size[1])
    flat = tuple(float(x) for x in np.asarray(pose).ravel())
    if abs(division) < 1e-6:
        return places.CameraPose(flat, float(focal), w, h, at)
    found = lens_model.Lens(w, h, (focal, focal, (w - 1) / 2, (h - 1) / 2), division)
    return places.CameraPose(flat, float(focal * lens_model.out_scale(found)), w, h, at, found.as_tuple())


def floor(
    points: np.ndarray, up: Sequence[float], poses: Mapping[str, np.ndarray], measured: Mapping[str, float]
) -> tuple[float, float, float, float] | None:
    """The group's floor: the lowest flat layer of its cloud along `up` (cameras.floor_along), then, with two or more
    measured camera heights, tilted (at most HEIGHT_TILT_DEG) so the cameras' heights over it keep the measured ratios.
    The heights fix the tilt along the line between the cameras; across it the layer's own tilt is kept."""
    from scipy.optimize import least_squares

    from liberational.spatial import cameras as cams

    found = cams.floor_along(points, up)
    if found is None:
        return None
    known = {c: float(m) for c, m in measured.items() if c in poses and float(m) > 0}
    if len(known) < 2:
        return found
    n0 = np.asarray(found[:3], dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    layer = pts[np.abs(pts @ n0 + found[3]) < cams.FLOOR_BAND]
    a = np.cross(n0, [1.0, 0.0, 0.0] if abs(n0[0]) < 0.9 else [0.0, 1.0, 0.0])
    a /= np.linalg.norm(a)
    b = np.cross(n0, a)
    centres = {c: np.asarray(poses[c])[:3, 3] for c in known}

    def plane(t: np.ndarray) -> tuple[np.ndarray, float]:
        n = n0 + t[0] * a + t[1] * b
        n /= np.linalg.norm(n)
        return n, -float(np.median(layer @ n))

    def residual(t: np.ndarray) -> np.ndarray:
        n, d = plane(t)
        logs_ = [math.log(max(float(n @ centres[c] + d), 1e-6) / known[c]) for c in known]
        mean = sum(logs_) / len(logs_)
        return np.r_[[x - mean for x in logs_], 0.05 * t]  # a little pull to the layer's own tilt

    got = least_squares(residual, np.zeros(2))
    tilt = math.degrees(math.atan(float(np.hypot(*got.x))))
    if tilt > HEIGHT_TILT_DEG:
        LOG.info("the floor would tilt %.1f degrees to the measured heights; kept as the cloud has it", tilt)
        return found
    n, d = plane(got.x)
    LOG.info("the floor tilted %.2f degrees so the cameras' heights keep the measured ratios", tilt)
    x, y, z = (float(v) for v in n)
    return (x, y, z, d)
