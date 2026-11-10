"""Checking and refining a fixed camera's pose from its owner spot against the plan's geometry (#292).

A view lifted from a fixed camera's spot on the plan (`spatial/spots.py`) could not be verified by measuring what share
of its 1.0 to 1.9 m wall band lay within 10 cm of `plan_import.walls` (`plan_lineup.measure`). On four fixed cameras on
the Linux machine, the spots as given put 1%, 9%, and 19% of each view's wall band on a plan wall, while poses 0.5 to
1.0 m away or turned 10 to 25 degrees scored up to 48 to 58%. Three causes combined:

1. `plan_import.walls` kept wide `cased` room openings and open doors as solid walls, and 82 to 99% of one view's band
   is foreground furniture (tables, couches, shelves) rather than walls, with unsigned distance ignoring visibility and
   surface normals.
2. Monocular metric depth (`da3metric-large`, run without camera intrinsics) is locally flat on each wall (0.3 to
   1.4 cm median plane residual) but warps near floor against far walls by 10 to 20%: scaling depth to bring the near
   floor to `z = 0` pulls far walls 0.4 to 0.6 m short of the plan, so shifting the camera forward or turning foreground
   clutter onto a cased opening puts more band points near a wall line.
3. For a camera with a wide barrel lens in `devices.toml`, `cameras.with_held` omitted the lens tuple on a fresh pose,
   leaving `spots.aim` to fit raw distorted rays against the undone focal (~6 degrees too steep on one live camera).

This module replaces unsigned band proximity with five geometric cues that separate orientation from position without
trusting monocular wall range to the centimetre:

- **Vanishing points (`vp`).** Straight segments in the undone picture are split into vertical candidates (scored
  against projected `[0, 0, 1]`, 35%) and horizontal candidates (scored against the plan's dominant wall axes, 65%), so
  vertical trim cannot mask a wrong yaw, and yaw, pitch, and roll lock from the RGB image alone.
- **Floor level and flatness (`floor_flat`).** Lifted floor patches must bring the floor to `z = 0` within `SPOT_SCALE`
  and face straight up.
- **Visibility and free space (`free_space`).** A 2D raycast from `(x, y)` against solid walls (`door` and `cased`
  openings cut) and opaque walls (`window` openings also cut) penalizes depth rays that pass through opaque walls at
  wall height or through solid walls along the floor.
- **Projected wireframe (`baseboard`, `wire`).** Visible `z = 0` wall-floor lines, room corners, and doorway or cased
  jambs are projected into the image and scored against depth creases, depth jumps, and RGB edges where unoccluded,
  locking `(x, y)` by 2D parallax rather than depth range.
- **Oriented wall planes and multi-wall consistency (`wall_norm`, `wall_coplanar`, `multi_wall`).** Background vertical
  patches must face inward like the raycast solid wall (`cos^6`), lie flat on their own plane (`wall_near`, within
  `PLANE_INLIER_M` of each wall's median depth), and keep consistent floor-calibrated range ratios across walls.

**The owner's spot stays the authority (`verify`).** `verify` searches inside `(spots.REFINE_M, spots.REFINE_DEG)`
(0.3 m, 5 degrees) and across a wider neighbourhood (`WIDE_M`, `WIDE_YAW_DEG`). An in-bound refinement is applied only
when it beats the spot pose by `REFINE_GAIN` and no pose beyond the bound beats the best in-bound pose by
`SUGGEST_MARGIN`. When a pose beyond the bound wins by at least `SUGGEST_MARGIN`, the owner's spot holds untouched and
`SpotCheck.suggestion` reports how far off the view wants to sit for the log and the Places page.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from liberational.spatial import plan_import, spots

N_BINS = 720  # 0.5-degree 2D visibility bins around the camera
_BIN_ANGLES = -math.pi + (np.arange(N_BINS) + 0.5) * (2.0 * math.pi / N_BINS)
_BIN_DIRS = np.stack([np.cos(_BIN_ANGLES), np.sin(_BIN_ANGLES)], axis=1)

MIN_VIEW_SIDE = 96  # smaller test fixtures have too few grid cells to refine a pose
MIN_GRID_POINTS = 200
MAX_DEPTH_M = 12.0
SPOT_FLOOR_M = 0.05
SPOT_SCALE = (0.75, 1.35)
PLANE_INLIER_M = 0.08  # metres from a wall's own median plane that count as coplanar on it
REFINE_GAIN = 0.015  # minimum score gain over the spot pose before an in-bound refinement moves it
SUGGEST_MARGIN = 0.05  # margin by which a beyond-bound pose must beat every in-bound pose to suggest moving the spot
SUGGEST_MIN_SCORE = 0.35
WIDE_M = 0.85
WIDE_YAW_DEG = 20.0
WIDE_PITCH_DEG = 10.0


@dataclass(frozen=True)
class PlanGeometry:
    """Clean wall, opening, corner, and axis geometry of an owner plan in its plan frame."""

    plan: plan_import.Parsed
    full: plan_import.Walls
    solid: plan_import.Walls  # doors and cased openings cut
    opaque: plan_import.Walls  # doors, cased openings, and windows cut
    up_3d: np.ndarray  # (1, 3) [0, 0, 1]
    horiz_3d: np.ndarray  # (K, 3) dominant horizontal wall directions
    corners: np.ndarray  # (C, 2) room polygon corners
    jambs: np.ndarray  # (J, 2) door and cased opening jambs


@dataclass(frozen=True)
class ViewFeatures:
    """Precomputed image lines, edge fields, and lifted camera-space 3D normals for one frame."""

    h: int
    w: int
    focal: float
    stride: int
    xm: np.ndarray
    ym: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    seg_len: np.ndarray
    seg_total: float
    pts_cam: np.ndarray
    n_cam: np.ndarray
    smooth: np.ndarray
    pts_grid: np.ndarray
    jump_s: np.ndarray
    crease_s: np.ndarray
    edge_vx_s: np.ndarray
    edge_mag_s: np.ndarray


@dataclass(frozen=True)
class PoseScore:
    """How well one candidate camera-to-plan pose aligns a view's lines, floor, free space, wireframe, and walls."""

    total: float
    vp: float = 0.0
    vp_horiz: float = 0.0
    vp_up: float = 0.0
    free_space: float = 0.0
    wall_norm: float = 0.0
    wall_coplanar: float = 0.0
    multi_wall: float = 1.0
    baseboard: float = 0.0
    wire: float = 0.0
    floor_flat: float = 0.5
    floor: float | None = None
    scale: float = 1.0
    wall_points: int = 0
    wall_median: float | None = None
    wall_near: float | None = None
    why: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": round(self.total, 4),
            "vp": round(self.vp, 4),
            "vp_horiz": round(self.vp_horiz, 4),
            "free_space": round(self.free_space, 4),
            "wall_norm": round(self.wall_norm, 4),
            "wall_coplanar": round(self.wall_coplanar, 4),
            "multi_wall": round(self.multi_wall, 4),
            "baseboard": round(self.baseboard, 4),
            "wire": round(self.wire, 4),
            "floor_flat": round(self.floor_flat, 4),
            "floor": None if self.floor is None else round(self.floor, 3),
            "scale": round(self.scale, 3),
            "wall_points": self.wall_points,
            "wall_median": None if self.wall_median is None else round(self.wall_median, 3),
            "wall_near": None if self.wall_near is None else round(self.wall_near, 3),
            "why": self.why,
        }


@dataclass(frozen=True)
class SpotCheck:
    """Result of verifying and refining a spot pose against the plan."""

    at_spot: PoseScore
    chosen: PoseScore
    pose: tuple[float, ...]  # 16 floats: refined pose when `refined` is True, else the spot's own pose
    refined: bool
    shift_m: float
    turn_deg: float
    suggestion: dict[str, float] | None
    best_in_bounds: PoseScore
    best_wide: PoseScore
    wide_shift_m: float
    wide_turn_deg: float


def prepare_plan(plan: plan_import.Parsed) -> PlanGeometry:
    """Extract solid walls, opaque walls, jambs, corners, and dominant wall directions from `plan`."""
    full = plan_import.walls(plan)
    solid = plan_import.sight_walls(plan, ("door", "cased"))
    opaque = plan_import.sight_walls(plan, ("door", "cased", "window"))
    jambs = plan_import.opening_jambs(plan, ("door", "cased"))

    vecs = full.b - full.a
    lengths = np.hypot(vecs[:, 0], vecs[:, 1]) if len(vecs) else np.zeros(0)
    horiz_axes: list[np.ndarray] = []
    for idx in np.argsort(-lengths):
        if lengths[idx] < 0.3:
            continue
        u = vecs[idx] / lengths[idx]
        if all(abs(float(u @ h)) < math.cos(math.radians(8.0)) for h in horiz_axes):
            horiz_axes.append(u)
        if len(horiz_axes) >= 4:
            break
    horiz_3d = np.array([[u[0], u[1], 0.0] for u in horiz_axes], dtype=np.float64).reshape(-1, 3)

    corners: list[tuple[float, float]] = []
    for room in plan.rooms:
        for cx, cy in room.polygon:
            if all(math.hypot(cx - ox, cy - oy) > 0.05 for ox, oy in corners):
                corners.append((cx, cy))

    return PlanGeometry(
        plan=plan,
        full=full,
        solid=solid,
        opaque=opaque,
        up_3d=np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
        horiz_3d=horiz_3d,
        corners=np.array(corners, dtype=np.float64).reshape(-1, 2),
        jambs=jambs,
    )


def _sample_bilinear(grid: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    gh, gw = grid.shape
    x0 = np.clip(np.floor(gx).astype(np.int64), 0, gw - 2)
    y0 = np.clip(np.floor(gy).astype(np.int64), 0, gh - 2)
    fx = np.clip(gx - x0, 0.0, 1.0)
    fy = np.clip(gy - y0, 0.0, 1.0)
    return (
        (1.0 - fx) * (1.0 - fy) * grid[y0, x0]
        + fx * (1.0 - fy) * grid[y0, x0 + 1]
        + (1.0 - fx) * fy * grid[y0 + 1, x0]
        + fx * fy * grid[y0 + 1, x0 + 1]
    )


def prepare_view(
    rgb: np.ndarray | None,
    depth: np.ndarray,
    mask: np.ndarray | None,
    focal: float,
    stride: int = 4,
) -> ViewFeatures:
    """Precompute straight segments, edge fields, and camera-space surface normals for one prepared frame."""
    import cv2

    h, w = depth.shape
    valid_px = depth > 0
    if mask is not None and mask.shape == (h, w):
        valid_px &= mask <= 127
    safe_px = cv2.erode(valid_px.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0

    xm = ym = ux = uy = seg_len = np.zeros(0, dtype=np.float64)
    if rgb is not None and rgb.shape[:2] == (h, w):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        raw_lines, _, _, _ = lsd.detect(gray)
        min_len = 0.035 * max(h, w)
        segs: list[tuple[float, float, float, float, float]] = []
        if raw_lines is not None:
            for x1, y1, x2, y2 in raw_lines.reshape(-1, 4):
                length = math.hypot(float(x2 - x1), float(y2 - y1))
                if length < min_len:
                    continue
                ts = np.linspace(0.0, 1.0, 7)
                xs = np.clip(np.round(x1 + ts * (x2 - x1)).astype(int), 0, w - 1)
                ys = np.clip(np.round(y1 + ts * (y2 - y1)).astype(int), 0, h - 1)
                if not safe_px[ys, xs].all():
                    continue
                segs.append((float(x1), float(y1), float(x2), float(y2), length))
        segs_arr = np.array(segs, dtype=np.float64).reshape(-1, 5)
        if len(segs_arr):
            xm = 0.5 * (segs_arr[:, 0] + segs_arr[:, 2]) + 0.5 - w / 2.0
            ym = 0.5 * (segs_arr[:, 1] + segs_arr[:, 3]) + 0.5 - h / 2.0
            ux = (segs_arr[:, 2] - segs_arr[:, 0]) / segs_arr[:, 4]
            uy = (segs_arr[:, 3] - segs_arr[:, 1]) / segs_arr[:, 4]
            seg_len = segs_arr[:, 4]
        gray_f = gray.astype(np.float32)
        gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
        gmag = np.hypot(gx, gy)
        gmag[~safe_px] = 0.0
        scale_g = max(float(np.percentile(gmag[safe_px], 92)) if safe_px.any() else 1.0, 10.0)
    else:
        gx = gmag = np.zeros((h, w), dtype=np.float32)
        scale_g = 1.0

    gv = np.arange(stride // 2, h, stride)
    gu = np.arange(stride // 2, w, stride)
    vv, uu = np.meshgrid(gv, gu, indexing="ij")
    z = depth[vv, uu].astype(np.float64)
    ok = (z > 0.1) & (z <= MAX_DEPTH_M)
    px = (uu + 0.5 - w / 2.0) * z / focal
    py = (vv + 0.5 - h / 2.0) * z / focal
    pts_grid = np.stack([px, py, z], axis=-1)

    dz_u = np.zeros_like(z)
    dz_v = np.zeros_like(z)
    if z.shape[1] >= 3:
        dz_u[:, 1:-1] = 0.5 * (z[:, 2:] - z[:, :-2])
    if z.shape[0] >= 3:
        dz_v[1:-1, :] = 0.5 * (z[2:, :] - z[:-2, :])
    dp_u = np.zeros_like(pts_grid)
    dp_v = np.zeros_like(pts_grid)
    if z.shape[1] >= 3:
        dp_u[:, 1:-1] = pts_grid[:, 2:] - pts_grid[:, :-2]
    if z.shape[0] >= 3:
        dp_v[1:-1, :] = pts_grid[2:, :] - pts_grid[:-2, :]

    n_raw = np.cross(dp_v, dp_u)
    n_norm = np.linalg.norm(n_raw, axis=-1)
    inner_ok = np.zeros_like(ok)
    if z.shape[0] >= 3 and z.shape[1] >= 3:
        inner_ok[1:-1, 1:-1] = ok[1:-1, 1:-1] & ok[1:-1, :-2] & ok[1:-1, 2:] & ok[:-2, 1:-1] & ok[2:, 1:-1]
    smooth = inner_ok & (np.abs(dz_u) < 0.08 * z) & (np.abs(dz_v) < 0.08 * z) & (n_norm > 1e-9)
    n_grid = np.zeros_like(pts_grid)
    n_grid[smooth] = n_raw[smooth] / n_norm[smooth, None]

    jump_rel = np.zeros_like(z)
    if z.shape[1] >= 3:
        jump_rel[:, 1:-1] = np.maximum(jump_rel[:, 1:-1], np.abs(z[:, 2:] - z[:, :-2]) / np.maximum(z[:, 1:-1], 0.5))
    if z.shape[0] >= 3:
        jump_rel[1:-1, :] = np.maximum(jump_rel[1:-1, :], np.abs(z[2:, :] - z[:-2, :]) / np.maximum(z[1:-1, :], 0.5))
    jump_rel[~inner_ok] = 0.0

    crease = np.zeros_like(z)
    if z.shape[1] >= 3:
        both_u = smooth[:, 2:] & smooth[:, :-2]
        crease[:, 1:-1][both_u] = np.maximum(
            crease[:, 1:-1][both_u],
            1.0 - np.clip(np.sum(n_grid[:, 2:][both_u] * n_grid[:, :-2][both_u], axis=-1), -1.0, 1.0),
        )
    if z.shape[0] >= 3:
        both_v = smooth[2:, :] & smooth[:-2, :]
        crease[1:-1, :][both_v] = np.maximum(
            crease[1:-1, :][both_v],
            1.0 - np.clip(np.sum(n_grid[2:, :][both_v] * n_grid[:-2, :][both_v], axis=-1), -1.0, 1.0),
        )

    crease_s = cv2.GaussianBlur(np.clip(crease / 0.35, 0.0, 1.0).astype(np.float32), (3, 3), 0.75)
    jump_s = cv2.GaussianBlur(np.clip(jump_rel / 0.12, 0.0, 1.0).astype(np.float32), (3, 3), 0.75)
    edge_mag_s = cv2.GaussianBlur(np.clip(gmag[vv, uu] / scale_g, 0.0, 1.0).astype(np.float32), (3, 3), 0.75)
    edge_vx_s = cv2.GaussianBlur(np.clip(np.abs(gx[vv, uu]) / scale_g, 0.0, 1.0).astype(np.float32), (3, 3), 0.75)

    return ViewFeatures(
        h=h,
        w=w,
        focal=focal,
        stride=stride,
        xm=xm,
        ym=ym,
        ux=ux,
        uy=uy,
        seg_len=seg_len,
        seg_total=float(seg_len.sum()) if len(seg_len) else 0.0,
        pts_cam=pts_grid[ok],
        n_cam=n_grid[ok],
        smooth=smooth[ok],
        pts_grid=pts_grid,
        jump_s=jump_s,
        crease_s=crease_s,
        edge_vx_s=edge_vx_s,
        edge_mag_s=edge_mag_s,
    )


def raycast_2d(origin: np.ndarray, w: plan_import.Walls) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distance, inward normal, and segment index of the nearest visible wall face in each of `N_BINS` directions."""
    r_hit = np.full(N_BINS, np.inf)
    n_hit = np.zeros((N_BINS, 2))
    idx_hit = np.full(N_BINS, -1, dtype=np.int64)
    if not len(w.a):
        return r_hit, n_hit, idx_hit
    ab = w.b - w.a
    seg_len = np.hypot(ab[:, 0], ab[:, 1])
    keep = seg_len > 1e-6
    a, ab, seg_len, half = w.a[keep], ab[keep], seg_len[keep], w.half[keep]
    t_hat = ab / seg_len[:, None]
    n0 = np.stack([-t_hat[:, 1], t_hat[:, 0]], axis=1)
    rel = origin[None, :] - a
    signed = np.sum(rel * n0, axis=1)
    inward = np.where(signed[:, None] >= 0, n0, -n0)
    shift = np.minimum(half, np.maximum(np.abs(signed) - 0.01, 0.0))
    fa = a + inward * shift[:, None]
    fa_rel = fa - origin[None, :]
    denom = _BIN_DIRS[None, :, 0] * ab[:, None, 1] - _BIN_DIRS[None, :, 1] * ab[:, None, 0]
    num_r = fa_rel[:, 0] * ab[:, 1] - fa_rel[:, 1] * ab[:, 0]
    num_s = fa_rel[:, None, 0] * _BIN_DIRS[None, :, 1] - fa_rel[:, None, 1] * _BIN_DIRS[None, :, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num_r[:, None] / denom
        s = num_s / denom
    valid = (np.abs(denom) > 1e-9) & (r > 0.02) & (s >= -1e-4) & (s <= 1.0001)
    r_masked = np.where(valid, r, np.inf)
    best_s = np.argmin(r_masked, axis=0)
    best_r = r_masked[best_s, np.arange(N_BINS)]
    hit = np.isfinite(best_r)
    r_hit[hit] = best_r[hit]
    n_hit[hit] = inward[best_s[hit]]
    idx_hit[hit] = best_s[hit]
    return r_hit, n_hit, idx_hit


def _dir_weights(
    vf: ViewFeatures, r_mat: np.ndarray, dirs_3d: np.ndarray, tol_px: float = 1.75, max_deg: float = 3.0
) -> np.ndarray:
    if not len(dirs_3d) or not len(vf.seg_len):
        return np.zeros(len(vf.seg_len))
    v_cam = dirs_3d @ r_mat
    lx = vf.focal * v_cam[:, 0:1] - v_cam[:, 2:3] * vf.xm[None, :]
    ly = vf.focal * v_cam[:, 1:2] - v_cam[:, 2:3] * vf.ym[None, :]
    lnorm = np.hypot(lx, ly)
    ok = lnorm > 1e-6
    dot = np.where(ok, np.abs(lx * vf.ux[None, :] + ly * vf.uy[None, :]) / np.maximum(lnorm, 1e-9), 0.0)
    sin_th = np.sqrt(np.maximum(0.0, 1.0 - dot * dot))
    err_px = 0.5 * vf.seg_len[None, :] * sin_th
    w = np.where(sin_th <= math.sin(math.radians(max_deg)), np.maximum(0.0, 1.0 - (err_px / tol_px) ** 2) ** 2, 0.0)
    return np.max(w, axis=0)


def vp_score(
    vf: ViewFeatures, r_mat: np.ndarray, up_3d: np.ndarray, horiz_3d: np.ndarray
) -> tuple[float, float, float]:
    """Combined vanishing-point score, horizontal wall-axis score, and vertical up-axis score."""
    if vf.seg_total <= 0:
        return 0.0, 0.0, 0.0
    w_up = _dir_weights(vf, r_mat, up_3d)
    w_hz = _dir_weights(vf, r_mat, horiz_3d)
    v_up = (up_3d @ r_mat)[0]
    lx_u = vf.focal * v_up[0] - v_up[2] * vf.xm
    ly_u = vf.focal * v_up[1] - v_up[2] * vf.ym
    ln_u = np.maximum(np.hypot(lx_u, ly_u), 1e-9)
    cos_to_up = np.abs(lx_u * vf.ux + ly_u * vf.uy) / ln_u
    vert_cand = cos_to_up >= math.cos(math.radians(18.0))
    non_vert = ~vert_cand
    len_v = float(vf.seg_len[vert_cand].sum())
    len_h = float(vf.seg_len[non_vert].sum())
    s_up = float(np.sum(w_up[vert_cand] * vf.seg_len[vert_cand]) / len_v) if len_v > 20.0 else 0.0
    s_hz = float(np.sum(w_hz[non_vert] * vf.seg_len[non_vert]) / len_h) if len_h > 20.0 else 0.0
    combined = 0.65 * s_hz + 0.35 * s_up if (len_v > 20.0 and len_h > 20.0) else max(s_hz, s_up)
    return combined, s_hz, s_up


def pose_delta(base_c2w: Sequence[float] | np.ndarray, other_c2w: Sequence[float] | np.ndarray) -> tuple[float, float]:
    """Shift in metres and 3D rotation angle in degrees between two camera-to-plan poses (`spots.within_bounds`)."""
    a = np.asarray(base_c2w, dtype=np.float64).reshape(4, 4)
    b = np.asarray(other_c2w, dtype=np.float64).reshape(4, 4)
    cos = (float(np.trace(a[:3, :3].T @ b[:3, :3])) - 1.0) / 2.0
    turned = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])), turned


def offset_pose(
    base_c2w: np.ndarray,
    dx: float = 0.0,
    dy: float = 0.0,
    dyaw: float = 0.0,
    dpitch: float = 0.0,
    droll: float = 0.0,
) -> np.ndarray:
    """`base_c2w` (right-handed camera-to-plan, +z up) shifted by `(dx, dy)` in plan metres at fixed height and
    turned by `dyaw` (clockwise seen from above, matching `Spot.yaw`), `dpitch` (down), and `droll` (clockwise)."""
    if dx == 0.0 and dy == 0.0 and dyaw == 0.0 and dpitch == 0.0 and droll == 0.0:
        return base_c2w.copy()
    r0 = base_c2w[:3, :3]
    g0 = r0[2, :3]
    pitch0 = math.degrees(math.asin(max(-1.0, min(1.0, float(-g0[2])))))
    roll0 = math.degrees(math.atan2(float(-g0[0]), float(-g0[1])))
    ahead0 = np.array([0.0, 0.0, 1.0]) - g0[2] * g0
    norm_a0 = float(np.linalg.norm(ahead0))
    if norm_a0 < 1e-6:
        return base_c2w.copy()
    ahead0 /= norm_a0
    d0 = r0[:2, :3] @ ahead0
    norm_d0 = float(np.linalg.norm(d0))
    if norm_d0 < 1e-6:
        return base_c2w.copy()
    d0 /= norm_d0
    # In a right-handed plan frame (+x east, +y north), a clockwise bearing increment turns `d0` by -dyaw.
    rad = -math.radians(dyaw)
    c, s = math.cos(rad), math.sin(rad)
    d = (float(d0[0] * c - d0[1] * s), float(d0[0] * s + d0[1] * c))
    side = (d[1], -d[0])
    g = spots.up_at(pitch0 + dpitch, roll0 + droll)
    ahead = np.array([0.0, 0.0, 1.0]) - g[2] * g
    norm_a = float(np.linalg.norm(ahead))
    if norm_a < 1e-6:
        return base_c2w.copy()
    ahead /= norm_a
    right = np.cross(ahead, g)
    out = np.eye(4, dtype=np.float64)
    out[0, :3] = d[0] * ahead + side[0] * right
    out[1, :3] = d[1] * ahead + side[1] * right
    out[2, :3] = g
    out[:3, 3] = (float(base_c2w[0, 3]) + dx, float(base_c2w[1, 3]) + dy, float(base_c2w[2, 3]))
    return out


def score_pose(
    vf: ViewFeatures,
    geom: PlanGeometry,
    c2w: np.ndarray,
    expected_room: str | None = None,
    allow_outside: bool = False,
) -> PoseScore:
    """Score one candidate camera-to-plan pose `c2w` against `geom`."""
    r_mat = c2w[:3, :3]
    t = c2w[:3, 3]
    h_cam = float(t[2])

    room = plan_import._room_of(geom.plan, float(t[0]), float(t[1]))
    if expected_room is not None and room != expected_room:
        return PoseScore(total=-1.0, why="outside_room")
    if expected_room is None and room is None and not allow_outside:
        return PoseScore(total=-1.0, why="outside_room")

    has_lines = vf.seg_total > 0
    vp, vp_hz, vp_up = vp_score(vf, r_mat, geom.up_3d, geom.horiz_3d) if has_lines else (0.0, 0.0, 0.0)
    if not len(vf.pts_cam):
        return PoseScore(total=vp, vp=vp, vp_horiz=vp_hz, vp_up=vp_up)

    pts_w_raw = vf.pts_cam @ r_mat.T + t
    n_w = vf.n_cam @ r_mat.T
    smooth = vf.smooth
    cos_15 = math.cos(math.radians(15.0))
    floor_mask = smooth & (n_w[:, 2] > cos_15) & (pts_w_raw[:, 2] < 0.5 * h_cam)
    floor_z = pts_w_raw[floor_mask, 2]
    scale = 1.0
    floor_level: float | None = None
    floor_flat = 0.5
    if len(floor_z) >= 50:
        bins = np.arange(floor_z.min(), floor_z.max() + 0.08, 0.04)
        if len(bins) >= 2:
            counts, edges = np.histogram(floor_z, bins)
            best = int(np.argmax(counts))
            if counts[best] >= 50:
                mid = 0.5 * (edges[best] + edges[best + 1])
                in_floor = floor_mask & (np.abs(pts_w_raw[:, 2] - mid) < 0.06)
                floor_level = float(np.median(pts_w_raw[in_floor, 2]))
                if abs(floor_level) > SPOT_FLOOR_M and h_cam - floor_level > 0.1:
                    scale = h_cam / (h_cam - floor_level)
                if not (SPOT_SCALE[0] <= scale <= SPOT_SCALE[1]):
                    return PoseScore(
                        total=-1.0, vp=vp, vp_horiz=vp_hz, vp_up=vp_up, floor=floor_level, scale=scale, why="floor"
                    )
                mean_n = n_w[in_floor].mean(axis=0)
                mean_n /= max(float(np.linalg.norm(mean_n)), 1e-9)
                tilt_err = math.degrees(math.acos(max(-1.0, min(1.0, float(mean_n[2])))))
                floor_flat = max(0.0, 1.0 - (tilt_err / 8.0) ** 2)

    r_solid, n_solid, idx_solid = raycast_2d(t[:2], geom.solid)
    r_opaque, _, _ = raycast_2d(t[:2], geom.opaque)

    pts_w = t + scale * (pts_w_raw - t)
    dxy_raw = pts_w_raw[:, :2] - t[:2]
    r_obs_raw = np.hypot(dxy_raw[:, 0], dxy_raw[:, 1])
    dxy = pts_w[:, :2] - t[:2]
    r_obs = np.hypot(dxy[:, 0], dxy[:, 1])
    ang = np.arctan2(dxy[:, 1], dxy[:, 0])
    bins_i = np.clip(((ang + math.pi) / (2.0 * math.pi) * N_BINS).astype(np.int64), 0, N_BINS - 1)

    mid_h = (pts_w[:, 2] >= 0.15) & (pts_w[:, 2] <= 2.15) & (r_obs > 0.25)
    if mid_h.any():
        rw_op = r_opaque[bins_i[mid_h]]
        over_op = np.maximum(0.0, np.minimum(r_obs[mid_h], r_obs_raw[mid_h]) - (1.10 * rw_op + 0.12))
        passthrough_mid = float(np.mean(np.minimum(over_op / 0.5, 1.0)))
    else:
        passthrough_mid = 0.0

    low_h = smooth & (n_w[:, 2] > cos_15) & (np.abs(pts_w[:, 2]) < 0.12) & (r_obs > 0.25)
    if low_h.any():
        rw_fl = r_solid[bins_i[low_h]]
        over_fl = np.maximum(0.0, r_obs[low_h] - (1.04 * rw_fl + 0.06))
        passthrough_floor = float(np.mean(np.minimum(over_fl / 0.25, 1.0)))
    else:
        passthrough_floor = 0.0
    free_space = 1.0 - min(1.0, 1.5 * passthrough_mid + 2.5 * passthrough_floor)

    w_im, h_im, f = vf.w, vf.h, vf.focal
    stride = vf.stride
    gh, gw = vf.crease_s.shape

    base_scores: list[np.ndarray] = []
    sw = geom.solid
    for si in range(len(sw.a)):
        wa, wb, wh = sw.a[si], sw.b[si], sw.half[si]
        ab = wb - wa
        slen = float(np.hypot(*ab))
        if slen < 0.25:
            continue
        t_hat = ab / slen
        n0 = np.array([-t_hat[1], t_hat[0]])
        if np.dot(t[:2] - wa, n0) < 0:
            n0 = -n0
        fa = wa + n0 * min(wh, max(abs(float(np.dot(t[:2] - wa, n0))) - 0.01, 0.0))
        ns = max(2, int(slen / 0.15))
        ss = (np.arange(ns) + 0.5) / ns * slen
        pts_2d = fa[None, :] + ss[:, None] * t_hat[None, :]
        d2 = pts_2d - t[:2]
        dist2 = np.hypot(d2[:, 0], d2[:, 1])
        angs = np.arctan2(d2[:, 1], d2[:, 0])
        bs = np.clip(((angs + math.pi) / (2.0 * math.pi) * N_BINS).astype(np.int64), 0, N_BINS - 1)
        vis = np.abs(r_solid[bs] - dist2) < 0.08
        if not vis.any():
            continue
        pts_3d = np.column_stack([pts_2d[vis], np.zeros(int(vis.sum()))])
        p_cam = (pts_3d - t) @ r_mat
        in_front = p_cam[:, 2] > 0.4
        if not in_front.any():
            continue
        p_cam = p_cam[in_front]
        gx_f = (f * p_cam[:, 0] / p_cam[:, 2] + w_im / 2.0 - 0.5 - stride // 2) / stride
        gy_f = (f * p_cam[:, 1] / p_cam[:, 2] + h_im / 2.0 - 0.5 - stride // 2) / stride
        in_img = (gx_f >= 1.0) & (gx_f <= gw - 2.0) & (gy_f >= 1.0) & (gy_f <= gh - 2.0)
        if not in_img.any():
            continue
        gx_f, gy_f, z_ex = gx_f[in_img], gy_f[in_img], p_cam[in_img, 2]
        z_ob = scale * vf.pts_grid[np.round(gy_f).astype(int), np.round(gx_f).astype(int), 2]
        unocc = z_ob >= 0.65 * z_ex
        if not unocc.any():
            continue
        c_vals = _sample_bilinear(vf.crease_s, gx_f[unocc], gy_f[unocc])
        e_vals = _sample_bilinear(vf.edge_mag_s, gx_f[unocc], gy_f[unocc])
        base_scores.append(0.65 * c_vals + 0.35 * e_vals)
    baseboard_score = float(np.concatenate(base_scores).mean()) if base_scores else 0.0

    vert_scores: list[float] = []
    for kind_f, pts_feat in (("corner", geom.corners), ("jamb", geom.jambs)):
        for fx, fy in pts_feat:
            dx_f, dy_f = float(fx - t[0]), float(fy - t[1])
            dist_f = math.hypot(dx_f, dy_f)
            if dist_f < 0.4 or dist_f > 9.0:
                continue
            b_f = int(np.clip(((math.atan2(dy_f, dx_f) + math.pi) / (2.0 * math.pi) * N_BINS), 0, N_BINS - 1))
            b_lo, b_hi = (b_f - 2) % N_BINS, (b_f + 2) % N_BINS
            if min(r_solid[b_lo], r_solid[b_f], r_solid[b_hi]) < dist_f - 0.22:
                continue
            zs = np.linspace(0.40, 1.85, 8)
            pts_3d = np.column_stack([np.full_like(zs, fx), np.full_like(zs, fy), zs])
            p_cam = (pts_3d - t) @ r_mat
            in_front = p_cam[:, 2] > 0.4
            if in_front.sum() < 3:
                continue
            p_cam = p_cam[in_front]
            gx_f = (f * p_cam[:, 0] / p_cam[:, 2] + w_im / 2.0 - 0.5 - stride // 2) / stride
            gy_f = (f * p_cam[:, 1] / p_cam[:, 2] + h_im / 2.0 - 0.5 - stride // 2) / stride
            in_img = (gx_f >= 1.0) & (gx_f <= gw - 2.0) & (gy_f >= 1.0) & (gy_f <= gh - 2.0)
            if in_img.sum() < 2:
                continue
            gx_f, gy_f, z_ex = gx_f[in_img], gy_f[in_img], p_cam[in_img, 2]
            z_ob = scale * vf.pts_grid[np.round(gy_f).astype(int), np.round(gx_f).astype(int), 2]
            unocc = (z_ob <= 0) | (z_ob >= 0.65 * z_ex)
            if unocc.sum() < 2:
                continue
            c_v = _sample_bilinear(vf.crease_s, gx_f[unocc], gy_f[unocc])
            ex_v = _sample_bilinear(vf.edge_vx_s, gx_f[unocc], gy_f[unocc])
            if kind_f == "jamb":
                j_v = _sample_bilinear(vf.jump_s, gx_f[unocc], gy_f[unocc])
                vert_scores.append(float(np.mean(np.maximum(j_v, 0.5 * ex_v + 0.5 * c_v))))
            else:
                vert_scores.append(float(np.mean(0.55 * c_v + 0.45 * ex_v)))
    wire_score = float(np.mean(vert_scores)) if vert_scores else 0.0

    vert_mask = (
        smooth
        & (np.abs(n_w[:, 2]) < math.sin(math.radians(22.0)))
        & (pts_w[:, 2] >= 0.30)
        & (pts_w[:, 2] <= 2.20)
        & (r_obs > 0.30)
    )
    wall_norm_score = 0.0
    wall_coplanar_score = 0.0
    multi_wall = 1.0
    wall_near: float | None = None
    wall_points = 0
    wall_median: float | None = None
    if vert_mask.sum() >= 30:
        b_v = bins_i[vert_mask]
        rw_v = r_solid[b_v]
        nw_v = n_solid[b_v]
        wid_v = idx_solid[b_v]
        ro_v = r_obs[vert_mask]
        dxy_v = dxy[vert_mask]
        nxy_v = n_w[vert_mask, :2]
        nxy_unit = nxy_v / np.maximum(np.hypot(nxy_v[:, 0], nxy_v[:, 1])[:, None], 1e-9)
        cos_n = np.sum(nxy_unit * nw_v, axis=1)

        bg = np.isfinite(rw_v) & (ro_v >= 0.62 * rw_v) & (ro_v <= 1.35 * rw_v)
        if bg.sum() >= 25:
            wall_norm_score = float(np.mean(np.maximum(0.0, cos_n[bg]) ** 6))
            aligned = bg & (cos_n > math.cos(math.radians(20.0)))
            wall_points = int(aligned.sum())
            if wall_points >= 20:
                wall_ratios: list[float] = []
                wall_weights: list[int] = []
                plane_res_list: list[np.ndarray] = []
                for wid in np.unique(wid_v[aligned]):
                    if wid < 0:
                        continue
                    sel = aligned & (wid_v == wid)
                    cnt = int(sel.sum())
                    if cnt < 15:
                        continue
                    n_s = nw_v[sel][0]
                    d_obs_perp = -(dxy_v[sel] @ n_s)
                    ray_cos = np.maximum(np.abs(_BIN_DIRS[b_v[sel]] @ n_s), 0.15)
                    d_plan_med = float(np.median(rw_v[sel] * ray_cos))
                    d_obs_med = float(np.median(d_obs_perp))
                    if d_plan_med < 0.3:
                        continue
                    plane_res_list.append(np.abs(d_obs_perp - d_obs_med))
                    wall_ratios.append(d_obs_med / d_plan_med)
                    wall_weights.append(cnt)

                if plane_res_list:
                    cat_res = np.concatenate(plane_res_list)
                    wall_median = float(np.median(cat_res))
                    wall_near = float((cat_res <= PLANE_INLIER_M).mean())
                    wall_coplanar_score = wall_near * min(1.0, len(cat_res) / 250.0)

                if wall_ratios:
                    ratios_arr = np.array(wall_ratios, dtype=np.float64)
                    wts_arr = np.array(wall_weights, dtype=np.float64)
                    wts_arr /= wts_arr.sum()
                    mean_r = float(wts_arr @ ratios_arr)
                    excess = max(0.0, 0.72 - mean_r, mean_r - 1.18)
                    range_ok = max(0.0, 1.0 - (excess / 0.20) ** 2)
                    if len(wall_ratios) >= 2:
                        rel_dev = float(wts_arr @ np.abs(ratios_arr - mean_r)) / max(mean_r, 0.5)
                        multi_wall = range_ok * max(0.0, 1.0 - (rel_dev / 0.14) ** 2)
                    else:
                        multi_wall = range_ok

    if has_lines:
        raw_total = (
            0.35 * vp
            + 0.20 * wall_norm_score
            + 0.15 * wall_coplanar_score * multi_wall
            + 0.20 * (0.5 * baseboard_score + 0.5 * wire_score)
            + 0.10 * floor_flat
        )
    else:
        raw_total = (
            0.35 * wall_norm_score
            + 0.25 * wall_coplanar_score * multi_wall
            + 0.25 * (0.5 * baseboard_score + 0.5 * wire_score)
            + 0.15 * floor_flat
        )
    total = raw_total * free_space * (0.5 + 0.5 * multi_wall)
    return PoseScore(
        total=float(total),
        vp=float(vp),
        vp_horiz=float(vp_hz),
        vp_up=float(vp_up),
        free_space=float(free_space),
        wall_norm=float(wall_norm_score),
        wall_coplanar=float(wall_coplanar_score),
        multi_wall=float(multi_wall),
        baseboard=float(baseboard_score),
        wire=float(wire_score),
        floor_flat=float(floor_flat),
        floor=floor_level,
        scale=float(scale),
        wall_points=wall_points,
        wall_median=wall_median,
        wall_near=wall_near,
        why=None,
    )


def verify(vf: ViewFeatures, geom: PlanGeometry, spot_c2w: np.ndarray, refine: bool = True) -> SpotCheck:
    """Verify `spot_c2w` against `geom`, refine within `(spots.REFINE_M, spots.REFINE_DEG)` when warranted, and detect
    any clear disagreement beyond that bound without moving past it."""
    base = np.asarray(spot_c2w, dtype=np.float64).reshape(4, 4)
    expected_room = plan_import._room_of(geom.plan, float(base[0, 3]), float(base[1, 3]))
    at_spot = score_pose(vf, geom, base, expected_room, allow_outside=True)
    flat_spot = tuple(float(x) for x in base.ravel())
    if not refine or min(vf.h, vf.w) < MIN_VIEW_SIDE or len(vf.pts_cam) < MIN_GRID_POINTS:
        return SpotCheck(at_spot, at_spot, flat_spot, False, 0.0, 0.0, None, at_spot, at_spot, 0.0, 0.0)

    def in_bound(cand: np.ndarray) -> bool:
        sh, tr = pose_delta(base, cand)
        return sh <= spots.REFINE_M - 1e-6 and tr <= spots.REFINE_DEG - 1e-6

    best_in = (at_spot.total, (0.0, 0.0, 0.0, 0.0, 0.0), base, at_spot)
    best_rot = (at_spot.total, (0.0, 0.0, 0.0))
    for dyaw in np.linspace(-4.5, 4.5, 7):
        for dpitch in np.linspace(-4.5, 4.5, 7):
            for droll in (-2.0, 0.0, 2.0):
                cand = offset_pose(base, 0.0, 0.0, float(dyaw), float(dpitch), float(droll))
                if not in_bound(cand):
                    continue
                sc = score_pose(vf, geom, cand, expected_room)
                if sc.total > best_rot[0]:
                    best_rot = (sc.total, (float(dyaw), float(dpitch), float(droll)))
                if sc.total > best_in[0]:
                    best_in = (sc.total, (0.0, 0.0, float(dyaw), float(dpitch), float(droll)), cand, sc)

    seed_rots = {(0.0, 0.0, 0.0), best_rot[1]}
    for ry, rp, rr in seed_rots:
        for dx in np.linspace(-0.28, 0.28, 7):
            for dy in np.linspace(-0.28, 0.28, 7):
                if math.hypot(float(dx), float(dy)) > spots.REFINE_M:
                    continue
                cand = offset_pose(base, float(dx), float(dy), ry, rp, rr)
                if not in_bound(cand):
                    continue
                sc = score_pose(vf, geom, cand, expected_room)
                if sc.total > best_in[0]:
                    best_in = (sc.total, (float(dx), float(dy), ry, rp, rr), cand, sc)

    # Fine in-bound polish around the best joint offset.
    _, (bx, by, byaw, bpitch, broll), _, _ = best_in
    for dyaw in byaw + np.array([-1.0, -0.5, 0.0, 0.5, 1.0]):
        for dpitch in bpitch + np.array([-1.0, -0.5, 0.0, 0.5, 1.0]):
            for droll in broll + np.array([-1.0, 0.0, 1.0]):
                cand = offset_pose(base, bx, by, float(dyaw), float(dpitch), float(droll))
                if not in_bound(cand):
                    continue
                sc = score_pose(vf, geom, cand, expected_room)
                if sc.total > best_in[0]:
                    best_in = (sc.total, (bx, by, float(dyaw), float(dpitch), float(droll)), cand, sc)
    _, (bx, by, byaw, bpitch, broll), _, _ = best_in
    for dx in bx + np.array([-0.08, -0.04, 0.0, 0.04, 0.08]):
        for dy in by + np.array([-0.08, -0.04, 0.0, 0.04, 0.08]):
            cand = offset_pose(base, float(dx), float(dy), byaw, bpitch, broll)
            if not in_bound(cand):
                continue
            sc = score_pose(vf, geom, cand, expected_room)
            if sc.total > best_in[0]:
                best_in = (sc.total, (float(dx), float(dy), byaw, bpitch, broll), cand, sc)

    # Wider check beyond the spot bound: does a pose outside (0.3 m, 5 deg) fit clearly better?
    best_wide = best_in
    wide_rot = (best_in[0], (byaw, bpitch, broll))
    for dyaw in np.linspace(-WIDE_YAW_DEG, WIDE_YAW_DEG, 17):
        for dpitch in (-10.0, -6.0, -3.0, 0.0, 3.0, 6.0, 10.0):
            for droll in (-3.0, 0.0, 3.0):
                cand = offset_pose(base, 0.0, 0.0, float(dyaw), float(dpitch), float(droll))
                sc = score_pose(vf, geom, cand, expected_room)
                if sc.total > wide_rot[0]:
                    wide_rot = (sc.total, (float(dyaw), float(dpitch), float(droll)))
                if sc.total > best_wide[0]:
                    best_wide = (sc.total, (0.0, 0.0, float(dyaw), float(dpitch), float(droll)), cand, sc)

    for ry, rp, rr in {(0.0, 0.0, 0.0), (byaw, bpitch, broll), wide_rot[1]}:
        for dx in np.linspace(-WIDE_M, WIDE_M, 9):
            for dy in np.linspace(-WIDE_M, WIDE_M, 9):
                if math.hypot(float(dx), float(dy)) > WIDE_M + 0.1:
                    continue
                cand = offset_pose(base, float(dx), float(dy), ry, rp, rr)
                sc = score_pose(vf, geom, cand, expected_room)
                if sc.total > best_wide[0]:
                    best_wide = (sc.total, (float(dx), float(dy), ry, rp, rr), cand, sc)

    _, (wx, wy, wyaw, wpitch, wroll), _, _ = best_wide
    for dyaw in wyaw + np.array([-1.5, 0.0, 1.5]):
        for dpitch in wpitch + np.array([-1.5, 0.0, 1.5]):
            cand = offset_pose(base, wx, wy, float(dyaw), float(dpitch), wroll)
            sc = score_pose(vf, geom, cand, expected_room)
            if sc.total > best_wide[0]:
                best_wide = (sc.total, (wx, wy, float(dyaw), float(dpitch), wroll), cand, sc)
    _, (wx, wy, wyaw, wpitch, wroll), _, _ = best_wide
    for dx in wx + np.array([-0.10, 0.0, 0.10]):
        for dy in wy + np.array([-0.10, 0.0, 0.10]):
            cand = offset_pose(base, float(dx), float(dy), wyaw, wpitch, wroll)
            sc = score_pose(vf, geom, cand, expected_room)
            if sc.total > best_wide[0]:
                best_wide = (sc.total, (float(dx), float(dy), wyaw, wpitch, wroll), cand, sc)

    _, _, in_c2w, in_sc = best_in
    _, _, wide_c2w, wide_sc = best_wide
    in_shift, in_turn = pose_delta(base, in_c2w)
    wide_shift, wide_turn = pose_delta(base, wide_c2w)
    beyond = wide_shift > spots.REFINE_M + 1e-4 or wide_turn > spots.REFINE_DEG + 1e-3

    if (
        beyond
        and wide_sc.total >= max(at_spot.total, in_sc.total) + SUGGEST_MARGIN
        and wide_sc.total >= SUGGEST_MIN_SCORE
    ):
        suggestion = {
            "shift_m": round(wide_shift, 2),
            "turn_deg": round(wide_turn, 1),
            "score": round(wide_sc.total, 3),
            "at_spot": round(at_spot.total, 3),
        }
        return SpotCheck(
            at_spot=at_spot,
            chosen=at_spot,
            pose=flat_spot,
            refined=False,
            shift_m=0.0,
            turn_deg=0.0,
            suggestion=suggestion,
            best_in_bounds=in_sc,
            best_wide=wide_sc,
            wide_shift_m=wide_shift,
            wide_turn_deg=wide_turn,
        )

    if in_sc.total >= at_spot.total + REFINE_GAIN and (in_shift > 1e-3 or in_turn > 0.05):
        return SpotCheck(
            at_spot=at_spot,
            chosen=in_sc,
            pose=tuple(float(x) for x in in_c2w.ravel()),
            refined=True,
            shift_m=in_shift,
            turn_deg=in_turn,
            suggestion=None,
            best_in_bounds=in_sc,
            best_wide=wide_sc,
            wide_shift_m=wide_shift,
            wide_turn_deg=wide_turn,
        )

    return SpotCheck(
        at_spot=at_spot,
        chosen=at_spot,
        pose=flat_spot,
        refined=False,
        shift_m=0.0,
        turn_deg=0.0,
        suggestion=None,
        best_in_bounds=in_sc,
        best_wide=wide_sc,
        wide_shift_m=wide_shift,
        wide_turn_deg=wide_turn,
    )
