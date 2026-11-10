"""Replay a video through the learn-surroundings gate and map step into scratch state (#281).

Feeds a recorded video's frames through the real sampler (`spatial/surroundings_worker.py`), gate
(`spatial/surroundings_frames.py`), and map step (`spatial/surroundings_map.py`) as if a camera were being carried,
writing only into a scratch state directory. When an ARKit `odometry.csv` is given with `--odometry`, it is read
only after the run finishes to score viewpoint coverage, pose accuracy, chain breaks, walk drift, and held-out
splat quality.

    PACK_ROOT=/path/to/checkout .venv/bin/python tools/replay_surroundings.py VIDEO.mp4 [--rotate 90] \\
        [--odometry odometry.csv] [--state /tmp/replay-state] [--gate-only]

Prints a JSON report to stdout. Defaults to a temporary scratch directory that is cleaned up on exit unless
`--state` names one to keep.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from liberational import config as cfg
from liberational import logs, nvml
from liberational.memory import places
from liberational.spatial import cameras, lens, surroundings_positions
from liberational.spatial import surroundings_frames as frames
from liberational.spatial import surroundings_map as sm
from liberational.spatial import surroundings_worker as worker

#: Quarter turns clockwise (degrees -> (3x3 matrix taking turned camera coords to sensor coords, cv2.rotate code)).
TURNS: dict[int, tuple[np.ndarray, int | None]] = {
    0: (np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), None),
    90: (np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), cv2.ROTATE_90_CLOCKWISE),
    180: (np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]), cv2.ROTATE_180),
    270: (np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), cv2.ROTATE_90_COUNTERCLOCKWISE),
}

COVER_M = 1.5  # a walk pose is covered when a view sits within this distance...
COVER_DEG = 35.0  # ...and looks within this angle
REDUNDANT_M = 0.25  # two kept views closer than this...
REDUNDANT_DEG = 8.0  # ...and turned less than this are near-duplicates
BREAK_M = 0.75  # consecutive placed views whose step disagrees with ground truth past this...
BREAK_DEG = 20.0  # ...or past this turn count as a chain break
REVISIT_GAP_S = 30.0  # seconds apart before two nearby poses count as a revisit / loop
REVISIT_M = 1.5  # ground-truth distance under which a revisit is scored
SHIFT_REACH = 3
SHIFT_AGREE_S = 0.004
SHIFT_SHARE = 0.8
SHIFT_GAIN = 0.1
HOLDOUT_COUNT = 12
HOLDOUT_SIDE = 448


@dataclass(frozen=True)
class SampleRecord:
    """What the sampler saw at one sampled video frame (kept in memory for scoring only)."""

    video_frame: int
    t: float
    outcome: str
    kept_id: str | None
    sharpness: float
    brightness: float


@dataclass(frozen=True)
class OdometryRow:
    """One ground-truth pose and intrinsics row from an ARKit `odometry.csv`."""

    t: float
    c2w: np.ndarray
    k: np.ndarray
    frame: int


def rotate_frame(pixels: np.ndarray, degrees: int) -> np.ndarray:
    """Turn `pixels` clockwise by `degrees` (0, 90, 180, or 270)."""
    code = TURNS[degrees][1]
    return pixels if code is None else cv2.rotate(pixels, code)


def turned_intrinsics(k: np.ndarray, size: tuple[int, int], target_size: tuple[int, int], degrees: int) -> np.ndarray:
    """Intrinsics `k` of a raw `size` (width, height) video frame after scaling and rotating clockwise by `degrees`
    to `target_size` (width, height)."""
    w, h = size
    tw, th = target_size
    raw_w, raw_h = (th, tw) if degrees in (90, 270) else (tw, th)
    sx, sy = raw_w / w, raw_h / h
    scale = np.diag([sx, sy, 1.0])
    turn = TURNS[degrees][0]
    rot = {
        0: np.eye(3),
        90: np.array([[0.0, -1.0, raw_h - 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        180: np.array([[-1.0, 0.0, raw_w - 1.0], [0.0, -1.0, raw_h - 1.0], [0.0, 0.0, 1.0]]),
        270: np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, raw_w - 1.0], [0.0, 0.0, 1.0]]),
    }[degrees]
    return rot @ scale @ np.asarray(k, dtype=np.float64) @ turn


def iter_video(
    video: Path,
    every_s: float,
    rotate_deg: int,
    max_side: int | None,
    limit_s: float | None,
    skip_through: int = -1,
) -> tuple[tuple[int, int], list[tuple[int, float]], Iterator[tuple[int, float, np.ndarray]]]:
    """Open `video`, return its raw `(width, height)`, a timing list populated during iteration, and an iterator of
    `(frame_index, seconds, bgr_pixels)` sampled every `every_s` seconds."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video}")
    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    timed: list[tuple[int, float]] = []

    def gen() -> Iterator[tuple[int, float, np.ndarray]]:
        due = 0.0
        idx = 0
        start_t: float | None = None
        try:
            while True:
                if idx <= skip_through:
                    if not cap.grab():
                        break
                    msec = float(cap.get(cv2.CAP_PROP_POS_MSEC))
                    raw_t = msec / 1000.0 if msec > 0 or idx == 0 else idx / fps
                    if start_t is None:
                        start_t = raw_t
                    if idx < 300 and idx % 10 == 0:
                        timed.append((idx, raw_t))
                    rel_t = max(0.0, raw_t - start_t)
                    if limit_s is not None and rel_t > limit_s:
                        break
                    if rel_t + 1e-6 >= due:
                        due = (rel_t if rel_t - due > every_s else due) + every_s
                    idx += 1
                    continue
                ok, frame = cap.read()
                if not ok:
                    break
                msec = float(cap.get(cv2.CAP_PROP_POS_MSEC))
                raw_t = msec / 1000.0 if msec > 0 or idx == 0 else idx / fps
                if start_t is None:
                    start_t = raw_t
                if idx < 300 and idx % 10 == 0:
                    timed.append((idx, raw_t))
                rel_t = max(0.0, raw_t - start_t)
                if limit_s is not None and rel_t > limit_s:
                    break
                if rel_t + 1e-6 >= due:
                    due = (rel_t if rel_t - due > every_s else due) + every_s
                    img = rotate_frame(frame, rotate_deg)
                    if max_side is not None and max(img.shape[:2]) > max_side:
                        h, w = img.shape[:2]
                        s = max_side / max(w, h)
                        img = cv2.resize(
                            img, (max(2, round(w * s)), max(2, round(h * s))), interpolation=cv2.INTER_AREA
                        )
                    yield idx, rel_t, img
                idx += 1
        finally:
            cap.release()

    return (raw_w, raw_h), timed, gen()


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity `(scale, rotation, shift)` mapping `src` (N x 3) onto `dst`: `dst ~ s * R @ src + t`."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    u, d, vt = np.linalg.svd(xd.T @ xs / len(src))
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1.0
    r = u @ sign @ vt
    var = float((xs**2).sum() / len(src))
    s = float(np.trace(np.diag(d) @ sign) / var) if var > 1e-12 else 1.0
    return s, r, mu_d - s * (r @ mu_s)


def rot_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in degrees between two 3x3 rotation matrices."""
    cos = (float(np.trace(a.T @ b)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def read_odometry(path: Path) -> dict[int, OdometryRow]:
    """Read an ARKit `odometry.csv` keyed by recording frame index."""
    fallback: np.ndarray | None = None
    matrix_file = path.with_name("camera_matrix.csv")
    if matrix_file.is_file():
        fallback = np.loadtxt(matrix_file, delimiter=",").reshape(3, 3)
    out: dict[int, OdometryRow] = {}
    with path.open(newline="") as f:
        rows = csv.reader(f)
        header = [h.strip() for h in next(rows)]
        for raw in rows:
            if not raw:
                continue
            row = {h: v.strip() for h, v in zip(header, raw, strict=False)}
            c2w = np.eye(4, dtype=np.float64)
            c2w[:3, :3] = Rotation.from_quat([float(row[q]) for q in ("qx", "qy", "qz", "qw")]).as_matrix()
            c2w[:3, 3] = [float(row[a]) for a in ("x", "y", "z")]
            if row.get("fx"):
                k = np.array(
                    [
                        [float(row["fx"]), 0.0, float(row["cx"])],
                        [0.0, float(row["fy"]), float(row["cy"])],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )
            elif fallback is not None:
                k = fallback.copy()
            else:
                raise ValueError(f"{path}: missing intrinsics")
            idx = int(row["frame"])
            out[idx] = OdometryRow(float(row["timestamp"]), c2w, k, idx)
    return out


def video_shift(rows: Mapping[int, OdometryRow], timed: Sequence[tuple[int, float]]) -> int:
    """Decoded frame offset against `odometry.csv` (FFmpeg drops pre-zero edit-list frames on some exports)."""
    if len(timed) < 2:
        return 0
    index = np.array([i for i, _ in timed])
    seconds = np.array([t for _, t in timed], dtype=np.float64)

    def agreement(shift: int) -> float:
        known = np.array([i + shift in rows for i in index])
        both = known[:-1] & known[1:]
        if not both.any():
            return 0.0
        row_t = np.array([rows[i + shift].t if ok else np.nan for i, ok in zip(index, known, strict=True)])
        return float((np.abs(np.diff(seconds) - np.diff(row_t))[both] <= SHIFT_AGREE_S).mean())

    shares = {s: agreement(s) for s in range(-SHIFT_REACH, SHIFT_REACH + 1)}
    best = max(shares, key=lambda s: (shares[s], -abs(s)))
    return best if shares[best] >= SHIFT_SHARE and shares[best] >= shares[0] + SHIFT_GAIN else 0


def _apply_drops(store: frames.Store) -> None:
    state = sm.State.read(store.folder)
    for frame_id in state.drop:
        store.remove(frame_id)


def _snapshot_splats(state_dir: Path, snap_dir: Path, saved: dict[str, Path]) -> None:
    """Copy any newly written `.gs.ply` files into `snap_dir` before later revisions prune them."""
    for place in places.place_ids(state_dir):
        for rev in places.revisions(state_dir, place):
            if rev.id in saved:
                continue
            src = places.splat_path(state_dir, rev, "gs.ply")
            if src.is_file():
                dst = snap_dir / f"{place}__{rev.id}.gs.ply"
                shutil.copy2(src, dst)
                saved[rev.id] = dst


def score_run(
    *,
    video: Path,
    odometry_path: Path,
    raw_size: tuple[int, int],
    timed: Sequence[tuple[int, float]],
    rotate_deg: int,
    max_side: int | None,
    samples: Sequence[SampleRecord],
    kept_frames: Mapping[str, int],
    state_dir: Path,
    splat_snaps: Mapping[str, Path],
    limit_s: float | None,
    score_splat: bool,
) -> dict[str, Any]:
    """Score the finished scratch state against `odometry.csv` without mutating state."""
    raw_rows = read_odometry(odometry_path)
    shift = video_shift(raw_rows, timed)
    turn = TURNS[rotate_deg][0]
    t0 = min(r.t for r in raw_rows.values())
    by_video: dict[int, OdometryRow] = {}
    for v_idx, row in ((k - shift, v) for k, v in raw_rows.items() if k >= shift):
        if limit_s is not None and row.t - t0 > limit_s:
            continue
        c2w = row.c2w.copy()
        c2w[:3, :3] = c2w[:3, :3] @ turn
        by_video[v_idx] = OdometryRow(row.t - t0, c2w, row.k, row.frame)

    ordered_keys = sorted(by_video)
    all_poses = np.stack([by_video[k].c2w for k in ordered_keys])
    all_centres = all_poses[:, :3, 3]
    all_fwd = all_poses[:, :3, 2]
    path_m = float(np.linalg.norm(np.diff(all_centres, axis=0), axis=1).sum()) if len(all_centres) > 1 else 0.0
    duration_s = float(by_video[ordered_keys[-1]].t - by_video[ordered_keys[0]].t) if len(ordered_keys) > 1 else 0.0
    start_end_gt_m = float(np.linalg.norm(all_centres[-1] - all_centres[0])) if len(all_centres) > 1 else 0.0

    store = frames.Store(frames.root(state_dir))
    map_state = sm.State.read(store.folder)
    kept_final = [
        f for f in sorted(store.kept, key=lambda x: x.t) if f.id in kept_frames and kept_frames[f.id] in by_video
    ]

    def coverage(view_poses_gt: np.ndarray) -> dict[str, float]:
        if not len(view_poses_gt):
            return {"covered_share": 0.0, "nearest_m_median": -1.0, "nearest_m_p90": -1.0}
        v_pos = view_poses_gt[:, :3, 3]
        v_fwd = view_poses_gt[:, :3, 2]
        dists = np.linalg.norm(all_centres[:, None, :] - v_pos[None, :, :], axis=2)  # N x M
        cos = np.clip(all_fwd @ v_fwd.T, -1.0, 1.0)
        degs = np.degrees(np.arccos(cos))
        ok = ((dists <= COVER_M) & (degs <= COVER_DEG)).any(axis=1)
        nearest_m = dists.min(axis=1)
        return {
            "covered_share": round(float(ok.mean()), 4),
            "nearest_m_median": round(float(np.median(nearest_m)), 3),
            "nearest_m_p90": round(float(np.percentile(nearest_m, 90)), 3),
        }

    kept_gt = np.stack([by_video[kept_frames[f.id]].c2w for f in kept_final]) if kept_final else np.zeros((0, 4, 4))
    redundancy: dict[str, Any] = {"step_m_median": None, "step_deg_median": None, "near_duplicate_share": 0.0}
    if len(kept_gt) >= 2:
        step_m = np.linalg.norm(np.diff(kept_gt[:, :3, 3], axis=0), axis=1)
        step_deg = [rot_angle_deg(kept_gt[i, :3, :3], kept_gt[i + 1, :3, :3]) for i in range(len(kept_gt) - 1)]
        dup = 0
        for i in range(1, len(kept_gt)):
            d_m = np.linalg.norm(kept_gt[:i, :3, 3] - kept_gt[i, :3, 3], axis=1)
            d_deg = np.array([rot_angle_deg(kept_gt[j, :3, :3], kept_gt[i, :3, :3]) for j in range(i)])
            if bool(((d_m <= REDUNDANT_M) & (d_deg <= REDUNDANT_DEG)).any()):
                dup += 1
        redundancy = {
            "step_m_median": round(float(np.median(step_m)), 3),
            "step_deg_median": round(float(np.median(step_deg)), 2),
            "near_duplicate_share": round(dup / len(kept_gt), 4),
        }

    # Score placed poses per place
    place_ids = sorted(places.place_ids(state_dir), key=lambda pid: -len(sm.views(state_dir, pid)))
    all_placed_gt_list: list[np.ndarray] = []
    place_reports: dict[str, Any] = {}
    main_alignment: tuple[str, float, np.ndarray, np.ndarray, list[sm.View]] | None = None

    for pid in place_ids:
        p_views = [v for v in sm.views(state_dir, pid) if v.id in kept_frames and kept_frames[v.id] in by_video]
        if not p_views:
            continue
        est = np.stack([np.asarray(v.pose, dtype=np.float64).reshape(4, 4) for v in p_views])
        gt = np.stack([by_video[kept_frames[v.id]].c2w for v in p_views])
        all_placed_gt_list.append(gt)
        entry: dict[str, Any] = {"views": len(p_views)}
        if len(p_views) >= 3:
            s, r, t = umeyama(est[:, :3, 3], gt[:, :3, 3])
            aligned_pos = (s * (r @ est[:, :3, 3].T)).T + t
            err = np.linalg.norm(aligned_pos - gt[:, :3, 3], axis=1)
            turns = [rot_angle_deg(r @ est[i, :3, :3], gt[i, :3, :3]) for i in range(len(p_views))]
            # Consecutive RPE (both raw metric and similarity-scaled)
            rpe_m, rpe_scaled_m, rpe_deg = [], [], []
            for i in range(len(p_views) - 1):
                rel_est = np.linalg.inv(est[i]) @ est[i + 1]
                rel_gt = np.linalg.inv(gt[i]) @ gt[i + 1]
                rpe_m.append(float(np.linalg.norm(rel_est[:3, 3] - rel_gt[:3, 3])))
                rel_scaled = rel_est.copy()
                rel_scaled[:3, 3] *= s
                rpe_scaled_m.append(float(np.linalg.norm(rel_scaled[:3, 3] - rel_gt[:3, 3])))
                rpe_deg.append(rot_angle_deg(rel_est[:3, :3], rel_gt[:3, :3]))
            # Initial-segment drift (align first segment, measure at 120s and end)
            seg_n = min(len(p_views), max(4, len(p_views) // 4))
            while seg_n < len(p_views) and float(np.linalg.norm(gt[seg_n - 1, :3, 3] - gt[0, :3, 3])) < 1.5:
                seg_n += 1
            s0, r0, t0_vec = umeyama(est[:seg_n, :3, 3], gt[:seg_n, :3, 3])
            drift_pos = (s0 * (r0 @ est[:, :3, 3].T)).T + t0_vec
            drift_err = np.linalg.norm(drift_pos - gt[:, :3, 3], axis=1)
            times_v = [by_video[kept_frames[v.id]].t for v in p_views]
            at_2m = [i for i, tv in enumerate(times_v) if tv <= 120.0]
            # Loop / revisit closure error in the map's own metres and similarity-scaled metres
            revisit_errs_raw: list[float] = []
            revisit_errs_scaled: list[float] = []
            revisit_degs: list[float] = []
            for i in range(len(p_views)):
                for j in range(i + 1, len(p_views)):
                    if times_v[j] - times_v[i] < REVISIT_GAP_S:
                        continue
                    gt_d = float(np.linalg.norm(gt[j, :3, 3] - gt[i, :3, 3]))
                    if gt_d <= REVISIT_M:
                        rel_est = np.linalg.inv(est[i]) @ est[j]
                        rel_gt = np.linalg.inv(gt[i]) @ gt[j]
                        revisit_errs_raw.append(float(np.linalg.norm(rel_est[:3, 3] - rel_gt[:3, 3])))
                        revisit_errs_scaled.append(float(np.linalg.norm(s * rel_est[:3, 3] - rel_gt[:3, 3])))
                        revisit_degs.append(rot_angle_deg(rel_est[:3, :3], rel_gt[:3, :3]))
            start_end_est_m = float(np.linalg.norm(est[-1, :3, 3] - est[0, :3, 3]))
            start_end_gt_place_m = float(np.linalg.norm(gt[-1, :3, 3] - gt[0, :3, 3]))
            entry.update(
                {
                    "scale": round(s, 4),
                    "centre_rmse_m": round(float(np.sqrt((err**2).mean())), 3),
                    "centre_median_m": round(float(np.median(err)), 3),
                    "centre_max_m": round(float(err.max()), 3),
                    "rotation_median_deg": round(float(np.median(turns)), 2),
                    "rotation_max_deg": round(float(np.max(turns)), 2),
                    "rpe_raw_m_median": round(float(np.median(rpe_m)), 3),
                    "rpe_scaled_m_median": round(float(np.median(rpe_scaled_m)), 3),
                    "rpe_scaled_m_p90": round(float(np.percentile(rpe_scaled_m, 90)), 3),
                    "rpe_scaled_m_max": round(float(np.max(rpe_scaled_m)), 3),
                    "rpe_deg_median": round(float(np.median(rpe_deg)), 2),
                    "rpe_deg_p90": round(float(np.percentile(rpe_deg, 90)), 2),
                    "rpe_deg_max": round(float(np.max(rpe_deg)), 2),
                    "drift_at_120s_m": round(float(drift_err[at_2m[-1]]), 3) if at_2m else None,
                    "drift_at_end_m": round(float(drift_err[-1]), 3),
                    "start_end_gt_m": round(start_end_gt_place_m, 3),
                    "start_end_map_m": round(start_end_est_m, 3),
                    "start_end_scaled_m": round(s * start_end_est_m, 3),
                    "revisit_pairs": len(revisit_errs_raw),
                    "revisit_err_raw_m_median": round(float(np.median(revisit_errs_raw)), 3)
                    if revisit_errs_raw
                    else None,
                    "revisit_err_scaled_m_median": round(float(np.median(revisit_errs_scaled)), 3)
                    if revisit_errs_scaled
                    else None,
                    "revisit_deg_median": round(float(np.median(revisit_degs)), 2) if revisit_degs else None,
                }
            )
            if main_alignment is None:
                # Inverse similarity (gt -> est) for scoring held-out views in the place's frame
                s_inv, r_inv, t_inv = umeyama(gt[:, :3, 3], est[:, :3, 3])
                main_alignment = (pid, s_inv, r_inv, t_inv, p_views)
        place_reports[pid] = entry

    placed_gt = np.concatenate(all_placed_gt_list, axis=0) if all_placed_gt_list else np.zeros((0, 4, 4))

    # Diagnose chain breaks along kept_final in time order
    main_place = place_ids[0] if place_ids else None
    main_poses = map_state.placed(main_place) if main_place is not None else {}
    breaks: list[dict[str, Any]] = []
    main_scale = place_reports.get(main_place or "", {}).get("scale") or 1.0
    for i, f in enumerate(kept_final):
        info = map_state.frames.get(f.id) or {}
        v_idx = kept_frames[f.id]
        gt_curr = by_video[v_idx]
        prev_gt = by_video[kept_frames[kept_final[i - 1].id]] if i > 0 else None
        gt_move = float(np.linalg.norm(gt_curr.c2w[:3, 3] - prev_gt.c2w[:3, 3])) if prev_gt else 0.0
        gt_turn = rot_angle_deg(prev_gt.c2w[:3, :3], gt_curr.c2w[:3, :3]) if prev_gt else 0.0
        dt = float(gt_curr.t - prev_gt.t) if prev_gt else 0.0
        if f.id not in main_poses:
            status = f"other_place:{info.get('place')}" if info.get("pose") is not None else f"parked:{info.get('why')}"
            breaks.append(
                {
                    "t": round(gt_curr.t, 2),
                    "dt_s": round(dt, 2),
                    "gt_move_m": round(gt_move, 2),
                    "gt_turn_deg": round(gt_turn, 1),
                    "sharpness": round(f.sharpness, 1),
                    "status": status,
                }
            )
        elif i > 0 and kept_final[i - 1].id in main_poses and prev_gt is not None:
            p_prev = main_poses[kept_final[i - 1].id]
            p_curr = main_poses[f.id]
            rel_est = np.linalg.inv(p_prev) @ p_curr
            rel_gt = np.linalg.inv(prev_gt.c2w) @ gt_curr.c2w
            err_m = float(np.linalg.norm(main_scale * rel_est[:3, 3] - rel_gt[:3, 3]))
            err_deg = rot_angle_deg(rel_est[:3, :3], rel_gt[:3, :3])
            if err_m > BREAK_M or err_deg > BREAK_DEG:
                breaks.append(
                    {
                        "t": round(gt_curr.t, 2),
                        "dt_s": round(dt, 2),
                        "gt_move_m": round(gt_move, 2),
                        "gt_turn_deg": round(gt_turn, 1),
                        "sharpness": round(f.sharpness, 1),
                        "status": "pose_jump",
                        "step_err_m": round(err_m, 2),
                        "step_err_deg": round(err_deg, 1),
                    }
                )

    # Held-out splat progression on the main place
    holdout_scores: list[dict[str, Any]] = []
    if score_splat and main_alignment is not None:
        pid, s_inv, r_inv, t_inv, p_views = main_alignment
        revs = [r for r in places.revisions(state_dir, pid) if r.id in splat_snaps]
        if revs:
            picked_revs = revs if len(revs) <= 3 else [revs[0], revs[len(revs) // 2], revs[-1]]
            holdout_scores = _score_holdout_revisions(
                video=video,
                by_video=by_video,
                kept_video_frames=set(kept_frames.values()),
                raw_size=raw_size,
                rotate_deg=rotate_deg,
                max_side=max_side,
                t_range=(by_video[kept_frames[p_views[0].id]].t, by_video[kept_frames[p_views[-1].id]].t),
                transform=(s_inv, r_inv, t_inv),
                revisions=[(r.id, splat_snaps[r.id]) for r in picked_revs],
            )

    outcomes: dict[str, int] = {}
    for rec in samples:
        outcomes[rec.outcome] = outcomes.get(rec.outcome, 0) + 1

    return {
        "duration_s": round(duration_s, 2),
        "path_m": round(path_m, 2),
        "start_end_gt_m": round(start_end_gt_m, 3),
        "video_shift": shift,
        "sampled": len(samples),
        "outcomes": outcomes,
        "kept_ever": len(kept_frames),
        "kept_final": len(store.kept),
        "kept_per_min": round(len(store.kept) / max(duration_s / 60.0, 1e-6), 2),
        "coverage_kept": coverage(kept_gt),
        "coverage_placed": coverage(placed_gt),
        "redundancy": redundancy,
        "places_count": len(place_reports),
        "parked_final": len(sm.waiting(store.kept, map_state)),
        "places": place_reports,
        "chain_breaks_count": len(breaks),
        "chain_breaks": breaks[:25],
        "holdout_progression": holdout_scores,
    }


def _score_holdout_revisions(
    *,
    video: Path,
    by_video: Mapping[int, OdometryRow],
    kept_video_frames: set[int],
    raw_size: tuple[int, int],
    rotate_deg: int,
    max_side: int | None,
    t_range: tuple[float, float],
    transform: tuple[float, np.ndarray, np.ndarray],
    revisions: Sequence[tuple[str, Path]],
) -> list[dict[str, Any]]:
    """Extract evenly spaced held-out frames from `video` and score each revision's splat with `splat_holdout.py`."""
    candidates = [
        idx
        for idx, row in sorted(by_video.items())
        if t_range[0] <= row.t <= t_range[1] and all(abs(idx - k) >= 15 for k in kept_video_frames)
    ]
    if not candidates:
        return []
    picks = [
        candidates[round(i * (len(candidates) - 1) / max(1, min(HOLDOUT_COUNT, len(candidates)) - 1))]
        for i in range(min(HOLDOUT_COUNT, len(candidates)))
    ]
    wanted = set(picks)
    runtime = cfg.path("spatial.splat_runtime")
    python = runtime / ".venv" / "bin" / "python"
    if not python.exists():
        return []
    gpu = cameras.pick_gpu(int(cfg.get("spatial.gpu")), nvml.gpus())
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    s_inv, r_inv, t_inv = transform
    tool = Path(__file__).with_name("splat_holdout.py")

    with tempfile.TemporaryDirectory(prefix="replay-holdout-") as tmp:
        folder = Path(tmp)
        cap = cv2.VideoCapture(str(video))
        views: list[dict[str, Any]] = []
        idx = 0
        try:
            while len(views) < len(wanted):
                if idx not in wanted:
                    if not cap.grab():
                        break
                    idx += 1
                    continue
                ok, frame = cap.read()
                if not ok:
                    break
                if idx in wanted:
                    img = rotate_frame(frame, rotate_deg)
                    if max_side is not None and max(img.shape[:2]) > max_side:
                        h, w = img.shape[:2]
                        sc = max_side / max(w, h)
                        img = cv2.resize(
                            img, (max(2, round(w * sc)), max(2, round(h * sc))), interpolation=cv2.INTER_AREA
                        )
                    if cameras.daylight(img):
                        tw, th = sm.train_size(img.shape[1], img.shape[0])
                        img_t = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
                        img_path = folder / f"h{idx:06d}.png"
                        cv2.imwrite(str(img_path), img_t)
                        row = by_video[idx]
                        k_t = turned_intrinsics(row.k, raw_size, (tw, th), rotate_deg)
                        pose = np.eye(4, dtype=np.float64)
                        pose[:3, :3] = r_inv @ row.c2w[:3, :3]
                        pose[:3, 3] = s_inv * (r_inv @ row.c2w[:3, 3]) + t_inv
                        views.append(
                            {
                                "image": str(img_path),
                                "pose": pose.ravel().tolist(),
                                "focal": float((k_t[0, 0] + k_t[1, 1]) / 2.0),
                                "fx": float(k_t[0, 0]),
                                "fy": float(k_t[1, 1]),
                                "cx": float(k_t[0, 2]),
                                "cy": float(k_t[1, 2]),
                                "width": tw,
                                "height": th,
                            }
                        )
                idx += 1
        finally:
            cap.release()
        if not views:
            return []
        job = folder / "job.json"
        job.write_text(json.dumps({"views": views, "holdout": list(range(len(views)))}))
        out: list[dict[str, Any]] = []
        for round_i, (rev_id, splat_path) in enumerate(revisions):
            cmd = [
                "nice",
                "-n",
                str(sm.NICE),
                str(python),
                str(tool),
                "score",
                str(job),
                str(splat_path),
                "--side",
                str(HOLDOUT_SIDE),
            ]
            done = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
            if done.returncode == 0:
                scored = json.loads(done.stdout.strip().splitlines()[-1])
                out.append(
                    {
                        "index": round_i,
                        "revision": rev_id,
                        "gaussians": scored["gaussians"],
                        "held_out": scored["held_out"],
                        "psnr_as_posed": scored["psnr_as_posed"],
                        "psnr_aligned": scored["psnr_aligned"],
                    }
                )
        return out


def replay(
    video: Path,
    *,
    state_dir: Path,
    camera: str = "walk",
    every_s: float | None = None,
    rotate_deg: int = 0,
    max_side: int | None = 960,
    limit_s: float | None = None,
    map_gap_s: float = worker.MAP_GAP_S,
    gate_only: bool = False,
    odometry_path: Path | None = None,
    score_splat: bool = True,
    max_steps: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Run `video` through the real sampler and map step into `state_dir`, and optionally score against `odometry_path`."""
    import collections as _collections

    live_state = cfg.path("paths.state").resolve()
    if state_dir.resolve() == live_state:
        raise ValueError(f"refusing to replay into the live state directory {live_state}")
    state_dir.mkdir(parents=True, exist_ok=True)
    scratch_meta = state_dir / "_replay_meta"
    scratch_meta.mkdir(parents=True, exist_ok=True)
    manifest_path = scratch_meta / "devices.toml"
    manifest_path.write_text(f'[[device]]\nname = "{camera}"\nkind = "camera"\n')
    pack_toml = scratch_meta / "pack.toml"
    pack_toml.write_text("")
    splat_snap_dir = scratch_meta / "splats"
    splat_snap_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = scratch_meta / "checkpoint.json"

    ckpt: dict[str, Any] = json.loads(ckpt_path.read_text()) if resume and ckpt_path.is_file() else {}
    splat_snaps: dict[str, Path] = {
        str(k): splat_snap_dir / str(v)
        for k, v in (ckpt.get("splat_snaps") or {}).items()
        if (splat_snap_dir / str(v)).is_file()
    }

    sample_every = worker.SAMPLE_EVERY_S if every_s is None else every_s
    old_manifest = lens.manifest
    old_pack_config = os.environ.get("PACK_CONFIG")
    os.environ["PACK_CONFIG"] = str(pack_toml)
    lens.manifest = lambda: manifest_path  # ty: ignore[invalid-assignment] (scratch manifest for isolated replay)
    try:
        with cfg.override({"paths": {"state": str(state_dir)}}):
            store = frames.Store(frames.root(state_dir))
            describer = frames.Describer()
            detect = worker.Living()
            sampler = worker.Sampler(store, describer, detect, cameras.daylight, cameras.sharpness)
            if ckpt.get("sharp_history"):
                sampler._sharp[camera] = _collections.deque(
                    (float(x) for x in ckpt["sharp_history"]), maxlen=worker.SHARP_WINDOW
                )
            if ckpt.get("judged_thumb") is not None and ckpt.get("judged_id"):
                sampler._judged[camera] = (
                    np.asarray(ckpt["judged_thumb"], dtype=np.float32),
                    str(ckpt["judged_id"]),
                )

            samples: list[SampleRecord] = [
                SampleRecord(
                    video_frame=int(r["video_frame"]),
                    t=float(r["t"]),
                    outcome=str(r["outcome"]),
                    kept_id=r["kept_id"],
                    sharpness=float(r["sharpness"]),
                    brightness=float(r["brightness"]),
                )
                for r in (ckpt.get("samples") or [])
            ]
            kept_frames: dict[str, int] = {str(k): int(v) for k, v in (ckpt.get("kept_frames") or {}).items()}
            step_logs: list[dict[str, Any]] = list(ckpt.get("step_logs") or [])
            base_epoch = 1_700_000_000.0
            last_map_t: float | None = None if ckpt.get("last_map_t") is None else float(ckpt["last_map_t"])
            skip_through = int(ckpt.get("last_video_frame", -1))
            stream_done = bool(ckpt.get("stream_done", False))
            drain_offset = float(ckpt.get("drain_offset", 1.0))

            raw_size_saved = ckpt.get("raw_size")
            timed_saved = ckpt.get("timed")
            raw_size: tuple[int, int]
            timed: list[tuple[int, float]]
            steps_this_call = 0
            paused = False

            def save_ckpt() -> None:
                judged = sampler._judged.get(camera)
                ckpt_path.write_text(
                    json.dumps(
                        {
                            "raw_size": list(raw_size),
                            "timed": [list(x) for x in timed],
                            "last_video_frame": skip_through,
                            "stream_done": stream_done,
                            "last_map_t": last_map_t,
                            "drain_offset": drain_offset,
                            "sharp_history": list(sampler._sharp.get(camera, ())),
                            "judged_thumb": None if judged is None else judged[0].tolist(),
                            "judged_id": None if judged is None else judged[1],
                            "kept_frames": kept_frames,
                            "step_logs": step_logs,
                            "splat_snaps": {k: p.name for k, p in splat_snaps.items()},
                            "samples": [
                                {
                                    "video_frame": s.video_frame,
                                    "t": s.t,
                                    "outcome": s.outcome,
                                    "kept_id": s.kept_id,
                                    "sharpness": s.sharpness,
                                    "brightness": s.brightness,
                                }
                                for s in samples
                            ],
                        }
                    )
                )

            if not stream_done:
                raw_size, timed_live, stream = iter_video(
                    video, sample_every, rotate_deg, max_side, limit_s, skip_through=skip_through
                )
                timed = [(int(a), float(b)) for a, b in timed_saved] if timed_saved else timed_live
                for v_idx, rel_t, img in stream:
                    now = base_epoch + rel_t
                    before_ids = {f.id for f in store.kept}
                    outcome = sampler.look(camera, img, now)
                    after_ids = {f.id for f in store.kept}
                    added_ids = after_ids - before_ids
                    kept_id = next(iter(added_ids), None)
                    if kept_id is not None:
                        kept_frames[kept_id] = v_idx
                    samples.append(
                        SampleRecord(
                            video_frame=v_idx,
                            t=rel_t,
                            outcome=outcome,
                            kept_id=kept_id,
                            sharpness=cameras.sharpness(img),
                            brightness=cameras.light(img)[0],
                        )
                    )
                    skip_through = v_idx
                    if not gate_only and (last_map_t is None or rel_t - last_map_t >= map_gap_s):
                        _apply_drops(store)
                        due_now = sm.due(store.kept, sm.State.read(store.folder), now, {})
                        if due_now:
                            if max_steps is not None and steps_this_call >= max_steps:
                                paused = True
                                save_ckpt()
                                break
                            did = sm.step(state_dir, now=now)
                            steps_this_call += 1
                            if did is not None:
                                surroundings_positions.advance(state_dir, now=now, manifest=manifest_path)
                                _snapshot_splats(state_dir, splat_snap_dir, splat_snaps)
                                step_logs.append(_slim_step(did))
                            _apply_drops(store)
                            last_map_t = rel_t
                            save_ckpt()
                else:
                    stream_done = True
                    save_ckpt()
            else:
                assert raw_size_saved is not None and timed_saved is not None
                raw_size = (int(raw_size_saved[0]), int(raw_size_saved[1]))
                timed = [(int(a), float(b)) for a, b in timed_saved]

            # Drain remaining due batches, just as the background service does once walking stops.
            if not gate_only and not paused:
                end_now = base_epoch + (samples[-1].t if samples else 0.0) + drain_offset
                while True:
                    _apply_drops(store)
                    due_now = sm.due(store.kept, sm.State.read(store.folder), end_now, {})
                    if not due_now:
                        break
                    if max_steps is not None and steps_this_call >= max_steps:
                        paused = True
                        save_ckpt()
                        break
                    did = sm.step(state_dir, now=end_now)
                    steps_this_call += 1
                    end_now += 1.0
                    drain_offset += 1.0
                    if did is None:
                        break
                    surroundings_positions.advance(state_dir, now=end_now, manifest=manifest_path)
                    _snapshot_splats(state_dir, splat_snap_dir, splat_snaps)
                    step_logs.append(_slim_step(did))
                    save_ckpt()
                if not paused:
                    _apply_drops(store)
                    save_ckpt()

            report: dict[str, Any] = {
                "complete": not paused,
                "sample_every_s": sample_every,
                "rotate_deg": rotate_deg,
                "sampled": len(samples),
                "kept_ever": len(kept_frames),
                "kept_final": len(store.kept),
                "steps_run": len(step_logs),
                "steps": step_logs,
            }
            if odometry_path is not None and not paused:
                report["score"] = score_run(
                    video=video,
                    odometry_path=odometry_path,
                    raw_size=raw_size,
                    timed=timed,
                    rotate_deg=rotate_deg,
                    max_side=max_side,
                    samples=samples,
                    kept_frames=kept_frames,
                    state_dir=state_dir,
                    splat_snaps=splat_snaps,
                    limit_s=limit_s,
                    score_splat=score_splat and not gate_only,
                )
            return report
    finally:
        lens.manifest = old_manifest
        if old_pack_config is None:
            os.environ.pop("PACK_CONFIG", None)
        else:
            os.environ["PACK_CONFIG"] = old_pack_config


def _slim_step(did: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "taken": did.get("taken"),
        "unplaced": did.get("unplaced"),
        "seconds": did.get("seconds"),
        "places": {
            pid: {
                k: p.get(k)
                for k in (
                    "founded",
                    "views",
                    "added",
                    "already_drawn",
                    "gaussians",
                    "train_psnr",
                    "pose_shift",
                    "points",
                )
            }
            for pid, p in (did.get("places") or {}).items()
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path, help="recorded video file to feed through the sampler and map step")
    parser.add_argument("--state", type=Path, default=None, help="scratch state directory to keep (default: temporary)")
    parser.add_argument("--camera", default="walk", help="camera name in the scratch manifest")
    parser.add_argument("--every", type=float, default=None, help="seconds between sampled frames (default: sampler's)")
    parser.add_argument(
        "--rotate", type=int, choices=(0, 90, 180, 270), default=0, help="clockwise turn for raw frames"
    )
    parser.add_argument("--max-side", type=int, default=960, help="downscale frames above this long side (0 for raw)")
    parser.add_argument("--limit-s", type=float, default=None, help="stop after this many seconds of video")
    parser.add_argument(
        "--map-gap-s", type=float, default=worker.MAP_GAP_S, help="video seconds between live map steps"
    )
    parser.add_argument(
        "--gate-only", action="store_true", help="run only the sampler and first gate, no GPU map steps"
    )
    parser.add_argument(
        "--odometry", type=Path, default=None, help="ARKit odometry.csv for post-run ground-truth scoring"
    )
    parser.add_argument("--no-splat-score", action="store_true", help="skip held-out splat rendering when scoring")
    parser.add_argument("--max-steps", type=int, default=None, help="pause after at most this many map steps")
    parser.add_argument("--resume", action="store_true", help="resume a paused replay in --state")
    args = parser.parse_args(argv)
    logs.setup()

    with contextlib.ExitStack() as stack:
        state_dir = args.state
        if state_dir is None:
            state_dir = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="replay-surroundings-")))
        result = replay(
            args.video,
            state_dir=state_dir,
            camera=args.camera,
            every_s=args.every,
            rotate_deg=args.rotate,
            max_side=None if args.max_side <= 0 else args.max_side,
            limit_s=args.limit_s,
            map_gap_s=args.map_gap_s,
            gate_only=args.gate_only,
            odometry_path=args.odometry,
            score_splat=not args.no_splat_score,
            max_steps=args.max_steps,
            resume=args.resume,
        )
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
