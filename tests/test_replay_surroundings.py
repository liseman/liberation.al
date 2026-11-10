"""Tests for the moving-camera surroundings replay and scoring harness (`tools/replay_surroundings.py`, #281)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def _load_replay():
    spec = importlib.util.spec_from_file_location("replay_surroundings", ROOT / "tools" / "replay_surroundings.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["replay_surroundings"] = module
    spec.loader.exec_module(module)
    return module


def test_umeyama_recovers_known_similarity():
    rs = _load_replay()
    rng = np.random.default_rng(7)
    src = rng.normal(size=(12, 3))
    true_r = Rotation.from_euler("xyz", [18.0, -32.0, 41.0], degrees=True).as_matrix()
    true_s = 1.35
    true_t = np.array([2.4, -1.1, 0.7])
    dst = (true_s * (true_r @ src.T)).T + true_t

    s, r, t = rs.umeyama(src, dst)
    assert s == pytest.approx(true_s, abs=1e-6)
    assert np.allclose(r, true_r, atol=1e-6)
    assert np.allclose(t, true_t, atol=1e-6)
    assert np.linalg.det(r) == pytest.approx(1.0, abs=1e-6)


def test_turned_intrinsics_projects_same_pixel_after_clockwise_turn():
    rs = _load_replay()
    w, h = 640, 480
    k = np.array([[520.0, 0.0, 318.0], [0.0, 515.0, 241.0], [0.0, 0.0, 1.0]])
    img = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)

    # Pick an arbitrary 3D point in the turned camera frame and verify pixel lookup agrees with rotate_frame.
    for deg, target in [(0, (640, 480)), (90, (480, 640)), (180, (640, 480)), (270, (480, 640))]:
        k_t = rs.turned_intrinsics(k, (w, h), target, deg)
        turn = rs.TURNS[deg][0]
        p_turned = np.array([0.15, -0.1, 2.0])
        p_raw = turn @ p_turned
        uv_raw = k @ (p_raw / p_raw[2])
        uv_turned = k_t @ (p_turned / p_turned[2])
        u0, v0 = round(float(uv_raw[0])), round(float(uv_raw[1]))
        ut, vt = round(float(uv_turned[0])), round(float(uv_turned[1]))
        turned_img = rs.rotate_frame(img, deg)
        assert np.array_equal(turned_img[vt, ut], img[v0, u0])


def test_video_shift_detects_dropped_initial_frame():
    rs = _load_replay()
    eye = np.eye(4)
    k = np.eye(3)
    # Irregular frame deltas where video index 0 corresponds to odometry row 1.
    ts = [0.0, 0.031, 0.072, 0.105, 0.155, 0.189]
    rows = {i: rs.OdometryRow(t=t, c2w=eye, k=k, frame=i) for i, t in enumerate(ts)}
    timed = [(i, ts[i + 1] - ts[1]) for i in range(len(ts) - 1)]
    assert rs.video_shift(rows, timed) == 1


def test_gate_only_replay_scores_coverage_into_scratch_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    rs = _load_replay()

    class FakeDescriber:
        def __call__(self, bgr: np.ndarray) -> np.ndarray:
            # Distinct unit descriptors derived from mean spatial quadrants so distinct frames stay distinct.
            g = bgr.mean(axis=2, dtype=np.float32)
            h, w = g.shape
            vec = np.zeros(64, dtype=np.float32)
            vec[0] = float(g[: h // 2, : w // 2].mean())
            vec[1] = float(g[: h // 2, w // 2 :].mean())
            vec[2] = float(g[h // 2 :, : w // 2].mean())
            vec[3] = float(g[h // 2 :, w // 2 :].mean()) + 1.0
            # Encode stripe phase so stepped frames differ below SAME_VIEW.
            idx = int(round(float(bgr[0, 0, 0]) / 40.0)) % 60
            vec[4 + idx] = 200.0
            vec /= np.linalg.norm(vec)
            return vec

    class FakeLiving:
        def __call__(self, bgr: np.ndarray) -> list[tuple[float, float, float, float]]:
            return []

    monkeypatch.setattr(rs.frames, "Describer", FakeDescriber)
    monkeypatch.setattr(rs.worker, "Living", FakeLiving)

    video_path = tmp_path / "walk.mp4"
    odom_path = tmp_path / "odometry.csv"
    state_dir = tmp_path / "scratch_state"
    w, h, fps, n_frames = 320, 240, 5.0, 35
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert writer.isOpened()

    lines = ["timestamp,frame,x,y,z,qx,qy,qz,qw,fx,fy,cx,cy"]
    for i in range(n_frames):
        img = np.full((h, w, 3), 140, dtype=np.uint8)
        img[0, 0] = (i * 40) % 240
        # Crisp synthetic vertical and horizontal bars so sharpness passes.
        x0 = 30 + (i * 5) % 200
        cv2.rectangle(img, (x0, 30), (x0 + 40, 200), (30, 30, 30), -1)
        cv2.rectangle(img, (20, 90), (290, 120), (220, 220, 220), -1)
        writer.write(img)
        t = i / fps
        lines.append(f"{100.0 + t:.6f},{i},{i * 0.12:.4f},0.0,0.0,0.0,0.0,0.0,1.0,260.0,260.0,160.0,120.0")
    writer.release()
    odom_path.write_text("\n".join(lines) + "\n")

    report = rs.replay(
        video_path,
        state_dir=state_dir,
        every_s=2.0,
        gate_only=True,
        odometry_path=odom_path,
    )
    assert report["complete"] is True
    assert report["kept_final"] >= 3
    assert report["score"]["coverage_kept"]["covered_share"] > 0.8
    assert (state_dir / "surroundings" / "index.json").is_file()
