"""A walkthrough recorded with Stray Scanner (iOS, MIT, github.com/strayrobots/scanner), read for pack-walkthrough:
ARKit's camera poses and the phone's LiDAR depth, so the walk needs no pose model and its cloud no learned
depth.

The export folder (the app's format, docs/format.md upstream): `rgb.mp4` (the camera's frames), `odometry.csv` (a
header, then per frame: `timestamp, frame, x, y, z, qx, qy, qz, qw`, and in newer exports `fx, fy, cx, cy` and the
distortion centre), `camera_matrix.csv` (3 x 3 intrinsics, the last frame's: what older exports have instead of the
per-frame ones), `depth/000000.png` (16-bit millimetres, 256 x 192), and `confidence/000000.png` (0, 1, or 2). The
intrinsics are for `rgb.mp4`'s size; a depth map is scaled from them.

Poses: ARKit's world is metric and gravity-aligned, +y up. The app writes each camera's rotation turned 180 degrees
about its x axis (`OdometryEncoder.swift`), so a pose is camera to world with OpenCV camera axes (x right, y down, z
forward), as the rest of spatial/ uses. ARKit records in the sensor's landscape orientation, so a phone held upright
(portrait) has its camera's -y pointing sideways: `upright` finds the quarter turn about the view axis that brings
-y up, and frames, poses, and points are all turned by it, so the walkthrough's checks (a camera's up, a pair's
tilt) and MASt3R (which wants upright images) see an upright camera.

Pure reading: numpy and OpenCV, no model, so it runs anywhere.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

WORLD_UP = (0.0, 1.0, 0.0)  # ARKit's
#: Depth pixels kept: the app's highest confidence (2), within this range of the camera (metres). LiDAR reaches about
#: 5 m; nearer than 0.2 m is the phone's own hand or a doorframe brushed past.
MIN_CONFIDENCE, NEAR_M, FAR_M = 2, 0.2, 5.0
#: The quarter turns about the view axis, as (the matrix taking a turned camera's coordinates to the sensor's, the
#: cv2.rotate code that turns a sensor image the same way, or None). A clockwise turn makes the sensor's -y (the top
#: of a landscape frame) the turned camera's +x.
TURNS: tuple[tuple[tuple[tuple[float, ...], ...], int | None], ...] = (
    (((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), None),
    (((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)), 0),  # cv2.ROTATE_90_CLOCKWISE
    (((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)), 1),  # cv2.ROTATE_180
    (((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)), 2),  # cv2.ROTATE_90_COUNTERCLOCKWISE
)


@dataclass(frozen=True)
class Frame:
    """One recorded frame's pose (4 x 4, camera to world, OpenCV axes, the sensor's orientation) and intrinsics (3 x
    3, at rgb.mp4's size)."""

    t: float
    c2w: np.ndarray
    k: np.ndarray


def is_export(path: Path) -> bool:
    """Whether `path` is a Stray Scanner export folder."""
    return path.is_dir() and (path / "odometry.csv").is_file() and (path / "rgb.mp4").is_file()


def read_odometry(folder: Path) -> dict[int, Frame]:
    """Every frame of `odometry.csv` by its frame number."""
    fallback: np.ndarray | None = None
    if (folder / "camera_matrix.csv").is_file():
        fallback = np.loadtxt(folder / "camera_matrix.csv", delimiter=",").reshape(3, 3)
    out: dict[int, Frame] = {}
    with open(folder / "odometry.csv", newline="") as f:
        rows = csv.reader(f)
        header = [h.strip() for h in next(rows)]
        for raw in rows:
            if not raw:
                continue
            row: Mapping[str, str] = {h: v.strip() for h, v in zip(header, raw, strict=False)}
            c2w = np.eye(4)
            c2w[:3, :3] = Rotation.from_quat([float(row[q]) for q in ("qx", "qy", "qz", "qw")]).as_matrix()
            c2w[:3, 3] = [float(row[a]) for a in ("x", "y", "z")]
            if row.get("fx"):
                k = np.array(
                    [[float(row["fx"]), 0, float(row["cx"])], [0, float(row["fy"]), float(row["cy"])], [0, 0, 1]]
                )
            elif fallback is not None:
                k = fallback
            else:
                raise ValueError(f"{folder}: no intrinsics (no fx column and no camera_matrix.csv)")
            out[int(row["frame"])] = Frame(float(row["timestamp"]), c2w, k)
    return out


def upright(c2ws: np.ndarray) -> tuple[np.ndarray, int | None]:
    """The quarter turn (from `TURNS`) that best brings the cameras' -y to the world's up, over the walk: the matrix
    taking a turned camera's coordinates to the sensor camera's, and the cv2.rotate code for its images."""
    up = np.asarray(WORLD_UP)
    best = max(TURNS, key=lambda t: float(np.median(np.einsum("nij,jk,k->ni", c2ws[:, :3, :3], t[0], (0, -1, 0)) @ up)))
    return np.array(best[0]), best[1]


def turned(c2w: np.ndarray, turn: np.ndarray) -> np.ndarray:
    """A sensor camera's pose as the turned camera's."""
    out = np.array(c2w, dtype=np.float64)
    out[:3, :3] = c2w[:3, :3] @ turn
    return out


def saved_intrinsics(
    k: np.ndarray, size: tuple[int, int], side: int, code: int | None, turn: np.ndarray
) -> tuple[np.ndarray, tuple[int, int]]:
    """The intrinsics (3 x 3) and size (width, height) of a frame as pack-walkthrough saves it: `rgb.mp4`'s frame of
    `size` with intrinsics `k`, scaled down to at most `side` a side, then turned by the cv2.rotate `code` that goes
    with `turn` (TURNS). A turned pixel is the rotation's pixel map of the scaled sensor pixel of `turn` p, so this is
    that map times the scale times `k` times `turn`; a quarter turn swaps fx and fy and moves the principal point."""
    w, h = size
    s = side / max(w, h) if max(w, h) > side else 1.0
    sw, sh = round(w * s), round(h * s)
    scale = np.diag([s, s, 1.0])
    rot = {
        None: (np.eye(3), (sw, sh)),
        0: (np.array([[0.0, -1.0, sh - 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), (sh, sw)),  # clockwise
        1: (np.array([[-1.0, 0.0, sw - 1.0], [0.0, -1.0, sh - 1.0], [0.0, 0.0, 1.0]]), (sw, sh)),
        2: (np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, sw - 1.0], [0.0, 0.0, 1.0]]), (sh, sw)),  # anticlockwise
    }
    pixels, out_size = rot[code]
    return pixels @ scale @ np.asarray(k, dtype=np.float64) @ np.asarray(turn, dtype=np.float64), out_size


def _depth(folder: Path, frame: int) -> tuple[np.ndarray, np.ndarray] | None:
    """One frame's LiDAR depth map as recorded (16-bit millimetres, the sensor's orientation) and which of its pixels
    are kept (confident and in range); None when the frame has no depth map."""
    import cv2

    depth = cv2.imread(str(folder / "depth" / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None
    conf = cv2.imread(str(folder / "confidence" / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED)
    z = depth.astype(np.float64) / 1000.0
    keep = (z >= NEAR_M) & (z <= FAR_M)
    if conf is not None:
        keep &= conf >= MIN_CONFIDENCE
    return depth, keep


def depth_map(folder: Path, frame: int, code: int | None) -> np.ndarray | None:
    """One frame's LiDAR depth as a map to train a splat against (spatial/splats.py): 16-bit millimetres along the
    view axis, 0 where the depth isn't kept (not confident, or out of range), turned by the cv2.rotate `code` its
    saved frame was turned by (`upright`); None when the frame has no depth map. It covers the frame's whole field of
    view at the LiDAR's own size."""
    import cv2

    got = _depth(folder, frame)
    if got is None:
        return None
    depth, keep = got
    kept = np.where(keep, depth, 0).astype(np.uint16)
    return kept if code is None else cv2.rotate(kept, code)


def points(
    folder: Path, frame: int, k: np.ndarray, rgb_size: tuple[int, int], image: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """One frame's LiDAR points (N x 3, the sensor camera's frame, metres), confident and in range, and their colours
    (N x 3 uint8, RGB) from `image` (the frame from rgb.mp4, BGR, at `rgb_size` width and height) when given, else
    grey."""
    import cv2

    got = _depth(folder, frame)
    if got is None:
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
    depth, keep = got
    h, w = depth.shape[:2]
    z = depth.astype(np.float64) / 1000.0
    sx, sy = w / rgb_size[0], h / rgb_size[1]
    v, u = np.nonzero(keep)
    zz = z[v, u]
    # Pixel centres are at whole coordinates (OpenCV's convention, and the intrinsics'): a depth pixel's centre in
    # rgb.mp4's pixels is (u + 0.5) / sx - 0.5.
    ur, vr = (u + 0.5) / sx - 0.5, (v + 0.5) / sy - 0.5
    pts = np.c_[(ur - k[0, 2]) * zz / k[0, 0], (vr - k[1, 2]) * zz / k[1, 1], zz]
    if image is None:
        return pts, np.full((len(pts), 3), 160, np.uint8)
    small = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    return pts, small[v, u][:, ::-1].copy()
