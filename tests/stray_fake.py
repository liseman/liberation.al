"""A tiny Stray Scanner export for tests: a box room, a camera walking through it, and the LiDAR depth that walk sees,
written in the app's format (spatial/stray.py)."""

import itertools
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOM = ((0.0, 4.0), (0.0, 2.6), (0.0, 3.0))  # x, y (up), z extents, metres: ARKit's world is y-up
RGB = (320, 240)  # rgb.mp4's width and height (the sensor's landscape)
DEPTH = (64, 48)
K = np.array([[250.0, 0.0, 159.5], [0.0, 250.0, 119.5], [0.0, 0.0, 1.0]])
CLOCKWISE = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # stray.TURNS[1]: turned to sensor


def upright_pose(x: float, heading: float = 0.0) -> np.ndarray:
    """An upright camera (OpenCV axes) 1.5 m up at (x, 1.5), looking along +z turned `heading` about up."""
    c, s = math.cos(heading), math.sin(heading)
    forward = np.array([s, 0.0, c])
    down = np.array([0.0, -1.0, 0.0])
    pose = np.eye(4)
    pose[:3, :3] = np.stack([np.cross(down, forward), down, forward], axis=1)
    pose[:3, 3] = [x, 1.5, 1.5]
    return pose


def depth_map(c2w: np.ndarray) -> np.ndarray:
    """Metres to the room's walls, floor, or ceiling for each depth pixel of the sensor camera at `c2w`."""
    w, h = DEPTH
    sx, sy = w / RGB[0], h / RGB[1]
    v, u = np.mgrid[0:h, 0:w]
    ur, vr = (u + 0.5) / sx - 0.5, (v + 0.5) / sy - 0.5
    rays = np.stack([(ur - K[0, 2]) / K[0, 0], (vr - K[1, 2]) / K[1, 1], np.ones_like(ur, dtype=float)], -1)
    d = rays @ c2w[:3, :3].T
    o = c2w[:3, 3]
    best = np.full((h, w), np.inf)
    for axis, (lo, hi) in enumerate(ROOM):
        for plane in (lo, hi):
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (plane - o[axis]) / d[..., axis]
            best = np.where((t > 0) & (t < best), t, best)
    return best  # a ray's t is its depth, since the ray's camera z is 1


def write(folder: Path, sensor_poses: list[np.ndarray], intrinsics_columns: bool = True) -> Path:
    """The export of a walk whose sensor camera has these poses (camera to world, OpenCV axes), one rgb frame each and
    four more between (so the walk's frame picker has frames to choose from, at 30 fps)."""
    (folder / "depth").mkdir(parents=True)
    (folder / "confidence").mkdir()
    rng = np.random.default_rng(1)
    scene = cv2.GaussianBlur(rng.uniform(0, 255, (RGB[1], 4000, 3)).astype(np.uint8), (0, 0), 2)
    writer = cv2.VideoWriter(str(folder / "rgb.mp4"), cv2.VideoWriter.fourcc(*"mp4v"), 30, RGB)
    header = "timestamp, frame, x, y, z, qx, qy, qz, qw"
    lines = [header + (", fx, fy, cx, cy, distortion_center_x, distortion_center_y" if intrinsics_columns else "")]
    frame = 0
    for a, b in itertools.pairwise(sensor_poses):
        for k in range(5):
            pose = a.copy()
            pose[:3, 3] = a[:3, 3] + (b[:3, 3] - a[:3, 3]) * k / 5
            writer.write(np.ascontiguousarray(scene[:, frame * 6 : frame * 6 + RGB[0]]))
            depth = depth_map(pose)
            conf = np.full(depth.shape, 2, np.uint8)
            conf[:, :4] = 1  # a low-confidence strip, which the import leaves out
            cv2.imwrite(str(folder / "depth" / f"{frame:06d}.png"), np.round(depth * 1000).astype(np.uint16))
            cv2.imwrite(str(folder / "confidence" / f"{frame:06d}.png"), conf)
            q = Rotation.from_matrix(pose[:3, :3]).as_quat()
            row = [frame / 30, f"{frame:06d}", *pose[:3, 3], *q]
            if intrinsics_columns:
                row += [K[0, 0], K[1, 1], K[0, 2], K[1, 2], "", ""]
            lines.append(", ".join(str(x) for x in row))
            frame += 1
    writer.release()
    (folder / "odometry.csv").write_text("\n".join(lines) + "\n")
    np.savetxt(folder / "camera_matrix.csv", K, delimiter=",")
    return folder


def distance_to_room(points: np.ndarray) -> np.ndarray:
    """Each world point's distance to the nearest of the room's six faces."""
    return np.min(np.abs(np.stack([points[:, a] - v for a, ext in enumerate(ROOM) for v in ext], -1)), axis=1)
