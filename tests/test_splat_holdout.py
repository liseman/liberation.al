"""tools/splat_holdout.py: the pose comparison, the square crop's intrinsics, and reading a 3DGS PLY (the GPU scoring
needs gsplat's runtime and isn't tested here)."""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "splat_holdout.py"
spec = importlib.util.spec_from_file_location("splat_holdout", SCRIPT)
assert spec and spec.loader
sh = importlib.util.module_from_spec(spec)
sys.modules["splat_holdout"] = sh
spec.loader.exec_module(sh)


def rot_z(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def walk() -> np.ndarray:
    """Four cameras on an L-shaped path, each turned a little differently."""
    out = np.tile(np.eye(4), (4, 1, 1))
    for i, (centre, yaw) in enumerate([((0, 0, 0), 0), ((1, 0, 0), 10), ((2, 0, 0), 20), ((2, 1, 0.5), 30)]):
        out[i, :3, :3] = rot_z(yaw)
        out[i, :3, 3] = centre
    return out


def moved(c2w: np.ndarray, scale: float, turn: np.ndarray, shift: np.ndarray) -> np.ndarray:
    out = c2w.copy()
    out[:, :3, :3] = turn @ c2w[:, :3, :3]
    out[:, :3, 3] = scale * (turn @ c2w[:, :3, 3].T).T + shift
    return out


def test_poses_that_differ_by_a_similarity_agree_exactly() -> None:
    a = walk()
    b = moved(a, 2.0, rot_z(90), np.array([5.0, -1.0, 3.0]))
    got = sh.compare_poses(a, b)
    assert got["scale"] == pytest.approx(2.0)
    assert got["centre_rmse"] == pytest.approx(0.0, abs=1e-6)
    assert got["rotation_max_deg"] == pytest.approx(0.0, abs=1e-3)
    assert got["path_length"] == pytest.approx(2 * (1 + 1 + math.sqrt(1.25)), abs=1e-3)


def test_one_camera_turned_shows_in_the_rotation_error_only() -> None:
    a = walk()
    b = a.copy()
    b[3, :3, :3] = rot_z(12) @ b[3, :3, :3]
    got = sh.compare_poses(a, b)
    assert got["centre_max"] == pytest.approx(0.0, abs=1e-6)
    assert got["rotation_max_deg"] == pytest.approx(12.0, abs=1e-3)
    assert got["rotation_median_deg"] == pytest.approx(0.0, abs=1e-3)


def test_square_crop_scales_the_short_side_and_moves_the_principal_point() -> None:
    k = np.array([[600.0, 0, 384], [0, 600.0, 512], [0, 0, 1]])
    # 768 x 1024 portrait to 448 a side: scale 448/768, 597 px tall, 74 px cut off the top
    got, scale, left, top = sh.square_crop(k, 768, 1024, 448)
    assert (scale, left, top) == (pytest.approx(448 / 768), 0, 74)
    assert got[0, 0] == pytest.approx(350.0)
    assert got[0, 2] == pytest.approx(224.0)
    assert got[1, 2] == pytest.approx(512 * 448 / 768 - 74)


def test_read_ply_takes_sh_channel_by_channel(tmp_path: Path) -> None:
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(9)] + ["opacity", "scale_0", "scale_1", "scale_2"]
    names += ["rot_0", "rot_1", "rot_2", "rot_3"]
    # f_rest: red's three band-1 coefficients (1, 2, 3), then green's (4, 5, 6), then blue's (7, 8, 9)
    row = [1, 2, 3, 0, 0, 0, 0.1, 0.2, 0.3, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0.0, 0.0, math.log(2), 0.0, 2, 0, 0, 0]
    header = "ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
    header += "".join(f"property float {n}\n" for n in names) + "end_header\n"
    path = tmp_path / "one.ply"
    path.write_bytes(header.encode() + np.array(row, "<f4").tobytes())
    got = sh.read_ply(path)
    assert got["sh"].shape == (1, 4, 3)
    np.testing.assert_allclose(got["sh"][0, 0], [0.1, 0.2, 0.3], rtol=1e-6)
    np.testing.assert_allclose(got["sh"][0, 1], [1, 4, 7])
    np.testing.assert_allclose(got["sh"][0, 3], [3, 6, 9])
    np.testing.assert_allclose(got["quats"][0], [1, 0, 0, 0])
    np.testing.assert_allclose(got["scales"][0], [1, 2, 1], rtol=1e-6)
    assert got["opacities"][0] == pytest.approx(0.5)
    np.testing.assert_allclose(got["means"][0], [1, 2, 3])
