"""The fisheye calibration tool (tools/calibrate_fisheye.py, #1891): it recovers a known lens from checkerboard views,
finds a board's corners, and writes the result where see-actions reads it."""

import importlib.util
import pathlib
import sys

import cv2
import numpy as np
import pytest
import torch

from liberational.see import fisheye

PATH = pathlib.Path(__file__).resolve().parents[1] / "tools" / "calibrate_fisheye.py"
spec = importlib.util.spec_from_file_location("calibrate_fisheye", PATH)
assert spec and spec.loader
tool = importlib.util.module_from_spec(spec)
sys.modules["calibrate_fisheye"] = tool
spec.loader.exec_module(tool)

LENS = fisheye.Lens(1920, 1080, (690.0, 688.0, 955.0, 545.0), (0.05, -0.01, 0.002, 0.0))
BOARD = (9, 6)


def views_through(lens: fisheye.Lens, n: int = 20, noise: float = 0.1) -> list[np.ndarray]:
    """Board corners as the lens would see them from `n` poses spread over the image, with a little pixel noise."""
    fx, fy, cx, cy = lens.k
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
    D = np.array(lens.d, np.float64).reshape(4, 1)
    rng = np.random.default_rng(3)
    out = []
    for i in range(n):
        angle = np.radians(rng.uniform(0, 60))  # the board's center from the lens axis, out toward the rim
        turn = 2 * np.pi * i / n
        distance = rng.uniform(8, 14)  # in squares
        center = distance * np.array([np.sin(angle) * np.cos(turn), np.sin(angle) * np.sin(turn), np.cos(angle)])
        tvec = center - np.array([4.0, 2.5, 0.0])  # the board's middle corner at the center
        rvec = rng.uniform(-0.4, 0.4, 3)
        pts, _ = cv2.fisheye.projectPoints(tool.board_points(BOARD), rvec, tvec, K, D)
        out.append(pts + rng.normal(0, noise, pts.shape))
    return out


def test_solve_recovers_the_lens_its_views_came_through():
    lens, rms = tool.solve(views_through(LENS), BOARD, (1920, 1080))
    np.testing.assert_allclose(lens.k, LENS.k, rtol=1e-2)
    np.testing.assert_allclose(lens.d, LENS.d, atol=2e-2)
    assert rms < 0.5
    # What matters to rectification: where rays land, across the field of view.
    rays = fisheye.unproject(LENS, torch.tensor([[100.0, 540.0], [1800.0, 200.0], [960.0, 1000.0]]))
    np.testing.assert_allclose(fisheye.project(lens, rays).numpy(), fisheye.project(LENS, rays).numpy(), atol=2.0)


def test_too_few_views_are_refused():
    with pytest.raises(SystemExit, match="at least"):
        tool.solve(views_through(LENS, n=4), BOARD, (1920, 1080))


def test_a_drawn_board_has_all_its_inner_corners_found():
    square, (cols, rows) = 40, BOARD
    image = np.full(((rows + 3) * square, (cols + 3) * square), 255, np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                y, x = (r + 1) * square, (c + 1) * square
                image[y : y + square, x : x + square] = 0
    corners = tool.find_corners(image, BOARD)
    assert corners is not None and corners.shape == (1, cols * rows, 2)
    assert tool.find_corners(np.full_like(image, 128), BOARD) is None


def test_the_calibration_is_written_where_see_actions_reads_it(tmp_path):
    manifest = tmp_path / "devices.toml"
    manifest.write_text(
        "# the house's cameras\n"
        '[[device]]\nname = "cam0"\nkind = "camera"\n\n'
        '[[device]]\nname = "fish0"\nkind = "camera"  # the front lens\nvideo = { flip = false }\n'
    )
    tool.write_calibration(manifest, "fish0", LENS, 0.42)
    text = manifest.read_text()
    assert "# the house's cameras" in text and "# the front lens" in text, "the rest of the file is kept"
    got = fisheye.lens_for(manifest, "fish0")
    assert got is not None
    np.testing.assert_allclose(got.k, LENS.k, atol=1e-3)
    np.testing.assert_allclose(got.d, LENS.d, atol=1e-6)
    assert fisheye.lens_for(manifest, "cam0") is None
    with pytest.raises(SystemExit, match="no device named"):
        tool.write_calibration(manifest, "nobody", LENS, 0.4)
