"""What the spatial jobs that work from the cameras' frames share: which cameras are fixed, and how frames are
scored."""

import textwrap

import numpy as np
import pytest

from liberational.spatial import cameras


def test_fixed_cameras_are_the_enabled_cameras_without_a_fisheye_lens(tmp_path):
    manifest = tmp_path / "devices.toml"
    manifest.write_text(
        textwrap.dedent(
            """
            [[device]]
            name = "desk"
            kind = "camera"
            [[device]]
            name = "mic"
            kind = "microphone"
            [[device]]
            name = "rig-front"
            kind = "camera"
            fisheye = { width = 1920, height = 1080, k = [1, 1, 1, 1], d = [0, 0, 0, 0] }
            [[device]]
            name = "porch"
            kind = "rtsp"
            [[device]]
            name = "pack-back"
            kind = "camera"
            mount = { rig = "pack", hfov_deg = 120.0 }
            [[rig]]
            name = "pack"
            [[device]]
            name = "attic"
            kind = "rtsp"
            enabled = false
            """
        )
    )
    assert cameras.fixed_cameras(manifest) == ["desk", "porch"]
    assert cameras.fixed_cameras(tmp_path / "missing.toml") == []


def frame(b, g, r):
    return np.full((48, 64, 3), (b, g, r), dtype=np.uint8)


@pytest.mark.parametrize(
    ("pixels", "lit"),
    [
        pytest.param(frame(90, 120, 160), True, id="lit-colour"),
        pytest.param(frame(20, 25, 30), False, id="dark"),
        pytest.param(frame(140, 140, 141), False, id="infrared-grey"),
        pytest.param(np.full((48, 64), 150, dtype=np.uint8), False, id="single-channel"),
    ],
)
def test_only_lit_colour_frames_are_daylight(pixels, lit):
    assert cameras.daylight(pixels) is lit


def test_a_frames_light_is_its_mean_brightness_and_channel_spread():
    assert cameras.light(frame(90, 120, 160)) == pytest.approx((370 / 3, 70.0))
    assert cameras.light(np.zeros((4, 4), dtype=np.uint8)) == (0.0, 0.0)


def test_a_blurred_frame_scores_less_sharp_than_the_frame_it_was_blurred_from():
    cv2 = pytest.importorskip("cv2")
    crisp = np.random.default_rng(0).integers(0, 255, (72, 128, 3), dtype=np.uint8)
    assert cameras.sharpness(cv2.GaussianBlur(crisp, (9, 9), 3.0)) < cameras.sharpness(crisp) / 10
    assert cameras.sharpness(np.full((72, 128, 3), 90, np.uint8)) == 0.0
