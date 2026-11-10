"""A camera's own intrinsics, held: stored per camera, and put on a registration in place of tonight's."""

import numpy as np
import pytest

from liberational.memory import places
from liberational.spatial import cameras, lens, registration

T = 1_800_000_000.0


def test_the_newest_held_intrinsics_of_each_camera_are_read_back(tmp_path):
    assert lens.held(tmp_path) == {}, "no store, nothing held, and none made"
    lens.hold(tmp_path, "den", lens.Held(1.1, -0.3, "walk fit, 12 pairs", 2.0), now=T)
    lens.hold(tmp_path, "hall", lens.Held(0.66, 0.0, "median of 30 nightly fits"), now=T)
    lens.hold(tmp_path, "den", lens.Held(1.05, -0.25, "walk fit, 20 pairs", 1.5), now=T)  # same second: still newest
    assert lens.held(tmp_path) == {
        "den": lens.Held(1.05, -0.25, "walk fit, 20 pairs", 1.5),
        "hall": lens.Held(0.66, 0.0, "median of 30 nightly fits"),
    }


def test_a_focal_that_isnt_positive_isnt_held(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        lens.hold(tmp_path, "den", lens.Held(0.0, 0.0, "x"))


def pose(focal=300.0, width=640, height=360, lens_tuple=None):
    m = np.eye(4)
    m[:3, 3] = (1.0, 2.0, 1.5)
    return places.CameraPose(tuple(float(v) for v in m.ravel()), focal, width, height, T, lens_tuple)


def test_a_lone_cameras_pose_takes_its_held_focal_and_lens(monkeypatch):
    monkeypatch.setattr(lens, "as_registered", lambda camera: None)
    got = cameras.with_held("den", pose(), lens.Held(0.5, -0.2, "walk"))
    assert got.pose == pose().pose and got.width == 640
    assert got.lens is not None and got.lens[0] == pytest.approx(0.5) and got.lens[4] == pytest.approx(-0.2)
    same = registration.camera_pose(np.asarray(pose().pose).reshape(4, 4), 0.5 * 640, -0.2, (640, 360), T)
    assert got.focal == pytest.approx(same.focal), "kept as a registration keeps a fitted lens"


def test_a_camera_whose_configured_lens_undoes_its_frames_takes_only_the_held_focal(monkeypatch):
    configured = (0.5, 0.875, 0.5, 0.5, -0.3)
    monkeypatch.setattr(lens, "as_registered", lambda camera: configured)
    got = cameras.with_held("studio", pose(lens_tuple=configured), lens.Held(0.44, 0.0, "nightly"))
    assert got.focal == pytest.approx(0.44 * 640) and got.lens == configured


def test_nothing_held_leaves_the_pose_as_it_is():
    assert cameras.with_held("den", pose(), None) == pose()
