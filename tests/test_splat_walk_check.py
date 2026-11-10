"""tools/splat_walk_check.py: the known error it adds to a walk's poses, the views it keeps, and the exact copy."""

import importlib.util
import json
import pathlib

import numpy as np
import pytest

from liberational.spatial import splats

_spec = importlib.util.spec_from_file_location(
    "splat_walk_check", pathlib.Path(__file__).resolve().parents[1] / "tools" / "splat_walk_check.py"
)
assert _spec is not None and _spec.loader is not None
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)

# A camera 3 m along x and 1 m up, turned a quarter turn about z: not the identity, so a move made in the world's
# frame instead of the camera's would show.
POSE = np.array([[0.0, -1.0, 0.0, 3.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]])


def view(i: int, depth: bool = True) -> dict:
    made = {"image": f"f{i:03d}.jpg", "pose": POSE.ravel().tolist(), "focal": 500.0, "width": 48, "height": 64}
    return {**made, "depth": f"d{i:03d}.png"} if depth else made


def moves(before: dict, after: dict) -> tuple[np.ndarray, float]:
    """How far `after`'s camera centre is from `before`'s, in the world, and the angle between their rotations."""
    a, b = (np.asarray(v["pose"]).reshape(4, 4) for v in (before, after))
    cos = (np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2
    return b[:3, 3] - a[:3, 3], float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def test_every_pose_is_moved_by_the_jitter_each_in_its_own_direction():
    views = [view(i) for i in range(3)]
    got = check.jittered(views, 0.02, 0.3)
    shifts, turns = zip(*(moves(a, b) for a, b in zip(views, got, strict=True)), strict=True)
    assert [float(np.linalg.norm(s)) for s in shifts] == pytest.approx([0.02] * 3)
    assert list(turns) == pytest.approx([0.3] * 3, abs=1e-6)
    unit = [s / 0.02 for s in shifts]
    assert abs(float(unit[0] @ unit[1])) < 0.999 and abs(float(unit[1] @ unit[2])) < 0.999, "not one shared direction"
    assert [v["image"] for v in got] == [v["image"] for v in views] and got[0]["focal"] == 500.0
    assert views[0]["pose"] == POSE.ravel().tolist(), "the views given are left as they were"


def test_a_work_folder_with_its_views_kept_needs_no_recording(tmp_path):
    (tmp_path / "views.json").write_text(json.dumps({"views": [view(0)], "scale": 0.5}))
    views, scale = check.walk_views(tmp_path / "gone", tmp_path)
    assert views == [view(0)] and scale == 0.5


def test_the_exact_copy_is_drawn_once_and_stands_in_for_every_views_image_and_depth(tmp_path, monkeypatch):
    views = [view(0), view(1, depth=False)]
    calls = []

    def train_walk(views, points, colors, *, scale, steps, render):
        calls.append({"views": len(views), "scale": scale, "steps": steps})
        render.mkdir()
        for i in range(len(views)):
            (render / f"f{i:03d}.png").write_bytes(b"drawn")
        return {}, b""

    monkeypatch.setattr(splats, "train_walk", train_walk)
    folder = tmp_path / "exact"
    got = check.exact(views, np.zeros((1, 3)), None, 0.5, 70, folder)
    assert [v["image"] for v in got] == [str(folder / "f000.png"), str(folder / "f001.png")]
    assert got[0]["depth"] == str(folder / "d000.png") and "depth" not in got[1], "only a view with depth gets one"
    assert got[0]["pose"] == views[0]["pose"], "at the poses as given"
    assert calls == [{"views": 2, "scale": 0.5, "steps": 70}], "every view trains, none held out"
    check.exact(views, np.zeros((1, 3)), None, 0.5, 70, folder)
    assert len(calls) == 1, "a copy already drawn is used as it is"
