"""Movement and walk-bys from see-actions' tracks (see/action_movement.py), and the camera shift they're judged with."""

from typing import Any

import numpy as np
import pytest
import torch

from liberational import segment_log
from liberational.see import action_movement, actions, gpu_pose
from liberational.see.vision_backend import People
from test_see_actions import STEP, Decoded, Script, T, frame_id, nobody

LINE = {"kind": "vertical", "position": 0.5}


class Shifting(Script):
    """A scripted backend whose camera shifts by `step` (x, as a fraction of the frame) between samples."""

    def __init__(self, frames: list[People], step: float = 0.0) -> None:
        super().__init__(frames)
        self.step = step

    def shift(self, image: str) -> tuple[float, float] | None:
        return (self.step, 0.0)


def walker(x: float) -> People:
    box = np.array([[x, 0.2, x + 0.1, 0.9, 0.9]], np.float32)
    return People(box, np.full((1, actions.JOINTS, 3), 0.9, np.float32))  # a confident pose: a person


def walk_past(step: float = 0.0) -> dict[str, list[dict[str, Any]]]:
    """Someone crossing the camera's centre line left to right over 3 s, then 13 s of nobody (25 fps)."""
    frames = [walker(0.1 + 0.7 * i / 75) for i in range(75)] + [nobody()] * 325
    written: dict[str, list[dict[str, Any]]] = {}
    movement = action_movement.Movement("cam0", lambda stream, rows: written.setdefault(stream, []).extend(rows))
    movement.walks._state["line"] = dict(LINE)
    pipeline = actions.ActionPipeline(Shifting(frames, step), _Rows(), _Rows(), "m", movement=movement)
    for i in range(len(frames)):
        pipeline.feed(Decoded(frame_id(i), T + i * STEP))
    pipeline.close()
    return written


class _Rows:
    def append(self, columns: dict[str, Any]) -> None:
        pass


def test_someone_walking_past_is_one_walk_by_with_movement_and_coverage():
    written = walk_past()
    (walk,) = written[segment_log.WALKBY_SPANS]
    assert walk["attributes"]["direction"] == "left_to_right"
    assert walk["camera_moved"] is False and walk["uncertainty"] == {}
    assert T / 1e9 < walk["ts_start"] < (T + 75 * STEP) / 1e9
    assert written[segment_log.MOVEMENT_TRACKS]
    assert {r["cam"] for r in written[segment_log.MOVEMENT_TRACKS]} == {"cam0"}
    covered = written[segment_log.WALKBY_COVERAGE]
    assert covered[0]["ts_start"] == pytest.approx(T / 1e9)
    assert sum(r["frames"] for r in covered) == pytest.approx(400 / 5, abs=2)  # a sample every 0.2 s
    for stream, rows in written.items():
        for record in rows:
            segment_log.STREAMS[stream].to_row(record)  # every record encodes as its segment row


def test_a_walk_by_while_the_camera_moves_says_so():
    (walk,) = walk_past(step=0.2)[segment_log.WALKBY_SPANS]
    assert walk["camera_moved"] is True


def test_a_camera_shift_is_found_by_phase_correlation():
    rng = np.random.default_rng(0)
    base = torch.from_numpy(rng.integers(0, 256, (180, 320, 3), dtype=np.uint8))
    moved = torch.roll(base, shifts=(0, 20), dims=(0, 1))  # 20 px right: 10 px of the 160-wide thumbnail
    first, spectrum = gpu_pose.phase_shift(base, None)
    assert first is None
    shift, _ = gpu_pose.phase_shift(moved, spectrum)
    assert shift == pytest.approx((10 / 160, 0.0))
    still, _ = gpu_pose.phase_shift(base, spectrum)
    assert still == pytest.approx((0.0, 0.0))
