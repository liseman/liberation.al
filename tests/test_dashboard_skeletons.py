"""The skeleton playback page's reads (dashboard/skeletons.py) on a real segment store."""

from pathlib import Path

import numpy as np
import pytest

from liberational import segment_log
from liberational.dashboard import skeletons
from liberational.see.actions import JOINTS, POSE_SCHEMA, SCALE
from liberational.see.actions_worker import POSES

T = 1_800_000_000.0  # a fixed moment, in seconds


@pytest.fixture
def state(tmp_path: Path) -> Path:
    pytest.importorskip("continuum_py").SegmentWriter  # noqa: B018 (the native store must be built)
    return tmp_path


def write_poses(state: Path, cam: str, rows: list[tuple[float, int, float]]) -> None:
    """(time, track id, the nose's x) per row; every other joint is unplaced."""
    from continuum_py import SegmentWriter

    joints = np.full((len(rows), JOINTS * 3), -SCALE, np.int16)
    joints[:, 2::3] = 0
    for i, (_, _, x) in enumerate(rows):
        joints[i, :3] = (round(x * SCALE), round(0.25 * SCALE), round(0.9 * SCALE))
    writer = SegmentWriter(segment_log.root(state), POSES, cam, POSE_SCHEMA)
    writer.append(
        {
            "ts_ns": np.array([round(t * 1e9) for t, _, _ in rows], np.int64),
            "frame_id": [segment_log.new_id() for _ in rows],
            "track_id": np.array([tid for _, tid, _ in rows], np.int64),
            "box": np.array([[0.1, 0.2, 0.3, 0.9, 0.8]] * len(rows), np.float32),
            "joints": joints,
            "model": ["test"] * len(rows),
        }
    )
    writer.close()


def test_poses_are_one_tracks_frames_in_the_window_decoded_to_the_frame(state):
    write_poses(state, "cam0", [(T, 7, 0.5), (T + 0.1, 8, 0.9), (T + 0.2, 7, 0.6), (T + 9, 7, 0.7)])
    got = skeletons.poses(state, cam="cam0", track_id="7", start=T - 2, end=T + 5)
    assert [f["ts"] for f in got["frames"]] == [pytest.approx(T), pytest.approx(T + 0.2)]
    first = got["frames"][0]
    assert first["keypoints"][0] == pytest.approx([0.5, 0.25, 0.9])
    assert first["keypoints"][1] == [-1.0, -1.0, 0.0]  # unplaced
    assert len(first["keypoints"]) == JOINTS
    assert skeletons.poses(state, cam="cam1", track_id="7", start=T - 2, end=T + 5)["frames"] == []


@pytest.mark.parametrize(("start", "end"), [(T, T), (T, T + skeletons.MAX_WINDOW_S + 1)], ids=["empty", "too-long"])
def test_a_window_that_is_empty_or_too_long_is_refused(state, start, end):
    with pytest.raises(ValueError, match="window"):
        skeletons.poses(state, cam="cam0", track_id="7", start=start, end=end)


def test_spans_list_actions_and_walk_bys_newest_first_with_their_kind_and_person(state, monkeypatch):
    # conftest swaps the segment store's reads for a fake; this test reads the real one it writes
    monkeypatch.setattr(segment_log, "at", lambda state=None: segment_log.Log(segment_log.root(state)))
    writers = segment_log.Writers(segment_log.root(state))
    action = {
        "id": segment_log.new_id(),
        "ts_start": T - 600,
        "ts_end": T - 590,
        "track_id": "7",
        "subject_type": "person",
        "kind": "squat",
        "person": "p1",
        "person_name": "Luca",
    }
    walk = {
        "id": segment_log.new_id(),
        "ts_start": T - 60,
        "ts_end": T - 55,
        "track_id": "9",
        "attributes": {"direction": "left"},
    }
    old = {**action, "id": segment_log.new_id(), "ts_start": T - 90_000, "ts_end": T - 89_990}
    writers.append(segment_log.ACTION_SPANS, "cam0", [old, action])
    writers.append(segment_log.WALKBY_SPANS, "cam1", [walk])
    writers.close()
    got = skeletons.spans(state, hours=24, now=T)
    assert [(s["source"], s["cam"], s["kind"], s["track_id"], s["person"]) for s in got] == [
        ("walk-by", "cam1", "walk-by left", "9", None),
        ("action", "cam0", "squat", "7", "Luca"),
    ]
