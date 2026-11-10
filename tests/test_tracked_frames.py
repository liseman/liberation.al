"""see-actions' analyzed frames rebuilt from its segment rows (src/liberational/see/tracked_frames.py), on a real
segment store with see-actions' own schemas."""

import uuid
from pathlib import Path

import numpy as np
import pytest

from liberational import segment_log
from liberational.see import tracked_frames as tf
from liberational.see.actions import COVERAGE_SCHEMA, JOINTS, POSE_SCHEMA

T = 1_800_000_000 * tf.NS
S = tf.NS  # a second, in ns


class ActionsRows:
    """Writes `pose.track` and `action.coverage` rows as see-actions does, one camera at a time."""

    def __init__(self, root: Path, cam: str = "cam0") -> None:
        from continuum_py import SegmentWriter

        self.poses = SegmentWriter(root, tf.POSES, cam, POSE_SCHEMA)
        self.coverage = SegmentWriter(root, tf.COVERAGE, cam, COVERAGE_SCHEMA)

    def people(self, ts_ns: int, tracks: list[tuple[int, list[float]]]) -> None:
        """One frame's tracked people: (track id, [x1, y1, x2, y2]) each."""
        if not tracks:
            return
        n = len(tracks)
        frame = str(uuid.uuid7())
        self.poses.append(
            {
                "ts_ns": np.full(n, ts_ns, np.int64),
                "frame_id": [frame] * n,
                "track_id": np.array([t for t, _ in tracks], np.int64),
                "box": np.array([[*b, 0.9] for _, b in tracks], np.float32),
                "joints": np.zeros((n, JOINTS * 3), np.int16),
                "model": ["test"] * n,
            }
        )

    def covered(self, start_ns: int, end_ns: int, frames: int = 30) -> None:
        self.coverage.append(
            {
                "ts_ns": np.array([start_ns], np.int64),
                "end_ns": np.array([end_ns], np.int64),
                "frames": np.array([frames], np.int64),
                "model": ["test"],
            }
        )

    def sync(self) -> None:
        self.poses.sync()
        self.coverage.sync()

    def close(self) -> None:
        self.poses.close()
        self.coverage.close()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    pytest.importorskip("continuum_py").SegmentFollowers  # noqa: B018 (the native store must be built)
    return segment_log.root(tmp_path)


def test_a_coverage_row_gives_its_frames_and_an_empty_one_where_nobody_was(root):
    follow = tf.Follow(root)  # from now: nothing written yet
    rows = ActionsRows(root)
    rows.people(T, [(7, [0.1, 0.2, 0.3, 0.9]), (8, [0.5, 0.2, 0.7, 0.9])])
    rows.people(T + S // 2, [(7, [0.12, 0.2, 0.32, 0.9])])
    rows.covered(T, T + S)
    rows.sync()
    got = follow.poll(2.0)
    assert [(f.ts_ns - T, f.track_ids) for f in got] == [(0, ["7", "8"]), (S // 2, ["7"]), (S, [])]
    assert got[0].boxes[1] == pytest.approx([0.5, 0.2, 0.7, 0.9, 0.9])
    rows.close()


def test_rows_from_before_the_follow_started_are_not_replayed(root):
    rows = ActionsRows(root)
    rows.people(T, [(7, [0.1, 0.2, 0.3, 0.9])])
    rows.covered(T, T + S)
    rows.sync()
    follow = tf.Follow(root)
    rows.covered(T + S, T + 2 * S)
    rows.sync()
    assert [(f.ts_ns - T, f.track_ids) for f in follow.poll(2.0)] == [(2 * S, [])]
    rows.close()


def test_people_waiting_for_their_coverage_are_kept_in_the_cursor_across_a_restart(root):
    follow = tf.Follow(root)
    rows = ActionsRows(root)
    rows.people(T, [(7, [0.1, 0.2, 0.3, 0.9])])
    rows.sync()
    assert follow.poll(0.5) == []  # no coverage yet: the frame isn't known to be complete
    saved = follow.cursor
    assert saved[tf.POSES]["cam0"] == 0  # resumes at the held row, not past it
    again = tf.Follow(root, saved)
    rows.covered(T, T + S)
    rows.sync()
    assert [(f.ts_ns - T, f.track_ids) for f in again.poll(2.0)] == [(0, ["7"]), (S, [])]
    rows.close()


def test_a_window_of_history_has_the_frames_inside_it(root):
    rows = ActionsRows(root)
    rows.people(T, [(7, [0.1, 0.2, 0.3, 0.9])])
    rows.covered(T, T + S)
    rows.people(T + 3 * S // 2, [(7, [0.1, 0.2, 0.3, 0.9])])
    rows.covered(T + S, T + 2 * S)
    rows.close()
    got = tf.between(root, "cam0", (T + S // 2) / tf.NS, (T + 2 * S) / tf.NS + 0.001)
    assert [(f.ts_ns - T, f.track_ids) for f in got] == [(S, []), (3 * S // 2, ["7"]), (2 * S, [])]
