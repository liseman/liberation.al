"""The segment store from Python: typed columns written with SegmentWriter read back exactly with
read_segments and SegmentFollower, NumPy in and out, with the part set a read saw."""

from pathlib import Path

import numpy as np
import pytest
from continuum_py import SegmentFollower, SegmentWriter, read_segments

H = 3_600_000_000_000
T0 = 490_000 * H
POSES = [
    ("ts_ns", "i64"),
    ("frame_id", "id"),
    ("track_id", "i64"),
    ("joints", "i16[6]"),
    ("box", "f32[5]"),
    ("model", "str"),
]
FRAME = "0199a000-0000-7000-8000-000000000001"


def poses(ts: list[int], track: int = 7) -> dict:
    n = len(ts)
    return {
        "ts_ns": np.array(ts, np.int64),
        "frame_id": [FRAME] * n,
        "track_id": np.full(n, track, np.int64),
        "joints": np.arange(n * 6, dtype=np.int16).reshape(n, 6),
        "box": np.tile(np.array([0.1, 0.2, 0.3, 0.4, 0.9], np.float32), (n, 1)),
        "model": ["pose-v1"] * n,
    }


def test_numpy_columns_read_back_exactly_with_their_row_seq_and_the_part_set(tmp_path: Path) -> None:
    w = SegmentWriter(tmp_path, "pose.body", "cam0", POSES)
    assert w.append(poses([T0 + 1, T0 + 2])) == 0
    assert w.next_seq == 2
    got = read_segments(tmp_path, "pose.body", POSES, T0, T0 + H)
    assert got["camera"] == ["cam0", "cam0"]
    assert got["ts_ns"].tolist() == [T0 + 1, T0 + 2]
    assert got["row_seq"].tolist() == [0, 1]
    assert got["frame_id"] == [FRAME, FRAME]
    assert got["joints"].dtype == np.int16 and got["joints"].shape == (2, 6)
    assert got["joints"].tolist() == [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]
    assert got["box"].shape == (2, 5) and got["box"][0, 4] == pytest.approx(0.9)
    assert got["model"] == ["pose-v1", "pose-v1"]
    assert [(p["camera"], p["end"], p["rows"]) for p in got["parts"]] == [("cam0", None, 2)], "open, as far as read"
    w.close()
    assert [p["end"] for p in read_segments(tmp_path, "pose.body", POSES, T0, T0 + H)["parts"]] == [2]


def test_lists_strings_and_nullable_strings_round_trip(tmp_path: Path) -> None:
    schema = [("ts_ns", "i64"), ("boxes", "f32[]"), ("names", "i64[]"), ("who", "str?"), ("ok", "bool")]
    w = SegmentWriter(tmp_path, "body.detection", "cam1", schema)
    w.append(
        {
            "ts_ns": [T0 + 5, T0 + 6],
            "boxes": [np.array([0.5, 0.5, 0.6, 0.9, 0.8], np.float32), []],
            "names": [[1, 2], [3]],
            "who": ["Sam", None],
            "ok": [True, False],
        }
    )
    got = read_segments(tmp_path, "body.detection", schema, T0, T0 + H, cameras=["cam1"])
    assert [b.tolist() for b in got["boxes"]] == [pytest.approx([0.5, 0.5, 0.6, 0.9, 0.8]), []]
    assert [n.tolist() for n in got["names"]] == [[1, 2], [3]], "an empty frame is an empty list, not a missing row"
    assert got["who"] == ["Sam", None]
    assert got["ok"].tolist() == [True, False]


def test_a_follower_resumes_from_its_saved_cursor(tmp_path: Path) -> None:
    w = SegmentWriter(tmp_path, "pose.body", "cam0", POSES)
    f = SegmentFollower(tmp_path, "pose.body", POSES)
    assert f.poll()["row_seq"] == []
    w.append(poses([T0 + 1, T0 + 2]))
    assert f.poll()["row_seq"].tolist() == [0, 1]
    saved = f.cursor
    assert saved == {"cam0": 2}
    w.append(poses([T0 + 3]))
    resumed = SegmentFollower(tmp_path, "pose.body", POSES, cursor=saved)
    assert resumed.poll()["ts_ns"].tolist() == [T0 + 3]


def test_bad_schemas_columns_and_a_second_writer_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no segment column type"):
        SegmentWriter(tmp_path, "s", "cam0", [("ts_ns", "i64"), ("x", "complex")])
    with pytest.raises(ValueError, match="ts_ns"):
        SegmentWriter(tmp_path, "s", "cam0", [("x", "f32")])
    w = SegmentWriter(tmp_path, "pose.body", "cam0", POSES)
    with pytest.raises(RuntimeError, match="another writer"):
        SegmentWriter(tmp_path, "pose.body", "cam0", POSES)
    rows = poses([T0 + 1])
    rows["joints"] = np.zeros((1, 5), np.int16)
    with pytest.raises(ValueError, match="fixed list of 6"):
        w.append(rows)
    rows = poses([T0 + 1])
    del rows["model"]
    with pytest.raises(KeyError):
        w.append(rows)
    with pytest.raises(ValueError, match="can't hold None"):
        w.append({**poses([T0 + 1]), "model": [None]})
    w.close()
    with pytest.raises(RuntimeError, match="closed"):
        w.append(poses([T0 + 1]))


def test_sync_then_resume_at_writes_each_replayed_row_once(tmp_path: Path) -> None:
    w = SegmentWriter(tmp_path, "pose.body", "cam0", POSES)
    w.append(poses([T0 + 1]))
    saved = w.sync()
    w.append(poses([T0 + 2]))  # past the checkpoint: replayed after the restart
    w.close()
    w = SegmentWriter(tmp_path, "pose.body", "cam0", POSES, resume_at=saved)
    w.append(poses([T0 + 2]))
    w.close()
    assert read_segments(tmp_path, "pose.body", POSES, T0, T0 + H)["ts_ns"].tolist() == [T0 + 1, T0 + 2]


def test_an_overlapping_read_finds_a_visit_that_began_before_the_window(tmp_path: Path) -> None:
    schema = [("ts_ns", "i64"), ("end_ns", "i64"), ("zone", "str")]
    w = SegmentWriter(tmp_path, "zone.visit", "cam0", schema)
    w.append({"ts_ns": [T0 - 100], "end_ns": [T0 + 50], "zone": ["door"]})
    w.close()
    assert read_segments(tmp_path, "zone.visit", schema, T0, T0 + H)["zone"] == []
    assert read_segments(tmp_path, "zone.visit", schema, T0, T0 + H, overlapping=True)["zone"] == ["door"]


def test_followers_wait_for_rows_across_streams_with_one_cursor(tmp_path: Path) -> None:
    from continuum_py import SegmentFollowers

    plain = [("ts_ns", "i64"), ("score", "f32")]
    bodies = SegmentWriter(tmp_path, "body.detection", "cam0", plain)
    motion = SegmentWriter(tmp_path, "camera.motion", "cam0", plain)
    f = SegmentFollowers(tmp_path, {"body.detection": plain, "camera.motion": plain})
    assert f.poll(timeout_s=0.05) == []
    motion.append({"ts_ns": [T0 + 2], "score": np.array([0.5], np.float32)})
    bodies.append({"ts_ns": [T0 + 1], "score": np.array([0.9], np.float32)})
    got = f.poll(timeout_s=5.0)
    assert [(s, rows["ts_ns"].tolist()) for s, rows in got] == [
        ("body.detection", [T0 + 1]),
        ("camera.motion", [T0 + 2]),
    ]
    assert f.cursor == {"body.detection": {"cam0": 1}, "camera.motion": {"cam0": 1}}


def test_a_forget_hides_rows_and_the_feed_says_so(tmp_path: Path) -> None:
    from continuum_py import forget_segments, segment_deletions, segment_revision

    w = SegmentWriter(tmp_path, "pose.body", "cam0", POSES)
    w.append(poses([T0 + 1, T0 + 2]))
    before = read_segments(tmp_path, "pose.body", POSES, T0, T0 + H)
    assert before["revision"] == 0
    d = forget_segments(tmp_path, T0 + 2, T0 + 3, streams=["pose.body"])
    assert (d["revision"], d["kind"], d["streams"], d["cameras"]) == (1, "forget", ["pose.body"], [])
    after = read_segments(tmp_path, "pose.body", POSES, T0, T0 + H)
    assert after["ts_ns"].tolist() == [T0 + 1]
    assert after["revision"] == 1
    assert [x["revision"] for x in segment_deletions(tmp_path, after=before["revision"])] == [1]
    assert segment_revision(tmp_path) == 1
    with pytest.raises(ValueError, match="forget"):
        forget_segments(tmp_path, T0, T0 + 1, kind="erase")  # ty: ignore[invalid-argument-type]  # a wrong kind on purpose


def test_forgetting_a_person_by_an_id_column_keeps_rows_that_name_nobody(tmp_path: Path) -> None:
    from continuum_py import forget_segments_where, segment_deletions

    schema = [("ts_ns", "i64"), ("voiceprint", "id"), ("person", "id?")]
    sam, dana = "0199a000-0000-7000-8000-00000000000a", "0199a000-0000-7000-8000-00000000000b"
    w = SegmentWriter(tmp_path, "voice.embedding", "wave3", schema)
    w.append({"ts_ns": [T0 + 1, T0 + 2, T0 + 3], "voiceprint": [FRAME] * 3, "person": [sam, dana, None]})
    d = forget_segments_where(tmp_path, "person", sam, streams=["voice.embedding"])
    assert (d["column"], d["value"]) == ("person", sam)
    left = read_segments(tmp_path, "voice.embedding", schema, 0, 2**63)
    assert left["person"] == [dana, None]
    assert [x["value"] for x in segment_deletions(tmp_path)] == [sam]
