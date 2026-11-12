"""tools/measure_vitals.py: the parts that turn recorded rows into a rate a person can compare with a watch."""

import importlib.util
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("measure_vitals", ROOT / "tools" / "measure_vitals.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


measure_vitals = load()
NS = measure_vitals.NS


def test_a_time_is_read_as_epoch_seconds_either_way():
    assert measure_vitals.parse_time("1800000000") == 1_800_000_000.0
    # A local wall-clock time, which is what the owner types after reading his watch.
    got = measure_vitals.parse_time("2027-01-15T20:40")
    import time

    assert time.strftime("%Y-%m-%dT%H:%M", time.localtime(got)) == "2027-01-15T20:40"
    with pytest.raises(SystemExit):
        measure_vitals.parse_time("quarter past eight")


def test_a_frame_takes_the_nearest_recorded_tick():
    index = np.array([100, 200, 300], np.int64)
    assert measure_vitals.nearest(index, 140) == 0
    assert measure_vitals.nearest(index, 160) == 1
    assert measure_vitals.nearest(index, 10**9) == 2  # past the end: the last one
    assert measure_vitals.nearest(np.zeros(0, np.int64), 5) is None


def test_the_recorded_joints_of_a_tick_are_frame_pixels_for_each_track():
    # Two tracks on one frame: their joints as pose.track stores them, normalized to the frame times JOINT_EDGE.
    joints = np.zeros((2, 2, 3), np.float64)
    joints[0, 0] = (0.5 * measure_vitals.JOINT_EDGE, 0.25 * measure_vitals.JOINT_EDGE, 0.9 * measure_vitals.JOINT_EDGE)
    joints[1, 0] = (0.1 * measure_vitals.JOINT_EDGE, 0.2 * measure_vitals.JOINT_EDGE, 0.8 * measure_vitals.JOINT_EDGE)
    rows = {"ts_ns": np.array([10, 10], np.int64), "track_id": np.array([4, 9], np.int64), "joints": joints}
    got = measure_vitals.joints_at(rows, np.sort(rows["ts_ns"]), 10, width=200, height=100)
    assert [(track, tuple(person[0][:2])) for track, person in got] == [
        (4, (100.0, 25.0)),
        (9, (20.0, 20.0)),
    ]
    assert got[0][1][0][2] == pytest.approx(0.9)


def test_a_tick_with_no_recorded_frame_near_it_has_no_people():
    rows = {"ts_ns": np.array([10], np.int64), "track_id": np.array([4], np.int64), "joints": np.zeros((1, 2, 3))}
    other = 10 + NS // 2  # half a second away: no frame close enough to say where the face was
    assert measure_vitals.joints_at(rows, np.sort(rows["ts_ns"]), other, 200, 100) == []


def test_the_readings_are_counted_by_whole_minute_from_the_first():
    readings = [
        {"at": 1_000.0, "bpm": None},
        {"at": 1_010.0, "bpm": 70.0},
        {"at": 1_050.0, "bpm": 72.0},
        {"at": 1_075.0, "bpm": 74.0},
    ]
    # From the first reading at 1000 s: 1010 and 1050 are minute 0, and 1075 is 75 s in, minute 1.
    assert measure_vitals.minutes(readings) == {0: [70.0, 72.0], 1: [74.0]}
    assert measure_vitals.minutes([]) == {}
