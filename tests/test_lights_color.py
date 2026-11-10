"""Lamps-on time from the cameras' colour cast (memory/lights_color.py), sampled from their frames on the broker."""

import datetime as dt
import pathlib
import tempfile
from zoneinfo import ZoneInfo

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from liberational.memory import lights_color

T = 1_800_000_000.0
LA = ZoneInfo("America/Los_Angeles")


def test_each_sample_is_the_camera_frame_nearest_its_time(tmp_path, vision):
    warm = np.zeros((36, 64, 3), np.uint8)
    warm[:] = (60, 120, 160)  # BGR: lamplight, blue well below green
    neutral = np.full((36, 64, 3), 90, np.uint8)
    vision.frames[("cam0", T + 0.5)] = warm  # half a second off its slot: still that slot's frame
    vision.frames[("cam0", T + 300)] = neutral
    rows = lights_color.sample(tmp_path, "cam0", T, T + 900)  # the third slot has no video: no sample
    assert [(r["ts"], round(r["gap"]), round(r["mean"])) for r in rows] == [(T, -60, 113), (T + 300, 0, 90)]


def _local_day_seconds(day: dt.date, tz: dt.tzinfo) -> float:
    start = dt.datetime.combine(day, dt.time(), tzinfo=tz)
    return (start + dt.timedelta(days=1)).timestamp() - start.timestamp()


def test_analyse_day_uses_the_real_local_day_length_on_dst_transition_days(tmp_path):
    spring = lights_color.analyse_day(tmp_path, dt.date(2026, 3, 8), cameras=(), tz=LA)
    fall = lights_color.analyse_day(tmp_path, dt.date(2026, 11, 1), cameras=(), tz=LA)
    assert spring["window_s"] == 23 * 3600
    assert fall["window_s"] == 25 * 3600


@given(day=st.dates(min_value=dt.date(2026, 3, 1), max_value=dt.date(2026, 3, 15)))
def test_analyse_day_window_matches_local_midnight_delta_across_spring_dst(day):
    with tempfile.TemporaryDirectory() as tmpdir:
        res = lights_color.analyse_day(pathlib.Path(tmpdir), day, cameras=(), tz=LA)
        assert res["window_s"] == _local_day_seconds(day, LA)


@given(day=st.dates(min_value=dt.date(2026, 10, 25), max_value=dt.date(2026, 11, 8)))
def test_analyse_day_window_matches_local_midnight_delta_across_fall_dst(day):
    with tempfile.TemporaryDirectory() as tmpdir:
        res = lights_color.analyse_day(pathlib.Path(tmpdir), day, cameras=(), tz=LA)
        assert res["window_s"] == _local_day_seconds(day, LA)
