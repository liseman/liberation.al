"""Times someone was in view, from analyzed frames (src/liberational/memory/presence_episodes.py)."""

import itertools

import pytest
from hypothesis import given
from hypothesis import strategies as st

from liberational.memory.presence_episodes import Episodes

T = 1_800_000_000.0


def feed(e: Episodes, frames: list[tuple[float, list[str]]]) -> list[dict]:
    out = []
    for ts, tracks in frames:
        out += e.frame(T + ts, tracks)
    return out


def every(start: float, end: float, step: float, tracks: list[str]) -> list[tuple[float, list[str]]]:
    n = round((end - start) / step)
    return [(start + i * step, tracks) for i in range(n)]


def test_someone_in_view_then_gone_for_the_absence_is_one_episode():
    e = Episodes("cam0")
    frames = [(0.0, []), *every(1, 5, 0.5, ["7"]), *every(5, 20, 0.5, [])]
    (ep,) = feed(e, frames)
    assert (ep["ts_start"] - T, ep["ts_end"] - T) == (1.0, 4.5)
    assert (ep["tracks"], ep["max_people"], ep["left_censored"], ep["right_censored"]) == (1, 1, False, False)


def test_a_short_absence_or_a_track_split_is_still_one_episode():
    e = Episodes("cam0")
    frames = [(0.0, []), *every(1, 5, 0.5, ["7"]), *every(5, 12, 0.5, []), *every(12, 14, 0.5, ["9", "7"])]
    frames += every(14, 30, 0.5, [])
    (ep,) = feed(e, frames)
    assert (ep["ts_start"] - T, ep["ts_end"] - T, ep["tracks"], ep["max_people"]) == (1.0, 13.5, 2, 2)


def test_nobody_seen_for_the_absence_starts_a_new_episode():
    e = Episodes("cam0")
    frames = [(0.0, []), *every(1, 2, 0.5, ["7"]), *every(2, 12, 0.5, []), *every(12, 13, 0.5, ["8"])]
    frames += every(13, 30, 0.5, [])
    starts = [ep["ts_start"] - T for ep in feed(e, frames)]
    assert starts == [1.0, 12.0]


def test_an_unwatched_gap_censors_both_sides():
    e = Episodes("cam0")
    frames = [*every(0, 3, 0.5, ["7"]), *every(40, 42, 0.5, ["7"]), *every(42, 60, 0.5, [])]
    first, second = feed(e, frames)
    assert (first["left_censored"], first["right_censored"]) == (True, True)  # it began as the camera started
    assert (second["left_censored"], second["right_censored"]) == (True, False)
    assert second["ts_start"] - T == 40.0


def test_an_older_frame_changes_nothing():
    e = Episodes("cam0")
    feed(e, [(0.0, []), (1.0, ["7"])])
    before = e.snapshot()
    assert e.frame(T + 0.5, ["8"]) == []
    assert e.snapshot() == before


@given(st.lists(st.tuples(st.floats(0.1, 40), st.booleans()), max_size=60), st.integers(0, 59))
def test_a_restored_snapshot_continues_exactly_as_if_nothing_stopped(steps, cut):
    frames, ts = [], 0.0
    for dt, someone in steps:
        ts += dt
        frames.append((ts, ["7"] if someone else []))
    cut = min(cut, len(frames))
    whole = Episodes("cam0")
    expected = feed(whole, frames)
    first = Episodes("cam0")
    got = feed(first, frames[:cut])
    resumed = Episodes("cam0", state=first.snapshot())
    got += feed(resumed, frames[cut:])
    assert got == expected
    assert resumed.snapshot() == whole.snapshot()


@pytest.mark.parametrize("absence", [0.0, 10.0])
def test_episodes_never_overlap_and_end_after_they_start(absence):
    e = Episodes("cam0", absence_s=absence)
    frames = [(i * 0.7, ["7"] if (i // 9) % 2 == 0 else []) for i in range(200)]
    eps = feed(e, frames)
    assert all(a["ts_end"] < b["ts_start"] for a, b in itertools.pairwise(eps))
    assert all(ep["ts_end"] >= ep["ts_start"] for ep in eps)
