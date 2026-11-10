"""Each fixed camera's pose over time: observations stored and read back, split into still stretches where the
camera moved, and smoothed within a stretch."""

import math

import numpy as np
import pytest

from liberational import segment_log as S
from liberational.spatial import poses as P

T = 1_790_000_000.0  # before the test run's clock: `series` reads up to now
DAY = 86_400.0
HOME, FRAME = "0190c0de-0000-7000-8000-000000000001", "frame-1"
AWAY = "0190c0de-0000-7000-8000-000000000002"


@pytest.fixture
def store(monkeypatch, vision):
    """The real segment store under the test's state (conftest fakes its reads in every test)."""
    monkeypatch.setattr(S, "at", lambda state=None: S.Log(S.root(state)))


def rx(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rz(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# A camera turned 30 degrees and tipped 35 down: the truth the noisy seats scatter around.
TRUE = rz(30) @ rx(35)


def pose(xyz, rotation=TRUE):
    m = np.eye(4)
    m[:3, :3] = rotation
    m[:3, 3] = xyz
    return m


def seen(t, xyz, rotation=TRUE, source="seat", camera="cam", place=HOME, frame=FRAME):
    return P.Observation(t, camera, source, place, frame, pose(xyz, rotation), 0.9)


def owner(t, xyz, camera="cam"):
    return P.Observation(t, camera, "owner", HOME, FRAME, pose(xyz, np.eye(3)), 1.0)


def test_a_still_cameras_noisy_seats_smooth_to_the_true_rotation_and_the_median_spot():
    # Tilts 3 degrees either way and turns 4 either way about the truth: the sum is the truth times a positive
    # diagonal, so its nearest rotation is the truth. x's median is 1.0 where its mean would be 1.05.
    found = P.Series(
        "cam",
        FRAME,
        (
            seen(T, (0.9, 2.0, 1.5), TRUE @ rx(3)),
            seen(T + DAY, (1.0, 1.9, 1.4), TRUE @ rx(-3)),
            seen(T + 2 * DAY, (1.0, 2.1, 1.5), TRUE @ rz(4)),
            seen(T + 3 * DAY, (1.3, 2.0, 1.6), TRUE @ rz(-4), source="walk"),
        ),
    )
    got = found.at(T + 3 * DAY)
    assert got is not None
    assert got[:3, :3] == pytest.approx(TRUE, abs=1e-9)
    assert got[:3, 3] == pytest.approx([1.0, 2.0, 1.5], abs=1e-9)
    assert got[3] == pytest.approx([0.0, 0.0, 0.0, 1.0])


@pytest.mark.parametrize(
    ("after", "sizes"),
    [
        pytest.param(pose((1.3, 2.0, 1.5)), [5], id="shifted-30cm-stays"),
        pytest.param(pose((2.0, 2.0, 1.5)), [3, 2], id="shifted-1m-moved"),
        pytest.param(pose((1.0, 2.0, 1.5), TRUE @ rz(5)), [5], id="turned-5deg-stays"),
        pytest.param(pose((1.0, 2.0, 1.5), TRUE @ rz(15)), [3, 2], id="turned-15deg-moved"),
    ],
)
def test_a_seat_far_enough_from_its_stretch_starts_a_new_one(after, sizes):
    before = [seen(T + i * DAY, (1.0, 2.0, 1.5)) for i in range(3)]
    later = [seen(T + i * DAY, after[:3, 3], after[:3, :3]) for i in (3, 4)]
    assert [len(s) for s in P.stretches([*before, *later])] == sizes


MOVED = P.Series(
    "cam",
    FRAME,
    (
        seen(T, (1.0, 2.0, 1.5)),
        seen(T + DAY, (1.0, 2.0, 1.5)),
        seen(T + 2 * DAY, (2.0, 2.0, 1.5), TRUE @ rz(20)),
        seen(T + 3 * DAY, (2.0, 2.0, 1.5), TRUE @ rz(20)),
    ),
)


@pytest.mark.parametrize(
    ("t", "xyz", "rotation"),
    [
        pytest.param(T - 10 * DAY, (1.0, 2.0, 1.5), TRUE, id="before-the-first-reads-the-first"),
        pytest.param(T + 1.5 * DAY, (1.0, 2.0, 1.5), TRUE, id="before-the-move"),
        pytest.param(T + 2 * DAY, (2.0, 2.0, 1.5), TRUE @ rz(20), id="at-the-move"),
        pytest.param(T + 30 * DAY, (2.0, 2.0, 1.5), TRUE @ rz(20), id="after-the-last-reads-the-last"),
    ],
)
def test_a_moment_reads_the_stretch_it_falls_in(t, xyz, rotation):
    got = MOVED.at(t)
    assert got is not None
    assert got[:3, 3] == pytest.approx(xyz, abs=1e-9)
    assert got[:3, :3] == pytest.approx(rotation, abs=1e-9)


def test_the_owners_spot_places_its_stretch_without_turning_or_splitting_it():
    # 80 cm from the seats: a seat that far would be a move, the owner's tape is not
    found = [
        seen(T, (1.0, 2.0, 1.5)),
        owner(T + 0.5 * DAY, (1.8, 2.0, 1.5)),
        seen(T + DAY, (1.0, 2.0, 1.5)),
        seen(T + 2 * DAY, (1.0, 2.0, 1.5)),
    ]
    assert [len(s) for s in P.stretches(found)] == [4]
    got = P.Series("cam", FRAME, tuple(found)).at(T + 2 * DAY)
    assert got is not None
    assert got[:3, 3] == pytest.approx([1.8, 2.0, 1.5], abs=1e-9)
    assert got[:3, :3] == pytest.approx(TRUE, abs=1e-9)


def test_the_latest_owner_spot_in_a_stretch_holds():
    found = [seen(T, (1.0, 2.0, 1.5)), owner(T + DAY, (1.2, 2.0, 1.5)), owner(T + 2 * DAY, (1.1, 2.0, 1.5))]
    got = P.smoothed(found)
    assert got is not None
    assert got[:3, 3] == pytest.approx([1.1, 2.0, 1.5], abs=1e-9)


@pytest.mark.parametrize(
    "found",
    [pytest.param((), id="none"), pytest.param((owner(T, (1.0, 2.0, 1.5)),), id="only-the-owner")],
)
def test_a_series_with_no_rotation_has_no_pose(found):
    assert P.Series("cam", FRAME, found).at(T) is None


def test_recorded_observations_read_back_per_camera_in_time_order(tmp_path, store):
    later = P.Observation(T + 60, "cam-a", "walk", HOME, None, pose((1.0, 2.0, 1.5), TRUE @ rz(10)), 0.5)
    P.record(tmp_path, [later, seen(T, (1.0, 2.0, 1.5), camera="cam-a"), seen(T + 30, (4.0, 0.0, 2.0), camera="cam-b")])
    got = P.read(tmp_path, T - 1, T + 3600, ["cam-a"])
    assert [(o.t, o.camera, o.source, o.place, o.frame) for o in got] == [
        (T, "cam-a", "seat", HOME, FRAME),
        (T + 60, "cam-a", "walk", HOME, None),
    ]
    assert got[1].pose == pytest.approx(later.pose, abs=1e-6)  # stored as float32
    assert [o.quality for o in got] == pytest.approx([0.9, 0.5])


def test_a_series_keeps_only_its_place_and_frame(tmp_path, store):
    P.record(
        tmp_path,
        [
            seen(T, (1.0, 2.0, 1.5)),
            seen(T + DAY, (5.0, 5.0, 1.0), frame="frame-0"),
            seen(T + 2 * DAY, (9.0, 9.0, 1.0), place=AWAY),
            seen(T, (3.0, 3.0, 2.0), camera="unasked"),
        ],
    )
    found = P.series(tmp_path, HOME, FRAME, ["cam", "unseen"])
    assert list(found) == ["cam"]
    got = P.poses_at(found, "cam", T + 3 * DAY)
    assert got is not None
    assert got[:3, 3] == pytest.approx([1.0, 2.0, 1.5], abs=1e-6)
