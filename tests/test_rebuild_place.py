"""tools/rebuild_place.py: the cameras put on the owner's measured spots."""

import importlib.util
import math
import pathlib
import sys

import numpy as np
import pytest

from liberational.memory import places
from liberational.spatial import cameras, fuse, ply
from liberational.spatial import plan_import as pi

_spec = importlib.util.spec_from_file_location(
    "rebuild_place", pathlib.Path(__file__).resolve().parents[1] / "tools" / "rebuild_place.py"
)
assert _spec is not None and _spec.loader is not None
rp = importlib.util.module_from_spec(_spec)
sys.modules["rebuild_place"] = rp  # its dataclasses look their module up
_spec.loader.exec_module(rp)

LEVEL = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)  # plan = map on a levelled floor


def facing(heading_deg, x=1.0, y=1.0, z=1.5):
    """A camera at (x, y, z) looking along `heading_deg` (counterclockwise from +x) and 30 degrees down."""
    h, p = math.radians(heading_deg), math.radians(30.0)
    forward = np.array([math.cos(p) * math.cos(h), math.cos(p) * math.sin(h), -math.sin(p)])
    right = np.array([math.sin(h), -math.cos(h), 0.0])
    m = np.eye(4)
    m[:3, :3] = np.c_[right, np.cross(forward, right), forward]
    m[:3, 3] = (x, y, z)
    return places.CameraPose(tuple(float(v) for v in m.ravel()), 300.0, 640, 360)


def revision(cams):
    return places.Revision(
        "0" * 8 + "-0000-7000-8000-" + "0" * 12, "1" * 8 + "-0000-7000-8000-" + "0" * 12, 0.0, "t", (), cameras=cams
    )


def test_a_camera_goes_to_its_measured_spot_and_height_turned_to_the_facing_given():
    plan = pi.parse(
        'room r "Den" other poly 0,0 5,0 5,5 0,5\n'
        "x-camera a at 2,3 height 2.0 yaw 90\n"  # a page bearing: 90 is +x (the page's y grows down)
        "x-camera b at 4,1\n"
    )
    cams = {"a": facing(30.0), "b": facing(200.0, 3.0, 1.0, 1.2), "walk-0m00s": facing(0.0), "c": facing(10.0)}
    got = rp.spots(revision(cams), plan, LEVEL)
    a = np.asarray(got["a"].pose).reshape(4, 4)
    assert a[:3, 3] == pytest.approx([2.0, 3.0, 2.0])
    assert math.degrees(math.atan2(a[1, 2], a[0, 2])) == pytest.approx(0.0, abs=1e-6), "turned to face +x"
    assert -a[2, 2] == pytest.approx(math.sin(math.radians(30.0))), "its tilt kept"
    b = np.asarray(got["b"].pose).reshape(4, 4)
    assert b[:3, 3] == pytest.approx([4.0, 1.0, 1.2]), "no height given: its own kept"
    assert math.degrees(math.atan2(b[1, 2], b[0, 2])) == pytest.approx(-160.0), "no facing given: its own kept"
    assert got["c"] == cams["c"], "no spot: as registered"
    assert "walk-0m00s" not in got


T = 1_800_000_000.0
BOX = 'plan "Home" walls 0.02/0.02\nroom r1 "Box" other poly 20,5 24,5 24,8 20,8\n'


def box_room(width, depth):
    """A levelled room's walls and floor (z up, floor at z = 0), `width` by `depth` metres and 2.6 m high."""
    rng = np.random.default_rng(0)
    n = 6000
    x = np.r_[rng.uniform(0, width, 2 * n), np.zeros(n), np.full(n, float(width))]
    y = np.r_[np.zeros(n), np.full(n, float(depth)), rng.uniform(0, depth, 2 * n)]
    walls = np.c_[x, y, rng.uniform(0, 2.6, 4 * n)]
    floor = np.c_[rng.uniform(0, width, n), rng.uniform(0, depth, n), np.zeros(n)]
    return np.r_[walls, floor]


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A place whose fused map is a far-off yard, with the owner's plan of a 4 x 3 m room drawn on its frame."""
    monkeypatch.setattr(cameras, "layout", lambda *a, **k: cameras.Layout(None, None, fuse.LEVEL_FLOOR))
    yard = box_room(30, 20) * [1.0, 1.0, 0.15] - [45.0, 10.0, 0.0]
    rev = places.revise(
        tmp_path,
        None,
        producer="cams",
        map_bytes=ply.dumps(yard, np.full((len(yard), 3), 90, np.uint8)),
        map_format="ply",
        floor=fuse.LEVEL_FLOOR,
        now=T,
    )
    fused = fuse.refine(tmp_path, rev.place, now=T + 1)
    assert fused is not None
    places.correct_plan(tmp_path, fused.place, BOX, now=T + 2, base=fused.id)
    return fused


def walk_piece(room):
    """A walk's cloud in a frame of its own: `room` turned 30 degrees and shifted, on a level floor."""
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    own = room @ np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]) + [3.0, -7.0, 0.0]
    look = places.CameraPose(tuple(float(v) for v in np.eye(4).ravel()), 300.0, 640, 360, T)
    return fuse.Piece(own.astype(np.float32), None, fuse.LEVEL_FLOOR, {"walk-0m00s": look})


def test_a_walk_read_from_its_folder_is_placed_where_the_owners_plan_has_its_room(tmp_path, home):
    piece = walk_piece(box_room(4, 3))
    placed = rp.place_folder(tmp_path, home.place, "71de12f9", piece, pi.parse(BOX), LEVEL)
    assert not isinstance(placed, str) and placed.name == "71de12f9"
    there = fuse.apply(placed.transform, piece.points)
    assert there[:, :2].min(axis=0) == pytest.approx([20.0, 5.0], abs=0.05)
    assert there[:, :2].max(axis=0) == pytest.approx([24.0, 8.0], abs=0.05)
    assert placed.after.median < 0.03


def test_a_walk_that_doesnt_fit_the_owners_plan_is_not_placed(tmp_path, home):
    refused = rp.place_folder(tmp_path, home.place, "71de12f9", walk_piece(box_room(9, 9)), pi.parse(BOX), LEVEL)
    assert isinstance(refused, str) and refused.endswith("not a fit")


def test_two_walks_looks_keep_names_of_their_own():
    look = places.CameraPose(tuple(float(v) for v in np.eye(4).ravel()), 300.0, 640, 360, T)
    taken: set[str] = set()
    first = rp.distinct({"walk-0m00s": look, "walk-0m30s": look}, taken)
    second = rp.distinct({"walk-0m00s": look, "walk-1m00s": look}, taken)
    assert sorted(first) == ["walk-0m00s", "walk-0m30s"]
    assert sorted(second) == ["walk-0m00sb", "walk-1m00s"]


def sketched(room, share, seed=3):
    """`room` with clutter the plan doesn't draw: wall-height points inside it, away from its walls, making up `share`
    of its wall band, as a walk through rooms the owner only sketched has."""
    rng = np.random.default_rng(seed)
    walls = int((room[:, 2] > 0.3).sum())
    n = int(walls * share / (1 - share))
    inner = np.c_[rng.uniform(1.0, 3.0, n), rng.uniform(0.8, 2.2, n), rng.uniform(0.4, 1.9, n)]
    return np.r_[room, inner]


def in_the_box(points):
    """Where the plan has the room: the made-up room moved to the box at 20,5."""
    return points + [20.0, 5.0, 0.0]


def test_a_near_miss_is_a_fit_short_only_of_wall_share_and_not_by_much():
    def fit(median, near, inside):
        return fuse.PlanFit(np.eye(4), median, near, inside, "")

    assert fit(0.057, 0.59, 0.97).near_miss and not fit(0.057, 0.59, 0.97).ok
    assert fit(0.03, 0.82, 1.0).ok and not fit(0.03, 0.82, 1.0).near_miss
    assert not fit(0.057, 0.49, 0.97).near_miss, "too few walls near the plan's"
    assert not fit(0.07, 0.59, 0.97).near_miss, "the median misses"
    assert not fit(0.057, 0.59, 0.85).near_miss, "the floor misses"


def test_a_near_miss_walk_is_placed_when_a_walk_already_placed_agrees(tmp_path, home):
    piece = walk_piece(sketched(box_room(4, 3), 0.36))
    plan = pi.parse(BOX)
    fit = fuse.plan_fit(tmp_path, home.place, piece.points)
    assert fit is not None and fit.near_miss  # about 54% of its walls are near the plan's
    alone = rp.place_folder(tmp_path, home.place, "second", piece, plan, LEVEL)
    assert isinstance(alone, str) and "no walk placed yet" in alone, "not on the plan's word alone"
    earlier = in_the_box(box_room(4, 3))  # the first walk, placed: the same room, as LiDAR saw it
    placed = rp.place_folder(tmp_path, home.place, "second", piece, plan, LEVEL, earlier)
    assert not isinstance(placed, str) and "a near miss" in placed.found and "agree" in placed.found
    there = fuse.apply(placed.transform, piece.points)
    assert there[:, :2].min(axis=0) == pytest.approx([20.0, 5.0], abs=0.05)
    assert there[:, :2].max(axis=0) == pytest.approx([24.0, 8.0], abs=0.05)


def test_a_near_miss_walk_is_refused_when_the_walks_already_placed_dont_agree(tmp_path, home):
    piece = walk_piece(sketched(box_room(4, 3), 0.36))
    plan = pi.parse(BOX)
    fit = fuse.plan_fit(tmp_path, home.place, piece.points)
    assert fit is not None and fit.near_miss
    elsewhere = in_the_box(box_room(4, 3)) + [40.0, 0.0, 0.0]  # nothing where the plan put the walk
    nothing_there = rp.place_folder(tmp_path, home.place, "second", piece, plan, LEVEL, elsewhere)
    assert isinstance(nothing_there, str) and "don't agree" in nothing_there
    other = in_the_box(box_room(9, 9) * [0.3, 0.3, 1.0] + [0.6, 0.2, 0.0])  # something there, but not this room
    another_room = rp.place_folder(tmp_path, home.place, "second", piece, plan, LEVEL, other)
    assert isinstance(another_room, str)


def test_a_walk_that_misses_the_median_is_refused_whatever_else_is_placed(tmp_path, home):
    wrong = walk_piece(box_room(9, 9))
    refused = rp.place_folder(tmp_path, home.place, "x", wrong, pi.parse(BOX), LEVEL, in_the_box(box_room(4, 3)))
    assert isinstance(refused, str) and refused.endswith("not a fit")


def test_the_walks_the_plan_places_go_first_whatever_order_they_were_given_in(tmp_path, home):
    near_miss, good = walk_piece(sketched(box_room(4, 3), 0.36)), walk_piece(box_room(4, 3))
    placed, refused = rp.place_folders(
        tmp_path, home.place, [("second", near_miss), ("first", good)], pi.parse(BOX), LEVEL
    )
    assert [f.name for f in placed] == ["second", "first"] and refused == {}
    alone, refused = rp.place_folders(tmp_path, home.place, [("second", near_miss)], pi.parse(BOX), LEVEL)
    assert alone == [] and list(refused) == ["second"] and "no walk placed yet" in refused["second"]


def export(folder):
    folder.mkdir()
    (folder / "rgb.mp4").write_bytes(b"video")
    (folder / "odometry.csv").write_text("timestamp, frame\n")
    return folder


def test_a_walk_read_once_isnt_read_again_until_its_recording_changes(tmp_path):
    folder = export(tmp_path / "71de12f9")
    look = places.CameraPose(tuple(float(v) for v in np.arange(16)), 300.0, 640, 360)
    reads = []

    def read(path):
        reads.append(path)
        return np.ones((5, 3), np.float32), np.full((5, 3), 7, np.uint8), (0.0, 0.0, 1.0, 0.0), {"walk-0m00s": look}

    first, cached = rp.read_piece(folder, tmp_path / "cache", read)
    again, cached_again = rp.read_piece(folder, tmp_path / "cache", read)
    assert (cached, cached_again, reads) == (False, True, [folder])
    assert again.points.tolist() == first.points.tolist() and again.colors.tolist() == first.colors.tolist()
    assert tuple(again.floor) == (0.0, 0.0, 1.0, 0.0) and again.cameras == {"walk-0m00s": look}
    (folder / "odometry.csv").write_text("timestamp, frame\n0, 0\n")  # the recording changed
    rp.read_piece(folder, tmp_path / "cache", read)
    rp.read_piece(folder, None, read)  # and with no cache it is read every time
    assert len(reads) == 3
