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
    assert placed is not None and placed.name == "71de12f9"
    there = fuse.apply(placed.transform, piece.points)
    assert there[:, :2].min(axis=0) == pytest.approx([20.0, 5.0], abs=0.05)
    assert there[:, :2].max(axis=0) == pytest.approx([24.0, 8.0], abs=0.05)
    assert placed.after.median < 0.03


def test_a_walk_that_doesnt_fit_the_owners_plan_is_not_placed(tmp_path, home):
    assert rp.place_folder(tmp_path, home.place, "71de12f9", walk_piece(box_room(9, 9)), pi.parse(BOX), LEVEL) is None


def test_two_walks_looks_keep_names_of_their_own():
    look = places.CameraPose(tuple(float(v) for v in np.eye(4).ravel()), 300.0, 640, 360, T)
    taken: set[str] = set()
    first = rp.distinct({"walk-0m00s": look, "walk-0m30s": look}, taken)
    second = rp.distinct({"walk-0m00s": look, "walk-1m00s": look}, taken)
    assert sorted(first) == ["walk-0m00s", "walk-0m30s"]
    assert sorted(second) == ["walk-0m00sb", "walk-1m00s"]
