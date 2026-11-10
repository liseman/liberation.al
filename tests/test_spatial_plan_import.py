"""A plan as the fitting code reads it, registered onto a synthetic map's wall band with a known move, and moved
into the map's frame. Every plan here is made up, written in the tests' short notation (tests/plans.py)."""

import math

import numpy as np
import pytest

import plans
from liberational.spatial import plan_import as pi

SAMPLE = """plan "Somewhere" walls 0.2/0.1
room den "Den" living rect 0,0 4x3
room study "Study" office poly 4,0 7,0 7,5 4,5
room nook "Nook" other rect 0,3 4x2
door den>study at 4,1.5 w0.8
window exterior>den at 2,0 w1.2
cased den>nook at 2,3 w1
x-camera cam1 at 0.3,0.3 height 2.1 yaw 135 fov 90
x-camera cam2 at 6.7,4.7 height 1.5
"""


def parse(notation: str) -> pi.Parsed:
    """A plan written in the tests' notation, stored as a scene and read back as the fitting code takes it."""
    return pi.parse(plans.scene(notation))


def test_the_sample_reads_as_its_rooms_openings_and_camera_spots():
    plan = parse(SAMPLE)
    assert (plan.title, plan.walls) == ("Somewhere", pytest.approx((0.2, 0.1)))
    assert [(r.id, r.name, r.kind) for r in plan.rooms] == [
        ("den", "Den", "living"),
        ("study", "Study", "office"),
        ("nook", "Nook", "other"),
    ]
    assert plan.rooms[0].polygon == ((0, 0), (4, 0), (4, 3), (0, 3))
    assert plan.rooms[1].polygon == ((4, 0), (7, 0), (7, 5), (4, 5))
    assert sorted((o.kind, frozenset(o.between.split(">")), o.at, o.width) for o in plan.openings) == sorted(
        [
            ("door", frozenset({"den", "study"}), pytest.approx((4, 1.5)), 0.8),
            ("window", frozenset({"exterior", "den"}), pytest.approx((2, 0)), 1.2),
            ("cased", frozenset({"den", "nook"}), pytest.approx((2, 3)), 1.0),
        ],
        key=lambda o: o[0],
    )
    assert plan.cameras == (pi.Camera("cam1", (0.3, 0.3), 2.1, 135.0, 90.0), pi.Camera("cam2", (6.7, 4.7), 1.5))


def test_a_plan_with_no_rooms_is_refused():
    with pytest.raises(pi.PlanError, match="no rooms"):
        parse('plan "P"\n')


def _segments(w: pi.Walls) -> dict[tuple[tuple[float, float], tuple[float, float]], float]:
    return {(tuple(a), tuple(b)): h for a, b, h in zip(w.a.tolist(), w.b.tolist(), w.half.tolist(), strict=True)}


def test_a_shared_stretch_of_wall_is_one_partition_and_the_rest_exterior():
    plan = parse('plan "P" walls 0.2/0.1\nroom a "A" other rect 0,0 4x3\nroom b "B" other rect 4,0 2x5\n')
    segs = _segments(pi.walls(plan))
    assert segs[((4.0, 0.0), (4.0, 3.0))] == pytest.approx(0.05), "shared with a: a partition, counted once"
    assert segs[((4.0, 3.0), (4.0, 5.0))] == pytest.approx(0.1), "b's edge past a's corner is exterior"
    assert segs[((0.0, 0.0), (4.0, 0.0))] == pytest.approx(0.1)
    assert len(segs) == 8  # a's 3 outer sides, the shared piece, b's 3 outer sides, and its edge above a


# A synthetic map: both faces of every wall sampled every 2 cm, 1 cm of noise, and clutter in the rooms (furniture
# in the band), moved by a known move into the map's frame.

TRUE = pi.Move(math.radians(37.0), (3.0, -2.0), mirror=True)


def _band(plan: pi.Parsed, move: pi.Move, scale: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = pi.walls(plan)
    out = []
    for a, b, half in zip(w.a, w.b, w.half, strict=True):
        length = float(np.hypot(*(b - a)))
        t = np.linspace(0.0, 1.0, max(int(length / 0.02), 2))[:, None]
        along = a + t * (b - a)
        normal = np.array([-(b - a)[1], (b - a)[0]]) / length
        out += [along + normal * half, along - normal * half]
    pts = np.vstack(out) + rng.normal(0.0, 0.01, (sum(len(o) for o in out), 2))
    lo, hi = np.min(w.a, axis=0), np.max(w.b, axis=0)
    clutter = rng.uniform(lo, hi, (len(pts) // 10, 2))
    return move.points(np.vstack([pts, clutter]) * scale)


def _corner_error(found: pi.Move, plan: pi.Parsed, true: pi.Move = TRUE) -> float:
    corners = np.array([p for r in plan.rooms for p in r.polygon], dtype=np.float64)
    return float(np.max(np.hypot(*(found.points(corners) - true.points(corners)).T)))


def test_a_map_of_the_plan_moved_by_a_known_move_registers_back_to_it():
    plan = parse(SAMPLE)
    cams = {"cam1": TRUE.point(0.5, 0.5), "cam2": TRUE.point(6.5, 4.5)}
    fit = pi.register(plan, _band(plan, TRUE), cams, {"cam1": "den", "cam2": "study"})
    assert fit.move.mirror is True, "both mirrors tried; the right one fits"
    assert _corner_error(fit.move, plan) < 0.02
    assert fit.near_share > 0.85 and fit.outside == []
    assert [(c.camera, c.room) for c in fit.cameras] == [("cam1", "den"), ("cam2", "study")]
    assert fit.scale == pytest.approx(1.0, abs=0.005)


def test_the_room_a_camera_sees_settles_a_plan_its_walls_cant():
    # One rectangle split down the middle: half a turn fits the walls as well, and only the cameras tell which.
    plan = parse('plan "P" walls 0.2/0.1\nroom a "A" other rect 0,0 3x4\nroom b "B" other rect 3,0 3x4\n')
    true = pi.Move(math.radians(20.0), (1.0, 1.0), mirror=False)
    cams = {"door": true.point(1.0, 2.0)}  # in a
    fit = pi.register(plan, _band(plan, true), cams, {"door": "a"}, mirror=False)
    assert _corner_error(fit.move, plan, true) < 0.02
    flipped = pi.register(plan, _band(plan, true), cams, {"door": "b"}, mirror=False)
    assert flipped.cameras[0].room == "b"
    assert math.degrees(abs(flipped.move.turn - fit.move.turn)) % 360 == pytest.approx(180.0, abs=1.0)


def test_cameras_that_cant_all_be_in_their_rooms_are_refused():
    rooms = 'room a "A" other rect 0,0 3x4\nroom hall "Hall" other rect 3,0 2x4\nroom b "B" other rect 5,0 3x4\n'
    plan = parse('plan "P" walls 0.2/0.1\n' + rooms)
    cams = {"one": TRUE.point(1.0, 2.0), "two": TRUE.point(1.2, 2.0)}  # 20 cm apart: a and b are 2 m apart
    with pytest.raises(pi.PlanError, match="every camera"):
        pi.register(plan, _band(plan, TRUE), cams, {"one": "a", "two": "b"}, mirror=True)


def test_measured_camera_spots_hold_the_fit_and_report_their_residuals():
    plan = parse(SAMPLE)
    # the walls say TRUE; the cameras' registrations sit 30 cm along x from where the owner measured them
    off = pi.Move(TRUE.turn, (TRUE.shift[0] + 0.3, TRUE.shift[1]), mirror=True)
    cams = {"cam1": off.point(0.3, 0.3), "cam2": off.point(6.7, 4.7)}
    anchors = {"cam1": (0.3, 0.3), "cam2": (6.7, 4.7)}
    held = pi.register(plan, _band(plan, TRUE), cams, anchors=anchors, anchor_share=1.0)
    assert held.seeded_by == "anchors"
    assert [round(c.anchor_error or 0, 3) for c in held.cameras] == [0.0, 0.0]
    assert _corner_error(held.move, plan, off) < 0.01
    walls = pi.register(plan, _band(plan, TRUE), cams, anchors=anchors, anchor_share=0.0)
    assert _corner_error(walls.move, plan) < 0.02, "a share of 0 only measures the spots: the walls fit"
    assert all(c.anchor_error == pytest.approx(0.3, abs=0.03) for c in walls.cameras)


def test_a_camera_far_from_its_measured_spot_is_left_out_of_the_fit_and_named():
    plan = parse(SAMPLE)
    anchors = {"a": (0.5, 0.5), "b": (6.5, 0.5), "c": (6.5, 4.5), "lost": (1.0, 4.0)}
    cams = {c: TRUE.point(*at) for c, at in anchors.items()}
    cams["lost"] = TRUE.point(5.0, 1.0)  # its registration put it 4 m from where it is
    fit = pi.register(plan, _band(plan, TRUE), cams, anchors=anchors, mirror=True)
    assert fit.dropped == ("lost",)
    assert _corner_error(fit.move, plan) < 0.02
    errors = {c.camera: c.anchor_error for c in fit.cameras}
    assert errors["lost"] == pytest.approx(math.dist((5.0, 1.0), (1.0, 4.0)), abs=0.03)
    assert max(errors[c] or 0 for c in "abc") < 0.02


def test_a_map_at_another_scale_is_reported_not_fitted():
    plan = parse(SAMPLE)
    fit = pi.register(plan, _band(plan, TRUE, scale=1.03), {}, mirror=True)
    assert fit.scale == pytest.approx(1.03, abs=0.005)


def test_the_wall_error_of_another_map_is_measured_on_the_move_given():
    plan = parse('plan "P" walls 0.1/0.1\nroom a "A" other rect 0,0 4x3\nroom b "B" other rect 4,0 2x5\n')
    median, p90, n = pi.wall_error(plan, TRUE, _band(plan, TRUE))
    assert median < 0.015, "the band lies on the walls' faces, half a thickness from their centrelines"
    assert p90 < 0.1 and n > 1000
    assert pi.wall_error(plan, TRUE, np.zeros((0, 2)))[2] == 0


def test_the_plan_is_moved_into_the_target_frame():
    quarter = pi.Move(math.pi / 2, (10.0, 0.0), mirror=False)  # (x, y) to (10 - y, x)
    again = pi.parse(pi.moved(plans.scene(SAMPLE), quarter))
    assert again.rooms[0].polygon == ((10, 0), (10, 4), (7, 4), (7, 0))
    door = next(o for o in again.openings if o.kind == "door")
    assert (frozenset(door.between.split(">")), door.at) == (frozenset({"den", "study"}), pytest.approx((8.5, 4)))
    # cam1 looked south-east (135); a quarter turn from +x toward +y is clockwise on a page whose y grows down: it
    # looks south-west
    assert again.cameras[0] == pi.Camera("cam1", (9.7, 0.3), 2.1, 225.0, 90.0)


def test_a_mirrored_plan_keeps_its_rooms_and_turns_a_bearing_over():
    mirror = pi.Move(0.0, (0.0, 0.0), mirror=True)  # (x, y) to (x, -y)
    again = pi.parse(pi.moved(plans.scene(SAMPLE), mirror))
    assert set(again.rooms[0].polygon) == {(0, -3), (4, -3), (4, 0), (0, 0)}
    assert again.cameras[0].yaw == 45.0, "south-east mirrored over the x axis is north-east"


@pytest.mark.parametrize(
    ("flip", "want"),
    [pytest.param(1.0, True, id="right-handed"), pytest.param(-1.0, False, id="left-handed")],
)
def test_the_page_is_mirrored_into_a_right_handed_frame(flip, want):
    transform = (1.0, 0.0, 0.0, 0.0, 0.0, flip, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    pose = tuple(np.eye(4).ravel().tolist())
    assert pi.page_is_mirrored({"cam": pose}, transform) is want
    assert pi.page_is_mirrored({}, transform) is None


def test_camera_spots_and_the_band_are_read_in_the_plan_frame():
    shifted = (1.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, -0.1)  # x + 2, height = z - 0.1
    pose = np.eye(4)
    pose[:3, 3] = (1.0, 1.0, 2.0)
    assert pi.camera_spots({"cam": tuple(pose.ravel().tolist())}, shifted) == {"cam": (3.0, 1.0, 1.9)}
    points = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 1.5], [0.0, 0.0, 2.5]])
    assert pi.band(points, shifted).tolist() == [[2.0, 0.0]]


def _rotated(points: np.ndarray, about: tuple[float, float], degrees: float) -> np.ndarray:
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    off = points - np.asarray(about)
    return off @ np.array([[c, -s], [s, c]]).T + np.asarray(about)


def test_a_cameras_view_turned_off_its_room_is_turned_back_onto_it():
    # A camera in the den's top-left corner, looking diagonally into it: the floor and the den's walls it sees. Its
    # registration has it turned 30 degrees off, so its view crosses the den's walls until turned back.
    plan = parse(SAMPLE)
    rng = np.random.default_rng(1)
    cam = (0.3, 0.3)
    floor = rng.uniform((0.2, 0.2), (3.9, 2.9), (600, 2))
    edge = np.r_[
        np.c_[np.full(150, 3.95), rng.uniform(0.0, 3.0, 150)],
        np.c_[rng.uniform(0.0, 4.0, 150), np.full(150, 2.95)],
    ]
    page = np.r_[floor, edge]
    heights = np.r_[np.zeros(len(floor)), np.full(len(edge), 1.5)]
    there = TRUE.points(page)
    at = TRUE.point(*cam)
    registered = _rotated(there, at, -30.0)  # the view as its wrong yaw has it
    view = pi.view_fit(plan, TRUE, np.c_[registered, heights], at, at)
    assert math.degrees(view.turn) == pytest.approx(30.0, abs=3.0)
    assert view.inside > 0.99
    assert view.inside_now < 0.9, "unturned, part of the view lies beyond the den's walls"
    assert view.rooms[0][0] == "den"


def test_a_view_with_no_points_says_nothing():
    view = pi.view_fit(parse(SAMPLE), TRUE, np.zeros((0, 3)), (0.0, 0.0), (0.0, 0.0))
    assert view.rooms == () and math.isnan(view.inside)


def test_a_view_is_mirrored_across_the_cameras_line_of_sight():
    points = np.array([[2.0, 1.0, 1.5], [1.0, 3.0, 0.0]])
    # a camera at (1, 1) looking along +x: a point a metre to its left (+y) goes a metre to its right, height kept
    assert pi.mirrored(points, (1.0, 1.0), (2.0, 0.0)).tolist() == [[2.0, 1.0, 1.5], [1.0, -1.0, 0.0]]


HALLS = """plan "P" walls 0.1/0.1
room a "A" other rect 0,0 3x3
room b "B" other rect 3,0 3x3
door a>b at 3,1.5 w0.8
"""


def test_a_camera_sees_through_a_door_but_not_through_a_wall():
    plan = parse(HALLS)
    sight = pi.sight_walls(plan)
    points = np.array([[5.0, 1.5], [5.0, 0.5], [2.0, 2.0], [7.0, 1.5]])
    # from (1, 1.5): straight through the door, through the wall beside it, in its own room, and outside the plan
    assert pi.in_sight(plan, sight, (1.0, 1.5), points).tolist() == [True, False, True, False]


def test_a_moved_plans_shared_wall_is_split_once_and_its_archway_seen_through():
    # A scene moved by a turn and a mirror (`moved`) still gives one partition for the shared stretch, and a cased
    # archway in it is a gap the camera sees through, while the wall beside it is not.
    raw = plans.scene(
        'plan "P" walls 0.2/0.1\nroom a "A" other rect 0,0 4x3\nroom b "B" other rect 4,0 2x5\ncased a>b at 4,1.5 w0.8\n'
    )
    moved = pi.parse(pi.moved(raw, TRUE))
    w = pi.walls(moved)
    assert len(w.a) == 8
    assert int(np.sum(np.isclose(w.half, 0.05))) == 1
    sight = pi.sight_walls(moved)
    eye = TRUE.point(2.0, 1.5)
    through_cased = TRUE.points(np.array([[5.0, 1.5], [5.0, 2.8]]))
    assert pi.in_sight(moved, sight, eye, through_cased).tolist() == [True, False]
