"""Standing a cloud upright by its walls and squaring it to the plan (spatial/cloud_align.py), on made-up rooms: a
room is its plan's outline, and a cloud is what a camera standing in it would see of its walls and floor."""

import math

import numpy as np
import pytest

from liberational.spatial import cloud_align as align
from liberational.spatial import cloud_scene as scenes

#: A room 4 m by 5 m, as its plan draws it in the scene's frame.
ROOM = ((0.0, 0.0), (4.0, 0.0), (4.0, 5.0), (0.0, 5.0))
HEIGHT = 1.5


def wall(a, b, n: int = 4000, tall: float = 2.4, seed: int = 0) -> np.ndarray:
    """Points on the wall standing on the line from `a` to `b`."""
    rng = np.random.default_rng(seed)
    along = rng.random(n)[:, None]
    foot = np.asarray(a) + along * (np.asarray(b) - np.asarray(a))
    return np.c_[foot, rng.random(n) * tall] + rng.normal(0.0, 0.004, (n, 3))


def floor(n: int = 6000, seed: int = 9) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.c_[rng.random(n) * 4.0, rng.random(n) * 5.0, rng.normal(0.0, 0.004, n)]


def far_corner() -> np.ndarray:
    """The room's far wall and its right wall, as the scene has them."""
    return np.concatenate([wall(ROOM[3], ROOM[2], seed=1), wall(ROOM[1], ROOM[2], seed=2)])


def seen_from(points: np.ndarray, camera: scenes.At) -> np.ndarray:
    """Scene points in the level frame of a cloud whose camera really stands at `camera`: the camera on the z axis,
    looking along +y."""
    move = scenes._transform(camera)
    return (points - move[:3, 3]) @ move[:3, :3]


def at(x: float, y: float, degrees: float) -> scenes.At:
    return scenes.At(x, y, math.radians(degrees))


def tipped(points: np.ndarray, tilt: float, roll: float) -> tuple[np.ndarray, np.ndarray]:
    """A levelled cloud as it is kept when its camera, really tilted `tilt` degrees down and rolled `roll`, was taken
    to be upright: the points, and the pose that says upright."""
    t, r = math.radians(tilt), math.radians(roll)
    about_x = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(t), -math.sin(t)], [0.0, math.sin(t), math.cos(t)]])
    about_y = np.array([[math.cos(r), 0.0, math.sin(r)], [0.0, 1.0, 0.0], [-math.sin(r), 0.0, math.cos(r)]])
    turn = about_y @ about_x
    eye = np.array([0.0, 0.0, HEIGHT])
    pose = np.eye(4)
    pose[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])  # right +x, down -z, ahead +y
    pose[:3, 3] = eye
    return (points - eye) @ turn.T + eye, pose


def leans(points: np.ndarray) -> list[float]:
    """How far each wall found in `points` is from vertical, in degrees."""
    return [math.degrees(math.asin(abs(float(p.normal[2])))) for p in align.planes(points) if abs(p.normal[2]) < 0.5]


# ---- the flat surfaces


def test_a_rooms_floor_and_walls_are_found_as_planes_the_largest_first():
    cloud = np.concatenate([floor(), far_corner()])
    found = align.planes(cloud)
    assert len(found) == 3
    assert abs(found[0].normal[2]) == pytest.approx(1.0, abs=1e-3)  # the floor, with the most points
    assert sorted(round(abs(float(p.normal @ axis)), 2) for p in found[1:] for axis in ([1.0, 0, 0], [0, 1.0, 0])) == [
        0.0,
        0.0,
        1.0,
        1.0,
    ]
    assert all(p.rms < 0.01 for p in found)


def test_a_strip_of_points_is_not_a_plane():
    strip = wall((0.0, 3.0), (4.0, 3.0), tall=0.2)
    assert align.planes(strip) == []


# ---- level


def test_a_cloud_with_no_floor_is_stood_upright_by_two_walls_that_meet():
    cloud, pose = tipped(seen_from(far_corner(), at(1.0, 1.0, 20.0)), tilt=10.0, roll=4.0)
    assert max(leans(cloud)) > 3.0  # as kept, a wall leans
    upright = align.stand(cloud, pose)
    assert upright is not None
    assert max(leans(align.apply(upright, cloud))) < 0.3
    tilt, roll = align.tilt_roll(upright @ pose)
    assert (tilt, abs(roll)) == (pytest.approx(10.0, abs=0.5), pytest.approx(4.0, abs=0.5))
    eye = (upright @ pose)[:3, 3]
    assert eye[:2] == pytest.approx([0.0, 0.0], abs=1e-9)  # the camera stays on the z axis
    assert (upright @ pose)[0, 2] == pytest.approx(0.0, abs=1e-9)  # looking along +y


def test_one_wall_takes_out_the_lean_across_it_only():
    cloud, pose = tipped(seen_from(wall(ROOM[3], ROOM[2]), at(2.0, 1.0, 0.0)), tilt=10.0, roll=4.0)
    upright = align.stand(cloud, pose)
    assert upright is not None
    assert max(leans(align.apply(upright, cloud))) < 0.3  # the wall ahead stands upright
    tilt, roll = align.tilt_roll(upright @ pose)
    assert tilt == pytest.approx(10.0, abs=0.7)  # the tilt is across the wall ahead: found
    assert roll == pytest.approx(0.0, abs=0.7)  # the roll runs along it: nothing says, so it stays as it was


def test_a_cloud_with_no_wall_is_left_as_it_was():
    cloud, pose = tipped(seen_from(floor(), at(2.0, 1.0, 0.0)), tilt=40.0, roll=0.0)
    assert align.stand(cloud, pose) is None


# ---- square


def squared(
    truth: scenes.At, said: scenes.At, cloud: np.ndarray | None = None, rooms=(ROOM,)
) -> tuple[align.Square, scenes.At]:
    """Square the cloud of a camera that really stands at `truth` and whose spot says `said`: what came of it, and
    where it puts the camera."""
    own = seen_from(np.concatenate([floor(), far_corner()]) if cloud is None else cloud, truth)
    found = align.square(align.lines(align.planes(own)), said.x, said.y, said.bearing, align.edges_of(rooms))
    return found, said.step(*found.rel)


def test_walls_a_few_degrees_off_the_plan_are_turned_onto_it_and_slid_to_its_walls():
    truth = at(1.0, 1.0, 20.0)
    found, put = squared(truth, at(1.2, 0.85, 24.0))
    assert (found.state, found.walls, found.slid_on, found.slide) == ("squared", 2, 2, "")
    assert found.turn_deg == pytest.approx(-4.0, abs=0.3)
    assert found.off_deg == pytest.approx(4.0, abs=0.3)
    assert (put.x, put.y) == pytest.approx((truth.x, truth.y), abs=0.03)
    assert math.degrees(put.bearing) == pytest.approx(20.0, abs=0.3)
    assert found.bend_deg < 0.5  # a rigid room: nothing is left over


def test_a_cloud_that_is_square_already_stays_where_its_spot_puts_it():
    truth = at(1.0, 1.0, 20.0)
    found, put = squared(truth, truth)
    assert found.state == "squared"
    assert (put.x, put.y, math.degrees(put.bearing)) == pytest.approx((1.0, 1.0, 20.0), abs=0.03)


def test_walls_that_only_line_up_past_the_bound_are_left_alone_and_say_so():
    truth = at(1.0, 1.0, 20.0)
    said = at(1.0, 1.0, 45.0)
    found, put = squared(truth, said)
    assert found.state == "beyond"
    assert found.turn_deg == pytest.approx(-25.0, abs=0.5)  # what it would have taken
    assert put == said  # nothing moved


def test_a_slide_past_the_bound_is_not_made_but_the_turn_is():
    truth = at(1.0, 1.0, 20.0)
    found, put = squared(truth, at(1.8, 1.0, 23.0))
    assert (found.state, found.slide, found.slid_on) == ("squared", "beyond", 0)
    assert (put.x, put.y) == pytest.approx((1.8, 1.0))
    assert math.degrees(put.bearing) == pytest.approx(20.0, abs=0.3)


def test_a_wall_that_could_be_either_of_two_plan_walls_asks_for_no_slide():
    # a second room's wall 0.6 m behind the first room's far wall: which of the two the cloud's wall is, nobody knows
    rooms = (ROOM, ((0.0, 5.6), (4.0, 5.6), (4.0, 9.0), (0.0, 9.0)))
    truth = at(2.0, 1.0, 0.0)
    found, put = squared(truth, at(2.0, 1.2, 3.0), cloud=wall(ROOM[3], ROOM[2]), rooms=rooms)
    assert (found.state, found.slide) == ("squared", "no_counterpart")
    assert (put.x, put.y) == pytest.approx((2.0, 1.2))
    assert math.degrees(put.bearing) == pytest.approx(0.0, abs=0.3)


def test_a_cloud_with_no_wall_or_a_plan_with_none_nearby_is_not_squared():
    spot = at(2.0, 1.0, 0.0)
    assert squared(spot, spot, cloud=floor())[0].state == "no_walls"
    far_away = (tuple((x + 100.0, y) for x, y in ROOM),)
    assert squared(spot, spot, rooms=far_away)[0].state == "no_plan"


def test_walls_bent_against_each_other_are_squared_by_the_bigger_and_the_rest_is_reported():
    # the right wall runs 6 degrees off square to the far wall, as a depth estimate bends a room
    bent = np.concatenate(
        [
            wall(ROOM[3], ROOM[2], n=6000, seed=1),
            wall(ROOM[1], (4.0 + 5.0 * math.tan(math.radians(6.0)), 5.0), n=2500, seed=2),
        ]
    )
    spot = at(1.0, 1.0, 20.0)
    found, _ = squared(spot, spot, cloud=bent)
    assert found.state == "squared"
    assert abs(found.turn_deg) < 0.5  # the far wall, with more points, is on the plan already
    assert found.bend_deg == pytest.approx(6.0, abs=0.5)


# ---- fit to the room


def fitted(truth: scenes.At, said: scenes.At, cloud: np.ndarray, rooms=(ROOM,), size: float = 1.0, **more):
    """Fit the cloud of a camera that really stands at `truth`, whose spot says `said` and whose cloud came out
    `size` times too big: the fit, the cloud's own points, and the move that places them."""
    own = seen_from(cloud, truth) * size
    found = align.fit(align.lines(align.planes(own)), said.x, said.y, said.bearing, align.edges_of(rooms), **more)
    stands = scenes.At(said.x + found.dx, said.y + found.dy, said.bearing + math.radians(found.turn_deg))
    move = scenes._transform(stands) @ np.diag([found.scale, found.scale, found.up, 1.0])
    return found, own, move


def bent_room() -> np.ndarray:
    """The room's floor, its far wall, and its right wall, the right wall running 4 degrees off square."""
    lean = 5.0 * math.tan(math.radians(4.0))
    return np.concatenate([floor(), wall(ROOM[3], ROOM[2], n=6000, seed=1), wall((4.0 - lean, 0.0), ROOM[2], seed=2)])


def test_a_cloud_a_tenth_too_big_with_bent_walls_ends_with_its_walls_on_the_plans():
    spot = at(1.0, 1.0, 20.0)
    found, own, move = fitted(spot, spot, bent_room(), size=1.1)
    assert (found.state, len(found.walls), found.seen) == ("fitted", 2, 2)
    assert found.scale == pytest.approx(1 / 1.1, abs=0.03)
    assert math.hypot(found.dx, found.dy) < 0.2  # the bent wall asks for a little: it is shared, not stretched
    assert max(abs(w.before_m) for w in found.walls) > 0.25  # they stood well off the plan's walls
    assert max(abs(w.after_m) for w in found.walls) < 0.08
    assert max(w.off_deg for w in found.walls) == pytest.approx(4.0, abs=0.5)  # the bend is kept as a number
    drawn, kind = align.flatten(own, move, found)
    on_wall = drawn[kind == align.WALL]
    far, right = on_wall[on_wall[:, 0] < 3.999], on_wall[on_wall[:, 0] >= 3.999]
    assert len(far) > 1000 and len(right) > 1000
    assert far[:, 1] == pytest.approx(5.0, abs=1e-9)  # exactly on the plan's far wall
    assert right[:, 0] == pytest.approx(4.0, abs=1e-9)  # and on its right wall, straight
    assert drawn[kind == align.FLOOR][:, 2] == pytest.approx(0.0, abs=1e-12)
    assert (kind == align.FLOOR).sum() > 5000


def test_what_stands_in_front_of_a_wall_stays_points_and_is_not_laid_on_it():
    cupboard = wall((1.0, 4.5), (2.5, 4.5), n=2500, tall=1.8, seed=5)  # its front, half a metre off the far wall
    spot = at(2.0, 1.0, 0.0)
    found, own, move = fitted(spot, spot, np.concatenate([bent_room(), cupboard]))
    assert found.state == "fitted" and found.seen == 3 and len(found.walls) == 2
    drawn, kind = align.flatten(own, move, found)
    front = (np.abs(drawn[:, 1] - 4.5) < 0.05) & (drawn[:, 0] > 0.9) & (drawn[:, 0] < 2.6) & (drawn[:, 2] > 0.1)
    assert front.sum() > 2000
    assert (kind[front] == align.POINT).all()


def test_with_flat_off_or_no_fit_every_point_is_where_its_cloud_has_it():
    spot = at(1.0, 1.0, 20.0)
    found, own, move = fitted(spot, spot, bent_room())
    plain, kind = align.flatten(own, move, align.Fit("no_match"))
    assert (kind == align.POINT).all()
    assert plain == pytest.approx(own @ move[:3, :3].T + move[:3, 3])
    drawn, kind = align.flatten(own, move, found)
    assert drawn[kind == align.POINT] == pytest.approx(plain[kind == align.POINT])  # only walls and floor move
    assert len(drawn) == len(own)  # nothing added, nothing dropped


def test_one_wall_gives_the_size_alone():
    truth = at(2.0, 1.0, 0.0)
    found, _, _ = fitted(truth, at(2.3, 1.0, 0.0), wall(ROOM[3], ROOM[2]), size=1.1)
    assert (found.state, len(found.walls)) == ("fitted", 1)
    assert found.scale == pytest.approx(1 / 1.1, abs=0.01)
    assert (found.dx, found.dy) == pytest.approx((0.0, 0.0), abs=0.02)  # nothing says how far across it stands


def test_walls_on_both_sides_give_the_slide_too():
    truth = at(2.0, 1.0, 0.0)
    sides = np.concatenate([wall(ROOM[0], ROOM[3], seed=1), wall(ROOM[1], ROOM[2], seed=2)])
    found, _, _ = fitted(truth, at(2.25, 1.0, 0.0), sides)
    assert (found.state, len(found.walls)) == ("fitted", 2)
    assert found.scale == pytest.approx(1.0, abs=0.02)
    assert (found.dx, found.dy) == pytest.approx((-0.25, 0.0), abs=0.03)


def test_a_fit_that_asks_for_a_slide_past_the_bound_is_not_made_and_says_what_it_would_take():
    hall = ((0.0, 0.0), (12.0, 0.0), (12.0, 5.0), (0.0, 5.0))
    sides = np.concatenate([wall(hall[0], hall[3], seed=1), wall(hall[1], hall[2], seed=2)])
    found, _, _ = fitted(at(6.0, 1.0, 0.0), at(6.7, 1.0, 0.0), sides, rooms=(hall,))
    assert found.state == "beyond"
    assert found.dx == pytest.approx(-0.7, abs=0.05)


def test_a_wall_too_far_from_the_plan_wall_the_camera_faces_is_not_that_wall():
    # an island a metre and a half short of the far wall: no resize within the bound puts it there
    spot = at(2.0, 1.0, 0.0)
    found, _, _ = fitted(spot, spot, wall((0.5, 3.5), (3.5, 3.5)))
    assert (found.state, found.seen, found.walls) == ("no_match", 1, ())
    assert fitted(spot, spot, floor())[0].state == "no_walls"


def test_a_cloud_wall_is_matched_to_the_first_plan_wall_the_camera_sees_there_not_one_behind_it():
    rooms = (ROOM, ((0.0, 5.3), (4.0, 5.3), (4.0, 9.0), (0.0, 9.0)))  # a second room behind the far wall
    spot = at(2.0, 1.0, 0.0)
    found, _, _ = fitted(spot, spot, wall(ROOM[3], ROOM[2]), rooms=rooms, size=1.06)  # it reads 4.24 m: nearer 5.3
    (matched,) = found.walls
    assert {matched.a[1], matched.b[1]} == {5.0}
    assert found.scale == pytest.approx(1 / 1.06, abs=0.01)


def test_a_measured_height_sizes_the_cloud_upwards_when_it_is_plausible():
    spot = at(2.0, 1.0, 0.0)
    cloud = wall(ROOM[3], ROOM[2])
    assert fitted(spot, spot, cloud, height=(1.8, 1.5))[0].up == pytest.approx(1.2)
    found = fitted(spot, spot, cloud, height=(3.0, 1.5))[0]  # twice the cloud's: not this camera's floor
    assert found.up == found.scale


# ---- what is kept


def test_a_squaring_is_kept_and_read_back_and_holds_only_for_its_cloud_spot_and_plan(tmp_path):
    own = seen_from(np.concatenate([floor(), far_corner()]), at(1.0, 1.0, 20.0))
    made = align.make("cloud-1", own, (1.0, 1.0, math.radians(23.0)), "spot-1", (ROOM,))
    align.keep(tmp_path, made)
    kept = align.kept(tmp_path)
    assert kept == made
    assert kept.fit is not None and kept.fit.state == "fitted" and len(kept.fit.walls) == 2
    key = align.plan_key((ROOM,))
    assert kept.holds("cloud-1", "spot-1", key)
    assert not kept.holds("cloud-2", "spot-1", key)
    assert not kept.holds("cloud-1", "spot-2", key)
    assert not kept.holds("cloud-1", "spot-1", align.plan_key((ROOM[:3],)))


def test_a_record_kept_before_clouds_were_fitted_no_longer_holds_so_it_is_made_again(tmp_path):
    import json

    own = seen_from(np.concatenate([floor(), far_corner()]), at(1.0, 1.0, 20.0))
    made = align.make("cloud-1", own, (1.0, 1.0, math.radians(23.0)), "spot-1", (ROOM,))
    align.keep(tmp_path, made)
    record = json.loads((tmp_path / align.ALIGN).read_text())
    del record["fit"]
    (tmp_path / align.ALIGN).write_text(json.dumps(record))
    kept = align.kept(tmp_path)
    assert kept is not None and kept.fit is None
    assert not kept.holds("cloud-1", "spot-1", align.plan_key((ROOM,)))


def test_a_record_that_doesnt_read_is_no_squaring(tmp_path):
    assert align.kept(tmp_path) is None
    (tmp_path / align.ALIGN).write_text('{"cloud": "c"}')
    assert align.kept(tmp_path) is None
