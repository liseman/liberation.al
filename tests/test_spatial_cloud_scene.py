"""A place's scene of its cameras' clouds (spatial/cloud_scene.py): placed from their spots, tightened where two
cameras' pictures match, and following a camera that was turned or carried. The model never runs: a match is made
up from where the two cameras really stand, and a cloud from what a camera standing in made-up rooms would see."""

import math

import numpy as np
import pytest

from liberational.spatial import camera_clouds as clouds
from liberational.spatial import cloud_align, ply
from liberational.spatial import cloud_scene as scenes
from liberational.spatial import pointmap_runner as runner

T = 1_762_500_000.0
HEIGHT = 1.5
#: A camera hung upright at HEIGHT, in its cloud's level frame: its right is +x, its forward +y, its down -z.
UPRIGHT = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, HEIGHT, 0.0, 0.0, 0.0, 1.0)
PLACE = "11111111-1111-7111-8111-111111111111"


def at(x: float, y: float, degrees: float) -> scenes.At:
    return scenes.At(x, y, math.radians(degrees))


def spot(camera: str, where: scenes.At | None, place: str = PLACE, basis: str | None = None) -> scenes.SpotAt:
    plan = (0.0, 0.0) if where is None else (where.x, -where.y)
    return scenes.SpotAt(camera, place, where, plan, False, basis or f"{camera}@{where}")


def seen(
    other: scenes.At, mine: scenes.At, sure=(0.6, 0.25), found=(0.22, 0.11), disagree=(3.0, 0.15, 0.03), scale=1.0
):
    """What the runner would say of two cameras that really stand at `mine` and `other`: `other` as seen from
    `mine`, in `mine`'s camera frame (x right, y down, z forward)."""
    right, ahead, turn = other.seen_from(mine)
    s, c = math.sin(turn), math.cos(turn)
    return {
        "match": list(found),
        "sure": list(sure),
        "rotation": [c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c],  # the other camera's right, down, and forward
        "shift": [right, 0.0, ahead],
        "scale": scale,
        "disagree": list(disagree),
    }


APART = {"match": [0.0, 0.05], "sure": [0.0, 0.0]}  # two pictures of different space


@pytest.fixture
def store(tmp_path):
    return clouds.Store(clouds.root(tmp_path))


def keep(store, camera: str, n: int = 1, points=None) -> clouds.Cloud:
    """Keep a made-up cloud for `camera` (its `n`th), with the picture it was made from."""
    cloud = clouds.Cloud(
        id=f"00000000-0000-7000-8000-{abs(hash(camera)) % 10**8:08d}{n:04d}", camera=camera, at=T + n, points=3,
        share=0.8, size="approximate", floor_found=True, pose=UPRIGHT, focal=300.0, width=64, height=36,
    )  # fmt: skip
    own = np.array([[0.0, 1.0, 0.0], [0.5, 2.0, 0.0], [0.0, 2.0, 1.0]]) if points is None else points
    store.replace(
        clouds.Made(cloud, ply.dumps(own, np.full((len(own), 3), 100, np.uint8)), np.zeros((36, 64, 3), np.uint8))
    )
    return cloud


def drawn(store, found: scenes.Scene) -> np.ndarray:
    """The scene's points as the 3D view gets them."""
    rows = ply.vertices(scenes.points(store, found))
    assert rows is not None
    return rows


def placed(found: scenes.Scene, camera: str) -> scenes.Placed:
    return next(p for p in found.cameras if p.camera == camera)


def stands(found: scenes.Scene, camera: str) -> scenes.At:
    """Where `camera` is placed: its position on the floor and its bearing, read back from its pose in the scene."""
    pose = np.array(placed(found, camera).pose).reshape(4, 4)
    return scenes.At(float(pose[0, 3]), float(pose[1, 3]), float(math.atan2(pose[0, 2], pose[1, 2])))


def xyb(found: scenes.At) -> tuple[float, float, float]:
    return found.x, found.y, found.bearing


def where(spot: scenes.SpotAt) -> scenes.At:
    assert spot.at is not None
    return spot.at


# ---- 1. place


def test_each_cameras_cloud_is_put_on_its_spot_facing_its_facing(store):
    spots = {"hall": spot("hall", at(3.0, 2.0, 90.0)), "porch": spot("porch", at(-1.0, 0.0, 0.0))}
    keep(store, "hall")
    keep(store, "porch")
    found = scenes.scene(store, PLACE, spots)
    assert [(p.camera, p.how, p.placed) for p in found.cameras] == [("hall", "spot", True), ("porch", "spot", True)]
    # the hall camera faces +x: a point 1 m ahead of it in its own cloud is 1 m east of its spot, on the floor
    move = np.array(placed(found, "hall").transform).reshape(4, 4)
    assert move @ [0.0, 1.0, 0.0, 1.0] == pytest.approx([4.0, 2.0, 0.0, 1.0], abs=1e-9)
    assert np.array(placed(found, "hall").pose).reshape(4, 4)[:3, 3] == pytest.approx([3.0, 2.0, HEIGHT])


def test_a_camera_with_no_spot_or_no_facing_is_not_in_the_scene_and_says_why(store):
    keep(store, "hall")
    keep(store, "porch")
    found = scenes.scene(store, PLACE, {"porch": spot("porch", None)})
    assert [(p.camera, p.how, p.placed) for p in found.cameras] == [
        ("hall", "no_spot", False),
        ("porch", "no_facing", False),
    ]
    assert len(drawn(store, found)) == 0


def test_a_camera_with_a_spot_on_another_places_plan_is_in_that_places_scene_only(store):
    keep(store, "hall")
    spots = {"hall": spot("hall", at(0.0, 0.0, 0.0), place="22222222-2222-7222-8222-222222222222")}
    assert scenes.scene(store, PLACE, spots).cameras == ()


def test_the_scenes_points_are_each_clouds_own_moved_to_its_place_and_marked_with_whose_they_are(store):
    spots = {"hall": spot("hall", at(3.0, 2.0, 90.0)), "porch": spot("porch", at(0.0, 0.0, 0.0))}
    keep(store, "hall", points=np.array([[0.0, 1.0, 0.0]]))
    keep(store, "porch", points=np.array([[0.0, 2.0, 0.5], [1.0, 0.0, 0.0]]))
    rows = drawn(store, scenes.scene(store, PLACE, spots))
    assert np.c_[rows["x"], rows["y"], rows["z"]] == pytest.approx(
        np.array([[4, 2, 0], [0, 2, 0.5], [1, 0, 0]]), abs=1e-6
    )
    assert rows["view"].tolist() == [0, 1, 1]  # hall's point, then porch's two: nothing is merged


def test_a_page_drawn_spot_facing_up_the_page_looks_along_the_scenes_y_and_east_is_to_its_right():
    north = scenes.scene_at(2.0, 3.0, 0.0, mirrored=False)  # a page is x east, y south
    assert (north.x, north.y) == (2.0, -3.0)
    ahead = north.step(0.0, 1.0, 0.0)
    east = north.step(1.0, 0.0, 0.0)
    assert (ahead.x, ahead.y) == pytest.approx((2.0, -2.0)) and (east.x, east.y) == pytest.approx((3.0, -3.0))


# ---- square to the plan

#: A room 4 m by 5 m on the plan, in the scene's frame.
HOUSE = ((0.0, 0.0), (4.0, 0.0), (4.0, 5.0), (0.0, 5.0))


def corner_seen_from(truth: scenes.At) -> np.ndarray:
    """What a camera really standing at `truth` sees of the room's floor, far wall, and right wall, as its own
    levelled cloud."""
    rng = np.random.default_rng(3)

    def wall(a, b, n=3000):
        foot = np.asarray(a) + rng.random(n)[:, None] * (np.asarray(b) - np.asarray(a))
        return np.c_[foot, rng.random(n) * 2.4]

    ground = np.c_[rng.random(4000) * 4.0, rng.random(4000) * 5.0, np.zeros(4000)]
    every = np.concatenate([ground, wall(HOUSE[3], HOUSE[2]), wall(HOUSE[1], HOUSE[2])])
    move = scenes._transform(truth)
    return (every - move[:3, 3]) @ move[:3, :3]


def rooms_of_the_plan(place: str, mirrored: bool):
    return (HOUSE,)


def test_a_cloud_whose_walls_are_off_the_plan_is_turned_and_slid_onto_it_and_says_so(store, tmp_path):
    truth = at(1.0, 1.0, 20.0)
    spots = {"hall": spot("hall", at(1.15, 0.9, 24.0))}
    keep(store, "hall", points=corner_seen_from(truth))
    before = scenes.scene(store, PLACE, spots, (HOUSE,))
    assert xyb(stands(before, "hall")) == pytest.approx(xyb(where(spots["hall"])))  # not squared yet: at its spot
    assert scenes.align(store, "hall", spots, rooms_of_the_plan)
    found = scenes.scene(store, PLACE, spots, (HOUSE,))
    hall = placed(found, "hall")
    assert (hall.how, hall.squared) == ("spot", True)
    assert hall.square is not None and hall.square["state"] == "squared"
    assert hall.square["turn_deg"] == pytest.approx(-4.0, abs=0.3)
    assert xyb(stands(found, "hall")) == pytest.approx(xyb(truth), abs=0.03)
    assert (hall.shift_m, hall.turn_deg) == (pytest.approx(0.18, abs=0.03), pytest.approx(4.0, abs=0.3))
    assert found.id != before.id  # what is drawn changed
    # its far wall is drawn on the plan's far wall
    rows = drawn(store, found)
    far = rows[(rows["z"] > 0.3) & (rows["x"] < 3.5)]
    assert far["y"] == pytest.approx(5.0, abs=0.03)
    # nothing but the camera's own folder was written
    assert [path.name for path in tmp_path.iterdir()] == ["camera_clouds"]
    assert not scenes.align(store, "hall", spots, rooms_of_the_plan)  # and it is made once


def test_a_squaring_is_dropped_when_the_spot_the_plan_or_the_cloud_is_another(store):
    truth = at(1.0, 1.0, 20.0)
    spots = {"hall": spot("hall", at(1.0, 1.0, 24.0))}
    keep(store, "hall", points=corner_seen_from(truth))
    scenes.align(store, "hall", spots, rooms_of_the_plan)
    assert placed(scenes.scene(store, PLACE, spots, (HOUSE,)), "hall").squared
    turned = {"hall": spot("hall", at(1.0, 1.0, 30.0))}
    assert not placed(scenes.scene(store, PLACE, turned, (HOUSE,)), "hall").squared
    assert xyb(stands(scenes.scene(store, PLACE, turned, (HOUSE,)), "hall")) == pytest.approx(xyb(at(1.0, 1.0, 30.0)))
    redrawn = (((0.0, 0.0), (4.5, 0.0), (4.5, 5.0), (0.0, 5.0)),)
    assert not placed(scenes.scene(store, PLACE, spots, redrawn), "hall").squared
    keep(store, "hall", n=2, points=corner_seen_from(truth))
    assert not placed(scenes.scene(store, PLACE, spots, (HOUSE,)), "hall").squared
    assert scenes.align(store, "hall", spots, rooms_of_the_plan)  # made again for the new cloud
    assert placed(scenes.scene(store, PLACE, spots, (HOUSE,)), "hall").squared


def test_a_spot_set_facing_the_wrong_way_is_turned_to_where_its_walls_fit_however_far(store):
    truth = at(1.0, 1.0, 20.0)
    spots = {"hall": spot("hall", at(1.0, 1.0, 85.0))}  # 65 degrees out: far past what a squaring turns
    keep(store, "hall", points=corner_seen_from(truth))
    scenes.align(store, "hall", spots, rooms_of_the_plan)
    found = scenes.scene(store, PLACE, spots, (HOUSE,))
    hall = placed(found, "hall")
    assert (hall.how, hall.squared, (hall.follow or {})["by"]) == ("following", True, "turned")
    assert xyb(stands(found, "hall")) == pytest.approx(xyb(truth), abs=0.03)
    assert hall.turn_deg == pytest.approx(65.0, abs=0.5)
    assert spots["hall"].at == at(1.0, 1.0, 85.0)  # the spot itself is the owner's: it stays


def test_a_camera_with_no_spot_has_nothing_to_be_squared_from(store):
    keep(store, "hall", points=corner_seen_from(at(1.0, 1.0, 20.0)))
    assert not scenes.align(store, "hall", {}, rooms_of_the_plan)
    assert not scenes.align(store, "hall", {"hall": spot("hall", None)}, rooms_of_the_plan)


def test_between_two_squared_cameras_the_pictures_are_a_check_that_moves_neither(store):
    truth = {"hall": at(1.0, 1.0, 20.0), "porch": at(3.0, 1.5, -10.0)}
    spots = {"hall": spot("hall", at(1.0, 1.0, 23.0)), "porch": spot("porch", at(3.0, 1.5, -13.0))}
    scenes.record(store, keep(store, "hall", points=corner_seen_from(truth["hall"])), {}, spots)
    cloud = keep(store, "porch", points=corner_seen_from(truth["porch"]))
    # the pictures say the porch is 0.2 m farther right than it is
    scenes.record(store, cloud, {"hall": seen(truth["hall"], truth["porch"].step(0.2, 0.0, 0.0))}, spots)
    for camera in spots:
        scenes.align(store, camera, spots, rooms_of_the_plan)
    found = scenes.scene(store, PLACE, spots, (HOUSE,))
    (pair,) = found.pairs
    assert pair.state == "checked"
    assert pair.off_m == pytest.approx(0.2, abs=0.05)
    for camera in spots:
        assert placed(found, camera).how == "spot"
        assert xyb(stands(found, camera)) == pytest.approx(xyb(truth[camera]), abs=0.04)


def test_a_squared_camera_holds_and_its_unsquared_partner_takes_the_pairs_correction(store):
    truth = {"hall": at(1.0, 1.0, 20.0), "porch": at(3.0, 1.5, -10.0)}
    spots = {"hall": spot("hall", at(1.0, 1.0, 23.0)), "porch": spot("porch", at(3.2, 1.4, -14.0))}
    scenes.record(store, keep(store, "hall", points=corner_seen_from(truth["hall"])), {}, spots)
    scenes.record(store, keep(store, "porch"), {"hall": seen(truth["hall"], truth["porch"])}, spots)  # no wall in it
    for camera in spots:
        scenes.align(store, camera, spots, rooms_of_the_plan)
    found = scenes.scene(store, PLACE, spots, (HOUSE,))
    assert (placed(found, "hall").squared, placed(found, "porch").squared) == (True, False)
    assert (placed(found, "porch").square or {})["state"] == "no_walls"
    assert found.pairs[0].state == "tightened"
    assert xyb(stands(found, "hall")) == pytest.approx(xyb(truth["hall"]), abs=0.04)
    assert xyb(stands(found, "porch")) == pytest.approx(xyb(truth["porch"]), abs=0.04)
    assert placed(found, "porch").how == "adjusted"


def test_a_placed_camera_says_what_levelled_its_cloud_its_tilt_and_when_its_picture_was_taken(store):
    spots = {"hall": spot("hall", at(1.0, 1.0, 0.0))}
    cloud = keep(store, "hall")
    hall = placed(scenes.scene(store, PLACE, spots), "hall")
    assert (hall.at, hall.level, hall.tilt_deg, hall.roll_deg) == (cloud.at, "floor", 0.0, 0.0)
    assert (hall.square, hall.squared) == (None, False)


# ---- fit to the rooms, and draw them flat


def test_a_cloud_a_tenth_too_big_is_resized_to_the_plans_rooms_and_its_walls_and_floor_drawn_flat(store, tmp_path):
    truth = at(1.0, 1.0, 20.0)
    spots = {"hall": spot("hall", truth)}
    keep(store, "hall", points=corner_seen_from(truth) * 1.1)
    assert scenes.align(store, "hall", spots, rooms_of_the_plan)
    found = scenes.scene(store, PLACE, spots, (HOUSE,))
    hall = placed(found, "hall")
    assert (hall.how, hall.fitted) == ("spot", True)
    assert hall.fit is not None and hall.fit["state"] == "fitted" and len(hall.fit["walls"]) == 2
    assert hall.fit["scale"] == pytest.approx(1 / 1.1, abs=0.02)
    move = np.array(hall.transform).reshape(4, 4)
    assert move[2, 2] == pytest.approx(1 / 1.1, abs=0.02)  # the cloud is resized, not only moved
    assert np.array(hall.pose).reshape(4, 4)[2, 3] == pytest.approx(HEIGHT / 1.1, abs=0.03)
    flat = drawn(store, found)
    walls = flat[flat["z"] > 0.3]
    far, right = walls[walls["x"] < 3.9], walls[walls["x"] >= 3.9]
    assert far["y"] == pytest.approx(5.0, abs=1e-6)  # on the plan's far wall exactly
    assert right["x"] == pytest.approx(4.0, abs=1e-6)
    assert flat[(flat["x"] < 3.9) & (flat["y"] < 4.9)]["z"] == pytest.approx(0.0, abs=1e-6)  # the floor, flat
    raw = ply.vertices(scenes.points(store, found, flat=False))
    assert raw is not None and len(raw) == len(flat)
    assert float(np.abs(raw["y"][raw["z"] > 0.3] - 5.0).min()) < 0.05  # resized onto the wall, to the fit's few cm
    assert not np.allclose(raw["x"], flat["x"])  # and not flattened
    assert [path.name for path in tmp_path.iterdir()] == ["camera_clouds"]  # only beside the clouds


def test_the_flat_and_the_raw_drawing_of_one_scene_are_kept_side_by_side_and_an_older_scenes_go(store):
    truth = at(1.0, 1.0, 20.0)
    spots = {"hall": spot("hall", truth)}
    keep(store, "hall", points=corner_seen_from(truth))
    scenes.align(store, "hall", spots, rooms_of_the_plan)
    found = scenes.scene(store, PLACE, spots, (HOUSE,))
    flat, raw = scenes.drawn(store, found), scenes.drawn(store, found, flat=False)
    assert flat != raw and flat.is_file() and raw.is_file()
    assert flat.read_bytes() != raw.read_bytes()
    keep(store, "hall", n=2, points=corner_seen_from(truth))
    scenes.align(store, "hall", spots, rooms_of_the_plan)
    later = scenes.drawn(store, scenes.scene(store, PLACE, spots, (HOUSE,)))
    assert later.is_file() and not flat.exists() and not raw.exists()


def test_a_cloud_whose_walls_cant_be_matched_is_left_as_it_was_squared_and_says_why(store):
    spots = {"hall": spot("hall", at(2.0, 1.0, 0.0))}
    rng = np.random.default_rng(4)
    island = np.c_[rng.random(4000) * 3.0 + 0.5, np.full(4000, 3.5), rng.random(4000) * 2.4]  # 1.5 m short of the wall
    move = scenes._transform(at(2.0, 1.0, 0.0))
    keep(store, "hall", points=(island - move[:3, 3]) @ move[:3, :3])
    scenes.align(store, "hall", spots, rooms_of_the_plan)
    found = scenes.scene(store, PLACE, spots, (HOUSE,))
    hall = placed(found, "hall")
    assert (hall.fitted, (hall.fit or {})["state"], hall.squared) == (False, "no_match", True)
    assert np.array(hall.transform).reshape(4, 4)[2, 2] == pytest.approx(1.0)  # its own size
    assert drawn(store, found)["y"] == pytest.approx(3.5, abs=0.01)  # nothing laid on a wall


def test_a_fitted_cameras_own_view_is_its_cloud_at_the_rooms_size_with_flat_walls(store):
    truth = at(1.0, 1.0, 20.0)
    spots = {"hall": spot("hall", truth)}
    cloud = keep(store, "hall", points=corner_seen_from(truth) * 1.1)
    scenes.align(store, "hall", spots, rooms_of_the_plan)
    hall = placed(scenes.scene(store, PLACE, spots, (HOUSE,)), "hall")
    own = ply.vertices(scenes.own_points(store, hall, cloud))
    assert own is not None and len(own) > 0
    pose = scenes.own_pose(hall, cloud)
    assert pose[:3, 3] == pytest.approx([0.0, 0.0, HEIGHT / 1.1], abs=0.03)  # the camera, on its own z axis still
    # put back where the camera stands, its points are the scene's flat ones
    back = scenes._transform(stands(scenes.scene(store, PLACE, spots, (HOUSE,)), "hall"))
    xyz = np.c_[own["x"], own["y"], own["z"]] @ back[:3, :3].T + back[:3, 3]
    far = xyz[(xyz[:, 2] > 0.3) & (xyz[:, 0] < 3.9)]
    assert far[:, 1] == pytest.approx(5.0, abs=1e-4)
    assert scenes.drawn_own(store, hall, cloud).is_file()


def test_stored_clouds_are_brought_up_to_date_with_no_picture_and_no_model(store, tmp_path, monkeypatch):
    truth = at(1.0, 1.0, 20.0)
    spots = {"hall": spot("hall", at(1.0, 1.0, 23.0))}
    keep(store, "hall", points=corner_seen_from(truth))
    keep(store, "yard")  # a camera with no spot: nothing to square it to
    monkeypatch.setattr(scenes, "spots_of", lambda state: spots)
    monkeypatch.setattr(scenes, "plan_rooms", lambda state, place, mirrored: (HOUSE,))
    assert scenes.update(tmp_path, store) == 1
    hall = placed(scenes.scene(store, PLACE, spots, (HOUSE,)), "hall")
    assert (hall.squared, hall.fitted) == (True, True)
    assert scenes.update(tmp_path, store) == 0  # nothing is stale: nothing is done
    assert scenes.update(tmp_path, store, place="another") == 0  # and not for a place it has no spot in


# ---- 2. tighten


def tightened(store, truth: dict[str, scenes.At], spots: dict[str, scenes.SpotAt], **said) -> None:
    """Keep a cloud for each camera of `truth`, the second compared with the first as the runner would."""
    first, second = sorted(truth)
    scenes.record(store, keep(store, first), {}, spots)
    cloud = keep(store, second)
    scenes.record(store, cloud, {first: seen(truth[first], truth[second], **said)}, spots)


def test_two_cameras_whose_spots_are_slightly_off_are_pulled_to_what_their_pictures_say(store):
    truth = {"hall": at(0.0, 0.0, 0.0), "porch": at(2.0, 2.5, -80.0)}
    spots = {"hall": spot("hall", at(0.1, -0.1, 3.0)), "porch": spot("porch", at(2.2, 2.4, -85.0))}
    tightened(store, truth, spots)
    found = scenes.scene(store, PLACE, spots)
    assert [(p.camera, p.how, p.against) for p in found.cameras] == [
        ("hall", "adjusted", ("porch",)),
        ("porch", "adjusted", ("hall",)),
    ]
    # between them the cameras now sit as they really do, to a few centimetres and a fraction of a degree
    right, ahead, turn = stands(found, "porch").seen_from(stands(found, "hall"))
    want = truth["porch"].seen_from(truth["hall"])
    assert (right, ahead) == pytest.approx(want[:2], abs=0.03) and math.degrees(turn) == pytest.approx(-80.0, abs=0.5)
    # and neither left its spot by much: they shared the correction, and the scene stayed on the plan
    for camera in truth:
        off_m, off_deg = scenes.moved_by(stands(found, camera), where(spots[camera]))
        assert 0.02 < off_m < 0.3 and off_deg < 6.0
        assert (placed(found, camera).shift_m, placed(found, camera).turn_deg) == pytest.approx(
            (off_m, off_deg), abs=0.06
        )
    assert [(p.state, round(p.off_deg)) for p in found.pairs] == [("tightened", 8)]


@pytest.mark.parametrize(
    ("said", "state"),
    [
        pytest.param({"sure": (0.0, 0.0), "found": (0.0, 0.09)}, "apart", id="different-space"),
        pytest.param({"sure": (0.5, 0.4), "found": (0.09, 0.1)}, "apart", id="too-little-matched"),
        pytest.param({"disagree": (12.0, 0.1, 0.02)}, "unsure", id="directions-disagree-in-turn"),
        pytest.param({"disagree": (2.0, 0.9, 0.02)}, "unsure", id="directions-disagree-in-shift"),
    ],
)
def test_a_pair_that_doesnt_match_confidently_stays_where_the_spots_put_it(store, said, state):
    truth = {"hall": at(0.0, 0.0, 0.0), "porch": at(2.0, 2.5, -80.0)}
    spots = {"hall": spot("hall", at(0.1, -0.1, 3.0)), "porch": spot("porch", at(2.2, 2.4, -85.0))}
    tightened(store, truth, spots, **said)
    found = scenes.scene(store, PLACE, spots)
    assert [(p.how, p.shift_m, p.turn_deg) for p in found.cameras] == [("spot", 0.0, 0.0), ("spot", 0.0, 0.0)]
    assert [xyb(stands(found, c)) for c in truth] == [pytest.approx(xyb(where(spots[c]))) for c in truth]
    assert [p.state for p in found.pairs] == [state]


def test_the_runners_own_numbers_from_live_cameras_fall_on_the_right_side_of_the_test():
    room_pair = {"match": [0.218, 0.114], "sure": [0.613, 0.245], "rotation": [1] * 9, "shift": [2.8, 0, 0.3],
                 "disagree": [3.1, 0.158, 0.027]}  # fmt: skip
    turned = {**room_pair, "match": [0.094, 0.156], "sure": [0.729, 0.689], "disagree": [5.5, 0.734, 0.064]}
    assert scenes.confident(room_pair) is None
    assert scenes.confident(turned) == "unsure"
    assert scenes.confident({"match": [0.089, 0.094], "sure": [0.0, 0.0]}) == "apart"
    assert scenes.confident(None) == "none"


def test_a_match_that_would_move_a_camera_past_the_bound_moves_nothing_and_says_the_spots_disagree(store):
    truth = {"hall": at(0.0, 0.0, 0.0), "porch": at(2.0, 2.5, -80.0)}
    spots = {"hall": spot("hall", at(0.0, 0.0, 0.0)), "porch": spot("porch", at(4.5, 2.5, -30.0))}  # one spot is stale
    tightened(store, truth, spots)
    found = scenes.scene(store, PLACE, spots)
    assert [(p.how, p.shift_m) for p in found.cameras] == [("spot", 0.0), ("spot", 0.0)]
    (pair,) = found.pairs
    assert pair.state == "disagrees" and pair.off_m > 2.0 and pair.off_deg == pytest.approx(50, abs=1)


def test_a_correction_made_at_another_spot_is_dropped(store):
    truth = {"hall": at(0.0, 0.0, 0.0), "porch": at(2.0, 2.5, -80.0)}
    spots = {"hall": spot("hall", at(0.1, -0.1, 3.0)), "porch": spot("porch", at(2.2, 2.4, -85.0))}
    tightened(store, truth, spots)
    moved = {**spots, "porch": spot("porch", at(2.0, 2.5, -80.0), basis="the owner moved it")}
    found = scenes.scene(store, PLACE, moved)
    assert [p.how for p in found.cameras] == ["spot", "spot"] and [p.state for p in found.pairs] == ["stale"]


def test_a_daylight_correction_is_kept_through_a_night_look_that_doesnt_match(store):
    truth = {"hall": at(0.0, 0.0, 0.0), "porch": at(2.0, 2.5, -80.0)}
    spots = {"hall": spot("hall", at(0.1, -0.1, 3.0)), "porch": spot("porch", at(2.2, 2.4, -85.0))}
    tightened(store, truth, spots)
    by_day = [stands(scenes.scene(store, PLACE, spots), c) for c in truth]
    # at night the porch's picture is made again, still (it matches its anchor), and no longer matches the hall's
    again = keep(store, "porch", n=2)
    scenes.record(store, again, {"anchor": seen(truth["porch"], truth["porch"]), "hall": APART}, spots)
    found = scenes.scene(store, PLACE, spots)
    assert [xyb(stands(found, c)) for c in truth] == [pytest.approx(xyb(p)) for p in by_day]
    (pair,) = found.pairs
    assert (pair.state, pair.latest, pair.measured) == ("tightened", "apart", T + 1)


def test_with_no_measurement_the_solved_positions_are_the_spots():
    spots = {"hall": at(1.0, 2.0, 30.0), "porch": at(-1.0, 0.0, 200.0)}
    assert scenes.solve(spots, []) == spots


# ---- 3. a camera that was turned or carried is followed

#: Two made-up rooms that share a wall: a hall 4 m by 5 m, and beside it a den shaped like an L.
HALL = ((0.0, 0.0), (4.0, 0.0), (4.0, 5.0), (0.0, 5.0))
DEN = ((4.0, 0.0), (10.0, 0.0), (10.0, 3.5), (7.0, 3.5), (7.0, 6.0), (4.0, 6.0))
ROOMS = (HALL, DEN)


def view_from(truth: scenes.At, fov: float = 100.0, walls: bool = True) -> np.ndarray:
    """What a camera really standing at `truth` in ROOMS sees, as its own levelled cloud: along each sight line
    across its field of view, the floor as far as the first wall, and that wall up to 2.4 m (or, without `walls`,
    the floor alone)."""
    rng = np.random.default_rng(3)
    ends = np.array([[ring[i], ring[(i + 1) % len(ring)]] for ring in ROOMS for i in range(len(ring))])
    starts, runs = ends[:, 0], ends[:, 1] - ends[:, 0]
    eye = np.array([truth.x, truth.y])
    out = []
    for angle in truth.bearing + np.radians(np.linspace(-fov / 2, fov / 2, 500)):
        ray = np.array([math.sin(angle), math.cos(angle)])
        cross = ray[0] * runs[:, 1] - ray[1] * runs[:, 0]
        gap = starts - eye
        with np.errstate(divide="ignore", invalid="ignore"):
            along_ray = (gap[:, 0] * runs[:, 1] - gap[:, 1] * runs[:, 0]) / cross
            along_wall = (gap[:, 0] * ray[1] - gap[:, 1] * ray[0]) / cross
        met = (np.abs(cross) > 1e-12) & (along_ray > 1e-6) & (along_wall >= 0) & (along_wall <= 1)
        far = float(along_ray[met].min())
        out.append(np.c_[eye + ray * (rng.random(6)[:, None] * far), np.zeros(6)])
        if walls:
            out.append(np.c_[np.tile(eye + ray * far, (8, 1)), rng.random(8) * 2.4])
    move = scenes._transform(truth)
    return (np.concatenate(out) - move[:3, 3]) @ move[:3, :3]


def two_rooms(place: str, mirrored: bool):
    return ROOMS


def anchored(store, spots, camera="hall", points=None) -> None:
    """`camera` gets its first cloud at its spot: that picture is its anchor."""
    scenes.record(store, keep(store, camera, points=points), {}, spots)


def looks_elsewhere(store, spots, camera: str, truth: scenes.At | None, n: int = 2, **view) -> clouds.Cloud:
    """`camera`'s `n`th cloud, of what it sees from `truth` (nothing to speak of, with None): its picture shares
    nothing with the one it was placed with, or with another camera's."""
    points = np.array([[0.0, 9.0, 0.0]]) if truth is None else view_from(truth, **view)
    cloud = keep(store, camera, n=n, points=points)
    scenes.record(store, cloud, {"anchor": APART, **{other: APART for other in spots if other != camera}}, spots)
    return cloud


def test_a_camera_that_registers_against_its_old_picture_is_placed_by_what_the_pictures_measured(store):
    spots = {"hall": spot("hall", at(3.0, 2.0, 90.0))}
    anchored(store, spots)
    now = at(3.0, 2.0, 90.0).step(0.4, 0.1, math.radians(35.0))  # turned 35 degrees on its mount
    scenes.record(store, keep(store, "hall", n=2), {"anchor": seen(now, at(3.0, 2.0, 90.0))}, spots)
    assert placed(scenes.scene(store, PLACE, spots), "hall").how == "lost"  # until it is placed again, it isn't drawn
    assert scenes.align(store, "hall", spots, two_rooms)
    found = scenes.scene(store, PLACE, spots)
    hall = placed(found, "hall")
    assert hall.how == "following" and xyb(stands(found, "hall")) == pytest.approx(xyb(now))
    assert (hall.shift_m, hall.turn_deg) == pytest.approx((math.hypot(0.4, 0.1), 35.0), abs=0.05)
    assert hall.follow == {"by": "pictures", "since": T + 2, "lost": "", "pinned": False}
    assert spots["hall"].at == at(3.0, 2.0, 90.0)  # the spot itself is the owner's: it stays


def test_a_camera_whose_new_picture_matches_its_old_one_closely_hasnt_moved(store):
    spots = {"hall": spot("hall", at(3.0, 2.0, 90.0))}
    anchored(store, spots)
    nudged = at(3.0, 2.0, 90.0).step(0.05, 0.02, math.radians(1.5))
    scenes.record(store, keep(store, "hall", n=2), {"anchor": seen(nudged, at(3.0, 2.0, 90.0))}, spots)
    hall = placed(scenes.scene(store, PLACE, spots), "hall")
    assert (hall.how, hall.follow, hall.shift_m) == ("spot", None, 0.0)


def test_a_camera_that_only_registers_against_another_camera_is_placed_from_that_one(store):
    spots = {"hall": spot("hall", at(0.0, 0.0, 0.0)), "porch": spot("porch", at(2.0, 2.5, -80.0))}
    anchored(store, spots, "hall")
    anchored(store, spots, "porch")
    now = at(1.0, 4.0, 170.0)  # the porch camera was carried across the room: its old picture shows other space
    carried = keep(store, "porch", n=2)
    scenes.record(store, carried, {"anchor": APART, "hall": seen(where(spots["hall"]), now)}, spots)
    scenes.align(store, "porch", spots, two_rooms)
    found = scenes.scene(store, PLACE, spots)
    porch = placed(found, "porch")
    assert (porch.how, (porch.follow or {})["by"]) == ("following", "pictures")
    assert xyb(stands(found, "porch")) == pytest.approx(xyb(now), abs=1e-6)
    assert placed(found, "hall").how == "spot"  # the camera that stayed isn't moved by it


def test_a_camera_turned_ninety_degrees_is_placed_again_by_its_walls_without_being_asked(store, tmp_path):
    spots = {"hall": spot("hall", at(1.0, 1.0, 20.0))}
    anchored(store, spots, points=view_from(at(1.0, 1.0, 20.0)))
    scenes.align(store, "hall", spots, two_rooms)
    truth = at(1.0, 1.0, 110.0)
    looks_elsewhere(store, spots, "hall", truth)
    assert scenes.align(store, "hall", spots, two_rooms)
    found = scenes.scene(store, PLACE, spots, ROOMS)
    hall = placed(found, "hall")
    assert (hall.how, hall.fitted, (hall.follow or {})["by"]) == ("following", True, "turned")
    assert xyb(stands(found, "hall")) == pytest.approx(xyb(truth), abs=0.03)
    assert (hall.shift_m, hall.turn_deg) == (pytest.approx(0.0, abs=0.03), pytest.approx(90.0, abs=0.5))
    # the wall it now faces is drawn on the plan's wall between the two rooms
    rows = drawn(store, found)
    ahead = rows[(rows["z"] > 0.3) & (rows["y"] > 0.3)]
    assert ahead["x"] == pytest.approx(4.0, abs=0.03)
    # nothing but the camera's own folder was written: the owner's spots and plan are as they were
    assert [path.name for path in tmp_path.iterdir()] == ["camera_clouds"]
    assert spots["hall"].at == at(1.0, 1.0, 20.0)


def test_a_cupboard_that_fills_the_new_view_doesnt_keep_a_turned_camera_from_being_placed(store):
    spots = {"hall": spot("hall", at(1.0, 1.0, 20.0))}
    anchored(store, spots, points=view_from(at(1.0, 1.0, 20.0)))
    truth = at(1.0, 1.0, 110.0)
    # a cupboard 60 cm out from the wall it now faces, askew to it, holding more points than the walls do
    rng = np.random.default_rng(5)
    run = np.array([math.sin(math.radians(13.0)), math.cos(math.radians(13.0))])
    front = np.c_[np.array([3.4, 0.9]) + (rng.random(6000)[:, None] - 0.5) * run, rng.random(6000) * 1.6]
    move = scenes._transform(truth)
    cloud = np.concatenate([view_from(truth), (front - move[:3, 3]) @ move[:3, :3]])
    scenes.record(store, keep(store, "hall", n=2, points=cloud), {"anchor": APART}, spots)
    scenes.align(store, "hall", spots, two_rooms)
    found = scenes.scene(store, PLACE, spots, ROOMS)
    assert placed(found, "hall").how == "following"
    assert xyb(stands(found, "hall")) == pytest.approx(xyb(truth), abs=0.05)


def test_a_camera_carried_to_another_room_is_placed_on_that_rooms_walls(store):
    spots = {"hall": spot("hall", at(1.0, 1.0, 20.0))}
    anchored(store, spots, points=view_from(at(1.0, 1.0, 20.0)))
    truth = at(5.5, 1.5, 40.0)  # in the den, looking at the corner that makes it an L
    looks_elsewhere(store, spots, "hall", truth)
    scenes.align(store, "hall", spots, two_rooms)
    found = scenes.scene(store, PLACE, spots, ROOMS)
    hall = placed(found, "hall")
    assert (hall.how, (hall.follow or {})["by"]) == ("following", "walls")
    assert xyb(stands(found, "hall")) == pytest.approx(xyb(truth), abs=0.03)
    assert hall.shift_m == pytest.approx(math.hypot(4.5, 0.5), abs=0.03)
    assert not drawn(store, found)["outside"].any()  # every point of its cloud lies in the plan's rooms


def test_two_cameras_moved_at_once_are_each_placed_again_and_neither_is_taken_to_be_fixed(store):
    spots = {"hall": spot("hall", at(1.0, 1.0, 20.0)), "porch": spot("porch", at(8.0, 1.0, -30.0))}
    anchored(store, spots, "hall", view_from(at(1.0, 1.0, 20.0)))
    anchored(store, spots, "porch", view_from(at(8.0, 1.0, -30.0)))
    truth = {"hall": at(5.0, 5.0, 150.0), "porch": at(8.0, 1.0, 60.0)}  # one carried into the den, one turned there
    looks_elsewhere(store, spots, "hall", truth["hall"])
    looks_elsewhere(store, spots, "porch", truth["porch"])
    scenes.align(store, "hall", spots, two_rooms)
    scenes.align(store, "porch", spots, two_rooms)
    found = scenes.scene(store, PLACE, spots, ROOMS)
    assert [(p.camera, p.how) for p in found.cameras] == [("hall", "following"), ("porch", "following")]
    assert xyb(stands(found, "hall")) == pytest.approx(xyb(truth["hall"]), abs=0.03)
    assert xyb(stands(found, "porch")) == pytest.approx(xyb(truth["porch"]), abs=0.03)


@pytest.mark.parametrize(
    ("truth", "view", "why"),
    [
        pytest.param(None, {}, "no_walls", id="nothing-to-go-by"),
        pytest.param(at(2.0, 2.5, 0.0), {"walls": False}, "no_walls", id="floor-alone"),
        # from the middle of the hall, its end wall and both sides look the same facing either end
        pytest.param(at(2.0, 2.5, 0.0), {}, "ambiguous", id="a-view-that-fits-two-ways"),
    ],
)
def test_a_view_that_cant_be_placed_with_confidence_is_not_drawn_keeps_its_last_pose_and_says_why(
    store, truth, view, why
):
    spots = {"hall": spot("hall", at(1.0, 1.0, 20.0)), "porch": spot("porch", at(8.0, 1.0, -30.0))}
    anchored(store, spots, "hall", view_from(at(1.0, 1.0, 20.0)))
    anchored(store, spots, "porch")
    stood = at(1.0, 1.0, 110.0)
    looks_elsewhere(store, spots, "hall", stood)
    scenes.align(store, "hall", spots, two_rooms)  # followed once: that is its last good pose
    looks_elsewhere(store, spots, "hall", truth, n=3, **view)
    scenes.align(store, "hall", spots, two_rooms)
    found = scenes.scene(store, PLACE, spots, ROOMS)
    hall = placed(found, "hall")
    assert (hall.how, hall.placed, hall.transform, (hall.follow or {})["lost"]) == ("lost", False, None, why)
    assert set(drawn(store, found)["view"].tolist()) == {1}  # none of its points are drawn, anywhere
    kept = scenes.following(store, "hall", spots["hall"].basis)
    assert kept is not None and kept.at == pytest.approx(xyb(stood), abs=0.03)
    assert store.current("hall") is not None  # its own cloud is still its own


def test_a_lost_camera_whose_old_view_comes_back_stands_where_it_last_did(store):
    spots = {"hall": spot("hall", at(1.0, 1.0, 20.0))}
    anchored(store, spots, points=view_from(at(1.0, 1.0, 20.0)))
    looks_elsewhere(store, spots, "hall", None)
    scenes.align(store, "hall", spots, two_rooms)
    assert placed(scenes.scene(store, PLACE, spots, ROOMS), "hall").how == "lost"
    back = keep(store, "hall", n=3, points=view_from(at(1.0, 1.0, 20.0)))
    scenes.record(store, back, {"anchor": seen(at(1.0, 1.0, 20.0), at(1.0, 1.0, 20.0))}, spots)
    scenes.align(store, "hall", spots, two_rooms)
    found = scenes.scene(store, PLACE, spots, ROOMS)
    assert placed(found, "hall").how == "spot" and xyb(stands(found, "hall")) == pytest.approx(
        (1.0, 1.0, math.radians(20.0)), abs=0.03
    )


def test_the_way_back_puts_a_followed_camera_on_the_plans_spot_until_its_view_next_changes(store):
    spots = {"hall": spot("hall", at(1.0, 1.0, 20.0))}
    anchored(store, spots, points=view_from(at(1.0, 1.0, 20.0)))
    looks_elsewhere(store, spots, "hall", at(5.5, 1.5, 40.0))
    scenes.align(store, "hall", spots, two_rooms)
    assert placed(scenes.scene(store, PLACE, spots, ROOMS), "hall").how == "following"

    assert scenes.use_spot(store, "hall", spots["hall"].basis)
    scenes.align(store, "hall", spots, two_rooms)
    found = scenes.scene(store, PLACE, spots, ROOMS)
    hall = placed(found, "hall")
    assert (hall.how, hall.shift_m, (hall.follow or {})["pinned"]) == ("spot", 0.0, True)
    assert xyb(stands(found, "hall")) == pytest.approx(xyb(at(1.0, 1.0, 20.0)))  # as the plan has it, not re-turned

    assert scenes.follow_again(store, "hall", spots["hall"].basis)  # and the way back can itself be taken back
    scenes.align(store, "hall", spots, two_rooms)
    again = scenes.scene(store, PLACE, spots, ROOMS)
    assert placed(again, "hall").how == "following"
    assert xyb(stands(again, "hall")) == pytest.approx(xyb(at(5.5, 1.5, 40.0)), abs=0.03)
    assert not scenes.follow_again(store, "hall", spots["hall"].basis)  # nothing to take back twice

    scenes.use_spot(store, "hall", spots["hall"].basis)
    looks_elsewhere(store, spots, "hall", at(1.0, 1.0, 110.0), n=3)  # its view changes again: it is followed again
    scenes.align(store, "hall", spots, two_rooms)
    assert placed(scenes.scene(store, PLACE, spots, ROOMS), "hall").how == "following"


def test_setting_a_followed_cameras_spot_again_places_it_from_the_new_spot(store):
    spots = {"hall": spot("hall", at(1.0, 1.0, 20.0))}
    anchored(store, spots, points=view_from(at(1.0, 1.0, 20.0)))
    looks_elsewhere(store, spots, "hall", at(5.5, 1.5, 40.0))
    scenes.align(store, "hall", spots, two_rooms)
    reset = {"hall": spot("hall", at(5.0, 1.0, 45.0), basis="set again")}
    hall = scenes.scene(store, PLACE, reset)
    assert placed(hall, "hall").how == "spot" and xyb(stands(hall, "hall")) == pytest.approx(xyb(at(5.0, 1.0, 45.0)))


def test_a_new_picture_is_compared_with_its_anchor_and_its_nearest_neighbours_of_the_same_place(store):
    spots = {
        "hall": spot("hall", at(0.0, 0.0, 0.0)),
        "porch": spot("porch", at(9.0, 0.0, 0.0)),
        "den": spot("den", at(1.0, 1.0, 0.0)),
        "shed": spot("shed", at(0.5, 0.0, 0.0), place="22222222-2222-7222-8222-222222222222"),
        "yard": spot("yard", None),
    }
    for camera in ("porch", "den", "shed", "yard"):
        keep(store, camera)
    assert [label for label, _ in scenes.compare(store, "hall", spots)] == ["den", "porch"]
    anchored(store, spots)
    assert [label for label, _ in scenes.compare(store, "hall", spots)] == ["anchor", "den", "porch"]
    assert scenes.compare(store, "yard", spots) == []  # no facing: it isn't placed, so there is nothing to compare


# ---- what the view leaves out, and the cleaning


ROOM = [((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0))]


def test_points_well_outside_every_room_of_the_plan_are_marked_and_those_just_past_a_wall_are_not():
    xy = np.array([[2.0, 1.5], [4.2, 1.5], [4.6, 1.5], [-0.2, -0.2], [2.0, 9.0]])
    assert cloud_align.outside(xy, ROOM).tolist() == [False, False, True, False, True]
    assert not cloud_align.outside(xy, []).any()  # with no plan there is nothing to be outside of


def test_the_scenes_points_say_which_lie_outside_the_plan_without_dropping_them(store):
    spots = {"hall": spot("hall", at(1.0, 1.0, 0.0))}
    keep(store, "hall", points=np.array([[0.0, 1.0, 0.0], [0.0, 6.0, 0.0]]))
    rows = drawn(store, scenes.scene(store, PLACE, spots, ROOM))
    assert rows["outside"].tolist() == [0, 1] and len(rows) == 2


def test_a_scenes_id_changes_when_a_cloud_moves_or_the_plans_rooms_do(store):
    spots = {"hall": spot("hall", at(1.0, 1.0, 0.0))}
    keep(store, "hall")
    first = scenes.scene(store, PLACE, spots, ROOM).id
    assert scenes.scene(store, PLACE, spots, ROOM).id == first
    assert scenes.scene(store, PLACE, {"hall": spot("hall", at(1.5, 1.0, 0.0))}, ROOM).id != first
    assert scenes.scene(store, PLACE, spots, []).id != first


def wall_and_box() -> tuple[np.ndarray, np.ndarray]:
    """A point map of a wall 4 m ahead with a box 2 m ahead in front of its middle, and every pixel sure."""
    h, w, focal = 60, 80, 60.0
    v, u = np.mgrid[0:h, 0:w]
    depth = np.where((abs(u - w / 2) < 12) & (abs(v - h / 2) < 12), 2.0, 4.0)
    rays = np.stack([(u - w / 2) / focal, (v - h / 2) / focal, np.ones((h, w))], axis=2)
    return rays * depth[..., None], np.ones((h, w), dtype=bool)


def test_cleaning_drops_the_pixels_at_a_jump_in_depth_and_keeps_the_surfaces():
    points, sure = wall_and_box()
    kept = clouds.clean(points, sure)
    assert not kept[30, 29] and not kept[30, 28]  # the box's left edge and the wall pixel beside it
    assert kept[30, 40] and kept[5, 5]  # the middle of the box, and the wall
    assert 0.9 < kept.mean() < 0.99


def test_cleaning_drops_a_speck_and_a_surface_seen_edge_on():
    points, sure = wall_and_box()
    alone = np.zeros_like(sure)
    alone[3:6, 3:6] = True  # nine sure pixels with nothing round them
    assert not clouds.clean(points, alone).any()
    h, w = sure.shape
    v, u = np.mgrid[0:h, 0:w]
    # a wall running away from the camera along its line of sight, 0.2 m to its side: seen at a few degrees
    sheet = np.stack([np.full((h, w), 0.2), (v - h / 2) / 30.0, 2.0 + u * 0.2], axis=2)
    assert clouds.clean(sheet, sure).mean() < 0.05


# ---- the runner's fit


def test_the_runners_similarity_finds_the_turn_shift_and_scale_between_two_sets_of_points():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(400, 3))
    angle = math.radians(40.0)
    turn = np.array([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]])
    dst = 1.3 * src @ turn.T + [0.5, -0.2, 2.0]
    dst[:20] += rng.normal(scale=3.0, size=(20, 3))  # a twentieth of the points are wild
    scale, found, shift = runner.similarity(src, dst)
    assert scale == pytest.approx(1.3, abs=0.01) and shift == pytest.approx([0.5, -0.2, 2.0], abs=0.02)
    assert found == pytest.approx(turn, abs=0.01)
