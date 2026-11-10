"""Fixed-camera footprints: a tracked person's foot point, sampled per track, projected onto a place's floor,
and the walked floor closing a room that a fixed camera's cloud has no walls for."""

import math
from dataclasses import replace

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from liberational import segment_log
from liberational.memory import places
from liberational.spatial import floorplan as fp
from liberational.spatial import footprints as F

SCALE = F.SCALE


def joints(**placed: tuple[float, float, float]) -> list[int]:
    """A halpe26 joint list with the named joints placed (x, y, confidence), the rest not placed."""
    index = {
        "left_hip": 11,
        "right_hip": 12,
        "left_ankle": 15,
        "right_ankle": 16,
        "head": 17,
        "left_heel": 24,
        "right_heel": 25,
    }
    out = [-SCALE, -SCALE, 0] * 26
    for name, (x, y, c) in placed.items():
        j = index[name]
        out[3 * j : 3 * j + 3] = [round(x * SCALE), round(y * SCALE), round(c * SCALE)]
    return out


BOX = [0.2, 0.1, 0.4, 0.9, 0.8]  # x1, y1, x2, y2, detector score


@pytest.mark.parametrize(
    "placed,expected",
    [
        pytest.param(
            {"left_heel": (0.25, 0.88, 0.9), "right_heel": (0.35, 0.86, 0.7), "left_ankle": (0.1, 0.1, 0.9)},
            (0.30, 0.87, 0.8 * 0.7),
            id="heels-first",
        ),
        pytest.param(
            {"left_heel": (0.25, 0.88, 0.1), "left_ankle": (0.26, 0.84, 0.6), "right_ankle": (0.34, 0.82, 0.8)},
            (0.30, 0.83, 0.8 * 0.6),
            id="ankles-when-heels-unsure",
        ),
        pytest.param({}, None, id="no-feet-placed"),
    ],
)
def test_the_foot_point_is_the_heels_then_the_ankles_never_the_box(placed, expected):
    # The box's bottom isn't the feet when furniture hides the legs
    got = F.foot_point(BOX, joints(**placed))
    assert got == (pytest.approx(expected) if expected else None)


def test_feet_pose_placed_count_even_at_the_frames_bottom_edge():
    clipped = [0.2, 0.3, 0.4, 0.99, 0.8]
    assert F.foot_point(clipped, joints(left_heel=(0.3, 0.97, 0.9))) == pytest.approx((0.3, 0.97, 0.8 * 0.9))


def test_each_track_is_sampled_once_a_second():
    ts = [0.0, 0.3, 0.9, 1.0, 1.5, 2.2]  # track 1 every few frames; track 2 once
    rows = {
        "ts_ns": [round((1_800_000_000 + t) * F.NS) for t in ts] + [round((1_800_000_000 + 0.5) * F.NS)],
        "track_id": [1] * len(ts) + [2],
        "box": [BOX] * (len(ts) + 1),
        "joints": [joints(left_heel=(0.3, 0.9, 0.9))] * (len(ts) + 1),
    }
    got = F.sample("den", rows)
    assert [(f.track, round(f.t - 1_800_000_000, 1)) for f in got] == [(1, 0.0), (2, 0.5), (1, 1.0), (1, 2.2)]


def down_camera(x: float = 0.0, y: float = 0.0, height: float = 2.0) -> places.CameraPose:
    """A camera `height` m over (x, y) looking straight down (its +z to the world's -z), 100 px square, focal 100."""
    pose = (1, 0, 0, x, 0, -1, 0, y, 0, 0, -1, height, 0, 0, 0, 1)
    return places.CameraPose(tuple(float(v) for v in pose), 100.0, 100, 100)


FLOOR = (0.0, 0.0, 1.0, 0.0)  # z up, the floor at z = 0


def test_a_footprint_lands_where_its_pixel_ray_meets_the_floor():
    cams = {"den": down_camera(x=3.0)}
    cells = [
        # straight down: under the camera; half the width right: 0.5 m per metre of drop, to the feet's plane
        # (FOOT_M = 0.07 m up, people's heels) 1.93 m down, so 0.965 m right
        F.Cells("den", np.array([[0.5, 0.5], [1.0, 0.5]]), np.array([5, 2])),
        F.Cells("hall", np.array([[0.5, 0.5]]), np.array([9])),  # a camera not registered here: left out
    ]
    points, weights = F.project(cells, cams, FLOOR)
    np.testing.assert_allclose(points, [[3.0, 0.0, 0.0], [3.965, 0.0, 0.0]], atol=1e-9)  # on the floor under them
    assert weights.tolist() == [5, 2]  # each point stands for its cell's footprints
    assert F.project(cells, cams, None)[0].shape == (0, 3)  # no floor yet: nothing lands


# A camera 2 m up looking level along the world's +x (its +y down the image is the world's -z): 100 px, focal 100.
LEVEL = places.CameraPose(
    (0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0), 100.0, 100, 100
)


def test_a_person_lands_where_the_ray_meets_their_heels_plane_and_a_thing_on_the_floor_where_it_meets_the_floor():
    # LEVEL 2 m up, a pixel 0.25 down per metre ahead: the floor 8 m out; heels 0.07 m up, 7.72 m out
    assert places.floor_point(LEVEL, FLOOR, 0.5, 0.75) == pytest.approx((8.0, 0.0, 0.0))
    assert places.floor_point(LEVEL, FLOOR, 0.5, 0.75, 0.07) == pytest.approx((7.72, 0.0, 0.0))
    assert places.floor_point(LEVEL, (0.0, 0.0, 3.0, 0.0), 0.5, 0.75, 0.07) == pytest.approx((7.72, 0.0, 0.0)), (
        "a floor given with a normal not of length 1 still measures metres"
    )
    assert places.floor_point(LEVEL, FLOOR, 0.5, 0.75, 2.0) is None, "nothing at or above the camera's height"


def test_a_pixel_just_under_the_horizon_lands_nowhere_rather_than_far_away():
    near = F.Footprint("hall", 0.0, 0.5, 0.75, 1, 0.9)  # 0.25 down per metre ahead (14 degrees): the floor 8 m out
    far = F.Footprint("hall", 1.0, 0.5, 0.52, 1, 0.9)  # 0.02 down per metre: 100 m out, beyond MAX_FLOOR_M
    assert places.floor_point(LEVEL, FLOOR, near.u, near.v) == pytest.approx((8.0, 0.0, 0.0))
    assert places.floor_point(LEVEL, FLOOR, far.u, far.v) is None
    points, weights = F.project(
        [F.Cells("hall", np.array([[near.u, near.v], [far.u, far.v]]), np.array([1, 4]))], {"hall": LEVEL}, FLOOR
    )
    np.testing.assert_allclose(points, [[7.72, 0.0, 0.0]], atol=1e-9)  # a person's heels: 1.93 m down at 0.25 a metre
    assert weights.tolist() == [1]  # the far cell is dropped with its footprints


def test_a_ray_too_shallow_for_its_error_lands_nowhere_even_within_reach():
    # LEVEL's pixel v looks (v - 0.5) down per metre ahead. Just under MIN_RAY_DEG (12) the floor is still within
    # MAX_FLOOR_M, about 9.9 m out, but a degree of floor tilt moves it metres; just over, it's kept.
    shallow = 0.5 + math.tan(math.radians(places.MIN_RAY_DEG - 0.2))
    steep = 0.5 + math.tan(math.radians(places.MIN_RAY_DEG + 0.2))
    assert places.floor_point(LEVEL, FLOOR, 0.5, shallow) is None
    hit = places.floor_point(LEVEL, FLOOR, 0.5, steep)
    assert hit is not None and hit[0] == pytest.approx(2 / math.tan(math.radians(places.MIN_RAY_DEG + 0.2)))
    points, weights = F.project(
        [F.Cells("hall", np.array([[0.5, shallow], [0.5, steep]]), np.array([3, 5]))], {"hall": LEVEL}, FLOOR
    )
    np.testing.assert_allclose(points, [places.floor_point(LEVEL, FLOOR, 0.5, steep, F.FOOT_M)], atol=1e-9)
    assert weights.tolist() == [5]


def floor_and_furniture(rng: np.random.Generator) -> np.ndarray:
    """What a high camera sees of a 4 x 5 m room: its floor and a table, and no wall above the furniture."""
    floor = np.column_stack([rng.uniform(0, 4, 20000), rng.uniform(0, 5, 20000), rng.normal(0, 0.005, 20000)])
    table = np.column_stack([rng.uniform(1.5, 2.5, 3000), rng.uniform(2, 3, 3000), rng.uniform(0, 0.75, 3000)])
    return np.concatenate([floor, table])


def walked(rng: np.random.Generator) -> np.ndarray:
    """Feet all over the room, kept 0.4 m off its walls, round the table, and a few stray ones far away."""
    feet = np.column_stack([rng.uniform(0.4, 3.6, 8000), rng.uniform(0.4, 4.6, 8000), np.zeros(8000)])
    feet = feet[~((np.abs(feet[:, 0] - 2) < 0.6) & (np.abs(feet[:, 1] - 2.5) < 0.6))]
    stray = np.column_stack([rng.uniform(-15, 15, 20), rng.uniform(-15, 15, 20), np.zeros(20)])
    return np.concatenate([feet, stray])


def test_where_people_walk_closes_a_room_the_cloud_has_no_walls_for():
    rng = np.random.default_rng(0)
    cloud = floor_and_furniture(rng)
    with pytest.raises(ValueError):
        fp.floorplan(cloud, floor=FLOOR)  # the cloud alone: no wall, no room
    plan = fp.floorplan(cloud, floor=FLOOR, walkable=walked(rng))
    (room,) = plan.rooms  # the stray feet make no rooms of their own
    xs, ys = [x for x, _ in room.polygon], [y for _, y in room.polygon]
    sides = sorted([max(xs) - min(xs), max(ys) - min(ys)])
    # The walked floor reaches WALK_PAD_M past the feet and is squared off to WALK_BLOCK_M, so its walls land within
    # about a block of the truth.
    assert sides == pytest.approx([4.0, 5.0], abs=fp.WALK_BLOCK_M)
    assert len(room.polygon) == 4


def test_a_given_floor_is_used_instead_of_one_found():
    rng = np.random.default_rng(1)
    cloud = floor_and_furniture(rng)
    tilted = (math.sin(0.02), 0.0, math.cos(0.02), 0.0)  # a floor a degree off the cloud's: heights follow it
    plan = fp.floorplan(cloud, floor=tilted, walkable=walked(rng))
    assert plan.transform[2, :3] == pytest.approx(tilted[:3])


NOW = 1_800_000_000.0


@pytest.fixture
def tracks(monkeypatch, tmp_path):
    """pose.track stood in for at the read: one footprint 10 s into each window asked for; the windows asked for.
    The position layer is the real segment store under tmp_path (the suite otherwise reads an in-memory one)."""
    monkeypatch.setattr(segment_log, "at", lambda state=None: segment_log.Log(segment_log.root(tmp_path)))
    asked = []

    def collect(state, start, end, cameras):
        asked.append((start, end, tuple(cameras)))
        return [F.Footprint(cameras[0], start + 10, 0.4, 0.9, 7, 0.8)]

    monkeypatch.setattr(F, "collect", collect)
    return asked


def test_footprints_go_into_the_position_layer_once_and_read_back_as_written(tmp_path, tracks):
    assert F.record(tmp_path, ["den", "hall"], NOW) == 2
    assert F.record(tmp_path, ["den"], NOW + 3600) == 1  # from where the last run stopped, never again from before
    upto = NOW - F.FLUSH_S  # rows this recent may not be on disk yet: the next run reads them
    assert tracks == [
        (upto - F.FIRST_S, upto, ("den",)),
        (upto - F.FIRST_S, upto, ("hall",)),
        (upto, upto + 3600, ("den",)),
    ]
    (den,) = F.walked(tmp_path, ["den"], NOW + 3600)
    # both of den's footprints fell at (0.4, 0.9): one cell, counted twice, its centre within a cell of the point
    assert (den.camera, den.counts.tolist()) == ("den", [2])
    assert den.uv[0] == pytest.approx((0.4, 0.9), abs=F.CELL)
    assert F.walked(tmp_path, ["hall"], NOW + 3600, days=0.5) == []  # hall's one footprint is a day old


def test_the_walkable_read_turns_many_footprints_into_few_counted_cells(tmp_path):
    from liberational.spatial import positions

    # A person standing still for 10 minutes (one footprint a second, jittering by under a cell), and one step away.
    rng = np.random.default_rng(5)
    still = [
        dict(
            t=NOW + i,
            source=positions.FIXED_CAMERA,
            confidence=0.9,
            track="den:7",
            camera="den",
            u=0.3 + rng.uniform(0, F.CELL / 4),
            v=0.6 + rng.uniform(0, F.CELL / 4),
        )
        for i in range(600)
    ]
    step = dict(t=NOW + 700, source=positions.FIXED_CAMERA, confidence=0.9, track="den:7", camera="den", u=0.5, v=0.6)
    positions.write_samples(tmp_path, [*still, step])
    (den,) = F.walked(tmp_path, ["den"], NOW + 3600)
    assert sorted(den.counts.tolist()) == [1, 600]  # 601 footprints, two cells


def test_weighted_walkable_points_make_the_plan_the_repeated_ones_would():
    # Each footprint counted by its cell's weight is the same evidence as that footprint repeated (the concern on
    # A lone foot carrying a weight is many feet there, not a busy cell from one).
    rng = np.random.default_rng(6)
    cloud = floor_and_furniture(rng)
    feet = walked(rng)[::4]
    repeated = np.repeat(feet, 3, axis=0)
    by_points = fp.floorplan(cloud, floor=FLOOR, walkable=repeated)
    by_weights = fp.floorplan(cloud, floor=FLOOR, walkable=feet, walk_weights=np.full(len(feet), 3.0))
    assert [r.polygon for r in by_weights.rooms] == [r.polygon for r in by_points.rooms]
    np.testing.assert_allclose(by_weights.transform, by_points.transform)


def test_a_camera_not_recorded_for_long_picks_up_a_week_back_at_most(tmp_path, tracks):
    F.record(tmp_path, ["den"], NOW)
    tracks.clear()
    F.record(tmp_path, ["den"], NOW + 30 * F.DAY_S)
    upto = NOW + 30 * F.DAY_S - F.FLUSH_S
    assert tracks == [(upto - F.MAX_BACKLOG_S, upto, ("den",))]


def test_an_animals_footprint_is_its_box_bottom_centre_once_a_second_at_half_its_score():
    s = 1_800_000_000
    rows = {
        "ts_ns": [round((s + t) * F.NS) for t in (0.0, 0.4, 1.2, 0.5, 0.7)],
        "track_id": ["r:cat:1", "r:cat:1", "r:cat:1", "r:dog:2", "r:dog:2"],
        "box": [
            [0.2, 0.5, 0.4, 0.8, 0.9],
            [0.2, 0.5, 0.4, 0.8, 0.9],  # 0.4 s after the last: not sampled
            [0.3, 0.5, 0.5, 0.7, 0.6],
            [0.1, 0.6, 0.3, 0.99, 0.8],  # cut off by the frame's bottom edge: its feet aren't in view
            [0.1, 0.6, 0.3, 0.9, 0.8],
        ],
    }
    got = [(f.track, round(f.t - s, 1), f.u, f.v, f.confidence) for f in F.sample_animals("den", rows)]
    assert got == [
        ("r:cat:1", 0.0, pytest.approx(0.3), pytest.approx(0.8), pytest.approx(0.45)),
        ("r:dog:2", 0.7, pytest.approx(0.2), pytest.approx(0.9), pytest.approx(0.4)),
        ("r:cat:1", 1.2, pytest.approx(0.4), pytest.approx(0.7), pytest.approx(0.3)),
    ]


def test_animals_seen_by_a_camera_are_collected_beside_its_people(tmp_path):
    import uuid

    writers = segment_log.Writers(segment_log.root(tmp_path))
    record = dict(frame_id=str(uuid.uuid7()), track_id="r:cat:1", species="cat", verified=True, model="test")
    writers.append(segment_log.ANIMAL_DETECTIONS, "den", [{**record, "ts_ns": round((NOW + 5) * F.NS), "box": BOX}])
    writers.append(segment_log.ANIMAL_DETECTIONS, "hall", [{**record, "ts_ns": round((NOW + 6) * F.NS), "box": BOX}])
    writers.close()
    got = F.collect(tmp_path, NOW, NOW + 60, ["den"])
    assert [(f.camera, f.track, f.u, f.v) for f in got] == [("den", "r:cat:1", pytest.approx(0.3), pytest.approx(0.9))]


def test_the_walkable_read_leaves_animals_footprints_out(tmp_path):
    from liberational.spatial import positions

    row = dict(source=positions.FIXED_CAMERA, confidence=0.9, camera="den", v=0.6)
    positions.write_samples(
        tmp_path,
        [
            {**row, "t": NOW, "track": "den:7", "u": 0.3},
            {**row, "t": NOW + 1, "track": "den:r:cat:1", "u": 0.7},  # a cat on the sofa isn't on the floor
        ],
    )
    (den,) = F.walked(tmp_path, ["den"], NOW + 3600)
    assert den.counts.tolist() == [1]
    assert den.uv[0] == pytest.approx((0.3, 0.6), abs=F.CELL)


@pytest.mark.parametrize(
    "head,expected",
    [
        pytest.param((0.3, 0.15, 0.9), (0.3, 0.15), id="placed"),
        pytest.param((0.3, 0.15, 0.1), None, id="unsure"),
        pytest.param(None, None, id="not-placed"),
    ],
)
def test_a_sighting_has_the_feet_and_the_head_only_when_pose_was_sure_of_it(head, expected):
    placed = {"left_heel": (0.25, 0.88, 0.9), "right_heel": (0.35, 0.86, 0.9)}
    if head is not None:
        placed["head"] = head
    rows = {"ts_ns": [5 * F.NS], "track_id": [7], "box": [BOX], "joints": [joints(**placed)]}
    (got,) = F.sightings_of("cam", rows)
    assert (got.camera, got.t, got.track) == ("cam", 5.0, 7)
    assert (got.u, got.v) == pytest.approx((0.30, 0.87))
    assert got.head == (pytest.approx(expected) if expected else None)


@pytest.mark.parametrize(
    "placed,upright",
    [
        pytest.param(
            {"head": (0.3, 0.1, 0.9), "left_hip": (0.3, 0.5, 0.9), "left_heel": (0.3, 0.88, 0.9)}, True, id="standing"
        ),
        pytest.param(
            {"head": (0.3, 0.1, 0.9), "left_hip": (0.3, 0.75, 0.9), "left_heel": (0.3, 0.88, 0.9)}, False, id="sitting"
        ),
        pytest.param({"left_hip": (0.3, 0.5, 0.9), "left_heel": (0.3, 0.88, 0.9)}, False, id="no-head"),
    ],
)
def test_a_sighting_is_upright_only_with_its_feet_hips_and_head_placed_and_long_legs(placed, upright):
    # Hips at 0.5 between a head at 0.1 and heels at 0.88: legs 49% of the body's height in the frame; at 0.75, 17%
    rows = {"ts_ns": [5 * F.NS], "track_id": [7], "box": [[0.2, 0.05, 0.4, 0.9, 0.8]], "joints": [joints(**placed)]}
    (got,) = F.sightings_of("cam", rows)
    assert got.upright == upright


def test_a_foot_point_is_at_an_edge_where_what_its_camera_sees_behind_it_hides_enough_floor():
    # LEVEL, 2 m up looking level: the bottom row drops 0.5 m a metre, the row at v = 0.75 0.25 m. Its frame's
    # top half sees a couch's back 0.9 m up (0.45 of 2 m), the bottom left a chair 0.4 m up, the bottom right nothing
    # sure.
    cam = replace(LEVEL, background=places.Background(2, 2, (0.45, 0.45, 0.2, None)))
    # v = 0.25 looks above the horizon: no floor, unknown
    assert F.at_edge(cam, FLOOR, 0.25, 0.25) is None
    shallow = replace(LEVEL, background=places.Background(1, 1, (0.45,)))
    # at v = 0.75 the couch hides (0.9 - 0.07) / 0.25 = 3.32 m of floor; at v = 1.0, 1.66 m: both over EDGE_M
    assert F.at_edge(shallow, FLOOR, 0.5, 0.75) is True
    assert F.at_edge(shallow, FLOOR, 0.5, 0.999) is True
    # the chair, from the bottom row: (0.4 - 0.07) / 0.5 = 0.66 m, under it: a guess there is off by little
    assert F.at_edge(cam, FLOOR, 0.25, 0.999) is False
    assert F.at_edge(cam, FLOOR, 0.75, 0.999) is None, "nothing sure behind the foot"
    assert F.at_edge(LEVEL, FLOOR, 0.5, 0.75) is None, "no background"
    assert F.at_edge(shallow, None, 0.5, 0.75) is None, "no floor"
    # The same couch seen from straight overhead hides no floor along the ray: not an edge
    above = replace(down_camera(height=2.5), background=places.Background(1, 1, (0.36,)))
    assert F.at_edge(above, FLOOR, 0.5, 0.5) is False


def test_a_surface_given_for_an_unseen_cell_is_judged_as_the_background_would_be():
    # The bottom right cell is unknown. From the bottom row (dropping 0.5 m a metre) a 1 m surface would hide
    # (1.0 - 0.07) / 0.5 = 1.86 m of floor, a 0.4 m one 0.66 m
    cam = replace(LEVEL, background=places.Background(2, 2, (0.45, 0.45, 0.2, None)))
    assert F.at_edge(cam, FLOOR, 0.75, 0.999, surface=1.0) is True
    assert F.at_edge(cam, FLOOR, 0.75, 0.999, surface=0.4) is False
    assert F.at_edge(LEVEL, FLOOR, 0.5, 0.75, surface=1.0) is True, "a camera with no background at all"
    assert F.at_edge(cam, None, 0.75, 0.999, surface=1.0) is None, "no floor"


def test_the_walked_floor_leaves_out_feet_on_an_occluding_edge():
    # LEVEL's bottom row (dropping 0.5 m a metre) over a 0.4 m chair hides 0.66 m of floor: kept. Over a cell
    # no registration saw, judged as FURNITURE_M (1 m) would hide 1.86 m: left out, as positions leaves it uncertain.
    cam = replace(LEVEL, background=places.Background(2, 2, (0.45, 0.45, 0.2, None)))
    cells = [F.Cells("hall", np.array([[0.25, 0.999], [0.75, 0.999]]), np.array([3, 4]))]
    points, weights = F.project(cells, {"hall": cam}, FLOOR)
    assert weights.tolist() == [3]
    np.testing.assert_allclose(points, [places.floor_point(cam, FLOOR, 0.25, 0.999, F.FOOT_M)], atol=1e-9)
    # a camera with no background (registered alone) has no edges to judge by: both kept
    assert F.project(cells, {"hall": LEVEL}, FLOOR)[1].tolist() == [3, 4]


heights = st.one_of(st.none(), st.floats(0.0, 0.95))


@given(
    background=st.lists(heights, min_size=9, max_size=9),
    u=st.floats(0.0, 0.999),
    v=st.floats(0.6, 0.999),
    pitch=st.floats(0.0, 0.6),
)
def test_a_walked_cell_is_left_out_exactly_when_at_edge_says_its_foot_is_on_one(background, u, v, pitch):
    # LEVEL pitched down by `pitch` radians about its x axis, so rays fall at many slopes
    c, s = math.cos(pitch), math.sin(pitch)
    m = np.asarray(LEVEL.pose, dtype=np.float64).reshape(4, 4)
    m[:3, :3] = m[:3, :3] @ np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    seen = places.Background(3, 3, tuple(background))
    cam = replace(LEVEL, pose=tuple(float(x) for x in m.ravel()), background=seen)
    cell = [F.Cells("hall", np.array([[u, v]]), np.array([1]))]
    lands = len(F.project(cell, {"hall": replace(cam, background=None)}, FLOOR)[1]) == 1
    unseen = seen.at(u, v) is None
    edge = F.at_edge(cam, FLOOR, u, v, F.FURNITURE_M if unseen else None) is True
    assert len(F.project(cell, {"hall": cam}, FLOOR)[1]) == (lands and not edge)


def grid(fill: float | None, **cells: float | None) -> places.Background:
    """A 20 by 20 background of one height, with `cells` ("c<i>": height) set apart."""
    heights = [fill] * 400
    for name, h in cells.items():
        heights[int(name[1:])] = h
    return places.Background(20, 20, tuple(heights))


def test_an_unseen_cell_takes_the_newest_earlier_background_of_the_same_view_that_saw_it():
    tonight = grid(0.45, c95=None, c3=None)  # someone sat at cell 95 in tonight's frame; nothing saw cell 3 either
    moved = grid(0.0, c95=0.9)  # the camera aimed elsewhere: none of its cells agree
    newer = grid(0.45, c95=None, c3=0.5)  # same view; someone sat there that night too
    older = grid(0.47, c95=0.6, c3=0.7)  # same view, within 0.1 of tonight's everywhere
    got = F.filled(tonight, [moved, newer, older])
    assert (got.heights[95], got.heights[3]) == (0.6, 0.5)
    assert got.heights[:3] == (0.45, 0.45, 0.45)


@pytest.mark.parametrize(
    "earlier",
    [
        pytest.param([grid(0.0, c95=0.9)], id="another-view"),
        pytest.param([places.Background(5, 5, (0.45,) * 25)], id="another-grid"),
        pytest.param([grid(None, c95=0.6, c0=0.45)], id="too-few-cells-to-compare"),
        pytest.param([], id="none-earlier"),
    ],
)
def test_an_unseen_cell_stays_unknown_without_an_earlier_background_of_the_same_view(earlier):
    assert F.filled(grid(0.45, c95=None), earlier).heights[95] is None


FULL = grid(0.45)
SPARSE = grid(None, c0=0.45, c1=0.45, c2=0.9)  # knows 3 cells: too few to tell its view


@pytest.mark.parametrize(
    ("new", "old", "want"),
    [
        pytest.param(SPARSE, FULL, FULL, id="sparse-tonight-keeps-the-full-one"),
        pytest.param(FULL, SPARSE, FULL, id="full-tonight-wins-over-sparse"),
        pytest.param(grid(0.45, c95=None), FULL, FULL, id="same-view-gaps-filled"),
        pytest.param(grid(0.0, c95=None), FULL, grid(0.0, c95=None), id="re-aimed-keeps-tonights-own"),
        pytest.param(None, FULL, FULL, id="none-tonight-keeps-the-old"),
        pytest.param(SPARSE, None, SPARSE, id="nothing-before"),
        pytest.param(None, None, None, id="none-either"),
    ],
)
def test_a_reregistered_cameras_background_is_never_left_sparser_than_its_view_had(new, old, want):
    assert F.kept(new, old) == want
