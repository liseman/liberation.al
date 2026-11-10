"""Reconstruction output to a plan whose rooms tile: gaps, overlaps, skew, unseen parts, and openings on their
walls. Each case is also written as a scene and read back (`problems`), which is what a stored plan must survive."""

import json
import math

import numpy as np
import pytest

from liberational.spatial import floorplan as fp
from liberational.spatial import plan_import, plan_scene
from liberational.spatial import tiling as T


def problems(t: T.Tiled) -> list[str]:
    """What goes wrong writing `t` as a scene and reading it back: an opening that fits no wall of the tiled rooms,
    or rooms or openings that don't come back. Empty for a plan that tiles."""
    before = len(t.notes)
    text = T.to_scene(t, "Plan")
    found = list(t.notes[before:])
    plan = plan_import.parse(text)
    if [r.id for r in plan.rooms] != [r.id for r in t.rooms]:
        found.append(f"rooms read back as {[r.id for r in plan.rooms]}")
    if len(plan.openings) != len(t.openings):
        found.append(f"{len(plan.openings)} of {len(t.openings)} openings read back")
    # every stretch of wall has a room on at least one side, and no two rooms claim the same floor
    area = sum(abs(T._area(r.polygon)) for r in t.rooms)
    xs = [x for r in t.rooms for x, _ in r.polygon]
    ys = [y for r in t.rooms for _, y in r.polygon]
    if area > (max(xs) - min(xs)) * (max(ys) - min(ys)) + 1e-6:
        found.append("the rooms overlap")
    return found


def rect(x0: float, y0: float, x1: float, y1: float) -> tuple[T.Point, ...]:
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def box(room: T.Room) -> tuple[float, float, float, float]:
    xs, ys = [x for x, _ in room.polygon], [y for _, y in room.polygon]
    return (min(xs), min(ys), max(xs), max(ys))


def test_two_rooms_inner_faces_meet_on_their_shared_centreline_with_the_door_on_it():
    # Truth on centrelines: A is 0,0 to 4,3 and B 4,0 to 7,3. A room model sees the faces: half an exterior wall
    # (0.1) in from the outside, half a partition (0.05) back from the wall they share.
    rooms = [T.Room("a", rect(0.1, 0.1, 3.95, 2.9)), T.Room("b", rect(4.05, 0.1, 6.9, 2.9))]
    door = T.Opening("door", (4.02, 1.4), 0.9)
    t = T.tile(rooms, [door])
    assert [box(r) for r in t.rooms] == [pytest.approx((0, 0, 4, 3)), pytest.approx((4, 0, 7, 3))]
    assert [(o.between, o.at, o.width) for o in t.openings] == [(("a", "b"), pytest.approx((4.0, 1.4)), 0.9)]
    assert problems(t) == []


def test_overlapping_rooms_are_split_on_one_line():
    rooms = [T.Room("a", rect(0, 0, 4.06, 3)), T.Room("b", rect(3.96, 0, 7, 3))]  # 10 cm of overlap
    t = T.tile(rooms, settings=T.Settings(faces=False))
    a, b = (box(r) for r in t.rooms)
    assert a[2] == b[0] == pytest.approx(4.01)  # one shared line, between the two edges
    assert problems(t) == []


def test_skewed_rooms_are_squared_up():
    turn = math.radians(4)

    def turned(poly):
        return tuple(
            (math.cos(turn) * x - math.sin(turn) * y, math.sin(turn) * x + math.cos(turn) * y) for x, y in poly
        )

    rooms = [T.Room("a", turned(rect(0, 0, 4, 3))), T.Room("b", turned(rect(4, 0, 7, 3)))]
    t = T.tile(rooms, settings=T.Settings(faces=False))
    assert t.angle == pytest.approx(-turn)
    sizes = [(round(b[2] - b[0], 6), round(b[3] - b[1], 6)) for b in map(box, t.rooms)]
    assert sizes == [(4.0, 3.0), (3.0, 3.0)]
    assert problems(t) == []


def ring(hole: tuple[float, float, float, float]) -> list[T.Room]:
    """Four rooms round a 6 x 6 m square, leaving `hole` (x0, y0, x1, y1) inside them uncovered."""
    x0, y0, x1, y1 = hole
    return [
        T.Room("s", rect(0, 0, 6, y0)),
        T.Room("w", rect(0, y0, x0, y1)),
        T.Room("e", rect(x1, y0, 6, y1)),
        T.Room("n", rect(0, y1, 6, 6)),
    ]


def test_a_small_gap_the_rooms_enclose_goes_to_a_neighbour():
    t = T.tile(ring((2.5, 2.5, 3.5, 3.5)), settings=T.Settings(faces=False))  # 1 m2
    assert t.unexplored == []
    assert sum(abs(T._area(r.polygon)) for r in t.rooms) == pytest.approx(36)
    assert problems(t) == []


def test_a_large_gap_the_rooms_enclose_is_a_part_not_seen_yet():
    t = T.tile(ring((2, 2, 4, 4)), settings=T.Settings(faces=False))  # 4 m2: a closed door, say
    (area,) = t.unexplored
    assert abs(T._area(area)) == pytest.approx(4)
    unseen = plan_scene.records(json.loads(T.to_scene(t, "Large gap")))["unexplored"]
    assert unseen == {"u1": {"name": T.UNSEEN, "poly": [[2, 2], [4, 2], [4, 4], [2, 4]]}}
    assert problems(t) == []


def test_a_sliver_room_goes_to_a_neighbour():
    rooms = [T.Room("a", rect(0, 0, 4, 3)), T.Room("bit", rect(4, 0, 4.4, 2)), T.Room("b", rect(4, 2, 7, 3))]
    t = T.tile(rooms, settings=T.Settings(faces=False))
    assert [r.id for r in t.rooms] == ["a", "b"]
    assert any("bit" in n for n in t.notes)


@pytest.mark.parametrize(
    "at,width,expected_at,expected_width",
    [
        pytest.param((4.0, 1.2), 1.5, (4.0, 0.75), 1.3, id="wider-than-its-wall"),  # the 1.5 m wall takes 1.3
        pytest.param((4.0, 0.3), 0.9, (4.0, 0.55), 0.9, id="too-near-a-corner"),  # slid so 0.1 m of wall is left
    ],
)
def test_an_opening_is_clamped_to_its_wall(at, width, expected_at, expected_width):
    # A and B share a 1.5 m wall (B is the short room on A's east side, from y = 0 to 1.5).
    rooms = [T.Room("a", rect(0, 0, 4, 3)), T.Room("b", rect(4, 0, 6, 1.5)), T.Room("c", rect(4, 1.5, 6, 3))]
    t = T.tile(rooms, [T.Opening("door", at, width)], T.Settings(faces=False))
    ((between, placed, w),) = [(o.between, o.at, o.width) for o in t.openings]
    assert (between, placed, w) == (("a", "b"), pytest.approx(expected_at), pytest.approx(expected_width))
    assert problems(t) == []


def test_an_opening_on_no_wall_is_dropped_and_said_so():
    t = T.tile([T.Room("a", rect(0, 0, 4, 3))], [T.Opening("window", (2, 1.5), 1.0)], T.Settings(faces=False))
    assert t.openings == []
    assert "dropped" in t.notes[0]


# Three rooms with stepped outlines and a notch between two of them.
STEPPED = (
    (
        (0, 0),
        (5.6, 0),
        (5.6, 0.5),
        (8, 0.5),
        (8, 2.4),
        (7.2, 2.4),
        (7.2, 3.4),
        (6.6, 3.4),
        (6.6, 5.4),
        (6.2, 5.4),
        (6.2, 6),
        (0, 6),
    ),
    ((8, 0), (11, 0), (11, 0.5), (11.5, 0.5), (11.5, 1.4), (8, 1.4)),
    ((6.6, 3.8), (7.6, 3.8), (7.6, 4.4), (8.5, 4.4), (8.5, 4.9), (9, 4.9), (9, 6), (6.6, 6)),
)


def test_a_stepped_plan_with_doors_wider_than_their_walls_comes_out_clean():
    # A made-up plan shaped like what a reconstruction writes: stepped outlines, a notch between two rooms, and doors
    # wider than the wall they're on, which overflowed their walls in the library before openings were clamped.
    rooms = [
        T.Room("r1", STEPPED[0], "Room 1"),
        T.Room("r2", STEPPED[1], "Room 2"),
        T.Room("r3", STEPPED[2], "Room 3"),
    ]
    openings = [
        T.Opening("door", (0.0, 3.0), 1.5, ("exterior", "r1")),
        T.Opening("door", (8.0, 1.05), 1.0, ("r2", "r1")),
        T.Opening("door", (6.6, 4.6), 1.5, ("r1", "r3")),
    ]
    t = T.tile(rooms, openings, T.Settings(faces=False))
    assert len(t.rooms) == 3
    # The outside door fits its 6 m wall. Rooms 1 and 2 share 0.9 m of wall (y 0.5 to 1.4) and rooms 1 and 3 share
    # 1.6 m (y 3.8 to 5.4): each door keeps 0.1 m of wall at both ends, and the first slides to that wall's middle.
    assert [(o.between, o.at, o.width) for o in t.openings] == [
        (("exterior", "r1"), pytest.approx((0.0, 3.0)), pytest.approx(1.5)),
        (("r1", "r2"), pytest.approx((8.0, 0.95)), pytest.approx(0.7)),
        (("r1", "r3"), pytest.approx((6.6, 4.6)), pytest.approx(1.4)),
    ]
    assert problems(t) == []


def test_the_baseline_plan_of_a_scan_comes_out_clean():
    import test_spatial_floorplan as S

    plan = fp.floorplan(S.scan(S.TWO_ROOMS, [("door", (4.0, 1.5), 0.9)], np.random.default_rng(2)))
    rooms, openings = T.from_plan(plan)
    t = T.tile(rooms, openings, T.Settings(faces=False))
    assert len(t.rooms) == 2 and [o.between for o in t.openings] == [("r1", "r2")]
    assert problems(t) == []


def test_the_scene_carries_rooms_unseen_parts_and_openings_and_reads_back():
    t = T.tile(ring((2, 2, 4, 4)), [T.Opening("door", (1.0, 2.0), 0.9)], T.Settings(faces=False))
    text = T.to_scene(t, "Ring", T.Settings(faces=False))
    plan = plan_import.parse(text)
    assert plan.title == "Ring" and sorted(r.id for r in plan.rooms) == ["e", "n", "s", "w"]
    assert [(o.kind, frozenset(o.between.split(">")), o.at, o.width) for o in plan.openings] == [
        ("door", frozenset({"s", "w"}), pytest.approx((1.0, 2.0)), 0.9)
    ]
    assert list(plan_scene.records(json.loads(text))["unexplored"]) == ["u1"]


def test_an_opening_the_tiled_rooms_have_no_wall_for_is_left_out_of_the_scene_and_said_so():
    t = T.Tiled([T.Room("a", rect(0, 0, 4, 3), "A")], [T.Opening("door", (9.0, 9.0), 0.9)], [])
    text = T.to_scene(t, "Stray")
    assert plan_import.parse(text).openings == ()
    assert t.notes == ["door at 9.00,9.00 is on no wall of the tiled rooms: dropped"]


def test_a_door_into_a_part_not_seen_yet_is_dropped_not_made_an_outside_door():
    # w's east wall is the not-seen-yet area's west wall; there is no floor there to open onto.
    t = T.tile(ring((2, 2, 4, 4)), [T.Opening("door", (2.0, 3.0), 0.9)], T.Settings(faces=False))
    assert t.openings == []
    assert "not seen yet" in t.notes[0]
    assert problems(t) == []
