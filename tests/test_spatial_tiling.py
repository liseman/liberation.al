"""Reconstruction output to a plan the floorplan library accepts: gaps, overlaps, skew, unseen parts, and
openings on their walls. Each case is also checked with the vendored library's own linter (frontend/src/vendor/
floorplan) through node, keeping the converter's rules: tiling, walls, and openings (the architect checks, such as a
missing entrance, are noise for an observed place)."""

import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from liberational.spatial import floorplan as fp
from liberational.spatial import tiling as T

LIBRARY = Path(__file__).resolve().parents[1] / "frontend" / "src" / "vendor" / "floorplan" / "index.js"
FIXTURE = Path(__file__).with_name("data") / "stepped_rooms.plan"
KEPT = ("tiling.", "wall.", "opening.")


def lint(doc: dict) -> list[str]:
    """The library's findings for `doc` that the converter answers for, as "rule: message"."""
    if shutil.which("node") is None:
        pytest.skip("needs node for the library's linter")
    script = (
        f"import {{ lint }} from {json.dumps(str(LIBRARY))};"
        "let s = ''; process.stdin.on('data', d => s += d);"
        "process.stdin.on('end', () => console.log(JSON.stringify(lint(JSON.parse(s)).findings)));"
    )
    done = subprocess.run(
        ["node", "--input-type=module", "-e", script], input=json.dumps(doc), capture_output=True, text=True, check=True
    )
    return [f"{f['rule']}: {f['message']}" for f in json.loads(done.stdout) if f["rule"].startswith(KEPT)]


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
    assert lint(T.to_document(t, "Two rooms")) == []


def test_overlapping_rooms_are_split_on_one_line():
    rooms = [T.Room("a", rect(0, 0, 4.06, 3)), T.Room("b", rect(3.96, 0, 7, 3))]  # 10 cm of overlap
    t = T.tile(rooms, settings=T.Settings(faces=False))
    a, b = (box(r) for r in t.rooms)
    assert a[2] == b[0] == pytest.approx(4.01)  # one shared line, between the two edges
    assert lint(T.to_document(t, "Overlap")) == []


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
    assert lint(T.to_document(t, "Skewed")) == []


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
    assert lint(T.to_document(t, "Small gap")) == []


def test_a_large_gap_the_rooms_enclose_is_a_part_not_seen_yet():
    t = T.tile(ring((2, 2, 4, 4)), settings=T.Settings(faces=False))  # 4 m2: a closed door, say
    (area,) = t.unexplored
    assert abs(T._area(area)) == pytest.approx(4)
    doc = T.to_document(t, "Large gap")
    assert doc["voids"]["u1"]["name"] == doc["x-unexplored"]["u1"]["name"] == T.UNSEEN
    assert lint(doc) == []  # without the void the library calls it a hole in the plan


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
    assert lint(T.to_document(t, "Clamped")) == []


def test_an_opening_on_no_wall_is_dropped_and_said_so():
    t = T.tile([T.Room("a", rect(0, 0, 4, 3))], [T.Opening("window", (2, 1.5), 1.0)], T.Settings(faces=False))
    assert t.openings == []
    assert "dropped" in t.notes[0]


def test_a_stepped_plan_with_doors_wider_than_their_walls_comes_out_clean():
    # A made-up plan shaped like what a reconstruction writes: stepped outlines, a notch between two rooms, and doors
    # wider than the wall they're on, which overflowed their walls in the library before openings were clamped.
    rooms, openings = T.read_dsl(FIXTURE.read_text())
    assert len(rooms) == 3 and len(openings) == 3
    t = T.tile(rooms, openings, T.Settings(faces=False))
    assert len(t.rooms) == 3
    # The outside door fits its 6 m wall. Rooms 1 and 2 share 0.9 m of wall (y 0.5 to 1.4) and rooms 1 and 3 share
    # 1.6 m (y 3.8 to 5.4): each door keeps 0.1 m of wall at both ends, and the first slides to that wall's middle.
    assert [(o.between, o.at, o.width) for o in t.openings] == [
        (("exterior", "r1"), pytest.approx((0.0, 3.0)), pytest.approx(1.5)),
        (("r1", "r2"), pytest.approx((8.0, 0.95)), pytest.approx(0.7)),
        (("r1", "r3"), pytest.approx((6.6, 4.6)), pytest.approx(1.4)),
    ]
    assert lint(T.to_document(t, "Example")) == []


def test_the_baseline_plan_of_a_scan_comes_out_clean():
    import test_spatial_floorplan as S

    plan = fp.floorplan(S.scan(S.TWO_ROOMS, [("door", (4.0, 1.5), 0.9)], np.random.default_rng(2)))
    rooms, openings = T.from_plan(plan)
    t = T.tile(rooms, openings, T.Settings(faces=False))
    assert len(t.rooms) == 2 and [o.between for o in t.openings] == [("r1", "r2")]
    assert lint(T.to_document(t, "Scan")) == []


def test_the_dsl_carries_rooms_voids_and_openings_and_reads_back():
    t = T.tile(ring((2, 2, 4, 4)), [T.Opening("door", (1.0, 2.0), 0.9)], T.Settings(faces=False))
    text = T.to_dsl(t, "Ring", T.Settings(faces=False))
    assert 'void u1 "Not seen yet" rect 2,2 2x2' in text
    rooms, openings = T.read_dsl(text)
    assert sorted(r.id for r in rooms) == ["e", "n", "s", "w"]
    assert [(o.between, o.at, o.width) for o in openings] == [(("s", "w"), (1.0, 2.0), 0.9)]


def test_a_door_into_a_part_not_seen_yet_is_dropped_not_made_an_outside_door():
    # w's east wall is the not-seen-yet void's west wall; the library refuses any opening into a void.
    t = T.tile(ring((2, 2, 4, 4)), [T.Opening("door", (2.0, 3.0), 0.9)], T.Settings(faces=False))
    assert t.openings == []
    assert "not seen yet" in t.notes[0]
    assert lint(T.to_document(t, "Into the unseen")) == []
