"""The classical floorplan baseline against synthetic scans of known layouts.

Each synthetic cloud is what a walkthrough would see: wall faces from inside each room (an exterior wall only on
its inner face, a partition on both), floor and ceiling, door and window gaps at their heights, Gaussian noise,
furniture below the wall band, stray outliers, and a random rigid pose. The truth is the layout's centreline
polygons; a recovered plan matches when, up to a quarter turn (the Manhattan frame is only known modulo 90 degrees)
and a translation (the plan is normalised to start at 0,0), every room's corners are within TOL of the truth's.
"""

import math

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from liberational.spatial import floorplan as fp
from liberational.spatial import ply

TOL = 0.1  # metres per corner: two density cells
EXT, PART, HEIGHT = 0.2, 0.1, 2.6

Poly = list[tuple[float, float]]


def rect(x: float, y: float, w: float, h: float) -> Poly:
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def _on_boundary(points: np.ndarray, poly: Poly, tol: float = 1e-6) -> np.ndarray:
    hit = np.zeros(len(points), dtype=bool)
    for (ax, ay), (bx, by) in zip(poly, poly[1:] + poly[:1], strict=True):
        if ax == bx:
            hit |= (
                (np.abs(points[:, 0] - ax) < tol)
                & (points[:, 1] >= min(ay, by) - tol)
                & (points[:, 1] <= max(ay, by) + tol)
            )
        else:
            hit |= (
                (np.abs(points[:, 1] - ay) < tol)
                & (points[:, 0] >= min(ax, bx) - tol)
                & (points[:, 0] <= max(ax, bx) + tol)
            )
    return hit


def _inside(points: np.ndarray, poly: Poly) -> np.ndarray:
    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    for (ax, ay), (bx, by) in zip(poly, poly[1:] + poly[:1], strict=True):
        crosses = (ay > y) != (by > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            at = ax + (y - ay) * (bx - ax) / (by - ay)
        inside ^= crosses & (x < at)
    return inside


def scan(
    rooms: list[Poly],
    openings: list[tuple[str, tuple[float, float], float]],
    rng: np.random.Generator,
    *,
    noise: float = 0.01,
    clutter: bool = True,
    angle: float = 0.0,
) -> np.ndarray:
    """A z-up cloud of rooms (counter-clockwise centreline polygons) with openings (kind, centre on a wall line,
    width), rotated by `angle` about up and moved off the origin."""
    parts = []
    for r, poly in enumerate(rooms):
        others = [p for i, p in enumerate(rooms) if i != r]
        for (ax, ay), (bx, by) in zip(poly, poly[1:] + poly[:1], strict=True):
            length = math.hypot(bx - ax, by - ay)
            n = int(length * HEIGHT * 400)
            t, z = rng.uniform(0, 1, n), rng.uniform(0, HEIGHT, n)
            centre = np.stack([ax + t * (bx - ax), ay + t * (by - ay)], axis=1)
            shared = np.zeros(n, dtype=bool)
            for other in others:
                shared |= _on_boundary(centre, other)
            inward = np.array([-(by - ay), bx - ax]) / length  # left of a counter-clockwise edge
            half = np.where(shared, PART / 2, EXT / 2)[:, None]
            face = centre + half * inward
            keep = np.ones(n, dtype=bool)
            for kind, (ox, oy), width in openings:
                on_line = abs(ax - ox) < 1e-6 if ax == bx else abs(ay - oy) < 1e-6
                if not on_line:
                    continue
                along = centre[:, 1] - oy if ax == bx else centre[:, 0] - ox
                hole = (z < 2.05) if kind == "door" else ((z > 0.9) & (z < 2.1))
                keep &= ~((np.abs(along) < width / 2) & hole)
            parts.append(np.column_stack([face, z])[keep])
        xs, ys = [x for x, _ in poly], [y for _, y in poly]
        for height in (0.0, HEIGHT):
            m = int((max(xs) - min(xs)) * (max(ys) - min(ys)) * 150)
            flat = np.column_stack([rng.uniform(min(xs), max(xs), m), rng.uniform(min(ys), max(ys), m)])
            flat = flat[_inside(flat, poly)]
            parts.append(np.column_stack([flat, np.full(len(flat), height)]))
        if clutter:  # a table-height box near the room's middle: furniture stays below the wall band
            cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            m = 3000
            box = np.column_stack(
                [rng.uniform(cx - 0.6, cx + 0.6, m), rng.uniform(cy - 0.4, cy + 0.4, m), rng.uniform(0, 0.8, m)]
            )
            parts.append(box)
    cloud = np.concatenate(parts)
    if clutter:  # stray points anywhere, 0.5% of the cloud
        lo, hi = cloud.min(axis=0), cloud.max(axis=0)
        parts.append(rng.uniform(lo, hi, (len(cloud) // 200, 3)))
        cloud = np.concatenate(parts)
    cloud = cloud + rng.normal(0, noise, cloud.shape)
    c, s = math.cos(angle), math.sin(angle)
    turn = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return cloud @ turn.T + np.array([3.7, -12.2, 0.4])


def _normalised(poly: Poly, quarter_turns: int, origin: tuple[float, float]) -> Poly:
    turned = poly
    for _ in range(quarter_turns):
        turned = [(-y, x) for x, y in turned]
    return [(x - origin[0], y - origin[1]) for x, y in turned]


def _same_polygon(a: Poly, b: tuple[tuple[float, float], ...]) -> bool:
    if len(a) != len(b):
        return False
    return any(all(math.dist(a[i], b[(i + shift) % len(b)]) <= TOL for i in range(len(a))) for shift in range(len(b)))


def matches(truth: list[Poly], plan: fp.Floorplan) -> bool:
    """Whether the plan's rooms are the truth's, up to a quarter turn and a translation."""
    if len(truth) != len(plan.rooms):
        return False
    for turns in range(4):
        turned = [_normalised(p, turns, (0.0, 0.0)) for p in truth]
        origin = (min(x for p in turned for x, _ in p), min(y for p in turned for _, y in p))
        moved = [[(x - origin[0], y - origin[1]) for x, y in p] for p in turned]
        if all(any(_same_polygon(t, r.polygon) for r in plan.rooms) for t in moved):
            return True
    return False


ONE_ROOM = [rect(0, 0, 5, 4)]
L_FLAT = [[(0, 0), (8, 0), (8, 3), (3.5, 3), (3.5, 7), (0, 7)]]
TWO_ROOMS = [rect(0, 0, 4, 3.5), rect(4, 0, 3, 3.5)]


@pytest.mark.parametrize(
    "rooms",
    [pytest.param(ONE_ROOM, id="one-room"), pytest.param(L_FLAT, id="l-shaped"), pytest.param(TWO_ROOMS, id="two")],
)
def test_recovers_the_rooms_of_a_noisy_cluttered_scan(rooms):
    cloud = scan(rooms, [], np.random.default_rng(1), angle=math.radians(23))
    assert matches(rooms, fp.floorplan(cloud))


def test_a_door_between_two_rooms_is_found_where_it_is():
    door = ("door", (4.0, 1.5), 0.9)
    plan = fp.floorplan(scan(TWO_ROOMS, [door], np.random.default_rng(2), angle=0.0))
    assert matches(TWO_ROOMS, plan)
    (opening,) = plan.openings
    assert (opening.kind, opening.between) == ("door", ("r1", "r2"))
    assert opening.at == pytest.approx((4.0, 1.5), abs=TOL)
    assert opening.width == pytest.approx(0.9, abs=TOL)


def test_exterior_gaps_are_a_window_over_a_sill_and_a_door_down_to_the_floor():
    openings = [("window", (2.5, 4.0), 1.6), ("door", (0.0, 1.5), 1.0)]
    plan = fp.floorplan(scan(ONE_ROOM, openings, np.random.default_rng(3), angle=0.0))
    assert matches(ONE_ROOM, plan)
    found = {(o.kind, o.between): o for o in plan.openings}
    assert set(found) == {("window", ("exterior", "r1")), ("door", ("exterior", "r1"))}
    assert found["window", ("exterior", "r1")].at == pytest.approx((2.5, 4.0), abs=TOL)
    assert found["door", ("exterior", "r1")].at == pytest.approx((0.0, 1.5), abs=TOL)


@settings(max_examples=12, deadline=None)
@given(
    width=st.floats(2.5, 8.0),
    depth=st.floats(2.5, 8.0),
    angle=st.floats(0.0, math.pi * 2),
    seed=st.integers(0, 2**32 - 1),
)
# #2851: a small room's sparse floor, whose RANSAC plane came out 6 degrees off and was dropped for the ceiling.
@example(width=2.5, depth=3.0, angle=0.0, seed=0)
def test_any_rectangular_room_at_any_heading_comes_back_as_its_rectangle(width, depth, angle, seed):
    rooms = [rect(0, 0, round(width, 2), round(depth, 2))]
    assert matches(rooms, fp.floorplan(scan(rooms, [], np.random.default_rng(seed), angle=angle)))


@settings(max_examples=10, deadline=None)
# A door whose edge is under min_wall (0.4 m) from a corner reads as running into it (a documented limit), so the
# door's centre stays at least 0.45 + 0.1 (the exterior face) + 0.4 m from each end of the 3.5 m wall.
@given(split=st.floats(2.0, 5.0), door_at=st.floats(0.95, 2.55), seed=st.integers(0, 2**32 - 1))
# #2851: each of these once lost or narrowed the door because a plane tilted 4 degrees through the walls was taken for
# the floor, which lifted the lintel into the wall band.
@example(split=4.9609375, door_at=2.2421875, seed=261)
@example(split=4.99, door_at=1.88, seed=5)
@example(split=3.5, door_at=0.95, seed=6)
def test_two_rooms_split_anywhere_keep_their_door(split, door_at, seed):
    split, door_at = round(split, 2), round(door_at, 2)
    rooms = [rect(0, 0, split, 3.5), rect(split, 0, 7 - split, 3.5)]
    plan = fp.floorplan(scan(rooms, [("door", (split, door_at), 0.9)], np.random.default_rng(seed)))
    assert matches(rooms, plan)
    assert [(o.kind, o.at) for o in plan.openings] == [("door", pytest.approx((split, door_at), abs=TOL))]


def test_a_camera_frame_cloud_finds_its_floor_with_an_up_hint():
    cloud = scan(ONE_ROOM, [], np.random.default_rng(4))
    camera = cloud[:, [0, 2, 1]] * np.array([1, -1, 1])  # y down, as video models write it
    assert matches(ONE_ROOM, fp.floorplan(camera, fp.Settings(up=(0, -1, 0))))


def test_the_transform_takes_cloud_points_to_plan_coordinates():
    cloud = scan(ONE_ROOM, [], np.random.default_rng(5), angle=math.radians(40), clutter=False, noise=0.0)
    plan = fp.floorplan(cloud)
    floor = cloud[np.abs(cloud[:, 2] - 0.4) < 1e-9]  # the floor sits at z = 0.4 after the pose
    mapped = floor @ plan.transform[:, :3].T + plan.transform[:, 3]
    assert np.abs(mapped[:, 2]).max() < 0.01  # on the floor: height 0
    (room,) = plan.rooms
    xs, ys = [x for x, _ in room.polygon], [y for _, y in room.polygon]
    # Floor points lie inside the room's walls (half an exterior wall in from the centreline).
    assert mapped[:, 0].min() >= min(xs) - TOL and mapped[:, 0].max() <= max(xs) + TOL
    assert mapped[:, 1].min() >= min(ys) - TOL and mapped[:, 1].max() <= max(ys) + TOL


def test_a_cloud_with_no_floor_is_refused():
    rng = np.random.default_rng(6)
    wall = np.column_stack([np.zeros(5000), rng.uniform(0, 5, 5000), rng.uniform(0, 2.6, 5000)])
    with pytest.raises(ValueError, match="no floor"):
        fp.floorplan(wall)


def test_dsl_writes_rect_rooms_poly_rooms_and_openings_in_the_librarys_grammar():
    plan = fp.Floorplan(
        rooms=(
            fp.Room("r1", ((0.0, 0.0), (4.0, 0.0), (4.0, 3.5), (0.0, 3.5))),
            fp.Room("r2", ((4.0, 0.0), (8.0, 0.0), (8.0, 3.0), (6.5, 3.0), (6.5, 3.5), (4.0, 3.5))),
        ),
        openings=(
            fp.Opening("door", ("r1", "r2"), (4.0, 1.25), 0.9),
            fp.Opening("window", ("exterior", "r1"), (2.0, 0.0), 1.2),
        ),
        transform=np.eye(3, 4),
    )
    assert fp.to_dsl(plan, title='Flat "B"') == (
        "plan \"Flat 'B'\" walls 0.2/0.1\n"
        "\n"
        'room r1 "Room 1" other rect 0,0 4x3.5\n'
        'room r2 "Room 2" other poly 4,0 8,0 8,3 6.5,3 6.5,3.5 4,3.5\n'
        "\n"
        "door r1>r2 at 4,1.25 w0.9\n"
        "window exterior>r1 at 2,0 w1.2\n"
    )


@pytest.mark.parametrize("fmt", ["ascii", "binary_little_endian", "binary_big_endian"])
def test_ply_reads_vertex_positions_past_other_properties(tmp_path, fmt):
    xyz = np.array([[0.5, -1.25, 2.0], [3.0, 4.5, -0.75]])
    header = (
        f"ply\nformat {fmt} 1.0\ncomment made by a test\nelement vertex 2\n"
        "property float x\nproperty float y\nproperty float z\nproperty uchar red\n"
        "element face 0\nproperty list uchar int vertex_indices\nend_header\n"
    )
    if fmt == "ascii":
        body = "".join(f"{x} {y} {z} 200\n" for x, y, z in xyz).encode()
    else:
        order = "<" if fmt == "binary_little_endian" else ">"
        rows = np.zeros(2, dtype=[("x", order + "f4"), ("y", order + "f4"), ("z", order + "f4"), ("red", "u1")])
        rows["x"], rows["y"], rows["z"], rows["red"] = xyz[:, 0], xyz[:, 1], xyz[:, 2], 200
        body = rows.tobytes()
    path = tmp_path / "cloud.ply"
    path.write_bytes(header.encode() + body)
    assert ply.load(path) == pytest.approx(xyz)


def test_ply_refuses_a_file_without_vertices(tmp_path):
    path = tmp_path / "mesh.ply"
    path.write_bytes(b"ply\nformat ascii 1.0\nelement face 0\nproperty list uchar int vertex_indices\nend_header\n")
    with pytest.raises(ply.PlyError, match="no vertex"):
        ply.load(path)


def test_the_floor_is_the_lowest_plane_under_the_room_not_at_the_origin():
    # The scan sits 12 m from the origin. Besides the floor and the ceiling, RANSAC's spare plane here is a slice
    # through the walls tilted 4 degrees, whose height at the origin is lower than the floor's (#2851).
    rooms = [rect(0, 0, 4.99, 3.5), rect(4.99, 0, 2.01, 3.5)]
    cloud = scan(rooms, [("door", (4.99, 1.88), 0.9)], np.random.default_rng(5))
    normal, d = fp.floor_plane(cloud, (0.0, 0.0, 1.0), 0.03, np.random.default_rng(0))
    assert math.degrees(math.acos(normal[2])) < 0.5
    assert d / normal[2] == pytest.approx(0.4, abs=0.02)  # scan() lifts the floor to z = 0.4


# Carrying an owner's plan into another frame (#2997)

OWNER_PLAN = """plan "Cottage" walls 0.2/0.1

room r1 "Bedroom" bedroom rect 1,2 3x4
room r2 "Closet" other poly 4,2 5,2 5,3 4,3
door r1>r2 at 4,2.5 w0.8
door r1.north @1 w0.9
fixture bath in:r1 at 1,2 size 1x2 id:tub
x-camera bedroom at 2,3 yaw 10
x-anchor 01a1b856-15c9-7622-9732-bc2aee79fdda yaw 0 origin 0,0,0
"""

QUARTER = fp.Similarity(1.0, math.pi / 2, (10.0, 0.0))  # (x, y) to (10 - y, x): clockwise on the page


def test_an_owner_plan_carried_a_quarter_turn_keeps_its_rooms_square_and_renames_its_ids():
    got = fp.carry(OWNER_PLAN, QUARTER, "m1", "Cottage")
    assert got.lines == (
        'room m1_r1 "Bedroom" bedroom rect 4,1 4x3',
        'room m1_r2 "Closet" other poly 8,4 8,5 7,5 7,4',
        "door m1_r1>m1_r2 at 7.5,4 w0.8",
        "door m1_r1.east @1 w0.9",  # its north wall faces east once turned
        "fixture bath in:m1_r1 at 6,1 size 2x1 id:m1_tub",  # its top-left corner is another corner once turned
        "x-camera bedroom at 7,2 yaw 100",
    ), "the header and the old frame's anchor are left out: the target plan's hold"
    assert got.renamed == {"r1": "m1_r1", "r2": "m1_r2", "tub": "m1_tub"}
    assert got.kept_as_comments == 0
    assert [name for _, name, _ in fp.rooms_from_dsl("\n".join(got.lines))] == ["Bedroom", "Closet"]


def test_under_a_turn_that_isnt_a_quarter_what_cant_turn_is_kept_as_comments():
    move = fp.Similarity(2.0, math.radians(30), (0.0, 0.0))
    got = fp.carry(OWNER_PLAN, move, "m1", "Cottage")
    (bedroom,) = (line for line in got.lines if line.startswith("room m1_r1"))
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    corner = (2 * (c * 4 - s * 2), 2 * (s * 4 + c * 2))  # (4, 2) turned and doubled
    assert bedroom.startswith('room m1_r1 "Bedroom" bedroom poly ')
    assert f"{fp._num(corner[0])},{fp._num(corner[1])}" in bedroom.split()
    assert "door m1_r1>m1_r2 at" in "\n".join(got.lines) and "w1.6" in "\n".join(got.lines), "lengths are scaled"
    assert "# from Cottage: door r1.north @1 w0.9" in got.lines, "a wall side can't turn 30 degrees"
    (bath,) = (line for line in got.lines if line.startswith("fixture"))
    assert bath.startswith("fixture bath in:m1_r1 poly ") and len(bath.split()) == 9, "a box turned is its corners"
    assert got.kept_as_comments == 1


def test_a_track_grid_and_its_picture_are_kept_as_comments():
    text = 'plan "P"\nlayout cols 3,3 rows 3\n  a b\nroom a "A" rect 0,0 3x3\n'
    got = fp.carry(text, QUARTER, "m2", "P")
    assert got.lines == ("# from P: layout cols 3,3 rows 3", "# from P: a b", 'room m2_a "A" rect 7,0 3x3')
    assert got.kept_as_comments == 2


def test_a_turn_near_a_quarter_snaps_to_it_keeping_its_centre_where_it_lands():
    near = fp.Similarity(1.0, math.radians(93), (5.0, 1.0))
    snapped = near.snapped((2.0, 3.0), within_deg=10)
    assert snapped.turn == pytest.approx(math.pi / 2) and snapped.quarters == 1
    assert snapped.point(2.0, 3.0) == pytest.approx(near.point(2.0, 3.0))
    assert near.snapped((2.0, 3.0), within_deg=2) == near, "3 degrees off is too far to round at 2"


def test_without_takes_out_a_room_and_whatever_names_it():
    text = (
        'room r1 "A" rect 0,0 2x2\nroom r2 "B" rect 2,0 2x2\ndoor r1>r2 at 2,1 w0.8\ndoor r2.east @1 w0.8\n'
        "door exterior>r1 at 0,1 w0.9\nfixture bath in:r2 at 2,0 size 1x1\nx-source room:r2 owner\n"
    )
    assert fp.without(text, ["r2"]) == 'room r1 "A" rect 0,0 2x2\ndoor exterior>r1 at 0,1 w0.9\n'


def test_covered_is_the_share_of_a_room_inside_others():
    square = ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0))
    assert fp.covered(square, [("a", "A", ((1.0, -1.0), (3.0, -1.0), (3.0, 3.0), (1.0, 3.0)))]) == 0.5
    assert fp.covered(square, []) == 0.0
