"""A learned map's floor and fixed cameras, and people placed on its floor through them."""

import math

import numpy as np
import pytest

from liberational import segment_log as S
from liberational.memory import places
from liberational.spatial import ply, surroundings_map
from liberational.spatial import positions as P
from liberational.spatial import surroundings_positions as SP

T = 1_800_000_000.0
SIZE = (512, 288)
# The bottom middle of the frame is 15 degrees below where the camera looks: 144 px down at this focal.
FOCAL = 144.0 / math.tan(math.radians(15.0))
ABOVE = 2.0  # the cameras' height over the floor, metres
PITCH = 30.0  # and how far down they look


def camera(x, y, heading=0.0, pitch=PITCH, above=ABOVE):
    """A camera-to-room pose (4x4): at (x, y), `above` the floor (the room's z is up), looking `heading` degrees
    from the room's +y towards its +x and `pitch` degrees down, with no roll."""
    h, p = math.radians(heading), math.radians(pitch)
    forward = np.array([math.sin(h) * math.cos(p), math.cos(h) * math.cos(p), -math.sin(p)])
    right = np.array([math.cos(h), -math.sin(h), 0.0])
    out = np.eye(4)
    out[:3, 0], out[:3, 1], out[:3, 2], out[:3, 3] = right, np.cross(forward, right), forward, (x, y, above)
    return out


FIRST = camera(2.0, 0.5)
# The map's frame is its first camera's, as a learned map's is: nothing in it is level.
TO_MAP = np.linalg.inv(FIRST)


def in_map(points):
    return np.asarray(points, dtype=np.float64).reshape(-1, 3) @ TO_MAP[:3, :3].T + TO_MAP[:3, 3]


def sheet(origin, a, b, step=0.05):
    """Points every `step` over the parallelogram from `origin` along `a` and `b` (room coordinates)."""
    o, a, b = (np.asarray(v, dtype=np.float64) for v in (origin, a, b))
    i, j = np.meshgrid(np.arange(0, 1, step / np.linalg.norm(a)), np.arange(0, 1, step / np.linalg.norm(b)))
    return o + i.reshape(-1, 1) * a + j.reshape(-1, 1) * b


ROOM = np.vstack(
    [
        sheet((0, 0, 0), (4, 0, 0), (0, 5, 0)),  # the floor, 4 x 5 m
        sheet((1, 3, 0.75), (1.2, 0, 0), (0, 0.8, 0)),  # a table top
        sheet((0, 5, 0), (4, 0, 0), (0, 0, 2.4)),  # the far wall
        sheet((0, 0, 0), (0, 5, 0), (0, 0, 2.4)),  # a side wall
    ]
)


def view(name, cam, pose, t=T):
    flat = tuple(float(x) for x in (TO_MAP @ pose).ravel())
    return surroundings_map.View(name, cam, t, flat, FOCAL / SIZE[0], *SIZE)


VIEWS = [view("a", "hall", FIRST), view("b", "hall", FIRST, T + 60)]


def height(plane, room_point):
    (p,) = in_map([room_point])
    return float(np.asarray(plane[:3]) @ p + plane[3])


def test_a_maps_floor_is_found_from_its_own_surfaces_though_nothing_in_it_is_level():
    plane = SP.floor(in_map(ROOM), VIEWS)
    assert plane is not None
    assert height(plane, (2.0, 2.0, 0.0)) == pytest.approx(0.0, abs=0.03)
    assert height(plane, (1.5, 3.5, 0.75)) == pytest.approx(0.75, abs=0.03)  # the table isn't taken for the floor
    assert height(plane, (2.0, 0.5, ABOVE)) == pytest.approx(ABOVE, abs=0.03)


@pytest.mark.parametrize(
    "points",
    [
        pytest.param(sheet((0, 5, 0), (4, 0, 0), (0, 0, 2.4)), id="only-a-wall"),
        # the only flat layer is over the camera's head: a ceiling, which a camera is never under the floor of
        pytest.param(sheet((0, 0, 2.4), (4, 0, 0), (0, 5, 0)), id="only-a-ceiling"),
    ],
)
def test_a_map_with_no_floor_in_it_has_none(points):
    assert SP.floor(in_map(points), VIEWS) is None


def test_a_floor_found_again_replaces_the_held_one_only_when_it_moved():
    plane = SP.floor(in_map(ROOM), VIEWS)
    assert plane is not None
    nudged = (*plane[:3], plane[3] + 0.02)
    lifted = (*plane[:3], plane[3] + 0.2)
    assert not SP.floor_moved(plane, nudged, VIEWS)
    assert SP.floor_moved(plane, lifted, VIEWS)


def test_a_camera_whose_views_sit_at_one_pose_is_fixed_and_one_seen_from_several_is_moving():
    views = [
        *VIEWS,
        view("c", "cart", camera(1.0, 1.0)),
        view("d", "cart", camera(3.0, 2.5, heading=40.0), T + 60),
        view("e", "porch", camera(3.0, 0.5)),  # one view of a camera the manifest has on no rig
        view("f", "pack-front", camera(1.0, 0.5)),  # one view of a camera on a rig
    ]
    registered, unplaced = SP.fixed(views, "p", ["hall", "cart", "porch"], {}, T + 100)
    assert sorted(registered) == ["hall", "porch"]
    assert unplaced == {"cart": "moving", "pack-front": "moving"}
    hall = registered["hall"]
    assert np.asarray(hall.pose).reshape(4, 4) == pytest.approx(np.eye(4), abs=1e-6)  # the first view is the origin
    assert (hall.focal, hall.width, hall.height, hall.lens) == (pytest.approx(FOCAL), *SIZE, None)


@pytest.fixture
def manifest(tmp_path):
    path = tmp_path / "devices.toml"
    path.write_text(
        '[[rig]]\nname = "pack"\n\n'
        '[[device]]\nname = "hall"\nkind = "rtsp"\n\n'
        '[[device]]\nname = "pack-front"\nkind = "camera"\nmount = { rig = "pack", hfov_deg = 110.0 }\n'
    )
    return path


def learned(tmp_path, monkeypatch, views=VIEWS, points=ROOM, place=None, at=T):
    """A learned revision of a place, as the map step writes one, and its views."""
    was = places.latest(tmp_path, place) if place is not None else None
    rev = places.revise(
        tmp_path,
        place,
        producer=surroundings_map.PRODUCER,
        map_bytes=ply.dumps(in_map(points)),
        map_format="ply",
        frame=was.frame if was is not None else None,
        floor=was.floor if was is not None else None,
        cameras=was.cameras if was is not None else None,
        now=at,
    )
    monkeypatch.setattr(surroundings_map, "views", lambda state, place: list(views) if place == rev.place else [])
    return rev


def test_a_learned_map_gets_its_floor_and_fixed_cameras_in_a_revision_of_their_own(tmp_path, monkeypatch, manifest):
    rev = learned(tmp_path, monkeypatch, [*VIEWS, view("f", "pack-front", camera(1.0, 0.5))])
    found = SP.advance(tmp_path, T + 10, manifest)[rev.place]
    assert (found["floor"], found["cameras"], found["unplaced"]) == (True, ["hall"], {"pack-front": "moving"})
    newest = places.latest(tmp_path, rev.place)
    assert newest is not None
    assert (newest.producer, newest.map, newest.frame) == (SP.PRODUCER, rev.map, rev.frame)
    assert newest.floor is not None and sorted(newest.cameras) == ["hall"]


def test_a_map_that_grew_without_changing_shape_writes_nothing_more(tmp_path, monkeypatch, manifest):
    rev = learned(tmp_path, monkeypatch)
    SP.advance(tmp_path, T + 10, manifest)
    SP.advance(tmp_path, T + 20, manifest)  # the same map again
    grown = learned(tmp_path, monkeypatch, place=rev.place, at=T + 30)  # a later step's map, the room as it was
    SP.advance(tmp_path, T + 40, manifest)
    chain = places.revisions(tmp_path, rev.place)
    assert [r.producer for r in chain] == [surroundings_map.PRODUCER, SP.PRODUCER, surroundings_map.PRODUCER]
    assert chain[-1].floor == chain[1].floor and chain[-1].cameras == chain[1].cameras
    status = SP.status(tmp_path, rev.place)
    assert status is not None and status["map"] == grown.map


def test_a_registered_camera_places_nobody_until_the_map_has_a_floor(tmp_path, monkeypatch, manifest):
    wall = sheet((0, 5, 0), (4, 0, 0), (0, 0, 2.4))
    rev = learned(tmp_path, monkeypatch, points=wall)
    found = SP.advance(tmp_path, T + 10, manifest)[rev.place]
    assert (found["floor"], found["unplaced"]) == (False, {"hall": "floor"})
    assert P.placing(tmp_path, rev.place) == [{"camera": "hall", "by": "map", "missing": ["floor"]}]


def foot(vision, t, v, cam="hall"):
    row = dict(source=P.FIXED_CAMERA, confidence=0.9, camera=cam, u=0.5, v=v, track=f"{cam}:7", entity=None)
    row |= dict(place=None, room=None, x=None, y=None, revision=None)
    return vision.add(S.POSITIONS, ts_start=t, cam=cam, **row)


def point(p: P.Position) -> tuple[float, float]:
    assert p.x is not None and p.y is not None
    return p.x, p.y


def test_a_fixed_cameras_foot_points_land_on_the_learned_maps_floor_in_metres(tmp_path, vision, monkeypatch, manifest):
    # The camera is 2 m up looking 30 degrees down. Heels (0.07 m up) at the bottom middle of its frame are 45 degrees
    # down its ray, so 1.93 m out along the floor; at the frame's centre, 30 degrees down, 1.93 / tan 30 = 3.343 m.
    rev = learned(tmp_path, monkeypatch)
    SP.advance(tmp_path, T + 10, manifest)
    foot(vision, T + 20, v=1.0)
    foot(vision, T + 21, v=0.5)
    first, second = P.derive(tmp_path, T, T + 60)
    assert (first.place, second.place) == (rev.place, rev.place)
    near, far = point(first), point(second)
    assert math.dist(near, far) == pytest.approx(3.343 - 1.93, abs=0.05)
    newest = places.latest(tmp_path, rev.place)
    assert newest is not None
    under = places.project(newest, "hall", 0.5, 1.0, ABOVE - 0.001)  # heels nearly at the lens: under the camera
    assert under is not None
    assert math.dist(near, under) == pytest.approx(1.93, abs=0.05)


def test_a_moving_cameras_foot_points_stay_unplaced_with_the_reason(tmp_path, vision, monkeypatch, manifest):
    rev = learned(tmp_path, monkeypatch, [*VIEWS, view("f", "pack-front", camera(1.0, 0.5))])
    SP.advance(tmp_path, T + 10, manifest)
    foot(vision, T + 20, v=1.0, cam="pack-front")
    (p,) = P.derive(tmp_path, T, T + 60)
    assert (p.place, p.x, p.y) == (None, None, None)
    assert P.placing(tmp_path, rev.place) == [
        {"camera": "hall", "by": "map", "missing": []},
        {"camera": "pack-front", "by": "map", "missing": ["moving"]},
    ]
