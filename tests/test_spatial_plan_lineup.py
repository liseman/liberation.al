"""Lining a learned map up with the owner's plan: found from the walls or the cameras' measured spots once the map
covers enough to tell, checked as the map grows, and never fitted twice on a map that stayed as it was."""

import math

import numpy as np
import pytest

import plans
from liberational import segment_log as S
from liberational.memory import places
from liberational.spatial import plan_import as pi
from liberational.spatial import plan_lineup as L
from liberational.spatial import ply, surroundings_map
from liberational.spatial import positions as P

T = 1_800_000_000.0
# Four rooms in an L, so no turn of the plan fits its own walls again.
PLAN = plans.scene(
    'plan "Home" walls 0.2/0.1\n'
    'room den "Den" other rect 0,0 5x4\n'
    'room kitchen "Kitchen" other rect 5,0 3x2.5\n'
    'room hall "Hall" other rect 5,2.5 3x1.5\n'
    'room bed "Bedroom" other rect 0,4 3x3\n'
)
SPOTS = "x-camera cam0 at 1,1 height 2\nx-camera cam1 at 6.5,1.5 height 2\n"
SQUARE = plans.scene('plan "Shed" walls 0.2/0.1\nroom shed "Shed" other rect 0,0 4x4\n')
EVERYWHERE = (-1.0, -1.0, 9.0, 8.0)
DEN_CORNER = (-1.0, -1.0, 2.0, 2.0)  # two 2 m stretches of wall: a corner like any other

# The learned map's frame, as its first camera made it: 8% larger than the plan's metres, pitched and turned, and
# somewhere else. A plan point (x, y) at height h is SCALE * TILT @ (x, y, h) + ORIGIN in the map.
SCALE = 1.08
ORIGIN = np.array([2.0, -1.5, 4.0])


def _rotation(axis: int, degrees: float) -> np.ndarray:
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    i, j = [k for k in range(3) if k != axis]
    out = np.eye(3)
    out[i, i], out[i, j], out[j, i], out[j, j] = c, -s, s, c
    return out


TILT = _rotation(0, 100.0) @ _rotation(2, 25.0)
UP = TILT @ np.array([0.0, 0.0, 1.0])
FLOOR = (*UP, -float(UP @ ORIGIN))  # a map point's height above the floor, in the map's units


def in_map(xyh: np.ndarray) -> np.ndarray:
    return SCALE * (np.asarray(xyh, dtype=np.float64).reshape(-1, 3) @ TILT.T) + ORIGIN


def cloud(plan: str, within=EVERYWHERE, shift=(0.0, 0.0), seed=0) -> bytes:
    """A learned map of `plan`'s walls inside the box `within` (plan metres), floor to ceiling on both faces of each
    wall with a centimetre of noise, as PLY in the map's frame; `shift` moves it on the floor first (a map that
    drifted)."""
    rng = np.random.default_rng(seed)
    w = pi.walls(pi.parse(plan))
    faces = []
    for a, b, half in zip(w.a, w.b, w.half, strict=True):
        length = float(np.hypot(*(b - a)))
        along = a + np.linspace(0.0, 1.0, max(int(length / 0.04), 2))[:, None] * (b - a)
        normal = np.array([-(b - a)[1], (b - a)[0]]) / length
        faces += [along + normal * half, along - normal * half]
    xy = np.vstack(faces)
    x0, y0, x1, y1 = within
    xy = xy[(xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)]
    heights = np.arange(0.05, 2.4, 0.2)
    xyh = np.c_[np.repeat(xy, len(heights), axis=0) + np.asarray(shift), np.tile(heights, len(xy))]
    return ply.dumps(in_map(xyh + rng.normal(0.0, 0.01, xyh.shape)))


def camera(x: float, y: float, height: float = 2.0) -> places.CameraPose:
    """A camera registered in the map at plan (x, y), `height` up, looking along the plan's +x and 45 degrees down."""
    r = (
        TILT
        @ np.array([[0.0, -1.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, -1.0]])
        / np.array([1.0, math.sqrt(2), math.sqrt(2)])
    )
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = r, in_map(np.array([x, y, height]))[0]
    return places.CameraPose(tuple(float(v) for v in pose.ravel()), 144.0, 512, 288, T)


def kept_place(state, plan: str) -> str:
    """A place as a reset leaves it: listed with the owner's plan, and no map."""
    rev = places.revise(state, None, producer="places reset v1", now=T)
    places.correct_plan(state, rev.place, plan, now=T + 1)
    return rev.place


def learn(state, place: str, data: bytes, now: float, floor=FLOOR, cameras=None) -> places.Revision:
    """One step of the map: a revision with a map, carrying over what the one before it had, as the map step does."""
    was = places.latest(state, place)
    assert was is not None
    return places.revise(
        state, place, producer="surroundings v1", map_bytes=data, map_format="ply", now=now,
        floor=floor if floor is not None else was.floor, plan_transform=was.plan_transform, frame=was.frame,
        cameras=cameras,
    )  # fmt: skip


def corner_error(transform, plan: str, shift=(0.0, 0.0)) -> float:
    """How far, in plan metres, the transform puts the map's own points of the plan's room corners from them."""
    corners = np.array([p for r in pi.parse(plan).rooms for p in r.polygon], dtype=np.float64)
    there = in_map(np.c_[corners + np.asarray(shift), np.zeros(len(corners))])
    back = pi.in_frame(there, transform)
    return float(np.max(np.hypot(*(back[:, :2] - corners).T)))


def test_a_learned_map_of_the_plans_rooms_is_lined_up_with_it(tmp_path):
    place = kept_place(tmp_path, PLAN)
    learned = learn(tmp_path, place, cloud(PLAN), T + 10)
    found = L.advance(tmp_path, now=T + 20)[place]
    assert (found["state"], found["reason"], found["by"]) == ("aligned", None, "walls")
    assert found["fit"]["scale"] == pytest.approx(SCALE, rel=0.02)
    rev = places.latest(tmp_path, place)
    assert rev is not None and rev.producer == L.PRODUCER and (rev.map, rev.floor) == (learned.map, learned.floor)
    assert rev.plan_transform is not None and corner_error(rev.plan_transform, PLAN) < 0.1
    # heights come out in the plan's metres too: a point 2 m up the wall
    assert pi.in_frame(in_map(np.array([5.0, 4.0, 2.0])), rev.plan_transform)[0, 2] == pytest.approx(2.0, abs=0.05)
    held = places.get(tmp_path, place)
    assert held is not None and (held.plan, held.plan_source, held.plan_revision) == (PLAN, "owner", rev.id)
    assert [r.value["source"] for r in places.plan_history(tmp_path, place)] == ["owner", "lineup"]


def test_a_map_that_covers_too_little_of_the_plan_is_left_unaligned_with_the_reason(tmp_path):
    place = kept_place(tmp_path, PLAN)
    learned = learn(tmp_path, place, cloud(PLAN, within=DEN_CORNER), T + 10)
    found = L.advance(tmp_path, now=T + 20)[place]
    assert (found["state"], found["reason"]) == ("unaligned", "too_little")
    assert places.latest(tmp_path, place) == learned, "no revision is written for a line-up not found"
    assert len(places.plan_history(tmp_path, place)) == 1
    assert L.status(tmp_path, place) == found


def test_a_room_that_fits_its_plan_more_than_one_way_is_left_unaligned(tmp_path):
    place = kept_place(tmp_path, SQUARE)
    learned = learn(tmp_path, place, cloud(SQUARE), T + 10)
    found = L.advance(tmp_path, now=T + 20)[place]
    assert (found["state"], found["reason"]) == ("unaligned", "ambiguous")
    assert places.latest(tmp_path, place) == learned


def test_one_measured_camera_spot_settles_a_room_its_walls_cant(tmp_path):
    place = kept_place(tmp_path, SQUARE + "x-camera cam0 at 1,0.5 height 2\n")
    learn(tmp_path, place, cloud(SQUARE), T + 10, cameras={"cam0": camera(1.0, 0.5)})
    found = L.advance(tmp_path, now=T + 20)[place]
    assert (found["state"], found["by"]) == ("aligned", "walls")
    rev = places.latest(tmp_path, place)
    assert rev is not None and rev.plan_transform is not None and corner_error(rev.plan_transform, SQUARE) < 0.1


def test_two_measured_camera_spots_line_up_a_map_with_too_few_walls(tmp_path):
    place = kept_place(tmp_path, PLAN + SPOTS)
    cams = {"cam0": camera(1.0, 1.0), "cam1": camera(6.5, 1.5)}
    learn(tmp_path, place, cloud(PLAN, within=DEN_CORNER), T + 10, cameras=cams)
    found = L.advance(tmp_path, now=T + 20)[place]
    assert (found["state"], found["by"]) == ("aligned", "anchors")
    assert max(found["fit"]["anchors"].values()) < 0.1
    rev = places.latest(tmp_path, place)
    assert rev is not None and rev.plan_transform is not None and corner_error(rev.plan_transform, PLAN) < 0.15


def test_a_camera_far_from_its_measured_spot_refuses_the_line_up(tmp_path):
    place = kept_place(tmp_path, PLAN + "x-camera cam0 at 1,1 height 2\n")
    # the walls put the map 3 m along the plan from where its one measured camera says it is
    learn(tmp_path, place, cloud(PLAN, shift=(3.0, 0.0)), T + 10, cameras={"cam0": camera(1.0, 1.0)})
    found = L.advance(tmp_path, now=T + 20)[place]
    assert (found["state"], found["reason"]) == ("unaligned", "poor_fit")


@pytest.mark.parametrize(
    ("data", "floor", "reason"),
    [
        pytest.param(None, None, "no_map", id="nothing-learned"),
        pytest.param(cloud(PLAN), None, "no_floor", id="no-floor-yet"),
    ],
)
def test_a_place_with_no_map_or_no_floor_says_which(tmp_path, data, floor, reason):
    place = kept_place(tmp_path, PLAN)
    if data is not None:
        places.revise(tmp_path, place, producer="surroundings v1", map_bytes=data, map_format="ply", now=T + 10)
    found = L.advance(tmp_path, now=T + 20)[place]
    assert (found["state"], found["reason"]) == ("unaligned", reason)


def test_a_place_without_an_owners_plan_is_not_lined_up(tmp_path):
    rev = places.revise(
        tmp_path, None, producer="surroundings v1", map_bytes=cloud(PLAN), map_format="ply", floor=FLOOR
    )
    assert L.advance(tmp_path) == {}
    assert L.status(tmp_path, rev.place) is None


@pytest.fixture
def fits(monkeypatch) -> list[int]:
    """How many band points each fit that ran was given."""
    ran: list[int] = []
    real = L.fit

    def counted(plan, band, anchors=None):
        ran.append(len(band))
        return real(plan, band, anchors)

    monkeypatch.setattr(L, "fit", counted)
    return ran


def test_a_grown_map_is_checked_on_its_line_up_and_not_fitted_again(tmp_path, fits):
    place = kept_place(tmp_path, PLAN)
    learn(tmp_path, place, cloud(PLAN, within=(-1.0, -1.0, 9.0, 4.5)), T + 10)
    assert L.advance(tmp_path, now=T + 20)[place]["state"] == "aligned"
    lined = places.latest(tmp_path, place)
    grown = learn(tmp_path, place, cloud(PLAN, seed=1), T + 30)  # the bedroom's walls too
    found = L.advance(tmp_path, now=T + 40)[place]
    assert (found["state"], found["reason"]) == ("aligned", None)
    assert found["check"]["median"] < 0.05 and found["check"]["cover"] > 0.9
    assert len(fits) == 1, "found once"
    assert places.latest(tmp_path, place) == grown and lined is not None
    assert grown.plan_transform == lined.plan_transform
    assert len(places.plan_history(tmp_path, place)) == 2


def test_a_map_that_stayed_as_it_was_is_never_fitted_twice(tmp_path, fits):
    place = kept_place(tmp_path, SQUARE)
    learn(tmp_path, place, cloud(SQUARE), T + 10)
    first = L.advance(tmp_path, now=T + 20)[place]
    assert L.advance(tmp_path, now=T + 30)[place] == first, "nothing changed: nothing is worked out"
    learn(tmp_path, place, cloud(SQUARE, seed=1), T + 40)  # a step that added no wall
    again = L.advance(tmp_path, now=T + 50)[place]
    assert (again["state"], again["reason"]) == ("unaligned", "ambiguous")
    assert len(fits) == 1


def test_a_map_that_left_its_plan_is_fitted_once_more(tmp_path, fits):
    place = kept_place(tmp_path, PLAN)
    learn(tmp_path, place, cloud(PLAN), T + 10)
    L.advance(tmp_path, now=T + 20)
    learn(tmp_path, place, cloud(PLAN, shift=(0.6, 0.0)), T + 30)
    found = L.advance(tmp_path, now=T + 40)[place]
    assert (found["state"], found["reason"]) == ("aligned", None)
    rev = places.latest(tmp_path, place)
    assert rev is not None and rev.producer == L.PRODUCER and rev.plan_transform is not None
    assert corner_error(rev.plan_transform, PLAN, shift=(0.6, 0.0)) < 0.1
    assert len(fits) == 2


def test_a_drifted_map_that_cant_be_fitted_again_keeps_its_line_up_and_says_so(tmp_path):
    place = kept_place(tmp_path, PLAN)
    learn(tmp_path, place, cloud(PLAN), T + 10)
    L.advance(tmp_path, now=T + 20)
    lined = places.latest(tmp_path, place)
    off = learn(tmp_path, place, cloud(PLAN, within=DEN_CORNER, shift=(0.6, 0.6)), T + 30)
    found = L.advance(tmp_path, now=T + 40)[place]
    assert (found["state"], found["reason"]) == ("aligned", "drifted")
    assert found["check"]["median"] > L.CHECK_MEDIAN
    assert places.latest(tmp_path, place) == off and lined is not None and off.plan_transform == lined.plan_transform


def test_a_plan_the_owner_drew_in_the_maps_own_frame_is_left_as_it_is(tmp_path):
    place = kept_place(tmp_path, PLAN)
    learned = learn(tmp_path, place, cloud(PLAN), T + 10)
    places.correct_plan(tmp_path, place, PLAN, now=T + 15, base=learned.id)
    found = L.advance(tmp_path, now=T + 20)[place]
    assert (found["state"], found["by"]) == ("aligned", "owner")
    assert places.latest(tmp_path, place) == learned


def test_positions_on_a_lined_up_map_read_in_the_plans_rooms(tmp_path, vision):
    place = kept_place(tmp_path, PLAN)
    # 2 m up at plan (5.5, 1.2), looking 45 degrees down along +x: the middle of its frame is the floor 2 m ahead
    learn(tmp_path, place, cloud(PLAN), T + 10, cameras={"cam0": camera(5.5, 1.2)})
    row = dict(source=P.FIXED_CAMERA, confidence=0.9, camera="cam0", u=0.5, v=0.5, track="cam0:7", entity=None)
    row |= dict(place=None, room=None, x=None, y=None, revision=None)
    vision.add(S.POSITIONS, ts_start=T + 100, cam="cam0", **row)
    (before,) = P.derive(tmp_path, T + 99, T + 105)
    assert before.room is None, "in the map's own frame until it's lined up"
    L.advance(tmp_path, now=T + 20)
    learn(tmp_path, place, cloud(PLAN, seed=1), T + 30)  # a later step of the map carries the line-up over
    held = places.get(tmp_path, place)
    assert held is not None
    (p,) = P.derive(tmp_path, T + 99, T + 105, place=place, frame_revision=held.plan_revision)
    assert (p.room, p.room_id, p.in_frame) == ("Kitchen", "kitchen", True)
    assert (p.x, p.y) == pytest.approx((7.43, 1.2), abs=0.1)  # a person's heels, 7 cm up, land that much nearer


def test_the_transform_takes_a_map_point_onto_the_plan_through_the_floors_frame():
    # a level map (z up, floor at z = 0): its floor frame is x, y as they are; the plan is turned a quarter, twice
    # the size in map metres, and starts at map (4, 2)
    move = L.floorplan.Similarity(2.0, math.pi / 2, (4.0, 2.0))
    found = L.transform((0.0, 0.0, 1.0, 0.0), move)
    assert found is not None
    # plan (1, 0) is 2 m along the map's +y from (4, 2); 3 map metres up is 1.5 of the plan's
    assert pi.in_frame(np.array([4.0, 4.0, 3.0]), found)[0] == pytest.approx([1.0, 0.0, 1.5])
    assert L.to_plan(move, L.to_map(move, np.array([[0.3, -1.2]])))[0] == pytest.approx([0.3, -1.2])


def test_the_map_step_lines_its_places_up_when_it_is_done(tmp_path):
    place = kept_place(tmp_path, PLAN)
    learn(tmp_path, place, cloud(PLAN), T + 10)
    surroundings_map.line_up(tmp_path)
    rev = places.latest(tmp_path, place)
    assert rev is not None and rev.producer == L.PRODUCER


def test_a_line_up_that_fails_is_logged_and_the_map_step_goes_on(tmp_path, monkeypatch, caplog):
    def broken(state):
        raise OSError("the disk is full")

    monkeypatch.setattr(L, "advance", broken)
    surroundings_map.line_up(tmp_path)
    assert "the disk is full" in caplog.text
