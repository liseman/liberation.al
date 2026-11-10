"""The position layer: raw footprints placed on the plan and named (`derive`), zone visits as the room-only
fallback, the placer that writes them to the layer, the layer's read, and the 10 s paths kept past the window."""

import math
from dataclasses import replace

import numpy as np
import pytest

from liberational import retention
from liberational import segment_log as S
from liberational.domain.identity import Basis
from liberational.memory import places
from liberational.people import face_nots, pets, registry
from liberational.spatial import floorplan, footprints, lens, poses
from liberational.spatial import positions as P
from liberational.vision import animal_embedder

T = 1_800_000_000.0
# A place whose map is MASt3R's y-down frame: floor at y = 0, plan (x, y) = map (x, z).
FLOOR = (0.0, -1.0, 0.0, 0.0)
TO_PLAN = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0)
# A level camera 2 m above the floor at x = 1, looking along +z (288 px tall frame, 144 px focal): the bottom middle
# of its frame drops a metre a metre, so an animal's footprint there (on the floor) lands 2 m ahead, at plan (1, 2),
# and a person's (their heels, footprints.FOOT_M = 0.07 up) 1.93 m ahead.
LEVEL = places.CameraPose((1.0, 0, 0, 1.0, 0, 1.0, 0, -2.0, 0, 0, 1.0, 0.0, 0, 0, 0, 1.0), 144.0, 512, 288, T)
PLAN = 'plan "Home" walls 0.2/0.1\nroom r1 "Living" other rect 0,0 4x3\nroom r2 "Hall" other rect 0,3 4x2\n'


def home(tmp_path, *, plan=True):
    return places.revise(
        tmp_path,
        None,
        producer="test",
        cameras={"cam0": LEVEL},
        floor=FLOOR,
        map_bytes=b"x",
        map_format="ply",
        plan=PLAN if plan else None,
        plan_transform=TO_PLAN if plan else None,
        now=T - 100,
    ).place


def foot(vision, t, u=0.5, v=1.0, cam="cam0", **kw):
    row = dict(source=P.FIXED_CAMERA, confidence=0.9, camera=cam, u=u, v=v, track=f"{cam}:7", entity=None)
    row |= dict(place=None, room=None, x=None, y=None, revision=None)
    return vision.add(S.POSITIONS, ts_start=t, cam=cam, **(row | kw))


def test_written_rows_read_back_from_the_store_with_missing_numbers_as_none(tmp_path):
    P.write_samples(
        tmp_path,
        [
            {
                "t": T,
                "source": P.FIXED_CAMERA,
                "confidence": 0.8,
                "camera": "cam0",
                "u": 0.25,
                "v": 0.75,
                "track": "cam0:3",
            },
            {"t": T + 1, "source": P.RIG_POSE, "confidence": 0.5, "entity": "rig", "place": "p", "x": 1.5, "y": -2.0},
        ],
    )
    got = S.Log(S.root(tmp_path)).range(S.POSITIONS, T - 1, T + 5)
    assert [(r["cam"], r["u"], r["v"], r["x"], r["y"], r["entity"], r["track"]) for r in got] == [
        ("cam0", 0.25, 0.75, None, None, None, "cam0:3"),
        ("rig", None, None, 1.5, -2.0, "rig", None),
    ]


def test_nothing_to_write_writes_nothing(tmp_path):
    P.write_samples(tmp_path, [])
    assert not S.root(tmp_path).exists()


def test_samples_are_the_rows_as_written_for_the_cameras_asked(tmp_path, vision):
    foot(vision, T, cam="cam0")
    foot(vision, T + 1, cam="porch")
    got = P.samples(tmp_path, T - 1, T + 5, cameras=["porch"])
    assert [(s.camera, s.u, s.v, s.place, s.x) for s in got] == [("porch", 0.5, 1.0, None, None)]


def test_a_footprint_is_placed_on_the_plan_through_its_cameras_registration(tmp_path, vision):
    place = home(tmp_path)
    foot(vision, T)
    (p,) = P.derive(tmp_path, T - 1, T + 5)
    assert (p.place, p.x, p.y, p.room, p.room_id, p.source) == (place, 1.0, 1.93, "Living", "r1", P.FIXED_CAMERA)
    rev = places.latest(tmp_path, place)
    assert rev is not None and p.revision == rev.id


def test_a_footprint_follows_a_new_registration_with_nothing_rewritten(tmp_path, vision):
    place = home(tmp_path)
    foot(vision, T, v=0.75)  # half a metre down a metre: the feet's plane 1.93 / 0.5 = 3.86 m ahead, the hall
    assert [(p.x, p.y, p.room) for p in P.derive(tmp_path, T - 1, T + 5)] == [(1.0, 3.86, "Hall")]
    moved = places.CameraPose(LEVEL.pose[:3] + (2.0,) + LEVEL.pose[4:], 144.0, 512, 288, T + 50)
    places.revise(tmp_path, place, producer="test", cameras={"cam0": moved}, now=T + 50)  # the camera moved 1 m
    assert [(p.x, p.y) for p in P.derive(tmp_path, T - 1, T + 5)] == [(2.0, 3.86)]


class Moved:
    """A camera's pose series (spatial/poses.py's `Series`): `before` until `moved_at`, `after` from then on."""

    def __init__(self, before, after, moved_at):
        self.before, self.after, self.moved_at = before, after, moved_at

    def at(self, t):
        pose = self.before if t < self.moved_at else self.after
        return None if pose is None else np.asarray(pose.pose, dtype=np.float64).reshape(4, 4)


def with_series(monkeypatch, found):
    """`poses.series` answering `found` for every place and frame; returns the calls it got."""
    calls = []

    def series(state, place, frame, cameras):
        calls.append((place, frame, tuple(cameras)))
        return found

    monkeypatch.setattr(poses, "series", series)
    return calls


def test_a_footprint_is_placed_through_the_cameras_pose_at_its_time(tmp_path, vision, monkeypatch):
    # The camera stood 2 m along x until T + 5, then 1.5 m along z: its heels land 1.93 m ahead of where it was then
    place = home(tmp_path)
    calls = with_series(monkeypatch, {"cam0": Moved(moved(LEVEL, dx=2.0), moved(LEVEL, dz=1.5), T + 5)})
    for t in (T, T + 1, T + 10, T + 11):
        foot(vision, t)
    got = [(p.t, p.x, p.y, p.room) for p in P.derive(tmp_path, T - 1, T + 15)]
    assert got == [
        (T, 3.0, 1.93, "Living"),
        (T + 1, 3.0, 1.93, "Living"),
        (T + 10, 1.0, 3.43, "Hall"),
        (T + 11, 1.0, 3.43, "Hall"),
    ]
    rev = places.latest(tmp_path, place)
    assert rev is not None
    assert calls == [(place, rev.frame, ("cam0",))], "read once for the whole read, not per row"


@pytest.mark.parametrize(
    "found",
    [
        pytest.param({}, id="no-series"),
        pytest.param({"cam0": Moved(None, None, T)}, id="a-series-with-no-pose-then"),
        pytest.param(
            {"porch": Moved(*[replace(LEVEL, pose=LEVEL.pose[:3] + (3.0,) + LEVEL.pose[4:])] * 2, T)},
            id="another-cameras-series",
        ),
    ],
)
def test_a_camera_with_no_pose_at_its_rows_time_is_placed_through_its_registration(
    tmp_path, vision, monkeypatch, found
):
    home(tmp_path)
    with_series(monkeypatch, found)
    foot(vision, T)
    assert [(p.x, p.y, p.room) for p in P.derive(tmp_path, T - 1, T + 5)] == [(1.0, 1.93, "Living")]


def test_a_pose_from_the_series_takes_the_cameras_held_focal(tmp_path, vision, monkeypatch):
    # Held at half the frame's width (256 px against the registration's 144), the bottom middle of the frame drops
    # 144 / 256 = 0.5625 m a metre, so the heels (1.93 m below the camera) land 1.93 / 0.5625 = 3.431 m ahead.
    home(tmp_path)
    monkeypatch.setattr(lens, "as_registered", lambda camera: None)  # a pinhole: no configured lens undoes its frames
    lens.hold(tmp_path, "cam0", lens.Held(0.5, 0.0, "test"), now=T)
    with_series(monkeypatch, {"cam0": Moved(LEVEL, LEVEL, T)})
    foot(vision, T)
    assert [(p.x, p.y, p.room) for p in P.derive(tmp_path, T - 1, T + 5)] == [(1.0, 3.431, "Hall")]


@pytest.mark.parametrize(
    ("registered", "plan", "v", "want"),
    [
        # No plan: the feet's floor point in the floor's own frame (the map's x, and z as y), with no room
        pytest.param(True, False, 1.0, (True, 1.0, 1.93, None), id="place-without-a-plan"),
        pytest.param(True, True, 0.3, (True, None, None, None), id="looking-above-the-floor"),
        pytest.param(False, False, 1.0, (False, None, None, None), id="unregistered-camera"),
    ],
)
def test_a_footprint_with_no_plan_point_keeps_what_is_known(tmp_path, vision, registered, plan, v, want):
    place = home(tmp_path, plan=plan) if registered else None
    foot(vision, T, v=v)
    (p,) = P.derive(tmp_path, T - 1, T + 5)
    assert (p.place is not None, p.x, p.y, p.room) == want
    assert p.place == place


def test_a_rig_row_keeps_its_own_place_and_point_and_finds_its_room(tmp_path, vision):
    place = home(tmp_path)
    rev = places.latest(tmp_path, place)
    assert rev is not None
    rig = dict(source=P.RIG_POSE, confidence=0.6, entity="rig", track=None, camera=None, u=None, v=None, room=None)
    vision.add(S.POSITIONS, ts_start=T, cam="rig", **rig, place=place, x=2.0, y=4.5, revision=rev.id)
    (p,) = P.derive(tmp_path, T - 1, T + 5)
    assert (p.place, p.x, p.y, p.room, p.room_id, p.revision) == (place, 2.0, 4.5, "Hall", "r2", rev.id)


def test_entities_are_named_when_read(tmp_path, vision, monkeypatch):
    monkeypatch.setattr("liberational.config.owner_name", lambda: "Sam")
    alice = registry.at(tmp_path).ensure("Alice", "test")
    biscuit = pets.at(tmp_path).save("Biscuit", "dog")["id"]
    foot(vision, T, entity=f"person:{alice}")
    foot(vision, T + 1, entity=f"pet:{biscuit}")
    foot(vision, T + 2, entity="rig")
    foot(vision, T + 3, entity="person:nobody")
    foot(vision, T + 4)
    assert [p.name for p in P.derive(tmp_path, T - 1, T + 9)] == ["Alice", "Biscuit", "Sam", None, None]


def zone(vision, kind, start, end, cam="cam0", name="Kitchen"):
    vision.add(
        S.ZONE_VISITS, ts_start=start, ts_end=end, cam=cam, zone="z1", zone_name=name, subject_kind=kind, track_id="4"
    )


def test_zone_visits_are_room_only_positions_under_the_cameras_place(tmp_path, vision):
    place = home(tmp_path)
    zone(vision, "person", T, T + 30)
    zone(vision, "animal", T + 40, T + 50, name="Hall")
    zone(vision, "vehicle", T + 60, T + 70)  # neither a person nor a pet
    got = P.derive(tmp_path, T - 1, T + 100)
    assert [(p.source, p.place, p.room, p.room_id, p.x, p.t, p.end, p.track) for p in got] == [
        (P.ZONE, place, "Kitchen", None, None, T, T + 30, "cam0:4"),
        (P.ZONE, place, "Hall", None, None, T + 40, T + 50, "cam0:4"),
    ]


def test_place_and_entity_keep_only_their_rows(tmp_path, vision):
    place = home(tmp_path)
    foot(vision, T, entity="person:a")
    foot(vision, T + 1, cam="porch", entity="person:a")  # an unregistered camera: no place
    foot(vision, T + 2, entity="person:b")
    zone(vision, "person", T, T + 5)
    assert [p.t for p in P.derive(tmp_path, T - 1, T + 9, place=place)] == [T, T, T + 2]
    assert [p.t for p in P.derive(tmp_path, T - 1, T + 9, entity="person:a")] == [T, T + 1]


def test_past_the_window_a_track_keeps_one_position_every_10_seconds_and_its_last():
    rows = [
        {"id": f"{track}-{t}", "ts_start": T + t, "track": track, "source": P.FIXED_CAMERA, "camera": "cam0"}
        for track, times in (("cam0:1", (0, 1, 9, 10, 25)), ("cam0:2", (0, 4, 7)))
        for t in times
    ] + [
        {"id": "rig", "ts_start": T + 3, "track": None, "entity": "rig", "source": P.RIG_POSE, "camera": None},
        {"id": "a", "ts_start": T + 1, "track": None, "entity": None, "source": P.FIXED_CAMERA, "camera": "cam0"},
        {"id": "b", "ts_start": T + 2, "track": None, "entity": None, "source": P.FIXED_CAMERA, "camera": "cam0"},
    ]
    kept = retention.thin_positions(rows)
    assert sorted(r["id"] for r in kept) == sorted(
        ["cam0:1-0", "cam0:1-10", "cam0:1-25", "cam0:2-0", "cam0:2-7", "rig", "a", "b"]
    )  # cam0:2 keeps its last (7); two untracked anonymous rows aren't merged into one path


def test_positions_are_condensed_like_the_other_dense_streams():
    assert S.POSITIONS in S.CONDENSED
    assert S.condensed(S.POSITIONS) in S.STREAMS


def test_rooms_are_read_from_rect_and_poly_lines_and_found_by_point():
    text = PLAN + 'room r3 "Nook" other poly 4,0 6,0 4,2\ndoor r1>r2 at 2,3 w0.9\nroom bad "Odd" other rect x\n'
    rooms = floorplan.rooms_from_dsl(text)
    assert [(i, n, len(poly)) for i, n, poly in rooms] == [("r1", "Living", 4), ("r2", "Hall", 4), ("r3", "Nook", 3)]
    assert floorplan.room_at(rooms, 1.0, 1.0) == ("r1", "Living")
    assert floorplan.room_at(rooms, 4.5, 0.5) == ("r3", "Nook")
    assert floorplan.room_at(rooms, 0.0, 1.0) == ("r1", "Living")  # on a wall line
    assert floorplan.room_at(rooms, 9.0, 9.0) is None
    assert not math.isnan(rooms[0][2][2][0])


def test_a_long_read_can_be_thinned_to_paths(tmp_path, vision):
    home(tmp_path)
    for t in (0, 3, 6, 9, 12, 14):
        foot(vision, T + t)
    assert [p.t - T for p in P.derive(tmp_path, T - 1, T + 20, every=10)] == [0, 12, 14]


def test_a_place_or_entity_read_places_only_its_own_rows(tmp_path, vision, monkeypatch):
    place = home(tmp_path)
    places.revise(tmp_path, None, producer="test", cameras={"porch": LEVEL}, floor=FLOOR, now=T - 50)
    foot(vision, T, entity="person:a")
    foot(vision, T + 1, cam="porch", entity="person:a")
    foot(vision, T + 2, entity="person:b")
    projected = []
    real = places.project
    monkeypatch.setattr(
        places, "project", lambda rev, cam, u, v, height=0.0: projected.append(cam) or real(rev, cam, u, v, height)
    )
    assert [p.t for p in P.derive(tmp_path, T - 1, T + 9, place=place) if p.source != P.ZONE] == [T, T + 2]
    assert projected == ["cam0", "cam0"]  # the porch's place was never read
    projected.clear()
    assert [p.t for p in P.derive(tmp_path, T - 1, T + 9, entity="person:b")] == [T + 2]
    assert projected == ["cam0"]


PLAN_2 = 'plan "Home" walls 0.2/0.1\nroom d1 "Den" other rect 0,0 6x6\n'


def test_a_rig_rows_room_comes_from_the_plan_of_the_revision_its_point_is_in(tmp_path, vision):
    place = home(tmp_path)
    old = places.latest(tmp_path, place)
    places.revise(tmp_path, place, producer="test", plan=PLAN_2, plan_transform=TO_PLAN, now=T)  # a new frame
    rig = dict(source=P.RIG_POSE, confidence=0.6, entity="rig", track=None, camera=None, u=None, v=None, room=None)
    assert old is not None
    vision.add(S.POSITIONS, ts_start=T + 1, cam="rig", **rig, place=place, x=1.0, y=1.0, revision=old.id)
    vision.add(S.POSITIONS, ts_start=T + 2, cam="rig", **rig, place=place, x=1.0, y=1.0, revision="gone")
    got = P.derive(tmp_path, T, T + 5)
    assert [(p.room, p.room_id) for p in got] == [("Living", "r1"), (None, None)]  # an unknown frame names no room


OWNER_PLAN = 'plan "Home" walls 0.2/0.1\nroom r1 "Lounge" other rect 0,0 4x3\n'


def test_the_owners_plan_names_rooms_only_while_no_newer_machine_plan_moved_the_frame(tmp_path, vision):
    place = home(tmp_path)
    places.correct_plan(tmp_path, place, OWNER_PLAN, now=T - 50)
    foot(vision, T)
    assert [p.room for p in P.derive(tmp_path, T - 1, T + 5)] == ["Lounge"]
    places.revise(tmp_path, place, producer="test", plan=PLAN_2, plan_transform=TO_PLAN, now=T - 10)
    assert [p.room for p in P.derive(tmp_path, T - 1, T + 5)] == ["Den"]


# The next night's frame: the same map moved 2 m along the plan's x.
TO_PLAN_SHIFTED = (1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0)


def two_nights(tmp_path):
    """A place whose first revision draws its plan in TO_PLAN's frame and whose newest in a frame moved 2 m."""
    place = home(tmp_path)
    first = places.latest(tmp_path, place)
    newest = places.revise(
        tmp_path, place, producer="test", plan=PLAN, plan_transform=TO_PLAN_SHIFTED, cameras={"cam0": LEVEL}, now=T
    )
    assert first is not None
    return place, first, newest


def test_camera_rows_are_put_in_the_frame_asked_for(tmp_path, vision):
    place, first, newest = two_nights(tmp_path)
    foot(vision, T + 1)
    read = lambda frame: [
        (p.x, p.y, p.revision, p.in_frame) for p in P.derive(tmp_path, T, T + 5, place, frame_revision=frame)
    ]
    assert read(None) == [(3.0, 1.93, newest.id, True)]  # the current frame
    assert read(first.id) == [(1.0, 1.93, first.id, True)]  # the first night's frame: same raw u, v
    assert read(newest.id) == [(3.0, 1.93, newest.id, True)]


def test_rows_that_cant_be_put_in_the_frame_keep_their_own_and_are_flagged(tmp_path, vision):
    place, first, newest = two_nights(tmp_path)
    rig = dict(source=P.RIG_POSE, confidence=0.6, entity="rig", track=None, camera=None, u=None, v=None, room=None)
    vision.add(S.POSITIONS, ts_start=T + 1, cam="rig", **rig, place=place, x=1.0, y=1.0, revision=first.id)
    # porch only, in a new map with no plan and no camera in common with the newest revision
    places.revise(
        tmp_path,
        place,
        producer="test",
        cameras={"porch": LEVEL},
        floor=FLOOR,
        map_bytes=b"z",
        map_format="ply",
        now=T + 2,
    )
    foot(vision, T + 3, cam="porch")
    got = P.derive(tmp_path, T, T + 5, place, frame_revision=newest.id)
    assert [(p.source, p.x, p.y, p.in_frame) for p in got] == [
        (P.RIG_POSE, 1.0, 1.0, False),  # its point is in the first night's frame
        (P.FIXED_CAMERA, 1.0, 1.93, False),  # its own floor's frame: that revision can't be related to the porch's
    ]
    assert [p.in_frame for p in P.derive(tmp_path, T, T + 2, place, frame_revision=first.id)] == [True]


def moved(pose, dx=0.0, dz=0.0):
    """`pose` shifted along the map's floor (x and z, MASt3R's frame)."""
    m = list(pose.pose)
    m[3] += dx
    m[11] += dz
    return places.CameraPose(tuple(m), pose.focal, pose.width, pose.height, pose.at)


OWNER_ROOMS = 'plan "Home" walls 0.2/0.1\nroom r1 "Lounge" other rect 0,0 4x3\nroom r2 "Bedroom" other rect 0,3 4x3\n'


@pytest.mark.parametrize(("place", "frame"), [(None, "any"), ("given", "no-such-revision")])
def test_a_frame_needs_its_place_and_must_exist(tmp_path, place, frame):
    real = home(tmp_path)
    with pytest.raises(ValueError, match="revision"):
        P.derive(tmp_path, T, T + 5, real if place else None, frame_revision=frame)


def test_the_holding_plans_frame_is_named_on_the_place(tmp_path):
    place, _, newest = two_nights(tmp_path)
    held = places.get(tmp_path, place)
    assert held is not None and held.plan_revision == newest.id  # a machine plan: the newest revision's frame
    places.correct_plan(tmp_path, place, OWNER_PLAN, now=T + 10)
    places.revise(tmp_path, place, producer="test", cameras={"cam0": LEVEL}, now=T + 20)  # carries the plan over
    held = places.get(tmp_path, place)
    assert held is not None and (held.plan_source, held.plan_revision) == ("owner", newest.id)


def test_the_owners_plan_names_rooms_in_any_revision_drawn_in_its_frame(tmp_path, vision):
    place, first, newest = two_nights(tmp_path)
    places.correct_plan(tmp_path, place, OWNER_PLAN, now=T + 10)
    places.revise(tmp_path, place, producer="test", cameras={"cam0": LEVEL}, now=T + 20)  # same frame, carried over
    foot(vision, T + 1, v=1.0)  # plan (3, 2) in the newest frame, (1, 2) in the first
    # The correction was made on the newest revision's plan, so it names rooms in that frame (and in the later
    # revision that carries the same plan over); the first revision's own machine plan names them in its frame.
    assert [p.room for p in P.derive(tmp_path, T, T + 5, place, frame_revision=newest.id)] == ["Lounge"]
    assert [p.room for p in P.derive(tmp_path, T, T + 5, place)] == ["Lounge"]
    assert [p.room for p in P.derive(tmp_path, T, T + 5, place, frame_revision=first.id)] == ["Living"]


def test_a_rig_row_from_another_plan_with_equal_transform_values_is_out_of_frame(tmp_path, vision):
    place = home(tmp_path)
    first = places.latest(tmp_path, place)
    newest = places.revise(tmp_path, place, producer="test", plan=PLAN_2, plan_transform=TO_PLAN, now=T)
    assert first is not None and first.plan_transform == newest.plan_transform  # equal values, different plans
    rig = dict(source=P.RIG_POSE, confidence=0.6, entity="rig", track=None, camera=None, u=None, v=None, room=None)
    vision.add(S.POSITIONS, ts_start=T + 1, cam="rig", **rig, place=place, x=1.0, y=1.0, revision=first.id)
    (p,) = P.derive(tmp_path, T, T + 5, place, frame_revision=newest.id)
    assert not p.in_frame


def test_a_camera_run_without_a_plan_leaves_no_plan_from_another_frame(tmp_path, vision):
    place = home(tmp_path)
    # the next camera run: a new map (a new frame) whose cloud closed no room, so it has no plan of its own
    places.revise(tmp_path, place, producer="test", map_bytes=b"y", map_format="ply", cameras={"cam0": LEVEL}, now=T)
    held = places.get(tmp_path, place)
    assert held is not None and (held.plan, held.plan_revision) == (None, None)
    foot(vision, T + 1)
    assert [(p.place, p.x, p.room) for p in P.derive(tmp_path, T, T + 5)] == [(place, None, None)]


def test_samples_read_only_the_asked_cameras_slots_from_the_store(tmp_path, monkeypatch):
    # The slot filter ran after every slot was read, so it saved nothing; the store now reads only those slots.
    monkeypatch.setattr(S, "at", lambda state=None: S.Log(S.root(tmp_path)))
    P.write_samples(
        tmp_path,
        [
            {"t": T, "source": P.FIXED_CAMERA, "confidence": 0.9, "camera": "cam0", "u": 0.5, "v": 0.9},
            {"t": T + 1, "source": P.FIXED_CAMERA, "confidence": 0.9, "camera": "porch", "u": 0.5, "v": 0.9},
            {"t": T + 2, "source": P.RIG_POSE, "confidence": 0.5, "entity": "rig", "place": "p", "x": 1.0, "y": 1.0},
        ],
    )
    read = []
    real = S.Log.range
    monkeypatch.setattr(
        S.Log,
        "range",
        lambda self, *a, **k: read.append(a[3] if len(a) > 3 else k.get("cameras")) or real(self, *a, **k),
    )
    assert [s.camera for s in P.samples(tmp_path, T - 1, T + 5, cameras=["porch"])] == ["porch"]
    assert read == [["porch"]]  # asked of the store, not filtered after
    assert [s.source for s in P.samples(tmp_path, T - 1, T + 5, cameras=["porch", P.RIG])] == [
        P.FIXED_CAMERA,
        P.RIG_POSE,
    ]
    assert P.samples(tmp_path, T - 1, T + 5, cameras=[]) == []


@pytest.mark.parametrize(
    ("track", "want"),
    [
        pytest.param("cam0:r1a2:cat:12", "r1a2:cat:12", id="animal"),
        pytest.param("cam0:1730000000000", None, id="person"),
        pytest.param("cam0:r1a2:Cat:12", None, id="species-not-lowercase"),
        pytest.param(None, None, id="no-track"),
    ],
)
def test_an_animals_track_is_told_from_a_persons(track, want):
    assert P.animal_track(track) == want


def test_a_persons_track_is_named_by_the_faces_matched_on_it_when_they_agree(tmp_path, vision):
    home(tmp_path)
    alice = registry.at(tmp_path).ensure("Alice", "test")
    registry.at(tmp_path).ensure("Dana", "test")
    for t in (1, 2, 3):
        vision.face(T + t, name="Alice", track_id="7")
    vision.face(T + 1, name="Alice", track_id="8")  # two people on one track: nobody
    vision.face(T + 2, name="Dana", track_id="8")
    for t in (1, 2):  # the same number on another camera is another track
        vision.face(T + t, cam="porch", name="Dana", track_id="7")
    vision.face(T + 4, track_id="9")  # a face nobody is named for
    for track in ("cam0:7", "cam0:8", "cam0:9"):
        foot(vision, T + 1, track=track)
    got = P.derive(tmp_path, T, T + 5)
    assert [(p.track, p.entity, p.name) for p in got] == [
        ("cam0:7", f"person:{alice}", "Alice"),
        ("cam0:8", None, None),
        ("cam0:9", None, None),
    ]
    assert [p.track for p in P.derive(tmp_path, T, T + 5, entity=f"person:{alice}")] == ["cam0:7"]


def test_faces_for_many_cameras_are_read_once_per_request(tmp_path, vision, monkeypatch):
    from liberational.vision import sightings

    two_cameras(tmp_path)
    alice = registry.at(tmp_path).ensure("Alice", "test")
    for t in (1, 2, 3):
        vision.face(T + t, name="Alice", track_id="7")
        vision.face(T + t, cam="living", name="Alice", track_id="4")
    seen(vision, "cam0:7", [1])
    seen(vision, "living:4", [1])
    calls = 0
    real = sightings.faces_many

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(sightings, "faces_many", counted)
    got = {p.track: p.entity for p in P.derive(tmp_path, T, T + 5)}
    assert calls == 1
    assert got == {"cam0:7": f"person:{alice}", "living:4": f"person:{alice}"}


def test_pet_recognition_is_batched_once_across_cameras(tmp_path, vision, monkeypatch):
    two_cameras(tmp_path)
    pets.at(tmp_path).save("Miso", "cat")
    seen(vision, "cam0:r:cat:1", [1])
    seen(vision, "living:r:cat:2", [1])
    calls: list[dict[str, set[str]]] = []

    def fake_resolve_many(state, start, end, tracks_by_camera, listed_pets):
        calls.append({cam: set(tracks) for cam, tracks in tracks_by_camera.items()})
        return {}

    monkeypatch.setattr("liberational.people.pet_recognition.resolve_many", fake_resolve_many)
    P.derive(tmp_path, T, T + 5)
    assert calls == [{"cam0": {"r:cat:1"}, "living": {"r:cat:2"}}]


def test_entity_names_refresh_registry_once_per_read(tmp_path, vision, monkeypatch):
    home(tmp_path)
    ids = [registry.at(tmp_path).ensure(name, "test") for name in ("Alice", "Dana", "Eve")]
    for i, pid in enumerate(ids):
        foot(vision, T + i, track=f"cam0:{10 + i}", entity=f"person:{pid}")
    calls = 0
    real = registry.Registry.refresh

    def counted(self, *streams):
        nonlocal calls
        calls += 1
        return real(self, *streams)

    monkeypatch.setattr(registry.Registry, "refresh", counted)
    assert [p.name for p in P.derive(tmp_path, T, T + 5)] == ["Alice", "Dana", "Eve"]
    assert calls == 1


def test_an_unnamed_persons_tracks_share_their_face_group_on_every_camera(tmp_path, vision):
    # Tracks whose faces agree on an unnamed group ("Face 3") are one entity, still each its own path
    home(tmp_path)
    alice = registry.at(tmp_path).ensure("Alice", "test")
    for t in (1, 2, 3):
        vision.face(T + t, cluster="Face 3", track_id="7")
        vision.face(T + t, cam="porch", cluster="Face 3", track_id="4")
    vision.face(T + 1, cluster="Face 3", track_id="8")  # named faces decide: a group never outvotes them
    for t in (2, 3, 4):
        vision.face(T + t, name="Alice", track_id="8")
    vision.face(T + 1, cluster="Face 3", track_id="9")  # two groups on one track: nobody
    vision.face(T + 2, cluster="Face 5", track_id="9")
    for track in ("cam0:7", "porch:4", "cam0:8", "cam0:9"):
        foot(vision, T + 1, cam=track.partition(":")[0], track=track)
    got = {p.track: (p.entity, p.name) for p in P.derive(tmp_path, T, T + 5)}
    assert got == {
        "cam0:7": ("face:3", "Face 3"),
        "porch:4": ("face:3", "Face 3"),
        "cam0:8": (f"person:{alice}", "Alice"),
        "cam0:9": (None, None),
    }
    assert {p.track for p in P.derive(tmp_path, T, T + 5, entity="face:3")} == {"cam0:7", "porch:4"}


def test_a_face_groups_tracks_follow_the_name_the_owner_gives_it(tmp_path, vision):
    home(tmp_path)
    for t in (1, 2):
        vision.face(T + t, cluster="Face 3", track_id="7")
    foot(vision, T + 1, track="cam0:7")
    assert [p.entity for p in P.derive(tmp_path, T, T + 5)] == ["face:3"]
    reg = registry.at(tmp_path)
    reg.claim(registry.face_group(3), "Kara", Basis.OWNER_TAG, "test")
    kara = reg.person_id("Kara")
    assert [(p.entity, p.name) for p in P.derive(tmp_path, T, T + 5)] == [(f"person:{kara}", "Kara")]


def test_a_face_just_outside_the_read_still_names_its_track(tmp_path, vision):
    home(tmp_path)
    alice = registry.at(tmp_path).ensure("Alice", "test")
    vision.face(T + 5 + P.LOOK_S - 1, name="Alice", track_id="7")  # they turned to the camera after the window
    foot(vision, T + 1)
    assert [p.entity for p in P.derive(tmp_path, T, T + 5)] == [f"person:{alice}"]


def test_a_zone_visit_is_named_by_its_tracks_faces(tmp_path, vision):
    home(tmp_path)
    alice = registry.at(tmp_path).ensure("Alice", "test")
    vision.face(T + 1, name="Alice", track_id="4")
    zone(vision, "person", T, T + 30)
    assert [(p.source, p.entity, p.name) for p in P.derive(tmp_path, T - 1, T + 40)] == [
        (P.ZONE, f"person:{alice}", "Alice")
    ]


def pet_label(state, track, label):
    import uuid

    from liberational import knowledge
    from liberational.people import pet_recognition

    ns = round(T * 1e9)
    body = {"track_id": track, "label": label, "active": True, "source": "owner", "species": "cat", "cam": "cam0"}
    record = knowledge.Record(str(uuid.uuid7()), pet_recognition.LABELS, track, ns, ns, ns, body)
    knowledge.Store(registry.knowledge_path(state)).put([record])


def test_an_animals_track_is_named_by_the_owners_label_of_it(tmp_path, vision):
    home(tmp_path)
    miso = pets.at(tmp_path).save("Miso", "cat")["id"]
    pet_label(tmp_path, "r:cat:1", miso)
    pet_label(tmp_path, "r:cat:2", "other")  # a visiting cat: an animal, but none of the pets
    for t, track in enumerate(("cam0:r:cat:1", "cam0:r:cat:2", "cam0:r:cat:3")):
        foot(vision, T + t, track=track)
    assert [(p.entity, p.name) for p in P.derive(tmp_path, T - 1, T + 5)] == [
        (f"pet:{miso}", "Miso"),
        (None, None),
        (None, None),
    ]


def embedding(state, ts, track, identity, k):
    """One animal embedding: 0.95 along the animal's own axis, 0.31 along the sighting's."""
    import numpy as np

    v = np.zeros(384)
    v[identity], v[100 + k] = 0.95, 0.31
    row = {"ts_start": ts, "cam": "cam0", "track_id": track, "species": "dog", "box": [0.1, 0.2, 0.4, 0.6, 0.9]}
    writers = S.Writers(S.root(state))
    row |= {"id": f"emb-{track}-{k}", "vector": (v / np.linalg.norm(v)).tolist(), "model": animal_embedder.space()}
    writers.append(S.ANIMAL_EMBEDDINGS, "cam0", [row])
    writers.close()


def test_an_unlabeled_animals_track_is_named_by_pet_recognition(tmp_path, vision):
    from liberational.people import pet_recognition

    home(tmp_path)
    bess = pets.at(tmp_path).save("Bess", "dog")
    for k in range(6):  # six visits an hour apart, labeled on the Pets page
        embedding(tmp_path, T - 30000 + 3600 * k, f"bess{k}", 5, k)
    pet_recognition.label(tmp_path, [f"bess{k}" for k in range(6)], bess["id"], [bess], now=T)
    embedding(tmp_path, T + 1, "r:dog:1", 5, 10)  # looks like Bess
    embedding(tmp_path, T + 1, "r:dog:2", 6, 11)  # looks like nobody labeled
    foot(vision, T + 1, track="cam0:r:dog:1")
    foot(vision, T + 2, track="cam0:r:dog:2")
    assert [p.entity for p in P.derive(tmp_path, T, T + 5)] == [f"pet:{bess['id']}", None]


def test_pet_recognition_failing_leaves_animals_unnamed_but_still_placed(tmp_path, vision, monkeypatch):
    from liberational.people import pet_recognition

    home(tmp_path)
    pets.at(tmp_path).save("Bess", "dog")

    def broken(*args, **kwargs):
        raise OSError("embeddings unreadable")

    monkeypatch.setattr(pet_recognition, "resolve", broken)
    foot(vision, T + 1, track="cam0:r:dog:1")
    assert [(p.entity, p.x, p.y) for p in P.derive(tmp_path, T, T + 5)] == [(None, 1.0, 2.0)]


# ---- a track no face names, by the one track another camera saw at the same time


def two_cameras(tmp_path, backgrounds=None):
    """One place seen by cam0 and living, both registered in it; `backgrounds` gives a camera a background that is
    one height (a fraction of the camera's 2 m) over its whole frame."""
    poses = {
        cam: replace(LEVEL, background=places.Background(1, 1, ((backgrounds or {})[cam],)))
        if cam in (backgrounds or {})
        else LEVEL
        for cam in ("cam0", "living")
    }
    return places.revise(
        tmp_path,
        None,
        producer="test",
        cameras=poses,
        floor=FLOOR,
        map_bytes=b"x",
        map_format="ply",
        plan=PLAN,
        plan_transform=TO_PLAN,
        now=T - 100,
    )


def test_a_read_fills_a_cameras_pose_with_its_background_once(tmp_path):
    # Every sample of a read asks for its camera's pose with the current background: one pose per camera
    rev = two_cameras(tmp_path, backgrounds={"cam0": 0.5})
    where = P._Places(tmp_path)
    first = where.camera(rev, "cam0")
    assert first is not None and first.background == places.Background(1, 1, (0.5,))
    assert where.camera(rev, "cam0") is first


def seen(vision, track, seconds, u=0.5, v=1.0):
    """`track` in view for `seconds`; at u 0.5 both cameras put it at plan (1, 2), and each 0.1 of u is 0.69 m along x."""
    for t in seconds:
        foot(vision, T + t, u=u, v=v, cam=track.partition(":")[0], track=track)


FAR = 0.9  # 2.7 m from u 0.5 on the plan: never beside it
NEARBY = 0.66  # 1.1 m from u 0.5: past JOIN_M, within RIVAL_M


@pytest.fixture
def alice_on_cam0(tmp_path, vision):
    """Alice's face names cam0:7, in view for seconds 0 to 9 at plan (1, 2)."""
    two_cameras(tmp_path)
    alice = registry.at(tmp_path).ensure("Alice", "test")
    for t in (1, 2, 3):
        vision.face(T + t, name="Alice", track_id="7")
    seen(vision, "cam0:7", range(10))
    return f"person:{alice}"


def entities(tmp_path):
    """Who each track was: a fused point on a joined path says it for every track of its path."""
    return {t: p.entity for p in P.derive(tmp_path, T - 1, T + 20) for t in (p.path or "").split("+") if t}


# ---- one entity's tracks on two cameras fused into one path


def sin2(v):
    """The squared sine of LEVEL's ray to the floor at row v of the frame (u 0.5): its slope is (v - 0.5) * 2."""
    slope = (v - 0.5) * 288 / 144
    return slope**2 / (slope**2 + 1)


def test_one_persons_tracks_on_two_cameras_are_one_path_fused_where_both_saw_them(tmp_path, vision, alice_on_cam0):
    # living:3 has no face; it was with Alice for seconds 2 to 7, its ray to her feet shallower than cam0's
    v = 0.5 + (2 / 3) * 144 / 288  # a ray falling 2 m over 3 m (cam0's at the frame's bottom falls 2 m over 2 m)
    seen(vision, "living:3", range(2, 8), v=v)
    got = P.derive(tmp_path, T - 1, T + 20)
    assert {p.path for p in got} == {"cam0:7+living:3"}
    # each camera casts a person's foot onto the plane FOOT_M up, 2 - FOOT_M below it, so it lands that over its slope
    drop = 2 - footprints.FOOT_M
    cam0, living = drop, drop * 1.5
    alone = [(p.t - T, p.camera, p.y) for p in got if p.track is not None]
    assert alone == [(t, "cam0", pytest.approx(cam0, abs=0.001)) for t in (0, 1, 8, 9)]
    fused = [p for p in got if p.track is None]
    # each second's point is the two cameras' weighted by their rays' steepness: cam0's at 45 degrees, living's shallower
    want = (sin2(1.0) * cam0 + sin2(v) * living) / (sin2(1.0) + sin2(v))
    assert [p.t - T for p in fused] == [2, 3, 4, 5, 6, 7]
    assert all(p.camera is None and p.entity == alice_on_cam0 for p in fused)
    assert [p.y for p in fused] == pytest.approx([want] * 6, abs=0.002)
    assert cam0 < want < (cam0 + living) / 2  # nearer the steeper camera's point than the plain mean


def test_one_persons_tracks_far_apart_on_the_plan_stay_two_paths(tmp_path, vision, alice_on_cam0):
    # both named Alice by their faces, but 2.7 m apart: not one path (one may be named wrong)
    for t in (3, 4, 5):
        vision.face(T + t, cam="living", name="Alice", track_id="5")
    seen(vision, "living:5", range(2, 8), u=FAR)
    got = P.derive(tmp_path, T - 1, T + 20)
    assert {p.path for p in got} == {"cam0:7", "living:5"} and all(p.track is not None for p in got)


def test_plan_points_are_where_the_tests_say(tmp_path, vision, alice_on_cam0):
    seen(vision, "living:3", [2], u=FAR)
    seen(vision, "living:4", [2], u=NEARBY)
    xs = {p.track: p.x for p in P.derive(tmp_path, T + 2, T + 3)}
    assert xs == {
        "cam0:7": pytest.approx(1.0),
        "living:3": pytest.approx(3.74, abs=0.01),
        "living:4": pytest.approx(2.10, abs=0.01),
    }


@pytest.mark.parametrize("u", [pytest.param(0.5, id="beside-her"), pytest.param(FAR, id="far-from-her")])
def test_a_faceless_track_is_the_one_person_another_camera_saw_at_the_same_time(tmp_path, vision, alice_on_cam0, u):
    seen(vision, "living:3", range(2, 8), u=u)  # six seconds, every one of them with only cam0:7 in view
    assert entities(tmp_path) == {"cam0:7": alice_on_cam0, "living:3": alice_on_cam0}


def test_a_faceless_track_is_the_person_beside_it_on_the_plan_when_others_are_in_view_too(
    tmp_path, vision, alice_on_cam0
):
    seen(vision, "cam0:8", range(10), u=FAR)  # someone else in cam0's view, across the room
    seen(vision, "living:3", range(2, 8))
    assert entities(tmp_path)["living:3"] == alice_on_cam0


@pytest.mark.parametrize(
    ("extra", "seconds", "u"),
    [
        pytest.param({}, range(2, 2 + P.CO_MIN_S - 1), 0.5, id="seen-together-too-briefly"),
        pytest.param({}, range(4, 14), FAR, id="mostly-seen-without-her-and-far-away"),
        pytest.param({"cam0:8": (range(10), FAR)}, range(2, 8), FAR, id="two-people-in-view-and-beside-neither"),
        pytest.param({"cam0:8": (range(10), NEARBY)}, range(2, 8), 0.5, id="beside-her-but-someone-else-close-too"),
    ],
)
def test_a_faceless_track_stays_unnamed_when_who_it_was_with_is_not_clear(
    tmp_path, vision, alice_on_cam0, extra, seconds, u
):
    for track, (also, at) in extra.items():
        seen(vision, track, also, u=at)
    seen(vision, "living:3", seconds, u=u)
    assert entities(tmp_path)["living:3"] is None


@pytest.mark.parametrize(("second", "want"), [("Alice", "alice"), ("Dana", "face:3")])
def test_a_face_group_track_is_the_one_person_its_partners_faces_named(tmp_path, vision, alice_on_cam0, second, want):
    # living-room sees only the back of a named person's head, so its track gets a fresh unnamed group. cam0
    # saw her face on two tracks in a row, each with living:3 alone in view (so each is living:3's partner, though
    # living:3, seeing two, has none of its own). When their faces name one person, the group yields to her.
    registry.at(tmp_path).ensure("Dana", "test")
    for t in (13, 14, 15):
        vision.face(T + t, name=second, track_id="8")
    seen(vision, "cam0:8", range(10, 20))
    for t in (2, 3, 4):
        vision.face(T + t, cam="living", cluster="Face 3", track_id="3")
    seen(vision, "living:3", range(20))
    got = entities(tmp_path)
    assert got["living:3"] == (alice_on_cam0 if want == "alice" else want)
    assert got["cam0:7"] == alice_on_cam0


def test_a_face_group_track_never_yields_to_a_person_the_owner_said_it_isnt(tmp_path, vision, alice_on_cam0):
    # As above, cam0's tracks partnering living:3 are Alice's, but the owner marked Face 3 not Alice
    face_nots.mark(tmp_path, "Alice", [3])
    for t in (2, 3, 4):
        vision.face(T + t, cam="living", cluster="Face 3", track_id="3")
    seen(vision, "living:3", range(10))
    assert entities(tmp_path)["living:3"] == "face:3"


def test_an_animal_is_joined_with_its_own_kind_and_never_with_a_person(tmp_path, vision):
    two_cameras(tmp_path)
    miso = pets.at(tmp_path).save("Miso", "cat")["id"]
    pet_label(tmp_path, "r:cat:1", miso)
    seen(vision, "cam0:r:cat:1", range(10))
    seen(vision, "living:r:cat:9", range(2, 8))
    seen(vision, "living:4", range(2, 8))  # a person in view, and beside, only the cat
    got = entities(tmp_path)
    assert (got["living:r:cat:9"], got["living:4"]) == (f"pet:{miso}", None)


def test_a_path_never_holds_two_tracks_one_camera_saw_at_the_same_time(tmp_path, vision, alice_on_cam0):
    # cam0:8 is named Alice too, at the same time as cam0:7 and 0.27 m from her, and so is a living track beside both:
    # it joins the nearer one only
    for t in (3, 4, 5):
        vision.face(T + t, name="Alice", track_id="8")
        vision.face(T + t, cam="living", name="Alice", track_id="3")
    seen(vision, "cam0:8", range(10), u=0.54)
    seen(vision, "living:3", range(2, 8), u=0.51)
    assert {p.path for p in P.derive(tmp_path, T - 1, T + 20)} == {"cam0:7+living:3", "cam0:8"}


# A background 1.2 m up (a dresser's top, an occluding edge: down rays falling 0.9 and 2/3 of a metre a metre, it hides
# (1.2 - 0.07) / 0.9 = 1.26 m and 1.70 m of floor, over footprints.EDGE_M), and the floor itself
COUCH, FLOOR_SEEN = 0.6, 0.0


@pytest.mark.parametrize(
    ("cam0_sees", "living_sees", "trusted"),
    [
        pytest.param(COUCH, FLOOR_SEEN, "living", id="cam0-at-an-edge-defers-to-living"),
        pytest.param(FLOOR_SEEN, COUCH, "cam0", id="living-at-an-edge-defers-to-cam0"),
        pytest.param(COUCH, COUCH, "both", id="both-at-an-edge-neither-preferred"),
        pytest.param(FLOOR_SEEN, FLOOR_SEEN, "both", id="neither-at-an-edge"),
    ],
)
def test_a_fused_point_defers_to_the_camera_whose_foot_is_not_on_an_occluding_edge(
    tmp_path, vision, cam0_sees, living_sees, trusted
):
    # A foot point on a couch's top is hidden and guessed, or just beyond it; the other camera can tell
    two_cameras(tmp_path, {"cam0": cam0_sees, "living": living_sees})
    alice = registry.at(tmp_path).ensure("Alice", "test")
    for t in (1, 2, 3):
        vision.face(T + t, name="Alice", track_id="7")
    near, far = 0.95, 0.5 + (2 / 3) * 144 / 288  # rays falling 0.9 and 2/3 of a metre a metre (inside the frame)
    seen(vision, "cam0:7", range(10), v=near)
    seen(vision, "living:3", range(2, 8), v=far)
    fused = [p for p in P.derive(tmp_path, T - 1, T + 20) if p.track is None]
    drop = 2 - footprints.FOOT_M
    cam0, living = drop / 0.9, drop * 1.5
    both = (sin2(near) * cam0 + sin2(far) * living) / (sin2(near) + sin2(far))
    want = {"cam0": cam0, "living": living, "both": both}[trusted]
    assert [p.entity for p in fused] == [f"person:{alice}"] * 6
    assert [p.y for p in fused] == pytest.approx([want] * 6, abs=0.002)


# ---- a track broken in two on one camera


def test_a_track_broken_in_two_on_one_camera_is_one_person_and_one_path(tmp_path, vision, alice_on_cam0):
    # cam0:7 (Alice) ends at second 9; cam0:8 starts 4 s later 0.69 m away, inside 0.6 m + 1 m/s of the gap
    seen(vision, "cam0:8", range(13, 18), u=0.6)
    got = P.derive(tmp_path, T - 1, T + 20)
    assert {(p.track, p.entity, p.path) for p in got} == {
        ("cam0:7", alice_on_cam0, "cam0:7+cam0:8"),
        ("cam0:8", alice_on_cam0, "cam0:7+cam0:8"),
    }


@pytest.mark.parametrize(
    ("after", "extra"),
    [
        pytest.param({"cam0:8": (range(9 + int(P.HANDOFF_S) + 2, 30), 0.5)}, {}, id="started-too-long-after"),
        pytest.param({"cam0:8": (range(11, 16), FAR)}, {}, id="too-far-to-have-walked-there"),
        pytest.param({"cam0:8": (range(12, 16), 0.5)}, {"cam0:9": (range(13, 17), 0.55)}, id="two-could-be-her"),
        pytest.param({"living:8": (range(12, 16), 0.5)}, {}, id="on-another-camera"),
    ],
)
def test_a_track_that_may_not_be_the_one_before_stays_unnamed(tmp_path, vision, alice_on_cam0, after, extra):
    for track, (seconds, u) in {**after, **extra}.items():
        seen(vision, track, seconds, u=u)
    got = entities(tmp_path) | {p.track: p.entity for p in P.derive(tmp_path, T - 1, T + 40) if p.track}
    assert got[next(iter(after))] is None


# ---- feet on an occluding edge outside a fused path


def on_path_of(track, got):
    """The rows drawn on `track`'s path: its own, and any fused or joined with it."""
    return [p for p in got if track in (p.path or "").split("+")]


def named_on(vision, cam, track_id, name):
    """`name`'s faces on the track; a "Face N" is an unnamed group's."""
    group = name.startswith("Face ")
    for t in (3, 4, 5):
        vision.face(T + t, cam=cam, name=None if group else name, cluster=name if group else None, track_id=track_id)


@pytest.mark.parametrize(
    ("living_name", "deferred"),
    [
        pytest.param("Alice", True, id="same-person-by-name"),
        pytest.param("Dana", False, id="someone-else"),
    ],
)
def test_a_foot_on_an_edge_is_put_where_the_other_camera_saw_the_same_person_that_second(
    tmp_path, vision, living_name, deferred
):
    # cam0 sees Alice's feet on a 1.2 m occluder's top, 1.7 m from where living-room (seeing the floor) puts her: too
    # far apart for one path, but where living-room saw her is where she was
    two_cameras(tmp_path, {"cam0": COUCH, "living": FLOOR_SEEN})
    for name in {"Alice", living_name}:
        registry.at(tmp_path).ensure(name, "test")
    named_on(vision, "cam0", "7", "Alice")
    named_on(vision, "living", "3", living_name)
    seen(vision, "cam0:7", range(10), v=0.95)
    seen(vision, "living:3", range(10), v=0.75)  # a ray falling half a metre a metre
    got = P.derive(tmp_path, T - 1, T + 20)
    cam0 = on_path_of("cam0:7", got)
    living_y = (2 - footprints.FOOT_M) / 0.5
    assert len(cam0) == 10
    if deferred:  # one person drawn once: one path, a point a second, where living-room saw her
        assert {p.path for p in got} == {"cam0:7+living:3"}
        assert all(p.track is None and p.y == pytest.approx(living_y, abs=0.002) and not p.uncertain for p in cam0)
    else:
        assert {p.path for p in got} == {"cam0:7", "living:3"}
        assert all(p.y == pytest.approx((2 - footprints.FOOT_M) / 0.9, abs=0.002) and p.uncertain for p in cam0)
    assert not any(p.uncertain for p in got if p.camera == "living")


def test_a_foot_on_an_edge_only_one_camera_saw_is_uncertain(tmp_path, vision):
    two_cameras(tmp_path, {"cam0": COUCH, "living": FLOOR_SEEN})
    seen(vision, "cam0:7", range(3), v=0.95)  # behind the couch, nobody else in view
    seen(vision, "living:3", range(5, 8), v=0.95)  # later, on living-room's floor
    got = {p.camera: p.uncertain for p in P.derive(tmp_path, T - 1, T + 20)}
    assert got == {"cam0": True, "living": False}


# cam0's ray at u 0.5, v 0.95 meets its feet's plane 2.14 m out; the 1.2 m occluder at that pixel hides the floor along
# it from (2 - 1.2) / (2 - 0.07) of the way, 0.89 m out. living's at v 1.0 puts someone 1.93 m out, on that stretch.
ON_RAY = (0.5, 1.0)


@pytest.mark.parametrize(
    ("other", "names", "settled"),
    [
        pytest.param((NEARBY, 1.0), {}, True, id="the-other-off-the-ray"),
        pytest.param((0.5, 0.95), {}, False, id="both-on-the-ray"),
        pytest.param(
            (NEARBY, 1.0),
            {("cam0", "7"): "Alice", ("living", "3"): "Dana", ("living", "4"): "Eve"},
            False,
            id="on-the-ray-named-apart",
        ),
        pytest.param(
            (NEARBY, 1.0),
            {("cam0", "7"): "Face 3", ("living", "3"): "Dana", ("living", "4"): "Eve"},
            True,
            id="on-the-ray-and-only-a-face-group",  # an unnamed group is no one in particular
        ),
    ],
)
def test_a_foot_on_an_edge_among_several_people_is_put_where_the_one_on_its_ray_was(
    tmp_path, vision, other, names, settled
):
    # living-room saw two people that second; neither is cam0's partner (both are within RIVAL_M), so only
    # cam0's ray (its foot's column is right, its depth isn't) can say which one cam0 saw behind the occluder
    two_cameras(tmp_path, {"cam0": COUCH, "living": FLOOR_SEEN})
    for (cam, track_id), name in names.items():
        if not name.startswith("Face "):
            registry.at(tmp_path).ensure(name, "test")
        named_on(vision, cam, track_id, name)
    seen(vision, "cam0:7", range(10), v=0.95)
    seen(vision, "living:3", range(10), u=ON_RAY[0], v=ON_RAY[1])
    seen(vision, "living:4", range(10), u=other[0], v=other[1])
    cam0 = on_path_of("cam0:7", P.derive(tmp_path, T - 1, T + 20))
    want = (2 - footprints.FOOT_M) if settled else (2 - footprints.FOOT_M) / 0.9
    assert len(cam0) == 10
    assert all(p.y == pytest.approx(want, abs=0.002) and p.uncertain is not settled for p in cam0)
    assert {p.path for p in cam0} == ({"cam0:7+living:3"} if settled else {"cam0:7"})


def behind_the_couch(tmp_path, vision, cam0_seconds, living_seconds, names):
    """cam0:7's feet on a 1.2 m occluder's top for `cam0_seconds`; living-room sees two people on the floor for
    `living_seconds`, living:3 on cam0:7's ray and living:4 off it; `names` as the ray test's. All rows read."""
    two_cameras(tmp_path, {"cam0": COUCH, "living": FLOOR_SEEN})
    for (cam, track_id), name in names.items():
        if not name.startswith("Face "):
            registry.at(tmp_path).ensure(name, "test")
        named_on(vision, cam, track_id, name)
    seen(vision, "cam0:7", cam0_seconds, v=0.95)
    seen(vision, "living:3", living_seconds, u=ON_RAY[0], v=ON_RAY[1])
    seen(vision, "living:4", living_seconds, u=NEARBY, v=1.0)
    return P.derive(tmp_path, T - 1, T + 30)


def test_a_track_whose_feet_went_to_another_cameras_track_is_drawn_on_that_path_as_that_person(tmp_path, vision):
    # Every second living-room saw her, cam0's foot was put on living:3's point, so cam0:7 is Alice drawn twice:
    # one path, Alice's, also over the two seconds after living-room lost her
    got = behind_the_couch(tmp_path, vision, range(12), range(10), {("living", "3"): "Alice", ("living", "4"): "Dana"})
    alice = f"person:{registry.at(tmp_path).person_id('Alice')}"
    cam0 = on_path_of("cam0:7", got)
    assert {(p.path, p.entity, p.name) for p in cam0} == {("cam0:7+living:3", alice, "Alice")}
    assert [p.t - T for p in cam0 if p.track is None] == list(range(10))  # one fused point a second
    assert [(p.t - T, p.uncertain) for p in cam0 if p.track is not None] == [(10, True), (11, True)]


@pytest.mark.parametrize(
    ("cam0_seconds", "living_seconds", "joins"),
    [
        pytest.param(range(3), range(3), True, id="a-short-track-all-put-on-another"),
        pytest.param(range(20), range(3), False, id="a-few-of-a-long-tracks-rows"),
        pytest.param(range(8), range(P.CO_MIN_S), True, id="enough-rows-though-not-most-of-the-track"),
    ],
)
def test_a_track_joins_the_one_its_feet_went_to_only_when_enough_of_them_did(
    tmp_path, vision, cam0_seconds, living_seconds, joins
):
    got = behind_the_couch(tmp_path, vision, cam0_seconds, living_seconds, {})
    assert {p.path for p in on_path_of("cam0:7", got)} == ({"cam0:7+living:3"} if joins else {"cam0:7"})


def test_a_track_the_tracker_started_a_second_early_still_joins_the_path_its_feet_went_to(tmp_path, vision):
    # cam0 lost Alice behind the couch and started cam0:8 a second before it let go of cam0:6 (both at second 5): two
    # tracks of one camera at once, but only for DEFER_OVERLAP_S or less, so both are drawn on living:3's path
    two_cameras(tmp_path, {"cam0": COUCH, "living": FLOOR_SEEN})
    registry.at(tmp_path).ensure("Alice", "test")
    named_on(vision, "living", "3", "Alice")
    seen(vision, "cam0:6", range(6), v=0.95)
    seen(vision, "cam0:8", range(5, 12), v=0.95)
    seen(vision, "living:3", range(12), u=ON_RAY[0], v=ON_RAY[1])
    seen(vision, "living:4", range(12), u=NEARBY, v=1.0)
    got = P.derive(tmp_path, T - 1, T + 30)
    assert {p.path for p in got if p.track in ("cam0:6", "cam0:8", None)} == {"cam0:6+cam0:8+living:3"}


def test_a_track_whose_clear_feet_were_far_from_the_one_its_hidden_feet_went_to_stays_its_own_path(tmp_path, vision):
    # cam0 sees a couch on the right half of its frame and the floor on the left. cam0:7's feet behind the couch were
    # put on living:3's point, but when cam0 saw them on the floor, 2.07 m from living:3: not one person
    cameras = {
        "cam0": replace(LEVEL, background=places.Background(2, 1, (FLOOR_SEEN, COUCH))),
        "living": replace(LEVEL, background=places.Background(1, 1, (FLOOR_SEEN,))),
    }
    places.revise(
        tmp_path,
        None,
        producer="test",
        cameras=cameras,
        floor=FLOOR,
        map_bytes=b"x",
        map_format="ply",
        plan=PLAN,
        plan_transform=TO_PLAN,
        now=T - 100,
    )
    seen(vision, "cam0:7", range(10), v=0.95)
    seen(vision, "cam0:7", range(10, 15), u=0.2)
    seen(vision, "living:3", range(15), u=ON_RAY[0], v=ON_RAY[1])
    seen(vision, "living:4", range(15), u=NEARBY, v=1.0)
    got = P.derive(tmp_path, T - 1, T + 30)
    assert {p.path for p in on_path_of("cam0:7", got)} == {"cam0:7"}
    assert len([p for p in got if p.track == "cam0:7" and not p.uncertain and p.x == pytest.approx(1.0)]) == 10


def test_a_face_group_the_owner_said_isnt_someone_is_never_put_on_their_point(tmp_path, vision):
    # Unmarked, cam0's group would be put on Dana's point, on its ray (the face-group case of the ray test)
    for name in ("Dana", "Eve"):
        registry.at(tmp_path).ensure(name, "test")
        face_nots.mark(tmp_path, name, [3])
    names = {("cam0", "7"): "Face 3", ("living", "3"): "Dana", ("living", "4"): "Eve"}
    cam0 = on_path_of("cam0:7", behind_the_couch(tmp_path, vision, range(10), range(10), names))
    assert {(p.path, p.entity) for p in cam0} == {("cam0:7", "face:3")}
    assert all(p.uncertain for p in cam0)


def test_two_feet_on_an_edge_on_one_ray_that_pick_one_person_both_stay_uncertain(tmp_path, vision):
    two_cameras(tmp_path, {"cam0": COUCH, "living": FLOOR_SEEN})
    seen(vision, "cam0:7", range(10), v=0.95)
    seen(vision, "cam0:8", range(10), v=0.96)  # someone else behind the occluder, just in front on the same ray
    seen(vision, "living:3", range(10), u=ON_RAY[0], v=ON_RAY[1])
    seen(vision, "living:4", range(10), u=NEARBY, v=1.0)
    got = [p for p in P.derive(tmp_path, T - 1, T + 20) if p.camera == "cam0"]
    assert len(got) == 20
    assert all(p.uncertain for p in got)


def test_feet_on_an_edge_are_judged_in_a_plan_frame_from_before_backgrounds_were_kept(tmp_path, vision):
    # The owner corrected the plan of a revision with no backgrounds; the newest has them. Drawn in the corrected
    # plan's frame, cam0's foot behind the occluder is still put where living-room saw her, and is no longer uncertain
    first = two_cameras(tmp_path)
    poses = {"cam0": COUCH, "living": FLOOR_SEEN}
    places.revise(
        tmp_path,
        first.place,
        producer="test",
        cameras={c: replace(LEVEL, background=places.Background(1, 1, (h,))) for c, h in poses.items()},
        floor=FLOOR,
        plan=PLAN,
        plan_transform=TO_PLAN,
        now=T - 50,
    )
    seen(vision, "cam0:7", range(10), v=0.95)
    seen(vision, "living:3", range(10), u=ON_RAY[0], v=ON_RAY[1])
    seen(vision, "living:4", range(10), u=NEARBY, v=1.0)
    got = P.derive(tmp_path, T - 1, T + 20, place=first.place, frame_revision=first.id)
    cam0 = on_path_of("cam0:7", got)
    assert len(cam0) == 10
    assert all(p.y == pytest.approx(2 - footprints.FOOT_M, abs=0.002) and not p.uncertain for p in cam0)


def test_two_cameras_tracks_of_one_unnamed_person_are_one_path(tmp_path, vision):
    # nobody's face is seen; each camera's track was the only one the other camera had in view
    two_cameras(tmp_path)
    seen(vision, "cam0:7", range(10))
    seen(vision, "living:3", range(10))
    got = P.derive(tmp_path, T - 1, T + 20)
    assert {(p.path, p.entity) for p in got} == {("cam0:7+living:3", None)}
    assert len([p for p in got if p.track is None]) == 10  # fused, a point a second


def cam0_registered(tmp_path, place, heights, now):
    """A revision of `place` (None for a new one) where cam0 sees `heights` (a 20 by 20 background, fractions of its
    2 m) and living is registered with none."""
    return places.revise(
        tmp_path,
        place,
        producer="test",
        cameras={"cam0": replace(LEVEL, background=places.Background(20, 20, tuple(heights))), "living": LEVEL},
        floor=FLOOR,
        map_bytes=b"x",
        map_format="ply",
        plan=PLAN,
        plan_transform=TO_PLAN,
        now=now,
    )


# Tonight's registration masked someone sitting at cell 390 (u 0.5 to 0.55, v 0.95 to 1.0), so it doesn't know
# what's there. At v 0.999 cam0's ray falls about a metre a metre: a 1.2 m couch (COUCH) hides 1.13 m of
# floor there, over EDGE_M, and a 1 m piece of furniture (FURNITURE_M) 0.93 m, under it. At v 0.95 (0.9 a metre) it
# hides 1.03 m.
TONIGHT = [COUCH] * 390 + [None] + [COUCH] * 9


@pytest.mark.parametrize(
    ("earlier", "v", "uncertain"),
    [
        pytest.param([COUCH] * 400, 0.999, True, id="an-earlier-night-saw-the-couch-there"),
        pytest.param(
            [FLOOR_SEEN] * 390 + [COUCH] + [FLOOR_SEEN] * 9, 0.999, False, id="an-earlier-view-of-another-aim-isnt-used"
        ),
        pytest.param(None, 0.95, True, id="unseen-and-furniture-could-hide-enough"),
        pytest.param(None, 0.999, False, id="unseen-and-not-even-furniture-could"),
    ],
)
def test_a_foot_where_tonights_registration_saw_nobody_behind_is_not_taken_as_clear(
    tmp_path, vision, earlier, v, uncertain
):
    first = cam0_registered(tmp_path, None, earlier, T - 100) if earlier else None
    cam0_registered(tmp_path, first.place if first else None, TONIGHT, T - 50)
    seen(vision, "cam0:7", range(3), v=v)  # nobody else in view to defer to
    got = P.derive(tmp_path, T - 1, T + 20)
    assert len(got) == 3
    assert all(p.uncertain is uncertain for p in got)


def test_an_owner_plan_drawn_in_a_revision_with_no_plan_file_names_rooms_in_its_frame(tmp_path, vision):
    # An imported plan drawn in the floor's own frame of a map with no machine plan: plan (x, y) = map (x, z)
    # for this y-down floor, as TO_PLAN has it. It holds there and in a later revision of the same map, not in a new map.
    place = home(tmp_path, plan=False)
    drawn = places.latest(tmp_path, place)
    assert drawn is not None and drawn.plan is None
    places.correct_plan(tmp_path, place, OWNER_PLAN, now=T - 50, base=drawn.id)
    foot(vision, T)  # plan (1, 1.93): in the Lounge
    assert [p.room for p in P.derive(tmp_path, T - 1, T + 5, place, frame_revision=drawn.id)] == ["Lounge"]
    places.revise(tmp_path, place, producer="test", sources=["v2"], now=T - 40)  # the same map, carried over
    assert [p.room for p in P.derive(tmp_path, T - 1, T + 5)] == ["Lounge"]
    places.revise(tmp_path, place, producer="test", map_bytes=b"y", map_format="ply", floor=FLOOR, now=T - 30)
    assert [p.room for p in P.derive(tmp_path, T - 1, T + 5)] == [None], "a new map is a new frame"


# The layer: the placer's writes and the read. Its records are in the real segment store under tmp_path; the raw
# footprints the placer places are still the in-memory `vision` log's.


def record(t, place, *, x=1.0, y=1.0, source=P.RIG_POSE, entity="rig", path=None, track=None, revision=None, end=None):
    return P.Position(
        id=str(S.new_id()), t=t, end=t if end is None else end, entity=entity, name=None, track=track, source=source,
        confidence=0.8, place=place, room="Living", room_id="r1", x=x, y=y, camera=None, u=None, v=None,
        revision=revision, path=path,
    )  # fmt: skip


def test_with_no_camera_posed_in_any_map_the_placer_places_nothing_and_says_so(tmp_path, vision, caplog):
    # A place the owner's plan keeps listed, with no map and no camera: a footprint has nothing to be placed by.
    places.revise(tmp_path, None, producer="test", now=T - 100)
    foot(vision, T)
    with caplog.at_level("INFO"):
        did = P.advance(tmp_path, T + 60, chunk_s=P.BACKFILL_S)
    assert did["records"] == 0
    assert "no camera has a pose in any place's map" in caplog.text


def test_the_placer_writes_a_footprint_placed_and_a_read_gives_it_back(tmp_path, vision):
    place = home(tmp_path)
    foot(vision, T)
    assert P.advance(tmp_path, T + 60, chunk_s=P.BACKFILL_S)["records"] == 1
    (p,) = P.positions(tmp_path, T - 1, T + 5, place=place)
    assert (p.t, p.x, p.y, p.room, p.room_id) == (T, 1.0, 1.93, "Living", "r1")
    assert (p.source, p.track, p.path, p.camera, p.entity, p.in_frame) == (
        P.FIXED_CAMERA, "cam0:7", "cam0:7", "cam0", None, True,
    )  # fmt: skip
    rev = places.latest(tmp_path, place)
    assert rev is not None and p.revision == rev.id


def test_a_read_never_places_raw_footprints(tmp_path, vision, monkeypatch):
    place = home(tmp_path)
    foot(vision, T)
    P.advance(tmp_path, T + 60, chunk_s=P.BACKFILL_S)
    foot(vision, T + 1)  # recorded since: not in the layer until the placer's next run
    monkeypatch.setattr(P, "derive", lambda *a, **k: pytest.fail("a read placed raw samples"))
    monkeypatch.setattr(places, "project", lambda *a, **k: pytest.fail("a read projected a footprint"))
    assert [p.t for p in P.positions(tmp_path, T - 1, T + 5, place=place)] == [T]


def test_the_placer_writes_a_zone_visit_as_a_room_only_record_over_its_span(tmp_path, vision):
    place = home(tmp_path)
    zone = dict(subject_kind="person", zone="z1", zone_name="Sofa", track_id=7)
    vision.add(S.ZONE_VISITS, ts_start=T, ts_end=T + 40, cam="cam0", **zone)
    P.advance(tmp_path, T + 60, chunk_s=P.BACKFILL_S)
    (p,) = P.positions(tmp_path, T + 10, T + 20, place=place)  # a window inside the visit still finds it
    assert (p.source, p.t, p.end, p.room, p.x, p.y, p.track) == (P.ZONE, T, T + 40, "Sofa", None, None, "cam0:7")


def test_each_run_places_only_what_came_since_the_last(tmp_path, vision):
    place = home(tmp_path)
    foot(vision, T)
    P.advance(tmp_path, T + 60, chunk_s=P.BACKFILL_S)
    foot(vision, T + 100)
    assert P.advance(tmp_path, T + 160, chunk_s=P.BACKFILL_S)["records"] == 1
    assert [p.t for p in P.positions(tmp_path, T - 1, T + 200, place=place)] == [T, T + 100]


def failing_save(monkeypatch, nth):
    """`P._save_marks` failing on its `nth` call from now, as a full disk would; returns a function that mends it."""
    save, calls = P._save_marks, []

    def flaky(state, marks):
        calls.append(marks)
        if len(calls) == nth:
            raise OSError("no space left on device")
        save(state, marks)

    monkeypatch.setattr(P, "_save_marks", flaky)
    return lambda: monkeypatch.setattr(P, "_save_marks", save)


def test_a_piece_that_failed_after_its_rows_were_written_isnt_kept_twice(tmp_path, vision, monkeypatch):
    place = home(tmp_path)
    foot(vision, T - 50)
    P.advance(tmp_path, T - 10, chunk_s=P.BACKFILL_S)
    foot(vision, T)
    mend = failing_save(monkeypatch, 2)  # the first save is of the registration checked; the second moves the mark
    with pytest.raises(OSError, match="no space"):
        P.advance(tmp_path, T + 60, chunk_s=P.BACKFILL_S)
    mend()
    P.advance(tmp_path, T + 60, chunk_s=P.BACKFILL_S)
    assert [p.t for p in P.positions(tmp_path, T - 100, T + 200, place=place)] == [T - 50, T]


def test_a_track_crossing_a_pieces_end_is_written_once_on_each_side(tmp_path, vision):
    place = home(tmp_path)
    cut = (T // 3600 + 1) * 3600  # the placer cuts on the clock
    for t in (cut - 2, cut - 1, cut, cut + 1):
        foot(vision, t)
    assert P.advance(tmp_path, cut + 60, chunk_s=3600.0)["records"] == 4
    assert [p.t for p in P.positions(tmp_path, cut - 10, cut + 10, place=place)] == [cut - 2, cut - 1, cut, cut + 1]


def test_a_long_read_keeps_one_record_per_path_per_step_and_each_paths_last(tmp_path):
    rows = [record(T + i, "home", path="a") for i in range(25)] + [record(T + i, "home", path="b") for i in range(3)]
    P.write(tmp_path, rows)
    got = P.positions(tmp_path, T, T + 100, place="home", every=10)
    assert [(p.path, p.t - T) for p in got if p.path == "a"] == [("a", 0), ("a", 10), ("a", 20), ("a", 24)]
    assert [p.t - T for p in got if p.path == "b"] == [0, 2]


def test_a_week_long_read_scans_a_tenth_of_what_was_written(tmp_path, monkeypatch):
    P.write(tmp_path, [record(T + i, "home", path="a") for i in range(100)])
    scanned = []
    scan = P._scan

    def counted(*a, **k):
        found = scan(*a, **k)
        scanned.append(sum(len(got["ts_ns"]) for got in found))
        return found

    monkeypatch.setattr(P, "_scan", counted)
    week = P.positions(tmp_path, T, T + 100, place="home", every=15)
    day = P.positions(tmp_path, T, T + 100, place="home", every=2)
    assert scanned == [11, 100]  # one in ten and the path's last, against every record
    assert [p.t - T for p in week] == [0, 20, 30, 50, 60, 80, 90, 99]  # the 10 s records in each 15 s step
    assert len(day) == 51


def test_thinning_keeps_every_zone_visit(tmp_path):
    visits = [record(T + i, "home", source=P.ZONE, x=None, y=None, track="cam0:7", end=T + i + 1) for i in range(5)]
    P.write(tmp_path, visits)
    assert len(P.positions(tmp_path, T, T + 100, place="home", every=60)) == 5


def test_a_read_keeps_to_its_place_and_entity(tmp_path):
    P.write(tmp_path, [record(T, "home"), record(T + 1, "home", entity="person:1"), record(T + 2, "cabin")])
    assert [p.t - T for p in P.positions(tmp_path, T, T + 10, place="home")] == [0, 1]
    assert [p.t - T for p in P.positions(tmp_path, T, T + 10, place="home", entity="person:1")] == [1]
    assert [p.place for p in P.positions(tmp_path, T, T + 10)] == ["home", "home", "cabin"]


def test_a_read_stops_between_days_once_nobody_is_waiting(tmp_path, monkeypatch):
    P.write(tmp_path, [record(T + day * 86400, "home") for day in range(3)])
    scanned = []
    scan = P._scan
    monkeypatch.setattr(P, "_scan", lambda state, lo, hi, *a, **k: scanned.append(lo) or scan(state, lo, hi, *a, **k))
    waiting = iter([False, True])  # the client hangs up during the first day's scan
    with pytest.raises(P.Cancelled):
        P.positions(tmp_path, T, T + 3 * 86400, place="home", cancelled=lambda: next(waiting))
    assert scanned == [T]


def test_a_record_placed_on_another_plan_of_the_same_map_is_moved_onto_the_plan_asked_for(tmp_path):
    # The newest revision draws its plan 2 m along x from the first's, on the same map: (3, 1.93) there is (1, 1.93)
    # on the first's plan.
    place, first, newest = two_nights(tmp_path)
    P.write(tmp_path, [record(T + 2, place, x=3.0, y=1.93, revision=newest.id)])
    (p,) = P.positions(tmp_path, T, T + 5, place, frame_revision=first.id)
    assert (p.x, p.y) == pytest.approx((1.0, 1.93), abs=1e-3)
    assert (p.revision, p.in_frame) == (first.id, True)


def test_a_record_placed_on_another_map_keeps_its_own_frame_and_is_flagged(tmp_path):
    # The owner's plan is drawn on the first revision. A later map of the place stands in a frame of its own, and
    # nothing relates the two: a record placed there can't be put on the plan.
    place = home(tmp_path)
    first = places.latest(tmp_path, place)
    assert first is not None
    cams = {"cam0": moved(LEVEL, dx=5.0), "porch": moved(LEVEL, dx=5.0, dz=3.0)}
    later = places.revise(
        tmp_path, place, producer="test", cameras=cams, floor=FLOOR, map_bytes=b"y", map_format="ply", now=T
    )
    places.correct_plan(tmp_path, place, OWNER_ROOMS, now=T + 1, base=first.id)
    P.write(tmp_path, [record(T + 2, place, x=6.0, y=4.93, revision=later.id)])
    (p,) = P.positions(tmp_path, T, T + 5, place, frame_revision=first.id)
    assert (p.x, p.y, p.revision, p.in_frame) == (6.0, 4.93, later.id, False)


def test_a_record_from_a_revision_the_place_no_longer_has_is_marked_out_of_frame(tmp_path):
    place = home(tmp_path)
    drawn = places.latest(tmp_path, place)
    assert drawn is not None
    P.write(
        tmp_path, [record(T + 1, place, revision="a-revision-since-removed"), record(T + 2, place, revision=drawn.id)]
    )
    got = P.positions(tmp_path, T, T + 5, place, frame_revision=drawn.id)
    assert [(p.t - T, p.in_frame) for p in got] == [(1, False), (2, True)]


@pytest.mark.parametrize(("place", "frame"), [(None, "any"), ("given", "no-such-revision")])
def test_a_reads_frame_needs_its_place_and_must_exist(tmp_path, place, frame):
    real = home(tmp_path)
    with pytest.raises(ValueError, match="revision"):
        P.positions(tmp_path, T, T + 5, real if place else None, frame_revision=frame)


def test_last_seen_is_each_places_newest_record_within_the_lookbacks(tmp_path):
    now = T + 10 * 86400
    P.write(
        tmp_path,
        [record(now - 60, "home", end=now - 30), record(now - 5 * 86400, "cabin"), record(now - 9 * 86400, "shed")],
    )
    found = P.last_seen(tmp_path, ["home", "cabin", "shed", "never"], now, (3600, 7 * 86400))
    assert found == {"home": now - 30, "cabin": now - 5 * 86400}


def test_a_reset_has_the_placer_place_everything_again_once(tmp_path, vision):
    place = home(tmp_path)
    foot(vision, T)
    P.advance(tmp_path, T + 60, chunk_s=P.BACKFILL_S)
    P.write(tmp_path, [record(T + 1, place)])  # the rig's own record: nothing can make it again
    assert P.reset(tmp_path) == 1
    assert [p.source for p in P.positions(tmp_path, T - 1, T + 5, place=place)] == [P.RIG_POSE]
    P.advance(tmp_path, T + 60, chunk_s=P.BACKFILL_S)
    assert [p.source for p in P.positions(tmp_path, T - 1, T + 5, place=place)] == [P.FIXED_CAMERA, P.RIG_POSE]


def registered_again(tmp_path, place, dx, at):
    """`place`'s camera registered again `dx` metres along x from where LEVEL has it."""
    cam = places.CameraPose(LEVEL.pose[:3] + (LEVEL.pose[3] + dx,) + LEVEL.pose[4:], 144.0, 512, 288, at)
    return places.revise(tmp_path, place, producer="test", cameras={"cam0": cam}, now=at)


def test_a_path_that_spans_a_re_registration_is_placed_again_in_one_frame(tmp_path, vision):
    place = home(tmp_path)
    foot(vision, T, v=0.75)  # the feet's plane 3.86 m ahead of the camera, which stands at x = 1
    P.advance(tmp_path, T + 60, chunk_s=P.BACKFILL_S)
    assert [(p.x, p.y) for p in P.positions(tmp_path, T - 1, T + 200, place=place)] == [(1.0, 3.86)]
    newest = registered_again(tmp_path, place, 1.0, T + 80)  # the camera was a metre further along x all along
    foot(vision, T + 100, v=0.75)
    assert P.advance(tmp_path, T + 160, chunk_s=P.BACKFILL_S)["replaced"] == 1
    got = P.positions(tmp_path, T - 1, T + 200, place=place, frame_revision=newest.id)
    assert [(p.t - T, p.x, p.y, p.revision, p.in_frame) for p in got] == [
        (0, 2.0, 3.86, newest.id, True),
        (100, 2.0, 3.86, newest.id, True),
    ]


@pytest.mark.parametrize("dx", [0.0, 0.1], ids=["carried-over-as-it-was", "a-refit-of-centimetres"])
def test_a_registration_that_puts_people_where_it_did_leaves_the_records_alone(tmp_path, vision, dx):
    place = home(tmp_path)
    foot(vision, T, v=0.75)
    P.advance(tmp_path, T + 60, chunk_s=P.BACKFILL_S)
    registered_again(tmp_path, place, dx, T + 80)
    did = P.advance(tmp_path, T + 160, chunk_s=P.BACKFILL_S)
    assert (did["replaced"], did["records"]) == (0, 0)
    assert [(p.x, p.y) for p in P.positions(tmp_path, T - 1, T + 200, place=place)] == [(1.0, 3.86)]


def test_refits_that_add_up_to_a_move_have_the_place_placed_again(tmp_path, vision):
    place = home(tmp_path)
    foot(vision, T, v=0.75)
    P.advance(tmp_path, T + 60, chunk_s=P.BACKFILL_S)
    registered_again(tmp_path, place, 0.15, T + 80)
    assert P.advance(tmp_path, T + 160, chunk_s=P.BACKFILL_S)["replaced"] == 0
    registered_again(tmp_path, place, 0.3, T + 180)  # 0.15 m from last night, 0.3 m from where its records are
    assert P.advance(tmp_path, T + 260, chunk_s=P.BACKFILL_S)["replaced"] == 1
    assert [(p.x, p.y) for p in P.positions(tmp_path, T - 1, T + 300, place=place)] == [(1.3, 3.86)]


def test_only_the_place_whose_camera_was_registered_again_is_placed_again(tmp_path, vision):
    place = home(tmp_path)
    other = places.revise(
        tmp_path, None, producer="test", cameras={"porch": LEVEL}, floor=FLOOR, map_bytes=b"x", map_format="ply",
        plan=PLAN, plan_transform=TO_PLAN, now=T - 100,
    ).place  # fmt: skip
    foot(vision, T, v=0.75)
    foot(vision, T + 1, v=0.75, cam="porch")
    P.advance(tmp_path, T + 60, chunk_s=P.BACKFILL_S)
    P.write(tmp_path, [record(T + 2, place)])  # the rig's own record there
    registered_again(tmp_path, place, 1.0, T + 80)
    did = P.advance(tmp_path, T + 160, chunk_s=P.BACKFILL_S)
    assert (did["replaced"], did["records"]) == (1, 1)  # the porch's record wasn't placed again
    assert [(p.source, p.x) for p in P.positions(tmp_path, T - 1, T + 200, place=place)] == [
        (P.FIXED_CAMERA, 2.0),
        (P.RIG_POSE, 1.0),
    ]
    assert [(p.t - T, p.x) for p in P.positions(tmp_path, T - 1, T + 200, place=other)] == [(1, 1.0)]


def test_placing_a_place_again_picks_up_where_a_failed_run_stopped(tmp_path, vision, monkeypatch):
    place = home(tmp_path)
    half = P.BACKFILL_S / 2
    cut = (T // half) * half  # the placer cuts on the clock: a footprint either side of a cut, in two pieces
    for t in (cut - 100, cut + 100):
        foot(vision, t, v=0.75)
    P.advance(tmp_path, cut + 200, chunk_s=half)
    registered_again(tmp_path, place, 1.0, cut + 300)
    mend = failing_save(monkeypatch, 3)  # after the place is marked and the registration saved: the first piece's
    with pytest.raises(OSError, match="no space"):
        P.advance(tmp_path, cut + 400, chunk_s=half)
    mend()
    assert P.advance(tmp_path, cut + 400, chunk_s=half)["replaced"] == 0  # it goes on; it doesn't start over
    got = P.positions(tmp_path, cut - 200, cut + 200, place=place)
    assert [(p.t - cut, p.x) for p in got] == [(-100, 2.0), (100, 2.0)]


def test_retention_thins_placed_records_by_path_and_keeps_every_zone_visit():
    step = dict(source=P.FIXED_CAMERA, entity=None, track=None, camera=None, path="cam0:7+porch:2")
    rows = [dict(id=f"s{i}", ts_start=T + i, **step) for i in range(25)]
    rows += [dict(id=f"z{i}", ts_start=T + i, source=P.ZONE, track="cam0:7", entity=None) for i in range(3)]
    kept = [r["id"] for r in retention.thin_positions(rows, every=10)]
    assert kept == ["s0", "z0", "z1", "z2", "s10", "s20", "s24"]
