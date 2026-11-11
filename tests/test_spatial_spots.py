"""A camera's pose from its owner's spot on the plan: the tilt fitted from walking people, what a spot lacks, and
footprints placed through it with no map."""

import math

import numpy as np
import pytest

import plans
from liberational import json_state
from liberational import segment_log as S
from liberational.dashboard import places as view
from liberational.memory import places
from liberational.spatial import footprints, lens, plan_import, spots
from liberational.spatial import positions as P

T = 1_800_000_000.0
SIZE = (512, 288)
FOCAL = 144.0  # pixels: the bottom middle of the frame drops a metre a metre from a level camera
ROOMS = plans.scene(
    'plan "Home" walls 0.2/0.1\nroom r1 "Living" other rect 0,0 4x3\nroom r2 "Hall" other rect 0,3 4x2\n'
)
# The same, with a wide doorway from the living room into the hall and a front door, so a camera at (1, 1) facing
# east sees into the hall and out of the front door at the frame's right edge
OPEN = plans.scene(
    'plan "Home" walls 0.2/0.1\nroom r1 "Living" other rect 0,0 4x3\nroom r2 "Hall" other rect 0,3 4x2\n'
    "cased r1>r2 at 2.2,3 w2.4\ndoor exterior>r1 at 1.8,0 w2.4\n"
)


def walkers(pitch, roll=0.0, above=2.0, n=120, seed=0, across=3.0, focal=FOCAL):
    """Foot and head pixels of people 1.5 to 1.8 m tall walking 2 to 7 m in front of a camera `above` the floor,
    pitched `pitch` degrees down and rolled `roll` clockwise, worked out from the room's side: the world has z up,
    the camera looks along +y. Returns (feet, heads, the people's median height)."""
    rng = np.random.default_rng(seed)
    tall = rng.uniform(1.5, 1.8, n)
    floor = np.column_stack([rng.uniform(-across, across, n), rng.uniform(2.0, 7.0, n)])
    feet = pixels(np.column_stack([floor, np.full(n, footprints.FOOT_M)]), pitch, roll, above, focal)
    heads = pixels(np.column_stack([floor, tall]), pitch, roll, above, focal)
    return feet, heads, float(np.median(tall))


def pixels(points, pitch, roll=0.0, above=2.0, focal=FOCAL):
    """Where (N, 3) points (x right of the camera, y ahead, z up, from the floor under it) are seen in a pinhole
    frame of SIZE at `focal`, from a camera `above` the floor pitched `pitch` degrees down and rolled `roll`."""
    p, r = math.radians(pitch), math.radians(roll)
    forward = np.array([0.0, math.cos(p), -math.sin(p)])
    level_right, level_down = np.array([1.0, 0.0, 0.0]), np.array([0.0, -math.sin(p), -math.cos(p)])
    right = math.cos(r) * level_right + math.sin(r) * level_down
    down = -math.sin(r) * level_right + math.cos(r) * level_down
    rel = np.asarray(points, dtype=np.float64) - np.array([0.0, 0.0, above])
    x, y, z = rel @ right, rel @ down, rel @ forward
    return np.column_stack([0.5 + focal * x / z / SIZE[0], 0.5 + focal * y / z / SIZE[1]])


# A made-up room seen down its length: 8 m wide, its far wall 6.5 m from a spot half a metre inside the near one, with
# walls 2.4 m tall. The camera's world here is the room's side (x right of it, y ahead, z up from the floor under it).
ROOM = plans.scene('plan "Home" walls 0.2/0.1\nroom r1 "Room" other rect -4,-7 8x7\n')
SPOT = plan_import.Camera("cam0", (0.0, -0.5), 2.0, 0.0)
WIDE_SHARE = 0.401  # a wide lens: its focal as a share of the frame's width, about 100 degrees across
TOP = 2.4  # metres: how tall the room's walls are, so a picture of them has a line at the top too
FAR = 6.5  # metres from that spot to the far wall


def wall_lines():
    """((start, end) in world metres) the straight lines the room's walls make in its picture: each wall's floor line,
    its top line, and the corners between walls."""
    lines = [((-4.0, FAR, 0.0), (4.0, FAR, 0.0)), ((-4.0, FAR, TOP), (4.0, FAR, TOP))]
    for x in (-4.0, 4.0):
        lines += [((x, 0.2, 0.0), (x, FAR, 0.0)), ((x, 0.2, TOP), (x, FAR, TOP)), ((x, FAR, 0.0), (x, FAR, TOP))]
    return lines


def room_picture(pitch, roll=0.0, above=2.0, focal=WIDE_SHARE):
    """A picture of the room from a camera `above` the floor, pitched `pitch` degrees down, rolled `roll` clockwise,
    and at `focal` (a share of the frame's width): its walls' lines drawn on a plain background, as that camera's
    frame shows them."""
    import cv2

    img = np.full((SIZE[1], SIZE[0], 3), 235, np.uint8)
    for start, end in wall_lines():
        along = np.linspace(0.0, 1.0, 120)
        points = np.array(start)[None, :] * (1.0 - along)[:, None] + np.array(end)[None, :] * along[:, None]
        seen = pixels(points, pitch, roll, above, focal * SIZE[0])
        kept = seen[(seen[:, 0] > -1.0) & (seen[:, 0] < 2.0) & (seen[:, 1] > -1.0) & (seen[:, 1] < 2.0)]
        drawn = np.column_stack([kept[:, 0] * SIZE[0], kept[:, 1] * SIZE[1]]).round().astype(np.int32)
        if len(drawn) > 1:
            cv2.polylines(img, [drawn], False, (30, 30, 30), 2)
    return img


HELD = lens.Held(FOCAL / SIZE[0], 0.0, "test")


@pytest.mark.parametrize(
    ("pitch", "roll", "across"),
    [
        pytest.param(25.0, 0.0, 3.0, id="pitched"),
        pytest.param(40.0, 8.0, 3.0, id="pitched-and-rolled"),
        pytest.param(10.0, 0.0, 0.3, id="walkers-in-one-strip-of-the-view"),
    ],
)
def test_the_tilt_is_found_from_walking_people_with_no_height_assumed(pitch, roll, across):
    feet, heads, tall = walkers(pitch, roll, across=across)
    spot = plan_import.Camera("cam0", (0.0, 0.0), 2.0, 0.0)
    found = spots.aim("cam0", spot, HELD, SIZE, feet, heads, T)
    assert found is not None
    assert (found.pitch, found.roll) == pytest.approx((pitch, roll), abs=0.5)
    assert found.tall == pytest.approx(tall, abs=0.05)


def test_a_pictures_straight_edges_recover_a_wide_lenses_tilt_roll_and_focal():
    # The room's lines across the whole frame, and walkers in one strip of it who would put the tilt anywhere with a
    # lens whose focal is a quarter off: the edges' tilt and focal are found, and the walkers agree with them there.
    pitch, roll = 18.0, 6.0
    edges = spots.edges_in(room_picture(pitch, roll))
    assert len(edges) >= spots.EDGE_MIN
    feet, heads, tall = walkers(pitch, roll, across=0.6, focal=WIDE_SHARE * SIZE[0])
    assert float(np.ptp(np.percentile(feet[:, 0], [5, 95]))) < spots.ROLL_SPAN  # one strip of the view
    held = lens.Held(WIDE_SHARE * 0.75, 0.0, "a plumb line's")
    shape = places.CameraPose(tuple(float(x) for x in np.eye(4).ravel()), held.focal * SIZE[0], *SIZE)
    walked = spots.up_from_people(spots.rays(shape, feet), spots.rays(shape, heads), roll=False)
    assert walked is not None
    assert abs(spots._pitch_roll(walked)[0] - pitch) > 3.0  # the walkers alone, at that focal, are the wrong tilt

    found = spots.aim(
        "cam0", SPOT, held, SIZE, feet, heads, T, edges=edges, room=spots.room(ROOM, SPOT), mirrored=False
    )

    assert found is not None
    assert (found.pitch, found.roll) == pytest.approx((pitch, roll), abs=1.5)
    assert found.focal == pytest.approx(WIDE_SHARE * SIZE[0], rel=0.08)
    assert found.confirmed
    assert found.edges >= spots.EDGE_MIN
    assert found.tall == pytest.approx(tall, abs=0.08)
    assert "confirmed" in spots.edge_note(found)


def test_the_plans_headings_and_the_walls_its_spot_should_see():
    assert spots.headings(ROOM) == (90.0, 0.0)  # the far wall across the view, the side walls running away
    assert spots.headings("not a scene") == ()
    seen = spots.room(ROOM, SPOT)
    assert len(seen.floor) > 10
    # every floor point is in front of the spot (which faces -y on the page) and inside the room, not behind it
    assert (seen.floor[:, 1] < SPOT.at[1]).all()
    assert (np.abs(seen.floor[:, 0]) <= 4.0).all()
    assert seen.eye == (0.0, -0.5, 2.0)
    assert not len(spots.room(ROOM, plan_import.Camera("cam0", (0.0, -0.5), None, 0.0)).floor)


def test_too_few_straight_edges_leave_the_walkers_tilt_and_say_so():
    # A picture with one line across it says nothing of a tilt: the walkers' own fit stands, and the note says why.
    picture = np.full((SIZE[1], SIZE[0], 3), 235, np.uint8)
    picture[140:143, 60:460] = 30
    edges = spots.edges_in(picture)
    feet, heads, _ = walkers(24.0, 4.0)
    found = spots.aim(
        "cam0",
        SPOT,
        lens.Held(WIDE_SHARE, 0.0, "test"),
        SIZE,
        feet,
        heads,
        T,
        edges=edges,
        room=spots.room(ROOM, SPOT),
        mirrored=False,
    )
    assert found is not None
    assert (found.pitch, found.roll) == pytest.approx((24.0, 4.0), abs=1.0)
    assert not found.confirmed
    assert found.focal is None
    assert "too few" in spots.edge_note(found) or "no straight edges" in spots.edge_note(found)


def test_a_fitted_focal_is_the_one_the_poses_rays_are_cast_with(tmp_path, monkeypatch, pinhole, vision):
    # The picture's edges fit the focal, so the aim keeps it and the pose reads frames through it.
    pitch, roll = 18.0, 6.0
    picture = room_picture(pitch, roll)
    vision.frames[("cam0", T)] = picture
    feet, heads, _ = walkers(pitch, roll, across=0.6, focal=WIDE_SHARE * SIZE[0])
    taught(tmp_path, "x-camera cam0 at 0,-0.5 height 2 yaw 0", rooms=ROOM, held=False)
    lens.hold(tmp_path, "cam0", lens.Held(WIDE_SHARE * 0.75, 0.0, "a plumb line's"), now=T - 50)
    monkeypatch.setattr(spots, "people", lambda state, camera, now: (feet, heads))
    monkeypatch.setattr(spots, "frame_size", lambda camera, t: SIZE)

    fitted = spots.fit(tmp_path, T)["cam0"]

    assert fitted.focal == pytest.approx(WIDE_SHARE * SIZE[0], rel=0.08)
    spot = spots.read(tmp_path)["cam0"]
    assert spot.missing == ()
    assert spot.pose is not None
    assert spot.pose.focal == pytest.approx(WIDE_SHARE * SIZE[0], rel=0.08)


def test_an_aim_kept_before_the_pictures_edges_were_read_still_holds(tmp_path):
    folder = places.root(tmp_path)
    folder.mkdir(parents=True, exist_ok=True)
    up = spots.up_at(20.0, 2.0)
    json_state.write(
        folder / spots.AIMS,
        {
            "cam0": {
                "up": [float(v) for v in up],
                "width": SIZE[0],
                "height": SIZE[1],
                "people": 120,
                "tall": 1.65,
                "at": T,
                "basis": "2.0|None|0.28125/0.00000",
            }
        },
        indent=1,
        sort_keys=True,
    )
    kept = spots.aims(tmp_path)["cam0"]
    assert (kept.edges, kept.edge_pitch, kept.edge_roll, kept.focal, kept.confirmed) == (0, None, None, None, False)
    assert (kept.pitch, kept.roll) == pytest.approx((20.0, 2.0), abs=0.01)


def test_the_dry_run_says_whether_a_pictures_edges_confirmed_a_cameras_tilt(tmp_path, capsys, monkeypatch):
    up = spots.up_at(18.0, 0.0)
    aim = spots.Aim(
        (float(up[0]), float(up[1]), float(up[2])),
        SIZE[0],
        SIZE[1],
        120,
        1.65,
        T,
        "basis",
        edges=40,
        edge_pitch=18.2,
        edge_roll=0.3,
        confirmed=True,
    )
    row = {
        "camera": "cam0",
        "place": None,
        "by": "spot",
        "footprints": 0,
        "placed": 0,
        "in_a_room": 0,
        "people_placed": 0,
        "behind_wall": None,
        "missing": [],
        "aim": aim,
    }
    monkeypatch.setattr(P, "dry_run", lambda state, cams, now, hours: [row])
    P._print_dry_run(tmp_path, ["cam0"], 24.0)
    assert "40 straight edges confirmed that tilt" in capsys.readouterr().out


def test_too_few_walkers_fit_no_tilt():
    feet, heads, _ = walkers(25.0, n=spots.AIM_MIN - 1)
    assert spots.aim("cam0", plan_import.Camera("cam0", (0.0, 0.0), 2.0, 0.0), HELD, SIZE, feet, heads, T) is None


def pose_rows(people):
    """pose.track columns for `people`: (track, second, foot (u, v), head (u, v)) with every joint placed on the
    line from head to foot, the hip halfway."""
    rows = {"ts_ns": [], "track_id": [], "box": [], "joints": []}
    for track, t, foot, head in people:
        joints = [-1, -1, 0] * 26
        hip = ((foot[0] + head[0]) / 2, (foot[1] + head[1]) / 2)
        for j, (u, v) in ((24, foot), (25, foot), (spots.HEAD, head), (spots.HIP, hip)):
            joints[3 * j : 3 * j + 3] = [round(u * footprints.SCALE), round(v * footprints.SCALE), 9000]
        rows["ts_ns"].append(round((T + t) * footprints.NS))
        rows["track_id"].append(track)
        rows["box"].append([head[0] - 0.05, head[1], foot[0] + 0.05, foot[1], 0.9])
        rows["joints"].append(joints)
    return rows


def test_only_people_whose_feet_moved_since_the_last_second_count_as_walking():
    still = [(1, t, (0.3, 0.9), (0.3, 0.5)) for t in range(5)]
    strolling = [(2, t, (0.5 + 0.03 * t, 0.9), (0.5 + 0.03 * t, 0.5)) for t in range(5)]
    feet, heads = spots.walking(pose_rows(still + strolling))
    # the stroller's first row has nothing before it to have moved from
    assert feet[:, 0].tolist() == pytest.approx([0.53, 0.56, 0.59, 0.62])
    assert heads[:, 1].tolist() == pytest.approx([0.5] * 4)


def test_a_bent_person_isnt_taken_for_upright():
    # the hip is placed off the line from head to feet by more than STRAIGHT_DEG: someone leaning or seated
    rows = pose_rows([(3, t, (0.5 + 0.03 * t, 0.9), (0.5 + 0.03 * t, 0.5)) for t in range(4)])
    for joints in rows["joints"]:
        joints[3 * spots.HIP] += round(0.2 * footprints.SCALE)
    feet, _ = spots.walking(rows)
    assert len(feet) == 0


def test_spots_are_read_from_a_plan_with_nothing_else_in_it():
    text = plans.scene("x-camera hall at 1,2 height 2.1 yaw 90 fov 100\nx-camera porch at 3,4\n")
    assert plan_import.spots(text) == (
        plan_import.Camera("hall", (1.0, 2.0), 2.1, 90.0, 100.0),
        plan_import.Camera("porch", (3.0, 4.0)),
    )


@pytest.fixture
def pinhole(monkeypatch):
    monkeypatch.setattr(lens, "as_registered", lambda camera: None)  # no configured lens undoes the frames


def taught(tmp_path, spot, *, view=None, held=True, rooms=ROOMS):
    """A place kept across a reset: no map, and the owner's plan with one camera spot."""
    place = places.revise(tmp_path, None, producer="places reset v1", now=T - 100).place
    places.correct_plan(tmp_path, place, rooms + spot + "\n", now=T - 50, view=view)
    if held:
        lens.hold(tmp_path, "cam0", HELD, now=T - 50)
    return place


def fit(tmp_path, monkeypatch, pitch=0.0, above=2.0):
    """Fit the cameras' aims from walkers seen by a camera `above` the floor pitched `pitch` down."""
    feet, heads, _ = walkers(pitch, above=above)
    monkeypatch.setattr(spots, "people", lambda state, camera, now: (feet, heads))
    return spots.fit(tmp_path, T, sizes=lambda camera, t: SIZE)


def foot(vision, t, u=0.5, v=1.0, cam="cam0", track="7"):
    row = dict(source=P.FIXED_CAMERA, confidence=0.9, camera=cam, u=u, v=v, track=f"{cam}:{track}", entity=None)
    row |= dict(place=None, room=None, x=None, y=None, revision=None)
    return vision.add(S.POSITIONS, ts_start=t, cam=cam, **row)


def test_a_footprint_is_placed_on_the_plan_through_the_owners_spot_with_no_map(tmp_path, vision, monkeypatch, pinhole):
    # A level camera 2 m up at (1, 1) facing east (yaw 90): the bottom middle of its frame drops a metre a metre, so
    # heels (0.07 m up) are 1.93 m ahead of it, at (2.93, 1), in the living room.
    place = taught(tmp_path, "x-camera cam0 at 1,1 height 2 yaw 90")
    fit(tmp_path, monkeypatch)
    foot(vision, T)
    held = places.get(tmp_path, place)
    assert held is not None
    for frame in (None, held.plan_revision):
        (p,) = P.derive(tmp_path, T - 1, T + 5, place=place, frame_revision=frame)
        assert (p.x, p.y) == pytest.approx((2.93, 1.0), abs=0.03)
        assert (p.place, p.room, p.revision, p.in_frame) == (place, "Living", held.plan_revision, True)


@pytest.mark.parametrize(
    ("view", "want"),
    [
        # On the page as drawn (y grows south), a camera facing east has south on its right: the frame's right edge
        # is 256 / 144 m to the right a metre ahead, so 3.43 m south of the heels, in the hall.
        pytest.param(None, (2.93, 4.43, "Hall"), id="the-page-as-drawn"),
        # In a plan frame that is the page's mirror image, the same facing has its right hand the other way.
        pytest.param(
            {"plan": OPEN, "turn": 0.0, "shift": [0.0, 0.0], "mirror": True}, (2.93, -2.43, None), id="mirrored"
        ),
    ],
)
def test_the_cameras_right_hand_follows_the_plans_handedness(tmp_path, vision, monkeypatch, pinhole, view, want):
    place = taught(tmp_path, "x-camera cam0 at 1,1 height 2 yaw 90", view=view, rooms=OPEN)
    fit(tmp_path, monkeypatch)
    foot(vision, T, u=1.0)
    (p,) = P.derive(tmp_path, T - 1, T + 5, place=place)
    assert (p.x, p.y) == pytest.approx(want[:2], abs=0.05)
    assert p.room == want[2]


@pytest.mark.parametrize(
    ("spot", "held", "walkers_above", "missing"),
    [
        pytest.param("x-camera cam0 at 1,1 height 2 yaw 90", True, 2.0, (), id="complete"),
        pytest.param("x-camera cam0 at 1,1 height 2", True, 2.0, ("facing",), id="no-yaw"),
        pytest.param("x-camera cam0 at 1,1 yaw 90", True, 2.0, ("height",), id="no-height"),
        pytest.param("x-camera cam0 at 1,1 height 2 yaw 90", False, 2.0, ("lens",), id="no-held-lens-and-no-fov"),
        pytest.param("x-camera cam0 at 1,1 height 2 yaw 90 fov 121.3", False, 2.0, (), id="fov-stands-in-for-a-lens"),
        # walkers seen from 4 m up, read from a spot that says 2 m, come out half their height
        pytest.param("x-camera cam0 at 1,1 height 2 yaw 90", True, 4.0, ("check",), id="people-the-wrong-height"),
    ],
)
def test_a_spot_says_what_it_lacks_for_a_pose(tmp_path, monkeypatch, pinhole, spot, held, walkers_above, missing):
    taught(tmp_path, spot, held=held)
    fit(tmp_path, monkeypatch, above=walkers_above)
    found = spots.read(tmp_path)["cam0"]
    assert found.missing == missing
    assert (found.pose is None) == bool(missing)


def test_a_camera_nobody_walked_past_waits_for_its_tilt(tmp_path, monkeypatch, pinhole):
    taught(tmp_path, "x-camera cam0 at 1,1 height 2 yaw 90")
    monkeypatch.setattr(spots, "people", lambda state, camera, now: (np.zeros((0, 2)), np.zeros((0, 2))))
    assert spots.fit(tmp_path, T, sizes=lambda camera, t: SIZE) == {}
    assert spots.read(tmp_path)["cam0"].missing == ("tilt",)


def test_a_spot_measured_again_needs_its_tilt_fitted_again(tmp_path, monkeypatch, pinhole):
    place = taught(tmp_path, "x-camera cam0 at 1,1 height 2 yaw 90")
    fit(tmp_path, monkeypatch)
    places.correct_plan(tmp_path, place, ROOMS + "x-camera cam0 at 1,1 height 2.4 yaw 90\n", now=T + 10)
    assert spots.read(tmp_path)["cam0"].missing == ("tilt",)


def test_an_unplaceable_camera_leaves_the_others_of_its_place_placed(tmp_path, vision, monkeypatch, pinhole):
    place = taught(tmp_path, "x-camera cam0 at 1,1 height 2 yaw 90\nx-camera porch at 3,1 height 2")
    lens.hold(tmp_path, "porch", HELD, now=T - 50)
    fit(tmp_path, monkeypatch)
    foot(vision, T)
    foot(vision, T + 1, cam="porch")
    got = {p.camera: (p.place, p.x) for p in P.derive(tmp_path, T - 1, T + 5, place=place)}
    assert got["cam0"] == (place, pytest.approx(2.93, abs=0.03))
    assert got["porch"] == (place, None)
    assert P.placing(tmp_path, place) == [
        {"camera": "cam0", "by": "spot", "missing": []},
        {"camera": "porch", "by": "spot", "missing": ["facing"]},
    ]


def test_the_placer_writes_spot_placed_records_and_the_page_reads_them_with_each_cameras_note(
    tmp_path, vision, monkeypatch, pinhole
):
    place = taught(tmp_path, "x-camera cam0 at 1,1 height 2 yaw 90\nx-camera porch at 3,1 height 2")
    lens.hold(tmp_path, "porch", HELD, now=T - 50)
    fit(tmp_path, monkeypatch)
    foot(vision, T)
    assert P.advance(tmp_path, T + 60, chunk_s=P.BACKFILL_S)["records"] == 1
    query = {"place": place, "start": str(T - 60), "end": str(T + 60)}
    code, reply = view.positions_view(query, tmp_path)
    assert code == 200
    ((point,),) = [t["points"] for t in reply["tracks"]]
    assert (point["x"], point["y"], point["room"]) == (
        pytest.approx(2.93, abs=0.03),
        pytest.approx(1.0, abs=0.03),
        "r1",
    )
    assert (reply["cameras"], reply["map"]) == (1, False)
    assert reply["camera_notes"] == [
        {"camera": "cam0", "by": "spot", "missing": []},
        {"camera": "porch", "by": "spot", "missing": ["facing"]},
    ]


def test_a_run_of_the_job_fits_the_tilts_records_the_footprints_and_places_them(tmp_path, monkeypatch, pinhole):
    place = taught(tmp_path, "x-camera cam0 at 1,1 height 2 yaw 90")
    feet, heads, _ = walkers(0.0)
    monkeypatch.setattr(spots, "people", lambda state, camera, now: (feet, heads))
    monkeypatch.setattr(spots, "frame_size", lambda camera, t: SIZE)
    seen = [footprints.Footprint("cam0", T - 900, 0.5, 1.0, 7, 0.9)]
    monkeypatch.setattr(footprints, "collect", lambda state, start, end, cameras: seen if start < T - 900 else [])
    # the placer reads the footprints the run recorded, from the store's own files
    monkeypatch.setattr(S, "at", lambda state=None: S.Log(S.root(tmp_path)))
    did = P.run(tmp_path, ["cam0"], T)
    assert (did["recorded"], did["records"]) == (1, 1)
    assert spots.read(tmp_path)["cam0"].missing == ()
    (p,) = P.positions(tmp_path, T - 3600, T, place=place)
    assert (p.x, p.y, p.room) == (pytest.approx(2.93, abs=0.03), pytest.approx(1.0, abs=0.03), "Living")
    again = P.run(tmp_path, ["cam0"], T + 3600)  # the next hour's run has nothing new to record or place
    assert (again["recorded"], again["records"]) == (0, 0)


def test_a_persons_footprint_its_camera_cant_see_by_the_plan_gets_no_point(tmp_path, vision, monkeypatch, pinhole):
    # The frame's bottom right corner from a level camera 2 m up at (1, 1) facing east lands about (2.9, 4.4): in the
    # hall, behind the wall at y = 3 from the camera. A person's foot point is pose's, and a pose that puts feet
    # behind a wall isn't a piece of furniture the person stood on.
    place = taught(tmp_path, "x-camera cam0 at 1,1 height 2 yaw 90")
    fit(tmp_path, monkeypatch)
    foot(vision, T, u=1.0, v=1.0, track="7")
    foot(vision, T + 2, u=0.5, v=1.0)  # in the living room, in view
    hidden: set = set()
    rows = P.derive(tmp_path, T - 1, T + 5, place=place, hidden=hidden)
    assert [(p.x, p.y) for p in rows if p.x is not None] == [pytest.approx((2.93, 1.0), abs=0.03)]
    assert hidden == {("cam0", "cam0:7", T)}


def test_an_animal_its_camera_cant_see_on_the_floor_is_placed_at_the_furniture_in_its_room(
    tmp_path, vision, monkeypatch, pinhole
):
    # The frame's bottom right corner from the same camera is (2.9, 4.4), in the hall behind the living room's far
    # wall (y = 3): on the floor that is where a pet up on a cabinet against that wall is cast to. It is put at the
    # furniture instead: the sight line from the spot (1, 1) leaves the living room at (2.13, 3), and the point is
    # three tenths back along it, so it is inside the room, at the cabinet, and says it is approximate.
    place = taught(tmp_path, "x-camera cam0 at 1,1 height 2 yaw 90")
    fit(tmp_path, monkeypatch)
    foot(vision, T, u=1.0, v=1.0, track="5:cat:1")
    hidden: set = set()
    (p,) = P.derive(tmp_path, T - 1, T + 5, place=place, hidden=hidden)
    assert (p.place, p.room, p.room_id) == (place, "Living", "r1")
    assert (p.x, p.y) == pytest.approx((1.98, 2.74), abs=0.02)
    assert p.y is not None and p.y < 3.0, "short of the wall the floor point was beyond"
    assert (p.approximate, p.uncertain, p.in_frame) == (True, False, True)
    assert hidden == set(), "put at the furniture, so not left out"


def test_an_animal_put_at_the_furniture_is_stored_approximate(tmp_path, vision, monkeypatch, pinhole):
    place = taught(tmp_path, "x-camera cam0 at 1,1 height 2 yaw 90")
    fit(tmp_path, monkeypatch)
    foot(vision, T, u=1.0, v=1.0, track="5:cat:1")
    assert P.advance(tmp_path, T + 60, chunk_s=P.BACKFILL_S)["records"] == 1
    (p,) = P.positions(tmp_path, T - 1, T + 5, place=place)
    assert (p.approximate, p.room) == (True, "Living")
    assert (p.x, p.y) == pytest.approx((1.98, 2.74), abs=0.02)


def test_the_owners_spot_outranks_the_maps_registration_of_the_camera(tmp_path, vision, monkeypatch, pinhole):
    # The map (y down, floor at y = 0, plan (x, y) = map (x, z)) registers cam0 at plan (1, 0) looking along the
    # plan's +y, which would put the heels at (1, 1.93). The owner's plan, drawn on that map's plan, says the camera
    # is at (3, 1) facing the same way (yaw 180), so they land at (3, 2.93).
    registered = places.CameraPose((1.0, 0, 0, 1.0, 0, 1.0, 0, -2.0, 0, 0, 1.0, 0.0, 0, 0, 0, 1.0), FOCAL, *SIZE, T)
    rev = places.revise(
        tmp_path,
        None,
        producer="test",
        cameras={"cam0": registered},
        floor=(0.0, -1.0, 0.0, 0.0),
        map_bytes=b"x",
        map_format="ply",
        plan=ROOMS,
        plan_transform=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0),
        now=T - 100,
    )
    foot(vision, T)
    assert [(p.x, p.y) for p in P.derive(tmp_path, T - 1, T + 5)] == [(1.0, 1.93)]
    places.correct_plan(tmp_path, rev.place, ROOMS + "x-camera cam0 at 3,1 height 2 yaw 180\n", now=T - 50)
    lens.hold(tmp_path, "cam0", HELD, now=T - 50)
    fit(tmp_path, monkeypatch)
    (p,) = P.derive(tmp_path, T - 1, T + 5)
    assert (p.x, p.y) == pytest.approx((3.0, 2.93), abs=0.03)
    assert (p.room, p.revision) == ("Living", rev.id)


def test_a_dry_run_counts_what_each_camera_would_place_and_keeps_nothing(tmp_path, monkeypatch, pinhole):
    taught(tmp_path, "x-camera cam0 at 1,1 height 2 yaw 90")
    feet, heads, _ = walkers(0.0)
    monkeypatch.setattr(spots, "people", lambda state, camera, now: (feet, heads))
    monkeypatch.setattr(spots, "frame_size", lambda camera, t: SIZE)
    seen = [
        footprints.Footprint("cam0", T - 900, 0.5, 1.0, 7, 0.9),  # on the floor, in the living room
        footprints.Footprint("cam0", T - 850, 1.0, 1.0, 7, 0.9),  # in the hall, behind the wall from the camera
        footprints.Footprint("cam0", T - 840, 1.0, 1.0, "5:cat:1", 0.9),  # an animal there too
        footprints.Footprint("cam0", T - 800, 0.5, 0.2, 7, 0.9),  # above the horizon: no floor point
        footprints.Footprint("porch", T - 700, 0.5, 1.0, 3, 0.9),  # a camera with no spot and no pose
    ]
    monkeypatch.setattr(footprints, "collect", lambda state, start, end, cameras: seen)
    cam0, porch = P.dry_run(tmp_path, ["cam0", "porch"], T, 24.0)
    # The animal behind the wall is put at the furniture, in the living room, so it's placed and only the person is
    # left unplaced behind a wall.
    assert (cam0["by"], cam0["footprints"], cam0["placed"], cam0["in_a_room"], cam0["missing"]) == ("spot", 4, 2, 2, [])
    assert (cam0["people_placed"], cam0["behind_wall"]) == (1, {"person": 1, "animal": 0})
    assert (porch["by"], porch["footprints"], porch["placed"], porch["behind_wall"]) == (None, 1, 0, None)
    assert spots.aims(tmp_path) == {}
    assert not S.root(tmp_path).exists()


def test_a_map_may_refine_a_spot_only_within_the_bound():
    spot = places.CameraPose(tuple(float(x) for x in np.eye(4).ravel()), FOCAL, *SIZE)
    near, far = np.eye(4), np.eye(4)
    near[0, 3], far[0, 3] = spots.REFINE_M - 0.01, spots.REFINE_M + 0.01
    assert spots.within_bounds(spot, places.CameraPose(tuple(float(x) for x in near.ravel()), FOCAL, *SIZE))
    assert not spots.within_bounds(spot, places.CameraPose(tuple(float(x) for x in far.ravel()), FOCAL, *SIZE))


WIDE = lens.Lens(SIZE[0], SIZE[1], (280.0, 280.0, SIZE[0] / 2 - 0.5, SIZE[1] / 2 - 0.5), -0.3)


def bend(uv: np.ndarray, undone: float) -> np.ndarray:
    """Pinhole pixels at FOCAL (the true rays) as WIDE's barrel shows them raw, for a camera whose undone frames
    (`lens.undistort_points`) are a pinhole of focal `undone` (a share of the width)."""
    fx, fy, cx, cy = WIDE.fractions()
    scale = undone / (lens.out_scale(WIDE) * fx)  # true normalized rays over the calibration's
    xu = (uv[:, 0] - 0.5) * SIZE[0] / FOCAL * scale
    yu = (uv[:, 1] - 0.5) * SIZE[1] / FOCAL * scale
    xd, yd = lens._distort(WIDE, xu, yu)
    return np.column_stack([cx + fx * xd, cy + fy * yd])


@pytest.mark.parametrize(
    ("configured", "held", "undone"),
    [
        pytest.param(None, HELD, None, id="pinhole"),
        # held on the raw frames before its lens was read (no `lens` on it): the undone frame isn't the one it was
        # measured on, so the calibration's own focal holds (0.366 of the width), not 0.6
        pytest.param(WIDE, lens.Held(0.6, 0.0, "fitted on raw frames"), lens.undone_focal(WIDE), id="wide-held-raw"),
        pytest.param(WIDE, None, lens.undone_focal(WIDE), id="wide-nothing-held"),
        pytest.param(WIDE, lens.Held(0.45, 0.0, "fit", lens=WIDE.as_tuple()), 0.45, id="wide-held-through-it"),
    ],
)
def test_a_footprint_at_a_known_floor_spot_is_placed_on_that_spot(
    tmp_path, vision, monkeypatch, configured, held, undone
):
    def seen(uv: np.ndarray) -> np.ndarray:
        return uv if undone is None else bend(uv, undone)

    monkeypatch.setattr(lens, "as_registered", lambda camera: None if configured is None else configured.as_tuple())
    place = taught(tmp_path, "x-camera cam0 at 1,1 height 2.2 yaw 90", held=False)
    if held is not None:
        lens.hold(tmp_path, "cam0", held, now=T - 50)
    feet, heads, _ = walkers(18.0, above=2.2)
    monkeypatch.setattr(spots, "people", lambda state, camera, now: (seen(feet), seen(heads)))
    spots.fit(tmp_path, T, sizes=lambda camera, t: SIZE)
    # Heels 2.5 m ahead of the camera and 0.8 m to its right, near the frame's edge: it faces east on the page, so
    # its right is south, and they stand at (3.5, 1.8).
    ((u, v),) = seen(pixels([[0.8, 2.5, footprints.FOOT_M]], 18.0, above=2.2))
    foot(vision, T, u=float(u), v=float(v))
    (p,) = P.derive(tmp_path, T - 1, T + 5, place=place)
    assert (p.x, p.y) == pytest.approx((3.5, 1.8), abs=0.05)


def test_a_configured_barrel_lens_is_undone_when_fitting_a_spots_tilt_and_placing_footprints(
    tmp_path, vision, monkeypatch
):
    # A camera whose barrel lens is in the device manifest holds the undone frame's focal in lens.held, while pose
    # tracks and footprints come in raw camera pixels.
    configured = lens.Lens(SIZE[0], SIZE[1], (280.0, 280.0, SIZE[0] / 2 - 0.5, SIZE[1] / 2 - 0.5), -0.3)
    held_undone = lens.Held(280.0 * lens.out_scale(configured) / SIZE[0], 0.0, "test", lens=configured.as_tuple())
    fx, fy, cx, cy = configured.fractions()

    def barrel(uv: np.ndarray) -> np.ndarray:
        xu = (uv[:, 0] - 0.5) * SIZE[0] / FOCAL
        yu = (uv[:, 1] - 0.5) * SIZE[1] / FOCAL
        xd, yd = lens._distort(configured, xu, yu)
        return np.column_stack([cx + fx * xd, cy + fy * yd])

    feet_pin, heads_pin, tall = walkers(18.0, roll=-4.0, above=2.2)
    feet, heads = barrel(feet_pin), barrel(heads_pin)
    place = taught(tmp_path, "x-camera cam0 at 1,1 height 2.2 yaw 90", held=False, rooms=OPEN)
    lens.hold(tmp_path, "cam0", held_undone, now=T - 50)
    monkeypatch.setattr(spots, "people", lambda state, camera, now: (feet, heads))

    # Fitted first without the manifest lens, then read after the manifest gets it: the stale aim must refit.
    monkeypatch.setattr(lens, "as_registered", lambda camera: None)
    spots.fit(tmp_path, T - 10, sizes=lambda camera, t: SIZE)
    monkeypatch.setattr(lens, "as_registered", lambda camera: configured.as_tuple())
    assert spots.read(tmp_path)["cam0"].missing == ("tilt",)

    fitted = spots.fit(tmp_path, T, sizes=lambda camera, t: SIZE)["cam0"]
    assert (fitted.pitch, fitted.roll) == pytest.approx((18.0, -4.0), abs=0.6)
    assert fitted.tall == pytest.approx(tall, abs=0.05)

    # A foot point near the edge of the raw barrel frame lands where its undone ray meets the floor.
    raw_foot = barrel(np.array([[0.85, 0.92]]))[0]
    foot(vision, T, u=float(raw_foot[0]), v=float(raw_foot[1]))
    (p,) = P.derive(tmp_path, T - 1, T + 5, place=place)
    monkeypatch.setattr(lens, "as_registered", lambda camera: None)
    lens.hold(tmp_path, "cam0", HELD, now=T + 1)
    ux, uy, uz = (float(v) for v in spots.up_at(18.0, -4.0))
    spots.save_aims(
        tmp_path,
        {"cam0": spots.Aim((ux, uy, uz), *SIZE, 120, tall, T + 2, "2.2|None|0.28125/0.00000")},
    )
    latest_rev = places.latest(tmp_path, place)
    assert latest_rev is not None
    want_rev = spots.overlay(latest_rev, spots.read(tmp_path))
    want = places.floor_point(want_rev.cameras["cam0"], want_rev.floor, 0.85, 0.92, height=footprints.FOOT_M)
    assert want is not None
    assert (p.x, p.y) == pytest.approx(want[:2], abs=0.04)


def test_a_configured_barrel_lens_is_undone_for_a_spot_with_only_a_field_of_view(tmp_path, monkeypatch):
    # Nothing held: the spot's `fov` is the undone frame's, and the manifest's lens still undoes the raw pixels.
    configured = lens.Lens(SIZE[0], SIZE[1], (280.0, 280.0, SIZE[0] / 2 - 0.5, SIZE[1] / 2 - 0.5), -0.3)
    undone_focal = 280.0 * lens.out_scale(configured)
    fov = math.degrees(2 * math.atan(SIZE[0] / (2 * undone_focal)))
    fx, fy, cx, cy = configured.fractions()

    # `walkers` gives pinhole pixels at FOCAL: their true rays, which the barrel lens bends.
    def barrel(uv: np.ndarray) -> np.ndarray:
        xu = (uv[:, 0] - 0.5) * SIZE[0] / FOCAL
        yu = (uv[:, 1] - 0.5) * SIZE[1] / FOCAL
        xd, yd = lens._distort(configured, xu, yu)
        return np.column_stack([cx + fx * xd, cy + fy * yd])

    feet_pin, heads_pin, _ = walkers(18.0, roll=-4.0, above=2.2)
    taught(tmp_path, f"x-camera cam0 at 1,1 height 2.2 yaw 90 fov {fov:.4f}", held=False)
    monkeypatch.setattr(spots, "people", lambda state, camera, now: (barrel(feet_pin), barrel(heads_pin)))
    monkeypatch.setattr(lens, "as_registered", lambda camera: configured.as_tuple())

    fitted = spots.fit(tmp_path, T, sizes=lambda camera, t: SIZE)["cam0"]

    assert (fitted.pitch, fitted.roll) == pytest.approx((18.0, -4.0), abs=0.6)
    pose = spots.read(tmp_path)["cam0"].pose
    assert pose is not None
    assert pose.lens == configured.as_tuple()
