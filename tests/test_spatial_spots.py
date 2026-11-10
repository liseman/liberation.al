"""A camera's pose from its owner's spot on the plan: the tilt fitted from walking people, what a spot lacks, and
footprints placed through it with no map."""

import math

import numpy as np
import pytest

from liberational import segment_log as S
from liberational.dashboard import places as view
from liberational.memory import places
from liberational.spatial import footprints, lens, plan_import, spots
from liberational.spatial import positions as P

T = 1_800_000_000.0
SIZE = (512, 288)
FOCAL = 144.0  # pixels: the bottom middle of the frame drops a metre a metre from a level camera
ROOMS = 'plan "Home" walls 0.2/0.1\nroom r1 "Living" other rect 0,0 4x3\nroom r2 "Hall" other rect 0,3 4x2\n'


def walkers(pitch, roll=0.0, above=2.0, n=120, seed=0, across=3.0):
    """Foot and head pixels of people 1.5 to 1.8 m tall walking 2 to 7 m in front of a camera `above` the floor,
    pitched `pitch` degrees down and rolled `roll` clockwise, worked out from the room's side: the world has z up,
    the camera looks along +y. Returns (feet, heads, the people's median height)."""
    rng = np.random.default_rng(seed)
    p, r = math.radians(pitch), math.radians(roll)
    forward = np.array([0.0, math.cos(p), -math.sin(p)])
    level_right, level_down = np.array([1.0, 0.0, 0.0]), np.array([0.0, -math.sin(p), -math.cos(p)])
    right = math.cos(r) * level_right + math.sin(r) * level_down
    down = -math.sin(r) * level_right + math.cos(r) * level_down
    eye = np.array([0.0, 0.0, above])
    tall = rng.uniform(1.5, 1.8, n)
    floor = np.column_stack([rng.uniform(-across, across, n), rng.uniform(2.0, 7.0, n)])

    def pixels(points):
        rel = points - eye
        x, y, z = rel @ right, rel @ down, rel @ forward
        return np.column_stack([0.5 + FOCAL * x / z / SIZE[0], 0.5 + FOCAL * y / z / SIZE[1]])

    feet = pixels(np.column_stack([floor, np.full(n, footprints.FOOT_M)]))
    heads = pixels(np.column_stack([floor, tall]))
    return feet, heads, float(np.median(tall))


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
    text = "x-camera hall at 1,2 height 2.1 yaw 90 fov 100\nx-camera porch at 3,4\nx-camera broken yaw\n# x-camera no at 0,0"
    assert plan_import.spots(text) == (
        plan_import.Camera("hall", (1.0, 2.0), 2.1, 90.0, 100.0),
        plan_import.Camera("porch", (3.0, 4.0)),
    )


@pytest.fixture
def pinhole(monkeypatch):
    monkeypatch.setattr(lens, "as_registered", lambda camera: None)  # no configured lens undoes the frames


def taught(tmp_path, spot, *, view=None, held=True):
    """A place kept across a reset: no map, and the owner's plan with one camera spot."""
    place = places.revise(tmp_path, None, producer="places reset v1", now=T - 100).place
    places.correct_plan(tmp_path, place, ROOMS + spot + "\n", now=T - 50, view=view)
    if held:
        lens.hold(tmp_path, "cam0", HELD, now=T - 50)
    return place


def fit(tmp_path, monkeypatch, pitch=0.0, above=2.0):
    """Fit the cameras' aims from walkers seen by a camera `above` the floor pitched `pitch` down."""
    feet, heads, _ = walkers(pitch, above=above)
    monkeypatch.setattr(spots, "people", lambda state, camera, now: (feet, heads))
    return spots.fit(tmp_path, T, sizes=lambda camera, t: SIZE)


def foot(vision, t, u=0.5, v=1.0, cam="cam0"):
    row = dict(source=P.FIXED_CAMERA, confidence=0.9, camera=cam, u=u, v=v, track=f"{cam}:7", entity=None)
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
            {"plan": ROOMS, "turn": 0.0, "shift": [0.0, 0.0], "mirror": True}, (2.93, -2.43, None), id="mirrored"
        ),
    ],
)
def test_the_cameras_right_hand_follows_the_plans_handedness(tmp_path, vision, monkeypatch, pinhole, view, want):
    place = taught(tmp_path, "x-camera cam0 at 1,1 height 2 yaw 90", view=view)
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
        footprints.Footprint("cam0", T - 800, 0.5, 0.2, 7, 0.9),  # above the horizon: no floor point
        footprints.Footprint("porch", T - 700, 0.5, 1.0, 3, 0.9),  # a camera with no spot and no pose
    ]
    monkeypatch.setattr(footprints, "collect", lambda state, start, end, cameras: seen)
    cam0, porch = P.dry_run(tmp_path, ["cam0", "porch"], T, 24.0)
    assert (cam0["by"], cam0["footprints"], cam0["placed"], cam0["in_a_room"], cam0["missing"]) == ("spot", 2, 1, 1, [])
    assert (porch["by"], porch["footprints"], porch["placed"]) == (None, 1, 0)
    assert spots.aims(tmp_path) == {}
    assert not S.root(tmp_path).exists()


def test_a_map_may_refine_a_spot_only_within_the_bound():
    spot = places.CameraPose(tuple(float(x) for x in np.eye(4).ravel()), FOCAL, *SIZE)
    near, far = np.eye(4), np.eye(4)
    near[0, 3], far[0, 3] = spots.REFINE_M - 0.01, spots.REFINE_M + 0.01
    assert spots.within_bounds(spot, places.CameraPose(tuple(float(x) for x in near.ravel()), FOCAL, *SIZE))
    assert not spots.within_bounds(spot, places.CameraPose(tuple(float(x) for x in far.ravel()), FOCAL, *SIZE))


def test_a_configured_barrel_lens_is_undone_when_fitting_a_spots_tilt_and_placing_footprints(
    tmp_path, vision, monkeypatch
):
    # A camera whose barrel lens is in the device manifest holds the undone frame's focal in lens.held, while pose
    # tracks and footprints come in raw camera pixels.
    configured = lens.Lens(SIZE[0], SIZE[1], (280.0, 280.0, SIZE[0] / 2 - 0.5, SIZE[1] / 2 - 0.5), -0.3)
    held_undone = lens.Held(280.0 * lens.out_scale(configured) / SIZE[0], 0.0, "test")
    fx, fy, cx, cy = configured.fractions()

    def barrel(uv: np.ndarray) -> np.ndarray:
        xu = (uv[:, 0] - 0.5) * SIZE[0] / FOCAL
        yu = (uv[:, 1] - 0.5) * SIZE[1] / FOCAL
        xd, yd = lens._distort(configured, xu, yu)
        return np.column_stack([cx + fx * xd, cy + fy * yd])

    feet_pin, heads_pin, tall = walkers(18.0, roll=-4.0, above=2.2)
    feet, heads = barrel(feet_pin), barrel(heads_pin)
    place = taught(tmp_path, "x-camera cam0 at 1,1 height 2.2 yaw 90", held=False)
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
