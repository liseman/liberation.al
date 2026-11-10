"""What the jobs that work from recorded frames share: which cameras are fixed and grouped, how frames are scored
and people masked, how MASt3R's runtime is called, and a cloud's floor and plan. MASt3R itself never runs here: a fake
runtime stands in for it at the subprocess boundary."""

import datetime as dt
import os
import sys
import textwrap
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from liberational.spatial import cameras, floorplan, plan_import, rig
from liberational.spatial.mast3r_runner import (
    floor_from_normals,
    groups,
    lone,
    pair_edges,
    pair_pose,
    pnp_inliers,
    up_from_normals,
)

IDENTITY = [float(x) for x in np.eye(4).ravel()]


# Match counts from the six fixed cameras on the Linux machine.
SPIKE = [
    ("s2", "s6", 3528),
    ("s2", "s5", 669),
    ("s5", "s6", 596),
    ("s1", "s2", 187),
    ("s3", "s5", 119),
    ("s1", "s3", 67),
    ("s4", "s6", 16),
]
SIX = ["s1", "s2", "s3", "s4", "s5", "s6"]


@pytest.mark.parametrize(
    ("matches", "min_matches", "want"),
    [
        pytest.param(SPIKE, 1000, [["s2", "s6"]], id="spike-only-the-living-room-pair"),
        pytest.param(SPIKE, 500, [["s2", "s5", "s6"]], id="spike-weak-tie-joins-at-a-lower-bar"),
        pytest.param(
            [("s1", "s2", 1000), ("s2", "s3", 1000), ("s4", "s5", 2000)],
            1000,
            [["s1", "s2", "s3"], ["s4", "s5"]],
            id="chains-join-largest-first",
        ),
        pytest.param([("s1", "s2", 999)], 1000, [], id="just-under-the-bar"),
        pytest.param([("s1", "s9", 5000)], 1000, [], id="unknown-camera"),
    ],
)
def test_cameras_joined_by_enough_matches_form_groups_of_two_or_more(matches, min_matches, want):
    assert groups(SIX, matches, min_matches) == want


def test_fixed_cameras_are_the_enabled_cameras_without_a_fisheye_lens(tmp_path):
    manifest = tmp_path / "devices.toml"
    manifest.write_text(
        textwrap.dedent(
            """
            [[device]]
            name = "desk"
            kind = "camera"
            [[device]]
            name = "mic"
            kind = "microphone"
            [[device]]
            name = "rig-front"
            kind = "camera"
            fisheye = { width = 1920, height = 1080, k = [1, 1, 1, 1], d = [0, 0, 0, 0] }
            [[device]]
            name = "porch"
            kind = "rtsp"
            [[device]]
            name = "pack-back"
            kind = "camera"
            mount = { rig = "pack", hfov_deg = 120.0 }
            [[rig]]
            name = "pack"
            [[device]]
            name = "attic"
            kind = "rtsp"
            enabled = false
            """
        )
    )
    assert cameras.fixed_cameras(manifest) == ["desk", "porch"]
    assert cameras.fixed_cameras(tmp_path / "missing.toml") == []


GPUS = [
    {"index": 0, "uuid": "GPU-aaa", "total": 24 * 2**30, "used": 19 * 2**30},
    {"index": 1, "uuid": "GPU-bbb", "total": 24 * 2**30, "used": 11 * 2**30},
]


@pytest.mark.parametrize(
    ("setting", "found", "want"),
    [(-1, GPUS, "GPU-bbb"), (0, GPUS, "GPU-aaa"), (5, GPUS, None), (-1, None, None), (-1, [], None)],
    ids=["most-free", "configured", "configured-missing", "no-nvml", "no-gpus"],
)
def test_mast3r_runs_on_the_gpu_with_the_most_free_memory_unless_one_is_set(setting, found, want):
    assert cameras.pick_gpu(setting, found) == want


# A camera turned a quarter turn about its view axis: its -y points along the map's +x.
ON_ITS_SIDE = [0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def test_a_level_camera_says_the_cloud_is_y_down():
    assert cameras.up_direction({"a": IDENTITY, "b": IDENTITY}) == pytest.approx((0.0, -1.0, 0.0))


def test_a_camera_on_its_side_with_portrait_frames_doesnt_tilt_up():
    poses = {"level": IDENTITY, "side": ON_ITS_SIDE}
    sizes = {"level": [512, 288], "side": [288, 512]}
    assert cameras.up_direction(poses, sizes) == pytest.approx((0.0, -1.0, 0.0))
    assert cameras.up_direction({"side": ON_ITS_SIDE}, {"side": [288, 512]}) == pytest.approx((1.0, 0.0, 0.0))


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    """A MASt3R runtime whose python is this one and whose runner is a fake that records its job and environment."""
    rt = tmp_path / "mast3r"
    (rt / ".venv" / "bin").mkdir(parents=True)
    (rt / "src" / "mast3r").mkdir(parents=True)
    os.symlink(sys.executable, rt / ".venv" / "bin" / "python")
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(b"")
    fake = tmp_path / "runner.py"
    fake.write_text(
        textwrap.dedent(
            """
            import json, os, sys
            job = json.load(open(sys.argv[1]))
            if job["images"].get("fail"):
                sys.exit("CUDA out of memory")
            out = {"matches": [["a", "b", 4000]], "groups": [], "job": job, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}
            json.dump(out, open(os.path.join(job["out"], "result.json"), "w"))
            """
        )
    )
    monkeypatch.setattr(cameras, "RUNNER", fake)
    monkeypatch.setattr(cameras, "checkpoint", lambda: ckpt)
    return rt


def test_the_runner_gets_the_images_and_the_bar_on_the_chosen_gpu(runtime, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    got = cameras.run_mast3r(runtime, {"a": tmp_path / "a.png", "b": tmp_path / "b.png"}, out, "GPU-bbb")
    assert got["matches"] == [["a", "b", 4000]]
    assert got["gpu"] == "GPU-bbb"
    assert got["job"]["images"] == {"a": str(tmp_path / "a.png"), "b": str(tmp_path / "b.png")}
    assert got["job"]["min_matches"] == cameras.MIN_MATCHES
    assert got["job"]["repo"] == str(runtime / "src")


def test_a_failed_run_says_why(runtime, tmp_path):
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        cameras.run_mast3r(runtime, {"fail": tmp_path / "x.png"}, tmp_path, None)


def test_a_missing_runtime_says_how_to_install_it(tmp_path):
    with pytest.raises(RuntimeError, match="mise run mast3r"):
        cameras.run_mast3r(tmp_path / "nowhere", {}, tmp_path, None)


def room(seed: int = 0, width: float = 4.0, depth: float = 3.0) -> np.ndarray:
    """A `width` x `depth` m room 2.6 m high, as MASt3R's y-down frame has it: floor at y = 0, ceiling at y = -2.6."""
    rng = np.random.default_rng(seed)
    n = 4000
    wall_x = np.r_[rng.uniform(0, width, n), rng.uniform(0, width, n), np.zeros(n), np.full(n, width)]
    wall_z = np.r_[np.zeros(n), np.full(n, depth), rng.uniform(0, depth, n), rng.uniform(0, depth, n)]
    wall_h = rng.uniform(0, 2.6, 4 * n)
    floor = np.c_[rng.uniform(0, width, n), rng.uniform(0, depth, n)]
    xs = np.r_[wall_x, floor[:, 0]]
    zs = np.r_[wall_z, floor[:, 1]]
    hs = np.r_[wall_h, np.zeros(n)]
    return np.c_[xs, -hs, zs].astype(np.float32) + rng.normal(0, 0.005, (5 * n, 3)).astype(np.float32)


def frame(b, g, r):
    return np.full((48, 64, 3), (b, g, r), dtype=np.uint8)


@pytest.mark.parametrize(
    ("pixels", "lit"),
    [
        pytest.param(frame(90, 120, 160), True, id="lit-colour"),
        pytest.param(frame(20, 25, 30), False, id="dark"),
        pytest.param(frame(140, 140, 141), False, id="infrared-grey"),
        pytest.param(np.full((48, 64), 150, dtype=np.uint8), False, id="single-channel"),
    ],
)
def test_only_lit_colour_frames_go_to_mast3r(pixels, lit):
    assert cameras.daylight(pixels) is lit


def test_a_frames_light_is_its_mean_brightness_and_channel_spread():
    assert cameras.light(frame(90, 120, 160)) == pytest.approx((370 / 3, 70.0))
    assert cameras.light(np.zeros((4, 4), dtype=np.uint8)) == (0.0, 0.0)


def test_a_place_plan_goes_through_the_tiling_converter(monkeypatch):
    # The baseline found a door between two rooms wider than the 1.2 m wall they share; the plan the place gets is
    # clamped to it, and written as a scene.
    rooms = (
        floorplan.Room("r1", ((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0))),
        floorplan.Room("r2", ((4.0, 0.0), (6.0, 0.0), (6.0, 1.2), (4.0, 1.2))),
        floorplan.Room("r3", ((4.0, 1.2), (6.0, 1.2), (6.0, 3.0), (4.0, 3.0))),
    )
    door = floorplan.Opening("door", ("r1", "r2"), (4.0, 0.6), 1.5)
    found = floorplan.Floorplan(rooms, (door,), np.eye(3, 4))
    monkeypatch.setattr(cameras.floorplan, "floorplan", lambda *a, **k: found)
    out = cameras.layout(room(), (0.0, -1.0, 0.0), "Den")
    assert out.plan is not None
    (found_door,) = plan_import.parse(out.plan).openings
    assert (found_door.kind, found_door.at, found_door.width) == ("door", pytest.approx((4.0, 0.6)), 1.0)
    assert out.transform == pytest.approx(np.eye(3, 4).ravel().tolist(), abs=1e-12)  # axis-aligned: no turn


@pytest.mark.parametrize(
    ("kw", "want"),
    [
        pytest.param({}, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], id="every-pair"),
        pytest.param({"window": 1}, [(0, 1), (1, 2), (2, 3)], id="neighbours"),
        pytest.param({"offsets": [1, 3, 0, 3]}, [(0, 1), (0, 3), (1, 2), (2, 3)], id="offsets-only"),
    ],
)
def test_the_runner_matches_the_pairs_asked_for(kw, want):
    assert sorted(pair_edges(4, **kw)) == want


def turned(deg):
    """A rotation of `deg` degrees about y."""
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def test_a_pairs_relative_pose_comes_from_one_views_two_pointmaps_over_its_confident_pixels():
    rng = np.random.default_rng(0)
    own = rng.uniform(-1, 1, (12, 16, 3)) + [0, 0, 3]  # camera b's points in its own frame
    rot, shift, scale = turned(30), np.array([0.5, -0.1, 2.0]), 1.1
    other = scale * own @ rot.T + shift  # the same points in camera a's frame, in that pair's scale
    conf = np.full((12, 16), 3.0)
    conf[:4] = 1.0  # unsure rows, and wrong
    other[:4] = rng.uniform(-50, 50, (4, 16, 3))
    pose, got_scale = pair_pose(own, other, conf, conf)
    assert pose[:3, :3] == pytest.approx(rot)
    assert pose[:3, 3] == pytest.approx(shift), "camera b's centre, its own origin, sits at the shift in a's frame"
    assert got_scale == pytest.approx(scale)
    with pytest.raises(ValueError, match="confident"):
        pair_pose(own, other, np.ones((12, 16)), conf)


def test_pnp_keeps_matches_that_project_where_they_were_seen_and_not_shuffled_ones():
    rng = np.random.default_rng(1)
    points = rng.uniform(-1, 1, (60, 3)) + [0, 0, 4]
    rot, shift = turned(10), np.array([0.3, 0.0, 0.2])
    cam = points @ rot.T + shift
    pixels = 300.0 * cam[:, :2] / cam[:, 2:] + [160.0, 120.0]
    assert pnp_inliers(points, pixels, 300.0, (160.0, 120.0)) == 60
    assert pnp_inliers(points, rng.permutation(pixels), 300.0, (160.0, 120.0)) < 30
    assert pnp_inliers(points[:5], pixels[:5], 300.0, (160.0, 120.0)) == 0


# --- a camera that sees no other camera's space


@pytest.mark.parametrize(
    ("matches", "want"),
    [
        pytest.param([("a", "b", 4000), ("a", "c", 212), ("b", "c", 150)], ["c"], id="no-tie"),
        pytest.param([("a", "b", 4000), ("a", "c", 600), ("b", "c", 150)], [], id="weak-tie-is-not-lone"),
        pytest.param([("a", "b", 299), ("a", "c", 0), ("b", "c", 299)], ["a", "b", "c"], id="all-alone"),
        pytest.param([("a", "b", 300)], ["c"], id="the-bar-is-a-tie"),
    ],
)
def test_a_camera_is_lone_when_none_of_its_pairs_reaches_the_bar(matches, want):
    assert lone(["a", "b", "c"], matches, 300) == want


def test_the_only_camera_is_lone():
    assert lone(["bedroom"], [], 300) == ["bedroom"]


def pitched_view(height=2.0, pitch_deg=35.0, w=320, h=240, f=200.0):
    """The point map (H x W x 3, OpenCV camera frame) of a camera `height` m above the floor, pitched down, looking
    along the room's +y: the top rows see a wall at y = 5, the middle a table top 0.75 m high (most of the frame), and
    the bottom rows the floor. Also the room's up in the camera's frame."""
    a = np.radians(pitch_deg)
    # camera axes in the room (z up): x right, y down, z forward (pitched down by a)
    right, fwd = np.array([1.0, 0, 0]), np.array([0, np.cos(a), -np.sin(a)])
    down = np.cross(fwd, right)
    rot = np.stack([right, down, fwd], axis=1)  # camera to room
    centre = np.array([0.0, 0.0, height])
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    rays = np.stack([(u - w / 2) / f, (v - h / 2) / f, np.ones_like(u)], axis=-1) @ rot.T
    wall = np.arange(h) < 60
    table = (np.arange(h) >= 60) & (np.arange(h) < 190)
    t = np.where(
        wall[:, None],
        (5.0 - centre[1]) / rays[..., 1],
        np.where(table[:, None], (0.75 - height) / rays[..., 2], -height / rays[..., 2]),
    )
    pts = centre + rays * t[..., None]
    in_camera = (pts - centre) @ rot  # room to camera
    return in_camera, rot.T @ np.array([0.0, 0.0, 1.0])


def test_a_lone_cameras_up_and_floor_come_from_its_surfaces_not_its_tilt():
    points, up = pitched_view()
    conf = np.full(points.shape[:2], 3.0)
    found = np.array(up_from_normals(points, conf))
    assert np.degrees(np.arccos(found @ up)) < 1.0  # the camera's own -y is 35 degrees off
    floor = floor_from_normals(points, conf, tuple(found))
    assert floor is not None
    # the floor, not the table top that fills most of the frame: 2 m under the camera, which sits at its origin
    assert np.array(floor[:3]) == pytest.approx(up, abs=0.01)
    assert floor[3] == pytest.approx(2.0, abs=0.02)


def test_a_view_with_no_horizontal_surface_has_no_floor():
    points, _ = pitched_view()
    conf = np.full(points.shape[:2], 3.0)
    conf[60:] = 0.0  # only the wall is confident
    assert floor_from_normals(points, conf, (0.0, -1.0, 0.0)) is None


def test_cameras_mounted_outdoors_are_left_out_in_order():
    mounts = rig.Mounts(
        cameras={
            "patio": rig.Mount("patio", outdoor=True),
            "bedroom": rig.Mount("bedroom", height_m=2.0),
            "driveway": rig.Mount("driveway", outdoor=True),
        }
    )
    assert cameras.indoor_cameras(["cam0", "patio", "bedroom", "driveway"], mounts) == ["cam0", "bedroom"]
    assert cameras.indoor_cameras(["cam0"], rig.Mounts()) == ["cam0"]


def test_the_runner_is_asked_for_lone_cameras_only_when_given_a_bar(runtime, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    images = {"a": tmp_path / "a.png", "b": tmp_path / "b.png"}
    assert "lone_matches" not in cameras.run_mast3r(runtime, images, out, None)["job"]
    assert cameras.run_mast3r(runtime, images, out, None, lone_matches=300)["job"]["lone_matches"] == 300


# --- each camera's best recorded frame

UTC = ZoneInfo("UTC")


def test_frames_are_sampled_only_around_midday_on_the_last_day():
    now = dt.datetime(2030, 1, 2, 12, 0, tzinfo=UTC).timestamp()
    times = cameras.sample_times(now, UTC)
    local = [dt.datetime.fromtimestamp(t, UTC) for t in times]
    assert all(dt.time(10, 30) <= w.time() < dt.time(14, 30) for w in local)
    assert min(times) >= now - 86400 and max(times) < now
    # a day's lookback holds one 4-hour midday: 12 tries 20 minutes apart, here split by noon across the two days
    assert len(times) == 12 and {w.day for w in local} == {1, 2}
    assert [w.strftime("%H:%M") for w in local][:2] == ["12:00", "12:20"] and local[-1].strftime("%H:%M") == "11:40"


def test_the_midday_window_follows_the_houses_zone():
    now = dt.datetime(2030, 1, 2, 12, 0, tzinfo=UTC).timestamp()
    la = ZoneInfo("America/Los_Angeles")
    assert all(
        dt.time(10, 30) <= dt.datetime.fromtimestamp(t, la).time() < dt.time(14, 30)
        for t in cameras.sample_times(now, la)
    )
    assert cameras.sample_times(now, la) != cameras.sample_times(now, UTC)


def test_a_cloud_with_no_flat_layer_along_up_has_no_floor_there():
    points, up = pitched_view()
    wall = points[:60].reshape(-1, 3)  # only the wall rows: every surface stands upright
    assert cameras.floor_along(wall, up) is None
    assert cameras.floor_along(points.reshape(-1, 3), up) is not None


# People left out of the frame a camera is registered from


def rfdetr(*people):
    """RF-DETR's raw answer for queries at (cx, cy, w, h, person logit) and one cat query: dets and 91 sparse logits."""
    dets = np.array([p[:4] for p in people] + [(0.5, 0.5, 0.2, 0.2)], np.float32)
    logits = np.full((len(dets), 91), -9.0, np.float32)
    for i, p in enumerate(people):
        logits[i, 1] = p[4]  # the person slot
    logits[-1, 17] = 9.0  # a sure cat (sparse COCO 17): never a person
    return dets, logits


def test_rfdetrs_sure_person_queries_are_the_people_in_a_frame():
    dets, logits = rfdetr((0.25, 0.5, 0.2, 0.6, 2.0), (0.9, 0.5, 0.4, 0.2, -3.0), (0.05, 0.95, 0.2, 0.2, 1.0))
    got = cameras.person_boxes(dets, logits)
    # the unsure one (sigmoid -3 = 0.05) is left out; the one over the frame's corner is clipped to it
    assert np.asarray(got) == pytest.approx(np.array([(0.15, 0.2, 0.35, 0.8), (0.0, 0.85, 0.15, 1.0)]))


def test_a_blurred_frame_scores_less_sharp_than_the_frame_it_was_blurred_from():
    cv2 = pytest.importorskip("cv2")
    crisp = np.random.default_rng(0).integers(0, 255, (72, 128, 3), dtype=np.uint8)
    assert cameras.sharpness(cv2.GaussianBlur(crisp, (9, 9), 3.0)) < cameras.sharpness(crisp) / 10
    assert cameras.sharpness(np.full((72, 128, 3), 90, np.uint8)) == 0.0


def test_overlapping_people_cover_their_union_of_the_frame():
    left, across = (0.0, 0.0, 0.5, 1.0), (0.25, 0.0, 0.75, 1.0)
    assert cameras.covered_share([]) == 0.0
    assert cameras.covered_share([left]) == pytest.approx(0.5)
    assert cameras.covered_share([left, across]) == pytest.approx(0.75)


def test_a_persons_mask_is_their_box_grown_a_tenth_each_way_and_clipped_to_the_frame():
    mask = cameras.person_mask([(0.4, 0.5, 0.6, 1.0)], 100, 50)
    assert mask.shape == (50, 100)
    # 20 px wide and 25 tall, grown 2 px and 2.5 px each way: columns 38 to 61, rows 22 down to the frame's bottom
    assert mask[:, 38:62].any(axis=0).all() and not mask[:, :38].any() and not mask[:, 62:].any()
    assert mask[22:, 50].all() and not mask[:22].any()
