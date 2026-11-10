"""Fixed-camera places (#2825): which cameras are grouped, how MASt3R's runtime is called, and what is written as a
place. MASt3R itself never runs here: a fake runtime stands in for it at the subprocess boundary."""

import datetime as dt
import json
import os
import sys
import textwrap
from dataclasses import replace
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from liberational import config as cfg
from liberational.memory import places
from liberational.spatial import cameras, floorplan, footprints, fuse, lens, ply
from liberational.spatial.mast3r_runner import (
    floor_from_normals,
    groups,
    lone,
    pair_edges,
    pair_pose,
    pnp_inliers,
    up_from_normals,
)

T = 1_800_000_000.0
IDENTITY = [float(x) for x in np.eye(4).ravel()]


# The #2786 spike's match counts on devbox's six fixed cameras.
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


def answer(out, group_cameras, points, k=0):
    """A runner answer with one group whose cameras all sit level at the origin."""
    colors = np.full((len(points), 3), 200, dtype=np.uint8)
    np.savez(out / f"group{k}.npz", points=points, colors=colors)
    return {
        "cameras": group_cameras,
        "cloud": f"group{k}.npz",
        "poses": {c: IDENTITY for c in group_cameras},
        "focals": {c: 300.0 for c in group_cameras},
        "sizes": {c: [512, 288] for c in group_cameras},
    }


def test_a_group_becomes_a_place_with_its_map_plan_and_camera_poses(tmp_path):
    cloud = room()
    result = {"groups": [answer(tmp_path, ["s2", "s6"], cloud)]}
    (rev,) = cameras.write_places(tmp_path, result, tmp_path, T)
    assert rev.signals == {"camera": ["s2", "s6"]}
    assert set(rev.cameras) == {"s2", "s6"}
    assert rev.cameras["s2"] == places.CameraPose(tuple(IDENTITY), 300.0, 512, 288, T)
    assert rev.map_format == "ply"
    (places.root(tmp_path) / rev.place / str(rev.map)).rename(tmp_path / "back.ply")
    assert ply.load(tmp_path / "back.ply") == pytest.approx(cloud.astype(np.float64))
    assert "rect" in str(places.machine_plan(tmp_path, rev))
    assert rev.floor == pytest.approx((0.0, -1.0, 0.0, 0.0), abs=0.02)  # the room's floor is y = 0, up is -y
    assert rev.plan_transform is not None and rev.plan_transform[8:] == pytest.approx(rev.floor, abs=1e-9)
    assert places.camera_places(tmp_path) == {"s2": rev.place, "s6": rev.place}


def test_the_next_night_revises_the_same_place_and_keeps_its_other_signals(tmp_path):
    home = places.revise(tmp_path, None, producer="wifi", signals={"wifi": ["aa:bb"], "camera": ["s2", "s6"]}, now=T)
    result = {"groups": [answer(tmp_path, ["s2", "s5", "s6"], room())]}
    (rev,) = cameras.write_places(tmp_path, result, tmp_path, T + 86400)
    assert rev.place == home.place
    assert rev.signals == {"wifi": ["aa:bb"], "camera": ["s2", "s5", "s6"]}


def test_a_cloud_with_no_floor_or_walls_still_registers_its_cameras_without_a_plan(tmp_path):
    scatter = np.random.default_rng(1).uniform(-1, 1, (50, 3)).astype(np.float32)
    (rev,) = cameras.write_places(tmp_path, {"groups": [answer(tmp_path, ["a", "b"], scatter)]}, tmp_path, T)
    assert (rev.plan, rev.plan_transform) == (None, None)
    assert set(rev.cameras) == {"a", "b"}


def test_a_floor_without_walls_records_the_floor_and_no_plan(tmp_path):
    rng = np.random.default_rng(3)
    floor_only = np.c_[rng.uniform(0, 4, 3000), np.zeros(3000), rng.uniform(0, 3, 3000)].astype(np.float32)
    (rev,) = cameras.write_places(tmp_path, {"groups": [answer(tmp_path, ["a", "b"], floor_only)]}, tmp_path, T)
    assert rev.plan is None
    assert rev.floor == pytest.approx((0.0, -1.0, 0.0, 0.0), abs=0.02)


def test_footprints_give_a_cloud_without_walls_its_first_plan(tmp_path):
    # The floor-only cloud above has no plan; people's feet all over that 4 x 3 m floor close it into one room.
    rng = np.random.default_rng(3)
    floor_only = np.c_[rng.uniform(0, 4, 3000), np.zeros(3000), rng.uniform(0, 3, 3000)].astype(np.float32)
    group = answer(tmp_path, ["a", "b"], floor_only)
    up_high = [1.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0, -2.0, 0.0, 0.0, 1.0, -2.0, 0.0, 0.0, 0.0, 1.0]  # 2 m up, at z = -2
    group["poses"] = {"a": up_high, "b": up_high}

    def feet(cams):
        assert cams == ["a", "b"]
        spots = rng.uniform([0.4, 0.4], [3.6, 2.6], (4000, 2))  # x, z on the floor, kept off the walls
        # Each spot as camera a's pixel: (x - 2, 2, z + 2) in its frame, through its 300 px focal over 512 x 288.
        return footprints.cells(
            footprints.Footprint(
                "a", float(i), (x - 2) / (z + 2) * 300 / 512 + 0.5, 2 / (z + 2) * 300 / 288 + 0.5, 1, 0.9
            )
            for i, (x, z) in enumerate(spots.tolist())
        )

    (rev,) = cameras.write_places(tmp_path, {"groups": [group]}, tmp_path, T, feet=feet)
    plan = places.machine_plan(tmp_path, rev)
    assert plan is not None and rev.plan_transform is not None
    (line,) = [x for x in plan.splitlines() if x.startswith("room ")]
    w, h = sorted(float(v) for v in line.split()[-1].split("x"))
    assert (w, h) == pytest.approx((3.0, 4.0), abs=floorplan.WALK_BLOCK_M)


def test_two_groups_become_two_places(tmp_path):
    result = {"groups": [answer(tmp_path, ["a", "b"], room(), 0), answer(tmp_path, ["c", "d"], room(2), 1)]}
    revs = cameras.write_places(tmp_path, result, tmp_path, T)
    assert len({r.place for r in revs}) == 2
    assert json.loads(json.dumps(places.camera_places(tmp_path))) == {
        "a": revs[0].place,
        "b": revs[0].place,
        "c": revs[1].place,
        "d": revs[1].place,
    }


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


def test_a_place_that_split_keeps_its_larger_part_and_the_rest_becomes_a_new_place(tmp_path):
    home = places.revise(tmp_path, None, producer="p", signals={"camera": ["a", "b", "c"]}, now=T).place
    result = {"groups": [answer(tmp_path, ["c", "d"], room(), 0), answer(tmp_path, ["a", "b"], room(1), 1)]}
    cd, ab = cameras.write_places(tmp_path, result, tmp_path, T + 86400)
    assert ab.place == home
    assert cd.place != home
    assert [len(places.revisions(tmp_path, p)) for p in (home, cd.place)] == [2, 1]  # one write into home this run


# A joined place's rooms come back as one group a room every night (#3115)

DAY = 86400.0
Y_DOWN_FLOOR = [0.0, -1.0, 0.0, 0.0]  # MASt3R's frame: y down, the floor at y = 0


def at(x: float, height: float, z: float, turn_deg: float = 0.0) -> np.ndarray:
    """A level camera `height` m above MASt3R's floor at (x, z), turned `turn_deg` about the vertical."""
    a = np.radians(turn_deg)
    m = np.eye(4)
    m[:3, :3] = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
    m[:3, 3] = (x, -height, z)
    return m


def night(turn_deg: float, x: float, z: float, scale: float = 1.0) -> np.ndarray:
    """A MASt3R run's own frame against another's: turned about the vertical, shifted along the floor, and scaled."""
    out = at(x, 0.0, z, turn_deg)
    out[:3, :3] *= scale
    return out


def group_in(out, k, cams, points, poses, frame):
    """A runner group of `cams` at `poses` seeing `points` (a room), all in the run's `frame`; lone with one camera."""
    pts = points @ frame[:3, :3].T + frame[:3, 3]
    np.savez(out / f"group{k}.npz", points=pts.astype(np.float32), colors=np.full((len(pts), 3), 200, np.uint8))
    moved = {c: frame @ poses[c] for c in cams}
    for m in moved.values():
        m[:3, :3] /= np.cbrt(np.linalg.det(m[:3, :3]))
    group = {
        "cameras": cams,
        "cloud": f"group{k}.npz",
        "poses": {c: [float(x) for x in m.ravel()] for c, m in moved.items()},
        "focals": {c: 300.0 for c in cams},
        "sizes": {c: [512, 288] for c in cams},
        "floor": Y_DOWN_FLOOR,
    }
    return group | ({"up": [0.0, -1.0, 0.0]} if len(cams) == 1 else {})


LIVING = {"cam0": at(1.0, 2.0, 0.5), "living": at(3.0, 1.8, 2.5, 180.0)}
BEDROOM = {"bedroom": at(1.5, 1.6, 0.4)}


def tonight(out, living_frame, bedroom_frame, bedroom_points=None):
    return {
        "groups": [
            group_in(out, 0, ["cam0", "living"], room(1), LIVING, living_frame),
            group_in(
                out,
                1,
                ["bedroom"],
                room(2, 3.0, 2.5) if bedroom_points is None else bedroom_points,
                BEDROOM,
                bedroom_frame,
            ),
        ]
    }


def position(rev, cam):
    return np.asarray(rev.cameras[cam].pose).reshape(4, 4)[:3, 3]


@pytest.fixture
def joined(tmp_path):
    """Place 1 (the living room's two cameras) and the bedroom's place, each fused, then joined: the bedroom stands
    10 m along the living room's x, turned 90 degrees."""
    living, bedroom = cameras.write_places(tmp_path, tonight(tmp_path, np.eye(4), np.eye(4)), tmp_path, T)
    assert living.place != bedroom.place
    fuse.refine_all(tmp_path, now=T + 1)
    move = np.eye(4)
    move[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    move[:3, 3] = (10.0, 0.0, 0.0)
    home = fuse.join(tmp_path, living.place, {bedroom.place: move}, now=T + 2)
    return home, bedroom.place


def test_every_room_of_a_joined_place_keeps_writing_into_it_night_after_night(tmp_path, joined):
    home, retired = joined
    for k, (living_frame, bedroom_frame) in enumerate(
        [(night(30, 0.4, -0.2, 1.1), night(-70, 2.0, 1.0, 0.9)), (night(-15, -1.0, 0.6), night(120, 0.3, 0.3))],
        start=1,
    ):
        revs = cameras.write_places(tmp_path, tonight(tmp_path, living_frame, bedroom_frame), tmp_path, T + k * DAY)
        assert [r.place for r in revs] == [home.place, home.place], f"night {k}: both groups wrote into the home"
        assert places.place_ids(tmp_path) == [home.place], f"night {k}: no new place, the bedroom's stays retired"
        assert places.merged_into(tmp_path, retired) == home.place
    latest = places.latest(tmp_path, home.place)
    assert latest is not None and latest.producer == fuse.PRODUCER and latest.frame == home.frame
    assert {c: p.at for c, p in latest.cameras.items()} == {
        "cam0": T + 2 * DAY,
        "living": T + 2 * DAY,
        "bedroom": T + 2 * DAY,
    }
    for cam in ("cam0", "living", "bedroom"):  # each night's run, in its own frame and scale, seated where it was
        assert position(latest, cam) == pytest.approx(position(home, cam), abs=0.03), cam
    assert sorted(latest.signals["camera"]) == ["bedroom", "cam0", "living"]
    assert places.camera_places(tmp_path) == dict.fromkeys(["bedroom", "cam0", "living"], home.place)


def test_rooms_seated_in_one_run_write_one_revision_each_built_on_the_one_before(tmp_path, joined):
    # #3220: one run's seats share its time, so places.revise keeps them apart by a millisecond each
    home, _ = joined
    living, bedroom = cameras.write_places(tmp_path, tonight(tmp_path, np.eye(4), np.eye(4)), tmp_path, T + DAY)
    assert bedroom.sources == (living.id,), "the bedroom's seat fuses into the map the living room's just wrote"
    assert living.sources == (home.id,)
    assert home.at < living.at < bedroom.at
    assert [r.id for r in places.revisions(tmp_path, home.place)[-2:]] == [living.id, bedroom.id]
    assert bedroom.cameras["cam0"] == living.cameras["cam0"], "the later seat keeps the earlier one's cameras"


def test_a_room_whose_snapshot_lands_off_the_map_leaves_the_joined_place_as_it_was(tmp_path, joined):
    # The bedroom camera's cloud tonight is 6 m from where its pose puts it in the home: it moved, or the run is wrong.
    home, _ = joined
    off = room(2, 3.0, 2.5) + np.array([6.0, 0.0, 0.0], np.float32)
    (living,) = cameras.write_places(tmp_path, tonight(tmp_path, np.eye(4), np.eye(4), off), tmp_path, T + DAY)
    assert living.place == home.place
    assert places.place_ids(tmp_path) == [home.place]
    latest = places.latest(tmp_path, home.place)
    assert latest is not None and latest.cameras["bedroom"] == home.cameras["bedroom"]


def bedroom_feet(frame):
    """The bedroom camera's walked footprints tonight, as cells of its frame: heels across the room's far half."""
    pose = frame @ BEDROOM["bedroom"]
    rot = pose[:3, :3] / np.cbrt(np.linalg.det(pose[:3, :3]))
    spots = np.random.default_rng(3).uniform([0.5, 1.2], [2.5, 2.4], (100, 2))
    heels = np.c_[spots[:, 0], np.full(len(spots), -footprints.FOOT_M), spots[:, 1]] @ frame[:3, :3].T + frame[:3, 3]
    c = (heels - pose[:3, 3]) @ rot  # each heel in the camera's frame
    uv = np.c_[c[:, 0] / c[:, 2] * 300 / 512 + 0.5, c[:, 1] / c[:, 2] * 300 / 288 + 0.5]
    return lambda cams: [footprints.Cells("bedroom", uv, np.ones(len(uv), np.int64))] if "bedroom" in cams else []


def test_a_seated_room_keeps_mast3rs_pose_when_geocalib_is_sure_so_it_stays_put_night_after_night(tmp_path, joined):
    # #3228: GeoCalib's pitch turned the bedroom's pose and slid it on its footprints, off its own cloud; the seat
    # started from it, ICP put the cloud back, and the pose was written 0.45 m off the map's bedroom
    home, _ = joined
    for k, unc in enumerate([2.0, cameras.MAX_PITCH_UNC_DEG + 0.2, 2.0], start=1):
        frame = night(25.0 * k, 0.3 * k, -0.2, 1.0)
        sure = {"bedroom": gravity(8.0, focal=0.7, unc=unc)}
        result = tonight(tmp_path, np.eye(4), frame)
        revs = cameras.write_places(tmp_path, result, tmp_path, T + k * DAY, feet=bedroom_feet(frame), gravity=sure)
        assert [r.place for r in revs] == [home.place, home.place], f"night {k}: both rooms seated"
        bedroom = revs[-1].cameras["bedroom"]
        assert position(revs[-1], "bedroom") == pytest.approx(position(home, "bedroom"), abs=0.03), f"night {k}"
        assert tilt_of(bedroom)[0] == pytest.approx(0.0, abs=0.5), f"night {k}: MASt3R's level camera, not 8 down"
        assert bedroom.focal == pytest.approx(300.0), f"night {k}: MASt3R's focal"


def test_a_repose_puts_a_camera_back_and_keeps_the_map_and_frame(tmp_path, joined):
    home, _ = joined
    cloud = ply.read(places.root(tmp_path) / home.place / home.map)[0]
    pose = np.asarray(home.cameras["bedroom"].pose).reshape(4, 4)
    pose[:3, 3] += (0.45, -0.3, 0.0)
    off = replace(home.cameras["bedroom"], pose=tuple(float(x) for x in pose.ravel()))
    bad = fuse.repose(tmp_path, home.place, {"bedroom": off}, now=T + DAY)
    assert bad is not None and bad.cameras["bedroom"] == off
    fixed = fuse.repose(tmp_path, home.place, {"bedroom": home.cameras["bedroom"]}, now=T + DAY + 1)
    assert fixed is not None and fixed.sources == (bad.id,)
    assert fixed.cameras == home.cameras and fixed.frame == home.frame and fixed.signals == home.signals
    assert ply.read(places.root(tmp_path) / home.place / fixed.map)[0] == pytest.approx(cloud)


def test_a_repose_of_a_camera_the_place_doesnt_have_writes_nothing(tmp_path, joined):
    home, _ = joined
    assert fuse.repose(tmp_path, home.place, {"porch": home.cameras["bedroom"]}, now=T + DAY) is None
    assert places.latest(tmp_path, home.place) == home


def test_a_place_keeps_its_map_when_a_camera_had_no_usable_frame(tmp_path):
    (was,) = cameras.write_places(tmp_path, {"groups": [answer(tmp_path, ["a", "b", "c"], room())]}, tmp_path, T)
    tonight = {"groups": [answer(tmp_path, ["a", "b"], room(1))]}
    assert cameras.write_places(tmp_path, tonight, tmp_path, T + 86400, absent=["c"]) == []
    assert places.latest(tmp_path, was.place) == was
    (moved,) = cameras.write_places(tmp_path, tonight, tmp_path, T + 2 * 86400)  # c was seen and no longer matches
    assert (moved.place, sorted(moved.cameras)) == (was.place, ["a", "b"])


def test_each_camera_is_registered_at_the_run_that_saw_it(tmp_path):
    (rev,) = cameras.write_places(tmp_path, {"groups": [answer(tmp_path, ["a", "b"], room())]}, tmp_path, T)
    assert {c: p.at for c, p in rev.cameras.items()} == {"a": T, "b": T}


def test_a_frames_light_is_its_mean_brightness_and_channel_spread():
    assert cameras.light(frame(90, 120, 160)) == pytest.approx((370 / 3, 70.0))
    assert cameras.light(np.zeros((4, 4), dtype=np.uint8)) == (0.0, 0.0)


def test_a_place_plan_goes_through_the_tiling_converter(monkeypatch):
    # The baseline found a door between two rooms wider than the 1.2 m wall they share; the plan the place gets is
    # clamped to it, so the library takes it as it is.
    rooms = (
        floorplan.Room("r1", ((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0))),
        floorplan.Room("r2", ((4.0, 0.0), (6.0, 0.0), (6.0, 1.2), (4.0, 1.2))),
        floorplan.Room("r3", ((4.0, 1.2), (6.0, 1.2), (6.0, 3.0), (4.0, 3.0))),
    )
    door = floorplan.Opening("door", ("r1", "r2"), (4.0, 0.6), 1.5)
    found = floorplan.Floorplan(rooms, (door,), np.eye(3, 4))
    monkeypatch.setattr(cameras.floorplan, "floorplan", lambda *a, **k: found)
    out = cameras.layout(room(), (0.0, -1.0, 0.0), "Den")
    assert out.plan is not None and "door r1>r2 at 4,0.6 w1\n" in out.plan
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


# --- a camera that sees no other camera's space (#3000)


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


def test_a_lone_group_becomes_a_place_on_the_floor_it_was_given(tmp_path):
    points, up = pitched_view()
    floor = [float(x) for x in (*up, 2.0)]
    group = answer(tmp_path, ["bedroom"], points.reshape(-1, 3).astype(np.float32))
    group.update(up=[float(x) for x in up], floor=floor)
    (rev,) = cameras.write_places(tmp_path, {"groups": [group]}, tmp_path, T)
    assert set(rev.cameras) == {"bedroom"} and rev.signals == {"camera": ["bedroom"]}
    assert rev.floor == pytest.approx(floor)  # a fit to this cloud would take the table top


def test_outdoor_cameras_are_left_out_in_order():
    assert cameras.indoor_cameras(["cam0", "patio", "bedroom", "driveway"], ["driveway", "patio"]) == [
        "cam0",
        "bedroom",
    ]
    assert cameras.indoor_cameras(["cam0"], []) == ["cam0"]


def test_the_runner_is_asked_for_lone_cameras_only_when_given_a_bar(runtime, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    images = {"a": tmp_path / "a.png", "b": tmp_path / "b.png"}
    assert "lone_matches" not in cameras.run_mast3r(runtime, images, out, None)["job"]
    assert cameras.run_mast3r(runtime, images, out, None, lone_matches=300)["job"]["lone_matches"] == 300


# --- each camera's best recorded frame (#3008)

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


def lit_frame(sharp: bool, seed: int = 0):
    """A lit, coloured BGR frame: fine stripes when `sharp`, a smooth gradient when not."""
    rng = np.random.default_rng(seed)
    frame = np.zeros((90, 160, 3), np.uint8)
    frame[..., 0], frame[..., 1], frame[..., 2] = 60, 120, 200
    if sharp:
        frame[:, ::2] = frame[:, ::2] // 2
    return frame + rng.integers(0, 3, frame.shape, dtype=np.uint8)


def test_the_best_frame_is_the_sharpest_lit_one_with_nobody_in_view():
    night = np.full((90, 160, 3), 20, np.uint8)
    shots = {1.0: lit_frame(False), 2.0: lit_frame(True), 3.0: night, 4.0: lit_frame(True, seed=1)}
    found = cameras.best_frame("cam0", list(shots), lambda cam, t: shots[t], people=[(3.5, 4.5)])
    assert found is not None and found[0] == 2.0  # 4.0 is as sharp, but someone was in view


def test_with_someone_in_view_at_every_lit_time_the_sharpest_is_still_taken():
    shots = {1.0: lit_frame(False), 2.0: lit_frame(True)}
    found = cameras.best_frame("cam0", list(shots), lambda cam, t: shots[t], people=[(0.0, 9.0)])
    assert found is not None and found[0] == 2.0


def test_a_camera_with_no_lit_frame_has_none():
    assert cameras.best_frame("cam0", [1.0, 2.0], lambda cam, t: np.full((90, 160, 3), 20, np.uint8)) is None
    assert cameras.best_frame("cam0", [1.0], lambda cam, t: None) is None


def test_frames_saves_each_cameras_best_and_leaves_out_one_with_none(tmp_path, vision):
    now = dt.datetime(2030, 1, 2, 12, 0, tzinfo=UTC).timestamp()

    def read(cam, t):
        return lit_frame(True) if cam == "cam0" else None

    got = cameras.frames(tmp_path, ["cam0", "hall"], tmp_path, now, read, UTC)
    assert list(got) == ["cam0"] and got["cam0"].is_file()


# GeoCalib's gravity and focal over MASt3R's (#2960). GeoCalib never runs here: a fake runtime stands in for it.

LEVEL = (0.0, 0.0, 1.0, 0.0)


def facing(heading_deg, pitch_deg, x=0.0, y=0.0, height=2.0, focal=300.0):
    """A camera in a levelled frame (floor z = 0) at (x, y, height), facing `heading_deg` and `pitch_deg` down."""
    h, p = np.radians(heading_deg), np.radians(pitch_deg)
    forward = np.array([np.cos(p) * np.cos(h), np.cos(p) * np.sin(h), -np.sin(p)])
    right = np.array([np.sin(h), -np.cos(h), 0.0])
    m = np.eye(4)
    m[:3, :3] = np.c_[right, np.cross(forward, right), forward]
    m[:3, 3] = (x, y, height)
    return places.CameraPose(tuple(float(v) for v in m.ravel()), focal, 640, 360, T)


def up_of(pitch_deg, roll_deg=0.0):
    """A camera's up in its own frame (x right, y down, z forward) at that pitch down and roll."""
    p, r = np.radians(pitch_deg), np.radians(roll_deg)
    return (float(np.cos(p) * np.sin(r)), float(-np.cos(p) * np.cos(r)), float(-np.sin(p)))


def gravity(pitch_deg, roll_deg=0.0, focal=0.5, unc=1.5):
    return cameras.Gravity(up_of(pitch_deg, roll_deg), focal, unc, 1.0)


def heading(pose):
    f = np.asarray(pose.pose).reshape(4, 4)[:3, 2]
    return np.degrees(np.arctan2(f[1], f[0]))


def tilt_of(pose, floor=LEVEL):
    return cameras.tilt(tuple(np.asarray(pose.pose).reshape(4, 4)[:3, :3].T @ np.asarray(floor[:3])))


@pytest.mark.parametrize("pitch,roll", [(12.0, 0.0), (8.0, 15.0), (30.0, -10.0)])
def test_geocalibs_gravity_sets_pitch_roll_and_focal_and_keeps_yaw_and_position(pitch, roll):
    was = facing(40.0, 21.0, x=1.0, y=2.0)
    got = cameras.regravitate(was, LEVEL, gravity(pitch, roll, focal=0.6))
    assert tilt_of(got) == pytest.approx((pitch, roll), abs=1e-6)
    assert got.focal == pytest.approx(0.6 * 640)
    assert np.asarray(got.pose)[[3, 7, 11]] == pytest.approx([1.0, 2.0, 2.0])
    assert heading(got) == pytest.approx(40.0)
    rot = np.asarray(got.pose).reshape(4, 4)[:3, :3]
    assert rot @ rot.T == pytest.approx(np.eye(3), abs=1e-9)


def walked_cells(camera, pose, spots):
    uv = []
    m = np.asarray(pose.pose).reshape(4, 4)
    for x, y in spots:
        c = m[:3, :3].T @ (np.array([x, y, footprints.FOOT_M]) - m[:3, 3])  # people's heels, a little up
        uv.append((c[0] / c[2] * pose.focal / pose.width + 0.5, c[1] / c[2] * pose.focal / pose.height + 0.5))
    return footprints.Cells(camera, np.array(uv), np.ones(len(uv), dtype=np.int64))


def test_a_turned_camera_slides_so_its_footprints_stay_where_they_were():
    was = facing(0.0, 25.0)
    spots = np.random.default_rng(0).uniform([2.5, -1.0], [5.0, 1.0], (200, 2))
    cells = walked_cells("a", was, spots)
    turned = cameras.regravitate(was, LEVEL, gravity(15.0))
    got = cameras.reanchor(was, turned, cells, LEVEL)

    def landed(pose):
        return [places.floor_point(pose, LEVEL, u, v, footprints.FOOT_M) for u, v in cells.uv.tolist()]

    both = [a is not None and b is not None for a, b in zip(landed(was), landed(turned), strict=True)]

    def median(pose):  # over the cells both poses place (the turned camera's shallower rays lose the farthest)
        return np.median(np.array([p for p, ok in zip(landed(pose), both, strict=True) if ok])[:, :2], axis=0)

    assert np.linalg.norm(median(turned) - median(was)) > 1.0, "turning alone swings the footprints out"
    assert median(got) == pytest.approx(median(was), abs=0.05)
    assert np.asarray(got.pose)[11] == pytest.approx(2.0), "the camera stays at its height"
    assert cameras.reanchor(was, turned, None, LEVEL) == turned


@pytest.mark.parametrize(
    "unc,applied",
    [pytest.param(2.9, True, id="sure"), pytest.param(cameras.MAX_PITCH_UNC_DEG + 0.1, False, id="unsure")],
)
def test_only_a_sure_gravity_replaces_mast3rs(unc, applied):
    poses = {"a": facing(0.0, 25.0), "b": facing(180.0, 30.0, x=6.0)}
    got = cameras.correct(poses, LEVEL, {"a": gravity(12.0, unc=unc)}, [])
    assert (tilt_of(got["a"])[0] == pytest.approx(12.0)) == applied
    assert got["b"] == poses["b"], "a camera GeoCalib didn't answer for keeps MASt3R's"
    assert cameras.correct(poses, None, {"a": gravity(12.0)}, []) == poses, "nothing without a floor"


def test_cameras_registered_together_keep_mast3rs_poses_against_a_sure_gravity(tmp_path):
    # #2960: turning one camera of a joint registration about its own centre moved its floor points 0.5 m off the
    # other camera's
    result = {"groups": [answer(tmp_path, ["s2", "s6"], room())]}
    (rev,) = cameras.write_places(tmp_path, result, tmp_path, T, gravity={"s2": gravity(10.0, focal=0.7)})
    assert list(rev.cameras["s2"].pose) == IDENTITY and rev.cameras["s2"].focal == 300.0


def test_a_lone_camera_takes_a_sure_gravity_and_focal(tmp_path):
    points, up = pitched_view(pitch_deg=35.0)
    group = answer(tmp_path, ["bedroom"], points.reshape(-1, 3).astype(np.float32))
    group.update(up=list(up), floor=[*up, 2.0])
    sure = cameras.Gravity(up_of(30.0), 0.7, cameras.MAX_PITCH_UNC_DEG, 1.0)  # GeoCalib reads 30 degrees down
    (rev,) = cameras.write_places(tmp_path, {"groups": [group]}, tmp_path, T, gravity={"bedroom": sure})
    assert rev.floor is not None
    assert tilt_of(rev.cameras["bedroom"], rev.floor)[0] == pytest.approx(30.0, abs=0.5)
    assert rev.cameras["bedroom"].focal == pytest.approx(0.7 * 512)


def test_a_lone_cameras_held_focal_wins_over_tonights(tmp_path):
    # #3289: a camera's own intrinsics are held; neither MASt3R's focal nor GeoCalib's replaces them
    points, up = pitched_view(pitch_deg=35.0)
    group = answer(tmp_path, ["bedroom"], points.reshape(-1, 3).astype(np.float32))
    group.update(up=list(up), floor=[*up, 2.0])
    sure = cameras.Gravity(up_of(30.0), 0.7, cameras.MAX_PITCH_UNC_DEG, 1.0)
    lens.hold(tmp_path, "bedroom", lens.Held(0.5, 0.0, "walk fit"), now=T)
    (rev,) = cameras.write_places(tmp_path, {"groups": [group]}, tmp_path, T, gravity={"bedroom": sure})
    assert rev.cameras["bedroom"].focal == pytest.approx(0.5 * 512)
    assert rev.cameras["bedroom"].lens is None


@pytest.fixture
def geocalib(tmp_path, monkeypatch):
    """A GeoCalib runtime whose python is this one, running a fake runner that answers for every image but `bad`."""
    rt = tmp_path / "geocalib"
    (rt / ".venv" / "bin").mkdir(parents=True)
    os.symlink(sys.executable, rt / ".venv" / "bin" / "python")
    ckpt = tmp_path / "geocalib.tar"
    ckpt.write_bytes(b"")
    fake = tmp_path / "geocalib_runner.py"
    fake.write_text(
        textwrap.dedent(
            """
            import json, os, sys
            job = json.load(open(sys.argv[1]))
            found = {c: {"up": [0.0, -0.98, -0.2], "focal": 0.63, "pitch_unc": 1.8, "roll_unc": 1.1, "vfov_unc": 5.0}
                     for c in job["images"] if c != "bad"}
            out = {"cameras": found, "errors": {"bad": "ValueError: no lines"} if "bad" in job["images"] else {}}
            json.dump(out, open(os.path.join(job["out"], "result.json"), "w"))
            """
        )
    )
    monkeypatch.setattr(cameras, "GEOCALIB_RUNNER", fake)
    monkeypatch.setattr(cameras, "geocalib_checkpoint", lambda: ckpt)
    return rt


def test_geocalib_answers_for_each_frame_it_could_read(geocalib, tmp_path):
    got = cameras.run_geocalib(geocalib, {"cam0": tmp_path / "a.png", "bad": tmp_path / "b.png"}, tmp_path / "out")
    assert got == {"cam0": cameras.Gravity((0.0, -0.98, -0.2), 0.63, 1.8, 1.1)}


def test_without_geocalibs_runtime_cameras_keep_mast3rs_gravity(tmp_path, caplog):
    assert cameras.run_geocalib(tmp_path / "nowhere", {"cam0": tmp_path / "a.png"}, tmp_path / "out") == {}
    assert "mise run geocalib" in caplog.text


def tipped(up, degrees):
    """`up` turned `degrees` about the camera's x axis: GeoCalib a little off."""
    a = np.radians(degrees)
    rot = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    return tuple(float(x) for x in rot @ up)


@pytest.mark.parametrize(
    "unc,found",
    [
        pytest.param(3.1, True, id="sure-enough"),
        pytest.param(cameras.LONE_UP_UNC_DEG + 0.5, False, id="unsure"),
        pytest.param(None, False, id="no-geocalib"),
    ],
)
def test_a_lone_cameras_floor_is_the_lowest_layer_along_geocalibs_up(tmp_path, unc, found):
    points, up = pitched_view()
    # The runner's floor went wrong: a plane tilted 18 degrees, through the room
    wrong_up = np.array(tipped(up, 18.0))
    wrong = [*(float(x) for x in wrong_up), 1.2]
    group = answer(tmp_path, ["bedroom"], points.reshape(-1, 3).astype(np.float32))
    group.update(up=[float(x) for x in wrong_up], floor=wrong)
    gravity = {} if unc is None else {"bedroom": cameras.Gravity(tipped(up, 3.0), 0.6, unc, 1.0)}
    (rev,) = cameras.write_places(tmp_path, {"groups": [group]}, tmp_path, T, gravity=gravity)
    assert rev.floor is not None
    if found:
        # the floor 2 m under the camera, not the table top that fills most of the frame, and level to within GeoCalib's
        # error but fitted to the floor itself
        assert np.array(rev.floor[:3]) == pytest.approx(up, abs=0.01)
        assert rev.floor[3] == pytest.approx(2.0, abs=0.02)
    else:
        assert rev.floor == pytest.approx(wrong)


def test_a_cloud_with_no_flat_layer_along_up_has_no_floor_there():
    points, up = pitched_view()
    wall = points[:60].reshape(-1, 3)  # only the wall rows: every surface stands upright
    assert cameras.floor_along(wall, up) is None
    assert cameras.floor_along(points.reshape(-1, 3), up) is not None


@pytest.mark.parametrize(
    "floor_d,measured,kept",
    [
        pytest.param(2.0, None, False, id="two-metres-up"),
        pytest.param(0.11, None, True, id="floor-right-under-the-camera"),
        pytest.param(2.0, 0.91, True, id="against-a-measured-0.91"),
        pytest.param(2.0, 1.5, False, id="against-a-measured-1.5"),
    ],
)
def test_a_lone_camera_with_an_impossible_floor_keeps_its_places_last_revision(tmp_path, floor_d, measured, kept):
    # 2026-11-08: a kitchen camera 0.91 m up was registered 0.11 above its floor, and its place scaled by 8.3 (#2960)
    points, up = pitched_view()
    good = answer(tmp_path, ["kitchen"], points.reshape(-1, 3).astype(np.float32))
    good.update(up=[float(x) for x in up], floor=[*(float(x) for x in up), 2.0])
    (was,) = cameras.write_places(tmp_path, {"groups": [good]}, tmp_path, T)
    tonight = dict(good, floor=[*(float(x) for x in up), floor_d])
    with cfg.override({"spatial": {"camera_heights_m": {"kitchen": measured} if measured else {}}}):
        got = cameras.write_places(tmp_path, {"groups": [tonight]}, tmp_path, T + 86400)
    assert (got == []) == kept
    latest = places.latest(tmp_path, was.place)
    assert latest is not None and (latest.id == was.id) == kept


# What a registered camera sees behind people, kept with its pose (#3070)


def test_a_registered_groups_cameras_keep_what_they_see_behind_people(tmp_path, monkeypatch):
    from liberational.spatial import registration

    cloud = room()  # MASt3R's y-down frame: the floor at y = 0
    pose = np.eye(4)
    pose[:3, 3] = (2.0, -1.5, 1.0)  # 1.5 m over the floor
    seen = np.zeros((16, 32, 3), np.float32)  # s2's point map: 2 x 4 cells of 8 px
    seen[:, :16, 1] = 0.0  # the left half sees the floor
    seen[:, 16:, 1] = -0.9  # the right half a couch's back, 0.9 m up
    seen[:8, 24:] = np.nan  # nothing sure in the top right cell
    found = registration.Registered(
        {"s2": pose, "s6": pose},
        {"s2": 300.0, "s6": 300.0},
        {"s2": 0.0, "s6": 0.0},
        cloud,
        np.full((len(cloud), 3), 200, np.uint8),
        "s2",
        {"s6": 1.0},
        len(cloud),
        {"s2": seen, "s6": seen},
    )
    monkeypatch.setattr(cameras, "reproject", lambda out, group, held=None: found)
    group = answer(tmp_path, ["s2", "s6"], cloud) | {"pairs": "group0-pairs.npz"}
    (rev,) = cameras.write_places(tmp_path, {"groups": [group]}, tmp_path, T)
    bg = rev.cameras["s2"].background
    assert bg is not None and (bg.cols, bg.rows) == (4, 2)
    assert bg.at(0.1, 0.75) == pytest.approx(0.0, abs=0.02)
    assert bg.at(0.6, 0.75) == pytest.approx(0.6, abs=0.02)  # 0.9 m over the camera's 1.5
    assert bg.at(0.9, 0.25) is None
    # read back from the revision as saved
    again = places.latest(tmp_path, rev.place)
    assert again is not None and again.cameras["s2"].background == bg
    # The camera looks level; at v = 0.75 its ray drops 0.24 m a metre, so the couch hides (0.9 - 0.07) / 0.24 m
    assert footprints.at_edge(again.cameras["s2"], again.floor, 0.6, 0.75) is True
    assert footprints.at_edge(again.cameras["s2"], again.floor, 0.4, 0.75) is False


def test_a_camera_registered_by_mast3rs_alignment_alone_has_no_background(tmp_path):
    (rev,) = cameras.write_places(tmp_path, {"groups": [answer(tmp_path, ["s2", "s6"], room())]}, tmp_path, T)
    assert rev.cameras["s2"].background is None
    assert footprints.at_edge(rev.cameras["s2"], rev.floor, 0.5, 0.9) is None


# People left out of the frame a camera is registered from (#3085)


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


def test_a_frame_with_nobody_in_it_is_chosen_over_a_sharper_one_with_someone():
    shots = {1.0: lit_frame(False), 2.0: lit_frame(True), 3.0: lit_frame(True, seed=1)}
    seen = {id(shots[1.0]): [], id(shots[2.0]): [(0.1, 0.1, 0.3, 0.9)], id(shots[3.0]): [(0.0, 0.0, 1.0, 1.0)]}
    found = cameras.best_frame("cam0", list(shots), lambda cam, t: shots[t], detect=lambda f: seen[id(f)])
    assert found is not None and (found[0], found[2]) == (1.0, [])
    # with someone in every frame, the one they cover least
    seen[id(shots[1.0])] = [(0.0, 0.0, 0.5, 1.0)]
    found = cameras.best_frame("cam0", list(shots), lambda cam, t: shots[t], detect=lambda f: seen[id(f)])
    assert found is not None and (found[0], found[2]) == (2.0, [(0.1, 0.1, 0.3, 0.9)])


def test_the_people_in_the_chosen_frame_are_masked_beside_it_grown_a_little(tmp_path):
    now = dt.datetime(2030, 1, 2, 12, 0, tzinfo=UTC).timestamp()
    frame = lit_frame(True)  # 160 x 90
    got = cameras.frames(
        tmp_path, ["cam0"], tmp_path, now, lambda cam, t: frame, UTC, detect=lambda f: [(0.25, 0.2, 0.5, 0.8)]
    )
    assert list(got) == ["cam0"]
    import cv2

    saved = cv2.imread(str(tmp_path / "cam0.mask.png"), cv2.IMREAD_GRAYSCALE)
    assert saved is not None
    rows, cols = np.nonzero(np.asarray(saved) > 127)
    # the box (x 40 to 80, y 18 to 72 px) grown by a tenth of its size each way: x 36 to 84, y 12.6 to 77.4
    assert (cols.min(), cols.max() + 1, rows.min(), rows.max() + 1) == (36, 84, 12, 78)


def bedroom_off(tmp_path, when):
    """A night whose bedroom snapshot lands 6 m off where its camera is registered in the home (it moved)."""
    off = room(2, 3.0, 2.5) + np.array([6.0, 0.0, 0.0], np.float32)
    return cameras.write_places(tmp_path, tonight(tmp_path, np.eye(4), np.eye(4), off), tmp_path, when)


def test_a_camera_refused_its_seat_three_runs_running_is_registered_on_its_own(tmp_path, joined):
    home, _ = joined
    for k in (1, 2):
        assert [r.place for r in bedroom_off(tmp_path, T + k * DAY)] == [home.place], f"night {k}: only cam0 wrote"
        assert places.camera_places(tmp_path)["bedroom"] == home.place, f"night {k}: one bad snapshot isn't a move"
    revs = bedroom_off(tmp_path, T + 3 * DAY)
    new = [r for r in revs if r.place != home.place]
    assert len(new) == 1 and set(new[0].cameras) == {"bedroom"} and new[0].signals["camera"] == ["bedroom"]
    assert sorted(places.place_ids(tmp_path)) == sorted([home.place, new[0].place])
    assert places.camera_places(tmp_path) == {"cam0": home.place, "living": home.place, "bedroom": new[0].place}
    latest = places.latest(tmp_path, home.place)
    assert latest is not None and set(latest.cameras) == {"cam0", "living"}
    assert "bedroom" not in latest.signals["camera"] and latest.frame == home.frame
    assert cameras.refused_seats(tmp_path) == {}, "the count starts again"


def test_a_seat_that_holds_between_refusals_starts_the_count_again(tmp_path, joined):
    home, _ = joined
    bedroom_off(tmp_path, T + DAY)
    bedroom_off(tmp_path, T + 2 * DAY)
    cameras.write_places(tmp_path, tonight(tmp_path, np.eye(4), np.eye(4)), tmp_path, T + 3 * DAY)  # it seats
    bedroom_off(tmp_path, T + 4 * DAY)
    bedroom_off(tmp_path, T + 5 * DAY)
    assert places.place_ids(tmp_path) == [home.place]
    assert places.camera_places(tmp_path)["bedroom"] == home.place
    assert cameras.refused_seats(tmp_path)["bedroom"]["nights"] == 2


# A camera moved or uncovered after midday (#2997): its frame since the change, the same night


def view(across: bool, light: float = 1.0):
    """A lit BGR frame of one view: stripes down it, or across it for another view; `light` dims it."""
    out = np.zeros((90, 160, 3), np.float64)
    out[...] = (60, 120, 200)
    if across:
        out[::6] *= 0.4
    else:
        out[:, ::6] *= 0.4
    return np.clip(out * light, 0, 255).astype(np.uint8)


NIGHT_RUN = dt.datetime(2030, 1, 2, 1, 0, tzinfo=UTC).timestamp()
MOVED_AT = dt.datetime(2030, 1, 1, 16, 5, tzinfo=UTC).timestamp()


def moved_at_four(cam, t):
    return view(across=t >= MOVED_AT)


def test_a_camera_whose_view_changed_after_midday_gets_a_frame_since_the_change(tmp_path, vision):
    picked = {}
    got = cameras.frames(tmp_path, ["kitchen"], tmp_path, NIGHT_RUN, moved_at_four, UTC, picked=picked)
    assert list(got) == ["kitchen"] and picked["kitchen"] >= MOVED_AT


def test_a_view_that_only_dims_toward_evening_keeps_its_midday_frame(tmp_path, vision):
    def dimming(cam, t):
        return view(across=False, light=1.0 if t < MOVED_AT else 0.75)

    picked = {}
    cameras.frames(tmp_path, ["kitchen"], tmp_path, NIGHT_RUN, dimming, UTC, picked=picked)
    local = dt.datetime.fromtimestamp(picked["kitchen"], UTC)
    assert dt.time(10, 30) <= local.time() < dt.time(14, 30)


def test_views_compare_by_what_they_show_not_how_bright():
    assert cameras.view_match(view(False), view(False, light=0.6)) == pytest.approx(1.0, abs=1e-3)
    assert cameras.view_match(view(False), view(True)) < cameras.VIEW_SAME


def test_a_floorless_lone_camera_is_tried_again_with_its_view_after_the_frame_it_was_tried_with(
    tmp_path, vision, monkeypatch
):
    midday = dt.datetime(2030, 1, 1, 12, 0, tzinfo=UTC).timestamp()
    asked = {}

    def mast3r(runtime, images, out, gpu, *, lone_matches=None, **_):
        asked.update(images)
        return {"groups": [{"cameras": ["kitchen"], "cloud": "alone-kitchen.npz"}], "floorless": []}

    monkeypatch.setattr(cameras, "run_mast3r", mast3r)
    monkeypatch.setattr(cameras, "run_geocalib", lambda runtime, images, out, nice=0: {})
    first = {"groups": [{"cameras": ["cam0"], "cloud": "alone-cam0.npz"}], "floorless": ["kitchen"], "matches": []}
    merged, _ = cameras.retry_floorless(
        tmp_path, first, tmp_path, NIGHT_RUN, {"kitchen": midday}, None, read=moved_at_four, zone=UTC
    )
    assert [g["cloud"] for g in merged["groups"]] == ["alone-cam0.npz", "late/alone-kitchen.npz"]
    assert merged["groups"][1]["folder"] == "late" and merged["floorless"] == []
    import cv2

    tried = cv2.imread(str(asked["kitchen"]))
    assert tried is not None and asked["kitchen"].parent == tmp_path / "late"
    assert cameras.view_match(tried, view(True)) > 0.99, "the frame it was tried with again shows its view now"


def test_with_no_floorless_camera_mast3r_isnt_run_again(tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("MASt3R ran again")

    monkeypatch.setattr(cameras, "run_mast3r", refuse)
    first = {"groups": [], "floorless": []}
    assert cameras.retry_floorless(tmp_path, first, tmp_path, NIGHT_RUN, {}, None) == (first, {})


def test_a_camera_that_stops_matching_its_rooms_others_and_doesnt_seat_splits_off_that_night(tmp_path):
    # living and cam0 were one group; tonight living comes back alone, its snapshot off where its pose was (#3146)
    (both,) = cameras.write_places(
        tmp_path, {"groups": [group_in(tmp_path, 0, ["cam0", "living"], room(1), LIVING, np.eye(4))]}, tmp_path, T
    )
    fuse.refine_all(tmp_path, now=T + 1)
    off = room(1) + np.array([6.0, 0.0, 0.0], np.float32)
    tonight = {
        "groups": [
            group_in(tmp_path, 0, ["cam0"], room(1), {"cam0": LIVING["cam0"]}, np.eye(4)),
            group_in(tmp_path, 1, ["living"], off, {"living": LIVING["living"]}, np.eye(4)),
        ]
    }
    revs = cameras.write_places(tmp_path, tonight, tmp_path, T + DAY)
    new = [r for r in revs if r.place != both.place]
    assert len(new) == 1 and set(new[0].cameras) == {"living"}
    assert places.camera_places(tmp_path) == {"cam0": both.place, "living": new[0].place}
    latest = places.latest(tmp_path, both.place)
    assert latest is not None and set(latest.cameras) == {"cam0"}


def test_a_run_that_tries_a_camera_and_doesnt_refuse_it_breaks_its_count(tmp_path, joined):
    bedroom_off(tmp_path, T + DAY)
    bedroom_off(tmp_path, T + 2 * DAY)
    # the whole home comes back as one group: the bedroom is tried and nothing refuses it
    whole = group_in(tmp_path, 0, ["cam0", "living", "bedroom"], room(1), {**LIVING, **BEDROOM}, np.eye(4))
    cameras.write_places(tmp_path, {"groups": [whole]}, tmp_path, T + 3 * DAY)
    assert "bedroom" not in cameras.refused_seats(tmp_path)


def test_a_camera_with_no_frame_tonight_keeps_its_count(tmp_path, joined):
    bedroom_off(tmp_path, T + DAY)
    bedroom_off(tmp_path, T + 2 * DAY)
    living_only = {"groups": [group_in(tmp_path, 0, ["cam0", "living"], room(1), LIVING, np.eye(4))]}
    cameras.write_places(tmp_path, living_only, tmp_path, T + 3 * DAY, absent=["bedroom"])
    assert cameras.refused_seats(tmp_path)["bedroom"]["nights"] == 2


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("{not json", id="unreadable"),
        pytest.param('{"bedroom": {"place": "p", "nights": "two", "at": 1}}', id="nights-not-a-number"),
        pytest.param('["bedroom"]', id="not-an-object"),
    ],
)
def test_a_malformed_seats_file_starts_the_counts_again(tmp_path, text):
    places.root(tmp_path).mkdir(parents=True)
    (places.root(tmp_path) / cameras.SEATS_FILE).write_text(text)
    assert cameras.refused_seats(tmp_path) == {}


def test_a_camera_whose_old_partner_had_no_frame_tonight_doesnt_split_off_on_one_refusal(tmp_path):
    (both,) = cameras.write_places(
        tmp_path, {"groups": [group_in(tmp_path, 0, ["cam0", "living"], room(1), LIVING, np.eye(4))]}, tmp_path, T
    )
    fuse.refine_all(tmp_path, now=T + 1)
    # living comes back alone and off, but cam0 had no frame: it couldn't have matched cam0 tonight
    off = room(1) + np.array([6.0, 0.0, 0.0], np.float32)
    alone = {"groups": [group_in(tmp_path, 1, ["living"], off, {"living": LIVING["living"]}, np.eye(4))]}
    cameras.write_places(tmp_path, alone, tmp_path, T + DAY, absent=["cam0"])
    assert places.place_ids(tmp_path) == [both.place]
    assert places.camera_places(tmp_path) == {"cam0": both.place, "living": both.place}
    assert cameras.refused_seats(tmp_path)["living"]["nights"] == 1


def held_by_another(state):
    """Whether the places lock (`places.writing`) is held: another open of it can't take it."""
    import fcntl

    from liberational import locking

    path = locking.lock_path(places.root(state))
    if not path.exists():
        return False
    with path.open("a") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return False


def test_the_nightly_run_writes_and_fuses_places_under_the_lock_a_join_holds(tmp_path, monkeypatch):
    held = {}
    real_get = cfg.get
    monkeypatch.setattr(cfg, "get", lambda key, *a: False if key == "spatial.splats_enabled" else real_get(key, *a))
    monkeypatch.setattr(cameras, "fixed_cameras", lambda path: ["cam0"])
    monkeypatch.setattr(cameras, "indoor_cameras", lambda cams, outdoor: cams)
    monkeypatch.setattr(cameras.footprints, "record", lambda state, cams, now: 0)
    monkeypatch.setattr(cameras, "detector", lambda: None)
    monkeypatch.setattr(cameras, "frames", lambda *a, **kw: {"cam0": tmp_path / "cam0.png"})
    monkeypatch.setattr(cameras.nvml, "gpus", list)
    monkeypatch.setattr(cameras, "run_mast3r", lambda *a, **kw: {"groups": [], "floorless": []})
    monkeypatch.setattr(cameras, "run_geocalib", lambda *a, **kw: {})

    def write_places(state, *a, **kw):
        held["write"] = held_by_another(state)
        return []

    def refine_all(state, *a, **kw):
        held["fuse"] = held_by_another(state)
        return []

    monkeypatch.setattr(cameras, "write_places", write_places)
    monkeypatch.setattr(fuse, "refine_all", refine_all)
    cameras.run(tmp_path, now=T)
    assert held == {"write": True, "fuse": True}
    assert not held_by_another(tmp_path), "released after"


@pytest.mark.parametrize("registration_fails", [False, True], ids=["after-the-places", "though-registration-failed"])
def test_the_nightly_run_places_the_days_positions_last(tmp_path, monkeypatch, registration_fails):
    from liberational.spatial import positions

    did = []
    real_get = cfg.get
    monkeypatch.setattr(cfg, "get", lambda key, *a: False if key == "spatial.splats_enabled" else real_get(key, *a))
    monkeypatch.setattr(cameras, "fixed_cameras", lambda path: ["cam0"])
    monkeypatch.setattr(cameras, "indoor_cameras", lambda cams, outdoor: cams)
    monkeypatch.setattr(cameras.footprints, "record", lambda state, cams, now: did.append("footprints") or 0)
    monkeypatch.setattr(cameras, "detector", lambda: None)
    monkeypatch.setattr(cameras, "frames", lambda *a, **kw: {"cam0": tmp_path / "cam0.png"})
    monkeypatch.setattr(cameras.nvml, "gpus", list)

    def run_mast3r(*a, **kw):
        if registration_fails:
            raise RuntimeError("out of memory")
        return {"groups": [], "floorless": []}

    monkeypatch.setattr(cameras, "run_mast3r", run_mast3r)
    monkeypatch.setattr(cameras, "run_geocalib", lambda *a, **kw: {})
    monkeypatch.setattr(cameras, "write_places", lambda *a, **kw: did.append("places") or [])
    monkeypatch.setattr(fuse, "refine_all", lambda *a, **kw: [])
    monkeypatch.setattr(positions, "advance", lambda state, upto: did.append(("positions", upto)) or {"records": 3})
    placed = ("positions", T - footprints.FLUSH_S)  # up to where the footprints are recorded
    if registration_fails:
        with pytest.raises(RuntimeError, match="out of memory"):
            cameras.run(tmp_path, now=T)
        assert did == ["footprints", placed]
    else:
        assert cameras.run(tmp_path, now=T)["positions"] == {"records": 3}
        assert did == ["footprints", "places", placed]
