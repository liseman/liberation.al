"""A splat of each fixed-camera place (spatial/splats.py): frames, start points, the depth prior, and the job."""

import datetime as dt
import json
import sys
import textwrap
import zipfile
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import pytest

from liberational import config as cfg
from liberational.memory import places
from liberational.spatial import cameras, ply, splats
from test_spatial_sog import gaussians

T = 1_800_000_000.0
UTC = ZoneInfo("UTC")
LEVEL = tuple(float(i == j) for i in range(4) for j in range(4))  # at the origin, looking along +z
CAM = places.CameraPose(LEVEL, focal=100.0, width=200, height=100)


def test_a_point_counts_as_seen_only_in_front_of_a_camera_and_inside_its_frame():
    points = np.array([[0, 0, 1], [0.9, 0, 1], [1.1, 0, 1], [0, 0, -1], [0, 0.6, 1], [0, 0, 0.01]], float)
    # u = 100 x / z + 100 and v = 100 y / z + 50: inside for x in (-1, 1) and y in (-0.5, 0.5) at z = 1
    assert splats.in_view(points, {"cam0": CAM}).tolist() == [True, True, False, False, False, False]


def test_a_second_camera_adds_what_it_sees():
    behind = places.CameraPose(
        (-1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, -1.0, 0, 0, 0, 0, 1.0), focal=100.0, width=200, height=100
    )  # turned to look along -z
    points = np.array([[0, 0, 1], [0, 0, -1]], float)
    assert splats.in_view(points, {"a": CAM, "b": behind}).tolist() == [True, True]


@pytest.mark.parametrize(
    ("n", "most", "want"),
    [(3, 5, [0, 1, 2]), (10, 4, [0, 3, 6, 9]), (5, 1, [2])],
    ids=["fewer", "spread", "one"],
)
def test_spread_keeps_an_even_subset_with_its_ends(n, most, want):
    assert splats.spread(list(range(n)), most) == want


@pytest.mark.parametrize(("w", "h"), [(1920, 1080), (640, 480), (1080, 1920)], ids=["16:9", "4:3", "portrait"])
def test_a_frame_is_scaled_to_cover_the_registered_size_and_centre_cropped(w, h):
    frame = np.zeros((h, w, 3), np.uint8)
    frame[h // 2 - 20 : h // 2 + 20, w // 2 - 20 : w // 2 + 20] = 255  # a white square at the centre
    out = splats.fit(frame, 512, 288)
    assert out.shape == (288, 512, 3)
    assert out[144, 256].tolist() == [255, 255, 255] and out[0, 0].tolist() == [0, 0, 0]


def lit(seed=0):
    """A lit, coloured frame (cameras.daylight passes it)."""
    rng = np.random.default_rng(seed)
    frame = np.zeros((100, 200, 3), np.uint8)
    frame[..., 0], frame[..., 1], frame[..., 2] = 60, 120, 200
    return frame + rng.integers(0, 5, frame.shape, dtype=np.uint8)


def test_day_frames_leave_out_dark_frames_and_cameras_without_any():
    night = np.full((100, 200, 3), 20, np.uint8)
    now = dt.datetime(2030, 1, 2, 12, 0, tzinfo=UTC).timestamp()

    def read(cam, t):
        if cam == "dark":
            return night
        return lit() if dt.datetime.fromtimestamp(t, UTC).hour < 12 else None

    got = splats.day_frames(["cam0", "dark"], now, read, UTC)
    assert list(got) == ["cam0"] and 1 < len(got["cam0"]) <= splats.FRAMES_PER_CAMERA


# --- a place's splat, end to end with a stand-in trainer

CLOUD = np.array([[0, 0, 2], [0.2, 0.1, 3], [0, 0, -2]], np.float32)  # the last one is behind the camera


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    """gsplat's runtime as this Python, with a stand-in for the trainer that writes a two-Gaussian 3DGS PLY and records
    the job it got."""
    rt = tmp_path / "gsplat"
    (rt / ".venv" / "bin").mkdir(parents=True)
    (rt / ".venv" / "bin" / "python").symlink_to(sys.executable)
    fake = tmp_path / "runner.py"
    fake.write_text(
        textwrap.dedent(
            """
            import json, os, sys
            job = json.load(open(sys.argv[1]))
            if os.environ.get("FAIL_SPLAT"):
                sys.exit("CUDA out of memory")
            names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "f_rest_0", "f_rest_1", "f_rest_2", "f_rest_3",
                     "f_rest_4", "f_rest_5", "f_rest_6", "f_rest_7", "f_rest_8", "opacity", "scale_0", "scale_1",
                     "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
            import struct
            head = "ply\\nformat binary_little_endian 1.0\\nelement vertex 2\\n"
            head += "".join(f"property float {k}\\n" for k in names) + "end_header\\n"
            body = struct.pack("<46f", *([0.5] * 46))
            open(os.path.join(job["out"], "splat.ply"), "wb").write(head.encode() + body)
            json.dump({"gaussians": 2, "seconds": 1.0, "train_psnr": 20.0, "job": job},
                      open(os.path.join(job["out"], "result.json"), "w"))
            """
        )
    )
    monkeypatch.setattr(splats, "RUNNER", fake)
    return rt


@pytest.fixture
def placed(tmp_path):
    rev = places.revise(
        tmp_path, None, producer="mapper v1", map_bytes=ply.dumps(CLOUD, np.full((3, 3), 90)), map_format="ply",
        cameras={"cam0": CAM}, now=T,
    )  # fmt: skip
    return rev


def always_lit(cam, t):
    return lit()


def test_a_place_gets_one_splat_for_the_revision_the_view_shows(tmp_path, runtime, placed, monkeypatch):
    monkeypatch.setattr("liberational.memory.history_qa.local_zone", lambda: UTC)
    did = splats.splat_place(tmp_path, placed.place, T, always_lit, runtime, None)
    assert did is not None
    assert (did["revision"], did["start_points"], did["gaussians"]) == (placed.id, 2, 2)
    assert places.has_splat(tmp_path, placed)
    master = places.splat_path(tmp_path, placed, "gs.ply").read_bytes()
    rows = ply.vertices(master)
    assert rows is not None and len(rows) == 2
    with zipfile.ZipFile(places.splat_path(tmp_path, placed, "sog")) as web:  # the view's copy (spatial/sog.py)
        assert json.loads(web.read("meta.json"))["count"] == 2
    # Written once: a second pass finds it there and does nothing.
    assert splats.splat_place(tmp_path, placed.place, T, always_lit, runtime, None) is None


def wall(depth, step=0.02):
    """A wall `depth` in front of CAM, filling its view: points every `step` across it."""
    xs, ys = np.meshgrid(np.arange(-2 * depth, 2 * depth, step), np.arange(-depth, depth, step))
    return np.c_[xs.ravel(), ys.ravel(), np.full(xs.size, depth)]


def test_a_cameras_depth_is_scaled_to_the_map_points_it_sees():
    model = np.full((100, 200), 1.6)  # the model has the wall at 1.6 m; the map has it at 2 m
    assert splats.map_scale(model, wall(2.0), CAM) == pytest.approx(1.25)


def test_the_scale_ignores_what_the_room_gained_since_the_map_and_the_points_behind_a_wall():
    model = np.full((100, 200), 1.6)
    model[:, :60] = 0.4  # a box put down close to the camera after the map was made
    behind = wall(6.0, step=0.3)  # another room's points, seen through the wall's gaps: nearer ones win a cell
    assert splats.map_scale(model, np.r_[wall(2.0), behind], CAM) == pytest.approx(1.25)


def test_a_camera_that_sees_too_few_map_points_keeps_the_models_metres():
    assert splats.map_scale(np.full((100, 200), 1.6), wall(2.0)[:50], CAM) is None


@pytest.fixture
def depth_model(tmp_path, runtime, monkeypatch):
    """Depth Anything 3's code and weights where splats looks for them, with the model itself faked at its
    subprocess boundary: frame k of a camera is given depth 1.5 + 0.1 k everywhere, and the requests are kept."""
    (runtime / splats.DEPTH_SOURCE).mkdir(parents=True)
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"")
    monkeypatch.setattr(splats, "depth_checkpoint", lambda: weights)
    asked = []

    def run(rt, job, gpu):
        request = json.loads(job.read_text())
        asked.append(request)
        for k, view in enumerate(request["views"]):
            np.save(view["out"], np.full((100, 200), 1.5 + 0.1 * k, np.float32))

    monkeypatch.setattr(splats, "run_depth", run)
    return asked


@pytest.fixture
def jobs(monkeypatch):
    """The jobs the trainer is given, read while their folder is still there."""
    seen = []
    real = splats.run_runner

    def run(rt, job, gpu, timeout_s=splats.RUN_TIMEOUT_S):
        seen.append(json.loads(job.read_text()))
        depth = seen[-1]["views"][0].get("depth")
        seen[-1]["depth_image"] = None if depth is None else cv2.imread(depth, cv2.IMREAD_UNCHANGED)
        return real(rt, job, gpu, timeout_s)

    monkeypatch.setattr(splats, "run_runner", run)
    return seen


def test_a_fixed_cameras_views_train_against_its_median_depth_scaled_to_the_map(
    tmp_path, runtime, depth_model, jobs, monkeypatch
):
    monkeypatch.setattr("liberational.memory.history_qa.local_zone", lambda: UTC)
    rev = places.revise(
        tmp_path, None, producer="mapper v1", map_bytes=ply.dumps(wall(2.0), None), map_format="ply",
        cameras={"cam0": CAM}, now=T,
    )  # fmt: skip
    did = splats.splat_place(tmp_path, rev.place, T, always_lit, runtime, None)
    (job,) = jobs
    frames = len(job["views"])
    assert frames == splats.FRAMES_PER_CAMERA and len(depth_model[0]["views"]) == frames
    assert depth_model[0]["views"][0]["focal"] == CAM.focal
    # The frames' depths are 1.5, 1.6, ... 2.6: their median, 2.05, is what the camera's one depth image holds,
    # scaled so the wall is where the map has it, 2 m, in thousandths.
    assert did is not None and did["depth_scales"] == {"cam0": pytest.approx(2.0 / 2.05, abs=1e-3)}
    assert {v["depth"] for v in job["views"]} == {job["views"][0]["depth"]} and job["depth_unit"] == 0.001
    assert job["depth_image"].dtype == np.uint16 and job["depth_image"].shape == (100, 200)
    assert np.unique(job["depth_image"]).tolist() == [2000]


def test_a_cameras_covered_pixels_have_no_depth(tmp_path, runtime, depth_model):
    rev = places.revise(
        tmp_path, None, producer="p", map_bytes=ply.dumps(wall(2.0), None), map_format="ply", cameras={"cam0": CAM},
        now=T,
    )  # fmt: skip
    folder = tmp_path / "job"
    folder.mkdir()
    job = splats.write_job(folder, rev, {"cam0": [lit(), lit(1)]}, (np.zeros((1, 3)), np.zeros((1, 3))))
    covered = np.zeros((100, 200), np.uint8)
    covered[:10] = 255  # an overlay along the top
    cv2.imwrite(str(folder / "overlay.png"), covered)
    spec = json.loads(job.read_text())
    for view in spec["views"]:
        view["mask"] = str(folder / "overlay.png")
    job.write_text(json.dumps(spec))
    splats.depth_priors(job, rev, wall(2.0), runtime, None)
    depth = cv2.imread(json.loads(job.read_text())["views"][0]["depth"], cv2.IMREAD_UNCHANGED)
    assert depth is not None and not depth[:10].any() and depth[10:].all()


def test_without_the_depth_models_code_a_splat_trains_on_its_frames_alone(
    tmp_path, runtime, placed, jobs, monkeypatch, caplog
):
    monkeypatch.setattr("liberational.memory.history_qa.local_zone", lambda: UTC)
    did = splats.splat_place(tmp_path, placed.place, T, always_lit, runtime, None)
    assert did is not None and did["depth_scales"] is None and places.has_splat(tmp_path, placed)
    assert all("depth" not in v for v in jobs[0]["views"]) and "depth_unit" not in jobs[0]
    assert "without the depth prior" in caplog.text and "mise run gsplat" in caplog.text


def test_a_failed_depth_model_leaves_the_splat_to_train_without_it(
    tmp_path, runtime, placed, depth_model, jobs, monkeypatch, caplog
):
    monkeypatch.setattr("liberational.memory.history_qa.local_zone", lambda: UTC)

    def fail(rt, job, gpu):
        raise RuntimeError("the depth model failed (1): CUDA out of memory")

    monkeypatch.setattr(splats, "run_depth", fail)
    did = splats.splat_place(tmp_path, placed.place, T, always_lit, runtime, None)
    assert did is not None and did["depth_scales"] is None
    assert all("depth" not in v for v in jobs[0]["views"]) and "CUDA out of memory" in caplog.text


def test_a_place_with_no_lit_frames_gets_no_splat(tmp_path, runtime, placed, monkeypatch):
    monkeypatch.setattr("liberational.memory.history_qa.local_zone", lambda: UTC)
    assert splats.splat_place(tmp_path, placed.place, T, lambda cam, t: None, runtime, None) is None
    assert not places.has_splat(tmp_path, placed)


def test_a_day_with_too_few_lit_frames_makes_no_splat_yet(tmp_path, runtime, placed, monkeypatch):
    monkeypatch.setattr("liberational.memory.history_qa.local_zone", lambda: UTC)
    times = sorted(cameras.sample_times(T, UTC))
    few = set(times[: splats.MIN_FRAMES_PER_CAMERA - 1])
    assert (
        splats.splat_place(tmp_path, placed.place, T, lambda cam, t: lit() if t in few else None, runtime, None) is None
    )
    assert not places.has_splat(tmp_path, placed)
    enough = set(times[: splats.MIN_FRAMES_PER_CAMERA])
    did = splats.splat_place(tmp_path, placed.place, T, lambda cam, t: lit() if t in enough else None, runtime, None)
    assert did is not None and did["frames"] == {"cam0": splats.MIN_FRAMES_PER_CAMERA}


def test_the_newest_map_without_cameras_gets_no_splat(tmp_path, placed):
    bare = places.revise(
        tmp_path, placed.place, producer="mapper v1", map_bytes=ply.dumps(CLOUD), map_format="ply", cameras={},
        now=T + 10,
    )  # fmt: skip
    assert splats.target(tmp_path, bare.place) is None


def two_gaussians():
    """A 3DGS PLY with two Gaussians at the origin, SH degree 1."""
    rows = gaussians(
        np.zeros((2, 3)), np.zeros((2, 3)), np.zeros(2), np.zeros((2, 3)), np.tile([1.0, 0, 0, 0], (2, 1)),
        np.zeros((2, 3, 3)),
    )  # fmt: skip
    head = "ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
    head += "".join(f"property float {k}\n" for k in rows.dtype.names) + "end_header\n"
    return head.encode() + rows.tobytes()


def test_a_splat_kept_from_before_sog_gets_its_copy_from_its_ply(tmp_path, placed):
    # What an install has before #76: the PLY and an SPZ beside it, and no SOG, so the view has nothing to load.
    master = places.splat_path(tmp_path, placed, "gs.ply")
    master.write_bytes(two_gaussians())
    old = master.with_name(f"{placed.id}.spz")
    old.write_bytes(b"spz")
    assert not places.has_splat(tmp_path, placed)
    assert splats.recode(tmp_path, placed.place) == 1
    with zipfile.ZipFile(places.splat_path(tmp_path, placed, "sog")) as web:
        assert json.loads(web.read("meta.json"))["count"] == 2
    assert places.has_splat(tmp_path, placed) and not old.exists()
    assert splats.recode(tmp_path, placed.place) == 0, "once"


def test_a_kept_ply_that_doesnt_read_is_left_alone(tmp_path, placed, caplog):
    places.splat_path(tmp_path, placed, "gs.ply").write_bytes(b"not a ply")
    assert splats.recode(tmp_path, placed.place) == 0
    assert not places.has_splat(tmp_path, placed) and "doesn't read" in caplog.text


def test_a_failed_training_is_logged_and_the_next_place_still_runs(tmp_path, runtime, placed, monkeypatch):
    places.revise(
        tmp_path, None, producer="mapper v1", map_bytes=ply.dumps(CLOUD), map_format="ply",
        cameras={"cam1": CAM}, now=T,
    )  # fmt: skip
    tried = []

    def run(rt, job, gpu, timeout_s=0.0):
        tried.append(job)
        if len(tried) == 1:
            raise RuntimeError("the splat trainer failed (1): CUDA out of memory")
        (job.parent / "splat.ply").write_bytes(two_gaussians())
        return {"gaussians": 2}

    monkeypatch.setattr(splats, "run_runner", run)
    monkeypatch.setattr(splats.cfg, "path", lambda key: runtime)
    monkeypatch.setattr(splats.cfg, "get", lambda key, *a: -1)
    monkeypatch.setattr(splats.nvml, "gpus", list)
    monkeypatch.setattr("liberational.memory.history_qa.local_zone", lambda: UTC)
    done = splats.splat_all(tmp_path, T, read=always_lit)
    assert len(tried) == 2 and len(done) == 1
    assert done[0]["place"] == places.place_ids(tmp_path)[1]
    assert not places.has_splat(tmp_path, placed)  # the first place, whose training failed


def test_without_the_runtime_nothing_is_read(tmp_path, placed, monkeypatch):
    monkeypatch.setattr(splats.cfg, "path", lambda key: tmp_path / "nowhere")

    def read(cam, t):
        raise AssertionError("no frame should be read")

    assert splats.splat_all(tmp_path, T, read=read) == []


def test_a_failed_trainer_says_why(tmp_path, runtime, monkeypatch):
    monkeypatch.setenv("FAIL_SPLAT", "1")
    job = tmp_path / "job.json"
    job.write_text('{"out": "%s"}' % tmp_path)
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        splats.run_runner(runtime, job, None)


def test_many_views_are_only_drawn_at_their_poses_when_asked(tmp_path, runtime, monkeypatch):
    monkeypatch.setattr(splats.nvml, "gpus", list)
    views = [{"image": str(tmp_path / "f000.jpg"), "pose": np.eye(4).ravel().tolist(), "focal": 100.0}]
    with cfg.override({"spatial": {"splat_runtime": str(runtime), "gpu": -1}}):
        plain, _ = splats.train_views(views, CLOUD, None)
        drawn, _ = splats.train_views(views, CLOUD, None, render=tmp_path / "exact")
    assert "render" not in plain["job"], "a splat draws nothing unless asked"
    assert drawn["job"]["render"] == str(tmp_path / "exact")
    assert drawn["job"]["holdout"] == [] and drawn["job"]["pose_opt"] is True


def test_posed_views_carry_each_frames_own_intrinsics_and_move_with_the_map(tmp_path):
    k = np.array([[120.0, 0.0, 31.0], [0.0, 130.0, 17.0], [0.0, 0.0, 1.0]])
    pose = np.eye(4)
    pose[:3, 3] = [1.0, 0.0, 0.0]
    quarter = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # x to y
    depth = tmp_path / "d.png"
    (plain,) = splats.posed_views([tmp_path / "f.jpg"], pose[None], [(k, (64, 36))])
    (moved,) = splats.posed_views(
        [tmp_path / "f.jpg"], pose[None], [(k, (64, 36))], (quarter, np.array([0.0, 0.0, 5.0]), 2.0), [depth]
    )
    assert (plain["fx"], plain["fy"], plain["cx"], plain["cy"], plain["focal"]) == (120.0, 130.0, 31.0, 17.0, 125.0)
    assert (plain["width"], plain["height"]) == (64, 36) and "depth" not in plain
    assert np.array(plain["pose"]).reshape(4, 4) == pytest.approx(pose)
    got = np.array(moved["pose"]).reshape(4, 4)
    assert got[:3, 3] == pytest.approx([0.0, 2.0, 5.0])  # turned a quarter, doubled, then lifted
    assert got[:3, :3] == pytest.approx(quarter)  # a camera's axes turn and never scale
    assert moved["depth"] == str(depth)
