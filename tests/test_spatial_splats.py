"""A splat of each fixed-camera place (spatial/splats.py): frames, start points, the SPZ copy, and the job."""

import datetime as dt
import gzip
import json
import math
import sys
import textwrap
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from liberational import config as cfg
from liberational.memory import places
from liberational.spatial import cameras, ply, splats

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


# --- SPZ, checked by a decoder written from Spark's (rust/spark-lib/src/spz.rs, v2.4.0) and Niantic's unpacking


def decode(data: bytes) -> dict:
    raw = gzip.decompress(data)
    magic, version, n = np.frombuffer(raw[:12], "<u4")
    degree, frac = raw[12], raw[13]
    assert (magic, version) == (0x5053474E, 3)
    coeffs = {0: 0, 1: 3, 2: 8, 3: 15}[degree]
    at = 16

    def take(size):
        nonlocal at
        chunk = np.frombuffer(raw[at : at + size], np.uint8)
        at += size
        return chunk

    pos = take(n * 9).reshape(n, 3, 3).astype(np.int32)
    fixed = pos[..., 0] | (pos[..., 1] << 8) | (pos[..., 2] << 16)
    fixed = np.where(fixed & 0x800000, fixed - (1 << 24), fixed)
    alpha = take(n).astype(float) / 255
    color = (take(n * 3).reshape(n, 3) / 255 - 0.5) / 0.15
    scale = take(n * 3).reshape(n, 3) / 16 - 10
    quats = []
    for b in take(n * 4).reshape(n, 4):
        comp = int(b[0]) | int(b[1]) << 8 | int(b[2]) << 16 | int(b[3]) << 24
        largest, q, rest = comp >> 30, [0.0] * 4, comp
        for j in (3, 2, 1, 0):
            if j != largest:
                v = math.sqrt(0.5) * (rest & 511) / 511
                q[j] = -v if (rest >> 9) & 1 else v
                rest >>= 10
        q[largest] = math.sqrt(max(0.0, 1 - sum(x * x for x in q)))
        quats.append(q)
    sh = (take(n * coeffs * 3).reshape(n, coeffs, 3).astype(float) - 128) / 128
    assert at == len(raw)
    return {"xyz": fixed / (1 << frac), "alpha": alpha, "dc": color, "scale": scale, "xyzw": np.array(quats), "sh": sh}


def gaussians(xyz, dc, opacity, scales, wxyz, rest):
    """A 3DGS PLY's vertices: `rest` is (N, 3 channels, coefficients), as the PLY lays it out channel by channel."""
    n, coeffs = len(xyz), rest.shape[2]
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", *(f"f_rest_{i}" for i in range(3 * coeffs))]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    cols = np.concatenate([xyz, dc, rest.reshape(n, -1), opacity[:, None], scales, wxyz], axis=1)
    return np.rec.fromarrays(cols.T.astype("<f4"), names=names)


def test_one_gaussian_encodes_to_the_values_worked_out_by_hand():
    rows = gaussians(
        xyz=np.array([[1.0, 2.0, -0.5]]),
        dc=np.zeros((1, 3)),
        opacity=np.array([0.0]),
        scales=np.full((1, 3), -4.0),
        wxyz=np.array([[1.0, 0.0, 0.0, 0.0]]),
        rest=np.zeros((1, 3, 0)),
    )
    got = decode(splats.spz(rows))
    assert got["xyz"].tolist() == [[1.0, -2.0, 0.5]]  # y and z negated: SPZ's right-up-back frame
    assert got["alpha"][0] == pytest.approx(128 / 255)  # sigmoid(0) = 0.5
    assert got["dc"][0] == pytest.approx([(128 / 255 - 0.5) / 0.15] * 3)
    assert got["scale"][0].tolist() == [-4.0, -4.0, -4.0]
    assert got["xyzw"][0] == pytest.approx([0, 0, 0, 1])


def unit(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@st.composite
def clouds(draw):
    n = draw(st.integers(1, 12))
    degree = draw(st.sampled_from([0, 1, 2, 3]))
    coeffs = (degree + 1) ** 2 - 1
    seed = draw(st.integers(0, 2**32 - 1))
    rng = np.random.default_rng(seed)
    quat = unit(rng.normal(size=(n, 4)) + 1e-3)
    return gaussians(
        xyz=rng.uniform(-50, 50, (n, 3)),
        dc=rng.uniform(-3, 3, (n, 3)),
        opacity=rng.uniform(-5, 5, n),
        scales=rng.uniform(-9, 5, (n, 3)),
        wxyz=quat,
        rest=rng.uniform(-0.9, 0.9, (n, 3, coeffs)),
    ), degree


FLIP_SH = np.array([-1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1])  # Niantic's flipSh, x = 1, y = z = -1


@settings(max_examples=60, deadline=None)
@given(clouds())
def test_every_gaussian_decodes_back_in_spzs_frame_within_its_quantization(cloud):
    rows, degree = cloud
    got = decode(splats.spz(rows))
    xyz = np.c_[rows["x"], -rows["y"], -rows["z"]]  # y and z negated
    assert np.abs(got["xyz"] - xyz).max() <= 0.5 / 4096 + 1e-6
    assert np.abs(got["alpha"] - 1 / (1 + np.exp(-rows["opacity"].astype(float)))).max() <= 0.5 / 255 + 1e-6
    dc = np.c_[rows["f_dc_0"], rows["f_dc_1"], rows["f_dc_2"]]
    assert np.abs(got["dc"] - dc).max() <= 0.5 / 255 / 0.15 + 1e-5
    scale = np.c_[rows["scale_0"], rows["scale_1"], rows["scale_2"]]
    assert np.abs(got["scale"] - scale).max() <= 0.5 / 16 + 1e-6
    # The rotation is the PLY's (w, x, y, z) with x, y, and z negated by the frame change, up to sign.
    w, x, y, z = (rows[f"rot_{i}"].astype(float) for i in range(4))
    want = np.c_[x, -y, -z, w]
    assert np.abs(np.abs((got["xyzw"] * want).sum(axis=1)) - 1).max() < 0.01
    coeffs = (degree + 1) ** 2 - 1
    rest = np.array([[rows[f"f_rest_{i}"][g] for i in range(3 * coeffs)] for g in range(len(rows))])
    rest = rest.reshape(len(rows), 3, coeffs)
    want_sh = rest.transpose(0, 2, 1) * FLIP_SH[:coeffs, None]
    bucket = np.where(np.arange(coeffs) < 3, 8, 16)[None, :, None] / 128  # 5 bits for band 1, 4 for the rest
    assert (np.abs(got["sh"] - want_sh) <= bucket / 2 + 1 / 256 + 1e-6).all()


def test_an_sh_count_that_isnt_a_degree_is_refused():
    rows = gaussians(
        np.zeros((1, 3)), np.zeros((1, 3)), np.zeros(1), np.zeros((1, 3)), np.eye(1, 4), np.zeros((1, 3, 2))
    )
    with pytest.raises(ValueError, match="SH degree"):
        splats.spz(rows)


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
        tmp_path, None, producer="tier1 fuse v1", map_bytes=ply.dumps(CLOUD, np.full((3, 3), 90)), map_format="ply",
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
    assert len(decode(places.splat_path(tmp_path, placed, "spz").read_bytes())["xyz"]) == 2
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
        tmp_path, None, producer="tier1 fuse v1", map_bytes=ply.dumps(wall(2.0), None), map_format="ply",
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
    walk = places.revise(
        tmp_path, placed.place, producer="walkthrough", map_bytes=ply.dumps(CLOUD), map_format="ply", cameras={},
        now=T + 10,
    )  # fmt: skip
    assert splats.target(tmp_path, walk.place) is None


def two_gaussians():
    """A 3DGS PLY with two Gaussians at the origin, SH degree 1."""
    rows = gaussians(
        np.zeros((2, 3)), np.zeros((2, 3)), np.zeros(2), np.zeros((2, 3)), np.tile([1.0, 0, 0, 0], (2, 1)),
        np.zeros((2, 3, 3)),
    )  # fmt: skip
    head = "ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
    head += "".join(f"property float {k}\n" for k in rows.dtype.names) + "end_header\n"
    return head.encode() + rows.tobytes()


def test_a_failed_training_is_logged_and_the_next_place_still_runs(tmp_path, runtime, placed, monkeypatch):
    places.revise(
        tmp_path, None, producer="tier1 fuse v1", map_bytes=ply.dumps(CLOUD), map_format="ply",
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


def test_a_walk_looks_from_evenly_spread_frames_named_by_their_time_into_it():
    poses = np.repeat(np.eye(4)[None], 5, axis=0)
    poses[:, 0, 3] = np.arange(5)  # frame i stands at x = i
    frames = [(np.diag([250.0, 260.0, 1.0]), (240, 320))] * 5
    looks = splats.look_from(poses, frames, [100.0, 130.0, 165.0, 165.5, 230.0], most=3)
    # frames 0, 2, and 4: 0, 65, and 130 s into the walk
    assert list(looks) == ["walk-0m00s", "walk-1m05s", "walk-2m10s"]
    assert [c.pose[3] for c in looks.values()] == [0.0, 2.0, 4.0]
    assert (looks["walk-1m05s"].focal, looks["walk-1m05s"].width, looks["walk-1m05s"].height) == (255.0, 240, 320)


def test_two_looks_in_the_same_second_keep_both():
    frames = [(np.eye(3), (10, 10))] * 2
    looks = splats.look_from(np.repeat(np.eye(4)[None], 2, axis=0), frames, [0.0, 0.5])
    assert list(looks) == ["walk-0m00s", "walk-0m00sb"]


def _pose(rot: np.ndarray, centre) -> places.CameraPose:
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = rot, centre
    return places.CameraPose(tuple(m.ravel().tolist()), 100.0, 200, 100)


def test_the_move_a_straight_walks_cameras_show_carries_any_other_pose_the_same_way():
    # A walk along x only: its centres alone can't say how far it turned about x. Their axes can.
    turn = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])  # 90 degrees about x
    shift, scale = np.array([1.0, 2.0, 3.0]), 0.5
    heading = np.array([[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    before = {f"walk-{i}": _pose(heading, [i, 0.0, 0.0]) for i in range(3)}
    after = {
        n: _pose(turn @ heading, scale * turn @ np.asarray(p.pose).reshape(4, 4)[:3, 3] + shift)
        for n, p in before.items()
    }
    moved = splats.carried(before, {**after, "cam0": CAM})
    assert moved is not None
    rot, got_shift, got_scale = moved
    assert rot == pytest.approx(turn, abs=1e-9)
    assert (got_shift, got_scale) == (pytest.approx(shift), pytest.approx(scale))


def test_one_shared_camera_is_enough_and_none_is_no_move():
    one = {"walk-0m00s": _pose(np.eye(3), [1.0, 0.0, 0.0])}
    moved = splats.carried(one, {"walk-0m00s": _pose(np.eye(3), [3.0, 0.0, 0.0])})
    assert moved is not None
    assert (moved[1], moved[2]) == (pytest.approx([2.0, 0.0, 0.0]), 1.0)
    assert splats.carried(one, {"cam0": CAM}) is None


def test_a_walks_look_from_cameras_are_not_read_for_the_fixed_camera_splat(tmp_path, runtime):
    rev = places.revise(
        tmp_path, None, producer="tier1 fuse v1", map_bytes=ply.dumps(CLOUD, np.full((3, 3), 90)), map_format="ply",
        cameras={"walk-0m00s": CAM}, now=T,
    )  # fmt: skip
    asked = []

    def read(cam, t):
        asked.append(cam)
        return lit()

    assert splats.splat_place(tmp_path, rev.place, T, read, runtime, None) is None
    assert asked == []


def test_a_walks_failed_training_leaves_the_walk_without_a_splat_and_says_why(tmp_path, runtime, monkeypatch, caplog):
    rev = places.revise(
        tmp_path, None, producer="tier1 fuse v1", map_bytes=ply.dumps(CLOUD, np.full((3, 3), 90)), map_format="ply",
        cameras={"walk-0m00s": CAM}, now=T,
    )  # fmt: skip
    monkeypatch.setenv("FAIL_SPLAT", "1")
    k = np.array([[100.0, 0, 100], [0, 100, 50], [0, 0, 1]])
    with cfg.override({"spatial": {"splat_runtime": str(runtime), "splats_enabled": True, "gpu": 0}}):
        did = splats.splat_walk(
            tmp_path, rev.place, rev.id, [tmp_path / "f000.jpg"], np.eye(4)[None], [(k, (200, 100))], rev.cameras
        )
    assert did is None
    assert not places.has_splat(tmp_path, rev)
    assert "CUDA out of memory" in caplog.text
