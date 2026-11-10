"""Learning surroundings (#258): placing kept frames in a place's map and continuing its splat."""

import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from liberational import config as cfg
from liberational.memory import places
from liberational.spatial import cameras, lens, ply, splat_runner, splats, spots
from liberational.spatial import surroundings_frames as frames
from liberational.spatial import surroundings_map as sm


def shifted(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 3] = (x, y, z)
    return pose


def unit(angle_deg: float) -> np.ndarray:
    a = math.radians(angle_deg)
    return np.array([math.cos(a), math.sin(a)], dtype=np.float32)


# ---- which pairs to measure


def test_each_new_frame_is_paired_with_the_frames_it_is_most_alike():
    ids = ["old-near", "old-far", "new"]
    descriptors = np.stack([unit(10), unit(80), unit(0)])
    assert sm.pairs_for(["new"], ["old-near", "old-far"], ids, descriptors, most=1) == [("old-near", "new")]


def test_two_new_frames_are_paired_once_and_a_frame_too_unlike_is_not_paired():
    ids = ["a", "b", "elsewhere"]
    descriptors = np.stack([unit(0), unit(20), unit(90)])  # a and b are 0.94 alike; "elsewhere" is 0 and 0.34 alike
    assert sm.pairs_for(["a", "b", "elsewhere"], [], ids, descriptors) == [("b", "a"), ("elsewhere", "b")]


# ---- from pairs to metric poses


def test_a_pairs_translation_is_scaled_by_metric_depth_over_the_pairs_own():
    # The pair saw both frames 2 units deep; the depth model says 3 m and 12 m: 1.5 and 6 times, whose geometric mean is 3.
    assert sm.pair_scale([2.0, 2.0], 3.0, 12.0) == pytest.approx(3.0)
    assert sm.pair_scale([2.0, 0.0], 3.0, 12.0) is None
    assert sm.pair_scale([2.0, 2.0], 0.0, 12.0) is None


def pair(a: str, b: str, forward: np.ndarray | None, inliers: tuple[int, int] = (500, 400), depths=(1.0, 1.0)) -> dict:
    return {
        "a": a,
        "b": b,
        "matches": 900,
        "depths": list(depths),
        "forward": None if forward is None else [float(x) for x in forward.ravel()],
        "backward": None,
        "inliers": list(inliers),
    }


def test_edges_are_in_metres_and_leave_out_what_gives_no_pose():
    found = sm.edges(
        [
            pair("a", "b", shifted(x=1.0), depths=(1.0, 1.0)),
            pair("a", "a", np.eye(4)),  # a frame with itself: only its focal was wanted
            pair("a", "c", None),  # MASt3R was unsure everywhere
            pair("a", "d", shifted(x=1.0)),  # d has no depth yet
        ],
        {"a": 2.0, "b": 2.0, "c": 2.0},
    )
    [edge] = found
    assert (edge.a, edge.b, edge.inliers) == ("a", "b", 400)  # the weaker direction's matches
    assert edge.b_in_a[:3, 3] == pytest.approx([2.0, 0.0, 0.0])  # 1 unit, at 2 m to the unit


def test_frames_are_placed_outward_through_the_strongest_pairs():
    found = [
        sm.Edge("root", "b", shifted(x=1.0), 300),
        sm.Edge("c", "b", shifted(x=-5.0), 900),  # c is where b is, less 5: b in c's frame is at -5... so c = b + 5
        sm.Edge("root", "c", shifted(x=9.0), 200),  # a weaker, disagreeing measure of c: never used
        sm.Edge("b", "weak", shifted(x=1.0), 20),
    ]
    placed = sm.grow({"root": np.eye(4)}, found, least=150)
    assert set(placed) == {"b", "c"}  # "weak" has too few agreeing matches
    assert placed["b"][:3, 3] == pytest.approx([1.0, 0.0, 0.0])
    assert placed["c"][:3, 3] == pytest.approx([6.0, 0.0, 0.0])  # through b (900 matches), not root's 200


def test_a_frame_with_no_pair_to_a_placed_one_is_not_placed():
    assert sm.grow({"root": np.eye(4)}, [sm.Edge("x", "y", shifted(x=1.0), 900)]) == {}


# ---- points


def test_depth_is_lifted_to_points_in_the_maps_frame():
    depth = np.full((4, 4), 2.0, dtype=np.float32)
    depth[0, 0] = 0.0  # no depth there
    depth[3, 3] = 50.0  # too far to trust
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[2, 2] = (10, 20, 30)
    points, colors = sm.lift(depth, image, focal=2.0, pose=shifted(z=10.0), stride=1)
    assert len(points) == 14
    at = points[(colors == (10, 20, 30)).all(axis=1)]
    # Pixel (2, 2) of a 4 x 4 picture is half a pixel right of and below the centre: 0.5 * 2 m / 2 px = 0.5 m each way.
    assert at == pytest.approx(np.array([[0.5, 0.5, 12.0]]))


def test_the_cloud_keeps_one_point_per_cube_and_the_maps_own_points_first():
    points = np.array([[0.0, 0.0, 0.0], [0.01, 0.01, 0.01], [1.0, 0.0, 0.0]], dtype=np.float32)
    colors = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.uint8)
    _, kept_colors = sm.thin(points, colors, voxel=0.03)
    assert kept_colors.tolist() == [[1, 1, 1], [3, 3, 3]]
    assert (
        len(
            sm.thin(
                np.random.default_rng(0).random((500, 3)).astype(np.float32),
                np.zeros((500, 3), np.uint8),
                0.001,
                most=100,
            )[0]
        )
        == 100
    )


# ---- the second gate


@pytest.mark.parametrize(
    ("before", "adds_nothing"),
    [
        ({"covered": 0.999, "psnr": 23.0}, True),
        ({"covered": 0.999, "psnr": 19.0}, False),  # covered, but it looks different now
        ({"covered": 0.90, "psnr": 25.0}, False),  # drawn well where drawn, but a tenth of it is new
        ({"covered": 0.0, "psnr": None}, False),
    ],
    ids=["already drawn", "changed", "uncovered area", "nothing of it"],
)
def test_a_frame_the_map_already_draws_well_adds_nothing(before, adds_nothing):
    assert sm.well_drawn(before) is adds_nothing


# ---- a camera's optics


def test_a_camera_with_nothing_known_uses_the_focal_learned_from_its_own_frames():
    assert sm.optics(None, [], None, None) == sm.Optics(None)
    assert sm.optics(None, [], None, 0.7) == sm.Optics(0.7)


def test_held_intrinsics_win_over_a_learned_focal_and_bring_their_lens():
    pinhole = lens.Held(0.66, 0.0, "fit")
    assert sm.optics(None, [], pinhole, 0.7) == sm.Optics(0.66)
    barrel = sm.optics(None, [], lens.Held(1.16, -0.29, "fit"), 0.7)
    assert barrel.focal == 1.16
    assert barrel.undo is not None
    undo = sm.held_lens(barrel.undo, 960, 540)
    assert (undo.width, undo.height, undo.division) == (960, 540, -0.29)
    assert undo.k == pytest.approx((1.16 * 960, 1.16 * 960, 479.5, 269.5))


def test_a_lens_in_the_manifest_is_undone_as_it_is_with_its_overlays_masked():
    configured = lens.Lens(896, 512, (448.0, 448.0, 447.5, 255.5), -0.3)
    optic = sm.optics(configured, [(0.0, 0.0, 0.2, 0.1)], None, 0.5)
    assert (optic.focal, optic.undo, optic.masks) == (0.5, configured, ((0.0, 0.0, 0.2, 0.1),))
    assert sm.held_lens(configured, 960, 540) is configured


# ---- the trainer reads back the splat it wrote


def splat_bytes(n: int, bands: int = 3) -> tuple[bytes, np.ndarray]:
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(3 * bands)]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    rows = np.arange(n * len(names), dtype="<f4").reshape(n, len(names))
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
    header += "".join(f"property float {k}\n" for k in names) + "end_header\n"
    return header.encode("ascii") + rows.tobytes(), rows


def test_a_trained_splat_reads_back_field_by_field():
    data, rows = splat_bytes(2)
    got = splat_runner.read_splat(data, np)
    assert got["means"].tolist() == rows[:, 0:3].tolist()
    assert got["sh0"].tolist() == rows[:, 6:9].tolist()
    assert got["rest"].shape == (2, 3, 3)  # channel by channel: red's three bands first
    assert got["rest"][0, 0].tolist() == rows[0, 9:12].tolist()
    assert got["opacity"].tolist() == rows[:, 18].tolist()
    assert got["scales"].tolist() == rows[:, 19:22].tolist()
    assert got["quats"].tolist() == rows[:, 22:26].tolist()


@pytest.mark.parametrize(
    "data",
    [b"not a ply", splat_bytes(2)[0][:-4], ply.dumps(np.zeros((2, 3)))],
    ids=["not a PLY", "cut short", "a point cloud, not a splat"],
)
def test_a_file_that_is_not_a_trained_splat_is_refused(data):
    with pytest.raises(ValueError):
        splat_runner.read_splat(data, np)


# ---- a whole step, with the three model runs stood in for


class Models:
    """Stand-ins for the three subprocesses a step runs: MASt3R's pairs, the depth model, and the splat trainer."""

    def __init__(self):
        self.pairs: dict[frozenset[str], dict] = {}  # what MASt3R answers for a pair, by its two frames
        self.before: dict[int, dict] = {}  # the trainer's `before` for a view index
        self.jobs: list[dict] = []
        self.measured: list[list[tuple[str, str]]] = []
        self.depth = lambda height, width, focal: np.full((height, width), 2.0, dtype=np.float32)  # a wall 2 m off

    def run_mast3r(self, runtime, images, out, gpu, *, pairs, nice=0, **_):
        self.measured.append([tuple(p) for p in pairs])
        answer = []
        for a, b in pairs:
            if a == b:
                answer.append({**pair(a, b, np.eye(4)), "focals": [358.4, 358.4], "sizes": [[512, 288], [512, 288]]})
            elif frozenset((a, b)) in self.pairs:
                answer.append({**self.pairs[frozenset((a, b))], "a": a, "b": b})
            else:
                answer.append(pair(a, b, None))
        return {"pairs": answer}

    def run_depth(self, runtime, request, gpu):
        import cv2

        for view in json.loads(Path(request).read_text())["views"]:
            image = cv2.imread(view["image"])
            assert image is not None
            height, width = image.shape[:2]
            np.save(view["out"], self.depth(height, width, view["focal"]))

    def run_runner(self, runtime, job, gpu, timeout_s=0.0):
        spec = json.loads(Path(job).read_text())
        self.jobs.append(spec)
        (Path(spec["out"]) / "splat.ply").write_bytes(splat_bytes(3)[0])
        scored = [{"view": i, **self.before.get(i, {"covered": 0.5, "psnr": 12.0})} for i in spec.get("score", [])]
        return {
            "gaussians": 3,
            "train_psnr": 25.0,
            "seconds": 1.0,
            "before": scored,
            "poses": [v["pose"] for v in spec["views"]],
        }


@pytest.fixture
def models(monkeypatch, tmp_path):
    fake = Models()
    monkeypatch.setattr(cameras, "run_mast3r", fake.run_mast3r)
    monkeypatch.setattr(splats, "run_depth", fake.run_depth)
    monkeypatch.setattr(splats, "run_runner", fake.run_runner)
    monkeypatch.setattr(sm.nvml, "gpus", lambda: None)
    # The depth model's code and weights, as far as the step checks for them.
    runtime = tmp_path / "gsplat"
    (runtime / splats.DEPTH_SOURCE).mkdir(parents=True)
    weights = tmp_path / "weights" / "model.safetensors"
    weights.parent.mkdir()
    weights.write_bytes(b"")
    monkeypatch.setattr(splats, "depth_checkpoint", lambda: weights)
    manifest = tmp_path / "devices.toml"
    manifest.write_text('[[device]]\nname = "front"\nkind = "camera"\n')
    monkeypatch.setattr(lens, "manifest", lambda: manifest)
    with cfg.override({"spatial": {"splat_runtime": str(runtime), "runtime": str(tmp_path / "mast3r")}}):
        yield fake


@pytest.fixture
def state(tmp_path):
    return tmp_path / "state"


def keep(state: Path, angle: float, t: float, camera: str = "front", size: tuple[int, int] = (64, 36)) -> frames.Kept:
    """Put a made-up frame in the kept store, as the sampler would have."""
    store = frames.Store(frames.root(state))
    picture = np.full((size[1], size[0], 3), 90, dtype=np.uint8)
    _, kept = store.offer(camera, picture, unit(angle), 100.0, now=t)
    assert kept is not None
    return kept


def run(state: Path, now: float) -> dict:
    did = sm.step(state, now=now, held={})
    assert did is not None
    return did


def newest(state: Path, place: str) -> places.Revision:
    rev = places.latest(state, place)
    assert rev is not None
    return rev


def test_a_step_with_nothing_waiting_does_nothing(models, state):
    assert sm.step(state, now=100.0, held={}) is None
    assert models.jobs == [] and models.measured == []
    assert places.place_ids(state) == []


def test_the_first_frame_founds_a_place_with_a_map_and_a_splat(models, state):
    first = keep(state, 0, t=1.0)
    did = run(state, 100.0)
    [(place, result)] = did["places"].items()
    assert (result["founded"], result["views"], result["added"], did["unplaced"]) == (True, 1, 1, 0)
    assert models.measured == [[(first.id, first.id)]]  # only its own focal was asked of MASt3R
    rev = newest(state, place)
    assert (rev.producer, rev.map_format, rev.frame) == ("surroundings v1", "ply", rev.id)
    assert places.splat_source(state, rev) == "surroundings"
    assert rev.map is not None
    points, colors, _ = ply.read(places.root(state) / place / rev.map)
    assert len(points) > 0
    assert colors is not None and bool((colors == 90).all())
    assert points[:, 2] == pytest.approx(2.0)  # the depth model's 2 m, from a camera at the origin looking along z
    [job] = models.jobs
    assert "resume" not in job and job["steps"] == sm.FOUND_STEPS
    assert job["views"][0]["focal"] == pytest.approx(0.7 * 64)  # 358.4 px of MASt3R's 512, at the picture's 64
    [view] = sm.views(state, place)
    assert (view.id, view.camera, view.pose, view.focal) == (
        first.id,
        "front",
        tuple(np.eye(4).ravel()),
        pytest.approx(0.7),
    )


def test_a_later_frame_joins_the_place_and_the_splat_resumes_from_where_it_was(models, state):
    first = keep(state, 0, t=1.0)
    sm.step(state, now=100.0, held={})
    [place] = places.place_ids(state)
    founding = newest(state, place)
    second = keep(state, 40, t=2.0)
    models.pairs[frozenset((first.id, second.id))] = pair(first.id, second.id, shifted(x=0.5), depths=(1.0, 1.0))
    did = run(state, 200.0)
    result = did["places"][place]
    assert (result["founded"], result["views"], result["added"], result["already_drawn"]) == (False, 2, 1, 0)
    job = models.jobs[-1]
    assert job["resume"] == str(places.splat_path(state, founding, "gs.ply"))
    assert (job["steps"], job["score"], job["pose_start"]) == (sm.ROUND_STEPS, [1], 0)
    rev = newest(state, place)
    assert rev.id != founding.id and rev.frame == founding.id  # a new revision in the same frame
    assert places.has_splat(state, rev)
    poses = {v.id: np.array(v.pose).reshape(4, 4) for v in sm.views(state, place)}
    assert poses[second.id][:3, 3] == pytest.approx([1.0, 0.0, 0.0])  # 0.5 of the pair's units, at 2 m to the unit
    summary = sm.summary(state)
    assert summary["places"][place] == {"revision": rev.id, "views": 2, "rounds": 2, "improved": 200.0}
    assert (summary["kept"], summary["waiting"], summary["parked"], summary["running"]) == (2, 0, 0, False)


def test_a_frame_the_map_already_draws_is_handed_back_to_be_dropped(models, state):
    first = keep(state, 0, t=1.0)
    sm.step(state, now=100.0, held={})
    [place] = places.place_ids(state)
    second = keep(state, 40, t=2.0)
    models.pairs[frozenset((first.id, second.id))] = pair(first.id, second.id, shifted(x=0.5))
    models.before[1] = {"covered": 0.999, "psnr": 24.0}
    did = run(state, 200.0)
    assert (did["places"][place]["added"], did["places"][place]["already_drawn"]) == (0, 1)
    saved = sm.State.read(frames.root(state))
    assert saved.drop == [second.id]
    assert [v.id for v in sm.views(state, place)] == [first.id]
    assert saved.places[place]["improved"] == 100.0  # nothing was added: the place didn't improve


def test_a_frame_no_pair_places_is_parked_and_starts_no_more_steps(models, state):
    keep(state, 0, t=1.0)
    sm.step(state, now=100.0, held={})
    stray = keep(state, 90, t=2.0)
    did = run(state, 200.0)
    assert (did["unplaced"], did["places"]) == (1, {})
    assert sm.step(state, now=260.0, held={}) is None  # parked: a step a minute later has nothing to do
    assert [f.id for f in frames.Store(frames.root(state)).kept if f.id == stray.id] == [stray.id]  # still kept
    summary = sm.summary(state, now=260.0)
    assert (summary["kept"], summary["waiting"], summary["parked"]) == (2, 0, 1)


def test_a_parked_frame_is_tried_again_once_the_map_has_gained_a_view(models, state):
    first = keep(state, 0, t=1.0)
    sm.step(state, now=100.0, held={})
    [place] = places.place_ids(state)
    stray = keep(state, 90, t=2.0)
    run(state, 200.0)
    bridge = keep(state, 45, t=3.0)  # a view between the two: it pairs with both
    models.pairs[frozenset((first.id, bridge.id))] = pair(first.id, bridge.id, shifted(x=0.5))
    models.pairs[frozenset((bridge.id, stray.id))] = pair(bridge.id, stray.id, shifted(x=0.5))
    did = run(state, 300.0)  # takes the bridge only: the stray is parked until the map grows
    assert (did["taken"], did["places"][place]["views"]) == (1, 2)
    assert sm.summary(state, now=300.0)["waiting"] == 1  # the map grew: the stray is due again
    did = run(state, 400.0)
    assert (did["taken"], did["places"][place]["views"], did["unplaced"]) == (1, 3, 0)


def test_a_frame_parked_for_a_day_is_dropped(models, state):
    keep(state, 0, t=1.0)
    sm.step(state, now=100.0, held={})
    stray = keep(state, 90, t=2.0)
    run(state, 200.0)
    just_under = 200.0 + sm.PARK_S - 1
    assert sm.due(frames.Store(frames.root(state)).kept, sm.State.read(frames.root(state)), just_under) == []
    assert sm.step(state, now=200.0 + sm.PARK_S, held={}) is None
    saved = sm.State.read(frames.root(state))
    assert (saved.drop, stray.id in saved.frames) == ([stray.id], False)


def test_a_frame_left_waiting_by_an_earlier_version_is_tried_once_and_parked(models, state):
    keep(state, 0, t=1.0)
    sm.step(state, now=100.0, held={})
    stray = keep(state, 90, t=2.0)
    saved = sm.State.read(frames.root(state))
    saved.frames[stray.id] = {"tries": 2}
    saved.write(frames.root(state))
    assert run(state, 200.0)["unplaced"] == 1
    assert sm.step(state, now=260.0, held={}) is None


def test_a_learned_revision_keeps_the_floor_set_in_its_frame(models, state):
    first = keep(state, 0, t=1.0)
    sm.step(state, now=100.0, held={})
    [place] = places.place_ids(state)
    places.revise(state, place, producer="floor v1", floor=(0.0, -1.0, 0.0, 1.4), now=150.0)
    second = keep(state, 40, t=2.0)
    models.pairs[frozenset((first.id, second.id))] = pair(first.id, second.id, shifted(x=0.5))
    sm.step(state, now=200.0, held={})
    assert newest(state, place).floor == (0.0, -1.0, 0.0, 1.4)


def test_the_first_learned_map_takes_the_one_place_that_has_no_map(models, state):
    kept_place = places.revise(state, None, producer="places reset v1", now=10.0).place
    keep(state, 0, t=1.0)
    did = run(state, 100.0)
    assert list(did["places"]) == [kept_place]


def test_what_the_step_knew_of_frames_no_longer_kept_is_forgotten(models, state):
    first = keep(state, 0, t=1.0)
    sm.step(state, now=100.0, held={})
    frames.Store(frames.root(state)).remove(first.id)
    assert sm.step(state, now=200.0, held={}) is None
    assert sm.State.read(frames.root(state)).frames == {}


def test_the_people_marked_in_a_kept_frame_are_left_out_of_its_depth_and_its_training(models, state):
    store = frames.Store(frames.root(state))
    covered = np.zeros((36, 64), dtype=bool)
    covered[18:, 24:40] = True  # someone in the lower middle
    store.offer("front", np.full((36, 64, 3), 90, dtype=np.uint8), unit(0), 100.0, now=1.0, covered=covered)
    did = run(state, 100.0)
    [place] = did["places"]
    [view] = models.jobs[0]["views"]
    assert view["mask"].endswith(".mask.png")  # the trainer's loss leaves those pixels out
    rev = newest(state, place)
    assert rev.map is not None
    points, _, _ = ply.read(places.root(state) / place / rev.map)
    # No point was lifted from where the person stood: at 2 m with a focal of 44.8 px, the columns 24 to 39 of the
    # lower half are x from -0.36 to 0.36 m and y above 0.
    under = (np.abs(points[:, 0]) < 0.3) & (points[:, 1] > 0.05)
    assert len(points) > 0 and not under.any()


# ---- what a running step says it's doing (#283), for the Places page


class Watching(Models):
    """The model runs, each noting what `summary` says while it runs; the trainer first says how far it is."""

    def __init__(self, state: Path):
        super().__init__()
        self.state = state
        self.seen: list[dict] = []

    def note(self):
        said = sm.summary(self.state)
        self.seen.append({"running": said["running"], **(said["step"] or {})})

    def run_mast3r(self, *args, **kwargs):
        self.note()
        return super().run_mast3r(*args, **kwargs)

    def run_depth(self, *args, **kwargs):
        self.note()
        return super().run_depth(*args, **kwargs)

    def run_runner(self, runtime, job, gpu, timeout_s=0.0):
        self.note()  # training has begun, and the trainer hasn't said a step yet
        spec = json.loads(Path(job).read_text())
        Path(spec["progress"]).write_text(json.dumps({"step": 1200, "steps": spec["steps"]}))
        self.note()
        return super().run_runner(runtime, job, gpu, timeout_s)


@pytest.fixture
def watching(models, monkeypatch, state):
    fake = Watching(state)
    monkeypatch.setattr(cameras, "run_mast3r", fake.run_mast3r)
    monkeypatch.setattr(splats, "run_depth", fake.run_depth)
    monkeypatch.setattr(splats, "run_runner", fake.run_runner)
    return fake


def test_a_running_step_says_each_phase_and_how_far_the_training_is(watching, state):
    keep(state, 0, t=1.0)
    run(state, 100.0)
    none = {"place": None, "views": None, "step": None, "steps": None}
    assert watching.seen == [
        {"running": True, **none, "phase": "pairing", "frames": 1},
        {"running": True, **none, "phase": "depth", "frames": 1},
        {"running": True, **none, "phase": "training", "frames": 1, "views": 1, "steps": sm.FOUND_STEPS},
        {"running": True, **none, "phase": "training", "frames": 1, "views": 1, "step": 1200, "steps": sm.FOUND_STEPS},
    ]


def test_a_step_that_continues_a_place_names_it_and_an_earlier_trainings_count_is_not_shown(watching, state):
    first = keep(state, 0, t=1.0)
    run(state, 100.0)
    [place] = places.place_ids(state)
    second = keep(state, 40, t=2.0)
    watching.pairs[frozenset((first.id, second.id))] = pair(first.id, second.id, shifted(x=0.5))
    watching.seen.clear()
    run(state, 200.0)
    began, later = [s for s in watching.seen if s["phase"] == "training"]
    assert (began["place"], began["step"], began["steps"]) == (place, None, sm.ROUND_STEPS)
    assert (later["place"], later["step"]) == (place, 1200)


def test_once_a_step_ends_nothing_is_running_and_the_summary_says_when_it_ended(watching, state):
    keep(state, 0, t=1.0)
    did = run(state, 100.0)
    said = sm.summary(state)
    assert (said["running"], said["step"]) == (False, None)
    assert said["last_step"] == pytest.approx(100.0 + did["seconds"])
    folder = frames.root(state)
    assert not (folder / sm.PROGRESS).exists() and not (folder / sm.TRAINING).exists()


def test_a_step_that_fails_leaves_no_progress_behind(models, monkeypatch, state):
    def fails(runtime, job, gpu, timeout_s=0.0):
        raise RuntimeError("the splat trainer failed")

    monkeypatch.setattr(splats, "run_runner", fails)
    keep(state, 0, t=1.0)
    with pytest.raises(RuntimeError):
        sm.step(state, now=100.0, held={})
    assert not (frames.root(state) / sm.PROGRESS).exists()
    assert sm.summary(state)["step"] is None


def test_what_a_step_that_died_left_behind_is_not_shown_as_running(state):
    folder = frames.root(state)
    folder.mkdir(parents=True)
    (folder / sm.PROGRESS).write_text(json.dumps({"phase": "training", "frames": 2, "steps": 1500}))
    said = sm.summary(state)
    assert (said["running"], said["step"], said["last_step"]) == (False, None, None)


@pytest.mark.parametrize(
    "written",
    [
        pytest.param("not json", id="not-json"),
        pytest.param(json.dumps(["training"]), id="not-an-object"),
        pytest.param(json.dumps({"phase": "dancing", "frames": -1, "place": 7, "views": True}), id="not-ours"),
    ],
)
def test_progress_that_isnt_a_steps_says_nothing(tmp_path, written):
    (tmp_path / sm.PROGRESS).write_text(written)
    assert sm.progress(tmp_path) == {
        "phase": None, "frames": None, "place": None, "views": None, "step": None, "steps": None,
    }  # fmt: skip


def test_the_summary_gives_the_sampled_cameras_recent_looks_from_the_samplers_status(monkeypatch, state):
    from liberational.spatial import surroundings

    folder = frames.root(state)
    folder.mkdir(parents=True)
    row = {"dark": 0, "same": 40, "person": 9, "view": 2, "last_kept": 50.0, "recent": {"same": 3, "person": 9}}
    status = {"t": 90.0, "on": True, "sampling": ["front"], "kept": 2, "cameras": {"front": row, "attic": row}}
    (folder / "status.json").write_text(json.dumps(status))
    monkeypatch.setattr(surroundings, "read", lambda: surroundings.Switch(True, ("front", "hall"), (), "front"))
    said = sm.summary(state)
    # only the cameras sampled now, and one the sampler hasn't looked at yet has no row
    assert said["cameras"] == {"front": {"recent": {"same": 3, "person": 9}, "last_kept": 50.0}}
    monkeypatch.setattr(surroundings, "read", lambda: surroundings.Switch(False, ("front",), (), "front"))
    assert sm.summary(state)["cameras"] == {}


# ---- a camera with a spot on the owner's plan (#288)

ROOMS = 'plan "Home" walls 0.2/0.1\nroom r1 "Living" other rect 0,0 4x3\nroom r2 "Hall" other rect 0,3 4x2\n'
#: A camera 1 m up at (2, 1.5) on the plan, level, looking along the plan's +x: its right is the plan's -y, its
#: down the plan's -z.
LOOKING_EAST = (0.0, 0.0, 1.0, 2.0, -1.0, 0.0, 0.0, 1.5, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def spot(place: str, camera: str = "front", pose=LOOKING_EAST, missing: tuple[str, ...] = ()) -> spots.Spot:
    posed = None if missing else places.CameraPose(tuple(pose), 44.8, 64, 36)  # a focal 0.7 of the width
    return spots.Spot(camera, place, "r", (pose[3], pose[7]), pose[11], 90.0, 90.0, True, posed, missing)


def owned(state: Path) -> str:
    """A place kept across a reset: no map, and the owner's plan of two rooms."""
    place = places.revise(state, None, producer="places reset v1", now=10.0).place
    places.correct_plan(state, place, ROOMS, now=20.0)
    return place


def floor_depth(long: float = 1.0):
    """The depth a level camera 1 m up sees of a floor, with a wall 4 m off above the horizon, as a depth model
    `long` times too long would give it."""

    def depth(height: int, width: int, focal: float) -> np.ndarray:
        down = (np.arange(height) + 0.5 - height / 2)[:, None] / focal  # how far below the axis each row looks
        with np.errstate(divide="ignore"):
            floor = np.where(down > 0, 1.0 / down, np.inf)
        return np.broadcast_to(np.minimum(floor, 4.0) * long, (height, width)).astype(np.float32)

    return depth


def test_a_views_floor_is_found_where_its_depth_puts_it_and_its_depth_scaled_to_the_plans():
    focal = 0.7 * 256
    depth = floor_depth(long=1.1)(144, 256, focal)
    fit = sm.spot_fit(depth, focal, np.array(LOOKING_EAST).reshape(4, 4), None)
    # A depth 1.1 times too long puts the floor of a camera 1 m up at 1 - 1.1 = -0.1 m, and 1 / 1.1 brings it back.
    assert fit.floor == pytest.approx(-0.1, abs=0.02)
    assert fit.scale == pytest.approx(1 / 1.1, abs=0.02)
    assert fit.why is None


def test_a_view_whose_floor_is_far_from_the_plans_is_refused():
    focal = 0.7 * 256
    fit = sm.spot_fit(floor_depth(long=2.0)(144, 256, focal), focal, np.array(LOOKING_EAST).reshape(4, 4), None)
    assert (fit.why, fit.scale) == ("floor", pytest.approx(0.5, abs=0.03))  # no depth model is twice too long


def test_a_view_with_its_floor_on_the_plans_keeps_its_depth_and_one_with_no_floor_too():
    focal = 0.7 * 256
    on = sm.spot_fit(floor_depth()(144, 256, focal), focal, np.array(LOOKING_EAST).reshape(4, 4), None)
    assert (on.floor, on.scale) == (pytest.approx(0.0, abs=0.02), 1.0)
    wall_only = sm.spot_fit(np.full((144, 256), 2.0, np.float32), focal, np.array(LOOKING_EAST).reshape(4, 4), None)
    assert (wall_only.floor, wall_only.scale, wall_only.why) == (None, 1.0, None)


def test_a_few_strays_under_the_floor_are_not_the_floor():
    rng = np.random.default_rng(0)
    floor = np.c_[rng.uniform(0, 3, (3000, 2)), rng.normal(0.0, 0.003, 3000)]
    strays = np.c_[rng.uniform(0, 0.4, (80, 2)), rng.normal(-1.2, 0.003, 80)]  # a level patch far below
    table = np.c_[rng.uniform(1, 2, (4000, 2)), rng.normal(0.75, 0.003, 4000)]  # larger than the floor, but too high
    assert sm.level_floor(np.vstack([floor, strays, table]), height=1.4) == pytest.approx(0.0, abs=0.02)


def test_a_camera_with_a_spot_is_placed_from_it_with_no_pairing_in_the_plans_frame(models, state):
    from liberational.spatial import plan_lineup

    place = owned(state)
    first = keep(state, 0, t=1.0)
    did = sm.step(state, now=100.0, held={}, spotted={"front": spot(place)})
    assert did is not None
    assert models.measured == []  # MASt3R was asked nothing
    assert (list(did["places"]), did["places"][place]["views"], did["unplaced"]) == ([place], 1, 0)
    rev = newest(state, place)
    assert (rev.floor, rev.plan_transform) == (spots.FLOOR, spots.PLAN)
    assert rev.cameras["front"].pose == LOOKING_EAST
    [view] = models.jobs[0]["views"]
    assert (view["pose"], view["fixed"]) == (list(LOOKING_EAST), True)
    [seen] = sm.views(state, place)
    assert (seen.id, seen.pose) == (first.id, LOOKING_EAST)
    assert sm.in_plan_frame(state, place)
    # The wall 2 m in front of a camera at x = 2 looking along +x is at x = 4, in the plan's metres.
    assert rev.map is not None
    points, _, _ = ply.read(places.root(state) / place / rev.map)
    assert points[:, 0] == pytest.approx(4.0)
    # The owner's plan is drawn in this map's frame, and the line-up finds it so without a search.
    held = places.get(state, place)
    assert held is not None and (held.plan_source, held.plan_revision) == ("owner", rev.id)
    assert held.plan == ROOMS
    lined = plan_lineup.advance(state, now=101.0)[place]
    assert (lined["state"], lined["by"]) == ("aligned", "owner")


def test_a_camera_whose_spot_lacks_its_facing_is_parked_with_the_reason_until_the_owner_gives_one(models, state):
    place = owned(state)
    keep(state, 0, t=1.0, camera="hall")
    keep(state, 90, t=2.0)
    lacking = {"hall": spot(place, "hall"), "front": spot(place, missing=("facing",))}
    did = sm.step(state, now=100.0, held={}, spotted=lacking)
    assert did is not None and did["unplaced"] == 1
    saved = sm.State.read(frames.root(state))
    [parked] = [entry for entry in saved.frames.values() if entry.get("parked") is not None]
    assert (parked["why"], parked["basis"]) == ("facing", "spot:facing")
    assert (
        sm.due(frames.Store(frames.root(state)).kept, saved, 160.0, sm.bases(lacking)) == []
    )  # no step a minute later
    given = {**lacking, "front": spot(place, pose=(*LOOKING_EAST[:3], 1.0, *LOOKING_EAST[4:]))}
    assert len(sm.due(frames.Store(frames.root(state)).kept, saved, 160.0, sm.bases(given))) == 1
    did = sm.step(state, now=200.0, held={}, spotted=given)
    assert did is not None and (did["unplaced"], did["places"][place]["views"]) == (0, 2)


def test_a_camera_with_no_spot_joins_through_a_view_that_overlaps_a_spot_view(models, state):
    place = owned(state)
    at_spot = keep(state, 0, t=1.0, camera="hall")
    other = keep(state, 40, t=2.0)
    models.pairs[frozenset((at_spot.id, other.id))] = pair(at_spot.id, other.id, shifted(z=0.5))
    did = sm.step(state, now=100.0, held={}, spotted={"hall": spot(place, "hall")})
    assert did is not None and did["places"][place]["views"] == 2
    poses = {v.id: np.array(v.pose).reshape(4, 4) for v in sm.views(state, place)}
    # Half a pair's unit forward of the spot view, at 2 m to the unit: 1 m along the plan's +x from x = 2.
    assert poses[other.id][:3, 3] == pytest.approx([3.0, 1.5, 1.0])
    fixed = {v["image"].split("/")[-1].split(".")[0]: v.get("fixed", False) for v in models.jobs[0]["views"]}
    assert fixed == {at_spot.id: True, other.id: False}


def test_a_map_learned_in_a_frame_of_its_own_is_started_again_when_a_spot_view_arrives(models, state):
    place = owned(state)
    old = keep(state, 0, t=1.0)
    did = sm.step(state, now=100.0, held={}, spotted={})
    assert did is not None and list(did["places"]) == [place]  # the one place with no map took it, at its own origin
    first = newest(state, place)
    assert first.plan_transform is None
    keep(state, 90, t=2.0, camera="hall")
    did = sm.step(state, now=200.0, held={}, spotted={"hall": spot(place, "hall")})
    assert did is not None
    rev = newest(state, place)
    assert rev.frame == rev.id != first.frame  # a new frame: the plan's
    assert "resume" not in models.jobs[-1]  # the old frame's splat is no start for it
    assert [v.camera for v in sm.views(state, place)] == ["hall"]
    saved = sm.State.read(frames.root(state))
    assert old.id not in saved.frames  # back to waiting, not parked: the next step tries it against the new map


def test_a_view_is_placed_again_when_its_cameras_spot_moves(models, state):
    place = owned(state)
    keep(state, 0, t=1.0)
    sm.step(state, now=100.0, held={}, spotted={"front": spot(place)})
    moved = (*LOOKING_EAST[:3], 1.0, *LOOKING_EAST[4:])  # the owner put the camera a metre to the west
    did = sm.step(state, now=200.0, held={}, spotted={"front": spot(place, pose=moved)})
    assert did is not None
    [seen] = sm.views(state, place)
    assert seen.pose == moved


def test_a_spot_view_whose_floor_is_far_off_is_parked_as_such(models, state):
    place = owned(state)
    keep(state, 0, t=1.0, size=(256, 144))
    models.depth = floor_depth(long=2.0)
    did = sm.step(state, now=100.0, held={}, spotted={"front": spot(place)})
    assert did is not None and (did["unplaced"], did["places"]) == (1, {})
    assert did["spots"]["front"]["why"] == "floor"
    summary = sm.summary(state, now=160.0, spotted={"front": spot(place)})
    assert (summary["parked"], summary["parked_cameras"]) == (1, {"front": "floor"})


# ---- spot verification and bounded refinement on a made-up room with known true pose (#292)


def _synthetic_plan(cx: float = 0.65, cy: float = 0.75, cz: float = 1.65, yaw: float = 125.0) -> str:
    return (
        'plan "Synthetic" walls 0.2/0.1\n'
        'room r1 "Main" living rect 0,0 5x4\n'
        'room r2 "Side" other rect 5,0 3.5x4\n'
        "cased r1>r2 at 5,2.0 w2.4\n"
        "door exterior>r1 at 2.4,0 w0.9\n"
        f"x-camera front at {cx:.3f},{cy:.3f} height {cz:.3f} yaw {yaw:.2f} fov 90\n"
    )


def _render_synthetic_room(
    plan_text: str,
    c2w: np.ndarray,
    width: int = 320,
    height: int = 180,
    focal: float = 160.0,
    floor_warp: float = 1.10,
    wall_warp: float = 0.99,
) -> tuple[np.ndarray, np.ndarray]:
    """Ray-trace a made-up two-room scene with an open cased archway, a foreground sofa in the wall band, and a
    depth warp between near floor and far walls like Depth Anything 3's."""
    from liberational.spatial import plan_import

    parsed = plan_import.parse(plan_text)
    solid_w = plan_import.sight_walls(parsed)
    full_w = plan_import.walls(parsed)
    solid = np.hstack([solid_w.a, solid_w.b])
    full = np.hstack([full_w.a, full_w.b])

    uu, vv = np.meshgrid(np.arange(width, dtype=np.float64) + 0.5, np.arange(height, dtype=np.float64) + 0.5)
    dirs_c = np.stack([(uu - width / 2.0) / focal, (vv - height / 2.0) / focal, np.ones_like(uu)], axis=-1)
    dirs_w = dirs_c @ c2w[:3, :3].T
    orig = c2w[:3, 3]

    best_t = np.full((height, width), np.inf, dtype=np.float64)
    shade = np.full((height, width, 3), 210, dtype=np.uint8)
    is_floor = np.zeros((height, width), dtype=bool)

    # Floor (z = 0) and ceiling (z = 2.6).
    with np.errstate(divide="ignore", invalid="ignore"):
        t_floor = np.where(dirs_w[..., 2] < -1e-6, -orig[2] / dirs_w[..., 2], np.inf)
        t_ceil = np.where(dirs_w[..., 2] > 1e-6, (2.6 - orig[2]) / dirs_w[..., 2], np.inf)
    hit_f = t_floor < best_t
    best_t = np.where(hit_f, t_floor, best_t)
    shade[hit_f] = (92, 86, 78)
    is_floor[hit_f] = True
    hit_c = t_ceil < best_t
    best_t = np.where(hit_c, t_ceil, best_t)
    shade[hit_c] = (228, 228, 224)
    is_floor[hit_c] = False

    def cast_segments(segs: np.ndarray, z_lo: float, z_hi: float, palette: Sequence[tuple[int, int, int]]) -> None:
        nonlocal best_t
        dx, dy = dirs_w[..., 0], dirs_w[..., 1]
        for idx, (ax, ay, bx, by) in enumerate(segs.tolist()):
            sx, sy = bx - ax, by - ay
            denom = dx * sy - dy * sx
            with np.errstate(divide="ignore", invalid="ignore"):
                t = ((ax - orig[0]) * sy - (ay - orig[1]) * sx) / denom
                u = ((ax - orig[0]) * dy - (ay - orig[1]) * dx) / denom
            z = orig[2] + t * dirs_w[..., 2]
            ok = (
                (np.abs(denom) > 1e-8) & (t > 0.05) & (u >= 0.0) & (u <= 1.0) & (z >= z_lo) & (z <= z_hi) & (t < best_t)
            )
            if not np.any(ok):
                continue
            best_t = np.where(ok, t, best_t)
            color = np.array(palette[idx % len(palette)], dtype=np.uint8)
            shade[ok] = color
            is_floor[ok] = False
            # Crisp baseboard and jamb trim for structural edge detection.
            trim = ok & ((z < 0.04) | (z > 2.56) | (u < 0.015) | (u > 0.985))
            shade[trim] = (38, 36, 34)

    wall_colors = [(172, 168, 160), (148, 152, 158), (186, 178, 166), (160, 156, 150)]
    cast_segments(solid, 0.0, 2.6, wall_colors)
    # Header above the 2.1 m cased archway and door.
    cast_segments(full, 2.1, 2.6, wall_colors)

    # A sofa/cabinet block inside r1 at x in [2.2, 3.2], y in [2.5, 3.5], z in [0, 1.45], sitting in the wall band.
    box_walls = np.array(
        [[2.2, 2.5, 3.2, 2.5], [3.2, 2.5, 3.2, 3.5], [3.2, 3.5, 2.2, 3.5], [2.2, 3.5, 2.2, 2.5]], dtype=np.float64
    )
    cast_segments(box_walls, 0.0, 1.45, [(68, 82, 98)])
    with np.errstate(divide="ignore", invalid="ignore"):
        t_top = np.where(dirs_w[..., 2] < -1e-6, (1.45 - orig[2]) / dirs_w[..., 2], np.inf)
    px = orig[0] + t_top * dirs_w[..., 0]
    py = orig[1] + t_top * dirs_w[..., 1]
    hit_top = (t_top > 0.05) & (t_top < best_t) & (px >= 2.2) & (px <= 3.2) & (py >= 2.5) & (py <= 3.5)
    best_t = np.where(hit_top, t_top, best_t)
    shade[hit_top] = (76, 90, 106)
    is_floor[hit_top] = False

    # Depth along the camera's +z axis (`best_t` multiplies unit-z camera rays `dirs_c`), warped like DA3: near floor
    # overestimated by `floor_warp`, far walls at `wall_warp`.
    warp = np.where(is_floor & (best_t < 3.6), floor_warp, wall_warp)
    depth = np.where(np.isfinite(best_t), best_t * warp, 0.0).astype(np.float32)
    return shade, depth


def _true_synthetic_pose(
    cx: float = 0.65, cy: float = 0.75, cz: float = 1.65, yaw: float = 125.0
) -> tuple[spots.Aim, np.ndarray]:
    from liberational.spatial import plan_import

    up = tuple(float(v) for v in spots.up_at(14.0, -2.0))
    aim = spots.Aim((up[0], up[1], up[2]), 320, 180, 30, 1.72, 50.0, f"{cz}|90.0|")
    cam = plan_import.Camera("front", (cx, cy), cz, yaw, 90.0)
    matrix = spots.pose(cam, aim, mirrored=True)
    assert matrix is not None
    return aim, np.asarray(matrix, dtype=np.float64).reshape(4, 4)


def _setup_synthetic_place(
    state: Path,
    rgb: np.ndarray,
    cx: float,
    cy: float,
    cz: float = 1.65,
    yaw: float = 125.0,
    pitch: float = 14.0,
    roll: float = -2.0,
    t: float = 1.0,
) -> tuple[str, frames.Kept]:
    import cv2

    plan_text = _synthetic_plan(cx, cy, cz, yaw)
    place = places.revise(state, None, producer="places reset v1", now=10.0).place
    places.correct_plan(
        state, place, plan_text, now=20.0, view={"plan": plan_text, "turn": 0.0, "shift": [0.0, 0.0], "mirror": True}
    )
    up = tuple(float(v) for v in spots.up_at(pitch, roll))
    aim = spots.Aim((up[0], up[1], up[2]), rgb.shape[1], rgb.shape[0], 30, 1.72, 50.0, f"{cz}|90.0|")
    spots.save_aims(state, {"front": aim})
    store = frames.Store(frames.root(state))
    _, kept = store.offer("front", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), unit(0), 100.0, now=t)
    assert kept is not None
    return place, kept


def test_spot_verification_tells_the_true_pose_from_shifted_and_turned_ones_despite_archways_furniture_and_depth_warp():
    from liberational.spatial import plan_import, spot_check

    plan_text = _synthetic_plan()
    parsed = plan_import.parse(plan_text)
    _, true_c2w = _true_synthetic_pose()
    focal = 160.0
    rgb, depth = _render_synthetic_room(plan_text, true_c2w, focal=focal)

    # Naive wall-band proximity (`spot_fit` given raw `Walls`) cannot tell the true pose from a shifted one: floor
    # scaling pulls warped walls short at the true pose (~6% near), while shifting +0.45 m in x and +0.30 m in y lands
    # those shortened walls and the sofa onto the plan's walls and cased partition line.
    wrong_shift = spot_check.offset_pose(true_c2w, dx=0.45, dy=0.30)
    old_walls = plan_import.walls(parsed)
    naive_true = sm.spot_fit(depth, focal, true_c2w, old_walls)
    naive_wrong = sm.spot_fit(depth, focal, wrong_shift, old_walls)
    assert (naive_wrong.wall_near or 0.0) > (naive_true.wall_near or 0.0) * 2

    # `spot_fit` given the parsed plan (`refine=False` to inspect the raw landscape) peaks clearly at the true pose
    # across shifts and turns, and stays idempotent after `_scale_depth` scales the depth to the plan's floor.
    at_true = sm.spot_fit(depth, focal, true_c2w, parsed, image=rgb, refine=False)
    assert at_true.score is not None
    for candidate in (
        wrong_shift,
        spot_check.offset_pose(true_c2w, dx=0.60),
        spot_check.offset_pose(true_c2w, dx=-0.50),
        spot_check.offset_pose(true_c2w, dy=0.60),
        spot_check.offset_pose(true_c2w, dy=-0.50),
        spot_check.offset_pose(true_c2w, dyaw=14.0),
        spot_check.offset_pose(true_c2w, dyaw=-14.0),
        spot_check.offset_pose(true_c2w, dpitch=8.0),
        spot_check.offset_pose(true_c2w, droll=8.0),
    ):
        off = sm.spot_fit(depth, focal, candidate, parsed, image=rgb, refine=False)
        assert off.score is not None and at_true.score > off.score + 0.05

    scaled_depth = depth * float(at_true.scale)
    again = sm.spot_fit(scaled_depth, focal, true_c2w, parsed, image=rgb, refine=False)
    assert again.score == pytest.approx(at_true.score, abs=0.01)


def test_a_spot_within_the_refinement_bound_is_refined_toward_the_true_pose_while_the_owners_spot_stays_authoritative(
    models, state
):
    from liberational.spatial import spot_check

    _, true_c2w = _true_synthetic_pose(0.65, 0.75, 1.65, 125.0)
    rgb, depth = _render_synthetic_room(_synthetic_plan(), true_c2w, focal=160.0)
    models.depth = lambda h, w, f: depth

    # The owner's spot is 0.19 m and ~3.4 deg off the true camera pose: inside `(spots.REFINE_M, spots.REFINE_DEG)`.
    place, _ = _setup_synthetic_place(state, rgb, cx=0.81, cy=0.65, cz=1.65, yaw=128.2, pitch=15.0, roll=-2.0)
    owner_spot = spots.read(state)["front"]
    assert owner_spot.pose is not None
    init_shift, init_turn = spot_check.pose_delta(owner_spot.pose.pose, true_c2w)
    assert 0.15 < init_shift < spots.REFINE_M and 2.5 < init_turn < spots.REFINE_DEG

    did = sm.step(state, now=100.0, held={})
    assert did is not None
    fit_info = did["spots"]["front"]
    assert fit_info["refined"] is True
    assert "suggestion" not in fit_info

    after = spots.read(state)["front"]
    assert after.pose == owner_spot.pose  # the owner's spot itself is untouched
    assert after.refined is not None
    assert spots.within_bounds(after.pose, after.refined)
    ref_shift, ref_turn = spot_check.pose_delta(after.refined.pose, true_c2w)
    assert ref_shift < init_shift and ref_turn < init_turn

    # `spots.overlay` and the placed view use the within-bound refined pose, and a second step does not mistake the
    # refined pose for an owner edit of the spot.
    assert spots.overlay(newest(state, place), spots.read(state)).cameras["front"].pose == after.refined.pose
    assert sm.views(state, place)[0].pose == after.refined.pose
    assert sm.summary(state, now=110.0)["spot_suggestions"] == {}
    assert sm.step(state, now=120.0, held={}) is None

    # Moving the spot on the plan immediately drops the old refinement.
    moved = _synthetic_plan(cx=0.65, cy=0.75, cz=1.65, yaw=125.0)
    places.correct_plan(
        state, place, moved, now=130.0, view={"plan": moved, "turn": 0.0, "shift": [0.0, 0.0], "mirror": True}
    )
    assert spots.read(state)["front"].refined is None


def test_a_spot_off_past_the_bound_holds_where_the_owner_put_it_and_surfaces_a_suggestion_until_moved(models, state):
    _, true_c2w = _true_synthetic_pose(0.65, 0.75, 1.65, 125.0)
    rgb, depth = _render_synthetic_room(_synthetic_plan(), true_c2w, focal=160.0)
    models.depth = lambda h, w, f: depth

    # The owner's spot is ~0.58 m and 12 deg off: well beyond `(0.3 m, 5 deg)`.
    place, _ = _setup_synthetic_place(state, rgb, cx=1.15, cy=1.05, cz=1.65, yaw=137.0, pitch=14.0, roll=-2.0)
    owner_spot = spots.read(state)["front"]
    assert owner_spot.pose is not None

    did = sm.step(state, now=100.0, held={})
    assert did is not None
    fit_info = did["spots"]["front"]
    assert fit_info["refined"] is False
    assert fit_info["suggestion"]["score"] > fit_info["at_spot"]
    assert fit_info["suggestion"]["shift_m"] > spots.REFINE_M or fit_info["suggestion"]["turn_deg"] > spots.REFINE_DEG

    # The owner's spot stays the authority: neither the placed view nor `spots.read` moves past it.
    assert sm.views(state, place)[0].pose == owner_spot.pose.pose
    assert newest(state, place).cameras["front"].pose == owner_spot.pose.pose
    assert spots.read(state)["front"].refined is None

    # The Places summary surfaces the suggestion until the owner updates the spot on the plan.
    summary = sm.summary(state, now=110.0)
    assert "front" in summary["spot_suggestions"]
    moved = _synthetic_plan(cx=0.65, cy=0.75, cz=1.65, yaw=125.0)
    places.correct_plan(
        state, place, moved, now=120.0, view={"plan": moved, "turn": 0.0, "shift": [0.0, 0.0], "mirror": True}
    )
    assert sm.summary(state, now=125.0)["spot_suggestions"] == {}
