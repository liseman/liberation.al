"""Learning surroundings (#258): placing kept frames in a place's map and continuing its splat."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from liberational import config as cfg
from liberational.memory import places
from liberational.spatial import cameras, lens, ply, splat_runner, splats
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
        for view in json.loads(Path(request).read_text())["views"]:
            np.save(view["out"], np.full((36, 64), 2.0, dtype=np.float32))

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


def keep(state: Path, angle: float, t: float, camera: str = "front") -> frames.Kept:
    """Put a made-up frame in the kept store, as the sampler would have."""
    store = frames.Store(frames.root(state))
    picture = np.full((36, 64, 3), 90, dtype=np.uint8)
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
