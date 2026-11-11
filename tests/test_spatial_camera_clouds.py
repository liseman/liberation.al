"""Each camera's own point cloud (spatial/camera_clouds.py): what a point map becomes, and how it's kept. The model
never runs here: a point map is made up by casting a made-up room's rays."""

import subprocess

import numpy as np
import pytest

from liberational.spatial import camera_clouds as clouds
from liberational.spatial import ply

T = 1_762_500_000.0
W, H, FOCAL = 96, 64, 70.0
PITCH = np.radians(25.0)  # the made-up camera looks this far below level


def room(height: float = 1.5, wall: float = 4.0, floor: bool = True) -> clouds.Pointmap:
    """What a camera `height` above a floor, pitched down, sees of that floor and of a wall `wall` ahead of it, as a
    point for every pixel in its own frame (x right, y down, z forward). Without `floor` the rays that would reach
    it stop at a side wall instead, so the view has depth and no floor."""
    v, u = np.mgrid[0:H, 0:W]
    rays = np.stack([(u - W / 2) / FOCAL, (v - H / 2) / FOCAL, np.ones((H, W))], axis=2)
    down = np.array([0.0, np.cos(PITCH), np.sin(PITCH)])  # the room's down, in the camera's frame
    ahead = np.array([0.0, -np.sin(PITCH), np.cos(PITCH)])  # level ahead
    to_wall = wall / (rays @ ahead)
    falling = rays @ down
    to_floor = np.where(falling > 1e-6, height / np.where(falling > 1e-6, falling, 1.0), np.inf)
    if not floor:
        to_floor = np.where(rays[..., 0] > 1e-6, 1.5 / np.where(rays[..., 0] > 1e-6, rays[..., 0], 1.0), np.inf)
    reach = np.minimum(to_wall, to_floor)
    colour = np.where((to_floor < to_wall)[..., None], [90, 60, 30], [200, 200, 190]).astype(np.uint8)
    return clouds.Pointmap(
        points=(rays * reach[..., None]).astype(np.float32),
        confidence=np.full((H, W), 3.0, dtype=np.float32),
        colour=colour,
        covered=np.zeros((H, W), dtype=bool),
        seconds=0.2,
        peak_gpu_gb=2.9,
    )


def points_of(made: clouds.Made) -> np.ndarray:
    rows = ply.vertices(made.ply)
    assert rows is not None
    return np.c_[rows["x"], rows["y"], rows["z"]].astype(np.float64)


def test_a_cloud_stands_level_on_its_floor_with_the_camera_above_it():
    made = clouds.make("hall", room(height=1.5), None, T)
    cloud = made.cloud
    pose = np.array(cloud.pose).reshape(4, 4)
    assert (cloud.size, cloud.floor_found) == ("approximate", True)
    assert pose[:3, 3] == pytest.approx([0.0, 0.0, 1.5], abs=0.02)  # the camera, on the z axis, as high as it hangs
    assert pose[:3, 2] == pytest.approx([0.0, np.cos(PITCH), -np.sin(PITCH)], abs=0.02)  # looking along +y, and down
    z = points_of(made)[:, 2]
    assert z.min() == pytest.approx(0.0, abs=0.03)  # nothing under the floor
    assert np.mean(np.abs(z) < 0.03) > 0.3  # and the floor's own points on it


def test_a_measured_mounting_height_doesnt_size_the_cloud():
    made = clouds.make("hall", room(height=1.5, wall=4.0), 2.0, T)  # hung at 2 m; the model reads 1.5
    assert made.cloud.size == "approximate" and made.cloud.floor_found
    assert np.array(made.cloud.pose).reshape(4, 4)[2, 3] == pytest.approx(1.5, abs=0.02)  # the model's own metres
    assert points_of(made)[:, 1].max() == pytest.approx(4.0, abs=0.1)  # and the wall where the model put it


def test_a_plane_far_from_the_cameras_measured_height_isnt_taken_for_its_floor():
    # a camera hung at 6 m whose "floor" reads 1.5 m under it is looking at a table top, not its floor
    made = clouds.make("hall", room(height=1.5), 6.0, T)
    assert made.cloud.floor_found is False


def test_a_surface_just_under_the_camera_isnt_its_floor():
    made = clouds.make("hall", room(height=0.3), None, T)  # a shelf the camera stands over
    assert made.cloud.floor_found is False


def test_a_view_with_no_floor_in_it_stays_approximate_even_with_a_height():
    made = clouds.make("hall", room(floor=False), 3.0, T)
    assert (made.cloud.size, made.cloud.floor_found) == ("approximate", False)
    assert points_of(made)[:, 2].min() == pytest.approx(0.0, abs=0.2)  # the lowest points stand in for the floor


def test_overlay_pixels_and_unsure_pixels_give_no_point():
    got = room()
    got.covered[:8] = True  # a clock burned into the top eighth
    got.confidence[:, :12] = 1.0  # and a strip the model isn't sure of
    made = clouds.make("hall", got, None, T)
    assert made.cloud.points == (H - 8) * (W - 12)
    assert made.cloud.share == pytest.approx((H - 8) * (W - 12) / (H * W), abs=0.001)


def test_a_picture_the_model_is_unsure_of_gives_no_cloud():
    got = room()
    got.confidence[:, : int(W * 0.8)] = 1.0
    with pytest.raises(clouds.Rejected) as refused:
        clouds.make("hall", got, None, T)
    assert refused.value.reason == "unsure"


def test_a_picture_with_no_depth_gives_no_cloud():
    got = room()
    got.points[..., 2] = 2.0  # everything the same distance away: a picture of a picture
    with pytest.raises(clouds.Rejected) as refused:
        clouds.make("hall", got, None, T)
    assert refused.value.reason == "flat"


@pytest.fixture
def store(tmp_path):
    return clouds.Store(clouds.root(tmp_path))


def run_room(picture, overlay, others=()):
    return room()


def test_a_new_cloud_replaces_the_cameras_old_one_whole(store, tmp_path):
    manifest = tmp_path / "devices.toml"
    first = clouds.refresh(store, "hall", np.zeros((36, 64, 3), np.uint8), T, run=run_room, manifest=manifest)
    second = clouds.refresh(store, "hall", np.zeros((36, 64, 3), np.uint8), T + 600, run=run_room, manifest=manifest)
    assert first is not None and second is not None and first.id != second.id
    assert store.current("hall") == second
    assert [p.name for p in (store.folder / "hall").glob("*.ply")] == [f"{second.id}.ply"]
    assert store.cameras() == ["hall"]


def refuse(error):
    def run(picture, overlay, others=()):
        raise error

    return run


@pytest.mark.parametrize(
    ("run", "reason"),
    [
        pytest.param(None, "no_runtime", id="no-runtime"),
        pytest.param(refuse(clouds.NoGpu("1.0 GB free")), "gpu", id="no-gpu-memory"),
        pytest.param(refuse(RuntimeError("the runner failed")), "failed", id="runner-failed"),
    ],
)
def test_a_try_that_makes_nothing_says_why_and_leaves_the_cloud_there_was(store, tmp_path, run, reason):
    manifest = tmp_path / "devices.toml"
    kept = clouds.refresh(store, "hall", np.zeros((36, 64, 3), np.uint8), T, run=run_room, manifest=manifest)
    assert clouds.refresh(store, "hall", np.zeros((36, 64, 3), np.uint8), T + 600, run=run, manifest=manifest) is None
    assert store.notes()["hall"]["reason"] == reason
    assert store.current("hall") == kept


def test_a_made_cloud_clears_the_note_of_the_try_before(store, tmp_path):
    manifest = tmp_path / "devices.toml"
    clouds.refresh(store, "hall", np.zeros((36, 64, 3), np.uint8), T, run=None, manifest=manifest)
    clouds.refresh(store, "hall", np.zeros((36, 64, 3), np.uint8), T + 60, run=run_room, manifest=manifest)
    assert store.notes() == {}


def test_the_model_gets_the_picture_with_the_cameras_overlays_marked(store, tmp_path):
    manifest = tmp_path / "devices.toml"
    manifest.write_text(
        '[[device]]\nname = "hall"\nkind = "rtsp"\nmasks = [[0.0, 0.0, 0.5, 0.25]]\nmount = { height_m = 2.25 }\n'
    )
    seen = {}

    def run(picture, overlay, others=()):
        seen["overlay"] = overlay
        return room(height=1.5)

    cloud = clouds.refresh(store, "hall", np.full((36, 64, 3), 120, np.uint8), T, run=run, manifest=manifest)
    assert seen["overlay"][:9, :32].all() and not seen["overlay"][9:].any() and not seen["overlay"][:, 32:].any()
    assert cloud is not None


def test_a_new_picture_is_compared_with_the_neighbours_in_the_same_run_and_kept_for_the_next(store, tmp_path):
    from liberational.spatial import cloud_scene

    manifest = tmp_path / "devices.toml"
    place = "11111111-1111-7111-8111-111111111111"
    spots = {
        name: cloud_scene.SpotAt(name, place, cloud_scene.At(x, 0.0, 0.0), (x, 0.0), False, name)
        for name, x in (("hall", 0.0), ("porch", 2.0))
    }
    asked = []

    def run(picture, overlay, others=()):
        asked.append([(path.name, first) for path, first in others])
        return clouds.replace(room(), matches=tuple({"match": [0.0, 0.0], "sure": [0.0, 0.0]} for _ in others))

    blank = np.full((36, 64, 3), 90, np.uint8)
    hall = clouds.refresh(store, "hall", blank, T, run=run, manifest=manifest, spots=lambda: spots)
    porch = clouds.refresh(store, "porch", blank, T + 60, run=run, manifest=manifest, spots=lambda: spots)
    again = clouds.refresh(store, "hall", blank, T + 120, run=run, manifest=manifest, spots=lambda: spots)
    assert hall is not None and porch is not None and again is not None
    # the first has nothing to compare with; the second, the first's picture; the third, its own anchor (as the
    # picture seen from) and the other camera's
    assert asked == [[], [(f"{hall.id}.png", False)], [("anchor.png", True), (f"{porch.id}.png", False)]]
    assert store.picture(again).is_file() and not store.picture(hall).is_file()  # one picture a cloud
    assert list(cloud_scene.pairs(store)) == ["hall|porch"]


def test_a_camera_whose_lens_calibration_cant_be_used_is_skipped_with_the_reason(store, tmp_path):
    manifest = tmp_path / "devices.toml"
    manifest.write_text(
        '[[device]]\nname = "hall"\nkind = "rtsp"\n'
        "lens = { width = 64, height = 36, k = [32.0, 32.0, 32.0, 18.0], division = -5.0 }\n"
    )
    assert clouds.refresh(store, "hall", np.zeros((36, 64, 3), np.uint8), T, run=run_room, manifest=manifest) is None
    assert store.notes()["hall"]["reason"] == "lens"


def test_the_gpu_with_the_most_free_memory_is_the_one_used(monkeypatch):
    def smi(*args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout="0, 2048\n1, 12288\n", stderr="")

    monkeypatch.setattr(clouds.subprocess, "run", smi)
    assert clouds.free_gpu() == (1, pytest.approx(12.0))


def test_a_pass_is_skipped_when_no_gpu_has_the_memory(monkeypatch, tmp_path):
    monkeypatch.setattr(clouds, "free_gpu", lambda: (0, 1.5))
    started = []
    monkeypatch.setattr(clouds.subprocess, "run", lambda *a, **k: started.append(a))
    found = clouds.Runtime(tmp_path / "python", tmp_path, tmp_path)
    with pytest.raises(clouds.NoGpu):
        clouds.pointmap(found, np.zeros((36, 64, 3), np.uint8), None)
    assert started == []  # the model was never started


def test_no_cloud_is_made_until_a_runtime_is_configured(monkeypatch, tmp_path):
    from liberational import config as cfg

    monkeypatch.setattr(cfg, "get", lambda key, default=None: {"spatial.pointmap_runtime": ""}.get(key, default))
    assert clouds.runtime() is None and clouds.runner() is None
