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


def test_a_cloud_stands_level_by_its_floor_and_says_its_cameras_tilt():
    from liberational.spatial import cloud_align

    made = clouds.make("hall", room(), None, T)
    assert made.cloud.level == "floor"
    tilt, roll = cloud_align.tilt_roll(np.array(made.cloud.pose).reshape(4, 4))
    assert (tilt, roll) == (pytest.approx(25.0, abs=0.3), pytest.approx(0.0, abs=0.3))


def test_a_view_with_no_floor_in_it_is_stood_upright_by_its_walls():
    from liberational.spatial import cloud_align

    made = clouds.make("hall", room(floor=False), None, T)
    assert (made.cloud.floor_found, made.cloud.level) == (False, "walls")
    tilt, roll = cloud_align.tilt_roll(np.array(made.cloud.pose).reshape(4, 4))
    assert (tilt, roll) == (pytest.approx(25.0, abs=0.5), pytest.approx(0.0, abs=0.5))  # the camera's real pitch
    own = points_of(made)
    ahead = own[own[:, 0] < 1.0]  # the wall ahead: upright, so every point of it is as far ahead
    assert ahead[:, 1] == pytest.approx(4.0, abs=0.05)


@pytest.mark.parametrize(
    ("new", "old", "expected"),
    [
        (("floor", 1000), ("floor", 1000), True),
        (("floor", 900), ("floor", 1000), True),  # as many points, near enough
        (("floor", 700), ("floor", 1000), False),
        (("none", 5000), ("floor", 1000), False),  # no floor found: worse, however many points
        (("walls", 5000), ("floor", 1000), False),
        (("walls", 1000), ("none", 1000), True),
        (("floor", 100), ("walls", 1000), True),
    ],
)
def test_a_cloud_is_better_for_a_floor_then_for_walls_then_for_its_points(new, old, expected):
    def cloud(level: str, points: int) -> clouds.Cloud:
        return clouds.Cloud("00000000-0000-7000-8000-000000000001", "hall", T, points, 0.8, "approximate",
                            level == "floor", tuple(np.eye(4).ravel()), 70.0, 64, 36, level=level)  # fmt: skip

    assert clouds.better(cloud(*new), cloud(*old)) is expected


def test_an_older_cloud_counts_as_levelled_by_its_floor_or_by_nothing():
    old = clouds.Cloud.from_dict({"id": "a", "camera": "hall", "at": T, "points": 3, "share": 0.8, "size": "approximate",
                                  "floor_found": False, "pose": list(np.eye(4).ravel()), "focal": 70.0, "width": 64,
                                  "height": 36})  # fmt: skip
    assert (old.level, old.levelled) == ("", "none")
    assert clouds.replace(old, floor_found=True).levelled == "floor"


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


SAME_PLACE = {"match": [0.22, 0.11], "sure": [0.6, 0.25], "rotation": [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0],
              "shift": [0.0, 0.0, 0.0], "scale": 1.0, "disagree": [1.0, 0.05, 0.01]}  # fmt: skip
ELSEWHERE = {"match": [0.0, 0.05], "sure": [0.0, 0.0]}


def spotted(camera: str = "hall"):
    from liberational.spatial import cloud_scene

    place = "11111111-1111-7111-8111-111111111111"
    return {camera: cloud_scene.SpotAt(camera, place, cloud_scene.At(0.0, 0.0, 0.0), (0.0, 0.0), False, camera)}


def test_a_dark_look_with_no_floor_doesnt_replace_the_cloud_of_a_camera_that_stayed_put(store, tmp_path):
    manifest, blank = tmp_path / "devices.toml", np.full((36, 64, 3), 90, np.uint8)
    spots = spotted()

    def by_night(picture, overlay, others=()):
        return clouds.replace(room(floor=False, wall=3.0), matches=tuple(SAME_PLACE for _ in others))

    day = clouds.refresh(store, "hall", blank, T, run=run_room, manifest=manifest, spots=lambda: spots)
    night = clouds.refresh(store, "hall", blank, T + 600, run=by_night, manifest=manifest, thumb=(0.5, 0.5),
                           spots=lambda: spots)  # fmt: skip
    assert day is not None and night is not None
    assert (night.id, night.level, night.at) == (day.id, "floor", T)  # the daylight cloud is the one kept
    assert night.also == (0.5, 0.5)  # and the night view is known, so it isn't tried again
    assert store.current("hall") == night
    assert [p.name for p in (store.folder / "hall").glob("*.ply")] == [f"{day.id}.ply"]


def test_a_worse_cloud_does_replace_the_old_one_when_the_camera_no_longer_sees_what_it_did(store, tmp_path):
    manifest, blank = tmp_path / "devices.toml", np.full((36, 64, 3), 90, np.uint8)
    spots = spotted()

    def elsewhere(picture, overlay, others=()):
        return clouds.replace(room(floor=False, wall=3.0), matches=tuple(ELSEWHERE for _ in others))

    day = clouds.refresh(store, "hall", blank, T, run=run_room, manifest=manifest, spots=lambda: spots)
    moved = clouds.refresh(store, "hall", blank, T + 600, run=elsewhere, manifest=manifest, spots=lambda: spots)
    assert day is not None and moved is not None
    assert (moved.id != day.id, moved.level) == (True, "walls")
    assert store.current("hall") == moved


def test_a_cloud_of_a_view_that_wasnt_placed_gives_way_when_the_old_view_comes_back_however_good_it_was(
    store, tmp_path
):
    manifest, blank = tmp_path / "devices.toml", np.full((36, 64, 3), 90, np.uint8)
    spots = spotted()

    def elsewhere(picture, overlay, others=()):
        return clouds.replace(room(), matches=tuple(ELSEWHERE for _ in others))

    def back_by_night(picture, overlay, others=()):
        return clouds.replace(room(floor=False, wall=3.0), matches=tuple(SAME_PLACE for _ in others))

    clouds.refresh(store, "hall", blank, T, run=run_room, manifest=manifest, spots=lambda: spots)
    away = clouds.refresh(store, "hall", blank, T + 600, run=elsewhere, manifest=manifest, spots=lambda: spots)
    back = clouds.refresh(store, "hall", blank, T + 1200, run=back_by_night, manifest=manifest, spots=lambda: spots)
    assert away is not None and back is not None
    # the look back has no floor, so it is the worse cloud; but the better one is of another view, so it goes
    assert (back.id != away.id, back.level, back.at) == (True, "walls", T + 1200)
    assert store.current("hall") == back


def test_a_camera_with_no_spot_keeps_its_newest_cloud(store, tmp_path):
    manifest, blank = tmp_path / "devices.toml", np.full((36, 64, 3), 90, np.uint8)
    day = clouds.refresh(store, "hall", blank, T, run=run_room, manifest=manifest)
    night = clouds.refresh(store, "hall", blank, T + 600, run=lambda *a: room(floor=False), manifest=manifest)
    assert day is not None and night is not None and night.id != day.id


def old_cloud(store, points: np.ndarray, floor_found: bool) -> clouds.Cloud:
    """A cloud as one made before walls were used was kept: no `level`, its camera upright at 1.5 m."""
    pose = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.0)
    cloud = clouds.Cloud("00000000-0000-7000-8000-000000000009", "hall", T, len(points), 0.8, "approximate",
                         floor_found, pose, 70.0, 64, 36)  # fmt: skip
    store.replace(
        clouds.Made(cloud, ply.dumps(points, np.full((len(points), 3), 100, np.uint8)), np.zeros((36, 64, 3), np.uint8))
    )
    return cloud


def leaning_walls(degrees: float) -> np.ndarray:
    """Two walls that meet, 3 m ahead and 2 m to the right of a camera at 1.5 m, tipped `degrees` about the camera's
    own right: what a cloud holds when its tilted camera was taken to be upright."""
    rng = np.random.default_rng(5)
    ahead = np.c_[rng.random(3000) * 4.0 - 2.0, np.full(3000, 3.0), rng.random(3000) * 2.4]
    right = np.c_[np.full(3000, 2.0), rng.random(3000) * 3.0, rng.random(3000) * 2.4]
    t = np.radians(degrees)
    tip = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(t), -np.sin(t)], [0.0, np.sin(t), np.cos(t)]])
    eye = np.array([0.0, 0.0, 1.5])
    return (np.concatenate([ahead, right]) - eye) @ tip.T + eye


def test_an_older_cloud_with_no_floor_is_stood_up_by_its_walls_once_as_a_new_cloud(store):
    from liberational.spatial import cloud_align

    was = old_cloud(store, leaning_walls(12.0), floor_found=False)
    stood = store.relevel("hall")
    assert stood is not None and stood.id != was.id
    assert (stood.level, stood.at, stood.points) == ("walls", was.at, was.points)
    assert cloud_align.tilt_roll(np.array(stood.pose).reshape(4, 4))[0] == pytest.approx(12.0, abs=0.5)
    own, _, _ = ply.read(store.file(stood))
    assert own[own[:, 0] < 1.9][:, 1] == pytest.approx(3.0, abs=0.05)  # the wall ahead stands upright again
    assert store.picture(stood).is_file()  # its picture goes with it
    assert store.current("hall") == stood
    assert store.relevel("hall") is None  # once


def test_an_older_cloud_with_a_floor_or_with_no_wall_is_only_marked(store):
    was = old_cloud(store, leaning_walls(0.0), floor_found=True)
    assert store.relevel("hall") is None
    assert store.current("hall") == clouds.replace(was, level="floor")
    flat = np.c_[np.random.default_rng(1).random((3000, 2)) * 4.0, np.zeros(3000)]
    was = old_cloud(store, flat, floor_found=False)
    assert store.relevel("hall") is None
    assert store.current("hall") == clouds.replace(was, level="none")


def test_a_new_cloud_leaves_the_anchor_picture_where_it_is(store):
    old_cloud(store, leaning_walls(0.0), floor_found=True)
    anchor = store.folder / "hall" / "anchor.png"
    anchor.write_bytes(b"png")
    old_cloud(store, leaning_walls(0.0), floor_found=True)
    assert anchor.is_file()


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
