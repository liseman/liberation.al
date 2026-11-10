"""Fixed cameras registered by reprojection, each with its own lens (#3069): a synthetic room seen by a pinhole camera
and a wide one with a barrel, as MASt3R's pair inference would hand them over."""

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

from liberational.memory import places
from liberational.spatial import registration as R

T = 1_800_000_000.0
REF_SIZE, CAM_SIZE = (512, 288), (288, 512)  # landscape and portrait, as cam0 and living-room on Place 1
REF_F, CAM_F, LAMBDA = 0.66 * 512, 1.13 * 288, -0.3


def looking(centre, target, roll_deg):
    """Camera to reference, for a camera at `centre` looking at `target` (x right, y down, z forward; up is -y),
    rolled about its view."""
    z = np.subtract(target, centre) / np.linalg.norm(np.subtract(target, centre))
    x = np.cross(z, [0.0, -1.0, 0.0])
    x /= np.linalg.norm(x)
    rot = np.c_[x, np.cross(z, x), z] @ Rotation.from_euler("z", roll_deg, degrees=True).as_matrix()
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = rot, centre
    return out


# The wide camera in the reference's frame: 4 m to the right, 0.8 m higher, 2 m ahead, looking back across the room
# and down, rolled 15 degrees.
CAM_TO_REF = looking((4.0, -0.8, 2.0), (0.5, 0.6, 4.0), 15.0)


def room(seed=0, n=6000):
    """Points spread through the room both cameras see, in the reference's frame (it looks along +z)."""
    rng = np.random.default_rng(seed)
    return np.c_[rng.uniform(-2.5, 3.5, n), rng.uniform(-1.5, 1.5, n), rng.uniform(1.5, 6.0, n)]


def ref_pixel(p):
    return p[:, :2] / p[:, 2:3] * REF_F + np.array(REF_SIZE) / 2


def cam_pixel(p, division=LAMBDA):
    c = (p - CAM_TO_REF[:3, 3]) @ CAM_TO_REF[:3, :3]
    x, y = R.distort(c[:, 0] / c[:, 2], c[:, 1] / c[:, 2], division)
    return np.c_[x, y] * CAM_F + np.array(CAM_SIZE) / 2, c[:, 2]


def inside(px, size):
    return (px[:, 0] >= 0) & (px[:, 1] >= 0) & (px[:, 0] < size[0]) & (px[:, 1] < size[1])


def pair(points, flip=False):
    """The pair as the runner saves it from one camera's side: that camera's own point map (depth at the matched
    pixels), the other camera's point map in its frame (a pinhole's rays at the true distance, as the network gives
    a barrel's pixels), and the matches (pixel indices)."""
    a_px, (b_px, b_depth) = ref_pixel(points), cam_pixel(points)
    ok = inside(a_px, REF_SIZE) & inside(b_px, CAM_SIZE) & (points[:, 2] > 0) & (b_depth > 0)
    pts, a_px, b_px = points[ok], a_px[ok], b_px[ok]
    a_xy, b_xy = np.floor(a_px).astype(int), np.floor(b_px).astype(int)
    in_cam = (pts - CAM_TO_REF[:3, 3]) @ CAM_TO_REF[:3, :3]
    if flip:  # from the wide camera's side: its own depth
        own_size, other_size, own_xy, other_xy, own_pts = CAM_SIZE, REF_SIZE, b_xy, a_xy, in_cam
    else:
        own_size, other_size, own_xy, other_xy, own_pts = REF_SIZE, CAM_SIZE, a_xy, b_xy, pts
    own = np.zeros((own_size[1], own_size[0], 3), np.float32)
    own[own_xy[:, 1], own_xy[:, 0]] = own_pts
    other = np.zeros((other_size[1], other_size[0], 3), np.float32)
    other_conf = np.zeros(other.shape[:2], np.float32)
    if not flip:  # the wide camera's pixels along a pinhole's rays at their true distance from it
        centred = other_xy + 0.5 - np.array(CAM_SIZE) / 2
        rays = np.c_[centred / CAM_F, np.ones(len(centred))]
        rays = rays / np.linalg.norm(rays, axis=1, keepdims=True) * np.linalg.norm(in_cam, axis=1, keepdims=True)
        other[other_xy[:, 1], other_xy[:, 0]] = rays @ CAM_TO_REF[:3, :3].T + CAM_TO_REF[:3, 3]
        other_conf[other_xy[:, 1], other_xy[:, 0]] = 3.0
    own_conf = (np.linalg.norm(own, axis=2) > 0).astype(np.float32) * 3.0
    return R.Pair(own, own_conf, other, other_conf, own_xy, other_xy, np.full(len(own_xy), 5.0, np.float32))


# MASt3R's own answer: roughly right rotations, focals a little off (the fit starts there)
NEAR = Rotation.from_rotvec([0.03, -0.02, 0.04]).as_matrix()
MAST3R_POSES = {"cam0": np.eye(4), "wide": np.r_[np.c_[CAM_TO_REF[:3, :3] @ NEAR, CAM_TO_REF[:3, 3]], [[0, 0, 0, 1]]]}
MAST3R_FOCALS = {"cam0": 0.74 * 512, "wide": 1.05 * 288}
SIZES = {"cam0": list(REF_SIZE), "wide": list(CAM_SIZE)}


def images():
    return {"cam0": np.zeros((288, 512, 3), np.uint8), "wide": np.zeros((512, 288, 3), np.uint8)}


def test_the_fit_finds_the_pose_focals_and_barrel_of_the_wide_camera():
    start = np.linalg.inv(MAST3R_POSES["wide"])[:3, :3]
    got = R.fit(pair(room()), REF_SIZE, CAM_SIZE, start, (MAST3R_FOCALS["cam0"], MAST3R_FOCALS["wide"]))
    assert got is not None
    assert got.ref_focal == pytest.approx(REF_F, rel=0.01)
    assert got.focal == pytest.approx(CAM_F, rel=0.01)
    assert got.division == pytest.approx(LAMBDA, abs=0.02)
    turn = Rotation.from_matrix(got.rotation @ CAM_TO_REF[:3, :3]).magnitude()
    assert math.degrees(turn) < 0.2
    centre = -got.rotation.T @ got.shift  # the wide camera's centre in the reference's frame
    assert np.linalg.norm(centre - CAM_TO_REF[:3, 3]) < 0.02
    assert got.residual < 1.0  # pixel indices are whole pixels


def test_as_a_pinhole_the_barrel_drags_the_reference_focal_off(monkeypatch):
    # #3069: without a lens for the wide camera the same fit puts the reference's focal where MASt3R did
    start = np.linalg.inv(MAST3R_POSES["wide"])[:3, :3]
    focals = (MAST3R_FOCALS["cam0"], MAST3R_FOCALS["wide"])
    with_lens = R.fit(pair(room()), REF_SIZE, CAM_SIZE, start, focals)
    monkeypatch.setattr(R, "MAX_RESIDUAL_PX", 1e9)
    pinhole = R.fit(pair(room()), REF_SIZE, CAM_SIZE, start, focals, lens=False)
    assert with_lens is not None and pinhole is not None
    assert pinhole.residual > 3 * with_lens.residual
    assert abs(pinhole.ref_focal / REF_F - 1) > 0.03


def test_too_few_matches_register_nothing():
    few = pair(room(n=60))
    assert R.fit(few, REF_SIZE, CAM_SIZE, np.eye(3), (300.0, 300.0)) is None


def test_the_pinhole_camera_is_the_reference_and_the_wide_ones_points_land_on_the_room():
    pts = room()
    pairs = {("cam0", "wide"): pair(pts), ("wide", "cam0"): pair(pts, flip=True)}
    got = R.register(["cam0", "wide"], pairs, SIZES, images(), MAST3R_POSES, MAST3R_FOCALS)
    assert got is not None and got.reference == "cam0"
    assert np.linalg.norm(got.poses["wide"][:3, 3] - CAM_TO_REF[:3, 3]) < 0.02
    assert got.divisions == {"cam0": 0.0, "wide": pytest.approx(LAMBDA, abs=0.02)}
    # The wide camera's points, re-cast along its true rays at the distance the network gave them, are the room's
    theirs = got.points[got.own :]
    assert len(theirs) > 100
    near, _ = KDTree(pts).query(theirs)
    assert np.median(near) < 0.02


def test_a_registered_wide_camera_puts_a_floor_pixel_on_its_floor_point():
    # The registered pose, focal, and lens as a place revision keeps them (spatial/lens.py's tuple, the undistorted
    # frame's focal), through places.floor_point
    floor = (0.0, -1.0, 0.0, 1.5)  # y down: the floor 1.5 m under the reference
    spot = np.array([[1.0, 1.5, 4.0]])
    px, _ = cam_pixel(spot)
    cam = R.camera_pose(CAM_TO_REF, CAM_F, LAMBDA, CAM_SIZE, T)
    assert cam.lens is not None
    hit = places.floor_point(cam, floor, px[0, 0] / CAM_SIZE[0], px[0, 1] / CAM_SIZE[1])
    assert hit == pytest.approx(tuple(spot[0]), abs=1e-6)
    assert R.camera_pose(np.eye(4), REF_F, 0.0, REF_SIZE, T).lens is None


def floor_scene(tilt_deg):
    """A floor 1.5 m under camera a and 2.25 m under b (b 3 m along x), its cloud's layer tilted `tilt_deg` about z
    so b seems higher or lower over it; up is +y here."""
    rng = np.random.default_rng(3)
    xz = rng.uniform([-1, 0], [5, 4], (8000, 2))
    a = math.radians(tilt_deg)
    layer = np.c_[xz[:, 0], -1.5 + math.tan(a) * xz[:, 0], xz[:, 1]]
    poses = {"a": np.eye(4), "b": np.eye(4)}
    poses["b"][:3, 3] = (3.0, 0.75, 0.0)
    return layer, poses


def test_the_floor_is_tilted_so_the_cameras_heights_keep_the_measured_ratio():
    layer, poses = floor_scene(1.0)  # a degree off: b stands 2.25 + 0.05 over it
    got = R.floor(layer, (0.0, 1.0, 0.0), poses, {"a": 1.52, "b": 2.28})
    assert got is not None
    n, d = np.array(got[:3]), got[3]
    heights = {c: float(n @ p[:3, 3] + d) for c, p in poses.items()}
    assert heights["b"] / heights["a"] == pytest.approx(2.28 / 1.52, rel=0.005)


def test_with_one_measured_height_the_floor_is_the_clouds():
    layer, poses = floor_scene(1.0)
    got = R.floor(layer, (0.0, 1.0, 0.0), poses, {"a": 1.52})
    assert got is not None
    assert math.degrees(math.acos(abs(got[1]))) == pytest.approx(1.0, abs=0.1)


def test_a_groups_saved_pairs_are_registered_as_the_runner_names_them(tmp_path):
    from liberational.spatial import cameras

    pts = room()
    arrays = {f"{c}__image": img for c, img in images().items()}
    for (a, b), p in {("cam0", "wide"): pair(pts), ("wide", "cam0"): pair(pts, flip=True)}.items():
        arrays |= {f"{a}__{b}__{name}": getattr(p, name) for name in R.PAIR_FIELDS}
    np.savez(tmp_path / "group0-pairs.npz", **arrays)
    group = {
        "cameras": ["cam0", "wide"],
        "pairs": "group0-pairs.npz",
        "poses": {c: m.ravel().tolist() for c, m in MAST3R_POSES.items()},
        "focals": MAST3R_FOCALS,
        "sizes": SIZES,
    }
    from PIL import Image

    mask = np.zeros((288, 512), np.uint8)
    mask[:, :100] = 255  # cam0's frame (as frames() saved it) with its left strip left out
    Image.fromarray(mask).save(tmp_path / "cam0.mask.png")
    got = cameras.reproject(tmp_path, group)
    assert got is not None and got.reference == "cam0"
    assert got.focals["wide"] == pytest.approx(CAM_F, rel=0.01)
    assert np.isnan(got.maps["cam0"][:, :100]).all(), "the mask beside the frame is left out"


def test_a_background_is_each_cells_surface_height_over_the_cameras():
    pose = np.eye(4)
    pose[:3, 3] = (0.0, 0.0, 2.0)  # 2 m over a z-up floor
    seen = np.zeros((16, 24, 3))
    seen[:, 8:, 2] = 0.9  # the middle and right thirds see a couch's back
    seen[:, 16:, 2] = 0.0  # the right third the floor again
    seen[8:, :8] = np.nan  # the bottom left cell: nothing sure
    seen[0, 0, 2] = 5.0  # one stray point a median outvotes
    got = R.background(seen, pose, (0.0, 0.0, 2.0, 0.0))  # a floor normal not of length 1
    assert got is not None and (got.cols, got.rows) == (3, 2)
    assert got.heights == (0.0, 0.45, 0.0, None, 0.45, 0.0)
    assert R.background(seen, np.eye(4), (0.0, 0.0, 1.0, 0.0)) is None, "a camera on the floor has no background"


def test_a_cameras_background_is_looked_up_through_the_lens_its_frames_were_undone_by():
    pose = np.eye(4)
    pose[:3, 3] = (0.0, 0.0, 2.0)
    seen = np.zeros((16, 24, 3))
    seen[:, 16:, 2] = 1.0  # the map's right third: something 1 m up
    # A lens that pushes every pixel of the frame to the map's right edge: every cell shows the right third
    got = R.background(seen, pose, (0.0, 0.0, 1.0, 0.0), lambda u, v: (np.full_like(u, 0.9), v))
    assert got is not None and set(got.heights) == {0.5}


def test_a_background_round_trips_and_finds_a_points_cell():
    bg = places.Background(2, 2, (0.0, 0.25, None, 0.5))
    assert places.Background.from_dict(bg.as_dict()) == bg
    assert (bg.at(0.1, 0.1), bg.at(0.9, 0.1), bg.at(0.1, 0.9), bg.at(0.9, 0.9)) == (0.0, 0.25, None, 0.5)
    assert bg.at(1.0, 0.5) is None and bg.at(-0.1, 0.5) is None
    pose = places.CameraPose(tuple(np.eye(4).ravel()), 300.0, 512, 288, T, None, bg)
    assert places.CameraPose.from_dict(pose.as_dict()) == pose
    with pytest.raises(ValueError, match="cols x rows"):
        places.Background.from_dict({"cols": 2, "rows": 2, "heights": [0.0]})


def test_a_cameras_masked_pixels_stay_out_of_its_cloud_and_background():
    # #3085: someone in the frame a camera is registered from isn't its furniture
    pts = room()
    pairs = {("cam0", "wide"): pair(pts), ("wide", "cam0"): pair(pts, flip=True)}
    clear = R.register(["cam0", "wide"], pairs, SIZES, images(), MAST3R_POSES, MAST3R_FOCALS)
    person = np.zeros((288, 512), bool)
    person[:, :256] = True  # the left half of cam0's frame
    got = R.register(["cam0", "wide"], pairs, SIZES, images(), MAST3R_POSES, MAST3R_FOCALS, covered={"cam0": person})
    assert clear is not None and got is not None
    assert np.isnan(got.maps["cam0"][:, :256]).all()
    assert np.isfinite(clear.maps["cam0"][:, :256]).any()
    assert got.own < clear.own and len(got.points) < len(clear.points)
    assert got.focals["wide"] == pytest.approx(clear.focals["wide"]), "the fit uses every match as before"


def test_a_fitted_cameras_background_map_is_its_own_sides_where_the_references_side_is_unsure():
    # #3220: from the reference's side, MASt3R was sure of 3% of cam0's pixels; from cam0's own side, of 83%. The own
    # side's inference has a scale of its own (here 2.5 times the reference side's), which the matches settle.
    pts = room()
    theirs, mine = pair(pts), pair(pts, flip=True)
    unsure = theirs.other_conf.copy()
    unsure[:, 40:] = 1.0  # sure only in a strip at the wide camera's left edge
    pairs = {
        ("cam0", "wide"): replace(theirs, other_conf=unsure),
        ("wide", "cam0"): replace(mine, own=mine.own * 2.5),
    }
    got = R.register(["cam0", "wide"], pairs, SIZES, images(), MAST3R_POSES, MAST3R_FOCALS)
    assert got is not None and got.reference == "cam0"
    sure_own = mine.own_conf > R.MIN_CONF
    assert (np.isfinite(got.maps["wide"][..., 0]) == sure_own).all(), "known wherever its own side is sure"
    near, _ = KDTree(pts).query(got.maps["wide"][sure_own])
    assert np.median(near) < 0.02, "and at the reference's scale"


# A camera's own intrinsics held (#3289)


def test_a_held_focal_and_lens_are_kept_and_only_the_pose_is_fitted():
    start = np.linalg.inv(MAST3R_POSES["wide"])[:3, :3]
    held = (1.10 * 288, -0.25)  # a little off the truth, as last week's fit might be
    got = R.fit(
        pair(room()),
        REF_SIZE,
        CAM_SIZE,
        start,
        (MAST3R_FOCALS["cam0"], MAST3R_FOCALS["wide"]),
        held=held,
        ref_held=REF_F,
    )
    assert got is not None
    assert (got.focal, got.division, got.ref_focal) == pytest.approx((*held, REF_F), abs=1e-5)
    turn = Rotation.from_matrix(got.rotation @ CAM_TO_REF[:3, :3]).magnitude()
    assert math.degrees(turn) < 1.5, "the pose still comes out near the truth"


def test_a_camera_held_with_a_lens_isnt_the_reference_and_keeps_it_in_the_group():
    pts = room()
    pairs = {("cam0", "wide"): pair(pts), ("wide", "cam0"): pair(pts, flip=True)}
    held = {"wide": (CAM_F / 288, LAMBDA), "cam0": (REF_F / 512, 0.0)}
    got = R.register(["wide", "cam0"], pairs, SIZES, images(), MAST3R_POSES, MAST3R_FOCALS, held=held)
    assert got is not None and got.reference == "cam0"
    assert got.focals == pytest.approx({"cam0": REF_F, "wide": CAM_F}, abs=1e-4)
    assert got.divisions["wide"] == pytest.approx(LAMBDA, abs=1e-5)


def test_a_held_lens_a_free_fit_beats_clearly_is_named_in_the_log(caplog):
    pts = room()
    pairs = {("cam0", "wide"): pair(pts), ("wide", "cam0"): pair(pts, flip=True)}
    wrong = {"wide": (0.9, 0.0)}  # held as a pinhole of the wrong focal: a free fit finds the barrel
    with caplog.at_level("WARNING"):
        R.register(["cam0", "wide"], pairs, SIZES, images(), MAST3R_POSES, MAST3R_FOCALS, held=wrong)
    assert "wide: its held intrinsics" in caplog.text


@pytest.mark.parametrize(
    ("held", "free", "clear"),
    [(3.0, 1.0, True), (1.8, 1.0, False), (1.4, 0.2, True), (1.0, 0.5, False)],
    ids=["much-worse", "under-a-pixel-worse", "worse-by-ratio-and-pixel", "worse-by-ratio-only"],
)
def test_a_held_lens_is_called_into_question_only_when_clearly_worse(held, free, clear):
    assert R.misfit(held, free) is clear
