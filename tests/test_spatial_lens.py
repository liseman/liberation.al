"""Wide-lens fixed cameras: the division model undone in frames and in pixels alike, the overlays painted over
and left out, and a registered pose's foot points undistorted before they're cast onto the floor."""

import json
import textwrap

import cv2
import numpy as np
import pytest

from liberational.memory import places
from liberational.spatial import footprints, lens
from liberational.spatial.mast3r_runner import overlay

W, H = 896, 512


def read(path, flags=cv2.IMREAD_COLOR):
    image = cv2.imread(str(path), flags)
    assert image is not None, path
    return image


BARREL = lens.Lens(W, H, (W / 2, W / 2, (W - 1) / 2, (H - 1) / 2), -0.31)  # the Reolinks' joint fit
OVERLAYS = [(0.25, 0.0, 0.74, 0.09), (0.84, 0.92, 1.0, 1.0)]


@pytest.mark.parametrize("division", [-0.31, -0.1, 0.05])
def test_the_undistorted_frame_and_undistorted_pixels_agree(division):
    # Each pixel of the undistorted frame is filled from a pixel of the frame (`undistort_maps`); that frame pixel,
    # undistorted as a point, must land back on it, or foot points and MASt3R's frame would disagree.
    lz = lens.Lens(W, H, BARREL.k, division)
    mx, my = lens.undistort_maps(lz, W, H)
    rows, cols = np.mgrid[20:H:60, 20:W:60]
    src_u, src_v = (mx[rows, cols] + 0.5) / W, (my[rows, cols] + 0.5) / H
    inside = (src_u > 0) & (src_u < 1) & (src_v > 0) & (src_v < 1)
    u, v = lens.undistort_points(lz, src_u[inside], src_v[inside])
    np.testing.assert_allclose(u * W - 0.5, cols[inside], atol=1e-3)
    np.testing.assert_allclose(v * H - 0.5, rows[inside], atol=1e-3)


def border(width, height):
    """Every pixel centre on a frame's four edges, as fractions."""
    u = np.r_[np.arange(width), np.arange(width), np.zeros(height), np.full(height, width - 1)] + 0.5
    v = np.r_[np.zeros(width), np.full(width, height - 1), np.arange(height), np.arange(height)] + 0.5
    return u / width, v / height


PORTRAIT = lens.Lens(288, 512, (144.0, 144.0, 143.5, 255.5), -0.1)  # a living-room-shaped frame, on its side


@pytest.mark.parametrize("found", [BARREL, PORTRAIT], ids=["landscape", "portrait"])
def test_the_undistorted_frame_keeps_the_whole_frame_on_both_axes(found):
    # At the old scale a landscape frame lost its corners (the kitchen's floor strip) and a portrait one most of
    # its top and bottom, its long axis. Every pixel on the frame's edges must land in the undistorted frame...
    u, v = lens.undistort_points(found, *border(found.width, found.height))
    assert u.min() >= -1e-9 and u.max() <= 1 + 1e-9 and v.min() >= -1e-9 and v.max() <= 1 + 1e-9
    # ...as large as that allows: some of the frame reaches the undistorted frame's edge.
    assert min(u.min(), v.min(), 1 - u.max(), 1 - v.max()) == pytest.approx(0, abs=0.01)
    centre = lens.undistort_points(found, np.array([0.5]), np.array([0.5]))
    assert (centre[0][0], centre[1][0]) == pytest.approx((0.5, 0.5), abs=1e-3)


def test_a_floor_strip_along_the_frames_bottom_survives_undistortion():
    # The kitchen camera sits 0.91 m up and sees the floor only along the bottom of its frame, corner to corner:
    # every pixel of the bottom tenth must land in the undistorted frame, on a pixel that isn't left out.
    rows, cols = np.mgrid[int(H * 0.9) : H, 0:W]
    u, v = lens.undistort_points(BARREL, (cols + 0.5) / W, (rows + 0.5) / H)
    assert u.min() >= 0 and u.max() <= 1 and v.min() >= 0 and v.max() <= 1
    _, covered = lens.prepare(np.full((H, W, 3), 90, np.uint8), BARREL, [])
    x, y = np.clip((u * W).astype(int), 0, W - 1), np.clip((v * H).astype(int), 0, H - 1)
    assert covered[y, x].mean() < 0.02  # all but the outline's one-pixel rim


def test_a_calibration_that_undoes_its_corners_past_the_models_pole_is_refused():
    # Living-room's plumb-line fit on the Linux machine: division -0.43 at a focal of 360 on a 720 x 1280 frame put its
    # top and bottom past 1 + lambda r^2 = 0, where the model has no undistorted point.
    entry = {"name": "living-room", "lens": {"width": 720, "height": 1280, "k": [360.0, 360.0, 359.5, 639.5]}}
    with pytest.raises(ValueError, match="living-room: lens division -0.43"):
        lens.lens_of({**entry, "lens": {**entry["lens"], "division": -0.43}})
    assert lens.lens_of({**entry, "lens": {**entry["lens"], "division": -0.1}}) is not None


def test_no_distortion_is_the_identity():
    flat = lens.Lens(W, H, BARREL.k, 0.0)
    u, v = lens.undistort_points(flat, np.array([0.1, 0.9]), np.array([0.05, 0.7]))
    np.testing.assert_allclose(np.c_[u, v], [[0.1, 0.05], [0.9, 0.7]], atol=1e-12)


def test_a_straight_edge_bent_by_the_barrel_comes_out_straight():
    # A straight vertical line near the frame's side, drawn through the lens (the undistorted frame's maps run the
    # other way, so distort by inverting them numerically: find each row's frame pixel that undistorts onto x = 820).
    rows = np.arange(30, H - 30, 10)
    xs = []
    for r in rows:
        cand = np.linspace(0, W - 1, 4000)
        u, _ = lens.undistort_points(BARREL, (cand + 0.5) / W, np.full_like(cand, (r + 0.5) / H))
        xs.append(cand[np.argmin(np.abs(u * W - 0.5 - 820))])
    xs = np.array(xs)
    assert np.ptp(xs) > 10  # seen through the lens, the line bows by over 10 px
    u, _ = lens.undistort_points(BARREL, (xs + 0.5) / W, (rows + 0.5) / H)
    assert np.ptp(u * W) < 0.5  # undone, it's straight again (x within half a pixel)


@pytest.mark.parametrize(
    "entry",
    [
        {"name": "a", "lens": {"width": 896, "height": 512, "k": [448.0, 448.0, 447.5], "division": -0.3}},
        {"name": "a", "lens": {"width": 896, "height": 512, "k": [0.0, 448.0, 447.5, 255.5], "division": -0.3}},
        {"name": "a", "lens": {"width": 0, "height": 512, "k": [448.0, 448.0, 447.5, 255.5], "division": -0.3}},
        {"name": "a", "lens": {"width": 896, "height": 512, "k": [448.0, 448.0, 447.5, 255.5], "division": "nan"}},
    ],
)
def test_a_malformed_lens_is_refused(entry):
    with pytest.raises(ValueError, match="a: lens"):
        lens.lens_of(entry)


@pytest.mark.parametrize("bad", [[0.5, 0.0, 0.4, 0.1], [0.0, 0.0, 1.2, 0.1], [0.0, 0.0, 0.1]])
def test_a_malformed_mask_is_refused(bad):
    with pytest.raises(ValueError):
        lens.masks_of({"name": "a", "masks": [bad]})


def test_a_cameras_lens_and_masks_are_read_from_the_manifest(tmp_path):
    manifest = tmp_path / "devices.toml"
    manifest.write_text(
        textwrap.dedent(
            """
            [[device]]
            name = "bedroom"
            kind = "rtsp"
            lens = { width = 896, height = 512, k = [448.0, 448.0, 447.5, 255.5], division = -0.31 }
            masks = [[0.25, 0.0, 0.74, 0.09]]

            [[device]]
            name = "cam0"
            kind = "camera"
            """
        )
    )
    found, masks = lens.for_camera(manifest, "bedroom")
    assert found == BARREL and masks == [(0.25, 0.0, 0.74, 0.09)]
    assert lens.for_camera(manifest, "cam0") == (None, [])
    assert lens.for_camera(manifest, "porch") == (None, [])
    assert lens.for_camera(tmp_path / "missing.toml", "bedroom") == (None, [])


def test_an_overlay_is_painted_over_and_left_out_with_what_the_frame_doesnt_reach():
    frame = np.full((H, W, 3), 120, np.uint8)
    frame[10:40, 300:600] = 255  # a burned-in clock, white on the grey wall
    out, covered = lens.prepare(frame, None, [(300 / W, 10 / H, 600 / W, 40 / H)])
    assert out[25, 450].tolist() == pytest.approx([120, 120, 120], abs=3)  # painted from its surroundings
    assert covered[25, 450] and not covered[200, 450]
    assert covered.sum() == 300 * 30  # without a lens, exactly the overlay
    out, covered = lens.prepare(frame, BARREL, [])
    # Undone, the barrel's frame doesn't reach the top and bottom middles of the undistorted one: left out.
    assert covered[0, W // 2] and covered[H - 1, W // 2] and not covered[H // 2, W // 2]
    assert out.shape == frame.shape


# A camera 2 m above the floor (z up), looking straight down, 100 x 100 px at focal 100.
DOWN = places.CameraPose((1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 2.0, 0, 0, 0, 1), 100.0, 100, 100, 0.0, BARREL.as_tuple())
FLOOR = (0.0, 0.0, 1.0, 0.0)


def test_a_registered_pose_keeps_its_lens():
    assert places.CameraPose.from_dict(DOWN.as_dict()) == DOWN
    plain = places.CameraPose(DOWN.pose, 100.0, 100, 100)
    assert "lens" not in plain.as_dict() and places.CameraPose.from_dict(plain.as_dict()).lens is None


def test_a_foot_point_is_undistorted_before_its_ray_is_cast():
    pinhole = places.CameraPose(DOWN.pose, DOWN.focal, DOWN.width, DOWN.height)
    u, v = 0.9, 0.8  # off towards a corner, where the barrel moved it most
    uu, vv = lens.undistort_points(BARREL, u, v)
    expected = places.floor_point(pinhole, FLOOR, float(uu), float(vv))
    assert places.floor_point(DOWN, FLOOR, u, v) == pytest.approx(expected)
    assert places.floor_point(DOWN, FLOOR, u, v) != pytest.approx(places.floor_point(pinhole, FLOOR, u, v))


def test_many_foot_points_of_one_camera_work_out_its_lens_frame_once(monkeypatch):
    # A positions read undistorts every foot point of a day; the undistorted frame's scale is the lens's own,
    # so it's worked out once, not per point. A lens no other test uses, so nothing has worked it out yet.
    cam = places.CameraPose(DOWN.pose, DOWN.focal, DOWN.width, DOWN.height, 0.0, (*BARREL.as_tuple()[:4], -0.27))
    boundary = lens._boundary
    calls = []
    monkeypatch.setattr(lens, "_boundary", lambda found: calls.append(found) or boundary(found))
    points = [places.floor_point(cam, FLOOR, u, 0.8) for u in (0.1, 0.5, 0.9)]
    assert len(calls) == 1
    assert len({p[0] for p in points if p is not None}) == 3


def test_the_walkable_floor_undistorts_its_footprints_as_a_single_foot_point_does():
    uv = np.array([[0.9, 0.8], [0.5, 0.5], [0.2, 0.7]])
    points, _ = footprints.project([footprints.Cells("bedroom", uv, np.ones(3))], {"bedroom": DOWN}, FLOOR)
    expected = [places.floor_point(DOWN, FLOOR, u, v, footprints.FOOT_M) for u, v in uv]  # people's heels
    assert all(e is not None for e in expected)
    np.testing.assert_allclose(points, np.array(expected, dtype=float), atol=1e-9)


def test_the_runners_overlay_mask_lands_where_dust3r_puts_the_frame(tmp_path):
    # dust3r's loader takes 896 x 512 to 512 x 293 and centre-crops it to 512 x 288 (whole 16 px patches).
    image = tmp_path / "bedroom.png"
    cv2.imwrite(str(image), np.zeros((H, W, 3), np.uint8))
    assert overlay(str(image), 288, 512, 512) is None  # no mask beside it
    mask = np.zeros((H, W), np.uint8)
    mask[0:46, 224:672] = 255  # a band across the top, a quarter in from each side
    cv2.imwrite(str(tmp_path / "bedroom.mask.png"), mask)
    got = overlay(str(image), 288, 512, 512)
    assert got.shape == (288, 512)
    rows, cols = np.nonzero(got)
    assert (cols.min(), cols.max()) == (128, 383)  # 224 * 512 / 896 and 672 * 512 / 896, less one
    assert rows.min() == 0 and rows.max() == pytest.approx(46 * 293 / 512 - 2 - 1, abs=1)


def test_a_frame_from_a_wide_camera_goes_to_mast3r_undistorted_with_its_mask(tmp_path, vision, monkeypatch):
    import datetime as dt
    from zoneinfo import ZoneInfo

    from liberational.spatial import cameras

    manifest = tmp_path / "devices.toml"
    lens_line = "lens = { width = 896, height = 512, k = [448.0, 448.0, 447.5, 255.5], division = -0.31 }"
    manifest.write_text(
        f'[[device]]\nname = "bedroom"\nkind = "rtsp"\n{lens_line}\nmasks = [[0.25, 0.0, 0.74, 0.09]]\n'
        '[[device]]\nname = "cam0"\nkind = "camera"\n'
    )
    monkeypatch.setattr(lens, "manifest", lambda: manifest)
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (H, W, 3), np.uint8)
    frame[..., 2] = 220  # lit colour, so it's taken
    utc = ZoneInfo("UTC")
    now = dt.datetime(2030, 1, 2, 12, 0, tzinfo=utc).timestamp()
    got = cameras.frames(tmp_path, ["bedroom", "cam0"], tmp_path, now, lambda cam, t: frame, utc)
    written = read(got["bedroom"])
    assert not np.array_equal(written, frame)  # undistorted
    assert read(tmp_path / "bedroom.mask.png", cv2.IMREAD_GRAYSCALE)[20, 448] == 255  # the clock's band
    assert np.array_equal(read(got["cam0"]), frame)  # a pinhole camera's frame goes as it is
    assert not (tmp_path / "cam0.mask.png").exists()


def test_a_splat_trains_on_the_frame_mast3r_saw_with_its_overlays_left_out(tmp_path, monkeypatch):
    from liberational.spatial import splats

    manifest = tmp_path / "devices.toml"
    manifest.write_text('[[device]]\nname = "bedroom"\nkind = "rtsp"\nmasks = [[0.84, 0.92, 1.0, 1.0]]\n')
    monkeypatch.setattr(lens, "manifest", lambda: manifest)
    pose = places.CameraPose(tuple(float(x) for x in np.eye(4).ravel()), 300.0, 512, 288, 0.0, BARREL.as_tuple())
    rev = places.Revision("0" * 8 + "-0000-0000-0000-" + "0" * 12, "1" * 8 + "-0000-0000-0000-" + "1" * 12, 0.0, "t", (),
                          cameras={"bedroom": pose})  # fmt: skip
    frame = np.full((H, W, 3), 128, np.uint8)
    job = json.loads(
        splats.write_job(tmp_path, rev, {"bedroom": [frame]}, (np.zeros((1, 3)), np.zeros((1, 3)))).read_text()
    )
    (view,) = job["views"]
    mask = read(view["mask"], cv2.IMREAD_GRAYSCALE)
    assert mask.shape == (288, 512) and read(view["image"]).shape == (288, 512, 3)
    assert mask[-1, -1] == 255 and mask[144, 256] == 0  # the name's corner (and the barrel's) out, the middle in


def test_a_persons_height_undoes_the_lens_on_their_head_as_on_their_feet():
    from liberational.spatial import fuse

    # A level camera 2 m up looking along +y (z up), 100 x 100 px at focal 100, through the Reolinks' barrel.
    level = (1.0, 0, 0, 0, 0, 0, 1.0, 0, 0, -1.0, 0, 2.0, 0, 0, 0, 1.0)
    wide = places.CameraPose(level, 100.0, 100, 100, 0.0, BARREL.as_tuple())
    pinhole = places.CameraPose(level, 100.0, 100, 100)
    # Someone 1.7 m tall, 6 m ahead and 1.5 m to the side: where a pinhole sees their head (0.3 m under the camera)
    # and their heels (FOOT_M = 0.07 m up, 1.93 m under it)...
    head, feet = (0.5 + 1.5 / 6, 0.5 + 0.3 / 6), (0.5 + 1.5 / 6, 0.5 + 1.93 / 6)
    assert fuse.stature(pinhole, footprints.Sighting("c", 0.0, 1, *feet, head)) == pytest.approx(1.7)

    # ...and where the barrel moved those pixels to: the wide camera, undoing it on both, measures the same.
    def seen(p):
        cand = np.linspace(0, 1, 20001)
        best = None
        for v in np.linspace(max(0, p[1] - 0.2), min(1, p[1] + 0.2), 401):
            u, vv = lens.undistort_points(BARREL, cand, np.full_like(cand, v))
            i = int(np.argmin((u - p[0]) ** 2 + (vv - p[1]) ** 2))
            err = (u[i] - p[0]) ** 2 + (vv[i] - p[1]) ** 2
            if best is None or err < best[0]:
                best = (err, cand[i], v)
        assert best is not None
        return best[1], best[2]

    s = footprints.Sighting("c", 0.0, 1, *seen(feet), seen(head))
    assert fuse.stature(wide, s) == pytest.approx(1.7, abs=0.02)
