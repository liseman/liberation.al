"""Fisheye rectification (src/liberational/see/fisheye.py): the lens model is OpenCV's, a person's virtual
view resamples what that view would see, and joints map back to where the lens put them. With no fisheye here, the
frames are synthetic renders through a known lens."""

import cv2
import numpy as np
import pytest
import torch

from liberational import config as cfg
from liberational.see import fisheye, gpu_pose
from liberational.vision import pose_model

# About an IMX291 behind a 160 degree lens at 1080p: 80 degrees off axis lands near the frame's side edge.
LENS = fisheye.Lens(1920, 1080, (690.0, 690.0, 959.5, 539.5), (0.05, -0.01, 0.002, 0.0))
SIZE: pose_model.Size = (192, 256)


def cv_matrix(lens: fisheye.Lens) -> tuple[np.ndarray, np.ndarray]:
    fx, fy, cx, cy = lens.k
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64), np.array(lens.d, np.float64).reshape(4, 1)


def test_rays_project_as_opencvs_fisheye_model_does():
    rng = np.random.default_rng(0)
    theta, phi = rng.uniform(0, np.radians(78), 200), rng.uniform(0, 2 * np.pi, 200)
    rays = np.stack((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)), 1)
    K, D = cv_matrix(LENS)
    want, _ = cv2.fisheye.projectPoints(rays[:, None, :], np.zeros(3), np.zeros(3), K, D)
    got = fisheye.project(LENS, torch.from_numpy(rays)).numpy()
    np.testing.assert_allclose(got, want[:, 0], atol=1e-6)


def test_pixels_unproject_to_the_rays_opencv_undistorts_them_to():
    pixels = np.array([[959.5, 539.5], [100.0, 540.0], [1800.0, 200.0], [960.0, 1000.0]], np.float64)
    K, D = cv_matrix(LENS)
    normalized = cv2.fisheye.undistortPoints(pixels[:, None, :], K, D)[:, 0]  # x / z, y / z of the ray
    rays = np.concatenate((np.asarray(normalized, np.float64), np.ones((len(pixels), 1))), 1)
    want = rays / np.linalg.norm(rays, axis=1, keepdims=True)
    got = fisheye.unproject(LENS, torch.from_numpy(pixels)).numpy()
    np.testing.assert_allclose(got, want, atol=1e-6)
    np.testing.assert_allclose(fisheye.project(LENS, torch.from_numpy(got)).numpy(), pixels, atol=1e-4)


def texture(rays: np.ndarray) -> np.ndarray:
    """A smooth scene on the sphere of directions, BGR uint8, so resampling differs only by interpolation."""
    a = np.arctan2(rays[..., 0], rays[..., 2])
    b = np.arctan2(rays[..., 1], np.hypot(rays[..., 0], rays[..., 2]))
    bgr = np.stack((128 + 90 * np.sin(5 * a), 128 + 90 * np.cos(4 * b), 128 + 60 * np.sin(3 * a + 2 * b)), -1)
    return np.clip(bgr, 0, 255).astype(np.uint8)


def render_fisheye(lens: fisheye.Lens) -> np.ndarray:
    v, u = np.mgrid[0 : lens.height, 0 : lens.width].astype(np.float64)
    rays = fisheye.unproject(lens, torch.from_numpy(np.stack((u, v), -1))).numpy()
    return texture(rays)


def render_view(rot: np.ndarray, focal: float, size: pose_model.Size) -> np.ndarray:
    """What the virtual pinhole camera sees of the same scene, pixel by pixel."""
    iw, ih = size
    v, u = np.mgrid[0:ih, 0:iw].astype(np.float64)
    rays = np.stack(((u - iw / 2) / focal, (v - ih / 2) / focal, np.ones_like(u)), -1)
    return texture(rays @ rot)  # the view's axes (rows of rot) back to the lens's: r_lens = rot.T @ r_view


HALF = LENS.scaled(960, 540)  # renders at half size, to keep the tests quick


@pytest.mark.parametrize(
    "box",
    [
        pytest.param((430, 150, 530, 450), id="center"),
        pytest.param((40, 175, 150, 400), id="near-the-left-rim"),
        pytest.param((750, 25, 875, 250), id="up-to-the-right"),
    ],
)
def test_a_persons_view_is_what_a_pinhole_camera_aimed_at_them_would_see(box):
    frame = torch.from_numpy(render_fisheye(HALF))
    blob, (rot, focal) = fisheye.crops(frame, torch.tensor([box], dtype=torch.float32), SIZE, HALF)
    mean, std = gpu_pose._norm(torch.device("cpu"))
    got = (blob * std + mean).flip(1)[0].permute(1, 2, 0).numpy()  # back to BGR pixels
    r, f = rot[0].double().numpy(), float(focal[0])
    want = render_view(r, f, SIZE).astype(np.float64)
    # Only where the view's ray lands in the frame: past its edge the crop is padding, as a plain crop's is.
    iw, ih = SIZE
    v, u = np.mgrid[0:ih, 0:iw].astype(np.float64)
    rays = np.stack(((u - iw / 2) / f, (v - ih / 2) / f, np.ones_like(u)), -1) @ r
    px = fisheye.project(HALF, torch.from_numpy(rays)).numpy()
    inside = (px[..., 0] >= 0) & (px[..., 0] <= HALF.width - 1) & (px[..., 1] >= 0) & (px[..., 1] <= HALF.height - 1)
    assert inside.mean() > 0.9
    error = np.abs(got - want).mean(-1)[inside]
    assert error.mean() < 1.0 and np.percentile(error, 99) < 3.0  # bilinear resampling of a smooth scene


def test_the_view_covers_the_padded_box_at_the_poses_aspect():
    box = torch.tensor([[80.0, 350.0, 300.0, 800.0]])
    rot, focal = fisheye.views(LENS, box, SIZE)
    corners = torch.tensor([[[80.0, 350.0], [300.0, 350.0], [80.0, 800.0], [300.0, 800.0]]])
    inview = torch.einsum("nij,nkj->nki", rot, fisheye.unproject(LENS, corners))
    x, y = inview[..., 0] / inview[..., 2] * focal, inview[..., 1] / inview[..., 2] * focal
    iw, ih = SIZE
    assert (x.abs() <= iw / 2 / pose_model.PADDING + 1e-3).all() and (
        y.abs() <= ih / 2 / pose_model.PADDING + 1e-3
    ).all()
    assert max(x.abs().max().item() / (iw / 2), y.abs().max().item() / (ih / 2)) == pytest.approx(
        1 / pose_model.PADDING, rel=1e-3
    )


def test_a_joint_in_the_view_maps_back_to_where_the_lens_put_it():
    box = torch.tensor([[1500.0, 50.0, 1750.0, 500.0]])
    rot, focal = fisheye.views(LENS, box, SIZE)
    # A point in the scene, a little off the view's axis.
    point = (rot[0].T @ torch.tensor([0.1, -0.2, 1.0])).double().numpy() * 3.0
    K, D = cv_matrix(LENS)
    want, _ = cv2.fisheye.projectPoints(point[None, None, :], np.zeros(3), np.zeros(3), K, D)
    iw, ih = SIZE
    in_view = torch.tensor([[[0.1 * float(focal[0]) + iw / 2, -0.2 * float(focal[0]) + ih / 2]]])
    got = fisheye.to_frame(in_view, (rot, focal), SIZE, LENS, (1080, 1920)).numpy()
    np.testing.assert_allclose(got[0, 0], want[0, 0], atol=1e-3)


def test_a_lens_follows_its_frames_size_and_mounting():
    half = LENS.scaled(960, 540)
    assert half.k == pytest.approx((345.0, 345.0, 479.5, 269.5))
    flipped = LENS.upright(True)
    assert flipped.k[2:] == pytest.approx((1919 - 959.5, 1079 - 539.5)) and flipped.d == LENS.d


def test_a_manifest_entry_gives_its_lens_turned_upright_or_none():
    entry = {"name": "fish0", "fisheye": {"width": 1920, "height": 1080, "k": [690, 690, 900, 500], "d": [0, 0, 0, 0]}}
    assert fisheye.lens_of(entry) == fisheye.Lens(1920, 1080, (690.0, 690.0, 900.0, 500.0), (0.0, 0.0, 0.0, 0.0))
    flipped = fisheye.lens_of({**entry, "video": {"flip": True}})
    assert flipped is not None and flipped.k[2:] == (1019.0, 579.0)
    assert fisheye.lens_of({"name": "cam0"}) is None
    with pytest.raises(ValueError, match="four terms"):
        fisheye.lens_of({"name": "bad", "fisheye": {"width": 1, "height": 1, "k": [1, 1, 0], "d": [0, 0, 0, 0]}})


def test_lens_for_reads_the_camera_from_the_manifest(tmp_path):
    manifest = tmp_path / "devices.toml"
    manifest.write_text(
        '[[device]]\nname = "cam0"\nkind = "camera"\n\n'
        '[[device]]\nname = "fish0"\nkind = "camera"\n'
        "fisheye = { width = 1920, height = 1080, k = [690.0, 690.0, 959.5, 539.5], d = [0.05, -0.01, 0.002, 0.0] }\n"
    )
    assert fisheye.lens_for(manifest, "fish0") == LENS
    assert fisheye.lens_for(manifest, "cam0") is None
    assert fisheye.lens_for(tmp_path / "missing.toml", "fish0") is None


def test_pose_through_a_lens_gives_frame_joints_and_their_view_joints():
    detector, pose_path = cfg.model_path("person_detector"), cfg.model_path("pose_live")
    if not (detector.exists() and pose_path.exists()):
        pytest.skip("model weights are missing; run mise run models")
    pose = gpu_pose.GpuPose(detector, pose_path, torch.device("cpu"), lens=LENS)
    frame = torch.from_numpy(render_fisheye(HALF))  # a half-size frame through a full-size calibration: it scales
    boxes = np.array([[40, 175, 150, 400], [430, 150, 530, 450]], np.float32)
    joints, view_joints = pose.skeletons(frame, boxes, views=True)
    assert joints.shape == view_joints.shape == (2, 26, 3)
    placed = view_joints[..., 2] > 0
    np.testing.assert_array_equal(placed, joints[..., 2] > 0)
    # Each placed joint in frame pixels is its view joint through the view and the lens.
    rot, focal = fisheye.views(HALF, torch.from_numpy(boxes), SIZE)
    back = fisheye.to_frame(torch.from_numpy(view_joints[..., :2]), (rot, focal), SIZE, LENS, (540, 960)).numpy()
    np.testing.assert_allclose(back[placed], joints[..., :2][placed], atol=1e-2)
