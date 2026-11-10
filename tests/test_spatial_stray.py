"""Reading a Stray Scanner export (#2931): ARKit's poses as OpenCV camera to world, the quarter turn that stands a
portrait recording upright, and the LiDAR depth as confident points."""

import cv2
import numpy as np
import pytest

from liberational.spatial import stray
from stray_fake import CLOCKWISE, RGB, K, distance_to_room, upright_pose, write


def walk(turn=None):
    poses = [upright_pose(1.0 + 0.1 * k, heading=0.05 * k) for k in range(4)]
    if turn is not None:
        poses = [stray.turned(p, turn.T) for p in poses]  # the sensor camera, rolled from the upright one
    return poses


@pytest.mark.parametrize("columns", [True, False], ids=["per-frame-intrinsics", "camera-matrix-only"])
def test_odometry_gives_each_frames_pose_and_intrinsics(tmp_path, columns):
    poses = walk()
    rows = stray.read_odometry(write(tmp_path / "x", poses, intrinsics_columns=columns))
    assert sorted(rows) == list(range(15))  # 5 frames for each of the 3 steps
    assert rows[5].c2w == pytest.approx(poses[1], abs=1e-6)
    assert rows[5].t == pytest.approx(5 / 30)
    assert rows[5].k == pytest.approx(K)


def test_a_landscape_recording_needs_no_turn():
    turn, code = stray.upright(np.stack(walk()))
    assert turn == pytest.approx(np.eye(3)) and code is None


def test_a_portrait_recording_is_turned_so_its_cameras_stand_upright():
    sensor = np.stack(walk(CLOCKWISE))
    turn, code = stray.upright(sensor)
    assert turn == pytest.approx(CLOCKWISE) and code == 0  # cv2.ROTATE_90_CLOCKWISE
    for pose in sensor:
        assert stray.turned(pose, turn)[:3, :3] @ [0, -1, 0] == pytest.approx([0, 1, 0], abs=1e-9)


@pytest.mark.parametrize("turn", [None, CLOCKWISE], ids=["landscape", "portrait"])
def test_depth_becomes_points_on_the_rooms_surfaces_without_the_unsure_ones(tmp_path, turn):
    folder = write(tmp_path / "x", walk(turn))
    rows = stray.read_odometry(folder)
    pts, colors = stray.points(folder, 7, rows[7].k, RGB, None)
    assert len(pts) == 60 * 48, "the 4-pixel low-confidence strip is left out"
    world = pts @ rows[7].c2w[:3, :3].T + rows[7].c2w[:3, 3]
    assert distance_to_room(world).max() < 0.002  # millimetre depth
    assert colors.shape == (len(pts), 3)


def test_a_frame_without_depth_has_no_points(tmp_path):
    folder = write(tmp_path / "x", walk())
    pts, colors = stray.points(folder, 999, K, RGB, None)
    assert len(pts) == 0 and len(colors) == 0


@pytest.mark.parametrize("which", range(4), ids=["landscape", "clockwise", "upside-down", "counterclockwise"])
def test_the_turned_frame_shows_up_as_up(which):
    cv2 = pytest.importorskip("cv2")
    matrix = np.array(stray.TURNS[which][0])
    sensor = stray.turned(upright_pose(2.0), matrix.T)
    _, code = stray.upright(np.stack([sensor]))
    image = np.zeros((RGB[1], RGB[0]), np.uint8)
    for value, height in ((255, 1.8), (128, 1.2)):  # 0.3 m above the camera's eye and below it, 1 m ahead
        cam = (np.array([2.0, height, 2.5]) - sensor[:3, 3]) @ sensor[:3, :3]
        u, v = K[0, 0] * cam[0] / cam[2] + K[0, 2], K[1, 1] * cam[1] / cam[2] + K[1, 2]
        image[round(v), round(u)] = value
    shown = image if code is None else cv2.rotate(image, code)
    above, below = np.argwhere(shown == 255)[0], np.argwhere(shown == 128)[0]
    assert above[0] < below[0], "the higher point is nearer the top of the turned frame"
    assert abs(above[1] - below[1]) <= 1, "and straight above the lower one"


@pytest.mark.parametrize("which", range(4), ids=["landscape", "clockwise", "upside-down", "counterclockwise"])
def test_a_saved_frames_intrinsics_put_a_point_where_the_turned_image_shows_it(which):
    # an off-centre principal point and unequal focals, so a swapped axis or a wrong edge shows
    k = np.array([[240.0, 0.0, 150.0], [0.0, 260.0, 110.0], [0.0, 0.0, 1.0]])
    w, h = 320, 240
    matrix, code = stray.TURNS[which]
    turn = np.array(matrix)
    got, size = stray.saved_intrinsics(k, (w, h), 1024, code, turn)
    # where the sensor's pixel (x, y) lands in the image cv2.rotate turns: an index image tells
    index = np.arange(w * h, dtype=np.int32).reshape(h, w)
    turned = index if code is None else cv2.rotate(index, code)
    assert size == (turned.shape[1], turned.shape[0])
    for p in ([0.3, -0.2, 2.0], [-0.5, 0.4, 3.0], [0.0, 0.0, 1.0]):
        sensor = k @ (turn @ np.array(p))
        x, y = round(sensor[0] / sensor[2]), round(sensor[1] / sensor[2])
        ty, tx = np.argwhere(turned == y * w + x)[0]
        mine = got @ np.array(p)
        assert (round(mine[0] / mine[2]), round(mine[1] / mine[2])) == (tx, ty)


def test_a_frame_scaled_down_scales_its_intrinsics():
    k = np.array([[1400.0, 0.0, 960.0], [0.0, 1400.0, 720.0], [0.0, 0.0, 1.0]])
    got, size = stray.saved_intrinsics(k, (1920, 1440), 1024, None, np.eye(3))
    assert size == (1024, 768)
    s = 1024 / 1920
    assert got == pytest.approx(np.array([[1400 * s, 0, 960 * s], [0, 1400 * s, 720 * s], [0, 0, 1]]))
