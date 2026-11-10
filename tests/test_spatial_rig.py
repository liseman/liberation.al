import math

import pytest

from liberational.memory import places
from liberational.spatial import rig

# A place whose map has the floor at z = 0 with z up, and a plan drawn in the map's own x and y.
REV = places.Revision(
    "rev1",
    "place1",
    0.0,
    "test",
    (),
    floor=(0.0, 0.0, 1.0, 0.0),
    plan_transform=(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
)


def rig_facing_x(x: float, y: float, z: float) -> tuple[float, ...]:
    """The rig standing at (x, y, z) in the map, facing the map's +x, upright: its right is the map's -y, its down
    the map's -z, and its forward the map's +x."""
    return (0.0, 0.0, 1.0, x, -1.0, 0.0, 0.0, y, 0.0, -1.0, 0.0, z, 0.0, 0.0, 0.0, 1.0)


@pytest.mark.parametrize(
    ("rig_at", "expected"),
    [
        # The camera's lens is 0.2 m above the rig's origin, so 1.7 m up; it looks to the rig's right (the map's -y)
        # and 45 degrees down, so its centre pixel meets the floor 1.7 m to the rig's right.
        pytest.param((2.0, 3.0, 1.5), (2.0, 1.3), id="rig-at-2-3"),
        pytest.param((3.0, 3.0, 1.5), (3.0, 1.3), id="rig-walked-a-metre-forward"),
    ],
)
def test_a_rig_cameras_pixel_lands_on_the_plan_from_the_rigs_pose(mounts, rig_at, expected):
    mounts(
        {"pack-right": {"rig": "pack", "at": [0.0, -0.2, 0.0], "turn": [90.0, -45.0, 0.0], "hfov_deg": 90.0}}, ["pack"]
    )
    mount = rig.load().get("pack-right")

    hit = rig.to_plan(REV, mount, rig_facing_x(*rig_at), 640, 480, 0.5, 0.5)

    assert hit == pytest.approx(expected, abs=1e-3)


def test_a_rig_cameras_focal_comes_from_its_field_of_view():
    mount = rig.Mount("pack-front", rig="pack", hfov_deg=90.0)
    pose = rig.camera_pose(mount, rig_facing_x(0.0, 0.0, 1.5), 640, 480)
    assert pose.focal == pytest.approx(320.0)  # half the width over tan(45 degrees)
    # With no offset and no turn, the camera has the rig's own pose
    assert pose.pose == pytest.approx(rig_facing_x(0.0, 0.0, 1.5))


def test_a_fixed_camera_has_no_pose_from_a_rig():
    with pytest.raises(ValueError, match="fixed"):
        rig.camera_pose(rig.Mount("hall", height_m=2.0), rig_facing_x(0.0, 0.0, 1.5), 640, 480)


@pytest.mark.parametrize(
    ("turn", "column", "expected"),
    [
        pytest.param((90.0, 0.0, 0.0), 2, (1.0, 0.0, 0.0), id="yaw-right-looks-along-x"),
        pytest.param((0.0, 90.0, 0.0), 2, (0.0, -1.0, 0.0), id="pitch-up-looks-up"),
        pytest.param((0.0, 0.0, 90.0), 0, (0.0, 1.0, 0.0), id="roll-clockwise-turns-right-down"),
        pytest.param((90.0, -90.0, 0.0), 2, (0.0, 1.0, 0.0), id="pitch-after-yaw-still-looks-down"),
    ],
)
def test_a_turn_points_the_camera_the_way_it_says(turn, column, expected):
    r = rig.turn_matrix(*turn)
    assert [r[i][column] for i in range(3)] == pytest.approx(expected, abs=1e-12)


def test_mounts_split_fixed_cameras_from_rig_cameras():
    parsed = rig.parse(
        {
            "rig": [{"name": "pack"}],
            "device": [
                {"name": "hall", "kind": "rtsp", "mount": {"height_m": 2.1}},
                {"name": "patio", "kind": "camera", "mount": {"outdoor": True}},
                {"name": "desk", "kind": "camera"},
                {"name": "pack-front", "kind": "camera", "mount": {"rig": "pack", "hfov_deg": 100}},
                {"name": "wave3", "kind": "microphone"},
            ],
        }
    )
    assert parsed.heights() == {"hall": 2.1}
    assert parsed.outdoor() == {"patio"}
    assert parsed.on_rig() == ["pack-front"] == parsed.on_rig("pack")
    assert parsed.get("desk") == rig.Mount("desk")
    assert parsed.get("not-listed").fixed
    assert "wave3" not in parsed.cameras
    front = parsed.get("pack-front")
    assert (front.at, front.turn, front.hfov_deg) == ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 100.0)


def test_a_missing_manifest_has_no_mounts(tmp_path):
    assert rig.load(tmp_path / "devices.toml") == rig.Mounts()


@pytest.mark.parametrize(
    ("mount", "message"),
    [
        pytest.param({"rig": "cart", "hfov_deg": 90}, "no \\[\\[rig\\]\\]", id="undeclared-rig"),
        pytest.param({"rig": "pack", "hfov_deg": 90, "height_m": 1.2}, "for a fixed camera", id="height-on-a-rig"),
        pytest.param({"rig": "pack", "hfov_deg": 90, "outdoor": True}, "for a fixed camera", id="outdoor-on-a-rig"),
        pytest.param({"rig": "pack"}, "needs hfov_deg", id="rig-camera-without-fov"),
        pytest.param({"rig": "pack", "hfov_deg": 180}, "between 0 and 180", id="fov-not-a-pinholes"),
        pytest.param({"rig": "pack", "hfov_deg": 90, "at": [0, 0]}, "three numbers", id="short-offset"),
        pytest.param({"rig": "pack", "hfov_deg": 90, "turn": [0, math.nan, 0]}, "finite", id="nan-turn"),
        pytest.param({"at": [0, 0, 0]}, "for a camera on a rig", id="offset-on-a-fixed-camera"),
        pytest.param({"height_m": 0}, "more than 0", id="height-at-the-floor"),
        pytest.param({"height_m": True}, "is a number", id="height-not-a-number"),
        pytest.param({"outdoor": "yes"}, "true or false", id="outdoor-not-a-bool"),
        pytest.param({"room": "kitchen"}, "mount has no room", id="unknown-key"),
    ],
)
def test_a_mount_that_makes_no_sense_names_its_camera(mount, message):
    manifest = {"rig": [{"name": "pack"}], "device": [{"name": "cam0", "kind": "camera", "mount": mount}]}
    with pytest.raises(ValueError, match="cam0") as raised:
        rig.parse(manifest)
    assert raised.match(message)


def test_a_rig_named_twice_is_refused():
    with pytest.raises(ValueError, match="named twice"):
        rig.parse({"rig": [{"name": "pack"}, {"name": "pack"}]})
