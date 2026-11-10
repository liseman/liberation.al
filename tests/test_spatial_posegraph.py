"""The walkthrough's pose graph (#2931): drift closed by a loop, gravity, the gauge, robustness to a false loop, and
the acceptance checks a MASt3R pair passes to become a loop edge."""

import itertools
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.spatial.transform import Rotation

from liberational.spatial import posegraph as pg

UP = np.array([0.0, 0.0, 1.0])  # the tests' world is z-up


def upright(heading, position):
    """A camera (OpenCV axes: x right, y down, z forward) held level, looking along `heading` (radians from +x)."""
    forward = np.array([math.cos(heading), math.sin(heading), 0.0])
    down = -UP
    pose = np.eye(4)
    pose[:3, :3] = np.stack([np.cross(down, forward), down, forward], axis=1)
    pose[:3, 3] = position
    return pose


def rectangle(width, depth, step=0.25):
    """A walk once round a width x depth rectangle at 1.5 m, facing the way it goes, ending a step short of the start."""
    corners = [(0.0, 0.0), (width, 0.0), (width, depth), (0.0, depth), (0.0, 0.0)]
    poses = []
    for (ax, ay), (bx, by) in itertools.pairwise(corners):
        length = math.hypot(bx - ax, by - ay)
        heading = math.atan2(by - ay, bx - ax)
        for k in range(round(length / step)):
            u = k * step / length
            poses.append(upright(heading, (ax + u * (bx - ax), ay + u * (by - ay), 1.5)))
    return np.array(poses)


def about(axis, radians):
    """A 4x4 turn about a camera axis (0 x, 1 y, 2 z)."""
    out = np.eye(4)
    out[:3, :3] = Rotation.from_rotvec(np.eye(3)[axis] * radians).as_matrix()
    return out


def chain(start, steps):
    poses = [start]
    for step in steps:
        poses.append(poses[-1] @ step)
    return np.array(poses)


def drifted(true, axis, total_deg):
    """The walk as an odometry that turns an extra total_deg about a camera axis, spread evenly over its steps."""
    bias = about(axis, math.radians(total_deg) / (len(true) - 1))
    return chain(true[0], [pg.relative(true[k], true[k + 1]) @ bias for k in range(len(true) - 1)])


def position_error(poses, true):
    return float(np.linalg.norm(poses[:, :3, 3] - true[:, :3, 3], axis=1).max())


def tilt_deg(poses):
    ups = np.einsum("nij,j->ni", poses[:, :3, :3], pg.CAMERA_UP)
    return np.degrees(np.arccos(np.clip(ups @ UP, -1.0, 1.0)))


def closing_loop(true):
    return pg.Edge(len(true) - 1, 0, pg.relative(true[-1], true[0]), pg.LOOP_DEG, pg.LOOP_M, loop=True)


@settings(max_examples=25, deadline=None)
@given(st.floats(2.0, 6.0), st.floats(2.0, 6.0), st.floats(5.0, 40.0), st.sampled_from([-1, 1]))
def test_a_loop_edge_closes_heading_drift_and_the_first_pose_stays(width, depth, drift_deg, sign):
    true = rectangle(width, depth)
    start = drifted(true, 1, sign * drift_deg)  # about camera y: heading, the drift CUT3R has
    solved = pg.solve(start, [*pg.odometry(start), closing_loop(true)])
    assert np.array_equal(solved.poses[0], start[0]), "the gauge"
    # The loop doesn't close all of the drift, and shouldn't: the chain of odometry edges is as stiff as their noise
    # times the square root of their number (2 degrees x sqrt(31), about 11 degrees, round a 2 x 2 m rectangle)
    # against the loop's 5, so least squares leaves the loop about a sixth of the drift, and Huber, softening a loop
    # more than 1 sigma out, about 30% on the smallest rectangle with the most drift (#2992). Without the loop
    # nothing improves (a ratio of 1).
    assert position_error(solved.poses, true) < 0.5 * position_error(start, true)
    assert solved.edge_deg[-1] < 0.35 * drift_deg
    assert solved.dropped == ()


@settings(max_examples=25, deadline=None)
@given(
    st.lists(st.tuples(st.floats(-0.5, 0.5), st.floats(0.05, 0.5)), min_size=1, max_size=40),
    st.floats(-math.pi, math.pi),
)
def test_a_graph_its_poses_already_fit_is_left_as_it_is(turns_and_steps, heading):
    """Exact odometry, level cameras at one height: nothing to correct, so nothing moves."""
    steps = [
        about(1, turn) @ np.r_[np.c_[np.eye(3), [0.0, 0.0, step]], [[0, 0, 0, 1]]] for turn, step in turns_and_steps
    ]
    poses = chain(upright(heading, (1.0, -2.0, 1.5)), steps)
    solved = pg.solve(poses, pg.odometry(poses))
    np.testing.assert_allclose(solved.poses, poses, atol=1e-9)
    assert solved.cost == pytest.approx(0.0, abs=1e-12)


def test_a_loop_twice_the_true_length_doesnt_shrink_or_stretch_the_walk():
    """No scale in the state: a loop translation off by 2x can only be fit by bending, which the odometry refuses,
    so the loop is dropped and the walk keeps its size (what a stitch of free-scale segments got wrong)."""
    true = rectangle(5.0, 3.0)
    long = pg.relative(true[0], true[20])
    long[:3, 3] *= 2.0
    solved = pg.solve(true, [*pg.odometry(true), pg.Edge(0, 20, long, pg.LOOP_DEG, pg.LOOP_M, loop=True)])
    assert solved.dropped == (len(true) - 1,)
    assert position_error(solved.poses, true) < 1e-6


def test_gravity_pulls_a_slow_pitch_ramp_back_toward_up():
    true = rectangle(5.0, 3.0)
    start = drifted(true, 0, 40.0)  # about camera x: the nose dips a little more every step
    solved = pg.solve(start, pg.odometry(start), up=UP)
    assert tilt_deg(start).max() > 10.0
    assert tilt_deg(solved.poses).max() < 0.2 * tilt_deg(start).max()
    assert np.array_equal(solved.poses[0], start[0])


def test_without_the_priors_the_same_ramp_stays_tilted():
    true = rectangle(5.0, 3.0)
    start = drifted(true, 0, 40.0)
    solved = pg.solve(start, pg.odometry(start), up=UP, gravity=False, height=False)
    np.testing.assert_allclose(tilt_deg(solved.poses), tilt_deg(start), atol=1e-6)


@settings(max_examples=30, deadline=None)
@given(
    st.integers(0, 62),
    st.integers(1, 30),
    st.tuples(st.floats(-1, 1), st.floats(-1, 1), st.floats(-1, 1)).filter(lambda v: np.linalg.norm(v) > 0.1),
    st.floats(45.0, 180.0),
)
def test_a_false_loop_doesnt_fold_the_walk_and_a_true_one_still_closes_it(i, gap, axis, wrong_deg):
    true = rectangle(5.0, 3.0)  # 64 poses
    j = min(i + gap, len(true) - 1)
    start = drifted(true, 1, 20.0)
    false = pg.relative(true[i], true[j])
    false[:3, :3] = (
        Rotation.from_rotvec(np.radians(wrong_deg) * np.asarray(axis) / np.linalg.norm(axis)).as_matrix()
        @ (false[:3, :3])
    )
    edges = [*pg.odometry(start), closing_loop(true), pg.Edge(i, j, false, pg.LOOP_DEG, pg.LOOP_M, loop=True)]
    solved = pg.solve(start, edges)
    assert solved.dropped == (len(edges) - 1,)
    assert position_error(solved.poses, true) < 0.25 * position_error(start, true)


def test_the_height_prior_pulls_a_sinking_walk_level_and_can_be_turned_off():
    true = rectangle(5.0, 3.0)
    sag = true.copy()
    sag[:, 2, 3] -= np.linspace(0.0, 0.6, len(true))  # the odometry sinks 0.6 m over the walk
    edges = pg.odometry(sag)
    assert np.ptp(pg.solve(sag, edges, up=UP).poses[:, 2, 3]) < 0.7 * np.ptp(sag[:, 2, 3])  # a prior, not a rule
    np.testing.assert_allclose(pg.solve(sag, edges, up=UP, height=False).poses, sag, atol=1e-9)


def test_the_height_prior_levels_a_walk_at_its_own_height_not_where_the_drift_put_it():
    true = rectangle(5.0, 3.0)
    sag = true.copy()
    sag[:, 2, 3] -= np.linspace(0.0, 3.0, len(true))  # the odometry sinks 3 m, so the drifted median is 1.5 m low
    edges = [*pg.odometry(true), closing_loop(true)]
    solved = pg.solve(sag, edges, up=UP).poses
    assert np.abs(solved[:, 2, 3] - 1.5).max() < 0.05, "every camera at the first one's height, the first step too"


def pair(back_rot_deg=0.0, back_dir_deg=0.0, tilt=0.0, shift=(1.0, 0.0, 0.5)):
    """A MASt3R pair: camera j 1.1 m from camera i and turned 30 degrees, both ways round, with the backward
    direction's rotation and translation direction knocked off by the given degrees."""
    forward = np.eye(4)
    forward[:3, :3] = (
        Rotation.from_euler("x", tilt, degrees=True).as_matrix()
        @ Rotation.from_euler("y", 30, degrees=True).as_matrix()
    )
    forward[:3, 3] = shift
    back = np.linalg.inv(forward)
    back[:3, :3] = Rotation.from_euler("y", back_rot_deg, degrees=True).as_matrix() @ back[:3, :3]
    # Turn the backward translation direction (as seen from i: -R_b^T t_b) by back_dir_deg about up.
    seen = -back[:3, :3].T @ back[:3, 3]
    seen = Rotation.from_euler("y", back_dir_deg, degrees=True).as_matrix() @ seen
    back[:3, 3] = -back[:3, :3] @ seen
    return forward, back


@pytest.mark.parametrize(
    ("change", "matches", "refused"),
    [
        pytest.param({}, 150, None, id="agreeing-at-the-floor"),
        pytest.param({}, 149, "matches", id="under-the-floor"),
        pytest.param({"back_rot_deg": 15.0}, 400, "rotations disagree", id="rotations-disagree"),
        pytest.param({"back_dir_deg": 30.0}, 400, "point", id="translations-disagree"),
        pytest.param({"tilt": 30.0}, 400, "tilts", id="up-axes-apart"),
        pytest.param({"shift": (20.0, 0.0, 10.0)}, 400, "baseline", id="not-house-sized"),
    ],
)
def test_a_pair_becomes_a_loop_edge_only_when_both_directions_agree(change, matches, refused):
    forward, back = pair(**change)
    guess = forward.copy()
    guess[:3, 3] *= 2.0  # the odometry's own scale: twice MASt3R's, as `scale` says
    reason = pg.loop_refusal(forward, back, matches, guess, scale=2.0)
    assert (reason is None) if refused is None else (refused in (reason or ""))


def test_a_pair_far_off_the_drifted_odometry_is_refused():
    forward, back = pair()
    guess = np.eye(4)
    guess[:3, 3] = (2.0 + 5.0, 0.0, 1.0)  # 5 m off the scaled pair
    assert "off the odometry" in (pg.loop_refusal(forward, back, 400, guess, scale=2.0) or "")


def test_a_loop_edge_carries_the_pair_in_the_odometry_scale():
    forward, _ = pair()
    edge = pg.loop_edge(forward, 3, 40, scale=2.0)
    np.testing.assert_allclose(edge.measured[:3, 3], (2.0, 0.0, 1.0))
    np.testing.assert_allclose(edge.measured[:3, :3], forward[:3, :3])
    assert (edge.i, edge.j, edge.loop) == (3, 40, True)


def test_the_scale_is_the_median_ratio_and_a_spread_one_is_refused():
    assert pg.loop_scale([2.0, 4.1, 0.9, 3.0], [1.0, 2.0, 0.5, 1.5]) == pytest.approx(2.0, rel=0.02)
    with pytest.raises(ValueError, match="spread"):
        pg.loop_scale([1.0, 1.0, 1.0, 4.0, 4.0, 4.0], [1.0] * 6)
    with pytest.raises(ValueError, match="no lengths"):
        pg.loop_scale([0.0], [0.0])


def wall(c2w, flip=False):
    """What a camera sees of a wall 2 m ahead, as a world pointmap (rows x columns x 3), row 0 at the image's top."""
    v, u = np.mgrid[-1.0:1.0:12j, -1.5:1.5:16j]
    if flip:
        v = -v  # a convention where image rows run up the camera's y
    camera = np.stack([u, v, np.full_like(u, 2.0)], axis=-1)
    return camera @ c2w[:3, :3].T + c2w[:3, 3]


def test_the_up_axis_check_passes_a_camera_to_world_pose_with_y_down():
    pose = upright(0.7, (1.0, 2.0, 1.5)) @ about(0, math.radians(12))  # looking a little down
    assert pg.up_axis_error(pose, wall(pose)) < 1.0


def test_the_up_axis_check_flags_y_up_and_a_world_to_camera_pose():
    pose = upright(0.7, (1.0, 2.0, 1.5)) @ about(0, math.radians(12))
    assert pg.up_axis_error(pose, wall(pose, flip=True)) > 170.0
    assert pg.up_axis_error(np.linalg.inv(pose), wall(pose)) > 30.0


def test_huber_alone_bounds_a_false_loops_pull(monkeypatch):
    """With the post-solve drop switched off, Huber is what's left: a loop 90 degrees off bends the walk far less
    than it does under plain least squares."""
    monkeypatch.setattr(pg, "REJECT", math.inf)
    true = rectangle(5.0, 3.0)
    false = pg.relative(true[8], true[34])
    false[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix() @ false[:3, :3]
    edges = [*pg.odometry(true), pg.Edge(8, 34, false, pg.LOOP_DEG, pg.LOOP_M, loop=True)]
    robust = position_error(pg.solve(true, edges).poses, true)
    monkeypatch.setattr(pg, "HUBER", 1e9)
    plain = position_error(pg.solve(true, edges).poses, true)
    assert robust < 0.5 * plain


def facing(true, grid_deg=0.0, quarters=None):
    """Each frame's wall direction in its own camera frame: the world's x axis turned `grid_deg` about up, and for
    frame k `quarters[k]` quarter turns more (any of a square room's four walls)."""
    turns = np.zeros(len(true)) if quarters is None else np.asarray(quarters)
    angles = math.radians(grid_deg) + turns * math.pi / 2
    world = np.stack([np.cos(angles), np.sin(angles), np.zeros(len(true))], axis=1)
    return np.einsum("nji,nj->ni", true[:, :3, :3], world)


@pytest.mark.parametrize("grid_deg", [0.0, 20.0, 44.0])
def test_walls_hold_the_heading_of_a_walk_that_never_comes_back(grid_deg):
    true = rectangle(6.0, 4.0)[:60]  # three sides: no revisit
    start = drifted(true, 1, 30.0)
    edges = pg.odometry(start)
    assert position_error(pg.solve(start, edges, up=UP).poses, true) > 1.0, "nothing else holds it"
    solved = pg.solve(start, edges, up=UP, walls=facing(true, grid_deg))
    assert np.array_equal(solved.poses[0], start[0]), "the gauge"
    assert position_error(solved.poses, true) < 0.15


def test_any_of_the_four_walls_holds_a_frame_the_same_way():
    true = rectangle(6.0, 4.0)[:60]
    start = drifted(true, 1, 30.0)
    one = pg.solve(start, pg.odometry(start), up=UP, walls=facing(true))
    any_wall = pg.solve(start, pg.odometry(start), up=UP, walls=facing(true, quarters=np.arange(60) % 4))
    assert any_wall.poses == pytest.approx(one.poses, abs=1e-6)


def test_a_few_slanted_walls_and_frames_with_none_dont_turn_the_walk():
    true = rectangle(6.0, 4.0)[:60]
    start = drifted(true, 1, 30.0)
    walls = facing(true)
    walls[10:60:10] = facing(true, 30.0)[10:60:10]  # five frames see a wall 30 degrees off the rest
    walls[5:60:10] = np.nan
    solved = pg.solve(start, pg.odometry(start), up=UP, walls=walls)
    assert position_error(solved.poses, true) < 0.3
