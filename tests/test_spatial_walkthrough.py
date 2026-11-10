"""Walkthrough video to place (#2787, #2931): frame choice, revisit proposals, loop closure, the CUT3R and MASt3R jobs,
and the place written. Neither model runs here: fake runtimes stand in for them at the subprocess boundary."""

import itertools
import json
import math
import os
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest

import stray_fake
from liberational import config as cfg
from liberational.memory import places
from liberational.spatial import cut3r_runner, fuse, ply, splats, stray
from liberational.spatial import posegraph as pg
from liberational.spatial import walkthrough as W

T = 1_800_000_000.0


@pytest.fixture(autouse=True)
def no_splats():
    """No walk trains a splat unless its test says so: a checkout's own gsplat runtime would train one for real."""
    with cfg.override({"spatial": {"splats_enabled": False}}):
        yield


def frames(seconds, moves, sharp=None):
    """Scored frames, 10 a second for `seconds`, the view moving `moves` (fraction of a frame width) between frames."""
    out, moved = [], 0.0
    for i in range(seconds * 10):
        moved += moves if i else 0.0
        out.append(W.Scored(i, i / 10, moved, (sharp or {}).get(i, 100.0)))
    return out


def test_a_walk_gets_the_sharpest_frame_of_each_second():
    got = W.pick(frames(4, 0.03, sharp={3: 150.0, 17: 120.0, 25: 300.0}))
    assert got == [3, 17, 25, 30]  # the first frame of the last second, all equally sharp


def test_a_second_in_which_the_view_barely_moved_adds_no_frame():
    moving = [W.Scored(i, i / 10, 0.03 * i, 200.0 if i == 9 else 100.0) for i in range(10)]  # frame 9 taken
    standing = [W.Scored(10 + i, 1 + i / 10, 0.27 + 0.001 * i, 100.0) for i in range(10)]
    on = [W.Scored(20 + i, 2 + i / 10, 0.4 + 0.03 * i, 100.0) for i in range(10)]
    assert W.pick(moving + standing + on) == [9, 20]


def test_a_long_walk_is_held_to_the_cap_by_wider_seconds():
    got = W.pick(frames(100, 0.03), cap=10)
    assert len(got) == 10
    assert got[-1] >= 900


def test_a_standing_video_has_one_frame_and_nothing_at_all_has_none():
    assert W.pick(frames(5, 0.0)) == [0]
    assert W.pick([]) == []


def upright(heading, x, z):
    """A level camera (OpenCV axes) in a y-down world, at (x, 0, z), looking along `heading` (radians from +z)."""
    c, s = math.cos(heading), math.sin(heading)
    pose = np.eye(4)
    pose[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    pose[:3, 3] = [x, 0.0, z]
    return pose


def rectangle(width=4.0, depth=3.0, step=0.25):
    """A walk once round a rectangle, facing the way it goes, ending a step short of the start."""
    corners = [(0.0, 0.0), (0.0, depth), (width, depth), (width, 0.0), (0.0, 0.0)]
    poses = []
    for (ax, az), (bx, bz) in itertools.pairwise(corners):
        length = math.hypot(bx - ax, bz - az)
        heading = math.atan2(bx - ax, bz - az)
        for k in range(round(length / step)):
            u = k * step / length
            poses.append(upright(heading, ax + u * (bx - ax), az + u * (bz - az)))
    return np.array(poses)


def drifted(true, deg_per_step):
    """The walk as a drifting odometry sees it: every step's own motion, then a turn of `deg_per_step` more."""
    turn = upright(math.radians(deg_per_step), 0.0, 0.0)
    out = [true[0]]
    for k in range(len(true) - 1):
        out.append(out[-1] @ pg.relative(true[k], true[k + 1]) @ turn)
    return np.array(out)


DEPTH = 4.0  # every frame's median depth as CUT3R sees it


def measured(true, pairs, scale):
    """MASt3R's answer for `pairs`: the true relative poses both ways, its translations and depths `scale` times the
    walk's (each frame DEPTH deep, as CUT3R sees it)."""
    out = []
    for i, j in pairs:
        fwd, back = pg.relative(true[i], true[j]), pg.relative(true[j], true[i])
        fwd[:3, 3] *= scale
        back[:3, 3] *= scale
        out.append(
            {
                "a": i,
                "b": j,
                "matches": 600,
                "depths": [DEPTH * scale, DEPTH * scale],
                "inliers": [540, 520],
                "forward": fwd.ravel().tolist(),
                "backward": back.ravel().tolist(),
            }
        )
    return out


def steps(n):
    return [(k, k + 1) for k in range(n - 1)]


def seen(true):
    """Each frame's view of the walls of a box round the rectangle walk, in its own camera frame."""
    rng = np.random.default_rng(3)
    u = rng.uniform(0, 1, 4000)
    h = rng.uniform(-1.0, 1.0, 4000)
    walls = np.concatenate(
        [
            np.c_[-1 + 6 * u[:1000], h[:1000], np.full(1000, -1.0)],
            np.c_[-1 + 6 * u[1000:2000], h[1000:2000], np.full(1000, 4.0)],
            np.c_[np.full(1000, -1.0), h[2000:3000], -1 + 5 * u[2000:3000]],
            np.c_[np.full(1000, 5.0), h[3000:], -1 + 5 * u[3000:]],
        ]
    )
    return [(walls - pose[:3, 3]) @ pose[:3, :3] for pose in true]


def test_a_revisit_closes_the_drifted_walk_in_the_odometrys_scale():
    true = rectangle()
    walk = drifted(true, 0.6)
    n = len(true)
    loops = [(0, n - 1), (1, n - 2)]
    closed = W.close_loops(walk, measured(true, loops + steps(n), 0.5), loops, np.full(n, DEPTH), gravity=True)
    assert closed.scale == pytest.approx(2.0)
    assert closed.repaired == [], "a drift of 0.6 degrees a step isn't a break"
    assert closed.loops == loops and closed.dropped == [] and closed.refused == {}
    assert np.linalg.norm(walk[-1, :3, 3] - true[-1, :3, 3]) > 1.0
    assert np.linalg.norm(closed.poses[-1, :3, 3] - true[-1, :3, 3]) < 0.1


def test_a_step_where_cut3r_spun_takes_mastr3rs_measurement():
    true = rectangle()
    n = len(true)
    walk = np.array(true)
    walk[20:] = walk[20] @ upright(math.radians(70), 0.0, 0.0) @ np.linalg.inv(walk[20]) @ walk[20:]  # a 70 degree spin
    closed = W.close_loops(walk, measured(true, steps(n), 0.5), [], np.full(n, DEPTH), gravity=True, clouds=seen(true))
    assert closed.repaired == [19] and closed.reverted == []
    assert np.abs(closed.poses[:, :3, 3] - true[:, :3, 3]).max() < 0.05


def test_a_sure_but_wrong_mastr3r_step_isnt_repaired_when_cut3rs_points_fit_its_own_step():
    true = rectangle()
    n = len(true)
    loops = [(0, n - 1), (1, n - 2), (2, n - 3)]
    answers = measured(true, loops + steps(n), 0.5)
    wrong = pg.relative(true[19], true[20]) @ upright(math.radians(60), 0.0, 0.0)  # MASt3R sees a turn there wasn't
    answers[len(loops) + 19] = {
        **answers[len(loops) + 19],
        "forward": wrong.ravel().tolist(),
        "backward": np.linalg.inv(wrong).ravel().tolist(),
    }
    closed = W.close_loops(true, answers, loops, np.full(n, DEPTH), gravity=True, clouds=seen(true))
    assert closed.repaired == [], "CUT3R's own points fit its step, not MASt3R's"
    assert closed.loops == loops
    assert np.abs(closed.poses[:, :3, 3] - true[:, :3, 3]).max() < 0.05


def test_a_wrong_match_far_along_a_long_walk_is_refused_by_the_capped_gate():
    true = rectangle(20.0, 15.0)  # 280 frames, 70 m round
    n = len(true)
    loops = [(0, 250)]
    wrong = pg.relative(true[0], true[250]) @ upright(math.radians(90), 0.0, 0.0)
    wrong[:3, 3] += [4.0, 0.0, 6.0]  # where frame 250 isn't, 90 degrees off
    answer = {
        **measured(true, loops, 1.0)[0],
        "forward": wrong.ravel().tolist(),
        "backward": np.linalg.inv(wrong).ravel().tolist(),
    }
    closed = W.close_loops(true, measured(true, steps(n), 1.0) + [answer], loops, np.full(n, DEPTH), gravity=True)
    assert closed.loops == [] and closed.refused == {"# m and # degrees off the odometry": 1}


def test_a_long_walk_drifted_past_the_gates_cap_closes_where_its_return_frames_agree():
    true = rectangle(20.0, 15.0)  # 280 frames, 70 m round
    walk = drifted(true, 0.6)  # 167 degrees off by the end: past the gate's cap
    n = len(true)
    loops = [(0, n - 1), (3, n - 4)]
    closed = W.close_loops(walk, measured(true, loops + steps(n), 0.5), loops, np.full(n, DEPTH), gravity=True)
    assert closed.loops == loops and closed.supported == 2 and closed.refused == {}
    assert max(closed.drift_deg) == pytest.approx(0.6 * (n - 1), abs=3.0)
    assert np.linalg.norm(walk[-1, :3, 3] - true[-1, :3, 3]) > 5.0
    assert np.linalg.norm(closed.poses[-1, :3, 3] - true[-1, :3, 3]) < 0.1


def test_two_revisits_past_the_gate_that_disagree_dont_support_each_other():
    true = rectangle(20.0, 15.0)
    walk = drifted(true, 0.6)
    n = len(true)
    loops = [(0, n - 1), (3, n - 4)]
    answers = measured(true, loops + steps(n), 0.5)
    wrong = pg.relative(true[3], true[n - 4]) @ upright(math.radians(90), 0.0, 0.0)
    wrong[:3, 3] += [2.0, 0.0, 3.0]  # 0.5 scale: 4 m and 6 m from where frame n - 4 is
    answers[1] = {**answers[1], "forward": wrong.ravel().tolist(), "backward": np.linalg.inv(wrong).ravel().tolist()}
    closed = W.close_loops(walk, answers, loops, np.full(n, DEPTH), gravity=True)
    assert closed.loops == [] and closed.supported == 0
    assert closed.refused == {"# m and # degrees off the odometry": 2}


def test_a_revisit_two_laps_on_passes_the_gate_once_the_nearer_ones_are_closed():
    true = np.concatenate([rectangle()] * 3)  # three laps of 56 frames
    walk = drifted(true, 0.6)  # a lap on is 34 degrees off, two laps on 67: past the cap
    n = len(true)
    near = [(5, 62), (20, 77), (40, 97), (60, 117), (80, 137)]  # a lap and a step on (a lap on is the same pose)
    far = [(10, 123), (30, 143)]
    loops = sorted(near + far)
    closed = W.close_loops(walk, measured(true, loops + steps(n), 0.5), loops, np.full(n, DEPTH), gravity=True)
    assert sorted(closed.loops) == loops and closed.supported == 0, "the far ones are too far apart for support"
    assert np.abs(walk[:150, :3, 3] - true[:150, :3, 3]).max() > 2.0
    assert np.abs(closed.poses[:150, :3, 3] - true[:150, :3, 3]).max() < 0.5


def test_mast3r_isnt_used_when_the_two_models_disagree_on_the_scene():
    true = rectangle()
    walk = drifted(true, 0.6)
    n = len(true)
    loops = [(0, n - 1)]
    answers = measured(true, loops + steps(n), 0.5)
    for k, m in enumerate(answers):
        m["depths"] = [DEPTH * (0.2 if k % 2 else 1.0)] * 2  # half the pairs see the scene five times as deep
    closed = W.close_loops(walk, answers, loops, np.full(n, DEPTH), gravity=True)
    assert closed.scale is None and "spread" in (closed.why_no_scale or "")
    assert closed.loops == [] and closed.repaired == []
    assert np.linalg.norm(closed.poses[-1, :3, 3] - true[-1, :3, 3]) > 1.0


def off(pose, deg, dx, dz):
    """`pose` turned `deg` about its y axis and shifted (dx, 0, dz): an error a level walk's own walls can show."""
    return pose @ upright(math.radians(deg), dx, dz)


def test_icp_fits_a_frames_walls_onto_another_frames_from_a_start_half_a_metre_off():
    true = rectangle()
    walls = seen(true)
    truth = pg.relative(true[0], true[30])
    pose, fit = W.icp(walls[0], walls[30], off(truth, 8.0, 0.4, -0.3))
    assert fit > 0.95
    assert np.abs(pose - truth).max() < 0.01


def test_icp_fit_is_low_when_the_two_frames_see_different_walls():
    true = rectangle()
    walls = seen(true)
    elsewhere = walls[30] + [0.0, 0.0, 20.0]  # the same walls, 20 m on: nothing within reach to fit to
    _, fit = W.icp(walls[0], elsewhere, pg.relative(true[0], true[30]))
    assert fit < W.ICP_FIT


def test_icp_with_too_few_points_keeps_the_start_and_fits_nothing():
    start = upright(0.3, 1.0, 2.0)
    pose, fit = W.icp(np.zeros((5, 3)), np.ones((500, 3)), start)
    assert fit == 0.0
    assert pose == pytest.approx(start)


def test_a_revisit_whose_near_points_meet_is_closed_on_their_fit_not_mast3rs_pose():
    true = rectangle()
    walk = drifted(true, 0.6)
    n = len(true)
    loops = [(0, n - 1)]
    answers = measured(true, steps(n), 0.5)
    loop = measured(true, loops, 0.5)[0]
    wrong = off(pg.relative(true[0], true[n - 1]), 5.0, 0.3, 0.2)  # MASt3R a little off, within its checks
    wrong[:3, 3] *= 0.5
    answers.append({**loop, "forward": wrong.ravel().tolist(), "backward": np.linalg.inv(wrong).ravel().tolist()})
    depths = np.full(n, DEPTH)
    plain = W.close_loops(walk, answers, loops, depths, gravity=True)
    fitted = W.close_loops(walk, answers, loops, depths, gravity=True, clouds=seen(true))
    truth = pg.relative(true[0], true[n - 1])
    assert plain.fitted == {} and plain.loops == loops
    assert list(fitted.fitted) == loops and fitted.fitted[loops[0]] > 0.95
    assert fitted.loops == loops
    assert np.linalg.norm(pg.relative(plain.poses[0], plain.poses[n - 1])[:3, 3] - truth[:3, 3]) > 0.15
    assert np.linalg.norm(pg.relative(fitted.poses[0], fitted.poses[n - 1])[:3, 3] - truth[:3, 3]) < 0.05


def test_a_revisit_the_drift_gate_refuses_is_still_closed_when_its_near_points_meet():
    true = rectangle()
    walk = drifted(true, 3.0)  # drifted 150 degrees by the end: MASt3R's true pose is far off the walk
    n = len(true)
    loops = [(10, 40)]
    answers = measured(true, loops, 0.5)  # no steps measured: none repaired, and the gate sees only the drift
    gated = W.close_loops(walk, answers, loops, np.full(n, DEPTH), gravity=False)
    fitted = W.close_loops(walk, answers, loops, np.full(n, DEPTH), gravity=False, clouds=seen(true))
    assert gated.loops == [] and gated.refused == {"# m and # degrees off the odometry": 1}
    assert fitted.loops == loops
    got = pg.relative(fitted.poses[10], fitted.poses[40])
    assert np.linalg.norm(got[:3, 3] - pg.relative(true[10], true[40])[:3, 3]) < 0.1


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        pytest.param({"matches": 40}, "# matches (needs #)", id="too-few-matches"),
        pytest.param({"forward": None}, "no pose: MASt3R isn't sure of the pair", id="no-pose"),
        pytest.param({"depths": [2.0, 6.0]}, "the two frames' depths give different scales", id="depths-disagree"),
        pytest.param({"inliers": [500, 100]}, "PnP keeps too few of its matches", id="pnp-disagrees"),
    ],
)
def test_a_pair_the_checks_refuse_is_not_a_loop(change, reason):
    true = rectangle()
    walk = drifted(true, 0.6)
    n = len(true)
    loops = [(0, n - 1)]
    answers = measured(true, steps(n), 0.5) + [{**measured(true, loops, 0.5)[0], **change}]
    closed = W.close_loops(walk, answers, loops, np.full(n, DEPTH), gravity=True)
    assert closed.loops == [] and closed.refused == {reason: 1}


@pytest.mark.parametrize(
    ("depths", "want"),
    [
        pytest.param((4.0, 6.0, 2.0, 3.0), 2.0, id="both-twice-as-deep"),
        pytest.param((4.0, 5.0, 2.0, 2.0), math.sqrt(2.0 * 2.5), id="within-the-agreement"),
        pytest.param((4.0, 6.0, 2.0, 1.5), None, id="frames-disagree"),
        pytest.param((4.0, 6.0, 0.0, 3.0), None, id="no-depth"),
    ],
)
def test_a_pairs_scale_is_how_much_deeper_cut3r_sees_its_frames(depths, want):
    got = W.pair_scale(*depths)
    assert got == (None if want is None else pytest.approx(want))


def test_each_frames_points_move_with_its_own_pose():
    points = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, 0, 2.0]])
    poses = np.stack([np.eye(4), upright(math.pi / 2, 5.0, 0.0)])  # the second looks along +x from x = 5
    got = W.rebuild(points, [1, 2], poses)
    assert got == pytest.approx(np.array([[1.0, 0, 0], [6.0, 0, 0], [7.0, 0, 0]]))


def pan_video(path, n=150, still=False):
    """A synthetic video panning across a textured scene (or standing still on it)."""
    import cv2

    rng = np.random.default_rng(1)
    scene = cv2.GaussianBlur((rng.uniform(0, 255, (240, 1600, 3))).astype(np.uint8), (0, 0), 2)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 30, (320, 240))
    for i in range(n):
        x = 0 if still else i * 5
        writer.write(np.ascontiguousarray(scene[:, x : x + 320]))
    writer.release()
    return path


def test_a_panning_video_scores_its_motion_and_saves_its_frames_in_time_order(tmp_path):
    video = pan_video(tmp_path / "pan.mp4", n=90)
    scored = W.score(video)
    assert len(scored) == 30  # 90 frames at 30 fps, sampled 10 a second
    assert scored[-1].moved == pytest.approx(445 / 320, rel=0.15)  # it panned 445 px across a 320 px frame
    picked = W.pick(scored)
    assert len(picked) == 3  # one a second
    saved = W.save_frames(video, picked, tmp_path)
    assert sorted(saved) == ["f000", "f001", "f002"]


def room():
    """A 4 x 3 m room 2.6 m high, y down: floor at y = 0."""
    rng = np.random.default_rng(0)
    n = 4000
    wall_x = np.r_[rng.uniform(0, 4, n), rng.uniform(0, 4, n), np.zeros(n), np.full(n, 4.0)]
    wall_z = np.r_[np.zeros(n), np.full(n, 3.0), rng.uniform(0, 3, n), rng.uniform(0, 3, n)]
    wall_h = rng.uniform(0, 2.6, 4 * n)
    floor = np.c_[rng.uniform(0, 4, n), rng.uniform(0, 3, n)]
    xs, zs, hs = np.r_[wall_x, floor[:, 0]], np.r_[wall_z, floor[:, 1]], np.r_[wall_h, np.zeros(n)]
    return np.c_[xs, -hs, zs].astype(np.float32)


UP = np.array([0.0, -1.0, 0.0])  # room()'s up: its y is down


def test_a_walks_floor_is_level_with_its_cameras_at_the_floors_height():
    rng = np.random.default_rng(1)
    ceiling = np.c_[rng.uniform(0, 4, 8000), np.full(8000, -2.6), rng.uniform(0, 3, 8000)]  # twice the floor's points
    floor = W.level_floor(np.r_[room(), ceiling], UP)
    assert floor is not None
    assert floor[:3] == pytest.approx(UP)
    assert np.array([2.0, -1.5, 1.0]) @ floor[:3] + floor[3] == pytest.approx(1.5, abs=0.03)  # 1.5 m over the floor


def test_a_tilted_surface_with_more_points_doesnt_tilt_a_walks_floor(tmp_path):
    rng = np.random.default_rng(2)
    x, z = rng.uniform(4, 16, 12000), rng.uniform(0, 3, 12000)
    slope = np.c_[x, -math.tan(math.radians(12)) * (x - 4), z]  # 12 degrees off level, rising away from the room
    points = np.r_[room(), slope].astype(np.float32)
    colors = np.full((len(points), 3), 128, dtype=np.uint8)
    rev = W.write_place(tmp_path, None, "walk.mp4", points, colors, (0.0, -1.0, 0.0))
    assert rev.floor is not None
    assert rev.floor[:3] == pytest.approx(UP)
    assert np.array([2.0, 0.0, 1.0]) @ rev.floor[:3] + rev.floor[3] == pytest.approx(0.0, abs=0.03)


def test_too_few_points_have_no_floor():
    assert W.level_floor(np.zeros((10, 3)), UP) is None


def python(runtime):
    """A runtime's venv python that runs this test's own interpreter, with its packages."""
    exe = runtime / ".venv" / "bin" / "python"
    exe.write_text(f'#!/bin/sh\nexec {sys.executable} "$@"\n')
    exe.chmod(0o755)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    """CUT3R's and MASt3R's runtimes with fake runners. CUT3R's walks frame k to x = 0.5 + 0.1 k, 1.5 m up, every
    frame seeing all of `room()` from 1 m outside it; MASt3R's answers every pair with the true relative pose at half scale. Both write
    their job and niceness into their answer, and CUT3R's fails when FAKE_FAIL is set."""
    from liberational import config as cfg
    from liberational.spatial import cameras

    m3 = tmp_path / "mast3r"
    (m3 / ".venv" / "bin").mkdir(parents=True)
    (m3 / "src" / "mast3r").mkdir(parents=True)
    python(m3)
    c3 = tmp_path / "cut3r"
    (c3 / ".venv" / "bin").mkdir(parents=True)
    (c3 / "src" / "src" / "dust3r").mkdir(parents=True)
    python(c3)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(b"")
    (ckpt / "cut3r.pth").write_bytes(b"")
    np.save(tmp_path / "room.npy", room())
    fake_cut3r = tmp_path / "cut3r_runner.py"
    fake_cut3r.write_text(
        textwrap.dedent(
            f"""
            import json, os, sys
            import numpy as np
            job = json.load(open(sys.argv[1]))
            if os.environ.get("FAKE_FAIL"):
                sys.exit("CUDA out of memory")
            n = len(job["images"])
            world = np.load({str(tmp_path / "room.npy")!r})
            c2w = np.stack([np.eye(4)] * n)
            c2w[:, 0, 3] = 0.5 + 0.1 * np.arange(n)
            c2w[:, 1, 3] = -1.5
            c2w[:, 2, 3] = -1.0
            rows, cols = np.mgrid[0:8, 0:6]
            grid = np.stack([(cols - 2.5) * 0.3, (rows - 3.5) * 0.3, np.full(rows.shape, 2.0)], -1)
            np.savez(os.path.join(job["out"], "result.npz"), c2w=c2w, focal=np.full(n, 300.0), size=np.array([288, 512]),
                     points=np.concatenate([world - c2w[k, :3, 3] for k in range(n)]).astype(np.float32),
                     colors=np.full((n * len(world), 3), 200, np.uint8), counts=np.full(n, len(world)),
                     grid=np.stack([grid] * n), world_grid=np.stack([grid + c2w[k, :3, 3] for k in range(n)]))
            json.dump({{"frames": n, "seconds": 1.0, "peak_gib": 0.1, "job": job, "nice": os.nice(0)}},
                      open(os.path.join(job["out"], "result.json"), "w"))
            """
        )
    )
    fake_mast3r = tmp_path / "mast3r_runner.py"
    fake_mast3r.write_text(
        textwrap.dedent(
            """
            import json, os, sys
            job = json.load(open(sys.argv[1]))
            depth = HALF_DEPTH
            pairs = []
            for a, b in job["pairs"]:
                dx = 0.05 * (int(b[1:]) - int(a[1:]))
                fwd = [1.0, 0, 0, dx, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0]
                back = [1.0, 0, 0, -dx, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0]
                pairs.append({"a": a, "b": b, "matches": 500, "forward": fwd, "backward": back, "depths": [depth, depth],
                              "scales": [1.0, 1.0], "inliers": [400, 400]})
            out = {"pairs": pairs, "job": job, "nice": os.nice(0), "seconds": 1.0}
            json.dump(out, open(os.path.join(job["out"], "result.json"), "w"))
            """
        ).replace("HALF_DEPTH", repr(float(np.median(room()[:, 2] + 1.0)) / 2))  # CUT3R's frames' depth, halved
    )
    monkeypatch.setattr(W, "RUNNER", fake_cut3r)
    monkeypatch.setattr(cameras, "RUNNER", fake_mast3r)
    monkeypatch.setattr(cameras, "checkpoint", lambda: ckpt)
    real_path, real_get, real_model = cfg.path, cfg.get, cfg.model_path
    runtimes = {"spatial.runtime": m3, "spatial.cut3r_runtime": c3}
    monkeypatch.setattr(cfg, "path", lambda key: runtimes.get(key) or real_path(key))
    monkeypatch.setattr(cfg, "get", lambda key, *a: -1 if key == "spatial.gpu" else real_get(key, *a))
    monkeypatch.setattr(cfg, "model_path", lambda name: ckpt / "cut3r.pth" if name == "cut3r" else real_model(name))
    monkeypatch.setattr(W.nvml, "gpus", list)
    return tmp_path


def test_a_video_becomes_a_place_with_its_cloud_floor_and_plan(tmp_path, runtime):
    did = W.run(tmp_path, pan_video(tmp_path / "walk.mp4"), poses_out=tmp_path / "poses.npz")
    rev = places.latest(tmp_path, did["place"])
    assert rev is not None
    assert rev.signals["walkthrough"] == ["walk.mp4"]
    assert rev.map_format == "ply" and places.read_map(tmp_path, rev)
    assert rev.floor is not None and rev.plan is not None and rev.plan_transform is not None
    assert did["frames"] == 5 and did["gravity"] and did["scale"] == pytest.approx(2.0)
    with np.load(tmp_path / "poses.npz") as saved:
        assert saved["c2w"].shape == (5, 4, 4)
        assert saved["c2w"][:, 0, 3] == pytest.approx(0.5 + 0.1 * np.arange(5), abs=1e-6)


def test_both_models_run_niced_on_the_frames_in_walk_order(tmp_path, runtime, monkeypatch):
    seen = {}
    real_cut3r, real_mast3r = W.run_cut3r, W.cameras.run_mast3r

    def cut3r(runtime_dir, images, out, *a, **kw):
        got = real_cut3r(runtime_dir, images, out, *a, **kw)
        seen["cut3r"] = json.loads((out / "result.json").read_text())
        return got

    def mast3r(*a, **kw):
        seen["mast3r"] = real_mast3r(*a, **kw)
        return seen["mast3r"]

    monkeypatch.setattr(W, "run_cut3r", cut3r)
    monkeypatch.setattr(W.cameras, "run_mast3r", mast3r)
    W.run(tmp_path, pan_video(tmp_path / "walk.mp4"))
    images = seen["cut3r"]["job"]["images"]
    assert [os.path.basename(p) for p in images] == [f"f{k:03d}.jpg" for k in range(5)]
    assert seen["cut3r"]["nice"] >= 10 and seen["mast3r"]["nice"] >= 10
    assert seen["mast3r"]["job"]["pairs"] == [["f000", "f001"], ["f001", "f002"], ["f002", "f003"], ["f003", "f004"]]


POSE = places.CameraPose(tuple(float(x) for x in np.eye(4).ravel()), 300.0, 640, 360, T)


def test_a_second_video_of_a_place_is_fused_into_its_map_and_its_fixed_cameras_keep_their_poses(tmp_path, runtime):
    first = W.run(tmp_path, pan_video(tmp_path / "a.mp4"))
    places.revise(tmp_path, first["place"], producer=fuse.PRODUCER, cameras={"s2": POSE}, now=T)  # a camera seated
    second = W.run(tmp_path, pan_video(tmp_path / "b.mp4"), place=first["place"])
    assert second["place"] == first["place"] and second["fused"] and second["registered"] == "map"
    producers = [r.producer for r in places.revisions(tmp_path, first["place"])]
    assert producers == [W.PRODUCER, fuse.PRODUCER, fuse.PRODUCER, fuse.PRODUCER], "no unregistered map in between"
    latest = places.latest(tmp_path, first["place"])
    assert latest is not None and latest.id == second["revision"]
    assert latest.cameras == {"s2": POSE}
    assert latest.signals["walkthrough"] == ["a.mp4", "b.mp4"]


def unrelated_place(state, cameras):
    """A place whose fused map is nothing like the walk's room: a 30 x 20 m yard's walls, knee high, well away from
    where the walk stands, with `cameras`."""
    rng = np.random.default_rng(5)
    t = rng.uniform(0, 1, 40_000)
    side = rng.integers(0, 4, 40_000)
    x = np.select([side == 0, side == 1, side == 2], [t * 30, t * 30, np.zeros_like(t)], np.full_like(t, 30.0))
    y = np.select([side == 0, side == 1, side == 2], [np.zeros_like(t), np.full_like(t, 20.0), t * 20], t * 20)
    scatter = np.c_[x - 45.0, y - 10.0, rng.uniform(0, 0.4, 40_000)]
    rev = places.revise(
        state,
        None,
        producer="cams",
        signals={"walkthrough": ["a.mp4"]},
        map_bytes=ply.dumps(scatter, np.full((len(scatter), 3), 90, np.uint8)),
        map_format="ply",
        floor=fuse.LEVEL_FLOOR,
        cameras=cameras,
        now=T,
    )
    fused = fuse.refine(state, rev.place, now=T + 1)
    assert fused is not None
    return fused


def no_frames(*a, **kw):
    return {}


def broker_down(*a, **kw):
    raise FileNotFoundError("No such file or directory (os error 2)")  # the broker's socket, while it restarts


@pytest.mark.parametrize(
    ("fused", "frames"),
    [
        pytest.param(True, no_frames, id="its-map-doesnt-fit-and-no-frame-seats-it"),
        pytest.param(True, broker_down, id="its-map-doesnt-fit-and-the-frames-cant-be-read"),
        pytest.param(False, no_frames, id="no-fused-map-yet"),
    ],
)
def test_a_walk_that_doesnt_register_in_a_place_is_its_own_place_and_the_place_keeps_its_cameras(
    tmp_path, runtime, monkeypatch, fused, frames
):
    if fused:
        old = unrelated_place(tmp_path, {"s2": POSE})
    else:
        old = places.revise(tmp_path, None, producer="cams", signals={"walkthrough": ["a.mp4"]}, cameras={"s2": POSE})
    monkeypatch.setattr(W.cameras, "frames", frames)  # s2's recorded frames, to seat it by
    did = W.run(tmp_path, pan_video(tmp_path / "b.mp4"), place=old.place)
    assert places.latest(tmp_path, old.place) == old
    assert places.camera_places(tmp_path) == {"s2": old.place}
    assert did["meant_for"] == old.place and did["place"] != old.place and did["registered"] is None
    alone = places.latest(tmp_path, did["place"])
    assert alone is not None and alone.signals["walkthrough"] == ["b.mp4"] and alone.map is not None
    assert did["join"]["why"] == "the walk didn't register in the place it was meant for"


def test_a_walk_its_places_map_cant_place_is_fused_where_the_fixed_cameras_seat_it(tmp_path, runtime, monkeypatch):
    old = unrelated_place(tmp_path, {"s2": POSE})
    there = np.eye(4)
    there[0, 3] = -20.0  # the place stands 20 m along -x in the walk's levelled frame: the walk 20 m along +x in it

    def seated(state, held, *a, **kw):
        assert set(held) == {old.place} and held[old.place].cameras == {"s2": POSE}
        return {old.place: there}, {old.place: {"seats": 3, "seated": True}}

    monkeypatch.setattr(W, "seat_places", seated)  # MASt3R's pairs of s2's frame with the walk's
    did = W.run(tmp_path, pan_video(tmp_path / "b.mp4"), place=old.place)
    assert did["place"] == old.place and did["registered"] == "cameras" and did["fused"]
    rev = places.latest(tmp_path, old.place)
    assert rev is not None and rev.cameras == {"s2": POSE} and rev.frame == old.frame
    assert rev.signals["walkthrough"] == ["a.mp4", "b.mp4"]
    cloud, _, _ = ply.read(places.root(tmp_path) / rev.place / str(rev.map))
    walls = cloud[(cloud[:, 0] > 19.9) & (cloud[:, 0] < 24.1)]
    assert len(walls) > 1000, "the walk's 4 m room, 20 m along x"
    assert np.ptp(walls[:, 0]) == pytest.approx(4.0, abs=0.1)


def test_a_failed_cut3r_run_changes_no_place(tmp_path, runtime, monkeypatch):
    first = W.run(tmp_path, pan_video(tmp_path / "good.mp4"))
    monkeypatch.setenv("FAKE_FAIL", "1")
    with pytest.raises(RuntimeError, match="CUT3R failed"):
        W.run(tmp_path, pan_video(tmp_path / "bad.mp4"), place=first["place"])
    latest = places.latest(tmp_path, first["place"])
    assert latest is not None and latest.id == first["revision"]


def test_a_gpu_without_room_for_the_models_fails_before_anything_runs(tmp_path, runtime, monkeypatch):
    gib = 2**30
    monkeypatch.setattr(W.nvml, "gpus", lambda: [{"index": 0, "uuid": "GPU-a", "total": 24 * gib, "used": 22 * gib}])
    with pytest.raises(RuntimeError, match="GB free"):
        W.run(tmp_path, pan_video(tmp_path / "walk.mp4"))
    assert places.place_ids(tmp_path) == []


def test_a_missing_cut3r_runtime_says_how_to_install_it(tmp_path, runtime):
    os.remove(runtime / "cut3r" / ".venv" / "bin" / "python")
    with pytest.raises(RuntimeError, match="mise run cut3r"):
        W.run(tmp_path, pan_video(tmp_path / "walk.mp4"))
    assert places.place_ids(tmp_path) == []


def test_a_video_that_doesnt_move_or_an_unknown_place_is_refused(tmp_path, runtime):
    with pytest.raises(RuntimeError, match="move through"):
        W.run(tmp_path, pan_video(tmp_path / "still.mp4", still=True))
    with pytest.raises(RuntimeError, match="no place"):
        W.run(tmp_path, pan_video(tmp_path / "x.mp4"), place="0190a000-0000-7000-8000-000000000000")
    assert places.place_ids(tmp_path) == []


def test_keyframes_only_saves_them_and_runs_nothing(tmp_path, runtime):
    did = W.run(tmp_path, pan_video(tmp_path / "walk.mp4"), keyframes_only=tmp_path / "kf")
    assert did["keyframes"] == len(list((tmp_path / "kf").glob("f*.jpg"))) == 5
    assert places.place_ids(tmp_path) == []
    assert json.dumps(did)


def test_cut3rs_rope_turns_each_half_by_its_position_even_a_negative_one():
    torch = pytest.importorskip("torch")
    tokens = torch.tensor([[[[1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]]])  # batch, head, token, 8 features
    positions = torch.tensor([[[-1, 2]]])  # the token's y and x
    got = cut3r_runner.rope_forward(SimpleNamespace(base=100.0), tokens, positions)
    # Each half has two frequencies, 1 and 100^-0.5 = 0.1; feature k pairs with feature k + 2 in its half.
    want = [math.cos(-1), math.cos(-0.1), math.sin(-1), math.sin(-0.1), 0.0, math.cos(0.2), 0.0, math.sin(0.2)]
    assert got.ravel().tolist() == pytest.approx(want, abs=1e-6)


@pytest.mark.parametrize("turn", [None, stray_fake.CLOCKWISE], ids=["landscape", "portrait"])
def test_a_stray_scanner_export_becomes_a_place_on_arkits_poses_and_lidar(tmp_path, turn):
    upright = [stray_fake.upright_pose(0.5 + 0.1 * k, heading=0.02 * k) for k in range(30)]
    sensor = upright if turn is None else [stray.turned(p, turn.T) for p in upright]
    folder = stray_fake.write(tmp_path / "71de12f9", sensor)
    did = W.run(tmp_path, folder, poses_out=tmp_path / "poses.npz")
    rev = places.revisions(tmp_path, did["place"])[0]
    assert rev.producer == W.PRODUCER_STRAY and rev.signals["walkthrough"] == ["71de12f9"]
    cloud = ply.vertices(places.read_map(tmp_path, rev) or b"")
    assert cloud is not None
    off = stray_fake.distance_to_room(np.c_[cloud["x"], cloud["y"], cloud["z"]])
    assert np.percentile(off, 90) < 0.01 and off.max() < 0.03, "the LiDAR points sit on the room's surfaces"
    assert rev.floor is not None
    with np.load(tmp_path / "poses.npz") as saved:
        frames = saved["frame"]
        assert len(frames) == did["frames"] >= W.MIN_FRAMES
        want = np.stack([upright[min(f // 5, 29)] for f in frames])[:, :3, :3]
        assert saved["c2w"][:, :3, :3] == pytest.approx(want, abs=1e-6), "upright cameras, whichever way it was held"


def test_a_portrait_stray_export_saves_upright_frames(tmp_path):
    sensor = [stray.turned(stray_fake.upright_pose(0.5 + 0.1 * k), stray_fake.CLOCKWISE.T) for k in range(30)]
    folder = stray_fake.write(tmp_path / "x", sensor)
    W.run(tmp_path, folder, keyframes_only=tmp_path / "kf")
    import cv2

    first = cv2.imread(str(tmp_path / "kf" / "f000.jpg"))
    assert first is not None
    assert first.shape[:2] == (stray_fake.RGB[0], stray_fake.RGB[1]), "portrait: as tall as the sensor is wide"


def test_a_stray_walk_gets_a_splat_in_its_maps_frame_and_cameras_to_look_from(tmp_path, monkeypatch):
    upright = [stray_fake.upright_pose(0.5 + 0.1 * k, heading=0.02 * k) for k in range(30)]
    sensor = [stray.turned(p, stray_fake.CLOCKWISE.T) for p in upright]  # held upright: portrait
    folder = stray_fake.write(tmp_path / "walk", sensor)
    runtime = tmp_path / "gsplat"
    (runtime / ".venv" / "bin").mkdir(parents=True)
    (runtime / ".venv" / "bin" / "python").write_text("")
    jobs = []

    def trained(rt, job, gpu, timeout_s=0.0):
        jobs.append(json.loads(job.read_text()))
        n = 2
        names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", *(f"f_rest_{i}" for i in range(9))]
        names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
        head = f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
        head += "".join(f"property float {k}\n" for k in names) + "end_header\n"
        rows = np.zeros((n, len(names)), "<f4")
        rows[:, names.index("rot_0")] = 1.0
        (job.parent / "splat.ply").write_bytes(head.encode() + rows.tobytes())
        return {"gaussians": n, "train_psnr": 20.0, "seconds": 1.0}

    monkeypatch.setattr(splats, "run_runner", trained)  # the subprocess boundary: gsplat's own runtime
    with cfg.override({"spatial": {"splat_runtime": str(runtime), "splats_enabled": True}}):
        did = W.run(tmp_path, folder, poses_out=tmp_path / "poses.npz")
    rev = places.latest(tmp_path, did["place"])
    assert rev is not None and did["splat"] is not None and did["splat"]["revision"] == rev.id
    assert rev.producer == fuse.PRODUCER, "the splat is of the fused revision, the one the 3D view shows"
    assert places.has_splat(tmp_path, rev)
    looks = [c for c in rev.cameras if c.startswith("walk-")]
    assert len(looks) == min(splats.LOOK_FROM, did["frames"])
    (job,) = jobs
    assert len(job["views"]) == did["frames"] and job["reg"] == 0.0
    view = job["views"][0]
    assert (view["width"], view["height"]) == (stray_fake.RGB[1], stray_fake.RGB[0]), "the upright frame's size"
    assert (view["cx"], view["cy"]) == pytest.approx((stray_fake.RGB[1] - 1 - stray_fake.K[1, 2], stray_fake.K[0, 2]))
    # The first view is the first look-from camera's frame: carried into the fused map's frame (levelled, so turned
    # from ARKit's y-up), it sits where that camera does.
    with np.load(tmp_path / "poses.npz") as saved:
        walked = saved["c2w"][0].ravel().tolist()
    assert view["pose"] != pytest.approx(walked, abs=1e-3), "tier 1 moved the walk"
    assert view["pose"] == pytest.approx(list(rev.cameras["walk-0m00s"].pose), abs=1e-6)


def fixed_camera(x, y, height, heading_deg, pitch_deg=30.0):
    """A camera in a levelled place (z up) at (x, y, height), facing `heading_deg`, pitched down: x right, y down."""
    h, p = math.radians(heading_deg), math.radians(pitch_deg)
    forward = np.array([math.cos(p) * math.cos(h), math.cos(p) * math.sin(h), -math.sin(p)])
    right = np.array([math.sin(h), -math.cos(h), 0.0])
    m = np.eye(4)
    m[:3, :3] = np.c_[right, np.cross(forward, right), forward]
    m[:3, 3] = (x, y, height)
    return m


def turned(yaw_deg, shift):
    a = math.radians(yaw_deg)
    m = np.eye(4)
    m[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    m[:3, 3] = shift
    return m


PLACE_TO_WALK = turned(30.0, [5.0, 2.0, 0.0])  # where the fixed place stands in the walk's levelled frame


def pair_answer(cam, frame, scale=0.5, rise=0.0):
    """MASt3R's pair of a fixed camera (pose `cam` in its place) with a walk frame (pose `frame` in the walk), in a
    pair scale `scale` times the walk's, the walk frame DEPTH deep; `rise` lifts the camera by that much (a bad pair)."""
    in_walk = PLACE_TO_WALK @ cam
    in_walk[2, 3] += rise
    fwd = np.linalg.inv(in_walk) @ frame
    fwd[:3, 3] *= scale
    return {
        "a": "cam",
        "b": "f000",
        "matches": 900,
        "inliers": [700, 650],
        "depths": [2.0, DEPTH * scale],
        "forward": fwd.ravel().tolist(),
        "backward": np.linalg.inv(fwd).ravel().tolist(),
    }


def pose_of(m):
    return places.CameraPose(tuple(float(x) for x in m.ravel()), 300.0, 640, 360, T)


WALK_FRAME = fixed_camera(6.0, 4.0, 1.4, 200.0, pitch_deg=5.0)  # the phone, nearly level, 1.4 m up


def test_a_sure_pair_seats_the_fixed_place_in_the_walk():
    cam = fixed_camera(1.0, 0.5, 2.2, 45.0)
    got = W.seat(pair_answer(cam, WALK_FRAME), WALK_FRAME, DEPTH, pose_of(cam))
    assert got == pytest.approx(PLACE_TO_WALK, abs=1e-6)


@pytest.mark.parametrize(
    ("change", "rise"),
    [
        pytest.param({}, 1.0, id="camera-a-metre-off-its-height"),
        pytest.param({"matches": 100}, 0.0, id="too-few-matches"),
        pytest.param({"inliers": [700, 100]}, 0.0, id="pnp-disagrees"),
        pytest.param({"forward": None}, 0.0, id="no-pose"),
    ],
)
def test_a_pair_mastr3r_isnt_sure_of_seats_nothing(change, rise):
    cam = fixed_camera(1.0, 0.5, 2.2, 45.0)
    answer = {**pair_answer(cam, WALK_FRAME, rise=rise), **change}
    assert W.seat(answer, WALK_FRAME, DEPTH, pose_of(cam)) is None


def test_seats_from_two_frames_that_agree_place_it_and_an_outlier_is_left_out():
    near = PLACE_TO_WALK @ turned(4.0, [0.3, 0.0, 0.0])
    wrong = turned(120.0, [-3.0, 8.0, 0.0])
    got = W.agree([(3, PLACE_TO_WALK), (6, near), (9, wrong)])
    assert got is not None
    yaw = math.degrees(math.atan2(got[1, 0], got[0, 0]))
    assert yaw == pytest.approx(32.0, abs=0.1), "the mean of the two that agree"
    assert got[:2, 3] == pytest.approx(PLACE_TO_WALK[:2, 3], abs=0.4)


def test_a_place_seated_from_one_walk_frame_isnt_placed():
    assert W.agree([(3, PLACE_TO_WALK), (3, PLACE_TO_WALK)]) is None
    assert W.agree([]) is None


def test_the_place_the_owner_taught_most_about_is_the_one_others_merge_into(tmp_path):
    def made(now, cams):
        return places.revise(tmp_path, None, producer="p", cameras=cams, plan="plan", now=now).place

    cam = pose_of(np.eye(4))
    walk, two_cams, corrected = made(T, {}), made(T + 1, {"a": cam, "b": cam}), made(T + 2, {"c": cam})
    assert W.join_target(tmp_path, [walk, two_cams]) == two_cams, "more cameras"
    places.name(tmp_path, walk, "Home", now=T + 3)
    assert W.join_target(tmp_path, [walk, two_cams]) == walk, "a name the owner gave"
    places.correct_plan(tmp_path, corrected, "plan", now=T + 4)
    assert W.join_target(tmp_path, [walk, two_cams, corrected]) == corrected, "a plan the owner drew"


def test_the_walk_is_written_and_joined_under_the_places_lock(tmp_path, runtime, monkeypatch):
    import fcntl

    from liberational import locking

    held = []

    def taken(state):
        with locking.lock_path(places.root(state)).open("a") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return False

    real_finish, real_join = W.finish, W.join_new

    def finish(state, *a, **kw):
        held.append(("finish", taken(state)))
        return real_finish(state, *a, **kw)

    def join_new(state, *a, **kw):
        held.append(("join", taken(state)))
        return real_join(state, *a, **kw)

    monkeypatch.setattr(W, "finish", finish)
    monkeypatch.setattr(W, "join_new", join_new)
    W.run(tmp_path, pan_video(tmp_path / "walk.mp4"))
    assert held == [("finish", True), ("join", True)]


def test_a_lidar_walk_into_a_place_with_an_owner_plan_is_placed_by_the_plan(tmp_path, monkeypatch):
    # #3286: the place's map has nothing like the walk's room, and no camera sees it, but the owner's plan has the room
    monkeypatch.setattr(W.fuse.cameras, "layout", lambda *a, **k: W.fuse.cameras.Layout(None, None, fuse.LEVEL_FLOOR))
    old = unrelated_place(tmp_path, {"s2": POSE})
    assert old.plan is None
    plan = 'plan "Home" walls 0.02/0.02\nroom r1 "Box" other poly 20,5 24,5 24,8 20,8\n'
    places.correct_plan(tmp_path, old.place, plan, now=T + 2, base=old.id)
    folder = stray_fake.write(tmp_path / "71de12f9", [stray_fake.upright_pose(0.5 + 0.1 * k) for k in range(30)])
    did = W.run(tmp_path, folder, place=old.place)
    assert did["place"] == old.place and did["registered"] == "plan"
    rev = places.latest(tmp_path, old.place)
    assert rev is not None and rev.frame == old.frame and rev.cameras["s2"] == POSE
    cloud, _, counts = ply.read(places.root(tmp_path) / rev.place / str(rev.map))
    walk = cloud[cloud[:, 0] > -10.0]  # the yard is at x -45 to -15
    assert len(walk) > 1000
    # it saw the far half of the room (the camera walked along it, looking at one long wall): all of it in the box
    assert (walk[:, :2] > [19.9, 4.9]).all() and (walk[:, :2] < [24.1, 8.1]).all(), "where the plan has the room"
    assert np.ptp(walk[:, 0]) > 3.5, "it spans the room's long side"
    assert int((counts["walked"] > 0).sum()) == len(walk), "its voxels, and only its, count as walked"
