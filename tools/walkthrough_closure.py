#!/usr/bin/env python3
"""How well a long plain-video walkthrough closes: simulated walks of several minutes through the real loop closure
(spatial/walkthrough.py `close_loops`, spatial/retrieval.py `shortlist`, and spatial/posegraph.py), measured against
the walk that was simulated.

    .venv/bin/python tools/walkthrough_closure.py                      # 2, 4, 6, and 10 minutes, 5 walks each
    .venv/bin/python tools/walkthrough_closure.py --minutes 6 --seeds 10 --one-lap --breaks 0.1

No model and no GPU: about a minute on the CPU for the default table. Each walk goes through a house-sized set of
rooms at a frame a second and 0.75 m a frame, and comes back to where it began, walking on a few frames past it:
several laps of a 14 x 9 m loop with a room detour, or with `--one-lap` one loop as long as the whole walk, which
sees its start again only at the end. The pose model's drift is simulated at what one real two-minute walk showed
(its revisits were 7 to 12 m and 9 to 52 degrees off after 30 to 70 frames): each step turns `BIAS_DEG` more than
it should, one way for the whole walk, with `STEP_NOISE_DEG` and `STEP_NOISE_M` of noise, and with `--breaks` that
share of the steps also spin 40 to 120 degrees. The pair model's answers are the true relative poses with
`PAIR_NOISE_DEG` and `PAIR_NOISE_M` of noise, in half the pose model's scale. Revisits are proposed from how close
and how alike in heading two frames truly are, with noise, through the real shortlist.

Each row is one walk length: the gap between the walk's start and its return before and after (the worst pair of a
frame in the first `RETURN` and one in the last `RETURN` that stand in the same spot: metres, and degrees of heading
about up), the share of walks inside the stated tolerance (`walkthrough.CLOSE_M`, `walkthrough.CLOSE_DEG`), and the
root mean square position error of every frame. The first frame keeps its own tilt (it is the fixed one), so tilt
isn't part of the heading here. The numbers are a model's, not a camera's: they say the graph closes a walk whose
revisits the models measure, not that the models will.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import math
import statistics
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from liberational.spatial import posegraph, retrieval, walkthrough

STEP_M = 0.75  # a frame's walk
LOOP_M = (14.0, 9.0)  # the lap's width and depth
TAIL = 4  # frames walked on past the start
BIAS_DEG = 0.5
STEP_NOISE_DEG, STEP_NOISE_M = 1.5, 0.05
PAIR_NOISE_DEG, PAIR_NOISE_M = 2.0, 0.1
PAIR_SCALE = 0.5
DEPTH = 3.0  # every frame's median depth as the pose model sees it
RETURN = 10  # frames at each end that count as the start and the return
SAME_M, SAME_DEG = 1.0, 30.0  # two frames stand in the same spot


def upright(heading: float, x: float, z: float) -> np.ndarray:
    """A level camera (OpenCV axes) in a y-down world, at (x, 0, z), looking along `heading` (radians from +z)."""
    c, s = math.cos(heading), math.sin(heading)
    pose = np.eye(4)
    pose[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    pose[:3, 3] = [x, 0.0, z]
    return pose


def route(frames: int, one_lap: bool = False) -> np.ndarray:
    """The true walk, about `frames` poses: laps of the loop with its room detour (or one loop the whole walk long),
    back to the start and `TAIL` frames on, facing the way it goes."""
    w, d = LOOP_M
    if one_lap:
        grow = (frames - TAIL) * STEP_M / (2 * (w + d))
        w, d = w * grow, d * grow
        lap = [(0.0, 0.0), (0.0, d), (w, d), (w, 0.0)]
        laps = 1
    else:
        lap = [(0.0, 0.0), (0.0, d), (0.4 * w, d), (0.4 * w, 0.45 * d), (0.7 * w, 0.45 * d), (0.7 * w, d), (w, d)]
        lap.append((w, 0.0))
        length = sum(math.dist(a, b) for a, b in itertools.pairwise([*lap, lap[0]]))
        laps = max(1, round((frames - TAIL) * STEP_M / length))
    corners = [*(lap * laps), (0.0, 0.0), (0.0, TAIL * STEP_M)]
    poses: list[np.ndarray] = []
    carry = 0.0
    for (ax, az), (bx, bz) in itertools.pairwise(corners):
        length = math.dist((ax, az), (bx, bz))
        heading = math.atan2(bx - ax, bz - az)
        at = carry
        while at < length:
            u = at / length
            poses.append(upright(heading, ax + u * (bx - ax), az + u * (bz - az)))
            at += STEP_M
        carry = at - length
    return np.array(poses)


def _noisy(rel: np.ndarray, rng: np.random.Generator, deg: float, m: float, bias_deg: float = 0.0) -> np.ndarray:
    """`rel` with noise: mostly about up and along the floor, a third of it in tilt and a fifth in height."""
    out = rel.copy()
    turn = np.radians(rng.normal(0.0, [deg / 3, deg, deg / 3]) + [0.0, bias_deg, 0.0])
    out[:3, :3] = rel[:3, :3] @ Rotation.from_rotvec(turn).as_matrix()
    out[:3, 3] = rel[:3, 3] + rng.normal(0.0, m, 3) * [1.0, 0.2, 1.0]
    return out


def drifted(true: np.ndarray, rng: np.random.Generator, breaks: float = 0.0) -> np.ndarray:
    """The walk as the pose model gives it: every step's own motion with its bias and noise, and `breaks` of the
    steps spun as well."""
    bias = BIAS_DEG * float(rng.choice([-1.0, 1.0]))
    spun = set(rng.choice(len(true) - 1, round(breaks * (len(true) - 1)), replace=False).tolist())
    out = [true[0]]
    for k in range(len(true) - 1):
        step = _noisy(posegraph.relative(true[k], true[k + 1]), rng, STEP_NOISE_DEG, STEP_NOISE_M, bias)
        if k in spun:
            step = step @ upright(math.radians(float(rng.choice([-1.0, 1.0]) * rng.uniform(40.0, 120.0))), 0.0, 0.0)
        out.append(out[-1] @ step)
    return np.array(out)


def apart(true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every pair of frames' distance along the floor (metres) and difference in heading (degrees), n x n each."""
    floor = true[:, :3, 3][:, [0, 2]]
    forward = true[:, :3, 2]
    return (
        np.linalg.norm(floor[:, None] - floor[None], axis=2),
        np.degrees(np.arccos(np.clip(forward @ forward.T, -1.0, 1.0))),
    )


def answers(true: np.ndarray, pairs: Sequence[tuple[int, int]], rng: np.random.Generator) -> list[dict[str, Any]]:
    """The pair model's answers for `pairs`, in the shape spatial/mast3r_runner.py's pairs mode gives."""
    out = []
    for i, j in pairs:
        forward = _noisy(posegraph.relative(true[i], true[j]), rng, PAIR_NOISE_DEG, PAIR_NOISE_M)
        backward = np.linalg.inv(forward)
        forward[:3, 3] *= PAIR_SCALE
        backward[:3, 3] *= PAIR_SCALE
        out.append(
            {
                "a": i,
                "b": j,
                "matches": 600,
                "depths": [DEPTH * PAIR_SCALE, DEPTH * PAIR_SCALE],
                "inliers": [540, 520],
                "forward": forward.ravel().tolist(),
                "backward": backward.ravel().tolist(),
            }
        )
    return out


def gap(poses: np.ndarray, true: np.ndarray, i: int, j: int) -> tuple[float, float]:
    """How far `poses` puts frame j from where `true` has it, seen from frame i: metres, and degrees of heading
    about the camera's up."""
    off = np.linalg.inv(posegraph.relative(true[i], true[j])) @ posegraph.relative(poses[i], poses[j])
    return float(np.linalg.norm(off[:3, 3])), abs(math.degrees(math.atan2(off[0, 2], off[2, 2])))


def simulate(frames: int, seed: int, one_lap: bool = False, breaks: float = 0.0) -> dict[str, Any]:
    """One simulated walk through `close_loops`: what it kept, and its closure before and after against the truth."""
    rng = np.random.default_rng(seed)
    true = route(frames, one_lap)
    n = len(true)
    walk = drifted(true, rng, breaks)
    dist, angle = apart(true)
    scores = 1.0 - dist / 3.0 - angle / 90.0 + rng.normal(0.0, 0.05, dist.shape)
    loops = retrieval.shortlist(scores, gap=retrieval.LOOP_GAP, cap=retrieval.cap_for(n))
    real = [(i, j) for i, j in loops if dist[i, j] < 2.5 and angle[i, j] < 50.0]  # the rest get no sure answer
    measured = answers(true, [*((k, k + 1) for k in range(n - 1)), *real], rng)
    closed = walkthrough.close_loops(walk, measured, loops, np.full(n, DEPTH), gravity=True)
    back = [
        (i, j) for i in range(RETURN) for j in range(n - RETURN, n) if dist[i, j] < SAME_M and angle[i, j] < SAME_DEG
    ]
    before = [gap(walk, true, i, j) for i, j in back]
    after = [gap(closed.poses, true, i, j) for i, j in back]
    error = np.linalg.norm(closed.poses[:, :3, 3] - true[:, :3, 3], axis=1)
    return {
        "frames": n,
        "metres": float(np.linalg.norm(np.diff(true[:, :3, 3], axis=0), axis=1).sum()),
        "revisits": len(real),
        "kept": len(closed.loops),
        "supported": closed.supported,
        "before_m": max(m for m, _ in before),
        "before_deg": max(d for _, d in before),
        "after_m": max(m for m, _ in after),
        "after_deg": max(d for _, d in after),
        "rms_m": float(np.sqrt((error**2).mean())),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Simulated long walkthroughs through the loop closure.")
    parser.add_argument("--minutes", type=float, nargs="+", default=[2.0, 4.0, 6.0, 10.0])
    parser.add_argument("--seeds", type=int, default=5, help="walks per length")
    parser.add_argument("--one-lap", action="store_true", help="one loop the whole walk long")
    parser.add_argument("--breaks", type=float, default=0.0, help="the share of steps where the pose model spins")
    args = parser.parse_args(argv)
    logging.disable(logging.WARNING)
    print(f"tolerance: {walkthrough.CLOSE_M} m and {walkthrough.CLOSE_DEG:.0f} degrees")
    print("minutes frames metres revisits kept | gap before (median) | gap after (median, worst) | inside | rms m")
    for minutes in args.minutes:
        runs = [simulate(round(minutes * 60), seed, args.one_lap, args.breaks) for seed in range(args.seeds)]
        inside = sum(r["after_m"] <= walkthrough.CLOSE_M and r["after_deg"] <= walkthrough.CLOSE_DEG for r in runs)

        def mid(key: str, runs: list[dict[str, Any]] = runs) -> float:
            return statistics.median(r[key] for r in runs)

        print(
            f"{minutes:7.0f} {runs[0]['frames']:6d} {runs[0]['metres']:6.0f} {mid('revisits'):8.0f} {mid('kept'):4.0f} | "
            f"{mid('before_m'):5.1f} m {mid('before_deg'):5.0f} deg | "
            f"{mid('after_m'):.2f} m {mid('after_deg'):.1f} deg, "
            f"{max(r['after_m'] for r in runs):.2f} m {max(r['after_deg'] for r in runs):.1f} deg | "
            f"{inside}/{len(runs)} | {mid('rms_m'):.2f}"
        )


if __name__ == "__main__":
    main()
