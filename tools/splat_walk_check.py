"""Score a walkthrough's splat on frames it didn't train on (#21): how well a plain-video walk's splat holds up
against a Stray Scanner walk's.

    PYTHONPATH=src .venv/bin/python tools/splat_walk_check.py WALK --work DIR [--refine on|off|both] [--steps N]
        [--jitter CM,DEG] [--exact]

WALK is a video or a Stray Scanner export folder, as `pack-walkthrough` takes it. A Stray export's `rgb.mp4` given
as WALK is a plain video of the same walk, so the two poses (ARKit's and CUT3R's) compare on the same frames.

1. **Poses.** `pack-walkthrough`'s own steps (spatial/walkthrough.py `run`) into a scratch state under `--work`,
   never the pack's, with splats off; the corrected poses are kept in `--work` and reused on the next run.
2. **Frames.** The walk's saved frames again (`save_frames`), with their intrinsics as `run` gives them. The views
   they make are kept in `--work` (`views.json`), so a later run needs only `--work`: WALK can be gone by then.
3. **Splat.** spatial/splats.py `train_walk`, as `splat_walk` trains it (in the revision's map frame, from its map
   points), with every `EVERY`-th frame held out, once per `--refine` setting. Each prints the runner's result as one
   JSON line: `test_psnr` (each held-out view's pose fitted to the frozen splat first) and `test_psnr_raw` (at the
   pose as given), the time, and the GPU memory.

`--jitter CM,DEG` moves every view's pose by that much first, each in its own random direction and about its own
random axis (`jittered`): a known error for the trainer's pose refinement to find (#197). Trained with refinement,
the correction it reports (`pose_shift`, `pose_turn_deg`) should be near the jitter, and near what the walk's own
poses need without it.

`--exact` swaps the walk for a copy whose frames sit exactly at their poses: the walk is trained once on every view,
and each view's image and depth as that splat draws them at its given pose (the trainer's `render`, kept under
`--work`/exact) stand in for the recording's. The walk's own frames are off their poses by an unknown amount; this
copy's are off by nothing, so trained with refinement it should report a correction near zero, and with `--jitter`
near the jitter. Spatial/splat_runner.py has what one walk measured.

It runs CUT3R or MASt3R and gsplat on the GPU, niced, so it's one run at a time, on a quiet GPU when its timings are
to be compared. Nothing it writes leaves `--work`.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from liberational import config as cfg
from liberational.memory import places
from liberational.spatial import ply, splats, stray, walkthrough

EVERY = 8  # every 8th frame (from the 4th) is held out
FIRST = 4


def poses(walk: Path, work: Path) -> Path:
    """The walk's corrected poses (`pack-walkthrough --poses`), run into `work`'s scratch state once and kept."""
    saved = work / "poses.npz"
    if not saved.exists():
        with cfg.override({"spatial": {"splats_enabled": False}}):
            did = walkthrough.run(work / "state", walk, poses_out=saved)
        (work / "walk.json").write_text(json.dumps(did, default=str))
    return saved


def frames(
    walk: Path, saved: Path, folder: Path
) -> tuple[list[Path], np.ndarray, list[tuple[np.ndarray, tuple[int, int]]], list[float], list[Path | None]]:
    """The walk's saved frames in `folder`, their poses, intrinsics and sizes, and times, as `run` (or `run_stray`)
    had them, and a Stray export's depth maps (none for a video)."""
    import cv2

    folder.mkdir(parents=True, exist_ok=True)
    with np.load(saved) as z:
        got = {k: z[k] for k in z.files}
    picked = [int(i) for i in got["frame"]]
    if stray.is_export(walk):
        video = walk / "rgb.mp4"
        recorded = stray.read_odometry(walk)
        # `frame` is each saved frame's index in the decoded video, `row` its number in the recording.
        rows = {i: recorded[int(n)] for i, n in zip(picked, got["row"], strict=True)}
        turn, code = stray.upright(np.stack([rows[i].c2w for i in picked]))
        images = walkthrough.save_frames(video, picked, folder, turn=code)
        cap = cv2.VideoCapture(str(video))
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
        paths = [images[n] for n in sorted(images)]
        intr = [stray.saved_intrinsics(rows[i].k, size, walkthrough.FRAME_SIDE, code, turn) for i in picked]
        depths = walkthrough.stray_depths(walk, [rows[i].frame for i in picked], code, folder)
    else:
        images = walkthrough.save_frames(walk, picked, folder)
        paths = [images[n] for n in sorted(images)]
        intr = walkthrough.cut3r_intrinsics(got["focal"], got["size"], paths)
        depths = []
    return paths, got["c2w"], intr, [float(t) for t in got["t"]], depths


def walk_views(walk: Path, work: Path) -> tuple[list[dict[str, Any]], float]:
    """The trainer's views of the walk in its scratch revision's map frame (`splats.walk_views`), and that frame's
    units to a metre; kept in `work`/views.json and read back from there when it has them."""
    kept = work / "views.json"
    if kept.exists():
        got = json.loads(kept.read_text())
        return got["views"], float(got["scale"])
    saved = poses(walk, work)
    paths, c2w, intr, times, depths = frames(walk, saved, work / "frames")
    state = work / "state"
    (place,) = places.place_ids(state)
    rev = places.latest(state, place)
    looks = splats.look_from(c2w, intr, times)
    moved = splats.carried(looks, rev.cameras) if rev is not None else None
    if moved is None:
        raise SystemExit("the scratch place has none of the walk's cameras")
    views = splats.walk_views(paths, c2w, intr, moved, depths)
    kept.write_text(json.dumps({"views": views, "scale": moved[2]}))
    return views, moved[2]


def exact(
    views: list[dict[str, Any]], points: np.ndarray, colors: np.ndarray | None, scale: float, steps: int, folder: Path
) -> list[dict[str, Any]]:
    """`views` with each one's image, and depth when it has one, swapped for what a splat trained on all of them
    draws at its given pose: a walk whose frames sit exactly at their poses. Trained and drawn once into `folder`."""
    if not (folder / f"f{len(views) - 1:03d}.png").exists():
        splats.train_walk(views, points, colors, scale=scale, steps=steps, render=folder)
    return [
        {
            **v,
            "image": str(folder / f"f{i:03d}.png"),
            **({"depth": str(folder / f"d{i:03d}.png")} if "depth" in v else {}),
        }
        for i, v in enumerate(views)
    ]


def jittered(views: list[dict[str, Any]], shift_m: float, turn_deg: float, seed: int = 0) -> list[dict[str, Any]]:
    """`views` with every pose moved in its camera's own frame by `shift_m` in a random direction and turned by
    `turn_deg` about a random axis, a different one for each view."""
    rng = np.random.default_rng(seed)
    out = []
    for view in views:
        direction, axis = (v / np.linalg.norm(v) for v in rng.normal(size=(2, 3)))
        delta = np.eye(4)
        delta[:3, :3] = Rotation.from_rotvec(axis * np.radians(turn_deg)).as_matrix()
        delta[:3, 3] = direction * shift_m
        pose = np.asarray(view["pose"], dtype=np.float64).reshape(4, 4) @ delta
        out.append({**view, "pose": pose.ravel().tolist()})
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("walk", type=Path, help="a video, or a Stray Scanner export folder")
    parser.add_argument("--work", type=Path, required=True, help="a scratch folder for the state, frames, and poses")
    parser.add_argument("--refine", choices=("on", "off", "both"), default="both")
    parser.add_argument("--steps", type=int, default=splats.WALK_STEPS)
    parser.add_argument("--jitter", help="move every pose by CM,DEG first, e.g. 2,0.3")
    parser.add_argument("--exact", action="store_true", help="train a copy of the walk drawn at exactly its poses")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args.work.mkdir(parents=True, exist_ok=True)
    views, scale = walk_views(args.walk, args.work)
    state = args.work / "state"
    (place,) = places.place_ids(state)
    rev = places.latest(state, place)
    if rev is None or rev.map is None:
        raise SystemExit("the scratch place has no map")
    points, colors, _ = ply.read(places.root(state) / rev.place / str(rev.map))
    if args.exact:
        views = exact(views, points, colors, scale, args.steps, args.work / "exact")
    if args.jitter:
        cm, deg = (float(x) for x in args.jitter.split(","))
        views = jittered(views, cm / 100.0, deg)
    holdout = list(range(FIRST, len(views), EVERY))
    for refine in {"on": [True], "off": [False], "both": [False, True]}[args.refine]:
        result, _ = splats.train_walk(
            views, points, colors, scale=scale, refine=refine, holdout=holdout, steps=args.steps
        )
        said = {"walk": args.walk.name, "views": len(views), "held_out": len(holdout), "jitter": args.jitter}
        said["exact"] = args.exact
        print(json.dumps({**said, **result}), flush=True)


if __name__ == "__main__":
    main()
