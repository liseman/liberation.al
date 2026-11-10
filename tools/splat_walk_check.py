"""Score a walkthrough's splat on frames it didn't train on (#21): how well a plain-video walk's splat holds up
against a Stray Scanner walk's.

    PYTHONPATH=src .venv/bin/python tools/splat_walk_check.py WALK --work DIR [--refine on|off|both] [--steps N]

WALK is a video or a Stray Scanner export folder, as `pack-walkthrough` takes it. A Stray export's `rgb.mp4` given
as WALK is a plain video of the same walk, so the two poses (ARKit's and CUT3R's) compare on the same frames.

1. **Poses.** `pack-walkthrough`'s own steps (spatial/walkthrough.py `run`) into a scratch state under `--work`,
   never the pack's, with splats off; the corrected poses are kept in `--work` and reused on the next run.
2. **Frames.** The walk's saved frames again (`save_frames`), with their intrinsics as `run` gives them.
3. **Splat.** spatial/splats.py `train_walk`, as `splat_walk` trains it (in the revision's map frame, from its map
   points), with every `EVERY`-th frame held out, once per `--refine` setting. Each prints the runner's result as one
   JSON line: `test_psnr` (each held-out view's pose fitted to the frozen splat first) and `test_psnr_raw` (at the
   pose as given), the time, and the GPU memory.

It runs CUT3R or MASt3R and gsplat on the GPU, niced, so it's one run at a time, on a quiet GPU when its timings are
to be compared. Nothing it writes leaves `--work`.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

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
        rows = stray.read_odometry(walk)
        turn, code = stray.upright(np.stack([rows[i].c2w for i in picked]))
        images = walkthrough.save_frames(video, picked, folder, turn=code)
        cap = cv2.VideoCapture(str(video))
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
        paths = [images[n] for n in sorted(images)]
        intr = [stray.saved_intrinsics(rows[i].k, size, walkthrough.FRAME_SIDE, code, turn) for i in picked]
        depths = walkthrough.stray_depths(walk, picked, code, folder)
    else:
        images = walkthrough.save_frames(walk, picked, folder)
        paths = [images[n] for n in sorted(images)]
        intr = walkthrough.cut3r_intrinsics(got["focal"], got["size"], paths)
        depths = []
    return paths, got["c2w"], intr, [float(t) for t in got["t"]], depths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("walk", type=Path, help="a video, or a Stray Scanner export folder")
    parser.add_argument("--work", type=Path, required=True, help="a scratch folder for the state, frames, and poses")
    parser.add_argument("--refine", choices=("on", "off", "both"), default="both")
    parser.add_argument("--steps", type=int, default=splats.WALK_STEPS)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args.work.mkdir(parents=True, exist_ok=True)
    saved = poses(args.walk, args.work)
    paths, c2w, intr, times, depths = frames(args.walk, saved, args.work / "frames")
    state = args.work / "state"
    (place,) = places.place_ids(state)
    rev = places.latest(state, place)
    looks = splats.look_from(c2w, intr, times)
    moved = splats.carried(looks, rev.cameras) if rev is not None else None
    if rev is None or rev.map is None or moved is None:
        raise SystemExit("the scratch place has no map or none of the walk's cameras")
    views = splats.walk_views(paths, c2w, intr, moved, depths)
    points, colors, _ = ply.read(places.root(state) / rev.place / str(rev.map))
    holdout = list(range(FIRST, len(views), EVERY))
    for refine in {"on": [True], "off": [False], "both": [False, True]}[args.refine]:
        result, _ = splats.train_walk(
            views, points, colors, scale=moved[2], refine=refine, holdout=holdout, steps=args.steps
        )
        print(json.dumps({"walk": args.walk.name, "views": len(views), "held_out": len(holdout), **result}))


if __name__ == "__main__":
    main()
