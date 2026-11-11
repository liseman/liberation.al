"""A 3D point for every pixel of one picture, from MASt3R-SLAM's single-frame step, run in its own runtime.
spatial/camera_clouds.py runs it; nothing that runs in the pack's environment imports it.

    <runtime>/venv/bin/python pointmap_runner.py job.json

MASt3R-SLAM (rmurai0610/MASt3R-SLAM, CC BY-NC-SA 4.0, on MASt3R) lives in a folder of its own because it needs its
own torch, a compiled solver, and upstream code the pack's processes stay clear of. The pack downloads none of it:
`spatial.pointmap_runtime` names a folder someone set up, with the checkout in `src/` and its environment in `venv/`.
So this file imports nothing from the pack and only the standard library at module level; numpy, torch, and
MASt3R-SLAM are imported inside `main`.

The job (JSON): `repo` (the MASt3R-SLAM checkout), `weights` (a directory holding MASt3R's `model.safetensors` and
`config.json`, loaded with safetensors, never a pickle), `out` (a file for the report), and `views`
([{"image": path, "mask": path or null, "out": path}]). Each view is one picture, alone: this is what MASt3R-SLAM
does with the first frame of a stream (the frame paired with itself), which gives every pixel a point in the
camera's own frame (x right, y down, z forward, in MASt3R's metres) and a confidence.

Each view's `out` is an .npz the size the model saw the picture at (its long side 512, cropped to a multiple of 16):
`points` (H x W x 3 float32), `confidence` (H x W float32), `colour` (H x W x 3 uint8, the picture as the model saw
it), and `covered` (H x W bool: the view's `mask`, white where a camera burns in its clock or name, resized the
same way; all False without one). The report is {"load_seconds", "views": [{"seconds", "peak_gpu_gb"}]}.

Measured on six cameras' pictures beside the live services: 7.5 s to load, then 0.2 s and 2.9 GB of GPU memory a
picture.
"""

from __future__ import annotations

import json
import os
import sys
import time

SIZE = 512  # the long side MASt3R takes


def main() -> None:
    with open(sys.argv[1]) as f:
        job = json.load(f)
    repo = os.path.abspath(job["repo"])
    weights = os.path.abspath(job["weights"])
    views = [{k: (os.path.abspath(v) if isinstance(v, str) else v) for k, v in view.items()} for view in job["views"]]
    report_path = os.path.abspath(job["out"])
    os.chdir(repo)  # MASt3R-SLAM reads its config relative to its checkout
    sys.path.insert(0, repo)

    import lietorch  # ty: ignore[import-not-found] (the runtime's own)
    import numpy as np
    import torch
    from mast3r_slam.config import load_config  # ty: ignore[import-not-found]
    from mast3r_slam.frame import create_frame  # ty: ignore[import-not-found]
    from mast3r_slam.mast3r_utils import load_mast3r, mast3r_inference_mono, resize_img  # ty: ignore[import-not-found]
    from PIL import Image

    load_config("config/base.yaml")
    device = "cuda:0"
    started = time.perf_counter()
    model = load_mast3r(path=weights, device=device)
    report = {"load_seconds": round(time.perf_counter() - started, 2), "views": []}
    for view in views:
        rgb = np.asarray(Image.open(view["image"]).convert("RGB"), dtype=np.float32) / 255.0
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        frame = create_frame(0, rgb, lietorch.Sim3.Identity(1, device=device), img_size=SIZE, device=device)
        points, confidence = mast3r_inference_mono(model, frame)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t
        h, w = (int(v) for v in frame.img_true_shape[0])
        covered = np.zeros((h, w), dtype=bool)
        if view.get("mask"):
            mask = np.asarray(Image.open(view["mask"]).convert("RGB"), dtype=np.float32) / 255.0
            covered = np.asarray(resize_img(mask, SIZE)["unnormalized_img"])[:h, :w, 0] > 127
        np.savez(
            view["out"],
            points=points.reshape(h, w, 3).float().cpu().numpy(),
            confidence=confidence.reshape(h, w).float().cpu().numpy(),
            colour=(frame.uimg.cpu().numpy()[:h, :w, :3] * 255).astype(np.uint8),
            covered=covered,
        )
        report["views"].append(
            {"seconds": round(seconds, 3), "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        )
    with open(report_path, "w") as f:
        json.dump(report, f)


if __name__ == "__main__":
    main()
