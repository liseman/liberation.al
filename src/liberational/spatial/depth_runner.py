"""Metric depth of single images from Depth Anything 3, run in gsplat's runtime (#20). spatial/splats.py runs it for a
fixed-camera splat's depth prior; nothing that runs in the pack's environment imports it.

    <runtime>/.venv/bin/python depth_runner.py job.json

Depth Anything 3 (ByteDance-Seed/Depth-Anything-3, code Apache-2.0, pinned by `mise run gsplat` under the runtime's
`depth-anything-3/`) has a monocular metric model, DA3METRIC-LARGE (Apache-2.0, a DINOv2 ViT-L with a DPT head,
0.35B parameters; [models.hf.depth_anything], safetensors only). This file imports nothing from the pack and only
the standard library at module level. It builds the network from the repository's own config and loads the weights
itself, without the package's `api` module: that one imports its exporters (open3d, pycolmap, moviepy, and more),
none of which a depth map needs.

The job (JSON): `source` (the repository's `src` folder), `checkpoint` (the model's `model.safetensors`), and
`views` ([{"image": path, "focal": px at the image's size, "out": path}]). Each view's depth is written to its
`out` as a float32 .npy the image's size, in metres along the view axis: the network's output times the focal at
the size it ran at, over 300, as the model card says. Each image runs alone, its long side scaled to `LONG_SIDE` and
both sides to a multiple of the patch size.

Measured on 48 frames of 512 x 288: 6 s after the load, 2.0 GiB of GPU memory.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

MODEL = "da3metric-large"
LONG_SIDE = 504  # the repository's default processing size
PATCH = 14
FOCAL_OVER = 300.0  # metres = output * focal / 300 (the model card)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet's, as DINOv2 takes its input


def main(argv: Sequence[str]) -> int:
    job = json.loads(Path(argv[0]).read_text())
    sys.path.insert(0, str(job["source"]))
    import numpy as np
    import torch
    import torch.nn.functional as F
    from depth_anything_3.cfg import create_object, load_config  # ty: ignore[import-not-found] (gsplat's runtime)
    from depth_anything_3.registry import MODEL_REGISTRY  # ty: ignore[import-not-found] (gsplat's runtime)
    from PIL import Image
    from safetensors.torch import load_file

    net = create_object(load_config(MODEL_REGISTRY[MODEL]))
    weights = {k.removeprefix("model."): v for k, v in load_file(job["checkpoint"]).items()}
    net.load_state_dict(weights, strict=True)
    net = net.to("cuda").eval()
    mean = torch.tensor(MEAN, device="cuda")[:, None, None]
    std = torch.tensor(STD, device="cuda")[:, None, None]
    for view in job["views"]:
        image = Image.open(view["image"]).convert("RGB")
        w, h = image.size
        scale = LONG_SIDE / max(w, h)
        pw, ph = max(PATCH, round(w * scale / PATCH) * PATCH), max(PATCH, round(h * scale / PATCH) * PATCH)
        pixels = np.asarray(image.resize((pw, ph), Image.Resampling.BICUBIC), np.float32) / 255.0
        x = (torch.from_numpy(pixels).to("cuda").permute(2, 0, 1) - mean) / std
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            # (batch, views, 3, H, W); no poses or intrinsics, no feature export, no Gaussians, no ray pose
            out = net(x[None, None], None, None, [], False, False, "first")
        depth = out["depth"].float().reshape(1, 1, ph, pw) * (float(view["focal"]) * pw / w / FOCAL_OVER)
        full = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=False)[0, 0]
        np.save(view["out"], full.cpu().numpy().astype(np.float32))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
