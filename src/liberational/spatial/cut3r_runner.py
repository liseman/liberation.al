"""CUT3R over a walkthrough's frames, run in CUT3R's own runtime. spatial/walkthrough.py runs it; nothing in the
pack's environment imports it.

    <runtime>/.venv/bin/python cut3r_runner.py job.json

CUT3R (CUT3R/CUT3R, pinned by `mise run cut3r`; code and weights CC BY-NC-SA 4.0, accepted for now) is a
recurrent model: it reads the frames in order, keeps a state of the scene so far, and gives each frame its camera's
pose and a pointmap in that camera's frame. Over a long walk the poses hold locally and keep the walk level, but drift
in heading; spatial/posegraph.py corrects that afterwards. Like mast3r_runner.py, this file imports nothing from the
pack and only the standard library at module level; numpy, torch, and CUT3R are imported inside `main`.

The job (JSON): `repo` (the pinned CUT3R checkout), `checkpoint` (its `cut3r_512_dpt_4_64.pth`), `images` (frame paths
in walk order), `size` (the long side CUT3R sees, 512), `stride` (keep every stride-th pixel's point, 1 for all), and
`out` (a directory for the answer).

Three choices that differ from CUT3R's demo:
- **Loading.** The checkpoint is a pickle holding the weights and an omegaconf config, and the demo's loader unpickles
  it freely and then `eval`s a model string from it. Here it's loaded with `weights_only=True`, allowing only the
  config's omegaconf classes, and the model is built from `MODEL`, the checkpoint's own settings written out, so the
  file can't run code. Every weight the model has must come from the checkpoint (a strict load), so none is left at
  its random start. It needs CUDA, and fails at once without it.
- **Memory.** The demo encodes every frame at once and keeps every frame's output on the GPU (about 12 GB for 253
  frames, which ran out of memory on a shared 24 GB GPU). This steps one frame at a time and moves each
  frame's output to the CPU, so only the recurrent state stays on the GPU; the result is the same.
- **RoPE.** CUT3R's CUDA RoPE kernel would need nvcc and an older host compiler to build. Its PyTorch fallback can't
  take the pose token's negative position (it looks positions up in a table), so `rope_forward` replaces the
  fallback's forward with the same rotation computed from the positions directly, as the kernel does.

The answer is `out/result.npz`: `c2w` (n x 4 x 4, camera to world, OpenCV camera axes: x right, y down, z forward),
`focal` (n, pixels at CUT3R's input size), `size` (that input's width and height), `points` (N x 3 float32, each in
its own frame's camera frame, confidence over `MIN_CONF`), `colors` (N x 3 uint8), `counts` (n: how many of the points
are each frame's, in order), `grid` (n x rows x columns x 3: every `GRID`-th pixel's camera-frame point), and
`world_grid` (the same pixels' points in the world as CUT3R's own world head puts them, so the up-axis check tests
the pose against something it wasn't computed from); and `out/result.json` with the seconds it took and the GPU memory it peaked at.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

#: Confidence a point needs to be kept (CUT3R's confidences start at 1; its demo shows points over 1.5).
MIN_CONF = 1.5
#: Every GRID-th pixel's point goes into `grid`.
GRID = 8
#: The checkpoint's model settings (its `args.model`), with the patch embedding that takes any aspect ratio, as the
#: demo's loader rewrites them.
MODEL: dict[str, Any] = dict(
    freeze="encoder",
    state_size=768,
    state_pe="2d",
    pos_embed="RoPE100",
    rgb_head=True,
    pose_head=True,
    patch_embed_cls="PatchEmbedDust3R",
    img_size=(512, 512),
    head_type="dpt",
    output_mode="pts3d+pose",
    depth_mode=("exp", -float("inf"), float("inf")),
    conf_mode=("exp", 1, float("inf")),
    pose_mode=("exp", -float("inf"), float("inf")),
    enc_embed_dim=1024,
    enc_depth=24,
    enc_num_heads=16,
    dec_embed_dim=768,
    dec_depth=12,
    dec_num_heads=12,
    landscape_only=False,
)


def rope_forward(self: Any, tokens: Any, positions: Any) -> Any:
    """CroCo's 2D RoPE (the fallback `RoPE2D.forward`'s signature): `tokens` is batch x heads x tokens x dim, and
    `positions` batch x tokens x 2 (y, x). The first half of each token's features turns by its y position and the
    second half by its x, each half as 1D RoPE does (frequencies base^(-2k/d)), with the angles computed from the
    positions directly, so negative positions (CUT3R's pose token is at -1) work as they do in the CUDA kernel."""
    import torch

    half = tokens.size(3) // 2
    inv_freq = 1.0 / (self.base ** (torch.arange(0, half, 2, device=tokens.device).float() / half))

    def turn(t: Any, pos: Any) -> Any:
        angle = pos.to(inv_freq.dtype)[..., None] * inv_freq  # batch x tokens x half/2
        angle = torch.cat((angle, angle), dim=-1)[:, None]  # batch x 1 x tokens x half
        cos, sin = angle.cos().to(t.dtype), angle.sin().to(t.dtype)
        a, b = t[..., : half // 2], t[..., half // 2 :]
        return t * cos + torch.cat((-b, a), dim=-1) * sin

    y, x = tokens.chunk(2, dim=-1)
    return torch.cat((turn(y, positions[:, :, 0]), turn(x, positions[:, :, 1])), dim=-1)


def main(argv: Sequence[str]) -> int:
    job: Mapping[str, Any] = json.loads(Path(argv[0]).read_text())
    sys.path[:0] = [str(Path(job["repo"]) / "src")]  # CUT3R imports its dust3r and croco copies as top-level packages
    import collections
    import typing

    import numpy as np
    import omegaconf  # ty: ignore[import-not-found] (CUT3R's runtime only)
    import torch
    from dust3r.model import ARCroco3DStereo, ARCroco3DStereoConfig  # ty: ignore[import-not-found] (CUT3R's runtime)
    from dust3r.post_process import estimate_focal_knowing_depth  # ty: ignore[import-not-found] (CUT3R's runtime)
    from dust3r.utils.camera import pose_encoding_to_camera  # ty: ignore[import-not-found] (CUT3R's runtime only)
    from dust3r.utils.image import load_images  # ty: ignore[import-not-found] (CUT3R's runtime only)
    from models import pos_embed  # ty: ignore[import-not-found] (CroCo's, on the path once dust3r.model is imported)
    from omegaconf.base import ContainerMetadata, Metadata  # ty: ignore[import-not-found] (CUT3R's runtime only)
    from omegaconf.nodes import AnyNode  # ty: ignore[import-not-found] (CUT3R's runtime only)

    if pos_embed.RoPE2D.__name__ == "RoPE2D":  # the PyTorch fallback, not the compiled kernel
        pos_embed.RoPE2D.forward = rope_forward
    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise SystemExit("CUT3R needs CUDA, and none is available here")
    device = "cuda"
    began = time.monotonic()
    safe: list[Any] = [
        Metadata,
        ContainerMetadata,
        AnyNode,
        typing.Any,
        omegaconf.DictConfig,
        dict,
        collections.defaultdict,
    ]
    with torch.serialization.safe_globals(safe):
        weights = torch.load(job["checkpoint"], map_location="cpu", weights_only=True, mmap=True)["model"]
    model = ARCroco3DStereo(ARCroco3DStereoConfig(**MODEL))
    model.load_state_dict(weights)  # strict: every weight the model has comes from the checkpoint (it has them all)
    model = model.to(device).eval()
    del weights
    size, stride = int(job.get("size", 512)), max(1, int(job.get("stride", 1)))
    c2w, focal, points, colors, counts, grid, world_grid = [], [], [], [], [], [], []
    state: tuple[Any, ...] = ()
    shape_wh: list[int] = []
    with torch.no_grad():
        for i, path in enumerate(job["images"]):
            img = load_images([str(path)], size=size, verbose=False)[0]
            view = {
                "img": img["img"].to(device),
                "ray_map": torch.full((1, 6, *img["img"].shape[-2:]), torch.nan, device=device),
                "true_shape": torch.from_numpy(img["true_shape"]).to(device),
                "img_mask": torch.tensor([True], device=device),
                "ray_mask": torch.tensor([False], device=device),
                "update": torch.tensor([True], device=device),
                "reset": torch.tensor([False], device=device),
            }
            shapes, feats, pos = model._encode_views([view])
            feat, pos_i, shape = feats[-1][0], pos[0], shapes[0]
            if i == 0:
                state_feat, state_pos = model._init_state(feat, pos_i)
                mem = model.pose_retriever.mem.expand(feat.shape[0], -1, -1)
                state = (state_feat.clone(), mem.clone(), state_feat, state_pos, mem)
            init_state_feat, init_mem, state_feat, state_pos, mem = state
            res, (state_feat, mem) = model._forward_decoder_step(
                {i: view}, i, feat, pos_i, shape, init_state_feat, init_mem, state_feat, state_pos, mem
            )
            state = (init_state_feat, init_mem, state_feat, state_pos, mem)
            pts = res["pts3d_in_self_view"].float().cpu()  # 1 x H x W x 3
            conf = res["conf_self"].float().cpu()[0]
            pose = pose_encoding_to_camera(res["camera_pose"].float().clone()).cpu()[0]
            h, w = pts.shape[1:3]
            shape_wh = [int(w), int(h)]
            pp = torch.tensor([[w // 2, h // 2]], dtype=torch.float32)
            focal.append(float(estimate_focal_knowing_depth(pts, pp, focal_mode="weiszfeld").ravel()[0]))
            c2w.append(pose.numpy().astype(np.float64))
            rgb = ((view["img"][0].permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            cam = pts[0].numpy()
            keep = (conf.numpy() > MIN_CONF)[::stride, ::stride]
            points.append(cam[::stride, ::stride][keep].astype(np.float32))
            colors.append(rgb[::stride, ::stride][keep])
            counts.append(int(keep.sum()))
            grid.append(cam[GRID // 2 :: GRID, GRID // 2 :: GRID].astype(np.float32))
            world = res["pts3d_in_other_view"].float().cpu()[0].numpy()
            world_grid.append(world[GRID // 2 :: GRID, GRID // 2 :: GRID].astype(np.float32))
            del res, view, feat, feats
    np.savez(
        out / "result.npz",
        c2w=np.stack(c2w),
        focal=np.array(focal),
        size=np.array(shape_wh),
        points=np.concatenate(points),
        colors=np.concatenate(colors),
        counts=np.array(counts),
        grid=np.stack(grid),
        world_grid=np.stack(world_grid),
    )
    peak = torch.cuda.max_memory_allocated() / 2**30
    (out / "result.json").write_text(
        json.dumps({"frames": len(c2w), "seconds": round(time.monotonic() - began, 1), "peak_gib": round(peak, 2)})
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
