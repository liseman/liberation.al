"""A 3D Gaussian splat of a place from its registered cameras' frames, run in gsplat's own runtime (#2877).
spatial/splats.py runs it; nothing that runs in the pack's environment imports it.

    <runtime>/.venv/bin/python splat_runner.py job.json

gsplat (nerfstudio-project/gsplat, Apache-2.0, pinned by `mise run gsplat`) needs torch with CUDA, which the pack's
own environment doesn't carry, so it lives in a venv of its own. This file imports nothing from the pack and only the
standard library at module level; numpy, torch, Pillow, and gsplat are imported inside `main`.

The job (JSON): `views` ([{"image": path, "pose": 16 numbers, camera to world row by row, the camera looking along
its +z (OpenCV), "focal": px, "width": px, "height": px}], every image already that size, its principal point at its
centre unless the view gives "fx", "fy", "cx", and "cy" (a phone's own intrinsics: a Stray Scanner walk's), and
optionally "mask": a PNG the same size whose white pixels (a camera's burned-in overlays, #2998) are left
out of the loss), `init` (an npz of `points`, N x 3 float32 metres, and `colors`, N x 3 uint8: the Gaussians' starting
centres and colours), `steps`, `cap` (the most Gaussians), `sh_degree`, `out` (a directory), and optionally
`holdout` (indices into `views` scored but not trained on, for measuring) and `reg` (the opacity and scale
regularizers' weight, REG_WEIGHT by default: a walkthrough's many views train better without them, #2877).

The training is the minimal trainer measured on #2877 (on gsplat's `rasterization`): gsplat's MCMC
strategy with a capped Gaussian count, an L1 plus 0.2 SSIM loss with small opacity and scale regularizers (none for a
walk: `reg`), and SH
degree raised one band every 1,000 steps. The poses are taken as they are (no pose refinement yet).

The answer is `out/splat.ply` (the standard 3DGS PLY: x, y, z, nx, ny, nz, f_dc_0..2, f_rest_* channel by channel,
opacity as a logit, log scales, and a w, x, y, z rotation) and `out/result.json`: {"seconds", "peak_vram_gib",
"gaussians", "init_points", "train_views", "train_psnr", "test_psnr" (null without a holdout), "gpu"}.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

C0 = 0.28209479177387814  # the SH band-0 constant: a colour c is the DC coefficient (c - 0.5) / C0
SSIM_WEIGHT = 0.2
REG_WEIGHT = 0.01  # each of the opacity and scale regularizers, as gsplat's MCMC examples use
INIT_OPACITY = 0.1
LR = {"scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
MEANS_LR = 1.6e-4  # times the scene's extent, decaying to 1% over the run
CDIST_FLOATS = 2**26  # 256 MiB of float32 distances per chunk when the start points' scales are measured


def main(argv: Sequence[str]) -> int:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from gsplat import rasterization
    from gsplat.strategy import MCMCStrategy
    from PIL import Image

    job = json.loads(Path(argv[0]).read_text())
    out = Path(job["out"])
    steps, sh_degree = int(job["steps"]), int(job["sh_degree"])
    reg = float(job.get("reg", REG_WEIGHT))
    dev = "cuda"
    torch.manual_seed(0)
    held = {int(i) for i in job.get("holdout", [])}

    views: list[dict[str, Any]] = []
    for i, v in enumerate(job["views"]):
        w, h, f = int(v["width"]), int(v["height"]), float(v["focal"])
        pixels = np.asarray(Image.open(v["image"]).convert("RGB"), np.uint8)  # kept as bytes: a walk has hundreds
        if pixels.shape[:2] != (h, w):
            raise ValueError(f"{v['image']} is {pixels.shape[1]}x{pixels.shape[0]}, not {w}x{h}")
        c2w = np.asarray(v["pose"], np.float64).reshape(4, 4)
        fx, fy = float(v.get("fx", f)), float(v.get("fy", f))
        k = np.array([[fx, 0, float(v.get("cx", w / 2))], [0, fy, float(v.get("cy", h / 2))], [0, 0, 1]], np.float32)
        covered = np.asarray(Image.open(v["mask"]).convert("L")) > 127 if v.get("mask") else None
        views.append(
            {
                "image": torch.from_numpy(pixels).to(dev),
                "viewmat": torch.from_numpy(np.linalg.inv(c2w).astype(np.float32)).to(dev),
                "K": torch.from_numpy(k).to(dev),
                "w": w,
                "h": h,
                "centre": c2w[:3, 3],
                "test": i in held,
                "covered": None if covered is None else torch.from_numpy(covered[..., None]).to(dev),
            }
        )
    train = [v for v in views if not v["test"]]
    test = [v for v in views if v["test"]]
    if not train:
        raise ValueError("no views to train on")
    centres = np.array([v["centre"] for v in views])
    extent = float(np.linalg.norm(centres - centres.mean(0), axis=1).max() * 1.1) or 1.0

    with np.load(job["init"]) as init:
        pts = np.asarray(init["points"], np.float32)
        rgb = np.asarray(init["colors"], np.float32) / 255.0
    if not len(pts):
        raise ValueError("no starting points")
    # Each point's starting scale: the RMS distance to its 3 nearest neighbours, in chunks of at most CDIST_FLOATS
    # distances (a walk's half million points would take 4 GiB a chunk of 2,048).
    p_t = torch.from_numpy(pts).to(dev)
    near = []
    chunk = max(1, CDIST_FLOATS // len(pts))
    for i in range(0, len(pts), chunk):
        d = torch.cdist(p_t[i : i + chunk], p_t).topk(min(4, len(pts)), largest=False).values[:, 1:]
        near.append((d**2).mean(1).sqrt() if d.shape[1] else torch.ones(len(d), device=dev))
    dist = torch.cat(near).clamp_min(1e-4)
    del p_t
    n, k_sh = len(pts), (sh_degree + 1) ** 2
    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.from_numpy(pts).to(dev)),
            "scales": torch.nn.Parameter(torch.log(dist)[:, None].repeat(1, 3)),
            "quats": torch.nn.Parameter(torch.rand(n, 4, device=dev)),
            "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), INIT_OPACITY, device=dev))),
            "sh0": torch.nn.Parameter(((torch.from_numpy(rgb).to(dev) - 0.5) / C0)[:, None, :]),
            "shN": torch.nn.Parameter(torch.zeros(n, k_sh - 1, 3, device=dev)),
        }
    )
    lrs = {"means": MEANS_LR * extent, **LR}
    optimizers = {k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15) for k, lr in lrs.items()}
    decay = 0.01 ** (1.0 / steps)
    strategy = MCMCStrategy(cap_max=int(job["cap"]), refine_start_iter=500, refine_stop_iter=int(steps * 0.8))
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state()

    def render(v: dict[str, Any], degree: int) -> tuple[Any, dict[str, Any]]:
        colors = torch.cat([params["sh0"], params["shN"]], dim=1)
        img, _, info = rasterization(
            params["means"],
            F.normalize(params["quats"], dim=-1),
            torch.exp(params["scales"]),
            torch.sigmoid(params["opacities"]),
            colors,
            v["viewmat"][None],
            v["K"][None],
            v["w"],
            v["h"],
            sh_degree=degree,
            packed=False,
        )
        return img[0].clamp(0, 1), info

    def ssim(a: Any, b: Any) -> Any:
        """Mean SSIM of two (H, W, 3) images in 0..1, with an 11 px Gaussian window."""
        g = torch.exp(-((torch.arange(11, device=dev) - 5) ** 2) / (2 * 1.5**2))
        g = g / g.sum()
        win = (g[:, None] @ g[None, :]).expand(3, 1, 11, 11)
        x, y = a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]

        def blur(z: Any) -> Any:
            return F.conv2d(z, win, padding=5, groups=3)

        mx, my = blur(x), blur(y)
        vx, vy, cov = blur(x * x) - mx**2, blur(y * y) - my**2, blur(x * y) - mx * my
        c1, c2 = 0.01**2, 0.03**2
        return (((2 * mx * my + c1) * (2 * cov + c2)) / ((mx**2 + my**2 + c1) * (vx + vy + c2))).mean()

    def truth(v: dict[str, Any]) -> Any:
        """`v`'s image as floats in 0..1 (it's kept on the GPU as bytes)."""
        return v["image"].float() / 255.0

    torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    rng = np.random.default_rng(0)
    for step in range(steps):
        v = train[int(rng.integers(len(train)))]
        img, info = render(v, min(sh_degree, step // 1000))
        want = truth(v)
        if v["covered"] is not None:  # an overlay's pixels match by fiat: no loss, no gradient
            img = torch.where(v["covered"], want, img)
        loss = (1 - SSIM_WEIGHT) * (img - want).abs().mean() + SSIM_WEIGHT * (1 - ssim(img, want))
        loss = loss + reg * (torch.sigmoid(params["opacities"]).mean() + torch.exp(params["scales"]).mean())
        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        optimizers["means"].param_groups[0]["lr"] *= decay
        strategy.step_post_backward(params, optimizers, state, step, info, lr=optimizers["means"].param_groups[0]["lr"])
    torch.cuda.synchronize()
    seconds = time.perf_counter() - began

    def psnr(group: list[dict[str, Any]]) -> float | None:
        if not group:
            return None
        with torch.no_grad():
            scores = [float(-10 * torch.log10(((render(v, sh_degree)[0] - truth(v)) ** 2).mean())) for v in group]
        return round(sum(scores) / len(scores), 2)

    m = len(params["means"])
    sh_rest = params["shN"].detach().transpose(1, 2).reshape(m, -1)  # channel by channel, as the 3DGS PLY has it
    cols = [
        params["means"].detach(),
        torch.zeros(m, 3, device=dev),
        params["sh0"].detach().reshape(m, 3),
        sh_rest,
        params["opacities"].detach()[:, None],
        params["scales"].detach(),
        F.normalize(params["quats"].detach(), dim=-1),
    ]
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(sh_rest.shape[1])]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    rows = torch.cat(cols, dim=1).cpu().numpy().astype("<f4")
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {m}\n"
    header += "".join(f"property float {k}\n" for k in names) + "end_header\n"
    (out / "splat.ply").write_bytes(header.encode("ascii") + rows.tobytes())
    result = {
        "seconds": round(seconds, 1),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "gaussians": m,
        "init_points": n,
        "train_views": len(train),
        "train_psnr": psnr(train),
        "test_psnr": psnr(test),
        "gpu": torch.cuda.get_device_name(),
    }
    (out / "result.json").write_text(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
