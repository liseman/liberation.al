"""A 3D Gaussian splat of a place from its registered cameras' frames, run in gsplat's own runtime.
spatial/splats.py runs it; nothing that runs in the pack's environment imports it.

    <runtime>/.venv/bin/python splat_runner.py job.json

gsplat (nerfstudio-project/gsplat, Apache-2.0, pinned by `mise run gsplat`) needs torch with CUDA, which the pack's
own environment doesn't carry, so it lives in a venv of its own. This file imports nothing from the pack and only the
standard library at module level; numpy, torch, Pillow, and gsplat are imported inside `main`.

The job (JSON): `views` ([{"image": path, "pose": 16 numbers, camera to world row by row, the camera looking along
its +z (OpenCV), "focal": px, "width": px, "height": px}], every image already that size, its principal point at its
centre unless the view gives "fx", "fy", "cx", and "cy" (a phone's own intrinsics: a Stray Scanner walk's), and
optionally "mask": a PNG the same size whose white pixels (a camera's burned-in overlays) are left
out of the loss, and "depth": a 16-bit image of the view's measured depth along its axis, 0 where there's none, covering
the same field of view at any size), `init` (an npz of `points`, N x 3 float32 metres, and `colors`, N x 3 uint8: the
Gaussians' starting centres and colours), `steps`, `cap` (the most Gaussians), `sh_degree`, `out` (a directory), and
optionally `holdout` (indices into `views` scored but not trained on, for measuring), `reg` (the opacity and scale
regularizers' weight, REG_WEIGHT by default: a walkthrough's many views train better without them), `bound` (true to
keep the Gaussians inside the start points' box), `depth_unit` (what one count of a depth image is in the poses'
units, 0.001 by default: millimetres in a metric scene), `pose_opt` (true to refine every training view's pose
while the splat trains), and `render` (a directory: every view as the trained splat draws it at its given pose,
`f000.png` on, and for a view with a depth map its drawn depth, `d000.png` on, at that map's size and unit; a walk
whose frames sit exactly at their poses, for checking what the refinement reports).

The training is the minimal trainer on gsplat's `rasterization`: gsplat's MCMC strategy with a capped Gaussian count,
an L1 plus 0.2 SSIM loss with small opacity and scale regularizers (none for a walk: `reg`), and SH degree raised one
band every 1,000 steps.

A walkthrough's job sets `bound` and `pose_opt` and gives its LiDAR depth (#43). Trained without them, one LiDAR walk
of a home (120 frames) put 10% of its Gaussians outside the map's box, the farthest 985 m away,
and scored 18.8 dB on held-out frames; with them 0.2% end outside, the farthest 8.7 m from the box's centre, at 21.6
dB, in about the same time and GPU memory (10 minutes, 1.1 GiB).

**Bounds** (`bound`): a walk trains without the regularizers, and its views look out of windows and through open
doors, so nothing stops a Gaussian from sliding outward to paint what's far away. And MCMC's position noise is a
Gaussian's covariance times the means' learning rate, which is itself scaled by the scene's extent: at an extent of
E metres the noise is E squared too large for the scene, so a large, faint Gaussian is thrown tens of metres a step.
With `bound` the noise is divided by the extent squared, as if the scene were scaled to unit extent. The box is the
start points' (`BOX_PERCENTILE` to 100 minus it on each axis, so a few stray map points don't widen it), grown by
`BOX_PAD` of its size on every side. Every `BOUND_EVERY` steps a Gaussian outside it is made transparent, which the
MCMC strategy reads as dead and relocates onto a live one in the scene while it still refines; every step a
Gaussian's scale is held under `MAX_SCALE` of the box's diagonal. At the end the Gaussians outside the box, and the
transparent ones, are dropped. What's beyond the box is painted on its edge, not left out: the views still match.

**Depth** (a view's "depth"): the rendered expected depth at each measured pixel is pulled to the measurement (L1 in
metres, `DEPTH_WEIGHT` beside the image loss), which keeps surfaces where the LiDAR saw them: 2 cm median error on
held-out frames. On its own it didn't change the held-out image score (the map the Gaussians start from is the
same LiDAR's); it holds the geometry in place while the poses move.

**Pose refinement** (`pose_opt`): a walk's frames don't sit exactly at their poses (the LiDAR walk's needed a median
4 cm and 0.8 degrees each, several pixels; a plain video's are worse), and a splat trained at them blurs: every
frame fits on its own, and a view between them doesn't. With `pose_opt` each training view gets its own small
correction, as gsplat's `simple_trainer --pose_opt` does: a translation (in units of the scene's extent) and a
rotation (the 6D form, from identity), applied in the camera's own frame (c2w @ delta), from `POSE_START` steps on,
once the Gaussians have a rough shape to align to. The corrections are rows of one sparse embedding stepped by
SparseAdam, so each view's row moves only on the steps that render it, about `POSE_LR` a visit, the rate decaying
to 1% over the run like the means'. Without it the poses are taken as they are. It was worth 1.3 dB on held-out
frames at 7,000 steps (21.3 against 20.0).

The correction it reports (`pose_shift`, `pose_turn_deg`) is what the frames need, not the step's own wandering
(#197; one LiDAR walk, 105 training views, 7,000 steps, `tools/splat_walk_check.py`). A copy of the walk drawn from
its trained splat, so every frame sits exactly at its pose (`render`), reports 0.3 cm and 0.04 degrees; with every
pose then moved 2 cm and 0.3 degrees in its own direction it reports 1.9 cm and 0.29 degrees, and the refined poses
end a median 0.5 cm and 0.09 degrees from the truth. On the walk's own frames it reports 2.4 cm and 0.38 degrees
(4.7 px for a point 2.5 m away), and two runs with different seeds agree view by view: the part they share is 3.0 cm
and 0.56 degrees RMS, the part they don't 0.45 cm and 0.10 degrees. So the reading's floor is about half a
centimetre and a tenth of a degree, and what's above it is in the poses (or in what a pose can stand in for: a
rolling shutter, the lens). A smaller step lowers the reading only by not reaching the frames: at an eighth of the
rate a known 2 cm reads 1.3 cm, and the held-out score drops 0.1 dB.

**Scores** (PSNR) count only the pixels the loss counts: a view's covered pixels (its "mask") are never trained, so
counting them only measured how much of the frame they cover (a camera with a wide lens undone scored 17 dB for a
29 dB splat).

**Held-out views** are scored twice: at their poses as given (`test_psnr_raw`), and after `ALIGN_STEPS` steps that
fit only that view's pose correction to its image with the Gaussians frozen (`test_psnr`). A held-out frame's given
pose is as far off as a training frame's was, and a refined splat has settled where its training frames agree, so
the given pose no longer lands exactly; the fit measures the splat, not that offset (the usual way pose-refining
trainers are scored). Both are reported for every run, refined or not, so two runs compare.

The answer is `out/splat.ply` (the standard 3DGS PLY: x, y, z, nx, ny, nz, f_dc_0..2, f_rest_* channel by channel,
opacity as a logit, log scales, and a w, x, y, z rotation) and `out/result.json`: {"seconds", "peak_vram_gib",
"gaussians", "init_points", "train_views", "train_psnr", "test_psnr" and "test_psnr_raw" (null without a holdout),
"test_depth_m" (the held-out views' median depth error, null without a holdout that has depth), "pose_opt",
"pose_shift" and "pose_turn_deg" (the median correction a training view ended with, null without `pose_opt`),
"dropped" (Gaussians dropped at the end for being outside the box or transparent, 0 without `bound`), "scale_p99"
(the 99th percentile of the Gaussians' largest scale), "gpu"}.
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
BOX_PERCENTILE = 0.5  # the box (`bound`) spans these percentiles of the start points on each axis...
BOX_PAD = 0.1  # ...grown by this much of its size on every side
BOUND_EVERY = 100  # steps between sending escaped Gaussians back (MCMC relocates every 100 steps)
POSE_START = 1000  # steps before the poses start to move (`pose_opt`)
POSE_LR = 1e-3  # SparseAdam's rate for a view's correction, per visit; decays to 1% over the run
ALIGN_STEPS = 200  # steps fitting a held-out view's pose to the frozen splat before it's scored
ALIGN_LR = 1e-3
IDENTITY_6D = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)  # the 6D rotation form's identity: the first two rows of I
DEPTH_WEIGHT = 0.2  # the depth loss (metres of mean error) beside the image loss
MAX_SCALE = 0.03  # of the box's diagonal: a Gaussian's largest axis (a 18 m home: about half a metre)
DEAD_OPACITY = 0.005  # gsplat's MCMC `min_opacity`: at or under it a Gaussian is relocated, and dropped at the end


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
    depth_unit = float(job.get("depth_unit", 0.001))

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
        at = metres = None
        if v.get("depth"):
            # The depth map covers the image's field of view at its own, smaller size: each measured depth pixel's
            # centre lands on one image pixel, whose rendered depth it's compared with.
            d = np.asarray(Image.open(v["depth"]), np.float32) * depth_unit
            dv, du = np.nonzero(d > 0)
            x = np.clip(np.round((du + 0.5) * w / d.shape[1] - 0.5), 0, w - 1).astype(np.int64)
            y = np.clip(np.round((dv + 0.5) * h / d.shape[0] - 0.5), 0, h - 1).astype(np.int64)
            if len(dv):
                at, metres = torch.from_numpy(y * w + x).to(dev), torch.from_numpy(d[dv, du]).to(dev)
        views.append(
            {
                "image": torch.from_numpy(pixels).to(dev),
                "c2w": torch.from_numpy(c2w.astype(np.float32)).to(dev),
                "K": torch.from_numpy(k).to(dev),
                "w": w,
                "h": h,
                "centre": c2w[:3, 3],
                "test": i in held,
                "covered": None if covered is None else torch.from_numpy(covered[..., None]).to(dev),
                "depth_at": at,
                "depth": metres,
                "depth_shape": d.shape if v.get("depth") else None,
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
    bound = bool(job.get("bound", False))
    lo, hi = np.percentile(pts, BOX_PERCENTILE, axis=0), np.percentile(pts, 100 - BOX_PERCENTILE, axis=0)
    pad = BOX_PAD * (hi - lo)
    box_lo = torch.tensor(lo - pad, dtype=torch.float32, device=dev)
    box_hi = torch.tensor(hi + pad, dtype=torch.float32, device=dev)
    max_log_scale = float(np.log(MAX_SCALE * np.linalg.norm(hi - lo + 2 * pad)))
    if bound:
        dist = dist.clamp_max(float(np.exp(max_log_scale)))
    # MCMC's position noise is a Gaussian's covariance times the means' rate. Both grow with the scene's size (the
    # covariance by its square), so at an extent of E metres the noise is E^2 too large for the scene; a bounded job
    # divides it back out.
    noise_div = extent**2 if bound else 1.0

    def outside() -> Any:
        """Which Gaussians are outside the box."""
        m = params["means"]
        return ((m < box_lo) | (m > box_hi)).any(dim=1)

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

    def corrected(c2w: Any, delta: Any) -> Any:
        """`c2w` moved by `delta` (9 numbers: a translation in units of `extent`, then the 6D rotation's offset
        from identity), in the camera's own frame."""
        d6 = delta[3:] + torch.tensor(IDENTITY_6D, device=dev)
        b1 = F.normalize(d6[:3], dim=-1)
        b2 = F.normalize(d6[3:] - (b1 * d6[3:]).sum() * b1, dim=-1)
        m = torch.eye(4, device=dev)
        m[:3, :3] = torch.stack([b1, b2, torch.linalg.cross(b1, b2)], dim=-2)
        m[:3, 3] = delta[:3] * extent
        return c2w @ m

    refine = bool(job.get("pose_opt", False))
    index = {id(v): i for i, v in enumerate(train)}
    poses = torch.nn.Embedding(len(train), 9, sparse=True).to(dev)
    torch.nn.init.zeros_(poses.weight)
    pose_opt = torch.optim.SparseAdam(list(poses.parameters()), lr=POSE_LR)

    def viewmat(v: dict[str, Any], delta: Any | None = None) -> Any:
        """`v`'s world-to-camera matrix: its pose, moved by `delta` when given, or by its trained correction."""
        if delta is None and refine and not v["test"]:
            delta = poses(torch.tensor([index[id(v)]], device=dev))[0]
        return torch.linalg.inv(v["c2w"] if delta is None else corrected(v["c2w"], delta))

    def render(
        v: dict[str, Any], degree: int, depth: bool = False, delta: Any | None = None
    ) -> tuple[Any, Any, dict[str, Any]]:
        """`v` rendered (at its pose moved by `delta` when given, else by its trained correction): its image (H, W,
        3) in 0..1, its expected depth (H, W) when `depth`, else None, and gsplat's info."""
        colors = torch.cat([params["sh0"], params["shN"]], dim=1)
        img, _, info = rasterization(
            params["means"],
            F.normalize(params["quats"], dim=-1),
            torch.exp(params["scales"]),
            torch.sigmoid(params["opacities"]),
            colors,
            viewmat(v, delta)[None],
            v["K"][None],
            v["w"],
            v["h"],
            sh_degree=degree,
            packed=False,
            render_mode="RGB+ED" if depth else "RGB",
        )
        return img[0, ..., :3].clamp(0, 1), img[0, ..., 3] if depth else None, info

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

    def photometric(img: Any, v: dict[str, Any]) -> Any:
        """The L1 plus SSIM loss of `img` against `v`'s image, its overlay pixels left out."""
        want = truth(v)
        if v["covered"] is not None:  # an overlay's pixels match by fiat: no loss, no gradient
            img = torch.where(v["covered"], want, img)
        return (1 - SSIM_WEIGHT) * (img - want).abs().mean() + SSIM_WEIGHT * (1 - ssim(img, want))

    torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    rng = np.random.default_rng(0)
    for step in range(steps):
        v = train[int(rng.integers(len(train)))]
        moving = refine and step >= POSE_START
        still = None if moving or not refine else torch.zeros(9, device=dev)  # no gradient reaches the rows yet
        img, seen, info = render(v, min(sh_degree, step // 1000), depth=v["depth"] is not None, delta=still)
        loss = photometric(img, v)
        loss = loss + reg * (torch.sigmoid(params["opacities"]).mean() + torch.exp(params["scales"]).mean())
        if seen is not None:
            loss = loss + DEPTH_WEIGHT * (seen.reshape(-1)[v["depth_at"]] - v["depth"]).abs().mean()
        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        optimizers["means"].param_groups[0]["lr"] *= decay
        if moving:
            pose_opt.step()
            pose_opt.zero_grad(set_to_none=True)
            pose_opt.param_groups[0]["lr"] *= decay
        strategy.step_post_backward(
            params, optimizers, state, step, info, lr=optimizers["means"].param_groups[0]["lr"] / noise_div
        )
        if bound:
            with torch.no_grad():
                params["scales"].clamp_(max=max_log_scale)
                if step % BOUND_EVERY == BOUND_EVERY - 1:
                    params["opacities"][outside()] = float(np.log(DEAD_OPACITY / 2 / (1 - DEAD_OPACITY / 2)))
    torch.cuda.synchronize()
    seconds = time.perf_counter() - began

    def score(v: dict[str, Any], delta: Any | None = None) -> float:
        """`v`'s PSNR, over the pixels the loss counts: a covered pixel (an overlay, or the border a lens's undoing
        leaves) is never trained, so it would only say how much of the frame is covered."""
        with torch.no_grad():
            err = (render(v, sh_degree, delta=delta)[0] - truth(v)) ** 2
            if v["covered"] is not None:
                err = err[~v["covered"][..., 0]]
            return float(-10 * torch.log10(err.mean()))

    def aligned(v: dict[str, Any]) -> tuple[float, Any]:
        """`v`'s score after fitting only its pose correction to its image, the Gaussians frozen, and that
        correction."""
        delta = torch.zeros(9, device=dev, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=ALIGN_LR)
        for _ in range(ALIGN_STEPS):
            photometric(render(v, sh_degree, delta=delta)[0], v).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        return score(v, delta.detach()), delta.detach()

    def mean(scores: list[float]) -> float | None:
        return round(sum(scores) / len(scores), 2) if scores else None

    def depth_error(group: list[dict[str, Any]]) -> float | None:
        """The median, over `group`'s views with a depth map, of each one's median depth error in metres."""
        with torch.no_grad():
            errs = [
                float(
                    (render(v, sh_degree, depth=True, delta=fits.get(id(v)))[1].reshape(-1)[v["depth_at"]] - v["depth"])
                    .abs()
                    .median()
                )
                for v in group
                if v["depth"] is not None
            ]
        return round(float(np.median(errs)), 4) if errs else None

    dropped = 0
    if bound:
        with torch.no_grad():
            keep = ~outside() & (torch.sigmoid(params["opacities"]) > DEAD_OPACITY)
        dropped = int((~keep).sum())
        for k in list(params.keys()):
            params[k] = torch.nn.Parameter(params[k].detach()[keep])
    train_psnr = mean([score(v) for v in train])
    test_raw = mean([score(v, torch.zeros(9, device=dev)) for v in test])
    for param in params.values():
        param.requires_grad_(False)
    fitted = [(v, *aligned(v)) for v in test]
    fits = {id(v): delta for v, _, delta in fitted}
    shift = turn = None
    if refine:
        moved = poses.weight.detach()
        shift = round(float(moved[:, :3].norm(dim=1).median() * extent), 4)
        rel = torch.stack([corrected(torch.eye(4, device=dev), d)[:3, :3] for d in moved])
        cos = ((rel.diagonal(dim1=1, dim2=2).sum(1) - 1) / 2).clamp(-1, 1)
        turn = round(float(torch.rad2deg(torch.arccos(cos)).median()), 3)
    if job.get("render"):
        drawn = Path(job["render"])
        drawn.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            for i, v in enumerate(views):
                img, seen, _ = render(v, sh_degree, depth=True, delta=torch.zeros(9, device=dev))
                Image.fromarray((img * 255).round().byte().cpu().numpy()).save(drawn / f"f{i:03d}.png")
                if v["depth_shape"] is None:
                    continue
                # Each depth pixel takes the drawn depth at the image pixel its centre lands on, as the loss reads it.
                dh, dw = v["depth_shape"]
                x = ((torch.arange(dw, device=dev) + 0.5) * v["w"] / dw - 0.5).round().clamp(0, v["w"] - 1).long()
                y = ((torch.arange(dh, device=dev) + 0.5) * v["h"] / dh - 0.5).round().clamp(0, v["h"] - 1).long()
                counts = (seen[y][:, x] / depth_unit).round().clamp(0, 65535).cpu().numpy().astype(np.uint16)
                Image.fromarray(counts).save(drawn / f"d{i:03d}.png")
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
        "train_psnr": train_psnr,
        "test_psnr": mean([got for _, got, _ in fitted]),
        "test_psnr_raw": test_raw,
        "pose_opt": refine,
        "pose_shift": shift,
        "pose_turn_deg": turn,
        "test_depth_m": depth_error(test),
        "dropped": dropped,
        "scale_p99": round(float(torch.quantile(torch.exp(params["scales"].detach()).max(1).values, 0.99)), 4),
        "gpu": torch.cuda.get_device_name(),
    }
    (out / "result.json").write_text(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
