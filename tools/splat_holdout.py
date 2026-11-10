"""Score a Gaussian splat on held-out views, and compare two sets of camera poses (#22).

Measures a splat method against gsplat (spatial/splats.py) on the same frames, so a new trainer or a feed-forward
model can be compared on our own walks rather than on published numbers.

    <gsplat runtime>/.venv/bin/python tools/splat_holdout.py score JOB SPLAT.ply [--side 448] [--align-steps 300]
    .venv/bin/python tools/splat_holdout.py poses A.npz B.npz

`score` reads a spatial/splat_runner.py job (its `views` and `holdout`) and a standard 3DGS PLY (what the runner
writes, or any trainer's export in the same frame as the job's poses), and prints JSON: the mean PSNR over the
held-out views as posed, and after each view's camera is aligned to its frame with the splat frozen (`ALIGN_STEPS`
of Adam on the L1 error over a rotation, a translation, and one focal scale). Scoring a method whose cameras aren't
exactly the reference's needs that alignment, and it's the usual way to score pose-free methods; give both numbers.
With `--side`, each frame and its intrinsics are scaled so the short side is that many pixels and centre-cropped
square (what a feed-forward model at 448 px sees), so methods trained at different sizes are scored alike. It needs
torch and gsplat, so run it with gsplat's runtime (`mise run gsplat`, under `spatial.splat_runtime`); it imports
nothing from the pack.

`poses` reads two npz files, each with `c2w` (N x 4 x 4, camera to world, the same N frames in the same order), fits
the similarity that best maps A's camera centres onto B's (Umeyama), and prints the centres' error after it (RMSE,
median, and max, in B's units), the cameras' rotation error after it (median and max degrees), the fitted scale, and
B's path length.

On #22 (a 16-frame stretch of a Stray Scanner walk, 12 trained and 4 held out, scored at 448 px): AnySplat 13.9 dB
as posed and 15.0 aligned, gsplat from the LiDAR points 14.6 and 15.7, and gsplat started from AnySplat's Gaussians
14.6 and 16.2. gsplat on the whole walk (105 trained, 15 held out) scored 18.0 and 19.9.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

ALIGN_STEPS = 300
ALIGN_LR = 2e-3


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """The scale s, rotation r, and shift t that best map the points `src` (N x 3) onto `dst`: dst ~ s r src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    u, d, vt = np.linalg.svd(xd.T @ xs / len(src))
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1
    r = u @ sign @ vt
    s = float(np.trace(np.diag(d) @ sign) / ((xs**2).sum() / len(src)))
    return s, r, mu_d - s * r @ mu_s


def compare_poses(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """How far the cameras `a` (N x 4 x 4, camera to world) sit from `b` after the best similarity of `a` onto `b`."""
    s, r, t = umeyama(a[:, :3, 3], b[:, :3, 3])
    err = np.linalg.norm((s * (r @ a[:, :3, 3].T)).T + t - b[:, :3, 3], axis=1)
    turn = [
        math.degrees(math.acos(float(np.clip((np.trace((r @ x[:3, :3]).T @ y[:3, :3]) - 1) / 2, -1.0, 1.0))))
        for x, y in zip(a, b, strict=True)
    ]
    return {
        "path_length": round(float(np.linalg.norm(np.diff(b[:, :3, 3], axis=0), axis=1).sum()), 3),
        "centre_rmse": round(float(np.sqrt((err**2).mean())), 4),
        "centre_median": round(float(np.median(err)), 4),
        "centre_max": round(float(err.max()), 4),
        "rotation_median_deg": round(float(np.median(turn)), 2),
        "rotation_max_deg": round(float(np.max(turn)), 2),
        "scale": round(s, 5),
    }


def square_crop(k: np.ndarray, width: int, height: int, side: int) -> tuple[np.ndarray, float, int, int]:
    """Intrinsics `k` of a `width` x `height` frame once its short side is scaled to `side` and the middle `side`
    square cut out; with the scale and the crop's left and top offsets (in the scaled frame)."""
    s = side / min(width, height)
    left, top = (round(width * s) - side) // 2, (round(height * s) - side) // 2
    out = np.array(k, dtype=np.float64)
    out[:2] *= s
    out[0, 2] -= left
    out[1, 2] -= top
    return out, s, left, top


def read_ply(path: Path) -> dict[str, np.ndarray]:
    """A binary little-endian 3DGS PLY of float properties: means, unit quaternions (w, x, y, z), scales, opacities in
    0..1, and SH coefficients (N x K x 3, band 0 first)."""
    raw = path.read_bytes()
    head, body = raw.split(b"end_header\n", 1)
    props = [line.split()[-1].decode() for line in head.splitlines() if line.startswith(b"property")]
    rows = np.frombuffer(body, dtype="<f4").reshape(-1, len(props))
    col = {p: rows[:, j] for j, p in enumerate(props)}
    rest = sorted((p for p in props if p.startswith("f_rest_")), key=lambda p: int(p.rsplit("_", 1)[1]))
    dc = np.stack([col["f_dc_0"], col["f_dc_1"], col["f_dc_2"]], 1)[:, None, :]
    sh = dc
    if rest:  # stored channel by channel: all of red's coefficients, then green's, then blue's
        sh = np.concatenate([dc, np.stack([col[p] for p in rest], 1).reshape(len(rows), 3, -1).transpose(0, 2, 1)], 1)
    quats = np.stack([col[f"rot_{i}"] for i in range(4)], 1)
    return {
        "means": np.stack([col["x"], col["y"], col["z"]], 1),
        "quats": quats / np.linalg.norm(quats, axis=1, keepdims=True),
        "scales": np.exp(np.stack([col[f"scale_{i}"] for i in range(3)], 1)),
        "opacities": 1 / (1 + np.exp(-col["opacity"])),
        "sh": np.ascontiguousarray(sh),
    }


def score(job: dict[str, Any], splat: dict[str, np.ndarray], side: int | None, steps: int) -> dict[str, Any]:
    """Mean PSNR of `splat` over `job`'s held-out views, as posed and after aligning each view's camera."""
    import torch
    from gsplat import rasterization
    from PIL import Image

    dev = "cuda"
    g = {k: torch.from_numpy(np.ascontiguousarray(v)).float().to(dev) for k, v in splat.items()}
    degree = math.isqrt(g["sh"].shape[1]) - 1
    held = [int(i) for i in job.get("holdout", [])]
    if not held:
        raise ValueError("the job holds no views out")
    centres = np.array([np.reshape(u["pose"], (4, 4))[:3, 3] for u in job["views"]])
    unit = float(np.median(np.linalg.norm(np.diff(centres, axis=0), axis=1))) or 1.0  # the alignment's step in metres
    before, after = [], []
    for i in held:
        v = job["views"][i]
        w, h = int(v["width"]), int(v["height"])
        f = float(v["focal"])
        k = np.array(
            [
                [float(v.get("fx", f)), 0, float(v.get("cx", w / 2))],
                [0, float(v.get("fy", f)), float(v.get("cy", h / 2))],
                [0, 0, 1],
            ]
        )
        image = Image.open(v["image"]).convert("RGB")
        if side is not None:
            k, s, left, top = square_crop(k, w, h, side)
            image = image.resize((round(w * s), round(h * s)), Image.Resampling.BILINEAR)
            image = image.crop((left, top, left + side, top + side))
            w = h = side
        truth = torch.from_numpy(np.asarray(image, np.float32) / 255.0).to(dev)
        w2c = torch.linalg.inv(torch.tensor(v["pose"], dtype=torch.float32, device=dev).reshape(4, 4))
        k_t = torch.from_numpy(k).float().to(dev)
        rot = torch.zeros(3, device=dev, requires_grad=True)
        shift = torch.zeros(3, device=dev, requires_grad=True)
        zoom = torch.zeros(1, device=dev, requires_grad=True)
        opt = torch.optim.Adam([rot, shift, zoom], lr=ALIGN_LR)

        def render(
            rot: Any = rot, shift: Any = shift, zoom: Any = zoom, w2c: Any = w2c, k_t: Any = k_t, w: int = w, h: int = h
        ) -> Any:
            zero = torch.zeros((), device=dev)
            skew = torch.stack(
                [
                    torch.stack([zero, -rot[2], rot[1]]),
                    torch.stack([rot[2], zero, -rot[0]]),
                    torch.stack([-rot[1], rot[0], zero]),
                ]
            )
            move = torch.eye(4, device=dev)
            move = torch.cat([torch.cat([torch.linalg.matrix_exp(skew), (shift * unit)[:, None]], 1), move[3:]], 0)
            focal = torch.cat([torch.exp(zoom).repeat(2), torch.ones(1, device=dev)])[:, None]
            img, _, _ = rasterization(
                g["means"],
                g["quats"],
                g["scales"],
                g["opacities"],
                g["sh"],
                (move @ w2c)[None],
                (k_t * focal)[None],
                w,
                h,
                sh_degree=degree,
                packed=False,
            )
            return img[0].clamp(0, 1)

        def psnr(img: Any, truth: Any = truth) -> float:
            return float(-10 * torch.log10(((img - truth) ** 2).mean()))

        with torch.no_grad():
            before.append(psnr(render()))
        for _ in range(steps):
            loss = (render() - truth).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            after.append(psnr(render()))
    return {
        "held_out": len(held),
        "gaussians": len(splat["means"]),
        "side": side,
        "psnr_as_posed": round(float(np.mean(before)), 2),
        "psnr_aligned": round(float(np.mean(after)), 2),
        "per_view_aligned": [round(x, 2) for x in after],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """`score` or `poses` (see the module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score", help="PSNR of a splat on a runner job's held-out views")
    s.add_argument("job", type=Path)
    s.add_argument("splat", type=Path)
    s.add_argument("--side", type=int, default=None, help="score square crops this many pixels a side")
    s.add_argument("--align-steps", type=int, default=ALIGN_STEPS)
    p = sub.add_parser("poses", help="two pose sets' agreement after a similarity")
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    args = parser.parse_args(argv)
    if args.command == "score":
        result = score(json.loads(args.job.read_text()), read_ply(args.splat), args.side, args.align_steps)
    else:
        with np.load(args.a) as a, np.load(args.b) as b:
            result = compare_poses(np.asarray(a["c2w"], np.float64), np.asarray(b["c2w"], np.float64))
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
