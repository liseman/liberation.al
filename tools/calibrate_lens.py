"""Calibrate a wide-lens fixed camera's radial distortion from straight edges in its own frame: a plumb-line fit,
so no checkerboard is needed. spatial/lens.py says how the calibration is used.

    uv run tools/calibrate_lens.py --view bedroom.png:bedroom-lines.json --view kitchen.png:kitchen-lines.json
    uv run tools/calibrate_lens.py --view kitchen.png:kitchen-lines.json --check bedroom.png:bedroom-lines.json
    uv run tools/calibrate_lens.py --view bedroom.png:bedroom-lines.json --view kitchen.png:kitchen-lines.json \\
        --write config/devices.toml --camera bedroom --camera kitchen --mask 0.25,0,0.74,0.09 --mask 0.84,0.92,1,1

Each `--view` is a frame and a JSON list of rough polylines in it, `[[x, y], [x, y], ...]` in the frame's pixels, each
along one edge that's straight in the room (a door frame, a wardrobe's side, a rug's border; a long edge near the
frame's sides tells the most). Each is snapped to the strongest edge within `--width` pixels of it (the gradient
across the polyline, to a sub-pixel), its stray points dropped (`trim`), and the rest fitted. Views from cameras with
the same lens at the same size fit one calibration together; `--check` scores another view's lines without fitting
them.

The models are scored by how straight the lines come out (a line's RMS distance from its own best straight line over
its length, read as pixels of the frame along that length), each line also held out in turn (the others fit, it's
scored), so the score that counts is of lines the fit didn't see:

- **division** (one or two terms): Fitzgibbon's division model, what spatial/lens.py uses and `--write` writes.
- **brown** (k1, k2, or k1 alone): OpenCV's standard radial polynomial. On the Reolinks it folds before the
  frame's edges at the barrel they need (`folds`), so it can't undistort a whole frame.
- **fisheye**: OpenCV's equidistant fisheye model (see/fisheye.py's), its focal and k1, k2 free.

Each model's focal is half the frame's width (the fisheye's is fitted): a plumb-line fit fixes the distortion, not the
focal, which MASt3R finds on the undistorted frame. `--write` puts the division calibration, and any `--mask` overlay
rectangles (x1, y1, x2, y2 in fractions of the frame), on each `--camera`'s `[[device]]` entry in the manifest, keeping
the rest of the file as it is. Capture ignores both (restart it so it reads the manifest again); the next
pack-spatial-cameras run uses them.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import pathlib
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from liberational.spatial import lens as L

Points = np.ndarray  # (N, 2) pixels


@dataclass(frozen=True)
class Fit:
    model: str
    params: tuple[float, ...]
    score: float  # mean straightness over the lines, pixels
    held_out: float  # the same, each line scored by a fit without it


def snap(gray: np.ndarray, polyline: Sequence[Sequence[float]], width: float, step: float = 2.0) -> Points:
    """Points on the strongest edge within `width` pixels of `polyline`, about every `step` pixels along it: at each
    sample, the peak of the image's gradient across the polyline, to a sub-pixel by a parabola. Weak samples (under a
    third of the line's median peak) are left out."""
    import cv2

    blur = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
    gx, gy = cv2.Sobel(blur, cv2.CV_32F, 1, 0), cv2.Sobel(blur, cv2.CV_32F, 0, 1)
    out, strength = [], []
    pts = np.asarray(polyline, float)
    for a, b in itertools.pairwise(pts):
        seg = b - a
        length = float(np.hypot(*seg))
        if length < 1:
            continue
        t_hat = seg / length
        n_hat = np.array([-t_hat[1], t_hat[0]])
        offsets = np.arange(-width, width + 0.5, 0.5)
        for s in np.arange(0, length, step):
            p = a + s * t_hat
            xs, ys = p[0] + offsets * n_hat[0], p[1] + offsets * n_hat[1]
            inside = (xs >= 1) & (ys >= 1) & (xs < gray.shape[1] - 2) & (ys < gray.shape[0] - 2)
            if not inside.all():
                continue
            mx = cv2.remap(gx, xs.astype(np.float32)[None], ys.astype(np.float32)[None], cv2.INTER_LINEAR)[0]
            my = cv2.remap(gy, xs.astype(np.float32)[None], ys.astype(np.float32)[None], cv2.INTER_LINEAR)[0]
            g = np.abs(mx * n_hat[0] + my * n_hat[1])
            i = int(np.argmax(g))
            if 0 < i < len(g) - 1:
                den = g[i - 1] - 2 * g[i] + g[i + 1]
                i_f = i + (0.5 * (g[i - 1] - g[i + 1]) / den if den != 0 else 0.0)
            else:
                i_f = float(i)
            o = -width + 0.5 * i_f
            out.append(p + o * n_hat)
            strength.append(float(g[i]))
    if not out:
        return np.zeros((0, 2))
    keep = np.asarray(strength) >= np.median(strength) / 3
    return trim(np.asarray(out)[keep])


def trim(points: Points, rounds: int = 3) -> Points:
    """`points` without the ones that left the edge (onto a neighbouring one, or a bit of clutter): those further than
    3 MADs (and a pixel) from a quadratic through them along the line. A lens bends a straight edge into a smooth
    curve, never a step, so a quadratic follows what's left."""
    if len(points) < 8:
        return points
    for _ in range(rounds):
        c = points - points.mean(axis=0)
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        along, across = c @ vt[0], c @ vt[1]
        resid = across - np.polyval(np.polyfit(along, across, 2), along)
        mad = float(np.median(np.abs(resid - np.median(resid)))) * 1.4826
        ok = np.abs(resid) <= max(1.0, 3 * mad)
        if ok.all() or ok.sum() < 8:
            break
        points = points[ok]
    return points


def straightness(points: Points) -> float:
    """A line's RMS distance from its best straight line, over its length: 0 for a straight one."""
    c = points - points.mean(axis=0)
    _, s, _ = np.linalg.svd(c, full_matrices=False)
    length = float(np.ptp(c @ np.linalg.svd(c, full_matrices=False)[2][0]))
    return float(s[1] / math.sqrt(len(points)) / max(length, 1e-9))


def folds(width: int, height: int, k1: float, k2: float) -> bool:
    """Whether the brown model with focal half the width can't reach the frame's corners: its distorted radius stops
    growing (r (1 + k1 r^2 + k2 r^4) turns back) before it gets there, so the corners have no undistorted point."""
    corner = math.hypot(width / 2, height / 2) / (width / 2)
    r = np.linspace(0, 4, 4001)
    g = r * (1 + k1 * r**2 + k2 * r**4)
    reached = np.nonzero(g >= corner)[0]
    return not len(reached) or bool((np.diff(g[: reached[0] + 1]) <= 0).any())


def brown(width: int, height: int, params: Sequence[float]) -> Callable[[Points], Points]:
    """Undistortion of pixels under the brown model with k1, k2 (`params`), focal half the width, centred, by
    fixed-point iteration. A model that folds before the frame's corners (`folds`) can't undistort the frame: its
    points come back as NaN."""
    k1, k2 = params[0], params[1]
    if folds(width, height, k1, k2):
        return lambda p: np.full_like(p, np.nan, dtype=float)
    f, cx, cy = width / 2, (width - 1) / 2, (height - 1) / 2

    def undo(p: Points) -> Points:
        xd, yd = (p[:, 0] - cx) / f, (p[:, 1] - cy) / f
        x, y = xd.copy(), yd.copy()
        for _ in range(30):
            r2 = x * x + y * y
            g = 1 + k1 * r2 + k2 * r2 * r2
            x, y = xd / g, yd / g
        return np.column_stack([cx + f * x, cy + f * y])

    return undo


def fisheye(width: int, height: int, params: Sequence[float]) -> Callable[[Points], Points]:
    """Undistortion of pixels under the equidistant fisheye model: focal (as a fraction of the width), k1, k2."""
    f = params[0] * width
    k1, k2 = params[1], params[2]
    cx, cy = (width - 1) / 2, (height - 1) / 2

    def undo(p: Points) -> Points:
        x, y = (p[:, 0] - cx) / f, (p[:, 1] - cy) / f
        rd = np.hypot(x, y)
        theta = rd.copy()
        for _ in range(20):  # Newton on theta (1 + k1 theta^2 + k2 theta^4) = rd
            g = theta * (1 + k1 * theta**2 + k2 * theta**4) - rd
            dg = 1 + 3 * k1 * theta**2 + 5 * k2 * theta**4
            theta = np.clip(theta - g / dg, 0, 1.5)
        scale = np.where(rd > 1e-12, np.tan(theta) / np.maximum(rd, 1e-12), 1.0)
        return np.column_stack([cx + f * x * scale, cy + f * y * scale])

    return undo


def division(width: int, height: int, params: Sequence[float]) -> Callable[[Points], Points]:
    """Undistortion of pixels under the division model (Fitzgibbon's), focal half the width, centred: an undistorted
    point is the distorted one over (1 + l1 r^2 + l2 r^4), r its distance from the centre. It never folds, and undoes
    in closed form, but only short of its pole: a model whose frame corners come within spatial/lens.py's
    MIN_DENOMINATOR of it can't undo the frame (its points come back as NaN), as `lens_of` would refuse it."""
    f = width / 2
    l1, l2 = params[0], params[1] if len(params) > 1 else 0.0
    cx, cy = (width - 1) / 2, (height - 1) / 2
    corner = (width / 2 / f) ** 2 + (height / 2 / f) ** 2
    if 1 + l1 * corner + l2 * corner * corner < L.MIN_DENOMINATOR:
        return lambda p: np.full_like(p, np.nan, dtype=float)

    def undo(p: Points) -> Points:
        x, y = (p[:, 0] - cx) / f, (p[:, 1] - cy) / f
        r2 = x * x + y * y
        s = 1 / (1 + l1 * r2 + l2 * r2 * r2)
        return np.column_stack([cx + f * x * s, cy + f * y * s])

    return undo


MODELS: dict[str, tuple[Callable[[int, int, Sequence[float]], Callable[[Points], Points]], tuple[float, ...]]] = {
    "division": (division, (0.0,)),
    "division2": (division, (0.0, 0.0)),
    "brown": (brown, (0.0, 0.0)),
    "brown-k1": (lambda w, h, p: brown(w, h, (p[0], 0.0)), (0.0,)),
    "fisheye": (fisheye, (0.5, 0.0, 0.0)),
}


def score(lines: Sequence[Points], raw: Sequence[Points], undo: Callable[[Points], Points]) -> float:
    """Mean straightness of the lines after `undo`, each read as pixels along its length in the frame."""
    out = []
    for pts, orig in zip(lines, raw, strict=True):
        length = float(np.hypot(*np.ptp(orig, axis=0)))
        out.append(straightness(undo(pts)) * length)
    return float(np.mean(out))


def fit(model: str, lines: Sequence[Points], width: int, height: int) -> tuple[float, ...]:
    """The model's parameters that make `lines` straightest."""
    from scipy.optimize import least_squares

    make, start = MODELS[model]

    def residuals(p: np.ndarray) -> np.ndarray:
        undo = make(width, height, p.tolist())
        out = np.array([straightness(undo(pts)) if np.isfinite(undo(pts)).all() else 1.0 for pts in lines])
        return out

    return tuple(float(x) for x in least_squares(residuals, np.array(start), method="lm").x)


def evaluate(model: str, lines: Sequence[Points], width: int, height: int) -> Fit:
    make, _ = MODELS[model]
    params = fit(model, lines, width, height)
    held = []
    for i in range(len(lines)):
        rest = [p for j, p in enumerate(lines) if j != i]
        held.append(score([lines[i]], [lines[i]], make(width, height, fit(model, rest, width, height))))
    return Fit(model, params, score(lines, lines, make(width, height, params)), float(np.mean(held)))


def load(frame: pathlib.Path, lines_file: pathlib.Path, width: float) -> tuple[list[Points], int, int]:
    import cv2

    img = cv2.imread(str(frame), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"{frame}: not an image")
    polylines = json.loads(lines_file.read_text())
    lines = [p for p in (snap(img, pl, width) for pl in polylines) if len(p) >= 8]
    return lines, img.shape[1], img.shape[0]


def write(manifest: pathlib.Path, cams: Sequence[str], lens: L.Lens, masks: Sequence[Sequence[float]]) -> None:
    """Sets `lens` (and `masks`, when given) on each camera's entry in the manifest, keeping the rest as it is."""
    import tomlkit

    doc = tomlkit.parse(manifest.read_text())
    found = set()
    for device in doc.get("device", []):
        if device.get("name") in cams:
            table = tomlkit.inline_table()
            table.update(
                {
                    "width": lens.width,
                    "height": lens.height,
                    "k": [round(x, 3) for x in lens.k],
                    "division": round(lens.division, 4),
                }
            )
            device["lens"] = table
            if masks:
                device["masks"] = [[round(v, 4) for v in m] for m in masks]
            found.add(device.get("name"))
    missing = set(cams) - found
    if missing:
        raise SystemExit(f"not in {manifest}: {', '.join(sorted(missing))}")
    manifest.write_text(tomlkit.dumps(doc))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--view", action="append", required=True, help="FRAME:LINES to fit (repeat: one lens, many views)")
    ap.add_argument("--width", type=float, default=8.0, help="how far from a polyline its edge may be, pixels")
    ap.add_argument("--check", action="append", default=[], help="FRAME:LINES scored, not fitted")
    ap.add_argument("--write", type=pathlib.Path, help="the manifest to put the division calibration in")
    ap.add_argument("--camera", action="append", default=[], help="a camera to write it on (repeat for more)")
    ap.add_argument("--mask", action="append", default=[], help="an overlay rectangle x1,y1,x2,y2 (fractions)")
    args = ap.parse_args(argv)
    lines: list[Points] = []
    sizes = set()
    for spec in args.view:
        frame, lines_file = spec.split(":", 1)
        got, width, height = load(pathlib.Path(frame), pathlib.Path(lines_file), args.width)
        sizes.add((width, height))
        print(f"{frame}: {width} x {height}, {len(got)} lines; as it is {score(got, got, lambda p: p):.2f} px")
        lines += got
    if len(sizes) != 1:
        raise SystemExit(f"the views are different sizes ({sorted(sizes)}): fit one lens at one size")
    width, height = sizes.pop()
    if len(lines) < 3:
        raise SystemExit(f"only {len(lines)} lines snapped to an edge: give at least 3")
    print(
        f"all: {len(lines)} lines, {sum(len(p) for p in lines)} edge points; as it is {score(lines, lines, lambda p: p):.2f} px"
    )
    fits = {m: evaluate(m, lines, width, height) for m in MODELS}
    for f in fits.values():
        print(f"  {f.model}: {tuple(round(x, 4) for x in f.params)} -> {f.score:.2f} px (held out {f.held_out:.2f} px)")
    for spec in args.check:
        frame, lines_file = spec.split(":", 1)
        other, w2, h2 = load(pathlib.Path(frame), pathlib.Path(lines_file), args.width)
        print(f"check {frame}: {len(other)} lines; as it is {score(other, other, lambda p: p):.2f} px", end="")
        for m, f in fits.items():
            print(f", {m} {score(other, other, MODELS[m][0](w2, h2, f.params)):.2f} px", end="")
        print()
    found = fits["division"]
    lens = L.Lens(width, height, (width / 2, width / 2, (width - 1) / 2, (height - 1) / 2), found.params[0])
    if L.denominator(lens) < L.MIN_DENOMINATOR:
        raise SystemExit(
            f"the division fit ({lens.division:.4f}) can't undo the frame's corners: give edges nearer the corners"
        )
    print(f"calibration (division): k = {list(lens.k)}, division = {lens.division:.4f}, scale {L.out_scale(lens):.3f}")
    if args.write:
        if not args.camera:
            raise SystemExit("--write needs at least one --camera")
        masks = [tuple(float(v) for v in m.split(",")) for m in args.mask]
        write(args.write, args.camera, lens, masks)
        print(f"wrote it to {', '.join(args.camera)} in {args.write}; the next pack-spatial-cameras run uses it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
