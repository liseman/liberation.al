"""SOG, the compact copy of a place's Gaussian splat that the Places 3D view loads (#76): PlayCanvas's "spatially
ordered Gaussians" format, version 2, written from a 3DGS PLY's vertices (`encode`, for `places.write_splat`).

A `.sog` is a zip of a `meta.json` and lossless WebP images, a pixel a Gaussian, in Morton order (neighbours in
space are neighbours in the image, which is what makes the images small):

- `means_l.webp`, `means_u.webp`: each centre's x, y, z after sign(v) * ln(|v| + 1), as 16 bits between the
  `means.mins` and `means.maxs` meta.json names (low bytes in one image, high bytes in the other).
- `quats.webp`: the rotation (w, x, y, z), smallest three: alpha is 252 plus the index of its largest component
  (made positive), and red, green, and blue are the other three, in order, each (c * sqrt(2) + 1) / 2 as a byte.
- `scales.webp`: each log scale as an index into `scales.codebook`, 256 values.
- `sh0.webp`: each colour's DC coefficient as an index into `sh0.codebook`, and the opacity (its sigmoid) in alpha.
- `shN_centroids.webp`, `shN_labels.webp` (only with SH bands past 0): each Gaussian's higher SH coefficients are
  one of `shN.count` palette entries, 64 to a row, each entry one pixel a coefficient with red, green, and blue
  indexing `shN.codebook`; a Gaussian's entry is a 16-bit label, red the low byte and green the high.

The splat stays in the map's frame, as its PLY is: nothing is flipped. Checked by decoding with PlayCanvas's own
`splat-transform` and by drawing in the engine. On a 1M-Gaussian walk splat it takes about 10 s and gives a file
about a sixth the PLY's size.

The codebooks are 1-D k-means from quantile starts; the SH palette is k-means on a sample, every Gaussian then
given its nearest entry. The format is PlayCanvas's (MIT); this writer is our own.
"""

from __future__ import annotations

import io
import json
import math
import zipfile
from typing import Any

import numpy as np

VERSION = 2
CODEBOOK = 256
PALETTE = 4096  # SH palette entries at most (the format allows 65,536)
PALETTE_ROW = 64
PALETTE_SAMPLE = 100_000  # Gaussians the palette is fitted to
KMEANS_ROUNDS = 8
SH_COEFFS = (0, 3, 8, 15)  # coefficients past band 0, per channel, by SH degree
MIN_ALPHA = 1  # an opacity byte of 0 would let the WebP encoder drop the pixel's colour


def _webp(rgba: np.ndarray) -> bytes:
    """An (H, W, 4) uint8 RGBA image as a lossless WebP."""
    import cv2

    ok, data = cv2.imencode(".webp", np.ascontiguousarray(rgba[..., [2, 1, 0, 3]]), [cv2.IMWRITE_WEBP_QUALITY, 101])
    if not ok:
        raise ValueError("the WebP encoder refused an image")
    return data.tobytes()


def _image(pixels: np.ndarray, width: int, height: int) -> bytes:
    """`pixels` (N x 4 uint8, a Gaussian each) laid out row by row in a `width` x `height` image, the rest opaque
    black, as a WebP."""
    full = np.zeros((width * height, 4), np.uint8)
    full[:, 3] = 255
    full[: len(pixels)] = pixels
    return _webp(full.reshape(height, width, 4))


def codebook(values: np.ndarray, size: int = CODEBOOK) -> tuple[np.ndarray, np.ndarray]:
    """`size` values (sorted, float32) that stand for `values`, and each value's index into them (uint8, `values`'
    shape): the values themselves when there are no more than `size` of them, else 1-D k-means, so the entries
    crowd where the values do."""
    flat = np.asarray(values, np.float64).ravel()
    if not len(flat):
        return np.zeros(size, np.float32), np.zeros(np.shape(values), np.uint8)
    distinct = np.unique(flat)
    if len(distinct) <= size:  # few enough to keep each as it is
        centres = np.r_[distinct, np.full(size - len(distinct), distinct[-1])]
        return centres.astype(np.float32), np.searchsorted(distinct, flat).astype(np.uint8).reshape(np.shape(values))
    sample = flat if len(flat) <= 2_000_000 else np.random.default_rng(0).choice(flat, 2_000_000, replace=False)
    # Starts: half at the values' quantiles, half spread evenly over their range, so the few values far from the
    # crowd (a splat's largest scales) get entries too.
    half = size // 2
    centres = np.sort(
        np.r_[np.quantile(sample, (np.arange(half) + 0.5) / half), np.linspace(flat.min(), flat.max(), size - half)]
    )
    for _ in range(KMEANS_ROUNDS):
        labels = np.searchsorted((centres[1:] + centres[:-1]) / 2, sample)
        counts = np.bincount(labels, minlength=size)
        sums = np.bincount(labels, weights=sample, minlength=size)
        centres = np.sort(np.where(counts > 0, sums / np.maximum(counts, 1), centres))
    labels = np.searchsorted((centres[1:] + centres[:-1]) / 2, flat)
    return centres.astype(np.float32), labels.astype(np.uint8).reshape(np.shape(values))


def morton(xyz: np.ndarray) -> np.ndarray:
    """The order that sorts points (N x 3) along a Morton curve through their box: 10 bits an axis, interleaved."""
    lo, hi = xyz.min(0), xyz.max(0)
    cells = np.clip((xyz - lo) / np.maximum(hi - lo, 1e-12) * 1023, 0, 1023).astype(np.uint64)
    code = np.zeros(len(xyz), np.uint64)
    for bit in range(10):
        for axis in range(3):
            code |= ((cells[:, axis] >> np.uint64(bit)) & np.uint64(1)) << np.uint64(3 * bit + axis)
    return np.argsort(code, kind="stable")


def _quats(wxyz: np.ndarray) -> np.ndarray:
    """Unit rotations (N x 4: w, x, y, z) as smallest-three pixels (see the module docstring)."""
    q = wxyz / np.maximum(np.linalg.norm(wxyz, axis=1, keepdims=True), 1e-12)
    largest = np.argmax(np.abs(q), axis=1)
    q = q * np.where(q[np.arange(len(q)), largest] < 0, -1.0, 1.0)[:, None]
    others = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])[largest]
    packed = np.take_along_axis(q, others, axis=1) * math.sqrt(2) * 0.5 + 0.5
    out = np.empty((len(q), 4), np.uint8)
    out[:, :3] = np.clip(np.round(packed * 255), 0, 255)
    out[:, 3] = 252 + largest
    return out


def _palette(rest: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The SH palette for `rest` (N x 3K, a Gaussian's higher coefficients, channel by channel): its entries (at
    most PALETTE) and each Gaussian's entry."""
    from scipy.cluster.vq import kmeans2
    from scipy.spatial import KDTree

    rng = np.random.default_rng(0)
    sample = rest if len(rest) <= PALETTE_SAMPLE else rest[rng.choice(len(rest), PALETTE_SAMPLE, replace=False)]
    unique = np.unique(sample, axis=0)
    if len(unique) <= PALETTE:
        centres = unique
    else:
        centres, _ = kmeans2(sample, PALETTE, iter=KMEANS_ROUNDS, minit="points", seed=0)
    return centres, KDTree(centres).query(rest, workers=4)[1]


def encode(rows: np.ndarray) -> bytes:
    """A 3DGS PLY's vertices (`ply.vertices`: x, y, z, f_dc_*, f_rest_* channel by channel, opacity as a logit, log
    scale_*, and rot_0..3 as w, x, y, z) as a `.sog` file's bytes, in the PLY's own frame."""
    names = rows.dtype.names or ()
    rest_names = sorted((k for k in names if k.startswith("f_rest_")), key=lambda k: int(k.rsplit("_", 1)[1]))
    if len(rest_names) // 3 not in SH_COEFFS or len(rest_names) % 3:
        raise ValueError(f"{len(rest_names)} f_rest properties isn't SH degree 0 to 3")
    bands = SH_COEFFS.index(len(rest_names) // 3)

    def col(*keys: str) -> np.ndarray:
        return np.stack([np.asarray(rows[k], np.float64) for k in keys], axis=1) if keys else np.zeros((len(rows), 0))

    order = morton(col("x", "y", "z")) if len(rows) else np.zeros(0, np.int64)
    n = len(order)
    width = max(4, math.ceil(math.sqrt(n) / 4) * 4)
    height = max(4, math.ceil(n / width / 4) * 4)
    files: dict[str, bytes] = {}

    xyz = col("x", "y", "z")[order]
    logged = np.sign(xyz) * np.log1p(np.abs(xyz))
    mins = logged.min(0) if n else np.zeros(3)
    maxs = logged.max(0) if n else np.zeros(3)
    q16 = np.round((logged - mins) / np.where(maxs > mins, maxs - mins, 1.0) * 65535).astype(np.uint32)
    low = np.full((n, 4), 255, np.uint8)
    high = np.full((n, 4), 255, np.uint8)
    low[:, :3], high[:, :3] = q16 & 255, q16 >> 8
    files["means_l.webp"], files["means_u.webp"] = _image(low, width, height), _image(high, width, height)

    files["quats.webp"] = _image(_quats(col("rot_0", "rot_1", "rot_2", "rot_3")[order]), width, height)

    scale_book, scale_at = codebook(col("scale_0", "scale_1", "scale_2")[order])
    files["scales.webp"] = _image(np.c_[scale_at, np.full(n, 255, np.uint8)], width, height)

    colour_book, colour_at = codebook(col("f_dc_0", "f_dc_1", "f_dc_2")[order])
    alpha = np.round(255.0 / (1.0 + np.exp(-np.asarray(rows["opacity"], np.float64)[order])))
    files["sh0.webp"] = _image(np.c_[colour_at, np.clip(alpha, MIN_ALPHA, 255).astype(np.uint8)], width, height)

    meta: dict[str, Any] = {
        "version": VERSION,
        "asset": {"generator": "liberational spatial/sog.py"},
        "count": n,
        "means": {"mins": mins.tolist(), "maxs": maxs.tolist(), "files": ["means_l.webp", "means_u.webp"]},
        "scales": {"codebook": scale_book.tolist(), "files": ["scales.webp"]},
        "quats": {"files": ["quats.webp"]},
        "sh0": {"codebook": colour_book.tolist(), "files": ["sh0.webp"]},
    }
    if bands and n:
        coeffs = SH_COEFFS[bands]
        entries, label = _palette(col(*rest_names)[order])
        sh_book, entry_at = codebook(entries)
        # An entry is `coeffs` pixels in a row: pixel j holds coefficient j's red, green, and blue.
        by_channel = entry_at.reshape(len(entries), 3, coeffs).transpose(0, 2, 1)
        rows_high = math.ceil(len(entries) / PALETTE_ROW)
        wide = np.zeros((rows_high * PALETTE_ROW, coeffs, 4), np.uint8)
        wide[..., 3] = 255
        wide[: len(entries), :, :3] = by_channel
        wide = wide.reshape(rows_high, PALETTE_ROW * coeffs, 4)
        files["shN_centroids.webp"] = _webp(wide)
        labels = np.c_[label & 255, label >> 8, np.zeros(n), np.full(n, 255)].astype(np.uint8)
        files["shN_labels.webp"] = _image(labels, width, height)
        meta["shN"] = {
            "count": len(entries),
            "bands": bands,
            "codebook": sh_book.tolist(),
            "files": ["shN_centroids.webp", "shN_labels.webp"],
        }
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as bundle:
        stamp = (1980, 1, 1, 0, 0, 0)  # a fixed date, so the same splat gives the same bytes
        bundle.writestr(zipfile.ZipInfo("meta.json", stamp), json.dumps(meta))
        for name, data in files.items():
            bundle.writestr(zipfile.ZipInfo(name, stamp), data)
    return out.getvalue()
