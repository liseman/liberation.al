"""Revisit proposals for a walkthrough: which pairs of a walk's frames may see the same place again, so MASt3R
measures only those (spatial/walkthrough.py).

MegaLoc (gberton/MegaLoc, MIT, CVPR 2025 Workshops) is a place-recognition model: one 8448-number descriptor per image,
whose cosine similarity says how much two views look like the same place. The pinned ONNX export
(`[models.files.megaloc]`, fetched by `mise run models`) runs here through onnxruntime on the CPU, so it takes no GPU
memory from CUT3R, MASt3R, or the live services; nothing from MegaLoc's repository runs. Each frame is resized to
322 x 322 and ImageNet-normalized, as its authors evaluate it.

`proposals` keeps the `PROPOSALS` most similar pairs at least `LOOP_GAP` frames apart (`shortlist`). On IMG_7676 (125
frames at 1 fps) that replaced ORB matching plus pairs at fixed steps of 32 and 64 frames, which proposed 79 pairs:
MegaLoc's best three of about 4,600 were exactly the three revisits whose near walls ICP fits, its 25 came within two
frames of 10 of the 12 revisits MASt3R had been sure of, and 14 of the 20 pairs the old proposals never made had more
than 1,000 MASt3R matches. The map came out the same (wall band within noise), for a third of the MASt3R pairs.
It takes about 0.35 s a frame on 4 CPU threads.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from liberational import config as cfg
from liberational import logs

LOG = logs.get("walkthrough")

# A pair is at least LOOP_GAP frames apart (30 s at one a second), so a walk's own neighbours aren't proposed as
# revisits.
LOOP_GAP = 30
PROPOSALS = 25  # pairs over the whole walk at most (MASt3R checks each in a few seconds)
PARTNERS = 2  # proposals per frame at most
SPREAD = 2  # a proposal within this many frames, at both ends, of a better one adds nothing new
SIDE = 322  # MegaLoc's input, square
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # ImageNet's, RGB
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
THREADS = 4


def similarity(images: Sequence[Path], model: Path | None = None) -> np.ndarray:
    """Every pair of `images`' MegaLoc cosine similarity (n x n), from the ONNX export at `model` (by default
    `[models.files.megaloc]`). Raises RuntimeError when the model or an image is missing."""
    import cv2
    import onnxruntime as ort

    model = cfg.model_path("megaloc") if model is None else model
    if not model.exists():
        raise RuntimeError(f"MegaLoc isn't at {model}; run `mise run models`")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = THREADS
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model), opts, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    out = []
    for path in images:
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise RuntimeError(f"can't read the frame {path}")
        rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (SIDE, SIDE), interpolation=cv2.INTER_AREA)
        x = ((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
        out.append(np.asarray(session.run(None, {name: x})[0], dtype=np.float64)[0])
    d = np.array(out).reshape(len(out), -1)
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    return d @ d.T


def shortlist(
    scores: np.ndarray,
    gap: int = LOOP_GAP,
    partners: int = PARTNERS,
    cap: int = PROPOSALS,
    spread: int = SPREAD,
) -> list[tuple[int, int]]:
    """The pairs (i, j), j at least `gap` after i, worth asking MASt3R about from `scores` (n x n, higher is more alike):
    best first, each frame in at most `partners` of them, none within `spread` frames at both ends of a better one (the
    same revisit seen from the next frame), and at most `cap`; returned sorted."""
    n = len(scores)
    i_s, j_s = np.nonzero(np.triu(np.ones((n, n), dtype=bool), gap))
    order = sorted(zip(i_s.tolist(), j_s.tolist(), strict=True), key=lambda p: (-float(scores[p]), p))
    used = np.zeros(n, dtype=np.int64)
    taken: list[tuple[int, int]] = []
    for i, j in order:
        if len(taken) >= cap:
            break
        if used[i] >= partners or used[j] >= partners:
            continue
        if any(abs(i - a) <= spread and abs(j - b) <= spread for a, b in taken):
            continue
        taken.append((i, j))
        used[i] += 1
        used[j] += 1
    return sorted(taken)


def proposals(images: Sequence[Path], required: bool = True, model: Path | None = None) -> list[tuple[int, int]]:
    """The possible revisits among a walk's frames (`images`, in walk order) for MASt3R to measure: `shortlist` of their
    MegaLoc `similarity`. A walk too short to hold a pair `LOOP_GAP` apart has none, and loads no model. When MegaLoc
    can't run and isn't `required`, none (logged)."""
    if len(images) <= LOOP_GAP:
        return []
    try:
        scores = similarity(images, model)
    except RuntimeError as e:
        if required:
            raise
        LOG.warning("no revisits proposed: %s", e)
        return []
    return shortlist(scores, gap=LOOP_GAP)
