"""Place recognition between frames: how much two views look like the same place (part of spatial memory).

MegaLoc (gberton/MegaLoc, MIT, CVPR 2025 Workshops) is a place-recognition model: one 8448-number descriptor per image,
whose cosine similarity says how much two views look like the same place. The pinned ONNX export
(`[models.files.megaloc]`, fetched by `mise run models`) runs here through onnxruntime on the CPU, so it takes no GPU
memory from MASt3R or the live services; nothing from MegaLoc's repository runs. Each frame is resized to
322 x 322 and ImageNet-normalized, as its authors evaluate it. It takes about 0.35 s a frame on 4 CPU threads.

`similarity` gives every pair's score, for a caller to pick the pairs worth measuring with MASt3R.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from liberational import config as cfg

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
