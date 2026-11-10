"""Inspect a few local animal/non-animal crops, then compare a coordinated SigLIP model run with a saved baseline.

  PYTHONPATH=src uv run python tools/probe_animal_verifier.py /tmp/crops/manifest.json
  PYTHONPATH=src uv run python tools/probe_animal_verifier.py /tmp/crops/manifest.json --run \
      --device cuda:1 --repo <installed-checkpoint> --baseline /tmp/clip.json --output /tmp/siglip.json

A manifest is a JSON list of {path, species, observation}; paths are relative to the manifest. Observations are
manual notes, never trusted pet tags or a ground-truth accuracy claim. Without --run this validates and describes
inputs without importing torch or loading weights. --run needs the coordinator's model-load window on a shared
machine. Load one model at a time, retain the same crop hashes for paired examples, and keep private crops/reports
outside the repository. Reported torch allocation excludes driver/context memory; it is not total GPU residency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil

from liberational.vision import animal_verifier as verifier


def samples(manifest: Path) -> list[dict[str, Any]]:
    """Validate local image paths and candidate species; do not infer observations from labels."""
    rows = json.loads(manifest.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("the crop manifest must be a nonempty list")
    out = []
    for row in rows:
        if not isinstance(row, dict) or row.get("species") not in verifier.SPECIES:
            raise ValueError("each crop needs a supported candidate species")
        path = manifest.parent / str(row.get("path", ""))
        image = cv2.imread(str(path))
        if image is None or image.size == 0:
            raise ValueError(f"cannot read crop {path}")
        out.append(
            {
                "path": str(path),
                "species": row["species"],
                "observation": str(row.get("observation", "")),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size": [int(image.shape[1]), int(image.shape[0])],
            }
        )
    return out


def run(rows: list[dict[str, Any]], repo: str | None, device: str, minimum: float) -> dict[str, Any]:
    """One sequential pass through unique crops; this is a small paired observation, not a load benchmark."""
    import torch

    torch.set_num_threads(1)
    cuda = device.startswith("cuda")
    if cuda:
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model = verifier.SiglipVerifier(repo, device, min_score=minimum)
    loaded_s = time.perf_counter() - started
    cache: dict[str, tuple[np.ndarray, float]] = {}
    output = []
    for row in rows:
        key = row["sha256"]
        if key not in cache:
            image = cv2.imread(row["path"])
            if image is None or image.size == 0:
                raise ValueError(f"cannot read crop {row['path']}")
            started = time.perf_counter()
            scores = model.probabilities(image)
            cache[key] = scores, (time.perf_counter() - started) * 1000
        scores, elapsed = cache[key]
        index = verifier.SPECIES.index(row["species"])
        rival = max((i for i in range(len(scores)) if i != index), key=lambda i: float(scores[i]))
        output.append(
            {
                **row,
                "score": float(scores[index]),
                "passes": verifier.passes(scores, row["species"], minimum),
                "competitor": verifier.LABELS[rival],
                "competitor_score": float(scores[rival]),
                "scores": {label: float(score) for label, score in zip(verifier.LABELS, scores, strict=True)},
                "crop_ms": elapsed,
            }
        )
    return {
        "repo": model.repo,
        "device": device,
        "score_kind": "independent-sigmoid",
        "precision": "fp16" if model.half else "fp32",
        "min_score": minimum,
        "load_s": loaded_s,
        "unique_crops": len(cache),
        "median_crop_ms": float(np.median([elapsed for _, elapsed in cache.values()])),
        "rss_mib": psutil.Process().memory_info().rss / 1024**2,
        "torch_allocated_mib": torch.cuda.memory_allocated(device) / 1024**2 if cuda else None,
        "torch_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2 if cuda else None,
        "rows": output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--run", action="store_true", help="load the model only in an approved coordinator window")
    parser.add_argument("--repo", help="an installed checkpoint; otherwise the configured animal verifier")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-score", type=float, default=verifier.MIN_SCORE)
    parser.add_argument("--baseline", type=Path, help="earlier JSON with rows keyed by crop sha256 and species")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = samples(args.manifest)
    report = (
        run(rows, args.repo, args.device, args.min_score)
        if args.run
        else {"model_loaded": False, "unique_crops": len({row["sha256"] for row in rows}), "rows": rows}
    )
    if args.baseline:
        old = json.loads(args.baseline.read_text())
        by_crop = {(r["sha256"], r["species"]): r for r in old["rows"]}
        for row in report["rows"]:
            previous = by_crop.get((row["sha256"], row["species"]))
            if previous is not None:
                row["baseline"] = {k: previous.get(k) for k in ("score", "passes", "crop_ms")}
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
