"""Freeze aligned face crops, then score the same small pair set with old and new recognizers.

Run only during the operator-approved model comparison window, through tools/test-slot.sh. This tool never
reads the runtime gallery or writes enrollment/state. Copy a few examples to scratch first; visually review
pairs as same/different/uncertain. Existing identity tags are selection hints, not ground truth.

Manifest JSON: {"images": {"a1": "a1.jpg", "a2": "a2.jpg", "b1": "b1.jpg"},
"pairs": [{"a": "a1", "b": "a2", "relation": "same"}, {"a": "a1", "b": "b1", "relation": "different"}]}.
Image paths are relative to the manifest. All outputs should go to private scratch, never the repository.

  python tools/compare_face_models.py align --manifest /tmp/pairs.json --model <SCRFD.onnx> \
    --provider CPUExecutionProvider --output /tmp/aligned
  python tools/compare_face_models.py score --manifest /tmp/pairs.json --model <recognizer.onnx> \
    --recognizer arcface --provider CPUExecutionProvider --aligned /tmp/aligned --output /tmp/before.json
  # Same command with --recognizer adaface and the approved AdaFace export gives after.json. Use separate
  # processes so the old and new recognition sessions are never resident together.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from liberational.vision.faces import AdaFaceRecognition
from liberational.vision.onnx_runtime import preload_gpu_libraries


def manifest(path: Path) -> tuple[dict[str, Path], list[dict[str, str]]]:
    data = json.loads(path.read_text())
    raw = data["images"]
    if not 2 <= len(raw) <= 12 or any(not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", key) for key in raw):
        raise ValueError("Use two to twelve scratch images with simple anonymous IDs.")
    images = {key: path.parent / value for key, value in raw.items()}
    pairs = data["pairs"]
    if not 1 <= len(pairs) <= 24:
        raise ValueError("Use a small, manually reviewed pair set (one to 24 pairs).")
    for pair in pairs:
        if (
            pair["a"] not in images
            or pair["b"] not in images
            or pair["relation"] not in {"same", "different", "uncertain"}
        ):
            raise ValueError("Each pair needs two listed images and a same/different/uncertain visual judgment.")
    return images, pairs


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if vector.shape != (512,) or not np.isfinite(vector).all() or not np.isfinite(norm) or norm <= 0:
        raise ValueError("Recognizer produced an invalid face vector.")
    return vector / norm


def score_pairs(vectors: dict[str, np.ndarray], pairs: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Keep the human relation as context; report cosines, without treating tags as labels or tuning a threshold."""
    return [{**pair, "cosine": float(unit(vectors[pair["a"]]) @ unit(vectors[pair["b"]]))} for pair in pairs]


def align(images: dict[str, Path], detector: Any, output: Path) -> None:
    from insightface.utils import face_align

    output.mkdir(parents=True, exist_ok=True)
    for key, path in images.items():
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read scratch image {key}.")
        boxes, landmarks = detector.detect(image, max_num=0, metric="default")
        if len(boxes) != 1 or landmarks is None:
            raise ValueError(f"Scratch image {key} needs exactly one detected face; choose another example.")
        crop = face_align.norm_crop(image, landmark=landmarks[0], image_size=112)
        if not cv2.imwrite(str(output / f"{key}.png"), crop):
            raise OSError(f"Cannot save aligned scratch image {key}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["align", "score"])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--provider",
        choices=["CPUExecutionProvider", "CUDAExecutionProvider", "CoreMLExecutionProvider"],
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aligned", type=Path)
    parser.add_argument("--recognizer", choices=["arcface", "adaface"])
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error("The model must already exist; this tool never fetches weights.")
    if args.action == "score" and (args.aligned is None or args.recognizer is None):
        parser.error("Scoring needs --aligned and --recognizer.")
    import onnxruntime as ort
    from insightface.model_zoo import get_model

    if args.provider not in ort.get_available_providers():
        parser.error("The explicitly requested provider is not installed.")
    preload_gpu_libraries(ort)
    providers = [args.provider] + ([] if args.provider == "CPUExecutionProvider" else ["CPUExecutionProvider"])
    images, pairs = manifest(args.manifest)
    if args.action == "align":
        detector = get_model(str(args.model), providers=providers)
        detector.prepare(
            ctx_id=-1 if args.provider == "CPUExecutionProvider" else 0, input_size=(640, 640), det_thresh=0.5
        )
        align(images, detector, args.output)
        return
    if args.recognizer == "adaface":
        recognition = AdaFaceRecognition(args.model, providers)
        embed = recognition.embed
        used = recognition.session.get_providers()
    else:
        recognition = get_model(str(args.model), providers=providers)
        recognition.prepare(ctx_id=-1 if args.provider == "CPUExecutionProvider" else 0)
        embed = recognition.get_feat
        used = recognition.session.get_providers()
    vectors = {}
    for key in images:
        crop = cv2.imread(str(args.aligned / f"{key}.png"))
        if crop is None or crop.shape != (112, 112, 3):
            raise ValueError(f"Missing frozen 112px crop for {key}.")
        vectors[key] = unit(embed(crop))
    args.output.write_text(
        json.dumps({"model": args.model.name, "providers": used, "pairs": score_pairs(vectors, pairs)}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
