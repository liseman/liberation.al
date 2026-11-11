"""Measure detection and pose models on one public labelled slice, under one harness.

The point is comparison, not a leaderboard: the same pictures, the same preprocessing as the paths this system runs,
the same decoder rules, and the same metrics for every model, so a candidate's numbers are read against the model
that runs today, in the same session, on the same machine.

    .venv/bin/python tools/measure_detection.py detect --model data/models/rfdetr-small.onnx --kind rfdetr \\
        --name rfdetr-s --images ~/scratch/val2017 --ann ~/scratch/instances_val2017.json --limit 300 --seed 0
    .venv/bin/python tools/measure_detection.py pose --model data/models/rtmpose-m_halpe26_256x192.onnx \\
        --name rtmpose-m --images ~/scratch/val2017 --ann ~/scratch/person_keypoints_val2017.json --limit 150

Each command prints a JSON report and, with `--out`, writes it. Nothing is downloaded: point `--model` at a local
file, and `--images`/`--ann` at a public, labelled set already on the machine. COCO val2017 fits because it labels
every class this system uses and every person's joints.

What the numbers are, exactly:

- Detection: COCO average precision over per-category boxes, computed here rather than with pycocotools so the tool
  carries no dependency the repository does not already have. It is COCO's rule: detections ranked by score within a
  category, greedy matching at each IoU threshold, precision envelope, 101-point interpolated AP, at most
  `--max-dets` detections a picture. `ap` is over IoU 0.50 to 0.95, `ap50` and `ap75` are the two ends, and
  `ap_small`/`ap_medium`/`ap_large` ignore ground truth outside COCO's area bands, so a detection that matches an
  ignored box is neither a hit nor a false positive. `recall_at_floor` is the share of people (or animals, or
  screens) found at the score floor the running path sets, one operating point, which is the number that changes
  behaviour.
- Pose: PCK@0.1 with the ground-truth box's longer side as the normaliser (a joint is right when it lands within a
  tenth of that), the mean joint error as a share of the same box, and OKS AP with each person's crop taken from
  their ground-truth box, so the pose model is measured on its own and the detector is not in the number. Top-down:
  the live path runs it on its own detector's boxes, and on a small set the detector's misses would otherwise be
  counted three times.
- Masks: `mask_iou_matrix` and `match_masks`, for the on-demand object path, where one run is a grid of point
  prompts today and one text-prompt call is a candidate. They are here so one run and the other are scored by the
  same code.

Latency is p50 and p95 of the whole per-picture call, split into preprocessing, the model run, and postprocessing,
because the model alone hides the resize. On a machine that also runs services, milliseconds are only comparable
against another model measured in the same session; the ratios are the finding, not the absolute numbers.
`--vram` samples the GPU while a model runs and reports the peak above the pre-load baseline.

Nothing here writes to the repository or to a running install, and no picture leaves the machine.

# An editor's pyright here resolves the system interpreter, which has neither of these; ty, the repository's
# checker, resolves the project's environment and passes. The ignores keep that environment mismatch from reading as
# a missing dependency in this file.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import pathlib
import subprocess
import sys
import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np  # pyright: ignore[reportMissingImports]

# COCO categories in the official 91-slot layout, where a logit slot's index is the category id.
PERSON = 1
ANIMALS = (16, 17, 18, 19, 20, 21, 22, 23, 24, 25)  # bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
SCREENS = (72, 73)  # tv, laptop: what the screens pass watches
GROUPS = {"person": (PERSON,), "animals": ANIMALS, "screens": SCREENS}
# The dense 80-class layout most third-party COCO exports use, by dense index, as official category ids.
DENSE80 = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
)  # fmt: skip
KEYPOINTS = 17  # COCO's joints, and the first 17 of halpe26 in the same order
POSE_BATCHES = (1, 2, 4, 8)  # the fixed batches the live pose path keeps a graph for
RFDETR_CANDIDATES = 300  # the running path's global query/class top-k, and the exports' query count
AREAS = {
    "all": (-1.0, float("inf")),
    "small": (-1.0, 32.0**2),
    "medium": (32.0**2, 96.0**2),
    "large": (96.0**2, float("inf")),
}
IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * step, 2) for step in range(10))
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], np.float32)
# RTMPose's crop and normalisation, as vision/pose_model.py has them.
POSE_PADDING = 1.25
POSE_MEAN = np.array([123.675, 116.28, 103.53], np.float32)
POSE_STD = np.array([58.395, 57.12, 57.375], np.float32)
POSE_SPLIT = 2.0  # SimCC bins per pixel
# COCO's per-joint constants for OKS, in the keypoint order both models use.
OKS_CONSTANTS = np.array(
    [0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89],
    np.float64,
)
# The joints the action model and the head crop lean on: wrists, knees, ankles.
INTEREST = (7, 8, 9, 10, 13, 14, 15, 16)


# ---- metrics


def iou_xyxy(left: np.ndarray, right: np.ndarray, crowd: np.ndarray | None = None) -> np.ndarray:
    """Pairwise overlap between N x 4 and M x 4 pixel boxes (x1, y1, x2, y2).

    A crowd column measures the share of the *detection* that falls inside it, which is COCO's own convention for a
    crowd region: a small box wholly inside a crowd is a full match there and is then dropped rather than counted
    against the detector. With `crowd` unset it is ordinary intersection over union.
    """
    a = np.asarray(left, np.float64).reshape(-1, 4)[:, None, :]
    b = np.asarray(right, np.float64).reshape(-1, 4)[None, :, :]
    x1, y1 = np.maximum(a[..., 0], b[..., 0]), np.maximum(a[..., 1], b[..., 1])
    x2, y2 = np.minimum(a[..., 2], b[..., 2]), np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(a[..., 2] - a[..., 0], 0, None) * np.clip(a[..., 3] - a[..., 1], 0, None)
    area_b = np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)
    union = area_a + area_b - inter
    plain = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
    if crowd is None:
        return plain
    inside = np.where(area_a > 0, inter / np.maximum(area_a, 1e-12), 0.0)
    return np.where(np.asarray(crowd, bool)[None, :], inside, plain)


def oks(pred: np.ndarray, truth: np.ndarray, visible: np.ndarray, box: np.ndarray) -> np.ndarray:
    """COCO object keypoint similarity: (N, K, 2) joints, (N, K) visibility truth, (N, 4) boxes, one number each.

    COCO's own form: a squared joint error scaled by that joint's constant and by the person's area, so the same
    pixel error counts for more on a small person."""
    area = np.clip((box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]), 1.0, None)[:, None]
    variances = (OKS_CONSTANTS[None, :] * 2.0) ** 2
    distance = ((np.asarray(pred, np.float64) - np.asarray(truth, np.float64)) ** 2).sum(-1)
    seen = np.asarray(visible, np.float64)
    per_joint = np.exp(-distance / variances / (area * 2.0)) * seen
    counts = seen.sum(1)
    return np.where(counts > 0, per_joint.sum(1) / np.maximum(counts, 1), 0.0)


def pck(pred: np.ndarray, truth: np.ndarray, visible: np.ndarray, box: np.ndarray, fraction: float = 0.1) -> float:
    """Share of visible joints within `fraction` of the ground-truth box's longer side."""
    if not len(box):
        return float("nan")
    scale = np.maximum(box[:, 2] - box[:, 0], box[:, 3] - box[:, 1])[:, None]
    distance = np.sqrt(((np.asarray(pred, np.float64) - np.asarray(truth, np.float64)) ** 2).sum(-1))
    hit = (distance <= fraction * scale) & (np.asarray(visible) > 0)
    total = int((np.asarray(visible) > 0).sum())
    return float(hit.sum() / total) if total else float("nan")


def joint_error(pred: np.ndarray, truth: np.ndarray, visible: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Per-joint mean distance as a share of the ground-truth box's longer side; NaN where nobody's joint was seen."""
    scale = np.maximum(box[:, 2] - box[:, 0], box[:, 3] - box[:, 1])[:, None]
    distance = np.sqrt(((np.asarray(pred, np.float64) - np.asarray(truth, np.float64)) ** 2).sum(-1)) / scale
    seen = np.asarray(visible) > 0
    counts = seen.sum(0)
    return np.where(counts > 0, np.where(seen, distance, 0.0).sum(0) / np.maximum(counts, 1), np.nan)


@dataclass(frozen=True)
class Scored:
    """One picture's predictions for one category, ranked by score: their areas, each one's similarity to that
    picture's ground truth, and which ground-truth boxes are ignored (a crowd region, or outside the area band)."""

    scores: np.ndarray
    areas: np.ndarray
    similarity: np.ndarray
    ignored: np.ndarray
    crowd: np.ndarray


def average_precision(per_image: Sequence[Scored], threshold: float, band: str = "all", max_dets: int = 100) -> float:
    """COCO's AP for one category at one similarity threshold, following its own matching rule.

    Per picture: ground truth that is ignored sorts last, each detection takes the best still-free ground truth it
    reaches, and it stops looking once a real match is in hand rather than taking an ignored one with a higher
    overlap. A detection with no match is a false positive unless its own box falls outside the area band, in which
    case it is dropped. The kept detections from every picture are then ranked by score as one list, because that is
    where COCO reads the precision envelope and the 101 recall points from. A picture in which nobody carries the
    category still contributes its detections, as false positives: that is what COCO's own evaluator does.
    """
    rows: list[tuple[float, bool, bool]] = []
    positives = 0
    low, high = AREAS[band]
    for scored in per_image:
        positives += int((~scored.ignored).sum())
        order = np.argsort(-scored.scores, kind="stable")[:max_dets]
        ground = np.argsort(scored.ignored, kind="stable")
        claimed = np.zeros(len(scored.ignored), bool)
        for index in order:
            best_iou, match = threshold - 1e-10, -1
            for candidate in ground:
                if claimed[candidate] and not scored.crowd[candidate]:
                    continue
                if match > -1 and not scored.ignored[match] and scored.ignored[candidate]:
                    break
                value = scored.similarity[index, candidate]
                if value < best_iou:
                    continue
                best_iou, match = value, candidate
            if match == -1:
                outside = scored.areas[index] < low or scored.areas[index] >= high
                rows.append((float(scored.scores[index]), False, bool(outside)))
            elif scored.ignored[match]:
                rows.append((float(scored.scores[index]), False, True))
            else:
                claimed[match] = True
                rows.append((float(scored.scores[index]), True, False))
    if not rows or not positives:
        return float("nan")
    rows.sort(key=lambda row: -row[0])
    hit = np.array([row[1] for row in rows if not row[2]], bool)
    if not hit.size:
        return float("nan")
    true_positive = np.cumsum(hit)
    false_positive = np.cumsum(~hit)
    recall = true_positive / positives
    precision = true_positive / np.maximum(true_positive + false_positive, 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    points = np.linspace(0.0, 1.0, 101)
    found = np.searchsorted(recall, points, side="left")
    sampled = np.where(found < len(recall), envelope[np.minimum(found, len(recall) - 1)], 0.0)
    return float(sampled.mean())


def mean_ap(
    per_image: Sequence[Scored], thresholds: Iterable[float] = IOU_THRESHOLDS, band: str = "all"
) -> dict[str, float]:
    """AP over the thresholds, plus the two ends COCO publishes."""
    values = np.array([average_precision(per_image, threshold, band) for threshold in thresholds])
    return {
        "ap": float(np.nanmean(values)),
        "ap50": float(values[0]),
        "ap75": float(values[5]),
    }


def recall_at(per_image: Sequence[Scored], floor: float, threshold: float = 0.5) -> float:
    """Share of kept ground truth reached by a prediction scoring at or above `floor`: one operating point."""
    matched = positive = 0
    for scored in per_image:
        keep = scored.scores >= floor
        positive += int((~scored.ignored).sum())
        if keep.any() and scored.similarity[keep].size:
            reached = scored.similarity[keep].max(0) >= threshold
            matched += int((reached & ~scored.ignored).sum())
    return float(matched / positive) if positive else float("nan")


def mask_iou_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pairwise IoU between N and M binary masks of one shape."""
    a = np.asarray(left, bool).reshape(len(left), -1)
    b = np.asarray(right, bool).reshape(len(right), -1)
    inter = a.astype(np.int64) @ b.T.astype(np.int64)
    union = a.sum(1)[:, None] + b.sum(1)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0)


def match_masks(pred: np.ndarray, truth: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Recall, precision and mean best IoU of predicted masks against one picture's ground-truth masks."""
    if not len(pred) or not len(truth):
        return {
            "recall": 0.0 if len(truth) else float("nan"),
            "precision": 0.0 if len(pred) else float("nan"),
            "best_iou": float("nan"),
        }
    overlap = mask_iou_matrix(pred, truth)
    return {
        "recall": float((overlap.max(0) >= threshold).mean()),
        "precision": float((overlap.max(1) >= threshold).mean()),
        "best_iou": float(overlap.max(0).mean()),
    }


# ---- models


class Session(Protocol):
    """The part of an onnxruntime session this tool uses."""

    def run(self, output_names: Any, input_feed: dict[str, Any], /) -> Sequence[Any]: ...

    def get_inputs(self) -> Sequence[Any]: ...

    def get_outputs(self) -> Sequence[Any]: ...

    def get_providers(self) -> Sequence[str]: ...


@dataclass(frozen=True)
class Detection:
    """One picture's decoded detections: N x 4 pixel boxes, N scores, N category ids."""

    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray


def square_input(image: np.ndarray, size: int, normalize: bool = True) -> np.ndarray:
    """A BGR picture as the detector's input: the 0 to 1 RGB floats, square bilinear resize, then the COCO
    normalisation when the export wants it (the running path does; the D-FINE export does not)."""
    import cv2  # pyright: ignore[reportMissingImports]

    rgb = image[:, :, ::-1].astype(np.float32) / 255.0
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    if normalize:
        resized = (resized - IMAGE_MEAN) / IMAGE_STD
    return np.ascontiguousarray(resized.transpose(2, 0, 1)[None])


def top_k_decode(
    boxes: np.ndarray, logits: np.ndarray, labels: Sequence[int] | None = None, candidates: int = RFDETR_CANDIDATES
) -> Detection:
    """The running path's decode in numpy (`see/gpu_pose.rfdetr_candidates`): sigmoid over every query/class pair,
    a global top-k before any class filter, normalised cxcywh to xyxy. No NMS, by design: adjacent people and
    animals both stay. D-FINE's export takes the same decode with its own label layout and a different
    normalisation, which is exactly what `labels` is for."""
    flat_boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
    flat_logits = np.asarray(logits, np.float64).reshape(len(flat_boxes), -1)
    scores = 1.0 / (1.0 + np.exp(-np.clip(flat_logits, -88.0, 88.0)))
    flat = scores.reshape(-1)
    order = np.argsort(-flat, kind="stable")[: min(candidates, flat.size)]
    queries, classes = np.divmod(order, flat_logits.shape[1])
    if labels is not None:
        classes = np.asarray(labels, np.int64)[classes]
    chosen = flat_boxes[queries]
    centres, extents = chosen[:, :2], chosen[:, 2:]
    xyxy = np.concatenate((centres - extents / 2, centres + extents / 2), 1)
    valid = (xyxy[:, 2] > xyxy[:, 0]) & (xyxy[:, 3] > xyxy[:, 1])
    return Detection(np.clip(xyxy[valid], 0, 1), flat[order][valid], classes[valid].astype(np.int64))


def non_max_suppression(detection: Detection, iou: float) -> Detection:
    """Multiclass greedy NMS, for the families whose own inference applies one (D-FINE does, RF-DETR does not).
    Kept apart from the decode so a model measured both ways reports which post-processing produced its numbers."""
    if iou <= 0:
        return detection
    keep: list[int] = []
    for category in np.unique(detection.labels):
        members = np.flatnonzero(detection.labels == category)
        members = members[np.argsort(-detection.scores[members], kind="stable")]
        while members.size:
            best = int(members[0])
            keep.append(best)
            rest = members[1:]
            if not rest.size:
                break
            overlap = iou_xyxy(detection.boxes[best : best + 1], detection.boxes[rest])[0]
            members = rest[overlap < iou]
    order = np.argsort(-detection.scores[keep], kind="stable")
    chosen = np.array(keep, np.int64)[order]
    return Detection(detection.boxes[chosen], detection.scores[chosen], detection.labels[chosen])


def pose_crop(
    image: np.ndarray, box: Sequence[float], size: tuple[int, int]
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """`vision/pose_model.crop`: the box grown 1.25x, widened to the input's aspect, as a normalised RGB patch."""
    import cv2  # pyright: ignore[reportMissingImports]

    width, height = size
    x1, y1, x2, y2 = (float(value) for value in box[:4])
    w, h = (x2 - x1) * POSE_PADDING, (y2 - y1) * POSE_PADDING
    if w > h * width / height:
        h = w * height / width
    else:
        w = h * width / height
    cx, cy, scale = (x1 + x2) / 2, (y1 + y2) / 2, width / max(w, 1e-6)
    matrix = np.array([[scale, 0, width / 2 - scale * cx], [0, scale, height / 2 - scale * cy]], np.float32)
    patch = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR)
    rgb = patch[:, :, ::-1].astype(np.float32)
    return np.ascontiguousarray(((rgb - POSE_MEAN) / POSE_STD).transpose(2, 0, 1)), (cx, cy, scale)


def pose_decode(
    simcc_x: np.ndarray, simcc_y: np.ndarray, transform: tuple[float, float, float], size: tuple[int, int]
) -> np.ndarray:
    """`vision/pose_model.decode` then `to_frame`: (K, 3) joints in picture pixels with a confidence each."""
    x = np.argmax(simcc_x, axis=2).astype(np.float64) / POSE_SPLIT
    y = np.argmax(simcc_y, axis=2).astype(np.float64) / POSE_SPLIT
    confidence = np.maximum(np.minimum(simcc_x.max(2), simcc_y.max(2)), 0.0)
    cx, cy, scale = transform
    out = np.empty((len(simcc_x), int(simcc_x.shape[1]), 2), np.float64)
    out[..., 0] = (x - size[0] / 2) / scale + cx
    out[..., 1] = (y - size[1] / 2) / scale + cy
    return np.concatenate((out, confidence[..., None]), 2)


def open_session(path: pathlib.Path, device: int | None) -> tuple[Session, str, int]:
    """An ONNX Runtime session on CUDA when it is there, which is the runtime the running path uses, else the CPU.
    The provider it actually got is reported, because a silent CPU fallback makes every number meaningless."""
    import onnxruntime as ort  # pyright: ignore[reportMissingImports]

    providers: list[Any] = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        # The running path calls this before every CUDA session (`vision/onnx_runtime.preload_gpu_libraries`):
        # without it the provider's cuDNN handle is unavailable and every kernel refuses.
        with contextlib.suppress(Exception):
            ort.preload_dlls()
        options = {"device_id": str(device)} if device is not None else {}
        providers = [("CUDAExecutionProvider", options), "CPUExecutionProvider"]
    session = ort.InferenceSession(str(path), providers=providers)
    shape = session.get_inputs()[0].shape
    height = int(shape[2]) if isinstance(shape[2], int) else 0
    return session, str(session.get_providers()[0]), height


class Vram(threading.Thread):
    """Samples one GPU's used memory while a model runs, through `nvidia-smi`, so nothing else is imported."""

    def __init__(self, device: int, interval: float = 0.2) -> None:
        super().__init__(daemon=True)
        self.device, self.interval = device, interval
        self.peak = 0
        self.baseline = self.used_mib()
        self._stop = threading.Event()

    def used_mib(self) -> int:
        result = subprocess.run(
            ["nvidia-smi", "--id", str(self.device), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
        )
        text = result.stdout.strip().splitlines()
        return int(text[0]) if result.returncode == 0 and text else 0

    def run(self) -> None:
        while not self._stop.wait(self.interval):
            self.peak = max(self.peak, self.used_mib())

    def stop(self) -> int:
        self._stop.set()
        self.join(timeout=5)
        return max(self.peak - self.baseline, 0)


# ---- data


def load_coco(path: pathlib.Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return json.load(f)


def image_ids(coco: dict[str, Any], limit: int, seed: int) -> list[int]:
    """A fixed, seeded slice of the labelled set's pictures, in id order, so every model sees the same files."""
    ids = sorted(int(image["id"]) for image in coco["images"])
    if limit <= 0 or limit >= len(ids):
        return ids
    chosen = np.random.default_rng(seed).choice(len(ids), size=limit, replace=False)
    return sorted(ids[int(index)] for index in chosen)


def category_boxes(
    coco: dict[str, Any], category: int, ids: Iterable[int]
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Ground truth of one category per picture: N x 4 pixel boxes, an N-long crowd flag, and N areas.

    The area is the annotation's own (`area` in a COCO file, the segmentation's), not the box's: COCO cuts its small,
    medium, and large bands on it, and a person's outline is smaller than their bounding box, so a box area moves
    instances between bands and the numbers with them.

    COCO's crowd regions are kept rather than dropped, because that is what they are for: a detection that lands in
    one is neither a hit nor a false positive."""
    keep = set(ids)
    rows: dict[int, list[list[float]]] = {image: [] for image in keep}
    crowds: dict[int, list[bool]] = {image: [] for image in keep}
    areas: dict[int, list[float]] = {image: [] for image in keep}
    for annotation in coco["annotations"]:
        if annotation["image_id"] not in keep or annotation["category_id"] != category:
            continue
        x, y, w, h = (float(value) for value in annotation["bbox"])
        image = int(annotation["image_id"])
        rows[image].append([x, y, x + w, y + h])
        crowds[image].append(bool(annotation.get("iscrowd")))
        areas[image].append(float(annotation.get("area", w * h)))
    return {
        image: (
            np.array(rows[image], np.float64).reshape(-1, 4),
            np.array(crowds[image], bool),
            np.array(areas[image], np.float64),
        )
        for image in keep
    }


def scored(detection: Detection, truth: tuple[np.ndarray, np.ndarray, np.ndarray], category: int, band: str) -> Scored:
    """One picture's Scored row: that category's detections, their IoU to its ground truth, and the ground truth that
    is ignored either because it is a crowd region or because it falls outside the area band."""
    wanted = np.flatnonzero(detection.labels == category)
    order = np.argsort(-detection.scores[wanted], kind="stable")
    boxes, crowds, ground_areas = truth
    boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
    crowd = np.asarray(crowds, bool)
    areas = np.asarray(ground_areas, np.float64)
    low, high = AREAS[band]
    ignored = (areas < low) | (areas >= high) | crowd
    chosen = detection.boxes[wanted][order]
    sizes = np.clip(chosen[:, 2] - chosen[:, 0], 0, None) * np.clip(chosen[:, 3] - chosen[:, 1], 0, None)
    return Scored(detection.scores[wanted][order], sizes, iou_xyxy(chosen, boxes, crowd), ignored, crowd)


def percentiles(values: Sequence[float]) -> dict[str, float]:
    """p50, p95 and mean of a list of millisecond timings, NaN when nothing was timed."""
    if not values:
        return {"p50": float("nan"), "p95": float("nan"), "mean": float("nan")}
    array = np.asarray(values, np.float64)
    return {
        "p50": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "mean": float(array.mean()),
    }


def output_pair(session: Session) -> tuple[int, int]:
    """The box and logit outputs by name, never by shape: `dets`/`labels` on an RF-DETR export, `pred_boxes`/
    `logits` on a D-FINE one (the same trap the export documentation warns about, at three classes)."""
    names = [output.name for output in session.get_outputs()]
    boxes = next((i for i, name in enumerate(names) if "dets" in name or "boxes" in name), None)
    logits = next((i for i, name in enumerate(names) if "labels" in name or "logits" in name), None)
    if boxes is None or logits is None:
        raise SystemExit(f"cannot tell boxes from logits among {names}")
    return boxes, logits


# ---- the two measurements


def measure_detection(args: argparse.Namespace) -> dict[str, Any]:
    import cv2  # pyright: ignore[reportMissingImports]

    coco = load_coco(args.ann)
    ids = image_ids(coco, args.limit, args.seed)
    truth = {
        group: {category: category_boxes(coco, category, ids) for category in categories}
        for group, categories in GROUPS.items()
    }
    by_id = {int(image["id"]): image for image in coco["images"]}
    session, provider, height = open_session(args.model, args.device)
    size = args.size or height
    labels = DENSE80 if args.labels == "coco80" else None
    boxes_at, logits_at = output_pair(session)
    watcher = Vram(args.device or 0) if args.vram else None
    if watcher:
        watcher.start()

    pre, run, post, total = [], [], [], []
    rows: dict[str, dict[str, list[Scored]]] = {group: {band: [] for band in AREAS} for group in GROUPS}
    for index, image_id in enumerate(ids):
        picture = cv2.imread(str(args.images / by_id[image_id]["file_name"]))
        if picture is None:
            raise SystemExit(f"cannot read {args.images / by_id[image_id]['file_name']}")
        width, height_px = picture.shape[1], picture.shape[0]
        t0 = time.perf_counter()
        blob = square_input(picture, size, normalize=args.kind == "rfdetr")
        t1 = time.perf_counter()
        outputs = session.run(None, {session.get_inputs()[0].name: blob})
        t2 = time.perf_counter()
        detection = top_k_decode(outputs[boxes_at], outputs[logits_at], labels)
        detection = non_max_suppression(detection, args.nms)
        detection = Detection(
            detection.boxes * np.array([width, height_px, width, height_px], np.float64),
            detection.scores,
            detection.labels,
        )
        t3 = time.perf_counter()
        if index >= args.warmup:
            pre.append((t1 - t0) * 1000)
            run.append((t2 - t1) * 1000)
            post.append((t3 - t2) * 1000)
            total.append((t3 - t0) * 1000)
        for group, categories in GROUPS.items():
            for category in categories:
                for band in AREAS:
                    rows[group][band].append(scored(detection, truth[group][category][image_id], category, band))

    report: dict[str, Any] = {
        "model": str(args.model),
        "name": args.name,
        "kind": args.kind,
        "provider": provider,
        "input": size,
        "images": len(ids),
        "seed": args.seed,
        "labels": args.labels,
        "latency_ms": {
            "preprocess": percentiles(pre),
            "run": percentiles(run),
            "postprocess": percentiles(post),
            "total": percentiles(total),
        },
        "vram_peak_above_baseline_mib": watcher.stop() if watcher else None,
    }
    classes: dict[str, Any] = {}
    for group in GROUPS:
        instances = int(sum(int((~row.ignored).sum()) for row in rows[group]["all"]))
        if not instances:
            continue
        summary: dict[str, Any] = {"instances": instances, **mean_ap(rows[group]["all"])}
        for band in ("small", "medium", "large"):
            summary[f"ap_{band}"] = mean_ap(rows[group][band], band=band)["ap"]
            summary[f"{band}_instances"] = int(sum(int((~row.ignored).sum()) for row in rows[group][band]))
        summary["recall_at_floor"] = recall_at(rows[group]["all"], args.recall_floor.get(group, 0.5))
        summary["score_floor"] = args.recall_floor.get(group, 0.5)
        classes[group] = summary
    report["classes"] = classes
    return report


def people_with_joints(coco: dict[str, Any], image_id: int) -> list[dict[str, Any]]:
    """One picture's ground-truth people who carry COCO's 17 joints, crowds dropped."""
    return [
        annotation
        for annotation in coco["annotations"]
        if annotation["image_id"] == image_id
        and not annotation.get("iscrowd")
        and annotation.get("num_keypoints")
        and len(annotation["keypoints"]) >= 3 * KEYPOINTS
    ]


def pose_on_picture(
    picture: np.ndarray,
    people: Sequence[dict[str, Any]],
    session: Any,
    sessions: dict[int, Any],
    name: str,
    size: tuple[int, int],
    batch: int,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray, list[float], float]:
    """Pose for every labelled person in one picture, on their own boxes: predicted joints, their truth and
    visibility, the boxes, one timing per person (batch 1) and one for the padded live batch."""
    crops, joints, visible, boxes = [], [], [], []
    for person in people:
        x, y, w, h = (float(value) for value in person["bbox"])
        truth = np.array(person["keypoints"], np.float64).reshape(-1, 3)[:KEYPOINTS]
        boxes.append([x, y, x + w, y + h])
        joints.append(truth[:, :2])
        visible.append(truth[:, 2] > 0)
        crops.append(pose_crop(picture, boxes[-1], size))
    blob = np.stack([crop for crop, _ in crops]).astype(np.float32)
    single = []
    for index in range(len(blob)):
        start = time.perf_counter()
        session.run(None, {name: blob[index : index + 1]})
        single.append((time.perf_counter() - start) * 1000)
    # Pad to the next batch the live path keeps a graph for (1, 2, 4, 8): a dynamic-batch export fed a different
    # batch every picture re-plans its memory each time, which is not the shape that runs live.
    wanted = next(step for step in POSE_BATCHES if step >= min(len(blob), batch))
    padded = (
        blob[:wanted] if wanted <= len(blob) else np.concatenate([blob, np.repeat(blob[:1], wanted - len(blob), 0)])
    )
    start = time.perf_counter()
    outputs = sessions[wanted].run(None, {name: padded})
    elapsed = (time.perf_counter() - start) * 1000
    simcc_x, simcc_y = (np.asarray(token) for token in outputs[:2])
    count = min(len(simcc_x), len(crops))
    predicted = [
        pose_decode(simcc_x[index : index + 1], simcc_y[index : index + 1], crops[index][1], size)[0][:KEYPOINTS]
        for index in range(count)
    ]
    return (
        predicted,
        np.array(joints[:count], np.float64),
        np.array(visible[:count]),
        np.array(boxes[:count], np.float64),
        single,
        elapsed,
    )


def measure_pose(args: argparse.Namespace) -> dict[str, Any]:
    import cv2
    import onnxruntime as ort

    coco = load_coco(args.ann)
    ids = image_ids(coco, args.limit, args.seed)
    by_id = {int(image["id"]): image for image in coco["images"]}
    providers: list[Any] = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        with contextlib.suppress(Exception):
            ort.preload_dlls()
        options = {"device_id": str(args.device)} if args.device is not None else {}
        providers = [("CUDAExecutionProvider", options), "CPUExecutionProvider"]
    # One session per batch size, as the live path holds a fixed-shape graph for each of its batches
    # (see/gpu_pose POSE_BATCHES): a session that sees two shapes re-plans its memory on every swap.
    session = ort.InferenceSession(str(args.model), providers=providers)
    sessions = {step: ort.InferenceSession(str(args.model), providers=providers) for step in POSE_BATCHES}
    name = session.get_inputs()[0].name
    shape = session.get_inputs()[0].shape
    size = (int(shape[3]), int(shape[2]))
    watcher = Vram(args.device or 0) if args.vram else None
    if watcher:
        watcher.start()

    single: list[float] = []
    batched: list[float] = []
    predictions, truths, visibles, boxes_all = [], [], [], []
    for image_id in ids:
        people = people_with_joints(coco, image_id)
        if not people:
            continue
        picture = cv2.imread(str(args.images / by_id[image_id]["file_name"]))
        if picture is None:
            continue
        predicted, truth, visible, boxes, one, elapsed = pose_on_picture(
            picture, people, session, sessions, name, size, args.batch
        )
        predictions.extend(predicted)
        truths.append(truth)
        visibles.append(visible)
        boxes_all.append(boxes)
        single.extend(one)
        batched.append(elapsed)

    if not boxes_all:
        raise SystemExit("no ground-truth people with joints in the slice")
    pred, truth = np.array([one[:, :2] for one in predictions]), np.concatenate(truths)
    visible, boxes = np.concatenate(visibles), np.concatenate(boxes_all)
    rows = []
    for index in range(len(pred)):
        similarity = oks(
            pred[index : index + 1], truth[index : index + 1], visible[index : index + 1], boxes[index : index + 1]
        )
        score = float(predictions[index][:, 2].mean())  # mean joint confidence, so AP has a ranking to use
        box = boxes[index]
        area = float(np.clip(box[2] - box[0], 0, None) * np.clip(box[3] - box[1], 0, None))
        rows.append(
            Scored(np.array([score]), np.array([area]), similarity.reshape(1, 1), np.zeros(1, bool), np.zeros(1, bool))
        )
    errors = joint_error(pred, truth, visible, boxes)
    return {
        "model": str(args.model),
        "name": args.name,
        "provider": str(session.get_providers()[0]),
        "input": size,
        "people": len(pred),
        "images": len(boxes_all),
        "seed": args.seed,
        "pck@0.1": pck(pred, truth, visible, boxes),
        "oks_ap": mean_ap(rows),
        "joint_error_fraction": {
            "all": float(np.nanmean(errors)),
            **{str(joint): float(errors[joint]) for joint in INTEREST},
        },
        "latency_ms": {"per_person": percentiles(single), f"batch{args.batch}": percentiles(batched)},
        "vram_peak_above_baseline_mib": watcher.stop() if watcher else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=("Measure detection and pose models on one public labelled slice."))
    commands = parser.add_subparsers(dest="task", required=True)
    for task in ("detect", "pose"):
        command = commands.add_parser(task)
        command.add_argument("--model", type=pathlib.Path, required=True, help="the local model file")
        command.add_argument("--name", default="", help="how the report names this model")
        command.add_argument("--images", type=pathlib.Path, required=True, help="directory of pictures")
        command.add_argument("--ann", type=pathlib.Path, required=True, help="the COCO-format annotations")
        command.add_argument("--limit", type=int, default=300, help="pictures to measure (0 for all)")
        command.add_argument("--seed", type=int, default=0, help="the seed that picks the pictures")
        command.add_argument("--device", type=int, default=None, help="the CUDA device id")
        command.add_argument("--vram", action="store_true", help="sample the GPU's memory while it runs")
        command.add_argument("--out", type=pathlib.Path, default=None, help="write the report here")
        if task == "detect":
            command.add_argument(
                "--kind", choices=("rfdetr", "dfine"), default="rfdetr", help="the decode the export wants"
            )
            command.add_argument("--size", type=int, default=0, help="square input (0: the model's own)")
            command.add_argument(
                "--labels", choices=("coco91", "coco80"), default="coco91", help="the export's class layout"
            )
            command.add_argument("--warmup", type=int, default=5, help="pictures timed but not counted")
            command.add_argument(
                "--nms",
                type=float,
                default=0.0,
                help="multiclass NMS IoU after the decode (0: none, as this system runs)",
            )
            command.add_argument(
                "--recall-floor",
                type=json.loads,
                default={"person": 0.6, "animals": 0.35, "screens": 0.4},
                help="score floor per group, as the running path sets it",
            )
        else:
            command.add_argument("--batch", type=int, default=8, help="the live pose batch (up to 8 people)")
    args = parser.parse_args(argv)
    report = measure_detection(args) if args.task == "detect" else measure_pose(args)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
