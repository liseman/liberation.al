"""Notable objects with boxes, from the caption model's grounding ([vision] objects).

Qwen3-VL can point at things: asked for objects as JSON it answers [{"label", "bbox_2d": [x1, y1, x2, y2]}] with
coordinates on a 0 to 1000 scale relative to the image it was given (checked against live frames: boxes drawn on
that scale land on the TV, couches, and guitars; read as pixels they don't). That makes one answer usable at any
size, so it's stored relative and converted to tile pixels for scene.json (the same space as faces) and to frame
pixels for the live stream.

`clean` turns the reply's rows into the objects worth storing, `to_pixels` converts a box onto an image, and
`draw` paints the boxes. The caption path
(see/captions.py) asks after the view changes, at most every MIN_S and at least every MAX_IDLE_S, and keeps the
last boxes for TTL_S. Grounding is a separate, slower call from the caption: about 260 output tokens, about 2.6 s
on an RTX 3090 with qwen3-vl:4b through Ollama, against about 0.9 s for a caption, and Ollama answers one request
at a time. Only the Ollama backend grounds; the MLX backends leave objects empty (logged once).
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, TypedDict

import cv2
import numpy as np

MIN_S = 30.0  # at most one grounding call per view this often (each one holds up a caption)
MAX_IDLE_S = 300.0  # and at least this often while the view is steady
TTL_S = 900.0  # boxes older than this are dropped rather than carried forward
MAX_OBJECTS = 12  # drawn per frame
MIN_SIDE = 12  # boxes (on the 0 to 1000 scale) narrower or shorter than this are noise
MAX_AREA = 0.9  # a box covering this much of the image is the whole scene, not an object
MAX_PER_LABEL = 2  # the 4B model can list a wall of guitars and nothing else; keep the variety
# Output tokens. qwen3-vl:4b spends a token per digit, so the 8 objects PROMPT asks for take 254 to 258 tokens
# (measured on cam0); 240 cut every answer off mid-array, and a cut-off array is no observation.
# Left to itself the model sometimes runs on past 8 (3 of 8 replies), so Ollama is also given SCHEMA, which caps the
# array at 8 (10 of 10 replies then stopped on their own at 256 to 260 tokens). A reply still cut off is truncated.
NUM_PREDICT = 400
MAX_GROUNDED = 8  # PROMPT's "up to 8"
# Ollama's structured output: a JSON array of at most MAX_GROUNDED {label, bbox_2d} objects.
SCHEMA = {
    "type": "array",
    "maxItems": MAX_GROUNDED,
    "items": {
        "type": "object",
        "required": ["label", "bbox_2d"],
        "properties": {
            "label": {"type": "string"},
            "bbox_2d": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "integer"}},
        },
    },
}


class GroundingTruncated(ValueError):
    """The model stopped at NUM_PREDICT mid-answer: the list is incomplete, so it says nothing about what's gone."""

    reason = "grounding_truncated"


# Labels for a person. Regular plurals ("girls") are matched by stripping the "s"; irregular ones are listed.
PEOPLE = {
    *("person", "people", "man", "woman", "boy", "girl", "child", "kid", "face", "head", "human", "baby"),
    *("men", "women", "children", "babies"),
}
PROMPT = (
    "Find the notable objects in this image: up to 8, each a different kind of thing, the most prominent first. "
    "Not people. Reply with JSON only, one object per line, short lowercase labels: "
    '[{"label": "...", "bbox_2d": [x1, y1, x2, y2]}]'
)


class Item(TypedDict):
    """One grounded object: a short lowercase label and its box [x1, y1, x2, y2] on the 0 to 1000 scale."""

    label: str
    box: list[float]


def clean(rows: Iterable[Mapping[str, Any]]) -> list[Item]:
    """The model's grounding rows -> [{"label", "box": [x1, y1, x2, y2] on 0 to 1000}], one per object worth
    keeping: labels lowercased and shortened, swapped corners put in order, boxes off the 0 to 1000 scale dropped
    and the rest clamped, and malformed, tiny, whole-image, and repeated boxes dropped, with at most MAX_PER_LABEL a
    label. `see/captions.grounded_items` has already validated the whole reply (a real label, a finite box), so this
    is the filtering that decides what is worth storing; it reads rows, never the text they came from."""
    items: list[Item] = []
    for d in rows:
        label, box = str(d.get("label", "")).strip().lower(), d.get("bbox_2d", d.get("bbox"))
        if not label or not isinstance(box, list) or len(box) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in box)
        except TypeError, ValueError:
            continue
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x1 < -5 or y1 < -5 or x2 > 1005 or y2 > 1005:  # not on the 0 to 1000 scale: the model drifted
            continue
        box = [max(0.0, min(1000.0, v)) for v in (x1, y1, x2, y2)]
        w, h = box[2] - box[0], box[3] - box[1]
        if w < MIN_SIDE or h < MIN_SIDE or w * h > MAX_AREA * 1e6:
            continue
        same = [it for it in items if it["label"] == label[:40]]
        if len(same) >= MAX_PER_LABEL or any(iou(it["box"], box) > 0.9 for it in same):
            continue
        items.append({"label": label[:40], "box": [round(v, 1) for v in box]})
    return items


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def to_pixels(
    items: Iterable[Mapping[str, Any]], shape: Sequence[int], ts: float | None = None, norm: bool | None = None
) -> list[dict[str, Any]]:
    """Relative boxes -> [{"label", "bbox": [x1, y1, x2, y2] in pixels of an image of shape (h, w), "ts"}].
    norm=True adds "bbox_norm", the box as fractions 0 to 1 of the image the model saw (for plain views that's the
    whole capture frame, so recordings line up with it); norm=False adds "bbox_norm": None (a warped tile)."""
    h, w = shape[:2]
    return [
        {
            "label": it["label"],
            "bbox": [
                round(it["box"][0] * w / 1000),
                round(it["box"][1] * h / 1000),
                round(it["box"][2] * w / 1000),
                round(it["box"][3] * h / 1000),
            ],
            **({"ts": ts} if ts is not None else {}),
            **({"bbox_norm": [round(v / 1000, 4) for v in it["box"]] if norm else None} if norm is not None else {}),
        }
        for it in items
    ]


OBJECT_BGR = (225, 205, 175)  # a pale steel blue: calmer than the face boxes (green known, amber unknown)


def draw(img: np.ndarray, objs: Iterable[Mapping[str, Any]]) -> np.ndarray:
    """Draw labelled object boxes (pixels of img) in place, the largest first so small labels stay on top."""
    h = img.shape[0]
    scale = max(0.4, h / 1600)  # about 0.45 at 720p
    thick = 1 if h < 900 else 2
    for o in sorted(objs, key=lambda o: -(o["bbox"][2] - o["bbox"][0]) * (o["bbox"][3] - o["bbox"][1]))[:MAX_OBJECTS]:
        x1, y1, x2, y2 = (int(v) for v in o["bbox"])
        cv2.rectangle(img, (x1, y1), (x2, y2), OBJECT_BGR, thick, cv2.LINE_AA)
        (tw, th), base = cv2.getTextSize(o["label"], cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        ty = y1 - 4 if y1 - th - 6 >= 0 else y1 + th + 4  # above the box, or inside its top edge at the frame top
        cv2.rectangle(img, (x1, ty - th - 3), (x1 + tw + 6, ty + base - 1), (40, 34, 28), -1)
        cv2.putText(img, o["label"], (x1 + 3, ty - 1), cv2.FONT_HERSHEY_SIMPLEX, scale, OBJECT_BGR, 1, cv2.LINE_AA)
    return img
