"""What is kept beside a face row so the owner can look at it: its picture and where it came from.

A face row in the gallery (`vision/faces.py`: the owner's `db.npz` and the learned `learned.npz`) is a vector and a
name. People's Training images lists a person's rows so a wrong one can be found and removed, which needs a picture
and a source for each. They are kept here, in `<paths.faces>/examples/`, one small JSON file a row and a JPEG when the
row was made from a picture: `<key>.json` and `<key>.jpg`.

A row's **key** is a hash of its vector as stored (float32), so the gallery files keep their shape, a rename (which
changes names, never vectors) keeps every row's picture, and a row with nothing here (one enrolled before pictures
were kept) is still listed and removable, shown as having no picture.

`faces.py` writes these as it adds a row and removes them as the owner removes one. Nothing here reads the gallery.
"""

import hashlib
import json
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FOLDER = "examples"
PX = 160  # the longest side of a kept picture, as a face crop is (people/face_shots.py)
QUALITY = 82
MAX_JPG = 512 * 1024  # a picture larger than this is not kept

# How a row came to be, as People says it.
GROUP = "group"  # the owner named an unnamed face group
INTRODUCED = "introduced"  # the identity worker's confirmed name for a group
CORRECTION = "correction"  # the owner said a sighting is this person
REFERENCE = "reference"  # a reference photo the owner attached
ENROLLED = "enrolled"  # enrolled from image files on the command line
OWNER = "owner"  # an owner row with nothing more known
LEARNED = "learned"  # learned from a confident track (`faces.learn`)


@dataclass(frozen=True)
class Example:
    """Where a new face row came from: how (`source`), the camera and the time it was seen when known (else the time
    is when the row was added), what it was made from (`ref`: "group:<n>", "shot:<detection>", or a track), and its
    picture as JPEG bytes."""

    source: str = OWNER
    cam: str | None = None
    at: float | None = None
    ref: str | None = None
    jpg: bytes | None = None


def key(vec: Any) -> str:
    """A face row's key: a hash of its vector as float32, the form the gallery stores."""
    return hashlib.sha1(np.ascontiguousarray(vec, dtype=np.float32).tobytes()).hexdigest()[:24]


def valid_key(value: object) -> bool:
    return isinstance(value, str) and len(value) == 24 and all(c in "0123456789abcdef" for c in value)


def _write(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def record(folder: Path, vec: Any, example: Example | None = None) -> str:
    """Keep `example` for the row whose stored vector is `vec`, under `folder` (the gallery's directory). A row
    recorded already keeps what it has (a row put back by an undo is the same row). Returns its key."""
    k = key(vec)
    home = folder / FOLDER
    meta = home / f"{k}.json"
    if meta.exists():
        return k
    example = example or Example()
    home.mkdir(parents=True, exist_ok=True)
    kept = example.jpg is not None and 0 < len(example.jpg) <= MAX_JPG
    if kept and example.jpg is not None:
        _write(home / f"{k}.jpg", example.jpg)
    body = {
        "source": example.source,
        "cam": example.cam,
        "at": float(example.at if example.at is not None else time.time()),
        "ref": example.ref,
        "added": time.time(),
    }
    _write(meta, json.dumps(body).encode())
    return k


def read(folder: Path, k: str) -> dict[str, Any] | None:
    """What was recorded for row `k`, or None for a row from before these were kept (or a file that can't be read)."""
    try:
        body = json.loads((folder / FOLDER / f"{k}.json").read_text())
    except OSError, ValueError:
        return None
    return body if isinstance(body, dict) else None


def picture(folder: Path, k: str) -> Path | None:
    """Row `k`'s kept picture, or None."""
    path = folder / FOLDER / f"{k}.jpg"
    return path if valid_key(k) and path.is_file() else None


def forget(folder: Path, keys: Iterable[str]) -> int:
    """Remove what is kept for these rows. How many had something."""
    gone = 0
    for k in keys:
        if not valid_key(k):
            continue
        had = False
        for suffix in (".json", ".jpg"):
            path = folder / FOLDER / f"{k}{suffix}"
            had = path.exists() or had
            path.unlink(missing_ok=True)
        gone += had
    return gone


def encode(image: Any, box: Any = None) -> bytes | None:
    """A BGR image (cut to `box` = x1, y1, x2, y2 with a little of what's around it, when given) as a JPEG of at most
    PX on its longest side, or None when it can't be made."""
    import cv2

    try:
        patch = np.asarray(image)
        if box is not None:
            height, width = patch.shape[:2]
            x1, y1, x2, y2 = (float(v) for v in box)
            pad = 0.25 * max(x2 - x1, y2 - y1)
            left, top = max(0, int(x1 - pad)), max(0, int(y1 - pad))
            right, bottom = min(width, int(x2 + pad)), min(height, int(y2 + pad))
            patch = patch[top:bottom, left:right]
        if patch.size == 0:
            return None
        scale = PX / max(patch.shape[:2])
        if scale < 1:
            size = (max(1, round(patch.shape[1] * scale)), max(1, round(patch.shape[0] * scale)))
            patch = cv2.resize(patch, size, interpolation=cv2.INTER_AREA)
        ok, out = cv2.imencode(".jpg", patch, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
        return out.tobytes() if ok else None
    except Exception:
        return None
