#!/usr/bin/env python3
"""Score the saved photos of people's older shots by how much they read as a face, for the card's photo picker.

Shots from before see-actions wrote a face.signal for each look have nothing that tells a face from the back of a
head: the landmarks call both straight on. This runs SCRFD and AdaFace (vision.faces.app(), the live models) on each
such shot's saved crop and records AdaFace's raw norm, 0 when no face is found on it, in the state directory's
`shot_norms.json` (sightings.SHOT_NORMS), which the picker rereads when it changes (sightings.norms). Shots with a
signal, and shots already scored, are skipped, so it can be run again.

The crop is padded onto a canvas three times its size first: it's cut tight around the face, and SCRFD finds few
faces that fill the whole image. Measured on one household member's 231 card shots: padded, the crop's norm follows the
live one (correlation 0.84; clear faces 18 to 22 both ways), and no face is found on the backs of their head and on
steep top-down profiles, while unpadded no face was found on 93% of them.

Usage: score_shot_crops.py NAME [NAME ...] [--limit N]
"""

import argparse
import json
import os
import pathlib
import tempfile

import cv2
import numpy as np

from liberational import config as cfg
from liberational.vision import faces as face_identity
from liberational.vision import sightings


def crop_norm(app: object, image: np.ndarray) -> float:
    """AdaFace's raw norm of the most confident face SCRFD finds on the crop, padded; 0 when it finds none."""
    h, w = image.shape[:2]
    canvas = np.zeros((h * 3, w * 3, 3), np.uint8)
    canvas[h : 2 * h, w : 2 * w] = image
    found = app.get(canvas)  # ty: ignore[unresolved-attribute]
    if not found:
        return 0.0
    best = max(found, key=lambda f: f.det_score)
    return round(float(best.get("adaface_norm") or 0.0), 3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="+", help="the people whose card shots to score")
    ap.add_argument("--limit", type=int, default=400, help="at most this many of each person's best shots")
    args = ap.parse_args()
    state = cfg.path("paths.state")
    path = state / sightings.SHOT_NORMS
    try:
        scores: dict[str, float] = json.loads(path.read_text())
    except FileNotFoundError:
        scores = {}
    app = face_identity.app()
    live = sightings.norms(state)  # the signals' norms and the file's
    for name in args.names:
        done = found = 0
        for r in sightings.ranked(name, state)[: args.limit]:
            sid = str(r["id"])
            if sid in live:  # a signal's own norm, or scored already
                continue
            files = sorted((state / "shots").glob(f"{sid}-*.jpg"))
            image = cv2.imread(str(files[0])) if files else None
            if image is None:
                continue
            scores[sid] = crop_norm(app, image)
            done += 1
            found += scores[sid] > 0
        print(json.dumps({"name": name, "scored": done, "face_found": found, "no_face": done - found}), flush=True)
    with tempfile.NamedTemporaryFile("w", dir=state, prefix=".shot_norms.", delete=False) as tmp:
        json.dump(scores, tmp)
    os.replace(tmp.name, path)
    print(json.dumps({"file": str(pathlib.Path(path)), "shots": len(scores)}))


if __name__ == "__main__":
    main()
