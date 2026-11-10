#!/usr/bin/env python3
"""Give back the owner's pet labels their examples (#3018): a `pet.reference` for each labeled track with none in the
current embedding space, embedded from the track's saved photo (`state/pet_shots`, kept 30 days).

A label is stranded when a reset dropped its track's embeddings, or when the embedding space changed (the tight crop,
#3024) and its reference no longer compares (people/pet_recognition.py's `stranded`). A saved photo is the track's
largest box cut square with context (`animal_verifier.square`), so the animal sits in its middle. The tool finds it
again with see-actions' detector (RF-DETR Small, on the CPU), cuts the photo tight to that box as the embedder now
does, and embeds it. A label without a photo, or whose photo the detector finds no animal in, is left as it is: no
example rather than a wrong one.

Run it once by hand, where the animal models are installed (DINOv3 loads on `vision.animal_models_device`; the
detector takes a few minutes on the CPU for a couple of thousand photos):

    uv run python tools/restore_pet_references.py --dry-run   # count only
    uv run python tools/restore_pet_references.py

It only adds `pet.reference` records to the knowledge store, under the pet evidence lock, and running it again finds
nothing more to do.
"""

import argparse
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from liberational import config as cfg
from liberational.people import face_shots, pet_shots
from liberational.people import pet_recognition as recognition

BATCH = 64


def photo(state: Path, track: str) -> Path | None:
    """The track's newest saved photo (the highest deletion revision), or None. A track id has no glob characters
    (people/pets.py's TRACK)."""
    pattern = re.compile(rf"{re.escape(track)}-{re.escape(face_shots.VERSION)}-r(\d+)\.jpg")
    found = [
        (int(m.group(1)), p)
        for p in face_shots.folder(state, pet_shots.KIND).glob(f"{track}-{face_shots.VERSION}-r*.jpg")
        if (m := pattern.fullmatch(p.name))
    ]
    return max(found)[1] if found else None


def central(animals: np.ndarray) -> Sequence[float] | None:
    """The photo's own animal among the detector's (N x 6: x1, y1, x2, y2 normalized, score, class): the box whose
    centre is nearest the photo's, since the photo was cut around it. None when there are none."""
    if not len(animals):
        return None
    centres = (animals[:, :2] + animals[:, 2:4]) / 2
    return animals[int(np.argmin(np.abs(centres - 0.5).sum(axis=1)))][:4].tolist()


def detector() -> Any:
    """see-actions' detector on the CPU: `people(bgr tensor)`'s `animals` are the animal boxes."""
    import torch

    from liberational.see.gpu_pose import GpuPose

    models = Path(cfg.path("paths.models"))
    pose = models / "rtmpose-m_halpe26_256x192.onnx"
    return GpuPose(models / cfg.model_path("person_detector").name, pose, torch.device("cpu"))


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="count what would be restored, and write nothing")
    parser.add_argument("--state", type=Path, help="the state directory (default: paths.state)")
    args = parser.parse_args()
    state = args.state or Path(cfg.path("paths.state"))
    stranded = recognition.stranded(state)
    with_photo = [(row, path) for row in stranded if (path := photo(state, str(row["track_id"]))) is not None]
    print(f"{len(stranded)} labels have no example; {len(with_photo)} of them have a saved photo")
    if args.dry_run or not with_photo:
        return
    import cv2
    import torch

    from liberational.vision.animal_embedder import DinoEmbedder
    from liberational.vision.animal_verifier import tight

    find, embedder = detector(), DinoEmbedder()
    crops: list[tuple[dict[str, Any], np.ndarray]] = []
    for row, path in with_photo:
        image = cv2.imread(str(path))
        if image is None:
            continue
        box = central(find.people(torch.from_numpy(image)).animals)
        if box is not None:
            crops.append((row, tight(image, box)))
    records = []
    now = time.time()
    for first in range(0, len(crops), BATCH):
        batch = crops[first : first + BATCH]
        vectors = embedder.embed_crops([crop for _, crop in batch])
        for k, ((row, _), vector) in enumerate(zip(batch, vectors, strict=True)):
            records.append(recognition.reference(vector, row, int((now + (first + k) * 1e-6) * 1e9)))
    print(f"no animal found in {len(with_photo) - len(crops)} photos; those labels keep no example")
    print(f"restored {recognition.keep_references(state, records)} pet references")


if __name__ == "__main__":
    main()
