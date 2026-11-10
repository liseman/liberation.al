"""Pet thumbnails: each animal track's photo as a file, served as /pets/sightings/photo?track=.

A track's photo is its largest embedded box (`pet_recognition.sightings`' shot), cut square with a little context
(`animal_verifier.square`). The workers that embed animal boxes (see-actions' detector, `see/action_animals.py`, and
see-captions' VLM, `see/caption_animals.py`) offer each box to a `Writer`, which saves the crop whenever it is the
track's largest so far, from the frame it already has. The dashboard serves a crop of the current deletion revision
with no lookup, and cuts one from recorded video only for a track without one. The files live in `state/pet_shots/`
and follow face_shots' naming, retention, and pruning.
"""

from __future__ import annotations

import collections
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from liberational import logs
from liberational.people import face_shots

LOG = logs.get("pet-shots")
KIND = "pet_shots"
PX = 192
TRACKS = 4096  # tracks whose largest box is remembered; an older one's next box simply writes again


def path(state: Path, track: str, revision: str) -> Path:
    return face_shots.path(state, track, revision, KIND)


def others(state: Path, track: str, revision: str) -> list[Path]:
    return face_shots.others(state, track, revision, KIND)


def encode(patch: Any) -> bytes | None:
    """A square BGR patch as the PX x PX JPEG served."""
    import cv2

    if patch is None or patch.size == 0:
        return None
    resized = cv2.resize(patch, (PX, PX), interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, face_shots.QUALITY])
    return jpg.tobytes() if ok else None


def bounds(width: int, height: int, box: Any) -> tuple[int, int, int, int]:
    """The crop's pixel bounds (left, top, right, bottom) for a box in shares of the frame."""
    from liberational.vision.animal_verifier import square_bounds

    return square_bounds(height, width, box[:4])


def cut(state: Path, sighting: Any) -> bytes | None:
    """Sighting `sighting`'s photo (a `pet_recognition.Sighting` with a shot) cut from its recorded frame, or None
    when the frame is gone. The slow way (a video decode)."""
    from liberational import vision_log

    shot = sighting.shot
    evidence = vision_log.at(state).frame_evidence(sighting.cam, float(shot["ts"]))
    if evidence is None or evidence.image is None:
        return None
    frame = evidence.image
    a, b, c, d = bounds(frame.shape[1], frame.shape[0], shot["box"])
    return encode(frame[b:d, a:c])


def revision(state: Path) -> str:
    from continuum_py import segment_revision

    from liberational import segment_log

    return str(segment_revision(segment_log.root(state)))


class Writer:
    """The workers' side: `offer` each embedded box; a track's crop is (re)written when the box is its largest yet."""

    PRUNE_EVERY = 200

    def __init__(self, state: Path) -> None:
        self.state = state
        self.largest: collections.OrderedDict[str, float] = collections.OrderedDict()
        self.written = 0

    def offer(
        self, track: str, box: Any, width: int, height: int, crop: Callable[[tuple[int, int, int, int]], Any]
    ) -> None:
        area = (float(box[2]) - float(box[0])) * (float(box[3]) - float(box[1]))
        if area <= self.largest.get(track, -1.0):
            return
        self.largest[track] = area
        self.largest.move_to_end(track)
        while len(self.largest) > TRACKS:
            self.largest.popitem(last=False)
        jpg = encode(crop(bounds(width, height, box)))
        if jpg is None:
            return
        try:
            face_shots.write(self.state, track, revision(self.state), jpg, KIND)
        except OSError as e:  # a full disk costs the thumbnail (cut from video on request instead), not the worker
            LOG.warning("pet thumbnail %s not saved: %s", track, e)
            return
        self.written += 1
        if self.written % self.PRUNE_EVERY == 0:
            threading.Thread(target=face_shots.prune, args=(self.state,), kwargs={"kind": KIND}, daemon=True).start()


def backfill(state: Path, per_second: float = 8.0, now: float | None = None) -> dict[str, int]:
    """Photos for the listed animal tracks that have none, a camera at a time, oldest first, at most
    `per_second` cuts a second. Counts of what it did."""
    from liberational.people import pet_recognition

    now = time.time() if now is None else now
    rev = revision(state)
    found = [s for s in pet_recognition.recorded(state, now) if s.shot]
    done = {"tracks": len(found), "had": 0, "cut": 0, "gone": 0}
    for s in sorted(found, key=lambda s: (s.cam, s.shot["ts"])):
        if path(state, s.track_id, rev).exists() or others(state, s.track_id, rev):
            done["had"] += 1
            continue
        started = time.monotonic()
        jpg = cut(state, s)
        if jpg is None:
            done["gone"] += 1
        else:
            face_shots.write(state, s.track_id, rev, jpg, KIND)
            done["cut"] += 1
        time.sleep(max(0.0, 1 / per_second - (time.monotonic() - started)))
    face_shots.prune(state, kind=KIND)
    return done


def main() -> None:
    """`python -m liberational.people.pet_shots backfill [--per-second N]`: photos for the listed animal tracks that
    have none, once. Run it niced."""
    import argparse

    from liberational import config as cfg

    parser = argparse.ArgumentParser(prog="pet_shots")
    parser.add_argument("command", choices=["backfill"])
    parser.add_argument("--per-second", type=float, default=8.0)
    args = parser.parse_args()
    print(backfill(cfg.path("paths.state"), args.per_second), flush=True)


if __name__ == "__main__":
    main()
