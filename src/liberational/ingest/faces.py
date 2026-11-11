"""Faces in ingested photos and video, through see-actions' own head faces (`see/head_faces.py`).

Ingested media has no pose tracks, so faces are found on the whole frame (SCRFD), and each face's landmarks stand in
for the pose's head joints: the nose and eyes at its landmarks, and the ears at its box's sides at eye height, all at
the face's detection confidence, so the head crop is as wide around the face as a pose's would be. A face is
followed from one analyzed frame to the next on the same camera when its box overlaps one of the last frame's faces
(`LINK_IOU`), so a person in a video, or in photos taken a second apart, keeps one track and one name the way
see-actions' tracks do. From there `HeadFaces` does everything it does for the cameras: a look a second per track,
the match by the track's mean, unnamed groups, `face.match`, `face.embedding`, `face.signal`, and `face.visit`
records, People's thumbnails, and teaching the gallery when a track ends.

A gap of more than `GAP_S` between frames (or time running backwards: a new upload) ends every track.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from liberational.see.faces import Detector, Face
from liberational.see.head_faces import HeadFaces

NS = 1_000_000_000
LINK_IOU = 0.3  # a face continues a track when its box overlaps that track's last box this much
GAP_S = 5.0  # frames further apart than this end every track
# How often a track is looked at: just under the analyzed frames' one a second, so frames a second apart (whose
# offsets round a hair either side of it) are each looked at, where see-actions' EVERY_NS would skip some.
EVERY_NS = 900_000_000
JOINTS = 26  # halpe26, the pose layout HeadFaces reads (only the head joints, 0 to 4, are used)


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection over union of two (x1, y1, x2, y2) boxes."""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def joints(face: Face) -> np.ndarray | None:
    """Pose joints (JOINTS x 3) standing in for a face's head: nose and eyes at its landmarks, the ears at its box's
    left and right edges at eye height, all at its detection confidence. None without landmarks."""
    if face.kps is None:
        return None
    kps = np.asarray(face.kps, np.float64)
    eye_y = float(kps[0][1] + kps[1][1]) / 2
    out = np.zeros((JOINTS, 3), np.float64)
    out[0, :2], out[1, :2], out[2, :2] = kps[2], kps[0], kps[1]  # nose, left eye, right eye
    out[3, :2], out[4, :2] = (face.bbox[0], eye_y), (face.bbox[2], eye_y)  # left ear, right ear
    out[:5, 2] = face.det
    return out


@dataclass
class _Seen:
    track: int
    bbox: tuple[float, float, float, float]


class FrameFaces:
    """One camera's faces in ingested frames. `feed` each frame in time order; `flush` writes what was found."""

    def __init__(self, cam: str, detector: Detector, make: Callable[[str], HeadFaces]) -> None:
        self.cam, self.detector = cam, detector
        self.heads = make(cam)
        self.last: list[_Seen] = []
        self.last_ts: float | None = None
        self.next_track = time.time_ns() // 1_000_000  # unique across runs: tracks restart with each process
        self.looks = 0

    def feed(self, img: np.ndarray, ts: float, frame_id: str, seq: int = 0) -> int:
        """One BGR frame at `ts`: its faces, followed from the last frame, go to HeadFaces. Returns the faces it
        recorded (`face.match` records) from this frame."""
        if self.last_ts is not None and (ts < self.last_ts or ts - self.last_ts > GAP_S):
            self.gap()
        self.last_ts = ts
        faces = [f for f in self.detector.detect(img) if f.kps is not None]
        seen: list[_Seen] = []
        free = list(self.last)
        people: list[tuple[int, np.ndarray]] = []
        for face in sorted(faces, key=lambda f: -f.det):
            best = max(free, key=lambda s: iou(s.bbox, face.bbox), default=None)
            if best is not None and iou(best.bbox, face.bbox) >= LINK_IOU:
                free.remove(best)
                track = best.track
            else:
                track, self.next_track = self.next_track, self.next_track + 1
            j = joints(face)
            if j is None:
                continue
            seen.append(_Seen(track, face.bbox))
            people.append((track, j))
        for gone in free:
            self.heads.end(gone.track)
        self.last = seen
        height, width = img.shape[:2]
        before = len(self.heads.matches)

        def crop(box: tuple[int, int, int, int]) -> np.ndarray:
            x1, y1, x2, y2 = box
            return img[y1:y2, x1:x2]

        self.heads.feed(crop, width, height, int(ts * NS), frame_id, people, frame_seq=seq)
        found = len(self.heads.matches) - before
        self.looks += found
        return found

    def gap(self) -> None:
        """Every track ends (and teaches the gallery)."""
        self.heads.gap()
        self.last = []

    def flush(self, ts: float | None = None) -> None:
        """Writes what was found; visits closed by `ts` (every open one when None)."""
        self.heads.flush(None if ts is None else int(ts * NS))

    def close(self) -> None:
        self.gap()
        self.flush()


class Faces:
    """Each ingest camera's `FrameFaces`, made on first use."""

    def __init__(self, detector: Detector, make: Callable[[str], HeadFaces]) -> None:
        self.detector, self.make = detector, make
        self.cams: dict[str, FrameFaces] = {}

    def cam(self, name: str) -> FrameFaces:
        if name not in self.cams:
            self.cams[name] = FrameFaces(name, self.detector, self.make)
        return self.cams[name]

    def close(self) -> None:
        for cam in self.cams.values():
            cam.close()


def open_faces(writers: Any, state: Any) -> Faces:
    """The pack's face models and gallery, as see-actions opens them, writing to `writers`."""
    from liberational.people import face_shots, registry
    from liberational.see import faces as face_models
    from liberational.see import subjects
    from liberational.vision import faces as insightface

    insightface.app()
    detector, matcher = face_models.InsightFaceDetector(), face_models.GalleryMatcher()
    people = subjects.RegistryPeople(registry.at(state))
    shots = face_shots.Writer(state)
    return Faces(
        detector,
        lambda cam: HeadFaces(cam, detector, matcher, writers, every_ns=EVERY_NS, people=people, shots=shots),
    )
