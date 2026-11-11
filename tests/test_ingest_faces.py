"""ingest/faces.py: faces in ingested frames, followed from frame to frame and named through see-actions' head faces."""

import asyncio
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from liberational import segment_log
from liberational.ingest import analyze
from liberational.ingest import faces as ingest_faces
from liberational.see.faces import Face
from liberational.see.head_faces import HeadFaces


@pytest.fixture(autouse=True)
def _no_live_voice_gate(monkeypatch):
    monkeypatch.setattr(analyze.voice_turn, "wait_quiet", lambda max_wait_s=None: None)


def unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


DANA = unit(1)


def face_at(x: float, emb: np.ndarray = DANA) -> Face:
    """A clear, frontal 40-pixel face with its left edge at x."""
    kps = np.array([[x + 12, 60], [x + 28, 60], [x + 20, 70], [x + 14, 80], [x + 26, 80]], np.float64)
    return Face(bbox=(x, 45.0, x + 40, 95.0), det=0.9, kps=kps, emb=emb, norm=21.0)


class Detector:
    """Whole frames give the faces queued for them; a head crop gives one face centered in it."""

    def __init__(self, per_frame: list[list[Face]]) -> None:
        self.per_frame = per_frame

    def detect(self, img: np.ndarray) -> list[Face]:
        if img.shape[0] >= 200:  # a whole frame
            return self.per_frame.pop(0) if self.per_frame else []
        h, w = img.shape[:2]
        kps = np.array(
            [[w * 0.4, h * 0.4], [w * 0.6, h * 0.4], [w * 0.5, h * 0.5], [w * 0.42, h * 0.6], [w * 0.58, h * 0.6]]
        )
        return [Face(bbox=(w * 0.3, h * 0.3, w * 0.7, h * 0.7), det=0.9, kps=kps, emb=DANA, norm=21.0)]


class Matcher:
    def match(self, emb: np.ndarray) -> tuple[str | None, float]:
        sim = float(np.dot(emb, DANA) / max(float(np.linalg.norm(emb)), 1e-9))
        return ("Dana", sim) if sim > 0.45 else (None, sim)

    def unknown(self, emb: np.ndarray) -> str:
        return "Face 7"


class Writers:
    def __init__(self) -> None:
        self.rows: dict[str, list[dict[str, Any]]] = {}
        self.synced = 0

    def append(self, stream: str, cam: str, records: list[dict[str, Any]]) -> None:
        self.rows.setdefault(stream, []).extend(records)

    def position(self, stream: str, cam: str) -> int:
        return len(self.rows.get(stream, []))

    def sync(self) -> dict[str, dict[str, int]]:
        self.synced += 1
        return {}

    def rewind(self, to: dict[str, dict[str, int]]) -> None:
        for stream, cams in to.items():
            for at in cams.values():
                del self.rows.setdefault(stream, [])[at:]


def faces_for(per_frame: list[list[Face]], writers: Writers) -> ingest_faces.Faces:
    detector = Detector(per_frame)
    return ingest_faces.Faces(
        detector, lambda cam: HeadFaces(cam, detector, Matcher(), writers, sharpness=lambda *_: 1.0)
    )


FRAME = np.zeros((240, 320, 3), np.uint8)


def test_a_face_in_frames_a_second_apart_keeps_one_track_and_its_name():
    writers = Writers()
    faces = faces_for([[face_at(100)], [face_at(104)], [face_at(108)]], writers)
    cam = faces.cam("ingest-upload")
    assert [cam.feed(FRAME, 1000.0 + i, f"f{i}", i) for i in range(3)] == [1, 1, 1]
    cam.flush(1002.0)
    matches = writers.rows[segment_log.FACE_MATCHES]
    assert {m["track_id"] for m in matches} == {matches[0]["track_id"]}
    assert [m["ts_start"] for m in matches] == [1000.0, 1001.0, 1002.0]
    assert {m["name"] for m in matches} == {"Dana"} and {m["cam"] for m in matches} == {"ingest-upload"}
    assert len(writers.rows[segment_log.FACE_EMBEDDINGS]) == 3


def test_a_gap_or_a_face_elsewhere_starts_a_new_track():
    writers = Writers()
    faces = faces_for([[face_at(100)], [face_at(250)], [face_at(250)]], writers)
    cam = faces.cam("ingest-upload")
    cam.feed(FRAME, 1000.0, "a")
    cam.feed(FRAME, 1001.0, "b")  # moved far: no overlap
    cam.feed(FRAME, 1001.0 + ingest_faces.GAP_S + 1, "c")  # same place, after a gap
    cam.flush()
    tracks = [m["track_id"] for m in writers.rows[segment_log.FACE_MATCHES]]
    assert len(set(tracks)) == 3


def test_a_named_person_across_photos_makes_one_visit():
    writers = Writers()
    faces = faces_for([[face_at(100)], [face_at(102)]], writers)
    cam = faces.cam("ingest-upload")
    cam.feed(FRAME, 1000.0, "a")
    cam.feed(FRAME, 1001.0, "b")
    faces.close()
    (visit,) = writers.rows[segment_log.FACE_VISITS]
    assert (visit["name"], visit["ts_start"], visit["ts_end"], visit["samples"]) == ("Dana", 1000.0, 1001.0, 2)


@pytest.mark.parametrize("fail", [False, True])
def test_the_analyzer_records_faces_of_a_finished_pass_only(fail):
    writers = Writers()
    faces = faces_for([[face_at(100)], [face_at(104)], [face_at(108)]], writers)

    def frames(path, every):
        for i in range(3):
            if fail and i == 2:
                raise OSError("simulated decode failure")
            yield FRAME, float(i)

    a = analyze.Analyzer(faces=faces, frames=frames)
    item = SimpleNamespace(
        id=1,
        kind="video",
        stream="ingest-upload",
        name="v",
        status="done",
        original_ts=1000.0,
        added_ts=2000.0,
        outputs=[{"kind": "video", "path": "v.mp4", "chunk_id": "c1", "stream": "ingest-upload.video_v1"}],
    )
    done = asyncio.run(a.item(item))
    if fail:
        assert done.errors and writers.rows.get(segment_log.FACE_MATCHES, []) == []
    else:
        assert (done.frames, done.faces, done.errors) == (3, 3, [])
        assert [m["ts_start"] for m in writers.rows[segment_log.FACE_MATCHES]] == [1000.0, 1001.0, 1002.0]
        assert writers.synced == 1
