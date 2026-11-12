"""see-actions' heart rate hook (see/hearts.py): a colour trace of each tracked face, and the rows it writes."""

import math
from typing import Any

import numpy as np
import pytest

from liberational import segment_log, vitals
from liberational.see import hearts

FPS = 30.0
FRAME = (400, 300)  # width, height
EYE_GAP = 61.0  # px between the pupils: a face vitals.FACE_SIDE times that, 140 px across


class Writers:
    """A `segment_log.Writers` that keeps what was appended."""

    def __init__(self) -> None:
        self.appends: list[tuple[str, str, list[dict[str, Any]]]] = []

    def append(self, stream: str, cam: str, records: list[dict[str, Any]]) -> None:
        self.appends.append((stream, cam, list(records)))


def face_joints(eye_gap: float = EYE_GAP, centre: tuple[float, float] = (200.0, 100.0)) -> np.ndarray:
    """A pose's joints with the nose and both eyes placed (the three `vitals.face_regions` needs)."""
    joints = np.zeros((26, 3), np.float64)
    x, y = centre
    joints[0] = (x, y + 30.0, 0.9)  # nose
    joints[1] = (x - eye_gap / 2, y, 0.9)  # left eye
    joints[2] = (x + eye_gap / 2, y, 0.9)  # right eye
    return joints


def skin(level: tuple[float, float, float], pulse: float, size: tuple[int, int]) -> np.ndarray:
    """A patch of skin as `Backend.crop` gives it, BGR: `level` is R, G, B, and `pulse` the swell a heartbeat makes
    in it (green most of all)."""
    height, width = size
    out = np.empty((height, width, 3), np.float32)
    r, g, b = level
    out[:, :, 0] = b + 0.2 * pulse
    out[:, :, 1] = g + pulse
    out[:, :, 2] = r + 0.3 * pulse
    return out


def feed(
    subject: hearts.Hearts,
    seconds: float = 8.0,
    *,
    level: tuple[float, float, float] = (150.0, 110.0, 85.0),
    bpm: float = 72.0,
    eye_gap: float = EYE_GAP,
    tracks: tuple[int, ...] = (7,),
    from_ns: int = 1_700_000_000_000_000_000,
) -> int:
    """Feed `seconds` of frames at 30 fps, a beating face on each track, and return the last capture time."""
    frames = int(seconds * FPS)
    width, height = FRAME
    for i in range(frames):
        ns = from_ns + round(i * 1e9 / FPS)
        swell = 0.4 * math.sin(2 * math.pi * (bpm / 60.0) * (i / FPS))
        subject.feed(
            ns,
            width,
            height,
            [(track, face_joints(eye_gap)) for track in tracks],
            lambda box, swell=swell: skin(level, swell, (box[3] - box[1], box[2] - box[0])),
        )
        subject.due(ns)
    return from_ns + round(frames * 1e9 / FPS)


def measured(
    *,
    who: Any = None,
    level: tuple[float, float, float] = (150.0, 110.0, 85.0),
    eye_gap: float = EYE_GAP,
    bpm: float = 72.0,
    seconds: float = 10.0,
) -> Writers:
    """A camera's `seconds` of frames, as the writes it made."""
    writers = Writers()
    subject = hearts.Hearts("cam0", FPS, writers, who=who)
    feed(subject, seconds=seconds, level=level, eye_gap=eye_gap, bpm=bpm)
    subject.flush()
    return writers


def test_a_seated_face_is_a_confident_reading_about_a_second():
    writers = measured()
    assert [stream for stream, _cam, _rows in writers.appends] == [segment_log.VITALS_HEART_RATES]
    _stream, cam, rows = writers.appends[0]
    assert cam == "cam0"
    assert len(rows) == 2  # a reading is taken once a second: the window fills at 7.2 s, so 8.0 and 9.0 s
    row = rows[-1]
    assert row["track_id"] == 7
    assert row["method"] == "pos-v1"
    assert row["bpm"] == pytest.approx(72.0, abs=3.0)
    assert row["confidence"] > vitals.MIN_CONFIDENCE
    assert row["fps"] == pytest.approx(FPS, abs=0.5)
    assert row["face_px"] == round(vitals.FACE_SIDE * EYE_GAP)
    assert row["person"] is None


def test_a_camera_below_twenty_frames_a_second_is_never_measured():
    writers = Writers()
    subject = hearts.Hearts("patio", 10.0, writers)
    feed(subject)
    subject.flush()
    assert writers.appends == []
    assert subject.traces == {}


def test_a_head_with_no_eyes_placed_is_not_traced():
    assert measured(eye_gap=0.0).appends == []  # both eyes at one point: an edge-on head, no face to read


def test_every_reading_names_the_person_the_face_stage_matched():
    writers = measured(who=lambda track_id: (f"person-{track_id}", "Alice"))
    assert {row["person"] for _s, _c, rows in writers.appends for row in rows} == {"person-7"}


def test_a_monochrome_picture_is_stored_as_no_reading():
    rows = [row for _s, _c, batch in measured(level=(120.0, 120.0, 120.0)).appends for row in batch]
    assert rows, "an infrared night camera still gets rows, saying there is no reading"
    assert all(math.isnan(row["bpm"]) for row in rows)


def test_a_gap_starts_the_window_again():
    writers = Writers()
    subject = hearts.Hearts("cam0", FPS, writers)
    last = feed(subject, seconds=10.0)  # the window fills at 7.2 s, so readings at 8.0 and 9.0 s
    subject.gap()
    feed(subject, seconds=2.0, from_ns=last + 6_000_000_000)  # a 6 s gap, then two more seconds of frames
    subject.flush()
    rows = [row for _s, _c, batch in writers.appends for row in batch]
    assert rows, "the first window's readings are written"
    assert all(row["ts_end"] <= last / 1e9 for row in rows), "the two seconds after the gap are not a reading yet"
