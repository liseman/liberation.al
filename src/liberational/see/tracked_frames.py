"""see-actions' analyzed frames, rebuilt from its segment rows: who was tracked where in each frame, and the
frames where nobody was.

see-actions writes one `pose.track` row per tracked person per frame and `action.coverage` spans of the frames it
analyzed (consecutive rows of a run share their boundary). Confirmed animals are the same shape: one
`animal.detection` row per tracked animal per sample, and `animal.coverage` spans of the samples it looked at. A
frame inside coverage with no detection row had nobody of that kind tracked in it, so a reader can't see empty
frames from the detection rows alone. This module puts them back: each coverage row gives the frames of its
detection rows, grouped by capture time, and an empty frame at its end when nobody was tracked there, so a stretch
with nobody in view still counts as looked at. Outside coverage is unknown.

People and animals are separate kinds. A person's box is labeled "person". An animal's box is labeled with the
species on its detection (`cat`, `dog`), and its track id is the string the catalog stored (`<run>:<species>:<n>`),
which is what a later reader uses to say which pet it was.

`Follow` gives both kinds live, from a saved cursor or from now, in one read. `between` reads a window of people's
history. Zone visits (see/zones_worker.py) and presence read these frames; neither has a model of its own.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from liberational import segment_log
from liberational.see.actions import COVERAGE_SCHEMA, POSE_SCHEMA

NS = 1_000_000_000
MAX_PENDING = 100_000  # detection rows held for their coverage row, per camera (see-actions writes both every second)


@dataclass(frozen=True)
class Kind:
    """One kind of tracked subject, and the detection and coverage streams its frames are rebuilt from.

    A pose track id is an integer. An animal track id is already the catalog's string, so it is kept as written.
    `label_column` names each box (an animal's species). Without one, every box is labeled with the kind's name.
    """

    name: str
    detections: str
    coverage: str
    detection_schema: tuple[tuple[str, str], ...]
    coverage_schema: tuple[tuple[str, str], ...]
    label_column: str | None = None
    integral_track: bool = False

    def track_id(self, value: Any) -> str:
        return str(int(value)) if self.integral_track else str(value)

    def label_at(self, rows: Mapping[str, Any], index: int) -> str:
        if self.label_column is None:
            return self.name
        raw = rows[self.label_column][index]
        text = "" if raw is None else str(raw).strip()
        return text or self.name


PEOPLE = Kind(
    "person",
    segment_log.POSE_TRACKS,
    segment_log.ACTION_COVERAGE,
    tuple(POSE_SCHEMA),
    tuple(COVERAGE_SCHEMA),
    integral_track=True,
)
ANIMALS = Kind(
    "animal",
    segment_log.ANIMAL_DETECTIONS,
    segment_log.ANIMAL_COVERAGE,
    tuple(segment_log.ANIMAL_DETECTION_SCHEMA),
    tuple(segment_log.WALKBY_COVERAGE_COLUMNS),
    label_column="species",
)
KINDS = (PEOPLE, ANIMALS)


@dataclass
class Frame:
    """One analyzed frame of a camera: its tracked boxes (x1, y1, x2, y2 normalized, and the detector's score),
    track ids, and labels, in the same order. `kind` is "person" or "animal". No boxes: nobody of that kind was
    tracked in it."""

    cam: str
    ts_ns: int
    frame_id: str | None = None
    boxes: list[list[float]] = field(default_factory=list)
    track_ids: list[str] = field(default_factory=list)
    kind: str = "person"
    labels: list[str] = field(default_factory=list)

    @property
    def ts(self) -> float:
        return self.ts_ns / NS


@dataclass
class _Pose:
    ts_ns: int
    frame_id: str
    track_id: str
    box: list[float]
    seq: int
    label: str


def _subjects(rows: Mapping[str, Any], kind: Kind) -> list[tuple[str, _Pose]]:
    boxes = rows["box"]
    return [
        (
            str(cam),
            _Pose(
                int(rows["ts_ns"][i]),
                str(rows["frame_id"][i]),
                kind.track_id(rows["track_id"][i]),
                [float(v) for v in boxes[i]],
                int(rows["row_seq"][i]),
                kind.label_at(rows, i),
            ),
        )
        for i, cam in enumerate(rows["camera"])
    ]


def _spans(rows: Mapping[str, Any]) -> list[tuple[str, int, int]]:
    return [(str(cam), int(rows["ts_ns"][i]), int(rows["end_ns"][i])) for i, cam in enumerate(rows["camera"])]


def _frames(kind: Kind, cam: str, poses: Iterable[_Pose], end_ns: int) -> list[Frame]:
    """The frames of one coverage row ending at `end_ns`, from its detection rows."""
    by_time: dict[int, Frame] = {}
    for p in sorted(poses, key=lambda p: (p.ts_ns, p.track_id)):
        f = by_time.setdefault(p.ts_ns, Frame(cam, p.ts_ns, p.frame_id, kind=kind.name))
        f.boxes.append(p.box)
        f.track_ids.append(p.track_id)
        f.labels.append(p.label)
    if end_ns not in by_time:
        by_time[end_ns] = Frame(cam, end_ns, kind=kind.name)  # its last analyzed frame had nobody of this kind
    return [by_time[t] for t in sorted(by_time)]


class Assembler:
    """Detection rows held until their coverage row comes, per camera; frames out in time order."""

    def __init__(self) -> None:
        self.pending: dict[str, list[_Pose]] = {}

    def poses(self, rows: Mapping[str, Any], kind: Kind) -> None:
        for cam, pose in _subjects(rows, kind):
            held = self.pending.setdefault(cam, [])
            held.append(pose)
            if len(held) > MAX_PENDING:
                del held[: len(held) - MAX_PENDING]  # coverage stopped coming: the oldest can't be placed

    def coverage(self, rows: Mapping[str, Any], kind: Kind) -> list[Frame]:
        out: list[Frame] = []
        for cam, _, end_ns in sorted(_spans(rows), key=lambda s: s[2]):
            held = self.pending.get(cam, [])
            mine = [p for p in held if p.ts_ns <= end_ns]
            self.pending[cam] = [p for p in held if p.ts_ns > end_ns]
            out.extend(_frames(kind, cam, mine, end_ns))
        return out

    def held_from(self) -> dict[str, int]:
        """Each camera's first pose row still held: a cursor saved now resumes there, so none is lost."""
        return {cam: min(p.seq for p in held) for cam, held in self.pending.items() if held}


def _schemas() -> dict[str, list[tuple[str, str]]]:
    """Every kind's detection and coverage schema, for one follower."""
    out: dict[str, list[tuple[str, str]]] = {}
    for kind in KINDS:
        out[kind.detections] = list(kind.detection_schema)
        out[kind.coverage] = list(kind.coverage_schema)
    return out


class Follow:
    """New analyzed frames, people's and animals', as see-actions writes them, from `cursor` (a saved `cursor`) or,
    without one, from now. One poll reads every stream."""

    def __init__(
        self, root: Path, cursor: Mapping[str, Mapping[str, int]] | None = None, cameras: Sequence[str] | None = None
    ) -> None:
        from continuum_py import SegmentFollowers, segment_tail

        cams = list(cameras) if cameras is not None else None
        schemas = _schemas()
        start = (
            {s: dict(c) for s, c in cursor.items()}
            if cursor is not None
            else {s: segment_tail(root, s, cams) for s in schemas}
        )
        self._followers = SegmentFollowers(root, schemas, start, cams)
        self._assemblers = {kind.name: Assembler() for kind in KINDS}
        self._by_detection = {kind.detections: kind for kind in KINDS}
        self._by_coverage = {kind.coverage: kind for kind in KINDS}

    def poll(self, timeout_s: float = 1.0) -> list[Frame]:
        """The frames whose coverage came since the last poll, in time order per camera and kind (blocking: call it
        off the event loop)."""
        got = self._followers.poll(timeout_s)
        # A flush writes its detection rows before its coverage row, but one poll may list them in either order.
        for stream, rows in got:
            kind = self._by_detection.get(stream)
            if kind is not None:
                self._assemblers[kind.name].poses(rows, kind)
        out: list[Frame] = []
        for stream, rows in got:
            kind = self._by_coverage.get(stream)
            if kind is not None:
                out.extend(self._assemblers[kind.name].coverage(rows, kind))
        return out

    @property
    def cursor(self) -> dict[str, dict[str, int]]:
        """Where to resume: past every frame given out, and at the first detection row still waiting for its
        coverage."""
        out = {s: dict(c) for s, c in self._followers.cursor.items()}
        for kind in KINDS:
            out.setdefault(kind.detections, {}).update(self._assemblers[kind.name].held_from())
        return out


def between(root: Path, cam: str, start: float, end: float) -> list[Frame]:
    """The camera's analyzed frames of people in [start, end), in time order."""
    from continuum_py import read_segments

    t0, t1 = round(start * NS), round(end * NS)
    if t1 <= t0:
        return []
    assembler = Assembler()
    assembler.poses(read_segments(root, segment_log.POSE_TRACKS, POSE_SCHEMA, t0, t1, [cam]), PEOPLE)
    spans = read_segments(root, segment_log.ACTION_COVERAGE, COVERAGE_SCHEMA, t0, t1, [cam], overlapping=True)
    order = sorted(range(len(spans["camera"])), key=lambda i: int(spans["end_ns"][i]))
    frames = assembler.coverage({k: [spans[k][i] for i in order] for k in ("camera", "ts_ns", "end_ns")}, PEOPLE)
    return [f for f in frames if t0 <= f.ts_ns < t1]
