"""see-actions' analyzed frames, rebuilt from its segment rows: who was tracked where in each frame, and the
frames where nobody was.

see-actions writes one `pose.track` row per tracked person per frame and `action.coverage` spans of the frames it
analyzed (consecutive rows of a run share their boundary). A frame inside coverage with no pose row had nobody
tracked in it, so a reader can't see empty frames from the pose rows alone. This module puts them back: each
coverage row gives the frames of its pose rows, grouped by capture time, and an empty frame at its end when nobody
was tracked there, so a stretch with nobody in view still counts as looked at. Outside coverage is unknown.

`Follow` gives them live, from a saved cursor or from now; `between` reads a window of history. Zone visits
(see/zones_worker.py) and presence read these frames; neither has a model of its own.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from liberational.see.actions import COVERAGE_SCHEMA, POSE_SCHEMA
from liberational.see.actions_worker import COVERAGE, POSES

NS = 1_000_000_000
SCHEMAS = {POSES: POSE_SCHEMA, COVERAGE: COVERAGE_SCHEMA}
MAX_PENDING = 100_000  # pose rows held for their coverage row, per camera (see-actions writes both every second)


@dataclass
class Frame:
    """One analyzed frame of a camera: its tracked people's boxes (x1, y1, x2, y2 normalized, and the detector's
    score) and track ids, in the same order. No boxes: nobody was tracked in it."""

    cam: str
    ts_ns: int
    frame_id: str | None = None
    boxes: list[list[float]] = field(default_factory=list)
    track_ids: list[str] = field(default_factory=list)

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


def _poses(rows: Mapping[str, Any]) -> list[tuple[str, _Pose]]:
    boxes = rows["box"]
    return [
        (
            str(cam),
            _Pose(
                int(rows["ts_ns"][i]),
                str(rows["frame_id"][i]),
                str(int(rows["track_id"][i])),
                [float(v) for v in boxes[i]],
                int(rows["row_seq"][i]),
            ),
        )
        for i, cam in enumerate(rows["camera"])
    ]


def _spans(rows: Mapping[str, Any]) -> list[tuple[str, int, int]]:
    return [(str(cam), int(rows["ts_ns"][i]), int(rows["end_ns"][i])) for i, cam in enumerate(rows["camera"])]


def _frames(cam: str, poses: Iterable[_Pose], end_ns: int) -> list[Frame]:
    """The frames of one coverage row ending at `end_ns`, from its pose rows."""
    by_time: dict[int, Frame] = {}
    for p in sorted(poses, key=lambda p: (p.ts_ns, p.track_id)):
        f = by_time.setdefault(p.ts_ns, Frame(cam, p.ts_ns, p.frame_id))
        f.boxes.append(p.box)
        f.track_ids.append(p.track_id)
    if end_ns not in by_time:
        by_time[end_ns] = Frame(cam, end_ns)  # its last analyzed frame had nobody tracked
    return [by_time[t] for t in sorted(by_time)]


class Assembler:
    """Pose rows held until their coverage row comes, per camera; frames out in time order."""

    def __init__(self) -> None:
        self.pending: dict[str, list[_Pose]] = {}

    def poses(self, rows: Mapping[str, Any]) -> None:
        for cam, pose in _poses(rows):
            held = self.pending.setdefault(cam, [])
            held.append(pose)
            if len(held) > MAX_PENDING:
                del held[: len(held) - MAX_PENDING]  # coverage stopped coming: the oldest can't be placed

    def coverage(self, rows: Mapping[str, Any]) -> list[Frame]:
        out: list[Frame] = []
        for cam, _, end_ns in sorted(_spans(rows), key=lambda s: s[2]):
            held = self.pending.get(cam, [])
            mine = [p for p in held if p.ts_ns <= end_ns]
            self.pending[cam] = [p for p in held if p.ts_ns > end_ns]
            out.extend(_frames(cam, mine, end_ns))
        return out

    def held_from(self) -> dict[str, int]:
        """Each camera's first pose row still held: a cursor saved now resumes there, so none is lost."""
        return {cam: min(p.seq for p in held) for cam, held in self.pending.items() if held}


class Follow:
    """New analyzed frames as see-actions writes them, from `cursor` (a saved `cursor`) or, without one, from now."""

    def __init__(
        self, root: Path, cursor: Mapping[str, Mapping[str, int]] | None = None, cameras: Sequence[str] | None = None
    ) -> None:
        from continuum_py import SegmentFollowers, segment_tail

        cams = list(cameras) if cameras is not None else None
        start = (
            {s: dict(c) for s, c in cursor.items()}
            if cursor is not None
            else {s: segment_tail(root, s, cams) for s in SCHEMAS}
        )
        self._followers = SegmentFollowers(root, SCHEMAS, start, cams)
        self._assembler = Assembler()

    def poll(self, timeout_s: float = 1.0) -> list[Frame]:
        """The frames whose coverage came since the last poll, in time order per camera (blocking: call it off the
        event loop)."""
        got = self._followers.poll(timeout_s)
        # A flush writes its pose rows before its coverage row, but one poll may list them in either order.
        for stream, rows in got:
            if stream == POSES:
                self._assembler.poses(rows)
        out: list[Frame] = []
        for stream, rows in got:
            if stream == COVERAGE:
                out.extend(self._assembler.coverage(rows))
        return out

    @property
    def cursor(self) -> dict[str, dict[str, int]]:
        """Where to resume: past every frame given out, and at the first pose row still waiting for its coverage."""
        out = {s: dict(c) for s, c in self._followers.cursor.items()}
        out.setdefault(POSES, {}).update(self._assembler.held_from())
        return out


def between(root: Path, cam: str, start: float, end: float) -> list[Frame]:
    """The camera's analyzed frames in [start, end), in time order."""
    from continuum_py import read_segments

    t0, t1 = round(start * NS), round(end * NS)
    if t1 <= t0:
        return []
    assembler = Assembler()
    assembler.poses(read_segments(root, POSES, POSE_SCHEMA, t0, t1, [cam]))
    spans = read_segments(root, COVERAGE, COVERAGE_SCHEMA, t0, t1, [cam], overlapping=True)
    order = sorted(range(len(spans["camera"])), key=lambda i: int(spans["end_ns"][i]))
    frames = assembler.coverage({k: [spans[k][i] for i in order] for k in ("camera", "ts_ns", "end_ns")})
    return [f for f in frames if t0 <= f.ts_ns < t1]
