"""Each fixed camera's pose over time (#3289, part of the spatial memory goal #2794).

A camera's intrinsics are its own and held (spatial/lens.py `Held`); its pose is a time series. The rig will be
mobile, and even a fixed camera gets nudged, so nothing here pins a pose to the house: every source of evidence about
where a camera was appends an observation, and readers ask for the pose valid at a moment.

**Observations** (`Observation`, the `camera.pose` segment stream, segment_log.py): when, which camera, which place
and map frame (`Revision.frame`) the pose is in, the 4x4 camera-to-map pose, a quality, and the source:
- `seat`: a nightly seat's registration of the camera (spatial/fuse.py `seat`), quality its fitness;
- `walk`: a match of the camera's frame with a LiDAR walk's frame whose pose is known;
- `owner`: the owner's measured spot (position only: its rotation is not evidence), derived on read from the owner
  plan's `x-camera` statements, at the time that plan was saved (`owner_spots`); never stored.
`record` appends, `read` reads a span back. `backfill` makes the observations a place's revisions already hold.

**Reading a pose** (`series`, `Series.at`): a camera's observations in one map frame, in time order, split into still
stretches where a seat or walk observation lands more than MOVE_M or turns more than MOVE_DEG from the stretch so far
(the camera moved); within a stretch the pose is smoothed: the rotation the mean of its observations' rotations, the
position the owner's spot when the owner measured it during the stretch, else the median of its observations'. A
moment before the first observation reads the first stretch, and one after a move reads the stretch it falls in.
"""

from __future__ import annotations

import bisect
import functools
import math
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from liberational import segment_log as S
from liberational.memory import places

SOURCES = ("seat", "walk", "owner")


@dataclass(frozen=True)
class Observation:
    """One observation of where a camera was (see the module docstring)."""

    t: float  # epoch seconds
    camera: str
    source: str  # one of SOURCES
    place: str
    frame: str | None  # the map frame the pose is in (`Revision.frame`)
    pose: np.ndarray  # 4x4 camera to map; an `owner` observation's rotation is not evidence
    quality: float


def record(state: Path, observations: Sequence[Observation]) -> None:
    """Append `observations` to the `camera.pose` stream (one slot per camera). Nothing to write is fine."""
    if not observations:
        return
    by_slot: dict[str, list[dict[str, Any]]] = {}
    for o in observations:
        if o.source not in SOURCES:
            raise ValueError(f"an observation's source is one of {SOURCES}, not {o.source!r}")
        row = {
            "id": str(uuid.uuid7()),
            "ts_start": float(o.t),
            "ts_end": float(o.t),
            "source": o.source,
            "place": o.place,
            "frame": o.frame,
            "pose": [float(v) for v in np.asarray(o.pose, dtype=np.float64).reshape(16)],
            "quality": float(o.quality),
        }
        by_slot.setdefault(S.slot(o.camera), []).append(row)
    writers = S.Writers(S.root(state))
    try:
        for slot, rows in by_slot.items():
            writers.append(S.CAMERA_POSES, slot, rows)
        writers.sync()
    finally:
        writers.close()


def read(state: Path, start: float, end: float, cameras: Sequence[str] | None = None) -> list[Observation]:
    """The stored observations in [start, end), of `cameras` when given, in time order."""
    slots = sorted({S.slot(c) for c in cameras}) if cameras is not None else None
    if slots == []:
        return []
    out = [
        Observation(
            float(r["ts_start"]),
            str(r["cam"]),
            str(r["source"]),
            str(r["place"]),
            None if r.get("frame") is None else str(r["frame"]),
            np.asarray(r["pose"], dtype=np.float64).reshape(4, 4),
            float(r["quality"]),
        )
        for r in S.at(state).range(S.CAMERA_POSES, start, end, slots)
    ]
    return sorted(out, key=lambda o: (o.t, o.camera))


#: A seat or walk observation this far (metres) from its stretch's seats and walks so far (their median) starts a new
#: stretch: the camera moved. The seat's own refusal bound (spatial/fuse.py `SEAT_SHIFT_M`): a camera that stayed put
#: is registered far closer than that to where it was.
MOVE_M = 0.5
#: Or turned this far (degrees) from the stretch's smoothed rotation. A night's tilt swings a few degrees on a camera
#: that stayed put (camera-b's pitch: 35.2, 32.7, and 33.6 on three runs), so a turn past that is a nudge.
MOVE_DEG = 8.0
#: Evidence sources that give a rotation and can start a stretch; `owner` gives a position only.
_POSED = ("seat", "walk")


@dataclass(frozen=True)
class Series:
    """One camera's observations in one map frame, split into still stretches (see the module docstring)."""

    camera: str
    frame: str | None
    observations: tuple[Observation, ...]

    @functools.cached_property
    def _stretches(self) -> tuple[tuple[float, ...], tuple[np.ndarray | None, ...]]:
        """Each stretch's first time and its smoothed pose, in time order."""
        found = stretches(self.observations)
        return tuple(s[0].t for s in found), tuple(smoothed(s) for s in found)

    def at(self, t: float) -> np.ndarray | None:
        """The smoothed 4x4 pose valid at `t`, or None with no observation that gives a rotation."""
        starts, found = self._stretches
        if not found:
            return None
        pose = found[max(bisect.bisect_right(starts, t) - 1, 0)]
        return None if pose is None else pose.copy()


def series(state: Path, place: str, frame: str | None, cameras: Sequence[str]) -> dict[str, Series]:
    """Each of `cameras`' series in `place`'s map frame `frame`: its stored observations in that frame and the owner's
    spots there (`owner_spots`). A camera with none is left out."""
    # Up to now, not a far future: a reader that keeps the store's hours (segment_log `keep_hours`, the dashboard)
    # walks a window that runs past now an hour at a time, and this one starts at 0. An observation is never ahead.
    by_slot = {S.slot(c): c for c in cameras}
    stored = [
        replace(o, camera=by_slot.get(o.camera, o.camera))
        for o in read(state, 0.0, time.time(), cameras)
        if o.place == place and o.frame == frame
    ]
    wanted = set(cameras)
    found: dict[str, list[Observation]] = {}
    for o in [*stored, *owner_spots(state, place, frame)]:
        if o.camera in wanted:
            found.setdefault(o.camera, []).append(o)
    return {c: Series(c, frame, tuple(sorted(obs, key=lambda o: o.t))) for c, obs in sorted(found.items())}


def owner_spots(state: Path, place: str, frame: str | None) -> list[Observation]:
    """The owner's measured camera spots as `owner` observations, when the owner plan holds drawn on map frame
    `frame` (spatial/fuse.py `measured_spots`), at the time that plan record was saved; [] otherwise."""
    from liberational.spatial import fuse  # heavy: the map fusion and everything it reads

    fused = fuse.fused_revision(state, place)
    if fused is None or fused.frame != frame:
        return []
    spots = fuse.measured_spots(state, place)
    history = places.plan_history(state, place)
    if not spots or not history:
        return []
    t = history[-1].start_ns / 1e9
    out = []
    for camera, xyz in sorted(spots.items()):
        pose = np.eye(4)
        pose[:3, 3] = xyz
        out.append(Observation(t, camera, "owner", place, frame, pose, 1.0))
    return out


def backfill(state: Path, place: str) -> list[Observation]:
    """The `seat` observations `place`'s tier 1 revisions in its current map frame already hold: each time a revision
    registered a fixed camera afresh (its pose changed from the revision before), at its registration time
    (`CameraPose.at`). For a dry run to print and `record` to store."""
    from liberational.spatial import fuse, splats  # heavy: the map fusion and everything it reads

    fused = fuse.fused_revision(state, place)
    if fused is None:
        return []
    out = []
    before: dict[str, places.CameraPose] = {}
    for rev in places.revisions(state, place):
        if rev.producer != fuse.PRODUCER or rev.frame != fused.frame:
            continue
        for camera, cp in rev.cameras.items():
            was = before.get(camera)
            if camera.startswith(splats.LOOK_PREFIX) or (was is not None and was.pose == cp.pose):
                continue
            pose = np.asarray(cp.pose, dtype=np.float64).reshape(4, 4)
            out.append(Observation(cp.at or rev.at, camera, "seat", place, rev.frame, pose, 1.0))
        before = rev.cameras
    return sorted(out, key=lambda o: (o.t, o.camera))


def _turn_deg(a: np.ndarray, b: np.ndarray) -> float:
    """The angle (degrees) of the rotation between rotations `a` and `b`."""
    cos = (float(np.trace(a.T @ b)) - 1.0) / 2.0
    return math.degrees(math.acos(min(1.0, max(-1.0, cos))))


class _Smoother:
    """A stretch's smoothed pose kept as observations join it (`smoothed`)."""

    def __init__(self) -> None:
        self.rotations = np.zeros((3, 3))
        self.positions: list[np.ndarray] = []
        self.owner: Observation | None = None

    def add(self, o: Observation) -> None:
        if o.source in _POSED:
            self.rotations += o.pose[:3, :3]
            self.positions.append(o.pose[:3, 3])
        elif self.owner is None or o.t >= self.owner.t:
            self.owner = o

    def pose(self, owned: bool = True) -> np.ndarray | None:
        """The chordal mean of the rotations, at the owner's spot (when `owned`) or the per-axis median of the
        positions."""
        if not self.positions:
            return None
        u, _, vt = np.linalg.svd(self.rotations)
        out = np.eye(4)
        out[:3, :3] = u @ np.diag([1.0, 1.0, float(np.linalg.det(u @ vt))]) @ vt
        spot = self.owner.pose[:3, 3] if owned and self.owner is not None else np.median(self.positions, axis=0)
        out[:3, 3] = spot
        return out


def stretches(observations: Sequence[Observation]) -> list[list[Observation]]:
    """`observations` (one camera, one frame, time order) split where the camera moved (see the module docstring)."""
    out: list[list[Observation]] = []
    smoother = _Smoother()
    for o in sorted(observations, key=lambda o: o.t):
        # Against the stretch's own seats and walks, not the owner's spot: a registration's offset from the owner's
        # tape is the same night after night, and comparing with the tape would read it as a move.
        now = smoother.pose(owned=False) if o.source in _POSED else None
        if not out or (now is not None and _moved(now, o.pose)):
            out.append([])
            smoother = _Smoother()
        out[-1].append(o)
        smoother.add(o)
    return out


def _moved(was: np.ndarray, pose: np.ndarray) -> bool:
    return float(np.linalg.norm(pose[:3, 3] - was[:3, 3])) > MOVE_M or _turn_deg(was[:3, :3], pose[:3, :3]) > MOVE_DEG


def smoothed(stretch: Sequence[Observation]) -> np.ndarray | None:
    """A still stretch's pose (see the module docstring); None when no observation in it gives a rotation."""
    smoother = _Smoother()
    for o in stretch:
        smoother.add(o)
    return smoother.pose()


def poses_at(found: Mapping[str, Series], camera: str, t: float) -> np.ndarray | None:
    """`camera`'s pose at `t` from `found` (`series`), or None when it has no series."""
    s = found.get(camera)
    return None if s is None else s.at(t)
