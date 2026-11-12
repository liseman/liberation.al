"""Heart rate on one camera's frames, from the colour of the skin on each tracked person's face.

see-actions already decodes every frame of the close, full-rate camera and tracks every person in it, so this is
where the skin traces are taken: no extra decode, no model, and no GPU work beyond the patch averages. Per frame and
per tracked person `feed` puts three skin patches where the pose's head joints are (`vitals.face_regions`), averages
their colour, and appends it to that track's `vitals.Trace`; about once a second `due` takes each trace's reading
and appends it to the buffer `flush` writes as `vitals.heart_rate` rows.

The patches are read off the pose joints, not the face detector's landmarks: SCRFD runs about once a second
(see/head_faces.py), and a pulse needs every frame. A face the pose places with its nose and both eyes is all the
geometry this needs, and a person who turns away or steps back simply stops being sampled, which the reading's
coverage and face size report.

Only a camera see-actions analyzes at `vitals.MIN_FPS` or faster is measured; on the others `feed` does nothing, so
a room camera at 10 frames a second costs nothing at all. What the owner is shown comes only from rows whose bpm is
a number; the rest say why in their confidence.

A heart rate matters far less than the vision it is measured beside, so no call here may take a camera down: every
public method catches its own error, logs it once with its traceback, and switches the measurement off for that
camera for the rest of the run (`off` says why). The owner can also switch it off entirely with
`vision.heart_rate = false`.
"""

from __future__ import annotations

import math
import uuid
from collections import deque
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from liberational import logs, segment_log, vitals

LOG = logs.get("see-actions")
READING_EVERY_NS = 1_000_000_000  # a track's reading is taken this often, as its window slides
MOTION_STEP_NS = 300_000_000  # how far back a face's movement is measured: over 0.3 s, per-frame keypoint jitter
# isn't motion, while a head turn still is
PATH_KEPT = 32  # face centres kept per track, to measure that movement against
METHOD = "pos-v1"  # what the bpm column's maths is: POS, with CHROM as the agreement check

Box = tuple[int, int, int, int]


class Hearts:
    """One camera's heart rate: every tracked person's colour trace, and the readings they support."""

    def __init__(
        self,
        camera: str,
        fps: float,
        writers: Any,
        *,
        who: Callable[[int], tuple[str | None, str | None]] | None = None,
        every_ns: int = READING_EVERY_NS,
        window_s: float = vitals.WINDOW_S,
        min_face_px: int = vitals.MIN_FACE_PX,
    ) -> None:
        """`fps` is the camera's analysis rate (`vision.actions_camera_fps` by name, else `vision.actions_fps`):
        below `vitals.MIN_FPS` this camera is not measured. `who(track)` is the track's person as the face stage
        names it (`see.head_faces.HeadFaces.who`), their PersonId and name, or (None, None) while unknown. `writers`
        is a `segment_log.Writers`: the rows go to `vitals.heart_rate` under this camera."""
        self.camera, self.fps, self.writers, self.who = camera, fps, writers, who
        self.every_ns, self.window_s, self.min_face_px = every_ns, window_s, min_face_px
        self.measuring = fps >= vitals.MIN_FPS
        # Why the measurement is off for the rest of the run, or None while it goes on: an error that got this far,
        # or the owner's switch (never here: then there is no Hearts at all).
        self.off: str | None = None
        self.traces: dict[int, vitals.Trace] = {}
        self._centres: dict[int, deque[tuple[int, tuple[float, float]]]] = {}
        self._read_ns: dict[int, int] = {}
        self.rows: list[dict[str, Any]] = []

    def feed(
        self,
        capture_ns: int,
        width: int,
        height: int,
        people: Sequence[tuple[int, NDArray[np.floating]]],
        crop: Callable[[Box], NDArray[np.uint8]],
    ) -> None:
        """One analyzed frame: `people` is (track_id, joints in the frame's pixels) per confirmed track, and
        `crop(box)` cuts that patch of the frame out, BGR, as `vision_backend.Backend.crop` does."""
        if not self.measuring or self.off:
            return
        try:
            for track_id, joints in people:
                regions = vitals.face_regions(joints, width, height)
                if regions is None:
                    continue
                patch = crop(regions.box)
                if patch.ndim != 3 or patch.shape[0] < 1 or patch.shape[1] < 1:
                    continue
                trace = self.traces.get(track_id)
                if trace is None:
                    trace = self.traces[track_id] = vitals.Trace(self.window_s, self.min_face_px)
                trace.add(
                    capture_ns,
                    vitals.region_means(patch, regions.patches),
                    regions.side,
                    self._motion(track_id, capture_ns, regions.side, regions.centre),
                )
        except Exception:
            self._failed("sampling a face")

    def _motion(self, track_id: int, capture_ns: int, side: float, centre: tuple[float, float]) -> float:
        """How fast the face moved, in face widths a second, over about the last `MOTION_STEP_NS`."""
        path = self._centres.get(track_id)
        if path is None:
            path = self._centres[track_id] = deque(maxlen=PATH_KEPT)
        speed = 0.0
        before = capture_ns - MOTION_STEP_NS
        for ts, old in reversed(path):
            if ts <= before:
                spent = (capture_ns - ts) / 1e9
                if spent > 0:
                    speed = math.hypot(centre[0] - old[0], centre[1] - old[1]) / max(side, 1.0) / spent
                break
        path.append((capture_ns, centre))
        return speed

    def due(self, capture_ns: int) -> None:
        """About once a second, take every track's reading and keep it for the next flush.

        The wait is marked whether or not a reading came out: a track whose window can't be read yet (too short, or
        a face that turned away and left the trace going stale) would otherwise measure its spectrum on every frame,
        which is milliseconds a frame for nothing."""
        if self.off:
            return
        try:
            self._read_due(capture_ns)
        except Exception:
            self._failed("reading a window")

    def _read_due(self, capture_ns: int) -> None:
        """Every track's reading that is due at `capture_ns`, kept for the next flush."""
        for track_id in list(self.traces):
            if capture_ns - self._read_ns.get(track_id, -(1 << 62)) < self.every_ns:
                continue
            self._read_ns[track_id] = capture_ns
            trace = self.traces[track_id]
            if trace.last_ns < capture_ns - self.every_ns:  # no sample in the last second: nothing new to read
                continue
            reading = trace.reading()
            if reading is None:  # the window doesn't hold enough of a trace yet
                continue
            person, _name = self.who(track_id) if self.who is not None else (None, None)
            self.rows.append(
                {
                    "id": str(uuid.uuid7()),
                    "ts_start": self.traces[track_id].first_ns / 1e9,
                    "ts_end": self.traces[track_id].last_ns / 1e9,
                    "track_id": track_id,
                    "person": person,
                    "bpm": reading.bpm,
                    "confidence": reading.confidence,
                    "snr_db": reading.snr_db,
                    "coverage": reading.coverage,
                    "face_px": reading.face_px,
                    "fps": reading.fps,
                    "method": METHOD,
                }
            )

    def end(self, track_id: int) -> None:
        """The track ended: forget its trace, its face's path, and when it was last read."""
        if self.off:
            return
        try:
            self.traces.pop(track_id, None)
            self._centres.pop(track_id, None)
            self._read_ns.pop(track_id, None)
        except Exception:
            self._failed("ending a track")

    def gap(self) -> None:
        """Frames are missing, so every track is about to end: start the traces again."""
        if self.off:
            return
        try:
            for track_id in list(self.traces):
                self.traces[track_id].reset()
            self._centres.clear()
            self._read_ns.clear()
        except Exception:
            self._failed("starting again after a gap")

    def flush(self) -> None:
        """Append the readings since the last flush."""
        if self.off or not self.rows:
            return
        try:
            self.writers.append(segment_log.VITALS_HEART_RATES, self.camera, self.rows)
            self.rows = []
        except Exception:
            self._failed("writing the readings")

    def _failed(self, doing: str) -> None:
        """An error this camera's heart rate can't carry on after: say so once, with the traceback, and stop. The
        camera's vision carries on without it, which is the point: a heart rate is an extra, and a camera's people,
        skeletons, faces, and actions are not."""
        self.off = f"an error while {doing}"
        self.traces.clear()
        self._centres.clear()
        self._read_ns.clear()
        self.rows = []
        LOG.exception("heart rate is off for %s for the rest of this run: %s", self.camera, self.off)
