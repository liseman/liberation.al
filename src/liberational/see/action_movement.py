"""Movement and walk-bys from see-actions' tracks: `movement.track`, `movement.coverage`, `walkby.span`, and
`walkby.coverage`, in the segment shapes their readers already use.

see-actions follows people at full frame rate. About every SAMPLE_NS, `Movement.sample` hands one frame of its
counted tracks (and the camera's shift since the last sample) to the existing rules, used as libraries:

- see/tracks.py's `MovementPipeline` for movement fragments and their coverage,
- the crossing line check (`vision/crossings.crossings_of`, run on a track's path when it ends), and
- see/walk_bys.py's `WalkByPipeline`, which turns crossings into walk-bys once a chain ends.

Camera motion comes from the backend's shift estimate (a phase correlation on the GPU): a sample whose shift is
past the movement rules' `motion_shift` marks the camera moving, and a walk-by during it says so (`camera_moved`),
as walk-bys from a moving backpack rig aren't counted. A gap or a stop flushes everything open: a track then ends
censored, and coverage breaks, so nothing bridges frames that were never seen.

Track IDs are see-actions' own (unique across restarts), as strings.
"""

import uuid
from collections import deque
from collections.abc import Callable
from typing import Any

from liberational import segment_log
from liberational.see import tracks as movement_rules
from liberational.see import walk_bys
from liberational.vision import crossings

SAMPLE_NS = 200_000_000  # a sample about every 0.2 s: the movement rules sample at 1 s, walk-bys need a few a second
MOTION_KEEP_S = 300.0  # how long camera shifts are kept to judge a walk-by's camera motion
MAX_PATH = 3000  # points a track keeps for its crossing check: 10 min at one per SAMPLE_NS

Write = Callable[[str, list[dict[str, Any]]], None]


class Movement:
    """One camera's movement and walk-bys. `write(stream, records)` gets their records."""

    def __init__(self, cam: str, write: Write) -> None:
        self.cam, self.write = cam, write
        self.rules = movement_rules.MovementPipeline(cam)
        self.walks = walk_bys.WalkByPipeline(cam)
        self.crossing = crossings.settings()
        self.directions = crossings.direction_names(self.walks.line)
        self.epoch = str(uuid.uuid7())
        self._last_ns: int | None = None
        self._moved: deque[tuple[float, bool]] = deque()  # (sample time, camera moved since the last sample)
        self._cover: list[float] | None = None  # walk-by coverage being built: [start, end, frames]

    def due(self, capture_ns: int) -> bool:
        return self._last_ns is None or capture_ns - self._last_ns >= SAMPLE_NS

    def sample(
        self,
        frame_id: str,
        capture_ns: int,
        people: list[tuple[int, Any]],
        shift: tuple[float, float] | None,
    ) -> None:
        """One analyzed frame: its counted tracks as (track ID, box: x1, y1, x2, y2 normalized and the score), and
        the camera's shift (x and y as fractions of the frame) since the last sample, or None when unknown."""
        ts = capture_ns / 1e9
        records: list[tuple[str, dict[str, Any]]] = []
        if shift is not None and self._last_ns is not None:
            dx, dy = shift
            motion = {
                "id": f"{frame_id}:motion",
                "ts_start": ts,
                "cam": self.cam,
                "motion": {
                    "status": "measured",
                    "transform": [1.0, 0.0, float(dx), 0.0, 1.0, float(dy)],
                    "lost": False,
                    "reference_ts": self._last_ns / 1e9,
                    "current_ts": ts,
                    "epoch": self.epoch,
                },
            }
            records += self._outputs(self.rules.feed_motion(motion))
            moved = max(abs(dx), abs(dy)) > float(self.rules.conf["motion_shift"])
            self._moved.append((ts, moved))
            while self._moved and self._moved[0][0] < ts - MOTION_KEEP_S:
                self._moved.popleft()
        body = {
            "id": frame_id,
            "ts_start": ts,
            "cam": self.cam,
            "boxes": [[float(v) for v in box[:5]] for _, box in people],
            "track_ids": [str(track_id) for track_id, _ in people],
        }
        records += self._outputs(self.rules.feed_body(body))
        self._walks(self.walks.feed_body(body))
        self._last_ns = capture_ns
        if self._cover is None:
            self._cover = [ts, ts, 0]
        self._cover[1], self._cover[2] = ts, self._cover[2] + 1
        self._commit(records)

    def ended(self, track: crossings.Track) -> None:
        """A track ended: its line crossings, from its whole path, go to the walk-by rules."""
        for ts, direction in crossings.crossings_of(
            track,
            self.walks.line,
            float(self.crossing["hysteresis"]),
            int(self.crossing["min_track_frames"]),
            float(self.crossing["min_track_seconds"]),
        ):
            event = {
                "id": str(uuid.uuid7()),
                "ts_start": ts,
                "direction": self.directions[direction],
                "track_id": str(track.id),
            }
            self._walks(self.walks.feed_crossing(event))

    def gap(self) -> None:
        """Frames are missing, or it's stopping: flush every open track, walk, and coverage."""
        self._commit(self._outputs(self.rules.flush("gap")))
        self._walks(self.walks.flush())
        self.flush_coverage(end=True)
        self.rules = movement_rules.MovementPipeline(self.cam)
        self.epoch = str(uuid.uuid7())
        self._last_ns = None

    def flush_coverage(self, *, end: bool = False) -> None:
        """Write the walk-by coverage since the last row; `end` breaks the run, as a gap does."""
        cover = self._cover
        if cover is not None and cover[2] and cover[1] > cover[0]:
            row = {"id": str(uuid.uuid7()), "ts_start": cover[0], "ts_end": cover[1], "cam": self.cam}
            self.write(segment_log.WALKBY_COVERAGE, [{**row, "frames": int(cover[2])}])
            self._cover = [cover[1], cover[1], 0]
        if end:
            self._cover = None

    def _outputs(self, outputs: list[movement_rules.Output]) -> list[tuple[str, dict[str, Any]]]:
        out = []
        for output in outputs:
            record = dict(output.record)
            record.setdefault("id", str(uuid.uuid7()))
            stream = segment_log.MOVEMENT_TRACKS if output.kind == "track" else segment_log.MOVEMENT_COVERAGE
            out.append((stream, record))
        return out

    def _commit(self, records: list[tuple[str, dict[str, Any]]]) -> None:
        by_stream: dict[str, list[dict[str, Any]]] = {}
        for stream, record in records:
            by_stream.setdefault(stream, []).append(record)
        for stream, rows in by_stream.items():
            self.write(stream, rows)

    def _walks(self, walks: list[dict[str, Any]]) -> None:
        spans = []
        for walk in walks:
            ts = float(walk["ts"])
            first = min([ts, *(s[0] for s in self._moved)]) if self._moved else ts
            near = [moved for at, moved in self._moved if abs(at - ts) <= 2.0]
            unknown = not near or first > ts
            spans.append(
                {
                    "id": str(uuid.uuid7()),
                    "ts_start": ts,
                    "ts_end": ts,
                    "cam": self.cam,
                    "track_id": walk["tracks"][0],
                    "kind": "crossing",
                    "attributes": {
                        "direction": walk["direction"],
                        "line": self.walks.line,
                        "crossings": walk["crossings"],
                        "tracks": len(walk["tracks"]),
                        "followed": walk["verified"],
                    },
                    "uncertainty": {"camera_motion": "unknown"} if unknown else {},
                    "camera_moved": unknown or any(near),
                    "candidate": False,
                    "interpretation": {"producer": "see-actions", "version": walk_bys.VERSION},
                }
            )
        if spans:
            self.write(segment_log.WALKBY_SPANS, spans)
