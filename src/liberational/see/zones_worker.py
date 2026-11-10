"""`pack-see-zones`: zone visits and coverage, and times someone was in view, from see-actions' tracked people (#1316,
#1825).

It follows see-actions' analyzed frames (see/tracked_frames.py: `pose.track` and `action.coverage` in the segment
store) and feeds each camera's frames through see/zones.py's pipeline, which writes `zone.visit` and `zone.coverage`
rows to the segment store. A frame inside see-actions' coverage with nobody tracked still counts as looked at, so a
zone nobody visited reads as none, not unknown. The zones are the owner's saved sets (`zone.defined`, in the
knowledge store), read again every ZONES_S; a changed set ends that camera's open visits, as saving one always has.
No model runs here: a visit is a track's foot anchor (its box's bottom centre) inside a polygon. The same frames give
each camera's `presence.span` rows (memory/presence_episodes.py): times someone was in view, which presence answers
read with see-actions' coverage.

Visits are people's: see-actions' tracks are people, so animal zone counts are unknown.

The checkpoint is a small local file, `paths.state/see-zones.json`, replaced atomically every CHECKPOINT_S: the input
cursor, each output writer's next `row_seq`, and every camera's open visits and episode. Outputs are synced before the
checkpoint that covers them, and a restart reopens the writers at its positions, so frames read again after a crash
write their visits once. With no checkpoint (or one from another version) it starts from now.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import os
import signal
import time
from collections.abc import Callable, Generator, Mapping, Sequence
from pathlib import Path
from typing import Any

from liberational import config, logs, segment_log
from liberational.memory import presence_episodes
from liberational.see import tracked_frames, zones

LOG = logs.get("see-zones")
VERSION = 4  # 3: from see-actions' tracked frames, with a local checkpoint; 4: presence episodes too (#1825)
CHECKPOINT_S = 5.0
ZONES_S = 5.0
POLL_S = 1.0
# see-actions flushes each camera's rows about once a second, each at its own moment, and every read lists and reads
# every camera's open parts: reading at most once a second takes them all in one pass instead of one per flush
# (#3211). Visits are a second later at most; the checkpoint already trails by CHECKPOINT_S.
BATCH_S = 1.0
MAX_CAMERAS = 64
OUTPUTS = {"visit": segment_log.ZONE_VISITS, "coverage": segment_log.ZONE_COVERAGE}
STREAMS = (*OUTPUTS.values(), segment_log.PRESENCE_SPANS)


def checkpoint_path(state: Path) -> Path:
    return state / "see-zones.json"


@contextlib.contextmanager
def exclusive(path: Path) -> Generator[None]:
    """One see-zones per state directory: two would write the same segment streams."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another see-zones is running on this state directory") from error
        yield


class ZoneWorker:
    """Each camera's zone pipeline, fed see-actions' frames in time order."""

    def __init__(
        self,
        state: Path,
        *,
        follow: Callable[[Mapping[str, Mapping[str, int]] | None], Any] | None = None,
        current: Callable[[Path], Mapping[str, zones.Definition]] = zones.current,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.state, self.current, self.clock = state, current, clock
        root = segment_log.root(state)
        self._follow = follow or (lambda cursor: tracked_frames.Follow(root, cursor))
        self.pipelines: dict[str, zones.ZonePipeline] = {}
        self.episodes: dict[str, presence_episodes.Episodes] = {}
        saved = self._load()
        self.frames = self._follow(saved.get("cursor") if saved else None)
        self.writers = segment_log.Writers(root, saved.get("writers") if saved else None)
        self._zones_at = self._saved_at = -1e18

    # The checkpoint.

    def _load(self) -> dict[str, Any] | None:
        path = checkpoint_path(self.state)
        try:
            saved = json.loads(path.read_text())
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as error:
            LOG.warning("see-zones checkpoint unreadable (%s); starting from now", error)
            return None
        if not isinstance(saved, dict) or saved.get("version") != VERSION:
            LOG.info("see-zones checkpoint is from another version; starting from now")
            return None
        try:
            self.pipelines = {cam: zones.ZonePipeline(cam, state=s) for cam, s in saved["pipelines"].items()}
            self.episodes = {cam: self._new_episodes(cam, s) for cam, s in saved["episodes"].items()}
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            LOG.warning("see-zones checkpoint malformed (%s); starting from now", error)
            self.pipelines, self.episodes = {}, {}
            return None
        return saved

    def checkpoint(self) -> None:
        """Outputs made durable, then where everything stands, replaced in one rename."""
        # Every output a camera may write gets a position, written to yet or not: one left out would reopen at its end
        # after a crash and keep what the replay writes again.
        for cam in {*self.pipelines, *self.episodes}:
            for stream in STREAMS:
                self.writers.position(stream, cam)
        snapshot = {
            "version": VERSION,
            "writers": self.writers.sync(),
            "cursor": self.frames.cursor,
            "pipelines": {cam: p.snapshot() for cam, p in self.pipelines.items()},
            "episodes": {cam: e.snapshot() for cam, e in self.episodes.items()},
        }
        path = checkpoint_path(self.state)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(snapshot))
        os.replace(tmp, path)
        self._saved_at = self.clock()

    # The pipeline.

    def _pipeline(self, cam: str) -> zones.ZonePipeline:
        if cam not in self.pipelines:
            if len(self.pipelines) >= MAX_CAMERAS:
                raise RuntimeError("see-zones camera bound exceeded")
            self.pipelines[cam] = zones.ZonePipeline(cam)
        return self.pipelines[cam]

    @staticmethod
    def _new_episodes(cam: str, state: Mapping[str, Any] | None = None) -> presence_episodes.Episodes:
        return presence_episodes.Episodes(
            cam, absence_s=config.get("presence.absence_s"), gap_s=config.get("presence.gap_s"), state=state
        )

    def _episodes(self, cam: str) -> presence_episodes.Episodes:
        if cam not in self.episodes:
            if len(self.episodes) >= MAX_CAMERAS:
                raise RuntimeError("see-zones camera bound exceeded")
            self.episodes[cam] = self._new_episodes(cam)
        return self.episodes[cam]

    def refresh_zones(self) -> None:
        """The owner's current zone sets: a camera whose set changed starts again under the new one."""
        self._zones_at = self.clock()
        try:
            saved = self.current(self.state)
        except Exception as error:  # the broker out of reach: the zones in force stay
            LOG.warning("zone sets unreadable (%s); keeping the ones in force", error)
            return
        for cam, definition in saved.items():
            pipeline = self._pipeline(cam)
            if pipeline.definition is None or pipeline.definition.id != definition.id:
                self._write(pipeline.define(definition))

    def feed(self, frames: Sequence[tracked_frames.Frame]) -> None:
        for f in frames:
            record = {"id": f.frame_id or "", "ts_start": f.ts, "boxes": f.boxes, "track_ids": f.track_ids, "ok": True}
            self._write(self._pipeline(f.cam).frame("person", record, ["person"] * len(f.boxes)))
            ended = self._episodes(f.cam).frame(f.ts, f.track_ids)
            if ended:
                rows = [{**ep, "id": segment_log.new_id()} for ep in ended]
                self.writers.append(segment_log.PRESENCE_SPANS, f.cam, rows)

    def _write(self, outputs: Sequence[zones.Output]) -> None:
        # Visits first: coverage is the frontier a reader trusts (a stretch never runs ahead of its visits).
        for out in sorted(outputs, key=lambda o: o.kind != "visit"):
            record = {k: v for k, v in out.record.items() if not k.startswith("_")}
            self.writers.append(OUTPUTS[out.kind], str(record["cam"]), [{**record, "id": segment_log.new_id()}])

    def step(self, timeout_s: float = POLL_S) -> None:
        """One round: the zones when due, the frames that came, a checkpoint when due."""
        if self.clock() - self._zones_at >= ZONES_S:
            self.refresh_zones()
        self.feed(self.frames.poll(timeout_s))
        if self.clock() - self._saved_at >= CHECKPOINT_S:
            self.checkpoint()

    async def run(self, stop: asyncio.Event, batch_s: float = BATCH_S) -> None:
        """Steps until `stop`, starting one at most every `batch_s`; a stop ends the pause between them at once."""
        try:
            while not stop.is_set():
                began = self.clock()
                await asyncio.to_thread(self.step)
                rest = batch_s - (self.clock() - began)
                if rest > 0:
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(stop.wait(), rest)
            self.checkpoint()
        finally:
            self.writers.close()


def main(argv: Sequence[str] | None = None) -> None:
    """`pack-see-zones`: zone visits and coverage from see-actions' tracked people and the owner's zones."""
    import argparse

    parser = argparse.ArgumentParser(prog="pack-see-zones", description=main.__doc__)
    parser.parse_args(argv)
    state = config.path("paths.state")

    async def run() -> None:
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for signum in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(signum, stop.set)
        await ZoneWorker(state).run(stop)

    with exclusive(state / "see-zones.lock"):
        asyncio.run(run())
