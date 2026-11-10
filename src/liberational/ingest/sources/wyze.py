"""The Wyze source: Wyze cameras recorded through docker-wyze-bridge (docs/INGEST.md), polled by `pack-ingest poll`.

The bridge signs in to Wyze (with `ingest.wyze_*`'s email, password, API ID, and key, which it takes as its own
environment) and serves each camera as RTSP at `<ingest.wyze_rtsp_url>/<camera>`. Continuum's capture opens local
cameras only (V4L2, AVFoundation), not RTSP, so the cameras don't go through capture: each one in
`ingest.wyze_cameras` gets an ffmpeg that copies its stream, without re-encoding, into segments of
`ingest.wyze_segment_s` seconds under `<paths.ingest>/wyze/<camera>/`, named by their start time in UTC
(`20261101-083000Z.mp4`). Local names repeat the hour clocks fall back, which lost that hour's segments as already
seen (#3122); segments recorded with the old local names (no `Z`) are still read in local time. A segment
is new once it's finished (a newer one has started, or this source's tracked recorder has exited); the pipeline
then takes it (moving it out of the recording directory) onto the camera's own streams,
`wyze-<camera>.video_v1` and `.audio_v1`.

Copying, not encoding, keeps the cost per camera to a little I/O, so 20 or more cameras are one ffmpeg each. A
recorder that exits is started again on the next poll, at most once a minute per camera.

An old mtime does not prove the newest segment is closed: RTSP can stall with ffmpeg still alive. Dashboard polls
cannot observe the service's recorder process, so they conservatively defer the newest file, including an orphan
left after a service restart, until a newer segment exists. A known exit is snapshotted before restarting ffmpeg.
"""

import datetime as dt
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from liberational import config as cfg
from liberational import logs
from liberational.ingest.pipeline import stream_name
from liberational.ingest.sources import NotConfigured, RemoteItem, SourceState, Status

LOG = logs.get("ingest-wyze")
SEGMENT_NAME = "%Y%m%d-%H%M%SZ"  # in UTC: the recorder runs with TZ=UTC0, so ffmpeg's -strftime fills it in UTC
LOCAL_SEGMENT_NAME = "%Y%m%d-%H%M%S"  # segments named before #3122, in local time
RESTART_S = 60.0


def segment_start(path: Path) -> float | None:
    """A segment's start, from its name (UTC, or local time for an old name without the `Z`), or None for another
    file."""
    try:
        return dt.datetime.strptime(path.stem, SEGMENT_NAME).replace(tzinfo=dt.UTC).timestamp()
    except ValueError:
        pass
    try:
        return dt.datetime.strptime(path.stem, LOCAL_SEGMENT_NAME).timestamp()
    except ValueError:
        return None


def _by_start(path: Path) -> tuple[float, str]:
    return segment_start(path) or 0.0, path.name


def finished_segments(cam_dir: Path, *, recorder_stopped: bool = False) -> list[Path]:
    """Closed segments, oldest first (by start: an old local name and a new UTC one don't sort by name). Only a
    known recorder exit allows including the newest file."""
    segs = sorted((p for p in cam_dir.glob("*.mp4") if segment_start(p) is not None), key=_by_start)
    if not segs:
        return []
    return segs if recorder_stopped else segs[:-1]


def record_command(ffmpeg: str, url: str, out_dir: Path, segment_s: int) -> list[str]:
    """ffmpeg copying one camera's RTSP stream into `segment_s` second MP4 segments named by their start."""
    return [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-rtsp_transport", "tcp", "-i", url,
        "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy",
        "-f", "segment", "-segment_time", str(segment_s), "-segment_format", "mp4", "-reset_timestamps", "1",
        "-strftime", "1", str(out_dir / f"{SEGMENT_NAME}.mp4"),
    ]  # fmt: skip


class Proc(Protocol):
    """What Recorders needs of a process (subprocess.Popen, or a fake)."""

    @property
    def returncode(self) -> int | None: ...

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def kill(self) -> None: ...


def _spawn(cmd: list[str]) -> Proc:
    env = {**os.environ, "TZ": "UTC0"}  # segment names (SEGMENT_NAME) in UTC
    return subprocess.Popen(
        cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env
    )


@dataclass
class Recorders:
    """One ffmpeg per camera, started when missing and again after it exits (at most every RESTART_S)."""

    spawn: Callable[[list[str]], Proc] = _spawn
    clock: Callable[[], float] = time.monotonic
    procs: dict[str, Proc] = field(default_factory=dict)
    started: dict[str, float] = field(default_factory=dict)

    def ensure(self, cameras: dict[str, list[str]]) -> list[str]:
        """Starts the recorder of each camera (by name: its command) that isn't running. Returns those started."""
        out = []
        for cam, cmd in cameras.items():
            proc = self.procs.get(cam)
            if proc is not None and proc.poll() is None:
                continue
            if proc is not None:
                LOG.warning(f"wyze {cam}: recorder exited ({proc.returncode})")
            if self.clock() - self.started.get(cam, float("-inf")) < RESTART_S:
                continue
            self.procs[cam] = self.spawn(cmd)
            self.started[cam] = self.clock()
            out.append(cam)
        for cam in set(self.procs) - set(cameras):  # a camera taken out of the config
            self.procs.pop(cam).terminate()
        return out

    def stop(self) -> None:
        for proc in self.procs.values():
            proc.terminate()
        for proc in self.procs.values():
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        self.procs.clear()


def credentials_set() -> dict[str, bool]:
    """Which of the bridge's settings are filled in (never their values). Written out, so readers.py sees each key."""
    values = {
        "email": cfg.get("ingest.wyze_email"),
        "password": cfg.get("ingest.wyze_password"),
        "api_id": cfg.get("ingest.wyze_api_id"),
        "api_key": cfg.get("ingest.wyze_api_key"),
    }
    return {k: bool(str(v).strip()) for k, v in values.items()}


@dataclass
class WyzeSource:
    cameras: Sequence[str]
    rtsp_url: str
    root: Path
    segment_s: int = 300
    credentials: dict[str, bool] = field(default_factory=dict)  # which of the bridge's settings are filled in
    recorders: Recorders = field(default_factory=Recorders)
    record: bool = True  # start the recorders on each poll (pack-ingest poll); the dashboard only lists
    name: str = "wyze"
    label: str = "Wyze cameras"
    ongoing: bool = True

    @classmethod
    def from_config(cls, record: bool = False) -> WyzeSource:
        return cls(
            cameras=[str(c).strip() for c in cfg.get("ingest.wyze_cameras") if str(c).strip()],
            rtsp_url=str(cfg.get("ingest.wyze_rtsp_url")).rstrip("/"),
            root=cfg.path("paths.ingest") / "wyze",
            segment_s=int(cfg.get("ingest.wyze_segment_s")),
            credentials=credentials_set(),
            record=record,
        )

    def status(self) -> Status:
        if not self.cameras:
            missing = [k for k, ok in self.credentials.items() if not ok]
            why = "No cameras listed (ingest.wyze_cameras)"
            if missing:
                why += f"; the bridge still needs {', '.join(missing)}"
            return Status(False, why)
        return Status(True, f"{len(self.cameras)} cameras through the bridge at {self.rtsp_url}")

    def camera_dir(self, cam: str) -> Path:
        return self.root / stream_name(cam)

    def list_new(self, state: SourceState) -> list[RemoteItem]:
        if not self.cameras:
            raise NotConfigured("no Wyze cameras listed (ingest.wyze_cameras)")
        stopped = {
            cam: finished_segments(self.camera_dir(cam), recorder_stopped=True)
            for cam in self.cameras
            if (proc := self.recorders.procs.get(cam)) is not None and proc.poll() is not None
        }
        if self.record:
            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                raise NotConfigured("ffmpeg isn't installed (mise install)")
            commands = {}
            for cam in self.cameras:
                self.camera_dir(cam).mkdir(parents=True, exist_ok=True)
                url = f"{self.rtsp_url}/{cam}"
                commands[cam] = record_command(ffmpeg, url, self.camera_dir(cam), self.segment_s)
            for cam in self.recorders.ensure(commands):
                LOG.info(f"wyze {cam}: recording")
        items = []
        for cam in self.cameras:
            d = self.camera_dir(cam)
            if not d.is_dir():
                continue
            # A replacement may already have opened a new segment. Only files seen before that spawn are closed
            # by the old process's exit; an absent/foreign process is never evidence of an exit.
            closed = set(finished_segments(d)) | set(stopped.get(cam, []))
            for seg in sorted(closed, key=_by_start):
                key = f"{stream_name(cam)}/{seg.name}"
                if state.seen(key):
                    continue
                items.append(
                    RemoteItem(
                        key=key,
                        name=f"{cam} {seg.stem}.mp4",
                        taken_at=segment_start(seg),
                        stream=f"wyze-{stream_name(cam)}",
                        extra={"path": str(seg)},
                    )
                )
        return items

    def fetch(self, item: RemoteItem, into: Path) -> Path:
        """The segment itself: it's already on this machine (the pipeline moves it out of the recording dir)."""
        path = Path(str(item.extra.get("path", "")))
        if not path.is_file() or not re.fullmatch(r"\d{8}-\d{6}Z?\.mp4", path.name):
            raise FileNotFoundError(f"no segment at {path}")
        return path
