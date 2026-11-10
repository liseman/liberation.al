"""Full-rate shared decoded camera frames. The initial transport is host memory, not CUDA IPC.

Every yielded image is copied under a short shared-cache lease. NV12 stays in native orientation;
BGR is upright, matching the existing vision workers. Source headers always describe the original capture.
"""

import asyncio
import contextlib
import hashlib
import json
import os
import tempfile
import time
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
from continuum_py import PayloadHeader

from liberational import config as cfg
from liberational import logs
from liberational.see.decoded_cache import DecodedCacheReader

MAX_MESSAGE = 1024 * 1024
LOG = logs.get("shared-frames")
STARTUP_WAIT_S = 60.0
_waiting_said = False  # this process said it is waiting for the decoder, and hasn't connected since
# Pause before reconnecting to a decoder that dropped the connection (it restarts during a service cascade).
RECONNECT_S = 0.5


SHM = Path("/dev/shm")


def runtime_dir() -> Path:
    """Where see-decode keeps its socket, lock, and frame ring (`paths.decoded`). Empty is the default: a directory
    for this user and this install, in memory under /dev/shm where it exists, else under the system temp directory.
    The ring is rewritten at full frame rate, so on disk it would be written back to the SSD for nothing."""
    if str(cfg.get("paths.decoded")).strip():
        return cfg.path("paths.decoded")
    base = SHM if SHM.is_dir() else Path(tempfile.gettempdir())
    # Two installs on one machine (each with its own state directory) never share a ring.
    install = hashlib.sha256(str(cfg.path("paths.state")).encode()).hexdigest()[:8]
    return base / f"liberational-{os.getuid()}-{install}"


def socket_path() -> Path:
    """The private decoder socket, in `runtime_dir()`; its frame ring is the `decoded` directory beside it."""
    return runtime_dir() / "decoded.sock"


async def _connect(address: Path) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Allow the decoder owner to start alongside its consumers during a normal service cascade."""
    loop = asyncio.get_running_loop()
    global _waiting_said
    deadline = loop.time() + STARTUP_WAIT_S
    while True:
        try:
            connection = await asyncio.open_unix_connection(str(address), limit=MAX_MESSAGE)
        except FileNotFoundError, ConnectionRefusedError:
            if loop.time() >= deadline:
                raise TimeoutError("shared decoder did not start within its startup deadline") from None
            if not _waiting_said:  # one line for the process, not one per camera waiting
                LOG.warning("waiting for the shared camera decoder to start")
                _waiting_said = True
            await asyncio.sleep(min(1.0, max(0.0, deadline - loop.time())))
        else:
            _waiting_said = False  # a later outage says so again
            return connection


@dataclass(frozen=True)
class DecodedFrame:
    """An actual source capture, with an explicit discontinuity before this picture when needed."""

    source_header: PayloadHeader
    pixels: np.ndarray
    decoder_epoch: str
    gap_before: bool
    color_range: str = "unspecified"
    color_matrix: str = "unspecified"

    @property
    def id(self) -> str:
        return self.source_header["id"]

    @property
    def capture_ns(self) -> int:
        return self.source_header["start_ns"]


class Schedule:
    """Which pictures a reader's rate takes: a schedule, not a minimum spacing from the last picture taken, since a
    10 fps source's jitter would otherwise skip every other picture (kitchen and bedroom got 6 fps of 10). A picture
    within a quarter period of its due time is taken, and the next is due a period after that due time, not after
    the picture. A picture after a gap is always taken and starts the schedule again. `every_ns` 0 takes every
    picture. A picture turned down leaves the schedule as it was, so running it over any run of pictures that holds
    every one it takes picks the same ones."""

    def __init__(self, every_ns: int) -> None:
        self.every_ns = every_ns
        self.due_ns: int | None = None

    def take(self, capture_ns: int, gap_before: bool) -> bool:
        if self.due_ns is not None and not gap_before and capture_ns < self.due_ns - self.every_ns // 4:
            return False
        behind = self.due_ns is None or gap_before or capture_ns - self.due_ns > self.every_ns
        self.due_ns = (capture_ns if behind else self.due_ns) + self.every_ns
        return True


async def decoded_frames(
    camera: str,
    *,
    format: Literal["nv12", "bgr"] = "nv12",
    socket: Path | None = None,
    on_break: Callable[[], None] | None = None,
    every_ns: int = 0,
) -> AsyncGenerator[DecodedFrame]:
    """Follow the camera's pictures in order, at most one every `every_ns` of capture time (a `Schedule`; 0 for every
    picture). Cache laps, restarts and decode losses set `gap_before`.

    The rate is the decoder's to apply: it runs each reader's `Schedule` before a picture is downloaded from
    the GPU, so a picture no reader takes is never copied or published, and a reader is only ever sent its own
    pictures. A reply says how many pictures it passed over for other readers (`skipped`), so those are no gap. A BGR
    reader gets at most one picture every `decoded_worker.BGR_EVERY_NS`, whatever it asks.

    There is no private software decoder fallback. A camera with no picture yet is waited for, and a decoder that
    drops the connection (it restarts during a cascade) is reconnected within `STARTUP_WAIT_S`, the next picture
    marked as a gap; a decoder that stays away past that deadline, or refuses the request, raises, so the service
    supervisor restarts the consumer without silently multiplying decode work. This API returns host arrays; a GPU
    consumer may upload NV12 and convert on its GPU. It must apply the source header's flip flag itself.
    """
    address = socket or socket_path()
    root = address.parent / "decoded"
    reader, writer = await _connect(address)
    cache = DecodedCacheReader(root, max_maps=8)
    after = 0
    owner: str | None = None
    previous_break: int | None = None
    gap = True
    waiting = False
    try:
        while True:
            request = {
                "camera": camera,
                "format": format,
                "after": after,
                "owner": owner,
                "break_count": previous_break,
                "every_ns": every_ns,
            }
            try:
                writer.write(json.dumps(request, separators=(",", ":")).encode() + b"\n")
                await writer.drain()
                raw = await reader.readline()
            except ConnectionError:
                raw = b""
            if not raw:
                LOG.warning("the shared camera decoder dropped %s's connection; reconnecting", camera)
                await _close(writer)
                await asyncio.sleep(RECONNECT_S)
                reader, writer = await _connect(address)
                # A new decoder run starts its own serials: begin again, and mark the next picture as a gap.
                after, owner, previous_break, gap = 0, None, None, True
                if on_break is not None:
                    on_break()
                continue
            descriptor = json.loads(raw)
            if "error" in descriptor:
                raise RuntimeError(f"the shared camera decoder refused {camera}: {descriptor['error'] or 'no reason'}")
            if descriptor.get("type") == "DecodedWaiting":
                if not waiting:
                    LOG.info("waiting for %s's first decoded picture", camera)
                    waiting = True
                continue
            waiting = False
            if descriptor.get("type") == "DecodedGap":
                previous_break = int(descriptor["break_count"])
                gap = True
                if on_break is not None:
                    on_break()
                continue
            serial, current_owner = int(descriptor["serial"]), str(descriptor["owner"])
            breaks = int(descriptor["break_count"])
            follows = after + 1 + int(descriptor.get("skipped", 0))  # pictures passed over for other readers
            gap = gap or owner != current_owner or (after != 0 and serial != follows) or previous_break != breaks
            after, owner, previous_break = serial, current_owner, breaks
            copied = cache.read(descriptor)
            if copied is None:
                gap = True
                continue
            pixels, header = copied
            yield DecodedFrame(
                cast(PayloadHeader, header),
                pixels,
                current_owner,
                gap,
                descriptor.get("color_range", "unspecified"),
                descriptor.get("color_matrix", "unspecified"),
            )
            gap = False
    finally:
        cache.close()
        await _close(writer)


LATEST_WAIT_S = 3.0  # how long latest() may take, connecting included
CAUGHT_UP_S = 0.1  # no newer picture within this long: the last one read is the newest


async def latest(
    camera: str, *, within_s: float, socket: Path | None = None, wait_s: float = LATEST_WAIT_S
) -> DecodedFrame | None:
    """The camera's newest decoded picture (BGR) from the last `within_s` seconds, for a one-off look (the Objects
    page's snapshot), or None: no picture in time, or no decoder answering within `wait_s`. The decoder hands out
    the oldest picture it still holds first, so this reads on until no newer one comes within `CAUGHT_UP_S`. Like
    the workers' frames, it isn't flipped, so the picture matches what see-objects matches against."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + wait_s
    frames = decoded_frames(camera, format="bgr", socket=socket)
    best: DecodedFrame | None = None
    try:
        while (left := deadline - loop.time()) > 0:
            try:
                best = await asyncio.wait_for(anext(frames), left if best is None else min(left, CAUGHT_UP_S))
            except TimeoutError:
                break
    finally:
        with contextlib.suppress(Exception):
            await frames.aclose()
    if best is None or time.time_ns() - best.capture_ns > within_s * 1e9:
        return None
    return best


async def _close(writer: asyncio.StreamWriter) -> None:
    """Close a decoder connection that may already be gone."""
    writer.close()
    with contextlib.suppress(ConnectionError):
        await writer.wait_closed()
