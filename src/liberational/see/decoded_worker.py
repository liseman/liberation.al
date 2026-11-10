"""One full-rate shared decoder per manifest camera, with a bounded transient host-frame fanout.

H.264 cameras decode once on NVDEC (one shared software decoder if CUDA is unavailable). MJPEG and UYVY
cameras, such as cam0 until capture encodes it as H.264, decode once on the CPU with OpenCV and are
converted to limited-range BT.601 NV12 only while an NV12 reader is attached. A camera decodes only while
some reader is attached; H.264 then restarts at the next IDR and the first picture is marked as a gap.

NV12 reaches GPU consumers without a CPU color conversion. Existing numpy consumers share one upright
BGR conversion. The initial bridge downloads NVDEC output to host memory; it is not CUDA IPC. Source
recording continues independently of this service, which never publishes derived video to the catalog.
`vision.shared_decode_cameras` only chooses which cameras the existing vision workers read through here;
any manifest camera can be read with `shared_frames.decoded_frames`.

Each reader asks for a rate (`every_ns`), and the decoder runs that reader's `shared_frames.Schedule` on each
picture's capture time as its unit arrives. Every access unit still goes to NVDEC (the cameras' P pictures are
all references, so none can be left undecoded), but a picture no reader takes is never downloaded from the GPU,
copied, converted, or published, and each reader is sent only the pictures its own schedule took: see-actions reads
the 30 fps cameras at 10, and see-captions takes one picture a second.

Each camera's decoder thread waits on its broker subscription itself (`Subscription.take`, every unit already waiting
in one wakeup) and decodes in place; the event loop hears from it only with pictures to publish or a break to note,
once per batch. The NVDEC decoders on a GPU share its one CUDA context, whose driver thread wakes hundreds of times a
second whether or not it has work.
"""

import asyncio
import concurrent.futures
import contextlib
import fcntl
import functools
import json
import os
import shutil
import signal
import stat
import threading
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import continuum_py
import numpy as np
from continuum_py import CaptureInfo, Client, PayloadHeader

from liberational import config as cfg
from liberational import continuum_link, logs
from liberational.see import frames
from liberational.see.decoded_cache import DecodedCache, PixelFormat
from liberational.see.shared_frames import MAX_MESSAGE, Schedule, socket_path

LOG = logs.get("see-decode")
getuid = os.getuid  # whose runtime directory this is (a name of its own, so a test can stand in for it)
MAX_CAMERAS = 100
MAX_CLIENTS = 512
# How long a request waits for a camera's next picture before the reply says it is still waiting.
WAITING_REPLY_S = 10.0
# Source shapes decoded on the CPU, one picture per payload. Anything else but H264AnnexB stops the camera.
STILL_SHAPES = ("MJPEG", "UYVY")
# A BGR reader gets a picture at most this often, by capture time, whatever rate it asks for: BGR is
# converted on the CPU, and converting and copying every 30 fps frame to BGR was most of see-decode's cost.
BGR_EVERY_NS = 500_000_000
# How long a camera's decode thread waits on its subscription before looking whether it should stop (leaving the
# subscription ends the wait at once; this is only the backstop), and how many waiting units it takes at once. A batch's
# pictures are published once it is all decoded, so a small batch keeps a thread catching up from holding back its
# first picture for long (eight decodes, a few milliseconds each); in step with a camera, a batch is one unit.
FOLLOW_WAIT_S = 1.0
EVENTS_PER_TAKE = 8
# How many pictures' reader picks a camera keeps while their decode is in flight (a pick for a picture the decoder
# dropped is let go after this many).
PICKS_HELD = 64
# How often a camera's count of unreadable frames is logged, at most.
UNREADABLE_REPORT_S = 60.0
# How far behind the oldest picture a decoder holds its H.264 references can reach: the longest keyframe interval
# capture allows (VideoEncodingSettings, 10 s), with a margin. A purge whose spans all end before that never touches
# see-decode's copies.
REFERENCE_REACH_NS = 15_000_000_000
# After a purge that does touch them, how long to gather the rest of a burst before acknowledging it all at once.
PURGE_SETTLE_S = 1.0
# After NVDEC fails, how long the camera decodes on the CPU before NVDEC is tried again: the next wait each time it
# fails again, the last one repeating, and the first again once NVDEC has decoded steadily for the last wait (so one
# that comes back, decodes a picture, and fails again doesn't flap every minute). One CUDA out-of-memory error (GPU
# memory is shared with Ollama and the vision models) used to mean the CPU until a restart (18 hours, once).
NVDEC_RETRY_S = (60.0, 300.0, 900.0)
_monotonic = time.monotonic  # a name of its own, so a test can stand in for the clock


class NativeDecoder:
    """Created and called on one dedicated thread, as the native SDK decoder is thread-affine."""

    def __init__(self, gpu: int, name: str = "") -> None:
        self.gpu = gpu
        self.name = name  # the camera's, for its decoder's log lines
        self.breaks = 0
        # Reset after a loss and not yet decoding again. Further losses in that state are only counted: a reset is
        # expensive on NVDEC, and resetting on every unreadable frame made see-decode fall further behind.
        self.waiting = False
        self.lost = 0
        self.decoder: frames.H264Frames | None = None
        self.on_gpu = False
        self.retry_at: float | None = None  # when NVDEC is next tried, while decoding on the CPU
        self.retries = 0  # NVDEC failures in a row, which pick the next wait (NVDEC_RETRY_S)
        self.cpu_since: float | None = None  # when NVDEC failed, until it decodes again
        self.gpu_since: float | None = None  # when NVDEC came back, until it has decoded steadily (NVDEC_RETRY_S[-1])
        try:
            self.decoder = frames.H264Frames(self.broke, backend="cuda", gpu=gpu, name=name)
            self.on_gpu = True
        except ValueError as error:
            wait = self._wait()
            LOG.warning(
                "NVDEC GPU %s unavailable (%s); sharing one software decoder, increased CPU cost expected, trying NVDEC"
                " again in %s",
                gpu,
                error,
                _span(wait),
            )
            try:
                self.decoder = frames.H264Frames(self.broke, name=self.name)
            except NotImplementedError as missing:
                # MJPEG cameras still work; an H.264 camera fails loudly when its first unit arrives.
                LOG.warning("no H.264 decoder in continuum_py (%s); only MJPEG and UYVY cameras can be shared", missing)
                self.retry_at = None  # no software decoder means no ffmpeg build: NVDEC isn't coming either

    def _wait(self) -> float:
        """Schedule the next NVDEC try after one more failure, and return how long until it."""
        wait = NVDEC_RETRY_S[min(self.retries, len(NVDEC_RETRY_S) - 1)]
        self.retries += 1
        now = _monotonic()
        self.retry_at = now + wait
        if self.cpu_since is None:
            self.cpu_since = now
        return wait

    def _to_cpu(self, error: ValueError) -> None:
        """NVDEC failed while decoding: decode on the CPU from the next IDR, and try NVDEC again later. A failure after
        NVDEC had been decoding is the switch, and warns; one on a try's first unit (it started, then ran out of
        memory) is logged like a try that couldn't start, so a GPU that keeps doing that goes quiet too."""
        decoding = self.on_gpu and self.retries == 0  # steady on NVDEC, not on a try
        quiet = self.retries >= len(NVDEC_RETRY_S)
        self.gpu_since = None
        self.broke()
        self.on_gpu = False
        self.decoder = frames.H264Frames(self.broke, name=self.name)
        wait = self._wait()
        if decoding:
            LOG.warning("NVDEC failed (%s); decoding on the CPU, trying NVDEC again in %s", error, _span(wait))
        elif not quiet:
            again = "every" if self.retries >= len(NVDEC_RETRY_S) else "in"
            LOG.info("NVDEC still unavailable (%s); decoding on the CPU, trying again %s %s", error, again, _span(wait))

    def _try_gpu(self) -> None:
        """Try NVDEC again (at the time `retry_at` set): on success it takes over from the next IDR, and on failure
        the CPU decoder goes on and the next try waits longer. A try that fails at the longest wait isn't logged
        again, so a lasting failure logs once per wait step, not every 15 minutes forever."""
        try:
            gpu = frames.H264Frames(self.broke, backend="cuda", gpu=self.gpu, name=self.name)
        except ValueError as error:
            quiet = self.retries >= len(NVDEC_RETRY_S)
            wait = self._wait()
            if not quiet:
                again = "every" if self.retries >= len(NVDEC_RETRY_S) else "in"
                LOG.info(
                    "NVDEC still unavailable (%s); decoding on the CPU, trying again %s %s", error, again, _span(wait)
                )
            return
        self.broke()  # the new decoder starts at the next IDR: a gap until then
        self.decoder, self.on_gpu, self.retry_at = gpu, True, None

    def broke(self) -> None:
        self.breaks += 1

    def decoded(self) -> None:
        """A picture came through another way (an MJPEG or UYVY still): the camera is decoding again, so its next loss
        is a new gap."""
        self.waiting = False

    def close(self) -> None:
        """Drop the native decoder; call on the thread that made it. `H264Frames` keeps `broke` as its break callback,
        so the two form a cycle that only the garbage collector frees, on whatever thread runs it, and the native
        decoder (pyo3 `unsendable`) refuses to be dropped off its own thread."""
        self.decoder = None

    def push(
        self, header: PayloadHeader, data: bytes | None, want_bgr: bool = False, keep: bool = True
    ) -> tuple[list[tuple[bytes, int, int, PayloadHeader, str, str, bytes | None]], int]:
        """Decode one access unit (None: it was lost). `keep` False decodes it for the reference chain only: its
        picture comes back with no pixels (empty bytes), never downloaded from the GPU."""
        if data is None:
            self.lost += 1
            if not self.waiting:
                self.waiting = True
                if self.decoder is None:
                    self.broke()
                else:
                    self.decoder.broke()  # one reset per loss; the chain restarts at the next IDR
            return [], self.breaks
        if self.decoder is None:
            raise ValueError("H.264 sharing requires continuum_py built with the ffmpeg feature")
        if not self.on_gpu and self.retry_at is not None and _monotonic() >= self.retry_at:
            self._try_gpu()
        try:
            pictures = self.decoder.push_nv12(header, data, rich=True, want_bgr=want_bgr, keep=keep)
        except ValueError as error:
            self._to_cpu(error)
            pictures = self.decoder.push_nv12(header, data, rich=True, want_bgr=want_bgr, keep=keep)
        if pictures:
            self.waiting = False
            if self.on_gpu:
                now = _monotonic()
                if self.cpu_since is not None:
                    if self.retries < len(NVDEC_RETRY_S):  # at the longest wait, a flapping GPU is quiet both ways
                        LOG.info("NVDEC is back, after %s on the CPU", _span(now - self.cpu_since))
                    self.cpu_since, self.gpu_since = None, now
                elif self.gpu_since is not None and now - self.gpu_since >= NVDEC_RETRY_S[-1]:
                    self.retries, self.gpu_since = 0, None  # steady: the next failure starts the waits over
        return [
            (pixels, width, height, original, *self.decoder.last_outputs[original["id"]])
            for pixels, width, height, original in pictures
        ], self.breaks


def _span(seconds: float) -> str:
    """A wait in words: "1 min", "15 min", or "2.5 h"."""
    return f"{seconds / 3600:.1f} h" if seconds >= 3600 else f"{max(1, round(seconds / 60))} min"


def decode_still(
    header: PayloadHeader, data: bytes, want_nv12: bool, want_bgr: bool
) -> tuple[bytes | None, int, int, bytes | None] | None:
    """One MJPEG or UYVY payload as sensor-orientation NV12 and BGR, each only when asked; None if it doesn't
    decode. The NV12 is OpenCV's limited-range BT.601 conversion of the decoded image, so its colors are known."""
    import cv2

    capture = header.get("capture")
    # Decode in sensor orientation: the channel rotates BGR itself, and NV12 stays unrotated.
    unflipped = None if capture is None else cast(CaptureInfo, {**capture, "flip": False})
    image = frames.decode_frame(header["shape_name"], data, unflipped)
    if image is None:
        return None
    height, width = image.shape[:2]
    if width % 2 or height % 2:
        image = image[: height - height % 2, : width - width % 2]
        height, width = image.shape[:2]
    nv12 = None
    if want_nv12:
        planar = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420)
        quarter = width * height // 4
        chroma = planar[height:].reshape(-1)
        interleaved = np.empty(2 * quarter, np.uint8)
        interleaved[0::2], interleaved[1::2] = chroma[:quarter], chroma[quarter:]
        nv12 = planar[:height].tobytes() + interleaved.tobytes()
    return nv12, width, height, np.ascontiguousarray(image).tobytes() if want_bgr else None


# Per format, the readers (connection numbers) whose schedules took a picture. A format no reader took is absent.
Picks = dict[str, frozenset[int]]
# One thing a camera's decode thread asks of the event loop (publish a picture, note a break), run there in order.
Step = Callable[[], Awaitable[None]]
NO_HEADER = cast(PayloadHeader, {})  # what a loss is pushed to the decoder with


@dataclass
class Reader:
    """One connection's reading of a camera: its format, its rate's schedule, and the decoder's break count at its last
    pick (a loss since, a capture restart with its new clock included, is a gap, which its schedule takes at once)."""

    format: str
    schedule: Schedule
    breaks: int


class Channel:
    """One camera's small per-format history. Slow readers cannot block decoding or recording."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.condition = asyncio.Condition()
        self.caches: dict[str, DecodedCache] = {}
        # Each format's newest published pictures, with the readers each was published for.
        self.recent: dict[str, deque[tuple[dict[str, Any], frozenset[int]]]] = {
            "nv12": deque(maxlen=4),
            "bgr": deque(maxlen=4),
        }
        self.readers: dict[int, Reader] = {}
        # The camera's decode thread runs the readers' schedules (`pick`) while connections attach and detach on the
        # event loop: this guards `readers` and `wanted` between them.
        self.readers_lock = threading.Lock()
        self.wanted: set[str] = set()
        self.breaks = 0
        self.layout: tuple[int, int] | None = None
        self.source_breaks = 0
        self.warned_colors = False
        # The copies this camera holds, for purges: each format's cache slot -> its source payload's id and
        # capture start, and the capture start of the newest payload given to the decoder since it was last reset.
        self.held: dict[str, dict[int, tuple[str, int]]] = {"nv12": {}, "bgr": {}}
        self.pushed_ns: int | None = None

    def attach(self, reader: int, format: str, every_ns: int = 0) -> None:
        """Start a reader (a connection's number) on a representation, which is made only while some reader wants
        it, at a picture every `every_ns` of capture time (0: every picture; BGR at most every BGR_EVERY_NS)."""
        if format not in self.recent:
            raise ValueError("unknown decoded representation")
        if isinstance(every_ns, bool) or not isinstance(every_ns, int) or every_ns < 0:
            raise ValueError("every_ns must be a nonnegative integer")
        if format == "bgr":
            every_ns = max(every_ns, BGR_EVERY_NS)
        with self.readers_lock:
            self.readers[reader] = Reader(format, Schedule(every_ns), self.source_breaks)
            self.wanted.add(format)

    def detach(self, reader: int) -> None:
        """Stop a reader. A representation's last reader stops its conversion and discards it."""
        with self.readers_lock:
            gone = self.readers.pop(reader, None)
            if gone is None or any(other.format == gone.format for other in self.readers.values()):
                return
            format = gone.format
            self.wanted.discard(format)
        cache = self.caches.pop(format, None)
        if cache is not None:
            cache.close()
        self.recent[format].clear()
        self.held[format].clear()

    def pick(self, header: PayloadHeader, breaks: int) -> Picks:
        """Run every reader's schedule on a picture before it is decoded: the readers that take it, by format. A
        picture nobody takes needn't be downloaded, converted, or published. One with no capture time is taken by
        every reader. `breaks` is the decoder's break count now (the decode thread's, which `source_breaks` follows
        once its pictures are published): a reader whose last pick was before a break takes this picture as a gap."""
        start = header.get("start_ns")
        chosen: dict[str, set[int]] = {}
        with self.readers_lock:
            for number, reader in self.readers.items():
                if not isinstance(start, int) or reader.schedule.take(start, reader.breaks != breaks):
                    reader.breaks = breaks
                    chosen.setdefault(reader.format, set()).add(number)
        return {format: frozenset(numbers) for format, numbers in chosen.items()}

    async def invalidate(self) -> None:
        self.breaks += 1
        for cache in self.caches.values():
            cache.close()
        self.caches.clear()
        for history in self.recent.values():
            history.clear()
        for held in self.held.values():
            held.clear()
        async with self.condition:
            self.condition.notify_all()

    async def note_breaks(self, breaks: int) -> None:
        """The decoder's break count is now `breaks`: if that is a new break (a loss, a reset, or a capture restart),
        drop what was published before it."""
        if breaks != self.source_breaks:
            await self.invalidate()
            self.source_breaks = breaks

    async def publish(
        self,
        data: bytes | None,
        width: int,
        height: int,
        header: PayloadHeader,
        source_breaks: int,
        picks: Picks,
        color_range: str = "unspecified",
        color_matrix: str = "unspecified",
        bgr: bytes | None = None,
    ) -> None:
        """Publish a decoded picture in each format its readers took it for (`picks`, from `pick`)."""
        if self.layout != (width, height) or self.source_breaks != source_breaks:
            await self.invalidate()
        self.layout, self.source_breaks = (width, height), source_breaks
        native = None if not data else np.frombuffer(data, np.uint8).reshape(height * 3 // 2, width)
        for format, readers in picks.items():
            if format not in self.wanted or not readers:
                continue
            if format == "nv12":
                if native is None:
                    continue  # an NV12 reader attached after this picture was decoded without NV12
                pixels = native
            else:
                import cv2

                if bgr is not None:
                    pixels = np.frombuffer(bgr, np.uint8).reshape(height, width, 3)
                elif native is None or (color_range != "unspecified" and color_matrix != "unspecified"):
                    # A BGR subscriber arrived while this NV12-only decode was in flight. Wait for
                    # the next signaled-color conversion rather than guessing this picture's colors.
                    continue
                else:
                    if not self.warned_colors:
                        LOG.warning(
                            "source color signaling is unknown; shared BGR uses the existing limited BT.601 conversion"
                        )
                        self.warned_colors = True
                    pixels = cv2.cvtColor(native, cv2.COLOR_YUV2BGR_NV12)
                if (header.get("capture") or {}).get("flip"):
                    pixels = cv2.rotate(pixels, cv2.ROTATE_180)
            cache = self.caches.get(format)
            if cache is None:
                cache = DecodedCache(self.root, format=cast(PixelFormat, format), max_pixels=width * height)
                self.caches[format] = cache
            descriptor = cache.publish(
                pixels, header, break_count=self.breaks, color_range=color_range, color_matrix=color_matrix
            )
            if descriptor is None:
                self.breaks += 1
                self.recent[format].clear()
                continue
            self.recent[format].append((descriptor, readers))
            self.held[format][descriptor["slot"]] = (str(header.get("id")), int(header.get("start_ns") or 0))
        async with self.condition:
            self.condition.notify_all()

    def oldest_ns(self) -> int | None:
        """The capture start of the oldest picture this camera holds a copy of, or gives its decoder references from;
        None when it holds nothing."""
        starts = [start for held in self.held.values() for _, start in held.values()]
        if self.pushed_ns is not None:
            starts.append(self.pushed_ns)
        return min(starts, default=None)

    async def next(
        self, format: str, owner: str | None, after: int, break_count: int | None, reader: int
    ) -> dict[str, Any]:
        """The oldest held picture after `after` that was published for `reader`, waiting for one. Its `skipped`
        counts the held pictures before it, after `after`, that were for other readers, so the reader can tell them
        from a loss."""
        if format not in self.recent or after < 0:
            raise ValueError("invalid decoded frame request")
        async with self.condition:
            while True:
                if owner is not None and break_count is not None and break_count != self.breaks:
                    return {"type": "DecodedGap", "break_count": self.breaks}
                skipped = 0
                for descriptor, readers in self.recent[format]:
                    if descriptor["owner"] != owner or descriptor["serial"] > after:
                        if reader in readers:
                            return {**descriptor, "skipped": skipped}
                        skipped += 1
                await self.condition.wait()


class DecoderService:
    """Own the source subscriptions and transient cache, independently of model worker lifetimes."""

    def __init__(self, client: Client, cameras: Sequence[str], gpu: int, root: Path) -> None:
        if len(cameras) > MAX_CAMERAS or len(set(cameras)) != len(cameras):
            raise ValueError("shared decode requires at most 100 distinct camera names")
        if not 0 <= gpu <= 2_147_483_647:
            raise ValueError("shared decode GPU must be a nonnegative ordinal")
        self.client, self.gpu = client, gpu
        self.channels = {camera: Channel(root) for camera in cameras}
        self.clients = 0
        self.connections = 0  # numbers each connection, its reader number on its camera
        self.handlers: set[asyncio.Task[Any]] = set()
        self.cleaned = False

    def touches(self, notice: Mapping[str, Any]) -> bool:
        """Whether a PurgeNotice could name a copy this service holds: a cached picture's own payload, or a span
        reaching into what the decoders hold (REFERENCE_REACH_NS behind their oldest picture). Spans are compared by
        time across every stream, which may stop the decoders for a purge of another stream; never the reverse."""
        ids = set(notice.get("payload_ids") or ())
        if any(
            payload in ids
            for channel in self.channels.values()
            for held in channel.held.values()
            for payload, _ in held.values()
        ):
            return True
        oldest = min(
            (at for channel in self.channels.values() if (at := channel.oldest_ns()) is not None), default=None
        )
        if oldest is None:
            return False
        reach = oldest - REFERENCE_REACH_NS
        return any(int(span["range"][1]) > reach for span in notice.get("spans") or ())

    async def camera(self, name: str, channel: Channel) -> None:
        """Decode one camera on a thread of its own (the decoder is thread-affine): `follow` waits on the subscription
        there and hands this task what the event loop must do, once per batch of access units."""
        loop = asyncio.get_running_loop()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="camera-decode")
        decoder: NativeDecoder | None = None
        initializing = loop.run_in_executor(executor, NativeDecoder, self.gpu, name)
        following: asyncio.Future[None] | None = None
        stop = threading.Event()
        try:
            try:
                decoder = await asyncio.shield(initializing)
            except asyncio.CancelledError:
                # A thread-affine decoder may finish opening after cancellation. Retain its result
                # so finally can destroy it on the same owner thread, rather than the event loop.
                decoder = await initializing
                raise
            while True:
                try:
                    subscription = await self.client.subscribe([f"{name}.video"], pinned=True)
                except continuum_py.BrokerError:
                    await asyncio.sleep(frames.RETRY_S)
                    continue
                break
            batches: asyncio.Queue[list[Step] | None] = asyncio.Queue()
            async with subscription:
                try:
                    following = loop.run_in_executor(
                        executor,
                        self.follow,
                        name,
                        channel,
                        decoder,
                        subscription,
                        stop,
                        functools.partial(loop.call_soon_threadsafe, batches.put_nowait),
                    )
                    # Settles after the thread's last batch, which call_soon_threadsafe already queued ahead of it.
                    following.add_done_callback(lambda _: batches.put_nowait(None))
                    while (batch := await batches.get()) is not None:
                        for step in batch:
                            await step()
                    await following  # its error, if it stopped on one
                finally:
                    stop.set()  # and leaving the block unsubscribes, which ends the thread's wait at once
        finally:
            if following is not None:
                await asyncio.wait([following])  # the thread is done with the decoder before it is closed
            await channel.invalidate()

            # Native decoder destruction also belongs to its owner thread, including a decoder that finished opening
            # while the camera was being cancelled.
            owned = decoder
            if owned is None and initializing.done() and not initializing.cancelled() and not initializing.exception():
                owned = initializing.result()
            if owned is not None:
                await loop.run_in_executor(executor, owned.close)
            executor.shutdown(wait=False, cancel_futures=True)

    def follow(
        self,
        name: str,
        channel: Channel,
        decoder: NativeDecoder,
        subscription: continuum_py.Subscription,
        stop: threading.Event,
        post: Callable[[list[Step]], object],
    ) -> None:
        """A camera's decode thread, the decoder's own: wait on its subscription, take every access unit already
        waiting, decode them in order, and `post` the event loop that batch's steps (publish a picture, note a break,
        or drop what was published), in order, once, or not at all when no reader took any of them. Waiting here
        rather than on the event loop wakes this thread once per batch, where each unit used to be handed from the
        broker's thread to asyncio, then to this thread and back. Ends when the subscription does, or `stop`
        is set."""
        idle = False
        unreadable, report_at = 0, _monotonic() + UNREADABLE_REPORT_S
        noted: int | None = None  # the break count last posted
        breaks = channel.source_breaks  # the decoder's, as its last push returned it
        # Picks for pictures given to the decoder and not back yet, by source payload id.
        in_flight: dict[str, Picks] = {}
        while not stop.is_set():
            events = subscription.take(FOLLOW_WAIT_S, EVENTS_PER_TAKE)
            if events is None:
                return
            steps: list[Step] = []
            try:
                for event in events:
                    if event.get("type") != "Event" or "item" not in event:
                        steps.append(channel.invalidate)
                        _, breaks = decoder.push(NO_HEADER, None)
                        channel.pushed_ns = None
                        continue
                    if not channel.wanted:
                        # Nobody is reading: skip decoding. H.264 restarts at the next IDR once a reader attaches.
                        if not idle:
                            _, breaks = decoder.push(NO_HEADER, None)
                            channel.pushed_ns = None
                            idle = True
                        continue
                    idle = False
                    item = cast(Mapping[str, Any], event["item"])
                    header = continuum_py.decode_header(item["header"])
                    shape = header["shape_name"]
                    if shape != "H264AnnexB" and shape not in STILL_SHAPES:
                        raise ValueError(f"{name}: shared decode supports H264AnnexB, MJPEG, or UYVY, not {shape}")
                    still = shape in STILL_SHAPES
                    picks = channel.pick(header, breaks) if still else {}
                    if still and not picks:
                        continue  # no reader takes it, and a still is nobody's reference: not even read
                    data = self.read(item.get("locator"))
                    if data is None:
                        unreadable += 1
                        if _monotonic() >= report_at:
                            LOG.warning(
                                "%s: %d frames unreadable in the last minute (their Tier 0 slot was gone, so see-decode"
                                " is reading behind); each loss is a gap until the next keyframe",
                                name,
                                unreadable,
                            )
                            unreadable, report_at = 0, _monotonic() + UNREADABLE_REPORT_S
                    if still:
                        decoded = None
                        if data is not None:
                            decoded = decode_still(header, data, "nv12" in picks, "bgr" in picks)
                        if decoded is None:
                            # A lost or undecodable picture is a gap, like an H.264 reset.
                            _, breaks = decoder.push(NO_HEADER, None)
                            channel.pushed_ns = None
                            continue
                        nv12, width, height, bgr = decoded
                        decoder.decoded()
                        breaks = decoder.breaks
                        if breaks != noted:
                            steps.append(functools.partial(channel.note_breaks, breaks))
                            noted = breaks
                        steps.append(
                            functools.partial(
                                channel.publish, nv12, width, height, header, breaks, picks, "limited", "bt601", bgr
                            )
                        )
                        continue
                    channel.pushed_ns = int(header.get("start_ns") or 0)
                    # Every unit is decoded (each P picture is a reference), but only a picture some reader takes is
                    # downloaded from the GPU (`keep`).
                    if data is not None and (picks := channel.pick(header, breaks)):
                        in_flight[str(header.get("id"))] = picks
                        while len(in_flight) > PICKS_HELD:
                            del in_flight[next(iter(in_flight))]
                    pictures, breaks = decoder.push(header, data, "bgr" in picks, bool(picks))
                    if breaks != noted:
                        steps.append(functools.partial(channel.note_breaks, breaks))
                        noted = breaks
                    for pixels, width, height, original, color_range, color_matrix, bgr in pictures:
                        taken = in_flight.pop(str(original.get("id")), None)
                        if taken and pixels:
                            steps.append(
                                functools.partial(
                                    channel.publish,
                                    pixels,
                                    width,
                                    height,
                                    original,
                                    breaks,
                                    taken,
                                    color_range,
                                    color_matrix,
                                    bgr,
                                )
                            )
            finally:
                if steps:
                    post(steps)  # what this batch did, an error included, reaches the loop first

    def read(self, location: Mapping[str, Any] | None) -> bytes | None:
        """A unit's bytes from its Tier 0 slot, or None: not in shared memory, or its slot already reused (see-decode
        is reading behind)."""
        if location is None or location["kind"] != "Shm":
            return None
        try:
            arena = self.client.arena(location["arena"])
            with arena.lease(
                location["slab_class"], location["slot"], location["slot_generation"], frames.READ_LEASE_MS
            ) as lease:
                return bytes(memoryview(lease)[: location["length"]])
        except continuum_py.ContinuumError:
            return None

    async def connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.clients += 1
        handler = asyncio.current_task()
        if handler is not None:
            self.handlers.add(handler)
        binding: tuple[Channel, str, int] | None = None
        self.connections += 1
        number = self.connections
        try:
            if self.clients > MAX_CLIENTS:
                raise ValueError("too many shared decode clients")
            while raw := await reader.readline():
                request = json.loads(raw)
                channel = self.channels.get(request["camera"])
                if channel is None:
                    raise ValueError("camera is not configured for shared decoding")
                selected = (channel, request["format"], request.get("every_ns", 0))
                if binding is None:
                    channel.attach(number, request["format"], request.get("every_ns", 0))
                    binding = selected
                elif binding != selected:
                    raise ValueError("a decoded connection must keep the same camera, representation, and rate")
                try:
                    descriptor = await asyncio.wait_for(
                        channel.next(
                            request["format"],
                            request.get("owner"),
                            int(request["after"]),
                            request.get("break_count"),
                            number,
                        ),
                        WAITING_REPLY_S,
                    )
                except TimeoutError:
                    # No picture yet (capture is still starting, say): say so and keep the connection, so the reader
                    # asks again rather than treating a quiet camera as a failure.
                    descriptor = {"type": "DecodedWaiting"}
                writer.write(json.dumps(descriptor, separators=(",", ":")).encode() + b"\n")
                await writer.drain()
        except (ValueError, KeyError, TypeError, TimeoutError) as error:
            writer.write(json.dumps({"error": str(error) or type(error).__name__}).encode() + b"\n")
            with contextlib.suppress(ConnectionError):
                await writer.drain()
        except ConnectionError:
            pass
        finally:
            self.clients -= 1
            if binding is not None:
                binding[0].detach(number)
            if handler is not None:
                self.handlers.discard(handler)
            writer.close()
            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()

    async def run(self, address: Path, stop: asyncio.Event, *, register: bool = True) -> None:
        """Serve until `stop`. `register` registers the broker worker (as a keeper of copies): once per connection,
        since a decoder epoch started again after a purge reuses it."""
        address.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        # The default directory sits in a shared one (/dev/shm or the temp directory): use it only if it's ours.
        if address.parent.stat().st_uid != getuid():
            raise ValueError(f"{address.parent} belongs to another user; set paths.decoded to a directory of yours")
        root = address.parent / "decoded"
        root.mkdir(mode=0o700, exist_ok=True)
        # Only the lock holder may replace a stale socket. A real file is never removed.
        with (address.parent / "decoded.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for abandoned in root.iterdir():
                name = abandoned.name
                if not name.startswith("decoded-") or len(name) < 46 or name[44] != "-":
                    continue
                try:
                    if str(uuid.UUID(name[8:44])) != name[8:44]:
                        continue
                except ValueError:
                    continue
                info = abandoned.lstat()
                if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
                    raise ValueError("abandoned decoded cache is not an owned directory")
                if any(
                    child.name not in {f"slot-{index}.pixels" for index in range(4)}
                    or not stat.S_ISREG(child.lstat().st_mode)
                    for child in abandoned.iterdir()
                ):
                    raise ValueError("abandoned decoded cache contains unexpected files")
                shutil.rmtree(abandoned)
            # Register only after proving exclusive ownership and discarding any previous owner's files.
            if register:
                await self.client.register_worker("see-decode", [], keeps_copies=True)
            if address.exists():
                if not stat.S_ISSOCK(address.lstat().st_mode):
                    raise ValueError("decoder socket path is occupied by a non-socket file")
                address.unlink()
            server = await asyncio.start_unix_server(self.connection, str(address), limit=MAX_MESSAGE)
            os.chmod(address, 0o600)
            inode = address.stat().st_ino
            tasks = [asyncio.create_task(self.camera(name, channel)) for name, channel in self.channels.items()]
            stopping = asyncio.create_task(stop.wait())
            try:
                done, _ = await asyncio.wait([*tasks, stopping], return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    if task is not stopping:
                        await task
                        raise RuntimeError("a shared camera decoder stopped")
            finally:
                server.close()
                stopping.cancel()
                for task in tasks:
                    task.cancel()
                outcomes = await asyncio.gather(stopping, *tasks, return_exceptions=True)
                handlers = list(self.handlers)
                for handler in handlers:
                    handler.cancel()
                outcomes.extend(await asyncio.gather(*handlers, return_exceptions=True))
                # Close the consumers' transports here, not in their deallocators after the server is gone: there,
                # Python 3.14 logs "TypeError: 'NoneType' object is not iterable" for each one.
                server.close_clients()
                await server.wait_closed()
                if address.exists() and address.stat().st_ino == inode:
                    address.unlink()
                self.cleaned = all(
                    not isinstance(result, BaseException) or isinstance(result, asyncio.CancelledError)
                    for result in outcomes
                )
                self.cleaned = self.cleaned and all(not channel.caches for channel in self.channels.values())


class Purges:
    """Answers the broker's purge notices for the decoders' copies. A purge that can't touch them (an age-out
    of old video: see-decode holds seconds of pictures) is acknowledged at once. One that can stops the running
    decoder epoch (`restart`) and waits in `pending` until the epoch has dropped its copies, with any others that
    arrive meanwhile, so a burst is one stop, and the process stays up."""

    def __init__(self, client: Client) -> None:
        self.client = client
        self.service: DecoderService | None = None  # the running epoch; None between epochs
        self.restart = asyncio.Event()  # the running epoch's stop
        self.pending: list[str] = []
        self.closed = asyncio.Event()  # the broker connection ended

    async def watch(self) -> None:
        try:
            while (event := await self.client.next_event()) is not None:
                notice = continuum_py.decode_event(event[1])
                if notice.get("type") != "PurgeNotice":
                    continue
                purge = str(notice["purge_id"])
                if purge in self.pending:
                    continue  # a later page of a purge already waiting: it's acknowledged with the rest
                if self.service is not None and not self.service.touches(notice):
                    self.client.ack_purge(purge)
                    continue
                self.pending.append(purge)
                if self.service is not None and not self.restart.is_set():
                    LOG.info("purge reaches the decoded copies; restarting the decoders before acknowledging it")
                    self.restart.set()
        finally:
            self.closed.set()
            self.restart.set()

    def acknowledge(self) -> None:
        """Acknowledge every pending purge: called only once the epoch that held copies has dropped them."""
        for purge in self.pending:
            self.client.ack_purge(purge)
        self.pending.clear()


async def serve(
    purges: Purges,
    make: Callable[[], DecoderService],
    terminating: asyncio.Event,
    settle_s: float = PURGE_SETTLE_S,
) -> None:
    """Run decoder epochs until `terminating` or the broker connection ends. A purge that reaches the copies ends an
    epoch, which drops them; its acknowledgement follows, with the rest of its burst, and the next epoch starts afresh
    in this process. An epoch that didn't clean up returns without acknowledging: a fresh process starts clean."""
    watching = asyncio.create_task(purges.watch())
    try:
        first = True
        while True:
            service = make()
            purges.service = service
            await service.run(socket_path(), purges.restart, register=first)
            purges.service = None
            first = False
            if not service.cleaned:
                return
            if purges.pending and not terminating.is_set() and not purges.closed.is_set():
                await asyncio.sleep(settle_s)
            purges.acknowledge()
            if terminating.is_set() or purges.closed.is_set():
                return
            purges.restart = asyncio.Event()
    finally:
        watching.cancel()
        await asyncio.gather(watching, return_exceptions=True)


def main(argv: Sequence[str] | None = None) -> None:
    """`pack-see-decode`: share one full-rate decoder per manifest camera."""
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("config/devices.toml"))
    args = parser.parse_args(argv)
    configured = cfg.get("vision.shared_decode_cameras")
    discovered = frames.cameras_from_manifest(args.manifest)
    known = {camera.name for camera in discovered}
    if not isinstance(configured, list) or any(
        not isinstance(name, str) or (name != "*" and name not in known) for name in configured
    ):
        raise SystemExit("shared decode cameras must name configured manifest cameras or '*'")
    # Every camera is served; the setting only picks which ones the vision workers read through here.
    selected = [camera.name for camera in discovered]
    gpu = int(cfg.get("vision.shared_decode_gpu"))
    frames.one_opencv_thread()

    async def run() -> None:
        async with continuum_link.connected(client_name="see-decode") as client:
            purges = Purges(client)
            terminating = asyncio.Event()

            def terminate() -> None:
                terminating.set()
                purges.restart.set()

            for sig in (signal.SIGINT, signal.SIGTERM):
                asyncio.get_running_loop().add_signal_handler(sig, terminate)
            await serve(
                purges, lambda: DecoderService(client, selected, gpu, socket_path().parent / "decoded"), terminating
            )

    continuum_link.run_service(run())
