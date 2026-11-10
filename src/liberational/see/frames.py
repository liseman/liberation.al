"""Camera frames from the broker, for the vision workers: which cameras capture's manifest has, and each
camera's newest frame, read from its Tier 0 slot and decoded.

`newest` follows one camera's `<name>.video` stream with a pinned subscription (retried until capture registers it)
and yields only the newest frame each time the caller is ready for one: a model slower than the camera skips frames
rather than falling behind, as v1's live path did. A frame is read from its slot under a short lease, decoded (MJPEG,
or raw UYVY at the header's size), and flipped when capture says so. H264AnnexB streams feed every access unit
through the optional native stateful decoder before newest-frame sampling, retaining the original presentation header.
"""

import asyncio
import logging
import time
import tomllib
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import continuum_py
import numpy as np
from continuum_py import CaptureInfo, Client, Item, PayloadHeader

from liberational import continuum_link, logs

log = logs.get("see-frames")

NS = 1_000_000_000
# How often a camera whose stream isn't registered yet is looked for again.
RETRY_S = 1.0
# How long a frame's slot is pinned while it's read and decoded.
READ_LEASE_MS = 2000


@dataclass(frozen=True)
class Camera:
    """A camera a worker reads: its device name in capture's manifest (its stream is `<name>.video`), its kind there,
    and how far apart in capture time this worker analyzes its frames (`every_ns`, 0 for no cap: the caption worker
    slots network cameras by its own interval)."""

    name: str
    kind: str = "camera"
    every_ns: int = 0
    shared_decode: bool = False

    @property
    def stream(self) -> str:
        return f"{self.name}.video"


def one_opencv_thread() -> None:
    """Run OpenCV with one thread in a vision worker. Its pool (a thread a core) spun between the small per-frame
    calls these workers make, and the spinning cost more than the work: 1.4 of see-captions' 1.8 cores,
    about half a core in the old face worker. Models run in their own runtimes' threads."""
    import cv2

    cv2.setNumThreads(1)


def cameras_from_manifest(path: Path, vision: Mapping[str, Any] | None = None) -> list[Camera]:
    """The cameras in capture's device manifest (`config/devices.toml`), each marked for the shared decoder by
    `vision`, the `[vision]` settings (read from config when omitted)."""
    manifest = tomllib.loads(path.read_text())
    devices = [d for d in manifest.get("device", []) if d.get("kind") in {"camera", "rtsp"} or "video" in d]
    if vision is None:
        from liberational import config as cfg

        # Key by key, so environment overrides apply.
        vision = {"shared_decode_cameras": cfg.get("vision.shared_decode_cameras")}
    selected = (vision or {}).get("shared_decode_cameras") or []
    if not isinstance(selected, list) or any(not isinstance(name, str) for name in selected):
        raise ValueError("vision.shared_decode_cameras must be a list of camera names or '*'")
    global_encoding = (manifest.get("video") or {}).get("encoding") or {}

    def uses_shared(device: Mapping[str, Any]) -> bool:
        encoding = {**global_encoding, **((device.get("video") or {}).get("encoding") or {})}
        eligible = device.get("kind") == "rtsp" or bool(encoding.get("enabled"))
        return device["name"] in selected or ("*" in selected and eligible)

    return [Camera(d["name"], d.get("kind", "camera"), 0, uses_shared(d)) for d in devices]


def check_h264_decoder(cameras: Sequence[Camera]) -> None:
    """Log an ERROR at a worker's start when the manifest has network (`rtsp`) cameras but continuum_py was built
    without the ffmpeg feature, so their H.264 can't be decoded and the worker would see none of their frames."""
    network = [c.name for c in cameras if c.kind == "rtsp" and not c.shared_decode]
    if not network or continuum_py.h264_decoder_available():
        return
    log.error(
        "continuum_py has no H.264 decoder (built without the ffmpeg feature), so network cameras %s get NO vision. "
        "Rebuild it with `mise run continuum-py` on a machine with FFmpeg's development libraries "
        "(config/system-packages.toml), then restart the vision workers.",
        ", ".join(network),
    )


def decode_frame(shape: str, data: bytes | memoryview, capture: CaptureInfo | None) -> np.ndarray | None:
    """A frame's bytes as a BGR image, or None if they don't decode. `shape` is the stream's format (`MJPEG` or
    `UYVY`); a raw frame's size comes from the header's capture info, which also says whether to flip it."""
    import cv2

    raw = np.frombuffer(data, np.uint8)
    if shape == "MJPEG":
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    elif shape == "UYVY" and capture is not None and capture["width"] * capture["height"] * 2 <= raw.size:
        w, h = capture["width"], capture["height"]
        img = cv2.cvtColor(raw[: w * h * 2].reshape(h, w, 2), cv2.COLOR_YUV2BGR_UYVY)
    else:
        return None
    if img is not None and capture is not None and capture.get("flip"):
        img = cv2.rotate(img, cv2.ROTATE_180)
    return img


def read_frame(client: Client, item: Item) -> tuple[np.ndarray, PayloadHeader] | None:
    """The frame an item names, decoded, with its header; None for one that can't be read (gone, or not a frame)."""
    header = continuum_py.decode_header(item["header"])
    loc = item["locator"]
    if loc is None or loc["kind"] != "Shm":
        return None
    try:
        arena = client.arena(loc["arena"])
        with arena.lease(loc["slab_class"], loc["slot"], loc["slot_generation"], READ_LEASE_MS) as lease:
            img = decode_frame(header["shape_name"], memoryview(lease)[: loc["length"]], header.get("capture"))
    except continuum_py.ContinuumError as e:
        log.debug("a frame was gone before it was read: %s", e)
        return None
    return (img, header) if img is not None else None


class H264Frames:
    """Decode every access unit, retaining its original header through frame reordering. `on_break` is called when
    pictures are lost: a gap in the camera's sequence, a size change, a damaged unit, or too many held headers (not
    the first unit's own reset). `name` is the camera's, for the log."""

    def __init__(
        self, on_break: Callable[[], None] | None = None, *, backend: str = "software", gpu: int = 0, name: str = ""
    ) -> None:
        self.on_break = on_break
        self.name = name
        self.backend = backend
        self.decoder = (
            continuum_py.H264Decoder() if backend == "software" else continuum_py.H264Decoder(backend=backend, gpu=gpu)
        )
        self.pending: dict[int, PayloadHeader] = {}
        self.sequence: int | None = None
        self.layout: tuple[int, int, bool, int] | None = None
        self.tag = 0
        self.last_outputs: dict[str, tuple[str, str, bytes | None]] = {}

    def broke(self) -> None:
        """Discard unread input's reference chain and delayed pictures before reporting the loss."""
        self.decoder.reset()
        self.pending.clear()
        self.sequence = None
        self.layout = None
        if self.on_break is not None:
            self.on_break()

    def push(
        self, header: PayloadHeader, data: bytes, want: Callable[[PayloadHeader], bool] | None = None
    ) -> list[tuple[np.ndarray, PayloadHeader]]:
        """Decode every AU, converting only selected native pictures into upright BGR."""
        import cv2

        result = []
        for nv12, width, height, original in self.push_nv12(header, data, want):
            image = cv2.cvtColor(np.frombuffer(nv12, np.uint8).reshape(height * 3 // 2, width), cv2.COLOR_YUV2BGR_NV12)
            if (original.get("capture") or {}).get("flip"):
                image = cv2.rotate(image, cv2.ROTATE_180)
            result.append((image, original))
        return result

    def push_nv12(
        self,
        header: PayloadHeader,
        data: bytes,
        want: Callable[[PayloadHeader], bool] | None = None,
        *,
        rich: bool = False,
        want_bgr: bool = False,
        keep: bool = True,
    ) -> list[tuple[bytes, int, int, PayloadHeader]]:
        """Decode once into native-orientation host NV12, retaining each exact original source header. With `rich`,
        `keep` False decodes this unit for the reference chain only: its picture comes back with empty NV12 and no
        BGR, never downloaded or copied (see-decode's pictures that no reader takes)."""
        self.last_outputs.clear()
        capture = header.get("capture")
        if capture is None:
            self.broke()
            return []
        sequence = capture["device_seq"]
        layout = (capture["width"], capture["height"], capture.get("flip", False), capture.get("capture_epoch", 0))
        if (
            (self.sequence is not None and sequence != self.sequence + 1)
            or layout != self.layout
            or len(self.pending) >= 256
        ):
            self.decoder.reset()
            self.pending.clear()
            if self.sequence is not None and self.on_break is not None:
                self.on_break()
        self.sequence, self.layout = sequence, layout
        self.tag += 1
        try:
            if rich:
                pictures = self.decoder.push_outputs(
                    data, self.tag, header["start_ns"] if self.backend == "cuda" else None, want_bgr, keep
                )
            else:
                plain = (
                    self.decoder.push(data, self.tag)
                    if self.backend == "software"
                    else self.decoder.push(data, self.tag, header["start_ns"])
                )
                pictures = [
                    (tag, width, height, pixels, "unspecified", "unspecified", None)
                    for tag, width, height, pixels in plain
                ]
        except ValueError as error:
            # A damaged camera AU must not restart every vision model. Discard its reference chain and
            # pending headers; the native decoder suppresses deltas until the next actual IDR.
            self.decoder.reset()
            self.pending.clear()
            log.warning(
                "%s: an H264 access unit could not be decoded (%s); waiting for a keyframe",
                self.name or "a camera",
                logs.reason(error),
            )
            if self.on_break is not None:
                self.on_break()
            return []
        if not self.decoder.waiting_for_keyframe:
            self.pending[self.tag] = header
        dropped = self.decoder.dropped_tags
        for tag in dropped:
            self.pending.pop(tag, None)
        if dropped and self.on_break is not None:
            self.on_break()
        result: list[tuple[bytes, int, int, PayloadHeader]] = []
        for tag, width, height, nv12, color_range, color_matrix, bgr in pictures:
            original = self.pending.pop(tag, None)
            if original is None:
                raise ValueError("H264 decoder returned a frame without its original header")
            if want is not None and not want(original):
                continue
            if rich:
                self.last_outputs[original["id"]] = (color_range, color_matrix, bgr)
            result.append((nv12, width, height, original))
        return result


def read_h264(
    client: Client, item: Item, decoder: H264Frames, want: Callable[[PayloadHeader], bool] | None = None
) -> list[tuple[np.ndarray, PayloadHeader]]:
    """Copy an AU under its lease, then decode after releasing the shared-memory slot (`want` as `H264Frames.push`)."""
    header = continuum_py.decode_header(item["header"])
    loc = item.get("locator")
    if loc is None or loc["kind"] != "Shm":
        decoder.broke()
        return []
    try:
        arena = client.arena(loc["arena"])
        with arena.lease(loc["slab_class"], loc["slot"], loc["slot_generation"], READ_LEASE_MS) as lease:
            data = bytes(memoryview(lease)[: loc["length"]])
    except continuum_py.ContinuumError as error:
        log.debug("an H264 access unit was gone before it was read: %s", error)
        decoder.broke()
        return []
    return decoder.push(header, data, want)


async def newest(
    client: Client,
    cam: Camera,
    min_interval_s: float = 0.0,
    on_break: Callable[[], None] | None = None,
) -> AsyncGenerator[tuple[np.ndarray, PayloadHeader]]:
    """The camera's newest frame each time the caller asks, at most one every `min_interval_s`, decoded, with its
    header. Independent frames that arrive while the caller is busy are skipped, except the last; H.264 access
    units all decode first so skipped pictures can still provide references.

    The camera's `every_ns` caps the rate by capture time: capture time is cut into `every_ns` slots, and a
    frame is a candidate only in a slot no frame was taken or held from, so at most one frame a slot reaches the
    caller, and the slots, unlike `min_interval_s`, don't drift with load. Other frames are dropped before they're
    read or, for H.264, color converted; they are never witnessed. A capture clock that goes back or repeats is a
    break (`on_break`), and that frame reaches the caller even in a slot already used.

    A camera read through the shared decoder (`cam.shared_decode`) is paced by the decoder instead: it sends a
    picture each `every_ns`, else each `min_interval_s`, of capture time on its `shared_frames.Schedule`, so nothing
    over the rate is converted or sent, and the slots and the monotonic interval here don't apply.

    Skipping is the sampling policy. Losing a frame is not: `on_break` is called when one couldn't be read or decoded
    (gone from its slot, an H.264 reset), and the frame waiting from before the loss is dropped, so the next frame
    yielded comes after it.
    """
    latest: list[Item | tuple[np.ndarray, PayloadHeader]] = []
    decoder: H264Frames | None = None
    ready = asyncio.Event()
    every_ns = cam.every_ns
    pace_s = 0.0 if cam.shared_decode else min_interval_s  # the shared decoder paces its own pictures
    taken_slot: int | None = None  # the capture-time slot of the frame last yielded
    held_slot: int | None = None  # and of the frame waiting in `latest`
    seen_ns: int | None = None  # the capture time of the newest frame the cap looked at

    def wanted(header: PayloadHeader) -> bool:
        """Whether a frame is a candidate under the rate cap; one that is holds its slot. A capture clock that went
        back or repeated is a break, and its frame a candidate whatever its slot, so the caller sees the regression
        rather than a skipped frame hiding it."""
        nonlocal held_slot, taken_slot, seen_ns
        if not every_ns:
            return True
        start = header["start_ns"]
        if seen_ns is not None and start <= seen_ns:
            broke()
            taken_slot = None
        seen_ns = start
        slot = start // every_ns
        if slot in (taken_slot, held_slot):
            return False
        held_slot = slot
        return True

    def broke() -> None:
        nonlocal held_slot
        latest.clear()
        held_slot = None
        if on_break is not None:
            on_break()

    def replace_latest(value: Item | tuple[np.ndarray, PayloadHeader]) -> None:
        latest[:] = [value]

    async def follow() -> None:
        nonlocal decoder
        if cam.shared_decode:
            from liberational.see.shared_frames import decoded_frames

            # The decoder applies the rate: it sends only the pictures this reader's schedule takes, one each
            # `every_ns` (else `min_interval_s`) of capture time, rather than every shared BGR picture for the slots
            # here to drop half of them. Its pictures are in order, so there is no clock regression to catch.
            rate = every_ns or round(min_interval_s * 1e9)
            async for picture in decoded_frames(cam.name, format="bgr", on_break=broke, every_ns=rate):
                if picture.gap_before:
                    broke()
                replace_latest((picture.pixels, picture.source_header))
                ready.set()
            return
        # The camera's stream may not exist yet (capture starts after the worker, or restarts): try again every
        # second until it does.
        while True:
            try:
                sub = await client.subscribe([cam.stream], pinned=True)
            except continuum_py.BrokerError as e:
                log.info("%s: not there yet (%s); trying again in %.0f s", cam.stream, e, RETRY_S)
                await asyncio.sleep(RETRY_S)
                continue
            async with sub:
                async for event in sub:
                    if event["type"] == "Event" and "item" in event:
                        item = event["item"]
                        header = continuum_py.decode_header(item["header"])
                        if header["shape_name"] == "H264AnnexB":
                            if decoder is None:
                                decoder = H264Frames(on_break=broke)
                            decoded = read_h264(client, item, decoder, wanted)
                            if not decoded:
                                continue
                            for picture in decoded:
                                replace_latest(picture)
                        elif wanted(header):
                            replace_latest(item)
                        else:
                            continue
                        ready.set()
            return

    follower = asyncio.create_task(follow())
    follower.add_done_callback(lambda _: ready.set())
    last = -float("inf")
    try:
        while True:
            if follower.done() and not latest:
                await follower
                return
            await ready.wait()
            ready.clear()
            if not latest:
                if follower.done():
                    await follower
                    return
                continue
            (item,) = latest
            latest.clear()
            slot, held_slot = held_slot, None
            if time.monotonic() - last < pace_s:
                continue
            last = time.monotonic()
            frame = item if isinstance(item, tuple) else read_frame(client, item)
            if frame is not None:
                taken_slot = slot
                yield frame
            else:
                broke()
    finally:
        follower.cancel()
        await asyncio.gather(follower, return_exceptions=True)


class Refusals:
    """The payloads the broker refused this worker (IngestRejected), counted by reason."""

    def __init__(self) -> None:
        self.count = 0
        self.by_reason: dict[str, int] = {}


async def watch_refusals(client: Any, refusals: Refusals | None = None) -> None:
    """Logs each payload the broker refuses this client (an IngestRejected event: a parent it doesn't know, say),
    which `ingest` alone never says: the first REFUSALS_LOGGED of each reason in full, then every REFUSALS_EVERY-th.
    Ends when the connection closes."""
    from continuum_py import decode_event

    seen = refusals or Refusals()
    while (event := await client.next_event()) is not None:
        try:
            decoded = decode_event(event[1])
        except ValueError:
            continue
        if decoded.get("type") != "IngestRejected":
            continue
        reason = str(decoded.get("reason"))
        seen.count += 1
        n = seen.by_reason[reason] = seen.by_reason.get(reason, 0) + 1
        if n <= REFUSALS_LOGGED or n % REFUSALS_EVERY == 0:
            log.warning(
                "the broker refused payload %s on stream %s: %s (%d so far for this reason)",
                decoded.get("payload_id"),
                decoded.get("stream_id"),
                reason,
                n,
            )


REFUSALS_LOGGED = 5
REFUSALS_EVERY = 100


async def until_stopped(stop: asyncio.Event, cameras: Sequence[asyncio.Task[None]]) -> None:
    """Waits for `stop`. A camera's task that ends first (its model wouldn't load, say) has its error raised, logged at
    ERROR first, so the service exits and its supervisor restarts it, rather than living on doing nothing. A task
    that just ended (its frames stopped, as when the broker goes) raises a RuntimeError unlogged: the service's
    `continuum_link.connected` block tells a broker outage (one WARN) from anything else."""
    stopping = asyncio.ensure_future(stop.wait())
    try:
        done, _ = await asyncio.wait([stopping, *cameras], return_when=asyncio.FIRST_COMPLETED)
    finally:
        stopping.cancel()
    if stopping in done:
        return
    task = next(t for t in cameras if t in done)
    error = task.exception() if not task.cancelled() else None
    if error is not None:
        continuum_link.failed(log, f"a camera stopped: {str(error) or type(error).__name__}", error, logging.ERROR)
    raise error or RuntimeError("a camera's frames ended")


def source_content_rect(metadata: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """The upright source rectangle in actual decoded pixels; unknown geometry is never inferred."""
    if not isinstance(metadata.get("flip"), bool):
        return None
    width, height = metadata.get("width"), metadata.get("height")
    if any(isinstance(v, bool) or not isinstance(v, int) or not 0 < v <= 8192 for v in (width, height)):
        return None
    if not isinstance(width, int) or not isinstance(height, int):
        return None
    native = metadata.get("nativeGeometry")
    if (
        metadata.get("profileStatus") == "absent"
        and isinstance(native, dict)
        and native.get("width") == width
        and native.get("height") == height
    ):
        return 0.0, 0.0, float(width), float(height)
    return None


def source_box_to_decoded(box: Sequence[float], metadata: dict[str, Any]) -> list[float] | None:
    """Map upright source xyxy shares onto upright decoded pixels, before any CSS fitting."""
    import math

    rect = source_content_rect(metadata)
    if rect is None or len(box) != 4 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in box):
        return None
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return None
    x, y, w, h = rect
    width, height = metadata["width"], metadata["height"]
    return [(x + x1 * w) / width, (y + y1 * h) / height, (x + x2 * w) / width, (y + y2 * h) / height]


def decoded_box_to_source(box: Sequence[float], metadata: dict[str, Any]) -> list[float] | None:
    """Inverse of source mapping; selections reaching into letterbox padding are rejected."""
    import math

    rect = source_content_rect(metadata)
    if rect is None or len(box) != 4 or any(not math.isfinite(v) for v in box):
        return None
    x, y, w, h = rect
    width, height = metadata["width"], metadata["height"]
    x1, y1, x2, y2 = box
    out = [(x1 * width - x) / w, (y1 * height - y) / h, (x2 * width - x) / w, (y2 * height - y) / h]
    if out[2] <= out[0] or out[3] <= out[1] or any(v < -1e-9 or v > 1 + 1e-9 for v in out):
        return None
    return [max(0.0, min(1.0, v)) for v in out]
