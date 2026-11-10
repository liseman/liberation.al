"""The vision log on Continuum: captions, detections, matches, poses, crossings, motion, and visits as broker
payloads, read back as dicts.

Each is a stream of small JSON payloads stored inline in the catalog (a pose or a
detection is far under the 64 KiB inline cap). The header carries the time (UTC nanoseconds, half open), the producer,
and the provenance (the frames it came from, as parents); the JSON body carries the rest. Face embeddings aren't in
the body: they go to the vector worker, keyed by the detection's payload id.

Each stream is one per kind, not one per camera, with the camera in the body (`cam`), as `speech_log` keeps the
device: a reader asks for "captions over this hour" more often than for one camera's, and a camera's are filtered
from the same query.

Readers get dicts with `ts_start` and `ts_end` (seconds, from the header), `id` (the payload's id), and the stream's
body keys. `VisionLog` is the synchronous face for threaded code; async code uses `read_range` and `follow` with
its own client.
"""

import collections
import contextlib
import json
import threading
import time
import uuid
from collections.abc import AsyncGenerator, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import continuum_py
from continuum_py import Client, DomainLayer, Item, PayloadHeader, Producer, read

from liberational import continuum_link, logs
from liberational.continuum import from_ns, to_ns
from liberational.continuum_link import BrokerLink

CAPTIONS = "camera.caption"
OBJECTS = "camera.objects"
PET_SIGHTINGS = "pet.sighting"
ZONE_VISITS = "zone.visit"
ZONE_COVERAGE = "zone.coverage"


@dataclass(frozen=True)
class StreamDef:
    """A vision stream: its layer, and which body keys its payloads carry."""

    name: str
    layer: DomainLayer
    keys: tuple[str, ...]


STREAMS = {
    d.name: d
    for d in (
        # A scene.json entry (camera_history's row): the VLM's observation of one view.
        StreamDef(
            CAPTIONS,
            "Layer1Raw",
            (
                "cam",
                "frame_seq",
                "yaw_deg",
                "caption",
                "faces",
                "objects",
                "source",
                "synthetic",
                "gated",
                "observation",
                "change",
                "brightness",
                "ms",
                "unchanged",
                "since",
                "deferred",
            ),
        ),
        StreamDef(OBJECTS, "Layer1Raw", ("cam", "frame_seq", "objects", "source", "observation")),
        # The owner's label for one animal sighting (people/pet_recognition.py): a named pet's id, "other",
        # or cleared (`active` false), parented to the embeddings it was judged from. The newest per track wins.
        StreamDef(
            PET_SIGHTINGS,
            "Layer2Decision",
            ("cam", "track_id", "species", "label", "active", "source", "sighting_start"),
        ),
        # A subject's visit to a zone, and where visits could have been seen (see/zones.py, see/zones_worker.py).
        StreamDef(
            ZONE_VISITS,
            "Layer3Interpretation",
            (
                "cam",
                "zone",
                "zone_name",
                "definition",
                "subject_kind",
                "track_id",
                "label",
                "dwell_s",
                "frames",
                "entry_seen",
                "exit_seen",
                "ended",
                "candidate",
                "interpretation",
            ),
        ),
        StreamDef(ZONE_COVERAGE, "Layer3Interpretation", ("cam", "subject_kind", "definition", "zones", "frames")),
    )
}


async def register(client: Client, stream: str) -> int:
    """Registers a vision stream (or finds it, when it's registered already) and returns its id."""
    d = STREAMS[stream]
    registered = await continuum_link.until_not_busy(
        lambda: client.register_stream(
            d.name, kind="derived", format="Text", shape_name=d.name, layer=d.layer, slab_class=0
        )
    )
    return registered["stream_id"]


def encode(
    stream: str,
    stream_id: int,
    record: dict[str, Any],
    *,
    payload_id: str | None = None,
    producer: Producer | None = None,
    parents: Sequence[str] = (),
    subjects: Sequence[str] = (),
    span_ns: tuple[int, int] | None = None,
    seq: int = 0,
    keep_null: Sequence[str] = (),
) -> tuple[bytes, bytes]:
    """A record as a payload: its header and its inline JSON body. `record` has `ts_start` (seconds), optionally
    `ts_end` (a frame's record is a point in time), and the stream's body keys; keys it lacks are left out, None
    values too (except those in `keep_null`, written as null), and any others are refused so nothing is silently
    dropped. `span_ns` gives the header's span exactly
    (a frame's capture time, say), which float seconds can't always hold to the nanosecond. `seq` tells apart records
    that start in the same nanosecond on one stream (the catalog keys a stream's payloads by start and seq).
    Evaluated-empty animal rows omit empty class-id and withheld diagnostics, retaining every required sample field."""
    d = STREAMS[stream]
    extra = set(record) - set(d.keys) - {"ts_start", "ts_end", "id"}
    if extra:
        msg = f"{stream} records don't carry {sorted(extra)}"
        raise ValueError(msg)
    if span_ns is not None:
        start_ns, end_ns = span_ns[0], max(span_ns)
    else:
        start_ns = to_ns(float(record["ts_start"]))
        end_ns = max(start_ns, to_ns(float(record.get("ts_end", record["ts_start"]))))
    header: PayloadHeader = {
        "id": payload_id or str(uuid.uuid7()),
        "stream_id": stream_id,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "layer": d.layer,
        "format": "Text",
        "shape_name": d.name,
        "parents": list(parents),
        "subjects": list(subjects),
        "seq": seq,
    }
    if producer is not None:
        header["producer"] = producer
    body = {k: record[k] for k in d.keys if record.get(k) is not None or (k in keep_null and k in record)}
    return continuum_py.encode_header(header), json.dumps(body, separators=(",", ":")).encode()


def decode(item: Item) -> tuple[str, dict[str, Any]] | None:
    """A payload as (its stream, its record), or None for one that isn't a vision stream's or has no inline body."""
    locator = item["locator"]
    if locator is None or locator["kind"] != "Inline" or "bytes" not in locator:
        return None
    header = continuum_py.decode_header(item["header"])
    stream = header["shape_name"]
    if stream not in STREAMS:
        return None
    _count_decoded(1)
    record: dict[str, Any] = json.loads(locator["bytes"])
    record["ts_start"] = from_ns(header["start_ns"])
    record["ts_end"] = from_ns(header["end_ns"])
    record["id"] = header["id"]
    return stream, record


async def records_of(
    client: Client, stream: str, item: Item, start_ns: int = 0, end_ns: int = 2**63 - 1, *, keep_ns: bool = False
) -> list[dict[str, Any]]:
    """The records a query or subscription `item` on `stream` holds, wherever its bytes are (`continuum_py.read`):
    one for an inline payload, or a flushed chunk's records overlapping [start_ns, end_ns). Each has `ts_start`,
    `ts_end`, and `id` from its own payload. None of them for a payload purged since, or one that isn't a record."""
    if stream not in STREAMS:
        return []
    found = await read.records(client, stream, item, start_ns=start_ns, end_ns=end_ns)
    return _decode_records(found or [], keep_ns=keep_ns)


LOG = logs.get("vision-log")
DECODE_SUMMARY_S = 60.0  # how often a process logs who had the vision log decode records for it
_decoded: collections.Counter[str] = collections.Counter()
_decoded_lock = threading.Lock()
_decoded_since = time.monotonic()


def _decode_records(found: Sequence[read.Record], *, keep_ns: bool = False) -> list[dict[str, Any]]:
    """Decode records only after a caller has applied any raw-record budget."""
    out = []
    for rec in found:
        record = _decode_record(rec.data, rec.payload_id, rec.start_ns, rec.end_ns, keep_ns=keep_ns)
        if record is not None:
            out.append(record)
    return out


def _count_decoded(n: int) -> None:
    """Count records decoded for whoever asked (continuum_link.CALLER), and once a minute log the biggest readers:
    decoding runs on the link's loop for every caller, so its cost shows only as that one thread's."""
    global _decoded_since
    if not n:
        return
    with _decoded_lock:
        _decoded[continuum_link.CALLER.get()] += n
        elapsed = time.monotonic() - _decoded_since
        if elapsed < DECODE_SUMMARY_S:
            return
        top, total = _decoded.most_common(6), sum(_decoded.values())
        _decoded.clear()
        _decoded_since = time.monotonic()
    readers = "; ".join(f"{who} {count}" for who, count in top)
    LOG.info("vision log decoded %d records in %.0f s: %s", total, elapsed, readers)


def _decode_record(
    data: bytes, payload_id: str, start_ns: int, end_ns: int, *, keep_ns: bool = False
) -> dict[str, Any] | None:
    """One JSON object, with identity and time from its header rather than its body."""
    _count_decoded(1)
    try:
        record = json.loads(data)
    except ValueError:
        return None
    if not isinstance(record, dict):
        return None
    record.update(ts_start=from_ns(start_ns), ts_end=from_ns(end_ns), id=payload_id)
    if keep_ns:
        record.update(_start_ns=start_ns, _end_ns=end_ns)
    return record


async def read_range(
    client: Client, stream: str, start: float, end: float, *, keep_ns: bool = False
) -> list[dict[str, Any]]:
    """Every record on `stream` whose span overlaps [start, end) (seconds), in time order, inline or flushed."""
    lo, hi = to_ns(start), to_ns(end)
    # Older SDKs retain the full-item path during the reader-first rollout. New SDKs project inline identity,
    # span, and bytes directly from their validated Rust headers, in the same order as the full query.
    compact = getattr(client, "query_inline_records", None)
    items: Sequence[tuple[str, int, int, bytes] | Item]
    if compact is None:
        items = (await client.query(stream, lo, hi))["items"]
    else:
        items = await compact(stream, lo, hi)
    if stream not in STREAMS:
        return []
    records = []
    for item in items:
        if isinstance(item, tuple):
            payload_id, first, last, data = item
            record = _decode_record(data, payload_id, first, last, keep_ns=keep_ns)
            if record is not None:
                records.append(record)
            continue
        locator = item["locator"]
        if locator is not None and locator["kind"] == "Inline":
            # Inline bytes are already in the reply. Avoid the generic reader's two full header projections
            # and per-item coroutine/Record wrappers for dense detector history. Other locators still need
            # its lease checks, chunk identity, and stale-location retry.
            header = continuum_py.decode_header(item["header"])
            record = _decode_record(
                bytes(locator["bytes"]), header["id"], header["start_ns"], header["end_ns"], keep_ns=keep_ns
            )
            if record is not None:
                records.append(record)
        else:
            records.extend(await records_of(client, stream, item, lo, hi, keep_ns=keep_ns))
    records.sort(key=lambda r: (r["ts_start"], r["id"]))
    return records


async def follow(
    client: Client,
    streams: Sequence[str],
    *,
    start: Literal["Live", "Time", "Saved"] = "Live",
    since: float = 0.0,
    durable_name: str | None = None,
    wait_for_streams_s: float = 0.0,
) -> AsyncGenerator[tuple[str, dict[str, Any]]]:
    """(stream, record) as each is committed: from now, from `since`, or after a durable name's saved position. Each
    record has its `position`. Following under a durable name, a record's position is acknowledged when the caller
    asks for the next one, so a record the caller never finished with comes again next time (at least once).
    `wait_for_streams_s` optionally waits for producer registration at startup; live subscription failures still raise.
    Each record is read through `client`, so it can't be a Backpressure reader, and a caller slow enough to leave
    4,096 events waiting gets the connection closed under the SDK's default policy: use a connection of
    its own and connect again after a close."""

    # Pinned, so each event carries its locator, which `records_of` reads like a query item's.
    async def subscribe() -> continuum_py.Subscription:
        return await client.subscribe(
            list(streams), start=start, start_ns=to_ns(since), durable_name=durable_name, pinned=True
        )

    sub = (
        await continuum_link.until_streams_ready(subscribe, wait_for_streams_s)
        if wait_for_streams_s > 0
        else await subscribe()
    )
    async with sub:
        async for event in sub:
            if event["type"] != "Event" or "item" not in event:
                continue
            stream = continuum_py.decode_header(event["item"]["header"])["shape_name"]
            # A flushed chunk can straddle `since`: only its records from then on.
            from_ns = to_ns(since) if start == "Time" else 0
            for record in await records_of(client, stream, event["item"], from_ns):
                record["position"] = event["position"]
                yield stream, record
            if durable_name is not None:
                sub.ack(event["position"])


def scene_entries(captions: Iterable[dict[str, Any]], limit: int = 30) -> list[dict[str, Any]]:
    """Caption records as the scene's entries: the newest `limit`, oldest first, each with `ts` (its start,
    seconds)."""
    newest = sorted(captions, key=lambda r: (r["ts_start"], r["id"]))[-limit:] if limit > 0 else []
    return [{**r, "ts": r["ts_start"]} for r in newest]


def latest_by_camera(captions: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The newest caption record of each camera."""
    out: dict[str, dict[str, Any]] = {}
    for r in sorted(captions, key=lambda r: (r["ts_start"], r["id"])):
        out[r["cam"]] = r
    return out


def is_real(caption: Mapping[str, Any]) -> bool:
    """Whether a `camera.caption` record is a description the VLM gave then: not a repeat of an earlier one for an
    unchanged view (`unchanged`), and not empty (warming up, or a failed call)."""
    text = caption.get("caption")
    return isinstance(text, str) and bool(text.strip()) and not caption.get("unchanged")


def as_v1_row(caption: Mapping[str, Any]) -> dict[str, Any]:
    """A `camera.caption` record with two more names beside its own, which some readers use: `ts` (its start,
    seconds) and `yaw` (`yaw_deg`)."""
    return {**caption, "ts": caption["ts_start"], "yaw": caption.get("yaw_deg", 0.0)}


class VisionLog:
    """The vision log for synchronous code: one broker client on a background event loop (see `BrokerLink`)."""

    #: How far back `scene` looks for the newest captions (the last 30 entries are about half a
    #: minute of captioning; an hour covers a camera that's been unchanged for a while).
    SCENE_WINDOW_S = 3600.0

    def __init__(
        self,
        socket: Path | None = None,
        *,
        client_name: str = "vision-log",
        timeout: float = 10.0,
        state: Path | None = None,
    ) -> None:
        self.link = BrokerLink(socket, client_name=client_name, timeout=timeout)
        # Pictures decode on a link of their own: an H.264 frame decodes its whole stretch up to that frame, on
        # the link's loop thread, and every other read queued behind it. After a restart People's face shots and the
        # unknown-face photos decoded at once, and plain reads (/people, /faces/unknown/groups, /scene) timed out.
        # Made on the first picture, so a process that never decodes one never starts its thread.
        self._frames_args = (socket, f"{client_name}-frames", timeout)
        self._frames_link: BrokerLink | None = None
        self._frames_lock = threading.Lock()
        self._scene_lock = threading.Lock()
        self._scene: tuple[float, int, list[dict[str, Any]]] | None = None  # (when read, limit, entries)
        self.state = state  # whose segment store holds the captions (`scene`, `latest`, `captions`); None: the pack's
        self._registered: dict[str, Client] = {}

    def range(self, stream: str, start: float, end: float) -> list[dict[str, Any]]:
        """An interactive read, retaining the link's normal ten-second deadline."""
        return self._range(stream, start, end, self.link.timeout)

    def _range(
        self, stream: str, start: float, end: float, timeout: float, *, window_s: float | None = None
    ) -> list[dict[str, Any]]:
        """Every record on `stream` overlapping [start, end) (seconds), in time order. The stream is registered
        first, once per client (registering again is harmless): on a fresh broker a stream its worker hasn't written
        yet isn't there, and asking about it would fail every read until it is (as `speech_log` does)."""

        async def go(c: Client) -> list[dict[str, Any]]:
            if self._registered.get(stream) is not c:
                await register(c, stream)
                self._registered[stream] = c
            if window_s is None:
                return await read_range(c, stream, start, end)
            # Bound catalog item batches for dense detector history rather than materializing the whole day.
            records: dict[str, dict[str, Any]] = {}
            at = start
            while at < end:
                stop = min(at + window_s, end)
                for record in await read_range(c, stream, at, stop):
                    records[record["id"]] = record
                at = stop
            return sorted(records.values(), key=lambda record: (record["ts_start"], record["id"]))

        return self.link.call(go, timeout=timeout)

    #: How long one read of the scene serves every caller.
    SCENE_TTL_S = 5.0

    def scene(self, now: float, limit: int = 30) -> list[dict[str, Any]]:
        """`scene.json`'s entries as of `now`: the newest `limit` captions of the last hour, oldest first.

        One read serves every caller for SCENE_TTL_S, and callers that arrive during a read wait for it instead of
        starting their own. A read costs 0.4 to 1.5 s holding the GIL whatever its stretch (the segment
        files are read whole), and with several pages polling /scene, reads back to back stalled the dashboard's
        other requests. Captions change about once a minute per camera, so a few seconds' age is never seen."""
        with self._scene_lock:
            cached = self._scene
            if cached is not None and cached[1] == limit and 0 <= time.monotonic() - cached[0] < self.SCENE_TTL_S:
                return list(cached[2])
            entries = scene_entries(self._captions(max(0.0, now - self.SCENE_WINDOW_S), now + 1.0), limit)
            self._scene = (time.monotonic(), limit, entries)
            return list(entries)

    def latest(self, now: float) -> dict[str, dict[str, Any]]:
        """The newest caption of each camera captioned in the last hour."""
        return latest_by_camera(self._captions(max(0.0, now - self.SCENE_WINDOW_S), now + 1.0))

    def close(self) -> None:
        """Closes the connections and stops the background loops."""
        self.link.close()
        with self._frames_lock:
            if self._frames_link is not None:
                self._frames_link.close()
                self._frames_link = None

    def frames_link(self) -> BrokerLink:
        """The link pictures decode on, apart from every other read (made on first use)."""
        with self._frames_lock:
            if self._frames_link is None:
                socket, name, timeout = self._frames_args
                self._frames_link = BrokerLink(socket, client_name=name, timeout=timeout)
            return self._frames_link

    def frame(self, cam: str, ts: float, within_s: float = 0.001) -> Any:
        """The camera's frame captured nearest `ts` (seconds; a record's time is its frame's), within `within_s` (a
        millisecond by default), decoded to a BGR image (numpy), or None when the broker no longer holds it."""
        return self.frames_link().call(lambda c: frame_at(c, cam, ts, within_s))

    def clip(self, cam: str, start: float, end: float, fps: float) -> list[Any]:
        """The camera's frames over [start, end) at `fps`, decoded in one pass (`clip_frames`), None where none."""
        return self.frames_link().call(lambda c: clip_frames(c, cam, start, end, fps))

    def frame_evidence(self, cam: str, ts: float) -> FrameEvidence | None:
        """Exact capture and per-picture geometry for source-normalized crop consumers."""
        return self.frames_link().call(lambda c: frame_evidence_at(c, cam, ts))

    def video_spans(self, cam: str, start: float, end: float) -> list[list[float]]:
        """Where `cam` has video in [start, end) (`video_spans`)."""
        return self.link.call(lambda c: video_spans(c, cam, start, end))

    def aged_out(self, cam: str, start: float, end: float) -> list[list[float]]:
        """Where `cam`'s video in [start, end) was aged out (the broker's record of what video_window.py removed), as
        [a, b] seconds: what lets a reader say "footage expired" rather than "there was none"."""

        async def go(client: Client) -> list[list[float]]:
            try:
                spans = await client.aged_out(f"{cam}.video", to_ns(start), to_ns(end))
            except continuum_py.ContinuumError:
                return []
            return [[from_ns(a), from_ns(b)] for a, b in spans]

        return self.link.call(go)

    def video_days(self) -> list[str]:
        """The local days with video (`video_days`)."""
        return self.link.call(video_days)

    def video_cameras(self) -> list[str]:
        """The cameras with a video stream (`video_cameras`)."""
        return self.link.call(video_cameras)

    def has_video(self, start: float, end: float) -> bool:
        """Whether any camera has video in [start, end) (`has_video`)."""
        return self.link.call(lambda c: has_video(c, start, end))

    def newest(self, cam: str, within_s: float = 5.0) -> tuple[Any, float, str] | None:
        """The camera's newest frame from the last `within_s` seconds, as (BGR image, capture time in seconds, its
        source payload's id), or None. It comes from see-decode's shared decode (see/shared_frames.py): stored camera
        frames are H.264 access units, which can't be decoded one at a time."""
        import asyncio

        from liberational.see import shared_frames

        got = asyncio.run(shared_frames.latest(cam, within_s=within_s))
        return None if got is None else (got.pixels, from_ns(got.capture_ns), got.id)

    def captions(self, start: float, end: float, cams: Iterable[str] = ()) -> list[dict[str, Any]]:
        """Every `camera.caption` record overlapping [start, end) (seconds), on `cams` when given, in time order, each
        with `ts` and `yaw` too (`as_v1_row`)."""
        wanted = set(cams)
        rows = self._captions(start, end, sorted(wanted) if wanted else None)
        return [as_v1_row(r) for r in rows if not wanted or r.get("cam") in wanted]

    def _captions(self, start: float, end: float, cams: list[str] | None = None) -> list[dict[str, Any]]:
        """`camera.caption` records over [start, end), only `cams`' when given (reading one camera is a third of the
        time of reading all): they're in the segment store, not the broker."""
        from liberational import segment_log

        return segment_log.at(self.state).range(segment_log.CAPTIONS, start, end, cams)


@dataclass(frozen=True)
class FrameEvidence:
    """Decoded upright pixels and geometry belonging to one actual capture identity."""

    image: Any
    captured_at_ns: int
    metadata: dict[str, Any]


H264 = "H264AnnexB"
# How far before a wanted picture to start decoding: from the GOP's keyframe (capture's encoder puts one every 2 s;
# a network camera's GOP can be longer, so the second, longer try). Decoding runs this far past it too, so the next
# crop in the same stretch continues the open decoder instead of starting again.
GOP_LOOKBACK_S = (4.0, 30.0)
MAX_OPEN_GOPS = 8  # open decoders kept, one per camera stream, newest use last


def _unit(data: bytes) -> bytes:
    """A Tier 1 chunk record's access unit: continuum-tier's one-unit record (count u32, then one 25-byte row of start
    u64, end u64, flags u8, offset u32, length u32, then the bytes) unwrapped; anything else (a Tier 0 access unit) as
    it is."""
    if len(data) < 29 or int.from_bytes(data[:4], "little") != 1:
        return data
    offset, length = int.from_bytes(data[21:25], "little"), int.from_bytes(data[25:29], "little")
    return data[29 + offset : 29 + offset + length] if offset == 0 and 29 + length == len(data) else data


def _codec(unit: bytes) -> str:
    """`av1` for an archived chunk's AV1 temporal unit (it opens with a temporal delimiter OBU), else `h264`:
    an Annex B access unit opens with a zero byte."""
    return "av1" if unit[:1] and unit[0] & 0x80 == 0 and (unit[0] >> 3) & 0xF == 2 else "h264"


@dataclass
class _Gop:
    """One decoded-forward stretch of a camera's recorded video: its records (start_ns, bytes) in decode order, and the
    decoder part-way through. `codecs` is each record's codec when the stretch has AV1 in it (the archive pass): the
    decoder is drained and replaced where the codec changes."""

    records: list[tuple[int, bytes]]
    flip: bool
    decoder: Any
    codecs: list[str] | None = None
    codec: str = "h264"
    pushed: int = 0
    pictures: dict[int, tuple[bytes, int, int]] = field(default_factory=dict)  # decoded ahead (reordered) by index
    emitted: set[int] = field(
        default_factory=set
    )  # every index the decoder has output (output order isn't decode order)
    finished: bool = False  # drained: the decoder takes no more

    def passed(self, index: int) -> bool:
        """Whether `index`'s picture is gone from this stretch: output and not kept, or the decoder drained."""
        return index not in self.pictures and (index in self.emitted or self.finished)


# Per thread: a native decoder must be used and dropped on the thread that made it (the broker link's loop thread).
_OPEN = threading.local()


def _gops() -> collections.OrderedDict[str, _Gop]:
    """This thread's open stretches. Made on a running loop, so at exit they're dropped there (see _drop_at_exit)."""
    gops = getattr(_OPEN, "gops", None)
    if gops is None:
        import asyncio
        import atexit

        gops = _OPEN.gops = collections.OrderedDict()
        atexit.register(_drop_at_exit, asyncio.get_running_loop(), gops)
    return gops


def _drop_at_exit(loop: Any, gops: collections.OrderedDict[str, _Gop]) -> None:
    """Drop the decoders on their own loop's thread before the interpreter finalizes that thread's locals elsewhere."""
    import asyncio

    async def drop() -> None:
        gops.clear()

    if loop.is_running():
        with contextlib.suppress(Exception):
            asyncio.run_coroutine_threadsafe(drop(), loop).result(timeout=2)


def _nearest(records: Sequence[tuple[int, bytes]], target_ns: int, tolerance_ns: int) -> int | None:
    best = min(range(len(records)), key=lambda i: abs(records[i][0] - target_ns), default=None)
    return best if best is not None and abs(records[best][0] - target_ns) <= tolerance_ns else None


def _keep(gop: _Gop, index: int, pictures: Iterable[Any]) -> None:
    """Keep the decoder's pictures at or after `index` as upright-to-be BGR bytes with their width and height. The
    BGR is the decoder's own, converted with the range and matrix the stream signals (full-range BT.601 from
    capture's encoder, BT.709 from a network camera). A stream that signals neither gets no BGR from the decoder,
    so its NV12 is converted as limited-range BT.601, the shared decoder's fallback (see/decoded_worker.py)."""
    import cv2
    import numpy as np

    for tag, width, height, nv12, _range, _matrix, bgr in pictures:
        gop.emitted.add(int(tag))
        if tag < index:
            continue
        if bgr is None:
            native = np.frombuffer(nv12, np.uint8).reshape(height * 3 // 2, width)
            bgr = cv2.cvtColor(native, cv2.COLOR_YUV2BGR_NV12).tobytes()
        gop.pictures[int(tag)] = (bytes(bgr), int(width), int(height))


def _advance(gop: _Gop, index: int) -> tuple[bytes, int, int] | None:
    """Push gop's records (in decode order) up to `index` and give that record's picture as (BGR, width, height),
    or None (before the stretch's first keyframe, or damaged). A picture the decoder holds back for reordering
    comes out a few records later, so this pushes on until it does, and drains the decoder only at the stretch's
    end. Pictures before it are dropped; later ones are kept for the next request."""
    if index in gop.pictures:
        return gop.pictures.pop(index)
    if gop.passed(index):
        return None  # the caller starts a new stretch
    try:
        while (gop.pushed <= index or index not in gop.pictures) and gop.pushed < len(gop.records):
            codec = gop.codecs[gop.pushed] if gop.codecs else gop.codec
            if codec != gop.codec:  # an H.264 chunk next to an archived AV1 one: each has its own decoder
                _keep(gop, index, gop.decoder.finish_outputs(True))
                gop.decoder, gop.codec = continuum_py.H264Decoder(codec=codec), codec
            _keep(gop, index, gop.decoder.push_outputs(gop.records[gop.pushed][1], gop.pushed, None, True))
            gop.pushed += 1
        if index not in gop.pictures:  # still held back at the stretch's end: drain it, which ends this decoder
            _keep(gop, index, gop.decoder.finish_outputs(True))
            gop.finished = True
    except ValueError:
        return None
    return gop.pictures.pop(index, None)


async def _h264_stretch(client: Client, stream: str, lo: int, hi: int) -> _Gop | None:
    """The stream's stored H.264 records in [lo, hi), any tier, in decode order, ready to decode forward.

    Decode order is not time order: with B-frames, presentation times go backwards. As continuum-tier's recent
    window does it: a Tier 1 chunk's records are stored in decode order and kept so, chunks by their start; a Tier 0
    access unit goes by its capture epoch (epochs by their first start), then its device sequence, or by its time
    when capture didn't attest them. Tier 0 follows the chunks flushed from it, and one unit in both is kept once."""
    result = await client.query(stream, lo, hi)
    entries: list[tuple[tuple[int, int, int, int], str, int, bytes]] = []
    tier0: list[tuple[Mapping[str, Any], str, int, bytes]] = []
    flip = False
    for item in result["items"]:
        header = continuum_py.decode_header(item["header"])
        if header["shape_name"] != H264:
            return None
        capture = header.get("capture") or {}
        flip = flip or bool(capture.get("flip"))
        chunk = (item.get("locator") or {}).get("kind") == "Chunk"
        records = await read.records(client, stream, item, start_ns=lo, end_ns=hi) or []
        for position, record in enumerate(records):
            got = (str(record.payload_id), record.start_ns, _unit(bytes(record.data)))
            if chunk:
                entries.append(((0, header["start_ns"], position, 0), *got))
            else:
                tier0.append((capture, *got))
    first: dict[int, int] = {}
    for capture, _id, start_ns, _data in tier0:
        epoch = int(capture.get("capture_epoch") or 0)
        first[epoch] = min(first.get(epoch, start_ns), start_ns)
    for capture, payload_id, start_ns, data in tier0:
        epoch, sequence = int(capture.get("capture_epoch") or 0), capture.get("device_seq")
        attested = epoch != 0 and sequence is not None
        key = (1, first[epoch], int(sequence) if attested else start_ns, 0 if attested else 1)
        entries.append((key, payload_id, start_ns, data))
    entries.sort(key=lambda e: e[0])
    seen: set[str] = set()
    ordered = []
    for _key, payload_id, start_ns, data in entries:
        if payload_id not in seen:
            seen.add(payload_id)
            ordered.append((start_ns, data))
    if not ordered:
        return None
    codecs = [_codec(data) for _start, data in ordered]
    if "av1" not in codecs:
        return _Gop(ordered, flip, continuum_py.H264Decoder())
    return _Gop(ordered, flip, continuum_py.H264Decoder(codec=codecs[0]), codecs, codecs[0])


async def h264_picture(client: Client, cam: str, target_ns: int, tolerance_ns: int) -> tuple[Any, int, bool] | None:
    """`cam`'s stored H.264 picture nearest `target_ns` (within `tolerance_ns`), decoded from the keyframe at or
    before it: (upright BGR image, its capture time in ns, whether capture flipped it), or None. A delta picture
    can't be decoded alone. The open decoder is kept per stream, so a later picture in the same stretch
    (a visit's face shots) decodes on from where the last one stopped."""
    import cv2
    import numpy as np

    stream = f"{cam}.video"
    for lookback_s in GOP_LOOKBACK_S:
        gops = _gops()
        gop = gops.get(stream)
        index = None if gop is None else _nearest(gop.records, target_ns, tolerance_ns)
        if gop is None or index is None or gop.passed(index):
            fresh = await _h264_stretch(
                client, stream, target_ns - to_ns(lookback_s), target_ns + to_ns(GOP_LOOKBACK_S[0]) + 1
            )
            if fresh is None:
                return None
            gop, index = fresh, _nearest(fresh.records, target_ns, tolerance_ns)
            if index is None:
                return None
        gops[stream] = gop
        gops.move_to_end(stream)
        while len(gops) > MAX_OPEN_GOPS:
            gops.popitem(last=False)
        picture = _advance(gop, index)
        if picture is not None:
            bgr, width, height = picture
            image = np.frombuffer(bgr, np.uint8).reshape(height, width, 3).copy()
            if gop.flip:
                image = cv2.rotate(image, cv2.ROTATE_180)
            return image, gop.records[index][0], gop.flip
        if gops.get(stream) is gop:
            del gops[stream]  # no keyframe early enough: try again from further back
    return None


async def clip_frames(client: Client, cam: str, start: float, end: float, fps: float) -> list[Any]:
    """`cam`'s frames over [start, end), one per 1 / fps tick: each the picture nearest its tick within half a tick
    (an upright BGR image), or None where there is none. H.264 video is read as one stretch and decoded once, forward
    from the keyframe before `start` (the second, longer lookback when the first finds none); anything else is one
    frame_at per tick. The stretch is walked in decode order and each tick takes the picture nearest it in time, so
    a camera whose decode order isn't time order (B-frames) still fills every tick; only pictures that are some
    tick's nearest are decoded to images and kept."""
    import cv2
    import numpy as np

    ticks = [start + i / fps for i in range(max(0, round((end - start) * fps)))]
    half = to_ns(0.5 / fps)
    stream = f"{cam}.video"
    for lookback_s in GOP_LOOKBACK_S:
        gop = await _h264_stretch(client, stream, to_ns(start - lookback_s), to_ns(end) + 1)
        if gop is None:
            break
        targets = [to_ns(t) for t in ticks]
        best: list[tuple[int, Any] | None] = [None] * len(ticks)  # (distance, image): a slow camera's picture
        for index, (at_ns, _data) in enumerate(gop.records):  # serves every tick nearest it
            near = [
                k
                for k, target in enumerate(targets)
                if abs(at_ns - target) <= half and ((b := best[k]) is None or abs(at_ns - target) < b[0])
            ]
            if not near:
                continue
            # Asking in decode order keeps every later picture the decoder emits early, so none is passed by.
            picture = None if gop.passed(index) else _advance(gop, index)
            if picture is None:
                continue
            bgr, width, height = picture
            image = np.frombuffer(bgr, np.uint8).reshape(height, width, 3).copy()
            if gop.flip:
                image = cv2.rotate(image, cv2.ROTATE_180)
            for k in near:
                best[k] = (abs(at_ns - targets[k]), image)
        out = [None if b is None else b[1] for b in best]
        if any(f is not None for f in out):
            return out
    try:
        kept = (await client.query(stream, to_ns(start), to_ns(end) + 1, all=False, max_items=1))["items"]
    except continuum_py.ContinuumError:
        return [None] * len(ticks)
    if not kept:
        return [None] * len(ticks)
    return [await frame_at(client, cam, t, 0.5 / fps) for t in ticks]


async def _h264_evidence(client: Client, cam: str, target_ns: int) -> FrameEvidence | None:
    got = await h264_picture(client, cam, target_ns, 1000)
    if got is None:
        return None
    image, at_ns, flip = got
    height, width = image.shape[:2]
    return FrameEvidence(
        image,
        at_ns,
        {
            "capturedAtUs": at_ns // 1000,
            "width": width,
            "height": height,
            "flip": flip,
            "profileStatus": "absent",
            "nativeGeometry": {"width": width, "height": height},
        },
    )


async def frame_evidence_at(client: Client, cam: str, ts: float) -> FrameEvidence | None:
    """Exact source-crop evidence; at most one microsecond compensates float timestamp roundoff.

    Old SDKs cannot decode rich archived evidence and fail closed. General nearest-frame playback remains
    available through frame_at, but is never substituted for detection-bound cropping.
    """

    from liberational.see.frames import decode_frame, source_content_rect

    target = to_ns(ts)
    lo, hi = max(0, target - 1000), target + 1001
    found: list[FrameEvidence] = []
    try:
        result = await client.query(f"{cam}.video", lo, hi, all=False, max_items=8)
        if result.get("truncated"):
            return None
        items = result["items"]
        if items and continuum_py.decode_header(items[0]["header"])["shape_name"] == H264:
            return await _h264_evidence(client, cam, target)
        for item in items:
            header = continuum_py.decode_header(item["header"])
            locator = item["locator"]
            if locator is None:
                continue
            if locator["kind"] == "Chunk":
                get_chunk = getattr(client, "get_chunk", None)
                if get_chunk is None:
                    continue
                before = await get_chunk(header["id"])
                if locator not in before["locations"]:
                    continue
                matches = [start for start, cut in before["frames"] if not cut and abs(start - target) <= 1000]
                if len(matches) != 1:
                    continue
                actual = matches[0]
                records = await read.records(client, f"{cam}.video", item, start_ns=actual, end_ns=actual + 1) or []
                exact = [record for record in records if record.start_ns == actual]
                if len(exact) != 1:
                    continue
                image = decode_frame(header["shape_name"], exact[0].data, header.get("capture"))
                if image is None:
                    continue
                at_ns = actual
                height, width = image.shape[:2]
                flip = bool((header.get("capture") or {}).get("flip"))
                after = await get_chunk(header["id"])
                if before != after:
                    continue
                metadata = {
                    "capturedAtUs": at_ns // 1000,
                    "width": width,
                    "height": height,
                    "flip": flip,
                    "profileStatus": "absent",
                    "nativeGeometry": {
                        "width": (header.get("capture") or {}).get("width"),
                        "height": (header.get("capture") or {}).get("height"),
                    },
                }
                if source_content_rect(metadata) is not None:
                    found.append(FrameEvidence(image, at_ns, metadata))
                continue
            for record in await read.records(client, f"{cam}.video", item, start_ns=lo, end_ns=hi) or []:
                if abs(record.start_ns - target) > 1000:
                    continue
                image = decode_frame(header["shape_name"], record.data, header.get("capture"))
                if image is not None:
                    height, width = image.shape[:2]
                    found.append(
                        FrameEvidence(
                            image,
                            record.start_ns,
                            {
                                "capturedAtUs": record.start_ns // 1000,
                                "width": width,
                                "height": height,
                                "flip": bool((header.get("capture") or {}).get("flip")),
                                "profileStatus": "absent",
                                "nativeGeometry": {"width": width, "height": height},
                            },
                        )
                    )
    except continuum_py.ContinuumError, OSError, ValueError, NotImplementedError:
        return None
    return found[0] if len(found) == 1 else None


async def frame_at(client: Client, cam: str, ts: float, within_s: float = 0.001) -> Any:
    """`cam`'s frame captured nearest `ts` (within `within_s`: a millisecond, for a frame named by its time; a second,
    say, for a moment), decoded to a BGR image, or None."""
    from liberational.see.frames import decode_frame

    stream = f"{cam}.video"
    lo, hi = to_ns(ts) - to_ns(within_s), to_ns(ts) + to_ns(within_s)
    try:
        result = await client.query(stream, lo, hi)
    except continuum_py.ContinuumError:
        return None
    if result["items"] and continuum_py.decode_header(result["items"][0]["header"])["shape_name"] == H264:
        try:
            got = await h264_picture(client, cam, to_ns(ts), to_ns(within_s))
        except continuum_py.ContinuumError, ValueError:
            return None
        return None if got is None else got[0]
    best: tuple[int, bytes, PayloadHeader] | None = None
    for item in result["items"]:
        header = continuum_py.decode_header(item["header"])
        for rec in await read.records(client, stream, item, start_ns=lo, end_ns=hi) or []:
            off = abs(rec.start_ns - to_ns(ts))
            if best is None or off < best[0]:
                best = (off, rec.data, header)
    if best is None:
        return None
    _, data, header = best
    return decode_frame(header["shape_name"], data, header.get("capture"))


async def video_spans(client: Client, cam: str, start: float, end: float, gap_s: float = 2.5) -> list[list[float]]:
    """The stretches of [start, end) where `cam` has video on the broker (`<cam>.video`, any tier), as [a, b] seconds,
    oldest first: its payloads' spans, joined across gaps up to `gap_s` (a frame or a GOP apart is one stretch)."""
    try:
        result = await client.query(f"{cam}.video", to_ns(start), to_ns(end))
    except continuum_py.ContinuumError:
        return []
    out: list[list[float]] = []
    for item in result["items"]:
        h = continuum_py.decode_header(item["header"])
        a, b = max(from_ns(h["start_ns"]), start), min(from_ns(max(h["end_ns"], h["start_ns"])), end)
        if b < a:
            continue
        if out and a - out[-1][1] <= gap_s:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


async def video_days(client: Client) -> list[str]:
    """The local days (YYYY-MM-DD) on which any camera has video on the broker, oldest first: each camera's first
    frame, then one item asked per day from there to today. Each asks for one item only (`all=False`: the default
    pages through every frame, ignoring max_items, which made /review/days take seconds)."""
    import datetime as dt

    days: set[str] = set()
    now = time.time()
    for cam in await video_cameras(client):
        stream = f"{cam}.video"
        try:
            first = (await client.query(stream, 0, to_ns(now + 1), all=False, max_items=1))["items"]
        except continuum_py.ContinuumError:
            continue
        if not first:
            continue
        day = dt.date.fromtimestamp(from_ns(continuum_py.decode_header(first[0]["header"])["start_ns"]))
        while day <= dt.date.fromtimestamp(now):
            if day.isoformat() not in days:
                start = dt.datetime.combine(day, dt.time()).timestamp()
                found = await client.query(stream, to_ns(start), to_ns(start + 86400), all=False, max_items=1)
                if found["items"]:
                    days.add(day.isoformat())
            day += dt.timedelta(days=1)
    return sorted(days)


async def has_video(client: Client, start: float, end: float) -> bool:
    """Whether any camera has a frame on the broker in [start, end) (seconds): one item asked per camera."""
    for cam in await video_cameras(client):
        try:
            found = await client.query(f"{cam}.video", to_ns(start), to_ns(end), all=False, max_items=1)
        except continuum_py.ContinuumError:
            continue
        if found["items"]:
            return True
    return False


async def video_cameras(client: Client) -> list[str]:
    """The cameras with a video stream on the broker (`<cam>.video`), by name."""
    return sorted(s["name"].removesuffix(".video") for s in await client.list_streams() if s["name"].endswith(".video"))


_LOGS: dict[Path, VisionLog] = {}
_LOGS_LOCK = threading.Lock()


def at(state: str | Path | None = None) -> VisionLog:
    """The vision log of the broker that `state` (a pack's state directory) belongs to: `continuum/broker.sock` under
    it, or `continuum_link.socket_path()` for the pack's own (and by default). One per socket, shared, made on first
    use; nothing connects until a read. Tests replace this with an in-memory log."""
    from liberational import config as cfg

    own = state is None or Path(state).resolve() == Path(cfg.path("paths.state")).resolve()
    socket = continuum_link.socket_path() if own else Path(state) / "continuum" / "broker.sock"
    with _LOGS_LOCK:
        log = _LOGS.get(socket)
        if log is None:
            log = _LOGS[socket] = VisionLog(socket, state=None if own else Path(state))
        return log
