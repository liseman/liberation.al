"""`face.boxes`: the face worker's live face boxes, in a Tier 0 ring that never reaches the catalog (#1762).

Every analyzed frame with a face gets one record: the camera, the frame's id, sequence and start, and each face's box
with who it is. Only live readers use it (the gateway's Live overlay, capture's exposure steering, the body worker's
face hints), so it is a fixed-stride ring of `SECONDS` at up to `RATE_HZ` records, and the writer sends no ingest
notice: no catalog row, no archive copy, and nothing to age out. Recorded video's overlays read `face.detection` and
`face.match`.

A record is a little-endian u32 length and then that many bytes of JSON, zero-padded to `STRIDE`. A frame whose faces
don't fit keeps its largest boxes.
"""

import asyncio
import json
import struct
from collections.abc import AsyncGenerator
from typing import Any

from continuum_py import BrokerError, Client, RingWriter

STREAM = "face.boxes"
STRIDE = 2048
RATE_HZ = 64.0  # records a second, across every camera, the ring is sized for
SECONDS = 60.0
RETRY_S = 1.0
_LEN = struct.Struct("<I")


def encode(record: dict[str, Any]) -> bytes:
    """One ring record for `record`; faces that don't fit go, smallest box first."""
    faces = sorted(record.get("faces", []), key=lambda f: -_area(f.get("bbox")))
    while True:
        body = json.dumps({**record, "faces": faces}, separators=(",", ":")).encode()
        if _LEN.size + len(body) <= STRIDE or not faces:
            return _LEN.pack(len(body)) + body
        faces = faces[:-1]


def decode(data: bytes) -> dict[str, Any] | None:
    """The record a ring slot holds, or None for an empty, torn, or foreign one."""
    if len(data) < _LEN.size:
        return None
    (n,) = _LEN.unpack_from(data)
    if not 0 < n <= len(data) - _LEN.size:
        return None
    try:
        record = json.loads(data[_LEN.size : _LEN.size + n])
    except ValueError:
        return None
    return record if isinstance(record, dict) else None


def _area(bbox: Any) -> float:
    if not isinstance(bbox, list) or len(bbox) < 4:
        return 0.0
    try:
        return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
    except TypeError, ValueError:
        return 0.0


async def writer(client: Client) -> RingWriter:
    """Registers the ring stream (the same ring on every registration) and takes its writer."""
    registered = await client.register_stream(
        STREAM,
        kind="derived",
        format="Text",
        shape_name=STREAM,
        nominal_rate_hz=RATE_HZ,
        ring_stride_bytes=STRIDE,
        ring_seconds=SECONDS,
        layer="Layer2Decision",
    )
    lease = registered["lease"]
    if lease is None:
        raise RuntimeError(f"the broker gave {STREAM} no ring")
    return client.ring_writer(lease)


async def follow(client: Client) -> AsyncGenerator[dict[str, Any]]:
    """Every record written from now on, as it lands. Waits for the face worker to register the ring; a reader that
    falls behind the ring skips to the oldest record still in it."""
    while True:
        try:
            sub = await client.subscribe([STREAM], start="Live")
        except BrokerError:
            await asyncio.sleep(RETRY_S)  # no face worker has registered it yet
            continue
        try:
            if not sub.rings:
                raise RuntimeError(f"{STREAM} has no ring")
            reader = client.ring_reader(sub.rings[0]["lease"])
            async for _, data, _ in reader:
                if (record := decode(data)) is not None:
                    yield record
        finally:
            await sub.unsubscribe()


async def between(client: Client, start_ns: int, end_ns: int) -> list[dict[str, Any]]:
    """The records still in the ring (the last `SECONDS` or so) whose frame starts in `[start_ns, end_ns]`, oldest
    first; none before the face worker registered the ring."""
    try:
        sub = await client.subscribe([STREAM], start="Live")
    except BrokerError:
        return []
    try:
        if not sub.rings:
            return []
        reader = client.ring_reader(sub.rings[0]["lease"], from_oldest=True)
        out = []
        while (got := reader.read_record()) is not None:
            record = decode(got[1])
            if record is not None and isinstance(at := record.get("start_ns"), int) and start_ns <= at <= end_ns:
                out.append(record)
        return out
    finally:
        await sub.unsubscribe()


def near(state: Any, start_ns: int, end_ns: int) -> list[dict[str, Any]]:
    """`between` from synchronous code, over the vision log's broker link for `state`."""
    from liberational import vision_log

    return vision_log.at(state).link.call(lambda c: between(c, start_ns, end_ns))
