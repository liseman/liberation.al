"""Reading a query item's bytes from wherever its locator says they are, for consumers that query and then read.

`records` turns any query item into the payloads it holds, each with its id, span, and bytes:

- Inline: one payload, the bytes in the locator.
- Shm (Tier 0): one payload in a slot of the broker's arena, copied under a lease, and kept only if the lease was
  still valid after the copy (a lapsed lease means the slot may have been reused mid-copy).
- Chunk (Tier 1 or 2): the tier worker flushed a stream's payloads into a chunk file, and the catalog keeps the
  chunk as one payload whose locator names the whole file; its records are the payloads it absorbed, each with its
  own id. They're read straight from the local file (`read_chunk_records`), records a purge cut left out. A FLAC
  file registered in place is one payload, the whole file.
- Ring: a batch of ring records; use `Client.ring_reader` for those.

A locator can go stale between the query and the read: a flush frees a slot (the payload is in a chunk now), or a
purge takes a whole chunk. Then the item is asked for again by id, once, over its own span, and read from where it is
now. An item that's gone by then (purged) reads as None. Stale bytes are never returned.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

from ._native import NotReadyError, StaleError, decode_header, read_chunk_records

if TYPE_CHECKING:
    from ._native import Client
    from .schema import Item, Locator

_NIL = "00000000-0000-0000-0000-000000000000"

# A lease the broker refuses because the slot was freed, reused, or isn't a live slot: the payload moved on.
_SLOT_MOVED = (StaleError, NotReadyError, ValueError)


class PcmFormat(NamedTuple):
    """Exact source format retained by new microphone archives, independent of capture timing jitter."""

    rate_hz: int
    channels: int
    stride: int


@dataclass(frozen=True)
class Record:
    """One payload's bytes, in its stream's native form, with its id and span (UTC nanoseconds, half open)."""

    payload_id: str
    start_ns: int
    end_ns: int
    data: bytes
    pcm: PcmFormat | None = None


class StaleLocator(LookupError):
    """An item's locator went stale twice running: it moved again between the second query and the read."""


async def records(
    client: Client, stream: str, item: Item, *, start_ns: int = 0, end_ns: int = 2**63 - 1
) -> list[Record] | None:
    """The payloads a query `item` on `stream` holds, oldest first: one for an inline or slot item, a chunk's
    records for a chunk (only those overlapping [start_ns, end_ns)). An empty list for a header-only payload (no
    bytes), and None when the item is gone (purged since the query). Raises ValueError for a locator this can't read
    (a ring batch, a chunk that isn't a local file) and StaleLocator if the item moves again during the retry."""
    header = decode_header(item["header"])
    if item["locator"] is None:
        return []
    got = await _read(client, item, start_ns, end_ns)
    if got is not None:
        return got
    # Stale: look again over the item's own span (a short ask). It's there under its own id, or, once a flush
    # absorbed it, as a record of a chunk there (an absorbed payload is no longer a row of its own).
    want = header["id"]
    span = (header["start_ns"], max(header["end_ns"], header["start_ns"] + 1))
    result = await client.query(stream, *span, all=True)
    for again in result["items"]:
        if decode_header(again["header"])["id"] == want:
            if again["locator"] is None:
                return []
            got = await _read(client, again, start_ns, end_ns)
            if got is None:
                raise StaleLocator(f"{want} on {stream} moved again while it was being read")
            return got
    for again in result["items"]:
        locator = again["locator"]
        if locator is not None and locator["kind"] == "Chunk":
            got = await _read(client, again, *span)
            if got is None:
                raise StaleLocator(f"{want} on {stream} moved again while it was being read")
            if mine := [r for r in got if r.payload_id == want]:
                return mine
    return None


async def _read(client: Client, item: Item, start_ns: int, end_ns: int) -> list[Record] | None:
    """The records at `item`'s locator, or None when it went stale."""
    locator = item["locator"]
    assert locator is not None
    header = decode_header(item["header"])
    whole = (header["id"], header["start_ns"], header["end_ns"])
    kind = locator["kind"]
    if kind == "Inline":
        return [Record(*whole, bytes(locator["bytes"]))]
    if kind == "Shm":
        data = _read_slot(client, locator)
        return None if data is None else [Record(*whole, data)]
    if kind == "Chunk":
        # A camera's chunk is committed in parts while its file is still being written, so the file can hold records
        # past what the item names: its next part, whose payloads are still Tier 0 items of their own. Reading only
        # the item's own span keeps a caller that walks a query's items from getting those twice.
        start_ns = max(start_ns, header["start_ns"])
        end_ns = min(end_ns, max(header["end_ns"], header["start_ns"] + 1))
        if start_ns >= end_ns:
            return []
        # File I/O (and LZ4 or FLAC for some streams): off the event loop.
        found = await asyncio.to_thread(read_chunk_records, _local(locator["uri"]), start_ns, end_ns)
        if found is None:
            return None
        if len(found) == 1 and found[0][:3] == (_NIL, 0, 0):
            # A FLAC file registered in place: the item is its one-frame chunk, and the file holds no record
            # ids, so the record takes the item's id and span.
            return [Record(*whole, found[0][3])]
        return [
            Record(r[0], r[1], r[2], r[3], PcmFormat(*r[4]) if len(r) > 4 and r[4] is not None else None) for r in found
        ]
    raise ValueError(f"a {kind} locator isn't a payload's bytes (read ring batches with Client.ring_reader)")


def _local(uri: str) -> str:
    """A chunk's uri as an absolute path. A tier worker given a relative directory wrote relative paths (before it
    made them absolute): those are under the services' working directory, the checkout, which is this process's too."""
    path = uri.removeprefix("file://")
    if "://" in path or os.path.isabs(path):
        return uri
    return os.path.abspath(path)


def _read_slot(client: Client, locator: Locator) -> bytes | None:
    """A Tier 0 slot's bytes, copied under a lease; None when the slot moved on (freed, reused, its arena recreated)
    or the lease lapsed before the copy was done (the slot may have been reused under it)."""
    try:
        arena = client.arena(locator["arena"])
    except KeyError:  # an arena this session doesn't map (a recreated one, say)
        return None
    if arena.generation != locator["arena_generation"]:
        return None
    try:
        lease = arena.lease(locator["slab_class"], locator["slot"], locator["slot_generation"])
    except _SLOT_MOVED:
        return None
    data = bytes(memoryview(lease)[: locator["length"]])
    # release() says whether the lease was still valid: a copy made after it lapsed can't be trusted.
    return data if lease.release() else None
