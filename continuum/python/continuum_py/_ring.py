"""next_record and the async iterator for RingReader, on the event loop.

The read happens here, in the coroutine, after the wait: a cancelled await raises at the await, before reading, so
cancelling never loses a record. The wait itself (RingReader._ready) runs on a tokio blocking thread through the ring's
sequence word, so it holds no GIL and doesn't poll.
"""

import asyncio
from collections.abc import AsyncIterator

from continuum_py._native import RingReader

# How long one readiness wait lasts when the caller gave no timeout. A publish ends it at once; this only bounds how
# often an idle follower goes around its loop.
_IDLE_WAIT_MS = 1000


async def next_record(reader: RingReader, timeout_ms: int | None = None) -> tuple[int, bytes, int] | None:
    loop = asyncio.get_running_loop()
    deadline = None if timeout_ms is None else loop.time() + timeout_ms / 1000
    while True:
        record = reader.read_record()
        if record is not None:
            return record
        if deadline is None:
            wait_ms = _IDLE_WAIT_MS
        else:
            wait_ms = round((deadline - loop.time()) * 1000)
            if wait_ms <= 0:
                return None
        await reader._ready(wait_ms)


async def follow(reader: RingReader) -> AsyncIterator[tuple[int, bytes, int]]:
    while True:
        record = await next_record(reader)
        assert record is not None  # no timeout, so it waits until there is one
        yield record
