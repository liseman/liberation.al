"""Similarity and full-text search through the vector worker (docs/CONTINUUM-STORAGE.md).

A Searcher commits a query payload on `vector.query` or `text.query` and waits for the vector worker's result, the
`vector.search` or `text.search` payload whose parent is the query. Until the SDK has the consumer's promise calls,
the worker follows the query streams live and answers on its own, so a query nobody answers (the worker isn't
running, or the query was refused) ends in a TimeoutError after `timeout` seconds rather than a hang.

    async with Searcher(client) as search:
        hits = await search.nearest("campp", "v1", vector, k=20, subjects=[person], timeout=0.5)

Hits come nearest first: `score` is cosine distance for a vector search (lower is nearer) and relevance for a text
search (higher is better). Each carries its payload's id, entity (first subject), parent (first parent), and span.
"""

import asyncio
import contextlib
import math
import secrets
import time
import uuid
from collections.abc import Iterable, Sequence
from types import TracebackType
from typing import Self, TypedDict

from continuum_py._native import (
    BrokerError,
    Client,
    Subscription,
    decode_header,
    decode_hits,
    encode_header,
    encode_text_query,
    encode_vector_query,
)
from continuum_py.schema import PayloadHeader

VECTOR_QUERY = "vector.query"
VECTOR_SEARCH = "vector.search"
TEXT_QUERY = "text.query"
TEXT_SEARCH = "text.search"
MAX_K = 40
"""The most hits one query asks for: what keeps a result an inline payload."""


def uuid7() -> str:
    """A new UUIDv7 (RFC 9562): 48 bits of Unix milliseconds, then random bits. uuid.uuid7 needs Python 3.14, and the
    SDK supports 3.12."""
    value = (time.time_ns() // 1_000_000) << 80 | int.from_bytes(secrets.token_bytes(10))
    value = (value & ~(0xF << 76)) | (0x7 << 76)  # version 7
    value = (value & ~(0x3 << 62)) | (0x2 << 62)  # the RFC 4122 variant
    return str(uuid.UUID(int=value))


def check_k(k: int) -> None:
    """Raises ValueError unless 1 <= k <= MAX_K."""
    if not 1 <= k <= MAX_K:
        raise ValueError(f"k is {k}; a query asks for 1 to {MAX_K} hits")


def check_during(during: tuple[int, int] | None) -> None:
    """Raises ValueError for a time range that's negative, empty, or backwards."""
    if during is not None and not 0 <= during[0] < during[1]:
        raise ValueError(f"a time range is [start, end) in UTC ns with start < end, not {during}")


class Hit(TypedDict):
    """One search result."""

    payload_id: str
    entity_id: str | None
    parent_id: str | None
    start_ns: int
    end_ns: int
    score: float


class SearchUnavailable(TimeoutError):
    """The search streams don't exist yet: no vector worker has registered them. A TimeoutError, so a caller that
    handles a slow answer handles this too."""


class Searcher:
    """Searches through the vector worker on one client. Use it as an async context manager, or call start and close.
    One Searcher serves any number of concurrent queries."""

    def __init__(self, client: Client) -> None:
        self._client = client
        self._streams: dict[str, int] = {}
        self._waiting: dict[str, asyncio.Future[list[Hit]]] = {}
        self._results: Subscription | None = None
        self._reader: asyncio.Task[None] | None = None
        self._ended: BaseException | None = None

    async def start(self) -> None:
        """Registers the query streams the way the vector worker does and follows the result streams.

        Raises SearchUnavailable when no vector worker has registered the result streams yet."""
        if self._reader is not None:
            raise RuntimeError("this Searcher is started already")
        self._ended = None
        for name in (VECTOR_QUERY, TEXT_QUERY):
            registered = await self._client.register_stream(
                name,
                kind="derived",
                format="ArrowRecordBatch",
                shape_name=name,
                shape_version=1,
                slab_class=0,
                layer="Layer1Raw",
            )
            self._streams[name] = registered["stream_id"]
        try:
            self._results = await self._client.subscribe([VECTOR_SEARCH, TEXT_SEARCH], start="Live", pinned=True)
        except BrokerError as e:
            raise SearchUnavailable(f"no vector worker has registered the search streams: {e}") from e
        self._reader = asyncio.create_task(self._read(self._results), name="continuum search results")

    async def close(self) -> None:
        """Stops following results; queries still waiting get a TimeoutError."""
        self._fail_waiting(TimeoutError("the searcher closed"))
        reader, results = self._reader, self._results
        self._reader = self._results = None
        if reader is not None:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
        if results is not None:
            # The connection may be gone already, which ends the subscription anyway.
            with contextlib.suppress(ConnectionError, BrokerError):
                await results.unsubscribe()

    def _fail_waiting(self, error: BaseException) -> None:
        for waiting in self._waiting.values():
            if not waiting.done():
                waiting.set_exception(error)
        self._waiting.clear()

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(
        self, kind: type[BaseException] | None, error: BaseException | None, tb: TracebackType | None
    ) -> None:
        await self.close()

    async def nearest(
        self,
        model: str,
        version: str,
        vector: Sequence[float],
        k: int = 20,
        *,
        during: tuple[int, int] | None = None,
        subjects: Iterable[str] = (),
        labelled: bool | None = None,
        timeout: float = 0.5,
    ) -> list[Hit]:
        """The k nearest embeddings to `vector` in the space `embedding/<model>/<version>`, among the rows whose
        span overlaps `during` (UTC ns, half open), whose entity is one of `subjects`, and that have an entity
        (`labelled=True`) or don't (`False`). Raises TimeoutError when no answer comes within `timeout` seconds, and
        ValueError at once for a query the worker would refuse."""
        check_k(k)
        check_during(during)
        if not vector or not all(math.isfinite(x) for x in vector):
            raise ValueError("a query vector needs at least one value, every one finite")
        query = encode_vector_query(
            model, version, list(vector), k, during=during, subjects=list(subjects), labelled=labelled
        )
        return await self._ask(VECTOR_QUERY, query, timeout)

    async def text_search(
        self,
        text: str,
        k: int = 20,
        *,
        during: tuple[int, int] | None = None,
        subjects: Iterable[str] = (),
        labelled: bool | None = None,
        timeout: float = 0.5,
    ) -> list[Hit]:
        """Full-text search over interpretation text, most relevant first, with the same filters as nearest."""
        check_k(k)
        check_during(during)
        query = encode_text_query(text, k, during=during, subjects=list(subjects), labelled=labelled)
        return await self._ask(TEXT_QUERY, query, timeout)

    async def _ask(self, stream: str, query: bytes, timeout: float) -> list[Hit]:
        if self._reader is None:
            raise RuntimeError("start the Searcher first (or use it with async with)")
        if self._reader.done():
            raise ConnectionError("this Searcher stopped receiving results") from self._ended
        query_id = uuid7()
        now = time.time_ns()
        header: PayloadHeader = {
            "id": query_id,
            "stream_id": self._streams[stream],
            "start_ns": now,
            "end_ns": now,
            # Two queries in the same nanosecond on one stream need different seqs.
            "seq": secrets.randbits(32),
            "layer": "Layer1Raw",
            "format": "ArrowRecordBatch",
            "shape_name": stream,
            "shape_version": 1,
        }
        answer: asyncio.Future[list[Hit]] = asyncio.get_running_loop().create_future()
        # Waiting before the query goes out, so an answer can't arrive first and be missed.
        self._waiting[query_id] = answer
        try:
            self._client.ingest(encode_header(header), inline=query)
            return await asyncio.wait_for(answer, timeout)
        finally:
            self._waiting.pop(query_id, None)

    async def _read(self, results: Subscription) -> None:
        """Hands each result to the query waiting for it. When the results end (the connection closed) or this fails,
        every query still waiting gets the cause at once, not a timeout."""
        try:
            await self._dispatch(results)
            self._ended = ConnectionError("the search results ended (the connection closed?)")
        except asyncio.CancelledError:
            raise
        except Exception as e:  # anything: the waiters must hear of it, not wait out their timeouts
            self._ended = e
        self._fail_waiting(ConnectionError(f"this Searcher stopped receiving results: {self._ended}"))

    async def _dispatch(self, results: Subscription) -> None:
        async for event in results:
            item = event.get("item")
            if event["type"] != "Event" or item is None:
                continue
            parents = decode_header(item["header"]).get("parents", [])
            waiting = self._waiting.get(parents[0]) if parents else None
            if waiting is None or waiting.done():
                continue  # another client's answer, or one that came after its query gave up
            locator = item["locator"]
            if locator is None or locator["kind"] != "Inline":
                waiting.set_exception(ValueError("a search result that isn't inline"))
                continue
            try:
                hits: list[Hit] = decode_hits(locator["bytes"])
            except ValueError as e:
                waiting.set_exception(e)
                continue
            waiting.set_result(hits)
