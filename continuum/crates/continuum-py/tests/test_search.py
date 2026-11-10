"""continuum_py.search against a real broker, with a stand-in vector worker in Python: a query is answered by the
result whose parent is it, concurrent queries each get their own answer, a query nobody answers times out, and a
Searcher with no vector worker registered says so. The worker's own side is tested in continuum-vector.

Run with `mise run continuum-test`, which builds the broker first.
"""

import asyncio
import math
import time
import uuid

import pytest
from continuum_py import Client, decode_header, encode_header
from continuum_py._native import encode_hits, encode_vector_query
from continuum_py.search import (
    TEXT_QUERY,
    TEXT_SEARCH,
    VECTOR_QUERY,
    VECTOR_SEARCH,
    Hit,
    Searcher,
    SearchUnavailable,
    uuid7,
)


def hit(n: int, *, parent: str | None = None) -> Hit:
    return {
        "payload_id": str(uuid.UUID(int=n)),
        "entity_id": None,
        "parent_id": parent,
        "start_ns": n,
        "end_ns": n + 1,
        "score": n / 8,  # exact in f32, which hits carry
    }


async def stand_in_worker(socket: str, answers: dict[int, list[Hit]]) -> tuple[Client, asyncio.Task[None]]:
    """Registers the search streams as the vector worker does and answers each vector query with the hits filed
    under its seq, or doesn't answer one with no entry."""
    worker = await Client.connect(socket, client_name="vector")
    streams = {}
    for name, layer in (
        (VECTOR_QUERY, "Layer1Raw"),
        (TEXT_QUERY, "Layer1Raw"),
        (VECTOR_SEARCH, "Layer2Decision"),
        (TEXT_SEARCH, "Layer2Decision"),
    ):
        registered = await worker.register_stream(
            name, kind="derived", format="ArrowRecordBatch", shape_name=name, slab_class=0, layer=layer
        )
        streams[name] = registered["stream_id"]
    queries = await worker.subscribe([VECTOR_QUERY], start="Live", pinned=True)

    async def answer() -> None:
        async for event in queries:
            item = event.get("item")
            if item is None:
                continue
            query = decode_header(item["header"])
            hits = answers.get(query["seq"])
            if hits is None:
                continue
            now = time.time_ns()
            header = encode_header(
                {
                    "id": uuid7(),
                    "stream_id": streams[VECTOR_SEARCH],
                    "start_ns": now,
                    "end_ns": now,
                    "layer": "Layer2Decision",
                    "format": "ArrowRecordBatch",
                    "shape_name": VECTOR_SEARCH,
                    "parents": [query["id"]],
                }
            )
            worker.ingest(header, inline=encode_hits(hits))

    return worker, asyncio.create_task(answer())


def test_a_query_gets_the_result_whose_parent_it_is(socket: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # The Searcher picks each query's seq at random; pin them so the stand-in knows which query is which.
    seqs = iter([1, 2, 3])
    monkeypatch.setattr("continuum_py.search.secrets.randbits", lambda _bits: next(seqs))
    first, second = [hit(1, parent=str(uuid.UUID(int=99))), hit(2)], [hit(5)]

    async def run() -> tuple[list[list[Hit]], bool]:
        worker, answering = await stand_in_worker(socket, {1: first, 2: second})
        consumer = await Client.connect(socket, client_name="consumer")
        async with Searcher(consumer) as search:
            # Two at once: each gets its own answer, not the other's.
            got = await asyncio.gather(
                search.nearest("arcface", "r100", [1.0, 0.0], k=5),
                search.nearest("arcface", "r100", [0.0, 1.0], k=5),
            )
            # Seq 3 has no answer: the wait ends in a TimeoutError, not a hang.
            try:
                await search.nearest("arcface", "r100", [1.0, 0.0], timeout=0.2)
                timed_out = False
            except TimeoutError:
                timed_out = True
        answering.cancel()
        await consumer.close()
        await worker.close()
        return list(got), timed_out

    got, timed_out = asyncio.run(run())
    assert got == [first, second]
    assert timed_out


def test_a_searcher_with_no_vector_worker_says_so(socket: str) -> None:
    async def run() -> None:
        consumer = await Client.connect(socket, client_name="consumer")
        try:
            await Searcher(consumer).start()
        finally:
            await consumer.close()

    with pytest.raises(SearchUnavailable):
        asyncio.run(run())


def test_a_query_asks_for_1_to_40_hits() -> None:
    assert encode_vector_query("arcface", "r100", [1.0], 40)
    for k in (0, 41):
        with pytest.raises(ValueError, match="1 to 40"):
            encode_vector_query("arcface", "r100", [1.0], k)
    with pytest.raises(ValueError, match="bad id"):
        encode_vector_query("arcface", "r100", [1.0], 5, subjects=["not an id"])


def test_a_waiting_query_hears_at_once_when_the_results_end(socket: str) -> None:
    async def run() -> tuple[type[BaseException] | None, float]:
        worker, answering = await stand_in_worker(socket, {})  # answers nothing
        consumer = await Client.connect(socket, client_name="consumer")
        search = Searcher(consumer)
        await search.start()
        waiting = asyncio.create_task(search.nearest("arcface", "r100", [1.0], timeout=10))
        await asyncio.sleep(0.2)
        started = time.monotonic()
        await consumer.close()  # the results subscription ends with the connection
        raised: type[BaseException] | None = None
        try:
            await waiting
        except ConnectionError as e:
            raised = type(e)
        took = time.monotonic() - started
        answering.cancel()
        await search.close()
        await worker.close()
        return raised, took

    raised, took = asyncio.run(run())
    assert raised is not None
    assert issubclass(raised, ConnectionError)
    assert took < 5, "it heard well before its 10 s timeout"


@pytest.mark.parametrize(
    ("vector", "k", "during"),
    [
        pytest.param([], 5, None, id="empty-vector"),
        pytest.param([1.0, math.nan], 5, None, id="nan"),
        pytest.param([1.0], 0, None, id="k-0"),
        pytest.param([1.0], -1, None, id="k-negative"),
        pytest.param([1.0], 41, None, id="k-41"),
        pytest.param([1.0], 5, (9, 5), id="backwards"),
        pytest.param([1.0], 5, (-1, 5), id="negative-time"),
    ],
)
def test_a_query_the_worker_would_refuse_fails_at_once(
    socket: str, vector: list[float], k: int, during: tuple[int, int] | None
) -> None:
    async def run() -> None:
        worker, answering = await stand_in_worker(socket, {})
        consumer = await Client.connect(socket, client_name="consumer")
        try:
            async with Searcher(consumer) as search:
                await search.nearest("arcface", "r100", vector, k, during=during, timeout=10)
        finally:
            answering.cancel()
            await consumer.close()
            await worker.close()

    with pytest.raises(ValueError):
        asyncio.run(run())


def test_query_ids_are_uuid7_and_rise_with_time() -> None:
    first, second = uuid7(), uuid7()
    for u in (first, second):
        parsed = uuid.UUID(u)
        assert (parsed.version, parsed.variant) == (7, uuid.RFC_4122)
    assert first != second
    # The leading 48 bits are Unix milliseconds: within a second of now.
    assert abs(int(first.replace("-", "")[:12], 16) - time.time_ns() // 1_000_000) < 1000
