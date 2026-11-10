"""Time queries from Python against a real broker: one page, every item paged by the cursor in time order, a range
filter, and argument checks.

Run with `mise run continuum-test`, which builds the broker first.
"""

import asyncio
from pathlib import Path

import pytest
from continuum_py import Client, QueryResult, ShmLocation, decode_header, encode_header


def uuid_of(n: int) -> str:
    raw = bytearray(16)
    raw[6], raw[8], raw[15] = 0x70, 0x80, n
    h = raw.hex()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


async def events(socket: str, starts: list[int]) -> Client:
    """A client with "events" registered and payload n (1-based) ingested inline at starts[n - 1], committed."""
    c = await Client.connect(socket)
    stream = (
        await c.register_stream(
            "events", kind="derived", format="Text", shape_name="event", layer="Layer2Decision", slab_class=0
        )
    )["stream_id"]
    live = await c.subscribe(["events"])
    for n, start in enumerate(starts, 1):
        header = encode_header(
            {
                "id": uuid_of(n),
                "stream_id": stream,
                "start_ns": start,
                "end_ns": start + 10,
                "seq": n,
                "layer": "Layer2Decision",
                "format": "Text",
                "shape_name": "event",
            }
        )
        c.ingest(header, inline=bytes([n]))
    for _ in starts:
        assert await asyncio.wait_for(live.next(), timeout=5) is not None
    await live.unsubscribe()
    return c


def numbers(result: QueryResult) -> list[int]:
    return [int(decode_header(it["header"])["id"][-2:], 16) for it in result["items"]]


def test_a_time_query_and_every_item_in_time_order(socket: str) -> None:
    async def run() -> tuple[QueryResult, QueryResult, QueryResult]:
        # Committed out of time order, as an import would be.
        c = await events(socket, [300, 100, 200, 200, 400])
        one = await asyncio.wait_for(c.query("events", 150, 350, all=False), timeout=10)
        every = await asyncio.wait_for(c.query("events", 0, 1_000), timeout=10)
        some = await asyncio.wait_for(c.query("events", 150, 350, ids=[uuid_of(3)]), timeout=10)
        await c.close()
        return one, every, some

    one, every, some = asyncio.run(run())
    assert sorted(numbers(one)) == [1, 3, 4]
    assert one["items"][0]["locator"] is not None and one["items"][0]["locator"]["kind"] == "Inline"
    assert numbers(every) == [2, 3, 4, 1, 5], "every item once, by start time then seq"
    assert every["next_after"] == 0 and every["complete"]
    assert numbers(some) == [3]


def test_bad_arguments_are_refused(socket: str) -> None:
    async def run() -> None:
        async with await Client.connect(socket) as c:
            with pytest.raises(ValueError, match="after start_ns"):
                await c.query("events", 10, 5)
            with pytest.raises(ValueError, match="after start_ns"):
                await c.query("events", 5, 5)  # an empty range, which would otherwise read the whole stream
            with pytest.raises(ValueError, match="only with all=True"):
                await c.query("events", 0, 5, all=False, ids=[uuid_of(1)])
            with pytest.raises(ValueError, match="canonical UUID"):
                await c.query("events", 0, 5, subjects=["someone"])

    asyncio.run(run())


def test_truncation_is_separate_from_gaps_in_a_sparse_time_range(socket: str) -> None:
    async def run() -> tuple[QueryResult, QueryResult, QueryResult]:
        async with await events(socket, [100, 300]) as c:
            gap = await c.query("events", 0, 400, all=False)
            capped = await c.query("events", 0, 400, all=False, max_items=1)
            drained = await c.query("events", 0, 400, all=True, max_items=1)
            return gap, capped, drained

    gap, capped, drained = asyncio.run(run())
    assert numbers(gap) == [1, 2]
    assert gap["complete"] is False
    assert gap["missing"] == [(0, 100), (110, 300), (310, 400)]
    assert gap["truncated"] is False
    assert numbers(capped) == [1]
    assert capped["complete"] is False
    assert capped["truncated"] is True
    assert gap["next_after"] == capped["next_after"] == 0, "time queries have no sequence continuation cursor"
    assert numbers(drained) == [1, 2]
    assert drained["complete"] is True
    assert drained["truncated"] is False


def test_a_page_of_large_inline_payloads_stops_at_the_result_byte_cap_and_pages_on(socket: str) -> None:
    # 130 payloads of 64 KiB (8.1 MiB) are more than one result's 8 MiB of inline bytes.
    big = 64 * 1024

    async def run() -> tuple[int, bool, bool, bool, list[int]]:
        async with await Client.connect(socket) as c:
            stream = (
                await c.register_stream(
                    "blobs", kind="derived", format="Text", shape_name="blob", layer="Layer2Decision", slab_class=0
                )
            )["stream_id"]
            live = await c.subscribe(["blobs"])
            for n in range(1, 131):
                header = encode_header(
                    {
                        "id": uuid_of(n),
                        "stream_id": stream,
                        "start_ns": n * 100,
                        "end_ns": n * 100 + 10,
                        "layer": "Layer2Decision",
                        "format": "Text",
                        "shape_name": "blob",
                    }
                )
                c.ingest(header, inline=bytes([n]) * big)
            for _ in range(130):
                assert await asyncio.wait_for(live.next(), timeout=10) is not None
            await live.unsubscribe()
            one = await c.query("blobs", 0, 1_000_000, all=False)
            every = await c.query("blobs", 0, 1_000_000, all=True)
            compact = await c.query_inline_records("blobs", 0, 1_000_000, max_items=1000)
            assert compact == [(uuid_of(n), n * 100, n * 100 + 10, bytes([n]) * big) for n in range(1, 131)]
            return len(one["items"]), one["complete"], one["truncated"], every["truncated"], numbers(every)

    first_page, complete, truncated, drained_truncated, every = asyncio.run(run())
    assert first_page == 128, "8 MiB of 64 KiB payloads"
    assert complete is False
    assert truncated is True
    assert drained_truncated is False
    assert every == list(range(1, 131)), "the sequence query pages on past each full result"


def test_header_heavy_results_page_below_the_wire_limit_without_losing_items(socket: str) -> None:
    # 900 * 20 KiB of header strings exceeds a 16 MiB control frame, while all inline bodies total only 900 bytes.
    count = 900
    shape = "header-heavy-" + "x" * (20 * 1024)

    async def run() -> None:
        async with await Client.connect(socket) as c:
            stream = (
                await c.register_stream(
                    "headers", kind="derived", format="Text", shape_name="event", layer="Layer2Decision", slab_class=0
                )
            )["stream_id"]
            live = await c.subscribe(["headers"])
            expected = []
            for first in range(1, count + 1, 30):
                batch = range(first, min(count + 1, first + 30))
                for n in batch:
                    identity = f"00000000-0000-7000-8000-{n:012x}"
                    expected.append(identity)
                    c.ingest(
                        encode_header(
                            {
                                "id": identity,
                                "stream_id": stream,
                                "start_ns": n,
                                "end_ns": n + 1,
                                "seq": n,
                                "layer": "Layer2Decision",
                                "format": "Text",
                                "shape_name": shape,
                            }
                        ),
                        inline=b"x",
                    )
                for _ in batch:
                    assert await asyncio.wait_for(live.next(), timeout=10) is not None
            await live.unsubscribe()
            page = await asyncio.wait_for(c.query("headers", 0, count + 2, all=False, max_items=2000), timeout=10)
            assert 0 < len(page["items"]) < count
            assert page["truncated"] is True and page["complete"] is False
            every = await asyncio.wait_for(c.query("headers", 0, count + 2, all=True, max_items=2000), timeout=20)
            assert [decode_header(item["header"])["id"] for item in every["items"]] == expected
            assert all(item["locator"] is not None and item["locator"]["bytes"] == b"x" for item in every["items"])
            assert every["complete"] is True and every["truncated"] is False
            compact = await asyncio.wait_for(
                c.query_inline_records("headers", 0, count + 2, max_items=2000), timeout=20
            )
            assert compact == [(identity, n, n + 1, b"x") for n, identity in enumerate(expected, 1)]
            # A large response must not poison the session's frame reader.
            again = await asyncio.wait_for(c.query("headers", 1, 2, all=False, max_items=1), timeout=10)
            assert [decode_header(item["header"])["id"] for item in again["items"]] == expected[:1]

    asyncio.run(run())


def test_compact_inline_query_drains_sparse_pages_and_keeps_order_filters_and_raw_bytes(socket: str) -> None:
    async def run() -> None:
        async with await events(socket, [300, 100, 200, 200, 400]) as c:
            compact = await c.query_inline_records("events", 0, 1000, max_items=1, band="P3Background")
            assert compact == [
                (uuid_of(2), 100, 110, b"\x02"),
                (uuid_of(3), 200, 210, b"\x03"),
                (uuid_of(4), 200, 210, b"\x04"),
                (uuid_of(1), 300, 310, b"\x01"),
                (uuid_of(5), 400, 410, b"\x05"),
            ]
            assert await c.query_inline_records("events", 150, 350, ids=[uuid_of(3)], max_items=1) == [
                (uuid_of(3), 200, 210, b"\x03")
            ]
            assert await c.query_inline_records("events", 0, 1000, subjects=[uuid_of(20)]) == []
            assert await c.query_inline_records("events", 500, 600) == []

    asyncio.run(run())


@pytest.mark.parametrize(
    ("start", "end", "kwargs", "message"),
    [
        (10, 5, {}, "after start_ns"),
        (5, 5, {}, "after start_ns"),
        (0, 5, {"subjects": ["someone"]}, "canonical UUID"),
        (0, 5, {"ids": ["not an id"]}, "canonical UUID"),
    ],
)
def test_compact_inline_query_rejects_invalid_arguments(
    socket: str, start: int, end: int, kwargs: dict[str, list[str]], message: str
) -> None:
    async def run() -> None:
        async with await Client.connect(socket) as c:
            with pytest.raises(ValueError, match=message):
                await c.query_inline_records(
                    "events", start, end, subjects=kwargs.get("subjects", []), ids=kwargs.get("ids", [])
                )

    asyncio.run(run())


def test_compact_query_keeps_ordered_header_only_slot_and_chunk_items_unchanged(socket: str, tmp_path: Path) -> None:

    # Registering a checked synthetic FLAC in place makes a real Chunk without starting a tier service.
    archive = tmp_path / "tone.flac"
    archive.write_bytes((Path(__file__).parent / "fixtures" / "v1_tone.flac").read_bytes())

    async def run() -> None:
        async with await Client.connect(socket) as c:
            stream = (
                await c.register_stream(
                    "mixed", kind="sensor", format="RawBytes", shape_name="PCM_S16LEx1", layer="Layer1Raw", slab_class=0
                )
            )["stream_id"]
            await c.register_worker("compact-query-fixture", [])
            await c.register_v1_flac(stream, archive, 4_000_000_000)
            live = await c.subscribe(["mixed"])
            arena = c.arena(c.arena_names[0])
            slot, generation, offset = arena.write(0, b"slot bytes")
            location: ShmLocation = {
                "arena": arena.name,
                "arena_generation": arena.generation,
                "slab_class": 0,
                "slot": slot,
                "slot_generation": generation,
                "offset": offset,
                "length": 10,
            }
            bodies = {1: b"not JSON\xff", 5: b'{"value":NaN,"large":123456789012345678901234567890}'}
            for n in (1, 2, 3, 5):
                header = encode_header(
                    {
                        "id": uuid_of(n),
                        "stream_id": stream,
                        "start_ns": n * 1_000_000_000,
                        "end_ns": n * 1_000_000_000 + 100,
                        "seq": n,
                        "layer": "Layer1Raw",
                        "format": "RawBytes",
                        "shape_name": "PCM_S16LEx1",
                    }
                )
                c.ingest(header, inline=bodies.get(n), shm=location if n == 2 else None)
                assert await asyncio.wait_for(live.next(), timeout=5) is not None
            await live.unsubscribe()
            original = (await c.query("mixed", 0, 6_000_000_000))["items"]
            compact = await c.query_inline_records("mixed", 0, 6_000_000_000, max_items=1)
            assert len(compact) == 5
            assert compact[0] == (uuid_of(1), 1_000_000_000, 1_000_000_100, bodies[1])
            assert compact[1:4] == original[1:4]
            assert original[1]["locator"] is not None and original[1]["locator"]["kind"] == "Shm"
            assert original[2]["locator"] is None
            assert original[3]["locator"] is not None and original[3]["locator"]["kind"] == "Chunk"
            assert compact[4] == (uuid_of(5), 5_000_000_000, 5_000_000_100, bodies[5])

    asyncio.run(run())
