"""Acknowledged ingest, the sequence query, the stream list, and purges with their records, from Python against a real
broker (what the event log adapter needs). These go through continuum-client, so they test its calls too.

Run with `mise run continuum-test`, which builds the broker first.
"""

import asyncio
import json

import pytest
from continuum_py import BrokerError, Client, decode_header, encode_header

PERSON = "0199a000-0000-7000-8000-00000000a11c"


def uuid_of(n: int) -> str:
    raw = bytearray(16)
    raw[6], raw[8], raw[15] = 0x70, 0x80, n
    h = raw.hex()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def header(stream: int, n: int, *, subjects: tuple[str, ...] = (), text: str = "") -> bytes:
    return encode_header(
        {
            "id": uuid_of(n),
            "stream_id": stream,
            "start_ns": n * 100,
            "end_ns": n * 100,
            "layer": "Layer2Decision",
            "format": "RawBytes",
            "shape_name": text or "event",
            "subjects": list(subjects),
        }
    )


async def events(socket: str) -> tuple[Client, int]:
    c = await Client.connect(socket, client_name="events")
    registered = await c.register_stream(
        "events:test", kind="derived", format="RawBytes", shape_name="event", slab_class=0, layer="Layer2Decision"
    )
    return c, registered["stream_id"]


def test_ingest_acked_gives_rising_seqs_and_the_same_seq_for_a_repeat(socket: str) -> None:
    async def run() -> tuple[list[int], int, type[BaseException] | None]:
        c, stream = await events(socket)
        seqs = [await c.ingest_acked(header(stream, n), inline=b"x") for n in (1, 2, 3)]
        again = await c.ingest_acked(header(stream, 2), inline=b"x")
        conflict: type[BaseException] | None = None
        try:  # the same id under a different header
            await c.ingest_acked(header(stream, 2, text="other"), inline=b"x")
        except BrokerError as e:
            conflict = type(e)
        await c.close()
        return seqs, again, conflict

    seqs, again, conflict = asyncio.run(run())
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == 3
    assert again == seqs[1]
    assert conflict is BrokerError


def test_the_sequence_query_filters_and_pages_in_commit_order(socket: str) -> None:
    async def run() -> tuple[list[tuple[int, int]], list[int], list[int], list[int], list[int]]:
        c, stream = await events(socket)
        # Committed out of time order: the query follows commit order, not time.
        for n, subjects in ((3, ()), (1, (PERSON,)), (2, ()), (4, (PERSON,))):
            await c.ingest_acked(header(stream, n, subjects=subjects), inline=b"x")
        number = lambda item: int(decode_header(item["header"])["id"][-2:], 16)
        everything = await c.query_by_seq("events:test")
        pairs = [(number(i), i["commit_seq"]) for i in everything["items"]]
        about = await c.query_by_seq("events:test", subjects=[PERSON])
        early = await c.query_by_seq("events:test", start_before=250)
        picked = await c.query_by_seq("events:test", ids=[uuid_of(2), uuid_of(4)])
        # Paged two at a time by next_after.
        paged: list[int] = []
        after = 0
        while True:
            page = await c.query_by_seq("events:test", after_seq=after, max_items=2)
            paged += [number(i) for i in page["items"]]
            if page["next_after"] == 0:
                break
            after = page["next_after"]
        await c.close()
        return (
            pairs,
            [number(i) for i in about["items"]],
            [number(i) for i in early["items"]],
            [number(i) for i in picked["items"]],
            paged,
        )

    pairs, about, early, picked, paged = asyncio.run(run())
    assert [n for n, _ in pairs] == [3, 1, 2, 4]
    assert [s for _, s in pairs] == sorted(s for _, s in pairs)
    assert about == [1, 4]
    assert early == [1, 2]
    assert picked == [2, 4]
    assert paged == [3, 1, 2, 4]


def test_list_streams_names_what_is_registered(socket: str) -> None:
    async def run() -> list[dict]:
        c, _ = await events(socket)
        streams = await c.list_streams()
        await c.close()
        return streams

    (stream,) = [s for s in asyncio.run(run()) if s["name"] == "events:test"]
    assert (stream["layer"], stream["format"], stream["shape_name"]) == ("Layer2Decision", "RawBytes", "event")


def test_purges_by_ids_and_entity_are_recorded_without_naming_anyone(socket: str) -> None:
    async def run() -> tuple[dict, dict, list[dict], dict, list[int]]:
        c, stream = await events(socket)
        for n, subjects in ((1, (PERSON,)), (2, ()), (3, (PERSON,)), (4, ())):
            await c.ingest_acked(header(stream, n, subjects=subjects), inline=b"x")
        detail = json.dumps({"stream": "events:test", "count": 1})
        by_ids = await c.purge(ids=[uuid_of(2)], reason="mistake", detail=detail)
        by_entity = await c.purge(entity=PERSON, reason="PersonForgotten")
        purges = await c.list_purges()
        record = await c.get_purge(by_ids["purge_id"])
        left = await c.query_by_seq("events:test")
        await c.close()
        return (
            by_ids,
            by_entity,
            purges,
            record,
            [int(decode_header(i["header"])["id"][-2:], 16) for i in left["items"]],
        )

    by_ids, by_entity, purges, record, left = asyncio.run(run())
    assert left == [4]
    assert [p["purge_id"] for p in purges] == [by_ids["purge_id"], by_entity["purge_id"]]
    ids, entity = purges
    assert (ids["kind"], ids["id_count"], ids["reason"], ids["removed"]) == ("Ids", 1, "mistake", 1)
    assert json.loads(ids["detail"]) == {"stream": "events:test", "count": 1}
    assert (entity["kind"], entity["reason"], entity["removed"]) == ("Entity", "PersonForgotten", 2)
    # The record kept for good never names the person.
    assert PERSON not in json.dumps(purges)
    assert record["purge_id"] == by_ids["purge_id"]
    assert sum(n for _, _, n in record["removed_by"]) == 1


@pytest.mark.parametrize(
    "target",
    [
        pytest.param({}, id="none"),
        pytest.param({"entity": PERSON, "ids": [uuid_of(1)]}, id="entity-and-ids"),
        pytest.param({"start_ns": 0, "end_ns": 5, "entity": PERSON}, id="range-and-entity"),
        pytest.param({"entity": PERSON, "streams": ["events:test"]}, id="entity-with-streams"),
        # One bound beside an entity or ids must be refused, not dropped (which would purge all of them).
        pytest.param({"entity": PERSON, "end_ns": 5}, id="entity-with-one-bound"),
        pytest.param({"ids": [uuid_of(1)], "start_ns": 5}, id="ids-with-one-bound"),
        pytest.param({"start_ns": 5}, id="one-bound-alone"),
    ],
)
def test_a_purge_takes_exactly_one_target(socket: str, target: dict[str, object]) -> None:
    async def run() -> None:
        c, _ = await events(socket)
        try:
            await c.purge(**target)  # ty: ignore[invalid-argument-type]  # the bad combinations under test
        finally:
            await c.close()

    with pytest.raises(ValueError, match="exactly one"):
        asyncio.run(run())


def test_stream_revision_tracks_deletion_but_not_appends(socket: str) -> None:
    async def run() -> tuple[int, int, int]:
        c, stream_id = await events(socket)

        async def revision() -> int:
            stream = next(s for s in await c.list_streams() if s["name"] == "events:test")
            value = stream["invalidation_revision"]
            assert isinstance(value, int) and not isinstance(value, bool)
            return value

        try:
            before = await revision()
            await c.ingest_acked(header(stream_id, 1), inline=b"x")
            appended = await revision()
            await c.purge(ids=[uuid_of(1)], reason="test deletion")
            removed = await revision()
            return before, appended, removed
        finally:
            await c.close()

    before, appended, removed = asyncio.run(run())
    assert appended == before
    assert removed > appended


def test_list_forgotten_pages_opaque_hashes_without_repeats(socket: str) -> None:
    async def run() -> None:
        async with await Client.connect(socket, client_name="forgotten-export") as client:
            assert await client.list_forgotten() == {"hashes": [], "next": None}
            for byte in (255, 0, 2, 1):
                await client.remember_forgotten(bytes([byte]) * 32)
            first = await client.list_forgotten(limit=2)
            assert first == {"hashes": [bytes(32), bytes([1]) * 32], "next": bytes([1]) * 32}
            assert await client.list_forgotten(first["next"], 2) == {
                "hashes": [bytes([2]) * 32, bytes([255]) * 32],
                "next": None,
            }
            assert await client.list_forgotten(bytes([255]) * 32) == {"hashes": [], "next": None}
            assert await client.is_forgotten(bytes(32))

    asyncio.run(run())


@pytest.mark.parametrize("cursor", [b"", b"x" * 31, b"x" * 33])
def test_list_forgotten_rejects_non_hash_cursor(socket: str, cursor: bytes) -> None:
    async def run() -> None:
        async with await Client.connect(socket) as client:
            with pytest.raises(ValueError, match="32"):
                await client.list_forgotten(cursor)

    asyncio.run(run())


@pytest.mark.parametrize("limit", [0, 1025])
def test_list_forgotten_rejects_unbounded_pages(socket: str, limit: int) -> None:
    async def run() -> None:
        async with await Client.connect(socket) as client:
            with pytest.raises(ValueError, match="limit"):
                await client.list_forgotten(limit=limit)

    asyncio.run(run())
