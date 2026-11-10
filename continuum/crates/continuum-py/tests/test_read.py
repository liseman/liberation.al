"""continuum_py.read.records against a real broker (and, for chunks, the real tier worker): a query item's payloads
from each kind of locator, and what a stale locator (the payload moved or was purged between the query and the read)
reads as.

Run with `mise run continuum-test`, which builds the broker and the tier worker first.
"""

import asyncio
import shutil
import time
from pathlib import Path
from typing import TypedDict
from unittest.mock import create_autospec

import continuum_py.read as reads
import pytest
from continuum_py import Client, Item, Locator, ShmLocation, decode_header, encode_header
from continuum_py.read import Record, records

from conftest import spawn_broker, spawn_tier_worker, stop_broker

TONE = Path(__file__).parent / "fixtures" / "tone.flac"


def uuid_of(n: int) -> str:
    return f"0199a000-0000-7000-8000-{n:012x}"


def header(stream_id: int, n: int) -> bytes:
    return encode_header(
        {
            "id": uuid_of(n),
            "stream_id": stream_id,
            "start_ns": n * 1_000,
            "end_ns": n * 1_000 + 500,
            "layer": "Layer2Decision",
            "format": "Text",
            "shape_name": "note",
        }
    )


async def notes(client: Client) -> int:
    registered = await client.register_stream(
        "notes", kind="derived", format="Text", shape_name="note", slab_class=0, layer="Layer2Decision"
    )
    return registered["stream_id"]


async def commit(client: Client, n: int, **at: object) -> None:
    """Ingests note `n` and waits until it's committed (the broker's live event for it)."""
    stream = await notes(client)
    live = await client.subscribe(["notes"], start="Live")
    client.ingest(header(stream, n), **at)  # ty: ignore[invalid-argument-type]  # inline= or shm=, as the test gives
    assert await asyncio.wait_for(live.next(), timeout=5) is not None
    await live.unsubscribe()


def in_slot(client: Client, data: bytes) -> ShmLocation:
    """`data` written into a Tier 0 slot of the broker's arena, as an ingest's shm location."""
    arena = client.arena(client.arena_names[0])
    slot, generation, offset = arena.write(0, data)
    return {
        "arena": arena.name,
        "arena_generation": arena.generation,
        "slab_class": 0,
        "slot": slot,
        "slot_generation": generation,
        "offset": offset,
        "length": len(data),
    }


async def items(client: Client, stream: str = "notes") -> list[Item]:
    return (await client.query(stream, 0, 2**63 - 1, all=True))["items"]


def kind(item: Item) -> str | None:
    return item["locator"]["kind"] if item["locator"] else None


def test_an_inline_payload_reads_back(socket: str) -> None:
    async def run() -> tuple[str | None, list[Record] | None]:
        async with await Client.connect(socket) as c:
            await commit(c, 1, inline=b'{"text":"hello"}')
            (item,) = await items(c)
            return kind(item), await records(c, "notes", item)

    assert asyncio.run(run()) == ("Inline", [Record(uuid_of(1), 1_000, 1_500, b'{"text":"hello"}')])


def test_a_slot_payload_reads_back_whole(socket: str) -> None:
    body = bytes(range(256)) * 40  # 10 KiB, in a 16 KiB slot

    async def run() -> tuple[str | None, list[Record] | None]:
        async with await Client.connect(socket) as c:
            await commit(c, 1, shm=in_slot(c, body))
            (item,) = await items(c)
            return kind(item), await records(c, "notes", item)

    assert asyncio.run(run()) == ("Shm", [Record(uuid_of(1), 1_000, 1_500, body)])


def test_flushed_payloads_read_back_from_their_chunk_even_through_a_stale_slot_locator(
    socket: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bodies = [bytes([n]) * 5_000 for n in (1, 2, 3)]

    async def run() -> tuple[list[Record] | None, list[Record] | None, list[Record] | None, list[Record] | None]:
        async with await Client.connect(socket) as c:
            registered = await c.register_stream("notes", kind="sensor", format="Text", shape_name="note", slab_class=0)
            stream = registered["stream_id"]
            worker = spawn_tier_worker(Path(socket), tmp_path / "chunks", "notes")
            try:
                for n, body in enumerate(bodies, 1):
                    raw = encode_header(
                        {
                            "id": uuid_of(n),
                            "stream_id": stream,
                            "start_ns": n * 1_000,
                            "end_ns": n * 1_000 + 500,
                            "layer": "Layer1Raw",
                            "format": "Text",
                            "shape_name": "note",
                        }
                    )
                    live = await c.subscribe(["notes"], start="Live")
                    c.ingest(raw, shm=in_slot(c, body))
                    assert await asyncio.wait_for(live.next(), timeout=5) is not None
                    await live.unsubscribe()
                (slot_item, *_) = await items(c)
                for _ in range(500):  # the worker closes a chunk after 2 s: wait (bounded) for its commit
                    if kind(chunk := (await items(c))[0]) == "Chunk":
                        break
                    await asyncio.sleep(0.02)
            finally:
                worker.kill()
                worker.wait()
            # A query made before the flush handed out note 1's slot; say the slot has since been reused.
            assert slot_item["locator"] is not None
            stale: Locator = {**slot_item["locator"], "slot_generation": slot_item["locator"]["slot_generation"] + 1}
            # A chunk recorded at a relative path (a tier worker given a relative directory) is read from the
            # working directory.
            assert chunk["locator"] is not None
            monkeypatch.chdir(tmp_path)
            relative: Locator = {**chunk["locator"], "uri": str(Path(chunk["locator"]["uri"]).relative_to(tmp_path))}
            return (
                await records(c, "notes", chunk),
                await records(c, "notes", chunk, start_ns=2_000, end_ns=2_600),
                await records(c, "notes", {**slot_item, "locator": stale}),
                await records(c, "notes", {**chunk, "locator": relative}),
            )

    every, one, from_stale, from_relative = asyncio.run(run())
    assert every == [Record(uuid_of(n), n * 1_000, n * 1_000 + 500, bodies[n - 1]) for n in (1, 2, 3)]
    assert one == [Record(uuid_of(2), 2_000, 2_500, bodies[1])], "only the records in the range"
    assert from_stale == [Record(uuid_of(1), 1_000, 1_500, bodies[0])], "found in the chunk that absorbed it"
    assert from_relative == every, "a relative chunk path, read from the working directory"


def test_a_flac_registered_in_place_reads_as_one_payload(socket: str, tmp_path: Path) -> None:
    device = tmp_path / "wave"
    device.mkdir()
    shutil.copy(TONE, device / "a.flac")
    (device / "index.jsonl").write_text('{"file": "a.flac", "start": 1793300000.25, "sr": 48000}\n')

    # The query's item is the file's one-frame chunk, and a FLAC file holds no record ids: the record is the chunk's.
    async def run() -> tuple[str, str | None, list[Record] | None]:
        async with await Client.connect(socket) as c:
            await c.register_worker("audio-import", [])
            (registered,) = await c.register_audio_index("wave.audio_v1", str(device))
            (item,) = await items(c, "wave.audio_v1")
            return registered["chunk_id"], kind(item), await records(c, "wave.audio_v1", item)

    chunk_id, got_kind, got = asyncio.run(run())
    assert got_kind == "Chunk"
    assert got is not None
    assert [(r.payload_id, r.data) for r in got] == [(chunk_id, TONE.read_bytes())]


def gone(item: Item, tmp_path: Path) -> Item:
    """`item` as a query before a purge took its chunk would have given it."""
    locator: Locator = {"kind": "Chunk", "tier": "Tier1Warm", "uri": str(tmp_path / "gone.cnk"), "offset": 0}
    return {**item, "locator": locator}


def test_a_stale_locator_is_looked_up_again_and_read_from_where_the_payload_is_now(socket: str, tmp_path: Path) -> None:
    async def run() -> list[Record] | None:
        async with await Client.connect(socket) as c:
            await commit(c, 1, inline=b"current")
            (item,) = await items(c)
            return await records(c, "notes", gone(item, tmp_path))

    assert asyncio.run(run()) == [Record(uuid_of(1), 1_000, 1_500, b"current")]


def test_a_payload_purged_since_the_query_reads_as_none(socket: str, tmp_path: Path) -> None:
    async def run() -> list[Record] | None:
        async with await Client.connect(socket) as c:
            await commit(c, 1, inline=b"soon gone")
            (item,) = await items(c)
            await c.purge_range(0, 2**63 - 1, streams=["notes"], reason="test")
            return await records(c, "notes", gone(item, tmp_path))

    assert asyncio.run(run()) is None


def test_a_header_only_payload_has_no_records(socket: str) -> None:
    async def run() -> list[Record] | None:
        async with await Client.connect(socket) as c:
            await commit(c, 1)
            (item,) = await items(c)
            return await records(c, "notes", item)

    assert asyncio.run(run()) == []


@pytest.mark.parametrize(
    "locator",
    [
        pytest.param({"kind": "Chunk", "uri": "s3://bucket/c.cnk", "offset": 0}, id="off-box-chunk"),
        pytest.param({"kind": "Ring", "arena": "a", "ring": 0, "first_seq": 1, "count": 4}, id="ring-batch"),
    ],
)
def test_a_locator_that_isnt_readable_here_is_refused_at_once(socket: str, locator: Locator) -> None:
    async def run() -> None:
        async with await Client.connect(socket) as c:
            await records(c, "notes", {"header": header(1, 1), "locator": locator})

    with pytest.raises(ValueError, match="not a local file|isn't a payload"):
        asyncio.run(run())


def test_chunk_reads_keep_explicit_audio_format_while_legacy_reads_stay_unknown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:

    client = create_autospec(Client, instance=True)
    item = Item(
        header=header(1, 1),
        locator=Locator(kind="Chunk", tier="Tier1Warm", uri=str(tmp_path / "audio.cnk"), offset=0, length=1),
    )

    def chunk_records(uri: str, start: int, end: int) -> list[tuple[str, int, int, bytes, tuple[int, int, int] | None]]:
        return [(uuid_of(1), 1_000, 1_500, b"audio", (12_000, 1, 496)), (uuid_of(2), 2_000, 2_500, b"legacy", None)]

    monkeypatch.setattr(reads, "read_chunk_records", chunk_records)
    found = asyncio.run(records(client, "mic.audio", item))
    assert found == [
        Record(uuid_of(1), 1_000, 1_500, b"audio", reads.PcmFormat(12_000, 1, 496)),
        Record(uuid_of(2), 2_000, 2_500, b"legacy"),
    ]
    client.query.assert_not_called()


def copy_files(chunks: Path) -> list[Path]:
    """The derived copies under the chunk directory. The tier worker files each one in its stream's hour directory
    (`streams/<stream>/<hour>/`), so the search is recursive."""
    return list(chunks.rglob("*-copy-*.cnk"))


class PurgeArgs(TypedDict, total=False):
    ids: list[str]
    entity: str
    reason: str


@pytest.mark.parametrize(
    "purge_args",
    [
        pytest.param({"ids": [uuid_of(700)], "reason": "mistake"}, id="parent"),
        pytest.param({"entity": uuid_of(600), "reason": "PersonForgotten"}, id="entity"),
    ],
)
def test_derived_copies_survive_restart_and_purge(tmp_path: Path, purge_args: PurgeArgs) -> None:
    socket = tmp_path / "b.sock"
    chunks = tmp_path / "chunks"
    broker = spawn_broker(socket, tier0_classes="131072x16")
    worker = None
    bodies = [bytes([value]) * 70_000 for value in (1, 2, 3)]
    person = uuid_of(600)
    parent = uuid_of(700)

    async def archive() -> tuple[list[bytes], list[int]]:
        nonlocal worker
        async with await Client.connect(str(socket)) as client:
            source = await client.register_stream(
                "source", kind="sensor", format="RawBytes", shape_name="source", slab_class=0
            )
            await client.ingest_acked(
                encode_header(
                    {
                        "id": parent,
                        "stream_id": source["stream_id"],
                        "start_ns": 1,
                        "end_ns": 2,
                        "layer": "Layer1Raw",
                        "format": "RawBytes",
                        "shape_name": "source",
                    }
                ),
                inline=b"source",
            )
            stream = await notes(client)
            # The explicit list is a sensor only: discovery must find derived streams independently.
            worker = spawn_tier_worker(socket, chunks, "source")
            headers = []
            live = await client.subscribe(["notes"], start="Live")
            for number, body in enumerate(bodies, 1):
                encoded = encode_header(
                    {
                        "id": uuid_of(number),
                        "stream_id": stream,
                        "start_ns": number * 1_000,
                        "end_ns": number * 1_000 + 500,
                        "layer": "Layer2Decision",
                        "format": "Text",
                        "shape_name": "note",
                        "parents": [parent],
                        "subjects": [person],
                    }
                )
                headers.append(encoded)
                client.ingest(encoded, shm=in_slot(client, body))
                assert await asyncio.wait_for(live.next(), timeout=5) is not None
            sibling = encode_header(
                {
                    "id": uuid_of(4),
                    "stream_id": stream,
                    "start_ns": 1_000,
                    "end_ns": 1_500,
                    "seq": 1,
                    "layer": "Layer2Decision",
                    "format": "Text",
                    "shape_name": "note",
                    "parents": [],
                    "subjects": [uuid_of(601)],
                }
            )
            client.ingest(sibling, shm=in_slot(client, b"s" * 70_000))
            assert await asyncio.wait_for(live.next(), timeout=5) is not None
            await live.unsubscribe()
            deadline = time.monotonic() + 8
            while True:
                page = await client.query("notes", 0, 10_000, all=True, max_items=1)
                # Queries prefer still-readable hot slots. File publication plus orderly Tier shutdown
                # lets the in-flight attachment finish; warm query/read assertions follow broker restart.
                copies = copy_files(chunks)
                if len(page["items"]) == 4 and len(copies) == 4:
                    break
                assert time.monotonic() < deadline, (
                    f"derived slot payloads never acquired durable copies: {len(page['items'])} of 4 items, "
                    f"copies {sorted(path.name for path in copies)}"
                )
                assert worker.poll() is None, "Tier exited while archiving derived payloads"
                await asyncio.sleep(0.02)
            positions = await client.query_by_seq("notes", max_items=1)
            return headers, [item["commit_seq"] for item in positions["items"]]

    async def read_and_forget(headers: list[bytes], positions: list[int]) -> None:
        nonlocal worker
        async with await Client.connect(str(socket)) as client:
            worker = spawn_tier_worker(socket, chunks, "source")
            page = await client.query("notes", 0, 10_000, all=True, max_items=1, subjects=[person])
            assert not page["truncated"] and len(page["items"]) == 3
            assert [item["header"] for item in page["items"]] == headers
            after = await client.query_by_seq("notes", max_items=1)
            assert [item["commit_seq"] for item in after["items"]] == positions
            for number, item in enumerate(page["items"], 1):
                assert kind(item) == "Chunk", "Tier 0's recreated arena must not supply the bytes"
                assert decode_header(item["header"])["parents"] == [parent]
                assert await records(client, "notes", item) == [
                    Record(uuid_of(number), number * 1_000, number * 1_000 + 500, bodies[number - 1])
                ]
            await client.purge(**purge_args)
            deadline = time.monotonic() + 8
            while True:
                left = await client.query("notes", 0, 10_000, all=True)
                files = copy_files(chunks)
                ids = [decode_header(item["header"])["id"] for item in left["items"]]
                if ids == [uuid_of(4)] and len(files) == 1:
                    break
                assert time.monotonic() < deadline, (
                    "purge must remove doomed sidecars and preserve the unrelated same-time sibling: "
                    f"items {ids}, copies {sorted(path.name for path in files)}"
                )
                assert worker.poll() is None, "Tier exited before removing its purged copies"
                await asyncio.sleep(0.02)
            assert await records(client, "notes", left["items"][0]) == [Record(uuid_of(4), 1_000, 1_500, b"s" * 70_000)]

    try:
        headers, positions = asyncio.run(archive())
        assert worker is not None
        worker.terminate()
        assert worker.wait(timeout=10) == 0, "Tier must finish its in-flight durable-copy commit before restart"
        worker = None
        stop_broker(broker, socket)
        broker = spawn_broker(socket, tier0_classes="131072x16")
        asyncio.run(read_and_forget(headers, positions))
    finally:
        if worker is not None:
            worker.terminate()
            worker.wait(timeout=10)
        stop_broker(broker, socket)
