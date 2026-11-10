import asyncio
import collections
import json
from pathlib import Path
from typing import Any, Literal, cast

import continuum_py
import pytest
from continuum_py import read

from liberational import continuum_link, segment_log, vision_log

CAPTION = {
    "ts_start": 1_800_000_000.5,
    "cam": "cam0",
    "yaw_deg": 0.0,
    "caption": "Sam at the desk, typing",
    "faces": [{"name": "Sam", "sim": 0.71, "bearing_deg": -4.0, "bbox": [120, 80, 180, 150]}],
    "objects": [{"label": "laptop", "bbox_norm": [0.4, 0.5, 0.7, 0.8]}],
    "source": "mlx-server",
    "observation": {"version": 1, "status": "fresh", "camera": "cam0"},
    "brightness": {"mean": 0.42, "dark_fraction": 0.1},
}
PID = "0190c8e0-0000-7000-8000-000000000001"


def item(header: bytes, body: bytes | None) -> continuum_py.Item:
    locator: continuum_py.Locator | None = None if body is None else {"kind": "Inline", "bytes": body}
    return {"header": header, "locator": locator}


def test_a_caption_reads_back_as_the_record_it_was() -> None:
    header, body = vision_log.encode(vision_log.CAPTIONS, 4, CAPTION, payload_id=PID)
    h = continuum_py.decode_header(header)
    assert (h["start_ns"], h["end_ns"]) == (1_800_000_000_500_000_000, 1_800_000_000_500_000_000)
    assert (h["layer"], h["shape_name"], h["stream_id"]) == ("Layer1Raw", "camera.caption", 4)
    assert vision_log.decode(item(header, body)) == (
        vision_log.CAPTIONS,
        {**CAPTION, "ts_end": 1_800_000_000.5, "id": PID},
    )


def test_vision_reads_register_streams_after_reconnecting_to_a_fresh_catalog(broker, tmp_path):
    from continuum_broker import start_broker

    replacement_state = tmp_path / "replacement"
    replacement_state.mkdir()
    log = vision_log.VisionLog(broker)
    try:

        def read_window():
            return log.range(vision_log.OBJECTS, 0, 1)

        assert read_window() == []
        log.link.call(lambda client: client.close())
        replacement = start_broker(replacement_state)
        log.link.socket = next(replacement)
        try:
            assert read_window() == [], "a stream without its producer must still be registered on the new session"
        finally:
            next(replacement, None)
    finally:
        log.close()


def test_a_pet_sighting_is_a_layer_2_span_and_none_values_are_left_out() -> None:
    sighting = {"ts_start": 10.0, "ts_end": 70.0, "label": "Rex", "cam": "cam0", "species": "dog", "source": None}
    header, body = vision_log.encode(vision_log.PET_SIGHTINGS, 9, sighting, payload_id=PID)
    h = continuum_py.decode_header(header)
    assert (h["layer"], h["start_ns"], h["end_ns"]) == ("Layer2Decision", 10_000_000_000, 70_000_000_000)
    assert json.loads(body) == {"cam": "cam0", "species": "dog", "label": "Rex"}


def test_keys_a_stream_doesnt_carry_are_refused_not_dropped() -> None:
    with pytest.raises(ValueError, match=r"\['embedding'\]"):
        vision_log.encode(vision_log.OBJECTS, 3, {"ts_start": 1.0, "cam": "cam0", "embedding": [0.1]})


@pytest.mark.parametrize(
    ("shape", "body"),
    [
        pytest.param("camera.caption", None, id="no-inline-body"),
        pytest.param("speech.utterance", b"{}", id="not-a-vision-stream"),
    ],
)
def test_payloads_it_cant_read_are_skipped(shape: str, body: bytes | None) -> None:
    header = continuum_py.encode_header(
        {
            "id": PID,
            "stream_id": 1,
            "start_ns": 1,
            "end_ns": 1,
            "layer": "Layer1Raw",
            "format": "Text",
            "shape_name": shape,
            "parents": [],
            "subjects": [],
        }
    )
    assert vision_log.decode(item(header, body)) is None


def caption(ts: float, cam: str, text: str, pid: int) -> dict:
    return {"ts_start": ts, "ts_end": ts, "cam": cam, "caption": text, "id": f"0190c8e0-0000-7000-8000-{pid:012d}"}


def test_scene_entries_are_the_newest_oldest_first_with_v1s_ts() -> None:
    records = [caption(30.0, "cam1", "c", 3), caption(10.0, "cam0", "a", 1), caption(20.0, "cam0", "b", 2)]
    scene = vision_log.scene_entries(records, limit=2)
    assert [(e["caption"], e["ts"]) for e in scene] == [("b", 20.0), ("c", 30.0)]
    assert vision_log.scene_entries(records, limit=0) == []


def test_the_latest_caption_of_each_camera() -> None:
    records = [caption(30.0, "cam1", "c", 3), caption(10.0, "cam0", "a", 1), caption(20.0, "cam0", "b", 2)]
    latest = vision_log.latest_by_camera(records)
    assert {cam: r["caption"] for cam, r in latest.items()} == {"cam0": "b", "cam1": "c"}


# ---- Against a real broker (the `broker` fixture is in conftest.py)


async def _commit(sock: Path, records: list[tuple[str, dict]]) -> None:
    async with await continuum_py.Client.connect(str(sock), client_name="writer") as c:
        ids = {s: await vision_log.register(c, s) for s in {s for s, _ in records}}
        live = await c.subscribe(list(ids))
        for stream, record in records:
            header, body = vision_log.encode(stream, ids[stream], record)
            c.ingest(header, inline=body)
        for _ in records:
            assert await live.next() is not None


def test_the_log_reads_a_streams_records_in_time_order_and_the_scene(
    broker: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(segment_log, "at", lambda state=None: segment_log.Log(segment_log.root(state)))

    asyncio.run(
        _commit(
            broker,
            [(vision_log.OBJECTS, {"ts_start": 100.5, "cam": "cam0", "objects": [], "source": "fake"})],
        )
    )
    # Captions are the caption worker's segment rows: the scene reads them from the log's state.
    writers = segment_log.Writers(segment_log.root(tmp_path))
    for ts, cam, text in [(100.0, "cam0", "empty room"), (101.0, "cam1", "a cat"), (99.0, "cam0", "Sam leaves")]:
        writers.append(segment_log.CAPTIONS, cam, [{"id": segment_log.new_id(), "ts_start": ts, "caption": text}])
    writers.append(segment_log.CAPTIONS, "cam0", [{"id": segment_log.new_id(), "ts_start": 900.0, "caption": "later"}])
    writers.close()
    log = vision_log.VisionLog(broker, state=tmp_path)
    try:
        captions = log.captions(98.0, 150.0)
        grounded = log.range(vision_log.OBJECTS, 98.0, 150.0)
        scene = log.scene(now=150.0, limit=2)
        latest = log.latest(now=150.0)
    finally:
        log.close()
    assert [r["caption"] for r in captions] == ["Sam leaves", "empty room", "a cat"]
    assert [(r["cam"], r["source"]) for r in grounded] == [("cam0", "fake")]
    assert [e["caption"] for e in scene] == ["empty room", "a cat"]
    assert {cam: r["caption"] for cam, r in latest.items()} == {"cam0": "empty room", "cam1": "a cat"}


def test_following_from_a_time_catches_up_then_yields_each_record_with_its_stream(
    broker: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Each item is read from `since` on, so a flushed chunk straddling it gives only its later records.
    bounds = []
    original = read.records

    async def spy(client, stream, item, *, start_ns=0, end_ns=2**63 - 1):
        bounds.append(start_ns)
        return await original(client, stream, item, start_ns=start_ns, end_ns=end_ns)

    monkeypatch.setattr(read, "records", spy)
    asyncio.run(
        _commit(
            broker,
            [
                (vision_log.CAPTIONS, {"ts_start": 5.0, "cam": "cam0", "caption": "early"}),
                (
                    vision_log.OBJECTS,
                    {"ts_start": 50.0, "cam": "cam0", "frame_seq": 7, "objects": {"status": "measured"}},
                ),
            ],
        )
    )

    async def first() -> tuple[str, dict]:
        async with await continuum_py.Client.connect(str(broker), client_name="reader") as c:
            records = vision_log.follow(c, [vision_log.CAPTIONS, vision_log.OBJECTS], start="Time", since=40.0)
            got = await anext(records)
            await records.aclose()
            return got

    stream, record = asyncio.run(first())
    assert (stream, record["frame_seq"], record["objects"]) == (vision_log.OBJECTS, 7, {"status": "measured"})
    assert bounds == [40_000_000_000]
    assert record["position"] > 0


def test_a_durable_follower_resumes_after_the_last_record_it_moved_past(broker: Path) -> None:
    asyncio.run(
        _commit(
            broker,
            [(vision_log.CAPTIONS, {"ts_start": float(i), "cam": "cam0", "caption": f"c{i}"}) for i in (1, 2, 3)],
        )
    )

    async def take(n: int, start: Literal["Time", "Saved"]) -> list[str]:
        async with await continuum_py.Client.connect(str(broker), client_name="reader") as c:
            records = vision_log.follow(c, [vision_log.CAPTIONS], start=start, since=0.5, durable_name="importer")
            got = [(await anext(records))[1]["caption"] for _ in range(n)]
            await records.aclose()
            return got

    # The first run moves past c1 and is handed c2, but stops before finishing it.
    assert asyncio.run(take(2, "Time")) == ["c1", "c2"]
    asyncio.run(_saved_cursor(broker, "importer"))
    # So the next run picks up at c2, not c1 and not c3.
    assert asyncio.run(take(2, "Saved")) == ["c2", "c3"]


async def _saved_cursor(broker: Path, durable_name: str) -> None:
    """Waits until the broker shows a saved position for the name: acks are saved through its writer, so a new
    subscription may need a moment to see them."""
    async with await continuum_py.Client.connect(str(broker), client_name="cursor") as c:
        for _ in range(250):
            sub = await c.subscribe([vision_log.CAPTIONS], start="Saved", durable_name=durable_name)
            await sub.unsubscribe()
            if sub.start_position > 0:
                return
            await asyncio.sleep(0.02)
    raise AssertionError("the acknowledged position wasn't saved")


def test_a_given_span_is_kept_to_the_nanosecond() -> None:
    frame_ns = 1_793_345_180_123_456_789  # float seconds would round this to ...456_717
    header, _ = vision_log.encode(
        vision_log.OBJECTS, 4, {"ts_start": frame_ns / 1e9, "cam": "cam0"}, span_ns=(frame_ns, frame_ns)
    )
    h = continuum_py.decode_header(header)
    assert (h["start_ns"], h["end_ns"]) == (frame_ns, frame_ns)


def test_a_flushed_chunks_records_each_read_back_with_their_own_id_and_time(monkeypatch: pytest.MonkeyPatch) -> None:
    # What continuum_py.read gives for a chunk the tier worker flushed: the payloads it absorbed, each its own.
    chunk = [
        read.Record(PID, 5_000_000_000, 5_000_000_000, b'{"cam":"cam0","objects":{"status":"reset"}}'),
        read.Record(PID[:-1] + "2", 6_000_000_000, 6_000_000_000, b"not a record"),
    ]
    asked = []

    async def records(client, stream, item, *, start_ns=0, end_ns=2**63 - 1):
        asked.append((stream, start_ns, end_ns))
        return chunk

    monkeypatch.setattr(read, "records", records)
    got = asyncio.run(vision_log.records_of(None, vision_log.OBJECTS, {"header": b"", "locator": None}, 1, 9))  # ty: ignore[invalid-argument-type] - read.records is stubbed, so no client is used
    assert got == [{"cam": "cam0", "objects": {"status": "reset"}, "ts_start": 5.0, "ts_end": 5.0, "id": PID}]
    assert asked == [(vision_log.OBJECTS, 1, 9)]
    assert asyncio.run(vision_log.records_of(None, "speech.utterance", {"header": b"", "locator": None})) == []  # ty: ignore[invalid-argument-type] - not a vision stream: nothing is read


def test_a_chunk_at_a_relative_path_is_read_from_the_working_directory(monkeypatch, tmp_path) -> None:
    # A tier worker given a relative directory wrote relative chunk paths; the reader (Rust) only opens absolute ones.
    record = {"cam": "cam0", "objects": {}, "ts_start": 5e-9}
    header, _ = vision_log.encode(vision_log.OBJECTS, 1, record, payload_id=PID)
    opened = []

    def read_chunk_records(uri, start_ns, end_ns):
        opened.append(uri)
        return [(PID, 5, 5, b"{}")]

    monkeypatch.setattr(read, "read_chunk_records", read_chunk_records)
    monkeypatch.chdir(tmp_path)
    item = {"header": header, "locator": {"kind": "Chunk", "uri": "data/state/continuum/tier/7-1-1.cnk"}}
    assert asyncio.run(read.records(None, vision_log.OBJECTS, item)) == [read.Record(PID, 5, 5, b"{}")]  # ty: ignore[invalid-argument-type] - a chunk read needs no client
    absolute = {"header": header, "locator": {"kind": "Chunk", "uri": "/elsewhere/7-1-1.cnk"}}
    asyncio.run(read.records(None, vision_log.OBJECTS, absolute))  # ty: ignore[invalid-argument-type] - a chunk read needs no client
    assert opened == [str(tmp_path / "data/state/continuum/tier/7-1-1.cnk"), "/elsewhere/7-1-1.cnk"]


def test_video_days_asks_the_broker_for_one_frame_at_a_time() -> None:
    import datetime as dt

    today = dt.date.today()
    noon = [dt.datetime.combine(today - dt.timedelta(days=d), dt.time(12)).timestamp() for d in (2, 0)]
    frames = [vision_log.to_ns(noon[0] + k) for k in range(500)] + [vision_log.to_ns(noon[1])]
    handed: list[int] = []

    class Broker:
        """The query semantics of continuum_py: `all` (the default) pages through every item, ignoring max_items."""

        async def list_streams(self) -> list[dict]:
            return [{"name": "cam0.video"}, {"name": "cam0.caption"}]

        async def query(self, stream: str, lo: int, hi: int, *, all: bool = True, max_items: int = 0) -> dict:
            found = [f for f in frames if lo <= f < hi]
            found = found if all else found[: max_items or 1024]
            handed.append(len(found))
            header = lambda ns: continuum_py.encode_header(
                {"id": PID, "stream_id": 1, "start_ns": ns, "end_ns": ns, "layer": "Layer1Raw", "format": "EncodedVideo",
                 "shape_name": "camera.frame", "parents": [], "subjects": []}
            )  # fmt: skip
            return {"items": [{"header": header(f)} for f in found]}

    days = asyncio.run(vision_log.video_days(Broker()))  # ty: ignore[invalid-argument-type] - a stand-in for the client
    assert days == [(today - dt.timedelta(days=2)).isoformat(), today.isoformat()]
    assert max(handed) == 1  # never the whole day's frames for a yes or no


def test_startup_wait_does_not_retry_a_failure_after_subscription_was_established():
    from types import SimpleNamespace
    from typing import cast

    calls = []

    class Subscription:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def __aiter__(self):
            raise continuum_py.BrokerError("NotFound", "runtime subscription failed")
            yield  # make an async iterator without emitting any synthetic payload

    async def subscribe(*args, **kwargs):
        calls.append(kwargs)
        return Subscription()

    async def run():
        client = cast(continuum_py.Client, SimpleNamespace(subscribe=subscribe))
        following = vision_log.follow(client, [vision_log.OBJECTS], durable_name="activity-test", wait_for_streams_s=1)
        with pytest.raises(continuum_py.BrokerError, match="runtime subscription failed"):
            await anext(following)

    asyncio.run(run())
    assert calls == [{"start": "Live", "start_ns": 0, "durable_name": "activity-test", "pinned": True}]


def test_range_decodes_inline_headers_once_without_changing_mixed_locator_evidence(monkeypatch):
    from unittest.mock import call, create_autospec

    def header(n, start, *, shape=vision_log.OBJECTS):
        return continuum_py.encode_header(
            {
                "id": f"0190c8e0-0000-7000-8000-{n:012d}",
                "stream_id": 1,
                "start_ns": start,
                "end_ns": start + 100,
                "layer": "Layer1Raw",
                "format": "Text",
                "shape_name": shape,
            }
        )

    box = [0.1, 0.2, 0.3, 0.4, 0.9]
    inline = [
        item(
            header(2, 2_000_000_000),
            json.dumps({"cam": "cam0", "boxes": [box], "id": "forged", "ts_start": -1, "ts_end": -1}).encode(),
        ),
        item(header(1, 2_000_000_000), b'{"cam":"cam0","boxes":[],"error":null}'),
        item(header(3, 1_000_000_000), b'{"cam":"cam1","error":"detector failed"}'),
        item(header(4, 3_000_000_000), b'["not an object"]'),
        item(header(5, 4_000_000_000), b"not JSON"),
        # read_range uses its requested stream, without adding decode()'s shape-name restriction.
        item(header(6, 5_000_000_000, shape="speech.utterance"), b'{"cam":"cam1","boxes":[]}'),
    ]
    chunk = {"header": header(7, 1_500_000_000), "locator": {"kind": "Chunk", "uri": "/fixture/chunk.cnk"}}
    shm = {"header": header(8, 2_500_000_000), "locator": {"kind": "Shm", "arena": "fixture"}}
    empty = item(header(9, 6_000_000_000), None)

    class Broker:
        async def query(self, stream, start_ns, end_ns):
            raise NotImplementedError

    client = create_autospec(Broker, instance=True)
    client.query.return_value = {
        "items": [*inline, chunk, shm, empty],
        "complete": True,
        "truncated": False,
        "next_after": 0,
        "missing": [],
        "tail": None,
    }
    original_read = read.records

    async def fallback(client, stream, selected, *, start_ns, end_ns):
        if selected is chunk:
            return [
                read.Record("chunk-child", 1_500_000_000, 1_500_000_200, b'{"cam":"cam0","bbox":[0.1,0.2,0.3,0.4]}')
            ]
        if selected is shm:
            return [read.Record("slot-child", 2_500_000_000, 2_500_000_300, b'{"cam":"cam1","boxes":[]}')]
        return await original_read(client, stream, selected, start_ns=start_ns, end_ns=end_ns)

    fallback_read = create_autospec(original_read, side_effect=fallback)
    decode = create_autospec(continuum_py.decode_header, side_effect=continuum_py.decode_header)
    monkeypatch.setattr(read, "records", fallback_read)
    monkeypatch.setattr(read, "decode_header", decode)
    monkeypatch.setattr(continuum_py, "decode_header", decode)
    got = asyncio.run(vision_log.read_range(client, vision_log.OBJECTS, 0, 10))
    assert got == [
        {
            "id": "0190c8e0-0000-7000-8000-000000000003",
            "ts_start": 1.0,
            "ts_end": 1.0000001,
            "cam": "cam1",
            "error": "detector failed",
        },
        {"id": "chunk-child", "ts_start": 1.5, "ts_end": 1.5000002, "cam": "cam0", "bbox": [0.1, 0.2, 0.3, 0.4]},
        {
            "id": "0190c8e0-0000-7000-8000-000000000001",
            "ts_start": 2.0,
            "ts_end": 2.0000001,
            "cam": "cam0",
            "boxes": [],
            "error": None,
        },
        {
            "id": "0190c8e0-0000-7000-8000-000000000002",
            "ts_start": 2.0,
            "ts_end": 2.0000001,
            "cam": "cam0",
            "boxes": [box],
        },
        {"id": "slot-child", "ts_start": 2.5, "ts_end": 2.5000003, "cam": "cam1", "boxes": []},
        {
            "id": "0190c8e0-0000-7000-8000-000000000006",
            "ts_start": 5.0,
            "ts_end": 5.0000001,
            "cam": "cam1",
            "boxes": [],
        },
    ]
    client.query.assert_awaited_once_with(vision_log.OBJECTS, 0, 10_000_000_000)
    inline_headers = [selected["header"] for selected in inline]
    inline_decodes = [seen.args[0] for seen in decode.call_args_list if seen.args[0] in inline_headers]
    assert len(inline_decodes) == len(inline_headers)
    assert inline_decodes == inline_headers
    assert fallback_read.await_args_list == [
        call(client, vision_log.OBJECTS, chunk, start_ns=0, end_ns=10_000_000_000),
        call(client, vision_log.OBJECTS, shm, start_ns=0, end_ns=10_000_000_000),
        call(client, vision_log.OBJECTS, empty, start_ns=0, end_ns=10_000_000_000),
    ]


@pytest.mark.parametrize("compact", [False, True], ids=["older-sdk", "compact-sdk"])
def test_range_compact_batch_preserves_python_json_evidence_and_mixed_fallback(monkeypatch, compact):
    import math
    from unittest.mock import call, create_autospec

    def header(n, start):
        return continuum_py.encode_header(
            {
                "id": f"0190c8e0-0000-7000-8000-{n:012d}",
                "stream_id": 1,
                "start_ns": start,
                "end_ns": start + 100,
                "layer": "Layer1Raw",
                "format": "Text",
                "shape_name": vision_log.OBJECTS,
            }
        )

    bodies = [
        b'{"cam":"cam0","boxes":[],"id":"forged","ts_start":-1,"ts_end":-1,"special":NaN,"large":1234567890123456789012345678901234567890}',
        b'{"cam":"cam1","error":"failed"}',
        b"not JSON",
        b"[1,2]",
    ]
    inline = [item(header(n, n * 1_000_000_000), body) for n, body in enumerate(bodies, 1)]
    chunk = {"header": header(5, 500_000_000), "locator": {"kind": "Chunk", "uri": "/fixture/chunk.cnk"}}
    shm = {"header": header(6, 2_500_000_000), "locator": {"kind": "Shm", "arena": "fixture"}}
    empty = item(header(7, 6_000_000_000), None)
    entries = [*inline, chunk, shm, empty]

    class OlderBroker:
        async def query(self, stream, start_ns, end_ns):
            return {"items": entries}

    class CompactBroker(OlderBroker):
        async def query_inline_records(self, stream, start_ns, end_ns):
            return [
                (f"0190c8e0-0000-7000-8000-{n:012d}", n * 1_000_000_000, n * 1_000_000_000 + 100, body)
                for n, body in enumerate(bodies, 1)
            ] + [chunk, shm, empty]

    broker = CompactBroker() if compact else OlderBroker()
    query = create_autospec(broker.query, side_effect=broker.query)
    monkeypatch.setattr(broker, "query", query)
    batch = None
    if compact:
        assert isinstance(broker, CompactBroker)
        batch = create_autospec(broker.query_inline_records, side_effect=broker.query_inline_records)
        monkeypatch.setattr(broker, "query_inline_records", batch)
    original_read = read.records

    async def fallback(client, stream, selected, *, start_ns, end_ns):
        if selected is chunk:
            return [read.Record("chunk-child", 500_000_000, 500_000_100, b'{"cam":"cam0","bbox":[0.1,0.2,0.3,0.4]}')]
        if selected is shm:
            return [read.Record("slot-child", 2_500_000_000, 2_500_000_100, b'{"cam":"cam1","boxes":[]}')]
        return await original_read(client, stream, selected, start_ns=start_ns, end_ns=end_ns)

    fallback_read = create_autospec(original_read, side_effect=fallback)
    monkeypatch.setattr(read, "records", fallback_read)
    got = asyncio.run(vision_log.read_range(cast(continuum_py.Client, broker), vision_log.OBJECTS, 0, 10))
    assert math.isnan(got[1].pop("special"))
    assert got == [
        {"id": "chunk-child", "ts_start": 0.5, "ts_end": 0.5000001, "cam": "cam0", "bbox": [0.1, 0.2, 0.3, 0.4]},
        {
            "id": "0190c8e0-0000-7000-8000-000000000001",
            "ts_start": 1.0,
            "ts_end": 1.0000001,
            "cam": "cam0",
            "boxes": [],
            "large": 1234567890123456789012345678901234567890,
        },
        {
            "id": "0190c8e0-0000-7000-8000-000000000002",
            "ts_start": 2.0,
            "ts_end": 2.0000001,
            "cam": "cam1",
            "error": "failed",
        },
        {"id": "slot-child", "ts_start": 2.5, "ts_end": 2.5000001, "cam": "cam1", "boxes": []},
    ]
    assert fallback_read.await_args_list == [
        call(broker, vision_log.OBJECTS, selected, start_ns=0, end_ns=10_000_000_000)
        for selected in (chunk, shm, empty)
    ]
    if compact:
        assert batch is not None
        batch.assert_awaited_once_with(vision_log.OBJECTS, 0, 10_000_000_000)
        query.assert_not_awaited()
    else:
        query.assert_awaited_once_with(vision_log.OBJECTS, 0, 10_000_000_000)


def test_decoding_is_counted_for_whoever_asked_and_summarized_each_minute(monkeypatch, caplog):
    from continuum_py import read

    rows = [
        read.Record(f"0190c8e0-0000-7000-8000-{i:012d}", 1_000_000_000 * i, 1_000_000_000 * i, b'{"cam": "cam0"}')
        for i in range(3)
    ]
    monkeypatch.setattr(vision_log, "_decoded", collections.Counter())
    monkeypatch.setattr(vision_log, "_decoded_since", 0.0)
    clock = [10.0]
    monkeypatch.setattr(vision_log.time, "monotonic", lambda: clock[0])
    caplog.set_level("INFO", logger="liberational.vision-log")
    token = continuum_link.CALLER.set("people-read dashboard.server._people_view")
    try:
        vision_log._decode_records(rows)  # 10 s since the last summary: counted, not logged
        clock[0] = 70.0
        vision_log._decode_records(rows[:1])
    finally:
        continuum_link.CALLER.reset(token)
    [line] = [r.getMessage() for r in caplog.records if "vision log decoded" in r.getMessage()]
    assert line == "vision log decoded 4 records in 70 s: people-read dashboard.server._people_view 4"


def test_the_inline_fast_path_is_counted_too(monkeypatch):
    class Client:
        async def query_inline_records(self, stream, lo, hi):
            return [(f"0190c8e0-0000-7000-8000-{i:012d}", lo + i, lo + i, b'{"cam": "cam0"}') for i in range(5)]

    monkeypatch.setattr(vision_log, "_decoded", collections.Counter())
    monkeypatch.setattr(vision_log, "_decoded_since", vision_log.time.monotonic())
    token = continuum_link.CALLER.set("live-scene dashboard.live.scene")
    try:
        found = asyncio.run(
            vision_log.read_range(cast(continuum_py.Client, Client()), vision_log.OBJECTS, 100.0, 200.0)
        )
    finally:
        continuum_link.CALLER.reset(token)
    assert len(found) == 5
    assert vision_log._decoded["live-scene dashboard.live.scene"] == 5


class _ThreadLink:
    """A broker link's shape: `call` runs on this link's one thread, as each real link runs on its own loop."""

    def __init__(self, socket: object = None, *, client_name: str, timeout: float = 10.0) -> None:
        import concurrent.futures

        self.name, self.timeout = client_name, timeout
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix=client_name)
        self.closed = False

    def call(self, fn, timeout: float | None = None):
        def run():
            got = fn(None)
            return (
                asyncio.run(got) if asyncio.iscoroutine(got) else got
            )  # awaited on this link's thread, as a loop does

        return self.pool.submit(run).result(timeout=timeout or 5)

    def close(self) -> None:
        self.closed = True
        self.pool.shutdown(wait=False)


def test_a_picture_decodes_apart_from_every_other_read(monkeypatch: pytest.MonkeyPatch) -> None:
    # After a restart, face shots decoded H.264 on the shared link's thread and plain reads timed out behind
    # them. A picture read must answer while that link is busy, and a read of the main link while a picture decodes.
    import threading

    monkeypatch.setattr(vision_log, "BrokerLink", _ThreadLink)
    release = threading.Event()
    log = vision_log.VisionLog(client_name="vision-log")

    async def stuck(_client):
        return release.wait(5)

    busy = threading.Thread(target=lambda: log.link.call(stuck))
    busy.start()  # the main link is stuck on a long read

    async def evidence(_client, cam, ts):
        return (cam, ts)

    async def frame(_client, cam, ts, _within):
        return (cam, ts)

    monkeypatch.setattr(vision_log, "frame_evidence_at", evidence)
    monkeypatch.setattr(vision_log, "frame_at", frame)
    try:
        assert log.frame_evidence("cam0", 12.5) == ("cam0", 12.5)
        assert log.frame("cam0", 13.0) == ("cam0", 13.0)
    finally:
        release.set()
        busy.join()
    frames = log.frames_link()
    assert frames is log.frames_link(), "one picture link, made once"
    assert cast(_ThreadLink, frames).name == "vision-log-frames"
    log.close()
    assert cast(_ThreadLink, frames).closed and cast(_ThreadLink, log.link).closed


def test_one_scene_read_serves_every_caller_for_a_few_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    # Each /scene poll read the whole hour of captions (0.4 to 1.5 s holding the GIL); several pages polling
    # stalled every other request. One read now serves all callers for SCENE_TTL_S.
    clock = [1000.0]
    monkeypatch.setattr(vision_log.time, "monotonic", lambda: clock[0])
    reads: list[float] = []

    def stored(start: float, end: float) -> list[dict[str, Any]]:
        reads.append(start)
        return [{"id": f"c{len(reads)}", "cam": "cam0", "ts_start": end - 2, "caption": "x"}]

    log = vision_log.VisionLog(client_name="scene-test")
    monkeypatch.setattr(log, "_captions", stored)
    first = log.scene(5000.0)
    clock[0] += vision_log.VisionLog.SCENE_TTL_S - 0.1
    assert log.scene(5004.0) == first and len(reads) == 1, "within a few seconds: the same read"
    clock[0] += 0.2
    assert log.scene(5005.0)[0]["id"] == "c2" and len(reads) == 2, "then a fresh one"
    log.close()


def test_callers_during_a_scene_read_wait_for_it_rather_than_read_again(monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    started, release = threading.Event(), threading.Event()
    reads: list[float] = []

    def slow(start: float, end: float) -> list[dict[str, Any]]:
        reads.append(start)
        started.set()
        release.wait(5)
        return [{"id": "c1", "cam": "cam0", "ts_start": end - 2, "caption": "x"}]

    log = vision_log.VisionLog(client_name="scene-test")
    monkeypatch.setattr(log, "_captions", slow)
    results: list[list[dict[str, Any]]] = []
    callers = [threading.Thread(target=lambda: results.append(log.scene(5000.0))) for _ in range(8)]
    for caller in callers:
        caller.start()
    started.wait(5)
    release.set()
    for caller in callers:
        caller.join(5)
    assert len(reads) == 1 and len(results) == 8 and all(r[0]["id"] == "c1" for r in results)
    log.close()


def test_captions_for_some_cameras_read_only_those_cameras(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[Any] = []

    def stored(start: float, end: float, cams: list[str] | None = None) -> list[dict[str, Any]]:
        asked.append(cams)
        return [
            {"id": "c1", "cam": "cam0", "ts_start": start + 1, "caption": "x"},
            {"id": "c2", "cam": "cam1", "ts_start": start + 2, "caption": "y"},
        ]

    log = vision_log.VisionLog(client_name="captions-test")
    monkeypatch.setattr(log, "_captions", stored)
    assert [r["id"] for r in log.captions(100.0, 200.0, ("cam0",))] == ["c1"]
    assert [r["id"] for r in log.captions(100.0, 200.0)] == ["c1", "c2"]
    assert asked == [["cam0"], None], "one camera's read, then every camera's"
    log.close()
