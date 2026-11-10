"""Ingested media registered in place on a real Continuum broker."""

import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest
from continuum_py import Client, decode_header

from liberational.ingest.broker import BrokerRegistrar, Refused
from liberational.ingest.pipeline import Incoming, Pipeline
from liberational.ingest.store import Store

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


async def found(client: Client, stream: str, ids: list[str]) -> int:
    """How many of `ids` the broker holds on `stream`, following the commit cursor once."""
    missing, after = set(ids), 0
    while missing:
        result = await client.query_by_seq(stream, after_seq=after, max_items=1000)
        missing.difference_update(decode_header(item["header"])["id"] for item in result["items"])
        after = result["next_after"]
        if not after:
            break
    return len(ids) - len(missing)


def held(sock: Path, stream: str, ids: list[str]) -> int:
    async def go() -> int:
        async with await Client.connect(str(sock), client_name="test") as c:
            return await found(c, stream, ids)

    return asyncio.run(go())


def test_a_video_with_sound_lands_on_its_streams_once_even_registered_twice(tmp_path, broker):
    clip = tmp_path / "talk.mov"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10", "-f", "lavfi",
         "-i", "sine=frequency=440", "-t", "2", "-c:v", "libx264", "-bf", "2", "-c:a", "aac",
         "-metadata", "creation_time=2026-05-01T12:00:00Z", str(clip)],
        check=True,
    )  # fmt: skip
    pipe = Pipeline(Store(tmp_path / "ingest.sqlite3"), tmp_path / "ingest", BrokerRegistrar(broker))
    item = pipe.ingest(Incoming(clip, "upload", "talk.mov"))
    assert item.status == "done", item.detail
    chunks = {o["stream"]: o["chunk_id"] for o in item.outputs}
    assert set(chunks) == {"ingest-upload.video_v1", "ingest-upload.audio_v1"}
    again = pipe.process(item.id)
    assert {o["stream"]: o["chunk_id"] for o in again.outputs} == chunks  # idempotent: the same chunks
    for stream, chunk in chunks.items():
        assert held(broker, stream, [chunk]) == 1


def test_a_file_the_broker_cant_take_is_refused_with_its_reason(tmp_path, broker):
    not_h264 = tmp_path / "x.mp4"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10", "-t", "1",
         "-c:v", "mpeg4", str(not_h264)],
        check=True,
    )  # fmt: skip
    with pytest.raises(Refused):
        BrokerRegistrar(broker).video("ingest-test", not_h264, 1_700_000_000.0, 10.0)


def test_no_broker_is_a_refusal_that_says_where_it_looked(tmp_path):
    with pytest.raises(Refused, match="isn't reachable"):
        BrokerRegistrar(tmp_path / "none.sock").audio("ingest-test", tmp_path / "a.flac", 1_700_000_000.0)


def test_item_purge_cascades_all_analysis_and_keeps_unrelated_rows(tmp_path, broker):
    from continuum_py import decode_header

    from liberational import vision_log
    from test_ingest import make_clip

    pipe = Pipeline(Store(tmp_path / "ingest.sqlite3"), tmp_path / "ingest", BrokerRegistrar(broker))
    item = pipe.ingest(Incoming(make_clip(tmp_path / "source.mov"), "upload", "source.mov"))
    assert item.status == "done"
    roots = {o["kind"]: o["chunk_id"] for o in item.outputs}
    assert item.original_ts is not None
    ns = int(item.original_ts * 1e9)
    streams = [vision_log.CAPTIONS, vision_log.OBJECTS]

    async def seed():
        async with await Client.connect(str(broker), client_name="test-delete") as client:
            registered = {name: await vision_log.register(client, name) for name in streams}
            record_seq = 0

            def text(stream, record, parents, at_ns):
                nonlocal record_seq
                record_seq += 1
                header, body = vision_log.encode(
                    stream, registered[stream], record, parents=parents, span_ns=(at_ns, at_ns + 1), seq=record_seq
                )
                client.ingest(header, inline=body)
                return decode_header(header)["id"]

            for cam, parents in [(item.stream, roots), ("unrelated-camera", {})]:
                at_ns = ns if parents else ns + 100_000_000
                owned = [parents["video"]] if parents else []
                text(vision_log.CAPTIONS, {"cam": cam, "caption": "An indoor scene"}, owned, at_ns)
                text(vision_log.OBJECTS, {"cam": cam, "objects": []}, owned, at_ns)
            # Ingest is one-way: wait until the real catalog has committed the complete graph.
            async with asyncio.timeout(5):
                while True:
                    counts = {
                        name: len((await client.query(name, ns - 1_000_000_001, ns + 3_000_000_000))["items"])
                        for name in streams
                    }
                    if counts == dict.fromkeys(streams, 2):
                        return counts
                    await asyncio.sleep(0)

    assert asyncio.run(seed()) == dict.fromkeys(streams, 2)
    purger = BrokerRegistrar(broker)
    purge_id = purger.purge_item(item)
    assert purge_id is not None
    assert not purger.purge_finished(purge_id)  # broker grace/keeper completion is separate from catalog removal
    assert purger.purge_item(item) == purge_id  # recover a lost response without a second empty purge

    async def remaining():
        async with await Client.connect(str(broker), client_name="verify-delete") as client:
            return {
                name: len((await client.query(name, ns - 1_000_000_001, ns + 3_000_000_000))["items"])
                for name in streams
            }

    assert asyncio.run(remaining()) == dict.fromkeys(streams, 1)  # the unrelated camera's rows stay
    assert held(broker, "ingest-upload.video_v1", [roots["video"]]) == 0
    assert held(broker, "ingest-upload.audio_v1", [roots["audio"]]) == 0
