"""The recent H.264 window from Python, against a real broker and tier worker: frames that reached a Tier 1
chunk, frames still in Tier 0, and frames written after the window opened come out in decode order from the keyframe
before the start, with a lost frame reported as a gap until the next keyframe, and only attested values."""

import asyncio
import time
from pathlib import Path

import pytest
from continuum_py import Client, H264AccessUnit, H264Gap, ShmLocation, decode_header, encode_header, open_h264_window

from conftest import spawn_broker, spawn_tier_worker, stop_broker

GOP = 10
FRAME_NS = 33_333_333
EPOCH = 3
LOST = 35


def uuid_of(n: int) -> str:
    return f"0199a000-0000-7000-8000-{n:012x}"


def access_unit(i: int) -> bytes:
    """A keyframe is SPS, PPS, and an IDR; the rest are P slices; each carries its index."""
    nals = (
        b"\x00\x00\x00\x01\x67\x42\x00\x00\x00\x01\x68\xce\x00\x00\x01\x65" if i % GOP == 0 else b"\x00\x00\x00\x01\x41"
    )
    return nals + i.to_bytes(4, "little") + bytes((k + i) % 251 for k in range(200))


def in_slot(client: Client, data: bytes) -> ShmLocation:
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


def test_a_window_reads_tier_1_tier_0_and_live_in_decode_order_with_explicit_gaps(tmp_path: Path) -> None:
    socket = tmp_path / "broker.sock"
    broker = spawn_broker(socket, tier0_classes="16384x256")
    t0 = time.time_ns()

    def at(i: int) -> int:
        return t0 + i * FRAME_NS

    async def write(c: Client, stream: int, i: int) -> None:
        if i != LOST:
            header = encode_header(
                {
                    "id": uuid_of(i + 1),
                    "stream_id": stream,
                    "start_ns": at(i),
                    "end_ns": at(i),
                    "seq": i,
                    "layer": "Layer1Raw",
                    "format": "EncodedVideo",
                    "shape_name": "H264AnnexB",
                    "capture": {
                        "capture_mono_ns": 0,
                        "device_ts": 0,
                        "arrival_ns": 0,
                        "stride": 0,
                        "width": 64,
                        "height": 48,
                        "flip": False,
                        "device_seq": i,
                        "capture_epoch": EPOCH,
                    },
                }
            )
            c.ingest(header, shm=in_slot(c, access_unit(i)))
        await asyncio.sleep(FRAME_NS / 1e9)

    async def run() -> list[H264AccessUnit | H264Gap]:
        async with (
            await Client.connect(socket) as c,
            await Client.connect(socket, subscription_overflow="backpressure") as live,
        ):
            registered = await c.register_stream(
                "cam.video", kind="sensor", format="EncodedVideo", shape_name="H264AnnexB", slab_class=0
            )
            stream = registered["stream_id"]
            worker = spawn_tier_worker(socket, tmp_path / "chunks", "cam.video")
            try:
                for i in range(30):
                    await write(c, stream, i)
                for _ in range(500):  # a chunk closes after a second without frames: wait (bounded) for its commit
                    found = await c.query("cam.video", t0, t0 + 10**10)
                    if any(it["locator"] and it["locator"].get("kind") == "Chunk" for it in found["items"]):
                        break
                    await asyncio.sleep(0.02)
                for i in range(30, 33):
                    await write(c, stream, i)
                window = await open_h264_window(c, live, "cam.video", at(12))
                items: list[H264AccessUnit | H264Gap] = []
                for i in range(33, 45):
                    await write(c, stream, i)
                while not items or items[-1].get("device_seq") != 44:
                    item = await asyncio.wait_for(window.next(), timeout=10)
                    assert item is not None
                    items.append(item)
                return items
            finally:
                worker.kill()
                worker.wait()

    try:
        items = asyncio.run(run())
    finally:
        stop_broker(broker, socket)

    seen = [(it["type"], it["device_seq"] if it["type"] == "au" else it["why"]) for it in items]
    expected = [("au", i) for i in range(10, LOST)] + [("gap", "missing")] + [("au", i) for i in range(40, 45)]
    assert seen == expected, "from the keyframe at 10, through the loss at 35, to the keyframe at 40"
    gap = next(it for it in items if it["type"] == "gap")
    assert (gap["from_ns"], gap["to_ns"]) == (at(LOST - 1), at(40))
    aus = [it for it in items if it["type"] == "au"]
    assert all(au["bytes"] == access_unit(au["device_seq"] or 0) for au in aus)
    assert all(
        (au["capture_epoch"], au["width"], au["height"], au["id"]) == (EPOCH, 64, 48, uuid_of(au["seq"] + 1))
        for au in aus
    )
    assert [au["random_access"] for au in aus if au["seq"] % GOP == 0] == [True, True, True, True]
    chunked = [au for au in aus if au["tier"] == "tier1"]
    hot = [au for au in aus if au["tier"] == "tier0"]
    assert chunked and hot, "the window crossed from Tier 1 into Tier 0"
    assert all(au["header"] is None for au in chunked), "a chunk doesn't keep the whole header, so none is claimed"
    first_hot = hot[0]
    assert first_hot["header"] is not None
    assert decode_header(first_hot["header"])["id"] == first_hot["id"]


def test_a_start_before_the_window_is_refused(tmp_path: Path) -> None:
    socket = tmp_path / "broker.sock"
    broker = spawn_broker(socket)

    async def run() -> None:
        async with (
            await Client.connect(socket) as c,
            await Client.connect(socket, subscription_overflow="backpressure") as live,
        ):
            await c.register_stream(
                "cam.video", kind="sensor", format="EncodedVideo", shape_name="H264AnnexB", slab_class=0
            )
            now = time.time_ns()
            with pytest.raises(ValueError, match="before the recent window"):
                await open_h264_window(c, live, "cam.video", now - 301 * 10**9, now_ns=now)

    try:
        asyncio.run(run())
    finally:
        stop_broker(broker, socket)
