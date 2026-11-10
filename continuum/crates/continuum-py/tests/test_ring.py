"""Following a ring live from asyncio: latency from a producer in another process, the event loop staying free,
overruns, and cancellation.

Run with `mise run continuum-test`, which builds the producer first. It comes from CONTINUUM_PCM_PRODUCER, or
continuum/target/debug/examples/pcm_producer. Each record the producer writes starts with the CLOCK_MONOTONIC time
it was published (u64 little-endian), then its block number.
"""

import asyncio
import os
import statistics
import struct
import subprocess
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from continuum_py import Arena, RingReader

PRODUCER = Path(
    os.environ.get(
        "CONTINUUM_PCM_PRODUCER",
        Path(__file__).resolve().parents[3] / "target" / "debug" / "examples" / "pcm_producer",
    )
)
STAMP = struct.Struct("<QQ")  # publish time (CLOCK_MONOTONIC ns), block number
STRIDE = 1920  # 20 ms of 48 kHz mono s16
TARGET_MS = 25.0  # capture to live reader (docs/CONTINUUM-BROKER.md, "Rings")
# The 25 ms target is asserted only on request (CONTINUUM_LATENCY_STRICT=1, on an otherwise idle machine): a shared
# box under load can delay a few wakes past it. By default the test asserts the bound a missed wake would break.
STRICT = os.environ.get("CONTINUUM_LATENCY_STRICT") == "1"
MISSED_WAKE_MS = 250.0


def mono_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC)  # the clock the producer stamps with, on Linux and macOS


@pytest.fixture
def ring_arena() -> Iterator[Arena]:
    name = f"/cnt.r.{os.getpid()}.{time.monotonic_ns() % 10**6}"
    arena = Arena.create(name, classes=[], rings=[(STRIDE, 500), (STRIDE, 16)])
    yield arena
    Arena.unlink(name)


def produce(arena: Arena, ring: int, blocks: int, period_us: int) -> subprocess.Popen[bytes]:
    return subprocess.Popen([PRODUCER, arena.name, str(ring), str(blocks), str(period_us)], stderr=subprocess.DEVNULL)


def test_a_live_reader_sees_each_block_within_the_latency_target(ring_arena: Arena) -> None:
    blocks = 100  # two seconds of 20 ms blocks

    async def run() -> tuple[list[int], list[float], int]:
        reader = ring_arena.ring_reader(0, from_oldest=False)
        producer = produce(ring_arena, 0, blocks, 20_000)
        seen: list[int] = []
        latencies: list[float] = []
        try:
            async for _seq, record, _lost in reader:
                published, n = STAMP.unpack_from(record)
                latencies.append((mono_ns() - published) / 1e6)
                seen.append(n)
                if len(seen) == blocks:
                    break
        finally:
            assert producer.wait(timeout=10) == 0
        return seen, latencies, reader.lost

    seen, latencies, lost = asyncio.run(asyncio.wait_for(run(), timeout=30))
    assert seen == list(range(blocks)), "every block, in order"
    assert lost == 0
    p95 = statistics.quantiles(latencies, n=20)[-1]
    print(
        f"capture to reader: median {statistics.median(latencies):.3f} ms, p95 {p95:.3f} ms, max {max(latencies):.3f} ms"
    )
    assert statistics.median(latencies) < MISSED_WAKE_MS, "wakes are being missed (each costs a whole wait slice)"
    if STRICT:
        assert p95 < TARGET_MS, f"p95 {p95:.2f} ms is over the {TARGET_MS} ms target"


def test_waiting_for_a_record_blocks_neither_the_event_loop_nor_other_threads(ring_arena: Arena) -> None:
    ticks = 0
    spins = 0
    stop = threading.Event()

    def spin() -> None:  # needs the GIL to count
        nonlocal spins
        while not stop.is_set():
            spins += 1

    async def tick() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    async def run() -> object:
        reader = ring_arena.ring_reader(0)
        ticker = asyncio.create_task(tick())
        got = await reader.next_record(timeout_ms=300)  # nothing is written
        ticker.cancel()
        return got

    spinner = threading.Thread(target=spin)
    spinner.start()
    started = time.monotonic()
    try:
        got = asyncio.run(run())
    finally:
        stop.set()
        spinner.join()
    assert got is None, "timed out with nothing published"
    assert time.monotonic() - started >= 0.3
    assert ticks >= 10, f"the event loop ran only {ticks} ticks while the read waited"
    assert spins > 0


def test_a_reader_that_falls_behind_skips_ahead_and_reports_what_it_lost(ring_arena: Arena) -> None:
    reader = ring_arena.ring_reader(1, from_oldest=False)  # capacity 16
    producer = produce(ring_arena, 1, 1000, 0)  # as fast as it can, with nobody reading
    try:
        assert producer.wait(timeout=10) == 0, "the writer finished without waiting for the reader"
    finally:
        producer.kill()

    async def drain() -> list[tuple[int, int, int]]:
        got = []
        while (record := await reader.next_record(timeout_ms=100)) is not None:
            seq, data, lost = record
            got.append((seq, STAMP.unpack_from(data)[1], lost))
        return got

    got = asyncio.run(drain())
    first_seq, first_block, first_lost = got[0]
    assert first_lost == first_seq == first_block, "it skipped straight to the oldest record still there"
    assert first_seq >= 1000 - 16
    assert [g[0] for g in got] == list(range(first_seq, 1000)), "then every record after it, in order"
    assert all(lost == 0 for _, _, lost in got[1:])
    assert reader.lost == first_lost


def test_a_cancelled_wait_loses_no_record(ring_arena: Arena) -> None:
    async def run() -> tuple[int, int]:
        reader: RingReader = ring_arena.ring_reader(0, from_oldest=False)
        pending = asyncio.ensure_future(reader.next_record())
        await asyncio.sleep(0.05)  # the wait is now blocked on the ring
        producer = produce(ring_arena, 0, 1, 0)
        try:
            pending.cancel()  # the record lands while the cancelled wait's thread is still waiting
            with pytest.raises(asyncio.CancelledError):
                await pending
            record = await reader.next_record(timeout_ms=5000)
            assert producer.wait(timeout=10) == 0
        finally:
            producer.kill()
        assert record is not None, "the cancelled wait didn't take the record with it"
        seq, data, _ = record
        return seq, STAMP.unpack_from(data)[1]

    assert asyncio.run(run()) == (0, 0)
