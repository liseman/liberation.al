"""Subscriptions from Python against a real broker: a live subscription as an async iterator, a time start that
catches up then goes live, a durable one resuming after its acknowledged position, the subject and shape filters
with a pinned locator, unsubscribing from another task ending a waiting reader, and `take` waiting on a thread.

Run with `mise run continuum-test`, which builds the broker first.
"""

import asyncio
import time

import pytest
from continuum_py import Client, RingLease, StreamLease, Subscription, SubscriptionEvent, decode_header, encode_header


def uuid_of(n: int, tag: int = 0) -> str:
    raw = bytearray(16)
    raw[6], raw[8], raw[14], raw[15] = 0x70, 0x80, tag, n
    h = raw.hex()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


async def producer(socket: str) -> tuple[Client, int]:
    p = await Client.connect(socket, client_name="producer")
    registered = await p.register_stream("cam0", kind="sensor", format="RawBytes", shape_name="frame", slab_class=0)
    return p, registered["stream_id"]


def ingest(p: Client, stream: int, n: int, start: int, *, subjects: tuple[int, ...] = (), shape: str = "frame") -> None:
    header = encode_header(
        {
            "id": uuid_of(n),
            "stream_id": stream,
            "start_ns": start,
            "end_ns": start + 100,
            "layer": "Layer1Raw",
            "format": "RawBytes",
            "shape_name": shape,
            "subjects": [uuid_of(s, 0xEE) for s in subjects],
        }
    )
    p.ingest(header, inline=bytes([n]))


async def take(sub: Subscription, n: int, *, skip_caught_up: bool = False) -> list[tuple[int, int]]:
    """The next n events as (payload number, position)."""
    got: list[tuple[int, int]] = []
    while len(got) < n:
        event = await asyncio.wait_for(sub.next(), timeout=5)
        assert event is not None, "the subscription ended"
        if event["type"] == "CaughtUp" and skip_caught_up:
            continue
        assert event["type"] == "Event", event
        header = decode_header(event["item"]["header"])
        got.append((int(header["id"][-2:], 16), event["position"]))
    return got


def test_a_live_subscription_iterates_commits_in_order(socket: str) -> None:
    async def run() -> list[tuple[int, int]]:
        p, cam = await producer(socket)
        async with await Client.connect(socket) as c:
            sub = await c.subscribe(["cam0"])
            assert sub.live_at_start
            for n in (1, 2, 3):
                ingest(p, cam, n, n * 100)
            got: list[tuple[int, int]] = []
            async with sub:
                async for event in sub:
                    assert event["type"] == "Event"
                    got.append((int(decode_header(event["item"]["header"])["id"][-2:], 16), event["position"]))
                    if len(got) == 3:
                        break
            return got

    got = asyncio.run(run())
    assert [n for n, _ in got] == [1, 2, 3]
    assert got[0][1] < got[1][1] < got[2][1]


def test_a_time_start_catches_up_then_goes_live(socket: str) -> None:
    async def run() -> tuple[list[tuple[int, int]], SubscriptionEvent, list[tuple[int, int]]]:
        p, cam = await producer(socket)
        async with await Client.connect(socket) as c:
            live = await c.subscribe(["cam0"])
            for n in (1, 2, 3):
                ingest(p, cam, n, n * 100)
            await take(live, 3)  # committed by now
            await live.unsubscribe()
            sub = await c.subscribe(["cam0"], start="Time", start_ns=250)
            assert not sub.live_at_start
            history = await take(sub, 2)
            caught = await asyncio.wait_for(sub.next(), timeout=5)
            assert caught is not None
            ingest(p, cam, 4, 400)
            return history, caught, await take(sub, 1)

    history, caught, after = asyncio.run(run())
    assert [n for n, _ in history] == [2, 3], "payload 2 reaches into 250, payload 1 doesn't"
    assert caught["type"] == "CaughtUp" and caught["position"] >= history[1][1]
    assert after[0][0] == 4 and after[0][1] > caught["position"]


def test_a_durable_subscription_resumes_after_its_ack(socket: str) -> None:
    async def run() -> tuple[int, int, list[tuple[int, int]]]:
        p, cam = await producer(socket)
        c = await Client.connect(socket)
        sub = await c.subscribe(["cam0"], start="Saved", durable_name="journal")
        assert sub.start_position == 0
        for n in (1, 2, 3):
            ingest(p, cam, n, n * 100)
        got = await take(sub, 3, skip_caught_up=True)
        sub.ack(got[1][1])
        await sub.unsubscribe()
        # The cursor is saved through the writer: a restarted consumer may need a moment to see it.
        d = await Client.connect(socket)
        for _ in range(250):
            resumed = await d.subscribe(["cam0"], start="Saved", durable_name="journal")
            if resumed.start_position == got[1][1]:
                return got[1][1], resumed.start_position, await take(resumed, 1)
            await resumed.unsubscribe()
            await asyncio.sleep(0.02)
        raise AssertionError("the acknowledged position wasn't saved")

    acked, start, rest = asyncio.run(run())
    assert start == acked
    assert rest[0][0] == 3


def test_filters_and_a_pinned_locator(socket: str) -> None:
    async def run() -> tuple[SubscriptionEvent, SubscriptionEvent]:
        p, cam = await producer(socket)
        async with await Client.connect(socket) as c:
            about = await c.subscribe(["cam0"], subjects=[uuid_of(7, 0xEE)], pinned=True)
            shaped = await c.subscribe(["cam0"], shape_name="pose")
            ingest(p, cam, 1, 100, subjects=(8,))
            ingest(p, cam, 2, 200, subjects=(7, 8))
            ingest(p, cam, 3, 300, shape="pose")
            first = await asyncio.wait_for(about.next(), timeout=5)
            second = await asyncio.wait_for(shaped.next(), timeout=5)
            assert first is not None and second is not None
            return first, second

    about, shaped = asyncio.run(run())
    assert "item" in about and "item" in shaped
    assert decode_header(about["item"]["header"])["id"] == uuid_of(2)
    assert about["item"]["locator"] == {"kind": "Inline", "bytes": b"\x02"}
    assert decode_header(shaped["item"]["header"])["id"] == uuid_of(3)
    assert shaped["item"]["locator"] is None


def test_unsubscribing_from_another_task_ends_a_waiting_reader(socket: str) -> None:
    async def run() -> object:
        await producer(socket)
        async with await Client.connect(socket) as c:
            sub = await c.subscribe(["cam0"])
            reader = asyncio.ensure_future(sub.next())
            await asyncio.sleep(0.05)
            await sub.unsubscribe()
            end = await asyncio.wait_for(reader, timeout=5)
            await sub.unsubscribe()  # already ended: nothing to do
            return end

    assert asyncio.run(run()) is None


def test_take_on_a_thread_returns_every_event_waiting_in_order_and_none_once_unsubscribed(socket: str) -> None:
    # #3235: a decoder thread waits here for a batch, not on the event loop for each event.
    def numbers(events: list[SubscriptionEvent]) -> list[int]:
        assert all(event["type"] == "Event" for event in events), events
        return [int(decode_header(event["item"]["header"])["id"][-2:], 16) for event in events]

    async def run() -> tuple[list[int], list[int], list[int], list[SubscriptionEvent] | None, float]:
        p, cam = await producer(socket)
        async with await Client.connect(socket) as c:
            sub = await c.subscribe(["cam0"])
            first = await asyncio.to_thread(sub.take, 0.05)  # nothing yet: the timeout passes with none
            for n in (1, 2, 3):
                ingest(p, cam, n, n * 100)
            await asyncio.sleep(0.5)  # all three have arrived and wait
            two = await asyncio.to_thread(sub.take, 5.0, 2)
            rest = await asyncio.to_thread(sub.take, 5.0)
            waiting = asyncio.ensure_future(asyncio.to_thread(sub.take, 30.0))
            await asyncio.sleep(0.05)  # the thread is waiting, the loop still runs: the GIL is released
            began = time.monotonic()
            await sub.unsubscribe()
            end = await asyncio.wait_for(waiting, timeout=5)
            return numbers(first or []), numbers(two or []), numbers(rest or []), end, time.monotonic() - began

    first, two, rest, end, ended_s = asyncio.run(run())
    assert (first, two, rest) == ([], [1, 2], [3])
    assert end is None
    assert ended_s < 5, "an unsubscribe ends a waiting take at once, not at its timeout"


def test_take_refuses_a_bad_timeout_or_max(socket: str) -> None:
    async def run() -> None:
        await producer(socket)
        async with await Client.connect(socket) as c, await c.subscribe(["cam0"]) as sub:
            for timeout_s, most in ((-1.0, 64), (float("nan"), 64), (0.0, 0)):
                with pytest.raises(ValueError):
                    sub.take(timeout_s, most)

    asyncio.run(run())


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param({"start": "Saved"}, "durable_name", id="saved-without-name"),
        pytest.param({"start": "Later"}, "Live, Time, or Saved", id="bad-start"),
        pytest.param({"subjects": ["someone"]}, "canonical UUID", id="bad-subject"),
    ],
)
def test_bad_arguments_are_refused_before_sending(socket: str, kwargs: dict[str, object], match: str) -> None:
    async def run() -> None:
        async with await Client.connect(socket) as c:
            with pytest.raises(ValueError, match=match):
                await c.subscribe(["cam0"], **kwargs)  # ty: ignore[invalid-argument-type]

    asyncio.run(run())


def test_a_subscription_to_a_ring_stream_hands_over_its_ring_to_read_live(socket: str) -> None:
    async def run() -> tuple[list[RingLease], StreamLease | None, object]:
        async with await Client.connect(socket) as c:
            registered = await c.register_stream(
                "mic.audio",
                kind="sensor",
                format="RawBytes",
                shape_name="PCM_F32LEx1",
                nominal_rate_hz=50.0,
                ring_stride_bytes=16 + 320 * 4,
                ring_seconds=2.0,
            )
            sub = await c.subscribe(["mic.audio"])
            rings = sub.rings
            reader = c.ring_reader(rings[0]["lease"])
            empty = reader.read()
            stale: StreamLease = {**rings[0]["lease"], "arena_generation": rings[0]["lease"]["arena_generation"] + 1}
            with pytest.raises(ValueError, match="generation"):
                c.ring_reader(stale)
            await sub.unsubscribe()
            return rings, registered["lease"], empty

    rings, registered, empty = asyncio.run(run())
    assert [r["stream_id"] for r in rings] == [1]
    assert rings[0]["lease"]["ring_stride"] == 16 + 320 * 4
    assert registered is not None and rings[0]["lease"]["ring_offset"] == registered["ring_offset"]
    assert empty is None, "nothing written yet"
