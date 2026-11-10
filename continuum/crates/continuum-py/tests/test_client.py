"""continuum_py.Client against a real broker process: the session, heartbeats, errors, and closing.

Run with `mise run continuum-test`, which builds the broker first. The broker binary comes from CONTINUUM_BROKER, or
continuum/target/debug/continuum-broker.
"""

import asyncio
import gc
import os
import socket as py_socket
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from continuum_py import BrokerError, Client, ConnectionClosedError, decode_event, decode_ingest_rejected, encode_header
from continuum_py.schema import VoiceTurnState

from conftest import spawn_broker, stop_broker


@pytest.fixture
def socket(tmp_path: Path) -> Iterator[str]:
    path = tmp_path / "b.sock"
    broker = spawn_broker(path, "--heartbeat-timeout-s", "30")
    try:
        yield str(path)
    finally:
        stop_broker(broker, path)


@pytest.fixture
def impatient_socket(tmp_path: Path) -> Iterator[str]:
    """A broker that closes a session after one quiet second."""
    path = tmp_path / "b.sock"
    broker = spawn_broker(path, "--heartbeat-timeout-s", "1")
    try:
        yield str(path)
    finally:
        stop_broker(broker, path)


def test_a_client_gets_a_session(socket: str) -> None:
    async def run() -> tuple[str, tuple[int, int], list[str]]:
        async with await Client.connect(socket, client_name="pytest") as client:
            return client.session_id, client.broker_schema, client.arena_names

    session_id, schema, arenas = asyncio.run(run())
    assert uuid.UUID(session_id).version == 7
    assert schema[0] == 0
    assert isinstance(arenas, list)


@pytest.mark.parametrize("state", ["Listening", "Thinking", "Speaking", "Idle"])
def test_voice_state_survives_mid_turn_connect_and_updates_without_event_reads(
    socket: str, state: VoiceTurnState
) -> None:
    async def run() -> None:
        async with await Client.connect(socket, client_name="voice") as voice:
            voice.set_system_state("Thinking")
            async with asyncio.timeout(5):
                while voice.voice_turn != "Thinking":
                    await asyncio.sleep(0)
            async with await Client.connect(socket, client_name="background") as background:
                assert background.voice_turn == "Thinking"
                voice.set_system_state(state)
                async with asyncio.timeout(5):
                    while background.voice_turn != state:
                        await asyncio.sleep(0)
                event = await asyncio.wait_for(background.next_event(), timeout=5)
                assert event is not None
                assert decode_event(event[1])["voice_turn"] == state

    asyncio.run(run())


def test_a_heartbeat_returns_the_brokers_utc_clock(socket: str) -> None:
    async def run() -> tuple[int, int, int]:
        async with await Client.connect(socket) as client:
            before = time.time_ns()
            broker_now = await client.heartbeat(queue_depth=2, models_resident=["pose"], ram_bytes=1 << 20)
            return before, broker_now, time.time_ns()

    before, broker_now, after = asyncio.run(run())
    assert before <= broker_now <= after


def test_background_heartbeats_keep_a_quiet_session_open(impatient_socket: str) -> None:
    async def run() -> int:
        client = await Client.connect(impatient_socket, heartbeat_interval_s=0.2)
        await asyncio.sleep(2.5)  # past two broker timeouts with nothing from the caller
        return await client.heartbeat()

    assert asyncio.run(run()) > 0


def test_registering_a_stream_returns_its_id_and_the_same_id_again(socket: str) -> None:
    async def run() -> tuple[int, int, int]:
        async with await Client.connect(socket) as client:
            first = await client.register_stream(
                "cam0.video", kind="sensor", format="RawBytes", shape_name="NV12", slab_class=2
            )
            other = await client.register_stream(
                "cam1.video", kind="sensor", format="RawBytes", shape_name="NV12", slab_class=2
            )
            again = await client.register_stream(
                "cam0.video", kind="sensor", format="RawBytes", shape_name="NV12", slab_class=2
            )
            return first["stream_id"], other["stream_id"], again["stream_id"]

    first, other, again = asyncio.run(run())
    assert first != other
    assert again == first


def test_a_refused_request_raises_broker_error_and_the_session_survives(socket: str) -> None:
    async def run() -> tuple[BrokerError, int]:
        async with await Client.connect(socket) as client:
            await client.register_stream(
                "cam0.video", kind="sensor", format="RawBytes", shape_name="NV12", slab_class=2
            )
            with pytest.raises(BrokerError) as refused:
                # The same name with a different shape is a different stream.
                await client.register_stream(
                    "cam0.video", kind="sensor", format="RawBytes", shape_name="MJPEG", slab_class=2
                )
            return refused.value, await client.heartbeat()

    refused, now = asyncio.run(run())
    assert refused.args[0] == "Conflict"
    assert now > 0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param({"slab_class": 1, "ring_stride_bytes": 8}, "exactly one", id="slab-and-ring"),
        pytest.param({}, "exactly one", id="neither"),
        pytest.param({"slab_class": 1, "kind": "camera"}, "'sensor' or 'derived'", id="bad-kind"),
        pytest.param({"slab_class": 1, "format": "jpeg"}, "format must be one of", id="bad-format"),
        pytest.param({"slab_class": 1, "anchor": ("nope", 1, 2)}, "canonical UUID", id="bad-anchor"),
        pytest.param({"slab_class": 1, "layer": "Layer9"}, "layer must be one of", id="bad-layer"),
        pytest.param({"slab_class": 70000}, "at most 65535", id="slab-class-past-u16"),
    ],
)
def test_a_bad_stream_spec_is_refused_before_it_is_sent(socket: str, kwargs: dict[str, object], match: str) -> None:
    async def run() -> None:
        async with await Client.connect(socket) as client:
            args: dict = {"kind": "derived", "format": "RawBytes", "shape_name": "pose", **kwargs}
            with pytest.raises(ValueError, match=match):
                await client.register_stream("pose.body", **args)

    asyncio.run(run())


def test_after_close_requests_fail_at_once(socket: str) -> None:
    async def run() -> float:
        client = await Client.connect(socket)
        await client.close()
        assert client.voice_turn is None
        started = time.monotonic()
        with pytest.raises(ConnectionClosedError):
            await client.heartbeat()
        return time.monotonic() - started

    assert asyncio.run(run()) < 1.0


def test_connecting_to_no_broker_is_an_os_error(tmp_path: Path) -> None:
    async def run() -> None:
        await Client.connect(str(tmp_path / "none.sock"))

    with pytest.raises(OSError):
        asyncio.run(run())


def test_a_ring_stream_with_a_clock_anchor_registers(socket: str) -> None:
    async def run() -> int:
        async with await Client.connect(Path(socket)) as client:
            anchor = (str(uuid.uuid4()), time.monotonic_ns(), time.time_ns())
            registered = await client.register_stream(
                "mic0.pcm",
                kind="sensor",
                format="RawBytes",
                shape_name="PCM_S16LE",
                nominal_rate_hz=50.0,  # 20 ms records; the broker sizes the ring from it
                ring_stride_bytes=1920,
                ring_seconds=10.0,
                anchor=anchor,
            )
            return registered["stream_id"]

    assert asyncio.run(run()) > 0


def test_an_empty_stream_name_is_a_bad_request(socket: str) -> None:
    async def run() -> None:
        async with await Client.connect(socket) as client:
            with pytest.raises(BrokerError) as refused:
                await client.register_stream("", kind="derived", format="Text", shape_name="note", slab_class=0)
            assert refused.value.args[0] == "BadRequest"

    asyncio.run(run())


@pytest.mark.parametrize("seconds", [0.0, 0.001, 1e-9, -1.0])
def test_a_heartbeat_interval_under_10_ms_is_refused(socket: str, seconds: float) -> None:
    async def run() -> None:
        await Client.connect(socket, heartbeat_interval_s=seconds)

    with pytest.raises(ValueError, match="at least"):
        asyncio.run(run())


def test_a_broker_that_never_answers_times_out(tmp_path: Path) -> None:
    path = tmp_path / "mute.sock"
    mute = py_socket.socket(py_socket.AF_UNIX)
    mute.bind(str(path))
    mute.listen(1)

    async def run() -> None:
        await Client.connect(str(path), request_timeout_s=0.2)

    try:
        with pytest.raises(TimeoutError):
            asyncio.run(run())
    finally:
        mute.close()


def test_when_the_broker_dies_events_end_and_requests_fail(tmp_path: Path) -> None:
    path = tmp_path / "b.sock"
    broker = spawn_broker(path, "--heartbeat-timeout-s", "30")

    async def run() -> tuple[object, type[BaseException] | None]:
        client = await Client.connect(str(path))
        stop_broker(broker, path, kill=True)
        event = await asyncio.wait_for(client.next_event(), timeout=5)
        try:
            await client.heartbeat()
        except Exception as e:
            return event, type(e)
        return event, None

    event, error = asyncio.run(run())
    assert event is None, "no more events once the broker is gone"
    assert error is ConnectionClosedError


def open_fds() -> int:
    return len(os.listdir("/dev/fd"))


def test_a_collected_client_closes_its_socket(socket: str) -> None:
    async def run() -> tuple[int, int]:
        # Warm up first: the runtime's threads and pollers start on the first connection, and stay.
        warmup = await Client.connect(socket)
        await warmup.close()
        del warmup
        gc.collect()
        await asyncio.sleep(0.2)
        before = open_fds()
        client = await Client.connect(socket)
        await client.heartbeat()
        del client
        gc.collect()
        await asyncio.sleep(0.2)  # let the runtime finish dropping the connection's tasks
        return before, open_fds()

    before, after = asyncio.run(run())
    assert after == before


def v7() -> str:
    """A fresh UUIDv7 string."""
    raw = bytearray(os.urandom(16))
    raw[:6] = (time.time_ns() // 1_000_000).to_bytes(6, "big")
    raw[6] = 0x70 | raw[6] & 0x0F
    raw[8] = 0x80 | raw[8] & 0x3F
    return str(uuid.UUID(bytes=bytes(raw)))


def encoded(stream_id: int) -> tuple[str, bytes]:
    payload_id = v7()
    now = time.time_ns()
    header = encode_header(
        {
            "id": payload_id,
            "stream_id": stream_id,
            "start_ns": now,
            "end_ns": now,
            "layer": "Layer2Decision",
            "format": "Text",
            "shape_name": "note",
        }
    )
    return payload_id, header


def test_an_accepted_ingest_draws_nothing_and_a_refused_one_comes_back_decoded(socket: str) -> None:
    async def run() -> tuple[str, dict[str, object]]:
        async with await Client.connect(socket) as client:
            registered = await client.register_stream(
                "notes", kind="derived", format="Text", shape_name="note", layer="Layer2Decision", slab_class=0
            )
            _, good = encoded(registered["stream_id"])
            client.ingest(good, inline=b"a short note")
            # A sentinel the broker refuses, after the good one: its rejection must be the first event.
            sentinel_id, sentinel = encoded(4244)
            client.ingest(sentinel)
            event = await asyncio.wait_for(client.next_event(), timeout=5)
            assert event is not None
            kind, envelope = event
            assert kind == "IngestRejected"
            # A refusal from the catalog writer would come later and asynchronously: wait a while for one.
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(client.next_event(), timeout=0.5)
            return sentinel_id, dict(decode_ingest_rejected(envelope))

    sentinel_id, rejected = asyncio.run(run())
    assert rejected["payload_id"] == sentinel_id, "the good notice drew no rejection"
    assert rejected["stream_id"] == 4244
    assert "didn't register" in str(rejected["reason"])


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        pytest.param({"inline": b"x" * 65537}, ValueError, "at most 65536", id="inline-too-big"),
        pytest.param({"inline": b"x", "shm": {}}, ValueError, "at most one", id="inline-and-shm"),
        pytest.param({"shm": {"arena": "hot"}}, KeyError, "arena_generation", id="incomplete-shm"),
    ],
)
def test_a_bad_ingest_is_refused_before_it_is_sent(
    socket: str, kwargs: dict[str, object], error: type[Exception], match: str
) -> None:
    async def run() -> None:
        async with await Client.connect(socket) as client:
            _, header = encoded(1)
            with pytest.raises(error, match=match):
                client.ingest(header, **kwargs)  # ty: ignore[invalid-argument-type]
            with pytest.raises(ValueError, match="header"):
                client.ingest(b"not a header")

    asyncio.run(run())


def test_ingest_after_close_raises(socket: str) -> None:
    async def run() -> None:
        client = await Client.connect(socket)
        await client.close()
        _, header = encoded(1)
        with pytest.raises(ConnectionClosedError):
            client.ingest(header)

    asyncio.run(run())


def test_decode_ingest_rejected_refuses_other_messages() -> None:
    with pytest.raises(ValueError):
        decode_ingest_rejected(b"not an envelope")


@pytest.mark.parametrize("policy", ["close", "backpressure"])
def test_subscription_overflow_policy_accepts_supported_modes(socket: str, policy: str) -> None:
    async def run() -> None:
        async with await Client.connect(socket, subscription_overflow=policy) as client:
            assert await client.heartbeat() > 0

    asyncio.run(run())


def test_unknown_subscription_overflow_policy_is_rejected_before_connect(tmp_path: Path) -> None:
    async def run() -> None:
        await Client.connect(str(tmp_path / "absent.sock"), subscription_overflow="discard")

    with pytest.raises(ValueError, match="subscription_overflow"):
        asyncio.run(run())
