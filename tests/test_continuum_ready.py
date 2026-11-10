"""Startup dependency waits use temporary endpoints, never the real service manager or devices."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from liberational import continuum_ready as ready


@pytest.fixture
def sdk():
    import continuum_py

    return continuum_py


def test_waits_for_a_successful_broker_request_not_only_a_connected_socket(monkeypatch, caplog, sdk):
    events = []
    attempts = 0

    class Client:
        async def list_streams(self):
            events.append("request")
            if attempts == 2:
                raise sdk.ConnectionClosedError("broker still starting")
            return []

        async def close(self, reason):
            events.append("close")

    async def connect(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        events.append("connect")
        if attempts == 1:
            raise FileNotFoundError("socket not created")
        return Client()

    monkeypatch.setattr(sdk, "Client", SimpleNamespace(connect=connect))
    asyncio.run(ready.wait_for_broker(Path("/fixture/broker.sock"), timeout=1))
    assert events == ["connect", "connect", "request", "close", "connect", "request", "close"]
    assert "waiting for broker readiness" in caplog.text
    assert "broker is ready" in caplog.text


@pytest.mark.parametrize("case", ["permission", "configuration", "schema", "request"])
def test_permanent_failures_are_not_retried(monkeypatch, case, sdk):
    error = {
        "permission": PermissionError("denied"),
        "configuration": ValueError("bad configuration"),
        "schema": sdk.ContinuumError("schema mismatch"),
        "request": sdk.BrokerError("BadRequest", "invalid protocol"),
    }[case]
    attempts = 0

    async def connect(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        raise error

    monkeypatch.setattr(sdk, "Client", SimpleNamespace(connect=connect))
    with pytest.raises(type(error)):
        asyncio.run(ready.wait_for_broker(Path("/fixture/broker.sock"), timeout=1))
    assert attempts == 1


def test_stalled_handshake_has_an_overall_deadline(monkeypatch, sdk):
    async def connect(*args, **kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(sdk, "Client", SimpleNamespace(connect=connect))
    with pytest.raises(TimeoutError, match="readiness timed out"):
        asyncio.run(ready.wait_for_broker(Path("/fixture/broker.sock"), timeout=0.03))


def test_dependency_failure_never_execs_the_service(monkeypatch, caplog):
    async def unavailable(*args):
        raise TimeoutError("broker did not become ready")

    executed = []
    monkeypatch.setattr(ready, "wait_for_broker", unavailable)
    monkeypatch.setattr(ready.os, "execv", lambda *args: executed.append(args))
    assert ready.main(["--socket", "/fixture/broker.sock", "--", "/fixture/daemon", "--flag"]) == 1
    assert executed == []
    assert "service was not started" in caplog.text


def test_exec_preserves_the_daemon_arguments_after_readiness(monkeypatch):
    events = []

    async def available(socket):
        events.append(("ready", socket))

    monkeypatch.setattr(ready, "wait_for_broker", available)
    monkeypatch.setattr(ready.os, "execv", lambda *args: events.append(("exec", args)))
    assert ready.main(["--socket", "/fixture/broker.sock", "--", "/fixture/daemon", "--flag", "two words"]) == 0
    assert events == [
        ("ready", Path("/fixture/broker.sock")),
        ("exec", ("/fixture/daemon", ["/fixture/daemon", "--flag", "two words"])),
    ]


def test_temporary_broker_answers_readiness(broker):
    asyncio.run(ready.wait_for_broker(broker, timeout=2))
    identity = asyncio.run(ready.broker_identity(broker))
    assert identity and all(generation > 0 for _, generation in identity)
    assert asyncio.run(ready.wait_for_broker_restart(broker, None, timeout=2)) == identity
    with pytest.raises(TimeoutError, match="broker instance has not changed"):
        asyncio.run(ready.wait_for_broker_restart(broker, identity, timeout=0.03))


class EmptySubscription:
    def __init__(self, events=(), rings=()):
        self.events = list(events)
        self.rings = list(rings)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.events:
            event = self.events.pop(0)
            if isinstance(event, BaseException):
                raise event
            return event
        await asyncio.Event().wait()

    async def unsubscribe(self):
        self.closed = True


class Slot(bytearray):
    released = False

    def release(self):
        self.released = True
        return True


class ReadyClient:
    voice_turn = "Idle"
    arena_names = ("/fixture-arena",)

    def __init__(self, generation=1, rows=()):
        self.generation = generation
        self.rows = list(rows)
        self.closed = False
        self.subscriptions = []
        self.slots = []
        self.events = {}
        self.rings = {}
        self.records = []
        self.reader_options = []

    def arena(self, name):
        def lease(*args, **kwargs):
            slot = Slot(32)
            self.slots.append(slot)
            return slot

        return SimpleNamespace(generation=self.generation, lease=lease)

    async def subscribe(self, streams, *, start, pinned):
        assert start == "Live" and pinned
        sub = EmptySubscription(self.events.get(streams[0], ()), self.rings.get(streams[0], ()))
        self.subscriptions.append(sub)
        return sub

    def ring_reader(self, lease, *, from_oldest):
        self.reader_options.append(from_oldest)
        records = self.records

        class Reader:
            stride = 24

            async def next_record(self, timeout_ms):
                if records:
                    return records.pop(0)
                await asyncio.Event().wait()

        return Reader()

    async def list_streams(self):
        return self.rows

    async def close(self):
        self.closed = True


def test_existing_broker_instance_cannot_satisfy_restart(monkeypatch, sdk):
    clients = []

    async def connect(*args, **kwargs):
        client = ReadyClient()
        clients.append(client)
        return client

    monkeypatch.setattr(sdk, "Client", SimpleNamespace(connect=connect))
    with pytest.raises(TimeoutError, match="broker"):
        asyncio.run(
            ready.wait_for_broker_restart(
                Path("/fixture/broker.sock"), frozenset({("/fixture-arena", 1)}), timeout=0.03
            )
        )
    assert all(client.closed for client in clients)


def test_registered_live_stream_without_fresh_frames_is_not_capture_ready(monkeypatch, sdk):
    async def connect(*args, **kwargs):
        return ReadyClient(rows=[{"name": "cam0.video", "stream_id": 7, "live": True}])

    monkeypatch.setattr(sdk, "Client", SimpleNamespace(connect=connect))
    with pytest.raises(TimeoutError, match="cam0.video"):
        asyncio.run(
            ready.wait_for_capture(
                Path("/fixture/broker.sock"), [ready.CaptureNeed("cam0.video", "video")], timeout=0.03
            )
        )


def install_client(monkeypatch, sdk, client):
    async def connect(*args, **kwargs):
        return client

    monkeypatch.setattr(sdk, "Client", SimpleNamespace(connect=connect))
    monkeypatch.setattr(ready.time, "monotonic_ns", lambda: 1000)


def live_event(sdk, kind="video", captured=1001, generation=1):
    from continuum_py import PayloadHeader

    header: PayloadHeader = {
        "id": "00000000-0000-7000-8000-000000000001",
        "stream_id": 7,
        "start_ns": captured,
        "end_ns": captured,
        "layer": "Layer1Raw",
        "format": "RawBytes",
        "shape_name": "UYVY" if kind == "video" else "PCM_S16LEx1",
        "capture": {
            "capture_mono_ns": captured,
            "device_ts": 0,
            "arrival_ns": captured,
            "stride": 4,
            "width": 2,
            "height": 2,
            "flip": False,
            "device_seq": 5,
            "sample_rate_hz": 16000,
        },
    }
    locator = {
        "kind": "Shm" if kind == "video" else "Ring",
        "arena": "/fixture-arena",
        "arena_generation": generation,
        "slab_class": 0,
        "slot": 0,
        "slot_generation": 1,
        "offset": 0,
        "length": 8,
        "ring": 0,
        "first_seq": 5,
        "count": 1,
    }
    return {"type": "Event", "item": {"header": sdk.encode_header(header), "locator": locator}}


def audio_client(sdk, *, record=True, batch=True):
    client = ReadyClient(rows=[{"name": "mic0.audio", "stream_id": 7, "live": True}])
    client.rings["mic0.audio"] = [
        {
            "stream_id": 7,
            "lease": {
                "lease_id": 1,
                "arena": "/fixture-arena",
                "arena_generation": 1,
                "first_slot": 0,
                "slot_count": 1,
                "slot_bytes": 32,
                "ring_offset": 0,
                "ring_records": 8,
                "ring_index": 0,
                "ring_data_offset": 0,
                "ring_stride": 24,
                "ring_generation": 1,
            },
        }
    ]
    if batch:
        client.events["mic0.audio"] = [live_event(sdk, "audio")]
    if record:
        client.records = [(5, (1001).to_bytes(8, "little") + bytes(16), 0)]
    return client


def test_new_broker_generation_is_ready_and_connection_is_closed(monkeypatch, sdk):
    client = ReadyClient(generation=2)
    install_client(monkeypatch, sdk, client)
    got = asyncio.run(ready.wait_for_broker_restart(Path("fixture.sock"), frozenset({("/fixture-arena", 1)})))
    assert got == frozenset({("/fixture-arena", 2)})
    assert client.closed


def test_unknown_voice_state_is_a_permanent_protocol_failure(monkeypatch, sdk):
    client = ReadyClient()
    client.voice_turn = "FutureState"
    install_client(monkeypatch, sdk, client)
    with pytest.raises(ValueError, match="voice state"):
        asyncio.run(ready.broker_identity(Path("fixture.sock")))
    assert client.closed


def test_video_requires_fresh_capture_and_current_valid_slot(monkeypatch, sdk):
    client = ReadyClient(rows=[{"name": "cam0.video", "stream_id": 7, "live": True}])
    client.events["cam0.video"] = [live_event(sdk, captured=999), live_event(sdk, generation=2), live_event(sdk)]
    install_client(monkeypatch, sdk, client)
    got = asyncio.run(ready.wait_for_capture(Path("fixture.sock"), [ready.CaptureNeed("cam0.video", "video")]))
    assert got == (ready.CaptureEvidence("cam0.video", frozenset({("/fixture-arena", 1)}), 1001, 5),)
    assert len(client.slots) == 1 and client.slots[0].released
    assert client.closed and all(sub.closed for sub in client.subscriptions)


@pytest.mark.parametrize("record,batch", [(False, True), (True, False), (False, False)])
def test_pcm_needs_new_record_and_live_batch_not_only_old_ring(monkeypatch, sdk, record, batch):
    client = audio_client(sdk, record=record, batch=batch)
    install_client(monkeypatch, sdk, client)
    with pytest.raises(TimeoutError, match="mic0.audio"):
        asyncio.run(
            ready.wait_for_capture(Path("fixture.sock"), [ready.CaptureNeed("mic0.audio", "audio")], timeout=0.03)
        )
    assert client.reader_options and not any(client.reader_options)
    assert client.closed and all(sub.closed for sub in client.subscriptions)


def test_new_pcm_record_plus_live_batch_proves_audio(monkeypatch, sdk):
    client = audio_client(sdk)
    install_client(monkeypatch, sdk, client)
    got = asyncio.run(ready.wait_for_capture(Path("fixture.sock"), [ready.CaptureNeed("mic0.audio", "audio")]))
    assert got[0].capture_mono_ns == 1001 and got[0].sequence == 5
    assert client.reader_options == [False]
    assert client.closed and client.subscriptions[0].closed


def test_every_configured_camera_and_microphone_is_required(monkeypatch, sdk):
    client = ReadyClient(rows=[{"name": "cam0.video", "stream_id": 7, "live": True}])
    install_client(monkeypatch, sdk, client)
    with pytest.raises(TimeoutError, match="mic0.audio"):
        asyncio.run(
            ready.wait_for_capture(
                Path("fixture.sock"),
                [ready.CaptureNeed("cam0.video", "video"), ready.CaptureNeed("mic0.audio", "audio")],
                timeout=0.03,
            )
        )


def test_manifest_needs_all_capture_kinds_and_reports_unsupported_teensy(tmp_path, caplog):
    manifest = tmp_path / "devices.toml"
    manifest.write_text(
        "\n".join(
            f'[[device]]\nname="{name}"\nkind="{kind}"'
            for name, kind in [
                ("cam0", "camera"),
                ("patio", "rtsp"),
                ("mic0", "microphone"),
                ("array0", "xvf3800"),
                ("sensor0", "teensy"),
            ]
        )
    )
    assert ready.capture_needs(manifest) == (
        ready.CaptureNeed("cam0.video", "video"),
        ready.CaptureNeed("patio.video", "video"),
        ready.CaptureNeed("mic0.audio", "audio"),
        ready.CaptureNeed("array0.audio", "audio"),
    )
    assert "teensy readiness is unsupported" in caplog.text


@pytest.mark.parametrize(
    "body",
    [
        "device=1",
        '[[device]]\nname="cam.0"\nkind="camera"',
        '[[device]]\nname="cam0"\nkind="unknown"',
        '[[device]]\nname="cam0"\nkind=1',
        '[[device]]\nname="cam0"\nkind="camera"\n[[device]]\nname="cam0"\nkind="microphone"',
    ],
)
def test_invalid_manifest_is_not_an_empty_capture_gate(tmp_path, body):
    manifest = tmp_path / "devices.toml"
    manifest.write_text(body)
    with pytest.raises(ValueError):
        ready.capture_needs(manifest)


def test_zero_needs_skips_only_capture_without_connecting(monkeypatch, sdk):
    async def connect(*args, **kwargs):
        pytest.fail("empty capture gate must not connect")

    monkeypatch.setattr(sdk, "Client", SimpleNamespace(connect=connect))
    assert asyncio.run(ready.wait_for_capture(Path("fixture.sock"), [])) == ()


def test_reconnect_discards_previously_proven_streams(monkeypatch, sdk):
    first = audio_client(sdk)
    second = audio_client(sdk)
    for client in (first, second):
        client.rows.append({"name": "cam0.video", "stream_id": 8, "live": True})
    # Camera succeeds in the first connection; audio fails. The second connection
    # proves audio, but must obtain camera evidence again rather than reuse it.
    camera = live_event(sdk)
    header = sdk.decode_header(camera["item"]["header"])
    header["stream_id"] = 8
    camera["item"]["header"] = sdk.encode_header(header)
    first.events["cam0.video"] = [camera]
    first.events["mic0.audio"] = [ConnectionError("disconnected")]
    clients = iter((first, second))

    async def connect(*args, **kwargs):
        return next(clients, second)

    monkeypatch.setattr(sdk, "Client", SimpleNamespace(connect=connect))
    monkeypatch.setattr(ready.time, "monotonic_ns", lambda: 1000)
    with pytest.raises(TimeoutError, match="cam0.video"):
        asyncio.run(
            ready.wait_for_capture(
                Path("fixture.sock"),
                [ready.CaptureNeed("cam0.video", "video"), ready.CaptureNeed("mic0.audio", "audio")],
                timeout=0.16,
            )
        )
    assert first.slots and first.closed and second.closed
    assert all(sub.closed for client in (first, second) for sub in client.subscriptions)


def test_hanging_client_cleanup_is_bounded_and_does_not_report_success(monkeypatch, sdk):
    client = ReadyClient(rows=[{"name": "cam0.video", "stream_id": 7, "live": True}])
    client.events["cam0.video"] = [live_event(sdk)]
    cancelled = []

    async def close():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    monkeypatch.setattr(client, "close", close)
    install_client(monkeypatch, sdk, client)

    async def probe():
        async with asyncio.timeout(0.15):
            with pytest.raises(TimeoutError, match="cleanup.*cam0.video"):
                await ready.wait_for_capture(
                    Path("fixture.sock"), [ready.CaptureNeed("cam0.video", "video")], timeout=0.03
                )

    asyncio.run(probe())
    assert cancelled == [True]
    assert client.subscriptions[0].closed and client.slots[0].released


def test_capture_permanent_subscription_error_is_not_retried(monkeypatch, sdk):
    client = ReadyClient(rows=[{"name": "cam0.video", "stream_id": 7, "live": True}])
    calls = []

    async def subscribe(*args, **kwargs):
        calls.append(True)
        raise sdk.BrokerError("BadRequest", "invalid stream selector")

    monkeypatch.setattr(client, "subscribe", subscribe)
    install_client(monkeypatch, sdk, client)
    with pytest.raises(sdk.BrokerError, match="BadRequest"):
        asyncio.run(ready.wait_for_capture(Path("fixture.sock"), [ready.CaptureNeed("cam0.video", "video")]))
    assert calls == [True] and client.closed


def test_undecodable_live_capture_is_permanent_not_silent_timeout(monkeypatch, sdk):
    client = ReadyClient(rows=[{"name": "cam0.video", "stream_id": 7, "live": True}])
    client.events["cam0.video"] = [{"type": "Undecodable", "reason": "invalid capture header"}]
    install_client(monkeypatch, sdk, client)
    with pytest.raises(ValueError, match="cam0.video.*undecodable"):
        asyncio.run(ready.wait_for_capture(Path("fixture.sock"), [ready.CaptureNeed("cam0.video", "video")]))
    assert client.closed and client.subscriptions[0].closed


def test_pcm_batch_must_have_a_live_sample_rate(monkeypatch, sdk):
    client = audio_client(sdk)
    event = client.events["mic0.audio"][0]
    header = sdk.decode_header(event["item"]["header"])
    header["capture"]["sample_rate_hz"] = 0
    event["item"]["header"] = sdk.encode_header(header)
    install_client(monkeypatch, sdk, client)
    with pytest.raises(ValueError, match="sample rate"):
        asyncio.run(ready.wait_for_capture(Path("fixture.sock"), [ready.CaptureNeed("mic0.audio", "audio")]))
    assert client.closed and client.subscriptions[0].closed


def test_healthy_audio_batch_can_finish_after_the_connection_attempt_budget(monkeypatch, sdk):
    # Scale the reported 2-second attempt / 3-second batch to milliseconds. PCM
    # arrives promptly, while its independently fresh sealed batch arrives later.
    monkeypatch.setattr(ready, "ATTEMPT_S", 0.02)
    client = audio_client(sdk)
    install_client(monkeypatch, sdk, client)
    connections = []

    async def connect(*args, **kwargs):
        connections.append(True)
        return client

    class DelayedBatch(EmptySubscription):
        async def __anext__(self):
            await asyncio.sleep(0.03)
            return await super().__anext__()

    async def subscribe(streams, *, start, pinned):
        assert start == "Live" and pinned
        sub = DelayedBatch(client.events[streams[0]], client.rings[streams[0]])
        client.subscriptions.append(sub)
        return sub

    monkeypatch.setattr(sdk, "Client", SimpleNamespace(connect=connect))
    monkeypatch.setattr(client, "subscribe", subscribe)
    result = asyncio.run(
        ready.wait_for_capture(Path("fixture.sock"), [ready.CaptureNeed("mic0.audio", "audio")], timeout=0.15)
    )
    assert result == (ready.CaptureEvidence("mic0.audio", frozenset({("/fixture-arena", 1)}), 1001, 5),)
    assert connections == [True], "slow healthy media must not reset the connection/barrier"
    assert client.closed and all(sub.closed for sub in client.subscriptions)


def test_capture_catalog_setup_keeps_short_budget_while_media_waits_longer(monkeypatch, sdk):
    monkeypatch.setattr(ready, "ATTEMPT_S", 0.02)
    slow = ReadyClient()
    healthy = ReadyClient(rows=[{"name": "cam0.video", "stream_id": 7, "live": True}])
    healthy.events["cam0.video"] = [live_event(sdk)]
    install_client(monkeypatch, sdk, healthy)
    cancelled = []
    connections = []

    async def stalled_catalog():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    async def connect(*args, **kwargs):
        client = slow if not connections else healthy
        connections.append(client)
        return client

    monkeypatch.setattr(slow, "list_streams", stalled_catalog)
    monkeypatch.setattr(sdk, "Client", SimpleNamespace(connect=connect))
    result = asyncio.run(
        ready.wait_for_capture(Path("fixture.sock"), [ready.CaptureNeed("cam0.video", "video")], timeout=0.5)
    )
    assert result[0].stream == "cam0.video"
    assert connections == [slow, healthy] and cancelled == [True]
    assert slow.closed and healthy.closed


@pytest.mark.parametrize("kind", ["camera", "rtsp", "microphone", "xvf3800"])
def test_disabled_manifest_device_does_not_block_capture_restart(tmp_path, kind):
    manifest = tmp_path / "devices.toml"
    manifest.write_text(f'[[device]]\nname="retained"\nkind="{kind}"\nenabled=false\n')
    assert ready.capture_needs(manifest) == ()


def test_manifest_enabled_requires_boolean(tmp_path):
    manifest = tmp_path / "devices.toml"
    manifest.write_text('[[device]]\nname="cam0"\nkind="camera"\nenabled="false"\n')
    with pytest.raises(ValueError, match="enabled"):
        ready.capture_needs(manifest)


def test_unregistered_names_the_needs_with_no_live_stream_in_one_look(monkeypatch, sdk):
    clients = []

    async def connect(*args, **kwargs):
        rows = [
            {"name": "cam0.video", "stream_id": 7, "live": True},
            {"name": "old.audio", "stream_id": 8, "live": False},
        ]
        clients.append(ReadyClient(rows=rows))
        return clients[-1]

    monkeypatch.setattr(sdk, "Client", SimpleNamespace(connect=connect))
    needs = (
        ready.CaptureNeed("cam0.video", "video"),
        ready.CaptureNeed("old.audio", "audio"),
        ready.CaptureNeed("desk.audio", "audio"),
    )
    gone = asyncio.run(ready.unregistered(Path("/fixture/broker.sock"), needs))
    assert gone == needs[1:]
    assert len(clients) == 1 and clients[0].closed
    assert asyncio.run(ready.unregistered(Path("/fixture/broker.sock"), ())) == ()
    assert len(clients) == 1  # nothing declared: the broker isn't asked


def test_unregistered_raises_when_the_broker_cannot_be_asked(monkeypatch, sdk):
    async def connect(*args, **kwargs):
        raise ConnectionRefusedError("no broker")

    monkeypatch.setattr(sdk, "Client", SimpleNamespace(connect=connect))
    with pytest.raises(ConnectionRefusedError):
        asyncio.run(ready.unregistered(Path("/fixture/broker.sock"), (ready.CaptureNeed("desk.audio", "audio"),)))
