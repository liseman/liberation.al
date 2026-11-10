"""The worker side from Python against a real broker: register_worker, quote, a WorkLease's renew, keep_alive, and
release, a voice turn preempting it (decode_event), and stage_id.

Run with `mise run continuum-test`, which builds the broker first.
"""

import asyncio
import os
import time
import uuid

import pytest
from continuum_py import (
    BrokerError,
    Capability,
    Client,
    ComputeDomain,
    WorkLease,
    decode_event,
    encode_header,
    stage_id,
)


async def worker(socket: str, domain: ComputeDomain) -> tuple[Client, int]:
    """A worker with "cam0" (sensor) and "pose" (derived from cam0) registered; returns it and the capability id."""
    client = await Client.connect(socket, client_name="py-worker")
    await client.register_stream("cam0", kind="sensor", format="RawBytes", shape_name="NV12", slab_class=0)
    await client.register_stream(
        "pose", kind="derived", format="FlatBuffer", shape_name="pose", layer="Layer2Decision", slab_class=0
    )
    capability: Capability = {
        "produces": "pose",
        "compute_domain": domain,
        "requires": [{"stream": "cam0"}],
        "max_batch": 8,
    }
    registered = await client.register_worker("worker:pose", [capability], version_hash="test")
    assert registered["refused_reasons"] == []
    (accepted,) = registered["accepted"]
    assert accepted["index"] == 0 and accepted["runnable"]
    return client, accepted["id"]


def test_a_capability_on_an_unknown_stream_is_refused_with_a_reason(socket: str) -> None:
    async def run() -> list[str]:
        async with await Client.connect(socket) as client:
            r = await client.register_worker("worker:x", [{"produces": "nothing-here", "compute_domain": "CPU"}])
            assert r["accepted"] == []
            return r["refused_reasons"]

    (reason,) = asyncio.run(run())
    assert "nothing-here" in reason


def test_a_lease_renews_and_keeps_itself_alive(socket: str) -> None:
    async def run() -> tuple[int, int, int]:
        client, cap = await worker(socket, "CPU")
        lease = await client.quote(cap, stage="pose", range=(0, 1_000))
        first = lease.expires_at_ns
        assert first is not None
        await asyncio.sleep(0.02)
        renewed = await lease.renew()
        lease.keep_alive(0.05)
        await asyncio.sleep(0.3)
        later = lease.expires_at_ns
        assert later is not None
        lease.release()
        assert lease.expires_at_ns is None, "released"
        await client.close()
        return first, renewed, later

    first, renewed, later = asyncio.run(run())
    assert renewed > first
    assert later > renewed, "the keep-alive renewed it again"


def test_a_voice_turn_preempts_a_background_gpu_lease(socket: str) -> None:
    async def run() -> tuple[int, dict[str, object], dict[str, object], dict[str, object]]:
        worker_client, cap = await worker(socket, "GPU")
        lease = await worker_client.quote(cap, stage="pose", range=(0, 1_000), compute_domain="GPU")
        voice = await Client.connect(socket, client_name="voice")
        voice.set_system_state("Listening")
        first = await asyncio.wait_for(worker_client.next_event(), timeout=5)
        second = await asyncio.wait_for(worker_client.next_event(), timeout=5)
        heard = await asyncio.wait_for(voice.next_event(), timeout=5)
        assert first is not None and second is not None and heard is not None
        lease.release()
        return lease.id, dict(decode_event(first[1])), dict(decode_event(second[1])), dict(decode_event(heard[1]))

    lease_id, first, second, heard = asyncio.run(run())
    assert first == {
        "type": "LeasePreempted",
        "lease_id": lease_id,
        "reason": first["reason"],
        "by_band": "P0Critical",
        "revoked": False,
    }
    assert second["type"] == "SystemStateChanged" and second["voice_turn"] == "Listening"
    assert heard["type"] == "SystemStateChanged", "every session hears the new state"


def test_leaving_async_with_releases_an_uncommitted_lease(socket: str) -> None:
    async def run() -> WorkLease:
        client, cap = await worker(socket, "CPU")
        async with await client.quote(cap, stage="pose", range=(0, 1_000)) as lease:
            assert lease.expires_at_ns is not None
        return lease

    lease = asyncio.run(run())
    assert lease.expires_at_ns is None, "the block gave it back"
    with pytest.raises(ValueError, match="already ended"):
        lease.keep_alive(1.0)


@pytest.mark.parametrize(
    ("call", "match"),
    [
        pytest.param(lambda c: c.quote(0, stage="s", range=(0, 1), band="P9"), "band must be one of", id="band"),
        pytest.param(
            lambda c: c.quote(0, stage="s", range=(0, 1), compute_domain="TPU"), "compute_domain", id="domain"
        ),
        pytest.param(lambda c: c.register_worker("w", [{"produces": "x"}]), "compute_domain", id="missing-key"),
    ],
)
def test_bad_arguments_are_refused_before_sending(socket: str, call: object, match: str) -> None:
    async def run() -> None:
        async with await Client.connect(socket) as client:
            with pytest.raises((ValueError, KeyError), match=match):
                await call(client)  # ty: ignore[call-non-callable]
            with pytest.raises(ValueError, match="voice_turn"):
                client.set_system_state("Shouting")  # ty: ignore[invalid-argument-type]

    asyncio.run(run())


def test_stage_ids_are_fnv_1a() -> None:
    assert stage_id("") == 0x811C9DC5
    assert stage_id("a") == 0xE40C292C
    assert stage_id("foobar") == 0xBF9CF968


def v7() -> str:
    raw = bytearray(os.urandom(16))
    raw[:6] = (time.time_ns() // 1_000_000).to_bytes(6, "big")
    raw[6] = 0x70 | raw[6] & 0x0F
    raw[8] = 0x80 | raw[8] & 0x3F
    return str(uuid.UUID(bytes=bytes(raw)))


def output_on(stream_id: int) -> bytes:
    return encode_header(
        {
            "id": v7(),
            "stream_id": stream_id,
            "start_ns": 100,
            "end_ns": 200,
            "layer": "Layer2Decision",
            "format": "FlatBuffer",
            "shape_name": "pose",
        }
    )


def test_a_commit_and_an_abort_return_once_the_broker_has_them(socket: str) -> None:
    async def run() -> None:
        client, cap = await worker(socket, "CPU")
        pose = (
            await client.register_stream(
                "pose", kind="derived", format="FlatBuffer", shape_name="pose", layer="Layer2Decision", slab_class=0
            )
        )["stream_id"]
        lease = await client.quote(cap, stage="pose", range=(0, 1_000), band="P2Nearline")
        await lease.commit(
            [(output_on(pose), None)],
            cost={
                "wall_time_us": 900,
                "queue_wait_us": 0,
                "ram_bytes": 0,
                "vram_bytes": 0,
                "device": "CPU",
                "items": 1,
            },
        )
        assert lease.expires_at_ns is None, "committed, so ended"
        with pytest.raises(ValueError, match="already ended"):
            await lease.commit([])
        lease = await client.quote(cap, stage="pose", range=(0, 1_000), band="P2Nearline")
        await lease.abort("out of memory")
        await client.close()

    asyncio.run(run())


def test_a_commit_onto_another_stream_raises_broker_error(socket: str) -> None:
    async def run() -> BrokerError:
        client, cap = await worker(socket, "CPU")
        cam = (await client.register_stream("cam0", kind="sensor", format="RawBytes", shape_name="NV12", slab_class=0))[
            "stream_id"
        ]
        lease = await client.quote(cap, stage="pose", range=(0, 1_000))
        with pytest.raises(BrokerError) as refused:
            await lease.commit([(output_on(cam), None)])
        assert lease.expires_at_ns is None, "a refused commit ends (and releases) the lease"
        with pytest.raises(ValueError, match="already ended"):
            await lease.commit([])
        await client.close()
        return refused.value

    assert asyncio.run(run()).args[0] == "NotAllowed"


def test_an_exception_in_async_with_aborts_the_lease(socket: str) -> None:
    async def run() -> WorkLease:
        client, cap = await worker(socket, "CPU")
        with pytest.raises(RuntimeError, match="the model failed"):
            async with await client.quote(cap, stage="pose", range=(0, 1_000)) as lease:
                raise RuntimeError("the model failed")
        await client.close()
        return lease

    assert asyncio.run(run()).expires_at_ns is None, "the block aborted it"


def test_release_during_a_renewal_isnt_refused_as_busy(socket: str) -> None:
    async def run() -> None:
        client, cap = await worker(socket, "CPU")
        lease = await client.quote(cap, stage="pose", range=(0, 1_000))
        renewal = asyncio.ensure_future(lease.renew())
        await asyncio.sleep(0)
        assert lease.expires_at_ns is not None, "still live while a renewal is in flight"
        lease.release()
        assert lease.expires_at_ns is None
        await asyncio.gather(renewal, return_exceptions=True)
        with pytest.raises(ValueError, match="already ended"):
            await lease.renew()
        await client.close()

    asyncio.run(run())


def test_cancelling_inside_async_with_ends_the_lease(socket: str) -> None:
    async def run() -> WorkLease:
        client, cap = await worker(socket, "CPU")
        lease = await client.quote(cap, stage="pose", range=(0, 1_000))

        async def job() -> None:
            async with lease:
                await asyncio.sleep(60)

        task = asyncio.ensure_future(job())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await client.close()
        return lease

    assert asyncio.run(run()).expires_at_ns is None
