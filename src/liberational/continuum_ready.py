"""Wait for a usable broker before execing a mandatory client service.

Service renderers opt in through services.toml's wait_for_broker. Both launchd and
systemd run this wrapper, which checks an SDK request, then replaces itself with
the daemon so signals and exit status still belong to the service manager.
Dashboard and gateway remain available while the broker is offline. Restart cascades additionally require a changed broker arena identity and fresh
video slots / PCM records for every configured camera and microphone. Live catalog
rows and pre-existing rings alone never prove capture readiness.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import time
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast, get_args

if TYPE_CHECKING:
    from continuum_py import Client, Locator, PayloadHeader, RingReader, StreamLease, Subscription

from liberational import logs

LOG = logs.get("continuum-ready")
WAIT_S = 60.0
ATTEMPT_S = 2.0


def transient(error: BaseException) -> bool:
    """Only dependency startup/transport failures merit another attempt."""
    import continuum_py

    return isinstance(error, (ConnectionError, FileNotFoundError, TimeoutError, continuum_py.NotReadyError)) or (
        isinstance(error, continuum_py.BrokerError) and bool(error.args) and error.args[0] == "Unavailable"
    )


async def wait_for_broker(socket: Path, timeout: float = WAIT_S) -> None:
    """Require a handshake and successful catalog request within one overall deadline."""
    import continuum_py

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    pause = 0.1
    waiting = False
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise TimeoutError(f"Broker readiness timed out after {timeout:g} s: {socket}")
        client = None
        try:
            async with asyncio.timeout(min(remaining, ATTEMPT_S)):
                client = await continuum_py.Client.connect(
                    str(socket), client_name="service-readiness", request_timeout_s=ATTEMPT_S
                )
                await client.list_streams()
            if waiting:
                LOG.info("broker is ready; starting the service")
            return
        except Exception as error:
            if not transient(error):
                raise
            if not waiting:
                LOG.info("waiting for broker readiness at %s: %s", socket, error)
                waiting = True
        finally:
            if client is not None:
                try:
                    async with asyncio.timeout(min(1.0, max(0.001, deadline - loop.time()))):
                        await client.close("readiness checked")
                except Exception as error:
                    if not transient(error):
                        raise
        await asyncio.sleep(min(pause, max(0.0, deadline - loop.time())))
        pause = min(pause * 2, 1.0)


BrokerIdentity = frozenset[tuple[str, int]]


@dataclass(frozen=True)
class CaptureNeed:
    """One required configured capture stream; absence is never treated as disabled."""

    stream: str
    kind: Literal["video", "audio"]


@dataclass(frozen=True)
class CaptureEvidence:
    """Fresh activity on one broker instance, without retaining media or a slot lease."""

    stream: str
    broker: BrokerIdentity
    capture_mono_ns: int
    sequence: int


class _Pending(Exception):
    """A capture dependency is registered incompletely or has not produced fresh data yet."""


async def _close(resource: Client | Subscription, deadline: float, *, subscription: bool = False) -> None:
    remaining = max(0.0, deadline - asyncio.get_running_loop().time())
    try:
        async with asyncio.timeout(min(0.2, remaining)):
            if subscription:
                await cast("Subscription", resource).unsubscribe()
            else:
                await cast("Client", resource).close()
    except TimeoutError as error:
        raise TimeoutError("readiness cleanup timed out") from error


async def _identity(client: Client) -> BrokerIdentity:
    import continuum_py

    await client.list_streams()
    voice = client.voice_turn
    if voice is None:
        raise ConnectionError("broker has no connected tracked voice state")
    if voice not in get_args(continuum_py.VoiceTurnState):
        raise ValueError(f"SDK cannot track broker voice state {voice!r}")
    identity = frozenset((name, client.arena(name).generation) for name in client.arena_names)
    if not identity:
        raise continuum_py.NotReadyError("broker has not supplied a Tier 0 arena")
    if any(not name or not isinstance(generation, int) or generation <= 0 for name, generation in identity):
        raise ValueError("broker supplied an invalid arena identity")
    return identity


async def broker_identity(socket: Path, timeout: float = ATTEMPT_S) -> BrokerIdentity:
    """One bounded compatible handshake/request probe; unavailable errors are returned to the caller."""
    import continuum_py

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    client = None
    try:
        async with asyncio.timeout(max(0.0, timeout * 0.8)):
            client = await continuum_py.Client.connect(
                str(socket), client_name="restart-readiness", request_timeout_s=min(timeout, ATTEMPT_S)
            )
            return await _identity(client)
    finally:
        if client is not None:
            await _close(client, deadline)


#: Tier 0 locators: a payload still in the broker's arena (a video slot, a PCM ring batch), not yet in a chunk.
TIER0_KINDS = frozenset({"Shm", "Ring"})
#: How far back a drain looks: past Tier 0's ring, so a frame still held there is in the window.
DRAIN_LOOKBACK_S = 30.0
DRAIN_POLL_S = 0.5


async def tier0_left(client: Client, streams: Sequence[str], now_ns: int) -> int:
    """How many payloads on `streams` from the last DRAIN_LOOKBACK_S still have only a Tier 0 copy."""
    left = 0
    for stream in streams:
        page = await client.query(stream, now_ns - int(DRAIN_LOOKBACK_S * 1e9), now_ns, max_items=0)
        left += sum(1 for item in page["items"] if (item.get("locator") or {}).get("kind") in TIER0_KINDS)
    return left


async def wait_for_tier0_drain(socket: Path, streams: Sequence[str], timeout: float) -> int:
    """Waits, after capture stopped, until the tier has flushed and committed what capture left in Tier 0, so a
    broker stop doesn't strand it (#1684). Returns how many payloads were still there when it gave up, 0 once
    drained. It only reads; a broker it can't reach counts as nothing to wait for."""
    import continuum_py

    if not streams:
        return 0
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    client = None
    try:
        client = await continuum_py.Client.connect(
            str(socket), client_name="restart-drain", request_timeout_s=min(max(timeout, 0.1), ATTEMPT_S)
        )
        now_ns = time.time_ns()
        while True:
            left = await tier0_left(client, streams, now_ns)
            if left == 0 or loop.time() + DRAIN_POLL_S > deadline:
                return left
            await asyncio.sleep(DRAIN_POLL_S)
    except Exception as error:
        if not transient(error):
            raise
        return 0
    finally:
        if client is not None:
            await _close(client, deadline)


async def wait_for_broker_restart(
    socket: Path, previous: BrokerIdentity | None, timeout: float = WAIT_S
) -> BrokerIdentity:
    """Wait for a usable broker, requiring a new arena identity when the prior instance is known."""
    deadline = asyncio.get_running_loop().time() + timeout
    reason = "broker instance has not changed"
    while (remaining := deadline - asyncio.get_running_loop().time()) > 0:
        try:
            identity = await broker_identity(socket, min(ATTEMPT_S, remaining))
            if previous is None or identity != previous:
                return identity
        except Exception as error:
            if not transient(error):
                raise
            reason = str(error)
        await asyncio.sleep(min(0.1, max(0.0, deadline - asyncio.get_running_loop().time())))
    raise TimeoutError(f"broker restart readiness timed out after {timeout:g} s: {reason}")


MIN_REQUEST_S = 0.01  # continuum_py's least request_timeout_s


def capture_needs(manifest: Path) -> tuple[CaptureNeed, ...]:
    """Validate the configured manifest identities and require enabled cameras/microphones/arrays."""
    data = tomllib.loads(manifest.read_text())
    devices = data.get("device", [])
    if not isinstance(devices, list):
        raise ValueError("capture manifest device must be an array of tables")
    result: list[CaptureNeed] = []
    names: set[str] = set()
    for device in devices:
        if not isinstance(device, dict):
            raise ValueError("capture manifest device entries must be tables")
        name, kind = device.get("name"), device.get("kind")
        if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9_-]+", name) is None or name in names:
            raise ValueError(f"invalid or duplicate capture device name {name!r}")
        names.add(name)
        if not isinstance(kind, str):
            raise ValueError(f"{name}: capture device kind must be a string")
        if kind not in {"camera", "rtsp", "microphone", "xvf3800", "teensy"}:
            raise ValueError(f"{name}: unsupported capture device kind {kind!r}")
        enabled = device.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(f"{name}: capture device enabled must be a boolean")
        if not enabled:
            continue
        if kind in {"camera", "rtsp"}:
            result.append(CaptureNeed(f"{name}.video", "video"))
        elif kind in {"microphone", "xvf3800"}:
            result.append(CaptureNeed(f"{name}.audio", "audio"))
        elif kind == "teensy":
            LOG.info("%s: teensy readiness is unsupported by the camera/microphone gate", name)
        else:
            raise ValueError(f"{name}: unsupported capture device kind {kind!r}")
    return tuple(result)


def _capture_fresh(header: PayloadHeader, barrier: int) -> bool:
    capture = header.get("capture")
    if capture is None:
        raise ValueError("capture header has no capture metadata")
    values = (capture.get("capture_mono_ns"), capture.get("arrival_ns"), capture.get("device_seq"))
    if any(not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("capture header has invalid monotonic/sequence metadata")
    return capture["capture_mono_ns"] >= barrier and capture["arrival_ns"] >= barrier


def _check_slot(client: Client, locator: Locator, identity: BrokerIdentity) -> None:
    import continuum_py

    if (locator["arena"], locator["arena_generation"]) not in identity:
        raise _Pending("video slot belongs to a different broker arena generation")
    try:
        arena = client.arena(locator["arena"])
        lease = arena.lease(locator["slab_class"], locator["slot"], locator["slot_generation"], ttl_ms=250)
    except (KeyError, continuum_py.StaleError, continuum_py.NotReadyError, ValueError) as error:
        raise _Pending("video slot moved before the readiness lease") from error
    try:
        if locator["length"] <= 0 or locator["length"] > memoryview(lease).nbytes:
            raise ValueError("capture video locator has an invalid payload length")
    finally:
        valid = lease.release()
    if not valid:
        raise _Pending("video readiness slot lease expired")


async def _fresh_event(
    client: Client,
    sub: Subscription,
    need: CaptureNeed,
    stream_id: int,
    barrier: int,
    identity: BrokerIdentity,
    ring: StreamLease | None = None,
) -> PayloadHeader:
    import continuum_py

    async for event in sub:
        if event["type"] == "Undecodable":
            raise ValueError(f"{need.stream}: undecodable live capture header: {event.get('reason', 'unknown error')}")
        if event["type"] != "Event" or "item" not in event:
            continue
        item = event["item"]
        header = continuum_py.decode_header(item["header"])
        if header["stream_id"] != stream_id:
            raise ValueError(f"{need.stream}: mismatched capture stream id")
        if not _capture_fresh(header, barrier):
            continue
        locator = item["locator"]
        if locator is None:
            continue
        if need.kind == "video":
            if locator["kind"] != "Shm":
                continue  # recorded chunks and replayed headers are not fresh Tier 0 capture
            capture = header["capture"]
            if capture is None or capture["width"] <= 0 or capture["height"] <= 0:
                raise ValueError(f"{need.stream}: invalid video geometry")
            try:
                _check_slot(client, locator, identity)
            except _Pending:
                continue  # a later live frame can replace an overwritten slot
        else:
            if (
                header["format"] != "RawBytes"
                or re.fullmatch(r"PCM_(?:S(?:16|24|32)LE|F32LE)x[1-9][0-9]*", header["shape_name"]) is None
            ):
                raise ValueError(f"{need.stream}: invalid live PCM batch format")
            capture = header["capture"]
            if capture is None or capture.get("sample_rate_hz", 0) <= 0:
                raise ValueError(f"{need.stream}: missing live PCM sample rate")
            if locator["kind"] != "Ring":
                continue
            if (
                ring is None
                or locator.get("arena") != ring["arena"]
                or locator.get("arena_generation") != ring["arena_generation"]
                or locator.get("ring") != ring["ring_index"]
            ):
                raise _Pending(f"{need.stream}: PCM batch belongs to a replaced ring")
        return header
    raise ConnectionError(f"{need.stream}: capture subscription ended")


async def _fresh_record(reader: RingReader, barrier: int) -> tuple[int, int]:
    while True:
        got = await reader.next_record(timeout_ms=250)
        if got is None:
            continue
        sequence, record, _lost = got
        if len(record) != reader.stride or len(record) < 16:
            raise ValueError("capture PCM record has an invalid stride")
        captured = int.from_bytes(record[:8], "little")
        if captured >= barrier:
            return sequence, captured


async def _capture_one(
    client: Client, need: CaptureNeed, stream_id: int, identity: BrokerIdentity, barrier: int, deadline: float
) -> CaptureEvidence:
    import continuum_py

    try:
        sub = await client.subscribe([need.stream], start="Live", pinned=True)
    except continuum_py.BrokerError as error:
        if error.args and error.args[0] in {"NotFound", "Unavailable"}:
            raise _Pending(f"{need.stream}: {error}") from error
        raise
    try:
        if need.kind == "video":
            header = await _fresh_event(client, sub, need, stream_id, barrier, identity)
            capture = header["capture"]
            if capture is None:
                raise ValueError(f"{need.stream}: missing video capture metadata")
            return CaptureEvidence(need.stream, identity, capture["capture_mono_ns"], capture["device_seq"])
        rings = sub.rings
        if not rings:
            raise _Pending(f"{need.stream}: capture has not carved its PCM ring")
        if len(rings) != 1 or rings[0]["stream_id"] != stream_id:
            raise ValueError(f"{need.stream}: ambiguous PCM ring grant")
        ring = rings[0]["lease"]
        if (ring["arena"], ring["arena_generation"]) not in identity:
            raise _Pending(f"{need.stream}: PCM ring belongs to a different broker generation")
        try:
            reader = client.ring_reader(ring, from_oldest=False)
        except (continuum_py.StaleError, continuum_py.NotReadyError) as error:
            raise _Pending(f"{need.stream}: PCM ring was replaced") from error
        record = asyncio.create_task(_fresh_record(reader, barrier))
        batch = asyncio.create_task(_fresh_event(client, sub, need, stream_id, barrier, identity, ring))
        try:
            (sequence, captured), _header = await asyncio.gather(record, batch)
            return CaptureEvidence(need.stream, identity, captured, sequence)
        finally:
            record.cancel()
            batch.cancel()
            await asyncio.gather(record, batch, return_exceptions=True)
    finally:
        await _close(sub, deadline, subscription=True)


async def _capture_tracked(
    client: Client,
    need: CaptureNeed,
    stream_id: int,
    identity: BrokerIdentity,
    barrier: int,
    deadline: float,
    pending: set[str],
) -> CaptureEvidence:
    evidence = await _capture_one(client, need, stream_id, identity, barrier, deadline)
    pending.discard(need.stream)
    return evidence


async def wait_for_capture(
    socket: Path, needs: Sequence[CaptureNeed], timeout: float = WAIT_S
) -> tuple[CaptureEvidence, ...]:
    """Require fresh video slots AND fresh PCM records/batches for every configured need on one broker instance."""
    import continuum_py

    needs = tuple(needs)
    if not needs:
        return ()
    if len({need.stream for need in needs}) != len(needs) or any(need.kind not in {"audio", "video"} for need in needs):
        raise ValueError("capture readiness needs must have unique streams and supported kinds")
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    pending = {need.stream for need in needs}
    reason = "no fresh capture activity"
    # The client refuses a request timeout under 10 ms; a sliver left at the deadline is a timeout, not a crash.
    while (remaining := deadline - loop.time()) >= MIN_REQUEST_S:
        client = None
        tasks: list[asyncio.Task[CaptureEvidence]] = []
        pending = {need.stream for need in needs}
        try:
            async with asyncio.timeout(min(ATTEMPT_S, remaining * 0.75)):
                client = await continuum_py.Client.connect(
                    str(socket), client_name="capture-readiness", request_timeout_s=min(ATTEMPT_S, remaining)
                )
                identity = await _identity(client)
                rows = await client.list_streams()
                live = {row["name"]: row["stream_id"] for row in rows if row.get("live")}
                missing = pending - live.keys()
                if missing:
                    pending = set(missing)
                    raise _Pending(f"capture streams are missing or not live: {', '.join(sorted(missing))}")
            barrier = time.monotonic_ns()
            # Keep time for concurrent unsubscribe cancellation plus the client close.
            cleanup_budget = min(0.4, max(0.0, deadline - loop.time()) * 0.25)
            async with asyncio.timeout(max(0.0, deadline - loop.time() - cleanup_budget)):
                tasks = [
                    asyncio.create_task(
                        _capture_tracked(client, need, live[need.stream], identity, barrier, deadline, pending)
                    )
                    for need in needs
                ]
                result = await asyncio.gather(*tasks)
                async with asyncio.timeout(min(ATTEMPT_S, max(0.0, deadline - loop.time()))):
                    if await _identity(client) != identity:
                        raise _Pending("broker arena identity changed while proving capture readiness")
                    rows = await client.list_streams()
                    live_names = {row["name"] for row in rows if row.get("live")}
                    if any(need.stream not in live_names for need in needs):
                        raise _Pending("a capture producer disconnected during readiness")
                return tuple(result)
        except Exception as error:
            if not isinstance(error, (_Pending, continuum_py.StaleError)) and not transient(error):
                raise
            reason = str(error) or "no fresh capture activity"
            pending = pending or {need.stream for need in needs}
        finally:
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if client is not None:
                try:
                    await _close(client, deadline)
                except TimeoutError as error:
                    raise TimeoutError(
                        f"capture readiness cleanup timed out; pending {', '.join(sorted(pending or {need.stream for need in needs}))}"
                    ) from error
        await asyncio.sleep(min(0.1, max(0.0, deadline - loop.time())))
    raise TimeoutError(
        f"capture readiness timed out after {timeout:g} s; pending {', '.join(sorted(pending))}: {reason}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--socket", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a service command is required")
    try:
        asyncio.run(wait_for_broker(args.socket))
    except Exception:
        LOG.exception("broker readiness failed; the service was not started")
        return 1
    os.execv(command[0], command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
