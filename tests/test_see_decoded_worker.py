"""Synthetic source, real daemon fanout, and real cache clients; no decoder or GPU is initialized."""

import asyncio
import json
import os
import threading
from collections import deque
from contextlib import aclosing
from typing import ClassVar, cast
from unittest.mock import AsyncMock, create_autospec

import cv2
import numpy as np
import pytest
from continuum_py import Client, PayloadHeader

from liberational.see import decoded_worker as worker
from liberational.see import shared_frames
from liberational.see.shared_frames import decoded_frames


def source(number=1):
    return {
        "id": f"01900000-0000-7000-8000-{number:012x}",
        "stream_id": 7,
        "start_ns": 1800000000000000000 + number,
        "end_ns": 1800000000000000000 + number,
        "shape_name": "H264AnnexB",
        "seq": number,
        "parents": [],
        "capture": {"width": 4, "height": 4, "device_seq": number, "flip": False},
    }


class Subscription:
    """continuum_py.Subscription as a camera's decode thread reads it: `take` blocks for the next events and returns
    them with every other one already waiting, and leaving `async with` (an unsubscribe) ends a waiting `take` with
    None. `put` adds units all at once, as units that arrive while the thread is busy decoding."""

    def __init__(self):
        self.ready = threading.Condition()
        self.waiting = deque()
        self.closed = False
        self.batches = []  # how many events each take returned

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        with self.ready:
            self.closed = True
            self.ready.notify_all()

    def put(self, *events):
        with self.ready:
            self.waiting.extend(events)
            self.ready.notify_all()

    def empty(self):
        with self.ready:
            return not self.waiting

    def take(self, timeout_s, max=64):
        with self.ready:
            self.ready.wait_for(lambda: self.waiting or self.closed, timeout_s)
            if self.closed:
                return None
            taken = [self.waiting.popleft() for _ in range(min(max, len(self.waiting)))]
            if taken:
                self.batches.append(len(taken))
            return taken


class Arena:
    def __init__(self, payload=b"annexb"):
        self.payload = payload

    def lease(self, slab_class, slot, generation, milliseconds):
        assert (slab_class, slot, generation) == (0, 3, 8)
        assert milliseconds > 0
        return memoryview(self.payload)


class SourceClient:
    def __init__(self, payload=b"annexb"):
        self.payload = payload
        self.subscription = Subscription()
        self.subscribed = asyncio.Event()
        self.calls = []
        self.registrations = []

    async def register_worker(self, name, capabilities, *, keeps_copies):
        self.registrations.append((name, capabilities, keeps_copies))

    def arena(self, name):
        assert name == "synthetic-arena"
        return Arena(self.payload)

    async def subscribe(self, streams, *, pinned):
        self.calls.append((streams, pinned))
        self.subscribed.set()
        return self.subscription


def event(length=6):
    return {
        "type": "Event",
        "item": {
            "header": b"synthetic",
            "locator": {
                "kind": "Shm",
                "arena": "synthetic-arena",
                "slab_class": 0,
                "slot": 3,
                "slot_generation": 8,
                "length": length,
            },
        },
    }


async def wanted(channel, format):
    async with asyncio.timeout(2):
        while format not in channel.wanted:
            await asyncio.sleep(0)  # Yield to the socket handler, without a wall-clock delay.


@pytest.mark.parametrize(("format", "native_bgr"), [("nv12", False), ("bgr", False), ("bgr", True)])
def test_two_consumers_share_one_decoder_and_one_conversion_per_source_frame(tmp_path, monkeypatch, format, native_bgr):
    async def run():
        client = SourceClient()
        header = source()
        pixels = bytes([16] * 16 + [128] * 8)
        native = create_autospec(worker.NativeDecoder, instance=True)
        header["capture"]["flip"] = native_bgr
        bgr = np.arange(48, dtype=np.uint8).reshape(4, 4, 3).tobytes() if native_bgr else None
        color_range, color_matrix = ("full", "bt709") if native_bgr else ("unspecified", "unspecified")
        native.push.return_value = ([(pixels, 4, 4, header, color_range, color_matrix, bgr)], 0)
        constructor = create_autospec(worker.NativeDecoder, return_value=native)
        monkeypatch.setattr(worker, "NativeDecoder", constructor)
        decode = create_autospec(worker.continuum_py.decode_header, return_value=header)
        monkeypatch.setattr(worker.continuum_py, "decode_header", decode)
        convert = create_autospec(cv2.cvtColor, side_effect=cv2.cvtColor)
        monkeypatch.setattr(cv2, "cvtColor", convert)
        root = tmp_path / "decoded"
        root.mkdir()
        address = tmp_path / "decoded.sock"
        service = worker.DecoderService(cast(Client, client), ["synthetic"], 0, root)
        stop = asyncio.Event()
        running = asyncio.create_task(service.run(address, stop))
        try:
            await asyncio.wait_for(client.subscribed.wait(), 2)
            async with (
                aclosing(decoded_frames("synthetic", format=format, socket=address)) as first,
                aclosing(decoded_frames("synthetic", format=format, socket=address)) as second,
            ):
                a = asyncio.create_task(anext(first))
                b = asyncio.create_task(anext(second))
                await wanted(service.channels["synthetic"], format)
                client.subscription.put(
                    {
                        "type": "Event",
                        "item": {
                            "header": b"synthetic",
                            "locator": {
                                "kind": "Shm",
                                "arena": "synthetic-arena",
                                "slab_class": 0,
                                "slot": 3,
                                "slot_generation": 8,
                                "length": 6,
                            },
                        },
                    }
                )
                one, two = await asyncio.wait_for(asyncio.gather(a, b), 2)
                assert one.source_header == two.source_header == header
                assert one.capture_ns == two.capture_ns == 1800000000000000001
                assert one.decoder_epoch == two.decoder_epoch
                np.testing.assert_array_equal(one.pixels, two.pixels)
                constructor.assert_called_once_with(0, "synthetic")
                native.push.assert_called_once_with(header, b"annexb", format == "bgr", True)
                assert client.calls == [(["synthetic.video"], True)]
                assert convert.call_count == (1 if format == "bgr" and not native_bgr else 0)
                assert (one.color_range, one.color_matrix) == (color_range, color_matrix)
                assert (two.color_range, two.color_matrix) == (color_range, color_matrix)
                expected = (
                    np.zeros((4, 4, 3), np.uint8) if format == "bgr" else np.frombuffer(pixels, np.uint8).reshape(6, 4)
                )
                if native_bgr:
                    expected = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)[::-1, ::-1]
                np.testing.assert_array_equal(one.pixels, expected)
        finally:
            stop.set()
            await asyncio.wait_for(running, 2)
        assert client.subscription.closed
        assert not address.exists()
        assert list(root.iterdir()) == []

    asyncio.run(run())


def test_shutdown_disconnects_an_idle_connected_consumer_and_removes_cache(tmp_path, monkeypatch):
    async def run():
        client = SourceClient()
        native = create_autospec(worker.NativeDecoder, instance=True)
        monkeypatch.setattr(worker, "NativeDecoder", create_autospec(worker.NativeDecoder, return_value=native))
        root = tmp_path / "decoded"
        root.mkdir()
        address = tmp_path / "decoded.sock"
        service = worker.DecoderService(cast(Client, client), ["synthetic"], 0, root)
        stop = asyncio.Event()
        running = asyncio.create_task(service.run(address, stop))
        waiting = None
        try:
            await asyncio.wait_for(client.subscribed.wait(), 2)
            async with aclosing(decoded_frames("synthetic", socket=address)) as frames:
                waiting = asyncio.create_task(anext(frames))
                await wanted(service.channels["synthetic"], "nv12")
                # The reader reconnects to a decoder that goes away, and gives up at its startup deadline.
                monkeypatch.setattr(shared_frames, "STARTUP_WAIT_S", 0.0)
                monkeypatch.setattr(shared_frames, "RECONNECT_S", 0.0)
                stop.set()
                await asyncio.wait_for(running, 2)
                with pytest.raises(TimeoutError, match="startup deadline"):
                    await asyncio.wait_for(waiting, 2)
        finally:
            stop.set()
            await asyncio.gather(running, return_exceptions=True)
            if waiting is not None:
                waiting.cancel()
                await asyncio.gather(waiting, return_exceptions=True)
        assert not address.exists()
        assert list(root.iterdir()) == []
        assert service.clients == 0

    asyncio.run(run())


def test_non_socket_path_is_never_replaced(tmp_path):
    async def run():
        address = tmp_path / "decoded.sock"
        address.write_bytes(b"keep synthetic sentinel")
        root = tmp_path / "decoded"
        root.mkdir()
        service = worker.DecoderService(cast(Client, SourceClient()), ["synthetic"], 0, root)
        with pytest.raises(ValueError, match="non-socket"):
            await service.run(address, asyncio.Event())
        assert address.read_bytes() == b"keep synthetic sentinel"

    asyncio.run(run())


def test_camera_format_failure_cleans_its_published_cache_and_socket(tmp_path, monkeypatch):
    async def run():
        client = SourceClient()
        good = source()
        bad = source(2) | {"shape_name": "Opus"}
        native = create_autospec(worker.NativeDecoder, instance=True)
        native.push.return_value = ([(bytes([16] * 16 + [128] * 8), 4, 4, good, "unspecified", "unspecified", None)], 0)
        monkeypatch.setattr(worker, "NativeDecoder", create_autospec(worker.NativeDecoder, return_value=native))
        monkeypatch.setattr(
            worker.continuum_py,
            "decode_header",
            create_autospec(worker.continuum_py.decode_header, side_effect=[good, bad]),
        )
        root = tmp_path / "decoded"
        root.mkdir()
        address = tmp_path / "decoded.sock"
        service = worker.DecoderService(cast(Client, client), ["synthetic"], 0, root)
        stop = asyncio.Event()
        running = asyncio.create_task(service.run(address, stop))
        try:
            await asyncio.wait_for(client.subscribed.wait(), 2)
            async with aclosing(decoded_frames("synthetic", socket=address)) as frames:
                waiting = asyncio.create_task(anext(frames))
                await wanted(service.channels["synthetic"], "nv12")
                event = {
                    "type": "Event",
                    "item": {
                        "header": b"synthetic",
                        "locator": {
                            "kind": "Shm",
                            "arena": "synthetic-arena",
                            "slab_class": 0,
                            "slot": 3,
                            "slot_generation": 8,
                            "length": 6,
                        },
                    },
                }
                client.subscription.put(event)
                assert (await asyncio.wait_for(waiting, 2)).source_header == good
                assert list(root.iterdir())
                client.subscription.put(event)
                with pytest.raises(ValueError, match="H264AnnexB"):
                    await asyncio.wait_for(running, 2)
        finally:
            stop.set()
            await asyncio.gather(running, return_exceptions=True)
        assert client.subscription.closed
        assert not address.exists()
        assert list(root.iterdir()) == []

    asyncio.run(run())


def test_decoder_initialization_failure_still_shuts_down_its_owner_executor(tmp_path, monkeypatch):
    async def run():
        real_executor = worker.concurrent.futures.ThreadPoolExecutor
        executors = []

        def executor_factory(*args, **kwargs):
            executor = real_executor(*args, **kwargs)
            shutdown = create_autospec(executor.shutdown, side_effect=executor.shutdown)
            monkeypatch.setattr(executor, "shutdown", shutdown)
            executors.append(executor)
            return executor

        monkeypatch.setattr(
            worker.concurrent.futures,
            "ThreadPoolExecutor",
            create_autospec(real_executor, side_effect=executor_factory),
        )
        monkeypatch.setattr(
            worker,
            "NativeDecoder",
            create_autospec(worker.NativeDecoder, side_effect=ValueError("synthetic decoder unavailable")),
        )
        root = tmp_path / "decoded"
        root.mkdir()
        address = tmp_path / "decoded.sock"
        service = worker.DecoderService(cast(Client, SourceClient()), ["synthetic"], 0, root)
        try:
            with pytest.raises(ValueError, match="synthetic decoder unavailable"):
                await asyncio.wait_for(service.run(address, asyncio.Event()), 2)
            assert len(executors) == 1
            executors[0].shutdown.assert_called_once_with(wait=False, cancel_futures=True)
            assert not address.exists()
            assert list(root.iterdir()) == []
        finally:
            for executor in executors:
                executor.shutdown(wait=True, cancel_futures=True)

    asyncio.run(run())


DAY_NS = 86_400_000_000_000


def notice(purge, ids=(), spans=()):
    return {"type": "PurgeNotice", "purge_id": purge, "payload_ids": list(ids), "spans": list(spans)}


async def holding(tmp_path, *numbers):
    """A service whose one camera holds NV12 copies of these source payloads."""
    service = worker.DecoderService(create_autospec(Client, instance=True), ["cam0"], 0, tmp_path)
    channel = service.channels["cam0"]
    channel.attach(1, "nv12")
    for number in numbers:
        picture = cast(PayloadHeader, source(number))
        await channel.publish(bytes(4 * 4 * 3 // 2), 4, 4, picture, 0, channel.pick(picture, 0), "limited", "bt601")
    return service


def test_only_a_purge_reaching_the_decoded_copies_touches_them(tmp_path):
    async def run():
        service = await holding(tmp_path, 1, 2)
        oldest = source(1)["start_ns"]
        never = await holding(tmp_path / "idle")  # nothing decoded yet
        return [
            service.touches(notice("p", ids=[source(2)["id"]])),  # one of its cached pictures
            service.touches(
                notice(
                    "p",
                    ids=["01900000-0000-7000-8000-0000000000ff"],
                    spans=[
                        {
                            "stream_id": 7,
                            "range": (oldest - 31 * DAY_NS, oldest - 30 * DAY_NS),
                        }  # an age-out: month-old video
                    ],
                )
            ),
            service.touches(notice("p", spans=[{"stream_id": 7, "range": (oldest - 20 * 10**9, oldest - 5 * 10**9)}])),
            never.touches(notice("p", ids=[source(1)["id"]], spans=[{"stream_id": 7, "range": (0, oldest)}])),
        ]

    assert asyncio.run(run()) == [True, False, True, False]


class Broker:
    """A client whose events are fed by the test, and whose acknowledgements are recorded in order."""

    def __init__(self, monkeypatch):
        self.events: asyncio.Queue = asyncio.Queue()
        self.log: list[str] = []
        self.client = create_autospec(Client, instance=True)
        self.client.next_event = AsyncMock(spec=Client.next_event, side_effect=self.next)
        self.client.ack_purge.side_effect = lambda purge: self.log.append(f"ack {purge}")
        monkeypatch.setattr(worker.continuum_py, "decode_event", lambda data: data)

    async def next(self):
        item = await self.events.get()
        return None if item is None else (0, item)


class Epoch:
    """A decoder epoch that touches every purge (live copies) and cleans up when stopped."""

    def __init__(self, log, touching=True):
        self.log, self.touching, self.cleaned = log, touching, False

    def touches(self, notice):
        return self.touching

    async def run(self, address, stop, *, register=True):
        self.log.append(f"start register={register}")
        await stop.wait()
        self.cleaned = True
        self.log.append("stopped")


def test_a_burst_of_purges_reaching_the_copies_is_one_restart_in_process_then_one_acknowledgement_each(monkeypatch):
    async def run():
        broker = Broker(monkeypatch)
        purges = worker.Purges(broker.client)
        epochs = []

        def make():
            epochs.append(Epoch(broker.log))
            return cast(worker.DecoderService, epochs[-1])

        serving = asyncio.create_task(worker.serve(purges, make, asyncio.Event(), settle_s=0.05))
        await asyncio.sleep(0.01)
        for purge in ("p1", "p1", "p2", "p3"):  # a burst, with a second page of p1
            broker.events.put_nowait(notice(purge))
        await asyncio.sleep(0.2)  # past the settle: acknowledged, and the next epoch is running
        broker.events.put_nowait(None)  # the broker goes away: the service ends
        await asyncio.wait_for(serving, 2)
        return broker.log, len(epochs)

    log, epochs = asyncio.run(run())
    assert log == ["start register=True", "stopped", "ack p1", "ack p2", "ack p3", "start register=False", "stopped"]
    assert epochs == 2


def test_purges_that_cannot_reach_the_copies_are_acknowledged_at_once_without_a_restart(monkeypatch):
    async def run():
        broker = Broker(monkeypatch)
        purges = worker.Purges(broker.client)
        terminating = asyncio.Event()
        serving = asyncio.create_task(
            worker.serve(
                purges,
                lambda: cast(worker.DecoderService, Epoch(broker.log, touching=False)),
                terminating,
                settle_s=0.05,
            )
        )
        await asyncio.sleep(0.01)
        for number in range(25):  # an age-out batch
            broker.events.put_nowait(notice(f"old{number}"))
        await asyncio.sleep(0.05)
        terminating.set()
        purges.restart.set()
        await asyncio.wait_for(serving, 2)
        return broker.log

    log = asyncio.run(run())
    assert log == ["start register=True", *(f"ack old{n}" for n in range(25)), "stopped"]


def test_an_epoch_that_did_not_clean_up_never_acknowledges(monkeypatch):
    class Leaky(Epoch):
        async def run(self, address, stop, *, register=True):
            await stop.wait()  # cleaned stays False

    async def run():
        broker = Broker(monkeypatch)
        purges = worker.Purges(broker.client)
        serving = asyncio.create_task(
            worker.serve(purges, lambda: cast(worker.DecoderService, Leaky(broker.log)), asyncio.Event(), settle_s=0.01)
        )
        await asyncio.sleep(0.01)
        broker.events.put_nowait(notice("p1"))
        await asyncio.wait_for(serving, 2)
        return broker.log

    assert asyncio.run(run()) == []


def test_cancellation_during_decoder_construction_releases_it_on_its_owner_thread(tmp_path, monkeypatch):
    import threading

    async def run():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        destroyed = asyncio.Event()
        release = threading.Event()
        threads = {}

        class ThreadAffineDecoder:
            def __init__(self, gpu, name=""):
                threads["owner"] = threading.get_ident()
                loop.call_soon_threadsafe(started.set)
                assert release.wait(2), "test did not release decoder construction"

            def close(self):
                threads["destructor"] = threading.get_ident()
                loop.call_soon_threadsafe(destroyed.set)

        monkeypatch.setattr(worker, "NativeDecoder", ThreadAffineDecoder)
        channel = worker.Channel(tmp_path)
        service = worker.DecoderService(cast(Client, SourceClient()), ["synthetic"], 0, tmp_path)
        camera = asyncio.create_task(service.camera("synthetic", channel))
        try:
            await asyncio.wait_for(started.wait(), 2)
            camera.cancel()
            # Let cancellation reach the awaiting coroutine before the constructor can finish.
            cancellation_delivered = asyncio.Event()
            loop.call_soon(cancellation_delivered.set)
            await cancellation_delivered.wait()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await camera
            await asyncio.wait_for(destroyed.wait(), 2)
            assert threads["destructor"] == threads["owner"]
            assert threads["owner"] != threading.get_ident()
        finally:
            release.set()
            camera.cancel()
            await asyncio.gather(camera, return_exceptions=True)

    asyncio.run(run())


def test_competing_decoder_never_registers_keeper_before_obtaining_lock(tmp_path):
    import fcntl

    async def run():
        root = tmp_path / "decoded"
        root.mkdir()
        client = SourceClient()
        service = worker.DecoderService(cast(Client, client), ["synthetic"], 0, root)
        address = tmp_path / "decoded.sock"
        address.write_bytes(b"existing owner sentinel")
        with (tmp_path / "decoded.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with pytest.raises(BlockingIOError):
                await service.run(address, asyncio.Event())
        assert client.registrations == []
        assert address.read_bytes() == b"existing owner sentinel"
        assert not service.cleaned

    asyncio.run(run())


def test_stale_real_cache_is_removed_before_keeper_registration_but_unrelated_files_survive(tmp_path, monkeypatch):
    from liberational.see.decoded_cache import DecodedCache

    async def run():
        root = tmp_path / "decoded"
        root.mkdir()
        abandoned = DecodedCache(root, format="nv12", max_pixels=16)
        descriptor = abandoned.publish(np.zeros((6, 4), np.uint8), source())
        assert descriptor is not None
        abandoned_directory = abandoned.path
        unrelated = root / "unrelated"
        unrelated.mkdir()
        (unrelated / "keep").write_bytes(b"synthetic sentinel")
        client = SourceClient()
        stop = asyncio.Event()

        async def register(name, capabilities, *, keeps_copies):
            assert not abandoned_directory.exists()
            assert (unrelated / "keep").read_bytes() == b"synthetic sentinel"
            client.registrations.append((name, capabilities, keeps_copies))
            stop.set()

        monkeypatch.setattr(client, "register_worker", create_autospec(client.register_worker, side_effect=register))
        monkeypatch.setattr(worker, "NativeDecoder", create_autospec(worker.NativeDecoder, instance=False))
        service = worker.DecoderService(cast(Client, client), ["synthetic"], 0, root)
        try:
            await asyncio.wait_for(service.run(tmp_path / "decoded.sock", stop), 2)
            assert client.registrations == [("see-decode", [], True)]
            assert not abandoned_directory.exists()
            assert (unrelated / "keep").read_bytes() == b"synthetic sentinel"
            assert service.cleaned
        finally:
            abandoned.close()

    asyncio.run(run())


def test_failed_camera_cleanup_does_not_claim_transient_copies_are_discarded(tmp_path, monkeypatch):
    async def run():
        root = tmp_path / "decoded"
        root.mkdir()
        client = SourceClient()
        native = create_autospec(worker.NativeDecoder, instance=True)
        monkeypatch.setattr(worker, "NativeDecoder", create_autospec(worker.NativeDecoder, return_value=native))
        service = worker.DecoderService(cast(Client, client), ["synthetic"], 0, root)
        channel = service.channels["synthetic"]
        monkeypatch.setattr(
            channel,
            "invalidate",
            create_autospec(channel.invalidate, side_effect=RuntimeError("synthetic cleanup failed")),
        )
        stop = asyncio.Event()
        running = asyncio.create_task(service.run(tmp_path / "decoded.sock", stop))
        try:
            await asyncio.wait_for(client.subscribed.wait(), 2)
            stop.set()
            await asyncio.wait_for(running, 2)
            assert not service.cleaned
        finally:
            stop.set()
            await asyncio.gather(running, return_exceptions=True)

    asyncio.run(run())


def test_mjpeg_camera_is_shared_as_sensor_nv12_and_upright_bgr(tmp_path, monkeypatch):
    # Left half black, right half white, so the 180 degree flip is visible in BGR and absent from NV12.
    image = np.zeros((8, 8, 3), np.uint8)
    image[:, 4:] = 255
    ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 100])
    assert ok
    payload = jpeg.tobytes()

    async def run():
        client = SourceClient(payload)
        header = source() | {"shape_name": "MJPEG"}
        header["capture"] = {"width": 8, "height": 8, "device_seq": 1, "flip": True}
        native = create_autospec(worker.NativeDecoder, instance=True)
        native.breaks = 0
        monkeypatch.setattr(worker, "NativeDecoder", create_autospec(worker.NativeDecoder, return_value=native))
        monkeypatch.setattr(
            worker.continuum_py,
            "decode_header",
            create_autospec(worker.continuum_py.decode_header, return_value=header),
        )
        root = tmp_path / "decoded"
        root.mkdir()
        address = tmp_path / "decoded.sock"
        service = worker.DecoderService(cast(Client, client), ["cam0"], 0, root)
        stop = asyncio.Event()
        running = asyncio.create_task(service.run(address, stop))
        try:
            await asyncio.wait_for(client.subscribed.wait(), 2)
            async with (
                aclosing(decoded_frames("cam0", format="nv12", socket=address)) as nv12_reader,
                aclosing(decoded_frames("cam0", format="bgr", socket=address)) as bgr_reader,
            ):
                a = asyncio.create_task(anext(nv12_reader))
                b = asyncio.create_task(anext(bgr_reader))
                await wanted(service.channels["cam0"], "nv12")
                await wanted(service.channels["cam0"], "bgr")
                client.subscription.put(event(len(payload)))
                nv12, bgr = await asyncio.wait_for(asyncio.gather(a, b), 2)
        finally:
            stop.set()
            await asyncio.wait_for(running, 2)
        assert nv12.source_header == bgr.source_header == header
        assert (nv12.color_range, nv12.color_matrix) == ("limited", "bt601")
        assert nv12.pixels.shape == (12, 8)
        luma, chroma = nv12.pixels[:8].astype(int), nv12.pixels[8:].astype(int)
        # Limited-range BT.601: black is Y 16, white Y 235, both with neutral chroma 128.
        assert np.abs(luma[:, :3] - 16).max() <= 4
        assert np.abs(luma[:, 5:] - 235).max() <= 4
        assert np.abs(chroma - 128).max() <= 4
        assert bgr.pixels.shape == (8, 8, 3)
        assert bgr.pixels[:, :3].min() >= 250, "flipped: white is now on the left"
        assert bgr.pixels[:, 5:].max() <= 5
        native.push.assert_not_called()

    asyncio.run(run())


def test_camera_without_readers_is_not_decoded(tmp_path, monkeypatch):
    async def run():
        client = SourceClient()
        native = create_autospec(worker.NativeDecoder, instance=True)
        native.push.return_value = ([], 1)
        monkeypatch.setattr(worker, "NativeDecoder", create_autospec(worker.NativeDecoder, return_value=native))
        decode = create_autospec(worker.continuum_py.decode_header, return_value=source())
        monkeypatch.setattr(worker.continuum_py, "decode_header", decode)
        root = tmp_path / "decoded"
        root.mkdir()
        service = worker.DecoderService(cast(Client, client), ["synthetic"], 0, root)
        stop = asyncio.Event()
        running = asyncio.create_task(service.run(tmp_path / "decoded.sock", stop))
        try:
            await asyncio.wait_for(client.subscribed.wait(), 2)
            for _ in range(3):
                client.subscription.put(event())
            async with asyncio.timeout(2):
                while not client.subscription.empty() or not native.push.call_count:
                    await asyncio.sleep(0)
        finally:
            stop.set()
            await asyncio.wait_for(running, 2)
        # One reset when the camera goes idle, then nothing decoded or even read.
        native.push.assert_called_once_with({}, None)
        decode.assert_not_called()

    asyncio.run(run())


def test_a_camera_with_no_picture_yet_is_answered_as_waiting_and_keeps_its_connection(tmp_path, monkeypatch):
    # The 12:24 cascade: capture wasn't up yet, the 10 s wait raised a bare TimeoutError, and every consumer
    # crashed on its empty "error" reply.
    async def run():
        client = SourceClient()
        native = create_autospec(worker.NativeDecoder, instance=True)
        monkeypatch.setattr(worker, "NativeDecoder", create_autospec(worker.NativeDecoder, return_value=native))
        monkeypatch.setattr(worker, "WAITING_REPLY_S", 0.05)
        root = tmp_path / "decoded"
        root.mkdir()
        address = tmp_path / "decoded.sock"
        service = worker.DecoderService(cast(Client, client), ["synthetic"], 0, root)
        stop = asyncio.Event()
        running = asyncio.create_task(service.run(address, stop))
        try:
            await asyncio.wait_for(client.subscribed.wait(), 2)
            reader, writer = await asyncio.open_unix_connection(str(address))
            request = b'{"camera":"synthetic","format":"nv12","after":0,"owner":null,"break_count":null}\n'
            for _ in range(2):
                writer.write(request)
                await writer.drain()
                assert json.loads(await asyncio.wait_for(reader.readline(), 2)) == {"type": "DecodedWaiting"}
            writer.close()
            await writer.wait_closed()
        finally:
            stop.set()
            await asyncio.gather(running, return_exceptions=True)

    asyncio.run(run())


def test_the_native_decoder_is_dropped_on_its_owner_thread_despite_its_break_callback_cycle(tmp_path, monkeypatch):
    # H264Frames keeps NativeDecoder.broke, so the pair is a cycle: without an explicit close, the garbage collector
    # freed the pyo3 decoder on the event loop's thread ("unsendable, but is being dropped on another thread").
    import gc
    import threading

    dropped = {}

    class Native:
        def __del__(self):
            dropped["thread"] = threading.current_thread().name

    class CyclicFrames:
        def __init__(self, on_break, **kwargs):
            self.on_break = on_break
            self.native = Native()

    monkeypatch.setattr(worker.frames, "H264Frames", CyclicFrames)

    async def run():
        client = SourceClient()
        service = worker.DecoderService(cast(Client, client), ["synthetic"], 0, tmp_path)
        camera = asyncio.create_task(service.camera("synthetic", service.channels["synthetic"]))
        await asyncio.wait_for(client.subscribed.wait(), 2)
        camera.cancel()
        await asyncio.gather(camera, return_exceptions=True)

    asyncio.run(run())
    gc.collect()
    assert dropped["thread"].startswith("camera-decode")


NO_HEADER = cast(PayloadHeader, {})


def unit(id: str) -> PayloadHeader:
    return cast(PayloadHeader, {"id": id})


class FakeNvdec:
    """frames.H264Frames' surface as NativeDecoder uses it: counts resets, and decodes a unit to one picture only
    once it has seen a keyframe since the last reset (as NVDEC restarts its chain at the next IDR)."""

    def __init__(self, on_break, *, backend: str = "software", gpu: int = 0, name: str = "") -> None:
        self.on_break, self.resets, self.keyed = on_break, 0, False
        self.last_outputs: dict[str, tuple[str, str, bytes | None]] = {}

    def broke(self) -> None:
        self.resets += 1
        self.keyed = False
        self.on_break()

    def push_nv12(self, header, data, rich=False, want_bgr=False, keep=True):
        self.keyed = self.keyed or data == b"idr"
        if not self.keyed:
            return []
        self.last_outputs[header["id"]] = ("limited", "bt601", None)
        return [(b"nv12", 64, 48, header)]


def test_a_run_of_unreadable_frames_resets_the_decoder_once_until_it_decodes_again(monkeypatch) -> None:
    monkeypatch.setattr(worker.frames, "H264Frames", FakeNvdec)
    native = worker.NativeDecoder(gpu=0)
    nvdec = cast(FakeNvdec, native.decoder)
    for _ in range(50):  # see-decode reading behind: every frame's slot is already gone
        assert native.push(NO_HEADER, None) == ([], 1)
    assert (nvdec.resets, native.breaks, native.lost) == (1, 1, 50)
    assert native.push(unit("p"), b"p") == ([], 1), "no picture until the next keyframe"
    assert native.push(NO_HEADER, None)[1] == 1, "still waiting for a keyframe: no second reset"
    pictures, breaks = native.push(unit("k"), b"idr")
    assert (len(pictures), breaks) == (1, 1)
    assert native.push(NO_HEADER, None)[1] == 2, "decoding again, so the next loss is a new gap"
    assert nvdec.resets == 2


class FlakyNvdec(FakeNvdec):
    """FakeNvdec whose NVDEC instances do what `plan` says, one entry per NVDEC start: "ok", "no-start" (cuCtxCreate
    out of memory), "dies" (fails on its first unit, as 11-06's did), or "ok-then-dies" (decodes one unit, then
    fails). Any NVDEC instance fails a unit of b"oom". The CPU one always works."""

    plan: ClassVar[list[str]] = []

    def __init__(self, on_break, *, backend: str = "software", gpu: int = 0, name: str = "") -> None:
        self.backend = backend
        self.fate = FlakyNvdec.plan.pop(0) if backend == "cuda" else "ok"
        if self.fate == "no-start":
            raise ValueError("cuCtxCreate: CUDA_ERROR_OUT_OF_MEMORY")
        super().__init__(on_break, backend=backend, gpu=gpu)

    def push_nv12(self, header, data, rich=False, want_bgr=False, keep=True):
        if self.fate == "dies" or (self.backend == "cuda" and data == b"oom"):
            raise ValueError("decode CUDA device: Generic error in an external library")
        if self.fate == "ok-then-dies":
            self.fate = "dies"
        return super().push_nv12(header, data, rich, want_bgr, keep)


def flaky(monkeypatch, plan, start=0.0):
    """A NativeDecoder over FlakyNvdec with `plan`, and a function that pushes a unit at a clock time (seconds from
    `start`) and answers which backend decoded and how many pictures came out."""
    clock = [start]
    monkeypatch.setattr(worker, "_monotonic", lambda: clock[0])
    monkeypatch.setattr(FlakyNvdec, "plan", list(plan))
    monkeypatch.setattr(worker.frames, "H264Frames", FlakyNvdec)
    native = worker.NativeDecoder(gpu=0)

    def at(t, data=b"idr"):
        clock[0] = start + t
        pictures, _ = native.push(unit(f"u{t}"), data)
        return cast(FlakyNvdec, native.decoder).backend, len(pictures)

    return at


def test_after_nvdec_fails_the_cpu_decodes_and_nvdec_is_tried_again_until_it_is_back(monkeypatch, capsys) -> None:
    # #2903: one CUDA out-of-memory error on 11-06 meant 18 hours of CPU decoding, until a restart.
    at = flaky(monkeypatch, ["dies", "no-start", "ok"], start=1000.0)
    assert at(0) == ("software", 1)  # NVDEC died on this unit, and the CPU decoded it
    assert at(59) == ("software", 1)  # not a minute yet
    assert at(60) == ("software", 1)  # NVDEC tried: out of memory, so the CPU goes on, 5 minutes until the next try
    assert at(359) == ("software", 1)
    assert at(360, b"p") == ("cuda", 0)  # NVDEC back, starting at the next keyframe
    assert at(361) == ("cuda", 1)
    log = capsys.readouterr().err
    assert log.count("NVDEC failed") == 1 and "trying NVDEC again in 1 min" in log
    assert "NVDEC still unavailable" in log and "trying again in 5 min" in log
    assert "NVDEC is back, after 6 min on the CPU" in log


def test_nvdec_that_stays_unavailable_is_tried_at_the_longest_wait_without_a_line_each_time(
    monkeypatch, capsys
) -> None:
    at = flaky(monkeypatch, ["no-start"] * 8)  # unavailable from the start
    for t in (60, 360, 1260, 2160, 3060):  # each try comes due: 1, 5, then every 15 minutes
        assert at(t) == ("software", 1)
    assert len(FlakyNvdec.plan) == 2, "the start and five tries"
    log = capsys.readouterr().err
    assert log.count("NVDEC still unavailable") == 2 and "trying again every 15 min" in log


def test_nvdec_that_starts_but_fails_its_first_unit_each_try_goes_quiet_too(monkeypatch, capsys) -> None:
    # The review of #2916: a context made but a decode out of memory, on every try.
    at = flaky(monkeypatch, ["dies"] * 7)
    for t in (0, 60, 360, 1260, 2160, 3060):
        assert at(t) == ("software", 1)
    assert len(FlakyNvdec.plan) == 1, "the start and five tries, each started and then failed"
    log = capsys.readouterr().err
    assert log.count("NVDEC failed") == 1, "the switch, once"
    assert log.count("NVDEC still unavailable") == 2 and "trying again every 15 min" in log


def test_nvdec_that_comes_back_and_fails_again_keeps_its_longer_wait(monkeypatch) -> None:
    # The review of #2916: reset on the first picture, this flapped every minute.
    at = flaky(monkeypatch, ["no-start", "ok-then-dies", "ok-then-dies", "ok"])
    assert at(60) == ("cuda", 1)  # back after a minute, for one picture
    assert at(61) == ("software", 1)  # failed again
    assert at(121) == ("software", 1), "not the first wait again"
    assert at(361) == ("cuda", 1)  # 5 minutes on
    assert at(362) == ("software", 1)
    assert at(1261) == ("software", 1)
    assert at(1262) == ("cuda", 1)  # 15 minutes on


def test_nvdec_steady_for_the_longest_wait_starts_the_waits_over(monkeypatch, capsys) -> None:
    at = flaky(monkeypatch, ["ok", "ok", "ok"])
    assert at(0) == ("cuda", 1)
    assert at(1, b"oom") == ("software", 0)  # out of memory: the CPU takes over from the next keyframe
    assert at(61) == ("cuda", 1)  # back
    assert at(961) == ("cuda", 1)  # decoding steadily for 15 minutes
    assert at(962, b"oom") == ("software", 0)
    assert at(1022) == ("cuda", 1), "a minute's wait again"
    assert capsys.readouterr().err.count("NVDEC failed") == 2, "each switch after steady decoding warns"


def test_with_no_native_decoder_a_run_of_losses_is_one_break(monkeypatch) -> None:
    def unavailable(*args, **kwargs):
        if kwargs.get("backend") == "cuda":
            raise ValueError("no CUDA here")
        raise NotImplementedError("continuum_py built without ffmpeg")

    monkeypatch.setattr(worker.frames, "H264Frames", unavailable)
    native = worker.NativeDecoder(gpu=0)
    assert native.decoder is None
    for _ in range(5):
        native.push(NO_HEADER, None)
    assert native.breaks == 1


def test_a_decoded_still_ends_the_wait_so_the_next_lost_still_is_a_new_gap(monkeypatch) -> None:
    monkeypatch.setattr(worker.frames, "H264Frames", FakeNvdec)
    native = worker.NativeDecoder(gpu=0)
    native.push(NO_HEADER, None)
    native.push(NO_HEADER, None)
    assert native.breaks == 1
    native.decoded()  # an MJPEG camera's next picture decoded on the CPU
    assert native.push(NO_HEADER, None)[1] == 2


def at_fps(fps: float, count: int) -> list[PayloadHeader]:
    """`count` pictures at `fps` of capture time from source(1)'s start, numbered from 1."""
    base = source(1)["start_ns"]
    starts = [base + round(n * 1e9 / fps) for n in range(count)]
    return [{**source(1 + n), "start_ns": at, "end_ns": at} for n, at in enumerate(starts)]


def test_a_bgr_reader_gets_a_picture_every_half_second_whatever_it_asks_while_nv12_takes_every_one(tmp_path) -> None:
    channel = worker.Channel(tmp_path)
    channel.attach(1, "bgr", 0)  # asks for every picture: BGR is converted on the CPU, so it gets one each 0.5 s
    channel.attach(2, "nv12", 0)
    picks = [channel.pick(header, 0) for header in at_fps(10, 12)]  # 10 fps for 1.2 s
    # Due at 500 and 1000 ms: each taken by the first picture within a quarter period (125 ms) before it.
    assert [n for n, pick in enumerate(picks) if "bgr" in pick] == [0, 4, 9]
    assert all(pick.get("nv12") == {2} for pick in picks)


def test_with_no_bgr_reader_no_picture_asks_for_bgr(tmp_path) -> None:
    channel = worker.Channel(tmp_path)
    channel.attach(1, "nv12", 0)
    assert channel.pick(at_fps(30, 1)[0], 0) == {"nv12": {1}}


def test_a_reader_asking_a_negative_rate_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match="every_ns"):
        worker.Channel(tmp_path).attach(1, "nv12", -1)


def test_each_reader_is_sent_its_own_rate_and_a_picture_nobody_takes_is_never_downloaded(tmp_path, monkeypatch):
    # #3218: two readers of a 30 fps camera, at 10 and 5 fps. Every unit is decoded (P pictures are references), but
    # only one some reader's schedule takes is downloaded (keep) and published, and only for the readers that took it.
    headers = at_fps(30, 13)  # 0 to 400 ms

    async def run():
        client = SourceClient()
        kept: list[bool] = []

        def push(header, data, want_bgr=False, keep=True):
            kept.append(keep)
            return [(bytes(24) if keep else b"", 4, 4, header, "limited", "bt601", None)], 0

        native = create_autospec(worker.NativeDecoder, instance=True)
        native.push.side_effect = push
        monkeypatch.setattr(worker, "NativeDecoder", create_autospec(worker.NativeDecoder, return_value=native))
        monkeypatch.setattr(
            worker.continuum_py,
            "decode_header",
            create_autospec(worker.continuum_py.decode_header, side_effect=headers),
        )
        root = tmp_path / "decoded"
        root.mkdir()
        service = worker.DecoderService(cast(Client, client), ["synthetic"], 0, root)
        channel = service.channels["synthetic"]
        channel.attach(10, "nv12", 100_000_000)
        channel.attach(5, "nv12", 200_000_000)
        stop = asyncio.Event()
        running = asyncio.create_task(service.run(tmp_path / "decoded.sock", stop))
        try:
            await asyncio.wait_for(client.subscribed.wait(), 2)
            for _ in headers:
                client.subscription.put(event())
            last = headers[-1]["id"]  # taken by the 10 fps reader, so published once decoded
            async with asyncio.timeout(2):
                while not channel.recent["nv12"] or channel.recent["nv12"][-1][0]["source_header"]["id"] != last:
                    await asyncio.sleep(0)
            held = list(channel.recent["nv12"])
            starts = [(d["source_header"]["start_ns"] - headers[0]["start_ns"]) // 1_000_000 for d, _ in held]
            owner, after = held[0][0]["owner"], held[0][0]["serial"]
            to_10 = await asyncio.wait_for(channel.next("nv12", owner, after, None, 10), 1)
            to_5 = await asyncio.wait_for(channel.next("nv12", owner, after, None, 5), 1)
            readers = [set(r) for _, r in held]
            return (
                kept,
                starts,
                readers,
                (to_10["serial"] - after, to_10["skipped"]),
                (to_5["serial"] - after, to_5["skipped"]),
            )
        finally:
            stop.set()
            await asyncio.wait_for(running, 2)

    kept, starts, readers, to_10, to_5 = asyncio.run(run())
    # 10 fps takes 0, 100, 200, 300, 400 ms; 5 fps takes 0, then 166 (within a quarter period of 200), then 366.
    assert [n for n, keep in enumerate(kept) if keep] == [0, 3, 5, 6, 9, 11, 12]
    assert starts == [200, 300, 366, 400], "the newest four published"
    assert readers == [{10}, {10}, {5}, {10}]
    assert to_10 == (1, 0), "the 10 fps reader's next is the very next picture"
    assert to_5 == (2, 1), "the 5 fps reader's next skips one picture published for the other"


def decoding(monkeypatch, headers, kept):
    """A decoder that decodes each unit to its own picture (pixels only when kept), counting a loss as a break, and
    that records each unit's id and keep; the service reads `headers` from the units in turn."""
    breaks = [0]

    def push(header, data, want_bgr=False, keep=True):
        if data is None:
            breaks[0] += 1
            return [], breaks[0]
        kept.append((header["id"], keep))
        return [(bytes(24) if keep else b"", 4, 4, header, "limited", "bt601", None)], breaks[0]

    native = create_autospec(worker.NativeDecoder, instance=True)
    native.push.side_effect = push
    monkeypatch.setattr(worker, "NativeDecoder", create_autospec(worker.NativeDecoder, return_value=native))
    monkeypatch.setattr(
        worker.continuum_py, "decode_header", create_autospec(worker.continuum_py.decode_header, side_effect=headers)
    )


def test_units_already_waiting_are_decoded_in_one_wakeup_and_published_in_order(tmp_path, monkeypatch):
    # #3235: the camera's thread takes every unit waiting on its subscription at once, not one per wakeup.
    headers = at_fps(30, 5)

    async def run():
        client = SourceClient()
        kept = []
        decoding(monkeypatch, headers, kept)
        root = tmp_path / "decoded"
        root.mkdir()
        service = worker.DecoderService(cast(Client, client), ["synthetic"], 0, root)
        channel = service.channels["synthetic"]
        channel.attach(1, "nv12", 0)
        stop = asyncio.Event()
        running = asyncio.create_task(service.run(tmp_path / "decoded.sock", stop))
        try:
            await asyncio.wait_for(client.subscribed.wait(), 2)
            client.subscription.put(*(event() for _ in headers))
            async with asyncio.timeout(2):
                while (
                    len(channel.recent["nv12"]) < 4
                    or channel.recent["nv12"][-1][0]["source_header"]["id"] != headers[-1]["id"]
                ):
                    await asyncio.sleep(0)
            return kept, [d["source_header"]["id"] for d, _ in channel.recent["nv12"]], client.subscription.batches
        finally:
            stop.set()
            await asyncio.wait_for(running, 2)

    kept, published, batches = asyncio.run(run())
    assert kept == [(header["id"], True) for header in headers], "decoded in order, each kept"
    assert published == [header["id"] for header in headers[1:]], "the newest four held, in order"
    assert batches == [5], "all five taken in one wakeup"


def test_a_loss_in_the_middle_of_a_batch_starts_each_readers_schedule_again_at_the_next_picture(tmp_path, monkeypatch):
    # The loop hears of the break only once the batch is done, so the thread runs the schedules on the decoder's own
    # break count: the picture after the loss is a gap, taken at once though it isn't due at 10 fps.
    headers = at_fps(30, 4)  # 0, 33, 67, and 100 ms; the loss comes between 33 and 67

    async def run():
        client = SourceClient()
        kept = []
        decoding(monkeypatch, headers, kept)
        root = tmp_path / "decoded"
        root.mkdir()
        service = worker.DecoderService(cast(Client, client), ["synthetic"], 0, root)
        service.channels["synthetic"].attach(1, "nv12", 100_000_000)
        stop = asyncio.Event()
        running = asyncio.create_task(service.run(tmp_path / "decoded.sock", stop))
        try:
            await asyncio.wait_for(client.subscribed.wait(), 2)
            client.subscription.put(event(), event(), {"type": "Undecodable"}, event(), event())
            async with asyncio.timeout(2):
                while len(kept) < 4:
                    await asyncio.sleep(0)
            return kept, client.subscription.batches
        finally:
            stop.set()
            await asyncio.wait_for(running, 2)

    kept, batches = asyncio.run(run())
    assert [keep for _, keep in kept] == [True, False, True, False]
    assert batches == [5]


def test_the_decoder_refuses_a_runtime_directory_another_user_owns(tmp_path, monkeypatch) -> None:
    # /dev/shm is shared by every user: a directory there with this install's name but someone else's is never used.
    monkeypatch.setattr(worker, "getuid", lambda: os.getuid() + 1)
    service = worker.DecoderService(cast(Client, object()), [], 0, tmp_path / "decoded")
    with pytest.raises(ValueError, match="belongs to another user"):
        asyncio.run(service.run(tmp_path / "decoded.sock", asyncio.Event()))
    assert not (tmp_path / "decoded").exists(), "nothing is created in it"
