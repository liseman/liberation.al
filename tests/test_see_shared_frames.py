"""Real socket/cache handoff keeps native identity and reports discontinuities."""

import asyncio
import json
import os
import time
from contextlib import aclosing

import numpy as np
import pytest

from liberational.see import shared_frames
from liberational.see.decoded_cache import DecodedCache
from liberational.see.shared_frames import decoded_frames


def source(number):
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


@pytest.mark.parametrize("discontinuity", ["serial", "owner", "break"])
def test_socket_client_marks_serial_owner_and_decoder_breaks(tmp_path, discontinuity):
    async def run():
        root = tmp_path / "decoded"
        root.mkdir()
        pixels = np.full((6, 4), 128, np.uint8)
        first_owner = DecodedCache(root, format="nv12", max_pixels=16)
        second_owner = DecodedCache(root, format="nv12", max_pixels=16)
        first = first_owner.publish(pixels, source(1))
        second = first_owner.publish(pixels, source(2))
        assert first is not None
        assert second is not None
        if discontinuity == "serial":
            first_owner.publish(pixels, source(3))
            third = first_owner.publish(pixels, source(4))
        elif discontinuity == "owner":
            third = second_owner.publish(pixels, source(3))
        else:
            third = first_owner.publish(pixels, source(3), break_count=1)
        assert third is not None
        descriptors = [first, second, third]
        requests = []

        async def connection(reader, writer):
            try:
                for descriptor in descriptors:
                    requests.append(json.loads(await reader.readline()))
                    writer.write(json.dumps(descriptor).encode() + b"\n")
                    await writer.drain()
                await reader.read()
            finally:
                writer.close()
                await writer.wait_closed()

        address = tmp_path / "decoded.sock"
        server = await asyncio.start_unix_server(connection, str(address))
        try:
            async with aclosing(decoded_frames("synthetic", socket=address)) as frames:
                a = await asyncio.wait_for(anext(frames), 2)
                b = await asyncio.wait_for(anext(frames), 2)
                c = await asyncio.wait_for(anext(frames), 2)
                assert (a.gap_before, b.gap_before, c.gap_before) == (True, False, True)
                assert (a.id, a.capture_ns) == (source(1)["id"], 1800000000000000001)
                assert b.source_header == source(2)
                np.testing.assert_array_equal(c.pixels, pixels)
                assert requests[1]["after"] == 1
                assert requests[1]["owner"] == first["owner"]
        finally:
            server.close()
            await server.wait_closed()
            first_owner.close()
            second_owner.close()

    asyncio.run(run())


def test_overwritten_generation_is_skipped_and_next_picture_is_a_gap(tmp_path):
    async def run():
        root = tmp_path / "decoded"
        root.mkdir()
        cache = DecodedCache(root, format="nv12", max_pixels=16)
        pixels = np.zeros((6, 4), np.uint8)
        stale = cache.publish(pixels, source(1))
        cache.publish(pixels, source(2))
        cache.publish(pixels, source(3))
        cache.publish(pixels, source(4))
        current = cache.publish(np.full((6, 4), 9, np.uint8), source(5))

        async def connection(reader, writer):
            try:
                await reader.readline()
                writer.write(json.dumps(stale).encode() + b"\n")
                await writer.drain()
                request = json.loads(await reader.readline())
                assert request["after"] == 1
                writer.write(json.dumps(current).encode() + b"\n")
                await writer.drain()
                await reader.read()
            finally:
                writer.close()
                await writer.wait_closed()

        address = tmp_path / "decoded.sock"
        server = await asyncio.start_unix_server(connection, str(address))
        try:
            async with aclosing(decoded_frames("synthetic", socket=address)) as frames:
                frame = await asyncio.wait_for(anext(frames), 2)
                assert frame.id == source(5)["id"]
                assert frame.gap_before
                np.testing.assert_array_equal(frame.pixels, np.full((6, 4), 9, np.uint8))
        finally:
            server.close()
            await server.wait_closed()
            cache.close()

    asyncio.run(run())


def test_standalone_socket_gap_calls_back_before_next_picture_without_yielding_a_frame(tmp_path):
    async def run():
        root = tmp_path / "decoded"
        root.mkdir()
        pixels = np.zeros((6, 4), np.uint8)
        with DecodedCache(root, format="nv12", max_pixels=16) as cache:
            first = cache.publish(pixels, source(1))
            second = cache.publish(pixels, source(2), break_count=1)
            assert first is not None and second is not None
            gap_called = asyncio.Event()
            waiting_for_picture = asyncio.Event()
            release_picture = asyncio.Event()
            disconnected = asyncio.Event()
            requests = []

            async def connection(reader, writer):
                try:
                    requests.append(json.loads(await reader.readline()))
                    writer.write(json.dumps(first).encode() + b"\n")
                    await writer.drain()
                    requests.append(json.loads(await reader.readline()))
                    writer.write(b'{"type":"DecodedGap","break_count":1}\n')
                    await writer.drain()
                    requests.append(json.loads(await reader.readline()))
                    waiting_for_picture.set()
                    await release_picture.wait()
                    writer.write(json.dumps(second).encode() + b"\n")
                    await writer.drain()
                    await reader.read()
                finally:
                    writer.close()
                    await writer.wait_closed()
                    disconnected.set()

            address = tmp_path / "decoded.sock"
            server = await asyncio.start_unix_server(connection, str(address))
            breaks = []

            def on_break():
                breaks.append("gap")
                gap_called.set()

            try:
                async with aclosing(decoded_frames("synthetic", socket=address, on_break=on_break)) as frames:
                    assert (await asyncio.wait_for(anext(frames), 2)).id == source(1)["id"]
                    next_frame = asyncio.create_task(anext(frames))
                    try:
                        await asyncio.wait_for(gap_called.wait(), 2)
                        await asyncio.wait_for(waiting_for_picture.wait(), 2)
                        assert breaks == ["gap"]
                        assert not next_frame.done(), "a standalone gap must not yield a phantom picture"
                        assert requests[1]["after"] == requests[2]["after"] == 1
                        assert requests[1]["break_count"] == 0
                        assert requests[2]["break_count"] == 1
                        release_picture.set()
                        frame = await asyncio.wait_for(next_frame, 2)
                        assert frame.id == source(2)["id"] and frame.gap_before
                        assert breaks == ["gap"]
                    finally:
                        release_picture.set()
                        if not next_frame.done():
                            next_frame.cancel()
                            await asyncio.gather(next_frame, return_exceptions=True)
                await asyncio.wait_for(disconnected.wait(), 2)
            finally:
                release_picture.set()
                server.close()
                await server.wait_closed()

    asyncio.run(run())


def test_shared_reader_waits_for_decoder_startup_race(monkeypatch):
    from pathlib import Path
    from unittest.mock import create_autospec

    from liberational.see import shared_frames

    async def run():
        expected = (object(), object())
        connect = create_autospec(asyncio.open_unix_connection, side_effect=[FileNotFoundError(), expected])
        pause = create_autospec(asyncio.sleep, return_value=None)
        monkeypatch.setattr(shared_frames.asyncio, "open_unix_connection", connect)
        monkeypatch.setattr(shared_frames.asyncio, "sleep", pause)
        assert await shared_frames._connect(Path("fixture.sock")) == expected
        assert connect.await_count == 2

    asyncio.run(run())


def test_cameras_waiting_for_the_decoder_say_so_once_until_it_connects(monkeypatch, caplog):
    from pathlib import Path
    from unittest.mock import create_autospec

    from liberational.see import shared_frames

    async def run():
        expected = (object(), object())
        down = [FileNotFoundError(), ConnectionRefusedError(), FileNotFoundError(), ConnectionRefusedError()]
        connect = create_autospec(
            asyncio.open_unix_connection, side_effect=[*down, expected, expected, FileNotFoundError(), expected]
        )
        real_sleep = asyncio.sleep

        async def pause(_seconds: float) -> None:
            await real_sleep(0)  # yield, so the two waiters take turns

        monkeypatch.setattr(shared_frames.asyncio, "open_unix_connection", connect)
        monkeypatch.setattr(shared_frames.asyncio, "sleep", pause)
        monkeypatch.setattr(shared_frames, "_waiting_said", False)
        # Two cameras wait through the same outage: one line, not one each (#3147).
        assert await asyncio.gather(*(shared_frames._connect(Path("fixture.sock")) for _ in range(2))) == [
            expected,
            expected,
        ]
        assert sum("waiting for the shared camera decoder" in r.getMessage() for r in caplog.records) == 1
        # Once connected, a later outage says so again.
        assert await shared_frames._connect(Path("fixture.sock")) == expected
        assert sum("waiting for the shared camera decoder" in r.getMessage() for r in caplog.records) == 2

    caplog.set_level("WARNING")
    asyncio.run(run())


def test_shared_reader_startup_wait_has_a_deadline(monkeypatch):
    from pathlib import Path
    from unittest.mock import create_autospec

    from liberational.see import shared_frames

    async def run():
        connect = create_autospec(asyncio.open_unix_connection, side_effect=ConnectionRefusedError())
        monkeypatch.setattr(shared_frames.asyncio, "open_unix_connection", connect)
        monkeypatch.setattr(shared_frames, "STARTUP_WAIT_S", 0.0)
        with pytest.raises(TimeoutError, match="startup deadline"):
            await shared_frames._connect(Path("fixture.sock"))
        assert connect.await_count == 1

    asyncio.run(run())


def test_reader_waits_through_a_quiet_camera_and_reconnects_to_a_restarted_decoder(tmp_path, monkeypatch):
    from liberational.see import shared_frames

    monkeypatch.setattr(shared_frames, "RECONNECT_S", 0.0)

    async def run():
        root = tmp_path / "decoded"
        root.mkdir()
        pixels = np.zeros((6, 4), np.uint8)
        first_owner = DecodedCache(root, format="nv12", max_pixels=16)
        second_owner = DecodedCache(root, format="nv12", max_pixels=16)
        first = first_owner.publish(pixels, source(1))
        second = second_owner.publish(pixels, source(2))
        assert first is not None and second is not None
        connections = []

        async def connection(reader, writer):
            connections.append(json.loads(await reader.readline()))
            try:
                if len(connections) == 1:
                    # No picture yet, then the first one; then the decoder goes away mid-request.
                    writer.write(b'{"type":"DecodedWaiting"}\n')
                    await writer.drain()
                    await reader.readline()
                    writer.write(json.dumps(first).encode() + b"\n")
                    await writer.drain()
                    await reader.readline()
                else:
                    writer.write(json.dumps(second).encode() + b"\n")
                    await writer.drain()
                    await reader.read()
            finally:
                writer.close()
                await writer.wait_closed()

        address = tmp_path / "decoded.sock"
        server = await asyncio.start_unix_server(connection, str(address))
        breaks = []
        try:
            frames_iter = decoded_frames("synthetic", socket=address, on_break=lambda: breaks.append("gap"))
            async with aclosing(frames_iter) as frames:
                a = await asyncio.wait_for(anext(frames), 2)
                b = await asyncio.wait_for(anext(frames), 2)
            assert (a.id, b.id) == (source(1)["id"], source(2)["id"])
            assert b.gap_before, "a reconnect marks the next picture as a gap"
            assert breaks == ["gap"]
            assert connections[1]["after"] == 0 and connections[1]["owner"] is None
        finally:
            server.close()
            await server.wait_closed()
            first_owner.close()
            second_owner.close()

    asyncio.run(run())


def test_a_refusal_with_no_message_still_names_the_camera(tmp_path):
    async def run():
        async def connection(reader, writer):
            await reader.readline()
            writer.write(b'{"error":""}\n')
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        (tmp_path / "decoded").mkdir()
        address = tmp_path / "decoded.sock"
        server = await asyncio.start_unix_server(connection, str(address))
        try:
            async with aclosing(decoded_frames("synthetic", socket=address)) as frames:
                with pytest.raises(RuntimeError, match="refused synthetic: no reason"):
                    await asyncio.wait_for(anext(frames), 2)
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(run())


def test_a_configured_decoded_path_holds_the_socket_and_its_ring(tmp_path) -> None:
    with shared_frames.cfg.override({"paths": {"decoded": str(tmp_path / "ring")}}):
        assert shared_frames.socket_path() == tmp_path / "ring" / "decoded.sock"


def test_by_default_the_ring_lives_in_memory_per_user_and_per_install(tmp_path, monkeypatch) -> None:
    shm = tmp_path / "shm"
    shm.mkdir()
    monkeypatch.setattr(shared_frames, "SHM", shm)
    with shared_frames.cfg.override({"paths": {"decoded": "", "state": str(tmp_path / "a")}}):
        first = shared_frames.runtime_dir()
    with shared_frames.cfg.override({"paths": {"decoded": "", "state": str(tmp_path / "b")}}):
        second = shared_frames.runtime_dir()
    assert first.parent == shm
    assert first.name.startswith(f"liberational-{os.getuid()}-")
    assert first != second, "two installs on one machine never share a ring"


def test_without_dev_shm_the_ring_goes_in_the_temp_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(shared_frames, "SHM", tmp_path / "missing")
    monkeypatch.setattr(shared_frames.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))
    with shared_frames.cfg.override({"paths": {"decoded": ""}}):
        assert shared_frames.runtime_dir().parent == tmp_path / "tmp"


def _serve_then_go_quiet(descriptors):
    """A decoder that answers with these pictures, then has nothing newer (it holds the next request open)."""

    async def connection(reader, writer):
        try:
            for descriptor in descriptors:
                await reader.readline()
                writer.write(json.dumps(descriptor).encode() + b"\n")
                await writer.drain()
            await reader.read()
        finally:
            writer.close()

    return connection


def _bgr_pictures(tmp_path, starts_ns):
    root = tmp_path / "decoded"
    root.mkdir()
    cache = DecodedCache(root, format="bgr", max_pixels=16)
    pixels = [np.full((4, 4, 3), n, np.uint8) for n in range(len(starts_ns))]
    descriptors = [
        cache.publish(p, dict(source(n + 1), start_ns=start)) for n, (p, start) in enumerate(zip(pixels, starts_ns))
    ]
    assert all(d is not None for d in descriptors)
    return cache, descriptors, pixels


@pytest.mark.parametrize(("age_s", "found"), [(1.0, True), (60.0, False)], ids=["recent", "too-old"])
def test_latest_reads_on_to_the_newest_picture_and_only_a_recent_one_counts(tmp_path, age_s, found):
    async def run():
        now = time.time_ns()
        starts = [now - int((age_s + 2) * 1e9), now - int((age_s + 1) * 1e9), now - int(age_s * 1e9)]
        cache, descriptors, pixels = _bgr_pictures(tmp_path, starts)
        address = tmp_path / "decoded.sock"
        server = await asyncio.start_unix_server(_serve_then_go_quiet(descriptors), str(address))
        try:
            got = await shared_frames.latest("synthetic", within_s=30.0, socket=address)
        finally:
            server.close()
            await server.wait_closed()
            cache.close()
        if found:
            assert got is not None and got.capture_ns == starts[2]
            np.testing.assert_array_equal(got.pixels, pixels[2])
        else:
            assert got is None

    asyncio.run(run())


def test_latest_gives_none_quickly_when_no_decoder_answers(tmp_path):
    async def run():
        loop = asyncio.get_running_loop()
        began = loop.time()
        assert await shared_frames.latest("synthetic", within_s=30.0, socket=tmp_path / "none.sock", wait_s=0.3) is None
        assert loop.time() - began < 2.0  # bounded by wait_s, not the 60 s startup wait

    asyncio.run(run())


@pytest.mark.parametrize(
    ("skipped", "gap"),
    [pytest.param(1, False, id="passed-over-for-another-reader"), pytest.param(0, True, id="lost")],
)
def test_a_serial_jump_is_a_gap_unless_the_decoder_passed_those_pictures_over_for_other_readers(tmp_path, skipped, gap):
    # #3218: the decoder applies each reader's rate, so a reader's pictures skip serials published for others.
    requests = []

    async def run():
        root = tmp_path / "decoded"
        root.mkdir()
        pixels = np.full((6, 4), 128, np.uint8)
        cache = DecodedCache(root, format="nv12", max_pixels=16)
        first, _, third = (cache.publish(pixels, source(n)) for n in (1, 2, 3))
        assert first is not None and third is not None
        replies = [{**first, "skipped": 0}, {**third, "skipped": skipped}]

        async def connection(reader, writer):
            try:
                for reply in replies:
                    requests.append(json.loads(await reader.readline()))
                    writer.write(json.dumps(reply).encode() + b"\n")
                    await writer.drain()
                await reader.read()
            finally:
                writer.close()
                await writer.wait_closed()

        address = tmp_path / "decoded.sock"
        server = await asyncio.start_unix_server(connection, str(address))
        try:
            async with aclosing(decoded_frames("synthetic", socket=address, every_ns=100_000_000)) as frames:
                a = await asyncio.wait_for(anext(frames), 2)
                b = await asyncio.wait_for(anext(frames), 2)
                return (a.id, a.gap_before), (b.id, b.gap_before)
        finally:
            server.close()
            await server.wait_closed()
            cache.close()

    a, b = asyncio.run(run())
    assert a == (source(1)["id"], True)
    assert b == (source(3)["id"], gap)
    assert [request["every_ns"] for request in requests] == [100_000_000, 100_000_000]
