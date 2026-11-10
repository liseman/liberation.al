"""continuum_py against a real arena: a Rust producer writes 4K NV12 frames, and Python reads them in place.

Run with `mise run continuum-test`, which builds the extension and the producer first. The producer binary comes from
CONTINUUM_FRAME_PRODUCER, or continuum/target/debug/examples/frame_producer.
"""

import os
import struct
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import continuum_py
import numpy as np
import pytest
from continuum_py import Arena, Lease, StaleError

NV12_4K = 3840 * 2160 * 3 // 2
SLOTS = 16  # about 266 ms of 60 fps frames, slack for a consumer on a loaded machine
RECORD = struct.Struct("<HxxII4xQQ")  # the producer's ring record: handle, frame number, length
END = 2**64 - 1
FRAMES = 120  # two seconds at 60 fps
FPS = 60
PRODUCER = Path(
    os.environ.get(
        "CONTINUUM_FRAME_PRODUCER",
        Path(__file__).resolve().parents[3] / "target" / "debug" / "examples" / "frame_producer",
    )
)


@pytest.fixture
def arena_name() -> Iterator[str]:
    name = f"/cnt.py.{os.getpid()}.{time.monotonic_ns() % 10**6}"
    yield name
    try:
        Arena.unlink(name)
    except FileNotFoundError:
        pass


@pytest.fixture
def frame_arena(arena_name: str) -> Arena:
    return Arena.create(arena_name, classes=[(NV12_4K, SLOTS)], rings=[(RECORD.size, 256)])


@dataclass
class Consumed:
    frames: list[int] = field(default_factory=list)
    stale: int = 0
    misaligned: list[int] = field(default_factory=list)
    copied: list[int] = field(default_factory=list)
    wrong_bytes: list[int] = field(default_factory=list)


def check_frame(lease: Lease, frame: int, out: Consumed) -> None:
    """Reads one leased frame three ways, all in place, and records anything off."""
    if lease.offset % continuum_py.PAGE_ALIGN or lease.address % continuum_py.PAGE_ALIGN:
        out.misaligned.append(frame)
    view = memoryview(lease)
    array = lease.numpy()
    tensor = np.from_dlpack(lease)
    if array.ctypes.data != lease.address or tensor.ctypes.data != lease.address or array.flags.owndata:
        out.copied.append(frame)
    expected = frame & 0xFF
    if (
        int.from_bytes(view[:8], "little") != frame
        or array[8::4099].min() != expected
        or array[8::4099].max() != expected
        or tensor[-1] != expected
    ):
        out.wrong_bytes.append(frame)
    view.release()
    del array, tensor
    out.frames.append(frame)


def consume(arena: Arena, producer: subprocess.Popen[bytes], timeout: float) -> Consumed:
    """Follows ring 0 until the end record, leasing and checking every frame it names."""
    out = Consumed()
    ring = arena.ring_reader(0)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = ring.read()
        if record is None:
            if producer.poll() not in (None, 0):
                break
            ring.wait(timeout_ms=100)  # woken by the producer's next push, across processes
            continue
        slab_class, index, generation, frame, _length = RECORD.unpack(record[1])
        if frame == END:
            break
        try:
            lease = arena.lease(slab_class, index, generation)
        except StaleError:
            out.stale += 1
            continue
        with lease:
            check_frame(lease, frame, out)
    return out


def test_python_reads_rust_4k_frames_in_place_at_60_fps(frame_arena: Arena) -> None:
    producer = subprocess.Popen([PRODUCER, frame_arena.name, str(FRAMES), str(FPS)])
    try:
        got = consume(frame_arena, producer, timeout=10.0)
        assert producer.wait(timeout=5) == 0
    finally:
        producer.kill()
    assert (got.misaligned, got.copied, got.wrong_bytes) == ([], [], [])
    assert got.frames == sorted(got.frames)
    assert len(got.frames) + got.stale == FRAMES
    assert len(got.frames) >= FRAMES * 0.9, f"read {len(got.frames)} of {FRAMES} frames, {got.stale} stale"


@pytest.fixture
def one_frame(frame_arena: Arena) -> tuple[Arena, tuple[int, int, int]]:
    """An arena holding one published frame (number 0), and its handle."""
    subprocess.run([PRODUCER, frame_arena.name, "1", str(FPS)], check=True, timeout=10)
    record = frame_arena.ring_reader(0).read()
    assert record is not None
    slab_class, index, generation, _frame, _length = RECORD.unpack(record[1])
    return frame_arena, (slab_class, index, generation)


def test_a_lease_pins_the_slot_until_released(one_frame: tuple[Arena, tuple[int, int, int]]) -> None:
    arena, handle = one_frame
    lease = arena.lease(*handle)
    assert arena.pin_count(*handle) == 1
    assert lease.release() is True
    assert arena.pin_count(*handle) == 0
    assert lease.is_valid() is False


def test_release_refuses_while_a_memoryview_is_open(one_frame: tuple[Arena, tuple[int, int, int]]) -> None:
    arena, handle = one_frame
    lease = arena.lease(*handle)
    view = memoryview(lease)
    with pytest.raises(BufferError, match="still uses"):
        lease.release()
    view.release()
    assert lease.release() is True


def test_leased_bytes_are_read_only(one_frame: tuple[Arena, tuple[int, int, int]]) -> None:
    arena, handle = one_frame
    with arena.lease(*handle) as lease:
        assert memoryview(lease).readonly
        assert not lease.numpy().flags.writeable
        assert not np.from_dlpack(lease).flags.writeable


def test_a_dlpack_array_keeps_the_pin_after_release(one_frame: tuple[Arena, tuple[int, int, int]]) -> None:
    arena, handle = one_frame
    lease = arena.lease(*handle)
    tensor = np.from_dlpack(lease)
    lease.release()
    assert arena.pin_count(*handle) == 1, "the capsule's array still reads the slot"
    del tensor
    assert arena.pin_count(*handle) == 0


def test_leaving_a_with_block_while_an_array_still_reads_the_slot_raises(
    one_frame: tuple[Arena, tuple[int, int, int]],
) -> None:
    arena, handle = one_frame
    with pytest.raises(BufferError, match="still uses"), arena.lease(*handle) as lease:
        array = lease.numpy()
    assert arena.pin_count(*handle) == 1, "the array still reads the slot"
    del array, lease
    assert arena.pin_count(*handle) == 0, "the pin goes with the last view"


def test_an_exception_in_the_with_block_is_not_masked_by_an_open_view(
    one_frame: tuple[Arena, tuple[int, int, int]],
) -> None:
    arena, handle = one_frame
    with pytest.raises(KeyError), arena.lease(*handle) as lease:
        array = lease.numpy()
        raise KeyError("from the block")
    del array, lease
    assert arena.pin_count(*handle) == 0


def test_numpy_views_take_a_dtype_and_shape(one_frame: tuple[Arena, tuple[int, int, int]]) -> None:
    arena, handle = one_frame
    with arena.lease(*handle) as lease:
        luma = lease.numpy(shape=[3240, 3840])
        words = lease.numpy("<u8")
        assert luma.shape == (3240, 3840)
        assert words[0] == 0  # frame number 0 in the first 8 bytes
        del luma, words


def test_a_handle_to_an_unpublished_slot_is_stale(frame_arena: Arena) -> None:
    with pytest.raises(StaleError):
        frame_arena.lease(0, 0, 1)
    assert frame_arena.pin_count(0, 0, 1) == 0


def test_opening_a_missing_arena_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        Arena.open("/cnt.no.such.arena")


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("no-slash", id="no-leading-slash"),
        pytest.param("/a/b", id="inner-slash"),
        pytest.param("/" + "x" * 31, id="over-macos-31-bytes"),
    ],
)
def test_names_must_be_portable(name: str) -> None:
    with pytest.raises(ValueError, match="31 bytes"):
        Arena.create(name, classes=[(1, 1)])


WAIT_WHILE_READING = """
import sys, threading
from continuum_py import Arena
arena = Arena.create(sys.argv[1], classes=[(1, 1)], rings=[(8, 4)])
ring = arena.ring_reader(0)
waiter = threading.Thread(target=ring.wait, args=(300,))
waiter.start()
for _ in range(1000):
    ring.read()  # takes the reader while the other thread waits on it
waiter.join()
Arena.unlink(sys.argv[1])
"""


def test_a_blocked_reader_is_woken_by_a_write_from_another_process(frame_arena: Arena) -> None:
    ring = frame_arena.ring_reader(0, from_oldest=False)
    producer = subprocess.Popen([PRODUCER, frame_arena.name, "1", str(FPS)])
    try:
        start = time.monotonic()
        woke = ring.wait(timeout_ms=5000)
        waited = time.monotonic() - start
        assert producer.wait(timeout=5) == 0
    finally:
        producer.kill()
    # A missed wake would sit out the full 5 s; process start and one 4K frame take well under 2 s.
    assert woke
    assert waited < 2.0
    assert ring.read() is not None


def test_reading_a_ring_while_another_thread_waits_on_it_does_not_deadlock(arena_name: str) -> None:
    # A deadlock would hang forever, so it runs in a child process with a timeout.
    done = subprocess.run([sys.executable, "-c", WAIT_WHILE_READING, arena_name], timeout=10, check=False)
    assert done.returncode == 0


def test_a_written_slot_reads_back_and_a_full_class_or_an_oversized_write_is_refused(arena_name: str) -> None:
    arena = Arena.create(arena_name, classes=[(16_384, 2)])
    index, generation, offset = arena.write(0, b"jpeg bytes")
    assert offset % 16_384 == 0
    with arena.lease(0, index, generation) as lease:
        assert bytes(memoryview(lease)) == b"jpeg bytes"
    arena.write(0, b"second")
    with pytest.raises(continuum_py.FullError):
        arena.write(0, b"third")
    with pytest.raises(ValueError, match="don't fit"):
        Arena.create(arena_name + "x", classes=[(16_384, 1)]).write(0, bytes(16_385))
    Arena.unlink(arena_name + "x")


def test_a_numpy_array_writes_in_place_and_a_large_one_too(arena_name: str) -> None:
    arena = Arena.create(arena_name, classes=[(16_384, 1), (1 << 20, 1)])
    small = np.arange(256, dtype=np.uint8)
    index, generation, _ = arena.write(0, small)
    with arena.lease(0, index, generation) as lease:
        assert bytes(memoryview(lease)) == small.tobytes()
    big = np.full(200_000, 7, np.uint8)  # over 64 KiB: copied with the GIL released
    index, generation, _ = arena.write(1, memoryview(big))
    with arena.lease(1, index, generation) as lease:
        assert lease.numpy().size == 200_000 and int(lease.numpy()[-1]) == 7


def test_a_non_contiguous_buffer_is_refused(arena_name: str) -> None:
    arena = Arena.create(arena_name, classes=[(16_384, 1)])
    with pytest.raises(ValueError, match="contiguous"):
        arena.write(0, np.arange(100, dtype=np.uint8)[::2])
