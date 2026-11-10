"""Device choice and slab views as torch tensors (continuum_py.device).

Run with `mise run continuum-test`. The CUDA tests run only where torch sees a GPU (devbox's 3090s) and the MPS ones
only on a Mac; everything else runs everywhere.
"""

import os
import struct
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from continuum_py import Arena
from continuum_py.device import DeviceUnavailable, pick_device, register_arena, to_tensor, unregister_arena

NV12_4K = 3840 * 2160 * 3 // 2
RECORD = struct.Struct("<HxxII4xQQ")
PRODUCER = Path(
    os.environ.get(
        "CONTINUUM_FRAME_PRODUCER",
        Path(__file__).resolve().parents[3] / "target" / "debug" / "examples" / "frame_producer",
    )
)
CUDA = torch.cuda.is_available()
MPS = torch.backends.mps.is_available()


@pytest.fixture
def frame() -> Iterator[tuple[Arena, tuple[int, int, int]]]:
    """An arena holding one published 4K NV12 frame (every byte 0, the frame number), and its handle."""
    name = f"/cnt.d.{os.getpid()}.{time.monotonic_ns() % 10**6}"
    arena = Arena.create(name, classes=[(NV12_4K, 2)], rings=[(RECORD.size, 8)])
    try:
        subprocess.run([PRODUCER, name, "1", "60"], check=True, timeout=10)
        record = arena.ring_reader(0).read()
        assert record is not None
        slab_class, index, generation, _frame, _length = RECORD.unpack(record[1])
        yield arena, (slab_class, index, generation)
    finally:
        Arena.unlink(name)


def test_an_explicit_cpu_setting_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTINUUM_DEVICE", "cuda")
    assert pick_device("cpu") == "cpu", "the argument wins over the environment"


def test_the_environment_setting_is_used_when_no_argument_is_given(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTINUUM_DEVICE", " CPU ")
    assert pick_device() == "cpu"


def test_auto_prefers_cuda_then_mps_then_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONTINUUM_DEVICE", raising=False)
    expected = "cuda" if CUDA else "mps" if MPS else "cpu"
    assert pick_device() == expected
    assert pick_device("auto") == expected


@pytest.mark.skipif(not CUDA, reason="needs a CUDA device")
def test_a_device_index_is_normalized() -> None:
    assert pick_device("cuda:00") == "cuda:0"


@pytest.mark.parametrize("setting", ["tpu", "cuda:x", "gpu", "cuda:-1"])
def test_an_unknown_setting_is_refused(setting: str) -> None:
    with pytest.raises(ValueError, match="device must be"):
        pick_device(setting)


def test_a_device_this_machine_lacks_is_refused_not_swapped() -> None:
    with pytest.raises(DeviceUnavailable):
        pick_device("cuda:99")
    if not MPS:
        with pytest.raises(DeviceUnavailable, match="mps"):
            pick_device("mps")


def test_a_frame_on_the_cpu_is_the_slab_itself_and_dropping_it_releases_the_pin(
    frame: tuple[Arena, tuple[int, int, int]],
) -> None:
    arena, handle = frame
    lease = arena.lease(*handle)
    luma = to_tensor(lease, "cpu", shape=(3240, 3840))
    assert luma.data_ptr() == lease.address, "no copy: the tensor reads the slot in place"
    assert lease.address % 16384 == 0
    assert luma.shape == (3240, 3840)
    assert int(luma[100, 100]) == 0  # frame 0's fill byte
    lease.release()
    assert arena.pin_count(*handle) == 1, "the tensor still holds the pin"
    del luma
    assert arena.pin_count(*handle) == 0, "and releases it when it goes"


def test_a_dtype_reinterprets_the_bytes_without_copying(frame: tuple[Arena, tuple[int, int, int]]) -> None:
    arena, handle = frame
    with arena.lease(*handle) as lease:
        words = to_tensor(lease, "cpu", dtype=torch.int64)
        assert words.data_ptr() == lease.address
        assert int(words[0]) == 0  # the first 8 bytes hold the frame number
        assert words.numel() == NV12_4K // 8
        del words


@pytest.mark.skipif(not CUDA, reason="needs a CUDA device")
def test_a_frame_on_cuda_is_a_copy_with_the_same_bytes_and_frees_the_slot(
    frame: tuple[Arena, tuple[int, int, int]],
) -> None:
    arena, handle = frame
    with arena.lease(*handle) as lease:
        on_gpu = to_tensor(lease, "cuda", shape=(NV12_4K,))
    assert arena.pin_count(*handle) == 0, "the copy doesn't need the slot"
    assert on_gpu.device.type == "cuda"
    assert int(on_gpu.sum()) == 0  # frame 0: every byte zero
    del on_gpu


@pytest.mark.skipif(not CUDA, reason="needs a CUDA device")
def test_an_async_copy_keeps_the_slot_pinned_until_the_gpu_tensor_goes(
    frame: tuple[Arena, tuple[int, int, int]],
) -> None:
    arena, handle = frame
    register_arena(arena)
    try:
        lease = arena.lease(*handle)
        on_gpu = to_tensor(lease, "cuda", non_blocking=True)
        lease.release()  # the caller lets go while the DMA may still be in flight
        assert arena.pin_count(*handle) == 1, "the GPU tensor holds the pin for the copy"
        torch.cuda.current_stream().synchronize()
        assert int(on_gpu[:8].sum()) == 0
        del on_gpu
        assert arena.pin_count(*handle) == 0
    finally:
        unregister_arena(arena)


@pytest.mark.skipif(not CUDA, reason="needs a CUDA device")
def test_registering_twice_is_refused_before_cuda_sees_it(frame: tuple[Arena, tuple[int, int, int]]) -> None:
    arena, _ = frame
    register_arena(arena)
    try:
        with pytest.raises(ValueError, match="already registered"):
            register_arena(arena)
        # CUDA was never asked, so it holds no error to fail the next, unrelated launch with.
        assert float((torch.ones(4, device="cuda") * 2).sum()) == 8.0
    finally:
        unregister_arena(arena)
    with pytest.raises(ValueError, match="isn't registered"):
        unregister_arena(arena)


@pytest.mark.skipif(not CUDA, reason="needs a CUDA device")
def test_a_registered_arena_stays_mapped_until_unregistered(frame: tuple[Arena, tuple[int, int, int]]) -> None:
    import gc

    from continuum_py import device

    arena, _ = frame
    base = arena.base_address
    register_arena(arena)
    try:
        del arena
        gc.collect()
        assert base in device._registered, "the registry holds the arena, so its mapping can't go while registered"
    finally:
        unregister_arena(device._registered[base])


def test_registering_without_cuda_is_device_unavailable(
    monkeypatch: pytest.MonkeyPatch, frame: tuple[Arena, tuple[int, int, int]]
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(DeviceUnavailable, match="CUDA"):
        register_arena(frame[0])


@pytest.mark.skipif(not MPS, reason="needs Metal (Apple Silicon)")
def test_a_frame_on_mps_is_a_copy_with_the_same_bytes(frame: tuple[Arena, tuple[int, int, int]]) -> None:
    arena, handle = frame
    with arena.lease(*handle) as lease:
        on_gpu = to_tensor(lease, "mps", shape=(3240, 3840))
    assert on_gpu.device.type == "mps"
    assert int(on_gpu[100, 100]) == 0
