"""NVIDIA GPUs' memory, utilization, and compute processes from NVML in this process, for the Resource monitor
(dashboard/machine_usage.py) and model status (model_runtime.py).

Each nvidia-smi call costs 30 to 60 ms of CPU (a new process, and NVML's own start-up), and the dashboard made three of
them every 5 s. NVML's library calls (libnvidia-ml, which nvidia-smi itself uses, installed with the driver) cost well
under a millisecond. `gpus()` is None when NVML can't be loaded (a Mac, no driver) or a call fails; callers then fall
back to nvidia-smi.
"""

import ctypes
import threading
from typing import Any

NOT_AVAILABLE = 2**64 - 1  # NVML_VALUE_NOT_AVAILABLE: a process whose memory the driver won't say
INSUFFICIENT_SIZE = 7  # NVML_ERROR_INSUFFICIENT_SIZE: more processes than the array holds
TEXT = 96  # NVML_DEVICE_UUID_V2_BUFFER_SIZE and NVML_DEVICE_NAME_V2_BUFFER_SIZE


class _Memory(ctypes.Structure):  # nvmlMemory_t
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


class _Memory2(
    ctypes.Structure
):  # nvmlMemory_v2_t: "used" without the driver's reserved memory, as nvidia-smi shows it
    _fields_ = [
        ("version", ctypes.c_uint),
        ("total", ctypes.c_ulonglong),
        ("reserved", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class _Utilization(ctypes.Structure):  # nvmlUtilization_t
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _Process(ctypes.Structure):  # nvmlProcessInfo_t, as nvmlDeviceGetComputeRunningProcesses_v3 fills it
    _fields_ = [
        ("pid", ctypes.c_uint),
        ("usedGpuMemory", ctypes.c_ulonglong),
        ("gpuInstanceId", ctypes.c_uint),
        ("computeInstanceId", ctypes.c_uint),
    ]


class _Failed(Exception):
    pass


_lock = threading.Lock()
_lib: Any = None
_tried = False


def _library() -> Any:
    """libnvidia-ml, initialized once; None when it isn't there or won't start."""
    global _lib, _tried
    if not _tried:
        _tried = True
        try:
            lib = ctypes.CDLL("libnvidia-ml.so.1")
            if lib.nvmlInit_v2() == 0:
                _lib = lib
        except OSError, AttributeError:
            _lib = None
    return _lib


def _check(code: int) -> None:
    if code != 0:
        raise _Failed(code)


def _memory(lib: Any, handle: ctypes.c_void_p) -> tuple[int, int]:
    """(total, used) bytes: the v2 call when the driver has it, else v1."""
    try:
        m2 = _Memory2(version=ctypes.sizeof(_Memory2) | (2 << 24))
        if lib.nvmlDeviceGetMemoryInfo_v2(handle, ctypes.byref(m2)) == 0:
            return int(m2.total), int(m2.used)
    except AttributeError:
        pass
    m = _Memory()
    _check(lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(m)))
    return int(m.total), int(m.used)


def _processes(lib: Any, handle: ctypes.c_void_p) -> dict[int, int]:
    """Each compute process's GPU memory on this device, by pid; one whose memory the driver won't say is left out."""
    size = 64
    while True:
        count = ctypes.c_uint(size)
        found = (_Process * size)()
        code = lib.nvmlDeviceGetComputeRunningProcesses_v3(handle, ctypes.byref(count), found)
        if code == INSUFFICIENT_SIZE and size < 4096:
            size = max(size * 2, int(count.value) + 8)
            continue
        _check(code)
        apps: dict[int, int] = {}
        for p in found[: count.value]:
            if int(p.usedGpuMemory) != NOT_AVAILABLE:
                apps[int(p.pid)] = apps.get(int(p.pid), 0) + int(p.usedGpuMemory)
        return apps


def gpus() -> list[dict[str, Any]] | None:
    """Each GPU: {"index", "uuid", "name", "total", "used" (bytes), "util" (% or None), "apps": {pid: bytes}}; None
    without NVML or when it fails."""
    lib = _library()
    if lib is None:
        return None
    try:
        with _lock:
            count = ctypes.c_uint()
            _check(lib.nvmlDeviceGetCount_v2(ctypes.byref(count)))
            out = []
            for index in range(count.value):
                handle = ctypes.c_void_p()
                _check(lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(index), ctypes.byref(handle)))
                name, uuid = ctypes.create_string_buffer(TEXT), ctypes.create_string_buffer(TEXT)
                _check(lib.nvmlDeviceGetName(handle, name, ctypes.c_uint(TEXT)))
                _check(lib.nvmlDeviceGetUUID(handle, uuid, ctypes.c_uint(TEXT)))
                total, used = _memory(lib, handle)
                util = _Utilization()
                busy = int(util.gpu) if lib.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(util)) == 0 else None
                out.append(
                    {
                        "index": index,
                        "uuid": uuid.value.decode(errors="replace"),
                        "name": name.value.decode(errors="replace"),
                        "total": total,
                        "used": used,
                        "util": busy,
                        "apps": _processes(lib, handle),
                    }
                )
            return out
    except _Failed, AttributeError, OSError:
        return None
