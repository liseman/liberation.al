"""Give the memory a long-lived process has freed back to the system.

glibc's malloc keeps freed memory in its arenas for reuse rather than returning it, and a process that works with
large short-lived buffers (decoded video frames, a full read of the speech history) holds on to its peak. On the Linux machine the
dashboard grew from 0.46 to 1.09 GB over 3.5 hours, and its People list worker from 0.77 to 1.62 GB, with every byte
private heap. In a probe, cutting face photos from camera frames took a process from 81 to 620 MB, where it stayed,
and one malloc_trim(0) brought it back to 216 MB: the memory was free, just not returned.

`start()` trims every TRIM_S on a daemon thread of its own, once per process. A trim takes each arena's lock, so a
thread that allocates meanwhile waits: in the probe a trim took 25 to 96 ms after a burst of cuts. glibc's own knobs
(MALLOC_ARENA_MAX=2 and MALLOC_MMAP_THRESHOLD_=1 MiB) only took the same probe from 634 to 590 MB, so they aren't set. On a system without glibc (macOS) it
does nothing: the other allocators return memory on their own.
"""

import ctypes
import functools
import os
import threading
import time
from collections.abc import Callable

from liberational import logs

TRIM_S = 60.0  # how often a process started with start() trims
LOG = logs.get("memtrim")

_started: set[int] = set()  # the pids that have a trim thread (a spawned or forked child starts its own)
_lock = threading.Lock()


@functools.cache
def _malloc_trim() -> Callable[[int], int] | None:
    """glibc's malloc_trim, or None where there's no glibc."""
    try:
        trim = ctypes.CDLL("libc.so.6").malloc_trim
    except OSError, AttributeError:
        return None
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    return trim


def trim() -> bool:
    """Return the freed memory at the heap's top and in every arena to the system now. True when some was returned;
    False when none was, or without glibc."""
    malloc_trim = _malloc_trim()
    return bool(malloc_trim(0)) if malloc_trim is not None else False


def start(every_s: float = TRIM_S) -> bool:
    """Trim every `every_s` for this process's whole life, on a daemon thread, unless this process already does or
    has no glibc. True when this call started it."""
    if _malloc_trim() is None:
        return False
    with _lock:
        if os.getpid() in _started:
            return False
        _started.add(os.getpid())

    def run() -> None:
        while True:
            time.sleep(every_s)
            try:
                trim()
            except Exception as error:  # a failed trim mustn't end the process's trimming
                LOG.warning("malloc trim failed: %s", error)

    threading.Thread(target=run, name="malloc-trim", daemon=True).start()
    return True
