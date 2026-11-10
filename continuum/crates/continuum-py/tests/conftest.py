"""Starting a real broker for a test so it can't outlive the test run.

A fixture's `finally` stops its broker when the test ends, but not when the test process itself is killed (a timeout,
a Ctrl-C at the wrong moment, SIGKILL), and a broker left that way runs on with its socket and arena (2026-10-29: one
ran for 7 hours after its worktree was deleted). So every broker is started through `spawn_broker`:

- On Linux the kernel kills it when the process that started it dies, however that dies: it's started through
  util-linux's `setpriv --pdeathsig KILL`, which sets PR_SET_PDEATHSIG and execs the broker in place (not a
  `preexec_fn`, which isn't safe in a process with threads, and these tests run tokio's and asyncio's).
- Elsewhere (macOS has no equivalent) it's killed at interpreter exit and on SIGTERM or SIGHUP; only a SIGKILL of the
  test process can leave it, and then `unlink_arena` cleans the next run's way.
- It runs with a small Tier 0 arena (16 slots of 16 KiB), not the profile's gigabytes.
"""

import atexit
import ctypes
import os
import shutil
import signal
import socket as py_socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from types import FrameType

import pytest

BROKER = Path(
    os.environ.get(
        "CONTINUUM_BROKER",
        Path(__file__).resolve().parents[3] / "target" / "debug" / "continuum-broker",
    )
)

TIER_WORKER = BROKER.parent / "continuum-tier-worker"

_children: set[subprocess.Popen[bytes]] = set()


# On Linux, `setpriv` makes the broker die with this process; where it's missing, the atexit and signal cleanup
# below still apply.
_DIE_WITH_PARENT = (
    ["setpriv", "--pdeathsig", "KILL"] if sys.platform == "linux" and shutil.which("setpriv") is not None else []
)


def _kill_children() -> None:
    for child in list(_children):
        if child.poll() is None:
            child.kill()
            child.wait()
    _children.clear()


def _on_signal(signum: int, frame: FrameType | None) -> None:
    _kill_children()
    previous = _previous.get(signum)
    if callable(previous):
        previous(signum, frame)
    else:
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)


atexit.register(_kill_children)
_previous = {sig: signal.signal(sig, _on_signal) for sig in (signal.SIGTERM, signal.SIGHUP)}


def accepts(socket: Path) -> bool:
    probe = py_socket.socket(py_socket.AF_UNIX)
    try:
        probe.connect(str(socket))
    except OSError:
        return False
    finally:
        probe.close()
    return True


def unlink_arena(socket: Path) -> None:
    """Removes the Tier 0 arena of a broker that was killed (SIGKILL skips its own cleanup). The arena's name is
    the second line of `<socket>.arena`."""
    state = Path(f"{socket}.arena")
    if not state.exists():
        return
    lines = state.read_text().splitlines()
    if len(lines) > 1:
        # libc's shm_unlink, on Linux and macOS alike; a name that's already gone is fine.
        ctypes.CDLL(None).shm_unlink(lines[1].encode())


def spawn_broker(socket: Path, *args: str, tier0_classes: str = "16384x16") -> subprocess.Popen[bytes]:
    """Starts a broker on `socket` (with extra `args`) and waits until it accepts connections. The caller stops it
    with `stop_broker`; if the test process dies first, it dies too (see the module's docs)."""
    broker = subprocess.Popen(
        [*_DIE_WITH_PARENT, BROKER, "--socket", socket, "--tier0-classes", tier0_classes, *args],
        stderr=subprocess.DEVNULL,
    )
    _children.add(broker)
    try:
        # Wait until a connection succeeds, not just until the file exists: bind creates the file a moment before
        # listen, and a connect in between is refused.
        deadline = time.monotonic() + 10
        while not accepts(socket):
            assert broker.poll() is None, "the broker exited before listening"
            assert time.monotonic() < deadline, "the broker didn't start listening"
            time.sleep(0.01)
    except BaseException:
        stop_broker(broker, socket, kill=True)
        raise
    return broker


def stop_broker(broker: subprocess.Popen[bytes], socket: Path, *, kill: bool = False) -> None:
    """Stops a broker from `spawn_broker`: SIGTERM and up to 10 s to exit (or SIGKILL at once with `kill`), then
    removes its arena if it didn't."""
    if broker.poll() is None:
        if kill:
            broker.kill()
        else:
            broker.terminate()
        try:
            broker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            broker.kill()
            broker.wait()
    _children.discard(broker)
    unlink_arena(socket)


@pytest.fixture
def socket(tmp_path: Path) -> Iterator[str]:
    """A running broker's socket, for one test."""
    path = tmp_path / "b.sock"
    broker = spawn_broker(path)
    try:
        yield str(path)
    finally:
        stop_broker(broker, path)


def spawn_tier_worker(socket: Path, chunks: Path, *streams: str) -> subprocess.Popen[bytes]:
    """Starts the tier worker on a running broker, flushing `streams` into chunk files under `chunks`. It dies with
    this process, like the broker; the caller kills it when done."""
    args = [str(TIER_WORKER), "--socket", str(socket), "--chunks", str(chunks), "--manifest", "/nonexistent"]
    for s in streams:
        args += ["--stream", s]
    worker = subprocess.Popen([*_DIE_WITH_PARENT, *args], stderr=subprocess.DEVNULL)
    _children.add(worker)
    return worker
