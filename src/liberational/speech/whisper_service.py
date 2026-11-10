"""Supervise native whisper-server and report its actual model holder after health readiness.

pack-whisper <binary> <original whisper-server args...> retains the declared service's native arguments. The
supervisor forwards termination, waits for the child, and removes its load evidence when the child exits. Later
probe failures cannot prove a verified model unloaded.
"""

import json
import signal
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from types import FrameType

import psutil

from liberational import model_runtime


def _option(args: Sequence[str], name: str) -> str:
    index = args.index(name)
    return args[index + 1]


def ready(pid: int, host: str, port: int) -> bool:
    """A healthy endpoint must be listened to by this child, not another server on its intended port."""
    try:
        process = psutil.Process(pid)
        if not any(
            c.status == psutil.CONN_LISTEN and c.laddr.port == port for c in process.net_connections(kind="tcp")
        ):
            return False
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=0.5) as response:
            body = json.loads(response.read(4096))
            return response.status == 200 and isinstance(body, dict) and body.get("status") == "ok"
    except OSError, ValueError, psutil.Error, urllib.error.URLError:
        return False


def supervise(args: Sequence[str]) -> int:
    """Own one native child's lifetime; loading is not reported merely because it spawned."""
    host, port = _option(args, "--host"), int(_option(args, "--port"))
    child = subprocess.Popen(list(args))
    stopping = threading.Event()
    token: model_runtime.LoadToken | None = None

    def stop(signum: int, _frame: FrameType | None) -> None:
        stopping.set()
        if child.poll() is None:
            child.send_signal(signum)

    previous = {signum: signal.signal(signum, stop) for signum in (signal.SIGTERM, signal.SIGINT)}
    try:
        with model_runtime.reporting("whisper"):
            while child.poll() is None and not stopping.is_set():
                healthy = ready(child.pid, host, port)
                if healthy and child.poll() is None and token is None:
                    token = model_runtime.loaded(
                        "models.files.asr_whisper",
                        Path(_option(args, "-m")).name,
                        "speech to text",
                        "unknown",
                        external_pid=child.pid,
                    )
                stopping.wait(0.5)
    finally:
        if token is not None:
            token.close()
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
        child.wait()
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    return child.returncode


def main() -> None:
    raise SystemExit(supervise(sys.argv[1:]))
