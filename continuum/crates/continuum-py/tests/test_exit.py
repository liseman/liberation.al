"""Exiting with a live client and awaitables still pending doesn't crash the process.

Each awaitable's result is handed to Python from a tokio thread. A process that exited with some pending used to
segfault now and then: the interpreter finalized, tearing down the pending futures, and the bridge then attached a
tokio thread to the dead interpreter (6 of 20 runs of this script on devbox, 2026-10-29). The script below exits that
way; it's run 20 times because the crash was a race (about 1 run in 4 before the fix).
"""

import subprocess
import sys
from pathlib import Path

SCRIPT = """
import asyncio, sys, threading, time
import continuum_py

loop = asyncio.new_event_loop()
threading.Thread(target=loop.run_forever, daemon=True).start()

async def connect():
    return await continuum_py.Client.connect(sys.argv[1])

client = asyncio.run_coroutine_threadsafe(connect(), loop).result(5)

async def wait_for_an_event():
    await client.next_event()

async def heartbeats():
    while True:
        await client.heartbeat()

asyncio.run_coroutine_threadsafe(wait_for_an_event(), loop)
asyncio.run_coroutine_threadsafe(heartbeats(), loop)
time.sleep(0.2)
"""


def test_exiting_with_pending_awaitables_never_crashes(socket: str, tmp_path: Path) -> None:
    script = tmp_path / "exit.py"
    script.write_text(SCRIPT)
    runs = [subprocess.run([sys.executable, script, socket], capture_output=True, timeout=30) for _ in range(20)]
    failed = [(r.returncode, r.stderr.decode(errors="replace")[-500:]) for r in runs if r.returncode != 0]
    assert not failed, f"exits that failed (-11 is a segfault): {failed}"
