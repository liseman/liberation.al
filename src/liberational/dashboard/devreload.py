"""Dev mode for the dashboard (`mise run dev`, or `pack-dashboard --dev`).

- Python or config changes: the server checks that it still imports, then re-executes itself in place.
  A change that doesn't import is logged and the old code keeps running until the next save.
- The frontend (frontend/): its page comes from Vite's dev server (ports.vite) and the dashboard passes
  /static/app/* through to it (server.py `app_page` and `vite`), so Vite's own hot reload runs it. `mise run dev`
  starts Vite too.

None of this runs unless the dashboard was started with --dev.
"""

import os
import pathlib
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterable

from liberational.config import ROOT, local_config

BOOT = uuid.uuid4().hex[:12]  # changes on every restart, so pages can tell the server came back with new code


def _newest(files: Iterable[pathlib.Path]) -> int:
    return max((f.stat().st_mtime_ns for f in files if f.is_file()), default=0)


def code_version() -> int:
    src = ROOT / "src" / "liberational"
    return _newest([*src.rglob("*.py"), *(ROOT / "config").glob("*.toml"), local_config()])


def _imports() -> tuple[bool, list[str]]:
    r = subprocess.run([sys.executable, "-c", "import liberational.dashboard.server"], capture_output=True, text=True)
    return r.returncode == 0, r.stderr.strip().splitlines()[-1:] or [""]


def watch(log: Callable[..., None]) -> None:
    """Re-exec the dashboard when Python or config changes (in a daemon thread)."""

    def loop() -> None:
        seen = code_version()
        while True:
            time.sleep(0.5)
            now = code_version()
            if now == seen:
                continue
            seen = now
            ok, err = _imports()
            if not ok:
                log("dev: change doesn't import, keeping the old code:", err[0])
                continue
            log("dev: code changed, restarting")
            sys.stdout.flush()
            os.execv(sys.executable, [sys.executable, *sys.argv])

    threading.Thread(target=loop, daemon=True, name="devreload").start()
