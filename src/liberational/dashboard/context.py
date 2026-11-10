"""What the dashboard's modules share (#61): where the state directory is, the log, and a few small things.

Every area of the dashboard reads state from one directory, and its tests point that at a temporary one. It lives
here, not in the server, so an area's module and its tests don't need the server to find it:
`monkeypatch.setattr(context, "STATE", tmp_path)`. Code reads `context.STATE` at the time it's needed, never a copy
taken at import.
"""

import os
import sys
import threading
from typing import Any

from liberational import config as cfg
from liberational import logs

type Reply = tuple[int, dict[str, Any]]  # (HTTP status, JSON body)

STATE = cfg.path("paths.state")  # PACK_STATE= for tests
PY = sys.executable
LOG = logs.get("dashboard")

# How many browsers hold the live feed (GET /events) open, in a list so every module sees the one count.
SSE_CLIENTS = [0]
SSE_LOCK = threading.Lock()


def low_priority(priority: int = 15) -> None:
    """Lower this thread's CPU priority (and the decoder threads it starts) so live vision comes first."""
    try:
        tid = threading.get_native_id()
        if os.getpriority(os.PRIO_PROCESS, tid) < priority:
            os.setpriority(os.PRIO_PROCESS, tid, priority)
    except OSError, AttributeError:
        pass  # shortcut: per-thread nice is Linux only; macOS decodes at normal priority
