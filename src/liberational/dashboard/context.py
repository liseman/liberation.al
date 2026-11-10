"""What the dashboard's modules share (#61): where the state directory is, the log, and a few small things.

Every area of the dashboard reads state from one directory, and its tests point that at a temporary one. It lives
here, not in the server, so an area's module and its tests don't need the server to find it:
`monkeypatch.setattr(context, "STATE", tmp_path)`. Code reads `context.STATE` at the time it's needed, never a copy
taken at import.
"""

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


def log(*a: object, level: int = logs.INFO) -> None:
    LOG.log(level, " ".join(str(x) for x in a))
