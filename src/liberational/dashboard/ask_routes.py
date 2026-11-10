"""Ask's routes on the dashboard (#118): the questions asked of the household's history and their saved answers.
The logic is liberational/memory/ask.py's `handle_http`; this registers its paths in the dashboard's table.

GET    /pack-ask/jobs/<id>     a question's job and its answer
POST   /pack-ask/jobs          ask a question
GET, POST, DELETE /pack-ask/history and /pack-ask/history/...   the saved questions
"""

from __future__ import annotations

from typing import Any

from liberational import logs
from liberational.dashboard import routes
from liberational.dashboard.context import log


def _ours(path: str) -> bool:
    return path in ("/pack-ask/history", "/pack-ask/jobs") or path.startswith(("/pack-ask/history/", "/pack-ask/jobs/"))


@routes.get_when("/pack-ask/jobs/<id>, /pack-ask/history, /pack-ask/history/...", _ours)
@routes.post_raw_when("/pack-ask/jobs, /pack-ask/history/...", _ours)
@routes.delete_when("/pack-ask/history/...", _ours)
def _route(h: Any) -> None:
    """Ask answers, and never raises into the server: a failure is a 503 that says so."""
    try:
        from liberational.memory.ask import handle_http

        handle_http(h)
    except Exception as ex:
        log("pack_ask failed:", ex, level=logs.ERROR)
        try:
            h._json({"error": f"pack_ask unavailable: {ex}"}, 503)
        except Exception:
            pass
