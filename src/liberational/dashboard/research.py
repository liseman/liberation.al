"""The Inbox's routes: research jobs (memory/research.py) for the owner, on this dashboard's origin only. A question
is researched through Ask's one route, POST /pack-ask/jobs with `deep: true` (memory/ask.py handle_http).

GET    /research/jobs              {jobs: [newest first, without reports], available, reason, unread, typical_s,
                                   reach: {quick, deep}} (research.reach: what each path looks through)
GET    /research/jobs/<id>         one job with its report
POST   /research/jobs/<id>/read    {read: bool} -> marks the report read or unread
POST   /research/jobs/<id>/delete  removes the job and its report
"""

import json
import re
from typing import Any

from liberational import logs
from liberational.dashboard import routes
from liberational.memory import ask, research

LOG = logs.get("dashboard")
JOB = re.compile(r"/research/jobs/([0-9a-f]{32})(/read|/delete)?")


def _body(handler: Any) -> dict[str, Any]:
    if handler.headers.get_content_type() != "application/json":
        raise TypeError("Use application/json.")
    length = int(handler.headers.get("Content-Length") or 0)
    if not 0 < length <= 8192:
        raise ValueError("Invalid request size.")
    body = json.loads(handler.rfile.read(length))
    if not isinstance(body, dict):
        raise ValueError("Expected a JSON object.")
    return body


def view(store: research.Store | None = None) -> dict[str, Any]:
    """GET /research/jobs, and the live feed's `research` event: the jobs, whether research can run, the unread
    count, how long research usually takes here, and what a quick answer and deep research each look through."""
    jobs = (store or research.Store()).jobs()
    reason = research.unavailable()
    unread = sum(1 for j in jobs if j["status"] in ("done", "failed") and not j["read"])
    return {
        "jobs": jobs,
        "available": reason is None,
        "reason": reason,
        "unread": unread,
        "typical_s": research.typical_s(jobs),
        "reach": {"quick": research.reach(False), "deep": research.reach(True)},
    }


def handle(handler: Any) -> bool:
    """Answer a /research request, or return False for any other path."""
    path = handler.path.split("?", 1)[0]
    if path != "/research/jobs" and not path.startswith("/research/jobs/"):
        return False
    if not ask._owner_http(handler):
        return True
    store = research.Store()
    match = JOB.fullmatch(path)
    try:
        if handler.command == "POST" and not ask._same_origin(handler.headers):
            handler._json({"error": "Cross-origin request rejected."}, 403)
        elif handler.command in ("GET", "HEAD") and path == "/research/jobs":
            handler._json(view(store))
        elif handler.command in ("GET", "HEAD") and match and not match[2]:
            job = store.get(match[1])
            handler._json(job if job else {"error": "No such research job."}, 200 if job else 404)
        elif handler.command == "POST" and match and match[2] == "/read":
            read = _body(handler).get("read")
            if not isinstance(read, bool):
                raise ValueError("read is true or false.")
            found = store.mark(match[1], read)
            handler._json(
                {"ok": True, "read": read} if found else {"error": "No such research job."}, 200 if found else 404
            )
        elif handler.command == "POST" and match and match[2] == "/delete":
            found = store.delete(match[1])
            handler._json({"ok": True} if found else {"error": "No such research job."}, 200 if found else 404)
        else:
            handler._json({"error": "Method not supported."}, 405)
    except TypeError as exc:
        handler._json({"error": str(exc)}, 415)
    except ValueError as exc:
        handler._json({"error": str(exc)}, 400)
    except OSError:
        LOG.warning("research jobs are unavailable", exc_info=True)
        handler._json({"error": "Research jobs are unavailable right now."}, 503)
    return True


# ---- routes (dashboard/routes.py) ----


def _ours(path: str) -> bool:
    return path == "/research/jobs" or path.startswith("/research/jobs/")


@routes.get_when("/research/jobs, /research/jobs/<id>", _ours)
@routes.post_raw_when("/research/jobs/<id>/read, /research/jobs/<id>/delete", _ours)
def _route(h: Any) -> None:
    handle(h)
