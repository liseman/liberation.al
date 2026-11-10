"""People's Training images: a person's face rows (the examples naming matches against), listed and removed.

`vision/faces.py` holds the rows and what is kept beside each (`vision/face_examples.py`); this is the dashboard's
three routes over them. Removing takes effect without a restart: see-actions' matcher rereads a gallery file that
changed on its next look.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from liberational import logs
from liberational.dashboard import people_list, routes
from liberational.vision import face_examples

if TYPE_CHECKING:
    from liberational.dashboard.server import H

LOG = logs.get("people")
MAX_KEYS = 500  # the most rows one request removes

type Reply = tuple[int, dict[str, Any]]


def listing(name: object) -> Reply:
    """GET /people/examples?name=: {name, count, examples}, the least like the rest first (`faces.examples`)."""
    if not isinstance(name, str) or not name.strip():
        return 400, {"error": "name a person"}
    rows = people_list._faces_mod().examples(name.strip())
    return 200, {"name": name.strip(), "count": len(rows), "examples": rows}


def remove(body: object) -> Reply:
    """POST /people/examples/remove {name, keys}: take these rows out of the person's training set
    (`faces.remove_examples`). Answers how many went and the listing as it is now."""
    if not isinstance(body, dict):
        return 400, {"ok": False, "error": "send a JSON object"}
    name, keys = body.get("name"), body.get("keys")
    if not isinstance(name, str) or not name.strip():
        return 400, {"ok": False, "error": "name a person"}
    if not isinstance(keys, list) or not keys or len(keys) > MAX_KEYS or not all(map(face_examples.valid_key, keys)):
        return 400, {"ok": False, "error": f"send 1 to {MAX_KEYS} training image keys"}
    removed = people_list._faces_mod().remove_examples(name.strip(), keys)
    code, reply = listing(name)
    return code, {"ok": True, "removed": removed, **reply}


def picture(key: str) -> bytes | None:
    """GET /people/examples/photo?key=: the row's kept picture, or None."""
    path = face_examples.picture(people_list._faces_mod().DB.parent, key)
    try:
        return path.read_bytes() if path is not None else None
    except OSError:
        return None


# ---- routes (dashboard/routes.py): what the server answers for this area ----


@routes.get("/people/examples")
def _get_people_examples(h: H) -> None:
    code, reply = listing(h._query().get("name"))
    return h._json(reply, code)


@routes.get("/people/examples/photo")
def _get_people_examples_photo(h: H) -> None:
    jpg = picture(h._query().get("key", ""))
    if jpg is None:
        return h._send(404, "text/plain", b"")
    h.send_response(200)
    h.send_header("Content-Type", "image/jpeg")
    h.send_header("Cache-Control", "private, max-age=86400")  # a key names one vector, whose picture never changes
    h.send_header("Content-Length", str(len(jpg)))
    h.end_headers()
    if h.command != "HEAD":
        h.wfile.write(jpg)


@routes.post("/people/examples/remove")
def _post_people_examples_remove(h: H, body: Any) -> None:
    code, reply = remove(body)  # gated before its body is read: people_writes.TEACHING_ROUTES
    return h._json(reply, code)
