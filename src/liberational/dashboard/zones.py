"""The zone editor's routes: read each camera's zones and their history, and save a camera's new set."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from liberational.dashboard import routes
from liberational.see import zones

if TYPE_CHECKING:
    from liberational.dashboard.server import H


def listing(state: Path) -> tuple[int, dict[str, Any]]:
    """GET /zones: every camera's current zone set, {"cameras": {cam: {id, cam, zones, at}}}."""
    return 200, {"cameras": {cam: d.as_dict() for cam, d in zones.current(state).items()}}


def history(cam: str, state: Path) -> tuple[int, dict[str, Any]]:
    """GET /zones/history?cam=: the camera's saved versions, newest first (the audit trail)."""
    if not cam:
        return 400, {"error": "cam is required"}
    return 200, {"cam": cam, "versions": [d.as_dict() for d in reversed(zones.history(state, cam))]}


def save(body: object, control: bool, state: Path) -> tuple[int, dict[str, Any]]:
    """POST /zones {cam, zones, base}: save the camera's whole set, edited from version `base` (null for none)."""
    if not control:
        return 403, {"ok": False, "error": "zones can only be changed on this machine"}
    if not isinstance(body, dict):
        return 400, {"ok": False, "error": "send a JSON object"}
    cam, base = body.get("cam"), body.get("base")
    if not isinstance(cam, str) or not cam or (base is not None and not isinstance(base, str)):
        return 400, {"ok": False, "error": "cam and base (the version you edited, or null) are required"}
    try:
        saved = zones.define(state, cam, zones.parse_zones(body.get("zones")), base)
    except zones.StaleBase as exc:
        return 409, {"ok": False, "error": str(exc)}
    except ValueError as exc:
        return 400, {"ok": False, "error": str(exc)}
    return 200, {"ok": True, "definition": saved.as_dict()}


# ---- routes (dashboard/routes.py): what the server answers for this area ----


@routes.get("/zones", "/zones/history")
def _get_zones(h: H) -> None:
    p = h.path.split("?")[0]
    if p == "/zones":
        code, reply = listing(h.state)
    else:
        code, reply = history(h._query().get("cam", ""), h.state)
    return h._json(reply, code)


@routes.post("/zones")
def _post_zones(h: H, body: Any) -> None:
    if not h._same_origin_json():
        return h._json({"ok": False, "error": "same-origin JSON only"}, 403)
    code, reply = save(body, h._can_teach(), h.state)
    return h._json(reply, code)
