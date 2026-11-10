"""The zone editor's routes (#1316): read each camera's zones and their history, and save a camera's new set."""

from pathlib import Path
from typing import Any

from liberational.see import zones


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
