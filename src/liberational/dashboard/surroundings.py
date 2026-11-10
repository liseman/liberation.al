"""The Live tab's "Learn surroundings" switch (#255): whether the cameras' pictures may refine a place's map, and
which cameras are sampled. spatial/surroundings.py holds the settings and how they're followed live.

GET  /surroundings  {"ok", "control", "on", "cameras": [{"name", "sampled", "default"}], "chosen": [name, ...]}:
     every camera in the device manifest, in its order; "sampled" says whether learning takes pictures from it
     while on, "default" marks the one used when none is chosen, and "chosen" is the choice as saved (empty: the
     default).
POST /surroundings  {"on"?: bool, "cameras"?: [name, ...]} -> the same view after the change. Either or both; an
     empty "cameras" goes back to the default. A name the manifest doesn't have is refused (400) and nothing is
     saved. This machine only, same-origin JSON only, like the other controls. Nothing restarts: the sampler
     follows pack.toml.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from liberational import config as cfg
from liberational.dashboard import routes
from liberational.spatial import surroundings

if TYPE_CHECKING:
    from liberational.dashboard.server import H


def view(control: bool) -> dict[str, Any]:
    switch = surroundings.read()
    return {
        "ok": True,
        "control": control,
        "on": switch.on,
        "cameras": [
            {"name": name, "sampled": name in switch.cameras, "default": name == switch.default}
            for name in surroundings.manifest_cameras()
        ],
        "chosen": list(switch.chosen),
    }


def change(body: Any) -> tuple[int, dict[str, Any]]:
    """Apply a POST body; the status and the reply."""
    shape = {"ok": False, "error": 'send {"on": true} or {"cameras": ["..."]}'}
    if not isinstance(body, dict) or not body or not set(body) <= {"on", "cameras"}:
        return 400, shape
    on, cameras = body.get("on"), body.get("cameras")
    if (on is not None and not isinstance(on, bool)) or (
        cameras is not None and (not isinstance(cameras, list) or any(not isinstance(c, str) for c in cameras))
    ):
        return 400, shape
    try:
        surroundings.save(on, cameras)
    except surroundings.UnknownCamera as error:
        return 400, {"ok": False, "error": str(error)}
    except cfg.SettingError as error:
        return 400, {"ok": False, "error": str(error)}
    return 200, view(True)


@routes.get("/surroundings")
def _get(h: H) -> None:
    h._json(view(h._can_control()))


@routes.post("/surroundings")
def _post(h: H, body: Any) -> None:
    if not h._can_control():
        return h._json({"ok": False, "error": "surroundings can only be changed from this machine"}, 403)
    if not h._same_origin_json():
        return h._json({"ok": False, "error": "same-origin JSON only"}, 403)
    code, reply = change(body)
    return h._json(reply, code)
