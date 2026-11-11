"""The Live tab's "Learn surroundings" switch (#255): whether pictures are collected from the cameras (for the 3D map
being rebuilt with a new method), and which cameras are sampled. spatial/surroundings.py holds the settings and how they're followed live.

GET  /surroundings  {"ok", "control", "on", "cameras": [{"name", "sampled", "default"}], "chosen": [name, ...]}:
     every camera in the device manifest, in its order; "sampled" says whether learning takes pictures from it
     while on, "default" marks the one used when none is chosen, and "chosen" is the choice as saved (empty: the
     default).
POST /surroundings  {"on"?: bool, "cameras"?: [name, ...]} -> the same view after the change. Either or both; an
     empty "cameras" goes back to the default. A name the manifest doesn't have is refused (400) and nothing is
     saved. This machine only, same-origin JSON only, like the other controls. Nothing restarts: the sampler
     follows pack.toml.
GET  /surroundings/status  what collecting pictures is doing now, for the Places page to show while it's open
     (#283): spatial/surroundings_frames.py's `status` as it is, {"on", "sampling", "kept", "cameras", "frames"}.
     Two small state files a read, so a page may ask every few seconds. "frames" is each camera's newest kept
     frame, for the picture below; it's empty when no picture may be shown.
GET  /surroundings/thumb/<frame id>.jpg  a kept frame as a small picture (THUMB_WIDTH wide), for the Places page's
     camera rows (#294). Whoever was in view is painted over first, from the pixels the sampler marked
     (spatial/surroundings_frames.py's `covered_path`), so a person or an animal is never shown; and without the
     sampler's person detector nothing was ever marked, so no picture is served at all (`pictures_allowed`). A frame's
     id names one picture that never changes, so a browser keeps it for a day, and the last THUMBS made are held here.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from liberational import config as cfg
from liberational.dashboard import routes
from liberational.spatial import surroundings

if TYPE_CHECKING:
    from liberational.dashboard.server import H

THUMB_WIDTH = 192  # px: a camera row shows it 64 wide, so this is sharp at three device pixels a pixel
THUMB_QUALITY = 72
THUMBS = 32  # pictures held in memory, about 8 kB each
COVER = (112, 116, 114)  # BGR: the flat grey painted over whoever was in view


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


def pictures_allowed() -> bool:
    """Whether a kept frame may be shown: only when the sampler's person detector is there, since without it the
    sampler marks nobody and a kept frame may show a person."""
    return cfg.model_path("person_detector").exists()


@functools.lru_cache(maxsize=THUMBS)
def thumbnail(folder: Path, frame_id: str) -> bytes | None:
    """The kept frame `frame_id` in `folder` as a small JPEG with whoever was in view painted over, or None when
    there's no such frame (or it doesn't read)."""
    import cv2
    import numpy as np

    store_path = folder / f"{frame_id}.jpg"
    if not store_path.is_file():
        return None
    pixels = cv2.imdecode(np.frombuffer(store_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if pixels is None:
        return None
    covered_path = folder / f"{frame_id}.covered.png"
    if covered_path.is_file():
        covered = cv2.imdecode(np.frombuffer(covered_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if covered is None:
            return None  # someone was in view and the mask doesn't read: no picture
        if covered.shape != pixels.shape[:2]:
            covered = cv2.resize(covered, (pixels.shape[1], pixels.shape[0]), interpolation=cv2.INTER_NEAREST)
        pixels[covered > 0] = COVER
    height = max(1, round(pixels.shape[0] * THUMB_WIDTH / pixels.shape[1]))
    small = cv2.resize(pixels, (THUMB_WIDTH, height), interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, THUMB_QUALITY])
    return jpeg.tobytes() if ok else None


@routes.get("/surroundings")
def _get(h: H) -> None:
    h._json(view(h._can_control()))


@routes.get("/surroundings/status")
def _get_status(h: H) -> None:
    from liberational.spatial import surroundings_frames

    said = surroundings_frames.status(h.state)
    if not pictures_allowed():
        said["frames"] = {}
    h._json(said)


@routes.get_re(r"/surroundings/thumb/([0-9a-f]{32})\.jpg")
def _get_thumb(h: H, match: re.Match[str]) -> None:
    from liberational.spatial import surroundings_frames

    jpg = thumbnail(surroundings_frames.root(h.state), match.group(1)) if pictures_allowed() else None
    if jpg is None:
        return h._send(404, "text/plain", b"")
    h.send_response(200)
    h.send_header("Content-Type", "image/jpeg")
    h.send_header("Cache-Control", "private, max-age=86400, immutable")  # an id names one picture, for good
    h.send_header("Content-Length", str(len(jpg)))
    h.end_headers()
    if h.command != "HEAD":
        h.wfile.write(jpg)


@routes.post("/surroundings")
def _post(h: H, body: Any) -> None:
    if not h._can_control():
        return h._json({"ok": False, "error": "surroundings can only be changed from this machine"}, 403)
    if not h._same_origin_json():
        return h._json({"ok": False, "error": "same-origin JSON only"}, 403)
    code, reply = change(body)
    return h._json(reply, code)
