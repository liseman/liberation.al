"""The Objects page and its API (frontend/src/objects/), served by the dashboard from `memory.objects`.

GET  /objects                 the page
GET  /objects/api/status      how many objects, which cameras have a fresh frame, and the worker's last sighting
GET  /objects/api/objects     every object with where it was last seen
GET  /objects/api/draft       the view a spoken "remember this as my wallet" saved, with its name, for 10 minutes
GET  /objects/api/query?q=    where an object named in q was last seen
GET  /objects/evidence/<ref>  a JPEG snapshot/template/sighting, or a transparent segmentation PNG
POST /objects/api/capture     {camera, draft?, name?}: a snapshot of the camera's newest frame to crop from
POST /objects/api/import      {camera, image, captured_at, kind?}: a Live crop or native Review full_frame JPEG
POST /objects/api/segment     {snapshot, bbox|point, crop?, job?}: an EdgeTAM mask for the box or point the owner chose
POST /objects/api/find        {snapshot, crop?, job?}: every object EdgeTAM finds in the crop (the view the owner is
                              looking at), each a selection with its outline
                              (never an automatic pass over the frame) and a snapshot-bound selection token
POST /objects/api/cancel     {job}: stop only that owned EdgeTAM request before replacing the frozen view
POST /objects/api/enroll      {name, camera, snapshot, selection?|full_crop?|direct_crop?, crop?, aliases, note}: remember explicitly
"""

import base64
import binascii
import io
import json
import math
import secrets
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from liberational import config as cfg
from liberational import logs, vision_log
from liberational.memory import objects
from liberational.vision import segmentation

LOG = logs.get("dashboard")
_snapshots = objects.Snapshots()
_lock = threading.Lock()
_store: objects.Store | None = None
DRAFT_S = 600.0
FULL_FRAME_DATA_URL_LIMIT = 16_000_000  # about 12 MB of JPEG, without resizing the recorded frame
FULL_FRAME_PIXELS = 16_000_000
FULL_FRAME_DIMENSION = 8192
IMPORT_JSON_LIMIT = FULL_FRAME_DATA_URL_LIMIT + 10_000


@dataclass
class Selection:
    snapshot: str
    bbox: list[float]
    result: segmentation.Result
    preview: bytes
    made: float = field(default_factory=time.time)


_selections: dict[str, Selection] = {}
MAX_SELECTIONS = 64  # held at once: a Find objects run makes up to segmentation.FIND_MAX
OUTLINE_POINTS = 120  # an outline is simplified until it has at most about this many points


def store() -> objects.Store:
    global _store
    with _lock:
        if _store is None:
            _store = objects.Store(cfg.path("paths.state"))
        return _store


def _cameras(now: float) -> list[dict[str, Any]]:
    """Each camera's id, its name (vision.camera_names; its id when it has none), and whether it has a fresh frame."""
    from liberational.vision import views

    named = views.names()
    out = []
    for cam in objects.camera_names():
        row: dict[str, Any] = {"camera": cam, "name": named.get(cam, cam)}
        try:
            got = vision_log.at(store().state).newest(cam, within_s=objects.FRESH_SECONDS)
        except Exception as e:  # the broker out of reach
            out.append({**row, "available": False, "error": str(e)[:120]})
            continue
        if got is None:
            out.append({**row, "available": False, "error": "no recent frame"})
        else:
            out.append({**row, "available": True, "stale": now - got[1] > objects.FRESH_SECONDS, "source": "camera"})
    return out


def status() -> dict[str, Any]:
    now = time.time()
    s = store()
    rows = s.objects()
    newest = [r for r in s.sightings(now - 3600, now + 1) if r.get("kind") == "visual_match"]
    return {
        "objects": len(rows),
        "visual_objects": sum(o.get("mode") == "visual" for o in rows),
        "cameras": _cameras(now),
        # the object worker (pack-see-objects) commits sightings only: its newest one in the last hour stands in for
        # the time an observer pass last finished, which the page shows
        "observer": {"finished_at": max(r["ts_start"] for r in newest)} if newest else None,
        "observer_error": None,
        "recognition": "CPU SIFT + RANSAC; two distinct consistent frames; no person recognition",
    }


def capture(data: dict[str, Any]) -> dict[str, Any]:
    cam = str(data.get("camera") or "cam0")
    if cam not in objects.camera_names():
        raise ValueError("camera must be one of " + ", ".join(objects.camera_names()))
    segmentation.cancel_current()
    got = vision_log.at(store().state).newest(cam, within_s=objects.FRESH_SECONDS)
    if got is None:
        raise ValueError("No recent picture from that camera. No object was remembered.")
    image, captured_at, frame_id = got
    snap = _snapshots.add(cam, captured_at, objects.jpeg(image, max_bytes=2_000_000), frame_id)
    if data.get("draft"):
        _snapshots.draft = (snap, str(data.get("name", "")).strip()[:80], time.time())
    return snap.meta()


def import_view(data: dict[str, Any]) -> dict[str, Any]:
    """Keep a clicked Live crop or a native recorded Review frame in the bounded temporary snapshot cache.

    Full frames keep their exact JPEG bytes and historical capture timestamp. An explicit validated rectangle can be
    remembered directly; masked enrollment requires a snapshot-bound EdgeTAM selection.
    """
    cam = str(data.get("camera", ""))
    if cam not in objects.camera_names():
        raise ValueError("Unknown camera.")
    kind = data.get("kind", "crop")
    if kind not in ("crop", "full_frame"):
        raise ValueError("Unknown object picture kind.")
    full_frame = kind == "full_frame"
    timestamp = data.get("captured_at", 0)
    if full_frame and (isinstance(timestamp, bool) or not isinstance(timestamp, int | float)):
        raise ValueError("Invalid recorded frame timestamp.")
    try:
        captured_at = float(timestamp)
    except OverflowError as exc:
        raise ValueError("Invalid object picture timestamp.") from exc
    age = time.time() - captured_at
    if full_frame:
        if "captured_at" not in data or not math.isfinite(captured_at) or captured_at < 0 or age < -2:
            raise ValueError("Invalid recorded frame timestamp.")
    elif not math.isfinite(captured_at) or not -2 <= age <= 60:
        raise ValueError("The Live object picture expired. Click the object again.")
    value = data.get("image")
    prefix = "data:image/jpeg;base64,"
    limit = FULL_FRAME_DATA_URL_LIMIT if full_frame else 2_700_000
    if not isinstance(value, str) or not value.startswith(prefix) or len(value) > limit:
        raise ValueError("Expected a bounded JPEG object picture.")
    try:
        raw = base64.b64decode(value[len(prefix) :], validate=True)
    except binascii.Error as exc:
        raise ValueError("Invalid object picture.") from exc
    dimension = FULL_FRAME_DIMENSION if full_frame else 1280
    try:
        with Image.open(io.BytesIO(raw)) as header:
            width, height = header.size
            if header.format != "JPEG" or max(width, height) > dimension or width * height > FULL_FRAME_PIXELS:
                raise ValueError("Invalid object picture dimensions.")
    except Image.DecompressionBombError as exc:
        raise ValueError("Invalid object picture dimensions.") from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Invalid object picture.") from exc
    image = objects.decode(raw)
    if image is None or max(image.shape[:2]) > dimension or image.shape[0] * image.shape[1] > FULL_FRAME_PIXELS:
        raise ValueError("Invalid object picture dimensions.")
    metadata = data.get("frame_metadata")
    if metadata is not None:
        from liberational.see.frames import source_content_rect

        if (
            not full_frame
            or not isinstance(metadata, dict)
            or source_content_rect(metadata) is None
            or metadata.get("width") != image.shape[1]
            or metadata.get("height") != image.shape[0]
            or isinstance(metadata.get("capturedAtUs"), bool)
            or not isinstance(metadata.get("capturedAtUs"), int)
            or abs(metadata["capturedAtUs"] - captured_at * 1e6) > 1
        ):
            raise ValueError("Recorded frame geometry does not match the captured picture")
    segmentation.cancel_current()
    encoded = raw if full_frame else objects.jpeg(image, max_bytes=2_000_000)
    snap = _snapshots.add(
        cam, captured_at, encoded, live_crop=not full_frame, full_frame=full_frame, frame_metadata=metadata
    )
    return snap.meta()


def draft() -> dict[str, Any] | None:
    held = _snapshots.draft
    if held is None or time.time() - held[2] > DRAFT_S or _snapshots.get(held[0].token) is None:
        return None
    snap, name, saved_at = held
    return {"snapshot": snap.meta(), "name": name, "saved_at": saved_at}


@dataclass
class Crop:
    """A validated pixel rectangle and its lossless EdgeTAM input, tied to the original frame."""

    image: bytes
    frame_shape: tuple[int, int]
    rectangle: tuple[int, int, int, int]
    input_shape: tuple[int, int]

    def restore(self, result: segmentation.Result) -> segmentation.Result:
        if result.mask.shape != self.input_shape or not math.isfinite(result.score):
            raise segmentation.Unavailable("Invalid cropped segmentation result")
        x, y, width, height = self.rectangle
        local = cv2.resize(result.mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        mask = np.zeros(self.frame_shape, dtype=bool)
        mask[y : y + height, x : x + width] = local.astype(bool)
        return segmentation.Result(mask, result.score, result.model)

    def prompt(self, bbox: Any, point: Any) -> tuple[list[float] | None, list[float] | None]:
        if (bbox is None) == (point is None):
            raise ValueError("Provide exactly one bbox or point")
        height, width = self.frame_shape
        x, y, w, h = self.rectangle
        if bbox is not None:
            bx, by, bw, bh = objects._box(bbox)
            local = [(bx * width - x) / w, (by * height - y) / h, bw * width / w, bh * height / h]
            if local[0] < -1e-9 or local[1] < -1e-9 or local[0] + local[2] > 1 + 1e-9 or local[1] + local[3] > 1 + 1e-9:
                raise ValueError("bbox must be inside the crop")
            # Clamp accepted boundary noise before the inference layer's strict box validation.
            left, top, right, bottom = [
                min(1.0, max(0.0, v)) for v in (local[0], local[1], local[0] + local[2], local[1] + local[3])
            ]
            return [left, top, right - left, bottom - top], None
        if not isinstance(point, list | tuple) or len(point) != 2:
            raise ValueError("point must contain two normalized coordinates")
        if any(
            isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v) or not 0 <= v <= 1
            for v in point
        ):
            raise ValueError("point must contain two normalized coordinates")
        local_point = [(point[0] * width - x) / w, (point[1] * height - y) / h]
        if any(v < -1e-9 or v > 1 + 1e-9 for v in local_point):
            raise ValueError("point must be inside the crop")
        return None, [min(1.0, max(0.0, v)) for v in local_point]


def _crop(data: dict[str, Any], image: bytes) -> Crop | None:
    if "crop" not in data:
        return None
    frame = objects.decode(image)
    if frame is None:
        raise ValueError("the frozen image is invalid")
    height, width = frame.shape[:2]
    box = data["crop"]
    if not isinstance(box, list | tuple) or len(box) != 4 or any(type(v) is not int for v in box):
        raise ValueError("crop must contain four integer frame pixels")
    x, y, w, h = box
    if x < 0 or y < 0 or w < 2 or h < 2 or x + w > width or y + h > height:
        raise ValueError("crop must be inside the frame and at least 2 pixels wide and high")
    scale = 1024 / max(w, h)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    resized = cv2.resize(frame[y : y + h, x : x + w], size, interpolation=cv2.INTER_LINEAR)
    return Crop(_png(resized), (height, width), (x, y, w, h), (size[1], size[0]))


def segment(data: dict[str, Any]) -> dict[str, Any]:
    token = str(data.get("snapshot", ""))
    snap = _snapshots.get(token)
    if snap is None:
        raise FileNotFoundError("snapshot")
    crop = _crop(data, snap.image)
    bbox, point = data.get("bbox"), data.get("point")
    if crop is not None:
        bbox, point = crop.prompt(bbox, point)
    result = segmentation.segment(crop.image if crop else snap.image, bbox=bbox, point=point, job=data.get("job"))
    if crop is not None:
        result = crop.restore(result)
        if not np.any(result.mask):
            raise segmentation.Unavailable("The cropped selection is empty")
    return _remember_selection(token, result)


def find(data: dict[str, Any]) -> dict[str, Any]:
    """Every object EdgeTAM finds in the snapshot's `crop` (the whole frame without one), best first: each a selection,
    as /api/segment gives one, with its outline, and how long it took."""
    token = str(data.get("snapshot", ""))
    snap = _snapshots.get(token)
    if snap is None:
        raise FileNotFoundError("snapshot")
    began = time.monotonic()
    crop = _crop(data, snap.image)
    found = segmentation.find_all(crop.image if crop else snap.image, job=data.get("job"))
    out = []
    for result in found:
        if crop is not None:
            result = crop.restore(result)
        if np.any(result.mask):
            out.append({**_remember_selection(token, result), "outline": outline(result.mask)})
    return {"snapshot": token, "objects": out, "ms": round(1000 * (time.monotonic() - began))}


def outline(mask: np.ndarray) -> list[list[list[float]]]:
    """A mask's outer edges as polygons of normalized [x, y] points, simplified to about OUTLINE_POINTS in all."""
    height, width = mask.shape
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= 0.05 * max(cv2.contourArea(k) for k in contours)]
    total = sum(cv2.arcLength(c, True) for c in contours)
    polygons = []
    for c in contours:
        simple = cv2.approxPolyDP(c, max(1.0, total / OUTLINE_POINTS), True)
        if len(simple) >= 3:
            polygons.append([[round(float(x) / width, 5), round(float(y) / height, 5)] for [[x, y]] in simple])
    return polygons


def _png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("the selection preview could not be encoded")
    return encoded.tobytes()


def _remember_selection(token: str, result: segmentation.Result) -> dict[str, Any]:
    ys, xs = np.nonzero(result.mask)
    height, width = result.mask.shape
    bbox = [
        float(xs.min() / width),
        float(ys.min() / height),
        float((xs.max() - xs.min() + 1) / width),
        float((ys.max() - ys.min() + 1) / height),
    ]
    preview = np.zeros((height, width, 4), dtype=np.uint8)
    preview[result.mask] = (209, 84, 52, 112)
    png = _png(preview)
    selection = secrets.token_hex(16)
    with _lock:
        now = time.time()
        expired = [key for key, held in _selections.items() if now - held.made > objects.SNAPSHOT_TTL]
        for key in expired:
            del _selections[key]
        while len(_selections) >= MAX_SELECTIONS:
            del _selections[next(iter(_selections))]
        _selections[selection] = Selection(token, bbox, result, png)
    return {
        "selection": selection,
        "snapshot": token,
        "bbox": bbox,
        "mask": f"segmentation:{selection}",
        "score": result.score,
        "model": result.model,
    }


def _selection(token: str, snapshot: str | None = None) -> Selection:
    with _lock:
        selected = _selections.get(token)
    if selected is None or time.time() - selected.made > objects.SNAPSHOT_TTL:
        raise FileNotFoundError("selection")
    if snapshot is not None and selected.snapshot != snapshot:
        raise ValueError("the EdgeTAM selection belongs to another snapshot")
    if _snapshots.get(selected.snapshot) is None:
        raise FileNotFoundError("snapshot")
    return selected


def evidence(ref: str) -> bytes:
    kind, _, key = ref.partition(":")
    if kind == "segmentation":
        return _selection(key).preview
    if kind == "snapshot":
        snap = _snapshots.get(key)
        if snap is None:
            raise FileNotFoundError(ref)
        return snap.image
    return store().evidence(ref)


def enroll(data: dict[str, Any]) -> dict[str, Any]:
    snap = _snapshots.get(str(data.get("snapshot", "")))
    if snap is None:
        raise FileNotFoundError("snapshot")
    if snap.camera != data.get("camera"):
        raise ValueError("snapshot camera mismatch")
    crop = _crop(data, snap.image)
    if not data.get("selection"):
        if data.get("direct_crop") is True:
            if crop is None:
                raise ValueError("A direct crop requires a crop rectangle")
            height, width = crop.frame_shape
            x, y, w, h = crop.rectangle
            return store().remember(
                name=data.get("name", ""),
                snapshot=snap,
                bbox=[x / width, y / height, w / width, h / height],
                note=data.get("note", ""),
                aliases=data.get("aliases"),
            )
        if not snap.live_crop or data.get("full_crop") is not True:
            raise ValueError("Select the object outline before remembering it.")
        # Only a validated, cached Live crop can use this explicit unmasked path. Never trust a client bbox.
        return store().remember(
            name=data.get("name", ""),
            snapshot=snap,
            bbox=[0.0, 0.0, 1.0, 1.0],
            note=data.get("note", ""),
            aliases=data.get("aliases"),
        )
    selected = _selection(str(data["selection"]), snap.token)
    return store().remember(
        name=data.get("name", ""),
        snapshot=snap,
        bbox=selected.bbox,
        note=data.get("note", ""),
        aliases=data.get("aliases"),
        mask=selected.result.mask,
        segmentation_model=selected.result.model,
        segmentation_score=selected.result.score,
    )


def handle(h: Any, method: str) -> bool:
    """Answers an /objects request on the dashboard's handler `h`; False for any other path."""
    u = urllib.parse.urlsplit(h.path)
    p = u.path
    if not (p == "/objects" or p.startswith("/objects/")):
        return False
    sub = p[len("/objects") :] or "/"
    try:
        if method == "GET":
            if sub in ("/", ""):
                from liberational.dashboard.server import app_page

                code, body = app_page()
                h._send(code, "text/html; charset=utf-8", body)
            elif sub == "/api/status":
                h._json(status())
            elif sub == "/api/objects":
                h._json({"objects": store().list_objects()})
            elif sub == "/api/draft":
                h._json({"draft": draft()})
            elif sub == "/api/query":
                q = urllib.parse.parse_qs(u.query).get("q", [""])[0]
                if not q or len(q) > 1000:
                    raise ValueError("Provide a question of 1 to 1000 characters.")
                h._json(store().query(q))
            elif sub.startswith("/evidence/"):
                ref = urllib.parse.unquote(sub[len("/evidence/") :])
                h._send(
                    200,
                    "image/png" if ref.startswith("segmentation:") else "image/jpeg",
                    evidence(ref),
                )
            else:
                h._json({"error": "not found"}, 404)
            return True
        # Same-origin JSON only; native local clients (voice) may omit Origin.
        origin = h.headers.get("Origin")
        if origin and origin != "http://" + (h.headers.get("Host") or ""):
            h._json({"error": "Cross-origin write rejected."}, 403)
            return True
        if h.headers.get_content_type() != "application/json":
            h._json({"error": "Use application/json."}, 415)
            return True
        n = int(h.headers.get("Content-Length", "0"))
        limit = IMPORT_JSON_LIMIT if sub == "/api/import" else 32768
        if not 0 < n <= limit:
            raise ValueError("Invalid request size.")
        data = json.loads(h.rfile.read(n))
        if not isinstance(data, dict):
            raise ValueError("Expected JSON object.")
        if sub == "/api/capture":
            h._json(capture(data))
        elif sub == "/api/import":
            h._json(import_view(data))
        elif sub == "/api/segment":
            h._json(segment(data))
        elif sub == "/api/find":
            h._json(find(data))
        elif sub == "/api/cancel":
            h._json({"ok": True, "cancelled": segmentation.cancel(str(data.get("job", "")))})
        elif sub == "/api/enroll":
            h._json({"object": enroll(data)}, 201)
        else:
            h._json({"error": "not found"}, 404)
    except segmentation.Cancelled as exc:
        h._json({"error": str(exc)}, 409)
    except segmentation.Unavailable as exc:
        h._json({"error": str(exc)}, 503)
    except FileNotFoundError:
        h._json({"error": "Snapshot, selection, or evidence expired. Capture a new one."}, 404)
    except (ValueError, TypeError, KeyError) as exc:
        h._json({"error": str(exc)}, 400)
    except Exception as exc:
        LOG.error(f"objects: {method} {sub} failed: {type(exc).__name__}: {exc}")
        h._json({"error": "Object memory is unavailable."}, 503)
    return True
