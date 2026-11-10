"""The photos the People pages show, as files: a face's saved crop, a pet's photo, a scene thumbnail from recorded
video, and the background worker that prepares them (moved out of the server, #61).

GET /faces/shot/<id>.jpg                 a sighting's face crop
GET /faces/unknown/<group>.jpg           an unnamed group's photo
GET /people/<name>/photos/<stamp>.jpg    one of a person's photos
GET /people/video-thumb.jpg              a scene thumbnail
GET /pets/sightings/photo                a pet sighting's photo
"""

import contextlib
import json
import math
import os
import pathlib
import re
import threading
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import TYPE_CHECKING

from liberational import config as cfg
from liberational import segment_log, vision_log
from liberational.dashboard import call_log, context, routes
from liberational.dashboard.context import LOG
from liberational.people import face_shots, registry, video_thumbs

if TYPE_CHECKING:
    from liberational.dashboard.server import H


_shot_decode = threading.BoundedSemaphore(2)  # at most two frames read and decoded at once


# A shot request with no crop on disk waits this long for a turn to look its sighting up and cut it, then is told to
# come back (503): a grid of old sightings held every one of the browser's six connections for seconds at a time, and
# every other request of the page waited behind them.
SHOT_WAIT_S = 1.0


class ShotsBusy(Exception):
    """Every turn to cut a crop from video is taken: ask again shortly."""


_shot_writes = 0


_pet_shot_writes = 0


_video_thumb_writes = 0


VIDEO_THUMB_PRUNE_EVERY = 200  # thumbnails saved between prunes of their cache (people/video_thumbs.py)


def _low_priority(priority: int = 15) -> None:
    """Lower this thread's CPU priority (and the decoder threads it starts) so live vision comes first."""
    try:
        tid = threading.get_native_id()
        if os.getpriority(os.PRIO_PROCESS, tid) < priority:
            os.setpriority(os.PRIO_PROCESS, tid, priority)
    except OSError, AttributeError:
        pass  # shortcut: per-thread nice is Linux only; macOS decodes at normal priority


PET_TRACK = re.compile(r"[0-9A-Za-z:_-]{1,128}")  # an animal track's id: "<run>:<species>:<number>"


SIGHTING_ID = re.compile(r"[0-9A-Za-z-]{1,64}")  # a sighting's id: its detection's payload id (a UUID)


def _shot_revision(state: pathlib.Path | None = None) -> str:
    """The segment store's deletion revision: a forget moves it (about 3 microseconds to read)."""
    from continuum_py import segment_revision

    return str(segment_revision(segment_log.root(context.STATE if state is None else state)))


def _shot_file(
    sid: str, state: pathlib.Path | None = None, *, on_decode: Callable[[], None] | None = None
) -> tuple[pathlib.Path | None, str | None]:
    """Sighting sid's crop and its ETag. A crop checked under the current deletion revision is served with no
    lookup: see-actions writes one with each sighting, so People's grids read files. Otherwise the sighting is looked
    up first, so a crop of a deleted one is never served: a crop from before a forget is kept when its sighting still
    is, and one with none is cut from its recorded frame."""
    global _shot_writes
    import hashlib

    from liberational.vision import sightings

    if not SIGHTING_ID.fullmatch(sid):
        return None, None
    state = context.STATE if state is None else state
    revision = _shot_revision(state)
    out = face_shots.path(state, sid, revision)
    etag = '"' + hashlib.sha256(f"{face_shots.VERSION}:{sid}:{revision}".encode()).hexdigest() + '"'
    if out.exists():
        return out, etag
    older = face_shots.others(state, sid, revision)
    s = sightings.shot(sid, state)
    if s is None or not s["bbox_norm"]:
        for f in older:
            f.unlink(missing_ok=True)
        return None, None
    for previous in older:
        if not face_shots.promote(previous, out):
            continue
        for f in older:
            f.unlink(missing_ok=True)
        return out, etag
    if not _shot_decode.acquire(timeout=SHOT_WAIT_S):
        raise ShotsBusy
    try:
        _low_priority()
        if on_decode is not None:
            on_decode()
        jpg = face_shots.cut(state, s)
    finally:
        _shot_decode.release()
    if jpg is None:
        return None, None
    face_shots.write(state, sid, revision, jpg)
    _shot_writes += 1
    if _shot_writes % 200 == 0:
        face_shots.prune(state)
    return out, etag


def _pet_file(track: str, state: pathlib.Path, *, on_decode: Callable[[], None] | None = None) -> bytes | None:
    """Read or cut one pet crop, sharing the face/video decode cap with background jobs."""
    global _pet_shot_writes
    from liberational.dashboard import pets
    from liberational.people import pet_shots

    if not PET_TRACK.fullmatch(track):
        return None
    revision = pet_shots.revision(state)
    saved = pet_shots.path(state, track, revision)
    if saved.exists():
        return saved.read_bytes()
    sighting = pets.photo_sighting(track, state, revision)
    older = pet_shots.others(state, track, revision)
    if sighting is None:
        for old in older:
            old.unlink(missing_ok=True)
        return None
    for previous in older:
        if not face_shots.promote(previous, saved):
            continue
        for old in older:
            old.unlink(missing_ok=True)
        return saved.read_bytes()
    if not _shot_decode.acquire(timeout=SHOT_WAIT_S):
        raise ShotsBusy
    try:
        _low_priority()
        if on_decode is not None:
            on_decode()
        jpg = pets.photo(track, state, sighting=sighting)
    finally:
        _shot_decode.release()
    if jpg is not None:
        _pet_shot_writes += 1
        if _pet_shot_writes % 200 == 0:
            face_shots.prune(state, kind=pet_shots.KIND)
    return jpg


def _video_file(
    state: pathlib.Path, cam: str, start: float, end: float, *, on_decode: Callable[[], None] | None = None
) -> pathlib.Path | None:
    """Prepare one full-scene thumbnail in the background, sharing the crop decode cap."""
    global _video_thumb_writes
    revision = _shot_revision(state)
    saved = video_thumbs.path(state, cam, start, revision)
    if saved.is_file():
        return saved
    if not _shot_decode.acquire(timeout=SHOT_WAIT_S):
        raise ShotsBusy
    try:
        for old in video_thumbs.others(state, cam, start, revision):
            old.unlink(missing_ok=True)
        if on_decode is not None:
            on_decode()
        jpg = video_thumb_jpeg(cam, start, end, turn=False, state=state)
    finally:
        _shot_decode.release()
    if jpg is None:
        return None
    video_thumbs.write(state, cam, start, revision, jpg)
    _video_thumb_writes += 1
    if _video_thumb_writes % VIDEO_THUMB_PRUNE_EVERY == 0:
        threading.Thread(target=video_thumbs.prune, args=(state,), name="video-thumbs-prune", daemon=True).start()
    return saved


def _crop_process(key: face_shots.CropKey) -> face_shots.CropResult | face_shots.CropRetry | None:
    state, kind, sid, _revision = key
    decoded = False

    def on_decode() -> None:
        nonlocal decoded
        decoded = True

    try:
        if kind.endswith("_grid"):
            original = face_shots.path(state, sid, _revision, kind.removesuffix("_grid"))
            if not original.is_file():
                if kind == "shots_grid":
                    _shot_file(sid, state, on_decode=on_decode)
                else:
                    _pet_file(sid, state, on_decode=on_decode)
            if not original.is_file():
                return face_shots.CropResult(False, decoded)
            import cv2
            import numpy as np

            stamp = original.stat().st_mtime_ns
            patch = cv2.imdecode(np.frombuffer(original.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            if patch is None:
                return face_shots.CropResult(False, decoded)
            ok, jpg = cv2.imencode(
                ".jpg",
                cv2.resize(patch, (face_shots.GRID_PX, face_shots.GRID_PX), interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, face_shots.QUALITY],
            )
            if not ok or _shot_revision(state) != _revision:
                return face_shots.CropResult(False, decoded)
            if not original.is_file() or original.stat().st_mtime_ns != stamp:
                return face_shots.CropRetry(decoded)
            face_shots.write(state, sid, _revision, jpg.tobytes(), kind)
            ready = True
        elif kind == "shots":
            ready = _shot_file(sid, state, on_decode=on_decode)[0] is not None
        elif kind == video_thumbs.KIND:
            cam, start, end = json.loads(sid)
            ready = _video_file(state, cam, start, end, on_decode=on_decode) is not None
        else:
            ready = _pet_file(sid, state, on_decode=on_decode) is not None
        return face_shots.CropResult(ready, decoded)
    except ShotsBusy:
        return None
    except TimeoutError:
        LOG.debug("thumbnail frame request timed out; keeping it pending for a backoff retry")
        return face_shots.CropRetry(decoded)
    except Exception:
        LOG.exception("thumbnail could not be prepared")
        return face_shots.CropResult(False, decoded)


_crop_jobs = face_shots.Pending(_crop_process)


def video_thumb_jpeg(
    cam: str, start: float, end: float, *, turn: bool = True, state: pathlib.Path | None = None
) -> bytes | None:
    """A scene thumbnail for a recorded visit, independent of face-photo quality. Try the opening frame, then
    the nearest retained span starts within five seconds of it. At most three fallback decodes. With `turn` False the
    caller already holds a decode turn (the background worker saves what it returns in people/video_thumbs.py)."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", cam) or not math.isfinite(start) or not math.isfinite(end):
        raise ValueError("A camera and finite video bounds are required.")
    if start < 0 or end < start:
        raise ValueError("Video bounds are out of order.")
    import cv2

    _low_priority()
    recording = vision_log.at(context.STATE if state is None else state)
    with _shot_decode if turn else contextlib.nullcontext():
        frame = recording.frame(cam, start)
        if frame is None:
            spans = recording.video_spans(cam, start, min(end + 0.05, start + 5.0))
            for at in sorted({max(start, float(span[0])) for span in spans})[:3]:
                frame = recording.frame(cam, at, within_s=0.05)
                if frame is not None:
                    break
    if frame is None:
        return None
    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        return None
    # Keep the full scene. CSS contains it instead of cropping a square face image a second time.
    scale = min(video_thumbs.WIDTH / width, video_thumbs.HEIGHT / height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    image = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return jpg.tobytes() if ok else None


def _photo_dir(name: str) -> pathlib.Path:
    """<paths.faces>/<name as a safe folder>/ for a person's face crops."""
    return cfg.path("paths.faces") / registry.photo_folder(name)


def _unknown_photo(gid: int | str) -> pathlib.Path:
    return cfg.path("paths.faces") / "_unknown" / f"{int(gid)}.jpg"


def _photos(name: str) -> list[pathlib.Path]:
    """A person's saved face crops, newest first."""
    d = _photo_dir(name)
    return sorted(d.glob("*.jpg"), key=lambda f: f.name, reverse=True) if d.is_dir() else []


@routes.get_when("/faces/unknown/<group>.jpg", lambda p: p.startswith("/faces/unknown/") and p.endswith(".jpg"))
def _get_faces_unknown_photo(h: H) -> None:
    p = h.path.split("?")[0]
    gid = p[len("/faces/unknown/") : -len(".jpg")]
    f = _unknown_photo(gid) if gid.isdigit() else None
    return h._send(200, "image/jpeg", f.read_bytes()) if f and f.exists() else h._send(404, "text/plain", b"")


@routes.get("/people/video-thumb.jpg")
def _get_people_video_thumb_jpg(h: H) -> None:
    return h._video_thumb()


@routes.get_re(r"/faces/shot/([0-9A-Za-z-]{1,64})\.jpg")
def _get_faces_shot_x_jpg(h: H, m: re.Match[str]) -> None:
    if h._query().get("size") == "80":
        return h._grid_photo(m.group(1), "shots_grid")
    try:
        path, etag = _shot_file(m.group(1))
    except ShotsBusy:
        h.send_response(503)
        h.send_header("Retry-After", "2")
        h.send_header("Content-Length", "0")
        h.end_headers()
        return
    if path is None:
        return h._send(404, "text/plain", b"")
    tags = {tag.strip() for tag in h.headers.get("If-None-Match", "").split(",")}
    if etag is not None and tags.intersection({etag, f"W/{etag}", "*"}):
        h.send_response(304)
        h.send_header("Cache-Control", "private, max-age=86400")
        h.send_header("ETag", etag)
        h.end_headers()
        return
    b = path.read_bytes()
    h.send_response(200)
    h.send_header("Content-Type", "image/jpeg")
    # A day: a crop's pixels don't change, and a deleted sighting's is no longer on any page that asks.
    h.send_header("Cache-Control", "private, max-age=86400")
    if etag is not None:
        h.send_header("ETag", etag)
    h.send_header("Content-Length", str(len(b)))
    h.end_headers()
    if h.command != "HEAD":
        h.wfile.write(b)
    return


@routes.get_when("/faces/shot/...", lambda p: p.startswith("/faces/shot/"))
def _get_faces_shot_missing(h: H) -> None:
    return h._send(404, "text/plain", b"")


# the card's photo URLs
@routes.get_re(r"/people/([^/]+)/photos/(\d{10,16})\.jpg")
def _get_people_x_photos_x_jpg(h: H, m: re.Match[str]) -> None:
    nm = urllib.parse.unquote(m.group(1))
    if not call_log._safe_name(nm):
        return h._json({"error": "bad name"}, 400)
    pick = [f for f in _photos(nm) if f.stem == m.group(2)][:1]
    return h._send(200, "image/jpeg", pick[0].read_bytes()) if pick else h._send(404, "text/plain", b"")


@routes.get("/pets/sightings/photo")
def _get_pets_sightings_photo(h: H) -> None:
    return h._pet_photo(h._query().get("track", ""))
