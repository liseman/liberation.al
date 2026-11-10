"""Photos for unnamed faces, cut from the camera's frames.

Each unnamed face group ("Face N", the face worker's cluster on its face.match) gets one photo: the group's best
sighting by quality, cut out of the camera frame it was seen in (`vision_log.at(state).frame`). Runs in a background
thread, a few groups per pass, so People loads at once. Photos go to data/faces/_unknown/<group id>.jpg (the dashboard
already serves that path).

A group's filmstrip on People (sightings_of) lists its recent sightings; the dashboard shows each as its shot
(/faces/shot/<sighting id>.jpg, people/face_shots.py).
"""

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from liberational import config as cfg
from liberational import logs, vision_log
from liberational.vision.sightings import State

DET_MIN = 0.5  # a sighting the detector is less sure of than this is left out of a group's pictures
PER_PASS = 6
STRIP_MAX = 24  # thumbnails in a face's filmstrip
PAD_PX = 12  # the photo takes this much of the frame around the face box
_busy = threading.Lock()
_last = {"t": 0.0}
# {face group: its count when its photo couldn't be cut} (its frame gone, or its box too small). Tried again only once
# the group has been seen again: each pass with one left read SHOT_DAYS of sightings, about 54,000 records every two
# minutes, for groups it could never cut (#1516).
_failed: dict[int, int] = {}


LOG = logs.get("face-photos")


def _dest(gid: int | np.integer) -> Path:
    # These photos can become chosen identity photos on enrollment/undo, so geometry cache revisions must not
    # silently replace them. Only disposable sighting shot caches are versioned by the dashboard.
    return cfg.path("paths.faces") / "_unknown" / f"{int(gid)}.jpg"


def _group(cluster: object) -> int | None:
    """The face group a cluster label names ("Face 7" is 7), or None."""
    if not isinstance(cluster, str) or not cluster.startswith("Face "):
        return None
    try:
        return int(cluster[5:])
    except ValueError:
        return None


def _unnamed(state: State = None, *, rows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """The unnamed sightings with a box the detector is sure of over the last SHOT_DAYS, newest first."""
    from liberational.vision import sightings as S

    now = time.time()
    found = S.faces(now - S.SHOT_DAYS * 86400, now + 1, state=state) if rows is None else rows
    keep = [f for f in found if f["name"] is None and f["bbox_norm"] and (f["det"] or 0.0) > DET_MIN]
    return sorted(keep, key=lambda f: (f["ts"], f["id"]), reverse=True)


def _best_sighting(gid: int, unnamed: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Group gid's best sighting by quality among `unnamed` (the newest of equals), or None when it has none."""
    found = [f for f in unnamed if _group(f["cluster"]) == gid]
    return max(found, key=lambda f: float(f["quality"] or 0.0)) if found else None


def _cut(row: dict[str, Any], dest: Path, state: State = None) -> bool:
    """Crops sighting `row` (its box, PAD_PX around) from its camera frame to a 160x160 JPEG at dest."""
    import cv2

    evidence = vision_log.at(state).frame_evidence(str(row["cam"]), float(row["ts"]))
    frame = evidence.image if evidence is not None else None
    if evidence is None or frame is None:  # the broker no longer holds the frame
        return False
    h, w = frame.shape[:2]
    from liberational.people.face_shots import in_frame
    from liberational.see.frames import source_box_to_decoded

    inside = in_frame(row["bbox_norm"])
    box = source_box_to_decoded(inside, evidence.metadata) if inside is not None else None
    if box is None:
        return False
    nx1, ny1, nx2, ny2 = box
    x1, y1 = max(0, int(nx1 * w) - PAD_PX), max(0, int(ny1 * h) - PAD_PX)
    x2, y2 = min(w, int(nx2 * w) + PAD_PX), min(h, int(ny2 * h) + PAD_PX)
    if x2 - x1 < 16 or y2 - y1 < 16:
        return False
    face = cv2.resize(frame[y1:y2, x1:x2], (160, 160), interpolation=cv2.INTER_LANCZOS4)
    dest.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(dest), face, [cv2.IMWRITE_JPEG_QUALITY, 90])) and dest.exists()


def sightings_of(gid: int, limit: int = STRIP_MAX, state: State = None, *, offset: int = 0) -> list[dict[str, Any]]:
    """Unnamed sightings of face group gid (the face worker's "Face <gid>"), newest first: [{"id", "ts"}],
    `limit` of them from `offset`."""
    return page_of(gid, offset, limit, state)[0]


def page_of(gid: int, offset: int, limit: int, state: State = None) -> tuple[list[dict[str, Any]], int]:
    """A page of face group gid's unnamed sightings, newest first, and how many it has in all."""
    found = [f for f in _unnamed(state) if _group(f["cluster"]) == int(gid)]
    page = found[max(0, offset) : max(0, offset) + max(0, limit)]
    return [{"id": f["id"], "ts": float(f["ts"])} for f in page], len(found)


KEPT_VECTORS = 256  # a group's embedding is rebuilt from at most this many of its newest kept shots (#2008)


def kept_embeddings(gid: int, limit: int = KEPT_VECTORS, state: State = None) -> np.ndarray:
    """The unit embeddings (N x 512) of face group gid's newest `limit` unnamed sightings, from their `face.embedding`
    rows: one read per hour the sightings fall in, so a click costs a few reads, not the shot window's."""
    from liberational import segment_log

    mine = [f for f in _unnamed(state) if _group(f["cluster"]) == int(gid)][:limit]
    wanted = {str(f["id"]) for f in mine}
    hours = sorted({int(float(f["ts"]) // 3600) for f in mine})
    log = segment_log.at(state)
    vectors = [
        r["vector"]
        for h in hours
        for r in log.range(segment_log.FACE_EMBEDDINGS, h * 3600.0, (h + 1) * 3600.0)
        if str(r.get("match_id")) in wanted
    ]
    return np.asarray(vectors, np.float32).reshape(-1, 512)


def recent_by_group(
    per: int = 4, scan: int = 8000, state: State = None, *, rows: list[dict[str, Any]] | None = None
) -> dict[int, list[dict[str, Any]]]:
    """Each unnamed face group's newest sightings among the newest `scan` unnamed ones, newest first: {gid: [{"id",
    "ts", "cam"}]}, up to `per` each, all groups in one read, so People's pickers cost one read however many groups
    there are. `rows` may supply the complete SHOT_DAYS projection, including named faces; None reads it,
    while [] is a known empty snapshot."""
    out: dict[int, list[dict[str, Any]]] = {}
    for f in _unnamed(state, rows=rows)[: int(scan)]:
        gid = _group(f["cluster"])
        if gid is None:
            continue
        got = out.setdefault(gid, [])
        if len(got) < per:
            got.append({"id": f["id"], "ts": float(f["ts"]), "cam": f["cam"]})
    return out


def refresh(max_groups: int = PER_PASS) -> int:
    """Give up to max_groups unnamed face groups without a photo one cut from a camera frame. Returns how many were
    added."""
    from liberational.vision import faces as fid

    ids, _vecs, counts, _ = fid._unk_load()
    todo = [
        (int(g), int(counts[i]))
        for i, g in enumerate(ids)
        if counts[i] >= 2 and not _dest(g).exists() and _failed.get(int(g)) != int(counts[i])
    ]
    todo.sort(key=lambda t: -t[1])
    added = 0
    unnamed = _unnamed() if todo else []
    for gid, n in todo[:max_groups]:
        row = _best_sighting(gid, unnamed)
        if row is not None and _cut(row, _dest(gid)):
            added += 1
            _failed.pop(gid, None)
        else:
            _failed[gid] = n
    return added


def refresh_async(min_gap_s: float = 120) -> None:
    """Run refresh() in the background at most every min_gap_s seconds."""
    if time.time() - _last["t"] < min_gap_s or _busy.locked():
        return

    def run() -> None:
        with _busy:
            _last["t"] = time.time()
            try:
                refresh()
            except Exception as ex:  # never let a photo pass break the People page
                LOG.error(f"failed: {ex}")

    threading.Thread(target=run, name="face-photos", daemon=True).start()
