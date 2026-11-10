"""Object memory on Continuum (#778): objects the owner names and crops on the Objects page, found again in the
cameras, and "where did I leave my wallet?" answered from where each was last seen.

- An object is an event on the `objects` stream (`ObjectRemembered`, `ObjectForgotten`), through the registry's
  event log: its name, aliases, note, and template image (a small JPEG, inline).
- A sighting is the object, the camera, its box, and the frame it was seen in as a small inline JPEG (`evidence`), so
  it outlives the frame's recompression. The owner's own crop is the first (`manual_label`), an `object.sighting`
  record in the knowledge store (taught state, outside the broker's reset); the object worker (`see.object_worker`) adds the rest (`visual_match`) as `object.match`
  rows in the segment store (#1791). `sightings` reads both.
- A snapshot (the view the owner crops from) is the camera's newest frame, kept in this process for SNAPSHOT_TTL.

Matching is v1's (`memory/object_memory.py`, #66): CPU SIFT features and a RANSAC homography, confirmed in two
distinct frames in a row; no person recognition.
"""

import base64
import math
import re
import secrets
import threading
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from cv2.typing import MatLike

from liberational import config as cfg
from liberational import knowledge, segment_log
from liberational.domain.kernel import Event, EventType, thaw

STREAM = "objects"  # the event stream objects are remembered and forgotten on
SNAPSHOT_TTL = 600.0  # how long a snapshot can be cropped from
MAX_SNAPSHOTS = 30
FRESH_SECONDS = 30.0  # a sighting or a frame older than this is stale
TEMPLATE_BYTES = 40_000  # a template image's largest JPEG (an event stays well under the 64 KiB inline cap)
EVIDENCE_WIDTH = 480  # a sighting's evidence frame is scaled to this width
EVIDENCE_BYTES = 40_000
LOOK_BACK_S = (86_400.0, 30 * 86_400.0)  # last seen: the last day first, then the last month
PERSON_WORDS = re.compile(r"\b(person|people|face|man|woman|child|human)\b", re.I)


def camera_names() -> list[str]:
    """The cameras in capture's manifest."""
    from liberational.see.frames import cameras_from_manifest

    return [c.name for c in cameras_from_manifest(cfg.ROOT / "config" / "devices.toml")]


# ---- matching (v1's) ---------------------------------------------------------------------------------------------


def features(image: MatLike, mask: MatLike | None = None) -> tuple[Sequence[cv2.KeyPoint], MatLike | None]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.SIFT.create(nfeatures=1200).detectAndCompute(gray, mask)


def template_mask(row: Mapping[str, Any], image: MatLike) -> np.ndarray | None:
    """A saved SAM2 template's mask, aligned to the decoded JPEG; old rectangular templates have none."""
    segmentation = row.get("segmentation")
    if not segmentation:
        return None
    raw = base64.b64decode(segmentation["mask"], validate=True)
    mask = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != image.shape[:2] or not mask.any():
        raise ValueError("the saved object segmentation mask is invalid")
    return mask


def masked_template(image: MatLike, mask: np.ndarray) -> tuple[bytes, bytes]:
    """A masked JPEG and aligned lossless mask, together small enough for one object event."""
    mask = (mask.astype(bool) * 255).astype(np.uint8)
    image = np.asarray(image).copy()
    image[mask == 0] = 0
    while True:
        encoded = jpeg(image, max_bytes=TEMPLATE_BYTES - 8_000)
        decoded = decode(encoded)
        if decoded is None:
            raise ValueError("object template could not be encoded")
        aligned = cv2.resize(mask, (decoded.shape[1], decoded.shape[0]), interpolation=cv2.INTER_NEAREST)
        ok, png = cv2.imencode(".png", aligned, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if ok and len(png) <= 8_000:
            return encoded, png.tobytes()
        h, w = image.shape[:2]
        if min(h, w) <= 32:
            raise ValueError("the object mask is too detailed to save")
        size = (max(24, w * 3 // 4), max(24, h * 3 // 4))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)


def match(
    template: tuple[Sequence[cv2.KeyPoint], MatLike | None, tuple[int, int]],
    frame: tuple[Sequence[cv2.KeyPoint], MatLike | None, tuple[int, int]],
) -> tuple[list[float], float] | None:
    """Where a template's object is in a frame, as a normalized [x, y, w, h] box and the inlier share, or None. Each
    side is (keypoints, descriptors, (height, width)), computed once."""
    kp1, d1, (h, w) = template
    kp2, d2, (H, W) = frame
    if d1 is None or d2 is None or len(d1) < 18 or len(d2) < 18:
        return None
    pairs = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [a for pair in pairs if len(pair) == 2 for a, b in [pair] if a.distance < 0.68 * b.distance]
    if len(good) < 14 or len({m.trainIdx for m in good}) < 14:
        return None
    src = np.array([kp1[m.queryIdx].pt for m in good], dtype=np.float32)
    dst = np.array([kp2[m.trainIdx].pt for m in good], dtype=np.float32)
    matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if matrix is None or mask is None:
        return None
    keep = mask.ravel().astype(bool)
    if keep.sum() < 12 or keep.mean() < 0.65:
        return None
    if cv2.contourArea(cv2.convexHull(src[keep])) / (h * w) < 0.12:
        return None
    corners = cv2.perspectiveTransform(np.array([[[0, 0], [w, 0], [w, h], [0, h]]], dtype=np.float32), matrix)[0]
    if not np.isfinite(corners).all() or not cv2.isContourConvex(corners):
        return None
    if corners[:, 0].min() < 0 or corners[:, 1].min() < 0 or corners[:, 0].max() > W or corners[:, 1].max() > H:
        return None
    area = cv2.contourArea(corners)
    edges = np.linalg.norm(corners - np.roll(corners, 1, axis=0), axis=1)
    if not 0.2 <= area / (w * h) <= 5 or edges.min() < 16 or edges.max() / edges.min() > 8:
        return None
    x, y = corners.min(axis=0)
    right, bottom = corners.max(axis=0)
    return [float(x / W), float(y / H), float((right - x) / W), float((bottom - y) / H)], float(keep.mean())


def consistent(a: Sequence[float], b: Sequence[float]) -> bool:
    """Whether two [x, y, w, h] boxes overlap enough (IoU 0.55) to be the same place."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    inter = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
    return inter / max(aw * ah + bw * bh - inter, 1e-9) >= 0.55


def jpeg(image: MatLike, *, max_bytes: int, width: int | None = None) -> bytes:
    """The image as a JPEG under max_bytes: scaled to `width` when given (never up), then lower quality, then
    smaller, until it fits."""
    if width and image.shape[1] > width:
        image = cv2.resize(image, (width, round(image.shape[0] * width / image.shape[1])), interpolation=cv2.INTER_AREA)
    while True:
        for quality in (90, 80, 70, 60, 50):
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ok and len(encoded) <= max_bytes:
                return encoded.tobytes()
        h, w = image.shape[:2]
        if min(h, w) < 32:
            raise ValueError("image can't be made small enough")
        image = cv2.resize(image, (w * 3 // 4, h * 3 // 4), interpolation=cv2.INTER_AREA)


def decode(data: bytes) -> MatLike | None:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


# ---- snapshots ---------------------------------------------------------------------------------------------------


@dataclass
class Snapshot:
    token: str
    camera: str
    captured_at: float
    image: bytes  # the frame as a JPEG
    frame_id: str | None = None
    made: float = field(default_factory=time.time)
    live_crop: bool = False  # set only by the validated Live import endpoint
    full_frame: bool = False  # native recorded Review import; still requires a segmented selection
    frame_metadata: dict[str, Any] | None = None

    def meta(self) -> dict[str, Any]:
        age = time.time() - self.captured_at
        return {
            **({"frame_metadata": self.frame_metadata} if self.frame_metadata is not None else {}),
            "token": self.token,
            "evidence": f"snapshot:{self.token}",
            "camera": self.camera,
            "captured_at": self.captured_at,
            "stale": age > FRESH_SECONDS or age < -2,
            "source": "full_frame" if self.full_frame else "live_crop" if self.live_crop else "camera",
        }


class Snapshots:
    """The views the owner can crop from, newest MAX_SNAPSHOTS, each for SNAPSHOT_TTL; and the draft a spoken
    "remember this as my wallet" left (a snapshot and a name)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._held: dict[str, Snapshot] = {}
        self.draft: tuple[Snapshot, str, float] | None = None

    def add(
        self,
        camera: str,
        captured_at: float,
        image: bytes,
        frame_id: str | None = None,
        *,
        live_crop: bool = False,
        full_frame: bool = False,
        frame_metadata: dict[str, Any] | None = None,
    ) -> Snapshot:
        s = Snapshot(
            secrets.token_hex(16),
            camera,
            captured_at,
            image,
            frame_id,
            live_crop=live_crop,
            full_frame=full_frame,
            frame_metadata=frame_metadata,
        )
        with self._lock:
            now = time.time()
            self._held = {t: h for t, h in self._held.items() if now - h.made <= SNAPSHOT_TTL}
            self._held[s.token] = s
            while len(self._held) > MAX_SNAPSHOTS:
                self._held.pop(next(iter(self._held)))
        return s

    def get(self, token: str) -> Snapshot | None:
        with self._lock:
            s = self._held.get(token)
        return s if s and time.time() - s.made <= SNAPSHOT_TTL else None


# ---- the store ---------------------------------------------------------------------------------------------------


def _aliases(aliases: str | Iterable[str] | None) -> list[str]:
    if isinstance(aliases, str):
        aliases = [a.strip() for a in aliases.split(",") if a.strip()]
    out = [a.strip() for a in aliases or [] if isinstance(a, str)]
    if len(out) > 20 or any(not a or len(a) > 80 for a in out):
        raise ValueError("aliases must be up to 20 short strings")
    return out


def _pixel_edge(value: float, size: int) -> int:
    """Floor fractional pixels without losing exact integer edges to floating-point noise."""
    return int(round(value * size, 9))


def _box(bbox: object) -> tuple[float, float, float, float]:
    if not isinstance(bbox, list | tuple) or len(bbox) != 4:
        raise ValueError("bbox must be normalized [x,y,w,h]")
    x, y, w, h = (float(v) for v in bbox)
    if not all(math.isfinite(v) for v in (x, y, w, h)) or min(x, y) < 0 or min(w, h) <= 0 or x + w > 1 or y + h > 1:
        raise ValueError("bbox outside image")
    return x, y, w, h


def match_record(
    *,
    object_id: str,
    name: str,
    cam: str,
    at: float,
    bbox: Sequence[float],
    score: float,
    frame: MatLike,
    frame_id: str | None,
    frame_seq: int | None,
) -> dict[str, Any]:
    """The object worker's visual match as an `object.match` segment record, with the frame as its evidence."""
    return {
        "id": str(uuid.uuid7()),
        "cam": cam,
        "ts_start": at,
        "object_id": object_id,
        "name": name,
        "bbox": [float(v) for v in bbox],
        "score": float(score),
        "frame_id": frame_id,
        "frame_seq": frame_seq,
        "evidence": base64.b64encode(jpeg(frame, max_bytes=EVIDENCE_BYTES, width=EVIDENCE_WIDTH)).decode(),
    }


SIGHTINGS = "object.sighting"  # the owner's crops, in the knowledge store


def _ns(seconds: float) -> int:
    return max(0, round(seconds * 1e9))


def _knowledge(state: str | Path) -> knowledge.Store:
    from liberational.people import registry

    return knowledge.at(registry.knowledge_path(state))


def sightings(state: str | Path, start: float, end: float) -> list[dict[str, Any]]:
    """Every sighting in [start, end), in time order: the owner's crops from the knowledge store and the object
    worker's matches from the segment store."""
    labelled = [
        {**row.record.value, "id": row.record.id}
        for row in _knowledge(state).query(SIGHTINGS, start_ns=_ns(start), end_ns=max(_ns(end), _ns(start) + 1))
    ]
    matched = segment_log.at(state).range(segment_log.OBJECT_MATCHES, start, end)
    return sorted([*labelled, *matched], key=lambda r: (r["ts_start"], r["id"]))


class Store:
    """Object memory for a pack's state directory: its objects (events) and their sightings."""

    def __init__(self, state: str | Path) -> None:
        self.state = Path(state)

    # objects

    def _events(self) -> list[Event]:
        from liberational.people import registry

        stored = registry.at(self.state).events(
            [EventType.OBJECT_REMEMBERED, EventType.OBJECT_FORGOTTEN], 0.0, 1e12, streams=(STREAM,)
        )
        return [s.event for s in stored]

    def objects(self) -> list[dict[str, Any]]:
        """Every object remembered and not forgotten, oldest first, each with its template (base64 JPEG)."""
        held: dict[str, dict[str, Any]] = {}
        for e in self._events():
            p = thaw(e.payload)
            oid = str(p.get("object_id", ""))
            if e.type == EventType.OBJECT_FORGOTTEN:
                held.pop(oid, None)
            elif oid:
                held[oid] = {**p, "id": oid, "created": e.occurred_at}
        return list(held.values())

    def remember(
        self,
        *,
        name: str,
        snapshot: Snapshot,
        bbox: object,
        note: str = "",
        aliases: str | Iterable[str] | None = None,
        mask: np.ndarray | None = None,
        segmentation_model: str | None = None,
        segmentation_score: float | None = None,
    ) -> dict[str, Any]:
        """Remembers the object the owner cropped from a snapshot; returns it as list_objects has it."""
        name = str(name).strip()
        note = str(note)
        if not name or len(name) > 80 or len(note) > 2000:
            raise ValueError("name required (max 80 characters); note max 2000")
        if PERSON_WORDS.search(name):
            raise ValueError("object memory does not support people or faces")
        alias_list = _aliases(aliases)
        x, y, w, h = _box(bbox)
        if time.time() - snapshot.made > SNAPSHOT_TTL:
            raise ValueError("snapshot expired; capture another preview")
        labels = {s.casefold() for o in self.objects() for s in [o["name"], *o.get("aliases", [])]}
        if labels.intersection(s.casefold() for s in [name, *alias_list]):
            raise ValueError("name or alias already enrolled")
        image = decode(snapshot.image)
        if image is None:
            raise ValueError("the snapshot isn't an image")
        H, W = image.shape[:2]
        x1, y1, x2, y2 = _pixel_edge(x, W), _pixel_edge(y, H), _pixel_edge(x + w, W), _pixel_edge(y + h, H)
        if x2 - x1 < 24 or y2 - y1 < 24:
            raise ValueError("crop must be at least 24px")
        from liberational.see.frames import decoded_box_to_source, source_box_to_decoded

        source_bbox = [x, y, w, h]
        bbox_space = (
            "decoded" if (snapshot.full_frame or snapshot.live_crop) and snapshot.frame_metadata is None else "source"
        )
        if snapshot.frame_metadata is not None:
            mapped = decoded_box_to_source([x, y, x + w, y + h], snapshot.frame_metadata)
            if mapped is None:
                raise ValueError("The crop reaches outside the recorded source image")
            source_bbox = [mapped[0], mapped[1], mapped[2] - mapped[0], mapped[3] - mapped[1]]
        for face in self._faces_near(snapshot.camera, snapshot.captured_at):
            if bbox_space == "decoded":
                raise ValueError("Recapture this picture to verify that the crop excludes faces")
            if snapshot.frame_metadata is not None:
                face = source_box_to_decoded(face, snapshot.frame_metadata)
                if face is None:
                    raise ValueError("The recorded source geometry is unavailable")
            fx1, fy1, fx2, fy2 = face
            if min(x2, fx2 * W) > max(x1, fx1 * W) and min(y2, fy2 * H) > max(y1, fy1 * H):
                raise ValueError("crop overlaps a face; choose only the object")
        crop = image[y1:y2, x1:x2]
        segmentation = None
        if mask is not None:
            if mask.shape != image.shape[:2] or not mask.any() or not segmentation_model:
                raise ValueError("the SAM2 selection mask is invalid")
            template, saved_mask = masked_template(crop, mask[y1:y2, x1:x2])
            crop = decode(template)
            if crop is None:
                raise ValueError("the object template is invalid")
            segmentation = {
                "model": segmentation_model,
                "score": segmentation_score,
                "mask": base64.b64encode(saved_mask).decode(),
            }
            keypoint_mask = template_mask({"segmentation": segmentation}, crop)
        else:
            template = jpeg(crop, max_bytes=TEMPLATE_BYTES)
            keypoint_mask = None
        kp, desc = features(crop, keypoint_mask)
        mode = "visual" if desc is not None and len(kp) >= 18 else "manual_only"
        oid = uuid.uuid4().hex
        payload = {
            "object_id": oid,
            "name": name,
            "aliases": alias_list,
            "note": note,
            "mode": mode,
            "template": base64.b64encode(template).decode(),
            "camera": snapshot.camera,
            "bbox": source_bbox,
            "bbox_space": bbox_space,
        }
        if segmentation is not None:
            payload["segmentation"] = segmentation
        from liberational.people import registry

        registry.at(self.state).log.append(
            [Event.new(EventType.OBJECT_REMEMBERED, stream=STREAM, source="objects-page", payload=payload)]
        )
        self.sighted(
            object_id=oid,
            name=name,
            cam=snapshot.camera,
            at=snapshot.captured_at,
            bbox=source_bbox,
            frame_metadata=snapshot.frame_metadata,
            bbox_space=bbox_space,
            score=None,
            kind="manual_label",
            frame=image,
            frame_id=snapshot.frame_id,
        )
        return next(o for o in self.list_objects() if o["id"] == oid)

    def forget(self, object_id: str) -> None:
        if not any(o["id"] == object_id for o in self.objects()):
            raise ValueError("no such object")
        from liberational.people import registry

        registry.at(self.state).log.append(
            [
                Event.new(
                    EventType.OBJECT_FORGOTTEN, stream=STREAM, source="objects-page", payload={"object_id": object_id}
                )
            ]
        )

    def _faces_near(self, cam: str, at: float) -> list[list[float]]:
        """Face boxes (normalized x1, y1, x2, y2) on the camera within a second of `at`."""
        out: list[list[float]] = []
        try:
            from liberational.see import face_boxes  # imports continuum_py

            ns = int(at * 1e9)
            for r in face_boxes.near(self.state, ns - 10**9, ns + 10**9):
                if r.get("cam") == cam:
                    out += [f["bbox"] for f in r.get("faces") or [] if isinstance(f.get("bbox"), list)]
        except Exception:  # the broker out of reach: no faces to keep the crop off
            return []
        return out

    # sightings

    def sighted(
        self,
        *,
        object_id: str,
        name: str,
        cam: str,
        at: float,
        bbox: Sequence[float],
        score: float | None,
        kind: str,
        frame: MatLike,
        frame_id: str | None,
        frame_seq: int | None = None,
        frame_metadata: dict[str, Any] | None = None,
        bbox_space: str = "source",
    ) -> None:
        """Commits a sighting, with the frame as its evidence."""
        record = {
            **({"frame_metadata": frame_metadata} if frame_metadata is not None else {}),
            "cam": cam,
            "bbox_space": bbox_space,
            "frame_seq": frame_seq,
            "object_id": object_id,
            "name": name,
            "bbox": [float(v) for v in bbox],
            "score": score,
            "kind": kind,
            "evidence": base64.b64encode(jpeg(frame, max_bytes=EVIDENCE_BYTES, width=EVIDENCE_WIDTH)).decode(),
            "ts_start": at,
        }
        ident, at_ns = str(uuid.uuid7()), _ns(at)
        value = {**record, "frame_id": frame_id}
        _knowledge(self.state).put([knowledge.Record(ident, SIGHTINGS, object_id, at_ns, at_ns, time.time_ns(), value)])

    def sightings(self, start: float, end: float) -> list[dict[str, Any]]:
        return sightings(self.state, start, end)

    def recent(self, object_ids: Iterable[str], limit: int = 1) -> dict[str, list[dict[str, Any]]]:
        """Each object's newest sightings (up to `limit`, newest first), from the last day, else the last month."""
        wanted = set(object_ids)
        found: dict[str, list[dict[str, Any]]] = {o: [] for o in wanted}
        now = time.time()
        for back in LOOK_BACK_S:
            missing = {o for o, s in found.items() if len(s) < limit}
            if not missing:
                break
            seen = sorted(self.sightings(now - back, now + 1), key=lambda r: (r["ts_start"], r["id"]), reverse=True)
            found = {o: [r for r in seen if r.get("object_id") == o][:limit] for o in wanted}
        return found

    @staticmethod
    def as_sighting(r: Mapping[str, Any]) -> dict[str, Any]:
        """A sighting record as the Objects page shows one."""
        bbox = r.get("bbox")
        metadata = r.get("frame_metadata")
        if metadata is not None and bbox is not None and r.get("bbox_space", "source") == "source":
            from liberational.see.frames import source_box_to_decoded

            x, y, w, h = bbox
            mapped = source_box_to_decoded([x, y, x + w, y + h], metadata)
            bbox = [mapped[0], mapped[1], mapped[2] - mapped[0], mapped[3] - mapped[1]] if mapped else None
        return {
            "camera": r.get("cam"),
            "captured_at": r["ts_start"],
            "kind": r.get("kind"),
            "evidence": f"sighting:{r['id']}",
            "bbox": bbox,
            "stale": time.time() - r["ts_start"] > FRESH_SECONDS,
        }

    def list_objects(self) -> list[dict[str, Any]]:
        objects = self.objects()
        last = self.recent([o["id"] for o in objects])
        return [
            {
                "id": o["id"],
                "name": o["name"],
                "aliases": o.get("aliases", []),
                "note": o.get("note", ""),
                "template": f"template:{o['id']}",
                "mode": o.get("mode", "manual_only"),
                "last_seen": self.as_sighting(last[o["id"]][0]) if last.get(o["id"]) else None,
            }
            for o in objects
        ]

    def evidence(self, ref: str) -> bytes:
        """The JPEG an evidence reference names: `template:<object id>` or `sighting:<record id>` (looked for in the
        last month)."""
        kind, _, key = ref.partition(":")
        if kind == "template":
            o = next((o for o in self.objects() if o["id"] == key), None)
            if o is None:
                raise FileNotFoundError(ref)
            return base64.b64decode(o["template"])
        if kind == "sighting":
            now = time.time()
            for back in LOOK_BACK_S:
                r = next((r for r in self.sightings(now - back, now + 1) if r["id"] == key), None)
                if r is not None:
                    return base64.b64decode(r["evidence"])
        raise FileNotFoundError(ref)

    def query(self, question: str) -> dict[str, Any]:
        """Where an object named in the question was last seen, as v1 answered it."""
        q = str(question).casefold()
        objects = self.objects()
        matches = [
            o
            for o in objects
            if any(
                re.search(r"(?<!\w)" + re.escape(s.casefold()) + r"(?!\w)", q)
                for s in [o["name"], *o.get("aliases", [])]
            )
        ]
        if len(matches) != 1:
            return {
                "status": "ambiguous" if matches else "unknown",
                "answer": "Which object do you mean?"
                if matches
                else "I do not have evidence for that object. Name and crop it in a camera preview first.",
                "objects": [o["name"] for o in matches],
            }
        obj = matches[0]
        history = [self.as_sighting(r) for r in self.recent([obj["id"]], limit=10)[obj["id"]]]
        if not history:
            return {"status": "unknown", "answer": f"I haven't seen {obj['name']} in the last month.", "objects": []}
        seen = history[0]
        when = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(seen["captured_at"]))
        kind = "You manually labeled" if seen["kind"] == "manual_label" else "A tentative visual match last showed"
        answer = (
            f"{kind} {obj['name']} in {seen['camera']} at {when}. "
            "I cannot confirm it is still there or infer a room/location from the camera."
        )
        if obj.get("mode") == "manual_only":
            answer += " This low-texture object is manual-only."
        return {
            "status": "last_seen",
            "answer": answer,
            "evidence": seen,
            "history": history,
            "uncertainty": "Camera view only; visual matches can be wrong. No current-presence guarantee.",
        }
