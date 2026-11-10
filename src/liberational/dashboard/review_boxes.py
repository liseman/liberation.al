"""Boxes for Review's large camera: the people and things one camera saw during a short stretch, for drawing over
the recorded video at the playhead (GET /review/boxes?cam=&start=&end=).

- Faces: the face worker's sightings with a box (vision/sightings.py, on the vision log), named by who they are now
  (the event log), else unknown.
- People's bodies: see-actions' `pose.track` boxes, which it stores for every tracked person on every analyzed
  frame (cam0 at 30 fps, the others at 10), read at most one a track every BODY_STRIDE_S so a minute of a busy
  camera stays a small response. A body is named by the majority of the window's known face matches on its pose
  track (see/head_faces.py writes each look's `track_id` as the pose track), else unknown. Faces still come back as
  their own boxes: Review draws them only for people without a body box (older data), and naming an unknown body
  crops the face of its track's latest look.
- Pets: see-actions' `animal.detection` source-frame boxes, keyed by their original animal track. Exact owner
  `pet.sighting` labels supply the current pet profile's name/species. A track without one is named only when
  appearance recognition (people/pet_recognition.py `resolve`) recognizes it as a pet from this window's own
  embeddings, with the Pets page's check and threshold; it never reads history. Other tracks keep the detected
  species.
- Things: object memory sightings (the owner's crops and the object worker's matches, memory/objects.py), by name.
- Labels: the caption worker's groundings (`camera.objects`, see/captions.py), the generic "chair" and "car" boxes Live
  draws. A camera grounds every several seconds and most records are cached copies of the last grounding, so each
  record's boxes carry "until": the camera's next record, at most LABEL_HOLD_S later. A record without boxes clears
  them. Records from LABEL_HOLD_S before the window are read too, so the window opens with what was standing then.

Every read is bounded: at most WINDOW_MAX_S of one camera, by time (the vision log's faces over the window), and at most LIMIT rows a source. Object memory holds only the objects someone asked it to remember, so its
small table is read by camera and time without an index of its own.

A face is named by who that sighting was then (the event log, as of its time). An unknown face keeps one id
while it stays put across samples (overlap), so its box moves rather than blinking out and in.

A box is {"ts", "id", "kind": "person" | "face" | "pet" | "object" | "label", "box": [x1, y1, x2, y2] as fractions
of the picture, "label", "tone": "known" | "unknown", "name"?, "group"?, "track"?, "until"?} ("until": a label's last
moment) (group: an unknown face's face group; track: its original evidence track, never the spatially smoothed face
box id; a person's track is its pose track, the same string its faces carry). Pets also carry "species" and,
when an exact owner label names an active pet, or recognition does, "pet_id" and "name"; a recognized one also has
"how": "recognized" and its "similarity". A pet's box id stays per track even when named.
"""

import math
import pathlib
import re
from collections import Counter
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from liberational import logs
from liberational.dashboard import routes
from liberational.people import registry, transcript_view
from liberational.vision import sightings, views

if TYPE_CHECKING:
    from liberational.dashboard.server import H

WINDOW_MAX_S = 120.0  # the longest stretch one request reads
LIMIT = 5000  # rows a source, at most
TRACK_IOU = 0.3  # an unknown face overlapping the last one this much is the same face
TRACK_GAP_S = 2.0  # and no longer apart than this
LABEL_HOLD_S = 15.0  # a grounding's boxes stand at most this long without a newer record (cameras ground every ~10 s)
BODY_STRIDE_S = 0.1  # a person's body box at most this often: 10 a second, which Review interpolates between
NS = 1_000_000_000
LOG = logs.get("review-boxes")


def _clip(v: float) -> float:
    return min(1.0, max(0.0, float(v)))


def _box(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    return [round(_clip(x1), 4), round(_clip(y1), 4), round(_clip(x2), 4), round(_clip(y2), 4)]


def _faces(state: pathlib.Path, cam: str, start: float, end: float) -> Iterator[dict[str, Any]]:
    found = [f for f in sightings.faces(start, end, cam, state) if f["bbox_norm"]][:LIMIT]
    if not found:
        return
    view = transcript_view.View(state)  # who each sighting is now: a rename since applies
    for f in found:
        ts, name = float(f["ts"]), f["name"]
        who = view.current(name, ts) if isinstance(name, str) else None  # who it was then, after a rename
        known = registry.is_name(who)
        track = f.get("track_id")
        yield {
            "ts": ts,
            "id": f"face:{str(who).strip().lower()}" if known else f"face:#{f['id']}",
            "kind": "face",
            "box": _box(*f["bbox_norm"][:4]),
            "label": str(who).strip() if known else "Who's this?",
            "tone": "known" if known else "unknown",
            **({"track": track} if isinstance(track, str) and track.strip() else {}),
            **({"name": str(who).strip()} if known else _group(f.get("cluster"))),
        }


def _bodies(
    state: pathlib.Path, cam: str, start: float, end: float, faces: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """The camera's tracked people in the window as body boxes, one a track every BODY_STRIDE_S, named by the
    majority of the known faces on their pose track. [] when the segment store can't be read."""
    from liberational import segment_log
    from liberational.see.actions import POSE_SCHEMA
    from liberational.see.actions_worker import POSES

    try:
        from continuum_py import read_segments  # the native store; a machine without it shows faces only

        rows = read_segments(segment_log.root(state), POSES, POSE_SCHEMA, round(start * NS), round(end * NS), [cam])
    except Exception as error:  # no pose segments here, or the store unreadable: faces still show
        LOG.debug("Review's body boxes unavailable for %s: %s", cam, error)
        return []
    votes: dict[str, Counter[str]] = {}
    group: dict[str, int] = {}
    for f in faces:  # in time order, so a track's group is its latest unknown face's
        track = f.get("track")
        if track is None:
            continue
        if f["tone"] == "known":
            votes.setdefault(track, Counter())[f["name"]] += 1
        elif "group" in f:
            group[track] = f["group"]
    order = sorted(range(len(rows["ts_ns"])), key=lambda i: (int(rows["ts_ns"][i]), int(rows["track_id"][i])))
    kept: dict[str, int] = {}  # each track's last kept sample, in ns
    stride = round(BODY_STRIDE_S * NS)
    out: list[dict[str, Any]] = []
    for i in order:
        ns, track = int(rows["ts_ns"][i]), str(int(rows["track_id"][i]))
        if track in kept and ns - kept[track] < stride - stride // 10:  # a tenth's slack: 30 fps keeps every third
            continue
        try:
            coords = [float(v) for v in rows["box"][i][:4]]
        except TypeError, ValueError:
            continue
        if len(coords) != 4 or not all(math.isfinite(v) for v in coords):
            continue
        box = _box(*coords)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        kept[track] = ns
        named = votes.get(track)
        name = named.most_common(1)[0][0] if named else None
        out.append(
            {
                "ts": ns / NS,
                "id": f"person:{track}",
                "kind": "person",
                "box": box,
                "label": name or "Who's this?",
                "tone": "known" if name else "unknown",
                "track": track,
                **({"name": name} if name else {"group": group[track]} if track in group else {}),
            }
        )
        if len(out) >= LIMIT:
            break
    return out


def _group(cluster: object) -> dict[str, int]:
    """An unknown face's face group ("Face 12" is 12), which Review's Who's this? names through /people/claim."""
    m = re.fullmatch(r"Face (\d+)", cluster) if isinstance(cluster, str) else None
    return {"group": int(m.group(1))} if m else {}


def _objects(state: pathlib.Path, cam: str, start: float, end: float) -> Iterator[dict[str, Any]]:
    """The camera's object memory sightings in the window (memory/objects.py), by the object's name."""
    from liberational.memory import objects

    try:
        rows = objects.sightings(state, start, end)
    except Exception:  # the broker out of reach: no object boxes
        return
    for r in [r for r in rows if r.get("cam") == cam][:LIMIT]:
        # Legacy imported JPEG coordinates cannot be projected onto another native/aged playback picture.
        if r.get("bbox_space", "source") != "source" or (r.get("kind") == "manual_label" and "bbox_space" not in r):
            continue
        try:
            x, y, w, h = (float(v) for v in r["bbox"])
        except KeyError, TypeError, ValueError:
            continue
        label = str(r.get("name") or r.get("object_id") or "Thing").strip()
        yield {
            "ts": float(r["ts_start"]),
            "id": f"object:{r.get('object_id') or r['id']}",
            "kind": "object",
            "box": _box(x, y, x + w, y + h),
            "label": label,
            "tone": "known",
            "name": label,
        }


def _labels(state: pathlib.Path, cam: str, start: float, end: float) -> Iterator[dict[str, Any]]:
    """The camera's grounded object labels standing during the window, each held until its camera's next record."""
    from liberational import segment_log

    rows = segment_log.Log(segment_log.root(state)).range(
        segment_log.CAMERA_OBJECTS, start - LABEL_HOLD_S, end, cameras=[cam]
    )[:LIMIT]
    rows.sort(key=lambda r: float(r["ts_start"]))
    for i, row in enumerate(rows):
        ts = float(row["ts_start"])
        until = min(ts + LABEL_HOLD_S, float(rows[i + 1]["ts_start"]) if i + 1 < len(rows) else math.inf)
        if until <= start:
            continue
        for n, found in enumerate(row.get("objects") or []):
            label, norm = str(found.get("label") or "").strip(), found.get("bbox_norm")
            if not label or norm is None:
                continue  # a box on a warped tile has no place on the picture
            try:
                coords = [float(v) for v in norm[:4]]
            except TypeError, ValueError:
                continue
            if len(coords) != 4 or not all(math.isfinite(v) for v in coords):
                continue
            box = _box(*coords)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            yield {
                "ts": ts,
                "until": round(until, 3),
                "id": f"label:{label.lower()}:{n}",
                "kind": "label",
                "box": box,
                "label": label,
                "tone": "unknown",
            }


def pet_boxes(state: pathlib.Path, cam: str, start: float, end: float) -> Iterator[dict[str, Any]]:
    """Bounded source-frame animal detections, joined to exact owner track labels and current pet profiles. Live's
    Here now reads the last seconds of them for which pets are in view (dashboard/pets.py here)."""
    from liberational import segment_log
    from liberational.people import pet_recognition, pets

    rows = segment_log.Log(segment_log.root(state)).range(segment_log.ANIMAL_DETECTIONS, start, end, cameras=[cam])[
        :LIMIT
    ]
    if not rows:
        return
    listing = pets.at(state).listing()
    profiles = {p["id"]: p for p in listing}
    labels = pet_recognition.labels(state)
    unlabeled = {t for r in rows if isinstance(t := r.get("track_id"), str) and t not in labels}
    try:
        recognized = pet_recognition.resolve(state, cam, start, end, unlabeled, listing) if unlabeled else {}
    except Exception as error:  # recognition is extra: without it the boxes still show, by species
        LOG.warning("Pet recognition for Review's boxes failed: %s", error)
        recognized = {}
    for row in rows:
        track, species = row.get("track_id"), row.get("species")
        if not isinstance(track, str) or not track.strip() or not isinstance(species, str) or not species.strip():
            continue
        try:
            coords = [float(v) for v in row["box"][:4]]
        except KeyError, TypeError, ValueError:
            continue
        if len(coords) != 4 or not all(math.isfinite(v) for v in coords):
            continue
        box = _box(*coords)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        profile = profiles.get(labels.get(track, {}).get("label"))
        match = recognized.get(track) if profile is None else None
        profile = profile or (profiles.get(match["label"]) if match else None)
        species = str(profile["species"]) if profile else species.strip()
        name = str(profile["name"]) if profile else None
        yield {
            "ts": float(row["ts_start"]),
            "id": f"pet:{track}",
            "kind": "pet",
            "track": track,
            "species": species,
            "box": box,
            "label": f"{name} · {species}" if name else species,
            "tone": "known" if profile else "unknown",
            **({"pet_id": profile["id"], "name": name} if profile else {}),
            **({"how": "recognized", "similarity": match["similarity"]} if profile and match else {}),
        }


def _iou(a: list[float], b: list[float]) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _track(items: list[dict[str, Any]]) -> None:
    """Give unknown faces (in time order) one id per stretch they stay put: each takes the id of the
    last box of its kind it overlaps by TRACK_IOU within TRACK_GAP_S, else a new one. Greedy, one box per id a
    moment."""
    tracks: list[dict[str, Any]] = []  # {"id", "kind", "box", "ts"}
    n = 0
    for b in items:
        if b["kind"] != "face" or b["tone"] != "unknown":
            continue
        taken = {t["id"] for t in tracks if t["ts"] == b["ts"]}
        best = max(
            (
                t
                for t in tracks
                if t["kind"] == b["kind"] and t["id"] not in taken and 0 < b["ts"] - t["ts"] <= TRACK_GAP_S
            ),
            key=lambda t: _iou(t["box"], b["box"]),
            default=None,
        )
        if best is not None and _iou(best["box"], b["box"]) >= TRACK_IOU:
            b["id"] = best["id"]
            best.update(box=b["box"], ts=b["ts"])
        else:
            n += 1
            b["id"] = f"{b['kind']}:?{n}"
            tracks.append({"id": b["id"], "kind": b["kind"], "box": b["box"], "ts": b["ts"]})


def boxes(cam: str, start: float, end: float, state: str | pathlib.Path) -> dict[str, Any]:
    """{"cam", "start", "end", "boxes": [...]} for camera cam between start and end (at most WINDOW_MAX_S), in time
    order. Raises ValueError for a bad camera or times."""
    if not isinstance(cam, str) or not views.valid_camera_id(cam):
        raise ValueError("cam must contain only ASCII letters, digits, _ or -")
    start, end = float(start), float(end)
    if not end > start:
        raise ValueError("end must be after start")
    end = min(end, start + WINDOW_MAX_S)
    root = pathlib.Path(state)
    faces = list(_faces(root, cam, start, end))
    out = faces + _bodies(root, cam, start, end, faces)
    out += list(pet_boxes(root, cam, start, end)) + list(_objects(root, cam, start, end))
    out += _labels(root, cam, start, end)
    out.sort(key=lambda b: (b["ts"], b["kind"], b["id"]))
    _track(out)
    return {"cam": cam, "start": start, "end": end, "boxes": out}


# ---- routes (dashboard/routes.py): what the server answers for this area ----


@routes.get("/review/boxes")
def _get_review_boxes(h: H) -> None:
    q = h._query()
    try:
        start, end = float(q.get("start") or ""), float(q.get("end") or "")
        return h._json(boxes(q.get("cam", ""), start, end, h.state))
    except (TypeError, ValueError) as ex:
        return h._json({"error": str(ex)}, 400)
