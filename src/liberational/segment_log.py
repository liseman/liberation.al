"""Derived streams in the segment store: typed schemas, row codecs, writers, and reads.

The segment store (`continuum_py.SegmentWriter`, `read_segments`) keeps per-frame and per-span machine output as
typed columns, one writer per stream and camera, under `paths.state/segments`. A stream here is declared once: its
column schema and how one record (the dict its producers and readers already use) becomes a row and back. Readers
call `at(state).range(stream, start, end)` and get the same records `vision_log` gave them, so moving a stream is a
change of where it is read, not of what a reader does with it.

Span streams have `end_ns` and are read with `overlapping=True`, so a span that started before a window but reaches
into it is found. Times are kept as nanoseconds: epoch seconds in float32 would lose about two minutes. Optional
values with no nullable column type (a boundary's time, a count that can be unknown) are lists that are empty when
absent. A record's `id` is its column `id`: activity naming and feedback name a span by it.
"""

from __future__ import annotations

import json
import math
import re
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import orjson

from liberational import config

MOVEMENT_TRACKS = "movement.track"
MOVEMENT_COVERAGE = "movement.coverage"
ANIMAL_DETECTIONS = "animal.detection"  # see-actions' confirmed animal boxes
ANIMAL_EMBEDDINGS = "animal.embedding"  # and their appearance embeddings, for telling pets apart
ANIMAL_COVERAGE = "animal.coverage"  # and the spans it looked for animals in
# see-captions' animals, grounded by the VLM when a caption names one, in the same shapes: their own streams,
# since a stream has one writer per camera
ANIMAL_GROUNDINGS = "animal.grounding"
ANIMAL_GROUNDING_EMBEDDINGS = "animal.grounding_embedding"
ANIMAL_EMBEDDING_STREAMS = (ANIMAL_EMBEDDINGS, ANIMAL_GROUNDING_EMBEDDINGS)  # every animal embedding, for pets
WALKBY_SPANS = "walkby.span"
WALKBY_COVERAGE = "walkby.coverage"
ZONE_VISITS = "zone.visit"
ZONE_COVERAGE = "zone.coverage"
OBJECT_MATCHES = "object.match"
ACTION_SPANS = "action.span"  # see-actions' model-based actions, in the ACTIVITY span shape
CAPTIONS = "camera.caption"
CAMERA_OBJECTS = "camera.objects"
PRESENCE_SPANS = "presence.span"
ACTION_COVERAGE = "action.coverage"
ACTION_WINDOWS = "action.window"  # see-actions' scored windows: each track's top model classes
FACE_MATCHES = "face.match"  # see/head_faces.py: faces from see-actions' tracked heads
FACE_VISITS = "face.visit"
FACE_EMBEDDINGS = "face.embedding"  # one per face.match, for the vector worker's nearest-face search
FACE_SIGNALS = "face.signal"  # one per look: how much it reads as a face; a set-aside one names no match

NS = 1_000_000_000
Record = dict[str, Any]
Row = dict[str, Any]


def ns(seconds: float) -> int:
    return round(seconds * NS)


def secs(nanos: int) -> float:
    return nanos / NS


def _pairs(intervals: Iterable[Sequence[float]]) -> list[int]:
    return [ns(v) for lo, hi in intervals for v in (lo, hi)]


def _unpairs(flat: Sequence[int]) -> list[list[float]]:
    values = [secs(int(v)) for v in flat]
    return [[values[i], values[i + 1]] for i in range(0, len(values), 2)]


def _maybe(value: int | None) -> list[int]:
    return [] if value is None else [int(value)]


def _unmaybe(values: Sequence[int]) -> int | None:
    return int(values[0]) if len(values) else None


def _boundary_row(prefix: str, b: Mapping[str, Any]) -> Row:
    time = b.get("time")
    return {
        f"{prefix}_status": str(b.get("status", "unknown")),
        f"{prefix}_reason": str(b.get("reason") or ""),
        f"{prefix}_edge": b.get("edge"),
        f"{prefix}_time_ns": _pairs([time]) if time else [],
    }


def _boundary(prefix: str, row: Mapping[str, Any]) -> Record:
    b: Record = {"status": row[f"{prefix}_status"], "reason": row[f"{prefix}_reason"]}
    if row[f"{prefix}_edge"] is not None:
        b["edge"] = row[f"{prefix}_edge"]
    if len(row[f"{prefix}_time_ns"]):
        b["time"] = _unpairs(row[f"{prefix}_time_ns"])[0]
    return b


BOUNDARY = [("status", "str"), ("reason", "str"), ("edge", "str?"), ("time_ns", "i64[]")]
SPAN = [("ts_ns", "i64"), ("end_ns", "i64"), ("id", "id")]
TRACK = [
    *SPAN,
    ("track_id", "str"),
    ("version", "str"),
    ("point_t_ns", "i64[]"),
    ("point_x", "f32[]"),
    ("point_y", "f32[]"),
    ("supported_ns", "i64[]"),
    ("gaps_ns", "i64[]"),
    *[(f"entry_{n}", t) for n, t in BOUNDARY],
    *[(f"exit_{n}", t) for n, t in BOUNDARY],
    ("direction", "str"),
    ("camera_motion", "str"),
    ("finished", "bool"),
    ("closure", "str"),
]
COVERAGE = [
    *SPAN,
    ("version", "str"),
    ("frames", "i64"),
    ("supported_ns", "i64[]"),
    ("gaps_ns", "i64[]"),
    ("tracking", "str"),
    ("camera_motion", "str"),
    ("finished", "bool"),
    ("closure", "str"),
]

ANIMAL = [("subject_type", "str"), ("species", "str?")]


def _span_row(r: Mapping[str, Any]) -> Row:
    return {"ts_ns": ns(r["ts_start"]), "end_ns": ns(r["ts_end"]), "id": str(r["id"])}


def _span(row: Mapping[str, Any], cam: str) -> Record:
    return {"id": row["id"], "ts_start": secs(int(row["ts_ns"])), "ts_end": secs(int(row["end_ns"])), "cam": cam}


def track_row(r: Mapping[str, Any]) -> Row:
    points = r.get("points") or []
    return {
        **_span_row(r),
        "track_id": str(r["track_id"]),
        "version": str(r["version"]),
        "point_t_ns": [ns(p[0]) for p in points],
        "point_x": [float(p[1]) for p in points],
        "point_y": [float(p[2]) for p in points],
        "supported_ns": _pairs(r.get("supported") or []),
        "gaps_ns": _pairs(r.get("gaps") or []),
        **_boundary_row("entry", r.get("entry") or {}),
        **_boundary_row("exit", r.get("exit") or {}),
        "direction": str(r["direction"]),
        "camera_motion": str(r["camera_motion"]),
        "finished": bool(r["finished"]),
        "closure": str(r["closure"]),
    }


def track(row: Mapping[str, Any], cam: str) -> Record:
    return {
        **_span(row, cam),
        "track_id": row["track_id"],
        "version": row["version"],
        "points": [
            [secs(int(t)), float(x), float(y)]
            for t, x, y in zip(row["point_t_ns"], row["point_x"], row["point_y"], strict=True)
        ],
        "supported": _unpairs(row["supported_ns"]),
        "gaps": _unpairs(row["gaps_ns"]),
        "entry": _boundary("entry", row),
        "exit": _boundary("exit", row),
        "direction": row["direction"],
        "camera_motion": row["camera_motion"],
        "finished": bool(row["finished"]),
        "closure": row["closure"],
    }


def coverage_row(r: Mapping[str, Any]) -> Row:
    return {
        **_span_row(r),
        "version": str(r["version"]),
        "frames": int(r["frames"]),
        "supported_ns": _pairs(r.get("supported") or []),
        "gaps_ns": _pairs(r.get("gaps") or []),
        "tracking": str(r["tracking"]),
        "camera_motion": str(r["camera_motion"]),
        "finished": bool(r["finished"]),
        "closure": str(r["closure"]),
    }


def coverage(row: Mapping[str, Any], cam: str) -> Record:
    return {
        **_span(row, cam),
        "version": row["version"],
        "frames": int(row["frames"]),
        "supported": _unpairs(row["supported_ns"]),
        "gaps": _unpairs(row["gaps_ns"]),
        "tracking": row["tracking"],
        "camera_motion": row["camera_motion"],
        "finished": bool(row["finished"]),
        "closure": row["closure"],
    }


ACTIVITY = [
    *SPAN,
    ("track_id", "str"),
    *ANIMAL,
    ("person", "str?"),  # an action span's person (a PersonId) when the track's face was recognized, else None
    ("person_name", "str?"),  # and their name as the face was matched
    ("animals_visible", "i64[]"),
    ("kind", "str"),
    ("direction", "str"),
    ("supported_ns", "i64[]"),
    ("observations", "i64"),
    ("interpretation", "str"),
]


def activity_row(r: Mapping[str, Any]) -> Row:
    attributes = r.get("attributes") or {}
    return {
        **_span_row(r),
        "track_id": str(r["track_id"]),
        "subject_type": str(r["subject_type"]),
        "species": r.get("species"),
        "person": r.get("person"),
        "person_name": r.get("person_name"),
        "animals_visible": _maybe(r.get("animals_visible")),
        "kind": str(r["kind"]),
        "direction": str(attributes.get("direction", "")),
        "supported_ns": _pairs(attributes.get("supported") or []),
        "observations": int(attributes.get("observations", 0)),
        "interpretation": str((r.get("uncertainty") or {}).get("interpretation", "")),
    }


def activity(row: Mapping[str, Any], cam: str) -> Record:
    return {
        **_span(row, cam),
        "track_id": row["track_id"],
        "subject_type": row["subject_type"],
        "species": row["species"],
        "person": row["person"],
        "person_name": row["person_name"],
        "animals_visible": _unmaybe(row["animals_visible"]),
        "kind": row["kind"],
        "candidate": True,
        "camera_moved": False,
        "attributes": {
            "direction": row["direction"],
            "supported": _unpairs(row["supported_ns"]),
            "observations": int(row["observations"]),
        },
        "uncertainty": {"identity": "unidentified", "interpretation": row["interpretation"]},
    }


UTTERANCES = "speech.utterance"
REPLIES = "assistant.reply"
SPEECH_ACTIVITY = "speech.activity"
VOICE_EMBEDDINGS = "voice.embedding"
VOICE_DIM = 192  # ReDimNet2 B3 (vox2-lm): 192 floats a voiceprint

# A filed voice embedding: its voiceprint's subject, the person that voiceprint named when it was filed (the
# entity a match names), the line it came from, and the vector. The vector worker follows this stream.
VOICE = [
    *SPAN,
    ("voiceprint", "id"),
    ("person", "id?"),
    ("utterance", "id?"),
    ("vector", f"f32[{VOICE_DIM}]"),
]


def voice_row(r: Mapping[str, Any]) -> Row:
    return {
        "ts_ns": int(r["start_ns"]),
        "end_ns": max(int(r["start_ns"]), int(r["end_ns"])),
        "id": str(r.get("id") or new_id()),
        "voiceprint": str(r["voiceprint"]),
        "person": None if r.get("person") is None else str(r["person"]),
        "utterance": None if r.get("utterance") is None else str(r["utterance"]),
        "vector": [float(x) for x in r["vector"]],
    }


def voice(row: Mapping[str, Any], _slot: str) -> Record:
    return {
        "id": row["id"],
        "start_ns": int(row["ts_ns"]),
        "end_ns": int(row["end_ns"]),
        "ts_start": secs(int(row["ts_ns"])),
        "voiceprint": row["voiceprint"],
        "person": row["person"],
        "utterance": row["utterance"],
        "vector": [float(x) for x in row["vector"]],
    }


LATENCIES = ("gate", "asr", "emb")  # the transcriber's steps, each an optional `lat_<step>_ms` column
UTTERANCE = [
    *SPAN,
    ("device", "str"),
    ("text", "str"),
    ("speaker", "str?"),
    ("speaker_id", "i64[]"),
    ("speaker_sim", "f32[]"),
    *[(f"lat_{step}_ms", "i64[]") for step in LATENCIES],
    ("password", "bool"),
    ("audio_stream", "str?"),
    ("audio_ns", "i64[]"),
]
REPLY = [
    *SPAN,
    ("speaker", "str?"),
    ("text", "str"),
    ("reply_to", "str?"),
    ("route", "str?"),
    ("error", "str?"),
]
GATE = [("ts_ns", "i64"), ("end_ns", "i64"), ("device", "str"), ("stream", "str"), ("in_speech", "bool"),
        ("level_dbfs", "f32"), ("threshold_dbfs", "f32")]  # fmt: skip


def _line_span(r: Mapping[str, Any]) -> tuple[int, int]:
    """A speech line's span in nanoseconds: `span_ns` when the writer has it exactly, else from its seconds."""
    if (span := r.get("span_ns")) is not None:
        return int(span[0]), max(int(span[0]), int(span[1]))
    start = ns(float(r["ts_start"]))
    return start, max(start, ns(float(r.get("ts_end") or r["ts_start"])))


def _speech_line(row: Mapping[str, Any]) -> Record:
    return {"id": row["id"], "ts_start": secs(int(row["ts_ns"])), "ts_end": secs(int(row["end_ns"]))}


def _some(record: Record, key: str, value: object) -> None:
    """Sets `key` only for a value that's there: a line never carried a key it had no value for."""
    if value is not None:
        record[key] = value


def utterance_row(r: Mapping[str, Any]) -> Row:
    start, end = _line_span(r)
    lat = r.get("lat_ms") or {}
    audio = r.get("audio") or {}
    sim = r.get("speaker_sim")
    return {
        "ts_ns": start,
        "end_ns": end,
        "id": str(r.get("id") or new_id()),
        "device": str(r.get("device") or ""),
        "text": str(r["text"]),
        "speaker": r.get("speaker"),
        "speaker_id": _maybe(r.get("speaker_id")),
        "speaker_sim": [] if sim is None else [float(sim)],
        **{f"lat_{step}_ms": _maybe(lat.get(step)) for step in LATENCIES},
        "password": bool(r.get("password")),
        "audio_stream": audio.get("stream"),
        "audio_ns": [int(audio["start_ns"]), int(audio["end_ns"])] if audio else [],
    }


def utterance(row: Mapping[str, Any], _slot: str) -> Record:
    out = _speech_line(row)
    out["device"] = row["device"]
    _some(out, "speaker", row["speaker"])
    _some(out, "speaker_id", _unmaybe(row["speaker_id"]))
    if len(row["speaker_sim"]):
        out["speaker_sim"] = round(float(row["speaker_sim"][0]), 3)  # it was rounded to 3 places before float32
    out["text"] = row["text"]
    lat = {step: v for step in LATENCIES if (v := _unmaybe(row[f"lat_{step}_ms"])) is not None}
    if lat:
        out["lat_ms"] = lat
    if row["password"]:
        out["password"] = True
    if row["audio_stream"] is not None and len(row["audio_ns"]) == 2:
        start, end = (int(v) for v in row["audio_ns"])
        out["audio"] = {"stream": row["audio_stream"], "start_ns": start, "end_ns": end}
    return out


def reply_row(r: Mapping[str, Any]) -> Row:
    start, end = _line_span(r)
    return {
        "ts_ns": start,
        "end_ns": end,
        "id": str(r.get("id") or new_id()),
        **{k: (None if r.get(k) is None else str(r[k])) for k in ("speaker", "reply_to", "route", "error")},
        "text": str(r["text"]),
    }


def reply(row: Mapping[str, Any], _slot: str) -> Record:
    out = _speech_line(row)
    for k in ("speaker", "text", "reply_to", "route", "error"):
        _some(out, k, row[k])
    out["assistant"] = True
    return out


def gate_row(r: Mapping[str, Any]) -> Row:
    start, _ = _line_span(r)
    return {
        "ts_ns": start,
        "end_ns": start,
        "device": str(r["device"]),
        "stream": str(r.get("stream") or ""),
        "in_speech": bool(r["in_speech"]),
        "level_dbfs": float(r.get("level_dbfs", math.nan)),
        "threshold_dbfs": float(r.get("threshold_dbfs", math.nan)),
    }


def gate(row: Mapping[str, Any], _slot: str) -> Record:
    out: Record = {
        "ts_start": secs(int(row["ts_ns"])),
        "device": row["device"],
        "stream": row["stream"],
        "in_speech": bool(row["in_speech"]),
    }
    for k in ("level_dbfs", "threshold_dbfs"):
        if math.isfinite(v := float(row[k])):
            out[k] = v
    return out


WALKBY = [
    *SPAN,
    ("track_id", "str"),
    ("version", "str"),
    ("direction", "str?"),
    ("line_kind", "str"),
    ("line_values", "f32[]"),
    ("crossings", "i64"),
    ("tracks", "i64"),
    ("followed", "bool"),
    ("camera_moved", "bool"),
    ("motion_unknown", "bool"),
]
WALKBY_FROM = "see-actions' person tracks"


def _line_row(line: Mapping[str, Any]) -> Row:
    kind = str(line.get("kind", "vertical"))
    values = [*line["p1"], *line["p2"]] if kind == "segment" else [line["position"]]
    return {"line_kind": kind, "line_values": [float(v) for v in values]}


def _line(row: Mapping[str, Any]) -> Record:
    values = [float(v) for v in row["line_values"]]
    if row["line_kind"] == "segment":
        return {"kind": "segment", "p1": values[:2], "p2": values[2:4]}
    return {"kind": row["line_kind"], "position": values[0]}


def walkby_row(r: Mapping[str, Any]) -> Row:
    attributes = r.get("attributes") or {}
    return {
        **_span_row(r),
        "track_id": str(r["track_id"]),
        "version": str((r.get("interpretation") or {}).get("version", "")),
        "direction": attributes.get("direction"),
        **_line_row(attributes.get("line") or {"kind": "vertical", "position": 0.5}),
        "crossings": int(attributes.get("crossings", 1)),
        "tracks": int(attributes.get("tracks", 1)),
        "followed": bool(attributes.get("followed", True)),
        "camera_moved": bool(r.get("camera_moved", True)),
        "motion_unknown": (r.get("uncertainty") or {}).get("camera_motion") == "unknown",
    }


def walkby(row: Mapping[str, Any], cam: str) -> Record:
    """A walk-by span as the walk-by worker made it; its evidence block is rebuilt from the stored fields."""
    uncertainty: Record = {"value": None, "joints": [], "from": WALKBY_FROM}
    if row["motion_unknown"]:
        uncertainty["camera_motion"] = "unknown"
    attributes = {
        "direction": row["direction"],
        "line": _line(row),
        "crossings": int(row["crossings"]),
        "tracks": int(row["tracks"]),
        "followed": bool(row["followed"]),
    }
    evidence = {
        key: {
            "value": value,
            "uncertainty": dict(uncertainty),
            "producer": {"model": "walk-bys", "version": row["version"]},
            "interpretation": {"input": "pose.track", "source": "interpretation.records"},
        }
        for key, value in {**attributes, "kind": "crossing"}.items()
    }
    return {
        **_span(row, cam),
        "track_id": row["track_id"],
        "kind": "crossing",
        "candidate": False,
        "attributes": attributes,
        "attribute_evidence": evidence,
        "uncertainty": uncertainty,
        "camera_moved": bool(row["camera_moved"]),
        "interpretation": {"producer": "see-actions", "version": row["version"], "input": "pose.track"},
    }


WALKBY_COVERAGE_COLUMNS = [*SPAN, ("frames", "i64")]


def walkby_coverage_row(r: Mapping[str, Any]) -> Row:
    return {**_span_row(r), "frames": int(r["frames"])}


def walkby_coverage(row: Mapping[str, Any], cam: str) -> Record:
    return {**_span(row, cam), "frames": int(row["frames"])}


# Zone visits and coverage (see/zones.py, see/zones_worker.py). A visit's dwell is its rounded span; its
# interpretation is fixed but for the version. Coverage names its zone ids as a JSON list (no string-list column).
ZONE_INTERPRETATION = {"producer": "zone-visits", "anchor": "box_bottom_centre"}
ZONE_VISIT_SCHEMA = [
    *SPAN,
    ("zone", "str"),
    ("zone_name", "str"),
    ("definition", "str"),
    ("subject_kind", "str"),
    ("track_id", "str"),
    ("label", "str?"),
    ("dwell_s", "f64"),
    ("frames", "i64"),
    ("entry_seen", "bool"),
    ("exit_seen", "bool"),
    ("ended", "str"),
    ("version", "str"),
]
ZONE_COVERAGE_SCHEMA = [*SPAN, ("subject_kind", "str"), ("definition", "str"), ("zones", "str"), ("frames", "i64")]


def zone_visit_row(r: Mapping[str, Any]) -> Row:
    return {
        **_span_row(r),
        "zone": str(r["zone"]),
        "zone_name": str(r["zone_name"]),
        "definition": str(r["definition"]),
        "subject_kind": str(r["subject_kind"]),
        "track_id": str(r["track_id"]),
        "label": r.get("label"),
        "dwell_s": float(r["dwell_s"]),
        "frames": int(r["frames"]),
        "entry_seen": bool(r["entry_seen"]),
        "exit_seen": bool(r["exit_seen"]),
        "ended": str(r["ended"]),
        "version": str((r.get("interpretation") or {}).get("version", "")),
    }


def zone_visit(row: Mapping[str, Any], cam: str) -> Record:
    return {
        **_span(row, cam),
        "zone": row["zone"],
        "zone_name": row["zone_name"],
        "definition": row["definition"],
        "subject_kind": row["subject_kind"],
        "track_id": row["track_id"],
        "label": row["label"],
        "dwell_s": float(row["dwell_s"]),
        "frames": int(row["frames"]),
        "entry_seen": bool(row["entry_seen"]),
        "exit_seen": bool(row["exit_seen"]),
        "ended": row["ended"],
        "candidate": True,
        "interpretation": {**ZONE_INTERPRETATION, "version": row["version"]},
    }


def zone_coverage_row(r: Mapping[str, Any]) -> Row:
    return {
        **_span_row(r),
        "subject_kind": str(r["subject_kind"]),
        "definition": str(r["definition"]),
        "zones": json.dumps([str(z) for z in r.get("zones") or []]),
        "frames": int(r["frames"]),
    }


def zone_coverage(row: Mapping[str, Any], cam: str) -> Record:
    return {
        **_span(row, cam),
        "subject_kind": row["subject_kind"],
        "definition": row["definition"],
        "zones": json.loads(row["zones"]),
        "frames": int(row["frames"]),
    }


# The object worker's visual matches of remembered objects (memory/objects.py, see/object_worker.py): a moment, with
# the box in source pixels and the frame as a small JPEG (`evidence`, base64). The owner's own crops
# (`manual_label`) are taught data, kept as `object.sighting` records in the knowledge store.
OBJECT_MATCH_SCHEMA = [
    *SPAN,
    ("object_id", "str"),
    ("name", "str"),
    ("bbox", "f32[]"),
    ("score", "f64"),
    ("frame_id", "str?"),
    ("frame_seq", "i64[]"),
    ("evidence", "str"),
]


def object_match_row(r: Mapping[str, Any]) -> Row:
    return {
        "ts_ns": ns(r["ts_start"]),
        "end_ns": ns(r.get("ts_end", r["ts_start"])),  # a match is a moment: it ends where it starts
        "id": str(r["id"]),
        "object_id": str(r["object_id"]),
        "name": str(r["name"]),
        "bbox": [float(v) for v in r["bbox"]],
        "score": float(r["score"]),
        "frame_id": r.get("frame_id"),
        "frame_seq": _maybe(r.get("frame_seq")),
        "evidence": str(r["evidence"]),
    }


def object_match(row: Mapping[str, Any], cam: str) -> Record:
    return {
        **_span(row, cam),
        "object_id": row["object_id"],
        "name": row["name"],
        "bbox": [float(v) for v in row["bbox"]],
        "score": float(row["score"]),
        "kind": "visual_match",
        "bbox_space": "source",
        "frame_id": row["frame_id"],
        "frame_seq": _unmaybe(row["frame_seq"]),
        "evidence": row["evidence"],
    }


# The caption worker's captions (see/captions.py, see/caption_worker.py): one per looked-at frame, with the grounded
# boxes it carries. A caption's `objects` is None when its pipeline doesn't ground, and each box's `bbox_norm` is None
# when the model saw a warped tile; labels are a JSON list (there's no string-list column). `faces` is always None
# here: the face worker's records hold them.
BOXES = [("obj_labels", "str"), ("obj_bbox", "i64[]"), ("obj_norm", "f32[]"), ("obj_ts_ns", "i64[]")]


def _boxes_row(objs: Sequence[Mapping[str, Any]]) -> Row:
    """Grounded boxes as columns: labels (a JSON list), pixel boxes and normalized ones flat, four values each (none
    when any box has no `bbox_norm`), and the one grounding time they share (none when they carry no `ts`)."""
    normed = bool(objs) and all(o.get("bbox_norm") is not None for o in objs)
    stamps = {o["ts"] for o in objs if o.get("ts") is not None}
    if len(stamps) > 1:
        raise ValueError("boxes in one record come from one grounding")
    return {
        "obj_labels": json.dumps([str(o["label"]) for o in objs]),
        "obj_bbox": [int(v) for o in objs for v in o["bbox"]],
        "obj_norm": [float(v) for o in objs for v in o["bbox_norm"]] if normed else [],
        "obj_ts_ns": [ns(t) for t in stamps],
    }


def _boxes(row: Mapping[str, Any]) -> list[Record]:
    flat, norm = [int(v) for v in row["obj_bbox"]], [float(v) for v in row["obj_norm"]]
    stamp = [secs(int(t)) for t in row["obj_ts_ns"]]
    return [
        {
            "label": label,
            "bbox": flat[4 * i : 4 * i + 4],
            **({"ts": stamp[0]} if stamp else {}),
            "bbox_norm": norm[4 * i : 4 * i + 4] if norm else None,
        }
        for i, label in enumerate(json.loads(row["obj_labels"]))
    ]


CAPTION_SCHEMA = [
    *SPAN,
    ("frame_id", "str?"),
    ("frame_seq", "i64"),
    ("yaw_deg", "f64"),
    ("caption", "str"),
    ("source", "str"),
    ("synthetic", "bool"),
    ("gated", "bool"),
    ("status", "str"),
    ("observed_ns", "i64[]"),
    ("backend", "str"),
    ("prompt_version", "str"),
    ("change_mad", "f64"),
    ("change_frac", "f64"),
    ("bright_mean", "f64"),
    ("dark_fraction", "f64"),
    ("vlm_ms", "i64"),
    ("unchanged", "bool"),
    ("since_ns", "i64[]"),
    ("deferred", "bool"),
    ("grounded", "bool"),
    *BOXES,
]


def caption_row(r: Mapping[str, Any]) -> Row:
    obs = r.get("observation") or {}
    change = r.get("change")
    bright = r.get("brightness") or {}
    objs = r.get("objects")
    return {
        **_span_row({**r, "ts_end": r.get("ts_end", r["ts_start"])}),
        "frame_id": r.get("frame_id"),
        "frame_seq": int(r.get("frame_seq") or 0),
        "yaw_deg": float(r.get("yaw_deg") or 0.0),
        "caption": str(r.get("caption") or ""),
        "source": str(r.get("source") or ""),
        "synthetic": bool(r.get("synthetic")),
        "gated": bool(r.get("gated")),
        "status": str(obs.get("status", "")),
        "observed_ns": _maybe(None if obs.get("observed_at") is None else ns(obs["observed_at"])),
        "backend": str(obs.get("backend", "")),
        "prompt_version": str(obs.get("prompt_version", "")),
        "change_mad": math.nan if change is None else float(change["mad"]),
        "change_frac": math.nan if change is None else float(change["frac"]),
        "bright_mean": float(bright.get("mean", math.nan)),
        "dark_fraction": float(bright.get("dark_fraction", math.nan)),
        "vlm_ms": int((r.get("ms") or {}).get("vlm", 0)),
        "unchanged": bool(r.get("unchanged")),
        "since_ns": _maybe(None if r.get("since") is None else ns(r["since"])),
        "deferred": bool(r.get("deferred")),
        "grounded": objs is not None,
        **_boxes_row(objs or []),
    }


def caption(row: Mapping[str, Any], cam: str) -> Record:
    objs = _boxes(row)
    observed = _unmaybe(row["observed_ns"])
    mad, frac = float(row["change_mad"]), float(row["change_frac"])
    out: Record = {
        **_span(row, cam),
        "frame_id": row["frame_id"],
        "frame_seq": int(row["frame_seq"]),
        "yaw_deg": float(row["yaw_deg"]),
        "caption": row["caption"],
        "faces": None,
        "objects": objs if row["grounded"] else None,
        "source": row["source"],
        "synthetic": bool(row["synthetic"]),
        "gated": bool(row["gated"]),
        "observation": {
            "version": 1,
            "status": row["status"],
            "observed_at": None if observed is None else secs(observed),
            "camera": cam,
            "backend": row["backend"],
            "prompt_version": row["prompt_version"],
        },
        "change": None if math.isnan(mad) or math.isnan(frac) else {"mad": mad, "frac": frac},
        "brightness": {"mean": float(row["bright_mean"]), "dark_fraction": float(row["dark_fraction"])},
        "ms": {"vlm": int(row["vlm_ms"])},
    }
    if row["unchanged"]:
        since = _unmaybe(row["since_ns"])
        out.update(unchanged=True, since=None if since is None else secs(since))
    if row["deferred"]:
        out["deferred"] = True
    return out


# The caption worker's groundings (see/captions.py): the boxes standing in a camera's view at a frame, fresh or a cached
# copy of the last grounding, which the gateway draws over Live (continuum-gateway segment_overlays.rs reads these
# columns by name).
CAMERA_OBJECTS_SCHEMA = [
    *SPAN,
    ("frame_id", "str?"),
    ("frame_seq", "i64"),
    ("source", "str"),
    ("status", "str"),
    ("observed_ns", "i64[]"),
    ("attempted_ns", "i64[]"),
    ("reason", "str?"),
    *BOXES,
]


def camera_objects_row(r: Mapping[str, Any]) -> Row:
    obs = r.get("observation") or {}
    return {
        **_span_row({**r, "ts_end": r.get("ts_end", r["ts_start"])}),
        "frame_id": r.get("frame_id"),
        "frame_seq": int(r.get("frame_seq") or 0),
        "source": str(r.get("source") or ""),
        "status": str(obs.get("status", "")),
        "observed_ns": _maybe(None if obs.get("observed_at") is None else ns(obs["observed_at"])),
        "attempted_ns": _maybe(None if obs.get("attempted_at") is None else ns(obs["attempted_at"])),
        "reason": obs.get("reason"),
        **_boxes_row(r.get("objects") or []),
    }


def camera_objects(row: Mapping[str, Any], cam: str) -> Record:
    observed, attempted = _unmaybe(row["observed_ns"]), _unmaybe(row["attempted_ns"])
    observation: Record = {
        "version": 1,
        "status": row["status"],
        "observed_at": None if observed is None else secs(observed),
        "attempted_at": None if attempted is None else secs(attempted),
    }
    if row["reason"] is not None:
        observation["reason"] = row["reason"]
    return {
        **_span(row, cam),
        "frame_id": row["frame_id"],
        "frame_seq": int(row["frame_seq"]),
        "objects": _boxes(row),
        "source": row["source"],
        "observation": observation,
    }


# Times someone was in view of a camera (memory/presence_episodes.py), written by see-zones from see-actions'
# frames: never a count of people, and `tracks` is distinct camera-local track ids, never identities.
PRESENCE_SCHEMA = [
    *SPAN,
    ("frames", "i64"),
    ("max_people", "i64"),
    ("tracks", "i64"),
    ("left_censored", "bool"),
    ("right_censored", "bool"),
]


def presence_row(r: Mapping[str, Any]) -> Row:
    return {
        **_span_row(r),
        "frames": int(r["frames"]),
        "max_people": int(r["max_people"]),
        "tracks": int(r["tracks"]),
        "left_censored": bool(r["left_censored"]),
        "right_censored": bool(r["right_censored"]),
    }


def presence(row: Mapping[str, Any], cam: str) -> Record:
    return {
        **_span(row, cam),
        "frames": int(row["frames"]),
        "max_people": int(row["max_people"]),
        "tracks": int(row["tracks"]),
        "left_censored": bool(row["left_censored"]),
        "right_censored": bool(row["right_censored"]),
    }


# see-actions' analyzed spans (see/actions.py COVERAGE_SCHEMA, which this must equal: see-actions writes them), read
# here as records so presence reads them like any stream. A row has no id of its own: its camera and start name it.
ACTION_COVERAGE_SCHEMA = [("ts_ns", "i64"), ("end_ns", "i64"), ("frames", "i64"), ("model", "str")]


# see/actions.py's WINDOW_SCHEMA: a scored window of one track, its TOP classes (NTU-120 indices) and their scores.
ACTION_WINDOW_SCHEMA = [
    ("ts_ns", "i64"),
    ("end_ns", "i64"),
    ("track_id", "i64"),
    ("classes", "i16[5]"),
    ("scores", "f32[5]"),
    ("model", "str"),
]


def action_window_row(r: Mapping[str, Any]) -> Row:
    return {
        "ts_ns": ns(r["ts_start"]),
        "end_ns": ns(r["ts_end"]),
        "track_id": int(r["track_id"]),
        "classes": [int(c) for c in r["classes"]],
        "scores": [float(x) for x in r["scores"]],
        "model": str(r["model"]),
    }


def action_window(row: Mapping[str, Any], cam: str) -> Record:
    return {
        "id": f"{cam}:{int(row['track_id'])}:{int(row['ts_ns'])}:{row['model']}",  # both models score a track
        "cam": cam,
        "ts_start": secs(int(row["ts_ns"])),
        "ts_end": secs(int(row["end_ns"])),
        "track_id": str(int(row["track_id"])),  # as zone visits and face matches name see-actions' tracks
        "classes": [int(c) for c in row["classes"]],
        "scores": [float(x) for x in row["scores"]],
        "model": row["model"],
    }


def action_coverage_row(r: Mapping[str, Any]) -> Row:
    return {"ts_ns": ns(r["ts_start"]), "end_ns": ns(r["ts_end"]), "frames": int(r["frames"]), "model": str(r["model"])}


def action_coverage(row: Mapping[str, Any], cam: str) -> Record:
    return {
        "id": f"{cam}:{int(row['ts_ns'])}",
        "cam": cam,
        "ts_start": secs(int(row["ts_ns"])),
        "ts_end": secs(int(row["end_ns"])),
        "frames": int(row["frames"]),
        "model": row["model"],
    }


# Faces from see-actions' tracked heads (see/head_faces.py): one match per look at a track's head, with the face's box
# normalized to the frame, and the visits a named person's matches make on one camera. A match is a moment; its `id`
# is also its `detection`, which a visit's `shots` name (People's photos). `shots` is a JSON list: there's no
# list-of-records column.
FACE_MATCH_SCHEMA = [
    *SPAN,
    ("track_id", "str"),
    ("frame_id", "str"),
    ("bbox", "f32[]"),
    ("det", "f64"),
    ("name", "str?"),
    ("sim", "f64"),
    ("cluster", "str?"),
    ("quality", "f64"),
]
FACE_VISIT_SCHEMA = [*SPAN, ("name", "str"), ("best_sim", "f64"), ("samples", "i64"), ("shots", "str")]
# What each look's face is made of, beside its face.match: the pose's confidence in the five head joints it was
# cropped from (nose, left eye, right eye, left ear, right ear; a back of a head has ears but no nose or eyes), the
# detector's landmark frontalness (face_quality.frontal), and AdaFace's raw feature norm (NaN when not given). Kept
# apart from face.match, whose stored segments have no room for new columns.
FACE_SIGNAL_SCHEMA = [*SPAN, ("match_id", "id"), ("head", "f32[5]"), ("frontal", "f64"), ("norm", "f64")]
FACE_EMBEDDING_SCHEMA = [
    *SPAN,
    ("match_id", "id"),  # the face.match row it came from
    ("track_id", "str"),
    ("name", "str?"),  # for display; the vector worker keys on person, else group
    ("person", "id?"),  # the named person's PersonId; None while unknown
    ("group", "id?"),  # the unknown face's group subject (see.subjects.group); None when there's none
    ("vector", "f32[512]"),  # AdaFace IR50, L2-normalized
    ("model", "str"),
]


def face_embedding_row(r: Mapping[str, Any]) -> Row:
    return {
        "ts_ns": ns(r["ts_start"]),
        "end_ns": ns(r.get("ts_end", r["ts_start"])),
        "id": str(r["id"]),
        "match_id": str(r["match_id"]),
        "track_id": str(r["track_id"]),
        "name": r.get("name"),
        "person": r.get("person"),
        "group": r.get("group"),
        "vector": [float(v) for v in r["vector"]],
        "model": str(r["model"]),
    }


def face_embedding(row: Mapping[str, Any], cam: str) -> Record:
    """A face.embedding row as a record. Its vector stays the float32 (512,) array the store read (a row of its
    (rows, 512) block, not a copy): a list of 512 Python floats cost about 66 us a row, 1.25 s of autoname's 2.2 s
    over 7 days (18,889 rows), and every reader turns it straight back into numpy. It is a read-only view, so
    a reader can't change the block it shares (or an hour `keep_hours` kept), and the caller's own array stays as it
    was."""
    vector = np.asarray(row["vector"], np.float32).view()
    vector.flags.writeable = False
    return {
        **_span(row, cam),
        "match_id": row["match_id"],
        "track_id": row["track_id"],
        "name": row["name"],
        "person": row["person"],
        "group": row["group"],
        "vector": vector,
        "model": row["model"],
    }


def face_match_row(r: Mapping[str, Any]) -> Row:
    return {
        "ts_ns": ns(r["ts_start"]),
        "end_ns": ns(r.get("ts_end", r["ts_start"])),
        "id": str(r["id"]),
        "track_id": str(r["track_id"]),
        "frame_id": str(r["frame_id"]),
        "bbox": [float(v) for v in r["bbox"]],
        "det": float(r["det"]),
        "name": r.get("name"),
        "sim": float(r["sim"]),
        "cluster": r.get("cluster"),
        "quality": float(r["quality"]),
    }


def face_match(row: Mapping[str, Any], cam: str) -> Record:
    return {
        **_span(row, cam),
        "detection": row["id"],
        "track_id": row["track_id"],
        "frame_id": row["frame_id"],
        "bbox": [float(v) for v in row["bbox"]],
        "det": float(row["det"]),
        "name": row["name"],
        "sim": float(row["sim"]),
        "cluster": row["cluster"],
        "quality": float(row["quality"]),
    }


def face_signal_row(r: Mapping[str, Any]) -> Row:
    return {
        **_span_row(r),
        "match_id": str(r["match_id"]),
        "head": [float(v) for v in r["head"]],
        "frontal": float(r["frontal"]),
        "norm": float("nan") if r.get("norm") is None else float(r["norm"]),
    }


def face_signal(row: Mapping[str, Any], cam: str) -> Record:
    norm = float(row["norm"])
    return {
        **_span(row, cam),
        "match_id": row["match_id"],
        "head": [float(v) for v in row["head"]],
        "frontal": float(row["frontal"]),
        "norm": None if math.isnan(norm) else norm,
    }


def face_visit_row(r: Mapping[str, Any]) -> Row:
    return {
        **_span_row(r),
        "name": str(r["name"]),
        "best_sim": float(r["best_sim"]),
        "samples": int(r["samples"]),
        "shots": json.dumps(r.get("shots") or []),
    }


def face_visit(row: Mapping[str, Any], cam: str) -> Record:
    return {
        **_span(row, cam),
        "name": row["name"],
        "best_sim": float(row["best_sim"]),
        "samples": int(row["samples"]),
        "shots": json.loads(row["shots"]),
    }


# see-actions' animals (see/action_animals.py): each confirmed box of a frame, and the embeddings its tracks
# are due. Their coverage has walk-by coverage's shape. A record's `ts_ns`, when it has one, is kept exactly; else
# its `ts_start` is converted.
EMBEDDING_DIM = 384  # DINOv3 ViT-S/16 (vision/animal_embedder.py); old DINOv2 vectors remain incompatible
ANIMAL_DETECTION_SCHEMA = [
    ("ts_ns", "i64"),
    ("frame_id", "id"),
    ("track_id", "str"),
    ("species", "str"),
    ("box", "f32[5]"),  # x1, y1, x2, y2 normalized to the frame, and the detector's score
    ("verified", "bool"),  # the track passed the image check (vision/animal_verifier.py)
    ("model", "str"),
]
ANIMAL_EMBEDDING_SCHEMA = [
    ("ts_ns", "i64"),
    ("track_id", "str"),
    ("species", "str"),
    ("box", "f32[5]"),
    ("vector", f"f32[{EMBEDDING_DIM}]"),  # unit length
    ("model", "str"),
]


def _ts_ns(r: Mapping[str, Any]) -> int:
    return int(r["ts_ns"]) if r.get("ts_ns") is not None else ns(r["ts_start"])


def animal_detection_row(r: Mapping[str, Any]) -> Row:
    return {
        "ts_ns": _ts_ns(r),
        "frame_id": str(r["frame_id"]),
        "track_id": str(r["track_id"]),
        "species": str(r["species"]),
        "box": [float(v) for v in r["box"][:5]],
        "verified": bool(r["verified"]),
        "model": str(r["model"]),
    }


def animal_detection(row: Mapping[str, Any], cam: str) -> Record:
    return {
        "ts_start": secs(int(row["ts_ns"])),
        "cam": cam,
        "frame_id": row["frame_id"],
        "track_id": row["track_id"],
        "species": row["species"],
        "box": [float(v) for v in row["box"]],
        "verified": bool(row["verified"]),
        "model": row["model"],
    }


def animal_embedding_row(r: Mapping[str, Any]) -> Row:
    return {
        "ts_ns": _ts_ns(r),
        "track_id": str(r["track_id"]),
        "species": str(r["species"]),
        "box": [float(v) for v in r["box"][:5]],
        "vector": [float(v) for v in r["vector"]],
        "model": str(r["model"]),
    }


def animal_embedding(row: Mapping[str, Any], cam: str) -> Record:
    return {
        "ts_start": secs(int(row["ts_ns"])),
        "cam": cam,
        "track_id": row["track_id"],
        "species": row["species"],
        "box": [float(v) for v in row["box"]],
        "vector": [float(v) for v in row["vector"]],
        "model": row["model"],
    }


@dataclass(frozen=True)
class Stream:
    """One segment stream: its column schema and its record codec."""

    name: str
    schema: list[tuple[str, str]]
    to_row: Callable[[Mapping[str, Any]], Row]
    to_record: Callable[[Mapping[str, Any], str], Record]


STREAMS: dict[str, Stream] = {
    s.name: s
    for s in (
        Stream(MOVEMENT_TRACKS, TRACK, track_row, track),
        Stream(MOVEMENT_COVERAGE, COVERAGE, coverage_row, coverage),
        Stream(UTTERANCES, UTTERANCE, utterance_row, utterance),
        Stream(REPLIES, REPLY, reply_row, reply),
        Stream(SPEECH_ACTIVITY, GATE, gate_row, gate),
        Stream(VOICE_EMBEDDINGS, VOICE, voice_row, voice),
        Stream(WALKBY_SPANS, WALKBY, walkby_row, walkby),
        Stream(WALKBY_COVERAGE, WALKBY_COVERAGE_COLUMNS, walkby_coverage_row, walkby_coverage),
    )
}
# see-zones' outputs.
STREAMS |= {
    s.name: s
    for s in (
        Stream(ZONE_VISITS, ZONE_VISIT_SCHEMA, zone_visit_row, zone_visit),
        Stream(ZONE_COVERAGE, ZONE_COVERAGE_SCHEMA, zone_coverage_row, zone_coverage),
    )
}
STREAMS[ACTION_SPANS] = Stream(ACTION_SPANS, ACTIVITY, activity_row, activity)
# The object worker's matches, and the caption worker's captions and object tracks.
STREAMS[OBJECT_MATCHES] = Stream(OBJECT_MATCHES, OBJECT_MATCH_SCHEMA, object_match_row, object_match)
STREAMS[CAPTIONS] = Stream(CAPTIONS, CAPTION_SCHEMA, caption_row, caption)
STREAMS[CAMERA_OBJECTS] = Stream(CAMERA_OBJECTS, CAMERA_OBJECTS_SCHEMA, camera_objects_row, camera_objects)
STREAMS[FACE_MATCHES] = Stream(FACE_MATCHES, FACE_MATCH_SCHEMA, face_match_row, face_match)
STREAMS[FACE_VISITS] = Stream(FACE_VISITS, FACE_VISIT_SCHEMA, face_visit_row, face_visit)
STREAMS[FACE_EMBEDDINGS] = Stream(FACE_EMBEDDINGS, FACE_EMBEDDING_SCHEMA, face_embedding_row, face_embedding)
STREAMS[FACE_SIGNALS] = Stream(FACE_SIGNALS, FACE_SIGNAL_SCHEMA, face_signal_row, face_signal)
STREAMS[PRESENCE_SPANS] = Stream(PRESENCE_SPANS, PRESENCE_SCHEMA, presence_row, presence)
STREAMS[ACTION_COVERAGE] = Stream(ACTION_COVERAGE, ACTION_COVERAGE_SCHEMA, action_coverage_row, action_coverage)
STREAMS[ACTION_WINDOWS] = Stream(ACTION_WINDOWS, ACTION_WINDOW_SCHEMA, action_window_row, action_window)
STREAMS[ANIMAL_DETECTIONS] = Stream(ANIMAL_DETECTIONS, ANIMAL_DETECTION_SCHEMA, animal_detection_row, animal_detection)
STREAMS[ANIMAL_EMBEDDINGS] = Stream(ANIMAL_EMBEDDINGS, ANIMAL_EMBEDDING_SCHEMA, animal_embedding_row, animal_embedding)
STREAMS[ANIMAL_COVERAGE] = Stream(ANIMAL_COVERAGE, WALKBY_COVERAGE_COLUMNS, walkby_coverage_row, walkby_coverage)
STREAMS[ANIMAL_GROUNDINGS] = Stream(ANIMAL_GROUNDINGS, ANIMAL_DETECTION_SCHEMA, animal_detection_row, animal_detection)
STREAMS[ANIMAL_GROUNDING_EMBEDDINGS] = Stream(
    ANIMAL_GROUNDING_EMBEDDINGS, ANIMAL_EMBEDDING_SCHEMA, animal_embedding_row, animal_embedding
)


# Place visits: a span of time the rig spent in one indoor place, derived from the place signals. Machine-
# made, so a re-derivation writes new rows rather than editing old ones. The slot is what decided the visit (the
# rig, or a fixed camera); `memory/places.py` reads them.
PLACE_VISITS = "place.visit"
PLACE_VISIT_SCHEMA = [*SPAN, ("place", "str"), ("producer", "str"), ("basis", "str")]


def place_visit_row(r: Mapping[str, Any]) -> Row:
    return {**_span_row(r), "place": str(r["place"]), "producer": str(r["producer"]), "basis": str(r["basis"])}


def place_visit(row: Mapping[str, Any], cam: str) -> Record:
    return {**_span(row, cam), "place": row["place"], "producer": row["producer"], "basis": row["basis"]}


STREAMS[PLACE_VISITS] = Stream(PLACE_VISITS, PLACE_VISIT_SCHEMA, place_visit_row, place_visit)


def slot(name: str) -> str:
    """`name` as a writer slot (the store's camera): letters, digits, '.', '_', and '-', other characters as '-'."""
    out = re.sub(r"[^A-Za-z0-9._-]", "-", name).lstrip(".")[:128]
    return out or "unnamed"


# The network inventory's polls (proximity/network_worker.py): one row per device a poll saw, and one per poll. The
# camera is the poll's source ("unifi", "host"). A device's connected state is 1, 0, or -1 for unknown (a cached ARP
# or DHCP row).
NET_OBSERVATIONS = "network.device_observation"
NET_INVENTORY = "network.inventory"
_NET_TEXT = ("mac", "hostname", "display_name", "ip", "vendor", "ap_id", "ap_name", "connection_type")
NET_OBSERVATION_SCHEMA = [
    ("ts_ns", "i64"),
    ("end_ns", "i64"),
    ("id", "id"),
    ("inventory_id", "id"),
    ("observer", "str"),
    ("identifier_kind", "str"),
    ("identifier", "str"),
    *((name, "str?") for name in _NET_TEXT),
    ("connected", "i16"),
]
NET_INVENTORY_SCHEMA = [
    ("ts_ns", "i64"),
    ("end_ns", "i64"),
    ("id", "id"),
    ("observer", "str"),
    ("complete", "bool"),
    ("authoritative", "bool"),
    ("device_count", "i64"),
    ("warnings", "str"),
]


def net_observation_row(r: Mapping[str, Any]) -> Row:
    connected = r.get("connected")
    return {
        "ts_ns": ns(float(r["start"])),
        "end_ns": ns(float(r["end"])),
        "id": str(r["id"]),
        "inventory_id": str(r["inventory_id"]),
        "observer": str(r["observer"]),
        "identifier_kind": str(r["identifier_kind"]),
        "identifier": str(r["identifier"]),
        **{name: (None if r.get(name) is None else str(r[name])) for name in _NET_TEXT},
        "connected": 1 if connected is True else 0 if connected is False else -1,
    }


def net_observation(row: Mapping[str, Any], cam: str) -> Record:
    connected = int(row["connected"])
    return {
        "id": row["id"],
        "source": cam,
        "start": secs(int(row["ts_ns"])),
        "end": secs(int(row["end_ns"])),
        "ts_start": secs(int(row["ts_ns"])),
        "inventory_id": row["inventory_id"],
        "observer": row["observer"],
        "identifier_kind": row["identifier_kind"],
        "identifier": row["identifier"],
        **{name: row[name] for name in _NET_TEXT},
        "connected": True if connected == 1 else False if connected == 0 else None,
    }


def net_inventory_row(r: Mapping[str, Any]) -> Row:
    return {
        "ts_ns": ns(float(r["start"])),
        "end_ns": ns(float(r["end"])),
        "id": str(r["id"]),
        "observer": str(r["observer"]),
        "complete": bool(r["complete"]),
        "authoritative": bool(r["authoritative"]),
        "device_count": int(r["device_count"]),
        "warnings": "\n".join(str(w) for w in r.get("warnings", [])),
    }


def net_inventory(row: Mapping[str, Any], cam: str) -> Record:
    return {
        "id": row["id"],
        "source": cam,
        "start": secs(int(row["ts_ns"])),
        "end": secs(int(row["end_ns"])),
        "ts_start": secs(int(row["ts_ns"])),
        "observer": row["observer"],
        "complete": bool(row["complete"]),
        "authoritative": bool(row["authoritative"]),
        "device_count": int(row["device_count"]),
        "warnings": [w for w in str(row["warnings"]).split("\n") if w],
    }


STREAMS[NET_OBSERVATIONS] = Stream(NET_OBSERVATIONS, NET_OBSERVATION_SCHEMA, net_observation_row, net_observation)
STREAMS[NET_INVENTORY] = Stream(NET_INVENTORY, NET_INVENTORY_SCHEMA, net_inventory_row, net_inventory)


# Bluetooth sensors' sightings and coverage (proximity/worker.py), under the camera name "ble". The time and id are
# columns; the rest of a record (signal statistics, advertising payloads, sensor counters) is its JSON text, since
# only proximity/devices.py reads it, whole.
BLE_SIGHTINGS = "bluetooth.sighting"
BLE_COVERAGE = "bluetooth.coverage"
BLE_SCHEMA = [("ts_ns", "i64"), ("end_ns", "i64"), ("id", "id"), ("record", "str")]


def ble_row(r: Mapping[str, Any]) -> Row:
    start = float(r["start"])
    rest = {k: v for k, v in r.items() if k not in ("id", "start", "end")}
    return {
        "ts_ns": ns(start),
        "end_ns": ns(max(start, float(r.get("end", start)))),
        "id": str(r["id"]),
        "record": json.dumps(rest, separators=(",", ":"), ensure_ascii=False),
    }


def _record(text: str) -> dict[str, Any]:
    """A row's JSON record: orjson, about four times faster than json (a day of Bluetooth sightings is about half a
    million rows), and json for what orjson refuses (a NaN, an infinity, a lone surrogate). One difference: an
    int past 64 bits comes back a float, where json keeps it; a Bluetooth record holds signal strengths, counts, and
    times, none that large."""
    try:
        return orjson.loads(text)
    except orjson.JSONDecodeError:
        return json.loads(text)


def ble(row: Mapping[str, Any], cam: str) -> Record:
    start = secs(int(row["ts_ns"]))
    return {
        **_record(row["record"]),
        "id": row["id"],
        "start": start,
        "end": secs(int(row["end_ns"])),
        "ts_start": start,
    }


STREAMS[BLE_SIGHTINGS] = Stream(BLE_SIGHTINGS, BLE_SCHEMA, ble_row, ble)
STREAMS[BLE_COVERAGE] = Stream(BLE_COVERAGE, BLE_SCHEMA, ble_row, ble)


# The Wi-Fi access points this machine hears (proximity/wifi.py, written by proximity/network_worker.py), a place
# signal, under the camera name "wifi". One row per access point per written scan; a scan's rows share `scan_id`. A
# scan is written when the set of BSSIDs changes, and again every WIFI_REWRITE_SECONDS while it doesn't, so the
# latest set stands until the next one, and no rows for longer than that means nothing was heard (or no adapter).
# `signal` is in `signal_unit`: "percent" (NetworkManager's 0 to 100 quality) or "dbm" (CoreWLAN's RSSI).
WIFI_ACCESS_POINTS = "wifi.access_point"
WIFI_REWRITE_SECONDS = 600.0
WIFI_SCHEMA = [
    ("ts_ns", "i64"),
    ("end_ns", "i64"),
    ("id", "id"),
    ("scan_id", "id"),
    ("observer", "str"),
    ("bssid", "str"),
    ("ssid", "str?"),
    ("signal", "i16"),
    ("signal_unit", "str"),
    ("freq_mhz", "i32"),
]


def wifi_row(r: Mapping[str, Any]) -> Row:
    return {
        "ts_ns": ns(float(r["start"])),
        "end_ns": ns(float(r["end"])),
        "id": str(r["id"]),
        "scan_id": str(r["scan_id"]),
        "observer": str(r["observer"]),
        "bssid": str(r["bssid"]),
        "ssid": None if r.get("ssid") is None else str(r["ssid"]),
        "signal": int(r["signal"]),
        "signal_unit": str(r["signal_unit"]),
        "freq_mhz": int(r["freq_mhz"]),
    }


def wifi(row: Mapping[str, Any], cam: str) -> Record:
    start = secs(int(row["ts_ns"]))
    return {
        "id": row["id"],
        "source": cam,
        "start": start,
        "end": secs(int(row["end_ns"])),
        "ts_start": start,
        **{name: row[name] for name in ("scan_id", "observer", "bssid", "ssid", "signal_unit")},
        "signal": int(row["signal"]),
        "freq_mhz": int(row["freq_mhz"]),
    }


STREAMS[WIFI_ACCESS_POINTS] = Stream(WIFI_ACCESS_POINTS, WIFI_SCHEMA, wifi_row, wifi)


# Fixed cameras' footprints (spatial/footprints.py records them, spatial/positions.py's `write_samples`): one row per
# observation, raw. A row is the foot point as the camera saw it (`u`, `v`, fractions of the frame; the camera is the
# row's slot, since the store keeps a column of that name for slots). The position layer's placer
# (`positions.advance`) puts them on a place's plan and writes the result to `position.placed`, below; the walkable
# floor reads these rows as they are. The store has no nullable floats, so a missing u, v, x, or y is NaN in the
# column and None in the record.
POSITIONS = "position.sample"
POSITION_SCHEMA = [
    *SPAN,
    ("source", "str"),
    ("confidence", "f32"),
    ("entity", "str?"),
    ("track", "str?"),
    ("u", "f32"),
    ("v", "f32"),
    ("place", "str?"),
    ("room", "str?"),
    ("x", "f64"),
    ("y", "f64"),
    ("place_revision", "str?"),  # a record's `revision` (the store's reads keep `camera` and `revision` for their own)
]
_POSITION_TEXT = ("entity", "track", "place", "room")
_POSITION_FLOATS = ("u", "v", "x", "y")


def position_row(r: Mapping[str, Any]) -> Row:
    return {
        **_span_row(r),
        "source": str(r["source"]),
        "confidence": float(r["confidence"]),
        **{k: None if r.get(k) is None else str(r[k]) for k in _POSITION_TEXT},
        **{k: math.nan if r.get(k) is None else float(r[k]) for k in _POSITION_FLOATS},
        "place_revision": None if r.get("revision") is None else str(r["revision"]),
    }


def position(row: Mapping[str, Any], cam: str) -> Record:
    return {
        **_span(row, cam),
        "source": row["source"],
        "confidence": float(row["confidence"]),
        **{k: row[k] for k in _POSITION_TEXT},
        **{k: None if math.isnan(float(row[k])) else float(row[k]) for k in _POSITION_FLOATS},
        "camera": cam if row["source"] == "fixed-camera" else None,  # a camera's rows are in its slot
        "revision": row["place_revision"],
    }


STREAMS[POSITIONS] = Stream(POSITIONS, POSITION_SCHEMA, position_row, position)


# The position layer (spatial/positions.py writes and reads it): where each person, pet, and the rig wearer was, one
# record shape for every source. A row is a subject (`entity`, None until someone is identified) at a time, in a
# place, with the room and plan coordinates when they are known (`x`, `y` in the plan frame of the place revision
# `place_revision` names; NaN for a room-only row), how sure, and which source said so ("fixed-camera", "rig-pose",
# or "zone"). `path` groups rows into the line a view draws, and `seen_by` is the camera of a fixed-camera row (the
# store keeps `camera` for the slot). The slot is the place and its writer (`positions.layer_slot`), so a read of
# one place opens only that place's files.
PLACED = "position.placed"
PLACED_SCHEMA = [
    *SPAN,
    ("source", "str"),
    ("confidence", "f32"),
    ("entity", "str?"),
    ("track", "str?"),
    ("path", "str?"),
    ("place", "str"),
    ("room", "str?"),
    ("room_id", "str?"),
    ("x", "f64"),
    ("y", "f64"),
    ("place_revision", "str?"),
    ("seen_by", "str?"),
    ("uncertain", "bool"),
]
_PLACED_TEXT = ("entity", "track", "path", "room", "room_id")


def placed_row(r: Mapping[str, Any]) -> Row:
    return {
        **_span_row(r),
        "source": str(r["source"]),
        "confidence": float(r["confidence"]),
        "place": str(r["place"]),
        **{k: None if r.get(k) is None else str(r[k]) for k in _PLACED_TEXT},
        **{k: math.nan if r.get(k) is None else float(r[k]) for k in ("x", "y")},
        "place_revision": None if r.get("revision") is None else str(r["revision"]),
        "seen_by": None if r.get("camera") is None else str(r["camera"]),
        "uncertain": bool(r.get("uncertain", False)),
    }


def placed(row: Mapping[str, Any], cam: str) -> Record:
    return {
        **_span(row, cam),
        "source": row["source"],
        "confidence": float(row["confidence"]),
        "place": row["place"],
        **{k: row[k] for k in _PLACED_TEXT},
        **{k: None if math.isnan(float(row[k])) else float(row[k]) for k in ("x", "y")},
        "revision": row["place_revision"],
        "camera": row["seen_by"],
        "uncertain": bool(row["uncertain"]),
    }


STREAMS[PLACED] = Stream(PLACED, PLACED_SCHEMA, placed_row, placed)
# The same records thinned to paths as they are written (one per path per `positions.COARSE_S`, and each path's last
# of a write): what a read of a long window scans, a tenth of the rows.
PLACED_PATHS = "position.path"
STREAMS[PLACED_PATHS] = Stream(PLACED_PATHS, PLACED_SCHEMA, placed_row, placed)


# Each fixed camera's pose over time (spatial/poses.py writes and reads it): one row per observation of where a
# camera was, in the map frame it was observed in (`frame`, a place revision's `Revision.frame`): a nightly seat's
# registration, a match with a walk's frame, or the owner's measured spot (position only). The slot is the camera.
# A camera that stays put is smoothed over its rows on read; one that moves starts a new stretch.
CAMERA_POSES = "camera.pose"
CAMERA_POSE_SCHEMA = [
    *SPAN,
    ("source", "str"),
    ("place", "str"),
    ("frame", "str?"),
    ("pose", "f32[16]"),  # camera to map, a 4x4 row by row
    ("quality", "f32"),  # a seat's fitness, a walk pair's inlier share; 1 for the owner's spot
]


def camera_pose_row(r: Mapping[str, Any]) -> Row:
    return {
        **_span_row(r),
        "source": str(r["source"]),
        "place": str(r["place"]),
        "frame": None if r.get("frame") is None else str(r["frame"]),
        "pose": [float(v) for v in r["pose"]],
        "quality": float(r["quality"]),
    }


def camera_pose(row: Mapping[str, Any], cam: str) -> Record:
    return {
        **_span(row, cam),
        "source": row["source"],
        "place": row["place"],
        "frame": row["frame"],
        "pose": [float(v) for v in row["pose"]],
        "quality": float(row["quality"]),
    }


STREAMS[CAMERA_POSES] = Stream(CAMERA_POSES, CAMERA_POSE_SCHEMA, camera_pose_row, camera_pose)


# Condensed history: past the 30-day window, the nightly retention pass (`retention.py`) merges the dense
# coverage rows, the device sightings, and the positions of each ended day into far fewer rows (contiguous intervals,
# visits, and 10 s paths) and drops the originals. A forget hides every row of its range, later appends included, so
# the merged rows go to a companion stream of the same schema, `<stream>.condensed`, and `Log` reads a stream and its
# companion together: a reader sees the condensed rows where the originals were.
CONDENSED_SUFFIX = ".condensed"


def condensed(stream: str) -> str:
    """The companion stream holding `stream`'s condensed history."""
    return stream + CONDENSED_SUFFIX


CONDENSED = (
    ACTION_COVERAGE,
    WALKBY_COVERAGE,
    ANIMAL_COVERAGE,
    ZONE_COVERAGE,
    NET_OBSERVATIONS,
    BLE_SIGHTINGS,
    POSITIONS,
    PLACED,
    PLACED_PATHS,
)
for _name in CONDENSED:
    _spec = STREAMS[_name]
    STREAMS[condensed(_name)] = Stream(condensed(_name), _spec.schema, _spec.to_row, _spec.to_record)


def root(state: str | Path | None = None) -> Path:
    """Where the segment store lives: `paths.state/segments`."""
    return Path(state if state is not None else config.path("paths.state")) / "segments"


def columns(stream: str, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Records as the column dict `SegmentWriter.append` takes."""
    spec = STREAMS[stream]
    rows = [spec.to_row(r) for r in records]
    out: dict[str, Any] = {name: [row[name] for row in rows] for name, _ in spec.schema}
    for name, kind in spec.schema:
        if re.fullmatch(r"f32\[\d+\]", kind):  # a fixed-size list goes in as one (rows, N) array
            out[name] = np.asarray(out[name], dtype=np.float32).reshape(len(rows), int(kind[4:-1]))
    return out


def records(stream: str, got: Mapping[str, Any]) -> list[Record]:
    """A `read_segments` (or follower) answer as records, in time order."""
    spec = STREAMS[stream]
    cams = got["camera"]
    out = []
    for i, cam in enumerate(cams):
        row = {name: got[name][i] for name, _ in spec.schema}
        out.append(spec.to_record(row, cam))
    out.sort(key=lambda r: (r["ts_start"], r.get("id", "")))
    return out


class Writers:
    """One `SegmentWriter` per (stream, camera), opened on first use. `resume` maps (stream, camera) to the
    `row_seq` a checkpoint saved: a writer reopened there drops what was written after it, so outputs a restarted
    worker derives again aren't kept twice."""

    def __init__(self, base: Path, resume: Mapping[str, Mapping[str, int]] | None = None) -> None:
        from continuum_py import SegmentWriter

        self._new = SegmentWriter
        self.base = base
        self.resume = {s: dict(c) for s, c in (resume or {}).items()}
        self._open: dict[tuple[str, str], Any] = {}

    def _writer(self, stream: str, cam: str) -> Any:
        key = (stream, cam)
        if key not in self._open:
            at = self.resume.get(stream, {}).get(cam)
            self._open[key] = self._new(self.base, stream, cam, STREAMS[stream].schema, resume_at=at)
        return self._open[key]

    def append(self, stream: str, cam: str, records: Sequence[Mapping[str, Any]]) -> None:
        if records:
            self._writer(stream, cam).append(columns(stream, records))

    def position(self, stream: str, cam: str) -> int:
        """The `row_seq` the camera's next row of `stream` gets (opening its writer)."""
        return int(self._writer(stream, cam).next_seq)

    def rewind(self, to: Mapping[str, Mapping[str, int]]) -> None:
        """Drops what was written past `to` ({stream: {camera: row_seq}}, from `position` or `sync`): the writers
        named reopen there on their next append. A pass that failed part way is then written again, not twice."""
        for (stream, cam), writer in list(self._open.items()):
            if cam in to.get(stream, {}):
                writer.close()
                del self._open[(stream, cam)]
        for stream, cams in to.items():
            self.resume.setdefault(stream, {}).update({c: int(v) for c, v in cams.items()})

    def sync(self) -> dict[str, dict[str, int]]:
        """Makes every write durable; returns each writer's next `row_seq`, by stream and camera, to checkpoint."""
        out: dict[str, dict[str, int]] = {s: dict(c) for s, c in self.resume.items()}
        for (stream, cam), writer in self._open.items():
            out.setdefault(stream, {})[cam] = int(writer.sync())
        return out

    def close(self) -> None:
        for writer in self._open.values():
            writer.close()
        self._open.clear()


HOUR_S = 3600.0
SETTLE_S = 120.0  # an hour this long over is kept: rows reach their files within a few seconds of their time
MAX_HOURS = 1000  # kept hours, before the cache starts over (a day of every stream read, with room)
_KEEP_HOURS = False
_HOURS: dict[tuple[Any, ...], tuple[Any, list[Record]]] = {}


def keep_hours(on: bool = True) -> None:
    """Keep each ended hour's decoded records of a window still going in this process, for a reader that reads it
    again and again (Review's day history of today, every minute). Each kept hour is checked against its files'
    stamp (`Log.stamp`) on every read, so a row written late or a purge shows at once; only the newest hours are read
    from the store. Off by default: a process that reads a window once gains nothing. Today's hours are a lot of
    records: Review's day-history worker was about 1.1 GB by late afternoon on the Linux machine, dropped when the
    day turns."""
    global _KEEP_HOURS
    _KEEP_HOURS = on
    if not on:
        _HOURS.clear()


def _segment_moved(exc: BaseException) -> bool:
    """Whether `read_segments` failed because a part vanished or was renamed while it was being opened."""
    if isinstance(exc, OSError) and exc.errno == 2:
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc)
        return "(os error 2)" in msg or "No such file" in msg
    return False


class Log:
    """Reads of the segment streams, as records."""

    def __init__(self, base: Path) -> None:
        self.base = base

    def stamp(self, start: float, end: float, streams: Sequence[str] | None = None) -> tuple[tuple[str, int, int], ...]:
        """What the store holds for [start, end), cheaply: the files a read of the window opens, with their sizes and
        mtimes, and the deletions log. The store files each row under its own UTC hour, and a read opens the parts from
        the window's first hour, less the camera's longest span (`.maxspan`, so a track that began earlier is found),
        to its last; this lists the same parts (from an hour earlier at least). It moves when a row of the window is
        written late or purged, and not otherwise: a reader of an ended stretch keys its answer on it.
        `streams` limits it to those; missing directories are nothing."""
        import os

        last = time.strftime("%Y%m%d%H", time.gmtime(max(start, end - 1e-6)))
        out: list[tuple[str, int, int]] = []
        try:
            deletions = (self.base / ".deletions").stat()
            out.append((".deletions", deletions.st_size, deletions.st_mtime_ns))
        except OSError:
            pass
        try:
            names = (
                sorted({*streams, *(condensed(s) for s in streams if s in CONDENSED)})
                if streams is not None
                else sorted(e.name for e in os.scandir(self.base))
            )
        except OSError:
            return tuple(out)
        for name in names:
            if name.startswith("."):
                continue
            try:
                cams = [c for c in os.scandir(self.base / name) if c.is_dir()]
            except OSError:
                continue
            for cam in cams:
                try:
                    reach = int((Path(cam.path) / ".maxspan").read_text().strip()) / 1e9
                except OSError, ValueError:
                    reach = 0.0
                first = time.strftime("%Y%m%d%H", time.gmtime(start - max(3600.0, reach)))
                for f in os.scandir(cam.path):
                    if first <= f.name[:10] <= last:
                        try:
                            st = f.stat()
                        except OSError:
                            continue
                        out.append((f"{name}/{cam.name}/{f.name}", st.st_size, st.st_mtime_ns))
        return tuple(sorted(out))

    def range(self, stream: str, start: float, end: float, cameras: Sequence[str] | None = None) -> list[Record]:
        """Every record of `stream` overlapping [start, end), in time order. With `keep_hours` on, a window over an hour
        long is read an hour at a time, each ended hour's records kept (see `keep_hours`)."""
        if end <= start:
            return []
        if _KEEP_HOURS and end - start > HOUR_S and end > time.time():
            return _face_groups_now(stream, self._by_hour(stream, start, end, cameras))
        return _face_groups_now(stream, self._read(stream, start, end, cameras))

    def _by_hour(self, stream: str, start: float, end: float, cameras: Sequence[str] | None) -> list[Record]:
        settled = time.time() - SETTLE_S
        # Only the window still going keeps its hours (an ended day's reader keeps its whole answer instead): what's
        # kept from before this window, yesterday's hours once the day has turned, goes.
        for old in [k for k in _HOURS if k[0] == str(self.base) and k[1] == stream and k[3] <= start]:
            del _HOURS[old]
        out: list[Record] = []
        at = start
        while at < end:
            upto = min(end, (math.floor(at / HOUR_S) + 1) * HOUR_S)
            if upto <= settled:
                key = (str(self.base), stream, at, upto, tuple(cameras) if cameras is not None else None)
                stamp = self.stamp(at, upto, (stream,))
                kept = _HOURS.get(key)
                if kept is None or kept[0] != stamp:
                    kept = (stamp, self._read(stream, at, upto, cameras))
                    if len(_HOURS) >= MAX_HOURS:
                        _HOURS.clear()
                    _HOURS[key] = kept
                rows = kept[1]
            else:
                rows = self._read(stream, at, upto, cameras)
            # A record spanning an hour's end is in both hours' reads, and ids aren't unique in every stream: each
            # record is taken from the one hour holding its start (or the window's, for one that began before it),
            # whose read always has it.
            out.extend(dict(r) for r in rows if at <= max(r["ts_start"], start) < upto)  # copies: a reader that
            # changes a record can't change the kept hour
            at = upto
        out.sort(key=lambda r: (r["ts_start"], r.get("id", "")))
        return out

    def _read(self, stream: str, start: float, end: float, cameras: Sequence[str] | None) -> list[Record]:
        from continuum_py import read_segments

        got = None
        for attempt in range(2):
            try:
                got = read_segments(
                    self.base,
                    stream,
                    STREAMS[stream].schema,
                    ns(start),
                    ns(end),
                    list(cameras) if cameras is not None else None,
                    overlapping=True,
                )
                break
            except (OSError, RuntimeError) as e:
                if not _segment_moved(e):
                    raise
                if attempt == 0:
                    continue
                raise
        assert got is not None
        out = records(stream, got)
        if stream in CONDENSED:  # its condensed history too
            out = sorted(
                [*out, *self._read(condensed(stream), start, end, cameras)],
                key=lambda r: (r["ts_start"], str(r.get("id", ""))),
            )
        return out


_FACE_GROUP = re.compile(r"Face (\d+)")


def _face_groups_now(stream: str, rows: list[Record]) -> list[Record]:
    """Face sightings with their unnamed group as it is now: a group folded into another since (a stranger
    split across groups, vision/faces.py FOLD_SIM) is read as the group it went into, so its looks stay with it. The
    rows keep what was recorded; only the read changes. Every other stream's rows pass through."""
    if stream not in (FACE_MATCHES, FACE_EMBEDDINGS) or not rows:
        return rows
    from liberational.vision import faces

    if stream == FACE_MATCHES:
        aliases = faces.group_aliases()
        if not aliases:
            return rows
        out = []
        for r in rows:
            m = _FACE_GROUP.fullmatch(r["cluster"]) if isinstance(r.get("cluster"), str) else None
            to = aliases.get(int(m.group(1))) if m else None
            out.append(r if to is None else {**r, "cluster": f"Face {to}"})
        return out
    subjects = faces.group_alias_subjects()
    if not subjects:
        return rows
    return [r if r.get("group") not in subjects else {**r, "group": subjects[r["group"]]} for r in rows]


class Tail:
    """New records of one stream from `since` on, as they're appended: the store's follower, started at each
    camera's newest row seen in the last `lookback_s` (a follower with no cursor would replay every row kept). A
    camera with nothing that recent starts from its first row, and what's older than `since` is skipped."""

    def __init__(self, base: Path, stream: str, since: float, lookback_s: float = 86400.0) -> None:
        from continuum_py import SegmentFollower, read_segments

        self.stream, self.since = stream, since
        schema = STREAMS[stream].schema
        got = read_segments(base, stream, schema, ns(since - lookback_s), ns(since + lookback_s), None)
        cursor: dict[str, int] = {}
        for cam, seq in zip(got["camera"], got["row_seq"], strict=True):
            cursor[cam] = max(cursor.get(cam, 0), int(seq) + 1)
        self._follower = SegmentFollower(base, stream, schema, cursor)
        self._pending = [r for r in records(stream, got) if r["ts_start"] >= since]

    def poll(self, timeout_s: float = 1.0) -> list[Record]:
        """What's new since the last poll, waiting up to `timeout_s` for it (blocking: call it off the event loop)."""
        found, self._pending = self._pending, []
        if not found:
            found = records(self.stream, self._follower.poll(timeout_s))
        return _face_groups_now(self.stream, [r for r in found if r["ts_start"] >= self.since])


def at(state: str | Path | None = None) -> Log:
    return Log(root(state))


def new_id() -> str:
    return str(uuid.uuid7())
