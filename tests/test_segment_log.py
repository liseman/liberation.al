"""The segment streams (src/liberational/segment_log.py): records survive the typed columns, and the store keeps them
where a checkpoint said."""

import json
import math
import threading
import time

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from liberational import segment_log as S

T = 1_800_000_000.0

# Producer times are float seconds; the store keeps nanoseconds, so a round trip is exact to well under a microsecond.
times = st.floats(min_value=T, max_value=T + 86400, allow_nan=False)
coords = st.floats(min_value=0, max_value=1, allow_nan=False, width=32)
words = st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=12)


@st.composite
def intervals(draw):
    return [sorted(draw(st.lists(times, min_size=2, max_size=2))) for _ in range(draw(st.integers(0, 3)))]


@st.composite
def boundaries(draw):
    b = {"status": draw(st.sampled_from(["supported", "unknown"])), "reason": draw(words)}
    if draw(st.booleans()):
        b["edge"] = draw(st.sampled_from(["left", "right", "top", "bottom"]))
    if draw(st.booleans()):
        b["time"] = sorted(draw(st.lists(times, min_size=2, max_size=2)))
    return b


def span(draw):
    start = draw(times)
    return {"id": S.new_id(), "ts_start": start, "ts_end": start + draw(st.floats(0, 600)), "cam": "cam0"}


@st.composite
def track_records(draw):
    return {
        **span(draw),
        "track_id": draw(words),
        "version": "movement-1",
        "points": [[draw(times), draw(coords), draw(coords)] for _ in range(draw(st.integers(0, 5)))],
        "supported": draw(intervals()),
        "gaps": draw(intervals()),
        "entry": draw(boundaries()),
        "exit": draw(boundaries()),
        "direction": draw(st.sampled_from(["left", "right", "up", "down", "stationary", "unknown"])),
        "camera_motion": draw(st.sampled_from(["stable", "moving", "unknown"])),
        "finished": draw(st.booleans()),
        "closure": draw(words),
    }


@st.composite
def coverage_records(draw):
    return {
        **span(draw),
        "version": "movement-1",
        "frames": draw(st.integers(0, 10_000)),
        "supported": draw(intervals()),
        "gaps": draw(intervals()),
        "tracking": draw(st.sampled_from(["complete", "partial"])),
        "camera_motion": draw(st.sampled_from(["stable", "moving", "unknown"])),
        "finished": draw(st.booleans()),
        "closure": draw(words),
    }


@st.composite
def zone_visit_records(draw):
    return {
        **span(draw),
        "zone": "zone-" + draw(words),
        "zone_name": draw(st.text(min_size=1, max_size=20)),
        "definition": S.new_id(),
        "subject_kind": draw(st.sampled_from(["person", "animal"])),
        "track_id": draw(words),
        "label": draw(st.one_of(st.none(), st.sampled_from(["person", "cat"]))),
        "dwell_s": draw(st.floats(0, 3600)),
        "frames": draw(st.integers(0, 10_000)),
        "entry_seen": draw(st.booleans()),
        "exit_seen": draw(st.booleans()),
        "ended": draw(st.sampled_from(["left", "lost", "censored", "zones_changed"])),
        "candidate": True,
        "interpretation": {"producer": "zone-visits", "anchor": "box_bottom_centre", "version": "zone-visits-1"},
    }


@st.composite
def zone_coverage_records(draw):
    return {
        **span(draw),
        "subject_kind": draw(st.sampled_from(["person", "animal"])),
        "definition": S.new_id(),
        # Zone ids are "zone-" and up to 35 more characters of any kind (see/zones.py).
        "zones": draw(st.lists(st.text(max_size=35).map(lambda t: "zone-" + t), max_size=16)),
        "frames": draw(st.integers(0, 10_000)),
    }


@st.composite
def object_match_records(draw):
    start = draw(times)
    return {
        "id": S.new_id(),
        "ts_start": start,
        "ts_end": start,
        "cam": "cam0",
        "object_id": draw(words),
        "name": draw(st.text(min_size=1, max_size=20)),
        "bbox": [draw(st.floats(0, 4000, width=32)) for _ in range(4)],
        "score": draw(st.floats(0, 1)),
        "kind": "visual_match",
        "bbox_space": "source",
        "frame_id": draw(st.one_of(st.none(), st.uuids().map(str))),
        "frame_seq": draw(st.one_of(st.none(), st.integers(0, 2**40))),
        "evidence": draw(st.text(alphabet="ABCDEFabcdef0123456789+/=", max_size=64)),
    }


@st.composite
def face_match_records(draw):
    start, mid = draw(times), S.new_id()
    return {
        "id": mid,
        "ts_start": start,
        "ts_end": start,
        "cam": "cam0",
        "detection": mid,
        "track_id": str(draw(st.integers(1, 2**60))),
        "frame_id": str(draw(st.uuids())),
        "bbox": [draw(st.floats(0, 1, width=32)) for _ in range(4)],
        "det": draw(st.floats(0, 1)),
        "name": draw(st.one_of(st.none(), st.text(min_size=1, max_size=20))),
        "sim": draw(st.floats(-1, 1)),
        "cluster": draw(st.one_of(st.none(), st.text(min_size=1, max_size=12))),
        "quality": draw(st.floats(0, 1)),
    }


@st.composite
def face_visit_records(draw):
    start = draw(times)
    shot = st.fixed_dictionaries(
        {
            "detection": st.uuids().map(str),
            "ts": st.floats(0, 2e9),
            "bbox": st.lists(st.floats(0, 1), min_size=4, max_size=4),
            "quality": st.floats(0, 1),
            "sim": st.floats(-1, 1),
        }
    )
    return {
        "id": S.new_id(),
        "ts_start": start,
        "ts_end": start + draw(st.floats(0, 3600)),
        "cam": "cam0",
        "name": draw(st.text(min_size=1, max_size=20)),
        "best_sim": draw(st.floats(-1, 1)),
        "samples": draw(st.integers(1, 10_000)),
        "shots": draw(st.lists(shot, max_size=3)),
    }


def unit_vector(seed):
    v = np.random.default_rng(seed).standard_normal(512).astype(np.float32)
    return [float(x) for x in v / np.linalg.norm(v)]


@st.composite
def face_embedding_records(draw):
    start = draw(times)
    return {
        "id": S.new_id(),
        "ts_start": start,
        "ts_end": start,
        "cam": "cam0",
        "match_id": S.new_id(),
        "track_id": str(draw(st.integers(1, 2**60))),
        "name": draw(st.one_of(st.none(), st.text(min_size=1, max_size=20))),
        "person": draw(st.one_of(st.none(), st.uuids().map(str))),
        "group": draw(st.one_of(st.none(), st.uuids().map(str))),
        # 512 drawn floats is too large an input for Hypothesis; a seeded unit vector covers the column as well.
        "vector": unit_vector(draw(st.integers(0, 2**32 - 1))),
        "model": "insightface-buffalo_l/arcface",
    }


@st.composite
def face_signal_records(draw):
    start = draw(times)
    return {
        "id": S.new_id(),
        "ts_start": start,
        "ts_end": start,
        "cam": "cam0",
        "match_id": S.new_id(),
        "head": [draw(st.floats(0, 1, width=32)) for _ in range(5)],
        "frontal": draw(st.floats(0, 1)),
        # NaN on disk stands for a recognizer that gave no norm, so None must come back as None.
        "norm": draw(st.one_of(st.none(), st.floats(0, 100))),
    }


@st.composite
def caption_records(draw):
    start = draw(times)
    grounded = draw(st.booleans())
    n = draw(st.integers(0, 4)) if grounded else 0
    normed = draw(st.booleans())
    stamp = draw(times)
    objs = [
        {
            "label": draw(st.text(min_size=1, max_size=12)),
            "bbox": [draw(st.integers(0, 4000)) for _ in range(4)],
            "ts": stamp,
            "bbox_norm": [draw(coords) for _ in range(4)] if normed else None,
        }
        for _ in range(n)
    ]
    fresh = draw(st.booleans())
    record = {
        "id": S.new_id(),
        "ts_start": start,
        "ts_end": start,
        "cam": "cam0",
        "frame_id": draw(st.one_of(st.none(), st.uuids().map(str))),
        "frame_seq": draw(st.integers(0, 2**40)),
        "yaw_deg": 0.0,
        "caption": draw(st.text(max_size=80)),
        "faces": None,
        "objects": objs if grounded else None,
        "source": draw(st.sampled_from(["camera", "ingest"])),
        "synthetic": False,
        "gated": draw(st.booleans()),
        "observation": {
            "version": 1,
            "status": "fresh" if fresh else "not_observed",
            "observed_at": start if fresh else None,
            "camera": "cam0",
            "backend": "ollama",
            "prompt_version": "independent-v1",
        },
        "change": draw(
            st.one_of(st.none(), st.fixed_dictionaries({"mad": st.floats(0, 255), "frac": st.floats(0, 1)}))
        ),
        "brightness": {"mean": draw(st.floats(0, 255)), "dark_fraction": draw(st.floats(0, 1))},
        "ms": {"vlm": draw(st.integers(0, 60_000))},
    }
    if draw(st.booleans()):
        record.update(unchanged=True, since=draw(st.one_of(st.none(), times)))
    if draw(st.booleans()):
        record["deferred"] = True
    return record


@st.composite
def camera_objects_records(draw):
    start = draw(times)
    normed = draw(st.booleans())
    stamp = draw(times)
    objs = [
        {
            "label": draw(st.text(min_size=1, max_size=12)),
            "bbox": [draw(st.integers(0, 4000)) for _ in range(4)],
            "ts": stamp,
            "bbox_norm": [draw(coords) for _ in range(4)] if normed else None,
        }
        for _ in range(draw(st.integers(0, 4)))
    ]
    observation = {
        "version": 1,
        "status": draw(st.sampled_from(["fresh", "cached", "failed", "unevaluated"])),
        "observed_at": draw(st.one_of(st.none(), times)),
        "attempted_at": draw(st.one_of(st.none(), times)),
    }
    if draw(st.booleans()):
        observation["reason"] = draw(st.sampled_from(["cached_grounding", "grounding_failed"]))
    return {
        "id": S.new_id(),
        "ts_start": start,
        "ts_end": start,
        "cam": "cam0",
        "frame_id": draw(st.one_of(st.none(), st.uuids().map(str))),
        "frame_seq": draw(st.integers(0, 2**40)),
        "objects": objs,
        "source": "ollama",
        "observation": observation,
    }


@st.composite
def action_coverage_records(draw):
    start = draw(times)
    return {
        "ts_start": start,
        "ts_end": start + draw(st.floats(0, 5)),
        "cam": "cam0",
        "frames": draw(st.integers(0, 300)),
        "model": "yolox+rtmpose",
    }


def test_action_coverage_is_read_with_the_schema_see_actions_writes():
    from liberational.see.actions import COVERAGE_SCHEMA

    assert S.ACTION_COVERAGE_SCHEMA == COVERAGE_SCHEMA


@settings(max_examples=60, deadline=None)
@given(record=action_coverage_records())
def test_a_coverage_span_survives_its_columns_and_is_named_by_camera_and_start(record):
    spec = S.STREAMS[S.ACTION_COVERAGE]
    row = spec.to_row(record)
    back = spec.to_record(row, "cam0")
    assert back["id"] == f"cam0:{row['ts_ns']}"
    assert same({k: v for k, v in back.items() if k != "id"}, record)


@st.composite
def presence_records(draw):
    return {
        **span(draw),
        "frames": draw(st.integers(1, 10**6)),
        "max_people": draw(st.integers(1, 20)),
        "tracks": draw(st.integers(1, 256)),
        "left_censored": draw(st.booleans()),
        "right_censored": draw(st.booleans()),
    }


@st.composite
def place_visit_records(draw):
    return {**span(draw), "place": S.new_id(), "producer": draw(words), "basis": draw(words)}


RECORDS = {
    S.MOVEMENT_TRACKS: track_records(),
    S.MOVEMENT_COVERAGE: coverage_records(),
    S.ZONE_VISITS: zone_visit_records(),
    S.ZONE_COVERAGE: zone_coverage_records(),
    S.OBJECT_MATCHES: object_match_records(),
    S.CAPTIONS: caption_records(),
    S.CAMERA_OBJECTS: camera_objects_records(),
    S.FACE_MATCHES: face_match_records(),
    S.FACE_VISITS: face_visit_records(),
    S.FACE_EMBEDDINGS: face_embedding_records(),
    S.FACE_SIGNALS: face_signal_records(),
    S.PRESENCE_SPANS: presence_records(),
    S.PLACE_VISITS: place_visit_records(),
}


def same(a, b):
    """Equal records, with times to the microsecond and coordinates to float32."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):  # a vector column read back as its float32 array
        a, b = (x.tolist() if isinstance(x, np.ndarray) and x.dtype == np.float32 else x for x in (a, b))
        return isinstance(a, list) and same(a, b)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, float) and not isinstance(b, bool):
        return isinstance(b, int | float) and math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)
    return a == b


@pytest.mark.parametrize("stream", sorted(RECORDS))
@settings(max_examples=60, deadline=None)
@given(data=st.data())
def test_a_record_survives_its_typed_columns(stream, data):
    record = data.draw(RECORDS[stream])
    spec = S.STREAMS[stream]
    row = spec.to_row(record)
    assert set(row) == {name for name, _ in spec.schema}, "the row is exactly the schema's columns"
    assert same(spec.to_record(row, "cam0"), record)


def test_a_bluetooth_record_json_wrote_with_a_nan_still_reads():
    row = S.ble_row({"id": "a", "start": 10.0, "end": 12.0, "rssi": {"mean": float("nan"), "max": -60}})
    assert "NaN" in row["record"]  # what json.dumps writes, and orjson refuses
    got = S.ble(row, "cam0")
    assert math.isnan(got["rssi"]["mean"]) and got["rssi"]["max"] == -60


TRACK = {
    "track_id": "run:cam0:1",
    "version": "movement-1",
    "points": [[T + 1, 0.25, 0.5], [T + 2, 0.75, 0.5]],
    "supported": [[T + 1, T + 2]],
    "gaps": [],
    "entry": {"status": "supported", "reason": "edge_motion_and_evaluated_absence", "edge": "left", "time": [T, T + 1]},
    "exit": {"status": "unknown", "reason": "shutdown"},
    "direction": "right",
    "camera_motion": "stable",
    "finished": True,
    "closure": "absence",
}
COVERAGE = {
    "version": "movement-1",
    "frames": 20,
    "supported": [[T, T + 10]],
    "gaps": [],
    "tracking": "complete",
    "camera_motion": "stable",
    "finished": False,
    "closure": "fragment",
}
FIXED = {
    S.MOVEMENT_TRACKS: TRACK,
    S.MOVEMENT_COVERAGE: COVERAGE,
    S.ZONE_VISITS: {
        "zone": "zone-litter",
        "zone_name": "Litter box",
        "definition": "0190a5b8-0000-7000-8000-000000000001",
        "subject_kind": "animal",
        "track_id": "run:cam0:animal:cat:1",
        "label": "cat",
        "dwell_s": 14.5,
        "frames": 30,
        "entry_seen": True,
        "exit_seen": True,
        "ended": "left",
        "candidate": True,
        "interpretation": {"producer": "zone-visits", "anchor": "box_bottom_centre", "version": "zone-visits-1"},
    },
    S.ZONE_COVERAGE: {
        "subject_kind": "person",
        "definition": "0190a5b8-0000-7000-8000-000000000001",
        "zones": ["zone-litter", "zone-door"],
        "frames": 120,
    },
    S.FACE_MATCHES: {
        "detection": "",
        "track_id": "1793733746000001",
        "frame_id": "0190a5b8-0000-7000-8000-000000000003",
        "bbox": [0.25, 0.125, 0.375, 0.3125],
        "det": 0.875,
        "name": "Joe",
        "sim": 0.625,
        "cluster": None,
        "quality": 0.5,
    },
    S.FACE_VISITS: {
        "name": "Joe",
        "best_sim": 0.625,
        "samples": 12,
        "shots": [
            {
                "detection": "0190a5b8-0000-7000-8000-000000000004",
                "ts": T + 11,
                "bbox": [0.25, 0.125, 0.375, 0.3125],
                "quality": 0.5,
                "sim": 0.625,
            }
        ],
    },
    S.FACE_EMBEDDINGS: {
        "match_id": "0190a5b8-0000-7000-8000-000000000004",
        "track_id": "1793733746000001",
        "name": None,
        "person": None,
        "group": "0190a5b8-0000-7000-8000-000000000005",
        "vector": unit_vector(7),
        "model": "insightface-buffalo_l/arcface",
    },
    S.FACE_SIGNALS: {
        "match_id": "0190a5b8-0000-7000-8000-000000000004",
        "head": [0.125, 0.25, 0.0, 0.875, 0.75],
        "frontal": 0.0,
        "norm": None,
    },
    S.OBJECT_MATCHES: {
        "object_id": "obj-wallet",
        "name": "wallet",
        "bbox": [120.5, 80.25, 64.0, 48.0],
        "score": 0.82,
        "kind": "visual_match",
        "bbox_space": "source",
        "frame_id": "0190a5b8-0000-7000-8000-000000000002",
        "frame_seq": 1234,
        "evidence": "/9j/4AAQSkZJRg==",
    },
    S.CAPTIONS: {
        "frame_id": "0190a5b8-0000-7000-8000-000000000003",
        "frame_seq": 42,
        "yaw_deg": 0.0,
        "caption": "a grey cat on the sofa",
        "faces": None,
        "objects": [{"label": "cat", "bbox": [10, 20, 30, 40], "ts": T + 9, "bbox_norm": [0.25, 0.5, 0.375, 0.75]}],
        "source": "camera",
        "synthetic": False,
        "gated": False,
        "observation": {
            "version": 1,
            "status": "fresh",
            "observed_at": T + 10,
            "camera": "cam0",
            "backend": "ollama",
            "prompt_version": "independent-v1",
        },
        "change": {"mad": 12.5, "frac": 0.25},
        "brightness": {"mean": 91.5, "dark_fraction": 0.125},
        "ms": {"vlm": 850},
    },
    S.PRESENCE_SPANS: {"frames": 90, "max_people": 2, "tracks": 3, "left_censored": False, "right_censored": True},
    S.PLACE_VISITS: {"place": "0190a5b8-0000-7000-8000-000000000009", "producer": "place-detect-1", "basis": "wifi"},
    S.CAMERA_OBJECTS: {
        "frame_id": "0190a5b8-0000-7000-8000-000000000005",
        "frame_seq": 7,
        "objects": [{"label": "lamp", "bbox": [64, 96, 192, 288], "ts": T + 9, "bbox_norm": [0.1, 0.2, 0.3, 0.6]}],
        "source": "ollama",
        "observation": {
            "version": 1,
            "status": "cached",
            "observed_at": T + 9,
            "attempted_at": None,
            "reason": "cached_grounding",
        },
    },
}


def one(stream, *, ts_start, ts_end):
    record = {**FIXED[stream], "id": S.new_id(), "ts_start": ts_start, "ts_end": ts_end, "cam": "cam0"}
    if "detection" in record:  # a face match is its own detection
        record["detection"] = record["id"]
    return record


@pytest.fixture
def store(tmp_path):
    pytest.importorskip("continuum_py").SegmentWriter  # noqa: B018 (the native store must be built)
    return tmp_path / "segments"


def test_every_stream_round_trips_through_the_native_store(store):
    written = {stream: one(stream, ts_start=T + 10, ts_end=T + 20) for stream in RECORDS}
    writers = S.Writers(store)
    for stream, record in written.items():
        writers.append(stream, "cam0", [record])
    writers.sync()
    writers.close()
    log = S.Log(store)
    for stream, record in written.items():
        (back,) = log.range(stream, T, T + 60)
        assert same(back, record), stream


def test_face_embedding_vectors_read_back_as_read_only_float32_rows(store):
    first, second = (
        one(S.FACE_EMBEDDINGS, ts_start=T + 1, ts_end=T + 1),
        one(S.FACE_EMBEDDINGS, ts_start=T + 2, ts_end=T + 2),
    )
    second["vector"] = unit_vector(8)
    writers = S.Writers(store)
    writers.append(S.FACE_EMBEDDINGS, "cam0", [first, second])
    writers.sync()
    writers.close()
    a, b = S.Log(store).range(S.FACE_EMBEDDINGS, T, T + 60)
    assert a["vector"].dtype == np.float32 and a["vector"].shape == (512,)
    assert np.allclose(a["vector"], first["vector"]) and np.allclose(b["vector"], second["vector"])
    with pytest.raises(ValueError, match="read-only"):
        a["vector"][0] = 2.0  # it shares the store's block with the other rows
    assert np.allclose(b["vector"], second["vector"])


def test_a_face_embedding_record_leaves_the_callers_array_writable():
    mine = np.asarray(unit_vector(3), np.float32)
    row = S.face_embedding_row(one(S.FACE_EMBEDDINGS, ts_start=T, ts_end=T) | {"vector": mine})
    row["vector"] = mine
    got = S.face_embedding(row, "cam0")
    assert not got["vector"].flags.writeable
    mine[0] = 2.0  # still the caller's to change
    assert got["vector"][0] == 2.0, "a view, not a copy"


def test_redimnet_voice_embedding_round_trips_through_the_native_store(store):
    vector = [1.0, *([0.0] * 191)]
    record = {
        "id": S.new_id(),
        "start_ns": int(T * 1e9),
        "end_ns": int((T + 2) * 1e9),
        "voiceprint": S.new_id(),
        "person": None,
        "utterance": S.new_id(),
        "vector": vector,
    }
    writers = S.Writers(store)
    writers.append(S.VOICE_EMBEDDINGS, "mic", [record])
    writers.close()
    (back,) = S.Log(store).range(S.VOICE_EMBEDDINGS, T, T + 3)
    assert back["id"] == record["id"]
    assert back["vector"] == vector
    assert back["voiceprint"] == record["voiceprint"]


def test_a_span_that_started_before_the_window_is_found_in_it(store):
    early = one(S.MOVEMENT_TRACKS, ts_start=T, ts_end=T + 100)
    later = one(S.MOVEMENT_TRACKS, ts_start=T + 300, ts_end=T + 310)
    writers = S.Writers(store)
    writers.append(S.MOVEMENT_TRACKS, "cam0", [early, later])
    writers.close()
    log = S.Log(store)
    assert [r["id"] for r in log.range(S.MOVEMENT_TRACKS, T + 50, T + 60)] == [early["id"]]
    assert log.range(S.MOVEMENT_TRACKS, T + 200, T + 250) == []


def test_a_writer_reopened_at_a_checkpoint_drops_what_was_written_after_it(store):
    rows = [one(S.MOVEMENT_COVERAGE, ts_start=T + n, ts_end=T + n + 1) for n in range(5)]
    writers = S.Writers(store)
    writers.append(S.MOVEMENT_COVERAGE, "cam0", rows[:3])
    saved = writers.sync()
    assert saved == {S.MOVEMENT_COVERAGE: {"cam0": 3}}
    writers.append(S.MOVEMENT_COVERAGE, "cam0", rows[3:])  # past the checkpoint, then a crash
    writers.close()
    again = S.Writers(store, saved)
    again.append(S.MOVEMENT_COVERAGE, "cam0", [rows[3]])  # the replay derives the fourth again
    again.close()
    got = S.Log(store).range(S.MOVEMENT_COVERAGE, T, T + 10)
    assert [r["id"] for r in got] == [r["id"] for r in rows[:4]]


def test_a_rewound_writer_drops_a_failed_pass_so_its_retry_lands_once(store):
    rows = [one(S.CAMERA_OBJECTS, ts_start=T + n, ts_end=T + n) for n in range(4)]
    writers = S.Writers(store)
    writers.append(S.CAMERA_OBJECTS, "cam0", rows[:1])
    mark = {S.CAMERA_OBJECTS: {"cam0": writers.position(S.CAMERA_OBJECTS, "cam0")}}
    writers.append(S.CAMERA_OBJECTS, "cam0", rows[1:3])  # a pass that then fails
    writers.rewind(mark)
    writers.append(S.CAMERA_OBJECTS, "cam0", rows[1:])  # its retry
    writers.close()
    got = S.Log(store).range(S.CAMERA_OBJECTS, T, T + 10)
    assert [r["id"] for r in got] == [r["id"] for r in rows]


def test_a_tail_gives_what_is_appended_after_it_starts_and_nothing_before(store):
    old = [one(S.CAPTIONS, ts_start=T + n, ts_end=T + n) for n in range(3)]
    writers = S.Writers(store)
    writers.append(S.CAPTIONS, "cam0", old)
    writers.sync()
    tail = S.Tail(store, S.CAPTIONS, since=T + 100)
    assert tail.poll(0.0) == []
    new = one(S.CAPTIONS, ts_start=T + 101, ts_end=T + 101)
    writers.append(S.CAPTIONS, "cam0", [new])
    writers.sync()
    assert [r["id"] for r in tail.poll(2.0)] == [new["id"]]
    writers.close()


@pytest.fixture
def kept_hours():
    S.keep_hours()
    yield
    S.keep_hours(False)


def write(store, records, stream=S.MOVEMENT_TRACKS):
    writers = S.Writers(store)
    writers.append(stream, "cam0", records)
    writers.sync()
    writers.close()


def as_rows(records):
    return [json.dumps(r, sort_keys=True, default=str) for r in records]


def test_a_window_still_going_reads_its_ended_hours_once_and_answers_as_a_plain_read(store, kept_hours, monkeypatch):
    """#2398: Review rereads today's history every minute; its ended hours' records are kept."""
    now = time.time()
    hour = (now // 3600) * 3600
    start = hour - 3 * 3600
    rows = [one(S.MOVEMENT_TRACKS, ts_start=start + k * 1500, ts_end=start + k * 1500 + 2400) for k in range(7)]
    rows.append({**rows[0], "ts_start": start + 10, "ts_end": start + 20})  # the same id twice: both are rows
    write(store, rows)
    log = S.Log(store)
    plain = log._read(S.MOVEMENT_TRACKS, start, now + 3600, None)
    reads = []
    real = S.Log._read
    monkeypatch.setattr(S.Log, "_read", lambda self, *a: reads.append(a[1:3]) or real(self, *a))
    first = log.range(S.MOVEMENT_TRACKS, start, now + 3600)
    assert as_rows(first) == as_rows(plain), "the same records, a span across hours once, duplicate ids kept"
    reads.clear()
    first[0]["ts_start"] = 0  # a reader changing its copy changes nothing kept
    again = log.range(S.MOVEMENT_TRACKS, start, now + 3600)
    assert as_rows(again) == as_rows(plain)
    assert all(lo >= hour - S.HOUR_S for lo, _ in reads), f"only the newest hours read again: {reads}"


def test_a_row_written_late_into_a_kept_hour_shows_on_the_next_read(store, kept_hours):
    now = time.time()
    start = (now // 3600) * 3600 - 3 * 3600
    write(store, [one(S.MOVEMENT_TRACKS, ts_start=start + 60, ts_end=start + 70)])
    log = S.Log(store)
    assert len(log.range(S.MOVEMENT_TRACKS, start, now + 3600)) == 1
    write(store, [one(S.MOVEMENT_TRACKS, ts_start=start + 120, ts_end=start + 130)])  # late, into the oldest hour
    assert len(log.range(S.MOVEMENT_TRACKS, start, now + 3600)) == 2


def test_an_ended_window_keeps_nothing(store, kept_hours):
    write(store, [one(S.MOVEMENT_TRACKS, ts_start=T + 60, ts_end=T + 70)])
    assert len(S.Log(store).range(S.MOVEMENT_TRACKS, T, T + 4 * 3600)) == 1
    assert not S._HOURS, "an ended day's reader keeps its whole answer instead"


def test_a_part_listed_then_renamed_before_open_is_retried(store, monkeypatch):
    """#3100: a pass lists an open part, the writer seals it, the read's ENOENT is retried once."""
    continuum_py = pytest.importorskip("continuum_py")
    record = one(S.FACE_MATCHES, ts_start=T, ts_end=T)
    writers = S.Writers(store)
    writers.append(S.FACE_MATCHES, "living-room", [record])
    writers.close()
    real = continuum_py.read_segments
    calls = {"n": 0}

    def missing_once(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError(f"{store}/face.match/living-room/x.arrows: No such file or directory (os error 2)")
        return real(*args, **kwargs)

    monkeypatch.setattr(continuum_py, "read_segments", missing_once)
    got = S.Log(store).range(S.FACE_MATCHES, T, T + 10)
    assert [r["id"] for r in got] == [record["id"]]
    assert calls["n"] == 2


def test_a_part_still_missing_after_retry_raises(store, monkeypatch):
    continuum_py = pytest.importorskip("continuum_py")
    record = one(S.FACE_MATCHES, ts_start=T, ts_end=T)
    writers = S.Writers(store)
    writers.append(S.FACE_MATCHES, "living-room", [record])
    writers.close()
    missing = f"{store}/face.match/living-room/x.arrows: No such file or directory (os error 2)"
    calls = {"n": 0}

    def always_missing(*args, **kwargs):
        calls["n"] += 1
        raise RuntimeError(missing)

    monkeypatch.setattr(continuum_py, "read_segments", always_missing)
    with pytest.raises(RuntimeError, match=r"os error 2"):
        S.Log(store).range(S.FACE_MATCHES, T, T + 10)
    assert calls["n"] == 2


def test_a_part_sealed_while_a_range_read_runs(store):
    """#3100: sealing an open part under a concurrent range read still returns its rows."""
    record = one(S.FACE_MATCHES, ts_start=T, ts_end=T)
    writers = S.Writers(store)
    writers.append(S.FACE_MATCHES, "living-room", [record])
    writers.sync()
    rows: list[dict] = []
    err: list[BaseException] = []

    def reader() -> None:
        try:
            rows.extend(S.Log(store).range(S.FACE_MATCHES, T, T + 10))
        except BaseException as e:
            err.append(e)

    t = threading.Thread(target=reader)
    t.start()
    for _ in range(200):
        if not t.is_alive():
            break
        time.sleep(0.001)
    writers.close()
    t.join(timeout=10.0)
    assert not err
    assert t.is_alive() is False
    assert [r["id"] for r in rows] == [record["id"]]


@pytest.fixture
def folded(tmp_path, monkeypatch):
    """A group store where Face 5 was folded into Face 3 (vision/faces.py, #3181)."""
    from liberational.vision import faces

    monkeypatch.setattr(faces, "UNK_DB", tmp_path / "unknown_clusters.npz")
    faces._unk_save([3, 9], np.stack([unit_vector(0), unit_vector(1)]), [2, 1], 10, aliases={5: 3})


def group_of(stream, n):
    """(field, value): how a sighting of `stream` names unnamed group `n`."""
    from liberational.people import registry

    if stream == S.FACE_MATCHES:
        return "cluster", f"Face {n}"
    return "group", str(registry.subject_uuid(registry.face_group(n)))


def sighting(stream, n, at):
    field, value = group_of(stream, n)
    return one(stream, ts_start=at, ts_end=at) | {"name": None, field: value}


@pytest.mark.parametrize("stream", [S.FACE_MATCHES, S.FACE_EMBEDDINGS])
def test_a_sighting_of_a_folded_group_reads_as_the_group_it_went_into(store, folded, stream):
    write(store, [sighting(stream, 5, T + 1), sighting(stream, 9, T + 2)], stream)
    field, _ = group_of(stream, 0)
    assert [r[field] for r in S.Log(store).range(stream, T, T + 60)] == [group_of(stream, 3)[1], group_of(stream, 9)[1]]


def test_a_tail_reads_a_folded_groups_new_sightings_as_the_group_it_went_into(store, folded):
    writers = S.Writers(store)
    tail = S.Tail(store, S.FACE_MATCHES, since=T)
    writers.append(S.FACE_MATCHES, "cam0", [sighting(S.FACE_MATCHES, 5, T + 1)])
    writers.sync()
    assert [r["cluster"] for r in tail.poll(2.0)] == ["Face 3"]
    writers.close()


def test_without_a_fold_sightings_read_as_recorded(store, tmp_path, monkeypatch):
    from liberational.vision import faces

    monkeypatch.setattr(faces, "UNK_DB", tmp_path / "unknown_clusters.npz")  # no group store at all
    write(store, [sighting(S.FACE_MATCHES, 5, T + 1)], S.FACE_MATCHES)
    assert [r["cluster"] for r in S.Log(store).range(S.FACE_MATCHES, T, T + 60)] == ["Face 5"]
