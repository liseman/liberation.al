"""encode_header and decode_header: the schema's PayloadHeader between Python dicts and FlatBuffer bytes.

Run with `mise run continuum-test` (or pytest in continuum/.venv once continuum_py is built into it).
"""

import uuid
from typing import Any, get_args

import pytest
from continuum_py import DomainLayer, PayloadFormat, PayloadHeader, StorageTier, decode_header, encode_header
from hypothesis import example, given, settings
from hypothesis import strategies as st

U64_MAX = 2**64 - 1
U64 = st.integers(min_value=0, max_value=U64_MAX)
U32 = st.integers(min_value=0, max_value=2**32 - 1)
# Payload and entity ids are UUIDv7 (decision 8): random bits with the version and variant set.
ids = st.binary(min_size=16, max_size=16).map(
    lambda b: str(uuid.UUID(bytes=bytes([*b[:6], 0x70 | b[6] & 0x0F, b[7], 0x80 | b[8] & 0x3F, *b[9:]])))
)
text = st.text(max_size=200)


@st.composite
def spans(draw: st.DrawFn) -> tuple[int, int]:
    start = draw(U64)
    return start, draw(st.integers(min_value=start, max_value=2**64 - 1))


producers = st.fixed_dictionaries(
    {
        "producer_id": text,
        "model": text,
        "version": text,
        "version_hash": text,
        "prompt_id": st.none() | text,
        "prompt_hash": st.none() | text,
    }
)
evidence = st.fixed_dictionaries({"payload_id": ids, "start_ns": st.none() | U64, "end_ns": st.none() | U64})
computes = st.fixed_dictionaries(
    {"wall_time_us": U64, "queue_wait_us": U64, "ram_bytes": U64, "vram_bytes": U64, "device": text, "items": U64}
)
captures = st.fixed_dictionaries(
    {
        "capture_mono_ns": U64,
        "device_ts": U64,
        "arrival_ns": U64,
        "stride": U32,
        "width": U32,
        "height": U32,
        "flip": st.booleans(),
        "device_seq": U64,
        "sample_rate_hz": U32,
        "capture_epoch": U32,
    }
)


@st.composite
def headers(draw: st.DrawFn) -> dict[str, Any]:
    """A header with every key, as decode_header returns it."""
    start, end = draw(spans())
    return {
        "id": draw(ids),
        "stream_id": draw(U32),
        "start_ns": start,
        "end_ns": end,
        "seq": draw(U32),
        "layer": draw(st.sampled_from(get_args(DomainLayer))),
        "format": draw(st.sampled_from(get_args(PayloadFormat))),
        "shape_name": draw(text),
        "shape_version": draw(U32),
        "producer": draw(st.none() | producers),
        "parents": draw(st.lists(ids, max_size=40)),
        "evidence": draw(st.lists(evidence, max_size=20)),
        "source_tier": draw(st.sampled_from(get_args(StorageTier))),
        "compute": draw(st.none() | computes),
        # The schema's confidence is an f32: finite f32 values survive the trip, and others are refused.
        "confidence": draw(st.none() | st.floats(width=32, allow_nan=False, allow_infinity=False)),
        "subjects": draw(st.lists(ids, max_size=40)),
        "capture": draw(st.none() | captures),
    }


def edge(**changes: Any) -> dict[str, Any]:
    """A full header with the given fields changed, for @example."""
    return {
        "id": "01890a5d-ac96-774b-bcce-b302099a8057",
        "stream_id": 0,
        "start_ns": 0,
        "end_ns": 0,
        "seq": 0,
        "layer": "Layer1Raw",
        "format": "FlatBuffer",
        "shape_name": "",
        "shape_version": 0,
        "producer": None,
        "parents": [],
        "evidence": [],
        "source_tier": "Tier0Hot",
        "compute": None,
        "confidence": None,
        "subjects": [],
        "capture": None,
        **changes,
    }


@settings(deadline=None)
@given(headers())
@example(edge(start_ns=1_761_700_000_000_000_000, end_ns=1_761_700_000_000_000_000))  # an instant
@example(edge(start_ns=U64_MAX, end_ns=U64_MAX, stream_id=2**32 - 1, seq=2**32 - 1))  # the largest values
@example(edge(shape_name="é" * 5000))  # long, non-ASCII
def test_a_header_survives_encoding_and_decoding(header: dict[str, Any]) -> None:
    assert decode_header(encode_header(header)) == header  # ty: ignore[invalid-argument-type]


def minimal() -> PayloadHeader:
    return {
        "id": "01890a5d-ac96-774b-bcce-b302099a8057",
        "stream_id": 3,
        "start_ns": 1_761_700_000_000_000_000,
        "end_ns": 1_761_700_000_016_666_667,
        "layer": "Layer1Raw",
        "format": "RawBytes",
        "shape_name": "NV12",
    }


def test_omitted_keys_take_the_schemas_defaults() -> None:
    decoded = decode_header(encode_header(minimal()))
    assert decoded["seq"] == 0
    assert decoded["source_tier"] == "Tier0Hot"
    assert decoded["parents"] == [] and decoded["subjects"] == [] and decoded["evidence"] == []
    assert decoded["producer"] is None and decoded["compute"] is None and decoded["capture"] is None
    assert decoded["confidence"] is None, "an unset confidence stays absent, not 0.0"


def test_ids_come_back_as_canonical_uuidv7_strings() -> None:
    decoded = decode_header(encode_header(minimal()))
    assert uuid.UUID(decoded["id"]).version == 7
    assert decoded["id"] == minimal()["id"]


@pytest.mark.parametrize(
    ("change", "error", "match"),
    [
        pytest.param({"id": "01890A5D-AC96-774B-BCCE-B302099A8057"}, ValueError, "canonical", id="uppercase-id"),
        pytest.param({"id": "nope"}, ValueError, "canonical", id="not-a-uuid"),
        pytest.param({"end_ns": 1}, ValueError, "before start_ns", id="end-before-start"),
        pytest.param({"layer": "Layer9"}, ValueError, "layer must be one of", id="unknown-layer"),
        pytest.param({"format": "jpeg"}, ValueError, "format must be one of", id="unknown-format"),
        pytest.param({"stream_id": -1}, ValueError, "stream_id: out of range", id="negative-stream"),
        pytest.param({"stream_id": 2**32}, ValueError, "stream_id: out of range", id="stream-past-u32"),
        pytest.param({"start_ns": 2**64}, ValueError, "start_ns: out of range", id="start-past-u64"),
        pytest.param({"stream_id": "3"}, TypeError, "stream_id", id="stream-as-text"),
        pytest.param({"parents": ["x"]}, ValueError, "canonical", id="bad-parent"),
        pytest.param({"id": "b6d1f3a2-5c4e-4d6f-8a1b-2c3d4e5f6a7b"}, ValueError, "UUIDv7", id="v4-id"),
        pytest.param({"subjects": ["b6d1f3a2-5c4e-0d6f-8a1b-2c3d4e5f6a7b"]}, ValueError, "RFC 9562", id="v0-subject"),
        pytest.param({"subjects": ["b6d1f3a2-5c4e-5d6f-0a1b-2c3d4e5f6a7b"]}, ValueError, "RFC 9562", id="ncs-subject"),
        pytest.param({"confidence": float("nan")}, ValueError, "finite f32", id="nan-confidence"),
        pytest.param({"confidence": 1e300}, ValueError, "finite f32", id="confidence-past-f32"),
    ],
)
def test_a_bad_header_is_refused(change: dict[str, Any], error: type[Exception], match: str) -> None:
    with pytest.raises(error, match=match):
        encode_header({**minimal(), **change})


def test_a_subject_may_be_any_rfc_uuid_while_a_payload_id_must_be_a_v7() -> None:
    # A voiceprint's subject is a v5 named from its ref (people.registry.subject_uuid); a person's is their v7 id.
    v5 = str(uuid.uuid5(uuid.NAMESPACE_URL, "voiceprint:speaker:7"))
    person = "01890a5d-ac96-774b-bcce-b302099a8057"
    assert decode_header(encode_header({**minimal(), "subjects": [person, v5]}))["subjects"] == [person, v5]
    with pytest.raises(ValueError, match="UUIDv7"):
        encode_header({**minimal(), "id": v5})


def test_a_missing_required_key_is_a_key_error() -> None:
    header = dict(minimal())
    del header["shape_name"]
    with pytest.raises(KeyError, match="shape_name"):
        encode_header(header)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("wrap", [bytes, bytearray, memoryview], ids=["bytes", "bytearray", "memoryview"])
def test_any_bytes_like_object_decodes(wrap: type) -> None:
    assert decode_header(wrap(encode_header(minimal())))["id"] == minimal()["id"]


def test_bytes_that_arent_a_header_are_refused() -> None:
    with pytest.raises(ValueError):
        decode_header(b"not a flatbuffer")
