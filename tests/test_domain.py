"""The domain types (docs/DOMAIN-MODEL.md, "Records" and "Events"): their invariants, the envelope's format, and the import rules."""

import ast
import dataclasses
import datetime
import json
import pathlib
import uuid

import pytest

from liberational.domain import identity, kernel, memory
from liberational.domain.kernel import (
    AssertionId,
    Event,
    EventType,
    EvidenceKind,
    EvidenceRef,
    PersonId,
    Producer,
    VoiceprintId,
)

T = 1_760_000_000.0  # an instant in 2025, in epoch seconds
DOMAIN = pathlib.Path(kernel.__file__).parent
NAN, INF = float("nan"), float("inf")
SUBJECT = identity.SubjectRef(identity.SubjectKind.VOICEPRINT, "v1")
MATCHER = Producer("ecapa-tdnn", "1.0")


def ref(kind=EvidenceKind.UTTERANCE, id="u1", **kw):
    return EvidenceRef(kind, id, **kw)


def utterance_event(**kw):
    return Event.new(
        EventType.UTTERANCE_HEARD, stream="mic:desk", source="ambient", payload={"text": "hi Dana"}, occurred_at=T, **kw
    )


def scene_event(payload):
    return Event.new("SceneObserved", stream="cam:front", source="vision", payload=payload)


def assertion(basis, confidence=1.0, asserted_at=T, source="dashboard", id="a1", **kw):
    return identity.IdentityAssertion(
        AssertionId(id), SUBJECT, PersonId("p1"), basis, confidence, asserted_at=asserted_at, source=source, **kw
    )


def test_seconds_pass_through_and_milliseconds_are_converted():
    assert kernel.instant(T) == T
    assert kernel.instant(int(T * 1000)) == T


def test_an_aware_datetime_converts():
    aware = datetime.datetime(2026, 10, 25, 12, tzinfo=datetime.timezone(datetime.timedelta(hours=-7)))
    assert kernel.instant(aware) == aware.timestamp()
    assert kernel.to_datetime(kernel.instant(aware)) == aware


def test_ids_are_uuid7_and_sort_by_creation():
    ids = [kernel.new_id() for _ in range(200)]
    assert ids == sorted(ids)
    assert len(set(ids)) == 200
    assert uuid.UUID(ids[0]).version == 7


def test_a_media_ref_takes_offsets_and_a_kind_string():
    assert ref(EvidenceKind.RECORDING, "r1", start=12.5).to_dict() == {"kind": "recording", "id": "r1", "start": 12.5}
    assert ref(EvidenceKind.CALL, "c1", start=1.0, end=4.0).end == 4.0
    assert EvidenceRef("scene", "o1").kind is EvidenceKind.SCENE  # ty: ignore[invalid-argument-type]  # converted


def test_new_fills_the_envelope(monkeypatch):
    monkeypatch.setattr(kernel, "now", lambda: T + 5)
    e = utterance_event()
    assert e.schema_version == kernel.SCHEMA_VERSIONS[EventType.UTTERANCE_HEARD]
    assert (e.occurred_at, e.recorded_at, e.subject) == (T, T + 5, None)


def test_the_envelope_survives_json():
    e = utterance_event(
        subject="p1",
        producer=Producer("parakeet-tdt", "0.6b-v2", prompt_id="p", prompt_hash="ab"),
        evidence=[ref(EvidenceKind.AUDIO, "desk/2026-10-25/1200.flac", start=3.0, end=4.5), ref()],
        correlation_id="chain-1",
    )
    d = json.loads(json.dumps(e.to_dict(), allow_nan=False))
    assert sorted(d) == sorted(f.name for f in dataclasses.fields(Event))
    assert Event.from_dict(d) == e


def test_the_event_is_immutable_and_detached_from_the_callers_values():
    payload = {"text": "hi"}
    evidence = [ref()]
    e = Event.new("AssistantSpoke", stream="voice", source="voice", payload=payload, evidence=evidence)
    payload["text"] = "changed"
    evidence.append(ref(id="u2"))
    assert e.payload["text"] == "hi"
    assert e.evidence == (ref(),)
    with pytest.raises(TypeError):
        e.payload["text"] = "x"  # ty: ignore[invalid-assignment]  # the payload is read-only, which is the point
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.stream = "other"  # ty: ignore[invalid-assignment]  # the event is frozen, which is the point


def test_nested_payload_values_are_frozen_too():
    face = {"faceprint": "f1"}
    e = scene_event({"faces": [face]})
    face["faceprint"] = "f2"
    assert e.payload["faces"][0]["faceprint"] == "f1"
    with pytest.raises(TypeError):
        e.payload["faces"][0]["faceprint"] = "f3"
    out = e.to_dict()
    out["payload"]["faces"][0]["faceprint"] = "f4"
    assert e.to_dict()["payload"] == {"faces": [{"faceprint": "f1"}]}


def test_the_catalog_matches_the_document():
    doc = (DOMAIN.parents[2] / "docs" / "DOMAIN-MODEL.md").read_text()
    catalog = doc.split("### Catalog", 1)[1].split("\n\n", 2)[1]
    names = {n.strip(" `") for row in catalog.splitlines()[2:] for n in row.split("|")[1].split(",")}
    assert names == {str(t) for t in EventType}


def test_records_freeze_the_lists_they_are_given():
    # Each list below becomes a tuple, which is the point, so ty is told to accept a list.
    assert identity.Person(PersonId("p1"), "Al", aliases=["Al"]).aliases == ("Al",)  # ty: ignore[invalid-argument-type]
    v = identity.Voiceprint(VoiceprintId("v1"), [0.3, 0.4], references=[[0.3, 0.4]])  # ty: ignore[invalid-argument-type]
    assert v.references == ((0.3, 0.4),)


def test_values_at_the_edge_of_each_rule_are_accepted():
    assert assertion("owner_tag", evidence=[ref()]).basis is identity.Basis.OWNER_TAG
    assert assertion("correction", id="a2", supersedes="a1", retracted=True).retracted
    assert assertion("owner_tag").producer is None
    assert assertion("owner_tag", producer=MATCHER).producer == MATCHER
    assert assertion("embedding_match", 0.8, producer=MATCHER).producer == MATCHER
    assert assertion("backfill_match", 0.8, producer=MATCHER).producer == MATCHER
    e = memory.Encounter(PersonId("p1"), T, T + 300, ["seen", "heard"])  # ty: ignore[invalid-argument-type]  # converted
    assert e.seen_or_heard == {memory.Modality.SEEN, memory.Modality.HEARD}


@pytest.mark.parametrize(
    "make, error",
    [
        pytest.param(lambda: kernel.instant(-1), ValueError, id="instant-negative"),
        pytest.param(lambda: kernel.instant(NAN), ValueError, id="instant-nan"),
        pytest.param(lambda: kernel.instant(INF), ValueError, id="instant-inf"),
        pytest.param(lambda: kernel.instant(True), TypeError, id="instant-bool"),
        pytest.param(lambda: kernel.instant("1760000000"), TypeError, id="instant-string"),  # ty: ignore[invalid-argument-type]
        pytest.param(lambda: kernel.instant(datetime.datetime(2026, 10, 25, 12)), ValueError, id="instant-naive"),
        pytest.param(lambda: ref(start=1.0), ValueError, id="ref-offset-on-a-record"),
        pytest.param(lambda: ref(EvidenceKind.AUDIO, start=-1.0), ValueError, id="ref-negative-start"),
        pytest.param(lambda: ref(EvidenceKind.AUDIO, end=2.0), ValueError, id="ref-end-without-start"),
        pytest.param(lambda: ref(EvidenceKind.AUDIO, start=3.0, end=2.0), ValueError, id="ref-end-before-start"),
        pytest.param(lambda: ref(EvidenceKind.AUDIO, start=0.0, end=INF), ValueError, id="ref-infinite-end"),
        pytest.param(lambda: EvidenceRef("photo", "p1"), ValueError, id="ref-unknown-kind"),  # ty: ignore[invalid-argument-type]
        pytest.param(lambda: scene_event({"seen": {"f1"}}), TypeError, id="payload-set"),
        pytest.param(lambda: scene_event({1: "x"}), ValueError, id="payload-int-key"),
        pytest.param(lambda: scene_event(["a", "list"]), TypeError, id="payload-not-a-mapping"),
        pytest.param(lambda: scene_event({"v": [NAN]}), ValueError, id="payload-nan"),
        pytest.param(lambda: scene_event({"v": [-INF]}), ValueError, id="payload-inf"),
        pytest.param(lambda: identity.Person(PersonId("p1"), " "), ValueError, id="person-blank-name"),
        pytest.param(lambda: identity.Person(PersonId("p1"), "Alice", role=""), ValueError, id="person-empty-role"),
        pytest.param(lambda: identity.Person(PersonId("p1"), "A", added="yes"), ValueError, id="person-added"),  # ty: ignore[invalid-argument-type]
        pytest.param(
            lambda: identity.Voiceprint(VoiceprintId("v1"), (0.1, 0.2), references=((0.1,),)), ValueError, id="refs"
        ),
        pytest.param(lambda: assertion("correction", retracted=True), ValueError, id="retraction-names-nothing"),
        pytest.param(lambda: assertion("correction", supersedes="a1"), ValueError, id="self-supersede"),
        pytest.param(lambda: assertion("guess", 0.5), ValueError, id="unknown-basis"),
        pytest.param(lambda: assertion("embedding_match", 1.2, producer=MATCHER), ValueError, id="confidence-over-1"),
        pytest.param(lambda: assertion("owner_tag", asserted_at=T * 1000), ValueError, id="asserted-in-ms"),
        pytest.param(lambda: assertion("owner_tag", source=" "), ValueError, id="blank-source"),
        pytest.param(lambda: assertion("embedding_match", 0.8), ValueError, id="embedding-match-without-producer"),
        pytest.param(lambda: assertion("backfill_match", 0.8), ValueError, id="backfill-match-without-producer"),
        pytest.param(lambda: assertion("embedding_match", 0.8, producer="ecapa"), ValueError, id="producer-a-string"),
        pytest.param(lambda: memory.Encounter(PersonId("p1"), T, T + 1, frozenset()), ValueError, id="encounter-empty"),
        pytest.param(lambda: Producer("qwen3", ""), ValueError, id="producer-without-version"),
    ],
)
def test_a_value_that_breaks_a_rule_is_refused(make, error):
    with pytest.raises(error):
        make()


@pytest.mark.parametrize(
    "change, match",
    [
        pytest.param({"type": "UtteranceHeared"}, "UtteranceHeared", id="unknown-type"),
        pytest.param({"schema_version": 2}, "newer", id="newer-version"),
        pytest.param({"schema_version": 0}, "at least 1", id="version-zero"),
    ],
)
def test_an_unknown_type_or_version_is_refused(change, match):
    with pytest.raises(ValueError, match=match):
        Event.from_dict({**utterance_event().to_dict(), **change})


# The domain does no I/O, and each context imports only the shared kernel (see the package docstring).
STDLIB = {"__future__", "collections", "dataclasses", "datetime", "enum", "math", "time", "types", "typing", "uuid"}


def imported(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            yield "." * node.level + (node.module or "")


@pytest.mark.parametrize("path", sorted(DOMAIN.glob("*.py")), ids=lambda p: p.name)
def test_the_domain_imports_only_the_kernel_and_pure_stdlib(path):
    bad = [m for m in imported(path) if m != "liberational.domain.kernel" and m.split(".")[0] not in STDLIB]
    assert bad == []
