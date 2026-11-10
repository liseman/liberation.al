"""The shared kernel: ids, the instant, evidence references, producers, and the event envelope and catalog.

Every bounded context imports this module and no other context's module. See "Contexts" and "Events" in
docs/DOMAIN-MODEL.md.

Ids are UUIDv7 strings, so they sort by creation time. Person ids are stored in the event log,
so changing the format now means a migration.
"""

import dataclasses
import datetime
import enum
import math
import time
import types
import uuid
from collections.abc import Callable, Iterable, Mapping
from typing import Any, NewType

type Instant = float
"""A UTC time as epoch seconds (rule 5). Convert other clocks at the edge with `instant()`."""

PersonId = NewType("PersonId", str)
VoiceprintId = NewType("VoiceprintId", str)
AssertionId = NewType("AssertionId", str)
EventId = NewType("EventId", str)

# Epoch seconds reach 1e12 in the year 33658, so a larger value is milliseconds. Readers guess this today.
_MILLISECONDS_ABOVE = 1e12


def new_id() -> str:
    """A new id that sorts after every id made before it in this process."""
    return str(uuid.uuid7())


def now() -> Instant:
    return time.time()


def instant(value: float | datetime.datetime) -> Instant:
    """Convert a value from the edge to an instant: epoch seconds or milliseconds, or a datetime with a zone.

    A naive datetime is refused, because its zone is a guess. Attach the zone where the value is read.
    """
    if isinstance(value, datetime.datetime):
        if value.utcoffset() is None:
            raise ValueError(f"naive datetime {value!r}: attach its zone where it is read")
        return value.timestamp()
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"an instant is a number or a datetime, not {type(value).__name__}")
    t = float(value)
    if not math.isfinite(t) or t < 0:
        raise ValueError(f"not an instant: {value!r}")
    return t / 1000 if t > _MILLISECONDS_ABOVE else t


def to_datetime(t: Instant) -> datetime.datetime:
    """The instant as an aware UTC datetime, for display."""
    return datetime.datetime.fromtimestamp(t, datetime.UTC)


# Checks the context modules share. Each raises ValueError (or TypeError for a wrong type) naming the field.


def check(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


# The checks below build their message only when they fail: a check runs millions of times in one presence read, and
# formatting a message (a repr) every time was most of its cost.


def check_text(value: object, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, not {type(value).__name__}")
    if not value.strip():
        raise ValueError(f"{name} is empty")


def check_optional_text(value: object, name: str) -> None:
    if value is not None:
        check_text(value, name)


def check_instant(t: object, name: str) -> None:
    """A record field holds epoch seconds already; conversion belongs at the edge, not here."""
    if isinstance(t, bool) or not isinstance(t, int | float):
        raise TypeError(f"{name} must be epoch seconds, not {type(t).__name__}")
    if not (math.isfinite(t) and t >= 0):
        raise ValueError(f"{name} is not an instant: {t!r}")
    if t > _MILLISECONDS_ABOVE:
        raise ValueError(f"{name} looks like milliseconds ({t!r}); convert it with instant()")


def check_span(start: object, end: object) -> None:
    check_instant(start, "start")
    check_instant(end, "end")
    if end < start:  # ty: ignore[unsupported-operator]  # check_instant refused anything but a number
        raise ValueError(f"end {end!r} is before start {start!r}")


def check_unit(x: object, name: str, low: float = 0.0) -> None:
    """`x` is a number from `low` to 1."""
    if isinstance(x, bool) or not isinstance(x, int | float):
        raise TypeError(f"{name} must be a number, not {type(x).__name__}")
    check(low <= x <= 1, f"{name} must be from {low} to 1, not {x!r}")


def vector(values: Iterable[float]) -> tuple[float, ...]:
    """An embedding as a tuple of finite floats."""
    v = tuple(float(x) for x in values)
    check(len(v) > 0, "an embedding is empty")
    check(all(math.isfinite(x) for x in v), "an embedding has a value that is not finite")
    return v


def vectors(values: Iterable[Iterable[float]]) -> tuple[tuple[float, ...], ...]:
    """Several embeddings of one size."""
    vs = tuple(vector(v) for v in values)
    check(len({len(v) for v in vs}) <= 1, "embeddings differ in size")
    return vs


def coerce(record: object, name: str, convert: Callable[[Any], Any]) -> None:
    """Replace a field of a frozen record with `convert(field)` while it is built.

    Sequences become tuples this way, so a caller cannot change a record through the list it passed in.
    """
    object.__setattr__(record, name, convert(getattr(record, name)))


def freeze(value: Any) -> Any:
    """A deep, read-only copy of JSON-like data: mappings become read-only mappings and lists become tuples.

    Anything that JSON cannot hold is refused, so a frozen value always serializes.
    """
    if isinstance(value, Mapping):
        check(all(isinstance(k, str) for k in value), "payload keys must be strings")
        return types.MappingProxyType({k: freeze(v) for k, v in value.items()})
    if isinstance(value, list | tuple):
        return tuple(freeze(v) for v in value)
    if isinstance(value, float):
        check(math.isfinite(value), f"a payload number must be finite, not {value!r}")
        return value
    if value is None or isinstance(value, str | int):
        return value
    raise TypeError(f"a payload holds JSON-like values, not {type(value).__name__}")


def thaw(value: Any) -> Any:
    """A frozen value as new plain dicts and lists, for JSON. Changing the result does not change the record."""
    if isinstance(value, Mapping):
        return {k: thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [thaw(v) for v in value]
    return value


class EvidenceKind(enum.StrEnum):
    UTTERANCE = "utterance"
    SCENE = "scene"
    FACE = "face"
    RECORDING = "recording"
    CALL = "call"
    AUDIO = "audio"
    EVENT = "event"
    INTERPRETATION = "interpretation"


MEDIA_KINDS = frozenset({EvidenceKind.RECORDING, EvidenceKind.CALL, EvidenceKind.AUDIO})


@dataclasses.dataclass(frozen=True, slots=True)
class EvidenceRef:
    """A stable pointer to the source of a claim.

    For media (a recording, a call, an audio file) `id` names the media and `start` and `end` are seconds from its
    beginning, so a frame is a recording plus `start`. For every other kind `id` names the record alone.
    """

    kind: EvidenceKind
    id: str
    start: float | None = None
    end: float | None = None

    def __post_init__(self) -> None:
        coerce(self, "kind", EvidenceKind)
        check_text(self.id, "evidence id")
        if self.kind not in MEDIA_KINDS:
            check(self.start is None and self.end is None, f"{self.kind} evidence has no offsets")
            return
        if self.start is not None:
            check(math.isfinite(self.start) and self.start >= 0, f"evidence start is negative: {self.start!r}")
        if self.end is not None:
            check(self.start is not None, "evidence has an end but no start")
            check(math.isfinite(self.end), f"evidence end is not finite: {self.end!r}")  # json would refuse it later
            check(
                self.end >= self.start,  # ty: ignore[unsupported-operator]  # the check above refused a None start
                f"evidence end {self.end!r} is before start {self.start!r}",
            )

    def to_dict(self) -> dict:
        d = {"kind": str(self.kind), "id": self.id}
        if self.start is not None:
            d["start"] = self.start
        if self.end is not None:
            d["end"] = self.end
        return d

    @classmethod
    def from_dict(cls, d: Mapping) -> EvidenceRef:
        return cls(d["kind"], d["id"], d.get("start"), d.get("end"))


@dataclasses.dataclass(frozen=True, slots=True)
class Producer:
    """The model behind a record (rules 4 and 13): its name and version, and the prompt if there was one."""

    model: str
    version: str
    prompt_id: str | None = None
    prompt_hash: str | None = None

    def __post_init__(self) -> None:
        check_text(self.model, "producer model")
        check_text(self.version, "producer version")
        check_optional_text(self.prompt_id, "producer prompt_id")
        check_optional_text(self.prompt_hash, "producer prompt_hash")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping) -> Producer:
        return cls(d["model"], d["version"], d.get("prompt_id"), d.get("prompt_hash"))


class EventType(enum.StrEnum):
    """The event catalog in docs/DOMAIN-MODEL.md. Names are past tense."""

    UTTERANCE_HEARD = "UtteranceHeard"
    SCENE_OBSERVED = "SceneObserved"
    FACE_DETECTED = "FaceDetected"
    RECORDING_SEGMENT_CLOSED = "RecordingSegmentClosed"
    INTRODUCTION_HEARD = "IntroductionHeard"
    PERSON_REGISTERED = "PersonRegistered"
    PERSON_ATTRIBUTES_CHANGED = "PersonAttributesChanged"
    IDENTITY_ASSERTED = "IdentityAsserted"
    IDENTITY_RETRACTED = "IdentityRetracted"
    UTTERANCE_LABELED = "UtteranceLabeled"
    PERSONS_MERGED = "PersonsMerged"
    PERSON_FORGOTTEN = "PersonForgotten"
    ACTIVITY_DETECTED = "ActivityDetected"
    MEASUREMENT_RECORDED = "MeasurementRecorded"
    ASSISTANT_SPOKE = "AssistantSpoke"
    APPROVAL_RECORDED = "ApprovalRecorded"
    CALL_ENDED = "CallEnded"
    SUMMARY_WRITTEN = "SummaryWritten"
    INTERPRETATION_RECORDED = "InterpretationRecorded"
    INTERPRETATION_SUPERSEDED = "InterpretationSuperseded"
    INTERPRETATION_PROMOTED = "InterpretationPromoted"
    CARD_PREFERENCE_SET = "CardPreferenceSet"
    INBOX_SKIPPED = "InboxSkipped"
    WHISPER_ANNOUNCED = "WhisperAnnounced"
    OBJECT_SIGHTED = "ObjectSighted"
    OBJECT_REMEMBERED = "ObjectRemembered"
    OBJECT_FORGOTTEN = "ObjectForgotten"


# The newest schema version of each event type. Bump one when its payload changes shape.
SCHEMA_VERSIONS: Mapping[EventType, int] = types.MappingProxyType({t: 1 for t in EventType})


@dataclasses.dataclass(frozen=True, slots=True)
class Event:
    """The envelope every event shares. The payload differs by type.

    `source` is the device or process that wrote the event, and `producer` is the model behind it, if any.
    """

    event_id: EventId
    type: EventType
    schema_version: int
    occurred_at: Instant
    recorded_at: Instant
    stream: str
    source: str
    payload: Mapping[str, Any]
    producer: Producer | None = None
    subject: PersonId | None = None
    causation_id: EventId | None = None
    correlation_id: str | None = None
    evidence: tuple[EvidenceRef, ...] = ()

    def __post_init__(self) -> None:
        check_text(self.event_id, "event_id")
        coerce(self, "type", EventType)
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int):
            raise TypeError(f"schema_version must be an integer, not {type(self.schema_version).__name__}")
        newest = SCHEMA_VERSIONS[self.type]
        check(self.schema_version >= 1, f"schema_version must be at least 1, not {self.schema_version}")
        check(
            self.schema_version <= newest,
            f"{self.type} version {self.schema_version} is newer than this code knows ({newest})",
        )
        check_instant(self.occurred_at, "occurred_at")
        check_instant(self.recorded_at, "recorded_at")
        check_text(self.stream, "stream")
        check_text(self.source, "source")
        check_optional_text(self.subject, "subject")
        check_optional_text(self.causation_id, "causation_id")
        check_optional_text(self.correlation_id, "correlation_id")
        if not isinstance(self.payload, Mapping):
            raise TypeError(f"payload must be a mapping, not {type(self.payload).__name__}")
        coerce(self, "payload", freeze)
        coerce(self, "evidence", tuple)
        check(all(isinstance(e, EvidenceRef) for e in self.evidence), "evidence must be EvidenceRef values")
        check(self.producer is None or isinstance(self.producer, Producer), "producer must be a Producer")

    @classmethod
    def new(
        cls,
        type: EventType | str,
        *,
        stream: str,
        source: str,
        payload: Mapping[str, Any],
        occurred_at: Instant | None = None,
        **optional: Any,
    ) -> Event:
        """A new event at the newest schema version for its type, recorded now."""
        recorded = now()
        type = EventType(type)
        return cls(
            event_id=EventId(new_id()),
            type=type,
            schema_version=SCHEMA_VERSIONS[type],
            occurred_at=recorded if occurred_at is None else occurred_at,
            recorded_at=recorded,
            stream=stream,
            source=source,
            payload=payload,
            **optional,
        )

    def to_dict(self) -> dict:
        """The envelope as plain JSON-ready values. Every field is present, with None where it is unset."""
        return {
            "event_id": self.event_id,
            "type": str(self.type),
            "schema_version": self.schema_version,
            "occurred_at": self.occurred_at,
            "recorded_at": self.recorded_at,
            "stream": self.stream,
            "source": self.source,
            "producer": self.producer.to_dict() if self.producer else None,
            "subject": self.subject,
            "causation_id": self.causation_id,
            "correlation_id": self.correlation_id,
            "evidence": [e.to_dict() for e in self.evidence],
            "payload": thaw(self.payload),
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> Event:
        """Read an envelope. Refuses an unknown type, and a version newer than this code knows."""
        producer = d.get("producer")
        return cls(
            event_id=d["event_id"],
            type=d["type"],
            schema_version=d["schema_version"],
            occurred_at=d["occurred_at"],
            recorded_at=d["recorded_at"],
            stream=d["stream"],
            source=d["source"],
            payload=d["payload"],
            producer=Producer.from_dict(producer) if producer else None,
            subject=d.get("subject"),
            causation_id=d.get("causation_id"),
            correlation_id=d.get("correlation_id"),
            evidence=tuple(EvidenceRef.from_dict(e) for e in d.get("evidence", ())),
        )
