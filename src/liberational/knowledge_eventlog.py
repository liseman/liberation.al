"""The domain event log on durable owner knowledge, independent of broker observations.

Event envelopes remain unchanged JSON. Indexed collection, subject, and occurrence-time fields are only lookup
metadata; replay positions and consumer cursors belong to this SQLite store, not to a broker session.
"""

import time
import uuid
from collections.abc import Iterable, Iterator, Mapping
from fractions import Fraction
from pathlib import Path

from liberational import knowledge
from liberational.domain import kernel
from liberational.domain.eventlog import EVERYTHING, Filter, Position, PurgeRecord, PurgeScope, Selector, Stored
from liberational.domain.identity import SubjectKind, SubjectRef
from liberational.domain.kernel import Event, EventType
from liberational.knowledge import MAX_TIME_NS, Record, StoredRecord


def _ns(instant: float) -> int:
    return round(Fraction(instant) * 1_000_000_000)


def _uuid(value: str) -> None:
    parsed = uuid.UUID(value)
    kernel.check(str(parsed) == value and parsed.version == 7, "event and subject ids must be canonical UUIDv7")


def subject_key(subject: SubjectRef) -> str:
    """An index key distinct from person UUIDs, preserving the complete domain subject identity."""
    return f"ref:{subject.kind}:{subject.id}"


def _subjects(event: Event) -> tuple[str, ...]:
    subjects: list[str] = [event.subject] if event.subject is not None else []
    assertion = event.payload.get("assertion")
    if event.type in (EventType.IDENTITY_ASSERTED, EventType.IDENTITY_RETRACTED) and isinstance(assertion, Mapping):
        ref = assertion.get("subject")
        if (
            isinstance(ref, Mapping)
            and ref.get("kind") in tuple(SubjectKind)
            and isinstance(ref.get("id"), str)
            and ref["id"].strip()
        ):
            subjects.append(subject_key(SubjectRef(SubjectKind(ref["kind"]), ref["id"])))
    if event.type is EventType.UTTERANCE_LABELED and isinstance(utterance := event.payload.get("utterance"), str):
        subjects.append(subject_key(SubjectRef(SubjectKind.OBSERVATION, utterance)))
    return tuple(subjects)


def _stored(item: StoredRecord) -> Stored:
    return Stored(item.position, Event.from_dict(item.record.value))


def record_of(event: Event) -> Record:
    """A validated domain envelope as independent storage, also used by the explicit pre-reset exporter."""
    if not isinstance(event, Event):
        raise TypeError(f"the log holds events, not {type(event).__name__}")
    _uuid(event.event_id)
    if event.subject is not None:
        _uuid(event.subject)
    return Record(
        id=event.event_id,
        collection=f"event:{event.stream}",
        key=str(event.type),
        start_ns=_ns(event.occurred_at),
        end_ns=_ns(event.occurred_at),
        recorded_ns=_ns(event.recorded_at),
        value=event.to_dict(),
        subjects=_subjects(event),
    )


class SQLiteEventLog:
    """An EventLog adapter whose owner assertions survive broker resets."""

    def __init__(self, path: Path) -> None:
        self.store = knowledge.at(path)

    def append(self, events: Iterable[Event]) -> list[Position]:
        return self.store.put(record_of(event) for event in events)

    def read(self, stream: str, after: Position = 0, limit: int | None = None) -> list[Stored]:
        kernel.check_text(stream, "stream")
        kernel.check(type(after) is int, "after must be an integer")
        if limit is not None:
            kernel.check(type(limit) is int and limit > 0, "limit must be a positive integer")
        return [_stored(item) for item in self.store.query(f"event:{stream}", after=max(after, 0), limit=limit)]

    def subscribe(self, consumer: str, filter: Filter = EVERYTHING) -> Iterator[Stored]:
        kernel.check_text(consumer, "consumer")
        kernel.check(isinstance(filter, Filter), "subscribe takes a Filter")
        after = self.store.cursor(consumer)
        head = self.store.head()
        # Materialize before returning: later appends must wait for the next subscription, even on another stream.
        found = [
            _stored(item)
            for collection in self.store.collections(prefix="event:")
            if not filter.streams or collection.removeprefix("event:") in filter.streams
            for item in self.store.query(collection, after=after, through=head)
        ]
        found.sort(key=lambda item: item.position)
        return iter(item for item in found if not filter.types or item.event.type in filter.types)

    def ack(self, consumer: str, position: Position) -> None:
        kernel.check_text(consumer, "consumer")
        kernel.check(type(position) is int and position >= 0, "a position must be a nonnegative integer")
        self.store.ack(consumer, position)

    def purge(self, selector: Selector, reason: str) -> int:
        kernel.check(isinstance(selector, Selector), "purge takes a Selector")
        kernel.check_text(reason, "reason")
        collections = [f"event:{selector.stream}"] if selector.stream is not None else []
        before_ns = _ns(selector.before) + 1 if selector.before is not None else None
        upper = before_ns if before_ns is not None and before_ns <= MAX_TIME_NS else None
        now = time.time_ns()
        audit = Record(
            id=str(uuid.uuid7()),
            collection="audit:eventlog",
            key=None,
            start_ns=now,
            end_ns=now,
            recorded_ns=now,
            value={
                "at": now / 1_000_000_000,
                "scope": PurgeScope.of(selector).to_dict(),
                "reason": reason,
                "removed": 0,
            },
        )
        return self.store.remove_matching(
            collections,
            collection_prefix="event:" if selector.stream is None else None,
            subjects=selector.subjects,
            ids=selector.event_ids,
            before_ns=upper,
            predicate=(lambda record: record.value["occurred_at"] < selector.before)
            if selector.before is not None
            else None,
            audit=audit,
        )

    def purges(self) -> list[PurgeRecord]:
        return [
            PurgeRecord(
                item.record.value["at"],
                PurgeScope(**item.record.value["scope"]),
                item.record.value["reason"],
                item.record.value["removed"],
            )
            for item in self.store.query("audit:eventlog")
        ]

    def version(self) -> int:
        """A cross-process change revision, including deletions that leave no events."""
        return self.store.revision()

    def identity_version(self) -> tuple[str, int, int]:
        """Only identity projection inputs, including erasures and database replacement, invalidate its readers."""
        return self.store.generation(("event:people", "event:identity", "event:identity.sightings"))

    def close(self) -> None:
        self.store.close()
