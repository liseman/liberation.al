"""The event log port: the small interface every context depends on instead of a product (docs/DOMAIN-MODEL.md).

An adapter keeps these guarantees, and no more:

- At least once. A consumer can see an event again, and ignores it by `event_id`.
- Order within one stream only. Positions rise within a stream; there is no promise across streams.
- Replay. A new consumer starts from the beginning of every stream.
- Events are small. Media stays in files and is named by evidence references.

A position is an adapter's own number. Only compare positions from the same log, and never reuse one: after a purge,
new events still get positions after every position handed out before.

`tests/test_eventlog.py` holds the contract test that every adapter must pass.
"""

import dataclasses
from collections.abc import Iterable, Iterator
from typing import Any, Protocol

from liberational.domain.kernel import (
    Event,
    EventId,
    EventType,
    Instant,
    PersonId,
    check,
    check_instant,
    check_optional_text,
    check_text,
    coerce,
)

type Position = int


@dataclasses.dataclass(frozen=True, slots=True)
class Stored:
    """An event and the position the log gave it."""

    position: Position
    event: Event


@dataclasses.dataclass(frozen=True, slots=True)
class Filter:
    """Which events a consumer wants. An empty set means any."""

    types: frozenset[EventType] = frozenset()
    streams: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        coerce(self, "types", lambda ts: frozenset(EventType(t) for t in ts))
        coerce(self, "streams", frozenset)
        for s in self.streams:
            check_text(s, "stream")


EVERYTHING = Filter()


@dataclasses.dataclass(frozen=True, slots=True)
class Selector:
    """Which events a purge removes: those that match every field that is set. At least one must be set.

    `subjects` matches events whose envelope names any of them: a person, and the ids merged into them, forgotten in
    one purge with one record (#725). Finding the rest of a person's records (rule 12) is the job of whatever applies
    `PersonForgotten`, not of the log.
    """

    subjects: frozenset[PersonId] = frozenset()
    stream: str | None = None
    event_ids: frozenset[EventId] = frozenset()
    before: Instant | None = None

    def __post_init__(self) -> None:
        coerce(self, "subjects", frozenset)
        for s in self.subjects:
            check_text(s, "subject")
        check_optional_text(self.stream, "stream")
        coerce(self, "event_ids", frozenset)
        for i in self.event_ids:
            check_text(i, "event id")
        if self.before is not None:
            check_instant(self.before, "before")
        check(
            any((self.subjects, self.stream, self.event_ids, self.before is not None)),
            "a purge selector needs at least one field, so nothing purges the whole log by accident",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "subjects": sorted(self.subjects),
            "stream": self.stream,
            "event_ids": sorted(self.event_ids),
            "before": self.before,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class PurgeScope:
    """What a purge selected, without naming anyone: whether it was by person (not which, nor how many ids stood for
    them), the stream, the time bound, and how many event ids. A log keeps purge records for good, so they never hold a person's id."""

    subject: bool = False
    stream: str | None = None
    before: Instant | None = None
    event_count: int = 0

    @classmethod
    def of(cls, selector: Selector) -> PurgeScope:
        return cls(bool(selector.subjects), selector.stream, selector.before, len(selector.event_ids))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class PurgeRecord:
    """What a purge removed, kept by the log so forgetting can be shown to have happened (rule 7), without saying who
    was forgotten (`scope`)."""

    at: Instant
    scope: PurgeScope
    reason: str
    removed: int


class EventLog(Protocol):
    def append(self, events: Iterable[Event]) -> list[Position]:
        """Write events durably, in order, and return their positions.

        An event whose `event_id` is already in the log is not written again; its existing position is returned.
        """
        ...

    def read(self, stream: str, after: Position = 0, limit: int | None = None) -> list[Stored]:
        """The events of one stream after a position, in order."""
        ...

    def subscribe(self, consumer: str, filter: Filter = EVERYTHING) -> Iterator[Stored]:
        """The events a consumer has not acknowledged yet, that match `filter`, in order within each stream.

        Delivery starts after the consumer's last `ack`, so anything it did not acknowledge comes again.
        """
        ...

    def ack(self, consumer: str, position: Position) -> None:
        """Record that a consumer has handled everything up to `position`. A consumer's position never moves back."""
        ...

    def purge(self, selector: Selector, reason: str) -> int:
        """Remove every event that matches, completely, and record the purge. Returns how many were removed."""
        ...

    def purges(self) -> list[PurgeRecord]:
        """Every purge so far, oldest first. A record never names the person a purge was about. An adapter may leave
        no record for a purge that removed nothing."""
        ...
