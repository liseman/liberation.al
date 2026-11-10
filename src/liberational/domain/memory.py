"""Memory: encounters, a layer 2 projection of when a person was seen, heard, or both."""

import dataclasses
import enum

from liberational.domain.kernel import (
    Instant,
    PersonId,
    check,
    check_span,
    check_text,
    coerce,
)


class Modality(enum.StrEnum):
    SEEN = "seen"
    HEARD = "heard"


@dataclasses.dataclass(frozen=True, slots=True)
class Encounter:
    """A contiguous period in which a person was seen or heard. Derived, never entered by hand."""

    person: PersonId
    start: Instant
    end: Instant
    seen_or_heard: frozenset[Modality]

    def __post_init__(self) -> None:
        check_text(self.person, "person")
        check_span(self.start, self.end)
        coerce(self, "seen_or_heard", lambda ms: frozenset(Modality(m) for m in ms))
        check(len(self.seen_or_heard) > 0, "an encounter is seen, heard, or both")
