"""Identity: people, their voiceprints and faceprints, and the assertions that link observations to them.

This is the only context that decides who someone is (rule 2). A person is referenced by id everywhere (rule 3), and
a voiceprint or faceprint belongs to a person only through assertions, so a wrong match can be undone without losing
the print. A correction supersedes an assertion; it never deletes one (rule 6).

An assertion carries its own time and source (rule 4). People and prints are projections rebuilt from events (rule
8), so they carry only an id, and the events that build them carry the rest.

`decide` is the one written rule for choosing between assertions about one subject (rule 2, open decision 2).
"""

import dataclasses
import enum
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from liberational.domain.kernel import (
    AssertionId,
    EvidenceRef,
    Instant,
    PersonId,
    Producer,
    VoiceprintId,
    check,
    check_instant,
    check_optional_text,
    check_text,
    check_unit,
    coerce,
    vector,
    vectors,
)


@dataclasses.dataclass(frozen=True, slots=True)
class Person:
    """A human the pack knows about. The name is an attribute, so a rename changes one record.

    A projection of events such as `PersonRegistered` and `PersonAttributesChanged`. It carries only an id: the
    instant, source, producer, and schema version are on those events (rule 4).

    `aliases` are the names the person had before a rename, and the names of people merged into them. `added` means
    the owner added them by hand (People: Add), so they are listed even before anything has seen or heard them.
    """

    id: PersonId
    display_name: str
    aliases: tuple[str, ...] = ()
    role: str | None = None
    organization: str | None = None
    phone: str | None = None
    added: bool = False

    def __post_init__(self) -> None:
        check_text(self.id, "person id")
        check_text(self.display_name, "display_name")
        coerce(self, "aliases", tuple)
        for alias in self.aliases:
            check_text(alias, "alias")
        check_optional_text(self.role, "role")
        check_optional_text(self.organization, "organization")
        check_optional_text(self.phone, "phone")
        check(isinstance(self.added, bool), "added must be True or False")


@dataclasses.dataclass(frozen=True, slots=True)
class Voiceprint:
    """A speaker cluster: the centroid and reference embeddings of one voice.

    A projection rebuilt from events, so it carries only an id. The events carry the instant, source, producer, and
    schema version (rule 4).
    """

    id: VoiceprintId
    centroid: tuple[float, ...]
    references: tuple[tuple[float, ...], ...] = ()

    def __post_init__(self) -> None:
        check_text(self.id, "voiceprint id")
        coerce(self, "centroid", vector)
        coerce(self, "references", vectors)
        check(all(len(r) == len(self.centroid) for r in self.references), "references differ in size from centroid")


class SubjectKind(enum.StrEnum):
    VOICEPRINT = "voiceprint"
    FACEPRINT = "faceprint"
    OBSERVATION = "observation"


@dataclasses.dataclass(frozen=True, slots=True)
class SubjectRef:
    """What an assertion is about: a voiceprint, a faceprint, or one observation."""

    kind: SubjectKind
    id: str

    def __post_init__(self) -> None:
        coerce(self, "kind", SubjectKind)
        check_text(self.id, "subject id")


class Basis(enum.StrEnum):
    """Why an assertion was made."""

    OWNER_TAG = "owner_tag"
    INTRODUCTION = "introduction"
    NAME_BADGE = "name_badge"
    EMBEDDING_MATCH = "embedding_match"
    CLUSTER_MERGE = "cluster_merge"
    BACKFILL_MATCH = "backfill_match"
    CORRECTION = "correction"


# The bases where a model decides, so the assertion must name that model (rule 4).
MODEL_BASES = frozenset({Basis.EMBEDDING_MATCH, Basis.BACKFILL_MATCH})


@dataclasses.dataclass(frozen=True, slots=True)
class IdentityAssertion:
    """A claim that a subject belongs to a person, with its basis, confidence, and evidence.

    `asserted_at` is when it was made and `source` is the device or process that made it, such as "dashboard" or
    "ambient". `producer` is the model behind it. It is required when a model decided (an embedding match or a
    backfill match), and optional otherwise (rule 4). These are on the record, not only on its event, because "newer
    wins" breaks ties (open decision 2) and an audit of when and by what someone was identified should not need the
    envelope.

    A later assertion replaces this one by naming it in `supersedes`. A retraction is an assertion with `retracted`
    set that supersedes the one it withdraws.
    """

    id: AssertionId
    subject: SubjectRef
    person: PersonId
    basis: Basis
    confidence: float
    asserted_at: Instant
    source: str
    producer: Producer | None = None
    evidence: tuple[EvidenceRef, ...] = ()
    supersedes: AssertionId | None = None
    retracted: bool = False

    def __post_init__(self) -> None:
        check_text(self.id, "assertion id")
        check(isinstance(self.subject, SubjectRef), "subject must be a SubjectRef")
        check_text(self.person, "person")
        coerce(self, "basis", Basis)
        check_unit(self.confidence, "confidence")
        check_instant(self.asserted_at, "asserted_at")
        check_text(self.source, "assertion source")
        check(self.producer is None or isinstance(self.producer, Producer), "producer must be a Producer")
        check(
            self.producer is not None or self.basis not in MODEL_BASES, f"a {self.basis} assertion names its producer"
        )
        coerce(self, "evidence", tuple)
        check(all(isinstance(e, EvidenceRef) for e in self.evidence), "evidence must be EvidenceRef values")
        check_optional_text(self.supersedes, "supersedes")
        check(self.supersedes != self.id, "an assertion cannot supersede itself")
        check(not self.retracted or self.supersedes is not None, "a retraction names the assertion it withdraws")

    def to_dict(self) -> dict[str, Any]:
        """The assertion as plain JSON-ready values, for an event payload."""
        return {
            "id": self.id,
            "subject": {"kind": str(self.subject.kind), "id": self.subject.id},
            "person": self.person,
            "basis": str(self.basis),
            "confidence": self.confidence,
            "asserted_at": self.asserted_at,
            "source": self.source,
            "producer": self.producer.to_dict() if self.producer else None,
            "evidence": [e.to_dict() for e in self.evidence],
            "supersedes": self.supersedes,
            "retracted": self.retracted,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> IdentityAssertion:
        producer = d.get("producer")
        return cls(
            id=d["id"],
            subject=SubjectRef(d["subject"]["kind"], d["subject"]["id"]),
            person=d["person"],
            basis=d["basis"],
            confidence=d["confidence"],
            asserted_at=d["asserted_at"],
            source=d["source"],
            producer=Producer.from_dict(producer) if producer else None,
            evidence=tuple(EvidenceRef.from_dict(e) for e in d.get("evidence", ())),
            supersedes=d.get("supersedes"),
            retracted=bool(d.get("retracted", False)),
        )


class Tier(enum.IntEnum):
    """How much an assertion counts under decision 2. A higher tier wins."""

    SYSTEM = 1  # a model matched an embedding: compared by confidence
    CONFIRMED = 2  # a spoken introduction, a name badge, or a cluster merge
    OWNER = 3  # the owner tagged or corrected it


OWNER_BASES = frozenset({Basis.OWNER_TAG, Basis.CORRECTION})


def tier(a: IdentityAssertion) -> Tier:
    if a.basis in OWNER_BASES:
        return Tier.OWNER
    return Tier.SYSTEM if a.basis in MODEL_BASES else Tier.CONFIRMED


def strength(a: IdentityAssertion) -> tuple[Tier, float]:
    """How much an assertion counts before time breaks a tie: the tier, then confidence for system matches only."""
    t = tier(a)
    return (t, a.confidence if t is Tier.SYSTEM else 0.0)


def rank(a: IdentityAssertion) -> tuple[Tier, float, Instant, AssertionId]:
    """Sort key, best last: the strength, then the newer one (decision 2)."""
    return (*strength(a), a.asserted_at, a.id)


@dataclasses.dataclass(frozen=True, slots=True)
class Decision:
    """Who a subject is, and the assertions behind it.

    `person` is None when no assertion stands. `standing` holds every assertion still in force, best first, so the
    first one is the winner. `retractions` are the withdrawals that apply to this subject.
    """

    person: PersonId | None
    standing: tuple[IdentityAssertion, ...] = ()
    retractions: tuple[IdentityAssertion, ...] = ()

    @property
    def winner(self) -> IdentityAssertion | None:
        return self.standing[0] if self.standing else None


def decide(
    assertions: Iterable[IdentityAssertion],
    resolve: Callable[[PersonId], PersonId | None] = lambda p: p,
    rematched_at: float | None = None,
) -> Decision:
    """Apply decision 2 to every assertion about one subject.

    1. An assertion named in another's `supersedes` no longer counts (rule 6). A retraction is no exception, so a
       later assertion lifts one by superseding it. A retraction that supersedes a retraction undoes it and
       withdraws nothing itself.
    2. A retraction beats everything: it withdraws every assertion that names the same person and was made before
       it, of any tier, and every later one that is not an owner tag. Only the owner can name that person again.
    3. A rematch (`BACKFILL_MATCH`) reran matching against the current face or voice gallery, so a system match made
       before the newest rematch no longer counts: confidences from an older gallery are not comparable. The cutoff
       holds even after that rematch is superseded or retracted, since the gallery was still corrected. A caller
       that drops superseded assertions passes the newest rematch time as `rematched_at`. Owner tags and confirmed
       assertions are untouched by this.
    4. Of the rest, an owner tag beats a confirmed assertion, which beats a system match. System matches compare by
       confidence. A tie goes to the newer one (`asserted_at`).

    `resolve` maps a person to the one they were merged into, or to None once they are forgotten. Assertions about a
    forgotten person do not count.
    """
    every = list(assertions)
    replaced = {a.supersedes for a in every if a.supersedes}
    retraction_ids = {a.id for a in every if a.retracted}
    retractions = [a for a in every if a.retracted and a.id not in replaced and a.supersedes not in retraction_ids]
    standing = []
    for a in every:
        if a.retracted or a.id in replaced:
            continue
        person = resolve(a.person)
        if person is None:
            continue
        withdrawn = any(
            resolve(r.person) == person and (r.asserted_at >= a.asserted_at or tier(a) is not Tier.OWNER)
            for r in retractions
        )
        if not withdrawn:
            standing.append(a)
    times = [a.asserted_at for a in every if a.basis is Basis.BACKFILL_MATCH]
    rematched = max(times + ([rematched_at] if rematched_at is not None else []), default=None)
    if rematched is not None:
        standing = [a for a in standing if tier(a) is not Tier.SYSTEM or a.asserted_at >= rematched]
    standing.sort(key=rank, reverse=True)
    return Decision(resolve(standing[0].person) if standing else None, tuple(standing), tuple(retractions))
