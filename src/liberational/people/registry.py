"""The person registry and the person_of projection (docs/DOMAIN-MODEL.md, "Records" and rule 2).

Every person gets a `PersonId` that never changes, a UUIDv7. Each place that writes a label appends the matching event
to the independent knowledge SQLite event log through this module.

Transcript lines are different (rule 1: observations are immutable). A label given to one line is only an event here: a relabel, an unlabel, a
"multiple speakers", and a backfill voice match never rewrite `transcript.jsonl`. Readers see each line through
`people/transcript_view.py`, which asks `said` who said it.

- `Registry` writes the events and is also the projection: it is rebuilt from the log (rule 8), reading new events
  when it is asked something.
- `Registry.person_of(subject)` says which person a voiceprint, faceprint, or observation is, by the identity
  precedence ("Identity precedence", `domain.identity.decide`), and keeps the assertions behind the answer. `Registry.said(lines)` answers it for many
  transcript lines at once.
- A line can also carry a label that names nobody, "unknown" or "multiple speakers" (`UtteranceLabeled`), so a line
  the owner took away from its voice stays that way even when the voice had no name to withdraw.
- `safely(...)` wraps every dual write: a failure is logged and never breaks the store write it sits beside. A line
  label has no store write beside it, so its writers let a failure through to the person who asked.
- `describe(name, ...)` sets a person's role and organization, from a tag such as "that's Alice, CEO of Acme" (step
  6, parsed in `people/tags.py`) or the dashboard. The log is where a person's attributes live; an identity
  record (`people/identity.py`, on the broker) also keeps what its tag said.
- `contact(name, ...)` sets a person's phone number, or marks them as added by hand (People: Add). `contacts.json`
  and `people_added.json` are read models of these, written beside the log by `people/contacts.py`. A
  rename or a merge changes one `Person`, so the number, the mark, and the old names (`aliases`) go with it.
- `backfill(...)` gives every name in today's stores a person id, records today's voice labels and hand-tagged
  lines as assertions, and copies phone numbers and added names in. Writing nothing that is already there makes it
  safe to rerun: `pack-registry backfill`.

Streams. Person events go on "people", and assertions and line labels on "identity", so one reader sees each in one
order. Backfill face matches on recorded sightings can be many, so they have their own stream, "identity.sightings",
which is read only when someone asks about a sighting.

Subjects. Today's stores have no ids for prints, so a subject id is the store's own key:

    speaker:<id>        a voice cluster in speakers.json (its ids are never reused)
    face-group:<id>     an unnamed face group ("Face N") that someone named
    identity:<path>     the face candidate of an imported identity record, by its saved embedding (history only)
    detection:<id>      one face observation, what a face match or correction is about: a `face.match` row's id in
                        the segment store (see-actions' head faces), or a `face.detection` payload before
    enroll:<uuid>       a face enrolled from images by `pack-faces enroll`
    utterance:<key>     one transcript line, by `line_key`: `<device slug>@<ts_start to the millisecond>`, or the
                        millisecond alone for a line with no device. Earlier lines had the millisecond alone,
                        and what was said under that id still counts for each line it names (`legacy_of`), until
                        something said about one of them supersedes it for that line.
    sighting:<row id>   one face found in recorded video (sightings.sqlite3)
    photo:<ms>          one saved face photo of a person

Forgetting ("Forgetting" in docs/DOMAIN-MODEL.md): `forget` appends `PersonForgotten` and purges every event whose envelope names the person
or anyone merged into them. Assertion events name their person on the envelope for this reason.
"""

import argparse
import bisect
import collections
import contextlib
import dataclasses
import hashlib
import hmac
import json
import math
import os
import pathlib
import re
import secrets
import sys
import threading
import uuid
from collections.abc import Generator, Iterable, Iterator, Mapping
from typing import Any, TypeGuard

from liberational import config as cfg
from liberational import logs
from liberational.domain import kernel
from liberational.domain.eventlog import EventLog, Selector, Stored
from liberational.domain.identity import (
    MODEL_BASES,
    Basis,
    Decision,
    IdentityAssertion,
    Person,
    SubjectKind,
    SubjectRef,
    Tier,
    decide,
    rank,
    strength,
    tier,
)
from liberational.domain.kernel import AssertionId, Event, EventType, EvidenceKind, EvidenceRef, PersonId, Producer

PEOPLE, ASSERTIONS, SIGHTINGS = "people", "identity", "identity.sightings"
MEASUREMENTS = "measurements"  # spoken quantities (memory/quantities.py), read back by the projections
_PAGE = 1000
DECIDED_MAX = 200_000  # decisions kept between events: every line and face observation a busy day asks about

ATTRIBUTES = ("role", "organization")  # what a tag or the Role button sets (Person)
FIELDS = (*ATTRIBUTES, "phone")  # every text field of a person besides their name (Person)
CONTACT = ("phone", "added")  # what `contact` sets (people/contacts.py)
AUTO = re.compile(r"(?:Speaker|Face) \d+|Caller \S*", re.IGNORECASE)  # in any capitals, as names compare
NOT_PEOPLE = {"", "unknown", "multiple speakers"}


LOG = logs.get("registry")


def norm(name: object) -> str:
    """The key a name is matched by: spaces collapsed, case folded (the stores compare names without case)."""
    return " ".join(str(name or "").split()).casefold()


def is_name(label: object) -> TypeGuard[str]:
    """A label that names a person, not an automatic one ("Speaker 4", "Face 2", "unknown", in any capitals) or the
    assistant."""
    n = " ".join(str(label or "").split())
    return bool(n) and n.casefold() not in NOT_PEOPLE | {norm(cfg.assistant_name())} and not AUTO.fullmatch(n)


# ---- subjects ----
def voiceprint(sid: int | str) -> SubjectRef:
    return SubjectRef(SubjectKind.VOICEPRINT, f"speaker:{int(sid)}")


#: The namespace of subject_uuid's v5 ids. Fixed for good: changing it changes every print's Continuum id.
SUBJECT_NAMESPACE = uuid.UUID("2da871d0-b163-4c60-bbf4-976d9d5cf73b")


def subject_uuid(ref: SubjectRef) -> str:
    """A voiceprint's, faceprint's, or face group's id as a Continuum subject: a UUIDv5 named from the ref's kind and
    id, so the same print always has the same id wherever it's written (voice.embedding, face payloads, a purge by
    entity). A person's subject is their PersonId as it is."""
    return str(uuid.uuid5(SUBJECT_NAMESPACE, f"{ref.kind.value}:{ref.id}"))


def face_group(gid: int | str) -> SubjectRef:
    return SubjectRef(SubjectKind.FACEPRINT, f"face-group:{int(gid)}")


def faceprint(key: str) -> SubjectRef:
    return SubjectRef(SubjectKind.FACEPRINT, key)


def slugify(s: str) -> str:
    """A device's name as lowercase letters, digits, and hyphens: its folder under state/audio, and its part of a line
    key."""
    s = re.sub(r"[^a-z0-9]+", "-", s.lower().replace("\u2019", "")).strip("-")
    return s or "device"


def line_key(ts: float | str, device: object = None) -> str:
    """The key of one transcript line: `<device slug>@<ts_start to the millisecond>`. Two microphones can start
    lines in the same millisecond, so the device is part of it. A line with no device has the millisecond alone, the
    key every line had before."""
    t = f"{float(ts):.3f}"
    return f"{slugify(device)}@{t}" if isinstance(device, str) and device.strip() else t


def key_of(e: Mapping[str, Any]) -> str:
    """The key of a transcript line as stored (its ts_start and device)."""
    return line_key(e["ts_start"], e.get("device"))


def split_key(key: str) -> tuple[str | None, float]:
    """(the device slug, or None, and the ts_start) of a line key or an utterance subject id. A slug has no "@"."""
    slug, _, t = key.removeprefix("utterance:").rpartition("@")
    return slug or None, float(t)


def utterance(ts: float | str, device: object = None) -> SubjectRef:
    """One transcript line, by its ts_start and the device that heard it (`line_key`)."""
    return SubjectRef(SubjectKind.OBSERVATION, f"utterance:{line_key(ts, device)}")


def utterance_of(e: Mapping[str, Any]) -> SubjectRef:
    """The subject of a transcript line as stored."""
    return SubjectRef(SubjectKind.OBSERVATION, f"utterance:{key_of(e)}")


def legacy_of(subject: SubjectRef) -> SubjectRef | None:
    """The id a line keyed by its device had before device-prefixed keys (the millisecond alone), or None for any other
    subject. What was said under that id still counts for every line that starts in that millisecond, until something
    said about one of them supersedes it for that line."""
    if not subject.id.startswith("utterance:"):
        return None
    slug, ts = split_key(subject.id)
    return utterance(ts) if slug else None


def sighting(row: int | str) -> SubjectRef:
    return SubjectRef(SubjectKind.OBSERVATION, f"sighting:{int(row)}")


def detection(payload_id: str) -> SubjectRef:
    """One face observation (a `face.match` row's id, which is its own detection): what the identity worker's
    claims and the owner's face corrections name, and what readers ask `person_of` about."""
    return SubjectRef(SubjectKind.OBSERVATION, f"detection:{payload_id}")


def photo(ms: int | str) -> SubjectRef:
    return SubjectRef(SubjectKind.OBSERVATION, f"photo:{int(ms)}")


def photo_folder(name: str | None) -> str:
    """The folder under `paths.faces` that holds a person's saved face photos, named `<ms>.jpg`."""
    return re.sub(r"[^\w-]+", "-", re.sub(r"\s*\(phone\)$", "", str(name or "")).strip().lower()).strip("-") or "_"


def subject_from_id(text: str) -> SubjectRef:
    """A subject from its id as printed, such as "speaker:4" or "utterance:1761234567.123"."""
    prefix = text.split(":", 1)[0]
    if prefix == "speaker":
        return SubjectRef(SubjectKind.VOICEPRINT, text)
    if prefix in ("face-group", "identity", "enroll"):
        return SubjectRef(SubjectKind.FACEPRINT, text)
    return SubjectRef(SubjectKind.OBSERVATION, text)


def stream_of(subject: SubjectRef) -> str:
    return SIGHTINGS if subject.id.startswith("sighting:") else ASSERTIONS


def voice_producer() -> Producer:
    return Producer("speaker embedding", pathlib.Path(cfg.get("models.files.speaker_embedding.path")).name)


def face_producer() -> Producer:
    return Producer("adaface", cfg.get("models.adaface.recognition"))


def utterance_evidence(ts: float | str, device: object = None) -> EvidenceRef:
    return EvidenceRef(EvidenceKind.UTTERANCE, utterance(ts, device).id)


@dataclasses.dataclass(frozen=True, slots=True)
class Claim:
    """One assertion to write: `subject` is the person called `name`, with this confidence and evidence."""

    subject: SubjectRef
    name: str
    confidence: float = 1.0
    evidence: tuple[EvidenceRef, ...] = ()


def _key(subject: SubjectRef) -> tuple[str, str]:
    return str(subject.kind), subject.id


@dataclasses.dataclass(frozen=True, slots=True)
class Said:
    """Who said one transcript line, and why.

    `person` is who it was, or None. `own` is the decision about the line itself: when it names someone, the owner
    labeled the line or the backfill matched it. Otherwise the person came through `voice`, the decision about the
    voice it matched. `detached` means the owner took the line away from its voice, and `label` is what they called it
    instead ("unknown", "multiple speakers"), when the log says.
    """

    person: PersonId | None
    own: Decision
    voice: Decision | None = None
    label: str | None = None
    detached: bool = False


class Registry:
    """People and identity assertions, written to the event log and projected from it.

    Safe to share across threads. Every answer starts by reading the events other processes appended since the last
    one, so two processes that share a log agree.
    """

    def __init__(self, log: EventLog) -> None:
        self.log = log
        self.lock = threading.RLock()
        self._reset()

    def _reset(self) -> None:
        """Drop projection state after erasure, database replacement, or an aborted worker transaction."""
        self._pos = collections.defaultdict(int)
        self._seen: dict[str, object] = {}  # per stream, the log's version when it was last read through
        self._generation: tuple[str, int] | None = None
        self._revision = 0
        self._voiceprints: set[int] = set()
        self._voiceprint_cache: dict[PersonId | None, tuple[int, ...]] = {}
        self._people: dict[PersonId, Person] = {}
        self._names: dict[str, PersonId] = {}
        # Everyone who has had each name, oldest first, for labels written before a rename. A list, so that when the
        # newest holder is forgotten the name falls back to the one before, as a projection rebuilt without them does.
        self._known: dict[str, list[PersonId]] = {}
        # name -> (when it was given, to whom), by when the log recorded it: a backfill may date a registration to
        # old evidence, but a name is taken when it is given. The holder is None for a time the name was given to
        # someone since forgotten.
        self._held: dict[str, list[tuple[float, PersonId | None]]] = {}
        # name -> when a rename or a merge took it away from the person who had it, by when the log recorded it
        self._ended: dict[str, list[float]] = {}
        # When each name was given to someone since forgotten, by salted hash, and when the forget was recorded.
        self._tombs: list[tuple[bytes, dict[str, list[float]], float]] = []
        self._merged: dict[PersonId, PersonId] = {}
        self._forgotten: set[PersonId] = set()
        self._claims: dict[tuple[str, str], list[IdentityAssertion]] = collections.defaultdict(list)
        self._line_labels: dict[str, str] = {}  # utterance subject id -> a label that names nobody
        self._rematched: dict[tuple[str, str], float] = {}  # newest rematch per subject, kept past supersession
        # What was said under a line's legacy id counts for every line in that millisecond until one of them changes
        # it: then it no longer counts for that line alone. The legacy assertions each line superseded, and the
        # lines whose own label was cleared, so a legacy label doesn't show through.
        self._shadowed: dict[tuple[str, str], set[AssertionId]] = collections.defaultdict(set)
        self._unlabeled: set[str] = set()
        self._dead: set[AssertionId] = set()  # superseded by a forgotten person's assertion, whose purge outlived it
        self._raced: set[PersonId] = set()  # registrations that lost a race for their name (see _apply)
        # `_decide`'s answers until the next event applies: a People read decides every retained line under
        # the lock, and request threads queued behind it. Everything a decision reads changes only in `_apply`.
        self._decided: dict[SubjectRef, Decision] = {}

    # ---- the projection ----
    def refresh(self, *streams: str) -> None:
        """Apply the events appended to these streams (people and identity by default) since the last refresh. A
        stream is read only when the log has been written to since it was last read through: a People request
        refreshes about a hundred times, and reading found nothing new nearly every time."""
        probe = getattr(self.log, "identity_version", None) or getattr(self.log, "version", None)
        with self.lock:
            now = probe() if probe is not None else None
            if isinstance(now, tuple):
                generation = (now[0], now[2])
                if self._generation is not None and (generation != self._generation or now[1] < self._revision):
                    self._reset()
                self._generation = generation
                self._revision = now[1]
            for s in streams or (PEOPLE, ASSERTIONS):
                if now is not None and self._seen.get(s) == now:
                    continue
                while True:
                    page = self.log.read(s, after=self._pos[s], limit=_PAGE)
                    for stored in page:
                        self._apply(stored.event)
                        self._pos[s] = stored.position
                    if len(page) < _PAGE:
                        break
                if now is not None:
                    self._seen[s] = now  # read as of `now` at least: a write since moves the version again

    def _apply(self, e: Event) -> None:
        self._decided.clear()
        p = e.payload
        if e.stream == PEOPLE:
            self._voiceprint_cache.clear()
        elif e.type in (EventType.IDENTITY_ASSERTED, EventType.IDENTITY_RETRACTED):
            subject = p["assertion"]["subject"]
            if subject["kind"] == SubjectKind.VOICEPRINT and subject["id"].startswith("speaker:"):
                try:
                    self._voiceprints.add(int(subject["id"].removeprefix("speaker:")))
                except ValueError:
                    pass
                self._voiceprint_cache.clear()
        if e.type is EventType.PERSON_REGISTERED:
            pid, name = p["person"], p["display_name"]
            if pid in self._people or pid in self._forgotten:
                return
            first = self._names.get(norm(name))
            self._had(name, first if first in self._people else pid)  # ty: ignore[invalid-argument-type]  # in _people, so not None
            attrs = {k: p.get(k) or None for k in FIELDS}
            added = bool(p.get("added"))
            if first in self._people:  # two processes registered one name at once: the first keeps it
                self._merged[pid] = first  # ty: ignore[invalid-assignment]  # in _people, so not None
                self._raced.add(pid)
                self._fill(first, attrs, added)  # ty: ignore[invalid-argument-type]  # in _people, so not None
                return
            self._people[pid] = Person(
                pid, name, role=attrs["role"], organization=attrs["organization"], phone=attrs["phone"], added=added
            )
            self._names[norm(name)] = pid
            self._hold(norm(name), e.recorded_at, pid)
        elif e.type is EventType.PERSON_ATTRIBUTES_CHANGED:
            person, changes = self._people.get(p["person"]), p["changes"]
            if person is None:
                return
            if name := changes.get("display_name"):
                if self._names.get(norm(person.display_name)) == person.id:
                    del self._names[norm(person.display_name)]
                    if norm(person.display_name) != norm(name):
                        self._ended.setdefault(norm(person.display_name), []).append(e.recorded_at)
                aliases = _aliases(name, *person.aliases, person.display_name)
                person = dataclasses.replace(person, display_name=name, aliases=aliases)
                self._names[norm(name)] = person.id
                self._had(name, person.id)
                self._hold(norm(name), e.recorded_at, person.id)
            attrs: dict[str, str | bool | None] = {k: changes[k] or None for k in FIELDS if k in changes}  # "" clears
            if "added" in changes:
                attrs["added"] = bool(changes["added"])
            self._people[person.id] = dataclasses.replace(person, **attrs)
            for assertion in p.get("owner_voice_assertions", ()):
                self._apply(_assertion_event(IdentityAssertion.from_dict(assertion)))
        elif e.type is EventType.PERSONS_MERGED:
            src, dst = p["from"], p["into"]
            if src == dst:
                return
            self._merged[src] = dst
            # What only the merged person had is kept, unless the survivor has its own, and their names become aliases.
            if gone := self._people.get(src):
                if self._names.get(norm(gone.display_name)) == src:
                    self._ended.setdefault(norm(gone.display_name), []).append(e.recorded_at)
                self._fill(dst, {k: getattr(gone, k) for k in FIELDS}, gone.added, (gone.display_name, *gone.aliases))
            self._drop(src)
            # Explicit This is me records the merge and its confirmed authority assertions in one broker item.
            # Ordinary merge payloads have no assertions and still cannot confer owner voice authority.
            for assertion in p.get("owner_voice_assertions", ()):
                self._apply(_assertion_event(IdentityAssertion.from_dict(assertion)))
        elif e.type is EventType.PERSON_FORGOTTEN:
            group = {p["person"], *p.get("merged", ())}
            for pid in group:
                self._forgotten.add(pid)
                self._merged.pop(pid, None)
                self._drop(pid)
            # The names they took go back to whoever had them before, as in a projection rebuilt without their events.
            for holders in self._known.values():
                holders[:] = [h for h in holders if h not in group]
            # But a name written while they had it stays theirs, so nobody else gets it: their time with each name
            # becomes a holder of None here, and the times kept by hash put the same holders in a rebuilt projection.
            for held in self._held.values():
                held[:] = [(since, None if pid in group else pid) for since, pid in held]
            if tenures := p.get("held"):
                tomb = (
                    bytes.fromhex(tenures["salt"]),
                    {d: [float(t) for t in ts] for d, ts in tenures["since"].items()},
                    e.recorded_at,
                )
                self._tombs.append(tomb)
                for key in self._held:
                    self._bury(key, [tomb])
            # What their purged assertions did to everyone else's stays done (see `forget`).
            dead = set(p.get("superseded", ()))
            self._dead |= dead
            for claims in self._claims.values():
                claims[:] = [c for c in claims if c.id not in dead]
            for sid, t in (p.get("rematched") or {}).items():
                k = _key(subject_from_id(sid))
                self._rematched[k] = max(self._rematched.get(k, t), t)
        elif e.type is EventType.UTTERANCE_LABELED:
            if p.get("label"):
                self._line_labels[p["utterance"]] = p["label"]
                self._unlabeled.discard(p["utterance"])
            else:
                self._line_labels.pop(p["utterance"], None)
                self._unlabeled.add(p["utterance"])
        elif e.type in (EventType.IDENTITY_ASSERTED, EventType.IDENTITY_RETRACTED):
            a = IdentityAssertion.from_dict(p["assertion"])
            claims = self._claims[_key(a.subject)]
            if any(c.id == a.id for c in claims):
                return
            if a.basis is Basis.BACKFILL_MATCH:
                k = _key(a.subject)
                self._rematched[k] = max(self._rematched.get(k, a.asserted_at), a.asserted_at)
            if a.supersedes:  # a superseded assertion never counts again, so it need not be kept
                gone = [c for c in claims if c.id == a.supersedes]
                claims[:] = [c for c in claims if c.id != a.supersedes]
                # A line's assertions include those under its legacy id (`_decide`). One it supersedes there stops
                # counting for this line, and still counts for any other line in that millisecond.
                if (old := legacy_of(a.subject)) is not None:
                    hit = [c for c in self._claims.get(_key(old), ()) if c.id == a.supersedes]
                    if hit:
                        gone += hit
                        self._shadowed[_key(a.subject)].add(a.supersedes)
                if a.retracted and any(c.retracted for c in gone):
                    return  # it undid a retraction (decide, rule 1) and says nothing more
            if a.id in self._dead:  # read after the PersonForgotten that keeps it superseded (another stream)
                return
            claims.append(a)

    def _keys(self, subject: SubjectRef) -> list[tuple[str, str]]:
        """Where what was said about `subject` is kept: its own key, and for a line keyed by its device the key it had
        before device-prefixed keys."""
        old = legacy_of(subject)
        return [_key(subject), *([_key(old)] if old else [])]

    def _decide(self, subject: SubjectRef) -> Decision:
        """Decision 2 over the cached claims about `subject` (a line's include those under its legacy id), with its
        rematch cutoff (superseded ones are not kept)."""
        if (hit := self._decided.get(subject)) is not None:
            return hit
        keys = self._keys(subject)
        shadowed = self._shadowed.get(keys[0], set())
        claims = [c for k in keys for c in self._claims.get(k, ()) if c.id not in shadowed]
        cutoff = max((self._rematched[k] for k in keys if k in self._rematched), default=None)
        if len(self._decided) >= DECIDED_MAX:
            self._decided.clear()
        found = self._decided[subject] = decide(claims, self.resolve, cutoff)
        return found

    def _label(self, subject: SubjectRef) -> str | None:
        """The label that names nobody a line has: its own, else the one given under its legacy id, unless the line's
        own label was cleared since."""
        if subject.id in self._line_labels:
            return self._line_labels[subject.id]
        old = legacy_of(subject)
        if old is None or subject.id in self._unlabeled:
            return None
        return self._line_labels.get(old.id)

    def _unlabel(self, subject: SubjectRef, source: str, at: float | None) -> list[Event]:
        """The event that clears a line's label (its own, or the legacy one it shows), for this line alone."""
        return [_line_label(subject, None, source, at)] if self._label(subject) is not None else []

    def _fill(
        self, pid: PersonId, attrs: dict[str, str | None], added: bool = False, names: tuple[str, ...] = ()
    ) -> None:
        """Give `pid` each field in `attrs` that they have no value for, the added mark if `added`, and `names` as
        aliases (a merge: the merged person's name and old names)."""
        person = self._people.get(pid)
        if person is not None:
            blank = {k: v for k, v in attrs.items() if v and getattr(person, k) is None}
            aliases = _aliases(person.display_name, *person.aliases, *names)
            self._people[pid] = dataclasses.replace(person, **blank, aliases=aliases, added=person.added or added)

    def _had(self, name: str, pid: PersonId) -> None:
        """`pid` is now the newest person to have had `name`."""
        holders = self._known.setdefault(norm(name), [])
        if pid in holders:
            holders.remove(pid)
        holders.append(pid)

    def _hold(self, key: str, since: float, pid: PersonId) -> None:
        """The normalized name `key` was given to `pid` at `since`."""
        if key not in self._held:
            self._held[key] = []
            self._bury(key, self._tombs)
        bisect.insort(self._held[key], (since, pid), key=lambda h: h[0])

    def _bury(self, key: str, tombs: list[tuple[bytes, dict[str, list[float]], float]]) -> None:
        """Add to `key`'s holders each time `tombs` says it was given to someone since forgotten."""
        held = self._held[key]
        for salt, since, _ in tombs:
            for t in since.get(_digest(salt, key), ()):
                if (t, None) not in held:
                    bisect.insort(held, (t, None), key=lambda h: h[0])

    def _last_known(self, key: str) -> PersonId | None:
        """The newest person who had the normalized name `key` and has not been forgotten, followed through merges."""
        return next((r for pid in reversed(self._known.get(key, ())) if (r := self.resolve(pid))), None)

    def _drop(self, pid: PersonId) -> None:
        person = self._people.pop(pid, None)
        if person and self._names.get(norm(person.display_name)) == pid:
            del self._names[norm(person.display_name)]

    def resolve(self, pid: PersonId | None) -> PersonId | None:
        """The person `pid` is now: followed through merges, or None once forgotten or never registered."""
        seen = set()
        while pid in self._merged and pid not in seen:
            seen.add(pid)
            pid = self._merged[pid]  # ty: ignore[invalid-argument-type]  # pid is a key of _merged, so not None
        return pid if pid in self._people else None

    def person_id(self, name: str) -> PersonId | None:
        with self.lock:
            self.refresh(PEOPLE)
            return self.resolve(self._names.get(norm(name)))

    def owner(self) -> PersonId | None:
        """The owner's person: the one called owner.speaker_label or owner.name now. None when nobody is, or when the
        two settings name two different people (the owner check fails closed on that)."""
        keys = {norm(cfg.owner_label()), norm(cfg.owner_name())} - {""}
        with self.lock:
            self.refresh(PEOPLE)
            found = {self.resolve(self._names.get(k)) for k in keys} - {None}
        return next(iter(found)) if len(found) == 1 else None

    def same_registration(self, pid: PersonId | None, person: PersonId) -> bool:
        """Whether an assertion that names `pid` was made about `person` itself: the same id, or a second registration
        of their name that lost a race to theirs (two processes registered it at once). Never through a merge or a
        rename someone asked for, so moving a person into someone never moves the owner tags on their voices."""
        with self.lock:
            return pid == person or (pid in self._raced and self._merged.get(pid) == person)

    def known_as(
        self, name: str, at: float | None = None, refresh: bool = True, strict: bool = False
    ) -> PersonId | None:
        """The person called `name` now, or else the person last called it before a rename or a merge (skipping anyone
        forgotten since). Old labels that were never rewritten, such as the name on a hand-tagged transcript line,
        resolve this way.

        With `at`, a label written at that time: the person who was called `name` then (the first one ever, for a
        time before anyone was), followed through merges, so someone new who takes an old name later never gets
        what was written under it before. None when that person was forgotten. With `refresh` False it answers from
        what was last read, for a caller that just refreshed.

        A label can be written under an old name for a while after a rename (a service that has not read it yet), so
        a time after a rename or a merge took the name away still gives the person who had it last. With `strict`,
        for a request made at `at` (a tag), it gives None then: nobody was called that, so the request means someone
        new."""
        with self.lock:
            if refresh:
                self.refresh(PEOPLE)
            key = norm(name)
            if at is not None and (held := self._held.get(key)):
                then = [(since, pid) for since, pid in held if since <= at]
                if strict and then and any(then[-1][0] <= t <= at for t in self._ended.get(key, ())):
                    return None
                return self.resolve(then[-1][1] if then else held[0][1])
            return self.resolve(self._names.get(key)) or self._last_known(key)

    def forgotten_at(self, name: str, at: float | None = None, refresh: bool = True) -> bool:
        """Did `name`, written at `at`, mean someone since forgotten? True when the registry dates the name and
        `known_as` finds nobody living for it: the time a forgotten person had a name stays a holder of None, live
        and in a projection rebuilt from `PersonForgotten` (whose hashed times name it even when nobody living has
        had it since). It needs no hash in `FORGOTTEN`, so it holds with assistant.forgotten_reply off.
        Someone forgotten before those times were kept is not known here."""
        with self.lock:
            if refresh:
                self.refresh(PEOPLE)
            key = norm(name)
            if key not in self._held:
                return any(_digest(salt, key) in since for salt, since, _ in self._tombs)
            return self.known_as(name, at, refresh=False) is None

    def forgotten_since(self, name: str, requested_at: float | None) -> bool:
        """Whether a forget superseded this queued teaching request, using projected salted name tombstones.

        Check inside the owner's write transaction before registering a name. A request made strictly after the
        forget is fresh teaching and may register someone new; equal times and undated legacy requests fail closed.
        Unlike `forgotten_at`, this compares the request with the forget's recorded time, not the name's tenure.
        """
        with self.lock:
            self.refresh(PEOPLE)
            key = norm(name)
            return any(
                (requested_at is None or forgotten_at >= requested_at) and _digest(salt, key) in since
                for salt, since, forgotten_at in self._tombs
            )

    def events(
        self,
        types: Iterable[EventType | str],
        start: float,
        end: float,
        streams: Iterable[str] = (PEOPLE, ASSERTIONS, SIGHTINGS, MEASUREMENTS),
    ) -> list[Stored]:
        """The stored events of these types that occurred in `[start, end)`, in time order, across `streams` (the
        registry's own and the measurements by default), read from the log as it is now. For the projections
        (event counts, the timeline), which otherwise would open the log themselves."""
        wanted = {EventType(t) for t in types}
        out: list[Stored] = []
        for s in streams:
            after = 0
            while True:
                page = self.log.read(s, after=after, limit=_PAGE)
                for stored in page:
                    after = stored.position
                    e = stored.event
                    if e.type in wanted and start <= e.occurred_at < end:
                        out.append(stored)
                if len(page) < _PAGE:
                    break
        out.sort(key=lambda stored: (stored.event.occurred_at, stored.position))
        return out

    def mark(self) -> tuple[int, ...]:
        """Where the log stands: the position of the last event applied per registry stream, after reading what is
        new. It changes whenever any process appends to a registry stream (a label, a rename, a forget), so a cache
        keyed on it refreshes then; the dashboard's `log_mark` and the projections' signatures use it in place of the
        event log file's size and time."""
        with self.lock:
            self.refresh(PEOPLE, ASSERTIONS, SIGHTINGS)
            generation = (int(uuid.UUID(self._generation[0])), self._generation[1]) if self._generation else ()
            return (*[self._pos[s] for s in (PEOPLE, ASSERTIONS, SIGHTINGS)], *generation)

    def person(self, pid: PersonId) -> Person | None:
        with self.lock:
            self.refresh(PEOPLE)
            return self._people.get(self.resolve(pid))

    def people(self) -> list[Person]:
        with self.lock:
            self.refresh(PEOPLE)
            return sorted(self._people.values(), key=lambda p: norm(p.display_name))

    def person_of(self, subject: SubjectRef, via: SubjectRef | None = None) -> Decision:
        """Who `subject` is, by decision 2, with the assertions behind the answer.

        `via` is a print the subject matched, such as the voiceprint of a transcript line. It answers when nothing
        about the subject itself stands, unless the subject has a retraction that withdrew that very person.
        """
        with self.lock:
            self.refresh(PEOPLE, stream_of(subject))
            own = self._decide(subject)
            if own.person is not None or via is None:
                return own
            if (label := self._label(subject)) is not None:  # the owner took it away from its voice
                if (moved := split_voice(label)) is not None:  # to another voice: that voice's person
                    return self.person_of(voiceprint(moved))
                return Decision(None, (), own.retractions)
            through = self.person_of(via)
            if through.person is not None and any(self.resolve(r.person) == through.person for r in own.retractions):
                return Decision(None, (), own.retractions)
            return through

    def said(self, lines: Iterable[tuple[SubjectRef | float, int | None]], refresh: bool = True) -> list[Said]:
        """Who said each transcript line, given as (its subject, the voice id it matched, merges followed); a bare
        ts_start stands for a line with no device. One read of the log answers them all; with `refresh` False it
        answers from what was last read, for a caller that just refreshed. The same rule as
        `person_of(line, via=voiceprint(sid))`, with the reason kept."""
        with self.lock:
            if refresh:
                self.refresh()
            voices: dict[int, Decision] = {}
            out = []
            for line, sid in lines:
                subject = line if isinstance(line, SubjectRef) else utterance(line)
                own = self._decide(subject)
                if own.person is not None:
                    out.append(Said(own.person, own))
                    continue
                if (label := self._label(subject)) is not None:
                    if (moved := split_voice(label)) is not None:  # split off to a voice of its own
                        if moved not in voices:
                            voices[moved] = self._decide(voiceprint(moved))
                        out.append(Said(voices[moved].person, own, voices[moved], label=label, detached=True))
                        continue
                    out.append(Said(None, own, label=label, detached=True))
                    continue
                if sid is None:
                    out.append(Said(None, own))
                    continue
                if sid not in voices:
                    voices[sid] = self._decide(voiceprint(sid))
                voice = voices[sid]
                if voice.person is not None and any(self.resolve(r.person) == voice.person for r in own.retractions):
                    out.append(Said(None, own, voice, detached=True))  # unlabeled before lines carried a label
                    continue
                out.append(Said(voice.person, own, voice))
            return out

    def voices(self, sids: Iterable[int], refresh: bool = True) -> dict[int, PersonId | None]:
        """The person each voice (a speakers.json id) is, or None."""
        with self.lock:
            if refresh:
                self.refresh()
            return {int(i): self._decide(voiceprint(i)).person for i in sids}

    def voice_names(self, sids: Iterable[int], refresh: bool = True) -> dict[int, str | None]:
        """The name each voice (a speakers.json id) goes by now: its person's name, or None while it names nobody.
        speakers.json holds no names, so this is where every voice's name comes from."""
        with self.lock:
            out: dict[int, str | None] = {}
            for i, pid in self.voices(sids, refresh).items():
                p = self._people.get(pid) if pid is not None else None
                out[i] = p.display_name if p else None
            return out

    def has_claims(self, subject: SubjectRef) -> bool:
        """Whether anything was ever said about `subject` that is still in the log, withdrawn or not."""
        with self.lock:
            self.refresh(PEOPLE, stream_of(subject))
            return any(k in self._claims for k in self._keys(subject)) or self._label(subject) is not None

    def observed(self, prefix: str) -> list[str]:
        """The ids of the observations starting with `prefix` (`"detection:"`, say) that anything was said about and is
        still in the log, withdrawn or not."""
        with self.lock:
            self.refresh(PEOPLE, ASSERTIONS)
            kind = str(SubjectKind.OBSERVATION)
            return [sid for k, sid in self._claims if k == kind and sid.startswith(prefix)]

    def utterances_of(self, name: str) -> list[str]:
        """The key (`line_key`) of each transcript line labeled as the person called `name` (or anyone merged into
        them), so forgetting them can put those lines back to unnamed the way it does for lines that carry the name.
        A key with no device is a millisecond, and stands for every line that starts in it (`legacy_of`)."""
        with self.lock:
            pid = self.person_id(name)
            if pid is None:
                return []
            self.refresh(ASSERTIONS)
            return sorted(
                sid.removeprefix("utterance:")
                for kind, sid in self._claims
                if sid.startswith("utterance:") and self._decide(SubjectRef(SubjectKind(kind), sid)).person == pid
            )

    def legacy_lines(self) -> set[str]:
        """The legacy (millisecond-only) line ids anything was said under, withdrawn or not: what `pack-registry
        ambiguous` checks against the transcript."""
        with self.lock:
            self.refresh(ASSERTIONS)
            ids = {sid for _, sid in self._claims} | set(self._line_labels)
            return {i for i in ids if i.startswith("utterance:") and split_key(i)[0] is None}

    # ---- writing ----
    def _append(self, events: list[Event]) -> None:
        if events:
            self.log.append(events)
            self.refresh(PEOPLE, *dict.fromkeys(e.stream for e in events))

    def ensure(self, name: str, source: str, at: float | None = None, **attributes: str | bool | None) -> PersonId:
        """The id of the person called `name`, registering them first if nobody has that name. `attributes` (a role,
        an organization, a phone number, and the added mark) go on the registration of someone new; `describe` and
        `contact` change them for anyone."""
        kernel.check(is_name(name), f"not a person's name: {name!r}")
        attributes = _attributes(attributes, (*FIELDS, "added"))
        with self.lock:
            pid = self.person_id(name)
            if pid is None:
                pid = PersonId(kernel.new_id())
                payload = {"person": pid, "display_name": " ".join(name.split())}
                payload.update({k: v for k, v in attributes.items() if v})
                self._append([_event(EventType.PERSON_REGISTERED, PEOPLE, source, payload, pid, at)])
                pid = self.person_id(name)  # the first registration wins a race between processes
            return pid  # ty: ignore[invalid-return-type]  # the registration just appended gives the name a person

    def claim(self, subject: SubjectRef, name: str, basis: Basis, source: str, **kw: Any) -> AssertionId | None:
        """Record that `subject` is the person called `name`. See `claim_many`."""
        confidence, evidence = kw.pop("confidence", 1.0), kw.pop("evidence", ())
        made = self.claim_many([Claim(subject, name, confidence, tuple(evidence))], basis, source, **kw)
        return made[0] if made else None

    def claim_many(
        self,
        claims: Iterable[Claim],
        basis: Basis,
        source: str,
        *,
        producer: Producer | None = None,
        at: float | None = None,
        replace: bool = False,
    ) -> list[AssertionId]:
        """Append an `IdentityAsserted` for each claim, in one write. Returns the ids of the assertions written.

        A claim whose name is an automatic label ("Speaker 4") is skipped, and so is one whose confidence is not a
        finite number (a NaN similarity from a zero embedding is no match, not a full-confidence one). So is one that
        is not the owner's about a line the owner labeled "unknown" or "multiple speakers" (decision 2), and one whose
        person already stands at least as strongly (the same tier, and for a system match a confidence as high), so
        mirroring a store write twice writes once.
        A model's match (an embedding or backfill match) never registers anyone: it is skipped when nobody has the
        name now. A gallery read before People: Delete can still name the person it forgot, and only the owner (or a
        confirmed tag) brings a name into the registry.
        A new assertion supersedes the subject's current winner when it ranks at least as high by decision 2 (tier,
        then confidence for system matches, then newer), or always with `replace`, for a store write that overwrote
        the old answer (a merged voice belongs to the survivor). A weaker one stands beside the winner.
        A rematch (`BACKFILL_MATCH`) is a newer gallery's answer, so it supersedes a system match whatever the
        confidence, and `decide` drops older system matches (the owner chose this). It never supersedes an owner tag
        or a confirmed assertion, and it is skipped only when its person already wins.
        With no winner, it supersedes the newest retraction a model made (not the owner) at its tier or below, so a
        rematch that gives a sighting its name back lifts the rematch that took it away.
        """
        at = kernel.now() if at is None else at
        # An automatic label names nobody, and a confidence that isn't a finite number (a NaN) is no match.
        claims = [c for c in claims if is_name(c.name) and math.isfinite(float(c.confidence))]
        events, made = [], []
        with self.lock:
            self.refresh(PEOPLE, *dict.fromkeys(stream_of(c.subject) for c in claims))
            for c in claims:
                pid = self.resolve(self._names.get(norm(c.name)))
                if pid is None and basis in MODEL_BASES:  # a match names someone in a gallery, never someone new
                    continue
                pid = pid or self.ensure(c.name, source, at)
                d = self._decide(c.subject)
                winner = d.winner
                a = IdentityAssertion(
                    id=AssertionId(kernel.new_id()),
                    subject=c.subject,
                    person=pid,
                    basis=basis,
                    confidence=max(0.0, min(1.0, float(c.confidence))),
                    asserted_at=at,
                    source=source,
                    producer=producer,
                    evidence=c.evidence,
                )
                detached = self._label(c.subject) is not None
                if detached and tier(a) is not Tier.OWNER:  # the owner took the line from its voice: only they undo it
                    continue
                same = [s for s in d.standing if self.resolve(s.person) == pid]  # best first
                rematch = basis is Basis.BACKFILL_MATCH
                if same and strength(same[0]) >= strength(a) and (not (replace or rematch) or d.person == pid):
                    continue
                if winner and (replace or rank(a) >= rank(winner) or (rematch and tier(winner) is Tier.SYSTEM)):
                    a = dataclasses.replace(a, supersedes=winner.id)
                elif winner is None:
                    lifted = [r for r in d.retractions if tier(r) is not Tier.OWNER and tier(r) <= tier(a)]
                    if lifted:
                        a = dataclasses.replace(a, supersedes=max(lifted, key=lambda r: r.asserted_at).id)
                events.append(_assertion_event(a))
                made.append(a.id)
                if detached:  # a line the owner gives a person is no longer "unknown" or "several"
                    events += self._unlabel(c.subject, source, at)
            self._append(events)
        return made

    def retract(self, subject: SubjectRef, source: str, **kw: Any) -> list[AssertionId]:
        """Withdraw what `subject` was said to be. See `retract_many`."""
        return self.retract_many([subject], source, **kw)

    def retract_many(
        self,
        subjects: Iterable[SubjectRef],
        source: str,
        *,
        via: SubjectRef | None = None,
        basis: Basis = Basis.CORRECTION,
        producer: Producer | None = None,
        at: float | None = None,
    ) -> list[AssertionId]:
        """Withdraw what each subject was said to be, in one write: each standing assertion about it, and with `via`
        also the winner of `via` for this subject alone (a line unlabeled from its voice's person). `basis` is a
        correction (the owner's) unless a model withdrew the name, as a rematch does. Returns the retractions."""
        at = kernel.now() if at is None else at
        subjects = list(subjects)
        with self.lock:
            events = self._retractions(subjects, source, via, basis, producer, at)
            self._append(events)
            return [e.payload["assertion"]["id"] for e in events]

    def _retractions(
        self,
        subjects: list[SubjectRef],
        source: str,
        via: SubjectRef | None,
        basis: Basis,
        producer: Producer | None,
        at: float,
    ) -> list[Event]:
        self.refresh(PEOPLE, *dict.fromkeys(stream_of(s) for s in subjects))
        events = []
        for subject in subjects:
            targets = list(self._decide(subject).standing)
            if via is not None and (w := self.person_of(via).winner):
                targets.append(w)
            events += [
                _retraction(subject, t, self.resolve(t.person) or t.person, source, basis, producer, at)
                for t in targets
            ]
        return events

    def detach(
        self,
        line: SubjectRef | float,
        label: str,
        source: str,
        via: SubjectRef | None = None,
        at: float | None = None,
    ) -> None:
        """The line is nobody's: "unknown", or "multiple speakers". Withdraws what it was said to be (its own
        assertions, and its voice's person for this line alone) and gives it `label`, in one write. A bare ts_start
        stands for a line with no device."""
        kernel.check_text(label, "line label")
        at = kernel.now() if at is None else at
        subject = line if isinstance(line, SubjectRef) else utterance(line)
        with self.lock:
            events = self._retractions([subject], source, via, Basis.CORRECTION, None, at)
            events.append(_line_label(subject, label, source, at))
            self._append(events)

    def release(self, subject: SubjectRef, source: str, at: float | None = None) -> list[AssertionId]:
        """Withdraw everything said about `subject` itself, its retractions and line label included, so it follows its
        `via` again (a line whose label was undone goes back to following its voice). Returns the retractions written.

        A standing assertion is withdrawn and the withdrawal undone at once (a retraction that supersedes a
        retraction, decide rule 1), so nothing is left that would also withdraw that person from the voice."""
        at = kernel.now() if at is None else at
        with self.lock:
            self.refresh(PEOPLE, stream_of(subject))
            d = self._decide(subject)
            events = []
            for t in (*d.standing, *d.retractions):
                gone = _retraction(subject, t, self.resolve(t.person) or t.person, source, Basis.CORRECTION, None, at)
                events.append(gone)
                if not t.retracted:
                    first = IdentityAssertion.from_dict(gone.payload["assertion"])
                    events.append(_retraction(subject, first, first.person, source, Basis.CORRECTION, None, at))
            events += self._unlabel(subject, source, at)
            self._append(events)
            return [e.payload["assertion"]["id"] for e in events if "assertion" in e.payload]

    def rename(self, old: str, new: str, source: str) -> PersonId:
        """The person called `old` is now called `new`, keeping their id. To a name someone else has, it is a merge."""
        with self.lock:
            pid, other = self.person_id(old), self.person_id(new)
            if pid is None:
                return self.ensure(new, source)
            if other is not None and other != pid:
                return self.merge(old, new, source)
            display, before = " ".join(new.split()), self._people[pid].display_name
            if display != before:
                payload = {"person": pid, "changes": {"display_name": display}, "previous": {"display_name": before}}
                self._append([_event(EventType.PERSON_ATTRIBUTES_CHANGED, PEOPLE, source, payload, pid)])
            return pid

    def describe(
        self, name: str, source: str, given: Mapping[str, str | None], at: float | None = None, fill: bool = False
    ) -> PersonId:
        """Set the role or organization of the person called `name`, registering them with it if nobody has that
        name. `given` maps "role" or "organization" to a value: an empty value clears one, and one left out stays.
        Writes nothing when nothing changes. With `fill`, only an attribute the person has no value for is set, so a
        guess never replaces what the owner said."""
        return self._change(name, source, _attributes(given), at, fill)

    def contact(
        self,
        name: str,
        source: str,
        *,
        phone: str | None = None,
        added: bool = False,
        fill: bool = False,
        at: float | None = None,
    ) -> PersonId:
        """Give the person called `name` a phone number, or mark them as added by hand (People: Add), registering
        them first if nobody has that name. `phone` None leaves the number as it is, and "" clears it. `added` False
        leaves the mark as it is: forgetting someone is how they leave the list. With `fill`, a number is set only for
        a person who has none, so the backfill never replaces one set since. Writes nothing when nothing changes.
        Callers go through `people/contacts.py`, which also writes the read models."""
        given: dict[str, str | bool] = {"phone": phone} if phone is not None else {}
        if added:
            given["added"] = True
        return self._change(name, source, _attributes(given, CONTACT), at, fill)

    def _change(
        self, name: str, source: str, changes: dict[str, str | bool | None], at: float | None, fill: bool
    ) -> PersonId:
        with self.lock:
            pid = self.person_id(name)
            if pid is None:
                return self.ensure(name, source, at, **changes)
            person = self._people[pid]
            diff = {k: v for k, v in changes.items() if getattr(person, k) != v and not (fill and getattr(person, k))}
            if diff:
                payload = {"person": pid, "changes": diff, "previous": {k: getattr(person, k) for k in diff}}
                self._append([_event(EventType.PERSON_ATTRIBUTES_CHANGED, PEOPLE, source, payload, pid, at)])
            return pid

    def merge(self, from_name: str, into_name: str, source: str) -> PersonId:
        """The person called `from_name` is the person called `into_name`. Assertions about either now resolve there."""
        with self.lock:
            src, dst = self.person_id(from_name), self.ensure(into_name, source)
            if src is not None and src != dst:
                self._append([_event(EventType.PERSONS_MERGED, PEOPLE, source, {"from": src, "into": dst}, dst)])
            return dst

    def merge_as_owner(self, from_name: str, voice_ids: list[int], source: str) -> PersonId:
        """Explicit local owner confirmation: merge and grant the confirmed voices in one broker item.

        Ordinary merges deliberately do not confer owner authority. The caller validates the current voice set
        under its People mutation lock; this operation rechecks each voice against the source registration. Without
        an owner record after a reset, the selected person is renamed to configured owner.name in the same item.
        """
        with self.lock:
            self.refresh()
            src, dst = self.person_id(from_name), self.owner()
            owner_name = cfg.owner_name().strip()
            if src is None or not owner_name or src == dst:
                raise ValueError("source and configured owner must exist and differ")
            if dst is None and any(self.person_id(n) is not None for n in (owner_name, cfg.owner_label()) if n):
                raise ValueError("owner names refer to conflicting identities")
            establishing = dst is None
            dst = dst or src
            kernel.check(sorted(voice_ids) == self.voiceprints(src), "voices changed before owner confirmation")
            assertions = []
            for sid in voice_ids:
                subject = voiceprint(sid)
                decision = self.person_of(subject)
                kernel.check(decision.person == src, "voice no longer belongs to this person")
                for previous in decision.standing or (None,):
                    assertions.append(
                        IdentityAssertion(
                            id=AssertionId(kernel.new_id()),
                            subject=subject,
                            person=dst,
                            basis=Basis.OWNER_TAG,
                            confidence=1.0,
                            asserted_at=kernel.now(),
                            source=source,
                            supersedes=previous.id if previous else None,
                        ).to_dict()
                    )
            if establishing:
                event = _event(
                    EventType.PERSON_ATTRIBUTES_CHANGED,
                    PEOPLE,
                    source,
                    {
                        "person": src,
                        "changes": {"display_name": owner_name},
                        "previous": {"display_name": from_name},
                        "owner_voice_assertions": assertions,
                    },
                    src,
                )
            else:
                event = _event(
                    EventType.PERSONS_MERGED,
                    PEOPLE,
                    source,
                    {"from": src, "into": dst, "owner_voice_assertions": assertions},
                    dst,
                )
            self._append([event])
            return dst

    def _group(self, pid: PersonId) -> set[PersonId]:
        return {pid} | {p for p in self._merged if self.resolve(p) == pid}

    def group(self, name: str) -> set[PersonId]:
        """The person called `name` and everyone merged into them; empty when nobody has that name."""
        with self.lock:
            pid = self.person_id(name)
            return self._group(pid) if pid is not None else set()

    def names(self, name: str, others: bool = False) -> set[str]:
        """Every name the person called `name` has had, and the names of anyone merged into them, normalized. A name
        someone else has now is left out, unless `others`: then the caller tells their records apart by time
        (`known_as` with `at`). Empty when nobody has that name. For people/forgotten.py and the journal, before
        `forget`."""
        with self.lock:
            pid = self.person_id(name)
            if pid is None:
                return set()
            group = self._group(pid)
            person = self._people[pid]
            had = {k for k in self._known if self._last_known(k) == pid}
            had |= {k for k, held in self._held.items() if any(p in group for _, p in held)}
            had |= {norm(n) for n in (person.display_name, *person.aliases)}
            return had if others else {k for k in had if self.resolve(self._names.get(k)) in (None, pid)}

    def forget_subjects(self, name: str) -> set[str]:
        """Every Continuum subject id that is the person called `name` now: their PersonId and the ids merged into
        them, and the `subject_uuid` of each voiceprint, faceprint, and face group whose `person_of` is them. A purge
        by entity of each forgets them on the broker, voice and face alike. Read it before `forget`, which takes
        their assertions away. Empty when nobody has that name."""
        with self.lock:
            self.refresh(PEOPLE, ASSERTIONS, SIGHTINGS)
            pid = self.person_id(name)
            if pid is None:
                return set()
            group = self._group(pid)
            out = {str(p) for p in group}
            for kind, sid in list(self._claims):
                if kind not in (SubjectKind.VOICEPRINT, SubjectKind.FACEPRINT):
                    continue
                ref = SubjectRef(SubjectKind(kind), sid)
                if self.resolve(self._decide(ref).person) == pid:
                    out.add(subject_uuid(ref))
            return out

    def voiceprints(self, pid: PersonId | None = None) -> list[int]:
        """The numbers of the voiceprints the log holds claims about (`voiceprint(n)`), lowest first: every one, or
        only those whose `person_of` is the person `pid` is now."""
        with self.lock:
            self.refresh(PEOPLE, ASSERTIONS, SIGHTINGS)
            want = self.resolve(pid) if pid is not None else None
            if pid is not None and want is None:  # forgotten, or never registered: nobody's voices
                return []
            if want not in self._voiceprint_cache:
                self._voiceprint_cache[want] = tuple(
                    n
                    for n in sorted(self._voiceprints)
                    if want is None or self.resolve(self._decide(voiceprint(n)).person) == want
                )
            return list(self._voiceprint_cache[want])

    def forget(self, name: str, source: str) -> int:
        """Forget the person called `name` ("Forgetting" in docs/DOMAIN-MODEL.md): append `PersonForgotten`, then purge every event about
        them or anyone merged into them. Returns how many events were purged; 0 when nobody has that name.

        `PersonForgotten` names only ids, never the name, and has no envelope subject, so it outlives the purge and a
        projection rebuilt later still knows those ids are gone (rule 7). It also keeps what the purged assertions did
        to other people's, by id and time only: the assertions they superseded, and each subject's newest rematch
        time. Without them a rebuilt projection would bring a superseded assertion back, and give the forgotten
        person's lines and voices to whoever had them before (rule 8). It keeps when each of their names was given to
        them, under a salted hash of the name, so a rebuilt projection dates a name (`known_as` with `at`) as the
        live one does: a label written while they had it names nobody."""
        with self.lock:
            pid = self.person_id(name)
            if pid is None:
                return 0
            group = self._group(pid)
            payload = {"person": pid, "merged": sorted(group - {pid}), **self._outliving(group)}
            if tenures := self._tenures(group):
                payload["held"] = tenures
            self._append([_event(EventType.PERSON_FORGOTTEN, PEOPLE, source, payload, None)])
            removed = 0
            try:
                # Them and everyone merged into them, in one purge with one record.
                removed = self.log.purge(Selector(subjects=frozenset(group)), reason="PersonForgotten")
            except RuntimeError as e:
                LOG.info(f"forget: {e}")
            for claims in self._claims.values():
                claims[:] = [c for c in claims if c.person not in group]
            return removed

    def _tenures(self, group: set[PersonId]) -> dict[str, Any] | None:
        """For `forget`: when each name was given to someone in `group`, by a salted hash of the name, or None when
        nothing was. {"salt": hex, "since": {hash: [recorded_at, ...]}}"""
        salt = secrets.token_bytes(16)
        since = {
            _digest(salt, key): ts
            for key, held in self._held.items()
            if (ts := sorted({t for t, pid in held if pid in group}))
        }
        return {"salt": salt.hex(), "since": since} if since else None

    def _outliving(self, group: set[PersonId]) -> dict[str, Any]:
        """For `forget`: what the assertion events about `group` (the ones its purge removes) did to the rest. The
        ids they superseded that the purge leaves ("superseded"), and the newest rematch time per subject id
        ("rematched"), since a rematch's cutoff holds after the rematch is gone (decide, rule 3)."""
        purged: list[IdentityAssertion] = []
        for stream in (ASSERTIONS, SIGHTINGS):
            after = 0
            while page := self.log.read(stream, after=after, limit=_PAGE):
                purged += [
                    IdentityAssertion.from_dict(s.event.payload["assertion"])
                    for s in page
                    if s.event.subject in group and "assertion" in s.event.payload
                ]
                after = page[-1].position
        ids = {a.id for a in purged}
        rematched: dict[str, float] = {}
        for a in purged:
            if a.basis is Basis.BACKFILL_MATCH:
                rematched[a.subject.id] = max(rematched.get(a.subject.id, a.asserted_at), a.asserted_at)
        superseded = sorted({a.supersedes for a in purged if a.supersedes and a.supersedes not in ids})
        return {"superseded": superseded, "rematched": rematched}


def _attributes(given: Mapping[str, object], allowed: tuple[str, ...] = ATTRIBUTES) -> dict[str, str | bool | None]:
    """Attributes as stored: `allowed` keys only, spaces collapsed, None for an empty value, and the added mark as a
    bool."""
    unknown = set(given) - set(allowed)
    kernel.check(not unknown, f"not a person's attribute: {', '.join(sorted(unknown))}")
    return {
        k: bool(v) if k == "added" else (" ".join(str(v).split()) if v is not None else "") or None
        for k, v in given.items()
    }


def _digest(salt: bytes, key: str) -> str:
    """The salted hash `PersonForgotten` keeps of a normalized name, in place of the name."""
    return hmac.new(salt, key.encode(), hashlib.sha256).hexdigest()


def _aliases(name: str, *names: str) -> tuple[str, ...]:
    """The other names of the person called `name`: one spelling per name, in order, never `name` itself."""
    out: dict[str, str] = {}
    for n in names:
        out.setdefault(norm(n), n)
    out.pop(norm(name), None)
    return tuple(out.values())


def _event(
    type: EventType, stream: str, source: str, payload: dict, subject: PersonId | None, at: float | None = None
) -> Event:
    return Event.new(type, stream=stream, source=source, payload=payload, occurred_at=at, subject=subject)


def _retraction(
    subject: SubjectRef,
    target: IdentityAssertion,
    person: PersonId,
    source: str,
    basis: Basis,
    producer: Producer | None,
    at: float,
) -> Event:
    """The event that withdraws `target`, an assertion about `subject` (or about the print it came through)."""
    a = IdentityAssertion(
        id=AssertionId(kernel.new_id()),
        subject=subject,
        person=person,
        basis=basis,
        confidence=1.0,
        asserted_at=at,
        source=source,
        producer=producer,
        supersedes=target.id,
        retracted=True,
    )
    return _assertion_event(a)


def split_label(voice: int) -> str:
    """The label of a line split off a named turn to a voice of its own (a voice tag whose lines disagreed): the
    line goes with that voice from then on, its person included, once the voice is named."""
    return f"Speaker {int(voice)}"


def split_voice(label: str | None) -> int | None:
    """The voice a line's label moved it to, or None for a label that names no voice ("unknown")."""
    m = re.fullmatch(r"Speaker (\d+)", label or "")
    return int(m[1]) if m else None


def _line_label(subject: SubjectRef, label: str | None, source: str, at: float | None) -> Event:
    """A label for a line that names nobody ("unknown", "multiple speakers"), or None to clear it. It names no person,
    so forgetting someone never purges it, and it holds nothing about them."""
    payload = {"utterance": subject.id, "label": label}
    return _event(EventType.UTTERANCE_LABELED, ASSERTIONS, source, payload, None, at)


def _assertion_event(a: IdentityAssertion) -> Event:
    return Event.new(
        EventType.IDENTITY_RETRACTED if a.retracted else EventType.IDENTITY_ASSERTED,
        stream=stream_of(a.subject),
        source=a.source,
        payload={"assertion": a.to_dict()},
        occurred_at=a.asserted_at,
        subject=a.person,
        producer=a.producer,
        evidence=a.evidence,
    )


_registries: dict[str, Registry] = {}
_registries_lock = threading.Lock()
_worker_local = threading.local()
_registry_epoch = 0


def _socket(state: str | pathlib.Path | None) -> pathlib.Path:
    """The broker's socket: `CONTINUUM_SOCKET`, else `<state>/continuum/broker.sock` (`paths.state` by default)."""
    env = os.environ.get("CONTINUUM_SOCKET")
    if env:
        return pathlib.Path(env)
    if state is not None:
        return pathlib.Path(state) / "continuum" / "broker.sock"
    from liberational import continuum_link

    return continuum_link.socket_path()


def at(state: str | pathlib.Path | None = None) -> Registry:
    """The registry in `<state>/knowledge.sqlite3`, independent of broker availability and reset.

    The state directory defaults to central configuration. Each process caches the registry projection by resolved
    database path; SQLite change revisions make writes by other processes visible on the next refresh.
    """
    from liberational.knowledge_eventlog import SQLiteEventLog

    path = knowledge_path(state)
    active = getattr(_worker_local, "active", {})
    if str(path) in active:
        return active[str(path)]
    with _registries_lock:
        if str(path) not in _registries:
            _registries[str(path)] = Registry(SQLiteEventLog(path))
        return _registries[str(path)]


def local(state: str | pathlib.Path | None = None) -> Registry:
    """A persistent projection confined to this calling thread, for bulk reads or the serialized tag worker.

    Never hand this Registry to another thread. Its refreshes and writes do not hold the shared projection's lock.
    """
    from liberational.knowledge_eventlog import SQLiteEventLog

    if getattr(_worker_local, "epoch", None) != _registry_epoch:
        _worker_local.registries = {}
        _worker_local.epoch = _registry_epoch
    path = str(knowledge_path(state))
    registries = _worker_local.registries
    if path not in registries:
        registries[path] = Registry(SQLiteEventLog(pathlib.Path(path)))
    return registries[path]


@contextlib.contextmanager
def transaction(state: str | pathlib.Path | None = None) -> Generator[Registry]:
    """One worker-thread knowledge transaction with a private projection, including nested registry.at calls.

    Prepare embeddings and sync segment files first. SQLite cannot roll back segment writes. No shared Registry
    lock spans BEGIN, durable writes, or commit; other threads continue to read committed identities. Failures
    roll back knowledge and discard the worker projection before it can answer another request.
    """
    from liberational import knowledge

    reg = local(state)
    path = str(knowledge_path(state))
    active = getattr(_worker_local, "active", {})
    previous = active.get(path)
    _worker_local.active = active
    try:
        with knowledge.at(path).transaction():
            active[path] = reg
            yield reg
    except BaseException:
        reg._reset()
        raise
    finally:
        if previous is None:
            active.pop(path, None)
        else:
            active[path] = previous


def knowledge_path(state: str | pathlib.Path | None = None) -> pathlib.Path:
    """Owner knowledge lives outside the broker catalog reset boundary."""
    return (pathlib.Path(state) if state is not None else cfg.path("paths.state")).resolve() / "knowledge.sqlite3"


def log_exists(state: str | pathlib.Path | None = None) -> bool:
    """Whether knowledge exists. A corrupt existing database remains an error when opened, never an empty roster."""
    return knowledge_path(state).exists()


def close_all() -> None:
    """Release cached registry projections; subsequent calls replay the durable SQLite records."""
    global _registry_epoch
    with _registries_lock:
        logs = [r.log for r in _registries.values()]
        _registries.clear()
        _registry_epoch += 1
    for log in logs:
        close = getattr(log, "close", None)
        if callable(close):
            close()


def owner_voice(state: str | pathlib.Path | None, sid: object) -> bool:
    """Whether voice `sid` is the owner's. This decides who gets the owner path (assistant/voice.py), so it fails
    closed: False whenever anything is missing, unreadable, or unclear.

    - The voice must be a voiceprint listen has handed out (state/voiceprints.json, listen/voices.py) and must not be
      a voice clone.
    - Its person by person_of must be the owner (`Registry.owner`).
    - The assertion that decides it must be an owner tag made about the owner's own person: the owner naming the
      voice on the dashboard or the CLI, "That voice is mine" on the dashboard, a voice tag, or enrolling the owner's
      reference clip. A spoken introduction, a badge, a cluster merge, a correction, or a model's match never makes a
      voice the owner's. Nor does merging or renaming someone into the owner: an owner tag about that other person
      stays theirs (`Registry.same_registration`). The explicit local This is me confirmation records new owner
      assertions together with its merge in one broker item (`merge_as_owner`). Nor does a name stored on a transcript line.
    - Any other owner-tier assertion still standing on the voice makes the answer unclear, so False."""
    from liberational.people import transcript_view

    if not isinstance(sid, int) or isinstance(sid, bool):
        return False  # a real voice id only: "1", " 1", "+1", or "١" read as 1 would key apart from voice 1
    state = pathlib.Path(state if state is not None else cfg.path("paths.state"))
    try:
        if not 1 <= sid < handed_out(state):
            return False
        if sid in transcript_view.clone_only_ids(state):
            return False
        if not log_exists(state):
            return False  # reading must not create the log
        reg = at(state)
        pid = reg.owner()
        if pid is None:
            return False
        decision = reg.person_of(voiceprint(sid))
        winner = decision.winner
        if decision.person != pid or winner is None or winner.basis is not Basis.OWNER_TAG:
            return False
        return all(reg.same_registration(a.person, pid) for a in decision.standing if tier(a) is Tier.OWNER)
    except Exception:
        return False


def handed_out(state: pathlib.Path) -> int:
    """The next voiceprint number listen would hand out (state/voiceprints.json, listen/voices.py): every number
    below it is a voice. 1 (no voice) when the file is missing or unreadable."""
    try:
        n = json.loads((state / "voiceprints.json").read_text())["next"]
    except OSError, ValueError, KeyError, TypeError:
        return 1
    return n if isinstance(n, int) and not isinstance(n, bool) and n >= 1 else 1


def voice_names(state: str | pathlib.Path | None, sids: Iterable[int]) -> dict[int, str | None]:
    """`Registry.voice_names` on the log under `state`, for a reader: every voice names nobody while there is no log
    yet, and reading never creates one."""
    state = pathlib.Path(state if state is not None else cfg.path("paths.state"))
    sids = [int(i) for i in sids]
    if not log_exists(state):
        return dict.fromkeys(sids)
    return at(state).voice_names(sids)


@contextlib.contextmanager
def safely(what: str) -> Generator[None]:
    """Run a registry read that has a fallback: a failure is logged and swallowed, and the caller goes on without it
    (a name not looked up, a suggestion left out). Writes use `recorded`, so a failed one is never reported as done."""
    try:
        yield
    except Exception as e:
        LOG.error(f"{what}: the event log couldn't be read: {e!r}")


class WriteFailed(RuntimeError):
    """A write the person registry couldn't record. What it was for wasn't done, or was undone, so the caller says
    so rather than reporting it done."""


@contextlib.contextmanager
def recorded(what: str) -> Generator[None]:
    """Run a registry write that must stand: a failure is logged and raised as WriteFailed."""
    try:
        yield
    except WriteFailed:
        raise
    except Exception as e:
        LOG.error(f"{what}: not recorded in the event log: {e!r}")
        raise WriteFailed(f"The person registry couldn't record this ({what}), so it wasn't saved; try again.") from e


# ---- backfill from today's stores ----
def _json(path: pathlib.Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except OSError, ValueError:
        return default


def _transcript(path: pathlib.Path) -> Iterator[dict]:
    try:
        with path.open() as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                if isinstance(r, dict):
                    yield r
    except OSError:
        return


def _face_names(db: pathlib.Path) -> list[str]:
    if not db.exists():
        return []
    import numpy as np

    # Our own file, written by vision/faces.save_db and read the way faces.load_db reads it.
    return [str(n) for n in np.load(db, allow_pickle=True)["names"]]


# Identity record kinds (people/identity.py). The owner's own spoken "that's Alice" (collar_intro, from the voice
# service or the CLI) is their word, the same as a confirm. Every other kind is a spoken introduction someone overheard.
IDENTITY_BASES = {"badge": Basis.NAME_BADGE, "collar": Basis.OWNER_TAG, "cli": Basis.OWNER_TAG}


def identity_basis(rec: dict) -> Basis:
    """The basis of an identity record: the owner's when they confirmed it by hand or said it themselves."""
    if rec.get("confirmed_by"):
        return Basis.OWNER_TAG
    return IDENTITY_BASES.get(rec.get("kind"), Basis.INTRODUCTION)


def backfill(
    state: str | pathlib.Path | None = None,
    faces_db: str | pathlib.Path | None = None,
    source: str = "registry-backfill",
) -> dict:
    """Give every person named in today's stores a person id, and record what today's stores already say as
    assertions: each named voice, each voice merged into one, each hand-labeled transcript line, and each saved face
    photo. Copy each phone number in contacts.json to a person who has none, and mark each name in people_added.json
    as added by hand. Skips what the log already has, so a rerun writes nothing new.

    Names come only from the stores that make up today's roster: every named voice counts as an owner tag, and
    enrolled faces are the gallery's rows. (The paragraph below describes the stores that remain.) A name still on the
    roster enrolled. Otherwise a rerun would bring a forgotten person back.

    Today's stores do not say how a voice got its name, so a voice named by an identity.jsonl record gets that
    record's basis and every other named voice counts as an owner tag. Returns counts of what was written."""
    state = pathlib.Path(state if state is not None else cfg.path("paths.state"))
    faces_db = pathlib.Path(faces_db) if faces_db else cfg.path("paths.faces") / "db.npz"
    reg = at(state)
    before = len(reg.people())
    # A voiceprint's name is the log's, so there's none to
    # copy. A line keeps the name it was hand-tagged with.
    labels: dict[int, str] = {}
    merged: dict[str, int] = {}
    lines: list[dict[str, Any]] = [  # only what a backfill needs from each line, since the transcript can be large
        {k: e.get(k) for k in ("ts_start", "device", "speaker", "speaker_id", "speaker_sim")}
        for e in _transcript(state / "transcript.jsonl")
        if e.get("ts_start") is not None
    ]
    # The name a line goes by today. A rename or merge rewrites no lines, so a line with a voice takes the voice's
    # current label (as the dashboard shows it) and a hand-tagged one follows its old name to the person who had it.
    current = {}
    for e in lines:
        sid = e["speaker_id"]
        sid = merged.get(str(sid), sid) if sid is not None else None
        if sid in labels:
            e["speaker"] = labels[sid]
        elif is_name(e["speaker"]):
            if e["speaker"] not in current:
                pid = reg.known_as(e["speaker"])
                current[e["speaker"]] = reg.person(pid).display_name if pid else e["speaker"]  # ty: ignore[unresolved-attribute]  # known_as names a registered person
            e["speaker"] = current[e["speaker"]]
    names: list[Any] = list(labels.values())
    names += _face_names(faces_db)
    added = [n for n in _json(state / "people_added.json", []) if is_name(n)]
    numbers = {  # "Alice (phone)" is Alice, as the People tab reads it
        re.sub(r"\s*\(phone\)$", "", k): v.strip() if isinstance(v, str) else ""
        for k, v in _json(state / "contacts.json", {}).items()
        if not k.startswith("_")
    }
    names += added + list(numbers)
    names += [e.get("speaker") for e in lines]
    names = list(dict.fromkeys(n for n in names if is_name(n)))
    for name in names:
        reg.ensure(name, source)
    by_basis = collections.defaultdict(list)
    for sid, label in labels.items():
        if is_name(label):
            by_basis[Basis.OWNER_TAG].append(Claim(voiceprint(sid), label))
    for old, into in merged.items():
        if is_name(labels.get(into)):
            by_basis[Basis.CLUSTER_MERGE].append(Claim(voiceprint(old), labels[into]))
    for e in lines:  # a line a person labeled: no similarity score and a name (see vision/sightings.hand_labeled)
        # A label is never written onto a line, so a line the log already knows keeps what the log says:
        # a rerun must not put back a name the owner changed since.
        if e.get("speaker_sim") is None and is_name(e.get("speaker")) and not reg.has_claims(utterance_of(e)):
            by_basis[Basis.OWNER_TAG].append(Claim(utterance_of(e), e["speaker"]))
    for key, name in {photo_folder(n): n for n in names}.items():  # "Alice" and "alice" share a folder
        folder = faces_db.parent / key
        for f in sorted(folder.glob("*.jpg")) if folder.is_dir() else ():
            if re.fullmatch(r"\d{10,16}", f.stem):
                by_basis[Basis.OWNER_TAG].append(Claim(photo(f.stem), name))
    written = sum(len(reg.claim_many(claims, basis, source)) for basis, claims in by_basis.items())
    phones = marked = 0
    for name, number in numbers.items():
        if number and is_name(name) and _field(reg, name, "phone") is None:  # a number set since is kept
            reg.contact(name, source, phone=number, fill=True)
            phones += 1
    for name in added:
        if not _field(reg, name, "added"):
            reg.contact(name, source, added=True)
            marked += 1
    return {"people": len(reg.people()) - before, "assertions": written, "phones": phones, "added": marked}


MIGRATE_SECTION = "Voice names move to the event log"


def _field(reg: Registry, name: str, field: str) -> object:
    """A field of the person called `name`, or None when nobody has that name."""
    pid = reg.person_id(name)
    person = reg.person(pid) if pid else None
    return getattr(person, field) if person else None


def _phone(state: str | None, name: str, number: str | None, clear: bool) -> int:
    """`pack-registry phone`: show, set, or clear the number of someone the registry knows. Returns the exit code."""
    from liberational.people import contacts

    book = contacts.Book.at(state)
    pid = at(state).person_id(name)
    if pid is None:
        print(f"Nobody is called {name}. Run pack-registry backfill first, or add them on the People tab.")
        return 1
    if number is not None and not contacts.PHONE.fullmatch(number):
        print("A number looks like +14155550123.")
        return 1
    if clear or number is not None:
        book.set_phone(name, "" if clear else number or "", "cli")
    print(f"{name}: {book.number(name) or 'no number'}")
    return 0


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="The person registry over the event log (docs/DOMAIN-MODEL.md).")
    ap.add_argument("--state-dir", default=None, help="state directory (paths.state by default)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("backfill", help="give every name in today's stores a person id (safe to rerun)")
    sub.add_parser("people", help="list people, their ids, their role and organization, and their phone number")
    phone = sub.add_parser("phone", help="show, set, or clear a person's phone number (contacts.json follows)")
    phone.add_argument("name")
    phone.add_argument("number", nargs="?", help="such as +14155550123; leave out to show the number")
    phone.add_argument("--clear", action="store_true", help="remove the person's number")
    who = sub.add_parser("who", help="who a subject is, with the assertions behind it")
    who.add_argument("subject", help="such as speaker:4, utterance:elgato-wave-3@1761234567.123, or sighting:812")
    a = ap.parse_args(argv)
    if a.cmd == "backfill":
        print(json.dumps(backfill(a.state_dir)))
        return
    if a.cmd == "phone":
        sys.exit(_phone(a.state_dir, a.name, a.number, a.clear))
    reg = at(a.state_dir)
    if a.cmd == "people":
        for p in reg.people():
            said = ", ".join(x for x in (p.role, p.organization) if x)
            print(
                f"{p.id}  {p.display_name}"
                + (f"  [{said}]" if said else "")
                + (f"  phone {p.phone}" if p.phone else "")
                + (f"  (was {', '.join(p.aliases)})" if p.aliases else "")
            )
        return
    d = reg.person_of(subject_from_id(a.subject))
    person = reg.person(d.person) if d.person else None
    print(f"{a.subject}: {person.display_name if person else 'nobody'}")
    for x in (*d.standing, *d.retractions):
        named = reg.person(x.person)
        print(
            f"  {'retracted' if x.retracted else x.basis:<15} {named.display_name if named else '?':<16} "
            f"conf {x.confidence:.2f}  {kernel.to_datetime(x.asserted_at):%Y-%m-%d %H:%M:%S}Z  by {x.source}"
        )


if __name__ == "__main__":
    main()
