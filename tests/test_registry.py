"""The person registry and person_of (docs/DOMAIN-MODEL.md): which assertion wins ("Identity precedence"), the
projection rebuilt from the log, forgetting ("Forgetting"), the backfill from today's stores, and each dual write
beside today's label writes. Everything runs on temp dirs, with made-up people."""

import json
import pathlib
import sqlite3
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import create_autospec

import numpy as np
import pytest

from liberational import config as cfg
from liberational import knowledge
from liberational.dashboard import known_voices
from liberational.domain import kernel
from liberational.domain.eventlog import Selector
from liberational.domain.identity import Basis, Decision, IdentityAssertion, SubjectKind, SubjectRef, decide
from liberational.domain.kernel import Event, EventType, PersonId, Producer
from liberational.knowledge_eventlog import SQLiteEventLog
from liberational.people import identity, registry
from liberational.vision import faces
from speech_fake import fake
from voices import write_voices

VOICE = SubjectRef(SubjectKind.VOICEPRINT, "speaker:1")
MODEL = Producer("test-model", "1")
NOTHING = {"people": 0, "assertions": 0, "phones": 0, "added": 0}  # a backfill rerun writes nothing


def assertion(id, person, basis, at, confidence=1.0, **kw):
    producer = MODEL if basis in (Basis.EMBEDDING_MATCH, Basis.BACKFILL_MATCH) else None
    return IdentityAssertion(id, VOICE, person, basis, confidence, at, "test", producer=producer, **kw)


def name_of(reg: registry.Registry, pid: PersonId | None) -> str | None:
    person = reg.person(pid) if pid else None
    return person.display_name if person else None


def who(reg: registry.Registry, subject: SubjectRef, via: SubjectRef | None = None) -> str | None:
    """The name of whoever `subject` is (through `via` when it has no assertion of its own), or None."""
    return name_of(reg, reg.person_of(subject, via=via).person)


def winner(d: Decision) -> IdentityAssertion:
    """The assertion that decides `d`, which the test expects to have one."""
    assert d.winner is not None
    return d.winner


_logs: dict[pathlib.Path, SQLiteEventLog] = {}


def make_log(state: pathlib.Path) -> SQLiteEventLog:
    """A real SQLite event log in this state directory, also used by fresh Registry projections."""
    path = (state / "knowledge.sqlite3").resolve()
    if path not in _logs:
        _logs[path] = SQLiteEventLog(path)
    return _logs[path]


def rebuilt(state: pathlib.Path) -> registry.Registry:
    """A registry projected from the log alone, as a restart or another process sees it."""
    return registry.Registry(make_log(state))


def events(state, stream=None):
    log = make_log(state)
    streams = [stream] if stream else [registry.PEOPLE, registry.ASSERTIONS, registry.SIGHTINGS]
    return [s.event for st in streams for s in log.read(st)]


@pytest.fixture(autouse=True)
def backend(tmp_path):
    """Every registry entry point, including face-gallery writes, uses the same temporary install."""
    with cfg.override({"paths": {"state": str(tmp_path)}}):
        yield
    registry.close_all()
    for log in _logs.values():
        log.close()
    _logs.clear()


def types(state, stream=None):
    return [str(e.type) for e in events(state, stream)]


@pytest.fixture
def reg(tmp_path):
    return registry.at(tmp_path)


@pytest.fixture
def clock(monkeypatch):
    """The time the log records events at, set by the test."""
    c = SimpleNamespace(t=100.0)
    monkeypatch.setattr(kernel, "now", lambda: c.t)
    return c


# ---- identity precedence: retraction, then owner tag, then confirmed, then system match by confidence; newer wins ties ----
def test_owner_beats_confirmed_beats_system():
    system = assertion("a1", "p-sys", Basis.EMBEDDING_MATCH, 30.0, 0.99)
    confirmed = assertion("a2", "p-conf", Basis.INTRODUCTION, 20.0)
    owner = assertion("a3", "p-owner", Basis.OWNER_TAG, 10.0)
    d = decide([system, confirmed, owner])
    assert d.person == "p-owner"
    assert [a.id for a in d.standing] == ["a3", "a2", "a1"]  # the assertions behind the answer, best first
    assert decide([system, confirmed]).person == "p-conf"
    assert decide([assertion("c", "p-fix", Basis.CORRECTION, 1.0), confirmed]).person == "p-fix"


def test_system_matches_compare_by_confidence_then_time():
    old_strong = assertion("a1", "p1", Basis.EMBEDDING_MATCH, 10.0, 0.9)
    new_weak = assertion("a2", "p2", Basis.EMBEDDING_MATCH, 20.0, 0.6)
    assert decide([old_strong, new_weak]).person == "p1"
    tie = assertion("a3", "p3", Basis.EMBEDDING_MATCH, 30.0, 0.9)
    assert decide([old_strong, new_weak, tie]).person == "p3"


def test_a_rematch_outranks_older_system_matches_but_not_the_owner():
    alice = assertion("a1", "p-alice", Basis.BACKFILL_MATCH, 10.0, 0.8)
    carol = assertion("a2", "p-carol", Basis.EMBEDDING_MATCH, 12.0, 0.75)  # an old match standing beside it
    dana = assertion("a3", "p-dana", Basis.BACKFILL_MATCH, 20.0, 0.62)  # the rematch, against a fixed gallery
    d = decide([alice, carol, dana])
    assert d.person == "p-dana"
    assert [a.id for a in d.standing] == ["a3"]
    later_live = assertion("a4", "p-erin", Basis.EMBEDDING_MATCH, 30.0, 0.9)  # after the rematch, confidence again
    assert decide([alice, carol, dana, later_live]).person == "p-erin"
    assert decide([assertion("o1", "p-alice", Basis.OWNER_TAG, 5.0), alice, dana]).person == "p-alice"
    assert decide([assertion("i1", "p-alice", Basis.INTRODUCTION, 5.0), alice, dana]).person == "p-alice"


def test_a_superseded_or_passed_in_rematch_still_sets_the_cutoff():
    carol = assertion("a1", "p-carol", Basis.EMBEDDING_MATCH, 10.0, 0.8)
    dana = assertion("a2", "p-dana", Basis.BACKFILL_MATCH, 20.0, 0.6)
    erin = assertion("a3", "p-erin", Basis.EMBEDDING_MATCH, 30.0, 0.7, supersedes="a2")
    assert decide([carol, dana, erin]).person == "p-erin"
    assert decide([carol, erin], rematched_at=20.0).person == "p-erin"  # a cache that dropped a2
    assert decide([carol, erin]).person == "p-carol"  # without it, nothing marks the gallery fix


def test_newer_wins_a_tie_between_owner_tags():
    newer, older = assertion("a1", "p1", Basis.OWNER_TAG, 20.0), assertion("a2", "p2", Basis.OWNER_TAG, 10.0)
    assert decide([newer, older]).person == "p1"


def test_a_retraction_beats_everything_until_the_owner_says_so_again():
    tag = assertion("a1", "p1", Basis.OWNER_TAG, 10.0)
    gone = assertion("r1", "p1", Basis.CORRECTION, 20.0, supersedes="a1", retracted=True)
    assert decide([tag, gone]).person is None
    later_match = assertion("a2", "p1", Basis.BACKFILL_MATCH, 30.0, 0.95)
    earlier_owner = assertion("a3", "p1", Basis.OWNER_TAG, 15.0)
    d = decide([tag, gone, later_match, earlier_owner])
    assert d.person is None
    assert [r.id for r in d.retractions] == ["r1"]
    assert decide([tag, gone, assertion("a4", "p1", Basis.OWNER_TAG, 40.0)]).person == "p1"
    other = assertion("a5", "p2", Basis.BACKFILL_MATCH, 30.0, 0.7)  # a retraction withdraws one person only
    assert decide([tag, gone, other]).person == "p2"


def test_a_superseded_retraction_no_longer_withdraws():
    first = assertion("a1", "p1", Basis.INTRODUCTION, 10.0)
    gone = assertion("r1", "p1", Basis.CORRECTION, 20.0, supersedes="a1", retracted=True)
    back = assertion("a2", "p1", Basis.INTRODUCTION, 30.0, supersedes="r1")
    d = decide([first, gone, back])
    assert (d.person, d.retractions) == ("p1", ())
    undo = assertion("r2", "p1", Basis.CORRECTION, 30.0, supersedes="r1", retracted=True)
    assert decide([first, gone, undo]).retractions == ()  # retracting a retraction undoes it


def test_superseded_assertions_and_forgotten_people_do_not_count():
    first = assertion("a1", "p1", Basis.OWNER_TAG, 10.0)
    second = assertion("a2", "p2", Basis.BACKFILL_MATCH, 20.0, 0.5, supersedes="a1")
    assert decide([first, second]).person == "p2"
    p1, p2, p9 = PersonId("p1"), PersonId("p2"), PersonId("p9")
    forgotten: dict[PersonId, PersonId | None] = {p1: p1, p2: None}  # p2 was forgotten
    assert decide([first, second], forgotten.get).person is None
    merged: dict[PersonId, PersonId | None] = {p1: p9, p2: p2}  # p1 was merged into p9
    assert decide([first], merged.get).person == "p9"


def test_assertions_round_trip_through_a_payload():
    a = assertion("a1", "p1", Basis.BACKFILL_MATCH, 10.0, 0.7, supersedes="a0")
    assert IdentityAssertion.from_dict(json.loads(json.dumps(a.to_dict()))) == a


# ---- the registry ----
def test_a_name_gets_one_id_that_never_changes(tmp_path, reg):
    alice = reg.ensure("Alice", "test")
    assert reg.ensure("  alice ", "test") == alice
    assert reg.rename("Alice", "Alicia", "test") == alice
    assert (reg.person_id("Alicia"), reg.person_id("Alice")) == (alice, None)
    assert reg.person(alice).aliases == ("Alice",)
    assert types(tmp_path) == ["PersonRegistered", "PersonAttributesChanged"]
    assert rebuilt(tmp_path).person_id("alicia") == alice
    with pytest.raises(ValueError, match="not a person's name"):
        reg.ensure("Speaker 4", "test")


def test_two_registrations_of_one_name_resolve_to_the_first(tmp_path):
    first = PersonId(kernel.new_id())
    payload = {"person": first, "display_name": "Dana"}
    theirs = [Event.new(EventType.PERSON_REGISTERED, stream=registry.PEOPLE, source="other", payload=payload)]

    class RacingLog(SQLiteEventLog):
        """Another process registers Dana between this one's lookup and its write."""

        def append(self, events):
            if theirs:
                super().append([theirs.pop()])
            return super().append(events)

    reg = registry.Registry(RacingLog(tmp_path / "knowledge.sqlite3"))
    assert reg.ensure("Dana", "test") == first
    assert types(tmp_path).count("PersonRegistered") == 2
    late = next(e.payload["person"] for e in events(tmp_path) if e.payload["person"] != first)
    assert rebuilt(tmp_path).person(late).id == first  # ty: ignore[unresolved-attribute]  # merged into first
    assert [p.id for p in rebuilt(tmp_path).people()] == [first]


def test_person_of_shows_the_assertions_behind_it(tmp_path, reg):
    reg.ensure("Dana", "test")  # a match names someone already known (claim_many)
    reg.claim(VOICE, "Alice", Basis.INTRODUCTION, "identity", confidence=0.9)
    reg.claim(VOICE, "Dana", Basis.BACKFILL_MATCH, "backfill", confidence=0.99, producer=MODEL)
    d = reg.person_of(VOICE)
    assert name_of(reg, d.person) == "Alice"
    assert [a.basis for a in d.standing] == [Basis.INTRODUCTION, Basis.BACKFILL_MATCH]
    assert rebuilt(tmp_path).person_of(VOICE) == d
    reg.claim(VOICE, "Dana", Basis.OWNER_TAG, "dashboard")  # the owner overrides, superseding the intro
    d = reg.person_of(VOICE)
    assert name_of(reg, d.person) == "Dana"
    assert [a.basis for a in d.standing] == [Basis.OWNER_TAG, Basis.BACKFILL_MATCH]


def test_a_decision_follows_a_claim_another_process_wrote(tmp_path, reg):
    reg.claim(VOICE, "Alice", Basis.INTRODUCTION, "identity")
    assert who(reg, VOICE) == "Alice"
    rebuilt(tmp_path).claim(VOICE, "Dana", Basis.OWNER_TAG, "dashboard")  # another process, the same log
    assert who(reg, VOICE) == "Dana"


def test_asking_again_between_events_decides_once(reg, monkeypatch):
    reg.claim(VOICE, "Alice", Basis.INTRODUCTION, "identity")
    counted = create_autospec(decide, side_effect=decide)
    monkeypatch.setattr(registry, "decide", counted)
    first = reg.person_of(VOICE)
    decided = counted.call_count
    assert reg.person_of(VOICE) == first
    assert reg.said([(VOICE, None)], refresh=False)[0].person == first.person
    assert counted.call_count == decided, "nothing applied since: the same decision, not decided again"
    reg.claim(VOICE, "Dana", Basis.OWNER_TAG, "dashboard")
    assert name_of(reg, reg.person_of(VOICE).person) == "Dana"


def test_claiming_what_already_stands_writes_nothing(tmp_path, reg):
    assert reg.claim(VOICE, "Alice", Basis.OWNER_TAG, "test") is not None
    assert reg.claim(VOICE, "alice", Basis.OWNER_TAG, "test") is None
    assert reg.claim(VOICE, "Alice", Basis.CLUSTER_MERGE, "test") is None  # weaker than what stands
    assert reg.claim(VOICE, "Speaker 3", Basis.OWNER_TAG, "test") is None  # an automatic label
    assert types(tmp_path, registry.ASSERTIONS) == ["IdentityAsserted"]


@pytest.mark.parametrize(
    "label",
    ["Speaker 4", "speaker 4", "SPEAKER 4", "face 2", "caller +14155550123"],
    ids=["canonical", "lower", "upper", "lower-face", "lower-caller"],
)
def test_an_automatic_label_names_nobody_in_any_capitals(label):
    assert not registry.is_name(label)


@pytest.mark.parametrize("confidence", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_a_match_without_a_real_confidence_is_no_match(tmp_path, reg, confidence):
    # A zero embedding gives a NaN similarity, which a clamp to [0, 1] turned into a full-confidence match.
    reg.ensure("Alice", "test")
    s = registry.sighting(1)
    made = reg.claim(s, "Alice", Basis.EMBEDDING_MATCH, "identity", confidence=confidence, producer=MODEL)
    assert made is None
    assert reg.person_of(s).person is None
    assert "IdentityAsserted" not in types(tmp_path)


def test_a_weaker_system_match_never_supersedes_a_stronger_one(tmp_path, reg):
    s = registry.sighting(1)

    def match(name, confidence):
        return reg.claim(s, name, Basis.EMBEDDING_MATCH, "identity", confidence=confidence, producer=MODEL)

    for name in ("Alice", "Dana"):
        reg.ensure(name, "test")  # a match names someone already known (claim_many)
    match("Alice", 0.95)
    assert match("Dana", 0.60) is not None
    d = reg.person_of(s)
    assert name_of(reg, d.person) == "Alice"
    assert [a.confidence for a in d.standing] == [0.95, 0.60]  # the weaker one stands beside it
    assert rebuilt(tmp_path).person_of(s) == d
    assert match("Dana", 0.60) is None  # already stands that strongly
    match("Dana", 0.97)  # stronger: it supersedes Alice's
    assert [a.confidence for a in reg.person_of(s).standing] == [0.97, 0.60]
    assert match("Dana", 0.99) is not None  # stronger evidence for the same person still counts
    assert match("Dana", 0.50) is None
    assert [a.confidence for a in reg.person_of(s).standing] == [0.99, 0.60]


def test_a_rematch_wins_over_older_system_matches(tmp_path, reg):
    s = registry.sighting(2)

    def claim(name, basis, confidence=1.0):
        producer = MODEL if basis in (Basis.EMBEDDING_MATCH, Basis.BACKFILL_MATCH) else None
        return reg.claim(s, name, basis, "test", confidence=confidence, producer=producer)

    for name in ("Alice", "Dana"):
        reg.ensure(name, "test")  # a match names someone already known (claim_many)
    claim("Alice", Basis.BACKFILL_MATCH, 0.80)
    claim("Dana", Basis.EMBEDDING_MATCH, 0.70)  # weaker: stands beside Alice
    assert who(reg, s) == "Alice"
    assert claim("Dana", Basis.BACKFILL_MATCH, 0.62) is not None  # the rematch is written though Dana stood
    assert who(reg, s) == "Dana"
    assert rebuilt(tmp_path).person_of(s) == reg.person_of(s)
    assert claim("Dana", Basis.BACKFILL_MATCH, 0.55) is None  # Dana already wins: nothing to write
    claim("Alice", Basis.OWNER_TAG)
    claim("Dana", Basis.BACKFILL_MATCH, 0.99)  # a rematch never beats the owner
    assert who(reg, s) == "Alice"


def test_the_rematch_cutoff_outlives_the_rematch(tmp_path, reg):
    s = registry.sighting(3)

    def live(name, confidence):
        return reg.claim(s, name, Basis.EMBEDDING_MATCH, "identity", confidence=confidence, producer=MODEL)

    for name in ("Alice", "Carol", "Dana", "Erin"):
        reg.ensure(name, "test")  # a match names someone already known (claim_many)
    live("Alice", 0.90)
    live("Carol", 0.80)  # stands beside Alice, from the old gallery
    reg.claim(s, "Dana", Basis.BACKFILL_MATCH, "backfill", confidence=0.60, producer=MODEL)
    live("Erin", 0.70)  # supersedes the rematch, which drops out of the cache
    assert who(reg, s) == who(rebuilt(tmp_path), s) == "Erin"  # not Carol: her match predates the gallery fix
    reg.retract(s, "dashboard")
    assert "Carol" not in (who(reg, s), who(rebuilt(tmp_path), s))


def test_a_line_falls_back_to_its_voice_unless_withdrawn(reg):
    reg.claim(VOICE, "Alice", Basis.OWNER_TAG, "test")
    line = registry.utterance(1700000000.1234)
    assert line.id == "utterance:1700000000.123"
    assert (who(reg, line, via=VOICE), who(reg, line)) == ("Alice", None)
    assert len(reg.retract(line, "dashboard", via=VOICE)) == 1  # unlabeled from its voice's person
    assert who(reg, line, via=VOICE) is None
    assert who(reg, VOICE) == "Alice"  # the voice itself is still Alice's


def test_merged_people_answer_as_the_survivor(tmp_path, reg):
    reg.claim(VOICE, "Dana", Basis.OWNER_TAG, "test")
    alice = reg.ensure("Alice", "test")
    assert reg.merge("Dana", "Alice", "test") == alice
    assert reg.person_of(VOICE).person == rebuilt(tmp_path).person_of(VOICE).person == alice
    assert reg.person_id("Dana") is None


def test_a_name_written_at_a_time_is_whoever_had_it_then(tmp_path, reg, clock):
    dana = reg.ensure("Dana", "test", at=900.0)  # dated to old evidence, but given the name at 100
    reg.rename("Dana", "Dee", "dashboard")
    reg.merge("Dee", "Carl", "dashboard")
    carl = reg.person_id("Carl")
    clock.t = 200.0
    newcomer = reg.ensure("Dana", "dashboard")  # someone new takes the old name later
    assert dana != carl
    assert reg.known_as("Dana") == newcomer
    assert reg.known_as("Dana", at=150.0) == carl  # written while Dana was the one now merged in Carl
    assert reg.known_as("Dana", at=50.0) == carl  # before anyone had the name: its first holder
    assert reg.known_as("Dana", at=250.0) == newcomer
    assert reg.known_as("Dee", at=150.0) == carl
    clock.t = 300.0
    reg.forget("Carl", "dashboard")
    assert reg.known_as("Dana", at=150.0) is None  # forgotten, and never handed to the newcomer
    assert_dates_names_alike(tmp_path, reg, ("Dana", "Dee", "Carl"), (50.0, 100.0, 150.0, 200.0, 250.0))


def test_strictly_a_name_taken_away_names_nobody_until_someone_takes_it(tmp_path, reg, clock):
    """`known_as(name, at, strict=True)`, for a tag: after a rename or a merge took the name away, nobody was
    called that. Without `strict`, a label written late under the old name still finds its last holder."""
    dana = reg.ensure("Dana", "test")
    carl = reg.ensure("Carl", "test")
    clock.t = 200.0
    reg.rename("Dana", "Dee", "dashboard")
    clock.t = 300.0
    reg.merge("Carl", "Dee", "dashboard")
    clock.t = 400.0
    newcomer = reg.ensure("Dana", "dashboard")
    assert reg.known_as("Dana", 150.0, strict=True) == dana
    assert reg.known_as("Dana", 250.0, strict=True) is None
    assert reg.known_as("Dana", 250.0) == dana
    assert reg.known_as("Dana", 450.0, strict=True) == newcomer
    assert reg.known_as("Carl", 250.0, strict=True) == dana  # Carl, merged into Dee since
    assert reg.known_as("Carl", 350.0, strict=True) is None
    assert reg.known_as("Dee", 350.0, strict=True) == dana
    assert carl != dana
    assert_dates_names_alike(tmp_path, reg, ("Dana", "Dee", "Carl"), (50.0, 150.0, 250.0, 350.0, 450.0))


def test_a_rebuilt_registry_dates_names_as_the_live_one_after_a_forget(tmp_path, reg, clock):
    """Rule 8 for `known_as(name, at)`: a forgotten person's time with a name stays theirs after a restart, and
    never goes to whoever had the name before them or takes it after them."""
    reg.ensure("Ann", "test")
    clock.t = 150.0
    reg.rename("Ann", "Zed", "dashboard")
    clock.t = 200.0
    reg.ensure("Ann", "test")  # someone new takes the name, and is forgotten
    clock.t = 250.0
    reg.forget("Ann", "dashboard")
    clock.t = 300.0
    reg.ensure("Bea", "test")  # the first Bea is forgotten, and someone new takes her name
    clock.t = 350.0
    reg.forget("Bea", "dashboard")
    clock.t = 400.0
    bea = reg.ensure("Bea", "test")
    assert reg.known_as("Ann", at=250.0) is None
    assert reg.known_as("Bea", at=350.0) is None
    assert reg.known_as("Bea", at=450.0) == bea
    times = (50.0, 100.0, 125.0, 150.0, 175.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0)
    assert_dates_names_alike(tmp_path, reg, ("Ann", "Zed", "Bea"), times)
    forgotten = [e for e in events(tmp_path, registry.PEOPLE) if e.type is EventType.PERSON_FORGOTTEN]
    assert len(forgotten) == 2
    assert "ann" not in json.dumps([e.to_dict() for e in forgotten]).casefold()  # never the name itself


def assert_dates_names_alike(state, reg, names, times):
    """A registry rebuilt from the log answers `known_as` (strict or not) as the live one, for each name and time."""
    fresh = rebuilt(state)
    cases = [(n, t, strict) for n in names for t in times for strict in (False, True)]
    assert {c: fresh.known_as(c[0], c[1], strict=c[2]) for c in cases} == {
        c: reg.known_as(c[0], c[1], strict=c[2]) for c in cases
    }


# ---- forgetting: PersonForgotten, and the person's events purged from the log ----
def test_forgetting_purges_their_events_and_keeps_others(tmp_path, reg):
    reg.claim(VOICE, "Dana", Basis.OWNER_TAG, "test")
    reg.claim(registry.utterance(5.0), "Dana Cole", Basis.OWNER_TAG, "test")
    reg.merge("Dana Cole", "Dana", "test")  # someone merged into her goes with her
    reg.claim(registry.sighting(7), "Dana", Basis.BACKFILL_MATCH, "backfill", confidence=0.8, producer=MODEL)
    reg.claim(registry.voiceprint(2), "Alice", Basis.OWNER_TAG, "test")
    dana = reg.person_id("Dana")
    assert reg.forget("dana", "dashboard") > 0
    left = events(tmp_path)
    # Alice's, the record that forgetting happened, and Alice's voice
    assert [str(e.type) for e in left] == ["PersonRegistered", "PersonForgotten", "IdentityAsserted"]
    assert "Dana" not in json.dumps([e.to_dict() for e in left])
    forgotten = next(e for e in left if e.type is EventType.PERSON_FORGOTTEN)
    assert (forgotten.subject, forgotten.payload["person"]) == (None, dana)
    # Dana and the person merged into her go in one purge, with one record.
    assert [(p.reason, p.scope.subject) for p in make_log(tmp_path).purges()] == [("PersonForgotten", True)]

    def answers(r):
        return r.person_id("Dana"), who(r, VOICE), who(r, registry.sighting(7)), who(r, registry.voiceprint(2))

    assert answers(reg) == answers(rebuilt(tmp_path)) == (None, None, None, "Alice")
    assert reg.ensure("Dana", "test") != dana  # a new Dana is someone new
    assert reg.forget("Nobody", "dashboard") == 0


def assert_rebuilt_agrees(state, reg, voices=(), lines=(), sightings=()):
    """A registry rebuilt from the log alone (a restart, or another process) answers like the live one."""
    fresh = rebuilt(state)
    subjects = [*map(registry.voiceprint, voices), *map(registry.sighting, sightings)]
    assert [fresh.person_of(s).person for s in subjects] == [reg.person_of(s).person for s in subjects]
    assert fresh.said(lines) == reg.said(lines)
    return fresh


def test_forgetting_whoever_superseded_someone_keeps_the_old_assertion_dropped(tmp_path, reg):
    v7 = registry.voiceprint(7)
    reg.claim(v7, "Dana", Basis.OWNER_TAG, "ambient")
    reg.claim(v7, "Alice", Basis.OWNER_TAG, "ambient")  # the owner corrects the voice: Dana's is superseded
    reg.retract(v7, "ambient")  # forgetting voice 7, as People: Delete does
    v8 = registry.voiceprint(8)
    reg.claim(v8, "Bob", Basis.INTRODUCTION, "identity")
    reg.claim(v8, "Cara", Basis.INTRODUCTION, "identity")  # supersedes Bob's
    reg.claim(v8, "Alice", Basis.NAME_BADGE, "identity")  # supersedes Cara's, which kept Bob's superseded
    reg.forget("Alice", "dashboard")
    assert reg.person_of(v7).person is None
    fresh = assert_rebuilt_agrees(tmp_path, reg, voices=[7, 8], lines=[(3000.0, 7)])
    assert fresh.person_of(v7).person is None  # not Dana
    assert fresh.person_of(v8).person is None  # neither Cara nor Bob


def test_forgetting_whoever_relabeled_a_line_keeps_the_rematch_and_the_label(tmp_path, reg):
    for name in ("Dana", "Cara", "Carol"):
        reg.ensure(name, "test")
    line = registry.utterance(3000.0)
    reg.claim(line, "Dana", Basis.BACKFILL_MATCH, "backfill", confidence=0.7, producer=MODEL)
    reg.claim(line, "Alice", Basis.OWNER_TAG, "dashboard")  # the owner relabels it
    other = registry.utterance(3001.0)
    reg.claim(other, "Cara", Basis.INTRODUCTION, "identity")
    reg.claim(other, "Alice", Basis.CORRECTION, "dashboard")
    reg.detach(3001.0, "unknown", "dashboard")  # then calls it nobody's
    s = registry.sighting(9)
    reg.claim(s, "Carol", Basis.EMBEDDING_MATCH, "identity", confidence=0.9, producer=MODEL)
    reg.claim(s, "Dana", Basis.EMBEDDING_MATCH, "identity", confidence=0.6, producer=MODEL, at=1.0)
    reg.claim(s, "Alice", Basis.BACKFILL_MATCH, "backfill", confidence=0.8, producer=MODEL)  # drops both
    reg.forget("Alice", "dashboard")
    lines = [(3000.0, None), (3001.0, None)]
    fresh = assert_rebuilt_agrees(tmp_path, reg, lines=lines, sightings=[9])
    said = fresh.said(lines)
    assert [x.person for x in said] == [None, None]
    assert (said[1].label, said[1].detached) == ("unknown", True)
    assert fresh.person_of(s).person is None  # the rematch cutoff outlived the rematch's purge
    assert [e for e in events(tmp_path) if "Alice" in json.dumps(e.to_dict())] == []  # only ids and times are kept


def test_known_as_falls_back_to_the_earlier_holder_of_a_forgotten_name(tmp_path, reg):
    reg.claim(registry.voiceprint(1), "Alice", Basis.OWNER_TAG, "dashboard")
    reg.claim(registry.voiceprint(2), "Dana", Basis.OWNER_TAG, "dashboard")
    dana = reg.person_id("Dana")
    reg.rename("Alice", "Dana", "dashboard")  # Alice was Dana all along: a merge
    reg.claim(registry.voiceprint(3), "Alice", Basis.OWNER_TAG, "dashboard")  # someone new takes the name
    assert reg.known_as("Alice") != dana
    assert reg.names("Alice") == {"alice"}
    assert reg.names("Dana") == {"dana"}  # someone else has "Alice" now
    reg.forget("Alice", "dashboard")
    fresh = rebuilt(tmp_path)
    assert fresh.known_as("Alice") == reg.known_as("Alice") == dana  # the live registry agrees, before any restart
    assert reg.names("Dana") == fresh.names("Dana")


def test_a_late_model_match_never_brings_a_forgotten_person_back(tmp_path, reg):
    reg.claim(registry.voiceprint(3), "Dana", Basis.OWNER_TAG, "ambient")
    backfill = rebuilt(tmp_path)  # another process, mid-pass
    backfill.person_id("Dana")
    reg.forget("Dana", "dashboard")
    line, s = registry.utterance(3000.0), registry.sighting(4)
    made = backfill.claim_many([registry.Claim(line, "Dana", 0.8)], Basis.BACKFILL_MATCH, "backfill", producer=MODEL)
    made += backfill.claim_many([registry.Claim(s, "Dana", 0.8)], Basis.EMBEDDING_MATCH, "identity", producer=MODEL)
    assert made == []
    assert [(r.person_id("Dana"), r.people()) for r in (reg, backfill)] == [(None, [])] * 2
    assert reg.claim(line, "Dana", Basis.OWNER_TAG, "dashboard") is not None  # the owner can name someone new
    assert reg.person_id("Dana") is not None


@pytest.mark.parametrize(
    ("requested_at", "forgotten"),
    [(None, True), (199.0, True), (200.0001, True), (200.0002, True), (200.0003, False), (201.0, False)],
)
def test_forgotten_since_distinguishes_stale_tags_from_fresh_teaching(tmp_path, reg, clock, requested_at, forgotten):
    reg.ensure("Alice", "test")
    observer = rebuilt(tmp_path)
    observer.refresh()
    clock.t = 200.0002
    reg.forget("Alice", "test")
    assert observer.forgotten_since(" ALICE ", requested_at) is forgotten
    assert rebuilt(tmp_path).forgotten_since("Alice", requested_at) is forgotten
    assert observer.forgotten_since("Dana", requested_at) is False
    clock.t = 300.0
    reg.ensure("Alice", "test")
    assert observer.forgotten_since("Alice", requested_at) is forgotten


def test_forgotten_since_checks_renamed_alias_from_cached_tombstone(tmp_path, reg, clock):
    reg.ensure("Alice", "test")
    clock.t = 150.0
    reg.rename("Alice", "Dana", "test")
    clock.t = 200.0
    reg.forget("Dana", "test")
    fresh = rebuilt(tmp_path)
    assert fresh.forgotten_since("Alice", 199.0)
    assert fresh.forgotten_since("Dana", 199.0)
    assert not fresh.forgotten_since("Alice", 201.0)


# ---- registry.backfill: every name in today's stores gets an id ----
@pytest.fixture
def stores(tmp_path):
    """Today's stores, with a face database beside the state directory. Returns the face database's path."""
    db = tmp_path / "faces" / "db.npz"
    db.parent.mkdir()
    np.savez(db, names=np.array(["Alice", "Erin", "Face 3"]), embs=np.zeros((3, 4), np.float32))
    write_voices(tmp_path, {1: None, 2: None})  # voiceprints with no name in the log
    (tmp_path / "people_added.json").write_text(json.dumps(["Gus", "alice"]))
    (tmp_path / "contacts.json").write_text(json.dumps({"Hana": "+14155550123", "_note": "x"}))
    lines = [
        {"ts_start": 10.0, "speaker": "Alice", "speaker_id": 1, "speaker_sim": 0.8},
        {"ts_start": 11.0, "speaker": "Ivy", "speaker_id": None, "speaker_sim": None, "tagged": True},
        {"ts_start": 12.0, "speaker": "unknown", "speaker_id": None, "speaker_sim": None},
        {"ts_start": 13.0, "speaker": "Alice", "speaker_id": None, "speaker_sim": None},  # tagged by hand, no voice
    ]
    (tmp_path / "transcript.jsonl").write_text("".join(json.dumps(e) + "\n" for e in lines))
    fake().write(lines)  # and on the speech log, for the dashboard
    return db


def test_every_name_gets_an_id_and_a_rerun_writes_nothing(tmp_path, stores):
    # the two hand-tagged lines (Ivy's and Alice's); Hana's number; Gus and Alice added by hand
    assert registry.backfill(tmp_path, stores) == {"people": 5, "assertions": 2, "phones": 1, "added": 2}
    count = len(events(tmp_path))
    assert registry.backfill(tmp_path, stores) == NOTHING
    assert len(events(tmp_path)) == count
    reg = registry.at(tmp_path)
    assert sorted(p.display_name for p in reg.people()) == ["Alice", "Erin", "Gus", "Hana", "Ivy"]
    assert who(reg, registry.utterance(11.0)) == "Ivy"  # a hand-tagged line
    assert winner(reg.person_of(registry.utterance(13.0))).basis == Basis.OWNER_TAG
    # A voice's name is only ever the log's: a voice-matched line's stored name never names its voice.
    assert reg.person_of(registry.voiceprint(1)).person is None
    assert reg.person_of(registry.utterance(10.0)).person is None  # voice-matched: follows its voice


def test_a_forgotten_person_stays_gone_after_a_rerun(tmp_path, stores):
    registry.backfill(tmp_path, stores)
    reg = registry.at(tmp_path)
    # Today's delete_person drops Gus from the roster stores.
    (tmp_path / "people_added.json").write_text(json.dumps(["alice"]))
    assert reg.forget("Gus", "dashboard") > 0
    assert registry.backfill(tmp_path, stores) == NOTHING
    assert reg.person_id("Gus") is None


def test_old_names_on_transcript_lines_resolve_after_a_rename_or_merge(tmp_path, stores):
    registry.backfill(tmp_path, stores)
    reg = registry.at(tmp_path)
    alice = reg.person_id("Alice")
    # The dashboard renames Alice to Alicia and merges Ivy into Erin. Neither rewrites transcript lines.
    reg.rename("Alice", "Alicia", "dashboard")
    reg.merge("Ivy", "Erin", "dashboard")
    np.savez(stores, names=np.array(["Alicia", "Erin", "Face 3"]), embs=np.zeros((3, 4), np.float32))
    (tmp_path / "people_added.json").write_text(json.dumps(["Gus", "Alicia"]))
    assert registry.backfill(tmp_path, stores) == NOTHING
    assert (reg.person_id("Alice"), reg.person_id("Alicia")) == (None, alice)
    assert reg.person_of(registry.utterance(13.0)).person == alice
    assert reg.person_of(registry.utterance(11.0)).person == reg.person_id("Erin")
    assert reg.known_as("alice") == alice
    assert reg.ensure("Alice", "test") != alice  # someone new may take the old name
    assert reg.known_as("Alice") != alice


def test_saved_photos_are_owner_tags_on_the_photo(tmp_path, stores):
    (stores.parent / "alice").mkdir()
    (stores.parent / "alice" / "1700000000000.jpg").write_bytes(b"jpg")
    assert registry.backfill(tmp_path, stores)["assertions"] == 3  # the photo and the two hand-tagged lines
    reg = registry.at(tmp_path)
    d = reg.person_of(registry.photo(1700000000000))
    assert (name_of(reg, d.person), winner(d).basis) == ("Alice", Basis.OWNER_TAG)


# ---- voice names from introductions and the owner's spoken tag ----
def test_the_cli_and_introductions_say_where_a_label_came_from(tmp_path, reg):
    write_voices(tmp_path, {1: None, 2: None, 3: None})
    assert identity.label_speaker(identity.Paths(tmp_path), 2, "Dana", Basis.NAME_BADGE)[0]
    w = winner(reg.person_of(registry.voiceprint(2)))
    assert (w.basis, w.source) == (Basis.NAME_BADGE, "identity")


def test_the_owners_spoken_tag_is_an_owner_tag(tmp_path, reg, speech):
    """Review 4, finding 2: the owner's "that's Dana" (collar_intro) was recorded as an overheard introduction, so an
    earlier owner correction that took Dana off the voice withdrew it, and the voice named nobody."""
    write_voices(tmp_path, {1: None, 2: "Dana", 3: None})  # the owner names voice 2 on the dashboard
    reg.retract(registry.voiceprint(2), "dashboard")  # and takes it back
    now = time.time()
    speech.say({"ts_start": now - 2, "ts_end": now - 1, "speaker_id": 2, "speaker": "Speaker 2", "text": "hi"})
    with cfg.override({"owner": {"name": "Alice"}}):
        r = identity.collar_intro("Dana", tmp_path)  # the owner: "that's Dana", with no unknown face in view
    assert r["record"] is not None
    assert r["record"]["status"] == "applied"
    w = winner(reg.person_of(registry.voiceprint(2)))
    assert (name_of(reg, w.person), w.basis) == ("Dana", Basis.OWNER_TAG)
    # what a backfill rerun gives the same record
    kinds = ("collar", "cli", "self_intro")
    assert [registry.identity_basis({"kind": k}) for k in kinds] == [Basis.OWNER_TAG] * 2 + [Basis.INTRODUCTION]


# ---- the face store's writes, and the backfill's ----
FACE = np.zeros(512, np.float32)
FACE[0] = 1


@pytest.fixture
def face_dbs(tmp_path, monkeypatch):
    monkeypatch.setattr(faces, "DB", tmp_path / "faces" / "db.npz")
    monkeypatch.setattr(faces, "UNK_DB", tmp_path / "faces" / "unknown.npz")


def test_an_identity_record_names_the_face_group_by_its_kind(tmp_path, reg, face_dbs):
    paths = identity.Paths(tmp_path)
    other = np.zeros(512, np.float32)
    other[1] = 1
    first, second = faces.group_unknown(FACE), faces.group_unknown(other)  # two strangers the face worker grouped
    rec = {"id": "i-2", "kind": "self_intro", "confidence": 0.9, "evidence": {"ts_start": 99.0}}
    identity.apply_record(paths, {**rec, "faces": [{"cluster": first, "detections": ["d-1"]}]}, "Dana")
    w = winner(reg.person_of(registry.face_group(first)))
    assert (name_of(reg, w.person), w.basis, w.confidence) == ("Dana", Basis.INTRODUCTION, 0.9)
    assert w.evidence[0].id == "utterance:99.000"
    assert winner(reg.person_of(registry.detection("d-1"))).evidence[0].id == "utterance:99.000"
    # Review 4, finding 2: collar_intro's face assertion was a spoken introduction, below an owner correction.
    rec = {"id": "i-3", "kind": "collar", "confidence": 0.95, "faces": [{"cluster": second}]}
    identity.apply_record(paths, rec, "Dana", owner=True)  # what collar_intro calls
    w = winner(reg.person_of(registry.face_group(second)))
    assert (name_of(reg, w.person), w.basis) == ("Dana", Basis.OWNER_TAG)
    assert faces.load_db()[0] == ["Dana", "Dana"]  # both groups' faces are rows of hers now


# ---- the dashboard's writes, through its HTTP handler ----
@pytest.fixture
def dash(tmp_path, monkeypatch, move_dashboard_state, serve_dashboard, face_dbs):
    """The dashboard on tmp_path: every path it keeps under its state directory, and the face store, move there.
    Voice 1 is Alice's, and she said two lines. Returns post(path, body) -> (status, reply)."""
    move_dashboard_state(tmp_path)
    lines = [
        {"ts_start": 1000.0, "text": "one", "speaker": "Alice", "speaker_id": 1, "speaker_sim": 0.4},
        {"ts_start": 1001.0, "text": "two", "speaker": "Alice", "speaker_id": 1, "speaker_sim": 0.5},
    ]
    (tmp_path / "transcript.jsonl").write_text("".join(json.dumps(e) + "\n" for e in lines))
    fake().write(lines)  # and on the speech log, for the dashboard
    write_voices(tmp_path, {1: "Alice"})
    base = serve_dashboard()

    def post(path, body):
        req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            return e.code, json.load(e)

    with cfg.override({"paths": {"state": str(tmp_path), "faces": str(tmp_path / "faces")}}):
        yield post


def test_refine_labels_and_undo(reg, dash):
    def line(ts):
        return who(reg, registry.utterance(ts), via=registry.voiceprint(1))

    assert dash("/speakers/refine", {"action": "assign", "ts_start": 1000.0, "name": "Dana"})[0] == 200
    assert line(1000.0) == "Dana"
    dash("/speakers/refine", {"action": "multi", "ts_start": 1001.0})
    assert line(1001.0) is None  # several speakers: not Alice's line any more
    dash("/speakers/refine/undo", {"ts_starts": [1000.0, 1001.0]})
    assert (line(1000.0), line(1001.0)) == ("Alice", "Alice")  # back to following the voice
    reg.claim(registry.voiceprint(1), "Erin", Basis.OWNER_TAG, "test")  # the voice was someone else
    assert (line(1000.0), line(1001.0)) == ("Erin", "Erin")


def test_rename_keeps_the_id(reg, dash):
    alice = reg.person_id("Alice")
    assert dash("/people/rename", {"from": "Alice", "to": "Alicia"})[0] == 200
    assert (reg.person_id("Alicia"), reg.person_id("Alice")) == (alice, None)


@pytest.mark.parametrize("on_people", [True, False], ids=["on-people", "registry-only"])
def test_a_rename_onto_someone_without_a_voice_merges_them(tmp_path, reg, dash, monkeypatch, on_people):
    """Review 2, findings 14 and 25: Gus has no voice, so no voice moves and the person merge alone makes Alice's voice
    his, whether he is on People or only in the registry."""
    cmds = create_autospec(known_voices._speaker_cmds)
    monkeypatch.setattr(known_voices, "_speaker_cmds", cmds)
    if on_people:
        dash("/people/add", {"name": "Gus"})
    else:
        reg.describe("Gus", "test", {"role": "plumber"})
    gus = reg.person_id("Gus")
    assert dash("/people/rename", {"from": "Alice", "to": "Gus"})[0] == 200
    cmds.assert_not_called()  # no voice command: the microphone service is left alone
    assert reg.person_of(registry.voiceprint(1)).person == gus
    assert reg.person_id("Alice") is None
    assert name_of(reg, gus) == "Gus"


def test_a_rename_onto_someone_with_a_voice_makes_both_voices_theirs(tmp_path, reg, dash, monkeypatch):
    """Naming replaces merging: renaming Alice onto Gus, who has a voice, is a person merge alone. No voice is
    folded into another, and each voice reads as Gus."""
    write_voices(tmp_path, {1: "Alice", 2: "Gus"})
    gus = reg.person_id("Gus")
    cmds = create_autospec(known_voices._speaker_cmds)
    monkeypatch.setattr(known_voices, "_speaker_cmds", cmds)
    assert dash("/people/rename", {"from": "Alice", "to": "Gus"})[0] == 200
    cmds.assert_not_called()
    assert [reg.person_of(registry.voiceprint(i)).person for i in (1, 2)] == [gus, gus]
    assert reg.person_id("Alice") is None


def test_refresh_reads_the_log_only_when_something_was_written_since(tmp_path):
    """People's requests refreshed about a hundred times each, and each opened two connections to find
    nothing new. A refresh now asks the log's version first."""
    reads: list[str] = []

    class Counting(SQLiteEventLog):
        def read(self, stream, after=0, limit=None):
            reads.append(stream)
            return super().read(stream, after, limit)

    reg = registry.Registry(Counting(tmp_path / "knowledge.sqlite3"))
    reg.ensure("Dana", "test")
    reg.refresh()
    reads.clear()
    for _ in range(20):
        reg.refresh()
    assert reads == []  # nothing new: no reads at all

    other = registry.Registry(make_log(tmp_path))  # another process names someone
    other.ensure("Alice", "test")
    assert reg.person_id("Alice") is not None  # seen at once
    assert reads  # because the log was read again

    reads.clear()
    other.claim(VOICE, "Alice", Basis.OWNER_TAG, "dashboard")
    reg.refresh(registry.PEOPLE)  # one stream brought up to date...
    assert reads == [registry.PEOPLE]
    assert name_of(reg, reg.person_of(VOICE).person) == "Alice"  # ...the other still reads what it missed


def test_reference_writes_do_not_reread_identity_streams(tmp_path):
    reads = []

    class Counting(SQLiteEventLog):
        def read(self, stream, after=0, limit=None):
            reads.append(stream)
            return super().read(stream, after, limit)

    reg = registry.Registry(Counting(tmp_path / "knowledge.sqlite3"))
    reg.claim(VOICE, "Alice", Basis.OWNER_TAG, "test")
    reg.refresh()
    reads.clear()
    knowledge.Store(tmp_path / "knowledge.sqlite3").put(
        [knowledge.Record("reference", "voice.reference", None, 1, 1, 1, {"vector": [1.0]})]
    )
    assert reg.voice_names([1]) == {1: "Alice"}
    assert reads == []


def test_voiceprint_list_reuses_projection_across_nineteen_line_reads(tmp_path):
    class Counting(registry.Registry):
        decisions = 0

        def _decide(self, subject):
            self.decisions += 1
            return super()._decide(subject)

    reg = Counting(make_log(tmp_path))
    reg.claim(VOICE, "Alice", Basis.OWNER_TAG, "test")
    pid = reg.person_id("Alice")
    reg.decisions = 0
    assert [reg.voiceprints(pid) for _ in range(19)] == [[1]] * 19
    assert reg.decisions == 1
    reg.retract(VOICE, "test")
    assert reg.voiceprints(pid) == []
    reg.claim(VOICE, "Alice", Basis.OWNER_TAG, "test")
    assert reg.voiceprints(pid) == [1]
    reg.forget("Alice", "test")
    assert reg.voiceprints(pid) == []


def test_identity_purge_rebuilds_cached_projection(tmp_path, reg):
    reg.claim(VOICE, "Alice", Basis.OWNER_TAG, "test")
    assert reg.voice_names([1]) == {1: "Alice"}
    other = SQLiteEventLog(tmp_path / "knowledge.sqlite3")
    assert other.purge(Selector(stream=registry.ASSERTIONS), "test") == 1
    assert reg.voice_names([1]) == {1: None}


def test_replacing_same_incarnation_database_with_equal_positions_changes_identity_and_mark(tmp_path, reg):
    with (
        sqlite3.connect(tmp_path / "knowledge.sqlite3") as original,
        sqlite3.connect(tmp_path / "replacement.sqlite3") as fork,
    ):
        original.backup(fork)
    reg.claim(VOICE, "Alice", Basis.OWNER_TAG, "test")
    before = reg.mark()
    replacement_log = SQLiteEventLog(tmp_path / "replacement.sqlite3")
    replacement = registry.Registry(replacement_log)
    replacement.claim(VOICE, "Dana", Basis.OWNER_TAG, "test")
    replacement_log.close()
    (tmp_path / "replacement.sqlite3").replace(tmp_path / "knowledge.sqlite3")
    assert reg.voice_names([1]) == {1: "Dana"}
    assert reg.person_id("Alice") is None
    assert reg.mark() != before


def test_restoring_backup_in_place_resets_projection_when_revision_rewinds(tmp_path, reg):
    reg.claim(VOICE, "Alice", Basis.OWNER_TAG, "test")
    with (
        sqlite3.connect(tmp_path / "knowledge.sqlite3") as original,
        sqlite3.connect(tmp_path / "backup.sqlite3") as backup,
    ):
        original.backup(backup)
    reg.rename("Alice", "Dana", "test")
    assert reg.voice_names([1]) == {1: "Dana"}
    before = (tmp_path / "knowledge.sqlite3").stat().st_ino
    with (
        sqlite3.connect(tmp_path / "backup.sqlite3") as backup,
        sqlite3.connect(tmp_path / "knowledge.sqlite3") as target,
    ):
        backup.backup(target)
    assert (tmp_path / "knowledge.sqlite3").stat().st_ino == before
    assert reg.voice_names([1]) == {1: "Alice"}
    assert reg.person_id("Dana") is None


def test_worker_transaction_rolls_back_claims_references_and_private_projection(tmp_path, reg):
    with pytest.raises(ValueError, match="abort batch"), registry.transaction(tmp_path) as worker:
        assert registry.at(tmp_path) is worker
        worker.claim(VOICE, "Alice", Basis.OWNER_TAG, "test")
        knowledge.at(registry.knowledge_path(tmp_path)).put(
            [knowledge.Record("reference", "voice.reference", None, 1, 1, 1, {"vector": [1.0]})]
        )
        assert worker.voice_names([1]) == {1: "Alice"}
        raise ValueError("abort batch")
    assert registry.at(tmp_path) is reg
    assert registry.local(tmp_path).person_id("Alice") is None
    assert registry.local(tmp_path).voice_names([1]) == {1: None}
    assert reg.voice_names([1]) == {1: None}
    assert knowledge.at(registry.knowledge_path(tmp_path)).query("voice.reference") == []


def test_nineteen_line_claims_and_references_share_one_commit(tmp_path, reg, monkeypatch):
    reg.claim(VOICE, "Alice", Basis.OWNER_TAG, "test")
    registry.local(tmp_path)
    store = knowledge.at(registry.knowledge_path(tmp_path))
    store.close()
    statements, connections = [], []
    connect = sqlite3.connect

    def traced(*args, **kwargs):
        db = connect(*args, **kwargs)
        connections.append(db)
        db.set_trace_callback(statements.append)
        return db

    monkeypatch.setattr(sqlite3, "connect", traced)
    with registry.transaction(tmp_path) as worker:
        made = worker.claim_many(
            [registry.Claim(registry.utterance(n, "Mic"), "Alice") for n in range(19)], Basis.OWNER_TAG, "test"
        )
        store.put([knowledge.Record(str(n), "voice.reference", None, n, n, n, {"vector": [1.0]}) for n in range(19)])
        assert len(store.query("voice.reference")) == 19
    assert len(made) == 19
    assert len(connections) == 1
    assert statements.count("BEGIN IMMEDIATE") == 1
    assert statements.count("COMMIT") == 1
    assert reg.voice_names([1]) == {1: "Alice"}


def test_shared_readers_continue_during_worker_transaction_and_see_only_committed_claims(tmp_path, reg):
    reg.ensure("Alice", "test")
    writing, finish = threading.Event(), threading.Event()

    def write_batch():
        with registry.transaction(tmp_path) as worker:
            assert worker is not reg
            assert registry.at(tmp_path) is worker
            worker.claim(VOICE, "Alice", Basis.OWNER_TAG, "test")
            writing.set()
            assert finish.wait(5), "reader blocked behind the worker"

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(write_batch)
        assert writing.wait(5)
        assert registry.at(tmp_path) is reg
        assert reg.voice_names([1]) == {1: None}
        finish.set()
        pending.result(timeout=5)
    assert reg.voice_names([1]) == {1: "Alice"}


def test_a_prints_subject_id_is_a_stable_v5_that_differs_by_print_and_kind() -> None:
    first = registry.subject_uuid(registry.voiceprint(7))
    assert first == registry.subject_uuid(registry.voiceprint("7")), "the same print, the same id"
    assert uuid.UUID(first).version == 5
    others = {
        registry.subject_uuid(registry.voiceprint(8)),
        registry.subject_uuid(registry.face_group(7)),
        registry.subject_uuid(SubjectRef(SubjectKind.FACEPRINT, "speaker:7")),  # the same text, another kind
    }
    assert first not in others and len(others) == 3
    # Pinned: the namespace must never change, or every stored print's id would.
    assert first == str(uuid.uuid5(uuid.UUID("2da871d0-b163-4c60-bbf4-976d9d5cf73b"), "voiceprint:speaker:7"))


def test_forget_subjects_are_their_ids_and_the_prints_that_are_theirs(reg):
    reg.claim(registry.voiceprint(3), "Dana", Basis.OWNER_TAG, "test")
    reg.claim(registry.face_group(2), "Dana Cole", Basis.OWNER_TAG, "test")
    reg.merge("Dana Cole", "Dana", "test")  # someone merged into her: their face group is hers now
    reg.claim(registry.faceprint("enroll:abc"), "Dana", Basis.OWNER_TAG, "test")
    reg.claim(registry.voiceprint(4), "Dana", Basis.OWNER_TAG, "test")
    reg.claim(registry.voiceprint(4), "Alice", Basis.OWNER_TAG, "test")  # corrected: not hers any more
    reg.claim(registry.utterance(5.0), "Dana", Basis.OWNER_TAG, "test")  # a line is no print: not a subject
    dana, cole = reg.person_id("Dana"), next(iter(reg.group("Dana") - {reg.person_id("Dana")}))
    assert reg.forget_subjects("Dana") == {
        dana,
        cole,
        registry.subject_uuid(registry.voiceprint(3)),
        registry.subject_uuid(registry.face_group(2)),
        registry.subject_uuid(registry.faceprint("enroll:abc")),
    }
    assert reg.forget_subjects("Nobody") == set()


def test_events_of_given_types_in_a_window_come_in_time_order(reg):
    reg.claim(registry.voiceprint(1), "Alice", Basis.OWNER_TAG, "test")
    reg.ensure("Dana", "test", at=50.0)
    reg.log.append(
        [
            Event.new(
                EventType.MEASUREMENT_RECORDED,
                stream=registry.MEASUREMENTS,
                source="test",
                occurred_at=20.0,
                payload={"name": "weight", "value": 5.0},
            )
        ]
    )
    found = reg.events([EventType.MEASUREMENT_RECORDED, EventType.PERSON_REGISTERED], 0.0, 60.0)
    assert [(str(s.event.type), s.event.occurred_at) for s in found] == [
        ("MeasurementRecorded", 20.0),
        ("PersonRegistered", 50.0),
    ]
    assert reg.events([EventType.PERSON_REGISTERED], 51.0, 60.0) == []
    assert [str(s.event.type) for s in reg.events([EventType.IDENTITY_ASSERTED], 0.0, 1e12)] == ["IdentityAsserted"]


def test_a_face_detection_on_the_broker_is_an_observation_a_claim_can_name(reg):
    face = registry.detection("01a18be2-c1e9-740f-83c3-c4ebf8ea952c")
    assert face.id == "detection:01a18be2-c1e9-740f-83c3-c4ebf8ea952c"
    reg.ensure("Dana", "test")  # a model's match names only someone the registry knows
    reg.claim(face, "Dana", Basis.EMBEDDING_MATCH, "identity", confidence=0.7, producer=MODEL)
    assert who(reg, face) == "Dana"
    reg.claim(face, "Erin", Basis.OWNER_TAG, "dashboard")  # the owner's correction outranks the model's match
    assert who(reg, face) == "Erin"


def test_registry_persists_and_reopens_without_any_broker(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("registry must not discover or connect to a broker")

    monkeypatch.setattr(registry, "_socket", forbidden)
    monkeypatch.setattr("liberational.continuum_link.socket_path", forbidden)
    state = tmp_path / "standalone"
    assert not registry.log_exists(state)
    assert not state.exists()
    first = registry.at(state)
    person = first.ensure("Dana", "test")
    assert registry.log_exists(state)
    registry.close_all()
    reopened = registry.at(state)
    assert reopened.person_id("Dana") == person
    assert who(reopened, registry.voiceprint(7)) is None


def test_registry_states_are_independent_even_with_one_broker_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTINUUM_SOCKET", str(tmp_path / "no-broker.sock"))
    first = registry.at(tmp_path / "one")
    person = first.ensure("Dana", "test")
    second = registry.at(tmp_path / "two")
    assert second.person_id("Dana") is None
    assert first.person_id("Dana") == person
    assert registry.at(tmp_path / "one" / ".") is first
