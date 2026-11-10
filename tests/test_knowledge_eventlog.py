"""Persistence and atomicity beyond the shared EventLog contract."""

import dataclasses
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from liberational.domain.eventlog import Selector
from liberational.domain.kernel import Event, EventId, EventType
from liberational.knowledge_eventlog import SQLiteEventLog


def event(stream="people", **payload):
    return Event.new(EventType.PERSON_REGISTERED, stream=stream, source="test", payload=payload or {"name": "Alice"})


def test_reopen_keeps_exact_envelope_and_consumer_cursor(tmp_path):
    path = tmp_path / "knowledge.sqlite3"
    original = event(name="Alice", nested={"values": [None, False, 1.25]})
    first = SQLiteEventLog(path)
    (position,) = first.append([original])
    first.ack("reader", position)
    first.close()
    reopened = SQLiteEventLog(path)
    assert reopened.read("people")[0].event.to_dict() == original.to_dict()
    assert list(reopened.subscribe("reader")) == []
    assert reopened.append([original]) == [position]


def test_bad_event_late_in_generator_leaves_whole_batch_unwritten(tmp_path):
    log = SQLiteEventLog(tmp_path / "knowledge.sqlite3")
    good = event()
    bad = dataclasses.replace(good, event_id=EventId("invalid"))
    with pytest.raises(ValueError):
        log.append(e for e in [good, bad])
    assert log.read("people") == []


def test_conflicting_event_id_rolls_back_other_new_events(tmp_path):
    log = SQLiteEventLog(tmp_path / "knowledge.sqlite3")
    original = event()
    log.append([original])
    changed = dataclasses.replace(original, payload={"name": "Dana"})
    with pytest.raises(ValueError):
        log.append([event(name="Bob"), changed])
    assert [item.event for item in log.read("people")] == [original]


def test_purged_id_cannot_resurrect_and_other_handle_sees_deletion_revision(tmp_path):
    path = tmp_path / "knowledge.sqlite3"
    first, other = SQLiteEventLog(path), SQLiteEventLog(path)
    original = event()
    first.append([original])
    before = other.version()
    assert first.purge(Selector(event_ids=frozenset({original.event_id})), "test") == 1
    assert other.version() > before
    with pytest.raises(ValueError):
        other.append([original])
    assert other.read("people") == []
    assert other.purges()[0].removed == 1


def test_empty_purge_does_not_publish_audit(tmp_path):
    log = SQLiteEventLog(tmp_path / "knowledge.sqlite3")
    assert log.purge(Selector(stream="missing"), "test") == 0
    assert log.purges() == []


def test_subscribe_uses_common_head_across_stream_reads(tmp_path, monkeypatch):
    log = SQLiteEventLog(tmp_path / "knowledge.sqlite3")
    first, second = event("a"), event("b")
    log.append([first, second])
    late_a, late_b = event("a", name="Dana"), event("b", name="Bob")
    query = log.store.query
    added = False

    def interleaved(collection, **kwargs):
        nonlocal added
        rows = query(collection, **kwargs)
        if collection == "event:a" and not added:
            added = True
            log.append([late_a, late_b])
        return rows

    monkeypatch.setattr(log.store, "query", interleaved)
    batch = list(log.subscribe("reader"))
    assert [item.event for item in batch] == [first, second]
    log.ack("reader", batch[-1].position)
    assert [item.event for item in log.subscribe("reader")] == [late_a, late_b]


def test_registry_names_claims_and_forgetting_survive_reopen(tmp_path):
    from liberational.domain.identity import Basis
    from liberational.people.registry import Registry, voiceprint

    path = tmp_path / "knowledge.sqlite3"
    first = Registry(SQLiteEventLog(path))
    first.claim(voiceprint(1), "Alice", Basis.OWNER_TAG, "test")
    person = first.person_id("Alice")
    first.rename("Alice", "Dana", "test")
    reopened = Registry(SQLiteEventLog(path))
    assert reopened.person_id("Dana") == person
    assert reopened.person_of(voiceprint(1)).person == person
    assert reopened.known_as("Alice") == person
    assert reopened.forget("Dana", "test") > 0
    forgotten = Registry(SQLiteEventLog(path))
    assert forgotten.person_id("Dana") is None
    assert forgotten.person_of(voiceprint(1)).person is None
    assert forgotten.known_as("Alice") is None


scalars = st.none() | st.booleans() | st.integers(min_value=-(2**63), max_value=2**63 - 1) | st.text()
json_values = st.recursive(
    scalars,
    lambda children: st.lists(children, max_size=5) | st.dictionaries(st.text(), children, max_size=5),
    max_leaves=15,
)


@settings(deadline=None)
@given(json_values)
def test_event_json_roundtrip_property(value):
    with tempfile.TemporaryDirectory() as folder:
        log = SQLiteEventLog(Path(folder) / "knowledge.sqlite3")
        sent = event(value=value)
        (position,) = log.append([sent])
        assert log.append([sent, sent]) == [position, position]
        (got,) = log.read("people")
        assert got.event.to_dict() == sent.to_dict()


def test_purge_cannot_split_a_concurrent_atomic_append_across_collections(tmp_path, monkeypatch):
    path = tmp_path / "knowledge.sqlite3"
    log, writer = SQLiteEventLog(path), SQLiteEventLog(path)
    log.append([event("a"), event("b")])
    late_a, late_b = event("a", name="Dana"), event("b", name="Bob")
    selecting = threading.Event()
    decode = log.store._stored
    query = log.store.query

    def selected(row):
        item = decode(row)
        if item.record.collection == "event:a":
            selecting.set()
        return item

    def append_when_selecting():
        assert selecting.wait(5), "purge never reached selection"
        return writer.append([late_a, late_b])

    monkeypatch.setattr(log.store, "_stored", selected)
    with ThreadPoolExecutor(max_workers=1) as pool:
        appended = pool.submit(append_when_selecting)

        def old_separate_read(collection, **kwargs):
            rows = query(collection, **kwargs)
            if collection == "event:a":
                appended.result(timeout=5)
            return rows

        # Forces the old multi-transaction implementation's vulnerable interleaving. The atomic implementation
        # decodes under its write lock, so the writer instead completes after the purge transaction commits.
        monkeypatch.setattr(log.store, "query", old_separate_read)
        assert log.purge(Selector(before=late_b.occurred_at + 1), "test") == 2
        appended.result(timeout=5)
    assert [item.event for item in log.read("a")] == [late_a]
    assert [item.event for item in log.read("b")] == [late_b]
    assert log.purges()[0].removed == 2


def test_domain_subject_index_keeps_person_and_print_filters_separate(tmp_path):
    from liberational.domain.identity import Basis
    from liberational.people.registry import Registry, voiceprint

    log = SQLiteEventLog(tmp_path / "knowledge.sqlite3")
    reg = Registry(log)
    reg.claim(voiceprint(1), "Alice", Basis.OWNER_TAG, "test")
    reg.claim(voiceprint(2), "Dana", Basis.OWNER_TAG, "test")
    (claim,) = log.store.query("event:identity", subjects=("ref:voiceprint:speaker:1",))
    assert claim.record.value["payload"]["assertion"]["subject"]["id"] == "speaker:1"
    assert reg.person_id("Alice") in claim.record.subjects
    assert reg.forget("Alice", "test") > 0
    assert log.store.query("event:identity", subjects=("ref:voiceprint:speaker:1",)) == []
    assert len(log.store.query("event:identity", subjects=("ref:voiceprint:speaker:2",))) == 1


def test_unnamed_line_label_is_indexed_without_a_person_binding(tmp_path):
    log = SQLiteEventLog(tmp_path / "knowledge.sqlite3")
    labeled = Event.new(
        EventType.UTTERANCE_LABELED,
        stream="identity",
        source="test",
        payload={"utterance": "utterance:desk@1760000000.000", "label": "unknown"},
    )
    log.append([labeled])
    (found,) = log.store.query("event:identity", subjects=("ref:observation:utterance:desk@1760000000.000",))
    assert found.record.value == labeled.to_dict()
    assert found.record.subjects == ("ref:observation:utterance:desk@1760000000.000",)
