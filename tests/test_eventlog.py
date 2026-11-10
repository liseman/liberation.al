"""The event log port (docs/DOMAIN-MODEL.md, "Event log port"): the contract every adapter must pass, and the SQLite adapter.

A new adapter joins the contract by adding a param to the `log` fixture.
"""

import pytest

from liberational.domain.eventlog import Filter, PurgeScope, Selector
from liberational.domain.kernel import Event, EventType, PersonId

T = 1_760_000_000.0  # an instant in 2025, in epoch seconds
ALICE = PersonId("0199a000-0000-7000-8000-00000000a11c")
DANA = PersonId("0199a000-0000-7000-8000-0000000000da")


def ev(stream="mic:desk", type=EventType.UTTERANCE_HEARD, at=T, subject=None, **payload):
    return Event.new(
        type, stream=stream, source="test", payload=payload or {"text": "hi"}, occurred_at=at, subject=subject
    )


@pytest.fixture(params=["sqlite"])
def log(tmp_path, request):
    from liberational.knowledge_eventlog import SQLiteEventLog

    adapter = SQLiteEventLog(tmp_path / "knowledge.sqlite3")
    try:
        yield adapter
    finally:
        adapter.close()


def events(stored):
    return [s.event for s in stored]


# The contract: the guarantees in the port's docstring.


def test_append_returns_rising_positions_and_read_returns_the_events(log):
    sent = [ev(text="one"), ev(text="two"), ev(text="three")]
    positions = log.append(sent)
    assert positions == sorted(positions)
    assert len(set(positions)) == 3
    stored = log.read("mic:desk")
    assert [s.position for s in stored] == positions
    assert events(stored) == sent


def test_read_takes_one_stream_after_a_position_up_to_a_limit(log):
    a1, b1, a2, a3 = ev("a"), ev("b"), ev("a"), ev("a")
    p = log.append([a1, b1, a2, a3])
    assert events(log.read("a")) == [a1, a2, a3]
    assert events(log.read("a", after=p[0])) == [a2, a3]
    assert events(log.read("a", after=p[0], limit=1)) == [a2]
    assert log.read("nobody") == []
    with pytest.raises(ValueError):
        log.read("a", limit=0)


def test_an_event_appended_twice_is_stored_once(log):
    e = ev()
    first = log.append([e])
    assert log.append([e, e]) == first * 2
    assert len(log.read("mic:desk")) == 1


def test_only_events_go_in(log):
    with pytest.raises(TypeError):
        log.append([{"type": "UtteranceHeard"}])


def test_a_consumer_gets_what_it_has_not_acknowledged_again(log):
    p = log.append([ev(text="one"), ev(text="two")])
    assert len(list(log.subscribe("c"))) == 2
    assert len(list(log.subscribe("c"))) == 2  # nothing acked, so at least once means again
    log.ack("c", p[0])
    assert [s.position for s in log.subscribe("c")] == [p[1]]
    log.ack("c", p[1])
    assert list(log.subscribe("c")) == []
    later = ev(text="three")
    log.append([later])
    assert events(log.subscribe("c")) == [later]


def test_consumers_keep_their_own_positions(log):
    p = log.append([ev(), ev()])
    log.ack("one", p[1])
    assert list(log.subscribe("one")) == []
    assert len(list(log.subscribe("two"))) == 2


def test_a_consumer_position_never_moves_back(log):
    p = log.append([ev(), ev()])
    log.ack("c", p[1])
    log.ack("c", p[0])
    assert list(log.subscribe("c")) == []


def test_a_filter_picks_types_and_streams(log):
    heard = ev("mic:desk")
    seen = ev("cam:front", EventType.SCENE_OBSERVED)
    heard_elsewhere = ev("mic:kitchen")
    log.append([heard, seen, heard_elsewhere])
    by_type = Filter(types=frozenset({EventType.UTTERANCE_HEARD}))
    assert events(log.subscribe("c", by_type)) == [heard, heard_elsewhere]
    by_stream = Filter(streams=frozenset({"cam:front"}))
    assert events(log.subscribe("c", by_stream)) == [seen]
    by_name = {"UtteranceHeard"}  # Filter converts a type name to its EventType
    both = Filter(types=by_name, streams=frozenset({"mic:kitchen"}))  # ty: ignore[invalid-argument-type]  # converted
    assert events(log.subscribe("c", both)) == [heard_elsewhere]


def test_a_new_consumer_replays_in_order_across_many_events(log):
    sent = [ev("a", n=i) for i in range(1234)]  # more than one page of the SQLite adapter's reads
    log.append(sent)
    assert events(log.subscribe("c")) == sent


def test_a_purge_removes_a_persons_events_and_records_that_it_did(log):
    alice = [ev(subject=ALICE, text="from alice"), ev("cam:front", EventType.SCENE_OBSERVED, subject=ALICE)]
    dana = ev(subject=DANA)
    nobody = ev()
    p = log.append([alice[0], dana, alice[1], nobody])
    assert log.purge(Selector(subjects=frozenset({ALICE})), "PersonForgotten") == 2
    assert [(s.position, s.event) for s in log.subscribe("c")] == [(p[1], dana), (p[3], nobody)]
    (record,) = log.purges()
    # The record says a person was forgotten, never which one.
    assert (record.scope, record.reason, record.removed) == (PurgeScope(subject=True), "PersonForgotten", 2)


def test_forgetting_a_set_of_subjects_is_one_purge_with_one_record_naming_none(log):
    """A person and the ids merged into them go in one purge, with one record that says only that it was by
    person, not which ids, nor how many."""
    merged = PersonId("0199a000-0000-7000-8000-0000000a11ce")
    alice, merged_in = ev(subject=ALICE, text="alice"), ev("cam:front", EventType.SCENE_OBSERVED, subject=merged)
    dana, nobody = ev(subject=DANA), ev()
    p = log.append([alice, dana, merged_in, nobody])
    assert log.purge(Selector(subjects=frozenset({ALICE, merged})), "PersonForgotten") == 2
    assert [(s.position, s.event) for s in log.subscribe("c")] == [(p[1], dana), (p[3], nobody)]
    (record,) = log.purges()
    assert (record.scope, record.removed) == (PurgeScope(subject=True), 2)
    assert ALICE not in repr(record) and merged not in repr(record)


def test_a_position_is_never_reused_after_a_purge(log):
    p = log.append([ev(), ev()])
    log.purge(Selector(stream="mic:desk"), "test")
    (q,) = log.append([ev()])
    assert q > max(p)


def test_a_purge_by_age_or_by_id_takes_only_those(log):
    old, new, picked = ev(at=T - 100), ev(at=T), ev(at=T + 1)
    log.append([old, new, picked])
    assert log.purge(Selector(before=T), "retention") == 1
    assert log.purge(Selector(event_ids=frozenset({picked.event_id})), "mistake") == 1
    assert events(log.read("mic:desk")) == [new]
    assert [r.removed for r in log.purges()] == [1, 1]


def test_a_purge_record_keeps_the_kind_of_purge_but_no_person(log):
    picked = ev("a", subject=ALICE)
    log.append([picked, ev("a", at=T - 100, subject=ALICE)])
    log.purge(Selector(stream="a", before=T, subjects=frozenset({ALICE})), "test")
    log.purge(Selector(event_ids=frozenset({picked.event_id})), "mistake")
    assert [r.scope for r in log.purges()] == [
        PurgeScope(subject=True, stream="a", before=T),
        PurgeScope(event_count=1),
    ]


def test_a_purge_combines_its_fields(log):
    log.append([ev("a", subject=ALICE), ev("b", subject=ALICE)])
    assert log.purge(Selector(subjects=frozenset({ALICE}), stream="a"), "test") == 1
    assert len(log.read("b")) == 1


def test_a_purge_must_select_something():
    with pytest.raises(ValueError):
        Selector()


def test_a_purge_before_time_zero_takes_nothing(log):
    log.append([ev(at=0.5), ev()])
    assert log.purge(Selector(before=0.0), "nothing is older") == 0
    assert len(log.read("mic:desk")) == 2
