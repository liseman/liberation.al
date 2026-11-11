"""Count questions about a named event answered from detected events, through history_qa, with a
stand-in for memory/events.py (events() and coverage())."""

import datetime as dt
import types
from unittest.mock import create_autospec

import pytest

from asking import query
from liberational import config as cfg
from liberational.memory import activity_questions, event_counts, history_qa, question, recall
from liberational.memory import ask as pack_ask

UTC = dt.UTC
NOW = dt.datetime(2026, 10, 29, 18, 0, tzinfo=UTC).timestamp()


def at(hour, minute):
    return dt.datetime(2026, 10, 29, hour, minute, tzinfo=UTC).timestamp()


def span(start, end, count, name, cam="cam0"):
    return {"id": f"ev-{start:.0f}", "start": start, "end": end, "camera": cam, "count": count, "name": name,
            "segment": f"{cam}/2026-10-29/x.mp4", "person": None, "kind": "model_action"}  # fmt: skip


class FakeEvents(types.ModuleType):
    """memory/events.py as a stand-in: the spans it found and the recordings it covered, set per test."""

    def __init__(self) -> None:
        super().__init__("liberational.memory.events")
        self.found: list[dict] = []
        self.covered: list[tuple[str, float, float]] = []

    def events(self, state, start, end):
        return [e for e in self.found if e["start"] < end and e["end"] > start]

    def coverage(self, state, start, end):
        return [c for c in self.covered if c[1] < end and c[2] > start]

    def named(self, state):
        return [(e["name"], e.get("kind")) for e in self.found if e.get("name") and e["name"] != e.get("kind")]


@pytest.fixture
def events(monkeypatch, tmp_path):
    fake = FakeEvents()
    monkeypatch.setattr(event_counts, "module", lambda: fake)  # immune to the real module being imported
    monkeypatch.setattr(event_counts, "_NAMES", {})
    monkeypatch.setattr(history_qa, "TZ", UTC)
    return fake


def ask(question, state):
    return history_qa.collect(question, state, now=NOW)


def test_counts_per_span_with_coverage_and_a_review_link_each(events, tmp_path):
    events.covered = [("cam0", at(9, 0), at(9, 30))]
    events.found = [span(at(9, 14), at(9, 16), 2, "opened the fridge"), span(at(9, 20), at(9, 22), 1, "Opened the fridge"),
                    span(at(9, 25), at(9, 26), 3, None)]  # fmt: skip
    r = ask("How many times did I open the fridge today?", tmp_path)
    assert r["paths"] == ["events"]
    assert r["answer"].startswith(
        "opened the fridge: 3 times (2 at 09:14, 1 at 09:20), from 1 analyzed recording. "
        "Also 1 span of something not named yet (3 times)."
    )
    assert [e["link"] for e in r["evidence"]][:2] == [
        f"/review#day=2026-10-29&t={at(9, 14)}",
        f"/review#day=2026-10-29&t={at(9, 20)}",
    ]
    assert r["coverage"]["events"]["recordings"] == 1


def test_a_window_nothing_covers_says_so_and_is_never_zero(events, tmp_path):
    events.found = [span(at(6, 5), at(6, 6), 1, "dog on the couch")]
    events.covered = [("cam0", at(6, 0), at(6, 30))]  # analyzed, but yesterday's question asks about another day
    r = ask("How many times was the dog on the couch yesterday?", tmp_path)
    assert r["paths"] == ["events"]
    assert r["answer"].startswith("No recording in that time has been analyzed for events yet")
    assert "0" not in r["answer"]


def test_covered_with_nothing_found_is_a_real_zero(events, tmp_path):
    events.found = [span(at(6, 5) - 86400, at(6, 6) - 86400, 1, "dog on the couch")]  # named yesterday, not today
    events.covered = [("cam0", at(9, 0), at(9, 30))]
    r = ask("how often did the dog get on the couch today", tmp_path)
    assert r["answer"].startswith("dog on the couch: not found in 1 analyzed recording.")


def test_a_span_across_midnight_counts_in_both_days_it_covers(events, readings, tmp_path):
    readings["how many times do I generally open the fridge"] = question.Reading(
        measure="count", usual=True, people=("me",), source="cameras"
    )
    midnight = dt.datetime(2026, 10, 28, tzinfo=UTC).timestamp()
    events.covered = [("cam0", midnight - 86400, midnight), ("cam0", midnight, midnight + 86400)]
    events.found = [span(midnight - 300, midnight + 300, 1, "opened the fridge")]
    r = ask("how many times do I generally open the fridge", tmp_path)
    assert r["paths"] == ["events", "typical"]
    assert [p["count"] for p in r["coverage"]["events"]["periods"]] == [1, 1]


def test_a_usual_count_reads_the_store_once_for_all_its_periods(events, readings, tmp_path, monkeypatch):
    readings["how many times do I generally open the fridge"] = question.Reading(
        measure="count", usual=True, people=("me",), source="cameras"
    )
    day = dt.datetime(2026, 10, 28, tzinfo=UTC).timestamp()
    events.covered = [("cam0", day, day + 86400)]
    events.found = [span(day + 3600, day + 3660, 1, "opened the fridge")]
    reads: list[tuple[float, float]] = []
    original = events.events
    monkeypatch.setattr(
        events, "events", lambda state, start, end: reads.append((start, end)) or original(state, start, end)
    )
    r = ask("how many times do I generally open the fridge", tmp_path)
    assert r["paths"] == ["events", "typical"]
    assert len(reads) == 1, "one read for the whole run of periods, not one per period"


def test_without_the_events_module_no_question_is_one_of_these(monkeypatch, tmp_path):
    monkeypatch.setattr(event_counts, "module", lambda: None)  # memory/events.py not installed
    monkeypatch.setattr(event_counts, "_NAMES", {})
    assert event_counts.wanted(query("How many times did I open the fridge today?", tmp_path), tmp_path) is None
    assert "isn't set up here yet" in event_counts.answer("opened the fridge", {"start": 0, "end": NOW}, tmp_path,
                                                         history_qa.stamp)["answer"]  # fmt: skip


def test_people_coming_into_view_stay_with_presence(events, tmp_path):
    events.covered = [("cam0", at(9, 0), at(9, 30))]
    events.found = [span(at(9, 14), at(9, 16), 2, "walked in")]
    r = ask("How many times did someone walk in today?", tmp_path)
    assert "events" not in (r.get("paths") or [])


def test_the_switch_turns_it_off(events, tmp_path):
    events.found = [span(at(9, 14), at(9, 16), 2, "opened the fridge")]
    with cfg.override({"events": {"answer_counts": False}}):
        assert event_counts.wanted(query("How many times did I open the fridge today?", tmp_path), tmp_path) is None
        assert "events" not in (ask("How many times did I open the fridge today?", tmp_path).get("paths") or [])


@pytest.mark.parametrize(
    ("word", "stemmed"),
    [("opened", "open"), ("opening", "open"), ("opens", "open"), ("couches", "couch"), ("glass", "glass"),
     ("red", "red"), ("bed", "bed"), ("sing", "sing"), ("closed", "clos"), ("close", "clos"), ("closing", "clos"),
     ("stopped", "stop"), ("sitting", "sit"), ("buses", "bus"), ("called", "call")],
)  # fmt: skip
def test_stem_drops_only_the_endings_a_name_and_a_question_differ_by(word, stemmed):
    assert event_counts.stem(word) == stemmed


def test_a_question_about_someone_counts_only_what_they_were_seen_in(events, tmp_path):
    events.covered = [("cam0", at(9, 0), at(9, 30))]
    events.found = [
        span(at(9, 14), at(9, 16), 2, "opened the fridge"),
        {**span(at(9, 20), at(9, 22), 1, "opened the fridge"), "person": "p-dana"},
    ]
    r = event_counts.answer("opened the fridge", {"start": at(0, 0), "end": NOW}, tmp_path, history_qa.stamp, UTC,
                            person="p-dana")  # fmt: skip
    assert r["answer"] == (
        "opened the fridge: 1 time (1 at 09:20), from 1 analyzed recording. 1 span in that time couldn't be told "
        "whose, and aren't counted."
    )


def test_an_event_count_is_the_answer_with_no_model(events, tmp_path, monkeypatch):
    events.covered = [("cam0", at(9, 0), at(9, 30))]
    events.found = [span(at(9, 14), at(9, 16), 2, "opened the fridge")]
    monkeypatch.setattr(
        pack_ask, "local_answer", create_autospec(pack_ask.local_answer, side_effect=AssertionError("no model"))
    )
    monkeypatch.setattr(recall, "loop", create_autospec(recall.loop, side_effect=AssertionError("no recall")))
    result = pack_ask.execute("How many times did I open the fridge today?", tmp_path, now=NOW)
    assert result["paths"] == ["events"]
    assert result["answer"].startswith("opened the fridge: 2 times (2 at 09:14)")


def test_a_span_named_only_by_its_detectors_kind_is_not_named_yet(events, tmp_path):
    # memory/events.py gives an unnamed span its kind ("model_pause") as its name: that's no name anyone gave it.
    events.covered = [("cam0", at(9, 0), at(9, 30))]
    still = {**span(at(9, 10), at(9, 12), 2, "model_pause"), "kind": "model_pause"}
    named = {**span(at(9, 14), at(9, 16), 1, "opened the fridge"), "kind": "model_action"}
    events.found = [still, named]
    assert event_counts.known_names(tmp_path) == ["opened the fridge"]
    r = ask("How many times did I open the fridge today?", tmp_path)
    assert "1 time" in r["answer"]
    assert "1 span of something not named yet (2 times)" in r["answer"]
    assert ask("How many times was it model_pause today?", tmp_path).get("paths") != ["events"]


def test_actual_crossing_without_repetition_count_counts_one_event(tmp_path, vision):
    from liberational import segment_log
    from liberational.memory import events as real_events

    record = vision.add(
        segment_log.WALKBY_SPANS,
        ts_start=at(9, 10),
        ts_end=at(9, 10),
        cam="cam0",
        kind="crossing",
        count=None,
        candidate=False,
        camera_moved=False,
        uncertainty={"value": None},
    )
    real_events.name_activity(tmp_path, real_events.NameActivity(record["id"], at(9, 10), "crossed the doorway"))
    vision.add(segment_log.WALKBY_COVERAGE, ts_start=at(9, 0), ts_end=at(9, 30), cam="cam0", frames=300)
    result = event_counts.answer(
        "crossed the doorway", {"start": at(9, 0), "end": at(9, 30)}, tmp_path, history_qa.stamp
    )
    assert result["answer"].startswith("crossed the doorway: 1 time")
    assert "1 time" in result["evidence"][0]["quote"]


def test_the_name_lookup_reads_the_knowledge_store_and_no_spans(monkeypatch, tmp_path, vision):
    from liberational import segment_log
    from liberational.memory import events as real_events

    monkeypatch.setattr(event_counts, "_NAMES", {})
    named = []
    for minute, name in enumerate(("fed the cat", "opened the fridge", "crossed the doorway")):
        record = vision.add(
            segment_log.WALKBY_SPANS,
            ts_start=at(9, minute),
            ts_end=at(9, minute + 1),
            cam="cam0",
            kind="crossing",
            count=1,
            candidate=False,
            camera_moved=False,
            uncertainty={"value": None},
        )
        real_events.name_activity(tmp_path, real_events.NameActivity(record["id"], at(9, minute), name))
        named.append(name)

    def no_spans(*args, **kwargs):
        raise AssertionError("the name lookup read every span")

    monkeypatch.setattr(real_events, "count_evidence", no_spans)
    monkeypatch.setattr(real_events, "_spans", no_spans)
    assert sorted(event_counts.known_names(tmp_path)) == sorted(named)


def test_a_failed_name_read_is_logged_and_not_cached(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(event_counts, "_NAMES", {})

    class Broken(types.ModuleType):
        def __init__(self) -> None:
            super().__init__("liberational.memory.events")
            self.calls = 0

        def named(self, state):
            self.calls += 1
            raise RuntimeError("the knowledge store is busy")

    broken = Broken()
    monkeypatch.setattr(event_counts, "module", lambda: broken)
    assert event_counts.known_names(tmp_path) == []
    assert broken.calls == 1
    assert "could not be read" in caplog.text
    assert event_counts.known_names(tmp_path) == []
    assert broken.calls == 2, "a failed read is not cached: the next question asks again"
    assert event_counts._NAMES == {}


def test_named_unverified_candidate_never_becomes_not_found_with_processed_coverage(tmp_path, vision):
    from liberational import segment_log
    from liberational.memory import events as real_events

    record = vision.add(
        segment_log.WALKBY_SPANS,
        ts_start=at(9, 10),
        ts_end=at(9, 12),
        cam="cam0",
        kind="crossing",
        count=6,
        candidate=True,
        camera_moved=False,
        uncertainty={"value": 0.1},
    )
    real_events.name_activity(tmp_path, real_events.NameActivity(record["id"], at(9, 10), "stirring"))
    vision.add(segment_log.WALKBY_COVERAGE, ts_start=at(9, 0), ts_end=at(9, 30), cam="cam0", frames=300)
    result = event_counts.answer("stirring", {"start": at(9, 0), "end": at(9, 30)}, tmp_path, history_qa.stamp)
    assert result["warnings"], "an unmeasured named candidate is unknown, not a verified absence"
    assert "not found" not in result["answer"]
    assert "6 times" not in result["answer"], "a candidate's repetition count is not a promoted fact"


def test_actual_named_candidate_routes_to_explicit_unknown_count(monkeypatch, tmp_path, vision):
    from liberational import segment_log
    from liberational.memory import events as real_events

    monkeypatch.setattr(event_counts, "_NAMES", {})
    record = vision.add(
        segment_log.WALKBY_SPANS,
        ts_start=at(9, 10),
        ts_end=at(9, 12),
        cam="cam0",
        kind="crossing",
        count=6,
        candidate=True,
        camera_moved=False,
        uncertainty={"value": 0.1},
    )
    real_events.name_activity(tmp_path, real_events.NameActivity(record["id"], at(9, 10), "stirring"))
    vision.add(segment_log.WALKBY_COVERAGE, ts_start=at(9, 0), ts_end=at(9, 30), cam="cam0", frames=300)
    assert event_counts.wanted(query("How many times was I stirring today?", tmp_path), tmp_path) == "stirring"
    result = ask("How many times was I stirring today?", tmp_path)
    assert result["paths"] == ["events"]
    assert result["warnings"]
    assert "unverified" in result["answer"]
    assert "not found" not in result["answer"]


@pytest.mark.parametrize("scope", ["camera", "person"])
def test_unverified_evidence_outside_requested_scope_does_not_hide_verified_crossing(scope, tmp_path, vision):
    from liberational import segment_log
    from liberational.memory import events as real_events

    good = vision.add(
        segment_log.WALKBY_SPANS,
        ts_start=at(9, 10),
        ts_end=at(9, 10),
        cam="cam0",
        person="Dana",
        kind="crossing",
        count=None,
        candidate=False,
        camera_moved=False,
    )
    pending = vision.add(
        segment_log.WALKBY_SPANS,
        ts_start=at(9, 15),
        ts_end=at(9, 16),
        cam="cam1" if scope == "camera" else "cam0",
        person="Alice",
        kind="crossing",
        count=6,
        candidate=True,
        camera_moved=False,
    )
    for row in (good, pending):
        real_events.name_activity(tmp_path, real_events.NameActivity(row["id"], row["ts_start"], "crossed the doorway"))
    vision.add(segment_log.WALKBY_COVERAGE, ts_start=at(9, 0), ts_end=at(9, 30), cam="cam0", frames=300)
    result = event_counts.answer(
        "crossed the doorway",
        {"start": at(9, 0), "end": at(9, 30), "cameras": ["cam0"] if scope == "camera" else []},
        tmp_path,
        history_qa.stamp,
        person="Dana" if scope == "person" else None,
    )
    assert result["answer"].startswith("crossed the doorway: 1 time")
    assert result["warnings"] == []


@pytest.mark.parametrize(
    ("uncertain", "prefix", "warns"),
    [
        (False, "crossed the doorway: 1 time", False),
        (True, "I found unverified candidates", True),
    ],
)
def test_point_event_at_inclusive_query_start_is_not_discarded(uncertain, prefix, warns, tmp_path, vision):
    from liberational import segment_log
    from liberational.memory import events as real_events

    record = vision.add(
        segment_log.WALKBY_SPANS,
        ts_start=at(9, 0),
        ts_end=at(9, 0),
        cam="cam0",
        kind="crossing",
        count=None,
        candidate=False,
        camera_moved=uncertain,
    )
    real_events.name_activity(tmp_path, real_events.NameActivity(record["id"], at(9, 0), "crossed the doorway"))
    vision.add(segment_log.WALKBY_COVERAGE, ts_start=at(9, 0), ts_end=at(9, 30), cam="cam0", frames=300)
    result = event_counts.answer(
        "crossed the doorway", {"start": at(9, 0), "end": at(9, 30)}, tmp_path, history_qa.stamp
    )
    assert result["answer"].startswith(prefix)
    assert bool(result["warnings"]) is warns


@pytest.mark.parametrize("uncertain", [False, True])
@pytest.mark.parametrize("point", [at(8, 59), at(9, 30)])
def test_point_events_before_start_or_at_exclusive_end_stay_outside(uncertain, point, tmp_path, vision):
    from liberational import segment_log
    from liberational.memory import events as real_events

    record = vision.add(
        segment_log.WALKBY_SPANS,
        ts_start=point,
        ts_end=point,
        cam="cam0",
        kind="crossing",
        count=None,
        candidate=False,
        camera_moved=uncertain,
    )
    real_events.name_activity(tmp_path, real_events.NameActivity(record["id"], point, "crossed the doorway"))
    vision.add(segment_log.WALKBY_COVERAGE, ts_start=at(9, 0), ts_end=at(9, 30), cam="cam0", frames=300)
    result = event_counts.answer(
        "crossed the doorway", {"start": at(9, 0), "end": at(9, 30)}, tmp_path, history_qa.stamp
    )
    assert result["answer"].startswith("crossed the doorway: not found")
    assert result["warnings"] == []


def _feedback_calibrated_repetition(vision, tmp_path):
    from liberational import segment_log
    from liberational.memory import events as real_events

    # Ten distinct ratings, with one known false detection inside the answer window.
    for i in range(10):
        start = at(9, 15) if i == 9 else at(8, i)
        source = vision.add(
            segment_log.WALKBY_SPANS,
            ts_start=start,
            ts_end=start + 12,
            cam="cam0",
            kind="crossing",
            count=7,
            candidate=True,
            camera_moved=False,
            uncertainty={"value": 0.1},
        )
        real_events.feedback_activity(
            tmp_path, real_events.FeedbackActivity(source["id"], start, "right" if i < 8 else "wrong")
        )
        if i == 9:
            real_events.name_activity(tmp_path, real_events.NameActivity(source["id"], start, "stirring"))
    target = vision.add(
        segment_log.WALKBY_SPANS,
        ts_start=at(9, 10),
        ts_end=at(9, 12),
        cam="cam0",
        kind="crossing",
        count=3,
        candidate=True,
        camera_moved=False,
        uncertainty={"value": 0.1},
    )
    real_events.name_activity(tmp_path, real_events.NameActivity(target["id"], at(9, 10), "stirring"))
    vision.add(segment_log.WALKBY_COVERAGE, ts_start=at(9, 0), ts_end=at(9, 30), cam="cam0", frames=300)
    return target


def test_feedback_promoted_kind_counts_in_ask_but_known_wrong_events_do_not(tmp_path, vision, monkeypatch):
    monkeypatch.setattr(event_counts, "_NAMES", {})
    _feedback_calibrated_repetition(vision, tmp_path)
    assert event_counts.wanted(query("How many times was I stirring today?", tmp_path), tmp_path) == "stirring"
    result = event_counts.answer("stirring", {"start": at(9, 0), "end": at(9, 30)}, tmp_path, history_qa.stamp)
    assert result["answer"].startswith("stirring: 3 times")
    assert result["warnings"] == []
    assert len(result["evidence"]) == 1
    assert result["evidence"][0]["link"] == f"/review#day=2026-10-29&t={at(9, 10)}"


def test_feedback_demotion_returns_ask_to_unknown_instead_of_retaining_a_count(tmp_path, vision):
    from liberational import segment_log
    from liberational.memory import events as real_events

    _feedback_calibrated_repetition(vision, tmp_path)
    before = event_counts.answer("stirring", {"start": at(9, 0), "end": at(9, 30)}, tmp_path, history_qa.stamp)
    assert before["answer"].startswith("stirring: 3 times")
    source = vision.add(
        segment_log.WALKBY_SPANS,
        ts_start=at(8, 20),
        ts_end=at(8, 21),
        cam="cam0",
        kind="crossing",
        count=2,
        candidate=True,
        camera_moved=False,
        uncertainty={"value": 0.1},
    )
    real_events.feedback_activity(tmp_path, real_events.FeedbackActivity(source["id"], at(8, 20), "wrong"))
    result = event_counts.answer("stirring", {"start": at(9, 0), "end": at(9, 30)}, tmp_path, history_qa.stamp)
    assert result["warnings"]
    assert "unverified" in result["answer"]
    assert "3 times" not in result["answer"]
    assert "not found" not in result["answer"]


@pytest.mark.parametrize(
    "question",
    [
        "How long was I sitting still today?",
        "How long was I still today?",
        "How many still events happened today?",
        "Was there repeated movement today?",
        "How many eating_or_drinking events were detected?",
    ],
)
def test_what_the_camera_model_doesnt_detect_says_so_without_a_model(tmp_path, monkeypatch, question):
    monkeypatch.setattr(
        pack_ask, "local_answer", create_autospec(pack_ask.local_answer, side_effect=AssertionError("no model"))
    )
    monkeypatch.setattr(recall, "loop", create_autospec(recall.loop, side_effect=AssertionError("no recall")))
    result = pack_ask.execute(question, tmp_path, now=NOW)
    assert result["answer"] == event_counts.UNSUPPORTED
    assert result["evidence"] == []
    assert result["coverage"]["activity"] == {"available": False, "reason": "unsupported"}


def test_direct_retired_count_reader_does_not_report_zero(events, tmp_path):
    events.covered = [("cam0", at(9, 0), at(9, 30))]
    result = event_counts.answer("repeated_movement", {"start": at(9, 0), "end": at(9, 30)}, tmp_path, history_qa.stamp)
    assert result["answer"] == event_counts.UNSUPPORTED
    assert result["coverage"]["available"] is False
    assert "count" not in result


def test_retired_action_words_do_not_displace_explicit_zone_counts(events, tmp_path, monkeypatch):
    from liberational.memory import zone_counts

    monkeypatch.setattr(event_counts, "zone_names", lambda state: ["drinking station"])
    measured = {"answer": "zone visits unavailable", "evidence": [], "coverage": {}, "warnings": []}
    monkeypatch.setattr(zone_counts, "answer", create_autospec(zone_counts.answer, return_value=measured))
    result = ask("How many times did someone visit the drinking station today?", tmp_path)
    assert result["answer"] == "zone visits unavailable"
    assert result["paths"] == ["events"]


def test_a_model_kind_with_no_action_coverage_is_not_a_count(tmp_path, vision):
    result = event_counts.answer("sit_down", {"start": at(9, 0), "end": at(10, 0)}, tmp_path, history_qa.stamp)
    assert "No recording in that time has been analyzed" in result["answer"]
    assert "count" not in result


def test_a_counted_span_says_whose_it_was_when_their_face_was_recognized(events, tmp_path):
    events.covered = [("cam0", at(9, 0), at(9, 30))]
    events.found = [{**span(at(9, 14), at(9, 16), 1, "opened the fridge"), "person_name": "Dana"}]
    assert ask("How many times did someone open the fridge today?", tmp_path)["answer"].startswith(
        "opened the fridge: 1 time (1 at 09:14 by Dana)"
    )


def asking(**fields):
    """A structured query's routing fields, as history_qa.describe gives them."""
    return {"measure": "count", "kind": "video", "subjects": {}, "action": "", "reading": {"people": []}, **fields}


@pytest.mark.parametrize(
    ("query_", "asked"),
    [
        pytest.param(
            asking(subjects={"event": "opened the fridge", "zone": "Fridge"}), "opened the fridge", id="event-first"
        ),
        pytest.param(asking(subjects={"zone": "Litter box"}), "Litter box", id="a-zone"),
        pytest.param(asking(action="sit_down"), "sit_down", id="a-model-action"),
        pytest.param(asking(action="stillness"), None, id="an-action-the-model-doesnt-detect"),
        pytest.param(asking(), None, id="names-nothing"),
        pytest.param(asking(subjects={"event": "opened the fridge"}, measure="when"), None, id="not-a-count"),
        pytest.param(
            asking(subjects={"event": "opened the fridge"}, kind="transcript"), None, id="about-what-was-said"
        ),
    ],
)
def test_what_a_count_asks_for(query_, asked, tmp_path):
    assert event_counts.wanted(query_, tmp_path) == asked


def test_no_count_is_an_event_count_while_the_switch_is_off(tmp_path):
    with cfg.override({"events": {"answer_counts": False}}):
        assert event_counts.wanted(asking(subjects={"event": "opened the fridge"}), tmp_path) is None


@pytest.mark.parametrize(
    ("query_", "asked"),
    [
        pytest.param(
            asking(measure="when", action="sit_down"), activity_questions.Asked("when", kind="sit_down"), id="when"
        ),
        pytest.param(
            asking(measure="duration", action="phone_use"),
            activity_questions.Asked("when", kind="phone_use"),
            id="how-long",
        ),
        pytest.param(asking(action="sit_down"), None, id="a-count-is-event-counts"),
        pytest.param(
            asking(measure="when", action="repeated_movement"), activity_questions.Asked("unsupported"), id="undetected"
        ),
        pytest.param(asking(measure="when", action="sit_down", kind="transcript"), None, id="about-what-was-said"),
        pytest.param(
            asking(measure="doing", subjects={"zone": "Kitchen"}, people=["Sam"], first_person=False),
            activity_questions.Asked("doing", zone="Kitchen", who="Sam"),
            id="doing-in-a-zone",
        ),
        pytest.param(
            asking(measure="doing", people=["the owner"], first_person=True),
            activity_questions.Asked("doing"),
            id="the-asker",
        ),
        pytest.param(asking(measure="describe"), None, id="neither"),
    ],
)
def test_what_an_activity_question_asks(query_, asked, tmp_path):
    assert activity_questions.asked(query_, tmp_path) == asked
