"""Detected activity for count questions (memory/events.py): promoted spans, the owner's names, recounts, coverage."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from activity_annotations import name, rate, values
from liberational import config as cfg
from liberational import knowledge, segment_log
from liberational.memory import events

T = 1_800_000_000.0


def span(vision, start, end, kind="crossing", count=5, uncertainty=0.2, moved=False, **more):
    """A walk-by span. Real walk-bys are promoted (candidate=False); a test passes candidate=True for one that only
    owner feedback can promote."""
    return vision.add(
        segment_log.WALKBY_SPANS,
        ts_start=start,
        ts_end=end,
        cam="cam0",
        track_id="t1",
        kind=kind,
        count=count,
        uncertainty={"value": uncertainty, "joints": ["neck", "right_wrist"]},
        camera_moved=moved,
        **more,
    )


def test_promoted_spans_with_the_owners_name_and_recounts_replacing(tmp_path, vision):
    a = span(vision, T, T + 10)
    span(vision, T + 20, T + 30, uncertainty=0.8, moved=True)  # per-span camera uncertainty still excludes it
    span(vision, T + 40, T + 50, moved=True)  # the camera moved: not promoted
    b = span(vision, T + 60, T + 80, count=None)
    name(tmp_path, a, "stirring")
    recount = span(vision, T + 60, T + 80, count=None, replaces=b["id"])
    got = events.events(tmp_path, T, T + 100)
    assert [(e["id"], e["name"], e["count"]) for e in got] == [
        (a["id"], "stirring", 5),
        (recount["id"], "crossing", None),
    ]
    assert got[0]["camera"] == "cam0" and (got[0]["start"], got[0]["end"]) == (T, T + 10)


def test_coverage_is_each_cameras_stretches_joined(tmp_path, vision):
    for a, b, cam in ((T, T + 60, "cam0"), (T + 60, T + 120, "cam0"), (T + 300, T + 360, "cam0"), (T, T + 60, "cam1")):
        vision.add(segment_log.WALKBY_COVERAGE, ts_start=a, ts_end=b, cam=cam, frames=600)
    assert events.coverage(tmp_path, T + 30, T + 400) == [
        ("cam0", T + 30, T + 120),
        ("cam0", T + 300, T + 360),
        ("cam1", T + 30, T + 60),
    ]


def test_a_crossing_is_promoted_as_the_body_worker_decided_it(tmp_path, vision):
    c = vision.add(
        segment_log.WALKBY_SPANS,
        ts_start=T,
        ts_end=T,
        cam="cam0",
        track_id="t9",
        kind="crossing",
        uncertainty={"value": None, "joints": []},
        camera_moved=False,
    )
    assert [e["id"] for e in events.events(tmp_path, T - 1, T + 1)] == [c["id"]]


def test_owner_name_is_kept_over_the_source_span_in_the_knowledge_store_without_promoting(tmp_path, vision):
    source = span(vision, T, T + 10, candidate=True)
    returned = events.name_activity(tmp_path, events.NameActivity(source["id"], T + 5, " stirring "))
    assert (returned["ts_start"], returned["ts_end"], returned["name"]) == (T, T + 10, "stirring")
    (row,) = events._store(tmp_path).query(events.NAMED)
    assert (row.record.key, row.record.start_ns, row.record.end_ns) == (
        source["id"],
        round(T * 1e9),
        round((T + 10) * 1e9),
    )
    assert row.record.value == {
        "activity": source["id"],
        "name": "stirring",
        "source": "owner",
        "kind": "crossing",
        "cam": "cam0",
        "stream": segment_log.WALKBY_SPANS,
    }
    assert events.events(tmp_path, T, T + 20) == []
    assert events.detected(tmp_path, T, T + 20)["items"][0]["name"] == "stirring"


def test_naming_twice_keeps_the_latest_written_even_with_reverse_uuid_order(tmp_path, vision):
    source = span(vision, T, T + 10)
    start, end = round(T * 1e9), round((T + 10) * 1e9)
    for record_id, label in [
        ("ffffffff-ffff-7fff-bfff-ffffffffffff", "old"),
        ("00000000-0000-7000-8000-000000000000", "new"),
    ]:
        value = {"activity": source["id"], "name": label, "source": "owner"}
        events._store(tmp_path).put(
            [knowledge.Record(record_id, events.NAMED, source["id"], start, end, time.time_ns(), value)]
        )
    assert events.events(tmp_path, T, T + 20)[0]["name"] == "new"
    events.name_activity(tmp_path, events.NameActivity(source["id"], T, "newest"))
    assert events.events(tmp_path, T, T + 20)[0]["name"] == "newest"


def test_a_name_from_anyone_but_the_owner_is_not_shown(tmp_path, vision):
    source = span(vision, T, T + 10)
    name(tmp_path, source, "guessed", source="model")
    assert events.events(tmp_path, T, T + 20)[0]["name"] == "crossing"


def test_concurrent_names_of_one_span_both_land(tmp_path, vision):
    source = span(vision, T, T + 10)
    barrier = threading.Barrier(2)

    def label(text):
        barrier.wait()
        return events.name_activity(tmp_path, events.NameActivity(source["id"], T, text))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(label, ["waiting", "resting"]))
    assert {r["name"] for r in results} == {"waiting", "resting"}
    stored = values(tmp_path, events.NAMED)
    assert sorted(v["name"] for v in stored) == ["resting", "waiting"]
    assert events.events(tmp_path, T, T + 20)[0]["name"] == stored[-1]["name"]


@pytest.mark.parametrize(
    "annotate",
    [
        pytest.param(
            lambda state, span_id: events.name_activity(state, events.NameActivity(span_id, T, "ghost")), id="name"
        ),
        pytest.param(
            lambda state, span_id: events.feedback_activity(state, events.FeedbackActivity(span_id, T, "right")),
            id="feedback",
        ),
    ],
)
def test_annotating_an_unknown_span_raises_lookup_error_and_writes_nothing(tmp_path, vision, annotate):
    span(vision, T, T + 10)
    with pytest.raises(LookupError):
        annotate(tmp_path, segment_log.new_id())
    assert values(tmp_path, events.NAMED) == values(tmp_path, events.FEEDBACK) == []


def test_unmeasured_v2_candidates_are_visible_for_naming_but_never_promoted(tmp_path, vision):
    candidate = span(
        vision,
        T,
        T + 10,
        candidate=True,
        uncertainty=0.01,
        attribute_evidence={"count": {"value": 5, "uncertainty": {"value": 0.01}}},
    )
    crossing = span(vision, T + 20, T + 20, candidate=True, uncertainty=0.01)
    events.name_activity(tmp_path, events.NameActivity(candidate["id"], T, "stirring"))
    assert events.events(tmp_path, T - 1, T + 30) == []
    detected = events.detected(tmp_path, T - 1, T + 30)["items"]
    assert [(row["id"], row["promoted"]) for row in detected] == [(candidate["id"], False), (crossing["id"], False)]
    assert detected[0]["name"] == "stirring"


def test_review_dto_includes_evidence_and_latest_owner_feedback_without_changing_count_facts(tmp_path, vision):
    source = span(
        vision,
        T,
        T + 10,
        candidate=True,
        attributes={"duration_s": 10},
        attribute_evidence={
            "duration_s": {"value": 10, "interpretation": {"input": "pose.body", "records": ["large-source-id"]}}
        },
    )
    source["track_id"] = "track-a"
    before = events.count_evidence(tmp_path, T, T + 20)
    rate(tmp_path, source, "right")
    rate(tmp_path, source, "wrong")
    rate(tmp_path, source, "right", source="model")
    row = events.detected(tmp_path, T, T + 20)["items"][0]
    assert row.get("feedback") == "wrong"
    assert row.get("track_id") == "track-a"
    assert row.get("attributes") == {"duration_s": 10}
    assert row.get("attribute_evidence") == {"duration_s": {"value": 10, "interpretation": {"input": "pose.body"}}}
    assert row.get("uncertainty") == {"value": 0.2, "joints": ["neck", "right_wrist"]}
    assert before[1]
    assert events.count_evidence(tmp_path, T, T + 20) == ([], [])
    assert events.events(tmp_path, T, T + 20) == []


def ratings(state, vision, kind, verdicts, cam="cam0", stream=segment_log.WALKBY_SPANS):
    """One candidate span per verdict, rated by the owner, written to `stream` and rated as from it."""
    for offset, verdict in enumerate(verdicts):
        source = vision.add(stream, ts_start=T - 1000 + offset, cam=cam, kind=kind, candidate=True, camera_moved=False)
        rate(state, source, verdict, stream=stream)


def test_global_feedback_promotes_candidates_and_wrong_is_rejected_not_unknown(tmp_path, vision):
    ratings(tmp_path, vision, "crossing", ["right"] * 8 + ["wrong"] * 2)
    target = span(vision, T, T + 5, candidate=True, uncertainty=0.99)
    wrong = span(vision, T + 10, T + 15, candidate=True)
    rate(tmp_path, wrong, "wrong")
    # Eleven votes demote; first exclude the directly rejected source even while the whole kind is withheld.
    assert all(row["id"] != wrong["id"] for rows in events.count_evidence(tmp_path, T, T + 20) for row in rows)
    # Correcting the same source replaces its vote; it is not a twelfth independent rating.
    rate(tmp_path, wrong, "right")
    assert target["id"] in [row["id"] for row in events.events(tmp_path, T, T + 20)]
    summary = events.promotion_measurements(tmp_path)
    global_row = next(row for row in summary if row["kind"] == "crossing" and row["camera"] is None)
    assert (global_row["rated"], global_row["right"], global_row["status"]) == (11, 9, "counted")


def test_repeated_clicks_do_not_satisfy_the_minimum_distinct_event_ratings(tmp_path, vision):
    source = span(vision, T, T + 1, candidate=True)
    for _ in range(10):
        rate(tmp_path, source, "right")
    assert events.events(tmp_path, T, T + 2) == []
    row = next(row for row in events.promotion_measurements(tmp_path) if row["camera"] is None)
    assert (row["rated"], row["right"]) == (1, 1)


def test_camera_own_policy_overrides_the_kind_policy_after_ten_ratings(tmp_path, vision):
    ratings(tmp_path, vision, "crossing", ["wrong"] * 10, cam="cam0")
    ratings(tmp_path, vision, "crossing", ["right"] * 10, cam="cam1")
    for cam in ("cam0", "cam1", "cam2"):
        vision.add(segment_log.WALKBY_SPANS, ts_start=T, cam=cam, kind="crossing", candidate=True, camera_moved=False)
    assert [row["camera"] for row in events.events(tmp_path, T, T + 1)] == ["cam1"]
    assert [row["camera"] for row in events.count_evidence(tmp_path, T, T + 1)[1]] == ["cam0", "cam2"]


def test_crossing_defaults_are_preserved_but_sufficient_negative_feedback_demotes(tmp_path, vision):
    target = span(vision, T, T, kind="crossing", candidate=False)
    assert [row["id"] for row in events.events(tmp_path, T - 1, T + 1)] == [target["id"]]
    ratings(tmp_path, vision, "crossing", ["right"] * 7 + ["wrong"] * 3)
    assert events.events(tmp_path, T - 1, T + 1) == []
    assert [row["id"] for row in events.count_evidence(tmp_path, T - 1, T + 1)[1]] == [target["id"]]


@pytest.mark.parametrize("minimum", [True, 0, -1, 1.9])
def test_promotion_rejects_non_positive_integer_minimums(tmp_path, vision, minimum):
    with cfg.override({"activity": {"promotion_min_ratings": minimum}}), pytest.raises(ValueError):
        events.promotion_measurements(tmp_path)


@pytest.mark.parametrize("precision", [True, -0.1, 1.1, float("inf"), float("nan")])
def test_promotion_rejects_invalid_precision(tmp_path, vision, precision):
    with cfg.override({"activity": {"promotion_min_precision": precision}}), pytest.raises(ValueError):
        events.promotion_measurements(tmp_path)


def test_nine_distinct_ratings_wait_and_the_tenth_at_eighty_percent_promotes(tmp_path, vision):
    ratings(tmp_path, vision, "crossing", ["right"] * 7 + ["wrong"] * 2)
    target = span(vision, T, T + 2, candidate=True)
    assert events.events(tmp_path, T, T + 3) == []
    extra = vision.add(segment_log.WALKBY_SPANS, ts_start=T - 10, cam="cam0", kind="crossing", candidate=True)
    events.feedback_activity(tmp_path, events.FeedbackActivity(extra["id"], T - 10, "right"))
    assert [row["id"] for row in events.events(tmp_path, T, T + 3)] == [target["id"]]


def test_bad_camera_angle_demotes_even_when_global_precision_passes(tmp_path, vision):
    ratings(tmp_path, vision, "crossing", ["right"] * 7 + ["wrong"] * 3, cam="cam0")
    ratings(tmp_path, vision, "crossing", ["right"] * 33 + ["wrong"] * 7, cam="cam1")
    for cam in ("cam0", "cam1", "cam2"):
        vision.add(segment_log.WALKBY_SPANS, ts_start=T, cam=cam, kind="crossing", candidate=True)
    assert [row["camera"] for row in events.events(tmp_path, T, T + 1)] == ["cam1", "cam2"]
    assert [row["camera"] for row in events.count_evidence(tmp_path, T, T + 1)[1]] == ["cam0"]
    summaries = events.promotion_measurements(tmp_path)
    assert next(row for row in summaries if row["camera"] is None)["precision"] == 0.8
    assert next(row for row in summaries if row["camera"] == "cam0")["status"] == "needs_correct_ratings"


def test_review_item_limit_does_not_reduce_the_rating_cohort(tmp_path, vision):
    ratings(tmp_path, vision, "crossing", ["right"] * 8 + ["wrong"] * 2)
    span(vision, T, T + 1, candidate=True)
    span(vision, T + 2, T + 3, candidate=True)
    response = events.detected(tmp_path, T, T + 4, limit=1)
    assert len(response["items"]) == 1
    assert response["truncated"] is True
    assert response["items"][0]["promoted"] is True
    assert next(row for row in response["promotion"] if row["camera"] is None)["rated"] == 10


@pytest.mark.parametrize("verdicts", [[], ["right"] * 10], ids=["unrated", "rated-right"])
def test_review_does_not_claim_counted_in_ask_when_event_answers_are_switched_off(tmp_path, vision, verdicts):
    ratings(tmp_path, vision, "crossing", verdicts)
    span(vision, T, T + 1, candidate=False)
    with cfg.override({"events": {"answer_counts": False}}):
        response = events.detected(tmp_path, T, T + 2)
        assert next(row for row in response["promotion"] if row["camera"] is None)["status"] == "disabled"
    assert (
        next(row for row in events.detected(tmp_path, T, T + 2)["promotion"] if row["camera"] is None)["status"]
        == "counted"
    )


def test_candidates_count_only_after_feedback_promotes_their_kind(tmp_path, vision):
    from liberational.memory import event_counts, history_qa

    detected = span(vision, T, T + 2, kind="crossing", count=None, candidate=True)
    name(tmp_path, detected, "marked gesture")
    vision.add(segment_log.WALKBY_COVERAGE, ts_start=T - 60, ts_end=T + 60, cam="cam0", frames=1200)
    window = {"start": T - 30, "end": T + 30}

    def ask():
        return event_counts.answer("marked gesture", window, tmp_path, history_qa.stamp)

    assert events.events(tmp_path, T - 30, T + 30) == []
    assert ask()["coverage"]["count_available"] is False
    ratings(tmp_path, vision, "crossing", ["right"] * 7 + ["wrong"] * 3)
    assert ask()["coverage"]["count_available"] is False
    ratings(tmp_path, vision, "crossing", ["right"] * 10)
    assert [row["id"] for row in events.events(tmp_path, T - 30, T + 30)] == [detected["id"]]
    assert ask()["answer"].startswith("marked gesture: 1 time")


@pytest.mark.parametrize(
    "kind",
    [
        None,
        "still",
        "repeated_movement",
        "pickup",
        "putdown",
        "eating_or_drinking",
    ],
)
def test_retired_rule_rows_are_not_read_or_calibrated(tmp_path, vision, kind):
    source = span(vision, T, T + 1, kind=kind)
    ratings(tmp_path, vision, kind, ["right"] * 10)
    name(tmp_path, source, "legacy activity")
    assert events.count_evidence(tmp_path, T, T + 2) == ([], [])
    assert events.detected(tmp_path, T, T + 2)["items"] == []
    assert events.promotion_measurements(tmp_path) == []
    with pytest.raises(LookupError):
        events.name_activity(tmp_path, events.NameActivity(source["id"], T, "new label"))


def test_a_rating_from_before_its_source_stream_was_kept_does_not_count_toward_promotion(tmp_path, vision):
    rated = [span(vision, T - 100 + i, T - 100 + i, candidate=True) for i in range(10)]
    for source in rated:
        rate(tmp_path, source, "right", stream=None)
    target = span(vision, T, T + 1, candidate=True)
    assert events.promotion_measurements(tmp_path) == []
    assert events.events(tmp_path, T, T + 2) == []
    assert events.detected(tmp_path, T, T + 2)["items"][0]["feedback"] is None
    # The same verdicts with their stream promote the kind.
    for source in rated:
        rate(tmp_path, source, "right")
    assert [row["id"] for row in events.events(tmp_path, T, T + 2)] == [target["id"]]


def action(vision, start, end, kind="sit_down"):
    """A see-actions model action: always a candidate until owner feedback promotes its kind."""
    return vision.add(
        segment_log.ACTION_SPANS,
        ts_start=start,
        ts_end=end,
        cam="cam0",
        kind=kind,
        candidate=True,
        track_id="1",
        subject_type="person",
        camera_moved=False,
    )


def test_a_model_action_is_a_candidate_until_enough_right_ratings_promote_its_kind(tmp_path, vision):
    target = action(vision, T, T + 3)
    assert [(row["id"], row["promoted"]) for row in events.detected(tmp_path, T, T + 4)["items"]] == [
        (target["id"], False)
    ]
    assert events.events(tmp_path, T, T + 4) == []
    ratings(tmp_path, vision, "sit_down", ["right"] * 9, stream=segment_log.ACTION_SPANS)
    assert events.events(tmp_path, T, T + 4) == []
    ratings(tmp_path, vision, "sit_down", ["right"], stream=segment_log.ACTION_SPANS)
    assert [(row["id"], row["kind"]) for row in events.events(tmp_path, T, T + 4)] == [(target["id"], "sit_down")]
    row = next(row for row in events.promotion_measurements(tmp_path) if row["camera"] is None)
    assert (row["kind"], row["rated"], row["status"]) == ("sit_down", 10, "counted")


def test_feedback_on_a_model_action_records_its_stream_kind_and_camera(tmp_path, vision):
    target = action(vision, T, T + 3, kind="drink")
    returned = events.feedback_activity(tmp_path, events.FeedbackActivity(target["id"], T, "right"))
    assert returned == {
        "activity": target["id"],
        "verdict": "right",
        "source": "owner",
        "kind": "drink",
        "cam": "cam0",
        "stream": segment_log.ACTION_SPANS,
        "ts_start": T,
        "ts_end": T + 3,
    }
    assert values(tmp_path, events.FEEDBACK) == [{k: v for k, v in returned.items() if not k.startswith("ts_")}]


def test_feedback_wrong_excludes_the_span_from_counts_and_shows_on_review(tmp_path, vision):
    source = span(vision, T, T + 10)
    before = events.count_evidence(tmp_path, T, T + 20)
    for verdict in ("right", "wrong"):
        events.feedback_activity(tmp_path, events.FeedbackActivity(source["id"], T, verdict))
    assert [v["verdict"] for v in values(tmp_path, events.FEEDBACK)] == ["right", "wrong"]
    assert [row["id"] for row in before[0]] == [source["id"]]
    assert events.detected(tmp_path, T, T + 20)["items"][0]["feedback"] == "wrong"
    assert events.count_evidence(tmp_path, T, T + 20) == ([], [])
