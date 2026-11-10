"""Explicit activity names are runtime annotations, never measurement labels or implicit promotion."""

import pytest

import activity_annotations
from liberational import config as cfg
from liberational import segment_log
from liberational.dashboard import activities
from liberational.memory import events
from liberational.people import registry

T = 1_800_000_000.0


def test_owner_names_and_renames_an_existing_activity(tmp_path, vision):
    span = vision.add(
        segment_log.WALKBY_SPANS,
        ts_start=T,
        ts_end=T + 10,
        cam="cam0",
        kind="crossing",
        candidate=True,
        uncertainty={"value": 0.9},
    )
    with cfg.override({"paths": {"state": str(tmp_path)}}):
        for label in ["stirring", "mixing"]:
            code, reply = activities.name({"activity": span["id"], "start": T, "name": label}, True)
            assert (code, reply) == (200, {"ok": True, "activity": span["id"], "name": label})
    assert [v["name"] for v in activity_annotations.values(tmp_path, events.NAMED)] == ["stirring", "mixing"]
    assert not (tmp_path / "events.jsonl").exists()
    assert events.events(tmp_path, T, T + 20) == []


def written() -> list[dict]:
    """Every owner name and rating in the configured state's knowledge store (another test may have made the file)."""
    if not registry.knowledge_path().exists():
        return []
    state = registry.knowledge_path().parent
    return activity_annotations.values(state, events.NAMED) + activity_annotations.values(state, events.FEEDBACK)


@pytest.mark.parametrize(
    "body",
    [
        None,
        {},
        {"activity": "bad", "start": T, "name": "mixing"},
        {"activity": "00000000-0000-7000-8000-000000000001", "start": True, "name": "mixing"},
        {"activity": "00000000-0000-7000-8000-000000000001", "start": float("nan"), "name": "mixing"},
        {"activity": "00000000-0000-7000-8000-000000000001", "start": T, "name": " "},
        {"activity": "00000000-0000-7000-8000-000000000001", "start": T, "name": "x" * 121},
    ],
)
def test_invalid_names_never_write(body, vision):
    assert activities.name(body, True)[0] == 400
    assert not vision.records
    assert not written()


def test_missing_or_purged_source_cannot_be_named(vision):
    request = {"activity": "00000000-0000-7000-8000-000000000001", "start": T, "name": "mixing"}
    assert activities.name(request, True)[0] == 404
    assert not vision.records
    assert not written()


def test_existing_control_decision_is_required(vision):
    assert activities.name({}, False)[0] == 403
    assert not vision.records
    assert not written()


def test_list_includes_candidates_and_explicit_names_without_promoting(tmp_path, vision):
    from liberational.dashboard.presence import day_bounds

    start, _ = day_bounds("2027-01-15")
    old = vision.add(
        segment_log.WALKBY_SPANS,
        ts_start=start + 10,
        ts_end=start + 20,
        cam="cam0",
        kind="crossing",
        candidate=True,
        uncertainty={"value": 0.9},
    )
    latest = vision.add(
        segment_log.WALKBY_SPANS,
        ts_start=start + 10,
        ts_end=start + 20,
        cam="cam0",
        kind="crossing",
        candidate=True,
        uncertainty={"value": 0.8},
        replaces=old["id"],
    )
    activity_annotations.name(tmp_path, latest, "waiting")
    code, reply = activities.listing("2027-01-15", tmp_path)
    assert code == 200
    assert reply == {
        "items": [
            {
                "id": latest["id"],
                "camera": "cam0",
                "start": start + 10,
                "end": start + 20,
                "count": None,
                "name": "waiting",
                "kind": "crossing",
                "stream": "walkby.span",
                "person": None,
                "person_name": None,
                "track_id": None,
                "attributes": {},
                "attribute_evidence": {},
                "uncertainty": {"value": 0.8},
                "feedback": None,
                "promoted": False,
            }
        ],
        "truncated": False,
        "promotion": [
            {
                "kind": "crossing",
                "camera": None,
                "status": "counted",
                "remaining_ratings": None,
                "rated": 0,
                "right": 0,
                "precision": None,
                "min_ratings": 10,
                "min_precision": 0.8,
            }
        ],
    }


@pytest.mark.parametrize("day", ["", "bad", "2027-02-30", "20270115"])
def test_listing_rejects_invalid_day_without_reading(day, tmp_path, vision):
    assert activities.listing(day, tmp_path)[0] == 400
    assert vision.reads == 0


def test_several_days_are_listed_in_one_answer_each_with_its_own_activities(tmp_path, vision):
    from liberational.dashboard.presence import day_bounds

    first, _ = day_bounds("2027-01-15")
    second, _ = day_bounds("2027-01-16")
    vision.add(segment_log.WALKBY_SPANS, ts_start=first + 10, ts_end=first + 20, cam="cam0", kind="crossing")
    vision.add(segment_log.WALKBY_SPANS, ts_start=second + 30, ts_end=second + 40, cam="cam1", kind="crossing")
    code, reply = activities.listings("2027-01-16,2027-01-15,2027-01-14", tmp_path)
    assert code == 200
    assert list(reply["days"]) == ["2027-01-16", "2027-01-15", "2027-01-14"]
    assert [[item["start"] for item in day["items"]] for day in reply["days"].values()] == [
        [second + 30],
        [first + 10],
        [],
    ]
    assert reply["days"]["2027-01-15"] == activities.listing("2027-01-15", tmp_path)[1]


@pytest.mark.parametrize(
    "days",
    [
        pytest.param("", id="none"),
        pytest.param("2027-01-15,bad", id="one-invalid"),
        pytest.param(
            ",".join(f"2027-{month:02d}-{day:02d}" for month in (1, 3) for day in range(1, 17)), id="too-many"
        ),
    ],
)
def test_a_days_listing_is_refused_whole_when_any_part_is_wrong(days, tmp_path, vision):
    assert activities.listings(days, tmp_path)[0] == 400


def test_candidate_listing_is_bounded_and_reports_truncation(tmp_path, vision):
    for offset in range(5):
        vision.add(segment_log.WALKBY_SPANS, ts_start=T + offset, cam="cam0", kind="crossing")
    reply = events.detected(tmp_path, T, T + 10, limit=2)
    assert [r["start"] for r in reply["items"]] == [T + 3, T + 4]
    assert reply["truncated"] is True


@pytest.mark.parametrize("verdict", ["right", "wrong"])
def test_explicit_feedback_does_not_promote_candidates_or_rename_them(tmp_path, vision, verdict):
    source = vision.add(
        segment_log.WALKBY_SPANS,
        ts_start=T,
        ts_end=T + 10,
        cam="cam0",
        kind="crossing",
        candidate=True,
        uncertainty={"value": 0.1},
    )
    with cfg.override({"paths": {"state": str(tmp_path)}}):
        status, reply = activities.feedback({"activity": source["id"], "start": T, "verdict": verdict}, True)
    assert (status, reply) == (200, {"ok": True, "activity": source["id"], "verdict": verdict})
    assert activity_annotations.values(tmp_path, events.FEEDBACK) == [
        {
            "activity": source["id"],
            "verdict": verdict,
            "source": "owner",
            "kind": "crossing",
            "cam": "cam0",
            "stream": segment_log.WALKBY_SPANS,
        }
    ]
    assert events.events(tmp_path, T, T + 20) == []
    assert events.detected(tmp_path, T, T + 20)["items"][0]["name"] is None


@pytest.mark.parametrize(
    "body",
    [
        None,
        {},
        {"activity": "bad", "start": T, "verdict": "right"},
        {"activity": "00000000-0000-7000-8000-000000000001", "start": True, "verdict": "right"},
        {"activity": "00000000-0000-7000-8000-000000000001", "start": float("inf"), "verdict": "right"},
        {"activity": "00000000-0000-7000-8000-000000000001", "start": T, "verdict": "maybe"},
    ],
)
def test_invalid_feedback_never_writes(body, vision):
    assert activities.feedback(body, True)[0] == 400
    assert not vision.records
    assert not written()


def test_feedback_requires_local_control_and_an_existing_source(vision):
    assert activities.feedback({}, False)[0] == 403
    assert (
        activities.feedback({"activity": "00000000-0000-7000-8000-000000000001", "start": T, "verdict": "wrong"}, True)[
            0
        ]
        == 404
    )
    assert not vision.records
    assert not written()


def test_listing_uses_current_person_display_name_and_preserves_unknown_people(tmp_path, vision):
    from liberational.dashboard.presence import day_bounds

    reg = registry.at(tmp_path)
    person = reg.ensure("Dana", "test")
    reg.rename("Dana", "Dana Rivera", "test")
    start, _ = day_bounds("2027-01-15")
    vision.add(segment_log.WALKBY_SPANS, ts_start=start + 1, cam="cam0", kind="crossing", person=str(person))
    vision.add(segment_log.WALKBY_SPANS, ts_start=start + 2, cam="cam0", kind="crossing", person="unknown-id")
    status, reply = activities.listing("2027-01-15", tmp_path)
    assert status == 200
    assert [row["person_name"] for row in reply["items"]] == ["Dana Rivera", None]


def test_a_walk_by_written_before_its_face_was_known_is_named_from_its_track(tmp_path, vision):
    from liberational.dashboard.presence import day_bounds

    start, _ = day_bounds("2027-01-15")
    vision.add(
        segment_log.WALKBY_SPANS, ts_start=start + 10, ts_end=start + 14, cam="cam0", kind="crossing", track_id="7"
    )
    vision.add(
        segment_log.WALKBY_SPANS, ts_start=start + 20, ts_end=start + 24, cam="cam0", kind="crossing", track_id="9"
    )
    for at, name, track in [
        (11, "Dana", "7"),
        (12, "Dana", "7"),
        (13, "Bob", "7"),
        (12, "Bob", "8"),
        (3600, "Bob", "7"),
    ]:
        vision.face(
            start + at, "cam0", name, 0.8, track_id=track
        )  # track 8, and track 7 an hour on, aren't this walk-by
    status, reply = activities.listing("2027-01-15", tmp_path)
    assert status == 200
    assert [row["person_name"] for row in reply["items"]] == ["Dana", None]  # track 9 has no faces


def test_listing_resolves_merged_historical_person_ids_without_rewriting_activity(tmp_path, vision):
    from liberational.dashboard.presence import day_bounds

    reg = registry.at(tmp_path)
    alice = reg.ensure("Alice", "test")
    dana = reg.ensure("Dana", "test")
    reg.merge("Alice", "Dana", "test")
    reg.rename("Dana", "Dana Rivera", "test")
    start, _ = day_bounds("2027-01-15")
    source = vision.add(segment_log.WALKBY_SPANS, ts_start=start + 1, cam="cam0", kind="crossing", person=str(alice))
    status, reply = activities.listing("2027-01-15", tmp_path)
    assert status == 200
    assert reply["items"][0]["person_name"] == "Dana Rivera"
    assert reply["items"][0]["person"] == str(alice)
    assert source["person"] == str(alice)
    assert str(alice) != str(dana)
    reg.forget("Dana Rivera", "test")
    assert activities.listing("2027-01-15", tmp_path)[1]["items"][0]["person_name"] is None
    assert source["person"] == str(alice)


def test_a_rating_queue_reaches_unrated_candidates_past_the_bound(tmp_path, vision):
    # One unrated candidate before five rated ones: a 2-item listing of everything would show only rated ones.
    def crossing(start: float) -> dict:
        return vision.add(segment_log.WALKBY_SPANS, ts_start=start, ts_end=start + 1, cam="cam0", kind="crossing")

    waiting = crossing(T)
    rated = [crossing(T + 10 + offset) for offset in range(5)]
    with cfg.override({"paths": {"state": str(tmp_path)}}):
        for row in rated:
            assert (
                activities.feedback({"activity": row["id"], "start": row["ts_start"], "verdict": "right"}, True)[0]
                == 200
            )
    queue = events.detected(tmp_path, T, T + 20, limit=2, kind="crossing", unrated=True)
    assert [r["id"] for r in queue["items"]] == [waiting["id"]]
    assert queue["truncated"] is False
    other = events.detected(tmp_path, T, T + 20, kind="squat")
    assert other["items"] == []
    assert [row["kind"] for row in other["promotion"]] == ["crossing"]


def test_a_model_action_is_listed_rated_in_review_and_then_counted_by_ask(vision):
    from liberational.dashboard.presence import day_bounds
    from liberational.memory import event_counts, history_qa

    state = registry.knowledge_path().parent
    start, _ = day_bounds("2027-01-15")
    vision.add(segment_log.ACTION_COVERAGE, ts_start=start, ts_end=start + 3600, cam="cam0", frames=100, model="test")
    spans = [
        vision.add(segment_log.ACTION_SPANS, ts_start=start + 60 * n, ts_end=start + 60 * n + 2, cam="cam0",
                   kind="sit_down", candidate=True, track_id=str(n), subject_type="person", camera_moved=False)
        for n in range(3)
    ]  # fmt: skip
    window = {"start": start, "end": start + 3600}
    with cfg.override({"activity": {"promotion_min_ratings": 2}}):
        code, reply = activities.listing("2027-01-15", state)
        assert code == 200 and [item["kind"] for item in reply["items"]] == ["sit_down"] * 3
        assert next(r for r in reply["promotion"] if r["kind"] == "sit_down")["status"] == "needs_ratings"
        assert (
            "I can't count sit down yet" in event_counts.answer("sit_down", window, state, history_qa.stamp)["answer"]
        )
        for span in spans[:2]:
            body = {"activity": span["id"], "start": span["ts_start"], "verdict": "right"}
            assert activities.feedback(body, True)[0] == 200
        _, reply = activities.listing("2027-01-15", state)
        assert next(r for r in reply["promotion"] if r["kind"] == "sit_down")["status"] == "counted"
        assert event_counts.answer("sit_down", window, state, history_qa.stamp)["count"] == 3


def test_the_owner_adds_deletes_and_restores_an_activity_through_the_routes(tmp_path, vision):
    added = activities.edit_kind("/activity/kinds", {"name": "Squats", "description": None}, True, tmp_path)
    assert added[0] == 200
    assert (added[1]["kind"]["kind"], added[1]["kind"]["classes"]) == (
        "squats",
        {"ntu": ["squat down"], "kinetics": ["squat"]},
    )
    sat = vision.add(segment_log.ACTION_SPANS, ts_start=T, ts_end=T + 5, cam="cam0", kind="sit_down", candidate=True)
    activity_annotations.rate(tmp_path, sat, "right", stream=segment_log.ACTION_SPANS)
    assert activities.edit_kind("/activity/kinds/delete", {"kind": "sit_down"}, True, tmp_path)[0] == 200
    code, listed = activities.kinds(tmp_path)
    assert code == 200
    assert "squats" in [k["kind"] for k in listed["kinds"]]
    assert "sit_down" not in [k["kind"] for k in listed["kinds"]]
    assert listed["deleted"] == [{"kind": "sit_down", "name": "Sat down", "answers": 1}]
    code, restored = activities.edit_kind("/activity/kinds/restore", {"kind": "sit_down"}, True, tmp_path)
    assert (code, restored["kind"]["answers"]) == (200, 1)
    assert activities.kinds(tmp_path)[1]["deleted"] == []


@pytest.mark.parametrize(
    ("path", "body", "control", "code"),
    [
        pytest.param("/activity/kinds", {"name": "Squats"}, False, 403, id="not-this-machine"),
        pytest.param("/activity/kinds", ["Squats"], True, 400, id="not-an-object"),
        pytest.param("/activity/kinds", {"name": 3}, True, 400, id="name-not-text"),
        pytest.param("/activity/kinds", {"name": "Sat down"}, True, 400, id="name-taken"),
        pytest.param("/activity/kinds/delete", {}, True, 400, id="no-kind"),
        pytest.param("/activity/kinds/delete", {"kind": "zorbing"}, True, 404, id="unknown-kind"),
    ],
)
def test_a_refused_list_change_writes_nothing(tmp_path, path, body, control, code):
    assert activities.edit_kind(path, body, control, tmp_path)[0] == code
    assert not registry.knowledge_path(tmp_path).exists()
