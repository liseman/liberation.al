"""Questions about an animal: what it has been doing, and when it was last seen or last in a zone."""

import datetime as dt
from zoneinfo import ZoneInfo

import pytest

from liberational import knowledge, segment_log
from liberational.memory import history_qa as h
from liberational.memory import subject_activity, zone_counts
from liberational.see import zones

LA = ZoneInfo("America/Los_Angeles")
DAY = dt.datetime(2026, 10, 23, tzinfo=LA).timestamp()
NOW = DAY + 20 * 3600
H = 3600.0
BOX = [[0.1, 0.6], [0.3, 0.6], [0.3, 0.9], [0.1, 0.9]]


@pytest.fixture(autouse=True)
def roster(monkeypatch):
    monkeypatch.setattr(h, "TZ", LA)
    monkeypatch.setattr(subject_activity, "_pets", lambda state: [{"name": "May", "species": "cat"}])
    zone_counts._NAMES.clear()


@pytest.mark.parametrize(
    ("question", "path"),
    [
        ("what has the cat been doing lately?", "animal-activity"),
        ("what has May been up to?", "animal-activity"),
        ("when did I last see the cat?", "animal-last-seen"),
        ("where is May?", "animal-last-seen"),
    ],
)
def test_without_a_zone_an_animal_question_says_tracking_was_removed_not_that_nothing_happened(
    tmp_path, vision, question, path
):
    reply = h.collect(question, tmp_path, NOW)
    assert reply["paths"] == [path]
    assert reply["answer"].startswith("Animal tracking was removed, so I can't say")
    assert "(probably May, the only cat you've named)" in reply["answer"], "the pet's name reads its species"
    assert reply["evidence"] == []


@pytest.mark.parametrize(
    "question",
    ["what has Dana been doing lately?", "what has the team been up to this week?", "what is the plan?"],
)
def test_a_question_about_someone_else_is_left_to_the_other_paths(tmp_path, question):
    assert subject_activity.asked(question, tmp_path) is None


def recognition(
    monkeypatch, vision, *, on=False, sightings=(), pets=({"id": "p-may", "name": "May", "species": "cat"},)
):
    """The roster, the owner's labels (`pet.sighting`), and what people/pet_recognition.py's view says of each cat
    track: (track id, label, recognized)."""
    monkeypatch.setattr(subject_activity, "_pets", lambda state: [dict(p) for p in pets])
    from liberational.people import pet_recognition

    owner = {
        t: {"track_id": t, "label": label, "active": True, "source": "owner"} for t, label, _ in sightings if label
    }
    monkeypatch.setattr(pet_recognition, "labels", lambda state: owner)

    def view(state, roster, now=None):
        rows = [
            {"track_id": t, "species": "cat", "label": label, **({"recognized": {"label": seen}} if seen else {})}
            for t, label, seen in sightings
        ]
        return {"groups": [{"sightings": rows}], "recognition": [{"species": "cat", "on": on}]}

    monkeypatch.setattr(pet_recognition, "view", view)


@pytest.fixture
def litter(tmp_path, vision):
    """cam0 with a Litter box zone, watched for animals all week."""
    d = zones.define(tmp_path, "cam0", zones.parse_zones([{"name": "Litter box", "polygon": BOX}]), None, now=DAY)
    zone = d.zones[0].id
    vision.add(
        segment_log.ZONE_COVERAGE,
        ts_start=NOW - 7 * 86400,
        ts_end=NOW,
        cam="cam0",
        subject_kind="animal",
        definition=d.id,
        zones=[zone],
        frames=1000,
    )
    return vision, zone, d.id


def track(litter, track_id, hours_ago, species="cat"):
    """A minute-long visit to the Litter box by this track."""
    vision, zone, definition = litter
    a = NOW - hours_ago * H
    vision.add(
        segment_log.ZONE_VISITS,
        ts_start=a,
        ts_end=a + 60,
        cam="cam0",
        zone=zone,
        zone_name="Litter box",
        definition=definition,
        subject_kind="animal",
        track_id=track_id,
        label=species,
        dwell_s=60.0,
        frames=10,
        entry_seen=True,
        exit_seen=True,
        ended="left",
        candidate=True,
    )


LITTER = "when did May last use the litter box?"


def test_a_person_s_last_seen_question_stays_with_the_timeline(tmp_path):
    assert subject_activity.last_asked("when did I last see Dana?", tmp_path) is None
    assert subject_activity.last_asked("where is the remote?", tmp_path) is None


def test_with_two_named_cats_only_tracks_labeled_as_the_one_asked_about_count(tmp_path, litter, vision, monkeypatch):
    pets = ({"id": "p-may", "name": "May", "species": "cat"}, {"id": "p-tom", "name": "Tom", "species": "cat"})
    recognition(monkeypatch, vision, sightings=[("may-1", "p-may", None), ("tom-1", "p-tom", None)], pets=pets)
    track(litter, "may-1", 6)
    track(litter, "tom-1", 3)
    track(litter, "unknown", 1)
    reply = h.collect(LITTER, tmp_path, NOW)
    assert reply["answer"].startswith("May (the cat) was last in Litter box at 2026-10-23 at 14:01"), (
        "Tom's and the unlabeled"
    )
    assert "only sightings you labeled as May are counted" in reply["answer"]


def test_with_recognition_on_recognized_tracks_count_and_another_animal_never_does(
    tmp_path, litter, vision, monkeypatch
):
    recognition(
        monkeypatch,
        vision,
        on=True,
        sightings=[("seen-may", None, "p-may"), ("visitor", "other", None), ("stray", None, "other")],
    )
    track(litter, "seen-may", 6)
    track(litter, "visitor", 3)
    track(litter, "stray", 2)
    track(litter, "unrecognized", 1)
    reply = h.collect(LITTER, tmp_path, NOW)
    assert reply["answer"].startswith("May (the cat) was last in Litter box at 2026-10-23 at 14:01")
    assert "Only sightings labeled or recognized as May are counted." in reply["answer"]


def test_a_cat_labeled_as_another_animal_is_never_assumed_to_be_the_only_named_cat(
    tmp_path, litter, vision, monkeypatch
):
    recognition(monkeypatch, vision, sightings=[("visitor", "other", None)])
    track(litter, "may", 6)
    track(litter, "visitor", 1)
    reply = h.collect(LITTER, tmp_path, NOW)
    assert "last in Litter box at 2026-10-23 at 14:01" in reply["answer"]
    assert "except sightings you labeled as another animal" in reply["answer"]


@pytest.mark.parametrize(
    ("question", "routed"),
    [
        ("what has the cat been doing lately", True),
        ("where is May?", True),
        ("when did I last see the cat", True),
        ("where is the remote?", False),
        ("what is the plan?", False),
    ],
)
def test_voice_routes_animal_questions_to_history(tmp_path, monkeypatch, question, routed):
    from liberational import config as cfg

    with cfg.override({"paths": {"state": str(tmp_path)}}):
        assert h.is_history_question(question) is routed


@pytest.mark.parametrize(
    "question",
    [
        "when did I last see the cat yesterday?",
        "when did I last see the cat on camera 0?",
        "where is May now?",
        "when was the cat last seen this week",
    ],
)
def test_time_and_camera_qualifiers_are_not_part_of_the_animal(tmp_path, question):
    found = subject_activity.last_asked(question, tmp_path)
    assert found is not None and found[:2] == ("cat", "May")


def test_owner_labels_older_than_the_recognition_window_still_attribute_tracks(tmp_path, vision, monkeypatch):
    from liberational.people import pet_recognition

    pets = ({"id": "p-may", "name": "May", "species": "cat"}, {"id": "p-tom", "name": "Tom", "species": "cat"})
    monkeypatch.setattr(subject_activity, "_pets", lambda state: [dict(p) for p in pets])
    old = NOW - 60 * 86400  # past the view's 30 days of embeddings
    body = {"track_id": "may-old", "label": "p-may", "active": True, "source": "owner"}
    at = int(old * 1e9)
    record = knowledge.Record("019c0000-0000-7000-8000-0000000000aa", "pet.sighting", "may-old", at, at, at, body)
    knowledge.Store(tmp_path / "knowledge.sqlite3").put([record])
    monkeypatch.setattr(pet_recognition, "view", lambda state, roster, now=None: {"groups": [], "recognition": []})
    whose = subject_activity.attribute(tmp_path, "cat", "May", NOW)
    assert subject_activity.keeps(whose, "may-old")
    assert not subject_activity.keeps(whose, "someone-else")


@pytest.mark.parametrize(
    "question",
    ["where is Friday?", "when did I last see Friday yesterday?", "what has Friday been doing on Sunday"],
)
def test_a_pet_named_like_a_time_qualifier_is_still_that_pet(tmp_path, monkeypatch, question):
    monkeypatch.setattr(subject_activity, "_pets", lambda state: [{"name": "Friday", "species": "dog"}])
    found = subject_activity.last_asked(question, tmp_path) or subject_activity.asked(question, tmp_path)
    assert found is not None and found[:2] == ("dog", "Friday")
