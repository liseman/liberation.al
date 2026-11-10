"""Routine questions from explicitly named model candidates and human/animal zone visits."""

import datetime as dt
from zoneinfo import ZoneInfo

import pytest

import activity_annotations
from liberational import config as cfg
from liberational import segment_log, vision_log
from liberational.memory import event_counts, zone_counts
from liberational.memory import history_qa as h
from liberational.see import zones

LA = ZoneInfo("America/Los_Angeles")
DAY = dt.datetime(2026, 10, 23, tzinfo=LA).timestamp()  # local midnight
NOW = DAY + 20 * 3600
H = 3600.0
BOX = [[0.1, 0.6], [0.3, 0.6], [0.3, 0.9], [0.1, 0.9]]
OWNER = "p-sam"


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    monkeypatch.setattr(h, "TZ", LA)
    zone_counts._NAMES.clear()
    event_counts._NAMES.clear()
    from liberational.people import registry

    class Registry:
        def owner(self):
            return OWNER

    monkeypatch.setattr(registry, "at", lambda state=None: Registry())
    with cfg.override({"activity": {"promotion_min_ratings": 2}}):
        yield


def night(n: int) -> float:
    """22:00 local on the date n days before DAY."""
    return dt.datetime.fromtimestamp(DAY, LA).replace(hour=22).timestamp() - n * 86400.0


def watch(vision, a: float, b: float) -> None:
    vision.add(segment_log.WALKBY_COVERAGE, ts_start=a, ts_end=b, cam="cam0", frames=1000)


def span(
    state,
    vision,
    at: float,
    kind: str = "crossing",
    person: str | None = OWNER,
    rated: str | None = "right",
    name: str = "reading a book",
    cam: str = "cam0",
):
    row = vision.add(
        segment_log.WALKBY_SPANS, ts_start=at, ts_end=at + 3, cam=cam, kind=kind, person=person, candidate=True
    )
    activity_annotations.name(state, row, name)
    if rated:
        activity_annotations.rate(state, row, rated)
    return row


def test_usual_named_model_activity_counts_only_my_nighttime_events(tmp_path, vision):
    for n in (1, 2, 3):
        watch(vision, night(n), night(n) + 9 * H)
    span(tmp_path, vision, night(3) + 2 * H)  # 00:00
    span(tmp_path, vision, night(3) + 6 * H)  # 04:00
    span(tmp_path, vision, night(2) + 3 * H)
    span(tmp_path, vision, night(2) + 4 * H, person="p-dana")  # someone else reading
    span(tmp_path, vision, night(2) + 5 * H, person=None)  # nobody recognized: not counted as mine
    span(tmp_path, vision, night(1) - 6 * H)  # 16:00, a daytime reading session
    reply = h.collect("how many times do I usually read a book at night", tmp_path, NOW)
    assert reply["paths"] == ["events", "typical"]
    assert reply["answer"].startswith(
        "reading a book: typically 1 a night (median), from 0 to 2, over 3 nights with footage"
    )
    assert [p["count"] for p in reply["coverage"]["events"]["periods"]] == [2, 1, 0]


def test_unrated_named_model_candidates_are_not_a_verified_count(tmp_path, vision):
    watch(vision, DAY, NOW)
    span(tmp_path, vision, DAY + 9 * H, rated=None)
    reply = h.collect("how many times did I read a book today", tmp_path, NOW)
    assert reply["answer"].startswith("I found unverified candidates named reading a book")
    assert reply["coverage"]["events"]["count_available"] is False
    assert reply["coverage"]["events"]["unverified"] == 1


def test_single_window_counts_named_model_events_with_times_and_camera_filter(tmp_path, vision):
    watch(vision, DAY, NOW)
    span(tmp_path, vision, DAY + 9 * H)
    span(tmp_path, vision, DAY + 10 * H)
    span(tmp_path, vision, DAY + 11 * H, name="opening a parcel")
    span(tmp_path, vision, DAY + 12 * H, cam="cam1")
    reply = h.collect("how many times did I read a book today on camera cam0", tmp_path, NOW)
    assert reply["answer"].startswith("reading a book: 2 times (1 at 09:00, 1 at 10:00)")
    assert reply["query"]["cameras"] == ["cam0"]


@pytest.mark.parametrize(
    ("question", "what"),
    [("how many times did I get up today", "stand up"), ("how often do I usually sit down", "sit down")],
)
def test_an_unrated_model_kind_is_not_counted_and_says_how_to_check_it(tmp_path, vision, question, what):
    vision.add(segment_log.ACTION_COVERAGE, ts_start=DAY, ts_end=NOW, cam="cam0", frames=1000, model="test")
    kind = what.replace(" ", "_")
    vision.add(segment_log.ACTION_SPANS, ts_start=DAY + 9 * H, ts_end=DAY + 9 * H + 2, cam="cam0", kind=kind,
               candidate=True, track_id="1", subject_type="person", camera_moved=False)  # fmt: skip
    reply = h.collect(question, tmp_path, NOW)
    assert f"I can't count {what} yet" in reply["answer"]
    assert f"Rate {cfg.get('activity.promotion_min_ratings')} more" in reply["answer"]
    assert "times" not in reply["answer"].split(".")[0]


def test_how_often_the_cat_usually_uses_the_litter_box_is_a_typical_day_of_its_visits(tmp_path, vision):
    d = zones.define(tmp_path, "cam0", zones.parse_zones([{"name": "Litter box", "polygon": BOX}]), None, now=DAY)
    zone = d.zones[0].id
    for n in (1, 2, 3):
        a = DAY - n * 86400
        vision.add(
            vision_log.ZONE_COVERAGE,
            ts_start=a,
            ts_end=a + 86400,
            cam="cam0",
            subject_kind="animal",
            definition=d.id,
            zones=[zone],
            frames=1000,
        )
    for n, visits in ((3, 4), (2, 2), (1, 3)):
        for k in range(visits):
            at = DAY - n * 86400 + (8 + 2 * k) * H
            vision.add(
                vision_log.ZONE_VISITS,
                ts_start=at,
                ts_end=at + 60,
                cam="cam0",
                zone=zone,
                zone_name="Litter box",
                definition=d.id,
                subject_kind="animal",
                track_id=f"t{n}{k}",
                label="cat",
                dwell_s=60.0,
                ended="left",
                candidate=True,
            )
    reply = h.collect("how often does the cat usually use the litter box", tmp_path, NOW)
    assert reply["paths"] == ["events", "typical"]
    assert reply["answer"].startswith("Litter box: typically 3 a day (median), from 2 to 4, over 3 days with footage")


FACE = [0.2, 0.32, 0.3, 0.45]  # a face inside it


@pytest.fixture
def bathroom(tmp_path, vision, monkeypatch):
    """A Bathroom zone on cam0 covered for people all day, and a registry where a name is "p-" plus it."""
    from liberational.people import registry

    class Registry:
        def owner(self):
            return OWNER

        def person_id(self, name):
            return f"p-{name.lower()}"

    monkeypatch.setattr(registry, "at", lambda state=None: Registry())
    d = zones.define(tmp_path, "cam0", zones.parse_zones([{"name": "Bathroom", "polygon": BOX}]), None, now=DAY)
    zone = d.zones[0].id
    vision.add(
        vision_log.ZONE_COVERAGE,
        ts_start=night(3),
        ts_end=NOW,
        cam="cam0",
        subject_kind="person",
        definition=d.id,
        zones=[zone],
        frames=1000,
    )

    def trip(at: float, track: str, face: str | None, face_at: float = 0.0) -> None:
        """A 2 min Bathroom visit on `track`; `face` matched on that same track `face_at` s from the start."""
        if face:
            vision.face(at + face_at, "cam0", face, bbox=FACE, track_id=track)
        vision.add(
            vision_log.ZONE_VISITS,
            ts_start=at,
            ts_end=at + 120,
            cam="cam0",
            zone=zone,
            zone_name="Bathroom",
            definition=d.id,
            subject_kind="person",
            track_id=track,
            label="person",
            dwell_s=120.0,
            ended="left",
            candidate=True,
        )

    return tmp_path, trip


def test_my_bathroom_trips_are_the_visits_my_face_was_seen_on(bathroom):
    state, trip = bathroom
    trip(DAY + 8 * H, "t1", "Sam", face_at=60)  # my face in the bathroom doorway
    trip(DAY + 9 * H, "t2", "Sam", face_at=-90)  # seen walking up 90 s before going in: the same track
    trip(DAY + 10 * H, "t3", "Dana", face_at=30)  # someone else
    trip(DAY + 11 * H, "t4", None)  # no face ever seen: can't say whose
    reply = h.collect("how many times did I go to the bathroom today", state, NOW)
    text = reply["answer"]
    assert text.startswith("Bathroom: 2 visits of at least 10 s by you (08:00, 09:00).")
    assert "1 more visit by someone no face named on the camera (11:00) is not counted." in text
    assert "can't tell people apart" not in text


def test_a_track_two_people_s_faces_were_seen_on_is_nobody_s(bathroom, vision):
    state, trip = bathroom
    trip(DAY + 8 * H, "t1", "Sam", face_at=10)
    vision.face(DAY + 8 * H + 10, "cam0", "Dana", bbox=FACE, track_id="t1")  # a second face on the same track
    reply = h.collect("how many times did I go to the bathroom today", state, NOW)
    assert reply["answer"].startswith("Bathroom: no visit of at least 10 s by you in the checked footage.")
    assert "1 more visit by someone no face named" in reply["answer"]


def test_my_typical_night_counts_only_my_trips(bathroom):
    state, trip = bathroom
    for n, mine, theirs in ((1, 2, 1), (2, 1, 2), (3, 0, 0)):
        for k in range(mine):
            trip(night(n) + (2 + k) * H, f"m{n}{k}", "Sam", face_at=5)
        for k in range(theirs):
            trip(night(n) + (5 + k) * H, f"d{n}{k}", "Dana", face_at=5)
    reply = h.collect("how many times do I usually go to the bathroom at night", state, NOW)
    assert [p["count"] for p in reply["coverage"]["events"]["periods"]] == [0, 1, 2]
