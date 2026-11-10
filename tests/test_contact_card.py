"""The contact card: face shot quality (vision/face_quality.py), the best shots and the main photo
(vision/sightings.py, over the face worker's faces on the vision log), and GET /people/<name>/card. Made-up names and
synthetic faces; never the real data/."""

import datetime as dt
import http.client
import json
import time
from types import SimpleNamespace
from typing import cast
from unittest.mock import create_autospec

import cv2
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as hs

import speech_fake
from liberational import config as cfg
from liberational import segment_log, speech_log, vision_log
from liberational.dashboard import server
from liberational.memory import clip_labels
from liberational.people import face_shots, registry, video_thumbs
from liberational.vision import face_quality as fq
from liberational.vision import sightings as st
from vision_fake import FakeVisionLog

# Five landmarks of a face looking straight at the camera: eyes 40 px apart, nose centred a little over halfway
# down to the mouth (22 of 40 px), mouth corners below.
STRAIGHT = [(30, 40), (70, 40), (50, 62), (35, 80), (65, 80)]


def turned(nose_x, nose_y=62):
    return [STRAIGHT[0], STRAIGHT[1], (nose_x, nose_y), STRAIGHT[3], STRAIGHT[4]]


@pytest.mark.parametrize(
    ("kps", "want"),
    [
        pytest.param(STRAIGHT, 1.0, id="straight-on"),
        pytest.param(turned(52), 1.0, id="a-slight-turn-is-still-straight"),
        pytest.param(turned(61), 0.5, id="half-turned"),  # nose 11 px off = yaw 0.275, halfway from 0.05 to 0.5
        pytest.param(turned(70), 0.0, id="profile"),  # nose under the far eye: yaw 0.5
        pytest.param(turned(50, 76), 1 - 0.25 / 0.35, id="tipped-down"),  # nose 36 of 40 px down: pitch 0.35
        pytest.param([STRAIGHT[1], STRAIGHT[0], *STRAIGHT[2:]], 0.0, id="eyes-swapped"),
        pytest.param([(30, 40)] * 5, 0.0, id="all-on-one-point"),
        pytest.param(None, 0.0, id="no-landmarks"),
        pytest.param([(1, 2), (3, 4)], 0.0, id="too-few-landmarks"),
        pytest.param([(float("nan"), 40), *STRAIGHT[1:]], 0.0, id="nan"),
    ],
)
def test_frontal_reads_the_turn_from_the_landmarks(kps, want):
    assert fq.frontal(kps) == pytest.approx(want)


def test_a_sharp_face_scores_above_the_same_face_blurred():
    rng = np.random.default_rng(1)
    face = (rng.random((96, 96)) * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(face, (0, 0), 4)
    assert fq.sharpness(face) > 10 * fq.sharpness(blurred)
    assert fq.sharpness(np.full((96, 96), 128, np.uint8)) == 0.0  # a flat patch has no edges
    assert fq.crop_sharpness(np.zeros((10, 10, 3), np.uint8), (20, 20, 30, 30)) == 0.0  # box outside the frame


BIG = (0, 0, 112, 112)


@pytest.mark.parametrize(
    ("args", "want"),
    [
        pytest.param((0.9, BIG, 0.75, 1.0, 150.0), 0.5**0.2, id="all-best-but-sharpness-at-its-half"),
        pytest.param((0.9, BIG, 0.75), 0.5**0.5, id="unmeasured-frontal-and-sharp-count-as-half"),
        pytest.param((0.49, BIG, 0.75, 1.0, 1e9), 0.0, id="below-the-detection-floor"),
        pytest.param((0.9, BIG, 0.75, 0.0, 1e9), 0.0, id="a-profile-is-never-chosen"),
        pytest.param((0.9, (0, 0, 24, 200), 0.75, 1.0, 1e9), 0.0, id="too-thin-a-face"),
        pytest.param((0.9, BIG, 0.45, 1.0, 1e9), 0.0, id="at-the-match-threshold"),
        pytest.param((float("nan"), BIG, 0.75, 1.0, 1e9), 0.0, id="nan-detection"),
        pytest.param((0.7, (0, 0, 68, 68), 0.6, 0.5, 150.0), 0.5, id="every-factor-at-half"),
    ],
)
def test_score_is_the_weighted_geometric_mean(args, want):
    assert fq.score(*args) == pytest.approx(want)


unit = hs.floats(0, 1)


@given(det=hs.floats(0.5, 1), side=hs.floats(0, 300), sim=hs.floats(0.4, 1), f=unit, g=unit, sharp=hs.floats(0, 1e4))
def test_a_better_factor_never_lowers_the_score(det, side, sim, f, g, sharp):
    box = (0, 0, side, side)
    lo, hi = sorted((f, g))
    a, b = fq.score(det, box, sim, lo, sharp), fq.score(det, box, sim, hi, sharp)
    assert 0 <= a <= b <= 1


# ---- picking shots: the best by the quality the face worker scored ----
# Days within sightings.SHOT_DAYS of now, so the readers that look back that far find them.
DAY1, DAY2, DAY3 = (
    dt.datetime.combine(dt.date.today() - dt.timedelta(days=d), dt.time(12)).timestamp() for d in (6, 5, 4)
)


def best_id(state, *args, **kw) -> str | None:
    """The id of Alice's best shot, or None."""
    b = st.best("Alice", state, *args, **kw)
    return None if b is None else b["id"]


def shot(log: FakeVisionLog, ts: float, quality: float, name: str | None = "Alice", cam: str = "cam0") -> str:
    """One face of `name` found at ts with that quality; its sighting id."""
    return log.face(ts, cam, name, 0.75, quality=quality)["id"]


def photo(log: FakeVisionLog, ts: float, quality: float, name: str = "Alice", cam: str = "cam0") -> str:
    """shot(), in a visit of its own: a face People can pick as a photo (photos come from visits)."""
    log.visit(name, ts, ts, cam)
    return shot(log, ts, quality, name, cam)


def test_the_main_photo_looks_at_the_camera(tmp_path, vision):
    straight = photo(vision, DAY1, fq.score(0.9, BIG, 0.75, 1.0, 600.0))
    photo(vision, DAY1 + 1, fq.score(0.99, BIG, 0.75, 0.1, 5000.0))  # sharper, more confident, but turned away
    assert best_id(tmp_path) == straight


def test_a_shot_with_no_usable_face_is_never_chosen(tmp_path, vision):
    photo(vision, DAY1, fq.score(0.9, BIG, 0.75, 0.0, 900.0))  # a profile
    photo(vision, DAY1 + 1, fq.score(0.3, BIG, 0.75, 1.0, 900.0))  # below the detection floor
    assert best_id(tmp_path) is None
    fine = photo(vision, DAY1 + 2, 0.4)
    assert best_id(tmp_path) == fine


def test_best_can_be_limited_to_one_visit(tmp_path, vision):
    shot(vision, DAY1, 0.9)
    shot(vision, DAY1 + 100, 0.6, cam="cam1")
    in_visit = shot(vision, DAY1 + 100, 0.5)
    assert best_id(tmp_path, cam="cam0", start=DAY1 + 50, end=DAY1 + 150) == in_visit


def test_shots_spread_over_days_then_fill_with_distinct_moments(tmp_path, vision):
    a = photo(vision, DAY1, 0.9)
    photo(vision, DAY1 + 10, 0.8)  # 10 s after a on the same camera: looks the same, skipped
    c = photo(vision, DAY1 + 10, 0.7, cam="cam1")  # another camera: kept
    d = photo(vision, DAY1 + 120, 0.1)
    e = photo(vision, DAY2, 0.3)
    f = photo(vision, DAY3, 0.2)
    assert [s["id"] for s in st.best_spread("Alice", 6, tmp_path)] == [a, c, e, f, d]
    assert [s["id"] for s in st.best_spread("Alice", 2, tmp_path)] == [a, e]  # one per day while days last


def test_last_seen_takes_the_newest_camera_across_capitals(tmp_path, vision):
    vision.visit("Alice", 50, 100, cam="cam0")
    vision.visit("alice", 150, 200, cam="cam1")
    assert st.last_seen_where(tmp_path) == {"alice": (200, "cam1")}


# ---- GET /people/<name>/card ----
NO_FACES = SimpleNamespace(counts=lambda: {"Alice": 3}, unknown_groups=list, rename=lambda a, b: 0)
SHOTS = SimpleNamespace(early="", best="")  # the fixture's two shots of Alice, by sighting id


@pytest.fixture
def card_state(tmp_path, monkeypatch, vision):
    """Alice: two recorded shots, a visit on cam0 with a caption, two room lines on two mics, and in view now on
    cam1."""
    tmp = tmp_path
    (tmp / "speakers.json").write_text(json.dumps({"speakers": []}))
    for name, path in {
        "STATE": ".",
        "CALLS_DIR": "calls",
        "CONTACTS_FILE": "contacts.json",
        "PEOPLE_ADDED": "people_added.json",
    }.items():
        monkeypatch.setattr(server, name, tmp / path)
    monkeypatch.setattr(server, "_here_since", {})  # each test starts with no one in view
    monkeypatch.setattr(server, "_here_cam", {})
    monkeypatch.setattr(server, "_faces_mod", lambda: NO_FACES)
    monkeypatch.setattr(server, "_face_last_seen_at", lambda: {"alice": (DAY1 + 50, "cam0")})
    monkeypatch.setattr(SHOTS, "early", shot(vision, DAY1 + 5, 0.3))
    monkeypatch.setattr(SHOTS, "best", shot(vision, DAY1 + 40, 0.9))
    face_shots.write(tmp, SHOTS.early, "0", b"ready-jpeg")
    face_shots.write(tmp, SHOTS.best, "0", b"ready-jpeg")
    vision.visit("Alice", DAY1, DAY1 + 50, cam="cam0", best_sim=0.8, samples=9)
    video_thumbs.write(tmp, "cam0", DAY1, "0", b"ready-scene-jpeg")
    vision.caption(DAY1 + 25, "cam0", "A woman pours coffee. The kettle is on.")  # the clip's caption
    lines = [
        {"ts_start": DAY1 + 10, "ts_end": DAY1 + 12, "text": "Morning!", "speaker": "Alice", "device": "Desk Mic"},
        {"ts_start": DAY1 + 20, "ts_end": DAY1 + 21, "text": "  ", "speaker": "Alice"},  # no words: left out
        {"ts_start": DAY1 + 30, "ts_end": DAY1 + 33, "text": "Coffee?", "speaker": "Alice"},
        {"ts_start": DAY1 + 31, "ts_end": DAY1 + 32, "text": "Yes.", "speaker": "Bob", "device": "Desk Mic"},
    ]
    speech_fake.fake().write(lines)
    shot(vision, time.time() - 5, 0.0, cam="cam1")  # in view now (quality 0, so never a card shot)
    # The clip labels' model, faked: it replies in the first person, as the journal would, for the clip at DAY1 + 25.
    replies = {"A woman pours coffee. The kettle is on.": "Pouring coffee, kettle on."}
    labels = clip_labels.Labels(tmp, lambda text: replies.get(text, "I poured my coffee."), background=False)
    monkeypatch.setitem(clip_labels._labels, str(tmp), labels)
    with cfg.override({"owner": {"name": "Olive"}, "paths": {"state": str(tmp)}}):
        yield tmp


def test_the_card_has_the_best_shots_recent_video_and_voice(card_state):
    first = server.person_card("ALICE")
    assert first is not None
    assert first["videos"][0]["what"] is None  # the label isn't written yet: the card won't wait
    clip_labels.at(card_state).drain()
    card = server.person_card("ALICE")
    assert card is not None
    best = {"id": SHOTS.best, "url": f"/faces/shot/{SHOTS.best}.jpg", "ts": DAY1 + 40, "cam": "cam0", "score": 0.9}
    assert card["name"] == "Alice"
    assert card["photo"] == best
    assert [s["id"] for s in card["shots"]] == [SHOTS.best]  # the early one is 35 s before it on the same camera
    assert card["videos"] == [
        {
            "id": f"cam0@{DAY1:.2f}",  # for the card's menu (POST /people/correct)
            "cam": "cam0",
            "place": "cam0",  # no vision.camera_names: the id
            "start": DAY1,
            "end": DAY1 + 50,
            "duration_s": 50.0,
            "thumb": f"/people/video-thumb.jpg?cam=cam0&start={DAY1:.3f}&end={DAY1 + 50:.3f}",
            "what": "Pouring coffee, kettle on.",
            "review": f"/review#person=Alice&day={dt.date.fromtimestamp(DAY1).isoformat()}&t={DAY1:.2f}",
        }
    ]
    assert card["voices"] == [
        {"line": registry.line_key(DAY1 + 30), "ts_start": DAY1 + 30, "text": "Coffee?", "mic": None},
        {"line": f"desk-mic@{DAY1 + 10:.3f}", "ts_start": DAY1 + 10, "text": "Morning!", "mic": "Desk Mic"},
    ]
    assert card["last_seen"] == {"ts": DAY1 + 50, "cam": "cam0", "place": "cam0"}
    assert card["here_now"] == {"cam": "cam1", "place": "cam1", "since": scene_ts(card_state)}


def test_card_hides_uncached_faces_and_reports_processing_without_decoding(card_state, monkeypatch):
    face_shots.path(card_state, SHOTS.early, "0").unlink()
    face_shots.path(card_state, SHOTS.best, "0").unlink()
    jobs = face_shots.Pending(server._crop_process)
    monkeypatch.setattr(server, "_crop_jobs", jobs)
    card = server.person_card("Alice")
    assert card is not None
    assert card["shots"] == []
    assert card["photo"] is None
    assert card["shots_pending"] == 2
    assert len(jobs.pending) == 2  # the repeated roster/main-photo reads coalesce
    face_shots.write(card_state, SHOTS.best, "0", b"ready-jpeg")
    refreshed = server.person_card("Alice")
    assert refreshed is not None
    assert [s["id"] for s in refreshed["shots"]] == [SHOTS.best]
    assert refreshed["photo"]["id"] == SHOTS.best
    assert refreshed["shots_pending"] == 1


def test_a_first_person_caption_rewrite_never_reaches_the_card(card_state, vision):
    vision.records[vision_log.CAPTIONS][0]["caption"] = (
        "Foreground: man, seated, hand over mouth."  # the fake says "I ..."
    )
    server.person_card("Alice")
    clip_labels.at(card_state).drain()
    card = server.person_card("Alice")
    assert card is not None
    assert card["videos"][0]["what"] is None  # only the time and place show


def test_the_roster_shows_the_main_photo_and_where_they_are(card_state):
    alice = next(p for p in server.people_roster() if p["name"] == "Alice")
    assert alice["photo"] == f"/faces/shot/{SHOTS.best}.jpg"
    assert (alice["last_seen"], alice["last_seen_cam"], alice["here_now"]) == (
        DAY1 + 50,
        "cam0",
        {"cam": "cam1", "place": "cam1", "since": scene_ts(card_state)},
    )


def test_someone_tagged_live_gets_their_saved_crop(card_state, monkeypatch):
    with cfg.override({"paths": {"faces": str(card_state / "faces")}}):
        d = server._photo_dir("Bob")
        d.mkdir(parents=True)
        (d / "1793212345000.jpg").write_bytes(b"jpg")
        assert server.main_photo("Bob") == {
            "id": None,
            "url": "/people/Bob/photos/1793212345000.jpg",
            "ts": 1793212345.0,
            "cam": None,
            "score": None,
        }
        assert server.main_photo("Carol") is None


def fake(state) -> FakeVisionLog:
    return cast(FakeVisionLog, vision_log.at(state))  # the `vision` fixture's, through card_state


def scene_ts(state):
    """When the fixture saw Alice in view."""
    return next(r["ts_start"] for r in fake(state).records[segment_log.FACE_MATCHES] if r["cam"] == "cam1")


def seen(state, *frames):
    """Alice's faces in the last hour are now just these, each (time, camera)."""
    log = fake(state)
    stream = segment_log.FACE_MATCHES
    log.records[stream] = [r for r in log.records.get(stream, []) if r["ts_start"] < time.time() - 3600]
    for t, c in frames:
        shot(log, t, 0.0, cam=c)


def here(name="Alice"):
    me = next(p for p in server.people_roster() if p["name"] == name)
    return me["here_now"] and (me["here_now"]["cam"], me["here_now"]["since"])


def test_here_now_says_since_when_their_run_on_that_camera_began(card_state):
    now = time.time()
    seen(card_state, (now - 90, "cam1"), (now - 65, "cam1"), (now - 40, "cam1"), (now - 15, "cam1"))
    assert here() == ("cam1", now - 90)  # in view all along: no gap over 30 s


def test_here_now_starts_a_new_run_after_a_gap(card_state):
    now = time.time()
    seen(card_state, (now - 80, "cam1"), (now - 10, "cam1"))
    assert here() == ("cam1", now - 10)  # gone for 70 s in between


def test_here_now_remembers_a_run_longer_than_one_read_covers(card_state):
    now = time.time()
    seen(card_state, (now - 25, "cam1"), (now - 5, "cam1"))
    assert here() == ("cam1", now - 25)
    seen(card_state, (now - 5, "cam1"), (now - 3, "cam1"))  # the face at now - 25 has left the window read
    assert here() == ("cam1", now - 25)


def test_here_now_follows_them_to_the_camera_that_saw_them_last(card_state):
    now = time.time()
    seen(card_state, (now - 20, "cam0"), (now - 4, "cam1"))
    assert here() == ("cam1", now - 4)


def history(state, cam, times, name="Alice"):
    """Faces of `name` found on cam at each time."""
    for t in times:
        shot(fake(state), t, 0.0, name, cam)


def test_here_now_finds_when_a_run_began_in_the_faces_found(card_state):
    now = time.time()
    seen(card_state, (now - 5, "cam1"))
    history(card_state, "cam1", [now - 5000 + 10 * i for i in range(500)])  # every 10 s for more than an hour
    assert here() == ("cam1", now - 5000)


def test_here_now_stops_looking_back_at_a_gap_in_the_faces_found(card_state):
    now = time.time()
    seen(card_state, (now - 5, "cam1"))
    history(card_state, "cam1", [now - 900, now - 890, *(now - 300 + 10 * i for i in range(30))])  # gone 590 s before
    assert here() == ("cam1", now - 300)


def test_here_now_keeps_someone_on_the_camera_they_were_shown_on(card_state):
    now = time.time()
    seen(card_state, (now - 20, "cam0"), (now - 4, "cam1"))
    assert here()[0] == "cam1"
    seen(card_state, (now - 3, "cam1"), (now - 2, "cam0"))  # cam0 saw them last this time
    assert here()[0] == "cam1"


def test_a_stale_scene_is_not_here_now(card_state):
    seen(card_state, (time.time() - 120, "cam1"))
    card = server.person_card("Alice")
    assert card is not None
    assert card["here_now"] is None


def test_nobody_gets_an_empty_card_and_a_bad_name_none(card_state):
    assert server.person_card("Carol") == {
        "name": "Carol",
        "photo": None,
        "shots": [],
        "videos": [],
        "voices": [],
        "last_seen": None,
        "here_now": None,
    }
    assert server.person_card("a/b") is None


def test_the_card_and_the_roster_say_where_each_camera_is(card_state):
    with cfg.override({"vision": {"camera_names": ["cam0: living room"]}}):  # cam1 has no place: its id
        card = server.person_card("Alice")
        alice = next(p for p in server.people_roster() if p["name"] == "Alice")
    assert card is not None
    assert [v["place"] for v in card["videos"]] == ["living room"]
    assert card["last_seen"] == {"ts": DAY1 + 50, "cam": "cam0", "place": "living room"}
    assert card["here_now"] == {"cam": "cam1", "place": "cam1", "since": scene_ts(card_state)}
    assert (alice["last_seen_place"], alice["here_now"]) == (
        "living room",
        {"cam": "cam1", "place": "cam1", "since": scene_ts(card_state)},
    )


def test_a_recorded_visit_gets_a_scene_thumbnail_without_a_quality_face(card_state, monkeypatch):
    monkeypatch.setattr(st, "best", lambda *args, **kwargs: None)
    card = server.person_card("Alice")
    assert card is not None
    assert card["videos"][0]["thumb"].startswith("/people/video-thumb.jpg?cam=cam0&start=")


def test_card_reads_retained_speech_once_and_reuses_it_between_requests(card_state, monkeypatch, speech):
    reads = []
    original = speech.all_lines

    def read(streams=None):
        reads.append(streams)
        return original(streams)

    monkeypatch.setattr(speech, "all_lines", read)
    first = server.person_card("Alice")
    second = server.person_card("Alice")
    assert first is not None and second is not None
    assert [v["text"] for v in first["voices"]] == ["Coffee?", "Morning!"]
    assert second["voices"] == first["voices"]
    assert len(reads) == 1  # one speech read, not separate roster/voice tail reads


def test_card_names_the_retained_tail_once_without_losing_voice_samples(card_state, monkeypatch):
    batches = []
    original = server.transcript_view.View.named

    def named(view, items):
        batches.append(list(items))
        return original(view, batches[-1])

    monkeypatch.setattr(server.transcript_view.View, "named", named)
    server.people_view(wait_s=None)  # the list a person is opened from; its first read creates the registry,
    server.people_view(wait_s=None)  # which moves the list's key once in a new state
    batches.clear()
    card = server.person_card("Alice")
    assert card is not None
    assert [v["text"] for v in card["voices"]] == ["Coffee?", "Morning!"]
    assert len(batches) == 1  # the card names the tail once for voice samples; its roster row is the list's read
    assert len(batches[0]) == 4


def test_card_cached_speech_sees_a_new_line(card_state, speech):
    first = server.person_card("Alice")
    speech.say({"ts_start": DAY1 + 60, "text": "Another cup?", "speaker": "Alice"})
    second = server.person_card("Alice")
    assert first is not None and second is not None
    assert [v["text"] for v in second["voices"]] == ["Another cup?", "Coffee?", "Morning!"]
    assert [v["text"] for v in first["voices"]] == ["Coffee?", "Morning!"]


def test_card_cached_speech_uses_current_names_without_new_speech(card_state):
    reg = registry.at(card_state)
    reg.ensure("Alice", "test", at=DAY1 - 1)
    first = server.person_card("Alice")
    reg.rename("Alice", "Alicia", "test")
    renamed = server.person_card("Alicia")
    old = server.person_card("Alice")
    assert first is not None and renamed is not None and old is not None
    assert renamed["name"] == "Alicia"
    assert [v["text"] for v in renamed["voices"]] == ["Coffee?", "Morning!"]
    assert old["voices"] == []


def test_card_cached_speech_drops_purged_voice_samples(card_state, speech):
    first = server.person_card("Alice")
    speech.purge(speech_log.UTTERANCES, lambda e: e["ts_start"] == DAY1 + 30)
    second = server.person_card("Alice")
    assert first is not None and second is not None
    assert [v["text"] for v in second["voices"]] == ["Morning!"]
    assert [v["text"] for v in first["voices"]] == ["Coffee?", "Morning!"]


def test_opening_a_person_reads_their_row_from_the_list_without_building_a_roster(card_state, monkeypatch):
    server.people_view(wait_s=None)  # the list a person is opened from; its first read creates the registry,
    server.people_view(wait_s=None)  # which moves the list's key once in a new state
    listed = server._PEOPLE_READS.last
    assert listed is not None
    (alice,) = [p for p in listed["people"] if p["name"] == "Alice"]
    memory = card_state / "memory.jsonl"
    memory.write_text(
        "".join(
            json.dumps(e) + "\n"
            for e in (
                {"op": "remember", "text": "Alice likes tea"},
                {"op": "remember", "text": "Alice is in Lisbon this week"},
                {"op": "forget", "text": "Lisbon"},
            )
        )
    )
    monkeypatch.setattr(server, "MEMORY_FILE", memory)
    monkeypatch.setattr(server, "people_roster", create_autospec(server.people_roster, side_effect=AssertionError))

    summary = server.person_summary("alice")
    card = server.person_card("Alice")

    assert summary is not None and card is not None
    assert (summary["name"], summary["lines"], summary["room_lines"], summary["call_lines"]) == ("Alice", 2, 2, 0)
    assert (alice["lines"], alice["room"], alice["calls"]) == (2, 2, 0)  # the same counts the list shows
    assert summary["memory"] == ["Alice likes tea"]
    assert card["here_now"] == alice["here_now"] and card["here_now"] is not None


def _counted_card(monkeypatch):
    """person_card with a count of its reads."""
    reads = []
    original = server.person_card

    def read(name):
        reads.append(name)
        return original(name)

    monkeypatch.setattr(server, "person_card", read)
    return reads


def _settle(cards):
    """Wait out the cache's background reads."""
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        with cards.lock:
            if not cards.reading and not cards.warming:
                return
        time.sleep(0.01)
    raise AssertionError("background card reads did not finish")


def test_a_card_opened_again_answers_from_the_last_read_and_refreshes_it_in_the_background(card_state, monkeypatch):
    server.people_view(wait_s=None)  # the list a person is opened from; its first read creates the registry,
    server.people_view(wait_s=None)  # which moves the keys once in a new state
    reads = _counted_card(monkeypatch)
    cards = server._CARD_READS
    first = cards.get("Alice")
    again = cards.get("alice")
    assert first is not None and again == first
    assert reads == ["Alice"]  # the second open read nothing
    assert "refreshing" not in first and "refreshing" not in again
    monkeypatch.setattr(server, "CARD_REFRESH_S", 0.0)  # now the cached card is old
    assert cards.get("Alice") == {**first, "refreshing": True}  # answered from it, marked so the page asks again
    _settle(cards)
    assert reads == ["Alice", "Alice"]  # and read again behind it
    monkeypatch.setattr(server, "CARD_REFRESH_S", 3600.0)
    assert cards.get("Alice") == first  # the one read behind it, no longer marked
    assert reads == ["Alice", "Alice"]


def test_the_people_list_reads_ahead_only_cards_it_has_none_of_however_old_the_rest(card_state, monkeypatch):
    server.people_view(wait_s=None)  # the list a person is opened from; its first read creates the registry,
    server.people_view(wait_s=None)  # which moves the keys once in a new state
    cards = server._CARD_READS
    assert cards.get("Alice") is not None
    reads = _counted_card(monkeypatch)
    monkeypatch.setattr(server.time, "time", lambda: 1e12)  # every cached card is now very old
    cards.warm(["Alice", "Bob"])
    _settle(cards)
    assert reads == ["Bob"]  # Alice's open reads hers again behind it, marked refreshing


def test_a_card_read_that_began_before_an_edit_is_never_kept(card_state, monkeypatch):
    cards = server._CARD_READS
    original = server.person_card

    def read_across_an_edit(name):
        card = original(name)
        cards.clear()  # a People POST landed while this read was running
        return card

    monkeypatch.setattr(server, "person_card", read_across_an_edit)
    assert cards.get("Alice") is not None
    assert cards.last == {}


def test_the_people_list_reads_cards_ahead_so_the_first_open_is_cached(card_state, monkeypatch, serve):
    base = serve(server.H)
    c = http.client.HTTPConnection(base.removeprefix("http://"))
    c.request("GET", "/people", headers={"Accept": "application/json"})
    assert c.getresponse().status == 200
    c.close()
    _settle(server._CARD_READS)
    reads = _counted_card(monkeypatch)
    assert server._CARD_READS.get("Alice") is not None
    assert reads == []


def test_a_photo_hidden_from_the_card_is_gone_from_the_next_open(card_state, monkeypatch, serve):
    monkeypatch.setattr(server, "_card_key", lambda: ("unchanged",))  # only the POST's clear can show the change
    monkeypatch.setattr(server.H, "_authed", lambda self: True)
    base = serve(server.H)

    def request(method, path, body=None):
        c = http.client.HTTPConnection(base.removeprefix("http://"))
        c.request(method, path, json.dumps(body) if body else None, {"Content-Type": "application/json"})
        r = c.getresponse()
        out = (r.status, json.loads(r.read()))
        c.close()
        return out

    status, card = request("GET", "/people/Alice/card")
    assert status == 200 and card["shots"]
    shot = card["shots"][0]["id"]
    status, reply = request("POST", "/people/photo", {"name": "Alice", "action": "hide", "shot": shot})
    assert status == 200 and reply["ok"], reply
    status, card = request("GET", "/people/Alice/card")
    assert status == 200
    assert shot not in [s["id"] for s in card["shots"]]
