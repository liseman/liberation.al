import datetime as dt
import json
import os
import shutil
import time
from typing import Any
from unittest.mock import create_autospec
from zoneinfo import ZoneInfo

import pytest

import speech_fake
from liberational import config as cfg
from liberational import memory_log, vision_log
from liberational.memory import history_qa as h
from liberational.memory import question
from liberational.people import registry
from liberational.vision import sightings
from vision_fake import FakeVisionLog
from voices import write_voices

LA = ZoneInfo("America/Los_Angeles")  # a zone with DST, so the day-length cases are deterministic
NOW = dt.datetime(2026, 10, 23, 12, tzinfo=LA).timestamp()
YESTERDAY = NOW - 86400
TODAY = "2026-10-23"
SAM_WANTED = "all Sam wanted to work on yesterday"
PARIS = "/usr/share/zoneinfo/Europe/Paris"


@pytest.fixture(autouse=True)
def la_zone(monkeypatch):
    monkeypatch.setattr(h, "TZ", LA)


@pytest.fixture
def state(tmp_path):
    root = tmp_path / "state"
    root.mkdir()
    return root


def ask(state, question):
    return h.collect(question, state, NOW)


def transcripts(state, rows):
    for r in rows:
        speech_fake.fake().say(r)


def row(t, **kw):
    return dict(ts_start=t, speaker="Sam", text="I want to work on the garden.", **kw)


def camera(state, rows):
    """Caption records from v1-style rows: `ts`, and brightness as the 0..1 proxy (or the record's dict). A row
    without `observation` has none, as a legacy caption."""
    log = vision_log.at(state)
    assert isinstance(log, FakeVisionLog)
    for r in rows:
        fields = dict(r)
        ts = fields.pop("ts")
        light = fields.pop("brightness", None)
        if isinstance(light, (int, float)):
            light = dict(mean=light * 255, dark_fraction=0.0)
        if light is not None:
            fields["brightness"] = light
        fields.setdefault("yaw_deg", 0.0)
        log.add(vision_log.CAPTIONS, ts_start=ts, **fields)


def desk(n=13):
    """n camera0 captions 10 s apart, the last one at NOW."""
    return [dict(ts=NOW - 10 * (n - 1 - i), cam="cam0", caption=f"A desk, frame {i}") for i in range(n)]


def add_journal(state, text, start, level=30, people=()):
    """A journal entry on the (in-memory) memory log, about `people` (by name), as the journal worker writes one."""
    reg = registry.at(state)
    subjects = [str(reg.ensure(n, "test")) for n in people]
    record = {"ts_start": start, "ts_end": start + level, "text": text, "level": level}
    return memory_log.at(state).append(memory_log.JOURNAL, record, subjects=subjects)


@pytest.mark.parametrize("question", ["who's home?", "Who is home right now?", "who\u2019s here"])
def test_who_is_home_reads_who_was_recognized_in_the_last_hour(question):
    query = h.parse(question, NOW)
    assert (query["kind"], query["start"], query["end"]) == ("who", NOW - 3600, NOW)


def test_parse_reads_the_camera_number():
    assert h.parse("when Alex last on camera3", NOW)["cameras"] == ["cam3"]


@pytest.mark.parametrize(
    ("question", "message"),
    [
        ("camera banana yesterday", "actual camera ID"),
        ("camera1 yesterday at 25", "clock time"),
        pytest.param("who was here at 13pm", "clock time like 3pm", id="13pm"),
        pytest.param("who was here at 0am", "clock time like 3pm", id="0am"),
        pytest.param("camera1 2026-10-30", "in the future", id="future-date"),
    ],
)
def test_parse_rejects(question, message):
    with pytest.raises(ValueError, match=message):
        h.parse(question, NOW)


@pytest.mark.parametrize(
    ("question", "start"),
    [
        pytest.param("what did Dana say since 9am today", (23, 9, 0), id="clock-today"),
        pytest.param("what did Dana say since 10:30", (23, 10, 30), id="24h-clock"),
        pytest.param("what did Dana say since 11", (23, 11, 0), id="bare-hour-latest-past-half"),
        pytest.param("what did Dana say since 9 this morning", (23, 9, 0), id="this-morning"),
        pytest.param("what did Dana say since midnight", (23, 0, 0), id="midnight"),
        pytest.param("what did Dana say since 1 p.m. yesterday", (22, 13, 0), id="dotted-pm-yesterday"),
        pytest.param("what did Dana say yesterday since noon", (22, 12, 0), id="yesterday-since-noon"),
        pytest.param("what did Dana say since 11 last night", (22, 23, 0), id="last-night"),
        pytest.param("what did Dana say since 2 last night", (23, 2, 0), id="last-night-after-midnight"),
    ],
)
def test_since_a_clock_reads_from_that_clock_to_now(question, start):
    query = h.parse(question, NOW)
    begin = dt.datetime.fromtimestamp(query["start"], LA)
    assert ((begin.day, begin.hour, begin.minute), query["end"]) == (start, NOW)


@pytest.mark.parametrize(
    ("question", "start"),
    [
        pytest.param("what did Dana say since yesterday", (22, 0), id="yesterday-midnight"),
        pytest.param("what did Dana say since this morning", (23, 5), id="morning-starts-at-five"),
        pytest.param("what did Dana say since last night", (22, 18), id="last-night-starts-at-six-pm"),
    ],
)
def test_since_a_day_or_part_of_it_reads_from_its_start_to_now(question, start):
    query = h.parse(question, NOW, selected_day="2026-10-20")
    begin = dt.datetime.fromtimestamp(query["start"], LA)
    assert ((begin.day, begin.hour, begin.minute), query["end"]) == ((*start, 0), NOW)


def test_since_this_afternoon_starts_at_noon():
    later = dt.datetime(2026, 10, 23, 18, tzinfo=LA).timestamp()
    query = h.parse("what did Dana say since this afternoon", later)
    assert (query["start"], query["end"]) == (dt.datetime(2026, 10, 23, 12, tzinfo=LA).timestamp(), later)


@pytest.mark.parametrize("question", ["what did Dana say since this evening", "camera1 since this afternoon at 3pm"])
def test_since_a_part_of_the_day_refuses_one_not_started(question):
    with pytest.raises(ValueError, match="hasn't started yet|in the future"):
        h.parse(question, dt.datetime(2026, 10, 23, 11, tzinfo=LA).timestamp())


def test_since_a_clock_on_a_selected_day_ends_with_that_day():
    query = h.parse("what did Dana say since 3pm", NOW, selected_day="2026-10-22")
    expected = dt.datetime(2026, 10, 22, 15, tzinfo=LA).timestamp(), dt.datetime(2026, 10, 23, tzinfo=LA).timestamp()
    assert (query["start"], query["end"]) == expected


def test_since_a_clock_names_the_day_in_its_window():
    assert h.parse("what did Dana say since 1 p.m. yesterday", NOW)["window"] == "since 13:00 yesterday"


def test_between_two_clocks_reads_the_hour_the_pm_names():
    query = h.parse("what did Dana say between 2 and 3 PM yesterday", NOW)
    start, end = (dt.datetime.fromtimestamp(query[k], LA) for k in ("start", "end"))
    assert ((start.day, start.hour), (end.day, end.hour)) == ((22, 14), (22, 15))


@pytest.mark.parametrize(
    ("question", "message"),
    [
        pytest.param("what did Dana say since 5pm", "in the future", id="future"),
        pytest.param("what did Dana say since 3 this afternoon", "in the future", id="future-part"),
        pytest.param("what did Dana say since 3pm this morning", "isn't this morning", id="wrong-half"),
        pytest.param("what did Dana say since 25", "clock time like 3pm", id="bad-clock"),
        pytest.param("what did Dana say since 9am on 2026-10-20", "single time window", id="with-date"),
    ],
)
def test_since_a_clock_refuses_what_it_cannot_read(question, message):
    with pytest.raises(ValueError, match=message):
        h.parse(question, NOW)


@pytest.mark.parametrize(
    "question",
    [
        pytest.param("what did we talk about last month", id="last-month"),
        pytest.param("what did Dana say since Monday", id="since-weekday"),
        pytest.param("what did Dana say before 5pm", id="before-clock"),
        pytest.param("what did Dana say two weeks ago", id="ago"),
        pytest.param("what did Dana say over the past three weeks", id="past-weeks"),
        pytest.param("what did Dana say on Friday", id="weekday"),
    ],
)
def test_parse_still_refuses_time_phrases_it_cannot_read(question):
    with pytest.raises(ValueError, match="Unsupported time phrase"):
        h.parse(question, NOW)


@pytest.mark.parametrize(
    "question",
    [
        pytest.param("what did Dana say about the week of work with Sam", id="the-week-of"),
        pytest.param("what did Dana say about the van since its last oil change", id="since-its-last"),
        pytest.param("what did Dana say about the last chapter", id="last-noun"),
        pytest.param("what did Dana say about the past", id="the-past"),
        pytest.param("what did Dana say before leaving", id="before-verb"),
        pytest.param("what did Dana say about the previous owner", id="previous-noun"),
    ],
)
def test_ordinary_time_words_keep_the_selected_day(question):
    q = h.parse(question, NOW, selected_day="2026-10-22")
    assert (q["person"], q["window"]) == ("Dana", "2026-10-22")


@pytest.mark.parametrize(
    "question",
    [
        "What snack did someone ask about?",
        "What book did somebody mention?",
        "what did anyone say about the garden",
        "What recipe did anybody suggest",
    ],
)
def test_an_indefinite_speaker_is_nobody_in_particular(question):
    q = h.parse(question, NOW)
    assert (q["person"], q["participants"]) == (None, [])


@pytest.mark.parametrize(
    ("question", "person"),
    [
        ("What snack did Dana ask about?", "Dana"),
        ("What book did Alice mention yesterday?", "Alice"),
        ("what did Dana say", "Dana"),
        ("what Alice said about lunch", "Alice"),
    ],
)
def test_a_noun_before_did_is_not_part_of_the_speaker(question, person):
    assert h.parse(question, NOW)["person"] == person


@pytest.mark.parametrize(
    "question",
    [
        "What was said in the last 30 minutes?",
        "what else was said today",
        "What has been mentioned in the last 2 hours?",
        "What were they told yesterday?",
    ],
)
def test_a_passive_question_names_no_speaker(question):
    q = h.parse(question, NOW)
    assert (q["kind"], q["person"], q["participants"]) == ("transcript", None, [])


def test_a_passive_question_about_a_listener_keeps_the_name():
    assert h.parse("What was Dana told yesterday?", NOW)["person"] == "Dana"


@pytest.mark.parametrize(
    ("question", "hour"),
    [pytest.param("who was here at 12am", 0, id="midnight"), pytest.param("who was here at 12pm", 12, id="noon")],
)
def test_parse_reads_twelve_oclock(question, hour):
    assert dt.datetime.fromtimestamp(h.parse(question, NOW)["at"], LA).hour == hour


def test_a_future_date_is_answered_as_future(state):
    assert "in the future" in h.collect("camera1 2026-10-30", state, now=NOW)["answer"]


@pytest.mark.parametrize(
    ("question", "now", "hours"),
    [
        pytest.param("camera1 yesterday", dt.datetime(2026, 3, 9, 12, tzinfo=LA), 23, id="spring-forward"),
        pytest.param("camera1 yesterday", dt.datetime(2026, 11, 2, 12, tzinfo=LA), 25, id="fall-back"),
        pytest.param("camera1 2026-10-22", dt.datetime(2026, 10, 23, 12, tzinfo=LA), 24, id="plain-date"),
    ],
)
def test_a_day_spans_its_local_hours(question, now, hours):
    q = h.parse(question, now.timestamp())
    assert q["end"] - q["start"] == hours * 3600


def test_presence_answers_need_the_camera_and_a_face_match(state):
    camera(state, [dict(ts=NOW - 10, cam="cam2", caption="A man at a desk")])
    assert "cam3" in ask(state, "when Alex last on camera3")["answer"]
    r = ask(state, "when Alex last on camera2")
    assert r["evidence"] == []
    assert "unavailable" in r["answer"]
    camera(state, [dict(ts=NOW - 5, cam="cam2", caption="A person", faces=[dict(name="Alex", sim=0.8)])])
    r = ask(state, "when Alex last on camera2")
    assert "[C1]" in r["answer"]
    assert r["evidence"][0]["identity"]["name"] == "Alex"


def test_caption_is_not_identity(state):
    camera(state, [dict(ts=NOW - 3, cam="cam0", caption="Alex is here")])
    assert "does not establish" in ask(state, "when Alex last on camera0")["answer"]


def test_identity_and_the_brightness_proxy_come_from_the_record(state):
    camera(state, [dict(ts=NOW - 20, cam="cam0", caption="Desk")])
    assert ask(state, "camera0 today")["coverage"]["cameras"]["cam0"]["identity_available"] == 0
    camera(
        state, [dict(ts=NOW - 10, cam="cam0", caption="Desk", faces=[], brightness=dict(mean=25.5, dark_fraction=0.9))]
    )
    r = ask(state, "camera0 today")
    assert r["coverage"]["cameras"]["cam0"]["identity_available"] == 1
    assert [e.get("brightness") for e in r["evidence"]] == [None, pytest.approx(0.1)]  # 25.5 / 255


def test_a_repeat_for_an_unchanged_view_is_counted_but_never_quoted(state):
    camera(state, [dict(ts=NOW - 20, cam="cam0", caption="A desk with a red toolbox")])
    camera(state, [dict(ts=NOW - 10, cam="cam0", caption="A desk with a red toolbox", unchanged=True, since=NOW - 20)])
    r = ask(state, "when was the toolbox visible on camera 0 today")
    assert r["coverage"]["cameras"]["cam0"]["count"] == 2
    assert [e["ts"] for e in r["evidence"]] == [NOW - 20]


@pytest.mark.parametrize(
    ("question", "unknown"),
    [
        pytest.param("how many times lights went off today", h.LIGHTS_UNKNOWN, id="on-off-moments"),
        pytest.param("how long were the lights on in the last 3 hours", h.LIGHTS_WINDOW, id="not-whole-days"),
        pytest.param("how long were the lights on yesterday", h.LIGHTS_NO_RECORDINGS, id="no-recordings"),
    ],
)
def test_lights_without_the_colour_measure_are_unknown_with_the_reason_not_from_brightness(state, question, unknown):
    # Caption brightness swings with auto exposure, not the lamps: off, on, off again is no evidence of either.
    values = [0.5, 0.5, 0.05, 0.5, 0.5, 0.05, 0.05, 0.05, 0.05]
    camera(state, [dict(ts=NOW - 100 + i * 3, cam="cam0", caption="Desk", brightness=v) for i, v in enumerate(values)])
    r = ask(state, question)
    assert r["answer"] == unknown
    assert r["evidence"] == []


def test_lights_that_cant_be_read_say_so_and_warn(state, monkeypatch):
    from liberational.memory import lights_color

    def unreadable(*args, **kwargs):
        raise OSError("disk")

    monkeypatch.setattr(lights_color, "analyse_day", unreadable)
    r = ask(state, "how long were the lights on yesterday")
    assert r["answer"] == h.LIGHTS_UNREADABLE.format(error="OSError")
    assert any("couldn't be read for the lights" in w for w in r["warnings"])


def test_lights_day_window_uses_configured_timezone_not_process_timezone(state, monkeypatch):
    from liberational.memory import lights_color

    seen: list[tuple[dt.date, dt.tzinfo | None]] = []

    def fake_analyse_day(
        state_path,
        day,
        cameras=("cam0", "cam1", "cam2"),
        until=None,
        tz=None,
    ):
        seen.append((day, tz))
        return {"day": day.isoformat(), "on_s": 0.0, "measured_s": 0.0}

    monkeypatch.setattr(lights_color, "analyse_day", fake_analyse_day)
    saved_tz = os.environ.get("TZ")
    os.environ["TZ"] = "UTC"
    time.tzset()
    try:
        start = dt.datetime(2026, 3, 8, tzinfo=LA).timestamp()
        end = dt.datetime(2026, 3, 9, tzinfo=LA).timestamp()
        answer, coverage = h._lights_by_colour(state, {"start": start, "end": end}, [])
    finally:
        if saved_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = saved_tz
        time.tzset()

    assert answer == h.LIGHTS_NO_RECORDINGS
    assert coverage is None
    assert seen == [(dt.date(2026, 3, 8), LA)]


def test_alias_not_relabel_named(state):
    # Voice 1 was renamed from Alex to Sam: a line heard as Alex keeps that name, since a rename never rewrote it.
    write_voices(state, {1: "Sam"}, forwards={2: 1})
    transcripts(
        state,
        [
            dict(ts_start=YESTERDAY, speaker="Speaker 2", speaker_id=2, text="Garden please"),
            dict(ts_start=YESTERDAY + 1, speaker="Alex", speaker_id=1, text="Not Sam"),
        ],
    )
    r = ask(state, SAM_WANTED)
    assert len(r["evidence"]) == 1
    assert r["evidence"][0]["recorded_speaker"] == "Speaker 2"


def test_a_persons_several_voices_all_answer_for_them(state):
    # Naming replaces merging, so one person may have several voices.
    write_voices(state, {1: "Sam", 2: "Sam"})
    transcripts(
        state,
        [
            dict(ts_start=YESTERDAY, speaker="Speaker 1", speaker_id=1, text="Garden please"),
            dict(ts_start=YESTERDAY + 60, speaker="Speaker 2", speaker_id=2, text="And the fence"),
        ],
    )
    r = ask(state, SAM_WANTED)
    assert "ambiguous_speaker" not in r["coverage"]
    assert [e["text"] for e in r["evidence"]] == ["Garden please", "And the fence"]


def test_a_voice_clone_with_the_same_name_is_not_ambiguous(state):
    # Sam's room voice (1) and a voice clone recorded under Sam's name (2), as SpeakerDB.enroll(clone=True) makes.
    write_voices(state, {1: "Sam", 2: "Sam"})
    (state / "clones").mkdir()
    (state / "clones" / "2.json").write_text(json.dumps(dict(source="recorded", label="Sam")))
    transcripts(state, [row(YESTERDAY, speaker_id=1)])
    r = ask(state, SAM_WANTED)
    assert "ambiguous_speaker" not in r["coverage"]
    assert [e["text"] for e in r["evidence"]] == ["I want to work on the garden."]


def test_playback_is_left_out_and_truncation_is_partial(state, monkeypatch):
    texts = ["Garden", "password rosebud", "123456", "Playback", "Fence"]
    transcripts(state, [dict(ts_start=YESTERDAY + i, speaker="Sam", text=t) for i, t in enumerate(texts)])
    (state / "playback_windows.json").write_text(json.dumps([dict(start=YESTERDAY + 3, end=YESTERDAY + 3.5)]))
    monkeypatch.setattr(h, "MAX_EVIDENCE", 1)
    r = ask(state, SAM_WANTED)
    assert r["coverage"]["matched"] == 4  # everything but the playback
    assert r["coverage"]["truncated"] and r["coverage"]["partial"]
    assert r["evidence"][0]["text"] == "Garden"


# "hot springs" is 11 characters and "last weekend" is 12, so 23 holds those two and not the 17-character line before them.
FOLLOW_UP = (
    dict(ts_start=YESTERDAY, speaker="Sam", text="morning fence talk"),
    dict(ts_start=YESTERDAY + 10, speaker="Sam", text="still the morning"),
    dict(ts_start=YESTERDAY + 3600, speaker="Sam", text="hot springs"),
    dict(ts_start=YESTERDAY + 3660, speaker="Sam", text="last weekend"),
)


def test_a_short_unscoped_transcript_stays_chronological(state):
    transcripts(state, list(FOLLOW_UP))
    r = ask(state, "anything said yesterday?")
    assert [e["text"] for e in r["evidence"]] == [row["text"] for row in FOLLOW_UP]
    assert r["coverage"]["truncated"] is False


@pytest.mark.parametrize(
    ("limit", "value"),
    [pytest.param("MAX_EVIDENCE", 2, id="count"), pytest.param("MAX_CHARS", 23, id="characters")],
)
def test_an_unscoped_question_keeps_the_lines_just_before_it(state, monkeypatch, limit, value):
    transcripts(state, list(FOLLOW_UP))
    monkeypatch.setattr(h, limit, value)
    r = ask(state, "anything said yesterday?")
    assert [e["text"] for e in r["evidence"]] == ["hot springs", "last weekend"]
    assert r["coverage"]["matched"] == 4
    assert r["coverage"]["truncated"] is True


UTOPIA = "We went to Utopia at the hot springs."


@pytest.mark.parametrize(
    ("day", "said_at", "now"),
    [
        # 22:40 PDT is 05:40 UTC on 10-31, so a day cut in UTC would put the line on the wrong day.
        pytest.param(
            "2026-10-30",
            dt.datetime(2026, 10, 30, 22, 40, tzinfo=LA),
            dt.datetime(2026, 11, 1, 14, tzinfo=LA),
            id="late-evening",
        ),
        # 2026-11-01 is 25 hours long (PDT to PST at 02:00); 23:30 PST is in its last hour.
        pytest.param(
            "2026-11-01",
            dt.datetime(2026, 11, 1, 23, 30, tzinfo=LA),
            dt.datetime(2026, 11, 2, 9, tzinfo=LA),
            id="fall-back-day",
        ),
    ],
)
def test_a_selected_day_keeps_the_lines_the_question_is_about(state, monkeypatch, day, said_at, now):
    """A busy day's newest lines don't push out the line the question names, and the day is cut at local
    midnight, so the same words just outside it stay out."""
    d = dt.date.fromisoformat(day)
    before = dt.datetime.combine(d, dt.time(), tzinfo=LA).timestamp() - 600
    after = dt.datetime.combine(d + dt.timedelta(days=1), dt.time(), tzinfo=LA).timestamp() + 600
    said = said_at.timestamp()
    lines = [
        dict(ts_start=before, speaker="Sam", text="Utopia was the day before."),
        dict(ts_start=said, speaker="Sam", text=UTOPIA),
        *(dict(ts_start=said + 300 + 60 * i, speaker="Sam", text=f"filler {i}") for i in range(20)),
        dict(ts_start=after, speaker="Sam", text="Utopia was the day after."),
    ]
    transcripts(state, lines)
    monkeypatch.setattr(h, "MAX_EVIDENCE", 2)
    r = h.collect("Where was the Utopia event held?", state, now.timestamp(), day=day)
    assert [e["text"] for e in r["evidence"]] == [UTOPIA, "filler 19"]


def test_a_matching_line_brings_the_lines_said_around_it(state, monkeypatch):
    lines = [
        dict(ts_start=YESTERDAY, speaker="Sam", text="Where did the pickles come from?"),
        dict(ts_start=YESTERDAY + 5, speaker="Dana", text="A farm stand in the valley."),
        *(dict(ts_start=YESTERDAY + 600 + 60 * i, speaker="Sam", text=f"filler {i}") for i in range(10)),
    ]
    transcripts(state, lines)
    monkeypatch.setattr(h, "MAX_EVIDENCE", 3)
    r = h.collect("Where did the pickles come from?", state, NOW, day="2026-10-22")
    assert [e["text"] for e in r["evidence"]] == [
        "Where did the pickles come from?",
        "A farm stand in the valley.",
        "filler 9",
    ]


def test_a_matching_lines_context_stays_in_its_own_conversation():
    """A call overlapping room speech: the call's answer comes with the matching question, the room's lines don't."""
    room = [dict(ts=1000.0 + i * 0.5, quote=f"room chatter {i}", conversation_source="room") for i in range(100)]
    call = [
        dict(ts=1040.0, quote="Where did the pickles come from?", conversation_source="calls/2026-10-22/a"),
        dict(ts=1045.0, quote="A farm stand in the valley.", conversation_source="calls/2026-10-22/a"),
    ]
    rows = sorted([*room, *call], key=lambda r: r["ts"])
    ordered = h._about_first(rows, "Where did the pickles come from?")
    assert [r["quote"] for r in ordered[:2]] == ["Where did the pickles come from?", "A farm stand in the valley."]


def test_a_rare_word_outweighs_common_ones(state, monkeypatch):
    lines = [
        dict(ts_start=YESTERDAY, speaker="Dana", text="The pickles were from a farm stand."),
        *(dict(ts_start=YESTERDAY + 600 + 60 * i, speaker="Sam", text=f"Come on, {i} from here.") for i in range(10)),
    ]
    transcripts(state, lines)
    monkeypatch.setattr(h, "MAX_EVIDENCE", 1)
    r = h.collect("Where did the pickles come from?", state, NOW, day="2026-10-22")
    assert [e["text"] for e in r["evidence"]] == ["The pickles were from a farm stand."]


def test_a_question_that_names_someone_still_keeps_the_earliest_lines(state, monkeypatch):
    transcripts(state, list(FOLLOW_UP))
    monkeypatch.setattr(h, "MAX_EVIDENCE", 2)
    r = ask(state, "what did Sam say yesterday")
    assert [e["text"] for e in r["evidence"]] == ["morning fence talk", "still the morning"]


def test_journal_entries_stay_ahead_of_the_newest_lines(state, monkeypatch):
    transcripts(state, [dict(ts_start=NOW - 60 + 10 * i, speaker="Sam", text=f"line {i}") for i in range(4)])
    add_journal(state, "Sam talked about a trip.", NOW - 50)
    monkeypatch.setattr(h, "MAX_EVIDENCE", 3)
    r = h.collect("where was it?", state, NOW, day=TODAY)
    assert [(e["id"], e["text"]) for e in r["evidence"]] == [
        ("J1", "Sam talked about a trip."),
        ("T2", "line 2"),
        ("T3", "line 3"),
    ]


def test_lines_that_mention_passwords_or_codes_are_not_hidden(state):
    """History answers hide nothing based on what a line says."""
    texts = ["We moved here in 2019 and want to fix the fence", "The password is rosebud", "The code is 482913"]
    transcripts(state, [dict(ts_start=YESTERDAY + i, speaker="Sam", text=t) for i, t in enumerate(texts)])
    assert [e["text"] for e in ask(state, SAM_WANTED)["evidence"]] == texts


@pytest.mark.parametrize(
    ("extra", "partial"),
    [
        pytest.param([], False, id="complete"),
        pytest.param([{"speaker": "caller", "text": "no time"}], True, id="skipped-malformed-call-line"),
    ],
)
def test_transcript_answer_is_partial_only_when_rows_were_skipped(state, extra, partial):
    transcripts(state, [row(YESTERDAY + 60)])
    call = state.parent / "calls" / "2026-10-22" / "test"
    call.mkdir(parents=True)
    (call / "call.json").write_text(json.dumps(dict(started="2026-10-22T12:00:00-07:00")))
    (call / "transcript.jsonl").write_text("".join(json.dumps(e) + "\n" for e in extra))
    r = ask(state, "what did Sam want to work on yesterday?")
    assert r["evidence"]
    assert r["coverage"]["partial"] is partial


def with_error_row(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows[6]["observation"] = dict(status="error")
    return rows


@pytest.mark.parametrize(
    ("rows", "partial"),
    [
        pytest.param(desk(), False, id="steady"),
        pytest.param(desk()[:3] + desk()[9:], True, id="gap"),
        pytest.param(with_error_row(desk()), True, id="excluded-error-row"),
    ],
)
def test_camera_answer_is_partial_only_when_its_captions_have_a_gap(state, rows, partial):
    camera(state, rows)
    assert ask(state, "what did camera0 show in the last 2 minutes?")["coverage"]["partial"] is partial


def test_skipped_captions_are_excluded_and_legacy_rows_kept(state):
    camera(
        state,
        [
            dict(ts=NOW - 10, cam="cam0", caption="Zero people visible.", observation=dict(status="fresh")),
            dict(ts=NOW - 9, cam="cam0", caption="Skipped", observation=dict(status="not_observed")),
            dict(ts=NOW - 8, cam="cam0", caption="Legacy room"),
        ],
    )
    r = ask(state, "what was on camera0 today")
    assert [(e["quote"], e["observation"]["status"]) for e in r["evidence"]] == [
        ("Zero people visible.", "fresh"),
        ("Legacy room", "legacy_unknown"),
    ]
    assert any("excluded" in w for w in r["warnings"])


def test_question_match_not_lost_after_evidence_limit(state):
    rows: list[dict[str, Any]] = [
        dict(ts=NOW - 500 + i, cam="cam0", caption="Zero people; shelves visible.") for i in range(150)
    ]
    rows.append(
        dict(ts=NOW - 1, cam="cam0", caption="One person holding a red toolbox.", observation=dict(status="fresh"))
    )
    camera(state, rows)
    r = ask(state, "when was the toolbox visible on camera 0 today")
    assert "toolbox" in r["evidence"][0]["quote"]
    assert r["evidence"][0]["observation"]["status"] == "fresh"


def test_call_timestamp_and_bot(state):
    call = state.parent / "calls" / "2026-10-22" / "test"
    call.mkdir(parents=True)
    (call / "call.json").write_text(json.dumps(dict(started="2026-10-22T12:00:00-07:00")))
    (call / "transcript.jsonl").write_text(
        "\n".join(
            json.dumps(dict(t=i, speaker=s, speaker_label="Sam", text="Build a fence"))
            for i, s in enumerate(["caller", "bot"])
        )
    )
    r = ask(state, SAM_WANTED)
    assert len(r["evidence"]) == 1
    assert r["evidence"][0]["source"] == "call"
    assert r["evidence"][0]["href"] == "/calls/detail?day=2026-10-22&dir=test"


def test_transcript_citation_seeks_review_at_the_line_time(state):
    transcripts(state, [row(YESTERDAY + 0.25)])
    evidence = ask(state, SAM_WANTED)["evidence"][0]
    assert evidence["href"] == f"/review#day=2026-10-22&t={YESTERDAY + 0.25}"
    assert "camera" not in evidence  # audio alone does not identify a camera angle


def test_journal_without_matching_entries_changes_nothing(state):
    transcripts(state, [row(YESTERDAY), dict(ts_start=YESTERDAY + 5, speaker="Sam", text="And the fence.")])
    camera(state, [dict(ts=NOW - 10, cam="cam2", caption="A man at a desk")])
    questions = (SAM_WANTED, "what was on camera2 today", "what did Sam say yesterday")
    before = [json.dumps(ask(state, q)) for q in questions]
    add_journal(state, "Sam talked about the garden.", YESTERDAY - 7 * 86400)  # outside every window
    add_journal(state, "Nothing of note.", YESTERDAY)  # no matching words
    assert [json.dumps(ask(state, q)) for q in questions] == before


def test_journal_entries_join_transcript_and_camera_questions(state):
    write_voices(state, {1: "Sam"})
    transcripts(state, [row(YESTERDAY, speaker_id=1)])
    add_journal(state, "[S1] planned work on the garden and the fence.", YESTERDAY, 180, people=["Sam"])
    r = ask(state, SAM_WANTED)
    assert [e["id"] for e in r["evidence"]] == ["J1", "T2"]
    assert r["evidence"][0]["text"] == "Sam planned work on the garden and the fence."
    assert r["evidence"][0]["kind"] == "journal"
    assert r["coverage"]["journal"] == dict(selected=1)
    assert any("Journal entries are summaries" in w for w in r["warnings"])
    assert r["answer"] is None
    # A journal entry alone is evidence: no "no matching evidence" answer.
    assert ask(state, "what did Sam say about the fence today")["evidence"] == []
    add_journal(state, "[S1] fixed the fence.", NOW - 600, 30, people=["Sam"])
    r = ask(state, "what did Sam say about the fence today")
    assert [e["id"] for e in r["evidence"]] == ["J1"]
    assert r["answer"] is None


def test_a_question_with_no_search_words_gets_the_latest_journal_entries(state):
    # "where was it?" is all stop words, so the journal search has nothing to look for; the entries that start
    # latest in the last few minutes still join, in time order, ahead of the lines.
    transcripts(state, [dict(ts_start=NOW - 20, speaker="Sam", text="at the hot springs")])
    add_journal(state, "Too early to count.", NOW - h.JOURNAL_TAIL_S - 60)
    for ago in (136, 106, 76, 46):
        add_journal(state, f"Entry from {ago} s before.", NOW - ago)
    r = h.collect("where was it?", state, NOW, day=TODAY)  # a follow-up asked on its day
    assert [(e["id"], e["text"]) for e in r["evidence"]] == [
        ("J1", "Entry from 106 s before."),
        ("J2", "Entry from 76 s before."),
        ("J3", "Entry from 46 s before."),
        ("T4", "at the hot springs"),
    ]
    assert r["coverage"]["journal"] == dict(selected=3)


def test_a_question_with_no_search_words_and_no_recent_journal_gets_none(state):
    transcripts(state, [dict(ts_start=NOW - 20, speaker="Sam", text="at the hot springs")])
    add_journal(state, "Too early to count.", NOW - h.JOURNAL_TAIL_S - 60)
    r = h.collect("where was it?", state, NOW, day=TODAY)  # a follow-up asked on its day
    assert [e["text"] for e in r["evidence"]] == ["at the hot springs"]
    assert "journal" not in r["coverage"]


def test_journal_stays_out_of_a_question_about_one_camera(state):
    # Journal entries mix every camera and speech, so a question that names a camera gets only that camera.
    camera(state, [dict(ts=NOW - 30, cam="cam2", caption="An empty driveway with a parked car")])
    add_journal(state, "A man on camera 0 was in the kitchen, cooking at the stove.", NOW - 600)
    r = ask(state, "what did camera2 see in the kitchen today")
    assert {(e["source"], e["camera"]) for e in r["evidence"]} == {("camera", "cam2")}
    assert "journal" not in r["coverage"]
    r = ask(state, "what did the cameras see in the kitchen today")  # no camera named: the journal joins
    assert r["evidence"][0]["kind"] == "journal"


PILL = "what did Sam hide the pill in this morning"


@pytest.mark.parametrize(
    ("measure", "heard"),
    [
        pytest.param("describe", True, id="describe-reads-the-lines"),
        pytest.param("count", False, id="a-count-keeps-to-the-cameras"),
    ],
)
def test_a_question_read_as_seen_also_reads_what_was_said(state, readings, measure, heard):
    # Whether "what did Sam hide the pill in" was seen or said is the reading's guess, so a question read as about
    # the cameras still gets the lines said in its window.
    readings[PILL] = question.Reading(source="cameras", measure=measure, people=("Sam",), part="this morning")
    said = dt.datetime(2026, 10, 23, 7, 21, tzinfo=LA).timestamp()
    transcripts(state, [dict(ts_start=said, speaker="Sam", text="I hid the pill in a piece of cheese.")])
    camera(state, [dict(ts=said + 60, cam="cam0", caption="A kitchen counter with a bowl")])
    quotes = [e["quote"] for e in h.collect(PILL, state, NOW)["evidence"]]
    assert ("I hid the pill in a piece of cheese." in quotes) is heard
    assert "A kitchen counter with a bowl" in quotes


def test_journal_never_moves_a_recorded_answers_citation(state):
    camera(state, [dict(ts=NOW - 5, cam="cam2", caption="A person", faces=[dict(name="Alex", sim=0.8)])])
    add_journal(state, "Alex came by camera2 and waved.", NOW - 60)
    r = ask(state, "when Alex last on camera2")
    assert "[C1]" in r["answer"]
    assert [e["id"] for e in r["evidence"]] == ["C1"]


def test_topic_retrieval_retains_whole_window_and_unknown_speakers(tmp_path):
    rows = [
        dict(ts_start=10, speaker="Alice", text="Garden planning"),
        dict(ts_start=11, speaker="unknown", text="I prefer tomatoes"),
        dict(ts_start=900, speaker="Dana", text="Garden irrigation"),
    ]
    transcripts(tmp_path, rows)
    query = dict(start=0, end=1000, person=None, participants=["Alice", "Dana"], topics=True, intent=False)
    found, _ = h._transcripts(tmp_path, query, [])
    assert [r["participants_matched"] for r in found] == [["Alice"], [], ["Dana"]]


@pytest.mark.skipif(not os.path.exists(PARIS), reason="no system zoneinfo")
@pytest.mark.parametrize(
    "tz",
    [
        pytest.param(lambda tmp: PARIS, id="zone-file"),
        pytest.param(lambda tmp: ":" + shutil.copy(PARIS, str(tmp / "localtime")), id="alpine-copy-not-symlink"),
        pytest.param(lambda tmp: "Europe/Paris", id="zone-name"),
    ],
)
def test_local_zone_reads_tz(tz, tmp_path, monkeypatch):
    monkeypatch.setenv("TZ", tz(tmp_path))
    assert dt.datetime(2026, 7, 1, 12, tzinfo=h.local_zone()).utcoffset() == dt.timedelta(hours=2)


# ---- parts of the day: one set of hours, against the selected day or today ----


def span(q):
    """The window a parse read, as local (start, end) times."""
    return dt.datetime.fromtimestamp(q["start"], LA), dt.datetime.fromtimestamp(q["end"], LA)


@pytest.mark.parametrize(
    ("question", "selected", "first", "last"),
    [
        pytest.param("who was here in the evening", "2026-10-22", (22, 17), (22, 22), id="evening-of-the-selected-day"),
        pytest.param("what did Dana say this morning", None, (23, 5), (23, 12), id="morning-today-up-to-now"),
        pytest.param(
            "what did Dana say yesterday in the afternoon", None, (22, 12), (22, 17), id="afternoon-yesterday"
        ),
        pytest.param("who was here last night", None, (22, 18), (23, 5), id="last-night-runs-past-midnight"),
        pytest.param("camera1 2026-10-20 during the night", None, (20, 22), (21, 5), id="night-of-a-date"),
    ],
)
def test_a_part_of_the_day_reads_its_hours(question, selected, first, last):
    start, end = span(h.parse(question, NOW, selected))
    assert ((start.day, start.hour), (end.day, end.hour)) == (first, last)


def test_a_part_of_the_day_names_its_day_in_the_window():
    assert h.parse("who was here in the evening", NOW, "2026-10-22")["window"] == "the evening of 2026-10-22"


@pytest.mark.parametrize(
    ("question", "message"),
    [
        pytest.param("who was here this evening", "hasn't started yet", id="not-started-yet"),
        pytest.param("who was here tonight", "hasn't started yet", id="tonight-at-noon"),
        pytest.param("camera1 last 2 hours in the evening", "single time window", id="with-a-duration"),
    ],
)
def test_a_part_of_the_day_that_cant_be_read_is_refused(question, message):
    with pytest.raises(ValueError, match=message):
        h.parse(question, NOW)


def test_a_night_across_the_spring_change_is_an_hour_short():
    now = dt.datetime(2026, 3, 10, 12, tzinfo=LA).timestamp()
    q = h.parse("camera1 2026-03-07 during the night", now)
    assert q["end"] - q["start"] == 6 * 3600  # 22:00 to 05:00, with 02:00 to 03:00 skipped


def test_speech_heard_on_two_mics_is_one_piece_of_evidence(state):
    """Another mic's copy of a line is the same thing said, so an answer cites it once, even when the
    two copies were written down a little differently, or start in different seconds."""
    write_voices(state, {1: "Sam"})
    transcripts(
        state,
        [
            dict(ts_start=YESTERDAY + 0.8, ts_end=YESTERDAY + 3, speaker="Sam", speaker_id=1, device="Kitchen",
                 text="I want to work on the garden."),
            dict(ts_start=YESTERDAY + 1.2, ts_end=YESTERDAY + 3.4, speaker="Sam", speaker_id=1, device="Hall",
                 text="i want to work on the garden"),
            dict(ts_start=YESTERDAY + 60, speaker="Sam", speaker_id=1, device="Hall", text="I want to fix the fence."),
        ],
    )  # fmt: skip
    r = ask(state, SAM_WANTED)
    assert [e["text"] for e in r["evidence"]] == ["I want to work on the garden.", "I want to fix the fence."]


@pytest.mark.parametrize("available", [False, True])
def test_optional_movement_only_changes_partial_status_when_its_facts_are_available(available):
    coverage = {"presence": {"partial": False, "movement": {"available": available, "partial": True}}}
    assert h.was_cut(coverage) is available


def test_labeled_transcript_and_caption_keep_canonical_payload_ids(state):
    utterance_id = "01900000-0000-7000-8000-000000000001"
    caption_id = "01900000-0000-7000-8000-000000000002"
    forged_id = "01900000-0000-7000-8000-000000000099"
    transcripts(state, [row(NOW - 60, id=utterance_id, payload_id=forged_id, payload_ids=[forged_id])])
    camera(state, [dict(ts=NOW - 30, id=caption_id, payload_ids=[forged_id], cam="cam0", caption="A garden.")])
    transcript = ask(state, "what did Sam say about the garden today")["evidence"]
    caption = ask(state, "what was on camera0 today")["evidence"]
    assert [(e["id"], e.get("payload_id")) for e in transcript] == [("T1", utterance_id)]
    assert [(e["id"], e.get("payload_id")) for e in caption] == [("C1", caption_id)]
    assert speech_fake.fake().lines["speech.utterance"][0]["id"] == utterance_id


def test_grouped_face_visit_citation_keeps_each_source_payload_id(state, vision):
    first = vision.visit("Alex", NOW - 15, NOW - 10)
    second = vision.visit("Alex", NOW - 5, NOW)
    first["payload_ids"] = ["01900000-0000-7000-8000-000000000099"]
    grouped = sightings.who_at(NOW - 10, 30, state)
    evidence = h.labeled(h._visit_evidence(grouped[0]), 1)
    assert len(grouped) == 1
    assert evidence["id"] == "C1"
    assert evidence.get("payload_ids") == [first["id"], second["id"]]
    assert evidence["href"] == f"/review#day=2026-10-23&t={NOW - 15}&cam=cam0"


@pytest.mark.parametrize("legacy_id", [None, 42, "C1", "legacy-row"])
def test_camera_citation_does_not_invent_payload_id_for_legacy_row(legacy_id):
    record = dict(id=legacy_id, ts=NOW, cam="cam0", yaw=0.0, caption="A garden.")
    evidence = h.labeled(h.camera_evidence(h._camera_row(record)), 1)
    assert evidence["id"] == "C1"
    assert "payload_id" not in evidence and "payload_ids" not in evidence
    assert record["id"] == legacy_id


@pytest.mark.parametrize("camera_id", ["entryway", "0", "cam00", "CAM0"])
def test_parse_preserves_explicit_configured_camera_names(tmp_path, monkeypatch, camera_id):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "devices.toml").write_text(
        'device=[{name="entryway",kind="rtsp"},{name="0",kind="camera"},'
        '{name="cam00",kind="camera"},{name="CAM0",kind="rtsp"}]'
    )
    monkeypatch.setattr(cfg, "OWNER_ROOT", tmp_path)
    result = h.parse(f"what did camera {camera_id} show yesterday", NOW)
    assert result["cameras"] == [camera_id]
    assert result["kind"] == "camera"


@pytest.mark.parametrize("camera_id", ["last", "year"])
def test_camera_name_is_not_mistaken_for_an_unsupported_time_phrase(tmp_path, monkeypatch, camera_id):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "devices.toml").write_text(f'device=[{{name="{camera_id}",kind="rtsp"}}]')
    monkeypatch.setattr(cfg, "OWNER_ROOT", tmp_path)
    question = f"what did camera {camera_id} show yesterday"
    result = h.parse(question, NOW)
    assert result["cameras"] == [camera_id]
    assert result["window"] == "yesterday"


@pytest.mark.parametrize(
    ("question", "first", "last"),
    [
        ("who was here around 3 pm", (23, 14, 55), (23, 15, 5)),
        ("who was here about 3", (23, 14, 55), (23, 15, 5)),
        ("what did Dana say around 3 pm", (23, 14, 55), (23, 15, 5)),
        ("who was here this afternoon", (23, 12, 0), (23, 17, 0)),
        ("who was here this morning", (23, 5, 0), (23, 12, 0)),
        ("who was here last night", (22, 18, 0), (23, 5, 0)),
        ("who was here an hour ago", (23, 15, 55), (23, 16, 5)),
        ("who was here earlier today", (23, 0, 0), (23, 17, 0)),
        ("who was here between 2 and 4", (23, 14, 0), (23, 16, 0)),
        ("what did Dana say yesterday evening", (22, 17, 0), (22, 22, 0)),
        ("who was here around 8:45 last night", (22, 20, 40), (22, 20, 50)),
        ("who was here around 3 pm on October 22", (22, 14, 55), (22, 15, 5)),
    ],
)
def test_natural_time_words_constrain_the_actual_read_window(question, first, last):
    now = dt.datetime(2026, 10, 23, 17, tzinfo=LA).timestamp()
    query = h.parse(question, now)
    start = dt.datetime.fromtimestamp(query["start"], LA)
    end = dt.datetime.fromtimestamp(query["end"], LA)
    assert (start.day, start.hour, start.minute) == first
    assert (end.day, end.hour, end.minute) == last


@pytest.mark.parametrize(
    "phrase",
    [
        "around tea time",
        "between 2 and banana",
        "about 25 pm",
        "around 3pm and 4pm",
        "last night 8:45",
        "between 2 and 4 and 6",
    ],
)
def test_unreadable_time_words_are_answered_explicitly_without_evidence(state, phrase):
    result = h.collect(f"who was here {phrase}", state, now=NOW)
    assert result["evidence"] == []
    assert "time" in result["answer"].lower()
    assert result["warnings"]


def test_relative_night_uses_question_time_not_the_dashboard_selection():
    query = h.parse("who was here around 8:45 last night", NOW, "2026-10-20")
    start = dt.datetime.fromtimestamp(query["start"], LA)
    end = dt.datetime.fromtimestamp(query["end"], LA)
    assert (start.day, start.hour, start.minute) == (22, 20, 40)
    assert (end.day, end.hour, end.minute) == (22, 20, 50)


def test_a_clock_question_cannot_quote_a_line_from_the_rest_of_the_day(state):
    write_voices(state, {1: "Dana"})
    near = dt.datetime(2026, 10, 23, 15, tzinfo=LA).timestamp()
    far = dt.datetime(2026, 10, 23, 10, tzinfo=LA).timestamp()
    transcripts(
        state,
        [
            dict(ts_start=far, speaker_id=1, speaker="Dana", text="The blue bicycle is ready."),
            dict(ts_start=near, speaker_id=1, speaker="Dana", text="The green bicycle is ready."),
        ],
    )
    result = h.collect("what did Dana say around 3 pm", state, now=dt.datetime(2026, 10, 23, 17, tzinfo=LA).timestamp())
    quotes = [e["quote"] for e in result["evidence"]]
    assert any("green bicycle" in quote for quote in quotes)
    assert not any("blue bicycle" in quote for quote in quotes)


def test_a_reading_with_no_clock_keeps_the_whole_day():
    query = h.parse("what did Dana say about 3 bicycles yesterday", NOW)
    assert query["at"] is None
    assert query["end"] - query["start"] == 86400


@pytest.mark.parametrize(
    "question",
    ["what did Dana say around 1:30 am on November 1", "what did Dana say around 2:30 am on March 8"],
)
def test_unresolvable_dst_clocks_are_refused_instead_of_shifted(question):
    with pytest.raises(ValueError, match="clock time"):
        h.parse(question, dt.datetime(2026, 11, 3, 17, tzinfo=LA).timestamp())


def test_an_invalid_spelled_date_is_refused_not_dropped():
    with pytest.raises(ValueError, match="calendar date"):
        h.parse("what did Dana say on February 30", NOW)


def test_a_clock_window_never_reads_after_question_time():
    query = h.parse("who was here at 12pm", NOW)
    assert query["start"] == NOW - 300
    assert query["end"] == NOW


@pytest.mark.parametrize("question", ["what did Dana say before 5pm", "what did Dana say on Friday"])
def test_unsupported_time_still_routes_to_an_explicit_history_refusal(state, question):
    result = h.collect(question, state, now=NOW)
    assert "Unsupported time phrase" in result["answer"]
    assert not result["evidence"]


@pytest.mark.parametrize("clock", ["3am", "3"])
def test_last_nights_after_midnight_clock_is_this_morning(clock):
    query = h.parse(f"who was here around {clock} last night", NOW)
    start = dt.datetime.fromtimestamp(query["start"], LA)
    end = dt.datetime.fromtimestamp(query["end"], LA)
    assert (start.day, start.hour, start.minute) == (23, 2, 55)
    assert (end.day, end.hour, end.minute) == (23, 3, 5)


def test_last_night_includes_six_pm_but_not_the_previous_afternoon(state):
    write_voices(state, {1: "Dana"})
    evening = dt.datetime(2026, 10, 22, 18, tzinfo=LA).timestamp()
    afternoon = evening - 60
    transcripts(
        state,
        [
            dict(ts_start=afternoon, ts_end=afternoon + 1, speaker_id=1, speaker="Dana", text="Afternoon bicycle."),
            dict(ts_start=evening, ts_end=evening + 1, speaker_id=1, speaker="Dana", text="Evening bicycle."),
        ],
    )
    result = h.collect("what did Dana say last night", state, now=NOW)
    quotes = [e["quote"] for e in result["evidence"]]
    assert any("Evening bicycle" in quote for quote in quotes), result
    assert not any("Afternoon bicycle" in quote for quote in quotes)


@pytest.mark.parametrize("question", ["what was visible yesterday", "what was the person seated on yesterday"])
@pytest.mark.parametrize("selected_day", [None, "2026-10-22"])
def test_general_visual_context_survives_an_empty_journal_index(state, monkeypatch, question, selected_day):
    from liberational.memory import recall

    first = YESTERDAY + 3600
    ident = add_journal(state, "A person rested on a blue bench beside a cello.", first, 180)
    monkeypatch.setattr(recall, "_hits", create_autospec(recall._hits, return_value=([], [])))
    with cfg.override({"semantic": {"enabled": False}}):
        result = h.collect(question, state, now=NOW, day=selected_day)
    assert any(e.get("interpretation") == ident for e in result["evidence"])
    assert result["evidence"][0]["source"] == "journal"
    assert result["query"]["kind"] == "video"


def test_visual_overview_does_not_leak_into_an_exact_camera_question(state, monkeypatch):
    from liberational.memory import recall

    add_journal(state, "A person rested on a blue bench.", YESTERDAY + 3600, 180)
    overview = create_autospec(recall.overview, side_effect=AssertionError("camera isolation"))
    monkeypatch.setattr(recall, "overview", overview)
    result = h.collect("what was visible on camera2 yesterday", state, now=NOW, day="2026-10-22")
    assert not any(e["source"] == "journal" for e in result["evidence"])
    overview.assert_not_called()


def test_visual_overview_keeps_the_budget_and_ranked_context(state, monkeypatch):
    from liberational.memory import recall

    overview_id = add_journal(state, "A blue bench beside a cello.", YESTERDAY + 300, 180)
    native = [add_journal(state, f"Visible details {n}.", YESTERDAY + 3600 + n * 300, 180) for n in range(4)]
    hits = [
        {
            "payload_id": ident,
            "start_ns": int((YESTERDAY + 3600 + n * 300) * 1e9),
            "end_ns": int((YESTERDAY + 3600 + n * 300 + 180) * 1e9),
            "score": 1.0,
        }
        for n, ident in enumerate(native)
    ]
    monkeypatch.setattr(recall, "_hits", create_autospec(recall._hits, return_value=(hits, [])))
    with cfg.override({"semantic": {"enabled": False}}):
        result = h.collect("what was visible yesterday", state, now=NOW, day="2026-10-22")
    journal = [e for e in result["evidence"] if e["source"] == "journal"]
    assert [e["interpretation"] for e in journal] == [overview_id, native[0], native[1]]
    assert len(journal) == h.JOURNAL_ITEMS


def test_unreadable_visual_overview_keeps_independent_camera_evidence(state, monkeypatch):
    from liberational.memory import recall

    camera(state, [dict(ts=YESTERDAY + 10, cam="cam0", caption="A blue bench is visible.")])
    monkeypatch.setattr(recall, "overview", create_autospec(recall.overview, side_effect=OSError("unavailable")))
    result = h.collect("what was visible yesterday", state, now=NOW, day="2026-10-22")
    assert any(e["source"] == "camera" for e in result["evidence"])
    assert any("overview" in w.lower() and "OSError" in w for w in result["warnings"])


@pytest.mark.parametrize("camera_id", ["front", "rear-left", "side", "upper", "lower"])
def test_a_named_camera_request_is_scoped_and_cites_the_selected_angle(state, tmp_path, monkeypatch, camera_id):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "devices.toml").write_text(
        'device=[{name="cam0",kind="camera"},{name="front",kind="rtsp"},'
        '{name="rear-left",kind="rtsp"},{name="side",kind="rtsp"},'
        '{name="upper",kind="rtsp"},{name="lower",kind="rtsp"}]'
    )
    monkeypatch.setattr(cfg, "OWNER_ROOT", tmp_path)
    camera(
        state,
        [
            dict(ts=NOW - 10, cam="cam0", caption="A yellow umbrella."),
            dict(ts=NOW - 10, cam=camera_id, caption="A violet mug."),
        ],
    )
    add_journal(state, "A yellow umbrella appeared on another camera.", NOW - 30)
    result = ask(state, f"What did the {camera_id.replace('-', ' ')} camera show today?")
    assert result["query"]["cameras"] == [camera_id]
    assert [(item["source"], item["camera"], item["text"]) for item in result["evidence"]] == [
        ("camera", camera_id, "A violet mug.")
    ]
    assert result["evidence"][0]["href"].endswith(f"&cam={camera_id}")
    assert "journal" not in result["coverage"]


def test_in_a_room_does_not_scope_a_question_to_a_camera_named_like_it(state, tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "devices.toml").write_text('device=[{name="cam0",kind="camera"},{name="study",kind="rtsp"}]')
    monkeypatch.setattr(cfg, "OWNER_ROOT", tmp_path)
    camera(state, [dict(ts=NOW - 10, cam="study", caption="A violet mug.")])
    result = ask(state, "What happened in the study today?")
    assert result["query"]["cameras"] == []


def test_camera_caption_citation_seeks_the_actual_camera_angle():
    evidence = h.labeled(h.camera_evidence(dict(ts=NOW, cam="study", caption="A violet mug.")), 1)
    assert evidence["href"] == f"/review#day=2026-10-23&t={NOW}&cam=study"


def test_explicit_journal_request_does_not_mix_in_transcript_or_camera_topics(state, journal_search):
    target = add_journal(state, "A blue bicycle leaned against the fence.", NOW - 60)
    journal_search.near[:] = [target]
    transcripts(state, [dict(ts_start=NOW - 20, speaker="Sam", text="The vehicle is a freight train.")])
    camera(state, [dict(ts=NOW - 10, cam="cam0", caption="A red truck.")])
    result = ask(state, "What vehicle did the journal describe today?")
    assert result["query"]["kind"] == "journal"
    assert result["paths"] == ["journal"]
    assert [(item["source"], item["text"]) for item in result["evidence"]] == [
        ("journal", "A blue bicycle leaned against the fence.")
    ]
    assert "journal" not in result["query"]["search_text"]
    assert "today" not in result["query"]["search_text"]
    assert "vehicle" in result["query"]["search_text"]
    assert "transcript" not in result["coverage"]


def test_ambiguous_camera_name_returns_an_actionable_refusal(state, tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "devices.toml").write_text('device=[{name="north",kind="rtsp"},{name="south",kind="rtsp"}]')
    monkeypatch.setattr(cfg, "OWNER_ROOT", tmp_path)
    with cfg.override({"vision": {"camera_names": ["north: Side camera", "south: Side camera"]}}):
        result = ask(state, "What did the side camera see today?")
    assert result["is_history"]
    assert "Ambiguous camera name" in result["answer"]
    assert result["evidence"] == []


def test_discussion_of_a_room_or_journal_stays_a_transcript_question(state, tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "devices.toml").write_text('device=[{name="garden",kind="rtsp"}]')
    monkeypatch.setattr(cfg, "OWNER_ROOT", tmp_path)
    assert h.parse("What did Dana say about camping in the garden today?", NOW)["kind"] == "transcript"
    assert h.parse("What did Dana say about writing in the journal today?", NOW)["kind"] == "transcript"


def test_read_journal_without_keywords_keeps_the_requested_day(state):
    add_journal(state, "A violet mug was unpacked.", YESTERDAY + 3600)
    result = ask(state, "Read the journal yesterday")
    assert [item["text"] for item in result["evidence"]] == ["A violet mug was unpacked."]
    assert result["coverage"]["journal"]["partial"]


def test_unbounded_journal_browse_requests_a_window_instead_of_scanning_all_history(state):
    result = ask(state, "Read the journal")
    assert "at most two days" in result["answer"]
    assert result["evidence"] == []


@pytest.mark.parametrize("topic", ["using a phone", "drinking water", "sitting down"])
def test_speech_about_retired_actions_still_reads_the_transcript(state, topic):
    transcripts(state, [dict(ts_start=NOW - 10, speaker="Dana", text=f"We discussed {topic} during the picnic.")])
    result = ask(state, f"What did Dana say about {topic} today?")
    assert result["query"]["kind"] == "transcript"
    assert any(topic in evidence["quote"] for evidence in result["evidence"])
    assert "being rebuilt" not in str(result.get("answer"))
