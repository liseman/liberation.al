"""The model's reading of a question (memory/question.py): what a reply becomes, and what is refused."""

import datetime as dt
import json
from zoneinfo import ZoneInfo

import pytest

from liberational.assistant import voice_auth
from liberational.memory import history_qa, question, recall

LA = ZoneInfo("America/Los_Angeles")
NOW = dt.datetime(2026, 10, 23, 12, tzinfo=LA).timestamp()
READ = question.read  # before the suite's readings fixture stands in for it


@pytest.fixture(autouse=True)
def la_zone(monkeypatch):
    monkeypatch.setattr(history_qa, "TZ", LA)


def reply(**fields):
    """A model reply: the whole form under its lettered keys, with these fields filled in."""
    assert set(fields) <= set(question.FORM)
    return {question.KEY[name]: value for name, value in {**question.FORM, **fields}.items()}


def local(ts):
    return dt.datetime.fromtimestamp(ts, LA).strftime("%m-%d %H:%M")


def test_the_forms_keys_sort_in_the_order_the_model_fills_them():
    # Ollama writes a constrained reply's keys alphabetically: the time's words and kind must come before its fields.
    keys = sorted(question.SCHEMA["properties"])
    assert keys == list(question.SCHEMA["required"])
    assert keys.index("i_time_phrase") < keys.index("j_time_kind") < keys.index("k_day") < keys.index("n_clock")


def test_a_reply_becomes_a_reading():
    asked = "What did Dana say about the garden yesterday at 3pm?"
    r = question.from_reply(
        reply(people=["Dana"], time_phrase="yesterday at 3pm", time_kind="calendar", day="yesterday", clock="3pm"),
        asked,
    )
    assert r == question.Reading(people=("Dana",), day="yesterday", clock="3pm")


@pytest.mark.parametrize(
    ("asked", "fields", "kept"),
    [
        pytest.param(
            "what was on the desk last night around 11",
            dict(time_kind="calendar", day="last week", part="last night", clock="11pm"),
            dict(part="last night", clock="11"),
            id="a-day-and-a-half-of-the-day-nobody-said",
        ),
        pytest.param(
            "what did the camera see yesterday morning",
            dict(time_kind="calendar", day="yesterday", part="this morning"),
            dict(day="yesterday", part="morning"),
            id="this-morning-when-it-said-yesterday-morning",
        ),
        pytest.param(
            "who was here at three",
            dict(time_kind="calendar", clock="3"),
            dict(clock="3"),
            id="a-clock-said-as-a-word",
        ),
        pytest.param(
            "who was here at noon on October 20",
            dict(time_kind="calendar", date="10-20", clock="noon"),
            dict(day="10-20", clock="noon"),
            id="a-date-and-noon",
        ),
    ],
)
def test_time_values_the_question_doesnt_hold_are_dropped(asked, fields, kept):
    r = question.from_reply(reply(time_phrase="x", **fields), asked)
    assert r == question.Reading(**kept)


@pytest.mark.parametrize(
    ("asked", "fields", "phrase"),
    [
        pytest.param(
            "summarize what happened on October 20",
            dict(time_phrase="on October 20", time_kind="calendar", day="today"),
            "on October 20",
            id="only-a-day-nobody-said",
        ),
        pytest.param(
            "what happened in the last few hours",
            dict(time_phrase="in the last few hours", time_kind="last", span=3, span_unit="hours"),
            "in the last few hours",
            id="an-amount-nobody-said",
        ),
        pytest.param(
            "did anyone mention the budget last month",
            dict(time_phrase="last month", time_kind="other"),
            "last month",
            id="a-time-the-form-cant-hold",
        ),
    ],
)
def test_a_time_left_with_nothing_is_refused_not_dropped(asked, fields, phrase):
    r = question.from_reply(reply(**fields), asked)
    assert r.other_time == phrase
    with pytest.raises(ValueError, match="Unsupported time phrase"):
        history_qa.window_of(r, NOW)


def test_fields_of_another_time_kind_cant_combine_into_a_window():
    r = question.from_reply(
        reply(time_phrase="in the last 2 hours", time_kind="last", span=2, span_unit="hours", day="today", clock="2"),
        "what happened today in the last 2 hours",
    )
    assert r == question.Reading(span=2.0, span_unit="hours", span_kind="last")
    assert history_qa.window_of(r, NOW)[:3] == (NOW - 7200, NOW, "last 2 hours")


@pytest.mark.parametrize(
    ("asked", "people", "kept"),
    [
        pytest.param("what did we decide about the van", ["we", "Dana"], (), id="nobody-in-particular-or-unnamed"),
        pytest.param("what did Dana tell me yesterday", ["dana", "me"], ("dana", "me"), id="named-and-the-asker"),
        pytest.param("who was here yesterday", ["me"], (), id="the-asker-when-nobody-said-i"),
    ],
)
def test_people_are_the_names_the_question_holds(asked, people, kept):
    assert question.from_reply(reply(people=people), asked).people == kept


@pytest.mark.parametrize(
    ("fields", "source", "movement"),
    [
        pytest.param(dict(source="people", measure="count", movement="came_in"), "cameras", "came_in", id="a-count"),
        pytest.param(dict(source="people", measure="who", movement="came_in"), "people", "", id="who-was-here"),
        pytest.param(
            dict(source="cameras", measure="describe", movement="walked_by"), "cameras", "walked_by", id="seen"
        ),
    ],
)
def test_people_coming_in_are_a_camera_count_and_who_was_here_is_no_movement(fields, source, movement):
    r = question.from_reply(reply(**fields), "how many people came in")
    assert (r.source, r.movement) == (source, movement)


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param("not a form", id="not-an-object"),
        pytest.param(reply(source="memories"), id="an-unknown-choice"),
        pytest.param(reply(history="yes"), id="a-flag-that-isnt-one"),
        pytest.param(reply(people="Dana"), id="people-not-a-list"),
        pytest.param(reply(span="two"), id="an-amount-that-isnt-a-number"),
    ],
)
def test_a_reply_that_doesnt_fit_the_form_is_unreadable(bad):
    with pytest.raises(question.Unreadable):
        question.from_reply(bad, "who was here yesterday")


def test_every_worked_example_is_a_whole_form_that_reads_cleanly():
    for asked, answer in question.SHOTS:
        filled = json.loads(answer)
        assert list(filled) == list(question.KEY.values())
        r = question.from_reply(filled, asked)
        if filled[question.KEY["time_kind"]] not in ("none", "other"):
            assert not r.other_time, asked  # the example's own time fields are in its question's words


@pytest.fixture
def read():
    """question.read itself, not the suite's fixed readings, with nothing kept from another test."""
    question.forget()
    yield READ
    question.forget()


def test_read_asks_the_model_once_with_the_form_and_its_examples(monkeypatch, read):
    calls = []

    def chat(system, user, schema, num_predict, seconds, shots=()):
        calls.append((system, user, schema, shots))
        return reply(source="people", measure="who", time_phrase="yesterday", time_kind="calendar", day="yesterday")

    monkeypatch.setattr(recall, "chat_within", chat)
    first = read("who was  here yesterday")
    second = read("who was here yesterday")
    assert first == second == question.Reading(source="people", measure="who", day="yesterday")
    assert len(calls) == 1
    system, user, schema, shots = calls[0]
    assert (user, schema, shots) == ("who was here yesterday", question.SCHEMA, question.SHOTS)
    assert "i_time_phrase" in system


def test_no_reply_from_the_model_is_unreadable_and_not_kept(monkeypatch, read):
    def down(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(recall, "chat_within", down)
    with pytest.raises(question.Unreadable, match="OSError"):
        read("who was here yesterday")
    monkeypatch.setattr(recall, "chat_within", lambda *a, **k: reply())
    assert read("who was here yesterday") == question.Reading()


def test_a_reading_handed_to_another_process_is_kept_without_asking_the_model(monkeypatch, read):
    def refuse(*args, **kwargs):
        raise AssertionError("the model was asked again")

    monkeypatch.setattr(recall, "chat_within", refuse)
    reading = question.Reading(source="people", measure="who", people=("Dana",), day="yesterday")
    sent = json.loads(json.dumps(reading.as_dict()))
    received = question.handed(sent)
    assert received == reading
    question.keep("was Dana here yesterday", reading)
    assert read("was Dana here  yesterday") == reading
    assert question.kept("was Dana here yesterday") == reading
    assert question.kept("was Sam here yesterday") is None


@pytest.mark.parametrize("fields", [None, "speech", {"source": "speech", "mood": "curious"}])
def test_something_that_isnt_a_reading_is_not_handed_over(fields):
    assert question.handed(fields) is None


def test_the_oldest_reading_is_dropped_past_the_limit(monkeypatch, read):
    monkeypatch.setattr(question, "KEEP", 2)
    for asked in ("one", "two", "three"):
        question.keep(asked, question.Reading())
    assert [question.kept(asked) is not None for asked in ("one", "two", "three")] == [False, True, True]


def unreadable(monkeypatch):
    def read(asked):
        raise question.Unreadable("TimeoutError")

    monkeypatch.setattr(question, "read", read)


def test_collect_says_a_question_it_couldnt_read_and_finds_nothing(monkeypatch, tmp_path):
    unreadable(monkeypatch)
    result = history_qa.collect("who was here yesterday", tmp_path, NOW)
    assert result["answer"] == history_qa.UNREADABLE
    assert result["evidence"] == []
    assert result["coverage"] == {"read_error": True}
    assert result["query"] == {}


def test_routing_takes_an_unread_question_as_not_history_but_the_password_check_fails_closed(monkeypatch):
    unreadable(monkeypatch)
    assert history_qa.is_history_question("what did the camera see") is False
    assert voice_auth.asks_about_the_past("what did the camera see") is True
    assert history_qa.people_named("when did I last see Dana") == []


def test_routing_follows_the_readings_history(readings):
    readings["what did we decide"] = question.Reading(history=True)
    readings["what is a quasar"] = question.Reading(history=False)
    assert history_qa.is_history_question("what did we decide") is True
    assert history_qa.is_history_question("what is a quasar") is False


def test_people_named_leaves_out_the_asker(readings):
    readings["what did Dana tell me"] = question.Reading(people=("Dana", "me"))
    assert history_qa.people_named("what did Dana tell me") == ["Dana"]


@pytest.mark.parametrize(
    ("reading", "window", "first", "last"),
    [
        pytest.param(
            question.Reading(span=3, span_unit="days", span_kind="ago"),
            "2026-10-20",
            "10-20 00:00",
            "10-21 00:00",
            id="days-ago-is-that-day",
        ),
        pytest.param(
            question.Reading(part="this morning", clock="9"),
            "around 09:00",
            "10-23 08:55",
            "10-23 09:05",
            id="a-bare-clock-takes-its-part-of-the-days-half",
        ),
        pytest.param(
            question.Reading(day="yesterday", part="afternoon", clock="3"),
            "around 15:00",
            "10-22 14:55",
            "10-22 15:05",
            id="yesterday-afternoon-at-3",
        ),
        pytest.param(
            question.Reading(day="october 20"), "2026-10-20", "10-20 00:00", "10-21 00:00", id="a-months-name"
        ),
        pytest.param(
            question.Reading(day="2026-10-20", clock="2", clock_end="4pm"),
            "between 14:00 and 16:00",
            "10-20 14:00",
            "10-20 16:00",
            id="a-range-on-a-date",
        ),
        pytest.param(
            question.Reading(source="people", measure="who", now=True),
            "last hour",
            "10-23 11:00",
            "10-23 12:00",
            id="who-is-home",
        ),
    ],
)
def test_a_readings_time_fields_give_one_window(reading, window, first, last):
    start, end, words, _ = history_qa.window_of(reading, NOW)
    assert (words, local(start), local(end)) == (window, first, last)


@pytest.mark.parametrize(
    ("reading", "message"),
    [
        pytest.param(question.Reading(part="this morning", clock="3pm"), "isn't in the morning", id="wrong-half"),
        pytest.param(question.Reading(day="last week", clock="3pm"), "single time window", id="a-week-and-a-clock"),
        pytest.param(question.Reading(clock_end="4pm"), "complete time range", id="half-a-range"),
        pytest.param(question.Reading(day="02-30"), "valid calendar date", id="no-such-date"),
        pytest.param(question.Reading(day="next tuesday"), "Unsupported time phrase", id="not-a-date"),
        pytest.param(
            question.Reading(span=2.5, span_unit="days", span_kind="ago"), "Unsupported time phrase", id="half-days-ago"
        ),
    ],
)
def test_time_fields_that_name_no_single_window_are_refused(reading, message):
    with pytest.raises(ValueError, match=message):
        history_qa.window_of(reading, NOW)


@pytest.mark.parametrize(
    ("reading", "kind"),
    [
        (question.Reading(source="speech"), "transcript"),
        (question.Reading(source="cameras"), "video"),
        (question.Reading(source="both"), "mixed"),
        (question.Reading(source="people", measure="who"), "who"),
        (question.Reading(source="people", measure="last_time", people=("Dana",)), "camera"),
        (question.Reading(source="people", measure="last_time"), "who"),
        (question.Reading(source="lights", measure="duration"), "light_duration"),
        (question.Reading(source="lights", measure="count"), "lights"),
        (question.Reading(source="journal", text="trip"), "journal"),
        (question.Reading(source="people", measure="who", movement="came_in"), "who"),
    ],
)
def test_the_reading_decides_the_querys_kind(reading, kind):
    assert history_qa.parse("a question", NOW, reading=reading)["kind"] == kind


def test_the_asker_is_the_owner_in_a_transcript_query():
    query = history_qa.parse("what did I say", NOW, reading=question.Reading(people=("me",)))
    assert query["first_person"] is True
    assert query["person"] == query["participants"][0] == history_qa.cfg.owner_or()
