"""Right and wrong marks (dashboard/marks.py): People's are identification labels and Ask's are retrieval labels,
each one a label tools/measure.py scores."""

import datetime as dt
import http.client
import importlib.util
import json
import pathlib
from zoneinfo import ZoneInfo

import pytest

import speech_fake
from liberational import config as cfg
from liberational.dashboard import marks
from liberational.memory import history_qa
from voices import write_voices

_spec = importlib.util.spec_from_file_location(
    "measure", pathlib.Path(__file__).resolve().parents[1] / "tools" / "measure.py"
)
assert _spec is not None and _spec.loader is not None
measure = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(measure)

LINE = {"kind": "line", "ts_start": 1793000005.0, "ts_end": 1793000008.0, "device": "kitchen", "shown": "Alice"}
SHOT = {"kind": "shot", "sighting": "d-42", "ts": 1793000100.5, "camera": "cam1", "shown": "Alice"}


@pytest.fixture
def folder(tmp_path):
    with cfg.override({"paths": {"measurement": str(tmp_path / "measurement")}}):
        yield tmp_path / "measurement"


def saved(folder):
    return [json.loads(line) for line in (folder / marks.FILE).read_text().splitlines()]


def test_a_right_mark_on_a_line_scores_as_right(folder, tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    lines = [
        {"ts_start": 1793000005.0, "ts_end": 1793000008.0, "speaker": "Alice", "speaker_id": 1, "device": "kitchen"},
        {"ts_start": 1793000006.0, "ts_end": 1793000007.0, "speaker": "Dana", "speaker_id": 2, "device": "kitchen"},
    ]  # a second voice overlapping the first line: a mark is about its own line only
    speech_fake.fake().write(lines)
    write_voices(state, {1: "Alice", 2: "Dana"})
    assert marks.mark({**LINE, "right": True}, control=True)[0] == 200
    score = measure.score_identification(measure.load(folder, "identification"), state)
    assert (score["tp"], score["fp"], score["fn"]) == (1, 0, 0)


def test_a_wrong_mark_says_who_it_was_or_that_it_was_nobody_known(folder):
    marks.mark({**LINE, "right": False, "actual": "Dana"}, control=True)
    marks.mark({**SHOT, "right": False}, control=True)
    line, shot = saved(folder)
    assert line == {
        "id": "people-line-kitchen@1793000005.000",
        "modality": "voice",
        "start": 1793000005.0,
        "end": 1793000008.0,
        "line": 1793000005.0,
        "device": "kitchen",
        "people": ["Dana"],
        "note": "marked wrong in People (shown as Alice)",
    }
    assert shot == {
        "id": "people-shot-d-42",
        "modality": "face",
        "camera": "cam1",
        "at": 1793000100.5,
        "sighting": "d-42",
        "people": [],
        "note": "marked wrong in People (shown as Alice)",
    }
    assert [c.id for c in measure.load(folder, "identification")] == [
        "people-line-kitchen@1793000005.000",
        "people-shot-d-42",
    ]


def test_marking_the_same_record_again_replaces_its_label_and_keeps_the_rest(folder):
    folder.mkdir()
    (folder / marks.FILE).write_text(
        '{"id": "by-hand", "modality": "voice", "start": 1, "end": 2, "people": []}\nnot json\n'
    )
    marks.mark({**SHOT, "right": True}, control=True)
    marks.mark({**SHOT, "right": False, "actual": "Dana"}, control=True)
    text = (folder / marks.FILE).read_text().splitlines()
    assert text[:2] == ['{"id": "by-hand", "modality": "voice", "start": 1, "end": 2, "people": []}', "not json"]
    assert [json.loads(line)["people"] for line in text[2:]] == [["Dana"]]


@pytest.mark.parametrize(
    ("body", "error"),
    [
        ({**LINE, "right": "yes"}, "right is true"),
        ({**LINE, "shown": " "}, "shown is the name"),
        ({**LINE, "right": True, "actual": "Dana"}, "only on a wrong mark"),
        ({**LINE, "right": False, "actual": "alice"}, "someone other than the name shown"),
        ({**LINE, "right": True, "ts_end": 1793000000.0}, "a line needs its ts_start and ts_end"),
        ({**SHOT, "right": True, "sighting": True}, "a shot needs its sighting id"),
        ({**SHOT, "right": True, "kind": "frame"}, 'kind is "line", "shot", "answer", "event", or "entries"'),
    ],
)
def test_a_mark_that_cannot_be_a_label_is_refused_and_writes_nothing(folder, body, error):
    code, reply = marks.mark(body, control=True)
    assert (code, reply["ok"]) == (400, False)
    assert error in reply["error"]
    assert not folder.exists()


def test_marks_are_made_on_this_machine_only(folder):
    assert marks.mark({**LINE, "right": True}, control=False)[0] == 403
    assert not (folder / marks.FILE).exists()


def test_the_route_saves_a_mark_from_a_same_origin_page(folder, serve_dashboard, monkeypatch):
    host = serve_dashboard().removeprefix("http://")

    def post(body, origin):
        c = http.client.HTTPConnection(host, timeout=5)
        c.request("POST", "/measurement/mark", json.dumps(body), {"Content-Type": "application/json", "Origin": origin})
        r = c.getresponse()
        out = r.status, json.loads(r.read())
        c.close()
        return out

    assert post({**SHOT, "right": True}, "http://evil.example")[0] == 403
    assert not (folder / marks.FILE).exists()
    code, reply = post({**SHOT, "right": True}, f"http://{host}")
    assert (code, reply["label"]["people"]) == (200, ["Alice"])
    assert saved(folder)[0]["sighting"] == "d-42"


# Ask answers

LA = ZoneInfo("America/Los_Angeles")
ASKED = dt.datetime(2026, 10, 21, 9, 0, tzinfo=LA).timestamp()
SAID = dt.datetime(2026, 10, 20, 15, 0, 5, tzinfo=LA).timestamp()
QUESTION = "What did Dana say about the garden yesterday?"
ANSWER = {"kind": "answer", "question": QUESTION, "asked_at": ASKED}


def test_a_right_mark_keeps_the_cited_records_and_scores_as_passed(folder, tmp_path, monkeypatch):
    monkeypatch.setattr(history_qa, "TZ", LA)
    state = tmp_path / "state"
    state.mkdir()
    line = {"ts_start": SAID, "ts_end": SAID + 3, "speaker_id": 3, "device": "kitchen", "text": "Plant tomatoes."}
    speech_fake.fake().write([line])
    write_voices(state, {3: "Dana"})
    found = history_qa.collect(QUESTION, state, now=ASKED)["evidence"]  # what the dashboard's answer cited
    cited = [{"ts": e["ts"], "source": e["source"]} for e in found]
    assert marks.mark({**ANSWER, "right": True, "cited": cited}, control=True)[0] == 200
    assert saved_retrieval(folder)[0]["evidence"] == [{"at": SAID, "within_s": 1, "source": "transcript"}]
    score = measure.score_retrieval(measure.load(folder, "retrieval"), state)
    assert score["per_question"][0]["passed"], score


def saved_retrieval(folder):
    return [json.loads(line) for line in (folder / marks.RETRIEVAL).read_text().splitlines()]


def test_a_wrong_mark_keeps_the_correction_in_place_of_an_earlier_mark(folder):
    marks.mark({**ANSWER, "right": True, "says": "tomatoes"}, control=True)
    body = {**ANSWER, "day": "2026-10-20", "right": False, "says": "tomatoes", "not_says": "tulips", "moment": SAID}
    assert marks.mark(body, control=True)[0] == 200
    assert saved_retrieval(folder) == [
        {
            "id": f"ask-{ASKED:.3f}",
            "question": QUESTION,
            "asked_at": ASKED,
            "day": "2026-10-20",
            "answer_includes": ["tomatoes"],
            "answer_excludes": ["tulips"],
            "evidence": [{"at": SAID, "within_s": 30}],
            "note": "marked wrong in Ask",
        }
    ]
    assert [q.id for q in measure.load(folder, "retrieval")] == [f"ask-{ASKED:.3f}"]
    assert not (folder / marks.FILE).exists()  # an answer is never an identification label


@pytest.mark.parametrize(
    ("body", "error"),
    [
        pytest.param({**ANSWER, "right": False}, "a wrong mark needs", id="wrong-checks-nothing"),
        pytest.param(
            {**ANSWER, "right": False, "cited": [{"ts": SAID}]}, "cited is for a right mark", id="wrong-cites"
        ),
        pytest.param({**ANSWER, "right": True, "moment": SAID}, "for a wrong mark", id="right-with-moment"),
        pytest.param(
            {**ANSWER, "right": True, "cited": [{"ts": SAID, "source": "facts"}]}, "a source of", id="unknown-source"
        ),
        pytest.param({**ANSWER, "right": True, "says": "x" * 201}, "1 to 200", id="says-too-long"),
        pytest.param(
            {**ANSWER, "right": False, "says": "Alice and Dana", "not_says": "alice"},
            "could never pass",
            id="contradicts",
        ),
        pytest.param({**ANSWER, "right": True, "cited": [{"ts": 1}] * 201}, "at most 200", id="too-many-cited"),
        pytest.param({**ANSWER, "right": True, "says": "x", "day": "2026-10-32"}, "YYYY-MM-DD", id="bad-day"),
        pytest.param({**ANSWER, "right": True, "says": "x", "question": " "}, "question is", id="no-question"),
        pytest.param({**ANSWER, "right": True, "says": "x", "asked_at": None}, "asked_at", id="no-time"),
    ],
)
def test_an_answer_mark_that_cannot_be_a_label_is_refused_and_writes_nothing(folder, body, error):
    code, reply = marks.mark(body, control=True)
    assert (code, reply["ok"]) == (400, False)
    assert error in reply["error"]
    assert not folder.exists()


def test_a_plain_yes_with_nothing_cited_is_saved_as_feedback_not_a_label(folder):
    code, reply = marks.mark({**ANSWER, "right": True}, control=True)
    assert (code, reply["ok"]) == (200, True)
    assert not (folder / marks.RETRIEVAL).exists()  # measure.py needs something to check
    saved = [json.loads(line) for line in (folder / marks.FEEDBACK).read_text().splitlines()]
    assert saved == [{"id": f"ask-{ASKED:.3f}", "question": QUESTION, "asked_at": ASKED, "right": True}]


def test_a_right_mark_keeps_every_record_a_long_answer_cited(folder):
    # Ask keeps up to 100 records and adds up to 60 from its search rounds, any of which an answer can cite.
    cited = [{"ts": SAID + n, "source": "transcript"} for n in range(160)]
    assert marks.mark({**ANSWER, "right": True, "cited": cited}, control=True)[0] == 200
    assert len(saved_retrieval(folder)[0]["evidence"]) == 160


def test_two_microphones_lines_in_one_millisecond_are_two_marks(folder):
    marks.mark({**LINE, "right": True}, control=True)
    marks.mark({**LINE, "device": "hall", "right": False}, control=True)
    marks.mark({**{k: v for k, v in LINE.items() if k != "device"}, "right": True}, control=True)  # a line with none
    assert [x["id"] for x in saved(folder)] == [
        "people-line-kitchen@1793000005.000",
        "people-line-hall@1793000005.000",
        "people-line-1793000005.000",
    ]


# ---- events: a span of one camera where something happened, counted by watching ----

FRIDGE = {
    "kind": "event",
    "person": "Dana",
    "camera": "cam1",
    "what": "opened the fridge",
    "start": 1793000000.0,
    "end": 1793000040.5,
    "count": 3,
    "attributes": {"door open": {"value": 12, "unit": "s"}},
    "notes": [{"nth": 2, "note": " only half in view "}, {"note": "light on throughout"}],
}


def test_an_event_mark_is_a_label_the_measure_tool_loads(folder):
    assert marks.mark(FRIDGE, control=True)[0] == 200
    (label,) = measure.load(folder, "events")
    assert (label.person, label.camera, label.what, label.count) == ("Dana", "cam1", "opened the fridge", 3)
    assert (label.start, label.end) == (1793000000.0, 1793000040.5)
    assert label.attributes == {"door open": {"value": 12, "unit": "s"}}
    assert label.notes == ({"note": "only half in view", "nth": 2}, {"note": "light on throughout"})


@pytest.mark.parametrize("nobody", [pytest.param({"person": ""}, id="empty"), pytest.param({}, id="left-out")])
def test_an_event_mark_needs_no_one(folder, nobody):
    body = {k: v for k, v in FRIDGE.items() if k != "person"} | nobody
    assert marks.mark({**body, "what": "the door was left open"}, control=True)[0] == 200
    (label,) = measure.load(folder, "events")
    assert (label.person, label.what) == (None, "the door was left open")


def test_marking_the_same_span_again_replaces_its_label_and_another_span_adds_one(folder):
    marks.mark(FRIDGE, control=True)
    marks.mark({**FRIDGE, "count": 4, "attributes": {}, "notes": []}, control=True)
    marks.mark({**FRIDGE, "camera": "cam0"}, control=True)  # the same time on another camera is another span
    labels = measure.load(folder, "events")
    assert [(a.camera, a.count) for a in labels] == [("cam1", 4), ("cam0", 3)]
    assert labels[0].attributes == {} and labels[0].notes == ()


@pytest.mark.parametrize(
    ("change", "error"),
    [
        pytest.param({"person": " "}, "person is who it involved", id="blank-person"),
        pytest.param({"what": ""}, "what is what happened", id="no-what"),
        pytest.param({"end": 1793000000.0}, "a span needs its start and a later end", id="empty-span"),
        pytest.param({"count": -1}, "count is how many times", id="negative-count"),
        pytest.param({"count": 2.5}, "count is how many times", id="fractional-count"),
        pytest.param({"count": True}, "count is how many times", id="boolean-count"),
        pytest.param({"attributes": {"door open": {"value": 12}}}, "needs a unit", id="attribute-without-unit"),
        pytest.param({"attributes": {"door open": {"value": "long", "unit": "s"}}}, "a value, a number", id="word"),
        pytest.param({"notes": [{"nth": 4, "note": "past the count"}]}, "from 1 to the count", id="note-past-count"),
        pytest.param({"count": 0, "notes": [{"nth": 1, "note": "none"}]}, "from 1 to the count", id="note-of-none"),
        pytest.param({"notes": [{"note": "  "}]}, "each note says something", id="empty-note"),
    ],
)
def test_an_event_mark_that_cannot_be_a_label_is_refused_and_writes_nothing(folder, change, error):
    code, reply = marks.mark({**FRIDGE, **change}, control=True)
    assert code == 400 and error in reply["error"]
    assert not (folder / marks.EVENTS).exists()


DOOR = {
    "kind": "entries",
    "camera": "cam1",
    "start": 1793000000.0,
    "end": 1793000030.0,
    "events": [{"kind": "exit", "at": 1793000028.0}, {"kind": "entry", "at": 1793000004.0}],
    "unsure": [{"start": 1793000010.0, "end": 1793000015.0, "reason": " the door hides the edge "}],
    "conditions": ["occlusion"],
    "note": "kitchen door half shut",
}


def test_an_entries_mark_is_a_window_the_measure_tool_loads(folder):
    code, reply = marks.mark(DOOR, control=True)
    assert (code, reply["replaced"]) == (200, 0)
    (window,) = measure.load(folder, "entries")
    assert (window.camera, window.start, window.end) == ("cam1", 1793000000.0, 1793000030.0)
    assert [(e.kind, e.at) for e in window.events] == [("entry", 1793000004.0), ("exit", 1793000028.0)]
    assert [(u.start, u.end, u.reason) for u in window.unsure] == [
        (1793000010.0, 1793000015.0, "the door hides the edge")
    ]
    assert window.conditions == ("occlusion",)


def test_an_empty_window_needs_the_owner_to_say_nobody_came_or_went(folder):
    empty = {**DOOR, "events": [], "unsure": [], "conditions": []}
    code, reply = marks.mark(empty, control=True)
    assert code == 400 and "none_seen" in reply["error"]
    assert not (folder / marks.ENTRIES).exists()
    assert marks.mark({**empty, "none_seen": True}, control=True)[0] == 200
    (window,) = measure.load(folder, "entries")
    assert (window.events, window.conditions) == ((), ("no_events",))


def test_relabeling_a_window_replaces_it_and_an_overlap_needs_replace(folder):
    marks.mark(DOOR, control=True)
    marks.mark({**DOOR, "events": [], "none_seen": True, "unsure": []}, control=True)  # the same window again
    assert [w.events for w in measure.load(folder, "entries")] == [()]
    later = {**DOOR, "start": 1793000020.0, "end": 1793000060.0, "events": [], "unsure": [], "none_seen": True}
    code, reply = marks.mark(later, control=True)
    assert code == 409
    assert reply["overlaps"] == [
        {"id": "review-cam1-1793000000.000-1793000030.000", "start": 1793000000.0, "end": 1793000030.0}
    ]
    assert marks.mark({**later, "camera": "cam0"}, control=True)[0] == 200  # another camera never overlaps
    code, reply = marks.mark({**later, "replace": True}, control=True)
    assert (code, reply["replaced"]) == (200, 1)
    assert sorted((w.camera, w.start) for w in measure.load(folder, "entries")) == [
        ("cam0", 1793000020.0),
        ("cam1", 1793000020.0),
    ]
    adjoining = {**later, "start": 1793000060.0, "end": 1793000090.0}  # half-open: touching is not overlapping
    assert marks.mark(adjoining, control=True)[0] == 200


@pytest.mark.parametrize(
    ("change", "error"),
    [
        pytest.param({"camera": ""}, "camera is the camera", id="no-camera"),
        pytest.param({"end": 1793000000.0}, "a later end", id="empty-window"),
        pytest.param({"events": [{"kind": "arrival", "at": 1793000004.0}]}, "an entry or an exit", id="bad-kind"),
        pytest.param({"events": [{"kind": "entry", "at": 1793000030.0}]}, "before its end", id="event-at-end"),
        pytest.param({"events": [{"kind": "entry", "at": 1793000012.0}]}, "is a guess", id="event-in-unsure"),
        pytest.param(
            {"unsure": [{"start": 1793000010.0, "end": 1793000015.0, "reason": ""}]}, "says why", id="no-reason"
        ),
        pytest.param(
            {
                "unsure": [
                    {"start": 1793000010.0, "end": 1793000015.0, "reason": "door"},
                    {"start": 1793000014.0, "end": 1793000016.0, "reason": "dark"},
                ]
            },
            "must not overlap",
            id="overlapping-unsure",
        ),
        pytest.param(
            {"unsure": [{"start": 1793000025.0, "end": 1793000031.0, "reason": "door"}]}, "inside the window", id="out"
        ),
        pytest.param({"conditions": ["foggy"]}, "conditions are any of", id="unknown-condition"),
    ],
)
def test_an_entries_mark_that_cannot_be_a_label_is_refused_and_writes_nothing(folder, change, error):
    code, reply = marks.mark({**DOOR, **change}, control=True)
    assert code == 400 and error in reply["error"]
    assert not (folder / marks.ENTRIES).exists()
