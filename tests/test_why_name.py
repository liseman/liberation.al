"""Why a name was given: GET /identity/why explains a transcript line's or a recorded face's name from the
assertion that decided it, with the match score behind it. It changes nothing."""

import time

import pytest

from liberational.dashboard import context, server
from liberational.domain.identity import Basis
from liberational.listen import voices as live_voices
from liberational.people import registry
from speech_fake import fake
from voices import write_voices


def line(ts, text, sid=1, sim=0.71, device=None, speaker="Speaker 1", **extra):
    e = {"ts_start": ts, "ts_end": ts + 2, "text": text, "speaker": speaker, "speaker_id": sid}
    e |= {"speaker_sim": sim, **({"device": device} if device else {}), **extra}
    return e


LINES = [
    line(1000.0, "Morning."),
    line(1010.0, "Tea?", device="Kitchen"),  # two microphones' lines in one millisecond
    line(1010.0, "Coffee, please.", device="Hall", sim=0.6),
]


@pytest.fixture
def state(tmp_path, monkeypatch):
    """Voice 1, which the owner named Dana in People; three lines heard on it."""
    write_voices(tmp_path, {1: {"label": "Dana"}}, source="dashboard")
    fake().write(LINES)
    monkeypatch.setattr(context, "STATE", tmp_path)
    return tmp_path


def why(**q):
    return server.why_name_view({k: str(v) for k, v in q.items()})


def test_a_line_on_a_named_voice_is_explained_by_the_voices_name_and_its_match(state):
    body, code = why(line="1000.000")
    assert code == 200
    assert {k: body[k] for k in ("name", "subject", "via", "basis", "source", "similarity", "threshold")} == {
        "name": "Dana",
        "subject": "line",
        "via": "voice",
        "basis": "owner_tag",
        "source": "dashboard",
        "similarity": 0.71,
        "threshold": live_voices.THRESHOLD,  # the live matcher's own: 0.45 for ReDimNet2 since 196772ac
    }
    assert body["voice"] == {"id": 1, "label": "Dana"}
    assert body["mark"] == {"kind": "line", "ts_start": 1000.0, "ts_end": 1002.0}
    assert isinstance(body["asserted_at"], float)


def test_a_line_the_owner_named_itself_is_explained_by_that_and_only_that_line(state):
    registry.at(state).claim(registry.utterance(1010.0, "Hall"), "Eve", Basis.CORRECTION, "dashboard")
    hall, _ = why(line="hall@1010.000")
    kitchen, _ = why(line="kitchen@1010.000")
    assert (hall["name"], hall["via"], hall["basis"]) == ("Eve", "line", "correction")
    assert hall["similarity"] is None  # its score was for Dana's voice, not Eve
    assert hall["mark"]["device"] == "Hall"  # a mark names the very line
    assert (kitchen["name"], kitchen["via"], kitchen["basis"]) == ("Dana", "voice", "owner_tag")


def test_a_line_on_no_voice_with_no_name_says_nobody_named_it(tmp_path, monkeypatch):
    fake().write([line(1000.0, "Hi.", sid=None, sim=None)])
    monkeypatch.setattr(context, "STATE", tmp_path)
    body, code = why(line="1000.000")
    assert (code, body["via"], body["basis"], body["similarity"], body["voice"]) == (200, "none", None, None, None)


def test_a_line_taken_away_from_its_voice_says_so(state):
    registry.at(state).detach(registry.utterance(1000.0), "unknown", "dashboard", via=registry.voiceprint(1))
    body, _ = why(line="1000.000")
    assert (body["name"], body["via"], body["similarity"]) == ("unknown", "detached", None)


def test_a_name_written_on_the_line_before_the_log_is_the_lines_own(state):
    fake().write([line(1000.0, "Hi.", speaker="Eve", sim=None)])
    body, _ = why(line="1000.000")
    assert (body["name"], body["via"], body["basis"], body["similarity"]) == ("Eve", "hand", None, None)


def test_a_line_keeps_the_name_it_was_heard_as_after_its_voice_is_someone_elses(state):
    fake().write([line(1000.0, "Hi.", speaker="Dana")])
    registry.at(state).claim(registry.voiceprint(1), "Eve", Basis.CORRECTION, "dashboard")
    body, _ = why(line="1000.000")
    assert (body["name"], body["via"], body["basis"]) == ("Dana", "heard", None)
    assert body["voice"] == {"id": 1, "label": "Eve"}


@pytest.mark.parametrize(
    ("query", "code"),
    [
        pytest.param({"line": "1234.000"}, 404, id="no-such-line"),
        pytest.param({"ts_start": 1010.0}, 409, id="a-time-naming-two-microphones"),
        pytest.param({"ts_start": "soon"}, 400, id="not-a-time"),
        pytest.param({}, 400, id="nothing-asked"),
        pytest.param({"shot": "not/an-id"}, 400, id="not-a-sighting-id"),
        pytest.param({"shot": "no-such-id"}, 404, id="no-such-shot"),
    ],
)
def test_a_request_it_cant_answer_says_why(state, query, code):
    assert why(**query)[1] == code


def test_a_recorded_face_is_explained_by_its_match_until_the_owner_corrects_it(state, vision):
    ts = round(time.time() - 60.0, 3)  # shots are looked for over the last 30 days
    row = vision.face(ts, "cam0", "Dana", 0.62)["id"]
    seen, code = why(shot=row)
    assert code == 200
    assert {k: seen[k] for k in ("name", "subject", "via", "basis", "similarity", "threshold")} == {
        "name": "Dana",
        "subject": "shot",
        "via": "face",
        "basis": None,
        "similarity": 0.62,
        "threshold": 0.45,
    }
    assert seen["mark"] == {"kind": "shot", "sighting": row, "ts": ts, "camera": "cam0"}
    registry.at(state).claim(registry.detection(row), "Eve", Basis.CORRECTION, "dashboard")
    fixed, _ = why(shot=row)
    assert (fixed["name"], fixed["via"], fixed["basis"], fixed["similarity"]) == ("Eve", "shot", "correction", None)


def test_a_face_the_owner_says_is_nobody_they_know_says_so(state, vision):
    row = vision.face(round(time.time() - 60.0, 3), "cam0", "Dana", 0.62)["id"]
    registry.at(state).claim(registry.detection(row), "Eve", Basis.CORRECTION, "dashboard")
    registry.at(state).retract(registry.detection(row), "dashboard")  # then: not Eve, nobody they know
    body, _ = why(shot=row)
    assert (body["name"], body["via"], body["similarity"]) == (None, "shot", None)
    assert isinstance(body["asserted_at"], float)
