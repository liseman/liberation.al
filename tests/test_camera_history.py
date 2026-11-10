from unittest.mock import ANY, create_autospec, patch

import pytest

from liberational import config as cfg
from liberational.assistant import voice
from liberational.assistant.voice_auth import UNHEARD, Conversation
from liberational.memory import ask as pack_ask
from liberational.memory import history_qa, question
from liberational.phone import outbound
from liberational.vision import camera_history as h
from voices import write_voices

NOW = 1000000


def row(vision, age, cam="cam1", caption="Same. Desk.", **extra):
    """A caption on the vision log `age` seconds before NOW."""
    return vision.caption(NOW - age, cam, caption, **extra)


def query(vision, q="what do you see on cam 1"):
    return h.query(q, vision, NOW)


@pytest.mark.parametrize(
    ("question", "cameras"),
    [
        ("what's on cam 1", ["cam1"]),
        ("what does camera one see", ["cam1"]),
        ("show me camera two and cam zero", ["cam0", "cam2"]),
    ],
)
def test_the_cameras_a_question_names_are_looked_up(question, cameras):
    assert h.cameras(question) == cameras


def test_now_is_the_newest_caption_of_the_last_ten_minutes_on_its_camera(vision):
    for args in ((4000, "cam1", "Empty desk."), (50, "cam1", "A person at the desk."), (5, "cam2", "A hallway.")):
        row(vision, *args)
    r = query(vision)
    assert r["cameras"] == [dict(cam="cam1", count=1, last=NOW - 50, stale=True, latest="A person at the desk.")]
    assert r["answer"] == f"cam1, observed at {h.stamp(NOW - 50)} (stale): A person at the desk."


def test_a_camera_quiet_for_ten_minutes_says_when_it_was_last_seen_never_an_old_view(vision):
    row(vision, 4000, "cam1", "Empty desk.")
    answer = query(vision)["answer"]
    assert answer == f"cam1: no observations in the last 10 minutes. Last observation {h.stamp(NOW - 4000)}, stale."
    assert "Empty desk" not in answer


def test_no_saved_history(vision):
    assert "No saved history" in query(vision)["answer"]


@pytest.fixture(autouse=True)
def conversation(tmp_path, monkeypatch):  # the voice tests below must not share the module's pending-auth state
    monkeypatch.setattr(voice, "CONVERSATION", Conversation(tmp_path / "verifier"))


@pytest.fixture
def history_worker():
    # pack_ask.ask runs the history worker in a subprocess: the boundary.
    with patch.object(pack_ask, "ask", autospec=True, return_value={"answer": "camera result"}) as ask:
        yield ask


OWNER = {"speaker_id": 1, "speaker_sim": 0.9}  # the owner's voice, by an owner tag (owner_state)


@pytest.fixture
def owner_state(tmp_path, monkeypatch):
    """Alice is the owner, and voice 1 is hers; voice 2 is a guest's."""
    state = tmp_path / "state"
    write_voices(state, {1: "Alice", 2: "Guest"})
    monkeypatch.setattr(voice, "STATE_DIR", state)
    with cfg.override({"owner": {"name": "Alice"}}):
        yield state


def test_the_owner_gets_a_spoken_camera_history_answer(history_worker, owner_state):
    speaker = create_autospec(voice.Speaker, instance=True)
    reply = voice.handle("Hey Bob what changed on Cam One in the last hour", speaker, OWNER)
    assert reply == "camera result"
    # With a speaker, voice hands Ask a callback to speak a short answer early.
    history_worker.assert_called_once_with("what changed on Cam One in the last hour", on_short=ANY)
    speaker.speak.assert_called_once_with("camera result")


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param({"speaker_id": 9, "speaker_sim": 1}, id="unnamed-voice"),
        pytest.param({"speaker_id": 2, "speaker_sim": 0.9}, id="named-guest"),
        pytest.param({"speaker_id": 1, "speaker_sim": 0.2}, id="owners-voice-heard-faintly"),
    ],
)
def test_anyone_else_gives_the_password_for_a_camera_history_question(history_worker, owner_state, entry):
    """A camera question about the past is answered from the transcript and memory, so only the verified owner
    gets it without the password."""
    reply = voice.handle("Hey Bob what changed on Cam One in the last hour", entry=entry)
    assert reply == "Please say your password for this request, or say cancel."
    history_worker.assert_not_called()


def test_a_line_that_says_nothing_about_who_spoke_gets_no_camera_history(history_worker, owner_state):
    reply = voice.handle("Hey Bob what changed on Cam One in the last hour", entry={})
    assert reply == UNHEARD
    history_worker.assert_not_called()


def test_a_guests_question_about_what_the_cameras_see_now_needs_no_password(history_worker, owner_state):
    with patch.object(voice, "camera_query", autospec=True, return_value={"answer": "A desk and a lamp."}) as live:
        reply = voice.handle("Hey Bob what do the cameras see", entry={"speaker_id": 2, "speaker_sim": 0.9})
    assert reply == "A desk and a lamp."
    live.assert_called_once_with("what do the cameras see")
    history_worker.assert_not_called()


def test_misheard_chem_is_asked_as_cam(history_worker, owner_state):
    voice.handle("Hey Bob, what changed on Chem Zero in the last hour?", entry=OWNER)
    history_worker.assert_called_once_with("what changed on cam Zero in the last hour?")


def test_nonowner_actions_still_need_the_password():
    with patch.object(outbound, "place", autospec=True, side_effect=AssertionError("call placed")):
        reply = voice.handle("Hey Bob call Sam", entry={"speaker_id": 9, "speaker_sim": 0.9})  # a voice nobody named
    assert reply is not None
    assert "password" in reply


def test_the_voice_service_never_answers_a_guests_camera_history_question_without_the_password(
    history_worker, owner_state, monkeypatch
):
    """The second lock: even a camera Grant for a guest that asks about the past gets no history answer."""
    guest_grant = voice.Grant("what changed on cam one in the last hour", "camera", False, False, False, False, False)
    monkeypatch.setattr(voice.CONVERSATION, "accept", lambda *a, **k: guest_grant)
    assert (
        voice.handle("Hey Bob what changed on cam one in the last hour", entry=OWNER)
        == "That needs the voice password."
    )
    history_worker.assert_not_called()


@pytest.mark.parametrize(
    "question",
    [
        "what did camzero show in the last 2 hours",
        "what changed on camzero over the past 2 hours",
        "how was camone in the last 3 hours",
        "describe camtwo over the previous 2 hours",
        "show me camszero for the last 2 hours",
    ],
)
def test_a_guest_gives_the_password_for_any_camera_question_about_the_past(history_worker, owner_state, question):
    """A camera name run together ("camzero") is a camera to the camera history, which
    then reads its time window, but not to the history classifier. Whatever reads the past needs the password."""
    with patch.object(voice, "camera_query", autospec=True, side_effect=AssertionError("caption history read")):
        reply = voice.handle("Hey Bob " + question, entry={"speaker_id": 2, "speaker_sim": 0.9})
    assert reply == "Please say your password for this request, or say cancel."
    history_worker.assert_not_called()


def test_the_owners_question_with_a_time_in_it_goes_to_ask_never_the_current_view(history_worker, owner_state):
    # Its reading isn't marked history but has a window (tests/data/readings.json): that is still the past.
    with patch.object(voice, "camera_query", autospec=True, side_effect=AssertionError("current view read")):
        reply = voice.handle("Hey Bob what changed on camzero over the past 2 hours", entry=OWNER)
    assert reply == "camera result"
    history_worker.assert_called_once_with("what changed on camzero over the past 2 hours")


@pytest.mark.parametrize(
    ("reading", "past"),
    [
        pytest.param(question.Reading(history=False, source="cameras"), False, id="what-they-show-now"),
        pytest.param(question.Reading(history=True, source="cameras"), True, id="a-history-question"),
        pytest.param(question.Reading(history=False, source="cameras", clock="3pm"), True, id="a-clock"),
        pytest.param(question.Reading(history=False, source="cameras", day="yesterday"), True, id="a-day"),
        pytest.param(
            question.Reading(history=False, source="cameras", span=2, span_unit="hours", span_kind="last"),
            True,
            id="a-window",
        ),
        pytest.param(
            question.Reading(history=False, source="cameras", other_time="last month"), True, id="a-refused-time"
        ),
    ],
)
def test_a_camera_question_is_about_the_past_when_its_reading_has_a_time(readings, reading, past):
    readings["what about the cameras"] = reading
    assert history_qa.asks_about_the_past("what about the cameras") is past


def test_the_owners_in_person_camera_question_about_the_past_is_asks_too(history_worker, readings):
    readings["what did the cameras see at 3pm"] = question.Reading(history=False, source="cameras", clock="3pm")
    with patch.object(voice, "camera_query", autospec=True, side_effect=AssertionError("current view read")):
        assert voice.owner_answer("what did the cameras see at 3pm", "session") == "camera result"
    history_worker.assert_called_once_with("what did the cameras see at 3pm", mode="local")


# ---- clock times: what a camera showed at 3pm, never what it shows now ----


def test_a_guest_gives_the_password_for_a_clock_time_camera_question(history_worker, owner_state):
    """The lock covers clock times too: a moment is a past question (voice_auth.asks_about_the_past)."""
    with patch.object(voice, "camera_query", autospec=True, side_effect=AssertionError("caption history read")):
        reply = voice.handle("Hey Bob what did cam 0 show at 3pm", entry={"speaker_id": 2, "speaker_sim": 0.9})
    assert reply == "Please say your password for this request, or say cancel."


def test_the_owners_clock_time_camera_question_reads_that_time_through_the_shared_history(history_worker, owner_state):
    # A clock time is a history question (shared Ask and voice time phrases), so it never reads the current view.
    with patch.object(voice, "camera_query", autospec=True) as cam:
        assert voice.handle("Hey Bob what did cam 0 show at 3pm", entry=OWNER) == "camera result"
    history_worker.assert_called_once_with("what did cam 0 show at 3pm")
    cam.assert_not_called()


@pytest.mark.parametrize("camera_id", ["entryway", "0", "cam00", "CAM0"])
def test_configured_camera_names_are_kept_as_they_are(tmp_path, monkeypatch, camera_id):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "devices.toml").write_text(
        'device=[{name="entryway",kind="rtsp"},{name="0",kind="camera"},'
        '{name="cam00",kind="camera"},{name="CAM0",kind="rtsp"}]'
    )
    monkeypatch.setattr(cfg, "OWNER_ROOT", tmp_path)
    assert h.cameras(f"what's on camera {camera_id}") == [camera_id]


def test_a_question_that_names_no_camera_reads_the_manifests_cameras(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    path = tmp_path / "config" / "devices.toml"
    path.write_text('device=[{name="entryway",kind="rtsp"},{name="rear-2",kind="camera"}]')
    monkeypatch.setattr(cfg, "OWNER_ROOT", tmp_path)
    assert h.cameras("what do the cameras see") == ["entryway", "rear-2"]
    path.write_text("device=[]")
    assert h.cameras("what do the cameras see") == []
