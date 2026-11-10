import datetime as dt
from unittest.mock import ANY, create_autospec, patch

import pytest

from liberational import config as cfg
from liberational.assistant import voice
from liberational.assistant.voice_auth import UNHEARD, Conversation
from liberational.memory import ask as pack_ask
from liberational.phone import outbound
from liberational.vision import camera_history as h
from voices import write_voices

NOW = 1000000


def row(vision, age, cam="cam1", caption="Same. Desk.", **extra):
    """A caption on the vision log `age` seconds before NOW."""
    return vision.caption(NOW - age, cam, caption, **extra)


def query(vision, q="what changed on cam 1 in the last hour"):
    return h.query(q, vision, NOW)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("cam 1 last hour", (["cam1"], 3600)),
        ("camera one past one hour", (["cam1"], 3600)),
        ("cam1 previous an hour", (["cam1"], 3600)),
        ("camera two last five minutes", (["cam2"], 300)),
        ("cam zero now", (["cam0"], None)),
    ],
)
def test_parse_reads_camera_and_window(question, expected):
    assert h.parse(question) == expected


def test_parse_rejects_a_window_with_no_length():
    with pytest.raises(ValueError, match="relative window"):
        h.parse("camera 1 yesterday")


def test_partial_stale_window_counts_only_its_camera(vision):
    for args in ((4000,), (50, "cam1", "Changed. Chair moved."), (5, "cam2")):
        row(vision, *args)
    c = query(vision)["cameras"][0]
    assert c["count"] == 1
    assert c["partial"]
    assert c["stale"]
    assert c["change_count"] == 1


def test_no_saved_history(vision):
    assert "No saved history" in query(vision)["answer"]


def test_full_samples_then_a_gap_turns_partial(vision):
    for x in range(0, 3601, 10):
        if not 100 <= x <= 200:  # a gap of more than 30 s in the window
            row(vision, x)
    assert query(vision)["cameras"][0]["partial"]
    for x in range(110, 200, 10):
        row(vision, x)
    assert not query(vision)["cameras"][0]["partial"]


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


def test_the_owner_still_gets_a_camera_history_window(history_worker, owner_state):
    with patch.object(voice, "camera_query", autospec=True, return_value={"answer": "Two changes."}) as camera:
        assert voice.handle("Hey Bob what changed on camzero in the last 2 hours", entry=OWNER) == "Two changes."
    camera.assert_called_once_with("what changed on camzero in the last 2 hours")


# ---- clock times: what a camera showed at 3pm, never what it shows now ----

EVENING = dt.datetime(2026, 10, 24, 18, 0).timestamp()  # local time, as people ask


def local(day, hour, minute=0):
    return dt.datetime(2026, 10, day, hour, minute).timestamp()


@pytest.mark.parametrize(
    ("question", "moment"),
    [
        pytest.param("what did cam 0 show at 3pm", local(24, 15), id="pm-today"),
        pytest.param("what did cam 0 show at 3:30 p.m.", local(24, 15, 30), id="minutes-and-dots"),
        pytest.param("what did cam 0 show at 15:00", local(24, 15), id="24-hour"),
        pytest.param("what did cam 0 show at noon", local(24, 12), id="noon"),
        pytest.param("what did cam 0 show around 9 am", local(24, 9), id="around-am"),
        pytest.param("what did cam 0 show at 8pm", local(23, 20), id="not-yet-today-is-yesterday"),
        pytest.param("what did cam 0 show yesterday at 3pm", local(23, 15), id="yesterday"),
        pytest.param("what did cam 0 show at midnight", local(24, 0), id="midnight"),
    ],
)
def test_parse_reads_a_clock_time_as_a_moment(question, moment):
    assert h.parse(question, EVENING) == (["cam0"], h.At(moment))


@pytest.mark.parametrize(
    "question",
    [
        pytest.param("what did cam 0 show today at 8pm", id="later-today"),
        pytest.param("what did cam 0 show at 13pm", id="no-such-time"),
        pytest.param("what did cam 0 show at 25:00", id="no-such-24-hour-time"),
        pytest.param("what did cam 0 show this morning", id="part-of-day"),
        # A clock time with another time word must not quietly mean today or yesterday.
        pytest.param("what did cam 0 show two days ago at 3pm", id="days-ago"),
        pytest.param("what did cam 0 show at 3pm last week", id="last-week"),
        pytest.param("what did cam 0 show at 3pm on 2026-10-20", id="a-date"),
        pytest.param("what did cam 0 show on monday at 3pm", id="a-weekday"),
        pytest.param("what did cam 0 show at 3pm in the evening", id="with-a-part-of-day"),
    ],
)
def test_parse_refuses_a_past_time_it_cant_read(question):
    with pytest.raises(ValueError):
        h.parse(question, EVENING)


def test_a_number_alone_is_not_a_clock_time():
    assert h.parse("what do you see at cam 3", EVENING) == (["cam3"], None)


def test_a_clock_time_reads_the_caption_nearest_it_never_the_current_view(vision):
    three = local(24, 15)
    vision.caption(three - 400, "cam0", "Same. Empty desk.")
    vision.caption(three - 90, "cam0", "Changed. A person at the desk.")
    vision.caption(EVENING - 5, "cam0", "Same. Lights off.")
    answer = h.query("what did cam 0 show at 3pm", vision, EVENING)["answer"]
    assert answer == f"cam0 at {h.stamp(three - 90)}: Changed. A person at the desk."


def test_a_clock_time_on_an_unchanged_view_says_what_it_repeats_and_since_when(vision):
    three = local(24, 15)
    vision.caption(three - 200, "cam0", "A person at the desk.")
    vision.caption(three - 10, "cam0", "A person at the desk.", unchanged=True, since=three - 200, gated=True)
    answer = h.query("what did cam 0 show at 3pm", vision, EVENING)["answer"]
    assert answer == f"cam0 at {h.stamp(three - 10)}: unchanged since {h.stamp(three - 200)}: A person at the desk."


def test_a_clock_time_on_a_warm_up_says_so_and_never_shows_an_older_caption(vision):
    three = local(24, 15)
    vision.caption(three - 200, "cam0", "A man at a desk.")
    vision.caption(three - 5, "cam0", None)  # the VLM was still warming up
    answer = h.query("what did cam 0 show at 3pm", vision, EVENING)["answer"]
    assert "starting up" in answer and "A man at a desk" not in answer


def test_a_clock_time_with_nothing_near_it_says_so(vision):
    vision.caption(EVENING - 5, "cam0", "Same. Lights off.")
    answer = h.query("what did cam 0 show at 3pm", vision, EVENING)["answer"]
    assert answer == f"cam0: no observations within 5 minutes of {h.stamp(local(24, 15))}."


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
def test_history_parse_preserves_configured_camera_names(tmp_path, monkeypatch, camera_id):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "devices.toml").write_text(
        'device=[{name="entryway",kind="rtsp"},{name="0",kind="camera"},'
        '{name="cam00",kind="camera"},{name="CAM0",kind="rtsp"}]'
    )
    monkeypatch.setattr(cfg, "ROOT", tmp_path)
    assert h.parse(f"camera {camera_id} last hour") == ([camera_id], 3600)


def test_history_default_cameras_follow_the_authoritative_native_manifest(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    path = tmp_path / "config" / "devices.toml"
    path.write_text('device=[{name="entryway",kind="rtsp"},{name="rear-2",kind="camera"}]')
    monkeypatch.setattr(cfg, "ROOT", tmp_path)
    assert h.parse("what did cameras see in the last hour") == (["entryway", "rear-2"], 3600)
    path.write_text("device=[]")
    assert h.parse("what did cameras see in the last hour") == ([], 3600)
