"""The voice assistant's authorization (assistant/voice_auth.py, through assistant/voice.py handle): who is asked for the
password and a confirmation, the lockout, expiry, replay, and the gateway's own check of the grant."""

import hashlib
import json
import re
import types
from unittest.mock import create_autospec

import pytest
import requests

from liberational import config as cfg
from liberational.assistant import voice as b
from liberational.assistant import voice_auth
from liberational.assistant.voice_auth import Conversation, Grant
from liberational.domain.identity import Basis
from liberational.memory import ask as pack_ask
from liberational.people import registry
from liberational.phone import outbound
from voices import write_voices

# The agent backend is opt-in (empty by default); these tests run with one configured.
AGENT = {
    "url": "http://127.0.0.1:18789/v1/chat/completions",
    "openclaw_config": "~/.openclaw/openclaw.json",
    "phone_agent": "openclaw/phone",
}
SALT = b"test salt only"
VERIFIER = {
    "version": 1,
    "salt": SALT.hex(),
    "iterations": 100000,
    "hash": hashlib.pbkdf2_hmac("sha256", b"dummy secret", SALT, 100000).hex(),
}
OWNER = {"speaker_id": 1, "speaker_sim": 0.9}
GUEST = {"speaker_id": 2, "speaker_sim": 0.9}  # recognized non-owner: still gated
STRANGER = {"speaker_id": 9, "speaker_sim": 0.9}  # a voice nobody has named


@pytest.fixture
def clock():
    return types.SimpleNamespace(now=1000.0)


@pytest.fixture(autouse=True)
def conv(tmp_path, monkeypatch, clock):
    path = tmp_path / "verifier.json"
    path.write_text(json.dumps(VERIFIER))
    path.chmod(0o600)
    c = Conversation(path, lambda: clock.now)
    monkeypatch.setattr(b, "CONVERSATION", c)
    monkeypatch.setattr(b, "STATE_DIR", tmp_path / "state")
    with cfg.override({"agent": AGENT, "owner": {"name": "Alice"}}):
        write_voices(tmp_path / "state", {1: "Alice", 2: "Guest"})  # the owner's voice is 1, by an owner tag
        yield c


@pytest.mark.parametrize("text", ["What time is it?", "Set the volume to twenty percent", "Bob, what time is it?"])
def test_strict_wake_ignores_ambient_owner_requests(conv, text):
    with cfg.override({"assistant": {"name": "Bob", "require_wake_phrase": True}}):
        assert conv.accept(text, {"speaker": "Alice"}, True, lambda _: None) is None


def test_strict_wake_accepts_full_phrase(conv):
    with cfg.override({"assistant": {"name": "Bob", "require_wake_phrase": True}}):
        result = conv.accept("Hey Bob, explain stars", {"speaker": "Alice"}, True, lambda _: None)
        assert isinstance(result, Grant)
        assert result.request == "explain stars"


@pytest.mark.parametrize(
    ("text", "addressed"),
    [
        pytest.param("How many times has that happened?", False, id="the-owner-asking-the-room"),
        pytest.param("Hey Bob, how many times has that happened?", True, id="hey-bob"),
        pytest.param("Bob, how many times has that happened?", True, id="by-name"),
    ],
)
def test_a_grant_says_whether_the_line_called_the_assistant(conv, text, addressed):
    with cfg.override({"assistant": {"name": "Bob"}}):
        result = conv.accept(text, {"speaker": "Alice"}, True, lambda _: None)
    assert isinstance(result, Grant) and result.route == "gateway"
    assert isinstance(result, Grant)
    assert (result.request.lower(), result.addressed) == ("how many times has that happened?", addressed)


def test_a_camera_question_says_whether_the_line_called_the_assistant(conv):
    with cfg.override({"assistant": {"name": "Bob"}}):
        asked = conv.accept("What is on cam zero?", {"speaker": "Alice"}, True, lambda _: True)
        called = conv.accept("Hey Bob, what is on cam zero?", {"speaker": "Alice"}, True, lambda _: True)
    assert isinstance(asked, Grant) and isinstance(called, Grant)
    assert [(g.route, g.addressed) for g in (asked, called)] == [("camera", False), ("camera", True)]


def say(text, entry=OWNER):
    return b.handle(text, entry=entry)


def stub(monkeypatch, name, **kw):
    fake = create_autospec(getattr(b, name), **kw)
    monkeypatch.setattr(b, name, fake)
    return fake


def check_grant(gateway, request, password, confirmation):
    gateway.assert_called_once()
    grant = gateway.call_args.kwargs["grant"]
    assert (grant.needs_password, grant.needs_confirmation) == (password, confirmation)
    assert (grant.authenticated, grant.confirmed) == (password, confirmation)
    assert grant.authorized_for(request)
    assert not grant.authorized_for(request + " and more")


@pytest.mark.parametrize(
    "request_",
    [
        "Explain how stars form",
        "What does delete mean?",
        "How do I install software?",
        "Set the volume to twenty percent",
        "Change my preferred response length to short",
        "Delete all files in Downloads",
        "Could you install a new browser?",
        "Read my medical records",
        "Remember my private preference for short replies",
        "Change my account password",
        "Reset the computer",
    ],
)
def test_the_owner_is_never_prompted(conv, monkeypatch, request_):
    gateway = stub(monkeypatch, "owner_answer", return_value="done")
    assert say(request_) == "done"
    assert conv.pending is None
    check_grant(gateway, request_, password=False, confirmation=False)


def test_a_guest_says_the_password_even_to_ask_a_question(conv, monkeypatch):
    gateway = stub(monkeypatch, "owner_answer", return_value="done")
    assert "password" in say("Hey Bob, Explain how stars form", GUEST)
    gateway.assert_not_called()
    assert say("dummy secret", GUEST) == "done"
    assert conv.pending is None
    check_grant(gateway, "Explain how stars form", password=True, confirmation=False)


@pytest.mark.parametrize("request_", ["Reset the computer", "Change my account password"])
def test_a_guest_also_confirms_a_major_request(conv, monkeypatch, request_):
    gateway = stub(monkeypatch, "owner_answer", return_value="done")
    assert "password" in say("Hey Bob, " + request_, GUEST)
    assert "Confirm" in say("dummy secret", GUEST)
    say("delete something else", GUEST)  # neither a confirmation nor a new request
    gateway.assert_not_called()
    assert say("yes confirm", GUEST) == "done"
    assert conv.pending is None
    check_grant(gateway, request_, password=True, confirmation=True)


def test_no_owner_configured_means_nobody_is_owner(monkeypatch):
    stub(monkeypatch, "owner_answer", side_effect=AssertionError)
    with cfg.override({"owner": {"name": ""}}):
        assert not any(b.owner_entry({"speaker": s, "speaker_sim": 0.9}) for s in ("alice", "", "unknown"))
        assert say("Reset the computer") is None  # unaddressed: not for us
        assert "password" in say("Hey Bob, reset the computer")


def test_wake_words_follow_config(monkeypatch):
    answer = stub(monkeypatch, "answer", return_value="Noon")
    with cfg.override({"assistant": {"name": "Pax", "wake_words": ["ok pax"]}}):
        assert say("Ok, Pax, what time is it?") == "Noon"
        assert answer.call_args.args[0] == "what time is it?"
        assert say("Pax what time is it?") == "Noon"
        assert answer.call_args.args[0] == "what time is it?"


def test_wake_words_tolerate_odd_config():
    with cfg.override({"assistant": {"name": "Pax", "wake_words": ["hey, pax", "", "hey"]}}):
        assert cfg.wake_words() == ["hey, pax", "hey"]  # the empty entry would match everything
        assert re.match(voice_auth.wake_regex(), "hey pax what time is it")
        assert not re.match(voice_auth.wake_regex(), "the room looks nice")
        # longest phrase first, so "hey pax" isn't cut down to "hey"
        m = re.match(voice_auth.address_regex(), "hey pax what")
        assert m is not None
        assert m.group(0) == "hey pax"


def test_background_and_no_wake(monkeypatch):
    assert say("The room looks nice") is None
    answer = stub(monkeypatch, "answer", return_value="Noon")
    assert say("What time is it?") == "Noon"
    answer.assert_called_once()
    # a line a person tagged as the owner counts as the owner; a missing score alone does not
    assert say("What day is it?", {"speaker_id": 1, "speaker_sim": None, "tagged": True}) == "Noon"
    assert say("What day is it?", {"speaker_id": 1, "speaker_sim": None}) is None
    assert say("What day is it?", {"speaker_id": 1, "speaker_sim": 0.2}) is None


def test_unknown_public_and_no_promotion(monkeypatch):
    answer = stub(monkeypatch, "answer", return_value="Noon")
    assert say("What time is it?", STRANGER) is None  # strangers must say the wake phrase
    assert say("What time is it?", GUEST) is None
    assert "password" in say("Hey Bob, what time is it?", STRANGER)
    answer.assert_not_called()
    assert say("dummy secret", STRANGER) == "Noon"
    assert "password" in say("Hey Bob, what time is it?", STRANGER)


def test_call_no_early_dispatch(monkeypatch):
    call = create_autospec(outbound.place, return_value=(True, "calling"))
    monkeypatch.setattr(outbound, "place", call)
    assert "password" in say("Hey Bob, call Dana", GUEST)
    call.assert_not_called()
    assert "Confirm" in say("dummy secret", GUEST)
    call.assert_not_called()
    assert say("yes confirm", GUEST) == "calling"
    call.assert_called_once()
    assert say("Call Dana") == "calling"
    assert call.call_count == 2


def test_wrong_password_cooldown(conv, clock, monkeypatch):
    stub(monkeypatch, "owner_answer", side_effect=AssertionError)
    stub(monkeypatch, "answer", side_effect=AssertionError)
    say("Hey Bob, reset my private computer", GUEST)
    replies = [say("wrong", GUEST) for _ in range(3)]
    assert "Too many" in replies[-1]
    assert conv.pending is None
    assert "wait" in say("Hey Bob, reset my private computer", GUEST)
    clock.now += 61
    assert "password" in say("Hey Bob, reset my private computer", GUEST)


def test_expiry_cancel_and_mismatch(conv, clock):
    say("Hey Bob, reset my private computer", GUEST)
    assert say("dummy secret", {"speaker": "other"}) is None
    assert conv.pending is not None
    clock.now += 46
    assert "expired" in say("dummy secret", GUEST)
    say("Hey Bob, reset my private computer", GUEST)
    assert say("cancel", GUEST) == "Cancelled."
    assert say("yes confirm", GUEST) is None


def test_confirmation_expiry(clock, monkeypatch):
    say("Hey Bob, reset my private computer", GUEST)
    say("dummy secret", GUEST)
    clock.now += 46
    stub(monkeypatch, "owner_answer", side_effect=AssertionError)
    assert "expired" in say("yes", GUEST)


def test_self_speech_and_replay(conv, monkeypatch):
    conv.spoken_reply("What time is it?", 990, 995)
    assert say("What time is it?") is None
    assert say("Reset my private computer", {**OWNER, "ts_start": 993}) is None
    assert say("Reset my private computer", {**OWNER, "ts_start": 900}) is None
    camera = stub(monkeypatch, "camera_query", return_value={"answer": "desk"})
    say("What changed on cam zero?", {**OWNER, "ts_start": 999})
    say("What changed on cam zero?", {**OWNER, "ts_start": 999})
    camera.assert_called_once()


def test_no_password_in_notes_or_logs(tmp_path, monkeypatch, caplog):
    caplog.set_level("DEBUG")
    gateway = stub(monkeypatch, "owner_answer", return_value="done")
    say("Hey Bob, reset my private computer", GUEST)
    say("dummy secret", GUEST)
    say("yes confirm", GUEST)
    gateway.assert_called_once()
    assert "dummy secret" not in b.owner_notes()
    assert caplog.messages, "the turn logged nothing, so this checks nothing"
    assert "dummy secret" not in caplog.text + repr(gateway.call_args_list)


def test_the_owners_notes_leave_out_what_they_asked_to_forget(tmp_path, monkeypatch):
    memory = tmp_path / "memory.jsonl"
    memory.write_text(
        "".join(
            json.dumps(e) + "\n"
            for e in (
                {"op": "remember", "text": "the spare key is under the mat"},
                {"op": "forget", "text": "everything"},
                {"op": "remember", "text": "Dana likes tea"},
            )
        )
    )
    monkeypatch.setattr(b, "MEMORY", memory)
    notes = b.owner_notes()
    assert "Dana likes tea" in notes
    assert "spare key" not in notes


@pytest.mark.parametrize(
    "grant",
    [
        None,
        Grant("other", "gateway", True, True),
        Grant("Restart", "gateway", True, False),
        Grant("Restart", "gateway", False, False, needs_password=False, needs_confirmation=False),
    ],
    ids=["no-grant", "another-request", "unconfirmed", "a-guest-skipped-the-gate"],
)
def test_the_gateway_refuses_a_grant_that_does_not_cover_the_request(monkeypatch, grant):
    monkeypatch.setattr(b.requests, "post", create_autospec(requests.post, side_effect=AssertionError))
    with pytest.raises(PermissionError) as rejected:
        b.owner_answer("Restart", "test", grant)
    assert isinstance(rejected.value, b.VoiceAuthorizationError)


@pytest.fixture
def post(monkeypatch):
    response = create_autospec(requests.Response, instance=True)
    response.json.return_value = {"choices": [{"message": {"content": "Done."}}]}
    monkeypatch.setattr(b, "_gateway_token", lambda: "dummy")
    monkeypatch.setattr(b, "owner_notes", lambda: "")
    fake = create_autospec(requests.post, return_value=response)
    monkeypatch.setattr(b.requests, "post", fake)
    return fake


def owner_grant(request):
    return Grant(request, "gateway", False, False, needs_password=False, needs_confirmation=False, owner=True)


@pytest.mark.parametrize(
    "grant",
    [
        owner_grant("Explain gravity"),
        owner_grant("Set volume to twenty"),
        owner_grant("Restart service"),
        owner_grant("Read medical records"),
        Grant("Explain gravity", "gateway", authenticated=True, confirmed=False, needs_confirmation=False),
        Grant("Restart service", "gateway", authenticated=True, confirmed=True),
    ],
    ids=["owner-explain", "owner-volume", "owner-restart", "owner-medical", "guest-password", "guest-confirmed"],
)
def test_the_gateway_rechecks_a_valid_grant_and_scopes_the_agent_to_its_request(post, grant):
    assert b.owner_answer(grant.request, "test", grant) == "Done."
    post.assert_called_once()
    sent = post.call_args.kwargs["json"]["messages"][0]["content"]
    assert f'EXACT request "{grant.request}"' in sent
    assert "Native tool approvals remain mandatory" in sent  # the agent may not skip its own approval prompts


def test_mixed_camera_mutation_gated(conv, monkeypatch):
    stub(monkeypatch, "camera_query", side_effect=AssertionError)
    stub(monkeypatch, "owner_answer", return_value="done")
    assert "password" in say("Hey Bob, what is on camera one and delete the recordings", GUEST)
    conv.pending = None
    assert say("What is on camera one and delete the recordings") == "done"


def test_bad_verifier_fails_closed(conv):
    conv.verifier.chmod(0o644)
    say("Hey Bob, reset my private computer", GUEST)
    assert "not accepted" in say("dummy secret", GUEST)


@pytest.mark.parametrize(
    ("prompt_end", "now", "heard"),
    [(1040, 1090, {"ts_start": 1084, "ts_end": 1086}), (1005, 1007, {"ts_start": 1005.2})],
    ids=["delayed-asr", "no-three-second-blackout"],
)
def test_a_password_said_after_the_prompt_is_heard(conv, clock, monkeypatch, prompt_end, now, heard):
    say("Hey Bob, read my medical records", GUEST)
    conv.spoken_reply("Please say your password", 1000, prompt_end)
    clock.now = now
    stub(monkeypatch, "owner_answer", return_value="done")
    assert say("dummy secret", {**GUEST, **heard}) == "done"


def test_asking_friends_is_gone_but_still_behind_authorization(monkeypatch):
    module = create_autospec(pack_ask)
    monkeypatch.setattr(b, "optional_module", {"liberational.memory.ask": module}.get)
    assert say("Ask your friends why stars shine") == b.NO_OUTSIDE_MODELS
    assert "password" in say("Hey Bob, ask your friends why stars shine", STRANGER)
    module.ask.assert_not_called()


@pytest.mark.parametrize("named_by", ["introduction", "merge"])
def test_a_voice_that_only_names_the_owner_gets_no_owner_path(tmp_path, monkeypatch, named_by):
    """A second voice whose person is the owner, but not by the owner's own tag, asks like a guest, keeps its own
    pending request, and can't answer it on the owner's behalf. The owner can still cancel it."""
    state = tmp_path / "state"
    write_voices(state, {1: "Alice", 2: "Guest", 3: None})
    reg = registry.at(state)
    if named_by == "introduction":
        reg.claim(registry.voiceprint(3), "Alice", Basis.INTRODUCTION, "identity")  # "I'm Alice", overheard
    else:
        reg.claim(registry.voiceprint(3), "Dana", Basis.OWNER_TAG, "dashboard")
        reg.merge("Dana", "Alice", "dashboard")  # Dana's person merged into the owner's
    gateway = stub(monkeypatch, "owner_answer", side_effect=AssertionError)
    second = {"speaker_id": 3, "speaker_sim": 0.95}
    assert "password" in say("Hey Bob, reset the computer", second).lower()
    assert say("yes confirm", second) == "Password not accepted. Try again or say cancel."
    assert say("cancel", OWNER) == "Cancelled."  # the verified owner can end another voice's request
    gateway.assert_not_called()


def test_the_owner_still_passes_voice_verification_end_to_end_after_175(conv, monkeypatch):
    """Through voice.handle and the real owner check (owner_voice by the owner tag, canonical_entry): the owner is
    never asked, breaks in on a guest's waiting request, and another voice they named theirs is them too."""
    gateway = stub(monkeypatch, "owner_answer", return_value="done")
    assert "password" in say("Hey Bob, read my medical records", GUEST)  # a guest's request waits
    assert say("Hey Bob, change my account password", GUEST | {"speaker_id": 3}) == voice_auth.WAITING  # another
    assert say("Hey Bob, reset the computer") == "done"  # the owner replaces it, and isn't asked
    check_grant(gateway, "reset the computer", password=False, confirmation=False)
    write_voices(b.STATE_DIR, {7: "Alice"})  # the owner named voice 7 theirs too ("That was me")
    assert say("Hey Bob, delete all files in Downloads", OWNER | {"speaker_id": 7}) == "done"
    assert conv.pending is None


@pytest.mark.parametrize(
    ("failure", "error", "spoken"),
    [
        (RuntimeError("gateway token unavailable"), "gateway_error", "could not be completed"),
        (PermissionError("reply file is not writable"), "gateway_error", "could not be completed"),
        (
            b.VoiceAuthorizationError("Local authorization and confirmation required"),
            "not_authorized",
            "this request wasn't authorized",
        ),
    ],
    ids=["unexpected-failure", "dispatch-permission-failure", "not-authorized"],
)
def test_a_failed_request_logs_what_failed_and_says_so_honestly(conv, monkeypatch, caplog, failure, error, spoken):
    stub(monkeypatch, "owner_answer", side_effect=failure)
    caplog.set_level("INFO")
    assert spoken in say("Hey Bob, explain gravity").lower()
    assert any(str(failure) in line for line in caplog.messages)
    assert any("error=" + error in line for line in caplog.messages)
