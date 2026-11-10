import asyncio
import base64
import hashlib
import hmac
import json
import os
import stat
import threading
import time
from types import SimpleNamespace
from typing import cast
from unittest.mock import create_autospec

import numpy as np
import pytest
import requests
from fastapi.testclient import TestClient

from liberational import config as cfg
from liberational.assistant import voice_auth
from liberational.people import profiles
from liberational.phone import bridge as phone_bridge
from liberational.phone import call as phone_cli
from liberational.phone import outbound as outbound_call

ALICE_AND_AVA = {"owner": {"name": "Alice"}, "assistant": {"name": "Ava"}}


@pytest.fixture(autouse=True)
def env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("BRIDGE_PUBLIC_HOST", raising=False)
    path = tmp_path / ".env.phone"
    with cfg.override({"phone": {"env_file": str(path)}}):
        yield path


def test_owner_number_needs_owner_phone():
    assert not outbound_call.is_owner_number("+15555550123")
    with cfg.override({"owner": {"phone": "+15555550123"}}):
        assert not outbound_call.is_owner_number("+15555550123")  # a phone alone isn't an owner
    with cfg.override({"owner": {"phone": "+15555550123", "name": "Alice"}}):
        assert outbound_call.is_owner_number("+15555550123")
        assert not outbound_call.is_owner_number("+15555550124")
        assert not outbound_call.is_owner_number("")


def test_owner_phone_label_and_the_profiles_that_fold_it():
    assert outbound_call.owner_phone_label() == "Owner (phone)"
    assert profiles.phone_labels() == {}
    with cfg.override({"owner": {"name": "Alice"}}):
        assert profiles.phone_labels() == {"Alice (phone)": "alice"}


def test_caller_label_from_number():
    call = phone_bridge.Call(cast(phone_bridge.WebSocket, None))  # the label logic never touches the socket
    call.from_number = "+15555550123"
    assert call.caller_default == "Caller 0123"
    with cfg.override({"owner": {"name": "Alice", "phone": "+15555550123"}}):
        assert call.caller_default == "Caller 0123"  # the owner's number alone, unattested, isn't the owner
        call.verified_owner = True
        assert call.caller_default == "Alice (phone)"


@pytest.mark.parametrize("consented", [False, True])
def test_only_preconsented_openers_skip_the_recording_notice(monkeypatch, consented):
    seen = []

    def answer(history, system):
        seen.append(history[-1]["content"])
        return "What time works for you?"

    monkeypatch.setattr(phone_bridge, "ask_llm", answer)
    with cfg.override(ALICE_AND_AVA):
        spoken = phone_bridge.mission_opener("Arrange dinner", prior_consent=consented)
    assert ("This call is recorded" in spoken) is not consented
    assert spoken.startswith("It's Ava. " if consented else "Hi, this is Ava, calling for Alice. ")
    assert ("This call is recorded" in seen[0]) is not consented
    assert "Arrange dinner" not in spoken


@pytest.mark.parametrize(
    "status,provider_state,ok", [(200, "completed", True), (200, "in-progress", False), (403, "completed", False)]
)
def test_hangup_requires_provider_confirmation(env_file, monkeypatch, status, provider_state, ok):
    env_file.write_text("TWILIO_ACCOUNT_SID=ACfake\nTWILIO_AUTH_TOKEN=fake\n")
    post = create_autospec(
        outbound_call.requests.post,
        return_value=SimpleNamespace(status_code=status, json=lambda: {"status": provider_state}),
    )
    monkeypatch.setattr(outbound_call.requests, "post", post)
    sid = "CA" + "1" * 32
    assert outbound_call.hangup(sid)[0] is ok
    post.assert_called_once_with(
        f"https://api.twilio.com/2010-04-01/Accounts/ACfake/Calls/{sid}.json",
        auth=("ACfake", "fake"),
        data={"Status": "completed"},
        timeout=10,
    )


@pytest.mark.parametrize("sid", ["", "CA1", "../Calls", "CA" + "x" * 32])
def test_hangup_invalid_sid_cannot_reach_provider(monkeypatch, sid):
    post = create_autospec(outbound_call.requests.post)
    monkeypatch.setattr(outbound_call.requests, "post", post)
    assert not outbound_call.hangup(sid)[0]
    post.assert_not_called()


def test_hangup_cli_ends_one_call_without_placing_another(env_file, monkeypatch, capsys):
    import tomllib

    scripts = tomllib.loads((cfg.ROOT / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts["pack-phone-call"] == "liberational.phone.call:main"
    env_file.write_text("TWILIO_ACCOUNT_SID=ACfake\nTWILIO_AUTH_TOKEN=fake\n")
    post = create_autospec(
        outbound_call.requests.post,
        return_value=SimpleNamespace(status_code=200, json=lambda: {"status": "completed"}),
    )
    monkeypatch.setattr(outbound_call.requests, "post", post)
    sid = "CA" + "2" * 32
    monkeypatch.setattr("sys.argv", ["pack-phone-call", "--hangup", sid])
    phone_cli.main()
    assert capsys.readouterr().out == "Call ended.\n"
    assert post.call_args.args[0].endswith(f"/Calls/{sid}.json")
    assert post.call_args.kwargs["data"] == {"Status": "completed"}


def test_farewell_failure_does_not_prevent_hangup(talk, monkeypatch):
    hung = []

    async def broken_speech(text):
        raise RuntimeError("synthetic synthesis failure")

    monkeypatch.setattr(talk.call, "speak", broken_speech)
    monkeypatch.setattr(phone_bridge, "hangup_call", lambda sid: hung.append(sid) or "completed")
    asyncio.run(talk.call.say_and_hang_up("Okay, bye!"))
    assert hung == ["CA1"]


def test_public_url(monkeypatch):
    with pytest.raises(RuntimeError):
        outbound_call.public_url("/voice")
    with cfg.override({"phone": {"public_host": "pack.example.ts.net:8443"}}):
        assert outbound_call.public_url("/voice") == "https://pack.example.ts.net:8443/bridge/voice"
        assert phone_bridge.public_voice_urls() == (
            "https://pack.example.ts.net:8443/bridge/voice",
            "https://pack.example.ts.net/bridge/voice",
        )
    monkeypatch.setenv("BRIDGE_PUBLIC_HOST", "other.example.net")
    assert outbound_call.public_url() == "https://other.example.net/bridge"


def test_twilio_settings_come_from_the_env_file_unless_configured(env_file):
    assert outbound_call.twilio_number() == ""  # no env file, nothing configured
    env_file.write_text(
        "# Twilio\n\nTWILIO_ACCOUNT_SID=\"AC123\"\nTWILIO_AUTH_TOKEN = 'tok'\nTWILIO_FROM_NUMBER=+15555550100\n"
    )
    assert outbound_call.load_env() == {
        "TWILIO_ACCOUNT_SID": "AC123",
        "TWILIO_AUTH_TOKEN": "tok",
        "TWILIO_FROM_NUMBER": "+15555550100",
    }
    assert outbound_call.twilio_number() == "+15555550100"
    with cfg.override({"phone": {"twilio_number": "+15555550111"}}):
        assert outbound_call.twilio_number() == "+15555550111"


def test_parse_uses_assistant_name():
    assert outbound_call.parse("bob call Dana and tell her I'm running late") == ("dana", "I'm running late")
    with cfg.override({"assistant": {"name": "Alfred"}}):
        assert outbound_call.parse("hey alfred call dana") == ("dana", "")
        assert outbound_call.parse("bob call dana") is None


def test_place_refuses_without_public_host(tmp_path, monkeypatch):
    missions = tmp_path / "missions.json"
    monkeypatch.setattr(outbound_call, "contacts", lambda: {"dana": "+15555550199"})
    monkeypatch.setattr(outbound_call, "MISSIONS", missions)

    def dial(*args, **kwargs):
        raise AssertionError("must not dial")

    monkeypatch.setattr(outbound_call.requests, "post", dial)
    ok, detail = outbound_call.place("dana", "hi")
    assert not ok
    assert "public_host" in detail
    assert not missions.exists()


PROMPTS = {
    "greeting": phone_bridge.greeting,
    "greeting_other": phone_bridge.greeting_other,
    "owner_only": phone_bridge.owner_only,
    "disclosure": phone_bridge.mission_disclosure,
    "opener": lambda: phone_bridge.mission_opener("tell him a joke"),
    "voicemail": lambda: phone_bridge.voicemail_message(None),
    "system": phone_bridge.system_prompt,
}


@pytest.mark.parametrize("prompt", PROMPTS.values(), ids=PROMPTS.keys())
def test_prompts_name_the_owner_and_assistant_or_stay_neutral(prompt, monkeypatch):
    monkeypatch.setattr(phone_bridge, "ask_llm", lambda *args: "What time works for you?")
    neutral = prompt()
    assert " 's" not in neutral and " ," not in neutral and "None" not in neutral  # no blank where a name goes
    with cfg.override(ALICE_AND_AVA):
        named = prompt()
    assert ("Alice" in named or "Ava" in named) and "Bob" not in named


def test_mission_opening_keeps_instructions_private_and_identity_truthful(monkeypatch):
    seen = []

    def answer(history, system):
        seen.append((history, system))
        return "What time works for you?"

    monkeypatch.setattr(phone_bridge, "ask_llm", answer)
    with cfg.override(ALICE_AND_AVA):
        opener = phone_bridge.mission_opener("Arrange a meeting; ask which time works.")
    assert "Ava, calling for Alice" in opener
    assert "This call is recorded" in opener
    assert "AI" not in opener and "Arrange a meeting" not in opener
    assert "What time works for you?" in opener
    assert "Arrange a meeting" in seen[0][1]


def test_mission_opening_tells_the_model_the_disclosure_was_already_said(monkeypatch):
    # Without being told the disclosure was already spoken, the model greeted and introduced itself again
    seen = []

    def answer(history, system):
        seen.append(history)
        return "You wanted a haiku. What should it be about?"

    monkeypatch.setattr(phone_bridge, "ask_llm", answer)
    with cfg.override(ALICE_AND_AVA):
        opener = phone_bridge.mission_opener("Compose a haiku on a topic they choose.")
    heard = "Hi, this is Ava, calling for Alice. This call is recorded."
    assert opener == heard + " You wanted a haiku. What should it be about?"
    assert heard in seen[0][-1]["content"]


@pytest.mark.parametrize("number", ["", "4155550123", "+44 (0)20 1234 5678", "+123", "+1abc5550123"])
def test_explicit_invalid_target_never_dials(number, monkeypatch):
    monkeypatch.setattr(outbound_call.requests, "post", lambda *a, **kw: pytest.fail("dialed"))
    assert not outbound_call.place("", "Arrange a meeting", number=number)[0]


def test_explicit_target_normalized_and_contact_change_refused(tmp_path, monkeypatch, env_file):
    env_file.write_text("TWILIO_ACCOUNT_SID=fake\nTWILIO_AUTH_TOKEN=fake\nTWILIO_FROM_NUMBER=+15555550100\n")
    monkeypatch.setattr(outbound_call, "MISSIONS", tmp_path / "missions.json")
    monkeypatch.setattr(outbound_call, "contacts", lambda: {"dana": "+15555550199"})
    sent = []

    def dial(*args, **kwargs):
        sent.append(kwargs["data"])
        return SimpleNamespace(status_code=201)

    monkeypatch.setattr(outbound_call.requests, "post", dial)
    with cfg.override({"phone": {"public_host": "example.invalid"}}):
        assert outbound_call.place("", "Arrange a meeting", number="+1 (555) 555-0123")[0]
        assert not outbound_call.place("Dana", "Arrange a meeting", number="+15555550123")[0]
    assert len(sent) == 1
    assert sent[0]["To"] == "+15555550123"
    assert outbound_call.take_mission("+15555550123") == "Arrange a meeting"


def test_a_placed_call_keeps_whom_the_owner_asked_for_and_their_words(tmp_path, monkeypatch, env_file):
    """For the call's record: the mission keeps the owner's words and the contact's name beside what to say."""
    env_file.write_text("TWILIO_ACCOUNT_SID=fake\nTWILIO_AUTH_TOKEN=fake\nTWILIO_FROM_NUMBER=+15555550100\n")
    monkeypatch.setattr(outbound_call, "MISSIONS", tmp_path / "missions.json")
    monkeypatch.setattr(outbound_call, "contacts", lambda: {"dana": "+15555550199"})
    monkeypatch.setattr(outbound_call.requests, "post", lambda *a, **k: SimpleNamespace(status_code=201))
    with cfg.override({"phone": {"public_host": "example.invalid"}}):
        assert outbound_call.place("Dana", "Ask whether she's free Friday")[0]
    mission = outbound_call.take_mission_record("+15555550199")
    assert mission is not None
    assert (mission["name"], mission["message"]) == ("Dana", "Ask whether she's free Friday")
    assert outbound_call.take_mission_record("+15555550199") is None  # consumed once


def twilio_created(to: str = "+15555550199") -> requests.Response:
    """What Twilio's Calls API answers a placed call with: 201 and the new call resource."""
    r = requests.Response()
    r.status_code = 201
    r.headers["Content-Type"] = "application/json"
    r._content = json.dumps(
        {"sid": "CA" + "0" * 32, "to": to, "from": "+15555550100", "status": "queued", "direction": "outbound-api"}
    ).encode()
    return r


@pytest.fixture
def dialable(tmp_path, monkeypatch, env_file):
    """Twilio configured, two contacts, and the missions file in tmp_path; dialing succeeds unless a test says not."""
    env_file.write_text("TWILIO_ACCOUNT_SID=fake\nTWILIO_AUTH_TOKEN=fake\nTWILIO_FROM_NUMBER=+15555550100\n")
    monkeypatch.setattr(outbound_call, "MISSIONS", tmp_path / "missions.json")
    monkeypatch.setattr(outbound_call, "contacts", lambda: {"dana": "+15555550199", "eve": "+15555550177"})
    monkeypatch.setattr(outbound_call.requests, "post", lambda *a, **k: twilio_created(k["data"]["To"]))
    with cfg.override({"phone": {"public_host": "example.invalid"}}):
        yield tmp_path / "missions.json"


def test_two_processes_placing_calls_at_once_both_keep_their_mission(dialable, monkeypatch):
    """The second writer waits for the first one's read-modify-write instead of overwriting it."""
    first_read, release, second_read = threading.Event(), threading.Event(), threading.Event()

    class Missions(type(dialable)):
        def read_text(self, *args, **kwargs):
            if threading.current_thread().name == "first":
                first_read.set()
                release.wait(5)
            else:
                second_read.set()
            return super().read_text(*args, **kwargs)

    dialable.write_text("{}")
    monkeypatch.setattr(outbound_call, "MISSIONS", Missions(dialable))
    first = threading.Thread(target=outbound_call.place, args=("dana", "Dinner at seven"), name="first")
    second = threading.Thread(target=outbound_call.place, args=("eve", "Bring the ladder"), name="second")
    first.start()
    assert first_read.wait(5)
    second.start()
    assert not second_read.wait(0.5)  # held at the lock while the first is between its read and its write
    release.set()
    first.join(5)
    second.join(5)
    assert outbound_call.take_mission("+15555550199") == "Dinner at seven"
    assert outbound_call.take_mission("+15555550177") == "Bring the ladder"
    assert stat.S_IMODE(dialable.stat().st_mode) == 0o600


@pytest.mark.parametrize("bad", ["", '{"+1555', "[]"], ids=["empty", "half-written", "not-an-object"])
def test_a_bad_missions_file_never_stops_a_call(dialable, bad):
    dialable.write_text(bad)
    assert outbound_call.take_mission("+15555550199") is None
    assert outbound_call.place("dana", "Dinner at seven") == (True, "Calling dana now.")
    assert outbound_call.take_mission("+15555550199") == "Dinner at seven"


def test_a_bad_missions_file_is_kept_aside_not_written_over(dialable, caplog):
    # The next call's write used to replace a file it couldn't read, and every mission in it was gone unseen
    garbled = '{"15555550177": {"say": "Bring the ladder", "created": 17'
    dialable.write_text(garbled)
    assert outbound_call.place("dana", "Dinner at seven") == (True, "Calling dana now.")
    (kept,) = dialable.parent.glob("missions.json.unreadable-*")
    assert kept.read_text() == garbled
    assert any(r.levelname == "ERROR" and kept.name in r.getMessage() for r in caplog.records)
    assert outbound_call.take_mission("+15555550199") == "Dinner at seven"


def test_a_bad_missions_file_that_cant_be_moved_aside_refuses_the_call(dialable, monkeypatch):
    class Stuck(type(dialable)):
        def rename(self, target):
            raise PermissionError(13, "Permission denied")

    post = create_autospec(requests.post)
    monkeypatch.setattr(outbound_call.requests, "post", post)
    monkeypatch.setattr(outbound_call, "MISSIONS", Stuck(dialable))
    dialable.write_text('{"+1555')
    ok, _ = outbound_call.place("dana", "Dinner at seven")
    assert not ok
    assert outbound_call.take_mission("+15555550199") is None
    assert dialable.read_text() == '{"+1555'
    post.assert_not_called()


@pytest.mark.parametrize("error", [requests.ConnectTimeout, requests.ReadTimeout, requests.ConnectionError])
def test_twilio_unreachable_is_a_clear_failure_not_a_crash(dialable, monkeypatch, error):
    def post(*args, **kwargs):
        raise error("twilio down")

    monkeypatch.setattr(outbound_call.requests, "post", post)
    ok, detail = outbound_call.place("dana", "Dinner at seven")
    assert not ok
    assert detail == f"I couldn't reach Twilio to place the call ({error.__name__})."


def saved_call(tmp_path, **kw):
    """A finished call's call.json, as finalize writes it."""
    call = phone_call(STRANGER)
    call.dir = tmp_path
    call.in_frames = [np.zeros(800, np.int16)]
    call.transcript = [
        {"t": 0.0, "speaker": "caller", "text": "Hello?"},
        {"t": 1.0, "speaker": "bot", "text": "So that's Friday after seven. Thanks, Dana."},
    ]
    for k, v in kw.items():
        setattr(call, k, v)
    call.finalize()
    return json.loads((tmp_path / "call.json").read_text())


def test_a_placed_call_saves_its_direction_whom_we_called_what_for_and_what_came_of_it(tmp_path):
    meta = saved_call(tmp_path, direction="out", to_name="Dana", instructions="Ask whether she's free Friday")
    assert meta["direction"] == "out"
    assert meta["to"] == {"name": "Dana", "number": STRANGER}
    assert meta["instructions"] == "Ask whether she's free Friday"
    assert meta["outcome"] == "So that's Friday after seven. Thanks, Dana."
    assert meta["from"] == STRANGER  # unchanged: the other party's number


def test_a_call_answered_saves_its_direction_and_no_outcome_without_instructions(tmp_path):
    meta = saved_call(tmp_path)
    assert (meta["direction"], meta["to"], meta["instructions"], meta["outcome"]) == ("in", None, None, None)


def test_a_call_is_marked_live_while_it_is_on_and_not_after(tmp_path):
    """The dashboard's "on a call" reads this mark."""
    call = phone_call(STRANGER)
    call.dir = tmp_path
    call.direction, call.to_name = "out", "Dana"
    call.mark_live()
    live = json.loads((tmp_path / "live.json").read_text())
    assert {k: live[k] for k in ("callSid", "direction", "number", "name", "pid")} == {
        "callSid": "CA1",
        "direction": "out",
        "number": STRANGER,
        "name": "Dana",
        "pid": os.getpid(),
    }
    assert live["since"] == call.started
    call.end_live()
    assert not (tmp_path / "live.json").exists()
    call.end_live()  # twice is harmless


def test_an_owner_calling_in_is_named_on_the_live_mark(tmp_path):
    with cfg.override({"owner": {"name": "Alice"}}):
        call = phone_call(OWNER)
        call.dir = tmp_path
        call.verified_owner = True
        call.mark_live()
    assert json.loads((tmp_path / "live.json").read_text())["name"] == "Alice"


def test_tool_rule_uses_pack_services():
    rule = phone_bridge.phone_tool_rule()
    assert str(cfg.ROOT / ".venv" / "bin" / "pack-services") + " status" in rule
    assert "launchctl" not in rule


def test_a_caller_is_named_by_the_voice_lookup_and_unnamed_without_it(monkeypatch):
    """The bridge names a caller by the stored voice their clip is nearest (voice_lookup.py; its own tests cover the
    names), and by nobody while the vector worker can't answer."""
    from liberational.voice_lookup import Heard, VoiceLookup

    lookup = create_autospec(VoiceLookup, instance=True)
    lookup.identify.return_value = Heard(4, "Dana", 0.812)
    monkeypatch.setattr(phone_bridge, "_spk", lookup)
    assert phone_bridge.identify_speaker(np.zeros(16000, np.float32)) == ("Dana", 0.81)
    lookup.identify.side_effect = TimeoutError("no vector worker")
    assert phone_bridge.identify_speaker(np.zeros(16000, np.float32)) == (None, None)
    monkeypatch.setattr(phone_bridge, "_spk", None)
    assert phone_bridge.identify_speaker(np.zeros(16000, np.float32)) == (None, None)


OWNER, TWILIO, STRANGER = "+15555550123", "+15555550100", "+15555550199"
HOST = "pack.example.ts.net"


class Webhook:
    """POST a Twilio-signed voice webhook (Twilio's algorithm: HMAC-SHA1 over the URL and sorted params). Calling it
    returns the call record the setup leaves for the media stream, which then takes it (as a started stream does);
    raw posts without taking it and returns the response."""

    def __init__(self, client):
        self.client = client

    def raw(self, **form):
        signed = f"https://{HOST}/bridge/voice" + "".join(k + form[k] for k in sorted(form))
        sig = base64.b64encode(hmac.new(b"tok", signed.encode(), hashlib.sha1).digest()).decode()
        return self.client.post("/bridge/voice", data=form, headers={"X-Twilio-Signature": sig})

    def __call__(self, **form):
        assert self.raw(**form).status_code == 200
        (record,) = phone_bridge.PENDING.values()
        phone_bridge.PENDING.clear()
        return record


@pytest.fixture
def webhook(env_file, monkeypatch, tmp_path):
    env_file.write_text(f"TWILIO_ACCOUNT_SID=AC123\nTWILIO_AUTH_TOKEN=tok\nTWILIO_FROM_NUMBER={TWILIO}\n")
    monkeypatch.setattr(phone_bridge, "PENDING", {})
    monkeypatch.setattr(phone_bridge, "SEEN_CALLS", tmp_path / "phone_seen_calls.json")
    with cfg.override({"owner": {"name": "Alice", "phone": OWNER}, "phone": {"public_host": HOST}}):
        yield Webhook(TestClient(phone_bridge.app))


@pytest.mark.parametrize("ours", [TWILIO, "+1 555-555-0100", "+1 (555) 555.0100"], ids=["e164", "dashes", "mixed"])
def test_a_call_we_placed_is_recognized_whatever_the_format_of_our_number(webhook, ours):
    with cfg.override({"phone": {"twilio_number": ours}}):
        placed = webhook(CallSid="CA1", From=TWILIO, To=STRANGER, Direction="outbound-api")
        incoming = webhook(CallSid="CA2", From=STRANGER, To=TWILIO, Direction="inbound")
    assert (placed["remote"], incoming["remote"]) == (STRANGER, STRANGER)


# Owner treatment needs the owner's number AND grade A caller-ID attestation. Twilio signs Direction and
# StirVerstat, so a caller can spoof From but not those.
A, B, C = "TN-Validation-Passed-A", "TN-Validation-Passed-B", "TN-Validation-Passed-C"


@pytest.mark.parametrize(
    "form, remote, owner",
    [
        pytest.param(dict(From=OWNER, To=TWILIO, Direction="inbound", StirVerstat=A), OWNER, True, id="owner-grade-a"),
        pytest.param(dict(From=OWNER, To=TWILIO, Direction="inbound", StirVerstat=B), OWNER, False, id="grade-b"),
        pytest.param(dict(From=OWNER, To=TWILIO, Direction="inbound", StirVerstat=C), OWNER, False, id="grade-c"),
        pytest.param(
            dict(From=OWNER, To=TWILIO, Direction="inbound", StirVerstat="TN-Validation-Failed-A"),
            OWNER,
            False,
            id="failed",
        ),
        pytest.param(dict(From=OWNER, To=TWILIO, Direction="inbound"), OWNER, False, id="no-attestation"),
        pytest.param(
            dict(From=STRANGER, To=TWILIO, Direction="inbound", StirVerstat=A), STRANGER, False, id="stranger"
        ),
        pytest.param(dict(From=TWILIO, To=OWNER, Direction="inbound"), TWILIO, False, id="inbound-spoofing-our-number"),
        pytest.param(dict(From=TWILIO, To=OWNER, Direction="outbound-api"), OWNER, True, id="we-called-the-owner"),
    ],
)
def test_owner_treatment_needs_the_owners_number_and_grade_a(webhook, form, remote, owner):
    record = webhook(CallSid="CA1", **form)
    assert (record["remote"], record["owner"], record["verified"]) == (remote, owner, owner)


@pytest.fixture
def auth(tmp_path, monkeypatch):
    """The voice password "open sesame" (voice_auth's verifier format), and the state dir its lockouts live in."""
    salt = bytes(range(16))
    digest = hashlib.pbkdf2_hmac("sha256", b"open sesame", salt, 100000).hex()
    verifier = tmp_path / "voice_auth.json"
    verifier.write_text(json.dumps({"version": 1, "iterations": 100000, "salt": salt.hex(), "hash": digest}))
    verifier.chmod(0o600)
    monkeypatch.setattr(phone_bridge, "STATE_DIR", tmp_path)
    monkeypatch.setattr(phone_bridge, "LOCKOUT_NOTICE", tmp_path / "phone_lockout_notice.json")
    monkeypatch.setattr(phone_bridge, "GREETING_ONCE", tmp_path / "greeting_once.json")
    monkeypatch.setattr(phone_bridge, "_last_agent_notice", [0.0])
    with cfg.override({"owner": {"name": "Alice", "phone": OWNER}}):
        yield voice_auth.Lockouts(tmp_path / "voice_lockout.json")


def phone_call(number=OWNER):
    call = phone_bridge.Call(cast(phone_bridge.WebSocket, None))
    call.from_number, call.call_sid = number, "CA1"
    return call


def test_the_password_unlocks_owner_treatment_for_that_call_only(auth):
    call = phone_call()
    assert call.try_password("my password is open sesame") == "Thanks, Alice. Go ahead."
    assert call.owner and call.caller_default == "Alice (phone)"
    assert not phone_call().owner  # the next call starts as a guest again


def test_wrong_passwords_lock_out_the_number_across_calls_and_then_refuse_the_right_one(auth):
    replies = [phone_call().try_password("password is guess") for _ in range(voice_auth.TRIES)]
    assert replies == ["That isn't the password."] * (voice_auth.TRIES - 1) + [phone_bridge.LOCKED_OUT]
    call = phone_call()
    assert call.try_password("password is open sesame") == phone_bridge.LOCKED_OUT
    assert not call.owner


def test_spoofing_a_new_number_per_call_still_hits_the_phones_lockout_but_not_the_rooms(auth):
    for n in range(voice_auth.EVERYONE_TRIES):
        phone_call(f"+1555555{n:04d}").try_password("password is guess")
    assert phone_call("+15555559999").try_password("password is open sesame") == phone_bridge.LOCKED_OUT
    assert not voice_auth.Lockouts(auth.path).locked("voice:1")  # people in the room have their own budget


def test_an_in_person_lockout_leaves_the_phone_alone(auth):
    for n in range(voice_auth.EVERYONE_TRIES):
        auth.fail(f"voice:{n}")
    call = phone_call()
    assert call.try_password("password is open sesame") == "Thanks, Alice. Go ahead."


def test_after_an_owner_only_ask_the_next_thing_said_is_a_password_try(auth, monkeypatch):
    said = iter(["What's on the camera?", "open sesame"])
    monkeypatch.setattr(phone_bridge, "transcribe", lambda audio: next(said))
    monkeypatch.setattr(phone_bridge, "identify_speaker", lambda audio: (None, None))
    call, spoken = phone_call(), []

    async def speak(text):
        spoken.append(text)

    monkeypatch.setattr(call, "speak", speak)
    monkeypatch.setattr(phone_bridge, "owner_turn", lambda text, owner: (phone_bridge.owner_only(), None))
    for _ in range(2):
        asyncio.run(call.handle_utterance([np.full(160, 1000, np.int16)]))
    assert spoken == [phone_bridge.owner_only(), "Thanks, Alice. Go ahead."]
    assert call.owner
    assert all(t["content"] != "open sesame" for t in call.history)  # a password try never reaches the model


@pytest.mark.parametrize(
    "said, kept",
    [
        pytest.param("password is open sesame", "[password]", id="the-password"),
        pytest.param("password is guess", "password is guess", id="a-wrong-guess"),
        pytest.param("the gate code is 4821", "the gate code is 4821", id="a-number"),
    ],
)
def test_only_the_password_itself_is_blanked_in_the_transcript_and_log(auth, monkeypatch, tmp_path, caplog, said, kept):
    monkeypatch.setattr(phone_bridge, "transcribe", lambda audio: said)
    monkeypatch.setattr(phone_bridge, "identify_speaker", lambda audio: (None, None))
    monkeypatch.setattr(phone_bridge, "ask_llm", lambda history, system: "Okay.")
    call = phone_call()
    call.dir = tmp_path / "call"
    call.dir.mkdir()

    async def speak(text):
        pass

    monkeypatch.setattr(call, "speak", speak)
    audio = [np.full(160, 1000, np.int16)]
    with caplog.at_level("INFO", logger=phone_bridge.log.name):
        asyncio.run(call.handle_utterance(audio))
    written = json.loads((call.dir / "transcript.jsonl").read_text().splitlines()[0])["text"]
    assert (written, call.transcript[0]["text"]) == (kept, kept)
    assert ("open sesame" in caplog.text) is False
    assert (said in caplog.text) is (kept == said)
    assert audio[0].tolist() == [1000] * 160  # the recording is left as it was


@pytest.fixture
def mission_for(tmp_path, monkeypatch):
    """Leave a message for a number, as outbound.place does when it calls someone."""
    path = tmp_path / "outbound_missions.json"
    monkeypatch.setattr(outbound_call, "MISSIONS", path)

    def leave(number, say):
        key = number  # missions are keyed by the complete E.164 destination
        path.write_text(json.dumps({key: {"say": say, "created": time.time()}}))

    return leave


@pytest.mark.parametrize(
    "form, delivered",
    [
        pytest.param(dict(From=STRANGER, To=TWILIO, Direction="inbound"), None, id="unvouched-callback"),
        pytest.param(dict(From=STRANGER, To=TWILIO, Direction="inbound", StirVerstat=B), None, id="grade-b-callback"),
        pytest.param(dict(From=STRANGER, To=TWILIO, Direction="inbound", StirVerstat=A), "the gate code", id="grade-a"),
        pytest.param(dict(From=TWILIO, To=STRANGER, Direction="outbound-api"), "the gate code", id="we-called-them"),
    ],
)
def test_a_message_left_for_a_number_goes_only_to_a_vouched_for_caller(webhook, mission_for, form, delivered):
    mission_for(STRANGER, "the gate code")
    assert webhook(CallSid="CA1", **form)["say"] == delivered


def test_the_setup_says_which_way_a_call_went_and_for_one_we_placed_whom_and_what_for(webhook, tmp_path, monkeypatch):
    path = tmp_path / "outbound_missions.json"
    monkeypatch.setattr(outbound_call, "MISSIONS", path)
    mission = {
        "say": "Hi, I'm calling for Alice.",
        "message": "Ask about Friday",
        "name": "Dana",
        "created": time.time(),
    }
    path.write_text(json.dumps({STRANGER: mission}))
    placed = webhook(CallSid="CA1", From=TWILIO, To=STRANGER, Direction="outbound-api")
    assert (placed["direction"], placed["to_name"], placed["instructions"]) == ("out", "Dana", "Ask about Friday")
    incoming = webhook(CallSid="CA2", From=STRANGER, To=TWILIO, Direction="inbound")
    assert (incoming["direction"], incoming["to_name"], incoming["instructions"]) == ("in", None, None)


def test_a_call_we_placed_to_the_owner_with_a_message_is_not_owner_mode_but_is_the_owner(webhook, mission_for):
    mission_for(OWNER, "the gate code")
    record = webhook(CallSid="CA1", From=TWILIO, To=OWNER, Direction="outbound-api")
    assert (record["owner"], record["verified"], record["say"]) == (False, True, "the gate code")


@pytest.mark.parametrize(
    "said",
    ["Open sesame.", "The password is open sesame.", "Um, password: open sesame", "My password's open sesame!"],
)
def test_the_password_is_recognized_as_speech_recognition_writes_it(auth, said):
    call = phone_call()
    assert call.try_password(said) == "Thanks, Alice. Go ahead."


def test_a_save_that_fails_still_locks_out_a_guessing_caller(auth, monkeypatch):
    def disk_full(*args):
        raise OSError("no space left on device")

    monkeypatch.setattr(voice_auth.os, "replace", disk_full)
    replies = [phone_call().try_password("password is guess") for _ in range(voice_auth.TRIES)]
    assert replies[-1] == phone_bridge.LOCKED_OUT
    assert phone_call().try_password("password is open sesame") == phone_bridge.LOCKED_OUT


@pytest.fixture
def talk(auth, monkeypatch):
    """Say lines to a call in turn; returns (the call, what it said back, what reached a model)."""
    call, spoken, model = phone_call(), [], []

    async def speak(text):
        spoken.append(text)

    async def owner_agent_reply(text, hint):
        model.append(text)
        return "Okay."

    def ask_llm(history, system):
        model.append(history[-1]["content"])
        return "Okay."

    monkeypatch.setattr(call, "speak", speak)
    monkeypatch.setattr(call, "owner_agent_reply", owner_agent_reply)
    monkeypatch.setattr(phone_bridge, "ask_llm", ask_llm)
    monkeypatch.setattr(phone_bridge, "identify_speaker", lambda audio: (None, None))

    def say(*lines):
        heard = iter(lines)
        monkeypatch.setattr(phone_bridge, "transcribe", lambda audio: next(heard))
        for _ in lines:
            asyncio.run(call.handle_utterance([np.full(160, 1000, np.int16)]))

    return SimpleNamespace(call=call, spoken=spoken, model=model, say=say)


def test_an_owner_who_says_the_password_again_is_not_passed_on(talk):
    talk.call.owner = True
    talk.say("password is open sesame")
    assert (talk.spoken, talk.model, talk.call.transcript[0]["text"]) == (["You're already through."], [], "[password]")


def test_a_request_after_the_password_prompt_is_not_a_try_and_is_answered(talk, auth):
    talk.say("Remember to buy milk", "tell him I called")
    assert talk.spoken[0] == phone_bridge.owner_only()
    assert talk.model == ["tell him I called"]
    assert not auth.counts and not talk.call.awaiting_password  # no try used up, and the prompt is over


def test_never_mind_after_the_password_prompt_cancels_without_a_try(talk, auth):
    talk.say("Remember to buy milk", "never mind")
    assert talk.spoken[-1] == "Okay."
    assert not auth.counts


def test_after_a_wrong_try_the_next_line_is_not_a_try(talk):
    talk.say("Remember to buy milk", "sesame open", "how's the weather")
    assert talk.spoken[1] == "That isn't the password."
    assert talk.model == ["how's the weather"]


# A captured, signed /bridge/voice request replayed later can't set the call up again.
INBOUND = dict(From=STRANGER, To=TWILIO, Direction="inbound")


def test_a_replayed_setup_is_refused_once_the_call_has_started(webhook):
    webhook(CallSid="CA1", **INBOUND)  # set up, and its media stream took the token
    assert webhook.raw(CallSid="CA1", **INBOUND).status_code == 403
    assert phone_bridge.PENDING == {}


def test_twilios_retry_of_an_unanswered_setup_gets_the_same_answer(webhook, mission_for):
    mission_for(STRANGER, "the gate code")
    form = dict(CallSid="CA1", StirVerstat=A, **INBOUND)
    first, retry = webhook.raw(**form), webhook.raw(**form)
    assert (first.status_code, retry.status_code, retry.text) == (200, 200, first.text)  # the same token
    (record,) = phone_bridge.PENDING.values()
    assert record["say"] == "the gate code"  # the message the first setup took isn't lost on the retry


def test_a_setup_without_a_call_id_is_refused(webhook):
    assert webhook.raw(**INBOUND).status_code == 400


def test_the_seen_calls_survive_a_restart(webhook, monkeypatch):
    webhook(CallSid="CA1", **INBOUND)
    monkeypatch.setattr(phone_bridge, "PENDING", {})  # a new process: nothing in memory
    assert webhook.raw(CallSid="CA1", **INBOUND).status_code == 403
    assert stat.S_IMODE(os.stat(phone_bridge.SEEN_CALLS).st_mode) == 0o600


def test_the_seen_calls_are_bounded_by_age_and_count(webhook, monkeypatch):
    phone_bridge.SEEN_CALLS.write_text(json.dumps({"CAold": time.time() - phone_bridge.SEEN_KEEP_S - 1}))
    monkeypatch.setattr(phone_bridge, "SEEN_MAX", 3)
    for n in range(5):
        webhook(CallSid=f"CA{n}", **INBOUND)
    assert sorted(json.loads(phone_bridge.SEEN_CALLS.read_text())) == ["CA2", "CA3", "CA4"]


def test_an_unreadable_record_of_seen_calls_starts_over_rather_than_refusing_calls(webhook):
    phone_bridge.SEEN_CALLS.write_text("{not json")
    assert webhook(CallSid="CA1", **INBOUND)["remote"] == STRANGER


# Telling the owner when the phone locks out a password guesser.
def lock_out(number=STRANGER):
    for _ in range(voice_auth.TRIES):
        reply = phone_call(number).try_password("password is guess")
    return reply


def test_a_phone_lockout_is_logged_and_told_once_on_the_next_owner_call(auth, caplog):
    with caplog.at_level("WARNING", logger=phone_bridge.log.name):
        assert lock_out() == phone_bridge.LOCKED_OUT
    assert "phone password lockout" in caplog.text
    greeting = phone_bridge.take_greeting_once()
    assert greeting is not None and greeting.startswith("Heads up: someone tried the password on the phone")
    assert phone_bridge.take_greeting_once() is None  # once


def test_many_lockouts_make_one_notice_that_counts_them(auth):
    for n in range(2):  # a third number would already meet the phone-wide lockout
        lock_out(f"+1555555{n:04d}")
    greeting = phone_bridge.take_greeting_once()
    assert greeting is not None and "locked out 2 times" in greeting


def test_a_waiting_greeting_and_the_notice_are_both_said(auth):
    phone_bridge.GREETING_ONCE.write_text(json.dumps({"text": "Dana called.", "created": time.time()}))
    lock_out()
    greeting = phone_bridge.take_greeting_once()
    assert greeting is not None and greeting.startswith("Dana called. Heads up:")


def test_the_owner_who_unlocks_a_call_with_the_password_hears_the_notice(auth):
    lock_out()
    reply = phone_call().try_password("password is open sesame")
    assert reply.startswith("Thanks, Alice. Go ahead. Heads up: someone tried the password")


@pytest.fixture
def agent(monkeypatch):
    """A configured agent (its HTTP endpoint) that records what it is asked; the notice's thread runs in place."""
    told = []

    class Reply:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "Passed it on."}}]}

    def post(url, json, timeout, headers):
        told.append(json["messages"][0]["content"])
        return Reply()

    monkeypatch.setattr(phone_bridge.requests, "post", post)

    class Inline:
        def __init__(self, target, args, **kw):
            self.target, self.args = target, args

        def start(self):
            self.target(*self.args)

    monkeypatch.setattr(phone_bridge.threading, "Thread", Inline)
    with cfg.override({"agent": {"url": "http://127.0.0.1:18789/v1/chat/completions"}}):
        yield told


def test_the_agent_hears_of_a_lockout_at_most_once_per_half_hour(auth, agent):
    lock_out("+15555550001")
    lock_out("+15555550002")
    assert len(agent) == 1 and agent[0].startswith("Please pass this notice from the phone line on to Alice")


def test_without_an_agent_nothing_is_sent(auth, monkeypatch):
    sent = []
    monkeypatch.setattr(phone_bridge.requests, "post", lambda *a, **k: sent.append(a))
    lock_out()
    assert sent == []


def test_a_phone_wide_lockout_from_rotating_numbers_is_noticed(auth):
    for n in range(voice_auth.EVERYONE_TRIES):
        phone_call(f"+1555555{n:04d}").try_password("password is guess")  # one guess each: no number locks
    greeting = phone_bridge.take_greeting_once()
    assert greeting is not None and greeting.startswith("Heads up:")


def test_an_old_notice_expires_and_a_new_one_counts_from_one(auth):
    week_ago = time.time() - phone_bridge.LOCKOUT_NOTICE_KEEP_S - 60
    phone_bridge.notice_lockout(STRANGER, week_ago)
    assert phone_bridge.take_greeting_once() is None
    phone_bridge.notice_lockout(STRANGER, week_ago)
    phone_bridge.notice_lockout(STRANGER, time.time())
    greeting = phone_bridge.take_greeting_once()
    assert greeting is not None and "locked out once" in greeting


def test_the_notice_names_the_day_when_it_wasnt_today(auth):
    now = time.mktime(time.strptime("2026-10-28 09:00", "%Y-%m-%d %H:%M"))  # a Wednesday
    last = now - 86400  # Tuesday
    assert phone_bridge.notice_text(1, last, now).endswith("most recently on Tuesday at 09:00.")
    assert phone_bridge.notice_text(2, now, now).endswith("locked out 2 times, most recently at 09:00.")


@pytest.mark.parametrize("how", ["open-to-others", "a-link"])
def test_a_planted_notice_says_nothing(auth, tmp_path, how):
    numbers = json.dumps({"count": 9, "first": time.time(), "last": time.time()})
    if how == "a-link":
        target = tmp_path / "elsewhere.json"
        target.write_text(numbers)
        target.chmod(0o600)
        phone_bridge.LOCKOUT_NOTICE.symlink_to(target)
    else:
        phone_bridge.LOCKOUT_NOTICE.write_text(numbers)
        phone_bridge.LOCKOUT_NOTICE.chmod(0o644)
    assert phone_bridge.take_greeting_once() is None


def test_the_agent_hears_again_after_half_an_hour(auth, agent):
    start = time.time()
    phone_bridge.notice_lockout(STRANGER, start)
    phone_bridge.notice_lockout(STRANGER, start + 60)
    phone_bridge.notice_lockout(STRANGER, start + phone_bridge.LOCKOUT_AGENT_EVERY_S)
    assert len(agent) == 2


def test_mission_keeps_goal_and_prior_turns_for_interactive_followup(auth, monkeypatch):
    said = iter(["Tomorrow works", "At seven please"])
    monkeypatch.setattr(phone_bridge, "transcribe", lambda audio: next(said))
    monkeypatch.setattr(phone_bridge, "identify_speaker", lambda audio: (None, None))
    replies = iter(["Which time tomorrow?", "Seven tomorrow, understood."])
    seen = []

    def answer(history, system):
        seen.append((list(history), system))
        return next(replies)

    monkeypatch.setattr(phone_bridge, "ask_llm", answer)
    call, spoken = phone_call(STRANGER), []
    call.say = "Arrange a dinner time with the recipient."
    call.history = [{"role": "assistant", "content": "What day works for dinner?"}]

    async def speak(text):
        spoken.append(text)

    monkeypatch.setattr(call, "speak", speak)
    asyncio.run(call.handle_utterance([np.zeros(160, dtype=np.int16)]))
    asyncio.run(call.handle_utterance([np.zeros(160, dtype=np.int16)]))
    assert spoken == ["Which time tomorrow?", "Seven tomorrow, understood."]
    assert "Arrange a dinner time" in seen[0][1] and "Arrange a dinner time" in seen[1][1]
    assert seen[1][0][-3:] == [
        {"role": "user", "content": "Tomorrow works"},
        {"role": "assistant", "content": "Which time tomorrow?"},
        {"role": "user", "content": "At seven please"},
    ]


@pytest.mark.parametrize(
    "said",
    [
        "Bye.",
        "Okay, bye Ava.",
        "I'm your closed. Bye bomb.",
        "Thanks, goodbye!",
        "Talk to you later.",
        "Bye for now.",
        "Okay, bye then.",
    ],
)
@pytest.mark.parametrize("owner", [False, True], ids=["guest", "owner"])
def test_a_goodbye_ends_the_call_at_once_without_waiting_on_the_model(talk, monkeypatch, said, owner):
    talk.call.owner = owner
    hung = []
    monkeypatch.setattr(phone_bridge, "hangup_call", lambda sid: hung.append(sid) or "204")
    talk.call.say = "Tell them a joke."
    with cfg.override(ALICE_AND_AVA):  # "bye Ava" is a goodbye because Ava is the assistant's name
        talk.say(said)
    assert (talk.spoken, talk.model, hung) == (["Okay, bye!"], [], ["CA1"])


@pytest.mark.parametrize(
    "said",
    [
        "Wait, before you say bye, got another one?",
        "Say bye to Dana for me.",
        "Don't say bye yet.",
        "No need to say goodbye now!",
        "Before you say goodbye though.",
        "I wanted to say goodbye properly.",
        "Don't hang up.",
    ],
)
def test_bye_inside_a_question_or_a_request_is_not_a_goodbye(talk, monkeypatch, said):
    monkeypatch.setattr(phone_bridge, "hangup_call", lambda sid: pytest.fail("hung up"))
    talk.call.say = "Tell them a joke."
    talk.say(said)
    assert (talk.spoken, talk.model) == (["Okay."], [said])


@pytest.mark.parametrize(
    "reply,hangs_up",
    [("Okay, bye for now!", True), ("One more thing before we say goodbye though.", False)],
    ids=["goodbye", "talk-about-goodbye"],
)
def test_a_mission_reply_hangs_up_only_on_a_real_goodbye(talk, monkeypatch, reply, hangs_up):
    hung = []
    monkeypatch.setattr(phone_bridge, "hangup_call", lambda sid: hung.append(sid) or "204")
    monkeypatch.setattr(phone_bridge, "ask_llm", lambda history, system: reply)
    talk.call.say = "Tell them a joke."
    talk.say("Ha, good one.")
    assert (talk.spoken, hung) == ([reply], ["CA1"] if hangs_up else [])


def test_an_owner_who_says_not_to_hang_up_is_answered(talk, monkeypatch):
    monkeypatch.setattr(phone_bridge, "hangup_call", lambda sid: pytest.fail("hung up"))
    talk.call.owner = True
    talk.say("Don't hang up, one more thing.")
    assert (talk.spoken, talk.model) == (["Okay."], ["Don't hang up, one more thing."])


def caller_frame(level):
    return base64.b64encode(phone_bridge.ulaw_encode(np.full(160, level, np.int16))).decode()


FIRST_HALF, WHOLE = "That was a very ridiculous", "That was a very ridiculous limerick. You can do better."
REPLIES = {FIRST_HALF: "Thanks, glad you liked it!", WHOLE: "Fair, that one fell flat."}


@pytest.fixture
def pausing(auth, monkeypatch):
    """A call whose caller says a line, pauses, and may go on while the turn is made: while it is transcribed
    (during="transcribe") or while the model answers (during="model"). The model is the local one on a guest or
    mission call and the agent on an owner call."""
    call = phone_call(STRANGER)
    call.say = "Tell them a limerick."
    lines = [FIRST_HALF, WHOLE]  # what each transcription hears, in turn; a test may replace them
    held, made = threading.Event(), threading.Event()
    audio, asked, spoken = [], [], []
    hold = {"during": "model"}

    def wait_if(step, done):
        if hold["during"] == step and len(done) == 1:  # only the first turn is held
            held.set()
            made.wait(120)

    def transcribe(pcm):
        audio.append(len(pcm))
        wait_if("transcribe", audio)
        return lines[len(audio) - 1]

    def answer(text):
        asked.append(text)
        wait_if("model", asked)
        return REPLIES[text]

    async def owner_agent_reply(text, hint):
        return await asyncio.to_thread(answer, text)

    async def speak(text, then="listening"):
        spoken.append(text)
        call.state = then

    monkeypatch.setattr(phone_bridge, "transcribe", transcribe)
    monkeypatch.setattr(phone_bridge, "identify_speaker", lambda pcm: (None, None))
    monkeypatch.setattr(phone_bridge, "ask_llm", lambda history, system: answer(history[-1]["content"]))
    monkeypatch.setattr(call, "owner_agent_reply", owner_agent_reply)
    monkeypatch.setattr(call, "speak", speak)

    async def frames(level, count):
        for _ in range(count):
            await call.on_media(caller_frame(level))

    async def run(went_on):
        try:
            await frames(8000, 50)  # a second of speech, then the pause that ends it
            await frames(0, phone_bridge.END_SILENCE_FRAMES)
            assert await asyncio.to_thread(held.wait, 120), "the turn never got to the held step"
            await frames(8000, went_on)  # said while the turn is made
            made.set()
            deadline = time.monotonic() + 120  # generous: the Linux machine can be loaded
            while not spoken:  # the line goes quiet until something is said back
                assert time.monotonic() < deadline, "nothing was said back"
                await frames(0, 1)
                await asyncio.sleep(0.01)
        finally:
            made.set()  # a failed run must not leave the held thread waiting out its 120 s

    def go(went_on, during="model", owner=False):
        hold["during"], call.owner = during, owner
        asyncio.run(run(went_on))

    return SimpleNamespace(call=call, audio=audio, asked=asked, spoken=spoken, lines=lines, run=go)


def callers_turns(call):
    return [turn["text"] for turn in call.transcript if turn["speaker"] == "caller"]


@pytest.mark.parametrize("owner", [False, True], ids=["mission", "owner"])
def test_a_caller_who_goes_on_while_the_reply_is_made_is_answered_for_all_of_it(pausing, owner):
    # A caller who paused 1.2 s inside "That was a very ridiculous ... limerick, Bob, I think you can do better" got
    # a reply to the first half (taken as praise), and Bob talked over the rest.
    pausing.run(50, owner=owner)
    assert pausing.spoken == ["Fair, that one fell flat."]
    assert pausing.asked == [FIRST_HALF, WHOLE]
    assert pausing.audio[1] > pausing.audio[0]  # heard again from the start, not just the part after the pause
    assert callers_turns(pausing.call) == [WHOLE]  # the first half isn't left in the transcript as its own turn
    # Owner calls keep the exchange too: the fast local model answers from it (phone.fast_replies).
    assert pausing.call.history == [
        {"role": "user", "content": WHOLE},
        {"role": "assistant", "content": REPLIES[WHOLE]},
    ]


@pytest.mark.parametrize("owner", [False, True], ids=["mission", "owner"])
def test_a_caller_who_goes_on_while_they_are_transcribed_never_reaches_the_model_with_half(pausing, owner):
    pausing.run(50, during="transcribe", owner=owner)
    assert pausing.spoken == ["Fair, that one fell flat."]
    assert pausing.asked == [WHOLE]
    assert pausing.audio[1] > pausing.audio[0]
    assert callers_turns(pausing.call) == [WHOLE]


def test_a_blip_while_the_reply_is_made_does_not_hold_it_up(pausing):
    pausing.run(phone_bridge.RESUME_FRAMES - 1)
    assert pausing.spoken == ["Thanks, glad you liked it!"]
    assert pausing.asked == [FIRST_HALF]


def test_a_right_password_followed_by_more_speech_is_accepted_and_never_written_down(pausing, caplog):
    # A caller says the password, pauses, and goes on while it is checked. Dropping that turn to hear the rest made the
    # whole line ("password is open sesame, can you hear me?") the try: it failed the exact check, counted toward the
    # lockouts, and was logged and written to the transcript with the password in it.
    caplog.set_level("INFO")
    pausing.lines[:] = ["password is open sesame", "password is open sesame, can you hear me?"]
    pausing.run(50, during="transcribe")
    assert pausing.spoken == ["Thanks, Alice. Go ahead."]
    assert pausing.call.owner
    assert callers_turns(pausing.call) == [voice_auth.PASSWORD_SHOWN]
    assert "heard" in caplog.text and "sesame" not in caplog.text
    assert pausing.call.in_speech and len(pausing.call.buf) >= 50  # what they went on to say is the next utterance


def test_a_caller_who_keeps_going_on_is_answered_after_a_bounded_number_of_drops(talk, monkeypatch):
    # Background talk that starts while every turn is made would otherwise drop and re-ask forever.
    frame = np.full(160, 1000, np.int16)

    def transcribe(audio):
        talk.call.in_speech, talk.call.buf = True, [frame] * 10  # someone talks while each turn is made
        return "What time is it"

    monkeypatch.setattr(phone_bridge, "transcribe", transcribe)
    for _ in range(phone_bridge.MAX_WENT_ON):
        asyncio.run(talk.call.handle_utterance([frame] * 50))
        assert (talk.spoken, talk.call.state) == ([], "listening")  # dropped to hear the rest
    asyncio.run(talk.call.handle_utterance([frame] * 50))
    assert (talk.spoken, talk.model) == (["Okay."], ["What time is it"])


def test_going_on_past_the_longest_utterance_answers_the_first_part_and_keeps_the_rest(talk, monkeypatch):
    frame = np.full(160, 1000, np.int16)

    def transcribe(audio):
        talk.call.in_speech, talk.call.buf = True, [frame] * 20
        return "What time is it"

    monkeypatch.setattr(phone_bridge, "transcribe", transcribe)
    asyncio.run(talk.call.handle_utterance([frame] * (phone_bridge.MAX_UTT_FRAMES - 10)))
    assert (talk.spoken, talk.model) == (["Okay."], ["What time is it"])
    assert len(talk.call.buf) == 20  # not lost: the start of the next utterance


def test_the_loop_keeps_serving_while_twilio_is_slow_to_place_a_call(talk, dialable, monkeypatch):
    """The owner's "call Dana" waits on Twilio in a thread, so the event loop (other calls, this call's media
    stream) keeps running meanwhile. The fake Twilio answers only once the loop has run something else."""
    entered, served = threading.Event(), threading.Event()
    answered = []

    def post(*args, **kwargs):
        entered.set()
        answered.append(served.wait(2))  # False if the loop was stuck inside this call
        return twilio_created(kwargs["data"]["To"])

    monkeypatch.setattr(outbound_call.requests, "post", post)
    monkeypatch.setattr(phone_bridge, "transcribe", lambda audio: "call dana and tell her dinner at seven")
    talk.call.owner = True

    async def other_call() -> None:
        await asyncio.to_thread(entered.wait, 2)
        served.set()

    async def both() -> None:
        await asyncio.gather(talk.call.handle_utterance([np.full(160, 1000, np.int16)]), other_call())

    asyncio.run(both())
    assert answered == [True]
    assert talk.spoken == ["Calling dana now."]


def test_an_owner_turn_that_hangs_is_answered_with_a_clear_error(talk, monkeypatch):
    release = threading.Event()

    def owner_turn(text, owner):
        release.wait(5)
        return "Calling dana now.", None

    monkeypatch.setattr(phone_bridge, "owner_turn", owner_turn)
    monkeypatch.setattr(phone_bridge, "OWNER_TURN_TIMEOUT_S", 0.1)
    spoke = talk.call.speak

    async def speak(text):
        await spoke(text)
        release.set()  # lets the stuck thread finish, so the loop's shutdown doesn't wait on it

    monkeypatch.setattr(talk.call, "speak", speak)
    talk.call.owner = True
    talk.say("call dana")
    assert talk.spoken == [phone_bridge.OWNER_TURN_SLOW]
    assert talk.model == []
