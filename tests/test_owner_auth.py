"""Regression tests for the owner and lockout checks: the voice password lockout (and per speaker and across
restarts) and the owner's phone number match."""

import hashlib
import json
import os
import stat
import tomllib
from types import SimpleNamespace

import pytest

from liberational import config as cfg
from liberational.assistant import voice_auth
from liberational.assistant.voice_auth import UNHEARD, Conversation, Grant, Pending
from liberational.phone import numbers, outbound
from voices import write_voices

WAKE = "Hey Bob, reset my private computer"  # needs the password from anyone who isn't the owner
GUEST = {"speaker": "guest"}
WRONG = "Password not accepted. Try again or say cancel."
LOCKED = "Too many attempts. Please wait one minute."
WAIT = "Please wait one minute before trying again."


def make_verifier(folder):
    salt, iterations = b"test salt only", 100_000
    verifier = folder / "verifier.json"
    digest = hashlib.pbkdf2_hmac("sha256", b"dummy secret", salt, iterations).hex()
    verifier.write_text(json.dumps(dict(version=1, salt=salt.hex(), iterations=iterations, hash=digest)))
    verifier.chmod(0o600)
    return verifier


@pytest.fixture
def conversation(tmp_path):
    verifier = make_verifier(tmp_path)
    clock = [1000.0]
    c = Conversation(verifier, lambda: clock[0])

    def say(text, advance=0.0):
        clock[0] += advance
        return c.accept(text, GUEST, False, lambda _: None)

    return say


def test_a_lockout_gives_three_fresh_tries_after_the_cooldown(conversation):
    conversation(WAKE)
    assert [conversation("wrong") for _ in range(3)] == [WRONG, WRONG, LOCKED]
    assert "password" in conversation(WAKE, advance=61)
    assert [conversation("wrong") for _ in range(3)] == [WRONG, WRONG, LOCKED]


@pytest.mark.parametrize(
    "restart",
    [pytest.param([WAKE], id="wake-phrase"), pytest.param(["cancel", WAKE], id="cancel-then-ask")],
)
def test_asking_again_keeps_the_failure_count(conversation, restart):
    conversation(WAKE)
    assert [conversation("wrong") for _ in range(2)] == [WRONG, WRONG]
    assert "password" in [conversation(text) for text in restart][-1]
    assert conversation("wrong") == LOCKED


def test_the_right_password_resets_the_count(conversation):
    conversation(WAKE)
    conversation("wrong")
    conversation("wrong")
    conversation("dummy secret")
    conversation(WAKE)
    assert [conversation("wrong") for _ in range(3)] == [WRONG, WRONG, LOCKED]


class Room:
    """A voice process with separate wall and monotonic clocks, several speakers, and restarts."""

    def __init__(self, folder):
        self.verifier = make_verifier(folder)
        self.state = folder / "voice_lockout.json"
        self.wall, self.mono = [1_000_000.0], [50.0]
        self.restart()

    def restart(self, wall=None, mono=None):
        """A new process: the clocks may have moved, and nothing in memory survives."""
        self.wall[0] = self.wall[0] if wall is None else wall
        self.mono[0] = self.mono[0] if mono is None else mono
        self.c = Conversation(self.verifier, lambda: self.wall[0], lambda: self.mono[0])

    def say(self, text, who="guest", owner=False, voice=None):
        entry = {"speaker": who} if voice is None else {"speaker": who, "speaker_id": voice}
        return self.c.accept(text, entry, owner, lambda _: None)

    def wait(self, seconds):
        self.wall[0] += seconds
        self.mono[0] += seconds

    def wrong_tries(self, n, who="guest"):
        self.say(WAKE, who)
        return [self.say("wrong", who) for _ in range(n)]


@pytest.fixture
def room(tmp_path):
    return Room(tmp_path)


def test_one_speakers_wrong_tries_leave_another_speaker_three(room):
    assert room.wrong_tries(2, "dana") == [WRONG, WRONG]
    room.say("cancel", "dana")
    assert room.wrong_tries(3, "sam") == [WRONG, WRONG, LOCKED]
    assert room.say(WAKE, "dana") != WAIT  # Sam's lockout is Sam's alone


def test_only_the_speaker_who_asked_can_cancel(room):
    room.say(WAKE, "dana")
    assert room.say("cancel", "sam") is None
    assert room.c.pending is not None
    assert "Password accepted" in room.say("dummy secret", "dana")


def test_wrong_tries_from_many_voices_lock_everyone(room):
    # Four different guests. (A line with no voice id needs a person's name to count as anyone.)
    for who in ("eve", "fay"):
        assert room.wrong_tries(2, who) == [WRONG, WRONG]
        room.say("cancel", who)
    assert room.wrong_tries(2, "gus") == [WRONG, LOCKED]  # the sixth wrong try within a minute
    assert room.say(WAKE, "hal") == WAIT
    room.wait(61)
    assert "password" in room.say(WAKE, "hal")


def test_a_lockout_survives_a_restart(room):
    assert room.wrong_tries(3) == [WRONG, WRONG, LOCKED]
    room.restart(mono=7.0)  # monotonic time starts over in a new process
    assert room.say(WAKE) == WAIT
    room.wait(61)
    assert "password" in room.say(WAKE)


def test_a_failure_count_survives_a_restart(room):
    assert room.wrong_tries(2) == [WRONG, WRONG]
    room.restart()
    assert room.wrong_tries(1) == [LOCKED]


def test_the_lockout_file_is_private(room):
    room.wrong_tries(1)
    assert stat.S_IMODE(os.stat(room.state).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(room.state.with_name("voice_lockout.json.lock")).st_mode) == 0o600


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("not json", id="not-json"),
        pytest.param('{"version": 2}', id="unknown-version"),
        pytest.param(
            '{"version": 1, "saved": 1, "counts": {"guest": -4}, "until": {}, "recent": [], "everyone_until": 0}',
            id="negative-count",
        ),
        pytest.param("[]", id="wrong-shape"),
        pytest.param(
            '{"version": 1, "saved": NaN, "counts": {}, "until": {}, "recent": [], "everyone_until": 1e18}', id="nan"
        ),
        pytest.param(
            '{"version": 1, "saved": 1' + "0" * 400 + ', "counts": {}, "until": {}, "recent": [], "everyone_until": 0}',
            id="huge-number",
        ),
        pytest.param("[" * 100_000 + "]" * 100_000, id="deep-nesting"),
        pytest.param(b"\x8c\xff not utf-8", id="bad-bytes"),
    ],
)
def test_an_unreadable_lockout_file_locks_everyone_for_a_minute(room, text):
    room.state.write_bytes(text) if isinstance(text, bytes) else room.state.write_text(text)
    room.restart()
    assert room.say(WAKE, "anyone") == WAIT
    room.wait(61)
    assert "password" in room.say(WAKE, "anyone")


def test_a_corrupt_lockout_file_is_replaced_so_it_locks_only_once(room):
    room.state.write_text("{bad")
    room.restart()
    assert room.say(WAKE) == WAIT
    assert json.loads(room.state.read_text())["version"] == 2
    room.wait(61)
    room.restart()
    assert "password" in room.say(WAKE)


def test_the_owner_can_cancel_a_guests_request(room):
    room.say(WAKE, "guest")
    assert room.say("cancel", "luca", owner=True) == "Cancelled."
    assert room.c.pending is None


def test_the_owners_wake_phrase_replaces_a_guests_request(room):
    room.say(WAKE, "guest")
    assert room.say("Hey Bob, what time is it", "luca", owner=True) is not None
    assert room.c.pending is None or room.c.pending["speaker"] == "luca"


def test_a_relabeled_speaker_cannot_guess_while_its_voice_cools_down(room):
    assert room.wrong_tries(3, "alice")[-1] == LOCKED
    room.say(WAKE, "zed")
    assert room.c.pending is not None
    room.c.pending["speaker"] = "label:alice"  # the pending speaker turns out to be a voice that is cooling down
    assert room.say("wrong", "alice") == WAIT
    assert room.c.pending is None


def test_a_lockout_never_holds_up_the_owner(room):
    for who in ("eve", "fay", "gus"):
        room.wrong_tries(2, who)
        room.say("cancel", who)
    assert room.say(WAKE, "hal") == WAIT  # everyone else waits
    assert room.say("Hey Bob, what time is it", "luca", owner=True) != WAIT


def test_an_owners_pending_request_answers_only_to_the_verified_owner(room):
    """The owner is never asked to confirm today (request_policy), so no owner request waits. If one ever does, a
    line from the owner's voice that isn't the verified owner (heard too faintly, or no longer theirs) can neither
    confirm nor cancel it."""
    room.c.pending = Pending(
        request="reset the computer",
        route="gateway",
        speaker="voice:1",
        asked=room.wall[0],
        expires=room.wall[0] + 45,
        opened=room.wall[0],
        stage="confirm",
        authenticated=False,
        needs_password=False,
        needs_confirmation=True,
        owner=True,
        addressed=True,
    )
    assert room.say("yes confirm", "alice", owner=False, voice=1) is None
    assert room.say("cancel", "alice", owner=False, voice=1) is None
    assert room.c.pending is not None
    grant = room.say("yes confirm", "alice", owner=True, voice=1)
    assert isinstance(grant, Grant) and grant.confirmed and grant.owner


def test_a_line_that_says_nothing_about_who_spoke_starts_no_request(room):
    """Every clip too short to match a voice is "unknown" with no voice id. Shared, that key let one unknown
    person's "yes" confirm another's request, and their wrong passwords lock each other out."""
    assert room.say(WAKE, "unknown") == UNHEARD
    assert room.c.pending is None
    assert [room.say("wrong", "unknown") for _ in range(4)] == [None] * 4  # no request, so nothing to count


def test_a_line_that_says_nothing_about_who_spoke_never_answers_or_cancels(room):
    room.say(WAKE, "dana", voice=2)
    assert room.say("cancel", "unknown") is None
    assert room.say("dummy secret", "unknown") is None
    assert room.c.pending is not None and room.c.pending["speaker"] == "voice:2"
    assert "Password accepted" in (room.say("dummy secret", "dana", voice=2) or "")


def test_only_the_person_who_asked_hears_that_their_request_expired(room):
    room.say(WAKE, "dana", voice=2)
    room.wait(46)
    # Someone else's line ends it quietly and goes on as their own request.
    assert room.say(WAKE, "erin", voice=3) == "Please say your password for this request, or say cancel."
    assert room.c.pending is not None and room.c.pending["speaker"] == "voice:3"
    room.wait(46)
    assert room.say("dummy secret", "erin", voice=3) == "That request expired. Please ask again."


@pytest.mark.parametrize("field", ["ts_start", "ts_end"])
def test_a_line_with_an_unusable_time_is_ignored(room, field):
    """A NaN time passed every expiry and replay check, so a stale line could still answer a request."""
    room.say(WAKE, "dana", voice=2)
    entry = {"speaker": "dana", "speaker_id": 2, "ts_start": room.wall[0], "ts_end": room.wall[0] + 1}
    entry[field] = float("nan")
    assert room.c.accept("dummy secret", entry, False, lambda _: None) is None
    assert room.c.pending is not None


def test_a_guest_voice_carrying_someones_name_is_still_another_speaker(room):
    room.say(WAKE, "dana", voice=1)
    assert room.say("cancel", "dana", voice=7) is None  # voice 7 was given Dana's name, but isn't her voice
    assert room.c.pending is not None
    assert [room.say("wrong", "dana", voice=1) for _ in range(2)] == [WRONG, WRONG]
    room.say("cancel", "dana", voice=1)
    room.say(WAKE, "dana", voice=7)
    assert [room.say("wrong", "dana", voice=7) for _ in range(3)] == [WRONG, WRONG, LOCKED]  # its own three


def test_a_wall_clock_jump_does_not_end_a_cooldown(room):
    room.wrong_tries(3)
    room.wall[0] += 3600  # the clock is set an hour ahead while the process runs
    assert room.say(WAKE) == WAIT
    room.mono[0] += 61
    assert "password" in room.say(WAKE)


@pytest.mark.parametrize(
    "wall_at_restart",
    [pytest.param(1_000_000.0 - 86_400, id="clock-set-back-a-day"), pytest.param(1_000_000.0, id="no-jump")],
)
def test_a_saved_cooldown_never_outlasts_a_minute(room, wall_at_restart):
    room.wrong_tries(3)
    room.restart(wall=wall_at_restart)
    assert room.say(WAKE) == WAIT
    room.mono[0] += 61
    assert "password" in room.say(WAKE)


def test_a_far_future_deadline_on_disk_is_held_to_a_minute(room):
    room.state.write_text(
        json.dumps(
            dict(version=1, saved=room.wall[0], counts={}, until={"label:guest": 1e12}, recent=[], everyone_until=0)
        )
    )
    room.restart()
    assert room.say(WAKE) == WAIT
    room.mono[0] += 61
    assert "password" in room.say(WAKE)


@pytest.mark.parametrize(
    ("owner", "caller"),
    [
        pytest.param("+15555550123", "+15555550123", id="both-e164"),
        pytest.param("+1 (555) 555-0123", "+15555550123", id="formatted-owner"),
        pytest.param("+1.555.555.0123", "+15555550123", id="dotted-owner"),
        pytest.param(" +44 20 7946 0958 ", "+442079460958", id="uk-spaced"),
    ],
)
def test_owner_number_matches_however_it_was_typed(owner, caller):
    with cfg.override({"owner": {"name": "Alice", "phone": owner}}):
        assert outbound.is_owner_number(caller)


@pytest.mark.parametrize(
    ("owner", "caller"),
    [
        pytest.param("+15555550123", "+445555550123", id="different-country-code"),
        pytest.param("+15555550123", "+5555550123", id="caller-is-a-suffix"),
        pytest.param("+15555550123", "+155555501234", id="caller-is-longer"),
        pytest.param("+15555550123", "", id="empty-caller"),
        pytest.param("+15555550123", None, id="no-caller"),
        pytest.param("+15555550123", "anonymous", id="garbage-caller"),
        pytest.param("+15555550123", "15555550123", id="caller-without-plus"),
        pytest.param("555-555-0123", "+15555550123", id="owner-without-country-code"),
        pytest.param("15555550123", "+15555550123", id="owner-without-plus"),
        pytest.param("+1 555 CALL NOW", "+15555550123", id="garbage-owner"),
        pytest.param("", "", id="no-owner-phone"),
    ],
)
def test_owner_number_never_matches_a_different_or_unreadable_number(owner, caller):
    with cfg.override({"owner": {"name": "Alice", "phone": owner}}):
        assert not outbound.is_owner_number(caller)


def test_owner_number_needs_an_owner_name():
    with cfg.override({"owner": {"name": "", "phone": "+15555550123"}}):
        assert not outbound.is_owner_number("+15555550123")


@pytest.mark.parametrize(
    ("typed", "expected"),
    [
        pytest.param("+15555550123", "+15555550123", id="e164"),
        pytest.param("+1 (555) 555-0123", "+15555550123", id="formatted"),
        pytest.param("+1234567", None, id="too-short"),
        pytest.param("+1234567890123456", None, id="too-long"),
        pytest.param("+05555550123", None, id="country-code-zero"),
        pytest.param("+1-555-555-0123 ext 4", None, id="extension"),
        pytest.param("++15555550123", None, id="double-plus"),
        pytest.param("1+5555550123", None, id="plus-inside"),
        pytest.param("+44 (0)20 7946 0958", None, id="uk-trunk-zero"),
        pytest.param("+1\u0665\u0665\u0665\u0665\u0665\u0665\u0660\u0661\u0662\u0663", None, id="non-ascii-digits"),
    ],
)
def test_e164_reads_only_a_full_international_number(typed, expected):
    assert numbers.e164(typed) == expected


def test_settings_store_owner_phone_in_e164(tmp_path, monkeypatch):
    path = tmp_path / "pack.toml"
    monkeypatch.setattr(cfg, "local_config", lambda: path)
    cfg.save({"owner.phone": "+1 (555) 555-0123"})
    assert tomllib.loads(path.read_text())["owner"]["phone"] == "+15555550123"


@pytest.mark.parametrize(
    "typed",
    [
        pytest.param("555-555-0123", id="no-country-code"),
        pytest.param("+44 (0)20 7946 0958", id="trunk-zero"),
        pytest.param("call me", id="garbage"),
    ],
)
def test_settings_refuse_an_owner_phone_without_a_country_code(tmp_path, monkeypatch, typed):
    path = tmp_path / "pack.toml"
    monkeypatch.setattr(cfg, "local_config", lambda: path)
    with pytest.raises(cfg.SettingError, match=r"needs a \+ and the country code"):
        cfg.save({"owner.phone": typed})
    assert not path.exists()


@pytest.mark.parametrize("blank", [pytest.param("", id="empty"), pytest.param("  ", id="spaces")])
def test_settings_still_clear_owner_phone(tmp_path, monkeypatch, blank):
    path = tmp_path / "pack.toml"
    path.write_text('[owner]\nphone = "+15555550123"\n')
    monkeypatch.setattr(cfg, "local_config", lambda: path)
    cfg.save({"owner.phone": blank})
    assert "phone" not in tomllib.loads(path.read_text()).get("owner", {})


# One lockout file shared by two processes (the voice service and the phone bridge).
@pytest.fixture
def two_processes(tmp_path):
    clock = [1000.0]
    path = tmp_path / "voice_lockout.json"
    return [voice_auth.Lockouts(path, lambda: clock[0], lambda: clock[0]) for _ in range(2)]


def test_wrong_passwords_from_two_processes_add_up_for_one_speaker(two_processes):
    voice, phone = two_processes
    assert [voice.fail("x"), phone.fail("x"), voice.fail("x")] == [False, False, True]
    assert voice.locked("x") and phone.locked("x")


def test_wrong_passwords_from_two_processes_add_up_for_everyone(two_processes):
    first, second = two_processes
    for n in range(voice_auth.EVERYONE_TRIES // 2):
        first.fail(f"voice:{n}")
        second.fail(f"voice:{n + 10}")
    assert first.locked("voice:99") and second.locked("voice:99")


def test_a_right_password_in_one_process_clears_the_count_in_the_other(two_processes):
    voice, phone = two_processes
    voice.fail("x")
    voice.fail("x")
    phone.succeed("x")
    assert voice.fail("x") is False  # a fresh count: one wrong try, not the third


def test_a_save_that_fails_keeps_counting_in_memory(two_processes, monkeypatch):
    voice, _phone = two_processes
    voice.fail("x")

    def disk_full(*args):
        raise OSError("no space left on device")

    monkeypatch.setattr(voice_auth.os, "replace", disk_full)
    assert [voice.fail("x"), voice.fail("x")] == [False, True]  # never reread the older file and reset


def test_a_wall_clock_jump_while_another_process_saves_keeps_a_known_cooldown(tmp_path):
    wall, mono = [1000.0], [50.0]
    path = tmp_path / "voice_lockout.json"
    voice, phone = (voice_auth.Lockouts(path, lambda: mono[0], lambda: wall[0]) for _ in range(2))
    assert [voice.fail("x") for _ in range(voice_auth.TRIES)][-1] is True
    wall[0] += 3600  # the wall clock jumps an hour; no real time passes
    phone.fail("y")  # the other process reads the cooldown as long over, and saves
    assert voice.locked("x")


# A line the microphone service matched to the password reads "[password]" and carries password: true. The mark
# answers only the ask it was said for.
@pytest.fixture
def asked(tmp_path):
    """Voice 7 asked for something that needs the password; say(text, ts, ...) is one of their lines."""
    clock = [1000.0]
    c = Conversation(make_verifier(tmp_path), lambda: clock[0])

    def say(text, ts, voice=7, **extra):
        clock[0] = max(clock[0], ts + 2)
        entry = {"speaker": f"Speaker {voice}", "speaker_id": voice, "ts_start": ts, "ts_end": ts + 2, **extra}
        return c.accept(text, entry, False, lambda _: None)

    assert "password" in say(WAKE, 1000.0)
    return SimpleNamespace(c=c, say=say)


def test_a_marked_line_said_after_the_ask_answers_it(asked):
    assert "Password accepted" in str(asked.say("[password]", 1005.0, password=True))


@pytest.mark.parametrize(
    "line, reply",
    [
        pytest.param(dict(ts=995.0), WRONG, id="said-before-the-ask"),
        pytest.param(dict(ts=1005.0, voice=9), None, id="another-voice"),
        pytest.param(dict(ts=1100.0), "That request expired. Please ask again.", id="after-the-window"),
        pytest.param(dict(ts=1005.0, password=1), WRONG, id="a-mark-that-is-not-true"),
    ],
)
def test_a_marked_line_answers_only_its_own_ask(asked, line, reply):
    assert asked.say("[password]", **{"password": True, **line}) == reply


def test_one_marked_line_never_answers_two_asks(asked):
    assert "Password accepted" in str(asked.say("[password]", 1005.0, password=True))
    asked.say("cancel", 1008.0)
    assert "password" in asked.say(WAKE, 1010.0)
    copied = asked.say("[password]", 1009.0, password=True)  # a marked line from before this ask, not yet seen
    assert copied == WRONG


def test_without_the_mark_the_blanked_text_is_just_a_wrong_password(asked):
    assert asked.say("[password]", 1005.0) == WRONG


def test_the_pending_ask_is_shared_privately_and_closed_when_it_ends(asked):
    window = voice_auth.password_window(asked.c.verifier)
    asked.c.share_pending()
    assert json.loads(window.read_text()) == {"asked": 1002.0, "expires": 1002.0 + 45}
    assert stat.S_IMODE(os.stat(window).st_mode) == 0o600
    asked.say("cancel", 1003.0)
    asked.c.share_pending()
    assert json.loads(window.read_text()) == {"asked": 1002.0, "expires": 1005.0}  # kept, closed when it ended


@pytest.mark.parametrize(
    "said", ["password123", "Password123.", "password is password123", "the password's password123"]
)
def test_a_password_that_starts_with_password_is_still_recognized(tmp_path, said):
    salt = b"test salt only"
    verifier = tmp_path / "verifier.json"
    digest = hashlib.pbkdf2_hmac("sha256", b"password123", salt, 100_000).hex()
    verifier.write_text(json.dumps(dict(version=1, salt=salt.hex(), iterations=100_000, hash=digest)))
    verifier.chmod(0o600)
    assert voice_auth.said_password(verifier, said)


# Phone and room keep separate budgets, and phone cooldowns escalate.
@pytest.fixture
def clocks(tmp_path):
    """Lockouts on one file with a shared clock; new() is another process's Lockouts, t the monotonic time, and
    wall() the wall clock, which reads very differently from monotonic time (as it does in practice)."""
    t = [1_000_000.0]
    path = tmp_path / "voice_lockout.json"

    def wall():
        return t[0] + 1.7e9

    return SimpleNamespace(new=lambda: voice_auth.Lockouts(path, lambda: t[0], wall), t=t, wall=wall)


def lock(lockouts, who):
    return [lockouts.fail(who) for _ in range(voice_auth.TRIES)][-1]


def cooldown(lockouts, who, clocks):
    """How long who stays locked out from now (in whole seconds of 30 s steps), probing without moving time on."""
    start = clocks.t[0]
    while lockouts.locked(who):
        clocks.t[0] += 30
    left, clocks.t[0] = clocks.t[0] - start, start
    return left


@pytest.mark.parametrize(
    "guesser, spared", [("phone:+15555550199", "voice:1"), ("voice:1", "phone:+15555550199")], ids=["phone", "room"]
)
def test_one_budget_running_out_never_locks_out_the_other(clocks, guesser, spared):
    lockouts = clocks.new()
    prefix = guesser.split(":")[0]
    for n in range(voice_auth.EVERYONE_TRIES):
        lockouts.fail(f"{prefix}:{n}")
    assert lockouts.locked(f"{prefix}:99") and not lockouts.locked(spared)


def test_a_phone_numbers_lockouts_escalate_to_five_then_thirty_minutes(clocks):
    lockouts, who, seen = clocks.new(), "phone:+15555550199", []
    for _ in range(4):
        assert lock(lockouts, who)
        seen.append(cooldown(lockouts, who, clocks))
        clocks.t[0] += seen[-1]
    assert seen == [60, 300, 1800, 1800]


def test_the_room_never_escalates(clocks):
    lockouts, seen = clocks.new(), []
    for _ in range(3):
        lock(lockouts, "voice:1")
        seen.append(cooldown(lockouts, "voice:1", clocks))
        clocks.t[0] += seen[-1]
    assert seen == [60, 60, 60]


def test_rotating_caller_ids_escalate_the_phone_wide_limit(clocks):
    lockouts, seen, n = clocks.new(), [], 0
    for _ in range(3):
        for _ in range(voice_auth.EVERYONE_TRIES):
            lockouts.fail(f"phone:+1555555{n:04d}")  # a new number for every guess
            n += 1
        seen.append(cooldown(lockouts, "phone:+15555559999", clocks))
        clocks.t[0] += seen[-1]
    assert seen == [60, 300, 1800]


@pytest.mark.parametrize("reset", ["right-password", "a-quiet-day"])
def test_the_escalation_starts_over(clocks, reset):
    lockouts, who = clocks.new(), "phone:+15555550199"
    for _ in range(2):
        lock(lockouts, who)
        clocks.t[0] += cooldown(lockouts, who, clocks)
    if reset == "right-password":
        lockouts.succeed(who)
    else:
        clocks.t[0] += voice_auth.PHONE_QUIET_S
    lock(lockouts, who)
    assert cooldown(lockouts, who, clocks) == 60


def test_another_process_sees_the_escalation(clocks):
    who = clocks.new(), "phone:+15555550199"
    bridge, who = who
    lock(bridge, who)
    clocks.t[0] += 60
    other = clocks.new()  # say, the bridge after a restart
    lock(other, who)
    assert cooldown(other, who, clocks) == 300


def test_a_saved_phone_cooldown_never_outlasts_thirty_minutes(clocks):
    path = clocks.new().path
    state = dict(version=2, saved=clocks.wall(), counts={}, until={"phone:+15555550199": clocks.wall() + 10**9},
                 recent={}, everyone_until={}, strikes={})  # fmt: skip
    path.write_text(json.dumps(state))
    lockouts = clocks.new()
    assert cooldown(lockouts, "phone:+15555550199", clocks) == 1800


def test_a_lockout_file_from_before_the_split_holds_for_both_budgets(clocks):
    path = clocks.new().path
    old = dict(version=1, saved=clocks.wall(), counts={}, until={}, recent=[], everyone_until=clocks.wall() + 30)
    path.write_text(json.dumps(old))
    lockouts = clocks.new()
    assert lockouts.locked("voice:1") and lockouts.locked("phone:+15555550199")


def phone_wide(lockouts, first=0):
    """A wrong password from each of EVERYONE_TRIES new numbers: one phone-wide lockout."""
    for n in range(first, first + voice_auth.EVERYONE_TRIES):
        lockouts.fail(f"phone:+1555555{n:04d}")


def test_slow_guessing_from_rotating_numbers_still_locks_the_phone_within_an_hour(clocks):
    lockouts, locked_at = clocks.new(), None
    for n in range(voice_auth.PHONE_HOUR_TRIES):
        clocks.t[0] += 12  # under the per-minute limit, and a new number every guess
        if lockouts.fail(f"phone:+1555555{n:04d}"):
            locked_at = n + 1
    assert locked_at == voice_auth.PHONE_HOUR_TRIES and lockouts.locked("phone:+15555559999")


def test_a_right_password_from_one_number_leaves_the_phone_wide_escalation(clocks):
    lockouts = clocks.new()
    phone_wide(lockouts)
    clocks.t[0] += 60
    lockouts.succeed("phone:+15555550000")
    phone_wide(lockouts, 10)
    assert cooldown(lockouts, "phone:+15555559999", clocks) == 300


def test_the_phone_wide_escalation_starts_over_after_a_quiet_day(clocks):
    lockouts = clocks.new()
    phone_wide(lockouts)
    clocks.t[0] += voice_auth.PHONE_QUIET_S
    phone_wide(lockouts, 10)
    assert cooldown(lockouts, "phone:+15555559999", clocks) == 60


def test_quiet_phone_numbers_are_dropped_so_rotating_ones_cant_grow_the_file(clocks):
    lockouts = clocks.new()
    lockouts.fail("phone:+15555550001")
    lock(lockouts, "phone:+15555550002")
    clocks.t[0] += voice_auth.PHONE_QUIET_S
    lockouts.fail("voice:1")  # any save prunes
    saved = json.loads(lockouts.path.read_text())
    assert (saved["counts"], saved["strikes"], saved["touched"]) == ({"voice:1": 1}, {}, {})


def test_a_strike_saved_from_a_clock_that_ran_ahead_still_ages_out(clocks):
    path = clocks.new().path
    state = dict(version=2, saved=clocks.wall(), counts={}, until={}, recent={}, everyone_until={},
                 strikes={"phone": [5, clocks.wall() + 10**8]})  # fmt: skip
    path.write_text(json.dumps(state))
    lockouts = clocks.new()
    lockouts.locked("voice:1")  # the next check after the clock was corrected reads it: the strike counts as now
    clocks.t[0] += voice_auth.PHONE_QUIET_S
    phone_wide(lockouts)
    assert cooldown(lockouts, "phone:+15555559999", clocks) == 60


def test_an_unreadable_lockout_file_holds_up_the_phone_too(clocks):
    clocks.new().path.write_text("{bad")
    assert clocks.new().locked("phone:+15555550199")


def test_a_wall_clock_jump_keeps_a_known_everyone_lockout(tmp_path):
    wall, mono = [1.7e9], [50.0]
    path = tmp_path / "voice_lockout.json"
    first, second = (voice_auth.Lockouts(path, lambda: mono[0], lambda: wall[0]) for _ in range(2))
    phone_wide(first)
    wall[0] += 3600  # the wall clock jumps an hour; no real time passes
    second.fail("voice:1")  # the other process saves, reading the phone-wide lockout as long over
    assert first.locked("phone:+15555559999")


def test_recent_tries_in_a_file_from_before_the_split_count_for_the_phone_too(clocks):
    path = clocks.new().path
    tries = [clocks.wall() - 5] * (voice_auth.EVERYONE_TRIES - 1)
    path.write_text(
        json.dumps(dict(version=1, saved=clocks.wall(), counts={}, until={}, recent=tries, everyone_until=0))
    )
    lockouts = clocks.new()
    assert lockouts.fail("phone:+15555550199") is True  # the sixth wrong try within the minute


# ---- additional lockout and guest-count cases ----


def test_another_guest_hears_why_there_is_no_answer_while_a_request_waits(room):
    room.say(WAKE, "dana", voice=2)
    assert room.say(WAKE, "erin", voice=3) == voice_auth.WAITING  # not silence
    assert room.say("lovely weather", "erin", voice=3) is None  # talk without the wake phrase stays quiet
    assert room.c.pending is not None and room.c.pending["speaker"] == "voice:2"  # and it can't take the request


def test_asking_the_same_request_again_keeps_its_deadline(room):
    room.say(WAKE, "dana", voice=2)
    room.wait(40)
    assert "password" in room.say(WAKE, "dana", voice=2)  # the same request again, 40 s in
    room.wait(10)  # 50 s after it was first asked
    assert room.say("dummy secret", "dana", voice=2) == "That request expired. Please ask again."


def test_a_different_request_gets_its_own_time(room):
    room.say(WAKE, "dana", voice=2)
    room.wait(40)
    room.say("Hey Bob, open my email", "dana", voice=2)
    room.wait(10)
    assert isinstance(room.say("dummy secret", "dana", voice=2), Grant)  # in time for the second request


def test_replies_keep_a_request_open_only_so_long(room):
    room.say(WAKE, "dana", voice=2)
    p = room.c.pending
    assert p is not None
    for _ in range(5):  # the assistant keeps answering
        room.wait(40)
        room.c.spoken_reply("Password not accepted.", room.wall[0], room.wall[0] + 2)
    assert p["expires"] <= p["opened"] + voice_auth.HOLD_S


def test_a_long_line_that_runs_well_past_the_deadline_has_expired(room):
    room.say(WAKE, "dana", voice=2)
    deadline = room.c.pending["expires"]
    line = {"speaker": "dana", "speaker_id": 2, "ts_start": deadline - 1, "ts_end": deadline + voice_auth.LATE_S + 5}
    room.wall[0] = deadline + 21
    assert room.c.accept("dummy secret", line, False, lambda _: None) == "That request expired. Please ask again."


def test_a_yes_said_after_the_assistant_said_yes_is_heard(room):
    room.say("Hey Bob, delete the shopping list", "dana", voice=2)  # a guest: the password, then a confirmation
    assert "Confirm this exact request" in room.say("dummy secret", "dana", voice=2)
    room.c.spoken_reply("Yes?", room.wall[0], room.wall[0] + 1)
    echo = {"speaker": "dana", "speaker_id": 2, "ts_start": room.wall[0] + 1.5, "ts_end": room.wall[0] + 2}
    assert room.c.accept("yes", echo, False, lambda _: None) is None  # our own "Yes?" heard back
    room.wait(8)
    later = {"speaker": "dana", "speaker_id": 2, "ts_start": room.wall[0], "ts_end": room.wall[0] + 1}
    assert isinstance(room.c.accept("yes", later, False, lambda _: None), Grant)  # the confirmation, granted


def test_a_voice_id_that_isnt_a_real_int_is_no_voice(tmp_path, monkeypatch):
    from liberational.assistant import voice

    write_voices(tmp_path, {1: None})
    monkeypatch.setattr(voice, "STATE_DIR", tmp_path)
    e = voice.canonical_entry({"speaker": "guest", "speaker_id": True})
    assert e["speaker_id"] is True and e["owner_voice"] is False  # not voice 1
    assert voice_auth.speaker_key(e) == "label:guest"


def window(room):
    return json.loads(voice_auth.password_window(room.verifier).read_text())


def test_the_password_stays_blanked_45_s_after_we_last_asked_for_it_whatever_the_cap(room):
    prompt = "Please say your password for this request, or say cancel."
    room.say(WAKE, "dana", voice=2)
    room.c.spoken_reply(prompt, room.wall[0], room.wall[0] + 3)
    for _ in range(3):  # asked again at 40, 80, and 119 s, each prompt keeping it open up to the cap
        room.wait(39.5)
        assert room.say(WAKE, "dana", voice=2) == prompt
        room.c.spoken_reply(prompt, room.wall[0], room.wall[0] + 3)
    room.c.share_pending()
    assert room.c.pending is not None and room.c.pending["expires"] <= room.c.pending["opened"] + voice_auth.HOLD_S
    asked_until = room.wall[0] + 3 + 45  # past the capped deadline
    assert window(room)["expires"] >= asked_until  # a password said in time is blanked
    room.wait(60)
    room.say("dummy secret", "dana", voice=2)  # too late to answer: the request ends
    room.c.share_pending()
    assert window(room)["expires"] >= asked_until  # ending it never closes the window early


def test_a_request_ended_by_a_merge_tells_its_asker_and_keeps_its_password_blanked(room):
    room.say(WAKE, "dana", voice=5)
    room.c.spoken_reply("Please say your password for this request, or say cancel.", room.wall[0], room.wall[0] + 3)
    room.c.share_pending()
    room.c.follow = lambda key: "voice:3" if key == "voice:5" else key  # voice 5 merged into voice 3
    assert room.say("dummy secret", "dana", voice=3) == "That request ended. Please ask again."
    room.c.share_pending()
    assert window(room)["expires"] >= room.wall[0] + 3 + 40


def test_rewording_or_asking_something_else_buys_no_more_time(room):
    room.say(WAKE, "dana", voice=2)
    for request in ("Hey Bob, open my email", "Hey Bob, reset my private computer please", "Hey Bob, open my email"):
        room.wait(40)
        reply = room.say(request, "dana", voice=2)
    assert reply == "That request expired. Please ask again."  # 120 s in, however it's put


def test_another_guest_hears_about_the_wait_once(room):
    room.say(WAKE, "dana", voice=2)
    assert room.say(WAKE, "erin", voice=3) == voice_auth.WAITING
    assert room.say(WAKE, "erin", voice=3) is None  # not again: our reply would talk over the asker
