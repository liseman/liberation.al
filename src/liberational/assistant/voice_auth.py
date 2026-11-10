"""Local single-request authorization. No network, transcript, or candidate logging."""

import contextlib
import fcntl
import hashlib
import hmac
import json
import logging
import math
import os
import pathlib
import re
import stat
import tempfile
import time
from collections.abc import Callable, Generator, Mapping
from dataclasses import dataclass
from typing import Any, TypedDict

from liberational import config as cfg

QUESTION = re.compile(r"^(?:what|when|where|why|how|who|which|is|are|do|does|did|can|could|would|will)\b", re.I)
REQUEST = re.compile(
    r"^(?:please\s+)?(?:ask|tell|show|check|explain|help|call|reset|wipe|erase|format|reboot|shutdown|uninstall|upgrade|publish|purchase|buy|transfer|restart|stop|start|install|update|change|set|delete|remove|run|open|send|write|read|find|search|remember|forget|turn|make|enable|disable|list)\b",
    re.I,
)
CANCEL = {"cancel", "cancel that", "never mind", "nevermind", "stop"}
CONFIRM = {"yes", "yes confirm", "confirm", "yes do it"}
# Local time/date route; other authorized requests use the gateway.
PUBLIC = re.compile(
    r"^(?:what (?:time|day|date) is it|what is (?:the time|today.s date)|what.s (?:the time|today.s date))[?.! ]*$",
    re.I,
)


def _phrase(words: str) -> str:
    """Match a phrase by its words, with optional commas between them ("hey, bob")."""
    return r",?\s+".join(re.escape(w) for w in re.findall(r"\w+", words))


def wake_regex() -> str:
    """Any configured wake phrase ("hey bob", "hey, bob")."""
    return "(?:" + "|".join(_phrase(w) for w in sorted(cfg.wake_words(), key=len, reverse=True)) + ")"


def address_regex() -> str:
    """A wake phrase or the bare assistant name ("bob, ...")."""
    phrases = sorted(cfg.wake_words() + [cfg.assistant_name().lower()], key=len, reverse=True)
    return "(?:" + "|".join(_phrase(w) for w in phrases) + ")"


def wake_phrase() -> str:
    """The first wake phrase as spoken in prompts and hints ("Hey Bob")."""
    return cfg.wake_words()[0].title()


def camera_read(req: str, camera: Callable[[str], object]) -> bool:
    """A read-only camera question, the one route that needs no authorization. Strictly a read-only handler;
    never forwards mixed actions to tools."""
    return bool(
        camera(req)
        and re.match(r"^(?:what|how|show|describe|tell)\b", req, re.I)
        and not re.search(
            r"\b(?:delete|remove|restart|stop|start|install|update|change|set|send|call|password|medical|bank|private|secret|and then)\b",
            req,
            re.I,
        )
    )


def asks_about_the_past(req: str) -> bool:
    """Whether a camera request asks about the past rather than what the cameras show now: a history question
    (memory/history_qa.py is_history_question, which voice.handle answers from the transcript and memory), or one
    the camera history reads a time window for (vision/camera_history.py parse), or one that parse refuses as a
    past time it can't read. Both, because their wordings differ: "camzero" is a camera to parse but not to the
    history classifier."""
    from liberational.memory import history_qa
    from liberational.vision import camera_history

    if history_qa.is_history_question(req):
        return True
    try:
        return camera_history.parse(req)[1] is not None
    except ValueError:
        return True


def request_policy(request: str, owner: bool) -> tuple[bool, bool]:
    """Lexical intent/risk heuristic, not a perfect semantic security boundary.
    Owner (speaker-ID matched the configured owner): no password, no confirmation, no pending state."""
    if owner:
        return False, False
    sensitive = bool(
        re.search(
            r"\b(?:passwords?|passcodes?|credentials?|secrets?|private|medical|health records?|"
            r"bank(?:ing)?|financial|credit cards?|accounts?|balances?|payments?|transfer money|"
            r"authentication|authorization|permissions?|security|firewall|ssh keys?|api keys?|"
            r"access tokens?|two.factor|2fa)\b",
            request,
            re.I,
        )
    )
    informational = bool(
        re.match(
            r"^(?:explain\b|tell me (?:about|why|how)\b|what\b|why\b|when\b|where\b|"
            r"which\b|who\b|how (?:do|does|did|can|could|would|should|to|is|are)\b|"
            r"(?:can|could|would) (?:i|we|one)\b|is\b|are\b|does\b|do i\b)",
            request,
            re.I,
        )
    )
    mixed_action = bool(
        re.search(
            r"(?:\b(?:and|then|also)|;)\s+(?:then\s+)?(?:please\s+)?"
            r"(?:delete|remove|wipe|reset|install|update|restart|change|set|disable|enable|send|call)\b",
            request,
            re.I,
        )
    )
    major = (not informational or mixed_action) and bool(
        re.search(
            r"\b(?:delete|remove|erase|wipe|format|reset|uninstall|install|upgrade|update|"
            r"restart|reboot|shutdown|shut down|network|router|dns|vpn|firewall|security|"
            r"permissions?|authentication|authorization|passwords?|credentials?|"
            r"call|send|publish|purchase|buy|transfer)\b",
            request,
            re.I,
        )
    )
    return (not owner or sensitive), bool(major)


@dataclass(frozen=True)
class Grant:
    request: str
    route: str
    authenticated: bool
    confirmed: bool
    needs_password: bool = True
    needs_confirmation: bool = True
    owner: bool = False
    # Whether the line called the assistant by name. The verified owner may skip the wake phrase for a question, so
    # a question they ask someone else in the room reads as one too: a failed answer to it is never spoken.
    addressed: bool = True

    def authorized_for(self, request: str) -> bool:
        if self.request != request:
            return False
        if self.route == "camera":
            return True  # read-only route only
        return (
            request_policy(request, self.owner) == (self.needs_password, self.needs_confirmation)
            and (not self.needs_password or self.authenticated)
            and (not self.needs_confirmation or self.confirmed)
        )


log = logging.getLogger(__name__)


def speaker_key(entry: Mapping[str, Any]) -> str:
    """Who is speaking, for a pending request, its cancel, and wrong-password counts: the voice id (after merges, as
    voice.canonical_entry follows them) when the line has one, else its speaker label. Not the name alone: a name is
    a person's, and a guest's voice can carry it through an introduction or a mistaken merge."""
    sid = entry.get("speaker_id")
    if isinstance(sid, int) and not isinstance(sid, bool):
        return f"voice:{sid}"
    return "label:" + str(entry.get("speaker") or "unknown").lower()  # prefixed, so no label can pose as a voice


def anonymous(entry: Mapping[str, Any]) -> bool:
    """Whether a line says nothing about who spoke: no voice id, and no person's name, as the microphone service
    writes a clip too short to match any voice ("unknown"). All such lines would share one key, so different people
    would count as one speaker: such a line never answers, cancels, or makes a pending request."""
    from liberational.people import registry

    sid = entry.get("speaker_id")
    return not (isinstance(sid, int) and not isinstance(sid, bool)) and not registry.is_name(entry.get("speaker"))


UNHEARD = "I didn't catch who was speaking. Please say that again, a little longer."
WAITING = "Someone else's request is waiting. Please ask again in a minute."  # another guest's wake, mid-request
HOLD_S = 120.0  # longest a request stays open from when it was first asked, however often it's asked or answered
ECHO_S = 3.0  # a line captured this soon after our reply ends, with its words, is that reply heard back
LATE_S = 10.0  # how far past a request's deadline a line begun in time may run and still answer it


TRIES = 3  # wrong passwords one speaker gets before a cooldown
EVERYONE_TRIES = 6  # wrong passwords from all speakers within WINDOW_S before everyone waits
COOLDOWN_S = 60.0
WINDOW_S = 60.0
# Phone and room keep separate budgets: wrong passwords over the phone ("phone:<E.164>" speakers) count toward
# their own everyone-limit, never the room's, and the reverse. Phone cooldowns escalate, per number and for the
# phone-wide limit (so rotating caller IDs doesn't reset them): 1 minute, then 5, then 30 for every lockout after.
# The escalation starts over after that number's right password, or after PHONE_QUIET_S with no lockout. Guessing
# slowly from a new number every couple of tries stays under both the per-number and the per-minute limit, so the
# phone also has an hourly one: PHONE_HOUR_TRIES wrong passwords from any numbers within an hour lock the phone, and
# escalate like the per-minute limit.
PHONE_STEPS = (60.0, 300.0, 1800.0)
PHONE_QUIET_S = 24 * 3600.0
PHONE_HOUR_TRIES = 20
PHONE_HOUR_S = 3600.0
PHONE = "phone"  # the phone's budget; the room's is ROOM
ROOM = "room"


def budget(who: str) -> str:
    """Which everyone-limit a speaker's wrong passwords count toward: the phone's or the room's."""
    return PHONE if who.startswith("phone:") else ROOM


class Lockouts:
    """Wrong-password counts and cooldowns, per speaker, kept in a file so a restart doesn't clear them.

    Each speaker gets TRIES wrong passwords, then a COOLDOWN_S wait, so a guest can't use up someone else's tries.
    A new unknown voice is a new speaker label, so EVERYONE_TRIES wrong passwords from anyone within WINDOW_S make
    everyone wait: sounding like someone new doesn't buy more guesses. Only a correct password or a cooldown resets a
    speaker's count.

    Timing in the process is monotonic, so a wall-clock change doesn't end a cooldown. The file holds wall-clock
    deadlines (the only time that survives a restart), read back as at most COOLDOWN_S from when they were written:
    a clock set back can't stretch a cooldown, and one set forward while the process was down can shorten it. A
    missing file is a fresh start; one that can't be read makes everyone wait COOLDOWN_S, never resets to zero.

    The file is shared: the voice service and the phone bridge are separate processes. Each call takes an
    exclusive lock on <file>.lock and reads the file again first, so the read, the change, and the save happen as one
    step and no process misses or overwrites another's wrong passwords. When the file still holds what this
    instance last wrote, the in-memory state (on monotonic time) is kept rather than read back from wall-clock
    deadlines. When a save fails, this instance keeps counting in memory instead of rereading the older file, so a
    full disk can't reset anyone's count.
    """

    def __init__(
        self,
        path: pathlib.Path,
        monotonic: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
    ) -> None:
        self.path, self.monotonic, self.wall = path, monotonic, wall
        self.written: str | None = None  # the file's text as this instance last saved it
        self.unsaved = False  # the last save failed: the state in memory is newer than the file
        self.counts: dict[str, int] = {}
        self.until: dict[str, float] = {}  # speaker: monotonic end of their cooldown
        self.recent: dict[str, list[float]] = {}  # budget: monotonic times of its wrong passwords, within WINDOW_S
        self.everyone_until: dict[str, float] = {}  # budget: monotonic end of its everyone-cooldown
        self.strikes: dict[str, tuple[int, float]] = {}  # phone speaker, or PHONE: (lockouts so far, wall time of last)
        self.hourly: list[float] = []  # wall times of the phone's wrong passwords within PHONE_HOUR_S
        self.touched: dict[str, float] = {}  # phone speaker: wall time of their last wrong password

    @contextlib.contextmanager
    def _shared(self) -> Generator[None]:
        """Hold the file's lock and bring the state up to date with it, so this call sees every process's changes."""
        try:
            fd = os.open(self.path.with_name(self.path.name + ".lock"), os.O_RDWR | os.O_CREAT, 0o600)
        except OSError as ex:  # the state dir isn't writable: no other process can be saving either
            log.warning("voice lockout lock for %s not taken (%s)", self.path, ex)
            fd = None
        try:
            if fd is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            self._load()
            yield
        finally:
            if fd is not None:
                os.close(fd)

    def locked(self, who: str) -> bool:
        with self._shared():
            now = self.monotonic()
            return now < self.everyone_until.get(budget(who), 0.0) or now < self.until.get(who, 0.0)

    def fail(self, who: str) -> bool:
        """Count a wrong password; True when it starts a cooldown (for this speaker, or for everyone)."""
        with self._shared():
            return self._fail(who)

    def _fail(self, who: str) -> bool:
        now, wall, b = self.monotonic(), self.wall(), budget(who)
        self.recent[b] = [t for t in self.recent.get(b, []) if now - t < WINDOW_S] + [now]
        self.counts[who] = self.counts.get(who, 0) + 1
        locked = False
        if self.counts[who] >= TRIES:
            self.counts[who], self.until[who], locked = 0, now + self._cooldown(who, b, wall), True
        if len(self.recent[b]) >= EVERYONE_TRIES:
            self.recent[b], self.everyone_until[b], locked = [], now + self._cooldown(b, b, wall), True
        if b == PHONE:
            self.touched[who] = wall
            self.hourly = [t for t in self.hourly if wall - t < PHONE_HOUR_S] + [wall]
            if len(self.hourly) >= PHONE_HOUR_TRIES:
                end = now + self._cooldown(b, b, wall)
                self.hourly, self.everyone_until[b], locked = [], max(end, self.everyone_until.get(b, 0.0)), True
        self._save()
        return locked

    def _cooldown(self, key: str, b: str, wall: float) -> float:
        """How long a lockout of key (a speaker, or a whole budget) lasts: a minute in the room; on the phone, the next
        step of PHONE_STEPS, counting key's earlier lockouts unless PHONE_QUIET_S passed since the last one."""
        if b != PHONE:
            return COOLDOWN_S
        n, last = self.strikes.get(key, (0, 0.0))
        if wall - last >= PHONE_QUIET_S:
            n = 0
        self.strikes[key] = (n + 1, wall)
        return PHONE_STEPS[min(n, len(PHONE_STEPS) - 1)]

    def follow(self, resolve: Callable[[str], str]) -> None:
        """Move each speaker's count and cooldown to where resolve says that speaker is now (a voice merged into
        another), so a merge never buys fresh tries: counts add up, and the later cooldown holds."""
        with self._shared():
            self._follow(resolve)

    def _follow(self, resolve: Callable[[str], str]) -> None:
        moves = {k: resolve(k) for k in {*self.counts, *self.until}}
        if all(k == v for k, v in moves.items()):
            return
        now = self.monotonic()
        counts: dict[str, int] = {}
        until: dict[str, float] = {}
        for k, v in self.until.items():
            until[moves[k]] = max(until.get(moves[k], 0.0), v)
        for k, n in self.counts.items():
            counts[moves[k]] = counts.get(moves[k], 0) + n
        for k, n in list(counts.items()):
            if n >= TRIES:
                counts[k], until[k] = 0, max(until.get(k, 0.0), now + COOLDOWN_S)
        self.counts = {k: n for k, n in counts.items() if n}
        self.until = until
        self._save()

    def succeed(self, who: str) -> None:
        """A right password: the speaker's count starts over, and so does their phone escalation (not the
        phone-wide one: someone else may still be guessing)."""
        with self._shared():
            if self.counts.pop(who, 0) | (self.strikes.pop(who, None) is not None) | (who in self.touched):
                self.touched.pop(who, None)
                self._save()

    def _load(self) -> None:
        if self.unsaved:
            return  # newer than the file: keep counting in memory, and save it next time
        mono, wall = self.monotonic(), self.wall()
        try:
            text = self.path.read_text()
        except FileNotFoundError:
            if self.written is not None:  # removed since this instance saved it: a fresh start
                self.counts, self.until, self.recent, self.everyone_until, self.strikes = {}, {}, {}, {}, {}
                self.hourly, self.touched, self.written = [], {}, None
            return
        except Exception as ex:
            self._fail_closed(mono, type(ex).__name__)
            return
        if text == self.written:
            return  # no other process has changed it since this instance saved it
        known, known_everyone = self.until, self.everyone_until
        try:
            self._parse(text, mono, wall)
            self.written = text
            # A wall-clock jump can make another process read a cooldown this one still times as running, and save it
            # as over. A cooldown only ends with time, so keep the later end of each one this instance knows.
            for who, end in known.items():
                if end > max(mono, self.until.get(who, 0.0)):
                    self.until[who] = end
            for b, end in known_everyone.items():
                if end > self.everyone_until.get(b, 0.0):
                    self.everyone_until[b] = end
        except Exception as ex:  # unreadable or malformed, however it fails (bad bytes, deep nesting): fail closed
            self._fail_closed(mono, type(ex).__name__)

    def _parse(self, text: str, mono: float, wall: float) -> None:
        def finite(x: object) -> float:
            if isinstance(x, bool) or not isinstance(x, (int, float)):
                raise ValueError("not a number")
            v = float(x)  # a huge int raises OverflowError, caught by _load
            if not math.isfinite(v):
                raise ValueError("not a finite number")
            return v

        def no_constants(name: str) -> float:
            raise ValueError(name)  # NaN and Infinity, which json.loads takes by default

        d = json.loads(text, parse_constant=no_constants)
        if d["version"] not in (1, 2):
            raise ValueError("version")
        saved = finite(d["saved"])
        elapsed = max(0.0, wall - saved)  # a clock set back reads as no time passed

        def left(deadline: object, b: str) -> float:
            """Time left of a saved cooldown, never more than that budget's longest (so a clock set back can't
            stretch it)."""
            longest = PHONE_STEPS[-1] if b == PHONE else COOLDOWN_S
            return min(longest, max(0.0, finite(deadline) - saved - elapsed))

        counts = {str(k): int(v) for k, v in d["counts"].items()}
        if any(not 0 <= v < TRIES for v in counts.values()):
            raise ValueError("count")
        until = {str(k): mono + left(v, budget(str(k))) for k, v in d["until"].items()}
        if d["version"] == 1:  # one budget for everyone in v1: it holds for both now
            recent = {b: d["recent"] for b in (PHONE, ROOM)}
            everyone_at = {b: d["everyone_until"] for b in (PHONE, ROOM)}
            strikes: dict[str, tuple[int, float]] = {}
        else:
            recent = {str(b): list(ts) for b, ts in d["recent"].items()}
            everyone_at = {str(b): t for b, t in d["everyone_until"].items()}
            # A time ahead of now (a clock that ran fast, then was corrected) counts as now, so it still ages out.
            strikes = {str(k): (int(n), min(finite(t), wall)) for k, (n, t) in d["strikes"].items()}
            if any(b not in (PHONE, ROOM) for b in (*recent, *everyone_at)) or any(n < 1 for n, _ in strikes.values()):
                raise ValueError("budget")
            self.hourly = [min(finite(t), wall) for t in d.get("hourly", [])]
            self.touched = {str(k): min(finite(t), wall) for k, t in d.get("touched", {}).items()}
        self.counts = {k: v for k, v in counts.items() if v}
        self.until = {k: v for k, v in until.items() if v > mono}
        self.recent = {
            b: [mono - a for a in (max(0.0, saved - finite(t)) + elapsed for t in ts) if a < WINDOW_S]
            for b, ts in recent.items()
        }
        self.everyone_until = {b: mono + left(t, b) for b, t in everyone_at.items()}
        self.strikes = strikes

    def _fail_closed(self, mono: float, why: str) -> None:
        log.warning(
            "voice lockout state %s is unreadable (%s): everyone waits a minute before a password", self.path, why
        )
        self.counts, self.until, self.recent, self.strikes, self.hourly, self.touched = {}, {}, {}, {}, [], {}
        self.everyone_until = {PHONE: mono + COOLDOWN_S, ROOM: mono + COOLDOWN_S}
        self._save()  # replace it with a readable file, so the next restart doesn't lock everyone again

    def _save(self) -> None:
        mono, wall = self.monotonic(), self.wall()

        def at(t: float) -> float:
            return wall + (t - mono)

        # Phone numbers are the caller's choice, so their entries are dropped once quiet for PHONE_QUIET_S (an old
        # strike would start over by then anyway): a guesser rotating numbers can't grow the file without end.
        quiet = {k for k, t in self.touched.items() if wall - t >= PHONE_QUIET_S}
        self.counts = {k: n for k, n in self.counts.items() if k not in quiet}
        self.touched = {k: t for k, t in self.touched.items() if k not in quiet}
        self.strikes = {k: v for k, v in self.strikes.items() if wall - v[1] < PHONE_QUIET_S}
        self.hourly = [t for t in self.hourly if wall - t < PHONE_HOUR_S]
        state = dict(
            version=2,
            saved=wall,
            counts=self.counts,
            until={k: at(v) for k, v in self.until.items() if v > mono},
            recent={b: [at(t) for t in ts] for b, ts in self.recent.items()},
            everyone_until={b: at(t) if t > mono else 0.0 for b, t in self.everyone_until.items()},
            strikes={k: list(v) for k, v in self.strikes.items()},
            hourly=self.hourly,
            touched=self.touched,
        )
        text = json.dumps(state)
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(text)
            os.replace(tmp, self.path)
            self.written, self.unsaved = text, False
        except OSError as ex:
            self.unsaved = True
            log.warning("voice lockout state %s not saved (%s); it holds until a restart", self.path, ex)


class Pending(TypedDict):
    """A request waiting for its password or confirmation."""

    request: str
    route: str
    speaker: str
    asked: float  # when the request ended (wall clock): only a line said after it can answer it
    expires: float
    opened: float  # when this request was first asked (wall clock): replies and asking again extend it only so far
    stage: str  # "password" or "confirm"
    authenticated: bool
    needs_password: bool
    needs_confirmation: bool
    owner: bool
    addressed: bool  # Grant.addressed


def check_password(verifier: pathlib.Path, candidate: str) -> bool:
    """Whether candidate is the voice password, by the verifier file (a private PBKDF2 hash, set up by hand)."""
    try:
        if verifier.stat().st_mode & 0o077:
            return False
        d = json.loads(verifier.read_text())
        if d["version"] != 1 or not 100000 <= d["iterations"] <= 2000000:
            return False
        candidate = re.sub(r"^(?:my )?password (?:is )?", "", candidate.strip(), flags=re.I).rstrip(".!?").strip()
        value = hashlib.pbkdf2_hmac("sha256", candidate.encode(), bytes.fromhex(d["salt"]), d["iterations"]).hex()
        return hmac.compare_digest(value, d["hash"])
    except Exception:
        return False


# A spoken password is never written down as said: the line that matches it reads PASSWORD_SHOWN in the
# transcript and the logs, in person and on the phone. Only an actual match is blanked, never text that merely looks
# secret. Speech recognition capitalizes the first word, adds an end mark, and leads in several ways ("the
# password's ..."), so those are read past before the hash.
PASSWORD_SHOWN = "[password]"
_LEAD = r"^\W*(?:(?:um|uh|so|ok|okay)\W+)*(?:(?:the|my)\s+)?password\b"
PASSWORD_SAID = re.compile(_LEAD, re.I)
PASSWORD_LEAD = re.compile(_LEAD + r"(?:'s|\s+is|\s*[:,])?\s*", re.I)


def said_password(verifier: pathlib.Path, text: str) -> bool:
    """Whether text says the voice password, however speech recognition wrote it. A slow hash: call it only for a
    line that may be the password (may_be_password)."""
    whole = text.strip().rstrip(".!?").strip()  # a password can itself start with "password"
    said = PASSWORD_LEAD.sub("", whole).strip()
    return any(check_password(verifier, c) for c in dict.fromkeys((said, said.casefold(), whole)) if c)


def password_window(verifier: pathlib.Path) -> pathlib.Path:
    """When a password may be said (the asked and expires wall-clock times of the latest password ask), shared by the
    voice service (Conversation.share_pending) with the microphone service, beside the verifier."""
    return verifier.with_name("voice_password_window.json")


def may_be_password(verifier: pathlib.Path, entry: Mapping[str, Any], text: str) -> bool:
    """Whether a heard line is worth checking against the password: it leads with "password", or it was captured
    while a password was awaited (password_window), however late it is transcribed. Any speaker's line is checked
    then, since the microphone service may hear the asker as another voice; only a match is blanked, and
    Conversation.accept still takes the mark only from the asker."""
    if PASSWORD_SAID.match(text):
        return True
    window = password_window(verifier)
    try:
        st = window.lstat()
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid() or st.st_mode & 0o077:
            return False  # a link, someone else's file, or one others could write: no pending ask
        w = json.loads(window.read_text())
        return float(w["asked"]) <= float(entry.get("ts_start") or 0) <= float(w["expires"])
    except Exception:
        return False


class Conversation:
    def __init__(
        self,
        verifier: pathlib.Path,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] | None = None,
        follow: Callable[[str], str] | None = None,
    ) -> None:
        """clock is wall time (compared with transcript timestamps); cooldowns run on monotonic time. A test that
        passes only a clock drives both with it. Lockouts are kept beside the verifier, in voice_lockout.json.
        follow maps a speaker key to the one it is now (voice.follow_voice: a voice merged into another), so the
        lockouts stay with the voice through a merge; a pending request ends there instead."""
        self.verifier, self.clock, self.follow = verifier, clock, follow
        mono = monotonic or (time.monotonic if clock is time.time else clock)
        self.lockouts = Lockouts(verifier.with_name("voice_lockout.json"), mono, clock)
        self.pending: Pending | None = None
        self.suppress_until: float = 0
        self.spoken: list[tuple[str, float]] = []
        self.windows: list[tuple[float, float]] = []
        self.seen: dict[tuple[Any, str], float] = {}
        self.shared: tuple[float, float] | None = None  # the (asked, expires) last shared by share_pending
        # The password is blanked until 45 s after we last asked for it, whatever the request's own deadline: the
        # cap on how long a request stays open (HOLD_S) must never leave a password said in time in plain text.
        self.blank_until: float = 0.0
        self.told: set[str] = set()  # speakers told "someone else's request is waiting" for this request

    def spoken_reply(self, text: str, start: float, end: float) -> None:
        self.suppress_until = end
        self.windows.append((start, end))
        self.windows = self.windows[-16:]
        if self.pending:  # time to answer what was just said, but never past HOLD_S from the first ask
            p = self.pending
            p["expires"] = max(p["expires"], min(end + 45, p["opened"] + HOLD_S))
            if p["stage"] == "password":
                self.blank_until = max(self.blank_until, end + 45)
        self.spoken.append((self.normal(text), end))
        self.spoken = self.spoken[-16:]

    @staticmethod
    def normal(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

    def verify(self, candidate: str) -> bool:
        return check_password(self.verifier, candidate)

    def share_pending(self) -> None:
        """Tell the microphone service when a password may be said (password_window): from the end of the request to
        its deadline while one is awaited. When the ask ends, the window is kept, closed at that moment rather than
        removed, so a copy of the password that another microphone transcribes late is still blanked."""
        window = password_window(self.verifier)
        p = self.pending
        if p is not None and p["stage"] == "password":
            span = self.shared = (p["asked"], max(p["expires"], self.blank_until))
        elif self.shared is not None:  # closed now, but never before 45 s after the last ask for the password
            span, self.shared = (self.shared[0], min(self.shared[1], max(self.clock(), self.blank_until))), None
        else:
            return
        try:
            fd, tmp = tempfile.mkstemp(dir=window.parent, prefix=".voice_password_window.")  # private, never followed
            with os.fdopen(fd, "w") as f:
                json.dump({"asked": span[0], "expires": span[1]}, f)
            os.replace(tmp, window)
        except OSError as ex:
            log.warning("password window %s not shared (%s)", window, ex)

    def accept(
        self, text: str, entry: Mapping[str, Any], owner: bool, camera: Callable[[str], object]
    ) -> str | Grant | None:
        from liberational.speech.playback_guard import overlaps

        if entry.get("source") == "tts" or overlaps(entry.get("ts_start"), entry.get("ts_end")):
            return None
        now = self.clock()
        norm = self.normal(text)
        if self.follow is not None:
            self.lockouts.follow(self.follow)
            # A merge may join the asker's voice to someone else's (a guest into the owner's): the pending request
            # ends rather than passing to whoever the surviving voice is. Its lockouts still follow the voice.
            if self.pending and (moved := self.follow(self.pending["speaker"])) != self.pending["speaker"]:
                self.pending = None
                self.share_pending()  # its password stays blanked as long as it would have been
            else:
                moved = None
        else:
            moved = None
        # Timing uses capture start, not delayed ASR arrival. No timestamp in tests/manual replay.
        ts = entry.get("ts_start")
        if any(
            v is not None and (isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v))
            for v in (ts, entry.get("ts_end"))
        ):
            return None  # a time no comparison can use (NaN) would pass every expiry and replay check
        if ts is not None and (
            any(float(ts) <= z and float(entry.get("ts_end", ts)) >= a for a, z in self.windows)
            or now - float(entry.get("ts_end", ts)) > 45
        ):
            return None
        # Our own reply heard back: the same words, captured as it ended (by capture time when the line has one, so
        # "yes" said a while after the assistant said "Yes?" is the person's), else arriving within 120 s.
        if any(
            norm == spoken and (float(ts) <= end + ECHO_S if ts is not None else now - end < 120)
            for spoken, end in self.spoken
        ):
            return None
        if ts is not None:
            key = (ts, norm)
            if key in self.seen or any(n == norm and abs(float(t) - float(ts)) < 2 for t, n in self.seen):
                return None
            self.seen = {k: v for k, v in self.seen.items() if now - v < 180}
            self.seen[key] = now
        identity = speaker_key(entry)
        anon = anonymous(entry)
        again: Pending | None = None  # the speaker's own request a fresh wake phrase replaces, if any
        if moved is not None and identity == moved and not anon:  # only the asker hears it ended, so they ask again
            return "That request ended. Please ask again."
        # Too late when the line began after the deadline, or ran on past it by more than LATE_S: a long line begun
        # just in time no longer counts long after it, while a reply begun in time still does.
        began = float(ts) if ts is not None else now
        ended = float(entry.get("ts_end", ts)) if ts is not None else now
        if self.pending and (began > self.pending["expires"] or ended > self.pending["expires"] + LATE_S):
            # Only the person who asked hears that it expired; anyone else's line goes on as their own.
            mine = identity == self.pending["speaker"] and not anon
            self.pending = None
            if mine:
                return "That request expired. Please ask again."
        if self.pending:
            p = self.pending
            # The owner's own request answers only to the owner: a line from their voice that isn't the verified
            # owner (owner_entry: heard too faintly, or no longer theirs) is treated as another voice. Today the owner is
            # never asked to confirm (request_policy), so this only holds if that ever changes. A line that says nothing
            # about who spoke (anonymous) is always another voice.
            if anon or identity != p["speaker"] or (p["owner"] and not owner):
                # Another voice during someone's pending request is ignored, "cancel" too, so a guest can't clear
                # it. The owner can: their "cancel" ends it, and their wake phrase replaces it with their request.
                # A guest who asks with the wake phrase hears why there's no answer, rather than silence.
                if not owner:  # once per voice per request: a reply of ours mutes lines that overlap it
                    if not anon and identity not in self.told and re.match(r"^\W*" + wake_regex() + r"\b", text, re.I):
                        self.told.add(identity)
                        return WAITING
                    return None
                if norm in CANCEL:
                    self.pending = None
                    return "Cancelled."
                if not re.match(r"^\W*" + wake_regex() + r"\b", text, re.I):
                    return None
                self.pending = None
            elif norm in CANCEL:
                self.pending = None
                return "Cancelled."
            if re.match(r"^\W*" + wake_regex() + r"\b", text, re.I):
                # A fresh wake phrase is a new request, never a password/confirmation attempt. It keeps the failure
                # count: only a correct password or a lockout resets it, so asking again never buys more tries. The
                # same request asked again keeps its deadline (below), so it can't hold the floor forever.
                again = self.pending
                self.pending = None
        if self.pending:
            p = self.pending
            if p["stage"] == "password":
                if self.lockouts.locked(identity):  # a speaker relabeled onto a voice that is cooling down
                    self.pending = None
                    return "Please wait one minute before trying again."
                # A line the microphone service already matched to the password reads "[password]" (said_password).
                # Its mark counts only for this ask: said after the request (so an older, copied, or replayed marked
                # line never answers it), by the same speaker and in time (checked above); the ask then ends.
                flagged = entry.get("password") is True and ts is not None and float(ts) >= p["asked"]
                if not (flagged or self.verify(text)):
                    # The cooldown pays for the tries, so the count starts over with it. (Not at the next request:
                    # then cancelling after two tries and asking again would never lock out.)
                    if self.lockouts.fail(identity):
                        self.pending = None
                        return "Too many attempts. Please wait one minute."
                    return "Password not accepted. Try again or say cancel."
                self.lockouts.succeed(identity)
                p["authenticated"] = True
                if not p["needs_confirmation"]:
                    self.pending = None
                    return self.grant(p)
                p["stage"] = "confirm"
                p["expires"] = now + 45
                return (
                    "Password accepted. Confirm this exact request: " + p["request"] + ". Say yes confirm, or cancel."
                )
            if norm in CONFIRM:
                self.pending = None
                return self.grant(p, confirmed=True)
            return "Say yes confirm for the pending request, or cancel. New requests need a fresh authorization."
        if cfg.get("assistant.require_wake_phrase", False) and not re.match(r"^\W*" + wake_regex() + r"\b", text, re.I):
            return None
        req = re.sub(r"^\W*" + address_regex() + r"\b[\s,.!?:;-]*", "", text, flags=re.I).strip()
        addressed = req != text.strip()
        # Only the verified owner may skip the wake phrase. Anyone else asking a question into the room
        # (a guest, or the owner's voice mislabeled by the mic service) is talking to people, not to us.
        if not addressed and not (owner and (QUESTION.match(req) or REQUEST.match(req))):
            return None
        # The verified owner is never asked for a password, so a lockout never holds up their questions.
        if not owner and self.lockouts.locked(identity):
            return "Please wait one minute before trying again."
        if not req:
            return "Yes?"
        route = "camera" if camera_read(req, camera) else ("public" if PUBLIC.fullmatch(req) else "gateway")
        if route == "camera" and not owner and asks_about_the_past(req):
            # A camera question about the past is answered from the transcript and memory, not the live cameras, so
            # anyone but the verified owner gives the password for it, as for any other non-public request.
            route = "gateway"
        if route == "camera":
            return Grant(req, route, False, False, False, False, owner, addressed)
        needs_password, needs_confirmation = request_policy(req, owner)
        before = again if again is not None and again["speaker"] == identity else None
        same = before is not None and self.normal(before["request"]) == self.normal(req)
        # The same speaker's next request, while their last still waits, keeps its start: asking something else, or
        # rewording, buys no more than HOLD_S in all.
        opened = before["opened"] if before is not None else now
        expires = before["expires"] if same and before is not None else min(now + 45, opened + HOLD_S)
        if expires <= now:
            return "That request expired. Please ask again."
        p = Pending(
            request=req,
            route=route,
            speaker=identity,
            asked=float(entry.get("ts_end", ts)) if ts is not None else now,
            expires=expires,
            opened=opened,
            stage="password" if needs_password else "confirm",
            authenticated=False,
            needs_password=needs_password,
            needs_confirmation=needs_confirmation,
            owner=owner,
            addressed=addressed,
        )
        if not needs_password and not needs_confirmation:
            return self.grant(p)
        if anon:
            return UNHEARD  # nobody could tell its password or confirmation from anyone else's, so it starts nothing
        self.pending, self.told = p, set()
        if needs_password:
            return "Please say your password for this request, or say cancel."
        return "Confirm this exact request: " + req + ". Say yes confirm, or cancel."

    @staticmethod
    def grant(p: Pending, confirmed: bool = False) -> Grant:
        return Grant(
            p["request"],
            p["route"],
            p["authenticated"],
            confirmed,
            p["needs_password"],
            p["needs_confirmation"],
            p["owner"],
            p["addressed"],
        )
