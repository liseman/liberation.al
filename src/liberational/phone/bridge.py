"""Fully local two-way phone bridge: Twilio Media Streams <-> Parakeet ASR <-> Ollama LLM <-> the owner's cloned voice.

Flow per call (half-duplex, like a walkie-talkie):
  listening  -> energy VAD endpoints a caller utterance
  thinking   -> Parakeet transcribes, the speaker is matched against the stored voiceprints (voice_lookup.py), qwen3 replies
  speaking   -> reply is synthesized sentence by sentence in the cloned voice and streamed to Twilio;
                caller audio is not transcribed until Twilio's mark says playback finished

Voice cloning (Qwen3-TTS) needs MLX; elsewhere replies use Kokoro (kokoro-onnx) (tts.voice, no cloning).

Every call is recorded under paths.calls (default calls/), in <date>/<time>_<callSid>/:
  caller.wav, bot.wav, mixed.wav   8 kHz mono recordings (full call length)
  transcript.jsonl                  one line per turn, appended live
  transcript.txt                    readable transcript, written at call end
  call.json                         call metadata (from number, direction, whom we called and what for, its
                                    outcome, times, speakers)

Routes (public at phone.public_host through Tailscale Funnel at /bridge; nothing else is exposed):
  POST /bridge/voice         TwiML that connects the call to /bridge/media-stream
  WS   /bridge/media-stream  Twilio Media Streams (G.711 mu-law, 8 kHz, 20 ms frames)
  GET  /bridge/health        liveness

Run:  uv run python -m liberational.phone.bridge   (listens on 127.0.0.1, port ports.phone_bridge, 8790 by default)
"""

import asyncio
import base64
import datetime as dt
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import stat
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as xml_escape

import numpy as np
import requests
import soundfile as sf
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from scipy.signal import resample_poly

from liberational import config as cfg
from liberational import json_state, logs, model_runtime, ollama, speech_log, vision_log
from liberational.assistant import voice_auth
from liberational.memory import ask_history, remembered
from liberational.people import transcript_view
from liberational.phone import numbers
from liberational.phone import outbound as outbound_call

VOICE_WAV = cfg.path("owner.voice_ref")
VOICE_TXT = VOICE_WAV.with_suffix(".txt")
CALLS_DIR = cfg.path("paths.calls")
LIVE_FILE = "live.json"  # in a call's folder while it's on (Call.mark_live)
STATE_DIR = cfg.path("paths.state")
HOST, PORT = "127.0.0.1", cfg.get("ports.phone_bridge")
LLM = cfg.get("models.ollama.llm")


def _owner() -> str:
    return cfg.owner_or("the owner")


def system_prompt() -> str:
    return (
        f"You are {cfg.assistant_name()}, {_owner()}'s assistant, speaking on a live phone call. "
        "On this call you can only talk: you cannot send messages, set reminders, book anything, "
        "or look things up, and you have no calendar or file access. Never claim otherwise. "
        "Talk like a person on the phone: one or two short sentences, contractions, react to what they just said, "
        "ask a brief follow-up when it helps. Plain speech: no lists, no markdown, no emoji. "
        "A natural filler such as 'Um,' or 'Uh,' at the start is fine when it sounds right. "
        "If you don't know something, say so. Never end the call yourself: only say goodbye after THEY say bye or goodbye; "
        "'thanks' or 'ok' is not goodbye, keep talking."
    )


def mission_system(say: str) -> str:
    return (
        f"You are {cfg.assistant_name()}, calling for {_owner()}. "
        "Carry out the owner's goal through a live two-way conversation. "
        "The following is private task guidance, not a script to read aloud: " + say + "\n"
        "Listen, answer questions, ask relevant follow-ups, and adapt toward the goal. "
        "Never claim to be human or the owner; answer identity questions truthfully. "
        "Do not invent facts or completed actions. When finished, summarize the agreed outcome "
        "and say goodbye. You can negotiate on this call but cannot perform off-call actions. "
        "Use one or two short spoken sentences per turn. "
    )


# A goodbye at the end of what was said: "bye", "goodbye", "talk to you later", or "take care", followed by nothing,
# by "now", "then", or "for now", or by the assistant's name (bye_re). A sentence that is only "bye <one word>" also
# counts, since that word is usually the name misheard: "bye <name>" can come out of speech recognition as "Bye bomb."
# on a phone line. Any other trailing word ("Don't say bye yet.") is not a
# goodbye.
_BYE_WORDS = r"(?:good\s?)?bye|talk (?:to you )?later|take care"
LONE_BYE_RE = re.compile(r"(?:^|[.!?]\s*)(?:good\s?)?bye,?\s+\w+[.!]?\s*$", re.I)
# A goodbye or a hang-up asked for after one of these, in the same sentence, is talk about one, not one: "No need to
# say goodbye now!", "Before you say goodbye though", "don't hang up".
NOT_A_GOODBYE = re.compile(r"\b(?:don'?t|do not|not|never|no need to|before (?:you|we|I)|say|says|said|saying)\b", re.I)


def bye_re() -> re.Pattern[str]:
    """A goodbye at the end of what was said (see _BYE_WORDS); built per call, as the assistant's name is a setting."""
    name = re.escape(cfg.assistant_name())
    return re.compile(rf"\b(?:{_BYE_WORDS})(?:,?\s+(?:for now|now|then|{name}))?[.!]?\s*$", re.I)


def _unnegated(pattern: re.Pattern[str], text: str) -> bool:
    """Whether pattern matches text somewhere that no NOT_A_GOODBYE word comes before it in the same sentence."""
    return any(not NOT_A_GOODBYE.search(re.split(r"[.!?]", text[: m.start()])[-1]) for m in pattern.finditer(text))


def said_goodbye(text: str) -> bool:
    """Whether text ends with a goodbye (see bye_re and LONE_BYE_RE) that isn't talk about one."""
    return _unnegated(bye_re(), text) or _unnegated(LONE_BYE_RE, text)


def asks_to_hang_up(text: str) -> bool:
    """Whether text asks to end the call (HANGUP_RE), not "don't hang up" or "before you say goodbye"."""
    return _unnegated(HANGUP_RE, text)


# Voicemail / IVR: "message for 206..", "record your message after the tone", "press 1 to", "is not available", "mailbox is full"
VOICEMAIL_RE = re.compile(
    r"\b(?:record your message|after the (?:tone|beep)|leave (?:a|your) message|(?:is|are) not available|"
    r"can'?t take your call|mailbox|voice ?mail|press (?:the )?(?:pound|star|\d)|^message for \d|message is too short|"
    r"to (?:replay|re-record|delete|send)\b.*press|has been forwarded)\b",
    re.I,
)
VOICEMAIL_WAIT_S = 2.5  # after the prompt ends, wait this long for the beep before speaking the message


def greeting() -> str:
    """Brief identity for a verified owner who has already consented to recording."""
    return f"It's {cfg.assistant_name()}."


def greeting_other() -> str:
    """Opener for anyone else."""
    name = cfg.owner_name()
    if name:
        return f"Hi, this is {cfg.assistant_name()}, {name}'s assistant. {name} isn't available right now. What can I pass on?"
    return f"Hi, this is {cfg.assistant_name()}, an AI assistant. No one is available right now. What can I pass on?"


def owner_only() -> str:
    return f"That's only for {_owner()}, sorry. If you're {_owner()}, say the password."


# Owner treatment on a call. Caller ID can be spoofed, so the owner's number alone isn't enough: Twilio must also
# report the carrier's full STIR/SHAKEN attestation (StirVerstat grade A). Any other caller, the owner's number at a
# lower grade or with none included, is a guest until they say the voice password (assistant/voice_auth.py, with the
# same failure counts and lockouts as in person), which unlocks owner treatment for that call only.
GRADE_A = "TN-Validation-Passed-A"
LOCKED_OUT = "Too many wrong passwords. Try again later."  # a minute, or up to half an hour on the phone
_AUTH: dict[Path, voice_auth.Conversation] = {}


def auth() -> voice_auth.Conversation:
    """The voice password check, one per state dir for the life of the process: its lockouts keep counting in memory
    when a save fails, where a fresh one per try would reread the older file and let the guesses go on."""
    path = STATE_DIR / "voice_auth.json"
    return _AUTH.setdefault(path, voice_auth.Conversation(path))


def password_matches(text: str) -> bool:
    """Whether text says the voice password, by voice_auth's one definition (said_password), as in person."""
    return voice_auth.said_password(STATE_DIR / "voice_auth.json", text)


def answer_call(form: Mapping[str, Any]) -> tuple[str, bool, bool, bool]:
    """(the far end's number, whether we placed the call, whether that number is vouched for, whether it's the
    owner's) for a signed voice webhook. A call we placed is one Twilio marks outbound-api from our own number (both
    read as E.164), so an inbound call that spoofs our number isn't one; on it, the far end is the number we dialed, so
    it is vouched for. An inbound caller's number is vouched for only with grade A attestation, and the caller is the
    owner only with the owner's number and that."""
    frm, to = str(form.get("From", "")), str(form.get("To", ""))
    twilio = numbers.e164(outbound_call.twilio_number())
    outbound = form.get("Direction") == "outbound-api" and twilio is not None and numbers.e164(frm) == twilio
    remote = to if outbound else frm
    attested = outbound or form.get("StirVerstat") == GRADE_A
    return remote, outbound, attested, attested and outbound_call.is_owner_number(remote)


PEOPLE_DIR = STATE_DIR / "people"
MEM_PATH = STATE_DIR / "memory.jsonl"
CAPABILITY_CLAIM = re.compile(
    r"\b(take messages|set reminders?|book (an |a )?appointments?|send (a |an )?(message|text|email)|check my (calendar|email)|look (it )?up|i can (send|book|set|schedule|remind|text|call|order|buy)|i('| a)m able to (send|book|set))",
    re.I,
)
CAPABILITY_FALLBACK = "On this call I can only talk, so I can't do that. Want me to pass a note along?"
RECALL_LAST = re.compile(
    r"^\W*(?:what|which)\s+(?:did|do|have)\s+i\s+(?:just\s+|last\s+|previously\s+)?(?:ask|asked|say|said|tell|told)(?:\s+you|\s+them)?\W*$|^\W*what\s+was\s+my\s+(?:last|previous)\s+(?:question|request|thing|sentence)\W*$",
    re.I,
)


def owner_context() -> str:
    """Profile notes for the owner only. Other callers never get profile text."""
    notes = []
    for f in sorted(PEOPLE_DIR.glob("*.md")):
        if f.stem.lower() in ("index",):
            continue
        notes.append(f.read_text())
    return (
        (
            f"\n\nWhat you know about {_owner()} from their own recent speech (low confidence, may contain errors; "
            "use only if relevant, never invent details):\n" + "\n".join(notes)
        )
        if notes
        else ""
    )


# ---------------------------------------------------------------- owner voice commands ----
# Owner-only (owner.phone): camera questions, remember / forget / what-do-you-remember.
# Any caller: "hang up" / "goodbye" ends the call. Memory is the same file the in-person loop (pack.py) uses.
def owner_system() -> str:
    o = _owner()
    return (
        f"You are {cfg.assistant_name()}, {o}'s assistant, speaking with {o} on a live phone call. "
        f"You can talk, answer from the notes provided (camera notes, things {o} asked you to remember, "
        f"and what you know about {o}), and you can end the call when {o} asks. "
        "You cannot send messages, book anything, look things up on the internet, or access their calendar or files, "
        "and you cannot move or control the cameras. Never claim otherwise. "
        "Keep replies to one short sentence when possible. Plain speech: no lists, no markdown, no emoji. "
        "If the notes don't cover it, say so."
    )


REMEMBER_RE = re.compile(r"^\W*(?:please\s+|hey\s+|ok(?:ay)?\s+)*remember\s+(?:that\s+)?(.+?)\W*$", re.I)
FORGET_RE = re.compile(r"^\W*(?:please\s+|hey\s+|ok(?:ay)?\s+)*forget\s+(?:that\s+|about\s+)?(.+?)\W*$", re.I)
RECALL_MEM_RE = re.compile(r"^\W*what\s+(?:do\s+you|have\s+i\s+asked\s+you\s+to)\s+remember\W*$", re.I)
CAMERA_RE = re.compile(
    r"\bcameras?\b|\bwhat (?:do|can|could) you see\b|\bwhat(?:'s| is| are) (?:around|in front of me|in the room|in view|happening)\b"
    r"|\bdescribe (?:the )?(?:room|scene|view|surroundings)\b|\blook around\b"
    r"|\bwho(?:'s| is| are) (?:here|around|in (?:the )?(?:room|view))\b|\bwhat'?s around (?:me|us)\b",
    re.I,
)
HANGUP_RE = re.compile(r"\b(?:hang up|hang up now|end (?:the )?call|goodbye|good bye)\b", re.I)


def _ts(e: dict[str, Any]) -> float:
    t = float(e.get("ts") or 0)
    return t / 1000.0 if t > 1e12 else t


def _age(seconds: float) -> str:
    return f"{int(seconds)} seconds" if seconds < 90 else f"{int(seconds // 60)} minutes"


def scene_entries() -> list[dict[str, Any]]:
    """The newest captions from the vision log, oldest first ([] when it can't be asked)."""
    try:
        caps = vision_log.at(STATE_DIR).scene(time.time())
    except Exception:
        return []
    return [c for c in caps if c.get("caption")]


def scene_lines(max_age: float = 600) -> list[str]:
    """Newest caption per camera, within max_age seconds. The vision loop writes 'Same.' when nothing changed."""
    now, latest = time.time(), {}
    for e in scene_entries():
        latest[e.get("cam", "?")] = e
    lines = []
    for cam, e in sorted(latest.items()):
        age = now - _ts(e)
        if age > max_age:
            continue
        cap = str(e["caption"]).strip()
        if cap.lower().startswith("same."):
            cap = "(unchanged) " + cap[5:].strip()
        names = sorted(
            {
                f.get("name")
                for f in (e.get("faces") or [])
                if isinstance(f, dict) and f.get("name") not in (None, "", "unknown")
            }
        )
        extra = f" Recognised: {', '.join(names)}." if names else ""
        lines.append(f"{cam}, {_age(age)} ago: {cap}{extra}")
    return lines


def camera_fallback() -> str:
    entries = scene_entries()
    if not entries:
        return "I don't have any camera notes yet."
    return f"I don't have a live camera view right now. The newest camera note is {_age(time.time() - _ts(entries[-1]))} old."


# ---- approvals: changes proposed on a call wait here until the owner approves them by phone ----
APPROVALS = STATE_DIR / "approvals.jsonl"
LIST_APPROVALS_RE = re.compile(
    r"\b(what needs (my )?approval|what'?s (waiting|pending)|pending (approvals?|changes)|anything (to|i need to) approve)\b",
    re.I,
)
APPROVE_RE = re.compile(
    r"^\W*(?:please\s+)?(?:approve|authori[sz]e|go ahead(?: with)?|yes,? (?:do|go) (?:it|that|ahead))\b\W*(.*)$", re.I
)
DENY_RE = re.compile(r"^\W*(?:deny|reject|cancel that|don'?t do (?:that|it)|no,? (?:don'?t|cancel))\b", re.I)


def _approval_state() -> dict[str, dict[str, Any]]:
    """Replay the log: id -> {text, status: pending|approved|denied|done, t}."""
    items: dict[str, dict[str, Any]] = {}
    if not APPROVALS.exists():
        return items
    for raw in APPROVALS.read_text().splitlines():
        try:
            e = json.loads(raw)
        except Exception:
            continue
        i = e.get("id")
        if e.get("op") == "propose" and i:
            items[i] = {"text": e.get("text", ""), "status": "pending", "t": e.get("t", 0)}
        elif i in items and e.get("op") in ("approve", "deny", "done"):
            items[i]["status"] = {"approve": "approved", "deny": "denied", "done": "done"}[e["op"]]
    return items


def _append_approval(op: str, **fields: Any) -> None:
    with APPROVALS.open("a") as f:
        f.write(json.dumps({"op": op, "t": time.time(), **fields}) + "\n")


def approval_turn(text: str) -> tuple[str | None, str | None]:
    """Owner-only fast path. Returns (reply, hint) or (None, None)."""
    if LIST_APPROVALS_RE.search(text):
        pend = [(i, v) for i, v in _approval_state().items() if v["status"] == "pending"]
        if not pend:
            return "Nothing is waiting for your approval.", None
        return "Waiting for your approval: " + " ; ".join(f"{i}: {v['text']}" for i, v in pend[-3:]) + ".", None
    if DENY_RE.search(text):
        pend = [i for i, v in _approval_state().items() if v["status"] == "pending"]
        if not pend:
            return "Nothing to deny.", None
        _append_approval("deny", id=pend[-1])
        return "Denied.", None
    m = APPROVE_RE.match(text)
    if m:
        pend = [i for i, v in _approval_state().items() if v["status"] == "pending"]
        if not pend:
            return "Nothing is waiting for approval.", None
        target = pend[-1]
        _append_approval("approve", id=target)
        return (
            None,
            f"{_owner()} approved: {_approval_state()[target]['text']}. Carry it out now and say in one sentence what you did.",
        )
    return None, None


def approved_notes() -> str:
    """Approved items the agent should carry out now; each is handed over once, then marked done."""
    out = []
    for i, v in _approval_state().items():
        if v["status"] == "approved":
            out.append(f"Approved by {_owner()}, carry out now: {v['text']}")
            _append_approval("done", id=i)
    return " | ".join(out)


# owner_turn runs in a thread with this cap: past Twilio's own timeouts (outbound.TWILIO_TIMEOUT_S), so it only ends a
# turn stuck somewhere else, such as a DNS lookup that never returns.
OWNER_TURN_TIMEOUT_S = 30.0
OWNER_TURN_SLOW = "Sorry, that's taking too long, so I can't tell yet whether it worked."


def owner_turn(text: str, owner: bool) -> tuple[str | None, str | None]:
    """Returns (direct_reply, llm_hint). direct_reply is spoken as-is; llm_hint steers the next LLM reply."""
    if owner:
        from liberational.phone import outbound as _oc

        _p = _oc.parse(text)
        if _p:
            _ok, _detail = _oc.place(_p[0], _p[1])
            return _detail, None
    if LIST_APPROVALS_RE.search(text) or DENY_RE.search(text) or APPROVE_RE.match(text):
        if not owner:
            return owner_only(), None
        reply, hint = approval_turn(text)
        if reply is not None:
            return reply, None
        return None, hint
    if REMEMBER_RE.match(text) or FORGET_RE.match(text) or RECALL_MEM_RE.match(text):
        if not owner:
            return owner_only(), None
        m = REMEMBER_RE.match(text)
        if m:
            item = m.group(1).strip()
            remembered.append(MEM_PATH, "remember", item)
            return f"Okay, I'll remember that {item}.", None
        m = FORGET_RE.match(text)
        if m:
            item = m.group(1).strip()
            remembered.append(MEM_PATH, "forget", item)
            return (
                "Okay, forgotten."
                if item.lower() in ("everything", "all", "all of it")
                else f"Okay, forgotten: {item}."
            ), None
        mem = remembered.items(MEM_PATH)
        return (("I remember: " + "; ".join(mem[-5:]) + ".") if mem else "Nothing yet."), None
    if CAMERA_RE.search(text):
        if not owner:
            return owner_only(), None
        if not scene_lines():
            return camera_fallback(), None
        return None, (
            f"{_owner()} is asking what the cameras see. Answer in at most two short sentences, under 35 words in total, using only the camera notes: "
            "how many people, the main objects, and any hazard. If a note is older than a minute, say so."
        )
    return None, None


def hangup_call(call_sid: str) -> str:
    ok, detail = outbound_call.hangup(call_sid)
    if not ok:
        raise RuntimeError(detail)
    return "completed"


# ---------------------------------------------------------------- call security ----
# Only requests signed by Twilio (X-Twilio-Signature) can start a call. Each signed voice request mints a one-time
# token; the media stream must present it. Owner = what Twilio reported for that call (answer_call: the owner's
# number with grade A attestation, or a call we placed to it), never a value the websocket client claims; any other
# caller can say the voice password for owner treatment on that call.
def public_host() -> str:
    """phone.public_host, or "" when unset (then Twilio can't reach the bridge)."""
    return cfg.get("phone.public_host").strip()


def public_voice_urls() -> tuple[str, str]:
    """(voice webhook URL, the same without the port). Twilio signs the second form."""
    url = outbound_call.public_url("/voice")
    return url, "https://" + public_host().split(":")[0] + "/bridge/voice"


GREETING_ONCE = STATE_DIR / "greeting_once.json"  # written by phone_call.py --say; used by the next owner call
PENDING: dict[str, dict[str, Any]] = {}  # token -> {callSid, remote, owner, say, t}
PENDING_TTL_S = 180
# Every call this bridge set up, so a captured, signed /bridge/voice request replayed later can't start a second
# session for the same call. On disk, so a restart doesn't forget them. A call lasts at most 4 hours and Twilio
# retries a setup within seconds, so 30 days is long past any retry; SEEN_MAX bounds the file however busy the line.
SEEN_CALLS = STATE_DIR / "phone_seen_calls.json"
SEEN_KEEP_S = 30 * 86400
SEEN_MAX = 5000


def first_setup(call_sid: str, now: float) -> bool:
    """Record call_sid as set up and say whether it is new. An unreadable record starts over (logged): refusing every
    call would take the phone line down to stop a replay that needs TLS interception."""
    try:
        seen = json.loads(SEEN_CALLS.read_text())
        seen = {str(k): float(v) for k, v in seen.items()} if isinstance(seen, dict) else {}
    except FileNotFoundError:
        seen = {}
    except Exception as e:
        log.warning("seen calls %s unreadable (%s): starting over", SEEN_CALLS, type(e).__name__)
        seen = {}
    if call_sid in seen:
        return False
    seen = {k: t for k, t in seen.items() if now - t < SEEN_KEEP_S}
    seen[call_sid] = now
    kept = dict(sorted(seen.items(), key=lambda kv: kv[1])[-SEEN_MAX:])
    try:
        json_state.write(SEEN_CALLS, kept, private=True)
    except OSError as e:
        log.warning("seen calls %s not saved (%s)", SEEN_CALLS, e)
    return True


def twilio_signing_setup() -> tuple[str, list[str]]:
    vals = outbound_call.load_env()
    tokens = [vals["TWILIO_AUTH_TOKEN"]] + ([vals["TWILIO_SIGNING_TOKEN"]] if vals.get("TWILIO_SIGNING_TOKEN") else [])
    return vals["TWILIO_ACCOUNT_SID"], tokens


def twilio_signature_ok(url: str, params: Mapping[str, Any], signature: str | None) -> bool:
    """Twilio's algorithm (same as the voice-call plugin): HMAC-SHA1 over URL + sorted POST params, keyed by an auth token, base64.
    Accepts either token (rotation) and either URL form (with or without the non-default port)."""
    _, tokens = twilio_signing_setup()
    data = "".join(k + str(params[k]) for k in sorted(params))
    for tok in tokens:
        for u in (url, public_voice_urls()[1]):
            mac = base64.b64encode(hmac.new(tok.encode(), (u + data).encode(), hashlib.sha1).digest()).decode()
            if hmac.compare_digest(mac, signature or ""):
                return True
    return False


def take_greeting_once(max_age_s: float = 900) -> str | None:
    """One-shot opener for the next owner call, then deleted: a phone_call.py --say greeting, then any phone lockout
    notice (take_lockout_notice)."""
    say = None
    try:
        d = json.loads(GREETING_ONCE.read_text())
        GREETING_ONCE.unlink(missing_ok=True)
        if time.time() - float(d.get("created", 0)) <= max_age_s:
            say = (d.get("text") or "")[:600] or None
    except Exception:
        pass
    return " ".join(t for t in (say, take_lockout_notice()) if t) or None


# Telling the owner when the phone locks out a password guesser, through channels the pack already has: the
# one-shot greeting of their next owner call (or the call they unlock with the password), a warning in this log
# (the dashboard's Debug tab), and the agent when one is configured. The notice is one line that counts lockouts,
# so a guessing spree can't pile up messages; the agent hears at most once per LOCKOUT_AGENT_EVERY_S.
LOCKOUT_NOTICE = STATE_DIR / "phone_lockout_notice.json"
LOCKOUT_NOTICE_KEEP_S = 7 * 86400
LOCKOUT_AGENT_EVERY_S = 1800.0
_last_agent_notice = [0.0]


def notice_lockout(number: str, now: float) -> None:
    """A wrong password on the phone just started a lockout: log it, count it toward the next owner call's notice,
    and tell the agent (rate-limited, in the background, when one is configured)."""
    when = time.strftime("%H:%M", time.localtime(now))
    log.warning("phone password lockout: too many wrong passwords from %s at %s", number or "an unknown number", when)
    d = _read_notice()
    count = int(d["count"]) + 1 if d and now - float(d["first"]) <= LOCKOUT_NOTICE_KEEP_S else 1
    first = float(d["first"]) if d and count > 1 else now
    try:
        # numbers only: the words are made when said
        json_state.write(LOCKOUT_NOTICE, {"count": count, "first": first, "last": now}, private=True)
    except OSError as e:
        log.warning("phone lockout notice not saved (%s)", e)
    if cfg.agent_configured() and now - _last_agent_notice[0] >= LOCKOUT_AGENT_EVERY_S:
        _last_agent_notice[0] = now
        text = notice_text(count, now, now)
        threading.Thread(target=tell_agent, args=(text,), daemon=True, name="lockout-notice").start()


def notice_text(count: int, last: float, now: float) -> str:
    """The notice in words, with the day when the last lockout wasn't today."""
    times = "once" if count == 1 else f"{count} times"
    t = time.localtime(last)
    day = (
        ""
        if time.strftime("%Y-%m-%d", t) == time.strftime("%Y-%m-%d", time.localtime(now))
        else time.strftime(" on %A", t)
    )
    return (
        f"Heads up: someone tried the password on the phone and got locked out {times}, most recently{day} at "
        + time.strftime("%H:%M", t)
        + "."
    )


def _read_notice() -> dict[str, float] | None:
    """The notice's numbers, or None when there is none, or the file is a link, someone else's, open to others, or
    malformed (a planted file says nothing)."""
    try:
        st = LOCKOUT_NOTICE.lstat()
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid() or st.st_mode & 0o077:
            return None
        d = json.loads(LOCKOUT_NOTICE.read_text())
        out = {k: float(d[k]) for k in ("count", "first", "last")}
        return out if out["count"] >= 1 and all(math.isfinite(v) for v in out.values()) else None
    except Exception:
        return None


def tell_agent(text: str) -> None:
    """Ask the configured agent to pass a notice on to the owner, through whatever channel it reaches them on. Its
    reply is logged; a failure is too."""
    try:
        token = cfg.agent_token()
        owner = cfg.owner_or("the owner")
        body = {
            "model": cfg.get("agent.phone_agent"),
            "user": "pack-phone-notices",
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": f"Please pass this notice from the phone line on to {owner}, on the channel you usually "
                    f"reach them on, word for word: {text}",
                }
            ],
        }
        r = requests.post(
            cfg.get("agent.url"),
            json=body,
            timeout=60,
            headers={"Authorization": f"Bearer {token}", "x-openclaw-message-channel": "phone"},
        )
        r.raise_for_status()
        log.info("agent on the phone lockout notice: %s", str(r.json()["choices"][0]["message"]["content"])[:200])
    except Exception as e:
        log.warning("phone lockout notice not sent to the agent (%s)", type(e).__name__)


def take_lockout_notice() -> str | None:
    """The phone lockout notice for the owner, once, if it is recent enough to matter."""
    d = _read_notice()
    LOCKOUT_NOTICE.unlink(missing_ok=True)
    now = time.time()
    if d is None or now - d["last"] > LOCKOUT_NOTICE_KEEP_S:
        return None
    return notice_text(int(d["count"]), d["last"], now)


FRAME = 160  # 20 ms of 8 kHz mu-law
SR_PHONE, SR_ASR, SR_TTS = 8000, 16000, 24000
END_SILENCE_FRAMES = 20  # 400 ms of silence ends an utterance (was 500: felt laggy)
WIRED_BYTES = 8 << 30  # the phone voice models' MLX memory, kept resident (load_models)
BARGE_FRAMES = 15  # 300 ms of caller speech while Bob talks = interrupt
BARGE_RMS = 1500.0  # the owner's voice over Bob is 3500+, while line echo of Bob's own audio peaks at 2300-2800; 1500 clears echo with margin
START_FRAMES = 3  # 60 ms above threshold starts one
RESUME_FRAMES = 5  # 100 ms of speech while a reply is being made: the caller went on after a pause (see on_media)
# A turn drops its reply because the caller went on (Call.went_on) at most this many times in a row, then answers what
# it has: background talk that crosses RESUME_FRAMES on every think would otherwise drop and re-ask forever.
MAX_WENT_ON = 2
MAX_UTT_FRAMES = 1000  # 20 s cap
MIN_UTT_FRAMES = 20  # ignore blips shorter than 400 ms

# Its service log (logs/phone-bridge.log), which the call review and eval read through phone.bridge_log().
log = logs.get("phone-bridge")

# MLX keeps streams per thread. Load the models and run every MLX call on this one thread.
MLX_EXEC = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")


async def run_mlx[T](fn: Callable[..., T], *args: Any) -> T:
    return await asyncio.get_running_loop().run_in_executor(MLX_EXEC, fn, *args)


# ---------------------------------------------------------------- G.711 mu-law ----
def ulaw_decode(data: bytes) -> np.ndarray:
    u = (~np.frombuffer(data, dtype=np.uint8)).astype(np.int32)
    t = (((u & 0x0F) << 3) + 0x84) << ((u & 0x70) >> 4)
    return np.where(u & 0x80, 0x84 - t, t - 0x84).astype(np.int16)


def ulaw_encode(pcm: np.ndarray) -> bytes:
    x = pcm.astype(np.int32)
    sign = np.where(x < 0, 0x80, 0).astype(np.int32)
    mag = np.minimum(np.abs(x), 32635) + 0x84
    bits = np.zeros_like(mag)
    tmp = mag >> 8
    while np.any(tmp):
        bits += tmp > 0
        tmp >>= 1
    seg = np.clip(bits, 0, 7)
    mant = (mag >> (seg + 3)) & 0x0F
    return (~(sign | (seg << 4) | mant) & 0xFF).astype(np.uint8).tobytes()


# ---------------------------------------------------------------- models (loaded once) ----
# Any for the models: each has three backends (MLX, CUDA, and Kokoro or none) with no shared type, set by load_models().
_asr: Any = None
_tts: Any = None
FIXED_AUDIO: dict[str, np.ndarray] = {}  # pre-rendered greeting/notices/farewell; no inference on the call path
_ref: tuple[Any, str] | None = None  # (reference audio, transcript): an mx array on MLX, a path on CUDA; None on Kokoro
_spk: Any = None
FILLER_AUDIO: np.ndarray | None = None  # pre-rendered 'One moment.' so the filler plays with no synthesis wait
STILL_AUDIO: np.ndarray | None = None  # pre-rendered 'Still working on it.' for longer tasks


def qwen3_cuda_ok() -> bool:
    from liberational.speech import qwen3_cuda

    return qwen3_cuda.available()[0]


def load_models() -> None:
    global _asr, _tts, _ref, _spk
    from liberational.speech.asr import ASR, ASR_MODEL_NAME

    log.info("using %s", ASR_MODEL_NAME)
    _asr = ASR()
    if cfg.has_mlx():
        import mlx.core as mx
        from mlx_audio.tts.utils import load as load_tts

        # Keep the voice models wired: under memory pressure macOS swapped them out between calls, and the next
        # greeting's first audio took over 4 s (on the Mac, with 34 GB of swap in use).
        mx.set_wired_limit(min(WIRED_BYTES, int(mx.device_info()["max_recommended_working_set_size"])))

        if not VOICE_WAV.exists():
            raise FileNotFoundError(
                f"voice reference {VOICE_WAV} not found; set owner.voice_ref (with a .txt transcript next to it)"
            )
        log.info("loading Qwen3-TTS clone")
        _tts = load_tts(cfg.hf_repo("tts_clone"))
        model_runtime.loaded("models.hf.tts_clone", cfg.hf_repo("tts_clone"), "voice", "Apple GPU", owner=_tts)
        ref, sr = sf.read(str(VOICE_WAV), dtype="float32")
        if ref.ndim > 1:
            ref = ref.mean(axis=1)
        if sr != SR_TTS:
            ref = resample_poly(ref, SR_TTS, sr).astype(np.float32)
        _ref = (mx.array(ref), VOICE_TXT.read_text().strip())
    elif VOICE_WAV.exists() and qwen3_cuda_ok():
        # Same model on CUDA; it takes mlx-audio's generate() arguments, with the reference as a path.
        from liberational.speech import qwen3_cuda

        log.info("loading Qwen3-TTS clone on CUDA")
        _tts = qwen3_cuda.load()
        from liberational.speech.voice_clone import ref_text_for

        _ref = (str(VOICE_WAV), ref_text_for(VOICE_WAV))
    else:
        from liberational.speech.kokoro import Kokoro

        log.warning("no voice cloning here (MLX or a CUDA GPU needed); calls use Kokoro (%s)", cfg.get("tts.voice"))
        _tts = Kokoro(voice=cfg.get("tts.voice"), speed=1.1)
    try:
        # The voices the in-person mics hear (voiceprints on Continuum). A lookup is read-only, so calls never add one.
        from liberational.voice_lookup import VoiceLookup

        _spk = VoiceLookup(STATE_DIR)
        log.info("caller voices are looked up against the stored voiceprints")
    except Exception as e:
        log.warning("voice lookup unavailable, callers will be labelled by number: %s", e)
    global FILLER_AUDIO, STILL_AUDIO, FIXED_AUDIO
    FILLER_AUDIO = synthesize("One moment.")
    STILL_AUDIO = synthesize("Still working on it.")
    FIXED_AUDIO = {text: synthesize(text) for text in (greeting(), mission_disclosure().strip(), "Okay, bye!")}
    log.info("models ready (fillers, greeting and farewell pre-rendered)")


def transcribe(audio16k: np.ndarray) -> str:
    return _asr(audio16k.astype(np.float32))


def _to_phone(audio: np.ndarray, sr: int) -> np.ndarray:
    """float32 audio at sr -> 8 kHz int16 PCM."""
    if sr != SR_PHONE:
        audio = resample_poly(audio, SR_PHONE, sr)
    return np.clip(audio * 32767, -32768, 32767).astype(np.int16)


def synthesize(text: str) -> np.ndarray:
    """Return one sentence as 8 kHz int16 PCM in the cloned voice (Kokoro where cloning isn't available)."""
    if _ref is None:
        audio, sr = _tts.synth(text)
        return _to_phone(audio, sr) if len(audio) else np.zeros(0, dtype=np.int16)
    ref_audio, ref_text = _ref
    parts, out_sr = [], SR_TTS
    for r in _tts.generate(text=text, ref_audio=ref_audio, ref_text=ref_text, stream=False):
        parts.append(np.asarray(r.audio, dtype=np.float32).reshape(-1))
        out_sr = r.sample_rate
    if not parts:
        return np.zeros(0, dtype=np.int16)
    audio = np.concatenate(parts)
    if out_sr != SR_PHONE:
        audio = resample_poly(audio, SR_PHONE, out_sr)
    return np.clip(audio * 32767, -32768, 32767).astype(np.int16)


def synthesize_stream(text: str, on_chunk: Callable[[np.ndarray], None]) -> None:
    """Stream the reply in the cloned voice: on_chunk(int16 8 kHz) is called as each piece is ready (MLX thread)."""
    if _ref is None:  # Kokoro yields one sentence at a time
        for a, sr in _tts.generate(text):
            on_chunk(_to_phone(a, sr))
        return
    ref_audio, ref_text = _ref
    for r in _tts.generate(text=text, ref_audio=ref_audio, ref_text=ref_text, stream=True, streaming_interval=0.5):
        a = np.asarray(r.audio, dtype=np.float32).reshape(-1)
        if r.sample_rate != SR_PHONE:
            a = resample_poly(a, SR_PHONE, r.sample_rate)
        on_chunk(np.clip(a * 32767, -32768, 32767).astype(np.int16))


def identify_speaker(audio16k: np.ndarray) -> tuple[str | None, float | None]:
    """(name, similarity) of the stored voice a caller's clip is (voice_lookup.py), or (None, sim) for none close
    enough, or (None, None) when the lookup can't answer (no vector worker). A voice's name is read from the event
    log each time (its person), so a caller is never named after a voice People deleted, or by a name it renamed."""
    if _spk is None:
        return None, None
    try:
        heard = _spk.identify(audio16k)
    except TimeoutError as e:
        log.warning("caller voice lookup unavailable: %s", logs.reason(e))
        return None, None
    return heard.name, round(float(heard.sim), 2)


def ask_llm(history: list[dict[str, str]], system: str | None = None) -> str:
    reply = ollama.chat(
        LLM,
        [{"role": "system", "content": system or system_prompt()}, *history],
        num_ctx=4096,  # small ctx: keeps the 30B agent model resident
        num_predict=80,
        temperature=0.3,
        think=False,
        timeout=90,
    )
    return reply["message"]["content"].strip()


# ---------------------------------------------------------------- phone agent (owner calls) ----
# Owner calls go to the agent gateway's restricted phone agent (agent.phone_agent) over its chat endpoint (agent.url).
# The agent's tools are limited in config (web search, weather, fetch; no shell, files, or messaging).
OWNER_CLAIM = re.compile(
    r"\b((send|sending|sent|texting|texted|emailing|emailed|messaging|messaged|booking|booked|ordering|ordered|buying|bought|purchasing|purchased|placing|placed)\s+(a |an |the |your |my |that |this |it )?(message|messages|text|texts|email|emails|imessage|sms|note|reservation|order|booking|purchase|payment|call|reminder)s?"
    r"|(sending|sent|texting|texted|emailing|emailed|messaging|messaged) (it |that |this |you |him |her |them |mom |dad )?(now|to|for)"
    r"|i('ll| will| can| am going to) (send|book|schedule|remind|text|email|message|order|buy|purchase|place) "
    r"|i('ve| have) (booked|ordered|bought|purchased|scheduled|placed|sent|texted|emailed|messaged) "
    r"|take messages|set reminders?|book (an |a )?appointments?"
    r"|i('ll| will) (call you back|get back to you|adjust|integrate|update|rebuild|build|fix|set up|make) (the |your |this |that |a |an )?(dashboard|recordings?|transcript|page|app|system|service|change|update|fix|setup)?"
    r"|i('ll| will) (call you|text you|message you|email you) (back|later|when|once)?)\b",
    re.I,
)
TOOL_TEXT = re.compile(
    r'(?:^|\n)\s*(?:exec|web_search|web_fetch|read|write|edit|process|session_status|skills_read|skills_search|apply_patch)\s*\n?\s*\{|"command"\s*:|"pty"\s*:|<tool_call>|\{\s*"name"\s*:',
    re.I,
)
NARRATION = re.compile(
    r"\b(let me|i'?ll (check|search|look|find|see|run|book|try)|i'?m (checking|searching|looking|finding|running|going to)|i am (checking|searching|looking)|one moment|hold on|give me a (moment|second))\b",
    re.I,
)


def bad_reply(reply: str | None) -> bool:
    """A stall, a tool call written as text, or an empty reply: none of these may be spoken."""
    r = (reply or "").strip()
    return (
        (not r)
        or bool(STALL_REPLY.match(r))
        or bool(TOOL_TEXT.search(r))
        or bool(NARRATION.search(r))
        or ("{" in r and "}" in r)
    )


# Owner turns that need the phone agent's tools (the web, this machine, an action). Everything else on an owner call is
# answered by the fast local model (models.ollama.llm): a full agent turn takes 4 to 13 s on live calls,
# the local model about half a second.
NEEDS_AGENT = re.compile(
    r"\b(weather|forecast|rain(ing)?|snow|news|headlines?|search|google|look (it |that |this )?up|find out|prices?|"
    r"cost of|stocks?|scores?|traffic|directions|how far|open (now|until|today)|hours|websites?|url|links?|latest|"
    r"today'?s|tomorrow|tonight|this (week|weekend)|schedule|calendar|remind(er)?s?|e-?mails?|texts?|messages?|send|"
    r"book|order|buy|status|running|services?|uptime|cameras?|dashboard|recordings?|who'?s (here|home|there|in|at)|"
    r"what'?s (on|happening)|check|restart|turn (on|off)|when did|time in|how much|how many|who (is|was|won|wrote)|"
    r"where (is|are)|what happened|yesterday|last (night|week)|call (him|her|them|back)|hang up)\b",
    re.I,
)
# A fast-model reply that says it can't answer: the phone agent answers instead.
DEFERS = re.compile(
    r"\b(real[- ]time|(don'?t|do not) have (access|the ability|a way|that information)|can'?t (check|look|browse|access|"
    r"search|see)|unable to (check|access|browse|see)|as an ai|no (internet|web) access|i (don'?t|do not) know|"
    r"not sure)\b",
    re.I,
)


# A fast-model reply that promises something it can't do from a phone line (go, bring, meet, end the call),
# such as "I'll grab the map and meet you there" or "I'll end the call" without ending it.
FAST_PROMISE = re.compile(
    r"\b((i'?ll|i will|i'?m going to|let me) (end|hang|grab|meet|come|go|head|bring|pick|drive|walk|get|send|check|"
    r"look|call|text|remind|book|order|set)|meet you|end (the|this) call|hang up)\b",
    re.I,
)


def fast_system(base: str) -> str:
    """The fast model's system prompt: the owner turn's, plus what a small model gets wrong on a call."""
    o = _owner()
    return (
        f"{base}\nOn this turn: you are talking with {o} right now, so say 'you', never '{o}' in the third person. "
        "Answer from general knowledge, opinions, and ideas freely; only say you don't know for live facts you weren't "
        "given. Never promise to do, fetch, bring, meet, or end anything."
    )


def fast_replies() -> bool:
    """phone.fast_replies: answer everyday owner-call turns with the fast local model (default on)."""
    try:
        value = cfg.get("phone.fast_replies")
    except Exception:
        return True
    return True if value is None else bool(value)


STALL_REPLY = re.compile(
    r"^(?:checking|one moment|let me|looking|hold on|on it|working on it|still working on it|running|i'm checking|i am checking|i'll (?:run|check|look)|i will (?:run|check|look)|i'm going to|i am going to)\b[^.?!]{0,120}[.!]?$",
    re.I,
)


def phone_tool_rule(call_sid: str | None = None) -> str:
    """How the agent checks the pack's services on this machine (launchd or systemd, via pack-services)."""
    status = cfg.ROOT / ".venv" / "bin" / "pack-services"
    control = ""
    if call_sid and re.fullmatch(r"CA[0-9a-fA-F]{32}", call_sid):
        end_call = cfg.ROOT / ".venv" / "bin" / "pack-phone-call"
        control = (
            f" To end this call when the caller asks, call exec with: {end_call} --hangup {call_sid}. "
            "This ends this call only; do not place another call. "
        )
    return (
        "[Tool rule: for this machine's state, call exec first and answer from its output. Do not narrate, never write commands as text. "
        f"Pack services: run {status} status, which prints each installed service (vision = cameras, dashboard = web dashboard, "
        "ambient = microphones, identity = names, vlm = caption server) and whether it is running. "
        "Uptime: run uptime. For the web, use web_search or web_fetch." + control + "]"
    )


def phone_agent_reply(call_sid: str, text: str, notes: str | None = None, timeout: float = 60) -> str:
    """One turn in this call's phone session. Same user key per call, so the agent keeps the call's context."""
    if not cfg.agent_configured():
        raise RuntimeError("no agent configured")  # owner_agent_reply answers locally instead
    token = cfg.agent_token()
    if not token:
        raise RuntimeError("gateway token unavailable")
    parts = [phone_tool_rule(call_sid)]
    if notes:
        parts.append(f"[Notes: {notes}]")
    parts.append(f"{cfg.owner_or('Owner')}: {text}")
    body = {
        "model": cfg.get("agent.phone_agent"),
        "user": f"phone-call-{call_sid}",
        "stream": False,
        "messages": [{"role": "user", "content": "\n".join(parts)}],
    }
    r = requests.post(
        cfg.get("agent.url"),
        json=body,
        timeout=timeout,
        headers={"Authorization": f"Bearer {token}", "x-openclaw-message-channel": "phone"},
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def warm_phone_session(call_sid: str) -> None:
    """Runs while the greeting plays, so the first real turn of the call doesn't pay the cold prompt load."""
    if not cfg.agent_configured():
        return
    try:
        phone_agent_reply(call_sid + "-warm", "Call just started. Reply with the single word ready.", None, timeout=120)
        log.info("[%s] phone session warmed", call_sid)
    except Exception as e:
        log.warning("[%s] phone session warm-up failed: %s", call_sid, e)


def sanitize_spoken(text: str) -> str:
    t = re.sub(r"[*_`#>|]+", " ", text or "")
    t = re.sub(r"^\s*(?:[-•]|\d+[.)])\s+", "", t, flags=re.M)
    t = " ".join(t.split())
    parts = [x for x in re.split(r"(?<=[.!?])\s+", t) if x]
    return (" ".join(parts[:2]) if parts else t)[:400] or "Sorry, I didn't catch that."


def room_transcript_line(minutes: int = 5, max_rows: int = 15) -> str:
    """What was said in the room recently (the speech log on Continuum). Owner turns only."""
    try:
        now, rows = time.time(), []
        recent = speech_log.shared().recent(minutes * 60, now=now)
        for e in transcript_view.view(recent, STATE_DIR):  # who said each line, per the event log
            if now - float(e.get("ts_start", 0)) <= minutes * 60 and (e.get("text") or "").strip():
                rows.append(f"{e.get('speaker', '?')}: {e['text'].strip()}")
        return ("Room transcript, last %d minutes: " % minutes + " / ".join(rows[-max_rows:])) if rows else ""
    except Exception:
        return ""


def build_notes(hint: str | None) -> str | None:
    """Per-turn notes for an owner turn: cameras (always), room transcript (always), remembered items."""
    bits = []
    appr = approved_notes()
    if appr:
        bits.append(appr)
    lines = scene_lines()
    if lines:
        bits.append("Camera notes (newest per camera): " + " | ".join(lines[:4]))
    room = room_transcript_line()
    if room:
        bits.append(room)
    mem = remembered.items(MEM_PATH)
    if mem:
        bits.append(f"Things {_owner()} asked you to remember: " + "; ".join(mem[-8:]))
    return " ".join(bits) or None


# ---------------------------------------------------------------- call session ----
class Call:
    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws
        self.stream_sid: str | None = None
        self.call_sid = "unknown"
        self.from_number = ""  # the other party's number: who called in, or whom we called
        self.direction = "in"  # "out" for a call we placed
        self.to_name: str | None = None  # whom the owner asked to call, by name, for a call we placed
        self.instructions: str | None = None  # the owner's words for the call, when it had a mission
        self.state = "listening"
        self.history: list[dict[str, str]] = []
        self.noise = 300.0
        self.buf: list[np.ndarray] = []  # frames of current utterance (int16, 8 kHz)
        self.pre: list[np.ndarray] = []  # rolling pre-roll
        self.speech_run = 0
        self.silence_run = 0
        self.in_speech = False
        self.turn_frames = 0  # frames of the utterance handle_utterance is working on
        self.went_on_run = 0  # turns in a row that dropped their reply because the caller went on (MAX_WENT_ON)
        self.mark_expect: str | None = None
        self.mark_event = asyncio.Event()
        self.started = time.time()
        self.ended: float | None = None
        self.dir: Path | None = None
        self.in_frames: list[np.ndarray] = []  # every caller frame, always (full call)
        self.in_n = 0  # frames received so far (clock for bot audio placement)
        self.bot_segments: list[tuple[int, np.ndarray]] = []  # (start_sample, int16 audio)
        self.transcript: list[dict[str, Any]] = []
        self.speakers: dict[str, int] = {}
        self.saved = False
        self.utt_end_at: float | None = None
        self.system = system_prompt()
        self.last_user: str | None = None
        self.owner = False
        self.verified_owner = False  # the owner's number, attested (or the number we dialed): label them as the owner
        self.awaiting_password = False  # we just asked for the password, so the next thing said is a try
        self.say: str | None = None
        self.interrupted, self.barge_run, self.speak_peak = False, 0, 0.0
        self.sent_audio = False  # this reply's first audio has gone out (barge-in only counts after it)
        self.voicemail = False
        self.reply_model: str | None = None
        self.route = "agent"  # how this turn's reply was made: agent, fast (local model on an owner call), or local
        self.unseen: list[dict[str, str]] = []  # fast exchanges the phone agent hasn't seen; passed with its next turn
        self.timing: dict[str, Any] = {}  # this turn's marks (end of speech, ASR, reply), logged at its first audio
        self.timings: list[dict[str, Any]] = []  # every answered turn's timing, saved in call.json

    @property
    def caller_default(self) -> str:
        if self.verified_owner or self.owner:
            return outbound_call.owner_phone_label()
        return f"Caller {self.from_number[-4:]}" if self.from_number else "Caller"

    def open_dir(self) -> None:
        now = dt.datetime.now()
        self.dir = CALLS_DIR / now.strftime("%Y-%m-%d") / f"{now.strftime('%H%M%S')}_{self.call_sid}"
        os.makedirs(self.dir, mode=0o700, exist_ok=True)
        os.chmod(self.dir, 0o700)
        log.info("[%s] recording to %s", self.call_sid, self.dir)
        self.mark_live()

    def mark_live(self) -> None:
        """`live.json` in the call's folder while it's on, for the dashboard's "on a call" (GET /calls/live): which way,
        the other party, since when, and this process, so a call a crash left behind isn't on forever."""
        if not self.dir:
            return
        name = (
            self.to_name if self.direction == "out" else ((cfg.owner_name() or None) if self.verified_owner else None)
        )
        live = {
            "callSid": self.call_sid,
            "direction": self.direction,
            "number": self.from_number,
            "name": name,
            "since": self.started,
            "pid": os.getpid(),
        }
        try:
            json_state.write(self.dir / LIVE_FILE, live, private=True)
        except OSError as e:
            log.warning("[%s] couldn't mark the call live: %s", self.call_sid, e)

    def end_live(self) -> None:
        """The call is over, however it ended: no longer on."""
        if self.dir:
            (self.dir / LIVE_FILE).unlink(missing_ok=True)

    def log_turn(self, speaker: str, text: str, extra: dict[str, Any] | None = None) -> None:
        entry = {"t": round(time.time() - self.started, 2), "speaker": speaker, "text": text}
        if extra:
            entry.update(extra)
        self.transcript.append(entry)
        if self.dir:
            with open(self.dir / "transcript.jsonl", "a") as f:
                f.write(json.dumps(entry) + "\n")

    def unlog_caller_turn(self) -> None:
        """Take back the caller's last turn: a half-utterance whose reply was dropped because they went on (see
        went_on). The whole utterance is logged when it is answered, so the transcript shows it once."""
        last = next(
            (i for i in range(len(self.transcript) - 1, -1, -1) if self.transcript[i]["speaker"] == "caller"), None
        )
        if last is None:
            return
        entry = self.transcript.pop(last)
        label = entry.get("speaker_label")
        if isinstance(label, str) and self.speakers.get(label):
            self.speakers[label] -= 1
        if self.dir:
            lines = "".join(json.dumps(e) + "\n" for e in self.transcript)
            json_state.write_text(self.dir / "transcript.jsonl", lines)

    def going_on(self) -> bool:
        """Whether this turn should drop its reply for the speech that started while it was made (see went_on): the
        caller is talking, fewer than MAX_WENT_ON turns in a row have dropped theirs, and this turn's frames plus the
        new ones still fit in one utterance (MAX_UTT_FRAMES). Otherwise the turn answers what it has, and the new
        speech, kept by speak(), becomes the next utterance."""
        return self.in_speech and self.went_on_run < MAX_WENT_ON and self.turn_frames + len(self.buf) <= MAX_UTT_FRAMES

    def went_on(self, frames: list[np.ndarray], *, decided: bool = False) -> bool:
        """Whether the caller went on talking while this turn was made (on_media keeps listening while thinking, and
        going_on decides). If so, the turn's frames go back in front of the new speech and the call listens again, so
        the whole utterance is answered once they stop: a 1.2 s mid-sentence pause otherwise
        gets a reply to the first half while the caller is still finishing. On an owner call the agent may already have
        answered or acted on the first half; only its reply is dropped. decided: going_on already said so (reply_turn
        returned None), so drop regardless."""
        if not decided and not self.going_on():
            return False
        log.info("[%s] caller went on after a pause; listening for the rest", self.call_sid)
        self.buf = frames + self.buf
        self.state = "listening"
        self.went_on_run += 1
        return True

    async def send(self, obj: dict[str, Any]) -> None:
        await self.ws.send_text(json.dumps(obj))

    async def send_audio(self, audio: np.ndarray) -> None:
        payload = ulaw_encode(audio)
        for i in range(0, len(payload), FRAME):
            if self.interrupted:
                await self.send({"event": "clear", "streamSid": self.stream_sid})
                break
            await self.send(
                {
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {"payload": base64.b64encode(payload[i : i + FRAME]).decode()},
                }
            )
            await asyncio.sleep(0.018)

    def log_timing(self, now: float) -> None:
        """One answered turn's lag, by stage: end of speech to transcript, transcript to reply, reply to first audio."""
        t = self.timing
        end = t.get("end", now)
        asr, reply = t.get("asr", end), t.get("reply", end)
        row = {
            "route": t.get("route"),
            "asr_s": round(asr - end, 2),
            "reply_s": round(reply - asr, 2),
            "voice_s": round(now - reply, 2),
            "total_s": round(now - end, 2),
        }
        self.timings.append(row)
        self.timing = {}
        log.info(
            "[%s] turn timing: %s reply, asr %.2fs, reply %.2fs, voice %.2fs, total %.2fs",
            self.call_sid,
            row["route"],
            row["asr_s"],
            row["reply_s"],
            row["voice_s"],
            row["total_s"],
        )

    async def speak(self, text: str, then: str = "listening") -> None:
        """Say text, then go to state then ("thinking" for a filler while the turn is still being made). Speech the
        caller started before this (while the turn was made) is kept and goes on into their next utterance."""
        self.state = "speaking"
        self.log_turn("bot", text, {"speaker_label": cfg.assistant_name()})
        log.info("[%s] speaking: %s", self.call_sid, text)
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[np.ndarray | None] = asyncio.Queue()

        def on_chunk(a: np.ndarray) -> None:
            loop.call_soon_threadsafe(q.put_nowait, a)

        def produce() -> None:
            try:
                synthesize_stream(text, on_chunk)
            except Exception as e:
                log.exception("[%s] tts stream failed: %s", self.call_sid, e)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        cached = FIXED_AUDIO.get(text)
        if cached is not None:
            q.put_nowait(cached)
            q.put_nowait(None)
        else:
            MLX_EXEC.submit(produce)
        base, pos, first = self.in_n * FRAME, 0, True
        self.interrupted, self.barge_run, self.speak_peak = False, 0, 0.0
        self.sent_audio = False  # this reply's first audio has gone out (barge-in only counts after it)
        while True:
            a = await q.get()
            if a is None:
                break
            if self.interrupted:  # drop the rest of this reply; flush what Twilio has buffered
                try:
                    await self.send({"event": "clear", "streamSid": self.stream_sid})
                except Exception:
                    pass
                while not q.empty() and q.get_nowait() is not None:
                    pass
                self.log_turn("bot", "(interrupted)", {"speaker_label": cfg.assistant_name()})
                break
            if first and "reply" in self.timing:
                self.log_timing(time.time())
            if first and self.utt_end_at is not None:
                log.info(
                    "[%s] first audio %.2fs after end of caller speech", self.call_sid, time.time() - self.utt_end_at
                )
                self.utt_end_at = None
            first = False
            self.sent_audio = True
            self.bot_segments.append((base + pos, a))
            pos += len(a)
            await self.send_audio(a)
        mark = f"end-{uuid.uuid4().hex[:8]}"
        self.mark_expect = mark
        self.mark_event.clear()
        await self.send({"event": "mark", "streamSid": self.stream_sid, "mark": {"name": mark}})
        try:
            await asyncio.wait_for(self.mark_event.wait(), timeout=20)
        except TimeoutError:
            log.warning("[%s] mark timeout; resuming listening", self.call_sid)
        log.info(
            "[%s] caller peak rms while %s spoke: %.0f (barge threshold %.0f)",
            self.call_sid,
            cfg.assistant_name(),
            self.speak_peak,
            BARGE_RMS,
        )
        self.state = then
        if not self.in_speech:
            self.buf = []  # keep self.pre: words that started in Bob's last 200 ms are not lost
        self.interrupted = False

    async def greet(self) -> None:
        """Play fixed identity first while the mission opening is generated off the receive loop."""
        opening: asyncio.Task[str] | None = None
        try:
            if self.say:
                lead = mission_disclosure(prior_consent=self.verified_owner)
                opening = asyncio.create_task(
                    asyncio.to_thread(mission_opener, self.say, prior_consent=self.verified_owner)
                )
                self.history.append({"role": "assistant", "content": lead.strip()})
                await self.speak(lead.strip(), then="thinking")
                full = await opening
                # Barge-in during the intro is a real question, not permission to speak over the caller.
                if self.going_on():
                    self.state = "listening"
                    return
                rest = full.removeprefix(lead).strip()
                if rest:
                    self.history.append({"role": "assistant", "content": rest})
                    await self.speak(rest)
                else:
                    self.state = "listening"
            else:
                text = take_greeting_once() if self.owner else None
                await self.speak(text or (greeting() if self.owner else greeting_other()))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("[%s] opening unavailable: %s", self.call_sid, type(e).__name__)
            self.state = "listening"
        finally:
            if opening is not None and not opening.done():
                opening.cancel()
                await asyncio.gather(opening, return_exceptions=True)

    def system_for_turn(self, hint: str | None = None) -> str:
        if not self.owner:
            return mission_system(self.say) if self.say else system_prompt()
        parts = [owner_system(), owner_context()]
        mem = remembered.items(MEM_PATH)
        if mem:
            parts.append(f"Things {_owner()} asked you to remember:\n" + "\n".join(f"- {m}" for m in mem[-20:]))
        lines = scene_lines()
        if lines:
            parts.append("Camera notes right now (newest per camera):\n" + "\n".join(f"- {l}" for l in lines))
        parts.append("Time now: " + time.strftime("%A %H:%M"))
        if hint:
            parts.append("This turn: " + hint)
        return "\n\n".join(parts)

    async def say_and_hang_up(self, text: str) -> None:
        try:
            await asyncio.wait_for(self.speak(text), timeout=5 if text == "Okay, bye!" else 30)
        except Exception as e:
            log.warning("[%s] farewell unavailable; still ending the call: %s", self.call_sid, type(e).__name__)
        try:
            status = await asyncio.to_thread(hangup_call, self.call_sid)
            log.info("[%s] caller asked to hang up; provider status: %s", self.call_sid, status)
        except Exception as e:
            log.warning("[%s] hang-up failed: %s", self.call_sid, e)
        try:
            await self.ws.close()
        except Exception:
            pass

    async def play_clip(self, audio: np.ndarray, then: str = "listening") -> None:
        """Play an already-rendered clip and wait for Twilio's mark, same rules as speak()."""
        self.state = "speaking"
        self.bot_segments.append((self.in_n * FRAME, audio))
        await self.send_audio(audio)
        mark = f"end-{uuid.uuid4().hex[:8]}"
        self.mark_expect = mark
        self.mark_event.clear()
        await self.send({"event": "mark", "streamSid": self.stream_sid, "mark": {"name": mark}})
        try:
            await asyncio.wait_for(self.mark_event.wait(), timeout=5 + len(audio) / SR_PHONE)
        except TimeoutError:
            log.warning("[%s] filler mark timeout", self.call_sid)
        self.state = then
        if not self.in_speech:
            self.buf, self.pre = [], []

    async def fast_owner_reply(self, text: str) -> str | None:
        """An owner turn answered by the fast local model, with this call's history, or None to ask the phone agent
        (the model failed, deferred, or made a claim it can't keep)."""
        t0 = time.time()
        history = [*self.history[-10:], {"role": "user", "content": text}]
        try:
            reply = await asyncio.to_thread(ask_llm, history, fast_system(self.system_for_turn(None)))
        except Exception as e:
            log.warning("[%s] fast reply unavailable (%s); asking the agent", self.call_sid, type(e).__name__)
            return None
        # "Let me know if..." is an offer, not narration: bad_reply's "let me" sent every polite sign-off to the agent.
        checked = re.sub(r"\blet me know\b", "tell me", reply, flags=re.I)
        if (
            bad_reply(checked)
            or DEFERS.search(reply)
            or FAST_PROMISE.search(checked)
            or CAPABILITY_CLAIM.search(reply)
            or OWNER_CLAIM.search(reply)
        ):
            log.info(
                "[%s] fast reply deferred to the agent after %.2fs: %r", self.call_sid, time.time() - t0, reply[:80]
            )
            return None
        log.info("[%s] fast reply in %.2fs: %r", self.call_sid, time.time() - t0, reply[:120])
        self.reply_model = LLM
        self.route = "fast"
        return sanitize_spoken(reply)

    def agent_notes(self, hint: str | None) -> str | None:
        """The agent's per-turn notes, with what was said in fast turns it didn't see."""
        notes = build_notes(hint)
        if self.unseen:
            said = " | ".join(f"{m['role']}: {m['content']}" for m in self.unseen[-8:])
            notes = f"{notes} " if notes else ""
            notes += f"Earlier in this call, answered without you: {said}"
            self.unseen = []
        return notes

    async def owner_agent_reply(self, text: str, hint: str | None) -> str:
        self.route = "agent"
        if fast_replies() and not hint and not NEEDS_AGENT.search(text):
            fast = await self.fast_owner_reply(text)
            if fast is not None:
                return fast
        t0 = time.time()
        task = asyncio.create_task(
            asyncio.to_thread(phone_agent_reply, self.call_sid, text, self.agent_notes(hint), 240)
        )
        done, _ = await asyncio.wait({task}, timeout=3.0)
        if not done:
            if FILLER_AUDIO is not None:
                await self.play_clip(FILLER_AUDIO, then="thinking")
            else:
                await self.speak("One moment.", then="thinking")
            done, _ = await asyncio.wait({task}, timeout=15.0)
            if not done:
                if STILL_AUDIO is not None:
                    await self.play_clip(STILL_AUDIO, then="thinking")
                else:
                    await self.speak("Still working on it.", then="thinking")
        try:
            reply = await task
            log.info("[%s] agent reply in %.1fs: %r", self.call_sid, time.time() - t0, reply[:120])
            if bad_reply(reply):
                log.warning("[%s] unusable reply %r; retrying in a fresh session", self.call_sid, reply[:60])
                reply = await asyncio.to_thread(
                    phone_agent_reply,
                    f"{self.call_sid}-retry",
                    f"Answer with a real result. Use the exec or web tool to get it. Never write a command or tool call as text, never say you are checking. {_owner()} asked: "
                    + text,
                    build_notes(hint),
                    240,
                )
                if bad_reply(reply):
                    reply = "I didn't get a result for that. Ask me again?"
        except Exception as e:
            log.warning("[%s] phone agent unavailable (%s); answering locally", self.call_sid, e)
            self.reply_model = LLM
            reply = await asyncio.to_thread(ask_llm, [{"role": "user", "content": text}], self.system_for_turn(hint))
        reply = sanitize_spoken(reply)
        if OWNER_CLAIM.search(reply):
            log.warning("[%s] blocked owner-turn claim: %r", self.call_sid, reply)
            reply = "I can't do that on this call."
        return reply

    async def reply_turn(self, text: str, hint: str | None, *, record_history: bool = True) -> str | None:
        """The normal model turn, after authentication/control handling, with best-effort local history. None when the
        caller went on talking before or while the reply was made (Call.went_on): the reply and its history entry are
        dropped, and handle_utterance listens for the rest. On an owner call the agent session has by then seen the
        first half, and gets the whole utterance next."""
        if self.going_on():
            return None  # they went on before the model was asked: skip the round trip
        started = time.time()
        self.reply_model = None if self.owner else LLM
        self.route = "agent" if self.owner else "local"
        mode = "agent" if self.owner else "local"
        saved: tuple[ask_history.History, str] | None = None

        def start() -> tuple[ask_history.History, str]:
            history = ask_history.at(STATE_DIR)
            exchange = history.start(
                text, surface="call", scope="general", mode=mode, call_id=self.call_sid, created_at=started
            )
            return history, exchange

        async def finish(reply: str, status: str) -> None:
            if saved is None:
                return
            history, exchange = saved
            result = {
                "status": {"completed": "ok", "cancelled": "cancelled"}.get(status, "error"),
                "answer": reply,
                "model": self.reply_model,
                "evidence": [],
            }
            try:
                await asyncio.to_thread(
                    history.finish, exchange, result, started_at=started, finished_at=time.time(), status=status
                )
            except Exception as error:
                log.warning("[%s] Ask history completion unavailable (%s)", self.call_sid, type(error).__name__)

        if record_history:
            try:
                saved = await asyncio.to_thread(start)
            except Exception as error:
                log.warning("[%s] Ask history unavailable (%s)", self.call_sid, type(error).__name__)
        try:
            if self.owner:
                reply = await self.owner_agent_reply(text, hint)
                if self.going_on():
                    await finish("", "cancelled")
                    return None
                exchange = [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
                self.history += exchange  # the fast path's context, whichever answered
                if self.route == "fast":
                    self.unseen += exchange
            else:
                self.history.append({"role": "user", "content": text})
                reply = await asyncio.to_thread(ask_llm, self.history[-12:], self.system_for_turn(hint))
                if self.going_on():
                    self.history.pop()
                    await finish("", "cancelled")
                    return None
                if re.search(
                    r"text-based|I'?m just an? (?:text |AI )?assistant|not something I can do on a phone", reply, re.I
                ):
                    reply = (
                        re.sub(
                            r"\s*(?:I'?m just an? (?:text-based |text |AI )?assistant\.?|that'?s not something I can do on a phone call\.?)",
                            "",
                            reply,
                            flags=re.I,
                        ).strip()
                        or "Sorry, I can't do that from here."
                    )
                if CAPABILITY_CLAIM.search(reply):
                    log.warning("[%s] blocked capability claim: %r", self.call_sid, reply)
                    reply = CAPABILITY_FALLBACK
                self.history.append({"role": "assistant", "content": reply})
        except Exception:
            await finish("Call reply failed.", "failed")
            raise
        await finish(reply, "completed")
        self.timing["reply"] = time.time()
        self.timing["route"] = self.route
        return reply

    def try_password(self, text: str, right: bool | None = None) -> str:
        """Check a spoken password against the voice password (right, when already checked), counting a wrong one
        toward the same lockouts as in person (per caller number, and for everyone). A right one makes this call an
        owner call, for this call only."""
        self.awaiting_password = False
        if voice_auth.Conversation.normal(text) in voice_auth.CANCEL:
            return "Okay."
        lockouts = auth().lockouts
        who = "phone:" + (numbers.e164(self.from_number) or "unknown")
        if lockouts.locked(who):
            return LOCKED_OUT
        if password_matches(text) if right is None else right:
            lockouts.succeed(who)
            self.owner = True
            log.info("[%s] the voice password unlocked owner treatment for this call", self.call_sid)
            notice = take_lockout_notice()  # the owner often calls without attestation, so tell them here too
            return f"Thanks, {_owner()}. Go ahead." + (f" {notice}" if notice else "")
        if lockouts.fail(who):
            notice_lockout(self.from_number, time.time())
            return LOCKED_OUT
        return "That isn't the password."

    async def handle_utterance(self, frames: list[np.ndarray]) -> None:
        self.state = "thinking"
        self.turn_frames, went_on_before = len(frames), self.went_on_run
        try:
            pcm = np.concatenate(frames)
            pcm16k = resample_poly(pcm.astype(np.float32), 2, 1)
            self.timing = {"end": self.utt_end_at or time.time()}
            text, (label, sim) = await asyncio.gather(
                run_mlx(transcribe, pcm16k), asyncio.to_thread(identify_speaker, pcm16k)
            )
            self.timing["asr"] = time.time()
            # Right after we asked for the password, the next thing said is a try: the password, or a wrong one unless it
            # is plainly a question or a request ("tell him I called"), which shouldn't use up a try.
            said = bool(voice_auth.PASSWORD_SAID.match(text))
            maybe = said or self.awaiting_password
            right = maybe and await asyncio.to_thread(password_matches, text)  # a slow hash: off the event loop
            # Before anything is logged, tried, or said for the first half. Never for a right password: the whole line,
            # heard again, would fail the exact check, count as a wrong try, and be logged with the password in it.
            # What they went on to say stays buffered (speak keeps it) and is the next utterance.
            if not right and self.went_on(frames):
                return
            asked = voice_auth.QUESTION.match(text) or voice_auth.REQUEST.match(text)
            attempt = said or right or (self.awaiting_password and not asked)
            self.awaiting_password = self.awaiting_password and attempt
            # The password itself is never written down; anything else is.
            shown = voice_auth.PASSWORD_SHOWN if right else text
            log.info("[%s] heard (%s): %r", self.call_sid, label or self.caller_default, shown)
            if len(text.strip(" .?!")) < 2:
                self.state = "listening"
                return
            speaker = label or self.caller_default
            self.speakers[speaker] = self.speakers.get(speaker, 0) + 1
            self.log_turn("caller", shown, {"speaker_label": speaker, "match_sim": sim})
            if attempt and not self.owner:
                await self.speak(self.try_password(text, right))  # never passed on to the model
                return
            if right:
                await self.speak("You're already through.")
                return
            if not self.owner and VOICEMAIL_RE.search(text):
                # Voicemail/IVR, not a person (without this check, Bob can argue with a contact's voicemail). Leave the
                # message once after the prompt finishes, then hang up. Never answer menu prompts.
                if self.voicemail:
                    self.state = "listening"
                    return
                self.voicemail = True
                log.info("[%s] voicemail detected: %r", self.call_sid, text)
                await asyncio.sleep(VOICEMAIL_WAIT_S)
                await self.say_and_hang_up(
                    await asyncio.to_thread(voicemail_message, self.say, prior_consent=self.verified_owner)
                )
                return
            if RECALL_LAST.match(text):
                prev = (self.last_user or "").strip()
                reply = (
                    ("You asked: " + (prev if prev[-1:] in ".?!" else prev + "."))
                    if prev
                    else "You haven't asked me anything yet."
                )
                self.last_user = text
                self.history.append({"role": "user", "content": text})
                self.history.append({"role": "assistant", "content": reply})
                await self.speak(reply)
                return
            self.last_user = text
            if asks_to_hang_up(text) or said_goodbye(text):
                # They said goodbye, so say it back now instead of asking the model: waiting behind other Ollama
                # work can stall the reply for seconds until the caller hangs up on the silence.
                await self.say_and_hang_up("Okay, bye!")
                return
            # Off the event loop: placing a call waits on Twilio's API, and the approval and memory files are
            # blocking reads too. Every other call and this call's media stream keep going meanwhile.
            try:
                direct, hint = await asyncio.wait_for(
                    asyncio.to_thread(owner_turn, text, self.owner), OWNER_TURN_TIMEOUT_S
                )
            except TimeoutError:
                log.warning("[%s] owner turn took over %.0f s: %r", self.call_sid, OWNER_TURN_TIMEOUT_S, text)
                direct, hint = OWNER_TURN_SLOW, None
            if direct is not None:
                self.awaiting_password = not self.owner and direct == owner_only()
                await self.speak(direct)
                return
            reply = await self.reply_turn(text, hint, record_history=not attempt)
            if reply is None:  # they went on while the reply was made: answer all of it once they stop
                self.unlog_caller_turn()
                self.went_on(frames, decided=True)
                return
            if not self.owner and (asks_to_hang_up(text) or (self.say and said_goodbye(reply))):
                await self.say_and_hang_up(reply if said_goodbye(reply) else "Okay, bye.")  # the caller said goodbye
                return
            await self.speak(reply)
        except Exception as e:
            log.exception("[%s] turn failed: %s", self.call_sid, e)
            self.state = "listening"
        finally:
            if self.went_on_run == went_on_before:  # this turn was answered, not dropped
                self.went_on_run = 0

    async def on_media(self, payload_b64: str) -> None:
        pcm = ulaw_decode(base64.b64decode(payload_b64))
        self.in_frames.append(pcm)  # always record the caller, even while Bob talks
        self.in_n += 1
        rms = float(np.sqrt(np.mean((pcm.astype(np.float32) / 32768.0) ** 2))) * 32768.0
        thr = max(250.0, self.noise * 3.5)
        if self.state == "speaking":
            # Barge-in: the caller talking over Bob for BARGE_FRAMES (echo of our own audio is far quieter than a voice
            # at the handset) stops playback; the frames heard meanwhile become the start of the next utterance.
            self.speak_peak = max(self.speak_peak, rms)
            self.pre = (self.pre + [pcm])[-10:]
            # Before Bob's first audio goes out the caller hears silence and says "hello?": that isn't talking over
            # Bob, and taking it as a barge-in drops the whole greeting unheard.
            loud = self.sent_audio and rms > max(thr, BARGE_RMS)
            self.barge_run = self.barge_run + 1 if loud else 0
            if self.barge_run >= BARGE_FRAMES and not self.interrupted:
                self.interrupted = True
                # Speech carried in from while the turn was made (see speak) stays in front.
                self.buf = (self.buf if self.in_speech else []) + list(self.pre)
                self.in_speech, self.silence_run = True, 0
                log.info("[%s] barge-in: caller spoke over %s (rms %.0f)", self.call_sid, cfg.assistant_name(), rms)
            elif self.interrupted and self.in_speech:
                self.buf.append(pcm)  # keep capturing while playback drains
            return
        if self.state not in ("listening", "thinking"):
            return
        # Keep listening while a reply is made (see Call.went_on): speech that starts while thinking is buffered as
        # usual, and the turn then drops its reply and answers the whole of it.
        thinking = self.state == "thinking"
        if not self.in_speech:
            self.noise = 0.97 * self.noise + 0.03 * rms
            self.pre = (self.pre + [pcm])[-10:]
            self.speech_run = self.speech_run + 1 if rms > thr else 0
            if self.speech_run >= (RESUME_FRAMES if thinking else START_FRAMES):
                self.in_speech = True
                self.buf = list(self.pre)
                self.silence_run = 0
            return
        if not thinking or len(self.buf) < MAX_UTT_FRAMES:  # an owner turn can think for minutes: cap what it keeps
            self.buf.append(pcm)
        self.silence_run = 0 if rms > thr else self.silence_run + 1
        if thinking:
            return  # where it ends is found once the turn has dropped its reply and listens again (Call.went_on)
        if self.silence_run >= END_SILENCE_FRAMES or len(self.buf) >= MAX_UTT_FRAMES:
            frames = self.buf
            self.buf, self.pre, self.in_speech, self.speech_run = [], [], False, 0
            if len(frames) < MIN_UTT_FRAMES:
                return
            self.state = "thinking"
            self.utt_end_at = time.time()
            asyncio.create_task(self.handle_utterance(frames))

    def outcome(self) -> str | None:
        """What came of a call with instructions: the assistant's last line, which its mission asks to sum up what was
        agreed. None for a call with no instructions."""
        if not self.instructions:
            return None
        said = [e for e in self.transcript if e.get("speaker") == "bot" and str(e.get("text") or "").strip()]
        return str(said[-1]["text"]).strip()[:300] if said else None

    def finalize(self) -> None:
        """Write recordings, transcript and call.json. Safe to call twice."""
        if self.saved or not self.dir or not self.in_frames:
            return
        self.saved = True
        self.ended = time.time()
        caller = np.concatenate(self.in_frames).astype(np.float32)
        # cover every bot segment even if the inbound track ended early (short or dropped streams)
        need = max([len(caller)] + [st + len(sg) for st, sg in self.bot_segments])
        caller = np.pad(caller, (0, need - len(caller)))
        n = len(caller)
        bot = np.zeros(n, dtype=np.float32)
        for start, seg in self.bot_segments:
            seg = seg.astype(np.float32)
            if start >= n:
                continue
            end = min(n, start + len(seg))
            bot[start:end] += seg[: end - start]
        mix = np.clip(caller + bot, -32768, 32767).astype(np.int16)
        sf.write(self.dir / "caller.wav", np.clip(caller, -32768, 32767).astype(np.int16), SR_PHONE, subtype="PCM_16")
        sf.write(self.dir / "bot.wav", np.clip(bot, -32768, 32767).astype(np.int16), SR_PHONE, subtype="PCM_16")
        sf.write(self.dir / "mixed.wav", mix, SR_PHONE, subtype="PCM_16")
        with open(self.dir / "transcript.txt", "w") as f:
            for e in self.transcript:
                who = e.get("speaker_label") or (cfg.assistant_name() if e["speaker"] == "bot" else "Caller")
                f.write(f"[{e['t']:7.2f}s] {who}: {e['text']}\n")
        meta = {
            "callSid": self.call_sid,
            "from": self.from_number,
            # Which way it went, and for one we placed, whom and what for (the Calls notes). `from` stays the
            # other party's number either way, as before.
            "direction": self.direction,
            "to": {"name": self.to_name, "number": self.from_number} if self.direction == "out" else None,
            "instructions": self.instructions,
            "outcome": self.outcome(),
            "started": dt.datetime.fromtimestamp(self.started).isoformat(timespec="seconds"),
            "ended": dt.datetime.fromtimestamp(self.ended).isoformat(timespec="seconds"),
            "durationSeconds": round(n / SR_PHONE, 1),
            "turns": len(self.transcript),
            "turnTimings": self.timings,
            "speakers": self.speakers,
            "dir": self.dir.name,
        }
        (self.dir / "call.json").write_text(json.dumps(meta, indent=2))
        os.chmod(self.dir / "call.json", 0o600)
        log.info(
            "[%s] saved %.1fs recording + %d turns to %s", self.call_sid, n / SR_PHONE, len(self.transcript), self.dir
        )


def mission_disclosure(*, prior_consent: bool = False) -> str:
    """Identify the assistant; keep the recording notice for recipients without prior consent."""
    if prior_consent:
        return greeting() + " "
    a, name = cfg.assistant_name(), cfg.owner_name()
    if name:
        return f"Hi, this is {a}, calling for {name}. This call is recorded. "
    return f"Hi, this is {a}. This call is recorded. "


def voicemail_message(say: str | None, *, prior_consent: bool = False) -> str:
    """Leave a concise goal-directed message, not the private instructions."""
    guidance = say or f"{cfg.owner_or('Someone')} wanted to reach you and will follow up."
    message = ask_llm(
        [{"role": "user", "content": "Voicemail answered. Leave a brief relevant message, then say goodbye."}],
        mission_system(guidance),
    )
    return mission_disclosure(prior_consent=prior_consent) + message


MISSION_INSTR_RE = re.compile(r"^(?:please\s+)?(tell|ask|remind|let)\s+(?:him|her|them)\b", re.I)


def mission_opener(say: str, *, prior_consent: bool = False) -> str:
    """Generate an opening from private task guidance instead of reading it verbatim."""
    # The model hears the disclosure it follows. Told only to "start the conversation", it greets and introduces itself
    # a second time and runs long (37 words and a second "it's <assistant>"). On the phone LLM, 31 sampled openers over
    # 13 missions went from 24 second introductions to 0, and from 13 over 30 words (disclosure included) to 0.
    lead = mission_disclosure(prior_consent=prior_consent)
    heard = lead.strip()
    opening = ask_llm(
        [
            {
                "role": "user",
                "content": f'The recipient answered and has just heard you say: "{heard}" Continue straight from that. '
                "Don't greet them or say who you are again. In one or two short sentences, 15 words at most, "
                "get to the point: why you're calling, or the first question the goal needs answered.",
            }
        ],
        mission_system(say),
    )
    return lead + opening


def take_mission(number: str) -> dict[str, Any] | None:
    """The mission for a number (what to say, and the owner's words and whom they asked for), consumed once."""
    try:
        from liberational.phone import outbound as _oc

        return _oc.take_mission_record(number)
    except Exception:
        return None


def twiml_connect(token: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response><Connect>"
        f'<Stream url="wss://{public_host()}/bridge/media-stream">'
        f'<Parameter name="token" value="{xml_escape(token)}"/>'
        "</Stream></Connect></Response>"
    )


app = FastAPI()


@app.get("/bridge/health")
async def health() -> dict[str, bool]:
    return {"ok": True, "asr": _asr is not None, "tts": _tts is not None, "speakers": _spk is not None}


@app.api_route("/bridge/voice", methods=["POST"])
async def voice(request: Request) -> Response:
    form = dict(await request.form())
    if not public_host():
        log.warning("refused voice webhook: phone.public_host is not set, so Twilio signatures can't be checked")
        return Response(status_code=503)
    if not twilio_signature_ok(public_voice_urls()[0], form, request.headers.get("X-Twilio-Signature", "")):
        log.warning("rejected voice webhook without a valid Twilio signature")
        return Response(status_code=403)
    call_sid = str(form.get("CallSid", ""))
    if not call_sid:
        return Response(status_code=400)
    now = time.time()
    for old in [k for k, v in PENDING.items() if now - v["t"] > PENDING_TTL_S]:
        PENDING.pop(old, None)
    # Twilio sends a setup again when the first went unanswered (a timeout, or its fallback URL), with the same
    # CallSid. While that call's media stream hasn't started, answer it the same way: the same token, and the message
    # the first setup already took. Once it has started, or for a call set up before, it can only be a replay.
    retry = next((tok for tok, rec in PENDING.items() if rec["callSid"] == call_sid), None)
    if retry is not None:
        return Response(content=twiml_connect(retry), media_type="text/xml")
    if not first_setup(call_sid, now):
        log.warning("refused a second setup for call %s: a replayed webhook", call_sid)
        return Response(status_code=403)
    remote, outbound, attested, is_owner = answer_call(form)
    token = secrets.token_urlsafe(24)
    # A mission always wins on calls WE placed, including to the owner (otherwise "call <owner> and tell them a joke"
    # reaches the owner and asks "what do you need?"). Owner mode = the owner calling in, not us calling the owner.
    # An inbound caller (calling back) gets a message left for their number only when it is vouched for: a spoofed
    # number must not collect it.
    mission = take_mission(remote) if (outbound or (attested and not is_owner)) else None
    say = mission.get("say") if mission else None
    PENDING[token] = {
        "callSid": call_sid,
        "remote": remote,
        "owner": is_owner and not say,
        "verified": is_owner,
        "say": say,
        "direction": "out" if outbound else "in",
        "to_name": (mission or {}).get("name") or None,
        "instructions": (mission or {}).get("message") or None,
        "t": now,
    }
    return Response(content=twiml_connect(token), media_type="text/xml")


@app.websocket("/bridge/media-stream")
async def media_stream(ws: WebSocket) -> None:
    await ws.accept()
    call = Call(ws)
    greeted = False
    greeting_task: asyncio.Task[None] | None = None
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            ev = msg.get("event")
            if ev == "start":
                call.stream_sid = msg["start"]["streamSid"]
                call.call_sid = msg["start"].get("callSid", "unknown")
                tok = (msg["start"].get("customParameters") or {}).get("token", "")
                rec = PENDING.pop(tok, None)
                if rec is None or time.time() - rec["t"] > PENDING_TTL_S or rec["callSid"] != call.call_sid:
                    log.warning("[%s] refused media stream: no valid token", call.call_sid)
                    await ws.close(code=1008)
                    return
                call.from_number = rec["remote"]
                owner = rec["owner"]
                call.owner = owner
                call.verified_owner = rec["verified"]
                call.say = rec["say"]
                call.direction = rec.get("direction", "in")
                call.to_name = rec.get("to_name")
                call.instructions = rec.get("instructions")
                call.open_dir()
                log.info("[%s] stream start %s from %s", call.call_sid, call.stream_sid, call.caller_default)
                if not greeted:
                    greeted = True
                    if owner:
                        asyncio.create_task(asyncio.to_thread(warm_phone_session, call.call_sid))
                    call.state = "thinking"
                    greeting_task = asyncio.create_task(call.greet())
            elif ev == "media" and msg["media"].get("track", "inbound") == "inbound":
                await call.on_media(msg["media"]["payload"])
            elif ev == "mark":
                if msg["mark"].get("name") == call.mark_expect:
                    call.mark_event.set()
            elif ev == "stop":
                log.info("[%s] stream stop", call.call_sid)
                break
    except WebSocketDisconnect:
        log.info("[%s] disconnected", call.call_sid)
    except Exception as e:
        log.exception("media stream error: %s", e)
    finally:
        if greeting_task is not None and not greeting_task.done():
            greeting_task.cancel()
            await asyncio.gather(greeting_task, return_exceptions=True)
        call.end_live()  # first: finalize skips a call with no audio, and a call that's over must not stay on
        call.finalize()


@model_runtime.reporting("phone-bridge")
def main() -> None:
    if not public_host():
        log.warning(
            "phone.public_host is not set: Twilio can't reach this bridge and voice webhooks will be refused. "
            "Set it in pack.toml for real calls; the bridge still starts for local testing."
        )
    owner_phone = cfg.get("owner.phone").strip()
    if not owner_phone or not cfg.owner_name():
        log.info("owner.phone and owner.name are not both set: no caller gets owner treatment")
    elif numbers.e164(owner_phone) is None:
        log.warning(
            "owner.phone %r will never match a caller: it needs a + and the country code. Set owner.phone (in "
            "Settings, or [owner] phone in pack.toml) to the number as +<country code><number>, like +15551234567 "
            "for a US number, with no (0) after the country code",
            owner_phone,
        )
    MLX_EXEC.submit(load_models).result()
    # log_config=None keeps uvicorn off its own handlers; adopt gives its warnings and errors the one format, in the
    # service log the call review reads.
    logs.adopt("uvicorn")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning", log_config=None)


if __name__ == "__main__":
    main()
