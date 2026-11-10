"""Voice listener: wake phrase ('Hey Bob' by default) -> fast local answer -> spoken via the clone voice.

Follows the speech log (`speech_log`, in the segment store): utterances as the listen worker writes
them, replies written back on `assistant.reply`, and the voice turn set as broker system state. A line counts when its text starts with
a configured wake phrase ('Hey Bob' or 'Hey, Bob'), from any speaker. The words after the wake phrase are the request.
Answers come from the local Ollama LLM with the current local time; speech uses the shared TTS fallback.
The owner's voice (owner.speaker_label) gets the agent gateway and may skip the wake phrase for a question or
request; everyone else must say the wake phrase and is gated by voice_auth.

Usage:
  uv run python -m liberational.assistant.voice              # daemon (service: voice)
  uv run python -m liberational.assistant.voice --say "hey bob what time is it"   # one-shot test, prints + speaks
"""

import argparse
import datetime as dt
import importlib
import json
import pathlib
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import requests


def optional_module(name: str) -> ModuleType | None:
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


# In-memory, owner-only, short-lived follow-up; never ambient transcript or memory.
LAST_QUESTION: tuple[float, str] | None = None


def safe_question(request: str) -> bool:
    from liberational.assistant.ask_friends import sensitive
    from liberational.assistant.voice_auth import request_policy

    return (
        len(request) <= 1200
        and not any(request_policy(request, True))
        and not sensitive(request)
        and bool(re.match(r"^(?:why|what|how|where|when|which|is|are|explain)\b", request, re.I))
    )


QUESTION = re.compile(
    r"^\W*(?:who|what|when|where|why|how|which|whose|did|does|do|is|was|were|has|have|had|can|could|tell me|remind me)\b"
    r"|\?\s*$",
    re.I,
)


def asks_about_forgotten(request: str) -> bool:
    """A question that names a person deleted in People (people/forgotten.py). History questions meet the same check
    in ask.execute; this one also covers questions that would go elsewhere, such as "who's Dana?"."""
    if not QUESTION.search(request):
        return False
    from liberational.memory import history_qa
    from liberational.people import forgotten

    return forgotten.named(request, history_qa.people_named(request))


def _with_names(request: str, *, owner: bool) -> str:
    """The question with "she" or "he" read as the conversation's subject (memory/pronouns.py), so "what has she
    been doing" goes to history like "what has the cat been doing"; the question itself is asked unchanged. Only
    for the owner: anyone else's question is classified as said, so a pronoun can never turn it into a history
    question after the password check has passed it as one about now."""
    if not owner:
        return request
    from liberational.memory import pronouns

    try:
        resolved = pronouns.resolve(request, cfg.path("paths.state"), "voice", time.time())
    except Exception:
        return request
    return resolved.question if resolved else request


from liberational import config as cfg
from liberational import logs, model_runtime, ollama, vision_log
from liberational.speech import clone_server

STATE_DIR = cfg.path("paths.state")
SYSTEM = (
    "You are {name}, a personal assistant speaking out loud through a small wearable. "
    "Answer in one or two short plain sentences, under 30 words, no lists, no markdown, no quotes. "
    "If you do not know something, say so briefly. Current local time: {now}."
)


LOG = logs.get("voice")
# The speech log on Continuum, once the service runs: replies go there, and the voice turn is set as broker
# system state. None in a one-shot `--say` test, which only logs its reply.
CONTINUUM: Any = None
_TURN = "Idle"


def _turn(state: str) -> None:
    """Sets the voice turn on the broker, when on Continuum and it changed (decision 11). A failure is logged and
    otherwise ignored: the reply matters more than the scheduler's hint."""
    global _TURN
    if CONTINUUM is None or state == _TURN:
        return
    _TURN = state
    try:
        CONTINUUM.set_voice_turn(state)
    except Exception as e:
        LOG.warning("could not set the voice turn to %s: %s", state, type(e).__name__)


def status(text: str, level: int = logs.INFO) -> None:
    """Log a line and record it as a status entry for the dashboard's live transcript. A line that only describes
    a spoken reply goes to LOG alone (that detail is stored with the reply instead)."""
    LOG.log(level, text)
    _append({"ts_start": round(time.time(), 3), "text": text, "status": True})


def answer(request: str) -> str:
    from liberational.assistant.voice_auth import wake_phrase

    now = dt.datetime.now().strftime("%A %B %d %Y, %I:%M %p")
    reply = ollama.chat(
        cfg.get("models.ollama.llm"),
        [
            {"role": "system", "content": SYSTEM.format(name=cfg.assistant_name(), now=now)},
            {
                "role": "user",
                "content": request or f"{cfg.owner_or('Someone')} said {wake_phrase()} with no request.",
            },
        ],
        num_ctx=4096,
        num_predict=80,
        temperature=0.3,
        think=False,
        timeout=60,
    )
    text = ollama.text(reply)
    return text.replace('"', "").replace("\n", " ") or "Yes?"


# ---- owner path: the verified owner gets the full phone agent (cameras, transcript, memory, exec, web) ----
# Owner = a line whose voice is the owner's (owner_voice) at sim >= OWNER_MIN_SIM, or tagged by a person; the owner is
# never asked for a password. Anyone else gets, without a password, only the local talk-only answer for public
# questions (answer, above) and read-only questions about what the cameras show now. Everything else, a camera question
# about the past included, needs the voice password, and once given it reaches the same agent (owner_answer),
# whose scope says the speaker is not the recognized owner.
MEMORY = STATE_DIR / "memory.jsonl"
# Ambient labels short 1-2 s owner clips at 0.38-0.90; 0.60 rejected ~40% of the owner's own requests.
OWNER_MIN_SIM = 0.45
OWNER_RULE = (
    "[Tool rule: for this machine's state (services, processes, uptime, disk, files, versions), call exec first and answer "
    "from its output. The camera notes below ARE live access to this machine's cameras: never say you cannot see the cameras, answer from the notes. For weather: web_fetch https://wttr.in/<City>?format=3&u (spaces as +) and answer from that one line. For other web questions use web_search. "
    "Do not narrate and never write commands as text. Reply in one or two short spoken sentences: no lists, no markdown.]"
)


def owner_voice(sid: object, state: pathlib.Path | None = None) -> bool:
    """Whether voice `sid` is the owner's (people/registry.py owner_voice, which fails closed)."""
    from liberational.people import registry

    return bool(cfg.owner_label()) and registry.owner_voice(state if state is not None else STATE_DIR, sid)


def canonical_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    """The line with its speaker named as its voice is now, and `owner_voice` set by owner_voice: the only thing
    owner_entry trusts. Similarity alone, or a name stored on the line, never names the owner."""
    from liberational.people import registry

    e = dict(entry)
    e["owner_voice"] = owner_voice(e.get("speaker_id"))
    try:
        sid = e.get("speaker_id")
        if not isinstance(sid, int) or isinstance(sid, bool):
            return e  # a voice id is a real int: True, or "1", isn't voice 1
        if 1 <= sid < registry.handed_out(pathlib.Path(STATE_DIR)):
            name = registry.voice_names(STATE_DIR, [sid])[sid]
            e["speaker_id"], e["speaker"] = sid, name or f"Speaker {sid}"
    except Exception:
        pass
    return e


def owner_entry(e: Mapping[str, Any]) -> bool:
    """Whether a line (as canonical_entry gives it) is the owner speaking: their voice (owner_voice), matched well
    enough or tagged by a person."""
    try:
        sim = e.get("speaker_sim")
        # tagged: a person labeled the line (people/transcript_view.py reads it so), which outranks any score
        return (
            bool(cfg.owner_label())
            and e.get("owner_voice") is True
            and (bool(e.get("tagged")) or (sim is not None and float(sim) >= OWNER_MIN_SIM))
        )
    except Exception:
        return False


def _gateway_token() -> str | None:
    return cfg.agent_token() or None


def owner_notes() -> str:
    now, parts = time.time(), []
    try:  # cameras: newest caption per camera, within 10 minutes
        latest = {cam: c for cam, c in vision_log.at(STATE_DIR).latest(now).items() if c.get("caption")}
        lines = []
        for cam, c in sorted(latest.items()):
            ts = float(c["ts_start"])
            if now - ts <= 600:
                cap = str(c["caption"]).strip()
                if cap.lower().startswith("same."):
                    cap = "(unchanged) " + cap[5:].strip()
                lines.append(f"{cam}, {int(now - ts)}s ago: {cap}")
        if lines:
            parts.append("Camera notes: " + " | ".join(lines))
    except Exception:
        pass
    # Exclude ambient transcript entirely: it may contain spoken passwords.
    try:  # memory: things the owner asked the assistant to remember, less what they asked to forget
        from liberational.memory import remembered

        items = remembered.items(MEMORY)
        if items:
            parts.append(f"Things {cfg.owner_or('the owner')} asked you to remember: " + "; ".join(items[-8:]))
    except OSError:
        pass
    return "\n".join(parts)


BAD_OWNER_REPLY = re.compile(
    r"(\{|\}|\bexec\b|<tool|^\s*(checking|one moment|running|let me|hold on|i'm checking|i am checking)\b)", re.I
)

from liberational.vision.camera_history import CAMERA_Q
from liberational.vision.camera_history import query as camera_query


def owner_answer(request: str, session_key: str, grant: Grant | None = None) -> str:
    camera = camera_read(request, CAMERA_Q.search)
    if camera:
        try:
            from liberational.memory import ask as pack_ask
            from liberational.memory import history_qa

            if history_qa.asks_about_the_past(request):  # never the current view for a question with a time in it
                return pack_ask.spoken(pack_ask.ask(request, mode="local"))
            return camera_query(request)["answer"]
        except ValueError as ex:
            return str(ex)
        except Exception as ex:
            status(f"camera history unavailable: {ex}", level=logs.WARN)
            return "Camera history is unavailable right now; I cannot establish what changed."
    if not (grant and grant.route == "gateway" and grant.authorized_for(request)):
        raise VoiceAuthorizationError("Local authorization and confirmation required")
    if not cfg.agent_configured():
        raise NoAgent()
    scope = (
        "[Single-request policy authorization for EXACT request "
        + json.dumps(request)
        + ". Recognized owner: "
        + str(grant.owner)
        + "; password required: "
        + str(grant.needs_password)
        + "; password verified locally: "
        + str(grant.authenticated)
        + "; separate confirmation required: "
        + str(grant.needs_confirmation)
        + "; confirmation received: "
        + str(grant.confirmed)
        + ". No other action is authorized. Native tool approvals remain mandatory. "
        "If details or scope change, do not execute; ask for a new voice request.]"
    )
    tok = _gateway_token()
    if not tok:
        raise RuntimeError("gateway token unavailable")
    body = {
        "model": cfg.get("agent.phone_agent"),
        "user": session_key,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": "\n".join(
                    x
                    for x in [
                        scope,
                        OWNER_RULE,
                        owner_notes() and f"[Notes: {owner_notes()}]",
                        f"{cfg.owner_or('User')}: {request}",
                    ]
                    if x
                ),
            }
        ],
    }
    r = requests.post(
        cfg.get("agent.url"),
        json=body,
        timeout=150,
        headers={"Authorization": f"Bearer {tok}", "x-openclaw-message-channel": "phone"},
    )
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"].strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip().replace('"', "").replace("`", "")
    text = " ".join(text.split())
    if not text or BAD_OWNER_REPLY.search(text):
        raise MissingResult("missing_result")
    parts = [x for x in re.split(r"(?<=[.!?])\s+", text) if x]
    return " ".join(parts[: 3 if camera else 2])[:500]


class VoiceAuthorizationError(PermissionError):
    """The local voice grant does not authorize this exact dispatch request."""


class MissingResult(RuntimeError):
    pass


class NoAgent(RuntimeError):
    """The request needs the agent gateway and none is configured ([agent] url is empty)."""


NO_AGENT_REPLY = (
    "I can't do that without an agent, and none is set up. Add one under agent in the Liberational settings."
)


class Speaker:
    def __init__(self) -> None:
        from liberational.speech import tts

        t0 = time.time()
        self.tts = tts.TTS(clone=True)
        self.tts.synth("Ready.")  # warm the selected voice without playing startup audio
        status(f"voice ready in {time.time() - t0:.1f}s")

    def speak(self, text: str) -> None:
        from liberational.speech.voice_clone import play_chunks

        play_chunks(self.tts.generate(text))


@dataclass
class _HistorySpeech:
    """The already-spoken prefix of one authorized history request, kept on the speaker's own thread."""

    speaker: Speaker
    prefix: str = ""
    started: float | None = None
    failure: Exception | None = None

    def short(self, result: Mapping[str, Any]) -> None:
        from liberational.memory import ask as pack_ask

        if self.prefix or self.failure:
            return
        self.prefix = pack_ask.spoken(result)
        self.started = time.time()
        LOG.info("voice short answer ready")
        _turn("Speaking")
        try:
            self.speaker.speak(self.prefix)
        except Exception as error:
            self.failure = error
            raise
        finally:
            _turn("Thinking")


from liberational.assistant.voice_auth import Conversation, Grant, asks_about_the_past, camera_read, speaker_key


def follow_voice(key: str) -> str:
    """A speaker key (voice_auth.speaker_key) as it is now: "voice:5" is "voice:3" once voice 5 was merged into 3.
    Voice auth follows its lockouts through merges with it (a pending request ends at a merge); a requirement
    is never upgraded by this."""
    if match := re.fullmatch(r"voice:([0-9]+)", key):
        return speaker_key(canonical_entry(dict(speaker_id=int(match.group(1)))))
    return key


CONVERSATION = Conversation(STATE_DIR / "voice_auth.json", follow=follow_voice)


def handle(text_line: str, speaker: Speaker | None = None, entry: Mapping[str, Any] | None = None) -> str | None:
    """Answers a line if it's for the assistant (and speaks the answer with `speaker`). On Continuum the voice turn is
    Listening when a request is accepted, Thinking while it works out the answer, Speaking while it plays, and
    Idle after."""
    try:
        return _handle(text_line, speaker, entry)
    finally:
        _turn("Idle")


def _handle(text_line: str, speaker: Speaker | None = None, entry: Mapping[str, Any] | None = None) -> str | None:
    global LAST_QUESTION
    # No incoming text logging; pending-auth utterances are secret candidates.
    started = time.monotonic()
    text = text_line
    entry = canonical_entry(entry or {})
    if not CONVERSATION.pending:
        text = re.sub(r"\bchem\s*(?=zero\b|one\b|two\b|[012]\b)", "cam ", text, flags=re.I)
    result = CONVERSATION.accept(text, entry or {}, owner_entry(entry or {}), CAMERA_Q.search)
    if result is None:
        return None
    _turn("Listening")
    _turn("Thinking")
    if not isinstance(result, Grant) or not result.owner or result.needs_password or result.needs_confirmation:
        LAST_QUESTION = None
    error = None
    history_speech = _HistorySpeech(speaker) if speaker is not None else None
    history_options: dict[str, Any] = {"on_short": history_speech.short} if history_speech is not None else {}
    if isinstance(result, str):
        reply = result
    elif result.route == "camera":
        try:
            from liberational.memory import ask as pack_ask
            from liberational.memory import history_qa

            if not result.owner and asks_about_the_past(result.request):
                # accept() sends anyone else's question about the past to the password; never answer one here
                reply = "That needs the voice password."
                error = "unauthorized"
            elif history_qa.asks_about_the_past(_with_names(result.request, owner=result.owner)):
                # Anything with a time in it is Ask's: the camera answer below only knows what they show now.
                reply = pack_ask.spoken(pack_ask.ask(result.request, **history_options))
            else:
                reply = camera_query(result.request)["answer"]
        except ValueError as ex:
            reply = str(ex)
            error = "ambiguous_target"
        except Exception:
            reply = "Camera history is unavailable right now; I cannot establish what changed."
            error = "retrieval_failed"
    elif result.route == "public":
        try:
            reply = answer(result.request)
        except Exception:
            reply = "I could not get an answer."
            error = "missing_result"
    else:
        if not result.authorized_for(result.request):
            raise PermissionError("Unauthorized voice dispatch")
        from liberational.people import identity
        from liberational.phone import outbound as outbound_call

        call = outbound_call.parse(result.request)
        try:
            # Distinguish 'ask all your friends' (all models) vs 'ask your friends' (1 Anthropic + 1 OpenAI)
            all_friends = re.match(
                r"^(?:please\s+)?ask (?:all (?:your )?friends|everyone)\b[\s,:-]*(.*)", result.request, re.I
            )
            friends = re.match(r"^(?:please\s+)?ask (?:your )?friends\b[\s,:-]*(.*)", result.request, re.I)
            if all_friends or friends:
                module = optional_module("liberational.memory.ask")
                if module:
                    question = (all_friends or friends).group(1).strip()  # ty: ignore[unresolved-attribute]  # the if above saw a match
                    subset = "all" if all_friends else "pair"
                    if not question and LAST_QUESTION and time.time() - LAST_QUESTION[0] < 120:
                        question = LAST_QUESTION[1]
                    if not question:
                        reply = (
                            "What question should I ask all my friends?"
                            if all_friends
                            else "What question should I ask my friends?"
                        )
                        error = "ambiguous_target"
                    elif result.needs_password or result.needs_confirmation:
                        reply = "Sensitive requests are not sent to the friends panel."
                    else:
                        response = module.ask(question, mode="everyone" if subset == "all" else "friends")
                        reply = module.spoken(response)
                        if response.get("status") == "partial":
                            error = "friends_partial"
                        elif response.get("status") != "ok":
                            error = "friends_unavailable"
                        elif not response.get("answer"):
                            error = "missing_result"
                else:
                    reply = "My friends are not available yet."
                    error = "friends_unavailable"
            elif call:
                reply = outbound_call.place(*call)[1]
            elif intro := identity.INTRO_COMMAND.match(result.request):
                # Introductions are handled here, with the faces and voices Liberational already has; no agent needed.
                reply = identity.collar_intro(intro.group(1))["reply"]
            elif cfg.get("identity.command_attributes") and (tag := identity.tag_command(result.request)):
                # "that's Alice, CEO of Acme": the same introduction, with a role and organization (step 6)
                reply = identity.collar_intro(tag.name, attributes=tag.attributes())["reply"]
            elif asks_about_forgotten(result.request):
                from liberational.people import forgotten

                reply = forgotten.LINE
            else:
                # Existing authorization above remains the gate; no LLM for object retrieval.
                objects = optional_module("liberational.memory.objects_voice")
                object_reply = objects.handle(result.request) if objects else None
                from liberational.memory import ask as pack_ask
                from liberational.memory import history_qa

                if history_qa.is_history_question(_with_names(result.request, owner=result.owner)):
                    response = pack_ask.ask(result.request, mode="local", **history_options)
                    reply = pack_ask.spoken(response)
                    if response.get("status") not in ("ok", "partial"):
                        error = "history_qa_failed"
                else:
                    reply = (
                        object_reply
                        if object_reply is not None
                        else owner_answer(result.request, f"inperson-{time.time_ns()}", grant=result)
                    )
        except NoAgent:
            error = "no_agent"
            reply = NO_AGENT_REPLY
        except MissingResult:
            error = "missing_result"
            reply = "I did not get a verified result for that."
        except VoiceAuthorizationError as ex:
            # The local voice authorization didn't cover this exact request: nothing was sent to the agent.
            error = "not_authorized"
            reply = "I didn't send that because this request wasn't authorized. Please ask again."
            status(f"request not authorized locally: {ex}", level=logs.WARN)
        except Exception as ex:
            # Log the cause even when a local failure kept the request from reaching the gateway.
            error = "gateway_error" if not friends and not call else "dispatch_failed"
            reply = "That request could not be completed. Native tool approval may still be needed."
            status(f"{error}: {type(ex).__name__}: {str(ex)[:300]}", level=logs.WARN)
    if isinstance(result, Grant) and result.owner and not result.needs_password and not result.needs_confirmation:
        try:
            if safe_question(result.request):
                LAST_QUESTION = (time.time(), result.request)
            elif not re.search(r"ask your friends", result.request, re.I):
                LAST_QUESTION = None
        except Exception:
            LAST_QUESTION = None
    route = result.route if isinstance(result, Grant) else "authorization"
    LOG.info(
        "voice reply ready route=%s dispatch_seconds=%.3f error=%s", route, time.monotonic() - started, error or "none"
    )
    if (
        isinstance(result, Grant)
        and not result.addressed
        and error
        and error != "friends_partial"
        and not (history_speech and history_speech.prefix)
    ):  # that has an answer
        # A question the owner asked without the wake phrase may well have been for someone else in the room: a
        # failure ("I can't do that without an agent") is never spoken over them.
        LOG.info(f"not answering a line that didn't say {cfg.assistant_name()}'s name: error={error}")
        return None
    if speaker:
        start = history_speech.started if history_speech and history_speech.started is not None else time.time()
        try:
            _turn("Speaking")
            prefix = history_speech.prefix if history_speech else ""
            if history_speech and history_speech.failure is not None:
                error = "speech_failed"
                reply = prefix
                raise history_speech.failure
            if prefix:
                if reply.startswith(prefix):
                    rest = reply[len(prefix) :].strip()
                else:
                    rest = "I could not complete the explanation."
                    reply = prefix + " " + rest
                if rest:
                    speaker.speak(rest)
            else:
                speaker.speak(reply)
        finally:
            CONVERSATION.spoken_reply(reply, start, time.time())
            log_reply(reply, start, time.time(), entry, {"route": route, "error": error})
    return reply


def log_reply(
    text: str,
    start: float,
    end: float,
    heard: Mapping[str, Any] | None = None,
    meta: Mapping[str, Any] | None = None,
) -> None:
    """Record what the assistant said: an assistant.reply line in the speech log, which the dashboard shows in the
    live transcript."""
    if not text:
        return
    line = {
        "ts_start": round(start, 3),
        "ts_end": round(end, 3),
        "text": text,
        "speaker": cfg.assistant_name(),
        "reply_to": (heard or {}).get("id"),  # the utterance it answers, by line id
        **(meta or {}),
    }
    _append(line)


def _append(line: Mapping[str, Any]) -> None:
    """A spoken reply as an assistant.reply line. A status entry ("status": true) is only the log line it already
    was, and a one-shot `--say` test has no speech log, so its reply is only logged."""
    line = {"speaker": cfg.assistant_name(), **line}
    if line.get("status"):
        return
    if CONTINUUM is None:
        LOG.info("reply (not recorded outside the service): %s", line.get("text"))
        return
    from liberational import speech_log

    try:
        CONTINUUM.append(speech_log.REPLIES, line)
    except Exception as e:
        LOG.error("could not record the reply: %s", type(e).__name__)


def run_continuum() -> None:
    """The voice service on Continuum: utterances from the speech log as they're written, replies written
    back, and the voice turn set as system state. Lines from before it starts aren't answered (it follows
    from now)."""
    global CONTINUUM
    from liberational import speech_log

    speaker = Speaker()
    clone_server.serve()
    CONTINUUM = speech_log.SpeechLog(client_name="voice")
    _turn("Idle")
    status("listening on the speech log")
    for entry in CONTINUUM.follow(idle_s=0.5):
        if entry is None:
            # Between lines: any clone speech the dashboard asked for, on this thread.
            clone_server.run_pending(0)
            continue
        try:
            handle(entry.get("text", ""), speaker, entry)
            CONVERSATION.share_pending()
        except Exception as ex:
            status(f"error category: {type(ex).__name__}", level=logs.ERROR)


def _voiceprints() -> list[int]:
    """Every voiceprint listen has handed out (listen/voices.py)."""
    from liberational.people import registry

    return list(range(1, registry.handed_out(pathlib.Path(STATE_DIR))))


@model_runtime.reporting("voice")
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--say", help="test: run one wake line through the answer + speech path")
    ap.add_argument("--owner", action="store_true", help="test: treat the line as spoken by the owner")
    a = ap.parse_args()
    if a.say:
        sp = Speaker()
        entry = None
        if a.owner:  # the owner's own voice, as the microphone service would have matched it
            sid = next((n for n in _voiceprints() if owner_voice(n)), None)
            if sid is None:
                raise SystemExit("--owner: no voice is the owner's yet (enroll owner.voice_ref, or label it)")
            entry = {"speaker_id": sid, "speaker_sim": 0.9}
        print(handle(a.say, sp, entry))
    else:
        from liberational import continuum_link

        continuum_link.exit_on_outage(run_continuum)


if __name__ == "__main__":
    main()
