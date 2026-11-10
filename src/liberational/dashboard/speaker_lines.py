"""Lines to check and voice labels: the uncertain lines the owner confirms or corrects, undoing those labels,
naming a voice, and the transcript's own corrections (moved out of the server, #61).

GET  /speakers, /speakers/refine, /speakers/refine/log, /speaker/<id>/latest, /speaker/<id>/latest-clip
POST /speakers/tag, /speakers/label, /speakers/refine, /speakers/refine/undo
POST /transcript/reject, /transcript/undo, /transcript/media, /transcript/not-media, /transcript/unlabel,
     /voices/unknown/media
"""

import copy
import json
import re
import threading
import time
import uuid
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from liberational import logs, speech_log
from liberational.dashboard import (
    context,
    inbox_state,
    known_voices,
    live_status,
    people_list,
    people_reads,
    routes,
    transcript_lines,
    unnamed_groups,
)
from liberational.dashboard.context import LOG, Reply, log
from liberational.domain.identity import Basis, SubjectRef
from liberational.domain.kernel import PersonId
from liberational.people import registry, transcript_view

if TYPE_CHECKING:
    from liberational.dashboard.server import H


def reject_transcript(body: Mapping[str, Any]) -> Reply:
    """Record the owner's correction for every exact line in one visible turn, atomically."""
    from liberational.listen import revisions

    keys = body.get("lines")
    if keys is None and isinstance(body.get("line"), str):
        keys = [body["line"]]
    if not isinstance(keys, list) or not 0 < len(keys) <= 1000 or any(not isinstance(k, str) or not k for k in keys):
        return 400, {"ok": False, "error": "Send 1 to 1000 exact transcript line keys."}
    keys = list(dict.fromkeys(keys))
    if token := revisions.current_rejection(context.STATE, keys):
        return 200, {"ok": True, "lines": keys, "undo": token}
    available = {
        registry.key_of(e): e for e in transcript_lines.heard_lines() if isinstance(e.get("ts_start"), int | float)
    }
    if any(key not in available for key in keys):
        return 404, {"ok": False, "error": "One of those transcript lines is unavailable. Refresh the conversation."}
    try:
        token = revisions.reject_lines(context.STATE, [available[key] for key in keys])
    except ValueError as error:
        return 409, {"ok": False, "error": str(error)}
    return 200, {"ok": True, "lines": keys, "undo": token}


def media_transcript(body: Mapping[str, Any]) -> Reply:
    """The owner's "This is TV" on transcript lines: one atomic, undoable mark."""
    from liberational.listen import revisions

    keys = body.get("lines")
    if not isinstance(keys, list) or not 0 < len(keys) <= 1000 or any(not isinstance(k, str) or not k for k in keys):
        return 400, {"ok": False, "error": "Send 1 to 1000 exact transcript line keys."}
    keys = list(dict.fromkeys(keys))
    if token := revisions.current_media(context.STATE, keys):
        return 200, {"ok": True, "lines": keys, "undo": token}
    available = {
        registry.key_of(e): e for e in transcript_lines.heard_lines() if isinstance(e.get("ts_start"), int | float)
    }
    if any(key not in available for key in keys):
        return 404, {"ok": False, "error": "One of those transcript lines is unavailable. Refresh the conversation."}
    try:
        token = revisions.mark_media(context.STATE, [available[key] for key in keys])
    except ValueError as error:
        return 409, {"ok": False, "error": str(error)}
    return 200, {"ok": True, "lines": keys, "undo": token}


def media_voice(body: Mapping[str, Any]) -> Reply:
    """The owner's "This is TV" on an unnamed voice: every line it has that isn't a TV's already."""
    from liberational.listen import revisions

    group = body.get("group")
    if isinstance(group, bool) or not isinstance(group, int):
        return 400, {"ok": False, "error": "Send the voice's group number."}
    if registry.voice_names(context.STATE, [group]).get(group):
        return 409, {"ok": False, "error": "That voice has a name. Give its lines to someone else instead."}
    lines = [
        e
        for e in transcript_lines._view().lines(speech_log.retained())
        if e.get("speaker_id") == group and not e.get("media") and isinstance(e.get("ts_start"), int | float)
    ]
    if not lines:
        return 404, {"ok": False, "error": "That voice has no lines left to mark."}
    try:
        token = revisions.mark_media(context.STATE, lines)
    except ValueError as error:
        return 409, {"ok": False, "error": str(error)}
    return 200, {"ok": True, "lines": [registry.key_of(e) for e in lines], "undo": token}


def not_media(body: Mapping[str, Any]) -> Reply:
    """The owner's Not TV on transcript lines: every mark on them is taken back, so the tagger's mistakes are undone."""
    from liberational.listen import revisions

    keys = body.get("lines")
    if not isinstance(keys, list) or not 0 < len(keys) <= 1000 or any(not isinstance(k, str) or not k for k in keys):
        return 400, {"ok": False, "error": "Send 1 to 1000 exact transcript line keys."}
    token, lines = revisions.unmark_media(context.STATE, keys)
    if not lines:
        return 404, {"ok": False, "error": "None of those lines is marked as a TV's."}
    return 200, {"ok": True, "lines": lines, "undo": token}


def undo_transcript(body: Mapping[str, Any]) -> Reply:
    """Restore the batch named by its durable correction token."""
    from liberational.listen import revisions

    token = body.get("undo")
    if not isinstance(token, str) or not token:
        return 400, {"ok": False, "error": "Send the correction's Undo token."}
    try:
        lines = revisions.undo_correction(context.STATE, token)
    except KeyError:
        return 404, {"ok": False, "error": "Nothing to undo there."}
    except ValueError as error:
        return 409, {"ok": False, "error": str(error)}
    return 200, {"ok": True, "lines": lines}


def log_mark() -> tuple[int, ...]:
    """Changes whenever a process appends to a registry stream (a label, a rename, a forget): `Registry.mark`."""
    try:
        from liberational.listen import revisions

        return (*registry.at(context.STATE).mark(), revisions.review_revision(context.STATE))
    except Exception as e:  # the broker is down: nothing changed that a client could be told of
        LOG.error(f"log_mark: {e!r}")
        return ()


def latest_for_speaker(sid: int) -> dict[str, Any] | None:
    view = transcript_lines._view()
    for e in transcript_lines.iter_back():
        # A rejected line, or one with no voice, has no speaker_id.
        if transcript_lines.has_audio(e) and (e := view.line(e)).get("speaker_id") == sid:
            return e
    return None


_speakers_shared: tuple[tuple[Any, ...], dict[str, Any]] | None = None


_speakers_lock = threading.Lock()


def _speakers_key() -> tuple[Any, ...]:
    """What the voice list is worked out from: the event log (voices, names, enrollments, and the transcript's review),
    the speech log (each voice's lines and when it was last heard, and purges), and the voiceprint files."""
    return (
        log_mark(),
        speech_log.shared().mark(),
        people_reads.mtime(context.STATE / "voiceprints.json"),
        people_reads.mtime(context.STATE / "speakers.json"),
    )


def speakers_public() -> dict[str, Any]:
    """GET /speakers: every voice, most recently heard first, named as `_speakers` names it. Worked out once per change
    and shared: every open page's event stream asked for it on each line heard, about 235 registry lookups
    (15 ms of Python) each, a dozen at a time, and the requests behind them waited seconds on the registry's lock."""
    global _speakers_shared
    key = _speakers_key()
    with _speakers_lock:  # one works it out; the others with the same key wait for it and reuse it
        if _speakers_shared is None or _speakers_shared[0] != key:
            d = known_voices._speakers()
            value = {"speakers": sorted(d["speakers"], key=lambda s: -(s.get("last_seen") or 0)), "merged": d["merged"]}
            _speakers_shared = (key, value)
        return copy.deepcopy(_speakers_shared[1])


def _teach_voice(lines: list[dict[str, Any]], name: str) -> list[str]:
    """Confirm `lines` as `name`'s (`voices.teach`), so the voice the owner just named teaches listen. Returns
    their keys, which an undo releases (`_untaught`); [] when nothing could be written (logged, never raised: the
    naming itself stands)."""
    from liberational.listen import voices

    if not lines:
        return []
    try:
        voices.teach(context.STATE, lines, name)
    except Exception as e:
        log(f"teaching {name}'s voice from {len(lines)} line(s) failed:", repr(e), level=logs.ERROR)
        return []
    return [registry.key_of(e) for e in lines]


VOICE_TAGS = context.STATE / "voice_tags.jsonl"  # "this line was <name>" requests, handled by the microphone service


REFINE_LOG = (
    context.STATE / "refine_log.jsonl"
)  # every label given in "Lines to check", with what it replaced, so it can be undone


def request_voice_tags(ts_starts: object, name: object, control: bool = False, lines: object = None) -> Reply:
    """(code, body): queue "these lines were NAME" for the microphone service, which relabels them and learns the
    voice from their audio (audio/ambient.py Transcriber.process_tags). The dashboard never rewrites the transcript.
    Lines are tagged as the owner only from this machine (`control`): the tag teaches the owner's voice their sound,
    and an owner tag is what makes a voice the owner's (registry.owner_voice)."""
    name = people_list._person_name(name)
    if not name:
        return 400, {"ok": False, "error": "Names start with a letter and use letters, spaces, . ' or -"}
    if people_list._norm(name) in people_list._owner_keys() and not control:
        return 403, {"ok": False, "error": "Lines can be tagged as you only from this machine."}
    # Each line by its key, or by its time alone as before: then the first line in that millisecond.
    asked = lines if lines is not None else ts_starts
    if not isinstance(asked, list) or not 0 < len(asked) <= 200:
        return 400, {"ok": False, "error": "send the lines' keys (or ts_start values), 1 to 200"}
    try:
        tags = (
            [
                {"ts_start": at, **({"line": k} if slug else {})}  # a key with no device is its millisecond alone
                for k in asked
                if isinstance(k, str) and k
                for slug, at in [registry.split_key(k)]
            ]
            if lines is not None
            else [{"ts_start": round(float(x), 3)} for x in asked]
        )
    except TypeError, ValueError:
        return 400, {"ok": False, "error": "line keys and ts_start values must be well formed"}
    if len(tags) != len(asked):
        return 400, {"ok": False, "error": "line keys must be text"}
    now = time.time()
    keys = [tag.get("line") or registry.line_key(tag["ts_start"]) for tag in tags]
    with VOICE_TAGS.open("a") as f:
        f.write(
            json.dumps(
                {"lines": keys, "group": str(uuid.uuid4()), "name": name, "requested_at": now}, ensure_ascii=False
            )
            + "\n"
        )
    note = (
        None
        if live_status._listening()
        else "The microphone service isn't running; the lines are tagged when it starts."
    )
    if len(tags) > 1:
        note = " ".join(
            filter(
                None, [note, "Only lines that agree with this turn's majority are learned. Labels you set are kept."]
            )
        )
    return 200, {"ok": True, "queued": len(tags), "name": name, "note": note}


def label_speaker(sid: object, name: str, control: bool = False) -> Reply:
    """(code, body): name a voice (Live's turn rename). Never the owner's name: "That voice is mine" does that, from
    this machine (confirm_owner_voice). The owner's own voice is renamed only from this machine (`control`), since
    renaming it takes the owner path away from it."""
    name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip()
    if not re.fullmatch(r"\d{1,6}", str(sid)):
        return 400, {"ok": False, "error": "bad id"}
    if not (1 <= len(name) <= 40):
        return 400, {"ok": False, "error": "name must be 1-40 chars"}
    if people_list._norm(name) in people_list._owner_keys():
        return 403, {
            "ok": False,
            "error": "To say a voice is yours, use \u201cThat voice is mine\u201d on this machine.",
        }
    if not control and registry.owner_voice(
        context.STATE, int(str(sid))
    ):  # sid is digits (checked above): as the voice id
        return 403, {"ok": False, "error": "That's your voice. Rename it from this machine."}
    # A person's name, as rename and claim take it (never "unknown", the assistant's name, or another automatic
    # label: the voice would leave People, and same-label voices merge), or the voice's own automatic label back.
    if name != f"Speaker {sid}":
        if not (person := people_list._person_name(name)):
            return 400, {"ok": False, "error": "Names start with a letter and use letters, spaces, . ' or -"}
        name = person
    ok, out = known_voices._speaker_cmds([["label", str(sid), name]])
    log(f"label {sid} -> {name!r}: {'ok' if ok else 'FAILED'} {out[-120:]}", level=logs.INFO if ok else logs.ERROR)
    return (200 if ok else 500), {"ok": ok, "out": out, "speakers": speakers_public()}


REFINE_PAGE = 50  # uncertain lines sent at once; "remaining" counts them all


def speakers_refine_list() -> dict[str, Any]:
    """Get ambiguous utterances: clips with low speaker_sim that are borderline between recognized speakers.
    These are candidates where the auto-labeling was uncertain and manual review would help."""
    d = known_voices._speakers()
    spk_map = {s["id"]: s.get("label", "Speaker " + str(s["id"])) for s in d.get("speakers", [])}
    # Find enrolled speakers (real people with enough data)
    enrolled = [s for s in d.get("speakers", []) if s.get("enrolled")]
    if not enrolled:
        return {"items": [], "remaining": 0, "message": "no enrolled speakers yet"}
    # Scan transcript for utterances from enrolled speakers that have low similarity. "remaining" counts every
    # such line, while items holds the first REFINE_PAGE of them.
    items, remaining = [], 0
    skipped = set(inbox_state.inbox_skips()["line"])

    def add(cand: dict[str, Any]) -> None:
        nonlocal remaining
        remaining += 1
        if len(items) < REFINE_PAGE:
            items.append(cand)

    for e in transcript_lines._view().lines(transcript_lines.heard_lines()):
        sid = e.get("speaker_id")
        if sid is None or sid not in spk_map:
            continue
        if unnamed_groups.put_aside(e, skipped):
            continue
        sim = e.get("speaker_sim", 1.0)
        # Low similarity but still matched the speaker. A line with no speaker id has no voice to name, so it isn't one.
        if sim is not None and sim < 0.7 and sid in {s["id"] for s in enrolled}:
            add(
                {
                    "ts_start": e.get("ts_start", 0),
                    "line": registry.key_of(e),
                    "speaker_id": sid,
                    "text": (e.get("text") or "")[:200],
                    "speaker_sim": round(sim, 3),
                    "sim_hint": f"{sim * 100:.0f}% match to {spk_map[sid]}",
                }
            )
    return {"items": items, "remaining": remaining}


REFINE_FIELDS = ("speaker", "speaker_id", "speaker_sim", "multi_speaker", "tagged")  # a line's label, as read


def _line_at(
    ts: object = None, line: object = None
) -> tuple[dict[str, Any], dict[str, Any], transcript_view.View] | None:
    """(the line as stored, the line as readers see it, the view), or None when there's no such line. See
    _stored_line for how it is found."""
    e = people_reads._stored_line(ts, line)
    if e is None:
        return None
    view = transcript_lines._view()
    return e, view.line(e), view


def _label_line(
    line: SubjectRef,
    name: str | None,
    voice: int | None,
    basis: Basis = Basis.OWNER_TAG,
    label: str | None = None,
    **kw: Any,
) -> None:
    """Label the line (registry.utterance_of) in the event log; nothing rewrites the line (step 4). A person's name makes it theirs.
    Anything else takes it away from its voice `voice` and gives it `label` ("unknown", "multiple speakers"). Raises
    when the log can't be written, since the log is the only record of the label."""
    reg = registry.at(context.STATE)
    if registry.is_name(name):
        reg.claim(line, name, basis, "dashboard", **kw)
    else:
        via = None if voice is None else registry.voiceprint(voice)
        reg.detach(line, label or name or transcript_view.UNKNOWN, "dashboard", via=via)


def _was(said: registry.Said | None) -> str:
    """How a line read before a label, so an undo can put it back: "own" (labeled or matched itself), "detached"
    (taken away from its voice), or "voice" (following its voice)."""
    if said is not None and said.own.person is not None:
        return "own"
    return "detached" if said is not None and said.detached else "voice"


def _named_now(name: str | None, at: float | None) -> str | None:
    """What an undo calls the person a label's old state named `name`, as of `at` (when the label was given): their
    name now, after any rename, or "unknown" once they are forgotten. An undo never registers anyone, so it never
    brings back someone People: Delete forgot, nor gives their line to someone new who took the name since."""
    if not registry.is_name(name):
        return name
    reg = registry.at(context.STATE)
    pid = reg.known_as(name, at)
    person = reg.person(pid) if pid else None
    return person.display_name if person else transcript_view.UNKNOWN


def _restore(
    stored: dict[str, Any], old: dict[str, Any], was: str | None, view: transcript_view.View, at: float | None = None
) -> None:
    """Undo a label: make the line read as `old` again, with a superseding assertion, a line label, or a release.
    `was` is None for a label logged before step 4, when the label was written onto the line itself. `at` is when
    the label was given; a person the old state named is looked up as of then (`_named_now`)."""
    line, name, voice = registry.utterance_of(stored), old.get("speaker"), view.canonical(stored.get("speaker_id"))
    legacy = was is None
    if legacy:
        hand = old.get("speaker_sim") is None and registry.is_name(name)
        was = "own" if hand else "detached" if old.get("speaker_id") is None and name else "voice"
    if was == "detached":
        _label_line(line, None, voice, label=name)
    elif was == "own" and old.get("speaker_sim") is None:
        _label_line(line, _named_now(name, at), voice, Basis.CORRECTION)
    elif was == "own":  # the backfill had matched it: that match again, over the label
        sim = float(old["speaker_sim"])
        if (said := view.explain(stored)) is not None and said.detached:  # a match never outranks the owner's label
            registry.at(context.STATE).release(line, "dashboard")
        _label_line(
            line,
            _named_now(name, at),
            voice,
            Basis.BACKFILL_MATCH,
            confidence=sim,
            producer=registry.voice_producer(),
            replace=True,
        )
    else:
        registry.at(context.STATE).release(line, "dashboard")
        v = transcript_lines._view()
        now = v.line(stored).get("speaker")
        if legacy and name and registry.norm(now) != registry.norm(name):  # the line still carries the old label
            if voice is None and (was_voice := v.canonical(old.get("speaker_id"))) is not None:
                # The old label erased the line's voice, so it can't follow that voice again: name the voice's person
                # now, as the old undo's restored voice id would show it.
                current = v.voice_name(was_voice)
                name = current if registry.is_name(current) else _named_now(name, at)
            else:
                name = _named_now(name, at)
            if voice is None or registry.is_name(name):
                _label_line(line, name, voice, Basis.CORRECTION, label=name)


def _refine_log(entry: dict[str, Any]) -> None:
    with REFINE_LOG.open("a") as f:
        f.write(json.dumps({"ts": round(time.time(), 3), **entry}, ensure_ascii=False) + "\n")


def _refine_entries() -> list[dict[str, Any]]:
    out = []
    if REFINE_LOG.exists():
        for ln in REFINE_LOG.read_text().splitlines():
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
    return out


def refine_log_list(limit: int = 50) -> dict[str, Any]:
    """Labels given in Lines to check that haven't been undone, newest first."""
    undone = {(e.get("ts_start"), e.get("undone_ts")) for e in _refine_entries() if e.get("action") == "undo"}
    live = [
        e
        for e in _refine_entries()
        if e.get("action") in ("assign", "multi") and (e["ts_start"], e["ts"]) not in undone
    ]
    return {"labels": live[::-1][:limit], "total": len(live)}


def speakers_refine(action: object, body: dict[str, Any]) -> Reply:
    """Handle refine actions: 'assign' (label an utterance), 'skip', or 'multi' (multiple speakers). Each label is an
    event in the log (the line is never rewritten), and is logged with what it replaced (refine_log.jsonl), so
    speakers_refine_undo can put it back."""
    action = str(action).strip()
    ts, line = body.get("ts_start"), body.get("line")  # the line's key, or its time alone as before
    if line is not None and not isinstance(line, str):
        return 400, {"ok": False, "error": "line must be a line key"}
    if action == "skip":
        return 200, {"ok": True, "action": action}
    if action == "assign":
        name = re.sub(r"[\x00-\x1f\x7f]", "", str(body.get("name", "")).strip())
        if not (ts or line) or not name:
            return 400, {"ok": False, "error": "line (or ts_start) and name required"}
    elif action == "multi":
        if not (ts or line):
            return 400, {"ok": False, "error": "line (or ts_start) required"}
        name = unnamed_groups.MULTI
    else:
        return 400, {"ok": False, "error": f"unknown action: {action}"}
    try:
        found = _line_at(ts, line)
        if found is None:
            return 404, {"ok": False, "error": "no such utterance"}
        stored, before, view = found
        said = view.explain(stored)
        was = _was(said)
        _label_line(registry.utterance_of(stored), name, view.canonical(stored.get("speaker_id")))
        if action == "assign" and registry.is_name(name):
            # The owner's word on the line teaches listen its person's voice: its embedding becomes one of
            # their confirmed references. The undo puts the line back, and with it the reference goes.
            _teach_voice([stored], name)
        denied = _deny_voice(stored, said.person if said is not None else None, name) if action == "assign" else None
        hit = transcript_lines._view().line(stored)
    except people_reads.AmbiguousLine as ex:
        return 409, {"ok": False, "error": str(ex)}
    except Exception as ex:
        return 500, {"ok": False, "error": str(ex)}
    _refine_log(
        {
            "action": action,
            "ts_start": hit.get("ts_start"),
            "line": registry.key_of(stored),
            "text": (hit.get("text") or "")[:120],
            "old": {k: before[k] for k in REFINE_FIELDS if k in before},
            "new": {k: hit[k] for k in REFINE_FIELDS if k in hit},
            "was": was,
            **({"denied": denied} if denied else {}),
        }
    )
    transcript_view.rebuild_quietly(context.STATE, [transcript_view.day_of(hit["ts_start"])])
    return 200, {"ok": True, "action": action, "entry": hit}


def _deny_voice(line: dict[str, Any], was: PersonId | None, name: str) -> str | None:
    """The owner took `line` from the person it was (`was`) by labeling it `name` (someone else, unknown): its voice
    becomes one of their negatives (`voices.deny`), so a line like it isn't named them. Returns their id for the
    undo (`speakers_refine_undo` releases it), or None when nothing was kept (logged, never raised: the label stands)."""
    from liberational.listen import voices

    if was is None:
        return None
    reg = registry.at(context.STATE)
    to = reg.person_id(name) if registry.is_name(name) else None
    if to is not None and reg.resolve(to) == reg.resolve(was):
        return None
    try:
        return str(was) if voices.deny(context.STATE, [line], str(was)) else None
    except Exception as e:
        log("keeping a voice line as not its person failed:", repr(e), level=logs.ERROR)
        return None


def speakers_refine_undo(body: dict[str, Any]) -> Reply:
    """Put lines back the way they were before their latest label: {"lines": [their keys]}, {"ts_starts": [...]}
    by timestamp, or {"since": ts} (every label given after that time). Newest labels are undone first, so repeated
    labels unwind in order. An older label logged without a line key is found by its millisecond."""
    lines, ts_starts, since = body.get("lines"), body.get("ts_starts"), body.get("since")
    if since is not None and not isinstance(since, (int, float)):
        return 400, {"ok": False, "error": "since must be a time"}
    if ts_starts is not None and not (
        isinstance(ts_starts, list) and all(isinstance(t, (int, float)) for t in ts_starts)
    ):
        return 400, {"ok": False, "error": "ts_starts must be a list of times"}
    if lines is not None and not (isinstance(lines, list) and all(isinstance(k, str) and k for k in lines)):
        return 400, {"ok": False, "error": "lines must be a list of line keys"}
    if ts_starts is None and since is None and lines is None:
        return 400, {"ok": False, "error": 'send {"lines": [...]}, {"ts_starts": [...]}, or {"since": ts}'}
    keys = set(lines or ())
    try:
        ms = {registry.line_key(registry.split_key(k)[1]) for k in keys}
    except ValueError:
        return 400, {"ok": False, "error": "lines must be line keys, like mic@1761234567.123"}

    def asked(e: dict[str, Any]) -> bool:
        if since is not None and e["ts"] >= since:
            return True
        if ts_starts and e["ts_start"] in ts_starts:
            return True
        return bool(keys) and (e["line"] in keys if e.get("line") else registry.line_key(e["ts_start"]) in ms)

    live = refine_log_list(limit=100000)["labels"]  # newest first, not yet undone
    todo = [e for e in live if asked(e)]
    if ts_starts or keys:  # one undo per line: its latest label (newest first, so the first one seen)
        seen, done_ms, uniq = set(), set(), []
        for e in todo:
            at = registry.line_key(e["ts_start"])
            if e.get("line"):
                if e["line"] in seen:
                    continue
                seen.add(e["line"])
                done_ms.add(at)  # a label logged before keys in this millisecond is an older one of this line
            elif at in done_ms or at in seen:
                continue
            else:
                seen.add(at)
            uniq.append(e)
        todo = uniq
    undone, errors, days = [], [], set()
    for e in todo:
        old = e.get("old") or {}
        try:
            found = _line_at(e["ts_start"], e.get("line"))
            if found is None:
                errors.append(f"no line at {e['ts_start']}")
                continue
            stored, _, view = found
            _restore(stored, old, e.get("was"), view, e.get("ts"))
            if e.get("denied"):  # the line's "not them" goes with its label
                from liberational.listen import voices

                voices.deny(context.STATE, [stored], str(e["denied"]), active=False)
            hit = transcript_lines._view().line(stored)
        except Exception as ex:
            errors.append(str(ex))
            continue
        key = registry.key_of(stored)
        _refine_log({"action": "undo", "ts_start": e["ts_start"], "line": key, "undone_ts": e["ts"], "restored": old})
        undone.append({"ts_start": e["ts_start"], "line": key, "text": e.get("text"), "speaker": hit.get("speaker")})
        days.add(transcript_view.day_of(hit["ts_start"]))
    if days:
        transcript_view.rebuild_quietly(context.STATE, sorted(days))
    log(f"refine undo: {len(undone)} line(s) restored")
    return 200, {"ok": not errors, "undone": undone, "errors": errors}


@routes.get_re(r"/speaker/(\d{1,6})/(latest|latest-clip)")
def _get_speaker_x_x(h: H, m: re.Match[str]) -> None:
    e = latest_for_speaker(int(m.group(1)))
    if e is None:
        return h._json({"ok": False, "error": "no utterance with audio for that speaker"}, 404)
    return h._json(e) if m.group(2) == "latest" else h.clip(entry=e, info=h._query().get("info") in ("1", "true"))


@routes.get("/speakers")
def _get_speakers(h: H) -> None:
    return h._json(speakers_public())


@routes.get("/speakers/refine/log")
def _get_speakers_refine_log(h: H) -> None:
    return h._json(refine_log_list(min(500, max(1, int(h._query().get("limit", 50) or 50)))))


@routes.get("/speakers/refine")
def _get_speakers_refine(h: H) -> None:
    try:
        return h._json(speakers_refine_list())
    except Exception as ex:
        return h._json({"ok": False, "error": str(ex)}, 500)


@routes.post("/speakers/tag")
def _post_speakers_tag(h: H, body: Any) -> None:
    if not h._same_origin_json():
        return h._json({"ok": False, "error": "same-origin JSON only"}, 403)
    code, o = request_voice_tags(body.get("ts_starts"), body.get("name"), h._can_teach(), lines=body.get("lines"))
    return h._json(o, code)


@routes.post(
    "/transcript/reject",
    "/transcript/undo",
    "/transcript/media",
    "/transcript/not-media",
    "/voices/unknown/media",
)
def _post_transcript_reject(h: H, body: Any) -> None:
    p = h.path.split("?")[0]
    if not h._same_origin_json():
        return h._json({"ok": False, "error": "same-origin JSON only"}, 403)
    if not h._can_teach():
        return h._json({"ok": False, "error": "Controls unavailable"}, 403)
    if not isinstance(body, dict):
        return h._json({"ok": False, "error": "Send a JSON object."}, 400)
    action = {
        "/transcript/reject": reject_transcript,
        "/transcript/undo": undo_transcript,
        "/transcript/media": media_transcript,
        "/transcript/not-media": not_media,
        "/voices/unknown/media": media_voice,
    }[p]
    try:
        code, result = action(body)
    except (OSError, RuntimeError) as error:
        LOG.exception("transcript correction failed")
        return h._json({"ok": False, "error": f"The correction could not be saved: {error}"}, 500)
    return h._json(result, code)


@routes.post("/speakers/label")
def _post_speakers_label(h: H, body: Any) -> None:
    if not h._same_origin_json():
        return h._json({"ok": False, "error": "same-origin JSON only"}, 403)
    code, o = label_speaker(body.get("id"), str(body.get("name", "")), h._can_teach())
    return h._json(o, code)


@routes.post("/speakers/refine/undo")
def _post_speakers_refine_undo(h: H, body: Any) -> None:
    origin = h.headers.get("Origin")
    if not h._can_teach():
        return h._json({"ok": False, "error": "labels can only be undone from this machine"}, 403)
    if h.headers.get_content_type() != "application/json" or (
        origin and origin != "http://" + (h.headers.get("Host") or "")
    ):
        return h._json({"ok": False, "error": "same-origin JSON only"}, 403)
    code, o = speakers_refine_undo(body)
    return h._json(o, code)


@routes.post("/speakers/refine")
def _post_speakers_refine(h: H, body: Any) -> None:
    try:
        action = str(body.get("action", "")).strip()
        code, o = speakers_refine(action, body)
        return h._json(o, code)
    except Exception as ex:
        return h._json({"ok": False, "error": str(ex)}, 500)


@routes.post("/transcript/unlabel")
def _post_transcript_unlabel(h: H, body: Any) -> None:
    line = body.get("line")  # the line's key, or its time alone as before
    try:
        ts = None if isinstance(line, str) and line else float(body.get("ts_start"))
    except Exception:
        return h._json({"error": "line (or ts_start) required"}, 400)
    try:
        found = _line_at(ts, line)
        if found is None:
            return h._json({"error": "no such utterance"}, 404)
        stored, _, view = found
        voice = view.canonical(stored.get("speaker_id"))
        _label_line(registry.utterance_of(stored), None, voice, label=transcript_view.UNKNOWN)
        hit = transcript_lines._view().line(stored)
        transcript_view.rebuild_quietly(context.STATE, [transcript_view.day_of(hit["ts_start"])])
        return h._json({"ok": True, "entry": hit})
    except people_reads.AmbiguousLine as ex:
        return h._json({"error": str(ex)}, 409)
    except Exception as ex:
        return h._json({"error": str(ex)}, 500)
