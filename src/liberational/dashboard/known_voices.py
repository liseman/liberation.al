"""The voices the listen worker knows, as the dashboard reads them, and its way of asking the worker to change one
(moved out of the server, #61). No routes: clones, speakers, and people read these.
"""

import subprocess
from collections.abc import Sequence
from typing import Any

from liberational import speech_log
from liberational.config import workdir
from liberational.dashboard import context
from liberational.dashboard.context import LOG, PY
from liberational.people import registry, voice_edits, voice_list

_VOICES = voice_list.VoiceList(lambda: speech_log.shared())


def _speakers() -> dict[str, Any]:
    """The voice list (people/voice_list.py: each voiceprint the speech log or the event log knows) as the dashboard
    reads it: each voice with its name as "label" (its person by person_of, else "Speaker <id>"). "merged" is always
    empty: a voice is claimed by naming it, never folded into another. When the speech log can't be read the
    list is the event log's voices alone, and when the event log can't be read every voice reads as unnamed; both
    failures are logged, so the rest of the dashboard keeps working."""
    known: list[int] = []
    try:
        if registry.log_exists(context.STATE):
            known = registry.at(context.STATE).voiceprints()
    except Exception as e:
        LOG.warning("speakers: the event log's voices can't be read: %s", repr(e))
    try:
        sp = _VOICES.voices(known)
    except Exception as e:
        LOG.error("speakers: the voice list can't be read from the speech log: %s", repr(e))
        sp = voice_list.Tally().rows(known)
    try:
        names = registry.voice_names(context.STATE, [s["id"] for s in sp])
    except Exception as e:
        LOG.warning("speakers: voices read as unnamed, the event log is unreadable: %s", repr(e))
        names = {}
    enrolled = _enrolled_voices([s["id"] for s in sp])
    return {
        "speakers": [
            {**s, "label": names.get(s["id"]) or f"Speaker {s['id']}", "enrolled": s["id"] in enrolled} for s in sp
        ],
        "merged": {},
    }


def _enrolled_voices(sids: list[int]) -> set[int]:
    """The voices learned from a recording: a claim from an enroll (people/voice_edits.py) stands on them."""
    try:
        if not registry.log_exists(context.STATE):
            return set()
        reg = registry.at(context.STATE)
        return {
            n
            for n in sids
            if any(a.source == voice_edits.ENROLL_SOURCE for a in reg.person_of(registry.voiceprint(n)).standing)
        }
    except Exception as e:
        LOG.warning("speakers: which voices are enrolled can't be read: %s", repr(e))
        return set()


def _speakers_by_id() -> dict[int, str]:
    """{voice id: its name, or "Speaker <id>"}, as the dashboard names voices."""
    return {s["id"]: s["label"] for s in _speakers()["speakers"]}


def _speaker_cmds(cmds: Sequence[Sequence[object]]) -> tuple[bool, str]:
    """Run people/voice_edits.py commands in order (stopping at a failure), in a child process: an enroll loads the
    speaker model, which the dashboard never holds. Returns (ok, output)."""
    base = [str(PY), "-m", "liberational.people.voice_edits", "--state-dir", str(context.STATE)]
    ok, outs = True, []
    try:
        for c in cmds:
            r = subprocess.run(
                [*base, *[str(x) for x in c]], cwd=str(workdir()), capture_output=True, text=True, timeout=120
            )
            outs.append((r.stdout + r.stderr).strip())
            if r.returncode:
                ok = False
                break
    except Exception as e:
        ok = False
        outs.append(repr(e))
    return ok, "\n".join(outs)[-500:]
