"""Why a name was given: GET /identity/why?line=<key> for a transcript line, ?shot=<sighting id> for a recorded
face. Read only: it explains the name, and changes nothing.

A reply is {"name", "subject": "line" | "shot", "via", "basis", "asserted_at", "source", "confidence",
"similarity", "threshold", "voice", "mark"}:

- via, for a line, is the rule that named it (transcript_view.View.rule): "line" (something about the line itself:
  the owner named or corrected it, or a match named it), "detached" (taken away from its voice), "hand" (a
  label written on the line before the event log), "heard" (the name it was heard as, kept), "voice" (it follows its
  voice, whose assertion decided), or "none" (its voice names nobody). For a face: "shot" (the owner corrected it,
  or withdrew its name) or "face" (the face worker matched it to the named faces when it was seen).
- basis, asserted_at, source, and confidence describe the assertion that decided (registry Decision.winner), or are
  null when none did.
- similarity is how closely the line's voice or the face matched when it was heard or seen (cosine), with the
  threshold a match needs; null when there was no match score, or when the owner's word decided (the score was for
  the name it replaced).
- voice is {"id", "label"} for a line on a voice.
- mark is what POST /measurement/mark needs to mark the name right or wrong, less shown and right.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from liberational.domain.identity import Decision, Tier, tier
from liberational.people import registry, transcript_view
from liberational.vision.face_guess import MATCH as FACE_MATCH  # vision/faces.py THRESH, without loading cv2


def _assertion(d: Decision | None) -> dict[str, Any]:
    w = d.winner if d is not None else None
    if w is None:
        return {"basis": None, "asserted_at": None, "source": None, "confidence": None}
    return {"basis": str(w.basis), "asserted_at": w.asserted_at, "source": w.source, "confidence": w.confidence}


def _sim(v: object) -> float | None:
    return round(float(v), 3) if isinstance(v, int | float) and not isinstance(v, bool) else None


def for_line(e: dict[str, Any], view: transcript_view.View, near: Iterable[dict[str, Any]] = ()) -> dict[str, Any]:
    """Why a stored transcript line reads as it does: the same rule the view applied (View.rule). `near` is the lines
    stored around it, so a label on another mic's copy of it is the one explained."""
    from liberational.listen.voices import THRESHOLD as VOICE_MATCH  # loaded when asked

    near = list(near)
    seen = view.line(e, near)  # reads the log; explain() then answers from it
    said = view.explain(e, near)
    rule = view.rule(e, said)
    sid = seen.get("speaker_id")
    out: dict[str, Any] = {
        "name": seen.get("speaker") or transcript_view.UNKNOWN,
        "subject": "line",
        "via": "line" if rule == "own" else rule,
        "similarity": _sim(e.get("speaker_sim")),
        "threshold": VOICE_MATCH,
        "voice": {"id": sid, "label": view.voice_name(sid)} if isinstance(sid, int) else None,
        "mark": {"kind": "line", "ts_start": e["ts_start"], "ts_end": e.get("ts_end") or e["ts_start"]}
        | ({"device": e["device"]} if isinstance(e.get("device"), str) and e["device"] else {}),
    } | _assertion(None)
    if rule == "own" and said is not None:
        w = said.own.winner
        system = w is not None and tier(w) is Tier.SYSTEM
        # a match's score is its own confidence; the owner's word has none (the stored score was for the old name)
        return out | _assertion(said.own) | {"similarity": _sim(w.confidence) if system and w else None}
    if rule == "detached" and said is not None:
        when = max((r.asserted_at for r in said.own.retractions), default=None)
        return out | {"asserted_at": when, "similarity": None}
    if rule == "hand":
        return out | {"similarity": None}
    if rule == "voice" and said is not None:
        return out | _assertion(said.voice)
    return out


def for_shot(row: dict[str, Any], reg: registry.Registry | None) -> dict[str, Any]:
    """Why a face the face worker found (`shot_row`) carries its name: the owner's correction of it in the event log
    (a name, or none), else the face worker's match when it was seen."""
    out: dict[str, Any] = {
        "name": row["name"],
        "subject": "shot",
        "via": "face",
        "similarity": _sim(row["sim"]),
        "threshold": FACE_MATCH,
        "voice": None,
        "mark": {"kind": "shot", "sighting": row["id"], "ts": row["ts"], "camera": row["cam"]},
    } | _assertion(None)
    own = None
    if reg is not None:
        try:
            own = reg.person_of(registry.detection(row["id"]))
        except Exception as ex:  # an unreadable log: explain from the match
            registry.LOG.warning(f"why this name: the log is unreadable: {ex!r}")
    if own is not None and own.person is not None:
        person = reg.person(own.person) if reg is not None else None
        name = person.display_name if person is not None else row["name"]
        return out | {"name": name, "via": "shot", "similarity": None} | _assertion(own)
    if own is not None and own.retractions:  # the owner said it is nobody they know
        when = max(r.asserted_at for r in own.retractions)
        return out | {"name": None, "via": "shot", "similarity": None, "asserted_at": when}
    return out


def shot_row(state: Path, sighting: str) -> dict[str, Any] | None:
    """A sighting's id, see-actions' match for it (name and score), time, and camera, or None. The sighting is the
    `face.match` row itself."""
    from liberational import segment_log
    from liberational.vision import sightings

    found = sightings.shot(sighting, state)
    if found is None:
        return None
    ts = float(found["ts"])
    matches = segment_log.at(state).range(segment_log.FACE_MATCHES, ts, ts + 0.001)
    m = next((m for m in matches if m.get("id") == sighting), {})
    return {"id": sighting, "name": m.get("name"), "sim": m.get("sim"), "ts": ts, "cam": found["cam"]}
