"""Identity linking on Continuum: learn people's names the way humans do (fully local, no face
search).

The worker (`pack-identity`) follows the broker's streams and names what they show:
  speech.utterance   (the listen worker)    speaker_id / speaker_sim per utterance (voice clusters)
  face.match         (see-actions' faces)   who each face is, or which unknown face group ("Face N") it joined
and names people from two kinds of evidence:

  1. Introductions in speech: regex candidates ("I'm X", "my name is X", "this is X", "meet X",
     "X, nice to meet you") confirmed by the local LLM (self / third-party / none).
       self-intro   -> the utterance's speaker_id gets the name
       third-party  -> the most recent *other* speaker (within OTHER_SPEAKER_S) gets the name
     The unknown face group seen within ±WINDOW_S of the utterance is named: exactly one unknown person ->
     auto-apply; several -> 'ambiguous' (pending, candidates kept); none -> voice-only proposal (pending).
     The mic has no bearing yet, so face<->voice linkage is purely temporal.
  2. The collar (the voice assistant): "that's X" / "this is X" / "meet X" from the wearer names the unknown face
     group in view as X (and labels the most recent other speaker). See collar_intro().

An unknown person is a face group: the face worker gives every unnamed face a stable "Face N" through
`vision/faces.py`'s unknown clusters, so this worker needs no face model, frame, or embedding of its own. Naming a
group (`faces.name_cluster`) enrolls its face under the name, so the face worker recognizes the person from then on,
and claims the group in the event log (people/registry.py) with the record's basis; the detections seen in the
introduction's window are claimed too (`registry.detection`), so a reader asking `person_of` about one of them is
answered. A voice is named by a claim on its voiceprint in the event log (the only record of a voice's name).

A tag can carry a role and an organization ("that's Alice, CEO of Acme"), parsed by fixed rules in people/tags.py
(step 6 of docs/DOMAIN-MODEL.md). They are kept on the record as "attributes" and given to the person in the event
log once the name is applied; a plain name tag works as before.

Every proposal and what became of it is a record in the knowledge store's `identity.record` collection
(`paths.state/knowledge.sqlite3`, outside the broker's reset; append-only, the latest record per id wins):
  {id, ts, kind: self_intro|third_party|badge|collar, name, status: applied|pending|ambiguous|rejected,
   confidence, speaker_id, speaker_label, faces:[{cluster, cam, cams, ts, sim, n, detections}],
   attributes:{role, organization} (only when the tag said them),
   evidence:{text, ts_start, ts_end, regex, llm, caption, vlm}, applied:{speaker, face}}
A face candidate names its face group and the `face.detection` payloads it was seen in; the frames are in the
recording. The worker's place in the speech stream (the segment store) is a cursor saved in
`identity.speech.cursor`: it moves past a line once its measurements are written and its introduction read, so a
restart takes up after the last line handled.

CLI
  pack-identity                       the worker (services.toml: identity)
  pack-identity pending               list pending / ambiguous proposals
  pack-identity confirm <id> <name> [--face N]   apply a proposal (optionally pick face candidate N)
  pack-identity reject <id>
  pack-identity enroll-visible <name> what the collar hook does (name the one unknown face group in view)
  --state-dir DIR --faces-db FILE     run against a different state dir / face DB (tests)

The owner's own name is applied only on the owner's word. Heard from someone else or read off a badge, it stays
pending for `confirm`, so a guest's face never joins the owner's face row.
"""

import argparse
import asyncio
import collections
import datetime as dt
import functools
import json
import os
import pathlib
import queue
import re
import signal
import threading
import time
import uuid
from collections.abc import Callable, Collection, Iterable, Sequence
from typing import Any

from liberational import config as cfg
from liberational import json_state, knowledge, logs, ollama, ollama_health, segment_log, speech_log
from liberational.domain.identity import Basis
from liberational.memory import quantities
from liberational.people import registry, tags, transcript_view

LLM = os.environ.get("PACK_IDENTITY_LLM", cfg.get("models.ollama.llm"))
DEFAULT_STATE = cfg.path("paths.state")

WINDOW_S = 8.0  # face sighting window around an introduction (±)
OTHER_SPEAKER_S = 45.0  # "most recent other speaker" must have spoken this recently
SIGHTING_KEEP_S = 180.0  # ring buffer of unknown-face sightings
VISIBLE_S = 12.0  # the collar names a face group matched this recently
POLL_S = 1.0
POLL_FACES_S = 1.0  # how long one wait for new face matches blocks its thread (stopping waits at most this)
DUP_S = 30.0  # same name applied within this -> duplicate, skip
APPLY_CONF = 0.85
RECORDS = "identity.record"  # the knowledge store collection of the records
SPEECH_CURSOR = "identity.speech.cursor"  # the worker's place in the speech stream, under paths.state
NS = 1_000_000_000
FACE_GROUP = re.compile(r"Face (\d+)")

LOG = logs.get("identity")


def log(*a: object, level: int = logs.INFO) -> None:
    LOG.log(level, " ".join(str(x) for x in a))


def auto_label(label: str | None) -> bool:
    """A label no one gave: an automatic "Speaker 4", "unknown", or none, in any capitals."""
    return bool(re.fullmatch(r"Speaker \d+", label or "", re.IGNORECASE)) or (label or "").casefold() in ("", "unknown")


# ------------------------------------------------------------------ paths ----
class Paths:
    """The state directory the worker names people in (the broker's socket and the voice list are under it), and
    the face gallery it enrolls them in (`vision/faces.py`'s, unless a test gives another)."""

    def __init__(
        self, state_dir: str | pathlib.Path = DEFAULT_STATE, faces_db: str | pathlib.Path | None = None
    ) -> None:
        self.state = pathlib.Path(state_dir)
        self.faces_db = pathlib.Path(faces_db) if faces_db else None
        if self.faces_db:
            from liberational.vision import faces as face_id

            face_id.DB = self.faces_db


# ---------------------------------------------------------------- records ----
class Records:
    """The identity records in the knowledge store (`identity.record`, outside the broker's reset): one
    record per version, keyed by the record's id, the latest per id winning. These
    are the same records `pack-knowledge import` writes from the broker's old stream."""

    def __init__(self, store: knowledge.Store) -> None:
        self.store = store

    def append(self, rec: dict[str, Any]) -> dict[str, Any]:
        """Commits `rec` as the newest version of its id. A record without an id is refused."""
        if not rec.get("id"):
            raise ValueError("an identity record needs an id")
        # A plain JSON copy: a record can hold numpy numbers and other values that only print as text.
        value = json.loads(json.dumps(rec, ensure_ascii=False, default=str))
        now = time.time_ns()
        self.store.put([knowledge.Record(str(uuid.uuid7()), RECORDS, str(rec["id"]), now, now, now, value)])
        return rec

    def all(self) -> collections.OrderedDict[str, dict[str, Any]]:
        """The latest record per id, in first-seen order."""
        latest: collections.OrderedDict[str, dict[str, Any]] = collections.OrderedDict()
        for row in self.store.query(RECORDS):
            if row.record.key is not None:
                latest[row.record.key] = row.record.value
        return latest


def records(paths: Paths) -> Records:
    """The records in the knowledge store under `paths.state`."""
    return Records(knowledge.at(registry.knowledge_path(paths.state)))


def append_record(paths: Paths, rec: dict[str, Any]) -> dict[str, Any]:
    return records(paths).append(rec)


def load_records(paths: Paths) -> collections.OrderedDict[str, dict[str, Any]]:
    """Latest record per id, in first-seen order."""
    return records(paths).all()


# -------------------------------------------------------------- sightings ----
def face_group_of(match: dict[str, Any]) -> int | None:
    """The unknown face group a `face.match` record joined ("Face N" -> N), or None for a named face."""
    from liberational.vision import face_rejects

    if match.get("name") or str(match.get("id")) in face_rejects.rejected():  # rejected from its group
        return None
    m = FACE_GROUP.fullmatch(str(match.get("cluster") or ""))
    return int(m.group(1)) if m else None


def sighting_of(match: dict[str, Any]) -> dict[str, Any] | None:
    """An unknown-face sighting from a `face.match` record, or None for a named face (or a record with no group)."""
    group = face_group_of(match)
    if group is None:
        return None
    return {
        "ts": float(match.get("ts_start") or 0.0),
        "cam": match.get("cam") or "cam?",
        "cluster": group,
        "detection": match.get("detection"),
        "track_id": match.get("track_id"),
        "sim": float(match.get("sim") or 0.0),
    }


def cluster_sightings(sightings: Iterable[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Unknown sightings grouped into people: one group per face group, in the order first seen (the face worker
    already decided which sightings are one person)."""
    groups: dict[int, list[dict[str, Any]]] = {}
    for s in sightings:
        groups.setdefault(int(s["cluster"]), []).append(s)
    return list(groups.values())


def face_candidate(cluster: Sequence[dict[str, Any]], idx: int) -> dict[str, Any]:
    """One unknown person as a record's face candidate: its face group, where and when it was last seen, and the
    detections it was seen in (the newest 50), for the claims `apply_record` writes."""
    best = max(cluster, key=lambda s: (s["sim"], s["ts"]))
    last = max(cluster, key=lambda s: s["ts"])
    cams = collections.Counter(s["cam"] for s in cluster)
    return {
        "cluster": int(best["cluster"]),
        "cam": cams.most_common(1)[0][0],
        "cams": sorted(cams),
        "ts": last["ts"],
        "sim": best["sim"],
        "n": len(cluster),
        "detections": [s["detection"] for s in sorted(cluster, key=lambda s: s["ts"])[-50:] if s.get("detection")],
    }


# ------------------------------------------------------------ voices ----
def owners_name(name: str) -> bool:
    """Whether name is the owner's (owner.name or owner.speaker_label), matched as the event log matches names
    (registry.norm: any capitals and spacing), since a voice's name is its person there (assistant/voice.py
    owner_voice)."""
    return bool(cfg.owner_label()) and registry.norm(name) in (
        registry.norm(cfg.owner_label()),
        registry.norm(cfg.owner_name()),
    )


def label_speaker(
    paths: Paths, sid: int | str | None, name: str, basis: Basis = Basis.INTRODUCTION
) -> tuple[bool, str]:
    """Name a voiceprint (a claim in the event log, with `basis`). Refuses to rename a voice that already names
    someone (its person in the event log) to a different name."""
    if sid is None:
        return False, "no speaker id"
    try:
        n = int(sid)
    except ValueError:
        return False, f"bad speaker id {sid!r}"
    known = 1 <= n < registry.handed_out(pathlib.Path(paths.state))
    if not known and n not in registry.at(paths.state).voiceprints():
        return False, f"no voice {n}"
    current = registry.voice_names(paths.state, [n]).get(n)
    if current is not None and registry.norm(current) == registry.norm(name):
        return True, "already"
    # A guest saying "I'm <owner>" must not get the owner's label (it gates owner-only requests by voice).
    if owners_name(name):
        return False, "won't auto-label a voice as the owner; enroll the owner's voice or label it on the dashboard"
    if current is not None:
        return False, f"speaker {sid} already labelled {current!r}"
    registry.at(paths.state).claim_many([registry.Claim(registry.voiceprint(n), name)], basis, "identity")
    return True, f"Speaker {n} -> {name}"


# -------------------------------------------------------------- regex + LLM ----
NAME = r"([A-Z][a-z]+(?:[-'][A-Z][a-z]+)?(?:\s+[A-Z][a-z]+)?)"
INTRO_PATTERNS = [
    ("self", re.compile(r"\b(?i:I'm|I am|my name is|my name's|the name's|call me|it's me,?)\s+" + NAME + r"\b")),
    ("self", re.compile(r"\b(?i:my name is)\s+([a-z]+)\b")),  # lowercase ASR
    (
        "third",
        re.compile(
            r"\b(?i:this is|that's|that is|meet|say hi to|say hello to|introduce you to|introducing|here's|this here is)\s+(?:(?i:my)\s+(?:[a-z]+\s+){0,2})?"
            + NAME
            + r"\b"
        ),
    ),
    (
        "third",
        re.compile(
            r"^\W*"
            + NAME
            + r"\W*,?\s+(?i:nice|good|great|pleased|lovely|happy)\s+(?i:to)\s+(?i:finally\s+)?(?i:meet)\s+(?i:you)"
        ),
    ),
    (
        "third",
        re.compile(
            r"(?i:nice|good|great|pleased|lovely|happy)\s+(?i:to)\s+(?i:finally\s+)?(?i:meet)\s+(?i:you),?\s+"
            + NAME
            + r"\b"
        ),
    ),
]
STOP = {
    "not",
    "so",
    "just",
    "sure",
    "here",
    "sorry",
    "fine",
    "good",
    "okay",
    "ok",
    "done",
    "back",
    "glad",
    "happy",
    "really",
    "still",
    "also",
    "gonna",
    "going",
    "the",
    "this",
    "that",
    "what",
    "why",
    "how",
    "where",
    "when",
    "who",
    "it",
    "very",
    "all",
    "pretty",
    "kind",
    "sort",
    "only",
    "already",
    "about",
    "afraid",
    "right",
    "wrong",
    "trying",
    "thinking",
    "looking",
    "like",
    "me",
    "you",
    "he",
    "she",
    "they",
    "we",
    "your",
    "my",
    "a",
    "an",
    "in",
    "on",
    "at",
    "to",
    "from",
    "with",
    "there",
    "now",
    "then",
    "too",
    "up",
    "down",
    "out",
    "off",
    "over",
    "again",
    "never",
    "always",
    "sir",
    "man",
    "dude",
    "guys",
    "everyone",
    "everybody",
    "nobody",
    "someone",
    "somebody",
    "mom",
    "dad",
    "doctor",
    "mister",
    "mr",
    "mrs",
    "ms",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "american",
    "english",
    "chinese",
    "new",
    "old",
    "big",
    "little",
    "i",
    "oh",
    "hey",
    "hi",
    "well",
}


def clean_name(n: str | None) -> str | None:
    if not n:
        return None
    n = re.sub(r"[^A-Za-z' -]", "", n).strip(" '-")
    words = [w for w in n.split() if w]
    if not words or len(words) > 3:
        return None
    if words[0].lower() in STOP or n.lower() == cfg.assistant_name().lower():
        return None  # the assistant is never a person to enroll
    return " ".join(w.capitalize() if (w.isupper() or w.islower()) else w for w in words)  # ALEX / dana -> Alex / Dana


def spoken_name(raw: str) -> str | None:
    """A name as ASR writes one (capitalized, at most two words), cleaned, else None."""
    return clean_name(raw) if re.fullmatch(NAME, raw) else None


def spoken_tag(text: str, anywhere: bool = True) -> tags.Tag | None:
    """The name, role, and organization in an introduction ("this is Alice, CEO of Acme"), or None (people/tags.py)."""
    return tags.parse(text, spoken_name, anywhere=anywhere)


def same_name(a: str, b: str) -> bool:
    """Whether two names can be one person's: the same first name, and the same surname when both give one. So
    "Alice" and "Alice Smith" can be, and "Alice Smith" and "Alice Jones" cannot."""
    x, y = a.lower().split(), b.lower().split()
    return bool(x and y) and x[0] == y[0] and (len(x) == 1 or len(y) == 1 or x == y)


def regex_intro(text: str) -> tuple[str, str] | None:
    """-> (kind, name) for the first matching pattern, else None."""
    for kind, rx in INTRO_PATTERNS:
        m = rx.search(text)
        if m:
            n = clean_name(m.group(1))
            if n:
                return kind, n
    return None


LLM_PROMPT = """You are tagging an ambient speech transcript (automatic transcription, casual talk, possible errors) to learn people's names.
Task: decide whether THIS utterance introduces the name of a person who is physically present and talking, i.e. a moment where
a listener would learn what to call someone. Reason briefly, then answer.

Return ONLY a JSON object with these keys, in this order:
  "reason": one short sentence
  "named_thing": what the candidate word refers to: "person_present" | "person_absent" | "place" | "brand_or_thing" | "possessive_or_other" | "assistant"
  "intro": "self" if the speaker gives their own name ("I'm Dana", "my name is Dana", "it's Dana", a caller saying "this is Dana"),
           "third" if the speaker introduces or greets someone ELSE who is present ("this is Dana", "meet Dana", "Dana, nice to meet you"),
           "none" otherwise (e.g. "I'm sure", "this is Seattle", "this is Dana's phone", "meet Dana at five" = a plan, talking about someone absent)
  "name": the person's name as it should be stored (first name, or first + last if both were said), or ""

Previous line: {prev}
Utterance (speaker {spk}): "{text}"
"""


def llm_confirm(
    text: str, prev: str | None, spk: str, kind: str | None = None, name: str | None = None
) -> dict[str, Any] | None:
    """Local LLM, JSON mode, reasoning-first (the pattern candidate is deliberately NOT shown: with it the model just
    parroted the regex). -> dict(intro, name, raw) or None on failure. ~1 s per call, only on regex hits."""
    try:
        reply = ollama.chat(
            LLM,
            [{"role": "user", "content": LLM_PROMPT.format(prev=prev or "(none)", spk=spk, text=text)}],
            # num_ctx on EVERY model: one asked without it loads at 262k (24 GB) and evicts the agent brain
            num_ctx=ollama_health.VLM_NUM_CTX,
            num_predict=160,
            temperature=0.0,
            schema="json",
            think=False,
            truncate=False,
            timeout=45,
        )
        d = json.loads(ollama.text(reply))
        intro = str(d.get("intro", "none")).lower().strip()
        nm = clean_name(str(d.get("name", "")))
        if intro not in ("self", "third") or d.get("named_thing") not in (None, "person_present"):
            intro = "none"
        return {
            "intro": intro,
            "name": nm,
            "raw": json.dumps({k: d.get(k) for k in ("intro", "name", "named_thing", "reason")})[:300],
        }
    except ollama_health.Unavailable:
        return None  # logged once by ollama_health
    except Exception as e:
        log(f"llm confirm failed: {e}", level=logs.WARN)
        return None


def new_id() -> str:
    return "i-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]


def apply_record(
    paths: Paths,
    rec: dict[str, Any],
    name: str,
    face_idx: int | None = None,
    basis: Basis | None = None,
    owner: bool = False,
) -> dict[str, Any]:
    """Apply name to the record's speaker + face candidate. Mutates rec (status/applied), returns rec. `basis` is
    the assertions' basis; by default it follows the record's kind (registry.identity_basis). `owner` says the owner
    gave the name (their command), as an OWNER_TAG basis does (their confirm): see record_attributes. Only the owner
    gives their own name: one heard from someone else or read off a badge stays pending for `confirm`, and names no
    voice, face, or role. Else a guest's face joins the owner's face row (and is then recognized as the owner).
    Visual text, including badge OCR, is only a proposal until the owner confirms it: clothing or signage must
    never enroll a face or voice by itself. A `face_idx` the record has no face for raises ValueError before anything
    is applied."""
    faces = rec.get("faces") or []
    if face_idx is not None and not 0 <= face_idx < len(faces):
        raise ValueError(f"no face candidate {face_idx}: the record has {len(faces)}")
    basis = basis or registry.identity_basis(rec)
    owner = owner or basis is Basis.OWNER_TAG
    if not owner and rec.get("kind") == "badge":
        rec["name"] = name
        rec["applied"] = {
            "speaker": False,
            "face": False,
            "notes": ["visual text requires explicit confirmation before naming anyone"],
        }
        rec["status"] = "pending"
        return rec
    if not owner and owners_name(name):
        rec["name"] = name
        rec["applied"] = {
            "speaker": False,
            "face": False,
            "notes": ["won't name anyone as the owner unless the owner confirms it"],
        }
        rec["status"] = "pending"
        return rec
    applied = {"speaker": False, "face": False, "notes": []}
    if rec.get("speaker_id") is not None:
        ok, why = label_speaker(paths, rec["speaker_id"], name, basis)
        applied["speaker"] = ok
        applied["notes"].append(why)
    pick = None
    if face_idx is not None:
        pick = faces[face_idx]
    elif len(faces) == 1:
        pick = faces[0]
    elif len(faces) > 1:
        applied["notes"].append(f"{len(faces)} face candidates, none picked")
    if pick:
        try:
            from liberational.vision import faces

            heard = rec.get("evidence") or {}
            ts = heard.get("ts_start")
            evidence = [registry.utterance_evidence(ts, heard.get("device"))] if ts is not None else []
            confidence = rec.get("confidence") or 1.0
            faces.name_cluster(
                pick["cluster"], name, basis=basis, source="identity", confidence=confidence, evidence=evidence
            )
            applied["face"] = True
            applied["notes"].append(f"face group {pick['cluster']} named")
            with registry.recorded("identity detections"):
                claims = [
                    registry.Claim(registry.detection(d), name, confidence, tuple(evidence))
                    for d in pick.get("detections") or []
                ]
                if claims:
                    registry.at(paths.state).claim_many(claims, basis, "identity")
        except Exception as e:
            applied["notes"].append(f"face enroll failed: {e}")
    rec["name"] = name
    rec["applied"] = applied
    rec["status"] = "applied" if (applied["speaker"] or applied["face"]) else "pending"
    record_attributes(paths, rec, name, owner=owner)
    return rec


def record_attributes(paths: Paths, rec: dict[str, Any], name: str, owner: bool = False) -> None:
    """Give the person the role and organization their tag said, once the record is applied. A pending record waits:
    its name may still change at `confirm`. Only the owner's tag replaces a role or organization the person has; one
    the daemon overheard or read off a badge fills in only what is missing (decision 2: the owner's word wins)."""
    attrs = rec.get("attributes")
    if attrs and rec.get("status") == "applied":
        try:
            with registry.recorded("tag attributes"):
                registry.at(paths.state).describe(name, "identity", attrs, fill=not owner)
        except registry.WriteFailed as e:
            rec.setdefault("applied", {}).setdefault("notes", []).append(str(e))


def _ts(line: dict[str, Any]) -> float:
    """A transcript line's ts_start, or 0 when it has none."""
    try:
        return float(line.get("ts_start") or 0)
    except TypeError, ValueError:
        return 0.0


# ---------------------------------------------------------------- the brain ----


# ---------------------------------------------------------------- the brain ----
class Identity:
    """The worker's brain: fed records as they're committed (`on_match`, `on_lines`), it keeps the unknown-face
    sightings of the last SIGHTING_KEEP_S, proposes names, and applies the sure ones."""

    def __init__(self, paths: Paths, clock: Callable[[], float] = time.time) -> None:
        self.p = paths
        self.clock = clock
        self.sightings: collections.deque[dict[str, Any]] = collections.deque()  # unknown-face sightings
        self.pending_intros: list[tuple[float, dict[str, Any]]] = []  # intros waiting for their ±WINDOW_S of faces
        self.recent_lines: collections.deque[dict[str, Any]] = collections.deque(maxlen=50)

    # -- faces
    def trim(self) -> None:
        """Forgets sightings older than SIGHTING_KEEP_S."""
        now = self.clock()
        while self.sightings and self.sightings[0]["ts"] < now - SIGHTING_KEEP_S:
            self.sightings.popleft()

    def on_match(self, match: dict[str, Any]) -> None:
        """A `face.match` record: an unknown face is a sighting."""
        s = sighting_of(match)
        if s is not None:
            self.sightings.append(s)

    # -- speech
    def on_lines(self, lines: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Utterances as the speech log commits them: their spoken measurements are written first (so a failure
        leaves the lines to be read again), then each is read for an introduction. Returns the lines handled."""
        lines = [e for e in lines if isinstance(e, dict)]
        if not lines:
            return []
        view = transcript_view.View(self.p.state)
        resolved = view.speakers(lines)
        # An unreadable existing registry cannot safely become anonymous attribution.
        if view.reg is None and registry.log_exists(self.p.state):
            raise RuntimeError("cannot record spoken measurements while the identity log is unreadable")
        quantities.record(self.p.state, resolved)
        for e, _person in resolved:
            self.on_line(e)
        return [e for e, _person in resolved]

    def on_line(self, e: dict[str, Any]) -> dict[str, Any] | None:
        text = e.get("text")
        if not isinstance(text, str):
            return None  # a line with no words (or malformed) introduces nobody
        hit = regex_intro(text)
        prev = self.recent_lines[-1] if self.recent_lines else None
        self.recent_lines.append(e)
        if not hit:
            return None
        kind, name = hit
        spk = e.get("speaker") or f"Speaker {e.get('speaker_id')}"
        llm = llm_confirm(text, (prev or {}).get("text"), spk, kind, name)
        if llm is None:
            conf, kind_f, name_f = 0.4, kind, name  # Ollama down: regex only -> pending
        elif llm["intro"] == "none" or not llm["name"]:
            log(f"intro candidate rejected by LLM: {text!r} ({llm['raw']!r})")
            return None
        else:
            kind_f, name_f, conf = llm["intro"], llm["name"], 0.6
            if name_f.split()[0].lower() != name.split()[0].lower():
                conf -= 0.1  # LLM and regex disagree on the name
        kind_f = "self_intro" if kind_f == "self" else "third_party"
        tag = spoken_tag(text)
        same = tag and same_name(tag.name, name_f)  # the role belongs to this name
        rec: dict[str, Any] = {
            "id": new_id(),
            "ts": self.clock(),
            "kind": kind_f,
            "name": name_f,
            "confidence": conf,
            "evidence": {
                "text": text,
                "ts_start": e.get("ts_start"),
                "ts_end": e.get("ts_end"),
                "device": e.get("device"),
                "regex": f"{kind}:{name}",
                "llm": (llm or {}).get("raw"),
                "speaker": spk,
                "speaker_sim": e.get("speaker_sim"),
            },
        }
        if same and tag.attributes():
            rec["attributes"] = tag.attributes()
        # whose voice?
        if kind_f == "self_intro":
            rec["speaker_id"], rec["speaker_label"] = e.get("speaker_id"), spk
        else:
            other = next(
                (
                    l
                    for l in reversed(list(self.recent_lines)[:-1])
                    if l.get("speaker_id") not in (None, e.get("speaker_id"))
                    and e.get("ts_start", 0) - l.get("ts_end", 0) <= OTHER_SPEAKER_S
                ),
                None,
            )
            rec["speaker_id"] = other.get("speaker_id") if other else None
            rec["speaker_label"] = other.get("speaker") if other else None
            rec["evidence"]["other_line"] = other.get("text") if other else None
        if (
            rec["speaker_id"] is not None
            and not auto_label(rec["speaker_label"])
            and rec["speaker_label"].lower() != name_f.lower()  # ty: ignore[unresolved-attribute]  # auto_label(None) is True
        ):
            rec["evidence"]["note"] = f"speaker already labelled {rec['speaker_label']!r}; voice not relabelled"
            rec["speaker_id"] = None
            conf -= 0.1
        rec["confidence"] = conf
        # duplicate (e.g. the collar hook already applied the same name)
        for r in reversed(list(load_records(self.p).values())):
            if (
                r.get("status") == "applied"
                and r.get("name", "").lower() == name_f.lower()
                and self.clock() - r.get("ts", 0) <= DUP_S
            ):
                log(f"intro {name_f!r} duplicate of {r['id']} ({r['kind']}), skipping")
                return None
        deadline = (e.get("ts_end") or self.clock()) + WINDOW_S
        self.pending_intros.append((deadline, rec))
        log(f"intro candidate: {kind_f} {name_f!r} conf {conf:.2f}, waiting for scene until +{WINDOW_S:.0f}s")
        return rec

    def resolve_intros(self) -> list[dict[str, Any]]:
        now = self.clock()
        out = []
        for deadline, rec in list(self.pending_intros):
            if now < deadline:
                continue
            self.pending_intros.remove((deadline, rec))
            t0 = (rec["evidence"].get("ts_start") or rec["ts"]) - WINDOW_S
            t1 = (rec["evidence"].get("ts_end") or rec["ts"]) + WINDOW_S
            inwin = [s for s in self.sightings if t0 <= s["ts"] <= t1]
            clusters = cluster_sightings(inwin)
            rec["faces"] = [face_candidate(c, i) for i, c in enumerate(clusters)]
            rec["evidence"]["sightings"] = len(inwin)
            rec["evidence"]["unknown_people"] = len(clusters)
            if len(clusters) == 1:
                rec["confidence"] = round(rec["confidence"] + 0.3, 2)
            elif len(clusters) > 1:
                rec["confidence"] = round(rec["confidence"] - 0.1, 2)
            if len(clusters) == 1 and rec["confidence"] >= APPLY_CONF:
                apply_record(self.p, rec, rec["name"])
            elif len(clusters) > 1:
                rec["status"] = "ambiguous"
            else:
                rec["status"] = "pending"
            append_record(self.p, rec)
            out.append(rec)
            log(
                f"{rec['kind']} {rec['name']!r}: {len(inwin)} sightings / {len(clusters)} unknown people, conf {rec['confidence']:.2f} -> {rec['status']}"
                + (f" {rec['applied']['notes']}" if rec.get("applied") else "")
            )
        return out

    def tick(self) -> list[dict[str, Any]]:
        """Between records: resolve the intros whose window closed, and forget old sightings."""
        self.trim()
        return self.resolve_intros()

    def run(self, duration: float | None = None) -> None:
        """The worker: follows the speech stream from its saved place (every line handled moves it) and the face
        matches live, and ticks every POLL_S. The brain runs on one thread (the LLM calls take seconds); the
        followers only pump records to it."""
        log(f"worker on {self.p.state} (faces db {self.p.faces_db or 'default'}), LLM {LLM}")
        asyncio.run(self._run(duration))

    async def _run(self, duration: float | None) -> None:
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, stop.set)
            except NotImplementedError, RuntimeError:  # not the main thread
                pass
        if duration:
            loop.call_later(duration, stop.set)
        work: queue.Queue[tuple[str, Any] | None] = queue.Queue(maxsize=4096)
        acked: asyncio.Queue[int] = asyncio.Queue()
        brain = threading.Thread(target=self._consume, args=(work, loop, acked), name="identity-brain", daemon=True)
        brain.start()
        tasks = [
            asyncio.create_task(self._follow_speech(work, acked, stop)),
            asyncio.create_task(self._follow_faces(work, stop)),
        ]
        # Lines heard from a TV while one is on, marked by ear (listen/media_tag.py), on a thread of its own.
        from liberational.listen import media_tag

        tagging = threading.Event()
        media_tag.start(self.p.state, tagging)
        try:
            await stop.wait()
        finally:
            tagging.set()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            work.put(None)
            brain.join(timeout=5)

    def _consume(self, work: queue.Queue[tuple[str, Any] | None], loop: Any, acked: Any) -> None:
        """The brain thread: records off `work` in order, a tick every POLL_S whether or not records keep coming
        (matches arrive about once a second per camera), and the speech position acknowledged only up to the newest
        line handled with nothing failed before it: an acknowledgment covers everything before it, so a line whose
        measurements couldn't be written is tried again each tick and holds the acknowledgments back until it goes
        through (a restart then takes it up again)."""
        last_tick = time.monotonic()
        failed: list[tuple[int, dict[str, Any]]] = []  # lines whose handling failed, oldest first
        newest = -1  # the newest position handled
        acknowledged = -1

        def acknowledge() -> None:
            nonlocal acknowledged
            if not failed and newest > acknowledged:
                acknowledged = newest
                loop.call_soon_threadsafe(acked.put_nowait, newest)

        while True:
            try:
                got = work.get(timeout=max(0.0, POLL_S - (time.monotonic() - last_tick)))
            except queue.Empty:
                got = ()
            if time.monotonic() - last_tick >= POLL_S or got == ():
                last_tick = time.monotonic()
                self._safely(self.tick)
                while failed and self._safely(functools.partial(self.on_lines, [failed[0][1]])) is not None:
                    newest = max(newest, failed.pop(0)[0])
                acknowledge()
            if got == ():
                continue
            if got is None:
                return
            kind, payload = got
            if kind == "line":
                position, line = payload
                if self._safely(functools.partial(self.on_lines, [line])) is None:
                    failed.append((position, line))
                else:
                    newest = max(newest, position)
                    acknowledge()
            elif kind == segment_log.FACE_MATCHES:
                self._safely(functools.partial(self.on_match, payload))

    def _safely[T](self, fn: Callable[[], T]) -> T | None:
        try:
            return fn()
        except Exception as e:
            import traceback

            log("step error:", e, level=logs.ERROR)
            traceback.print_exc()
            return None

    async def _follow_speech(
        self,
        work: queue.Queue[tuple[str, Any] | None],
        acked: asyncio.Queue[int],
        stop: asyncio.Event,
    ) -> None:
        """Lines after the worker's saved place (or from now, the first time: history is never replayed, which once
        re-applied a stale introduction three times), each acknowledged once the brain handled it. The place is a
        segment cursor saved in SPEECH_CURSOR (state), moved to just past each line acknowledged."""
        path = self.p.state / SPEECH_CURSOR
        position = 0  # numbers lines for the brain; kept across rounds, since it acknowledges only higher ones
        while not stop.is_set():
            try:
                saved = _read_cursor(path)
                follower = await asyncio.to_thread(
                    speech_log.Follower, (speech_log.UTTERANCES,), saved, base=segment_log.root(self.p.state)
                )
                if saved is None:
                    _write_cursor(path, follower.cursor)
                past: dict[int, speech_log.Cursor] = {}  # each line's position and the cursor just past it
                saver = asyncio.create_task(_save_handled(path, past, acked))
                try:
                    while not stop.is_set():
                        for line, cursor in await asyncio.to_thread(follower.poll, 1.0):
                            position += 1
                            past[position] = cursor
                            await asyncio.to_thread(work.put, ("line", (position, line)))
                finally:
                    saver.cancel()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log(f"following {speech_log.UTTERANCES} stopped: {e}; again in {POLL_S:.0f} s", level=logs.WARN)
                await asyncio.sleep(POLL_S)

    async def _follow_faces(self, work: queue.Queue[tuple[str, Any] | None], stop: asyncio.Event) -> None:
        """Face matches from now, from the segment store (see-actions' head faces); sightings are a short
        memory, so nothing is replayed."""
        while not stop.is_set():
            try:
                tail = await asyncio.to_thread(
                    segment_log.Tail, segment_log.root(self.p.state), segment_log.FACE_MATCHES, time.time()
                )
                while not stop.is_set():
                    for record in await asyncio.to_thread(tail.poll, POLL_FACES_S):
                        await asyncio.to_thread(work.put, (segment_log.FACE_MATCHES, record))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log(f"following {segment_log.FACE_MATCHES} stopped: {e}; again in {POLL_S:.0f} s", level=logs.WARN)
                await asyncio.sleep(POLL_S)


def _read_cursor(path: pathlib.Path) -> speech_log.Cursor | None:
    """The saved speech cursor, or None when there's none yet (or it can't be read: then from now)."""
    try:
        got = json.loads(path.read_text())
    except OSError, ValueError:
        return None
    if not isinstance(got, dict):
        return None
    return {str(s): {str(k): int(v) for k, v in c.items()} for s, c in got.items() if isinstance(c, dict)}


def _write_cursor(path: pathlib.Path, cursor: speech_log.Cursor) -> None:
    json_state.write(path, cursor, sort_keys=True)


async def _save_handled(path: pathlib.Path, past: dict[int, speech_log.Cursor], acked: asyncio.Queue[int]) -> None:
    """Saves the cursor past each position the brain finished with (an acknowledgment covers every line before it),
    so the saved place moves."""
    while True:
        position = await acked.get()
        if (cursor := past.get(position)) is not None:
            await asyncio.to_thread(_write_cursor, path, cursor)
        for done in [p for p in past if p <= position]:
            del past[done]


# ------------------------------------------------------------- collar hook ----
# The wearer introducing someone: "that's Dana" / "this is Dana" / "meet Dana" (the assistant's wake word already removed).
INTRO_COMMAND = re.compile(
    r"^\W*(?i:that's|that is|this is|meet|say hi to)\s+(?:(?i:my)\s+(?:[a-z]+\s+){0,2})?([A-Z][a-z]+(?:[-'][A-Z][a-z]+)?(?:\s+[A-Z][a-z]+)?)\W*$"
)


def tag_command(text: str) -> tags.Tag | None:
    """The wearer's command with a role or organization ("that's Alice, CEO of Acme"), else None. A plain name is
    INTRO_COMMAND's, so it is handled exactly as before."""
    tag = tags.parse(text, spoken_name, leads=tags.COMMAND_LEADS)
    return tag if tag and tag.attributes() else None


def visible_unknown_people(paths: Paths, max_age_s: float = VISIBLE_S) -> list[list[dict[str, Any]]]:
    """The unknown face groups matched on any camera in the last `max_age_s`, each as its sightings."""
    now = time.time()
    matches = segment_log.at(paths.state).range(segment_log.FACE_MATCHES, now - max_age_s, now + 1.0)
    return cluster_sightings(s for m in matches if (s := sighting_of(m)) is not None)


def recent_other_speaker(
    paths: Paths, exclude_labels: Collection[str] | None = None, within_s: float = OTHER_SPEAKER_S
) -> dict[str, Any] | None:
    """The most recent utterance by someone other than the collar wearer (the owner by default), read through the
    event log."""
    if exclude_labels is None:
        exclude_labels = tuple(x for x in (cfg.owner_label(),) if x)
    excluded = {x.lower() for x in exclude_labels}
    recent = sorted(speech_log.shared().recent(within_s), key=lambda e: e.get("ts_end") or e.get("ts_start") or 0)
    for e in transcript_view.view(reversed(recent), paths.state):  # newest first
        if e.get("speaker_id") is not None and (e.get("speaker") or "").lower() not in excluded:
            return e
    return None


def collar_intro(
    name: str | None,
    state_dir: str | pathlib.Path = DEFAULT_STATE,
    faces_db: str | pathlib.Path | None = None,
    source: str = "collar",
    attributes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """pack.py hook: the wearer said "that's NAME" / "this is NAME". Enrol the one visible unknown face as NAME and label the most
    recent other speaker. `attributes` are the role and organization said with the name (tag_command), given to the
    person once the name is applied, as the owner's word. Returns dict(reply, record)."""
    paths = Paths(state_dir, faces_db)
    name = clean_name(name)
    if not name:
        return {"reply": "I didn't catch the name.", "record": None}
    attributes = {k: v for k, v in (attributes or {}).items() if v}
    said = tags.Tag(name, **attributes).described()
    clusters = visible_unknown_people(paths)
    other = recent_other_speaker(paths)
    rec: dict[str, Any] = {
        "id": new_id(),
        "ts": time.time(),
        "kind": source,
        "name": name,
        "speaker_id": other.get("speaker_id") if other else None,
        "speaker_label": other.get("speaker") if other else None,
        "evidence": {
            "text": f"collar: {name}",
            "other_line": other.get("text") if other else None,
            "unknown_people": len(clusters),
        },
    }
    if attributes:
        rec["attributes"] = attributes
    rec["faces"] = [face_candidate(c, i) for i, c in enumerate(clusters)]
    if len(clusters) == 1:
        rec["confidence"] = 0.95
        apply_record(paths, rec, name, owner=True)
        reply = f"Got it, that's {name}{', ' + said if said else ''}." + (
            " I've got their voice too." if rec["applied"]["speaker"] else ""
        )
    elif len(clusters) > 1:
        rec["confidence"] = 0.5
        rec["status"] = "ambiguous"
        reply = f"I see {len(clusters)} people I don't know. Which one is {name}? I've saved it as pending."
    else:
        rec["confidence"] = 0.5
        rec["status"] = "pending"
        if rec["speaker_id"] is not None and auto_label(rec["speaker_label"]):
            ok, why = label_speaker(paths, rec["speaker_id"], name, registry.identity_basis(rec))
            if ok:
                rec["status"] = "applied"
                rec["applied"] = {"speaker": True, "face": False, "notes": [why]}
                record_attributes(paths, rec, name, owner=True)
        reply = (
            f"I don't see an unknown face right now, but I've tagged the last voice as {name}."
            if rec["status"] == "applied"
            else f"I don't see an unknown face right now. Saved {name} as pending."
        )
    append_record(paths, rec)
    return {"reply": reply, "record": rec}


# ---------------------------------------------------------------------- CLI ----


# ---------------------------------------------------------------------- CLI ----
def fmt_ts(t: float | None) -> str:
    return dt.datetime.fromtimestamp(t).strftime("%m-%d %H:%M:%S") if t else "-"


def cmd_pending(paths: Paths) -> None:
    recs = [r for r in load_records(paths).values() if r.get("status") in ("pending", "ambiguous")]
    if not recs:
        print("(no pending proposals)")
        return
    for r in recs:
        faces = r.get("faces") or []
        print(
            f"{r['id']}  {fmt_ts(r['ts'])}  {r['kind']:<12} {r['name']:<14} {r['status']:<9} conf {r.get('confidence', 0):.2f}  "
            f"voice: {r.get('speaker_label') or '-'}  faces: {len(faces)}"
        )
        ev = r.get("evidence", {})
        if ev.get("text"):
            print(f"    said: {ev['text']!r}")
        for i, f in enumerate(faces):
            print(f"    face[{i}] Face {f['cluster']} on {f['cam']} n={f['n']} last {fmt_ts(f.get('ts'))}")
    print("\nconfirm: pack-identity confirm <id> <name> [--face N]   reject: pack-identity reject <id>")


def cmd_confirm(paths: Paths, rid: str, name: str, face_idx: int | None = None) -> None:
    recs = load_records(paths)
    r = recs.get(rid)
    if not r:
        raise SystemExit(f"no record {rid}")
    name = clean_name(name) or name
    try:
        apply_record(paths, r, name, face_idx, basis=Basis.OWNER_TAG)
    except ValueError as e:
        raise SystemExit(f"{rid}: {e}") from None
    r["ts_confirmed"] = time.time()
    r["confirmed_by"] = "cli"
    if r["status"] != "applied":
        r["status"] = "pending"
    append_record(paths, r)
    print(f"{rid}: {r['status']} {r['name']}: {r['applied']['notes']}")


def cmd_reject(paths: Paths, rid: str) -> None:
    recs = load_records(paths)
    r = recs.get(rid)
    if not r:
        raise SystemExit(f"no record {rid}")
    r["status"] = "rejected"
    r["ts_rejected"] = time.time()
    append_record(paths, r)
    print(f"{rid}: rejected")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state-dir", default=str(DEFAULT_STATE))
    ap.add_argument("--faces-db", default=None)
    ap.add_argument("--duration", type=float, default=None, help="worker: stop after N s (tests)")
    ap.add_argument("--face", type=int, default=None, help="confirm: pick face candidate N")
    ap.add_argument("cmd", nargs="?", default=None, choices=[None, "pending", "confirm", "reject", "enroll-visible"])
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    paths = Paths(a.state_dir, a.faces_db)
    if a.cmd is None:
        Identity(paths).run(a.duration)
    elif a.cmd == "pending":
        cmd_pending(paths)
    elif a.cmd == "confirm":
        if len(a.args) < 2:
            raise SystemExit("confirm <id> <name>")
        cmd_confirm(paths, a.args[0], " ".join(a.args[1:]), a.face)
    elif a.cmd == "reject":
        cmd_reject(paths, a.args[0])
    elif a.cmd == "enroll-visible":
        r = collar_intro(" ".join(a.args), a.state_dir, a.faces_db, source="cli")
        print(r["reply"])
        print(json.dumps(r["record"], default=str)[:400])


if __name__ == "__main__":
    main()
