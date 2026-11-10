"""Voices on Continuum: which voiceprint said each utterance, and the dashboard's voice tags.

v1 kept a voice list (`voices.sqlite3`: a running centroid per voice id) and matched each line against it. Here every
utterance's ReDimNet2 B3 embedding is its own row on `voice.embedding` in the segment store, and the vector worker
 follows and indexes them; a line is matched to the nearest stored embeddings through
`continuum_py.search.Searcher`.

- **Numbers.** A voiceprint keeps v1's form, an integer (`registry.voiceprint(n)`, "Speaker n" until it names
  someone), since the registry, voice auth (`voice:<n>`), and the dashboard all key voices that way. `Voiceprints`
  hands out the next number, kept in `state/voiceprints.json` and always above every number the registry knows.
- **Matching.** Unnamed clusters use the nearest embedding at cosine 0.45 (0.55 under one second). Named people
  use only individually human-confirmed utterances or explicit recording windows, never automatic observations, and
  the gate (`calibrate`, `match_person`) is set on this home's room audio, not clean read speech. A person's
  core is their confirmed lines of REF_MIN_S (2 s) or more that agree at REF_AGREE (0.5) with another of their
  confirmations; short or stray answers stay out of it instead of disabling the person. They're eligible with three
  distinct confirmations and a core from two. A line is named by its nearest core reference, not their average (a
  line sounds most like the answers from the same room and moment), when that reaches PERSON_FLOOR (0.5), raised to
  clear their nearest other person's reference and their own negatives by PERSON_MARGIN (0.15), and when it beats
  every other person and each of their negatives by PERSON_MARGIN and the nearest unnamed voice by UNNAMED_MARGIN
  (0.05). Measured on the Linux machine (counts only): same-person pairs of confirmed lines ranged 0.03 to 0.82 (lines
  under 2 s never above 0.31), different people's at most 0.26, and 0.16% of pairs of lines from different voiceprints
  reach 0.5. On a held-out replay of the confirmed lines (each left out, the gate built from the rest, the index's
  other lines competing) this named 2 of 8 rightly, none wrongly, and none of the 1,319 unconfirmed lines; the
  old LibriSpeech gate (every leave-one-out positive at 0.65) named nobody. Wrong names cost more than missed ones,
  so the floor and the unnamed margin sit where the replay named no unconfirmed line; more answers make more people
  nameable. Without evidence a line remains unnamed.
- **Negatives**. The owner taking a line from someone (a card's "Not <name>", or labeling it as somebody
  else) keeps its embedding as that person's negative (`deny`, `voice.not` in the knowledge store), released by the
  undo: their threshold rises to clear it and a line must beat it by PERSON_MARGIN, as faces' "not <person>".
- **Recent embeddings.** Only unnamed clusters use recent observations. The vector worker files an embedding a moment after it's committed, so a new speaker's
  second line could miss their first and open a second voiceprint (v1 merged such fragments later). The matcher
  keeps the last two minutes of its own embeddings and matches against them too.
- **Filing.** An embedding is filed only for a clip of a second or more, as a row in the writer's slot (listen's
  `listen`, an ingested item's `ingest-<id>`, the dashboard's enroll `enroll`) over its utterance's exact span, with
  its voiceprint's `subject_uuid`, the person that voiceprint names when it names one (the vector worker keys the row
  on the person, else the voiceprint), and the line it came from. Forgetting goes through the store
  (`forget_segments_where`): People: Delete forgets the person's rows and their voiceprints', and "forget a voice"
  its voiceprint's.
- **Names.** A line carries its voiceprint number and "Speaker n", never a person's name: who said it is read
  through the registry (`Registry.said`), so a rename, a merge, or a delete changes every line without rewriting
  one.
- **Owner answers teach**. Naming a voice confirms a few of its lines (`teaching_lines`, `teach`) and a
  line labeled as a person confirms that line, both as confirmed references (`confirm_lines`).
- **Voice tags.** "This line was Alice" from the dashboard (`state/voice_tags.jsonl`, as v1's microphone service
  read it) labels the line in the event log and teaches the voice: the line's embedding is filed again under
  Alice's voiceprint, or opens one for her. A batch syncs its segment rows before one owner-knowledge transaction.
  For a new person these prepared rows carry only the voiceprint and utterance; the committed registry claims
  and references supply the person. A failed knowledge transaction never grants those rows a named identity. A forget request (People: Delete) needs nothing more here than
  dropping the recent embeddings, since the dashboard purges the person's subjects itself.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import logging
import math
import os
import re
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from liberational import logs, segment_log, speech_log
from liberational.domain.identity import Basis
from liberational.people import registry

log = logging.getLogger(__name__)

EMBEDDINGS = segment_log.VOICE_EMBEDDINGS
MODEL, VERSION = "redimnet2-b3", "vox2-lm-v1"
SHAPE = f"embedding/{MODEL}/{VERSION}"
THRESHOLD = 0.45
SHORT_THRESHOLD = 0.55  # provisional: the calibration set has no clips under MIN_EMB_S
MIN_EMB_S = 1.0  # shorter clips never open a voiceprint and aren't filed
# Fewer words than this never open, join, or feed an unnamed voice: a one-word line ("Yeah.") is too little
# speech for a reliable voice print, and on the Linux machine after a reset 4 of the first 9 voiceprints were opened by
# one. Duration can't tell them apart: the span includes padding, and 5 of 6 one-word lines ran past MIN_EMB_S.
MIN_WORDS = 2
K = 20
SEARCH_TIMEOUT_S = 0.5
RECENT_S = 120.0
NS = 1_000_000_000
# The source of a voice tag's assertions, as v1's microphone service wrote them (people/transcript_view.VOICE_TAG).
VOICE_TAG = "ambient"


@dataclass(frozen=True)
class Match:
    """Which voiceprint said a line ("Speaker n" until the registry names it), and how close it was (cosine)."""

    speaker: str
    speaker_id: int
    sim: float


@dataclass(frozen=True)
class Reference:
    """A human-confirmed vector, keyed by its individual line or recording window."""

    person: str
    voiceprint: int
    key: str
    vector: np.ndarray
    source: str | None = None  # one enrollment remains one confirmation, regardless of its windows
    span: tuple[int, int] | None = None  # the confirmed line's audio (ns), whose original vector the index still holds


@dataclass(frozen=True)
class Negative:
    """The owner's "this line isn't them": a line's vector that naming `person` must clear by the margin."""

    person: str
    key: str
    vector: np.ndarray


# An unnamed index hit: (cosine, start_ns, end_ns) of the embedding it found.
Unnamed = tuple[float, int, int]
SAME_LINE_NS = 10_000_000  # a hit within 10 ms at both ends of a reference's span is that line's original vector

# The named-voice gate, set on this home's room audio ("Matching" in the module docs says how).
MIN_REFERENCES = 3  # distinct confirmations before anyone can be named
REF_MIN_S = 2.0  # a shorter confirmed line still counts as an answer but never joins the core
REF_AGREE = 0.5  # a core reference agrees this well with a reference from another confirmation
PERSON_FLOOR = 0.5  # the cosine a line needs to its nearest core reference
PERSON_MARGIN = 0.15  # over any other person (references and index hits) and over the person's own negatives
UNNAMED_MARGIN = 0.05  # over the nearest unnamed voice (an index hit or one of the matcher's recent lines)
TURN_ALIKE = 0.65  # two lines of one tagged turn are one speaker at this cosine (VoiceTags' turn split)


def _unit(vector: np.ndarray) -> np.ndarray | None:
    """`vector` at unit length, or None for one that can't be a ReDimNet2 embedding."""
    v = np.asarray(vector, np.float32)
    if v.shape != (segment_log.VOICE_DIM,) or not np.isfinite(v).all():
        return None
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else None


def _rows(vectors: Sequence[np.ndarray]) -> np.ndarray:
    matrix = np.asarray(vectors, np.float32).reshape(-1, segment_log.VOICE_DIM)
    matrix.flags.writeable = False
    return matrix


@dataclass(frozen=True)
class Voice:
    """One person's confirmed references as the gate reads them."""

    voiceprint: int
    vectors: np.ndarray  # every confirmed vector (unit rows): how much a line sounds like them when they compete
    core: np.ndarray  # the ones that can name them (`core`)
    confirmations: int  # distinct confirmations (an enrollment's windows are one)
    core_confirmations: int


def core(refs: Sequence[Reference]) -> list[int]:
    """Which of one person's unit references can name them: long enough to trust (REF_MIN_S; a reference with
    no span counts) and agreeing at REF_AGREE with a long-enough reference from another of their confirmations. A
    short or stray line is left out instead of disabling the person: on room audio one noisy answer would otherwise
    keep them from ever being named."""
    long = [i for i, r in enumerate(refs) if r.span is None or (r.span[1] - r.span[0]) / NS >= REF_MIN_S]
    return [
        i
        for i in long
        if any(
            (refs[j].source or refs[j].key) != (refs[i].source or refs[i].key)
            and float(np.dot(refs[i].vector, refs[j].vector)) >= REF_AGREE
            for j in long
        )
    ]


def voices_of(references: Sequence[Reference]) -> dict[str, Voice]:
    """Each person's `Voice`, from their usable references (one per key)."""
    grouped: dict[str, dict[str, Reference]] = {}
    for r in references:
        if (v := _unit(r.vector)) is not None:
            grouped.setdefault(r.person, {})[r.key] = Reference(r.person, r.voiceprint, r.key, v, r.source, r.span)
    out = {}
    for person, by_key in grouped.items():
        refs = list(by_key.values())
        inside = core(refs)
        out[person] = Voice(
            refs[0].voiceprint,
            _rows([r.vector for r in refs]),
            _rows([refs[i].vector for i in inside]),
            len({r.source or r.key for r in refs}),
            len({refs[i].source or refs[i].key for i in inside}),
        )
    return out


def negatives_of(negatives: Sequence[Negative]) -> dict[str, np.ndarray]:
    """Each person's negatives as unit rows."""
    grouped: dict[str, list[np.ndarray]] = {}
    for n in negatives:
        if (v := _unit(n.vector)) is not None:
            grouped.setdefault(n.person, []).append(v)
    return {person: _rows(vs) for person, vs in grouped.items()}


def calibrate(
    references: Sequence[Reference], negatives: Sequence[Negative] = ()
) -> dict[str, dict[str, float | int | bool]]:
    """Each person's gate from their confirmed references as a set and the owner's negatives.

    They're eligible with MIN_REFERENCES distinct confirmations and a core (`core`) from at least two of them. Their
    threshold is PERSON_FLOOR, raised so a core reference clears by PERSON_MARGIN every other person's reference and
    every one of their own negatives; past 1.0 (a core that sounds like someone else) nobody can be named it."""
    people = voices_of(references)
    nots = negatives_of(negatives)
    out: dict[str, dict[str, float | int | bool]] = {}
    for person, voice in people.items():
        against = [p.vectors for other, p in people.items() if other != person]
        if person in nots:
            against.append(nots[person])
        ceiling = max((float((voice.core @ m.T).max()) for m in against if len(voice.core)), default=-1.0)
        threshold = max(PERSON_FLOOR, ceiling + PERSON_MARGIN)
        out[person] = {
            "references": voice.confirmations,
            "vectors": len(voice.vectors),
            "core": len(voice.core),
            "negatives": len(nots.get(person, ())),
            "negative_ceiling": ceiling,
            "threshold": threshold,
            "eligible": voice.confirmations >= MIN_REFERENCES and voice.core_confirmations >= 2 and threshold <= 1.0,
        }
    return out


def _same_line(start: int, end: int, spans: Sequence[tuple[int, int]]) -> bool:
    return any(abs(start - a) <= SAME_LINE_NS and abs(end - b) <= SAME_LINE_NS for a, b in spans)


def word_count(text: str) -> int:
    """The words in a transcript line: runs of letters, digits, and apostrophes ("It's" is one)."""
    return len(re.findall(r"[\w']+", text))


def hit_span(hit: Mapping[str, Any], sim: float) -> Unnamed:
    """An unnamed search hit as `match_person` weighs it; a hit without a span can't be shown to be a taught line."""
    start, end = hit.get("start_ns"), hit.get("end_ns")
    if type(start) is int and type(end) is int:
        return sim, start, end
    return sim, -1, -1


def match_person(
    embedding: np.ndarray,
    references: Sequence[Reference],
    *,
    competitor: float = -1.0,
    unnamed: Sequence[Unnamed] = (),
    named_competitors: Mapping[str, float] | None = None,
    negatives: Sequence[Negative] = (),
    prepared: Calibration | None = None,
) -> Match | None:
    """One named assignment through the same confirmed-reference policy for listen, ingest, and phone lookups.

    A line is scored against each person's nearest core reference, not their average: on room audio a line sounds
    most like the answers from the same room and moment. The best eligible person is named when that score reaches
    their threshold, beats every other person (their references and `named_competitors`, their automatic index hits)
    and the nearest of their own negatives by PERSON_MARGIN, and beats the nearest unnamed voice by UNNAMED_MARGIN.

    `competitor` is an unnamed voice's best score with no provenance (the matcher's recent lines). `unnamed` are the
    index's unnamed hits with their spans: teaching a line copies its vector under the person, but its original stays
    under the unnamed voiceprint it was first heard as, so a hit over one of the person's own confirmed lines is that
    line again, not a competing voice. Every other unnamed hit still competes."""
    prepared = prepared or Calibration.build(references, negatives)
    policy = prepared.policy
    q = np.asarray(embedding, np.float32)
    named = {person: float((voice.vectors @ q).max()) for person, voice in prepared.voices.items()}
    eligible = [
        (float((voice.core @ q).max()), person)
        for person, voice in prepared.voices.items()
        if policy[person]["eligible"] and len(voice.core)
    ]
    if not eligible:
        return None
    score, person = max(eligible)
    own = [r.span for r in references if r.person == person and r.span is not None]
    rivals = [
        *[value for other, value in named.items() if other != person],
        *[value for other, value in (named_competitors or {}).items() if other != person],
    ]
    if person in prepared.negatives:
        rivals.append(float((prepared.negatives[person] @ q).max()))
    nearest_unnamed = max([competitor, *[sim for sim, start, end in unnamed if not _same_line(start, end, own)]])
    if (
        score >= float(policy[person]["threshold"])
        and score - max(rivals, default=-1.0) >= PERSON_MARGIN
        and score - nearest_unnamed >= UNNAMED_MARGIN
    ):
        voiceprint = prepared.voices[person].voiceprint
        return Match(f"Speaker {voiceprint}", voiceprint, score)
    return None


@dataclass(frozen=True)
class Calibration:
    """An immutable generation's policy, references, and negatives; never recalibrated on the audio event loop."""

    policy: Mapping[str, Mapping[str, float | int | bool]]
    voices: Mapping[str, Voice]
    negatives: Mapping[str, np.ndarray]

    @classmethod
    def build(cls, references: Sequence[Reference], negatives: Sequence[Negative] = ()) -> Calibration:
        return cls(calibrate(references, negatives), voices_of(references), negatives_of(negatives))


def confirmed_references(state: Path) -> list[Reference]:
    """Reference vectors are individual owner claims or explicit enrolled recordings, never a cluster's name.

    Explicit confirmation writes copy vectors to the existing owner-knowledge reference collection. Every read checks the current assertion again: a correction, forget, or replaced enrollment takes effect
    immediately. Old imported vectors without individual confirmation provenance remain stored but cannot teach.
    """
    from liberational import knowledge
    from liberational.domain.identity import SubjectKind, SubjectRef
    from liberational.listen import revisions

    if not registry.log_exists(state):
        return []
    reg = registry.local(state)
    store = knowledge.at(registry.knowledge_path(state))
    enrolled: dict[str, str] = {}
    try:
        enrolled = json.loads((state / "voice_enrollments.json").read_text())
    except OSError, ValueError:
        pass
    from liberational.people import transcript_view

    clones = transcript_view.clone_only_ids(state)
    existing = store.query(REFERENCES)
    reg.refresh()
    people = {p.id: p for p in reg.people()}
    available = reg.voices(reg.voiceprints(), refresh=False)
    by_person: dict[str, list[int]] = {}
    for voice, person in available.items():
        if person is not None and person in people and voice not in clones:
            by_person.setdefault(str(person), []).append(voice)
    rejected = revisions.rejected_subjects(state) | revisions.media_subjects(state)  # nor a TV's line
    subjects = {
        str(item.record.value.get("confirmation_subject"))
        for item in existing
        if item.record.value.get("confirmation_subject")
    }
    decisions = dict(
        zip(subjects, reg.said([(SubjectRef(SubjectKind.OBSERVATION, key), None) for key in subjects], refresh=False))
    )
    enrollments = {}
    out: dict[tuple[str, str], Reference] = {}
    for stored in existing:
        v = stored.record.value
        person = str(v.get("person") or "")
        n = v.get("voiceprint")
        if type(n) is not int or (v.get("model"), v.get("version")) != (MODEL, VERSION):
            continue
        key = str(v.get("confirmation_subject") or "")
        if key:
            if key in rejected:
                continue
            decision = decisions[key].own
            winner = decision.winner
            if decision.person is None or winner is None or winner.basis not in (Basis.OWNER_TAG, Basis.CORRECTION):
                continue
            person = str(decision.person)
        elif v.get("enrollment_id") and enrolled.get(str(n)) == v["enrollment_id"]:
            if n not in enrollments:
                enrollments[n] = reg.person_of(registry.voiceprint(n))
            decision = enrollments[n]
            if decision.person is None or decision.winner is None or decision.winner.basis != Basis.OWNER_TAG:
                continue
            person = str(decision.person)
            key = str(v["source_id"])
        else:
            continue
        vector = np.asarray(v.get("vector") or [], np.float32)
        norm = float(np.linalg.norm(vector))
        if vector.shape != (segment_log.VOICE_DIM,) or not np.isfinite(vector).all() or norm <= 0:
            continue
        candidates = by_person.get(person, [])
        if not candidates:
            continue
        vector = vector / norm
        vector.flags.writeable = False
        out[person, key] = Reference(
            person,
            candidates[0],
            key,
            vector,
            str(v["enrollment_id"]) if v.get("enrollment_id") else key,
            (stored.record.start_ns, stored.record.end_ns),
        )
    return list(out.values())


def confirm_lines(
    state: Path,
    items: Sequence[Mapping[str, Any]],
    *,
    base: Path | None = None,
    prepared: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> int:
    """Persist vectors only at an explicit line-confirmation write, with deterministic transactional IDs.

    No history scan or automatic match calls this. Immutable record IDs make simultaneous retries idempotent.
    The assertion's original person is stored; read-time resolution follows later merges and corrections.
    """
    from liberational import knowledge
    from liberational.listen import revisions

    items = revisions.filter_lines(state, items)
    reg = registry.at(state)
    decisions = reg.said([(registry.utterance_of(e), e.get("speaker_id")) for e in items])
    log = segment_log.Log(base if base is not None else segment_log.root(state))
    prints = Voiceprints(state / "voiceprints.json", lambda: max(reg.voiceprints(), default=0))
    records = []
    for e, decision in zip(items, decisions):
        winner = decision.own.winner
        if winner is None or winner.basis not in (Basis.OWNER_TAG, Basis.CORRECTION):
            continue
        rows = (
            prepared.get(str(e["id"]), ())
            if prepared is not None
            else log.range(
                EMBEDDINGS,
                float(e["ts_start"]),
                max(float(e.get("ts_end") or e["ts_start"]), float(e["ts_start"]) + 0.001),
            )
        )
        for row in rows:
            n = prints.number(row["voiceprint"])
            if row.get("utterance") != e.get("id") or n is None:
                continue
            person, subject_id = str(winner.person), registry.utterance_of(e).id
            value = {
                "voiceprint": n,
                "person": person,
                "vector": np.asarray(row["vector"], np.float32).tolist(),
                "dims": segment_log.VOICE_DIM,
                "model": MODEL,
                "version": VERSION,
                "source_id": row["id"],
                "source_kind": EMBEDDINGS,
                "confirmation_subject": subject_id,
            }
            rid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"confirmed-voice:{row['id']}:{winner.id}"))
            records.append(
                knowledge.Record(
                    rid,
                    REFERENCES,
                    f"speaker:{n}",
                    row["start_ns"],
                    row["end_ns"],
                    int(winner.asserted_at * NS),
                    value,
                    (person, subject(n)),
                )
            )
    if records:
        knowledge.at(registry.knowledge_path(state)).put(records)
    return len(records)


TEACH_LINES = 3  # the most lines one naming of a voice confirms


def teaching_lines(
    lines: Sequence[Mapping[str, Any]], offered: str | None, limit: int = TEACH_LINES
) -> list[Mapping[str, Any]]:
    """The lines a voice's naming confirms as the person's: the clip the owner was offered (its line key,
    `offered`) first, then the voice's most typical lines (best `speaker_sim`, then longest), at most `limit`. Only
    lines that can be a reference: a span of MIN_EMB_S or more (shorter ones file no embedding) and MIN_WORDS words."""

    def usable(e: Mapping[str, Any]) -> bool:
        start, end = e.get("ts_start"), e.get("ts_end")
        return (
            isinstance(start, int | float)
            and isinstance(end, int | float)
            and end - start >= MIN_EMB_S
            and word_count(str(e.get("text") or "")) >= MIN_WORDS
        )

    keep = [e for e in lines if usable(e)]
    first = [e for e in keep if offered is not None and registry.key_of(e) == offered][:1]
    rest = sorted(
        (e for e in keep if e not in first),
        key=lambda e: (-float(e.get("speaker_sim") or 0.0), -(float(e["ts_end"]) - float(e["ts_start"]))),
    )
    return [*first, *rest][: max(0, limit)]


def teach(state: Path, lines: Sequence[Mapping[str, Any]], name: str, source: str = "dashboard") -> int:
    """The owner named these lines `name`: each gets the owner's tag, and its embedding becomes one of the
    person's confirmed references (`confirm_lines`), which listen's gallery reads on its next revision check. An undo
    releases the lines' tags (registry `release`), and `confirmed_references` drops them at its next read. Returns the
    references written (0 for a line with no filed embedding; a repeat writes the same ids again)."""
    if not lines:
        return 0
    reg = registry.at(state)
    for e in lines:
        reg.claim(registry.utterance_of(e), name, Basis.OWNER_TAG, source)
    return confirm_lines(state, lines)


NOTS = "voice.not"  # the owner's "this line isn't them", in the knowledge store beside the references


def deny(
    state: Path,
    lines: Sequence[Mapping[str, Any]],
    person: str,
    *,
    active: bool = True,
    base: Path | None = None,
    now: float | None = None,
) -> int:
    """The owner said each of `lines` isn't the person with id `person`, or took it back (`active` False).

    Each line's filed embedding becomes one of the person's negatives (`confirmed_negatives`): their threshold rises to
    clear it, and a line must beat it by PERSON_MARGIN to be named them, as faces respect "not <person>". A
    record is keyed by line and person, the newest winning, so an undo writes an inactive one. Kept by person id, it
    follows a rename or a merge. Returns the records written (a line with no filed embedding writes none, unless
    it's a take-back)."""
    from liberational import knowledge

    if not lines:
        return 0
    log = segment_log.Log(base if base is not None else segment_log.root(state))
    at = int((time.time() if now is None else now) * NS)
    records = []
    for k, e in enumerate(lines):
        line = registry.utterance_of(e).id
        body: dict[str, Any] = {"person": person, "line": line, "active": active, "source": "owner"}
        start = end = at
        if active:
            row = next(
                (
                    r
                    for r in log.range(
                        EMBEDDINGS,
                        float(e["ts_start"]),
                        max(float(e.get("ts_end") or e["ts_start"]), float(e["ts_start"]) + 0.001),
                    )
                    if r.get("utterance") == e.get("id")
                ),
                None,
            )
            if row is None or _unit(np.asarray(row["vector"], np.float32)) is None:
                continue
            body |= {
                "vector": np.asarray(row["vector"], np.float32).tolist(),
                "dims": segment_log.VOICE_DIM,
                "model": MODEL,
                "version": VERSION,
                "source_id": row["id"],
            }
            start, end = row["start_ns"], row["end_ns"]
        records.append(
            knowledge.Record(str(uuid.uuid7()), NOTS, f"{line}:{person}", start, end, at + k, body, (person,))
        )
    if records:
        knowledge.at(registry.knowledge_path(state)).put(records)
    return len(records)


def confirmed_negatives(state: Path) -> list[Negative]:
    """The owner's standing voice negatives (`deny`), each under the person it's against now (merges followed). One
    taken back, against someone forgotten, or whose line the owner has since labeled as that very person is left out."""
    from liberational import knowledge
    from liberational.domain.identity import SubjectKind, SubjectRef
    from liberational.domain.kernel import PersonId

    if not registry.log_exists(state) or not registry.knowledge_path(state).exists():
        return []
    newest: dict[str, Mapping[str, Any]] = {}
    for stored in knowledge.at(registry.knowledge_path(state)).query(NOTS):  # oldest first: the last one wins
        newest[str(stored.record.key)] = stored.record.value
    standing = [
        v
        for v in newest.values()
        if v.get("active") is True and (v.get("model"), v.get("version")) == (MODEL, VERSION) and v.get("line")
    ]
    if not standing:
        return []
    reg = registry.local(state)
    reg.refresh()
    said = reg.said([(SubjectRef(SubjectKind.OBSERVATION, str(v["line"])), None) for v in standing], refresh=False)
    out = []
    for v, decision in zip(standing, said):
        person = reg.resolve(PersonId(str(v.get("person") or "")))
        vector = _unit(np.asarray(v.get("vector") or [], np.float32))
        if person is None or vector is None or reg.resolve(decision.own.person) == person:
            continue
        vector.flags.writeable = False
        out.append(Negative(str(person), str(v["line"]), vector))
    return out


def _identity_version(state: Path) -> tuple[str, int, int]:
    from liberational.knowledge_eventlog import SQLiteEventLog

    log = registry.local(state).log
    assert isinstance(log, SQLiteEventLog)
    return log.identity_version()


_REFERENCE_WORKER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voice-references")
_TAG_WORKER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voice-tags")


class ReferenceGallery:
    """Publish one calibrated generation off shared registry locks. Dirty generations cannot name anybody."""

    def __init__(self, state: Path) -> None:
        self.state = state
        self.lock = threading.Lock()
        self.build_lock = threading.Lock()
        self.mark: tuple[object, ...] | None = None
        self.cached: tuple[Reference, ...] = ()
        self.calibration: Calibration | None = None
        self.pending: Future[None] | None = None

    def _revision(self) -> tuple[object, ...]:
        from liberational import knowledge
        from liberational.listen import revisions

        def stamp(path: Path) -> tuple[int, int] | None:
            try:
                stat = path.stat()
                return stat.st_mtime_ns, stat.st_size
            except FileNotFoundError:
                return None

        if not registry.log_exists(self.state):
            return ()
        return (
            _identity_version(self.state),
            knowledge.at(registry.knowledge_path(self.state)).revision([REFERENCES, NOTS, revisions.CORRECTIONS]),
            stamp(self.state / "voice_enrollments.json"),
            stamp(self.state / "clones"),
        )

    def refresh(self) -> None:
        """A batch publishes once after commit; concurrent changes discard this snapshot rather than use stale refs."""
        with self.build_lock:
            mark = self._revision()
            with self.lock:
                if mark == self.mark and self.calibration is not None:
                    return
            refs = tuple(confirmed_references(self.state)) if mark else ()
            calibration = Calibration.build(refs, confirmed_negatives(self.state) if mark else ())
            if self._revision() == mark:
                with self.lock:
                    self.cached, self.calibration, self.mark = refs, calibration, mark

    def __call__(self) -> Sequence[Reference]:
        # Explicit synchronous readers/tests are not the audio matching path.
        if self._revision() != self.mark:
            self.refresh()
        mark = self._revision()
        with self.lock:
            return list(self.cached) if mark == self.mark else []

    def match(
        self,
        embedding: np.ndarray,
        *,
        competitor: float = -1.0,
        unnamed: Sequence[Unnamed] = (),
        named_competitors: Mapping[str, float] | None = None,
    ) -> Match | None:
        """Called off-thread: cheap revision probe, then cached arithmetic or one coalesced background rebuild."""
        mark = self._revision()
        with self.lock:
            if mark != self.mark or self.calibration is None:
                if self.pending is None or self.pending.done():
                    self.pending = _REFERENCE_WORKER.submit(self.refresh)
                return None
            refs, calibration = self.cached, self.calibration
        matched = match_person(
            embedding,
            refs,
            competitor=competitor,
            unnamed=unnamed,
            named_competitors=named_competitors,
            prepared=calibration,
        )
        return matched if self._revision() == mark else None


class Search(Protocol):
    """The part of `continuum_py.search.Searcher` the matcher uses."""

    async def nearest(
        self, model: str, version: str, vector: Sequence[float], k: int = 20, *, timeout: float = 0.5
    ) -> Sequence[Mapping[str, Any]]: ...


class People(Protocol):
    """Who a voiceprint is, and a person's voiceprint: the registry's answers, as the matcher needs them."""

    def person_of(self, n: int) -> str | None: ...
    def voice_of(self, person: str) -> int | None: ...


@dataclass(frozen=True)
class Membership:
    persons: Mapping[int, str]
    aliases: Mapping[str, str] | None = None

    def person_of(self, n: int) -> str | None:
        return self.persons.get(n)

    def voice_of(self, person: str) -> int | None:
        current = self.aliases.get(person, person) if self.aliases is not None else person
        return min((n for n, p in self.persons.items() if p == current), default=None)


class RegistryPeople:
    """Registry-backed membership, projected once per identity generation off the audio loop."""

    def __init__(self, state: Path) -> None:
        self.state = state
        self.lock = threading.Lock()
        self.mark: object = None
        self.cached = Membership({})

    def snapshot(self) -> Membership:
        reg = registry.local(self.state)
        mark = _identity_version(self.state)
        with self.lock:
            if mark != self.mark:
                persons = reg.voices(reg.voiceprints())
                aliases = {
                    str(alias): str(person.id) for person in reg.people() for alias in reg.group(person.display_name)
                }
                self.cached = Membership({n: str(p) for n, p in persons.items() if p is not None}, aliases)
                self.mark = mark
            return self.cached

    def person_of(self, n: int) -> str | None:
        return self.snapshot().person_of(n)

    def voice_of(self, person: str) -> int | None:
        return self.snapshot().voice_of(person)


def subject(n: int) -> str:
    """Voiceprint `n`'s Continuum subject id."""
    return registry.subject_uuid(registry.voiceprint(n))


class Voiceprints:
    """Voiceprint numbers: the next one to hand out (kept in `path`), and which subject id is which number. Listen
    and the dashboard's voice edits (people/voice_edits.py, enrolling a voice) both hand them out, so `new` takes the
    file's lock and reads it again each time: two processes never give out the same number."""

    def __init__(self, path: Path, highest_known: Callable[[], int]) -> None:
        self.path = path
        # Never a number the registry already has, even if the file was lost: a reused number would join two voices.
        self.next = max(self._saved(), highest_known() + 1, 1)
        self.by_subject = {subject(n): n for n in range(1, self.next)}

    def _saved(self) -> int:
        try:
            return int(json.loads(self.path.read_text())["next"])
        except OSError, ValueError, KeyError, TypeError:
            return 1

    def _grow(self, to: int) -> None:
        for n in range(self.next, to):
            self.by_subject[subject(n)] = n
        self.next = max(self.next, to)

    def new(self) -> int:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path.with_name(self.path.name + ".lock"), "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self._grow(self._saved())  # what another process handed out since
            n = self.next
            self._grow(n + 1)
            temporary = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
            temporary.write_text(json.dumps({"next": self.next}))
            os.replace(temporary, self.path)
        return n

    def number(self, entity: str) -> int | None:
        if entity not in self.by_subject:
            self._grow(self._saved())  # another process may have handed it out
        return self.by_subject.get(entity)


class VoiceMatcher:
    """Matches each line's embedding to a voiceprint (`match`) and files it (`learn`); see the module's docs."""

    def __init__(
        self,
        prints: Voiceprints,
        people: People,
        search: Search,
        *,
        clones: Callable[[], frozenset[int]] = frozenset,
        segments: Path | None = None,
        slot: str = "listen",
        references: Callable[[], Sequence[Reference]] | None = None,
    ) -> None:
        self.prints, self.people, self.search = prints, people, search
        self.references = references or (lambda: ())
        self.segments = segments if segments is not None else segment_log.root()
        self.slot = slot  # the store slot its embeddings go in, unless a call names another
        self._writers: dict[str, segment_log.Writers] = {}
        self.clones = clones  # the voice clones' voiceprints (clones/<n>.json): never a room line's voice
        self._recent: deque[tuple[float, np.ndarray, int]] = deque()  # (monotonic s, embedding, voiceprint)
        self._recent_lock = threading.Lock()
        self.lock = asyncio.Lock()
        self.unavailable = 0  # lines committed unmatched because the search couldn't answer

    async def match(
        self, embedding: np.ndarray, start_ns: int, end_ns: int, *, words: int | None = None
    ) -> Match | None:
        """The line's voice: a confirmed person, else an unnamed voiceprint (one it matches, or a new one). `words` is
        the line's word count (`word_count`): under MIN_WORDS it can only name a confirmed person, never join or open
        an unnamed voice. None (no transcript) counts as enough words."""
        long_enough = (end_ns - start_ns) / NS >= MIN_EMB_S
        few_words = words is not None and words < MIN_WORDS
        threshold = THRESHOLD if long_enough else SHORT_THRESHOLD
        # One at a time: two lines of a new speaker must not both open a voiceprint.
        async with self.lock:
            people = (
                await asyncio.to_thread(self.people.snapshot)
                if isinstance(self.people, RegistryPeople)
                else self.people
            )
            clones = await asyncio.to_thread(self.clones)
            best_sim, best = self._recent_best(embedding, people)
            recent_sim = best_sim
            named_scores: dict[str, float] = {}
            unnamed: list[Unnamed] = []
            try:
                hits: Sequence[Mapping[str, Any]] | None = await self.search.nearest(
                    MODEL, VERSION, [float(x) for x in embedding], k=K, timeout=SEARCH_TIMEOUT_S
                )
            except TimeoutError as e:
                hits = None
                self.unavailable += 1
                if self.unavailable == 1 or self.unavailable % 100 == 0:
                    log.warning(
                        "voice search unavailable (%s); %d lines unmatched so far", logs.reason(e), self.unavailable
                    )
            for h in hits or []:
                n = self._number(h.get("entity_id"), people, clones)
                # A named cluster's automatic observations can never become its person's reference set.
                if n is not None:
                    sim = 1.0 - float(h["score"])
                    if not math.isfinite(sim):
                        continue
                    person = people.person_of(n)
                    if person is not None:
                        named_scores[person] = max(named_scores.get(person, -1.0), sim)
                        continue
                    unnamed.append(hit_span(h, sim))
                    if sim > best_sim:
                        best_sim, best = sim, n
            try:
                if isinstance(self.references, ReferenceGallery):
                    matched = await asyncio.to_thread(
                        self.references.match,
                        embedding,
                        competitor=recent_sim,
                        unnamed=unnamed,
                        named_competitors=named_scores,
                    )
                else:
                    references = await asyncio.to_thread(self.references)
                    matched = await asyncio.to_thread(
                        match_person,
                        embedding,
                        references,
                        competitor=recent_sim,
                        unnamed=unnamed,
                        named_competitors=named_scores,
                    )
                if matched is not None:
                    return matched
            except Exception:
                # Failure to read confirmed evidence never licenses a match from the auto-trained index.
                log.exception("confirmed voice references unavailable; leaving named voices unmatched")
            if few_words:
                return None  # unlabeled; learn files it unassigned, so it feeds no unnamed voice
            if best is not None and best_sim >= threshold:
                return Match(f"Speaker {best}", best, best_sim)
            if max(named_scores.values(), default=-1.0) >= threshold:
                return None  # a likely named cluster with a closed gate stays unknown, without new-cluster churn
            if hits is None or not long_enough:
                return None
            n = self.prints.new()
            log.info("a new voice: Speaker %d (nearest at %.2f)", n, best_sim)
            # Remembered now, under the lock, so the next line of this voice finds it before the search does.
            self._remember(embedding, n)
            return Match(f"Speaker {n}", n, max(best_sim, 0.0))

    def learn(
        self,
        embedding: np.ndarray,
        match: Match | None,
        utterance_id: str,
        start_ns: int,
        end_ns: int,
        *,
        payload_id: str | None = None,
        slot: str | None = None,
    ) -> None:
        """Files the line's embedding under its voiceprint, once the line itself is written.

        `payload_id` fixes the embedding's id (an ingest retry). The live path leaves it empty. `slot` is the store
        slot it goes in (an ingested item's own, so forgetting the item forgets it), by default the matcher's.
        """
        if (end_ns - start_ns) / NS < MIN_EMB_S:
            return
        if match is None:
            self.file(embedding, 0, utterance_id, start_ns, end_ns, own_subject=True, payload_id=payload_id, slot=slot)
            return
        if self.people.person_of(match.speaker_id) is None:
            self._remember(embedding, match.speaker_id)
        self.file(
            embedding,
            match.speaker_id,
            utterance_id,
            start_ns,
            end_ns,
            payload_id=payload_id,
            slot=slot,
        )

    def file(
        self,
        embedding: np.ndarray,
        n: int,
        utterance: str | None,
        start_ns: int,
        end_ns: int,
        *,
        own_subject: bool = False,
        payload_id: str | None = None,
        slot: str | None = None,
    ) -> str:
        """Writes `embedding` on voice.embedding under voiceprint `n` (and the person it names, when it names one), in
        `slot` (by default the matcher's), synced before it returns. `utterance` is the line it came from (None for
        an enrolled recording, which is no line). Returns its id. `own_subject` files it under the voiceprint alone,
        never its person: a voice clone's, which must never be found as the person's voice (the vector index keys on
        the person)."""
        person = None if own_subject else self.people.person_of(n)
        row = {
            "id": payload_id or segment_log.new_id(),
            "start_ns": start_ns,
            "end_ns": max(end_ns, start_ns),
            "voiceprint": subject(n),
            "person": person,
            "utterance": utterance,
            "vector": np.asarray(embedding, np.float32),
        }
        where = segment_log.slot(slot or self.slot)
        if where not in self._writers:
            self._writers[where] = segment_log.Writers(self.segments)
        self._writers[where].append(EMBEDDINGS, where, [row])
        self._writers[where].sync()
        return row["id"]

    def close(self, slot: str | None = None) -> None:
        """Closes the writer of `slot` (every slot when None), so another process may take it."""
        for where in [segment_log.slot(slot)] if slot is not None else list(self._writers):
            if (w := self._writers.pop(where, None)) is not None:
                w.close()

    @property
    def recent(self) -> tuple[tuple[float, np.ndarray, int], ...]:
        """An ordered snapshot: tag/learn threads can mutate the deque while scoring runs without this lock."""
        with self._recent_lock:
            return tuple(self._recent)

    def forget(self) -> None:
        """A person was deleted: their recent embeddings mustn't match anyone again."""
        with self._recent_lock:
            self._recent.clear()

    def _number(self, entity: object, people: People | None = None, clones: frozenset[int] | None = None) -> int | None:
        """The voiceprint a hit is filed under, or None for none or a voice clone's: a clone's recording is never a
        person in the room (v1 never matched room speech to one)."""
        if not isinstance(entity, str):
            return None
        n = self.prints.number(entity)
        n = n if n is not None else (people or self.people).voice_of(entity)
        return None if n is not None and n in (self.clones() if clones is None else clones) else n

    def _remember(self, embedding: np.ndarray, n: int) -> None:
        now = time.monotonic()
        vector = np.asarray(embedding, np.float32)
        with self._recent_lock:
            self._recent.append((now, vector, n))
            while self._recent and now - self._recent[0][0] > RECENT_S:
                self._recent.popleft()

    def _recent_best(self, embedding: np.ndarray, people: People | None = None) -> tuple[float, int | None]:
        now = time.monotonic()
        best_sim, best = -1.0, None
        for t, e, n in self.recent:
            if (
                (people or self.people).person_of(n) is None
                and now - t <= RECENT_S
                and (sim := float(np.dot(embedding, e))) > best_sim
            ):
                best_sim, best = sim, n
        return best_sim, best


@dataclass
class TagBatch:
    """One owner request, including old queue entries already waiting during an upgrade."""

    request: dict[str, Any]
    lines: list[dict[str, Any]]
    turn: list[dict[str, Any]]
    after: int


class VoiceTags:
    """Drain owner requests on one serialized worker, never on the audio event loop.

    Segment copies sync before one SQLite transaction for the batch's claims and references. An interrupted owner
    transaction can leave ordinary prepared segment observations, but never confirmed person evidence. The byte
    cursor advances only after a batch completes; deterministic segment/reference IDs make a replay harmless.
    """

    FILE, DONE = "voice_tags.jsonl", "voice_tags.offset"

    def __init__(self, state: Path, matcher: VoiceMatcher, source: str = VOICE_TAG) -> None:
        self.state, self.matcher, self.source = state, matcher, source
        self._forget_scanned = 0
        self._forgets: dict[str, int] = {}
        self._retry_at = 0.0
        self._failures = 0

    async def run(self, stop: asyncio.Event, every_s: float = 1.0) -> None:
        while not stop.is_set():
            await self.poll()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), every_s)

    async def poll(self) -> None:
        await asyncio.get_running_loop().run_in_executor(_TAG_WORKER, self._drain)

    def _backoff(self) -> None:
        self._failures += 1
        self._retry_at = time.monotonic() + min(30.0, 2.0 ** min(self._failures - 1, 5))
        log.exception("voice tag batch failed; request remains queued for retry")

    def _drain(self) -> None:
        try:
            self._poll()
        except Exception:
            self._backoff()

    @staticmethod
    def _spec(value: object) -> dict[str, Any]:
        if isinstance(value, str):
            slug, at = registry.split_key(value)
            return {"ts_start": at, **({"line": value} if slug else {})}
        if isinstance(value, Mapping):
            return {"ts_start": float(value["ts_start"]), **({"line": value["line"]} if value.get("line") else {})}
        raise ValueError("a queued line must have its exact key or timestamp")

    def _scan_forgets(self) -> None:
        """Parse each new byte once, including deletions appended while a batch was preparing vectors."""
        path = self.state / self.FILE
        size = path.stat().st_size
        if size < self._forget_scanned:
            self._forget_scanned, self._forgets = 0, {}
        with path.open("rb") as stream:
            stream.seek(self._forget_scanned)
            data = stream.read()
        cut = data.rfind(b"\n")
        for raw in data[: cut + 1].splitlines(keepends=True) if cut >= 0 else []:
            self._forget_scanned += len(raw)
            try:
                request = json.loads(raw)
                if not isinstance(request, dict):
                    raise ValueError("voice tag request must be an object")
                if "forget" in request:
                    self._forgets[registry.norm(request.get("name"))] = self._forget_scanned
            except ValueError, TypeError:
                continue

    def _poll(self) -> None:
        if time.monotonic() < self._retry_at:
            return
        path, done = self.state / self.FILE, self.state / self.DONE
        try:
            size = path.stat().st_size
        except OSError:
            return
        try:
            off = int(done.read_text().strip() or 0)
        except OSError, ValueError:
            off = 0
        if off > size:
            off = 0
        if off == size:
            return
        with path.open("rb") as stream:
            stream.seek(off)
            data = stream.read(size - off)
        cut = data.rfind(b"\n")
        work: list[TagBatch] = []
        for raw in data[: cut + 1].splitlines(keepends=True) if cut >= 0 else []:
            off += len(raw)
            try:
                request = json.loads(raw)
                if not isinstance(request, dict):
                    raise ValueError("voice tag request must be an object")
                if "forget" in request:
                    work.append(TagBatch(request, [], [], off))
                    continue
                if "lines" in request:
                    specs = [self._spec(item) for item in request["lines"]]
                    work.append(TagBatch(request, specs, specs if len(specs) > 1 else [], off))
                else:
                    spec = self._spec(request)
                    group = request.get("group")
                    if group and work and work[-1].request.get("group") == group:
                        work[-1].lines.append(spec)
                        work[-1].after = off
                    else:
                        turn = [self._spec(item) for item in request.get("turn_lines", [])]
                        work.append(TagBatch(request, [spec], turn, off))
            except ValueError, TypeError, KeyError:
                log.warning("invalid voice tag request skipped")
                work.append(TagBatch({"invalid": True}, [], [], off))
        self._scan_forgets()
        for batch in work:
            try:
                if "forget" in batch.request:
                    self.matcher.forget()
                elif "invalid" not in batch.request:
                    self._batch(batch)
                temporary = done.with_name(done.name + ".tmp")
                temporary.write_text(str(batch.after))
                os.replace(temporary, done)
            except Exception:
                # Owner requests are durable: leave this batch and everything after it queued for a retry.
                self._backoff()
                return
            self._failures = 0
            self._retry_at = 0.0

    @staticmethod
    def _plan(
        turn: list[dict[str, Any]],
        lines: Mapping[str, dict[str, Any]],
        vectors: Mapping[str, Mapping[str, Any]],
        known: Sequence[np.ndarray] = (),
    ) -> TurnPlan:
        """What a named turn does with its lines. Its lines' voices are grouped by likeness. The group that sounds
        like the person's voice (`known`, their confirmed vectors), else a majority group, takes the name and teaches
        the voice; each other group splits off as a voice of its own, to be named later. With no two lines alike there is nothing to split
        from: the owner's word names every line, and nothing is taught. Groups with no majority and none
        that sounds like the person are left unknown."""
        if len(turn) < 2:
            return TurnPlan(set(lines), set(lines), [])
        keys = set()
        for spec in turn:
            key = spec.get("line")
            if key is None:
                key = next(
                    (key for key, line in lines.items() if abs(float(line["ts_start"]) - spec["ts_start"]) < 0.0015),
                    None,
                )
            if key is not None:
                keys.add(key)
        found = [
            (key, np.asarray(vectors[str(line["id"])]["vector"], np.float32))
            for key, line in lines.items()
            if key in keys and str(line["id"]) in vectors
        ]
        short = {key for key, line in lines.items() if key in keys and str(line["id"]) not in vectors}
        if not found:
            return TurnPlan(keys, set(), [])
        matrix = np.asarray([vector / max(float(np.linalg.norm(vector)), 1e-9) for _, vector in found])
        groups = _alike(matrix @ matrix.T >= TURN_ALIKE)
        if len(groups[0]) < 2:
            return TurnPlan(keys, set(), [])  # no two lines alike: nothing stands out to split off
        kept: list[int] | None = None
        if len(known):
            refs = np.asarray([v / max(float(np.linalg.norm(v)), 1e-9) for v in known], np.float32)
            sounds = [int(((matrix[g] @ refs.T).max(axis=1) >= TURN_ALIKE).sum()) for g in groups]
            if max(sounds) > 0:
                kept = groups[int(np.argmax(sounds))]
        if kept is None and len(groups[0]) > len(found) / 2:
            kept = groups[0]
        if kept is None:
            return TurnPlan(set(), set(), [])  # a split with no side theirs: left unknown
        named = {found[i][0] for i in kept} | short
        return TurnPlan(named, named, [{found[i][0] for i in g} for g in groups if g is not kept])

    def _batch(self, batch: TagBatch) -> None:
        from liberational.listen import revisions

        req = batch.request
        asked = str(req["name"]).strip()
        if not re.fullmatch(r"[^\W\d_][\w .'-]{0,39}", asked) or not registry.is_name(asked):
            log.warning("voice tag batch has an invalid person name; skipped")
            return
        requested = req.get("requested_at")
        at = (
            float(requested)
            if isinstance(requested, int | float) and not isinstance(requested, bool) and math.isfinite(requested)
            else None
        )
        reg = registry.local(self.state)
        pid = reg.known_as(asked, at, strict=True)
        person = reg.person(pid) if pid is not None else None
        name = person.display_name if person is not None else asked
        if any(self._forgets.get(registry.norm(n), 0) > batch.after for n in (asked, name)):
            log.info("voice tag batch predates a person deletion; skipped")
            return
        specs = batch.turn + batch.lines
        if not specs:
            return
        start, end = min(float(s["ts_start"]) for s in specs), max(float(s["ts_start"]) for s in specs)
        heard = speech_log.lines(start - 0.002, end + 0.002, base=self.matcher.segments)
        by_key = {registry.key_of(line): line for line in heard}
        selected = []
        for spec in batch.lines:
            line = (
                by_key.get(str(spec.get("line")))
                if spec.get("line")
                else next((line for line in heard if abs(float(line["ts_start"]) - spec["ts_start"]) < 0.0015), None)
            )
            if line is not None and line not in selected:
                selected.append(line)
        if not selected:
            log.info("voice tag batch has no retained transcript lines; skipped")
            return
        rows = segment_log.Log(self.matcher.segments).range(
            EMBEDDINGS, start - 0.002, max(float(line["ts_end"]) for line in heard) + 0.002
        )
        vectors = {str(row["utterance"]): row for row in rows if row.get("utterance")}
        known = [r.vector for r in confirmed_references(self.state) if r.person == str(pid)] if pid is not None else []
        plan = self._plan(batch.turn, by_key, vectors, known)
        allowed, taught = plan.named, plan.taught
        group_of = {key: g for g, keys in enumerate(plan.splits) for key in keys}
        accepted = [line for line in selected if registry.key_of(line) in allowed]
        learned = [line for line in accepted if registry.key_of(line) in taught]
        split = [line for line in selected if registry.key_of(line) in group_of]
        rejected = [line for line in selected if registry.key_of(line) not in allowed and line not in split]
        theirs = reg.voiceprints(pid) if pid is not None else []
        batch_key = str(req.get("group") or json.dumps(req, sort_keys=True))

        def copied_id(line: Mapping[str, Any], voice: int) -> str:
            return str(uuid.uuid5(uuid.NAMESPACE_URL, f"voice-tag:{batch_key}:{line['id']}:{voice}"))

        # A retry/restart reuses the number and rows prepared before its failed knowledge commit.
        old_ids = {str(row["id"]) for row in rows}
        accepted_by_id = {str(line["id"]): line for line in learned}
        prepared_number = (
            next(
                (
                    number
                    for row in rows
                    if (line := accepted_by_id.get(str(row.get("utterance")))) is not None
                    and (number := self.matcher.prints.number(row["voiceprint"])) is not None
                    and copied_id(line, number) == row["id"]
                ),
                None,
            )
            if not theirs
            else None
        )
        n = (
            theirs[0]
            if theirs
            else prepared_number
            if prepared_number is not None
            else self.matcher.prints.new()
            if any(str(e["id"]) in vectors for e in learned)
            else None
        )
        copies = [
            {
                "id": copied_id(line, n),
                "start_ns": speech_log.to_ns(line["ts_start"]),
                "end_ns": speech_log.to_ns(line["ts_end"]),
                "voiceprint": subject(n),
                "person": str(pid) if pid is not None else None,
                "utterance": line["id"],
                "vector": vectors[str(line["id"])]["vector"],
            }
            for line in learned
            if n is not None and str(line["id"]) in vectors
        ]
        # Each group split off is a voice of its own, taught from its lines with no person; a retry reuses its number.
        voice_of_group: dict[int, int] = {}
        for g in range(len(plan.splits)):
            members = {str(line["id"]): line for line in split if group_of[registry.key_of(line)] == g}
            if not any(i in vectors for i in members):
                continue
            voice_of_group[g] = (
                next(
                    (
                        number
                        for row in rows
                        if (line := members.get(str(row.get("utterance")))) is not None
                        and (number := self.matcher.prints.number(row["voiceprint"])) is not None
                        and copied_id(line, number) == row["id"]
                    ),
                    None,
                )
                or self.matcher.prints.new()
            )
        copies += [
            {
                "id": copied_id(line, m),
                "start_ns": speech_log.to_ns(line["ts_start"]),
                "end_ns": speech_log.to_ns(line["ts_end"]),
                "voiceprint": subject(m),
                "person": None,
                "utterance": line["id"],
                "vector": vectors[str(line["id"])]["vector"],
            }
            for line in split
            if (m := voice_of_group.get(group_of[registry.key_of(line)])) is not None and str(line["id"]) in vectors
        ]
        new_copies = [row for row in copies if row["id"] not in old_ids]
        # This native file commit precedes, and cannot be rolled back by, the owner-knowledge transaction.
        if new_copies:
            writer = segment_log.Writers(self.matcher.segments)
            try:
                writer.append(EMBEDDINGS, "voice-tags", new_copies)
                writer.sync()
            finally:
                writer.close()
        self._scan_forgets()
        if any(self._forgets.get(registry.norm(who), 0) > batch.after for who in (asked, name)):
            log.info("prepared voice tag batch predates a person deletion; skipped")
            return
        preserved = unknown = moved = 0
        with registry.transaction(self.state) as reg:
            # Queue markers can lag Delete. Check authoritative identity under the same transaction as the claim,
            # and resolve a concurrent rename/merge without recreating its old display name.
            current_pid = reg.known_as(asked, at, strict=True)
            if reg.forgotten_since(asked, at) or (pid is not None and current_pid is None):
                log.info("voice tag batch's registration was forgotten; skipped")
                return
            current = reg.person(current_pid) if current_pid is not None else None
            name = current.display_name if current is not None else asked
            accepted = [line for line in revisions.filter_lines(self.state, accepted) if not line.get("media")]
            learned = [line for line in accepted if registry.key_of(line) in taught]
            claims = [registry.Claim(registry.utterance_of(line), name) for line in accepted]
            if n is not None and learned and not theirs:
                claims.insert(0, registry.Claim(registry.voiceprint(n), name))
            reg.claim_many(claims, Basis.OWNER_TAG, self.source)
            for line, decision in zip(
                rejected, reg.said([(registry.utterance_of(e), e.get("speaker_id")) for e in rejected])
            ):
                if decision.own.winner is not None and decision.own.winner.basis in (Basis.OWNER_TAG, Basis.CORRECTION):
                    preserved += 1
                else:
                    sid = line.get("speaker_id")
                    reg.detach(
                        registry.utterance_of(line),
                        "unknown",
                        "turn-majority",
                        via=registry.voiceprint(sid) if type(sid) is int else None,
                    )
                    unknown += 1
            split = [line for line in revisions.filter_lines(self.state, split) if not line.get("media")]
            for line, decision in zip(
                split, reg.said([(registry.utterance_of(e), e.get("speaker_id")) for e in split])
            ):
                m = voice_of_group.get(group_of[registry.key_of(line)])
                if decision.own.winner is not None and decision.own.winner.basis in (Basis.OWNER_TAG, Basis.CORRECTION):
                    preserved += 1
                    continue
                sid = line.get("speaker_id")
                reg.detach(
                    registry.utterance_of(line),
                    registry.split_label(m) if m is not None else "unknown",
                    "turn-split",
                    via=registry.voiceprint(sid) if type(sid) is int else None,
                )
                if m is None:
                    unknown += 1
                else:
                    moved += 1
            confirmed = confirm_lines(
                self.state,
                learned,
                base=self.matcher.segments,
                prepared={str(row["utterance"]): [row] for row in copies},
            )
        if isinstance(self.matcher.references, ReferenceGallery):
            self.matcher.references.refresh()
        log.info(
            "voice tag batch: %d labelled, %d reference vectors, %d left unknown, %d human labels preserved, "
            "%d named without teaching the voice (its lines agreed on nothing), %d split off to %d new voices",
            len(accepted),
            confirmed,
            unknown,
            preserved,
            len(accepted) - len(learned),
            moved,
            len(voice_of_group),
        )


@dataclass
class TurnPlan:
    """What a named turn does with its lines, by key: those it names, those of them it teaches the voice from, and
    the groups it splits off as voices of their own. Any other line is left unknown."""

    named: set[str]
    taught: set[str]
    splits: list[set[str]]


def _alike(close: np.ndarray) -> list[list[int]]:
    """Groups of lines whose voices are alike, linked through each other (a square boolean matrix), largest first,
    then earliest."""
    seen: set[int] = set()
    groups = []
    for start in range(len(close)):
        if start in seen:
            continue
        group, todo = [], [start]
        seen.add(start)
        while todo:
            i = todo.pop()
            group.append(i)
            for j in np.flatnonzero(close[i]):
                if int(j) not in seen:
                    seen.add(int(j))
                    todo.append(int(j))
        groups.append(sorted(group))
    return sorted(groups, key=lambda g: (-len(g), g[0]))


TAUGHT = "taught"  # the slot the taught voice vectors are written to after a reset
REFERENCES = "voice.reference"  # the knowledge store's taught voice vectors (pack-knowledge import)


def seed_taught(state: Path, people: People, *, base: Path | None = None) -> int:
    """Writes the knowledge store's taught voice vectors (each named voiceprint's exact set, imported across the
    reset) as `voice.embedding` rows in slot `taught`, when the segment store has never had that slot: after a reset,
    the vector worker indexes them like any other row, so a voice named before the reset matches after it. Each row
    is filed under its voiceprint and the person that voiceprint names now. A vector of another model or size is
    left out. Returns how many were written; 0 when the slot was written before (a forget since then stays)."""
    from continuum_py import segment_tail

    root = base if base is not None else segment_log.root(state)
    if segment_tail(root, EMBEDDINGS, [TAUGHT]).get(TAUGHT, 0) > 0 or not registry.log_exists(state):
        return 0
    from liberational import knowledge

    taught = knowledge.at(registry.knowledge_path(state)).query(REFERENCES)
    matcher = VoiceMatcher(
        Voiceprints(state / "voiceprints.json", lambda: 0), people, _NoSearch(), segments=root, slot=TAUGHT
    )
    written = skipped = 0
    try:
        for stored in taught:
            v, r = stored.record.value, stored.record
            vector = np.asarray(v.get("vector") or [], np.float32)
            if (v.get("model"), v.get("version")) != (MODEL, VERSION) or len(vector) != segment_log.VOICE_DIM:
                skipped += 1
                continue
            at = int(r.start_ns)
            try:
                payload_id: str | None = str(uuid.UUID(r.id))  # the broker payload it was exported from
            except ValueError:
                payload_id = None
            matcher.file(vector, int(v["voiceprint"]), None, at, max(int(r.end_ns), at), payload_id=payload_id)
            written += 1
    finally:
        matcher.close()
    if skipped:
        log.warning(
            "skipped %d of %d taught voice vectors: not %s %s with %d floats",
            skipped,
            written + skipped,
            MODEL,
            VERSION,
            segment_log.VOICE_DIM,
        )
    return written


class _NoSearch:
    """For a matcher that only files (`seed_taught`): it never searches."""

    async def nearest(self, *a: object, **k: object) -> list[dict[str, object]]:
        return []


def forget_voice(n: int, *, base: Path | None = None) -> None:
    """Forgets every embedding of voiceprint `n`, whoever it named (the store's forget by column; the vector worker
    drops them from its index as it reads the deletion)."""
    from continuum_py import forget_segments_where

    root = base if base is not None else segment_log.root()
    forget_segments_where(root, "voiceprint", subject(n), streams=[EMBEDDINGS])


def forget_embedding(embedding_id: str, *, base: Path | None = None) -> None:
    """Forgets one filed embedding, by its id (a replaced enrollment)."""
    from continuum_py import forget_segments_where

    root = base if base is not None else segment_log.root()
    forget_segments_where(root, "id", embedding_id, streams=[EMBEDDINGS])


def prints(ns: Sequence[int], limit: int = 64, *, base: Path | None = None) -> dict[int, list[np.ndarray]]:
    """The embeddings stored for each voiceprint in `ns` (its newest `limit`), as unit vectors: what "Who's this?"
    compares voices by (people/voice_guess.py). A voiceprint with none maps to an empty list."""
    wanted = {subject(n): n for n in ns}
    out: dict[int, list[np.ndarray]] = {n: [] for n in ns}
    rows = segment_log.Log(base if base is not None else segment_log.root()).range(EMBEDDINGS, 0.0, time.time() + 1.0)
    for r in rows:  # oldest first, so the newest `limit` are each list's tail
        if (n := wanted.get(r["voiceprint"])) is not None:
            v = np.asarray(r["vector"], np.float32)
            if len(v) and np.all(np.isfinite(v)):
                out[n].append(v / (np.linalg.norm(v) + 1e-9))
    return {n: vectors[-limit:] for n, vectors in out.items()}
