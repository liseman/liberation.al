"""Questions about an animal (#1318): "what has the cat been doing lately", "when did the cat last use the litter box".

history_qa sends a question here when it asks about an animal the detector knows ("the cat") or a named pet
(people/pets.py, applied to its species by the single-pet rule).

- What it has been doing (`asked`, `answer`), and when it was last seen with no zone named: these were animal
  tracking's questions (see-tracks), which was removed (#1808). The answer says so plainly, never "nothing happened".
- When it was last in a zone (`last_asked`, `last_seen`): its newest visit to the zone named (`zone.visit`, #1316),
  over the zone's coverage that says whether the cameras were watching for animals there since.

Which visits are the pet (`attribute`), when a question names one: a track the owner labeled as that pet always is,
and one labeled as another animal never is. While recognition by appearance is off for the species
(people/pet_recognition.py), the only named pet of a species is assumed to be any other track of it, and the answer
says "probably"; with several named pets of the species, only labeled tracks count. Once recognition is on, tracks it
recognizes as the pet count too, and no others.
"""

import dataclasses
import re
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from liberational import segment_log
from liberational.memory import zone_counts

LATELY_DAYS = 7
WHAT = re.compile(
    r"\bwhat\s+(?:has|have|had|did|does|is|was)\s+(?P<a>.+?)\s+(?:been\s+)?(?:doing|up to|done)\b"
    r"|\bwhat's\s+(?P<b>.+?)\s+been\s+(?:doing|up to)\b"
    r"|\bhow\s+(?:has|is)\s+(?P<c>.+?)\s+been(?:\s+doing)?\b",
    re.I,
)
# "when did I last see the cat", "when was May last seen", "where is the cat", "have you seen May lately", "when did
# the cat last use the litter box": the subject is group s; a zone named after "last" is group z.
LAST = re.compile(
    r"\bwhen\s+(?:did|have)\s+(?:i|we|you)\s+last\s+(?:see|seen|spot|spotted)\s+(?P<a>.+?)\s*(?:\?|$|\btoday\b|\bthis\b)"
    r"|\bwhen\s+(?:was|were|is)\s+(?P<b>.+?)\s+last\s+seen\b"
    r"|\bwhere(?:\s+is|'s|\s+was)\s+(?P<c>.+?)\s*(?:\?|$|\bnow\b|\bright now\b)"
    r"|\b(?:have|did)\s+(?:you|i|we)\s+seen?\s+(?P<d>.+?)\s*(?:\?|$|\blately\b|\btoday\b|\brecently\b)"
    r"|\bwhen\s+did\s+(?P<e>.+?)\s+last\s+(?:use|go\s+to|visit|go\s+in(?:to)?|sit\s+in|get\s+in(?:to)?)\s+(?P<z>.+?)\s*(?:\?|$)",
    re.I,
)
DETERMINERS = frozenset({"the", "my", "our", "a", "your"})
MAX_EVIDENCE = 30


def _pets(state: Path) -> list[dict[str, Any]]:
    """The named pets, or none when the roster can't be read."""
    try:
        from liberational.people import pets

        return pets.at(state).listing()
    except Exception:
        return []


# Time and camera qualifiers history_qa.parse reads for the window and cameras: never part of a subject or zone name
# ("when did I last see the cat yesterday", "the cat on camera 2", "the litter box today").
QUALIFIERS = re.compile(
    r"\b(?:on|from|in)\s+(?:the\s+)?cam(?:era)?\s*[\w-]+"
    r"|\b(?:this|last|past|yesterday|today|tonight)\s+(?:morning|afternoon|evening|night|week|month|weekend)\b"
    r"|\b(?:on\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"
    r"|\b(?:today|yesterday|tonight|lately|recently|now|earlier|so far|anymore|again)\b"
    r"|\b(?:in\s+the\s+)?(?:last|past)\s+\d+\s+(?:minutes?|hours?|days?|weeks?)\b"
    r"|\b(?:since|before|after|at|around|by)\s+\S+",
    re.I,
)


def bare(phrase: str) -> str:
    """The phrase without time and camera qualifiers, or configured camera names said as cameras."""
    from liberational.vision import views

    try:
        phrase = views.configured_camera_mentions(phrase)[1]
    except Exception:
        pass
    return " ".join(QUALIFIERS.sub(" ", phrase).replace("?", " ").split())


def subject(phrase: str, state: Path) -> tuple[str, str | None] | None:
    """(species, pet name) for a phrase naming one animal ("the cat", "May"), or None. A pet's name is looked for
    before time qualifiers are stripped, so a pet called Friday is still Friday ("where is Friday yesterday")."""
    roster = _pets(state)
    said = [w for w in re.findall(r"[a-z']+", phrase.lower()) if w not in DETERMINERS]
    if said:
        first = said[0].removesuffix("'s")
        for p in roster:
            if str(p.get("name", "")).casefold() == first and (len(said) == 1 or bare(" ".join(said[1:])) == ""):
                return str(p["species"]), str(p["name"])
    words = [w for w in re.findall(r"[a-z']+", bare(phrase).lower()) if w not in DETERMINERS]
    if len(words) != 1:
        return None
    word = words[0].removesuffix("'s")
    for p in roster:
        if str(p.get("name", "")).casefold() == word:
            return str(p["species"]), str(p["name"])
    species = word.removesuffix("s") if word.removesuffix("s") in zone_counts.ANIMALS else word
    if species not in zone_counts.ANIMALS:
        return None
    named = [p for p in roster if p.get("species") == species]
    return species, (str(named[0]["name"]) if len(named) == 1 else None)


def asked(question: str, state: Path) -> tuple[str, str | None] | None:
    """(species, pet name) the question asks about, or None when it isn't asking what an animal has been doing."""
    m = WHAT.search(question)
    return None if m is None else subject(m["a"] or m["b"] or m["c"], state)


def last_asked(question: str, state: Path) -> tuple[str, str | None, str | None] | None:
    """(species, pet name, zone name or None) when the question asks when an animal was last seen, or last used a
    zone; None otherwise (a person's last-seen question is the timeline's)."""
    m = LAST.search(question.strip())
    if m is None:
        return None
    found = subject(m["a"] or m["b"] or m["c"] or m["d"] or m["e"] or "", state)
    if found is None:
        return None
    zone = None
    if m["z"]:
        zone = next((n for n in zone_counts.names(state) if _same(n, bare(m["z"]))), None)
        if zone is None:
            return None
    return found[0], found[1], zone


def _same(name: str | None, phrase: str) -> bool:
    from liberational.memory import event_counts

    return event_counts.same(name, phrase)


@dataclasses.dataclass(frozen=True)
class Attribution:
    """Which of a species' tracks are the animal asked about, and how to say so."""

    who: str  # "the cat", "the cat (probably May, the only cat you've named)", "May (the cat)"
    tracks: frozenset[str] | None  # track ids that count; None: every track of the species
    excluded: frozenset[str] = frozenset()  # tracks labeled or recognized as another animal
    note: str = ""


def attribute(state: Path, species: str, pet: str | None, now: float | None = None) -> Attribution:
    """See the module's docs: labels, then recognition when it is on, then the single-pet assumption with "probably"."""
    if pet is None:
        return Attribution(f"the {species}", None)
    roster = _pets(state)
    mine = next((p for p in roster if str(p.get("name")) == pet and p.get("species") == species), None)
    same_species = [p for p in roster if p.get("species") == species]
    labeled: set[str] = set()
    others: set[str] = set()
    on = False
    try:
        from liberational.people import pet_recognition

        at = time.time() if now is None else now
        # Owner labels hold for every track they name, however old: read them all, not the view's rolling window.
        for tid, row in pet_recognition.labels(state).items():
            said = row["label"]
            (labeled if mine is not None and said == mine.get("id") else others).add(str(tid))
        # Recognition by appearance compares embeddings, which the view reads over its own window.
        view = pet_recognition.view(state, roster, at)
        on = any(c["species"] == species and c["on"] for c in view["recognition"])
        if on:
            for g in view["groups"]:
                for s in g["sightings"]:
                    said = (s.get("recognized") or {}).get("label")
                    if s["species"] != species or s.get("label") is not None or said is None:
                        continue
                    (labeled if mine is not None and said == mine.get("id") else others).add(str(s["track_id"]))
    except Exception:
        pass  # no embeddings or labels: the assumption below, or the species alone
    if on:
        return Attribution(
            f"{pet} (the {species})",
            frozenset(labeled),
            frozenset(others),
            f"Only sightings labeled or recognized as {pet} are counted.",
        )
    if len(same_species) == 1:
        return Attribution(
            f"the {species} (probably {pet}, the only {species} you've named)",
            None,
            frozenset(others),
            f"Animals aren't told apart by appearance yet, so any {species} seen is taken to be {pet}"
            + (", except sightings you labeled as another animal." if others else "."),
        )
    return Attribution(
        f"{pet} (the {species})",
        frozenset(labeled),
        frozenset(others),
        f"You've named more than one {species} and they aren't told apart by appearance yet, so only sightings you "
        f"labeled as {pet} are counted.",
    )


def keeps(attribution: Attribution, track: object) -> bool:
    """Whether a record with this track id counts for the attributed animal."""
    tid = str(track) if track is not None else None
    if tid is not None and tid in attribution.excluded:
        return False
    return attribution.tracks is None or (tid is not None and tid in attribution.tracks)


def _dur(seconds: float) -> str:
    """A plain length: "40 s", "3 min", "1 h", "2 h 5 min", or "3 days" from two days on."""
    if seconds < 60:
        return f"{int(seconds + 0.5)} s"
    minutes = int(seconds / 60 + 0.5)
    if minutes < 60:
        return f"{minutes} min"
    if seconds >= 2 * 86400:
        return f"{int(seconds / 86400 + 0.5)} days"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes} min" if minutes else f"{hours} h"


def _union(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        elif b > a:
            out.append((a, b))
    return out


def _seconds(spans: list[tuple[float, float]]) -> float:
    return sum(b - a for a, b in _union(spans))


def _clip(r: Mapping[str, Any], start: float, end: float) -> tuple[float, float]:
    return max(float(r["ts_start"]), start), min(float(r.get("ts_end") or r["ts_start"]), end)


def answer(species: str, pet: str | None, state: Path, now: float) -> dict[str, Any]:
    """{answer, evidence, coverage, warnings} for what the animal has been doing: animal tracking was removed."""
    who = attribute(state, species, pet, now).who
    return removed(f"what {who} has been doing", species)


LAST_LOOKBACK_DAYS = 30  # how far back a last-seen question looks
REMOVED = "Animal tracking was removed"


def removed(what: str, species: str) -> dict[str, Any]:
    """The plain answer for a question only animal tracking (see-tracks, deleted in #1808) could answer."""
    return dict(
        answer=f"{REMOVED}, so I can't say {what}.",
        evidence=[],
        coverage={"available": False, "species": species, "reason": "removed"},
        warnings=[f"{REMOVED} (#1808): this is not a report that it wasn't there."],
    )


def last_seen(
    species: str,
    pet: str | None,
    zone: str | None,
    query: Mapping[str, Any],
    state: Path,
    stamp: Callable[[float], str],
    now: float,
) -> dict[str, Any]:
    """{answer, evidence, coverage, warnings}: when the animal was last in `zone`, and whether the cameras were
    watching for animals there after that, so "not seen since" means something. Without a zone it was animal
    tracking's question, and that was removed."""
    start, end = float(query.get("start") or 0), min(float(query.get("end") or now), now)
    if start <= 0 or end - start < 60:
        start, end = now - LAST_LOOKBACK_DAYS * 86400.0, now
    cameras = set(query.get("cameras") or [])
    whose = attribute(state, species, pet, now)
    who = whose.who
    if zone is None:
        return removed(f"when {who} was last seen", species)
    segments = segment_log.at(state)

    def rows(stream: str) -> list[dict[str, Any]]:
        return [r for r in segments.range(stream, start, end) if not cameras or r.get("cam") in cameras]

    # A zone's visits are only seen where its own coverage says animals were looked for in it.
    ids = set(zone_counts._matching(state, zone, _same))
    checks = [
        _clip(c, start, end)
        for c in rows(segment_log.ZONE_COVERAGE)
        if c.get("subject_kind") == "animal" and set(c.get("zones") or []) & ids
    ]
    coverage: dict[str, Any] = {"start": start, "end": end, "checked_seconds": _seconds(checks), "species": species}
    found = [
        v
        for v in rows(segment_log.ZONE_VISITS)
        if v.get("label") == species
        and v.get("zone") in ids  # the zone's stable ids: visits from before a rename still count
        and float(v.get("dwell_s") or 0) >= zone_counts.MIN_DWELL_S
        and keeps(whose, v.get("track_id"))
    ]
    what = f"in {zone}"
    method = "ZoneVisit: a tracked animal inside an owner-drawn zone (#1316), an unconfirmed candidate"
    days = round((end - start) / 86400.0)
    if not checks:
        return dict(
            answer=f"No footage in the last {days} days has been checked for animals, so I can't say when {who} was "
            f"last {what}.",
            evidence=[],
            coverage={**coverage, "available": False},
            warnings=["Nothing covers the window: this is not a report that it wasn't there."],
        )
    if not found:
        text = f"I haven't seen {who} in {zone} in the last {days} days of checked footage ({_dur(_seconds(checks))})."
        if whose.note:
            text += f" {whose.note}"
        return dict(answer=text, evidence=[], coverage={**coverage, "available": True, "found": False}, warnings=[])
    last = max(found, key=lambda r: float(r.get("ts_end") or r["ts_start"]))
    at = float(last.get("ts_end") or last["ts_start"])
    since = _seconds([(max(a, at), b) for a, b in checks if b > at])
    head = f"{who[0].upper()}{who[1:]}"
    place = f" on {last.get('cam')}" if last.get("cam") else ""
    text = f"{head} was last {what} at {stamp(at)[:16].replace('T', ' at ')}{place}"
    text += f", for {_dur(float(last.get('dwell_s') or at - float(last['ts_start'])))}"
    text += "."
    if now - at > 120:
        text += (
            f" The cameras checked {_dur(since)} of footage for animals since then."
            if since >= 60
            else " No footage since then has been checked for animals yet."
        )
    if whose.note:
        text += f" {whose.note}"
    text += " This is a detected visit, not a confirmed sighting."
    ts = float(last["ts_start"])
    evidence = [
        dict(
            ts=ts,
            camera=last.get("cam"),
            source="camera",
            quote=f"a {species} {what} on {last.get('cam')}",
            link="/review#" + urlencode(dict(day=stamp(ts)[:10], t=ts, cam=last.get("cam") or "")),
            method=method,
            event=last.get("id"),
        )
    ]
    coverage.update(available=True, found=True, last=at, checked_since_seconds=since)
    return dict(answer=text, evidence=evidence, coverage=coverage, warnings=[])
