"""Who "she", "he", or "they" is in a question to Ask or voice: "how many times did she use the litter box today".

A question that names nobody but uses a third-person pronoun is about the subject of the conversation so far: the
newest earlier question on the same surface (Ask's history, memory/ask_history.py), asked at most RECENT_S before it,
that named someone or an animal. A person comes from the names history_qa finds; an animal from a pet's name or a
species the detector knows ("the cat"). That earlier question may itself have used a pronoun, so it is resolved the
same way, at most DEPTH questions back.

With no such question, "she" or "he" (her, him, his, hers) is the one named pet whose sex the owner set to match,
else the only named pet. A person's sex is never guessed, and "they" is only ever the conversation's subject.

The question is answered with the pronoun replaced by the name, and the answer says so first ("Taking "she" as
May."), so a wrong guess shows. A question that names a subject itself keeps its pronouns: "did Dana say she was
coming" is about Dana.
"""

import dataclasses
import re
from pathlib import Path
from typing import Any

RECENT_S = 600.0  # an earlier question this recent sets who "she" is
DEPTH = 3  # how many earlier questions back a pronoun can be followed
SUBJECT = re.compile(r"\b(she|he|they|her|him|them|his|hers|their|theirs)\b", re.I)
POSSESSIVE = frozenset({"his", "hers", "their", "theirs"})
FEMALE = frozenset({"she", "her", "hers"})
MALE = frozenset({"he", "him", "his"})


@dataclasses.dataclass(frozen=True)
class Resolved:
    question: str  # with the pronoun replaced
    said: str  # the pronoun as asked ("she")
    meant: str  # who it was taken as ("May", "the cat", "Dana")

    @property
    def note(self) -> str:
        return f'Taking "{self.said}" as {self.meant}.'


def _pets(state: Path) -> list[dict[str, Any]]:
    from liberational.memory import subject_activity

    return subject_activity._pets(state)


def subject_of(question: str, state: Path) -> str | None:
    """The one person or animal a question names, as it would be said again ("Dana", "May", "the cat"), or None."""
    from liberational.memory import history_qa, zone_counts

    people = [p for p in history_qa.people_named(question) if p]
    pets = {str(p["name"]).casefold(): str(p["name"]) for p in _pets(state)}
    words = [w.removesuffix("'s") for w in re.findall(r"[A-Za-z']+", question)]
    named = sorted({pets[w.casefold()] for w in words if w.casefold() in pets})
    species = sorted(
        {
            w.casefold().removesuffix("s")
            for w in words
            if w.casefold().removesuffix("s") in zone_counts.ANIMALS or w.casefold() in zone_counts.ANIMALS
        }
    )
    people = [p for p in people if p.casefold() not in pets]
    found = [*people, *named, *(f"the {s}" for s in species if not named)]
    if len(found) == 1:
        return found[0]
    if len(people) > 1 and not named and not species:
        return " and ".join(people)  # "what did Dana and Eve talk about" ... "what did they decide"
    return None


def _earlier(state: Path, surface: str, before: float, depth: int) -> str | None:
    """The subject of the newest earlier question on `surface` within RECENT_S of `before`."""
    if depth <= 0:
        return None
    try:
        from liberational.memory import ask_history

        # Only a question within RECENT_S can set the subject, so the read is bounded to that window: this runs on
        # every question that says she, he, or they, up to DEPTH times.
        entries = ask_history.at(state).list_entries(surface=surface, limit=10, since=before - RECENT_S)
    except Exception:
        return None
    for e in entries:
        at = float(e.get("created_at") or 0)
        if at >= before:
            continue
        if before - at > RECENT_S:
            return None
        question = str(e.get("question") or "")
        if (who := subject_of(question, state)) is not None:
            return who
        if SUBJECT.search(question):
            return _earlier(state, surface, at, depth - 1)
        before = at
    return None


def _pet_by_sex(state: Path, pronoun: str) -> str | None:
    pets = _pets(state)
    sex = "female" if pronoun in FEMALE else "male" if pronoun in MALE else None
    if sex is None:
        return None
    matching = [p for p in pets if p.get("sex") == sex]
    if len(matching) == 1:
        return str(matching[0]["name"])
    return str(pets[0]["name"]) if len(pets) == 1 and pets[0].get("sex") in (None, sex) else None


def resolve(question: str, state: Path, surface: str, now: float) -> Resolved | None:
    """The question with its pronoun replaced (see the module's docs), or None to ask it as it is."""
    m = SUBJECT.search(question)
    if m is None or subject_of(question, state) is not None:
        return None
    said = m.group(1)
    meant = _earlier(state, surface, now, DEPTH) or _pet_by_sex(state, said.lower())
    if meant is None:
        return None

    def put(match: re.Match[str]) -> str:
        word = match.group(1)
        out = f"{meant}'s" if word.lower() in POSSESSIVE else meant
        return out[0].upper() + out[1:] if word[0].isupper() and match.start() == 0 else out

    return Resolved(SUBJECT.sub(put, question), said, meant)
