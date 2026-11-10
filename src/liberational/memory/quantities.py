"""Extract only quantities explicitly present in speech, without activity inference.

Supported numbers are unsigned finite decimal literals (optionally correctly
comma-grouped), conventional English cardinals below one million, and English
cardinals followed by ``point`` and digit words. Tens may hyphenate their ones;
``and`` may introduce the remainder of a hundred or thousand. Fractions, signs,
exponents, ranges, percentages, and malformed contiguous number-word runs are
rejected rather than partially interpreted. Units are canonical labels, never
converted. Without a unit, only ``putting on N`` and ``that's N`` are recognized.
Spans cover the original number through its unit, or just the bare number; cues
are excluded. A parse does not assert an activity, an observation, or its truth.
"""

import dataclasses
import hashlib
import json
import math
import pathlib
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from liberational.continuum import keyed_id
from liberational.domain.eventlog import Selector
from liberational.domain.kernel import Event, EventId, EventType, PersonId, Producer, check_instant
from liberational.people import registry


@dataclass(frozen=True)
class SpokenQuantity:
    """An explicit number with its exact half-open character span in source text."""

    value: float
    unit: str | None
    name: str
    start: int
    end: int
    rule: str


_ONES = dict(
    zip(
        (
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
        ),
        range(20),
        strict=True,
    )
)
_TENS = dict(
    zip(
        ("twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"),
        range(20, 100, 10),
        strict=True,
    )
)
_WORDS = set(_ONES) | set(_TENS) | {"hundred", "thousand", "and", "point"}
# Recognize unsupported numeric language as a blocker, rather than extracting its
# last supported suffix (for example "one million five kilograms").
_BLOCKERS = {
    "million",
    "billion",
    "trillion",
    "minus",
    "negative",
    "plus",
    "half",
    "quarter",
    "dozen",
    "percent",
    "percentage",
    "to",
    "through",
}
_TOKEN = re.compile(r"\d+(?:[.,:/%\-]\d+)*|[A-Za-z_][\w]*(?:[-’'][A-Za-z]+)*|[^\s]")
_LITERAL = re.compile(r"(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?\Z")
_ALIASES: dict[str, tuple[str, str]] = {}
for _canonical, _name, _forms in (
    ("kg", "mass", "kg kilogram kilograms"),
    ("g", "mass", "g gram grams"),
    ("lb", "mass", "lb lbs pound pounds"),
    ("oz", "mass", "oz ounce ounces"),
    ("km", "distance", "km kilometer kilometers kilometre kilometres"),
    ("m", "distance", "m meter meters metre metres"),
    ("cm", "distance", "cm centimeter centimeters centimetre centimetres"),
    ("mm", "distance", "mm millimeter millimeters millimetre millimetres"),
    ("mi", "distance", "mi mile miles"),
    ("ft", "distance", "ft foot feet"),
    ("in", "distance", "in inch inches"),
    ("rep", "count", "rep reps repetition repetitions"),
    ("step", "count", "step steps"),
    ("lap", "count", "lap laps"),
):
    for _form in _forms.split():
        _ALIASES[_form] = (_canonical, _name)
_UNIT = re.compile(r"(?<![A-Za-z_])(?:" + "|".join(sorted(_ALIASES, key=len, reverse=True)) + r")(?!\w)", re.IGNORECASE)
_CUE = re.compile(r"\b(?:putting\s+on|that['’]s)\s+", re.IGNORECASE)


def _small(words: list[str]) -> int | None:
    if len(words) == 1:
        return _ONES.get(words[0], _TENS.get(words[0]))
    if len(words) == 2 and words[0] in _TENS and words[1] in _ONES and 0 < _ONES[words[1]] < 10:
        return _TENS[words[0]] + _ONES[words[1]]
    return None


def _group(words: list[str]) -> int | None:
    if "hundred" not in words:
        return _small(words)
    if len(words) < 2 or words[1] != "hundred" or words[0] not in _ONES or not 0 < _ONES[words[0]] < 10:
        return None
    tail = words[2:]
    if not tail:
        return _ONES[words[0]] * 100
    if tail[0] == "and":
        tail = tail[1:]
    remainder = _small(tail)
    return _ONES[words[0]] * 100 + remainder if remainder is not None and remainder > 0 else None


def _cardinal(words: list[str]) -> int | None:
    if "thousand" not in words:
        return _group(words)
    if words.count("thousand") != 1:
        return None
    at = words.index("thousand")
    head = _group(words[:at])
    if head is None or head <= 0:
        return None
    tail = words[at + 1 :]
    if not tail:
        return head * 1000
    if tail[0] == "and":
        tail = tail[1:]
    remainder = _group(tail)
    return head * 1000 + remainder if remainder is not None and remainder > 0 else None


def _number(source: str) -> float | None:
    if _LITERAL.fullmatch(source):
        try:
            value = float(source.replace(",", ""))
        except ValueError:
            return None
        return value if math.isfinite(value) else None
    # A hyphen is grammatical only between a tens word and a nonzero digit.
    for token in source.lower().split():
        if "-" in token:
            parts = token.split("-")
            if len(parts) != 2 or parts[0] not in _TENS or parts[1] not in _ONES or not 0 < _ONES[parts[1]] < 10:
                return None
    words = source.lower().replace("-", " ").split()
    if not words or any(word not in _WORDS for word in words):
        return None
    if "point" not in words:
        number = _cardinal(words)
        return float(number) if number is not None else None
    if words.count("point") != 1:
        return None
    at = words.index("point")
    whole = _cardinal(words[:at])
    digits = words[at + 1 :]
    if whole is None or not digits or any(word not in _ONES or _ONES[word] > 9 for word in digits):
        return None
    value = float(str(whole) + "." + "".join(str(_ONES[word]) for word in digits))
    return value if math.isfinite(value) else None


def _number_token(token: str) -> bool:
    return bool(token[0].isdigit()) or all(part in _WORDS | _BLOCKERS for part in token.lower().split("-"))


def _range_sign(token: str) -> bool:
    """Reject ASCII operators and every Unicode dash as numeric separators."""
    return token in {"−", "+", "/", "%"} or (len(token) == 1 and unicodedata.category(token) == "Pd")


def _before(text: str, end: int) -> tuple[int, int, float] | None:
    tokens = list(_TOKEN.finditer(text, 0, end))
    if not tokens or text[tokens[-1].end() : end].strip():
        return None
    at = len(tokens)
    while at and _number_token(tokens[at - 1].group()):
        at -= 1
    if at == len(tokens):
        return None
    # "and" after another clause is not part of the numeric phrase.
    if tokens[at].group().lower() == "and":
        at += 1
    if at == len(tokens):
        return None
    start = tokens[at].start()
    if start and (text[start - 1].isalnum() or text[start - 1] in "_."):
        return None
    if at and _range_sign(tokens[at - 1].group()):
        return None
    if at > 1 and tokens[at - 1].group() in {".", ","} and _number_token(tokens[at - 2].group()):
        return None
    finish = tokens[-1].end()
    value = _number(text[start:finish])
    return (start, finish, value) if value is not None else None


def parse(text: str) -> tuple[SpokenQuantity, ...]:
    """Return explicit unit quantities and cue-qualified bare numbers in order."""
    found: list[SpokenQuantity] = []
    for match in _UNIT.finditer(text):
        # The inch abbreviation is also a common preposition. A following word
        # needs the unambiguous spelled-out unit instead.
        tail = text[match.end() :].lstrip()
        if match.group().lower() == "in" and tail and tail[0] not in ",.;:!?)]}":
            continue
        number = _before(text, match.start())
        if number is None:
            continue
        start, _, value = number
        unit, name = _ALIASES[match.group().lower()]
        found.append(SpokenQuantity(value, unit, name, start, match.end(), "explicit_unit"))
    for cue in _CUE.finditer(text):
        tokens = list(_TOKEN.finditer(text, cue.end()))
        end = cue.end()
        for token in tokens:
            if not _number_token(token.group()):
                break
            end = token.end()
        if end == cue.end():
            continue
        # A number prefix followed by numeric syntax is not a complete number.
        tail = text[end:]
        if tail and (tail[0].isalnum() or tail[0] == "_"):
            continue
        stripped = tail.lstrip()
        if stripped and _range_sign(stripped[0]):
            continue
        if stripped.startswith((".", ",")) and re.match(r"[.,]\s*[\d.,]", stripped):
            continue
        value = _number(text[cue.end() : end])
        if value is None or any(cue.end() < item.end and end > item.start for item in found):
            continue
        found.append(SpokenQuantity(value, None, "quantity", cue.end(), end, "bare_cue"))
    return tuple(sorted(found, key=lambda item: (item.start, item.end)))


_VERSION = "1"


def record(state: pathlib.Path, lines: Iterable[tuple[dict[str, Any], PersonId | None]]) -> None:
    """Write quantities once per utterance/match before its transcript is acknowledged.

    The subject is the reader's canonical identity snapshot, never a display name.
    Anonymous lines retain a voice reference when available. Receipts exclude
    mutable attribution and measurement content, and survive person purges.
    Invalid timestamps and assistant replies cannot become human measurements.
    """
    keyed: list[tuple[str, Event]] = []
    for line, person in lines:
        text, timestamp = line.get("text"), line.get("ts_start")
        if line.get("assistant") or not isinstance(text, str):
            continue
        if isinstance(timestamp, bool) or not isinstance(timestamp, int | float):
            continue
        try:
            check_instant(timestamp, "utterance start")
        except ValueError, TypeError:
            continue
        device = line.get("device")
        device = device if isinstance(device, str) else None
        evidence = registry.utterance_evidence(timestamp, device)
        sid = line.get("speaker_id")
        voice = None
        if isinstance(sid, int) and not isinstance(sid, bool):
            voice = registry.voiceprint(sid).id
        for ordinal, quantity in enumerate(parse(text)):
            # The event's id comes from the exact source time, device, and the quantity's place in the line (the key
            # earlier receipts used), so a line measured twice retries the same immutable event.
            identity = json.dumps([_VERSION, float(timestamp).hex(), device, ordinal])
            key = hashlib.sha256(identity.encode()).hexdigest()
            event = Event.new(
                EventType.MEASUREMENT_RECORDED,
                stream=registry.MEASUREMENTS,
                source="identity",
                occurred_at=timestamp,
                subject=person,
                producer=Producer("spoken-quantity-rules", _VERSION),
                evidence=(evidence,),
                payload={
                    "name": quantity.name,
                    "value": quantity.value,
                    "unit": quantity.unit,
                    "source": "speech",
                    "utterance_id": evidence.id,
                    "device": device,
                    "voice_id": voice,
                    "span": [quantity.start, quantity.end],
                    "rule": quantity.rule,
                },
            )
            keyed.append(
                (
                    key,
                    dataclasses.replace(
                        event, event_id=EventId(keyed_id(timestamp, "spoken-quantity", key)), recorded_at=timestamp
                    ),
                )
            )
    if keyed:
        reg = registry.at(state)
        # A person forgotten between the transcript's reading and this write (their events purged, PersonForgotten
        # applied) gets no new event about them: the registry no longer knows them.
        kept = [e for _, e in keyed if e.subject is None or reg.person(PersonId(e.subject)) is not None]
        if kept:
            reg.log.append(kept)
            # Forgotten between that check and the write: the broker's log has no receipt to refuse the event with,
            # so take it out again at once, under the purge's own reason.
            gone = [e for e in kept if e.subject is not None and reg.person(PersonId(e.subject)) is None]
            if gone:
                reg.log.purge(Selector(event_ids=frozenset(e.event_id for e in gone)), reason="PersonForgotten")
