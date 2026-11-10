"""Count questions about an event, answered from detected events, never from captions.

"How many times did I open the fridge today", "how often did the dog get on the couch this week": history_qa sends a
question here when events.answer_counts is on, its reading's measure is a count (memory/question.py), and it names
an event the detector or the owner has named (memory/events.py), a drawn zone, or an action the camera model
detects (`wanted`, from the structured query; nothing here reads the question's text). The answer is the count per span from the promoted events, the
coverage (the recordings the detector has finished in the window), and a Review link for each span. When no finished
recording covers the window it says so instead of estimating, and a window with recordings covered but nothing found
is a real zero, unless matching unverified evidence is being withheld from promotion. A named unverified candidate
routes to an explicit unknown count, never absence; a promoted non-repeating event counts once. Events nobody has named yet are counted apart ("N spans of something not named yet"), never as the
event asked about.

Nothing here knows any kind of event in advance: a question is one of these only when it names one the store has, so
"how many tablespoons in a cup" or "how many meetings tomorrow" go the way they always did. A question naming a zone
drawn on a camera's view ("how many times did the cat use the litter box") is answered from zone visits instead
(memory/zone_counts.py), unless a detected event has that name too.

Retired rules-based activity kinds are unavailable during the model-detector rebuild. They are never
counted from old rows; history_qa answers those requests explicitly before consulting evidence.
"""

import datetime as dt
import re
import time
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import urlencode

from liberational import config as cfg
from liberational.memory import activity_kinds
from liberational.memory import question as reader
from liberational.memory.events import RETIRED_KINDS

WORD = re.compile(r"[a-z0-9]+")
# Words a name and a question share without saying what happened: left out of both before they're compared.
STOP = frozenset({
    "a", "an", "the", "my", "our", "your", "his", "her", "their", "its", "i", "we", "you", "he", "she", "they", "it",
    "me", "us", "them", "this", "that", "these", "those", "did", "do", "does", "was", "were", "is", "are", "be", "been",
    "being", "has", "have", "had", "get", "got", "gets", "getting", "go", "went", "goes", "going", "on", "in", "at",
    "of", "to", "into", "onto", "from", "by", "for", "with", "up", "down", "out", "off", "over", "today", "yesterday",
})  # fmt: skip


# What the camera model doesn't detect: the retired rules' stillness and repeated movement, say.
UNSUPPORTED = (
    "The camera model doesn't detect that. It recognizes sitting down, standing up, picking something up, putting "
    "something down, falling, drinking, eating, phone calls, and phone use: ask when one happened, how many times, "
    "or what someone was doing."
)
# The retired rules' kinds, some of which see-actions' model now produces under the same names: a question naming
# one is about activity (retired_question), answered from the model where it can be (action_kind).
REBUILDING_KINDS = ("stand_up", "sit_down", "repeated_movement", "pickup", "putdown", "phone_use", "eating_or_drinking")
# see-actions' model kinds a question can ask about by name (the reading's `action`, memory/question.py).
MODEL_ACTIONS = frozenset(reader.ACTIONS) - {""} - reader.UNDETECTED


def stem(word: str) -> str:
    """A word without the endings a name and a question differ by: opened, opens, and opening are open; closed,
    closes, and close are clos; stopped is stop; couches is couch."""
    w = word.lower()
    for end, keep in (("sses", "ss"), ("shes", "sh"), ("ches", "ch"), ("xes", "x"), ("ies", "y")):
        if w.endswith(end) and len(w) > len(end) + 1:
            return w[: -len(end)] + keep
    for end in ("ing", "ed"):
        if w.endswith(end) and len(w) > len(end) + 2:
            w = w[: -len(end)]
            if len(w) > 2 and w[-1] == w[-2] and w[-1] not in "lsz":  # stopped, sitting
                w = w[:-1]
            break
    else:
        if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
            w = w[:-1]
    if w.endswith("e") and len(w) > 3:  # close and closed are one
        w = w[:-1]
    return w


def content(text: str) -> list[str]:
    """The stemmed words that say what happened, without articles, pronouns, and the like (STOP), hyphens and
    apostrophes joined: "the check-ins" is ["checkin"]."""
    return [stem(w) for w in WORD.findall(re.sub(r"['-]", "", text.lower())) if w not in STOP]


def usable(name: str) -> bool:
    """Whether a name says enough to be asked about: a word of its own longer than two letters ("in" and "it" can't
    take over questions)."""
    return any(len(w) > 2 for w in content(name))


def mentions(question: str, name: str) -> bool:
    """Whether the question names the event: every word of the name that says something is in it, in any order.
    "how many times did I open the fridge" and "how often was the fridge opened" mention "opened the fridge"."""
    want = content(name)
    have = set(content(question))
    return bool(want) and all(w in have for w in want)


def same(name: str | None, asked: str) -> bool:
    """Whether an event's name is the one asked about: any case, plural, tense, hyphens, and spaces aside."""
    return name is not None and sorted(content(name)) == sorted(content(asked))


def owner_name(e: Mapping[str, Any]) -> str | None:
    """The name the owner gave an event, or None: memory/events.py gives an unnamed span its detector's kind
    as its name, which is no name anyone asked about."""
    name = e.get("name")
    return None if name is None or name == e.get("kind") else str(name)


def module() -> Any:
    """memory/events.py, or None while it isn't installed: then nothing is analyzed here yet."""
    try:
        from liberational.memory import events
    except ImportError:
        return None
    return events


# state -> (when read, the names events have, the detector kinds each name was given to by its content words)
_NAMES: dict[str, tuple[float, list[str], dict[tuple[str, ...], set[str]]]] = {}
NAMES_TTL_S = 60.0


def known_names(state: Any) -> list[str]:
    """The names the detector or the owner has given events, as first spelled, read at most once a minute."""
    return _read_names(state)[1]


def _read_names(state: Any) -> tuple[float, list[str], dict[tuple[str, ...], set[str]]]:
    key = str(state)
    hit = _NAMES.get(key)
    if hit is not None and time.monotonic() - hit[0] < NAMES_TTL_S:
        return hit
    names: dict[tuple[str, ...], str] = {}
    kinds: dict[tuple[str, ...], set[str]] = {}
    act = module()
    if act is not None:
        try:
            projection = getattr(act, "count_evidence", None)
            if callable(projection):
                promoted, unverified = projection(state, 0.0, 1e12)
                rows = [*promoted, *unverified]
            else:
                rows = act.events(state, 0.0, 1e12)
            for e in rows:
                if e.get("kind") in RETIRED_KINDS:
                    continue
                if name := owner_name(e):
                    words = tuple(sorted(content(name)))
                    names.setdefault(words, name)
                    if isinstance(e.get("kind"), str):
                        kinds.setdefault(words, set()).add(e["kind"])
        except Exception:
            names, kinds = {}, {}
    found = [n for k, n in names.items() if k]
    _NAMES[key] = (time.monotonic(), found, kinds)
    return _NAMES[key]


def zone_names(state: Any) -> list[str]:
    """The names of the zones drawn on cameras' views: a count of visits to one is answered here too."""
    try:
        from liberational.memory import zone_counts

        return zone_counts.names(state)
    except Exception:
        return []


def _is_zone(state: Any, asked: str) -> bool:
    """Whether `asked` is a zone's name and no detected event's: an event of that name keeps its own answer."""
    return any(same(n, asked) for n in zone_names(state)) and not any(same(n, asked) for n in known_names(state))


def wanted(query: Mapping[str, Any], state: Any) -> str | None:
    """The event, zone, or model action a count asks about, while events.answer_counts is on: a name the reading
    picked from the owner's own (an event before a zone), else the camera model's kind. None for anything that
    isn't a count, and for a question about what was said."""
    if not cfg.get("events.answer_counts") or query.get("measure") != "count":
        return None
    if query.get("kind") in ("transcript", "journal"):
        return None
    subjects = query.get("subjects") or {}
    action = query.get("action")
    return subjects.get("event") or subjects.get("zone") or (action if action in MODEL_ACTIONS else None)


def _by(e: Mapping[str, Any], person: Any) -> str:
    """by Dana, after a span whose person is known, unless the question already names whose it was."""
    return f" by {e['person_name']}" if e.get("person_name") and person is None else ""


def _clock(ts: float, tz: dt.tzinfo | None) -> str:
    return dt.datetime.fromtimestamp(ts, tz).strftime("%H:%M")


def _n(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def _and(items: list[str]) -> str:
    return (
        items[0] if len(items) == 1 else ", ".join(items[:-1]) + (", and " if len(items) > 2 else " and ") + items[-1]
    )


def _overlaps(event: Mapping[str, Any], start: float, end: float) -> bool:
    """Point events include the query start; ordinary spans retain half-open interval overlap."""
    first, last = float(event["start"]), float(event["end"])
    if end <= start:
        return False
    return start <= first < end if first == last else first < end and last > start


def _occurrences(event: Mapping[str, Any]) -> int:
    """An event occurs once unless its producer supplied a repetition count."""
    count = event.get("count")
    return 1 if count is None else int(count)


def uncalibrated(state: Any, kind: str) -> int | None:
    """How many more Yes or No answers `kind` needs before Ask counts it (memory/events.py's promotion), or None once
    it counts. A model kind with too few answers has no trustworthy count yet."""
    from liberational.memory import events

    rows = [r for r in events.promotion_measurements(state) if r["kind"] == kind and r["camera"] is None]
    if rows and rows[0]["status"] == "counted":
        return None
    minimum = int(cfg.get("activity.promotion_min_ratings"))
    return rows[0]["remaining_ratings"] if rows and rows[0]["remaining_ratings"] is not None else minimum


def _model_kind_unchecked(
    kind: str,
    remaining: int,
    unverified: list[dict[str, Any]],
    coverage: dict[str, Any],
    start: float,
    end: float,
    cameras: list[str],
    stamp: Callable[[float], str],
) -> dict[str, Any]:
    """The answer for a model kind the owner has not checked with Yes or No in Known, Activities: what it saw,
    unverified, and how to check it, never a count."""
    what = kind.replace("_", " ")
    seen = [
        e
        for e in unverified
        if e.get("kind") == kind and (not cameras or e.get("camera") in cameras) and _overlaps(e, start, end)
    ]
    watched = sum(max(0.0, min(c["end"], end) - max(c["start"], start)) for c in coverage["covered"])
    text = (
        f"I can't count {what} yet: the camera model's {what} detections aren't checked. "
        f"Answer {remaining} more of them Yes or No in Known, Activities "
        f"(Not sure sets one aside), and I'll count them."
    )
    text += (
        f" In that time it flagged {_n(len(seen), 'possible ' + what, 'possible ' + what + 's')}, unverified."
        if seen
        else f" It flagged none in the {round(watched / 60)} min it watched, which isn't a count of zero."
    )
    return dict(
        answer=text,
        evidence=[
            dict(
                ts=float(e["start"]),
                camera=e.get("camera"),
                source="camera",
                quote=f"Unverified: {what}",
                event=e.get("id"),
                link="/review#" + urlencode(dict(day=stamp(float(e["start"]))[:10], t=float(e["start"]))),
                method="see-actions model candidate (action.span), not yet checked by ratings",
                stream="action.span",
            )
            for e in seen
        ],
        coverage={**coverage, "count_available": False, "unverified": len(seen), "ratings_needed": remaining},
        warnings=["The model's detections of this kind aren't checked yet: this is not a count."],
    )


def answer(
    asked: str,
    query: Mapping[str, Any],
    state: Any,
    stamp: Callable[[float], str],
    tz: dt.tzinfo | None = None,
    person: Any = None,
) -> dict[str, Any]:
    """{answer, evidence, coverage, warnings, count, dwell_s} for how many times `asked` happened in query's window;
    `count` is there only when the window could be counted (memory/typical_counts.py reads it), and `dwell_s` beside
    it is the counted spans' time inside the window (memory/journal_counts.py's durations). `person` (a PersonId)
    keeps to the spans someone was recognized in, when the question names who."""
    if _is_zone(state, asked):
        from liberational.memory import zone_counts

        return zone_counts.answer(asked, query, state, stamp, tz, same=same, person=person)
    model_kinds = activity_kinds.model_kinds(state)  # the owner's activities read from a model
    if asked not in model_kinds and (asked in REBUILDING_KINDS or asked in reader.ACTIONS):
        return dict(
            answer=UNSUPPORTED, evidence=[], coverage={"available": False, "reason": "unsupported"}, warnings=[]
        )
    start, end, cameras = float(query["start"]), float(query["end"]), list(query.get("cameras") or [])
    act = module()
    if act is None:
        return dict(
            answer=f"Event detection isn't set up here yet, so I can't count {asked}.",
            evidence=[],
            coverage={"recordings": 0, "available": False},
            warnings=["memory/events.py is not installed."],
        )
    projection = getattr(act, "count_evidence", None)
    promoted, unverified = (
        projection(state, start, end) if callable(projection) else (act.events(state, start, end), [])
    )
    promoted = [e for e in promoted if e.get("kind") not in RETIRED_KINDS]
    unverified = [e for e in unverified if e.get("kind") not in RETIRED_KINDS]

    def matches(e: Mapping[str, Any]) -> bool:
        # A model kind counts its own spans, named or not; any other event counts by the owner's name.
        return (asked in model_kinds and e.get("kind") == asked) or same(owner_name(e), asked)

    found = act.coverage(state, start, end, kind=asked) if asked in model_kinds else act.coverage(state, start, end)
    covered = [(c, a, z) for c, a, z in found if not cameras or c in cameras]
    covered = [(c, a, z) for c, a, z in covered if a < end and z > start]
    spans = [e for e in promoted if not cameras or e.get("camera") in cameras]
    spans = [e for e in spans if _overlaps(e, start, end)]
    coverage = {
        "recordings": len(covered),
        "covered": [{"camera": c, "start": a, "end": z} for c, a, z in covered],
        "available": True,
    }
    if not covered:
        return dict(
            answer=f"No recording in that time has been analyzed for events yet, so I can't count {asked}.",
            evidence=[],
            coverage=coverage,
            warnings=["Nothing covers the window: this is not a count of zero."],
        )
    if asked in model_kinds and (gate := uncalibrated(state, asked)) is not None:
        return _model_kind_unchecked(asked, gate, unverified, coverage, start, end, cameras, stamp)
    pending = [
        e
        for e in unverified
        if (not cameras or e.get("camera") in cameras)
        and _overlaps(e, start, end)
        and matches(e)
        and (person is None or e.get("person") in (None, person))
    ]
    if pending:
        unverified_text = (
            f"I found unverified candidates named {asked}, so I can't give a verified count for that time."
        )
        return dict(
            answer=unverified_text,
            evidence=[
                dict(
                    ts=float(e["start"]),
                    camera=e.get("camera"),
                    source="camera",
                    quote=f"Unverified candidate: {asked}",
                    event=e.get("id"),
                    link="/review#" + urlencode(dict(day=stamp(float(e["start"]))[:10], t=float(e["start"]))),
                    method="ActivityDetected candidate, not a promoted measurement",
                    stream=e.get("stream"),
                )
                for e in pending
            ],
            coverage={**coverage, "count_available": False, "unverified": len(pending)},
            warnings=["Unverified activity evidence overlaps the window: this is not a count of zero."],
        )
    unknown_who = 0
    if person is not None:  # "how many times did Dana open the fridge": only spans Dana was recognized in
        unknown_who = sum(1 for e in spans if e.get("person") is None)
        spans = [e for e in spans if e.get("person") == person]
    mine = [e for e in spans if matches(e)]
    unnamed = [e for e in spans if owner_name(e) is None and not matches(e)]
    evidence = [
        dict(
            ts=float(e["start"]),
            camera=e.get("camera"),
            source="camera",
            quote=f"{owner_name(e) or 'something not named yet'}: {_n(_occurrences(e), 'time', 'times')} on "
            f"{e.get('camera')}, {_clock(float(e['start']), tz)} to {_clock(float(e['end']), tz)}",
            link="/review#" + urlencode(dict(day=stamp(float(e["start"]))[:10], t=float(e["start"]))),
            method="EventDetected: a detector's promoted span, not a caption",
            stream=e.get("stream"),
            event=e.get("id"),
        )
        for e in [*mine, *unnamed]
    ]
    recs = _n(len(covered), "analyzed recording", "analyzed recordings")
    label = asked.replace("_", " ") if asked in model_kinds else asked
    total = sum(_occurrences(e) for e in mine)
    if mine:
        parts = ", ".join(f"{_occurrences(e)} at {_clock(float(e['start']), tz)}{_by(e, person)}" for e in mine)
        text = f"{label}: {_n(total, 'time', 'times')} ({parts}), from {recs}."
    else:
        text = f"{label}: not found in {recs}."
    if unnamed:
        counts = _and([str(_occurrences(e)) for e in unnamed])
        text += f" Also {_n(len(unnamed), 'span', 'spans')} of something not named yet ({counts} times)."
    if unknown_who:
        text += f" {_n(unknown_who, 'span', 'spans')} in that time couldn't be told whose, and aren't counted."
    dwell = sum(max(0.0, min(float(e["end"]), end) - max(float(e["start"]), start)) for e in mine)
    return dict(answer=text, evidence=evidence, coverage=coverage, warnings=[], count=total, dwell_s=dwell)
