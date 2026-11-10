"""Read-only local historical evidence. Records are untrusted data, never instructions.

Journal entries (memory/journal.py) join transcript and general visual/recap questions as evidence when the
journal has any in the window: the best few by full-text rank, fused with the nearest by meaning while the semantic index is on
(memory/recall.py, with no chat-model call; the semantic index embeds only the question, with a short timeout), placed
ahead of the raw lines so a long list of lines can't push the overview out, and only when there is no recorded answer
already, so a fixed citation like [C1] in one still points at its row. A question with no words to search for (a
spoken follow-up like "where was it?") still gets the latest entries from the last few minutes of its window. A
question that names a camera gets none: entries mix every camera and speech. Without a journal store, or with no
entries in the window, the result is exactly what it was before the journal existed.

A transcript question that names nobody, and no topic or intent, keeps the lines that hold the question's words when
the evidence budget (MAX_EVIDENCE, MAX_CHARS) cuts the window, the best matches first (rare words count most), each
with the lines said within CONTEXT_S of it, then fills what's left with the lines nearest the question time, and shows
them in time order. A question about a busy Review day finds its lines that way (#1303: the day's last minutes used to
fill the budget), and a follow-up with no words to match ("where was it?") still gets the minute before it. Naming
someone, a topic, or an intent still keeps the earliest lines.

With the timeline on (`timeline.enabled`, memory/timeline.py), "who was here" and "when was X last seen" (and "when did
I last see X") answer from encounters: face visits and transcript lines joined per person through the event
log, so someone only heard counts too. The answers cite the visits and lines behind them. When the timeline is off,
or there is no event log yet, they answer from face visits alone, exactly as before. A question that names a camera
also answers from that camera's face visits alone, since encounters mix every camera and speech. An answer from
encounters carries "paths": ["timeline"], which the dashboard shows as a "New" tag linked to the switch.

"Last seen" answers from the timeline (on) or the face worker's visits (off); v1's detector journal and its own path
are gone (#549).

Ask and voice share the temporal parser here. Clock and relative-instant questions read a bounded five-minute
neighborhood, clamped at question time; ranges read their stated interval. An unqualified 1 to 12 clock selects the
most recent eligible half of the chosen local day, and an unqualified range prefers the shortest increasing span,
then the latest start. Explicit AM/PM avoids this inference. Relative parts of the day use question time rather
than the dashboard selection; an unqualified part uses the selected date. "Last night" spans the previous
evening at 18:00 through 05:00; other night phrases retain 22:00 to 05:00. A clock qualified by "last night" selects
the previous evening or the following early morning. Unreadable, conflicting, nonexistent, or
ambiguous local times are refused instead of expanding the evidence search.
"""

import bisect
import calendar
import concurrent.futures
import datetime as dt
import itertools
import json
import math
import os
import pathlib
import re
import sqlite3
import time
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlencode
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from liberational import config as cfg
from liberational import logs, speech_log, vision_log
from liberational.audio import twins
from liberational.domain.kernel import PersonId
from liberational.domain.memory import Modality
from liberational.listen.revisions import NOT_SAID
from liberational.memory import activity_kinds, event_counts, timeline
from liberational.people import registry, transcript_view
from liberational.speech.playback_guard import overlaps
from liberational.vision import views

LOG = logs.get("history-qa")


def local_zone() -> ZoneInfo:
    """time.timezone, else TZ, else the system zone from /etc/localtime, else UTC."""
    name = cfg.get("time.timezone", "") or os.environ.get("TZ", "").lstrip(":") or "/etc/localtime"
    try:
        if name.startswith("/"):  # a zone file: TZ=/usr/share/zoneinfo/X, TZ=:/etc/localtime, or the default
            with open(name, "rb") as f:
                return ZoneInfo.from_file(f, key=pathlib.Path(name).resolve().name)
        return ZoneInfo(name)
    except OSError, ZoneInfoNotFoundError, ValueError:
        LOG.warning(f"can't load time zone {name!r}; using UTC for day boundaries")
        return ZoneInfo("UTC")


TZ = local_zone()
STATE = cfg.path("paths.state")
MAX_EVIDENCE = 100
MAX_CHARS = 18000
JOURNAL_ITEMS = 3  # journal entries added to a transcript or camera question's evidence
JOURNAL_TAIL_S = 180  # a question with no search words still gets entries from this long at the window's end
NO_EVIDENCE = "No matching safe, timestamped evidence in the available history."
CONTEXT_S = 60.0  # lines said this close to a matching line come with it (a reply is often within a minute)
# Words a question is made of rather than about, left out when its words are matched against lines and captions.
QUESTION_WORDS = frozenset(
    [
        "what",
        "when",
        "where",
        "which",
        "were",
        "was",
        "the",
        "and",
        "did",
        "does",
        "how",
        "many",
        "long",
        "today",
        "yesterday",
        "camera",
        "video",
        "people",
        "person",
        "visible",
        "show",
        "tell",
        "told",
        "about",
        "this",
        "that",
        "who",
        "why",
        "you",
        "your",
        "they",
        "them",
        "their",
        "there",
        "say",
        "said",
        "says",
        "talk",
        "talked",
        "now",
        "for",
        "with",
        "have",
        "has",
        "had",
        "are",
        "our",
    ]
)
WORDS = dict(
    zip(
        ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"],
        range(13),
    )
)
CAM = re.compile(r"\bcam(?:era)?\s*(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\b", re.I)


WHO = re.compile(r"\bwho\s+(?:was|were|did\s+(?:i|you)\s+see)\s*(?:here|there|around|in view|seen|in the room)?\b")
# "Who's home", "who is here right now": who was recognized in the last hour, from the presence answer. The cameras
# and microphones see who is around, not who is home, so the answer says what it read and when.
HOME_NOW = re.compile(
    r"\bwho(?:['\u2019]s|\s+is|\s+all\s+is)\s+(?:at\s+)?(?:home|here|around|in(?:\s+the\s+house)?)\b"
    r"(?:\s+(?:right\s+)?now)?\s*\??\s*$"
)
HOME_NOW_AS = "who was here in the last hour"
# "What happened today" with a day named in it: the day's journal and transcript, as "what did we talk about today" reads.
HAPPENED = re.compile(r"\bwhat (?:all )?happened\b")
CLOCK = r"\d{1,2}(?::\d{2})?\s*(?:am|pm)?(?![:\d])"
AT = re.compile(r"\b(at|around|about)\s+(" + CLOCK + r")\b")
CLOCK_RANGE = re.compile(r"\bbetween\s+(" + CLOCK + r")\s+and\s+(" + CLOCK + r")\b")
# "since 1pm", "since 13:00 today", "since 9 this morning", "yesterday since noon": a clock anchor up to now (#2219).
# "since yesterday", "since this morning", "since last night": the start of that day or part of the day up to now (#2228).
_SINCE_DAY = r"(today|yesterday|this morning|this afternoon|this evening|last night)"
SINCE = re.compile(
    r"\b(?:(today|yesterday)\s+)?since\s+(?:("
    + CLOCK
    + r"|noon|midnight)(?:\s*"
    + _SINCE_DAY
    + r")?|"
    + _SINCE_DAY
    + r")\b"
)
AGO = re.compile(
    r"\b(\d+(?:\.\d+)?|an?|one|two|three|four|five|six|seven|eight|nine|ten)\s+(seconds?|minutes?|hours?)\s+ago\b"
)
MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
MONTH_DATE = re.compile(r"\b(" + "|".join(MONTHS) + r")\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?\b")
WHO_WINDOW_S = 300  # "who was here at 3pm": anyone seen within 5 minutes of it
# The parts of a day, as local hours [start, end) of the day they belong to; night runs past midnight to 05:00 the next
# morning. Defined once, so "in the evening" means the same hours wherever a question is read (#195). A part resolves
# against the selected day, else a date or "yesterday" in the question, else today; "last night" is the night before.
LAST_NIGHT_HOURS = (18, 29)  # previous evening through 05:00 on the question day, shared by Ask and voice
PARTS_OF_DAY = {"morning": (5, 12), "afternoon": (12, 17), "evening": (17, 22), "night": (22, 29)}
PART = re.compile(r"\b(?:last night|tonight|(?:this|in the|during the|the)\s+(?:morning|afternoon|evening|night))\b")
# "when did I last see Dana (today)": parsed only with the timeline on, which answers it from encounters. The name
# stops before any time word parse() checks for, so "Dana this morning" leaves "this morning" for that check, and
# before "on", "in", or "at", so "Dana at the office" asks about Dana. The detector path stops it the same way.
TIME_WORDS = (
    r"today|yesterday|this|last|past|previous|ago|since|before|after|earlier|tonight|morning|afternoon|evening|week"
    r"|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday"
)
# A part of the day ("during the night", "the afternoon") ends a name too, so it stays a time (Codex on #219).
NAME_STOP = (
    r"\s+(?:(?:"
    + TIME_WORDS
    + r")\b|"
    + PART.pattern
    + "|"
    + AGO.pattern
    + "|"
    + AT.pattern
    + "|"
    + CLOCK_RANGE.pattern
    + "|"
    + MONTH_DATE.pattern
    + r"|on\s|in\s|at\s|\d{4}-\d{2}-\d{2})"
)
# A time word left over once parse() has read the windows it supports is refused only when it is part of a time
# expression parse() can't read ("last month", "since Monday", "two weeks ago", "before 5pm"). The same words used
# ordinarily ("the week of work", "since its last oil change", "the previous owner") leave the question's day or window
# as it is (#1299). TIME_MARK stands in for a phrase parse() already read, so "since yesterday" still refuses.
TIME_MARK = "\x00"
_UNIT = r"(?:seconds?|minutes?|hours?|days?|nights?|weeks?|weekends?|fortnights?|months?|years?)"
_WEEKDAY = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?"
_COUNT = r"(?:\d+(?:\.\d+)?|" + "|".join(WORDS) + r"|few|couple(?:\s+of)?|several|many)"
UNSUPPORTED_TIME = re.compile(
    r"\bago\b|\btonight\b|\bearlier\b|\b(?:morning|afternoon|evening)\b|\b" + _WEEKDAY + r"\b|\bat\s+\d"
    r"|\b(?:last|past|previous|this)\s+(?:" + _COUNT + r"\s+)?" + _UNIT + r"\b"
    r"|\b" + _COUNT + r"\s+" + _UNIT + r"\s+(?:ago|back|before|after|earlier)\b"
    r"|\b(?:since|before|after|until)\s+(?:" + TIME_MARK + r"|\d"
    r"|noon|midnight|lunch|dinner|breakfast|then|now)"
)
# Speakers who are nobody in particular: "what did someone ask" reads every line, the way "what did we say" does.
ANYONE = frozenset(
    ["we", "us", "you", "they", "them", "someone", "somebody", "anyone", "anybody", "everyone", "everybody"]
)
# Passive-voice and filler words around a "what ... said" capture ("What was said", "what else was said", "What was
# Dana told"): never a speaker's name (#2217).
NOT_A_SPEAKER = frozenset(["was", "were", "is", "are", "be", "been", "being", "has", "have", "had", "else"])
LAST_MET = re.compile(r"\bwhen\s+did\s+(?:i|you|we)\s+last\s+see\s+(.+?)(?=" + NAME_STOP + r"|\s*[?.!]?\s*$)", re.I)


def _clock_phrase(question: str) -> re.Match[str] | None:
    match = AT.search(question)
    # A bare number after 'about' is usually a topic/quantity, except in a who-time question.
    if match and match[1] == "about" and not WHO.search(question) and not re.search(r"[:apm]", match[2]):
        return None
    return match


JOURNAL_SOURCE = re.compile(
    r"\b(?:(?:the|my|our)\s+)?journal(?:\s+entries)?\s+"
    r"(?:describ(?:e[ds]?|ing)|mention(?:ed|s)?|sa(?:y[s]?|id)|record(?:ed|s)?|report(?:ed|s)?|show(?:ed|s)?|not(?:e[ds]?|ing)|contain(?:ed|s)?)\b"
    r"|\b(?:in|from|according\s+to)\s+(?:(?:the|my|our)\s+)?journal(?:\s+entries)?\b"
    r"|\b(?:summari[sz]e|recap|search|read|review)\s+(?:(?:the|my|our)\s+)?journal(?:\s+entries)?\b",
    re.I,
)


def _journal_source(text: str) -> bool:
    match = JOURNAL_SOURCE.search(text)
    if match is None:
        return False
    prefix = text[: match.start()]
    return not (SPEECH.search(prefix) and re.search(r"\babout\b", prefix, re.I))


def is_history_question(question: str) -> bool:
    try:
        cameras, remaining = views.configured_camera_mentions(question)
    except ValueError:
        return True  # collect/parse returns the actionable camera-selection error, never general-model fallback.
    q = remaining.lower()
    temporal = (
        re.search(r"\b(yesterday|today|last|past|previous|ago|since|when|history|earlier)\b|\d{4}-\d{2}-\d{2}", q)
        or PART.search(q)
        or _clock_phrase(q)
        or CLOCK_RANGE.search(q)
        or MONTH_DATE.search(q)
        or UNSUPPORTED_TIME.search(q)
    )
    subject = re.search(
        r"\b(cam(?:era)?s?\s*\d*|lights?|said|say|wanted|discuss|discussed|conversation|transcript|talk|talked|mentioned|work on|seen|see)\b",
        q,
    )
    return bool(
        _journal_source(q)
        or (temporal and (subject or cameras or SCENE.search(q)))
        or re.search(
            r"how many times.*lights?|how long.*lights?|\bvideo\b|walk(?:ed)? (?:by|past)|get done|\b(?:this|last) week\b",
            q,
        )
        or WHO.search(q)
        or (
            is_direction_question(q)
            and re.search(r"\b(?:did|was|were|walked|crossed|passed|moved|went|headed|observed|recorded)\b", q)
        )
        or is_entry_count(q)
        or event_counts.retired_question(question)
        or event_counts.asks(q)  # "how many times did I open the fridge today" (#290, #518)
        or _asks_about_animal(question)  # "what has the cat been doing", "where is May"
        or HOME_NOW.search(q)
        or (HAPPENED.search(q) and temporal)
        or _asks_about_device(question)  # "is Sam home", "where are my keys": collect() answers from devices (#1329)
        or _asks_if_home(question)  # "is Carly home" with no linked device: camera and voice presence (#1596)
    )


FIRST_PERSON = re.compile(r"\b(?:I|me|my|myself|I'm|I've)\b")


def _owner_if_first_person(question: str, state: pathlib.Path) -> Any:
    """The owner's PersonId when a count asks about "I" or "me" ("how many times do I get up at night"), so only spans
    the owner was recognized in count; None when it doesn't, or when the owner isn't set up."""
    if not FIRST_PERSON.search(question):
        return None
    try:
        from liberational.people import registry

        return registry.at(state).owner()
    except Exception:
        return None


def _asks_if_home(question: str) -> bool:
    """Whether the question is "is X home" about someone the registry knows, for routing with no state of its own."""
    from liberational.memory import device_answers

    m = device_answers.HOME.search(question)
    try:
        return m is not None and timeline.person(device_answers._who(m["who"]), cfg.path("paths.state")) is not None
    except Exception:
        return False


def _asks_about_device(question: str) -> bool:
    """Whether an owner link names who or what the question asks about, for routing that has no state of its own
    (voice): the same check collect() makes before answering from linked devices."""
    from liberational.memory import device_answers

    try:
        return device_answers.asked(question, cfg.path("paths.state")) is not None
    except Exception:
        return False


def _asks_about_animal(question: str) -> bool:
    """Whether the question asks what an animal has been doing or when it was last seen, for routing that has no
    state of its own (voice): the configured state's pets and zones."""
    from liberational.memory import subject_activity

    try:
        state = cfg.path("paths.state")
        return (
            subject_activity.asked(question, state) is not None
            or subject_activity.last_asked(question, state) is not None
        )
    except Exception:
        return False


# "How many times did someone walk into frame", "how many people came in", "how often did anyone enter the room": a
# count of people coming into view. It needs a count word, a coming-in verb, and a person or the view as the subject,
# so "how many times did I enter my password" stays a transcript question.
ENTRY_COUNT = re.compile(r"\b(?:how many|how often|count|number of)\b", re.I)
ENTRY_VERB = re.compile(
    r"\bwalk(?:s|ed|ing)?\s+(?:in|into|by|past|through)\b|\b(?:came|come|comes|coming)\s+in(?:to)?\b"
    r"|\benter(?:s|ed|ing)?\b|\bpass(?:es|ed|ing)?\s+(?:by|through)\b",
    re.I,
)
ENTRY_SUBJECT = re.compile(
    r"\b(?:people|persons?|someone|somebody|anyone|anybody|visitors?|guests?)\b"
    r"|\binto\s+(?:the\s+)?(?:frame|view|shot)\b",
    re.I,
)
# Questions about talking are transcript questions even when they say "how many people".
SPEECH = re.compile(
    r"\b(?:talk(?:s|ed|ing)?|say(?:s|ing)?|said|speak(?:s|ing)?|spoken?|call(?:s|ed|ing)?|discuss(?:es|ed|ing)?)\b",
    re.I,
)


def is_entry_count(question: str) -> bool:
    """A count of people coming into view or walking by, however it is phrased."""
    return bool(
        ENTRY_COUNT.search(question)
        and ENTRY_VERB.search(question)
        and ENTRY_SUBJECT.search(question)
        and not SPEECH.search(question)
    )


# Asked as walking by, passing, or crossing, rather than as coming into view: how a walk-by answer words itself.
WALKED_BY = re.compile(r"\b(?:walk\w*|pass\w*|went|go(?:ne|es|ing)?) (?:by|past)\b|\bcross\w*", re.I)
CROSSING = re.compile(
    r"how many\b.*\b(?:people|persons?|someone|anyone|walk\w*|pass\w*|cross\w*|went|go(?:ne)?|came|come)\b"
    r"|\b(?:count|number of|how often)\b.*\b(?:people|persons?|walk\w*|pass\w*|cross\w*)\b"
    r"|walk(?:ed|ing)? (?:by|past)|\bcross(?:ed|ings?)\b|\bpass(?:ed|ing)? (?:by|through)\b",
    re.I,
)


SCENE = re.compile(
    r"\b(?:scene|visible|appearance|wearing|wore|seated|sitting|standing|hanging|colou?rs?)\b"
    r"|\blooks? like\b|\blooked like\b|\b(?:on|under|beside|behind|near|next to)\s+(?:the|a)\b",
    re.I,
)
DIRECTION = re.compile(
    r"\b(?:which|what)\s+way\b|\bdirections?\b"
    r"|\b(?:left|right|top|bottom)[ -]+to[ -]+(?:left|right|top|bottom)\b"
    r"|\b(?:toward|towards|from)\s+(?:the\s+)?(?:left|right|top|bottom)\b"
    r"|\bside[ _-]+[ab][ _-]+(?:to|towards?)[ _-]+(?:line[ _-]+)?(?:side[ _-]+)?[ab]\b",
    re.I,
)


def is_direction_question(question: str) -> bool:
    """Movement direction, distinct from passage totals and conversational uses of direction words."""
    return bool(
        DIRECTION.search(question)
        and re.search(
            r"\b(?:walk(?:s|ed|ing)?|cross(?:es|ed|ing|ings)?|pass(?:es|ed|ing)?|mov(?:e[ds]?|ing|ement)|go(?:ing|ne)?|went|head(?:ed|ing)?)\b",
            question,
            re.I,
        )
        and not re.search(r"\b(?:plan(?:s|ned|ning)?|should|intend(?:s|ed|ing)?)\b", question, re.I)
        and not SPEECH.search(question)
    )


def is_crossing_question(question: str, query: Mapping[str, Any]) -> bool:
    """A walk-by / passage count question on recordings (not lights, not a plain scene description)."""
    return query.get("kind") in ("video", "mixed") and bool(
        CROSSING.search(question) or is_entry_count(question) or is_direction_question(question)
    )


def crossing_paths(state: pathlib.Path, query: Mapping[str, Any], warnings: list[str]) -> dict[str, Any]:
    """The walk-by count for a walk-by question: see-actions' walkby.span rows (memory/crossing_counts.py), the only
    source. Where nothing watched, the answer says the count is unknown, never zero. It never claims a unique-person
    count, and cameras aren't summed."""
    from liberational.memory import crossing_counts

    out: dict[str, Any] = dict(path_b=None, answer=None)
    try:
        result = crossing_counts.from_spans(state, query["start"], query["end"], query["cameras"])
        out["path_b"] = dict(status="done", result=result, answer=crossing_counts.answer_text(result))
        out["answer"] = out["path_b"]["answer"]
    except (OSError, ValueError, vision_log.continuum_py.ContinuumError) as exc:
        reason = str(exc) or type(exc).__name__
        out["path_b"] = dict(status="unavailable", reason=reason)
        out["answer"] = f"How many people walked past is unknown: the walk-by count couldn't be read ({reason})."
        warnings.append("Walk-by count unavailable (" + type(exc).__name__ + ").")
    return out


def stamp(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, TZ).isoformat(timespec="seconds")


def _clock_choices(token: str, base: dt.datetime, *, night: bool = False) -> list[dt.datetime]:
    """Read a local clock; an unqualified 1 to 12 offers both halves of the day."""
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", token)
    if m is None:
        raise ValueError("Use a clock time like 3pm or 15:30.")
    hour, minute, ap = int(m[1]), int(m[2] or 0), m[3]
    if minute >= 60 or hour >= 24 or (ap and not 1 <= hour <= 12):
        raise ValueError("Use a clock time like 3pm or 15:30.")
    hours = [hour % 12 + (12 if ap == "pm" else 0)] if ap else [hour]
    if not ap and 1 <= hour <= 12:
        hours = [hour % 12, hour % 12 + 12]
    result = [base.replace(hour=h, minute=minute, second=0, microsecond=0) for h in hours]
    if night:
        result = [
            value + dt.timedelta(days=1) if value.hour <= 5 else value
            for value in result
            if value.hour <= 5 or value.hour >= LAST_NIGHT_HOURS[0]
        ]
    # A nonexistent or ambiguous local clock cannot safely identify a historical window.
    for value in result:
        if dt.datetime.fromtimestamp(value.timestamp(), TZ).replace(tzinfo=None) != value.replace(tzinfo=None):
            raise ValueError("That local clock time does not exist because of a clock change.")
        if value.replace(fold=0).utcoffset() != value.replace(fold=1).utcoffset():
            raise ValueError("That local clock time is ambiguous because of a clock change; specify another time.")
    return result


def _since_window(
    match: re.Match[str], now: float, day: dt.datetime, selected_day: str | dt.date | None
) -> tuple[float, float, str]:
    """The window a SINCE match names: from the most recent past reading of its clock, or the start of the day or
    part of the day it names, to now.

    The clock is on today unless the phrase names yesterday or last night; with no day word and a selected day, it
    is on that day and the window ends with it. A part of the day picks the clock's half: "since 9 this morning"
    is 09:00, "since 3 this afternoon" is 15:00. An unqualified 1 to 12 reads as the latest of its two halves that
    has already passed. A named day or part of the day ("since yesterday", "since last night") starts where
    PARTS_OF_DAY and LAST_NIGHT_HOURS say and is always relative to today, whatever day is selected.
    """
    if match[4]:
        if match[1]:
            raise ValueError("Use a single time window; combined time expressions are not supported.")
        part = match[4].split()[-1]
        hour = (
            0 if part in ("today", "yesterday") else LAST_NIGHT_HOURS[0] if part == "night" else PARTS_OF_DAY[part][0]
        )
        begin = (day - dt.timedelta(days=match[4] in ("yesterday", "last night"))).replace(hour=hour)
        if begin.timestamp() > now:
            raise ValueError(f"{match[4].capitalize()} hasn't started yet, so there's no history for it.")
        return begin.timestamp(), now, f"since {_clock(begin.timestamp())} {'yesterday' if begin < day else 'today'}"
    day_word = match[1] or match[3]
    clock = match[2].strip()
    token = {"noon": "12pm", "midnight": "12am"}.get(clock, clock)
    base, end = day, now
    if day_word in ("yesterday", "last night"):
        base = day - dt.timedelta(days=1)
    elif day_word is None and selected_day:
        base = dt.datetime.combine(dt.date.fromisoformat(str(selected_day)), dt.time(), tzinfo=TZ)
        end = min(now, (base + dt.timedelta(days=1)).timestamp())
    choices = _clock_choices(token, base, night=day_word == "last night")
    if day_word == "this morning":
        choices = [c for c in choices if c.hour < 12]
    elif day_word in ("this afternoon", "this evening"):
        choices = [c for c in choices if c.hour >= 12]
    if not choices:
        raise ValueError(f"{clock} isn't {day_word}; use a clock time like 3pm or 15:30.")
    past = [c.timestamp() for c in choices if c.timestamp() <= now]
    if not past:
        raise ValueError("That clock time is in the future.")
    start = max(past)
    date = dt.datetime.fromtimestamp(start, TZ).date()
    on = "today" if date == day.date() else "yesterday" if date == (day - dt.timedelta(days=1)).date() else f"on {date}"
    return start, end, f"since {_clock(start)} {on}"


def _natural_dates(question: str, day: dt.datetime) -> str:
    """Normalize spelled-out dates, leaving selection and range validation to parse."""

    def replace(match: re.Match[str]) -> str:
        try:
            return dt.date(int(match[3] or day.year), MONTHS[match[1]], int(match[2])).isoformat()
        except ValueError:
            raise ValueError("Specify a valid calendar date.") from None

    return MONTH_DATE.sub(replace, question)


def parse(question: str, now: float, selected_day: str | dt.date | None = None) -> dict[str, Any]:
    met = LAST_MET.search(question) if timeline.enabled() else None
    configured_cameras, remaining = views.configured_camera_mentions(question)
    q = remaining.lower()
    journal_only = _journal_source(q)
    if met:  # the phrase names a person, not a time: it must not read as "last <unit>" below
        q = q[: met.start()] + " last seen " + q[met.end() :]
    q = HOME_NOW.sub(HOME_NOW_AS, q)
    day = dt.datetime.fromtimestamp(now, TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    start, end, window = 0.0, now, "all available retained history"
    q = _natural_dates(q, day)
    q = re.sub(r"\byesterday\s+(morning|afternoon|evening|night)\b", r"yesterday in the \1", q)
    q = re.sub(r"\bearlier\s+today\b", "today", q)
    q = re.sub(r"(\d)\s*([ap])\.\s?m\b\.?", r"\1\2m", q)  # "1 p.m." as speech recognition writes it
    since = SINCE.search(q)
    if since:
        q = q[: since.start()] + f" {TIME_MARK} " + q[since.end() :]
    ago = AGO.search(q)
    if ago:
        q = q[: ago.start()] + f" {TIME_MARK} " + q[ago.end() :]
    clock_range = CLOCK_RANGE.search(q)
    clock = _clock_phrase(q) if not clock_range else None
    dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", q)
    duration = re.search(
        r"\b(?:last|past|previous)\s+(?:(\d+(?:\.\d+)?|[a-z]+)\s+)?(seconds?|minutes?|hours?|days?)\b", q
    )
    week = re.search(r"\b(this|last) week\b", q)
    if ago and (
        dates
        or _clock_phrase(q)
        or CLOCK_RANGE.search(q)
        or PART.search(q)
        or re.search(r"\b(?:today|yesterday|past)\b|\blast\s+(?!seen\b|see\b)", q)
    ):
        raise ValueError("Use a single time window; combined time expressions are not supported.")
    if sum((bool(dates), bool(duration), bool(week), "yesterday" in q, bool(re.search(r"\btoday\b", q)))) > 1:
        raise ValueError("Use a single time window; combined time expressions are not supported.")
    if since and (
        ago
        or dates
        or duration
        or week
        or clock
        or clock_range
        or PART.search(q)
        or re.search(r"\b(?:today|yesterday)\b", q)
    ):
        raise ValueError("Use a single time window; combined time expressions are not supported.")
    if dates:
        if len(dates) != 1:
            raise ValueError("Use one explicit YYYY-MM-DD date, not a date range.")
        d = dt.datetime.fromisoformat(dates[0]).replace(tzinfo=TZ)
        start, end, window = d.timestamp(), (d + dt.timedelta(days=1)).timestamp(), dates[0]
    elif "yesterday" in q:
        start, end, window = (day - dt.timedelta(days=1)).timestamp(), day.timestamp(), "yesterday"
    elif re.search(r"\btoday\b", q):
        start, window = day.timestamp(), "today"
    elif week:
        monday = day - dt.timedelta(days=day.weekday())
        d = monday - dt.timedelta(days=7 if week[1] == "last" else 0)
        start, end, window = d.timestamp(), (d + dt.timedelta(days=7)).timestamp(), week[0]
    elif duration:
        token = duration[1] or "one"
        n = WORDS.get(token, 1 if token in ("a", "an") else None)
        if n is None:
            try:
                n = float(token)
            except ValueError:
                raise ValueError("Specify a numeric history duration.")
        if not 0 < n <= 100000:
            raise ValueError("History duration is out of range.")
        start = now - n * {"s": 1, "m": 60, "h": 3600, "d": 86400}[duration[2][0]]
        window = duration[0]
    if selected_day and not (dates or duration or week or "yesterday" in q or re.search(r"\btoday\b", q)):
        d = dt.datetime.combine(dt.date.fromisoformat(str(selected_day)), dt.time(), tzinfo=TZ)
        start, end, window = d.timestamp(), (d + dt.timedelta(days=1)).timestamp(), str(selected_day)
    if since:
        start, end, window = _since_window(since, now, day, selected_day)
    if part := PART.search(q):
        if duration or week or clock_range or (clock and part[0] != "last night"):
            raise ValueError("Use a single time window; combined time expressions are not supported.")
        name = "night" if part[0] in ("tonight", "last night") else part[0].split()[-1]
        relative_part = part[0] in ("last night", "tonight") or part[0].startswith("this ")
        if relative_part and (dates or "yesterday" in q):
            raise ValueError("Use a single time window; combined time expressions are not supported.")
        explicit = not relative_part and (dates or "yesterday" in q or selected_day)
        base = dt.datetime.fromtimestamp(start, TZ) if explicit else day
        if part[0] == "last night":
            base -= dt.timedelta(days=1)
        first, last = LAST_NIGHT_HOURS if part[0] == "last night" else PARTS_OF_DAY[name]
        part_start = base.replace(hour=first)
        part_end = (base + dt.timedelta(days=last // 24)).replace(hour=last % 24)
        if part_start.timestamp() > now:
            raise ValueError(
                f"The {name} of {base.date().isoformat()} hasn't started yet, so there's no history for it."
            )
        start, end, window = part_start.timestamp(), part_end.timestamp(), f"the {name} of {base.date().isoformat()}"
        q = q.replace(part[0], f" {TIME_MARK} ")
    at_ts = None
    who = bool(WHO.search(q))
    if ago:
        token = ago[1]
        n = WORDS.get(token, 1 if token in ("a", "an") else None)
        amount = float(token) if n is None else float(n)
        if not 0 < amount <= 100000:
            raise ValueError("History duration is out of range.")
        center = now - amount * {"s": 1, "m": 60, "h": 3600}[ago[2][0]]
        start, end = center - WHO_WINDOW_S, center + WHO_WINDOW_S
        window = "around " + _clock(center)
        at_ts = center if WHO.search(q) else None
    elif clock or clock_range:
        if duration or week:
            raise ValueError("Use a single time window; combined time expressions are not supported.")
        explicit_day = bool(dates or "yesterday" in q or selected_day or part)
        base = dt.datetime.fromtimestamp(start, TZ) if explicit_day else day
        match = clock_range or clock
        assert match is not None
        if clock_range:
            pairs = [
                (a, b)
                for a in _clock_choices(clock_range[1], base)
                for b in _clock_choices(clock_range[2], base)
                if a < b
            ]
            # Prefer a same-half-day range, then the most recent range that has started.
            pairs = [(a, b) for a, b in pairs if a.timestamp() <= now]
            if not pairs:
                raise ValueError("That time window is in the future or has invalid clock times.")
            a, b = min(pairs, key=lambda pair: ((pair[1] - pair[0]).total_seconds(), -pair[0].timestamp()))
            start, end = a.timestamp(), b.timestamp()
            window = f"between {_clock(start)} and {_clock(end)}"
        else:
            assert clock is not None
            choices = _clock_choices(clock[2], base, night=bool(part and part[0] == "last night"))
            choices = [c for c in choices if c.timestamp() <= now]
            if not choices:
                raise ValueError("That clock time is in the future.")
            center = max(c.timestamp() for c in choices)
            start, end = center - WHO_WINDOW_S, center + WHO_WINDOW_S
            window = "around " + _clock(center)
            at_ts = center if WHO.search(q) else None
        q = q.replace(match[0], f" {TIME_MARK} ")
    if re.search(r"\b(?:at|around|between)\s+\d", q) or (
        WHO.search(q) and re.search(r"\b(?:around|about|between)\s+(?!today\b|yesterday\b|last\b|this\b)\w+", q)
    ):
        raise ValueError("Unsupported time phrase; specify a clock time or a complete time range.")
    residual = q
    for phrase in dates + [x[0] for x in (week, duration) if x] + ["yesterday", "today"]:
        residual = residual.replace(phrase, f" {TIME_MARK} ")
    residual = re.sub(r"walk(?:ed)? past", "walked by", residual)
    residual = re.sub(r"\blast\s+(?:seen|on|appear(?:ed)?|time)\b", "", residual)
    leftover_clock = re.search(r"\b\d{1,2}:\d{2}\b|\b\d{1,2}\s*(?:am|pm)\b", residual)
    incomplete_range = clock_range and re.search(r"\band\s+\d", residual)
    if UNSUPPORTED_TIME.search(residual) or leftover_clock or incomplete_range:
        raise ValueError(
            "Unsupported time phrase; use today, yesterday, this/last week, last N hours/minutes/days, YYYY-MM-DD, "
            "or a part of the day (this morning, in the afternoon, this evening, tonight, last night)."
        )
    cameras = sorted(
        set(configured_cameras) | {"cam" + str(int(WORDS.get(x.lower(), x))) for x in CAM.findall(remaining)}
    )
    stripped = CAM.sub("", q)
    if re.search(r"\bcam(?:era)?\s+\w+", stripped) and not re.search(r"\bcameras\b", stripped):
        raise ValueError("Specify the actual camera ID, for example camera3; IDs are not renumbered.")
    kind = (
        "who"
        if who
        else "lights"
        if re.search(r"\blights?\b", q)
        else "camera"
        if cameras or CAM.search(q) or re.search(r"\bcameras?\b|last seen", q)
        else "transcript"
    )
    if journal_only:
        if cameras:
            raise ValueError("Journal summaries mix cameras; ask about the journal or a specific camera separately.")
        kind = "journal"
    person = None
    m = re.search(r"\bwhen\s+(?:was\s+)?(.+?)\s+last\b|\b(?:last seen|last see)\s+([\w-]+)", question, re.I)
    if m:
        person = (m[1] or m[2]).strip()
    last_seen = bool(re.search(r"\blast (?:see|seen)\b", q))
    if last_seen:
        direct = re.search(
            r"\blast see\s+(.+?)(?="
            + NAME_STOP
            + r"|\s+(?:over\s|during\b|the\s+(?:last|past)\b|cam(?:era)?\s*\d+\b)|[?!.]|$)",
            question,
            re.I,
        )
        if direct:
            person = direct[1].strip()
        kind = "camera"
    if met:
        person = met[1].strip()
    if kind == "transcript":
        m = re.search(
            r"\b(?:what(?:\s+all)?|everything(?:\s+that)?|all(?:\s+that)?)(?:(?:\s+[\w'-]+){0,2}?\s+(?:did|does|has|had))?\s+([\w'-]+(?:\s+[\w'-]+){0,3}?)\s+(?:tell|told|wanted|want|said|say|discussed|discuss|mentioned|mention|talked|talk|asked|ask|suggested|suggest)\b",
            question,
            re.I,
        )
        if m:
            words = m[1].split()
            while words and words[0].casefold() in NOT_A_SPEAKER:
                words.pop(0)
            while words and words[-1].casefold() in NOT_A_SPEAKER:
                words.pop()
            person = " ".join(words) or None
    participants: list[str] = []
    if kind == "transcript" and person:
        participants = [p.strip() for p in re.split(r"\s+and\s+", person, flags=re.I)]
        participants = [cfg.owner_or() if p.casefold() in ("i", "me") else p for p in participants]
        if any(p.casefold() in ANYONE for p in participants):
            participants = []
        person = participants[0] if len(participants) == 1 else None
    intent = bool(re.search(r"\b(?:want(?:ed)?|get done|work on|plan(?:ned)?|to do|should|need(?:ed)?)\b", q))
    if re.search(r"\blights?\b", q) and re.search(r"how long|duration|time.*on", q):
        kind = "light_duration"
    elif (
        re.search(r"\bvideo\b|walk(?:ed)? (?:by|past)|cross(?:ed|ings)?", q)
        or is_entry_count(q)
        or is_direction_question(q)
    ):
        kind = "video"
    if kind == "transcript" and not intent:
        if re.search(r"\bwhat (?:all )?happened\b|\b(?:recap|summarize)\b", q):
            kind = "mixed"
        elif not SPEECH.search(q) and (
            SCENE.search(q)
            or re.search(
                r"\b(?:people|person|someone|anyone)\b.*\b(?:doing|did|came|come|pass|passed|walk|walking|wear)\b", q
            )
        ):
            kind = "video"
    if journal_only:
        kind = "journal"
    if start > now:  # the end is clamped to now below, so a future window would read as one with no evidence
        raise ValueError("That date is in the future, so there's no history for it yet.")
    return dict(
        question=question,
        participants=participants,
        topics=kind == "transcript" and bool(re.search(r"\bmost\b|\btopics?\b|main (?:theme|subject)", q)),
        intent=intent,
        last_seen=last_seen,
        start=start,
        end=min(end, now),
        timezone=str(TZ),
        window=window,
        cameras=cameras,
        kind=kind,
        person=person,
        at=at_ts,
        **(
            {"search_text": " ".join(JOURNAL_SOURCE.sub(" ", residual).replace(TIME_MARK, " ").split())}
            if journal_only
            else {}
        ),
    )


def people_named(question: str) -> list[str]:
    """The people `parse` finds in a question ("when was Dana last seen", "what did Dana and Eve say"), or none when it
    can't parse it. For the forgotten-person check (people/forgotten.py), which runs before collect."""
    try:
        query = parse(question, time.time())
    except ValueError:
        return []
    return [p for p in (query["person"], *query["participants"]) if p]


def _json(path: pathlib.Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except OSError, ValueError:
        return default


def _has_text(text: Any) -> bool:
    """Whether text read from a record (a JSON line, a camera row, a cached caption) has anything to show. Nothing
    is hidden for what it says: lines mentioning passwords, codes, or keys reach history answers too (Joe's
    decision, #92)."""
    return isinstance(text, str) and bool(text.strip())


def bot_labels() -> tuple[str, ...]:
    return (cfg.assistant_name().casefold(), "bot", "assistant", "system")


def calls_dir(state: pathlib.Path) -> pathlib.Path:
    """Configured calls/ for the configured state dir; a sibling calls/ for any other (tests)."""
    return cfg.path("paths.calls") if state == cfg.path("paths.state").resolve() else state.parent / "calls"


def _recorded_visits(state: pathlib.Path, query: Mapping[str, Any]) -> list[dict[str, Any]]:
    """A person's visits from recorded video, newest first, on the cameras the question named (any camera when it
    named none). Falls through to camera history when nothing matches.

    A visit is the person's when the name on it belongs to them now (a rematch names visits by the face gallery now,
    as timeline._visits reads them), so a renamed person keeps the visits named before the next rematch. Each one is
    shown under the person's current name. Without a log, or for a name the registry does not know, names are
    compared, as before."""
    from liberational.vision import sightings

    view = transcript_view.View(state)
    target = view.person_at(query["person"])
    if target is None:
        visits = sightings.visits(query["person"], query["start"], query["end"], state)
    else:
        owner: dict[str, bool] = {}
        visits = []
        for v in sightings.visits_between(query["start"], query["end"], state):
            if (name := str(v["name"])) not in owner:
                owner[name] = (p := view.person_at(name)) is not None and p.id == target.id
            if owner[name]:
                visits.append({**v, "name": target.display_name})
        visits.sort(key=lambda v: v["end"], reverse=True)
    return [v for v in visits if not query["cameras"] or v["cam"] in query["cameras"]]


def _camera(
    state: pathlib.Path, query: Mapping[str, Any], warnings: list[str]
) -> tuple[list[dict[str, Any]], list[str], dict[str, dict[str, Any]]]:
    records = vision_log.at(state).captions(query["start"], query["end"])
    rows = [_camera_row(r) for r in records if query["start"] <= r["ts"] < query["end"]]
    rows.sort(key=lambda r: (r["ts"], r["cam"], r["yaw"]))
    available = sorted({r["cam"] for r in rows})
    excluded: dict[str, int] = {}
    for r in rows:
        if r["observation"]["status"] in ("not_observed", "error"):
            excluded[r["cam"]] = excluded.get(r["cam"], 0) + 1
    skipped = sum(excluded.values())
    rows = [r for r in rows if r["observation"]["status"] not in ("not_observed", "error")]
    if skipped:
        warnings.append(f"{skipped} gate-skipped/error rows excluded: these are not fresh visual observations.")
    if any(r["observation"]["status"] == "legacy_unknown" for r in rows):
        warnings.append(
            "Legacy captions lack freshness provenance and may reuse earlier descriptions; not proof of continuous presence or absence."
        )
    wanted = query["cameras"] or available
    rows = [r for r in rows if r["cam"] in wanted]
    coverage: dict[str, dict[str, Any]] = {}
    for cam in wanted:
        rr = [r for r in rows if r["cam"] == cam]
        gaps = [b["ts"] - a["ts"] for a, b in itertools.pairwise(rr)]
        coverage[cam] = dict(
            count=len(rr),
            first=rr[0]["ts"] if rr else None,
            last=rr[-1]["ts"] if rr else None,
            missing_camera=cam not in available,
            identity_available=sum(r["faces"] is not None for r in rr),
            max_gap_seconds=max(gaps, default=0),
            excluded_rows=excluded.get(cam, 0),
            partial=not rr
            or excluded.get(cam, 0) > 0
            or rr[0]["ts"] > query["start"] + 30
            or rr[-1]["ts"] < query["end"] - 30
            or max(gaps, default=0) > 30,
        )
    return rows, available, coverage


def _review_link(ts: float, camera: str | None = None) -> str:
    """Seek Review to the source's local day and exact time, with a known camera angle if any."""
    place: dict[str, str | float] = dict(day=stamp(ts)[:10], t=ts)
    if camera:
        place["cam"] = camera
    return "/review#" + urlencode(place)


def _provenance(record: Mapping[str, Any]) -> dict[str, Any]:
    """Keep source identity independently of the citation's later C/T display label."""
    ids = timeline.payload_ids(record)
    if len(ids) == 1:
        return {"payload_id": ids[0]}
    return {"payload_ids": list(ids)} if ids else {}


def _camera_row(record: Mapping[str, Any]) -> dict[str, Any]:
    """A caption record as a camera row: brightness as the 0..1 proxy (mean / 255), observation {"status":
    "legacy_unknown"} when it has none, and `unchanged` True for a repeat of an unchanged view's caption."""
    light = record.get("brightness")
    mean = light.get("mean") if isinstance(light, dict) else None
    meta = record.get("observation")
    return dict(
        **_provenance(record),
        ts=record["ts"],
        cam=record["cam"],
        yaw=record["yaw"],
        caption=record.get("caption"),
        unchanged=bool(record.get("unchanged")),
        faces=record.get("faces"),
        brightness=mean / 255 if isinstance(mean, (int, float)) else None,
        observation=meta if isinstance(meta, dict) and meta else {"status": "legacy_unknown"},
    )


def camera_evidence(r: Mapping[str, Any]) -> dict[str, Any]:
    """A camera row (`_camera`) as an evidence item."""
    return dict(
        **_provenance(r),
        ts=r["ts"],
        camera=r["cam"],
        quote=r["caption"],
        source="camera",
        link=_review_link(r["ts"], r["cam"]),
        **{k: r[k] for k in ("identity", "method", "brightness", "observation") if k in r},
    )


def labeled(r: Mapping[str, Any], n: int) -> dict[str, Any]:
    """An evidence item with its citation id n: C for camera, J for journal, D for a device sighting, T for speech."""
    prefix = {"camera": "C", "journal": "J", "device": "D"}.get(r["source"], "T")
    out = dict(r, label=prefix + str(n), timestamp=stamp(r["ts"]))
    out.update(id=out["label"], text=out["quote"], href=out["link"])
    return out


def _journal(
    question: str, query: Mapping[str, Any], state: pathlib.Path, result: dict[str, Any], warnings: list[str]
) -> list[dict[str, Any]]:
    """A bounded visual/recap overview plus ranked journal hits, or ordinary ranked/latest journal context.

    One early, fully-contained summary anchors a visual/recap window without relying on index candidates.
    Ranked hits fill the unchanged item budget; named-camera and recorded-answer exclusions live at the caller.
    """
    from liberational.memory import recall

    text = str(query.get("search_text", question))
    limit = MAX_EVIDENCE if query["kind"] == "journal" else JOURNAL_ITEMS
    try:
        if recall.searchable(text, state):
            items = recall.search(text, query["start"], query["end"], query["person"], state, rank=False, limit=limit)
        else:
            end = float(query["end"]) if math.isfinite(query["end"]) else time.time() + 1.0
            start = float(query["start"])
            if query["kind"] == "journal":
                if not 0 < end - start <= 2 * 86400:
                    result["answer"] = "Choose a journal window of at most two days, or add a search topic."
                    return []
            else:
                start = max(start, end - JOURNAL_TAIL_S)
            items = recall.latest(start, end, query["person"], state, limit)
    except (OSError, sqlite3.Error, ValueError) as e:
        warnings.append("The journal could not be read (" + type(e).__name__ + ").")
        return []
    if query["kind"] in ("video", "camera", "mixed"):
        try:
            context = recall.overview(float(query["start"]), float(query["end"]), query["person"], state, limit=1)
        except (OSError, sqlite3.Error, ValueError, vision_log.continuum_py.ContinuumError) as error:
            warnings.append("Journal overview context could not be read (" + type(error).__name__ + ").")
            context = []
        combined = []
        seen = set()
        for item in context + items:
            ident = item.get("interpretation")
            if ident is not None and ident in seen:
                continue
            if ident is not None:
                seen.add(ident)
            combined.append(item)
        items = combined[:JOURNAL_ITEMS]
    if items:
        result["coverage"]["journal"] = dict(selected=len(items))
        warnings.append(recall.JOURNAL_WARNING)
    return items


def _transcripts(
    state: pathlib.Path, query: Mapping[str, Any], warnings: list[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    view = transcript_view.View(state)  # who said each line, and each voice's name, through the event log (step 4)
    person = query["person"]
    participants = {p.casefold() for p in query.get("participants", [])}
    # With an event log, a room line is said by a person (View.speakers), and a call line by the person its label
    # meant at the line's time (as profiles.load_calls reads it), so a renamed person keeps the lines heard under
    # their old name, and someone new who takes that name does not get them. A name the registry does not know, and a
    # state with no log, compare names, as before.
    said_by: dict[int, PersonId | None] = {}  # id(row) -> its person, for kept lines read through the log
    asked: dict[str, PersonId | None] = {}

    def speaks(name: str, row: Mapping[str, Any]) -> bool:
        """Whether the person asked about as `name` said this row."""
        if id(row) in said_by:
            if (key := name.casefold()) not in asked:
                p = view.person_at(name)
                asked[key] = p.id if p is not None else None
            if asked[key] is not None:
                return said_by[id(row)] == asked[key]
            if said_by[id(row)] is not None:
                return False
        return row["speaker"].casefold() == name.casefold()

    topic_query = bool(query.get("topics"))
    contextual = not topic_query and ((query.get("intent") and not person) or len(participants) > 1)
    # A voice clone may carry a person's name, and the microphone never matches it (View.clones).
    ids = {
        view.canonical(sid)
        for sid, name in view.names.items()
        if person and name and name.casefold() == person.casefold() and sid not in view.clones
    }
    # A person may have several voices (#772: naming replaces merging); a name is ambiguous only when its voices are
    # different people's.
    owners = (
        set(view.reg.voices(sorted(i for i in ids if i is not None), refresh=False).values()) if view.reg else set()
    )
    if len(ids) > 1 and (len(owners) != 1 or None in owners):
        warnings.append("Ambiguous speaker name: multiple independent speaker IDs; no name-filtered evidence selected.")
        return [], dict(ambiguous_speaker=True)
    rows: list[dict[str, Any]] = []
    malformed = 0
    excluded = 0
    files = 0

    def keep(
        e: dict[str, Any],
        ts: float,
        txt: str,
        label: str,
        resolved: str,
        pid: PersonId | None,
        path: pathlib.Path | None,
    ) -> None:
        """One line as an evidence row, when it's about the question's person (or the question isn't). `path` is a
        call's transcript, or None for a room line."""
        is_call = path is not None
        row = dict(
            **_provenance(e),
            ts=ts,
            speaker=resolved,
            recorded_speaker=label,
            speaker_id=e.get("speaker_id"),
            ts_end=e.get("ts_end", ts),
            conversation_source=str(path.parent) if path is not None else "room",
            quote=txt,
            source="call" if is_call else "transcript",
            link="/calls/detail?" + urlencode(dict(day=path.parent.parent.name, dir=path.parent.name))
            if path is not None
            else _review_link(ts),
        )
        if view.reg is not None and (not is_call or pid is not None):
            said_by[id(row)] = pid
        row["participants_matched"] = [p for p in query.get("participants", []) if speaks(p, row)]
        if person and not contextual and not topic_query and not speaks(person, row):
            said_by.pop(id(row), None)  # kept rows only, since a dropped row's id can be reused
            return
        rows.append(row)

    def read_room(batch: list[tuple[dict[str, Any], float, str, str]]) -> None:
        """Who said these room lines comes from the event log, as every reader sees it (step 4), shown under the
        current name of the person who said it. They're read together, so another mic's copy of a line is known
        (#284) and counted once (#476)."""
        if not batch:
            return
        original = {str(e.get("id") or registry.key_of(e)): (e, ts, txt, label) for e, ts, txt, label in batch}
        for heard, pid in view.named_speakers([b[0] for b in batch]):
            if heard.get("twin_of"):
                continue
            e, ts, txt, label = original[str(heard.get("id") or registry.key_of(heard))]
            keep(e, ts, txt, label, heard.get("speaker") or transcript_view.UNKNOWN, pid, None)

    def skipped(e: Mapping[str, Any], ts: float, txt: str, label: str, room: bool) -> bool:
        """Whether a line is left out: no words, the assistant's, or played back rather than said."""
        return (
            not _has_text(txt)
            or label.casefold() in bot_labels()
            or any(e.get(k) for k in NOT_SAID)
            or (room and overlaps(ts, e.get("ts_end", ts), state / "playback_windows.json"))
        )

    # Room lines come from the speech log over the question's window, read through the event log in batches, each
    # ending at a gap wider than a copy can lag its line.
    apart = 2 * max(twins.settings()[0], 0.0) + 1.0
    lo = max(float(query["start"]), 0.0)
    hi = float(query["end"]) if math.isfinite(query["end"]) else time.time() + 1.0
    from liberational.listen import revisions

    room = revisions.filter_lines(state, speech_log.shared().range(lo, hi)) if lo < hi else []
    files += bool(room)
    pending: list[tuple[dict[str, Any], float, str, str]] = []
    for e in room:
        ts = float(e["ts_start"])
        if not lo <= ts < hi:
            continue  # a line that only overlaps the window
        txt, label = str(e.get("text", "")), str(e.get("speaker", "unknown"))
        if skipped(e, ts, txt, label, room=True):
            excluded += 1
            continue
        if pending and ts - pending[-1][1] > apart:
            read_room(pending)  # a quiet gap: no copy of anything before it comes after it
            pending = []
        pending.append((e, ts, txt, label))
    read_room(pending)
    # Phone calls: the bridge's own transcripts, timed from each call's start.
    for path in sorted(calls_dir(state).glob("*/*/transcript.jsonl")):
        files += 1
        meta = _json(path.parent / "call.json", {})
        try:
            d = dt.datetime.fromisoformat(meta["started"])
            if d.tzinfo is None:
                if d.replace(tzinfo=TZ, fold=0).utcoffset() != d.replace(tzinfo=TZ, fold=1).utcoffset():
                    raise ValueError("ambiguous")
                d = d.replace(tzinfo=TZ)
            base = d.timestamp()
        except ValueError, KeyError, TypeError:
            malformed += 1
            continue
        with path.open(errors="replace") as f:
            for line in f:
                try:
                    e = json.loads(line)
                    ts = float(base + float(e["t"]))
                    if not math.isfinite(ts):
                        raise ValueError()
                except ValueError, TypeError, KeyError:
                    malformed += 1
                    continue
                if not query["start"] <= ts < query["end"]:
                    continue
                txt, label = str(e.get("text", "")), str(e.get("speaker", "unknown"))
                if skipped(e, ts, txt, label, room=False):
                    excluded += 1
                    continue
                pid: PersonId | None = None
                # A caller line keeps the name the bridge heard it under then, so it is read as of its time.
                label = str(e.get("speaker_label") or label)
                resolved = label
                if (who := view.person_at(label, ts)) is not None:
                    pid = who.id
                    resolved = label if label.casefold() == who.display_name.casefold() else who.display_name
                keep(e, ts, txt, label, resolved, pid, path)
    # Another mic's copy of a line with the same words in the same second is one thing said.
    seen: set[tuple[int, str]] = set()
    unique: list[dict[str, Any]] = []
    for r in rows:
        key = (int(r["ts"]), r["quote"].strip().casefold())
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    if contextual:
        anchors = [
            r
            for r in unique
            if (not participants or any(speaks(p, r) for p in participants))
            and (
                not query.get("intent")
                or re.search(
                    r"\b(?:want|wanted|need|should|plan|let.s|will|going to|done|finish|work|do)\b", r["quote"], re.I
                )
            )
        ]
        if len(participants) > 1:
            joint = [
                a
                for a in anchors
                if all(
                    any(speaks(p, r) for r in unique if abs(r["ts"] - a["ts"]) <= 120 and r["source"] == a["source"])
                    for p in participants
                )
            ]
            if joint:
                anchors = joint
            else:
                warnings.append(
                    "No planning exchange with both participant labels nearby; selected individual planning remarks, not proof of a shared conversation."
                )
        unique = [
            r for r in unique if any(abs(r["ts"] - a["ts"]) <= 120 and r["source"] == a["source"] for a in anchors)
        ]
        # Round-robin through day buckets so a busy first day cannot consume the
        # entire weekly evidence budget. Preserve chronological order within days.
        if len(unique) > MAX_EVIDENCE:
            buckets: dict[str, list[dict[str, Any]]] = {}
            for row in sorted(unique, key=lambda r: r["ts"]):
                buckets.setdefault(stamp(row["ts"])[:10], []).append(row)
            unique = [r for group in itertools.zip_longest(*buckets.values()) for r in group if r is not None]
            warnings.append("Planning evidence prioritized across days rather than earliest records only.")
    warnings.extend(
        [
            "Transcripts are sampled ASR, not a complete conversation record; speaker labels may be wrong.",
            "Playback windows retain only one day; older unmarked cloned playback cannot be reliably excluded. Sensitive-looking utterances are excluded, but unlabeled spoken credentials cannot be detected reliably.",
        ]
    )
    if malformed:
        warnings.append("Malformed or ambiguous timestamp records skipped: " + str(malformed))
    return (unique if contextual and len(unique) > MAX_EVIDENCE else sorted(unique, key=lambda r: r["ts"])), dict(
        files_scanned=files, excluded_playback_or_bot=excluded, malformed=malformed
    )


def _unscoped(query: Mapping[str, Any]) -> bool:
    """Whether a transcript question names nobody, and no topic or intent, so its budget keeps the newest lines."""
    return not (query.get("person") or query.get("topics") or query.get("intent") or query.get("participants"))


def _terms(question: str) -> set[str]:
    """The words of a question that say what it's about: three letters or more, not in QUESTION_WORDS."""
    return {w for w in re.findall(r"[a-z]{3,}", question.lower()) if w not in QUESTION_WORDS}


def _about_first(rows: list[dict[str, Any]], question: str) -> list[dict[str, Any]]:
    """Transcript lines, given in time order, in the order the evidence budget should take them: the lines that best
    match the question's words first (the newest first among equals), each followed by the lines of the same
    conversation (a call, or the room) said within CONTEXT_S of it, then the rest newest first. A word matches at the start of a word in the line, and a plural
    matches its singular too ("pickles" finds "pickle"). Each word a line holds adds its inverse document frequency
    over the window's lines, so one rare word ("pickles") outweighs several common ones ("come", "from")."""
    stems = [t[:-1] if t.endswith("s") and len(t) > 4 else t for t in _terms(question)]
    patterns = [re.compile(r"\b" + re.escape(stem), re.I) for stem in stems]
    matched = [[bool(p.search(r["quote"])) for p in patterns] for r in rows]
    weights = [math.log((len(rows) + 1) / (sum(m[k] for m in matched) + 1)) for k in range(len(patterns))]
    scores = [sum(w for w, hit in zip(weights, m, strict=True) if hit) for m in matched]
    times = [float(r["ts"]) for r in rows]
    taken: dict[int, None] = {}  # an ordered set of row indexes
    for i in sorted((i for i, score in enumerate(scores) if score), key=lambda i: (-scores[i], -times[i])):
        taken.setdefault(i)
        lo, hi = bisect.bisect_left(times, times[i] - CONTEXT_S), bisect.bisect_right(times, times[i] + CONTEXT_S)
        # Only from the same conversation: a call overlapping room speech mustn't fill the budget with the room's lines.
        source = rows[i].get("conversation_source")
        for j in range(lo, hi):
            if rows[j].get("conversation_source") == source:
                taken.setdefault(j)
    return [rows[i] for i in taken] + [rows[i] for i in reversed(range(len(rows))) if i not in taken]


def _within_budget(selected: list[dict[str, Any]], ordered_after: int | None) -> list[dict[str, Any]]:
    """The leading items of `selected` that fit MAX_EVIDENCE and MAX_CHARS, stopping at the first that doesn't.
    With `ordered_after`, the items after that many leading ones (a transcript's lines after its journal entries) are
    in the order to keep them rather than in time order (_about_first), and the kept ones are put back in time order."""
    kept: list[dict[str, Any]] = []
    size = 0
    for r in selected:
        if len(kept) >= MAX_EVIDENCE or size + len(r["quote"]) > MAX_CHARS:
            break
        kept.append(r)
        size += len(r["quote"])
    if ordered_after is not None:
        kept = kept[:ordered_after] + sorted(kept[ordered_after:], key=lambda r: float(r["ts"]))
    return kept


def _clock(ts: float) -> str:
    return stamp(ts)[11:16]


def _visit_evidence(v: Mapping[str, Any]) -> dict[str, Any]:
    return dict(
        **_provenance(v),
        ts=v["end"],
        camera=v["cam"],
        quote=f"{v['name']} seen on {v['cam']} from {_clock(v['start'])} to {_clock(v['end'])} in recorded video "
        f"(best face match {v['best_sim']:.2f})",
        source="camera",
        link=_review_link(v["start"], v["cam"]),
        method="face recognition over camera frames (face worker)",
    )


def _who(state: pathlib.Path, query: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """(answer, evidence) for "who was here [at T]" from the backfill's visits, on the cameras the question named (any
    camera when it named none)."""
    if query["at"] is not None:
        found = _all_visits(state, query["start"], query["end"])
        when = "around " + _clock(query["at"])
    else:
        found = _all_visits(state, query["start"], query["end"])
        when = _during(query["window"])
    found = [v for v in found if not query["cameras"] or v["cam"] in query["cameras"]]
    # A rematch names every visit by the face gallery now, even one from before its person was tagged, so a visit's
    # name is resolved as of now, not as of the visit (as timeline._visits does). Before the next rematch, an old name
    # still finds the person renamed from it.
    view = transcript_view.View(state)
    found = [{**v, "name": view.current(v["name"])} for v in found]
    people: list[str] = []
    for v in found:
        if v["name"] not in people:
            people.append(v["name"])
    if not people:
        return (
            f"Nobody was recognized {when} in the recorded video I've gone through. This does not establish absence.",
            [],
        )
    return (
        ", ".join(people) + f": recognized {when}. Face matching is uncertain, not proof of identity.",
        [_visit_evidence(v) for v in found],
    )


def _met(how: frozenset[Modality]) -> str:
    """How someone was met, in words: seen, heard, or seen and heard."""
    return " and ".join(m.value for m in (Modality.SEEN, Modality.HEARD) if m in how)


def _caveat(how: frozenset[Modality]) -> str:
    if how == {Modality.SEEN}:
        return "Face matching is uncertain, not proof of identity."
    if how == {Modality.HEARD}:
        return "Voice matching is uncertain, not proof of identity."
    return "Face and voice matching are uncertain, not proof of identity."


def _encounter_evidence(
    tl: timeline.Timeline, entry: timeline.Entry, start: float, end: float, visits: int | None = None
) -> list[dict[str, Any]]:
    """An encounter's visits that overlap [start, end] as camera evidence (only the last `visits` of them when
    given), and its last line there whose text may be shown as transcript evidence, oldest first."""
    name = tl.name(entry.person)
    shown = [v for v in entry.visits if v.end >= start and v.start <= end]
    items = [
        _visit_evidence(
            dict(name=name, cam=v.cam, start=v.start, end=v.end, best_sim=v.best_sim, payload_ids=v.payload_ids)
        )
        for v in sorted(shown, key=lambda v: v.end)[-visits if visits else 0 :]
    ]
    said = [ln for ln in entry.lines if start <= ln.start <= end and _has_text(ln.text)]
    if said:
        ln = said[-1]
        items.append(
            dict(
                **_provenance({"payload_ids": ln.payload_ids}),
                ts=ln.start,
                speaker=name,
                quote=ln.text,
                source="transcript",
                link=_review_link(ln.start),
            )
        )
    return sorted(items, key=lambda r: r["ts"])


def _during(window: str) -> str:
    """A question's window as words after "recognized": "today", "yesterday", "this week", "on 2026-10-20", "in the
    last 2 hours", "in the morning of 2026-10-30"."""
    if window in ("today", "yesterday") or window.endswith(" week"):
        return window
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", window):
        return "on " + window
    if window.startswith(("last ", "past ", "previous ")):
        return "in the " + window
    return "in " + window


def _unregistered(state: pathlib.Path, start: float, end: float) -> list[dict[str, Any]]:
    """The face visits overlapping [start, end] under a name the registry doesn't know (named by the face gallery
    now, as `_who` names them). The timeline has no person for such a name, so without these a registry that lost
    its people (a state wipe that kept the face gallery) would answer "nobody" while the cameras recognized someone.
    A forgotten person's visits are purged, so none of theirs is here."""
    view = transcript_view.View(state)
    visits = [{**v, "name": view.current(v["name"])} for v in _all_visits(state, start, end)]
    return [v for v in visits if timeline.person(v["name"], state) is None]


def _who_met(state: pathlib.Path, query: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]] | None:
    """(answer, evidence) for "who was here [at T]" from encounters, seen and heard, and the visits under names the
    registry doesn't know, as seen; or None with the timeline off or no event log (the caller answers from visits
    alone, as before)."""
    tl = timeline.load(state) if timeline.enabled() else None
    if tl is None:
        return None
    if query["at"] is not None:
        start, end, when = query["start"], query["end"], "around " + _clock(query["at"])
    else:
        start, end, when = query["start"], query["end"], _during(query["window"])
    found = tl.present(start, end)
    known = {p.name for p in found}
    unknown = [v for v in _unregistered(state, start, end) if v["name"] not in known]
    if not found and not unknown:
        nobody = f"Nobody was recognized {when} in the recorded video I've gone through or the transcript."
        return nobody + " This does not establish absence.", []
    evidence = [item for p in found for e in p.entries for item in _encounter_evidence(tl, e, start, end)]
    evidence += [_visit_evidence(v) for v in unknown]
    names = [f"{p.name} ({_met(p.seen_or_heard)})" for p in found]
    names += [f"{n} (seen)" for n in dict.fromkeys(v["name"] for v in unknown)]
    how = frozenset(m for p in found for m in p.seen_or_heard) | ({Modality.SEEN} if unknown else frozenset())
    return (
        ", ".join(names) + f": recognized {when}. " + _caveat(how),
        sorted(evidence, key=lambda r: r["ts"]),
    )


def _last_met(
    state: pathlib.Path, query: Mapping[str, Any], *, newer_than: float | None = None
) -> tuple[str, list[dict[str, Any]]] | None:
    """(answer, evidence) for "when was X last seen" from X's newest encounter, seen or heard, or None when the
    timeline is off, there is no event log, nobody is called X, or X has no encounter in the window (the caller
    answers as before)."""
    tl = timeline.load(state) if timeline.enabled() else None
    who = timeline.person(query["person"], state) if tl is not None else None
    entry = tl.last(who, query["start"], query["end"]) if tl is not None and who is not None else None
    if tl is None or who is None or entry is None:
        return None
    # An encounter can run past the window (seen at 23:59, heard at 00:00), so answer from the part inside it.
    lo, hi = max(entry.start, query["start"]), min(entry.end, query["end"])
    visits = [v for v in entry.visits if v.end >= lo and v.start <= hi]
    lines = [ln for ln in entry.lines if lo <= ln.start <= hi]
    if not visits and not lines:  # the window falls in a gap of the encounter: answer as before
        return None
    how = frozenset(({Modality.SEEN} if visits else set()) | ({Modality.HEARD} if lines else set()))
    last = max([min(v.end, hi) for v in visits] + [min(ln.end, hi) for ln in lines])
    if newer_than is not None and last <= newer_than:
        return None
    evidence = _encounter_evidence(tl, entry, lo, hi, visits=1)
    cams = sorted({v.cam for v in visits})
    where = " on " + " and ".join(cams) if cams else ""
    cites = "".join(f"[{'C' if r['source'] == 'camera' else 'T'}{i}]" for i, r in enumerate(evidence, 1))
    answer = (
        f"{stamp(last)}: {tl.name(who)} was last {_met(how)}{where}. In an encounter from "
        f"{_clock(lo)} to {_clock(last)}{' ' + cites if cites else ''}. {_caveat(how)}"
    )
    return answer, evidence


HOME_LOOKBACK_S = 86400.0  # "is X home" with no linked device reads X's last day of encounters (#1596)


def _minutes_ago(seconds: float) -> str:
    minutes = round(seconds / 60)
    if minutes < 1:
        return "just now"
    if minutes < 90:
        return "1 minute ago" if minutes == 1 else f"{minutes} minutes ago"
    hours = round(minutes / 60)
    return "about an hour ago" if hours == 1 else f"about {hours} hours ago"


def _watched_since(state: pathlib.Path, since: float, now: float) -> str:
    """How much of the time since `since` the cameras were following people (movement coverage's supported spans,
    which trail the cameras by seconds; activity coverage trails by minutes), in words; "" for a gap too short to
    matter or when the coverage can't be read."""
    if now - since < 120:
        return ""
    try:
        from liberational import segment_log

        rows = segment_log.at(state).range(segment_log.MOVEMENT_COVERAGE, since, now)
    except OSError, TimeoutError, RuntimeError, ValueError:
        return ""
    spans = sorted(
        (max(since, float(a)), min(now, float(b)))
        for r in rows
        if r.get("camera_motion") == "stable"
        for a, b in (r.get("supported") or [])
        if isinstance(a, int | float) and isinstance(b, int | float)
    )
    covered, cursor = 0.0, since
    for lo, hi in spans:
        lo = max(lo, cursor)
        if hi > lo:
            covered, cursor = covered + hi - lo, hi
    gap = round((now - since) / 60)
    watched = min(gap, round(covered / 60))
    return f" The cameras were watching for {watched} of the {gap} minutes since."


def _home_seen(state: pathlib.Path, name: str, now: float) -> tuple[str, list[dict[str, Any]]] | None:
    """(answer, evidence) for "is X home" when X has no linked device: when X was last seen or heard, and how much of
    the time since the cameras watched (#1596). Never "home" or "away": the cameras and microphones only cover where
    they point. None with the timeline off or nobody called X (the question goes the way it did)."""
    tl = timeline.load(state) if timeline.enabled() else None
    who = timeline.person(name, state) if tl is not None else None
    if tl is None or who is None:
        return None
    shown, start = tl.name(who), now - HOME_LOOKBACK_S
    entry = tl.last(who, start, now)
    if entry is None:
        return (
            (
                f"{shown} hasn't been seen on camera or heard in the last 24 hours. The cameras and microphones only "
                f"cover where they point, so that doesn't say whether {shown} is home."
            ),
            [],
        )
    visits = [v for v in entry.visits if v.end >= start and v.start <= now]
    lines = [ln for ln in entry.lines if start <= ln.start <= now]
    how = frozenset(({Modality.SEEN} if visits else set()) | ({Modality.HEARD} if lines else set()))
    last = max([min(v.end, now) for v in visits] + [min(ln.end, now) for ln in lines] + [min(entry.end, now)])
    cams = sorted({v.cam for v in visits})
    where = " on " + " and ".join(cams) if cams else ""
    answer = (
        f"{shown} was last {_met(how) or 'recognized'}{where} {_minutes_ago(now - last)} ({_clock(last)})."
        + _watched_since(state, last, now)
        + f" That's when {shown} was last recognized, not whether {shown} is home. {_caveat(how)}"
    )
    return answer, _encounter_evidence(tl, entry, start, now, visits=1)


def _all_visits(state: pathlib.Path, start: float, end: float) -> list[dict[str, Any]]:
    from liberational.vision import sightings

    keys = ("name", "cam", "start", "end", "best_sim")
    return [{k: v[k] for k in keys} for v in sightings.visits_between(start, end, state)]


def was_cut(coverage: Mapping[str, Any]) -> bool:
    """True when some evidence for the window was cut: a source said it is partial, or rows were dropped, skipped, or
    left unread, or part of the window has no informative samples. A source with no coverage record of its own (the
    transcript's lines, face visits) counts as complete unless it skipped records."""
    if coverage.get("partial") is True or coverage.get("truncated") or coverage.get("read_error"):
        return True
    if coverage.get("deadline_reached") or (coverage.get("malformed") or 0) > 0:
        return True
    if (coverage.get("unknown_seconds") or 0) > 0 or (coverage.get("unobserved_rows") or 0) > 0:
        return True
    scanned, available = coverage.get("rows_scanned"), coverage.get("rows_available")
    if isinstance(scanned, int) and isinstance(available, int) and scanned < available:
        return True
    return any(
        was_cut(value)
        for key, value in coverage.items()
        if isinstance(value, Mapping) and not (key == "movement" and not value.get("available"))
    )


def _movement_evidence(measured: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Evidence for the separately stated boundary counts and supported visibility durations."""
    if not measured.get("available"):
        return []
    output = []
    for camera, observed in measured.get("cameras", {}).items():
        if not observed.get("available"):
            continue
        for kind, key in (("entry", "entries"), ("exit", "exits")):
            for event in observed[key]:
                lo, hi = event["time"]
                at = (lo + hi) / 2
                output.append(
                    dict(
                        ts=at,
                        camera=camera,
                        source="camera",
                        quote=f"Supported image-boundary {kind} on {camera}; time bounded by {stamp(lo)} and {stamp(hi)}. This is not room entry or unique-person evidence.",
                        link="/review#" + urlencode(dict(day=stamp(at)[:10], t=at)),
                        method="source-backed camera-local movement edge",
                        stream="movement.track",
                        movement_evidence=event,
                        **_provenance({"payload_ids": event["source_ids"]}),
                    )
                )
        for pair in observed.get("coarrivals", ())[:5]:
            at = min(pair["times"][0][0], pair["times"][1][0])
            output.append(
                dict(
                    ts=at,
                    camera=camera,
                    source="camera",
                    quote=f"Two camera-local supported entry-time bounds are close on {camera}; this is timing proximity, not a social group or cross-camera identity.",
                    link="/review#" + urlencode(dict(day=stamp(at)[:10], t=at)),
                    method="camera-local coincident entry times",
                    movement_evidence=pair,
                    **_provenance({"payload_ids": pair["source_ids"]}),
                )
            )
        visible = [
            track for track in observed.get("tracks", ()) if track["supported_s"] > 0 and track["supported_intervals"]
        ]
        for track in visible[:5]:
            at = track["supported_intervals"][0][0]
            directions = sorted(
                {row["direction"] for row in track.get("directions", ()) if row["direction"] != "unknown"}
            )
            direction = ", ".join(directions) if directions else "unavailable"
            output.append(
                dict(
                    ts=at,
                    camera=camera,
                    source="camera",
                    quote=f"A camera-local track has {track['supported_s']:.1f} s of supported visibility on {camera}; observed image direction: {direction}. Gaps are excluded and identity is not established.",
                    link="/review#" + urlencode(dict(day=stamp(at)[:10], t=at)),
                    method="source-backed tracked visibility intervals",
                    stream="movement.track",
                    movement_evidence=track,
                    **_provenance({"payload_ids": track["source_ids"]}),
                )
            )
        if not observed["entries"] and not observed["exits"]:
            intervals = observed.get("supported_intervals") or ()
            if intervals:
                at = intervals[0][0]
                output.append(
                    dict(
                        ts=at,
                        camera=camera,
                        source="camera",
                        quote=f"Supported movement-analysis coverage on {camera}; no supported boundary event was recorded in the covered portion.",
                        link="/review#" + urlencode(dict(day=stamp(at)[:10], t=at)),
                        method="movement coverage",
                        stream="movement.coverage",
                        **_provenance({"payload_ids": observed["source_ids"]}),
                        movement_evidence={
                            "source_ids": observed["source_ids"],
                            "supported_intervals": intervals,
                            "partial": observed["partial"],
                        },
                    )
                )
    return output


def _presence_saved(state: pathlib.Path, query: Mapping[str, Any], now: float) -> bool:
    """Whether the presence count can answer within Ask's deadline (presence_counts.ready): a window of a day or
    two, or one whose ended days are kept. Otherwise the walk-by question is answered from the walk-by spans alone."""
    from liberational.memory import presence_counts

    try:
        return presence_counts.ready(float(query["start"]), float(query["end"]), query.get("cameras") or ())
    except ValueError:
        return False


def _day_label(start: float, end: float) -> str:
    """'today', 'yesterday', a weekday and date, or 'then' for a window that isn't one local day."""
    first = dt.datetime.fromtimestamp(start, TZ).date()
    if end - start > 86400 + 3600:
        return "then"
    today = dt.datetime.now(TZ).date()
    if first == today:
        return "today"
    if first == today - dt.timedelta(days=1):
        return "yesterday"
    return "on " + first.strftime("%A %b %-d")


# Lamps are measured only from the recordings' colour cast, as time on per whole day (memory/lights_color.py).
# Caption brightness follows the cameras' auto exposure, not the lamps, so anything else is unknown, with its reason.
LIGHTS_UNKNOWN = (
    "When the lights went on or off is unknown: lamps are only measured as time on over whole days, from the "
    "recordings' colour."
)
LIGHTS_WINDOW = (
    "How long the lights were on is unknown for this window: lamps are measured from the recordings' colour over one "
    "to three whole days, and this window isn't."
)
LIGHTS_NO_RECORDINGS = (
    "How long the lights were on is unknown: there are no recordings in this window to measure the lamps from."
)
LIGHTS_UNREADABLE = "How long the lights were on is unknown: the recordings couldn't be read ({error})."


def _lights_by_colour(
    state: pathlib.Path, query: Mapping[str, Any], warnings: list[str]
) -> tuple[str, dict[str, Any] | None]:
    """Lamps-on time from the recordings' colour cast (memory/lights_color.py), for windows of one to three whole
    local days: the answer and its coverage, or why it's unknown (the window, no recordings, or a failed read) and
    None."""
    from liberational.memory import lights_color

    start, end = float(query["start"]), float(query["end"])
    first = dt.datetime.fromtimestamp(start, TZ).date()
    first_start = dt.datetime.combine(first, dt.time(), tzinfo=TZ).timestamp()
    if first_start != start or end - start > 3 * 86400 + 3600:
        return LIGHTS_WINDOW, None
    days, total, measured = [], 0.0, 0.0
    d = first
    try:
        while dt.datetime.combine(d, dt.time(), tzinfo=TZ).timestamp() < end:
            res = lights_color.analyse_day(state, d, until=min(end, time.time()), tz=TZ)
            days.append(res)
            total += res["on_s"]
            measured += res["measured_s"]
            d += dt.timedelta(days=1)
    except (OSError, ValueError) as error:
        warnings.append(f"The recordings couldn't be read for the lights ({type(error).__name__}).")
        return LIGHTS_UNREADABLE.format(error=type(error).__name__), None
    if not measured:
        return LIGHTS_NO_RECORDINGS, None
    if len(days) == 1:
        return lights_color.answer_text(days[0], label=_day_label(start, end), tz=TZ), {"days": days}
    parts = [f"{r['day']}: {lights_color._hm(r['on_s'])}" for r in days]
    return f"The lights were on for about {lights_color._hm(total)} in total ({'; '.join(parts)}).", {"days": days}


def _conversation_summary(state: pathlib.Path, query: Mapping[str, Any]) -> tuple[str, list[Any]] | None:
    """Two or more named people: their conversations, summarised by the local model, media playing in the room set
    aside, ranked by time (word counts returned an audiobook's "khan, mongols", 2026-10-29). None when the local model
    gave nothing (offline, tests): the lexical ranking below answers instead."""
    from liberational.memory import conversation_summary

    try:
        res = conversation_summary.analyse(
            state, list(query["participants"]), float(query["start"]), float(query["end"]), background=False
        )
    except OSError:
        return None
    if not any(c["topic"] for c in res["conversations"]):
        return None
    return conversation_summary.answer_text(res, label=_day_label(query["start"], query["end"])), res["conversations"]


def collect(
    question: str,
    state_dir: str | pathlib.Path | None = None,
    now: float | dt.datetime | None = None,
    day: str | None = None,
) -> dict[str, Any]:
    from liberational.memory import activity_questions, device_answers, subject_activity, typical_counts

    state = pathlib.Path(state_dir or STATE).resolve()
    last = subject_activity.last_asked(question, state)
    animal = None if last is not None else subject_activity.asked(question, state)
    activity = (
        activity_questions.asked(question, state)
        if last is None
        and animal is None
        and not any(event_counts.counts(question, name) for name in event_counts.zone_names(state))
        else None
    )
    if activity is not None and activity.mode == "unsupported":  # stillness, repeated movement (#1863)
        return dict(
            is_history=True,
            answer=event_counts.UNSUPPORTED,
            evidence=[],
            coverage={"activity": {"available": False, "reason": "unsupported"}},
            warnings=[],
            query={},
            paths=["activity"],
        )
    device = device_answers.asked(question, state)
    home = device_answers.unlinked_home(question, state) if device is None else None
    result: dict[str, Any] = dict(
        is_history=bool(day)
        or is_history_question(question)
        or event_counts.wanted(question, state) is not None
        or activity is not None
        or animal is not None
        or last is not None
        or device is not None
        or (home is not None and timeline.person(home, state) is not None),
        answer=None,
        evidence=[],
        coverage={},
        warnings=[],
        query={},
    )
    if not result["is_history"]:
        return result
    now = time.time() if now is None else now.timestamp() if isinstance(now, dt.datetime) else float(now)
    try:
        query = parse(question, now, selected_day=day)
    except ValueError as e:
        from liberational.memory import presence_counts

        # An unsupported count window is a deterministic refusal, not a model prompt.
        if is_entry_count(question) and presence_counts.enabled():
            result["paths"] = ["presence"]
        result["answer"] = str(e)
        result["warnings"] = [str(e)]
        return result
    result["query"] = query
    warnings: list[str] = result["warnings"]
    selected: list[dict[str, Any]] = []
    reordered = False  # an unscoped transcript question keeps the lines about it, then those nearest the question time
    # Current identity must never be reconstructed from stale caption labels.
    try:
        from liberational.memory import presence_counts

        # A count of a named event is answered from detected events, never from captions (#290, #518): the count
        # per span, the recordings analyzed, and a Review link per span, or that nothing covers the window. People
        # coming into view stay with presence below.
        # Who is home and where things are, from linked devices (#1329).
        if query["kind"] == "journal":
            result["paths"] = ["journal"]
            selected = _journal(question, query, state, result, warnings)
            result["coverage"].setdefault("journal", {})["partial"] = True
            warnings.append("Journal searches are partial; no matching summary is not proof an event never occurred.")
        elif device is not None:
            result["paths"] = ["devices"]
            measured = device_answers.answer(device[0], device[1], query, state, stamp, TZ, now)
            result["answer"] = measured["answer"]
            result["coverage"]["devices"] = measured["coverage"]
            warnings.extend(measured["warnings"])
            selected = measured["evidence"]
        # "Is X home" for someone with no linked device: when they were last seen or heard (#1596).
        elif home is not None and (seen := _home_seen(state, home, now)) is not None:
            result["paths"] = ["timeline"]
            result["answer"], selected = seen
        # When an animal was last seen, or last in a zone: its newest track or visit, and coverage since.
        elif last is not None:
            result["paths"] = ["animal-last-seen"]
            measured = subject_activity.last_seen(last[0], last[1], last[2], query, state, stamp, now)
            result["answer"] = measured["answer"]
            result["coverage"]["animal"] = measured["coverage"]
            warnings.extend(measured["warnings"])
            selected = measured["evidence"]
        # What an animal has been doing (#1318): animal tracking was removed (#1808), and the answer says so.
        elif animal is not None:
            result["paths"] = ["animal-activity"]
            measured = subject_activity.answer(animal[0], animal[1], state, now)
            result["answer"] = measured["answer"]
            result["coverage"]["animal"] = measured["coverage"]
            warnings.extend(measured["warnings"])
            selected = measured["evidence"]
        # When a model action happened, or what someone did in a zone (#1863): see-actions' spans and guesses.
        elif activity is not None and activity.mode in ("when", "doing"):
            result["paths"] = ["activity"]
            measured = (
                activity_questions.when(activity, query, state, stamp, lambda ts: event_counts._clock(ts, TZ))
                if activity.mode == "when"
                else activity_questions.doing(activity, query, state, stamp)
            )
            result["answer"] = measured["answer"]
            result["coverage"]["activity"] = measured["coverage"]
            warnings.extend(measured["warnings"])
            selected = measured["evidence"]
        elif not is_entry_count(question) and (asked := event_counts.wanted(question, state)) is not None:
            result["paths"] = ["events"]
            who = transcript_view.View(state).person_at(query["person"]) if query.get("person") else None
            person = who.id if who is not None else _owner_if_first_person(question, state)
            unchecked = (
                asked in activity_kinds.model_kinds(state) and event_counts.uncalibrated(state, asked) is not None
            )
            # What's usual per day or night (#1318), not one window's count; a model kind not yet checked by the
            # owner's ratings has no count to make typical, so it gets the plain answer saying so.
            if typical_counts.asks(question) and not unchecked:
                result["paths"] = ["events", "typical"]
                measured = typical_counts.answer(
                    asked, query, state, stamp, TZ, now, event_counts.answer, person=person
                )
            else:
                measured = event_counts.answer(asked, query, state, stamp, TZ, person=person)
            result["answer"] = measured["answer"]
            result["coverage"]["events"] = measured["coverage"]
            warnings.extend(measured["warnings"])
            selected = measured["evidence"]
        # Walk-bys from the activity worker's spans and entries from movement tracks (#1591): a day reads in a few
        # seconds. The sampled presence visits (Review's chips) aren't read here: one missed
        # detection splits a visit, so their count is not a number to say.
        elif is_direction_question(question) and is_crossing_question(question, query):
            from liberational.memory import crossing_counts, walked_past

            result["paths"] = ["directions"]
            if not _presence_saved(state, query, now):
                result["answer"] = "Directional evidence is unavailable for this window; try a day or two at a time."
                result["coverage"]["partial"] = True
            else:
                crossed = crossing_counts.from_spans(state, query["start"], query["end"], query["cameras"])
                result["walked_past"] = crossed
                result["answer"] = walked_past.direction_text(
                    crossed, include_counts=bool(ENTRY_COUNT.search(question))
                )
                result["coverage"]["walked_past"] = {"partial": crossed["partial"]}
                for cam, info in crossed["cameras"].items():
                    for example in info["direction_examples"]:
                        selected.append(
                            dict(
                                ts=example["ts"],
                                camera=cam,
                                source="camera",
                                quote=f"Observed crossing in the camera image: {example['direction'].replace('_', ' ')}.",
                                link=example["link"],
                                method="see-actions' directional crossing witness",
                                stream="walkby.span",
                                **({"payload_id": example["payload_id"]} if example.get("payload_id") else {}),
                            )
                        )
        elif is_crossing_question(question, query) and presence_counts.enabled() and _presence_saved(state, query, now):
            from liberational.memory import crossing_counts, movement_counts, walked_past

            result["paths"] = ["presence"]
            with concurrent.futures.ThreadPoolExecutor(1) as pool:  # the two reads overlap; the link serves both
                walks = pool.submit(crossing_counts.from_spans, state, query["start"], query["end"], query["cameras"])
                tracked = movement_counts.collect(
                    state, start=query["start"], end=query["end"], cameras=query["cameras"]
                )
            try:
                crossed = walks.result()
            except Exception as error:  # the walk-by count is extra here: without it the answer says less
                LOG.warning("walk-by count unavailable for a presence answer: %r", error)
                crossed = None
            result["walked_past"] = crossed
            seen_to = min(float(query["end"]), now)
            result["answer"] = walked_past.answer_text(
                crossed,
                tracked,
                max(0.0, seen_to - float(query["start"])),
                TZ,
                so_far=float(query["end"]) > now,
                entries=not WALKED_BY.search(question),
            )
            if crossed is None and not tracked.get("available") and tracked.get("errors"):
                result["answer"] = "History source could not be read safely."  # neither source: never a zero
                result["coverage"]["read_error"] = True
            result["movement"] = tracked
            result["coverage"]["movement"] = tracked
            result["coverage"]["walked_past"] = (
                {"partial": crossed["partial"]} if crossed is not None else {"read_error": True}
            )
            if tracked.get("available"):
                result["paths"].append("movement")
                selected.extend(_movement_evidence(tracked))
            for cam, info in ((crossed or {}).get("cameras") or {}).items():
                for example in info["examples"]:
                    selected.append(
                        dict(
                            ts=example["ts"],
                            camera=cam,
                            source="camera",
                            quote=f"Walked past {cam}'s counting line"
                            + (f" ({example['direction'].replace('_', ' ')})" if example.get("direction") else "")
                            + ".",
                            link=example["link"],
                            method="see-actions' walk-by span",
                            stream="walkby.span",
                            **({"payload_id": example["payload_id"]} if example.get("payload_id") else {}),
                        )
                    )
        # Otherwise a walk-by question is answered from the walk-by spans alone (crossing_paths), even with
        # recorded-frame analysis off: never from sampled captions, which the model would count ("3 times in 24
        # hours", 2026-10-28).
        elif query["kind"] == "light_duration":
            result["answer"], result["coverage"]["lights"] = _lights_by_colour(state, query, warnings)
            if result["coverage"]["lights"] is not None:
                result["paths"] = ["lights-colour"]
        elif query["kind"] in ("video", "mixed") and (
            cfg.get("review.recorded_frame_analysis") or is_crossing_question(question, query)
        ):
            from liberational.memory import video_evidence

            if query["kind"] == "video" and is_crossing_question(question, query):
                # Walk-by counts are answered by the walk-by spans below; tiny VLM windows would only add a misleading
                # six-second denominator.
                selected, coverage, video_warnings = [], {}, []
            else:
                selected, coverage, video_warnings = video_evidence.collect(state, query)
            result["coverage"] = coverage
            warnings.extend(video_warnings)
            if query["kind"] == "mixed":
                transcript, transcript_coverage = _transcripts(state, query, warnings)
                selected = [
                    item for pair in itertools.zip_longest(selected, transcript) for item in pair if item is not None
                ]
                result["coverage"]["transcript"] = transcript_coverage
            if (
                query["kind"] == "video"
                and not is_crossing_question(question, query)
                and re.search(r"how many|count|times", question, re.I)
            ):
                result["answer"] = video_evidence.measured_answer(coverage, query["kind"])
            if is_crossing_question(question, query):
                result["crossings"] = crossing_paths(state, query, warnings)
                counted = (result["crossings"]["path_b"] or {}).get("result")
                result["coverage"]["walked_past"] = (
                    {"partial": counted["partial"]} if counted is not None else {"read_error": True}
                )
                if result["crossings"].get("answer"):
                    result["answer"] = result["crossings"]["answer"]
        elif query["kind"] == "transcript":
            selected, coverage = _transcripts(state, query, warnings)
            result["coverage"] = coverage
            if (
                query.get("topics")
                and len(query.get("participants") or []) >= 2
                and (summary := _conversation_summary(state, query))
            ):
                result["answer"], result["coverage"]["conversations"] = summary
                result["paths"] = ["conversation-summary"]
            elif query.get("topics"):
                from liberational.memory import conversation_topics

                ranked = conversation_topics.analyse(selected, query)
                result["answer"] = ranked["answer"]
                result["coverage"]["topics"] = ranked["coverage"]
                selected = ranked["evidence"]
                warnings.extend(ranked["warnings"])
            elif _unscoped(query):
                selected, reordered = _about_first(selected, question), True
        # Encounters mix every camera and speech, so a question about named cameras keeps to their visits.
        elif query["kind"] == "who" and not query["cameras"] and (met := _who_met(state, query)) is not None:
            result["answer"], selected = met
            result["paths"] = ["timeline"]
            warnings.append(
                "Only recorded video the backfill has processed and transcript lines are searched; both may be partial."
            )
        elif query["kind"] == "who":
            result["answer"], selected = _who(state, query)
            warnings.append("Only recorded video the backfill has processed is searched; recordings may be partial.")
        elif (
            query["kind"] == "camera"
            and query["person"]
            and not query["cameras"]
            and (met := _last_met(state, query)) is not None
        ):
            result["answer"], selected = met
            result["paths"] = ["timeline"]
        elif query["person"] and (recorded := _recorded_visits(state, query)):
            v = recorded[0]
            selected = [_visit_evidence(v)]
            result["answer"] = (
                f"{v['name']} was last seen at {stamp(v['end'])} on {v['cam']}, in a visit from {_clock(v['start'])} "
                f"to {_clock(v['end'])} [C1]. Face matching is uncertain, not proof of identity."
            )
        else:
            rows, available, coverage = _camera(state, query, warnings)
            rows = [r for r in rows if _has_text(r["caption"])]
            rows = [r for r in rows if not r["unchanged"]]  # a repeat of an unchanged view is never quoted
            if not rows:
                warnings.append("No safe camera samples in the requested window; no event total can be established.")
            camera_coverage: dict[str, Any] = dict(cameras=coverage, available_cameras=available)
            result["coverage"] = camera_coverage
            warnings.append(
                "Camera history is sampled and retention-limited; missing observations cannot establish absence."
            )
            missing = [cam for cam in query["cameras"] if cam not in available]
            if missing:
                result["answer"] = "No saved history for " + ", ".join(missing) + ". Camera IDs are not renumbered."
            elif query["kind"] == "lights":
                # Lamp on and off moments have no measure: caption brightness follows auto exposure, not the lamps.
                selected = []
                result["answer"] = LIGHTS_UNKNOWN
            elif query["person"]:
                person = query["person"]
                # A live camera row keeps the face name it was written with, so a face is the person when that name
                # meant them then (a renamed person keeps their rows, someone new with the old name gets none). A
                # name that meant someone since forgotten is nobody's. Without a log, or for a name the registry
                # does not know, names are compared, as before.
                view = transcript_view.View(state)
                target = view.person_at(person)
                shown = target.display_name if target is not None else person

                def is_person(name: str, at: float) -> bool:
                    if target is not None and (p := view.person_at(name, at)) is not None:
                        return p.id == target.id
                    if view.forgotten_at(name, at):
                        return False
                    return name.casefold() == person.casefold()

                matches: list[dict[str, Any]] = []
                mentions: list[dict[str, Any]] = []
                for r in rows:
                    faces = r["faces"] if isinstance(r["faces"], list) else []
                    for face in faces:
                        try:
                            matched = float(face.get("sim", 0)) > 0.45 and is_person(str(face.get("name", "")), r["ts"])
                        except ValueError, TypeError, AttributeError:
                            matched = False
                        if matched:
                            matches.append(dict(r, identity=dict(name=shown, similarity=face["sim"], threshold=0.45)))
                            break
                    if re.search(r"\b" + re.escape(person) + r"\b", r["caption"], re.I):
                        mentions.append(r)
                if matches:
                    selected = [matches[-1]]
                    result["answer"] = (
                        shown
                        + " was last recognized in available samples at "
                        + stamp(matches[-1]["ts"])
                        + " on "
                        + matches[-1]["cam"]
                        + " [C1]. Face matching is uncertain, not proof of identity."
                    )
                elif mentions:
                    selected = [dict(mentions[-1], method="caption name mention only; NOT recognized identity")]
                    result["answer"] = (
                        "No supported face-recognition match. Latest caption mentioning "
                        + person
                        + " is "
                        + stamp(mentions[-1]["ts"])
                        + " [C1]; this does not establish that person was present."
                    )
                else:
                    result["answer"] = (
                        "No supported face-recognition match in saved samples; identity data may be unavailable. This does not establish absence."
                    )
            else:
                # Rank actual question content before the bounded evidence slice; keep neighboring
                # observations so a matching object/action does not lose its transition context.
                terms = _terms(question)
                scores = [sum(term in r["caption"].lower() for term in terms) for r in rows]
                hits = {i for i, score in enumerate(scores) if score}
                context = {j for i in hits for j in (i - 1, i, i + 1) if 0 <= j < len(rows)}
                selected = (
                    [rows[i] for i in sorted(context, key=lambda i: (-scores[i], rows[i]["ts"]))] if hits else rows
                )
                if query["kind"] == "video":
                    warnings.append(
                        "Caption-first sampled evidence: visible counts are not passage totals or unique people; timestamps do not prove continuous duration. Unobserved intervals remain unknown."
                    )
            selected = [camera_evidence(r) for r in selected if _has_text(r["caption"])]
    except (OSError, sqlite3.Error, ValueError, TypeError, vision_log.continuum_py.ContinuumError) as e:
        result["answer"] = "History source could not be read safely."
        warnings.append(type(e).__name__)
        result["coverage"]["read_error"] = True
    counted = "events" in result.get("paths", [])  # an event count answers from its events alone (#290)
    if cfg.get("review.recorded_frame_analysis") and query["kind"] == "camera" and not query["person"] and not counted:
        from liberational.memory import video_evidence

        frames, video_coverage, video_warnings = video_evidence.collect(state, query)
        selected = frames + selected
        if frames:
            result["answer"] = None
        result["coverage"]["recorded_video"] = video_coverage
        warnings.extend(video_warnings)
    # Journal entries mix every camera and speech, so a question about named cameras keeps to their own evidence.
    if (
        query["kind"] in ("transcript", "camera", "video", "mixed")
        and result["answer"] is None
        and not query["cameras"]
        and not is_crossing_question(question, query)
    ):
        journal = _journal(question, query, state, result, warnings)
        selected = journal + selected
    else:
        journal = []
    if (
        query["kind"] == "mixed"
        and not cfg.get("review.recorded_frame_analysis")
        and "presence" not in result.get("paths", [])
        and not counted
    ):
        transcript, transcript_coverage = _transcripts(state, query, warnings)
        selected = [item for pair in itertools.zip_longest(selected, transcript) for item in pair if item is not None]
        result["coverage"]["transcript"] = transcript_coverage
    count = len(selected)
    for r in _within_budget(selected, len(journal) if reordered else None):
        result["evidence"].append(labeled(r, len(result["evidence"]) + 1))
    truncated = len(result["evidence"]) < count
    result["coverage"].update(matched=count, selected=len(result["evidence"]), truncated=truncated)
    result["coverage"]["partial"] = was_cut(result["coverage"])
    if truncated:
        warnings.append("Evidence limit reached; selected matching records are partial, not all findings.")
    if (
        (query["kind"] in ("lights", "who") or is_crossing_question(question, query) or counted)
        and result["evidence"]
        and result["answer"]
    ):
        result["answer"] += " Evidence: " + ", ".join("[" + r["id"] + "]" for r in result["evidence"][:5]) + "."
    if not result["evidence"] and result["answer"] is None:
        result["answer"] = NO_EVIDENCE
    return result
