"""The same speech heard on two microphones (#285, #284): one rule for when two lines are one thing said.

Two mics in one room both hear a sentence, so the transcriber gets it twice, once per device, a moment apart and
transcribed almost alike. Two lines are the same speech when they come from different devices, overlap in time (an
echo of one sentence does; two people saying "yes" a second apart don't), start within audio.dedup_window_s of each
other, and their text is at least audio.dedup_similarity alike (difflib's ratio, 0 to 1, on lowercased words). The transcriber uses this to write only the first copy (#285); a reader can use it to link the
pairs written before that, or left when the rule was off (#284), with pairs().
"""

import collections
import dataclasses
import difflib
import functools
import re

from liberational import config as cfg


@dataclasses.dataclass(frozen=True)
class Heard:
    """One line as the rule sees it: its line key (registry.key_of), mic, when it started, and its words."""

    key: str
    device: str
    start: float
    end: float
    text: str
    shown: str = ""  # its text as written, for a log line ("[password]" for the voice password)


@functools.lru_cache(maxsize=8192)  # a day's reading compares each line with a few others: work its words out once
def words(text: str) -> str:
    """Text compared as words: lowercased, punctuation dropped, spaces collapsed."""
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


def similarity(a: str, b: str) -> float:
    """How alike two lines' words are, 0 to 1 (difflib's ratio). Two empty lines are not alike."""
    wa, wb = words(a), words(b)
    if not wa or not wb:
        return 0.0
    return difflib.SequenceMatcher(None, wa, wb).ratio()


@functools.lru_cache(maxsize=65536)  # readers ask about the same pairs request after request
def alike(a: str, b: str, min_similarity: float) -> bool:
    """Whether similarity(a, b) >= min_similarity: the same words at once, else difflib's cheap upper bounds before
    its ratio (a two-mic transcript read whole compares about one pair per line, #284 review)."""
    wa, wb = words(a), words(b)
    if not wa or not wb:
        return False
    if wa == wb:
        return True
    m = difflib.SequenceMatcher(None, wa, wb)
    return m.real_quick_ratio() >= min_similarity and m.quick_ratio() >= min_similarity and m.ratio() >= min_similarity


def same_speech(a: Heard, b: Heard, window_s: float, min_similarity: float) -> bool:
    """Whether a and b are one thing said, heard on two mics."""
    return (
        a.device != b.device
        and a.start < b.end
        and b.start < a.end
        and abs(a.start - b.start) <= window_s
        and alike(a.text, b.text, min_similarity)
    )


def is_primary(device: str, primary: list[str]) -> bool:
    """Whether device is one whose line is kept even when another mic's copy came first (audio.primary_devices:
    case-insensitive pieces of a device name)."""
    name = device.lower()
    return any(p.strip() and p.strip().lower() in name for p in primary)


class Recent:
    """The lines written in the last window, to tell a second mic's copy from new speech (#285)."""

    def __init__(self, keep: int = 64) -> None:
        self.lines: collections.deque[Heard] = collections.deque(maxlen=keep)

    def add(self, h: Heard) -> None:
        self.lines.append(h)

    def twin_of(self, h: Heard, window_s: float, min_similarity: float, primary: list[str]) -> Heard | None:
        """The written line h repeats, if it is a copy to drop: the same speech from another mic, unless h's mic is a
        primary one and the written copy's isn't (then both stay, as a primary mic's line always does)."""
        if window_s <= 0 or min_similarity <= 0:  # off (a similarity of 0 would make every line a copy)
            return None
        for w in reversed(self.lines):  # two mics' lines can be written out of start order: look at all of them
            if same_speech(w, h, window_s, min_similarity):
                if is_primary(h.device, primary) and not is_primary(w.device, primary):
                    return None
                return w
        return None


def pairs(lines: list[Heard], window_s: float, min_similarity: float, primary: list[str]) -> dict[int, int]:
    """Lines written as copies of the same speech (#284): {index of a copy: index of the line shown for it}. Each
    group of one thing said shows one line, its primary mic's when it has one (as the transcriber would have kept),
    else its earliest; the others point at it. Lines that repeat nothing aren't in the result."""
    if window_s <= 0 or min_similarity <= 0:
        return {}
    order = sorted(range(len(lines)), key=lambda i: lines[i].start)
    group: dict[int, int] = {}  # line index -> its group's first line
    mics: dict[int, set[str]] = {}  # a group's first line -> the mics its lines came from
    for pos, i in enumerate(order):
        for k in reversed(order[max(0, pos - 64) : pos]):  # the lines just before it, nearest first
            if lines[i].start - lines[k].start > window_s:
                break
            first = group.get(k, k)
            if lines[i].device in mics.get(first, {lines[first].device}):
                continue  # a mic says a thing once per group: the same words again from it are said again
            if same_speech(lines[k], lines[i], window_s, min_similarity):
                group[i] = first
                mics.setdefault(first, {lines[first].device}).add(lines[i].device)
                break
    members: dict[int, list[int]] = collections.defaultdict(list)
    for i, first in group.items():
        members[first].append(i)
    out: dict[int, int] = {}
    for first, rest in members.items():
        everyone = [first, *rest]
        shown = next((i for i in everyone if is_primary(lines[i].device, primary)), first)
        out.update({i: shown for i in everyone if i != shown})
    return out


DEFAULTS = (1.5, 0.9, ["reSpeaker"])  # as defaults.toml has them, for a pack.toml value that can't be read


def settings() -> tuple[float, float, list[str]]:
    """(window in seconds, least similarity, primary devices) from [audio] in config. A value that can't be read
    falls back to its default rather than stopping the transcriber; one device name alone counts as a list of one."""
    out: list[float] = []
    for key, default in zip(("dedup_window_s", "dedup_similarity"), DEFAULTS[:2], strict=True):
        try:
            out.append(float(cfg.get(f"audio.{key}")))
        except TypeError, ValueError:
            out.append(float(default))
    primary = cfg.get("audio.primary_devices")
    if isinstance(primary, str):
        primary = [primary]
    if not isinstance(primary, list):
        primary = DEFAULTS[2]
    return out[0], out[1], [str(p) for p in primary]
