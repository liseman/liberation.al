"""Plain-language lines from the local model, cached (#404, and Live's "What's happening", #354).

A source text (a journal entry, a run of conversation) goes in; a short neutral line comes out, with topic tags:
"Worked at the desk for about 3 hours" ["work"]. clean() checks every reply. It must be JSON with a text that has no
first person (the journal is written as "I"), no subject for whoever the line is about ("He worked", "A person sat",
"The owner had lunch", "You read"), and at most MAX_CHARS characters; names of other people stay, and so do the
pronouns that go with them ("Talked with Dana about her new job"). A reply that breaks a rule is asked for once
more, with the rule it broke; one that breaks it again is dropped, never shown, and not asked for again while its
source stays the same.

Nobody waits for the model: many() returns what is cached and queues the rest for one background thread, the most
recent request first (so the stretch on screen comes before ones looked at earlier), MAX_QUEUE at most. The thread
yields before each model call through the shared bounded voice gate, as the journal does, so a
rewrite never delays a spoken reply. Lines are cached by their source text, so an entry rebuilt with the same words
costs no new call. A model that is down is tried again after RETRY_S, not on every read. Rewrites keeps its lines
in the process; a user can keep them elsewhere by overriding _rows, _save, and meta, as memory/review_summary.py
keeps them as review.summary records on the memory log (#778).

Named lines (check_named, NAMED_SYSTEM) are the other rule set, for Live's log: who did it
comes first, by name ("Kara asked Luca about the couch"), never a pronoun, an automatic label, or "the owner".

memory/clip_labels.py is the same idea, from camera captions, with its own stricter rules (no subject, no name).
"""

import collections
import hashlib
import json
import pathlib
import re
import threading
import time
from collections.abc import Callable, Iterable
from typing import Any

from liberational import logs

MAX_CHARS = 90
MAX_QUEUE = 200
RETRY_S = 600.0
VOICE_WAIT_S = 300.0  # waits at most this long for a voice turn to end, for an injected custom busy predicate
TOPICS = (
    "work",
    "meal",
    "conversation",
    "call",
    "visit",
    "decision",
    "rest",
    "exercise",
    "chores",
    "errand",
    "media",
    "music",
    "travel",
    "planning",
)

LOG = logs.get("plain-text")

SYSTEM = (
    "You write one short line for a timeline of someone's day, from a diary entry written in the first person. Say "
    "what happened, neutrally and with no subject: start with a verb in the past tense or a short noun phrase, as in "
    '"Worked at the desk for about 3 hours", "Lunch in the kitchen", or "Talked with Dana about her trip". Never '
    "use I, me, my, we, our, or you, and never start with who did it (he, she, they, the person, the owner). Other "
    "people's names may stay. At most 90 characters. Then give the line's topics, chosen only from this list: "
    + ", ".join(TOPICS)
    + '. Reply with JSON only: {"text": "...", "topics": ["..."]}.'
)
AGAIN = "\n\nYour last line broke a rule: {why}. Write it again, following every rule."

# "I" only as a word on its own in capitals; "us" only in lower case, so "the US" is fine.
FIRST_PERSON_I = re.compile(r"(?<![\w'])I(?:'(?:m|ve|d|ll))?(?![\w'])")
FIRST_PERSON = re.compile(r"\bus\b|\b(me|my|mine|myself|we|our|ours|ourselves|let's)\b")
FIRST_PERSON_ANY_CASE = re.compile(r"\b(me|my|mine|myself|we|our|ours|ourselves|let's)\b", re.I)


def _says_i(text: str) -> bool:
    """Whether text says "I" about the diarist: a capital "I" on its own, except after a capitalized word that
    isn't the first, where it is part of a name ("World War I", "Henry I")."""
    for m in FIRST_PERSON_I.finditer(text):
        before = text[: m.start()].split()
        if len(before) >= 2 and before[-1][:1].isupper() and m.group() == "I":
            continue
        return True
    return False


SUBJECT = re.compile(
    # who did it, as the line's first word: "He worked", "A person sat", "Someone cooked", "Man had lunch"
    r"^(he|she|they|it|themself|themselves|(an? |the )?(man|woman|person|people|boy|girl|guy|lady|someone|somebody|"
    r"user|owner|narrator|author|writer|diarist|individual))\b|"
    # the reader, anywhere ("the owner of the cafe" mid-line is someone else, and fine)
    r"\b(you|your|yours|yourself)\b",
    re.I,
)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _json_in(reply: str) -> Any:
    """The first JSON object in a reply (a model may add words before or after it), or None."""
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", reply):
        try:
            return dec.raw_decode(reply, m.start())[0]
        except ValueError:
            continue
    return None


def check(reply: str | None, max_chars: int = MAX_CHARS) -> tuple[dict[str, Any] | None, str]:
    """(the reply as {"text", "topics"}, "") or (None, the rule it broke). Topics not in TOPICS are dropped. The
    text gets a capital first letter and no full stop at the end (a timeline row). A longer line (a part of the day's
    rollup, #2276) passes its own `max_chars`."""
    d = _json_in(str(reply or ""))
    if not isinstance(d, dict) or not isinstance(d.get("text"), str):
        return None, 'reply with JSON: {"text": "...", "topics": [...]}'
    t = " ".join(d["text"].split()).strip().strip("\"'`“”").rstrip(" .")
    if not t:
        return None, "the text was empty"
    if len(t) > max_chars:
        return None, f"at most {max_chars} characters"
    if _says_i(t) or FIRST_PERSON.search(t) or FIRST_PERSON_ANY_CASE.search(t):
        return None, "no I, me, my, we, or our"
    if SUBJECT.search(t):
        return None, "no subject: never start with who did it, and never say you"
    topics = d.get("topics") if isinstance(d.get("topics"), list) else []
    line = {"text": t[0].upper() + t[1:], "topics": sorted({str(x).lower() for x in topics} & set(TOPICS))}
    return line, ""


def clean(reply: str | None) -> dict[str, Any] | None:
    """check()'s line, or None when the reply broke a rule."""
    return check(reply)[0]


# ---- named lines (Live's What's happening, #354): who did it comes first, by name ----
NAMED_MAX_CHARS = 80
NAMED_SYSTEM = (
    "You write one short line for a live log of a room, from a few lines of conversation, each given as "
    '"Name: what they said". Say who talked with whom and about what, in the past tense, like "Kara asked Luca '
    'about the couch" or "Dana told Sam about her trip". Start with the name of who spoke. Use only the names '
    'given; anyone called "someone" stays "someone". Never use I, me, my, we, our, or you, and never "the owner", '
    '"the user", or "the person". At most 80 characters. Reply with JSON only: {"text": "...", "topics": []}.'
)
UNNAMED = re.compile(
    r"\bspeaker[ _]?\d+\b|\bface \d+\b|\bthe (owner|user|person|narrator|diarist)\b|\b(you|your|yours|yourself)\b", re.I
)
PRONOUN_FIRST = re.compile(r"^(he|she|they|it|you)\b", re.I)


def check_named(reply: str | None) -> tuple[dict[str, Any] | None, str]:
    """check() for a named line: (line, "") or (None, the rule it broke). The line starts with who did it (a name,
    or "someone"), never a pronoun; no first person; no automatic label ("Speaker 4") or "the owner"; at most
    NAMED_MAX_CHARS characters."""
    d = _json_in(str(reply or ""))
    if not isinstance(d, dict) or not isinstance(d.get("text"), str):
        return None, 'reply with JSON: {"text": "...", "topics": []}'
    t = " ".join(d["text"].split()).strip().strip("\"'`\u201c\u201d").rstrip(" .")
    if not t:
        return None, "the text was empty"
    if len(t) > NAMED_MAX_CHARS:
        return None, f"at most {NAMED_MAX_CHARS} characters"
    # Named lines are full of names, so a capital "I" after one is never a name part here ("Kara told Luca I...").
    if FIRST_PERSON_I.search(t) or FIRST_PERSON.search(t) or FIRST_PERSON_ANY_CASE.search(t):
        return None, "no I, me, my, we, or our"
    if UNNAMED.search(t):
        return None, 'use the names given, or "someone"; never "Speaker 4", "the owner", or "you"'
    if PRONOUN_FIRST.search(t):
        return None, "start with the name of who did it"
    return {"text": t[0].upper() + t[1:], "topics": []}, ""


def _ask(system: str, text: str) -> str:
    """One local model call (journal.model, else models.ollama.llm) with this system prompt: the reply, unchecked."""
    from liberational.memory import journal

    return journal.chat(system, f"{text}\n\nJSON:", journal.model_name(), 120)


def _rewrite(text: str) -> str:
    return _ask(SYSTEM, f"Diary entry:\n{text}")


class Rewrites:
    """One cache of plain lines and its background writer. rewrite is the model call and busy says when to wait for
    a voice turn (a test passes its own of each, and background False to run the queue itself with drain())."""

    def __init__(
        self,
        state: pathlib.Path,
        rewrite: Callable[[str], str] = _rewrite,
        background: bool = True,
        clock: Callable[[], float] = time.time,
        idle_s: float = 30.0,
        busy: Callable[[], bool] | None = None,
        checker: Callable[[str | None], tuple[dict[str, Any] | None, str]] = check,
    ) -> None:
        self.state = pathlib.Path(state)
        self.checker = checker
        self.rewrite, self.background, self.clock, self.idle_s = rewrite, background, clock, idle_s
        self.busy = busy if busy is not None else (lambda: False)
        self.lock = threading.Lock()
        self.todo: collections.deque[str] = collections.deque()
        self.queued: set[str] = set()
        self.unsaved: dict[str, float] = {}  # sources whose line couldn't be saved: not asked again until then
        self.wake = threading.Event()
        self.worker: threading.Thread | None = None
        self.kept: dict[str, tuple[Any, Any]] = {}
        self.marks: dict[str, str] = {}

    def _rows(self, keys: list[str]) -> dict[str, tuple[Any, Any]]:
        """{key: (line or None, retry_at or None)} for the keys kept (a subclass reads its own store)."""
        return {k: self.kept[k] for k in keys if k in self.kept}

    def _save(self, source: str, entry: dict[str, Any]) -> None:
        self.kept[_hash(source)] = (entry.get("line"), entry.get("retry_at"))

    def meta(self, key: str, value: str | None = None) -> str | None:
        """A value kept with the lines (a writer's own bookmark): set it when value is given; returns it."""
        if value is not None:
            self.marks[key] = value
        return self.marks.get(key)

    def _judge(self, sources: list[str]) -> tuple[dict[str, tuple[str, dict[str, Any] | None]], list[str]]:
        """(each source's stage, as many() gives it, the sources write_one would write now: no line yet, or a failed
        try whose RETRY_S has passed), from one look at the store."""
        rows = self._rows([_hash(src) for src in sources])
        now = self.clock()
        out: dict[str, tuple[str, dict[str, Any] | None]] = {}
        due = []
        for source in sources:
            hit = rows.get(_hash(source))
            if hit is not None and hit[0] is not None:
                out[source] = ("ready", hit[0])
                continue
            if hit is not None and not hit[1]:
                out[source] = ("rejected", None)
                continue
            out[source] = ("pending", None)
            if hit is None or now >= hit[1]:
                due.append(source)
        return out, due

    def due(self, sources: Iterable[str]) -> list[str]:
        """The sources write_one would write now (no line yet, or a failed try whose RETRY_S has passed), in order;
        not one whose line couldn't be saved until its RETRY_S has passed, as many() does."""
        now = self.clock()
        return [src for src in self._judge(list(dict.fromkeys(sources)))[1] if now >= self.unsaved.get(src, 0)]

    def many(self, sources: Iterable[str]) -> dict[str, tuple[str, dict[str, Any] | None]]:
        """Each source text as (state, line): ("ready", {"text", "topics"}) when its line is written; ("rejected",
        None) when the model couldn't write one that keeps the rules (nothing to show, and not asked again while the
        source is the same); ("pending", None) while it waits to be written (or the model was down, until RETRY_S
        has passed). The ones pending go to the front of the queue, in order. Looks up only these, never waits."""
        out, due = self._judge(list(dict.fromkeys(sources)))
        now = self.clock()
        ask = [src for src in due if now >= self.unsaved.get(src, 0)]
        if ask:
            self._queue(ask)
        return out

    def _queue(self, sources: list[str]) -> None:
        with self.lock:
            for source in reversed(sources):  # the newest request first, and in its own order
                if source in self.queued:
                    if source not in self.todo:
                        continue  # being written right now
                    self.todo.remove(source)
                self.todo.appendleft(source)
                self.queued.add(source)
            while len(self.todo) > MAX_QUEUE:  # the oldest requests give way
                self.queued.discard(self.todo.pop())
            self.wake.set()
            if self.background and self.worker is None:
                self.worker = threading.Thread(target=self._run, name=f"plain-{type(self).__name__}", daemon=True)
                self.worker.start()

    def write_one(self, source: str) -> dict[str, Any] | None:
        """Write and cache one line now (the worker's step): asked once more with the rule a reply broke. Returns
        the line, or None."""
        entry: dict[str, Any] = {"at": self.clock(), "line": None}
        try:
            line, why = self.checker(self.rewrite(source))
            if line is None:
                line, why = self.checker(self.rewrite(source + AGAIN.format(why=why)))
            entry["line"] = line
        except Exception as ex:  # the model is down or failed: try again later, not on every read
            LOG.warning(f"rewrite failed: {type(ex).__name__}: {ex}")
            entry["retry_at"] = self.clock() + RETRY_S
        try:
            self._save(source, entry)
        except Exception as ex:  # the store out of reach: hold off this one, or every read would ask again
            LOG.error(f"could not save a line: {ex}")
            self.unsaved[source] = self.clock() + RETRY_S
        return entry["line"]

    def _next(self, wait: float) -> str | None:
        """The next source to write, waiting up to `wait` seconds for one; None when there is none."""
        deadline = time.monotonic() + wait
        while True:
            with self.lock:
                if self.todo:
                    source = self.todo.popleft()
                    return source
                self.wake.clear()
            left = deadline - time.monotonic()
            if left <= 0 or not self.wake.wait(left):
                return None

    def _run(self, wait: float | None = None) -> None:
        wait = self.idle_s if wait is None else wait
        while True:
            source = self._next(wait)
            if source is None:
                with self.lock:  # under the lock, so a source queued now is seen here or starts a new worker
                    if self.todo:
                        continue
                    if threading.current_thread() is self.worker:
                        self.worker = None
                    return
            started = time.monotonic()
            while wait and self.busy() and time.monotonic() - started < VOICE_WAIT_S:  # a voice turn comes first
                time.sleep(1.0)
            try:
                self.write_one(source)
            except Exception as ex:  # one bad source never stops the rest
                LOG.error(f"failed: {type(ex).__name__}: {ex}")
            finally:
                with self.lock:
                    self.queued.discard(source)

    def drain(self) -> None:
        """Write every queued line now, in this thread (the voice wait is skipped)."""
        self._run(wait=0)
