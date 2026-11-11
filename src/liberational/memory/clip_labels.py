"""Short labels for the contact card's recent video clips: "Sitting on a chair, hand over mouth, glasses on."

A label says what is happening in the clip with no subject, in the present tense: no "I", "my", or "me" (the journal
covers a wider stretch, so its text is never used), no "the person", no pronoun, and no name (the card shows whose clip it
is). At most MAX_CHARS characters.

It is written from the camera's captions during the clip (the Continuum vision log, third person), the real caption
nearest the clip's middle (repeats for an unchanged view are skipped). The local model rewrites that one
caption into the label (one short call, `journal.model`, else `models.ollama.llm`), and clean() checks the result
against the rules above; one that breaks them is dropped, never shown. The card never waits for the model: label()
returns what is cached, and queues the clip for one background thread when nothing is. A clip with no label (not
written yet, no caption, the model down, or a rewrite that broke the rules) gets None, and the card shows only its
time and place.

Labels are cached in the process, MAX_CACHED clips at most (the oldest written goes first): one per clip (camera and
start), with the hash of the caption it came from, so a clip is rewritten again only when its caption changes. A
rewrite that broke the rules is cached with no label, so it isn't asked for again; a model that was down is tried
again after RETRY_S. A restart starts the cache empty, and labels are written again as cards ask for them.
"""

import hashlib
import pathlib
import queue
import re
import threading
import time
from collections.abc import Callable
from typing import Any

from liberational import config as cfg
from liberational import logs, vision_log

MAX_CHARS = 60
WINDOW_S = 15.0  # captions this far either side of a clip count as its own
SETTLED_S = 60.0  # a clip whose caption window ended this long ago gets no new captions: its lookup is kept
KEEP_S = 600.0  # for this long, so a card opened again doesn't read the segment store again
KEPT = 512  # lookups kept at most
_kept: dict[tuple[str, str, float, float], tuple[float, dict[str, Any] | None]] = {}
_kept_lock = threading.Lock()
RETRY_S = 600.0  # a clip whose rewrite failed because the model was down is tried again after this long
MAX_CACHED = 5000  # clips with a cached label; a card shows a few dozen, so this is weeks of browsing

LOG = logs.get("clip-labels")

SYSTEM = (
    "You write a very short label for a video clip, from a camera caption of it. Describe what is happening, as a "
    "present-tense phrase with no subject: start with a verb ending in -ing or a short noun phrase. Never use I, me, "
    "my, he, she, they, his, her, their, the person, the man, the woman, or anyone's name. If the caption has "
    "several people and you can't tell which one matters, describe the scene briefly instead. At most 60 "
    "characters. Reply with the label alone.\n"
    'Example caption: "Foreground: man, seated, facing camera, hand over mouth, wearing glasses."\n'
    'Example label: "Sitting, hand over mouth, glasses on."'
)

FIRST_PERSON = re.compile(r"\b(i|me|my|mine|myself|we|us|our|ours)\b|\bi'(m|ve|d|ll)\b", re.IGNORECASE)
SUBJECT = re.compile(
    r"\b(he|she|they|him|her|them|his|hers|their|theirs|himself|herself|themselves)\b|"
    r"\bthe (person|man|woman|guy|lady|people)\b|"
    # a label that starts with who, bare ("Man seated", "A woman pouring", "Someone sitting"); "Two people talking"
    # describes the scene and stays
    r"^(an? |the )?(man|woman|person|boy|girl|child|kid|guy|lady|someone|somebody|individual|figure)\b",
    re.IGNORECASE,
)
LEAD = re.compile(r"^((label|caption|foreground|background)( person)?\s*:\s*)+", re.IGNORECASE)
QUOTES = "\"'`\u201c\u201d\u2018\u2019*_ "  # straight and curly quotes, backticks, and markdown emphasis


def _key(cam: str, start: float) -> str:
    return f"{cam}@{start:.2f}"


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def clean(text: str | None, name: str | None = None) -> str | None:
    """The model's reply as a label, or None when it breaks a rule: first person, a pronoun or "the person", the
    person's name (any part of it), or longer than MAX_CHARS. Quotes, a "Label:" lead, and extra space go, the first
    letter is a capital, and it ends with a full stop. Only the reply's first line counts (a note after it goes)."""
    lines = [line for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return None
    t = " ".join(lines[0].split()).strip(QUOTES)
    t = LEAD.sub("", t).strip(QUOTES)
    if not t or FIRST_PERSON.search(t) or SUBJECT.search(t):
        return None
    for part in re.findall(r"[^\W\d_]{2,}", name or ""):
        if re.search(rf"\b{re.escape(part)}\b", t, re.IGNORECASE):
            return None
    t = t.rstrip(" .,;:!") + "."
    if len(t) > MAX_CHARS or len(t) < 4:
        return None
    return t[0].upper() + t[1:]


def caption(cam: str, start: float, end: float, state: pathlib.Path) -> str | None:
    """The real caption of `cam` nearest the clip's middle, within WINDOW_S of the clip, or None."""
    found = caption_record(cam, start, end, state)
    return None if found is None else " ".join(found["caption"].split())


def caption_record(cam: str, start: float, end: float, state: pathlib.Path) -> dict[str, Any] | None:
    """The `camera.caption` record `caption` reads, with its id, or None. A settled clip's answer is kept for KEEP_S."""
    key, now = (str(state), cam, start, end), time.time()
    with _kept_lock:
        hit = _kept.get(key)
    if hit is not None and now - hit[0] < KEEP_S:
        return hit[1]
    mid = (start + end) / 2
    try:
        records = vision_log.at(state).captions(start - WINDOW_S, end + WINDOW_S, (cam,))
    except OSError, vision_log.continuum_py.ContinuumError:
        return None  # unknown, not none: never kept
    real = [r for r in records if vision_log.is_real(r)]
    found = min(real, key=lambda r: abs(r["ts"] - mid)) if real else None
    if end + WINDOW_S + SETTLED_S <= now:
        with _kept_lock:
            _kept[key] = (now, found)
            while len(_kept) > KEPT:
                del _kept[next(iter(_kept))]
    return found


def _rewrite(text: str) -> str:
    """One local model call: the caption rewritten as a label (unchecked)."""
    from liberational.memory import journal

    return journal.chat(SYSTEM, f'Caption: "{text}"\nLabel:', journal.model_name(), 40)


class Labels:
    """The cache and its one background writer. rewrite is the model call; a test passes its own, and background
    False to run the queue itself with drain()."""

    def __init__(
        self,
        state: pathlib.Path,
        rewrite: Callable[[str], str] = _rewrite,
        background: bool = True,
        clock: Callable[[], float] = time.time,
        idle_s: float = 30.0,
    ) -> None:
        self.state = pathlib.Path(state)
        self.clock = clock
        self.idle_s = idle_s  # the worker stops after this long with nothing queued
        self.rewrite = rewrite
        self.background = background
        self.lock = threading.Lock()
        self.todo: queue.Queue[tuple[str, float, float, str | None]] = queue.Queue()
        self.queued: set[str] = set()
        self.worker: threading.Thread | None = None
        self.retry_at: dict[str, tuple[str, float]] = {}  # a clip whose rewrite failed: (its caption's hash, when)
        self.written: dict[str, tuple[str, str | None]] = {}  # {clip key: (its caption's hash, label or None)}

    def get(self, cam: str, start: float, end: float, name: str | None = None) -> str | None:
        """The clip's label if one is cached for its current caption, else None, with the clip queued to write one.
        Never waits on the model."""
        text = caption(cam, start, end, self.state)
        if text is None:
            return None
        key, h = _key(cam, start), _hash(text)
        with self.lock:
            hit = self.written.get(key)
            failed = self.retry_at.get(key)
        if hit is not None and hit[0] == h:
            return clean(hit[1], name)  # checked again: the name may have changed since
        if failed is not None and failed[0] == h and self.clock() < failed[1]:
            return None  # the model was down a moment ago
        self._queue(key, cam, start, end, name)
        return None

    def _queue(self, key: str, cam: str, start: float, end: float, name: str | None) -> None:
        with self.lock:
            if key in self.queued:
                return
            self.queued.add(key)
            self.todo.put((cam, start, end, name))
            if self.background and self.worker is None:
                self.worker = threading.Thread(target=self._run, name="clip-labels", daemon=True)
                self.worker.start()

    def write_one(self, cam: str, start: float, end: float, name: str | None) -> str | None:
        """Write and cache one clip's label now (the worker's step). Returns it, or None."""
        found = caption_record(cam, start, end, self.state)
        if found is None:
            return None
        text = " ".join(found["caption"].split())
        try:
            label = clean(self.rewrite(text), name)
        except Exception as ex:  # the model is down or failed: try again later, not on every card load
            LOG.warning(f"rewrite failed: {type(ex).__name__}: {ex}")
            with self.lock:
                self.retry_at[_key(cam, start)] = (_hash(text), self.clock() + RETRY_S)
            return None
        # A label that broke the rules is cached too, as None, so it isn't asked for again for this caption.
        key = _key(cam, start)
        with self.lock:
            self.written.pop(key, None)  # a rewrite moves the clip to the newest end
            self.written[key] = (_hash(text), label)
            while len(self.written) > MAX_CACHED:
                del self.written[next(iter(self.written))]
        return label

    def _run(self, wait: float | None = None) -> None:
        wait = self.idle_s if wait is None else wait
        while True:
            try:
                cam, start, end, name = self.todo.get(timeout=wait) if wait else self.todo.get_nowait()
            except queue.Empty:
                with self.lock:  # under the lock, so a clip queued now either is seen here or starts a new worker
                    if self.todo.empty():
                        if threading.current_thread() is self.worker:
                            self.worker = None
                        return
                continue
            try:
                self.write_one(cam, start, end, name)
            except Exception as ex:  # never let one clip stop the rest
                LOG.error(f"failed: {type(ex).__name__}: {ex}")
            finally:
                with self.lock:
                    self.queued.discard(_key(cam, start))

    def drain(self) -> None:
        """Write every queued clip's label now, in this thread."""
        self._run(wait=0)


_labels: dict[str, Labels] = {}
_made = threading.Lock()


def at(state: pathlib.Path | None = None) -> Labels:
    """The labels of a state directory (one per process, so one background writer)."""
    path = pathlib.Path(state or cfg.path("paths.state"))
    with _made:
        if str(path) not in _labels:
            _labels[str(path)] = Labels(path)
        return _labels[str(path)]
