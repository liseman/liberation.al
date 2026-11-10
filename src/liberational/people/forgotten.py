"""Recognize a person who was forgotten, without keeping their name.

Deleting a person on the dashboard (People: Delete) calls `remember` with every name they had: their name now, their
earlier names, and their aliases (`Registry.names`). Only a keyed one-way HMAC of each name is kept, under a fixed
key (a per-install salt is a follow-up), normalized the way the registry matches names (`registry.norm`). Asked about
one of those names later, the pack answers with the line from Westworld, `LINE`, and gives no evidence and makes no
model call (`named`). The hashes let it recognize a name it is asked about, but nobody can read from them who was
forgotten.

Where the hashes live. The independent knowledge SQLite database (`knowledge.sqlite3` under `paths.state`),
collection `forgotten.names`. Suppression records have deterministic HMAC digest IDs and no person subjects, so
forgetting registry events cannot erase the suppression. Existing broker or JSON hashes are not migrated.

Which names a question names. The people history_qa's parser finds ("what did Dana say", "when was Dana last seen"),
the words after "who's", "what about", and the like, and each capitalized word or pair of words. A capitalized word
that opens a sentence and is a common opener ("Will", "May") is skipped, as are month and weekday names, so a
forgotten Will or May fires only where the parser or "who's" found the name. The owner's and the assistant's names
never fire, and neither does a name a living person has now (someone new can be called Dana).

`assistant.forgotten_reply`, on by default, keeps those hashes and lets the line fire. Off keeps no hash and the line
never fires.
"""

import hashlib
import hmac
import pathlib
import re
import threading
from collections.abc import Iterable

from liberational import config as cfg
from liberational import knowledge
from liberational.people import registry

LINE = "Doesn't look like anything to me."

WORD = re.compile(r"[^\W\d_][\w'’-]*")
POSSESSIVE = re.compile(r"(?:['’]s|['’])$", re.I)
ABOUT = re.compile(
    r"\b(?:who(?:['’]s|\s+is|\s+was)|what\s+about|how\s+about|tell\s+me\s+about|know\s+about|remember)\s+"
    r"([^\W\d_][\w'’-]*)(?:\s+([^\W\d_][\w'’-]*))?",
    re.I,
)
# A capitalized word that is one of these opens a sentence, or names a day or a month, rather than a person
OPENER = re.compile(
    r"who|what|when|where|why|how|which|whose|whom|did|do|does|is|was|were|are|am|has|have|had|can|could|will|would|"
    r"should|shall|may|might|must|tell|show|give|list|remind|find|and|but|so|or|hey|hi|ok|okay|please|i|the|an?|any|"
    r"anyone|anybody|someone|somebody|there|this|that|these|those|it|yes|no",
    re.I,
)
CALENDAR = re.compile(
    r"january|february|march|april|may|june|july|august|september|october|november|december|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday",
    re.I,
)


BROKER_KEY = b"liberational.forgotten"
"""The existing fixed HMAC key, retained across the storage cut so normalized names hash identically.
A per-install salt is a follow-up; the database holds only these hashes, never names."""

COLLECTION = "forgotten.names"


def enabled() -> bool:
    return bool(cfg.get("assistant.forgotten_reply"))


def broker_digest(name: str) -> bytes:
    """A name's stable suppression hash: HMAC-SHA256 of its normalized form under `BROKER_KEY`."""
    return hmac.new(BROKER_KEY, registry.norm(name).encode(), hashlib.sha256).digest()


class SQLiteStore:
    """Hash-only name suppression outside the registry's person erasure scope."""

    def __init__(self, backend: knowledge.Store) -> None:
        self.backend = backend

    def remember(self, keys: set[str]) -> int:
        """Keep normalized name hashes; return how many were not already present."""
        new = 0
        for digest in sorted({broker_digest(key).hex() for key in keys}):
            if self.backend.get(digest) is None:
                # Immutable content must also be identical across processes and retries. No event clock,
                # raw name, or person subject belongs to this durable suppression set.
                self.backend.put([knowledge.Record(digest, COLLECTION, digest, 0, 0, 0, {})])
                new += 1
        return new

    def has(self, key: str) -> bool:
        row = self.backend.get(broker_digest(key).hex())
        return row is not None and row.record.collection == COLLECTION


_stores: dict[pathlib.Path, SQLiteStore] = {}
_store_lock = threading.Lock()


def store(state: str | pathlib.Path | None = None) -> SQLiteStore:
    """One adapter per install; every operation reads the SQLite source of truth."""
    path = (pathlib.Path(state) if state is not None else cfg.path("paths.state")) / "knowledge.sqlite3"
    path = path.resolve()
    with _store_lock:
        if path not in _stores:
            _stores[path] = SQLiteStore(knowledge.Store(path))
        return _stores[path]


def reset() -> None:
    """Drop cached adapters without deleting suppression; the next call reopens its install's database."""
    with _store_lock:
        _stores.clear()


def remember(names: Iterable[str], state: str | pathlib.Path | None = None) -> int:
    """Keep name hashes in the independent knowledge store; return how many were new. Off keeps nothing."""
    keys = {registry.norm(n) for n in names} - {""}
    if not enabled() or not keys:
        return 0
    return store(state).remember(keys)


def _ignored() -> set[str]:
    """The owner's and the assistant's names, whole and each word of them: never a forgotten person."""
    names = {cfg.owner_name(), cfg.owner_label(), cfg.assistant_name()}
    return {registry.norm(n) for n in names} | {w for n in names for w in registry.norm(n).split()}


def candidates(question: str, people: Iterable[str] = ()) -> set[str]:
    """The normalized names `question` may name: `people` (from history_qa's parser), the words after "who's" and the
    like, and each run of adjacent capitalized words, whole. A word inside a longer run counts only as part of it."""
    found = [str(p) for p in people if p]
    for m in ABOUT.finditer(question):
        first = POSSESSIVE.sub("", m[1])
        found += [first, f"{first} {POSSESSIVE.sub('', m[2])}"] if m[2] else [first]
    words = list(WORD.finditer(question))
    caps: list[tuple[int, str]] = []
    for i, m in enumerate(words):
        word = POSSESSIVE.sub("", m[0])
        opens = question[: m.start()].rstrip()[-1:] in ("", ".", "!", "?")
        if word[:1].isupper() and not CALENDAR.fullmatch(word) and not (opens and OPENER.fullmatch(word)):
            caps.append((i, word))
    runs: list[list[str]] = []  # adjacent capitalized words, separated only by spaces: one name, any length
    for (i, w), prev in zip(caps, [None, *caps]):
        joined = prev is not None and prev[0] == i - 1 and question[words[i - 1].end() : words[i].start()].isspace()
        if joined:
            runs[-1].append(w)
        else:
            runs.append([w])
    found += [" ".join(run) for run in runs]
    # A full name such as "Dana White" is someone; its first name alone ("Dana") is not who was asked about.
    within = {registry.norm(w) for run in runs if len(run) > 1 for w in run}
    return {registry.norm(n) for n in found} - {""} - within - _ignored()


def named(question: str, people: Iterable[str] = (), state: str | pathlib.Path | None = None) -> bool:
    """Does `question` name a person who was forgotten? `people` are the names history_qa's parser found in it."""
    if not enabled():
        return False
    kept = store(state)
    return any(kept.has(name) and not _living(name, state) for name in candidates(question, people))


def _living(name: str, state: str | pathlib.Path | None) -> bool:
    """Does someone the registry knows have this name now? A failure to read it counts as nobody."""
    try:
        return registry.at(state).known_as(name) is not None
    except Exception as e:
        registry.LOG.warning(f"forgotten: registry lookup failed: {e!r}")
        return False
