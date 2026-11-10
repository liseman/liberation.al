"""Ingest sources: one small interface for everywhere media comes from.

A source can:
- `status()`: whether it's set up, and if not, what's missing, in words for the Ingest tab;
- `list_new(state)`: the items it has that `state` (its record in the ingest store) hasn't seen yet;
- `fetch(item, into)`: put one item's file in the directory `into` and return its path.

A one-off source (link, google_photos) is driven by the person picking what to bring in; an ongoing one (wyze) is
polled by `pack-ingest poll` (runner.py). A source that isn't set up, or whose data can't be reached yet, raises
NotConfigured with a message that says so. `SOURCES` lists every source by name; `make(name)` builds one from config.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from liberational.ingest.store import Store


class NotConfigured(Exception):
    """The source isn't set up (or can't be yet); the message says what's missing."""


@dataclass(frozen=True)
class RemoteItem:
    """One thing a source has: its own stable `key`, a file name, when the source says it was made, and whatever the
    source needs to fetch it."""

    key: str
    name: str
    taken_at: float | None = None
    url: str | None = None
    stream: str | None = None  # the stream it goes on, when the source picks one (a Wyze camera's)
    extra: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def public(self) -> dict[str, Any]:
        return {"key": self.key, "name": self.name, "taken_at": self.taken_at, "url": self.url, **self.extra}


class SourceState:
    """A source's view of the ingest store: which of its keys it has handed over, and a JSON blob of its own."""

    def __init__(self, store: Store, source: str) -> None:
        self.store, self.source = store, source

    def seen(self, key: str) -> bool:
        return self.store.seen(self.source, key)

    def mark(self, key: str, item_id: int | None = None) -> None:
        self.store.mark_seen(self.source, key, item_id)

    def count(self) -> int:
        return self.store.seen_count(self.source)

    def get(self) -> dict[str, Any]:
        return self.store.source_state(self.source)

    def put(self, data: dict[str, Any]) -> None:
        self.store.put_source_state(self.source, data)


@dataclass(frozen=True)
class Status:
    configured: bool
    message: str  # for the owner, in plain words
    blocked: bool = False  # setting it up can't go on from the dashboard (a package missing): its sign-in waits
    fix: str | None = None  # what whoever runs this machine does about it (a command), for `pack-ingest sources`
    later: bool = False  # no way to bring it in yet (a stub): Settings lists it under "Coming later", not as a setup


class Source(Protocol):
    @property
    def name(self) -> str: ...  # its key in SOURCES and in the ingest store

    @property
    def label(self) -> str: ...  # what the Ingest tab calls it

    @property
    def ongoing(self) -> bool: ...  # polled by pack-ingest poll (else: driven by hand)

    def status(self) -> Status: ...

    def list_new(self, state: SourceState) -> list[RemoteItem]: ...

    def fetch(self, item: RemoteItem, into: Path) -> Path: ...


def _link() -> Source:
    from liberational.ingest.sources.link import LinkSource

    return LinkSource.from_config()


def _wyze() -> Source:
    from liberational.ingest.sources.wyze import WyzeSource

    return WyzeSource.from_config()


def _google_photos() -> Source:
    from liberational.ingest.sources.google_photos import GooglePhotosSource

    return GooglePhotosSource.from_config()


def _stub(name: str) -> Callable[[], Source]:
    def make() -> Source:
        from liberational.ingest.sources.stubs import STUBS

        return STUBS[name]

    return make


# Every source, in the order the Ingest tab lists them.
SOURCES: dict[str, Callable[[], Source]] = {
    "link": _link,
    "wyze": _wyze,
    "google_photos": _google_photos,
    "icloud_photos": _stub("icloud_photos"),
    "allsound": _stub("allsound"),
    "thinking_hat": _stub("thinking_hat"),
}


def make(name: str) -> Source:
    """The source called `name`, built from config; KeyError for one that doesn't exist."""
    return SOURCES[name]()
