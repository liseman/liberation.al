"""The page shell's status on the live feed (#52): the privacy pauses, the services badge, and the Inbox's research
jobs go out on GET /events with the transcript and the camera status, instead of each being polled by every page.

Each open feed holds a `Status`: it reads a kind when the kind's period has passed and reports the ones whose answer
changed. `shared` keeps one read for every feed that asks within the period, so more open pages don't mean more
reads. The same views still answer their own GET routes (/devices/privacy, /services, and /research/jobs), which the
page uses while its stream is down and after a change it made itself.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Hashable
from typing import Any

# How often each kind is read for an open feed. Privacy follows capture applying a pause, which the header icon
# shows as it happens; the other two are a badge and a count.
PERIOD_S: dict[str, float] = {"privacy": 1.0, "services": 15.0, "research": 5.0}
RETRY_S = 2.0  # a view that says it isn't ready ({"ok": false}: the first process scan) is asked again this soon

_lock = threading.Lock()
_kept: dict[Hashable, tuple[float, Any]] = {}


def _waiting(value: object) -> bool:
    """A view that says it isn't ready yet, such as /services before its first process scan."""
    return isinstance(value, dict) and value.get("ok") is False


def shared(key: Hashable, ttl: float, read: Callable[[], Any]) -> Any:
    """`read()`, or the answer another caller got for `key` within the last `ttl` seconds. An answer that wasn't
    ready is kept for half of RETRY_S only, so every feed's retry reads again."""
    now = time.monotonic()
    with _lock:
        kept = _kept.get(key)
        if kept is not None and now - kept[0] < (min(ttl, RETRY_S / 2) if _waiting(kept[1]) else ttl):
            return kept[1]
    value = read()
    with _lock:
        _kept[key] = (time.monotonic(), value)
    return value


class Status:
    """One feed's shell status: `changed()` gives the (kind, view) pairs to send now."""

    def __init__(self, sources: dict[str, Callable[[], Any]]) -> None:
        self.sources = sources
        self.due = dict.fromkeys(sources, 0.0)
        self.sent: dict[str, Any] = {}

    def changed(self, now: float | None = None) -> list[tuple[str, Any]]:
        now = time.monotonic() if now is None else now
        out = []
        for kind, read in self.sources.items():
            if now < self.due[kind]:
                continue
            value = read()
            self.due[kind] = now + (RETRY_S if _waiting(value) else PERIOD_S[kind])
            if kind not in self.sent or self.sent[kind] != value:
                self.sent[kind] = value
                out.append((kind, value))
        return out
