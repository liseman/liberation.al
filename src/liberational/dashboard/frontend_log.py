"""The dashboard frontend's own error log and trace: `POST /frontend/log` appends what the page reports to
`logs/frontend.log`, one JSON object a line, so a failure seen in a browser can be read and fixed later. Local only:
nothing leaves the machine.

The page (frontend/src/shared/telemetry.ts) sends batches of events: errors and unhandled rejections, chunks that
failed to load, error boundaries, failed or slow requests, and slow navigations with the requests and chunks they
waited on. Whatever the page sends, only known fields of the right types are kept, and strings are cut short. Paths
keep no query or hash (a hash like #card=<name> names a person). A body is at most MAX_BODY bytes, at most
MAX_PER_MINUTE events are kept a minute across all pages, and the file rotates once at MAX_FILE bytes.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from liberational import config as cfg

PATH = "/frontend/log"
MAX_BODY = 64 * 1024
MAX_EVENTS = 50  # in one batch
MAX_PER_MINUTE = 120
MAX_FILE = 5 * 1024 * 1024
TYPES = {"error", "rejection", "chunk", "boundary", "request", "navigation"}
TEXT = {"message": 500, "stack": 2000, "name": 80, "method": 8, "build": 40}  # field: longest kept
NUMBERS = ("at", "status", "ms")
MAX_ITEMS = 20  # a navigation's requests or chunks

_lock = threading.Lock()
_kept: deque[float] = deque()


def log_path() -> Path:
    return cfg.path("paths.logs") / "frontend.log"


def _path_only(value: object) -> str | None:
    """A route or request path with no query or hash, or None for anything that isn't an in-app path."""
    if not isinstance(value, str) or not value.startswith("/") or value.startswith("//"):
        return None
    return value.split("?", 1)[0].split("#", 1)[0][:200]


def _number(value: object) -> float | None:
    """A finite number, rounded to a tenth (a status, a duration in ms, a time), or None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return round(number, 1) if math.isfinite(number) and abs(number) < 1e15 else None


def clean(event: object) -> dict[str, Any] | None:
    """The fields of one reported event worth keeping, or None for one that isn't an event."""
    if not isinstance(event, dict) or event.get("type") not in TYPES:
        return None
    out: dict[str, Any] = {"type": event["type"]}
    for key in ("route", "url"):
        if (path := _path_only(event.get(key))) is not None:
            out[key] = path
    for key, longest in TEXT.items():
        if isinstance(value := event.get(key), str) and value:
            out[key] = value[:longest]
    for key in NUMBERS:
        if (number := _number(event.get(key))) is not None:
            out[key] = number
    for key in ("requests", "chunks"):
        items = event.get(key)
        if isinstance(items, list):
            kept = [_item(item) for item in items[:MAX_ITEMS]]
            out[key] = [item for item in kept if item]
    return out


def _item(item: object) -> dict[str, Any] | None:
    """A navigation's request or chunk: its path, and its status and time."""
    if not isinstance(item, dict) or (url := _path_only(item.get("url"))) is None:
        return None
    out: dict[str, Any] = {"url": url}
    for key in ("status", "ms"):
        if (number := _number(item.get(key))) is not None:
            out[key] = number
    return out


def _admit(now: float) -> bool:
    """Whether one more event fits this minute's cap (the caller holds the lock)."""
    while _kept and now - _kept[0] >= 60:
        _kept.popleft()
    if len(_kept) >= MAX_PER_MINUTE:
        return False
    _kept.append(now)
    return True


def append(events: list[Any], path: Path | None = None, now: float | None = None) -> int:
    """Appends the batch's valid events, as far as the minute's cap allows, and returns how many it kept."""
    path = path or log_path()
    now = time.time() if now is None else now
    received = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now))
    lines = []
    with _lock:
        for event in events[:MAX_EVENTS]:
            if (kept := clean(event)) is None or not _admit(now):
                continue
            lines.append(json.dumps({"received": received, **kept}, separators=(",", ":")))
        if not lines:
            return 0
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > MAX_FILE:
            path.replace(path.with_name(path.name + ".1"))  # one older file, so the log can't grow without bound
        with path.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    return len(lines)


def handle(handler: Any) -> bool:
    """`POST /frontend/log` from a page this dashboard served; anything else isn't this module's."""
    if handler.path.split("?", 1)[0] != PATH:
        return False
    if not handler._same_origin_json():
        handler._json({"error": "same-origin JSON only"}, 403)
        return True
    try:
        length = int(handler.headers.get("Content-Length") or 0)
    except ValueError:
        length = -1
    if not 0 < length <= MAX_BODY:
        handler._json({"error": f"a log batch is 1 to {MAX_BODY} bytes"}, 413)
        return True
    try:
        body = json.loads(handler.rfile.read(length))
    except ValueError, UnicodeDecodeError:
        handler._json({"error": "not JSON"}, 400)
        return True
    events = body.get("events") if isinstance(body, dict) else None
    if not isinstance(events, list):
        handler._json({"error": "expected {events: [...]}"}, 400)
        return True
    handler._json({"kept": append(events)})
    return True
