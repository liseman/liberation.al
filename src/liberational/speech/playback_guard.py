"""Cross-process playback exclusion before ASR, enrollment and authorization.
No text/audio stored. Human barge-in overlapping playback is also excluded.
"""

import contextlib
import fcntl
import json
import math
import os
import pathlib
import time
import uuid
from collections.abc import Callable, Generator
from typing import TypedDict

from liberational import config as cfg

PATH = cfg.path("paths.state") / "playback_windows.json"
TAIL = 0.6
RETENTION = 86400


class Window(TypedDict):
    """A playback window as stored in the file. Times are epoch seconds."""

    id: str
    start: float
    end: float


def _read(path: pathlib.Path) -> list[Window]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return []


def _update(path: pathlib.Path, change: Callable[[list[Window]], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = [r for r in _read(path) if r["end"] >= time.time() - RETENTION]
        change(rows)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows))
        os.replace(tmp, path)


@contextlib.contextmanager
def playback_window(duration: float, path: str | os.PathLike[str] | None = None) -> Generator[None]:
    path = PATH if path is None else pathlib.Path(path)
    token = uuid.uuid4().hex
    start = time.time()
    # Covers player startup/timeout even if writer crashes; history covers delayed ASR.
    _update(
        path, lambda rows: rows.append(dict(id=token, start=start - 0.1, end=start + max(15, duration + 10) + TAIL))
    )
    try:
        yield
    finally:

        def finish(rows: list[Window]) -> None:
            for row in rows:
                if row["id"] == token:
                    row["end"] = time.time() + TAIL

        _update(path, finish)


def open_window(duration: float, path: str | os.PathLike[str] | None = None) -> str:
    """Open a window for audio the dashboard is about to play. Returns a token to close it with close_window."""
    path = PATH if path is None else pathlib.Path(path)
    token = uuid.uuid4().hex
    start = time.time()
    _update(path, lambda rows: rows.append(dict(id=token, start=start - 0.1, end=start + duration + TAIL)))
    return token


def close_window(token: str | None, path: str | os.PathLike[str] | None = None) -> None:
    """End a window opened by open_window right now (plus the tail), e.g. when playback stops early."""
    path = PATH if path is None else pathlib.Path(path)
    if not token:
        return

    def finish(rows: list[Window]) -> None:
        for row in rows:
            if row["id"] == token:
                row["end"] = min(row["end"], time.time() + TAIL)

    _update(path, finish)


def overlaps(
    start: float | str | None, end: float | str | None = None, path: str | os.PathLike[str] | None = None
) -> bool:
    if start is None:
        return False  # manual text, not captured audio
    try:
        start = float(start)
        end = start if end is None else float(end)
        if not (math.isfinite(start) and math.isfinite(end)):
            return True  # NaN compares False with every window, so it would pass as clear: fail closed like "abc"
        return any(
            start <= row["end"] and end >= row["start"] for row in _read(PATH if path is None else pathlib.Path(path))
        )
    except OSError, ValueError, TypeError, KeyError:
        return True
