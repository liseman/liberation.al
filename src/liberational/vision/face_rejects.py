"""Shots the owner took out of their unnamed face group ("Not this person" on Who's this?, #2008).

A shot is one `face.match` sighting, by its id. A rejected one keeps its row in the segment store but belongs to no
group: readers that join sightings to groups (vision/sightings.py, people/identity.py) treat its cluster as none, so it
leaves the group's pictures, its photo, and its guesses, and naming the group later doesn't name it. The group's
embedding is rebuilt from the shots it keeps (vision/faces.py `recenter`).

The ids are a small JSON file next to the face groups, `paths.faces/unknown_rejected.json` (`{"ids": [...]}`), read
again only when it changes.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterable
from pathlib import Path

from liberational import config as cfg

MAX_IDS = 100_000  # far more than an owner rejects; the oldest go first past it
_lock = threading.Lock()
_cache: dict[str, tuple[tuple[int, int] | None, frozenset[str]]] = {}


def path() -> Path:
    return cfg.path("paths.faces") / "unknown_rejected.json"


def _stamp(p: Path) -> tuple[int, int] | None:
    try:
        st = p.stat()
    except OSError:
        return None
    return st.st_size, st.st_mtime_ns


def _read(p: Path) -> list[str]:
    try:
        ids = json.loads(p.read_text()).get("ids")
    except OSError, ValueError, AttributeError:
        return []
    return [str(i) for i in ids] if isinstance(ids, list) else []


def rejected() -> frozenset[str]:
    """Every rejected shot id."""
    p = path()
    stamp = _stamp(p)
    got = _cache.get(str(p))
    if got is not None and got[0] == stamp:
        return got[1]
    ids = frozenset(_read(p)) if stamp is not None else frozenset()
    _cache[str(p)] = (stamp, ids)
    return ids


def stamp() -> tuple[int, int] | None:
    """The file's (size, mtime), for caches of what the groups show."""
    return _stamp(path())


def reject(ids: Iterable[str]) -> int:
    """Adds `ids` to the rejected shots; returns how many weren't already."""
    p = path()
    with _lock:
        have = _read(p)
        known = set(have)
        new = [i for i in dict.fromkeys(str(i) for i in ids) if i not in known]
        if new:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps({"ids": [*have, *new][-MAX_IDS:]}))
            os.replace(tmp, p)
    return len(new)
