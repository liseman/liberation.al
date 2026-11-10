"""Shots the owner took out of their unnamed face group on Who's this?: ignored or blocked ones, which leave it,
and deselected ones, which move to a group of their own.

A shot is one `face.match` sighting, by its id. A rejected one keeps its row in the segment store but belongs to no
group: readers that join sightings to groups (vision/sightings.py, people/identity.py) treat its cluster as none, so it
leaves the group's pictures, its photo, and its guesses, and naming the group later doesn't name it. A moved one is
read as a member of the group it was moved to (`group_of`), so the pictures the owner left out of an answer wait
together for their own. The group's embedding is rebuilt from the shots it keeps (vision/faces.py `recenter`).

The ids are a small JSON file next to the face groups, `paths.faces/unknown_rejected.json`
(`{"ids": [...], "moved": {id: group}}`), read again only when it changes. What the owner ignored or blocked is also
kept as taught state (people/face_hidden.py); this file is how the readers apply it.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from liberational import config as cfg
from liberational import json_state

MAX_IDS = 100_000  # far more than an owner rejects or moves; the oldest go first past it
_lock = threading.Lock()
_cache: dict[str, tuple[tuple[int, int] | None, frozenset[str], Mapping[str, int]]] = {}


def path() -> Path:
    return cfg.path("paths.faces") / "unknown_rejected.json"


def _stamp(p: Path) -> tuple[int, int] | None:
    try:
        st = p.stat()
    except OSError:
        return None
    return st.st_size, st.st_mtime_ns


def _read(p: Path) -> tuple[list[str], dict[str, int]]:
    try:
        d: Any = json.loads(p.read_text())
        ids, moved = d.get("ids"), d.get("moved")
    except OSError, ValueError, AttributeError:
        return [], {}
    return (
        [str(i) for i in ids] if isinstance(ids, list) else [],
        {str(k): v for k, v in moved.items() if type(v) is int} if isinstance(moved, dict) else {},
    )


def _load() -> tuple[frozenset[str], Mapping[str, int]]:
    p = path()
    stamp = _stamp(p)
    got = _cache.get(str(p))
    if got is not None and got[0] == stamp:
        return got[1], got[2]
    ids, moved = _read(p) if stamp is not None else ([], {})
    _cache[str(p)] = (stamp, frozenset(ids), MappingProxyType(moved))
    return _cache[str(p)][1], _cache[str(p)][2]


def rejected() -> frozenset[str]:
    """Every rejected shot id."""
    return _load()[0]


def moved() -> Mapping[str, int]:
    """Every moved shot's group: {shot id: the face group it was moved to}."""
    return _load()[1]


def group_of(shot: object, cluster: object) -> object:
    """A shot's face group label as the owner left it: none for a rejected shot, the group a moved one went to
    ("Face N"), else `cluster`, the one the face worker gave it. A shot with no group has none to change."""
    if not cluster:
        return cluster
    ids, went = _load()
    key = str(shot)
    if key in ids:
        return None
    return f"Face {went[key]}" if key in went else cluster


def stamp() -> tuple[int, int] | None:
    """The file's (size, mtime), for caches of what the groups show."""
    return _stamp(path())


def _write(p: Path, ids: list[str], went: dict[str, int]) -> None:
    keep = dict(list(went.items())[-MAX_IDS:])
    json_state.write(p, {"ids": ids[-MAX_IDS:], "moved": keep}, durable=True)  # the owner's choices


def reject(ids: Iterable[str]) -> int:
    """Adds `ids` to the rejected shots; returns how many weren't already."""
    p = path()
    with _lock:
        have, went = _read(p)
        known = set(have)
        new = [i for i in dict.fromkeys(str(i) for i in ids) if i not in known]
        if new:
            _write(p, [*have, *new], went)
    return len(new)


def restore(ids: Iterable[str]) -> int:
    """Takes `ids` off the rejected shots, so each is in its group again; returns how many were rejected."""
    p = path()
    back = {str(i) for i in ids}
    with _lock:
        have, went = _read(p)
        kept = [i for i in have if i not in back]
        if len(kept) != len(have):
            _write(p, kept, went)
    return len(have) - len(kept)


def move(ids: Iterable[str], group: int) -> int:
    """Moves `ids` to face group `group`: they are read as its members from now on. Returns how many."""
    p = path()
    new = list(dict.fromkeys(str(i) for i in ids))
    with _lock:
        have, went = _read(p)
        if new:
            _write(p, have, {**{k: v for k, v in went.items() if k not in new}, **dict.fromkeys(new, int(group))})
    return len(new)
