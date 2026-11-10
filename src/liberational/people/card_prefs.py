"""What the owner chose on a person's contact card, and the undo log for People's corrections.

<paths.state>/card_prefs.json holds, per person (keyed by registry.norm of their name):

    main    the sighting id chosen as their main photo ("Use as main photo"), or absent for the best-scoring one
    hidden  {"shot": [sighting ids], "video": ["<cam>@<start>"], "voice": [line keys]}: "Hide from the card"

A rename moves a person's entry to the new name and a merge folds it into the survivor's; deleting a person drops it.

<paths.state>/people_undo.json keeps the newest UNDO_KEEP undo records from the last UNDO_DAYS of People's corrections and assignments
({token, at, kind, name, ops, undone}); the dashboard writes each one and reads it back to undo it. Both files are
small and rewritten whole, each under one lock, since the dashboard serves requests on several threads. Taking an
undo record claims it before execution; failed operations reopen the token with only those operations remaining.
"""

import json
import os
import pathlib
import secrets
import threading
import time
from typing import Any

from liberational.people import registry

PREFS = "card_prefs.json"
UNDO = "people_undo.json"
UNDO_KEEP = 50
UNDO_DAYS = 1  # Undo is for the toast just after a change; older records go
KINDS = ("shot", "video", "voice")
_lock = threading.Lock()


def _read(path: pathlib.Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except OSError, ValueError:
        return default


def _write(path: pathlib.Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)


def _entry(d: dict[str, Any], name: str) -> dict[str, Any]:
    e = d.setdefault(registry.norm(name), {})
    hidden = e.setdefault("hidden", {})
    for k in KINDS:
        hidden.setdefault(k, [])
    return e


def get(state: pathlib.Path, name: str) -> dict[str, Any]:
    """{"main": sighting id | None, "hidden": {"shot": [...], "video": [...], "voice": [...]}} for a person."""
    e = _read(state / PREFS, {}).get(registry.norm(name)) or {}
    hidden = e.get("hidden") or {}
    return {"main": e.get("main"), "hidden": {k: list(hidden.get(k) or []) for k in KINDS}}


def set_main(state: pathlib.Path, name: str, sid: str | None) -> str | None:
    """Make sighting sid the person's main photo (None: back to the best-scoring one). Returns the one before."""
    with _lock:
        d = _read(state / PREFS, {})
        e = _entry(d, name)
        before = e.get("main")
        if sid is None:
            e.pop("main", None)
        else:
            e["main"] = str(sid)  # a detection's payload id
        _write(state / PREFS, d)
    return before


def hide(state: pathlib.Path, name: str, kind: str, ident: Any, hidden: bool = True) -> bool:
    """Hide (or, with hidden False, show again) one shot, video, or voice line on a person's card. Returns whether it
    was hidden before."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    with _lock:
        d = _read(state / PREFS, {})
        items = _entry(d, name)["hidden"][kind]
        was = ident in items
        if hidden and not was:
            items.append(ident)
        elif not hidden and was:
            items.remove(ident)
        _write(state / PREFS, d)
    return was


def rename(state: pathlib.Path, old: str, new: str) -> None:
    """A rename or merge: old's choices go to new. A main photo new already chose is kept, and hidden items join."""
    with _lock:
        d = _read(state / PREFS, {})
        gone = d.pop(registry.norm(old), None)
        if not gone:
            return
        e = _entry(d, new)
        if "main" not in e and gone.get("main") is not None:
            e["main"] = gone["main"]
        for k in KINDS:
            e["hidden"][k] += [i for i in (gone.get("hidden") or {}).get(k) or [] if i not in e["hidden"][k]]
        _write(state / PREFS, d)


def forget(state: pathlib.Path, name: str) -> None:
    with _lock:
        d = _read(state / PREFS, {})
        if d.pop(registry.norm(name), None) is not None:
            _write(state / PREFS, d)


# ---- undo ----
def record(state: pathlib.Path, kind: str, name: str, ops: list[dict[str, Any]]) -> str:
    """Keep the ops that undo one correction or assignment (the newest UNDO_KEEP from the last UNDO_DAYS kept).
    Returns its token."""
    token = secrets.token_urlsafe(9)
    with _lock:
        cutoff = time.time() - UNDO_DAYS * 86400
        log = [r for r in _read(state / UNDO, []) if r.get("at", 0) >= cutoff]
        log.append({"token": token, "at": time.time(), "kind": kind, "name": name, "ops": ops, "undone": False})
        _write(state / UNDO, log[-UNDO_KEEP:])
    return token


def take(state: pathlib.Path, token: str) -> dict[str, Any] | None:
    """Claim token before executing its ops; None when missing, already claimed, or successfully undone."""
    with _lock:
        log = _read(state / UNDO, [])
        for r in log:
            if r.get("token") == token and not r.get("undone"):
                r["undone"] = True
                _write(state / UNDO, log)
                return r
    return None


def retry_failed(state: pathlib.Path, token: str, ops: list[dict[str, Any]]) -> None:
    """Reopen a claimed token with only its failed ops, in their original order, so successes never replay."""
    if not ops:
        return
    with _lock:
        log = _read(state / UNDO, [])
        for r in log:
            if r.get("token") == token and r.get("undone"):
                r["ops"] = ops
                r["undone"] = False
                _write(state / UNDO, log)
                return


def known(state: pathlib.Path, token: str) -> bool:
    """Whether token names an undo record at all (undone or not)."""
    return any(r.get("token") == token for r in _read(state / UNDO, []))
