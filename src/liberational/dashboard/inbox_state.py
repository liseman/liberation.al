"""What the owner set aside in the Inbox ("not now"): the faces, voices, and lines skipped, kept in
`inbox_skips.json` (moved out of the server, #61).

GET /inbox/skips  how many of each kind are set aside
"""

import json
import os
import pathlib
from typing import TYPE_CHECKING, Any, TypedDict

from liberational.dashboard import context, routes
from liberational.dashboard.context import Reply
from liberational.people import registry

if TYPE_CHECKING:
    from liberational.dashboard.server import H


def read_json(p: pathlib.Path, default: Any) -> Any:
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


INBOX_SKIPS = (
    context.STATE / "inbox_skips.json"
)  # put aside for good in People's inbox: {"face": [ids], "voice": [ids], "line": [ts]}


class Skips(TypedDict):
    """What People's inbox put aside (inbox_skips)."""

    face: list[float]
    voice: list[float]
    line: list[float | str]  # each by its key (registry.key_of), or by its ts_start as before


def inbox_skips() -> Skips:
    """What People's inbox put aside: face and voice ids, and lines, each by its key (registry.key_of) or,
    for older entries, by its ts_start, which puts aside every line that starts in that millisecond."""
    d = read_json(INBOX_SKIPS, {})

    def numbers(k: str) -> list[float]:
        return [round(float(x), 3) for x in (d.get(k) or []) if isinstance(x, int | float) and not isinstance(x, bool)]

    lines: list[float | str] = []
    for x in d.get("line") or []:
        if isinstance(x, str):
            try:
                slug, at = registry.split_key(x)
            except ValueError:
                continue
            lines.append(x if slug else at)  # a key with no device is its millisecond
        elif isinstance(x, int | float) and not isinstance(x, bool):
            lines.append(round(float(x), 3))
    return {"face": numbers("face"), "voice": numbers("voice"), "line": lines}


def skip_counts(d: Skips) -> dict[str, int]:
    """How many of each kind are put aside, and in all."""
    n = {"face": len(d["face"]), "voice": len(d["voice"]), "line": len(d["line"])}
    return {**n, "total": sum(n.values())}


def inbox_skip(kind: Any, ident: object, control: bool) -> Reply:
    """(code, reply) for POST /inbox/skip: put one face, voice, or line aside for good (it stays out of the inbox and
    its counts across reloads), or bring them all back with kind "clear"."""
    if not control:
        return 403, {"ok": False, "error": "the inbox can only be changed from this machine"}
    if kind == "clear":
        INBOX_SKIPS.unlink(missing_ok=True)
        return 200, {"ok": True, "skipped": 0}
    if kind == "line" and isinstance(ident, str):  # a line by its key; one with no device is its millisecond
        try:
            slug, at = registry.split_key(ident)
        except ValueError:
            return 400, {"ok": False, "error": "a line is put aside by its key, like mic@1761234567.123"}
        ident = ident if slug else at
    keyed = kind == "line" and isinstance(ident, str)
    if (
        kind not in ("face", "voice", "line")
        or not keyed
        and (isinstance(ident, bool) or not isinstance(ident, int | float))
    ):
        return 400, {"ok": False, "error": 'send {"kind": "face" | "voice" | "line", "id": number, or a line key}'}
    d = inbox_skips()
    value: float | str = ident if isinstance(ident, str) else round(float(ident), 3)
    put: list[Any] = d["line"] if kind == "line" else d["face"] if kind == "face" else d["voice"]
    if value not in put:
        put.append(value)
        tmp = INBOX_SKIPS.with_suffix(".tmp")
        tmp.write_text(json.dumps(d))
        os.replace(tmp, INBOX_SKIPS)
    return 200, {"ok": True, "skipped": skip_counts(d)["total"]}


@routes.get("/inbox/skips")
def _get_inbox_skips(h: H) -> None:
    counts = skip_counts(inbox_skips())
    return h._json({k: counts[k] for k in ("face", "voice", "line")})
