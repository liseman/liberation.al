"""Calls: the recorded calls, the one on now, and placing or ending one (moved out of the server, #61).

GET  /calls         the recorded calls, newest first (`calls_list`)
GET  /calls/live    the call on now, from the phone bridge's mark
GET  /calls/detail  one call's transcript
GET  /calls/audio   one call's recording
POST /calls/place   place a confirmed call
POST /calls/hangup  end the call on now
"""

import json
import os
import re
import time
from typing import TYPE_CHECKING, Any

from liberational import config as cfg
from liberational.dashboard import routes

if TYPE_CHECKING:
    from liberational.dashboard.server import H


CALLS_DIR = cfg.path("paths.calls")


def _safe_name(n: str) -> bool:
    return bool(n) and len(n) <= 80 and not re.search(r"[\\\\/\x00]", n)


def calls_list(limit: int = 100) -> list[dict[str, Any]]:
    out = []
    for meta in sorted(CALLS_DIR.glob("*/*/call.json"), reverse=True)[:limit]:
        try:
            m = json.loads(meta.read_text())
        except Exception:
            continue
        m["day"], m["dir"] = meta.parent.parent.name, meta.parent.name
        # A call saved before these were kept reads as unknown: no direction, nobody called, no instructions.
        m["direction"] = m.get("direction") if m.get("direction") in ("in", "out") else "unknown"
        for key in ("to", "instructions", "outcome"):
            m.setdefault(key, None)
        out.append(m)
    return out


LIVE_CALL_MAX_S = 6 * 3600  # a live mark older than this is a call a crash left behind, whatever its process


def calls_live(now: float | None = None) -> dict[str, Any]:
    """GET /calls/live: the call on now, from the phone bridge's `live.json` in its folder (phone/bridge.py
    Call.mark_live): {"live": {"direction": "in" | "out", "number", "name" (or null), "since" (epoch seconds),
    "day", "dir"} or null when idle, "count": calls on}. A mark whose bridge process is gone, or older than
    LIVE_CALL_MAX_S, isn't on: a crash leaves its mark behind."""
    now = time.time() if now is None else now
    on: list[tuple[float, dict[str, Any]]] = []
    for path in CALLS_DIR.glob("*/*/live.json"):
        try:
            m = json.loads(path.read_text())
            since, pid = float(m["since"]), int(m["pid"])
        except OSError, ValueError, KeyError, TypeError:
            continue
        if now - since > LIVE_CALL_MAX_S or not _running(pid):
            continue
        on.append(
            (
                since,
                {
                    "direction": m.get("direction") if m.get("direction") in ("in", "out") else "unknown",
                    "number": m.get("number") or None,
                    "name": m.get("name") or None,
                    "since": since,
                    "day": path.parent.parent.name,
                    "dir": path.parent.name,
                },
            )
        )
    on.sort(key=lambda c: -c[0])
    return {"live": on[0][1] if on else None, "count": len(on)}


def calls_hangup(day: str, name: str) -> tuple[int, dict[str, Any]]:
    """End the selected live call, not whichever one happens to be current after a stale click."""
    if not day or not name:
        return 400, {"ok": False, "error": "Choose the live call to end."}
    live = calls_live()["live"]
    if live is None or (live["day"], live["dir"]) != (day, name):
        return 409, {"ok": False, "error": "That call is no longer the active call. Refresh before trying again."}
    # These path components came from the live-call scan, not an unchecked request path.
    try:
        mark = json.loads((CALLS_DIR / live["day"] / live["dir"] / "live.json").read_text())
    except OSError, ValueError:
        return 409, {"ok": False, "error": "That call has already ended."}
    from liberational.phone import outbound

    try:
        ok, detail = outbound.hangup(str(mark.get("callSid") or ""))
    except OSError, ValueError, KeyError:
        return 502, {"ok": False, "error": "Could not confirm hang-up with the phone provider. Try again."}
    return (200 if ok else 409), {"ok": ok, "detail": detail, **({} if ok else {"error": detail})}


def _running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # someone else's process, but running
    return pid > 0


def call_transcript(day: str, name: str) -> dict[str, Any] | None:
    d = CALLS_DIR / day / name
    if not (_safe_name(day) and _safe_name(name)) or not d.is_dir():
        return None
    rows = []
    tp = d / "transcript.jsonl"
    if tp.exists():
        for line in tp.read_text().splitlines():
            try:
                e = json.loads(line)
                e["who"] = e.get("speaker_label") or (cfg.assistant_name() if e.get("speaker") == "bot" else "Caller")
                rows.append(e)
            except Exception:
                pass
    meta = json.loads((d / "call.json").read_text()) if (d / "call.json").exists() else {}
    return {"meta": meta, "transcript": rows}


@routes.get("/calls/audio")
def _get_calls_audio(h: H) -> None:
    q = h._query()
    day, name, which = q.get("day", ""), q.get("dir", ""), q.get("which", "mixed")
    if not (_safe_name(day) and _safe_name(name)) or which not in ("mixed", "bot", "caller"):
        return h._json({"error": "bad call"}, 400)
    return h._wav_file(CALLS_DIR / day / name / f"{which}.wav")


@routes.get("/calls")
def _get_calls(h: H) -> None:
    return h._json(calls_list())


@routes.get("/calls/live")
def _get_calls_live(h: H) -> None:
    return h._json(calls_live())


@routes.get("/calls/detail")
def _get_calls_detail(h: H) -> None:
    q = dict(kv.split("=", 1) for kv in (h.path.split("?", 1) + [""])[1].split("&") if "=" in kv)
    t = call_transcript(q.get("day", ""), q.get("dir", ""))
    return h._json(t or {"error": "not found"}, 200 if t else 404)


@routes.post("/calls/hangup")
def _post_calls_hangup(h: H, body: Any) -> None:
    if not h._same_origin_json():
        return h._json({"error": "same-origin JSON only"}, 403)
    if body.get("confirmed") is not True:
        return h._json({"error": "Confirm the selected call first"}, 400)
    code, result = calls_hangup(str(body.get("day") or ""), str(body.get("dir") or ""))
    return h._json(result, code)


@routes.post("/calls/place")
def _post_calls_place(h: H, body: Any) -> None:
    if not h._same_origin_json():
        return h._json({"error": "same-origin JSON only"}, 403)
    if body.get("confirmed") is not True:
        return h._json({"error": "Confirm the exact call target first"}, 400)
    name, msg = str(body.get("name", "")).strip(), str(body.get("message", "")).strip()[:500]
    number = str(body.get("number", "")).strip()
    if not number:
        return h._json({"error": "confirmed number required"}, 400)
    try:
        from liberational.phone import outbound as outbound_call

        ok, detail = outbound_call.place(name, msg, number=number)
        return h._json({"ok": ok, "detail": detail}, 200 if ok else 409)
    except Exception as ex:
        return h._json({"ok": False, "error": str(ex)}, 500)
