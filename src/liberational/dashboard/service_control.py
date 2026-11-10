"""Settings' Services: what each service is doing, and turning one on or off (moved out of the server, #61).

GET  /services          every service for this platform and whether this viewer may switch them (`services_view`)
GET  /services/history  lifecycle history over 24 hours or 7 days
POST /services          {"name", "action"} or {"all"}: this machine only, same-origin JSON, one action at a time
"""

import subprocess
import threading
from typing import TYPE_CHECKING, Any

from liberational import config as cfg
from liberational import service_cascade, services
from liberational.config import ROOT
from liberational.dashboard import context, routes
from liberational.dashboard.context import LOG, PY, Reply, log

if TYPE_CHECKING:
    from liberational.dashboard.server import H


def services_view(control: bool, fresh: bool = False) -> dict[str, Any]:
    """The admin sidebar's view: every service for this platform, and whether this viewer may switch them. The
    actions ask for fresh pids, so a service started or stopped by hand a moment ago counts."""
    try:
        return {"ok": True, "control": control, "platform": cfg.platform_key(), "services": services.snapshot(fresh)}
    except services.ProcessScanPending as ex:
        return {"ok": False, "control": control, "error": str(ex), "services": []}
    except SystemExit as ex:  # pack.toml names a service that doesn't exist
        return {"ok": False, "control": control, "error": str(ex), "services": []}


def service_history_view(hours: str = "24") -> Reply:
    """Read-only lifecycle history, with one of the two bounded windows chosen by the UI."""
    if hours not in {"24", "168"}:
        return 400, {"error": "History must cover 24 hours or 7 days."}
    try:
        return 200, dict(services.history(context.STATE, int(hours)))
    except (OSError, ValueError) as ex:
        LOG.warning(f"Service history could not be read: {type(ex).__name__}")
        return 503, {"error": "Could not read service history."}


# What each action on one service does: its pack-services command, and the words its log line says it in.
SERVICE_ACTIONS: dict[str, tuple[str, str]] = {
    "on": ("install", "installed and started; runs at every login"),
    "off": ("uninstall", "uninstalled; stays off until turned on"),
    "stop": ("stop", "stopped now; starts again when the computer restarts"),
    "start": ("start", "started"),
}


def _logged(action: str, what: str, why: str, ok: bool | None, out: str) -> None:
    """One INFO line per service action from the dashboard, done or not, so the log says what changed."""
    lines = out.strip().splitlines()
    bad = next((line for line in lines if "FAILED" in line), lines[-1] if lines else "?")  # the error, not a note after
    result = "done" if ok else "timed out" if ok is None else f"failed: {bad}"
    log(f"services: {action} {what} ({why}): {result}")


# Held around each POST /services action: Enable and Disable read and rewrite pack.toml in a pack-services process, and
# two at once would lose one's change; the others mustn't race an install or uninstall of the same units either.
SERVICES_LOCK = threading.Lock()


# Enable and Disable: pack.toml's [services] lists first, so the choice outlasts a restart, then the service itself.
SERVICE_CHOICES: dict[str, tuple[str, str]] = {
    "enable": ("install", "enabled for this machine, installed, and started"),
    "disable": ("uninstall", "disabled for this machine and stopped; stays off after a restart"),
}


def service_action(name: str, action: str) -> Reply:
    """(code, body): one service's action: on (pack-services install), off (uninstall: stays off), stop (stopped now,
    still installed, so it starts again when the computer restarts, or on Repair), start (an installed one that was
    stopped), enable (one not enabled here: pack.toml's [services] names it, and it's installed), or disable (the
    reverse: taken out of pack.toml's choice and uninstalled, so it stays off after a restart)."""
    if action not in SERVICE_ACTIONS and action not in SERVICE_CHOICES:
        return 400, {
            "ok": False,
            "error": 'send {"name": ..., "action": "on", "off", "stop", "start", "enable", or "disable"}',
        }
    svc = next((s for s in services_view(True, fresh=True)["services"] if s["name"] == name), None)
    if svc is None or (not svc["selected"] and action != "enable"):
        return 400, {
            "ok": False,
            "error": f"{name or '?'} is not enabled for this machine (see [services] in pack.toml)",
        }
    if name == "dashboard":
        return 400, {"ok": False, "error": "this page is served by the dashboard; use pack-services in a terminal"}
    if action == "enable" and svc["selected"]:
        return 409, {"ok": False, "error": f"{name} is already enabled for this machine"}
    if action == "disable" and svc.get("required"):
        return 409, {"ok": False, "error": f"{name} is needed by another setting (asr.engine), so it stays on"}
    if action in ("on", "enable") and not svc["installed"] and svc["pids"]:
        return 409, {
            "ok": False,
            "error": f"{name} is already running outside the service manager (pid {svc['pids'][0]})",
        }
    if action in ("stop", "start") and not svc["installed"]:
        return 409, {"ok": False, "error": f"{name} isn't installed: turn it on instead"}
    if action in SERVICE_CHOICES:
        command, why = SERVICE_CHOICES[action]
        # Back to its default where that is the choice, so pack.toml names only what differs from services.toml.
        default = bool(svc.get("default"))
        listed = ("reset" if default else "enable") if action == "enable" else ("disable" if default else "reset")
        steps = [[listed, name], [command, name]]
    else:
        command, why = SERVICE_ACTIONS[action]
        steps = [[command, name]]
    ok, outs = True, []
    for step in steps:
        # The outer process must outlive the cascade's own readiness and recovery deadlines.
        timeout = (
            service_cascade.subprocess_timeout()
            if step[0] == "install" and name in {"continuum-broker", "continuum-capture"}
            else 120
        )
        ok, out = _pack_services(step, timeout)
        outs.append(out)
        if step[0] in ("enable", "disable", "reset"):
            cfg.load.cache_clear()  # pack-services wrote pack.toml; read it again here
        if not ok:
            break
    out = "\n".join(outs)
    _logged(action, name, why, ok, out)
    if ok is None:
        return 504, {
            **services_view(True, fresh=True),
            "ok": False,
            "error": f"pack-services timed out: {action} {name}",
        }
    return (200 if ok else 500), {
        **services_view(True, fresh=True),
        "ok": ok,
        "out": out[-2000:],
        "error": None if ok else out[-500:],
    }


def _pack_services(args: list[str], timeout: float) -> tuple[bool | None, str]:
    """Run `pack-services <args>` from this checkout: (worked, output), with worked None on a timeout."""
    try:
        r = subprocess.run(
            [PY, "-m", "liberational.services", *args], capture_output=True, text=True, timeout=timeout, cwd=ROOT
        )
    except subprocess.TimeoutExpired:
        return None, ""
    out = (r.stdout + r.stderr).strip()
    return r.returncode == 0 and "FAILED" not in out, out


def all_services(action: str) -> Reply:
    """(code, body): every service but the dashboard at once. "repair" reinstalls each service enabled for
    this machine (units rendered fresh, enabled, and restarted), which also brings back one left installed but
    stopped. "off" removes every installed one. The dashboard is left alone (this page runs on it), and so is a
    service running by hand outside the service manager, which an install would start a second time."""
    if action not in ("repair", "off"):
        return 400, {"ok": False, "error": 'send {"all": "repair"} or {"all": "off"}'}
    view = services_view(True, fresh=True)
    if not view["ok"]:
        return 500, view
    rows = [r for r in view["services"] if r["name"] != "dashboard"]
    by_hand = [r["name"] for r in rows if not r["installed"] and r["pids"]]
    core = False
    if action == "repair":
        names = [r["name"] for r in rows if r["selected"] and r["name"] not in by_hand]
        core = bool({"continuum-broker", "continuum-capture"}.intersection(names))
        # The cascade must see the original PIDs before stopping anything. A separate uninstall would discard
        # that evidence and bypass its reader-first shutdown, including capture-worker ownership checks.
        steps = [["install", "--cascade", *names]] if core else [["uninstall", *names], ["install", *names]]
    else:
        names = [r["name"] for r in rows if r["installed"]]
        steps = [["uninstall", *names]]
    note = f"Left alone, running by hand: {', '.join(by_hand)}." if by_hand else None
    if not names:  # `pack-services uninstall` with no names means every installed service, the dashboard included
        return 200, {**services_view(True, fresh=True), "ok": True, "note": note or "Nothing to do."}
    outs = []
    why = "uninstalled and installed again" if action == "repair" else "every installed service but the dashboard"
    if action == "repair" and core:
        why = "units rendered and services restarted in dependency order"
    for step in steps:
        timeout = service_cascade.subprocess_timeout() if step[0] == "install" and core else 300
        ok, out = _pack_services(step, timeout)
        outs.append(out)
        if not ok:
            _logged(action, ", ".join(names), why, ok, out)
            error = (
                f"pack-services {step[0]} timed out" if ok is None else out[-500:] or f"pack-services {step[0]} failed"
            )
            return (504 if ok is None else 500), {
                **services_view(True, fresh=True),
                "ok": False,
                "error": error,
                "note": note,
            }
    _logged(action, ", ".join(names), why, True, "")
    return 200, {**services_view(True, fresh=True), "ok": True, "out": "\n".join(outs)[-2000:], "note": note}


@routes.get("/services/history")
def _get_services_history(h: H) -> None:
    code, body = service_history_view(h._query().get("hours", "24"))
    return h._json(body, code)


@routes.get("/services")
def _get_services(h: H) -> None:
    return h._json(services_view(h._can_control()))


@routes.post("/services")
def _post_services(h: H, body: Any) -> None:
    origin = h.headers.get("Origin")
    if not h._can_control():
        return h._json({"ok": False, "error": "services can only be changed from this machine"}, 403)
    if h.headers.get_content_type() != "application/json" or (
        origin and origin != "http://" + (h.headers.get("Host") or "")
    ):
        return h._json({"ok": False, "error": "same-origin JSON only"}, 403)
    with SERVICES_LOCK:  # one action at a time: two Disables at once would each rewrite pack.toml
        if "all" in body:
            code, o = all_services(str(body.get("all")))
        else:
            code, o = service_action(str(body.get("name", "")), str(body.get("action", "")))
    return h._json(o, code)
