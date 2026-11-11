"""Settings' Services: what each service is doing, and turning one on or off (moved out of the server, #61).

GET  /services          every service for this platform and whether this viewer may switch them (`services_view`)
GET  /services/history  lifecycle history over 24 hours or 7 days
POST /services          {"name", "action"} or {"all"}: this machine only, same-origin JSON, one action at a time
"""

import contextlib
import threading
from typing import TYPE_CHECKING, Any

from liberational import config as cfg
from liberational import service_cascade, services, supervisor
from liberational.dashboard import context, routes
from liberational.dashboard.context import LOG, Reply

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


# What each action on one service does to the application's set, and the words its log line says it in.
SERVICE_ACTIONS: dict[str, str] = {
    "on": "started; runs as long as the application does",
    "off": "stopped and kept stopped while the application runs",
    "stop": "stopped now; starts again when the application restarts",
    "start": "started",
}


def _no_application() -> Reply:
    """Turning a service on, off, or stopping it needs the application: nothing renders an operating-system unit."""
    return 503, {"ok": False, "error": f"{services.APPLICATION}, then try again"}


def _logged(action: str, what: str, why: str, ok: bool, out: str) -> None:
    """One INFO line per service action from the dashboard, done or not, so the log says what changed."""
    lines = out.strip().splitlines()
    bad = next((line for line in lines if "FAILED" in line), lines[-1] if lines else "?")  # the error, not a note after
    LOG.info(f"services: {action} {what} ({why}): {'done' if ok else f'failed: {bad}'}")


# Held around each POST /services action: Enable and Disable read and rewrite pack.toml, and two at once would lose
# one's change; the others mustn't race a start or stop of the same service either.
SERVICES_LOCK = threading.Lock()


# Enable and Disable: pack.toml's [services] lists first, so the choice outlasts the application, then the service.
SERVICE_CHOICES: dict[str, str] = {
    "enable": "enabled for this machine, and started",
    "disable": "disabled for this machine and stopped; stays off after a restart",
}


def service_action(name: str, action: str) -> Reply:
    """(code, body): one service's action: on (running, and kept running), off (stopped, and kept stopped), stop
    (stopped now, so it starts again when the application does, or on Repair), start (one the application has
    stopped), enable (one not enabled here: pack.toml's [services] names it, and it starts), or disable (the
    reverse: taken out of pack.toml's choice, so it stays off after a restart). All of them need the application
    open, since it, not an operating-system unit, runs the services."""
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
    if supervisor.read_status() is None:
        return _no_application()
    return _supervised(svc, name, action)


def _supervised(svc: dict[str, Any], name: str, action: str) -> Reply:
    """One service's action while a supervisor runs the services (#155): a message to it, not a unit install. On and
    start have it running; off and stop stop it and keep it stopped while the application runs. Enable and disable
    write pack.toml's [services] choice, which is what lasts, and have the supervisor follow it."""
    try:
        if action in SERVICE_CHOICES:
            default = bool(svc.get("default"))
            # Back to its default where that is the choice, so pack.toml names only what differs from services.toml.
            choice = (None if default else True) if action == "enable" else (False if default else None)
            with contextlib.suppress(SystemExit):  # choose() exits on a name services.toml doesn't declare
                services.choose([name], enable=choice)
            cfg.load.cache_clear()
            reply = supervisor.request({"op": "reload"}, timeout=120)
        else:
            op = "start" if action in ("on", "start") else "stop"
            reply = supervisor.request({"op": op, "names": [name]}, timeout=120)
    except ConnectionError as error:
        reply = {"ok": False, "error": str(error)}
    ok = bool(reply.get("ok"))
    LOG.info(f"services: {action} {name} (supervisor): {'done' if ok else 'failed: ' + str(reply.get('error'))}")
    return (200 if ok else 500), {
        **services_view(True, fresh=True),
        "ok": ok,
        "out": "",
        "error": None if ok else str(reply.get("error")),
    }


def all_services(action: str) -> Reply:
    """(code, body): every service but the dashboard at once. "repair" is the application's staged restart of every
    selected service, which also brings back one it had stopped. "off" stops every other service, and keeps it
    stopped while the application runs. The dashboard is left alone: this page runs on it."""
    if action not in ("repair", "off"):
        return 400, {"ok": False, "error": 'send {"all": "repair"} or {"all": "off"}'}
    status = supervisor.read_status()
    if status is None:
        return _no_application()
    # The application runs the services (#155): repair is its staged restart of everything, and off stops all but
    # this page.
    try:
        if action == "repair":
            reply = supervisor.request({"op": "restart"}, timeout=service_cascade.subprocess_timeout())
        else:
            names = [name for name in status["services"] if name != "dashboard"]
            reply = supervisor.request({"op": "stop", "names": names}, timeout=120)
    except ConnectionError as error:
        reply = {"ok": False, "error": str(error)}
    ok = bool(reply.get("ok"))
    LOG.info(f"services: {action} all (supervisor): {'done' if ok else 'failed: ' + str(reply.get('error'))}")
    return (200 if ok else 500), {
        **services_view(True, fresh=True),
        "ok": ok,
        "error": None if ok else str(reply.get("error")),
    }


@routes.get("/services/history")
def _get_services_history(h: H) -> None:
    code, body = service_history_view(h._query().get("hours", "24"))
    return h._json(body, code)


@routes.get("/services")
def _get_services(h: H) -> None:
    return h._json(services_view(h._can_control()))


@routes.post("/services")
def _post_services(h: H, body: Any) -> None:
    if not h._can_control():
        return h._json({"ok": False, "error": "services can only be changed from this machine"}, 403)
    if not h._same_origin_json():
        return h._json({"ok": False, "error": "same-origin JSON only"}, 403)
    with SERVICES_LOCK:  # one action at a time: two Disables at once would each rewrite pack.toml
        if "all" in body:
            code, o = all_services(str(body.get("all")))
        else:
            code, o = service_action(str(body.get("name", "")), str(body.get("action", "")))
    return h._json(o, code)
