"""The pack's services, declared once in services/services.toml: which run here, how each starts, and what runs now.

  uv run pack-services list                 # the services selected on this machine
  uv run pack-services status
  uv run pack-services restart --cascade   # staged restart of the selected services
  uv run pack-services install [NAME ...]   # follow pack.toml's selection and (re)start what was named
  uv run pack-services uninstall [NAME ...] # stop them
  uv run pack-services migrate              # remove the per-service units of an install from before the application

The application (pack-app, or pack-supervisor run) runs the selected services as its own children, using `selected`,
`argv`, and `environment` from here. The verbs above ask it (`_ask_supervisor`); with no application running they say
so and change nothing, since nothing here renders, installs, or uninstalls an operating-system unit any more. Which
services are selected: each one's `default`, adjusted by [services] enable / disable in pack.toml, limited to its
platforms. The dashboard's admin sidebar shows snapshot() and turns services on and off through the same verbs.

An install from before the application has one unit per service: launchd agents labelled pack.<name> in
~/Library/LaunchAgents on macOS, and systemd user units named pack-<name> in ~/.config/systemd/user on Linux.
`pack-services migrate` (unit_migration.py) finds and removes them; it reads their file names alone, so it never
needs the rendering this module no longer does.
"""

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import threading
import time
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast

from liberational import config as cfg
from liberational import logs

if TYPE_CHECKING:
    from liberational import proc_table

SPEC = cfg.RESOURCES / "services.toml"
DARWIN = sys.platform == "darwin"

# One [name] table from services.toml: command, args, env, default, platforms, schedule, and so on.
type Service = dict[str, Any]


def declared() -> dict[str, Service]:
    """Every service services.toml declares that this install can run: one marked `source_build` runs a program a
    checkout builds from source (whisper-server), so an installed package, which builds nothing, doesn't have it."""
    services = tomllib.loads(SPEC.read_text())
    if cfg.CHECKOUT is None:
        services = {name: svc for name, svc in services.items() if not svc.get("source_build")}
    return services


def required() -> set[str]:
    """Services the current settings depend on, whether or not [services] enables them."""
    if str(cfg.get("asr.engine")).lower() != "whisper":
        return set()
    if "whisper" not in declared():
        raise SystemExit(
            "asr.engine is whisper, and whisper-server is built from source: it isn't in an installed package. "
            "Set asr.engine to another engine, or run from a checkout."
        )
    return {"whisper"}


def selected() -> dict[str, Service]:
    """Services to install here, by name, in declaration order: defaults, [services] enable, and whatever the
    settings require (required()), minus [services] disable."""
    enable, disable = set(cfg.get("services.enable")) | required(), set(cfg.get("services.disable"))
    unknown = (enable | disable) - set(declared())
    if unknown:
        raise SystemExit(f"unknown service(s) in [services]: {', '.join(sorted(unknown))}")
    out: dict[str, Service] = {}
    for name, svc in declared().items():
        configured = bool(cfg.get(str(svc["enabled_by"]))) if svc.get("enabled_by") else False
        on = (svc.get("default", False) or configured or name in enable) and name not in disable
        if on and (not svc.get("platforms") or cfg.platform_key() in svc["platforms"]):
            out[name] = svc
    return out


_TEMPLATE = re.compile(r"\{([^}]+)\}")
OURS = ("pack-", "continuum-")  # commands installed with the package, in its environment's bin directory


def _expand(arg: str) -> str:
    def sub(m: re.Match[str]) -> str:
        key = m[1]
        if key.startswith("env:"):
            _, var, default = (key.split(":", 2) + [""])[:3]
            return os.environ.get(var) or default
        if key == "file:devices":
            return str(cfg.devices_file())
        if key.startswith("owner:"):  # one of the owner's files by its checkout path, wherever this install has it
            return str(cfg.owner_file(key.removeprefix("owner:")))
        value = str(cfg.get(key))
        # A path is given whole, so it means the same wherever the service's working directory is.
        return str(cfg.owner_file(value)) if value and (key.startswith("paths.") or key.endswith(".path")) else value

    return os.path.expanduser(_TEMPLATE.sub(sub, arg))


def argv(svc: Service, *, startup: bool = False, name: str = "") -> list[str]:
    """Absolute command line for a service. At startup, its wrappers too: a service with `log_relay = true` (a
    program we don't own) runs under log_relay, tagged with its name, so its lines carry the one log format."""
    cmd = svc["command"]
    exe = [str(cfg.program(cmd))] if cmd.startswith(OURS) else [shutil.which(cmd) or cmd]
    if startup and svc.get("fresh_config"):
        command = [
            sys.executable,
            "-m",
            "liberational.services",
            "exec",
            str(svc["fresh_config"]),
        ]
    else:
        command = exe + [_expand(a) for a in svc.get("args", [])]
    if startup and svc.get("wait_for_gpu"):
        command = [sys.executable, "-m", "liberational.gpu_ready", "--", *command]
    if startup and svc.get("wait_for_broker"):
        command = [
            sys.executable,
            "-m",
            "liberational.continuum_ready",
            "--socket",
            str(cfg.path("paths.state") / "continuum/broker.sock"),
            "--",
            *command,
        ]
    if startup and svc.get("log_relay"):
        command = [sys.executable, "-m", "liberational.log_relay", "--tag", name, "--", *command]
    return command


def _templated_env(svc: Service) -> dict[str, str]:
    """The service's own env, with {section.key} templates filled from config, as in args. A variable whose template
    fills in empty is left out (CUDA_VISIBLE_DEVICES="" would hide every GPU); one written empty is kept empty."""
    env = {}
    for k, v in (svc.get("env") or {}).items():
        value = _expand(str(v))
        if value or not _TEMPLATE.search(str(v)):
            env[k] = value
    return env


# What a service gets from whoever opened the application (#183): the variables that say which desktop session this
# is, so PipeWire, D-Bus, and the display can be reached, and the locale. Nothing else is inherited.
SESSION = frozenset(
    {
        "XDG_RUNTIME_DIR",
        "DBUS_SESSION_BUS_ADDRESS",
        "WAYLAND_DISPLAY",
        "DISPLAY",
        "XAUTHORITY",
        "XDG_SESSION_TYPE",
        "PIPEWIRE_REMOTE",
        "PIPEWIRE_RUNTIME_DIR",
        "PULSE_SERVER",
        "PULSE_RUNTIME_PATH",
        "USER",
        "LOGNAME",
        "LANG",
        "LANGUAGE",
        "TZ",
    }
)
# Variables a launcher sets for itself, which would change what a service loads: the dynamic loader's, and those of
# an AppImage, a Flatpak, or a Snap whose terminal the application was opened from. Never inherited; a service
# that needs one names it in services.toml's env. Listed only to say in the log what was dropped.
LAUNCHER = frozenset({"LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT", "APPDIR", "APPIMAGE", "ARGV0", "OWD", "SNAP"})
LAUNCHER_PREFIXES = ("DYLD_", "APPIMAGE_", "FLATPAK_", "SNAP_")
APPIMAGE_MOUNTS = "/tmp/.mount_"  # where an AppImage mounts itself; matched, never written


def _launcher_roots(launcher: Mapping[str, str]) -> list[str]:
    """The directories a launcher put itself in: its mount (AppImage), or its own tree (Snap)."""
    return [launcher[key].rstrip("/") for key in ("APPDIR", "SNAP") if launcher.get(key, "").startswith("/")]


def _from_launcher(directory: str, roots: Sequence[str]) -> bool:
    return directory.startswith(APPIMAGE_MOUNTS) or any(directory == r or directory.startswith(r + "/") for r in roots)


def dropped(launcher: Mapping[str, str]) -> list[str]:
    """What of a launcher's environment `environment(..., launcher=...)` keeps from a service, for the log: its
    loader and packaging variables by name, and the PATH directories inside the launcher."""
    roots = _launcher_roots(launcher)
    names = sorted(k for k in launcher if k in LAUNCHER or k.startswith(LAUNCHER_PREFIXES))
    path = [d for d in launcher.get("PATH", "").split(os.pathsep) if d and _from_launcher(d, roots)]
    return names + [f"PATH entry {d}" for d in dict.fromkeys(path)]


def _session(launcher: Mapping[str, str]) -> dict[str, str]:
    """The session variables in a launcher's environment (SESSION, and the locale's LC_*)."""
    env = {k: v for k, v in launcher.items() if v and (k in SESSION or k.startswith("LC_"))}
    runtime = pathlib.Path(f"/run/user/{os.getuid()}")
    if "XDG_RUNTIME_DIR" not in env and not DARWIN and runtime.is_dir():
        env["XDG_RUNTIME_DIR"] = str(runtime)  # opened from a shell with no session (cron, a bare ssh command)
    return env


def environment(svc: Service, name: str = "", *, launcher: Mapping[str, str] | None = None) -> dict[str, str]:
    """A service's whole environment. `launcher` (the environment the application was opened with) gives the child
    the session variables from it and nothing else (#183), so a terminal's loader paths and packaging variables
    never reach a service; with none, the current environment stands in."""
    # The environment's own bin first; in a checkout, mise's shims next (tools like ffmpeg from mise.toml; an
    # installed package has no mise and gets them from the system); then the starting shell's PATH; then the system
    # directories whatever that shell had: capture runs sysctl, which is in /usr/sbin.
    source = os.environ if launcher is None else launcher
    roots = _launcher_roots(source)
    parts = [str(cfg.BIN)]
    parts += [str(pathlib.Path.home() / ".local/share/mise/shims")] if cfg.CHECKOUT is not None else []
    parts += [d for d in source.get("PATH", "").split(os.pathsep) if not _from_launcher(d, roots)]
    parts += ["/usr/local/bin"] if DARWIN else []
    parts += ["/usr/bin", "/bin", "/usr/sbin", "/sbin"]
    env = _session(launcher) if launcher is not None else {}
    env.update({"PATH": os.pathsep.join(dict.fromkeys(p for p in parts if p)), "HOME": str(pathlib.Path.home())})
    # No progress bars in a service log (Hugging Face downloads, transformers' "Loading weights", any tqdm): a bar
    # redraws itself, which a log file keeps as a pile of partial lines. A CLI run by hand still shows them. (tqdm
    # reads any non-empty TQDM_DISABLE as true, so "0" doesn't turn its bars back on; an empty value does.)
    env.update({"HF_HUB_DISABLE_PROGRESS_BARS": "1", "TQDM_DISABLE": "1"})
    # Native libraries' own logs at errors only (#59): they write to stderr with no stamp or level. transformers'
    # handler prints its deprecation notes bare, and OpenCV's FFmpeg its decoder warnings (16 is AV_LOG_ERROR).
    # FFmpeg inside continuum_py is set the same way where it initializes (continuum-tier's codec_open), and ONNX
    # Runtime's where a session is built (vision/onnx_runtime.py). A Rust daemon's level is RUST_LOG, from the
    # same logging.level and logging.per_service the Python services read, so it takes a reinstall to change.
    env.update({"TRANSFORMERS_VERBOSITY": "error", "OPENCV_FFMPEG_LOGLEVEL": "16", "OPENCV_LOG_LEVEL": "ERROR"})
    env["RUST_LOG"] = logs.rust_filter(name)
    env.update(_templated_env(svc))
    if int(env.get("LLAMA_ARG_N_CPU_MOE", "0")) > 0:
        # The runner's generic whole-model fitter cannot budget CPU experts separately. Let Ask's explicit
        # placement keep its attention/KV on GPU; with the portable zero-layer default, retain the stock fitter.
        env["LLAMA_ARG_FIT"] = "off"
    return env


def log_path(name: str) -> pathlib.Path:
    """A service's log. The log-trim service keeps it and its one older copy (logs.older) under logs.CAP_BYTES."""
    return cfg.path("paths.logs") / f"{name}.log"


APPLICATION = "the application isn't running: open it with pack-app (or pack-supervisor run, without the window)"


def _supervised() -> dict[str, Any] | None:
    """The running application's services (supervisor.py), or None when the application isn't open."""
    from liberational import supervisor

    status = supervisor.read_status()
    return status["services"] if status is not None else None


def _needs_application() -> None:
    """Refuse a verb only the application can carry out, plainly: nothing here renders or installs a unit."""
    raise SystemExit(APPLICATION)


def _ask_supervisor(op: str, names: Sequence[str] = (), timeout: float = 120.0) -> None:
    """Send the running supervisor one request, and print one line a name (or for the whole set). Exits nonzero
    with its reason when it refuses or doesn't answer."""
    from liberational import supervisor

    said = {"start": "running", "stop": "stopped", "restart": "restarted", "reload": "reloaded"}[op]
    try:
        reply = supervisor.request({"op": op, "names": list(names)}, timeout=timeout)
    except ConnectionError as error:
        reply = {"ok": False, "error": str(error)}
    for name in names or ["the application's services"]:
        print(f"pack-{name}: {said if reply.get('ok') else 'FAILED ' + str(reply.get('error'))}")
    if not reply.get("ok"):
        raise SystemExit(1)


def install(names: Sequence[str], *, cascade: bool = False) -> None:
    """Have the application follow pack.toml's selection, then (re)start what was named, or everything through the
    staged cascade with no names or --cascade. There is nothing to install: the application runs the services."""
    if _supervised() is None:
        _needs_application()
    from liberational import service_cascade

    _ask_supervisor("reload")
    _ask_supervisor("restart", [] if cascade else names, timeout=service_cascade.subprocess_timeout())


def installed() -> list[str]:
    """The services the running application runs, by name (empty while it isn't open)."""
    return list(_supervised() or {})


def uninstall(names: Sequence[str]) -> None:
    """Stop services now: the application starts them again the next time it opens. With no names, quit the
    application, which stops everything."""
    if _supervised() is None:
        _needs_application()
    if not names:
        raise SystemExit("under the application, quit it to stop everything (pack-supervisor quit)")
    return _ask_supervisor("stop", names)


def stop(names: Sequence[str]) -> None:
    """Stop services now, leaving them selected: they start again on the application's next start, on Repair, or on
    `start`. A stop lasts while one application runs."""
    if _supervised() is None:
        _needs_application()
    return _ask_supervisor("stop", names)


def start(names: Sequence[str]) -> None:
    """Start selected services that were stopped (a scheduled job's timer starts its schedule)."""
    if _supervised() is None:
        _needs_application()
    return _ask_supervisor("start", names)


def restart(names: Sequence[str], *, cascade: bool = False) -> None:
    """Restart services. Core changes always use the same strict staged cascade, with the same readiness checks and
    recovery, as the application's own start and quit."""
    if _supervised() is None:
        _needs_application()
    from liberational import service_cascade

    if not names and not cascade:
        raise SystemExit("restart needs service names or --cascade")
    return _ask_supervisor("restart", [] if cascade else names, timeout=service_cascade.subprocess_timeout())


def states() -> dict[str, str]:
    """The application's state for each service it runs (running, stopped, failed, ...; empty while it isn't open)."""
    supervised = _supervised()
    return {name: str(service["state"]) for name, service in supervised.items()} if supervised else {}


LOG_TAIL_BYTES = 64 * 1024  # how much of the end of a service's log last_error reads
# A line's stamp: an ISO local date and time with offset, or in lines written before it, the time alone.
STAMP = r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d|\d\d:\d\d:\d\d)"
ERROR_LINE = re.compile(STAMP + r" \[[^\]]*\] ERROR (.*)")
# Any line logs.py or the broker wrote, and the stamped free text continuum-capture and the tier worker write.
TIMED = re.compile(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d) |(\d\d:\d\d:\d\d) \[")


def runs() -> dict[str, dict[str, Any]]:
    """How each service the application runs has run: {"restarts": how often the application restarted it, "up_s":
    how long its current run has lasted, or None}. A crash loop is restarts climbing with a short current run.
    run_id identifies the current run without changing as uptime advances, so acknowledging its history cannot hide
    a later run. Scheduled jobs are left out: their runs are by design."""
    from liberational import supervisor

    status = supervisor.read_status()
    return supervisor.managed(status, time.time())[1] if status is not None else {}


# Public systemd lifecycle identifiers: src/systemd/sd-messages.h. Read manager metadata only, never MESSAGE.
type LifecycleKind = Literal["started", "stopped", "restart", "crashed", "start-failed"]

LIFECYCLE_IDS: dict[str, LifecycleKind] = {
    "39f53479d3a045ac8e11786248231fbf": "started",
    "9d1aaa27d60140bd96365438aad20286": "stopped",
    "7ad2d189f7e94e70a38c781354912448": "stopped",  # successful deactivation without a stop job
    "5eb03494b6584870a536b337290809b3": "restart",
    "d9b373ed55a64feb8242e02dbe79a49c": "crashed",
    "be02cf6855d2428ba40df7e9d022f03d": "start-failed",
}
PROCESS_EXIT_ID = "98e322203f7a4ed290d09fe03c09fe15"
HISTORY_RECORDS = 4000
HISTORY_BYTES = 4 * 1024 * 1024
HISTORY_TTL_S = 30.0
HISTORY_FIELDS = (
    "MESSAGE_ID,USER_UNIT,UNIT,__REALTIME_TIMESTAMP,_BOOT_ID,USER_INVOCATION_ID,INVOCATION_ID,"
    "UNIT_RESULT,JOB_RESULT,EXIT_CODE,EXIT_STATUS"
)


class HistorySpan(TypedDict):
    start: float
    end: float
    state: Literal["up", "stopped", "unknown"]


class HistoryEvent(TypedDict):
    at: float
    kind: LifecycleKind
    reason: str | None


# The public event plus the boot and invocation that own its transition.
type _HistoryTransition = tuple[HistoryEvent, str | None, str | None]


class ServiceHistory(TypedDict):
    name: str
    spans: list[HistorySpan]
    events: list[HistoryEvent]


class HistoryView(TypedDict):
    start: float
    end: float
    source: Literal["journal", "sampled", "unavailable"]
    available: bool
    sample_s: float | None
    truncated: bool
    note: str | None
    services: list[ServiceHistory]


_history_lock = threading.Lock()
_history_cache: dict[tuple[str, int, tuple[str, ...], bool], tuple[float, HistoryView]] = {}


def _history_span(
    spans: list[HistorySpan], start: float, end: float, state: Literal["up", "stopped", "unknown"]
) -> None:
    if end <= start:
        return
    if spans and spans[-1]["end"] == start and spans[-1]["state"] == state:
        spans[-1]["end"] = end
    else:
        spans.append({"start": start, "end": end, "state": state})


def _journal_view(
    records: list[dict[str, Any]], names: Sequence[str], start: float, end: float, current_boot: str | None
) -> HistoryView:
    """Project metadata in time order. No events means unknown, not a healthy or stopped window."""
    names = sorted(names)
    events: dict[str, list[_HistoryTransition]] = {name: [] for name in names}
    anchors: dict[str, _HistoryTransition] = {}
    exits: dict[tuple[str, str, str], str] = {}
    ordered: list[tuple[float, dict[str, Any]]] = []
    for row in records:
        try:
            at = int(row["__REALTIME_TIMESTAMP"]) / 1e6
        except KeyError, TypeError, ValueError, OverflowError:
            continue
        if at <= end:
            ordered.append((at, row))
    ordered.sort(key=lambda item: item[0])
    unit_failures: set[tuple[str, str, str]] = set()
    for _, row in ordered:
        if LIFECYCLE_IDS.get(str(row.get("MESSAGE_ID"))) != "crashed":
            continue
        invocation = str(row.get("USER_INVOCATION_ID") or row.get("INVOCATION_ID") or "")
        if invocation:
            unit_failures.add(
                (str(row.get("USER_UNIT") or row.get("UNIT")), str(row.get("_BOOT_ID") or ""), invocation)
            )
    seen: set[tuple[str, str, str]] = set()
    for at, row in ordered:
        unit = row.get("USER_UNIT") or row.get("UNIT")
        name = (
            str(unit)[5:-8] if isinstance(unit, str) and unit.startswith("pack-") and unit.endswith(".service") else ""
        )
        if name not in events:
            continue
        boot = str(row.get("_BOOT_ID") or "")
        invocation = str(row.get("USER_INVOCATION_ID") or row.get("INVOCATION_ID") or "")
        key = (str(unit), boot, invocation)
        message = row.get("MESSAGE_ID")
        if message == PROCESS_EXIT_ID:
            code, status = row.get("EXIT_CODE"), row.get("EXIT_STATUS")
            if invocation and isinstance(code, str) and isinstance(status, str):
                exits[key] = f"{code}, status {status}"[:160]
            continue
        kind = LIFECYCLE_IDS.get(str(message))
        if kind is None:
            continue
        # Prefer the actual unit failure over a duplicate failed-start job, in either logging order.
        if kind == "start-failed" and invocation and key in unit_failures:
            continue
        if kind == "crashed" and invocation and key in seen:
            continue  # repeated unit-failure messages describe one invocation
        if kind == "crashed" and invocation:
            seen.add(key)
        result = row.get("UNIT_RESULT") or row.get("JOB_RESULT")
        reason = str(result)[:160] if result else None
        if kind == "crashed" and invocation and key in exits:
            reason = ": ".join(x for x in (reason, exits[key]) if x)
        item: _HistoryTransition = ({"at": at, "kind": kind, "reason": reason}, boot or None, invocation or None)
        if kind == "start-failed":
            if at >= start:
                events[name].append(item)
            continue  # a job result says nothing about whether the unit itself stopped
        if at < start:
            anchors[name] = item
        else:
            events[name].append(item)
    services: list[ServiceHistory] = []
    for name, found in events.items():
        spans: list[HistorySpan] = []
        state: Literal["up", "stopped", "unknown"] = "unknown"
        previous = start
        anchor = anchors.get(name)
        boot = anchor[1] if anchor else None
        invocation = anchor[2] if anchor else None
        if anchor:
            state = "up" if anchor[0]["kind"] == "started" else "stopped"
        for event, event_boot, event_invocation in found:
            if event["kind"] == "start-failed":
                continue
            same_run = bool(invocation and event_invocation and invocation == event_invocation)
            changed_owner = bool(invocation and event_invocation and invocation != event_invocation)
            missing_end = state == "up" and (changed_owner or (event["kind"] == "started" and not same_run))
            # Neither a different boot nor a new invocation proves when the previous run actually ended.
            _history_span(
                spans, previous, event["at"], state if boot and boot == event_boot and not missing_end else "unknown"
            )
            state = "up" if event["kind"] == "started" else "stopped"
            previous, boot, invocation = event["at"], event_boot, event_invocation
        _history_span(spans, previous, end, state if current_boot and boot == current_boot else "unknown")
        services.append({"name": name, "spans": spans, "events": [event for event, _, _ in found]})
    return {
        "start": start,
        "end": end,
        "source": "journal",
        "available": True,
        "sample_s": None,
        "truncated": False,
        "note": None,
        "services": services,
    }


def _read_journal(names: Sequence[str], start: float, end: float) -> HistoryView:
    binary = shutil.which("journalctl")
    if binary is None:
        raise OSError("Lifecycle history is unavailable")
    command = [
        binary,
        "--user",
        "--no-pager",
        "-o",
        "json",
        f"--output-fields={HISTORY_FIELDS}",
        f"--since=@{start:.6f}",
        f"--until=@{end:.6f}",
        "-n",
        str(HISTORY_RECORDS + 1),
    ]
    for name in names:
        command += ["-u", f"pack-{name}.service"]
    command += ["_COMM=systemd", *[f"MESSAGE_ID={message}" for message in (*LIFECYCLE_IDS, PROCESS_EXIT_ID)]]

    def read(args: list[str]) -> list[dict[str, Any]]:
        reply = subprocess.run(args, capture_output=True, timeout=5, check=True)
        if len(reply.stdout) > HISTORY_BYTES:
            raise ValueError("Lifecycle history exceeded its bounded reply")
        rows = [json.loads(line) for line in reply.stdout.splitlines() if line.strip()]
        if not all(isinstance(row, dict) for row in rows):
            raise ValueError("Invalid lifecycle history")
        return rows

    rows = read(command)
    truncated = len(rows) > HISTORY_RECORDS
    anchors: list[dict[str, Any]] = []
    anchor_limited = False
    if not truncated:
        # Only a complete in-window read can extend prior evidence through the window. A bounded reverse lookup
        # finds long-lived services' latest transition without reading application output or all past records.
        prior = [arg for arg in command if not arg.startswith(("--since=", "--until="))]
        prior += [f"--until=@{start - 0.000001:.6f}"]
        anchors = read(prior)
        anchor_limited = len(anchors) > HISTORY_RECORDS
        anchors = anchors[-HISTORY_RECORDS:]
    try:
        boot = pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip().replace("-", "")
    except OSError:
        boot = None
    view = _journal_view([*anchors, *rows[-HISTORY_RECORDS:]], names, start, end, boot)
    view["truncated"] = truncated or anchor_limited
    if view["truncated"]:
        view["note"] = "Some earlier lifecycle events could not be included; unproven intervals remain unknown."
    return view


def history(state: pathlib.Path, hours: int = 24) -> HistoryView:
    """Bounded, cached uptime history. Current status/counters never stand in for missing past observations."""
    if hours not in (24, 168):
        raise ValueError("History must cover 24 hours or 7 days")
    names = tuple(
        sorted(
            name
            for name, spec in declared().items()
            if not spec.get("schedule") and (not spec.get("platforms") or cfg.platform_key() in spec["platforms"])
        )
    )
    key = (str(state), hours, names, DARWIN)
    with _history_lock:
        now = time.monotonic()
        if key in _history_cache and now - _history_cache[key][0] < HISTORY_TTL_S:
            return _history_cache[key][1]
        end = time.time()
        start = end - hours * 3600
        if _supervised() is not None:  # the application's own record of each start and exit (#165)
            from liberational import supervisor

            # supervisor.history() builds the same HistoryView (its docstring says so) but is typed as a plain dict,
            # since the supervisor does not depend on this module's history types.
            view = cast(HistoryView, supervisor.history(supervisor.home(), names, start, end))
        else:
            try:
                if DARWIN or not names:
                    raise OSError("Exact lifecycle history is unavailable")
                view = _read_journal(names, start, end)
            except OSError, ValueError, subprocess.SubprocessError:
                from liberational.dashboard import report_health

                view = report_health.service_history(state, names, start, end)
        # At most the two supported ranges for the current state/platform/roster are retained.
        for old in list(_history_cache):
            if old[0] != str(state) or old[2:] != key[2:]:
                _history_cache.pop(old)
        _history_cache[key] = (now, view)
        return view


TRACEBACK = "Traceback (most recent call last):"
CLOCK_SLACK_S = 120  # threads can write lines a little out of order; a bigger step back in time is a new day
UP_STATES = {"active", "running"}  # the application's and systemd's word for a service that's running
RESTARTED_RECENTLY_S = 600  # the frontend's CRASH_RECENT_S: a restart this recent is a crash loop's
CRASH_LOOP_RESTARTS = 3  # the frontend's CRASH_LOOP: fewer restarts don't establish a loop


def _dated(stamps: Sequence[str], end: float) -> list[float]:
    """Epoch seconds for a log's stamps, in file order. A dated stamp is its own time. A "HH:MM:SS" one, from
    before stamps carried the date, has none, so it's dated backward from the next dated line, or the log's last write
    (end, its mtime): a line later in the day than the one after it is from the day before."""
    last = datetime.datetime.fromtimestamp(end)
    day, bound = last.date(), last.hour * 3600 + last.minute * 60 + last.second + CLOCK_SLACK_S
    out: list[float] = []
    for stamp in reversed(stamps):
        if "T" in stamp:
            at = datetime.datetime.fromisoformat(stamp)
            out.append(at.timestamp())
            local = datetime.datetime.fromtimestamp(at.timestamp())
            day, bound = local.date(), local.hour * 3600 + local.minute * 60 + local.second + CLOCK_SLACK_S
            continue
        h, m, sec = (int(x) for x in stamp.split(":"))
        tod = h * 3600 + m * 60 + sec
        if tod > bound:
            day -= datetime.timedelta(days=1)
        bound = tod + CLOCK_SLACK_S
        out.append(datetime.datetime.combine(day, datetime.time(h, m, sec)).timestamp())
    return out[::-1]


def _lines(parts: list[logs.Part]) -> Iterator[tuple[os.stat_result, int, bytes]]:
    """Each line of a log's tail with the file it's in and its byte offset there."""
    for part in parts:
        offset = part.offset
        for raw in part.data.splitlines(keepends=True):
            yield part.stat, offset, raw
            offset += len(raw)


# Each log's newest error, kept with the files it was read from (the log and its older copy: device, inode, size,
# and mtime of each): /services is polled every 15 s on every page, and reading the last 64 KiB of every service's
# log each time was 28 opens a poll on the Linux machine. A log nobody wrote to since is not opened again.
type _LogIdentity = tuple[tuple[int, int, int, int] | None, ...]
_log_errors: dict[str, tuple[_LogIdentity, dict[str, Any] | None]] = {}
_log_errors_lock = threading.Lock()


def _log_identity(path: pathlib.Path) -> _LogIdentity:
    """What tells a log and its older copy from the ones last read; None for one that isn't there."""
    out: list[tuple[int, int, int, int] | None] = []
    for file in (path, logs.older(path)):
        try:
            stat = file.stat()
        except OSError:
            out.append(None)
        else:
            out.append((stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns))
    return tuple(out)


def _newest_error(path: pathlib.Path) -> dict[str, Any] | None:
    """The newest error in the last LOG_TAIL_BYTES of a log, its older file included just after a rotation: {"at",
    "ts", "text", "id"}, or None. Read again only when either file's size or modification time changed, or it's
    another file (rotated)."""
    key = str(path)
    identity = _log_identity(path)
    with _log_errors_lock:
        kept = _log_errors.get(key)
    if kept is not None and kept[0] == identity:
        return kept[1]
    error = _parse_error(logs.tail(path, LOG_TAIL_BYTES))
    with _log_errors_lock:
        # Kept under what the files were before the read: a write during it is seen as a change by the next call.
        _log_errors[key] = (identity, error)
    return error


def _parse_error(parts: list[logs.Part]) -> dict[str, Any] | None:
    """_newest_error's answer from a log's tail."""
    if not parts:
        return None
    end = parts[-1].stat.st_mtime
    # (stamp index, or -1; text; the file and byte offset of its occurrence)
    found: tuple[int, str, tuple[os.stat_result, int]] | None = None
    stamps: list[str] = []  # every timestamped line's time, in file order
    in_trace = after_error = False
    for stat, at, raw in _lines(parts):
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        position = (stat, at)
        if line.startswith(TRACEBACK):
            in_trace = True
            continue
        if in_trace:
            if not line or line[0].isspace() or line.startswith(("During handling", "The above exception")):
                continue
            in_trace = False
            if not after_error and not TIMED.match(line):  # its exception line, unless an ERROR line said it already
                found = (len(stamps) - 1, line.strip()[:300], position)
                continue
        if m := TIMED.match(line):
            stamps.append(m[1] or m[2])
            e = ERROR_LINE.fullmatch(line.strip())
            after_error = bool(e)
            if e:
                found = (len(stamps) - 1, e[2][:300], position)
    if found is None:
        return None
    i, text, position = found
    ts = _dated(stamps, end)[i] if i >= 0 else None
    stat, at = position
    occurrence = f"{stat.st_dev}:{stat.st_ino}:{at}:{ts}:{text}"
    return {
        "at": stamps[i][-14:-6] if i >= 0 and "T" in stamps[i] else stamps[i] if i >= 0 else "",
        "ts": ts,
        "text": text,
        "id": hashlib.sha256(occurrence.encode()).hexdigest(),
    }


def last_error(name: str, since: float | None = None, *, include_id: bool = False) -> dict[str, Any] | None:
    """What last went wrong in a service, from the last LOG_TAIL_BYTES of its log, its older file included just after
    a rotation (a long log is never read whole, and an unchanged one isn't read again: `_newest_error`):
    {"at": "HH:MM:SS", "ts": epoch seconds, "text"}, or None. That's its newest ERROR line, or a crash's exception
    (the last line of a bare traceback, which an uncaught exception writes with no ERROR line; its time is the time
    of the line before it, and "at" is "" and "ts" None when there's none).

    since (epoch seconds) leaves out an error from before then: a running service's errors from before its current
    start were fixed by that start, and only confuse the page. An error with no time counts as before. include_id
    adds an opaque log occurrence id for dismissals: appending the same error in the same second is a new event,
    while later unrelated log writes leave its id alone."""
    found = _newest_error(log_path(name))
    if found is None:
        return None
    ts = found["ts"]
    if since is not None and (ts is None or ts + 1 <= since):  # the log's seconds are whole; since isn't
        return None
    return {k: v for k, v in found.items() if include_id or k != "id"}


def _run_began(state: str | None, run: dict[str, Any] | None) -> float | None:
    """The oldest error to show: a healthy service's current run or the last ten minutes, whichever is newer.

    Failed services and current crash loops retain their reason regardless of age. The cutoff advances with wall
    time, not with polling, so an old error expires even when the process never restarts or logs recovery."""
    if state not in UP_STATES:
        return None
    up = run.get("up_s") if run else None
    if run and (run.get("restarts") or 0) >= CRASH_LOOP_RESTARTS and (up is None or up < RESTARTED_RECENTLY_S):
        return None
    now = time.time()
    return max(now - RESTARTED_RECENTLY_S, now - up if up is not None else 0.0)


# The process scan reads every process's /proc entry, which takes seconds while the services restart and load models.
# Keep the last answer for PROCESS_SCAN_TTL_S, and while a rescan runs, answer with the last one instead of
# waiting.
PROCESS_SCAN_TTL_S = 5.0
_scan_lock = threading.Lock()
_scan_work_lock = threading.Lock()  # an explicit fresh scan waits for any background scan, then takes its own
_scan: dict[str, Any] = {"at": None, "pids": {}, "running": False, "thread": None}


class ProcessScanPending(Exception):
    """A nonblocking status poll has no current process evidence yet; this is not an empty process list."""


def _commands() -> tuple[dict[str, str], dict[str, str]]:
    """How each declared service shows in a command line, as two maps to its name: a pack-* command by its own name,
    wherever it runs from (uv run, the venv's python), and a command in the repo (Continuum's daemons, whisper-server,
    a script) by its absolute path in this checkout, as argv() runs it, so a build in another checkout isn't counted."""
    by_name: dict[str, str] = {}
    by_path: dict[str, str] = {}
    for name, svc in declared().items():
        cmd = svc["command"]
        if cmd.startswith("pack-"):
            by_name[cmd] = name
        elif cmd.startswith(OURS):
            by_path[str(cfg.program(cmd))] = name
            by_path[str(cfg.program(cmd).resolve())] = name  # a checkout links it there from its release build
    return by_name, by_path


DEMO_COMMANDS = frozenset({"pack-demo", "liberational.demo.cli"})
"""How pack-demo shows in a command line (its script, or the module run with -m)."""


def _demo_processes(table: Mapping[int, proc_table.Proc]) -> set[int]:
    """Pids of pack-demo and everything under it: its own broker and dashboard run the services' programs on a made-up
    household, apart from this install, and aren't its services (they would stop the application from starting)."""
    roots = {p.pid for p in table.values() if any(arg.rsplit("/", 1)[-1] in DEMO_COMMANDS for arg in p.cmdline)}
    found: dict[int, bool] = {}

    def under_demo(pid: int) -> bool:
        chain: list[int] = []
        answer = False
        while pid not in found:
            if pid in roots:
                answer = True
                break
            chain.append(pid)
            p = table.get(pid)
            if p is None or p.ppid is None or p.ppid in chain or len(chain) > 64:
                break
            pid = p.ppid
        else:
            answer = found[pid]
        for seen in chain:
            found[seen] = answer
        return answer

    return {pid for pid in table if under_demo(pid)}


def _scan_processes() -> dict[str, list[int]]:
    """Pids of running services by name, from every process (proc_table, each one's command line read once): each
    service's own command, and the processes it started (continuum-capture runs one worker per device), which count as
    it."""
    from liberational import proc_table

    by_name, by_path = _commands()
    table = proc_table.table()
    demo = _demo_processes(table)
    out: dict[str, list[int]] = {}
    others: list[tuple[int, int | None]] = []  # (pid, parent pid) of the processes no command matched
    for p in table.values():
        if p.pid in demo:
            continue
        name = next(
            (n for arg in p.cmdline if (n := by_name.get(arg.rsplit("/", 1)[-1]) or by_path.get(arg))),
            None,
        )
        if name:
            out.setdefault(name, []).append(p.pid)
        else:
            others.append((p.pid, p.ppid))
    owner = {pid: name for name, pids in out.items() for pid in pids}
    for pid, parent in others:
        if parent is not None and (name := owner.get(parent)):
            out[name].append(pid)
    return out


def _rescan(*, background: bool = False) -> None:
    with _scan_work_lock:
        try:
            pids = _scan_processes()
        except Exception as error:
            with _scan_lock:
                _scan.update(running=False, error=type(error).__name__)
            if not background:
                raise
            return  # the next poll reports the failure and retries; daemon threads never leak a traceback
        with _scan_lock:
            _scan.update(at=time.monotonic(), pids=pids, running=False, error=None)


def processes(fresh: bool = False, *, nonblocking: bool = False) -> dict[str, list[int]]:
    """Pids of running services by name (as _scan_processes finds them), whether the application started them or a
    person did.

    fresh=False (the sidebar's poll): the last scan's answer, scanning only on the first call. A scan older than
    PROCESS_SCAN_TTL_S is still returned as it is and starts one rescan in the background, so the poll after it
    gets the new one. fresh=True (before turning a service on or off): a new scan, waited for. nonblocking=True
    lets dashboard polls start the first scan in the background and raises ProcessScanPending until it succeeds;
    other callers retain the synchronous first-scan contract. An explicit fresh=True always waits."""
    if fresh:
        _rescan()
    with _scan_lock:
        at = _scan["at"]
        pending = nonblocking and not fresh and (at is None or _scan.get("error") is not None)
        if (pending or (at is not None and time.monotonic() - at >= PROCESS_SCAN_TTL_S)) and not _scan["running"]:
            _scan["running"] = True
            _scan["thread"] = threading.Thread(
                target=_rescan, kwargs={"background": True}, daemon=True, name="service-pids"
            )
            _scan["thread"].start()
        if pending:
            error = _scan.get("error")
            raise ProcessScanPending(
                f"Service process scan failed ({error}); retrying…" if error else "Checking service processes…"
            )
        if at is not None:
            return {name: list(pids) for name, pids in _scan["pids"].items()}
    _rescan()
    with _scan_lock:
        return {name: list(pids) for name, pids in _scan["pids"].items()}


# What the application said of its services (states() and runs()), kept for a few seconds: several pages poll
# /services at once, and a state changes only when a service starts, stops, or fails.
MANAGER_TTL_S = 5.0
_manager_lock = threading.Lock()
_manager: dict[str, Any] = {"at": None, "states": {}, "runs": {}}


def _managed(fresh: bool = False) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """(states(), runs()) for a status poll: the application's own record of its services (#155). The last answer
    under MANAGER_TTL_S old is reused, with each run's `up_s` moved on by the answer's age; fresh=True (an action's
    reply) asks now. Polls that arrive together wait for one read instead of each starting their own; with no
    application open there is nothing to ask."""
    from liberational import supervisor

    status = supervisor.read_status()
    if status is not None:
        return supervisor.managed(status, time.time())
    with _manager_lock:
        at = _manager["at"]
        if fresh or at is None or time.monotonic() - at >= MANAGER_TTL_S:
            state, ran = states(), runs()
            _manager.update(at=time.monotonic(), states=state, runs=ran)
            return dict(state), {name: dict(run) for name, run in ran.items()}
        age = time.monotonic() - at
        return dict(_manager["states"]), {
            name: {**run, "up_s": run["up_s"] + age} if run.get("up_s") is not None else dict(run)
            for name, run in _manager["runs"].items()
        }


def snapshot(fresh: bool = False) -> list[dict[str, Any]]:
    """Every service declared for this platform: selected for this machine, installed, its state, and running pids
    (fresh as in processes()). A poll reuses the application's last answer for MANAGER_TTL_S (`_managed`) and opens
    only the logs written to since the last one (`_newest_error`)."""
    # A cold poll reports pending before asking the manager or reading logs; actions still get a fresh truth.
    procs = processes(fresh, nonblocking=not fresh)
    state, ran = _managed(fresh)
    chosen, need = selected(), required()
    return [
        {
            "name": name,
            "description": svc.get("description", ""),
            "group": svc.get("group", ""),
            "default": bool(svc.get("default", False)),
            "required": name in need,
            "selected": name in chosen,
            "installed": name in state,
            "state": state.get(name, "not installed"),
            "scheduled": bool(svc.get("schedule")),
            "pids": procs.get(name, []),
            "restarts": ran.get(name, {}).get("restarts"),  # None: not installed, a job, or the manager didn't say
            "up_s": ran.get(name, {}).get("up_s"),  # how long the current run has lasted
            "run_id": ran.get(name, {}).get("run_id"),
            "last_error": last_error(name, since=_run_began(state.get(name), ran.get(name)), include_id=True),
        }
        for name, svc in declared().items()
        if not svc.get("platforms") or cfg.platform_key() in svc["platforms"]
    ]


def status() -> None:
    """The application's state for each service it runs; with none open, plainly that, and nothing else."""
    running = states()
    for name, state in running.items():
        print(f"{name:16} {state}")
    if not running:
        print(APPLICATION)


def choose(names: Sequence[str], *, enable: bool | None) -> tuple[list[str], list[str]]:
    """Edits pack.toml's [services] lists for `names`: `enable=True` adds them to enable (and takes them out of
    disable), `False` to disable (and out of enable), `None` takes them out of both, back to their defaults. Changes no
    service: install, stop, or start them after. Returns the new (enable, disable)."""
    unknown = sorted(set(names) - set(declared()))
    if unknown:
        raise SystemExit(f"not a declared service: {', '.join(unknown)} (see `pack-services list`)")
    on = [n for n in cfg.get("services.enable") if n not in names]
    off = [n for n in cfg.get("services.disable") if n not in names]
    if enable is True:
        on += list(names)
    elif enable is False:
        off += list(names)
    cfg.save({"services.enable": on, "services.disable": off})
    return on, off


def exec_service(name: str) -> None:
    """Resolve current settings at startup, then replace the launcher with a selected service's command.

    A service with `fresh_config` starts through this launcher rather than a snapshot of policy flags. Use the
    ordinary command here, without startup wrappers, so this cannot recurse through fresh_config or repeat the
    readiness wait. The service's templated env is filled in again too, over what was already set (the tier worker's
    video_archive settings).
    """
    cfg.load.cache_clear()
    services = selected()
    if name not in services:
        raise SystemExit(f"cannot launch unselected or unknown service: {name}")
    command = argv(services[name])
    try:
        os.chdir(cfg.workdir())
        os.execve(command[0], command, {**os.environ, **_templated_env(services[name])})
    except OSError as error:
        raise SystemExit(f"cannot start {name}: {error}") from error


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("installed", help="the services the running application runs, one name a line")
    execute = sub.add_parser("exec", help="start one selected service using current settings (internal launcher)")
    execute.add_argument("name")
    for c in ("enable", "disable", "reset"):
        p = sub.add_parser(c, help=f"{c} services in pack.toml's [services] (changes no running service)")
        p.add_argument("names", nargs="+")
    for c in ("install", "uninstall", "stop", "start", "restart"):
        p = sub.add_parser(c)
        p.add_argument("names", nargs="+" if c in ("stop", "start") else "*")  # never every service by default
        if c in {"restart", "install"}:
            p.add_argument("--cascade", action="store_true", help="stage the whole selected set")
    sub.add_parser("status")
    move = sub.add_parser("migrate", help="remove the per-service units of an earlier install (see MIGRATING.md)")
    move.add_argument("--dry-run", action="store_true", help="print exactly what would be done, and do nothing")
    move.add_argument("--disable-linger", action="store_true", help="also turn linger off for this user")
    sub.add_parser("trim-logs", help="rotate every log under paths.logs past its cap (the log-trim service)")
    a = ap.parse_args()
    if a.cmd == "trim-logs":
        for file, freed in logs.trim(log_path("x").parent).items():
            print(f"{file}: rotated, {freed / 1e6:.1f} MB freed")
        return
    if a.cmd == "migrate":
        from liberational import unit_migration

        return unit_migration.main(a.dry_run, a.disable_linger)
    if a.cmd == "exec":
        exec_service(a.name)
    elif a.cmd == "installed":
        print(*installed(), sep="\n")
    elif a.cmd == "list":
        chosen = selected()
        for name, svc in declared().items():
            print(f"{'*' if name in chosen else ' '} {name:16} {svc.get('description', '')}")
        print("\n* = selected on this machine")
    elif a.cmd in ("enable", "disable", "reset"):
        on, off = choose(a.names, enable={"enable": True, "disable": False, "reset": None}[a.cmd])
        print(f"pack.toml [services]: enable = {on}, disable = {off}")
    elif a.cmd == "install":
        install(a.names, cascade=a.cascade)
    elif a.cmd == "uninstall":
        uninstall(a.names)
    elif a.cmd == "stop":
        stop(a.names)
    elif a.cmd == "start":
        start(a.names)
    elif a.cmd == "restart":
        if not a.names and not a.cascade:
            ap.error("restart needs service names or --cascade")
        restart(a.names, cascade=a.cascade)
    else:
        status()


if __name__ == "__main__":
    main()
