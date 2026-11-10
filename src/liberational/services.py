"""Render and install the pack's services from services/services.toml.

  uv run pack-services list                 # what would be installed on this machine
  uv run pack-services render [--out DIR]   # print (or write) the launchd plists / systemd units
  uv run pack-services install [NAME ...]   # write them to the user's service dir and start them
  uv run pack-services uninstall [NAME ...] # stop and remove them
  uv run pack-services restart --cascade   # staged restart of the selected installed services
  uv run pack-services status

The dashboard's admin sidebar shows snapshot() and turns services on and off with install and uninstall.

macOS gets launchd agents labelled pack.<name> in ~/Library/LaunchAgents; Linux gets systemd user
units named pack-<name> in ~/.config/systemd/user. Which services are installed: each one's
`default`, adjusted by [services] enable / disable in pack.toml, limited to its platforms.
"""

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import plistlib
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from collections.abc import Iterator, Sequence
from typing import Any, Literal, TypedDict

from liberational import config as cfg
from liberational import logs

SPEC = cfg.ROOT / "services" / "services.toml"
DARWIN = sys.platform == "darwin"

# One [name] table from services.toml: command, args, env, default, platforms, schedule, and so on.
type Service = dict[str, Any]


def declared() -> dict[str, Service]:
    return tomllib.loads(SPEC.read_text())


def required() -> set[str]:
    """Services the current settings depend on, whether or not [services] enables them."""
    return {"whisper"} if str(cfg.get("asr.engine")).lower() == "whisper" else set()


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


def _expand(arg: str) -> str:
    def sub(m: re.Match[str]) -> str:
        key = m[1]
        if key.startswith("env:"):
            _, var, default = (key.split(":", 2) + [""])[:3]
            return os.environ.get(var) or default
        return str(cfg.get(key))

    return os.path.expanduser(_TEMPLATE.sub(sub, arg))


def argv(svc: Service, *, startup: bool = False, name: str = "") -> list[str]:
    """Absolute command line for a service. At startup, its wrappers too: a service with `log_relay = true` (a
    program we don't own) runs under log_relay, tagged with its name, so its lines carry the one log format."""
    cmd = svc["command"]
    if cmd.startswith("pack-"):
        exe = [str(cfg.ROOT / ".venv" / "bin" / cmd)]
    elif "/" in cmd:
        path = cfg.ROOT / cmd
        exe = ["/bin/bash", str(path)] if path.suffix == ".sh" else [str(path)]
    else:
        exe = [shutil.which(cmd) or cmd]
    if startup and svc.get("fresh_config"):
        command = [
            str(cfg.ROOT / ".venv/bin/python"),
            "-m",
            "liberational.services",
            "exec",
            str(svc["fresh_config"]),
        ]
    else:
        command = exe + [_expand(a) for a in svc.get("args", [])]
    if startup and svc.get("wait_for_gpu"):
        command = [str(cfg.ROOT / ".venv/bin/python"), "-m", "liberational.gpu_ready", "--", *command]
    if startup and svc.get("wait_for_broker"):
        command = [
            str(cfg.ROOT / ".venv/bin/python"),
            "-m",
            "liberational.continuum_ready",
            "--socket",
            str(cfg.path("paths.state") / "continuum/broker.sock"),
            "--",
            *command,
        ]
    if startup and svc.get("log_relay"):
        command = [str(cfg.ROOT / ".venv/bin/python"), "-m", "liberational.log_relay", "--tag", name, "--", *command]
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


def environment(svc: Service, name: str = "") -> dict[str, str]:
    # The venv first, then mise's shims (tools like ffmpeg from mise.toml), then the installing shell's PATH, then the
    # system directories whatever that shell had: capture runs sysctl, which is in /usr/sbin.
    parts = [str(cfg.ROOT / ".venv" / "bin"), str(pathlib.Path.home() / ".local/share/mise/shims")]
    parts += os.environ.get("PATH", "").split(os.pathsep)
    parts += ["/usr/local/bin"] if DARWIN else []
    parts += ["/usr/bin", "/bin", "/usr/sbin", "/sbin"]
    env = {"PATH": os.pathsep.join(dict.fromkeys(p for p in parts if p)), "HOME": str(pathlib.Path.home())}
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


OWN_LOGS = frozenset({"cascade", "frontend"})  # logs under paths.logs that the pack writes but no service owns


def prune_logs() -> list[str]:
    """Delete the logs, and their older copies, of services services.toml no longer declares. Returns the file
    names deleted."""
    keep = set(declared()) | OWN_LOGS
    directory = log_path("x").parent
    gone = []
    for path in sorted([*directory.glob("*.log"), *directory.glob("*.log" + logs.OLDER)]):
        if path.name.removesuffix(logs.OLDER).removesuffix(".log") not in keep and path.is_file():
            path.unlink()
            gone.append(path.name)
    return gone


def render_launchd(name: str, svc: Service) -> dict[str, str]:
    d: dict[str, Any] = {
        "Label": f"pack.{name}",
        "ProgramArguments": argv(svc, startup=True, name=name),
        "WorkingDirectory": str(cfg.ROOT),
        "EnvironmentVariables": environment(svc, name),
        "StandardOutPath": str(log_path(name)),
        "StandardErrorPath": str(log_path(name)),
        # Seconds after SIGTERM before launchd kills it, the same as systemd's TimeoutStopSec.
        "ExitTimeOut": int(svc.get("stop_timeout", 20)),
    }
    if svc.get("schedule"):
        hour, minute = svc["schedule"].split(":")
        # "*:MM" runs every hour, as systemd reads the same OnCalendar time.
        d["StartCalendarInterval"] = {"Minute": int(minute)} | ({} if hour == "*" else {"Hour": int(hour)})
    else:
        d["RunAtLoad"] = True
        if svc.get("restart", True):
            d["KeepAlive"] = True
            d["ThrottleInterval"] = svc.get("throttle", 10)
    if svc.get("background"):
        d["ProcessType"] = "Background"
        d["LowPriorityIO"] = True
    return {f"pack.{name}.plist": plistlib.dumps(d).decode()}


def render_systemd(name: str, svc: Service) -> dict[str, str]:
    # systemd opens StandardOutput before ExecStartPre, so mkdir there cannot repair
    # a removed log directory. Open the log only after mkdir, then exec to preserve signals.
    log = log_path(name)
    script = 'mkdir -p -- "$1" && shift && log=$1 && shift && exec "$@" >> "$log" 2>&1'
    command = ["/bin/sh", "-c", script, "pack-log", str(log.parent), str(log), *argv(svc, startup=True, name=name)]
    # ':' disables systemd's $ expansion; %% preserves literal % specifiers in paths/args.
    start = "ExecStart=:" + " ".join(shlex.quote(a).replace("%", "%%") for a in command)
    env = "\n".join(f"Environment={shlex.quote(f'{k}={v}')}" for k, v in environment(svc, name).items())
    scheduled = bool(svc.get("schedule"))
    lines = [
        "[Unit]",
        f"Description=Pack: {svc.get('description', name)}",
        "",
        "[Service]",
        f"Type={'oneshot' if scheduled else 'simple'}",
        start,
        f"WorkingDirectory={cfg.ROOT}",
        env,
        "StandardOutput=journal",
        "StandardError=journal",
        # Seconds after SIGTERM before systemd kills it; its 90 s default could hold a stop from Settings' Services.
        f"TimeoutStopSec={svc.get('stop_timeout', 20)}",
    ]
    if not scheduled and svc.get("restart", True):
        lines += ["Restart=always", f"RestartSec={svc.get('throttle', 10)}"]
    if svc.get("background"):
        lines += ["Nice=10", "IOSchedulingClass=idle"]
    units = {
        f"pack-{name}.service": "\n".join(lines + ([] if scheduled else ["", "[Install]", "WantedBy=default.target"]))
        + "\n"
    }
    if scheduled:
        units[f"pack-{name}.timer"] = (
            "\n".join(
                [
                    "[Unit]",
                    f"Description=Pack: {svc.get('description', name)} (schedule)",
                    "",
                    "[Timer]",
                    f"OnCalendar=*-*-* {svc['schedule']}:00",
                    "Persistent=true",
                    "",
                    "[Install]",
                    "WantedBy=timers.target",
                ]
            )
            + "\n"
        )
    return units


def render(name: str, svc: Service) -> dict[str, str]:
    return render_launchd(name, svc) if DARWIN else render_systemd(name, svc)


def unit_dir() -> pathlib.Path:
    return pathlib.Path.home() / ("Library/LaunchAgents" if DARWIN else ".config/systemd/user")


def _run(*cmd: str, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def _systemctl(verb: Sequence[str], units: Sequence[str]) -> dict[str, str | None]:
    """Run one systemctl verb on every unit in a single call (they stop or start in parallel), and say per unit
    whether it worked: None, or the error. Only when the batch fails is each unit looked at, to tell which: start and
    restart ask whether it's running (running them again would restart the healthy ones twice), and the rest are
    tried one at a time.
    One call matters: enable and disable each reload the whole manager unless told not to (about 0.7 s apiece on
    the Linux machine), which per unit made All off take about 8 s for nine services."""
    if not units:
        return {}
    batch = _run("systemctl", "--user", *verb, *units)
    if batch.returncode == 0:
        return dict.fromkeys(units)
    out: dict[str, str | None] = {}
    for unit in units:
        if verb[0] in ("start", "restart"):  # don't run it again: that would restart the healthy ones twice
            up = _run("systemctl", "--user", "is-active", unit).returncode == 0
            out[unit] = None if up else (batch.stderr or "").strip() or "failed"
        else:  # enable and disable are all or nothing as a batch; one at a time is harmless
            r = _run("systemctl", "--user", *verb, unit)
            out[unit] = None if r.returncode == 0 else (r.stderr or "").strip() or "failed"
    return out


def checkout_note() -> str:
    """Why services must not be installed from here: a git worktree or a temporary directory isn't the checkout they
    should run from (their units would point at files that go away). Empty for a real checkout."""
    root = cfg.ROOT.resolve()
    if (root / ".git").is_file():  # a linked worktree: .git is a pointer file, not the repository
        return f"{root} is a git worktree"
    tmp = pathlib.Path(tempfile.gettempdir()).resolve()
    if root == tmp or tmp in root.parents or root.parts[:2] == ("/", "tmp"):
        return f"{root} is under a temporary directory"
    return ""


def _check_checkout(anywhere: bool) -> None:
    if (why := checkout_note()) and not anywhere:
        raise SystemExit(
            f"not touching services: {why}. Liberational's services run from the checkout they're installed from;"
            " run this from the real checkout, or pass --anywhere if you mean it."
        )


def preflight(names: Sequence[str]) -> None:
    """Warn about system packages these services need and this machine is missing (config/system-packages.toml),
    with the command that installs them. It doesn't stop the install: some packages matter only to some features."""
    from liberational import system_packages

    try:
        gone = system_packages.missing(system_packages.load(), services=names)
    except (OSError, ValueError) as e:
        print(f"warning: can't check system packages: {e}")
        return
    if gone:
        print(
            "warning: missing system packages these services use:\n"
            + system_packages.describe(gone)
            + "\nOr run `uv run pack-system install`, which asks before installing."
        )


def _bootstrap(name: str) -> str | None:
    """Load a launchd job from its plist: None, or the error. A bootout returns before the job is gone; a bootstrap in
    that window fails with "Input/output error" (5) and the service stays down (ambient died this way after an
    owner-voice merge). So wait for the label to disappear, then retry the bootstrap a few times."""
    domain = f"gui/{os.getuid()}"
    for _ in range(40):
        if _run("launchctl", "print", f"{domain}/pack.{name}").returncode != 0:
            break
        time.sleep(0.25)
    r = None
    for attempt in range(6):
        r = _run("launchctl", "bootstrap", domain, str(unit_dir() / f"pack.{name}.plist"))
        if r.returncode == 0 or "already" in (r.stderr or "").lower():
            return None
        time.sleep(0.5 * (attempt + 1))
    return (r.stderr or "").strip() if r else "not tried"


def install(names: Sequence[str], anywhere: bool = False, *, cascade: bool = False) -> None:
    _check_checkout(anywhere)
    log_path("x").parent.mkdir(parents=True, exist_ok=True)
    for gone in prune_logs():
        print(f"removed {gone}: no service by that name is declared")
    services = selected()
    full = not names
    names = names or list(services)
    missing = [n for n in names if n not in services]
    if missing:
        raise SystemExit(f"not selected for this machine: {', '.join(missing)} (see `pack-services list`)")
    if cascade or {"continuum-broker", "continuum-capture"}.intersection(names):
        from liberational import service_cascade

        service_cascade.run_selected(list(names), install=True, anywhere=anywhere, start_stopped=full or cascade)
        return
    preflight(names)
    log_path("x").parent.mkdir(parents=True, exist_ok=True)
    unit_dir().mkdir(parents=True, exist_ok=True)
    for name in names:
        for fname, text in render(name, services[name]).items():
            (unit_dir() / fname).write_text(text)
        if DARWIN:
            _run("launchctl", "bootout", f"gui/{os.getuid()}/pack.{name}")
            err = _bootstrap(name)
            print(f"pack.{name}: {'loaded' if err is None else 'FAILED ' + err}")
    if not DARWIN:
        _run("systemctl", "--user", "daemon-reload", check=True)  # once, so enable needn't reload per unit
        unit = {n: f"pack-{n}.timer" if services[n].get("schedule") else f"pack-{n}.service" for n in names}
        enabled = _systemctl(["enable", "--no-reload"], list(unit.values()))
        ok = [u for u in unit.values() if enabled[u] is None]
        # A timer starts its schedule; a service restarts, not just starts, so a reinstall picks up new settings.
        ran = _systemctl(["start"], [u for u in ok if u.endswith(".timer")])
        ran |= _systemctl(["restart"], [u for u in ok if u.endswith(".service")])
        for u in unit.values():
            err = enabled[u] or ran.get(u)
            print(f"{u}: {'running' if err is None else 'FAILED ' + err}")
        if _run("loginctl", "show-user", os.environ.get("USER", ""), "-p", "Linger", "--value").stdout.strip() == "no":
            print("note: user services only run while you're logged in; to start them at boot: loginctl enable-linger")


def installed() -> list[str]:
    pat = "pack.*.plist" if DARWIN else "pack-*.service"
    return sorted(re.sub(r"^pack[.-]|\.(plist|service)$", "", p.name) for p in unit_dir().glob(pat))


def uninstall(names: Sequence[str], anywhere: bool = False) -> None:
    _check_checkout(anywhere)
    names = list(names) or installed()  # before any unit file goes
    for name in names:
        if DARWIN:
            _run("launchctl", "bootout", f"gui/{os.getuid()}/pack.{name}")
            (unit_dir() / f"pack.{name}.plist").unlink(missing_ok=True)
            print(f"removed {name}")
    if not DARWIN:
        # Every unit at once: one stop (in parallel), one disable without a reload, then one reload.
        units = [
            u for name in names for u in (f"pack-{name}.timer", f"pack-{name}.service") if (unit_dir() / u).exists()
        ]
        stopped = _systemctl(["stop"], units)
        disabled = _systemctl(["disable", "--no-reload"], units)
        for u in units:
            (unit_dir() / u).unlink(missing_ok=True)
        for name in names:
            err = next(
                (e for u in units if u.startswith(f"pack-{name}.") for e in (stopped[u], disabled[u]) if e), None
            )
            print(f"removed {name}" if err is None else f"pack-{name}: FAILED {err}")
        _run("systemctl", "--user", "daemon-reload")


def _installed_only(names: Sequence[str]) -> list[str]:
    """names, refusing any service that isn't installed (stop and start act on an installed unit; install adds one)."""
    have = set(installed())
    missing = [n for n in names if n not in have]
    if missing:
        raise SystemExit(f"not installed: {', '.join(missing)} (turn it on with `pack-services install`)")
    return list(names)


def stop(names: Sequence[str], anywhere: bool = False) -> None:
    """Stop services now, leaving them installed and enabled: they start again when the computer restarts
    (systemd starts the user's units at boot with lingering on, else at the first login; launchd loads the kept plist
    at the next login), on Repair, or on `start`. `uninstall` is what keeps one off."""
    _check_checkout(anywhere)
    names = _installed_only(names)
    if DARWIN:
        for name in names:
            # Unloaded, with its plist kept: launchd loads it again at the next login. (A kill would only make
            # KeepAlive start it again at once.)
            r = _run("launchctl", "bootout", f"gui/{os.getuid()}/pack.{name}")
            print(f"pack.{name}: {'stopped' if r.returncode == 0 else 'FAILED ' + r.stderr.strip()}")
        return
    units = [u for n in names for u in (f"pack-{n}.timer", f"pack-{n}.service") if (unit_dir() / u).exists()]
    done = _systemctl(["stop"], units)
    for n in names:  # a scheduled job is named by its timer, which is what brings it back
        u = f"pack-{n}.timer" if (unit_dir() / f"pack-{n}.timer").exists() else f"pack-{n}.service"
        err = done.get(u) or done.get(f"pack-{n}.service")
        print(f"{u}: {'stopped' if err is None else 'FAILED ' + err}")


def start(names: Sequence[str], anywhere: bool = False) -> None:
    """Start installed services that were stopped (a scheduled job's timer starts its schedule)."""
    _check_checkout(anywhere)
    names = _installed_only(names)
    if DARWIN:
        domain = f"gui/{os.getuid()}"
        for name in names:
            if _run("launchctl", "print", f"{domain}/pack.{name}").returncode == 0:
                # Still loaded (between KeepAlive restarts, or a calendar job waiting): a bootstrap would fail with
                # "Input/output error", so start it where it is.
                r = _run("launchctl", "kickstart", f"{domain}/pack.{name}")
                err = None if r.returncode == 0 else (r.stderr or "").strip()
            else:
                err = _bootstrap(name)
            print(f"pack.{name}: {'running' if err is None else 'FAILED ' + err}")
        return
    units = [f"pack-{n}.timer" if (unit_dir() / f"pack-{n}.timer").exists() else f"pack-{n}.service" for n in names]
    done = _systemctl(["start"], units)
    for u in units:
        print(f"{u}: {'running' if done.get(u) is None else 'FAILED ' + str(done[u])}")


def restart(names: Sequence[str], anywhere: bool = False, *, cascade: bool = False) -> None:
    """Restart installed services. Core changes always use the same strict selected/installed cascade as install."""
    from liberational import service_cascade

    _check_checkout(anywhere)
    chosen, have = selected(), set(installed())
    missing = set(names) - (chosen.keys() & have)
    if missing:
        raise SystemExit("not selected and installed: " + ", ".join(sorted(missing)))
    if cascade:
        names = [name for name in chosen if name in have]
    elif not names:
        raise SystemExit("restart needs service names or --cascade")
    if cascade or {service_cascade.BROKER, service_cascade.CAPTURE}.intersection(names):
        service_cascade.run_selected(list(names), anywhere=anywhere, start_stopped=cascade)
        return
    # Leaf-only changes remain selective and do not cycle capture or the broker.
    deadline = time.monotonic() + service_cascade.DEFAULT_TIMEOUT
    manager = service_cascade.ServiceManager(chosen)
    try:
        before = manager.snapshot(names, deadline)
        manager.stop(names, deadline)
        manager.wait_stopped(before, deadline)
        manager.start(names, deadline)
        service_cascade.wait_running_snapshot(manager, names, deadline)
    except Exception as error:
        # Start them again before failing, so a stop that half worked doesn't leave them unloaded.
        retry = time.monotonic() + service_cascade.RECOVERY_WAIT
        failures = service_cascade.start_each(manager, names, retry)
        detail = str(error) + "".join(f"; recovery start failed: {f}" for f in failures)
        raise service_cascade.CascadeError("restart " + ", ".join(names), detail) from error
    for name in names:
        print(f"pack-{name}: running")


def states() -> dict[str, str]:
    """The service manager's state for each installed service (systemd: active, failed, ...; launchd: running, ...)."""
    names = installed()
    if DARWIN:
        out: dict[str, str] = {}
        for name in names:
            r = _run("launchctl", "print", f"gui/{os.getuid()}/pack.{name}")
            out[name] = next(
                (l.split("=", 1)[1].strip() for l in r.stdout.splitlines() if l.strip().startswith("state =")),
                "not loaded",
            )
        return out
    # Poll every unit in one manager request instead of spawning a process per service. A scheduled job's service
    # is idle between runs, so ask about its timer. Missing replies stay unknown rather than implying inactivity.
    units = {
        (timer.name if (timer := unit_dir() / f"pack-{name}.timer").exists() else f"pack-{name}.service"): name
        for name in names
    }
    out = dict.fromkeys(names, "unknown")
    if not units:
        return out
    r = _run("systemctl", "--user", "show", "-p", "Id", "-p", "ActiveState", *units)
    for block in r.stdout.split("\n\n"):
        props = dict(line.split("=", 1) for line in block.splitlines() if "=" in line)
        if (name := units.get(props.get("Id", ""))) is not None:
            out[name] = props.get("ActiveState") or "unknown"
    return out


LOG_TAIL_BYTES = 64 * 1024  # how much of the end of a service's log last_error reads
# A line's stamp: an ISO local date and time with offset, or in lines written before it, the time alone.
STAMP = r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d|\d\d:\d\d:\d\d)"
ERROR_LINE = re.compile(STAMP + r" \[[^\]]*\] ERROR (.*)")
# Any line logs.py or the broker wrote, and the stamped free text continuum-capture and the tier worker write.
TIMED = re.compile(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d) |(\d\d:\d\d:\d\d) \[")


def _launchd(name: str) -> dict[str, str]:
    """`launchctl print`'s key = value lines for one service, stripped."""
    r = _run("launchctl", "print", f"gui/{os.getuid()}/pack.{name}")
    out = {}
    for line in r.stdout.splitlines():
        if " = " in line:
            k, v = line.split(" = ", 1)
            out.setdefault(k.strip(), v.strip())  # the first: nested blocks repeat keys
    return out


def runs() -> dict[str, dict[str, Any]]:
    """How each installed daemon has run: {"restarts": how often the service manager restarted it, "up_s": how
    long its current run has lasted, or None}. A crash loop is restarts climbing with a short current run; restarts
    alone can't tell one, since systemd's count only resets when the unit is started by hand. run_id identifies the
    current run without changing as uptime advances, so acknowledging its history cannot hide a later run.

    systemd: NRestarts and ActiveEnterTimestampMonotonic for every unit in one systemctl call. launchd: its runs after
    the first, and the age of its pid. Scheduled jobs, and services with restart = false, are left out: their runs
    are by design."""
    spec = declared()
    names = [n for n in installed() if not spec.get(n, {}).get("schedule") and spec.get(n, {}).get("restart", True)]
    if not names:
        return {}
    if DARWIN:
        import psutil

        out: dict[str, dict[str, Any]] = {}
        for name in names:
            info = _launchd(name)
            if not info.get("runs", "").isdigit():
                continue
            up, started = None, None
            if info.get("pid", "").isdigit():
                try:
                    started = psutil.Process(int(info["pid"])).create_time()
                    up = max(0.0, time.time() - started)
                except psutil.Error, OSError:
                    up = None
            out[name] = {
                "restarts": max(0, int(info["runs"]) - 1),
                "up_s": up,
                "run_id": f"launchd:{info['runs']}:{info.get('pid', '')}:{started}",
            }
        return out
    units = [f"pack-{n}.service" for n in names]
    r = _run(
        "systemctl",
        "--user",
        "show",
        "-p",
        "Id",
        "-p",
        "NRestarts",
        "-p",
        "ActiveEnterTimestampMonotonic",
        "-p",
        "ActiveEnterTimestamp",
        *units,
    )
    if r.returncode != 0:
        return {}
    out = {}
    for block in r.stdout.split("\n\n"):  # one block per unit; the properties in whatever order systemd prints them
        props = dict(line.split("=", 1) for line in block.splitlines() if "=" in line)
        unit, n, since = props.get("Id", ""), props.get("NRestarts", ""), props.get("ActiveEnterTimestampMonotonic", "")
        if unit not in units or not n.isdigit():
            continue
        up = time.monotonic() - int(since) / 1e6 if since.isdigit() and int(since) > 0 else None
        out[unit[len("pack-") : -len(".service")]] = {
            "restarts": int(n),
            "up_s": up,
            "run_id": f"systemd:{n}:{since}:{props.get('ActiveEnterTimestamp', '')}",
        }
    return out


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
UP_STATES = {"active", "running"}  # systemd's and launchd's word for a service that's running
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
        elif "/" in cmd:
            by_path[str(cfg.ROOT / cmd)] = name
            by_path[str((cfg.ROOT / cmd).resolve())] = name
    return by_name, by_path


def _scan_processes() -> dict[str, list[int]]:
    """Pids of running services by name, from every process (proc_table, each one's command line read once): each
    service's own command, and the processes it started (continuum-capture runs one worker per device), which count as
    it."""
    from liberational import proc_table

    by_name, by_path = _commands()
    out: dict[str, list[int]] = {}
    others: list[tuple[int, int | None]] = []  # (pid, parent pid) of the processes no command matched
    for p in proc_table.table().values():
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
    """Pids of running services by name (as _scan_processes finds them), whether the service manager started them or a
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


# What the service manager said of every unit (states() and runs(), one systemctl call each), kept for a few seconds:
# several pages poll /services at once, and a state changes only when a service starts, stops, or fails.
MANAGER_TTL_S = 5.0
_manager_lock = threading.Lock()
_manager: dict[str, Any] = {"at": None, "states": {}, "runs": {}}


def _managed(fresh: bool = False) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """(states(), runs()) for a status poll: the last answer when it is under MANAGER_TTL_S old, with each run's
    `up_s` moved on by the answer's age. fresh=True (an action's reply) asks the manager now. Polls that arrive
    together wait for one read instead of each starting their own."""
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
    (fresh as in processes()). A poll reuses the service manager's last answer for MANAGER_TTL_S (`_managed`) and
    opens only the logs written to since the last one (`_newest_error`)."""
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
    for name, state in states().items():
        print(f"{name:16} {state}")


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

    Installed units keep this launcher rather than a snapshot of policy flags. Use the ordinary command here,
    without startup wrappers, so this cannot recurse through fresh_config or repeat the readiness wait. The
    service's templated env is filled in again too, over what the unit was installed with (the tier worker's
    video_archive settings).
    """
    cfg.load.cache_clear()
    services = selected()
    if name not in services:
        raise SystemExit(f"cannot launch unselected or unknown service: {name}")
    command = argv(services[name])
    try:
        os.chdir(cfg.ROOT)
        os.execve(command[0], command, {**os.environ, **_templated_env(services[name])})
    except OSError as error:
        raise SystemExit(f"cannot start {name}: {error}") from error


def main() -> None:
    from liberational.service_cascade import CascadeError

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("installed", help="the services installed here, one name a line")
    execute = sub.add_parser("exec", help="start one selected service using current settings (internal launcher)")
    execute.add_argument("name")
    r = sub.add_parser("render")
    r.add_argument("--out", type=pathlib.Path)
    for c in ("enable", "disable", "reset"):
        p = sub.add_parser(c, help=f"{c} services in pack.toml's [services] (changes no running service)")
        p.add_argument("names", nargs="+")
    for c in ("install", "uninstall", "stop", "start", "restart"):
        p = sub.add_parser(c)
        p.add_argument("names", nargs="+" if c in ("stop", "start") else "*")  # never every service by default
        p.add_argument("--anywhere", action="store_true", help="even from a git worktree or a temporary directory")
        if c in {"restart", "install"}:
            p.add_argument("--cascade", action="store_true", help="stage the whole selected installed set")
    sub.add_parser("status")
    sub.add_parser("trim-logs", help="rotate every log under paths.logs past its cap (the log-trim service)")
    a = ap.parse_args()
    if a.cmd == "trim-logs":
        for file, freed in logs.trim(log_path("x").parent).items():
            print(f"{file}: rotated, {freed / 1e6:.1f} MB freed")
        return
    if a.cmd == "exec":
        exec_service(a.name)
    elif a.cmd == "installed":
        print(*installed(), sep="\n")
    elif a.cmd == "list":
        chosen = selected()
        for name, svc in declared().items():
            print(f"{'*' if name in chosen else ' '} {name:16} {svc.get('description', '')}")
        print("\n* = installed by `pack-services install` on this machine")
    elif a.cmd == "render":
        for name, svc in selected().items():
            for fname, text in render(name, svc).items():
                if a.out:
                    a.out.mkdir(parents=True, exist_ok=True)
                    (a.out / fname).write_text(text)
                else:
                    print(f"# --- {fname}\n{text}")
        if a.out:
            print(f"wrote {a.out}")
    elif a.cmd in ("enable", "disable", "reset"):
        on, off = choose(a.names, enable={"enable": True, "disable": False, "reset": None}[a.cmd])
        print(f"pack.toml [services]: enable = {on}, disable = {off}")
    elif a.cmd == "install":
        try:
            install(a.names, a.anywhere, cascade=a.cascade)
        except CascadeError as error:
            print(f"cascade: FAILED {error}", file=sys.stderr)
            raise SystemExit(1) from error
    elif a.cmd == "uninstall":
        uninstall(a.names, a.anywhere)
    elif a.cmd == "stop":
        stop(a.names, a.anywhere)
    elif a.cmd == "start":
        start(a.names, a.anywhere)
    elif a.cmd == "restart":
        if not a.names and not a.cascade:
            ap.error("restart needs service names or --cascade")
        try:
            restart(a.names, a.anywhere, cascade=a.cascade)
        except CascadeError as error:
            print(f"cascade: FAILED {error}", file=sys.stderr)
            raise SystemExit(1) from error
    else:
        status()


if __name__ == "__main__":
    main()
