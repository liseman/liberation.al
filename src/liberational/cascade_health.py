"""Read-only final stability and selected vision checks for a staged service cascade.

The runner supplies the first healthy snapshot. Never replace it with a later
healthy PID: a daemon or capture worker restarting during readiness is failure.
Scheduled jobs only need an armed schedule, since their short-lived PIDs and run
counts legitimately change. All manager calls share the caller's deadline.
"""

import subprocess
import sys
from collections.abc import Mapping, Sequence
from time import monotonic, sleep
from typing import TYPE_CHECKING

from liberational import config as cfg
from liberational import logs
from liberational.continuum_status import VISION

if TYPE_CHECKING:
    from liberational.service_cascade import Manager, ServiceFact


def _remaining(deadline: float) -> float:
    left = deadline - monotonic()
    if left <= 0:
        raise TimeoutError("service health deadline expired")
    return left


def _compare(names: Sequence[str], baseline: Mapping[str, ServiceFact], current: Mapping[str, ServiceFact]) -> None:
    from liberational.service_cascade import _stopped

    for name in names:
        old, new = baseline.get(name), current.get(name)
        if old is None or new is None:
            raise RuntimeError(f"{name}: service health snapshot missing")
        if old.kept_stopped:
            if not _stopped(new):
                raise RuntimeError(f"{name}: kept-stopped service unexpectedly started or failed ({new.state})")
            continue
        if old.scheduled:
            if not new.scheduled or new.state != "scheduled":
                raise RuntimeError(f"{name}: schedule is no longer armed ({new.state})")
            continue
        if new.state != "running":
            raise RuntimeError(f"{name}: service is not running ({new.state})")
        if old.pid is None or old.pid.pid <= 0 or new.pid is None or new.pid.pid <= 0:
            raise RuntimeError(f"{name}: daemon process identity unavailable")
        if new.pid != old.pid:
            raise RuntimeError(f"{name}: daemon process identity changed during service health")
        if old.restarts is None or new.restarts is None:
            raise RuntimeError(f"{name}: restart counter unavailable")
        if new.restarts != old.restarts:
            raise RuntimeError(f"{name}: restart counter changed during service health")
        if name == "continuum-capture" and set(new.children) != set(old.children):
            raise RuntimeError(f"{name}: capture worker identities changed during service health")


def _snapshot(manager: Manager, names: Sequence[str], baseline: Mapping[str, ServiceFact], deadline: float) -> None:
    _remaining(deadline)
    current = manager.snapshot(names, deadline)
    _remaining(deadline)
    _compare(names, baseline, current)


log = logs.get("cascade-health")  # the restart command shows these on stderr, beside its own output


def _diagnostic(stdout: str | bytes | None, stderr: str | bytes | None) -> str:
    def text(value: str | bytes | None) -> str:
        return value.decode(errors="replace") if isinstance(value, bytes) else value or ""

    return (text(stdout) + "\n" + text(stderr)).strip()[-4096:] or "no status diagnostics"


def _report(names: Sequence[str], baseline: Mapping[str, ServiceFact]) -> None:
    kept = sum(baseline[name].kept_stopped for name in names)
    armed = sum(baseline[name].scheduled and not baseline[name].kept_stopped for name in names)
    running = len(names) - kept - armed
    log.info(f"service health: {running} running-ready, {kept} stopped (kept), {armed} scheduled armed")
    for name in names:
        fact = baseline[name]
        if fact.kept_stopped:
            log.info(f"  {name}: stopped (kept); state={fact.state}")
            continue
        if fact.scheduled:
            log.info(f"  {name}: schedule armed")
            continue
        pid = fact.pid
        if pid is None:
            raise RuntimeError(f"{name}: daemon process identity unavailable")
        log.info(f"  {name}: pid={pid.pid} created_at={pid.created_at!r} restarts={fact.restarts} unchanged")
        if name == "continuum-capture":
            children = ", ".join(
                f"pid={child.pid} created_at={child.created_at!r}"
                for child in sorted(fact.children, key=lambda child: (child.pid, child.created_at))
            )
            log.info(f"    capture workers: {children or 'none'}; unchanged")


def wait_for_health(
    manager: Manager, names: Sequence[str], baseline: Mapping[str, ServiceFact], deadline: float
) -> None:
    """Require one stable second, then selected vision health, keeping the original baseline throughout."""
    names = tuple(dict.fromkeys(names))
    baseline = dict(baseline)
    _compare(names, baseline, baseline)
    stable_until = monotonic() + 1.0
    while True:
        _snapshot(manager, names, baseline, deadline)
        left = stable_until - monotonic()
        if left <= 0:
            break
        sleep(min(0.1, left, _remaining(deadline)))
    active = tuple(name for name in names if not baseline[name].kept_stopped)
    if any(name.startswith("see-") for name in active):
        args = [sys.executable, "-m", "liberational.continuum_status", "--port", "vision", "--settle", "60"]
        if not set(VISION).issubset(active):
            args.append("--selected-only")
            for name in VISION:
                if name in active:
                    args.extend(("--expected-service", name))
        # Check on both sides: a successful probe cannot conceal a daemon/worker restart.
        _snapshot(manager, names, baseline, deadline)
        try:
            result = subprocess.run(args, cwd=cfg.ROOT, capture_output=True, text=True, timeout=_remaining(deadline))
        except subprocess.TimeoutExpired as error:
            raise TimeoutError(f"vision health probe timed out: {_diagnostic(error.stdout, error.stderr)}") from error
        if result.returncode != 0:
            raise RuntimeError(
                f"vision health probe exited {result.returncode}: {_diagnostic(result.stdout, result.stderr)}"
            )
        _snapshot(manager, names, baseline, deadline)
    _remaining(deadline)
    _report(names, baseline)
