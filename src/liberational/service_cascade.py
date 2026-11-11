"""Portable staged service restarts, with strict readiness and process identity checks.

A successful stop or start is not readiness. A cascade stops readers before capture, lets the tier worker flush what
capture left in Tier 0, then stops it and the broker, proves old processes gone, then starts broker, capture, and
readers across explicit SDK boundaries. Diagnostics stay available
through core stages. Every operation, and every wait inside it, has a deadline; errors retain their stage.

The application (supervisor.py) is the only Manager: the cascade walks its own children, so there is no
operating-system unit to render, enable, or reload.
"""

import asyncio
import itertools
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from liberational import config as cfg
from liberational import logs

type BrokerIdentity = frozenset[tuple[str, int]]

BROKER = "continuum-broker"
CAPTURE = "continuum-capture"
TIER = "continuum-tier-worker"
#: How long a cascade waits for the tier to drain Tier 0 once capture stopped. Past it the restart goes on: some
#: frames strand, but a stuck tier never blocks a deploy.
DRAIN_TIMEOUT = 20.0
DIAGNOSTICS = frozenset({"dashboard", "continuum-gateway"})
# A cold broker may rebuild its entire availability index before accepting requests. Leave bounded room for
# that work plus capture evidence and final health checks; callers can still request a shorter deadline.
DEFAULT_TIMEOUT = 600.0
RECOVERY_RESERVE = 10.0
# After a failure, how long recovery waits for the services it started before reporting.
RECOVERY_WAIT = 15.0


def subprocess_timeout() -> float:
    """Bound an outer controller without killing the cascade's own recovery.

    The cascade reserves recovery inside its main deadline, while leaf restarts may recover after
    theirs. Allow both recovery windows plus finite interpreter/service-controller exit overhead.
    """
    return DEFAULT_TIMEOUT + RECOVERY_RESERVE + RECOVERY_WAIT + 30.0


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    pid: int
    created_at: float


@dataclass(frozen=True, slots=True)
class ServiceFact:
    name: str
    state: str
    pid: ProcessIdentity | None
    restarts: int | None
    children: tuple[ProcessIdentity, ...] = ()
    scheduled: bool = False
    kept_stopped: bool = False
    # Children the service replaces by design while healthy: capture's RTSP workers, which its supervisor starts again
    # whenever a network camera's clock fit fails or its stream stalls.
    replaceable: frozenset[ProcessIdentity] = frozenset()


class CascadeError(RuntimeError):
    def __init__(self, stage: str, detail: str) -> None:
        self.stage = stage
        super().__init__(f"{stage}: {detail}")


class Manager(Protocol):
    def snapshot(self, names: Sequence[str], deadline: float) -> dict[str, ServiceFact]: ...
    def stop(self, names: Sequence[str], deadline: float) -> None: ...
    def start(self, names: Sequence[str], deadline: float) -> None: ...
    def wait_stopped(self, before: Mapping[str, ServiceFact], deadline: float) -> None: ...


class Readiness(Protocol):
    def previous(self, deadline: float) -> BrokerIdentity | None: ...
    def broker(self, previous: BrokerIdentity | None, deadline: float) -> BrokerIdentity: ...
    def capture(self, broker: BrokerIdentity, deadline: float) -> None: ...
    def verify(self, broker: BrokerIdentity, deadline: float) -> None: ...
    def drain(self, deadline: float) -> int: ...


type Health = Callable[[Manager, Sequence[str], Mapping[str, ServiceFact], float], None]


@dataclass(frozen=True, slots=True)
class CascadeReport:
    names: tuple[str, ...]
    retained: tuple[str, ...]


def remaining(deadline: float, cap: float | None = None) -> float:
    left = deadline - time.monotonic()
    if left <= 0:
        raise TimeoutError("cascade deadline expired")
    return min(left, cap) if cap is not None else left


def wait_running_snapshot(manager: Manager, names: Sequence[str], deadline: float) -> dict[str, ServiceFact]:
    """Obtain a healthy post-start baseline before the separate stability/vision check."""
    while True:
        facts = manager.snapshot(names, deadline)
        pending = []
        for name in names:
            fact = facts.get(name)
            if fact is None:
                raise CascadeError("starting services", f"{name}: manager returned no state")
            if fact.state in {"failed", "error"}:
                raise CascadeError("starting services", f"{name}: manager state {fact.state}")
            if fact.scheduled:
                if fact.state != "scheduled":
                    pending.append(f"{name}: {fact.state}")
            elif fact.state != "running" or fact.pid is None or fact.pid.pid <= 0:
                pending.append(f"{name}: {fact.state}, pid={fact.pid}")
            elif fact.restarts is None:
                raise CascadeError("starting services", f"{name}: restart counter unavailable")
        if not pending:
            return facts
        try:
            time.sleep(remaining(deadline, 0.1))
        except TimeoutError as error:
            raise CascadeError("starting services", "; ".join(pending)) from error


def _same_runs(before: Mapping[str, ServiceFact], after: Mapping[str, ServiceFact]) -> None:
    for name, old in before.items():
        new = after.get(name)
        if new is None:
            raise RuntimeError(f"{name}: manager state disappeared")
        if old.kept_stopped:
            if not _stopped(new):
                raise RuntimeError(f"{name}: previously stopped service unexpectedly started ({new.state})")
            continue
        if old.scheduled:
            if new.state != "scheduled":
                raise RuntimeError(f"{name}: schedule is no longer armed ({new.state})")
        elif new.state != "running" or new.pid != old.pid or new.restarts != old.restarts:
            raise RuntimeError(f"{name}: service identity/state/restart counter changed during cascade")
        # Capture's own PID and restart counter are held above. A USB camera or microphone worker that changed means a
        # device failed to come up; an RTSP worker replaced is a network camera's reconnect, which isn't.
        if name == CAPTURE and set(new.children) - new.replaceable != set(old.children) - old.replaceable:
            raise RuntimeError(f"{name}: capture worker identities changed during cascade")


def rtsp_devices(manifest: Path) -> frozenset[str]:
    """The names of the manifest's network (RTSP) cameras. None when it can't be read: every worker is then held to
    the strict check."""
    import tomllib

    try:
        devices = tomllib.loads(manifest.read_text()).get("device", [])
    except OSError, tomllib.TOMLDecodeError:
        return frozenset()
    if not isinstance(devices, list):
        return frozenset()
    return frozenset(
        d["name"] for d in devices if isinstance(d, dict) and d.get("kind") == "rtsp" and isinstance(d.get("name"), str)
    )


def _device_of(command: Sequence[str]) -> str | None:
    """The `--device NAME` a capture worker was started with."""
    for flag, value in itertools.pairwise(command):
        if flag == "--device":
            return value
    return None


def _stopped(fact: ServiceFact) -> bool:
    return (
        fact.state in {"inactive", "inactive/dead", "not loaded", "dead", "exited"}
        and fact.pid is None
        and not fact.children
    )


def _check_required_core(
    facts: Mapping[str, ServiceFact], requested: Sequence[str], start_stopped: bool, capture_required: bool
) -> None:
    if start_stopped:
        return
    for name in (BROKER, CAPTURE):
        if name in facts and name not in requested and _stopped(facts[name]) and (name == BROKER or capture_required):
            raise CascadeError("planning", f"{name} is stopped; start it explicitly or use --cascade")


def _report_failure_state(
    manager: Manager,
    names: Sequence[str],
    before: Mapping[str, ServiceFact],
    deadline: float,
    log: Callable[[str], None],
    diagnostics_restored: bool,
) -> None:
    """Report observed aftermath without converting failed readiness into recovery success."""
    try:
        snapshot_deadline = time.monotonic() + remaining(deadline, 2)
        facts = manager.snapshot(names, snapshot_deadline)
        remaining(deadline)
        stopped = [name for name in names if name in facts and _stopped(facts[name])]
        log("cascade: left stopped: " + (", ".join(stopped) or "none observed"))
        for name in names:
            fact = facts.get(name)
            if name in DIAGNOSTICS:
                if fact is None:
                    status = "state unavailable"
                elif _stopped(fact):
                    status = f"stopped; state={fact.state}"
                elif fact.state == "running" and fact.pid is not None:
                    original = before.get(name)
                    if original is not None and original.pid == fact.pid:
                        disposition = "retained"
                    elif diagnostics_restored:
                        disposition = "restored"
                    else:
                        disposition = "current state; restoration not confirmed"
                    status = f"running ({disposition})"
                else:
                    status = f"not confirmed running; state={fact.state}"
                log(f"cascade: diagnostic {name}: {status}")
            elif fact is None or not _stopped(fact):
                log(f"cascade: after failure {name}: {fact.state if fact is not None else 'state unavailable'}")
    except Exception as error:
        log(f"cascade: post-failure state unavailable: {error}")
    log("cascade: recovery: pack-services restart --cascade")


def start_each(manager: Manager, names: Sequence[str], deadline: float) -> list[str]:
    """Start names, so one that fails doesn't keep the rest down: the failures, one line each."""
    if not names:
        return []
    try:
        manager.start(names, deadline)
        return []
    except Exception:
        pass
    failures = []
    for name in names:
        try:
            manager.start((name,), deadline)
        except Exception as error:
            failures.append(f"{name}: {error}")
    return failures


def run(
    names: Sequence[str],
    *,
    manager: Manager,
    readiness: Readiness,
    health: Health,
    refresh_diagnostics: Sequence[str] = (),
    start_stopped: bool = True,
    requested: Sequence[str] = (),
    capture_required: bool = True,
    timeout: float = DEFAULT_TIMEOUT,
    log: Callable[[str], None] = print,
) -> CascadeReport:
    """Run a complete selected/installed cascade. A failed stage never becomes a successful diagnostic restore."""
    names = tuple(dict.fromkeys(names))
    if BROKER not in names:
        raise CascadeError("planning", "continuum-broker must be selected and installed")
    overall = time.monotonic() + timeout
    deadline = overall - min(RECOVERY_RESERVE, timeout / 4)
    diagnostics = tuple(name for name in names if name in DIAGNOSTICS)
    refresh = tuple(name for name in diagnostics if name in refresh_diagnostics)
    retained = tuple(name for name in diagnostics if name not in refresh)
    readers = tuple(name for name in names if name not in DIAGNOSTICS and name not in {BROKER, CAPTURE})
    stage = "preflight"
    changed = False
    old: dict[str, ServiceFact] = {}
    active: tuple[str, ...] = ()
    try:
        remaining(deadline)
        previous = readiness.previous(deadline)
        old = manager.snapshot(names, deadline)
        _check_required_core(old, requested, start_stopped, capture_required)
        kept = {
            name: replace(fact, kept_stopped=True)
            for name, fact in old.items()
            if not start_stopped and name not in requested and _stopped(fact)
        }
        active = tuple(name for name in names if name not in kept)
        readers = tuple(name for name in readers if name not in kept)
        diagnostics = tuple(name for name in diagnostics if name not in kept)
        refresh = tuple(name for name in refresh if name not in kept)
        for name in kept:
            log(f"cascade: {name}: stopped (kept)")
        for name, fact in old.items():
            if name not in kept and _stopped(fact):
                log(f"cascade: {name}: previously stopped; starting explicitly")
        # The tier worker stops after capture, once it has flushed what capture left in Tier 0: stopped first, the
        # frames between it and capture would strand when the broker's arena goes.
        tier = TIER in readers
        for label, group in (
            ("stop readers", tuple(name for name in readers if name != TIER)),
            ("stop capture", (CAPTURE,) if CAPTURE in active else ()),
            ("drain Tier 0", ()),
            ("stop tier", (TIER,) if tier else ()),
            ("stop broker", (BROKER,)),
        ):
            if label == "drain Tier 0":
                if tier and CAPTURE in active:
                    stage = label
                    try:
                        left = readiness.drain(min(deadline, time.monotonic() + DRAIN_TIMEOUT))
                    except Exception as error:  # best effort: a drain that can't tell never fails the restart
                        log(f"cascade: {stage}: couldn't check ({error}); going on")
                    else:
                        log(f"cascade: {stage}: " + (f"{left} payloads still in Tier 0; going on" if left else "done"))
                continue
            if not group:
                continue
            stage = label
            log(f"cascade: {stage}: {', '.join(group)}")
            changed = True
            manager.stop(group, deadline)
            manager.wait_stopped({name: old[name] for name in group}, deadline)
        stage = "start broker"
        manager.start((BROKER,), deadline)
        core = wait_running_snapshot(manager, (BROKER,), deadline)
        stage = "broker readiness"
        broker = readiness.broker(previous, deadline)
        # The tier worker starts before capture: until it subscribes, the broker owes capture's streams to no archive
        # and may reclaim their first Tier 0 frames under pressure before anything reads them.
        if tier:
            stage = "start tier"
            manager.start((TIER,), deadline)
            wait_running_snapshot(manager, (TIER,), deadline)
        if CAPTURE in active:
            stage = "start capture"
            manager.start((CAPTURE,), deadline)
            core.update(wait_running_snapshot(manager, (CAPTURE,), deadline))
        stage = "capture readiness"
        readiness.capture(broker, deadline)
        # Initialization may add children. Freeze that cohort, then prove every stream afresh without rebasing it.
        if CAPTURE in core:
            current = manager.snapshot((CAPTURE,), deadline)[CAPTURE]
            expected = core[CAPTURE]
            if current.state != "running" or current.pid != expected.pid or current.restarts != expected.restarts:
                raise RuntimeError(f"{CAPTURE}: supervisor restarted while capture was becoming ready")
            core[CAPTURE] = current
            readiness.capture(broker, deadline)
            _same_runs(core, manager.snapshot(tuple(core), deadline))
        stage = "start readers"
        manager.start(tuple(name for name in readers if name != TIER), deadline)
        stage = "diagnostics"
        if refresh:
            snapshot = manager.snapshot(refresh, deadline)
            manager.stop(refresh, deadline)
            manager.wait_stopped(snapshot, deadline)
        manager.start(diagnostics, deadline)
        if retained:
            log(f"cascade: retained diagnostics without restart: {', '.join(retained)}")
        stage = "initial service health"
        baseline = wait_running_snapshot(manager, active, deadline)
        baseline.update(kept)
        _same_runs(kept, manager.snapshot(tuple(kept), deadline))
        # Never rebase an upstream restart into a newly healthy baseline.
        _same_runs(core, baseline)
        stage = "final service health"
        health(manager, names, baseline, deadline)
        stage = "final broker identity"
        readiness.verify(broker, deadline)
        _same_runs(baseline, manager.snapshot(names, deadline))
        running = tuple(name for name in active if not baseline[name].scheduled)
        scheduled = tuple(name for name in active if baseline[name].scheduled)
        log("cascade: ready: " + ", ".join(running))
        if scheduled:
            log("cascade: schedules armed: " + ", ".join(scheduled))
        if kept:
            log("cascade: kept stopped: " + ", ".join(kept))
        return CascadeReport(names, retained)
    except BaseException as error:
        restore_error = None
        if changed:
            # Never leave what this cascade stopped down: a service the application keeps wanted comes back on its
            # own. After a failed stop the core is left as it is, since an old process may still hold the devices.
            # The failure still exits nonzero.
            core_unsafe = stage.startswith("stop ")
            recover = [name for name in active if not (core_unsafe and name in {BROKER, CAPTURE})]
            groups = (
                [n for n in (BROKER, CAPTURE) if n in recover],
                [n for n in recover if n not in DIAGNOSTICS and n not in {BROKER, CAPTURE}],
                [n for n in recover if n in DIAGNOSTICS],
            )
            ordered = [n for group in groups for n in group]
            recovery_deadline = min(overall - min(2, timeout / 8), time.monotonic() + RECOVERY_WAIT)
            failures = [f for group in groups for f in start_each(manager, group, recovery_deadline)]
            if failures:
                restore_error = "; ".join(failures)
                for failure in failures:
                    log(f"cascade: recovery start failed: {failure}")
            elif ordered:
                try:
                    wait_running_snapshot(manager, ordered, recovery_deadline)
                except Exception as recovery:
                    restore_error = str(recovery)
        _report_failure_state(
            manager, names, old, overall, log, bool(changed and diagnostics and restore_error is None)
        )
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        detail = str(error)
        if restore_error:
            detail += f"; diagnostic restoration failed: {restore_error}"
        raise CascadeError(stage, detail) from error


class SDKReadiness:
    """Synchronous lifecycle adapter for the SDK agent's bounded, strict readiness functions."""

    def __init__(self, manifest: Path, socket: Path) -> None:
        from liberational import continuum_ready

        self.socket = socket
        self.manifest = manifest
        self.needs = continuum_ready.capture_needs(manifest)

    def missing(self) -> list[dict[str, str]]:
        """The devices the manifest declares now (enabled ones) that have no live stream at the broker, as
        {"name", "kind": "camera" | "microphone"}: what the application's window names (#183). Raises when the
        broker can't be asked."""
        from liberational import continuum_ready

        needs = continuum_ready.capture_needs(self.manifest)  # read again: a device may have been added since
        gone = asyncio.run(continuum_ready.unregistered(self.socket, needs))
        kinds = {"video": "camera", "audio": "microphone"}
        return [{"name": need.stream.rsplit(".", 1)[0], "kind": kinds[need.kind]} for need in gone]

    def previous(self, deadline: float) -> BrokerIdentity | None:
        from liberational import continuum_ready

        try:
            return asyncio.run(continuum_ready.broker_identity(self.socket, timeout=remaining(deadline, 2)))
        except Exception as error:
            if not continuum_ready.transient(error):
                raise
            return None

    def broker(self, previous: BrokerIdentity | None, deadline: float) -> BrokerIdentity:
        from liberational import continuum_ready

        return asyncio.run(
            continuum_ready.wait_for_broker_restart(
                self.socket,
                previous,
                # The runner already reserved diagnostic recovery time. Cold catalog/purge
                # startup may exceed the standalone wrapper's 60-second wait.
                timeout=remaining(deadline),
            )
        )

    def capture(self, broker: BrokerIdentity, deadline: float) -> None:
        from liberational import continuum_ready

        if not self.needs:
            return
        evidence = asyncio.run(
            continuum_ready.wait_for_capture(self.socket, self.needs, timeout=remaining(deadline, 60))
        )
        if {e.stream for e in evidence} != {need.stream for need in self.needs}:
            raise RuntimeError("capture readiness returned incomplete configured-device evidence")
        if any(e.broker != broker for e in evidence):
            raise RuntimeError("capture evidence belongs to a different broker/arena identity")

    def drain(self, deadline: float) -> int:
        from liberational import continuum_ready

        return asyncio.run(
            continuum_ready.wait_for_tier0_drain(
                self.socket, [need.stream for need in self.needs], timeout=remaining(deadline)
            )
        )

    def verify(self, broker: BrokerIdentity, deadline: float) -> None:
        from liberational import continuum_ready

        current = asyncio.run(continuum_ready.broker_identity(self.socket, timeout=remaining(deadline, 2)))
        if current != broker:
            raise RuntimeError("broker/arena identity changed after capture readiness")


def disk_line(line: str, ts: float) -> str:
    """A cascade line as cascade.log keeps it, in the one log format (logs.py): `<stamp> [cascade] LEVEL message`,
    ERROR for a failure, else INFO."""
    message = line.removeprefix("cascade: ")
    level = "ERROR" if "failed" in message.lower() else "INFO"
    return f"{logs.stamp(ts)} [cascade] {level} {message}"


def _on_disk(line: str) -> None:
    """A cascade line in <paths.logs>/cascade.log, timestamped: the install's terminal is often gone when its failure
    is looked into (keeping only the "after failure" tail lost the stage that failed)."""
    try:
        path = cfg.path("paths.logs") / "cascade.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as out:
            out.write(disk_line(line, time.time()) + "\n")
    except OSError:
        pass  # the log is a convenience: a full or read-only disk never fails a cascade


def file_log(line: str) -> None:
    """A cascade line on stdout, as before (the command's own output: the CLI prints these), and on disk."""
    sys.stdout.write(line + "\n")
    _on_disk(line)
