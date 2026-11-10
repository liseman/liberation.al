"""Portable staged service restarts, with strict readiness and process identity checks.

Service-manager success is not readiness. A cascade stops readers before capture, lets the tier worker flush what
capture left in Tier 0, then stops it and the broker, proves old processes gone, then starts broker, capture, and
readers across explicit SDK boundaries. Diagnostics stay available
through core stages. Every manager command and the entire operation have deadlines; errors retain their stage.
"""

import asyncio
import contextlib
import fcntl
import itertools
import os
import subprocess
import sys
import time
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import psutil

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
            # Never leave what this cascade stopped unloaded: launchd doesn't bring an unloaded job back, while
            # a loaded one that fails keeps retrying. After a failed stop the core is left as it is, since an old
            # process may still hold the devices. The failure still exits nonzero.
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


class ServiceManager:
    """Bounded systemd/launchd adapter; it never resets failure counters or signals guessed PIDs."""

    def __init__(self, spec: Mapping[str, dict[str, Any]]) -> None:
        from liberational import services

        self.spec = spec
        self.darwin = services.DARWIN
        self.units = services.unit_dir()

    def command(self, args: Sequence[str], deadline: float, *, check: bool = True) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run(list(args), capture_output=True, text=True, timeout=remaining(deadline, 2))
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f"{' '.join(args)}: {error}") from error
        if check and result.returncode:
            raise RuntimeError(
                f"{' '.join(args)}: {(result.stderr or result.stdout).strip()[-1000:] or f'exit {result.returncode}'}"
            )
        return result

    def unit(self, name: str) -> str:
        return f"pack-{name}.timer" if self.spec[name].get("schedule") else f"pack-{name}.service"

    @staticmethod
    def _identity(pid: int) -> ProcessIdentity | None:
        if pid <= 0:
            return None
        try:
            process = psutil.Process(pid)
            if process.status() == psutil.STATUS_ZOMBIE:
                return None
            return ProcessIdentity(pid, process.create_time())
        except psutil.NoSuchProcess:
            return None
        except psutil.Error as error:
            raise RuntimeError(f"cannot inspect process {pid}: {error}") from error

    def _children(self, identity: ProcessIdentity | None) -> tuple[ProcessIdentity, ...]:
        if identity is None:
            return ()
        try:
            parent = psutil.Process(identity.pid)
            if parent.create_time() != identity.created_at:
                raise RuntimeError(f"process {identity.pid} changed identity during inspection")
            children = (self._identity(p.pid) for p in parent.children(recursive=True))
            return tuple(sorted((p for p in children if p is not None), key=lambda p: (p.pid, p.created_at)))
        except psutil.NoSuchProcess:
            return ()
        except psutil.Error as error:
            raise RuntimeError(f"cannot inspect children of {identity.pid}: {error}") from error

    def _capture_workers(self) -> tuple[tuple[ProcessIdentity, ...], frozenset[ProcessIdentity]]:
        """Capture's worker processes, and which of them run an RTSP camera (their `--device` is a `kind = "rtsp"`
        device in the manifest)."""
        from liberational import services

        if CAPTURE not in self.spec:
            return (), frozenset()
        worker = Path(services.argv(self.spec[CAPTURE])[0]).with_name("continuum-capture-worker").resolve()
        network = rtsp_devices(cfg.devices_file())
        found, rtsp = [], set()
        for process in psutil.process_iter(["exe", "cmdline"]):
            command = process.info.get("cmdline") or []
            executable = process.info.get("exe") or (command[0] if command else None)
            if executable and Path(executable).resolve() == worker:
                identity = self._identity(process.pid)
                if identity is not None:
                    found.append(identity)
                    if _device_of(command) in network:
                        rtsp.add(identity)
        return tuple(sorted(found, key=lambda p: (p.pid, p.created_at))), frozenset(rtsp)

    def snapshot(self, names: Sequence[str], deadline: float) -> dict[str, ServiceFact]:
        if not names:
            return {}
        result: dict[str, ServiceFact] = {}
        if not self.darwin:
            units = list(
                dict.fromkeys(
                    unit
                    for name in names
                    for unit in (
                        (self.unit(name), f"pack-{name}.service")
                        if self.spec[name].get("schedule")
                        else (self.unit(name),)
                    )
                )
            )
            output = self.command(
                [
                    "systemctl",
                    "--user",
                    "show",
                    "--property=Id,LoadState,ActiveState,SubState,MainPID,NRestarts",
                    *units,
                ],
                deadline,
            ).stdout
            records = {}
            for block in output.strip().split("\n\n"):
                props = dict(line.split("=", 1) for line in block.splitlines() if "=" in line)
                if props.get("Id") in units:
                    records[props["Id"]] = props
            for name in names:
                props = records.get(self.unit(name))
                if props is None:
                    continue
                active, sub = props.get("ActiveState", "unknown"), props.get("SubState", "unknown")
                scheduled = bool(self.spec[name].get("schedule"))
                state = (
                    "scheduled"
                    if scheduled and active == "active" and sub in {"waiting", "running"}
                    else "running"
                    if active == "active" and sub == "running"
                    else "failed"
                    if active == "failed"
                    else f"{active}/{sub}"
                )
                if props.get("LoadState") == "not-found":
                    state = "not loaded"
                process_props = records.get(f"pack-{name}.service", props)
                raw_pid, counter = process_props.get("MainPID", ""), process_props.get("NRestarts", "")
                pid = self._identity(int(raw_pid)) if raw_pid.isdigit() else None
                result[name] = ServiceFact(
                    name, state, pid, int(counter) if counter.isdigit() else None, self._children(pid), scheduled
                )
        else:
            for name in names:
                output = self.command(["launchctl", "print", f"gui/{os.getuid()}/pack.{name}"], deadline, check=False)
                if output.returncode:
                    if (
                        output.returncode == 113
                        or "could not find service" in output.stderr.lower()
                        or "not found" in output.stderr.lower()
                    ):
                        result[name] = ServiceFact(
                            name, "not loaded", None, None, scheduled=bool(self.spec[name].get("schedule"))
                        )
                        continue
                    raise RuntimeError(f"{name}: launchctl print failed: {output.stderr.strip()}")
                props = {}
                for line in output.stdout.splitlines():
                    if " = " in line:
                        key, value = line.split(" = ", 1)
                        props.setdefault(key.strip(), value.strip())
                scheduled = bool(self.spec[name].get("schedule"))
                raw_pid, runs = props.get("pid", ""), props.get("runs", "")
                pid = self._identity(int(raw_pid)) if raw_pid.isdigit() else None
                state = "scheduled" if scheduled else props.get("state", "unknown")
                result[name] = ServiceFact(
                    name, state, pid, max(0, int(runs) - 1) if runs.isdigit() else None, self._children(pid), scheduled
                )
        missing = set(names) - result.keys()
        if missing:
            raise RuntimeError("manager returned no state for: " + ", ".join(sorted(missing)))
        if CAPTURE in result:
            fact = result[CAPTURE]
            workers, rtsp = self._capture_workers()
            children = tuple(sorted(set(fact.children) | set(workers), key=lambda p: (p.pid, p.created_at)))
            result[CAPTURE] = replace(fact, children=children, replaceable=rtsp)
        return result

    def stop(self, names: Sequence[str], deadline: float) -> None:
        if not names:
            return
        if self.darwin:
            facts = self.snapshot(names, deadline)
            for name in names:
                if facts[name].state != "not loaded":
                    self.command(["launchctl", "bootout", f"gui/{os.getuid()}/pack.{name}"], deadline)
        else:
            units = [
                unit
                for name in names
                for unit in (
                    (self.unit(name), f"pack-{name}.service") if self.spec[name].get("schedule") else (self.unit(name),)
                )
            ]
            self.command(["systemctl", "--user", "--no-block", "stop", *dict.fromkeys(units)], deadline)

    def start(self, names: Sequence[str], deadline: float) -> None:
        if not names:
            return
        # Every service is tried even when one fails; the failures are raised together after.
        failures = []
        if self.darwin:
            facts = self.snapshot(names, deadline)
            for name in names:
                if facts[name].state in {"running", "scheduled"}:
                    continue
                if facts[name].state == "not loaded":
                    args = ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(self.units / f"pack.{name}.plist")]
                else:
                    args = ["launchctl", "kickstart", f"gui/{os.getuid()}/pack.{name}"]
                try:
                    self.command(args, deadline)
                except RuntimeError as error:
                    failures.append(str(error))
        else:
            units = [self.unit(name) for name in names]
            try:
                self.command(["systemctl", "--user", "--no-block", "start", *units], deadline)
            except RuntimeError:
                # One bad unit fails the whole call; start each so the rest still go.
                for unit in units:
                    try:
                        self.command(["systemctl", "--user", "--no-block", "start", unit], deadline)
                    except RuntimeError as error:
                        failures.append(str(error))
        if failures:
            raise RuntimeError("; ".join(failures))

    def wait_stopped(self, before: Mapping[str, ServiceFact], deadline: float) -> None:
        while True:
            pending = []
            for name, fact in before.items():
                for identity in (*fact.children, fact.pid) if fact.pid is not None else fact.children:
                    if self._identity(identity.pid) == identity:
                        pending.append(f"{name}: old process {identity.pid} is still alive")
            current = self.snapshot(tuple(before), deadline)
            for name, fact in current.items():
                if fact.pid is not None or fact.children or fact.state in {"running", "scheduled"}:
                    pending.append(f"{name}: manager/child processes have not stopped ({fact.state})")
            if not pending:
                return
            try:
                time.sleep(remaining(deadline, 0.1))
            except TimeoutError as error:
                raise RuntimeError("; ".join(pending)) from error

    def prepare_install(self, names: Sequence[str], deadline: float) -> None:
        """Render/enable units without starting them; activation belongs exclusively to the staged runner."""
        from liberational import services

        services.log_path("x").parent.mkdir(parents=True, exist_ok=True)
        self.units.mkdir(parents=True, exist_ok=True)
        for name in names:
            for filename, text in services.render(name, self.spec[name]).items():
                (self.units / filename).write_text(text)
        if not self.darwin:
            self.command(["systemctl", "--user", "daemon-reload"], deadline)
            self.command(
                ["systemctl", "--user", "enable", "--no-reload", *(self.unit(name) for name in names)], deadline
            )


@contextlib.contextmanager
def cascade_lock(path: Path, deadline: float) -> Generator[None]:
    """Serialize controllers across CLI/dashboard processes; the persistent lock inode is never unlinked."""
    remaining(deadline)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0), 0o600
    )
    try:
        os.fchmod(descriptor, 0o600)
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                try:
                    time.sleep(remaining(deadline, 0.05))
                except TimeoutError as error:
                    raise CascadeError("cascade lock", "another service cascade still holds the lock") from error
        yield
    finally:
        os.close(descriptor)


def _run_selected(
    original: Sequence[str],
    *,
    install: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
    anywhere: bool = False,
    start_stopped: bool = False,
) -> CascadeReport:
    """Production boundary shared by explicit restart and every install touching broker/capture."""
    from liberational import services
    from liberational.cascade_health import wait_for_health

    deadline = time.monotonic() + timeout
    services._check_checkout(anywhere)
    chosen = services.selected()
    unknown = set(original) - chosen.keys()
    if unknown:
        raise CascadeError("planning", "not selected: " + ", ".join(sorted(unknown)))
    have = set(services.installed())
    if not install and set(original) - have:
        raise CascadeError("planning", "not installed: " + ", ".join(sorted(set(original) - have)))
    names = tuple(name for name in chosen if name in have or install and name in original)
    if BROKER not in names:
        raise CascadeError("planning", "continuum-broker must be selected and installed")
    ready = SDKReadiness(cfg.devices_file(), cfg.path("paths.state") / "continuum/broker.sock")
    if ready.needs and CAPTURE not in names:
        raise CascadeError("planning", "configured capture devices require selected/installed continuum-capture")
    # Permanent SDK/configuration errors precede all unit writes and manager mutations.
    ready.previous(deadline)
    with cascade_lock(cfg.path("paths.state") / "continuum/service-cascade.lock", deadline):
        # A waiting controller must not apply an obsolete selection or device manifest.
        if services.selected() != chosen or set(services.installed()) != have:
            raise CascadeError("planning", "service selection/installation changed while waiting; retry the cascade")
        fresh = SDKReadiness(cfg.devices_file(), cfg.path("paths.state") / "continuum/broker.sock")
        if fresh.needs != ready.needs:
            raise CascadeError("planning", "configured capture streams changed while waiting; retry the cascade")
        ready = fresh
        manager = ServiceManager(chosen)
        before = manager.snapshot(tuple(name for name in names if name in have), deadline)
        _check_required_core(before, original, start_stopped, bool(ready.needs))
        if install:
            services.preflight(original)
            manager.prepare_install(original, deadline)
        return run(
            names,
            manager=manager,
            readiness=ready,
            health=wait_for_health,
            refresh_diagnostics=tuple(name for name in original if name in DIAGNOSTICS),
            start_stopped=start_stopped,
            requested=original,
            capture_required=bool(ready.needs),
            timeout=remaining(deadline),
            log=file_log,
        )


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


def run_selected(
    original: Sequence[str],
    *,
    install: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
    anywhere: bool = False,
    start_stopped: bool = False,
) -> CascadeReport:
    """Expose named preflight/stage failures to CLI callers instead of silent manager output or empty state."""
    _on_disk(f"cascade: {'install' if install else 'restart'} {' '.join(original) or '(all selected)'}")
    try:
        return _run_selected(original, install=install, timeout=timeout, anywhere=anywhere, start_stopped=start_stopped)
    except CascadeError as error:
        _on_disk(f"cascade: failed at {error}")  # the line the CLI prints, which is the one that names the stage
        raise
    except Exception as error:
        _on_disk(f"cascade: failed at preflight: {error}")
        raise CascadeError("preflight", str(error)) from error
