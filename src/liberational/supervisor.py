"""One supervisor process runs the pack's services as its own children (#155, part of #151).

  pack-supervisor run                   # the application: start everything selected, in order, until told to quit
  pack-supervisor status                # what each service is doing, from the running supervisor
  pack-supervisor start|stop NAME ...   # one service, now
  pack-supervisor restart [NAME ...]    # the staged restart (service_cascade.py) when the broker or capture is in it
  pack-supervisor reload                # follow pack.toml's [services] after a change: start what was turned on,
                                        # stop what was turned off
  pack-supervisor quit                  # drain, then stop, everything, and exit

The operating system's service manager isn't involved: nothing is rendered, installed, or registered. The system runs
while this process does. It lands beside pack-services (which still renders units for the installs that have them,
until #165 moves them over) and the two must never run the same services at once: `run` refuses to start while a
service of the pack is running outside it.

What it does that systemd and launchd did:

- **Starts** each selected service (services.selected()) with the command line, environment, log file, and priority
  the units gave it (services.argv and services.environment), in the cascade's order: the dashboard and the gateway
  and the broker first, then the tier worker, then capture, then the readers. Readiness is checked between the core
  stages when a Readiness is given; a check that fails is logged and the start goes on, since each reader still
  waits for the broker itself (wait_for_broker).
- **Restarts** a service that exits, after its `throttle` doubled for each failure in a row, up to BACKOFF_MAX_S. A
  run that lasted STABLE_S starts the count again. A service with `restart = false` is left as it ended.
- **Stops** with SIGTERM to the service's process group, then SIGKILL after its `stop_timeout`.
- **Drains, then stops, on quit**: the readers, then capture, then the tier worker once it has flushed what capture
  left in Tier 0 (bounded by the cascade's DRAIN_TIMEOUT), then the broker, then the dashboard and the gateway.
  Nothing it started is left running. If the supervisor is killed outright, Linux signals each service too
  (PR_SET_PDEATHSIG); macOS has no such call, so there a service can outlive a supervisor that was killed with
  SIGKILL.
- **Runs the scheduled jobs** on its own clock ("HH:MM" daily, "*:MM" hourly). A job still running at its next time
  is skipped. A daily job whose time passed while the supervisor wasn't running runs once, CATCH_UP_S after the
  start (as the timers' Persistent=true did); an hourly job waits for its next time.
- **Keeps health**: each service's state, pid, restarts, and when its run began, written to
  `<paths.state>/supervisor/status.json` whenever one changes, and every start and exit appended to `events.jsonl`
  beside it. Settings' Services reads the status (services._managed) while a supervisor runs.

Every service stays its own process (AGENTS.md, docs/CONTINUUM.md): a capture crash ends capture, and the supervisor
starts it again.

It is the cascade's Manager (snapshot, stop, start, wait_stopped), so `restart` with the broker or capture in it is
service_cascade.run over this process's own children, with the same stages, readiness checks, and recovery.

Control is a Unix socket, `<paths.state>/supervisor/control.sock`, one JSON object a line each way, open to this
user only.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import json
import os
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Protocol

from liberational import config as cfg
from liberational import logs, service_cascade, services
from liberational.service_cascade import BROKER, CAPTURE, DIAGNOSTICS, TIER, ProcessIdentity, ServiceFact

LOG = logs.get("supervisor")

BACKOFF_MAX_S = 300.0  # the longest wait between restarts of a service that keeps failing
STABLE_S = 60.0  # a run this long was a good one: the next failure waits the plain throttle again
CATCH_UP_S = 120.0  # how soon after a start a daily job that was missed runs
TICK_S = 0.5
REQUEST_TIMEOUT_S = 5.0


def home() -> Path:
    return cfg.path("paths.state") / "supervisor"


def status_path() -> Path:
    return home() / "status.json"


def socket_path() -> Path:
    return home() / "control.sock"


class Process(Protocol):
    """What the supervisor needs of a child: subprocess.Popen, or a stand-in in tests."""

    pid: int

    def poll(self) -> int | None: ...


class Spawn(Protocol):
    def __call__(self, name: str, svc: services.Service) -> Process: ...


def _child_setup(background: bool) -> Callable[[], None]:
    """Runs in the child before exec: on Linux, be signalled if the supervisor dies; lower priority for a service
    declared `background`."""

    def setup() -> None:
        if sys.platform == "linux":
            import ctypes

            ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
        if background:
            os.nice(10)

    return setup


def spawn(name: str, svc: services.Service) -> subprocess.Popen[bytes]:
    """Start a service as the units did: its startup command line (wrappers included), its environment, the owner's root
    as its working directory, and its log file. It gets a process group of its own, so a stop reaches what it started."""
    log = services.log_path(name)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as out:
        return subprocess.Popen(
            services.argv(svc, startup=True, name=name),
            cwd=cfg.workdir(),
            env=services.environment(svc, name),
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=_child_setup(bool(svc.get("background"))),  # noqa: PLW1509 - set before exec, on purpose
        )


def signal_group(pid: int, sig: int) -> None:
    """Signal a service and everything it started; one that is already gone is fine."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, sig)


def next_time(schedule: str, after: dt.datetime) -> dt.datetime:
    """The first time after `after` that a schedule ("HH:MM" daily, "*:MM" hourly) comes round, in local time."""
    hour, minute = schedule.split(":")
    if hour == "*":
        at = after.replace(minute=int(minute), second=0, microsecond=0)
        return at if at > after else at + dt.timedelta(hours=1)
    at = after.replace(hour=int(hour), minute=int(minute), second=0, microsecond=0)
    return at if at > after else at + dt.timedelta(days=1)


def backoff(throttle: float, failures: int) -> float:
    """Seconds to wait before restarting a service that has now failed `failures` times in a row."""
    return min(float(throttle) * 2 ** max(0, failures - 1), BACKOFF_MAX_S)


@dataclass
class Worker:
    name: str
    svc: services.Service
    wanted: bool = False  # the supervisor should have it running (or, for a job, armed)
    proc: Process | None = None
    created_at: float = 0.0  # when the current (or last) run began, wall clock
    began: float = 0.0  # the same, on the supervisor's monotonic clock
    restarts: int = 0  # how often the supervisor restarted it after an exit
    failures: int = 0  # exits in a row without a stable run between
    retry_at: float | None = None  # when the next restart is due
    last_exit: int | None = None
    next_run: dt.datetime | None = None  # a job's next time

    @property
    def scheduled(self) -> bool:
        return bool(self.svc.get("schedule"))

    @property
    def state(self) -> str:
        """In the service managers' words, which Settings' Services and the cascade already read."""
        if self.proc is not None:
            return "running"
        if self.scheduled and self.wanted:
            return "scheduled"
        if self.retry_at is not None:
            return "activating"  # waiting out its backoff, as systemd says of a unit between restarts
        if self.wanted and self.last_exit not in (None, 0) and not self.svc.get("restart", True):
            return "failed"
        return "inactive"


@dataclass
class Supervisor:
    """The services of one machine as child processes. Every method takes the lock; `tick` is the loop's step."""

    spec: Mapping[str, services.Service]
    spawn: Spawn = spawn
    kill: Callable[[int, int], None] = signal_group
    clock: Callable[[], float] = time.monotonic
    now: Callable[[], float] = time.time
    sleep: Callable[[float], None] = time.sleep
    directory: Path | None = None  # where status.json and events.jsonl go; None keeps nothing on disk
    readiness: service_cascade.Readiness | None = None
    workers: dict[str, Worker] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock)
    started_at: float = 0.0
    _written: str = ""

    def __post_init__(self) -> None:
        self.started_at = self.now()
        self.workers = {name: Worker(name, svc) for name, svc in self.spec.items()}

    # ---- one service

    def _start(self, worker: Worker) -> None:
        if worker.proc is not None:
            return
        worker.retry_at = None
        try:
            worker.proc = self.spawn(worker.name, worker.svc)
        except OSError as error:
            LOG.error(f"{worker.name}: could not start: {error}")
            self._failed(worker, None)
            return
        worker.created_at, worker.began = self.now(), self.clock()
        self._event(worker, "start", pid=worker.proc.pid)
        LOG.info(f"{worker.name}: started (pid {worker.proc.pid})")

    def _failed(self, worker: Worker, code: int | None) -> None:
        """A wanted service is down: arrange its restart, unless it is one that runs once."""
        worker.last_exit = code
        if worker.scheduled or not worker.svc.get("restart", True) or not worker.wanted:
            return
        stable = worker.began and self.clock() - worker.began >= STABLE_S
        worker.failures = 1 if stable else worker.failures + 1
        wait = backoff(worker.svc.get("throttle", 10), worker.failures)
        worker.retry_at = self.clock() + wait
        LOG.warning(f"{worker.name}: exited ({code}); restarting in {wait:.0f} s")

    def _reap(self, worker: Worker) -> None:
        if worker.proc is None:
            return
        code = worker.proc.poll()
        if code is None:
            return
        pid, worker.proc = worker.proc.pid, None
        self._event(worker, "exit", pid=pid, code=code)
        if worker.scheduled:
            worker.last_exit = code
            LOG.log(logs.INFO if code == 0 else logs.WARN, f"{worker.name}: job ended ({code})")
            return
        self._failed(worker, code)

    def start(self, names: Sequence[str], deadline: float | None = None) -> None:
        """Have these services running (a job: armed for its next time). Also the cascade's Manager.start."""
        with self.lock:
            for name in names:
                worker = self.workers[name]
                worker.wanted, worker.failures = True, 0
                if worker.scheduled:
                    worker.next_run = worker.next_run or next_time(worker.svc["schedule"], self._local())
                else:
                    self._start(worker)
            self._publish()

    def stop(self, names: Sequence[str], deadline: float | None = None) -> None:
        """Stop these services and keep them stopped: SIGTERM to each one's group, then SIGKILL to what is left after
        its stop_timeout. Also the cascade's Manager.stop, which gives the deadline."""
        with self.lock:
            running = []
            for name in names:
                worker = self.workers[name]
                worker.wanted, worker.retry_at, worker.next_run = False, None, None
                if worker.proc is not None:
                    self.kill(worker.proc.pid, signal.SIGTERM)
                    running.append(worker)
            began = self.clock()
            while running:
                for worker in running:
                    self._reap(worker)
                running = [worker for worker in running if worker.proc is not None]
                late = [w for w in running if self.clock() - began >= float(w.svc.get("stop_timeout", 20))]
                for worker in late:
                    if worker.proc is not None:
                        LOG.warning(f"{worker.name}: still running after its stop timeout; killing it")
                        self.kill(worker.proc.pid, signal.SIGKILL)
                if running:
                    if deadline is not None and self.clock() >= deadline and not late:
                        raise TimeoutError("stop: " + ", ".join(w.name for w in running) + " still running")
                    self.sleep(0.05)
            self._publish()

    # ---- the loop's step

    def tick(self) -> None:
        """Notice exits, start what is due (a restart whose backoff is over, a job whose time has come), and write
        the status if it changed."""
        with self.lock:
            local = self._local()
            for worker in self.workers.values():
                self._reap(worker)
                if not worker.wanted:
                    continue
                if worker.scheduled:
                    if worker.next_run is not None and local >= worker.next_run:
                        worker.next_run = next_time(worker.svc["schedule"], local)
                        if worker.proc is None:
                            self._start(worker)
                        else:
                            LOG.warning(f"{worker.name}: still running at its next time; skipped")
                elif worker.proc is None and worker.retry_at is not None and self.clock() >= worker.retry_at:
                    worker.restarts += 1
                    self._start(worker)
            self._publish()

    # ---- everything, in order

    def _groups(self, names: Sequence[str]) -> list[tuple[str, tuple[str, ...]]]:
        """The start order, as the cascade has it; a stop goes through it backwards."""
        chosen = [name for name in self.workers if name in names]
        readers = tuple(n for n in chosen if n not in DIAGNOSTICS and n not in {BROKER, CAPTURE, TIER})
        return [
            ("diagnostics", tuple(n for n in chosen if n in DIAGNOSTICS)),
            ("broker", tuple(n for n in chosen if n == BROKER)),
            ("tier", tuple(n for n in chosen if n == TIER)),
            ("capture", tuple(n for n in chosen if n == CAPTURE)),
            ("readers", readers),
        ]

    def _confirm(
        self, stage: str, broker: service_cascade.BrokerIdentity | None
    ) -> service_cascade.BrokerIdentity | None:
        """The readiness check after a core start stage. It never stops the start: a reader waits for the broker
        itself. Returns the broker's identity once the broker answers."""
        if self.readiness is None:
            return broker
        deadline = time.monotonic() + 60.0
        try:
            if stage == "broker":
                return self.readiness.broker(None, deadline)
            if stage == "capture" and broker is not None:
                self.readiness.capture(broker, deadline)
        except Exception as error:
            LOG.warning(f"start: {stage} not confirmed ready ({logs.reason(error)}); going on")
        return broker

    def start_all(self, missed: Sequence[str] = ()) -> None:
        """Start every service in order. `missed` names the daily jobs to run once soon (see missed_jobs)."""
        broker = None
        for stage, group in self._groups(tuple(self.workers)):
            if not group:
                continue
            LOG.info(f"start: {stage}: {', '.join(group)}")
            self.start(group)
            broker = self._confirm(stage, broker)
        with self.lock:
            soon = self._local() + dt.timedelta(seconds=CATCH_UP_S)
            for name in missed:
                self.workers[name].next_run = soon

    def shutdown(self) -> None:
        """Drain, then stop: readers, capture, the tier worker once Tier 0 is flushed, the broker, and last the
        dashboard and the gateway."""
        running = tuple(name for name, worker in self.workers.items() if worker.wanted or worker.proc is not None)
        groups = dict(self._groups(running))
        for stage in ("readers", "capture", "tier", "broker", "diagnostics"):
            group = groups[stage]
            if stage == "tier" and group and groups["capture"] and self.readiness is not None:
                try:
                    left = self.readiness.drain(time.monotonic() + service_cascade.DRAIN_TIMEOUT)
                    LOG.info("quit: Tier 0 " + (f"still holds {left} payloads; going on" if left else "drained"))
                except Exception as error:
                    LOG.warning(f"quit: couldn't check Tier 0 ({logs.reason(error)}); going on")
            if group:
                LOG.info(f"quit: stop {stage}: {', '.join(group)}")
                self.stop(group)

    def reload(self, spec: Mapping[str, services.Service]) -> dict[str, list[str]]:
        """Follow a new selection: stop what left it, and start what joined it. A service still selected keeps
        running as it is, even if its declaration changed (restart it to pick that up)."""
        with self.lock:
            gone = [name for name in self.workers if name not in spec]
            self.stop(gone)
            for name in gone:
                del self.workers[name]
            new = [name for name in spec if name not in self.workers]
            for name in new:
                self.workers[name] = Worker(name, spec[name])
            self.spec = spec
            self.start(new)
            return {"started": new, "stopped": gone}

    # ---- the cascade's Manager

    def snapshot(self, names: Sequence[str], deadline: float | None = None) -> dict[str, ServiceFact]:
        with self.lock:
            facts = {}
            for name in names:
                worker = self.workers[name]
                self._reap(worker)
                pid = ProcessIdentity(worker.proc.pid, worker.created_at) if worker.proc is not None else None
                state = "exited" if worker.state in ("inactive", "failed") else worker.state
                facts[name] = ServiceFact(name, state, pid, worker.restarts, scheduled=worker.scheduled)
            return facts

    def wait_stopped(self, before: Mapping[str, ServiceFact], deadline: float) -> None:
        """stop() returns only once each process is gone, so there is nothing left to wait for; say so if not."""
        left = [name for name in before if self.workers[name].proc is not None]
        if left:
            raise RuntimeError("still running after stop: " + ", ".join(left))

    def restart(self, names: Sequence[str]) -> str:
        """Restart services. With the broker or capture among them (or no names: everything) it is the staged
        cascade over this supervisor's children; a reader alone is stopped and started."""
        chosen = tuple(names) or tuple(self.workers)
        unknown = [name for name in chosen if name not in self.workers]
        if unknown:
            raise KeyError(f"not a selected service: {', '.join(unknown)}")
        if not {BROKER, CAPTURE} & set(chosen):
            self.stop(chosen)
            self.start(chosen)
            return "restarted: " + ", ".join(chosen)
        if self.readiness is None:
            raise RuntimeError("no readiness checks here; a core restart needs them")
        from liberational.cascade_health import wait_for_health

        report = service_cascade.run(
            tuple(self.workers),
            manager=self,
            readiness=self.readiness,
            health=wait_for_health,
            refresh_diagnostics=tuple(name for name in chosen if name in DIAGNOSTICS),
            start_stopped=False,
            requested=chosen,
            capture_required=CAPTURE in self.workers,
            log=service_cascade.file_log,
        )
        return "cascade done: " + ", ".join(report.names)

    # ---- health

    def _local(self) -> dt.datetime:
        return dt.datetime.fromtimestamp(self.now())

    def status(self) -> dict[str, Any]:
        """What each service is doing, as status.json keeps it."""
        with self.lock:
            return {
                "pid": os.getpid(),
                "started_at": self.started_at,
                "services": {
                    name: {
                        "state": worker.state,
                        "pid": worker.proc.pid if worker.proc is not None else None,
                        "restarts": worker.restarts,
                        "since": worker.created_at if worker.proc is not None else None,
                        "last_exit": worker.last_exit,
                        "scheduled": worker.scheduled,
                        "next_run": worker.next_run.timestamp() if worker.next_run is not None else None,
                    }
                    for name, worker in self.workers.items()
                },
            }

    def _publish(self) -> None:
        """Write status.json when something changed, whole or not at all."""
        if self.directory is None:
            return
        text = json.dumps(self.status(), separators=(",", ":"))
        if text == self._written:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = self.directory / "status.json.tmp"
        tmp.write_text(text)
        os.replace(tmp, self.directory / "status.json")
        self._written = text

    def _event(self, worker: Worker, event: str, **fields: Any) -> None:
        if self.directory is None:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        line = {"ts": self.now(), "name": worker.name, "event": event, "restarts": worker.restarts, **fields}
        with (self.directory / "events.jsonl").open("a") as out:
            out.write(json.dumps(line, separators=(",", ":")) + "\n")

    def last_runs(self) -> dict[str, float]:
        """When each job last started, from events.jsonl."""
        return last_starts(self.directory) if self.directory is not None else {}


def last_starts(directory: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        lines = (directory / "events.jsonl").read_text().splitlines()
    except OSError:
        return out
    for line in lines:
        with contextlib.suppress(ValueError, KeyError, TypeError):
            event = json.loads(line)
            if event["event"] == "start":
                out[str(event["name"])] = float(event["ts"])
    return out


def missed_jobs(spec: Mapping[str, services.Service], last: Mapping[str, float], now: float) -> list[str]:
    """The daily jobs whose latest time has passed with no run since: each runs once soon after the start. An hourly
    job, and a job that has never run, waits for its next time."""
    at = dt.datetime.fromtimestamp(now)
    missed = []
    for name, svc in spec.items():
        schedule = str(svc.get("schedule") or "")
        if not schedule or schedule.startswith("*") or name not in last:
            continue
        latest = next_time(schedule, at) - dt.timedelta(days=1)
        if last[name] < latest.timestamp():
            missed.append(name)
    return missed


# ---- status for other processes, and control


def read_status(path: Path | None = None) -> dict[str, Any] | None:
    """The running supervisor's status, or None when none runs here (no file, or the process that wrote it is gone)."""
    try:
        status = json.loads((path or status_path()).read_text())
        os.kill(int(status["pid"]), 0)
    except OSError, ValueError, KeyError, TypeError:
        return None
    return status if isinstance(status.get("services"), dict) else None


def managed(status: Mapping[str, Any], now: float) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """A status as services.states() and services.runs() give the service manager's: each service's state, and for a
    daemon how often it was restarted, how long its current run has lasted, and an id that names that run."""
    states, runs = {}, {}
    for name, service in status["services"].items():
        states[name] = str(service["state"])
        if not service.get("scheduled"):
            since = service.get("since")
            runs[name] = {
                "restarts": service.get("restarts"),
                "up_s": max(0.0, now - since) if since else None,
                "run_id": f"{service.get('pid')}-{int(since)}" if since else None,
            }
    return states, runs


def request(message: Mapping[str, Any], path: Path | None = None, timeout: float = REQUEST_TIMEOUT_S) -> dict[str, Any]:
    """Send one request to the running supervisor and return its reply. Raises ConnectionError when none runs."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
            conn.settimeout(timeout)
            conn.connect(str(path or socket_path()))
            conn.sendall(json.dumps(message).encode() + b"\n")
            with conn.makefile("r") as replies:
                return json.loads(replies.readline() or "{}")
    except (OSError, ValueError) as error:
        raise ConnectionError(f"no supervisor answered ({logs.reason(error)})") from error


class _Control(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        try:
            message = json.loads(self.rfile.readline() or b"{}")
            reply = self.server.app.handle(message)  # ty: ignore[unresolved-attribute] - set by App.serve
        except Exception as error:
            reply = {"ok": False, "error": logs.reason(error)}
        with contextlib.suppress(OSError):
            self.wfile.write(json.dumps(reply).encode() + b"\n")


class _Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True


@dataclass
class App:
    """The running application: a Supervisor, its loop, and its control socket."""

    supervisor: Supervisor
    selected: Callable[[], Mapping[str, services.Service]] = services.selected
    quitting: threading.Event = field(default_factory=threading.Event)

    def handle(self, message: Mapping[str, Any]) -> dict[str, Any]:
        """One control request. Names are checked against what is selected before anything is touched."""
        op, names = str(message.get("op") or ""), [str(n) for n in message.get("names") or []]
        unknown = [name for name in names if name not in self.supervisor.workers]
        if unknown and op in ("start", "stop", "restart"):
            return {"ok": False, "error": f"not a selected service: {', '.join(unknown)}"}
        if op == "status":
            return {"ok": True, **self.supervisor.status()}
        if op == "start":
            self.supervisor.start(names)
        elif op == "stop":
            self.supervisor.stop(names)
        elif op == "restart":
            return {"ok": True, "detail": self.supervisor.restart(names)}
        elif op == "reload":
            cfg.load.cache_clear()
            return {"ok": True, **self.supervisor.reload(self.selected())}
        elif op == "quit":
            self.quitting.set()
        else:
            return {"ok": False, "error": "op is status, start, stop, restart, reload, or quit"}
        return {"ok": True}

    @contextlib.contextmanager
    def serve(self, path: Path) -> Generator[None]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
        old = os.umask(0o177)  # the socket is this user's alone
        try:
            server = _Server(str(path), _Control)
        finally:
            os.umask(old)
        server.app = self  # ty: ignore[unresolved-attribute] - read by _Control.handle
        threading.Thread(target=server.serve_forever, daemon=True, name="supervisor-control").start()
        try:
            yield
        finally:
            server.shutdown()
            server.server_close()
            path.unlink(missing_ok=True)

    def run(self, missed: Sequence[str] = ()) -> None:
        """Start everything, keep it running until told to quit, then drain and stop."""
        try:
            self.supervisor.start_all(missed)
            while not self.quitting.wait(TICK_S):
                self.supervisor.tick()
        finally:
            self.supervisor.shutdown()


@contextlib.contextmanager
def single(directory: Path) -> Generator[None]:
    """One supervisor at a time for this state directory."""
    directory.mkdir(parents=True, exist_ok=True)
    lock: IO[str] = (directory / "lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.close()
        raise SystemExit("a supervisor is already running here (pack-supervisor status)") from None
    try:
        yield
    finally:
        lock.close()


def run() -> None:
    """`pack-supervisor run`: the application, in the foreground."""
    logs.setup()
    logs.adopt("supervisor")
    spec = services.selected()
    outside = {name: pids for name, pids in services.processes(fresh=True).items() if name in spec and pids}
    if outside:
        raise SystemExit(
            "these services are already running outside the supervisor (pack-services, or by hand): "
            + ", ".join(sorted(outside))
            + ". Stop them first: the two must not run the same service."
        )
    directory = home()
    with single(directory):
        readiness = service_cascade.SDKReadiness(cfg.devices_file(), cfg.path("paths.state") / "continuum/broker.sock")
        missed = missed_jobs(spec, last_starts(directory), time.time())
        app = App(Supervisor(spec, directory=directory, readiness=readiness))
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: app.quitting.set())
        LOG.info(f"supervising {len(spec)} services" + (f"; catching up: {', '.join(missed)}" if missed else ""))
        with app.serve(socket_path()):
            app.run(missed)
        status_path().unlink(missing_ok=True)
        LOG.info("stopped")


def main() -> None:
    parser = argparse.ArgumentParser(prog="pack-supervisor", description=(__doc__ or "").split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run", help="start everything selected and keep it running until quit")
    sub.add_parser("status", help="what each service is doing")
    for name, text in (("start", "start services"), ("stop", "stop services and keep them stopped")):
        sub.add_parser(name, help=text).add_argument("names", nargs="+")
    sub.add_parser("restart", help="restart services; everything, staged, with no names").add_argument(
        "names", nargs="*"
    )
    sub.add_parser("reload", help="follow [services] in pack.toml after a change")
    sub.add_parser("quit", help="drain, then stop, everything")
    args = parser.parse_args()
    if args.command == "run":
        return run()
    timeout = service_cascade.subprocess_timeout() if args.command == "restart" else 120.0
    try:
        reply = request({"op": args.command, "names": getattr(args, "names", [])}, timeout=timeout)
    except ConnectionError as error:
        raise SystemExit(str(error)) from error
    if not reply.get("ok"):
        raise SystemExit(str(reply.get("error") or "the supervisor refused"))
    if args.command == "status":
        for name, service in reply["services"].items():
            since = f" since {logs.stamp(service['since'])}" if service.get("since") else ""
            print(f"{name:24} {service['state']:11} restarts {service['restarts']}{since}")
    else:
        print(reply.get("detail") or json.dumps({k: v for k, v in reply.items() if k != "ok"}) or "ok")


if __name__ == "__main__":
    main()
