"""The supervisor (#155): order, backoff, stop, drain, the schedule, health, and control, with stand-in workers."""

import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time

import pytest

from liberational import cascade_health, service_cascade, services, supervisor
from liberational.service_cascade import BROKER, CAPTURE, TIER

SPEC = {
    "listen": {"command": "pack-listen", "throttle": 10},
    "dashboard": {"command": "pack-dashboard"},
    BROKER: {"command": "continuum-broker"},
    CAPTURE: {"command": "continuum-capture"},
    TIER: {"command": "continuum-tier-worker", "stop_timeout": 60},
    "journal": {"command": "pack-journal"},
    "video-window": {"command": "pack-video-window", "schedule": "04:15", "restart": False},
}


class Proc:
    """A worker that runs until the test ends it, or until it is signalled (unless it ignores SIGTERM)."""

    def __init__(self, pid, stubborn=False):
        self.pid, self.code, self.stubborn = pid, None, stubborn

    def poll(self):
        return self.code


class World:
    """The supervisor's surroundings: what it spawned and signalled, a clock the test moves, and Tier 0's drain."""

    def __init__(self):
        self.t, self.wall = 1000.0, dt.datetime(2026, 10, 27, 12, 0).timestamp()
        self.spawned, self.signals, self.procs, self.log = [], [], {}, []
        self.stubborn, self.unspawnable = set(), set()

    def spawn(self, name, svc):
        if name in self.unspawnable:
            raise OSError("no such program")
        proc = Proc(100 + len(self.spawned), name in self.stubborn)
        self.spawned.append(name)
        self.procs[name] = proc
        self.log.append(f"start {name}")
        return proc

    def kill(self, pid, sig):
        name = next(n for n, p in self.procs.items() if p.pid == pid)
        self.signals.append((name, sig))
        if sig == signal.SIGKILL or not self.procs[name].stubborn:
            self.procs[name].code = -sig
            self.log.append(f"stop {name}")

    def sleep(self, seconds):
        self.advance(seconds)

    def advance(self, seconds):
        self.t += seconds
        self.wall += seconds

    # Readiness, as the cascade's and the supervisor's start use it
    def previous(self, deadline):
        return frozenset({("epoch", 1)})

    def broker(self, previous, deadline):
        self.log.append("broker ready")
        return frozenset({("epoch", 2)})

    def capture(self, broker, deadline):
        self.log.append("capture ready")

    def verify(self, broker, deadline):
        self.log.append("verified")

    def drain(self, deadline):
        self.log.append("drained")
        return 0


@pytest.fixture
def world():
    return World()


@pytest.fixture
def sup(world, tmp_path):
    return supervisor.Supervisor(
        SPEC,
        spawn=world.spawn,
        kill=world.kill,
        clock=lambda: world.t,
        now=lambda: world.wall,
        sleep=world.sleep,
        directory=tmp_path,
        readiness=world,
    )


def test_everything_starts_in_the_cascades_order_with_readiness_between(sup, world):
    sup.start_all()
    assert world.log == [
        "start dashboard",
        f"start {BROKER}",
        "broker ready",
        f"start {TIER}",
        f"start {CAPTURE}",
        "capture ready",
        "start listen",
        "start journal",
    ]
    assert "video-window" not in world.spawned  # a job waits for its time
    assert sup.status()["services"]["video-window"]["state"] == "scheduled"


def test_a_readiness_check_that_fails_does_not_keep_the_readers_down(sup, world):
    def refuses(previous, deadline):
        raise RuntimeError("broker socket refused")

    world.broker = refuses
    sup.start_all()
    assert {"listen", "journal", CAPTURE} <= set(world.spawned)


def test_a_worker_that_dies_comes_back_after_its_throttle_and_the_others_keep_running(sup, world):
    sup.start_all()
    others = {name: proc.pid for name, proc in world.procs.items() if name != "listen"}
    world.procs["listen"].code = 1
    sup.tick()
    assert sup.status()["services"]["listen"]["state"] == "activating"
    world.advance(9.9)
    sup.tick()
    assert world.spawned.count("listen") == 1  # not before its throttle (10 s)
    world.advance(0.2)
    sup.tick()
    assert world.spawned.count("listen") == 2
    listen = sup.status()["services"]["listen"]
    assert (listen["state"], listen["restarts"]) == ("running", 1)
    assert {name: proc.pid for name, proc in world.procs.items() if name != "listen"} == others
    assert all(s["restarts"] == 0 for name, s in sup.status()["services"].items() if name != "listen")


def test_the_wait_doubles_for_each_failure_in_a_row_up_to_the_cap_and_a_stable_run_starts_it_again(sup, world):
    sup.start(["listen"])
    waits = []
    for _ in range(7):
        world.procs["listen"].code = 1
        sup.tick()
        waits.append(sup.workers["listen"].retry_at - world.t)
        world.advance(waits[-1])
        sup.tick()
    assert waits == [10, 20, 40, 80, 160, 300, 300]
    world.advance(supervisor.STABLE_S)  # this run held
    world.procs["listen"].code = 1
    sup.tick()
    assert sup.workers["listen"].retry_at - world.t == 10


def test_a_program_that_cannot_be_started_is_retried_with_the_same_backoff(sup, world):
    world.unspawnable.add("journal")
    sup.start(["journal"])
    assert sup.status()["services"]["journal"]["state"] == "activating"
    world.unspawnable.clear()
    world.advance(10)
    sup.tick()
    assert sup.status()["services"]["journal"]["state"] == "running"


def test_a_service_that_runs_once_is_left_as_it_ended(world, tmp_path):
    sup = supervisor.Supervisor(
        {"once": {"command": "pack-once", "restart": False}}, spawn=world.spawn, kill=world.kill, clock=lambda: world.t
    )
    sup.start(["once"])
    world.procs["once"].code = 3
    sup.tick()
    world.advance(1000)
    sup.tick()
    assert world.spawned == ["once"]
    assert sup.status()["services"]["once"]["state"] == "failed"


def test_stop_asks_first_then_kills_what_ignores_it_after_its_own_timeout_and_keeps_it_stopped(sup, world):
    world.stubborn.add(TIER)
    sup.start([TIER, "listen"])
    began = world.t
    sup.stop([TIER, "listen"])
    assert world.signals == [(TIER, signal.SIGTERM), ("listen", signal.SIGTERM), (TIER, signal.SIGKILL)]
    assert world.t - began == pytest.approx(60, abs=0.1)  # the tier worker's stop_timeout, not the default 20
    world.advance(1000)
    sup.tick()
    assert world.spawned == [TIER, "listen"]  # stopped on purpose: not restarted
    assert sup.status()["services"]["listen"]["state"] == "inactive"


def test_quitting_drains_then_stops_and_leaves_nothing_running(sup, world):
    sup.start_all()
    world.log.clear()
    sup.shutdown()
    assert world.log == [
        "stop listen",
        "stop journal",
        f"stop {CAPTURE}",
        "drained",
        f"stop {TIER}",
        f"stop {BROKER}",
        "stop dashboard",
    ]
    assert all(proc.code is not None for proc in world.procs.values())
    assert all(s["pid"] is None for s in sup.status()["services"].values())


def test_a_drain_that_cannot_be_checked_never_holds_up_quitting(sup, world):
    def unknown(deadline):
        raise RuntimeError("broker gone")

    world.drain = unknown
    sup.start_all()
    sup.shutdown()
    assert all(proc.code is not None for proc in world.procs.values())


@pytest.mark.parametrize(
    ("schedule", "after", "expected"),
    [
        pytest.param("04:15", (27, 4, 14), (27, 4, 15), id="later-today"),
        pytest.param("04:15", (27, 4, 15), (28, 4, 15), id="exactly-now-is-tomorrow"),
        pytest.param("04:15", (27, 23, 0), (28, 4, 15), id="tomorrow"),
        pytest.param("*:07", (27, 9, 6), (27, 9, 7), id="this-hour"),
        pytest.param("*:07", (27, 9, 7), (27, 10, 7), id="next-hour"),
        pytest.param("*:07", (27, 23, 30), (28, 0, 7), id="hourly-across-midnight"),
    ],
)
def test_a_schedule_names_its_next_time(schedule, after, expected):
    day, hour, minute = after
    got = supervisor.next_time(schedule, dt.datetime(2026, 10, day, hour, minute))
    assert (got.day, got.hour, got.minute) == expected


def test_a_job_runs_at_its_time_and_is_skipped_while_its_last_run_is_still_going(sup, world):
    sup.start(["video-window"])  # 12:00 now; due 04:15 tomorrow
    world.advance(16 * 3600)
    sup.tick()
    assert world.spawned == []  # 04:00
    world.advance(15 * 60)
    sup.tick()
    assert world.spawned == ["video-window"]
    world.advance(24 * 3600)  # still running a day later
    sup.tick()
    assert world.spawned == ["video-window"]
    world.procs["video-window"].code = 0
    world.advance(24 * 3600)
    sup.tick()
    sup.tick()
    assert world.spawned == ["video-window", "video-window"]


def test_only_a_daily_job_that_ran_before_and_missed_its_time_is_caught_up():
    spec = {
        "daily": {"schedule": "04:15"},
        "ran-today": {"schedule": "04:15"},
        "hourly": {"schedule": "*:07"},
        "never-ran": {"schedule": "01:00"},
        "daemon": {},
    }
    now = dt.datetime(2026, 10, 27, 12, 0).timestamp()
    last = {
        "daily": dt.datetime(2026, 10, 26, 4, 15).timestamp(),
        "ran-today": dt.datetime(2026, 10, 27, 4, 15, 2).timestamp(),
        "hourly": dt.datetime(2026, 10, 26, 9, 7).timestamp(),
    }
    assert supervisor.missed_jobs(spec, last, now) == ["daily"]


def test_a_missed_job_runs_once_soon_after_the_start(sup, world):
    sup.start_all(missed=["video-window"])
    world.advance(supervisor.CATCH_UP_S - 1)
    sup.tick()
    assert "video-window" not in world.spawned
    world.advance(2)
    sup.tick()
    assert world.spawned.count("video-window") == 1


def test_a_core_restart_is_the_cascade_over_the_supervisors_own_children(sup, world, monkeypatch):
    monkeypatch.setattr(cascade_health, "wait_for_health", lambda *a, **k: None)
    monkeypatch.setattr(service_cascade, "file_log", lambda line: None)
    monkeypatch.setattr(service_cascade.time, "sleep", lambda s: None)
    sup.start_all()
    dashboard = world.procs["dashboard"].pid
    world.log.clear()
    assert sup.restart([BROKER]).startswith("cascade done")
    assert world.log == [
        "stop listen",
        "stop journal",
        f"stop {CAPTURE}",
        "drained",
        f"stop {TIER}",
        f"stop {BROKER}",
        f"start {BROKER}",
        "broker ready",
        f"start {TIER}",
        f"start {CAPTURE}",
        "capture ready",
        "capture ready",
        "start listen",
        "start journal",
        "verified",
    ]
    assert world.procs["dashboard"].pid == dashboard  # diagnostics stay up through a core restart


def test_restarting_a_reader_touches_only_it(sup, world):
    sup.start_all()
    world.log.clear()
    sup.restart(["journal"])
    assert world.log == ["stop journal", "start journal"]


def test_health_is_written_when_it_changes_and_reads_as_the_service_managers_did(sup, world, tmp_path):
    sup.start(["listen", "video-window"])
    written = json.loads((tmp_path / "status.json").read_text())
    assert written["pid"] == os.getpid()
    assert written["services"]["listen"] == {
        "state": "running",
        "pid": world.procs["listen"].pid,
        "restarts": 0,
        "since": world.wall,
        "last_exit": None,
        "scheduled": False,
        "next_run": None,
    }
    states, runs = supervisor.managed(written, world.wall + 30)
    assert states["listen"] == "running" and states["video-window"] == "scheduled" and states["journal"] == "inactive"
    assert runs["listen"] == {"restarts": 0, "up_s": 30.0, "run_id": f"{world.procs['listen'].pid}-{int(world.wall)}"}
    assert "video-window" not in runs  # a job's runs are by design, as with the timers
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [(e["name"], e["event"]) for e in events] == [("listen", "start")]


def test_a_status_whose_supervisor_is_gone_is_no_status(tmp_path):
    path = tmp_path / "status.json"
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    path.write_text(json.dumps({"pid": gone.pid, "services": {}}))
    assert supervisor.read_status(path) is None
    path.write_text(json.dumps({"pid": os.getpid(), "services": {"listen": {"state": "running"}}}))
    status = supervisor.read_status(path)
    assert status is not None and status["services"] == {"listen": {"state": "running"}}
    assert supervisor.read_status(tmp_path / "missing.json") is None


def test_settings_services_reads_the_supervisors_record_while_one_runs(monkeypatch):
    status = {"pid": os.getpid(), "services": {"listen": {"state": "activating", "restarts": 4, "since": None}}}
    monkeypatch.setattr(supervisor, "read_status", lambda: status)
    monkeypatch.setattr(services, "states", lambda: pytest.fail("asked the service manager"))
    states, runs = services._managed(fresh=True)
    assert states == {"listen": "activating"}
    assert runs == {"listen": {"restarts": 4, "up_s": None, "run_id": None}}


def test_control_requests_reach_the_supervisor_over_its_socket(sup, world, tmp_path):
    app = supervisor.App(sup, selected=lambda: {**SPEC, "new": {"command": "pack-new"}})
    path = tmp_path / "control.sock"
    with app.serve(path):
        assert supervisor.request({"op": "start", "names": ["listen"]}, path) == {"ok": True}
        assert world.spawned == ["listen"]
        status = supervisor.request({"op": "status"}, path)
        assert status["ok"] and status["services"]["listen"]["state"] == "running"
        refused = supervisor.request({"op": "stop", "names": ["nope"]}, path)
        assert refused == {"ok": False, "error": "not a selected service: nope"}
        assert supervisor.request({"op": "dance"}, path)["ok"] is False
        assert os.stat(path).st_mode & 0o077 == 0  # this user's alone
        assert supervisor.request({"op": "quit"}, path) == {"ok": True}
        assert app.quitting.is_set()
    assert not path.exists()
    with pytest.raises(ConnectionError):
        supervisor.request({"op": "status"}, path)


def test_reload_starts_what_was_turned_on_and_stops_what_was_turned_off(sup, world):
    sup.start(["listen", "journal"])
    new = {name: svc for name, svc in SPEC.items() if name != "journal"} | {"extra": {"command": "pack-extra"}}
    assert sup.reload(new) == {"started": ["extra"], "stopped": ["journal"]}
    assert world.procs["journal"].code is not None
    assert world.procs["listen"].code is None  # still selected: untouched
    assert sup.status()["services"]["extra"]["state"] == "running"
    assert "journal" not in sup.status()["services"]


def test_a_second_supervisor_for_the_same_state_is_refused(tmp_path):
    with supervisor.single(tmp_path), pytest.raises(SystemExit, match="already running"), supervisor.single(tmp_path):
        pass
    with supervisor.single(tmp_path):  # free again once the first has gone
        pass


def test_a_stop_reaches_what_the_service_started():
    """signal_group against real processes: a shell that started a child of its own, as a wrapped service does."""
    shell = subprocess.Popen(
        ["/bin/sh", "-c", "sleep 60 & echo $!; wait"], stdout=subprocess.PIPE, start_new_session=True, text=True
    )
    assert shell.stdout is not None
    child = int(shell.stdout.readline())
    supervisor.signal_group(shell.pid, signal.SIGTERM)
    assert shell.wait(timeout=5) != 0
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        with open(f"/proc/{child}/stat") as stat:  # a zombie until init reaps it still answers kill(0)
            if stat.read().split()[2] == "Z":
                break
    else:
        pytest.fail("the service's own child outlived the stop")
    supervisor.signal_group(shell.pid, signal.SIGTERM)  # already gone: no error


def test_real_processes_are_started_restarted_and_all_stopped(tmp_path):
    """The real spawn and signals, with `sleep` standing in for a service: its log is opened, a kill from outside
    brings it back, and quitting leaves no process."""
    spec = {"napper": {"command": "sleep", "args": ["60"], "throttle": 0, "stop_timeout": 5}}
    sup = supervisor.Supervisor(spec, directory=tmp_path)
    try:
        sup.start(["napper"])
        first = sup.status()["services"]["napper"]["pid"]
        assert services.log_path("napper").exists()
        os.kill(first, signal.SIGKILL)
        deadline = time.monotonic() + 10
        while sup.status()["services"]["napper"]["pid"] in (first, None) and time.monotonic() < deadline:
            sup.tick()
            time.sleep(0.05)
        second = sup.status()["services"]["napper"]
        assert second["pid"] not in (first, None) and second["restarts"] == 1
    finally:
        sup.shutdown()
    assert sup.status()["services"]["napper"]["pid"] is None
    with pytest.raises(ProcessLookupError):
        os.kill(second["pid"], 0)
