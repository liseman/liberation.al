import time
from collections import namedtuple

import psutil

"""The admin sidebar's service status and switches: services.snapshot() and the dashboard's /services endpoints."""

import http.client
import json
import logging
import subprocess
import threading
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest

from liberational import config as cfg
from liberational import proc_table
from liberational import services as s
from liberational.dashboard import context, machine_stats, service_control


def proc(pid, cmdline):
    return SimpleNamespace(info={"pid": pid, "cmdline": cmdline})


def as_table(scan):
    """proc_table.table() answered from `scan()`: a list of process_iter-like rows ({"pid", "ppid"?, "name"?,
    "cmdline"}), so each test lists the processes the way it reads best."""

    def table():
        return {
            r.info["pid"]: proc_table.Proc(
                pid=r.info["pid"],
                ppid=r.info.get("ppid"),
                name=r.info.get("name") or "",
                cmdline=tuple(r.info.get("cmdline") or ()),
                created=0.0,
                read_at=0.0,
            )
            for r in scan()
        }

    return table


@pytest.fixture(autouse=True)
def no_saved_scan(monkeypatch):
    monkeypatch.setattr(s, "_scan", {"at": None, "pids": {}, "running": False, "thread": None}, raising=False)
    monkeypatch.setattr(s, "_manager", {"at": None, "states": {}, "runs": {}})


def test_processes_match_pack_commands_by_basename(monkeypatch):
    procs = [
        proc(1, ["/repo/.venv/bin/python3", "/repo/.venv/bin/pack-listen"]),
        proc(2, ["uv", "run", "pack-voice"]),
        proc(3, None),  # cmdline unreadable
        proc(4, ["vim", "notes.md"]),
    ]
    monkeypatch.setattr(proc_table, "table", as_table(create_autospec(lambda: None, return_value=procs)))
    assert s.processes() == {"listen": [1], "voice": [2]}


def test_processes_match_repo_commands_by_their_path_here_and_count_their_children(monkeypatch):
    # The continuum-* daemons are programs in the repo, not pack-* commands, so they had no pids
    broker = str(cfg.workdir() / "continuum/target/release/continuum-broker")
    capture = str(cfg.workdir() / "continuum/target/release/continuum-capture")
    gateway = str(cfg.workdir() / "continuum/target/release/continuum-gateway")
    procs = [
        SimpleNamespace(info={"pid": 1, "ppid": 0, "cmdline": [broker, "--state-dir", "data/state"]}),
        SimpleNamespace(info={"pid": 2, "ppid": 0, "cmdline": [capture, "--manifest", "config/devices.toml"]}),
        SimpleNamespace(info={"pid": 3, "ppid": 2, "cmdline": [capture + "-worker", "--device", "cam0"]}),
        SimpleNamespace(info={"pid": 4, "ppid": 0, "cmdline": ["/usr/bin/nice", gateway]}),
        SimpleNamespace(info={"pid": 5, "ppid": 0, "cmdline": ["/elsewhere/continuum/target/debug/continuum-broker"]}),
        SimpleNamespace(info={"pid": 6, "ppid": 0, "cmdline": ["cargo", "run", "--bin", "continuum-broker"]}),
        SimpleNamespace(info={"pid": 7, "ppid": 5, "cmdline": ["sleep", "1"]}),  # a child of something else
    ]
    monkeypatch.setattr(proc_table, "table", as_table(create_autospec(lambda: None, return_value=procs)))
    assert s.processes() == {"continuum-broker": [1], "continuum-capture": [2, 3], "continuum-gateway": [4]}


def test_processes_within_the_ttl_answer_without_scanning_again(monkeypatch):
    scan = create_autospec(lambda: None, return_value=[proc(1, ["pack-listen"])])
    monkeypatch.setattr(proc_table, "table", as_table(scan))
    monkeypatch.setattr(s, "PROCESS_SCAN_TTL_S", 3600.0)
    assert s.processes() == {"listen": [1]}
    assert s.processes() == {"listen": [1]}
    assert scan.call_count == 1


def test_a_fresh_scan_sees_a_process_started_since_the_last_one(monkeypatch):
    found = [[proc(1, ["pack-listen"])], [proc(1, ["pack-listen"]), proc(2, ["pack-voice"])]]
    monkeypatch.setattr(proc_table, "table", as_table(create_autospec(lambda: None, side_effect=found)))
    monkeypatch.setattr(s, "PROCESS_SCAN_TTL_S", 3600.0)
    assert s.processes() == {"listen": [1]}
    assert s.processes(fresh=True) == {"listen": [1], "voice": [2]}
    assert s.processes() == {"listen": [1], "voice": [2]}  # and the poll gets it from then on


def test_a_request_gets_the_last_answer_while_a_rescan_is_still_running(monkeypatch):
    # After a start, reading every /proc entry took 17 s and GET /services waited for all of it
    rescanning, release = threading.Event(), threading.Event()

    def slow_second_scan():
        if scan.call_count == 1:
            return [proc(1, ["pack-listen"])]
        rescanning.set()
        release.wait(10)
        return [proc(2, ["pack-voice"])]

    scan = create_autospec(lambda: None, side_effect=slow_second_scan)
    monkeypatch.setattr(proc_table, "table", as_table(scan))
    assert s.processes() == {"listen": [1]}
    monkeypatch.setattr(s, "PROCESS_SCAN_TTL_S", 0.0)
    assert s.processes() == {"listen": [1]}  # stale: starts a rescan, answers now
    assert rescanning.wait(5)
    assert s.processes() == {"listen": [1]}  # rescan still running: no second one, no waiting
    release.set()
    s._scan["thread"].join(5)
    monkeypatch.setattr(s, "PROCESS_SCAN_TTL_S", 3600.0)
    assert s.processes() == {"voice": [2]}
    assert scan.call_count == 2


def test_first_services_poll_returns_unknown_while_one_process_scan_runs(monkeypatch):
    scanning, release, replied = threading.Event(), threading.Event(), threading.Event()
    answers = []

    def scan_processes():
        scanning.set()
        release.wait(5)
        return [proc(7, ["pack-listen"])]

    scan = create_autospec(lambda: None, side_effect=scan_processes)
    monkeypatch.setattr(proc_table, "table", as_table(scan))
    monkeypatch.setattr(s, "states", dict)
    monkeypatch.setattr(s, "runs", dict)
    monkeypatch.setattr(s, "last_error", lambda *a, **kw: None)

    def poll():
        answers.append(service_control.services_view(True))
        replied.set()

    request = threading.Thread(target=poll)
    request.start()
    try:
        assert scanning.wait(1)
        assert replied.wait(1), "a first poll must not wait for the cold process scan"
        assert answers == [{"ok": False, "control": True, "error": "Checking service processes…", "services": []}]
        assert service_control.services_view(False)["ok"] is False
        assert scan.call_count == 1
    finally:
        release.set()
        request.join(5)
        if s._scan["thread"] is not None:
            s._scan["thread"].join(5)
    view = service_control.services_view(True)
    assert view["ok"] is True
    assert next(row for row in view["services"] if row["name"] == "listen")["pids"] == [7]


def test_failed_background_process_scan_is_unknown_and_retries_without_thread_exception(monkeypatch):
    scan = create_autospec(lambda: None, side_effect=[OSError("synthetic scan failure"), [proc(8, ["pack-listen"])]])
    monkeypatch.setattr(proc_table, "table", as_table(scan))
    monkeypatch.setattr(s, "states", dict)
    monkeypatch.setattr(s, "runs", dict)
    monkeypatch.setattr(s, "last_error", lambda *a, **kw: None)
    assert service_control.services_view(True)["ok"] is False
    s._scan["thread"].join(5)
    retry = service_control.services_view(True)
    assert retry["ok"] is False and "OSError" in retry["error"] and "retrying" in retry["error"]
    s._scan["thread"].join(5)
    ready = service_control.services_view(True)
    assert ready["ok"] is True
    assert next(row for row in ready["services"] if row["name"] == "listen")["pids"] == [8]
    assert scan.call_count == 2


def test_fresh_action_waits_for_cold_background_scan_then_reads_new_processes(monkeypatch):
    started, release, action_started, done = (threading.Event() for _ in range(4))
    answers = []

    def scan_processes():
        if scan.call_count == 1:
            started.set()
            release.wait(10)
            return [proc(7, ["pack-listen"])]
        return [proc(9, ["pack-listen"])]

    scan = create_autospec(lambda: None, side_effect=scan_processes)
    monkeypatch.setattr(proc_table, "table", as_table(scan))
    monkeypatch.setattr(s, "states", dict)
    monkeypatch.setattr(s, "runs", dict)
    monkeypatch.setattr(s, "last_error", lambda *a, **kw: None)
    assert service_control.services_view(True)["ok"] is False
    assert started.wait(5)

    def action():
        action_started.set()
        answers.append(service_control.services_view(True, fresh=True))
        done.set()

    request = threading.Thread(target=action)
    request.start()
    try:
        assert action_started.wait(5)
        assert not done.is_set()
        release.set()
        assert done.wait(5)
        assert next(row for row in answers[0]["services"] if row["name"] == "listen")["pids"] == [9]
        assert scan.call_count == 2
    finally:
        release.set()
        request.join(5)
        s._scan["thread"].join(5)


def test_fresh_service_view_waits_for_process_evidence_even_before_the_first_poll(monkeypatch):
    scan = create_autospec(lambda: None, return_value=[proc(9, ["pack-listen"])])
    monkeypatch.setattr(proc_table, "table", as_table(scan))
    monkeypatch.setattr(s, "states", dict)
    monkeypatch.setattr(s, "runs", dict)
    monkeypatch.setattr(s, "last_error", lambda *a, **kw: None)
    view = service_control.services_view(True, fresh=True)
    assert view["ok"] is True
    assert next(row for row in view["services"] if row["name"] == "listen")["pids"] == [9]
    assert s._scan["thread"] is None
    scan.assert_called_once_with()


def test_states_ask_about_the_timer_for_scheduled_jobs(tmp_path, monkeypatch):
    for f in ("pack-listen.service", "pack-deep-transcribe.service", "pack-deep-transcribe.timer"):
        (tmp_path / f).write_text("")
    asked = []

    def run(*cmd, check=False):
        asked.append(cmd)
        return subprocess.CompletedProcess(
            cmd,
            0,
            "Id=pack-deep-transcribe.timer\nActiveState=active\n\nId=pack-listen.service\nActiveState=failed\n",
            "",
        )

    monkeypatch.setattr(s, "DARWIN", False)
    monkeypatch.setattr(s, "unit_dir", lambda: tmp_path)
    monkeypatch.setattr(s, "_run", run)
    assert s.states() == {"listen": "failed", "deep-transcribe": "active"}
    assert asked == [
        (
            "systemctl",
            "--user",
            "show",
            "-p",
            "Id",
            "-p",
            "ActiveState",
            "pack-deep-transcribe.timer",
            "pack-listen.service",
        )
    ]


@pytest.mark.parametrize("reply", ["", "Id=pack-listen.service\n", "Id=unrelated.service\nActiveState=active\n"])
def test_states_preserve_unknown_when_the_manager_does_not_report_an_installed_unit(tmp_path, monkeypatch, reply):
    (tmp_path / "pack-listen.service").write_text("")
    monkeypatch.setattr(s, "DARWIN", False)
    monkeypatch.setattr(s, "unit_dir", lambda: tmp_path)
    monkeypatch.setattr(s, "_run", lambda *cmd, check=False: subprocess.CompletedProcess(cmd, 1, reply, "unavailable"))
    assert s.states() == {"listen": "unknown"}


def test_snapshot_lists_this_platform_with_selection_and_state(monkeypatch):
    monkeypatch.setattr(cfg, "platform_key", lambda: "linux-x86_64")
    monkeypatch.setattr(s, "states", lambda: {"listen": "active"})
    monkeypatch.setattr(s, "processes", lambda fresh=False, **kw: {"dashboard": [42]})
    monkeypatch.setattr(s, "runs", lambda: {"listen": {"restarts": 6, "up_s": 30.0}})
    began = {}

    def last_error(name, since=None, *, include_id=False):
        assert include_id  # the page dismisses an error by its id
        began[name] = since
        return {"at": "09:02:00", "text": "boom"} if name == "listen" else None

    monkeypatch.setattr(s, "last_error", last_error)
    monkeypatch.setattr(s.time, "time", lambda: 1000.0)
    rows = {r["name"]: r for r in s.snapshot()}
    assert (rows["listen"]["restarts"], rows["listen"]["up_s"]) == (6, 30.0)
    assert rows["listen"]["last_error"] == {"at": "09:02:00", "text": "boom"}
    assert (rows["whisper-names"]["restarts"], rows["whisper-names"]["last_error"]) == (None, None)
    assert "vlm" not in rows  # Apple Silicon only
    assert (rows["listen"]["installed"], rows["listen"]["state"]) == (True, "active")
    assert (rows["whisper-names"]["installed"], rows["whisper-names"]["state"]) == (False, "not installed")
    assert rows["dashboard"]["pids"] == [42]
    assert not rows["phone-bridge"]["selected"]
    assert (rows["listen"]["default"], rows["phone-bridge"]["default"]) == (True, False)
    assert rows["deep-transcribe"]["scheduled"]
    assert (rows["listen"]["group"], rows["continuum-broker"]["group"]) == ("workers", "core")
    assert (began["listen"], began["whisper-names"]) == (
        None,
        None,
    )  # listen restarted 30 s ago: a crash loop's error shows


@pytest.fixture
def manager(monkeypatch):
    """A service manager that counts how often it's asked, with one service that started 30 s before each answer."""
    asked = {"states": 0, "runs": 0}

    def states():
        asked["states"] += 1
        return {"listen": "active"}

    def runs():
        asked["runs"] += 1
        return {"listen": {"restarts": 0, "up_s": 30.0, "run_id": "systemd:0:1:x"}}

    monkeypatch.setattr(s, "states", states)
    monkeypatch.setattr(s, "runs", runs)
    monkeypatch.setattr(s, "processes", lambda fresh=False, **kw: {})
    monkeypatch.setattr(s, "last_error", lambda *a, **kw: None)
    return asked


def test_polls_close_together_share_one_answer_from_the_service_manager(manager):
    first = {r["name"]: r for r in s.snapshot()}
    second = {r["name"]: r for r in s.snapshot()}
    assert manager == {"states": 1, "runs": 1}
    assert (first["listen"]["state"], second["listen"]["state"]) == ("active", "active")
    assert first["listen"]["up_s"] == 30.0
    assert 30.0 < second["listen"]["up_s"] < 35.0, "a reused answer's run has lasted that much longer"


def test_an_action_s_reply_and_a_poll_after_the_wait_ask_the_service_manager_again(manager, monkeypatch):
    s.snapshot()
    s.snapshot(fresh=True)
    assert manager == {"states": 2, "runs": 2}
    monkeypatch.setattr(s, "MANAGER_TTL_S", 0.0)
    s.snapshot()
    assert manager == {"states": 3, "runs": 3}


SNAPSHOT = [
    {"name": "listen", "selected": True, "installed": False, "state": "not installed", "pids": []},
    {"name": "see-faces", "selected": True, "installed": False, "state": "not installed", "pids": [7]},
    {"name": "voice", "selected": True, "installed": True, "state": "active", "pids": [8], "default": True},
    {"name": "dashboard", "selected": True, "installed": False, "state": "not installed", "pids": [9]},
    {"name": "phone-bridge", "selected": False, "installed": False, "state": "not installed", "pids": []},
]

ON = {"name": "listen", "action": "on"}


class FakePackServices:
    """Stands in for subprocess.run: records the pack-services arguments of each call and replies with one result."""

    returncode, stderr = 0, ""

    def __init__(self):
        self.ran = []
        self.timeouts = []
        self.elapsed = 0

    def __call__(self, cmd, **kw):
        self.ran.append(cmd[cmd.index("liberational.services") + 1 :])
        self.timeouts.append(kw.get("timeout"))
        if kw["timeout"] < self.elapsed:
            raise subprocess.TimeoutExpired(cmd, kw["timeout"])
        return subprocess.CompletedProcess(cmd, self.returncode, "", self.stderr)


@pytest.fixture
def pack_services(monkeypatch):
    fake = FakePackServices()
    monkeypatch.setattr(subprocess, "run", fake)
    return fake


@pytest.fixture
def snapshots(monkeypatch):
    """Stands in for services.snapshot: returns SNAPSHOT and records whether each call asked for fresh pids."""
    asked: list[bool] = []

    def snapshot(fresh=False):
        asked.append(fresh)
        return SNAPSHOT

    monkeypatch.setattr(service_control.services, "snapshot", snapshot)
    return asked


@pytest.fixture
def base(serve_dashboard, monkeypatch, pack_services, snapshots):
    return serve_dashboard()


def call(base, method, body=None, *, path="/services", **headers):
    if body is not None:
        headers.setdefault("Content-Type", "application/json")
    c = http.client.HTTPConnection(base.removeprefix("http://"), timeout=5)
    c.request(method, path, json.dumps(body) if body is not None else None, headers)
    r = c.getresponse()
    out = r.status, json.loads(r.read())
    c.close()
    return out


def test_status_says_whether_this_viewer_may_switch(base):
    code, body = call(base, "GET")
    assert (code, body["control"], body["services"]) == (200, True, SNAPSHOT)
    assert not call(base, "GET", Host="pack.example.ts.net")[1]["control"]
    assert not call(base, "GET", **{"X-Forwarded-For": "100.64.0.9"})[1]["control"]


def test_the_poll_takes_cached_pids_and_every_action_a_fresh_scan(base, snapshots):
    # a service run by hand a moment ago must count before on or Repair, or the install starts it a second time
    call(base, "GET")
    assert snapshots == [False]
    snapshots.clear()
    call(base, "POST", {"name": "voice", "action": "stop"})
    call(base, "POST", {"all": "repair"})
    assert snapshots and all(snapshots)


def test_each_action_runs_its_pack_services_command(base, pack_services):
    assert call(base, "POST", ON)[0] == 200
    assert call(base, "POST", {"name": "voice", "action": "off"})[0] == 200  # turned off: stays off
    assert (
        call(base, "POST", {"name": "voice", "action": "stop"})[0] == 200
    )  # stopped now: back when the computer restarts
    assert call(base, "POST", {"name": "voice", "action": "start"})[0] == 200
    assert call(base, "POST", {"name": "listen", "on": True})[0] == 400  # no action, nothing runs
    assert call(base, "POST", {"name": "listen", "action": "restart"})[0] == 400
    assert pack_services.ran == [["install", "listen"], ["uninstall", "voice"], ["stop", "voice"], ["start", "voice"]]


@pytest.mark.parametrize("name", ["continuum-broker", "continuum-capture"])
@pytest.mark.parametrize("elapsed", [309, 630], ids=["cold-start-over-five-minutes", "late-recovery"])
def test_core_install_allows_the_bounded_cascade_to_finish(base, pack_services, monkeypatch, name, elapsed):
    row = {"name": name, "selected": True, "installed": True, "state": "active", "pids": [42]}
    monkeypatch.setattr(service_control.services, "snapshot", lambda fresh=False: [row])
    pack_services.elapsed = elapsed
    assert call(base, "POST", {"name": name, "action": "on"})[0] == 200
    assert 630 <= pack_services.timeouts[0] <= 900, "cascade plus recovery must remain finitely bounded"


def test_noncore_install_keeps_the_short_service_timeout(base, pack_services):
    assert call(base, "POST", {"name": "listen", "action": "on"})[0] == 200
    assert pack_services.timeouts == [120]


def test_stop_and_start_need_the_service_installed(base, pack_services):
    assert call(base, "POST", {"name": "listen", "action": "stop"})[0] == 409  # enabled, but not installed
    assert call(base, "POST", {"name": "listen", "action": "start"})[0] == 409
    assert pack_services.ran == []


def test_every_action_logs_one_line_with_the_service_and_what_was_done(base, pack_services, monkeypatch):
    records: list[logging.LogRecord] = []
    grab = logging.Handler(logging.INFO)
    grab.emit = records.append  # ty: ignore[invalid-assignment]
    monkeypatch.setattr(context.LOG, "level", logging.INFO)
    context.LOG.addHandler(grab)
    try:
        call(base, "POST", {"name": "voice", "action": "stop"})
        pack_services.returncode, pack_services.stderr = 1, "unit not found"
        call(base, "POST", {"name": "voice", "action": "off"})
        call(base, "POST", {"all": "off"})
    finally:
        context.LOG.removeHandler(grab)
    said = [(r.levelno, r.getMessage()) for r in records if r.getMessage().startswith("services:")]
    assert said == [
        (logging.INFO, "services: stop voice (stopped now; starts again when the computer restarts): done"),
        (logging.INFO, "services: off voice (uninstalled; stays off until turned on): failed: unit not found"),
        (logging.INFO, "services: off voice (every installed service but the dashboard): failed: unit not found"),
    ]


def test_refuses_remote_rebound_and_cross_origin_requests(base, pack_services):
    assert call(base, "POST", ON, **{"X-Forwarded-For": "100.64.0.9"})[0] == 403  # via Tailscale
    assert call(base, "POST", ON, Host="evil.example")[0] == 403  # DNS rebinding
    assert call(base, "POST", ON, Origin="http://evil.example")[0] == 403
    assert call(base, "POST", ON, **{"Content-Type": "text/plain"})[0] == 403
    assert call(base, "POST", {"all": "off"}, **{"X-Forwarded-For": "100.64.0.9"})[0] == 403
    assert pack_services.ran == []


def test_refuses_unselected_unknown_self_and_duplicate(base, pack_services):
    assert call(base, "POST", {"name": "phone-bridge", "action": "on"})[0] == 400
    assert call(base, "POST", {"name": "nope", "action": "on"})[0] == 400
    assert call(base, "POST", {"name": "dashboard", "action": "off"})[0] == 400
    assert call(base, "POST", {"name": "see-faces", "action": "on"})[0] == 409  # already running by hand
    assert call(base, "POST", {"all": "nope"})[0] == 400
    assert pack_services.ran == []


def test_disable_writes_pack_toml_then_uninstalls_so_it_stays_off_after_a_restart(base, pack_services):
    code, body = call(base, "POST", {"name": "voice", "action": "disable"})
    assert (code, body["ok"]) == (200, True), body
    code, body = call(base, "POST", {"name": "listen", "action": "disable"})  # off by default: back to its default
    assert (code, body["ok"]) == (200, True), body
    assert pack_services.ran == [
        ["disable", "voice"],
        ["uninstall", "voice"],
        ["reset", "listen"],
        ["uninstall", "listen"],
    ]


def test_each_action_runs_alone(base, pack_services, monkeypatch):
    # Two Disables at once each read pack.toml, and the second write dropped the first's change.
    held = []
    run = subprocess.run

    def checking(cmd, **kw):
        held.append(service_control.SERVICES_LOCK.locked())
        return run(cmd, **kw)

    monkeypatch.setattr(subprocess, "run", checking)
    call(base, "POST", {"name": "voice", "action": "disable"})
    call(base, "POST", {"all": "repair"})
    assert held and all(held)
    assert not service_control.SERVICES_LOCK.locked()


def test_enable_writes_pack_toml_then_installs(base, pack_services):
    code, body = call(base, "POST", {"name": "phone-bridge", "action": "enable"})
    assert (code, body["ok"]) == (200, True), body
    assert pack_services.ran == [["enable", "phone-bridge"], ["install", "phone-bridge"]]


def test_a_failed_pack_toml_write_changes_no_service(base, pack_services):
    pack_services.returncode, pack_services.stderr = 1, "pack.toml: permission denied"
    code, body = call(base, "POST", {"name": "voice", "action": "disable"})
    assert (code, body["ok"]) == (500, False)
    assert pack_services.ran == [["disable", "voice"]]


def test_enable_and_disable_refuse_what_they_cannot_change(base, pack_services, monkeypatch):
    assert call(base, "POST", {"name": "voice", "action": "enable"})[0] == 409  # already enabled
    assert call(base, "POST", {"name": "phone-bridge", "action": "disable"})[0] == 400  # not enabled
    assert call(base, "POST", {"name": "dashboard", "action": "disable"})[0] == 400  # this page runs on it
    assert call(base, "POST", {"name": "see-faces", "action": "enable"})[0] == 409  # selected already
    whisper = {"name": "whisper", "selected": True, "installed": True, "state": "active", "pids": [], "required": True}
    monkeypatch.setattr(service_control.services, "snapshot", lambda fresh=False: [whisper])
    assert call(base, "POST", {"name": "whisper", "action": "disable"})[0] == 409  # asr.engine needs it
    assert pack_services.ran == []


def test_repair_reinstalls_every_enabled_service_but_the_dashboard(base, pack_services):
    """Services left installed but stopped and disabled showed "All on", since Start all only counted ones not
    installed. Repair removes and reinstalls each enabled service, leaving out the dashboard (this page runs on
    it) and a service running by hand (see-faces here), which a reinstall would start twice."""
    code, body = call(base, "POST", {"all": "repair"})
    assert (code, body["ok"]) == (200, True), body
    assert pack_services.ran == [["uninstall", "listen", "voice"], ["install", "listen", "voice"]]
    assert "see-faces" in body["note"]


def test_core_repair_keeps_old_processes_visible_to_the_cascade(base, pack_services, monkeypatch):
    core = [
        {"name": name, "selected": True, "installed": True, "state": "active", "pids": [42]}
        for name in ("continuum-broker", "continuum-capture")
    ]
    monkeypatch.setattr(service_control.services, "snapshot", lambda fresh=False: [*SNAPSHOT, *core])
    pack_services.elapsed = 630  # A late failure must still have time to finish diagnostic recovery.
    code, body = call(base, "POST", {"all": "repair"})
    assert (code, body["ok"]) == (200, True)
    assert pack_services.ran == [["install", "--cascade", "listen", "voice", "continuum-broker", "continuum-capture"]]
    assert "see-faces" in body["note"]


def test_all_off_removes_every_installed_service_but_the_dashboard(base, pack_services, monkeypatch):
    installed_dashboard = {"name": "dashboard", "selected": True, "installed": True, "state": "active", "pids": []}
    monkeypatch.setattr(
        service_control.services, "snapshot", lambda fresh=False: [*SNAPSHOT[:3], installed_dashboard, SNAPSHOT[4]]
    )
    code, body = call(base, "POST", {"all": "off"})
    assert (code, body["ok"]) == (200, True), body
    assert pack_services.ran == [["uninstall", "voice"]]  # never the dashboard, never "uninstall" with no names


def test_bulk_actions_stop_at_a_failure(base, pack_services):
    pack_services.returncode, pack_services.stderr = 1, "systemctl: boom"
    code, body = call(base, "POST", {"all": "repair"})
    assert (code, body["ok"]) == (500, False)
    assert "boom" in body["error"]
    assert pack_services.ran == [["uninstall", "listen", "voice"]]  # no install after a failed removal


# ---- /stats: each process named for its service, and each Ollama runner for its model ------------------------

BLOB = "sha256-" + "ab" * 32


class FakeProcess:
    def __init__(self, pid, name, cmdline):
        self.pid = pid
        self.info = {"pid": pid, "name": name, "cmdline": cmdline}

    def cpu_percent(self, interval):
        return 12.5

    def memory_info(self):
        return SimpleNamespace(rss=1024)

    def num_threads(self):
        return 3

    def ppid(self):
        return 0

    def create_time(self):
        return 0.0


def test_stats_name_each_service_and_each_ollama_runner_by_its_model(monkeypatch):
    procs = [
        FakeProcess(1, "continuum-broker", ["/repo/continuum/target/release/continuum-broker"]),
        FakeProcess(2, "python", ["/repo/.venv/bin/python", "/repo/.venv/bin/pack-see-faces"]),
        FakeProcess(3, "llama-server", ["/usr/lib/ollama/llama-server", "--model", f"/models/blobs/{BLOB}"]),
        FakeProcess(4, "ollama", ["ollama", "runner", "--model", "/models/blobs/sha256-" + "cd" * 32]),
        FakeProcess(5, "ollama", ["ollama", "serve"]),
        FakeProcess(6, "vim", ["vim"]),
    ]
    monkeypatch.setattr(proc_table, "table", as_table(lambda: procs))
    monkeypatch.setattr(psutil, "Process", {p.pid: p for p in procs}.__getitem__)
    monkeypatch.setattr(
        service_control.services, "processes", lambda fresh=False: {"continuum-broker": [1], "see-faces": [2]}
    )
    asked = []

    def ollama_blob(name):
        asked.append(name)
        return BLOB if name == "qwen3:4b" else None

    monkeypatch.setattr(machine_stats, "ollama_blob", ollama_blob)
    stats = machine_stats.Stats()
    models = [{"name": "qwen3:4b", "digest": "d1"}, {"name": "emb:4b", "digest": "d2"}]
    assert stats.weights(models) == {BLOB: "qwen3:4b"}
    assert stats.weights(models) == {BLOB: "qwen3:4b"}
    assert asked == ["qwen3:4b", "emb:4b"]  # once per model, even when Ollama can't say
    stats.model_of = stats.weights(models)
    monkeypatch.setattr(stats, "ollama_ps", lambda: {"models": []})
    monkeypatch.setattr(stats, "system_status", dict)
    got = {p["pid"]: (p["role"], p["model"]) for p in stats.snapshot()["procs"]}
    assert got == {
        1: ("continuum-broker", None),
        2: ("see-faces", None),
        3: ("ollama runner", "qwen3:4b"),
        4: ("ollama runner", None),  # macOS's runner, whose model Ollama didn't name
        5: ("ollama", None),
    }


def test_a_runner_s_weights_come_from_its_model_argument():
    assert machine_stats.runner_blob(["llama-server", "--model", f"/m/blobs/{BLOB}", "--port", "1"]) == BLOB
    assert machine_stats.runner_blob(["llama-server", "--mmproj", f"/m/blobs/{BLOB}"]) is None
    assert machine_stats.runner_blob(["llama-server"]) is None


def test_stats_asked_again_within_a_second_give_the_same_measurement(monkeypatch):
    # Two pages polling a moment apart: the second read CPU "since the last call" as nearly nothing.
    monkeypatch.setattr(proc_table, "table", dict)
    monkeypatch.setattr(service_control.services, "processes", lambda fresh=False: {})
    clock = [1000.0]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    stats = machine_stats.Stats()
    monkeypatch.setattr(stats, "ollama_ps", lambda: {"models": []})
    monkeypatch.setattr(stats, "system_status", dict)
    # busy (user) and idle seconds since boot: 40% busy, then nothing, then 55%
    stats.cpu_t = times(user=0, idle=0)
    readings = iter([times(user=40, idle=60), times(user=40, idle=60), times(user=95, idle=105)])
    monkeypatch.setattr(psutil, "cpu_times", lambda: next(readings))
    assert stats.snapshot()["cpu"] == 40.0
    clock[0] += 0.5
    assert stats.snapshot()["cpu"] == 40.0  # the same measurement, not a half-second's
    clock[0] += 0.6
    assert stats.snapshot()["cpu"] == 0.0
    clock[0] += 2
    assert stats.snapshot()["cpu"] == 55.0


CpuTimes = namedtuple("CpuTimes", "user nice system idle iowait irq softirq steal guest guest_nice")


def times(user, idle, *, iowait=0.0, guest=0.0, system=0.0):
    """psutil.cpu_times() as Linux gives it: guest time is also counted in user."""
    return CpuTimes(user, 0.0, system, idle, iowait, 0.0, 0.0, 0.0, guest, 0.0)


def test_cpu_busy_counts_guest_once_and_iowait_as_idle():
    assert machine_stats.cpu_busy(times(0, 0), times(30, 50, iowait=10, system=10)) == 40.0  # 40 of 100 s busy
    assert machine_stats.cpu_busy(times(0, 0), times(50, 50, guest=20)) == 50.0  # guest is inside user, not added again
    assert machine_stats.cpu_busy(times(5, 5), times(5, 5)) == 0.0  # no time passed


def test_stats_read_the_cpu_the_same_from_any_request_thread(monkeypatch):
    # psutil 7 keeps cpu_percent's last reading per thread, and each /stats request is a new server thread: it read 0%
    # every time. Two snapshots taken on two fresh threads measure the time between them.
    monkeypatch.setattr(proc_table, "table", dict)
    monkeypatch.setattr(service_control.services, "processes", lambda fresh=False: {})
    clock = [1000.0]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    readings = iter([times(10, 90), times(85, 115)])
    monkeypatch.setattr(psutil, "cpu_times", lambda: next(readings))
    stats = machine_stats.Stats()
    monkeypatch.setattr(stats, "ollama_ps", lambda: {"models": []})
    monkeypatch.setattr(stats, "system_status", dict)
    clock[0] += 2
    got = []
    worker = threading.Thread(target=lambda: got.append(stats.snapshot()["cpu"]))
    worker.start()
    worker.join()
    assert got == [75.0]  # 75 of the 100 s between the two readings were busy


@pytest.mark.parametrize(
    "path,hours", [("/services/history", 24), ("/services/history?hours=24", 24), ("/services/history?hours=168", 168)]
)
def test_service_history_get_keeps_unknown_time_and_uses_the_selected_bounded_window(
    base, tmp_path, monkeypatch, path, hours
):
    monkeypatch.setattr(context, "STATE", tmp_path)
    monkeypatch.setattr(s, "_history_cache", {})
    monkeypatch.setattr(s, "DARWIN", False)
    monkeypatch.setattr(s, "declared", lambda: {"listen": {}})
    monkeypatch.setattr(s.time, "time", lambda: 100000.0)
    read = create_autospec(
        s._read_journal, side_effect=lambda names, start, end: s._journal_view([], names, start, end, "boot")
    )
    monkeypatch.setattr(s, "_read_journal", read)
    code, body = call(base, "GET", path=path)
    assert code == 200 and body["source"] == "journal"
    assert body["services"] == [
        {"name": "listen", "events": [], "spans": [{"start": 100000 - hours * 3600, "end": 100000, "state": "unknown"}]}
    ]
    read.assert_called_once_with(("listen",), 100000 - hours * 3600, 100000)


@pytest.mark.parametrize("hours", ["0", "169", "24.0", "-24", "wrong", ""])
def test_service_history_refuses_invalid_windows_before_reading_any_metadata(base, monkeypatch, hours):
    read = create_autospec(s._read_journal, side_effect=AssertionError("invalid ranges never reach the reader"))
    monkeypatch.setattr(s, "_read_journal", read)
    code, body = call(base, "GET", path=f"/services/history?hours={hours}")
    assert code == 400 and "24 hours or 7 days" in body["error"]


def test_service_history_missing_sample_fallback_is_unavailable_not_healthy(base, tmp_path, monkeypatch):
    monkeypatch.setattr(context, "STATE", tmp_path)
    monkeypatch.setattr(s, "_history_cache", {})
    monkeypatch.setattr(s, "DARWIN", True)
    monkeypatch.setattr(s, "declared", lambda: {"listen": {}})
    code, body = call(base, "GET", path="/services/history")
    assert code == 200 and not body["available"] and body["source"] == "unavailable"
    assert body["services"][0]["spans"][0]["state"] == "unknown"
    assert not (tmp_path / "reports.sqlite3").exists()


def test_service_history_failure_stays_actionable(base, monkeypatch):
    def broken_spec():
        raise OSError("unreadable service definitions")

    monkeypatch.setattr(s, "declared", broken_spec)
    code, body = call(base, "GET", path="/services/history")
    assert code == 503 and body == {"error": "Could not read service history."}
