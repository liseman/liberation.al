"""Final cascade checks use fake managers, clocks, and status subprocesses only."""

import subprocess
from dataclasses import replace

import pytest

from liberational import cascade_health as health
from liberational.service_cascade import ProcessIdentity, ServiceFact


class Clock:
    now = 100.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        assert 0 <= seconds <= 0.1
        self.now += seconds


class Manager:
    def __init__(self, facts):
        self.facts = dict(facts)
        self.snapshots = []

    def snapshot(self, names, deadline):
        self.snapshots.append((tuple(names), deadline))
        return {name: self.facts[name] for name in names if name in self.facts}

    def stop(self, names, deadline):
        pytest.fail("health check cannot stop services")

    def start(self, names, deadline):
        pytest.fail("health check cannot start services")

    def wait_stopped(self, before, deadline):
        pytest.fail("health check cannot change services")


@pytest.fixture
def setup(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(health, "monotonic", clock.monotonic, raising=False)
    monkeypatch.setattr(health, "sleep", clock.sleep, raising=False)
    facts = {"listen": ServiceFact("listen", "running", ProcessIdentity(101, 10.0), 0)}
    return clock, Manager(facts), facts


def test_changed_daemon_identity_is_not_rebased_as_healthy(setup):
    _, manager, baseline = setup
    manager.facts["listen"] = replace(baseline["listen"], pid=ProcessIdentity(102, 20.0))
    with pytest.raises(RuntimeError, match="listen.*identity"):
        health.wait_for_health(manager, tuple(baseline), baseline, 105.0)


def test_selected_vision_runs_read_only_status_probe(setup, monkeypatch):
    from types import SimpleNamespace

    _, manager, baseline = setup
    baseline["see-actions"] = ServiceFact("see-actions", "running", ProcessIdentity(103, 10), 0)
    manager.facts.update(baseline)
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, "vision ready", "")

    monkeypatch.setattr(health, "subprocess", SimpleNamespace(run=run), raising=False)
    health.wait_for_health(manager, tuple(baseline), baseline, 105.0)
    assert len(calls) == 1, "selected vision must pass the status health probe"
    assert "--selected-only" in calls[0][0]


@pytest.fixture
def status_probe(monkeypatch):
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, "ready", "")

    monkeypatch.setattr(health.subprocess, "run", run)
    return calls


def test_audio_only_services_stay_stable_for_a_second_without_vision_probe(setup, status_probe):
    clock, manager, baseline = setup
    health.wait_for_health(manager, tuple(baseline), baseline, 105)
    assert clock.now >= 101 and len(manager.snapshots) >= 11
    assert all(deadline == 105 for _, deadline in manager.snapshots)
    assert status_probe == []


@pytest.mark.parametrize(
    "change,diagnostic",
    [
        ({"state": "inactive"}, "not running"),
        ({"pid": None}, "identity unavailable"),
        ({"pid": ProcessIdentity(101, 11)}, "identity changed"),
        ({"restarts": 1}, "restart counter changed"),
        ({"restarts": None}, "restart counter unavailable"),
    ],
)
def test_daemon_failures_are_immediate(setup, change, diagnostic):
    clock, manager, baseline = setup
    manager.facts["listen"] = replace(baseline["listen"], **change)
    with pytest.raises(RuntimeError, match=diagnostic):
        health.wait_for_health(manager, tuple(baseline), baseline, 105)
    assert clock.now == 100


def test_missing_service_snapshot_fails_with_its_name(setup):
    _, manager, baseline = setup
    manager.facts.clear()
    with pytest.raises(RuntimeError, match="listen.*snapshot missing"):
        health.wait_for_health(manager, tuple(baseline), baseline, 105)


def test_counter_unavailable_in_initial_baseline_is_not_accepted(setup):
    _, manager, baseline = setup
    baseline["listen"] = replace(baseline["listen"], restarts=None)
    manager.facts.update(baseline)
    with pytest.raises(RuntimeError, match="restart counter unavailable"):
        health.wait_for_health(manager, tuple(baseline), baseline, 105)


def test_capture_worker_churn_fails_even_with_unchanged_supervisor(setup):
    _, manager, baseline = setup
    name = "continuum-capture"
    baseline[name] = ServiceFact(name, "running", ProcessIdentity(104, 10), 0, (ProcessIdentity(204, 10),))
    manager.facts.update(baseline)
    manager.facts[name] = replace(baseline[name], children=(ProcessIdentity(204, 20),))
    with pytest.raises(RuntimeError, match="capture worker identities changed"):
        health.wait_for_health(manager, tuple(baseline), baseline, 105)


def test_armed_job_can_have_no_pid_and_changing_run_counter(setup, status_probe):
    _, manager, baseline = setup
    baseline["maintenance"] = ServiceFact("maintenance", "scheduled", None, None, scheduled=True)
    manager.facts.update(baseline)
    manager.facts["maintenance"] = replace(baseline["maintenance"], pid=ProcessIdentity(300, 20), restarts=5)
    health.wait_for_health(manager, tuple(baseline), baseline, 105)
    assert status_probe == []


def test_disarmed_job_is_not_healthy(setup):
    _, manager, baseline = setup
    baseline["maintenance"] = ServiceFact("maintenance", "scheduled", None, None, scheduled=True)
    manager.facts.update(baseline)
    manager.facts["maintenance"] = replace(baseline["maintenance"], state="inactive")
    with pytest.raises(RuntimeError, match="maintenance.*no longer armed"):
        health.wait_for_health(manager, tuple(baseline), baseline, 105)


def test_short_deadline_cannot_pass_the_stability_window(setup):
    clock, manager, baseline = setup
    with pytest.raises(TimeoutError, match="deadline"):
        health.wait_for_health(manager, tuple(baseline), baseline, 100.25)
    assert clock.now == 100.25


def test_snapshot_that_exceeds_deadline_cannot_report_health(setup, monkeypatch):
    clock, manager, baseline = setup
    snapshot = manager.snapshot

    def late(names, deadline):
        result = snapshot(names, deadline)
        clock.now = deadline + 1
        return result

    monkeypatch.setattr(manager, "snapshot", late)
    with pytest.raises(TimeoutError, match="deadline"):
        health.wait_for_health(manager, tuple(baseline), baseline, 105)


@pytest.mark.parametrize("full", [False, True])
def test_vision_probe_exact_interpreter_deadline_and_selection(setup, status_probe, full):
    import sys

    from liberational import config
    from liberational.continuum_status import VISION

    clock, manager, baseline = setup
    for index, name in enumerate(VISION if full else ("see-actions",)):
        baseline[name] = ServiceFact(name, "running", ProcessIdentity(300 + index, 20), 0)
    manager.facts.update(baseline)
    health.wait_for_health(manager, tuple(baseline), baseline, 105)
    args, kwargs = status_probe[0]
    expected = [sys.executable, "-m", "liberational.continuum_status", "--port", "vision", "--settle", "60"]
    if not full:
        expected += ["--selected-only", "--expected-service", "listen", "--expected-service", "see-actions"]
    assert args == expected
    assert kwargs == {"cwd": config.ROOT, "capture_output": True, "text": True, "timeout": 105 - clock.now}


@pytest.mark.parametrize("failure", ["exit", "timeout"])
def test_vision_probe_failures_include_last_diagnostics(setup, monkeypatch, failure):
    _, manager, baseline = setup
    baseline["see-actions"] = ServiceFact("see-actions", "running", ProcessIdentity(200, 10), 0)
    manager.facts.update(baseline)

    def run(args, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args, kwargs["timeout"], output=b"partial report", stderr=b"camera missing")
        return subprocess.CompletedProcess(args, 1, "partial report", "camera missing")

    monkeypatch.setattr(health.subprocess, "run", run)
    with pytest.raises((RuntimeError, TimeoutError), match="camera missing"):
        health.wait_for_health(manager, tuple(baseline), baseline, 105)


def test_successful_vision_probe_cannot_hide_a_restarted_worker(setup, monkeypatch):
    _, manager, baseline = setup
    baseline["see-actions"] = ServiceFact("see-actions", "running", ProcessIdentity(200, 10), 0)
    manager.facts.update(baseline)

    def run(args, **kwargs):
        manager.facts["see-actions"] = replace(baseline["see-actions"], restarts=1)
        return subprocess.CompletedProcess(args, 0, "ready", "")

    monkeypatch.setattr(health.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="restart counter changed"):
        health.wait_for_health(manager, tuple(baseline), baseline, 105)


def test_baseline_mapping_cannot_be_mutated_into_accepting_restart(setup, monkeypatch):
    _, manager, baseline = setup
    snapshot = manager.snapshot
    calls = 0

    def changed(names, deadline):
        nonlocal calls
        calls += 1
        if calls == 2:
            baseline["listen"] = replace(baseline["listen"], restarts=1)
            manager.facts.update(baseline)
        return snapshot(names, deadline)

    monkeypatch.setattr(manager, "snapshot", changed)
    with pytest.raises(RuntimeError, match="restart counter changed"):
        health.wait_for_health(manager, tuple(baseline), baseline, 105)


def test_success_reports_verified_daemon_worker_and_schedule_facts(setup, capsys):
    _, manager, baseline = setup
    baseline["continuum-capture"] = ServiceFact(
        "continuum-capture", "running", ProcessIdentity(104, 12.5), 2, (ProcessIdentity(204, 13.5),)
    )
    baseline["maintenance"] = ServiceFact("maintenance", "scheduled", None, None, scheduled=True)
    manager.facts.update(baseline)
    health.wait_for_health(manager, tuple(baseline), baseline, 105)
    output = capsys.readouterr().err  # logged, so the restart command shows it beside its own output
    assert "2 running-ready" in output
    assert "0 stopped (kept)" in output
    assert "1 scheduled armed" in output
    for expected in (
        "listen",
        "pid=101",
        "created_at=10",
        "restarts=0",
        "continuum-capture",
        "pid=104",
        "created_at=12.5",
        "restarts=2",
        "pid=204",
        "created_at=13.5",
        "unchanged",
    ):
        assert expected in output
    job_line = next(line for line in output.splitlines() if "maintenance" in line)
    assert "armed" in job_line and "None" not in job_line and "pid=" not in job_line


def test_failure_never_prints_a_success_summary(setup, capsys):
    _, manager, baseline = setup
    manager.facts["listen"] = replace(baseline["listen"], restarts=1)
    with pytest.raises(RuntimeError):
        health.wait_for_health(manager, tuple(baseline), baseline, 105)
    assert "service health" not in capsys.readouterr().err


@pytest.mark.parametrize("state", ["inactive", "inactive/dead", "not loaded", "dead", "exited"])
def test_kept_stopped_leaves_are_reported_without_requiring_pid_or_counter(setup, status_probe, capsys, state):
    _, manager, baseline = setup
    baseline["see-actions"] = ServiceFact("see-actions", state, None, None, kept_stopped=True)
    manager.facts.update(baseline)
    health.wait_for_health(manager, tuple(baseline), baseline, 105)
    assert status_probe == [], "kept-stopped vision must not trigger a live vision check"
    line = next(line for line in capsys.readouterr().err.splitlines() if "see-actions" in line)
    assert "stopped (kept)" in line and f"state={state}" in line
    assert "None" not in line and "pid=" not in line


@pytest.mark.parametrize(
    "change",
    [
        {"state": "running", "pid": ProcessIdentity(202, 20)},
        {"state": "failed"},
        {"children": (ProcessIdentity(303, 20),)},
        {"pid": ProcessIdentity(202, 20)},
    ],
)
def test_kept_stopped_service_cannot_start_fail_or_retain_children(setup, change):
    _, manager, baseline = setup
    baseline["see-actions"] = ServiceFact("see-actions", "inactive/dead", None, None, kept_stopped=True)
    manager.facts.update(baseline)
    manager.facts["see-actions"] = replace(baseline["see-actions"], **change)
    with pytest.raises(RuntimeError, match="see-actions.*kept-stopped"):
        health.wait_for_health(manager, tuple(baseline), baseline, 105)


def test_partial_vision_scope_excludes_kept_stopped_tier(setup, status_probe):
    _, manager, baseline = setup
    baseline["see-actions"] = ServiceFact("see-actions", "running", ProcessIdentity(200, 20), 0)
    baseline["continuum-tier-worker"] = ServiceFact(
        "continuum-tier-worker", "inactive/dead", None, None, kept_stopped=True
    )
    manager.facts.update(baseline)
    health.wait_for_health(manager, tuple(baseline), baseline, 105)
    args, _kwargs = status_probe[0]
    assert "--selected-only" in args
    explicit = [args[index + 1] for index, arg in enumerate(args) if arg == "--expected-service"]
    assert explicit == ["listen", "see-actions"]


def test_summary_separates_running_daemons_kept_stopped_and_armed_jobs(setup, capsys):
    _, manager, baseline = setup
    baseline.update(
        {
            "continuum-capture": ServiceFact("continuum-capture", "running", ProcessIdentity(104, 12.5), 0),
            "voice": ServiceFact("voice", "inactive/dead", None, None, kept_stopped=True),
            "maintenance": ServiceFact("maintenance", "scheduled", None, None, scheduled=True),
            "paused-job": ServiceFact("paused-job", "inactive/dead", None, None, scheduled=True, kept_stopped=True),
        }
    )
    manager.facts.update(baseline)
    health.wait_for_health(manager, tuple(baseline), baseline, 105)
    summary = next(line for line in capsys.readouterr().err.splitlines() if "service health" in line)
    assert "2 running-ready" in summary
    assert "2 stopped (kept)" in summary
    assert "1 scheduled armed" in summary
