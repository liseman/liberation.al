"""Cascades use fake managers/SDK gates and temporary unit files, never real services."""

import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest

from liberational import continuum_ready, services
from liberational import service_cascade as cascade

NAMES = ("continuum-broker", "continuum-capture", "listen", "see-faces", "dashboard", "continuum-gateway")
OLD = frozenset({("old-arena", 1)})
NEW = frozenset({("new-arena", 2)})


class FakeManager:
    def __init__(self, events, names=NAMES):
        self.events = events
        self.names = names
        self.facts = {
            name: cascade.ServiceFact(name, "running", cascade.ProcessIdentity(i + 100, 1.0), 2)
            for i, name in enumerate(names)
        }
        self.facts["continuum-capture"] = replace(
            self.facts["continuum-capture"], children=(cascade.ProcessIdentity(200, 1),)
        )
        self.fail_stop = None
        self.orphan = False

    def snapshot(self, names, deadline):
        return {name: self.facts[name] for name in names}

    def stop(self, names, deadline):
        self.events.append(("stop", tuple(names)))
        if self.fail_stop in names:
            raise RuntimeError(f"{self.fail_stop}: manager refused stop")
        for name in names:
            self.facts[name] = replace(self.facts[name], state="inactive", pid=None, children=())

    def wait_stopped(self, before, deadline):
        self.events.append(("gone", tuple(before)))
        if self.orphan and "continuum-capture" in before:
            raise RuntimeError("continuum-capture: old child 200 remains alive")

    def start(self, names, deadline):
        self.events.append(("start", tuple(names)))
        for name in names:
            if self.facts[name].state != "running":
                pid = cascade.ProcessIdentity(self.names.index(name) + 300, 2)
                children = (cascade.ProcessIdentity(400, 2),) if name == "continuum-capture" else ()
                self.facts[name] = replace(self.facts[name], state="running", pid=pid, children=children)


class FakeReadiness:
    def __init__(self, events):
        self.events = events
        self.fail = None

    def previous(self, deadline):
        self.events.append(("previous",))
        return OLD

    def broker(self, previous, deadline):
        self.events.append(("broker", previous))
        if self.fail == "broker":
            raise RuntimeError("broker request did not become ready")
        return NEW

    def capture(self, broker, deadline):
        self.events.append(("capture", broker))
        if self.fail == "capture":
            raise RuntimeError("cam1.video: no fresh capture frames")

    def drain(self, deadline):
        self.events.append(("drain",))
        if self.fail == "drain":
            raise RuntimeError("broker went away")
        return 0

    def verify(self, broker, deadline):
        self.events.append(("verify", broker))
        if self.fail == "verify":
            raise RuntimeError("broker arena changed after capture readiness")


@pytest.fixture
def lifecycle():
    events = []
    manager, readiness = FakeManager(events), FakeReadiness(events)

    def health(manager, names, baseline, deadline):
        events.append(("health", tuple(names)))

    return manager, readiness, health, events


def test_cascade_orders_stops_and_starts_across_readiness_boundaries(lifecycle):
    manager, readiness, health, events = lifecycle
    report = cascade.run(NAMES, manager=manager, readiness=readiness, health=health, log=lambda _: None)
    assert events == [
        ("previous",),
        ("stop", ("listen", "see-faces")),
        ("gone", ("listen", "see-faces")),
        ("stop", ("continuum-capture",)),
        ("gone", ("continuum-capture",)),
        ("stop", ("continuum-broker",)),
        ("gone", ("continuum-broker",)),
        ("start", ("continuum-broker",)),
        ("broker", OLD),
        ("start", ("continuum-capture",)),
        ("capture", NEW),
        ("capture", NEW),
        ("start", ("listen", "see-faces")),
        ("start", ("dashboard", "continuum-gateway")),
        ("health", NAMES),
        ("verify", NEW),
    ]
    assert report.retained == ("dashboard", "continuum-gateway")


WITH_TIER = (*NAMES[:2], "continuum-tier-worker", *NAMES[2:])


def test_the_tier_stops_after_capture_once_it_has_drained_tier_0(lifecycle):
    # Stopped with the readers, the tier missed what capture wrote until it stopped, and the broker's new arena
    # stranded it.
    _manager, readiness, health, events = lifecycle
    manager = FakeManager(events, WITH_TIER)
    cascade.run(WITH_TIER, manager=manager, readiness=readiness, health=health, log=lambda _: None)
    stops = [event for event in events if event[0] in ("stop", "drain")]
    assert stops == [
        ("stop", ("listen", "see-faces")),
        ("stop", ("continuum-capture",)),
        ("drain",),
        ("stop", ("continuum-tier-worker",)),
        ("stop", ("continuum-broker",)),
    ]


def test_the_tier_starts_before_capture(lifecycle):
    # Started with the readers, the tier subscribed after capture's first frames, and the broker could reclaim them
    # from Tier 0 before anything archived them.
    _manager, readiness, health, events = lifecycle
    manager = FakeManager(events, WITH_TIER)
    cascade.run(WITH_TIER, manager=manager, readiness=readiness, health=health, log=lambda _: None)
    starts = [event for event in events if event[0] == "start"]
    assert starts[:4] == [
        ("start", ("continuum-broker",)),
        ("start", ("continuum-tier-worker",)),
        ("start", ("continuum-capture",)),
        ("start", ("listen", "see-faces")),
    ]


def test_a_drain_that_fails_never_fails_the_restart(lifecycle):
    _manager, readiness, health, events = lifecycle
    manager = FakeManager(events, WITH_TIER)
    readiness.fail = "drain"
    lines = []
    cascade.run(WITH_TIER, manager=manager, readiness=readiness, health=health, log=lines.append)
    assert "cascade: drain Tier 0: couldn't check (broker went away); going on" in lines
    assert all(fact.state == "running" for fact in manager.facts.values())


def test_without_the_tier_in_the_cascade_nothing_waits_for_a_drain(lifecycle):
    manager, readiness, health, events = lifecycle
    cascade.run(NAMES, manager=manager, readiness=readiness, health=health, log=lambda _: None)
    assert ("drain",) not in events


class DrainClient:
    """A broker whose camera stream holds `left[n]` Tier 0 payloads at the n-th look, beside a chunk."""

    def __init__(self, left):
        self.left = list(left)
        self.looks = 0

    async def query(self, stream, start_ns, end_ns, max_items=0):
        n = self.left[min(self.looks, len(self.left) - 1)]
        self.looks += 1
        chunk = {"header": b"", "locator": {"kind": "Chunk", "tier": "Tier1Warm"}}
        held = [{"header": b"", "locator": {"kind": kind}} for kind in (["Shm", "Ring"] * n)[:n]]
        return {"items": [chunk, *held, {"header": b"", "locator": None}]}

    async def close(self):
        pass


@pytest.mark.parametrize(
    ("left", "timeout", "expected"),
    [
        pytest.param([3, 1, 0], 5.0, 0, id="drains"),
        pytest.param([2], 0.1, 2, id="gives-up"),
    ],
)
def test_the_drain_waits_until_no_payload_is_left_only_in_tier_0(monkeypatch, left, timeout, expected):
    import asyncio

    import continuum_py

    client = DrainClient(left)

    async def connect(*args, **kwargs):
        return client

    monkeypatch.setattr(continuum_py.Client, "connect", connect)
    monkeypatch.setattr(continuum_ready, "DRAIN_POLL_S", 0.01)
    monkeypatch.setattr(continuum_ready, "_close", lambda *a, **k: client.close())
    got = asyncio.run(continuum_ready.wait_for_tier0_drain(Path("broker.sock"), ["patio.video"], timeout=timeout))
    assert got == expected


@pytest.mark.parametrize("stage", ["broker", "capture", "verify"])
def test_failed_gate_is_named_and_diagnostic_recovery_does_not_report_success(lifecycle, stage):
    manager, readiness, health, events = lifecycle
    readiness.fail = stage
    with pytest.raises(cascade.CascadeError, match=stage if stage != "verify" else "arena changed"):
        cascade.run(NAMES, manager=manager, readiness=readiness, health=health, log=lambda _: None)
    assert events[-1] == ("start", ("dashboard", "continuum-gateway"))


def test_orphan_capture_child_prevents_new_broker_and_capture_start(lifecycle):
    manager, readiness, health, events = lifecycle
    manager.orphan = True
    with pytest.raises(cascade.CascadeError, match="old child 200"):
        cascade.run(NAMES, manager=manager, readiness=readiness, health=health, log=lambda _: None)
    assert ("start", ("continuum-broker",)) not in events
    assert ("start", ("continuum-capture",)) not in events


@pytest.mark.parametrize("stage", ["broker", "capture"])
def test_a_failed_upstream_gate_still_starts_everything_it_stopped_and_fails(lifecycle, stage):
    # Bob's Mac: a capture failure left the readers unloaded, and launchd never brings those back.
    manager, readiness, health, events = lifecycle
    readiness.fail = stage
    with pytest.raises(cascade.CascadeError, match=stage):
        cascade.run(NAMES, manager=manager, readiness=readiness, health=health, log=lambda _: None)
    assert all(manager.facts[name].state == "running" for name in NAMES)
    started = [event for event in events if event[0] == "start"]
    assert started.index(("start", ("listen", "see-faces"))) < started.index(
        ("start", ("dashboard", "continuum-gateway"))
    )


def test_a_failed_stop_restores_the_readers_but_not_a_core_that_may_still_be_running(lifecycle):
    manager, readiness, health, _events = lifecycle
    manager.orphan = True
    with pytest.raises(cascade.CascadeError, match="old child 200"):
        cascade.run(NAMES, manager=manager, readiness=readiness, health=health, log=lambda _: None)
    assert manager.facts["listen"].state == manager.facts["see-faces"].state == "running"
    assert manager.facts["continuum-capture"].state == "inactive"


def test_one_service_failing_to_start_does_not_keep_the_rest_down(lifecycle):
    manager, readiness, health, _events = lifecycle
    readiness.fail = "capture"
    start = manager.start

    def refuse_listen(names, deadline):
        start([name for name in names if name != "listen"], deadline)
        if "listen" in names:
            raise RuntimeError("bootstrap failed: Input/output error")

    manager.start = refuse_listen
    output = []
    with pytest.raises(cascade.CascadeError, match="no fresh capture frames"):
        cascade.run(NAMES, manager=manager, readiness=readiness, health=health, log=output.append)
    assert manager.facts["see-faces"].state == "running"
    assert "cascade: recovery start failed: listen: bootstrap failed: Input/output error" in output
    assert "cascade: left stopped: listen" in output


def test_worker_identity_change_after_health_is_still_a_failure(lifecycle):
    manager, readiness, _health, _events = lifecycle

    def changed(manager, names, baseline, deadline):
        fact = manager.facts["continuum-capture"]
        manager.facts["continuum-capture"] = replace(fact, children=(cascade.ProcessIdentity(999, 3),))

    with pytest.raises(cascade.CascadeError, match="capture worker identities changed"):
        cascade.run(NAMES, manager=manager, readiness=readiness, health=changed, log=lambda _: None)


def _capture_worker_replaced(manager, *, network):
    """While the cascade watches, capture's worker (FakeManager.start's 400) is replaced by 999, as its supervisor does
    for a network camera whose clock fit failed. `network`: both are RTSP workers (snapshot's `replaceable`)."""
    start, worker, again = manager.start, cascade.ProcessIdentity(400, 2), cascade.ProcessIdentity(999, 3)

    def started(names, deadline):
        start(names, deadline)
        fact = manager.facts["continuum-capture"]
        manager.facts["continuum-capture"] = replace(fact, replaceable=frozenset({worker} if network else ()))

    def replaced(manager, names, baseline, deadline):
        fact = manager.facts["continuum-capture"]
        manager.facts["continuum-capture"] = replace(
            fact, children=(again,), replaceable=frozenset({again} if network else ())
        )

    manager.start = started
    return replaced


def test_an_rtsp_worker_reconnecting_while_the_cascade_watches_isnt_a_failure(lifecycle):
    """Capture's supervisor starts an RTSP worker again whenever its camera's clock fit fails."""
    manager, readiness, _health, _events = lifecycle
    health = _capture_worker_replaced(manager, network=True)
    cascade.run(NAMES, manager=manager, readiness=readiness, health=health, log=lambda _: None)
    assert manager.facts["continuum-capture"].children == (cascade.ProcessIdentity(999, 3),)


def test_a_usb_or_mic_worker_replaced_while_the_cascade_watches_is_still_a_failure(lifecycle):
    manager, readiness, _health, _events = lifecycle
    health = _capture_worker_replaced(manager, network=False)
    with pytest.raises(cascade.CascadeError, match="capture worker identities changed"):
        cascade.run(NAMES, manager=manager, readiness=readiness, health=health, log=lambda _: None)


def test_the_network_cameras_are_the_manifests_rtsp_devices(tmp_path):
    manifest = tmp_path / "devices.toml"
    manifest.write_text(
        '[[device]]\nname = "cam0"\nkind = "camera"\n'
        '[[device]]\nname = "patio"\nkind = "rtsp"\n'
        '[[device]]\nname = "wave3"\nkind = "microphone"\n'
    )
    assert cascade.rtsp_devices(manifest) == {"patio"}
    assert cascade.rtsp_devices(tmp_path / "missing.toml") == frozenset(), "unreadable: every worker held strictly"


def test_failed_stop_never_advances_to_the_next_core_stage(lifecycle):
    manager, readiness, health, events = lifecycle
    manager.fail_stop = "listen"
    with pytest.raises(cascade.CascadeError, match="listen: manager refused stop"):
        cascade.run(NAMES, manager=manager, readiness=readiness, health=health, log=lambda _: None)
    assert ("stop", ("continuum-capture",)) not in events


def test_explicitly_requested_diagnostics_refresh_only_after_capture_and_readers(lifecycle):
    manager, readiness, health, events = lifecycle
    cascade.run(
        NAMES,
        manager=manager,
        readiness=readiness,
        health=health,
        refresh_diagnostics=("dashboard", "continuum-gateway"),
        log=lambda _: None,
    )
    assert events.index(("stop", ("dashboard", "continuum-gateway"))) > events.index(("start", ("listen", "see-faces")))


def test_counter_change_between_start_and_healthy_baseline_cannot_be_rebased(lifecycle):
    manager, readiness, health, _events = lifecycle
    capture = readiness.capture

    def changed(broker, deadline):
        capture(broker, deadline)
        manager.facts["continuum-broker"] = replace(manager.facts["continuum-broker"], restarts=3)

    readiness.capture = changed
    with pytest.raises(cascade.CascadeError, match="continuum-broker: service identity"):
        cascade.run(NAMES, manager=manager, readiness=readiness, health=health, log=lambda _: None)


def test_scheduled_service_can_be_armed_without_a_pid_or_counter(lifecycle):
    manager, _readiness, _health, _events = lifecycle
    manager.facts["listen"] = cascade.ServiceFact("listen", "scheduled", None, None, scheduled=True)
    assert cascade.wait_running_snapshot(manager, ["listen"], time.monotonic() + 1)["listen"].scheduled


def test_missing_daemon_restart_counter_is_not_zero(lifecycle):
    manager, _readiness, _health, _events = lifecycle
    manager.facts["listen"] = replace(manager.facts["listen"], restarts=None)
    with pytest.raises(cascade.CascadeError, match="listen: restart counter unavailable"):
        cascade.wait_running_snapshot(manager, ["listen"], time.monotonic() + 1)


def test_failed_initial_daemon_is_named_without_polling(lifecycle):
    manager, _readiness, _health, _events = lifecycle
    manager.facts["listen"] = replace(manager.facts["listen"], state="failed", pid=None)
    with pytest.raises(cascade.CascadeError, match="listen: manager state failed"):
        cascade.wait_running_snapshot(manager, ["listen"], time.monotonic() + 1)


def test_cli_restart_cascade_uses_the_explicit_flag(monkeypatch):
    restart = create_autospec(services.restart)
    monkeypatch.setattr(services, "restart", restart)
    monkeypatch.setattr(sys, "argv", ["pack-services", "restart", "--cascade"])
    services.main()
    restart.assert_called_once_with([], cascade=True)


def test_cli_restart_without_names_or_cascade_is_rejected(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pack-services", "restart"])
    with pytest.raises(SystemExit) as error:
        services.main()
    assert error.value.code == 2


@pytest.mark.parametrize("wrong_broker", [False, True])
def test_sdk_capture_rejects_incomplete_or_foreign_arena_evidence(tmp_path, monkeypatch, wrong_broker):
    needs = (continuum_ready.CaptureNeed("cam1.video", "video"), continuum_ready.CaptureNeed("mic1.audio", "audio"))
    monkeypatch.setattr(continuum_ready, "capture_needs", lambda path: needs)
    evidence = tuple(
        continuum_ready.CaptureEvidence(need.stream, OLD if wrong_broker else NEW, 100, 2)
        for need in (needs if wrong_broker else needs[:1])
    )

    async def capture(socket, requested, timeout):
        assert requested == needs
        return evidence

    monkeypatch.setattr(continuum_ready, "wait_for_capture", capture)
    ready = cascade.SDKReadiness(tmp_path / "devices.toml", tmp_path / "broker.sock")
    with pytest.raises(RuntimeError, match="different broker|incomplete configured-device"):
        ready.capture(NEW, time.monotonic() + 1)


def test_sdk_final_probe_rejects_a_new_arena_after_health(tmp_path, monkeypatch):
    monkeypatch.setattr(continuum_ready, "capture_needs", lambda path: ())

    async def identity(socket, timeout):
        return OLD

    monkeypatch.setattr(continuum_ready, "broker_identity", identity)
    ready = cascade.SDKReadiness(tmp_path / "devices.toml", tmp_path / "broker.sock")
    with pytest.raises(RuntimeError, match="identity changed after capture readiness"):
        ready.verify(NEW, time.monotonic() + 1)


@pytest.mark.parametrize("startup_seconds", [87, 117])
def test_cold_broker_can_exceed_60_seconds_within_the_240_second_cascade(lifecycle, monkeypatch, startup_seconds):
    manager, readiness, health, events = lifecycle
    clock = [1000.0]
    monkeypatch.setattr(cascade, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    ready = object.__new__(cascade.SDKReadiness)
    ready.socket = Path("fixture.sock")

    async def cold_start(socket, previous, timeout):
        assert previous == OLD
        if timeout < startup_seconds:
            clock[0] += timeout
            raise TimeoutError("cold catalog/purge still loading after broker readiness budget")
        clock[0] += startup_seconds
        return NEW

    monkeypatch.setattr(continuum_ready, "wait_for_broker_restart", cold_start)
    readiness.broker = ready.broker
    report = cascade.run(NAMES, manager=manager, readiness=readiness, health=health, timeout=240, log=lambda _: None)
    assert report.names == NAMES
    assert clock[0] == 1000 + startup_seconds
    assert ("start", ("listen", "see-faces")) in events


def test_default_cascade_allows_244_second_rebuild_and_preserves_capture_and_verify_budget(lifecycle, monkeypatch):
    manager, readiness, _health, _events = lifecycle
    clock = [1000.0]
    monkeypatch.setattr(cascade, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    ready = object.__new__(cascade.SDKReadiness)
    ready.socket = Path("fixture.sock")
    capture_budgets = []
    verification_budgets = []

    async def rebuild(socket, previous, timeout):
        assert previous == OLD
        assert timeout == 590
        clock[0] += 244
        return NEW

    def capture(broker, deadline):
        assert broker == NEW
        capture_budgets.append(cascade.remaining(deadline))
        clock[0] += 30

    def health(manager, names, baseline, deadline):
        assert cascade.remaining(deadline) >= 5
        clock[0] += 5

    def verify(broker, deadline):
        assert broker == NEW
        verification_budgets.append(cascade.remaining(deadline))

    monkeypatch.setattr(continuum_ready, "wait_for_broker_restart", rebuild)
    readiness.broker = ready.broker
    readiness.capture = capture
    readiness.verify = verify
    report = cascade.run(NAMES, manager=manager, readiness=readiness, health=health, log=lambda _: None)
    assert report.names == NAMES
    assert capture_budgets == [346, 316]
    assert verification_budgets == [281]
    assert clock[0] == 1309


def test_explicit_short_cascade_timeout_still_bounds_broker_wait(lifecycle, monkeypatch):
    manager, readiness, health, _events = lifecycle
    clock = [1000.0]
    monkeypatch.setattr(cascade, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    ready = object.__new__(cascade.SDKReadiness)
    ready.socket = Path("fixture.sock")

    async def rebuilding(socket, previous, timeout):
        assert timeout == 22.5
        clock[0] += timeout
        raise TimeoutError("synthetic rebuild exceeded caller deadline")

    monkeypatch.setattr(continuum_ready, "wait_for_broker_restart", rebuilding)
    readiness.broker = ready.broker
    with pytest.raises(cascade.CascadeError, match="broker readiness: synthetic rebuild exceeded caller deadline"):
        cascade.run(NAMES, manager=manager, readiness=readiness, health=health, timeout=30, log=lambda _: None)
    assert clock[0] == 1022.5


def test_broker_wait_still_expires_before_the_cascade_diagnostic_reserve(lifecycle, monkeypatch):
    manager, readiness, health, events = lifecycle
    clock = [1000.0]
    monkeypatch.setattr(cascade, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    ready = object.__new__(cascade.SDKReadiness)
    ready.socket = Path("fixture.sock")

    async def unavailable(socket, previous, timeout):
        clock[0] += timeout
        raise TimeoutError("broker never became ready")

    monkeypatch.setattr(continuum_ready, "wait_for_broker_restart", unavailable)
    readiness.broker = ready.broker
    with pytest.raises(cascade.CascadeError, match="broker readiness: broker never became ready"):
        cascade.run(NAMES, manager=manager, readiness=readiness, health=health, timeout=240, log=lambda _: None)
    assert clock[0] == 1230
    assert manager.facts["listen"].state == "running", "recovery restores the readers after the failure"
    assert events[-1] == ("start", ("dashboard", "continuum-gateway"))


def test_sdk_previous_identity_is_refreshed_after_waiting_for_controller_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(continuum_ready, "capture_needs", lambda path: ())
    identities = iter((OLD, NEW))

    async def identity(socket, timeout):
        return next(identities)

    monkeypatch.setattr(continuum_ready, "broker_identity", identity)
    ready = cascade.SDKReadiness(tmp_path / "devices.toml", tmp_path / "broker.sock")
    assert ready.previous(time.monotonic() + 1) == OLD
    assert ready.previous(time.monotonic() + 1) == NEW


def test_cascade_lines_go_to_stdout_and_the_log(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cascade.cfg, "path", lambda key: tmp_path)
    cascade.file_log("cascade: ready: listen")
    assert capsys.readouterr().out == "cascade: ready: listen\n"
    assert (tmp_path / "cascade.log").read_text().endswith(" [cascade] INFO ready: listen\n")


@pytest.mark.parametrize("start_all", [False, True])
def test_automatic_cascade_preserves_stopped_leaves_but_explicit_cascade_starts_them(lifecycle, start_all):
    manager, readiness, _health, events = lifecycle
    for name in ("listen", "continuum-gateway"):
        manager.facts[name] = replace(manager.facts[name], state="inactive", pid=None)

    def health(manager, names, baseline, deadline):
        assert baseline["listen"].kept_stopped is not start_all
        assert baseline["continuum-gateway"].kept_stopped is not start_all

    cascade.run(
        NAMES,
        manager=manager,
        readiness=readiness,
        health=health,
        start_stopped=start_all,
        requested=("continuum-broker",),
        log=lambda _: None,
    )
    for name in ("listen", "continuum-gateway"):
        assert (manager.facts[name].state == "running") is start_all
        assert any(event[0] == "start" and name in event[1] for event in events) is start_all


def test_failed_leaf_is_recovered_instead_of_preserved_as_stopped(lifecycle):
    manager, readiness, health, _events = lifecycle
    manager.facts["listen"] = replace(manager.facts["listen"], state="failed", pid=None)
    cascade.run(NAMES, manager=manager, readiness=readiness, health=health, start_stopped=False, log=lambda _: None)
    assert manager.facts["listen"].state == "running"


def test_cli_install_cascade_exposes_repair_start_all_policy(monkeypatch):
    install = create_autospec(services.install)
    monkeypatch.setattr(services, "install", install)
    monkeypatch.setattr(sys, "argv", ["pack-services", "install", "--cascade", "continuum-capture"])
    services.main()
    install.assert_called_once_with(["continuum-capture"], cascade=True)


def test_kept_stopped_service_starting_during_health_is_not_accepted(lifecycle):
    manager, readiness, _health, _events = lifecycle
    manager.facts["listen"] = replace(manager.facts["listen"], state="inactive", pid=None)

    def start_stopped(manager, names, baseline, deadline):
        manager.start(("listen",), deadline)

    with pytest.raises(cascade.CascadeError, match="listen: previously stopped service unexpectedly started"):
        cascade.run(
            NAMES,
            manager=manager,
            readiness=readiness,
            health=start_stopped,
            start_stopped=False,
            requested=("continuum-broker",),
            log=lambda _: None,
        )


@pytest.mark.parametrize(
    "stopped,target", [("continuum-capture", "continuum-broker"), ("continuum-broker", "continuum-capture")]
)
def test_stopped_required_core_dependency_refuses_before_stopping_any_service(lifecycle, stopped, target):
    manager, readiness, health, events = lifecycle
    manager.facts[stopped] = replace(manager.facts[stopped], state="inactive", pid=None, children=())
    with pytest.raises(cascade.CascadeError, match=f"{stopped} is stopped; start it explicitly or use --cascade"):
        cascade.run(
            NAMES,
            manager=manager,
            readiness=readiness,
            health=health,
            start_stopped=False,
            requested=(target,),
            log=lambda _: None,
        )
    assert not any(event[0] in {"stop", "start"} for event in events)


def test_zero_devices_keeps_stopped_capture_during_broker_restart(lifecycle):
    manager, readiness, _health, events = lifecycle
    manager.facts["continuum-capture"] = replace(
        manager.facts["continuum-capture"], state="inactive", pid=None, children=()
    )

    def health(manager, names, baseline, deadline):
        assert baseline["continuum-capture"].kept_stopped

    cascade.run(
        NAMES,
        manager=manager,
        readiness=readiness,
        health=health,
        start_stopped=False,
        requested=("continuum-broker",),
        capture_required=False,
        log=lambda _: None,
    )
    assert not any(event[0] in {"stop", "start"} and "continuum-capture" in event[1] for event in events)


def test_replacement_capture_worker_must_prove_fresh_frames_again(lifecycle):
    manager, readiness, health, _events = lifecycle
    proofs = []

    def capture(broker, deadline):
        proofs.append(broker)
        if len(proofs) == 1:
            # A emitted evidence, then restarted while the first all-camera proof waited for B.
            manager.facts["continuum-capture"] = replace(
                manager.facts["continuum-capture"], children=(cascade.ProcessIdentity(999, 3),)
            )
        else:
            raise RuntimeError("cam1.video: replacement worker produced no fresh frames")

    readiness.capture = capture
    with pytest.raises(cascade.CascadeError, match="replacement worker produced no fresh frames"):
        cascade.run(NAMES, manager=manager, readiness=readiness, health=health, log=lambda _: None)
    assert proofs == [NEW, NEW]
    assert manager.facts["listen"].state == "running", "recovery restores the readers after the failure"


def test_capture_cohort_change_during_second_proof_is_rejected(lifecycle):
    manager, readiness, health, _events = lifecycle
    proofs = []

    def capture(broker, deadline):
        proofs.append(broker)
        if len(proofs) == 2:
            manager.facts["continuum-capture"] = replace(
                manager.facts["continuum-capture"], children=(cascade.ProcessIdentity(999, 3),)
            )

    readiness.capture = capture
    with pytest.raises(cascade.CascadeError, match="capture worker identities changed"):
        cascade.run(NAMES, manager=manager, readiness=readiness, health=health, log=lambda _: None)
    assert proofs == [NEW, NEW]
    assert manager.facts["listen"].state == "running", "recovery restores the readers after the failure"


def test_success_output_does_not_call_kept_stopped_service_ready(lifecycle):
    manager, readiness, health, _events = lifecycle
    manager.facts["listen"] = replace(manager.facts["listen"], state="inactive", pid=None)
    output = []
    cascade.run(
        NAMES,
        manager=manager,
        readiness=readiness,
        health=health,
        start_stopped=False,
        requested=("continuum-broker",),
        log=output.append,
    )
    ready = next(line for line in output if line.startswith("cascade: ready:"))
    assert "listen" not in ready
    assert "cascade: kept stopped: listen" in output


def test_failure_output_reports_stopped_services_diagnostics_and_explicit_recovery(lifecycle):
    manager, readiness, health, _events = lifecycle
    readiness.fail = "broker"
    output = []
    with pytest.raises(cascade.CascadeError, match="broker request did not become ready"):
        cascade.run(NAMES, manager=manager, readiness=readiness, health=health, log=output.append)
    assert "cascade: left stopped: none observed" in output
    assert "cascade: diagnostic dashboard: running (retained)" in output
    assert "cascade: diagnostic continuum-gateway: running (retained)" in output
    assert "cascade: recovery: pack-services restart --cascade" in output


def test_failure_reporting_unavailable_preserves_primary_failure(lifecycle, monkeypatch):
    manager, readiness, health, events = lifecycle
    readiness.fail = "broker"
    original = manager.snapshot

    def unavailable(names, deadline):
        if ("broker", OLD) in events:
            raise RuntimeError("manager query unavailable")
        return original(names, deadline)

    monkeypatch.setattr(manager, "snapshot", unavailable)
    output = []
    with pytest.raises(cascade.CascadeError, match="broker request did not become ready"):
        cascade.run(NAMES, manager=manager, readiness=readiness, health=health, log=output.append)
    assert "cascade: post-failure state unavailable: manager query unavailable" in output
    assert "cascade: recovery: pack-services restart --cascade" in output


def test_failure_does_not_restore_a_kept_stopped_diagnostic(lifecycle):
    manager, readiness, health, events = lifecycle
    readiness.fail = "broker"
    manager.facts["continuum-gateway"] = replace(manager.facts["continuum-gateway"], state="inactive", pid=None)
    output = []
    with pytest.raises(cascade.CascadeError, match="broker request did not become ready"):
        cascade.run(
            NAMES,
            manager=manager,
            readiness=readiness,
            health=health,
            start_stopped=False,
            requested=("continuum-broker",),
            log=output.append,
        )
    assert "cascade: diagnostic continuum-gateway: stopped; state=inactive" in output
    assert not any(event[0] == "start" and "continuum-gateway" in event[1] for event in events)


def test_final_output_separates_armed_schedule_from_running_daemons(lifecycle, monkeypatch):
    manager, readiness, health, _events = lifecycle
    manager.facts["listen"] = replace(manager.facts["listen"], state="scheduled", pid=None, scheduled=True)
    original = manager.start

    def start(names, deadline):
        original(names, deadline)
        if "listen" in names:
            manager.facts["listen"] = replace(manager.facts["listen"], state="scheduled", pid=None)

    monkeypatch.setattr(manager, "start", start)
    output = []
    cascade.run(NAMES, manager=manager, readiness=readiness, health=health, log=output.append)
    assert "listen" not in next(line for line in output if line.startswith("cascade: ready:"))
    assert "cascade: schedules armed: listen" in output
