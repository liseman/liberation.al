"""pack-continuum-status: the service states it reports, the broker it checks (against a real one where continuum_py
and a broker binary are available, as in continuum-test), and its exit code."""

import argparse
import os
import pathlib
import socket as py_socket
import subprocess
import time

import pytest

from liberational import continuum_status as cs


def args(**over: object) -> argparse.Namespace:
    return argparse.Namespace(
        **{"window": 60, "stale": 10.0, "lag": 30.0, "full": 0.9, "port": None, "through": None, **over}
    )


def test_a_service_that_isnt_installed_is_noted_and_one_that_isnt_running_fails() -> None:
    report = cs.Report()
    running = cs.check_services(
        report,
        installed=["continuum-broker", "continuum-tier-worker"],
        state={"continuum-broker": "active", "continuum-tier-worker": "failed"},
    )
    assert running == {"continuum-broker"}
    assert report.problems == ["continuum-tier-worker: failed"]
    assert "        continuum-gateway: not installed" in report.lines


def test_at_a_stage_a_service_it_needs_must_be_installed() -> None:
    report = cs.Report()
    cs.check_services(report, installed=["continuum-broker"], state={"continuum-broker": "active"}, expected=cs.AUDIO)
    assert "listen: not installed, and this stage needs it" in report.problems
    # The audio stage doesn't need the gateway or the vision workers: they're only noted.
    assert not any("gateway" in p or "see-" in p for p in report.problems)


def test_the_manifest_names_a_stream_per_device(tmp_path: pathlib.Path) -> None:
    manifest = tmp_path / "devices.toml"
    manifest.write_text(
        '[[device]]\nname = "wave3"\nkind = "microphone"\n\n[[device]]\nname = "facecam"\nkind = "camera"\n'
    )
    assert cs.manifest_streams(manifest) == {"wave3.audio", "facecam.video"}


def test_an_xvf3800_array_names_its_audio_stream(tmp_path: pathlib.Path) -> None:
    # Its audio is a stream like a microphone's; leaving the kind out silently skipped the check.
    manifest = tmp_path / "devices.toml"
    manifest.write_text('[[device]]\nname = "array1"\nkind = "xvf3800"\n')
    assert cs.manifest_streams(manifest) == {"array1.audio"}


def test_the_arena_name_is_the_second_line_beside_the_socket(tmp_path: pathlib.Path) -> None:
    socket = tmp_path / "broker.sock"
    assert cs.arena_name(socket) is None
    pathlib.Path(f"{socket}.arena").write_text("1234\n/cnt.hot.7\n")
    assert cs.arena_name(socket) == "/cnt.hot.7"


def test_problems_exit_1_and_a_clean_report_exits_0(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    bad = cs.Report()
    bad.bad("continuum-broker: failed")
    monkeypatch.setattr(cs, "run", lambda _args: bad)
    with pytest.raises(SystemExit) as e:
        cs.main([])
    assert e.value.code == 1
    assert "1 problem(s)" in capsys.readouterr().out
    monkeypatch.setattr(cs, "run", lambda _args: cs.Report())
    cs.main([])
    assert "all good" in capsys.readouterr().out


def test_a_broker_that_isnt_there_fails(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("continuum_py", reason="continuum_py is built by uv sync or in continuum/.venv")
    import asyncio

    report = cs.Report()
    asyncio.run(cs.check_broker(report, tmp_path / "nobody.sock", args()))
    assert report.problems == ["its socket doesn't accept connections"]


def test_a_running_broker_with_a_live_stream_checks_out(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("continuum_py", reason="continuum_py is built by uv sync or in continuum/.venv")
    broker = os.environ.get("CONTINUUM_BROKER")
    if not broker or not os.path.exists(broker):
        pytest.skip("no broker binary (CONTINUUM_BROKER): run continuum-test")
    import asyncio

    from continuum_py import Client, encode_header

    socket = tmp_path / "b.sock"
    process = subprocess.Popen([broker, "--socket", socket, "--tier0-classes", "16384x16"], stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 10
        while not cs.accepts(str(socket), py_socket.AF_UNIX):
            assert process.poll() is None and time.monotonic() < deadline, "the broker didn't start"
            time.sleep(0.01)

        async def scenario() -> tuple[cs.Report, cs.Report]:
            producer = await Client.connect(str(socket), client_name="camera")
            registered = await producer.register_stream(
                "cam0.video", kind="sensor", format="RawBytes", shape_name="frame", slab_class=0
            )
            now = time.time_ns()
            header = {
                "id": "0199a000-0000-7000-8000-000000000001",
                "stream_id": registered["stream_id"],
                "start_ns": now,
                "end_ns": now,
                "layer": "Layer1Raw",
                "format": "RawBytes",
                "shape_name": "frame",
            }
            await producer.ingest_acked(encode_header(header), inline=b"x")  # ty: ignore[invalid-argument-type]
            report = cs.Report()
            await cs.check_broker(report, socket, args())
            # At a stage whose manifest names a microphone capture hasn't registered, that's a problem.
            staged = cs.Report()
            await cs.check_broker(staged, socket, args(), {"cam0.video", "wave3.audio"}, expect_vector=True)
            await producer.close()
            return report, staged

        report, staged = asyncio.run(scenario())
    finally:
        process.terminate()
        process.wait(timeout=10)
    assert report.problems == [], report.lines
    text = "\n".join(report.lines)
    assert "connected (schema" in text
    assert "Tier 0 class 0 (16 KiB slots): 0 of 16 in use, 0 held" in text
    assert "cam0.video (live): 1 payloads, newest" in text
    assert staged.problems == [
        "wave3.audio: the manifest names its device, but capture hasn't registered it",
        "its search streams aren't registered",
    ], staged.lines


def test_a_stage_takes_the_ports_services_up_to_the_last_one_turned_on() -> None:
    assert cs.stage(None, None) == ()
    assert cs.stage("audio", "continuum-capture") == ("continuum-broker", "continuum-capture")
    assert cs.stage("audio", None) == cs.AUDIO
    assert cs.stage("vision", None)[-1] == "continuum-gateway"
    with pytest.raises(SystemExit, match="isn't one of the audio port's services"):
        cs.stage("audio", "continuum-gateway")


@pytest.mark.parametrize("stopped_unselected", [False, True])
@pytest.mark.parametrize("missing_selected", [False, True])
def test_selected_health_does_not_require_disabled_services(
    tmp_path, monkeypatch, stopped_unselected, missing_selected
):
    manifest = tmp_path / "devices.toml"
    manifest.write_text("device = []\n")
    chosen = ["continuum-broker", "continuum-capture", "see-actions"]
    installed = [*chosen, "continuum-gateway"] if stopped_unselected else chosen
    if missing_selected:
        installed = [name for name in installed if name != "see-actions"]
    states = dict.fromkeys(chosen, "active")
    if stopped_unselected:
        states["continuum-gateway"] = "inactive"
    monkeypatch.setattr(cs.services, "selected", lambda: {name: {} for name in chosen})
    monkeypatch.setattr(cs.services, "installed", lambda: installed)
    monkeypatch.setattr(cs.services, "states", lambda: states)
    gateway = []
    monkeypatch.setattr(cs, "check_gateway", lambda report: gateway.append(True))

    async def broker(report, socket, options, expected_streams=(), expect_vector=False):
        report.ok("broker request succeeded")

    monkeypatch.setattr(cs, "check_broker", broker)
    report = cs.run(args(port="vision", manifest=manifest, selected_only=True))
    expected = ["see-actions: not installed, and this stage needs it"] if missing_selected else []
    assert report.problems == expected
    assert gateway == [], "offline gateway must not become a disabled dependency"


@pytest.mark.parametrize("selected_only", [False, True])
@pytest.mark.parametrize(
    "tier_selected,expected_service",
    [(False, None), (True, None), (True, ["continuum-broker", "continuum-capture", "see-actions"])],
)
@pytest.mark.parametrize("stale_sensor", [False, True])
def test_partial_without_tier_ignores_archive_pressure_but_requires_fresh_capture(
    tmp_path, monkeypatch, selected_only, tier_selected, expected_service, stale_sensor
):
    import asyncio
    from types import SimpleNamespace

    import continuum_py
    from continuum_py import PayloadHeader

    now = 100_000_000_000
    monkeypatch.setattr(cs.time, "time_ns", lambda: now)
    chosen = {"continuum-broker": {}, "continuum-capture": {}, "see-actions": {}}
    if tier_selected:
        chosen["continuum-tier-worker"] = {}
    monkeypatch.setattr(cs.services, "selected", lambda: chosen)
    monkeypatch.setattr(cs, "accepts", lambda *args: True)
    monkeypatch.setattr(cs, "check_arena", lambda report, *args: report.bad("arena held pressure"))
    items = []
    for sequence, stamp in enumerate((10_000_000_000, now - (20 if stale_sensor else 1) * 1_000_000_000)):
        header: PayloadHeader = {
            "id": f"0199a000-0000-7000-8000-{sequence:012d}",
            "stream_id": 7,
            "start_ns": stamp,
            "end_ns": stamp,
            "layer": "Layer1Raw",
            "format": "RawBytes",
            "shape_name": "frame",
        }
        items.append({"header": continuum_py.encode_header(header), "locator": {"kind": "Shm"}})

    class Client:
        broker_schema = (1, 99)

        async def list_streams(self):
            return [{"name": "cam0.video", "kind": "Sensor", "live": True}]

        async def query(self, *args):
            return {"items": items}

        async def close(self, reason):
            pass

    async def connect(*args, **kwargs):
        return Client()

    monkeypatch.setattr(continuum_py, "Client", SimpleNamespace(connect=connect))
    report = cs.Report()
    asyncio.run(
        cs.check_broker(
            report,
            tmp_path / "fixture.sock",
            args(
                port="vision", selected_only=selected_only, expected_service=expected_service if selected_only else None
            ),
            {"cam0.video"},
        )
    )
    archive_required = not selected_only or (tier_selected and expected_service is None)
    assert any("arena held pressure" in problem for problem in report.problems) == archive_required
    assert any("flush lag" in problem for problem in report.problems) == archive_required
    assert any("older than" in problem for problem in report.problems) == stale_sensor
    if not archive_required and not stale_sensor:
        assert report.problems == []


@pytest.mark.parametrize(
    "flags",
    [
        ["--expected-service", "listen"],
        ["--port", "vision", "--expected-service", "listen"],
        ["--port", "audio", "--selected-only", "--expected-service", "see-actions"],
        ["--port", "vision", "--through", "listen", "--selected-only", "--expected-service", "see-actions"],
    ],
)
def test_explicit_expected_services_require_a_valid_selected_port_stage(monkeypatch, flags):
    def must_not_probe(args):
        pytest.fail("invalid explicit scope must fail before any health probes")

    monkeypatch.setattr(cs, "run", must_not_probe)
    with pytest.raises(SystemExit) as error:
        cs.main(flags)
    assert error.value.code == 2


def test_explicit_running_scope_does_not_require_kept_stopped_services(tmp_path, monkeypatch):
    manifest = tmp_path / "devices.toml"
    manifest.write_text("device = []\n")
    active = ["continuum-broker", "continuum-capture", "see-actions"]
    monkeypatch.setattr(cs.services, "selected", lambda: {name: {} for name in cs.VISION})
    monkeypatch.setattr(cs.services, "installed", lambda: list(cs.VISION))
    monkeypatch.setattr(
        cs.services, "states", lambda: {name: "active" if name in active else "inactive" for name in cs.VISION}
    )
    gateway = []
    monkeypatch.setattr(cs, "check_gateway", lambda report: gateway.append(True))

    async def broker(report, socket, options, expected_streams=(), expect_vector=False):
        assert not expect_vector
        report.ok("broker request succeeded")

    monkeypatch.setattr(cs, "check_broker", broker)
    report = cs.run(args(port="vision", manifest=manifest, selected_only=True, expected_service=active))
    assert report.problems == [] and gateway == []


def test_rtsp_manifest_is_checked_for_live_video(tmp_path):
    path = tmp_path / "devices.toml"
    path.write_text('[[device]]\nname="entryway"\nkind="rtsp"\nrtsp={url="rtsp://192.168.1.20/main"}\n')
    assert cs.manifest_streams(path) == {"entryway.video"}


PLAN = """tier0 source=config/devices.toml
baseline_bytes=11073339392 camera_increment_bytes=589955072 ring_region_bytes=67108864 total_arena_bytes=11663294464
camera=cam0 format=MJPEG fps=30 ring_seconds=10 slots_per_class=383 class_sizes=[65536, 2097152, 16777216]
camera=patio format=H264AnnexB nominal_fps=30 target_ring_seconds=10 byte_budget_limited=true class_slots=[(65536, 383), (16384, 383), (1048576, 49)]
camera=kitchen format=H264AnnexB nominal_fps=30 target_ring_seconds=10 byte_budget_limited=true class_slots=[(65536, 383), (16384, 383), (1048576, 49)]
class=0 configured_slot_bytes=65536 count=4096
class=1 configured_slot_bytes=16384 count=766
class=2 configured_slot_bytes=1048576 count=98
"""
# The live arena for PLAN: (slot bytes, count, used, held). The 1 MiB class is only the two rings, entirely held.
ROWS = [(65536, 4096, 900, 700), (16384, 766, 700, 300), (1048576, 98, 98, 98)]
HEALTHY = {"patio": cs.RingHealth(300, 0, 0), "kitchen": cs.RingHealth(300, 0, 0)}


def test_the_plan_sums_each_slot_size_the_network_rings_reserve() -> None:
    plan = cs.parse_plan(PLAN)
    assert plan == cs.Tier0Plan(
        classes={65536: 4096, 16384: 766, 1048576: 98},
        network={65536: 766, 16384: 766, 1048576: 98},
        cameras=("patio", "kitchen"),
    )


def test_output_without_classes_is_no_plan() -> None:
    assert cs.parse_plan("error: the manifest is invalid\n") is None


def judged(rows=ROWS, health=HEALTHY, plan=PLAN) -> cs.Report:
    report = cs.Report()
    cs.judge_arena(report, rows, 0.9, cs.parse_plan(plan) if plan else None, health)
    return report


def test_a_fully_held_network_ring_class_passes_while_its_cameras_drop_nothing() -> None:
    report = judged()
    assert report.problems == [], report.lines
    assert any("patio ring: 300 pictures in its last interval, none dropped" in line for line in report.lines)


@pytest.mark.parametrize(
    ("kitchen", "problem"),
    [
        pytest.param(
            cs.RingHealth(45, 255, 0),
            "kitchen ring, last interval: 255 dropped under pressure, 0 overwritten unflushed, 45 kept",
            id="pressure",
        ),
        pytest.param(
            cs.RingHealth(300, 0, 2),
            "kitchen ring, last interval: 0 dropped under pressure, 2 overwritten unflushed, 300 kept",
            id="overwrite",
        ),
        pytest.param(cs.RingHealth(0, 0, 0), "kitchen ring: no pictures in its last interval", id="stalled"),
    ],
)
def test_a_network_ring_fails_on_its_own_health(kitchen: cs.RingHealth, problem: str) -> None:
    assert judged(health={**HEALTHY, "kitchen": kitchen}).problems == [problem]


def test_a_ring_without_a_health_line_yet_is_noted() -> None:
    report = judged(health={"patio": HEALTHY["patio"]})
    assert report.problems == []
    assert "kitchen ring: no health line in capture's log yet" in "\n".join(report.lines)


def test_a_shared_class_keeps_room_outside_its_ring_slots() -> None:
    # 766 of the 4096 64 KiB slots are the rings'. 3700 held is 90% of the class but 2934 of the other 3330 (88%);
    # 3800 leaves 3034 of them (91%).
    assert judged(rows=[(65536, 4096, 4096, 3700), *ROWS[1:]]).problems == []
    assert judged(rows=[(65536, 4096, 4096, 3800), *ROWS[1:]]).problems == [
        "Tier 0 class 0 (64 KiB slots): 4096 of 4096 in use, 3800 held, at or above 90% held outside its 766 ring slots"
    ]


@pytest.mark.parametrize(
    "plan", [pytest.param(None, id="no-plan"), pytest.param(PLAN.replace("98", "147"), id="stale")]
)
def test_without_a_matching_plan_occupancy_alone_judges(plan: str | None) -> None:
    assert judged(plan=plan).problems == [
        "Tier 0 class 2 (1024 KiB slots): 98 of 98 in use, 98 held, at or above 90% held"
    ]


def test_ring_health_is_each_cameras_last_interval_and_restarts_on_reconnect() -> None:
    at = "2026-11-09T18:00:00-08:00 [capture-worker] "
    lines = [
        at + "INFO patio: RTSP frames=2791 reader_dropped=2 pressure_dropped=4 unflushed_overwrites=0",
        at + "INFO kitchen: RTSP frames=900 reader_dropped=1 pressure_dropped=13 unflushed_overwrites=0",
        at + "DEBUG patio: RTSP frames=3091 reader_dropped=2 pressure_dropped=4 unflushed_overwrites=0 full_waited=2",
        at + "WARN kitchen: reconnecting RTSP after host sleep",
        at + "INFO kitchen: RTSP frames=40 reader_dropped=0 pressure_dropped=3 unflushed_overwrites=0",
        at + "INFO bedroom: RTSP frames=185 reader_dropped=1 pressure_dropped=17 unflushed_overwrites=1",
    ]
    assert cs.ring_health(lines) == {
        "patio": cs.RingHealth(300, 0, 0),
        "kitchen": cs.RingHealth(40, 3, 0),
        "bedroom": cs.RingHealth(185, 17, 1),
    }


def test_a_log_tail_drops_the_partial_first_line(tmp_path: pathlib.Path) -> None:
    log = tmp_path / "capture.log"
    log.write_text("first line\nsecond line\nthird\n")
    assert cs.log_tail(log, limit=14) == ["third"]
    assert cs.log_tail(log) == ["first line", "second line", "third"]
    assert cs.log_tail(tmp_path / "missing.log") == []
