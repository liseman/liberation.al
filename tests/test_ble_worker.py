import asyncio
from collections.abc import Callable
from unittest.mock import create_autospec

import pytest

from liberational import config as cfg
from liberational import segment_log, services
from liberational.proximity import worker as w
from liberational.proximity.ble import Advertisement
from liberational.proximity.sources import HostSource, SerialSource


def observation(identifier="C0:00:00:00:00:01", at=100):
    return Advertisement(
        at,
        "esp32_serial",
        "AABBCCDDEEFF",
        "bluetooth_address",
        identifier,
        address_type="random",
        rssi=-60,
        manufacturer_data={0x1234: b"abcdefgh"},
        scan_mode="passive",
    )


def test_default_source_stays_esp32_without_host_discovery_authorization():
    assert isinstance(w.source_from_config(), SerialSource)


@pytest.mark.parametrize("source", ["bluez", "corebluetooth"])
def test_computer_sources_need_separate_active_scan_authorization(source):
    with cfg.override({"bluetooth": {"source": source}}), pytest.raises(ValueError, match="allow_active_scan"):
        w.source_from_config()


@pytest.mark.parametrize("source,platform", [("bluez", "linux"), ("corebluetooth", "darwin")])
def test_authorized_platform_source_is_constructed_without_native_api_access(monkeypatch, source, platform):
    monkeypatch.setattr(w.sys, "platform", platform)
    with cfg.override({"bluetooth": {"source": source, "allow_active_scan": True}}):
        chosen = w.source_from_config()
    assert isinstance(chosen, HostSource)
    assert chosen.metadata["backend"] == source
    assert chosen.stats["active_sensors"] == 0


@pytest.mark.parametrize("source,platform", [("bluez", "darwin"), ("corebluetooth", "linux")])
def test_wrong_platform_is_explicit_instead_of_silently_scanning_another_source(monkeypatch, source, platform):
    monkeypatch.setattr(w.sys, "platform", platform)
    with (
        cfg.override({"bluetooth": {"source": source, "allow_active_scan": True}}),
        pytest.raises(ValueError, match="requires"),
    ):
        w.source_from_config()


class Sink:
    def __init__(self):
        self.rows = []
        self.started = False

    async def start(self):
        self.started = True

    async def publish(self, stream, record):
        self.rows.append((stream, record))


class Source:
    def __init__(self):
        self.stats = {"dropped_reports": 0}
        self.sensors: dict[str, str] = {}

    async def run(self, emit, stop):
        emit(observation(at=100))
        emit(observation(at=101))


def test_a_flush_reserves_one_record_of_the_rate_for_coverage():
    sink = Sink()
    moment = [0.0]
    worker = w.Worker(Source(), sink, max_records_per_second=3, clock=lambda: moment[0])
    worker.aggregates.add(observation())
    worker.aggregates.add(observation(at=101))
    worker.aggregates.add(observation("C0:00:00:00:00:02"))
    worker.aggregates.add(observation("C0:00:00:00:00:02", at=101))
    asyncio.run(worker.flush(106))
    first = list(sink.rows)
    asyncio.run(worker.flush(106, force=True))
    assert sink.rows == first, "one flush a second"
    moment[0] = 1
    asyncio.run(worker.flush(107))
    second = sink.rows[len(first) :]
    assert [stream for stream, _ in first] == [w.SIGHTINGS, w.SIGHTINGS, w.COVERAGE]
    assert [stream for stream, _ in second] == [w.COVERAGE]
    assert len({row["identifier"] for stream, row in first if stream == w.SIGHTINGS}) == 2
    assert sink.rows[-1][1]["aggregation"]["pending_devices"] == 0


def test_a_running_worker_says_it_started_and_reports_what_it_recorded_every_ten_minutes(caplog):
    """bluetooth.log stayed empty on a healthy run; it logged only a stop or a failure."""
    moment = [0.0]
    source = Source()
    source.stats["lost_reports"] = 0
    source.sensors = {"/dev/ttyACM0": "Desk (AABBCCDDEE01)"}
    worker = w.Worker(source, Sink(), clock=lambda: moment[0])
    with caplog.at_level("INFO"):
        asyncio.run(worker.run(asyncio.Event()))  # the finite source ends the run
    assert "Bluetooth started: Source; a summary every 10 min" in caplog.text
    caplog.clear()
    worker.aggregates.add(observation())
    worker.aggregates.add(observation("C0:00:00:00:00:02"))
    source.stats["lost_reports"] = 40
    moment[0] = w.REPORT_S - 1
    with caplog.at_level("INFO"):
        asyncio.run(worker.flush(200, force=True))
    assert "sightings" not in caplog.text  # not yet ten minutes
    moment[0] = w.REPORT_S + 1
    with caplog.at_level("INFO"):
        asyncio.run(worker.flush(201, force=True))
    assert (
        "Bluetooth: 3 sightings in 10 min; 40 reports over the sensors' output cap, 0 dropped here; "
        "1 sensors on: Desk (AABBCCDDEE01)"
    ) in caplog.text


def ten_minute_summary(source, stats):
    """The worker's summary line after ten minutes in which the source's counters grew by stats."""
    moment = [0.0]
    worker = w.Worker(source, Sink(), clock=lambda: moment[0])
    for name, count in stats.items():
        source.stats[name] = source.stats.get(name, 0) + count
    moment[0] = w.REPORT_S + 1
    worker._report()
    return worker


@pytest.mark.parametrize(
    "stats,level,counts",
    [
        pytest.param(
            {"lost_reports": 40}, "INFO", "40 reports over the sensors' output cap, 0 dropped here", id="gaps"
        ),
        pytest.param(
            {"lost_reports": 40, "dropped_reports": 2},
            "WARNING",
            "40 reports over the sensors' output cap, 2 dropped here",
            id="queue-full",
        ),
    ],
)
def test_only_reports_dropped_on_this_machine_make_the_summary_a_warning(caplog, stats, level, counts):
    with caplog.at_level("INFO"):
        ten_minute_summary(Source(), stats)
    [record] = [r for r in caplog.records if "sightings" in r.getMessage()]
    assert record.levelname == level
    assert counts in record.getMessage()


def test_a_full_aggregator_counts_as_dropped_here(caplog):
    moment = [0.0]
    worker = w.Worker(Source(), Sink(), max_devices=1, clock=lambda: moment[0])
    worker.aggregates.add(observation("C0:00:00:00:00:01"))
    worker.aggregates.add(observation("C0:00:00:00:00:02"))  # no room for a second device
    moment[0] = w.REPORT_S + 1
    with caplog.at_level("INFO"):
        worker._report()
    assert "0 reports over the sensors' output cap, 1 dropped here" in caplog.text
    assert "WARNING" in [r.levelname for r in caplog.records]


class IdleScanner:
    """A host discovery session that hears nothing; tests set its counters."""

    def __init__(self) -> None:
        self.stats = {"invalid_reports": 0, "lost_reports": 0}
        self.metadata = {"backend": "bluez", "adapter": "hci0", "scan_mode": "os_managed_active"}

    async def start(self, emit: Callable[[Advertisement], None]) -> None:
        pass

    async def stop(self) -> None:
        pass

    def check(self) -> None:
        pass


def test_host_discovery_unreadable_reports_are_dropped_here_not_over_a_sensor_cap(caplog):
    scanner = IdleScanner()
    source = HostSource(scanner, allow_active_scan=True)
    scanner.stats.update(invalid_reports=5, lost_reports=5)  # bluez counts an unreadable report in both
    with caplog.at_level("INFO"):
        ten_minute_summary(source, {})
    assert "0 reports over the sensors' output cap, 5 dropped here" in caplog.text


def test_source_to_worker_is_bounded_and_flushes_a_finite_replay():
    sink = Sink()
    worker = w.Worker(Source(), sink)
    asyncio.run(worker.run(asyncio.Event()))
    sightings = [row for stream, row in sink.rows if stream == w.SIGHTINGS]
    assert sink.started and len(sightings) == 1
    assert sightings[0]["reports"] == 2
    assert sightings[0]["rssi"] == {"samples": 2, "min": -60, "max": -60, "mean": -60.0}


def test_the_segment_sink_keeps_ids_times_and_the_whole_record(tmp_path):
    sink = w.SegmentSink(tmp_path)
    sighting = {
        "id": "0199a2b0-0000-7000-8000-000000000001",
        "start": 100.0,
        "end": 101.0,
        "identifier": "C0:00:00:00:00:01",
        "rssi": {"samples": 2, "min": -60, "max": -55, "mean": -57.5},
    }
    asyncio.run(sink.publish(w.SIGHTINGS, sighting))
    asyncio.run(sink.publish(w.COVERAGE, {"id": "0199a2b0-0000-7000-8000-000000000002", "start": 101.0, "end": 101.0}))
    sink.close()
    log = segment_log.Log(tmp_path)
    (got,) = log.range(w.SIGHTINGS, 99, 102)
    assert (got["id"], got["start"], got["end"], got["identifier"], got["rssi"]) == (
        sighting["id"],
        100.0,
        101.0,
        "C0:00:00:00:00:01",
        sighting["rssi"],
    )
    assert [c["id"] for c in log.range(w.COVERAGE, 99, 102)] == ["0199a2b0-0000-7000-8000-000000000002"]


@pytest.mark.parametrize(
    ("enabled", "disable", "wanted"), [(False, [], False), (True, [], True), (True, ["bluetooth"], False)]
)
def test_service_enabled_by_flag_is_generic_and_explicit_disable_wins(tmp_path, monkeypatch, enabled, disable, wanted):
    spec = tmp_path / "services.toml"
    spec.write_text('[bluetooth]\ncommand="pack-bluetooth"\ndefault=false\nenabled_by="bluetooth.enabled"\n')
    monkeypatch.setattr(services, "SPEC", spec)
    with cfg.override({"bluetooth": {"enabled": enabled}, "services": {"disable": disable, "enable": []}}):
        assert ("bluetooth" in services.selected()) is wanted


def test_disabled_worker_does_not_even_construct_a_source(monkeypatch):
    factory = create_autospec(w.source_from_config, side_effect=AssertionError("disabled source"))
    monkeypatch.setattr(w, "source_from_config", factory)
    with cfg.override({"bluetooth": {"enabled": False}}):
        w.main([])


def test_stalled_ack_has_one_inflight_write():
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()

        class SlowSink(Sink):
            async def publish(self, stream, record):
                if stream == w.SIGHTINGS:
                    entered.set()
                    await release.wait()
                await super().publish(stream, record)

        sink = SlowSink()
        worker = w.Worker(Source(), sink, max_devices=2)
        worker.aggregates.add(observation())
        worker.aggregates.add(observation(at=101))
        flushing = asyncio.create_task(worker.flush(106))
        await asyncio.wait_for(entered.wait(), timeout=1)
        assert sink.rows == []
        worker.aggregates.add(observation("C0:00:00:00:00:02"))
        worker.aggregates.add(observation("C0:00:00:00:00:03"))
        worker.aggregates.add(observation("C0:00:00:00:00:04"))
        assert worker.aggregates.stats["pending_devices"] == 2
        assert worker.aggregates.stats["dropped_capacity"] == 1
        release.set()
        await flushing
        assert worker.published == 1

    asyncio.run(run())


def test_failed_ack_reports_uncommitted_batch():
    class RefusingSink(Sink):
        async def publish(self, stream, record):
            raise ConnectionError("broker gone")

    worker = w.Worker(Source(), RefusingSink())
    worker.aggregates.add(observation())
    worker.aggregates.add(observation("C0:00:00:00:00:02"))
    with pytest.raises(ConnectionError, match="broker gone"):
        asyncio.run(worker.flush(106))
    assert worker.unacknowledged_sightings == 2
    assert worker.published == 0


def test_slow_ack_cannot_allow_a_second_batch_to_burst_in_the_same_second():
    moment = [0.0]

    class SlowSink(Sink):
        async def publish(self, stream, record):
            if stream == w.SIGHTINGS:
                moment[0] += 10  # Simulate ten seconds spent waiting for commit, without a wall-clock sleep.
            await super().publish(stream, record)

    sink = SlowSink()
    worker = w.Worker(Source(), sink, max_records_per_second=3, clock=lambda: moment[0])
    worker.aggregates.add(observation())
    worker.aggregates.add(observation("C0:00:00:00:00:02"))
    worker.aggregates.add(observation("C0:00:00:00:00:03"))  # past this second's two sightings
    asyncio.run(worker.flush(106))
    before = list(sink.rows)
    asyncio.run(worker.flush(106, force=True))
    assert sink.rows == before
    assert worker.aggregates.stats["pending_devices"] == 1
