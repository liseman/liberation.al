import asyncio

import pytest

from liberational import segment_log
from liberational.proximity.network import NetworkDevice, NetworkInventory
from liberational.proximity.network_worker import (
    INVENTORY,
    OBSERVATIONS,
    WIFI,
    NetworkSource,
    NetworkWorker,
    SegmentSink,
    WifiRecorder,
)
from liberational.proximity.wifi import AccessPoint


class Source:
    def __init__(self, inventory):
        self.inventory = inventory
        self.calls = 0

    def poll(self, now):
        self.calls += 1
        return self.inventory


class Sink:
    def __init__(self, fail=None):
        self.rows = []
        self.fail = fail

    def write(self, stream, source, records):
        if stream == self.fail:
            raise OSError("synthetic unavailable store")
        self.rows.extend((stream, source, r) for r in records)

    def close(self):
        pass


def snapshot(at=100, devices=None, complete=True):
    if devices is None:
        devices = [
            NetworkDevice(
                "mac_address",
                "02:00:00:00:00:01",
                mac="02:00:00:00:00:01",
                connected=True,
                hostname="camera",
                access_point_name="Upstairs",
            )
        ]
    return NetworkInventory("unifi", "home", at, tuple(devices), complete, complete)


def test_a_poll_writes_its_devices_before_its_own_complete_row():
    sink = Sink()
    NetworkWorker(Source(snapshot()), sink).process(snapshot(), poll_started_at=99)
    assert [(stream, source) for stream, source, _ in sink.rows] == [(OBSERVATIONS, "unifi"), (INVENTORY, "unifi")]
    device, poll = sink.rows[0][2], sink.rows[1][2]
    assert (device["mac"], device["hostname"], device["ap_name"], device["connected"]) == (
        "02:00:00:00:00:01",
        "camera",
        "Upstairs",
        True,
    )
    assert device["inventory_id"] == poll["id"]
    assert (poll["start"], poll["end"], poll["complete"], poll["authoritative"], poll["device_count"]) == (
        99,
        100,
        True,
        True,
        1,
    )


def test_a_poll_whose_devices_failed_to_write_writes_no_complete_row():
    sink = Sink(fail=OBSERVATIONS)
    with pytest.raises(OSError, match="unavailable store"):
        NetworkWorker(Source(snapshot()), sink).process(snapshot())
    assert sink.rows == []


def test_empty_authoritative_inventory_still_writes_a_complete_poll():
    sink = Sink()
    NetworkWorker(Source(snapshot()), sink).process(snapshot(devices=[]))
    assert [stream for stream, _, _ in sink.rows] == [INVENTORY]
    assert sink.rows[0][2]["device_count"] == 0 and sink.rows[0][2]["complete"]


def test_cached_connection_unknown_makes_the_poll_partial_even_if_the_source_claims_complete():
    device = NetworkDevice("mac_address", "02:00:00:00:00:01", connected=None)
    sink = Sink()
    NetworkWorker(Source(snapshot()), sink).process(snapshot(devices=[device]))
    poll = sink.rows[-1][2]
    assert not poll["complete"] and not poll["authoritative"]
    assert sink.rows[0][2]["connected"] is None


def test_polls_read_back_from_the_segment_store_as_the_devices_reader_sees_them(tmp_path):
    sink = SegmentSink(tmp_path)
    worker = NetworkWorker(Source(snapshot()), sink)
    worker.process(snapshot(1_800_000_000.0), poll_started_at=1_799_999_990.0)
    cached = NetworkDevice("mac_address", "02:00:00:00:00:02", connected=None)
    worker.process(snapshot(1_800_000_060.0, devices=[cached]))
    sink.close()
    log = segment_log.Log(tmp_path)
    seen = log.range(OBSERVATIONS, 1_799_999_000.0, 1_800_001_000.0)
    assert [(r["identifier"], r["connected"], r["source"]) for r in seen] == [
        ("02:00:00:00:00:01", True, "unifi"),
        ("02:00:00:00:00:02", None, "unifi"),
    ]
    polls = log.range(INVENTORY, 1_799_999_000.0, 1_800_001_000.0)
    assert [(p["start"], p["complete"], p["warnings"]) for p in polls] == [
        (1_799_999_990.0, True, []),
        (1_800_000_060.0, False, []),
    ]


def test_authoritative_empty_controller_result_does_not_fall_back_to_stale_cache():
    host = Source(snapshot(100))
    controller = Source(snapshot(100, []))
    result = NetworkSource(host, controller).poll(100)
    assert result.devices == () and result.complete
    assert host.calls == 0


def test_partial_positive_controller_result_is_not_replaced_by_host_cache():
    host = Source(snapshot(100, []))
    controller = Source(snapshot(100, complete=False))
    result = NetworkSource(host, controller).poll(100)
    assert len(result.devices) == 1 and not result.complete
    assert host.calls == 0


def test_failed_controller_falls_back_without_exposing_exception_credentials():
    class Failed:
        def poll(self, now):
            raise RuntimeError("https://secret:password@192.168.1.1/token")

    cached = NetworkInventory("host_network", "home", 100, (), False, False)
    result = NetworkSource(Source(cached), Failed()).poll(100)
    assert not result.complete and not result.authoritative
    assert "password" not in str(result.warnings)
    assert "RuntimeError" in str(result.warnings)


def test_stop_after_poll_prevents_any_publication():
    async def run():
        stop = asyncio.Event()

        class StoppingSource:
            def poll(self, now):
                stop.set()
                return snapshot()

        sink = Sink()
        await NetworkWorker(StoppingSource(), sink).run(stop)
        assert sink.rows == []

    asyncio.run(run())


def ap(bssid, signal=60, ssid="home"):
    return AccessPoint(bssid, ssid, signal, "percent", 2437)


class Heard:
    def __init__(self, *scans):
        self.scans = list(scans)

    def scan(self):
        return self.scans.pop(0)


def written(sink):
    return [(stream, source, r["start"], r["bssid"]) for stream, source, r in sink.rows]


def test_wifi_is_written_when_the_set_changes_and_restated_after_the_rewrite_interval():
    a, b = "02:00:00:00:00:0a", "02:00:00:00:00:0b"
    sink = Sink()
    recorder = WifiRecorder(
        Heard([ap(a)], [ap(a, signal=20)], [ap(a), ap(b)], [ap(a), ap(b)], [ap(a), ap(b)]),
        sink,
        "desk",
        rewrite_seconds=600,
    )
    assert [recorder.poll(t) for t in (1000, 1060, 1120, 1719, 1720)] == [1, 0, 2, 0, 2]
    assert written(sink) == [
        (WIFI, "wifi", 1000, a),
        (WIFI, "wifi", 1120, a),
        (WIFI, "wifi", 1120, b),
        (WIFI, "wifi", 1720, a),
        (WIFI, "wifi", 1720, b),
    ]
    first, second = sink.rows[1][2], sink.rows[2][2]
    assert first["scan_id"] == second["scan_id"] != sink.rows[0][2]["scan_id"]
    assert first["observer"] == "desk"


def test_wifi_that_cannot_be_read_writes_nothing_and_a_later_scan_still_lands():
    sink = Sink()
    recorder = WifiRecorder(Heard(None, None, [ap("02:00:00:00:00:0a")]), sink, "desk")
    assert [recorder.poll(t) for t in (1000, 1060, 1120)] == [None, None, 1]
    assert written(sink) == [(WIFI, "wifi", 1120, "02:00:00:00:00:0a")]


def test_hearing_nothing_after_a_set_writes_no_rows_but_counts_as_a_change():
    a = "02:00:00:00:00:0a"
    sink = Sink()
    recorder = WifiRecorder(Heard([ap(a)], [], [ap(a)]), sink, "desk")
    assert [recorder.poll(t) for t in (1000, 1060, 1120)] == [1, 0, 1]
    assert written(sink) == [(WIFI, "wifi", 1000, a), (WIFI, "wifi", 1120, a)]


def test_a_failing_wifi_read_never_stops_the_network_inventory():
    async def run():
        stop = asyncio.Event()

        class Broken:
            def scan(self):
                stop.set()
                raise RuntimeError("adapter vanished")

        sink = Sink()
        await NetworkWorker(Source(snapshot()), sink, wifi=WifiRecorder(Broken(), sink, "desk")).run(stop)
        assert [stream for stream, _, _ in sink.rows] == [OBSERVATIONS, INVENTORY]

    asyncio.run(run())


def test_wifi_rows_read_back_from_the_segment_store(tmp_path):
    sink = SegmentSink(tmp_path)
    heard = [
        AccessPoint("02:00:00:00:00:0a", None, -48, "dbm", 5180),
        AccessPoint("02:00:00:00:00:0b", "Café: 2", 71, "percent", 2412),
    ]
    WifiRecorder(Heard(heard), sink, "desk").poll(1_800_000_000.0)
    sink.close()
    seen = segment_log.Log(tmp_path).range(WIFI, 1_799_999_000.0, 1_800_001_000.0)
    assert sorted((r["bssid"], r["ssid"], r["signal"], r["signal_unit"], r["freq_mhz"], r["source"]) for r in seen) == [
        ("02:00:00:00:00:0a", None, -48, "dbm", 5180, "wifi"),
        ("02:00:00:00:00:0b", "Café: 2", 71, "percent", 2412, "wifi"),
    ]
    assert {r["start"] for r in seen} == {1_800_000_000.0}
    assert len({r["scan_id"] for r in seen}) == 1
