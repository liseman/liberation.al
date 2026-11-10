import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest
from hypothesis import given
from hypothesis import strategies as st

from liberational.proximity import advertising as ad
from liberational.proximity import sources
from liberational.proximity.serial_protocol import Framer, Hello, Report, SensorClock, frame, parse

CHIP = "AABBCCDDEEFF"
BOOT = "01020304"
ADDR = "C0:00:00:00:00:01"


def test_os_advertising_fields_share_manufacturer_and_uuid_semantics_without_inventing_raw_packets():
    data = ad.platform_data(
        manufacturer_bytes=b"\x34\x12abcdefgh",
        service_data={"180D": b"heart"},
        service_uuids=["180D", "0000180D-0000-1000-8000-00805F9B34FB"],
        local_name=None,
        tx_power=127,
    )
    assert data.manufacturer_data == {0x1234: b"abcdefgh"}
    assert data.service_data == {"0000180d-0000-1000-8000-00805f9b34fb": b"heart"}
    assert data.service_uuids == ("0000180d-0000-1000-8000-00805f9b34fb",)
    assert data.local_name is None and data.tx_power is None
    assert not data.malformed and not data.truncated


def test_platform_payload_copy_is_bounded_before_conversion_and_reports_truncation():
    data = ad.platform_data(manufacturer_bytes=bytearray(b"\x34\x12" + b"x" * 100000))
    assert len(data.manufacturer_data[0x1234]) == 65534
    assert data.truncated
    bounded = ad.platform_data(manufacturer_data={k: b"x" for k in range(100)}, service_uuids=["180D"] * 100)
    assert len(bounded.manufacturer_data) == 64 and bounded.truncated


def test_platform_invalid_fields_are_unknown_with_explicit_malformed_evidence():
    data = ad.platform_data(manufacturer_bytes=b"\x01", manufacturer_data={True: b"x"}, service_data={"bad": b"x"})
    assert data.manufacturer_data == {} and data.service_data == {}
    assert data.malformed


def hello(chip=CHIP, boot=BOOT, uptime=1000000, name=""):
    return frame(["BLE1", "HELLO", chip, boot, str(uptime), name.encode().hex()])


def report(counter=1, uptime=2000000, payload=b"\x02\x01\x06", chip=CHIP, boot=BOOT, address=ADDR):
    return frame(["BLE1", "ADV", chip, boot, str(counter), str(uptime), address, "1", "-60", "0", payload.hex()])


def test_advertising_fields_use_wire_endianness_and_keep_unknowns_unknown():
    fields = ad.decode(bytes.fromhex("02010605094e6f6465020afc05ff3412010203030f1805160f180102"))
    assert fields.local_name == "Node"
    assert fields.manufacturer_data == {0x1234: b"\x01\x02"}
    assert fields.service_uuids == ("0000180f-0000-1000-8000-00805f9b34fb",)
    assert fields.service_data == {"0000180f-0000-1000-8000-00805f9b34fb": b"\x01\x02"}
    assert fields.tx_power == -4
    assert not fields.malformed
    empty = ad.decode(b"")
    assert empty.local_name is None and empty.tx_power is None


def test_uuid128_is_little_endian_not_guid_mixed_endian():
    assert ad.service_uuid(bytes(range(16))) == "0f0e0d0c-0b0a-0908-0706-050403020100"


@pytest.mark.parametrize("payload", [b"\x05\xff\x01", b"\x02\xff\x01", b"\x02\x03\x01", b"\x02\x16\x01"])
def test_malformed_ad_structures_are_not_certified_complete(payload):
    assert ad.decode(payload).malformed


def test_complete_name_wins_over_later_shortened_name():
    assert ad.decode(bytes.fromhex("05094e6f646503084e6f")).local_name == "Node"


def test_serial_checksum_detects_mutated_or_unversioned_data():
    wire = report()
    with pytest.raises(ValueError, match="checksum"):
        parse(wire.replace(b"-60", b"-61"))
    with pytest.raises(ValueError, match="protocol"):
        parse(frame(["BLE2", "HELLO", CHIP, BOOT, "0", ""]))


@given(st.binary(max_size=31), st.integers(min_value=0, max_value=2**32 - 1), st.integers(min_value=0, max_value=2**48))
def test_serial_roundtrip_preserves_raw_payload_counter_and_uptime(payload, counter, uptime):
    decoded = parse(report(counter=counter, uptime=uptime, payload=payload))
    assert isinstance(decoded, Report)
    assert (decoded.payload, decoded.counter, decoded.uptime_us) == (payload, counter, uptime)
    assert decoded.rssi == -60 and decoded.channel is None and decoded.address_type == "random"


@given(st.integers(min_value=0, max_value=150))
def test_partial_serial_reads_preserve_complete_frames(cut):
    wire = hello() + report()
    split = min(cut, len(wire))
    f = Framer()
    lines = f.feed(wire[:split]) + f.feed(wire[split:])
    assert [parse(x).__class__ for x in lines] == [Hello, Report]


def test_oversized_line_is_discarded_until_newline_then_recovers():
    f = Framer()
    assert f.feed(b"x" * 10000) == []
    assert len(f.buffer) <= 4096
    assert f.feed(b"remainder\n" + hello()) == [hello().rstrip(b"\n")]
    assert f.invalid == 1


def test_per_sensor_clock_counter_wrap_reboot_and_name_override():
    sensor = SensorClock({CHIP: "Kitchen"})
    assert sensor.accept(parse(hello()), 101) is None
    first = sensor.accept(parse(report(counter=2**32 - 1)), 102.2)
    assert first is not None
    assert first.observed_at == pytest.approx(102)
    assert first.received_at == 102.2
    assert first.sensor_name == "Kitchen" and first.adapter == CHIP and first.scan_mode == "passive"
    wrapped = sensor.accept(parse(report(counter=0, uptime=3000000)), 103)
    assert wrapped is not None and wrapped.device_counter == 0
    assert sensor.accept(parse(report(counter=0, uptime=3000000)), 104) is None
    assert sensor.reordered_reports == 1
    after_gap = sensor.accept(parse(report(counter=3, uptime=4000000)), 104)
    assert after_gap is not None and after_gap.device_counter == 3
    assert sensor.lost_reports == 2
    sensor.accept(parse(hello(boot="05060708", uptime=1000)), 200)
    reset = sensor.accept(parse(report(counter=1, uptime=2000, boot="05060708")), 200.1)
    assert reset is not None
    assert reset.observed_at == pytest.approx(200.001)
    assert reset.boot_id == "05060708"


def test_report_before_hello_and_other_sensor_are_refused():
    sensor = SensorClock()
    with pytest.raises(ValueError, match="hello"):
        sensor.accept(parse(report()), 100)
    sensor.accept(parse(hello()), 100)
    with pytest.raises(ValueError, match="hello"):
        sensor.accept(parse(report(chip="111111111111")), 101)


def test_two_sensors_keep_independent_host_offsets():
    left, right = SensorClock(), SensorClock()
    left.accept(parse(hello(uptime=1_000_000)), 101)
    right.accept(parse(hello(uptime=50_000_000, chip="111111111111")), 101)
    a = left.accept(parse(report(uptime=2_000_000)), 102)
    b = right.accept(parse(report(uptime=51_000_000, chip="111111111111")), 102)
    assert a is not None and b is not None
    assert a.observed_at == b.observed_at == 102
    assert a.clock_offset_s == 100 and b.clock_offset_s == 51
    assert a.adapter != b.adapter


def test_serial_source_is_bounded_recovers_on_unplug_and_closes_port():
    port = create_autospec(sources.SerialPort, instance=True)
    port.read.side_effect = [hello() + report() + report(counter=2), OSError("unplug")]
    source = sources.SerialSource(queue_size=1, opener=lambda path, baud: port, clock=lambda: 102)
    source.read_port("fake-port", threading.Event())
    assert source.pending.qsize() == 1
    assert source.stats["dropped_reports"] == 1
    assert source.stats["disconnects"] == 1
    assert source.stats["active_sensors"] == 0
    port.close.assert_called_once_with()


def test_a_sensor_says_in_the_log_when_it_attaches_and_detaches(caplog):
    """A healthy run shows in bluetooth.log: which sensor, by name and chip, on which port."""
    port = create_autospec(sources.SerialPort, instance=True)
    port.read.side_effect = [hello(name="Desk") + report(), OSError("unplug")]
    source = sources.SerialSource(opener=lambda path, baud: port, clock=lambda: 102)
    with caplog.at_level("INFO"):
        source.read_port("/dev/ttyACM0", threading.Event())
    assert f"sensor Desk ({CHIP}) attached on /dev/ttyACM0" in caplog.text
    assert f"sensor Desk ({CHIP}) detached from /dev/ttyACM0" in caplog.text
    assert source.sensors == {}


def test_serial_open_deasserts_reset_lines_and_never_writes(monkeypatch):
    events = []

    class Port:
        def __init__(self, **kwargs):
            self.dtr = self.rts = True
            self.port = ""
            events.append(kwargs)

        def open(self):
            events.append((self.dtr, self.rts, self.port))

    monkeypatch.setattr(sources.importlib, "import_module", lambda name: SimpleNamespace(Serial=Port))
    sources.open_serial("fake-port", 115200)
    assert events == [
        {"port": None, "baudrate": 115200, "timeout": 0.2, "write_timeout": 2, "exclusive": True},
        (False, False, "fake-port"),
    ]


def test_usb_discovery_includes_native_usb_and_uart_bridges(monkeypatch):
    module = SimpleNamespace(
        comports=lambda: [
            SimpleNamespace(device="ACM", vid=0x303A),
            SimpleNamespace(device="USB", vid=0x10C4),
            SimpleNamespace(device="other", vid=0x1234),
        ]
    )
    monkeypatch.setattr(sources.importlib, "import_module", lambda name: module)
    assert sources.serial_ports() == ["ACM", "USB"]


def test_replay_routes_multiple_named_sensors_without_real_serial(tmp_path):
    path = tmp_path / "replay.jsonl"
    lines = [hello(), report(), hello(chip="111111111111", name="Desk"), report(chip="111111111111")]
    path.write_text("".join(json.dumps({"received_at": 102, "line": line.decode()}) + "\n" for line in lines))
    seen = []
    source = sources.ReplaySource(path)
    asyncio.run(source.run(seen.append, asyncio.Event()))
    assert [(x.adapter, x.sensor_name, x.identifier) for x in seen] == [
        (CHIP, CHIP, ADDR),
        ("111111111111", "Desk", ADDR),
    ]
    assert source.stats["invalid_lines"] == 0
    assert {item.backend for item in seen} == {"replay"}
    assert {item.scan_mode for item in seen} == {"replay"}


def test_firmware_golden_vectors_match_python_crc_and_fields():
    h = parse(b"BLE1|HELLO|AABBCCDDEEFF|1234ABCD|123456|646f6f72|60BF9CC2\n")
    r = parse(b"BLE1|ADV|AABBCCDDEEFF|1234ABCD|1|123456|AA:BB:CC:DD:EE:FF|1|-60|0|020106|5050C1F7\n")
    assert h == Hello(CHIP, "1234ABCD", 123456, "door")
    assert isinstance(r, Report)
    assert (r.counter, r.uptime_us, r.payload) == (1, 123456, b"\x02\x01\x06")
    assert frame(["NAME1", "646f6f72"]) == b"NAME1|646f6f72|B9E21551\n"


def test_unplug_reconnect_waits_for_new_hello_and_resumes_reports():
    async def run():
        first = create_autospec(sources.SerialPort, instance=True)
        first.read.side_effect = [hello() + report(), OSError("unplug")]
        second = create_autospec(sources.SerialPort, instance=True)
        second.read.side_effect = [hello(boot="05060708") + report(boot="05060708"), OSError("unplug")]
        ports = iter([first, second])
        source = sources.SerialSource(ports=["fake"], opener=lambda path, baud: next(ports), discover_interval_s=0.01)
        seen, stop = [], asyncio.Event()

        def emit(item):
            seen.append(item)
            if len(seen) == 2:
                stop.set()

        await asyncio.wait_for(source.run(emit, stop), timeout=2)
        assert [x.boot_id for x in seen] == [BOOT, "05060708"]
        assert source.stats["disconnects"] >= 2
        first.write.assert_not_called()
        second.write.assert_not_called()

    asyncio.run(run())


def test_explicit_sensor_name_is_checksummed_and_confirmed_before_success():
    port = create_autospec(sources.SerialPort, instance=True)
    wire = frame(["NAME1", b"Desk".hex()])
    port.write.return_value = len(wire)
    port.read.return_value = hello(name="Desk")
    sources.set_sensor_name("fake", "Desk", opener=lambda path, baud: port)
    port.write.assert_called_once_with(wire)
    port.close.assert_called_once_with()


def test_live_serial_loss_reorder_and_malformed_counts_are_visible_before_disconnect():
    port = create_autospec(sources.SerialPort, instance=True)
    wire = (
        hello()
        + report(counter=1)
        + report(counter=3, uptime=3000000, payload=b"\x05\x09x")
        + report(counter=3, uptime=3000000)
    )
    responses = iter([wire, b""])
    stop, snapshots = threading.Event(), []
    source = sources.SerialSource(opener=lambda path, baud: port, clock=lambda: 104)

    def read(size):
        data = next(responses)
        if not data:
            snapshots.append(dict(source.stats))
            stop.set()
        return data

    port.read.side_effect = read
    source.read_port("fake-port", stop)
    assert snapshots[0]["lost_reports"] == 1
    assert snapshots[0]["reordered_reports"] == 1
    assert snapshots[0]["malformed_payloads"] == 1
    assert source.stats["lost_reports"] == 1
    assert source.stats["reordered_reports"] == 1
    assert source.stats["malformed_payloads"] == 1
    assert source.stats["disconnects"] == 0


@pytest.mark.parametrize("bad_row", [[], {"line": None, "received_at": 102}, {"line": 7, "received_at": 102}])
def test_malformed_replay_row_is_counted_and_does_not_hide_later_valid_reports(tmp_path, bad_row):
    path = tmp_path / "replay.jsonl"
    valid = [
        {"received_at": 102, "line": hello().decode()},
        {"received_at": 102, "line": report().decode()},
    ]
    path.write_text(json.dumps(bad_row) + "\n" + "".join(json.dumps(row) + "\n" for row in valid))
    source = sources.ReplaySource(path)
    seen = []
    asyncio.run(source.run(seen.append, asyncio.Event()))
    assert source.stats["invalid_lines"] == 1
    assert len(seen) == 1
    assert seen[0].identifier == ADDR


def test_autodetection_rotates_past_non_sensor_without_exceeding_reader_cap():
    async def run():
        opened, closed, seen = [], [], []
        active, peak = [0], [0]
        idle = threading.Event()

        class Port:
            def __init__(self, path):
                self.path = path
                self.frames = iter([hello() + report()] if path == "b-sensor" else [])
                opened.append(path)
                active[0] += 1
                peak[0] = max(peak[0], active[0])

            def read(self, size):
                data = next(self.frames, b"")
                if not data:
                    idle.wait(0.002)
                return data

            def close(self):
                closed.append(self.path)
                active[0] -= 1

            def write(self, data):
                raise AssertionError("acquisition must not write")

        stop = asyncio.Event()
        source = sources.SerialSource(
            max_sensors=1,
            discover=lambda: ["a-non-sensor", "b-sensor"],
            opener=lambda path, baud: Port(path),
            discover_interval_s=0.01,
        )
        source.hello_timeout_s = 0.02

        def emit(item):
            seen.append(item)
            stop.set()

        await asyncio.wait_for(source.run(emit, stop), timeout=2)
        assert opened == ["a-non-sensor", "b-sensor"]
        assert closed == ["a-non-sensor", "b-sensor"]
        assert peak[0] == 1
        assert seen[0].identifier == ADDR
        assert source.stats["hello_timeouts"] == 1

    asyncio.run(run())


def test_sensor_readers_cannot_starve_usb_discovery_or_shutdown(monkeypatch):
    async def run():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        second_discovery = asyncio.Event()
        calls = 0
        original_to_thread = asyncio.to_thread

        def discover():
            return ["fake-sensor"]

        async def observed_to_thread(function, *args, **kwargs):
            nonlocal calls
            if function is discover:
                calls += 1
                if calls == 2:
                    second_discovery.set()
            return await original_to_thread(function, *args, **kwargs)

        class Port:
            def __init__(self):
                self.first, self.closed = True, False
                self.pause = threading.Event()

            def read(self, size):
                if self.first:
                    self.first = False
                    return hello() + report()
                self.pause.wait(timeout=0.01)  # A timeout-bounded serial read, not a production timer.
                return b""

            def close(self):
                self.closed = True

            def write(self, data):
                raise AssertionError("acquisition must not write")

        port = Port()
        monkeypatch.setattr(sources.asyncio, "to_thread", observed_to_thread)
        source = sources.SerialSource(
            max_sensors=1, discover=discover, opener=lambda path, baud: port, discover_interval_s=0.01
        )
        stop = asyncio.Event()
        task = asyncio.create_task(source.run(lambda item: None, stop))
        try:
            await asyncio.wait_for(second_discovery.wait(), timeout=2)
            stop.set()
            await asyncio.wait_for(task, timeout=2)
            assert port.closed
        finally:
            stop.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())
