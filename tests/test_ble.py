from dataclasses import replace

import pytest

from liberational.proximity import ble
from liberational.proximity.ble import Advertisement, Aggregator


def ad(at=100.0, identifier="AA:BB:CC:DD:EE:01", **kwargs):
    return Advertisement(at, "esp32", "sensor-a", "bluetooth_address", identifier, **kwargs)


def aggregate(identifier="AA:BB:CC:DD:EE:01", at=100.0, **kwargs):
    a = Aggregator()
    a.add(ad(at, identifier, **kwargs))
    a.add(ad(at + 1, identifier, **kwargs))
    return a.drain(at + 5, 1)[0]


def test_native_backend_truncation_remains_visible_after_aggregation():
    result = aggregate(source_payload_truncated=True, manufacturer_data={42: b"abcd"})
    assert result["payload_truncated"] is True


PAYLOAD = {42: b"\x01\x23\x45\x67\x89\xab\xcd\xef"}


def test_repeated_reports_keep_rssi_statistics_without_inventing_unknown_samples():
    a = Aggregator(window_s=5)
    a.add(ad(rssi=-70))
    a.add(ad(101, rssi=None))
    a.add(ad(102, rssi=-40))
    assert a.drain(104.999, 5) == []
    result = a.drain(105, 5)
    assert len(result) == 1
    r = result[0]
    assert (r["start"], r["end"], r["reports"]) == (100, 102, 3)
    assert r["rssi"] == {"samples": 2, "min": -70, "max": -40, "mean": -55}
    assert r["local_name"] is None
    assert a.drain(110, 5) == []


@pytest.mark.parametrize("rssi", [None, 127, -128, True])
def test_unavailable_rssi_is_unknown(rssi):
    r = aggregate(rssi=rssi)
    assert r["rssi"] == {"samples": 0, "min": None, "max": None, "mean": None}


def test_same_native_string_with_different_identifier_kind_observer_or_address_type_stays_distinct():
    a = Aggregator()
    source = ad()
    a.add(source)
    a.add(replace(source, identifier_kind="corebluetooth_uuid"))
    a.add(replace(source, adapter="sensor-b"))
    a.add(replace(source, address_type="random"))
    assert len(a.drain(105, 10)) == 4


def test_capacity_and_drain_limits_are_explicit_and_do_not_evict_accepted_sightings():
    a = Aggregator(max_devices=2)
    a.add(ad())
    a.add(ad(identifier="AA:BB:CC:DD:EE:02"))
    a.add(ad(identifier="AA:BB:CC:DD:EE:03"))
    a.add(ad(101))
    assert a.stats["dropped_capacity"] == 1
    assert a.stats["pending_devices"] == 2
    first = a.drain(105, 1)
    assert first[0]["identifier"] == "AA:BB:CC:DD:EE:01"
    assert first[0]["reports"] == 2
    assert len(a.drain(105, 1)) == 1
    a.add(ad(identifier="AA:BB:CC:DD:EE:03"))
    assert a.drain(100, 0, force=True) == []
    assert a.drain(100, 1, force=True)[0]["identifier"] == "AA:BB:CC:DD:EE:03"


def test_payload_snapshot_is_immutable_and_aggregation_truncation_is_visible():
    data = {42: b"0123456789"}
    source = ad(manufacturer_data=data)
    data[42] = b"different"
    a = Aggregator(max_payload_bytes=6)
    a.add(source)
    r = a.drain(105, 1)[0]
    assert r["manufacturer_data"] == {"42": "30313233"}
    assert r["payload_truncated"]
    assert a.stats["truncated_reports"] == 1


def test_observation_bounds_name_uuid_lists_and_raw_bytes_before_queueing():
    source = ad(local_name="x" * 1000, service_uuids=tuple("uuid" for _ in range(1000)), raw_payload=b"x" * 100000)
    assert source.local_name == "x" * 256
    assert len(source.service_uuids) == 64
    assert len(source.raw_payload or b"") <= 65536
    assert source.payload_truncated


def test_latest_payload_and_sensor_clock_ranges_survive_out_of_order_reports():
    a = Aggregator()
    a.add(
        ad(
            102,
            manufacturer_data=PAYLOAD,
            raw_payload=b"new",
            rssi=-60,
            device_counter=9,
            device_uptime_us=200,
            received_at=103,
            clock_offset_s=100,
            boot_id="boot-a",
            channel=38,
            sensor_name="Desk",
            scan_mode="esp32_passive",
        )
    )
    a.add(
        ad(
            100,
            manufacturer_data={42: b"old"},
            raw_payload=b"old",
            rssi=-80,
            device_counter=7,
            device_uptime_us=100,
            received_at=101,
            clock_offset_s=99,
            boot_id="boot-a",
            channel=37,
            scan_mode="unknown",
        )
    )
    r = a.drain(105, 1)[0]
    assert r["manufacturer_data"] == {"42": "0123456789abcdef"}
    assert (r["start"], r["end"], r["device_counter"], r["device_uptime_us"]) == (100, 102, 9, 200)
    assert (r["received_at_start"], r["received_at_end"]) == (101, 103)
    assert (r["clock_offset_s_min"], r["clock_offset_s_max"]) == (99, 100)
    assert (r["device_uptime_us_min"], r["device_uptime_us_max"]) == (100, 200)
    assert r["boot_ids"] == ["boot-a"]
    assert r["channels"] == [37, 38]
    assert r["raw_payload"] == "6e6577"
    assert r["sensor_name"] == "Desk"
    assert r["scan_mode"] == "mixed"
    assert r["scan_modes"] == ["esp32_passive", "unknown"]


def test_multiple_boots_are_reported_as_ambiguous_clock_epoch():
    a = Aggregator()
    a.add(ad(100, boot_id="boot-a"))
    a.add(ad(101, boot_id="boot-b"))
    r = a.drain(105, 1)[0]
    assert r["boot_id"] is None
    assert r["boot_ids"] == ["boot-a", "boot-b"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"observed_at": float("nan")},
        {"identifier": "x" * 257},
        {"identifier_kind": "guessed_person"},
        {"device_counter": -1},
        {"device_uptime_us": 2**64},
        {"channel": 40},
        {"clock_offset_s": float("inf")},
    ],
)
def test_invalid_observation_metadata_is_rejected(kwargs):
    fields = {
        "observed_at": 100,
        "backend": "esp32",
        "adapter": "sensor-a",
        "identifier_kind": "bluetooth_address",
        "identifier": "AA:BB:CC:DD:EE:01",
    }
    with pytest.raises(ValueError):
        Advertisement(**(fields | kwargs))


def test_a_payload_fingerprint_hashes_the_decoded_payload_not_its_hex_spelling():
    record = aggregate(manufacturer_data=PAYLOAD)
    assert ble.payload_fingerprint(record) == ble.payload_fingerprint(
        record | {"manufacturer_data": {"42": "0123456789ABCDEF"}}
    )
    assert ble.payload_fingerprint(record | {"payload_truncated": True}) is None
    assert ble.payload_fingerprint(record | {"manufacturer_data": {"42": "0000000000000000"}}) is None
