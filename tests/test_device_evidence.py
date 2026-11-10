"""Live's linked-device evidence: synthetic broker records, no live links or network access."""

import time

from liberational.dashboard import device_links
from liberational.proximity import devices
from test_devices import PHONE, WATCH, home, http_, network, sighting  # noqa: F401 # shared temporary broker fixtures


def test_live_read_api_and_ask_share_recent_network_and_bluetooth_evidence(http_, home):  # noqa: F811
    reg, record = home
    now = time.time()
    reg.save("net_mac", PHONE, "person", "Alice")
    reg.save("ble_identifier", WATCH, "person", "Alice")
    record(devices.NET_OBSERVATIONS, [network(now - 30, PHONE)])
    record(devices.BLE_SIGHTINGS, [sighting(now - 10, WATCH)])
    code, body = http_("GET", "/device-presence")
    assert code == 200
    assert len(body["people"]) == 1
    person = body["people"][0]
    assert person["name"] == "Alice" and person["status"] == "recent"
    assert person["source"] == devices.BLE_SIGHTINGS and person["links"] == 2
    assert "not proof" in body["caveat"]
    state = reg.target("Alice", "person", body["start"], body["at"])
    assert state is not None
    assert devices.home_status(state, body["at"], body["gap_seconds"]) == person["status"]


def test_live_device_evidence_failure_is_unavailable_instead_of_zero_people(monkeypatch):
    def unavailable():
        raise RuntimeError("synthetic unavailable store")

    monkeypatch.setattr(devices, "at", unavailable)
    code, body = device_links.presence(now=1000)
    assert code == 503 and "error" in body
    assert "people" not in body


def test_live_device_evidence_no_links_is_an_explicit_setup_state(http_):  # noqa: F811
    code, body = http_("GET", "/device-presence")
    assert code == 200 and body["people"] == []
