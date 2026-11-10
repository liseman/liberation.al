from unittest.mock import create_autospec

import pytest

from liberational.proximity.unifi_network import UniFiNetwork
from liberational.unifi import UniFiError, UniFiSession

SITE = "00000000-0000-0000-0000-000000000010"
CLIENT = "00000000-0000-0000-0000-000000000001"
VPN = "00000000-0000-0000-0000-000000000002"
UPLINK = "00000000-0000-0000-0000-000000000003"
MAC = "02:00:00:00:00:01"
WIRELESS = {
    "id": CLIENT,
    "type": "WIRELESS",
    "name": "Test sensor",
    "macAddress": MAC,
    "ipAddress": "192.0.2.10",
    "uplinkDeviceId": UPLINK,
    "access": {"type": "DEFAULT"},
    "connectedAt": "2026-10-01T10:00:00Z",
}
VIRTUAL = {"id": VPN, "type": "VPN", "name": "Test tunnel", "access": {"type": "DEFAULT"}}


def page(data, offset=0, total=None):
    return {
        "data": data,
        "offset": offset,
        "limit": 200,
        "count": len(data),
        "totalCount": len(data) if total is None else total,
    }


def test_legacy_inventory_preserves_alias_hostname_vendor_and_source_times():
    session = create_autospec(UniFiSession, instance=True)
    session.legacy_clients.return_value = {
        "meta": {"rc": "ok"},
        "data": [
            {
                "mac": MAC.lower(),
                "name": "Test label",
                "hostname": "sensor-a",
                "ip": "192.0.2.10",
                "oui": "Example Instruments",
                "is_wired": False,
                "ap_mac": "02:00:00:00:00:F0",
                "ap_name": "Test AP",
                "first_seen": 100,
                "last_seen": 200,
                "assoc_time": 150,
            }
        ],
    }
    result = UniFiNetwork(session, observer="test-network").poll(300)
    assert result.complete and result.authoritative
    device = result.devices[0]
    assert (device.identifier_kind, device.identifier, device.mac) == ("mac_address", MAC, MAC)
    assert (device.display_name, device.hostname, device.vendor) == ("Test label", "sensor-a", "Example Instruments")
    assert (device.first_seen, device.last_seen, device.connected_at, device.disconnected_at) == (100, 200, 150, None)
    assert (device.access_point_id, device.access_point_name) == ("02:00:00:00:00:F0", "Test AP")
    assert result.observer == "test-network" and result.observed_at == 300


def test_integration_reads_all_pages_and_keeps_vpn_uuid_without_invented_mac():
    session = create_autospec(UniFiSession, instance=True)
    session.integration_clients.side_effect = [page([WIRELESS], total=2), page([VIRTUAL], offset=1, total=2)]
    result = UniFiNetwork(session, SITE, mode="integration").poll(300)
    assert result.complete and result.authoritative
    assert [device.identifier for device in result.devices] == [MAC, VPN]
    assert (result.devices[1].identifier_kind, result.devices[1].mac) == ("unifi_client_id", None)
    assert result.devices[0].hostname is None
    assert result.devices[0].display_name == "Test sensor"
    assert result.devices[0].access_point_id == UPLINK
    assert result.devices[0].connected_at == 1790848800
    assert result.devices[0].last_seen is None and result.devices[0].vendor is None
    assert [call.args[1] for call in session.integration_clients.call_args_list] == [0, 1]
    session.integration_sites.assert_not_called()


def test_short_site_reference_is_resolved_with_paginated_site_inventory():
    session = create_autospec(UniFiSession, instance=True)
    session.integration_sites.return_value = page(
        [{"id": SITE, "internalReference": "default", "name": "Test network"}]
    )
    session.integration_clients.return_value = page([])
    result = UniFiNetwork(session, mode="integration").poll(300)
    assert result.complete and result.devices == ()
    assert session.integration_clients.call_args.args[0] == SITE


@pytest.mark.parametrize(
    "reply",
    [
        {"meta": {"rc": "error", "msg": "api.err.LoginRequired"}, "data": []},
        {"meta": {"rc": "ok"}, "data": None},
    ],
)
def test_failed_legacy_response_is_unknown_not_a_complete_empty_inventory(reply):
    session = create_autospec(UniFiSession, instance=True)
    session.legacy_clients.return_value = reply
    result = UniFiNetwork(session).poll(300)
    assert not result.complete and not result.authoritative
    assert result.warnings


@pytest.mark.parametrize("second", [page([VIRTUAL], offset=1, total=3), UniFiError("UniFi request timed out")])
def test_changed_or_failed_later_page_keeps_positive_rows_without_certifying_absence(second):
    session = create_autospec(UniFiSession, instance=True)
    session.integration_clients.side_effect = [page([WIRELESS], total=2), second]
    result = UniFiNetwork(session, SITE, mode="integration").poll(300)
    assert [device.identifier for device in result.devices] == [MAC]
    assert not result.complete and not result.authoritative


@pytest.mark.parametrize("limit_setting", [{"max_devices": 1}, {"max_pages": 1}])
def test_local_bounds_preserve_unknown_coverage_instead_of_false_disconnects(limit_setting):
    session = create_autospec(UniFiSession, instance=True)
    session.integration_clients.return_value = page([WIRELESS], total=2)
    result = UniFiNetwork(session, SITE, mode="integration", **limit_setting).poll(300)
    assert len(result.devices) == 1
    assert not result.complete and not result.authoritative
    assert session.integration_clients.call_count == 1


@pytest.mark.parametrize("bad_row", [WIRELESS | {"macAddress": "invalid"}, WIRELESS | {"id": 7}, None])
def test_malformed_client_identity_prevents_authoritative_absence(bad_row):
    session = create_autospec(UniFiSession, instance=True)
    session.integration_clients.return_value = page([VIRTUAL, bad_row])
    result = UniFiNetwork(session, SITE, mode="integration").poll(300)
    assert not result.complete and not result.authoritative
    assert result.warnings


def test_duplicate_client_across_offset_pages_is_partial():
    session = create_autospec(UniFiSession, instance=True)
    session.integration_clients.side_effect = [page([WIRELESS], total=2), page([WIRELESS], offset=1, total=2)]
    result = UniFiNetwork(session, SITE, mode="integration").poll(300)
    assert len(result.devices) == 1
    assert not result.complete


def test_wired_switch_is_not_mislabeled_as_an_access_point():
    session = create_autospec(UniFiSession, instance=True)
    session.integration_clients.return_value = page([WIRELESS | {"type": "WIRED"}])
    result = UniFiNetwork(session, SITE, mode="integration").poll(300)
    assert result.complete
    assert result.devices[0].connection_type == "wired"
    assert result.devices[0].access_point_id is None


def test_malformed_page_keeps_valid_positive_rows_but_never_authorizes_absence():
    session = create_autospec(UniFiSession, instance=True)
    session.integration_clients.return_value = page([VIRTUAL, None])
    result = UniFiNetwork(session, SITE, mode="integration").poll(300)
    assert [device.identifier for device in result.devices] == [VPN]
    assert not result.complete and not result.authoritative


def test_vpn_stays_a_native_connection_id_even_if_an_extra_mac_field_is_present():
    session = create_autospec(UniFiSession, instance=True)
    session.integration_clients.return_value = page([VIRTUAL | {"macAddress": MAC}])
    result = UniFiNetwork(session, SITE, mode="integration").poll(300)
    assert result.devices[0].identifier_kind == "unifi_client_id"
    assert result.devices[0].identifier == VPN
    assert result.devices[0].mac is None
