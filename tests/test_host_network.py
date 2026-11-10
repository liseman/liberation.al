import datetime as dt
import os
import plistlib
import subprocess
import sys
from unittest.mock import create_autospec

import pytest

from liberational.proximity.host_network import CommandResult, HostNetworkSource, run_command

HEADER = "IP address       HW type     Flags       HW address            Mask     Device\n"
ARP_ROW = "192.0.2.10 0x1 0x2 aa:bb:cc:dd:ee:01 * eth0\n"


def cache(tmp_path, text, name="arp"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_linux_cache_and_expired_lease_merge_without_claiming_presence(tmp_path):
    arp = cache(tmp_path, HEADER + ARP_ROW)
    lease = cache(tmp_path, "946684800 aa:bb:cc:dd:ee:01 192.0.2.10 printer *\n", "dnsmasq.leases")
    inventory = HostNetworkSource("home", arp, [lease]).poll(1800000000)
    assert len(inventory.devices) == 1
    device = inventory.devices[0]
    assert (device.identifier_kind, device.identifier, device.mac, device.ip, device.hostname) == (
        "mac_address",
        "AA:BB:CC:DD:EE:01",
        "AA:BB:CC:DD:EE:01",
        "192.0.2.10",
        "printer",
    )
    assert device.lease_expires_at == 946684800
    assert (device.connected, device.last_seen, device.vendor, device.access_point_id, device.connection_type) == (
        None,
        None,
        None,
        None,
        None,
    )
    assert not inventory.complete and not inventory.authoritative
    assert inventory.warnings
    arp.write_text(HEADER)
    lease.write_text("")
    later = HostNetworkSource("home", arp, [lease]).poll(1800000001)
    assert later.devices == ()
    assert not later.complete and not later.authoritative


def test_unresolved_and_malformed_neighbors_are_unknown_not_zero_mac_devices(tmp_path):
    path = cache(tmp_path, HEADER + ARP_ROW + "192.0.2.11 0x1 0x0 00:00:00:00:00:00 * eth0\nwrong\n")
    result = HostNetworkSource("home", path).poll(1)
    assert [d.identifier for d in result.devices] == ["AA:BB:CC:DD:EE:01"]
    assert any("unresolved" in w for w in result.warnings)
    assert any("malformed" in w for w in result.warnings)


def test_darwin_command_is_only_numeric_cache_display(tmp_path):
    runner = create_autospec(
        run_command,
        return_value=CommandResult(
            b"? (192.0.2.20) at a:b:c:d:e:f on en0 ifscope [ethernet]\n"
            b"? (192.0.2.21) at (incomplete) on en0 ifscope [ethernet]\n",
            0,
        ),
    )
    result = HostNetworkSource("laptop", platform="darwin", runner=runner, which=lambda name: "/synthetic/arp").poll(1)
    runner.assert_called_once_with(["/synthetic/arp", "-an"], timeout=5.0, max_bytes=1048576)
    assert [(d.mac, d.ip, d.hostname, d.connected) for d in result.devices] == [
        ("0A:0B:0C:0D:0E:0F", "192.0.2.20", None, None)
    ]
    assert any("unresolved" in w for w in result.warnings)


@pytest.mark.parametrize(
    "failure", [TimeoutError("timed out"), OSError("no command"), subprocess.TimeoutExpired("arp", 1)]
)
def test_mac_command_failure_leaves_dhcp_positives_available(tmp_path, failure):
    lease = cache(tmp_path, "0 aa:bb:cc:dd:ee:02 192.0.2.22 * *\n", "leases")
    runner = create_autospec(run_command, side_effect=failure)
    result = HostNetworkSource(
        "home", dhcp_paths=[lease], platform="darwin", runner=runner, which=lambda _: "arp"
    ).poll(1)
    assert [(d.mac, d.lease_expires_at) for d in result.devices] == [("AA:BB:CC:DD:EE:02", None)]
    assert any("command failed" in w for w in result.warnings)
    assert not result.complete and not result.authoritative


def test_missing_files_and_unavailable_arp_are_explicit(tmp_path):
    result = HostNetworkSource("home", tmp_path / "absent-arp", [tmp_path / "absent-lease"]).poll(1)
    assert result.devices == ()
    assert any("absent-arp" in w and "could not be read" in w for w in result.warnings)
    assert any("absent-lease" in w and "could not be read" in w for w in result.warnings)
    missing_command = HostNetworkSource("home", platform="darwin", which=lambda _: None).poll(1)
    assert any("command is unavailable" in w for w in missing_command.warnings)
    assert not missing_command.authoritative


def test_isc_latest_declaration_replaces_reassigned_ip_and_dates_are_utc(tmp_path):
    leases = cache(
        tmp_path,
        """
lease 192.0.2.30 { hardware ethernet aa:bb:cc:dd:ee:03; ends never; }
lease 192.0.2.30 {
 hardware ethernet aa:bb:cc:dd:ee:04;
 client-hostname "workstation } # not a comment";
 ends 0 2000/01/02 00:00:00;
 binding state free;
 on expiry { execute("ignored"); }
}
lease 192.0.2.31 { hardware ethernet aa:bb:cc:dd:ee:05; ends epoch 946684800; }
""",
        "dhcpd.leases",
    )
    result = HostNetworkSource("home", dhcp_paths=[leases]).poll(1800000000)
    assert [(d.mac, d.ip, d.hostname, d.lease_expires_at, d.connected) for d in result.devices] == [
        ("AA:BB:CC:DD:EE:04", "192.0.2.30", "workstation } # not a comment", 946771200, None),
        ("AA:BB:CC:DD:EE:05", "192.0.2.31", None, 946684800, None),
    ]


def test_isc_missing_mac_in_latest_declaration_does_not_resurrect_old_owner(tmp_path):
    leases = cache(
        tmp_path,
        """
lease 192.0.2.30 { hardware ethernet aa:bb:cc:dd:ee:03; ends never; }
lease 192.0.2.30 { binding state free; }
""",
        "leases",
    )
    result = HostNetworkSource("home", dhcp_paths=[leases]).poll(1)
    assert result.devices == ()
    assert any("no valid Ethernet" in w for w in result.warnings)


def test_incomplete_isc_tail_preserves_earlier_valid_records(tmp_path):
    leases = cache(
        tmp_path,
        """
lease 192.0.2.30 { hardware ethernet aa:bb:cc:dd:ee:03; ends never; }
lease 192.0.2.31 { hardware ethernet aa:bb:cc:dd:ee:04; client-hostname "unfinished
""",
        "leases",
    )
    result = HostNetworkSource("home", dhcp_paths=[leases]).poll(1)
    assert [d.mac for d in result.devices] == ["AA:BB:CC:DD:EE:03"]
    assert any("unterminated quote" in w for w in result.warnings)


@pytest.mark.parametrize("binary", [False, True])
def test_apple_plist_uses_client_identifier_never_router_mac(tmp_path, binary):
    path = tmp_path / "lease.plist"
    path.write_bytes(
        plistlib.dumps(
            {
                "ClientIdentifier": bytes.fromhex("01aabbccddee06"),
                "IPAddress": "192.0.2.40",
                "LeaseStartDate": dt.datetime(2000, 1, 1),
                "LeaseLength": 3600,
                "RouterHardwareAddress": bytes.fromhex("aabbccddee99"),
            },
            fmt=plistlib.FMT_BINARY if binary else plistlib.FMT_XML,
        )
    )
    result = HostNetworkSource("home", dhcp_paths=[path]).poll(1)
    assert [(d.mac, d.ip, d.lease_expires_at) for d in result.devices] == [
        ("AA:BB:CC:DD:EE:06", "192.0.2.40", 946688400)
    ]
    assert result.devices[0].connected is None


def test_apple_plist_with_only_router_mac_has_no_client_identity(tmp_path):
    path = tmp_path / "lease.plist"
    path.write_bytes(
        plistlib.dumps({"IPAddress": "192.0.2.40", "RouterHardwareAddress": bytes.fromhex("aabbccddee99")})
    )
    result = HostNetworkSource("home", dhcp_paths=[path]).poll(1)
    assert result.devices == ()
    assert any("lacked an Ethernet ClientIdentifier" in w for w in result.warnings)


@pytest.mark.parametrize(
    "text",
    [
        "nan aa:bb:cc:dd:ee:01 192.0.2.10 printer *\n",
        "lease 192.0.2.10 { hardware ethernet aa:bb:cc:dd:ee:01; ends 0 1999/99/99 00:00:00; }",
    ],
)
def test_bad_expiry_remains_unknown_without_losing_native_device(tmp_path, text):
    path = cache(tmp_path, text, "leases")
    result = HostNetworkSource("home", dhcp_paths=[path]).poll(1)
    assert [d.mac for d in result.devices] == ["AA:BB:CC:DD:EE:01"]
    assert result.devices[0].lease_expires_at is None
    assert any("malformed" in w for w in result.warnings)


def test_conflicting_ip_observations_do_not_attach_another_ips_expiry(tmp_path):
    arp = cache(tmp_path, HEADER + ARP_ROW)
    lease = cache(tmp_path, "946684800 aa:bb:cc:dd:ee:01 192.0.2.99 printer *\n", "lease")
    result = HostNetworkSource("home", arp, [lease]).poll(1)
    assert (result.devices[0].ip, result.devices[0].hostname, result.devices[0].lease_expires_at) == (
        "192.0.2.10",
        "printer",
        None,
    )
    assert any("conflicting" in w for w in result.warnings)


def test_byte_row_and_device_limits_report_omissions(tmp_path):
    first = "946684800 aa:bb:cc:dd:ee:01 192.0.2.10 printer *\n"
    second = "946684800 aa:bb:cc:dd:ee:02 192.0.2.11 scanner *\n"
    lease = cache(tmp_path, first + second, "leases")
    by_bytes = HostNetworkSource("home", dhcp_paths=[lease], max_read_bytes=len(first) + 5).poll(1)
    assert [d.ip for d in by_bytes.devices] == ["192.0.2.10"]
    assert any("byte limit" in w for w in by_bytes.warnings)
    by_rows = HostNetworkSource("home", dhcp_paths=[lease], max_rows=1).poll(1)
    assert [d.ip for d in by_rows.devices] == ["192.0.2.10"]
    assert any("row limit" in w for w in by_rows.warnings)
    by_devices = HostNetworkSource("home", dhcp_paths=[lease], max_devices=1).poll(1)
    assert [d.ip for d in by_devices.devices] == ["192.0.2.10"]
    assert any("device limit" in w for w in by_devices.warnings)


def test_special_file_is_not_opened_as_a_blocking_cache(tmp_path):
    fifo = tmp_path / "cache.fifo"
    os.mkfifo(fifo)
    result = HostNetworkSource("home", fifo).poll(1)
    assert result.devices == ()
    assert any("not a regular file" in w for w in result.warnings)


def test_bounded_command_runner_caps_actual_synthetic_subprocess_output(tmp_path):
    script = tmp_path / "emit.py"
    script.write_text("print('x' * 10000)\n")
    result = run_command([sys.executable, str(script)], timeout=5, max_bytes=128)
    assert len(result.stdout) == 128
    assert result.truncated


def test_oversized_injected_command_result_is_also_bounded(tmp_path):
    first = b"? (192.0.2.20) at aa:bb:cc:dd:ee:01 on en0\n"
    runner = create_autospec(run_command, return_value=CommandResult(first + b"x" * 100, 0))
    result = HostNetworkSource(
        "home", platform="darwin", runner=runner, which=lambda _: "arp", max_read_bytes=len(first) + 1
    ).poll(1)
    assert [d.ip for d in result.devices] == ["192.0.2.20"]
    assert any("byte limit" in w for w in result.warnings)


@pytest.mark.parametrize(
    "bound", [{"max_devices": 0}, {"max_read_bytes": 0}, {"max_rows": 0}, {"command_timeout": float("inf")}]
)
def test_invalid_bounds_are_rejected(bound):
    with pytest.raises(ValueError):
        HostNetworkSource("home", **bound)


def test_malformed_apple_xml_is_unknown_coverage_not_a_poll_failure(tmp_path):
    path = tmp_path / "broken.plist"
    path.write_bytes(b'<?xml version="1.0"?><plist><dict><key>IPAddress</key><string>192.0.2.10')
    result = HostNetworkSource("home", dhcp_paths=[path]).poll(1)
    assert result.devices == ()
    assert any("plist could not be decoded" in w for w in result.warnings)
    assert not result.complete and not result.authoritative


def test_failed_arp_command_is_not_interpreted_as_a_valid_snapshot():
    runner = create_autospec(
        run_command, return_value=CommandResult(b"? (192.0.2.20) at aa:bb:cc:dd:ee:01 on en0\n", 1)
    )
    result = HostNetworkSource("home", platform="darwin", runner=runner, which=lambda _: "arp").poll(1)
    assert result.devices == ()
    assert any("exit 1" in w for w in result.warnings)
