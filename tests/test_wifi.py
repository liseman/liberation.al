import subprocess
from types import SimpleNamespace

import pytest

from liberational.proximity import wifi
from liberational.proximity.wifi import AccessPoint, CoreWlanScanner, NmcliScanner, channel_mhz, parse_nmcli

# `nmcli -t -f BSSID,SSID,SIGNAL,FREQ dev wifi list` output in NetworkManager's terse format: colons inside a field
# escaped as `\:` and backslashes as `\\`, a hidden network's SSID empty, and two adapters listing the same AP.
NMCLI = r"""AA\:BB\:CC\:00\:11\:22:Home:72:2437 MHz
AA\:BB\:CC\:00\:11\:23:Home:58:5180 MHz
DE\:AD\:BE\:EF\:00\:01::40:5745 MHz
DE\:AD\:BE\:EF\:00\:02:Café\: 2 \\ up:31:2412 MHz
AA\:BB\:CC\:00\:11\:22:Home:65:2437 MHz
not a line
AA\:BB\:CC\:00\:11:Short:50:2437 MHz
DE\:AD\:BE\:EF\:00\:03:NoSignal::2437 MHz
"""


def test_nmcli_terse_output_unescapes_fields_and_keeps_the_strongest_listing_of_each_ap():
    assert parse_nmcli(NMCLI) == [
        AccessPoint("aa:bb:cc:00:11:22", "Home", 72, "percent", 2437),
        AccessPoint("aa:bb:cc:00:11:23", "Home", 58, "percent", 5180),
        AccessPoint("de:ad:be:ef:00:01", None, 40, "percent", 5745),
        AccessPoint("de:ad:be:ef:00:02", "Café: 2 \\ up", 31, "percent", 2412),
    ]


def test_no_wifi_adapter_lists_nothing():
    assert parse_nmcli("") == []


def test_nmcli_is_asked_for_its_cached_list_and_never_to_rescan():
    calls = []

    def run(argv):
        calls.append(argv)
        return NMCLI

    assert len(NmcliScanner(run).scan() or []) == 4
    assert calls == [["nmcli", "-t", "-f", "BSSID,SSID,SIGNAL,FREQ", "dev", "wifi", "list", "--rescan", "no"]]


@pytest.mark.parametrize(
    "error",
    [FileNotFoundError("nmcli"), subprocess.CalledProcessError(8, "nmcli"), subprocess.TimeoutExpired("nmcli", 10)],
    ids=["missing", "failed", "hung"],
)
def test_an_nmcli_that_fails_means_wifi_is_unknown_not_empty(error):
    def run(argv):
        raise error

    assert NmcliScanner(run).scan() is None


@pytest.mark.parametrize(
    ("band", "number", "mhz"),
    [(1, 1, 2412), (1, 13, 2472), (1, 14, 2484), (2, 36, 5180), (2, 165, 5825), (3, 1, 5955), (0, 6, None)],
)
def test_corewlan_channels_map_to_their_centre_frequency(band, number, mhz):
    assert channel_mhz(band, number) == mhz


def network(bssid, ssid="Home", rssi=-50, band=2, number=36):
    channel = SimpleNamespace(channelBand=lambda: band, channelNumber=lambda: number)
    return SimpleNamespace(bssid=lambda: bssid, ssid=lambda: ssid, rssiValue=lambda: rssi, wlanChannel=lambda: channel)


def corewlan(*networks, interface=True):
    iface = SimpleNamespace(cachedScanResults=lambda: list(networks)) if interface else None
    client = SimpleNamespace(interface=lambda: iface)
    return SimpleNamespace(CWWiFiClient=SimpleNamespace(sharedWiFiClient=lambda: client))


def test_corewlan_cached_results_become_access_points_in_dbm():
    module = corewlan(network("AA:BB:CC:00:11:22"), network("aa:bb:cc:00:11:23", ssid=None, rssi=-71, band=1, number=6))
    assert CoreWlanScanner(module).scan() == [
        AccessPoint("aa:bb:cc:00:11:22", "Home", -50, "dbm", 5180),
        AccessPoint("aa:bb:cc:00:11:23", None, -71, "dbm", 2437),
    ]


def test_corewlan_hiding_every_bssid_means_no_location_permission_not_an_empty_room():
    assert CoreWlanScanner(corewlan(network(None), network(None))).scan() is None


def test_corewlan_with_nothing_cached_hears_nothing():
    assert CoreWlanScanner(corewlan()).scan() == []


def test_a_mac_without_wifi_hardware_is_unknown():
    assert CoreWlanScanner(corewlan(interface=False)).scan() is None


def test_linux_without_nmcli_and_other_platforms_have_no_scanner(monkeypatch):
    monkeypatch.setattr(wifi.shutil, "which", lambda name: None)
    assert wifi.scanner("linux").scan() is None
    assert wifi.scanner("win32").scan() is None
