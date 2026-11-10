"""The Wi-Fi access points this machine hears, as a place signal (part of the spatial-memory goal, #1).

The same set of access points means the same place. Passive only: this reads the scan results the operating system
already has and never connects, probes, or asks for a new scan.

- Linux: NetworkManager's cached list, `nmcli -t -f BSSID,SSID,SIGNAL,FREQ dev wifi list --rescan no`, which needs no
  privileges. `--rescan no` matters: without it nmcli starts a scan when its list is older than 30 s. In terse
  output a field's colons are escaped as `\\:` and backslashes as `\\\\`, so a BSSID reads `AA\\:BB\\:...`. SIGNAL
  is NetworkManager's 0 to 100 quality, not dBm.
- macOS: CoreWLAN's cached scan results (`cachedScanResults`), with RSSI in dBm. CoreWLAN hides BSSIDs from a
  process without Location Services permission; then the scanner logs once and reports nothing.
- Elsewhere, or with no Wi-Fi adapter, or no nmcli: nothing, logged once.

`proximity/network_worker.py` polls this once a minute and writes `wifi.access_point` rows (segment_log.py).
"""

import importlib
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from liberational import logs

LOG = logs.get("network-devices")

NMCLI_ARGS = ("-t", "-f", "BSSID,SSID,SIGNAL,FREQ", "dev", "wifi", "list", "--rescan", "no")
_BSSID = re.compile(r"[0-9a-f]{2}(:[0-9a-f]{2}){5}")


@dataclass(frozen=True)
class AccessPoint:
    bssid: str  # lowercase, colon-separated
    ssid: str | None  # None for a hidden network
    signal: int
    signal_unit: Literal["percent", "dbm"]
    freq_mhz: int


class Scanner(Protocol):
    def scan(self) -> list[AccessPoint] | None:
        """The access points heard now, or None when this machine can't tell (no adapter, tool, or permission)."""
        ...


def _terse_fields(line: str) -> list[str]:
    """One line of `nmcli -t` output split on its unescaped colons, with the escapes removed."""
    fields, current, escaped = [], [], False
    for ch in line:
        if escaped:
            current.append(ch)
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == ":":
            fields.append("".join(current))
            current = []
        else:
            current.append(ch)
    fields.append("".join(current))
    return fields


def _strongest(aps: Iterable[AccessPoint]) -> list[AccessPoint]:
    """One entry per BSSID (two adapters can both list it), the strongest, in BSSID order."""
    best: dict[str, AccessPoint] = {}
    for ap in aps:
        if ap.bssid not in best or ap.signal > best[ap.bssid].signal:
            best[ap.bssid] = ap
    return [best[k] for k in sorted(best)]


def parse_nmcli(text: str) -> list[AccessPoint]:
    """`nmcli -t -f BSSID,SSID,SIGNAL,FREQ dev wifi list` output as access points; malformed lines are skipped."""
    aps = []
    for line in text.splitlines():
        fields = _terse_fields(line)
        if len(fields) != 4:
            continue
        bssid, ssid, signal, freq = fields
        bssid = bssid.strip().lower()
        if not _BSSID.fullmatch(bssid):
            continue
        try:
            strength, mhz = int(signal), int(freq.split()[0])
        except ValueError, IndexError:
            continue
        aps.append(AccessPoint(bssid, ssid if ssid not in ("", "--") else None, strength, "percent", mhz))
    return _strongest(aps)


class NmcliScanner:
    """NetworkManager's cached Wi-Fi list (Linux)."""

    def __init__(self, run: Callable[[list[str]], str] | None = None) -> None:
        self._run = run or _run_nmcli

    def scan(self) -> list[AccessPoint] | None:
        try:
            return parse_nmcli(self._run(["nmcli", *NMCLI_ARGS]))
        except (OSError, subprocess.SubprocessError) as e:
            LOG.debug("nmcli Wi-Fi list failed (%s)", type(e).__name__)
            return None


def _run_nmcli(argv: list[str]) -> str:
    return subprocess.run(argv, capture_output=True, text=True, timeout=10, check=True).stdout


def channel_mhz(band: int, number: int) -> int | None:
    """A CoreWLAN channel's centre frequency: band 1 is 2.4 GHz, 2 is 5 GHz, 3 is 6 GHz (CWChannelBand)."""
    if band == 1:
        return 2484 if number == 14 else 2407 + 5 * number
    if band == 2:
        return 5000 + 5 * number
    if band == 3:
        return 5950 + 5 * number
    return None


class CoreWlanScanner:
    """CoreWLAN's cached scan results (macOS), when this process may see BSSIDs."""

    def __init__(self, corewlan: Any = None) -> None:
        self._corewlan = corewlan

    def scan(self) -> list[AccessPoint] | None:
        if self._corewlan is None:
            try:
                self._corewlan = importlib.import_module("CoreWLAN")
            except ImportError:
                return None
        interface = self._corewlan.CWWiFiClient.sharedWiFiClient().interface()
        if interface is None:
            return None
        aps, hidden = [], 0
        for network in interface.cachedScanResults() or ():
            bssid = network.bssid()
            if not bssid:
                hidden += 1
                continue
            channel = network.wlanChannel()
            mhz = channel_mhz(int(channel.channelBand()), int(channel.channelNumber())) if channel else None
            if mhz is None:
                continue
            ssid = network.ssid()
            aps.append(
                AccessPoint(str(bssid).lower(), str(ssid) if ssid else None, int(network.rssiValue()), "dbm", mhz)
            )
        if hidden and not aps:
            return None  # every BSSID hidden: this process has no Location Services permission
        return _strongest(aps)


class Unavailable:
    def scan(self) -> list[AccessPoint] | None:
        return None


def scanner(platform: str = sys.platform) -> Scanner:
    """This platform's passive scanner."""
    if platform.startswith("linux"):
        return NmcliScanner() if shutil.which("nmcli") else Unavailable()
    if platform == "darwin":
        return CoreWlanScanner()
    return Unavailable()
