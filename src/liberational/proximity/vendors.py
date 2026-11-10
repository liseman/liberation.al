"""Who made a device, and what kind it is, from public registries, offline (#3267).

Known › Devices names a device by what it calls itself; most Bluetooth devices call themselves nothing, so the card
says who made it instead. The sources, in order of trust:

1. The address prefix of a globally unique (public) address: the IEEE registry's MA-S (36-bit), MA-M (28-bit), and
   MA-L (24-bit) blocks, longest first. A locally administered address (the second-least-significant bit of the first
   octet set: a phone's private Wi-Fi address) or a Bluetooth random address carries no vendor, so it's skipped, as
   is a CoreBluetooth identifier (a per-app UUID, not an address).
2. A Bluetooth advertisement's manufacturer data: its company identifier, from the Bluetooth SIG's list.
3. Its advertised 16-bit service UUIDs: a member UUID names the company that registered it, and a SIG-defined service
   (Heart Rate, Glucose) says what kind of device it is.

A network device the router reported a vendor for (UniFi does) keeps that when the prefix says nothing.

The registries are files that `mise run models` fetches ([models.files.oui_*] and [models.files.bt_*] in
config/defaults.toml); nothing here touches the network. A missing file means fewer vendors, never an error. Where the
IEEE files weren't fetched, Debian and Ubuntu's ieee-data package (/usr/share/ieee-data) stands in for them.

Each process reads the files once, on its first lookup (about a tenth of a second); a refetch shows after a restart.
"""

import csv
import functools
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from liberational import config, logs

LOG = logs.get("devices")

OUI_FILES = ("oui_ma_s", "oui_ma_m", "oui_ma_l")  # [models.files.<name>]: IEEE CSVs, MA-S, MA-M, and MA-L
SYSTEM_OUI_DIR = Path("/usr/share/ieee-data")  # Debian and Ubuntu's ieee-data package, the same CSVs
BT_COMPANIES = "bt_company_ids"  # [models.files.<name>]: the Bluetooth SIG's company identifiers (YAML)
BT_MEMBER_UUIDS = "bt_member_uuids"  # 16-bit service UUIDs registered to member companies (YAML)
BT_SERVICE_UUIDS = "bt_service_uuids"  # 16-bit service UUIDs the SIG defines (YAML)

# Blocks the IEEE assigned to itself to subdivide (into MA-M and MA-S) or kept private: they name no maker.
_NOT_A_MAKER = {"ieee registration authority", "private"}
# Services nearly every device has: they say nothing about what it is.
_GENERIC_SERVICES = {0x1800, 0x1801, 0x180A, 0x180F, 0x1805, 0x1804}
_BASE_UUID = re.compile(r"0000([0-9a-f]{4})-0000-1000-8000-00805f9b34fb")
APPLE = 0x004C
# Apple's Continuity advertisements start with a type byte that says what kind of device sent it (from public research
# into the protocol: furiousMAC's continuity dissector). Only the types that name a device are here.
APPLE_KINDS = {
    0x02: "Beacon",
    0x06: "HomeKit accessory",
    0x07: "AirPods or Beats",
    0x09: "AirPlay speaker or TV",
    0x0B: "Apple Watch",
    0x0C: "iPhone, iPad, or Mac",
    0x10: "iPhone, iPad, Mac, or Watch",
    0x12: "Find My device",  # an AirTag, or an Apple device away from its owner
}
# Legal suffixes left off a maker's name on a card ("Apple, Inc." reads as Apple).
_SUFFIX = re.compile(
    r"[\s,.]+(inc|incorporated|corp|corporation|co|company|ltd|limited|llc|l\.l\.c|gmbh|ag|s\.?a|b\.?v|oy|ab|a/s|as|"
    r"plc|pte|pty|k\.?k|s\.?r\.?l|s\.?p\.?a|s\.?a\.?s|sp\. z o\.o)\.?$",
    re.IGNORECASE,
)


def short(name: str) -> str:
    """A maker's name without its legal suffixes: "Samsung Electronics Co.,Ltd" is Samsung Electronics."""
    text = " ".join(name.split())
    while (cut := _SUFFIX.sub("", text)) != text and cut:
        text = cut
    return text.rstrip(" ,.") or name.strip()


def mac_hex(value: object) -> str | None:
    """An address's 12 hex digits, upper case, or None for anything that isn't a 48-bit address."""
    digits = re.sub(r"[:\-.]", "", str(value or "")).upper()
    return digits if re.fullmatch(r"[0-9A-F]{12}", digits) else None


def locally_administered(digits: str) -> bool:
    """Whether an address (12 hex digits) was set locally rather than assigned by the IEEE: the second-least-
    significant bit of the first octet. A phone's private Wi-Fi address is one; so is every BLE random static or
    private address (their top two bits are 11 or 01, so the bit can be either; `address_type` says which)."""
    return bool(int(digits[:2], 16) & 0x02)


@dataclass
class Registry:
    """The registries, read: address prefixes by length in hex digits, and the Bluetooth SIG's numbers."""

    prefixes: dict[int, dict[str, str]] = field(default_factory=dict)  # {9: MA-S, 7: MA-M, 6: MA-L}: {prefix: maker}
    companies: dict[int, str] = field(default_factory=dict)  # company identifier: maker
    members: dict[int, str] = field(default_factory=dict)  # member service UUID: maker
    services: dict[int, str] = field(default_factory=dict)  # SIG service UUID: what it is

    def maker(self, digits: str) -> str | None:
        """The maker the IEEE assigned this address's block to, longest block first."""
        for size in sorted(self.prefixes, reverse=True):
            found = self.prefixes[size].get(digits[:size])
            if found:
                return found
        return None


def read_oui_csv(path: Path, into: dict[int, dict[str, str]]) -> int:
    """Add an IEEE registry CSV's assignments (Registry, Assignment, Organization Name, ...) to `into`, by length.
    Returns how many were added."""
    added = 0
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            prefix = (row.get("Assignment") or "").strip().upper()
            name = (row.get("Organization Name") or "").strip()
            if not re.fullmatch(r"[0-9A-F]{6,9}", prefix) or not name or name.casefold() in _NOT_A_MAKER:
                continue
            into.setdefault(len(prefix), {})[prefix] = short(name)
            added += 1
    return added


_YAML_NUMBER = re.compile(r"^\s*-?\s*(value|uuid):\s*(0x[0-9a-fA-F]+|\d+)\s*$")
_YAML_NAME = re.compile(r"^\s*-?\s*name:\s*(.+?)\s*$")


def read_sig_yaml(path: Path) -> dict[int, str]:
    """A Bluetooth SIG assigned-numbers YAML file (company_identifiers.yaml, member_uuids.yaml, service_uuids.yaml):
    a list of entries, each a `value:` or `uuid:` number and a `name:`. Read line by line: the files are flat."""
    out: dict[int, str] = {}
    number: int | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if m := _YAML_NUMBER.match(line):
            number = int(m[2], 0)
        elif (m := _YAML_NAME.match(line)) and number is not None:
            name = m[1]
            if len(name) >= 2 and name[0] == name[-1] and name[0] in "'\"":
                name = name[1:-1].replace("''", "'") if name[0] == "'" else name[1:-1]
            out[number] = name
            number = None
    return out


def _model_file(name: str) -> Path | None:
    try:
        return config.model_path(name)
    except KeyError:
        return None


def _oui_paths() -> list[Path]:
    """The IEEE CSVs: each one `mise run models` fetched, else the ieee-data package's file of the same name."""
    out = []
    for key in OUI_FILES:
        fetched = _model_file(key)
        if fetched is not None:
            out.append(fetched if fetched.exists() else SYSTEM_OUI_DIR / fetched.name)
    return out


def load(
    oui: Iterable[Path] | None = None,
    companies: Path | None = None,
    members: Path | None = None,
    services: Path | None = None,
) -> Registry:
    """Read the registries from the given files, or else the ones `mise run models` fetched (and the ieee-data
    package's for an IEEE file that wasn't). A file that's missing or unreadable is skipped, with a warning."""
    reg = Registry()
    for path in oui if oui is not None else _oui_paths():
        try:
            read_oui_csv(path, reg.prefixes)
        except OSError as error:
            LOG.warning("address prefixes not read from %s: %s (mise run models fetches them)", path, error.strerror)
    for path, table, attr in (
        (companies, BT_COMPANIES, "companies"),
        (members, BT_MEMBER_UUIDS, "members"),
        (services, BT_SERVICE_UUIDS, "services"),
    ):
        file = path if path is not None else _model_file(table)
        if file is None:
            continue
        try:
            setattr(reg, attr, read_sig_yaml(file))
        except OSError as error:
            LOG.warning("Bluetooth numbers not read from %s: %s (mise run models fetches them)", file, error.strerror)
    return reg


@functools.cache
def registry() -> Registry:
    """The registries this process reads, once."""
    reg = load()
    LOG.info(
        "device registries: %d address blocks, %d Bluetooth companies, %d service UUIDs",
        sum(len(t) for t in reg.prefixes.values()),
        len(reg.companies),
        len(reg.members) + len(reg.services),
    )
    return reg


def uuid16(value: object) -> int | None:
    """A 16-bit service UUID from its short form ("fe07", "0xFE07") or the Bluetooth base UUID's long form, else None
    (a vendor's own 128-bit UUID says nothing here)."""
    text = str(value or "").strip().lower().removeprefix("0x")
    if re.fullmatch(r"[0-9a-f]{4}", text):
        return int(text, 16)
    m = _BASE_UUID.fullmatch(text)
    return int(m[1], 16) if m else None


def _company_ids(record: Mapping[str, Any]) -> list[int]:
    out = []
    for key in record.get("manufacturer_data") or {}:
        try:
            company = int(key)
        except TypeError, ValueError:
            continue
        if 0 <= company <= 0xFFFF:
            out.append(company)
    return out


def _uuids(record: Mapping[str, Any]) -> list[int]:
    found = [*(record.get("service_uuids") or []), *(record.get("service_data") or {})]
    return [u for u in (uuid16(v) for v in found) if u is not None]


def bluetooth(record: Mapping[str, Any], reg: Registry | None = None) -> tuple[str | None, str | None]:
    """(maker, kind) for a Bluetooth sighting (proximity/ble.py's record), either or both None when nothing says."""
    reg = reg if reg is not None else registry()
    vendor: str | None = None
    digits = mac_hex(record.get("identifier")) if record.get("identifier_kind") == "bluetooth_address" else None
    if digits is not None and record.get("address_type") == "public" and not locally_administered(digits):
        vendor = reg.maker(digits)
    companies = _company_ids(record)
    uuids = _uuids(record)
    if vendor is None:
        vendor = next((short(reg.companies[c]) for c in companies if c in reg.companies), None)
    if vendor is None:
        vendor = next((short(reg.members[u]) for u in uuids if u in reg.members), None)
    kind: str | None = None
    apple = (record.get("manufacturer_data") or {}).get(str(APPLE))
    if isinstance(apple, str) and len(apple) >= 2:
        try:
            kind = APPLE_KINDS.get(int(apple[:2], 16))
        except ValueError:
            kind = None
    if kind is None:
        kind = next((reg.services[u] for u in uuids if u in reg.services and u not in _GENERIC_SERVICES), None)
    return vendor, kind


def network(record: Mapping[str, Any], reg: Registry | None = None) -> str | None:
    """The maker of a network device (proximity/network.py's observation): its address's IEEE block for a globally
    unique address, else what the router reported, else None."""
    reg = reg if reg is not None else registry()
    digits = mac_hex(record.get("mac") or record.get("identifier"))
    if digits is not None and not locally_administered(digits):
        found = reg.maker(digits)
        if found:
            return found
    reported = record.get("vendor")
    return short(reported) if isinstance(reported, str) and reported.strip() else None
