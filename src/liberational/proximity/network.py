"""Native network observations and coverage-aware presence transitions, never people or physical whereabouts.

A controller's complete connected-client list can bound an observed connection transition. ARP caches and DHCP
leases are positive, partial evidence only: their absence or expiry never proves a device disconnected. Every
source preserves its own native identifier; a MAC or controller UUID is not a canonical physical-device identity.
"""

import ipaddress
import math
import re
from dataclasses import dataclass


def mac_address(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    raw = value.replace("-", ":").upper()
    return raw if re.fullmatch(r"(?:[0-9A-F]{2}:){5}[0-9A-F]{2}", raw) else None


def ip_address(value: object) -> str | None:
    try:
        return str(ipaddress.ip_address(str(value))) if value is not None else None
    except ValueError:
        return None


@dataclass(frozen=True)
class NetworkDevice:
    identifier_kind: str
    identifier: str
    mac: str | None = None
    mac_randomized: bool | None = None
    hostname: str | None = None
    display_name: str | None = None
    ip: str | None = None
    vendor: str | None = None
    access_point_id: str | None = None
    access_point_name: str | None = None
    first_seen: float | None = None
    last_seen: float | None = None
    connected_at: float | None = None
    disconnected_at: float | None = None
    connection_type: str | None = None
    connected: bool | None = None
    lease_expires_at: float | None = None

    def __post_init__(self) -> None:
        if (
            self.identifier_kind not in ("mac_address", "unifi_client_id")
            or not self.identifier
            or len(self.identifier) > 128
        ):
            raise ValueError("invalid native network identifier")
        for value in (
            self.hostname,
            self.display_name,
            self.vendor,
            self.access_point_id,
            self.access_point_name,
            self.connection_type,
        ):
            if value is not None and len(value.encode()) > 512:
                raise ValueError("network metadata field exceeds 512 bytes")
        for value in (self.first_seen, self.last_seen, self.connected_at, self.disconnected_at, self.lease_expires_at):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("network timestamp must be a finite epoch time or unknown")

    @property
    def key(self) -> tuple[str, str]:
        return self.identifier_kind, self.identifier

    @property
    def locally_administered(self) -> bool | None:
        """The MAC U/L bit, not proof that an address is randomized or rotating."""
        value = mac_address(self.mac)
        return bool(int(value[:2], 16) & 2) if value else None


@dataclass(frozen=True)
class NetworkInventory:
    source: str
    observer: str
    observed_at: float
    devices: tuple[NetworkDevice, ...]
    complete: bool = False
    authoritative: bool = False
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.source or not self.observer or len(self.observer) > 128:
            raise ValueError("invalid network observer")
        if not math.isfinite(self.observed_at) or self.observed_at < 0:
            raise ValueError("inventory time must be finite")
        if len(self.devices) > 2048 or len(self.warnings) > 32:
            raise ValueError("network inventory exceeds hard bounds")
