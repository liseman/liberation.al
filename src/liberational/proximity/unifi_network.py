"""Read connected-client inventories from a configured local UniFi controller, without probing devices.

Integration v1 is documented at https://developer.ui.com/network/v10.6.106/openapi.json. Its offset pagination
has no snapshot token, so complete means every consistent page was read, not an atomic network-wide observation.
Legacy stat/sta is a compatibility API: https://raw.githubusercontent.com/Kane610/aiounifi/master/aiounifi/models/client.py
Controller timestamps and aliases remain distinct from this poll's observation time and actual hostnames.
"""

import datetime as dt
import math
import time
import uuid
from typing import Any

from liberational.proximity.network import NetworkDevice, NetworkInventory, ip_address, mac_address
from liberational.unifi import UniFiError, UniFiSession


def _text(value: object) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str) or len(value) > 512 or len(value.encode()) > 512:
        raise ValueError("invalid controller text field")
    return value


def _epoch(value: object, *, iso: bool = False) -> float | None:
    if value is None or value == "" or value == 0:
        return None
    if isinstance(value, bool):
        raise ValueError("invalid controller timestamp")
    if iso:
        if not isinstance(value, str) or len(value) > 64:
            raise ValueError("invalid controller timestamp")
        parsed = dt.datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            raise ValueError("controller timestamp needs a timezone")
        result = parsed.timestamp()
    elif isinstance(value, int | float):
        result = float(value)
    else:
        raise ValueError("invalid controller timestamp")
    if not math.isfinite(result) or result < 0:
        raise ValueError("invalid controller timestamp")
    return result


def _device(row: dict[str, Any], integration: bool) -> NetworkDevice:
    if integration:
        if not isinstance(row.get("id"), str):
            raise ValueError("invalid controller client identifier")
        client_id = str(uuid.UUID(row["id"]))
        kind = _text(row.get("type"))
        if kind is None:
            raise ValueError("missing client type")
        mac = mac_address(row.get("macAddress")) if kind in ("WIRED", "WIRELESS") else None
        if kind in ("WIRED", "WIRELESS") and mac is None:
            raise ValueError("physical client needs a valid MAC")
        raw_ip = row.get("ipAddress")
        ip = ip_address(raw_ip)
        if raw_ip not in (None, "") and ip is None:
            raise ValueError("invalid client IP")
        return NetworkDevice(
            identifier_kind="mac_address" if mac is not None else "unifi_client_id",
            identifier=mac or client_id,
            mac=mac,
            display_name=_text(row.get("name")),
            hostname=None,
            ip=ip,
            access_point_id=_text(row.get("uplinkDeviceId")) if kind == "WIRELESS" else None,
            connected_at=_epoch(row.get("connectedAt"), iso=True),
            connection_type=kind.lower(),
            connected=True,
        )
    mac = mac_address(row.get("mac"))
    if mac is None:
        raise ValueError("legacy client needs a valid MAC")
    raw_ip = row.get("ip")
    ip = ip_address(raw_ip)
    if raw_ip not in (None, "") and ip is None:
        raise ValueError("invalid client IP")
    wired = row.get("is_wired")
    if wired is not None and not isinstance(wired, bool):
        raise ValueError("invalid client connection type")
    return NetworkDevice(
        identifier_kind="mac_address",
        identifier=mac,
        mac=mac,
        hostname=_text(row.get("hostname")),
        display_name=_text(row.get("name")),
        ip=ip,
        vendor=_text(row.get("oui")),
        access_point_id=mac_address(row.get("ap_mac")) if wired is False else None,
        access_point_name=_text(row.get("ap_name")) if wired is False else None,
        first_seen=_epoch(row.get("first_seen")),
        last_seen=_epoch(row.get("last_seen")),
        connected_at=_epoch(row.get("latest_assoc_time") or row.get("assoc_time")),
        disconnected_at=_epoch(row.get("disconnect_timestamp")),
        connection_type="wired" if wired is True else "wireless" if wired is False else None,
        connected=True,
    )


class UniFiNetwork:
    """Bounded best-effort poll; malformed or incomplete replies never authorize absence transitions."""

    def __init__(
        self,
        session: UniFiSession,
        site: str = "default",
        *,
        mode: str = "legacy",
        observer: str = "local-network",
        max_devices: int = 1024,
        max_pages: int = 16,
        timeout_s: float = 10.0,
    ) -> None:
        if mode not in ("legacy", "integration") or not 1 <= max_devices <= 2048 or not 1 <= max_pages <= 32:
            raise ValueError("invalid UniFi inventory settings")
        if not math.isfinite(timeout_s) or not 0 < timeout_s <= 120 or not site or len(site) > 128:
            raise ValueError("invalid UniFi poll bounds")
        if not observer or len(observer) > 128:
            raise ValueError("invalid UniFi observer")
        self.session, self.site, self.mode, self.observer = session, site, mode, observer
        self.max_devices, self.max_pages, self.timeout_s = max_devices, max_pages, timeout_s

    def _pages(self, *, deadline: float, site_uuid: str | None = None) -> tuple[list[dict[str, Any]], bool]:
        rows: list[dict[str, Any]] = []
        total: int | None = None
        offset = 0
        for _ in range(self.max_pages):
            try:
                page = (
                    self.session.integration_sites(offset, 200, deadline=deadline)
                    if site_uuid is None
                    else self.session.integration_clients(site_uuid, offset, 200, deadline=deadline)
                )
            except UniFiError:
                if rows:
                    return rows, False
                raise
            if not isinstance(page, dict) or not isinstance(page.get("data"), list):
                if rows:
                    return rows, False
                raise UniFiError("UniFi pagination envelope is invalid")
            data = page["data"]
            count, echoed_offset, limit, advertised_total = (
                page.get(k) for k in ("count", "offset", "limit", "totalCount")
            )
            valid_fields = all(
                isinstance(v, int) and not isinstance(v, bool) for v in (count, echoed_offset, limit, advertised_total)
            )
            if (
                not valid_fields
                or len(data) > 200
                or any(not isinstance(row, dict) for row in data)
                or count != len(data)
                or echoed_offset != offset
                or not 1 <= limit <= 200
                or count > limit
                or advertised_total < 0
                or offset + count > advertised_total
            ):
                rows.extend(row for row in data[: min(200, self.max_devices - len(rows))] if isinstance(row, dict))
                return rows, False
            if total is not None and advertised_total != total:
                return rows, False
            total = advertised_total
            remaining = self.max_devices - len(rows)
            rows.extend(data[:remaining])
            if len(data) > remaining:
                return rows, False
            offset += count
            if offset == total:
                return rows, True
            if count == 0 or len(rows) >= self.max_devices:
                return rows, False
        return rows, False

    def _site_uuid(self, deadline: float) -> str:
        try:
            return str(uuid.UUID(self.site))
        except ValueError:
            pass
        sites, complete = self._pages(deadline=deadline)
        if not complete:
            raise UniFiError("UniFi site listing is incomplete")
        matching = [row for row in sites if self.site in (row.get("internalReference"), row.get("id"), row.get("name"))]
        if len(matching) != 1:
            raise UniFiError("UniFi site reference is missing or ambiguous")
        try:
            if not isinstance(matching[0].get("id"), str):
                raise ValueError()
            return str(uuid.UUID(matching[0]["id"]))
        except ValueError, KeyError, TypeError:
            raise UniFiError("UniFi site identifier is invalid") from None

    def poll(self, now: float) -> NetworkInventory:
        devices: list[NetworkDevice] = []
        warnings: list[str] = []
        complete = False
        try:
            deadline = time.monotonic() + self.timeout_s
            if self.mode == "integration":
                rows, complete = self._pages(deadline=deadline, site_uuid=self._site_uuid(deadline))
            else:
                envelope = self.session.legacy_clients(self.site, deadline=deadline)
                if (
                    not isinstance(envelope, dict)
                    or not isinstance(envelope.get("meta"), dict)
                    or envelope["meta"].get("rc") != "ok"
                    or not isinstance(envelope.get("data"), list)
                ):
                    raise UniFiError("UniFi legacy client envelope is invalid")
                rows = envelope["data"][: self.max_devices]
                complete = len(envelope["data"]) <= self.max_devices
            seen: set[tuple[str, str]] = set()
            invalid = 0
            for row in rows:
                try:
                    if not isinstance(row, dict):
                        raise ValueError("invalid client row")
                    device = _device(row, self.mode == "integration")
                except ValueError, TypeError, KeyError, OverflowError:
                    invalid += 1
                    continue
                if device.key in seen:
                    invalid += 1
                    continue
                seen.add(device.key)
                devices.append(device)
            if invalid:
                complete = False
                warnings.append(f"{invalid} UniFi client rows were invalid or duplicated; absence is unknown.")
            if not complete:
                warnings.append("The UniFi connected-client listing is incomplete; absence is unknown.")
        except UniFiError as e:
            warnings.append(str(e))
        return NetworkInventory(
            source="unifi",
            observer=self.observer,
            observed_at=now,
            devices=tuple(devices),
            complete=complete,
            authoritative=complete,
            warnings=tuple(warnings),
        )
