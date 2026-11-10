"""Bounded BLE advertisement aggregation and explicitly unconfirmed payload matches.

Observations retain native address or per-app identifier semantics. An aggregate counts reports received from the
backend, not radio packets: the operating system may coalesce or cache advertisement properties. No absence,
distance, person identity, or device identity across rotating identifiers is inferred here.
"""

import hashlib
import json
import math
import uuid
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

_MAX_FIELDS = 64
_MAX_OBSERVATION_BYTES = 65536
NativeKey = tuple[str, str, str, str, str | None]


def _text(value: str, limit: int) -> tuple[str, bool]:
    raw = value[:limit].encode("utf-8", errors="replace")
    return raw[:limit].decode("utf-8", errors="ignore"), len(value) > limit or len(raw) > limit


def _payload(
    manufacturer: Mapping[int, bytes],
    services: Mapping[str, bytes],
    name: str | None,
    uuids: tuple[str, ...],
    limit: int,
) -> tuple[dict[int, bytes], dict[str, bytes], str | None, tuple[str, ...], bool]:
    """Copy a bounded snapshot, sharing one byte budget across all advertisement payload fields."""
    left = limit
    truncated = False
    out_name: str | None = None
    if name is not None:
        out_name, cut = _text(name, min(256, left))
        truncated |= cut
        left -= len(out_name.encode())
    out_m: dict[int, bytes] = {}
    for i, (key, value) in enumerate(manufacturer.items()):
        if i >= _MAX_FIELDS or left < 2:
            truncated = True
            break
        if not isinstance(key, int) or isinstance(key, bool) or not 0 <= key <= 65535:
            truncated = True
            continue
        kept = bytes(value[: max(0, left - 2)])
        truncated |= len(kept) < len(value)
        out_m[key] = kept
        left -= len(kept) + 2
    out_s: dict[str, bytes] = {}
    for i, (key, value) in enumerate(services.items()):
        if i >= _MAX_FIELDS or len(key) > 128 or len(key.encode()) > min(128, left):
            truncated = True
            break
        left -= len(key.encode())
        kept = bytes(value[:left])
        truncated |= len(kept) < len(value)
        out_s[key] = kept
        left -= len(kept)
    out_u: list[str] = []
    for i, value in enumerate(uuids):
        if i >= _MAX_FIELDS or len(value) > 128 or len(value.encode()) > min(128, left):
            truncated = True
            break
        out_u.append(value)
        left -= len(value.encode())
    return out_m, out_s, out_name, tuple(out_u), truncated


def _payload_size(
    manufacturer: Mapping[int, bytes], services: Mapping[str, bytes], name: str | None, uuids: tuple[str, ...]
) -> int:
    return (
        len((name or "").encode())
        + sum(len(v) + 2 for v in manufacturer.values())
        + sum(len(k.encode()) + len(v) for k, v in services.items())
        + sum(len(u.encode()) for u in uuids)
    )


@dataclass(frozen=True)
class Advertisement:
    """A backend report; absent optional values remain unknown, never zero or an empty invented name."""

    observed_at: float
    backend: str
    adapter: str
    identifier_kind: str
    identifier: str
    address_type: str | None = None
    rssi: int | None = None
    local_name: str | None = None
    manufacturer_data: Mapping[int, bytes] = field(default_factory=dict)
    service_data: Mapping[str, bytes] = field(default_factory=dict)
    service_uuids: tuple[str, ...] = ()
    tx_power: int | None = None
    scan_mode: str = "unknown"
    raw_payload: bytes | None = None
    device_counter: int | None = None
    device_uptime_us: int | None = None
    boot_id: str | None = None
    channel: int | None = None
    received_at: float | None = None
    clock_offset_s: float | None = None
    sensor_name: str | None = None
    payload_malformed: bool = False
    source_payload_truncated: bool = False
    payload_truncated: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.observed_at) or self.observed_at < 0:
            raise ValueError("observed_at must be a finite nonnegative epoch timestamp")
        if self.identifier_kind not in ("bluetooth_address", "corebluetooth_uuid"):
            raise ValueError("unknown native identifier kind")
        for name, value, limit in (
            ("backend", self.backend, 64),
            ("adapter", self.adapter, 128),
            ("identifier", self.identifier, 256),
            ("scan_mode", self.scan_mode, 64),
        ):
            if not value or len(value) > limit or len(value.encode()) > limit:
                raise ValueError(f"{name} must be nonempty and at most {limit} bytes")
        if self.address_type is not None and (len(self.address_type) > 32 or len(self.address_type.encode()) > 32):
            raise ValueError("address_type is too long")
        m, s, name, uuids, truncated = _payload(
            self.manufacturer_data, self.service_data, self.local_name, self.service_uuids, _MAX_OBSERVATION_BYTES
        )
        truncated |= self.source_payload_truncated
        for field_name, value in (("device_counter", self.device_counter), ("device_uptime_us", self.device_uptime_us)):
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < 2**64):
                raise ValueError(f"{field_name} must be an unsigned 64-bit integer")
        for field_name, value in (("received_at", self.received_at), ("clock_offset_s", self.clock_offset_s)):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        if self.boot_id is not None and (len(self.boot_id) > 128 or len(self.boot_id.encode()) > 128):
            raise ValueError("boot_id is too long")
        if self.tx_power is not None and (
            not isinstance(self.tx_power, int) or isinstance(self.tx_power, bool) or not -127 <= self.tx_power <= 126
        ):
            object.__setattr__(self, "tx_power", None)
        if self.channel is not None and not 0 <= self.channel <= 39:
            raise ValueError("invalid BLE channel")
        if self.sensor_name is not None:
            sensor_name, cut = _text(self.sensor_name, 128)
            object.__setattr__(self, "sensor_name", sensor_name)
            truncated |= cut
        if self.raw_payload is not None:
            raw = bytes(self.raw_payload[: max(0, _MAX_OBSERVATION_BYTES - _payload_size(m, s, name, uuids))])
            truncated |= len(raw) < len(self.raw_payload)
            object.__setattr__(self, "raw_payload", raw)
        object.__setattr__(self, "manufacturer_data", MappingProxyType(m))
        object.__setattr__(self, "service_data", MappingProxyType(s))
        object.__setattr__(self, "local_name", name)
        object.__setattr__(self, "service_uuids", uuids)
        object.__setattr__(self, "payload_truncated", truncated)


def _key(ad: Advertisement) -> NativeKey:
    return ad.backend, ad.adapter, ad.identifier_kind, ad.identifier, ad.address_type


def _rssi(value: int | None) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and -127 <= value <= 20 else None


@dataclass
class _Window:
    first: float
    last: float
    latest: Advertisement
    reports: int = 0
    rssi_count: int = 0
    rssi_sum: int = 0
    rssi_min: int | None = None
    rssi_max: int | None = None
    changes: int = 0
    truncated: bool = False
    malformed: bool = False
    modes: set[str] = field(default_factory=set)
    boot_ids: set[str] = field(default_factory=set)
    boot_ids_truncated: bool = False
    channels: set[int] = field(default_factory=set)
    received_first: float | None = None
    received_last: float | None = None
    offset_min: float | None = None
    offset_max: float | None = None
    uptime_min: int | None = None
    uptime_max: int | None = None


class Aggregator:
    """One pending aggregate per native identifier, drained by first-report age.

    At capacity, reports for new identifiers are dropped and counted; existing windows are preserved. The caller
    drains frequently and publishes loss counters, so overload cannot masquerade as complete device coverage.
    """

    def __init__(self, window_s: float = 5.0, max_devices: int = 1024, max_payload_bytes: int = 4096) -> None:
        if not math.isfinite(window_s) or window_s <= 0 or max_devices < 1 or max_payload_bytes < 1:
            raise ValueError("aggregation limits must be positive and finite")
        self.window_s = window_s
        self.max_devices = max_devices
        self.max_payload_bytes = min(max_payload_bytes, _MAX_OBSERVATION_BYTES)
        self._windows: dict[NativeKey, _Window] = {}
        self._stats = dict(reports=0, accepted=0, dropped_capacity=0, truncated_reports=0, invalid_rssi=0, drained=0)

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats, pending_devices=len(self._windows))

    def add(self, ad: Advertisement) -> None:
        self._stats["reports"] += 1
        key = _key(ad)
        window = self._windows.get(key)
        if window is None and len(self._windows) >= self.max_devices:
            self._stats["dropped_capacity"] += 1
            return
        m, s, name, uuids, truncated = _payload(
            ad.manufacturer_data, ad.service_data, ad.local_name, ad.service_uuids, self.max_payload_bytes
        )
        raw = (
            None
            if ad.raw_payload is None
            else bytes(ad.raw_payload[: max(0, self.max_payload_bytes - _payload_size(m, s, name, uuids))])
        )
        truncated |= ad.raw_payload is not None and len(raw or b"") < len(ad.raw_payload)
        bounded = Advertisement(
            ad.observed_at,
            ad.backend,
            ad.adapter,
            ad.identifier_kind,
            ad.identifier,
            ad.address_type,
            ad.rssi,
            name,
            m,
            s,
            uuids,
            ad.tx_power,
            ad.scan_mode,
            raw_payload=raw,
            device_counter=ad.device_counter,
            device_uptime_us=ad.device_uptime_us,
            boot_id=ad.boot_id,
            channel=ad.channel,
            received_at=ad.received_at,
            clock_offset_s=ad.clock_offset_s,
            sensor_name=ad.sensor_name,
            payload_malformed=ad.payload_malformed,
        )
        truncated |= ad.payload_truncated
        if truncated:
            self._stats["truncated_reports"] += 1
        if window is None:
            window = self._windows[key] = _Window(ad.observed_at, ad.observed_at, bounded)
        else:
            if (window.latest.manufacturer_data, window.latest.service_data, window.latest.raw_payload) != (m, s, raw):
                window.changes += 1
            if ad.observed_at >= window.last:
                window.latest = bounded
        window.first = min(window.first, ad.observed_at)
        window.last = max(window.last, ad.observed_at)
        window.reports += 1
        window.truncated |= truncated
        window.malformed |= ad.payload_malformed
        if len(window.modes) < 4:
            window.modes.add(ad.scan_mode)
        elif ad.scan_mode not in window.modes:
            window.modes.add("mixed")
        if ad.boot_id is not None:
            if len(window.boot_ids) < 4 or ad.boot_id in window.boot_ids:
                window.boot_ids.add(ad.boot_id)
            else:
                window.boot_ids_truncated = True
        if ad.channel is not None:
            window.channels.add(ad.channel)
        if ad.received_at is not None:
            window.received_first = (
                ad.received_at if window.received_first is None else min(window.received_first, ad.received_at)
            )
            window.received_last = (
                ad.received_at if window.received_last is None else max(window.received_last, ad.received_at)
            )
        if ad.clock_offset_s is not None:
            window.offset_min = (
                ad.clock_offset_s if window.offset_min is None else min(window.offset_min, ad.clock_offset_s)
            )
            window.offset_max = (
                ad.clock_offset_s if window.offset_max is None else max(window.offset_max, ad.clock_offset_s)
            )
        if ad.device_uptime_us is not None:
            window.uptime_min = (
                ad.device_uptime_us if window.uptime_min is None else min(window.uptime_min, ad.device_uptime_us)
            )
            window.uptime_max = (
                ad.device_uptime_us if window.uptime_max is None else max(window.uptime_max, ad.device_uptime_us)
            )
        value = _rssi(ad.rssi)
        if value is not None:
            window.rssi_count += 1
            window.rssi_sum += value
            window.rssi_min = value if window.rssi_min is None else min(window.rssi_min, value)
            window.rssi_max = value if window.rssi_max is None else max(window.rssi_max, value)
        elif ad.rssi is not None:
            self._stats["invalid_rssi"] += 1
        self._stats["accepted"] += 1

    def drain(self, now: float, limit: int, force: bool = False) -> list[dict[str, Any]]:
        if not math.isfinite(now) or limit < 0:
            raise ValueError("drain requires a finite timestamp and nonnegative limit")
        ready = sorted(
            (key for key, w in self._windows.items() if force or now - w.first >= self.window_s),
            key=lambda key: self._windows[key].first,
        )[:limit]
        out: list[dict[str, Any]] = []
        for key in ready:
            w = self._windows.pop(key)
            ad = w.latest
            out.append(
                dict(
                    id=str(uuid.uuid7()),
                    schema_version=1,
                    start=w.first,
                    end=w.last,
                    backend=ad.backend,
                    adapter=ad.adapter,
                    identifier_kind=ad.identifier_kind,
                    identifier=ad.identifier,
                    address_type=ad.address_type,
                    reports=w.reports,
                    rssi=dict(
                        samples=w.rssi_count,
                        min=w.rssi_min,
                        max=w.rssi_max,
                        mean=w.rssi_sum / w.rssi_count if w.rssi_count else None,
                    ),
                    local_name=ad.local_name,
                    manufacturer_data={str(k): v.hex() for k, v in ad.manufacturer_data.items()},
                    service_data={k: v.hex() for k, v in ad.service_data.items()},
                    service_uuids=list(ad.service_uuids),
                    tx_power=ad.tx_power,
                    sensor_name=ad.sensor_name,
                    raw_payload=None if ad.raw_payload is None else ad.raw_payload.hex(),
                    device_counter=ad.device_counter,
                    device_uptime_us=ad.device_uptime_us,
                    boot_id=ad.boot_id if len(w.boot_ids) <= 1 and not w.boot_ids_truncated else None,
                    boot_ids=sorted(w.boot_ids),
                    boot_ids_truncated=w.boot_ids_truncated,
                    channel=ad.channel,
                    channels=sorted(w.channels),
                    received_at=ad.received_at,
                    received_at_start=w.received_first,
                    received_at_end=w.received_last,
                    clock_offset_s=ad.clock_offset_s,
                    clock_offset_s_min=w.offset_min,
                    clock_offset_s_max=w.offset_max,
                    device_uptime_us_min=w.uptime_min,
                    device_uptime_us_max=w.uptime_max,
                    payload_truncated=w.truncated,
                    payload_malformed=w.malformed,
                    payload_changes=w.changes,
                    scan_mode=next(iter(w.modes)) if len(w.modes) == 1 else "mixed",
                    scan_modes=sorted(w.modes),
                )
            )
        self._stats["drained"] += len(out)
        return out


def payload_fingerprint(record: dict[str, Any]) -> str | None:
    """The exact-payload fingerprint the candidate index groups sightings by, or None when the payload is too short,
    too plain, truncated, malformed, or changed within the window to be one (#1329 links a device by it)."""
    payload = CandidateIndex._eligible_payload(record)
    if payload is None:
        return None
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class _SightingEvidence:
    payload_id: str
    first: float
    last: float
    reports: int


@dataclass
class _Member:
    # Retain exactly the evidence we name as parents. Lifetime totals would survive purges of discarded parents.
    sightings: list[_SightingEvidence] = field(default_factory=list)

    @property
    def first(self) -> float:
        return min(sighting.first for sighting in self.sightings)

    @property
    def last(self) -> float:
        return max(sighting.last for sighting in self.sightings)

    @property
    def reports(self) -> int:
        return sum(sighting.reports for sighting in self.sightings)

    @property
    def ids(self) -> list[str]:
        return [sighting.payload_id for sighting in self.sightings]


@dataclass
class _Fingerprint:
    last: float
    payload: dict[str, Any]
    members: OrderedDict[NativeKey, _Member] = field(default_factory=OrderedDict)
    emitted: tuple[NativeKey, ...] = ()


class CandidateIndex:
    """Bounded, expiring hints from exact complete payload equality, never a merged device identity.

    At least eight payload bytes with four distinct values are required, and each identifier needs two backend
    reports. These conservative heuristics reject trivial constants, but shared factory payloads can still collide.
    Candidate evidence states that limitation explicitly. Names, service UUID lists, RSSI, and TX power are not proof.
    Member spans and report counts cover only the two retained sighting aggregates named as causal parents, not
    the identifier's lifetime totals. Purging any contributing raw sighting can therefore invalidate its candidate.
    """

    def __init__(self, max_entries: int = 1024, ttl_s: float = 3600, max_members: int = 8) -> None:
        if max_entries < 1 or not math.isfinite(ttl_s) or ttl_s <= 0 or max_members < 2:
            raise ValueError("candidate limits must be positive, with at least two members")
        self.max_entries, self.ttl_s, self.max_members = max_entries, ttl_s, max_members
        self._entries: OrderedDict[tuple[str, str, str, str], _Fingerprint] = OrderedDict()
        self._stats = dict(
            observed=0, candidates=0, expired=0, expired_members=0, evicted=0, dropped_members=0, ineligible=0
        )
        self._watermark = 0.0

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats, entries=len(self._entries))

    def observe(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        self._stats["observed"] += 1
        now = float(record["end"])
        start = float(record["start"])
        if not math.isfinite(now) or not math.isfinite(start) or start < 0 or now < start or int(record["reports"]) < 1:
            raise ValueError("invalid candidate observation span or report count")
        self._watermark = max(self._watermark, now)
        if now < self._watermark - self.ttl_s:
            self._stats["ineligible"] += 1
            return []
        for key in list(self._entries):
            if self._watermark - self._entries[key].last > self.ttl_s:
                del self._entries[key]
                self._stats["expired"] += 1
            else:
                entry = self._entries[key]
                for native in list(entry.members):
                    if self._watermark - entry.members[native].last > self.ttl_s:
                        del entry.members[native]
                        self._stats["expired_members"] += 1
                entry.emitted = tuple(native for native in entry.emitted if native in entry.members)
        for name, limit in (
            ("backend", 64),
            ("adapter", 128),
            ("identifier_kind", 32),
            ("identifier", 256),
            ("id", 64),
        ):
            value = record.get(name)
            if not isinstance(value, str) or not value or len(value) > limit or len(value.encode()) > limit:
                self._stats["ineligible"] += 1
                return []
        address_type = record.get("address_type")
        if address_type is not None and (not isinstance(address_type, str) or len(address_type) > 32):
            self._stats["ineligible"] += 1
            return []
        payload = self._eligible_payload(record)
        if payload is None:
            self._stats["ineligible"] += 1
            return []
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        scope = (str(record["backend"]), str(record["adapter"]), str(record["identifier_kind"]), fingerprint)
        native: NativeKey = (*scope[:3], str(record["identifier"]), record.get("address_type"))
        entry = self._entries.get(scope)
        if entry is None:
            if len(self._entries) >= self.max_entries:
                self._entries.popitem(last=False)
                self._stats["evicted"] += 1
            entry = self._entries[scope] = _Fingerprint(now, payload)
        self._entries.move_to_end(scope)
        entry.last = max(entry.last, now)
        member = entry.members.get(native)
        if member is None:
            if len(entry.members) >= self.max_members:
                self._stats["dropped_members"] += 1
                return []
            member = entry.members[native] = _Member()
        record_id = str(record["id"])
        if record_id in member.ids:
            return []  # replay does not turn one report into two independent observations
        member.sightings = [
            *member.sightings[-1:],
            _SightingEvidence(record_id, start, now, int(record["reports"])),
        ]
        qualified = tuple(
            sorted(
                (key for key, value in entry.members.items() if value.reports >= 2), key=lambda k: (*k[:4], k[4] or "")
            )
        )
        if len({key[3].casefold() for key in qualified}) < 2 or qualified == entry.emitted:
            return []
        entry.emitted = qualified
        self._stats["candidates"] += 1
        members = [
            dict(
                backend=k[0],
                adapter=k[1],
                identifier_kind=k[2],
                identifier=k[3],
                address_type=k[4],
                start=entry.members[k].first,
                end=entry.members[k].last,
                reports=entry.members[k].reports,
            )
            for k in qualified
        ]
        return [
            dict(
                id=str(uuid.uuid7()),
                schema_version=1,
                status="unconfirmed",
                method="exact_repeated_advertised_payload",
                fingerprint=fingerprint,
                fingerprint_algorithm="sha256",
                evidence=dict(payload={key: dict(value) for key, value in entry.payload.items()}),
                start=min(entry.members[k].first for k in qualified),
                end=max(entry.members[k].last for k in qualified),
                members=members,
                parents=list(dict.fromkeys(pid for key in qualified for pid in entry.members[key].ids)),
                caveat="Equal advertised payloads can be shared by unrelated devices; this is a candidate, not device identity or proof of address rotation.",
            )
        ]

    @staticmethod
    def _eligible_payload(record: dict[str, Any]) -> dict[str, Any] | None:
        if record.get("payload_truncated") or record.get("payload_malformed") or record.get("payload_changes"):
            return None
        manufacturer = record.get("manufacturer_data") or {}
        service = record.get("service_data") or {}
        if not isinstance(manufacturer, dict) or not isinstance(service, dict) or len(manufacturer) + len(service) > 64:
            return None
        values: list[bytes] = []
        normalized: list[dict[str, str]] = []
        size = 0
        for mapping in (manufacturer, service):
            out: dict[str, str] = {}
            for key, value in mapping.items():
                if not isinstance(key, str) or len(key) > 128 or not isinstance(value, str) or len(value) > 131072:
                    return None
                try:
                    data = bytes.fromhex(value)
                except ValueError:
                    return None
                size += len(data)
                if size > _MAX_OBSERVATION_BYTES:
                    return None
                values.append(data)
                out[key] = data.hex()
            normalized.append(out)
        raw = b"".join(values)
        if len(raw) < 8 or len(set(raw)) < 4:
            return None
        return dict(manufacturer_data=normalized[0], service_data=normalized[1])
