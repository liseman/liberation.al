"""Linux OS-managed LE discovery, with no pairing, connections, GATT reads, or adapter power changes.

BlueZ shares radio discovery across clients. This backend owns one session on its dedicated system-bus connection
and releases only that session. Initial Device1 objects seed bounded metadata, never fresh sightings. Properties
are OS reports, not reconstructed raw advertising packets. Shared Device1 updates do not expose per-report bearer:
another client's classic discovery can contribute properties, so requesting LE does not certify LE-only evidence.
See BlueZ's org.bluez.Adapter/Device API documentation.
"""

import asyncio
import importlib
import re
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import islice
from typing import Any, Protocol

from liberational.proximity.advertising import platform_data
from liberational.proximity.ble import Advertisement

ADAPTER = "org.bluez.Adapter1"
DEVICE = "org.bluez.Device1"
MANAGER = "org.freedesktop.DBus.ObjectManager"
PROPERTIES = "org.freedesktop.DBus.Properties"
DBUS = "org.freedesktop.DBus"
EVIDENCE = frozenset({"RSSI", "ManufacturerData", "ServiceData", "AdvertisingData", "AdvertisingFlags", "TxPower"})
FIELDS = EVIDENCE | {"Address", "AddressType", "Name", "UUIDs", "Adapter"}
MAC = re.compile(r"(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\Z")


@dataclass(frozen=True)
class Signal:
    path: str
    interface: str
    member: str
    body: Sequence[Any]
    sender: str = ""


class Bus(Protocol):
    @property
    def connected(self) -> bool: ...
    async def connect(self) -> None: ...
    async def call(
        self, destination: str, path: str, interface: str, member: str, signature: str = "", body: Sequence[Any] = ()
    ) -> list[Any]: ...
    def add_handler(self, handler: Callable[[Signal], None]) -> None: ...
    def remove_handler(self, handler: Callable[[Signal], None]) -> None: ...
    def disconnect(self) -> None: ...


class _SystemBus:
    """Lazy optional dependency boundary; construction alone never contacts Bluetooth."""

    def __init__(self) -> None:
        try:
            self.api = importlib.import_module("dbus_fast")
            aio = importlib.import_module("dbus_fast.aio")
        except ImportError as error:
            raise RuntimeError("BlueZ scanning requires the optional dbus-fast dependency") from error
        self.bus = aio.MessageBus(bus_type=self.api.BusType.SYSTEM)
        self.handlers: dict[Callable[[Signal], None], Callable[[Any], None]] = {}

    @property
    def connected(self) -> bool:
        return bool(self.bus.connected)

    async def connect(self) -> None:
        await self.bus.connect()

    async def call(
        self, destination: str, path: str, interface: str, member: str, signature: str = "", body: Sequence[Any] = ()
    ) -> list[Any]:
        if member == "SetDiscoveryFilter":
            body = [{"Transport": self.api.Variant("s", "le"), "DuplicateData": self.api.Variant("b", True)}]
        reply = await self.bus.call(
            self.api.Message(
                destination=destination,
                path=path,
                interface=interface,
                member=member,
                signature=signature,
                body=list(body),
            )
        )
        if reply is None or reply.message_type == self.api.MessageType.ERROR:
            name = getattr(reply, "error_name", "no reply")
            raise RuntimeError(f"BlueZ D-Bus {member} failed: {name}")
        return list(reply.body)

    def add_handler(self, handler: Callable[[Signal], None]) -> None:
        def receive(message: Any) -> None:
            if message.message_type == self.api.MessageType.SIGNAL:
                handler(
                    Signal(
                        message.path or "",
                        message.interface or "",
                        message.member or "",
                        message.body,
                        message.sender or "",
                    )
                )

        self.handlers[handler] = receive
        self.bus.add_message_handler(receive)

    def remove_handler(self, handler: Callable[[Signal], None]) -> None:
        callback = self.handlers.pop(handler, None)
        if callback is not None:
            self.bus.remove_message_handler(callback)

    def disconnect(self) -> None:
        self.bus.disconnect()


def _value(value: Any) -> Any:
    return getattr(value, "value", value)


def _properties(properties: Any) -> dict[str, Any]:
    """Unwrap only bounded identity and advertising fields, preserving explicit data loss flags."""
    if not isinstance(properties, dict):
        return {}
    values = {key: _value(properties[key]) for key in FIELDS if key in properties}
    malformed = False
    for key in ("ManufacturerData", "ServiceData"):
        if key in values:
            value = values[key]
            if not isinstance(value, dict):
                values[key], malformed = {}, True
            else:
                # Keep one excess entry so the common normalizer can mark truncation.
                values[key] = {k: _value(v) for k, v in islice(value.items(), 65)}
    uuids = values.get("UUIDs", ())
    if not isinstance(uuids, (list, tuple)):
        uuids, malformed = (), True
    data = platform_data(
        local_name=values.get("Name"),
        manufacturer_data=values.get("ManufacturerData"),
        service_data=values.get("ServiceData"),
        service_uuids=uuids,
        tx_power=values.get("TxPower"),
    )
    result: dict[str, Any] = {"_malformed": malformed or data.malformed, "_truncated": data.truncated}
    for key, value in (
        ("Name", data.local_name),
        ("ManufacturerData", data.manufacturer_data),
        ("ServiceData", data.service_data),
        ("UUIDs", data.service_uuids),
        ("TxPower", data.tx_power),
    ):
        if key in values:
            result[key] = value
    for key in ("Address", "AddressType", "Adapter"):
        if key in values:
            value = values[key]
            result[key] = value[:128] if isinstance(value, str) else None
    if "RSSI" in values:
        value = values["RSSI"]
        result["RSSI"] = (
            value if isinstance(value, int) and not isinstance(value, bool) and -127 <= value <= 20 else None
        )
    return result


class BlueZScanner:
    """One authorized OS-managed scan lease, with a bounded device cache and injectable bus."""

    def __init__(
        self,
        adapter: str = "",
        max_devices: int = 1024,
        *,
        bus_factory: Callable[[], Bus] = _SystemBus,
        clock: Callable[[], float] = time.time,
        timeout_s: float = 10.0,
    ) -> None:
        if not 1 <= max_devices <= 65536 or not 0 < timeout_s <= 60:
            raise ValueError("invalid BlueZ scanner bounds")
        self.adapter, self.max_devices, self.bus_factory, self.clock, self.timeout_s = (
            adapter,
            max_devices,
            bus_factory,
            clock,
            timeout_s,
        )
        self.stats = dict(
            reports=0, lost_reports=0, cache_evictions=0, invalid_reports=0, ignored_updates=0, cleanup_errors=0
        )
        self.metadata = {
            "backend": "bluez",
            "adapter": adapter,
            "scan_mode": "os_managed_active",
            "requested_transport": "le",
            "reported_transport": "unknown_shared_discovery",
        }
        self.devices: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.bus: Bus | None = None
        self.owner = ""
        self.selected = ""
        self.started = False
        self.failure: str | None = None
        self.emit: Callable[[Advertisement], None] | None = None
        self.listening = False

    async def _call(
        self,
        path: str,
        interface: str,
        member: str,
        signature: str = "",
        body: Sequence[Any] = (),
        *,
        daemon: bool = False,
    ) -> list[Any]:
        assert self.bus is not None
        try:
            return await asyncio.wait_for(
                self.bus.call(DBUS if daemon else self.owner, path, interface, member, signature, body), self.timeout_s
            )
        except TimeoutError as error:
            raise RuntimeError(f"BlueZ D-Bus {member} timed out") from error

    async def start(self, emit: Callable[[Advertisement], None]) -> None:
        if self.bus is not None:
            raise RuntimeError("BlueZ scanner already started")
        self.failure, self.emit = None, emit
        self.devices.clear()
        try:
            self.bus = self.bus_factory()
            await asyncio.wait_for(self.bus.connect(), self.timeout_s)
            [self.owner] = await self._call(
                "/org/freedesktop/DBus", DBUS, "GetNameOwner", "s", ["org.bluez"], daemon=True
            )
            self.bus.add_handler(self._signal)
            self.listening = True
            for rule in (
                f"type='signal',sender='{self.owner}',interface='{MANAGER}'",
                f"type='signal',sender='{self.owner}',interface='{PROPERTIES}'",
                "type='signal',sender='org.freedesktop.DBus',interface='org.freedesktop.DBus',member='NameOwnerChanged',arg0='org.bluez'",
            ):
                await self._call("/org/freedesktop/DBus", DBUS, "AddMatch", "s", [rule], daemon=True)
            [objects] = await self._call("/", MANAGER, "GetManagedObjects")
            adapters = {path: interfaces[ADAPTER] for path, interfaces in objects.items() if ADAPTER in interfaces}
            candidates = sorted(
                path for path in adapters if not self.adapter or self.adapter in (path, path.rsplit("/", 1)[-1])
            )
            if not candidates:
                raise RuntimeError("BlueZ Bluetooth adapter was not found")
            powered = [path for path in candidates if _value(adapters[path].get("Powered")) is True]
            if not powered:
                raise RuntimeError("BlueZ Bluetooth adapter is powered off; enable it in system settings")
            self.selected = powered[0]
            self.metadata["adapter"] = self.selected
            for path, interfaces in objects.items():
                if DEVICE in interfaces and path.startswith(self.selected + "/"):
                    self._cache(path, interfaces[DEVICE])
            await self._call(
                self.selected, ADAPTER, "SetDiscoveryFilter", "a{sv}", [{"Transport": "le", "DuplicateData": True}]
            )
            await self._call(self.selected, ADAPTER, "StartDiscovery")
            self.started = True
            self.check()
        except BaseException:
            self._disconnect()
            raise

    def check(self) -> None:
        if self.failure:
            raise RuntimeError(self.failure)
        if not self.started or self.bus is None or not self.bus.connected:
            raise RuntimeError("BlueZ discovery lease or system bus is unavailable")

    async def stop(self) -> None:
        bus = self.bus
        if bus is None:
            return
        try:
            if self.started and bus.connected:
                try:
                    await self._call(self.selected, ADAPTER, "StopDiscovery")
                except Exception:
                    self.stats["cleanup_errors"] += 1
        finally:
            self._disconnect()

    def _disconnect(self) -> None:
        """Closing this dedicated bus releases matches, filter, and any acquired discovery session without RPCs."""
        bus = self.bus
        if bus is None:
            return
        self.emit = None
        self.started = False
        try:
            if self.listening:
                bus.remove_handler(self._signal)
        finally:
            self.listening = False
            try:
                bus.disconnect()
            finally:
                self.bus = None
                self.devices.clear()

    def _cache(self, path: str, changed: Any, invalidated: Sequence[str] = ()) -> dict[str, Any]:
        props = self.devices.pop(path, {})
        normalized = _properties(changed)
        normalized["_malformed"] = normalized.get("_malformed", False) or props.get("_malformed", False)
        normalized["_truncated"] = normalized.get("_truncated", False) or props.get("_truncated", False)
        props.update(normalized)
        for key in invalidated:
            props.pop(key, None)
        self.devices[path] = props
        if len(self.devices) > self.max_devices:
            self.devices.popitem(last=False)
            self.stats["cache_evictions"] += 1
        return props

    def _signal(self, signal: Signal) -> None:
        try:
            if (
                signal.sender == DBUS
                and signal.interface == DBUS
                and signal.member == "NameOwnerChanged"
                and signal.body[0] == "org.bluez"
            ):
                if signal.body[1] == self.owner and signal.body[2] != self.owner:
                    self.failure = "BlueZ daemon disappeared or restarted; discovery lease was lost"
                return
            if signal.sender != self.owner:
                return
            if signal.interface == MANAGER and signal.member == "InterfacesRemoved":
                path, interfaces = signal.body
                if path == self.selected and ADAPTER in interfaces:
                    self.failure = "BlueZ Bluetooth adapter disappeared"
                if DEVICE in interfaces:
                    self.devices.pop(path, None)
                return
            if signal.interface == PROPERTIES and signal.member == "PropertiesChanged":
                interface, changed, invalidated = signal.body
                if interface == ADAPTER and signal.path == self.selected:
                    if any(
                        key in invalidated or (key in changed and _value(changed[key]) is not True)
                        for key in ("Powered", "Discovering")
                    ):
                        self.failure = "BlueZ adapter powered off or discovery lease was lost"
                    return
                if interface != DEVICE or not signal.path.startswith(self.selected + "/"):
                    return
                props = self._cache(signal.path, changed, invalidated)
            elif signal.interface == MANAGER and signal.member == "InterfacesAdded":
                path, interfaces = signal.body
                if DEVICE not in interfaces or not path.startswith(self.selected + "/"):
                    return
                changed = interfaces[DEVICE]
                props = self._cache(path, changed)
            else:
                return
            if not self.started or self.failure or not EVIDENCE.intersection(changed):
                self.stats["ignored_updates"] += 1
                return
            self._report(props)
        except KeyError, TypeError, ValueError, IndexError:
            self.stats["invalid_reports"] += 1
            self.stats["lost_reports"] += 1

    def _report(self, props: dict[str, Any]) -> None:
        address = props.get("Address")
        if not isinstance(address, str) or not MAC.fullmatch(address):
            self.stats["invalid_reports"] += 1
            self.stats["lost_reports"] += 1
            return
        data = platform_data(
            local_name=props.get("Name"),
            manufacturer_data=props.get("ManufacturerData"),
            service_data=props.get("ServiceData"),
            service_uuids=props.get("UUIDs", ()),
            tx_power=props.get("TxPower"),
        )
        received_at = self.clock()
        ad = Advertisement(
            observed_at=received_at,
            received_at=received_at,
            backend="bluez",
            adapter=self.selected,
            identifier_kind="bluetooth_address",
            identifier=address.upper(),
            address_type=props.get("AddressType"),
            rssi=props.get("RSSI"),
            local_name=data.local_name,
            manufacturer_data=data.manufacturer_data,
            service_data=data.service_data,
            service_uuids=data.service_uuids,
            tx_power=data.tx_power,
            payload_malformed=data.malformed or props.get("_malformed", False),
            source_payload_truncated=data.truncated or props.get("_truncated", False),
            scan_mode="os_managed_active",
        )
        if self.emit is not None:
            self.emit(ad)
            self.stats["reports"] += 1
