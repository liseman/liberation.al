import asyncio
from dataclasses import dataclass

import pytest

from liberational.proximity.bluez import ADAPTER, DBUS, DEVICE, MANAGER, PROPERTIES, BlueZScanner, Signal

ADAPTER_PATH = "/org/bluez/hci0"
DEVICE_PATH = ADAPTER_PATH + "/dev_AA_BB_CC_DD_EE_FF"


@dataclass
class Variant:
    value: object


def device(**extra):
    return {"Address": Variant("aa:bb:cc:dd:ee:ff"), "AddressType": Variant("random"), **extra}


class FakeBus:
    def __init__(self, objects=None, fail=None):
        self.connected = False
        self.objects = objects if objects is not None else {ADAPTER_PATH: {ADAPTER: {"Powered": Variant(True)}}}
        self.calls = []
        self.handlers = []
        self.fail = fail

    async def connect(self):
        self.connected = True

    async def call(self, destination, path, interface, member, signature="", body=()):
        self.calls.append((destination, path, interface, member, signature, body))
        if member == self.fail:
            raise RuntimeError("synthetic D-Bus failure")
        if member == "GetNameOwner":
            return [":1.42"]
        if member == "GetManagedObjects":
            return [self.objects]
        return []

    def add_handler(self, handler):
        self.handlers.append(handler)

    def remove_handler(self, handler):
        self.handlers.remove(handler)

    def disconnect(self):
        self.connected = False

    def signal(self, member, body, path=DEVICE_PATH, interface=PROPERTIES, sender=":1.42"):
        for handler in tuple(self.handlers):
            handler(Signal(path, interface, member, body, sender))

    def update(self, changed, invalidated=(), path=DEVICE_PATH):
        self.signal("PropertiesChanged", [DEVICE, changed, invalidated], path=path)


@pytest.fixture
def bus():
    return FakeBus()


def test_initial_cache_is_not_fresh_and_only_own_discovery_session_is_released(bus):
    async def run():
        bus.objects[DEVICE_PATH] = {DEVICE: device(RSSI=Variant(-42), Name=Variant("cached name"))}
        found = []
        scanner = BlueZScanner(bus_factory=lambda: bus, clock=lambda: 123)
        await scanner.start(found.append)
        assert found == []
        assert scanner.metadata == {
            "backend": "bluez",
            "adapter": ADAPTER_PATH,
            "scan_mode": "os_managed_active",
            "requested_transport": "le",
            "reported_transport": "unknown_shared_discovery",
        }
        scanner.check()
        bus.update({"RSSI": Variant(-60)})
        assert [(ad.identifier, ad.address_type, ad.rssi, ad.local_name, ad.raw_payload) for ad in found] == [
            ("AA:BB:CC:DD:EE:FF", "random", -60, "cached name", None)
        ]
        await scanner.stop()
        assert not bus.connected and bus.handlers == []
        assert [call[3] for call in bus.calls] == [
            "GetNameOwner",
            "AddMatch",
            "AddMatch",
            "AddMatch",
            "GetManagedObjects",
            "SetDiscoveryFilter",
            "StartDiscovery",
            "StopDiscovery",
        ]
        assert bus.calls[5][-1] == [{"Transport": "le", "DuplicateData": True}]
        assert bus.calls[7][0] == ":1.42"  # never stop a replacement daemon's session
        await scanner.stop()

    asyncio.run(run())


def test_added_unknown_device_and_duplicate_payload_updates_both_emit(bus):
    async def run():
        found = []
        scanner = BlueZScanner(bus_factory=lambda: bus, clock=lambda: 123)
        await scanner.start(found.append)
        props = device(ManufacturerData=Variant({65535: Variant(b"unknown vendor")}))
        bus.signal("InterfacesAdded", [DEVICE_PATH, {DEVICE: props}], interface=MANAGER, path="/")
        bus.update({"ManufacturerData": props["ManufacturerData"]})
        assert len(found) == 2
        assert found[0].manufacturer_data == {65535: b"unknown vendor"}
        assert found[0].rssi is None and found[0].local_name is None
        assert found[0].scan_mode == "os_managed_active"
        await scanner.stop()

    asyncio.run(run())


def test_invalidated_fields_are_unknown_and_metadata_only_updates_are_not_sightings(bus):
    async def run():
        found = []
        scanner = BlueZScanner(bus_factory=lambda: bus, clock=lambda: 123)
        await scanner.start(found.append)
        bus.signal(
            "InterfacesAdded",
            [DEVICE_PATH, {DEVICE: device(RSSI=Variant(-30), Name=Variant("old"))}],
            interface=MANAGER,
        )
        bus.update({}, ["RSSI", "Name"])
        bus.update({"Connected": Variant(True), "Name": Variant("metadata")})
        assert len(found) == 1
        bus.update({"ServiceData": Variant({})}, ["Name"])
        assert len(found) == 2
        assert found[-1].rssi is None and found[-1].local_name is None
        await scanner.stop()

    asyncio.run(run())


def test_bounded_cache_eviction_loses_old_identity_instead_of_inventing_it(bus):
    async def run():
        found = []
        scanner = BlueZScanner(max_devices=1, bus_factory=lambda: bus, clock=lambda: 123)
        await scanner.start(found.append)
        bus.signal("InterfacesAdded", [DEVICE_PATH, {DEVICE: device(RSSI=Variant(-30))}], interface=MANAGER)
        bus.signal("InterfacesAdded", [DEVICE_PATH + "2", {DEVICE: device(RSSI=Variant(-40))}], interface=MANAGER)
        assert scanner.stats["cache_evictions"] == 1
        bus.update({"RSSI": Variant(-20)})
        assert len(found) == 2
        assert scanner.stats["lost_reports"] == 1
        assert len(scanner.devices) == 1
        await scanner.stop()

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["power", "discovery", "invalidated", "adapter", "daemon", "bus"])
def test_current_adapter_lease_loss_is_fatal(bus, kind):
    async def run():
        scanner = BlueZScanner(bus_factory=lambda: bus)
        await scanner.start(lambda ad: None)
        if kind == "power":
            bus.signal("PropertiesChanged", [ADAPTER, {"Powered": Variant(False)}, []], path=ADAPTER_PATH)
        elif kind == "discovery":
            bus.signal("PropertiesChanged", [ADAPTER, {"Discovering": Variant(False)}, []], path=ADAPTER_PATH)
        elif kind == "invalidated":
            bus.signal("PropertiesChanged", [ADAPTER, {}, ["Powered"]], path=ADAPTER_PATH)
        elif kind == "adapter":
            bus.signal("InterfacesRemoved", [ADAPTER_PATH, [ADAPTER]], interface=MANAGER)
        elif kind == "daemon":
            bus.signal("NameOwnerChanged", ["org.bluez", ":1.42", ":1.99"], interface=DBUS, sender=DBUS)
        else:
            bus.connected = False
        with pytest.raises(RuntimeError, match="BlueZ"):
            scanner.check()
        await scanner.stop()

    asyncio.run(run())


@pytest.mark.parametrize(
    "objects,match", [({}, "not found"), ({ADAPTER_PATH: {ADAPTER: {"Powered": Variant(False)}}}, "powered off")]
)
def test_missing_or_powered_off_adapter_never_starts_discovery(objects, match):
    async def run():
        bus = FakeBus(objects)
        scanner = BlueZScanner(bus_factory=lambda: bus)
        with pytest.raises(RuntimeError, match=match):
            await scanner.start(lambda ad: None)
        assert not bus.connected
        assert "StartDiscovery" not in [call[3] for call in bus.calls]
        assert "StopDiscovery" not in [call[3] for call in bus.calls]

    asyncio.run(run())


def test_start_failure_never_stops_an_unacquired_session(bus):
    async def run():
        bus.fail = "StartDiscovery"
        scanner = BlueZScanner(bus_factory=lambda: bus)
        with pytest.raises(RuntimeError, match="synthetic"):
            await scanner.start(lambda ad: None)
        assert not bus.connected and not bus.handlers
        assert "StopDiscovery" not in [call[3] for call in bus.calls]

    asyncio.run(run())


def test_device_removal_and_unrelated_sender_do_not_create_sightings(bus):
    async def run():
        found = []
        scanner = BlueZScanner(bus_factory=lambda: bus, clock=lambda: 123)
        await scanner.start(found.append)
        bus.signal("InterfacesAdded", [DEVICE_PATH, {DEVICE: device(RSSI=Variant(-30))}], interface=MANAGER)
        bus.signal("InterfacesRemoved", [DEVICE_PATH, [DEVICE]], interface=MANAGER)
        bus.signal(
            "InterfacesAdded", [DEVICE_PATH, {DEVICE: device(RSSI=Variant(-30))}], interface=MANAGER, sender=":1.99"
        )
        assert scanner.devices == {} and len(found) == 1
        await scanner.stop()

    asyncio.run(run())


def test_platform_normalization_preserves_unknown_vendor_uuid_and_truncation(bus):
    async def run():
        found = []
        scanner = BlueZScanner(bus_factory=lambda: bus, clock=lambda: 123)
        await scanner.start(found.append)
        props = device(
            ManufacturerData=Variant({65535: Variant([1, 2, 3])}),
            ServiceData=Variant({"180F": Variant([42])}),
            UUIDs=Variant(["180F", "invalid"]),
            Name=Variant("n" * 1000),
        )
        bus.signal("InterfacesAdded", [DEVICE_PATH, {DEVICE: props}], interface=MANAGER)
        [report] = found
        assert report.manufacturer_data == {65535: b"\x01\x02\x03"}
        assert report.service_data == {"0000180f-0000-1000-8000-00805f9b34fb": b"*"}
        assert report.service_uuids == ("0000180f-0000-1000-8000-00805f9b34fb",)
        assert report.payload_truncated and report.payload_malformed
        assert report.local_name == "n" * 256
        await scanner.stop()

    asyncio.run(run())


def test_cleanup_failure_disconnects_and_does_not_retain_callbacks(bus):
    async def run():
        scanner = BlueZScanner(bus_factory=lambda: bus)
        await scanner.start(lambda ad: None)
        bus.fail = "StopDiscovery"
        await scanner.stop()
        assert not bus.connected and not bus.handlers
        assert scanner.stats["cleanup_errors"] == 1
        assert scanner.bus is None

    asyncio.run(run())


def test_optional_dependency_is_lazy_and_missing_dependency_is_clear(monkeypatch):
    from liberational.proximity import bluez

    imports = []

    def unavailable(name):
        imports.append(name)
        raise ImportError("synthetic optional dependency absence")

    monkeypatch.setattr(bluez.importlib, "import_module", unavailable)
    scanner = BlueZScanner()
    assert imports == []
    with pytest.raises(RuntimeError, match="dbus-fast"):
        asyncio.run(scanner.start(lambda ad: None))
    assert imports == ["dbus_fast"]


def test_adapter_selection_is_read_only_and_respects_explicit_adapter(bus):
    async def run():
        second = "/org/bluez/hci1"
        bus.objects[second] = {ADAPTER: {"Powered": Variant(True)}}
        scanner = BlueZScanner(adapter="hci1", bus_factory=lambda: bus)
        await scanner.start(lambda ad: None)
        assert scanner.metadata["adapter"] == second
        await scanner.stop()
        adapter_calls = [(call[1], call[3]) for call in bus.calls if call[2] == ADAPTER]
        assert adapter_calls == [(second, "SetDiscoveryFilter"), (second, "StartDiscovery"), (second, "StopDiscovery")]

    asyncio.run(run())


def test_call_timeout_disconnects_without_stopping_an_unacquired_session(bus):
    class StalledBus(FakeBus):
        async def call(self, destination, path, interface, member, signature="", body=()):
            if member == "GetManagedObjects":
                await asyncio.Event().wait()
            return await super().call(destination, path, interface, member, signature, body)

    async def run():
        stalled = StalledBus()
        scanner = BlueZScanner(bus_factory=lambda: stalled, timeout_s=0.01)
        with pytest.raises(RuntimeError, match="GetManagedObjects timed out"):
            await scanner.start(lambda ad: None)
        assert not stalled.connected and not stalled.handlers
        assert "StopDiscovery" not in [call[3] for call in stalled.calls]

    asyncio.run(run())


def test_cancelled_stop_still_removes_handler_and_disconnects():
    class CancellingBus(FakeBus):
        async def call(self, destination, path, interface, member, signature="", body=()):
            if member == "StopDiscovery":
                raise asyncio.CancelledError
            return await super().call(destination, path, interface, member, signature, body)

    async def run():
        bus = CancellingBus()
        scanner = BlueZScanner(bus_factory=lambda: bus)
        await scanner.start(lambda ad: None)
        with pytest.raises(asyncio.CancelledError):
            await scanner.stop()
        assert not bus.connected and not bus.handlers and scanner.bus is None

    asyncio.run(run())


def test_connection_failure_cleans_up_before_any_handler_is_registered():
    class BrokenBus(FakeBus):
        async def connect(self):
            raise RuntimeError("system bus unavailable")

    async def run():
        bus = BrokenBus()
        scanner = BlueZScanner(bus_factory=lambda: bus)
        with pytest.raises(RuntimeError, match="system bus unavailable"):
            await scanner.start(lambda ad: None)
        assert not bus.handlers and scanner.bus is None
        assert bus.calls == []

    asyncio.run(run())


def test_cancelled_stalled_start_disconnects_without_waiting_for_cleanup_rpc():
    class StalledBus(FakeBus):
        def __init__(self):
            super().__init__()
            self.read_started = asyncio.Event()
            self.cleanup_started = asyncio.Event()
            self.blocked = asyncio.Event()

        async def call(self, destination, path, interface, member, signature="", body=()):
            if member == "GetManagedObjects":
                self.read_started.set()
                await self.blocked.wait()
            if member in {"RemoveMatch", "StopDiscovery"}:
                self.cleanup_started.set()
                await self.blocked.wait()
            return await super().call(destination, path, interface, member, signature, body)

    async def run():
        bus = StalledBus()
        scanner = BlueZScanner(bus_factory=lambda: bus)
        task = asyncio.create_task(scanner.start(lambda ad: None))
        await bus.read_started.wait()
        task.cancel()
        cleanup = asyncio.create_task(bus.cleanup_started.wait())
        try:
            done, _ = await asyncio.wait([task, cleanup], return_when=asyncio.FIRST_COMPLETED)
            assert task in done, "startup cancellation must not wait for a cleanup RPC"
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not bus.connected and not bus.handlers and scanner.bus is None
            assert not bus.cleanup_started.is_set()
        finally:
            task.cancel()
            cleanup.cancel()
            await asyncio.gather(task, cleanup, return_exceptions=True)

    asyncio.run(run())


def test_failed_start_does_not_send_cleanup_rpcs_on_dedicated_connection():
    async def run():
        bus = FakeBus(fail="GetManagedObjects")
        scanner = BlueZScanner(bus_factory=lambda: bus)
        with pytest.raises(RuntimeError, match="synthetic"):
            await scanner.start(lambda ad: None)
        assert [call[3] for call in bus.calls] == [
            "GetNameOwner",
            "AddMatch",
            "AddMatch",
            "AddMatch",
            "GetManagedObjects",
        ]
        assert not bus.connected and not bus.handlers and scanner.bus is None

    asyncio.run(run())
