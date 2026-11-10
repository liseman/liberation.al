import asyncio
import threading
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest

from liberational.proximity import corebluetooth as cb

IDENTIFIER = "00000000-0000-4000-8000-000000000001"


class FakeCoreBluetooth:
    def __init__(self, state=5, scan_immediate=True, close_immediate=True):
        self.initial_state = state
        self.scan_immediate = scan_immediate
        self.close_immediate = close_immediate
        self.is_scanning = False
        self.opened = asyncio.Event()
        self.scan_called = asyncio.Event()
        self.closed = False
        self.on_state = None
        self.on_report = None
        self.failure = None
        self.done = None

    def open(self, state, report):
        self.on_state = state
        self.on_report = report
        state(self.initial_state)
        self.opened.set()

    def scan(self):
        self.scan_called.set()
        self.is_scanning = self.scan_immediate

    def scanning(self):
        return self.is_scanning

    def close(self, done):
        self.closed = True
        self.done = done
        self.is_scanning = False
        if self.close_immediate:
            done.set()

    def check(self):
        if self.failure is not None:
            raise self.failure

    def report(self, fields=None, rssi=-60, identifier=IDENTIFIER):
        assert self.on_report is not None
        self.on_report(identifier, fields or {}, rssi)

    def state(self, state):
        assert self.on_state is not None
        self.on_state(state)


def test_native_imports_wait_until_explicit_start(monkeypatch):
    forbidden = create_autospec(cb.importlib.import_module, side_effect=AssertionError("native import"))
    monkeypatch.setattr(cb.importlib, "import_module", forbidden)
    scanner = cb.CoreBluetoothScanner()
    assert scanner.metadata == {
        "backend": "corebluetooth",
        "adapter": "system-default",
        "scan_mode": "os_managed_active",
    }
    scanner.check()
    asyncio.run(scanner.stop())
    forbidden.assert_not_called()


def test_unnamed_unknown_rssi_and_duplicates_are_delivered_without_identity_invention():
    runtime = FakeCoreBluetooth()
    scanner = cb.CoreBluetoothScanner("radio-label", runtime_factory=lambda: runtime, clock=lambda: 1234.0)
    seen = []

    async def run():
        await scanner.start(seen.append)
        runtime.report(rssi=127)
        runtime.report(rssi=127)
        await scanner.stop()
        runtime.report()

    asyncio.run(run())
    assert len(seen) == 2
    assert seen[0].identifier == IDENTIFIER
    assert seen[0].identifier_kind == "corebluetooth_uuid"
    assert seen[0].address_type is None
    assert seen[0].rssi is None
    assert seen[0].local_name is None
    assert seen[0].raw_payload is None
    assert seen[0].observed_at == 1234.0
    assert seen[0].scan_mode == "os_managed_active"
    assert seen[0].adapter == "radio-label"
    assert runtime.closed


def test_advertised_fields_copy_company_prefix_services_and_local_name_only():
    runtime = FakeCoreBluetooth()
    scanner = cb.CoreBluetoothScanner(runtime_factory=lambda: runtime)
    seen = []
    value = bytearray(b"\x34\x12\x01\x02")

    async def run():
        await scanner.start(seen.append)
        runtime.report(
            {
                "manufacturer": value,
                "local_name": "Node",
                "service_data": {"180f": b"\x01"},
                "service_uuids": ["180f"],
                "tx_power": -4,
            }
        )
        value[:] = b"xxxx"
        await scanner.stop()

    asyncio.run(run())
    assert seen[0].manufacturer_data == {0x1234: b"\x01\x02"}
    assert seen[0].local_name == "Node"
    assert seen[0].service_data == {"0000180f-0000-1000-8000-00805f9b34fb": b"\x01"}
    assert seen[0].service_uuids == ("0000180f-0000-1000-8000-00805f9b34fb",)
    assert seen[0].tx_power == -4


@pytest.mark.parametrize(
    ("state", "reason"), [(1, "resetting"), (2, "unsupported"), (3, "unauthorized"), (4, "powered off")]
)
def test_startup_state_failures_stop_and_never_scan(state, reason):
    runtime = FakeCoreBluetooth(state)
    scanner = cb.CoreBluetoothScanner(runtime_factory=lambda: runtime)
    with pytest.raises(RuntimeError, match=reason):
        asyncio.run(scanner.start(lambda report: None))
    assert runtime.closed
    assert not runtime.scan_called.is_set()


def test_start_waits_for_power_and_actual_scanning_then_state_loss_fails_check():
    runtime = FakeCoreBluetooth(0, scan_immediate=False)
    scanner = cb.CoreBluetoothScanner(runtime_factory=lambda: runtime)

    async def run():
        task = asyncio.create_task(scanner.start(lambda report: None))
        await runtime.opened.wait()
        assert not task.done()
        runtime.state(5)
        await runtime.scan_called.wait()
        assert not task.done()
        runtime.is_scanning = True
        await asyncio.wait_for(task, 1)
        runtime.state(4)
        with pytest.raises(RuntimeError, match="powered off"):
            scanner.check()
        await scanner.stop()

    asyncio.run(run())


def test_start_cancellation_and_unknown_state_timeout_cleanup():
    runtime = FakeCoreBluetooth(0)
    scanner = cb.CoreBluetoothScanner(runtime_factory=lambda: runtime)

    async def run():
        task = asyncio.create_task(scanner.start(lambda report: None))
        await runtime.opened.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert runtime.closed
    waiting = FakeCoreBluetooth(0)
    scanner = cb.CoreBluetoothScanner(runtime_factory=lambda: waiting, startup_timeout_s=0.001)
    with pytest.raises(RuntimeError, match="startup timed out"):
        asyncio.run(scanner.start(lambda report: None))
    assert waiting.closed


def test_stop_timeout_disables_emission_and_prevents_overlapping_restart():
    runtime = FakeCoreBluetooth(close_immediate=False)
    scanner = cb.CoreBluetoothScanner(runtime_factory=lambda: runtime, stop_timeout_s=0.001)
    seen = []

    async def run():
        await scanner.start(seen.append)
        with pytest.raises(RuntimeError, match="stop timed out"):
            await scanner.stop()
        runtime.report()
        with pytest.raises(RuntimeError, match="already started"):
            await scanner.start(seen.append)
        assert runtime.done is not None
        runtime.done.set()
        await scanner.stop()

    asyncio.run(run())
    assert seen == []


def test_native_callback_failure_is_surfaced_by_check_and_cleanup_runs():
    runtime = FakeCoreBluetooth()
    scanner = cb.CoreBluetoothScanner(runtime_factory=lambda: runtime)

    async def run():
        await scanner.start(lambda report: None)
        runtime.failure = RuntimeError("native callback failed")
        with pytest.raises(RuntimeError, match="native callback failed"):
            scanner.check()
        with pytest.raises(RuntimeError, match="native callback failed"):
            await scanner.stop()

    asyncio.run(run())
    assert runtime.closed


def test_large_payload_and_invalid_identifier_are_bounded_before_emission():
    runtime = FakeCoreBluetooth()
    scanner = cb.CoreBluetoothScanner(runtime_factory=lambda: runtime)
    seen = []

    async def run():
        await scanner.start(seen.append)
        runtime.report({"manufacturer": b"\x34\x12" + b"x" * 100000, "local_name": "n" * 10000})
        runtime.report(identifier="not-a-native-uuid")
        await scanner.stop()

    asyncio.run(run())
    assert len(seen) == 1
    assert len(seen[0].manufacturer_data[0x1234]) <= 65536
    assert len(seen[0].local_name.encode()) <= 256
    assert seen[0].payload_truncated
    assert scanner.stats["invalid_reports"] == 1


def test_native_bridge_scans_all_with_duplicates_and_never_uses_peripheral_name(monkeypatch):
    class NSObject:
        def __init_subclass__(cls, **kwargs):
            pass

        @classmethod
        def alloc(cls):
            return cls()

        def init(self):
            return self

    class Manager:
        def __init__(self):
            self.active = False
            self.delegate = None
            self.scan_args = None
            self.options = None
            self.stopped = False

        def alloc(self):
            return self

        def initWithDelegate_queue_options_(self, delegate, queue, options):
            self.delegate = delegate
            self.options = options
            delegate.centralManagerDidUpdateState_(self)
            return self

        def state(self):
            return 5

        def scanForPeripheralsWithServices_options_(self, services, options):
            self.scan_args = (services, options)
            self.active = True

        def isScanning(self):
            return self.active

        def stopScan(self):
            self.active = False
            self.stopped = True

        def setDelegate_(self, delegate):
            self.delegate = delegate

    class Peripheral:
        def identifier(self):
            return SimpleNamespace(UUIDString=lambda: IDENTIFIER)

        def name(self):
            raise AssertionError("must use advertised local name only")

    manager = Manager()
    framework = SimpleNamespace(
        CBCentralManager=manager,
        CBCentralManagerOptionShowPowerAlertKey="power-alert",
        CBCentralManagerScanOptionAllowDuplicatesKey="duplicates",
        CBAdvertisementDataLocalNameKey="name",
        CBAdvertisementDataManufacturerDataKey="manufacturer",
        CBAdvertisementDataServiceDataKey="service-data",
        CBAdvertisementDataServiceUUIDsKey="service-uuids",
        CBAdvertisementDataTxPowerLevelKey="tx",
    )
    modules = {
        "CoreBluetooth": framework,
        "Foundation": SimpleNamespace(NSObject=NSObject),
        "objc": SimpleNamespace(protocolNamed=lambda name: name, autorelease_pool=nullcontext),
        "libdispatch": SimpleNamespace(
            DISPATCH_QUEUE_SERIAL=0,
            dispatch_queue_create=lambda name, mode: object(),
            dispatch_async=lambda queue, call: call(),
        ),
    }
    monkeypatch.setattr(cb, "_DELEGATE", None)
    monkeypatch.setattr(cb.sys, "platform", "darwin")
    monkeypatch.setattr(cb.importlib, "import_module", modules.__getitem__)
    scanner = cb.CoreBluetoothScanner()
    seen = []

    async def run():
        await scanner.start(seen.append)
        assert manager.delegate is not None
        manager.delegate.centralManager_didDiscoverPeripheral_advertisementData_RSSI_(
            manager, Peripheral(), {"name": "advertised"}, -61
        )
        await scanner.stop()

    asyncio.run(run())
    assert manager.scan_args == (None, {"duplicates": True})
    assert manager.options == {"power-alert": False}
    assert manager.stopped
    assert manager.delegate is None
    assert seen[0].local_name == "advertised"
    assert seen[0].rssi == -61


def test_callback_emission_is_synchronous_on_delivery_thread_with_no_event_loop_backlog():
    runtime = FakeCoreBluetooth()
    scanner = cb.CoreBluetoothScanner(runtime_factory=lambda: runtime)
    seen = []
    delivery_threads = []

    def emit(report):
        delivery_threads.append(threading.get_ident())
        seen.append(report)

    async def run():
        await scanner.start(emit)
        worker = threading.Thread(target=runtime.report)
        worker.start()
        worker.join(timeout=1)
        assert not worker.is_alive()
        assert len(seen) == 1
        assert delivery_threads == [worker.ident]
        await scanner.stop()

    asyncio.run(run())


def test_stopped_scanning_and_sink_failure_are_runtime_failures():
    runtime = FakeCoreBluetooth()
    scanner = cb.CoreBluetoothScanner(runtime_factory=lambda: runtime)

    async def run():
        await scanner.start(lambda report: None)
        runtime.is_scanning = False
        with pytest.raises(RuntimeError, match="scanning stopped"):
            scanner.check()
        await scanner.stop()

    asyncio.run(run())
    runtime = FakeCoreBluetooth()
    scanner = cb.CoreBluetoothScanner(runtime_factory=lambda: runtime)

    def broken_sink(report):
        raise ValueError("synthetic sink failure")

    async def run_broken():
        await scanner.start(broken_sink)
        runtime.report()
        with pytest.raises(RuntimeError, match="sink failed"):
            scanner.check()
        await scanner.stop()

    asyncio.run(run_broken())


def test_stale_callbacks_after_restart_do_not_change_new_run_state():
    first, second = FakeCoreBluetooth(), FakeCoreBluetooth()
    runtimes = iter([first, second])
    scanner = cb.CoreBluetoothScanner(runtime_factory=lambda: next(runtimes))
    seen = []

    async def run():
        await scanner.start(seen.append)
        await scanner.stop()
        await scanner.start(seen.append)
        first.report()
        first.state(4)
        scanner.check()
        second.report()
        await scanner.stop()

    asyncio.run(run())
    assert len(seen) == 1


def test_failed_native_open_still_detaches_and_optional_import_failure_is_clear(monkeypatch):
    class FailingOpen(FakeCoreBluetooth):
        def open(self, state, report):
            raise RuntimeError("native initialization failed")

    runtime = FailingOpen()
    scanner = cb.CoreBluetoothScanner(runtime_factory=lambda: runtime)
    with pytest.raises(RuntimeError, match="native initialization failed"):
        asyncio.run(scanner.start(lambda report: None))
    assert runtime.closed
    monkeypatch.setattr(cb.sys, "platform", "darwin")
    missing = create_autospec(cb.importlib.import_module, side_effect=ModuleNotFoundError("optional dependency"))
    monkeypatch.setattr(cb.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="pyobjc-framework-CoreBluetooth"):
        asyncio.run(cb.CoreBluetoothScanner().start(lambda report: None))
