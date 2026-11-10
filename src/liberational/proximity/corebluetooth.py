"""Explicitly selected macOS BLE observations, using a private serial dispatch queue.

CoreBluetooth manages active scanning and exposes per-app UUIDs, not Bluetooth addresses. It may merge/cache
advertisement properties. No peripheral objects or names are cached here. Native imports and manager creation
happen only in start(); callbacks copy bounded advertised fields directly into the caller's bounded sink.

Queue setup follows PyObjC libdispatch and Bleak's CoreBluetooth delegate implementation. There is no Python
worker thread or callback-to-asyncio task queue. Stop detaches the delegate on the same serial dispatch queue.
"""

import asyncio
import importlib
import sys
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from itertools import islice
from typing import Any, Protocol

from liberational.proximity.advertising import platform_data
from liberational.proximity.ble import Advertisement


class Runtime(Protocol):
    """Injected native boundary. All delivery is serialized; close acknowledges delegate detachment."""

    def open(self, state: Callable[[int], None], report: Callable[[str, Mapping[str, Any], int], None]) -> None: ...
    def scan(self) -> None: ...
    def scanning(self) -> bool: ...
    def close(self, done: threading.Event) -> None: ...
    def check(self) -> None: ...


def _snapshot(identifier: str, fields: Mapping[str, Any], rssi: int, adapter: str, now: float) -> Advertisement:
    data = platform_data(
        local_name=fields.get("local_name"),
        manufacturer_bytes=fields.get("manufacturer"),
        service_data=fields.get("service_data"),
        service_uuids=fields.get("service_uuids", ()),
        tx_power=fields.get("tx_power"),
    )
    return Advertisement(
        observed_at=now,
        received_at=now,
        backend="corebluetooth",
        adapter=adapter,
        identifier_kind="corebluetooth_uuid",
        identifier=str(uuid.UUID(identifier[:128])),
        rssi=rssi if -127 <= rssi <= 20 else None,
        local_name=data.local_name,
        manufacturer_data=data.manufacturer_data,
        service_data=data.service_data,
        service_uuids=data.service_uuids,
        tx_power=data.tx_power,
        scan_mode="os_managed_active",
        raw_payload=None,
        payload_malformed=data.malformed,
        source_payload_truncated=data.truncated or bool(fields.get("_truncated", False)),
    )


class CoreBluetoothScanner:
    """A lifecycle-controlled scanner; the caller must authorize OS-managed active scanning before start().

    adapter is an observation label. CoreBluetooth offers no public controller-selection API, so it does not
    select or power an adapter. Startup waits for poweredOn and isScanning, bounded by startup_timeout_s.
    """

    def __init__(
        self,
        adapter: str = "",
        *,
        runtime_factory: Callable[[], Runtime] | None = None,
        startup_timeout_s: float = 15.0,
        stop_timeout_s: float = 2.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if startup_timeout_s <= 0 or stop_timeout_s <= 0:
            raise ValueError("CoreBluetooth lifecycle timeouts must be positive")
        if len(adapter.encode()) > 128:
            raise ValueError("CoreBluetooth adapter label exceeds 128 bytes")
        self.metadata = {
            "backend": "corebluetooth",
            "adapter": adapter or "system-default",
            "scan_mode": "os_managed_active",
        }
        self.stats: dict[str, int] = {"reports": 0, "invalid_reports": 0}
        self._factory = runtime_factory or _NativeRuntime
        self._startup_timeout = startup_timeout_s
        self._stop_timeout = stop_timeout_s
        self._clock = clock
        self._lock = threading.RLock()
        self._runtime: Runtime | None = None
        self._emit: Callable[[Advertisement], None] | None = None
        self._error: RuntimeError | None = None
        self._powered_on = False
        self._started = False
        self._generation = 0
        self._closing: threading.Event | None = None

    def _state(self, state: int, generation: int) -> None:
        with self._lock:
            if generation != self._generation or self._runtime is None:
                return
            if state == 5:
                self._powered_on = True
            elif state != 0 or self._powered_on:
                reason = {
                    0: "unknown",
                    1: "resetting",
                    2: "unsupported",
                    3: "unauthorized: grant Bluetooth permission in macOS Privacy settings",
                    4: "powered off",
                }.get(state, "unrecognized state")
                self._error = RuntimeError(f"CoreBluetooth is {reason}")
                self._emit = None

    def _report(self, identifier: str, fields: Mapping[str, Any], rssi: int, generation: int) -> None:
        with self._lock:
            if generation != self._generation or self._emit is None or self._error is not None:
                return
            try:
                report = _snapshot(identifier, fields, rssi, self.metadata["adapter"], self._clock())
            except ValueError, TypeError, OverflowError, BufferError:
                self.stats["invalid_reports"] += 1
                return
            try:
                self._emit(report)
                self.stats["reports"] += 1
            except Exception:
                self._error = RuntimeError("CoreBluetooth observation sink failed")
                self._emit = None

    async def start(self, emit: Callable[[Advertisement], None]) -> None:
        with self._lock:
            if self._runtime is not None:
                raise RuntimeError("CoreBluetooth scanner is already started")
            self._error = None
            self._powered_on = False
            self._generation += 1
            generation = self._generation
            runtime = self._factory()
            self._runtime = runtime
            self._emit = emit
        try:
            runtime.open(
                lambda state: self._state(state, generation),
                lambda identifier, fields, rssi: self._report(identifier, fields, rssi, generation),
            )
            deadline = asyncio.get_running_loop().time() + self._startup_timeout
            requested = False
            while True:
                self.check()
                with self._lock:
                    powered_on = self._powered_on
                if powered_on and not requested:
                    runtime.scan()
                    requested = True
                if requested and runtime.scanning():
                    with self._lock:
                        self.check()
                        self._started = True
                    return
                if asyncio.get_running_loop().time() >= deadline:
                    raise RuntimeError(
                        "CoreBluetooth startup timed out waiting for powered-on scanning and Bluetooth permission"
                    )
                await asyncio.sleep(0.01)
        except BaseException:
            await self.stop()
            raise

    def check(self) -> None:
        with self._lock:
            if self._error is not None:
                raise self._error
            if self._runtime is not None:
                self._runtime.check()
                if self._started and not self._runtime.scanning():
                    raise RuntimeError("CoreBluetooth scanning stopped unexpectedly")

    async def stop(self) -> None:
        with self._lock:
            self._generation += 1
            self._emit = None
            self._started = False
            runtime = self._runtime
            if runtime is None:
                return
            if self._closing is None:
                self._closing = threading.Event()
                runtime.close(self._closing)
            done = self._closing
        deadline = asyncio.get_running_loop().time() + self._stop_timeout
        while not done.is_set():
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError("CoreBluetooth stop timed out detaching its native delegate")
            await asyncio.sleep(0.01)
        with self._lock:
            if self._runtime is runtime:
                self._runtime = None
                self._closing = None
        runtime.check()


class _NativeRuntime:
    """Lazy PyObjC bridge. GCD owns callback threads; no main-loop pumping or Python worker is needed."""

    _cb: Any
    _objc: Any
    _foundation: Any
    _dispatch: Any

    def __init__(self) -> None:
        if sys.platform != "darwin":
            raise RuntimeError("CoreBluetooth requires macOS")
        try:
            self._cb = importlib.import_module("CoreBluetooth")
            self._objc = importlib.import_module("objc")
            self._foundation = importlib.import_module("Foundation")
            self._dispatch = importlib.import_module("libdispatch")
        except ImportError as error:
            raise RuntimeError(
                "CoreBluetooth requires optional pyobjc-framework-CoreBluetooth and pyobjc-framework-libdispatch"
            ) from error
        self._queue: Any = None
        self._manager: Any = None
        self._delegate: Any = None
        self._failure: RuntimeError | None = None

    def open(self, state: Callable[[int], None], report: Callable[[str, Mapping[str, Any], int], None]) -> None:
        cb, objc = self._cb, self._objc
        # Reuse one Objective-C class: PyObjC class names are process-global even after a scanner stops.
        delegate_class = _delegate_class(self._foundation.NSObject, objc)
        self._delegate = delegate_class.alloc().init()
        self._delegate.state_callback = state
        self._delegate.report_callback = report
        self._delegate.objc_module = objc
        self._delegate.cb_module = cb
        self._delegate.failure_callback = self._failed
        self._queue = self._dispatch.dispatch_queue_create(
            b"liberational.corebluetooth", self._dispatch.DISPATCH_QUEUE_SERIAL
        )
        self._manager = cb.CBCentralManager.alloc().initWithDelegate_queue_options_(
            self._delegate, self._queue, {cb.CBCentralManagerOptionShowPowerAlertKey: False}
        )

    def _failed(self) -> None:
        self._failure = RuntimeError("CoreBluetooth native callback failed")

    def check(self) -> None:
        if self._failure is not None:
            raise self._failure

    def scan(self) -> None:
        def scan_on_queue() -> None:
            try:
                self._manager.scanForPeripheralsWithServices_options_(
                    None, {self._cb.CBCentralManagerScanOptionAllowDuplicatesKey: True}
                )
            except Exception:
                self._failed()

        self._dispatch.dispatch_async(self._queue, scan_on_queue)

    def scanning(self) -> bool:
        return bool(self._manager.isScanning())

    def close(self, done: threading.Event) -> None:
        if self._queue is None:
            done.set()
            return

        def close_on_queue() -> None:
            try:
                if self._manager is not None:
                    try:
                        self._manager.stopScan()
                    finally:
                        self._manager.setDelegate_(None)
            except Exception:
                self._failed()
            finally:
                if self._delegate is not None:
                    self._delegate.state_callback = None
                    self._delegate.report_callback = None
                self._manager = self._delegate = self._queue = None
                done.set()

        self._dispatch.dispatch_async(self._queue, close_on_queue)


_DELEGATE: Any = None


def _delegate_class(base: Any, objc: Any) -> Any:
    global _DELEGATE
    if _DELEGATE is not None:
        return _DELEGATE

    class LiberationalBluetoothDelegate(base, protocols=[objc.protocolNamed("CBCentralManagerDelegate")]):
        def centralManagerDidUpdateState_(self, manager: Any) -> None:
            try:
                callback = self.state_callback
                if callback is not None:
                    callback(int(manager.state()))
            except Exception:
                self.failure_callback()

        def centralManager_didDiscoverPeripheral_advertisementData_RSSI_(
            self, manager: Any, peripheral: Any, data: Any, rssi: Any
        ) -> None:
            try:
                callback = self.report_callback
                if callback is None:
                    return
                with self.objc_module.autorelease_pool():
                    cb = self.cb_module
                    service_data = data.get(cb.CBAdvertisementDataServiceDataKey, {})
                    service_uuids = data.get(cb.CBAdvertisementDataServiceUUIDsKey, ())
                    # Advertised fields only: never peripheral.name(), readRSSI(), or GATT.
                    fields = {
                        "local_name": data.get(cb.CBAdvertisementDataLocalNameKey),
                        "manufacturer": data.get(cb.CBAdvertisementDataManufacturerDataKey),
                        "service_data": {key.UUIDString(): value for key, value in islice(service_data.items(), 64)},
                        "service_uuids": tuple(value.UUIDString() for value in islice(service_uuids, 64)),
                        "tx_power": data.get(cb.CBAdvertisementDataTxPowerLevelKey),
                        "_truncated": len(service_data) > 64 or len(service_uuids) > 64,
                    }
                    callback(str(peripheral.identifier().UUIDString()), fields, int(rssi))
            except Exception:
                self.failure_callback()

    _DELEGATE = LiberationalBluetoothDelegate
    return _DELEGATE
