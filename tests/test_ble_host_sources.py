import asyncio
import threading
from collections.abc import Callable

import pytest

from liberational.proximity.ble import Advertisement
from liberational.proximity.sources import HostSource
from liberational.proximity.worker import Worker
from test_ble_worker import Sink


def report(at=100):
    return Advertisement(at, "bluez", "hci0", "bluetooth_address", "C0:00:00:00:00:01", scan_mode="os_managed_active")


class Scanner:
    def __init__(self, action=None, failure=False):
        self.action = action
        self.failure = failure
        self.stats = {"invalid_reports": 0}
        self.metadata = {"backend": "bluez", "adapter": "hci0", "scan_mode": "os_managed_active"}
        self.started = self.closed = False

    async def start(self, emit):
        self.started = True
        if self.action:
            self.action(emit)

    async def stop(self):
        self.closed = True

    def check(self):
        if self.failure:
            raise RuntimeError("adapter disappeared")


def test_host_source_requires_authorization_before_starting_a_scanner():
    scanner = Scanner()
    with pytest.raises(ValueError, match="allow_active_scan"):
        HostSource(scanner)
    assert not scanner.started


def test_host_report_queue_bounds_a_native_thread_burst_and_preserves_accepted_reports():
    async def run():
        stop = asyncio.Event()

        def burst(emit: Callable[[Advertisement], None]) -> None:
            def send() -> None:
                for n in range(20):
                    emit(report(100 + n))

            thread = threading.Thread(target=send)
            thread.start()
            thread.join()
            stop.set()

        scanner = Scanner(burst)
        source = HostSource(scanner, allow_active_scan=True, queue_size=2)
        found = []
        await source.run(found.append, stop)
        assert [ad.observed_at for ad in found] == [100, 101]
        assert source.stats["received_reports"] == 20
        assert source.stats["dropped_reports"] == 18
        assert source.stats["queued_reports"] == 0
        assert source.stats["active_sensors"] == 0
        assert source.metadata["scan_mode"] == "os_managed_active"
        assert scanner.closed

    asyncio.run(run())


def test_adapter_failure_stops_source_instead_of_reporting_a_successful_empty_scanner():
    scanner = Scanner(lambda emit: emit(report()), failure=True)
    source = HostSource(scanner, allow_active_scan=True)
    found = []
    with pytest.raises(RuntimeError, match="adapter disappeared"):
        asyncio.run(source.run(found.append, asyncio.Event()))
    assert len(found) == 1 and scanner.closed
    assert source.stats["source_failures"] == 1
    assert source.stats["active_sensors"] == 0


def test_failed_start_still_closes_partially_constructed_scanner():
    def fail(emit):
        raise RuntimeError("permission denied")

    scanner = Scanner(fail)
    source = HostSource(scanner, allow_active_scan=True)
    with pytest.raises(RuntimeError, match="permission denied"):
        asyncio.run(source.run(lambda ad: None, asyncio.Event()))
    assert scanner.closed and source.stats["active_sensors"] == 0


def test_normal_stop_interrupts_pending_native_startup_and_closes_the_scanner():
    async def run():
        entered, stop = asyncio.Event(), asyncio.Event()

        class WaitingScanner(Scanner):
            async def start(self, emit):
                entered.set()
                await asyncio.Event().wait()

        scanner = WaitingScanner()
        source = HostSource(scanner, allow_active_scan=True)
        running = asyncio.create_task(source.run(lambda ad: None, stop))
        await entered.wait()
        stop.set()
        await asyncio.wait_for(running, timeout=0.5)
        assert scanner.closed and source.stats["active_sensors"] == 0
        assert source.stats["source_failures"] == 0

    asyncio.run(run())


def test_shutdown_rejects_late_native_reports_and_does_not_start_after_stop():
    async def run():
        stop = asyncio.Event()
        stop.set()
        scanner = Scanner()
        source = HostSource(scanner, allow_active_scan=True)
        found = []
        await source.run(found.append, stop)
        source.offer(report())
        assert not scanner.started and not found
        assert source.stats["late_reports"] == source.stats["dropped_reports"] == 1

    asyncio.run(run())


@pytest.mark.parametrize(
    "backend,kind,identifier",
    [
        ("bluez", "bluetooth_address", "C0:00:00:00:00:01"),
        ("corebluetooth", "corebluetooth_uuid", "0190c8e0-0000-7000-8000-000000000001"),
    ],
)
def test_host_reports_publish_through_the_shared_worker_with_native_identity_and_scan_mode(backend, kind, identifier):
    async def run():
        stop = asyncio.Event()

        def observations(emit):
            emit(Advertisement(100, backend, "local-adapter", kind, identifier, scan_mode="os_managed_active"))
            emit(Advertisement(101, backend, "local-adapter", kind, identifier, scan_mode="os_managed_active"))
            stop.set()

        scanner = Scanner(observations)
        scanner.metadata = {"backend": backend, "adapter": "local-adapter", "scan_mode": "os_managed_active"}
        source = HostSource(scanner, allow_active_scan=True)
        sink = Sink()
        worker = Worker(source, sink)
        await source.run(worker.aggregates.add, stop)
        await worker.flush(106, force=True)
        sightings = [row for stream, row in sink.rows if stream == "bluetooth.sighting"]
        assert len(sightings) == 1
        assert sightings[0]["backend"] == backend
        assert sightings[0]["identifier_kind"] == kind and sightings[0]["identifier"] == identifier
        assert sightings[0]["reports"] == 2 and sightings[0]["scan_mode"] == "os_managed_active"
        assert sightings[0]["raw_payload"] is None
        coverage = next(row for stream, row in sink.rows if stream == "bluetooth.coverage")
        assert coverage["acquisition"] == {
            "backend": backend,
            "adapter": "local-adapter",
            "scan_mode": "os_managed_active",
        }
        assert coverage["absence_is_not_inferred"] is True

    asyncio.run(run())
