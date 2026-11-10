"""Off-by-default BLE sensor service, writing bounded observations and coverage to the segment store (#1755).

USB sensors only receive BLE advertisements. Optional authorized host discovery may send scan requests; no address
is connected, paired, queried through GATT, or mapped to a person by acquisition. Native identifiers stay separate.
Every queue and cache is bounded. Coverage reports count losses, so saturation or a missing sensor is not absence.
"""

import argparse
import asyncio
import logging
import math
import signal
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

from liberational import config as cfg
from liberational import logs, segment_log
from liberational.proximity.ble import Aggregator
from liberational.proximity.sources import (
    AdvertisementSource,
    HostScanner,
    HostSource,
    ReplaySource,
    SerialSource,
    set_sensor_name,
)

LOG = logs.get("bluetooth")
SIGHTINGS = segment_log.BLE_SIGHTINGS
COVERAGE = segment_log.BLE_COVERAGE
CAMERA = "ble"
REPORT_S = 600.0  # how often a running service says what it recorded, so a healthy run shows in bluetooth.log


class Sink(Protocol):
    async def start(self) -> None: ...

    async def publish(self, stream: str, record: dict[str, Any]) -> None: ...


class SegmentSink:
    """Bluetooth's sightings and coverage in the segment store (#1755), under the camera name "ble"; each coverage
    record (one a second) syncs what came before it."""

    def __init__(self, root: Path) -> None:
        self.writers = segment_log.Writers(root)

    async def start(self) -> None:
        pass

    async def publish(self, stream: str, record: dict[str, Any]) -> None:
        self.writers.append(stream, CAMERA, [record])
        if stream == COVERAGE:
            self.writers.sync()

    def close(self) -> None:
        self.writers.close()


class Worker:
    def __init__(
        self,
        source: AdvertisementSource,
        sink: Sink,
        *,
        window_s: float = 5,
        max_devices: int = 1024,
        max_records_per_second: int = 50,
        max_payload_bytes: int = 4096,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 3 <= max_records_per_second <= 500:
            raise ValueError("Bluetooth record rate must be between 3 and 500")
        self.source, self.sink = source, sink
        self.aggregates = Aggregator(window_s=window_s, max_devices=max_devices, max_payload_bytes=max_payload_bytes)
        self.rate = max_records_per_second
        self.published = 0
        self.unacknowledged_sightings = 0
        self.clock, self.next_flush = clock, -float("inf")
        self.reported_at, self.reported_published, self.reported_gaps, self.reported_dropped = clock(), 0, 0, 0

    async def flush(self, now: float, *, force: bool = False) -> None:
        moment = self.clock()
        if moment < self.next_flush:
            return
        self.next_flush = moment + 1.0
        # Reserve one record for coverage; sightings take the rest of the cap.
        batch = self.aggregates.drain(now, max(0, self.rate - 1), force=force)
        for index, record in enumerate(batch):
            try:
                await asyncio.wait_for(self.sink.publish(SIGHTINGS, record), timeout=5)
            except Exception, asyncio.CancelledError:
                self.unacknowledged_sightings += len(batch) - index
                raise
            self.published += 1
        await asyncio.wait_for(
            self.sink.publish(
                COVERAGE,
                dict(
                    id=str(uuid.uuid7()),
                    schema_version=1,
                    start=now,
                    end=now,
                    sightings_published=self.published,
                    source=dict(self.source.stats),
                    acquisition=dict(getattr(self.source, "metadata", {})),
                    aggregation=self.aggregates.stats,
                    absence_is_not_inferred=True,
                    unacknowledged_sightings=self.unacknowledged_sightings,
                ),
            ),
            timeout=5,
        )
        self.next_flush = self.clock() + 1.0  # A slow commit cannot make the next batch burst immediately.
        self._report()

    def _losses(self) -> tuple[int, int]:
        """Reports never recorded, so far: (over a sensor's output cap, dropped on this machine).

        A USB sensor counts every report it hears, so its counter gaps (`lost_reports`) are reports it heard but
        didn't send: more than its output rate, or cut by a reset. Host discovery's `lost_reports` are unreadable
        reports instead, already in `invalid_reports`. Only the second number is loss on this machine.
        """
        stats = dict(self.source.stats)
        dropped = int(stats.get("dropped_reports", 0)) + int(self.aggregates.stats["dropped_capacity"])
        if isinstance(self.source, HostSource):
            return 0, dropped + int(stats.get("invalid_reports", 0))
        return int(stats.get("lost_reports", 0)), dropped

    def _report(self, *, final: bool = False) -> None:
        """Every REPORT_S, what was recorded since the last report: sightings, reports the sensors didn't send, reports
        dropped here (a warning when any were), and the sensors on."""
        if not final and self.clock() - self.reported_at < REPORT_S:
            return
        stats = dict(self.source.stats)
        gaps, dropped = self._losses()
        sensors = sorted(getattr(self.source, "sensors", {}).values())
        LOG.log(
            logging.WARNING if dropped > self.reported_dropped else logging.INFO,
            "Bluetooth: %d sightings in %.0f min; %d reports over the sensors' output cap, %d dropped here; "
            "%s sensors on%s",
            self.published - self.reported_published,
            (self.clock() - self.reported_at) / 60,
            gaps - self.reported_gaps,
            dropped - self.reported_dropped,
            stats.get("active_sensors", len(sensors)),
            f": {', '.join(sensors)}" if sensors else "",
        )
        self.reported_at, self.reported_published = self.clock(), self.published
        self.reported_gaps, self.reported_dropped = gaps, dropped

    async def run(self, stop: asyncio.Event) -> None:
        await self.sink.start()
        describe = getattr(self.source, "describe", None)
        LOG.info(
            "Bluetooth started: %s; a summary every %.0f min",
            describe() if callable(describe) else type(self.source).__name__,
            REPORT_S / 60,
        )
        source_stop = asyncio.Event()
        source = asyncio.create_task(self.source.run(self.aggregates.add, source_stop))
        stopping = asyncio.create_task(stop.wait())
        failed = False
        try:
            while not stop.is_set():
                done, _ = await asyncio.wait([source, stopping], timeout=1, return_when=asyncio.FIRST_COMPLETED)
                await self.flush(time.time(), force=source in done)
                if source in done:
                    await source  # A source failure must stop the service, not silently leave an empty scanner.
                    break
        except Exception, asyncio.CancelledError:
            failed = True
            LOG.error(
                "Bluetooth stopped with incomplete evidence: unacknowledged=%d, pending=%d",
                self.unacknowledged_sightings,
                self.aggregates.stats["pending_devices"],
            )
            raise
        finally:
            source_stop.set()
            stopping.cancel()
            await asyncio.gather(source, stopping, return_exceptions=True)
            # One global deadline bounds graceful draining.
            if not failed:
                try:
                    async with asyncio.timeout(5):
                        while self.aggregates.stats["pending_devices"]:
                            await self.flush(time.time(), force=True)
                            if self.aggregates.stats["pending_devices"]:
                                await asyncio.sleep(1)
                except TimeoutError:
                    LOG.warning("Bluetooth shutdown drain timed out; evidence remains incomplete")
            LOG.info(
                "Bluetooth stopped; %d sightings submitted, aggregation=%s, source=%s",
                self.published,
                self.aggregates.stats,
                self.source.stats,
            )


def source_from_config() -> AdvertisementSource:
    source = str(cfg.get("bluetooth.source"))
    if source in {"bluez", "corebluetooth"}:
        if not cfg.get("bluetooth.allow_active_scan"):
            raise ValueError("Host Bluetooth discovery may send scan requests; set bluetooth.allow_active_scan=true")
        adapter = str(cfg.get("bluetooth.host_adapter"))
        scanner: HostScanner
        if source == "bluez":
            if not sys.platform.startswith("linux"):
                raise ValueError("bluetooth.source=bluez requires Linux; ESP32 serial works on either platform")
            from liberational.proximity.bluez import BlueZScanner

            scanner = BlueZScanner(adapter=adapter, max_devices=int(cfg.get("bluetooth.max_devices")))
        else:
            if sys.platform != "darwin":
                raise ValueError("bluetooth.source=corebluetooth requires macOS; ESP32 serial works on either platform")
            from liberational.proximity.corebluetooth import CoreBluetoothScanner

            scanner = CoreBluetoothScanner(adapter=adapter)
        return HostSource(scanner, allow_active_scan=True, queue_size=int(cfg.get("bluetooth.host_queue_size")))
    settings = cfg.get("bluetooth.sensor_names")
    if not isinstance(settings, list) or any(not isinstance(x, str) or "=" not in x for x in settings):
        raise ValueError("bluetooth.sensor_names must contain CHIPID=name strings")
    names = dict(x.split("=", 1) for x in settings)
    if len(names) != len(settings):
        raise ValueError("duplicate Bluetooth sensor name override")
    if len(names) > 128 or any(len(k) > 64 or len(v.encode()) > 64 for k, v in names.items()):
        raise ValueError("Bluetooth sensor names exceed the configured bounds")
    if source == "replay":
        path = str(cfg.get("bluetooth.replay_path"))
        if not path:
            raise ValueError("bluetooth.replay_path is required for replay")
        return ReplaySource(cfg.path("bluetooth.replay_path"), names, int(cfg.get("bluetooth.max_sensors")))
    if source != "esp32":
        raise ValueError("bluetooth.source must be esp32, replay, bluez, or corebluetooth")
    ports = cfg.get("bluetooth.serial_ports")
    if not isinstance(ports, list) or any(not isinstance(p, str) or not p or len(p) > 1024 for p in ports):
        raise ValueError("bluetooth.serial_ports must be serial device paths")
    return SerialSource(
        ports,
        names=names,
        baudrate=int(cfg.get("bluetooth.baudrate")),
        max_sensors=int(cfg.get("bluetooth.max_sensors")),
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pack-bluetooth", description=__doc__)
    parser.add_argument("--set-name", help="explicitly save this name on a local ESP32 sensor")
    parser.add_argument("--port", help="serial port for --set-name")
    args = parser.parse_args(argv)
    if args.set_name is not None:
        if not args.port:
            parser.error("--set-name requires --port")
        set_sensor_name(args.port, args.set_name)
        print("Sensor name saved and confirmed.")
        return
    if args.port:
        parser.error("--port is only used with --set-name; acquisition ports come from bluetooth.serial_ports")
    if not cfg.get("bluetooth.enabled"):
        LOG.info("Bluetooth sensing is disabled")
        return
    window = float(cfg.get("bluetooth.aggregate_seconds"))
    if not math.isfinite(window) or not 0.1 <= window <= 300:
        raise SystemExit("bluetooth.aggregate_seconds must be between 0.1 and 300")
    source = source_from_config()

    async def run() -> None:
        stop = asyncio.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_running_loop().add_signal_handler(sig, stop.set)
        sink = SegmentSink(segment_log.root())
        worker = Worker(
            source,
            sink,
            window_s=window,
            max_devices=int(cfg.get("bluetooth.max_devices")),
            max_records_per_second=int(cfg.get("bluetooth.max_records_per_second")),
        )
        try:
            await worker.run(stop)
        finally:
            sink.close()

    asyncio.run(run())


if __name__ == "__main__":
    main()
