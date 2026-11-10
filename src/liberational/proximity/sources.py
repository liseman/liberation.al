"""ESP32 USB, authorized OS discovery, and bounded replay sources.

ESP32/replay acquisition uses no host Bluetooth or serial-write API. Optional host discovery is explicitly
authorized separately because OS APIs may send scan requests. No source connects, pairs, or reads GATT attributes.
Only the explicitly invoked naming command writes to a local ESP32 sensor, to save its name in NVS.

Autodetection tries Espressif and common ESP32 USB bridges, accepting data only after a valid BLE1 hello.
Configured serial ports take precedence. Opening is exclusive where supported; DTR/RTS are deasserted before
open, so the scanner does not intentionally reset a board. USB-driver glitches can still reset some boards.
"""

import asyncio
import concurrent.futures
import importlib
import json
import queue
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

from liberational import logs
from liberational.proximity.ble import Advertisement
from liberational.proximity.serial_protocol import Framer, Hello, SensorClock, frame, parse

LOG = logs.get("bluetooth")
USB_VENDORS = {0x303A, 0x10C4, 0x1A86, 0x0403}


class AdvertisementSource(Protocol):
    @property
    def stats(self) -> dict[str, int]: ...

    async def run(self, emit: Callable[[Advertisement], None], stop: asyncio.Event) -> None: ...


class HostScanner(Protocol):
    """A selected OS discovery session. Callbacks must synchronously offer bounded immutable reports."""

    stats: dict[str, int]
    metadata: dict[str, str]

    async def start(self, emit: Callable[[Advertisement], None]) -> None: ...

    async def stop(self) -> None: ...

    def check(self) -> None: ...


class HostSource:
    """Bound a host scanner's cross-thread reports without scheduling one event-loop callback per discovery.

    OS discovery can transmit scan requests. This source requires explicit authorization in addition to selecting
    a backend; it is never an automatic fallback for a missing ESP32. Reports count OS callbacks, not packets.
    """

    def __init__(self, scanner: HostScanner, *, allow_active_scan: bool = False, queue_size: int = 512) -> None:
        if not allow_active_scan:
            raise ValueError("Host Bluetooth discovery may send scan requests; set bluetooth.allow_active_scan=true")
        if not 1 <= queue_size <= 4096:
            raise ValueError("Bluetooth host queue size must be between 1 and 4096")
        self.scanner = scanner
        self.pending: queue.Queue[Advertisement] = queue.Queue(queue_size)
        self._counts = dict(
            received_reports=0,
            dropped_reports=0,
            late_reports=0,
            source_failures=0,
            active_sensors=0,
            cleanup_failures=0,
        )
        self._accepting = False
        self._lock = threading.Lock()

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            return {**self.scanner.stats, **self._counts, "queued_reports": self.pending.qsize()}

    @property
    def metadata(self) -> dict[str, str]:
        return dict(self.scanner.metadata)

    def offer(self, report: Advertisement) -> None:
        """Called on either backend thread; no unbounded task or callback is queued on the asyncio loop."""
        with self._lock:
            self._counts["received_reports"] += 1
            if not self._accepting:
                self._counts["dropped_reports"] += 1
                self._counts["late_reports"] += 1
                return
            try:
                self.pending.put_nowait(report)
            except queue.Full:
                self._counts["dropped_reports"] += 1

    def _drain(self, emit: Callable[[Advertisement], None]) -> None:
        for _ in range(256):
            try:
                report = self.pending.get_nowait()
            except queue.Empty:
                break
            emit(report)

    async def run(self, emit: Callable[[Advertisement], None], stop: asyncio.Event) -> None:
        if stop.is_set():
            return
        with self._lock:
            self._accepting = True
        starting = asyncio.create_task(self.scanner.start(self.offer))
        stopping = asyncio.create_task(stop.wait())
        try:
            done, _ = await asyncio.wait((starting, stopping), timeout=15, return_when=asyncio.FIRST_COMPLETED)
            if not done:
                raise TimeoutError("Host Bluetooth discovery startup timed out")
            if starting not in done:
                return  # A normal stop also cancels a pending OS permission/adapter startup.
            await starting
            with self._lock:
                self._counts["active_sensors"] = 1
            while not stop.is_set():
                self.scanner.check()
                self._drain(emit)
                try:
                    await asyncio.wait_for(stop.wait(), timeout=0.05)
                except TimeoutError:
                    pass
        except Exception:
            with self._lock:
                self._counts["source_failures"] += 1
            raise
        finally:
            with self._lock:
                self._accepting = False
                self._counts["active_sensors"] = 0
            starting.cancel()
            stopping.cancel()
            await asyncio.gather(starting, stopping, return_exceptions=True)
            try:
                await asyncio.wait_for(self.scanner.stop(), timeout=5)
            except Exception:
                with self._lock:
                    self._counts["cleanup_failures"] += 1
                raise
            finally:
                while not self.pending.empty():
                    self._drain(emit)


class SerialPort(Protocol):
    def read(self, size: int) -> bytes: ...

    def close(self) -> None: ...

    def write(self, data: bytes) -> int: ...


def serial_ports() -> list[str]:
    ports = importlib.import_module("serial.tools.list_ports")
    return sorted({str(p.device) for p in ports.comports() if p.vid in USB_VENDORS})


def open_serial(path: str, baudrate: int) -> SerialPort:
    serial = importlib.import_module("serial")
    port = serial.Serial(port=None, baudrate=baudrate, timeout=0.2, write_timeout=2, exclusive=True)
    port.dtr = False
    port.rts = False
    port.port = path
    port.open()
    return port


class SerialSource:
    """At most max_sensors read threads, a bounded report queue, and hotplug enumeration every two seconds."""

    def __init__(
        self,
        ports: Sequence[str] = (),
        *,
        names: dict[str, str] | None = None,
        baudrate: int = 115200,
        max_sensors: int = 8,
        queue_size: int = 512,
        discover: Callable[[], list[str]] = serial_ports,
        opener: Callable[[str, int], SerialPort] = open_serial,
        clock: Callable[[], float] = time.time,
        discover_interval_s: float = 2.0,
        hello_timeout_s: float = 7.0,
    ) -> None:
        if not 1 <= max_sensors <= 32 or not 1 <= queue_size <= 4096 or not 1200 <= baudrate <= 2_000_000:
            raise ValueError("invalid serial source bounds")
        if not 0.01 <= discover_interval_s <= 60:
            raise ValueError("invalid USB discovery interval")
        if not 0.01 <= hello_timeout_s <= 60:
            raise ValueError("invalid BLE1 hello deadline")
        self.discover_interval_s, self.hello_timeout_s = discover_interval_s, hello_timeout_s
        self.ports, self.names, self.baudrate = tuple(ports), names or {}, baudrate
        self.max_sensors, self.discover, self.opener, self.clock = max_sensors, discover, opener, clock
        self.pending: queue.Queue[Advertisement] = queue.Queue(queue_size)
        self.stats = dict(
            invalid_lines=0,
            oversized_lines=0,
            dropped_reports=0,
            lost_reports=0,
            reordered_reports=0,
            malformed_payloads=0,
            disconnects=0,
            active_sensors=0,
            hello_timeouts=0,
        )
        self._lock = threading.Lock()
        self.sensors: dict[str, str] = {}  # each attached sensor's port, and "Name (CHIPID)" from its hello

    def describe(self) -> str:
        """What this source reads, for the log: its ports, or how it finds them."""
        return f"ESP32 sensors on {', '.join(self.ports)}" if self.ports else "ESP32 sensors found on USB"

    def _count(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self.stats[name] += amount

    def read_port(self, path: str, stop: threading.Event) -> None:
        """One connection. Unverified bridges time out so a non-sensor cannot monopolize detection forever.

        The default seven-second deadline spans the firmware's five-second HELLO interval. Verified sensors stay
        attached until stopped or disconnected; reopening starts a fresh hello/clock session.
        """
        port: SerialPort | None = None
        framer, sensor = Framer(), SensorClock(self.names)
        active = False
        reported = dict(oversized_lines=0, lost_reports=0, reordered_reports=0, malformed_payloads=0)

        def sync_stats() -> None:
            """Publish per-connection deltas while attached; final cleanup must not count them twice."""
            current = dict(
                oversized_lines=framer.invalid,
                lost_reports=sensor.lost_reports,
                reordered_reports=sensor.reordered_reports,
                malformed_payloads=sensor.malformed_payloads,
            )
            for name, count in current.items():
                delta = count - reported[name]
                if delta:
                    self._count(name, delta)
                    reported[name] = count

        try:
            port = self.opener(path, self.baudrate)
            hello_deadline = time.monotonic() + self.hello_timeout_s
            while not stop.is_set():
                if not active and time.monotonic() >= hello_deadline:
                    self._count("hello_timeouts")
                    return
                data = port.read(1024)
                if not data:
                    continue
                receipt = self.clock()
                for line in framer.feed(data):
                    try:
                        ad = sensor.accept(parse(line), receipt)
                        if sensor.hello is not None and not active:
                            active = True
                            self._count("active_sensors")
                            hello = sensor.hello
                            label = f"{self.names.get(hello.chip_id) or hello.name} ({hello.chip_id})"
                            with self._lock:
                                self.sensors[path] = label
                            LOG.info("sensor %s attached on %s", label, path)
                    except ValueError:
                        self._count("invalid_lines")
                        continue  # Includes boot logs; no identifier is accepted before a valid hello.
                    if ad is not None:
                        try:
                            self.pending.put_nowait(ad)
                        except queue.Full:
                            self._count("dropped_reports")
                sync_stats()
        except Exception as e:
            self._count("disconnects")
            LOG.warning("serial sensor %s unavailable (%s); retrying", path, type(e).__name__)
        finally:
            if port is not None:
                try:
                    port.close()
                except Exception as e:
                    LOG.warning("serial sensor close failed (%s)", type(e).__name__)
            if active:
                self._count("active_sensors", -1)
                with self._lock:
                    label = self.sensors.pop(path, path)
                LOG.info("sensor %s detached from %s", label, path)
            sync_stats()

    async def run(self, emit: Callable[[Advertisement], None], stop: asyncio.Event) -> None:
        # Long-lived blocking reads must not consume the executor needed for periodic USB enumeration.
        readers = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_sensors, thread_name_prefix="ble-sensor")
        loop = asyncio.get_running_loop()
        connections: dict[str, tuple[threading.Event, asyncio.Future[None]]] = {}
        next_discovery = 0.0
        attempt_cursor = 0
        try:
            while not stop.is_set():
                now = time.monotonic()
                if now >= next_discovery:
                    next_discovery = now + self.discover_interval_s
                    try:
                        available = list(self.ports) if self.ports else await asyncio.to_thread(self.discover)
                    except Exception as e:
                        available = []
                        LOG.warning("USB sensor enumeration unavailable (%s)", type(e).__name__)
                    available = list(dict.fromkeys(available))
                    wanted = set(available)
                    for path, (quit_port, task) in list(connections.items()):
                        if path not in wanted:
                            quit_port.set()
                        if task.done():
                            await task
                            del connections[path]
                    # Round-robin unclaimed candidates. A timed-out non-sensor must not keep later boards out,
                    # and adding an earlier-sorted USB bridge must not evict an already verified sensor.
                    for _ in range(len(available)):
                        if len(connections) >= self.max_sensors:
                            break
                        path = available[attempt_cursor % len(available)]
                        attempt_cursor += 1
                        if path in connections:
                            continue
                        quit_port = threading.Event()
                        connections[path] = (
                            quit_port,
                            loop.run_in_executor(readers, self.read_port, path, quit_port),
                        )
                for _ in range(256):
                    try:
                        ad = self.pending.get_nowait()
                    except queue.Empty:
                        break
                    emit(ad)
                try:
                    await asyncio.wait_for(stop.wait(), timeout=0.05)
                except TimeoutError:
                    pass
        finally:
            for quit_port, _ in connections.values():
                quit_port.set()
            try:
                await asyncio.gather(*(task for _, task in connections.values()))
            finally:
                readers.shutdown(wait=False, cancel_futures=True)
            while not self.pending.empty():
                emit(self.pending.get_nowait())


class ReplaySource:
    """Bounded JSONL replay: {received_at: epoch seconds, line: a complete BLE1 ASCII frame}. No real I/O devices."""

    def __init__(self, path: Path, names: dict[str, str] | None = None, max_sensors: int = 8) -> None:
        if not 1 <= max_sensors <= 32:
            raise ValueError("max_sensors must be between 1 and 32")
        self.path, self.names, self.max_sensors = path, names or {}, max_sensors
        self.stats = dict(invalid_lines=0, dropped_reports=0, lost_reports=0, reordered_reports=0, malformed_payloads=0)

    async def run(self, emit: Callable[[Advertisement], None], stop: asyncio.Event) -> None:
        sensors: dict[str, SensorClock] = {}
        with self.path.open("rb") as stream:
            while not stop.is_set():
                raw = stream.readline(8193)
                if not raw:
                    break
                if len(raw) > 8192:
                    raise ValueError("replay line exceeds 8192 bytes")
                try:
                    row: Any = json.loads(raw)
                    if not isinstance(row, dict) or not isinstance(row.get("line"), str):
                        raise ValueError("replay rows must contain an ASCII frame string")
                    message = parse(row["line"].encode("ascii"))
                    if message.chip_id not in sensors:
                        if len(sensors) >= self.max_sensors:
                            self.stats["dropped_reports"] += 1
                            continue
                        sensors[message.chip_id] = SensorClock(self.names)
                    ad = sensors[message.chip_id].accept(message, float(row["received_at"]))
                    if ad is not None:
                        emit(replace(ad, backend="replay", scan_mode="replay"))
                except ValueError, KeyError, TypeError, UnicodeError:
                    self.stats["invalid_lines"] += 1
                await asyncio.sleep(0)  # Let the publisher drain while a large replay is read.
        for sensor in sensors.values():
            for name in ("lost_reports", "reordered_reports", "malformed_payloads"):
                self.stats[name] += getattr(sensor, name)


def set_sensor_name(path: str, name: str, opener: Callable[[str, int], SerialPort] = open_serial) -> None:
    """Explicit local serial command, confirmed by the next checksummed hello; never used while scanning."""
    raw = name.encode("utf-8")
    if not raw or len(raw) > 32 or any(ord(c) < 32 or ord(c) == 127 for c in name):
        raise ValueError("sensor name must be 1 through 32 UTF-8 bytes without control characters")
    port = opener(path, 115200)
    try:
        wire = frame(["NAME1", raw.hex()])
        if port.write(wire) != len(wire):
            raise OSError("sensor naming command was not fully written")
        framer, deadline = Framer(), time.monotonic() + 6
        while time.monotonic() < deadline:
            for line in framer.feed(port.read(1024)):
                try:
                    message = parse(line)
                except ValueError:
                    continue
                if isinstance(message, Hello) and message.name == name:
                    return
        raise TimeoutError("sensor did not confirm its saved name")
    finally:
        port.close()
