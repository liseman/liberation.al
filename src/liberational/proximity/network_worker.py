"""Slow, read-only local network inventory into the segment store, with partial host-cache fallback.

No probes, DNS lookup, cloud API, alerts, or person/thing linking. Each poll writes one `network.device_observation`
row per device it saw and one `network.inventory` row for the poll, under the poll's source ("unifi" or
"host_network") as the segment camera. Complete controller inventories are distinct from cached ARP/DHCP
observations: only a complete, authoritative poll can support absence (proximity/devices.py).

Each poll also reads the Wi-Fi access points this machine hears (proximity/wifi.py, passive) and writes them as
`wifi.access_point` rows when the set changes, or every `segment_log.WIFI_REWRITE_SECONDS` while it doesn't.
"""

import argparse
import asyncio
import json
import math
import signal
import sys
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Protocol

from liberational import config as cfg
from liberational import logs, segment_log
from liberational.proximity.host_network import HostNetworkSource
from liberational.proximity.network import NetworkInventory
from liberational.proximity.unifi_network import UniFiNetwork
from liberational.proximity.wifi import AccessPoint, Scanner, scanner
from liberational.unifi import UniFiSession, configured_session

LOG = logs.get("network-devices")
OBSERVATIONS = segment_log.NET_OBSERVATIONS
INVENTORY = segment_log.NET_INVENTORY
WIFI = segment_log.WIFI_ACCESS_POINTS


class InventorySource(Protocol):
    def poll(self, now: float) -> NetworkInventory: ...


class Publisher(Protocol):
    def write(self, stream: str, source: str, records: Sequence[dict[str, Any]]) -> None: ...

    def close(self) -> None: ...


class NetworkSource:
    def __init__(self, host: InventorySource, unifi: InventorySource | None = None) -> None:
        self.host, self.unifi = host, unifi

    def poll(self, now: float) -> NetworkInventory:
        if self.unifi is None:
            return self.host.poll(now)
        try:
            result = self.unifi.poll(now)
        except Exception as e:
            # Never put exception strings (possibly URLs or credentials) in history or logs.
            result = NetworkInventory(
                "unifi", "unknown", now, (), warnings=(f"UniFi poll failed ({type(e).__name__}).",)
            )
        if result.complete or result.devices:
            return result
        cached = self.host.poll(now)
        return replace(
            cached,
            warnings=tuple(
                ["UniFi unavailable; host cache coverage is partial.", *result.warnings, *cached.warnings][:32]
            ),
        )


class SegmentSink:
    """The segment store's writers for both streams, one per source; each poll is synced as it's written."""

    def __init__(self, root: Path) -> None:
        self.writers = segment_log.Writers(root)

    def write(self, stream: str, source: str, records: Sequence[dict[str, Any]]) -> None:
        self.writers.append(stream, source, records)
        self.writers.sync()

    def close(self) -> None:
        self.writers.close()


class WifiRecorder:
    """The access points heard, written when the set of BSSIDs changes or the last write is `rewrite_seconds` old.

    A still machine hears the same set all day; writing only changes (and a periodic restatement, so a reader knows
    the set still stands) keeps it from filling the stream. Signal alone changing doesn't count as a change.
    """

    def __init__(
        self,
        scanner: Scanner,
        sink: Publisher,
        observer: str,
        *,
        rewrite_seconds: float = segment_log.WIFI_REWRITE_SECONDS,
    ) -> None:
        self.scanner, self.sink, self.observer = scanner, sink, observer
        self.rewrite_seconds = rewrite_seconds
        self._heard: frozenset[str] | None = None
        self._written_at = -math.inf
        self._unavailable_logged = False

    def poll(self, now: float) -> int | None:
        """Read the access points and write them if due: how many were written, or None when Wi-Fi can't be read."""
        aps = self.scanner.scan()
        if aps is None:
            if not self._unavailable_logged:
                LOG.info("No Wi-Fi scan results to record (no adapter, nmcli, or permission); skipping Wi-Fi")
                self._unavailable_logged = True
            return None
        return self.record(aps, now)

    def record(self, aps: Sequence[AccessPoint], now: float) -> int:
        heard = frozenset(ap.bssid for ap in aps)
        if heard == self._heard and 0 <= now - self._written_at < self.rewrite_seconds:
            return 0
        scan_id = str(uuid.uuid7())
        rows = [
            dict(asdict(ap), id=str(uuid.uuid7()), start=now, end=now, scan_id=scan_id, observer=self.observer)
            for ap in aps
        ]
        if rows:
            self.sink.write(WIFI, "wifi", rows)
        self._heard, self._written_at = heard, now
        return len(rows)


class NetworkWorker:
    def __init__(
        self,
        source: InventorySource,
        sink: Publisher,
        *,
        poll_seconds: float = 60,
        wifi: WifiRecorder | None = None,
    ) -> None:
        if not math.isfinite(poll_seconds) or not 30 <= poll_seconds <= 3600:
            raise ValueError("network poll interval must be between 30 and 3600 seconds")
        self.source, self.sink = source, sink
        self.poll_seconds = poll_seconds
        self.wifi = wifi
        self._watermark = 0.0

    def process(self, inventory: NetworkInventory, *, poll_started_at: float | None = None) -> None:
        devices = {device.key: device for device in inventory.devices}
        warnings = list(inventory.warnings)
        complete = inventory.complete
        if len(devices) != len(inventory.devices):
            complete = False
            warnings.append("Duplicate client identifiers; absence is unknown.")
        if any(device.connected is None for device in devices.values()):
            complete = False
        if inventory.observed_at < self._watermark:
            complete = False
            warnings.append("Host clock moved backwards; connection continuity is unknown.")
        at = inventory.observed_at
        inventory_id = str(uuid.uuid7())
        rows = [
            dict(
                asdict(device),
                id=str(uuid.uuid7()),
                start=at,
                end=at,
                inventory_id=inventory_id,
                observer=inventory.observer,
                ap_id=device.access_point_id,
                ap_name=device.access_point_name,
            )
            for device in devices.values()
        ]
        # Positives before the poll's own row: a poll cut short never looks complete with devices missing.
        self.sink.write(OBSERVATIONS, inventory.source, rows)
        self.sink.write(
            INVENTORY,
            inventory.source,
            [
                dict(
                    id=inventory_id,
                    start=min(poll_started_at or at, at),
                    end=at,
                    observer=inventory.observer,
                    complete=complete,
                    authoritative=inventory.authoritative and complete,
                    device_count=len(devices),
                    warnings=warnings[:32],
                )
            ],
        )
        self._watermark = max(self._watermark, at)

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            began = time.monotonic()
            started_at = time.time()
            inventory = await asyncio.to_thread(self.source.poll, started_at)
            inventory = replace(inventory, observed_at=time.time())
            if stop.is_set():
                return
            self.process(inventory, poll_started_at=started_at)
            if self.wifi is not None:
                try:
                    await asyncio.to_thread(self.wifi.poll, time.time())
                except Exception as e:
                    # A Wi-Fi failure never stops the network inventory.
                    LOG.warning("Wi-Fi access point read failed (%s)", type(e).__name__)
            delay = max(0.0, self.poll_seconds - (time.monotonic() - began))
            try:
                await asyncio.wait_for(stop.wait(), timeout=delay)
            except TimeoutError:
                pass


def configured_source() -> tuple[NetworkSource, UniFiSession | None]:
    observer = str(cfg.get("network_devices.observer"))
    maximum = int(cfg.get("network_devices.max_devices"))
    explicit = str(cfg.get("network_devices.arp_path"))
    arp = (
        cfg.path("network_devices.arp_path")
        if explicit
        else (cfg.path("network_devices.linux_arp_path") if sys.platform.startswith("linux") else None)
    )
    leases = cfg.get("network_devices.dhcp_lease_paths")
    if not isinstance(leases, list) or any(not isinstance(path, str) for path in leases):
        raise ValueError("network_devices.dhcp_lease_paths must be a list of local file paths")
    paths = [Path(path).expanduser() for path in leases]
    host = HostNetworkSource(
        observer, arp, [path if path.is_absolute() else cfg.ROOT / path for path in paths], maximum
    )
    session = configured_session() if cfg.get("unifi.url") else None
    unifi = (
        UniFiNetwork(
            session,
            site=str(cfg.get("network_devices.site")),
            mode=str(cfg.get("unifi.mode")),
            observer=observer,
            max_devices=maximum,
        )
        if session
        else None
    )
    return NetworkSource(host, unifi), session


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pack-network-devices", description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="read one inventory and print counts only, without publishing"
    )
    args = parser.parse_args(argv)
    if not cfg.get("network_devices.enabled") and not args.check:
        LOG.info("Network inventory is disabled")
        return
    source, session = configured_source()
    if args.check:
        try:
            result = source.poll(time.time())
            aps = scanner().scan()
            print(
                json.dumps(
                    dict(
                        source=result.source,
                        devices=len(result.devices),
                        complete=result.complete,
                        authoritative=result.authoritative,
                        warnings=result.warnings,
                        wifi_access_points=None if aps is None else len(aps),
                    )
                )
            )
        finally:
            if session:
                session.close()
        return

    async def run() -> None:
        stop = asyncio.Event()
        for sig in (signal.SIGTERM, signal.SIGINT):
            asyncio.get_running_loop().add_signal_handler(sig, stop.set)
        sink = SegmentSink(segment_log.root())
        try:
            wifi = WifiRecorder(scanner(), sink, str(cfg.get("network_devices.observer")))
            worker = NetworkWorker(source, sink, poll_seconds=float(cfg.get("network_devices.poll_seconds")), wifi=wifi)
            await worker.run(stop)
        finally:
            sink.close()
            if session:
                session.close()

    asyncio.run(run())


if __name__ == "__main__":
    main()
