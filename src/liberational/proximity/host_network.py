"""Read host ARP caches and configured DHCP lease files without sending network traffic.

These are partial, potentially stale observations, never a connected-client inventory. An expired lease, a
missing cache entry, or an unreadable file cannot establish a disconnect. No DNS, ping, neighbor solicitation,
DHCP request, or controller operation is performed. macOS uses only `arp -an` to display its existing cache.

ISC lease dates are UTC and the last declaration for an IP wins (ISC dhcpd.leases manual). Apple's DHCP client
plists identify the client through an Ethernet ClientIdentifier; RouterHardwareAddress belongs to the router,
not the client (apple-oss-distributions/bootp, IPConfiguration.bproj/DHCPLease.c).
"""

import datetime as dt
import math
import os
import plistlib
import re
import selectors
import shutil
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol
from xml.parsers.expat import ExpatError

from liberational.proximity.network import NetworkDevice, NetworkInventory, ip_address, mac_address

MAX_FILES = 16
MAX_READ_BYTES = 1_048_576
MAX_ROWS = 8192
PARTIAL = "Host ARP caches and DHCP leases are partial, possibly stale evidence; absence never proves disconnect."


@dataclass(frozen=True)
class CommandResult:
    stdout: bytes
    returncode: int
    truncated: bool = False


class CommandRunner(Protocol):
    def __call__(self, command: Sequence[str], *, timeout: float, max_bytes: int) -> CommandResult: ...


def run_command(command: Sequence[str], *, timeout: float, max_bytes: int) -> CommandResult:
    """Bound both command time and captured bytes, without a shell or unbounded communicate() buffer."""
    output = bytearray()
    deadline = time.monotonic() + timeout
    process = subprocess.Popen(
        list(command), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    assert process.stdout is not None
    try:
        os.set_blocking(process.stdout.fileno(), False)
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("host ARP cache command timed out")
                if not selector.select(remaining):
                    raise TimeoutError("host ARP cache command timed out")
                chunk = os.read(process.stdout.fileno(), min(65536, max_bytes + 1 - len(output)))
                if not chunk:
                    return CommandResult(bytes(output), process.wait(timeout=max(0.001, deadline - time.monotonic())))
                output.extend(chunk)
                if len(output) > max_bytes:
                    process.kill()
                    process.wait(timeout=1)
                    return CommandResult(bytes(output[:max_bytes]), process.returncode or 0, True)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=1)
        process.stdout.close()


class _Warnings:
    def __init__(self) -> None:
        self.items = [PARTIAL]

    def add(self, text: str) -> None:
        message = text[:512]
        if message in self.items:
            return
        if len(self.items) < 31:
            self.items.append(message)
        elif len(self.items) == 31:
            self.items.append("Additional host-cache warnings were omitted at the warning limit.")


def _mac(value: object, *, short_octets: bool = False) -> str | None:
    if short_octets and isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{1,2}(?::[0-9a-fA-F]{1,2}){5}", value):
        value = ":".join(part.zfill(2) for part in value.split(":"))
    result = mac_address(value)
    return result if result not in ("00:00:00:00:00:00", "FF:FF:FF:FF:FF:FF") else None


def _hostname(value: object, warnings: _Warnings) -> str | None:
    if value is None or value == "*" or value == "?":
        return None
    try:
        valid = (
            isinstance(value, str)
            and bool(value)
            and len(value.encode()) <= 512
            and not any(ord(c) < 32 for c in value)
        )
    except UnicodeError:
        valid = False
    if not valid:
        warnings.add("A cache or lease hostname was invalid or exceeded the metadata limit; its value is unknown.")
        return None
    return value if isinstance(value, str) else None


def _device(mac: str, ip: str, *, hostname: str | None = None, expiry: float | None = None) -> NetworkDevice:
    return NetworkDevice("mac_address", mac, mac=mac, ip=ip, hostname=hostname, lease_expires_at=expiry)


def _read(path: Path, limit: int, warnings: _Warnings) -> bytes | None:
    descriptor: int | None = None
    try:
        # Opening a configured FIFO/device must not block the worker. /proc/net/arp is a regular virtual file.
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            warnings.add(f"Host cache {path} is not a regular file; it was not read.")
            return None
        with os.fdopen(descriptor, "rb") as source:
            descriptor = None
            raw = source.read(limit + 1)
        if len(raw) > limit:
            warnings.add(
                f"Host cache {path} exceeded the byte limit; only complete records in its prefix are available."
            )
            raw = raw[:limit]
            if not raw.startswith((b"<?xml", b"bplist", b"<plist")):
                raw = raw.rsplit(b"\n", 1)[0] if b"\n" in raw else b""
        return raw
    except OSError as error:
        warnings.add(f"Host cache {path} could not be read ({type(error).__name__}); coverage is unknown.")
        return None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _arp(raw: bytes, *, darwin: bool, max_rows: int, warnings: _Warnings) -> list[NetworkDevice]:
    rows: list[NetworkDevice] = []
    malformed = 0
    incomplete = 0
    for number, line in enumerate(raw.decode("utf-8", errors="replace").splitlines()):
        if number >= max_rows:
            warnings.add("Host ARP cache exceeded the row limit; remaining entries are unknown.")
            break
        if not line.strip() or line.startswith("IP address"):
            continue
        host: str | None = None
        if darwin:
            match = re.fullmatch(r"\s*(\S+)\s+\(([^)]+)\)\s+at\s+(\S+)\s+on\s+.+", line)
            if not match:
                malformed += 1
                continue
            host = _hostname(match[1], warnings)
            ip = ip_address(match[2])
            mac = _mac(match[3], short_octets=True)
            if match[3] == "(incomplete)":
                incomplete += 1
                continue
        else:
            parts = line.split()
            if len(parts) != 6:
                malformed += 1
                continue
            ip, mac = ip_address(parts[0]), _mac(parts[3])
            try:
                flags = int(parts[2], 0)
                hardware = int(parts[1], 0)
            except ValueError:
                malformed += 1
                continue
            if not flags & 0x2:
                incomplete += 1
                continue
            if hardware != 1:
                malformed += 1
                continue
        if ip is None or mac is None:
            malformed += 1
            continue
        rows.append(_device(mac, ip, hostname=host))
    if malformed:
        warnings.add(f"Host ARP cache contained {malformed} malformed or unsupported entries.")
    if incomplete:
        warnings.add(f"Host ARP cache contained {incomplete} unresolved entries; their device identities are unknown.")
    return rows


def _expiry(value: object, warnings: _Warnings) -> float | None:
    try:
        if not isinstance(value, str | int | float) or isinstance(value, bool):
            raise ValueError()
        number = float(value)
        if not math.isfinite(number) or number < 0 or isinstance(value, bool):
            raise ValueError()
        return number
    except ValueError, TypeError, OverflowError:
        warnings.add("A DHCP lease expiry was malformed; its expiry is unknown.")
        return None


def _dnsmasq(text: str, max_rows: int, warnings: _Warnings) -> list[NetworkDevice]:
    latest: dict[str, NetworkDevice] = {}
    malformed = 0
    for number, line in enumerate(text.splitlines()):
        if number >= max_rows:
            warnings.add("DHCP leases exceeded the row limit; later declarations are unavailable.")
            break
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 5:
            malformed += 1
            continue
        mac, ip = _mac(fields[1]), ip_address(fields[2])
        if mac is None or ip is None:
            malformed += 1
            continue
        latest[ip] = _device(
            mac, ip, hostname=_hostname(fields[3], warnings), expiry=_expiry(fields[0], warnings) or None
        )
    if malformed:
        warnings.add(f"DHCP text contained {malformed} malformed or unsupported rows (including non-MAC client IDs).")
    return list(latest.values())


_ISC_TOKEN = re.compile(r'"(?:\\.|[^"\\])*"|\#[^\n]*|[{};]|[^\s{};#"]+')


def _isc_string(token: str) -> str:
    if not token.startswith('"'):
        return token
    return re.sub(
        r"\\([0-7]{1,3}|.)", lambda m: chr(int(m[1], 8)) if re.fullmatch(r"[0-7]{1,3}", m[1]) else m[1], token[1:-1]
    )


def _isc_date(parts: list[str], warnings: _Warnings) -> float | None:
    try:
        if parts == ["never"]:
            return None
        if len(parts) == 2 and parts[0] == "epoch":
            return _expiry(parts[1], warnings)
        if len(parts) != 3:
            raise ValueError()
        value = dt.datetime.strptime(" ".join(parts[1:]), "%Y/%m/%d %H:%M:%S").replace(tzinfo=dt.UTC).timestamp()
        if value < 0:
            raise ValueError()
        return value
    except ValueError, OverflowError:
        warnings.add("An ISC DHCP lease date was malformed; its expiry is unknown.")
        return None


def _isc(text: str, max_rows: int, warnings: _Warnings) -> list[NetworkDevice]:
    tokens: list[str] = []
    end = 0
    for match in _ISC_TOKEN.finditer(text):
        if text[end : match.start()].strip():
            warnings.add(
                "ISC DHCP lease text contained an unterminated quote or unsupported token; later records are unknown."
            )
            break
        end = match.end()
        token = match[0]
        if token.startswith("#"):
            continue
        if len(tokens) >= max_rows * 64:
            warnings.add("ISC DHCP lease token limit reached; later declarations are unavailable.")
            break
        tokens.append(token)
    latest: dict[str, NetworkDevice | None] = {}
    index = 0
    declarations = 0
    while index + 2 < len(tokens):
        if tokens[index] != "lease" or tokens[index + 2] != "{":
            index += 1
            continue
        if declarations >= max_rows:
            warnings.add("ISC DHCP lease declaration limit reached; later declarations are unavailable.")
            break
        declarations += 1
        ip = ip_address(tokens[index + 1])
        index += 3
        depth = 1
        statements: list[list[str]] = []
        statement: list[str] = []
        while index < len(tokens) and depth:
            token = tokens[index]
            index += 1
            if token == "{":
                depth += 1
                statement = []
            elif token == "}":
                depth -= 1
                statement = []
            elif depth == 1 and token == ";":
                statements.append(statement)
                statement = []
            elif depth == 1:
                statement.append(token)
        if depth:
            warnings.add("ISC DHCP lease ended inside a declaration; that incomplete declaration was not used.")
            break
        if ip is None:
            warnings.add("An ISC DHCP lease had an invalid IP address.")
            continue
        mac: str | None = None
        name: str | None = None
        expiry: float | None = None
        for fields in statements:
            if fields[:2] == ["hardware", "ethernet"] and len(fields) == 3:
                mac = _mac(fields[2])
            elif fields[:1] == ["client-hostname"] and len(fields) == 2:
                name = _hostname(_isc_string(fields[1]), warnings)
            elif fields[:1] == ["ends"]:
                expiry = _isc_date(fields[1:], warnings)
        latest[ip] = _device(mac, ip, hostname=name, expiry=expiry) if mac is not None else None
        if mac is None:
            warnings.add("An ISC DHCP lease had no valid Ethernet client MAC; its identity is unknown.")
    return [row for row in latest.values() if row is not None]


def _apple(raw: bytes, max_rows: int, warnings: _Warnings) -> list[NetworkDevice]:
    try:
        parsed: Any = plistlib.loads(raw)
    except ValueError, TypeError, OverflowError, RecursionError, plistlib.InvalidFileException, ExpatError:
        warnings.add("Apple DHCP plist could not be decoded; lease coverage is unknown.")
        return []
    entries = parsed if isinstance(parsed, list) else [parsed]
    rows: list[NetworkDevice] = []
    for number, entry in enumerate(entries):
        if number >= max_rows:
            warnings.add("Apple DHCP plist exceeded the lease row limit.")
            break
        if not isinstance(entry, dict):
            warnings.add("Apple DHCP plist contained an unsupported lease record.")
            continue
        client = entry.get("ClientIdentifier")
        # Type 1 followed by six bytes is an Ethernet client ID. The router's MAC is never the client's identity.
        mac = _mac(client[1:].hex(":")) if isinstance(client, bytes) and len(client) == 7 and client[0] == 1 else None
        ip = ip_address(entry.get("IPAddress"))
        if mac is None or ip is None:
            warnings.add("Apple DHCP lease lacked an Ethernet ClientIdentifier or IP address; its identity is unknown.")
            continue
        expiry: float | None = None
        start, duration = entry.get("LeaseStartDate"), entry.get("LeaseLength")
        if isinstance(start, dt.datetime) and isinstance(duration, int) and not isinstance(duration, bool):
            if 0 <= duration < 0xFFFFFFFF:
                try:
                    expiry = (start.replace(tzinfo=dt.UTC) if start.tzinfo is None else start).timestamp() + duration
                    if not math.isfinite(expiry) or expiry < 0:
                        raise ValueError()
                except ValueError, OverflowError, OSError:
                    expiry = None
                    warnings.add("An Apple DHCP lease expiry was outside the supported time range.")
            elif duration != 0xFFFFFFFF:
                warnings.add("An Apple DHCP lease duration was invalid; its expiry is unknown.")
        elif start is not None or duration is not None:
            warnings.add("An Apple DHCP lease expiry lacked a valid start date or duration.")
        rows.append(_device(mac, ip, expiry=expiry))
    return rows


class HostNetworkSource:
    def __init__(
        self,
        observer: str,
        arp_path: Path | None = None,
        dhcp_paths: Sequence[Path] = (),
        max_devices: int = 1024,
        *,
        platform: str = sys.platform,
        runner: CommandRunner | None = None,
        which: Callable[[str], str | None] | None = None,
        max_read_bytes: int = MAX_READ_BYTES,
        max_rows: int = MAX_ROWS,
        command_timeout: float = 5.0,
    ) -> None:
        if not observer or len(observer) > 128 or not 1 <= max_devices <= 2048:
            raise ValueError("invalid host network observer or device bound")
        if (
            len(dhcp_paths) > MAX_FILES
            or not 1 <= max_read_bytes <= 4 * MAX_READ_BYTES
            or not 1 <= max_rows <= MAX_ROWS
        ):
            raise ValueError("host network file/read bounds exceeded")
        if not math.isfinite(command_timeout) or not 0 < command_timeout <= 30:
            raise ValueError("host ARP command timeout must be finite and at most 30 seconds")
        self.observer, self.arp_path, self.dhcp_paths = observer, arp_path, tuple(dhcp_paths)
        self.max_devices, self.platform, self.runner = max_devices, platform, runner or run_command
        self.which = which or shutil.which
        self.max_read_bytes, self.max_rows, self.command_timeout = max_read_bytes, max_rows, command_timeout

    def poll(self, now: float) -> NetworkInventory:
        if not math.isfinite(now) or now < 0:
            raise ValueError("host network inventory time must be finite")
        warnings = _Warnings()
        rows: dict[tuple[str, str], NetworkDevice] = {}

        def keep(device: NetworkDevice) -> None:
            previous = rows.get(device.key)
            if previous is not None:
                same_ip = previous.ip == device.ip
                if not same_ip:
                    warnings.add("A native MAC had conflicting cached IP addresses; its lease expiry is unknown.")
                rows[device.key] = replace(
                    previous,
                    hostname=device.hostname or previous.hostname,
                    lease_expires_at=device.lease_expires_at if same_ip else None,
                )
            elif len(rows) < self.max_devices:
                rows[device.key] = device
            else:
                warnings.add("Host network device limit reached; additional device identities were omitted.")

        raw: bytes | None = None
        if self.arp_path is not None:
            raw = _read(self.arp_path, self.max_read_bytes, warnings)
        elif self.platform == "darwin":
            executable = self.which("arp")
            if executable is None:
                warnings.add("The read-only arp cache command is unavailable; neighbor coverage is unknown.")
            else:
                try:
                    result = self.runner(
                        [executable, "-an"], timeout=self.command_timeout, max_bytes=self.max_read_bytes
                    )
                    if result.truncated or len(result.stdout) > self.max_read_bytes:
                        warnings.add("Host ARP command output exceeded the byte limit; later entries are unknown.")
                        raw = result.stdout[: self.max_read_bytes]
                        raw = raw.rsplit(b"\n", 1)[0] if b"\n" in raw else b""
                    elif result.returncode:
                        warnings.add(
                            f"Host ARP command failed (exit {result.returncode}); neighbor coverage is unknown."
                        )
                    else:
                        raw = result.stdout[: self.max_read_bytes]
                except (OSError, TimeoutError, subprocess.SubprocessError) as error:
                    warnings.add(
                        f"Host ARP cache command failed ({type(error).__name__}); neighbor coverage is unknown."
                    )
        else:
            warnings.add("No host ARP cache path is configured; neighbor coverage is unknown.")
        if raw is not None:
            for device in _arp(raw, darwin=self.platform == "darwin", max_rows=self.max_rows, warnings=warnings):
                keep(device)
        for path in self.dhcp_paths:
            data = _read(path, self.max_read_bytes, warnings)
            if data is None:
                continue
            if data.lstrip().startswith((b"<?xml", b"bplist", b"<plist")):
                found = _apple(data, self.max_rows, warnings)
            else:
                text = data.decode("utf-8", errors="replace")
                found = (
                    _isc(text, self.max_rows, warnings)
                    if re.search(r"\blease\s+[^\s{}]+\s*\{", text)
                    else _dnsmasq(text, self.max_rows, warnings)
                )
            for device in found:
                keep(device)
        return NetworkInventory(
            "host_network", self.observer, now, tuple(rows.values()), False, False, tuple(warnings.items)
        )
