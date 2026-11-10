"""BLE1 transport-neutral ESP32 advertisement protocol, with bounded framing and per-sensor clocks.

Every line ends in |CRC32 (IEEE, uppercase hexadecimal) and LF; the CRC covers all ASCII before that separator.
HELLO carries chip/boot identity, receive uptime, and an optional UTF-8 sensor name as hex. ADV carries identity,
report counter, receive uptime, advertiser address/type, RSSI, channel, and unmodified AD bytes. The host supplies
receipt wall time. Clocks use a per-boot minimum observed receipt-minus-uptime offset: USB buffering can delay a
report but cannot make it precede its capture. This is an estimate, not clock synchronization or location proof.
"""

import math
import re
import zlib
from dataclasses import dataclass

from liberational.proximity import advertising
from liberational.proximity.ble import Advertisement

CHIP = re.compile(r"[0-9A-F]{12}\Z")
BOOT = re.compile(r"[0-9A-F]{8}\Z")
ADDRESS = re.compile(r"(?:[0-9A-F]{2}:){5}[0-9A-F]{2}\Z")
ADDRESS_TYPES = {0: "public", 1: "random", 2: "public_identity", 3: "random_identity"}
MAX_LINE = 4096


def frame(fields: list[str]) -> bytes:
    """Checksummed wire bytes, also useful for synthetic sources and the NAME1 configuration command."""
    body = "|".join(fields).encode("ascii")
    return body + f"|{zlib.crc32(body):08X}\n".encode("ascii")


def _number(text: str, maximum: int) -> int:
    if not text.isascii() or not text.isdecimal() or len(text) > 20:
        raise ValueError("invalid unsigned field")
    value = int(text)
    if value > maximum:
        raise ValueError("unsigned field out of range")
    return value


def _hex(text: str, maximum: int) -> bytes:
    if len(text) > maximum * 2 or len(text) % 2 or not re.fullmatch(r"[0-9a-fA-F]*", text):
        raise ValueError("invalid hexadecimal field")
    return bytes.fromhex(text)


@dataclass(frozen=True)
class Hello:
    chip_id: str
    boot_id: str
    uptime_us: int
    name: str


@dataclass(frozen=True)
class Report:
    chip_id: str
    boot_id: str
    counter: int
    uptime_us: int
    address: str
    address_type: str
    rssi: int | None
    channel: int | None
    payload: bytes


def parse(line: bytes) -> Hello | Report:
    """Validate framing, checksum, bounds, and version before accepting untrusted serial bytes."""
    if len(line) > MAX_LINE:
        raise ValueError("serial line too long")
    try:
        text = line.rstrip(b"\r\n").decode("ascii")
        body, crc = text.rsplit("|", 1)
        if not re.fullmatch(r"[0-9A-F]{8}", crc) or zlib.crc32(body.encode("ascii")) != int(crc, 16):
            raise ValueError("serial checksum mismatch")
        fields = body.split("|")
        if len(fields) < 4 or fields[0] != "BLE1" or not CHIP.fullmatch(fields[2]) or not BOOT.fullmatch(fields[3]):
            raise ValueError("unknown protocol or sensor identity")
        if fields[1] == "HELLO" and len(fields) == 6:
            return Hello(fields[2], fields[3], _number(fields[4], 2**64 - 1), _hex(fields[5], 32).decode("utf-8"))
        if fields[1] != "ADV" or len(fields) != 11:
            raise ValueError("unknown BLE1 message shape")
        if not ADDRESS.fullmatch(fields[6]):
            raise ValueError("invalid advertiser address")
        address_type = ADDRESS_TYPES[_number(fields[7], 3)]
        if not re.fullmatch(r"-?\d{1,3}", fields[8]):
            raise ValueError("invalid RSSI")
        rssi = int(fields[8])
        channel = _number(fields[9], 39)
        if not -127 <= rssi <= 127 or channel not in (0, 37, 38, 39):
            raise ValueError("RSSI or channel out of range")
        return Report(
            fields[2],
            fields[3],
            _number(fields[4], 2**32 - 1),
            _number(fields[5], 2**64 - 1),
            fields[6],
            address_type,
            None if rssi == 127 else rssi,
            channel or None,
            _hex(fields[10], 1650),
        )
    except (UnicodeError, IndexError, KeyError) as e:
        raise ValueError("invalid serial encoding or field") from e


class Framer:
    """Bounded line buffering; discard an oversized line through LF, then recover at the next frame."""

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.discarding = False
        self.invalid = 0

    def feed(self, data: bytes) -> list[bytes]:
        lines: list[bytes] = []
        for byte in data:
            if byte == 10:
                if not self.discarding:
                    lines.append(bytes(self.buffer))
                self.buffer.clear()
                self.discarding = False
            elif not self.discarding:
                if len(self.buffer) >= MAX_LINE:
                    self.buffer.clear()
                    self.discarding = True
                    self.invalid += 1
                else:
                    self.buffer.append(byte)
        return lines


class SensorClock:
    """One attached sensor, with bounded per-boot clock/counter evidence and no inferred device identity."""

    def __init__(self, names: dict[str, str] | None = None) -> None:
        self.names = names or {}
        self.hello: Hello | None = None
        self.offset_s: float | None = None
        self.counter: int | None = None
        self.uptime_us = 0
        self.lost_reports = 0
        self.reordered_reports = 0
        self.malformed_payloads = 0

    def accept(self, message: Hello | Report, received_at: float) -> Advertisement | None:
        if not math.isfinite(received_at) or received_at < 0:
            raise ValueError("receipt time must be a finite epoch time")
        if isinstance(message, Hello):
            if self.hello is not None and self.hello.chip_id != message.chip_id:
                raise ValueError("sensor changed on one serial connection")
            if self.hello is None or self.hello.boot_id != message.boot_id:
                self.offset_s = None
                self.counter = None
                self.uptime_us = 0
            self.hello = message
            self._offset(message.uptime_us, received_at)
            return None
        if self.hello is None or (message.chip_id, message.boot_id) != (self.hello.chip_id, self.hello.boot_id):
            raise ValueError("advertisement has no matching sensor hello")
        if self.counter is not None:
            delta = (message.counter - self.counter) % 2**32
            if delta == 0 or delta >= 2**31 or message.uptime_us < self.uptime_us:
                self.reordered_reports += 1
                return None
            self.lost_reports += delta - 1
        self.counter, self.uptime_us = message.counter, message.uptime_us
        offset = self._offset(message.uptime_us, received_at)
        fields = advertising.decode(message.payload)
        self.malformed_payloads += fields.malformed
        return Advertisement(
            observed_at=min(received_at, message.uptime_us / 1_000_000 + offset),
            backend="esp32_serial",
            adapter=message.chip_id,
            identifier_kind="bluetooth_address",
            identifier=message.address,
            address_type=message.address_type,
            rssi=message.rssi,
            local_name=fields.local_name,
            manufacturer_data=fields.manufacturer_data,
            service_data=fields.service_data,
            service_uuids=fields.service_uuids,
            tx_power=fields.tx_power,
            scan_mode="passive",
            raw_payload=message.payload,
            device_counter=message.counter,
            device_uptime_us=message.uptime_us,
            boot_id=message.boot_id,
            channel=message.channel,
            received_at=received_at,
            clock_offset_s=offset,
            sensor_name=self.names.get(message.chip_id) or self.hello.name or message.chip_id,
            payload_malformed=fields.malformed,
        )

    def _offset(self, uptime_us: int, received_at: float) -> float:
        candidate = received_at - uptime_us / 1_000_000
        self.offset_s = min(self.offset_s, candidate) if self.offset_s is not None else candidate
        return self.offset_s
