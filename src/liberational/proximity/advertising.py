"""Decode Bluetooth advertising data without contacting its sender.

AD structures are length/type/value tuples. Unknown types remain in raw_payload in the sighting; malformed
structures are reported rather than interpreted as absent fields. UUIDs use Bluetooth's little-endian wire order.
This decoder is shared by serial/replay sources and future computer Bluetooth backends.
"""

import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field


def service_uuid(value: bytes) -> str:
    """A 16-, 32-, or 128-bit little-endian Bluetooth UUID as its canonical string."""
    if len(value) in (2, 4):
        return f"{int.from_bytes(value, 'little'):08x}-0000-1000-8000-00805f9b34fb"
    if len(value) == 16:
        return str(uuid.UUID(bytes=value[::-1]))
    raise ValueError("a service UUID needs 2, 4, or 16 bytes")


@dataclass
class AdvertisingData:
    local_name: str | None = None
    manufacturer_data: dict[int, bytes] = field(default_factory=dict)
    service_data: dict[str, bytes] = field(default_factory=dict)
    service_uuids: tuple[str, ...] = ()
    tx_power: int | None = None
    malformed: bool = False
    truncated: bool = False


def _manufacturer(value: bytes, out: AdvertisingData) -> None:
    if len(value) < 2:
        out.malformed = True
        return
    company = int.from_bytes(value[:2], "little")
    if company in out.manufacturer_data:
        out.malformed = True
    out.manufacturer_data[company] = value[2:]


def _platform_uuid(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        if len(value) in (4, 8) and all(c in "0123456789abcdefABCDEF" for c in value):
            return f"{int(value, 16):08x}-0000-1000-8000-00805f9b34fb"
        if len(value) == 36:
            return str(uuid.UUID(value))
    except ValueError:
        pass
    return None


def _platform_bytes(value: object, limit: int) -> tuple[bytes | None, bool]:
    """Bound the copy before converting a native NSData, byte array, or D-Bus byte sequence."""
    try:
        if isinstance(value, list | tuple):
            return bytes(value[:limit]), len(value) > limit
        view = memoryview(value).cast("B")  # ty: ignore[invalid-argument-type] Native NSData buffers have no Python stub.
        return bytes(view[:limit]), len(view) > limit
    except TypeError, ValueError, BufferError, OverflowError:
        return None, False


def platform_data(
    *,
    local_name: object = None,
    manufacturer_data: Mapping[object, object] | None = None,
    manufacturer_bytes: object = None,
    service_data: Mapping[object, object] | None = None,
    service_uuids: Iterable[object] = (),
    tx_power: object = None,
) -> AdvertisingData:
    """Bound and normalize fields already decoded by an OS, without inventing a raw on-air AD packet.

    Manufacturer bytes from CoreBluetooth include the little-endian company prefix, using the same decoder as
    raw AD data. UUIDs from either OS become the Bluetooth canonical form. Native caches can combine reports;
    these fields describe an OS report, not a reconstructed radio packet or a GATT read.
    """
    out = AdvertisingData()
    left = 65536
    if local_name is not None:
        if isinstance(local_name, str):
            raw = local_name[:256].encode("utf-8", errors="replace")
            out.local_name = raw[:256].decode("utf-8", errors="ignore")
            out.truncated |= len(local_name) > 256 or len(raw) > 256
            left -= len(out.local_name.encode())
        else:
            out.malformed = True
    if manufacturer_bytes is not None:
        raw, cut = _platform_bytes(manufacturer_bytes, left)
        out.truncated |= cut
        if raw is None:
            out.malformed = True
        else:
            _manufacturer(raw, out)
            left -= len(raw)
    for i, (company, value) in enumerate((manufacturer_data or {}).items()):
        if i >= 64 or left < 2:
            out.truncated = True
            break
        if not isinstance(company, int) or isinstance(company, bool) or not 0 <= company <= 65535:
            out.malformed = True
            continue
        data, cut = _platform_bytes(value, left - 2)
        out.truncated |= cut
        if data is None:
            out.malformed = True
            continue
        _manufacturer(company.to_bytes(2, "little") + data, out)
        left -= len(data) + 2
    for i, (key, value) in enumerate((service_data or {}).items()):
        if i >= 64 or left < 36:
            out.truncated = True
            break
        name = _platform_uuid(key)
        if name is None:
            out.malformed = True
            continue
        data, cut = _platform_bytes(value, left - 36)
        out.truncated |= cut
        if data is None:
            out.malformed = True
            continue
        if name in out.service_data:
            out.malformed = True
        out.service_data[name] = data
        left -= len(data) + 36
    services: set[str] = set()
    for i, value in enumerate(service_uuids):
        if i >= 64 or left < 36:
            out.truncated = True
            break
        name = _platform_uuid(value)
        if name is None:
            out.malformed = True
            continue
        services.add(name)
        left -= 36
    out.service_uuids = tuple(sorted(services))
    if isinstance(tx_power, int) and not isinstance(tx_power, bool) and -127 <= tx_power <= 126:
        out.tx_power = tx_power
    elif tx_power not in (None, 127):
        out.malformed = True
    return out


def decode(payload: bytes) -> AdvertisingData:
    """Known fields from a bounded raw AD payload; never reads a remote name or GATT attribute."""
    if len(payload) > 1650:
        raise ValueError("advertising payload exceeds the extended advertising limit")
    out = AdvertisingData()
    services: set[str] = set()
    complete_name = False
    pos = 0
    while pos < len(payload):
        length = payload[pos]
        pos += 1
        if length == 0:
            break  # permitted padding
        if pos + length > len(payload):
            out.malformed = True
            break
        kind, value = payload[pos], payload[pos + 1 : pos + length]
        pos += length
        if kind in (0x08, 0x09) and (kind == 0x09 or not complete_name):
            out.local_name = value.decode("utf-8", errors="replace")
            complete_name = kind == 0x09
        elif kind == 0x0A:
            if len(value) != 1:
                out.malformed = True
            elif value[0] != 127:
                out.tx_power = int.from_bytes(value, "little", signed=True)
        elif kind == 0xFF:
            _manufacturer(value, out)
        elif kind in (2, 3, 4, 5, 6, 7):
            width = {2: 2, 3: 2, 4: 4, 5: 4, 6: 16, 7: 16}[kind]
            if len(value) % width:
                out.malformed = True
            for offset in range(0, len(value) - width + 1, width):
                services.add(service_uuid(value[offset : offset + width]))
        elif kind in (0x16, 0x20, 0x21):
            width = {0x16: 2, 0x20: 4, 0x21: 16}[kind]
            if len(value) < width:
                out.malformed = True
            else:
                key = service_uuid(value[:width])
                if key in out.service_data:
                    out.malformed = True
                out.service_data[key] = value[width:]
    out.service_uuids = tuple(sorted(services))
    return out
