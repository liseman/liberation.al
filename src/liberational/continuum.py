"""Ids and times as Continuum keeps them (#547, docs/CONTINUUM.md). Pure functions, no I/O.

- Ids: canonical UUIDv7 strings, kept as 16 bytes on the wire and in the catalog (decision 8). `check_id` refuses
  anything else. `keyed_id` makes the same id from the same key in any process.
- Times: Python instants are UTC epoch seconds (floats); Continuum times are UTC nanoseconds in a u64 (decision 9).
  Seconds to nanoseconds rounds to the nearest nanosecond, and back is the nearest float, so seconds survive the
  round trip within a nanosecond (exactly, for any time after April 1970). Nanoseconds don't survive the trip
  through seconds: a float near today's epoch is only exact to about 240 ns.
"""

import fractions
import hashlib
import operator
import uuid

from liberational.domain.kernel import (
    Instant,
    check_instant,
    check_span,
)

NS_PER_S = 1_000_000_000
U64_MAX = 2**64 - 1


# Ids


def keyed_id(seconds: float, *key: object) -> str:
    """A UUIDv7 that is the same every time for the same `key` at the same time: the 48-bit millisecond timestamp from
    `seconds`, the rest from a hash of `key`, with the version and variant bits set. For a record derived from a v1
    line or an event that must be written once whatever process writes it (a spoken quantity, an imported row): the
    broker keeps one copy of a repeated header, so the same key sent twice is stored once."""
    ms = max(0, int(seconds * 1000)) & ((1 << 48) - 1)
    digest = hashlib.sha256("|".join(map(str, key)).encode()).digest()[:10]
    raw = bytearray(ms.to_bytes(6, "big") + digest)
    raw[6] = 0x70 | (raw[6] & 0x0F)
    raw[8] = 0x80 | (raw[8] & 0x3F)
    return str(uuid.UUID(bytes=bytes(raw)))


def check_id(id: str) -> str:
    """id, if it's a canonical (lowercase, hyphenated) UUIDv7 string. Raises TypeError or ValueError otherwise."""
    if not isinstance(id, str):
        raise TypeError(f"an id is a string, not {type(id).__name__}")
    try:
        u = uuid.UUID(id)
    except ValueError as e:
        raise ValueError(f"not a UUID: {id!r}") from e
    if str(u) != id:
        raise ValueError(f"not a canonical UUID string (lowercase, hyphenated): {id!r}")
    if u.version != 7 or u.variant != uuid.RFC_4122:
        raise ValueError(f"not a UUIDv7: {id!r}")
    return id


# Times


def check_ns(ns: object, name: str = "nanoseconds") -> int:
    """ns as an int, if it's a u64 (numpy integers included). Raises TypeError or ValueError otherwise."""
    if isinstance(ns, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        n = operator.index(ns)  # ty: ignore[invalid-argument-type]  # a non-integer raises TypeError, handled below
    except TypeError:
        raise TypeError(f"{name} must be an integer, not {type(ns).__name__}") from None
    if not 0 <= n <= U64_MAX:
        raise ValueError(f"{name} is not a u64: {n!r}")
    return n


def to_ns(t: Instant) -> int:
    """Epoch seconds as UTC nanoseconds, to the nearest nanosecond. Raises like kernel.check_instant, and for a time
    past a u64 (the year 2554)."""
    check_instant(t, "instant")
    ns = round(fractions.Fraction(t) * NS_PER_S)  # exact: a float times 1e9 would round first
    if ns > U64_MAX:
        raise ValueError(f"{t!r} is past what Continuum's u64 nanoseconds hold")
    return ns


def from_ns(ns: int) -> Instant:
    """UTC nanoseconds as epoch seconds, the nearest float."""
    return check_ns(ns) / NS_PER_S  # int / int is correctly rounded


def span_ns(start: Instant, end: Instant | None = None) -> tuple[int, int]:
    """A record's time as a payload header's (start_ns, end_ns): equal for an instant (decision 9). The order is
    checked before rounding, as kernel.check_span does."""
    if end is None:
        s = to_ns(start)
        return s, s
    check_span(start, end)
    return to_ns(start), to_ns(end)


# Evidence


# Producers


# Layers
