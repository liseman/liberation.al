"""Devices linked to people and things: who is home, and where my things are.

The owner links a device to a person ("Alice's phone") or names it as a thing ("AirPods", "keys tag"). A link is an
owner assertion in the knowledge store's `device.link` collection (outside the broker's reset), latest revision per
link in the order written. It names the device by one selector:

- `ble_identifier`: one Bluetooth address or platform identifier (a watch or tag with a static address);
- `ble_fingerprint`: the exact advertising payload fingerprint the sightings' candidates use (proximity/ble.py), for
  a device whose address rotates but whose payload doesn't;
- `ble_name`: the advertised local name, exactly;
- `net_mac`: a Wi-Fi or wired MAC from the router's client list (`network.device_observation`). A phone's
  per-network private address is stable on that network.

Sightings are read as they are recorded (`bluetooth.sighting`, `network.device_observation`). A link turns
them into presence spans for a person (sightings no more than `devices.gap_s` apart are one stay) and a last seen
time, place (the sensor that heard it), and signal strength for a thing. None of this is an identity: a shared or
lent device, or a payload two devices share, is reported as the device, and every answer says these are device
sightings. Absence is never inferred where no sensor was running (`bluetooth.coverage`, `network.inventory`).

Every device seen is kept, linked or not; `summary` counts them. Nothing here alerts.

**Where a device lives**: beside its person or thing link, the owner can say a device lives in a place, or a
room of that place's plan (memory/places.py), as a `device.location` in the knowledge store (taught state, carried
across a reset by `pack-knowledge`), one per selector, revised in place. Known › Devices sets it and Ask says it
("Your TV lives in the Living room"). Nothing infers from it yet. Once the pack moves, these devices are beacons: a
located device heard with a strong signal (a smart plug, a TV, a speaker that never moves) says the rig is in or near
its room, and several heard together narrow it further. That inference belongs with the place signals (the Wi-Fi
access points, visual relocalization) that derive visits, as one more signal, weighted by signal strength.
"""

import math
import os
import re
import threading
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import orjson

from liberational import config, knowledge, logs, segment_log
from liberational.people import registry
from liberational.proximity import ble, vendors

LINKS = "device.link"
LOCATIONS = "device.location"  # where a device lives: a place, or a room of its plan
LOG = logs.get("devices")
BLE_SIGHTINGS = segment_log.BLE_SIGHTINGS
BLE_COVERAGE = segment_log.BLE_COVERAGE
NET_OBSERVATIONS = segment_log.NET_OBSERVATIONS
NET_INVENTORY = segment_log.NET_INVENTORY
SELECTORS = ("ble_identifier", "ble_fingerprint", "ble_name", "net_mac")
TARGETS = ("person", "thing")
MAX_LINK_REVISIONS = 4096
MAX_PAGES = 64
MAX_NAME = 80

Target = Literal["person", "thing"]


def gap_s() -> float:
    """Sightings this close together are one stay."""
    return float(config.get("devices.gap_s"))


def _text(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > MAX_NAME:
        raise ValueError(f"a device link's {what} must be 1 to {MAX_NAME} characters")
    return value.strip()


def selector(kind: object, value: object) -> dict[str, str]:
    """A checked selector. A MAC is lowercased with colons; a fingerprint is 64 hex digits."""
    if kind not in SELECTORS:
        raise ValueError(f"a device selector is one of {', '.join(SELECTORS)}")
    v = _text(value, "selector")
    if kind == "net_mac":
        digits = "".join(c for c in v.lower() if c in "0123456789abcdef")
        if len(digits) != 12:
            raise ValueError("a MAC address has 12 hex digits")
        v = ":".join(digits[i : i + 2] for i in range(0, 12, 2))
    elif kind == "ble_fingerprint":
        v = v.lower()
        if len(v) != 64 or any(c not in "0123456789abcdef" for c in v):
            raise ValueError("a payload fingerprint is 64 hex digits")
    return {"kind": str(kind), "value": v}


@dataclass(frozen=True)
class Link:
    """An active owner link: a device, by selector, to a person or a thing."""

    id: str
    kind: str
    value: str
    target: Target
    name: str
    evidence: str  # the assertion's payload id


@dataclass(frozen=True)
class Location:
    """An active owner location: a device, by selector, lives in a place, or in one room of its plan."""

    id: str
    kind: str
    value: str
    place: str  # the place id (memory/places.py)
    place_name: str  # the place's name when it was set; a reader shows the current one (`location_options`)
    room: str | None  # a room of the place's plan, by name; None for the place as a whole
    evidence: str

    def as_dict(self, names: Mapping[str, str] | None = None) -> dict[str, Any]:
        """For the dashboard: the place by its current name in `names` (id: name) when it's there."""
        place_name = (names or {}).get(self.place, self.place_name)
        return {"id": self.id, "place_id": self.place, "place": place_name, "room": self.room}

    def words(self, names: Mapping[str, str] | None = None) -> str:
        """Where it lives in words: "the Living room (Home)", or "Home" for the whole place."""
        place_name = (names or {}).get(self.place, self.place_name)
        return f"the {self.room} ({place_name})" if self.room else place_name


def location_options(state: Path | None = None) -> list[dict[str, Any]]:
    """The places a device can live in: each place (memory/places.py) by its name, with the rooms of the plan that
    holds for it, in plan order. [] when the places can't be read."""
    from liberational.memory import places
    from liberational.spatial import plan_scene

    base = config.path("paths.state") if state is None else Path(state)
    try:
        found = places.places(base)
    except (OSError, ValueError, KeyError) as error:
        LOG.warning("places unreadable for device locations: %s", error)
        return []
    out = []
    for p in found:
        rooms = list(dict.fromkeys(name for _, name, _ in plan_scene.rooms_of(p.plan or "") if name.strip()))
        out.append({"id": p.id, "name": p.name, "rooms": rooms})
    return out


def parse_location(body: Mapping[str, Any], evidence: str) -> tuple[str, bool, Location]:
    """(location id, active, the location) from a stored revision. Raises ValueError for a malformed one."""
    if body.get("source") != "owner" or type(body.get("active")) is not bool:
        raise ValueError("invalid device location revision")
    ident = str(uuid.UUID(str(body.get("location_id"))))
    sel = selector((body.get("selector") or {}).get("kind"), (body.get("selector") or {}).get("value"))
    room = body.get("room")
    if room is not None and not isinstance(room, str):
        raise ValueError("a device location's room is text")
    place = str(body.get("place") or "")
    if not place:
        raise ValueError("a device location names a place")
    where = Location(ident, sel["kind"], sel["value"], place, str(body.get("place_name") or ""), room, evidence)
    return ident, body["active"], where


def read_locations(store: knowledge.Store) -> list[Location]:
    """The active locations, latest revision of each. Raises rather than return a partial list."""
    latest: dict[str, tuple[bool, Location]] = {}
    rows = store.query(LOCATIONS, limit=MAX_LINK_REVISIONS + 1)
    if len(rows) > MAX_LINK_REVISIONS:
        raise RuntimeError("device location history exceeds the safe read bound")
    for row in rows:
        try:
            ident, active, where = parse_location(row.record.value, row.record.id)
        except (ValueError, TypeError, AttributeError) as error:
            raise RuntimeError("a stored device location is malformed") from error
        latest[ident] = (active, where)
    return sorted((x for active, x in latest.values() if active), key=lambda x: (x.place_name, x.room or "", x.id))


def write_location(
    store: knowledge.Store, location_id: str, sel: Mapping[str, str], where: Mapping[str, Any], active: bool
) -> str:
    """Writes one revision of a device's location (`where`: place, place_name, room) and returns its id."""
    revision = str(uuid.uuid7())
    now = time.time_ns()
    body = {"location_id": location_id, "selector": dict(sel), **where, "active": active, "source": "owner"}
    store.put([knowledge.Record(revision, LOCATIONS, location_id, now, now, now, body)])
    return revision


# ---- the owner's links, on the broker ----


def parse_link(body: Mapping[str, Any], evidence: str) -> tuple[str, bool, Link]:
    """(link id, active, the link) from a stored revision. Raises ValueError for a malformed one."""
    if body.get("source") != "owner" or type(body.get("active")) is not bool:
        raise ValueError("invalid device link revision")
    ident = str(uuid.UUID(str(body.get("link_id"))))
    sel = selector((body.get("selector") or {}).get("kind"), (body.get("selector") or {}).get("value"))
    target = body.get("target")
    if target not in TARGETS:
        raise ValueError("a device link is to a person or a thing")
    link = Link(ident, sel["kind"], sel["value"], target, _text(body.get("name"), "name"), evidence)
    return ident, body["active"], link


def read_links(store: knowledge.Store) -> list[Link]:
    """The active links, latest revision of each in the order they were written. Raises rather than return a partial
    roster."""
    latest: dict[str, tuple[bool, Link]] = {}
    rows = store.query(LINKS, limit=MAX_LINK_REVISIONS + 1)
    if len(rows) > MAX_LINK_REVISIONS:
        raise RuntimeError("device link history exceeds the safe read bound")
    for row in rows:
        try:
            ident, active, link = parse_link(row.record.value, row.record.id)
        except (ValueError, TypeError, AttributeError) as error:
            raise RuntimeError("a stored device link is malformed") from error
        latest[ident] = (active, link)
    return sorted((link for active, link in latest.values() if active), key=lambda x: (x.name, x.id))


def write_link(
    store: knowledge.Store, link_id: str, sel: Mapping[str, str], target: str, name: str, active: bool
) -> str:
    """Writes one revision to the knowledge store (outside the broker's reset) and returns its id."""
    revision = str(uuid.uuid7())
    now = time.time_ns()
    body = {
        "link_id": link_id,
        "selector": dict(sel),
        "target": target,
        "name": name,
        "active": active,
        "source": "owner",
    }
    store.put([knowledge.Record(revision, LINKS, link_id, now, now, now, body)])
    return revision


# ---- sightings ----


def matches(link: Link, record: Mapping[str, Any], source: str) -> bool:
    """Whether a Bluetooth sighting or a network observation is the linked device."""
    if source == NET_OBSERVATIONS:
        mac = record.get("mac") or (
            record.get("identifier") if record.get("identifier_kind") == "mac_address" else None
        )
        return link.kind == "net_mac" and isinstance(mac, str) and mac.lower().replace("-", ":") == link.value
    if link.kind == "ble_identifier":
        return str(record.get("identifier")) == link.value
    if link.kind == "ble_name":
        return record.get("local_name") == link.value
    if link.kind == "ble_fingerprint":
        return ble.payload_fingerprint(dict(record)) == link.value
    return False


Signature = tuple[str | None, str | None, str | None]  # (identifier or MAC, payload fingerprint, advertised name)


def signature(record: Mapping[str, Any], source: str) -> Signature:
    """What a link can match a sighting by (`matches`), small enough to keep: a network device's MAC; a Bluetooth
    device's address, payload fingerprint, and advertised name."""
    if source == NET_OBSERVATIONS:
        mac = record.get("mac") or (
            record.get("identifier") if record.get("identifier_kind") == "mac_address" else None
        )
        return (mac.lower().replace("-", ":") if isinstance(mac, str) else None, None, None)
    name = record.get("local_name")
    return (
        str(record.get("identifier")),
        ble.payload_fingerprint(dict(record)),
        name if isinstance(name, str) else None,
    )


def signature_matches(link: Link | Location, sig: Signature, source: str) -> bool:
    """`matches` for a sighting's `signature`."""
    if source == NET_OBSERVATIONS:
        return link.kind == "net_mac" and sig[0] is not None and sig[0] == link.value
    if link.kind == "ble_identifier":
        return sig[0] == link.value
    if link.kind == "ble_fingerprint":
        return sig[1] is not None and sig[1] == link.value
    if link.kind == "ble_name":
        return sig[2] is not None and sig[2] == link.value
    return False


def selectors(record: Mapping[str, Any], source: str) -> list[dict[str, str]]:
    """The selectors a sighting's device can be linked by, steadiest first: a network device's MAC; a Bluetooth
    device's address, then its payload fingerprint (for an address that rotates), then its advertised name."""
    if source == NET_OBSERVATIONS:
        mac = record.get("mac") or (
            record.get("identifier") if record.get("identifier_kind") == "mac_address" else None
        )
        found = [("net_mac", mac)]
    else:
        found = [
            ("ble_identifier", record.get("identifier")),
            ("ble_fingerprint", ble.payload_fingerprint(dict(record))),
            ("ble_name", record.get("local_name")),
        ]
    out: list[dict[str, str]] = []
    for kind, value in found:
        try:
            out.append(selector(kind, value))
        except ValueError:
            continue
    return out


_MAC = re.compile(r"(?:[0-9a-f]{2}[:-]){5}[0-9a-f]{2}", re.IGNORECASE)
AP_NAME_MAX = 40


def norm_mac(value: object) -> str | None:
    """A MAC address as aa:bb:cc:dd:ee:ff, or None for anything that isn't one."""
    text = str(value or "").strip()
    return text.lower().replace("-", ":") if _MAC.fullmatch(text) else None


def parse_access_points(entries: Sequence[object]) -> dict[str, str]:
    """devices.access_points, "MAC: name" each, as {mac: name}. Raises ValueError naming the first bad entry."""
    out: dict[str, str] = {}
    for entry in entries:
        text = str(entry).strip()
        m = re.fullmatch(r"((?:[0-9a-fA-F]{2}[:-]){5}[0-9a-fA-F]{2})\s*[:=]\s*(.+)", text)
        name = " ".join(m[2].split()) if m else ""
        if m is None or not name:
            raise ValueError(f'{text!r} isn\'t "MAC: name", like "aa:bb:cc:dd:ee:ff: Living room"')
        if len(name) > AP_NAME_MAX:
            raise ValueError(f"{name!r} is longer than {AP_NAME_MAX} characters")
        out[str(norm_mac(m[1]))] = name
    return out


def access_point_names() -> dict[str, str]:
    """The owner's names for access points (devices.access_points), {mac: name}. An entry set by hand that doesn't
    parse is left out here, rather than failing every read; saving refuses it."""
    names: dict[str, str] = {}
    for entry in config.get("devices.access_points") or []:
        try:
            names.update(parse_access_points([entry]))
        except ValueError:
            continue
    return names


def place(record: Mapping[str, Any], source: str, access_points: Mapping[str, str] | None = None) -> str:
    """Where a sighting was heard: the sensor's name (or chip id), or the access point for a network one, by the
    owner's name for it (devices.access_points), else the router's, else its MAC."""
    if source == NET_OBSERVATIONS:
        mac = norm_mac(record.get("ap_id"))
        named = (access_points if access_points is not None else access_point_names()).get(mac or "")
        return str(named or record.get("ap_name") or record.get("ap_id") or "the network")
    return str(record.get("sensor_name") or record.get("adapter") or "a Bluetooth sensor")


def place_id(record: Mapping[str, Any], source: str) -> str | None:
    """The access point's MAC a network sighting was heard through, for naming it; None for anything else."""
    return norm_mac(record.get("ap_id")) if source == NET_OBSERVATIONS else None


def rssi(record: Mapping[str, Any]) -> float | None:
    """A sighting's strongest signal in dBm, or None."""
    r = record.get("rssi")
    value = r.get("max") if isinstance(r, dict) else None
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


@dataclass(frozen=True)
class Stay:
    """Sightings of one target no more than a gap apart: arrived at `start`, last seen at `end`."""

    start: float
    end: float
    first_id: str
    last_id: str
    places: tuple[str, ...]
    sightings: int


Sighting = tuple[float, float, str, str] | tuple[float, float, str, str, int]


def stays(sightings: Iterable[Sighting], gap: float) -> list[Stay]:
    """(start, end, id, place[, count]) sightings merged into stays where each is within `gap` of the last. A
    sighting counts once, or `count` times (a compacted run of polls)."""
    out: list[Stay] = []
    for sighting in sorted(sightings):
        a, b, ident, where = sighting[:4]
        n = sighting[4] if len(sighting) > 4 else 1
        last = out[-1] if out else None
        if last is not None and a - last.end <= gap:
            places = last.places if where in last.places else (*last.places, where)
            out[-1] = Stay(last.start, max(last.end, b), last.first_id, ident, places, last.sightings + n)
        else:
            out.append(Stay(a, b, ident, ident, (where,), n))
    return out


@dataclass(frozen=True)
class Seen:
    """What the sightings say about one target over a window."""

    links: tuple[Link, ...]
    stays: tuple[Stay, ...]
    last: dict[str, Any] | None  # the newest matching sighting, with "source"
    sensing: tuple[tuple[float, float], ...]  # applicable source coverage, not an absence assertion
    coverage: tuple[dict[str, Any], ...] = ()  # preserve stream provenance and unknown/partial poll states


def _link_source(link: Link) -> str:
    return NET_INVENTORY if link.kind == "net_mac" else BLE_COVERAGE


def _negative_coverage(row: Mapping[str, Any]) -> bool:
    """Absence needs an explicit, applicable source assertion, never just a running collector.

    BLE coverage currently declares absence_is_not_inferred: advertisements can be missed even with an active
    radio. Partial/cached network inventories likewise retain unknown coverage. Positive sightings remain useful.
    """
    if row.get("absence_is_not_inferred") is True:
        return False
    if row.get("stream") == NET_INVENTORY:
        return row.get("complete") is True and row.get("authoritative") is True
    if row.get("stream") != BLE_COVERAGE or row.get("absence_is_not_inferred") is not False:
        return False
    source = row.get("source")
    if (
        not isinstance(source, Mapping)
        or not isinstance(source.get("active_sensors"), int)
        or source["active_sensors"] <= 0
    ):
        return False
    for counters in (source, row.get("aggregation", {}), row):
        if not isinstance(counters, Mapping):
            return False
        for key, value in counters.items():
            if (
                any(
                    word in str(key)
                    for word in ("lost", "drop", "invalid", "evict", "failure", "unacknowledged", "queued")
                )
                and value
            ):
                return False
    return True


def home_status(seen: Seen, now: float, gap: float) -> Literal["recent", "not_seen", "unknown"]:
    """Current linked-device evidence. A recent positive wins; every linked source must support a negative.

    The newest poll per source matters: a partial poll after a complete one restores uncertainty. This is device
    evidence only, never proof its owner is present or absent. Unmarked legacy coverage is conservatively unknown.
    """
    if seen.last is not None and 0 <= now - float(seen.last["end"]) <= gap:
        return "recent"
    required = {_link_source(link) for link in seen.links}
    if not required:
        return "unknown"
    for source in required:
        rows = [row for row in seen.coverage if row.get("stream") == source and float(row["end"]) <= now]
        if not rows:
            return "unknown"
        latest = max(float(row["end"]) for row in rows)
        if now - latest > gap or not all(_negative_coverage(row) for row in rows if float(row["end"]) == latest):
            return "unknown"
    return "not_seen"


def _sensing(rows: Iterable[Mapping[str, Any]], gap: float) -> list[tuple[float, float]]:
    """Coverage records merged into the stretches a source was running."""
    spans: list[tuple[float, float]] = []
    for r in sorted(rows, key=lambda r: r["start"]):
        a, b = float(r["start"]), float(r["end"])
        if spans and a - spans[-1][1] <= gap:
            spans[-1] = (spans[-1][0], max(spans[-1][1], b))
        else:
            spans.append((a, b))
    return spans


def seen(
    links: Sequence[Link], records: Mapping[str, list[dict[str, Any]]], coverage: Sequence[dict[str, Any]], gap: float
) -> Seen:
    """The stays, newest sighting, and sensing over the records read, for one target's links."""
    hits: list[Sighting] = []
    newest: dict[str, Any] | None = None
    aps = access_point_names()
    # A payload fingerprint is the one rule that has to parse the whole sighting, so a row's signature is built once
    # and only when a link matches by fingerprint; an address or name link reads a field and stays cheap.
    wants_fp = any(link.kind == "ble_fingerprint" for link in links)
    for source, rows in records.items():
        for r in rows:
            # A router row counts only when the controller says it is connected now: ARP and DHCP cache rows
            # (connected null) and expired leases are history, not presence.
            if source == NET_OBSERVATIONS and r.get("connected") is not True:
                continue
            sig = signature(r, source) if wants_fp else None
            hit = (
                any(signature_matches(link, sig, source) for link in links)
                if sig is not None
                else any(matches(link, r, source) for link in links)
            )
            if hit:
                hits.append((float(r["start"]), float(r["end"]), str(r["id"]), place(r, source, aps), 1))
                if newest is None or float(r["end"]) >= float(newest["end"]):
                    newest = {**r, "source": source}
    applicable = {_link_source(link) for link in links}
    scoped = [dict(row) for row in coverage if row.get("stream") in applicable]
    return Seen(tuple(links), tuple(stays(hits, gap)), newest, tuple(_sensing(scoped, gap)), tuple(scoped))


@dataclass
class _Tally:
    """One device's sightings over a stretch, as `Devices.summary` shows it: how many, the newest (by end), the newest
    name it gave, and each distinct `signature` with its newest end (what a link matches, whichever links exist)."""

    sightings: int = 0
    last_end: float = -math.inf
    last: dict[str, Any] = field(default_factory=dict)
    name: tuple[float, str] | None = None
    signatures: dict[Signature, float] = field(default_factory=dict)

    def add(self, r: Mapping[str, Any], source: str) -> None:
        end = float(r["end"])
        self.sightings += 1
        if end >= self.last_end:
            self.last_end, self.last = end, dict(r)
        named = r.get("local_name") or r.get("hostname") or r.get("display_name")
        if named and (self.name is None or end >= self.name[0]):
            self.name = (end, str(named))
        sig = signature(r, source)
        self.signatures[sig] = max(self.signatures.get(sig, -math.inf), end)

    def then(self, later: _Tally) -> None:
        """Fold in a later stretch's tally of the same device."""
        self.sightings += later.sightings
        if later.last_end >= self.last_end:
            self.last_end, self.last = later.last_end, later.last
        if later.name is not None and (self.name is None or later.name[0] >= self.name[0]):
            self.name = later.name
        for sig, end in later.signatures.items():
            self.signatures[sig] = max(self.signatures.get(sig, -math.inf), end)


def _key(r: Mapping[str, Any]) -> str:
    return str(r.get("identifier") or r.get("mac"))


def _tally(rows: Iterable[Mapping[str, Any]], source: str) -> dict[str, _Tally]:
    out: dict[str, _Tally] = {}
    for r in sorted(rows, key=lambda r: float(r["end"])):
        out.setdefault(_key(r), _Tally()).add(r, source)
    return out


SUMMARY_SOURCES = (BLE_SIGHTINGS, NET_OBSERVATIONS)
# Hour tallies kept per process, both sources together (about four days of each), oldest dropped first. Each holds the
# newest row of every device heard that hour: a few tens of MB in all on the Linux machine.
KEPT_HOURS = 24 * 8
TALLY_CACHE = Path("cache") / "device_tallies.json"  # under paths.state: the kept hours, across restarts
TALLY_CACHE_VERSION = 1  # a cache of another version is ignored and rebuilt


def _tally_json(t: _Tally) -> dict[str, Any]:
    return {
        "sightings": t.sightings,
        "last_end": t.last_end,
        "last": t.last,
        "name": list(t.name) if t.name is not None else None,
        "signatures": [[list(sig), end] for sig, end in t.signatures.items()],
    }


def _tally_of(d: Mapping[str, Any]) -> _Tally:
    name = d["name"]
    return _Tally(
        int(d["sightings"]),
        float(d["last_end"]),
        dict(d["last"]),
        (float(name[0]), str(name[1])) if name is not None else None,
        {(sig[0], sig[1], sig[2]): float(end) for sig, end in d["signatures"]},
    )


class Devices:
    """Synchronous access for Ask, the dashboard, and the CLI: links in the knowledge store, sightings in segments."""

    def __init__(
        self, *, segments: Path | None = None, knowledge_db: Path | None = None, tally_cache: Path | None = None
    ) -> None:
        self.segments = segments if segments is not None else segment_log.root()
        self.tally_cache = tally_cache  # where the kept hours are saved across restarts, or None to keep them in memory
        self.knowledge = knowledge.Store(knowledge_db if knowledge_db is not None else registry.knowledge_path())
        self.lock = threading.RLock()
        self._hours: dict[tuple[str, float], tuple[Any, dict[str, _Tally]]] = {}  # (source, hour): (stamp, tallies)
        self._hours_lock = threading.Lock()
        self._hours_loaded = False  # the saved hours read in (the first summary does it)
        self._hours_new = False  # an hour tallied since the last save

    def links(self) -> list[Link]:
        return read_links(self.knowledge)

    def save(self, kind: str, value: str, target: str, name: str) -> Link:
        """Links a device (a new link, or the same selector's existing one revised)."""
        sel = selector(kind, value)
        if target not in TARGETS:
            raise ValueError("a device is linked to a person or named as a thing")
        name = _text(name, "name")
        with self.lock:
            prior = next((x for x in self.links() if (x.kind, x.value) == (sel["kind"], sel["value"])), None)
            ident = prior.id if prior is not None else str(uuid.uuid7())

            evidence = write_link(self.knowledge, ident, sel, target, name, True)
        return Link(ident, sel["kind"], sel["value"], target, name, evidence)  # type: ignore[arg-type]

    def remove(self, link_id: str) -> None:
        """Retires a link; the sightings stay."""
        with self.lock:
            prior = next((x for x in self.links() if x.id == link_id), None)
            if prior is None:
                raise LookupError("device link not found")

            sel = {"kind": prior.kind, "value": prior.value}
            write_link(self.knowledge, prior.id, sel, prior.target, prior.name, False)

    def locations(self) -> list[Location]:
        return read_locations(self.knowledge)

    def locate(self, kind: str, value: str, place: str, room: str, state: Path | None = None) -> Location:
        """Says a device lives in `place`, in its plan's `room` ("" for the place as a whole): a new location, or the
        same selector's revised. Raises ValueError for a place or room that isn't there."""
        sel = selector(kind, value)
        option = next((o for o in location_options(state) if o["id"] == str(place)), None)
        if option is None:
            raise ValueError(f"there's no place {place}")
        room = " ".join(str(room or "").split())
        if room and room not in option["rooms"]:
            raise ValueError(f"{option['name']} has no room {room!r}")
        where = {"place": option["id"], "place_name": option["name"], "room": room or None}
        with self.lock:
            prior = next((x for x in self.locations() if (x.kind, x.value) == (sel["kind"], sel["value"])), None)
            ident = prior.id if prior is not None else str(uuid.uuid7())
            evidence = write_location(self.knowledge, ident, sel, where, True)
        return Location(ident, sel["kind"], sel["value"], option["id"], option["name"], room or None, evidence)

    def unlocate(self, location_id: str) -> None:
        """Retires a device's location."""
        with self.lock:
            prior = next((x for x in self.locations() if x.id == location_id), None)
            if prior is None:
                raise LookupError("device location not found")
            where = {"place": prior.place, "place_name": prior.place_name, "room": prior.room}
            write_location(self.knowledge, prior.id, {"kind": prior.kind, "value": prior.value}, where, False)

    def read(
        self, start: float, end: float, links: Sequence[Link] | None = None, *, coverage: bool = True
    ) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
        """(sightings by source, sensing coverage) over [start, end). With `links`, only what those links can match
        (`seen` reads nothing else): a source none of them names isn't read, and a Bluetooth sighting is parsed only
        when its text holds a linked address or name, unless a link matches by payload fingerprint, which takes the
        parsed payload. Without `coverage`, no sensing coverage is read."""

        log = segment_log.Log(self.segments)
        kinds = None if links is None else {link.kind for link in links}
        records: dict[str, list[dict[str, Any]]] = {BLE_SIGHTINGS: [], NET_OBSERVATIONS: []}
        if kinds is None or "ble_fingerprint" in kinds:
            records[BLE_SIGHTINGS] = log.range(BLE_SIGHTINGS, start, end)
        elif kinds & {"ble_identifier", "ble_name"}:
            wanted = sorted(
                {segment_log.json_text(x.value) for x in links or () if x.kind in ("ble_identifier", "ble_name")}
            )
            records[BLE_SIGHTINGS] = log.containing(BLE_SIGHTINGS, start, end, wanted)
        if kinds is None or "net_mac" in kinds:
            records[NET_OBSERVATIONS] = log.range(NET_OBSERVATIONS, start, end)
        if not coverage:
            return records, []
        # Preserve partial/inactive states and source provenance; consumers decide applicability.
        sources = (BLE_COVERAGE, NET_INVENTORY) if links is None else sorted({_link_source(x) for x in links})
        return records, [{**row, "stream": s} for s in sources for row in log.range(s, start, end)]

    def target(self, name: str, target: str, start: float, end: float) -> Seen | None:
        """What the sightings say about the person or thing called `name`, or None when no link names it."""
        mine = [x for x in self.links() if x.target == target and x.name.casefold() == name.casefold()]
        if not mine:
            return None
        records, coverage = self.read(start, end, mine)
        return seen(mine, records, coverage, gap_s())

    def people(self, start: float, end: float) -> list[tuple[str, Seen]]:
        """One snapshot of linked people: links once and shared observations/coverage once, without caching."""
        groups: dict[str, list[Link]] = {}
        for link in self.links():
            if link.target == "person":
                groups.setdefault(link.name.casefold(), []).append(link)
        if not groups:
            return []
        records, coverage = self.read(start, end, [link for mine in groups.values() for link in mine])
        gap = gap_s()
        return [(mine[0].name, seen(mine, records, coverage, gap)) for _, mine in sorted(groups.items())]

    def whereabouts(self, end: float, days: int) -> list[dict[str, Any]]:
        """Each linked person's and thing's newest sighting before `end`: when, where (the sensor or access point),
        and how strongly, or None when it wasn't seen. It looks back a day at a time, as Ask does for a thing
        (devices.thing_lookback_days), reading each day once for every link and stopping once each has been seen.
        Things come first, then people, each by name."""
        groups: dict[tuple[str, str], list[Link]] = {}
        aps = access_point_names()
        for link in self.links():
            groups.setdefault((link.target, link.name.casefold()), []).append(link)
        newest: dict[tuple[str, str], dict[str, Any] | None] = dict.fromkeys(groups)
        gap = gap_s()
        hi = end
        for _ in range(max(0, days)):
            if all(v is not None for v in newest.values()):
                break
            lo = hi - 86400.0
            unseen = [link for key, links in groups.items() if newest[key] is None for link in links]
            records, _ = self.read(lo, hi, unseen, coverage=False)
            for key, links in groups.items():
                if newest[key] is None:
                    newest[key] = seen(links, records, (), gap).last
            hi = lo
        rows = []
        for (target, _), links in sorted(groups.items(), key=lambda kv: (kv[0][0] != "thing", kv[0][1])):
            last = newest[(target, links[0].name.casefold())]
            rows.append(
                {
                    "target": target,
                    "name": links[0].name,
                    "links": len(links),
                    "last_seen": float(last["end"]) if last else None,
                    "place": place(last, last["source"], aps) if last else None,
                    "place_id": place_id(last, last["source"]) if last else None,
                    "rssi": rssi(last) if last else None,
                    "source": last["source"] if last else None,
                }
            )
        return rows

    def comings(self, start: float, end: float) -> dict[str, Any]:
        """Every device the router said was connected over [start, end): its stays (arrived, last seen), each run of
        observations no more than devices.gap_s apart, with its name and link, newest first. Only rows the router
        marks connected count, as for presence. `rows` is how many observations were read, for its cost."""
        links = self.links()
        located = self.locations()
        aps = access_point_names()
        rows = segment_log.Log(self.segments).range(NET_OBSERVATIONS, start, end)
        by_mac: dict[str, dict[str, Any]] = {}
        hits: dict[str, list[Sighting]] = {}
        for r in rows:
            if r.get("connected") is not True:
                continue
            mac = str(r.get("identifier") or r.get("mac"))
            sig = signature(r, NET_OBSERVATIONS)  # one MAC normalization per row, not one per location and link
            d = by_mac.setdefault(
                mac,
                {
                    "mac": mac,
                    "name": None,
                    "vendor": None,
                    "place": None,
                    "place_id": None,
                    "linked": None,
                    "location": next(
                        (x.as_dict() for x in located if signature_matches(x, sig, NET_OBSERVATIONS)),
                        None,
                    ),
                },
            )
            d["name"] = r.get("hostname") or r.get("display_name") or d["name"]
            if d["vendor"] is None:
                d["vendor"] = vendors.network(r)
            d["place"] = place(r, NET_OBSERVATIONS, aps)
            d["place_id"] = place_id(r, NET_OBSERVATIONS)
            if d["linked"] is None:
                hit = next((x for x in links if signature_matches(x, sig, NET_OBSERVATIONS)), None)
                if hit is not None:
                    d["linked"] = {"id": hit.id, "target": hit.target, "name": hit.name}
            hits.setdefault(mac, []).append((float(r["start"]), float(r["end"]), str(r["id"]), d["place"], 1))
        gap = gap_s()
        out = []
        for mac, d in by_mac.items():
            runs = stays(hits[mac], gap)
            out.append({**d, "stays": [{"start": s.start, "end": s.end} for s in runs]})
        out.sort(key=lambda d: -d["stays"][-1]["end"])
        return {"devices": out, "rows": len(rows), "access_points": aps}

    def summary(self, start: float, end: float) -> dict[str, Any]:
        """Every device seen over [start, end), linked or not, counted: unknown devices stay on record. Each says
        where it was last heard and how strongly, the selectors it can be linked by, and its link (with the link's
        id) when it has one; `links` is every active link, seen in the window or not.

        A day is about half a million Bluetooth rows on the Linux machine, 20 s of CPU to read and as much again to go through,
        and Known › Devices asks for it on every load. So each whole hour over by segment_log.SETTLE_S is
        tallied once per process and kept (`_Tally`, checked against the store's stamp, so a late row or a purge
        shows), and only the window's first part hour and the hours still going are read on each call."""
        links = self.links()
        located = self.locations()
        aps = access_point_names()
        devices: dict[tuple[str, str], dict[str, Any]] = {}
        for source in SUMMARY_SOURCES:
            for key, t in self._tallies(source, start, end).items():
                r = t.last
                hit = None
                where: Location | None = None
                for sig, _end in sorted(t.signatures.items(), key=lambda kv: -kv[1]):
                    hit = hit or next((x for x in links if signature_matches(x, sig, source)), None)
                    where = where or next((x for x in located if signature_matches(x, sig, source)), None)
                    if hit is not None and where is not None:
                        break
                vendor, kind = vendors.bluetooth(r) if source == BLE_SIGHTINGS else (vendors.network(r), None)
                devices[(source, key)] = {
                    "source": source,
                    "identifier": key,
                    "vendor": vendor,
                    "kind": kind,
                    "sightings": t.sightings,
                    "linked": {"id": hit.id, "target": hit.target, "name": hit.name} if hit is not None else None,
                    "location": where.as_dict() if where is not None else None,
                    "last_seen": t.last_end,
                    "place": place(r, source, aps),
                    "place_id": place_id(r, source),
                    "rssi": rssi(r),
                    "name": t.name[1] if t.name is not None else None,
                    "selectors": selectors(r, source),
                }
        self._save_hours()
        rows = sorted(devices.values(), key=lambda d: -d["last_seen"])
        return {
            "devices": rows,
            "total": len(rows),
            "linked": sum(1 for d in rows if d["linked"]),
            "unknown": sum(1 for d in rows if not d["linked"]),
            "links": [
                {"id": x.id, "kind": x.kind, "value": x.value, "target": x.target, "name": x.name} for x in links
            ],
            "locations": [{**x.as_dict(), "kind": x.kind, "value": x.value} for x in located],
        }

    def _tallies(self, source: str, start: float, end: float) -> dict[str, _Tally]:
        """`source`'s tallies over [start, end) by device, an hour at a time: a row counts in the hour it starts in
        (or the window's first, for one that began before it), so a row spanning an hour's end counts once."""
        log = segment_log.Log(self.segments)
        settled = time.time() - segment_log.SETTLE_S
        out: dict[str, _Tally] = {}
        at = start
        while at < end:
            upto = min(end, (math.floor(at / segment_log.HOUR_S) + 1) * segment_log.HOUR_S)
            if at > start and upto - at == segment_log.HOUR_S and upto <= settled:
                part = self._hour(log, source, at, upto)
            else:
                lo = at
                part = _tally((r for r in log.range(source, at, upto) if lo <= max(_began(r), start) < upto), source)
            for key, t in part.items():
                if key in out:
                    out[key].then(t)
                else:
                    out[key] = _Tally(t.sightings, t.last_end, t.last, t.name, dict(t.signatures))
            at = upto
        return out

    def _hour(self, log: segment_log.Log, source: str, at: float, upto: float) -> dict[str, _Tally]:
        """The whole, settled hour [at, upto)'s tallies, kept while the store's stamp of it holds."""
        stamp = log.stamp(at, upto, (source,))
        self._load_hours()
        with self._hours_lock:
            kept = self._hours.get((source, at))
        if kept is not None and kept[0] == stamp:
            return kept[1]
        tallies = _tally((r for r in log.range(source, at, upto) if at <= _began(r) < upto), source)
        with self._hours_lock:
            self._hours[(source, at)] = (stamp, tallies)
            self._hours_new = True
            while len(self._hours) > KEPT_HOURS:
                del self._hours[min(self._hours, key=lambda k: k[1])]
        return tallies

    def _load_hours(self) -> None:
        """Read the hours a past process saved (`tally_cache`), once. Each is still checked against the store's stamp
        before it's used, so one the store has changed since is tallied again. A missing, unreadable, or other-version
        file is nothing: the hours are tallied from the store."""
        with self._hours_lock:
            if self._hours_loaded:
                return
            self._hours_loaded = True
            if self.tally_cache is None:
                return
            try:
                saved = orjson.loads(self.tally_cache.read_bytes())
                if saved.get("version") != TALLY_CACHE_VERSION:
                    return
                hours = {
                    (str(h["source"]), float(h["hour"])): (
                        tuple(tuple(x) for x in h["stamp"]),
                        {key: _tally_of(t) for key, t in h["tallies"].items()},
                    )
                    for h in saved["hours"]
                }
            except OSError, ValueError, KeyError, TypeError, IndexError, AttributeError:
                return  # a cache: rebuilt from the store
            self._hours = {**hours, **self._hours}

    def _save_hours(self) -> None:
        """Save the kept hours (`tally_cache`) when a new one was tallied, so a restart starts with them. Written whole
        to a temporary file and moved into place; a failed save leaves the old file."""
        with self._hours_lock:
            if self.tally_cache is None or not self._hours_new:
                return
            self._hours_new = False
            hours = [
                {
                    "source": source,
                    "hour": hour,
                    "stamp": [list(x) for x in stamp],
                    "tallies": {key: _tally_json(t) for key, t in tallies.items()},
                }
                for (source, hour), (stamp, tallies) in self._hours.items()
            ]
        tmp = self.tally_cache.with_name(f".{self.tally_cache.name}.{os.getpid()}.{threading.get_ident()}")
        try:
            self.tally_cache.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(orjson.dumps({"version": TALLY_CACHE_VERSION, "hours": hours}))
            os.replace(tmp, self.tally_cache)
        except (OSError, TypeError) as error:  # TypeError: a value orjson can't write (an int past 64 bits)
            LOG.warning("device tallies not saved: %s", error)
            tmp.unlink(missing_ok=True)


def _began(r: Mapping[str, Any]) -> float:
    return float(r.get("ts_start", r.get("start", 0.0)))


_DEVICES: dict[Path, Devices] = {}
_LOCK = threading.Lock()


def at(state: str | Path | None = None) -> Devices:
    """The device links and sightings for the configured state directory, or another's (tests)."""
    own = state is None or Path(state).resolve() == config.path("paths.state").resolve()
    base = None if own else state
    key = registry.knowledge_path(base)
    with _LOCK:
        if key not in _DEVICES:
            state_dir = config.path("paths.state") if base is None else Path(base)
            _DEVICES[key] = Devices(
                segments=segment_log.root(base), knowledge_db=key, tally_cache=state_dir / TALLY_CACHE
            )
        return _DEVICES[key]


def main(argv: Sequence[str] | None = None) -> None:
    """`pack-devices`: list the devices seen, and link one to a person or name it as a thing."""
    import argparse
    import datetime as dt

    parser = argparse.ArgumentParser(prog="pack-devices", description=main.__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ls = sub.add_parser("list", help="every device seen, linked or not")
    ls.add_argument("--hours", type=float, default=24.0)
    links = sub.add_parser("links", help="the owner's links")
    links.set_defaults()
    add = sub.add_parser("link", help="link a device to a person or name it as a thing")
    add.add_argument("selector", choices=SELECTORS)
    add.add_argument("value", help="the address, fingerprint, advertised name, or MAC")
    group = add.add_mutually_exclusive_group(required=True)
    group.add_argument("--person")
    group.add_argument("--thing")
    rm = sub.add_parser("unlink", help="retire a link (its sightings stay)")
    rm.add_argument("link_id")
    args = parser.parse_args(argv)
    reg = at()
    if args.command == "list":
        now = time.time()
        found = reg.summary(now - args.hours * 3600, now)
        print(f"{found['total']} devices: {found['linked']} linked, {found['unknown']} unknown")
        for d in found["devices"]:
            seen_at = dt.datetime.fromtimestamp(d["last_seen"]).strftime("%Y-%m-%d %H:%M")
            linked = f"{d['linked']['target']} {d['linked']['name']}" if d["linked"] else "unknown"
            print(f"{seen_at}  {d['source']:<28} {d['identifier']:<40} {d['place']:<16} {d['sightings']:>6}  {linked}")
    elif args.command == "links":
        for x in reg.links():
            print(f"{x.id}  {x.target:<6} {x.name:<20} {x.kind} {x.value}")
    elif args.command == "link":
        made = reg.save(args.selector, args.value, "person" if args.person else "thing", args.person or args.thing)
        print(f"linked {made.kind} {made.value} to {made.target} {made.name} ({made.id})")
    else:
        reg.remove(args.link_id)
        print(f"unlinked {args.link_id}")
