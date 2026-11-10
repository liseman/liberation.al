"""Settings' phones, watches, and tags (#1363): the devices seen, and linking one to a person or naming it as a thing.

The routes read and write proximity/devices.py (#1329), the same links `pack-devices` makes. A link is an owner
assertion, so only this machine changes one.
GET /device-presence is a separate read-only, bounded linked-person evidence snapshot for Live, and
GET /device-whereabouts where each linked thing and person was last seen, for Known › Devices (#1326).
"""

import datetime as dt
import math
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from liberational import config, logs
from liberational.proximity import devices

LOG = logs.get("dashboard")
DEFAULT_HOURS = 24.0
MAX_HOURS = 24.0 * 31
UNAVAILABLE = "Couldn't reach the recordings store; try again in a minute."
PRESENCE_WINDOW_MAX_S = 3600.0
DEVICE_CAVEAT = "A linked device was seen; that is not proof its owner is here."


def presence(now: float | None = None) -> tuple[int, dict[str, Any]]:
    """GET /device-presence: one bounded, read-only snapshot for Live, using Ask's evidence policy.

    The current-state window covers the configured freshness gap, up to one hour. A larger configured gap can
    still produce a known positive, but a clipped read never certifies a negative. No owner link is created here.
    """
    end = time.time() if now is None else now
    try:
        gap = devices.gap_s()
        if not math.isfinite(gap) or gap <= 0:
            raise ValueError("devices.gap_s must be finite and positive")
        window = min(max(gap, 300.0), PRESENCE_WINDOW_MAX_S)
        start = end - window
        rows = []
        for name, seen in devices.at().people(start, end):
            status = devices.home_status(seen, end, gap)
            if gap > window and status == "not_seen":
                status = "unknown"
            last = seen.last
            rows.append(
                dict(
                    name=name,
                    status=status,
                    last_seen=float(last["end"]) if last else None,
                    source=last["source"] if last else None,
                    place=devices.place(last, last["source"]) if last else None,
                    links=len(seen.links),
                )
            )
    except Exception:
        LOG.warning("linked-device presence is unavailable", exc_info=True)
        return 503, {"error": UNAVAILABLE}
    return 200, {"at": end, "start": start, "gap_seconds": gap, "people": rows, "caveat": DEVICE_CAVEAT}


REUSE_S = 60.0  # how long a whereabouts, comings, or links reply is reused: a cold read is a day or more of rows
_held: dict[str, dict[str, Any]] = {}
_held_lock = threading.Lock()
_reading: dict[str, threading.Lock] = {}  # one per key: requests arriving together share one read
_epoch = [0]  # moved by _forget: a read that began before a link changed is never kept


def _reuse(key: str, end: float, make: Callable[[], dict[str, Any]], what: str) -> tuple[int, dict[str, Any]]:
    """`make()`'s reply for `key`, reused for REUSE_S, and shared by the requests that arrive while it's read (each
    key reads on its own, so a slow one holds up no other); a failed read isn't kept."""
    with _held_lock:
        reading = _reading.setdefault(key, threading.Lock())
    with reading:
        with _held_lock:
            held, epoch = _held.get(key), _epoch[0]
        if held is not None and 0 <= end - held["at"] < REUSE_S:
            return 200, held
        try:
            reply = {"at": end, **make()}
        except Exception:
            LOG.warning("%s are unavailable", what, exc_info=True)
            return 503, {"error": UNAVAILABLE}
        with _held_lock:
            if _epoch[0] == epoch:
                _held[key] = reply
        return 200, reply


def _forget() -> None:
    """A link changed: what's held names the old links."""
    with _held_lock:
        _held.clear()
        _epoch[0] += 1


def whereabouts(now: float | None = None) -> tuple[int, dict[str, Any]]:
    """GET /device-whereabouts: where each linked thing and person was last seen (#1326, "where my things are"), as
    Known › Devices shows it, looking back devices.thing_lookback_days."""
    end = time.time() if now is None else now
    days = int(config.get("devices.thing_lookback_days"))
    return _reuse(
        "whereabouts",
        end,
        lambda: {
            "days": days,
            "links": devices.at().whereabouts(end, days),
            "access_points": devices.access_point_names(),
            "caveat": DEVICE_CAVEAT,
        },
        "device whereabouts",
    )


def comings(now: float | None = None) -> tuple[int, dict[str, Any]]:
    """GET /device-comings: every device on the home Wi-Fi today (#1326, "what's on the Wi-Fi and when each comes
    and goes"): when it arrived and left, a stay at a time, newest first, as Known › Devices shows it."""
    end = time.time() if now is None else now
    local = dt.datetime.fromtimestamp(end).astimezone()
    start = local.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    return _reuse(
        "comings",
        end,
        lambda: {"start": start, "gap_seconds": devices.gap_s(), **devices.at().comings(start, end)},
        "Wi-Fi comings and goings",
    )


def listing(hours: str, now: float | None = None, state: Path | None = None) -> tuple[int, dict[str, Any]]:
    """GET /device-links?hours=: the devices seen in the last `hours` (24 by default) and every active link."""
    try:
        span = float(hours) if hours else DEFAULT_HOURS
    except ValueError:
        span = -1.0
    if not 0 < span <= MAX_HOURS:
        return 400, {"error": f"hours must be more than 0 and at most {MAX_HOURS:g}"}
    end = time.time() if now is None else now

    def make() -> dict[str, Any]:
        found = devices.at().summary(end - span * 3600, end)
        options = devices.location_options(state)
        names = {o["id"]: o["name"] for o in options}
        for where in [*found["locations"], *(d["location"] for d in found["devices"] if d["location"])]:
            where["place"] = names.get(where["place_id"], where["place"])  # the place's name now
        return {**found, "places": options}

    code, reply = _reuse(f"links:{span:g}", end, make, "device links")
    return code, ({k: v for k, v in reply.items() if k != "at"} | {"hours": span}) if code == 200 else reply


def _text(body: dict[str, Any], key: str) -> str:
    """A string field, or "" (which save() refuses with its own message)."""
    value = body.get(key)
    return value if isinstance(value, str) else ""


def save(body: object, control: bool) -> tuple[int, dict[str, Any]]:
    """POST /device-links {kind, value, target, name}: link a device (or revise its link)."""
    if not control:
        return 403, {"ok": False, "error": "devices can only be linked on this machine"}
    if not isinstance(body, dict):
        return 400, {"ok": False, "error": "send a JSON object"}
    try:
        made = devices.at().save(*(_text(body, k) for k in ("kind", "value", "target", "name")))
    except ValueError as exc:
        return 400, {"ok": False, "error": str(exc)}
    except Exception:
        LOG.warning("device link not saved", exc_info=True)
        return 503, {"ok": False, "error": UNAVAILABLE}
    link = {"id": made.id, "kind": made.kind, "value": made.value, "target": made.target, "name": made.name}
    _forget()
    return 200, {"ok": True, "link": link}


def name_access_point(body: object, control: bool) -> tuple[int, dict[str, Any]]:
    """POST /device-access-point {mac, name}: name a Wi-Fi access point the router reports by MAC alone, for Known ›
    Devices (devices.access_points in pack.toml, as camera names are in vision.places); an empty name clears it."""
    if not control:
        return 403, {"ok": False, "error": "access points can only be named on this machine"}
    if not isinstance(body, dict):
        return 400, {"ok": False, "error": "send a JSON object"}
    mac = devices.norm_mac(body.get("mac"))
    name = " ".join(str(body.get("name") or "").split())
    if mac is None:
        return 400, {"ok": False, "error": "mac must be an access point's MAC address, like aa:bb:cc:dd:ee:ff"}
    names = devices.access_point_names()
    if name:
        names[mac] = name
    else:
        names.pop(mac, None)
    try:
        config.save({"devices.access_points": [f"{m}: {n}" for m, n in sorted(names.items())]})
    except config.SettingError as exc:
        return 400, {"ok": False, "error": str(exc)}
    except OSError as exc:
        return 500, {"ok": False, "error": f"couldn't write {config.local_config()}: {exc.strerror}"}
    _forget()
    return 200, {"ok": True, "access_points": devices.access_point_names()}


def remove(body: object, control: bool) -> tuple[int, dict[str, Any]]:
    """POST /device-links/remove {id}: retire a link; its sightings stay."""
    if not control:
        return 403, {"ok": False, "error": "devices can only be unlinked on this machine"}
    ident = body.get("id") if isinstance(body, dict) else None
    if not isinstance(ident, str) or not ident:
        return 400, {"ok": False, "error": "id is required"}
    try:
        devices.at().remove(ident)
        _forget()
    except LookupError as exc:
        return 404, {"ok": False, "error": str(exc)}
    except Exception:
        LOG.warning("device link not removed", exc_info=True)
        return 503, {"ok": False, "error": UNAVAILABLE}
    return 200, {"ok": True}


def locate(body: object, control: bool, state: Path | None = None) -> tuple[int, dict[str, Any]]:
    """POST /device-locations {kind, value, place, room}: say a device lives in a place, or a room of its plan ("" for
    the whole place), or move it (#3274)."""
    if not control:
        return 403, {"ok": False, "error": "device locations can only be set on this machine"}
    if not isinstance(body, dict):
        return 400, {"ok": False, "error": "send a JSON object"}
    try:
        made = devices.at().locate(*(_text(body, k) for k in ("kind", "value", "place", "room")), state=state)
    except ValueError as exc:
        return 400, {"ok": False, "error": str(exc)}
    except Exception:
        LOG.warning("device location not saved", exc_info=True)
        return 503, {"ok": False, "error": UNAVAILABLE}
    _forget()
    return 200, {"ok": True, "location": {**made.as_dict(), "kind": made.kind, "value": made.value}}


def unlocate(body: object, control: bool) -> tuple[int, dict[str, Any]]:
    """POST /device-locations/remove {id}: retire a device's location."""
    if not control:
        return 403, {"ok": False, "error": "device locations can only be removed on this machine"}
    ident = body.get("id") if isinstance(body, dict) else None
    if not isinstance(ident, str) or not ident:
        return 400, {"ok": False, "error": "id is required"}
    try:
        devices.at().unlocate(ident)
    except LookupError as exc:
        return 404, {"ok": False, "error": str(exc)}
    except Exception:
        LOG.warning("device location not removed", exc_info=True)
        return 503, {"ok": False, "error": UNAVAILABLE}
    _forget()
    return 200, {"ok": True}
