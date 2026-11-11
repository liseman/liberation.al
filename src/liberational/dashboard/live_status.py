"""What's live now: the newest camera captions, the microphones and cameras' status, and what the cameras saw
around a moment (moved out of the server, #61).

GET /scene       the newest captions (`scene_entries`)
GET /ambient     the microphones (`mic_status`)
GET /vision      the cameras (`camera_status`)
GET /history/at  what the cameras saw around a time, on this machine unless remote camera history is on
"""

import math
import threading
import time
from typing import TYPE_CHECKING, Any

from liberational import services, speech_log, vision_log
from liberational.audio import gain
from liberational.dashboard import context, routes
from liberational.dashboard.context import LOG

if TYPE_CHECKING:
    from liberational.dashboard.server import H


SCENE_POLL_S = 2.0  # how often the live feed asks the vision log for new captions


_scene_lock = threading.Lock()


_scene_cache: dict[str, Any] = {"at": -math.inf, "value": None, "log": None}


def scene_entries() -> list[dict[str, Any]]:
    """The newest captions for Live (`camera.caption` on the vision log, as scene.json had them): an unchanged view
    shows its last real caption, unchanged since then. Empty when the broker can't be asked.

    Every open Live feed asks every SCENE_POLL_S, and each answer decodes the last hour of captions, so one read
    serves every caller for SCENE_POLL_S: more tabs no longer mean more decoding. A failed read isn't kept."""
    with _scene_lock:
        source = vision_log.at(context.STATE)  # one per broker, shared: the answer kept is this log's
        cached = _scene_cache["value"]
        if (
            cached is not None
            and _scene_cache["log"] is source
            and time.monotonic() - _scene_cache["at"] < SCENE_POLL_S
        ):
            return cached
        try:
            value = source.scene(time.time())
        except Exception as e:
            LOG.warning("scene: the vision log couldn't be read: %s", e)
            return []
        _scene_cache.update(at=time.monotonic(), value=value, log=source)
        return value


def history_at(ts: float, window: float = 30.0) -> dict[str, Any]:
    """What each camera saw nearest wall-clock ts, within +-window seconds (at most 600), from the vision log's
    captions: {"ts", "cams": {cam: {"ts", "dt", "caption", "unchanged", "since", "faces"}}}. faces is None when no
    faces were checked. A view that hadn't changed shows its last real caption, unchanged since then."""
    window = max(1.0, min(600.0, window))
    out: dict[str, Any] = {"ts": ts, "window": window, "cams": {}}
    try:
        rows = vision_log.at(context.STATE).captions(ts - window, ts + window)
    except Exception as ex:
        out["error"] = str(ex)
        return out
    nearest: dict[str, dict[str, Any]] = {}
    for r in rows:
        best = nearest.get(r["cam"])
        if best is None or abs(r["ts"] - ts) < abs(best["ts"] - ts):
            nearest[r["cam"]] = r
    for cam, row in nearest.items():
        faces = row.get("faces")
        out["cams"][cam] = {
            "ts": row["ts"],
            "dt": round(row["ts"] - ts, 2),
            "caption": row.get("caption"),
            "unchanged": bool(row.get("unchanged")),
            "since": row.get("since") if row.get("unchanged") else None,
            "faces": [{k: f.get(k) for k in ("name", "sim", "bearing_deg")} for f in faces if isinstance(f, dict)]
            if isinstance(faces, list)
            else None,
        }
    return _with_recorded_people(out, ts, window)


def _with_recorded_people(out: dict[str, Any], ts: float, window: float) -> dict[str, Any]:
    """Prefer who the face worker found (vision/sightings.py visits) over the caption's faces: named people it found
    come first, marked "source": "recordings", and the live sample's other faces follow."""
    from liberational.vision import sightings

    by_cam = {}
    for v in sightings.who_at(ts, min(window, 60.0), context.STATE):
        by_cam.setdefault(v["cam"], []).append(
            {"name": v["name"], "sim": round(v["best_sim"], 3), "bearing_deg": None, "source": "recordings"}
        )
    for cam, seen in by_cam.items():
        entry = out["cams"].setdefault(cam, {"ts": None, "dt": None, "caption": None, "faces": None})
        names = {f["name"].lower() for f in seen}
        entry["faces"] = seen + [f for f in entry["faces"] or [] if str(f.get("name", "")).lower() not in names]
    return out


def _listening() -> bool:
    """Whether the listen worker (which handles voice tags) is running here, as a service or by hand: the dashboard's
    cached process scan."""
    try:
        return bool(services.processes().get("listen"))
    except Exception:
        return False


MIC_FRESH_S = 10.0  # a microphone whose newest audio is older than this isn't being heard


_mic_cache: dict[str, Any] = {"at": 0.0, "state": None, "value": None}


def camera_status() -> dict[str, Any]:
    """The cameras capture is sending, from their `<name>.video` streams (dashboard/cameras.py), in the shape the
    page reads: {"ts", "cameras": {name: {"state", "size", "fps"}}}."""
    from liberational.dashboard import cameras, devices

    return cameras.states(devices._link())


def mic_status() -> dict[str, Any]:
    """The microphones as the Live tab shows them: {"ts", "devices":
    {name: {"name", "healthy", "gain"?}}}, "gain" from the listen worker's auto gain status. A microphone is healthy
    while the listen worker runs and its `<name>.audio` stream on the broker has audio from the last MIC_FRESH_S; "ts"
    is now while any is, and absent otherwise (not listening). Asked of the broker at most every 2 s, and the answer
    is kept per state directory, so pointing `context.STATE` elsewhere never serves the state before it."""
    state = context.STATE
    if (
        _mic_cache["value"] is not None
        and _mic_cache["state"] == str(state)
        and time.monotonic() - _mic_cache["at"] < 2.0
    ):
        return _mic_cache["value"]
    try:
        fresh = speech_log.shared().mic_freshness()
    except Exception as e:
        LOG.warning("mic status: %s", repr(e))
        fresh = {}
    listening, now_ns = _listening(), time.time_ns()
    gains = {g.get("name"): g for g in gain.read_status(state / gain.STATUS).values()}
    devices = {}
    for stream, newest in fresh.items():
        name = stream.removesuffix(".audio")
        ok = listening and newest is not None and now_ns - newest < MIC_FRESH_S * 1e9
        devices[name] = {"name": name, "healthy": ok}
        if (g := gains.get(name)) and g.get("kind"):
            devices[name]["gain"] = {"auto": g.get("auto"), "kind": g["kind"], "db": g.get("db")}
    out: dict[str, Any] = {"devices": devices}
    if any(d["healthy"] for d in devices.values()):
        out["ts"] = time.time()
    _mic_cache.update(at=time.monotonic(), state=str(state), value=out)
    return out


@routes.get("/history/at")
def _get_history_at(h: H) -> None:
    if not h._may_see_camera_history():
        return h._json({"error": "camera history is only shown on this machine"}, 403)
    q = h._query()
    try:
        ts, window = float(q.get("ts", "")), float(q.get("window", "30"))
    except ValueError:
        return h._json({"error": "bad ts or window"}, 400)
    return h._json(history_at(ts, window))


@routes.get("/scene")
def _get_scene(h: H) -> None:
    return h._json(scene_entries())


@routes.get("/ambient")
def _get_ambient(h: H) -> None:
    return h._json(mic_status())


@routes.get("/vision")
def _get_vision(h: H) -> None:
    return h._json(camera_status())
