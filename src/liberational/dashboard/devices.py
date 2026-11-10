"""The cameras and microphones this machine has and which ones Liberational uses, for Settings' Cameras and
microphones. Choosing them edits capture's device manifest (dashboard/cameras.py, POST /devices/camera and
POST /devices/mic).

GET  /devices   {"control", "mics": [...], "cameras": [...], "notes": {"mics", "cameras"}}. Device names only.
POST /devices/sensitivity  {"mic": device name, "sensitivity": 0..100 or null} -> one mic's sensitivity in
                pack.toml (null: back to audio.sensitivity); {"default": 0..100} -> audio.sensitivity, for every
                mic without its own. The microphone service applies either live, without restarting anything. This
                machine only, same-origin JSON.
POST /devices/camera  {"index": a camera's index from GET /devices, "on": bool} -> adds or removes its [[device]] in
                config/devices.toml and restarts continuum-capture. This machine only, same-origin JSON.
                {index, flip}, {index, auto_exposure}, {index, exposure_target}, or {index, width, height, fps} saves
                one camera's video preferences and requests only its worker to restart. A failed request retains
                the saved settings and offers an explicit all-capture restart; it never falls back automatically.
POST /devices/capture/restart {} -> explicitly requests all capture to restart, with the same local-control checks.
GET /devices/camera-controls?camera=<name> -> exposed native controls and current hardware values.
POST /devices/camera-controls {"camera", "values" | "auto_exposure" | "reset"} -> saved native overrides;
                capture applies them live on its control thread, without restarting the stream.
POST /devices/mic  {"id": a microphone's id from GET /devices, "on": bool} -> the same for a microphone.
GET/POST /devices/privacy -> pause or resume cameras or microphones for privacy (dashboard/privacy.py): capture
                stops a paused device within about a second, without restarting the others.
POST /devices/auto_gain  {"mic": device name, "on": bool} -> [audio.mic_auto_gain] in pack.toml. Live
                too: the listen worker follows it within about a second, and says where each gain is in
                state/listen.gain.json.
"""

import json
import pathlib
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlsplit

from liberational import config as cfg
from liberational import logs
from liberational.audio import gain, sensitivity
from liberational.dashboard import camera_controls, camera_modes, capture_control, privacy, routes
from liberational.dashboard import cameras as camera_manifest

if TYPE_CHECKING:
    from liberational.dashboard.server import H

LOG = logs.get("dashboard")
DISCOVERY_FRESH_S = 2.0  # share short-lived subprocess results between panels and requests


@dataclass
class _MicDiscovery:
    at: float
    raw: list[tuple[Any, ...]]
    names: dict[str, str]


_mic_discovery: _MicDiscovery | None = None
_discovery_lock = threading.Lock()
_rescan: threading.Thread | None = None  # the background scan of a stale discovery, while one runs


def _scan() -> _MicDiscovery:
    from liberational.audio import ambient

    raw = ambient.scan_raw_devices()
    names = _pipewire_names() if sys.platform.startswith("linux") else {}
    return _MicDiscovery(time.monotonic(), raw, names)


def _discover_mics(*, fresh: bool = False) -> tuple[list[tuple[Any, ...]], dict[str, str]]:
    """Only hardware discovery is cached, never selection, gain, sensitivity, or broker liveness. Mutations ask for
    fresh discovery before resolving their target; concurrent readers share one scan instead of spawning children.
    A read of a discovery older than DISCOVERY_FRESH_S answers from it at once and scans again in the background: the
    scan is a child process that took about a second, and Settings > Devices waited for it every 2 s.
    """
    global _mic_discovery, _rescan
    with _discovery_lock:
        hit = _mic_discovery
        if fresh or hit is None:
            hit = _mic_discovery = _scan()
        elif time.monotonic() - hit.at >= DISCOVERY_FRESH_S and _rescan is None:
            _rescan = threading.Thread(target=_rescan_in_background, daemon=True, name="mic-discovery")
            _rescan.start()
        return hit.raw, hit.names


def _rescan_in_background() -> None:
    """A stale discovery's next scan, kept unless a fresh one (a mutation's) came in while it ran."""
    global _mic_discovery, _rescan
    try:
        found = _scan()
        with _discovery_lock:
            if _mic_discovery is None or _mic_discovery.at < found.at:
                _mic_discovery = found
    except Exception as ex:
        LOG.warning(f"microphone discovery failed: {type(ex).__name__}: {ex}")
    finally:
        with _discovery_lock:
            _rescan = None


def _pipewire_names() -> dict[str, str]:
    """PipeWire's readable source names by node name ("Elgato Wave 3 Mono"), or {} without pw-dump."""
    exe = shutil.which("pw-dump")
    if not exe:
        return {}
    try:
        objs = json.loads(subprocess.run([exe], capture_output=True, text=True, timeout=5).stdout or "[]")
    except OSError, subprocess.SubprocessError, ValueError:
        return {}
    out: dict[str, str] = {}
    for o in objs:
        p = (o.get("info") or {}).get("props") or {}
        if p.get("media.class") == "Audio/Source" and p.get("node.name"):
            out[p["node.name"]] = p.get("node.description") or p["node.name"]
    return out


def mics(*, fresh: bool = False) -> tuple[list[dict[str, Any]], str]:
    """(rows, note): every real input, each used when a microphone in capture's manifest matches it by name (its
    `stream` is that microphone's name, which the listen worker keys its sensitivity and gain by), plus manifest
    microphones that aren't plugged in. In use: its `<stream>.audio` had audio in the last MIC_FRESH_S."""
    from liberational.audio import ambient  # selection rules only; the scan runs in a child process

    try:
        raw, names = _discover_mics(fresh=fresh)
    except Exception as ex:
        return [], f"Couldn't list microphones: {ex}"
    hosts = {d[1]: d[4] for d in raw}
    present = [d[1] for d in ambient.select_input_devices(raw, include=[])]
    manifest = camera_manifest.manifest_devices("microphone")
    live = _mic_freshness()
    gains = gain.read_status(cfg.path("paths.state") / gain.STATUS)
    rows, matched = [], set()
    for n in present:
        label = names.get(n, n)
        stream = next(
            (
                c["name"]
                for c in manifest
                if camera_manifest.matches(c["match"], label, None) or camera_manifest.matches(c["match"], n, None)
            ),
            None,
        )
        matched.add(stream)
        key = stream or n
        rows.append(
            {
                "id": n,
                "name": label,
                "host_api": hosts.get(n),
                "present": True,
                "selected": stream is not None,
                "stream": stream,
                "in_use": stream is not None and stream in live,
                "level_dbfs": None,
                **_sensitivity(key),
                **_auto_gain(key, gains.get(n)),  # the worker's status is by device name
            }
        )
    rows += [
        {
            "id": c["name"],
            "name": c["name"],
            "host_api": None,
            "present": False,
            "selected": True,
            "stream": c["name"],
            "in_use": False,
            "level_dbfs": None,
            **_sensitivity(c["name"]),
            **_auto_gain(c["name"], gains.get(c["name"])),
        }
        for c in manifest
        if c["name"] not in matched
    ]
    note = "" if present else "No microphones found. Plug one in, or check audio.exclude_devices."
    return rows, note


MIC_FRESH_S = 10.0  # a microphone whose stream had audio this recently is in use


def _mic_freshness() -> set[str]:
    """The microphones (by manifest name) whose `<name>.audio` had audio in the last MIC_FRESH_S; none without a
    broker."""
    from liberational import speech_log

    try:
        newest = speech_log.shared().mic_freshness()
    except Exception:
        return set()
    now = time.time_ns()
    return {s.removesuffix(".audio") for s, ns in newest.items() if ns is not None and now - ns < MIC_FRESH_S * 1e9}


def _mic_key(mic: str) -> str | None:
    """The name a microphone's settings are kept under: its manifest name when the manifest has it (what the listen
    worker reads), else its device name; None for no microphone /devices lists."""
    row = next((r for r in mics(fresh=True)[0] if mic in (r["id"], r["stream"])), None)
    return None if row is None else row["stream"] or row["id"]


def use_mic(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """POST /devices/mic {"id", "on"}: listens with one of the microphones /devices lists, or stops, by editing
    capture's manifest, then restarts capture to open what it names now."""
    mic, on = body.get("id"), body.get("on")
    if not isinstance(mic, str) or not mic or not isinstance(on, bool):
        return 400, {"ok": False, "error": "id (a microphone) and on (true or false) are needed"}
    rows, _ = mics(fresh=True)
    row = next((r for r in rows if r["id"] == mic), None)
    if row is None or on and not row["present"]:
        return 404, {"ok": False, "error": "no such microphone"}
    if not on and row["selected"] and row["present"] and sum(r["selected"] and r["present"] for r in rows) <= 1:
        return 409, {"ok": False, "error": "one microphone at least stays in use"}
    camera_manifest.use(row["name"], None, on, stream=None if on else row["stream"], kind="microphone")
    from liberational.dashboard import settings

    settings.restart(names=["continuum-capture"], dashboard=False)
    return 200, {**view(True), "restarted": ["continuum-capture"]}


def _auto_gain(device: str, live: dict[str, Any] | None) -> dict[str, Any]:
    """Whether auto gain is on for the mic, and its gain as the listen worker last said ({"auto", "kind": "hardware"
    or "software", "db"}, or None while it isn't running the mic). On: its own [audio.mic_auto_gain] entry (by
    device name, else by the devices.toml name the worker gave), else whether the worker found a hardware control
    (without the worker, nothing moves the gain, so it reads off)."""
    live = live if isinstance(live, dict) and live.get("kind") else None
    keys = [device] + ([str(live["name"])] if live and live.get("name") else [])
    try:
        on = gain.enabled(keys, bool(live) and live.get("kind") == "hardware")
    except OSError, ValueError:
        on = False
    return {"auto_gain": on, "gain": live}


def save_auto_gain(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """(code, reply) for POST /devices/auto_gain {"mic": device name, "on": bool}: live, nothing restarted."""
    mic, on = body.get("mic"), body.get("on")
    if not isinstance(mic, str) or not mic:
        return 400, {"ok": False, "error": "mic must be a device name"}
    if not isinstance(on, bool):
        return 400, {"ok": False, "error": "on must be true or false"}
    if (key := _mic_key(mic)) is None:
        return 400, {"ok": False, "error": f"no microphone named {mic!r}"}
    try:
        gain.save(key, on)
    except OSError as ex:
        return 500, {"ok": False, "error": f"couldn't write {cfg.local_config()}: {ex.strerror}"}
    LOG.info(f"devices: {mic} auto gain {'on' if on else 'off'}")
    return 200, view(True)


def _sensitivity(device: str) -> dict[str, Any]:
    """The mic's sensitivity now, and whether it has its own (custom) or follows audio.sensitivity."""
    try:
        return {"sensitivity": sensitivity.value(device), "sensitivity_custom": device in _custom()}
    except ValueError:
        return {"sensitivity": None, "sensitivity_custom": True}


def _custom() -> set[str]:
    table = sensitivity._local_audio().get(sensitivity.TABLE)
    return set(table) if isinstance(table, dict) else set()


def save_sensitivity(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """(code, reply) for POST /devices/sensitivity."""
    if "default" in body:
        return _save_default(body["default"])
    mic, value = body.get("mic"), body.get("sensitivity")
    if not isinstance(mic, str) or not mic:
        return 400, {"ok": False, "error": "mic must be a device name"}
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100):
        return 400, {"ok": False, "error": "sensitivity must be a whole number from 0 to 100, or null"}
    if (key := _mic_key(mic)) is None:
        return 400, {"ok": False, "error": f"no microphone named {mic!r}"}
    try:
        sensitivity.save(key, value)
    except OSError as ex:
        return 500, {"ok": False, "error": f"couldn't write {cfg.local_config()}: {ex.strerror}"}
    LOG.info(f"devices: {mic} sensitivity {'default' if value is None else value}")
    return 200, view(True)


def _save_default(value: object) -> tuple[int, dict[str, Any]]:
    """(code, reply) for {"default": N}: audio.sensitivity in pack.toml, nothing restarted."""
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100:
        return 400, {"ok": False, "error": "default must be a whole number from 0 to 100"}
    try:
        _, overridden = cfg.save({"audio.sensitivity": value})
    except cfg.SettingError as ex:
        return 400, {"ok": False, "error": str(ex)}
    except OSError as ex:
        return 500, {"ok": False, "error": f"couldn't write {cfg.local_config()}: {ex.strerror}"}
    LOG.info(f"devices: default sensitivity {value}")
    note = "PACK_AUDIO_SENSITIVITY is set in the environment and overrides this." if overridden else ""
    return 200, {**view(True), "default_sensitivity": sensitivity.default(), "note": note}


def _linux_cameras() -> list[tuple[int, str, str | None]]:
    """[(index, name, path)] for V4L2 capture nodes. Each UVC camera also has a metadata node (sysfs index 1)."""
    out: list[tuple[int, str, str | None]] = []
    for d in sorted(pathlib.Path("/sys/class/video4linux").glob("video*"), key=lambda p: int(p.name[5:] or 0)):
        try:
            if (d / "index").read_text().strip() != "0":
                continue
            out.append((int(d.name[5:]), (d / "name").read_text().strip(), f"/dev/{d.name}"))
        except OSError, ValueError:
            continue
    return out


def _mac_cameras() -> list[tuple[int, str, str | None]]:
    """[(index, name, path)] in AVFoundation order (what OpenCV's indexes follow), from ffmpeg's device list."""
    exe = shutil.which("ffmpeg")
    if not exe:
        return []
    r = subprocess.run(
        [exe, "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True,
        text=True,
        timeout=10,
    )
    out: list[tuple[int, str, str | None]] = []
    video = False
    for line in r.stderr.splitlines():
        if "AVFoundation video devices" in line:
            video = True
        elif "AVFoundation audio devices" in line:
            video = False
        elif video and (m := re.search(r"\] \[(\d+)\] (.+)$", line)) and not m.group(2).startswith("Capture screen"):
            out.append((int(m.group(1)), m.group(2).strip(), None))
    # Each camera's AVFoundation unique ID, which capture's manifest matches by (match = { path = "0x..." }): ffmpeg
    # lists names only, and three identical Arducams can't be told apart by name, so every manifest camera read as
    # unplugged and Live showed none. system_profiler lists the same devices, in the same order.
    ids = _mac_unique_ids()
    if len(ids) == len(out) and all(n == name for (_, n, _), (name, _) in zip(out, ids, strict=True)):
        out = [(i, n, uid) for (i, n, _), (_, uid) in zip(out, ids, strict=True)]
    return out


def _mac_unique_ids() -> list[tuple[str, str]]:
    """[(name, unique id)] of this Mac's cameras, from system_profiler; empty when it can't say."""
    try:
        r = subprocess.run(["system_profiler", "SPCameraDataType", "-json"], capture_output=True, text=True, timeout=15)
        rows = json.loads(r.stdout or "{}").get("SPCameraDataType", [])
    except OSError, ValueError, subprocess.SubprocessError:
        return []
    return [(str(c.get("_name", "")), str(c["spcamera_unique-id"])) for c in rows if c.get("spcamera_unique-id")]


def _matched_cameras(found: list[tuple[int, str, str | None]], manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join physical cameras to the first matching manifest row, without broker or mode discovery."""
    rows = []
    usbs = [camera_manifest.usb_id(path) for _, _, path in found]
    for (index, name, path), usb in zip(found, usbs, strict=True):
        links = camera_manifest.stable_paths(path) or ([path] if path and sys.platform == "darwin" else [])
        stream = next((c["name"] for c in manifest if camera_manifest.matches(c["match"], name, usb, links)), None)
        rows.append(
            {
                "index": index,
                "name": name,
                "path": path,
                "usb": usb,
                "paths": links,
                "shared_usb": usb is not None and usbs.count(usb) > 1,
                "present": True,
                "selected": stream is not None,
                "stream": stream,
            }
        )
    return rows


def camera_control_node(stream: str) -> str | None:
    """Resolve one selected Linux camera without querying frames, the broker, or capture modes."""
    if not sys.platform.startswith("linux"):
        return None
    return next(
        (
            row["path"]
            for row in _matched_cameras(_linux_cameras(), camera_manifest.manifest_cameras())
            if row["stream"] == stream
        ),
        None,
    )


def cameras() -> tuple[list[dict[str, Any]], str]:
    """(rows, note): the cameras this machine has, each used when a camera in capture's manifest matches it (its
    `stream` is that camera's name), plus manifest cameras that aren't plugged in (negative indexes)."""
    try:
        found = (
            _linux_cameras() if sys.platform.startswith("linux") else _mac_cameras() if sys.platform == "darwin" else []
        )
    except Exception as ex:
        return [], f"Couldn't list cameras: {ex}"
    manifest = camera_manifest.manifest_cameras()
    live = set(camera_manifest.states(_link()).get("cameras", {}))
    rows = _matched_cameras(found, manifest)
    matched = {row["stream"] for row in rows}
    for row in rows:
        row["in_use"] = row["stream"] in live
        row["flip"] = bool(camera_manifest.video_settings(row["stream"]).get("flip", False)) if row["stream"] else False
    absent = [c for c in manifest if c["name"] not in matched]
    rows += [
        {
            "index": -1 - n,
            "name": c["name"],
            "path": None,
            "usb": None,
            "paths": [],
            "shared_usb": False,
            "present": False,
            "selected": True,
            "stream": c["name"],
            "in_use": False,
            "flip": bool(camera_manifest.video_settings(c["name"]).get("flip", False)),
        }
        for n, c in enumerate(absent)
    ]
    for row in rows:
        row.update(_camera_controls(row))
    note = "" if found else "No cameras found. Plug one in (the Elgato Facecam needs a USB 3 port)."
    return rows, note


def _camera_controls(row: dict[str, Any]) -> dict[str, Any]:
    """Manifest preferences, with the same defaults as capture's VideoSettings, and native mode availability."""
    video = {"width": 1280, "height": 720, "fps": 30, "format": "MJPEG", "auto_exposure": True, "exposure_target": 110}
    if row["selected"]:
        settings = camera_manifest.video_settings(row["stream"])
        video.update({k: v for k, v in settings.items() if k in video})
    linux = sys.platform.startswith("linux")
    exposure_note = (
        ""
        if linux
        else "Automatic exposure steering is available only on Linux; this camera uses its own automatic exposure."
    )
    if linux and row["present"] and row["selected"] and row["path"]:
        try:
            modes = camera_modes.modes(row["path"], str(video["format"]))
        except ValueError as error:
            modes = {
                "status": "unavailable",
                "format": video["format"],
                "fourcc": "",
                "sizes": [],
                "choices": [],
                "diagnostics": [f"Invalid configured camera format {video['format']!r}: {error}"],
            }
    else:
        note = (
            "Camera mode discovery is available only on Linux"
            if not linux
            else "Select this camera to choose its mode"
            if not row["selected"]
            else "Camera is not attached"
        )
        modes = {
            "status": "unavailable",
            "format": video["format"],
            "fourcc": "",
            "sizes": [],
            "choices": [],
            "diagnostics": [note],
        }
    return {"video": video, "modes": modes, "exposure_supported": linux, "exposure_note": exposure_note}


_links: list[Any] = []


def _link() -> Any:
    """The dashboard's broker link for camera status, made on first use."""
    if not _links:
        from liberational import continuum_link

        _links.append(continuum_link.BrokerLink(client_name="dashboard-cameras", timeout=5.0))
    return _links[0]


def use_camera(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """POST /devices/camera {"index", "on"}: uses one of the cameras /devices lists, or stops using it, by editing
    capture's manifest, then restarts capture to open what it names now."""
    if any(k in body for k in ("flip", "auto_exposure", "exposure_target", "width", "height", "fps")):
        return save_camera_video(body)
    index, on = body.get("index"), body.get("on")
    if not isinstance(index, int) or isinstance(index, bool) or not isinstance(on, bool):
        return 400, {"ok": False, "error": "index (a number) and on (true or false) are needed"}
    rows, _ = cameras()
    row = next((r for r in rows if r["index"] == index), None)
    if row is None or on and not row["present"]:
        return 404, {"ok": False, "error": "no such camera"}
    if not on and row["selected"] and sum(r["selected"] and r["present"] for r in rows) <= 1 and row["present"]:
        return 409, {"ok": False, "error": "one camera at least stays in use"}
    camera_manifest.use(
        row["name"],
        row["usb"],
        on,
        stream=None if on else row["stream"],
        paths=row["paths"],
        shared=row["shared_usb"],
    )
    from liberational.dashboard import settings

    settings.restart(names=["continuum-capture"], dashboard=False)
    return 200, {**view(True), "restarted": ["continuum-capture"]}


def save_camera_video(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Save one video preference group, then request only that camera's worker to restart."""
    index = body.get("index")
    fields = set(body) - {"index"}
    if type(index) is not int or fields not in (
        {"flip"},
        {"auto_exposure"},
        {"exposure_target"},
        {"width", "height", "fps"},
    ):
        return 400, {"ok": False, "error": "send index and one camera setting group"}
    if fields in ({"flip"}, {"auto_exposure"}) and not isinstance(body[next(iter(fields))], bool):
        return 400, {"ok": False, "error": "camera switch must be true or false"}
    if fields == {"exposure_target"} and (
        type(body["exposure_target"]) is not int or not 60 <= body["exposure_target"] <= 180
    ):
        return 400, {"ok": False, "error": "exposure_target must be a whole number from 60 to 180"}
    rows, _ = cameras()
    row = next((r for r in rows if r["index"] == index and r["selected"] and r["present"]), None)
    if row is None:
        return 404, {"ok": False, "error": "no such attached selected camera"}
    if fields != {"flip"} and not sys.platform.startswith("linux"):
        return 400, {"ok": False, "error": "exposure steering and mode selection are available only on Linux"}
    try:
        values = (
            {k: body[k] for k in fields}
            if fields != {"width", "height", "fps"}
            else camera_modes.validate_choice(
                row["path"], body["width"], body["height"], body["fps"], row["video"]["format"]
            )
        )
        camera_manifest.update_video(row["stream"], values)
    except (OSError, ValueError) as error:
        return 400, {"ok": False, "error": str(error)}
    result = capture_control.restart_camera(row["stream"])
    LOG.info(f"camera settings: {row['stream']}: {result.status}: {result.message}")
    return 200, {
        **view(True),
        "restart": {"status": result.status, "device": row["stream"], "message": result.message},
        "restart_all_available": result.status != "restarting",
    }


def restart_capture(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Explicitly restart all capture, after the person chooses the fallback action."""
    if body:
        return 400, {"ok": False, "error": "send an empty object to restart all capture"}
    from liberational.dashboard import settings

    try:
        settings.restart(names=["continuum-capture"], dashboard=False)
        status, message = "restarting", "All capture restart requested"
    except (OSError, ValueError) as error:
        status, message = "failed", str(error)
    return 200, {
        **view(True),
        "restart": {"status": status, "device": "all", "message": message},
        "restart_all_available": status != "restarting",
    }


def view(control: bool) -> dict[str, Any]:
    m, mic_note = mics()
    c, cam_note = cameras()
    try:
        default: int | None = sensitivity.default()
    except ValueError:
        default = None
    return {
        "ok": True,
        "control": control,
        "mics": m,
        "cameras": c,
        "notes": {"mics": mic_note, "cameras": cam_note},
        "default_sensitivity": default,
    }


def handle(handler: H) -> bool:
    """Serve /devices and its POST routes (sensitivity, auto_gain, camera, mic); False for any other path."""
    path = handler.path.split("?")[0]
    if path not in (
        "/devices",
        "/devices/sensitivity",
        "/devices/auto_gain",
        "/devices/camera",
        "/devices/mic",
        "/devices/capture/restart",
        "/devices/camera-controls",
        "/devices/privacy",
    ):
        return False
    if handler.command in ("GET", "HEAD"):
        if path == "/devices/privacy":
            handler._json(privacy.view(handler._can_control()))
            return True
        if path == "/devices/camera-controls":
            camera = parse_qs(urlsplit(handler.path).query).get("camera", [""])[0]
            code, reply = camera_controls.view(camera, handler._can_control())
            handler._json(reply, code)
            return True
        if path != "/devices":
            return False
        handler._json(view(handler._can_control()))
        return True
    if handler.command != "POST" or path == "/devices":
        return False
    origin = handler.headers.get("Origin")
    if not handler._can_control():
        handler._json({"ok": False, "error": "devices can only be changed from this machine"}, 403)
        return True
    if handler.headers.get_content_type() != "application/json" or (
        origin and origin != "http://" + (handler.headers.get("Host") or "")
    ):
        handler._json({"ok": False, "error": "same-origin JSON only"}, 403)
        return True
    try:
        n = int(handler.headers.get("Content-Length") or 0)
        if not 0 < n <= 65536:
            raise ValueError
        body = json.loads(handler.rfile.read(n))
        if not isinstance(body, dict):
            raise ValueError
    except ValueError:
        shape = {
            "/devices/auto_gain": '{"mic": "...", "on": true}',
            "/devices/camera": '{"index": 0, "on": true}',
            "/devices/mic": '{"id": "...", "on": true}',
            "/devices/capture/restart": "{}",
            "/devices/privacy": '{"kind": "camera", "paused": true}',
        }.get(path, '{"mic": "...", "sensitivity": 70}')
        handler._json({"ok": False, "error": f"send {shape}"}, 400)
        return True
    if path == "/devices/privacy":
        code, reply = privacy.save(body)
    elif path == "/devices/camera-controls":
        code, reply = camera_controls.save(body)
    elif path == "/devices/capture/restart":
        code, reply = restart_capture(body)
    elif path == "/devices/camera":
        code, reply = use_camera(body)
    elif path == "/devices/mic":
        code, reply = use_mic(body)
    elif path == "/devices/auto_gain":
        code, reply = save_auto_gain(body)
    else:
        code, reply = save_sensitivity(body)
    handler._json(reply, code)
    return True


# ---- routes (dashboard/routes.py) ----


@routes.get("/devices", "/devices/privacy", "/devices/camera-controls")
@routes.post_raw(
    "/devices/sensitivity",
    "/devices/auto_gain",
    "/devices/camera",
    "/devices/mic",
    "/devices/capture/restart",
    "/devices/camera-controls",
    "/devices/privacy",
)
def _route(h: H) -> None:
    handle(h)
