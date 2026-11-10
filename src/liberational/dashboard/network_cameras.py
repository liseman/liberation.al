"""Explicit local camera discovery, with short-lived private stream proposals.

Protect is read from its own console (`[unifi_protect]`) when set, else `[unifi]`. ONVIF adds unicast probes to
`network_cameras.onvif_hosts` and, when `[unifi]` is configured, to the Network console's current client IPs.

RTSP URLs contain bearer credentials. Public replies expose only host and scheme; Add resolves opaque IDs and
writes an owner-only capture manifest. Discovery never enables streams or opens media.

Configured IDs remain stream identities; vision.camera_names supplies display aliases. The local use endpoint retains
disabled devices and their private credentials in place, guarded by the native enabled-field capability, and
explicitly restarts capture. Public replies contain only the redacted URL and the active camera roster.
"""

import json
import os
import re
import subprocess
import threading
import time
import tomllib
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlsplit, urlunsplit

import tomlkit

from liberational import config as cfg
from liberational.dashboard import cameras, settings
from liberational.proximity.network import ip_address
from liberational.proximity.unifi_network import UniFiNetwork
from liberational.unifi import UniFiError, configured_session, local_origin, protect_section, setting
from liberational.vision import views
from liberational.vision.onvif import MAX_UNICAST_HOSTS, OnvifDiscoverySource, probe_host
from liberational.vision.unifi_protect import UniFiProtect

if TYPE_CHECKING:
    from liberational.dashboard.server import H

_DISCOVERY = threading.Lock()
_CACHE_LOCK = threading.Lock()
# Only the latest bounded result is retained; restarting the dashboard expires proposals.
_proposals: dict[tuple[str, str, str], tuple[float, str]] = {}


def display_url(url: str) -> str:
    try:
        parts = urlsplit(url)
        return f"{parts.scheme}://{parts.hostname or 'camera'}/…"
    except ValueError:
        return "RTSP camera"


def capture_ready() -> bool:
    try:
        import continuum_py

        if not getattr(continuum_py, "h264_decoder_available", lambda: False)():
            return False
        for name in ("continuum-capture", "continuum-broker", "continuum-tier-worker"):
            binary = cfg.ROOT / "continuum" / "target" / "release" / name
            result = subprocess.run([str(binary), "--capabilities"], capture_output=True, timeout=3, check=False)
            if result.returncode != 0 or len(result.stdout) >= 4096:
                return False
            capabilities = json.loads(result.stdout)
            if not isinstance(capabilities, dict) or capabilities.get("rtsp_h264") is not True:
                return False
        return True
    except OSError, ValueError, subprocess.TimeoutExpired:
        return False


def controls_ready() -> bool:
    """An old capture rejects enabled fields; never write one before its binary supports them."""
    binary = cfg.ROOT / "continuum" / "target" / "release" / "continuum-capture"
    try:
        result = subprocess.run([str(binary), "--capabilities"], capture_output=True, timeout=3, check=False)
        if result.returncode != 0 or len(result.stdout) >= 4096:
            return False
        capabilities = json.loads(result.stdout)
        return (
            isinstance(capabilities, dict)
            and capabilities.get("device_enabled") is True
            and capabilities.get("rtsp_h264") is True
        )
    except OSError, ValueError, subprocess.TimeoutExpired:
        return False


def _configured(doc: Any) -> list[dict[str, Any]]:
    devices = doc.get("device", [])
    if not isinstance(devices, list) or any(not isinstance(row, dict) for row in devices):
        raise ValueError("invalid capture manifest device list")
    rows = [row for row in devices if row.get("kind") == "rtsp"]
    if any(
        not isinstance(row.get("name"), str)
        or not views.valid_camera_id(row["name"])
        or not isinstance(row.get("enabled", True), bool)
        or not isinstance(row.get("rtsp", {}), dict)
        or not isinstance(row.get("rtsp", {}).get("url", ""), str)
        for row in rows
    ):
        raise ValueError("invalid network camera configuration")
    return rows


def view() -> dict[str, Any]:
    path = cameras.manifest_path()
    # Malformed state is actionable, not an empty installation. Never expose a TOML source excerpt with a URL.
    try:
        doc = tomllib.loads(path.read_text())
        configured = _configured(doc)
    except FileNotFoundError:
        configured = []
    except OSError, ValueError:
        return {
            "cameras": [],
            "capture_ready": False,
            "error": "Capture manifest cannot be read; repair it before adding cameras.",
        }
    aliases = views.names()
    rows = [
        dict(
            id=row["name"],
            name=aliases.get(row["name"], row["name"]),
            source="rtsp",
            enabled=row.get("enabled", True),
            url_display=display_url(row.get("rtsp", {}).get("url", "")),
        )
        for row in configured
    ]
    return {"cameras": rows, "capture_ready": capture_ready(), "controls_ready": controls_ready()}


def use(body: dict[str, Any], log: Callable[..., None]) -> tuple[int, dict[str, Any]]:
    """Toggle an exact manifest ID without changing its private URL, display alias, or other settings."""
    name, enabled = body.get("id"), body.get("enabled")
    if set(body) != {"id", "enabled"} or not isinstance(name, str) or not isinstance(enabled, bool):
        return 400, {"error": "Send a camera id and enabled (true or false)."}
    if not controls_ready():
        return 409, {"error": "Build capture with device enable support before changing camera use."}
    path = cameras.manifest_path()
    with cameras._LOCK:
        try:
            doc = tomlkit.parse(path.read_text())
            rows = _configured(doc.unwrap())
            if not any(row["name"] == name for row in rows):
                return 404, {"error": "No such configured network camera."}
            row = next(row for row in doc["device"] if row.get("kind") == "rtsp" and row.get("name") == name)
            ordered = views.manifest_camera_names(path) or []
            active = set(views.manifest_camera_names(path, active_only=True) or [])
            if row.get("enabled", True) == enabled:
                return 200, {
                    "ok": True,
                    "restarted": [],
                    "active_cameras": [camera for camera in ordered if camera in active],
                }
            row["enabled"] = enabled
            if enabled:
                active.add(name)
            else:
                active.discard(name)
            os.chmod(path, 0o600)
            cameras._write_manifest(path, tomlkit.dumps(doc))
        except OSError, ValueError:
            return 500, {"error": "Could not update capture manifest; camera use was not changed."}
    try:
        settings.restart(log, names=["continuum-capture"], dashboard=False)
    except Exception:
        return 500, {"error": "Camera use was saved, but capture could not be restarted. Restart capture to apply it."}
    return 200, {
        "ok": True,
        "restarted": ["continuum-capture"],
        "active_cameras": [camera for camera in ordered if camera in active],
    }


def onvif_hosts(warnings: list[str]) -> list[str]:
    """Unicast probe targets: configured hosts first, then the Network console's current client IPs, capped."""
    configured = cfg.get("network_cameras.onvif_hosts")
    if not isinstance(configured, list):
        configured = []
        warnings.append("network_cameras.onvif_hosts must be a list of IP addresses.")
    hosts = [host for host in map(probe_host, configured) if host is not None]
    if len(hosts) < len(configured):
        warnings.append("Some network_cameras.onvif_hosts entries are not private IPv4 addresses; they were skipped.")
    if cfg.get("unifi.url"):
        session = None
        try:
            session = configured_session()
            inventory = UniFiNetwork(
                session,
                site=str(cfg.get("network_devices.site")),
                mode=str(cfg.get("unifi.mode")),
                timeout_s=float(cfg.get("unifi.timeout_seconds")),
            ).poll(time.time())
            if not inventory.devices:
                warnings.append("The UniFi client list was unavailable; ONVIF probed only listed hosts.")
            hosts.extend(
                host
                for host in (probe_host(device.ip) for device in inventory.devices if device.connected is not False)
                if host is not None
            )
        except Exception:
            warnings.append("The UniFi client list was unavailable; ONVIF probed only listed hosts.")
        finally:
            if session:
                session.close()
    unique = list(dict.fromkeys(hosts))
    if len(unique) > MAX_UNICAST_HOSTS:
        warnings.append(f"ONVIF unicast probes are capped at {MAX_UNICAST_HOSTS} hosts; the rest were skipped.")
    return unique[:MAX_UNICAST_HOSTS]


def discover(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    source = body.get("source", "all")
    if source not in ("all", "protect", "onvif"):
        return 400, {"error": "Choose all, protect, or onvif."}
    if not _DISCOVERY.acquire(blocking=False):
        return 409, {"error": "Camera discovery is already running."}
    try:
        found: list[tuple[str, Any]] = []
        warnings: list[str] = []
        if source in ("all", "protect"):
            section = protect_section()
            if section:
                session = None
                try:
                    session = configured_session(section)
                    result = UniFiProtect(session, mode=str(cfg.get(setting(section, "mode")))).discover()
                    found.extend(("protect", camera) for camera in result.cameras)
                    warnings.extend(result.warnings)
                except Exception:
                    warnings.append("Local Protect discovery failed; check controller configuration.")
                finally:
                    if session:
                        session.close()
            else:
                warnings.append("Local Protect controller is not configured.")
        if source in ("all", "onvif"):
            try:
                result = OnvifDiscoverySource(
                    username=str(cfg.get("network_cameras.onvif_username")),
                    password=str(cfg.get("network_cameras.onvif_password")),
                    hosts=onvif_hosts(warnings),
                ).discover()
                found.extend(("onvif", camera) for camera in result.cameras)
                warnings.extend(result.warnings)
            except Exception:
                warnings.append("Local ONVIF discovery failed.")
        discovery_id = str(uuid.uuid4())
        rows = []
        proposals = {}
        expires = time.monotonic() + 300
        for index, (kind, camera) in enumerate(found[:128]):
            camera_id = str(index)
            streams = []
            for number, stream in enumerate(camera.streams[:4]):
                if len(proposals) >= 256:
                    break
                stream_id = str(number)
                url = stream.url
                if kind == "onvif":
                    parts = urlsplit(url)
                    camera_ip = ip_address(camera.ip)
                    # Discovery pins media to the camera; check again before attaching configured secrets.
                    if parts.username is None and camera_ip is not None and ip_address(parts.hostname) == camera_ip:
                        username = str(cfg.get("network_cameras.onvif_username"))
                        password = str(cfg.get("network_cameras.onvif_password"))
                        if username:
                            authority = f"{quote(username, safe='')}:{quote(password, safe='')}@{parts.netloc}"
                            url = urlunsplit(parts._replace(netloc=authority))
                proposals[(discovery_id, camera_id, stream_id)] = (expires, url)
                streams.append(
                    dict(id=stream_id, quality=stream.quality, url_display=display_url(stream.url), ready=True)
                )
            rows.append(
                dict(
                    id=camera_id,
                    source=kind,
                    name=camera.name,
                    model=camera.model,
                    ip=camera.ip,
                    streams=streams,
                    requires_credentials=getattr(camera, "requires_credentials", False),
                )
            )
        with _CACHE_LOCK:
            _proposals.clear()
            _proposals.update(proposals)
        return 200, dict(discovery_id=discovery_id, cameras=rows, warnings=warnings[:32], capture_ready=capture_ready())
    finally:
        _DISCOVERY.release()


def add(body: dict[str, Any], log: Callable[..., None]) -> tuple[int, dict[str, Any]]:
    if not capture_ready():
        return 409, {"error": "Install RTSP-enabled capture and the H.264 vision decoder before adding cameras."}
    key = (body.get("discovery_id"), body.get("camera_id"), body.get("stream_id"))
    if any(not isinstance(part, str) or len(part) > 128 for part in key):
        return 400, {"error": "Choose a discovered camera stream."}
    with _CACHE_LOCK:
        proposal = _proposals.get(key)
    if proposal is None or proposal[0] < time.monotonic():
        return 409, {"error": "Discovery expired; discover cameras again."}
    url = proposal[1]
    try:
        parts = urlsplit(url)
        if parts.scheme not in ("rtsp", "rtsps") or parts.fragment:
            raise ValueError
        # Validate the host separately: ONVIF cameras may require userinfo in the private stream URL.
        local_origin("https://" + (f"[{parts.hostname}]" if ":" in (parts.hostname or "") else parts.hostname or ""))
    except ValueError, UniFiError:
        return 400, {"error": "Camera stream must use a local RTSP address."}
    name = body.get("name") or "network-camera"
    if not isinstance(name, str) or re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]{0,63}", name) is None:
        return 400, {
            "error": "Use a camera name starting with a letter, with letters, digits, underscores, or hyphens."
        }
    path = cameras.manifest_path()
    with cameras._LOCK:
        try:
            doc = tomlkit.parse(path.read_text()) if path.exists() else tomlkit.document()
            devices = doc.get("device", [])
            if any(row.get("name") == name for row in devices):
                return 409, {"error": "That camera name is already in use."}
            row = tomlkit.table()
            row.update(dict(name=name, kind="rtsp"))
            row["rtsp"] = tomlkit.inline_table()
            row["rtsp"].update(
                dict(
                    url=url,
                    transport="tcp",
                    connect_timeout_ms=10000,
                    read_timeout_ms=10000,
                    max_access_unit_bytes=4194304,
                    max_width=4096,
                    max_height=4096,
                    ring_bytes=67108864,
                )
            )
            row["video"] = tomlkit.inline_table()
            row["video"].update(dict(ring_seconds=10.0, flip=False))
            if "device" not in doc:
                doc["device"] = tomlkit.aot()
            doc["device"].append(row)
            path.parent.mkdir(parents=True, exist_ok=True)
            # The shared atomic writer preserves mode, so tighten before writing bearer credentials.
            if not path.exists():
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.close(fd)
            os.chmod(path, 0o600)
            cameras._write_manifest(path, tomlkit.dumps(doc))
        except OSError, ValueError:
            return 500, {"error": "Could not update capture manifest; no camera was started."}
    settings.restart(log, names=["continuum-capture"], dashboard=False)
    with _CACHE_LOCK:
        _proposals.pop(key, None)
    return 200, {"ok": True}


def handle(handler: H, log: Callable[..., None]) -> bool:
    path = handler.path.split("?")[0]
    if path not in ("/network-cameras", "/network-cameras/discover", "/network-cameras/add", "/network-cameras/use"):
        return False
    if handler.command in ("GET", "HEAD") and path == "/network-cameras":
        result = view()
        handler._json(result, 503 if "error" in result else 200)
        return True
    if handler.command != "POST" or path == "/network-cameras":
        return False
    origin = handler.headers.get("Origin")
    if (
        not handler._can_control()
        or handler.headers.get_content_type() != "application/json"
        or (origin and origin != "http://" + (handler.headers.get("Host") or ""))
    ):
        handler._json({"error": "Local same-origin JSON only."}, 403)
        return True
    try:
        length = int(handler.headers.get("Content-Length") or 0)
        if not 0 < length <= 4096:
            raise ValueError
        body = json.loads(handler.rfile.read(length))
        if not isinstance(body, dict):
            raise ValueError
    except ValueError:
        handler._json({"error": "Send a JSON object."}, 400)
        return True
    code, result = (
        discover(body) if path.endswith("/discover") else use(body, log) if path.endswith("/use") else add(body, log)
    )
    handler._json(result, code)
    return True
