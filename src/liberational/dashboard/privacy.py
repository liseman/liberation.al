"""Pausing cameras and microphones for privacy from the header's status icons.

A pause is `paused = true` on the device's `[[device]]` entry in config/devices.toml, so it lasts across restarts
(and state resets) until it's resumed. Capture's supervisor reads it about once a second and keeps no worker on a
paused device: the device isn't opened, so nothing from it is recorded, transcribed, matched, or captioned, and the
voice listener, which follows the transcript, hears nothing from a paused microphone either. The supervisor says what
it has applied in `<paths.state>/continuum/capture_paused.json`, so a saved pause reads "pausing" until capture has
stopped the device.

GET  /devices/privacy  {"control", "ready", "devices": [{"name", "kind", "paused", "state"}]}: kind "camera" or
     "microphone"; state "on", "pausing", "paused", or "resuming".
POST /devices/privacy  {"kind": "camera" | "microphone", "paused": bool, "device"?: name}: every enabled device of that
     kind, or just the one named. Local control and same-origin JSON only (dashboard/devices.py checks both).
"""

import json
import pathlib
import subprocess
from typing import Any

import tomlkit

from liberational import config as cfg
from liberational.dashboard import cameras

KINDS = {"camera": ("camera", "rtsp"), "microphone": ("microphone", "xvf3800")}


_ready: dict[
    tuple[str, int, int], bool
] = {}  # the capture binary as built (path, mtime, size): whether it follows pauses


def ready() -> bool:
    """Whether the built capture follows pauses: an older one rejects the manifest's paused field outright, so it is
    never written before the binary supports it. The binary is asked once per build, not once per read: the live
    feed reads this view every second (#52)."""
    binary = cfg.program("continuum-capture")
    try:
        stat = binary.stat()
    except OSError:
        return False
    key = (str(binary), stat.st_mtime_ns, stat.st_size)
    if key not in _ready:
        _ready.clear()
        _ready[key] = _follows_pauses(binary)
    return _ready[key]


def _follows_pauses(binary: pathlib.Path) -> bool:
    try:
        result = subprocess.run([str(binary), "--capabilities"], capture_output=True, timeout=3, check=False)
        if result.returncode != 0 or len(result.stdout) >= 4096:
            return False
        capabilities = json.loads(result.stdout)
        return isinstance(capabilities, dict) and capabilities.get("device_paused") is True
    except OSError, ValueError, subprocess.TimeoutExpired:
        return False


def status_path() -> pathlib.Path:
    return cfg.path("paths.state") / "continuum" / "capture_paused.json"


def applied() -> tuple[set[str], set[str]] | None:
    """(paused, still stopping) as capture last said, or None when it hasn't said (an older capture, or not running)."""
    try:
        data = json.loads(status_path().read_text())
        paused, stopping = data["paused"], data["stopping"]
        if not all(isinstance(x, list) and all(isinstance(n, str) for n in x) for x in (paused, stopping)):
            return None
        return set(paused), set(stopping)
    except OSError, ValueError, KeyError, TypeError:
        return None


def _kind(row: dict[str, Any]) -> str | None:
    return next((kind for kind, raw in KINDS.items() if row.get("kind") in raw), None)


def devices(path: pathlib.Path | None = None) -> list[dict[str, Any]]:
    """The enabled cameras and microphones in the manifest, in file order, with whether each is paused there."""
    try:
        doc = tomlkit.parse((path or cameras.manifest_path()).read_text()).unwrap()
    except OSError, ValueError:
        return []
    rows = doc.get("device", [])
    if not isinstance(rows, list):
        return []
    out = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str) or row.get("enabled", True) is not True:
            continue
        kind = _kind(row)
        if kind is not None:
            out.append({"name": row["name"], "kind": kind, "paused": row.get("paused", False) is True})
    return out


def state(paused: bool, name: str, said: tuple[set[str], set[str]] | None) -> str:
    """on, pausing, paused, or resuming: what was asked, against what capture has applied."""
    if said is None:
        return "pausing" if paused else "on"
    applied_paused, stopping = said
    if paused:
        return "paused" if name in applied_paused and name not in stopping else "pausing"
    return "resuming" if name in applied_paused else "on"


def _stamp(path: pathlib.Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


_rows: dict[str, Any] = {"stamp": None, "rows": []}


def view(control: bool) -> dict[str, Any]:
    """The devices and their pause states. The manifest and capture's status file are read again only when one of
    them changed on disk, since the live feed asks every second."""
    stamp = (str(cameras.manifest_path()), _stamp(cameras.manifest_path()), str(status_path()), _stamp(status_path()))
    if _rows["stamp"] != stamp:
        said = applied()
        rows = [dict(row, state=state(row["paused"], row["name"], said)) for row in devices()]
        _rows.update(stamp=stamp, rows=rows)
    return {"ok": True, "control": control, "ready": ready(), "devices": [dict(row) for row in _rows["rows"]]}


def save(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Pause or resume every enabled device of a kind, or one by name, in the manifest."""
    kind, paused, device = body.get("kind"), body.get("paused"), body.get("device")
    if (
        not set(body) <= {"kind", "paused", "device"}
        or kind not in KINDS
        or not isinstance(paused, bool)
        or (device is not None and not isinstance(device, str))
    ):
        return 400, {"ok": False, "error": 'send {"kind": "camera" or "microphone", "paused": true, "device"?: name}'}
    if not ready():
        return 409, {"ok": False, "error": "Rebuild and restart capture before pausing devices."}
    path = cameras.manifest_path()
    with cameras._LOCK:
        try:
            doc = tomlkit.parse(path.read_text())
            rows = doc.get("device", [])
            chosen = [
                row
                for row in rows
                if isinstance(row, dict)
                and row.get("enabled", True) is True
                and _kind(row) == kind
                and (device is None or row.get("name") == device)
            ]
            if not chosen:
                return 404, {"ok": False, "error": f"No such {kind}."}
            for row in chosen:
                if paused:
                    row["paused"] = True
                else:
                    row.pop("paused", None)
            cameras._write_manifest(path, tomlkit.dumps(doc))
        except OSError, ValueError:
            return 500, {"ok": False, "error": "Could not update the capture manifest; nothing was changed."}
    return 200, view(True)
