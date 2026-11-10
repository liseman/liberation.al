"""The cameras and microphones Continuum's capture uses, for Settings' Cameras and microphones.

Capture opens the devices its manifest (`config/devices.toml`) names: a `[[device]]` with `kind = "camera"` or
`"microphone"` and a `match` (`usb = "vendor:product"`, or `name`, a piece of the device's name; every field given must match). Settings
lists the cameras the machine has (dashboard/devices.py) and marks one used when a manifest camera matches it; ticking
one adds a `[[device]]` for it, unticking removes its block, and capture is restarted to pick that up (it reads the
manifest when it starts). The rest of the file, comments included, is left as it is.

Each used camera's status comes from its `<name>.video` stream on the broker: on, with the size and frame rate of
its last STATUS_WINDOW_S of frames, or absent when capture sent nothing in that time.
"""

import os
import pathlib
import re
import tempfile
import threading
import time
import tomllib
from collections.abc import Sequence
from typing import Any

import tomlkit

from liberational import config as cfg
from liberational import logs

LOG = logs.get("dashboard")
STATUS_WINDOW_S = 3.0  # a camera with a frame this recent is on; its frame rate is counted over this long
STATUS_EVERY_S = 2.0  # the broker is asked at most this often
_LOCK = threading.Lock()  # one manifest edit at a time
_cache: dict[str, Any] = {"at": -1e9, "value": None}
_status_lock = threading.Lock()
_status_read: threading.Thread | None = None  # the background read of a stale status, while one runs
_HEADER = re.compile(r"^\s*\[")
_SUBTABLE = re.compile(r"^\s*\[\s*device\s*\.")


def manifest_path() -> pathlib.Path:
    return cfg.ROOT / "config" / "devices.toml"


def manifest_cameras(path: pathlib.Path | None = None) -> list[dict[str, Any]]:
    """The manifest's cameras: [{name, match}], in file order."""
    return manifest_devices("camera", path)


def manifest_devices(kind: str, path: pathlib.Path | None = None, *, active_only: bool = True) -> list[dict[str, Any]]:
    """Devices of `kind`: [{name, match}], in file order, enabled only by default.

    Return none when there's no manifest or it doesn't parse. Edits include disabled rows so their identities
    and saved settings remain available when a device is selected again.
    """
    try:
        doc = tomllib.loads((path or manifest_path()).read_text())
    except OSError, tomllib.TOMLDecodeError:
        return []
    return [
        {"name": str(d["name"]), "match": dict(d.get("match") or {})}
        for d in doc.get("device", [])
        if isinstance(d, dict)
        and d.get("kind") == kind
        and d.get("name")
        and (not active_only or d.get("enabled", True) is True)
    ]


def video_settings(stream: str, path: pathlib.Path | None = None) -> dict[str, Any]:
    """Effective video settings, with a camera's fields overriding the manifest defaults."""
    doc = tomllib.loads((path or manifest_path()).read_text())
    camera = next(d for d in doc.get("device", []) if d.get("kind") == "camera" and d.get("name") == stream)
    return {**doc.get("video", {}), **camera.get("video", {})}


def _write_manifest(path: pathlib.Path, text: str) -> None:
    """Replace a complete manifest atomically; capture must never observe half a document."""
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as out:
            out.write(text)
        os.chmod(temporary, path.stat().st_mode & 0o777)
        os.replace(temporary, path)
    finally:
        pathlib.Path(temporary).unlink(missing_ok=True)


def update_video(stream: str, values: dict[str, bool | int], path: pathlib.Path | None = None) -> None:
    """Update one camera under the manifest lock, retaining every other camera and setting.

    Callers validate their public request shape. This boundary also validates supported native video fields so
    future exposure and resolution controls can share the same read-modify-write operation.
    """
    for key, value in values.items():
        if key in ("flip", "auto_exposure", "hold_fps"):
            valid = isinstance(value, bool)
        elif key in ("width", "height", "fps", "exposure_target"):
            valid = type(value) is int and value > 0 and (key != "exposure_target" or value <= 255)
        else:
            valid = False
        if not valid:
            raise ValueError(f"invalid video setting {key}")
    path = path or manifest_path()
    with _LOCK:
        doc = tomlkit.parse(path.read_text())
        for camera in doc.get("device", []):
            if camera.get("kind") == "camera" and camera.get("name") == stream:
                if "video" not in camera:
                    camera["video"] = tomlkit.table()
                camera["video"].update(values)
                _write_manifest(path, tomlkit.dumps(doc))
                return
        raise ValueError("no such manifest camera")


def update_controls(
    stream: str,
    values: dict[str, int],
    *,
    auto_exposure: bool | None = None,
    reset: bool = False,
    path: pathlib.Path | None = None,
) -> None:
    """Merge native overrides without losing concurrent settings edits; capture owns applying the saved values.

    Reset replaces overrides with the caller's advertised defaults, suppressing inherited global overrides.
    Exposure ownership stays unchanged.
    """
    if any(not key.isdecimal() or str(int(key)) != key or type(value) is not int for key, value in values.items()):
        raise ValueError("control IDs must be decimal numbers and values must be integers")
    path = path or manifest_path()
    with _LOCK:
        doc = tomlkit.parse(path.read_text())
        for camera in doc.get("device", []):
            if camera.get("kind") != "camera" or camera.get("name") != stream:
                continue
            if "video" not in camera:
                camera["video"] = tomlkit.table()
            video = camera["video"]
            existing = video.get("controls", doc.get("video", {}).get("controls", {}))
            merged = {} if reset else dict(existing)
            merged.update(values)
            table = tomlkit.inline_table()
            table.update(merged)
            video["controls"] = table
            if auto_exposure is not None:
                video["auto_exposure"] = auto_exposure
            _write_manifest(path, tomlkit.dumps(doc))
            return
        raise ValueError("no such manifest camera")


def matches(match: dict[str, Any], name: str, usb: str | None, paths: Sequence[str] = ()) -> bool:
    """Whether a manifest `match` picks the camera called `name` with USB id `usb` and stable links `paths`
    (`stable_paths`): every field given must match, and one this can't check (a serial, say) doesn't."""
    if not match:
        return False
    for key, want in match.items():
        if key == "usb":
            if usb is None or str(want).lower() != usb.lower():
                return False
        elif key == "name":
            if str(want).casefold() not in name.casefold():
                return False
        elif key == "path":
            if str(want) not in paths:
                return False
        else:
            return False
    return True


def stable_paths(node: str | None) -> list[str]:
    """The stable links udev makes for a V4L2 node (`/dev/v4l/by-id/*`, then `/dev/v4l/by-path/*`), as capture lists
    them for a `path` match; none off Linux or for no node."""
    if not node:
        return []
    out = []
    for sub in ("by-id", "by-path"):
        for link in sorted(pathlib.Path("/dev/v4l", sub).glob("*")):
            try:
                if link.resolve() == pathlib.Path(node).resolve():
                    out.append(str(link))
            except OSError:
                continue
    return out


def usb_id(path: str | None) -> str | None:
    """The USB "vendor:product" of a V4L2 node (/dev/videoN), or None."""
    if not path:
        return None
    dev = pathlib.Path("/sys/class/video4linux") / pathlib.Path(path).name / "device"
    for d in (dev, *dev.resolve().parents[:3]):
        try:
            return f"{(d / 'idVendor').read_text().strip()}:{(d / 'idProduct').read_text().strip()}"
        except OSError:
            continue
    return None


def _blocks(text: str) -> list[str]:
    """The manifest's text cut before each table header (`[video]`, `[[device]]`), so a block can be dropped whole. A
    device's own subtable (`[device.video]`, `[device.match]`) stays in its device's block."""
    out: list[str] = []
    for line in text.splitlines(keepends=True):
        if not out or _HEADER.match(line) and not _SUBTABLE.match(line):
            out.append(line)
        else:
            out[-1] += line
    return out


def use(
    name: str,
    usb: str | None,
    on: bool,
    path: pathlib.Path | None = None,
    stream: str | None = None,
    kind: str = "camera",
    paths: Sequence[str] = (),
    shared: bool = False,
) -> str | None:
    """Adds a `[[device]]` of `kind` for the device called `name` (matched by `usb` when known, else by its name), or
    removes every block of that kind that matches it (or, given `stream`, the one with that manifest name). Returns the
    manifest name it has now (None once removed). A camera is named camN, a microphone micN. A camera whose USB id
    another plugged-in camera `shared` is matched by its first stable link in `paths`, so
    each gets its own worker."""
    path = path or manifest_path()
    with _LOCK:
        try:
            text = path.read_text()
        except FileNotFoundError:
            text = ""
        cams = manifest_devices(kind, path, active_only=False) if text else []
        mine = [
            c["name"] for c in cams if c["name"] == stream or stream is None and matches(c["match"], name, usb, paths)
        ]
        if on:
            if mine:
                doc = tomlkit.parse(text)
                for device in doc.get("device", []):
                    if device.get("kind") == kind and device.get("name") == mine[0]:
                        if device.get("enabled", True) is False:
                            device["enabled"] = True
                            _write_manifest(path, tomlkit.dumps(doc))
                        break
                return mine[0]
            taken = {c["name"] for c in cams}
            prefix = "cam" if kind == "camera" else "mic"
            stream = next(f"{prefix}{n}" for n in range(len(taken) + 1) if f"{prefix}{n}" not in taken)
            if usb and not (shared and paths):
                match = f'{{ usb = "{usb}" }}'
            elif paths:
                match = f'{{ path = "{paths[0]}" }}'
            else:
                match = f'{{ name = "{name.replace(chr(34), "")}" }}'
            text = text.rstrip("\n") + ("\n\n" if text.strip() else "")
            text += f'[[device]]\nname = "{stream}"\nkind = "{kind}"\nmatch = {match}  # {name}, chosen in Settings\n'
            path.write_text(text)
            return stream
        if not mine:
            return None
        kept = []
        for block in _blocks(text):
            try:
                doc = tomllib.loads(block)
            except tomllib.TOMLDecodeError:
                doc = {}
            devices = doc.get("device")
            device = devices[0] if isinstance(devices, list) and devices else {}
            if device.get("kind") == kind and device.get("name") in mine:
                continue
            kept.append(block)
        path.write_text("".join(kept))
        return None


def states(link: Any) -> dict[str, Any]:
    """Each manifest camera's status: {"ts", "cameras": {name: {"state": "on",
    "size": [w, h], "fps"}}}, a camera absent while capture sends it nothing. `link` is a continuum_link.BrokerLink.
    Asked of the broker at most every STATUS_EVERY_S. A status older than that answers at once and is read again in
    the background, one read at a time: reading every camera's last frames took about a second, which every
    page's event stream (camera_status) and Settings > Devices waited for every 2 s. Only the first read waits."""
    global _status_read
    with _status_lock:
        value = _cache["value"]
        if value is not None:
            if time.monotonic() - _cache["at"] >= STATUS_EVERY_S and _status_read is None:
                _status_read = threading.Thread(
                    target=_read_in_background, args=(link,), daemon=True, name="camera-status"
                )
                _status_read.start()
            return value
    with _status_lock:  # the first read: callers arriving together share it
        if _cache["value"] is None:
            _read(link)
        return _cache["value"]


def _read_in_background(link: Any) -> None:
    global _status_read
    try:
        _read(link)
    except Exception as ex:
        LOG.warning(f"camera status not read: {type(ex).__name__}: {ex}")
    finally:
        with _status_lock:
            _status_read = None


def _read(link: Any) -> dict[str, Any]:
    """Each manifest camera's status from its last STATUS_WINDOW_S of frames, kept in `_cache`."""
    import continuum_py

    cameras: dict[str, Any] = {}
    now = time.time_ns()
    lo = now - int(STATUS_WINDOW_S * 1e9)
    for cam in [*manifest_cameras(), *manifest_devices("rtsp")]:
        stream = f"{cam['name']}.video"
        try:
            items = link.call(lambda c, s=stream: c.query(s, lo, now))["items"]
        except Exception as e:  # no broker, or the stream isn't registered yet: nothing from it
            LOG.debug(f"camera status {stream}: {e!r}")
            continue
        headers = [continuum_py.decode_header(i["header"]) for i in items]
        if not headers:
            continue
        capture = headers[-1].get("capture") or {}
        size = [capture["width"], capture["height"]] if capture.get("width") and capture.get("height") else None
        cameras[cam["name"]] = {"state": "on", "size": size, "fps": round(len(headers) / STATUS_WINDOW_S)}
    value = {"ts": time.time(), "cameras": cameras}
    _cache.update(at=time.monotonic(), value=value)
    return value
