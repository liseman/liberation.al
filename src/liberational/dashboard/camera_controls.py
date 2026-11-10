"""Read native camera controls and persist requests for capture's live control coordinator.

Only capture writes the device: its exposure steering and manual settings share one owner. Discovery uses a
separate, read-only V4L2 handle and never allocates buffers or changes streaming state.
"""

import errno
import os
import struct
import sys
from typing import Any

from liberational.dashboard import cameras as manifest

QUERY_CTRL = 0xC0445624
GET_CTRL = 0xC008561B
QUERY_MENU = 0xC02C5625
NEXT_CTRL = 0x80000000
MANAGED = frozenset((9963776, 10094849, 10094850))  # brightness, exposure mode, absolute exposure


def _ioctl(fd: int, request: int, data: bytearray) -> None:
    import fcntl

    fcntl.ioctl(fd, request, data, True)


def native_controls(path: str) -> list[dict[str, Any]]:
    """Enumerate readable scalar controls, preserving menu holes and the driver's current flags."""
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    try:
        controls: list[dict[str, Any]] = []
        previous = 0
        while True:
            query = bytearray(struct.pack("=II32siiiiI8s", previous | NEXT_CTRL, 0, b"", 0, 0, 0, 0, 0, b""))
            try:
                _ioctl(fd, QUERY_CTRL, query)
            except OSError as error:
                if error.errno == errno.EINVAL:
                    break
                raise
            ident, kind, name, minimum, maximum, step, default, flags, _ = struct.unpack("=II32siiiiI8s", query)
            if ident <= previous:
                raise OSError("camera returned an invalid control sequence")
            previous = ident
            if kind not in (1, 2, 3, 9) or flags & (1 | 64):  # disabled or write-only
                continue
            current = bytearray(struct.pack("=Ii", ident, 0))
            try:
                _ioctl(fd, GET_CTRL, current)
            except OSError:
                continue  # classes and controls without readable scalar values have no usable widget
            row: dict[str, Any] = {
                "id": ident,
                "name": name.split(b"\0", 1)[0].decode(errors="replace"),
                "type": "boolean" if kind == 2 else "menu" if kind in (3, 9) else "integer",
                "min": minimum,
                "max": maximum,
                "step": max(1, step),
                "default": default,
                "current": struct.unpack("=Ii", current)[1],
                "disabled": bool(flags & (2 | 4 | 16)),  # grabbed, read-only, inactive
            }
            if kind in (3, 9):
                options = []
                for index in range(minimum, maximum + 1):
                    menu = bytearray(struct.pack("=II32sI", ident, index, b"", 0))
                    try:
                        _ioctl(fd, QUERY_MENU, menu)
                    except OSError as error:
                        if error.errno == errno.EINVAL:
                            continue
                        raise
                    label = (
                        str(struct.unpack_from("=q", menu, 8)[0])
                        if kind == 9
                        else bytes(menu[8:40]).split(b"\0", 1)[0].decode(errors="replace")
                    )
                    options.append({"value": index, "label": label})
                row["options"] = options
            controls.append(row)
        return controls
    finally:
        os.close(fd)


def view(camera: str, can_control: bool) -> tuple[int, dict[str, Any]]:
    """Current hardware readback and effective per-camera ownership, without any writes."""
    from liberational.dashboard import devices

    base: dict[str, Any] = {
        "camera": camera,
        "available": False,
        "can_control": can_control,
        "auto_exposure": False,
        "pending": False,
        "controls": [],
    }
    if any(row["name"] == camera for row in manifest.manifest_devices("rtsp")):
        return 200, {**base, "note": "Camera settings live on the NVR."}
    if not any(row["name"] == camera for row in manifest.manifest_cameras()):
        return 404, {**base, "error": "no such camera"}
    if not sys.platform.startswith("linux"):
        return 200, {**base, "note": "Camera controls are not adjustable here."}
    try:
        settings = manifest.video_settings(camera)
        base["auto_exposure"] = bool(settings.get("auto_exposure", True))
        node = devices.camera_control_node(camera)
        if node is None:
            return 200, {**base, "note": "Camera is not attached."}
        controls = native_controls(node)
    except (OSError, ValueError, StopIteration) as error:
        return 200, {**base, "note": f"Couldn't read camera controls: {error}"}
    desired = settings.get("controls", {})
    for control in controls:
        if base["auto_exposure"] and control["id"] in MANAGED:
            control["disabled"] = True
        if not control["disabled"] and str(control["id"]) in desired:
            base["pending"] |= desired[str(control["id"])] != control["current"]
    return 200, {
        **base,
        "available": bool(controls),
        "controls": controls,
        "note": "" if controls else "This camera exposes no adjustable controls.",
    }


def save(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Validate advertised controls, then save atomically; capture applies them without restarting video."""
    camera = body.get("camera")
    if not isinstance(camera, str) or not camera or set(body) - {"camera", "values", "auto_exposure", "reset"}:
        return 400, {"error": "send camera and camera controls"}
    values, auto, reset = body.get("values", {}), body.get("auto_exposure"), body.get("reset", False)
    if (
        not isinstance(values, dict)
        or ("auto_exposure" in body and not isinstance(auto, bool))
        or not isinstance(reset, bool)
        or (reset and bool(values))
        or not (values or reset or "auto_exposure" in body)
    ):
        return 400, {"error": "send control values, auto_exposure, or reset"}
    code, report = view(camera, True)
    if code != 200:
        return code, report
    if not report["available"]:
        return 400, {"error": report.get("note", "Camera controls are unavailable")}
    descriptors = {str(control["id"]): control for control in report["controls"]}
    if reset:
        values = {key: control["default"] for key, control in descriptors.items() if not control["disabled"]}
    normalized: dict[str, int] = {}
    for key, value in values.items():
        control = descriptors.get(key)
        if control is None or control["disabled"] or auto is True and control["id"] in MANAGED:
            return 400, {"error": f"Control {key} is unavailable or controlled automatically"}
        if control["type"] == "boolean":
            valid = isinstance(value, bool) or type(value) is int and value in (0, 1)
        else:
            valid = type(value) is int
        if not valid:
            return 400, {"error": f"Invalid value for {control['name']}"}
        value = int(value)
        if not control["min"] <= value <= control["max"] or (value - control["min"]) % control["step"]:
            return 400, {"error": f"Value is outside {control['name']}'s range or step"}
        if control["type"] == "menu" and value not in {option["value"] for option in control["options"]}:
            return 400, {"error": f"Invalid choice for {control['name']}"}
        normalized[key] = value
    try:
        manifest.update_controls(camera, normalized, auto_exposure=auto, reset=reset)
    except ValueError as error:
        return 400, {"error": str(error)}
    except OSError as error:
        return 500, {"error": f"Couldn't save camera controls: {error}"}
    _, updated = view(camera, True)
    return 200, {**updated, "pending": True}
