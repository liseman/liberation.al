"""Read-only capture capability discovery for the camera settings picker.

The manifest accepts whole-number fps. Only explicitly enumerated discrete sizes and exact integral rates are
selectable; rational rates and symbolic ranges remain in the report without rounding or inventing choices.
"""

import copy
import json
import subprocess
import threading
import time
from typing import Any, TypedDict, TypeGuard

from liberational.dashboard import capture_control


class ModeChoice(TypedDict):
    width: int
    height: int
    fps: list[int]


class ModeReport(TypedDict):
    status: str
    format: str
    fourcc: str
    sizes: list[dict[str, Any]]
    choices: list[ModeChoice]
    diagnostics: list[str]


_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_lock = threading.Lock()
_reading: dict[str, threading.Thread] = {}  # a stale format's background read, while one runs
_TTL = 5.0
_TIMEOUT = 5.0


def _format(value: str) -> tuple[str, str]:
    if value.upper() in ("MJPEG", "MJPG"):
        return "MJPEG", "MJPG"
    if len(value) != 4 or not value.strip() or any(not 32 <= ord(c) <= 126 for c in value):
        raise ValueError("format must be MJPEG or a four-byte ASCII FourCC")
    return value, value


def _inventory(format: str, fresh: bool) -> dict[str, Any]:
    """The capture binary's capability list for a format. A cached one older than _TTL answers at once and is read
    again in the background, one read per format at a time (#2568): the read is a child process that took about a
    second, which Settings > Devices waited for every 5 s. A `fresh` ask, and the first, read it now. No read holds
    the lock while its child runs."""
    with _lock:
        cached = _cache.get(format)
        if not fresh and cached:
            if time.monotonic() - cached[0] >= _TTL and format not in _reading:
                _reading[format] = threading.Thread(
                    target=_read_in_background, args=(format,), daemon=True, name="camera-modes"
                )
                _reading[format].start()
            return copy.deepcopy(cached[1])
    return copy.deepcopy(_read(format))


def _read_in_background(format: str) -> None:
    try:
        _read(format)
    finally:
        with _lock:
            _reading.pop(format, None)


def _read(format: str) -> dict[str, Any]:
    """Read one format's capability list from the capture binary, and keep it."""
    report = _probe(format)
    with _lock:
        _cache[format] = (time.monotonic(), report)
    return report


def _probe(format: str) -> dict[str, Any]:
    try:
        binary = capture_control.capture_binary()
        if binary is None:
            raise ValueError("continuum-capture is not installed")
        result = subprocess.run(
            [binary, "--list", "--json", "--format", format],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            check=False,
        )
        if result.returncode:
            raise ValueError("camera capability discovery failed")
        report = json.loads(result.stdout)
        if (
            not isinstance(report, dict)
            or type(report.get("schema_version")) is not int
            or report["schema_version"] != 1
        ):
            raise ValueError("unsupported camera capability schema")
        if not isinstance(report.get("devices"), list) or report.get("format") != format:
            raise ValueError("invalid camera capability inventory")
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {"devices": [], "diagnostics": [str(exc)], "unavailable": True}
    return report


def _positive(value: object) -> TypeGuard[int]:
    return type(value) is int and 0 < value <= 0xFFFFFFFF


def _diagnostics(value: Any) -> list[str]:
    return [v for v in value if isinstance(v, str)] if isinstance(value, list) else []


def modes(node: str, format: str = "MJPEG", fresh: bool = False) -> ModeReport:
    """Return JSON-ready native descriptors and conservative selectable choices for one node or stable path."""
    format, fourcc = _format(format)
    inventory = _inventory(format, fresh)
    report: ModeReport = dict(
        status="unavailable",
        format=format,
        fourcc=fourcc,
        sizes=[],
        choices=[],
        diagnostics=_diagnostics(inventory.get("diagnostics")),
    )
    camera = next(
        (
            d
            for d in inventory["devices"]
            if isinstance(d, dict)
            and d.get("kind") == "camera"
            and (d.get("node") == node or isinstance(d.get("paths"), list) and node in d["paths"])
        ),
        None,
    )
    if camera is None:
        if not inventory.get("unavailable"):
            report["diagnostics"].append("Camera is absent from capability inventory")
        return report
    native = camera.get("camera_modes")
    if (
        not isinstance(native, dict)
        or native.get("fourcc") != fourcc
        or native.get("status") not in ("ok", "partial", "unavailable", "unsupported")
    ):
        report["diagnostics"].append("Invalid camera capability report")
        return report
    report["status"] = native["status"]
    report["diagnostics"].extend(_diagnostics(native.get("diagnostics")))
    sizes = native.get("sizes")
    if not isinstance(sizes, list):
        report["status"] = "unavailable"
        report["diagnostics"].append("Invalid camera size descriptors")
        return report
    report["sizes"] = [s for s in sizes if isinstance(s, dict)]
    if len(report["sizes"]) != len(sizes):
        report["diagnostics"].append("Invalid camera size descriptors were excluded")
    if report["status"] in ("unavailable", "unsupported"):
        return report
    choices: dict[tuple[int, int], set[int]] = {}
    for size in report["sizes"]:
        if size.get("kind") != "discrete":
            report["diagnostics"].append("Size ranges require a discrete mode and cannot be selected")
            continue
        width, height = size.get("width"), size.get("height")
        if not _positive(width) or not _positive(height):
            report["diagnostics"].append("Invalid camera dimensions were excluded")
            continue
        report["diagnostics"].extend(_diagnostics(size.get("diagnostics")))
        intervals = size.get("intervals")
        if not isinstance(intervals, list):
            report["diagnostics"].append("Missing camera interval descriptors")
            continue
        for interval in intervals:
            if not isinstance(interval, dict) or interval.get("kind") != "discrete":
                report["diagnostics"].append("Interval ranges cannot be selected as whole-number frame rates")
                continue
            rate, seconds = interval.get("fps"), interval.get("interval")
            if not isinstance(rate, dict) or not isinstance(seconds, dict):
                report["diagnostics"].append("Invalid camera rate descriptors were excluded")
                continue
            n, d = rate.get("numerator"), rate.get("denominator")
            sn, sd = seconds.get("numerator"), seconds.get("denominator")
            if not (_positive(n) and _positive(d) and _positive(sn) and _positive(sd)) or n * sn != d * sd:
                report["diagnostics"].append("Invalid or inconsistent camera rate rationals were excluded")
                continue
            if n % d:
                report["diagnostics"].append("Non-integral frame rates cannot be stored in camera settings")
                continue
            choices.setdefault((width, height), set()).add(n // d)
    report["choices"] = [dict(width=w, height=h, fps=sorted(rates)) for (w, h), rates in sorted(choices.items())]
    report["diagnostics"] = list(dict.fromkeys(report["diagnostics"]))
    return report


def validate_choice(node: str, width: int, height: int, fps: int, format: str = "MJPEG") -> dict[str, int]:
    """Refresh capabilities and return a validated cameras.update_video payload, or raise ValueError."""
    if not all(_positive(v) for v in (width, height, fps)):
        raise ValueError("width, height, and fps must be positive whole integers")
    report = modes(node, format, fresh=True)
    if not any(c["width"] == width and c["height"] == height and fps in c["fps"] for c in report["choices"]):
        raise ValueError("camera mode is not explicitly offered by current capability discovery")
    return {"width": width, "height": height, "fps": fps}
