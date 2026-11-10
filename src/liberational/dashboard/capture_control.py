"""Capture executable resolution and requests to its local supervisor.

An accepted restart is asynchronous. Only an unavailable control endpoint can authorize a caller's fallback;
timeouts and malformed replies may follow an accepted request, so they must never trigger another restart here.
"""

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from liberational import config, services


@dataclass(frozen=True, slots=True)
class RestartResult:
    status: Literal["restarting", "unavailable", "failed"]
    message: str


def capture_binary() -> str | None:
    """Prefer the executable used by the declared capture service, then a PATH installation."""
    service = services.declared().get("continuum-capture")
    if service is not None:
        executable = services.argv(service)[0]
        if Path(executable).is_file() and os.access(executable, os.X_OK):
            return executable
    return shutil.which("continuum-capture")


def restart_camera(device: str) -> RestartResult:
    """Ask the capture supervisor to restart exactly one camera; never restart services as a fallback."""
    if not isinstance(device, str) or not device.strip() or any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in device):
        return RestartResult("failed", "Camera name must be a nonempty string without control characters")
    try:
        if len(device.encode("utf-8")) > 256:
            return RestartResult("failed", "Camera name exceeds 256 UTF-8 bytes")
    except UnicodeError:
        return RestartResult("failed", "Camera name must be valid UTF-8")
    try:
        binary = capture_binary()
        if binary is None:
            return RestartResult("unavailable", "continuum-capture is not installed")
        result = subprocess.run(
            [binary, "--restart-camera", device, "--state-dir", str(config.path("paths.state"))],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except FileNotFoundError:
        return RestartResult("unavailable", "continuum-capture is not installed")
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return RestartResult("failed", f"Camera restart request failed: {exc}")
    if (
        result.returncode == 2
        and not result.stdout.strip()
        and result.stderr.lstrip().startswith("usage: continuum-capture ")
    ):
        return RestartResult("unavailable", "Installed continuum-capture does not support camera restart requests")
    try:
        reply = json.loads(result.stdout)
    except ValueError:
        return RestartResult("failed", "Invalid response to camera restart request")
    if (
        not isinstance(reply, dict)
        or reply.get("device") != device
        or not isinstance(reply.get("message"), str)
        or (result.returncode, reply.get("status")) not in ((0, "restarting"), (3, "unavailable"), (1, "failed"))
    ):
        return RestartResult("failed", "Invalid response to camera restart request")
    return RestartResult(reply["status"], reply["message"])
