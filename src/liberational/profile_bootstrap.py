"""Opt-in finite CPU profiles for existing pack console entry points.

PACK_PROFILE=1 profiles this process; a comma-separated service list selects console entry points by name.
PACK_PROFILE_SERVICE supplies a name for a Python module entry point. CLI administration remains excluded.
No command wrapper, signal handler, executable replacement, or runtime-data read is needed. Environment controls
are diagnostic overrides, independent of pack.toml, so enabling one service requires no configuration migration.
"""

import atexit
import math
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from liberational.service_profile import Profiler

_profile: Profiler | None = None


def _seconds(name: str, default: float, lower: float, upper: float) -> float:
    value = float(os.environ.get(name, str(default)))
    if not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper} seconds")
    return value


def _service() -> str | None:
    command = Path(sys.argv[0]).name if sys.argv else ""
    explicit = os.environ.get("PACK_PROFILE_SERVICE", "")
    if command in {"pack-services", "pack-config"}:
        return None
    service = explicit or (command.removeprefix("pack-") if command.startswith("pack-") else "")
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,79}", service):
        return None
    switch = os.environ.get("PACK_PROFILE", "").strip()
    selected = {name.strip() for name in switch.split(",")}
    return service if switch.lower() in {"1", "true", "yes"} or service in selected else None


def maybe_start() -> None:
    """Start once when selected, reporting diagnostic failure without preventing the service from starting."""
    global _profile
    service = _service()
    if service is None or _profile is not None:
        return
    from liberational import config, logs

    log = logs.get("profile")
    try:
        interval = _seconds("PACK_PROFILE_INTERVAL_SECONDS", 0.05, 0.01, 1.0)
        flush = _seconds("PACK_PROFILE_FLUSH_SECONDS", 30.0, 1.0, 300.0)
        duration = _seconds("PACK_PROFILE_SECONDS", 120.0, 1.0, 3600.0)
        delay = _seconds("PACK_PROFILE_DELAY_SECONDS", 30.0, 0.0, 600.0)
        from liberational.service_profile import Profiler

        profiler = Profiler(
            service,
            config.path("paths.logs") / "profiles",
            interval_s=interval,
            flush_s=flush,
            duration_s=duration,
            delay_s=delay,
        )
        _profile = profiler.start()
        atexit.register(profiler.stop)
        log.info(
            "CPU profile scheduled for %s: delay %.1fs, capture %.1fs, interval %.3fs",
            service,
            delay,
            duration,
            interval,
        )
    except (OSError, RuntimeError, ValueError) as error:
        log.warning("CPU profile unavailable for %s: %s", service, error)
