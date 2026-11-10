"""Learning surroundings: whether the cameras' pictures may refine a place's map, and from which cameras (#254).
docs/SPATIAL.md has the design.

**The switch** (#255) is two settings. `spatial.learn_surroundings` says whether to learn (off by default), and
`spatial.learn_cameras` names the cameras to sample; with none named, the first camera in the device manifest is
sampled. Both are live (`live = true` in config/labels.toml): `read` takes them from pack.toml as it is now, parsed
again only when the file changes, so what the Live tab saves (`save`) reaches a running sampler on its next look
with no restart. An environment variable (`PACK_SPATIAL_LEARN_SURROUNDINGS`, `PACK_SPATIAL_LEARN_CAMERAS`) still
wins, as for any setting.

A name the manifest doesn't have (a camera since renamed or removed) is never sampled, and never replaced by the
default: with only such names chosen, nothing is sampled, and the Live tab shows that none is chosen.
"""

from __future__ import annotations

import functools
import os
import threading
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from liberational import config as cfg

ON = "spatial.learn_surroundings"
CAMERAS = "spatial.learn_cameras"
CAMERA_KINDS = ("camera", "rtsp")  # the manifest's kinds that are cameras


class UnknownCamera(ValueError):
    """A camera name the device manifest doesn't have."""


@dataclass(frozen=True)
class Switch:
    """What to learn from now. `cameras` are the ones to sample: the chosen ones the manifest has, or its first
    camera when none is chosen. `chosen` is the setting as saved, and `default` the manifest's first camera."""

    on: bool
    cameras: tuple[str, ...]
    chosen: tuple[str, ...]
    default: str | None


def manifest_cameras(path: Path | None = None) -> list[str]:
    """The enabled cameras in the device manifest, in file order; none when it's missing or doesn't parse."""
    try:
        rows = tomllib.loads((path or cfg.devices_file()).read_text()).get("device", [])
    except OSError, ValueError:
        return []
    if not isinstance(rows, list):
        return []
    return [
        row["name"]
        for row in rows
        if isinstance(row, dict)
        and isinstance(row.get("name"), str)
        and row.get("enabled", True) is True
        and (row.get("kind") in CAMERA_KINDS or "video" in row)
    ]


_lock = threading.Lock()
_seen: tuple[Any, dict[str, Any]] | None = None  # (pack.toml's path and stat, its [spatial] table)


def _local() -> dict[str, Any]:
    """pack.toml's [spatial] table as the file is now, parsed again only when it changes; {} without the file or
    when it doesn't parse (a half-edited file never turns learning on)."""
    global _seen
    path = cfg.local_config()
    try:
        st = path.stat()
        stamp: Any = (str(path), st.st_mtime_ns, st.st_size, st.st_ino)
    except OSError:
        stamp = (str(path), None)
    with _lock:
        if _seen is not None and _seen[0] == stamp:
            return _seen[1]
        table: Any = {}
        if stamp[1] is not None:
            try:
                table = tomllib.loads(path.read_text()).get("spatial", {})
            except OSError, ValueError:
                table = {}
        _seen = (stamp, table if isinstance(table, dict) else {})
        return _seen[1]


@functools.cache
def _shipped() -> dict[str, Any]:
    return tomllib.loads(cfg.DEFAULTS.read_text())["spatial"]


#: Each setting through config, for when the environment sets it (the keys written out, so readers.py sees them).
_FROM_CONFIG = {
    "learn_surroundings": lambda: cfg.get("spatial.learn_surroundings"),
    "learn_cameras": lambda: cfg.get("spatial.learn_cameras"),
}


def _now(name: str) -> Any:
    """`spatial.<name>` now: from the environment, else pack.toml as it is now, else defaults.toml. Not the
    process's cached config, which would keep what pack.toml said when the process started."""
    if os.environ.get(f"PACK_SPATIAL_{name.upper()}"):
        return _FROM_CONFIG[name]()
    local = _local()
    return local[name] if name in local else _shipped()[name]


def read(manifest: Path | None = None) -> Switch:
    """The switch as saved now, against the cameras the manifest has now."""
    on = _now("learn_surroundings") is True
    saved = _now("learn_cameras")
    chosen = tuple(c for c in saved if isinstance(c, str)) if isinstance(saved, list) else ()
    have = manifest_cameras(manifest)
    default = have[0] if have else None
    if chosen:
        cameras = tuple(c for c in have if c in chosen)  # in the manifest's order, each once
    else:
        cameras = (default,) if default is not None else ()
    return Switch(on, cameras, chosen, default)


def save(on: bool | None = None, cameras: Sequence[str] | None = None, manifest: Path | None = None) -> Switch:
    """Turn learning on or off, choose the cameras, or both (None leaves that one as it is), in pack.toml; the
    switch as it then reads. An empty choice goes back to the default camera. Raises UnknownCamera for a name the
    manifest doesn't have, and writes nothing then."""
    values: dict[str, Any] = {}
    if on is not None:
        values[ON] = bool(on)
    if cameras is not None:
        have = manifest_cameras(manifest)
        unknown = [c for c in cameras if c not in have]
        if unknown:
            raise UnknownCamera(f"no camera named {unknown[0]!r} in the device manifest")
        values[CAMERAS] = [c for c in have if c in cameras]
    if values:
        cfg.save(values)
    return read(manifest)
