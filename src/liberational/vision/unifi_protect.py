"""Read local Protect camera metadata and already-enabled streams, without opening video or enabling anything.

Official routes: https://developer.ui.com/protect/v7.3.70/openapi.json . Public camera records omit IP addresses;
those remain unknown. Legacy bootstrap/channel compatibility follows the older maintained client:
https://raw.githubusercontent.com/uilibs/uiprotect/v5.4.0/src/uiprotect/data/devices.py . RTSPS URLs are bearer
credentials, including tokens in their path. They are kept only in private result fields excluded from repr.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, urlsplit

from liberational.proximity.network import ip_address, mac_address
from liberational.unifi import UniFiError, UniFiSession, local_origin


@dataclass(frozen=True)
class ProtectStream:
    quality: str
    url: str = field(repr=False)


@dataclass(frozen=True)
class ProtectCamera:
    id: str
    name: str | None = None
    model: str | None = None
    ip: str | None = None
    mac: str | None = None
    state: str | None = None
    streams: tuple[ProtectStream, ...] = field(default=(), repr=False)


@dataclass(frozen=True)
class ProtectDiscovery:
    cameras: tuple[ProtectCamera, ...]
    complete: bool
    warnings: tuple[str, ...] = ()


def _text(value: object, limit: int = 512) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str) or len(value) > limit or len(value.encode()) > limit:
        raise ValueError("invalid Protect metadata")
    return value


def _stream(quality: str, value: object) -> ProtectStream | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str) or len(value) > 2048:
        raise ValueError("invalid Protect stream")
    parts = urlsplit(value)
    if parts.scheme not in ("rtsp", "rtsps") or parts.fragment or not parts.path:
        raise ValueError("invalid Protect stream")
    local_origin("https://" + parts.netloc)
    return ProtectStream(quality, value)


class UniFiProtect:
    """A bounded camera-discovery pass. No stream creation, camera writes, or media probing is exposed."""

    def __init__(
        self, session: UniFiSession, *, mode: str = "legacy", max_cameras: int = 128, timeout_s: float = 10.0
    ) -> None:
        if mode not in ("legacy", "integration") or not 1 <= max_cameras <= 256:
            raise ValueError("invalid Protect discovery settings")
        if not math.isfinite(timeout_s) or not 0 < timeout_s <= 120:
            raise ValueError("invalid Protect discovery deadline")
        self.session, self.mode, self.max_cameras, self.timeout_s = session, mode, max_cameras, timeout_s

    def discover(self) -> ProtectDiscovery:
        cameras: list[ProtectCamera] = []
        warnings: list[str] = []
        complete = True
        try:
            deadline = time.monotonic() + self.timeout_s
            if self.mode == "integration":
                rows = self.session.protect_cameras(deadline=deadline)
                bootstrap: dict[str, Any] = {}
            else:
                bootstrap = self.session.protect_bootstrap(deadline=deadline)
                if not isinstance(bootstrap, dict):
                    raise UniFiError("Protect bootstrap is invalid")
                rows = bootstrap.get("cameras")
            if not isinstance(rows, list):
                raise UniFiError("Protect camera list is invalid")
            if len(rows) > self.max_cameras:
                complete = False
                warnings.append("Protect camera list exceeded the discovery bound.")
            seen: set[str] = set()
            for row in rows[: self.max_cameras]:
                try:
                    if (
                        not isinstance(row, dict)
                        or (camera_id := _text(row.get("id"), 128)) is None
                        or camera_id in seen
                    ):
                        raise ValueError("invalid or duplicate camera")
                    try:
                        streams = (
                            self._public_streams(camera_id, deadline)
                            if self.mode == "integration"
                            else self._legacy_streams(row, bootstrap)
                        )
                    except ValueError, TypeError, KeyError, UniFiError:
                        streams = ()
                        complete = False
                        if "Some Protect enabled streams could not be read." not in warnings:
                            warnings.append("Some Protect enabled streams could not be read.")
                    cameras.append(
                        ProtectCamera(
                            id=camera_id,
                            name=_text(row.get("name")),
                            model=_text(row.get("type")),
                            ip=ip_address(row.get("host")) if self.mode == "legacy" else None,
                            mac=mac_address(row.get("mac")),
                            state=_text(row.get("state")),
                            streams=streams,
                        )
                    )
                    seen.add(camera_id)
                except ValueError, TypeError, KeyError, UniFiError:
                    complete = False
                    if (
                        not warnings
                        or warnings[-1] != "Some Protect camera metadata or enabled streams could not be read."
                    ):
                        warnings.append("Some Protect camera metadata or enabled streams could not be read.")
        except UniFiError as e:
            complete = False
            warnings.append(str(e))
        return ProtectDiscovery(tuple(cameras), complete, tuple(warnings))

    def _public_streams(self, camera_id: str, deadline: float) -> tuple[ProtectStream, ...]:
        result = self.session.protect_streams(camera_id, deadline=deadline)
        if not isinstance(result, dict):
            raise ValueError("invalid Protect stream listing")
        streams = [_stream(quality, result.get(quality)) for quality in ("high", "medium", "low", "package")]
        return tuple(stream for stream in streams if stream is not None)

    def _legacy_streams(self, camera: dict[str, Any], bootstrap: dict[str, Any]) -> tuple[ProtectStream, ...]:
        channels = camera.get("channels")
        if not isinstance(channels, list) or len(channels) > 16:
            raise ValueError("invalid Protect channels")
        nvr = bootstrap.get("nvr")
        ports = nvr.get("ports") if isinstance(nvr, dict) else None
        if not isinstance(ports, dict):
            if any(isinstance(channel, dict) and channel.get("isRtspEnabled") is True for channel in channels):
                raise ValueError("Protect stream ports are unknown")
            return ()
        origin = urlsplit(self.session.origin)
        host = origin.hostname or ""
        host = f"[{host}]" if ":" in host else host
        streams: list[ProtectStream] = []
        for channel in channels:
            if not isinstance(channel, dict):
                raise ValueError("invalid Protect channel")
            if channel.get("isRtspEnabled") is not True:
                continue
            alias = _text(channel.get("rtspAlias"), 512)
            if alias is None:
                raise ValueError("Protect enabled stream alias is unknown")
            quality = {0: "high", 1: "medium", 2: "low", 3: "package"}.get(channel.get("id"), "other")
            for scheme in ("rtsps", "rtsp"):
                port = ports.get(scheme)
                if port is None:
                    continue
                if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
                    raise ValueError("invalid Protect stream port")
                value = f"{scheme}://{host}:{port}/{quote(alias, safe='')}" + (
                    "?enableSrtp" if scheme == "rtsps" else ""
                )
                stream = _stream(quality, value)
                if stream is not None:
                    streams.append(stream)
        return tuple(streams)
