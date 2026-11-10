"""Small local-only, read-only UniFi HTTP session shared by Network and Protect.

Network reads `[unifi]`. Protect reads `[unifi_protect]` when its url is set (a separate NVR console), else the
same `[unifi]` console. With verify_tls off, urllib3's unverified-TLS warning is silenced for that host only.

Local account cookies support viewer permissions. Integration API keys are optional; the key's permissions are
controller policy, not a guarantee this client can establish. Only the fixed GET routes below and the local login
POST are allowed. Credentials, cookies, response bodies, and stream tokens never enter exception messages.

Primary API references:
https://developer.ui.com/network/v10.6.106/openapi.json
https://developer.ui.com/protect/v7.3.70/openapi.json
Legacy local-login paths are compatibility APIs, demonstrated by:
https://raw.githubusercontent.com/Kane610/aiounifi/master/aiounifi/interfaces/connectivity.py
https://dl.ui.com/unifi/8.0.28/unifi_sh_api
"""

import ipaddress
import json
import math
import re
import threading
import time
import uuid
import warnings
from typing import Any
from urllib.parse import urlencode, urlsplit

import requests
from urllib3.exceptions import InsecureRequestWarning

from liberational import logs

LOG = logs.get("unifi")

_LOCAL_NETWORKS = tuple(
    ipaddress.ip_network(value)
    for value in (
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "fc00::/7",
        "fe80::/10",
        "::1/128",
    )
)
_SITE = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
_CAMERA = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_UNVERIFIED_LOCK = threading.Lock()
_unverified_hosts: set[str] = set()


def _allow_unverified(origin: str) -> None:
    """Silence urllib3's unverified-TLS warning for one console's host only, and say so once per process.

    The filter matches urllib3's message for this exact host, so other hosts (and verified requests, which never
    warn) are unaffected. warnings.catch_warnings would mutate the process-wide filter list per request, which is
    not thread-safe, so a host-scoped permanent filter is used instead.
    """
    host = urlsplit(origin).hostname or ""
    with _UNVERIFIED_LOCK:
        if host in _unverified_hosts:
            return
        _unverified_hosts.add(host)
        warnings.filterwarnings(
            "ignore",
            message=f"Unverified HTTPS request is being made to host '{re.escape(host)}'",
            category=InsecureRequestWarning,
        )
    LOG.warning("TLS certificate verification is off for the UniFi console at %s", origin)


class UniFiError(RuntimeError):
    """A bounded, credential-free error safe for a coverage warning."""


class _Unauthorized(UniFiError):
    pass


def local_origin(value: str) -> str:
    """An HTTPS origin with an explicit local unicast IP, never a hostname or cloud endpoint."""
    try:
        parts = urlsplit(value)
        host, port = parts.hostname, parts.port
        if (
            parts.scheme != "https"
            or not host
            or parts.username is not None
            or parts.password is not None
            or parts.path not in ("", "/")
            or parts.query
            or parts.fragment
            or "%" in host
        ):
            raise ValueError()
        address = ipaddress.ip_address(host)
        if address.is_unspecified or address.is_multicast or not any(address in network for network in _LOCAL_NETWORKS):
            raise ValueError()
        if port is not None and not 1 <= port <= 65535:
            raise ValueError()
        authority = f"[{address}]" if address.version == 6 else str(address)
        return "https://" + authority + (f":{port}" if port is not None else "")
    except ValueError, TypeError:
        raise UniFiError(
            "UniFi requires an HTTPS origin with a local unicast IP literal and no credentials or path"
        ) from None


class UniFiSession:
    """A serialized cookie/API-key session with fixed routes, bounded reads, and no redirects or proxies.

    Deadlines are checked between bounded raw reads. A single socket operation may additionally take up to its
    configured timeout. DNS resolution is avoided by accepting only local IP origins. Responses must be identity
    encoded, so compressed-body expansion cannot bypass the byte cap.
    """

    def __init__(
        self,
        base_url: str,
        *,
        username: str = "",
        password: str = "",
        api_key: str = "",
        unifi_os: bool = True,
        verify_tls: bool | str = True,
        timeout_s: float = 5.0,
        max_response_bytes: int = 2 * 1024 * 1024,
        session: requests.Session | None = None,
    ) -> None:
        self.origin = local_origin(base_url)
        if not math.isfinite(timeout_s) or not 0 < timeout_s <= 30 or not 1 <= max_response_bytes <= 16 * 1024 * 1024:
            raise ValueError("invalid UniFi HTTP bounds")
        if not isinstance(verify_tls, bool | str) or isinstance(verify_tls, str) and not verify_tls:
            raise ValueError("verify_tls must be a boolean or CA certificate path")
        if any(len(value) > 8192 for value in (username, password, api_key)):
            raise ValueError("UniFi credential exceeds bounds")
        self.unifi_os, self.verify_tls = unifi_os, verify_tls
        if verify_tls is False:
            _allow_unverified(self.origin)
        self.timeout_s, self.max_response_bytes = timeout_s, max_response_bytes
        self._username, self._password, self._api_key = username, password, api_key
        self._session = session or requests.Session()
        self._session.trust_env = False
        self._lock, self._logged_in = threading.RLock(), False

    def close(self) -> None:
        with self._lock:
            try:
                self._session.close()
            except Exception:
                raise UniFiError("UniFi session close failed") from None
            finally:
                self._logged_in = False

    def _remaining(self, deadline: float) -> float:
        remaining = min(self.timeout_s, deadline - time.monotonic())
        if remaining <= 0:
            raise UniFiError("UniFi request deadline reached")
        return remaining

    def _request(
        self,
        method: str,
        path: str,
        *,
        deadline: float,
        api_key: bool = False,
        body: dict[str, str] | None = None,
        query: dict[str, int] | None = None,
    ) -> Any:
        headers = {"Accept": "application/json", "Accept-Encoding": "identity"}
        if api_key:
            if not self._api_key:
                raise UniFiError("UniFi Integration API key is not configured")
            headers["X-API-KEY"] = self._api_key
        url = self.origin + path + ("?" + urlencode(query) if query else "")
        response: requests.Response | None = None
        try:
            remaining = self._remaining(deadline)
            response = self._session.request(
                method,
                url,
                headers=headers,
                json=body,
                timeout=(remaining, remaining),
                allow_redirects=False,
                stream=True,
                verify=self.verify_tls,
            )
            if 300 <= response.status_code < 400:
                raise UniFiError("UniFi redirect refused")
            if response.status_code == 401:
                raise _Unauthorized("UniFi authentication required")
            if response.status_code >= 400:
                raise UniFiError(f"UniFi request refused (HTTP {response.status_code})")
            if response.headers.get("Content-Encoding", "identity").lower() not in ("", "identity"):
                raise UniFiError("UniFi compressed response refused")
            length = response.headers.get("Content-Length")
            if length is not None and (not length.isdecimal() or int(length) > self.max_response_bytes):
                raise UniFiError("UniFi response exceeds byte bounds")
            data = bytearray()
            while True:
                self._remaining(deadline)
                chunk = response.raw.read1(min(16384, self.max_response_bytes + 1 - len(data)), decode_content=False)
                if not chunk:
                    break
                data.extend(chunk)
                if len(data) > self.max_response_bytes:
                    raise UniFiError("UniFi response exceeds byte bounds")
            try:
                return json.loads(data)
            except ValueError, UnicodeError:
                raise UniFiError("UniFi response is not valid JSON") from None
        except requests.Timeout:
            raise UniFiError("UniFi request timed out") from None
        except requests.RequestException, OSError:
            raise UniFiError("UniFi local connection failed") from None
        finally:
            if response is not None:
                response.close()

    def _deadline(self, deadline: float | None) -> float:
        return time.monotonic() + self.timeout_s if deadline is None else deadline

    def _login(self, deadline: float) -> None:
        if not self._username or not self._password:
            raise UniFiError("UniFi local username and password are not configured")
        result = self._request(
            "POST",
            "/api/auth/login" if self.unifi_os else "/api/login",
            deadline=deadline,
            body={"username": self._username, "password": self._password},
        )
        if not isinstance(result, dict) or isinstance(result.get("meta"), dict) and result["meta"].get("rc") != "ok":
            raise UniFiError("UniFi local authentication failed")
        self._logged_in = True

    def _cookie_get(self, path: str, deadline: float | None) -> Any:
        with self._lock:
            end = self._deadline(deadline)
            if not self._logged_in:
                self._login(end)
            for attempt in range(2):
                try:
                    result = self._request("GET", path, deadline=end)
                    if isinstance(result, dict) and isinstance(result.get("meta"), dict):
                        meta = result["meta"]
                        if meta.get("rc") != "ok":
                            if meta.get("msg") == "api.err.LoginRequired":
                                raise _Unauthorized("UniFi authentication required")
                            raise UniFiError("UniFi controller rejected the read request")
                    return result
                except _Unauthorized:
                    self._logged_in = False
                    if attempt:
                        raise UniFiError("UniFi local authentication expired") from None
                    self._login(end)
        raise UniFiError("UniFi local authentication failed")

    def _integration_get(self, path: str, deadline: float | None, query: dict[str, int] | None = None) -> Any:
        with self._lock:
            return self._request("GET", path, deadline=self._deadline(deadline), api_key=True, query=query)

    def legacy_clients(self, site: str = "default", *, deadline: float | None = None) -> Any:
        if not _SITE.fullmatch(site):
            raise ValueError("invalid UniFi legacy site reference")
        prefix = "/proxy/network" if self.unifi_os else ""
        return self._cookie_get(f"{prefix}/api/s/{site}/stat/sta", deadline)

    def integration_sites(self, offset: int = 0, limit: int = 200, *, deadline: float | None = None) -> Any:
        return self._integration_get("/proxy/network/integration/v1/sites", deadline, self._page(offset, limit))

    def integration_clients(
        self, site_uuid: str, offset: int = 0, limit: int = 200, *, deadline: float | None = None
    ) -> Any:
        try:
            site = str(uuid.UUID(site_uuid))
        except ValueError:
            raise ValueError("invalid UniFi Integration site UUID") from None
        return self._integration_get(
            f"/proxy/network/integration/v1/sites/{site}/clients", deadline, self._page(offset, limit)
        )

    @staticmethod
    def _page(offset: int, limit: int) -> dict[str, int]:
        if not 0 <= offset <= 2**31 - 1 or not 1 <= limit <= 200:
            raise ValueError("invalid UniFi pagination bounds")
        return dict(offset=offset, limit=limit)

    def protect_bootstrap(self, *, deadline: float | None = None) -> Any:
        if not self.unifi_os:
            raise UniFiError("Protect discovery requires a UniFi OS console")
        return self._cookie_get("/proxy/protect/api/bootstrap", deadline)

    def protect_cameras(self, *, deadline: float | None = None) -> Any:
        return self._integration_get("/proxy/protect/integration/v1/cameras", deadline)

    def protect_streams(self, camera_id: str, *, deadline: float | None = None) -> Any:
        if not _CAMERA.fullmatch(camera_id):
            raise ValueError("invalid UniFi camera identifier")
        return self._integration_get(f"/proxy/protect/integration/v1/cameras/{camera_id}/rtsps-stream", deadline)


# Each console's settings, by the start of their keys. configured_session reads the whole section, so the starts are
# written out here: the restart scan (readers.py) finds a setting only where its name is written, and setting(section, "url")
# names none, so saving unifi.password or any unifi_protect key restarted nothing that reads it.
CONSOLES = {"unifi": "unifi.", "unifi_protect": "unifi_protect."}


def setting(section: str, key: str) -> str:
    """The name of one console setting: setting("unifi_protect", "url") is "unifi_protect.url"."""
    return CONSOLES[section] + key


def configured_session(section: str = "unifi") -> UniFiSession:
    """A local session from one console's settings: `unifi` (Network) or `unifi_protect`."""
    from liberational import config as cfg

    ca_file = str(cfg.get(setting(section, "ca_file")))
    verify: bool | str = (
        str(cfg.path(setting(section, "ca_file"))) if ca_file else bool(cfg.get(setting(section, "verify_tls")))
    )
    return UniFiSession(
        str(cfg.get(setting(section, "url"))),
        username=str(cfg.get(setting(section, "username"))),
        password=str(cfg.get(setting(section, "password"))),
        api_key=str(cfg.get(setting(section, "api_key"))),
        unifi_os=bool(cfg.get(setting(section, "unifi_os"))),
        verify_tls=verify,
        timeout_s=float(cfg.get(setting(section, "timeout_seconds"))),
    )


def protect_section() -> str | None:
    """The settings section for the Protect console: its own when set, else the shared `[unifi]` console."""
    from liberational import config as cfg

    if cfg.get("unifi_protect.url"):
        return "unifi_protect"
    return "unifi" if cfg.get("unifi.url") else None
