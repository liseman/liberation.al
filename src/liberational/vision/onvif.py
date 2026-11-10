"""Bounded, explicit-action ONVIF discovery; never starts a media connection or changes a device.

Protocol references: https://www.onvif.org/specs/core/ONVIF-Core-Specification.pdf
and https://www.onvif.org/ver10/media/wsdl/media.wsdl . One WS-Discovery probe is
best effort, not an authoritative census. The same probe also goes by unicast to up to
256 listed hosts in one burst (cameras on Wi-Fi often miss multicast); replies from either path are
keyed by sender address, so a camera answering both is listed once. No TCP port is scanned. Media1 is supported; Media2-only devices
remain partial results. WSSE PasswordDigest requires a reasonably synchronized clock.
HTTP Digest follows https://www.rfc-editor.org/rfc/rfc7616.html with one challenge
retry, qop=auth, and MD5 or SHA-256; unsupported challenges remain partial.
The deadline is checked between raw reads; an in-flight socket read has at most
a two-second timeout. No redirects, proxy authentication, or cached challenges.
"""

import base64
import hashlib
import ipaddress
import math
import os
import re
import socket
import time
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast
from urllib.parse import unquote, urlsplit
from xml.sax.saxutils import escape

import requests

from liberational.unifi import UniFiError, local_origin

SOAP = "http://www.w3.org/2003/05/soap-envelope"
DEVICE = "http://www.onvif.org/ver10/device/wsdl"
MEDIA = "http://www.onvif.org/ver10/media/wsdl"
SCHEMA = "http://www.onvif.org/ver10/schema"
DISCOVERY = "http://schemas.xmlsoap.org/ws/2005/04/discovery"
ADDRESSING = "http://schemas.xmlsoap.org/ws/2004/08/addressing"
MAX_BYTES = 262144
MAX_UNICAST_HOSTS = 256


def probe_host(value: object) -> str | None:
    """A normalized literal private IPv4 address for a unicast probe, or None for anything else."""
    if not isinstance(value, str) or len(value) > 64:
        return None
    try:
        address = ipaddress.ip_address(value.strip())
        if address.version != 4:
            return None
        local_origin(f"https://{address}")
    except ValueError, UniFiError:
        return None
    return str(address)


@dataclass(frozen=True)
class OnvifStream:
    quality: str
    url: str = field(repr=False)


@dataclass(frozen=True)
class OnvifCamera:
    id: str
    ip: str
    name: str | None = None
    model: str | None = None
    streams: tuple[OnvifStream, ...] = field(default=(), repr=False)
    requires_credentials: bool = False


@dataclass(frozen=True)
class OnvifDiscovery:
    cameras: tuple[OnvifCamera, ...]
    complete: bool
    warnings: tuple[str, ...] = ()


class _Failure(Exception):
    pass


class _Credentials(_Failure):
    pass


def _xml(data: bytes) -> ET.Element:
    if len(data) > MAX_BYTES or b"\x00" in data or b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise _Failure("invalid XML")
    parser = ET.XMLPullParser(events=("start", "end"))
    depth = nodes = 0
    root: ET.Element | None = None
    try:
        for offset in range(0, len(data), 1024):
            parser.feed(data[offset : offset + 1024])
            for event, element in cast(Iterable[tuple[str, ET.Element]], parser.read_events()):
                if event == "start":
                    depth += 1
                    nodes += 1
                    if root is None:
                        root = element
                    if depth > 32 or nodes > 4096:
                        raise _Failure("XML bounds exceeded")
                else:
                    depth -= 1
        parser.close()
    except ET.ParseError:
        raise _Failure("invalid XML") from None
    if root is None:
        raise _Failure("empty XML")
    return root


def _text(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    value = value.strip()
    if len(value.encode()) > 512:
        raise _Failure("metadata bound exceeded")
    return value


def _url(value: str, *, sender: str | None = None, stream: bool = False) -> str:
    if len(value.encode()) > 2048 or any(ord(c) < 33 for c in value):
        raise _Failure("invalid endpoint")
    try:
        parts = urlsplit(value)
        if parts.scheme not in (("rtsp", "rtsps") if stream else ("http", "https")):
            raise ValueError
        if parts.fragment or not parts.hostname or (not stream and (parts.username or parts.password)):
            raise ValueError
        host = ipaddress.ip_address(parts.hostname)
        bracketed = f"[{host}]" if host.version == 6 else str(host)
        local_origin("https://" + bracketed)
        if sender is not None and host != ipaddress.ip_address(sender):
            raise ValueError
        if parts.port is not None and not 1 <= parts.port <= 65535:
            raise ValueError
    except ValueError, UniFiError:
        raise _Failure("invalid endpoint") from None
    return value


class OnvifDiscoverySource:
    """One multicast probe plus optional unicast probes, then bounded read-only SOAP calls, only on user action."""

    def __init__(
        self,
        username: str = "",
        password: str = "",
        *,
        timeout_s: float = 10,
        max_cameras: int = 32,
        max_profiles: int = 4,
        session: requests.Session | None = None,
        socket_factory: Callable[..., Any] = socket.socket,
        clock: Callable[[], float] = time.monotonic,
        hosts: Sequence[str] = (),
    ) -> None:
        if not math.isfinite(timeout_s) or not 0 < timeout_s <= 120:
            raise ValueError("invalid ONVIF timeout")
        if not 1 <= max_cameras <= 128 or not 1 <= max_profiles <= 16:
            raise ValueError("invalid ONVIF bounds")
        if len(username.encode()) > 512 or len(password.encode()) > 512:
            raise ValueError("invalid ONVIF credentials")
        normalized = [probe_host(host) for host in hosts]
        if None in normalized or len(normalized) > MAX_UNICAST_HOSTS:
            raise ValueError("ONVIF unicast hosts must be at most 256 literal private IPv4 addresses")
        self.hosts = tuple(dict.fromkeys(host for host in normalized if host is not None))
        self._username, self._password = username, password
        self.timeout_s, self.max_cameras, self.max_profiles = timeout_s, max_cameras, max_profiles
        self._session = session if session is not None else requests.Session()
        self._session.trust_env = False
        self._socket_factory, self._clock = socket_factory, clock

    def _remaining(self, deadline: float) -> float:
        remaining = deadline - self._clock()
        if remaining <= 0:
            raise _Failure("deadline exceeded")
        return remaining

    def _security(self) -> str:
        if not self._username:
            return ""
        nonce = os.urandom(20)
        created = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
        digest = base64.b64encode(hashlib.sha1(nonce + created.encode() + self._password.encode()).digest()).decode()
        ns = "http://docs.oasis-open.org/wss/2004/01/"
        return (
            f'<wsse:Security xmlns:wsse="{ns}oasis-200401-wss-wssecurity-secext-1.0.xsd" '
            f'xmlns:wsu="{ns}oasis-200401-wss-wssecurity-utility-1.0.xsd"><wsse:UsernameToken>'
            f"<wsse:Username>{escape(self._username)}</wsse:Username><wsse:Password "
            f'Type="{ns}oasis-200401-wss-username-token-profile-1.0#PasswordDigest">{digest}</wsse:Password>'
            f'<wsse:Nonce EncodingType="{ns}oasis-200401-wss-soap-message-security-1.0#Base64Binary">'
            f"{base64.b64encode(nonce).decode()}</wsse:Nonce><wsu:Created>{created}</wsu:Created>"
            "</wsse:UsernameToken></wsse:Security>"
        )

    def _digest(self, endpoint: str, challenge: str) -> str:
        """One RFC 7616 qop=auth challenge, without requests' automatic retry hooks."""
        if len(challenge.encode()) > 4096 or not challenge.lower().startswith("digest "):
            raise _Credentials
        pattern = re.compile(
            r'\s*([A-Za-z][A-Za-z0-9_-]*)\s*=\s*(?:"((?:[^"\\\r\n]|\\[\x20-\x7e])*)"|([A-Za-z0-9._/-]+))\s*(?:,|$)'
        )
        fields: dict[str, str] = {}
        rest = challenge[7:]
        while rest:
            match = pattern.match(rest)
            if match is None:
                raise _Credentials
            key = match[1].lower()
            value = re.sub(r"\\(.)", r"\1", match[2]) if match[2] is not None else match[3]
            if key in fields or len(value.encode()) > 1024 or any(ord(c) < 32 or ord(c) == 127 for c in value):
                raise _Credentials
            fields[key] = value
            rest = rest[match.end() :]
        if not fields.get("nonce") or "realm" not in fields or fields.get("userhash", "false").lower() != "false":
            raise _Credentials
        algorithm = fields.get("algorithm", "MD5").upper()
        if algorithm not in ("MD5", "SHA-256") or "auth" not in {
            v.strip() for v in fields.get("qop", "").lower().split(",")
        }:
            # qop=auth is required; no legacy no-qop, auth-int, or *-sess guesses.
            raise _Credentials
        charset = fields.get("charset", "ISO-8859-1").upper()
        if charset not in ("ISO-8859-1", "UTF-8"):
            raise _Credentials
        encoding = "utf-8" if charset == "UTF-8" else "latin-1"
        uri = requests.Request("POST", endpoint).prepare().path_url
        cnonce = os.urandom(16).hex()

        def hashed(value: str) -> str:
            try:
                return hashlib.new("md5" if algorithm == "MD5" else "sha256", value.encode(encoding)).hexdigest()
            except UnicodeError:
                raise _Credentials from None

        ha1 = hashed(f"{self._username}:{fields['realm']}:{self._password}")
        ha2 = hashed(f"POST:{uri}")
        digest = hashed(f"{ha1}:{fields['nonce']}:00000001:{cnonce}:auth:{ha2}")

        def quoted(value: str) -> str:
            if any(ord(c) < 32 or ord(c) == 127 for c in value):
                raise _Credentials
            return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'

        header = (
            f"Digest username={quoted(self._username)}, realm={quoted(fields['realm'])}, "
            f"nonce={quoted(fields['nonce'])}, uri={quoted(uri)}, algorithm={algorithm}, "
            f"response={quoted(digest)}, qop=auth, nc=00000001, cnonce={quoted(cnonce)}"
        )
        if "opaque" in fields:
            header += ", opaque=" + quoted(fields["opaque"])
        if len(header.encode()) > 4096:
            raise _Credentials
        return header

    def _soap(self, endpoint: str, namespace: str, operation: str, body: str, deadline: float) -> ET.Element:
        data = (
            f'<s:Envelope xmlns:s="{SOAP}" xmlns:t="{namespace}" xmlns:tt="{SCHEMA}">'
            f"<s:Header>{self._security()}</s:Header><s:Body><t:{operation}>{body}"
            f"</t:{operation}></s:Body></s:Envelope>"
        ).encode()
        response: requests.Response | None = None
        try:
            headers = {
                "Content-Type": f'application/soap+xml; charset=utf-8; action="{namespace}/{operation}"',
                "Accept-Encoding": "identity",
            }
            for attempt in range(2):
                response = self._session.post(
                    endpoint,
                    data=data,
                    headers=dict(headers),
                    timeout=min(2.0, self._remaining(deadline)),
                    allow_redirects=False,
                    stream=True,
                )
                if response.status_code == 403:
                    raise _Credentials
                if response.status_code != 401:
                    break
                if attempt or not self._username:
                    raise _Credentials
                authorization = self._digest(endpoint, response.headers.get("WWW-Authenticate", ""))
                # Never consume the untrusted challenge body, and never retry at a Location URL.
                response.close()
                response = None
                headers["Authorization"] = authorization
            assert response is not None
            if (
                300 <= response.status_code < 400
                or response.headers.get("Content-Encoding", "identity").lower() != "identity"
            ):
                raise _Failure
            chunks = bytearray()
            while True:
                self._remaining(deadline)
                chunk = response.raw.read1(min(16384, MAX_BYTES + 1 - len(chunks)), decode_content=False)
                if not chunk:
                    break
                chunks.extend(chunk)
                if len(chunks) > MAX_BYTES:
                    raise _Failure
            root = _xml(bytes(chunks))
            fault = root.find(f".//{{{SOAP}}}Fault")
            if fault is not None:
                if any("NotAuthorized" in (e.text or "") for e in fault.iter()):
                    raise _Credentials
                raise _Failure
            if response.status_code >= 400:
                raise _Failure
            result = root.find(f".//{{{namespace}}}{operation}Response")
            if result is None:
                raise _Failure
            return result
        except _Failure:
            raise
        except Exception:
            raise _Failure("ONVIF request failed") from None
        finally:
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass

    def _camera(
        self, identity: str, endpoint: str, ip: str, name: str | None, hardware: str | None, deadline: float
    ) -> tuple[OnvifCamera, bool]:
        # The discovery hardware scope names the model when GetDeviceInformation needs credentials.
        model = hardware
        streams: list[OnvifStream] = []
        failed = credentials = False
        try:
            info = self._soap(endpoint, DEVICE, "GetDeviceInformation", "", deadline)
            model = _text(info.findtext(f"{{{DEVICE}}}Model")) or hardware
        except _Credentials:
            credentials = failed = True
        except _Failure:
            failed = True
        try:
            caps = self._soap(endpoint, DEVICE, "GetCapabilities", "<t:Category>Media</t:Category>", deadline)
            media = caps.findtext(f".//{{{SCHEMA}}}Media/{{{SCHEMA}}}XAddr")
            if not media:
                raise _Failure
            media = _url(media, sender=ip)
            profiles = self._soap(media, MEDIA, "GetProfiles", "", deadline).findall(f"{{{MEDIA}}}Profiles")
            if len(profiles) > self.max_profiles:
                failed = True
            for profile in profiles[: self.max_profiles]:
                token = profile.get("token")
                if not token or len(token.encode()) > 512:
                    failed = True
                    continue
                quality = _text(profile.findtext(f"{{{SCHEMA}}}Name")) or "profile"
                try:
                    result = self._soap(
                        media,
                        MEDIA,
                        "GetStreamUri",
                        "<t:StreamSetup><tt:Stream>RTP-Unicast</tt:Stream>"
                        "<tt:Transport><tt:Protocol>RTSP</tt:Protocol></tt:Transport></t:StreamSetup>"
                        f"<t:ProfileToken>{escape(token)}</t:ProfileToken>",
                        deadline,
                    )
                    uri = result.findtext(f".//{{{SCHEMA}}}Uri")
                    if not uri:
                        raise _Failure
                    streams.append(OnvifStream(quality, _url(uri, sender=ip, stream=True)))
                except _Credentials:
                    credentials = failed = True
                except _Failure:
                    failed = True
        except _Credentials:
            credentials = failed = True
        except _Failure:
            failed = True
        return OnvifCamera(identity, ip, name, model, tuple(streams), credentials), failed

    def discover(self) -> OnvifDiscovery:
        cameras: list[OnvifCamera] = []
        warnings = ["ONVIF discovery is best effort; nonresponding devices remain unknown."]
        deadline = self._clock() + self.timeout_s
        message_id = "urn:uuid:" + str(uuid.uuid4())
        probe = (
            f'<s:Envelope xmlns:s="{SOAP}" xmlns:a="{ADDRESSING}" xmlns:d="{DISCOVERY}" '
            'xmlns:dn="http://www.onvif.org/ver10/network/wsdl"><s:Header>'
            f"<a:MessageID>{message_id}</a:MessageID><a:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</a:To>"
            f"<a:Action>{DISCOVERY}/Probe</a:Action></s:Header><s:Body><d:Probe>"
            "<d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></s:Body></s:Envelope>"
        ).encode()
        sock: Any = None
        endpoints: dict[str, tuple[str, str, str | None, str | None]] = {}
        try:
            sock = self._socket_factory(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
            sock.bind(("", 0))
            sock.settimeout(min(1.0, self._remaining(deadline)))
            sent = unsent = 0
            # One burst: multicast, then each unicast host; a send failure (no multicast route, or an unreachable
            # host) skips only that probe. Replies to all of them share the receive window below.
            for target in ("239.255.255.250", *self.hosts):
                try:
                    sock.sendto(probe, (target, 3702))
                    sent += 1
                except OSError:
                    unsent += 1
            if unsent:
                warnings.append(f"{unsent} ONVIF discovery probes could not be sent.")
            if not sent:
                raise _Failure("no probe sent")
            probe_deadline = min(deadline, self._clock() + min(2.0, self.timeout_s / 3))
            for _ in range(256 + len(self.hosts)):
                sock.settimeout(self._remaining(probe_deadline))
                try:
                    data, sender = sock.recvfrom(65536)
                except TimeoutError:
                    break
                try:
                    root = _xml(data)
                    if root.findtext(f".//{{{ADDRESSING}}}RelatesTo") != message_id:
                        continue
                    for match in root.findall(f".//{{{DISCOVERY}}}ProbeMatch")[: self.max_cameras]:
                        endpoint = next(
                            (
                                _url(v, sender=sender[0])
                                for v in (match.findtext(f"{{{DISCOVERY}}}XAddrs") or "").split()[:8]
                            ),
                            None,
                        )
                        if endpoint is None:
                            continue
                        identity = (
                            _text(match.findtext(f"{{{ADDRESSING}}}EndpointReference/{{{ADDRESSING}}}Address"))
                            or f"onvif:{sender[0]}"
                        )
                        scopes = (match.findtext(f"{{{DISCOVERY}}}Scopes") or "").split()[:32]
                        name, hardware = (
                            next(
                                (
                                    _text(unquote(v.split(f"/{kind}/", 1)[1]))
                                    for v in scopes
                                    if v.startswith(f"onvif://www.onvif.org/{kind}/")
                                ),
                                None,
                            )
                            for kind in ("name", "hardware")
                        )
                        if sender[0] not in endpoints and len(endpoints) >= self.max_cameras:
                            raise _Failure
                        endpoints[sender[0]] = identity, endpoint, name, hardware
                except _Failure, ValueError:
                    if len(warnings) < 16:
                        warnings.append("An ONVIF discovery response was rejected or exceeded a bound.")
            else:
                warnings.append("ONVIF datagram bound reached.")
        except Exception:
            warnings.append("ONVIF discovery was interrupted or unavailable.")
        finally:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass
        for ip, (identity, endpoint, name, hardware) in endpoints.items():
            camera, failed = self._camera(identity, endpoint, ip, name, hardware, deadline)
            cameras.append(camera)
            if failed and len(warnings) < 16:
                warnings.append("An ONVIF camera returned incomplete metadata or streams.")
        return OnvifDiscovery(tuple(cameras), False, tuple(warnings))
