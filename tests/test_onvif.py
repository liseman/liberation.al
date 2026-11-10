import base64
import hashlib
import io
import socket
import xml.etree.ElementTree as ET
from unittest.mock import create_autospec

import pytest
import requests

from liberational.vision.onvif import (
    ADDRESSING,
    DEVICE,
    DISCOVERY,
    MEDIA,
    SCHEMA,
    SOAP,
    OnvifDiscoverySource,
)


class Raw(io.BytesIO):
    def read1(self, size=-1, decode_content=False):
        return super().read1(size)


def response(body, status=200):
    result = requests.Response()
    result.status_code = status
    result.raw = Raw(body)
    return result


def envelope(body):
    return (
        f'<s:Envelope xmlns:s="{SOAP}" xmlns:d="{DISCOVERY}" xmlns:a="{ADDRESSING}" '
        f'xmlns:tds="{DEVICE}" xmlns:trt="{MEDIA}" xmlns:tt="{SCHEMA}">'
        f"<s:Body>{body}</s:Body></s:Envelope>"
    ).encode()


class DiscoverySocket:
    def __init__(
        self,
        *,
        endpoint="http://192.168.40.8/onvif/device_service",
        sender="192.168.40.8",
        extra=None,
        replies=1,
        scopes="onvif://www.onvif.org/name/Workshop%20camera",
        unreachable=(),
    ):
        self.endpoint, self.sender, self.extra = endpoint, sender, extra
        self.replies, self.scopes, self.unreachable = replies, scopes, unreachable
        self.sent = []
        self.options = []
        self.closed = False
        self.calls = 0

    def setsockopt(self, *args):
        self.options.append(args)

    def bind(self, address):
        self.bound = address

    def settimeout(self, value):
        assert 0 < value <= 10

    def sendto(self, data, address):
        if address[0] in self.unreachable:
            raise OSError("synthetic unreachable")
        self.sent.append((data, address))
        self.identity = ET.fromstring(data).findtext(f".//{{{ADDRESSING}}}MessageID")

    def recvfrom(self, size):
        self.calls += 1
        if self.calls > self.replies:
            raise TimeoutError
        if self.extra is not None:
            return self.extra, (self.sender, 3702)
        body = (
            f"<d:ProbeMatches><d:ProbeMatch><a:EndpointReference><a:Address>urn:uuid:camera-test</a:Address>"
            f"</a:EndpointReference><d:XAddrs>{self.endpoint}</d:XAddrs>"
            f"<d:Scopes>{self.scopes}</d:Scopes></d:ProbeMatch></d:ProbeMatches>"
        )
        data = envelope(body).replace(
            b"<s:Body>", f"<s:Header><a:RelatesTo>{self.identity}</a:RelatesTo></s:Header><s:Body>".encode()
        )
        return data, (self.sender, 3702)

    def close(self):
        self.closed = True


def operation(data):
    body = ET.fromstring(data).find(f"{{{SOAP}}}Body")
    assert body is not None
    return next(iter(body))


class CameraHTTP:
    def __init__(self, *, streams=None, status=200, profiles=None):
        self.calls = []
        self.status = status
        self.streams = streams or [
            "rtsp://viewer:synthetic-secret@192.168.40.8:554/h264Preview_01_main",
            "rtsp://192.168.40.8:554/h264Preview_01_sub",
        ]
        self.profiles = profiles or ["main", "sub"]

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        op = operation(kwargs["data"])
        name = op.tag.rsplit("}", 1)[1]
        if self.status != 200:
            return response(b"", self.status)
        bodies = {
            "GetDeviceInformation": "<tds:GetDeviceInformationResponse><tds:Manufacturer>Fixture vendor</tds:Manufacturer><tds:Model>RLC-test</tds:Model></tds:GetDeviceInformationResponse>",
            "GetCapabilities": "<tds:GetCapabilitiesResponse><tds:Capabilities><tt:Media><tt:XAddr>http://192.168.40.8/onvif/media_service</tt:XAddr></tt:Media></tds:Capabilities></tds:GetCapabilitiesResponse>",
            "GetProfiles": "<trt:GetProfilesResponse>"
            + "".join(f'<trt:Profiles token="{p}"><tt:Name>{p}</tt:Name></trt:Profiles>' for p in self.profiles)
            + "</trt:GetProfilesResponse>",
        }
        if name == "GetStreamUri":
            token = op.findtext(f"{{{MEDIA}}}ProfileToken")
            assert token is not None
            uri = self.streams[self.profiles.index(token)]
            return response(
                envelope(
                    f"<trt:GetStreamUriResponse><trt:MediaUri><tt:Uri>{uri}</tt:Uri></trt:MediaUri></trt:GetStreamUriResponse>"
                )
            )
        return response(envelope(bodies[name]))


def source(sock, http=None, **kwargs):
    factory = create_autospec(socket.socket, side_effect=lambda *args: sock)
    session = create_autospec(requests.Session, instance=True)
    fake = http or CameraHTTP()
    session.post.side_effect = fake.post
    return OnvifDiscoverySource(session=session, socket_factory=factory, **kwargs), fake, session


def test_reolink_style_profiles_preserve_private_uris_without_opening_media():
    sock = DiscoverySocket()
    discoverer, http, session = source(sock)
    result = discoverer.discover()
    assert len(result.cameras) == 1
    camera = result.cameras[0]
    assert (camera.id, camera.ip, camera.name, camera.model) == (
        "urn:uuid:camera-test",
        "192.168.40.8",
        "Workshop camera",
        "RLC-test",
    )
    assert [s.quality for s in camera.streams] == ["main", "sub"]
    assert camera.streams[0].url == "rtsp://viewer:synthetic-secret@192.168.40.8:554/h264Preview_01_main"
    assert "synthetic-secret" not in repr(result)
    assert "synthetic-secret" not in repr(camera.streams[0])
    assert not camera.requires_credentials
    assert not result.complete
    assert len(sock.sent) == 1
    assert sock.sent[0][1] == ("239.255.255.250", 3702)
    assert (socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1) in sock.options
    assert sock.closed
    assert len(http.calls) == 5
    assert session.trust_env is False
    assert all(call[0].startswith("http://192.168.40.8/onvif/") for call in http.calls)
    assert all(call[1]["allow_redirects"] is False for call in http.calls)


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://192.168.40.9/onvif",
        "http://example.invalid/onvif",
        "http://8.8.8.8/onvif",
        "http://0.0.0.0/onvif",
        "http://239.1.1.1/onvif",
        "http://user:pass@192.168.40.8/onvif",
    ],
)
def test_discovery_never_follows_untrusted_xaddrs(endpoint):
    discoverer, http, _ = source(DiscoverySocket(endpoint=endpoint))
    result = discoverer.discover()
    assert not result.cameras
    assert not result.complete
    assert http.calls == []


@pytest.mark.parametrize(
    "uri", ["rtsp://8.8.8.8/live", "rtsp://example.invalid/live", "rtsp://239.1.1.1/live", "http://192.168.40.8/live"]
)
def test_stream_uri_must_be_local_literal_and_rtsp(uri):
    discoverer, _, _ = source(DiscoverySocket(), CameraHTTP(streams=[uri, uri]))
    result = discoverer.discover()
    assert len(result.cameras) == 1
    assert result.cameras[0].streams == ()
    assert len(result.warnings) > 1


@pytest.mark.parametrize("status", [301, 302, 500])
def test_http_failures_leave_discovered_camera_partial(status):
    discoverer, _, _ = source(DiscoverySocket(), CameraHTTP(status=status))
    result = discoverer.discover()
    assert len(result.cameras) == 1
    assert result.cameras[0].model is None
    assert result.cameras[0].streams == ()
    assert not result.complete


@pytest.mark.parametrize("status", [401, 403])
def test_authentication_failure_is_explicit_and_does_not_bruteforce(status):
    discoverer, http, _ = source(
        DiscoverySocket(), CameraHTTP(status=status), username="viewer", password="secret-fixture"
    )
    result = discoverer.discover()
    assert result.cameras[0].requires_credentials
    assert len(http.calls) == 2
    assert "secret-fixture" not in repr(result)
    assert b"secret-fixture" not in http.calls[0][1]["data"]


def test_wsse_digest_is_nonce_created_and_password_not_plaintext():
    discoverer, http, _ = source(DiscoverySocket(), username="test-viewer", password="test-password")
    discoverer.discover()
    root = ET.fromstring(http.calls[0][1]["data"])
    values = {e.tag.rsplit("}", 1)[-1]: e.text or "" for e in root.iter()}
    expected = hashlib.sha1(base64.b64decode(values["Nonce"]) + values["Created"].encode() + b"test-password").digest()
    assert values["Password"] == base64.b64encode(expected).decode()
    assert values["Username"] == "test-viewer"


@pytest.mark.parametrize(
    "xml",
    [
        b'<!DOCTYPE x [<!ENTITY e "boom">]><x>&e;</x>',
        b"<a>" * 33 + b"</a>" * 33,
        b"<a>" + b"<b/>" * 4096 + b"</a>",
        b"\xff\xfe\x00x",
        b"<a>",
        b"x" * 262145,
    ],
)
def test_untrusted_discovery_xml_is_bounded_without_http(xml):
    discoverer, http, _ = source(DiscoverySocket(extra=xml))
    result = discoverer.discover()
    assert not result.cameras
    assert not result.complete
    assert not http.calls


def test_profiles_are_bounded_and_never_create_or_set_anything():
    discoverer, http, _ = source(DiscoverySocket(), max_profiles=1)
    result = discoverer.discover()
    assert len(result.cameras[0].streams) == 1
    operations = [operation(c[1]["data"]).tag.rsplit("}", 1)[1] for c in http.calls]
    assert operations == ["GetDeviceInformation", "GetCapabilities", "GetProfiles", "GetStreamUri"]
    assert len(result.warnings) > 1


def test_socket_failure_is_unknown_not_certified_empty():
    sock = DiscoverySocket()
    sock.recvfrom = create_autospec(sock.recvfrom, side_effect=OSError("private detail"))
    discoverer, http, _ = source(sock)
    result = discoverer.discover()
    assert result.cameras == ()
    assert not result.complete
    assert "private detail" not in repr(result)
    assert not http.calls
    assert sock.closed


def test_deadline_stops_http_after_discovery_without_waiting():
    sock = DiscoverySocket()
    ticks = iter([0, 0, 0, 0, 0, 11, 11, 11, 11])
    discoverer, http, _ = source(sock, clock=lambda: next(ticks))
    result = discoverer.discover()
    assert not result.complete
    assert len(http.calls) == 0


@pytest.mark.parametrize("payload", [b"<broken", b"x" * 262145, b"<!DOCTYPE x><x/>", b"<a>" * 33 + b"</a>" * 33])
def test_bad_soap_xml_preserves_discovered_device_and_does_not_leak_body(payload):
    discoverer, _, session = source(DiscoverySocket())
    session.post.side_effect = lambda *args, **kwargs: response(payload)
    result = discoverer.discover()
    assert len(result.cameras) == 1
    assert result.cameras[0].streams == ()
    assert not result.complete
    assert "broken" not in repr(result)


def test_soap_authorization_fault_on_http500_marks_missing_credentials():
    discoverer, _, session = source(DiscoverySocket())
    fault = envelope("<s:Fault><s:Code><s:Subcode><s:Value>ter:NotAuthorized</s:Value></s:Subcode></s:Code></s:Fault>")
    session.post.side_effect = lambda *args, **kwargs: response(fault, 500)
    result = discoverer.discover()
    assert result.cameras[0].requires_credentials
    assert result.cameras[0].streams == ()


def test_media_capability_cannot_redirect_credentials_to_a_different_local_device():
    discoverer, http, session = source(DiscoverySocket(), username="fixture-user", password="fixture-secret")
    original = http.post

    def post(url, **kwargs):
        if operation(kwargs["data"]).tag.endswith("GetCapabilities"):
            return response(
                envelope(
                    "<tds:GetCapabilitiesResponse><tt:Media><tt:XAddr>http://192.168.40.9/onvif/media</tt:XAddr></tt:Media></tds:GetCapabilitiesResponse>"
                )
            )
        return original(url, **kwargs)

    session.post.side_effect = post
    result = discoverer.discover()
    assert result.cameras[0].model == "RLC-test"
    assert result.cameras[0].streams == ()
    assert all(call.args[0].startswith("http://192.168.40.8/") for call in session.post.call_args_list)


def test_unrelated_discovery_reply_is_ignored():
    data = envelope(
        "<d:ProbeMatches><d:ProbeMatch><d:XAddrs>http://192.168.40.8/onvif</d:XAddrs></d:ProbeMatch></d:ProbeMatches>"
    )
    discoverer, http, _ = source(DiscoverySocket(extra=data))
    result = discoverer.discover()
    assert not result.cameras
    assert not http.calls


def test_datagram_flood_stops_at_fixed_bound_without_real_waits():
    sock = DiscoverySocket(extra=b"<bad")
    sock.recvfrom = create_autospec(sock.recvfrom, return_value=(b"<bad", ("192.168.40.8", 3702)))
    discoverer, http, _ = source(sock, clock=lambda: 0.0)
    result = discoverer.discover()
    assert sock.recvfrom.call_count == 256
    assert len(result.warnings) <= 17
    assert not result.complete
    assert not http.calls


def challenged(header, *, status=401):
    result = response(b"untrusted challenge body", status)
    result.headers["WWW-Authenticate"] = header
    return result


@pytest.mark.parametrize(
    ("algorithm", "digest"),
    [
        ("MD5", "10410c3ea0032ff693eac9c7e369c126"),
        ("SHA-256", "e58eab029b95e33e787a2d206110675fadb62ff487f6896251df522f53831973"),
    ],
)
def test_digest_authorization_matches_fixed_post_fixture_and_reads_no_challenge_body(monkeypatch, algorithm, digest):
    monkeypatch.setattr("liberational.vision.onvif.os.urandom", lambda size: bytes(size))
    discoverer, http, session = source(DiscoverySocket(), username="viewer", password="fixture-password")
    calls = []
    raw = create_autospec(Raw, instance=True)
    raw.read1.side_effect = AssertionError("challenge body must never be consumed")

    def post(url, **kwargs):
        calls.append((url, kwargs))
        if "Authorization" not in kwargs["headers"]:
            result = challenged(
                f'Digest realm="fixture-camera", nonce="fixture-nonce", qop="auth-int, auth", algorithm={algorithm}, opaque="opaque-fixture", domain="http://192.168.40.99/"'
            )
            result.raw = raw
            return result
        return http.post(url, **kwargs)

    session.post.side_effect = post
    result = discoverer.discover()
    # RFC 7616 A1 = viewer:fixture-camera:fixture-password; A2 = POST:/onvif/device_service.
    # MD5 HA1 af945d7cfc3bdb3a54f1a07d76f76517; HA2 bb7214af296c80b7b6e7d4b2e47f4ae6.
    assert calls[1][1]["headers"]["Authorization"] == (
        'Digest username="viewer", realm="fixture-camera", nonce="fixture-nonce", '
        f'uri="/onvif/device_service", algorithm={algorithm}, response="{digest}", '
        'qop=auth, nc=00000001, cnonce="00000000000000000000000000000000", opaque="opaque-fixture"'
    )
    assert len(result.cameras[0].streams) == 2
    assert len(calls) == 10
    assert all(c[0].startswith("http://192.168.40.8/") for c in calls)
    raw.read1.assert_not_called()
    assert "fixture-password" not in repr(result)


def test_digest_second401_never_causes_third_attempt():
    discoverer, _, session = source(DiscoverySocket(), username="viewer", password="fixture-password")
    session.post.side_effect = lambda *args, **kwargs: challenged(
        'Digest realm="camera", nonce="n", qop="auth", stale=true'
    )
    result = discoverer.discover()
    assert result.cameras[0].requires_credentials
    # Each of the two read-only operations has one initial call and at most one retry.
    assert session.post.call_count == 4
    assert "Authorization" not in session.post.call_args_list[0].kwargs["headers"]
    assert "Authorization" in session.post.call_args_list[1].kwargs["headers"]


@pytest.mark.parametrize(
    "challenge",
    [
        'Basic realm="camera"',
        'Digest realm="camera", nonce="n", qop="auth-int"',
        'Digest realm="camera", nonce="n", algorithm=MD5-sess, qop="auth"',
        'Digest realm="camera", nonce="n", algorithm=SHA-512-256, qop="auth"',
        'Digest realm="camera", nonce="n"',
        'Digest realm="camera", nonce="n", qop="auth", nonce="duplicate"',
        'Digest realm="camera", nonce="n", qop="auth", userhash=true',
        'Digest realm="camera", nonce="n", qop="auth", charset=unknown',
        'Digest realm="camera", nonce="bad\r\nnonce", qop="auth"',
        'Digest realm="camera", nonce="' + "n" * 1025 + '", qop="auth"',
        'Digest realm="' + "r" * 4097 + '", nonce="n", qop="auth"',
        'Digest realm="camera", nonce="n", qop="auth", Basic realm="other"',
    ],
)
def test_hostile_or_unsupported_digest_challenge_does_not_retry(challenge):
    discoverer, _, session = source(DiscoverySocket(), username="viewer", password="fixture-password")
    session.post.side_effect = lambda *args, **kwargs: challenged(challenge)
    result = discoverer.discover()
    assert result.cameras[0].requires_credentials
    assert session.post.call_count == 2
    assert all("Authorization" not in call.kwargs["headers"] for call in session.post.call_args_list)
    assert "fixture-password" not in repr(result)


def test_digest_retry_never_follows_redirect_even_to_private_third_party():
    discoverer, _, session = source(DiscoverySocket(), username="viewer", password="fixture-password")

    def post(url, **kwargs):
        if "Authorization" not in kwargs["headers"]:
            return challenged('Digest realm="camera", nonce="n", qop="auth"')
        result = response(b"", 302)
        result.headers["Location"] = "http://192.168.40.99/credentials"
        return result

    session.post.side_effect = post
    result = discoverer.discover()
    assert result.cameras[0].streams == ()
    assert session.post.call_count == 4
    assert all(call.args[0] == "http://192.168.40.8/onvif/device_service" for call in session.post.call_args_list)
    assert all(call.kwargs["allow_redirects"] is False for call in session.post.call_args_list)


def test_digest_retry_obeys_original_operation_deadline():
    now = [0.0]
    discoverer, _, session = source(
        DiscoverySocket(), username="viewer", password="fixture-password", clock=lambda: now[0]
    )

    def post(*args, **kwargs):
        now[0] = 11.0
        return challenged('Digest realm="camera", nonce="n", qop="auth"')

    session.post.side_effect = post
    result = discoverer.discover()
    assert len(result.cameras) == 1
    assert result.cameras[0].streams == ()
    assert session.post.call_count == 1
    assert not result.complete


@pytest.mark.parametrize("uri", ["rtsp://192.168.40.9/live", "rtsps://[fd00::9]/live"])
def test_stream_uri_cannot_redirect_camera_credentials_to_a_different_local_host(uri):
    discoverer, http, _ = source(
        DiscoverySocket(), CameraHTTP(streams=[uri, uri]), username="fixture-viewer", password="fixture-password"
    )
    result = discoverer.discover()
    assert len(result.cameras) == 1
    assert result.cameras[0].ip == "192.168.40.8"
    assert result.cameras[0].streams == ()
    assert len(result.warnings) > 1
    assert all(call[0].startswith("http://192.168.40.8/") for call in http.calls)


def test_unicast_probes_go_in_one_burst_and_a_camera_answering_twice_is_listed_once():
    sock = DiscoverySocket(replies=2)
    discoverer, http, _ = source(sock, hosts=["192.168.40.8", "192.168.40.9", "192.168.40.8"])
    result = discoverer.discover()
    assert [address for _, address in sock.sent] == [
        ("239.255.255.250", 3702),
        ("192.168.40.8", 3702),
        ("192.168.40.9", 3702),
    ]
    assert len({data for data, _ in sock.sent}) == 1
    assert [camera.ip for camera in result.cameras] == ["192.168.40.8"]
    assert len(http.calls) == 5


@pytest.mark.parametrize(
    "unreachable",
    [pytest.param(("239.255.255.250",), id="no-multicast-route"), pytest.param(("192.168.40.9",), id="one-host")],
)
def test_a_probe_that_cannot_be_sent_skips_only_that_target(unreachable):
    sock = DiscoverySocket(unreachable=unreachable)
    discoverer, _, _ = source(sock, hosts=["192.168.40.8", "192.168.40.9"])
    result = discoverer.discover()
    assert [camera.ip for camera in result.cameras] == ["192.168.40.8"]
    assert "1 ONVIF discovery probes could not be sent." in result.warnings


@pytest.mark.parametrize(
    "hosts",
    [
        pytest.param(["8.8.8.8"], id="public"),
        pytest.param(["camera.local"], id="hostname"),
        pytest.param(["fd00::20"], id="ipv6"),
        pytest.param(["239.255.255.250"], id="multicast"),
        pytest.param([f"10.0.{i // 256}.{i % 256}" for i in range(257)], id="over-256"),
    ],
)
def test_unicast_hosts_must_be_a_bounded_list_of_private_ipv4_literals(hosts):
    with pytest.raises(ValueError, match="unicast hosts"):
        source(DiscoverySocket(), hosts=hosts)


def test_hardware_scope_names_the_model_when_the_camera_needs_credentials():
    sock = DiscoverySocket(scopes="onvif://www.onvif.org/name/IPC onvif://www.onvif.org/hardware/E1Pro")
    discoverer, _, _ = source(sock, http=CameraHTTP(status=401))
    camera = discoverer.discover().cameras[0]
    assert (camera.name, camera.model, camera.requires_credentials) == ("IPC", "E1Pro", True)
