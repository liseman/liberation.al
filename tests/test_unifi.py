import io
import json
import time
import warnings
from typing import cast

import pytest
import requests
from urllib3.exceptions import InsecureRequestWarning

from liberational import config, unifi
from liberational.unifi import UniFiError, UniFiSession, configured_session, local_origin, protect_section


class Raw:
    def __init__(self, data):
        self.body = io.BytesIO(data)

    def read1(self, size, decode_content=False):
        return self.body.read(size)

    def close(self):
        self.body.close()


def response(data, status=200, **headers):
    r = requests.Response()
    r.status_code, r.raw = status, Raw(json.dumps(data).encode())
    r.headers.update(headers)
    return r


class FakeSession:
    def __init__(self, replies):
        self.replies, self.calls = iter(replies), []
        self.trust_env = True
        self.closed = False

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def close(self):
        self.closed = True


def session(replies, **kwargs):
    fake = FakeSession(replies)
    client = UniFiSession("https://192.168.50.1", session=cast(requests.Session, fake), **kwargs)
    return client, fake


@pytest.mark.parametrize(
    "url",
    [
        "http://192.168.1.1",
        "https://api.ui.com",
        "https://unifi.local",
        "https://8.8.8.8",
        "https://0.0.0.0",
        "https://224.0.0.1",
        "https://[::]",
        "https://[ff02::1]",
        "https://user:password@192.168.1.1",
        "https://192.168.1.1/path",
        "https://192.168.1.1?secret=x",
        "https://192.168.1.1#fragment",
    ],
)
def test_controller_origins_cannot_escape_local_https_without_url_credentials(url):
    with pytest.raises(UniFiError, match="local unicast"):
        local_origin(url)


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://192.168.50.1/", "https://192.168.50.1"),
        ("https://127.0.0.1:8443", "https://127.0.0.1:8443"),
        ("https://[fd00::1]:443", "https://[fd00::1]:443"),
        ("https://169.254.1.2", "https://169.254.1.2"),
    ],
)
def test_explicit_local_ip_origins_keep_management_ports(url, expected):
    assert local_origin(url) == expected


def test_viewer_cookie_login_and_client_get_use_only_fixed_local_routes():
    client, fake = session(
        [response({}), response({"meta": {"rc": "ok"}, "data": []})], username="viewer", password="sample-password"
    )
    assert client.legacy_clients() == {"meta": {"rc": "ok"}, "data": []}
    assert [(m, u) for m, u, _ in fake.calls] == [
        ("POST", "https://192.168.50.1/api/auth/login"),
        ("GET", "https://192.168.50.1/proxy/network/api/s/default/stat/sta"),
    ]
    assert fake.calls[0][2]["json"] == {"username": "viewer", "password": "sample-password"}
    assert fake.calls[1][2]["json"] is None
    assert fake.calls[0][2]["verify"] is True
    assert fake.calls[0][2]["allow_redirects"] is False
    assert not fake.trust_env
    assert "sample-password" not in repr(client)


def test_standalone_controller_does_not_receive_unifi_os_paths():
    client, fake = session(
        [response({"meta": {"rc": "ok"}}), response({"meta": {"rc": "ok"}, "data": []})],
        username="viewer",
        password="sample",
        unifi_os=False,
        verify_tls="/temporary/test-ca.pem",
    )
    client.legacy_clients("workshop")
    assert [(m, u) for m, u, _ in fake.calls] == [
        ("POST", "https://192.168.50.1/api/login"),
        ("GET", "https://192.168.50.1/api/s/workshop/stat/sta"),
    ]
    assert fake.calls[1][2]["verify"] == "/temporary/test-ca.pem"


def test_integration_key_needs_no_login_and_pagination_is_fixed():
    client, fake = session([response({"data": []})], api_key="sample-key", verify_tls=False)
    client.integration_clients("00000000-0000-0000-0000-000000000001", 200, 200)
    assert len(fake.calls) == 1
    method, url, options = fake.calls[0]
    assert method == "GET"
    assert (
        url
        == "https://192.168.50.1/proxy/network/integration/v1/sites/00000000-0000-0000-0000-000000000001/clients?offset=200&limit=200"
    )
    assert options["headers"]["X-API-KEY"] == "sample-key"
    assert options["verify"] is False


@pytest.mark.parametrize(
    "rejected", [response({}, 401), response({"meta": {"rc": "error", "msg": "api.err.LoginRequired"}, "data": []})]
)
def test_expired_session_reauthenticates_only_once(rejected):
    client, fake = session(
        [response({}), rejected, response({}), response({}, 401)], username="viewer", password="sample"
    )
    with pytest.raises(UniFiError, match="expired"):
        client.legacy_clients()
    assert [method for method, _, _ in fake.calls] == ["POST", "GET", "POST", "GET"]


@pytest.mark.parametrize(
    "reply,error",
    [
        (response({"echo": "sample-password"}, 302, Location="https://public.example"), "redirect"),
        (response({"echo": "sample-password"}, 403), "HTTP 403"),
        (requests.ConnectionError("request contains sample-password"), "connection failed"),
        (requests.Timeout("sample-password"), "timed out"),
        (response({}, **{"Content-Encoding": "gzip"}), "compressed"),
    ],
)
def test_transport_failures_never_include_credentials_or_response_text(reply, error):
    client, fake = session([reply], api_key="sample-password")
    with pytest.raises(UniFiError, match=error) as failure:
        client.integration_sites()
    assert "sample-password" not in str(failure.value)
    assert len(fake.calls) == 1


def test_body_limit_applies_without_content_length():
    client, _ = session([response({"data": "x" * 100})], api_key="sample", max_response_bytes=20)
    with pytest.raises(UniFiError, match="byte bounds"):
        client.integration_sites()


def test_expired_absolute_deadline_does_not_send_credentials():
    client, fake = session([], username="viewer", password="sample")
    with pytest.raises(UniFiError, match="deadline"):
        client.legacy_clients(deadline=time.monotonic() - 1)
    assert fake.calls == []


@pytest.mark.parametrize(
    "method,args",
    [
        ("legacy_clients", ("../private",)),
        ("integration_clients", ("not-a-uuid",)),
        ("protect_streams", ("camera?override=1",)),
    ],
)
def test_native_ids_cannot_inject_another_route(method, args):
    client, fake = session([], username="viewer", password="sample", api_key="sample")
    with pytest.raises(ValueError):
        getattr(client, method)(*args)
    assert fake.calls == []


def test_shared_session_reuses_local_login_and_only_reads_protect_routes():
    client, fake = session(
        [
            response({}),
            response({"meta": {"rc": "ok"}, "data": []}),
            response({"cameras": []}),
            response([]),
            response({"high": None}),
        ],
        username="viewer",
        password="sample",
        api_key="sample-key",
    )
    client.legacy_clients()
    client.protect_bootstrap()
    client.protect_cameras()
    client.protect_streams("000000000000000000000001")
    assert [(method, url) for method, url, _ in fake.calls] == [
        ("POST", "https://192.168.50.1/api/auth/login"),
        ("GET", "https://192.168.50.1/proxy/network/api/s/default/stat/sta"),
        ("GET", "https://192.168.50.1/proxy/protect/api/bootstrap"),
        ("GET", "https://192.168.50.1/proxy/protect/integration/v1/cameras"),
        ("GET", "https://192.168.50.1/proxy/protect/integration/v1/cameras/000000000000000000000001/rtsps-stream"),
    ]


@pytest.mark.parametrize(
    ("protect_url", "expected"),
    [
        pytest.param("https://192.168.50.224", "https://192.168.50.224", id="separate-nvr"),
        pytest.param("", "https://192.168.50.1", id="single-console-fallback"),
    ],
)
def test_protect_session_uses_its_own_console_or_falls_back_to_network(protect_url, expected):
    values = {
        "unifi": dict(url="https://192.168.50.1", username="router-viewer"),
        "unifi_protect": dict(url=protect_url, username="nvr-viewer", verify_tls=False),
    }
    with config.override(values):
        section = protect_section()
        assert section is not None
        client = configured_session(section)
    assert client.origin == expected
    assert client._username == ("nvr-viewer" if protect_url else "router-viewer")
    assert client.verify_tls is (not protect_url)


def test_protect_is_unconfigured_without_either_console():
    assert protect_section() is None


def test_protect_console_keeps_literal_private_ip_validation():
    with config.override({"unifi_protect": dict(url="https://unifi.local")}), pytest.raises(UniFiError):
        configured_session(protect_section() or "")


INSECURE = "Unverified HTTPS request is being made to host '{}'. Adding certificate verification is strongly advised."


def test_unverified_tls_warning_is_silenced_for_that_console_only_and_logged_once(monkeypatch, caplog):
    monkeypatch.setattr(unifi, "_unverified_hosts", set())
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        UniFiSession("https://192.168.50.224", verify_tls=False)
        UniFiSession("https://192.168.50.224", verify_tls=False)
        UniFiSession("https://192.168.50.1")
        for host in ("192.168.50.224", "192.168.50.2", "192.168.50.1"):
            warnings.warn(INSECURE.format(host), InsecureRequestWarning, stacklevel=1)
    assert [str(w.message).split("'")[1] for w in seen] == ["192.168.50.2", "192.168.50.1"]
    assert [r.getMessage() for r in caplog.records if "verification is off" in r.getMessage()] == [
        "TLS certificate verification is off for the UniFi console at https://192.168.50.224"
    ]
