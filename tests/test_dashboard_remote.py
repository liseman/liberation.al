"""Synthetic remote-preview and compressed-asset checks; no cameras or real state."""

import base64
import gzip
import http.client

import pytest

from liberational.dashboard import assets, server


@pytest.mark.parametrize(
    ("header", "accept"),
    [
        ("", False),
        ("br", False),
        ("gzip", True),
        ("GZip; q=0.4, br", True),
        ("gzip;q=0", False),
        ("gzip;q=0, *;q=1", False),
        ("*;q=0.5", True),
        ("gzip;q=bad", False),
        ("gzip;q=2", False),
        ("gzip;q=-1", False),
    ],
)
def test_gzip_negotiation(header, accept):
    assert assets.accepts_gzip(header) is accept


@pytest.fixture
def api(serve, monkeypatch):
    monkeypatch.setattr(server, "DASH_PASSWORD", "synthetic-password")
    host = serve(server.H).removeprefix("http://")

    def get(path, encoding="gzip", method="GET", auth=True):
        headers = {"Accept-Encoding": encoding, "Accept": "text/html", "X-Forwarded-For": "192.0.2.10"}
        if auth:
            headers["Authorization"] = "Basic " + base64.b64encode(b"user:synthetic-password").decode()
        c = http.client.HTTPConnection(host, timeout=5)
        c.request(method, path, headers=headers)
        r = c.getresponse()
        result = r.status, dict(r.getheaders()), r.read()
        c.close()
        return result

    return get


@pytest.fixture
def built(tmp_path, monkeypatch):
    """A frontend build in a temp static/app/, big enough to be worth compressing."""
    app = tmp_path / "static" / "app"
    (app / "assets").mkdir(parents=True)
    (app / ".vite").mkdir()
    (app / ".vite" / "manifest.json").write_text("{}")
    filler = "<!-- " + "padding " * 200 + "-->"
    (app / "index.html").write_text(
        f"<!doctype html><html><head><title>{{{{assistant}}}}</title></head><body>{filler}</body></html>"
    )
    (app / "assets" / "index-abc123.js").write_text("export const x = 1;\n" * 100)
    (app / "assets" / "index-abc123.css").write_text("body { margin: 0 }\n" * 100)
    for name, value in {
        "STATIC": app.parent,
        "APP": app,
        "MANIFEST": app / ".vite" / "manifest.json",
        "FRONTEND": tmp_path / "frontend",
    }.items():
        monkeypatch.setattr(assets, name, value)


@pytest.mark.parametrize(
    ("path", "cache"),
    [
        ("/", "no-store"),
        ("/review", "no-store"),
        ("/static/app/assets/index-abc123.js", assets.IMMUTABLE),
        ("/static/app/assets/index-abc123.css", assets.IMMUTABLE),
    ],
)
def test_gzip_body_head_and_auth_are_consistent(built, api, path, cache):
    code, h, body = api(path)
    assert code == 200 and h["Content-Encoding"] == "gzip"
    assert h["Vary"] == "Accept-Encoding" and h["Cache-Control"] == cache
    assert len(body) == int(h["Content-Length"])
    _, plain_h, plain = api(path, "gzip;q=0, *;q=1")
    assert "Content-Encoding" not in plain_h
    assert gzip.decompress(body) == plain
    code, head_h, head_body = api(path, method="HEAD")
    assert code == 200 and head_body == b""
    assert head_h["Content-Length"] == h["Content-Length"]
    assert api(path, auth=False)[0] == 401


def test_dynamic_api_data_not_compressed(api):
    code, headers, _ = api("/health")
    assert code == 200 and "Content-Encoding" not in headers


@pytest.mark.parametrize(
    ("host", "allowed"),
    [
        ("[::1]", True),
        ("[::1]:8765", True),
        ("localhost", True),
        ("localhost:8765", True),
        ("127.0.0.1", True),
        ("127.0.0.1:8765", True),
        ("attacker.example", False),
        ("attacker.example:8765", False),
        ("[::1", False),
        ("[::1]attacker.example", False),
        ("[::1]attacker.example:8765", False),
        ("[::1]:8765:1234", False),
        ("", False),
    ],
)
def test_controls_require_a_loopback_host(host, allowed):
    handler = object.__new__(server.H)
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = http.client.HTTPMessage()
    handler.headers["Host"] = host
    assert handler._can_control() is allowed


@pytest.mark.parametrize("host", ["localhost", "[::1]", "[::1]:8765"])
@pytest.mark.parametrize(
    ("client", "forwarded", "allowed"),
    [("127.0.0.1", "", True), ("::1", "", True), ("192.0.2.10", "", False), ("127.0.0.1", "192.0.2.10", False)],
)
def test_controls_still_require_a_direct_local_client(host, client, forwarded, allowed):
    handler = object.__new__(server.H)
    handler.client_address = (client, 12345)
    handler.headers = http.client.HTTPMessage()
    handler.headers["Host"] = host
    handler.headers["X-Forwarded-For"] = forwarded
    assert handler._can_control() is allowed


def test_controls_refuse_a_missing_host():
    handler = object.__new__(server.H)
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = http.client.HTTPMessage()
    assert not handler._can_control()


def test_real_dashboard_dispatches_history_delete_to_local_broker(serve, memory):
    import json

    from liberational.memory import ask, ask_history

    ident = ask_history.at(ask.STATE).start("Synthetic saved question")
    host = serve(server.H).removeprefix("http://")
    connection = http.client.HTTPConnection(host, timeout=5)
    try:
        connection.request(
            "DELETE", "/pack-ask/history/" + ident, body="{}", headers={"Content-Type": "application/json"}
        )
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read()) == {"deleted": 1}
        connection.request("GET", "/pack-ask/history/" + ident)
        response = connection.getresponse()
        assert response.status == 404
        response.read()
    finally:
        connection.close()
    assert ask_history.at(ask.STATE).get(ident) is None
