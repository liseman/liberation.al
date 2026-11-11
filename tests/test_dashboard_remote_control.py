"""dashboard.remote_control: a password-checked browser on another device of the tailnet may change what this machine
may (services, Settings, devices, privacy pauses); never over Funnel, never without a password, never from another
site."""

import base64
import http.client
import json
import urllib.parse

import pytest

from liberational.dashboard import privacy, server, surroundings


@pytest.fixture
def ask(serve_dashboard, monkeypatch):
    monkeypatch.setattr(privacy, "view", lambda control: {"ok": True, "control": control, "ready": True, "devices": []})
    monkeypatch.setattr(surroundings, "view", lambda control: {"ok": True, "control": control})
    base = serve_dashboard()
    netloc = urllib.parse.urlsplit(base).netloc

    def go(method, path, *, user="luke@example.com", funnel=False, auth=True, origin="https", body=None):
        headers = {"Content-Type": "application/json", "Host": netloc, "X-Forwarded-For": "100.64.0.9"}
        if user:
            headers["Tailscale-User-Login"] = user
        if funnel:
            headers["Tailscale-Funnel-Request"] = "?1"
        if auth:
            headers["Authorization"] = "Bas" + "ic " + base64.b64encode(b"luca:pw").decode()
        if origin == "evil":
            headers["Origin"] = "https://evil.example"
        elif origin:
            headers["Origin"] = f"{origin}://{netloc}"
        conn = http.client.HTTPConnection(netloc, timeout=120)
        conn.request(method, path, body=json.dumps(body) if body is not None else None, headers=headers)
        resp = conn.getresponse()
        code, data = resp.status, resp.read()
        conn.close()
        return code, (json.loads(data) if data.startswith(b"{") else None)

    return go


def control(ask, **kw):
    code, view = ask("GET", "/devices/privacy", **kw)
    assert code == 200
    return view["control"]


@pytest.mark.parametrize("on", [False, True])
def test_a_tailnet_user_controls_only_with_it_on(ask, monkeypatch, on):
    monkeypatch.setattr(server, "DASH_PASSWORD", "pw")
    monkeypatch.setattr(server, "remote_control", lambda: on)
    assert control(ask) is on


def test_never_over_funnel_or_without_a_tailnet_user(ask, monkeypatch):
    monkeypatch.setattr(server, "DASH_PASSWORD", "pw")
    monkeypatch.setattr(server, "remote_control", lambda: True)
    assert control(ask, funnel=True) is False
    assert control(ask, user="") is False


def test_never_without_a_password(ask, monkeypatch):
    monkeypatch.setattr(server, "DASH_PASSWORD", "")
    monkeypatch.setattr(server, "remote_control", lambda: True)
    assert control(ask, auth=False) is False


def test_the_password_gate_comes_first(ask, monkeypatch):
    monkeypatch.setattr(server, "DASH_PASSWORD", "pw")
    monkeypatch.setattr(server, "remote_control", lambda: True)
    assert ask("GET", "/devices/privacy", auth=False)[0] == 401


def test_another_site_cannot_post_a_change(ask, monkeypatch):
    monkeypatch.setattr(server, "DASH_PASSWORD", "pw")
    monkeypatch.setattr(server, "remote_control", lambda: True)
    code, reply = ask("POST", "/surroundings", origin="evil", body={"enabled": False})
    assert code == 403
    assert "this machine" in reply["error"]
