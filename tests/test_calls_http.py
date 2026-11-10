import base64
import http.client
import json
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest

from liberational import config as cfg
from liberational.dashboard import call_log
from liberational.phone import outbound


@pytest.fixture
def hangup_http(dashboard_server, serve_dashboard, monkeypatch, tmp_path):
    monkeypatch.setattr(call_log, "CALLS_DIR", tmp_path / "calls")
    monkeypatch.setattr(dashboard_server, "DASH_PASSWORD", "synthetic-password")
    monkeypatch.setattr(dashboard_server.H, "_local", lambda self: False)
    env = tmp_path / "phone.env"
    env.write_text("TWILIO_ACCOUNT_SID=ACfake\nTWILIO_AUTH_TOKEN=fake\n")
    sid = "CA" + "1" * 32
    day, name = "2026-11-06", "130200_" + sid
    _live(call_log.CALLS_DIR / day / name, callSid=sid)
    send = create_autospec(
        outbound.requests.post, return_value=SimpleNamespace(status_code=200, json=lambda: {"status": "completed"})
    )
    monkeypatch.setattr(outbound.requests, "post", send)
    host = serve_dashboard().removeprefix("http://")

    def post(body, *, auth=True, origin=None, content_type="application/json"):
        headers = {"Content-Type": content_type, "Origin": origin or f"https://{host}"}
        if auth:
            headers["Authorization"] = "Basic " + base64.b64encode(b"viewer:synthetic-password").decode()
        connection = http.client.HTTPConnection(host, timeout=5)
        try:
            connection.request("POST", "/calls/hangup", json.dumps(body), headers)
            response = connection.getresponse()
            raw = response.read()
            return response.status, json.loads(raw) if raw else {}
        finally:
            connection.close()

    with cfg.override({"phone": {"env_file": str(env)}}):
        yield post, send, {"day": day, "dir": name, "confirmed": True}


@pytest.mark.parametrize(
    "kwargs,status",
    [({"auth": False}, 401), ({"origin": "https://elsewhere.invalid"}, 403), ({"content_type": "text/plain"}, 403)],
)
def test_hangup_cannot_bypass_auth_or_same_origin(hangup_http, kwargs, status):
    post, send, body = hangup_http
    assert post(body, **kwargs)[0] == status
    send.assert_not_called()


@pytest.mark.parametrize(
    "change,status",
    [({"confirmed": False}, 400), ({"day": ""}, 400), ({"dir": "../../other"}, 409), ({"dir": "previous-call"}, 409)],
)
def test_hangup_cannot_end_an_unselected_or_stale_call(hangup_http, change, status):
    post, send, body = hangup_http
    assert post({**body, **change})[0] == status
    send.assert_not_called()


def test_hangup_https_targets_only_the_selected_live_call(hangup_http):
    post, send, body = hangup_http
    assert post(body) == (200, {"ok": True, "detail": "Call ended."})
    send.assert_called_once_with(
        "https://api.twilio.com/2010-04-01/Accounts/ACfake/Calls/CA" + "1" * 32 + ".json",
        auth=("ACfake", "fake"),
        data={"Status": "completed"},
        timeout=10,
    )


def test_hangup_reports_provider_failure_instead_of_success(hangup_http):
    post, send, body = hangup_http
    send.return_value.status_code = 503
    status, result = post(body)
    assert status == 409 and result["ok"] is False
    assert "503" in result["error"]


def test_hangup_reports_connection_failure_instead_of_success(hangup_http):
    post, send, body = hangup_http
    send.side_effect = outbound.requests.Timeout("synthetic timeout")
    status, result = post(body)
    assert status == 502 and result["ok"] is False


def test_call_route_requires_auth_origin_and_confirmation(dashboard_server, serve_dashboard, monkeypatch):
    monkeypatch.setattr(dashboard_server, "DASH_PASSWORD", "synthetic-password")
    monkeypatch.setattr(dashboard_server.H, "_local", lambda self: False)
    base = serve_dashboard()
    sent = []

    def place(name, message, *, number=None):
        sent.append((name, message, number))
        return True, "Synthetic only"

    monkeypatch.setattr(outbound, "place", place)

    def post(body, origin=None):
        c = http.client.HTTPConnection(base.removeprefix("http://"))
        headers = {"Content-Type": "application/json"}
        if origin:
            headers["Origin"] = origin
        c.request("POST", "/calls/place", json.dumps(body), headers)
        response = c.getresponse()
        status = response.status
        response.read()
        c.close()
        return status

    body = {"number": "+15555550123", "message": "Arrange dinner", "confirmed": True}
    assert post(body) == 401
    monkeypatch.setattr(dashboard_server.H, "_authed", lambda self: True)
    assert post(body, "https://evil.invalid") == 403
    assert post({"number": "+15555550123"}) == 400
    assert sent == []
    assert post(body) == 200
    assert sent == [("", "Arrange dinner", "+15555550123")]


def test_the_call_list_reads_an_older_call_without_a_direction_as_unknown(tmp_path, monkeypatch):
    """Calls saved before direction, to, instructions, and outcome were kept aren't migrated."""
    old = tmp_path / "2026-11-01" / "090000-CA1"
    new = tmp_path / "2026-11-06" / "130200-CA2"
    for d in (old, new):
        d.mkdir(parents=True)
    (old / "call.json").write_text(json.dumps({"callSid": "CA1", "from": "+15555550123"}))
    placed = {
        "callSid": "CA2",
        "from": "+15555550199",
        "direction": "out",
        "to": {"name": "Dana", "number": "+15555550199"},
    }
    (new / "call.json").write_text(
        json.dumps({**placed, "instructions": "Ask about Friday", "outcome": "Friday at 7."})
    )
    monkeypatch.setattr(call_log, "CALLS_DIR", tmp_path)
    by = {c["callSid"]: c for c in call_log.calls_list()}
    assert {k: by["CA1"][k] for k in ("direction", "to", "instructions", "outcome")} == {
        "direction": "unknown",
        "to": None,
        "instructions": None,
        "outcome": None,
    }
    assert (by["CA2"]["direction"], by["CA2"]["to"]["name"], by["CA2"]["outcome"]) == ("out", "Dana", "Friday at 7.")


def _live(folder, **over):
    folder.mkdir(parents=True)
    import os
    import time

    mark = {"callSid": "CA9", "direction": "out", "number": "+15555550199", "name": "Dana", "since": time.time() - 65}
    (folder / "live.json").write_text(json.dumps({**mark, "pid": os.getpid(), **over}))


def test_the_call_on_now_says_which_way_who_and_since_when(tmp_path, monkeypatch):
    """GET /calls/live for the Calls page's "on a call" row: from the bridge's live mark."""
    monkeypatch.setattr(call_log, "CALLS_DIR", tmp_path)
    assert call_log.calls_live() == {"live": None, "count": 0}
    _live(tmp_path / "2026-11-06" / "130200_CA9")
    got = call_log.calls_live()
    assert got["count"] == 1
    live = got["live"]
    assert {k: live[k] for k in ("direction", "number", "name", "day", "dir")} == {
        "direction": "out",
        "number": "+15555550199",
        "name": "Dana",
        "day": "2026-11-06",
        "dir": "130200_CA9",
    }


def test_a_live_mark_a_crash_left_behind_is_not_a_call_on(tmp_path, monkeypatch):
    import subprocess
    import time

    gone = subprocess.Popen(["true"])
    gone.wait()  # a process that has exited: its pid is free
    monkeypatch.setattr(call_log, "CALLS_DIR", tmp_path)
    _live(tmp_path / "2026-11-06" / "080000_CA1", pid=gone.pid)
    _live(tmp_path / "2026-11-06" / "090000_CA2", since=time.time() - call_log.LIVE_CALL_MAX_S - 1)
    (tmp_path / "2026-11-06" / "100000_CA3").mkdir()
    (tmp_path / "2026-11-06" / "100000_CA3" / "live.json").write_text("{not json")
    assert call_log.calls_live() == {"live": None, "count": 0}


def test_the_dashboard_serves_the_call_on_now(serve_dashboard, monkeypatch, tmp_path):
    monkeypatch.setattr(call_log, "CALLS_DIR", tmp_path)
    _live(tmp_path / "2026-11-06" / "130200_CA9", direction="in", name=None, number="+15555550123")
    base = serve_dashboard()
    c = http.client.HTTPConnection(base.removeprefix("http://"))
    c.request("GET", "/calls/live")
    response = c.getresponse()
    body = json.loads(response.read())
    c.close()
    assert response.status == 200
    assert (body["count"], body["live"]["direction"], body["live"]["name"]) == (1, "in", None)
