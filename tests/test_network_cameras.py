import json
import os
import time
import tomllib
from email.message import Message
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import create_autospec

import pytest

from liberational.dashboard import network_cameras as cameras
from liberational.proximity.network import NetworkDevice, NetworkInventory
from liberational.vision.onvif import OnvifCamera, OnvifDiscovery, OnvifStream
from liberational.vision.unifi_protect import ProtectCamera, ProtectDiscovery, ProtectStream

if TYPE_CHECKING:
    from liberational.dashboard.server import H

URL = "rtsps://192.168.1.20:7441/private-token?enableSrtp"
REAL_GET = cameras.cfg.get


def settings(values):
    """cfg.get with these keys replaced; every other key keeps its default from the suite's temp pack.toml."""
    return lambda key, *default: values[key] if key in values else REAL_GET(key, *default)


@pytest.fixture
def manifest(tmp_path, monkeypatch):
    path = tmp_path / "config" / "devices.toml"
    path.parent.mkdir()
    path.write_text('# existing USB camera\n[[device]]\nname="desk"\nkind="camera"\nmatch={name="USB"}\n')
    monkeypatch.setattr(cameras.cameras, "manifest_path", lambda: path)
    monkeypatch.setattr(cameras, "capture_ready", lambda: True)
    monkeypatch.setattr(cameras, "controls_ready", lambda: True)
    cameras._proposals.clear()
    return path


def proposal():
    cameras._proposals[("pass", "camera", "stream")] = (time.monotonic() + 300, URL)
    return dict(discovery_id="pass", camera_id="camera", stream_id="stream", name="entryway")


def test_add_preserves_usb_manifest_and_stores_private_url_with_owner_only_permissions(manifest, monkeypatch):
    restart = create_autospec(cameras.settings.restart)
    monkeypatch.setattr(cameras.settings, "restart", restart)
    code, result = cameras.add(proposal())
    assert code == 200 and result == {"ok": True}
    rows = tomllib.loads(manifest.read_text())["device"]
    assert rows[0] == dict(name="desk", kind="camera", match={"name": "USB"})
    assert rows[1]["kind"] == "rtsp" and rows[1]["rtsp"]["url"] == URL
    assert rows[1]["rtsp"]["transport"] == "tcp"
    assert os.stat(manifest).st_mode & 0o777 == 0o600
    assert "private-token" not in json.dumps(cameras.view())
    assert restart.call_args.kwargs == dict(names=["continuum-capture"], dashboard=False)


def test_network_camera_off_and_on_preserve_private_row_and_identity(manifest, monkeypatch):
    restart = create_autospec(cameras.settings.restart)
    monkeypatch.setattr(cameras.settings, "restart", restart)
    monkeypatch.setattr(cameras, "controls_ready", lambda: True)
    cameras.add(proposal())
    original = tomllib.loads(manifest.read_text())["device"]
    code, reply = cameras.use({"id": "entryway", "enabled": False})
    assert code == 200 and reply["ok"] and reply["restarted"] == ["continuum-capture"]
    disabled = tomllib.loads(manifest.read_text())["device"]
    assert disabled == [original[0], {**original[1], "enabled": False}]
    assert cameras.view()["cameras"][0]["enabled"] is False
    assert "private-token" not in json.dumps(reply)
    code, _ = cameras.use({"id": "entryway", "enabled": True})
    assert code == 200
    assert tomllib.loads(manifest.read_text())["device"] == [original[0], {**original[1], "enabled": True}]
    assert os.stat(manifest).st_mode & 0o777 == 0o600


@pytest.mark.parametrize("body", [{"id": "entryway", "enabled": "false"}, {"id": "desk", "enabled": False}])
def test_camera_use_refuses_invalid_toggle_without_writing(manifest, monkeypatch, body):
    monkeypatch.setattr(cameras, "controls_ready", lambda: True)
    before = manifest.read_bytes()
    code, _ = cameras.use(body)
    assert code in (400, 404) and manifest.read_bytes() == before


def test_camera_use_requires_native_enabled_support_before_writing(manifest, monkeypatch):
    monkeypatch.setattr(cameras, "controls_ready", lambda: False)
    before = manifest.read_bytes()
    code, _ = cameras.use({"id": "entryway", "enabled": False})
    assert code == 409 and manifest.read_bytes() == before


@pytest.mark.parametrize(
    "capabilities,ready",
    [
        ({"rtsp_h264": True}, False),
        ({"rtsp_h264": True, "device_enabled": True}, True),
        ({"rtsp_h264": False, "device_enabled": True}, False),
    ],
)
def test_camera_use_capability_requires_both_native_enabled_and_rtsp_support(monkeypatch, capabilities, ready):
    run = create_autospec(
        cameras.subprocess.run, return_value=SimpleNamespace(returncode=0, stdout=json.dumps(capabilities).encode())
    )
    monkeypatch.setattr(cameras.subprocess, "run", run)
    assert cameras.controls_ready() is ready


def test_camera_use_restart_failure_reports_saved_state_without_claiming_a_rollback(manifest, monkeypatch):
    monkeypatch.setattr(cameras.settings, "restart", lambda *a, **kw: None)
    cameras.add(proposal())

    def refused(*args, **kwargs):
        raise RuntimeError("synthetic restart refusal")

    monkeypatch.setattr(cameras.settings, "restart", refused)
    code, body = cameras.use({"id": "entryway", "enabled": False})
    assert code == 500 and "was saved" in body["error"]
    assert tomllib.loads(manifest.read_text())["device"][1]["enabled"] is False


def test_camera_use_write_failure_never_restarts_or_changes_private_configuration(manifest, monkeypatch):
    monkeypatch.setattr(cameras.settings, "restart", lambda *a, **kw: None)
    cameras.add(proposal())
    before = manifest.read_bytes()
    restart = create_autospec(cameras.settings.restart)
    monkeypatch.setattr(cameras.settings, "restart", restart)

    def unwritable(path, text):
        raise OSError("synthetic write failure")

    monkeypatch.setattr(cameras.cameras, "_write_manifest", unwritable)
    code, body = cameras.use({"id": "entryway", "enabled": False})
    assert code == 500 and "error" in body
    assert manifest.read_bytes() == before
    restart.assert_not_called()


def test_camera_use_corrupt_manifest_is_actionable_without_secret_excerpt(manifest):
    manifest.write_text('[[device]]\nrtsp = {url="private-token"\n')
    before = manifest.read_bytes()
    code, body = cameras.use({"id": "entryway", "enabled": False})
    assert code == 500 and "error" in body
    assert manifest.read_bytes() == before
    assert "private-token" not in json.dumps(body)


def test_network_camera_name_is_a_display_alias_without_changing_id(manifest, monkeypatch):
    monkeypatch.setattr(cameras.settings, "restart", lambda *a, **kw: None)
    cameras.add(proposal())
    before = manifest.read_bytes()
    with cameras.cfg.override({"vision": {"camera_names": ["entryway: Front door"]}}):
        row = cameras.view()["cameras"][0]
    assert row["name"] == "Front door" and row["id"] == "entryway"
    assert manifest.read_bytes() == before


def test_unsupported_capture_cannot_write_unknown_device_kind(manifest, monkeypatch):
    before = manifest.read_bytes()
    monkeypatch.setattr(cameras, "capture_ready", lambda: False)
    code, _ = cameras.add(proposal())
    assert code == 409 and manifest.read_bytes() == before


@pytest.mark.parametrize("key,value", [("stream_id", "arbitrary-url"), ("name", "../outside")])
def test_invalid_add_does_not_modify_manifest(manifest, key, value):
    before = manifest.read_bytes()
    body = {**proposal(), key: value}
    code, _ = cameras.add(body)
    assert code in (400, 409) and manifest.read_bytes() == before


def test_expired_proposal_is_not_an_arbitrary_url_write_capability(manifest):
    body = proposal()
    cameras._proposals[("pass", "camera", "stream")] = (0, URL)
    code, _ = cameras.add(body)
    assert code == 409
    assert len(tomllib.loads(manifest.read_text())["device"]) == 1


def test_corrupt_manifest_reports_actionable_error_without_leaking_toml_source(manifest):
    manifest.write_text('[[device]]\nrtsp = {url="private-token"\n')
    reply = cameras.view()
    assert "error" in reply and not reply["capture_ready"]
    assert "private-token" not in json.dumps(reply)


def test_protect_discovery_exposes_only_opaque_stream_ids_and_redacted_urls(manifest, monkeypatch):
    session = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(cameras.cfg, "get", settings({"unifi.url": "https://192.168.1.1", "unifi.mode": "legacy"}))
    monkeypatch.setattr(cameras, "configured_session", lambda section="unifi": session)
    result = ProtectDiscovery((ProtectCamera("native-id", name="Entry", streams=(ProtectStream("High", URL),)),), True)
    monkeypatch.setattr(cameras, "UniFiProtect", lambda *a, **k: SimpleNamespace(discover=lambda: result))
    code, reply = cameras.discover({"source": "protect"})
    assert code == 200
    assert reply["cameras"][0]["streams"][0]["ready"]
    assert "private-token" not in json.dumps(reply)
    key = (reply["discovery_id"], reply["cameras"][0]["id"], reply["cameras"][0]["streams"][0]["id"])
    assert cameras._proposals[key][1] == URL


def test_onvif_credentials_are_encoded_into_private_stream_only(manifest, monkeypatch):
    monkeypatch.setattr(
        cameras.cfg,
        "get",
        settings(
            {
                "network_cameras.onvif_username": "cam@local",
                "network_cameras.onvif_password": "p:ass/word",
            }
        ),
    )
    result = OnvifDiscovery(
        (
            OnvifCamera(
                "camera", "192.168.1.20", "Entry", "Test", (OnvifStream("Main", "rtsp://192.168.1.20/main"),), False
            ),
        ),
        False,
    )
    monkeypatch.setattr(cameras, "OnvifDiscoverySource", lambda **k: SimpleNamespace(discover=lambda: result))
    code, reply = cameras.discover({"source": "onvif"})
    assert code == 200
    assert "cam@local" not in json.dumps(reply) and "p:ass/word" not in json.dumps(reply)
    key = (reply["discovery_id"], "0", "0")
    assert cameras._proposals[key][1] == "rtsp://cam%40local:p%3Aass%2Fword@192.168.1.20/main"


@pytest.mark.parametrize("path", ["/network-cameras/discover", "/network-cameras/use"])
@pytest.mark.parametrize(
    "control,origin,content_type",
    [(False, None, "application/json"), (True, "http://other.invalid", "application/json"), (True, None, "text/plain")],
)
def test_discovery_rejects_remote_or_cross_origin_mutations_before_network(
    monkeypatch, control, origin, content_type, path
):
    from liberational.dashboard.server import H

    headers = Message()
    headers["Host"] = "localhost:8765"
    headers["Content-Type"] = content_type
    if origin:
        headers["Origin"] = origin
    replies = []
    handler = SimpleNamespace(
        path=path,
        command="POST",
        headers=headers,
        _can_control=lambda: control,
        _json=lambda body, code: replies.append((body, code)),
    )
    # The route asks the server's own guard, so the fake carries the real one over these headers.
    handler._same_origin_json = lambda: H._same_origin_json(cast("H", handler))
    # The fake has no request body or socket: reaching discovery would fail instead of touching the LAN.
    assert cameras.handle(cast("H", handler))
    assert replies[0][1] == 403


def test_corrupt_manifest_get_is_an_http_error_for_the_settings_ui(manifest):
    manifest.write_text('[[device]]\nrtsp = {url="private-token"\n')
    replies = []
    handler = SimpleNamespace(
        path="/network-cameras", command="GET", _json=lambda body, code: replies.append((body, code))
    )
    assert cameras.handle(cast("H", handler))
    assert replies[0][1] == 503
    assert "repair" in replies[0][0]["error"]
    assert "private-token" not in json.dumps(replies)


@pytest.mark.parametrize(
    "camera_ip,url",
    [
        ("192.168.1.21", "rtsp://192.168.1.20/main"),
        ("fd00::21", "rtsps://[fd00::20]/main"),
        (None, "rtsp://192.168.1.20/main"),
        ("", "rtsp://192.168.1.20/main"),
        ("camera.invalid", "rtsp://192.168.1.20/main"),
        ("camera.invalid", "rtsp://camera.invalid/main"),
    ],
)
def test_onvif_credentials_not_attached_when_stream_host_does_not_match_camera(manifest, monkeypatch, camera_ip, url):
    monkeypatch.setattr(
        cameras.cfg,
        "get",
        settings(
            {
                "network_cameras.onvif_username": "fixture-viewer",
                "network_cameras.onvif_password": "fixture-password",
            }
        ),
    )
    result = OnvifDiscovery((OnvifCamera("camera", camera_ip, streams=(OnvifStream("Main", url),)),), False)
    monkeypatch.setattr(cameras, "OnvifDiscoverySource", lambda **kwargs: SimpleNamespace(discover=lambda: result))
    code, reply = cameras.discover({"source": "onvif"})
    assert code == 200
    assert cameras._proposals[(reply["discovery_id"], "0", "0")][1] == url
    assert "fixture-viewer" not in json.dumps(reply)
    assert "fixture-password" not in str(cameras._proposals)


def test_onvif_credentials_attach_to_equivalent_ipv6_camera_address(manifest, monkeypatch):
    monkeypatch.setattr(
        cameras.cfg,
        "get",
        settings(
            {
                "network_cameras.onvif_username": "fixture-viewer",
                "network_cameras.onvif_password": "fixture-password",
            }
        ),
    )
    result = OnvifDiscovery(
        (
            OnvifCamera(
                "camera",
                "fd00:0000:0000:0000:0000:0000:0000:0020",
                streams=(OnvifStream("Main", "rtsp://[fd00::20]:554/main"),),
            ),
        ),
        False,
    )
    monkeypatch.setattr(cameras, "OnvifDiscoverySource", lambda **kwargs: SimpleNamespace(discover=lambda: result))
    code, reply = cameras.discover({"source": "onvif"})
    assert code == 200
    assert cameras._proposals[(reply["discovery_id"], "0", "0")][1] == (
        "rtsp://fixture-viewer:fixture-password@[fd00::20]:554/main"
    )
    assert "fixture-password" not in json.dumps(reply)


def test_onvif_stream_credentials_are_not_replaced_by_configured_credentials(manifest, monkeypatch):
    monkeypatch.setattr(
        cameras.cfg,
        "get",
        settings(
            {
                "network_cameras.onvif_username": "fixture-viewer",
                "network_cameras.onvif_password": "fixture-password",
            }
        ),
    )
    url = "rtsp://stream-user:stream-password@192.168.1.20/main"
    result = OnvifDiscovery((OnvifCamera("camera", "192.168.1.20", streams=(OnvifStream("Main", url),)),), False)
    monkeypatch.setattr(cameras, "OnvifDiscoverySource", lambda **kwargs: SimpleNamespace(discover=lambda: result))
    code, reply = cameras.discover({"source": "onvif"})
    assert code == 200
    assert cameras._proposals[(reply["discovery_id"], "0", "0")][1] == url
    assert "fixture-password" not in str(cameras._proposals)


def test_protect_nvr_stream_never_receives_onvif_or_controller_credentials(manifest, monkeypatch):
    session = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(
        cameras.cfg,
        "get",
        settings(
            {
                "unifi.url": "https://192.168.1.20",
                "unifi.mode": "legacy",
                "unifi.username": "controller-user",
                "unifi.password": "controller-password",
                "network_cameras.onvif_username": "fixture-viewer",
                "network_cameras.onvif_password": "fixture-password",
            }
        ),
    )
    monkeypatch.setattr(cameras, "configured_session", lambda section="unifi": session)
    result = ProtectDiscovery(
        (ProtectCamera("camera", ip="192.168.1.30", streams=(ProtectStream("High", URL),)),), True
    )
    monkeypatch.setattr(cameras, "UniFiProtect", lambda *args, **kwargs: SimpleNamespace(discover=lambda: result))
    code, reply = cameras.discover({"source": "protect"})
    assert code == 200
    assert cameras._proposals[(reply["discovery_id"], "0", "0")][1] == URL
    assert "fixture-password" not in str(cameras._proposals)
    assert "controller-password" not in str(cameras._proposals)


@pytest.mark.parametrize(
    ("protect", "origin", "mode"),
    [
        pytest.param(
            dict(url="https://192.168.1.224", mode="integration"), "https://192.168.1.224", "integration", id="nvr"
        ),
        pytest.param({}, "https://192.168.1.1", "legacy", id="single-console"),
    ],
)
def test_protect_discovery_reads_the_protect_console(manifest, monkeypatch, protect, origin, mode):
    used = []

    def protect_client(session, *, mode):
        used.append((session.origin, mode))
        return SimpleNamespace(discover=lambda: ProtectDiscovery((), True))

    monkeypatch.setattr(cameras, "UniFiProtect", protect_client)
    with cameras.cfg.override({"unifi": dict(url="https://192.168.1.1", mode="legacy"), "unifi_protect": protect}):
        code, reply = cameras.discover({"source": "protect"})
    assert code == 200 and reply["cameras"] == []
    assert used == [(origin, mode)]


def device(ip, connected=True):
    return NetworkDevice("unifi_client_id", f"client-{ip}", ip=ip, connected=connected)


class Inventory:
    def __init__(self, devices=(), error=None):
        self.devices, self.error, self.sessions = tuple(devices), error, []

    def __call__(self, session, **kwargs):
        self.sessions.append(session.origin)
        return self

    def poll(self, now):
        if self.error:
            raise self.error
        return NetworkInventory("unifi", "local-network", now, self.devices, complete=True)


def onvif_discovery(monkeypatch, values, inventory):
    probed = []

    def onvif_source(**kwargs):
        probed.append(kwargs["hosts"])
        return SimpleNamespace(discover=lambda: OnvifDiscovery((), False))

    monkeypatch.setattr(cameras, "OnvifDiscoverySource", onvif_source)
    monkeypatch.setattr(cameras, "UniFiNetwork", inventory)
    with cameras.cfg.override(values):
        code, reply = cameras.discover({"source": "onvif"})
    assert code == 200
    return probed[0], reply["warnings"]


def test_onvif_probes_listed_hosts_then_current_network_clients_once_each(manifest, monkeypatch):
    inventory = Inventory(
        [device("192.168.1.30"), device("192.168.1.40"), device("fd00::40"), device("192.168.1.50", connected=False)]
    )
    values = {
        "unifi": dict(url="https://192.168.1.1"),
        "unifi_protect": dict(url="https://192.168.1.224"),
        "network_cameras": dict(onvif_hosts=["192.168.1.30", "8.8.8.8", "camera.local"]),
    }
    hosts, warnings = onvif_discovery(monkeypatch, values, inventory)
    assert hosts == ["192.168.1.30", "192.168.1.40"]
    assert inventory.sessions == ["https://192.168.1.1"]
    assert any("onvif_hosts entries" in warning for warning in warnings)


def test_onvif_unicast_targets_are_capped_at_256(manifest, monkeypatch):
    inventory = Inventory([device(f"10.0.{i // 200}.{i % 200 + 1}") for i in range(300)])
    hosts, warnings = onvif_discovery(monkeypatch, {"unifi": dict(url="https://10.0.0.1")}, inventory)
    assert len(hosts) == 256 and hosts[0] == "10.0.0.1"
    assert any("capped at 256" in warning for warning in warnings)


@pytest.mark.parametrize(
    "inventory",
    [pytest.param(Inventory(error=RuntimeError("private detail")), id="raises"), pytest.param(Inventory(), id="empty")],
)
def test_unavailable_client_list_still_probes_listed_hosts(manifest, monkeypatch, inventory):
    values = {"unifi": dict(url="https://192.168.1.1"), "network_cameras": dict(onvif_hosts=["192.168.1.30"])}
    hosts, warnings = onvif_discovery(monkeypatch, values, inventory)
    assert hosts == ["192.168.1.30"]
    assert any("client list was unavailable" in warning for warning in warnings)
    assert "private detail" not in json.dumps(warnings)


def test_onvif_without_a_network_console_probes_only_listed_hosts(manifest, monkeypatch):
    inventory = Inventory([device("192.168.1.40")])
    values = {"network_cameras": dict(onvif_hosts=["192.168.1.30"])}
    hosts, _ = onvif_discovery(monkeypatch, values, inventory)
    assert hosts == ["192.168.1.30"]
    assert inventory.sessions == []
