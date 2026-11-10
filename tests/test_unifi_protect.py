from unittest.mock import create_autospec

import pytest

from liberational.unifi import UniFiError, UniFiSession
from liberational.vision.unifi_protect import UniFiProtect

CAMERA_ID = "000000000000000000000001"
CAMERA = {
    "id": CAMERA_ID,
    "name": "Workshop test camera",
    "type": "Example Camera",
    "state": "CONNECTED",
    "mac": "02:00:00:00:00:20",
    "host": "192.0.2.20",
}
TOKEN = "synthetic-stream-token"


def controller():
    client = create_autospec(UniFiSession, instance=True)
    client.origin = "https://192.168.50.1:443"
    return client


def test_legacy_bootstrap_discovers_only_enabled_streams_using_nvr_not_camera_ip():
    client = controller()
    client.protect_bootstrap.return_value = {
        "cameras": [
            CAMERA
            | {
                "channels": [
                    {"id": 0, "isRtspEnabled": True, "rtspAlias": TOKEN},
                    {"id": 1, "isRtspEnabled": False, "rtspAlias": "must-not-use-disabled"},
                ]
            }
        ],
        "nvr": {"ports": {"rtsp": 7447, "rtsps": 7441}},
    }
    result = UniFiProtect(client).discover()
    assert result.complete and len(result.cameras) == 1
    camera = result.cameras[0]
    assert (camera.name, camera.model, camera.ip) == ("Workshop test camera", "Example Camera", "192.0.2.20")
    assert [(stream.quality, stream.url) for stream in camera.streams] == [
        ("high", "rtsps://192.168.50.1:7441/synthetic-stream-token?enableSrtp"),
        ("high", "rtsp://192.168.50.1:7447/synthetic-stream-token"),
    ]
    assert TOKEN not in repr(result) and TOKEN not in repr(camera.streams[0])
    client.protect_cameras.assert_not_called()
    client.protect_streams.assert_not_called()


def test_public_camera_ip_stays_unknown_and_existing_urls_are_not_created_or_opened():
    client = controller()
    client.protect_cameras.return_value = [CAMERA]
    client.protect_streams.return_value = {
        "high": "rtsps://192.168.50.1:7441/synthetic-stream-token?enableSrtp",
        "medium": None,
        "low": None,
    }
    result = UniFiProtect(client, mode="integration").discover()
    assert result.complete
    assert result.cameras[0].ip is None
    assert len(result.cameras[0].streams) == 1
    assert client.protect_streams.call_args.args[0] == CAMERA_ID
    client.protect_bootstrap.assert_not_called()


def test_disabled_or_missing_streams_do_not_trigger_a_write():
    client = controller()
    client.protect_cameras.return_value = [CAMERA]
    client.protect_streams.return_value = {"high": None, "medium": None, "low": None, "package": None}
    result = UniFiProtect(client, mode="integration").discover()
    assert result.complete and result.cameras[0].streams == ()
    assert client.protect_streams.call_count == 1


@pytest.mark.parametrize(
    "url",
    [
        "https://192.168.50.1/live",
        "rtsps://8.8.8.8/live",
        "rtsps://public.example/live",
        "rtsps://user:sample-password@192.168.50.1/live",
        "rtsps://192.168.50.1/live#fragment",
    ],
)
def test_untrusted_stream_destinations_remain_unavailable_without_losing_camera_metadata(url):
    client = controller()
    client.protect_cameras.return_value = [CAMERA]
    client.protect_streams.return_value = {"high": url}
    result = UniFiProtect(client, mode="integration").discover()
    assert not result.complete
    assert len(result.cameras) == 1 and result.cameras[0].streams == ()
    assert url not in " ".join(result.warnings)
    assert "sample-password" not in repr(result)


def test_stream_permission_failure_retains_camera_and_has_safe_warning():
    client = controller()
    client.protect_cameras.return_value = [CAMERA]
    client.protect_streams.side_effect = UniFiError("UniFi request refused (HTTP 403)")
    result = UniFiProtect(client, mode="integration").discover()
    assert not result.complete
    assert result.cameras[0].id == CAMERA_ID
    assert result.cameras[0].streams == ()
    assert result.warnings


def test_camera_cap_is_explicitly_partial():
    client = controller()
    client.protect_cameras.return_value = [CAMERA, CAMERA | {"id": "000000000000000000000002"}]
    client.protect_streams.return_value = {}
    result = UniFiProtect(client, mode="integration", max_cameras=1).discover()
    assert len(result.cameras) == 1 and not result.complete
    assert client.protect_streams.call_count == 1
