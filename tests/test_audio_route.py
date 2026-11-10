import pytest

from liberational.audio import route as audio_route
from liberational.audio.route import select_output

PREFERRED_UID = "AA-BB-CC-DD-EE-FF:output"
HEADSET = {"uid": PREFERRED_UID}
BUILTIN = {"uid": "BuiltInSpeakerDevice"}
HDMI = {"uid": "hdmi"}


@pytest.fixture(autouse=True)
def headset_configured():
    with audio_route.cfg.override({"audio": {"bluetooth_output": "aa:bb:cc:dd:ee:ff"}}):
        yield


def test_configured_address():
    assert audio_route.address() == "aa-bb-cc-dd-ee-ff"
    assert audio_route.preferred_uid() == PREFERRED_UID


@pytest.mark.parametrize(
    "devices, connected, current, expected",
    [
        pytest.param([HDMI, BUILTIN, HEADSET], True, "hdmi", HEADSET, id="connected"),
        pytest.param([HEADSET, BUILTIN], False, PREFERRED_UID, BUILTIN, id="disconnected-stale-device"),
        pytest.param([HDMI, BUILTIN], False, PREFERRED_UID, BUILTIN, id="disconnected-removed-device"),
        pytest.param([BUILTIN], True, None, BUILTIN, id="connected-not-ready"),
        pytest.param([BUILTIN, HEADSET], True, BUILTIN["uid"], HEADSET, id="reconnect"),
        pytest.param([HDMI], False, "hdmi", HDMI, id="no-builtin-keeps-current"),
        pytest.param([HEADSET, HDMI], False, PREFERRED_UID, None, id="no-safe-fallback"),
        pytest.param([], False, None, None, id="no-devices"),
    ],
)
def test_select_output(devices, connected, current, expected):
    assert select_output(devices, connected, current) == expected


def test_no_headset_configured_falls_back_to_builtin():
    with audio_route.cfg.override({"audio": {"bluetooth_output": ""}}):
        assert audio_route.preferred_uid() is None
        assert select_output([HDMI, BUILTIN], True, "hdmi") == BUILTIN
