import sys
from unittest.mock import Mock

import pytest

from liberational.audio import ambient

# Shaped like PortAudio's list on a PipeWire machine: raw ALSA hw:, ALSA plugin aliases, and the PulseAudio host API
# (real sources, output monitors, and the "Default Source" alias). Names are made up.
LINUX_PIPEWIRE = [
    (0, "HDA Intel PCH: ALC1220 Analog (hw:0,0)", 2, 44100.0, "ALSA"),
    (1, "HDA Intel PCH: ALC1220 Digital (hw:0,1)", 0, 44100.0, "ALSA"),
    (2, "HDA Intel PCH: ALC1220 Alt Analog (hw:0,2)", 2, 44100.0, "ALSA"),
    (11, "Acme Mic: USB Audio (hw:3,0)", 1, 48000.0, "ALSA"),
    (12, "sysdefault", 128, 48000.0, "ALSA"),
    (19, "lavrate", 128, 44100.0, "ALSA"),
    (22, "pipewire", 128, 44100.0, "ALSA"),
    (23, "pulse", 32, 44100.0, "ALSA"),
    (26, "upmix", 8, 44100.0, "ALSA"),
    (29, "default", 128, 44100.0, "ALSA"),
    (31, "Default Source", 32, 48000.0, "PulseAudio"),
    (32, "Default Sink", 0, 48000.0, "PulseAudio"),
    (35, "alsa_output.pci-0000_00_1f.3.iec958-stereo.monitor", 2, 48000.0, "PulseAudio"),
    (36, "alsa_input.pci-0000_00_1f.3.analog-stereo", 2, 48000.0, "PulseAudio"),
    (37, "alsa_output.usb-Acme_Mic-00.analog-stereo.monitor", 2, 48000.0, "PulseAudio"),
    (38, "alsa_input.usb-Acme_Mic-00.mono-fallback", 1, 48000.0, "PulseAudio"),
    (39, "alsa_input.usb-Webcam_Co_Cam-02.analog-stereo", 2, 48000.0, "PulseAudio"),
]
ALSA_ONLY = [d for d in LINUX_PIPEWIRE if d[4] == "ALSA"]
PULSE_MONITORS_ONLY = [
    d for d in LINUX_PIPEWIRE if d[4] == "ALSA" or d[1].endswith(".monitor") or d[1] == "Default Source"
]

MACOS = [
    (0, "MacBook Pro Microphone", 1, 48000.0, "Core Audio"),
    (1, "MacBook Pro Speakers", 0, 48000.0, "Core Audio"),
    (2, "Acme Mic", 1, 48000.0, "Core Audio"),
    (3, "Acme Mic", 1, 48000.0, "Core Audio"),
    (4, "Webcam Co Cam", 2, 48000.0, "Core Audio"),
    (5, "default", 2, 48000.0, "Core Audio"),
]


@pytest.fixture(autouse=True)
def device_filters(monkeypatch):
    monkeypatch.setattr(ambient, "EXCLUDE_SUBSTR", ["webcam"])


@pytest.mark.parametrize(
    "devs,platform,expected",
    [
        pytest.param(
            LINUX_PIPEWIRE,
            "linux",
            [
                (36, "alsa_input.pci-0000_00_1f.3.analog-stereo", 2, 48000.0),
                (38, "alsa_input.usb-Acme_Mic-00.mono-fallback", 1, 48000.0),
            ],
            id="linux-pulse-real-sources-only",
        ),
        pytest.param(
            ALSA_ONLY,
            "linux",
            [
                (0, "HDA Intel PCH: ALC1220 Analog (hw:0,0)", 2, 44100.0),
                (2, "HDA Intel PCH: ALC1220 Alt Analog (hw:0,2)", 2, 44100.0),
                (11, "Acme Mic: USB Audio (hw:3,0)", 1, 48000.0),
            ],
            id="linux-no-pulse-raw-hw-only",
        ),
        pytest.param(PULSE_MONITORS_ONLY, "linux", [], id="linux-pulse-only-monitors"),
        pytest.param(
            MACOS,
            "darwin",
            [(0, "MacBook Pro Microphone", 1, 48000.0), (2, "Acme Mic", 1, 48000.0), (5, "default", 2, 48000.0)],
            id="macos-every-input-deduped",
        ),
    ],
)
def test_select_input_devices(devs, platform, expected):
    assert ambient.select_input_devices(devs, platform) == expected


def test_two_devices_whose_names_share_a_slug_open_only_the_first():
    # They would share a folder under state/audio and their lines' keys (#247).
    devs = [(2, "Acme Mic", 1, 48000.0, "Core Audio"), (3, "Acme-Mic", 1, 48000.0, "Core Audio")]
    assert ambient.select_input_devices(devs, "darwin") == [(2, "Acme Mic", 1, 48000.0)]


def test_an_included_device_opens_though_another_name_has_its_slug():
    devs = [(2, "Acme Mic", 1, 48000.0, "Core Audio"), (3, "Acme-Mic", 1, 48000.0, "Core Audio")]
    assert ambient.select_input_devices(devs, "darwin", include=["Acme-Mic"]) == [(3, "Acme-Mic", 1, 48000.0)]


def test_include_list_narrows_after_the_platform_rules():
    acme = "alsa_input.usb-Acme_Mic-00.mono-fallback"
    assert [d[1] for d in ambient.select_input_devices(LINUX_PIPEWIRE, "linux", include=[acme])] == [acme]
    # include=[] lists every real input, which is what the dashboard's source picker shows
    assert len(ambient.select_input_devices(LINUX_PIPEWIRE, "linux", include=[])) == 2
    # a monitor source can't be picked by name
    assert (
        ambient.select_input_devices(
            LINUX_PIPEWIRE, "linux", include=["alsa_output.usb-Acme_Mic-00.analog-stereo.monitor"]
        )
        == []
    )


class Stop(Exception):
    pass


@pytest.fixture
def portaudio(monkeypatch):
    sd = Mock()
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    return sd
