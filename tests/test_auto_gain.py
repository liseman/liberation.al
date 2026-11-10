"""Auto gain (#410): a mic's level steered toward speech peaks near audio.auto_gain_target_dbfs, by its own capture
volume when the OS can set it, else by software gain. No test touches a real mixer: amixer and pw-dump are fakes
holding what the Elgato Wave:3 shows on devbox."""

import json
import tomllib

import numpy as np
import pytest

from liberational import config as cfg
from liberational.audio import gain

WAVE = "alsa_input.usb-Elgato_Systems_Elgato_Wave_3_A017A41910AAFN-00.mono-fallback"
PW_DUMP = json.dumps(
    [
        {"info": {"props": {"media.class": "Audio/Source", "node.name": "alsa_input.pci-0000_00_1f.3.analog-stereo",
                            "api.alsa.pcm.card": "0"}}},
        {"info": {"props": {"media.class": "Audio/Source", "node.name": WAVE, "api.alsa.pcm.card": "1"}}},
    ]
)  # fmt: skip
SCONTENTS = """Simple mixer control 'PCM',0
  Capabilities: pvolume pvolume-joined pswitch pswitch-joined
  Playback channels: Mono
  Limits: Playback 0 - 120
  Mono: Playback 79 [66%] [-20.50dB] [on]
Simple mixer control 'Mic',0
  Capabilities: cvolume cvolume-joined cswitch cswitch-joined
  Capture channels: Mono
  Limits: Capture 0 - 80
  Mono: Capture 0 [0%] [0.00dB] [on]
"""
CONTENTS = """numid=6,iface=MIXER,name='Mic Capture Volume'
  ; type=INTEGER,access=rw---R--,values=1,min=0,max=80,step=0
  : values=0
  | dBminmax-min=0.00dB,max=40.00dB
"""


class FakeMixer:
    """amixer and pw-dump as the Wave:3 answers them, with the capture volume held here."""

    def __init__(self) -> None:
        self.db = 0.0
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str]) -> str:
        self.calls.append(args)
        if args == ["pw-dump"]:
            return PW_DUMP
        if args[-1] == "scontents":
            return SCONTENTS if args[2] == "1" else SCONTENTS.split("Simple mixer control 'Mic'")[0]  # card 0: none
        if args[-1] == "contents":
            return CONTENTS
        if "sget" in args:
            return f"Simple mixer control 'Mic',0\n  Limits: Capture 0 - 80\n  Mono: Capture 0 [0%] [{self.db:.2f}dB] [on]\n"
        if "sset" in args:
            self.db = float(args[-1].removesuffix("dB"))
            return ""
        raise AssertionError(f"unexpected {args}")


@pytest.fixture
def pack_toml(tmp_path, monkeypatch):
    path = tmp_path / "pack.toml"
    monkeypatch.setattr(cfg, "local_config", lambda: path)
    return path


def tone(dbfs: float, seconds: float = 1.0, sr: int = 16000) -> np.ndarray:
    """A sine whose 20 ms RMS sits at dbfs."""
    t = np.arange(int(sr * seconds)) / sr
    return (np.sqrt(2) * 10 ** (dbfs / 20) * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_a_speech_peak_is_the_loud_part_of_an_utterance_not_its_one_loudest_sample():
    quiet, loud = tone(-50, 0.9), tone(-20, 0.05)
    speech, top = gain.speech_peak(np.concatenate([quiet, loud, quiet]), 16000)
    assert speech == pytest.approx(-50, abs=0.5)  # under 3 % loud frames: below the 95th percentile
    assert top == pytest.approx(-20 + 3.01, abs=0.1)  # a sine's sample peak is 3 dB over its RMS
    assert gain.speech_peak(np.zeros(10, np.float32), 16000)[0] == -120.0  # shorter than one frame


def heard(auto: gain.AutoGain, dbfs: float, at: float, n: int = 3) -> None:
    for i in range(n):
        auto.heard(dbfs, dbfs + 3, 1.0, at + i)


@pytest.mark.parametrize(
    ("level", "step"),
    [
        pytest.param(-55.0, gain.UP_DB, id="quiet-goes-up-slowly"),
        pytest.param(-5.0, -gain.DOWN_DB, id="loud-comes-down-faster"),
        pytest.param(-21.0, None, id="near-the-target-stays"),
    ],
)
def test_it_moves_toward_the_target_one_bounded_step_at_a_time(level, step):
    auto = gain.AutoGain(0.0, 40.0)
    heard(auto, level, at=0.0)
    new = auto.step(now=100.0, in_speech=False, gain_db=10.0, target=-20.0)
    assert new == (None if step is None else 10.0 + step)


def test_it_never_moves_while_someone_talks_or_right_after():
    auto = gain.AutoGain(0.0, 40.0)
    heard(auto, -55.0, at=100.0)
    assert auto.step(now=150.0, in_speech=True, gain_db=10.0, target=-20.0) is None  # mid-sentence
    assert auto.step(now=102.0 + gain.QUIET_S - 0.5, in_speech=False, gain_db=10.0, target=-20.0) is None
    assert auto.step(now=102.0 + gain.QUIET_S + 0.5, in_speech=False, gain_db=10.0, target=-20.0) == 11.5


def test_moves_are_spaced_and_count_what_was_heard_at_the_new_gain():
    auto = gain.AutoGain(0.0, 40.0)
    heard(auto, -55.0, at=0.0)
    assert auto.step(now=10.0, in_speech=False, gain_db=0.0, target=-20.0) == 1.5
    assert auto.step(now=10.0 + gain.MIN_INTERVAL_S - 1, in_speech=False, gain_db=1.5, target=-20.0) is None
    auto.moved(1.0)  # the control rounded the step to 1 dB
    assert list(auto.peaks) == [-54.0, -54.0, -54.0]  # the earlier speech, as it would sound now


def test_clipped_clicks_in_quiet_speech_dont_step_down_so_the_gain_climbs_instead():
    # devbox, 2026-10-30: speech peaks near -42 dBFS with a few clipped transients stepped the Wave 3 down 3 dB, the
    # quiet speech raised it again, and it hunted: 30, 27, 28.5, 30, 27 dB. Target -26.
    auto = gain.AutoGain(0.0, 40.0)
    heard(auto, -42.5, at=0.0)
    auto.heard(-42.0, -0.2, 1.0, 3.0, clipped_samples=gain.CLIP_SAMPLES)  # a click clipped, the speech was quiet
    assert auto.step(now=30.0, in_speech=False, gain_db=30.0, target=-26.0) == 30.0 + gain.UP_DB


def test_a_loud_utterance_that_clips_steps_down_and_the_gain_holds_before_rising_again():
    auto = gain.AutoGain(0.0, 40.0)
    auto.heard(-18.0, -0.1, 2.0, 0.0, clipped_samples=40)  # loud, and clipping
    assert auto.step(now=20.0, in_speech=False, gain_db=30.0, target=-26.0) == 30.0 - gain.DOWN_DB
    heard(auto, -40.0, at=30.0)  # then quiet speech, which would raise it
    assert auto.step(now=60.0, in_speech=False, gain_db=27.0, target=-26.0) is None, "held after the clip"
    assert auto.step(now=20.0 + gain.CLIP_HOLD_S + 1, in_speech=False, gain_db=27.0, target=-26.0) == 27.0 + gain.UP_DB


def test_one_or_two_clipped_samples_are_not_a_clip_even_in_loud_speech():
    auto = gain.AutoGain(0.0, 40.0)
    auto.heard(-24.0, -0.1, 2.0, 0.0, clipped_samples=gain.CLIP_SAMPLES - 1)
    assert auto.step(now=20.0, in_speech=False, gain_db=30.0, target=-26.0) is None  # within the deadband


def test_it_stays_within_the_bounds_and_short_sounds_do_not_count():
    auto = gain.AutoGain(0.0, 40.0)
    heard(auto, -80.0, at=0.0)
    assert auto.step(now=10.0, in_speech=False, gain_db=40.0, target=-20.0) is None  # at the top already
    clicks = gain.AutoGain(0.0, 40.0)
    clicks.heard(-80.0, -77.0, 0.1, 0.0)
    assert clicks.step(now=10.0, in_speech=False, gain_db=10.0, target=-20.0) is None


def test_the_wave_3_uses_its_own_usb_capture_volume(monkeypatch):
    monkeypatch.setattr(gain.sys, "platform", "linux")
    mixer = FakeMixer()
    stage = gain.stage(WAVE, run=mixer)
    assert isinstance(stage, gain.HardwareGain)
    assert (stage.kind, stage.card, stage.control, stage.lo, stage.hi) == ("hardware", 1, "Mic", 0.0, 40.0)
    assert stage.step_db == 0.5  # 40 dB over 80 steps
    stage.set_db(12.5)
    assert mixer.calls[-1] == ["amixer", "-q", "-c", "1", "--", "sset", "Mic", "capture", "12.50dB"]
    assert stage.get_db() == 12.5
    stage.set_db(55.0)  # past the control's range
    assert mixer.calls[-1][-1] == "40.00dB"


def test_a_mic_without_a_capture_volume_gets_software_gain(monkeypatch):
    monkeypatch.setattr(gain.sys, "platform", "linux")
    mixer = FakeMixer()
    assert isinstance(gain.stage("alsa_input.pci-0000_00_1f.3.analog-stereo", run=mixer), gain.SoftwareGain)
    assert gain.stage("Not A PipeWire Node", run=mixer).kind == "software"
    soft = gain.SoftwareGain()
    soft.set_db(6.0)
    assert soft.factor == pytest.approx(10 ** (6 / 20))
    soft.set_db(99.0)
    assert soft.get_db() == gain.SOFTWARE_DB[1]


def test_macos_uses_software_gain_without_querying_a_linux_mixer(monkeypatch):
    monkeypatch.setattr(gain.sys, "platform", "darwin")
    mixer = FakeMixer()
    assert isinstance(gain.stage(WAVE, run=mixer), gain.SoftwareGain)
    assert mixer.calls == []


def test_a_mic_with_a_hardware_control_has_auto_gain_unless_its_own_entry_says_otherwise(pack_toml):
    keys = [WAVE, "wave3"]  # its device name, then its name in config/devices.toml
    assert gain.enabled(keys, hardware=True)
    assert not gain.enabled(keys, hardware=False)  # software gain waits to be turned on
    gain.save(WAVE, False)
    assert tomllib.loads(pack_toml.read_text())["audio"]["mic_auto_gain"] == {WAVE: False}
    assert not gain.enabled(keys, hardware=True)
    pack_toml.write_text("[audio.mic_auto_gain]\nwave3 = true\n")
    assert gain.enabled(keys, hardware=False)  # by its devices.toml name
    gain.save(WAVE, False)
    assert not gain.enabled(keys, hardware=False)  # the device name wins
    assert not gain.enabled(["Other"], hardware=False)


def test_the_target_is_read_live_and_checked(pack_toml):
    assert gain.target_dbfs() == -20.0
    pack_toml.write_text("[audio]\nauto_gain_target_dbfs = -26\n")
    assert gain.target_dbfs() == -26.0
    pack_toml.write_text("[audio]\nauto_gain_target_dbfs = 5\n")
    with pytest.raises(ValueError, match="-60 to -3"):
        gain.target_dbfs()


# ---- #449's review ----

BUILTIN = "alsa_input.pci-0000_00_1f.3.analog-stereo"
CARD0_SCONTENTS = """Simple mixer control 'Capture',0
  Capabilities: cvolume cswitch
  Capture channels: Front Left - Front Right
  Limits: Capture 0 - 63
  Front Left: Capture 39 [62%] [12.00dB] [on]
  Front Right: Capture 39 [62%] [12.00dB] [on]
"""
CARD0_CONTENTS = """numid=21,iface=MIXER,name='Capture Volume'
  ; type=INTEGER,access=rw---R--,values=2,min=0,max=63,step=0
  : values=39,39
  | dBscale-min=-17.25dB,step=0.75dB,mute=0
"""  # read-only from devbox's built-in mic (card 0)


def card0(args: list[str]) -> str:
    if args == ["pw-dump"]:
        return PW_DUMP
    return CARD0_SCONTENTS if args[-1] == "scontents" else CARD0_CONTENTS


def test_the_built_in_mic_uses_its_capture_control_and_its_db_scale(monkeypatch):
    monkeypatch.setattr(gain.sys, "platform", "linux")
    stage = gain.stage(BUILTIN, run=card0)
    assert isinstance(stage, gain.HardwareGain)
    assert (stage.control, stage.lo, stage.hi) == ("Capture", -17.25, pytest.approx(30.0))  # -17.25 + 63 * 0.75


def test_a_negative_gain_is_never_read_as_an_amixer_option():
    import getopt

    calls: list[list[str]] = []
    gain.HardwareGain(0, "Capture", -17.25, 30.0, run=lambda a: calls.append(a) or "").set_db(-3.0)
    _opts, rest = getopt.gnu_getopt(calls[-1][1:], "hc:D:qidnva:sRM")  # amixer's own option string
    assert rest[-1] == "-3.00dB"


SGET_BOTH = """Simple mixer control 'Mic',0
  Capabilities: pvolume pvolume-joined cvolume cvolume-joined pswitch pswitch-joined cswitch cswitch-joined
  Playback channels: Mono
  Capture channels: Mono
  Limits: Playback 0 - 31 Capture 0 - 35
  Mono: Playback 0 [0%] [-30.00dB] [off] Capture 23 [66%] [12.00dB] [on]
"""


def test_a_headsets_mic_with_a_sidetone_is_read_and_set_on_its_capture_side():
    calls: list[list[str]] = []
    stage = gain.HardwareGain(1, "Mic", 0.0, 35.0, run=lambda a: calls.append(a) or SGET_BOTH)
    assert stage.get_db() == 12.0
    stage.set_db(13.5)
    assert calls[-1][-2:] == ["capture", "13.50dB"]


class CoarseMixer:
    """A capture volume with 4 dB steps: amixer rounds a dB value to the nearest step."""

    def __init__(self) -> None:
        self.db = 0.0

    def __call__(self, args: list[str]) -> str:
        if "sset" in args:
            self.db = round(float(args[-1].removesuffix("dB")) / 4) * 4
            return ""
        return f"Mono: Capture 0 [0%] [{self.db:.2f}dB] [on]\n"


# ---- #798: devices.toml names to PipeWire nodes, and MicGain as the listen worker runs it

IPHONE = "alsa_input.usb-Apple_iPhone-00.mono-fallback"
NODES = [
    {"id": 50, "type": "PipeWire:Interface:Device", "info": {"props": {"device.vendor.id": "0x0fd9",
                                                                        "device.product.id": "0x0070"}}},
    {"id": 88, "type": "PipeWire:Interface:Node", "info": {"props": {
        "media.class": "Audio/Source", "node.name": WAVE, "node.description": "Elgato Wave 3 Mono",
        "node.nick": "Elgato Wave 3", "device.id": 50, "object.serial": 2063, "api.alsa.pcm.card": 1}}},
    {"id": 60, "type": "PipeWire:Interface:Node", "info": {"props": {
        "media.class": "Audio/Source", "node.name": BUILTIN, "node.description": "Built-in Audio Analog Stereo",
        "api.alsa.pcm.card": 0}}},
    {"id": 70, "type": "PipeWire:Interface:Node", "info": {"props": {
        "media.class": "Audio/Source", "node.name": IPHONE, "node.description": "iPhone Microphone"}}},
    {"id": 71, "type": "PipeWire:Interface:Node", "info": {"props": {
        "media.class": "Audio/Sink", "node.name": "alsa_output.usb-Elgato_Wave_3", "node.description": "Wave 3 Out"}}},
]  # fmt: skip


@pytest.mark.parametrize(
    ("match", "node"),
    [
        pytest.param({"name": "Wave 3"}, WAVE, id="name-piece-of-description"),
        pytest.param({"name": "wave 3"}, WAVE, id="name-any-case"),
        pytest.param({"usb": "0fd9:0070"}, WAVE, id="usb-from-its-device"),
        pytest.param({"path": WAVE}, WAVE, id="path-node-name"),
        pytest.param({"path": "2063"}, WAVE, id="path-serial"),
        pytest.param({"name": "Wave 3", "usb": "1234:5678"}, None, id="every-field-must-match"),
        pytest.param({"usb": "not-usb"}, None, id="bad-usb"),
        pytest.param({"name": "iPhone"}, None, id="excluded-like-capture"),
        pytest.param({"name": "Yeti"}, None, id="no-source-has-it"),
        pytest.param({}, None, id="ambiguous-is-none"),
    ],
)
def test_a_devices_toml_entry_finds_its_pipewire_node_as_capture_does(match, node):
    assert gain.pipewire_node(match, run=lambda args: json.dumps(NODES)) == node


def test_without_pipewire_no_node_is_found():
    def missing(args: list[str]) -> str:
        raise FileNotFoundError("pw-dump")

    assert gain.pipewire_node({"name": "Wave 3"}, run=missing) is None


@pytest.fixture
def wave(monkeypatch, pack_toml):
    """A MicGain for the Wave:3 (devices.toml name "wave3") over the fake mixer, with every stage it looks for."""
    monkeypatch.setattr(gain.sys, "platform", "linux")
    mixer = FakeMixer()
    found: list[str] = []

    def find(device: str) -> gain.Stage:
        found.append(device)
        return gain.stage(device, run=mixer)

    return gain.MicGain("wave3", lambda: WAVE, find), mixer, found


def test_a_quiet_wave_3_is_raised_between_utterances_and_each_move_is_logged(wave, caplog):
    g, mixer, _ = wave
    caplog.set_level("INFO", logger="liberational.audio.gain")
    g.tick(0.0, in_speech=False)  # finds the stage: a hardware control, so auto gain is on without an entry
    assert g.status() == {"name": "wave3", "auto": True, "kind": "hardware", "db": 0.0}
    for ended in (1.0, 2.0, 3.0):
        g.heard(tone(-50.0, 1.0), 16000, ended)
    g.tick(5.0, in_speech=True)  # someone is talking: never mid-sentence
    g.tick(5.5, in_speech=False)  # a second hasn't passed since the last look
    assert mixer.db == 0.0
    g.tick(6.5, in_speech=False)
    assert mixer.db == gain.UP_DB
    assert g.status()["db"] == gain.UP_DB
    (move,) = [r.getMessage() for r in caplog.records if "->" in r.getMessage()]
    assert move.startswith("wave3: auto gain +0.0 -> +1.5 dB (hardware") and "speech peaks" in move


def test_turned_off_a_hardware_control_stays_where_it_is(wave, pack_toml):
    g, mixer, _ = wave
    pack_toml.write_text(f'[audio.mic_auto_gain]\n"{WAVE}" = false\n')
    mixer.db = 30.0
    g.tick(0.0, in_speech=False)
    for ended in (1.0, 2.0, 3.0):
        g.heard(tone(-50.0, 1.0), 16000, ended)
    g.tick(10.0, in_speech=False)
    assert mixer.db == 30.0
    assert not any("sset" in c for c in mixer.calls)
    assert g.status()["auto"] is False


def test_software_gain_scales_samples_only_while_it_is_on(pack_toml):
    g = gain.MicGain("deskmic", lambda: None)  # no PipeWire node to find: software gain
    g.tick(0.0, in_speech=False)
    assert g.status() == {"name": "deskmic", "auto": False, "kind": "software", "db": None}
    pack_toml.write_text("[audio.mic_auto_gain]\ndeskmic = true\n")
    g.tick(1.0, in_speech=False)
    for ended in (2.0, 3.0, 4.0):
        g.heard(tone(-50.0, 1.0), 16000, ended)
    g.tick(8.0, in_speech=False)
    assert g.factor == pytest.approx(10 ** (gain.UP_DB / 20))
    pack_toml.write_text("[audio.mic_auto_gain]\ndeskmic = false\n")
    g.tick(9.0, in_speech=False)
    assert g.factor == 1.0  # back to as captured
    assert g.status()["db"] == 0.0


def test_a_control_that_fails_is_looked_for_again_after_a_while(wave, caplog):
    import subprocess

    g, mixer, found = wave
    g.tick(0.0, in_speech=False)
    mixer.calls.clear()

    def unplugged(args: list[str]) -> str:
        raise subprocess.CalledProcessError(1, args)

    assert isinstance(g.stage, gain.HardwareGain)
    g.stage.run = unplugged
    g.tick(gain.REREAD_S + 1, in_speech=False)  # its read back fails
    assert "auto gain: CalledProcessError" in caplog.text
    g.tick(gain.REREAD_S + 2, in_speech=False)
    assert found == [WAVE]
    g.tick(gain.REREAD_S + 2 + gain.FIND_AGAIN_S, in_speech=False)
    assert found == [WAVE, WAVE]
    assert g.status()["kind"] == "hardware"


def test_the_status_settings_reads_is_by_device_name_and_goes_stale(tmp_path, wave, monkeypatch):
    g, _, _ = wave
    g.tick(0.0, in_speech=False)
    path = tmp_path / gain.STATUS
    gain.write_status(path, [g, gain.MicGain("deskmic", lambda: None)])
    assert gain.read_status(path) == {
        WAVE: {"name": "wave3", "auto": True, "kind": "hardware", "db": 0.0},
        "deskmic": {"name": "deskmic", "auto": False, "kind": None, "db": None},
    }
    later = gain.time.time() + gain.FRESH_S + 1
    monkeypatch.setattr(gain.time, "time", lambda: later)
    assert gain.read_status(path) == {}
    assert gain.read_status(tmp_path / "missing.json") == {}


# ---- Codex on #809: software gain that stood in for a failed lookup, and macOS device names


def test_software_gain_standing_in_gives_way_to_the_hardware_control_once_it_is_found(wave, pack_toml):
    _, mixer, found = wave
    devices = iter([None, WAVE])  # PipeWire hadn't the node yet, then has
    g = gain.MicGain("wave3", lambda: next(devices), lambda device: found.append(device) or gain.stage(device, mixer))
    pack_toml.write_text("[audio.mic_auto_gain]\nwave3 = true\n")
    g.tick(0.0, in_speech=False)
    assert g.status()["kind"] == "software"
    for ended in (1.0, 2.0, 3.0):
        g.heard(tone(-50.0, 1.0), 16000, ended)
    g.tick(10.0, in_speech=False)
    assert g.factor > 1.0 and g.take_change() == gain.UP_DB
    g.tick(gain.FIND_AGAIN_S + 1, in_speech=False)
    assert g.status() == {"name": "wave3", "auto": True, "kind": "hardware", "db": 0.0}
    assert g.factor == 1.0
    assert g.take_change() == -gain.UP_DB  # the samples lost software gain's step
    assert found == [WAVE]


@pytest.mark.parametrize(
    ("match", "name"),
    [
        pytest.param({"name": "Wave 3"}, "Elgato Wave 3", id="name-piece"),
        pytest.param({"name": "iPhone"}, None, id="excluded-like-capture"),
        pytest.param({"name": "Mic"}, None, id="ambiguous"),
        pytest.param({"usb": "0fd9:0070"}, None, id="no-name-to-match"),
    ],
)
def test_on_macos_a_devices_toml_entry_finds_the_input_device_settings_shows(monkeypatch, match, name):
    monkeypatch.setattr(gain.sys, "platform", "darwin")
    names = ["MacBook Pro Microphone", "Elgato Wave 3", "iPhone Microphone", "Desk Mic"]
    assert gain.device_name(match, run=lambda args: json.dumps(names)) == name
