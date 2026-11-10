"""The Admin source picker: GET and POST /devices (device scans patched out; made-up device names)."""

import http.client
import json
import subprocess
import sys
import time
import tomllib
import types

import pytest

from liberational import config as cfg
from liberational import services
from liberational.audio import ambient
from liberational.dashboard import (
    camera_controls,
    camera_modes,
    cameras,
    capture_control,
    devices,
    privacy,
    server,
    settings,
)
from liberational.dashboard.camera_modes import modes as native_camera_modes

MIC_A, MIC_B = "alsa_input.pci-0000_00_1f.3.analog-stereo", "alsa_input.usb-Acme_Mic-00.mono-fallback"
RAW = [
    (31, "Default Source", 32, 48000.0, "PulseAudio"),
    (35, "alsa_output.pci-0000_00_1f.3.iec958-stereo.monitor", 2, 48000.0, "PulseAudio"),
    (36, MIC_A, 2, 48000.0, "PulseAudio"),
    (38, MIC_B, 1, 48000.0, "PulseAudio"),
]


@pytest.fixture
def pack_toml(tmp_path, monkeypatch):
    path = tmp_path / "pack.toml"
    monkeypatch.setattr(cfg, "local_config", lambda: path)
    return path


@pytest.fixture
def ran(tmp_path, monkeypatch):
    """The pack-services arguments of each service reinstall; ambient and vision are installed."""
    calls = []

    def run(cmd, **kw):
        calls.append(cmd[cmd.index("liberational.services") + 1 :])
        return subprocess.CompletedProcess(cmd, 0, "", "")

    units = tmp_path / "units"
    units.mkdir()
    for name in ("ambient", "vision"):
        (units / f"pack-{name}.service").touch()
    monkeypatch.setattr(services, "DARWIN", False)
    monkeypatch.setattr(services, "unit_dir", lambda: units)
    monkeypatch.setattr(devices.subprocess, "run", run)
    return calls


@pytest.fixture
def mode_calls(monkeypatch):
    calls = []

    def modes(node, format="MJPEG", fresh=False):
        calls.append((node, format, fresh))
        return {
            "status": "ok",
            "format": format,
            "fourcc": "MJPG" if format == "MJPEG" else format,
            "sizes": [],
            "choices": [{"width": 640, "height": 480, "fps": [30, 60]}, {"width": 1280, "height": 720, "fps": [30]}],
            "diagnostics": [],
        }

    monkeypatch.setattr(camera_modes, "modes", modes)
    return calls


@pytest.fixture
def camera_restarts(monkeypatch):
    calls = []

    def restart(device):
        calls.append(device)
        return capture_control.RestartResult("restarting", "camera restart scheduled")

    monkeypatch.setattr(capture_control, "restart_camera", restart)
    return calls


@pytest.fixture
def base(serve, tmp_path, monkeypatch, pack_toml, ran, manifest, mode_calls, camera_restarts):
    state = tmp_path / "state"
    state.mkdir()
    now = time.time()
    (state / "ambient.status.json").write_text(
        json.dumps({"ts": now, "devices": {"a": {"name": MIC_A, "level_dbfs": -50.0}}})
    )
    monkeypatch.setenv("PACK_PATHS_STATE", str(state))
    monkeypatch.setattr(server, "DASH_PASSWORD", "")
    monkeypatch.setattr(devices, "_mic_discovery", None)
    monkeypatch.setattr(ambient, "scan_raw_devices", lambda: RAW)
    monkeypatch.setattr(ambient, "EXCLUDE_SUBSTR", [])
    # Two modules read the platform: devices.py (PipeWire names) and audio.ambient.select_input_devices, which only
    # drops monitors and the "Default Source" alias on Linux. Patch both so the Linux filtering is reproducible on a Mac.
    linux = types.SimpleNamespace(platform="linux", executable=sys.executable)
    monkeypatch.setattr(devices, "sys", linux)
    monkeypatch.setattr(ambient, "sys", linux)
    monkeypatch.setattr(devices, "_pipewire_names", lambda: {MIC_B: "Acme Mic"})
    monkeypatch.setattr(devices, "_mic_freshness", lambda: {"acme"})  # acme.audio has audio on the broker
    monkeypatch.setattr(
        devices, "_linux_cameras", lambda: [(0, "Acme Cam", "/dev/video0"), (2, "Other Cam", "/dev/video2")]
    )
    return serve(server.H)


MANIFEST = """# capture's devices
[video]
fps = 30

[[device]]
name = "cam0"
kind = "camera"
match = { usb = "1234:0001" }  # Acme Cam

[[device]]
name = "acme"
kind = "microphone"
match = { name = "Acme Mic" }
"""


@pytest.fixture
def manifest(tmp_path, monkeypatch):
    path = tmp_path / "devices.toml"
    path.write_text(MANIFEST)
    monkeypatch.setattr(cameras, "manifest_path", lambda: path)
    usb = {"/dev/video0": "1234:0001", "/dev/video2": "1234:0002"}
    monkeypatch.setattr(cameras, "usb_id", lambda p: usb.get(p))
    links = {"/dev/video0": ["/dev/v4l/by-path/usb-0-1"], "/dev/video2": ["/dev/v4l/by-path/usb-0-2"]}
    monkeypatch.setattr(cameras, "stable_paths", lambda p: links.get(p, []))
    monkeypatch.setattr(
        cameras, "states", lambda link: {"ts": time.time(), "cameras": {"cam0": {"state": "on", "size": [1280, 720]}}}
    )
    monkeypatch.setattr(devices, "_link", lambda: None)
    return path


@pytest.fixture
def restarted(monkeypatch):
    """The services Settings would restart, recorded instead."""
    asked: list[list[str]] = []

    def restart(log, names=(), dashboard=True, before=None):
        asked.append(list(names))

    monkeypatch.setattr(settings, "restart", restart)
    return asked


def use(base, body, **headers):
    return call(base, "POST", body, path="/devices/camera", **headers)


def test_a_camera_is_used_when_capture_s_manifest_matches_it(base, manifest):
    cams = {c["index"]: c for c in call(base, "GET")[1]["cameras"]}
    assert (cams[0]["selected"], cams[0]["stream"], cams[0]["in_use"]) == (True, "cam0", True)
    assert (cams[2]["selected"], cams[2]["stream"], cams[2]["in_use"]) == (False, None, False)


def test_using_a_camera_adds_it_to_the_manifest_and_restarts_capture(base, manifest, restarted):
    code, v = use(base, {"index": 2, "on": True})
    assert code == 200 and restarted == [["continuum-capture"]]
    cams = {c["index"]: c for c in v["cameras"]}
    assert (cams[2]["selected"], cams[2]["stream"]) == (True, "cam1")
    text = manifest.read_text()
    assert text.startswith(MANIFEST.rstrip("\n")), "the rest of the file, comments included, stays as it was"
    assert tomllib.loads(text)["device"][-1] == {"name": "cam1", "kind": "camera", "match": {"usb": "1234:0002"}}


def test_leaving_a_camera_out_removes_its_block_only(base, manifest, restarted):
    use(base, {"index": 2, "on": True})
    code, v = use(base, {"index": 0, "on": False})
    assert code == 200
    text = manifest.read_text()
    assert [d["name"] for d in tomllib.loads(text)["device"]] == ["acme", "cam1"]
    assert "# capture's devices" in text and "fps = 30" in text
    assert {c["index"]: c["selected"] for c in v["cameras"]} == {0: False, 2: True}


def test_the_last_camera_in_use_stays(base, manifest, restarted):
    assert use(base, {"index": 0, "on": False})[0] == 409
    assert manifest.read_text() == MANIFEST and restarted == []


@pytest.mark.parametrize("body", [{"index": 0}, {"index": "0", "on": True}, {"index": True, "on": True}])
def test_a_bad_camera_choice_is_refused(base, manifest, restarted, body):
    assert use(base, body)[0] == 400
    assert manifest.read_text() == MANIFEST


def test_a_camera_choice_from_elsewhere_is_refused(base, manifest, restarted):
    assert use(base, {"index": 2, "on": True}, **{"X-Forwarded-For": "100.64.0.9"})[0] == 403
    assert manifest.read_text() == MANIFEST


def call(base, method, body=None, path="/devices", **headers):
    if body is not None:
        headers.setdefault("Content-Type", "application/json")
    c = http.client.HTTPConnection(base.removeprefix("http://"), timeout=10)
    c.request(method, path, json.dumps(body) if body is not None else None, headers)
    r = c.getresponse()
    out = r.status, json.loads(r.read())
    c.close()
    return out


def test_lists_real_inputs_with_what_is_in_use(base, manifest):
    code, v = call(base, "GET")
    assert code == 200
    assert [m["id"] for m in v["mics"]] == [MIC_A, MIC_B]  # no monitors, no Default Source
    a, b = v["mics"]
    assert (a["selected"], a["stream"], a["in_use"]) == (False, None, False)
    assert (b["selected"], b["stream"], b["in_use"], b["name"]) == (True, "acme", True, "Acme Mic")
    assert not call(base, "GET", **{"X-Forwarded-For": "100.64.0.9"})[1]["control"]


def sensitivity(base, body, **headers):
    return call(base, "POST", body, path="/devices/sensitivity", **headers)


def test_a_mics_sensitivity_is_saved_for_that_mic_without_a_restart(base, pack_toml, ran):
    code, v = sensitivity(base, {"mic": MIC_B, "sensitivity": 85})
    assert code == 200, v
    # Under its manifest name, which the listen worker reads it by
    assert tomllib.loads(pack_toml.read_text())["audio"]["mic_sensitivity"] == {"acme": 85}
    rows = {m["id"]: m for m in v["mics"]}
    assert (rows[MIC_B]["sensitivity"], rows[MIC_B]["sensitivity_custom"]) == (85, True)
    assert (rows[MIC_A]["sensitivity"], rows[MIC_A]["sensitivity_custom"]) == (70, False)  # audio.sensitivity
    assert ran == []  # the microphone service follows it live
    sensitivity(base, {"mic": MIC_B, "sensitivity": None})  # back to the default
    assert "audio" not in tomllib.loads(pack_toml.read_text())


def test_the_default_sensitivity_is_saved_for_every_mic_without_its_own_and_without_a_restart(base, pack_toml, ran):
    sensitivity(base, {"mic": MIC_B, "sensitivity": 85})
    code, v = sensitivity(base, {"default": 40})
    assert code == 200, v
    assert tomllib.loads(pack_toml.read_text())["audio"]["sensitivity"] == 40
    rows = {m["id"]: m for m in v["mics"]}
    assert (rows[MIC_A]["sensitivity"], rows[MIC_B]["sensitivity"], v["default_sensitivity"]) == (40, 85, 40)
    assert ran == []  # the microphone service follows it live
    assert call(base, "GET")[1]["default_sensitivity"] == 40


@pytest.mark.parametrize("value", [101, -1, 50.5, True, "50", None])
def test_a_bad_default_sensitivity_is_refused(base, pack_toml, value):
    assert sensitivity(base, {"default": value})[0] == 400
    assert not pack_toml.exists()


@pytest.mark.parametrize(
    "body",
    [
        {"mic": MIC_A, "sensitivity": 101},
        {"mic": MIC_A, "sensitivity": 50.5},
        {"mic": MIC_A, "sensitivity": True},
        {"mic": "Not A Mic", "sensitivity": 50},
        {"sensitivity": 50},
    ],
)
def test_a_bad_sensitivity_is_refused(base, pack_toml, body):
    assert sensitivity(base, body)[0] == 400
    assert not pack_toml.exists()


def test_a_sensitivity_from_elsewhere_is_refused(base, pack_toml):
    ok = {"mic": MIC_A, "sensitivity": 50}
    assert sensitivity(base, ok, **{"X-Forwarded-For": "100.64.0.9"})[0] == 403
    assert sensitivity(base, ok, Host="evil.example")[0] == 403
    assert sensitivity(base, ok, Origin="http://evil.example")[0] == 403
    assert sensitivity(base, ok, **{"Content-Type": "text/plain"})[0] == 403
    assert not pack_toml.exists()


def test_the_devices_are_chosen_through_settings_not_here(base, pack_toml, ran):
    assert call(base, "POST", {"mics": [MIC_A], "cameras": [0]})[0] == 404  # the Sources drawer's save is gone
    assert not pack_toml.exists()


def auto_gain(base, body, **headers):
    return call(base, "POST", body, path="/devices/auto_gain", **headers)


def test_auto_gain_is_turned_on_for_one_mic_without_a_restart(base, pack_toml, ran):
    assert {m["id"]: m["auto_gain"] for m in call(base, "GET")[1]["mics"]} == {MIC_A: False, MIC_B: False}
    code, v = auto_gain(base, {"mic": MIC_B, "on": True})
    assert code == 200, v
    assert tomllib.loads(pack_toml.read_text())["audio"]["mic_auto_gain"] == {"acme": True}  # its manifest name
    assert {m["id"]: m["auto_gain"] for m in v["mics"]} == {MIC_A: False, MIC_B: True}
    assert ran == []  # the microphone service follows it live
    auto_gain(base, {"mic": MIC_B, "on": False})
    # Kept as false: a mic with a hardware control has auto gain without an entry.
    assert tomllib.loads(pack_toml.read_text())["audio"]["mic_auto_gain"] == {"acme": False}


def test_auto_gain_shows_the_listen_workers_gain_and_is_on_for_a_hardware_control(base, pack_toml):
    from liberational.audio import gain

    status = {
        MIC_A: {"name": "builtin", "auto": False, "kind": "software", "db": 0.0},
        MIC_B: {"name": "acme", "auto": True, "kind": "hardware", "db": 30.0},
    }
    state = cfg.path("paths.state")
    (state / gain.STATUS).write_text(json.dumps({"ts": time.time(), "mics": status}))
    mics = {m["id"]: m for m in call(base, "GET")[1]["mics"]}
    assert {n: (m["auto_gain"], m["gain"]) for n, m in mics.items()} == {
        MIC_A: (False, status[MIC_A]),
        MIC_B: (True, status[MIC_B]),
    }
    pack_toml.write_text("[audio.mic_auto_gain]\nacme = false\n")  # by its config/devices.toml name
    assert {m["id"]: m["auto_gain"] for m in call(base, "GET")[1]["mics"]} == {MIC_A: False, MIC_B: False}


@pytest.mark.parametrize(
    "body",
    [{"mic": MIC_A, "on": "yes"}, {"mic": MIC_A}, {"mic": "Not A Mic", "on": True}, {"on": True}],
)
def test_a_bad_auto_gain_is_refused(base, pack_toml, body):
    assert auto_gain(base, body)[0] == 400
    assert not pack_toml.exists()


def test_auto_gain_from_elsewhere_is_refused(base, pack_toml):
    assert auto_gain(base, {"mic": MIC_A, "on": True}, **{"X-Forwarded-For": "100.64.0.9"})[0] == 403
    assert not pack_toml.exists()


def mic(base, body, **headers):
    return call(base, "POST", body, path="/devices/mic", **headers)


@pytest.mark.parametrize("kind,prefix", [("camera", "cam"), ("microphone", "mic")])
def test_use_reenables_retained_device_without_losing_configuration(tmp_path, kind, prefix):
    path = tmp_path / "devices.toml"
    original = (
        f'[[device]]\nname = "{prefix}0"\nkind = "{kind}"\nenabled = false # retained\n'
        'match = { name = "Acme" }\n[device.video]\nflip = true\n'
    )
    path.write_text(original)
    assert cameras.manifest_devices(kind, path) == []
    assert cameras.use("Acme", None, True, path, kind=kind) == f"{prefix}0"
    assert path.read_text() == original.replace("enabled = false", "enabled = true")
    assert cameras.manifest_devices(kind, path) == [{"name": f"{prefix}0", "match": {"name": "Acme"}}]


@pytest.mark.parametrize("kind,prefix", [("camera", "cam"), ("microphone", "mic")])
def test_use_reserves_disabled_device_ids_when_adding_another(tmp_path, kind, prefix):
    path = tmp_path / "devices.toml"
    path.write_text(f'[[device]]\nname = "{prefix}0"\nkind = "{kind}"\nenabled = false\nmatch = {{ name = "Acme" }}\n')
    assert cameras.use("Other", None, True, path, kind=kind) == f"{prefix}1"
    rows = tomllib.loads(path.read_text())["device"]
    assert [row["name"] for row in rows] == [f"{prefix}0", f"{prefix}1"]
    assert rows[0]["enabled"] is False


def test_listening_with_a_microphone_adds_it_to_the_manifest_by_name(base, manifest, restarted):
    code, v = mic(base, {"id": MIC_A, "on": True})
    assert code == 200 and restarted == [["continuum-capture"]]
    added = tomllib.loads(manifest.read_text())["device"][-1]
    assert added == {"name": "mic0", "kind": "microphone", "match": {"name": MIC_A}}
    assert {m["id"]: m["stream"] for m in v["mics"]} == {MIC_A: "mic0", MIC_B: "acme"}


def test_the_last_microphone_in_use_stays_and_another_can_go(base, manifest, restarted):
    assert mic(base, {"id": MIC_B, "on": False})[0] == 409
    mic(base, {"id": MIC_A, "on": True})
    code, _ = mic(base, {"id": MIC_B, "on": False})
    assert code == 200
    assert [d["name"] for d in tomllib.loads(manifest.read_text())["device"]] == ["cam0", "mic0"]


def test_a_camera_chosen_by_its_stable_path_is_used(base, manifest):
    manifest.write_text(
        MANIFEST.replace('match = { usb = "1234:0001" }', 'match = { path = "/dev/v4l/by-path/usb-0-2" }')
    )
    cams = {c["index"]: c for c in call(base, "GET")[1]["cameras"]}
    assert (cams[0]["selected"], cams[2]["selected"], cams[2]["stream"]) == (False, True, "cam0")


def test_two_cameras_with_one_usb_id_are_chosen_by_their_paths(base, manifest, restarted, monkeypatch):
    monkeypatch.setattr(cameras, "usb_id", lambda p: "1234:0001")  # two of the same model
    manifest.write_text(
        MANIFEST.replace('match = { usb = "1234:0001" }', 'match = { path = "/dev/v4l/by-path/usb-0-1" }')
    )
    code, v = use(base, {"index": 2, "on": True})
    assert code == 200
    added = tomllib.loads(manifest.read_text())["device"][-1]
    assert added["match"] == {"path": "/dev/v4l/by-path/usb-0-2"}, "not the shared USB id, which matches both"
    assert {c["index"]: c["stream"] for c in v["cameras"]} == {0: "cam0", 2: "cam1"}


def test_a_camera_with_its_own_subtable_is_removed_whole(base, manifest, restarted):
    use(base, {"index": 2, "on": True})
    text = manifest.read_text()
    manifest.write_text(
        text.replace(
            'kind = "camera"\nmatch = { usb = "1234:0001" }',
            'kind = "camera"\nmatch = { usb = "1234:0001" }\n[device.video]\nfps = 15',
            1,
        )
    )
    assert tomllib.loads(manifest.read_text())["device"][0]["video"] == {"fps": 15}
    code, _ = use(base, {"index": 0, "on": False})
    assert code == 200
    assert [d["name"] for d in tomllib.loads(manifest.read_text())["device"]] == ["acme", "cam1"]
    assert "fps = 15" not in manifest.read_text()


def test_repeated_inventory_reads_reuse_discovery_but_refresh_live_state(base, monkeypatch, pack_toml):
    from unittest.mock import create_autospec

    scan = create_autospec(ambient.scan_raw_devices, return_value=RAW)
    names = create_autospec(devices._pipewire_names, return_value={MIC_B: "Acme Mic"})
    monkeypatch.setattr(ambient, "scan_raw_devices", scan)
    monkeypatch.setattr(devices, "_pipewire_names", names)
    first = call(base, "GET")[1]
    pack_toml.write_text("[audio.mic_sensitivity]\nacme = 85\n")
    monkeypatch.setattr(devices, "_mic_freshness", lambda: set())
    second = call(base, "GET")[1]
    mic = next(row for row in second["mics"] if row["stream"] == "acme")
    assert first["mics"][1]["in_use"] is True
    assert (mic["in_use"], mic["sensitivity"]) == (False, 85)
    scan.assert_called_once_with()
    names.assert_called_once_with()


def rescanned() -> None:
    """Wait out a background scan of a stale discovery."""
    rescan = devices._rescan
    if rescan is not None:
        rescan.join(5)


def test_a_stale_microphone_discovery_answers_at_once_and_scans_again_behind_it(base, monkeypatch):
    from unittest.mock import create_autospec

    now = [100.0]
    monkeypatch.setattr(devices, "time", types.SimpleNamespace(monotonic=lambda: now[0]))
    scan = create_autospec(ambient.scan_raw_devices, side_effect=[RAW, []])
    monkeypatch.setattr(ambient, "scan_raw_devices", scan)
    assert devices._discover_mics()[0] == RAW
    now[0] = 101.0
    assert devices._discover_mics()[0] == RAW
    assert scan.call_count == 1, "still fresh"
    now[0] = 102.0
    assert devices._discover_mics()[0] == RAW, "stale: the last scan, at once (#2568)"
    rescanned()
    assert scan.call_count == 2
    assert devices._discover_mics()[0] == [], "the scan behind it"


def test_stale_reads_share_one_background_scan(base, monkeypatch):
    import threading

    now = [100.0]
    monkeypatch.setattr(devices, "time", types.SimpleNamespace(monotonic=lambda: now[0]))
    started, release = threading.Event(), threading.Event()
    scans = []

    def scan():
        scans.append(1)
        if len(scans) > 1:
            started.set()
            assert release.wait(5)
        return RAW

    monkeypatch.setattr(ambient, "scan_raw_devices", scan)
    devices._discover_mics()
    now[0] = 103.0
    for _ in range(5):
        assert devices._discover_mics()[0] == RAW
    assert started.wait(5)
    release.set()
    rescanned()
    assert len(scans) == 2


def test_a_fresh_scan_is_not_overwritten_by_an_older_background_one(base, monkeypatch):
    import threading

    now = [100.0]
    monkeypatch.setattr(devices, "time", types.SimpleNamespace(monotonic=lambda: now[0]))
    started, release = threading.Event(), threading.Event()
    results = iter([RAW, "background", []])

    def scan():
        got = next(results)
        if got == "background":
            started.set()
            assert release.wait(5)
            return [("old", "older scan", 1, 48000.0, "ALSA")]
        return got

    monkeypatch.setattr(ambient, "scan_raw_devices", scan)
    devices._discover_mics()
    now[0] = 103.0
    devices._discover_mics()  # starts the background scan, which waits
    assert started.wait(5)
    now[0] = 104.0
    assert devices._discover_mics(fresh=True)[0] == [], "a mutation's fresh scan, at once"
    now[0] = 103.5  # the background scan finished with a time before the fresh one's
    release.set()
    rescanned()
    assert devices._discover_mics()[0] == []


def test_a_failed_background_scan_keeps_the_last_discovery(base, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(devices, "time", types.SimpleNamespace(monotonic=lambda: now[0]))
    results = iter([RAW])

    def scan():
        try:
            return next(results)
        except StopIteration:
            raise RuntimeError("sounddevice crashed") from None

    monkeypatch.setattr(ambient, "scan_raw_devices", scan)
    devices._discover_mics()
    now[0] = 103.0
    first = devices._rescan
    rescanned()
    assert devices._discover_mics()[0] == RAW
    assert devices._rescan is not None and devices._rescan is not first, "a failure doesn't stop the next scan"
    rescanned()


def test_selecting_a_microphone_refreshes_discovery_even_while_cached(base, monkeypatch, restarted):
    call(base, "GET")
    monkeypatch.setattr(ambient, "scan_raw_devices", lambda: [row for row in RAW if row[1] != MIC_A])
    code, result = call(base, "POST", {"id": MIC_A, "on": True}, path="/devices/mic")
    assert (code, result["error"]) == (404, "no such microphone")
    assert restarted == []


def test_flip_only_updates_the_selected_camera(base, manifest, restarted, camera_restarts):
    manifest.write_text(
        MANIFEST
        + '\n[[device]]\nname="cam1"\nkind="camera"\nmatch={usb="1234:0002"}\n[device.video]\nwidth=640\nflip=false\n'
    )
    status, reply = call(base, "POST", {"index": 0, "flip": True}, path="/devices/camera")
    assert status == 200
    assert reply["restart"]["status"] == "restarting"
    assert reply["restart"]["device"] == "cam0"
    assert cameras.video_settings("cam0") == {"fps": 30, "flip": True}
    assert cameras.video_settings("cam1") == {"fps": 30, "width": 640, "flip": False}
    assert next(c for c in reply["cameras"] if c["stream"] == "cam0")["flip"] is True
    assert restarted == []
    assert camera_restarts == ["cam0"]


@pytest.mark.parametrize(
    "body", [{"index": 0, "flip": 1}, {"index": 0, "flip": True, "on": True}, {"index": True, "flip": False}]
)
def test_invalid_flip_does_not_write_or_restart(base, manifest, restarted, body):
    assert call(base, "POST", body, path="/devices/camera")[0] == 400
    assert manifest.read_text() == MANIFEST
    assert restarted == []


def test_camera_video_defaults_are_visible_without_writing_the_manifest(base, manifest):
    row = call(base, "GET")[1]["cameras"][0]
    assert row["video"] == {
        "width": 1280,
        "height": 720,
        "fps": 30,
        "format": "MJPEG",
        "auto_exposure": True,
        "exposure_target": 110,
    }
    assert row["exposure_supported"] is True
    assert row["modes"]["choices"][0] == {"width": 640, "height": 480, "fps": [30, 60]}
    assert manifest.read_text() == MANIFEST


@pytest.mark.parametrize(
    "body,field,expected",
    [
        ({"index": 0, "auto_exposure": False}, "auto_exposure", False),
        ({"index": 0, "auto_exposure": True}, "auto_exposure", True),
        ({"index": 0, "exposure_target": 60}, "exposure_target", 60),
        ({"index": 0, "exposure_target": 180}, "exposure_target", 180),
    ],
)
def test_exposure_settings_roundtrip_and_restart_only_that_camera(
    base, manifest, restarted, camera_restarts, body, field, expected
):
    status, reply = use(base, body)
    assert status == 200
    assert cameras.video_settings("cam0")[field] == expected
    assert reply["cameras"][0]["video"][field] == expected
    assert call(base, "GET")[1]["cameras"][0]["video"][field] == expected
    assert reply["restart"] == {"status": "restarting", "device": "cam0", "message": "camera restart scheduled"}
    assert reply["restart_all_available"] is False
    assert camera_restarts == ["cam0"]
    assert restarted == []


def test_resolution_save_uses_configured_format_and_preserves_fallback_and_other_camera(
    base, manifest, mode_calls, camera_restarts, restarted
):
    manifest.write_text(
        MANIFEST.replace("fps = 30", 'fps = 30\nformat = "UYVY"\nfallback = [[960, 540], [640, 480]]')
        + '\n[[device]]\nname="cam1"\nkind="camera"\nmatch={usb="1234:0002"}\nvideo={width=800, height=600, fps=15}\n'
    )
    status, reply = use(base, {"index": 0, "width": 640, "height": 480, "fps": 60})
    assert status == 200
    doc = tomllib.loads(manifest.read_text())
    assert doc["device"][0]["video"] == {"width": 640, "height": 480, "fps": 60}
    assert doc["video"]["fallback"] == [[960, 540], [640, 480]]
    assert next(d for d in doc["device"] if d["name"] == "cam1")["video"] == {"width": 800, "height": 600, "fps": 15}
    assert reply["cameras"][0]["video"] == {
        "width": 640,
        "height": 480,
        "fps": 60,
        "format": "UYVY",
        "auto_exposure": True,
        "exposure_target": 110,
    }
    assert ("/dev/video0", "UYVY", True) in mode_calls
    assert camera_restarts == ["cam0"]
    assert restarted == []


@pytest.mark.parametrize(
    "body",
    [
        {"index": 0, "auto_exposure": 1},
        {"index": 0, "exposure_target": True},
        {"index": 0, "exposure_target": 59},
        {"index": 0, "exposure_target": 181},
        {"index": 0, "exposure_target": 110.0},
        {"index": 0, "auto_exposure": False, "exposure_target": 100},
        {"index": 0, "width": 123, "height": 456, "fps": 30},
        {"index": 0, "width": 640, "height": 480, "fps": 29},
        {"index": 0, "width": 640, "height": 480, "fps": 30.0},
        {"index": 0, "width": 640},
        {"index": 0, "flip": True, "width": 640},
    ],
)
def test_bad_camera_preferences_do_not_write_or_restart(base, manifest, camera_restarts, restarted, body):
    assert use(base, body)[0] == 400
    assert manifest.read_text() == MANIFEST
    assert camera_restarts == []
    assert restarted == []


@pytest.mark.parametrize("body", [{"index": 2, "auto_exposure": False}, {"index": -1, "exposure_target": 100}])
def test_unselected_or_absent_camera_preferences_are_refused(base, manifest, camera_restarts, body):
    assert use(base, body)[0] == 404
    assert manifest.read_text() == MANIFEST
    assert not camera_restarts


@pytest.mark.parametrize("status", ["failed", "unavailable"])
def test_failed_target_restart_keeps_saved_settings_and_offers_explicit_fallback(
    base, manifest, restarted, monkeypatch, status
):
    monkeypatch.setattr(
        capture_control, "restart_camera", lambda device: capture_control.RestartResult(status, "request failed")
    )
    code, reply = use(base, {"index": 0, "auto_exposure": False})
    assert code == 200 and reply["ok"] is True
    assert cameras.video_settings("cam0")["auto_exposure"] is False
    assert reply["cameras"][0]["video"]["auto_exposure"] is False
    assert reply["restart"] == {"status": status, "device": "cam0", "message": "request failed"}
    assert reply["restart_all_available"] is True
    assert restarted == []


def test_explicit_capture_restart_is_the_only_new_control_that_restarts_all_capture(base, restarted):
    code, reply = call(base, "POST", {}, path="/devices/capture/restart")
    assert code == 200
    assert restarted == [["continuum-capture"]]
    assert reply["restart"]["status"] == "restarting"
    assert reply["restart"]["device"] == "all"


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "http://evil.example"},
        {"Host": "evil.example"},
        {"X-Forwarded-For": "100.64.0.9"},
        {"Content-Type": "text/plain"},
    ],
)
def test_explicit_capture_restart_requires_local_same_origin_json(base, restarted, headers):
    assert call(base, "POST", {}, path="/devices/capture/restart", **headers)[0] == 403
    assert restarted == []


def test_explicit_capture_restart_rejects_nonempty_body(base, restarted):
    assert call(base, "POST", {"device": "cam0"}, path="/devices/capture/restart")[0] == 400
    assert restarted == []


def test_mac_controls_are_unavailable_without_discovery_or_writes(
    base, manifest, monkeypatch, mode_calls, camera_restarts
):
    monkeypatch.setattr(devices, "sys", types.SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(devices, "_mac_cameras", lambda: [(0, "Acme Cam", None)])
    monkeypatch.setattr(cameras, "usb_id", lambda _: "1234:0001")
    row = devices.cameras()[0][0]
    assert row["exposure_supported"] is False
    assert "Linux" in row["exposure_note"]
    assert row["modes"]["status"] == "unavailable"
    assert mode_calls == []
    assert use(base, {"index": 0, "auto_exposure": False})[0] == 400
    assert manifest.read_text() == MANIFEST
    assert not camera_restarts


def test_existing_fractional_exposure_target_is_read_without_rounding(base, manifest):
    manifest.write_text(MANIFEST.replace("fps = 30", "fps = 30\nexposure_target = 110.5"))
    assert call(base, "GET")[1]["cameras"][0]["video"]["exposure_target"] == 110.5


def test_invalid_configured_format_keeps_camera_visible_with_actionable_mode_diagnostic(base, manifest, monkeypatch):
    manifest.write_text(MANIFEST.replace("fps = 30", 'fps = 30\nformat = "BAD"'))

    monkeypatch.setattr(camera_modes, "modes", native_camera_modes)
    code, reply = call(base, "GET")
    assert code == 200
    row = reply["cameras"][0]
    assert row["stream"] == "cam0" and row["video"]["format"] == "BAD"
    assert row["modes"]["status"] == "unavailable"
    assert row["modes"]["choices"] == []
    assert "BAD" in row["modes"]["diagnostics"][0]
    assert "four-byte" in row["modes"]["diagnostics"][0]


def test_absent_camera_shows_preferences_but_cannot_be_changed(
    base, manifest, monkeypatch, camera_restarts, mode_calls
):
    monkeypatch.setattr(devices, "_linux_cameras", list)
    row = call(base, "GET")[1]["cameras"][0]
    assert row["present"] is False
    assert row["video"]["width"] == 1280
    assert row["modes"]["status"] == "unavailable"
    assert mode_calls == []
    assert use(base, {"index": -1, "auto_exposure": False})[0] == 404
    assert manifest.read_text() == MANIFEST
    assert camera_restarts == []


def test_unavailable_modes_cannot_be_saved_as_arbitrary_dimensions(base, manifest, monkeypatch, camera_restarts):
    monkeypatch.setattr(
        camera_modes,
        "modes",
        lambda *args, **kwargs: {
            "status": "unavailable",
            "format": "MJPEG",
            "fourcc": "MJPG",
            "sizes": [],
            "choices": [],
            "diagnostics": ["driver unavailable"],
        },
    )
    assert use(base, {"index": 0, "width": 640, "height": 480, "fps": 30})[0] == 400
    assert manifest.read_text() == MANIFEST
    assert camera_restarts == []


def test_camera_preference_write_failure_does_not_restart(base, manifest, monkeypatch, camera_restarts):
    def fail(*args, **kwargs):
        raise OSError("manifest is read-only")

    monkeypatch.setattr(cameras, "update_video", fail)
    code, reply = use(base, {"index": 0, "auto_exposure": False})
    assert code == 400 and "read-only" in reply["error"]
    assert manifest.read_text() == MANIFEST
    assert camera_restarts == []


def test_camera_exposure_from_another_origin_does_not_write_or_restart(base, manifest, camera_restarts):
    assert use(base, {"index": 0, "auto_exposure": False}, Origin="http://evil.example")[0] == 403
    assert manifest.read_text() == MANIFEST
    assert camera_restarts == []


def test_explicit_capture_restart_failure_still_refreshes_settings_and_keeps_action(base, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("restart unavailable")

    monkeypatch.setattr(settings, "restart", fail)
    code, reply = call(base, "POST", {}, path="/devices/capture/restart")
    assert code == 200
    assert reply["cameras"][0]["video"]["auto_exposure"] is True
    assert reply["restart"] == {"status": "failed", "device": "all", "message": "restart unavailable"}
    assert reply["restart_all_available"] is True


def test_retired_room_is_not_effective_without_migration(pack_toml):
    pack_toml.write_text('[vision]\nroom="old room"\nflip=true\n')
    cfg.load.cache_clear()
    assert "room" not in cfg.load()["vision"]
    assert "flip" not in cfg.load()["vision"]


def test_rtsp_camera_is_in_live_status_without_becoming_a_usb_match(tmp_path, monkeypatch):
    manifest = tmp_path / "devices.toml"
    monkeypatch.setattr(cameras, "manifest_path", lambda: manifest)
    manifest.write_text('[[device]]\nname="entryway"\nkind="rtsp"\nrtsp={url="rtsp://192.168.1.20/main"}\n')
    monkeypatch.setattr(cameras, "_cache", {"at": -1e9, "value": None})
    monkeypatch.setattr("continuum_py.decode_header", lambda value: {"capture": {"width": 640, "height": 480}})
    queried = []

    class Link:
        def call(self, request):
            return request(self)

        def query(self, stream, lo, hi):
            queried.append(stream)
            return {"items": [{"header": b"synthetic"}]}

    reply = cameras.states(Link())
    assert reply["cameras"]["entryway"]["state"] == "on"
    assert reply["cameras"]["entryway"]["size"] == [640, 480]
    assert queried == ["entryway.video"]
    assert cameras.manifest_cameras() == []  # Network sources never match a physical USB camera.


@pytest.fixture
def native_controls(monkeypatch):
    state = {"value": 4}
    monkeypatch.setattr(camera_controls, "sys", types.SimpleNamespace(platform="linux"))

    def discover(path):
        assert path == "/dev/video0"
        return [
            {
                "id": 9963777,
                "name": "Contrast",
                "type": "integer",
                "min": 0,
                "max": 10,
                "step": 2,
                "default": 2,
                "current": state["value"],
                "disabled": False,
            },
            {
                "id": 9963800,
                "name": "Power line",
                "type": "menu",
                "min": 0,
                "max": 2,
                "step": 1,
                "default": 2,
                "current": 2,
                "disabled": False,
                "options": [{"value": 0, "label": "Off"}, {"value": 2, "label": "60 Hz"}],
            },
            {
                "id": 9963776,
                "name": "Brightness",
                "type": "integer",
                "min": 0,
                "max": 255,
                "step": 1,
                "default": 128,
                "current": 128,
                "disabled": False,
            },
            {
                "id": 9963788,
                "name": "Automatic white balance",
                "type": "boolean",
                "min": 0,
                "max": 1,
                "step": 1,
                "default": 1,
                "current": 1,
                "disabled": False,
            },
        ]

    monkeypatch.setattr(camera_controls, "native_controls", discover)
    return state


def test_live_controls_readback_save_pending_and_reset_without_restart(
    base, manifest, native_controls, camera_restarts, restarted
):
    route = "/devices/camera-controls"
    code, report = call(base, "GET", path=route + "?camera=cam0")
    assert code == 200 and report["available"] and not report["pending"]
    assert report["controls"][2]["disabled"]  # capture's automatic steering owns brightness
    assert manifest.read_text() == MANIFEST
    code, report = call(base, "POST", {"camera": "cam0", "values": {"9963777": 6}}, path=route)
    assert code == 200 and report["pending"]
    assert report["controls"][0]["current"] == 4  # saved is not applied readback
    assert cameras.video_settings("cam0")["controls"] == {"9963777": 6}
    assert call(base, "GET", path=route + "?camera=cam0")[1]["pending"]
    native_controls["value"] = 6
    assert not call(base, "GET", path=route + "?camera=cam0")[1]["pending"]
    assert call(base, "POST", {"camera": "cam0", "auto_exposure": False}, path=route)[0] == 200
    assert call(base, "POST", {"camera": "cam0", "reset": True}, path=route)[0] == 200
    assert cameras.video_settings("cam0")["controls"] == {
        "9963777": 2,
        "9963800": 2,
        "9963776": 128,
        "9963788": 1,
    }
    assert cameras.video_settings("cam0")["auto_exposure"] is False
    assert camera_restarts == restarted == []


@pytest.mark.parametrize(
    "values",
    [
        {"9963777": 3},
        {"9963777": 12},
        {"9963777": True},
        {"9963800": 1},
        {"9963776": 100},
        {"999": 1},
        {"09963777": 2},
        {"9963788": 2},
    ],
)
def test_live_control_invalid_values_do_not_persist(base, manifest, native_controls, values):
    code, _ = call(base, "POST", {"camera": "cam0", "values": values}, path="/devices/camera-controls")
    assert code == 400 and manifest.read_text() == MANIFEST


def test_live_controls_do_not_query_broker_or_modes(base, native_controls, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("camera control readback must not query the broker or camera modes")

    monkeypatch.setattr(devices, "_link", forbidden)
    monkeypatch.setattr(camera_modes, "modes", forbidden)
    code, report = call(base, "GET", path="/devices/camera-controls?camera=cam0")
    assert code == 200 and report["available"]


def test_control_camera_resolver_keeps_stable_path_and_first_manifest_match(base, manifest, native_controls):
    manifest.write_text(
        '[[device]]\nname="first"\nkind="camera"\nmatch={path="/dev/v4l/by-path/usb-0-1"}\n'
        '[[device]]\nname="second"\nkind="camera"\nmatch={usb="1234:0001"}\n'
    )
    assert devices.camera_control_node("first") == "/dev/video0"
    assert devices.camera_control_node("second") is None


@pytest.mark.parametrize("index,default", [(0, 3), (1, 1)])
def test_control_reset_rejects_invalid_advertised_defaults(
    base, manifest, native_controls, monkeypatch, index, default
):
    rows = camera_controls.native_controls("/dev/video0")
    rows[index]["default"] = default
    monkeypatch.setattr(camera_controls, "native_controls", lambda path: rows)
    code, _ = call(base, "POST", {"camera": "cam0", "reset": True}, path="/devices/camera-controls")
    assert code == 400 and manifest.read_text() == MANIFEST


def test_live_control_reset_restores_advertised_defaults_without_previous_overrides(base, manifest, native_controls):
    assert cameras.video_settings("cam0").get("controls") is None
    code, reply = call(base, "POST", {"camera": "cam0", "reset": True}, path="/devices/camera-controls")
    assert code == 200 and reply["pending"]
    assert cameras.video_settings("cam0")["controls"] == {"9963777": 2, "9963800": 2, "9963788": 1}
    assert "9963776" not in cameras.video_settings("cam0")["controls"]  # custom steering retains ownership


def test_live_controls_write_failure_is_reported_without_restart(
    base, manifest, native_controls, monkeypatch, camera_restarts
):
    def fail(stream, values, **kwargs):
        raise OSError("read-only manifest")

    monkeypatch.setattr(cameras, "update_controls", fail)
    code, reply = call(base, "POST", {"camera": "cam0", "values": {"9963777": 6}}, path="/devices/camera-controls")
    assert code == 500 and "read-only manifest" in reply["error"]
    assert manifest.read_text() == MANIFEST and camera_restarts == []


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "http://evil.example"},
        {"Host": "evil.example"},
        {"X-Forwarded-For": "100.64.0.9"},
        {"Content-Type": "text/plain"},
    ],
)
def test_live_control_writes_require_local_same_origin_json(base, manifest, native_controls, headers):
    assert call(base, "POST", {"camera": "cam0", "reset": True}, path="/devices/camera-controls", **headers)[0] == 403
    assert manifest.read_text() == MANIFEST


def test_live_controls_network_and_unavailable_camera_explain_without_discovery(base, manifest, monkeypatch):
    manifest.write_text(
        MANIFEST + '\n[[device]]\nname="entry"\nkind="rtsp"\nrtsp={url="rtsp://example.invalid/main"}\n'
    )

    def discover(path):
        pytest.fail("unsupported camera must not query native controls")

    monkeypatch.setattr(camera_controls, "native_controls", discover)
    assert (
        call(base, "GET", path="/devices/camera-controls?camera=entry")[1]["note"] == "Camera settings live on the NVR."
    )
    assert call(base, "GET", path="/devices/camera-controls?camera=unknown")[0] == 404
    monkeypatch.setattr(camera_controls, "sys", types.SimpleNamespace(platform="darwin"))
    report = call(base, "GET", path="/devices/camera-controls?camera=cam0")[1]
    assert not report["available"] and "not adjustable" in report["note"]


# Pausing devices for privacy from the header

PRIVACY_MANIFEST = """# capture's devices
[[device]]
name = "cam0"
kind = "camera"
match = { usb = "1234:0001" }  # Acme Cam

[[device]]
name = "acme"
kind = "microphone"
match = { name = "Acme Mic" }

[[device]]
name = "patio"
kind = "rtsp"
rtsp = { url = "rtsp://fixture.invalid/patio" }

[[device]]
name = "attic"
kind = "rtsp"
enabled = false
rtsp = { url = "rtsp://fixture.invalid/attic" }
"""


@pytest.fixture
def private(base, manifest, monkeypatch):
    manifest.write_text(PRIVACY_MANIFEST)
    monkeypatch.setattr(privacy, "ready", lambda: True)
    return base


def pause(base, body, **headers):
    return call(base, "POST", body, path="/devices/privacy", **headers)


def said(paused, stopping=()):
    path = privacy.status_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"paused": list(paused), "stopping": list(stopping)}))


def states(base):
    return {row["name"]: row["state"] for row in call(base, "GET", path="/devices/privacy")[1]["devices"]}


def test_pausing_all_cameras_marks_each_enabled_camera_and_nothing_else(private, manifest):
    code, reply = pause(private, {"kind": "camera", "paused": True})
    assert code == 200
    devices_ = {d["name"]: d for d in tomllib.loads(manifest.read_text())["device"]}
    assert {name for name, d in devices_.items() if d.get("paused")} == {"cam0", "patio"}
    assert "# Acme Cam" in manifest.read_text(), "the owner's comments stay"
    assert [(r["name"], r["kind"], r["paused"]) for r in reply["devices"]] == [
        ("cam0", "camera", True),
        ("acme", "microphone", False),
        ("patio", "camera", True),
    ]


def test_resuming_leaves_the_manifest_as_it_was(private, manifest):
    pause(private, {"kind": "microphone", "paused": True})
    assert tomllib.loads(manifest.read_text())["device"][1]["paused"] is True
    assert pause(private, {"kind": "microphone", "paused": False})[0] == 200
    assert manifest.read_text() == PRIVACY_MANIFEST


def test_one_named_device_is_paused_alone(private, manifest):
    assert pause(private, {"kind": "camera", "paused": True, "device": "patio"})[0] == 200
    paused = [d["name"] for d in tomllib.loads(manifest.read_text())["device"] if d.get("paused")]
    assert paused == ["patio"]


def test_the_state_follows_what_capture_has_applied(private):
    pause(private, {"kind": "camera", "paused": True, "device": "cam0"})
    assert states(private)["cam0"] == "pausing", "saved, but capture hasn't said"
    said(["cam0"], stopping=["cam0"])
    assert states(private)["cam0"] == "pausing", "its worker is still stopping"
    said(["cam0"])
    assert states(private) == {"cam0": "paused", "acme": "on", "patio": "on"}
    pause(private, {"kind": "camera", "paused": False, "device": "cam0"})
    assert states(private)["cam0"] == "resuming"
    said([])
    assert states(private)["cam0"] == "on"


def test_a_capture_that_cannot_follow_pauses_is_never_sent_one(private, manifest, monkeypatch):
    monkeypatch.setattr(privacy, "ready", lambda: False)
    code, reply = pause(private, {"kind": "camera", "paused": True})
    assert code == 409 and "Rebuild" in reply["error"]
    assert manifest.read_text() == PRIVACY_MANIFEST


def test_a_pause_from_elsewhere_is_refused(private, manifest):
    assert pause(private, {"kind": "camera", "paused": True}, **{"X-Forwarded-For": "100.64.0.9"})[0] == 403
    assert pause(private, {"kind": "camera", "paused": True}, Origin="http://evil.test")[0] == 403
    assert manifest.read_text() == PRIVACY_MANIFEST


@pytest.mark.parametrize(
    ("body", "code"),
    [
        pytest.param({"kind": "speaker", "paused": True}, 400, id="unknown-kind"),
        pytest.param({"kind": "camera", "paused": "yes"}, 400, id="paused-not-bool"),
        pytest.param({"kind": "camera", "paused": True, "extra": 1}, 400, id="extra-field"),
        pytest.param({"kind": "camera", "paused": True, "device": 3}, 400, id="device-not-name"),
        pytest.param({"kind": "camera", "paused": True, "device": "acme"}, 404, id="mic-named-as-camera"),
        pytest.param({"kind": "camera", "paused": True, "device": "attic"}, 404, id="disabled-camera"),
    ],
)
def test_a_bad_pause_is_refused_and_changes_nothing(private, manifest, body, code):
    assert pause(private, body)[0] == code
    assert manifest.read_text() == PRIVACY_MANIFEST


@pytest.mark.parametrize(
    ("stdout", "returncode", "ready"),
    [
        pytest.param(b'{"version": 1, "device_enabled": true, "device_paused": true}', 0, True, id="supports-pauses"),
        pytest.param(b'{"version": 1, "device_enabled": true}', 0, False, id="older-capture"),
        pytest.param(b'{"device_paused": "yes"}', 0, False, id="not-a-bool"),
        pytest.param(b"not json", 0, False, id="garbled"),
        pytest.param(b'{"device_paused": true}', 2, False, id="failed"),
    ],
)
def test_pauses_are_offered_only_by_a_capture_that_says_it_follows_them(
    monkeypatch, capture_binary, stdout, returncode, ready
):
    monkeypatch.setattr(
        privacy.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], returncode, stdout=stdout, stderr=b""),
    )
    assert privacy.ready() is ready


@pytest.fixture
def capture_binary(tmp_path, monkeypatch):
    """A built capture binary under a temporary root, which privacy.ready() asks for its capabilities."""
    binary = tmp_path / "continuum" / "target" / "release" / "continuum-capture"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"build one")
    monkeypatch.setattr(privacy.cfg, "ROOT", tmp_path)
    monkeypatch.setattr(privacy, "_ready", {})
    return binary


def test_capture_is_asked_once_per_build_not_once_per_read(monkeypatch, capture_binary):
    asked = []

    def run(command, **kwargs):
        asked.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=b'{"device_paused": true}', stderr=b"")

    monkeypatch.setattr(privacy.subprocess, "run", run)
    assert [privacy.ready(), privacy.ready(), privacy.ready()] == [True, True, True]
    assert len(asked) == 1
    capture_binary.write_bytes(b"build two, rebuilt")
    assert privacy.ready() is True
    assert len(asked) == 2


def test_no_built_capture_offers_no_pauses_and_runs_nothing(monkeypatch, capture_binary):
    capture_binary.unlink()
    monkeypatch.setattr(privacy.subprocess, "run", lambda *a, **k: pytest.fail("ran a binary that isn't there"))
    assert privacy.ready() is False
