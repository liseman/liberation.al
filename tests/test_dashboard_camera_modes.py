"""Camera choices use mocked capture discovery, never hardware or running services."""

import copy
import json
import subprocess
from unittest.mock import patch

import pytest

from liberational import services
from liberational.dashboard import camera_modes, capture_control


def rate(n=30, d=1):
    return {
        "kind": "discrete",
        "fps": {"numerator": n, "denominator": d},
        "interval": {"numerator": d, "denominator": n},
    }


@pytest.fixture
def inventory():
    return {
        "schema_version": 1,
        "format": "MJPEG",
        "fourcc": "MJPG",
        "diagnostics": [],
        "devices": [
            {"kind": "microphone", "key": "mic", "node": "mic", "name": "Acme Mic", "usb": None, "paths": []},
            {
                "kind": "camera",
                "key": "cam",
                "node": "/dev/video2",
                "name": "Acme Camera",
                "usb": None,
                "paths": ["/dev/v4l/by-id/acme"],
                "camera_modes": {
                    "status": "ok",
                    "fourcc": "MJPG",
                    "diagnostics": [],
                    "sizes": [
                        {
                            "kind": "discrete",
                            "width": 640,
                            "height": 480,
                            "diagnostics": [],
                            "intervals": [rate(), rate(60), rate(30000, 1001), rate(60, 2)],
                        }
                    ],
                },
            },
        ],
    }


@pytest.fixture
def discovery(monkeypatch, inventory, tmp_path):
    monkeypatch.setattr(camera_modes, "_cache", {})
    monkeypatch.setattr(camera_modes, "_reading", {})
    monkeypatch.setattr(services.cfg, "OWNER_ROOT", tmp_path)
    spec = tmp_path / "services.toml"
    spec.write_text('[continuum-capture]\ncommand="continuum/target/release/continuum-capture"\n')
    monkeypatch.setattr(services, "SPEC", spec)
    with (
        patch.object(capture_control.shutil, "which", autospec=True, return_value="/bin/continuum-capture"),
        patch.object(
            camera_modes.subprocess,
            "run",
            autospec=True,
            return_value=subprocess.CompletedProcess([], 0, json.dumps(inventory), ""),
        ) as run,
    ):
        yield run


def test_repository_service_binary_is_used_without_a_path_install(discovery, tmp_path, monkeypatch):
    binary = tmp_path / "continuum/target/release/continuum-capture"
    binary.parent.mkdir(parents=True)
    binary.touch(mode=0o755)
    with monkeypatch.context() as local:
        local.setattr(capture_control.shutil, "which", lambda _: None)
        report = camera_modes.modes("/dev/video2")
    assert report["choices"] == [{"width": 640, "height": 480, "fps": [30, 60]}]
    assert discovery.call_args.args[0] == [str(binary), "--list", "--json", "--format", "MJPEG"]


def test_declared_service_path_is_resolved_instead_of_hardcoding_release(discovery, tmp_path):
    services.SPEC.write_text('[continuum-capture]\ncommand="bin/capture-custom"\nargs=["--devices", "unused.toml"]\n')
    binary = tmp_path / "bin/capture-custom"
    binary.parent.mkdir()
    binary.touch(mode=0o755)
    assert camera_modes.modes("/dev/video2")["choices"]
    assert discovery.call_args.args[0] == [str(binary), "--list", "--json", "--format", "MJPEG"]


@pytest.mark.parametrize("kind", ["nonexecutable", "directory"])
def test_unusable_service_binary_falls_back_to_path(discovery, tmp_path, kind):
    binary = tmp_path / "continuum/target/release/continuum-capture"
    binary.parent.mkdir(parents=True)
    if kind == "directory":
        binary.mkdir()
    else:
        binary.touch(mode=0o644)
    assert camera_modes.modes("/dev/video2")["choices"]
    assert discovery.call_args.args[0][0] == "/bin/continuum-capture"


def test_choices_keep_native_rationals_but_only_offer_exact_integer_rates(discovery):
    report = camera_modes.modes("/dev/v4l/by-id/acme")
    assert report["choices"] == [{"width": 640, "height": 480, "fps": [30, 60]}]
    assert report["sizes"][0]["intervals"][2]["fps"] == {"numerator": 30000, "denominator": 1001}
    assert any("Non-integral" in d for d in report["diagnostics"])
    discovery.assert_called_once_with(
        ["/bin/continuum-capture", "--list", "--json", "--format", "MJPEG"],
        capture_output=True,
        text=True,
        timeout=5.0,
        check=False,
    )


def test_cache_is_short_lived_detached_and_save_refreshes(discovery, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(camera_modes.time, "monotonic", lambda: clock[0])
    camera_modes.modes("/dev/video2")["sizes"].clear()
    assert camera_modes.modes("/dev/video2")["choices"]
    assert discovery.call_count == 1
    assert camera_modes.validate_choice("/dev/video2", 640, 480, 30) == {"width": 640, "height": 480, "fps": 30}
    assert discovery.call_count == 2
    clock[0] = 16.0
    assert camera_modes.modes("/dev/video2")["choices"], "stale: the last list, at once"
    for reading in list(camera_modes._reading.values()):
        reading.join(5)
    assert discovery.call_count == 3, "and read again behind it"
    assert camera_modes._reading == {}


@pytest.mark.parametrize(
    "width,height,fps", [(320, 240, 30), (640, 480, 29), (True, 480, 30), (640, 480, 30.0), (640, 0, 30)]
)
def test_save_rejects_unadvertised_or_noninteger_choices(discovery, width, height, fps):
    with pytest.raises(ValueError):
        camera_modes.validate_choice("/dev/video2", width, height, fps)


def test_partial_results_keep_diagnostics_and_ranges_are_not_expanded(discovery, inventory):
    native = inventory["devices"][1]["camera_modes"]
    native["status"] = "partial"
    native["diagnostics"] = ["driver failed at size index 2"]
    native["sizes"].append(
        {
            "kind": "stepwise",
            "min_width": 320,
            "max_width": 1920,
            "step_width": 16,
            "min_height": 240,
            "max_height": 1080,
            "step_height": 16,
        }
    )
    native["sizes"][0]["intervals"].append(
        {
            "kind": "continuous",
            "min": {"numerator": 1, "denominator": 60},
            "max": {"numerator": 1, "denominator": 15},
            "step": {"numerator": 1, "denominator": 1},
        }
    )
    discovery.return_value.stdout = json.dumps(inventory)
    report = camera_modes.modes("/dev/video2")
    assert report["status"] == "partial"
    assert report["choices"] == [{"width": 640, "height": 480, "fps": [30, 60]}]
    assert "driver failed at size index 2" in report["diagnostics"]
    assert report["sizes"][1]["kind"] == "stepwise"


@pytest.mark.parametrize(
    "stdout",
    [
        "not json",
        "[]",
        '{"schema_version": 2}',
        '{"schema_version": true}',
        '{"schema_version":1,"format":"MJPEG","devices":null}',
    ],
)
def test_malformed_inventory_is_unavailable(discovery, stdout):
    discovery.return_value.stdout = stdout
    report = camera_modes.modes("/dev/video2")
    assert report["status"] == "unavailable"
    assert report["diagnostics"]
    assert not report["choices"]


@pytest.mark.parametrize("error", [OSError("cannot execute"), subprocess.TimeoutExpired("capture", 5)])
def test_discovery_failure_cannot_authorize_a_save(discovery, error):
    discovery.side_effect = error
    with pytest.raises(ValueError, match="not explicitly offered"):
        camera_modes.validate_choice("/dev/video2", 640, 480, 30)


def test_missing_binary_is_unavailable_without_running_anything(discovery, monkeypatch):
    with monkeypatch.context() as local:
        local.setattr(capture_control.shutil, "which", lambda _: None)
        assert camera_modes.modes("/dev/video2")["status"] == "unavailable"
    discovery.assert_not_called()


@pytest.mark.parametrize("status", ["unavailable", "unsupported"])
def test_unavailable_and_unsupported_keep_their_status_and_diagnostics(discovery, inventory, status):
    inventory["devices"][1]["camera_modes"]["status"] = status
    inventory["devices"][1]["camera_modes"]["diagnostics"] = ["driver does not support this format"]
    discovery.return_value.stdout = json.dumps(inventory)
    report = camera_modes.modes("/dev/video2")
    assert report["status"] == status
    assert report["diagnostics"] == ["driver does not support this format"]
    assert not report["choices"]


def test_missing_device_does_not_borrow_another_cameras_modes(discovery):
    report = camera_modes.modes("/dev/video99")
    assert report["status"] == "unavailable"
    assert not report["choices"]


def test_fourcc_case_is_preserved_and_cache_is_separate(discovery, inventory):
    report = copy.deepcopy(inventory)
    report["format"] = report["fourcc"] = "2vuy"
    report["devices"][1]["camera_modes"]["fourcc"] = "2vuy"
    discovery.return_value.stdout = json.dumps(report)
    assert camera_modes.modes("/dev/video2", "2vuy")["fourcc"] == "2vuy"
    assert discovery.call_args.args[0][-1] == "2vuy"


@pytest.mark.parametrize("format", ["mjpeg", "mjpg", "MJPG"])
def test_mjpeg_aliases_use_the_canonical_capture_format(discovery, format):
    assert camera_modes.modes("/dev/video2", format)["choices"]
    assert discovery.call_args.args[0][-1] == "MJPEG"


def test_inconsistent_rational_cannot_authorize_a_save(discovery, inventory):
    inventory["devices"][1]["camera_modes"]["sizes"][0]["intervals"] = [
        {
            "kind": "discrete",
            "fps": {"numerator": 30, "denominator": 1},
            "interval": {"numerator": 1, "denominator": 60},
        }
    ]
    discovery.return_value.stdout = json.dumps(inventory)
    with pytest.raises(ValueError, match="not explicitly offered"):
        camera_modes.validate_choice("/dev/video2", 640, 480, 30)


def test_failed_process_does_not_trust_even_valid_stdout(discovery):
    discovery.return_value.returncode = 1
    assert camera_modes.modes("/dev/video2")["status"] == "unavailable"


@pytest.mark.parametrize(
    "descriptor",
    [
        None,
        {"kind": "discrete", "width": "640", "height": 480},
        {"kind": "discrete", "width": 640, "height": 480, "intervals": [rate(30, 0)]},
    ],
)
def test_malformed_modes_are_excluded_with_diagnostics(discovery, inventory, descriptor):
    inventory["devices"][1]["camera_modes"]["sizes"] = [descriptor]
    discovery.return_value.stdout = json.dumps(inventory)
    report = camera_modes.modes("/dev/video2")
    assert not report["choices"]
    assert report["diagnostics"]
