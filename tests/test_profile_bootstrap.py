import json
import os
import subprocess
import sys
from unittest.mock import create_autospec

import pytest

from liberational import config
from liberational import profile_bootstrap as bootstrap


@pytest.mark.parametrize("switch", ["", "0", "false", "no"])
def test_disabled_package_import_loads_no_profiling_or_configuration(switch):
    environment = {**os.environ, "PACK_PROFILE": switch}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys,liberational; assert not any(k in sys.modules for k in ('liberational.service_profile','liberational.profile_bootstrap','liberational.config'))",
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux procfs CPU accounting")
def test_console_package_hook_creates_finite_profile_without_a_command_wrapper(tmp_path):
    settings = tmp_path / "pack.toml"
    settings.write_text(f"[paths]\nlogs = {json.dumps(str(tmp_path / 'logs'))}\n")
    environment = {
        **os.environ,
        "PACK_CONFIG": str(settings),
        "PACK_PROFILE": "1",
        "PACK_PROFILE_DELAY_SECONDS": "0",
        "PACK_PROFILE_SECONDS": "1",
    }
    source = """
import os, sys
sys.argv = ['pack-probe']
pid = os.getpid()
import liberational
from liberational.profile_bootstrap import _profile
assert _profile is not None and _profile.wait(5)
assert _profile.error is None
assert os.getpid() == pid
"""
    result = subprocess.run(
        [sys.executable, "-c", source],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    (artifact,) = (tmp_path / "logs" / "profiles").glob("*.json")
    profile = json.loads(artifact.read_text())
    assert profile["identity"]["service"] == "probe"
    assert profile["final"] is True


@pytest.mark.parametrize(
    "switch,command,explicit,expected",
    [
        ("", "pack-listen", "", None),
        ("0", "pack-listen", "", None),
        ("1", "pack-listen", "", "listen"),
        ("see-motion,listen", "pack-listen", "", "listen"),
        ("see-motion", "pack-listen", "", None),
        ("1", "pack-services", "listen", None),
        ("1", "python", "", None),
        ("1", "-m", "see-actions", "see-actions"),
        ("1", "pack-listen", "../bad", None),
    ],
)
def test_profile_selector_limits_startup_to_requested_service(monkeypatch, switch, command, explicit, expected):
    monkeypatch.setenv("PACK_PROFILE", switch)
    monkeypatch.setenv("PACK_PROFILE_SERVICE", explicit)
    monkeypatch.setattr(sys, "argv", [command])
    assert bootstrap._service() == expected


def test_selected_profile_starts_once_with_configured_output_and_finite_capture(monkeypatch, tmp_path):
    from liberational.service_profile import Profiler

    factory = create_autospec(Profiler)
    profiler = factory.return_value
    profiler.start.return_value = profiler
    monkeypatch.setattr("liberational.service_profile.Profiler", factory)
    register = create_autospec(bootstrap.atexit.register)
    monkeypatch.setattr(bootstrap.atexit, "register", register)
    monkeypatch.setattr(bootstrap, "_profile", None)
    monkeypatch.setattr(sys, "argv", ["pack-listen"])
    monkeypatch.setenv("PACK_PROFILE", "1")
    monkeypatch.setenv("PACK_PROFILE_SECONDS", "60")
    monkeypatch.setenv("PACK_PROFILE_DELAY_SECONDS", "5")
    with config.override({"paths": {"logs": str(tmp_path)}}):
        bootstrap.maybe_start()
        bootstrap.maybe_start()
    factory.assert_called_once_with(
        "listen",
        tmp_path / "profiles",
        interval_s=0.05,
        flush_s=30.0,
        duration_s=60.0,
        delay_s=5.0,
    )
    profiler.start.assert_called_once_with()
    register.assert_called_once_with(profiler.stop)


@pytest.mark.parametrize("value", ["nan", "inf", "-1", "0", "3601", "bad"])
def test_invalid_profile_duration_is_actionable_without_preventing_service_start(monkeypatch, capsys, value):
    monkeypatch.setattr(bootstrap, "_profile", None)
    monkeypatch.setattr(sys, "argv", ["pack-listen"])
    monkeypatch.setenv("PACK_PROFILE", "1")
    monkeypatch.setenv("PACK_PROFILE_SECONDS", value)
    bootstrap.maybe_start()
    assert bootstrap._profile is None
    assert "CPU profile unavailable for listen" in capsys.readouterr().err


def test_unavailable_cpu_counters_warn_without_changing_service_start(monkeypatch, capsys):
    from liberational.service_profile import Profiler

    factory = create_autospec(Profiler)
    factory.return_value.start.side_effect = RuntimeError("Linux procfs is unavailable")
    monkeypatch.setattr("liberational.service_profile.Profiler", factory)
    monkeypatch.setattr(bootstrap, "_profile", None)
    monkeypatch.setattr(sys, "argv", ["pack-listen"])
    monkeypatch.setenv("PACK_PROFILE", "1")
    bootstrap.maybe_start()
    assert bootstrap._profile is None
    assert "Linux procfs is unavailable" in capsys.readouterr().err
