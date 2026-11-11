"""pack-setup: what it installs and says on a machine with the package installed, with every command faked."""

import sys
from typing import Any

import pytest

from liberational import config as cfg
from liberational import fetch_models, pack_setup, system_packages
from liberational.system_packages import Package


@pytest.fixture
def installed(tmp_path, monkeypatch):
    """An installed package: no checkout, an environment in tmp_path/bin, and no release lock."""
    (tmp_path / "bin").mkdir()
    monkeypatch.setattr(cfg, "CHECKOUT", None)
    monkeypatch.setattr(cfg, "BIN", tmp_path / "bin")
    monkeypatch.setattr(pack_setup, "LOCK", tmp_path / "constraints.txt")
    monkeypatch.setattr(pack_setup.importlib.metadata, "version", {"liberational-pack": "1.2.3"}.__getitem__)
    monkeypatch.setattr(pack_setup.shutil, "which", {"uv": "/usr/bin/uv"}.get)
    monkeypatch.setattr(cfg, "platform_key", lambda: "darwin-arm64")  # every extra this platform gets, GPU or not
    return tmp_path


def recorder(code=0):
    ran = []

    def run(command):
        ran.append(command)
        return code

    return ran, run


@pytest.mark.parametrize(
    "platform, nvidia, want",
    [
        pytest.param("linux-x86_64", True, ["faces", "segmentation", "deep-asr", "voice-clone"], id="linux-nvidia"),
        pytest.param("linux-x86_64", False, ["faces", "segmentation", "deep-asr"], id="linux-no-gpu"),
        pytest.param("linux-aarch64", True, ["faces", "segmentation", "deep-asr"], id="linux-arm-nvidia"),
        pytest.param("darwin-arm64", False, ["faces", "segmentation", "deep-asr"], id="apple-silicon"),
    ],
)
def test_voice_cloning_is_only_for_linux_x86_with_an_nvidia_gpu(platform, nvidia, want):
    assert pack_setup.extras(platform, nvidia) == want


def test_the_extras_are_installed_as_the_installed_packages_own_at_its_version(installed):
    want = ["/usr/bin/uv", "pip", "install", "--python", sys.executable, "liberational-pack[faces,deep-asr]==1.2.3"]
    assert pack_setup.extras_command("/usr/bin/uv", ["faces", "deep-asr"]) == want
    (installed / "constraints.txt").write_text("numpy==2.0.0\n")
    assert pack_setup.extras_command("/usr/bin/uv", ["faces", "deep-asr"]) == [
        *want,
        "--constraint",
        str(installed / "constraints.txt"),
    ]


def test_a_bundled_uv_beside_the_interpreter_is_used_before_one_on_path(installed):
    assert pack_setup.uv() == "/usr/bin/uv"
    (installed / "bin" / "uv").touch()
    assert pack_setup.uv() == str(installed / "bin" / "uv")


def test_the_extras_step_runs_uv_once_and_passes_when_it_does(installed):
    ran, run = recorder()
    assert pack_setup.extras_step(check=False, run=run) is True
    assert [command[:3] for command in ran] == [["/usr/bin/uv", "pip", "install"]]


@pytest.fixture
def absent(monkeypatch):
    """This machine's environment as a bare one: no extra's module is importable."""
    monkeypatch.setattr(pack_setup.importlib.util, "find_spec", lambda name: None)


@pytest.fixture
def present(monkeypatch):
    monkeypatch.setattr(pack_setup.importlib.util, "find_spec", lambda name: object())


def test_check_names_the_missing_extras_and_fails_without_running_anything(installed, absent, capsys):
    ran, run = recorder()
    assert pack_setup.extras_step(check=True, run=run) is False
    assert ran == []
    said = capsys.readouterr().out
    assert "missing: faces (insightface), segmentation (timm, torchvision)" in said
    assert "install them with: /usr/bin/uv pip install" in said


def test_check_passes_when_the_extras_are_here(installed, present, capsys):
    ran, run = recorder()
    assert pack_setup.extras_step(check=True, run=run) is True
    assert ran == []
    assert "the extras are here" in capsys.readouterr().out


def test_a_failed_install_is_left_to_do(installed, capsys):
    _, run = recorder(code=2)
    assert pack_setup.extras_step(check=False, run=run) is False
    assert "exit 2" in capsys.readouterr().out


def test_without_uv_the_command_is_said_and_nothing_runs(installed, monkeypatch, capsys):
    monkeypatch.setattr(pack_setup.shutil, "which", lambda name: None)
    ran, run = recorder()
    assert pack_setup.extras_step(check=False, run=run) is False
    assert ran == []
    assert "uv pip install --python" in capsys.readouterr().out


def test_a_checkout_leaves_the_extras_to_mise(monkeypatch, capsys):
    monkeypatch.setattr(cfg, "CHECKOUT", cfg.PACKAGE.parents[1])
    ran, run = recorder()
    assert pack_setup.extras_step(check=False, run=run) is True
    assert ran == []
    assert "mise run setup" in capsys.readouterr().out


def test_continuum_is_here_only_with_every_daemon_and_the_sdk(installed, monkeypatch, capsys):
    monkeypatch.setattr(pack_setup.importlib.util, "find_spec", lambda name: object())
    for name in pack_setup.DAEMONS[:-1]:
        (installed / "bin" / name).touch()
    assert pack_setup.continuum_step() is False
    assert pack_setup.DAEMONS[-1] in capsys.readouterr().out
    (installed / "bin" / pack_setup.DAEMONS[-1]).touch()
    assert pack_setup.continuum_step() is True
    monkeypatch.setattr(pack_setup.importlib.util, "find_spec", lambda name: None)
    assert pack_setup.continuum_step() is False
    assert "continuum_py" in capsys.readouterr().out


def test_ollama_that_isnt_installed_says_how_to_get_it_for_this_platform(installed, monkeypatch, capsys):
    monkeypatch.setattr(fetch_models, "CHECKOUT", None)
    monkeypatch.setattr(fetch_models.sys, "platform", "linux")
    assert pack_setup.ollama_step() is False
    assert "https://ollama.com/download" in capsys.readouterr().out
    monkeypatch.setattr(fetch_models.sys, "platform", "darwin")
    pack_setup.ollama_step()
    assert "brew install ollama" in capsys.readouterr().out


def test_ollama_installed_but_not_running_is_not_a_failure(installed, monkeypatch, capsys):
    monkeypatch.setattr(pack_setup.shutil, "which", {"ollama": "/usr/bin/ollama"}.get)
    monkeypatch.setattr(pack_setup, "ollama_answers", lambda: False)
    assert pack_setup.ollama_step() is True
    assert "the application starts it" in capsys.readouterr().out


def test_check_names_the_missing_models_and_fails(monkeypatch, capsys):
    monkeypatch.setattr(fetch_models, "missing", lambda: ["face model: models/faces/2d106det.onnx", "pose: models/pose.onnx"])
    assert pack_setup.models_step(check=True, ollama=False) is False
    said = capsys.readouterr().out
    assert "missing: face model: models/faces/2d106det.onnx" in said
    assert "fetch them with pack-setup, or with pack-models" in said


def test_check_passes_when_every_model_is_here(monkeypatch, capsys):
    monkeypatch.setattr(fetch_models, "missing", list)
    assert pack_setup.models_step(check=True, ollama=True) is True
    assert "every model this machine wants is here" in capsys.readouterr().out


def test_a_download_that_stops_is_left_to_do_and_says_to_run_again(monkeypatch, capsys):
    def stopped(ollama):
        raise OSError("the download stopped at 5 of 9 bytes")

    monkeypatch.setattr(fetch_models, "fetch", stopped)
    assert pack_setup.models_step(check=False, ollama=True) is False
    assert "run pack-setup again to carry on" in capsys.readouterr().out


@pytest.fixture
def steps(installed, monkeypatch):
    """main() with every step faked: `steps` maps a step to whether it passes; "fetched" in it lists the models
    step's calls."""
    fetched: list[tuple[bool, bool]] = []
    result: dict[str, Any] = {"system": True, "extras": True, "continuum": True, "ollama": True, "models": True}
    result["fetched"] = fetched
    monkeypatch.setattr(pack_setup, "system_step", lambda: result["system"])
    monkeypatch.setattr(pack_setup, "extras_step", lambda check: result["extras"])
    monkeypatch.setattr(pack_setup, "continuum_step", lambda: result["continuum"])
    monkeypatch.setattr(pack_setup, "ollama_step", lambda: result["ollama"])

    def models(check, ollama):
        fetched.append((check, ollama))
        return result["models"]

    monkeypatch.setattr(pack_setup, "models_step", models)
    return result


def run_main(*argv):
    with pytest.raises(SystemExit) as stopped:
        pack_setup.main(list(argv))
    return stopped.value.code


def test_a_whole_setup_exits_0_and_ends_with_how_to_open_the_application(steps, capsys):
    assert run_main() == 0
    assert capsys.readouterr().out.splitlines()[-2:] == ["Set up.", "Open the application with: pack-app"]
    assert steps["fetched"] == [(False, True)]


@pytest.mark.parametrize("step", ["system", "extras", "continuum", "models"])
def test_anything_the_application_needs_still_missing_exits_1(steps, step, capsys):
    steps[step] = False
    assert run_main() == 1
    assert "Still to do: " in capsys.readouterr().out


def test_ollama_missing_alone_does_not_fail_setup(steps):
    steps["ollama"] = False
    assert run_main() == 0


def test_the_flags_reach_the_models_step(steps):
    assert run_main("--no-models") == 0
    assert run_main("--no-ollama") == 0
    assert run_main("--check") == 0
    assert steps["fetched"] == [(False, False), (True, True)]


def test_missing_needed_system_packages_are_listed_with_their_command_and_fail_the_step(monkeypatch, capsys):
    sox = Package("sox", "voice cloning shells out to it", command="sox", apt="sox")
    usb = Package("libusb", "a mic array's settings", library="usb-1.0", apt="libusb-1.0-0", optional=True)
    monkeypatch.setattr(system_packages, "load", lambda: [sox, usb])
    monkeypatch.setattr(system_packages, "describe", lambda gone: "DESCRIBED " + ",".join(p.name for p in gone))
    monkeypatch.setattr(system_packages, "missing", lambda packages: [sox, usb])
    assert pack_setup.system_step() is False
    assert "DESCRIBED sox,libusb" in capsys.readouterr().out
    monkeypatch.setattr(system_packages, "missing", lambda packages: [usb])
    assert pack_setup.system_step() is True  # an optional one alone is said, and doesn't fail it
