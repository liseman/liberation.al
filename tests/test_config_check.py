"""`pack-config --check` (src/liberational/config.py, #1907): what in an old pack.toml to delete after an update."""

import pytest

from liberational import config, readers

DEFAULTS = {
    "vision": {"actions_fps": 2.0, "shared_decode_gpu": 0, "actions_camera_fps": {"cam0": 30.0}},
    "audio": {"mic_channel": {}},  # empty in defaults: keyed by the owner's own device names
    "services": {"enable": [], "disable": []},
    "pets": {"label_days": 14},
}


@pytest.mark.parametrize(
    ("local", "gone"),
    [
        pytest.param({"vision": {"actions_fps": 4}}, [], id="a-setting-that-exists"),
        pytest.param({"aging": {"days": 3}}, ["aging"], id="a-removed-table"),
        pytest.param({"vision": {"live_pose_fps": 4, "actions_fps": 1}}, ["vision.live_pose_fps"], id="a-removed-key"),
        pytest.param({"audio": {"mic_channel": {"wave3": 1}}}, [], id="the-owners-own-keys"),
        pytest.param({"models": {"files": {"anything": "x"}}}, [], id="models-are-free-form"),
        pytest.param(
            {"vision": {"actions_camera_fps": {"porch": 15.0}}}, [], id="a-camera-keyed-table-with-a-default-key"
        ),
    ],
)
def test_settings_the_defaults_no_longer_have_are_listed(local, gone):
    assert config.unknown_settings(local, DEFAULTS) == gone


def test_services_named_in_pack_toml_that_are_no_longer_declared_are_listed():
    local = {"services": {"enable": ["see-actions"], "disable": ["see-faces", "listen"]}}
    assert config.unknown_services(local, ["see-actions", "listen", "dashboard"]) == ["services: see-faces"]
    assert config.unknown_services({}, ["listen"]) == []


def test_check_prints_what_to_delete_and_fails_until_its_gone(tmp_path, monkeypatch, capsys):
    pack = tmp_path / "pack.toml"
    pack.write_text('[aging]\ndays = 3\n\n[services]\ndisable = ["catalog-age"]\n')
    monkeypatch.setenv("PACK_CONFIG", str(pack))
    assert config.check() == 1
    out = capsys.readouterr().out
    assert "delete from pack.toml: aging" in out and "delete from pack.toml: services: catalog-age" in out
    pack.write_text("[vision]\nactions_fps = 4\n")
    assert config.check() == 0
    assert "every setting and service it names exists" in capsys.readouterr().out


def test_the_config_module_runs_without_the_services_module():
    # Every service imports config, so config importing services makes every service look like a reader of what
    # services.py reads (asr.engine), and a Settings save restarts them all (#1911 did, by importing it in check()).
    assert "liberational.services" not in readers._closure("liberational.config")
