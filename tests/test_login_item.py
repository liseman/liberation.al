"""Start at login (#157): the desktop's autostart entry follows one setting, off by default."""

import sys
from pathlib import Path

import pytest

from liberational import config as cfg
from liberational import login_item


@pytest.fixture
def desktop(tmp_path, monkeypatch):
    """A Linux desktop whose config directory is a temporary one. Returns where the entry goes."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(login_item.sys, "platform", "linux")
    monkeypatch.delenv("PACK_ROOT", raising=False)
    monkeypatch.delenv("PACK_CONFIG", raising=False)
    return tmp_path / "xdg" / "autostart" / "liberational.desktop"


def test_it_is_off_unless_the_owner_turns_it_on(desktop):
    assert cfg.get("app.start_at_login") is False
    assert login_item.sync(False) == "Start at login is off"
    assert not desktop.exists() and login_item.enabled() is False


def test_on_writes_an_entry_the_desktop_runs_and_off_removes_it(desktop):
    assert login_item.sync(True) == f"Start at login turned on: wrote {desktop}"
    lines = desktop.read_text().splitlines()
    assert lines[0] == "[Desktop Entry]" and "Type=Application" in lines and "Terminal=false" in lines
    run = next(line for line in lines if line.startswith("Exec=")).removeprefix("Exec=")
    assert run in (str(Path(sys.executable).parent / "pack-app"), f"{sys.executable} -m liberational.app_window")
    assert login_item.enabled() is True
    assert login_item.sync(False) == f"Start at login turned off: removed {desktop}"
    assert not desktop.exists() and login_item.enabled() is False


def test_syncing_again_changes_nothing(desktop):
    login_item.sync(True)
    stamp = desktop.stat().st_mtime_ns
    assert login_item.sync(True) == "Start at login is on"
    assert desktop.stat().st_mtime_ns == stamp
    login_item.sync(False)
    assert login_item.sync(False) == "Start at login is off"


def test_an_entry_from_another_install_is_rewritten_to_this_one(desktop):
    desktop.parent.mkdir(parents=True)
    desktop.write_text("[Desktop Entry]\nExec=/somewhere/else/pack-app\n")
    login_item.sync(True)
    assert "/somewhere/else" not in desktop.read_text()


def test_the_entry_opens_the_same_install_this_process_was_given(desktop, monkeypatch):
    monkeypatch.setenv("PACK_ROOT", "/srv/my pack")
    monkeypatch.setenv("PACK_CONFIG", "/srv/50%.toml")
    login_item.sync(True)
    run = next(l for l in desktop.read_text().splitlines() if l.startswith("Exec=")).removeprefix("Exec=")
    # a space is quoted and a percent doubled, as the desktop entry specification asks
    assert run.startswith('env "PACK_ROOT=/srv/my pack" PACK_CONFIG=/srv/50%%.toml ')


def test_sync_follows_the_setting_when_not_told(desktop):
    with cfg.override({"app": {"start_at_login": True}}):
        login_item.sync()
    assert desktop.exists()
    with cfg.override({"app": {"start_at_login": False}}):
        login_item.sync()
    assert not desktop.exists()


def test_on_a_mac_it_does_nothing_yet_and_says_so(desktop, monkeypatch):
    monkeypatch.setattr(login_item.sys, "platform", "darwin")
    assert "isn't available on this platform yet" in login_item.sync(True)
    assert not desktop.exists() and login_item.enabled() is False
    assert login_item.sync(False) == "Start at login is off"
