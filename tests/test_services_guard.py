"""Nothing installs or restarts real services by accident: the suite's guard, and what pack-services does with no
application open (nothing, and it says so)."""

import os
import pathlib
import socket
import sqlite3
import subprocess
import tempfile
import tomllib

import pytest

import conftest
from liberational import config as cfg
from liberational import services


@pytest.mark.parametrize(
    "cmd",
    [["systemctl", "--user", "restart", "pack-ambient"], ["python", "-m", "liberational.services", "install"]],
    ids=["systemctl", "pack-services"],
)
def test_service_commands_are_refused(cmd):
    with pytest.raises(RuntimeError, match="refused"):
        subprocess.run(cmd)
    assert cmd[0] in conftest.hits.pop()[0]


def test_exec_is_refused():
    with pytest.raises(RuntimeError, match="refused"):
        os.execv("/bin/true", ["true"])
    assert conftest.hits.pop()[0].startswith("execv")


def test_pack_toml_is_a_temp_file():
    assert cfg.local_config().is_relative_to(conftest._TMP)


@pytest.mark.parametrize("key", tomllib.loads(cfg.DEFAULTS.read_text())["paths"])
def test_every_data_path_is_a_temp_dir(key):
    assert cfg.path(f"paths.{key}").is_relative_to(conftest._TMP)


# Each opens a missing file read-only, so a broken guard changes nothing on disk.
@pytest.mark.parametrize(
    "opener",
    [open, lambda path: sqlite3.connect(f"file:{path}?mode=ro", uri=True)],
    ids=["open", "sqlite-uri"],
)
def test_opening_a_real_data_path_is_refused(opener):
    real = cfg.workdir() / "data" / "logs" / "no-such-file"
    with pytest.raises(RuntimeError, match="real data directory"):
        opener(real)
    assert "no-such-file" in conftest.hits.pop()[0]


def test_moving_a_file_into_a_real_data_path_is_refused(tmp_path):
    missing = tmp_path / "missing"  # a source that doesn't exist, so a broken guard still writes nothing
    with pytest.raises(RuntimeError, match="real data directory"):
        os.replace(missing, cfg.workdir() / "data" / "state" / "moved-in")
    assert "moved-in" in conftest.hits.pop()[0]


@pytest.mark.parametrize(
    "action",
    [services.install, services.uninstall, services.stop, services.start, services.restart],
    ids=["install", "uninstall", "stop", "start", "restart"],
)
def test_a_verb_with_no_application_open_touches_nothing(monkeypatch, action):
    """No verb renders, installs, or uninstalls an operating-system unit any more, from any directory: the suite's own
    guard (conftest refuses systemctl, launchctl, and pack-services) would fail this test if one shelled out."""
    monkeypatch.setattr(services, "_supervised", lambda: None)
    with pytest.raises(SystemExit, match="pack-app"):
        action(["listen"])


def test_temp_files_go_under_the_session_root():
    assert pathlib.Path(tempfile.gettempdir()).is_relative_to(conftest._TMP)
    assert os.environ["TMPDIR"] == tempfile.gettempdir()  # subprocesses too


def test_a_broker_socket_under_a_long_named_tests_state_dir_binds_with_room_to_spare(tmp_path):
    # A test that points paths.state at tmp_path puts the broker's socket at <tmp_path>/state/continuum/
    # broker.sock. Under the old temp root that came to about 118 bytes, past the 108 a unix socket path may use, and
    # a broker client's connect failed in a background thread. This test's name is long on purpose (pytest keeps 30
    # characters of it), and the 20 extra bytes leave room for a longer user name or a deeper state path.
    assert tmp_path.is_relative_to(conftest._TMP)
    path = tmp_path / "state" / "continuum" / ("x" * 20 + "broker.sock")
    path.parent.mkdir(parents=True)
    with socket.socket(socket.AF_UNIX) as server:
        server.bind(str(path))


@pytest.mark.parametrize(
    "fallback,warns",
    [
        pytest.param("/sandbox/tmp", False, id="short-fallback-is-quiet"),
        pytest.param("/sandbox/" + "deep/" * 12 + "tmp", True, id="long-fallback-warns"),
    ],
)
def test_a_session_root_falls_back_to_tmpdir_when_tmp_is_not_writable(monkeypatch, fallback, warns):
    # A sandbox that can only write under its own TMPDIR failed on importing conftest when the root was /tmp only.
    calls = []

    def mkdtemp(prefix, dir):
        calls.append(dir)
        if dir == "/tmp":
            raise PermissionError(13, "Permission denied", dir)
        return f"{dir}/{prefix}abcdefgh"

    monkeypatch.setattr(tempfile, "mkdtemp", mkdtemp)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: fallback)
    root, warning = conftest._session_root()
    assert (calls, root) == (["/tmp", fallback], pathlib.Path(f"{fallback}/lt-abcdefgh"))
    assert (warning is not None) is warns
    assert str(root) in (warning or str(root))


def test_a_writable_tmp_gives_a_short_root_without_a_warning():
    assert (conftest._TMP.parent, conftest._ROOT_WARNING) == (pathlib.Path("/tmp"), None)
