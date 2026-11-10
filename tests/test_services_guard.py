"""Nothing installs or restarts real services by accident: the suite's guard, and pack-services refusing to run
from a git worktree or a temporary directory."""

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


class Elsewhere:
    """The config module as it is, but for a working directory of `root`."""

    def __init__(self, root: pathlib.Path) -> None:
        self.workdir = lambda: root

    def __getattr__(self, name: str) -> object:
        return getattr(cfg, name)


@pytest.fixture
def temp_root(tmp_path, monkeypatch):
    """The working directory is a temp dir with nothing selected to install; returns the service-manager commands sent."""
    sent = []

    def run(*cmd, check=False):
        sent.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    # Only the checkout guard sees this root: services' view of the config has its working directory here and every
    # other setting as it is. Mutating the shared config module let lazy imports keep a temp manifest path after
    # monkeypatch restored it.
    monkeypatch.setattr(services, "cfg", Elsewhere(tmp_path))
    monkeypatch.setattr(services, "preflight", lambda names: None)
    monkeypatch.setattr(services, "DARWIN", False)
    monkeypatch.setattr(services, "_run", run)
    monkeypatch.setattr(services, "selected", dict)
    monkeypatch.setattr(services, "unit_dir", lambda: tmp_path / "units")
    monkeypatch.setattr(services, "log_path", lambda name: tmp_path / "logs" / f"{name}.log")
    return sent


@pytest.mark.parametrize("action", [services.install, services.uninstall], ids=["install", "uninstall"])
def test_a_temporary_directory_is_refused_unless_asked(temp_root, action):
    assert "temporary directory" in services.checkout_note()
    with pytest.raises(SystemExit, match="--anywhere"):
        action([])
    assert temp_root == []
    action([], anywhere=True)
    assert ("systemctl", "--user", "daemon-reload") in temp_root


def test_a_git_worktree_is_refused(temp_root, tmp_path, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(services.cfg, "workdir", lambda: pathlib.Path("/srv/liberational"))
        assert services.checkout_note() == ""  # a plain checkout passes
    (tmp_path / ".git").write_text("gitdir: /somewhere/.git/worktrees/x\n")  # checked before the temp dir
    assert "worktree" in services.checkout_note()
    with pytest.raises(SystemExit, match="worktree"):
        services.install([])


def test_checkout_guard_root_does_not_redirect_package_readers(temp_root, monkeypatch):
    from liberational import system_packages

    opened = []
    monkeypatch.setattr(pathlib.Path, "read_text", lambda path: opened.append(path) or "")
    system_packages.load()
    assert opened == [cfg.DEFAULTS.with_name("system-packages.toml")]


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
