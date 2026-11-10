"""tools/gated.sh: one Continuum gate at a time across a repository's worktrees, with flock and with the
Python fallback macOS uses. Each test runs real commands in a throwaway repository."""

import os
import pathlib
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time

import pytest

GATED = pathlib.Path(__file__).resolve().parents[1] / "tools" / "gated.sh"
WAITING = "waiting for another worktree's gate"


def git(cwd: pathlib.Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A repository with a subdirectory and a second worktree, which shares its common git directory."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    for var in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(var, "Dana")
    for var in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(var, "dana@example.com")
    main = tmp_path / "main"
    (main / "sub").mkdir(parents=True)
    git(main, "init", "-q", "-b", "main")
    git(main, "commit", "-q", "--allow-empty", "-m", "x")
    git(main, "worktree", "add", "-q", str(tmp_path / "other"))
    return main, tmp_path / "other"


@pytest.fixture(params=["flock", "python"])
def env(request):
    """The environment for each lock path: flock(1), or the Python fallback (forced here; macOS has no flock)."""
    if request.param == "flock" and shutil.which("flock") is None:
        pytest.skip("no flock(1) on this machine")
    extra = {} if request.param == "flock" else {"GATE_NO_FLOCK": "1", "GATE_PYTHON": sys.executable}
    return {**os.environ, **extra}


class Gate:
    """A gated command running in the background, its stderr read on a thread so a test can wait for a line without
    hanging if it never comes."""

    def __init__(self, cwd: pathlib.Path, env: dict, script: str) -> None:
        # Its own session, so its process group holds everything it starts (a disowned child too), for teardown.
        self.proc = subprocess.Popen(
            [str(GATED), "bash", "-c", script],
            cwd=cwd,
            env=env,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        self.lines: queue.Queue[str] = queue.Queue()
        stderr = self.proc.stderr
        assert stderr is not None
        threading.Thread(target=lambda: [self.lines.put(line) for line in stderr], daemon=True).start()

    def said(self, text: str, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while (left := deadline - time.monotonic()) > 0:
            try:
                if text in self.lines.get(timeout=left):
                    return True
            except queue.Empty:
                return False
        return False


def wait_for(path: pathlib.Path, text: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists() or text not in path.read_text():
        if time.monotonic() > deadline:
            raise AssertionError(f"{text!r} never appeared in {path}")
        time.sleep(0.02)


def hold(release: pathlib.Path) -> str:
    """A shell loop that waits for the test to create `release`, and gives up after 60 s on its own, so a test that
    dies before releasing it (or a runner killed outright) can't leave it spinning."""
    return f"for _ in $(seq 3000); do [ -e {release} ] && break; sleep 0.02; done"


@pytest.fixture
def release(tmp_path):
    """The file that lets a test's held gates finish; created at teardown whatever the test did."""
    path = tmp_path / "release"
    yield path
    path.touch()


@pytest.fixture
def gate():
    """Starts gated commands, and at teardown kills each one's whole process group (the command and anything it
    started, a disowned child included), whether the test passed or not."""
    started: list[Gate] = []

    def start(cwd: pathlib.Path, env: dict, script: str) -> Gate:
        g = Gate(cwd, env, script)
        started.append(g)
        return g

    yield start
    # Kill every group before waiting on any, so one stuck process can't spare the rest.
    for g in started:
        try:
            os.killpg(g.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # everything in it has already exited
    for g in started:
        g.proc.wait(timeout=10)


def test_gates_in_two_worktrees_run_one_after_the_other(repo, env, tmp_path, gate, release):
    main, other = repo
    log = tmp_path / "log"
    # The first gate holds the lock until the test releases it, so the second is certain to find it held.
    first = gate(main, env, f"echo a-start >> {log}; {hold(release)}; echo a-end >> {log}")
    wait_for(log, "a-start")
    second = gate(other, env, f"echo b-start >> {log}; echo b-end >> {log}")
    assert second.said(WAITING)
    assert "b-start" not in log.read_text()
    release.touch()
    assert first.proc.wait(timeout=10) == 0
    assert second.proc.wait(timeout=10) == 0
    assert log.read_text().split() == ["a-start", "a-end", "b-start", "b-end"]


def test_the_lock_lasts_as_long_as_anything_the_gate_started(repo, env, tmp_path, gate, release):
    # On both paths the command holds the lock itself, so a child it leaves running keeps the gate shut until it
    # exits, and the gate never opens while any of it is still running.
    main, other = repo
    started = tmp_path / "started"
    first = gate(main, env, f"(echo up > {started}; {hold(release)}) & disown")
    assert first.proc.wait(timeout=10) == 0
    wait_for(started, "up")
    second = gate(other, env, "true")
    assert second.said(WAITING)
    release.touch()
    assert second.proc.wait(timeout=10) == 0


def test_a_gate_started_in_a_subdirectory_takes_the_same_lock(repo, env, tmp_path, gate, release):
    main, other = repo
    log = tmp_path / "log"
    first = gate(main / "sub", env, f"echo a >> {log}; {hold(release)}")
    wait_for(log, "a")
    second = gate(other, env, "true")
    assert second.said(WAITING)
    release.touch()
    assert first.proc.wait(timeout=10) == 0
    assert second.proc.wait(timeout=10) == 0


def test_a_free_gate_runs_at_once_and_passes_on_the_exit_code(repo, env):
    main, _ = repo
    done = subprocess.run([str(GATED), "bash", "-c", "exit 3"], cwd=main, env=env, capture_output=True, text=True)
    assert done.returncode == 3
    assert "waiting" not in done.stderr
