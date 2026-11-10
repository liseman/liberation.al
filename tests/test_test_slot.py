"""tools/test-slot.sh: at most TEST_SLOTS non-Rust test runs at once machine-wide, each at the lowest priority."""

import json
import os
import pathlib
import subprocess
import sys
import time

import pytest

TEST_SLOT = pathlib.Path(__file__).resolve().parents[1] / "tools" / "test-slot.sh"


@pytest.fixture
def repo(tmp_path: pathlib.Path) -> pathlib.Path:
    subprocess.run(["git", "init", "-q", str(tmp_path / "repo")], check=True)
    return tmp_path / "repo"


def hold(repo: pathlib.Path, marker: pathlib.Path, release: pathlib.Path) -> subprocess.Popen[str]:
    """A wrapped command that says it started (a file) and runs until `release` exists."""
    program = (
        "import pathlib,sys,time;pathlib.Path(sys.argv[1]).touch()\n"
        "while not pathlib.Path(sys.argv[2]).exists(): time.sleep(0.05)"
    )
    return subprocess.Popen(
        [str(TEST_SLOT), sys.executable, "-c", program, str(marker), str(release)],
        cwd=repo,
        env=os.environ | {"TEST_SLOTS": "2"},
        stderr=subprocess.PIPE,
        text=True,
    )


def wait_for(path: pathlib.Path, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


def test_two_runs_share_the_slots_and_a_third_waits_for_one_to_finish(repo, tmp_path):
    release = [tmp_path / f"release{i}" for i in range(3)]
    started = [tmp_path / f"started{i}" for i in range(3)]
    runs = [hold(repo, started[i], release[i]) for i in range(2)]
    try:
        assert wait_for(started[0]) and wait_for(started[1]), "two slots: both run at once"
        third = hold(repo, started[2], release[2])
        runs.append(third)
        time.sleep(1.5)
        assert not started[2].exists(), "a third run waits while both slots are held"
        release[0].touch()
        assert wait_for(started[2]), "it starts once a slot frees"
    finally:
        for r in release:
            r.touch()
        for run in runs:
            run.wait(timeout=20)
    stderr = runs[2].stderr
    assert stderr is not None and "waiting for a test slot" in stderr.read()


def test_the_command_runs_at_the_lowest_priority_with_its_arguments_stdin_and_exit_status(repo):
    program = (
        "import json,os,sys;print(json.dumps([os.getpriority(os.PRIO_PROCESS,0),sys.argv[1:],sys.stdin.read()]));"
        "sys.exit(7)"
    )
    done = subprocess.run(
        [str(TEST_SLOT), sys.executable, "-c", program, "space here", "$(literal)", ""],
        cwd=repo,
        input="stdin for the command",
        capture_output=True,
        text=True,
    )
    assert done.returncode == 7
    assert json.loads(done.stdout) == [19, ["space here", "$(literal)", ""], "stdin for the command"]
