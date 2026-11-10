"""The Rust build wrapper serializes worktrees without changing command priority or arguments.

Run without application fixtures: pytest --noconftest tests/test_build_slot.py.
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

import test_gated as gates

pytest_plugins = ("test_gated",)

BUILD_SLOT = pathlib.Path(__file__).resolve().parents[1] / "tools" / "build-slot.sh"


@pytest.fixture(autouse=True)
def build_wrapper(monkeypatch):
    monkeypatch.setenv("RUSTC_WRAPPER", "")
    monkeypatch.setattr(gates, "GATED", BUILD_SLOT)
    monkeypatch.setattr(gates, "WAITING", "waiting for another Rust build")


def test_builds_in_two_worktrees_run_one_after_the_other(repo, env, tmp_path, gate, release):
    gates.test_gates_in_two_worktrees_run_one_after_the_other(repo, env, tmp_path, gate, release)


def test_build_lock_is_inherited_by_children(repo, env, tmp_path, gate, release):
    gates.test_the_lock_lasts_as_long_as_anything_the_gate_started(repo, env, tmp_path, gate, release)


@pytest.mark.parametrize(("jobs", "expected"), [(None, "8"), ("3", "3")])
def test_default_jobs_override_priority_and_literal_arguments(repo, env, jobs, expected):
    main, _ = repo
    command_env = env | {"RUSTC_WRAPPER": ""}
    command_env.pop("CARGO_BUILD_JOBS", None)
    if jobs is not None:
        command_env["CARGO_BUILD_JOBS"] = jobs
    program = "import json,os,sys;print(json.dumps([os.environ['CARGO_BUILD_JOBS'],os.getpriority(os.PRIO_PROCESS,0),sys.argv[1:]]))"
    done = subprocess.run(
        [str(BUILD_SLOT), sys.executable, "-c", program, "space here", "$(literal)", ""],
        cwd=main / "sub",
        env=command_env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(done.stdout) == [expected, os.getpriority(os.PRIO_PROCESS, 0), ["space here", "$(literal)", ""]]


def test_exit_status_and_stdin_are_preserved(repo, env):
    main, _ = repo
    program = "import sys;print(sys.stdin.read());sys.exit(7)"
    done = subprocess.run(
        [str(BUILD_SLOT), sys.executable, "-c", program],
        cwd=main,
        env=env | {"RUSTC_WRAPPER": ""},
        input="input for the wrapped command",
        capture_output=True,
        text=True,
    )
    assert done.returncode == 7
    assert done.stdout.strip() == "input for the wrapped command"


def test_sccache_startup_never_calls_priority_wrappers(repo, env, tmp_path):
    main, _ = repo
    commands = tmp_path / "bin"
    commands.mkdir()
    started = tmp_path / "started"
    demoted = tmp_path / "demoted"
    cache = commands / "sccache"
    cache.write_text(
        f"#!{sys.executable}\nimport pathlib,sys\npathlib.Path({str(started)!r}).write_text(sys.argv[1])\n"
    )
    cache.chmod(0o755)
    demotion = f"#!{sys.executable}\nimport pathlib\npathlib.Path({str(demoted)!r}).touch()\n"
    (commands / "nice").write_text(demotion)
    (commands / "nice").chmod(0o755)
    (commands / "ionice").write_text(demotion)
    (commands / "ionice").chmod(0o755)
    subprocess.run(
        [str(BUILD_SLOT), sys.executable, "-c", "pass"],
        cwd=main,
        env=env | {"RUSTC_WRAPPER": str(cache), "PATH": str(commands) + os.pathsep + env["PATH"]},
        check=True,
        capture_output=True,
        text=True,
    )
    assert not demoted.exists()
    assert started.read_text() == "--start-server"


@pytest.mark.skipif(shutil.which("flock") is None, reason="cross-backend test needs flock")
def test_flock_and_fcntl_contend_on_the_same_lock(repo, tmp_path, gate, release):
    main, other = repo
    log = tmp_path / "log"
    first = gate(main, os.environ | {"GATE_NO_FLOCK": ""}, f"echo first >> {log}; {gates.hold(release)}")
    gates.wait_for(log, "first")
    second = gate(
        other,
        os.environ | {"GATE_NO_FLOCK": "1", "GATE_PYTHON": sys.executable},
        f"echo second >> {log}",
    )
    assert second.said("waiting for another Rust build")
    assert log.read_text().split() == ["first"]
    release.touch()
    assert first.proc.wait(timeout=10) == 0
    assert second.proc.wait(timeout=10) == 0
    assert log.read_text().split() == ["first", "second"]
