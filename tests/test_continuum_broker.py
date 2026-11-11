"""Broker fixture discovery never skips a compatible local build or hides startup errors."""

import os
import pathlib
import sys
from pathlib import Path

import pytest

import continuum_broker as helper


def executable(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!{sys.executable}\n{body}\n")
    path.chmod(0o755)
    return path


def test_release_build_is_used_without_environment_override(tmp_path, monkeypatch):
    repo = tmp_path / "checkout"
    (repo / "tests").mkdir(parents=True)
    monkeypatch.setattr(helper, "__file__", str(repo / "tests" / "continuum_broker.py"))
    monkeypatch.delenv("CONTINUUM_BROKER", raising=False)
    release = executable(repo / "continuum/target/release/continuum-broker", 'print("--socket --tier0-classes")')
    try:
        selected = helper.broker_binary()
    except pytest.skip.Exception as error:
        pytest.fail(f"compatible release was skipped: {error}")
    assert Path(selected) == release


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    repo = tmp_path / "checkout"
    (repo / "tests").mkdir(parents=True)
    monkeypatch.setattr(helper, "__file__", str(repo / "tests/continuum_broker.py"))
    monkeypatch.delenv("CONTINUUM_BROKER", raising=False)
    return repo


def test_compatible_explicit_override_wins_over_release(checkout, tmp_path, monkeypatch):
    override = executable(tmp_path / "override", 'print("--socket --tier0-classes")')
    executable(checkout / "continuum/target/release/continuum-broker", 'print("--socket --tier0-classes")')
    monkeypatch.setenv("CONTINUUM_BROKER", str(override))
    assert helper.broker_binary() == str(override)


def test_obsolete_override_is_rejected_and_release_runs(checkout, tmp_path, monkeypatch, capsys):
    obsolete = executable(tmp_path / "old", 'print("--socket")')
    release = executable(checkout / "continuum/target/release/continuum-broker", 'print("--socket --tier0-classes")')
    monkeypatch.setenv("CONTINUUM_BROKER", str(obsolete))
    assert helper.broker_binary() == str(release)
    rejected = capsys.readouterr().err
    assert str(obsolete) in rejected and "missing --tier0-classes" in rejected


def test_worktree_finds_main_release_before_obsolete_shared_debug(checkout, tmp_path, monkeypatch):
    import subprocess

    main = tmp_path / "main"
    common = main / ".git"
    common.mkdir(parents=True)
    release = executable(main / "continuum/target/release/continuum-broker", 'print("--socket --tier0-classes")')
    executable(common / "continuum-target/debug/continuum-broker", 'print("--socket")')
    run = helper.subprocess.run

    def git_or_help(args, **kwargs):
        if args[0] == "git":
            return subprocess.CompletedProcess(args, 0, str(common), "")
        return run(args, **kwargs)

    monkeypatch.setattr(helper.subprocess, "run", git_or_help)
    assert helper.broker_binary() == str(release)


def test_compatible_checkout_debug_remains_a_fallback(checkout):
    debug = executable(checkout / "continuum/target/debug/continuum-broker", 'print("--socket --tier0-classes")')
    assert helper.broker_binary() == str(debug)


def test_no_compatible_binary_reports_paths_and_rejection_reasons(checkout, tmp_path, monkeypatch):
    obsolete = executable(tmp_path / "old", 'print("--socket --tier0-classes-extra")')
    monkeypatch.setenv("CONTINUUM_BROKER", str(obsolete))
    with pytest.raises(pytest.skip.Exception) as caught:
        helper.broker_binary()
    message = str(caught.value)
    assert str(obsolete) in message and "missing --tier0-classes" in message
    assert str(checkout / "continuum/target/release/continuum-broker") in message and "missing" in message


def test_nonexecutable_candidate_has_an_actionable_rejection(checkout, tmp_path, monkeypatch):
    invalid = tmp_path / "not-executable"
    invalid.write_text("not a program")
    monkeypatch.setenv("CONTINUUM_BROKER", str(invalid))
    with pytest.raises(pytest.skip.Exception, match="Permission denied"):
        helper.broker_binary()


def test_startup_failure_includes_binary_exit_and_stderr_and_removes_socket_dir(tmp_path, monkeypatch):
    binary = executable(
        tmp_path / "bad-start",
        """import sys
from pathlib import Path
if "--help" in sys.argv:
    print("--socket --tier0-classes")
else:
    Path("attempted-socket").write_text(sys.argv[sys.argv.index("--socket") + 1])
    print("invalid arena setting", file=sys.stderr, flush=True)
    sys.exit(7)""",
    )
    monkeypatch.setenv("CONTINUUM_BROKER", str(binary))
    with pytest.raises(AssertionError) as caught:
        next(helper.start_broker(tmp_path))
    assert str(binary) in str(caught.value)
    assert "exit=7" in str(caught.value) and "invalid arena setting" in str(caught.value)
    assert not Path((tmp_path / "attempted-socket").read_text()).parent.exists()


def test_temporary_real_broker_fixture_listens_and_cleans_up(broker):
    import socket

    assert broker.exists() and len(str(broker)) < 108
    with socket.socket(socket.AF_UNIX) as probe:
        probe.connect(str(broker))


def test_relative_override_still_starts_after_changing_to_fixture_directory(checkout, tmp_path, monkeypatch):
    binary = executable(
        tmp_path / "relative-broker",
        """import socket, sys, time
if "--help" in sys.argv:
    print("--socket --tier0-classes")
else:
    with socket.socket(socket.AF_UNIX) as listener:
        listener.bind(sys.argv[sys.argv.index("--socket") + 1])
        listener.listen()
        time.sleep(60)""",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CONTINUUM_BROKER", "./relative-broker")
    working = tmp_path / "different-working-directory"
    working.mkdir()
    try:
        selected = helper.broker_binary()
    except pytest.skip.Exception as error:
        pytest.fail(f"compatible relative override was skipped: {error}")
    assert selected == str(binary.resolve())
    generator = helper.start_broker(working)
    socket_path = next(generator)
    try:
        assert socket_path.exists()
    finally:
        assert next(generator, None) is None
    assert not socket_path.parent.exists()


@pytest.mark.parametrize("failure", [FileNotFoundError("git missing"), helper.subprocess.TimeoutExpired("git", 2)])
def test_git_discovery_failure_does_not_hide_compatible_release(checkout, monkeypatch, failure):
    release = executable(checkout / "continuum/target/release/continuum-broker", 'print("--socket --tier0-classes")')
    run = helper.subprocess.run

    def help_without_git(args, **kwargs):
        if args[0] == "git":
            raise failure
        return run(args, **kwargs)

    monkeypatch.setattr(helper.subprocess, "run", help_without_git)
    assert helper.broker_binary() == str(release)
    release.unlink()
    with pytest.raises(pytest.skip.Exception, match="git common-directory discovery failed"):
        helper.broker_binary()


def test_broker_that_never_listens_has_bounded_timeout_diagnostics(tmp_path, monkeypatch):
    from types import SimpleNamespace

    binary = executable(
        tmp_path / "stalled",
        """import sys, time
if "--help" in sys.argv:
    print("--socket --tier0-classes")
else:
    time.sleep(60)""",
    )
    monkeypatch.setenv("CONTINUUM_BROKER", str(binary))
    clock = iter((0, 11))
    monkeypatch.setattr(helper, "time", SimpleNamespace(monotonic=lambda: next(clock), sleep=lambda _: None))
    with pytest.raises(AssertionError) as caught:
        next(helper.start_broker(tmp_path))
    assert str(binary) in str(caught.value) and "listening timed out" in str(caught.value)
    assert "exit=None" in str(caught.value) and "stderr:" in str(caught.value)


def test_a_missing_build_only_skips_on_the_thread_running_the_test(checkout):
    """A skip can only be reported on the thread the test runs on. Every other thread a test has going (a request
    being answered, a worker doing broker work for the test) gets an ordinary error: a skip there ended a connection
    or a worker instead of the test."""
    import threading

    def from_that_thread(name: str) -> list[BaseException]:
        seen: list[BaseException] = []

        def run() -> None:
            try:
                helper.broker_or_skip()
            except BaseException as error:  # which signal comes out is the whole point
                seen.append(error)

        thread = threading.Thread(target=run, name=name)
        thread.start()
        thread.join()
        return seen

    for name in ("a-request", "a-worker-the-test-started"):
        assert [type(error) for error in from_that_thread(name)] == [RuntimeError]
    with pytest.raises(pytest.skip.Exception, match="no compatible continuum-broker"):
        helper.broker_or_skip()  # the test's own thread: a test that needs the broker still skips


def test_a_test_that_works_on_a_thread_resolves_the_broker_up_front(needs_broker):
    """`needs_broker` finds the binary on the test's own thread and gives every later start its path, so a thread
    that starts a broker never searches, and never skips."""
    env = os.environ["CONTINUUM_BROKER"]
    assert env == needs_broker
    assert helper._candidates()[0][0] == pathlib.Path(needs_broker)  # the first candidate: no search for a build
