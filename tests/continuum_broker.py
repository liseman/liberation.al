"""Temporary real brokers for tests, using a capability-checked local build.

CONTINUUM_BROKER wins, followed by checkout/main release builds, the shared debug
build, and checkout debug. Missing or obsolete builds produce an actionable skip;
startup failures retain stderr rather than masquerading as unavailable fixtures.
"""

import os
import pathlib
import re
import shutil
import socket as py_socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from typing import BinaryIO

import pytest


def _candidates() -> tuple[list[pathlib.Path], list[str]]:
    repo = pathlib.Path(__file__).resolve().parents[1]
    candidates = []
    if override := os.environ.get("CONTINUUM_BROKER"):
        candidates.append(pathlib.Path(override))
    candidates.append(repo / "continuum/target/release/continuum-broker")
    rejected = []
    shared = repo / ".git"
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"], cwd=repo, capture_output=True, text=True, timeout=2
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        rejected.append(f"git common-directory discovery failed: {error}; using {shared}")
    else:
        if common.returncode == 0 and common.stdout.strip():
            shared = (repo / common.stdout.strip()).resolve()
        else:
            rejected.append(
                f"git common-directory discovery exit {common.returncode}: {common.stderr.strip()}; using {shared}"
            )
    candidates.extend(
        [
            shared.parent / "continuum/target/release/continuum-broker",
            shared / "continuum-target/debug/continuum-broker",
        ]
    )
    candidates.append(repo / "continuum/target/debug/continuum-broker")
    return list(dict.fromkeys(candidates)), rejected


def broker_binary() -> str:
    """Select a binary supporting the tiny test arena, or skip with every rejected path/reason."""
    pytest.importorskip("continuum_py", reason="continuum_py unavailable: run continuum-test")
    candidates, rejected = _candidates()
    for candidate in candidates:
        path = candidate.resolve()
        if not path.is_file():
            rejected.append(f"{path}: missing")
            continue
        try:
            help_result = subprocess.run([path, "--help"], capture_output=True, text=True, timeout=2)
        except (OSError, subprocess.TimeoutExpired) as error:
            rejected.append(f"{path}: --help failed: {error}")
            print(f"broker fixture: rejected {rejected[-1]}", file=sys.stderr)
            continue
        output = help_result.stdout + help_result.stderr
        missing = [
            flag for flag in ("--socket", "--tier0-classes") if not re.search(rf"(?<![\w-]){flag}(?![\w-])", output)
        ]
        if help_result.returncode != 0 or missing:
            rejected.append(
                f"{path}: --help exit {help_result.returncode}; missing {', '.join(missing) or 'none'}; {output[-500:].strip()}"
            )
            print(f"broker fixture: rejected {rejected[-1]}", file=sys.stderr)
            continue
        return str(path.resolve())
    pytest.skip("no compatible continuum-broker (--socket and --tier0-classes required): " + "; ".join(rejected))


def _startup_error(binary: str, process: subprocess.Popen[bytes], stderr: BinaryIO, reason: str) -> str:
    stderr.seek(0, os.SEEK_END)
    stderr.seek(max(0, stderr.tell() - 8192))
    detail = stderr.read().decode(errors="replace").strip()
    return f"broker {binary}: {reason}; exit={process.poll()}; stderr: {detail or '(empty)'}"


#: The thread the test itself runs on (conftest points this at each test): the only thread where pytest's skip can
#: be reported. Every other thread a test has going (a request being answered, a worker the test started) has no test
#: to skip, so a missing build there is an ordinary error and the work fails or answers as an install with no broker
#: does, instead of a skip that ends somebody else's thread.
TEST_THREAD: threading.Thread | None = None


def from_the_test() -> bool:
    """Whether this is the thread the test runs on, where a skip means "this test needs a broker"."""
    return TEST_THREAD is not None and threading.current_thread() is TEST_THREAD


def broker_or_skip() -> str:
    """`broker_binary`, but a skip only on the thread the test runs on. A request being answered, or a worker the test
    started, has no test to skip: raising pytest's Skipped there drops a connection or kills a worker, so a missing
    build is an ordinary error on those threads."""
    try:
        return broker_binary()
    except pytest.skip.Exception as skip:
        if from_the_test():
            raise
        raise RuntimeError(str(skip)) from None


def start_broker(tmp_path: pathlib.Path) -> Iterator[pathlib.Path]:
    """Yield a bounded tiny broker on a short socket path; always reap it and remove the socket directory."""
    binary = broker_or_skip()
    die_with_parent = ["setpriv", "--pdeathsig", "KILL"] if sys.platform == "linux" and shutil.which("setpriv") else []
    with tempfile.TemporaryDirectory(prefix="cb-") as short, tempfile.TemporaryFile() as stderr:
        socket = pathlib.Path(short) / "b.sock"
        process = subprocess.Popen(
            [*die_with_parent, binary, "--socket", socket, "--tier0-classes", "16384x16"],
            stderr=stderr,
            cwd=tmp_path,
        )
        try:
            deadline = time.monotonic() + 10
            while True:
                with py_socket.socket(py_socket.AF_UNIX) as probe:
                    try:
                        probe.connect(str(socket))
                        break
                    except OSError:
                        assert process.poll() is None, _startup_error(
                            binary, process, stderr, "exited before listening"
                        )
                        assert time.monotonic() < deadline, _startup_error(
                            binary, process, stderr, "listening timed out"
                        )
                        time.sleep(0.01)
            yield socket
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
