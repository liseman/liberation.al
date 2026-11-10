"""Runs a third-party service (an Ollama server) and writes its output in the pack's one log format.

    python -m liberational.log_relay --tag ollama -- ollama serve

A program we don't own writes its own formats: Ollama's Go server logs `time=... level=INFO source=... msg=...`,
its HTTP router a `[GIN] ...` line a request, and the model runner (llama.cpp) bare lines. The Debug tab reads the
level from `<stamp> [tag] LEVEL message`, so this relay reads the program's stdout and stderr and writes each line
that way, at the level the line states (`level=WARN`, a `W ` prefix) or the one its kind gets: a request line is
DEBUG unless it failed, and a bare runner line is DEBUG unless it says something went wrong. Lines below the
service's level (`logging.level`, `logging.per_service`) are dropped, with the indented lines that continue them.
A line already in the one format passes through as it is (a pack wrapper such as gpu_ready logging before it
execs the program).

The relay forwards SIGTERM, SIGINT, and SIGHUP to the program and exits with its status, so the service manager
stops and restarts the program as it did. services.py puts it in front of a service that sets `log_relay = true`.
"""

import argparse
import re
import signal
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator
from typing import IO, NamedTuple, TextIO

from liberational import logs

OURS = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d \[[^\]]+\] (DEBUG|INFO|WARN|ERROR) ")
SLOG = re.compile(r"^time=\S+ level=(?P<level>[A-Z]+)(?: source=\S+)? (?P<rest>.*)$")
GIN = re.compile(
    r"^\[GIN\] [^|]+\|\s*(?P<status>\d{3})\s*\|\s*(?P<took>[^|]+?)\s*\|[^|]*\|\s*(?P<method>\w+)\s+(?P<path>.*)$"
)
PREFIXED = re.compile(r"^(?P<level>[DIWE]) (?P<rest>.*)$")  # llama.cpp with LLAMA_ARG_LOG_PREFIX
TROUBLE = re.compile(r"\b(error|failed|failure|fatal|panic|out of memory|cannot|unable)\b", re.IGNORECASE)
LEVELS = {"DEBUG": logs.DEBUG, "INFO": logs.INFO, "WARN": logs.WARN, "WARNING": logs.WARN, "ERROR": logs.ERROR}
LETTERS = {"D": logs.DEBUG, "I": logs.INFO, "W": logs.WARN, "E": logs.ERROR}


class Line(NamedTuple):
    """One line of the program's output as this relay writes it: its level and message, or None for a line that
    continues the one before (indented, or a traceback's), written as it is."""

    level: int | None
    text: str


def classify(raw: str) -> Line:
    """The level and message of one line the program wrote."""
    line = raw.rstrip("\r\n")
    if m := OURS.match(line):
        return Line(LEVELS[m[1]], line)
    if not line.strip() or line[:1].isspace():
        return Line(None, line)
    if m := SLOG.match(line):
        rest = m["rest"]
        if rest.startswith('msg="') and (end := rest.find('"', 5)) > 0:
            rest = rest[5:end] + rest[end + 1 :]
        elif rest.startswith("msg="):
            rest = rest[4:]
        return Line(LEVELS.get(m["level"], logs.INFO), rest.strip())
    if m := GIN.match(line):
        status = int(m["status"])
        level = logs.WARN if status >= 500 else logs.DEBUG
        return Line(level, f"{m['method']} {m['path'].strip()} {status} in {m['took']}")
    if m := PREFIXED.match(line):
        return Line(LETTERS[m["level"]], m["rest"])
    return Line(logs.WARN if TROUBLE.search(line) else logs.DEBUG, line)


def relay(lines: Iterable[str], tag: str, threshold: int, out: TextIO) -> None:
    """Writes each line in the one format, dropping those below threshold and their continuations."""
    writing = True
    for raw in lines:
        line = classify(raw)
        if line.level is None:
            if writing:
                out.write(line.text + "\n")
            continue
        writing = line.level >= threshold
        if not writing:
            continue
        if OURS.match(line.text):
            out.write(line.text + "\n")
        else:
            out.write(f"{logs.stamp(time.time())} [{tag}] {logs.level_name(line.level)} {line.text}\n")
        out.flush()


def _decoded(stream: IO[bytes]) -> Iterator[str]:
    for raw in iter(stream.readline, b""):
        yield raw.decode("utf-8", "replace")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a program and write its output in the pack's log format.")
    parser.add_argument("--tag", required=True, help="the service name, as services.toml names it")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- then the program and its arguments")
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no command after --")
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda signum, _frame: child.send_signal(signum))
    assert child.stdout is not None
    relay(_decoded(child.stdout), args.tag, logs.level_for(args.tag), sys.stdout)
    code = child.wait()
    return 128 - code if code < 0 else code  # killed by a signal: the shell's 128 + its number


if __name__ == "__main__":
    sys.exit(main())
