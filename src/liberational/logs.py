"""One log format for every service: `<date>T<time><offset> [tag] LEVEL message` (stamp), written to stderr (services.py sends each service's
stdout and stderr to logs/<service>.log, which the Debug tab tails, and `trim` below keeps each under a cap). Stderr keeps a CLI's stdout for its output.

LEVEL is DEBUG, INFO, WARN, or ERROR:

- ERROR: something failed and needs attention (a crash, a model that won't load, a write that failed).
- WARN: degraded, retrying, or falling back (a service it needs is down, a retry, a camera reconnecting).
- INFO: normal progress (started, a pass done, counts).
- DEBUG: per-item detail, off unless asked for.

The tag is the logger's name (`get("backfill")` writes `[backfill]`), so a module's lines say where they come from in
whichever process runs it. A traceback (`log.exception`, or `exc_info=True`) follows its line, and the Debug tab reads
its lines (no time, no level) as the level of the line before.

The level is `logging.level`, or this service's entry in `logging.per_service` ("backfill=debug"). The service is the
running `pack-<service>` command, as services.toml names it. The first logger set up configures the package's
loggers, so a daemon, a CLI, and a test all log the same way without calling anything first. The root logger is left
alone: a process that configures it (logging.basicConfig) keeps that, and other libraries keep Python's default of
warnings and errors only, in the one format. Services run with third-party progress bars off (services.py), since a
bar redrawn in a log file is noise.
"""

import datetime as dt
import logging
import os
import pathlib
import sys
import threading
from typing import NamedTuple, TextIO

from liberational import config as cfg

LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "warning": logging.WARNING,  # Python's own spelling
    "error": logging.ERROR,
    "critical": logging.ERROR,
}
DEBUG, INFO, WARN, ERROR = logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR
_NAMES = {logging.DEBUG: "DEBUG", logging.INFO: "INFO", logging.WARNING: "WARN"}
_ROOT = "liberational"
_lock = threading.Lock()
_handler: logging.Handler | None = None


class _Stderr(logging.StreamHandler):
    """Writes to whatever sys.stderr is when a line is written, so a test capturing stderr sees the lines."""

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stderr
        super().emit(record)


def level_name(levelno: int) -> str:
    """The word a line carries for a level: anything from ERROR up is ERROR, below DEBUG is DEBUG."""
    if levelno >= logging.ERROR:
        return "ERROR"
    return _NAMES.get(levelno) or ("WARN" if levelno > logging.INFO else "INFO" if levelno > logging.DEBUG else "DEBUG")


def stamp(ts: float) -> str:
    """A log line's time: ISO 8601 local date and time to the second with its UTC offset (2026-11-03T02:20:15-08:00).
    Lines carried only the time before the date was added, so a log a few days long read as one day."""
    return dt.datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")


class Formatter(logging.Formatter):
    """`<stamp> [tag] LEVEL message` (stamp: ISO local date and time with offset), with any traceback after it."""

    def format(self, record: logging.LogRecord) -> str:
        name = record.name.removeprefix(_ROOT + ".") if record.name != _ROOT else record.name
        line = f"{stamp(record.created)} [{name}] {level_name(record.levelno)} "
        line += record.getMessage()
        if record.exc_info:
            record.exc_text = record.exc_text or self.formatException(record.exc_info)
        if record.exc_text:
            line += "\n" + record.exc_text
        if record.stack_info:
            line += "\n" + self.formatStack(record.stack_info)
        return line


def service() -> str:
    """The service this process runs: `pack-backfill` is backfill. Anything else (a test, `python -m`) is its stem."""
    return pathlib.Path(sys.argv[0] if sys.argv and sys.argv[0] else "python").stem.removeprefix("pack-")


def level_for(name: str | None = None) -> int:
    """The configured level for a service (this one by default): its `logging.per_service` entry, else
    `logging.level`. An unknown word is INFO, so a typo in pack.toml never silences a service."""
    name = name or service()
    level = str(cfg.get("logging.level") or "info").strip().lower()
    entries = cfg.get("logging.per_service") or []
    for entry in [entries] if isinstance(entries, str) else entries:  # one "service=level" written as a string
        svc, _, lv = str(entry).partition("=")
        if svc.strip() == name and lv.strip():
            level = lv.strip().lower()
    return LEVELS.get(level, logging.INFO)


def setup(stream: TextIO | None = None, level: int | None = None) -> logging.Handler:
    """Send this package's log lines to stream (stderr, whatever it is at the time) in the one format, at the
    configured level, in place of an earlier setup. Only the package's own loggers: other libraries keep Python's
    default (warnings and errors to stderr), and the root logger is left for whoever configures it. Returns the
    handler, for tests."""
    global _handler
    with _lock:
        handler = logging.StreamHandler(stream) if stream is not None else _Stderr(sys.stderr)
        handler.setFormatter(Formatter())
        lib = logging.getLogger(_ROOT)
        for h in list(lib.handlers):
            lib.removeHandler(h)
        lib.addHandler(handler)
        lib.setLevel(level_for() if level is None else level)
        lib.propagate = False  # one line per record: ours, never a second copy from the root's handlers
        # Another library's warning, in a process nobody configured logging for, goes to Python's last-resort
        # handler, which prints the bare message. Give it the one format instead, still at WARNING and up only.
        # A process that configures the root (logging.basicConfig) never reaches it.
        last = logging.StreamHandler(stream) if stream is not None else _Stderr(sys.stderr)
        last.setFormatter(Formatter())
        last.setLevel(logging.WARNING)
        logging.lastResort = last
        _handler = handler
        return handler


def adopt(name: str) -> None:
    """Give another library's logger (uvicorn) this package's handler and format, in place of the root's."""
    if _handler is None:
        setup()
    lib = logging.getLogger(name)
    for h in list(lib.handlers):
        lib.removeHandler(h)
    if _handler is not None:
        lib.addHandler(_handler)
    lib.propagate = False


def get(tag: str) -> logging.Logger:
    """The logger that writes `[tag]`. The first one sets the process up."""
    if _handler is None:
        setup()
    return logging.getLogger(f"{_ROOT}.{tag}")


def reason(error: BaseException) -> str:
    """What a log line says went wrong: the exception's text, else its type's name. A bare TimeoutError (a wait that
    ran out) has no text, and logged as is it read "voice search unavailable ()"."""
    return str(error) or type(error).__name__


# Rotation (#58). A service's log is written by a process that holds it open in append mode (the systemd wrapper's
# `>>`, launchd's StandardOutPath), so it can't be renamed away: the writer would follow the rename. `trim` copies
# and truncates instead, as logrotate's copytruncate does: an append-mode writer's next line goes to the start of
# the emptied file. `pack-services trim-logs` runs it hourly (the log-trim service).
CAP_BYTES = 10 * 1024 * 1024  # the most a log file keeps, current and older alike
OLDER = ".1"  # the one older file kept beside each log: <name>.log.1


def older(path: pathlib.Path) -> pathlib.Path:
    """The kept older file of a log."""
    return path.with_name(path.name + OLDER)


def rotate(path: pathlib.Path, cap: int = CAP_BYTES) -> int:
    """If a log is past cap bytes, move its last cap bytes (from a line start) to its older file, replacing what
    was there, and empty it in place. Returns the bytes it freed, 0 when it was under the cap or missing.

    A line written between the copy and the truncate is lost: the window is one read long."""
    try:
        f = path.open("r+b")
    except OSError:
        return 0
    with f:
        size = os.fstat(f.fileno()).st_size
        if size <= cap:
            return 0
        f.seek(size - cap - 1)  # one byte more, to tell whether the window begins at a line's start
        data = f.read()  # to the end, so lines written since the size was taken are kept
        f.truncate(0)
    data = data[-(cap + 1) :]
    cut = data.find(b"\n")
    data = data[cut + 1 :] if 0 <= cut < len(data) - 1 else data[1:]  # from the first whole line in the window
    kept = older(path)
    temp = kept.with_name(kept.name + ".tmp")
    temp.write_bytes(data)
    os.replace(temp, kept)
    return size - len(data)


def trim(directory: pathlib.Path, cap: int = CAP_BYTES) -> dict[str, int]:
    """Rotate every log in directory that is past cap. Returns the bytes freed by file name, for each it rotated."""
    freed = {}
    for path in sorted(directory.glob("*.log")):
        if path.is_file() and (n := rotate(path, cap)):
            freed[path.name] = n
    return freed


class Part(NamedTuple):
    """A piece of a log's tail: which file it's from (its stat), where in that file it starts, and its bytes."""

    stat: os.stat_result
    offset: int
    data: bytes


def tail(path: pathlib.Path, limit: int) -> list[Part]:
    """The last limit bytes of a log, oldest first: the end of its older file when the current one is shorter than
    limit (just after a rotation), then the current one. Empty when neither can be read."""
    parts: list[Part] = []
    want = limit
    for file in (path, older(path)):
        try:
            with file.open("rb") as f:
                stat = os.fstat(f.fileno())
                begin = max(0, stat.st_size - want)
                f.seek(begin)
                data = f.read(stat.st_size - begin)
        except OSError:
            if file == path:
                continue
            break
        parts.insert(0, Part(stat, begin, data))
        want -= len(data)
        if want <= 0 or begin > 0:
            break
    return parts
