"""A live tail of every service's log, for the dashboard's Debug tab (issue #111).

Each service writes data/logs/<name>.log (services.log_path, the same file under launchd and systemd), so this reads
those files and nothing platform-specific. The service names come from services.toml, never from a request, so a
request can only pick among those files.

A tail starts with a bounded backlog (the last lines of each file, BACKLOG_LINES in all) and then sends new lines
only. A file that is renamed away and replaced (rotation) is read to its end and then the new file from its start; a
file cut shorter in place (truncation) is read again from its start; a line is held until its newline arrives.
"""

import os
import pathlib
from collections.abc import Iterable
from typing import BinaryIO

from liberational import config as cfg
from liberational import services

BACKLOG_LINES = 200  # lines sent on connect, across every file
BACKLOG_BYTES = 64 * 1024  # read from the end of each file for the backlog
READ_BYTES = 256 * 1024  # read from one file per poll, so one busy file can't hold up the others
MAX_LINE = 4000  # characters kept of one line
MAX_PARTIAL = 64 * 1024  # an unterminated line longer than this is sent as it stands


def known() -> dict[str, pathlib.Path]:
    """Every service declared for this platform's log file, by service name, in services.toml order (a Mac-only
    service never has a log here, so it isn't listed as missing one)."""
    here = cfg.platform_key()
    return {
        name: services.log_path(name)
        for name, svc in services.declared().items()
        if not svc.get("platforms") or here in svc["platforms"]
    }


def pick(wanted: Iterable[str] | None = None) -> dict[str, pathlib.Path]:
    """The log files of the named services (every declared one for None). Raises ValueError for a name services.toml
    doesn't declare, so a request can't name a file."""
    every = known()
    if wanted is None:
        return every
    names = list(dict.fromkeys(w for w in wanted if w))
    unknown = [n for n in names if n not in every]
    if unknown:
        raise ValueError("unknown service: " + ", ".join(unknown))
    return {n: every[n] for n in names}


def _line(raw: bytes) -> str:
    # A progress bar redraws itself with carriage returns: keep what a terminal would show, its last state.
    text = raw.decode("utf-8", "replace").rstrip("\r").rsplit("\r", 1)[-1]
    return text if len(text) <= MAX_LINE else text[:MAX_LINE] + " [line cut]"


class FileTail:
    """One log file: where the tail is in it, and the start of a line still waiting for its newline."""

    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        self.fh: BinaryIO | None = None
        self.offset = 0
        self.partial = b""

    def _open(self) -> bool:
        try:
            self.fh = open(self.path, "rb")  # noqa: SIM115 (kept open across polls, to finish a rotated file)
        except OSError:
            self.fh = None
        self.offset, self.partial = 0, b""
        return self.fh is not None

    def start(self) -> list[str]:
        """The file's last whole lines (from at most BACKLOG_BYTES of it), and a position at its end."""
        if not self._open() or self.fh is None:
            return []
        size = os.fstat(self.fh.fileno()).st_size
        begin = max(0, size - BACKLOG_BYTES)
        self.fh.seek(begin)
        data = self.fh.read(size - begin)
        self.offset = begin + len(data)
        lines = data.split(b"\n")
        self.partial = lines.pop()  # a line still being written, sent once its newline arrives
        if begin > 0 and lines:
            lines = lines[1:]  # the read began inside this line
        return [_line(raw) for raw in lines[-BACKLOG_LINES:]]

    def _replaced(self) -> bool:
        """Whether the path now names a different file than the one open (rotation)."""
        if self.fh is None:
            return False
        try:
            now, held = self.path.stat(), os.fstat(self.fh.fileno())
        except OSError:
            return False  # renamed away and not replaced yet: keep reading the open file
        return (now.st_dev, now.st_ino) != (held.st_dev, held.st_ino)

    def poll(self) -> list[str]:
        """The whole lines written since the last call."""
        if self.fh is None and not self._open():
            return []  # not there yet; once it appears it is read from its start
        assert self.fh is not None
        out: list[str] = []
        if os.fstat(self.fh.fileno()).st_size < self.offset:  # cut shorter in place
            self.offset, self.partial = 0, b""
        self.fh.seek(self.offset)
        data = self.fh.read(READ_BYTES)
        self.offset += len(data)
        if data:
            out += self._split(data)
        elif self._replaced():
            # The old file is read to its end: send what is left of its last line, then follow the new file.
            if self.partial:
                out.append(_line(self.partial))
            self.fh.close()
            self._open()
            out += self.poll()
        return out

    def _split(self, data: bytes) -> list[str]:
        lines = (self.partial + data).split(b"\n")
        self.partial = lines.pop()
        if len(self.partial) > MAX_PARTIAL:
            lines.append(self.partial)
            self.partial = b""
        return [_line(raw) for raw in lines]

    def close(self) -> None:
        if self.fh is not None:
            self.fh.close()
            self.fh = None


class Tailer:
    """Every picked file, tailed together. Lines come out as (service, text)."""

    def __init__(self, files: dict[str, pathlib.Path]) -> None:
        self.files = {name: FileTail(path) for name, path in files.items()}

    def start(self) -> list[tuple[str, str]]:
        """The backlog: each file's last lines, the least recently written file first, BACKLOG_LINES in all."""

        def written(name: str) -> float:
            try:
                return self.files[name].path.stat().st_mtime
            except OSError:
                return 0.0

        lines = [(name, text) for name in sorted(self.files, key=written) for text in self.files[name].start()]
        return lines[-BACKLOG_LINES:]

    def poll(self) -> list[tuple[str, str]]:
        return [(name, text) for name, tail in self.files.items() for text in tail.poll()]

    def missing(self) -> list[str]:
        """The services with no log file here (not installed on this machine, or never started)."""
        return [name for name, tail in self.files.items() if not tail.path.exists()]

    def close(self) -> None:
        for tail in self.files.values():
            tail.close()
