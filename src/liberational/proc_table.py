"""This machine's processes, each read once: pid, parent, name, and command line, for the scans that find which
processes are Liberational's (services.processes) and name them (the dashboard's /stats).

Reading every process's command line on each scan was most of the Resource monitor's cost (about 1,000 processes on
the Linux machine, 70 to 90 ms of CPU a scan). Listing the pids is nearly free, so `table()` lists them each time, drops the ones
that are gone, and reads only the new ones. A process younger than SETTLE_S when it was read may not have exec'd its
own program yet (a fork still shows its parent's command line), so it's read again on the next call until it's older.

What this trades: a pid that disappears and comes back between two calls (Linux hands out pids in order, so that takes a
whole cycle of the pid space) keeps the first process's entry, and a parent read once doesn't follow a reparenting.
"""

import threading
import time
from dataclasses import dataclass

import psutil

SETTLE_S = 2.0  # a process read younger than this is read again


@dataclass(frozen=True)
class Proc:
    pid: int
    ppid: int | None
    name: str
    cmdline: tuple[str, ...]
    created: float  # psutil's create_time
    read_at: float  # when this entry was read (time.time)


_lock = threading.Lock()
_table: dict[int, Proc] = {}


def _read(pid: int, now: float) -> Proc | None:
    """One process's entry, or None when it's gone. A field it may not read is left empty, as process_iter leaves it."""
    try:
        p = psutil.Process(pid)
        with p.oneshot():
            created = p.create_time()
            values: dict[str, object] = {}
            for field, read in (("name", p.name), ("cmdline", p.cmdline), ("ppid", p.ppid)):
                try:
                    values[field] = read()
                except psutil.AccessDenied, psutil.ZombieProcess:
                    values[field] = None
    except psutil.NoSuchProcess:  # gone (a zombie that can't say its creation time is gone too)
        return None
    except psutil.AccessDenied:
        return None
    ppid, cmdline = values["ppid"], values["cmdline"]
    return Proc(
        pid=pid,
        ppid=ppid if isinstance(ppid, int) else None,
        name=str(values["name"] or ""),
        cmdline=tuple(str(a) for a in cmdline) if isinstance(cmdline, list) else (),
        created=created,
        read_at=now,
    )


def table() -> dict[int, Proc]:
    """Every process now, by pid: new ones read, gone ones dropped, settled ones as they were read."""
    now = time.time()
    pids = set(psutil.pids())
    with _lock:
        for pid in [pid for pid in _table if pid not in pids]:
            del _table[pid]
        for pid in pids:
            known = _table.get(pid)
            if known is not None and known.read_at - known.created >= SETTLE_S:
                continue
            entry = _read(pid, now)
            if entry is None:
                _table.pop(pid, None)
            else:
                _table[pid] = entry
        return dict(_table)


def clear() -> None:
    """Forget every entry (tests)."""
    with _lock:
        _table.clear()
