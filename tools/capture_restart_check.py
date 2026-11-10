"""Stop and start capture once, and say how each device's teardown went and whether it came back.

On the Mac, stopping and starting Continuum has wedged the UVC cameras (a hub port cycle fixed them) and the
reSpeaker (a coreaudiod restart fixed it). Each capture worker logs its teardown step by step, and this reads
those lines for one stop and start:

    .venv/bin/python tools/capture_restart_check.py

It stops continuum-capture, waits for every capture process to exit, starts it, waits `--settle` seconds (30), and
then prints, for each device:

- its teardown: each step's time and the total, or the step it was still in when it was killed;
- whether the supervisor had to kill it ("ignored the stop") or gave up on it ("did not exit after SIGKILL");
- whether it opened again after the start (a camera's size and frame period, or a microphone's rate).

A device whose teardown finished but didn't come back points at the OS (recover it as docs/CONTINUUM-CAPTURE.md says). A
teardown that hung names the step, and that is the bug to fix. Send the whole output either way. It exits 1 when a
device didn't come back or a teardown didn't finish.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

WORKER = re.compile(r"capture worker ([^:\s]+): (.*)$")
SUPERVISOR_START = re.compile(r"capture supervisor: starting worker (\d+) for (\S+) on")
KILLED = re.compile(r"capture supervisor: worker (\d+) ignored the stop: killing it")
UNKILLABLE = "capture supervisor: ERROR workers did not exit after SIGKILL"
STEP = re.compile(r"teardown (.+) took (\d+) ms$")
STILL = re.compile(r"teardown still in (.+) after (\d+) ms$")
DONE = re.compile(r"teardown done in (\d+) ms$")
EARLIER_BYTES = 4 << 20  # how far back to look for the supervisor's worker numbers
OPENED = re.compile(r"^(\d+x\d+ \S+ at \d+ ns a frame.*|\d+ Hz, \d+ channels.*)$")


@dataclass
class Device:
    """What one stop and start showed for one device."""

    name: str
    steps: list[tuple[str, int]] = field(default_factory=list)
    done_ms: int | None = None
    stuck: tuple[str, int] | None = None  # the last "still in" step, and for how long
    killed: bool = False
    opened: str | None = None  # the line it logged on opening after the start

    @property
    def ok(self) -> bool:
        return self.done_ms is not None and not self.killed and self.opened is not None


@dataclass
class Report:
    devices: dict[str, Device]
    unkillable: bool = False

    @property
    def ok(self) -> bool:
        return not self.unkillable and all(d.ok for d in self.devices.values())


def read(stop_log: str, start_log: str, earlier: str = "") -> Report:
    """The report from the log lines written while capture stopped (`stop_log`) and after it started (`start_log`).
    `earlier` is the log before the stop, where the supervisor said which device each worker number runs."""
    devices: dict[str, Device] = {}
    workers = {m[1]: m[2] for m in SUPERVISOR_START.finditer(earlier)}  # worker number to device name
    unkillable = False

    def device(name: str) -> Device:
        return devices.setdefault(name, Device(name))

    for line in stop_log.splitlines():
        if m := SUPERVISOR_START.search(line):
            workers[m[1]] = m[2]
        elif m := KILLED.search(line):
            device(workers.get(m[1], f"worker {m[1]}")).killed = True
        elif UNKILLABLE in line:
            unkillable = True
        elif m := WORKER.search(line):
            name, rest = m[1], m[2]
            if s := DONE.search(rest):
                device(name).done_ms = int(s[1])
            elif s := STILL.search(rest):
                device(name).stuck = (s[1], int(s[2]))
            elif s := STEP.search(rest):
                device(name).steps.append((s[1], int(s[2])))
            elif rest == "teardown begins":
                device(name)
    for line in start_log.splitlines():
        if (m := WORKER.search(line)) and OPENED.match(m[2]) and m[1] in devices:
            devices[m[1]].opened = devices[m[1]].opened or m[2]
    return Report(devices=devices, unkillable=unkillable)


def describe(report: Report) -> str:
    """The report as text, one block per device."""
    if not report.devices:
        return "No teardown lines: capture didn't stop, or this build predates the teardown log."
    out = []
    for d in sorted(report.devices.values(), key=lambda d: d.name):
        steps = ", ".join(f"{name} {ms} ms" for name, ms in d.steps) or "no steps logged"
        if d.done_ms is not None:
            teardown = f"finished in {d.done_ms} ms ({steps})"
        elif d.stuck is not None:
            teardown = f"HUNG in {d.stuck[0]} ({d.stuck[1]} ms and counting; done so far: {steps})"
        else:
            teardown = f"never finished ({steps})"
        lines = [f"{d.name}: {'ok' if d.ok else 'PROBLEM'}", f"  teardown: {teardown}"]
        if d.killed:
            lines.append("  the supervisor killed it: it ignored the stop for stop_grace_ms")
        lines.append(f"  after the start: {d.opened or 'did NOT open again'}")
        if d.done_ms is not None and not d.killed and d.opened is None:
            lines.append("  its teardown finished cleanly, so the OS kept the device wedged: recover it as docs say")
        out.append("\n".join(lines))
    if report.unkillable:
        out.append("The supervisor gave up on a worker after SIGKILL: check the OS's device state before retrying.")
    return "\n".join(out)


def _capture_running() -> bool:
    """Whether any continuum-capture process (the supervisor or a worker) is still running, by executable name."""
    ps = subprocess.run(["ps", "-axo", "comm="], capture_output=True, text=True, check=True).stdout
    return any(Path(line.strip()).name.startswith("continuum-capture") for line in ps.splitlines())


def _services(verb: str) -> None:
    pack = Path(sys.executable).with_name("pack-services")
    subprocess.run([str(pack), verb, "continuum-capture"], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, default=Path("data/logs/continuum-capture.log"))
    parser.add_argument("--settle", type=float, default=30.0, help="seconds to wait after the start (30)")
    parser.add_argument("--stop-wait", type=float, default=40.0, help="seconds to wait for capture to exit (40)")
    args = parser.parse_args()
    if not args.log.exists():
        sys.exit(f"no capture log at {args.log}: run this from the checkout, or pass --log")
    before = args.log.stat().st_size
    print("stopping continuum-capture ...", flush=True)
    _services("stop")
    until = time.monotonic() + args.stop_wait
    while _capture_running() and time.monotonic() < until:
        time.sleep(0.5)
    if _capture_running():
        print(f"capture processes were still running {args.stop_wait:.0f} s after the stop", flush=True)
    middle = args.log.stat().st_size
    print("starting continuum-capture ...", flush=True)
    _services("start")
    time.sleep(args.settle)
    with args.log.open("rb") as f:
        f.seek(max(0, before - EARLIER_BYTES))
        earlier = f.read(before - f.tell()).decode(errors="replace")
        stop_log = f.read(middle - before).decode(errors="replace")
        start_log = f.read().decode(errors="replace")
    report = read(stop_log, start_log, earlier)
    print(describe(report))
    print(f"({os.uname().sysname} {os.uname().release}; log {args.log})")
    sys.exit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
