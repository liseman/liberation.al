"""Opt-in, finite CPU-weighted stack sampling for the current process (#1743).

Linux procfs counters are exact OS clock ticks, read independently (not an atomic process snapshot). Python stacks
are instantaneous statistical evidence: a thread's CPU since the previous tick is assigned to its sampled stack
only when its current procfs state is R, not measured inside each function. Each Python thread's state/counter read
is immediately followed by a fresh frame lookup before reading the next task. These are adjacent, not atomic;
scheduling and the GIL can still bias selection. Non-running endpoints retain their CPU as explicitly
unattributed, without redistribution. Native code retains its Python caller when one is visible and running;
threads without a Python mapping remain separately unattributed. Exited, newly observed, reused, or unreadable threads never acquire
invented deltas. The signed process-minus-observed-thread residual includes those gaps and counter read skew.

A separate wait-site table answers where threads block between short wakeups (#3180). Each thread's voluntary
context switch delta from /proc/<tid>/status counts its wakeups since the previous sample; when its endpoint is
not running, those wakeups and its non-running CPU ticks are recorded against that wait-site stack, whose leaf
carries the blocking call's current line. This table never feeds the CPU function estimates or collapsed stacks.

Only code locations are retained, never frame objects, source text, locals, arguments, or thread names. Collection
is bounded and ends without stopping the service. Importing this module does nothing; unsupported platforms fail
actionably only when start is requested. Artifacts are private, atomic cumulative snapshots with bounded rotation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Any

from liberational import logs

_LOG = logs.get("service-profile")
_STACK_BYTES = 2 * 1024 * 1024
_MAX_THREADS = 4096
_UNREAD = -1


@dataclass(frozen=True)
class _Counter:
    start: int
    ticks: int
    state: str = "R"
    voluntary: int = _UNREAD  # voluntary context switches: one per block, so one per later wakeup


@dataclass(frozen=True)
class _Snapshot:
    process: _Counter
    threads: dict[int, _Counter]
    missing: tuple[int, ...] = ()


def _stat(text: str) -> _Counter:
    """Parse stat despite spaces and parentheses in comm; suffix field 3 starts after the last closing paren."""
    end = text.rfind(")")
    if end < 0:
        raise ValueError("procfs stat has no comm terminator")
    fields = text[end + 1 :].split()
    if len(fields) < 20:
        raise ValueError("procfs stat is incomplete")
    return _Counter(start=int(fields[19]), ticks=int(fields[11]) + int(fields[12]), state=fields[0])


def _voluntary(text: str) -> int:
    """Read voluntary_ctxt_switches from a procfs status file; absent or malformed counts stay unread."""
    for line in text.splitlines():
        if line.startswith("voluntary_ctxt_switches:"):
            value = line.partition(":")[2].strip()
            return int(value) if value.isdigit() else _UNREAD
    return _UNREAD


class _LinuxCounters:
    def __init__(self) -> None:
        if not sys.platform.startswith("linux") or not Path("/proc/self/task").is_dir():
            raise RuntimeError(
                "CPU stack profiling requires Linux procfs (/proc/self/task); leave profiling disabled here"
            )
        self.tick_hz = int(os.sysconf("SC_CLK_TCK"))

    def snapshot(self) -> _Snapshot:
        return self._read(None)[0]

    def sample(self, depth: int) -> tuple[_Snapshot, dict[int, tuple[str, ...]], dict[int, int]]:
        """Pair each Python task's procfs read with a fresh frame snapshot before the next task's I/O.

        The third mapping is each Python thread's innermost current line, which locates a blocking call site.
        """
        return self._read(depth)

    def _read(self, depth: int | None) -> tuple[_Snapshot, dict[int, tuple[str, ...]], dict[int, int]]:
        identities = (
            {
                thread.native_id: thread.ident
                for thread in threading.enumerate()
                if thread.native_id is not None and thread.ident is not None
            }
            if depth is not None
            else {}
        )
        process = _stat(Path("/proc/self/stat").read_text())
        threads: dict[int, _Counter] = {}
        stacks: dict[int, tuple[str, ...]] = {}
        lines: dict[int, int] = {}
        missing = []
        for task in Path("/proc/self/task").iterdir():
            tid = int(task.name)
            try:
                voluntary = _voluntary((task / "status").read_text())
                counter = _stat((task / "stat").read_text())
            except OSError, ValueError:
                missing.append(tid)
                continue
            threads[tid] = _Counter(counter.start, counter.ticks, counter.state, voluntary)
            ident = identities.get(tid)
            if depth is not None and ident is not None:
                frames = sys._current_frames()
                try:
                    if ident in frames:
                        stacks[tid] = _stack(frames[ident], depth)
                        lines[tid] = frames[ident].f_lineno or 0
                finally:
                    frames.clear()  # release immediately, before another task's stat read releases the GIL
        return _Snapshot(process, threads, tuple(missing)), stacks, lines


def _stack(frame: FrameType, depth: int) -> tuple[str, ...]:
    locations = []
    current: FrameType | None = frame
    while current is not None and len(locations) < depth:
        code = current.f_code
        location = f"{code.co_filename[-256:]}:{code.co_firstlineno}:{code.co_qualname[-128:]}"
        locations.append(location.replace(";", ":").replace("\n", " ").replace("\r", " "))
        current = current.f_back
    if current is not None:
        locations.append("[stack truncated]")
    return tuple(reversed(locations))


class _Accumulator:
    def __init__(self, initial: _Snapshot, sampler: int, max_functions: int, max_stacks: int) -> None:
        self.previous = initial
        self.sampler = sampler
        self.max_functions, self.max_stacks = max_functions, max_stacks
        self.stacks: dict[tuple[str, ...], int] = {}
        self.functions: dict[str, dict[str, int]] = {}
        self.thread_ticks: dict[tuple[int, int], int] = {}
        self.native_ticks: dict[tuple[int, int], int] = {}
        self.nonrunning_ticks: dict[tuple[int, int], int] = {}
        self.thread_switches: dict[tuple[int, int], int] = {}
        self.waits: dict[tuple[str, ...], dict[str, int]] = {}
        self.wait_threads: dict[tuple[str, ...], set[tuple[int, int]]] = {}
        self.wait_bytes = 0
        self.stack_bytes = 0
        self.totals = dict.fromkeys(
            (
                "process_ticks",
                "observed_thread_ticks",
                "excluded_sampler_ticks",
                "python_attributed_ticks",
                "native_unattributed_ticks",
                "nonrunning_unattributed_ticks",
                "stack_overflow_ticks",
                "function_overflow_inclusive_ticks",
                "function_overflow_self_ticks",
                "thread_overflow_ticks",
                "new_thread_baselines",
                "exited_threads",
                "reused_or_reset_threads",
                "unreadable_thread_samples",
                "samples",
                "voluntary_switches",
                "excluded_sampler_switches",
                "wait_site_wakeups",
                "running_endpoint_wakeups",
                "native_wakeups",
                "wait_overflow_wakeups",
            ),
            0,
        )

    def add(self, snapshot: _Snapshot, stacks: dict[int, tuple[str, ...]], lines: dict[int, int] | None = None) -> None:
        prior = self.previous
        if snapshot.process.start != prior.process.start or snapshot.process.ticks < prior.process.ticks:
            raise ValueError("process CPU counter identity changed or reset")
        self.previous = snapshot
        self.totals["samples"] += 1
        self.totals["process_ticks"] += max(0, snapshot.process.ticks - prior.process.ticks)
        self.totals["exited_threads"] += len(set(prior.threads) - set(snapshot.threads) - set(snapshot.missing))
        self.totals["unreadable_thread_samples"] += len(snapshot.missing)
        for tid, counter in snapshot.threads.items():
            old = prior.threads.get(tid)
            if old is None:
                self.totals["new_thread_baselines"] += 1
                continue
            if counter.start != old.start or counter.ticks < old.ticks:
                self.totals["reused_or_reset_threads"] += 1
                continue
            delta = counter.ticks - old.ticks
            wakeups = (
                counter.voluntary - old.voluntary
                if counter.voluntary >= old.voluntary and old.voluntary != _UNREAD
                else 0
            )
            self.totals["observed_thread_ticks"] += delta
            if tid == self.sampler:
                self.totals["excluded_sampler_ticks"] += delta
                self.totals["excluded_sampler_switches"] += wakeups
                continue
            self.totals["voluntary_switches"] += wakeups
            key = (tid, counter.start)
            if key in self.thread_ticks or len(self.thread_ticks) < _MAX_THREADS:
                self.thread_ticks[key] = self.thread_ticks.get(key, 0) + delta
                if wakeups:
                    self.thread_switches[key] = self.thread_switches.get(key, 0) + wakeups
            else:
                self.totals["thread_overflow_ticks"] += delta
            if not delta and not wakeups:
                continue
            stack = stacks.get(tid)
            if not stack:
                self.totals["native_unattributed_ticks"] += delta
                self.totals["native_wakeups"] += wakeups
                if delta and key in self.thread_ticks:
                    self.native_ticks[key] = self.native_ticks.get(key, 0) + delta
                continue
            if counter.state != "R":
                self.totals["nonrunning_unattributed_ticks"] += delta
                if delta and key in self.thread_ticks:
                    self.nonrunning_ticks[key] = self.nonrunning_ticks.get(key, 0) + delta
                self._wait(stack, (lines or {}).get(tid), key, delta, wakeups)
                continue
            self.totals["running_endpoint_wakeups"] += wakeups
            if not delta:
                continue
            self.totals["python_attributed_ticks"] += delta
            stored_stack = (f"[thread {tid} start {counter.start}]", *stack)
            size = sum(len(location.encode()) for location in stored_stack)
            if stored_stack not in self.stacks and (
                len(self.stacks) >= self.max_stacks or self.stack_bytes + size > _STACK_BYTES
            ):
                self.totals["stack_overflow_ticks"] += delta
            else:
                if stored_stack not in self.stacks:
                    self.stack_bytes += size
                self.stacks[stored_stack] = self.stacks.get(stored_stack, 0) + delta
            for location in set(stack):
                if location == "[stack truncated]":
                    continue
                if location not in self.functions:
                    if len(self.functions) >= self.max_functions:
                        self.totals["function_overflow_inclusive_ticks"] += delta
                        if location == stack[-1]:
                            self.totals["function_overflow_self_ticks"] += delta
                        continue
                    self.functions[location] = {"inclusive_ticks_estimate": 0, "self_ticks_estimate": 0}
                self.functions[location]["inclusive_ticks_estimate"] += delta
                if location == stack[-1]:
                    self.functions[location]["self_ticks_estimate"] += delta

    def _wait(self, stack: tuple[str, ...], line: int | None, key: tuple[int, int], ticks: int, wakeups: int) -> None:
        """Record wakeups and non-running ticks at a blocking endpoint; never touches the CPU function tables."""
        site = (*stack[:-1], f"{stack[-1]} at line {line}") if line else stack
        if site not in self.waits:
            size = sum(len(location.encode()) for location in site)
            if len(self.waits) >= self.max_stacks or self.wait_bytes + size > _STACK_BYTES:
                self.totals["wait_overflow_wakeups"] += wakeups
                return
            self.wait_bytes += size
            self.waits[site] = {"wakeups": 0, "nonrunning_ticks": 0, "samples": 0}
            self.wait_threads[site] = set()
        row = self.waits[site]
        row["wakeups"] += wakeups
        row["nonrunning_ticks"] += ticks
        row["samples"] += 1
        self.totals["wait_site_wakeups"] += wakeups
        if len(self.wait_threads[site]) < _MAX_THREADS:
            self.wait_threads[site].add(key)

    def report(self) -> dict[str, Any]:
        return {
            **self.totals,
            "sampler_native_tid": self.sampler,
            "process_minus_observed_thread_ticks": self.totals["process_ticks"] - self.totals["observed_thread_ticks"],
            "functions": self.functions,
            "threads": [
                {
                    "native_tid": tid,
                    "start_ticks": start,
                    "cpu_ticks": ticks,
                    "native_or_unmapped_ticks": self.native_ticks.get((tid, start), 0),
                    "nonrunning_ticks": self.nonrunning_ticks.get((tid, start), 0),
                    "voluntary_switches": self.thread_switches.get((tid, start), 0),
                }
                for (tid, start), ticks in self.thread_ticks.items()
            ],
            "wait_sites": [
                {"stack": list(site), **row, "threads": len(self.wait_threads[site])}
                for site, row in sorted(self.waits.items(), key=lambda item: (-item[1]["wakeups"], item[0]))
            ],
        }


def _atomic(path: Path, data: str) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".profile-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as out:
            os.fchmod(out.fileno(), 0o600)
            out.write(data)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class Profiler:
    """Finite opt-in sampler. start returns self; stop joins and flushes; wait observes automatic completion.

    An instance captures once. Counter and artifact errors end profiling, log an error, and leave the service alive.
    `error` records that failure for a launcher that wants to surface it. Existing artifact files rotate per service.
    """

    def __init__(
        self,
        service: str,
        output: Path,
        interval_s: float = 0.05,
        flush_s: float = 60,
        duration_s: float = 120,
        delay_s: float = 0,
        *,
        max_depth: int = 64,
        max_functions: int = 2048,
        max_stacks: int = 2048,
        max_artifacts: int = 3,
    ) -> None:
        if not isinstance(service, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", service):
            raise ValueError("service must be a short name containing letters, numbers, dots, underscores, or hyphens")
        for name, value, lower, upper in (
            ("interval_s", interval_s, 0.005, 60),
            ("flush_s", flush_s, 0.005, 3600),
            ("duration_s", duration_s, 0.005, 3600),
            ("delay_s", delay_s, 0, 3600),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or not lower <= value <= upper
            ):
                raise ValueError(f"{name} must be finite and between {lower} and {upper}")
        for name, value, upper in (
            ("max_depth", max_depth, 128),
            ("max_functions", max_functions, 4096),
            ("max_stacks", max_stacks, 4096),
            ("max_artifacts", max_artifacts, 10),
        ):
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f"{name} must be an integer between 1 and {upper}")
        self.service, self.output = service, Path(output)
        self.interval_s, self.flush_s, self.duration_s = interval_s, flush_s, duration_s
        self.delay_s = delay_s
        self.max_depth, self.max_functions, self.max_stacks, self.max_artifacts = (
            max_depth,
            max_functions,
            max_stacks,
            max_artifacts,
        )
        self.error: str | None = None
        self._thread: threading.Thread | None = None
        self._stop, self._done = threading.Event(), threading.Event()
        self._lifecycle = threading.Lock()
        self._prefix = "profile-" + hashlib.sha256(service.encode()).hexdigest()[:16] + "-"

    @property
    def running(self) -> bool:
        return self._thread is not None and not self._done.is_set()

    def start(self) -> Profiler:
        with self._lifecycle:
            if self._thread is not None:
                raise RuntimeError("this profile capture has already started")
            reader = _LinuxCounters()  # explicit unsupported-platform error before starting any thread
            self.output.mkdir(parents=True, exist_ok=True, mode=0o700)
            thread = threading.Thread(target=self._run, args=(reader,), name="service-profile", daemon=True)
            thread.start()
            self._thread = thread
        return self

    def stop(self) -> None:
        with self._lifecycle:
            self._stop.set()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)

    def _write(
        self, accumulator: _Accumulator, identity: dict[str, Any], tick_hz: int, elapsed: float, final: bool
    ) -> None:
        stem = self._prefix + f"{time.time_ns():020d}-{identity['capture_id']}"
        report = {
            "version": 1,
            "identity": identity,
            "tick_hz": tick_hz,
            "elapsed_s": elapsed,
            "final": final,
            "interval_s": self.interval_s,
            "duration_s": self.duration_s,
            "delay_s": self.delay_s,
            "stack_filter": "running-only",
            "attribution_method": "thread-cpu-delta-running-state-v3",
            "stack_pairing": "per-python-thread-adjacent",
            "stack_scope": "native-thread",
            "wait_method": "voluntary-switch-delta-at-nonrunning-endpoint-v1",
            "method": "Exact OS CPU tick deltas; only threads sampled in procfs state R receive Python stack estimates. CPU at non-running Python endpoints remains unattributed, never redistributed. Each Python thread's state/counter read is immediately followed by a fresh frame lookup before reading the next task. These reads are not atomic; scheduling and the GIL can bias attribution. Native work may be assigned to its running visible Python caller.",
            "limits": {
                "depth": self.max_depth,
                "functions": self.max_functions,
                "stacks": self.max_stacks,
                "stack_bytes": _STACK_BYTES,
            },
            **accumulator.report(),
        }
        collapsed = [";".join(stack) + f" {ticks}\n" for stack, ticks in sorted(accumulator.stacks.items())]
        for field in ("native_unattributed_ticks", "nonrunning_unattributed_ticks", "stack_overflow_ticks"):
            if accumulator.totals[field]:
                collapsed.append(f"[{field}] {accumulator.totals[field]}\n")
        collapsed_path = self.output / (stem + ".collapsed")
        _atomic(collapsed_path, "".join(collapsed))
        try:
            _atomic(self.output / (stem + ".json"), json.dumps(report, sort_keys=True))
        except Exception:
            collapsed_path.unlink(missing_ok=True)
            raise
        for old in sorted(self.output.glob(self._prefix + "*.json"))[: -self.max_artifacts]:
            old.with_suffix(".collapsed").unlink(missing_ok=True)
            old.unlink(missing_ok=True)

    def _run(self, reader: _LinuxCounters) -> None:
        try:
            if self._stop.wait(self.delay_s):
                return
            initial = reader.snapshot()
            started = time.monotonic()
            identity = {
                "service": self.service,
                "pid": os.getpid(),
                "process_start_ticks": initial.process.start,
                "capture_id": uuid.uuid4().hex,
                "started_at_ns": time.time_ns(),
            }
            accumulator = _Accumulator(initial, threading.get_native_id(), self.max_functions, self.max_stacks)
            flush_at = started + self.flush_s
            while True:
                elapsed = time.monotonic() - started
                stopped = self._stop.wait(max(0, min(self.interval_s, self.duration_s - elapsed)))
                snapshot, stacks, lines = reader.sample(self.max_depth)
                accumulator.add(snapshot, stacks, lines)
                now = time.monotonic()
                final = stopped or now - started >= self.duration_s
                if final or now >= flush_at:
                    self._write(accumulator, identity, reader.tick_hz, now - started, final)
                    flush_at = now + self.flush_s
                if final:
                    return
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            _LOG.error("CPU stack capture ended: %s", self.error)
        finally:
            self._done.set()
