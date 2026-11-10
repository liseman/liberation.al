"""A watchdog for slow dashboard requests (#433): what a request is waiting on, said while it waits.

The log already says when a request took long (SLOW_REQUEST_S in server.py), but not why: after a start the
dashboard served /services and /people in 6 to 25 s while using little CPU, so the requests were waiting on
something. Attaching py-spy to the running dashboard isn't allowed here (ptrace), so the dashboard reports itself:
every request is registered while it runs, and a thread looks at them once a second. A request still running after
WATCH_S logs one WARN with its own stack (where it waits: a lock, a subprocess, a file) and a line per other busy
thread (who may hold that lock, or the GIL). Each route is reported at most once per PER_ROUTE_S, so a stall that
holds up every poll costs a few lines a minute, not a flood. A stream (an event stream or the camera mux) is long
on purpose and never reported.

Each request's own CPU time (its thread's, from begin to end) is added to its route's tally, and every TALLY_S the
watchdog logs one INFO line with the routes that used the most (#2418). A thread's CPU tells request work apart from
the background threads' in a /proc sample, but not which route spent it: a browser's keep-alive connection serves
every poll on one thread. A request still running when a tally is taken (an event stream a tab keeps open for hours)
has its CPU so far counted in that window, read from its thread's CPU clock, and the rest when it ends; where a
thread's clock can't be read (no `time.pthread_getcpuclockid`, as on macOS), it's all counted when it ends. It counts
only this process: a route that hands its work to a worker process (the face and voice groups, the People list, and
Reports' days) tallies only its wait, so it ranks low however much the worker spent.
"""

import sys
import threading
import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import FrameType

WATCH_S = 3.0  # a request running longer than this is reported
EVERY_S = 1.0  # how often the watchdog looks
PER_ROUTE_S = 60.0  # at most one report per route this often
STACK_FRAMES = 14  # the innermost frames of the slow request's stack
OTHER_THREADS = 25  # at most this many other threads, one line each
TALLY_S = 600.0  # how often the request CPU tally is logged and reset
TALLY_ROUTES = 8  # the routes named in each tally line


@dataclass
class Request:
    route: str  # "GET /services": the method and path without its query, as the "took" warning logs it
    started: float  # time.monotonic() when its request line was read
    streamed: Callable[[], bool]  # whether it has turned out to be a stream
    reported: bool = False
    cpu: float = 0.0  # its thread's CPU time (time.thread_time) when it began, or when a tally last counted it


@dataclass
class Spent:
    """A route's CPU and finished requests since the last tally line."""

    cpu: float = 0.0
    requests: int = 0


def thread_cpu(ident: int) -> float | None:
    """Thread `ident`'s CPU seconds from its own clock, or None where that can't be read (no
    pthread_getcpuclockid, or a thread that has exited)."""
    clock_id = getattr(time, "pthread_getcpuclockid", None)
    if clock_id is None:
        return None
    try:
        return time.clock_gettime(clock_id(ident))
    except OSError, OverflowError, ValueError:
        return None


@dataclass
class Watch:
    """The requests in flight, by thread, and when each route was last reported."""

    inflight: dict[int, Request] = field(default_factory=dict)
    last: dict[str, float] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    tally: dict[str, Spent] = field(default_factory=dict)  # by route, since the last tally line
    cpu_clock: Callable[[], float] = time.thread_time  # the calling thread's CPU seconds
    other_cpu: Callable[[int], float | None] = thread_cpu  # another thread's, by ident, or None when unreadable

    def begin(self, route: str, streamed: Callable[[], bool], now: float | None = None) -> None:
        """The calling thread has started serving route."""
        started = time.monotonic() if now is None else now
        with self.lock:
            self.inflight[threading.get_ident()] = Request(route, started, streamed, cpu=self.cpu_clock())

    def end(self) -> None:
        """The calling thread is done with its request: its CPU time goes to its route's tally."""
        spent = self.cpu_clock()
        with self.lock:
            r = self.inflight.pop(threading.get_ident(), None)
            if r is not None:
                t = self.tally.setdefault(r.route, Spent())
                t.cpu += max(0.0, spent - r.cpu)
                t.requests += 1

    def take_tally(self, span_s: float) -> str | None:
        """The tally since the last call as one line (the routes that used the most CPU first), and a fresh tally;
        None when nothing was spent. A request still running is counted up to now, and from now on next time."""
        with self.lock:
            for ident, r in self.inflight.items():
                if (now := self.other_cpu(ident)) is not None:
                    self.tally.setdefault(r.route, Spent()).cpu += max(0.0, now - r.cpu)
                    r.cpu = now
            tally, self.tally = self.tally, {}
        if not tally:
            return None
        total = sum(t.cpu for t in tally.values())
        top = sorted(tally.items(), key=lambda kv: -kv[1].cpu)[:TALLY_ROUTES]
        routes = ", ".join(f"{route} {t.cpu:.1f} s/{t.requests}" for route, t in top)
        requests = sum(t.requests for t in tally.values())
        return (
            f"request CPU over {span_s:.0f} s: {total:.1f} s in {requests} requests; by route (CPU/requests): {routes}"
        )

    def check(
        self,
        now: float,
        frames: Mapping[int, FrameType],
        names: Mapping[int, str],
        log: Callable[[str], None],
    ) -> int:
        """Report each request running longer than WATCH_S, once, and each route at most once per PER_ROUTE_S.
        Returns how many were reported."""
        with self.lock:
            due = [
                (ident, r)
                for ident, r in self.inflight.items()
                if not r.reported and now - r.started > WATCH_S and now - self.last.get(r.route, -1e9) > PER_ROUTE_S
            ]
            busy = {ident: (r.route, now - r.started) for ident, r in self.inflight.items()}
        n = 0
        for ident, r in due:
            if now - self.last.get(r.route, -1e9) <= PER_ROUTE_S:
                continue  # another request on this route was just reported, in this same pass
            try:
                if r.streamed():
                    r.reported = True  # a stream is long on purpose
                    continue
            except Exception:
                pass
            frame = frames.get(ident)
            if frame is None:  # it finished in the meantime
                continue
            r.reported = True
            self.last[r.route] = now
            log(report(ident, r, now, frame, frames, names, busy))
            n += 1
        return n


def _where(frame: FrameType) -> str:
    """A thread's innermost frame in our code (or its innermost frame at all), as "file:line in function"."""
    inner = frame
    f: FrameType | None = frame
    while f is not None:
        if "/liberational/" in f.f_code.co_filename and "site-packages" not in f.f_code.co_filename:
            inner = f
            break
        f = f.f_back
    name = inner.f_code.co_filename.rsplit("/", 1)[-1]
    top = f" (now in {frame.f_code.co_name})" if inner is not frame else ""
    return f"{name}:{inner.f_lineno} in {inner.f_code.co_name}{top}"


def report(
    ident: int,
    r: Request,
    now: float,
    frame: FrameType,
    frames: Mapping[int, FrameType],
    names: Mapping[int, str],
    busy: Mapping[int, tuple[str, float]],
) -> str:
    """The WARN for one slow request: its route and age, its stack, then a line per other thread."""
    stack = "".join(traceback.format_stack(frame)[-STACK_FRAMES:]).rstrip()
    others = []
    for other, f in frames.items():
        if other in (ident, threading.get_ident()):
            continue
        who = f"{busy[other][0]} for {busy[other][1]:.1f} s" if other in busy else names.get(other, str(other))
        others.append(f"  {who}: {_where(f)}")
    others.sort(key=lambda line: (not line.startswith(("  GET", "  POST", "  HEAD")), line))
    more = len(others) - OTHER_THREADS
    lines = others[:OTHER_THREADS] + ([f"  ... and {more} more"] if more > 0 else [])
    return (
        f"slow request: {r.route} still running after {now - r.started:.1f} s, "
        f"{len(busy)} in flight. Its stack:\n{stack}\nOther threads:\n" + "\n".join(lines)
    )


WATCH = Watch()


def watch(log: Callable[[str], None], info: Callable[[str], None] | None = None) -> threading.Thread:
    """Look at the requests in flight every EVERY_S, in a daemon thread, logging reports with log (a WARN), and the
    request CPU tally every TALLY_S with info (an INFO)."""

    def run() -> None:
        tallied = time.monotonic()
        while True:
            time.sleep(EVERY_S)
            try:
                names = {t.ident: t.name for t in threading.enumerate() if t.ident is not None}
                WATCH.check(time.monotonic(), sys._current_frames(), names, log)
            except Exception as ex:  # the watchdog must never take the dashboard down
                log(f"slow request watchdog: {type(ex).__name__}: {ex}")
            if info is None or (now := time.monotonic()) - tallied < TALLY_S:
                continue
            span, tallied = now - tallied, now  # a failed tally waits for the next window, not the next second
            try:  # its own try: a check that keeps failing mustn't stop the tally too
                line = WATCH.take_tally(span)
                if line is not None:
                    info(line)
            except Exception as ex:
                log(f"request CPU tally: {type(ex).__name__}: {ex}")

    t = threading.Thread(target=run, daemon=True, name="slow-requests")
    t.start()
    return t
