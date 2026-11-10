"""The slow-request watchdog (dashboard/slow.py), and the request's duration counted from its request line."""

import socket
import sys
import threading
import time
from urllib.parse import urlsplit

import pytest

from liberational.dashboard import server, slow


@pytest.fixture
def stuck():
    """A thread inside wait_for_the_lock until the test lets it go, registered as GET /services from t=100."""
    watch, go, ready = slow.Watch(), threading.Event(), threading.Event()

    def wait_for_the_lock():
        watch.begin("GET /services", lambda: False, now=100.0)
        ready.set()
        go.wait(5)
        watch.end()

    t = threading.Thread(target=wait_for_the_lock, name="request")
    t.start()
    ready.wait(5)
    yield watch, t
    go.set()
    t.join(5)


def check(watch, now):
    logged = []
    names = {t.ident: t.name for t in threading.enumerate() if t.ident is not None}
    watch.check(now, sys._current_frames(), names, logged.append)
    return logged


def test_a_request_still_running_after_a_few_seconds_says_where_it_waits(stuck):
    watch, _ = stuck
    assert check(watch, 100.0 + slow.WATCH_S - 0.5) == []  # not yet
    [report] = check(watch, 100.0 + slow.WATCH_S + 1)
    assert report.startswith("slow request: GET /services still running after 4.0 s, 1 in flight")
    assert "in wait_for_the_lock" in report  # its own stack
    assert "Other threads:" in report


def test_each_request_and_each_route_is_reported_once_a_minute_at_most(stuck):
    watch, _ = stuck
    assert len(check(watch, 105.0)) == 1
    assert check(watch, 110.0) == []  # the same request again: never
    watch.begin("GET /services", lambda: False, now=106.0)  # another poll of the same route, from this thread
    assert check(watch, 120.0) == []  # within the minute
    assert len(check(watch, 105.0 + slow.PER_ROUTE_S + 1)) == 1  # after it
    watch.end()


def test_a_stream_is_never_reported(stuck):
    watch, _ = stuck
    watch.begin("GET /events", lambda: True, now=100.0)
    assert [r.split(" still")[0] for r in check(watch, 110.0)] == ["slow request: GET /services"]
    watch.end()


def test_many_stalled_requests_on_one_route_are_one_report(stuck):
    watch, _ = stuck
    go = threading.Event()

    def poll():
        watch.begin("GET /services", lambda: False, now=100.0)
        go.wait(5)
        watch.end()

    more = [threading.Thread(target=poll) for _ in range(4)]
    for t in more:
        t.start()
    while len(watch.inflight) < 5:
        time.sleep(0.01)
    try:
        assert len(check(watch, 104.5)) == 1  # five tabs stuck on the same poll: one report
    finally:
        go.set()
        for t in more:
            t.join(5)
    assert len(watch.inflight) == 1  # each one's end() took it out


class Warnings:
    def __init__(self):
        self.lines, self.errors = [], []

    def warning(self, msg, *a, **k):
        self.lines.append(msg)

    def error(self, msg, *a, **k):
        self.errors.append(msg)

    def __getattr__(self, name):
        return lambda *a, **k: None


def test_the_server_registers_a_request_while_it_runs(serve_dashboard, monkeypatch):
    # The watch is process-wide: a fresh one keeps another test's request still winding down out of this count
    monkeypatch.setattr(slow, "WATCH", slow.Watch())
    inside, go = threading.Event(), threading.Event()

    def slow_get(self):
        inside.set()
        go.wait(5)
        self._json({"ok": True})

    monkeypatch.setattr(server.H, "do_GET", slow_get)
    port = urlsplit(serve_dashboard()).port
    with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
        s.sendall(b"GET /people/Alice?q=1 HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n")
        assert inside.wait(5)
        assert [r.route for r in slow.WATCH.inflight.values()] == ["GET /people/Alice"]  # never the query
        go.set()
        assert s.recv(64).startswith(b"HTTP/1.0 200")
    deadline = time.monotonic() + 5
    while slow.WATCH.inflight and time.monotonic() < deadline:
        time.sleep(0.01)
    assert slow.WATCH.inflight == {}


def test_an_early_connection_closed_unused_is_not_a_client_giving_up(serve_dashboard, monkeypatch):
    log = Warnings()
    monkeypatch.setattr(server, "LOG", log)
    monkeypatch.setattr(server, "SLOW_REQUEST_S", 0.2)
    port = urlsplit(serve_dashboard()).port
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    time.sleep(0.4)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00")  # reset, as a browser does
    s.close()
    time.sleep(0.2)
    assert [line for line in log.errors if "gave up" in line] == []


def test_a_request_counts_from_its_request_line_not_from_the_open_connection(serve_dashboard, monkeypatch):
    log = Warnings()
    monkeypatch.setattr(server, "LOG", log)
    monkeypatch.setattr(server, "SLOW_REQUEST_S", 0.3)
    port = urlsplit(serve_dashboard()).port
    with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
        time.sleep(0.6)  # a browser's early connection, idle before it sends
        s.sendall(b"GET /health HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n")
        assert s.recv(64).startswith(b"HTTP/1.0 200")
    time.sleep(0.1)
    assert [line for line in log.lines if "took" in line] == []


def test_a_playing_video_is_long_on_purpose_once_its_headers_are_out():
    h = object.__new__(server.H)
    h._content_type, h._headers_at = "video/mp4", None
    assert not h._long_on_purpose()  # still working out the reply: a stall the watchdog should see
    h._headers_at = 1.0
    assert h._long_on_purpose()  # the browser holds it open while it plays
    h._content_type = "application/json"
    assert not h._long_on_purpose()


def test_a_client_that_gave_up_on_a_slow_json_reply_is_still_named(serve_dashboard, monkeypatch):
    """_send takes a client leaving quietly, while a slow request it left is still logged as slow."""
    log = Warnings()
    monkeypatch.setattr(server, "LOG", log)
    monkeypatch.setattr(server, "SLOW_REQUEST_S", 0.2)

    def slow_get(self):
        time.sleep(0.5)
        self._json({"ok": True})

    monkeypatch.setattr(server.H, "do_GET", slow_get)
    port = urlsplit(serve_dashboard()).port
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    s.sendall(b"GET /people HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n")
    time.sleep(0.1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00")  # reset, as a browser does
    s.close()
    deadline = time.monotonic() + 3
    while not [line for line in log.errors if "gave up" in line] and time.monotonic() < deadline:
        time.sleep(0.02)
    assert [line for line in log.errors if "gave up" in line] == [
        next(line for line in log.errors if "gave up" in line)
    ]
    assert "GET /people: the client gave up after" in log.errors[0]
    assert [line for line in log.lines if "took" in line] == []  # one line about it, not a warning as well


def test_each_route_tallies_its_requests_cpu_and_the_busiest_route_comes_first():
    clock = iter([1.0, 1.5, 2.0, 4.0, 4.0, 4.25])  # thread CPU at begin and end of three requests
    watch = slow.Watch(cpu_clock=lambda: next(clock))
    for route in ("GET /people", "GET /voices", "GET /people"):
        watch.begin(route, lambda: False, now=0.0)
        watch.end()
    line = watch.take_tally(600.0)
    assert line == (
        "request CPU over 600 s: 2.8 s in 3 requests; by route (CPU/requests): GET /voices 2.0 s/1, GET /people 0.8 s/2"
    )


def test_the_tally_starts_over_once_taken_and_is_none_with_no_requests():
    watch = slow.Watch(cpu_clock=lambda: 0.0)
    assert watch.take_tally(600.0) is None
    watch.begin("GET /people", lambda: False, now=0.0)
    watch.end()
    assert watch.take_tally(600.0) is not None
    assert watch.take_tally(600.0) is None


def test_an_end_without_a_begin_tallies_nothing():
    watch = slow.Watch(cpu_clock=lambda: 0.0)
    watch.end()  # a connection closed before its request line arrived
    assert watch.take_tally(600.0) is None


def test_a_request_still_running_is_tallied_up_to_now_and_the_rest_when_it_ends():
    own = iter([10.0, 25.0])  # its thread's CPU when it began, and when it ended
    watch = slow.Watch(cpu_clock=lambda: next(own), other_cpu=lambda ident: 18.0)
    watch.begin("GET /events", lambda: True, now=0.0)
    assert watch.take_tally(600.0) == (
        "request CPU over 600 s: 8.0 s in 0 requests; by route (CPU/requests): GET /events 8.0 s/0"
    )
    watch.end()
    assert watch.take_tally(600.0) == (
        "request CPU over 600 s: 7.0 s in 1 requests; by route (CPU/requests): GET /events 7.0 s/1"
    )


def test_a_running_request_whose_thread_clock_cant_be_read_is_all_tallied_when_it_ends():
    own = iter([10.0, 25.0])
    watch = slow.Watch(cpu_clock=lambda: next(own), other_cpu=lambda ident: None)
    watch.begin("GET /events", lambda: True, now=0.0)
    assert watch.take_tally(600.0) is None
    watch.end()
    assert watch.take_tally(600.0) == (
        "request CPU over 600 s: 15.0 s in 1 requests; by route (CPU/requests): GET /events 15.0 s/1"
    )


@pytest.mark.skipif(not hasattr(time, "pthread_getcpuclockid"), reason="no per-thread CPU clocks here")
def test_this_threads_own_cpu_clock_is_readable_by_its_ident():
    assert slow.thread_cpu(threading.get_ident()) == pytest.approx(time.thread_time(), abs=0.05)
