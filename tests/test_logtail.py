"""The Debug tab's log tail: dashboard/logtail.py and GET /logs/stream (issue #111)."""

import http.client
import json
import os
import time

import pytest

from liberational import logs as liblogs
from liberational.dashboard import logtail, server, slow


@pytest.fixture
def logs(tmp_path, monkeypatch):
    monkeypatch.setenv("PACK_PATHS_LOGS", str(tmp_path))
    return tmp_path


def write(path, text, mode="a"):
    with open(path, mode) as f:
        f.write(text)


def test_backlog_is_each_files_last_lines_oldest_file_first(logs):
    write(logs / "see-actions.log", "v1\nv2\n")
    write(logs / "listen.log", "a1\na2\n")
    os.utime(logs / "see-actions.log", (1, 1))  # see-actions was written longer ago
    tail = logtail.Tailer(logtail.pick(["listen", "see-actions"]))
    assert tail.start() == [("see-actions", "v1"), ("see-actions", "v2"), ("listen", "a1"), ("listen", "a2")]


def test_backlog_is_capped_in_lines(logs, monkeypatch):
    monkeypatch.setattr(logtail, "BACKLOG_LINES", 3)
    write(logs / "listen.log", "".join(f"a{i}\n" for i in range(10)))
    assert [text for _, text in logtail.Tailer(logtail.pick(["listen"])).start()] == ["a7", "a8", "a9"]


def test_backlog_drops_the_line_its_byte_window_starts_inside(logs, monkeypatch):
    monkeypatch.setattr(logtail, "BACKLOG_BYTES", 8)
    write(logs / "listen.log", "first line\nab\ncd\n")
    assert logtail.FileTail(logs / "listen.log").start() == ["ab", "cd"]


def test_after_the_backlog_only_new_lines_come(logs):
    write(logs / "listen.log", "old\n")
    tail = logtail.FileTail(logs / "listen.log")
    tail.start()
    assert tail.poll() == []
    write(logs / "listen.log", "new 1\nnew 2\n")
    assert tail.poll() == ["new 1", "new 2"]
    assert tail.poll() == []


def test_a_line_waits_for_its_newline(logs):
    write(logs / "listen.log", "")
    tail = logtail.FileTail(logs / "listen.log")
    tail.start()
    write(logs / "listen.log", "half a")
    assert tail.poll() == []
    write(logs / "listen.log", " line\n")
    assert tail.poll() == ["half a line"]


def test_a_line_half_written_at_connect_comes_whole(logs):
    write(logs / "listen.log", "done\nhalf")
    tail = logtail.FileTail(logs / "listen.log")
    assert tail.start() == ["done"]
    write(logs / "listen.log", " of it\n")
    assert tail.poll() == ["half of it"]


def test_a_truncated_file_is_read_again_from_its_start(logs):
    write(logs / "listen.log", "a long first run of output\n")
    tail = logtail.FileTail(logs / "listen.log")
    tail.start()
    write(logs / "listen.log", "fresh\n", mode="w")
    assert tail.poll() == ["fresh"]


def test_a_rotated_file_is_finished_then_the_new_one_followed(logs):
    path = logs / "listen.log"
    write(path, "before\n")
    tail = logtail.FileTail(path)
    tail.start()
    os.rename(path, logs / "listen.log.1")
    write(logs / "listen.log.1", "last words\n")  # the service still had the old file open
    write(path, "new file\n")
    assert tail.poll() == ["last words"]
    assert tail.poll() == ["new file"]


def test_a_log_that_appears_later_is_read_from_its_start(logs):
    tail = logtail.FileTail(logs / "listen.log")
    assert tail.start() == []
    write(logs / "listen.log", "hello\n")
    assert tail.poll() == ["hello"]


def test_a_long_line_is_cut(logs, monkeypatch):
    monkeypatch.setattr(logtail, "MAX_LINE", 5)
    write(logs / "listen.log", "")
    tail = logtail.FileTail(logs / "listen.log")
    tail.start()
    write(logs / "listen.log", "abcdefgh\n")
    assert tail.poll() == ["abcde [line cut]"]


def test_a_progress_bar_shows_only_its_last_state(logs):
    write(logs / "listen.log", "")
    tail = logtail.FileTail(logs / "listen.log")
    tail.start()
    write(logs / "listen.log", "Loading 0%\rLoading 50%\rLoading 100%\r\n")
    assert tail.poll() == ["Loading 100%"]


def test_a_byte_that_is_not_utf8_is_replaced_not_fatal(logs):
    (logs / "listen.log").write_bytes(b"")
    tail = logtail.FileTail(logs / "listen.log")
    tail.start()
    with open(logs / "listen.log", "ab") as f:
        f.write(b"caf\xe9 ok\n")
    assert tail.poll() == ["caf\ufffd ok"]


def test_services_with_no_log_file_are_named(logs):
    write(logs / "listen.log", "here\n")
    assert logtail.Tailer(logtail.pick(["listen", "see-actions"])).missing() == ["see-actions"]


def test_every_declared_service_and_nothing_else_can_be_picked(logs):
    every = logtail.pick()
    assert "listen" in every and every["listen"] == logs / "listen.log"
    with pytest.raises(ValueError, match="unknown service: ../../etc/passwd"):
        logtail.pick(["listen", "../../etc/passwd"])


@pytest.fixture
def base(logs, serve, monkeypatch):
    monkeypatch.setattr(server, "DASH_PASSWORD", "")
    monkeypatch.setattr(server, "LOG_POLL_S", 0.02)
    return serve(server.H)


def events(response, count):
    """The first count SSE events of an open response, as (name, data)."""
    out, name = [], None
    while len(out) < count:
        line = response.fp.readline().decode().rstrip("\n")
        if line.startswith("event: "):
            name = line[7:]
        elif line.startswith("data: "):
            out.append((name, json.loads(line[6:])))
    return out


def test_the_stream_sends_the_backlog_then_new_lines(base, logs):
    write(logs / "listen.log", "earlier\n")
    c = http.client.HTTPConnection(base.removeprefix("http://"), timeout=5)
    c.request("GET", "/logs/stream?services=listen")
    r = c.getresponse()
    assert (r.status, r.getheader("Content-Type")) == (200, "text/event-stream")
    (hello, first), (_, backlog) = events(r, 2)
    assert (hello, first["services"], first["missing"]) == ("hello", ["listen"], [])
    assert backlog == [{"s": "listen", "l": "earlier", "t": None}]
    write(logs / "listen.log", "just now\n")
    ((_, fresh),) = events(r, 1)
    assert [(e["s"], e["l"]) for e in fresh] == [("listen", "just now")]
    assert isinstance(fresh[0]["t"], float)
    c.close()


def test_the_stream_ends_soon_after_its_client_hangs_up(base, logs, monkeypatch):
    watch = slow.Watch()  # the requests in flight; the stream is one until its handler returns
    monkeypatch.setattr(slow, "WATCH", watch)
    write(logs / "listen.log", "earlier\n")
    c = http.client.HTTPConnection(base.removeprefix("http://"), timeout=5)
    c.request("GET", "/logs/stream?services=listen")
    events(c.getresponse(), 2)
    assert [r.route for r in watch.inflight.values()] == ["GET /logs/stream"]
    c.close()
    deadline = time.monotonic() + 2  # well short of the 15 s keep-alive, the only write a quiet log makes
    while watch.inflight and time.monotonic() < deadline:
        time.sleep(0.01)
    assert watch.inflight == {}


def test_the_stream_refuses_an_unknown_service(base):
    c = http.client.HTTPConnection(base.removeprefix("http://"), timeout=5)
    c.request("GET", "/logs/stream?services=listen,nope")
    r = c.getresponse()
    assert (r.status, json.loads(r.read())) == (400, {"error": "unknown service: nope"})
    c.close()


def test_a_service_for_another_platform_has_no_log_to_miss(monkeypatch):
    monkeypatch.setattr(logtail.services, "declared", lambda: {"voice": {}, "vlm": {"platforms": ["darwin-arm64"]}})
    monkeypatch.setattr(logtail.cfg, "platform_key", lambda: "linux-x86_64")
    assert list(logtail.known()) == ["voice"]


def test_backlog_just_after_a_trim_comes_from_the_older_copy_and_new_lines_follow(logs):
    write(logs / "listen.log", "a1\na2\n")
    assert liblogs.rotate(logs / "listen.log", cap=4) > 0
    tail = logtail.Tailer(logtail.pick(["listen"]))
    assert tail.start() == [("listen", "a2")]
    write(logs / "listen.log", "a3\n")
    assert tail.poll() == [("listen", "a3")]
