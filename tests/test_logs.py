"""One log format for every service (liberational/logs.py), and the dashboard's request errors through it."""

import ast
import datetime as dt
import errno
import http.client
import io
import logging
import pathlib
import re
import subprocess
import sys
import time
from typing import Any, cast

import pytest

from liberational import config as cfg
from liberational import logs, ollama_health, phone
from liberational.assistant import voice
from liberational.dashboard import live_status, server
from liberational.people import registry


@pytest.fixture
def captured():
    """Every line the package logs while the test runs, at every level, in the one format."""
    buf = io.StringIO()
    logs.setup(buf, logs.DEBUG)
    yield buf
    logs.setup()  # back to stderr at the configured level


def lines(buf):
    return [line for line in buf.getvalue().splitlines() if line]


def test_a_line_is_time_tag_level_message(captured):
    log = logs.get("backfill")
    log.debug("one")
    log.info("two %s", 2)
    log.warning("three")
    log.error("four")
    log.critical("five")
    words = [line.split(" ", 3)[1:] for line in lines(captured)]
    assert words == [
        ["[backfill]", "DEBUG", "one"],
        ["[backfill]", "INFO", "two 2"],
        ["[backfill]", "WARN", "three"],
        ["[backfill]", "ERROR", "four"],
        ["[backfill]", "ERROR", "five"],
    ]
    # The stamp is ISO local date and time with its offset, and reads back as now.
    for line in lines(captured):
        stamp = line.split(" ", 1)[0]
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d", stamp), line
        assert abs(dt.datetime.fromisoformat(stamp).timestamp() - time.time()) < 60


def test_a_traceback_follows_its_line(captured):
    try:
        raise ValueError("boom")
    except ValueError:
        logs.get("journal").exception("pass failed")
    first, *rest = lines(captured)
    assert first.endswith("[journal] ERROR pass failed")
    assert rest[0] == "Traceback (most recent call last):"
    assert rest[-1] == "ValueError: boom"


def test_a_modules_log_helper_writes_the_level_it_is_given(captured):
    registry.log("video scan waits for retry:", "x.mp4", level=logs.WARN)
    registry.log("segment done")
    assert [line.split(" ", 2)[2] for line in lines(captured)] == [
        "WARN video scan waits for retry: x.mp4",
        "INFO segment done",
    ]


@pytest.mark.parametrize(
    ("settings", "service", "level"),
    [
        ({"level": "info", "per_service": []}, "backfill", logging.INFO),
        ({"level": "warn", "per_service": []}, "backfill", logging.WARNING),
        ({"level": "info", "per_service": ["backfill=debug"]}, "backfill", logging.DEBUG),
        ({"level": "info", "per_service": ["backfill=debug"]}, "vision", logging.INFO),
        ({"level": "loud", "per_service": []}, "backfill", logging.INFO),  # a typo never silences a service
        ({"level": "error", "per_service": ["vision = warn "]}, "vision", logging.WARNING),
        ({"level": "warning", "per_service": []}, "vision", logging.WARNING),  # Python's spelling
        ({"level": "info", "per_service": "backfill=debug"}, "backfill", logging.DEBUG),  # one entry, not a list
    ],
)
def test_the_level_is_the_services_own_else_the_global_one(settings, service, level):
    with cfg.override({"logging": settings}):
        assert logs.level_for(service) == level


def test_lines_below_the_level_are_not_written():
    buf = io.StringIO()
    logs.setup(buf, logs.WARN)
    try:
        log = logs.get("vision")
        log.info("quiet")
        log.warning("camera lost")
    finally:
        logs.setup()
    assert [line.split(" ", 2)[2] for line in lines(buf)] == ["WARN camera lost"]


def test_other_libraries_and_the_root_keep_their_own_logging(captured, caplog):
    caplog.set_level(logging.INFO)
    logging.getLogger("uvicorn.error").warning("port in use")
    logging.getLogger("httpx").info("GET /api 200")
    assert lines(captured) == []  # not ours to format, and the root's handlers (here caplog's) still get them
    assert [r.getMessage() for r in caplog.records] == ["port in use", "GET /api 200"]


def test_a_process_with_no_logging_yet_can_still_configure_the_root(captured, monkeypatch, tmp_path):
    # The phone bridge imported the package (whose loggers set up) before its own logging.basicConfig, which then did
    # nothing because the root already had a handler, and phone_bridge.log stayed empty.
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])
    monkeypatch.setattr(root, "level", root.level)
    logs.setup(captured)
    out = tmp_path / "app.log"
    logging.basicConfig(handlers=[logging.FileHandler(out)], format="%(message)s", level=logging.INFO)
    logging.getLogger("app").info("call started")
    root.handlers[0].close()
    assert out.read_text() == "call started\n"
    assert lines(captured) == []


def test_an_adopted_library_logs_in_the_one_format(captured, caplog):
    logs.adopt("uvicorn")
    try:
        logging.getLogger("uvicorn.error").warning("port in use")
    finally:
        logging.getLogger("uvicorn").handlers.clear()
        logging.getLogger("uvicorn").propagate = True
    assert [line.split(" ", 1)[1] for line in lines(captured)] == ["[uvicorn.error] WARN port in use"]
    assert caplog.records == []


def test_the_bridge_log_is_the_kept_older_copy_then_the_current_log(tmp_path):
    (tmp_path / "phone-bridge.log.1").write_text("2026-10-20 09:00:00 [CA1] first audio 1.2s\n")
    (tmp_path / "phone-bridge.log").write_text("09:00:00 [phone-bridge] INFO [CA2] first audio 0.8s\n")
    with cfg.override({"paths": {"logs": str(tmp_path)}}):
        assert phone.bridge_log().splitlines() == [
            "2026-10-20 09:00:00 [CA1] first audio 1.2s",
            "09:00:00 [phone-bridge] INFO [CA2] first audio 0.8s",
        ]


def test_no_bridge_log_is_empty(tmp_path):
    with cfg.override({"paths": {"logs": str(tmp_path)}}):
        assert phone.bridge_log() == ""


def test_ollama_going_down_is_a_warning_and_coming_back_is_info(captured):
    now = [1000.0]
    health = ollama_health.Health(lambda: now[0])
    with cfg.override({"ollama": {"url": "http://127.0.0.1:9", "retry_s": 30, "log_every_s": 60}}):
        with pytest.raises(ollama_health.Down), ollama_health.call("m", health):
            raise ConnectionRefusedError(111, "refused")
        now[0] += 120
        with ollama_health.call("m", health):
            pass
    assert [line.split(" ", 3)[1:3] for line in lines(captured)] == [["[ollama]", "WARN"], ["[ollama]", "INFO"]]


@pytest.mark.parametrize(
    ("argv0", "name"), [("/x/.venv/bin/pack-backfill", "backfill"), ("pack-whisper-names", "whisper-names")]
)
def test_the_service_is_the_running_pack_command(monkeypatch, argv0, name):
    monkeypatch.setattr(sys, "argv", [argv0])
    assert logs.service() == name


# The dashboard: a client that leaves is one debug line; a failing route is one error line with the route.


class Gone(io.RawIOBase):
    def __init__(self, ex: OSError) -> None:
        self.ex = ex

    def write(self, _b: Any) -> int:
        raise self.ex


@pytest.mark.parametrize(
    "ex",
    [
        BrokenPipeError(errno.EPIPE, "Broken pipe"),
        ConnectionResetError(errno.ECONNRESET, "Connection reset by peer"),
        OSError(errno.EPROTOTYPE, "Protocol wrong type for socket"),  # macOS, a socket mid-teardown
    ],
    ids=["broken-pipe", "reset", "macos-eprototype"],
)
def test_a_client_leaving_during_a_media_write_is_one_debug_line(captured, ex):
    h = object.__new__(server.H)
    h.command, h.path, h.headers = "GET", "/video/file?name=private", cast(Any, {"Range": "bytes=0-3"})
    h.wfile = cast(Any, Gone(ex))
    sent: list[Any] = []
    h.send_response = lambda code, *a: sent.append(code)  # ty: ignore[invalid-assignment]
    h.send_header = lambda *a: None  # ty: ignore[invalid-assignment]
    h.end_headers = lambda: None
    h._media(b"0123456789", "video/mp4")
    assert sent == [206]
    assert lines(captured) == [
        lines(captured)[0].split(" ", 1)[0]
        + f" [dashboard] DEBUG GET /video/file: the client left during the media write ({type(ex).__name__})"
    ]


def test_another_media_write_failure_is_not_swallowed(captured):
    h = object.__new__(server.H)
    h.command, h.path, h.headers = "GET", "/video/file", cast(Any, {})
    h.wfile = cast(Any, Gone(OSError(errno.ENOSPC, "No space left on device")))
    h.send_response = lambda *a: None  # ty: ignore[invalid-assignment]
    h.send_header = lambda *a: None  # ty: ignore[invalid-assignment]
    h.end_headers = lambda: None
    with pytest.raises(OSError, match="No space"):
        h._media(b"0123", "video/mp4")
    assert lines(captured) == []


def test_a_failing_route_is_one_error_line_naming_it(captured, serve_dashboard, monkeypatch, capsys):

    def broken(*a: Any) -> dict:
        raise ValueError("boom")

    monkeypatch.setattr(live_status, "camera_status", broken)
    url = serve_dashboard().removeprefix("http://")
    c = http.client.HTTPConnection(url, timeout=5)
    c.request("GET", "/vision?who=Dana")  # a route whose own work raises
    with pytest.raises((http.client.RemoteDisconnected, ConnectionResetError)):
        c.getresponse()
    c.close()
    first, *rest = lines(captured)
    assert first.endswith("[dashboard] ERROR GET /vision failed")  # no query: it can hold names
    assert rest[-1] == "ValueError: boom"
    assert "Exception occurred during processing of request" not in capsys.readouterr().err


def test_another_librarys_warning_in_an_unconfigured_process_is_in_the_one_format(captured, monkeypatch):
    # A library warning with no handler anywhere used to print bare ("Could not infer sample rate ...").
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])
    logs.setup(captured)
    logging.getLogger("faster_qwen3_tts.model").warning("Could not infer sample rate; defaulting to 24000 Hz.")
    logging.getLogger("httpx").info("GET /api 200")  # below WARNING: still not written
    assert [line.split(" ", 1)[1] for line in lines(captured)] == [
        "[faster_qwen3_tts.model] WARN Could not infer sample rate; defaulting to 24000 Hz."
    ]


def test_a_librarys_warnings_warn_is_in_the_one_format():
    # transformers' "`torch_dtype` is deprecated" printed bare, with no stamp or level. In its own process, since
    # pytest takes warnings over for itself.
    code = "from liberational import logs; logs.get('x'); import warnings; warnings.warn('`torch_dtype` is deprecated')"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=True)
    first = done.stderr.splitlines()[0]
    assert re.match(r"\S+ \[py\.warnings\] WARN .*UserWarning: `torch_dtype` is deprecated", first), done.stderr


def test_a_periodic_report_is_info_first_and_then_once_a_period():
    now = [0.0]
    clock = logs.Periodic(600.0, clock=lambda: now[0])
    seen = []
    for t in (0.0, 60.0, 599.0, 600.0, 660.0, 1199.0, 1200.0):
        now[0] = t
        seen.append(clock.level())
    info, debug = logs.INFO, logs.DEBUG
    assert seen == [info, debug, debug, info, debug, debug, info]


@pytest.mark.parametrize(
    ("settings", "service", "word"),
    [
        ({"level": "info", "per_service": ["continuum-gateway=debug"]}, "continuum-gateway", "debug"),
        ({"level": "info", "per_service": ["continuum-gateway=debug"]}, "continuum-broker", "info"),
        ({"level": "warn", "per_service": []}, "continuum-broker", "warn"),
        ({"level": "error", "per_service": []}, "continuum-broker", "error"),
        ({"level": "debug", "per_service": ["continuum-broker=info"]}, "", "debug"),  # no name: the global level
    ],
)
def test_a_rust_daemons_filter_follows_the_same_settings(settings, service, word):
    with cfg.override({"logging": settings}):
        assert logs.rust_filter(service) == word


SRC = pathlib.Path(logs.__file__).parent


def _has_cli(tree: ast.Module) -> bool:
    """A module with a command of its own (a main(), argparse, or a __main__ block) may print its output."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return True
        if isinstance(node, (ast.Import, ast.ImportFrom)) and any(a.name == "argparse" for a in node.names):
            return True
        if isinstance(node, ast.Constant) and node.value == "__main__":
            return True
    return False


def test_code_with_no_command_of_its_own_logs_instead_of_printing():
    # Such code only runs inside a service (or another command), where a print skips the level and the tag. A module
    # with a command of its own may print that command's output, so this can't see a print on a service's own path;
    # the service lines above have their own tests.
    printing = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text())
        if _has_cli(tree):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                printing.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert printing == []


def test_a_reply_that_cant_be_recorded_is_one_error_line(captured, monkeypatch):
    # v1 logged the failure with a status line, which went to the same unwritable file, and recursed. On the speech
    # log a failed append is said once, by the error's type: its message could quote the reply.
    class Refusing:
        def append(self, stream, line):
            raise ConnectionError("the broker is gone")

    monkeypatch.setattr(voice, "CONTINUUM", Refusing())
    voice._append({"text": "hello"})
    assert [line.split(" ", 2)[1:] for line in lines(captured)] == [
        ["[voice]", "ERROR could not record the reply: ConnectionError"]
    ]


def test_a_slow_request_is_a_warning_with_its_route_and_duration(captured, serve_dashboard, monkeypatch):
    monkeypatch.setattr(server, "SLOW_REQUEST_S", 0.0)
    url = serve_dashboard().removeprefix("http://")
    c = http.client.HTTPConnection(url, timeout=5)
    c.request("GET", "/no/such/route?who=Dana")
    c.getresponse().read()
    c.close()
    # The warning is written after the response has gone out, so it can land a moment after the client has it
    deadline = time.monotonic() + 5
    while not (slow := [line for line in lines(captured) if " WARN " in line]) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(slow) == 1
    assert re.search(r"\[dashboard\] WARN GET /no/such/route took \d+\.\d s$", slow[0])  # no query: it can hold names


def test_a_fast_request_is_not_logged(captured, serve_dashboard, monkeypatch):
    url = serve_dashboard().removeprefix("http://")
    c = http.client.HTTPConnection(url, timeout=5)
    c.request("GET", "/no/such/route")
    c.getresponse().read()
    c.close()
    assert [line for line in lines(captured) if " WARN " in line] == []


@pytest.mark.parametrize(
    "ex",
    [BrokenPipeError(errno.EPIPE, "Broken pipe"), ConnectionResetError(errno.ECONNRESET, "Connection reset by peer")],
    ids=["broken-pipe", "reset"],
)
def test_a_client_leaving_before_a_json_reply_is_one_debug_line_and_never_reaches_the_route(captured, ex):
    """A route that writes its reply inside its own try (GET /live/happening, /review/summary) logged the
    client leaving as its own ERROR, then wrote a 500 to the dead socket."""
    h = object.__new__(server.H)
    h.command, h.path, h.headers, h.request_version = "GET", "/live/happening?since=1", cast(Any, {}), "HTTP/1.1"
    h.wfile = cast(Any, Gone(ex))
    h.requestline, h.client_address = "GET /live/happening HTTP/1.1", ("127.0.0.1", 0)
    vars(h)["_headers_buffer"] = []  # http.server's own buffer, made in send_response's first header
    try:
        h._json({"items": []})  # raises nothing: the route's except never sees it
    except Exception as e:
        pytest.fail(f"_json raised {e!r}")
    assert h.close_connection
    out = lines(captured)
    assert len(out) == 1 and out[0].endswith(
        f"[dashboard] DEBUG GET /live/happening: the client left before the reply was written ({type(ex).__name__})"
    )


def test_another_reply_write_failure_is_not_swallowed(captured):
    h = object.__new__(server.H)
    h.command, h.path, h.headers, h.request_version = "GET", "/live/happening", cast(Any, {}), "HTTP/1.1"
    h.wfile = cast(Any, Gone(OSError(errno.ENOSPC, "No space left on device")))
    h.requestline, h.client_address = "GET /live/happening HTTP/1.1", ("127.0.0.1", 0)
    vars(h)["_headers_buffer"] = []  # http.server's own buffer, made in send_response's first header
    with pytest.raises(OSError, match="No space"):
        h._json({"items": []})


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        pytest.param(TimeoutError(), "TimeoutError", id="bare-timeout-names-its-type"),
        pytest.param(TimeoutError("no reply from the broker"), "no reply from the broker", id="text-wins"),
        pytest.param(ConnectionRefusedError(111, "refused"), "[Errno 111] refused", id="os-error"),
    ],
)
def test_a_logged_reason_is_never_empty(error, expected):
    # A bare TimeoutError logged as "voice search unavailable ()".
    assert logs.reason(error) == expected


def numbered(count: int) -> bytes:
    return b"".join(b"line %04d\n" % i for i in range(count))  # 10 bytes a line


def test_a_log_past_its_cap_keeps_its_newest_whole_lines_in_the_older_file_and_empties(tmp_path):
    log = tmp_path / "listen.log"
    log.write_bytes(numbered(100))  # 1,000 bytes
    assert logs.rotate(log, cap=95) == 1000 - 90
    assert log.read_bytes() == b""
    assert (tmp_path / "listen.log.1").read_bytes() == numbered(100)[-90:]  # 9 whole lines: line 0091 to 0099


def test_a_log_under_its_cap_is_left_alone(tmp_path):
    log = tmp_path / "listen.log"
    log.write_bytes(numbered(5))
    assert logs.rotate(log, cap=50) == 0
    assert log.read_bytes() == numbered(5)
    assert not (tmp_path / "listen.log.1").exists()


def test_an_append_mode_writer_keeps_writing_from_the_start_of_the_emptied_log(tmp_path):
    log = tmp_path / "listen.log"
    with log.open("ab", buffering=0) as writer:
        writer.write(numbered(50))
        logs.rotate(log, cap=100)
        writer.write(b"after\n")
    assert log.read_bytes() == b"after\n"
    assert (tmp_path / "listen.log.1").read_bytes() == numbered(50)[-100:]


def test_writing_past_the_cap_again_and_again_keeps_every_file_within_it_and_the_newest_lines(tmp_path):
    log = tmp_path / "ollama.log"
    with log.open("ab", buffering=0) as writer:
        for _ in range(5):
            writer.write(numbered(30))
            logs.trim(tmp_path, cap=100)
        writer.write(b"newest\n")
    assert log.read_bytes() == b"newest\n"
    assert (tmp_path / "ollama.log.1").read_bytes() == numbered(30)[-100:]


def test_trim_rotates_only_the_logs_past_the_cap(tmp_path):
    (tmp_path / "big.log").write_bytes(numbered(30))
    (tmp_path / "small.log").write_bytes(numbered(3))
    (tmp_path / "notes.txt").write_bytes(numbered(30))
    assert logs.trim(tmp_path, cap=100) == {"big.log": 200}
    assert (tmp_path / "small.log").read_bytes() == numbered(3)
    assert (tmp_path / "notes.txt").read_bytes() == numbered(30)


@pytest.mark.parametrize(
    ("current", "limit", "expected"),
    [
        pytest.param(b"", 40, numbered(4), id="just-rotated"),
        pytest.param(b"new\n", 24, numbered(4)[-20:] + b"new\n", id="reaches-into-older"),
        pytest.param(numbered(5), 30, numbered(5)[-30:], id="current-is-enough"),
    ],
)
def test_a_tail_reads_back_into_the_older_file_only_as_far_as_it_must(tmp_path, current, limit, expected):
    (tmp_path / "listen.log.1").write_bytes(numbered(4))
    (tmp_path / "listen.log").write_bytes(current)
    assert b"".join(p.data for p in logs.tail(tmp_path / "listen.log", limit)) == expected


def test_a_missing_log_has_an_empty_tail(tmp_path):
    assert logs.tail(tmp_path / "listen.log", 100) == []
