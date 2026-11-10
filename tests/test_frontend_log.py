"""The frontend's own log (dashboard/frontend_log.py): what a page reports is kept only as known fields, with
no query or hash on a path, capped per minute and in size, and appended as log lines carrying JSON."""

import io
import json
import re
from email.message import Message
from pathlib import Path
from typing import Any

import pytest

from liberational.dashboard import frontend_log as fl


@pytest.fixture(autouse=True)
def _fresh_minute():
    fl._kept.clear()
    yield
    fl._kept.clear()


LINE = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d \[frontend\] (DEBUG|INFO|WARN|ERROR) (\{.*\})$")


def records(path: Path) -> list[tuple[str, dict[str, Any]]]:
    """Each line's level and event, failing on a line that isn't `<stamp> [frontend] LEVEL {json}`."""
    out = []
    for line in path.read_text().splitlines():
        found = LINE.match(line)
        assert found, line
        out.append((found[1], json.loads(found[2])))
    return out


def lines(path: Path) -> list[dict[str, Any]]:
    return [event for _, event in records(path)]


def test_an_event_keeps_known_fields_and_its_path_loses_query_and_hash() -> None:
    event = {
        "type": "request",
        "route": "/people#card=Dana",
        "url": "/people/profile?name=Dana",
        "status": 500,
        "ms": 1234.567,
        "message": "x" * 900,
        "cookie": "secret",
        "body": {"name": "Dana"},
    }
    assert fl.clean(event) == {
        "type": "request",
        "route": "/people",
        "url": "/people/profile",
        "status": 500.0,
        "ms": 1234.6,
        "message": "x" * 500,
    }


@pytest.mark.parametrize(
    "event",
    [
        pytest.param({"type": "keylogger", "route": "/"}, id="unknown-type"),
        pytest.param(["type", "error"], id="not-an-object"),
        pytest.param("error", id="a-string"),
    ],
)
def test_anything_but_a_known_event_is_dropped(event: object) -> None:
    assert fl.clean(event) is None


def test_a_path_from_elsewhere_and_a_non_number_are_left_out() -> None:
    assert fl.clean({"type": "error", "route": "https://evil.example/x", "url": "//evil.example", "ms": True}) == {
        "type": "error"
    }


def test_a_navigation_keeps_its_requests_and_chunks_as_paths_with_times() -> None:
    event = {
        "type": "navigation",
        "route": "/pets",
        "ms": 2500,
        "requests": [{"url": "/pets?x=1", "status": 200, "ms": 2100}, {"url": "https://x.example/"}, "junk"],
        "chunks": [{"url": "/static/app/assets/Known-abc.js", "ms": 300}],
    }
    kept = fl.clean(event)
    assert kept is not None
    assert kept["requests"] == [{"url": "/pets", "status": 200.0, "ms": 2100.0}]
    assert kept["chunks"] == [{"url": "/static/app/assets/Known-abc.js", "ms": 300.0}]


def test_appends_json_lines_and_caps_the_minute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fl, "MAX_PER_MINUTE", 3)
    log = tmp_path / "frontend.log"
    events = [{"type": "error", "route": "/", "message": f"e{n}"} for n in range(5)]
    assert fl.append(events, log, now=1000.0) == 3
    assert fl.append(events, log, now=1030.0) == 0, "the same minute: nothing more"
    assert fl.append(events[:1], log, now=1061.0) == 1, "a minute later, room again"
    assert [line["message"] for line in lines(log)] == ["e0", "e1", "e2", "e0"]
    assert all(line["type"] == "error" and line["route"] == "/" for line in lines(log))


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        pytest.param({"type": "error", "message": "x"}, "ERROR", id="error"),
        pytest.param({"type": "rejection", "message": "x"}, "ERROR", id="rejection"),
        pytest.param({"type": "boundary", "message": "x"}, "ERROR", id="boundary"),
        pytest.param({"type": "chunk", "url": "/static/app/a.js"}, "WARN", id="chunk"),
        pytest.param({"type": "request", "url": "/x", "status": 500}, "WARN", id="server-error"),
        pytest.param({"type": "request", "url": "/x", "status": 0}, "WARN", id="network-failure"),
        pytest.param({"type": "request", "url": "/x", "status": 200, "ms": 4000}, "INFO", id="slow"),
        pytest.param({"type": "navigation", "route": "/", "ms": 3000}, "INFO", id="navigation"),
    ],
)
def test_each_event_is_a_line_at_its_level(tmp_path: Path, event: dict[str, Any], expected: str) -> None:
    log = tmp_path / "frontend.log"
    assert fl.append([event], log, now=1000.0) == 1
    assert [level for level, _ in records(log)] == [expected]


def test_rotates_once_past_the_size_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fl, "MAX_FILE", 100)
    log = tmp_path / "frontend.log"
    log.write_text("x" * 200 + "\n")
    assert fl.append([{"type": "error", "message": "new"}], log, now=1000.0) == 1
    assert (tmp_path / "frontend.log.1").read_text().startswith("x" * 200)
    assert [line["message"] for line in lines(log)] == ["new"]


class Handler:
    """The parts of the dashboard's request handler frontend_log.handle uses."""

    def __init__(self, body: bytes, *, path: str = "/frontend/log", same_origin: bool = True) -> None:
        self.path = path
        self.headers = Message()
        self.headers["Content-Length"] = str(len(body))
        self.rfile = io.BytesIO(body)
        self.same_origin = same_origin
        self.reply: tuple[dict[str, Any], int] | None = None

    def _same_origin_json(self) -> bool:
        return self.same_origin

    def _json(self, body: dict[str, Any], code: int = 200) -> None:
        self.reply = (body, code)


def test_the_endpoint_appends_a_batch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fl, "log_path", lambda: tmp_path / "frontend.log")
    handler = Handler(json.dumps({"events": [{"type": "rejection", "route": "/review", "message": "boom"}]}).encode())
    assert fl.handle(handler)
    assert handler.reply == ({"kept": 1}, 200)
    assert lines(tmp_path / "frontend.log")[0]["message"] == "boom"


@pytest.mark.parametrize(
    ("handler", "code"),
    [
        pytest.param(Handler(b'{"events": []}', same_origin=False), 403, id="another-site"),
        pytest.param(Handler(b"x" * (fl.MAX_BODY + 1)), 413, id="too-big"),
        pytest.param(Handler(b"not json"), 400, id="not-json"),
        pytest.param(Handler(b'{"events": "nope"}'), 400, id="no-event-list"),
    ],
)
def test_the_endpoint_refuses_what_it_cannot_take(handler: Handler, code: int) -> None:
    assert fl.handle(handler)
    assert handler.reply is not None and handler.reply[1] == code


def test_other_paths_are_not_its_own() -> None:
    assert not fl.handle(Handler(b"{}", path="/frontend/other"))
