"""The dashboard's media replies (H._media): a file, or the byte range of it the browser asks for, read from disk
as it's sent rather than loaded whole, since a browser keeps a range open for as long as it plays. The routes that
send files this way (/people/reference-media, room audio) share it; here a small handler serves one file through it.
v1's /video/file, which these tested first, is gone."""

import http.client
import inspect
import pathlib

import pytest

from liberational.dashboard import server

SEG = "cam0/2026-01-01/000000.mp4"
BODY = bytes(range(256)) * 12_000  # about 3 MB, past any one read the server makes


class Media(server.H):
    """The dashboard's handler, with GET /media sending one file through _media."""

    file: pathlib.Path

    def do_GET(self) -> None:
        if self.path.startswith("/media"):
            return self._media(self.file, "video/mp4", {"Cache-Control": "no-store"})
        return super().do_GET()

    def do_HEAD(self) -> None:
        if self.path.startswith("/media"):
            return self._media(self.file, "video/mp4", {"Cache-Control": "no-store"})
        return super().do_HEAD()


@pytest.fixture
def base(serve, tmp_path, monkeypatch):
    (tmp_path / "video" / "cam0" / "2026-01-01").mkdir(parents=True)
    (tmp_path / "video" / SEG).write_bytes(BODY)
    monkeypatch.setattr(Media, "file", tmp_path / "video" / SEG, raising=False)
    monkeypatch.setattr(server, "STATE", tmp_path)
    monkeypatch.setattr(server, "DASH_PASSWORD", "")

    def whole(self):
        raise AssertionError(f"read whole: {self}")

    monkeypatch.setattr(pathlib.Path, "read_bytes", whole)  # a segment is never loaded into memory
    return serve(Media)


def get(base: str, rng: str | None = None, method: str = "GET") -> tuple[int, dict[str, str], bytes]:
    c = http.client.HTTPConnection(base.removeprefix("http://"), timeout=5)
    c.request(method, "/media", headers={"Range": rng} if rng else {})
    r = c.getresponse()
    out = r.status, {k.lower(): v for k, v in r.getheaders()}, r.read()
    c.close()
    return out


@pytest.mark.parametrize(
    "rng, a, e",
    [
        pytest.param("bytes=0-1", 0, 1, id="first-bytes"),
        pytest.param("bytes=1000000-", 1_000_000, len(BODY) - 1, id="open-ended"),
        pytest.param("bytes=2000000-2099999", 2_000_000, 2_099_999, id="middle"),
        pytest.param("bytes=-100", len(BODY) - 100, len(BODY) - 1, id="suffix"),
        pytest.param(f"bytes=10-{len(BODY) + 50}", 10, len(BODY) - 1, id="past-the-end"),
    ],
)
def test_a_range_is_its_bytes_and_nothing_else(base, rng, a, e):
    code, h, body = get(base, rng)
    assert (code, h["content-range"], h["content-length"]) == (206, f"bytes {a}-{e}/{len(BODY)}", str(e - a + 1))
    assert body == BODY[a : e + 1]
    assert h["accept-ranges"] == "bytes"


def test_no_range_is_the_whole_segment(base):
    code, h, body = get(base)
    assert (code, h["content-length"], h["content-type"]) == (200, str(len(BODY)), "video/mp4")
    assert body == BODY


def test_a_range_past_the_end_is_416_and_head_sends_no_body(base):
    code, h, body = get(base, f"bytes={len(BODY)}-")
    assert (code, h["content-range"], body) == (416, f"bytes */{len(BODY)}", b"")
    code, h, body = get(base, "bytes=0-9", method="HEAD")
    assert (code, h["content-length"], body) == (206, "10", b"")


@pytest.mark.parametrize(
    "kind, headers_after, done_after, slow",
    [
        pytest.param("video/mp4", 0.05, 60.0, False, id="video-streamed-for-a-minute"),
        pytest.param("audio/wav", 0.05, 30.0, False, id="audio-streamed"),
        pytest.param("video/mp4", 9.0, 60.0, True, id="video-slow-to-start"),
        pytest.param("application/json", 0.05, 9.0, True, id="json-slow-to-finish"),
        pytest.param("application/json", 0.05, 1.0, False, id="json-quick"),
    ],
)
def test_media_is_slow_only_when_it_was_slow_to_start(kind, headers_after, done_after, slow):
    assert (server._server_time(kind, headers_after, done_after) > server.SLOW_REQUEST_S) is slow


def around_send_media(monkeypatch, change):
    """Wrap the real `H._send_media`: each call is bound to its own signature, `change` gets its arguments by name (and
    may replace one), and the real method runs with them. A parameter added to it later passes through untouched,
    and one a test reaches for that's renamed or gone fails here, not in a fake that went stale."""
    real = server.H._send_media
    signature = inspect.signature(real)

    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        change(bound.arguments)
        return real(*bound.args, **bound.kwargs)

    monkeypatch.setattr(server.H, "_send_media", wrapped)


def test_a_segment_deleted_after_it_was_opened_is_still_sent_whole(base, tmp_path, monkeypatch):
    def gone_meanwhile(arguments):
        assert arguments["f"] is not None  # a file was opened first
        (tmp_path / "video" / SEG).unlink()  # retention, between the open and the read

    around_send_media(monkeypatch, gone_meanwhile)
    code, _, body = get(base, "bytes=100-199")
    assert (code, body) == (206, BODY[100:200])
    assert get(base)[0] == 404  # and gone for the next request


def test_an_empty_segment_is_an_empty_200_and_any_range_416(base, tmp_path):
    (tmp_path / "video" / SEG).write_bytes(b"")
    code, h, body = get(base)
    assert (code, h["content-length"], body) == (200, "0", b"")
    assert get(base, "bytes=0-")[0] == 416


def test_a_media_body_that_takes_long_after_its_headers_is_not_logged_as_slow(base, monkeypatch):
    """End to end: once a segment's headers are out, the rest is paced by the browser reading as it plays, so time
    spent sending the body doesn't count as the server being slow. Here each chunk takes 0.1 s to arrive."""
    import logging
    import time

    monkeypatch.setattr(server, "SLOW_REQUEST_S", 0.2)
    monkeypatch.setattr(server, "MEDIA_CHUNK", 1 << 20)

    class Paced:
        def __init__(self, f):
            self.f = f

        def __getattr__(self, name):
            return getattr(self.f, name)

        def read(self, n):
            time.sleep(0.1)
            return self.f.read(n)

    around_send_media(monkeypatch, lambda arguments: arguments.update(f=Paced(arguments["f"])))
    records: list[logging.LogRecord] = []
    grab = logging.Handler(logging.WARNING)
    grab.emit = records.append  # ty: ignore[invalid-assignment]
    server.LOG.addHandler(grab)
    try:
        code, _, body = get(base)  # 3 MB: four chunks, 0.4 s after the headers
        time.sleep(0.1)  # the warning, if any, is logged once the response is done
    finally:
        server.LOG.removeHandler(grab)
    assert (code, body) == (200, BODY)
    assert [r.getMessage() for r in records if "took" in r.getMessage()] == []
