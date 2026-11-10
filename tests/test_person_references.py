import io
import socket
import subprocess
from unittest.mock import create_autospec

import pytest
from PIL import Image

from liberational.people import references as ref


@pytest.fixture
def photo():
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), "blue").save(buf, "PNG")
    return buf.getvalue()


def test_preview_attach_reload_remove(tmp_path, photo):
    item = ref.stage(tmp_path, photo, "sample.mp3")
    assert item["kind"] == "image"
    assert item["person"] is None
    assert ref.listing(tmp_path, "Dana") == []
    ref.attach(tmp_path, item["id"], "Dana")
    assert ref.listing(tmp_path, "Dana")[0]["enrollment"] == "reference only"
    with pytest.raises(ValueError, match="different person"):
        ref.attach(tmp_path, item["id"], "Alice")
    ref.remove(tmp_path, item["id"])
    assert ref.listing(tmp_path, "Dana") == []
    assert not ref.entry(tmp_path, item["id"]).exists()


@pytest.mark.parametrize(
    "raw", [b"", b"<html>video</html>", b"#EXTM3U\nhttp://127.0.0.1/secret", b"ID3corrupt", b"x" * (ref.MAX_BYTES + 1)]
)
def test_bad_content_leaves_no_attachment(tmp_path, raw):
    with pytest.raises(ValueError):
        ref.stage(tmp_path, raw, "photo.jpg")
    assert list(ref.root(tmp_path).glob("*/meta.json")) == []


@pytest.mark.parametrize(
    "value", ["file:///etc/passwd", "https://user:secret@example.com/a", "http://example.com:8080/a", "/tmp/audio.wav"]
)
def test_url_syntax_rejected_before_dns(value, monkeypatch):
    lookup = create_autospec(socket.getaddrinfo)
    monkeypatch.setattr(socket, "getaddrinfo", lookup)
    with pytest.raises(ValueError):
        ref.public_url(value)
    lookup.assert_not_called()


@pytest.mark.parametrize(
    "ip", ["127.0.0.1", "10.0.0.1", "169.254.169.254", "::1", "fe80::1", "192.168.0.1", "0.0.0.0", "::ffff:127.0.0.1"]
)
def test_dns_private_and_mixed_answers_rejected(ip, monkeypatch):
    lookup = create_autospec(
        socket.getaddrinfo, return_value=[(2, 1, 6, "", ("8.8.8.8", 443)), (2, 1, 6, "", (ip, 443))]
    )
    monkeypatch.setattr(socket, "getaddrinfo", lookup)
    with pytest.raises(ValueError, match="Private"):
        ref.public_url("https://example.com/media")


def test_redirect_rejected_without_following(monkeypatch):
    lookup = create_autospec(socket.getaddrinfo, return_value=[(2, 1, 6, "", ("8.8.8.8", 80))])
    monkeypatch.setattr(socket, "getaddrinfo", lookup)
    connection = create_autospec(ref.http.client.HTTPConnection("example.com"), instance=True)
    response = create_autospec(ref.http.client.HTTPResponse, instance=True)
    response.status = 302
    connection.getresponse.return_value = response
    factory = create_autospec(ref.http.client.HTTPConnection, return_value=connection)
    monkeypatch.setattr(ref.http.client, "HTTPConnection", factory)
    with pytest.raises(ValueError, match="Redirects"):
        ref.fetch_media("http://example.com/media")
    factory.assert_called_once_with("8.8.8.8", 80, timeout=10)
    connection.request.assert_called_once_with(
        "GET", "/media", headers={"Host": "example.com", "Accept": "image/*,audio/*,video/*", "Connection": "close"}
    )


@pytest.mark.parametrize("extension", ["wav", "mp3", "mp4"])
def test_real_synthetic_media_decode(tmp_path, extension):
    source = tmp_path / ("synthetic." + extension)
    video = ["-f", "lavfi", "-i", "color=c=blue:s=32x32:d=0.3", "-c:v", "libx264"]
    audio = ["-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", "0.3"]
    subprocess.run(
        ["ffmpeg", "-v", "error", *(video if extension == "mp4" else audio), str(source)], check=True, timeout=15
    )
    item = ref.stage(tmp_path, source.read_bytes(), "synthetic")
    assert item["kind"] == ("video" if extension == "mp4" else "audio")
    assert 0 < item["duration"] < 1
    assert item["enrollment"] == "reference only"
    assert item["person"] is None


def test_preview_expires(tmp_path, photo, monkeypatch):
    item = ref.stage(tmp_path, photo, "local")
    monkeypatch.setattr(ref.time, "time", lambda: item["created"] + 3601)
    with pytest.raises(ValueError, match="expired"):
        ref.attach(tmp_path, item["id"], "Dana")


@pytest.mark.parametrize("token", ["../secret", "", "a" * 31, "/" * 32])
def test_no_server_paths(tmp_path, token):
    with pytest.raises(ValueError, match="Invalid"):
        ref.entry(tmp_path, token)


def test_rename_keeps_attachment_without_reenrollment(tmp_path, photo):
    item = ref.stage(tmp_path, photo, "local")
    ref.attach(tmp_path, item["id"], "Dana")
    ref.rename(tmp_path, "Dana", "Alice")
    assert ref.listing(tmp_path, "Dana") == []
    assert ref.listing(tmp_path, "Alice")[0]["id"] == item["id"]
    assert ref.listing(tmp_path, "Alice")[0]["enrollment"] == "reference only"


def test_overlong_audio_rejected(tmp_path):
    import wave

    source = io.BytesIO()
    with wave.open(source, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\0\0" * 8000 * 121)
    with pytest.raises(ValueError, match="120 seconds"):
        ref.stage(tmp_path, source.getvalue(), "long.wav")


def test_expired_cleanup_never_removes_attached(tmp_path, photo, monkeypatch):
    preview = ref.stage(tmp_path, photo, "preview")
    attached = ref.stage(tmp_path, photo, "attached")
    ref.attach(tmp_path, attached["id"], "Dana")
    monkeypatch.setattr(ref.time, "time", lambda: attached["created"] + 3601)
    ref.stage(tmp_path, photo, "new preview")
    assert not ref.entry(tmp_path, preview["id"]).exists()
    assert ref.listing(tmp_path, "Dana")[0]["id"] == attached["id"]


def test_download_deadline_interrupts_incomplete_headers(monkeypatch):
    left, right = socket.socketpair()
    left.settimeout(0.3)
    right.sendall(b"HTTP/1.1 200 OK\r\nX-Unfinished: ")

    class Connection:
        sock = left

        def connect(self):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            response = ref.http.client.HTTPResponse(left)
            response.begin()
            return response

        def close(self):
            left.close()

    monkeypatch.setattr(ref, "DOWNLOAD_SECONDS", 0.02, raising=False)
    monkeypatch.setattr(ref, "public_url", lambda value: (ref.urllib.parse.urlsplit(value), "8.8.8.8"))
    monkeypatch.setattr(ref.http.client, "HTTPConnection", lambda *args, **kwargs: Connection())
    try:
        with pytest.raises(ValueError, match="timed out"):
            ref.fetch_media("http://example.com/media")
    finally:
        left.close()
        right.close()
