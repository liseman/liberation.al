"""Local reference attachments. A photo with exactly one face, once attached to a person, also teaches their face
(one owner face row, `faces.enroll_reference`, recorded here as "face" and taken back on Remove); group photos, audio,
and video never enroll .

Uploads are decoded locally. URL imports are direct media only, with pinned public
DNS addresses and no redirects. Unconfirmed previews expire after one hour.
"""

import concurrent.futures
import hashlib
import http.client
import io
import ipaddress
import json
import math
import pathlib
import shutil
import socket
import ssl
import subprocess
import threading
import time
import urllib.parse
import uuid
from typing import Any

from PIL import Image

MAX_BYTES = 20 * 1024 * 1024
MAX_SECONDS = 120
DOWNLOAD_SECONDS = 30.0
_DNS = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="reference-dns")
_DNS_SLOTS = threading.BoundedSemaphore(2)
_WRITES = threading.RLock()
_DECODES = threading.BoundedSemaphore(2)
_FETCHES = threading.BoundedSemaphore(2)


def public_url(value: str) -> tuple[urllib.parse.SplitResult, str]:
    """Validate every DNS answer and return one pinned public address."""
    url = urllib.parse.urlsplit(value)
    if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password:
        raise ValueError("Use a direct HTTP(S) media URL without credentials.")
    if url.port not in (None, 80, 443):
        raise ValueError("Only standard HTTP(S) ports are supported.")
    if not _DNS_SLOTS.acquire(blocking=False):
        raise ValueError("URL resolver busy. Retry shortly.")
    future = _DNS.submit(
        socket.getaddrinfo, url.hostname, url.port or (443 if url.scheme == "https" else 80), type=socket.SOCK_STREAM
    )
    future.add_done_callback(lambda _: _DNS_SLOTS.release())
    try:
        addresses = future.result(timeout=5)
    except TimeoutError as exc:
        raise ValueError("URL name resolution timed out. Retry later.") from exc
    ips = [str(answer[4][0]) for answer in addresses]
    if not ips or any(not ipaddress.ip_address(ip).is_global for ip in ips):
        raise ValueError("Private, local, and reserved network addresses are not allowed.")
    return url, ips[0]


def fetch_media(value: str) -> bytes:
    if not _FETCHES.acquire(blocking=False):
        raise ValueError("Media downloader busy. Retry shortly.")
    try:
        return _fetch_media(value)
    finally:
        _FETCHES.release()


def _fetch_media(value: str) -> bytes:
    """Pin validated DNS to the connection, including TLS hostname verification."""
    url, ip = public_url(value)
    port = url.port or (443 if url.scheme == "https" else 80)
    conn = http.client.HTTPConnection(ip, port, timeout=10)
    deadline = time.monotonic() + DOWNLOAD_SECONDS
    expired = threading.Event()
    active_socket: socket.socket | None = None

    def interrupt() -> None:
        expired.set()
        try:
            target = active_socket if active_socket is not None else conn.sock
            if target is not None:
                target.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    timer = threading.Timer(DOWNLOAD_SECONDS, interrupt)
    timer.daemon = True
    timer.start()
    try:
        conn.connect()
        if url.scheme == "https":
            conn.sock = ssl.create_default_context().wrap_socket(conn.sock, server_hostname=url.hostname)
        active_socket = conn.sock
        conn.request(
            "GET",
            urllib.parse.urlunsplit(("", "", url.path or "/", url.query, "")),
            headers={"Host": url.netloc, "Accept": "image/*,audio/*,video/*", "Connection": "close"},
        )
        response = conn.getresponse()
        if expired.is_set():
            raise ValueError("Media download timed out.")
        if response.status != 200:
            raise ValueError(
                "Link did not return direct media. Redirects and web pages are unsupported; download the media yourself and upload it."
            )
        length = response.getheader("Content-Length")
        if length and int(length) > MAX_BYTES:
            raise ValueError("Reference exceeds 20 MB.")
        data = bytearray()
        while len(data) <= MAX_BYTES:
            if time.monotonic() > deadline:
                raise ValueError("Media download timed out.")
            chunk = response.read1(min(65536, MAX_BYTES + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        if len(data) > MAX_BYTES:
            raise ValueError("Reference exceeds 20 MB.")
        if expired.is_set():
            raise ValueError("Media download timed out.")
        return bytes(data)
    except (OSError, http.client.HTTPException) as exc:
        if expired.is_set():
            raise ValueError("Media download timed out.") from exc
        raise ValueError("Media download failed. Check the direct URL and retry.") from exc
    finally:
        timer.cancel()
        conn.close()


def inspect_media(raw: bytes, folder: pathlib.Path) -> dict[str, Any]:
    """Decode actual bytes, ignoring filenames and advertised MIME types."""
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError("Choose a nonempty reference up to 20 MB.")
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.width * image.height > 20_000_000:
                raise ValueError("Image exceeds 20 megapixels.")
            image.load()
            image.convert("RGB").save(folder / "media.jpg", quality=90)
        return {"file": "media.jpg", "kind": "image", "mime": "image/jpeg"}
    except OSError, Image.DecompressionBombError:
        pass
    ffprobe, ffmpeg = shutil.which("ffprobe"), shutil.which("ffmpeg")
    if not ffprobe or not ffmpeg:
        raise ValueError("This is not a supported image. Audio/video requires local ffmpeg and ffprobe.")
    # Force a binary media demuxer; never allow playlists to open local files or URLs.
    if raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        demuxer = "wav"
    elif raw[:3] == b"ID3" or (len(raw) > 1 and raw[0] == 255 and raw[1] & 224 == 224):
        demuxer = "mp3"
    elif raw[4:8] == b"ftyp":
        demuxer = "mov"
    elif raw[:4] == b"\x1a\x45\xdf\xa3":
        demuxer = "matroska"
    elif raw[:4] == b"fLaC":
        demuxer = "flac"
    elif raw[:4] == b"OggS":
        demuxer = "ogg"
    else:
        raise ValueError(
            "Unsupported content. Use JPEG/PNG, MP3/WAV, MP4, WebM, FLAC, or Ogg; web pages and playlists cannot be imported."
        )
    source = folder / "input"
    source.write_bytes(raw)
    try:
        probe = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-protocol_whitelist",
                "file",
                "-show_format",
                "-show_streams",
                "-of",
                "json",
                "-f",
                demuxer,
                str(source),
            ],
            capture_output=True,
            timeout=15,
            check=True,
        )
        info = json.loads(probe.stdout)
        duration = float(info.get("format", {}).get("duration", 0))
        if not math.isfinite(duration) or not 0 < duration <= MAX_SECONDS:
            raise ValueError("Audio/video must have a known duration of at most 120 seconds.")
        streams = info.get("streams", [])
        video = any(
            s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic") for s in streams
        )
        if not video and not any(s.get("codec_type") == "audio" for s in streams):
            raise ValueError("No supported audio/video stream found.")
        filename = "media.mp4" if video else "media.wav"
        options = (
            [
                "-map",
                "0:v:0",
                "-map",
                "0:a:0?",
                "-vf",
                "scale=640:360:force_original_aspect_ratio=decrease",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
            ]
            if video
            else ["-vn", "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le"]
        )
        subprocess.run(
            [
                ffmpeg,
                "-threads",
                "2",
                "-v",
                "error",
                "-nostdin",
                "-protocol_whitelist",
                "file",
                "-f",
                demuxer,
                "-i",
                str(source),
                "-t",
                str(MAX_SECONDS),
                *options,
                "-threads",
                "2",
                "-fs",
                str(MAX_BYTES),
                str(folder / filename),
            ],
            capture_output=True,
            timeout=45,
            check=True,
        )
        if (folder / filename).stat().st_size >= MAX_BYTES:
            raise ValueError("Decoded media exceeds 20 MB. Use a shorter clip.")
        return {
            "file": filename,
            "kind": "video" if video else "audio",
            "mime": "video/mp4" if video else "audio/wav",
            "duration": duration,
        }
    except (subprocess.SubprocessError, ValueError, KeyError) as exc:
        raise ValueError(
            "Unsupported or invalid media. Use a decodable photo, MP3/WAV audio, or short video (at most 120 seconds)."
        ) from exc
    finally:
        source.unlink(missing_ok=True)


def root(state: pathlib.Path) -> pathlib.Path:
    return state / "person_references"


def entry(state: pathlib.Path, token: str) -> pathlib.Path:
    if len(token) != 32 or any(c not in "0123456789abcdef" for c in token):
        raise ValueError("Invalid reference ID.")
    return root(state) / token


def metadata(state: pathlib.Path, token: str) -> dict[str, Any]:
    folder = entry(state, token)
    try:
        data = json.loads((folder / "meta.json").read_text())
    except (OSError, ValueError) as exc:
        raise ValueError("Reference not found. Select it again to retry.") from exc
    if not data.get("person") and time.time() - data["created"] > 3600:
        raise ValueError("Preview expired. Upload it again.")
    return data


def stage(state: pathlib.Path, raw: bytes, source: str) -> dict[str, Any]:
    if not _DECODES.acquire(blocking=False):
        raise ValueError("Media decoder busy. Retry shortly.")
    try:
        return _stage(state, raw, source)
    finally:
        _DECODES.release()


def _stage(state: pathlib.Path, raw: bytes, source: str) -> dict[str, Any]:
    # Drop abandoned previews on the next upload, never attached references.
    for old in root(state).glob("*/meta.json"):
        try:
            data = json.loads(old.read_text())
            if data.get("person") is None and time.time() - data["created"] > 3600:
                remove(state, old.parent.name)
        except OSError, ValueError, KeyError:
            continue
    token = uuid.uuid4().hex
    folder = entry(state, token)
    folder.mkdir(parents=True)
    try:
        data = {
            **inspect_media(raw, folder),
            "id": token,
            "source": source[:1000],
            "created": time.time(),
            "person": None,
            "enrollment": "reference only",
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        (folder / "meta.json").write_text(json.dumps(data))
        return data
    except Exception:
        shutil.rmtree(folder)
        raise


def attach(state: pathlib.Path, token: str, person: str) -> dict[str, Any]:
    with _WRITES:
        return _attach(state, token, person)


def _attach(state: pathlib.Path, token: str, person: str) -> dict[str, Any]:
    data = metadata(state, token)
    if data["person"] is not None and data["person"] != person:
        raise ValueError("Reference is already attached to a different person.")
    data["person"] = person
    folder = entry(state, token)
    temporary = folder / "meta.tmp"
    temporary.write_text(json.dumps(data))
    temporary.replace(folder / "meta.json")
    return data


def note_face(state: pathlib.Path, token: str, face: dict[str, Any]) -> dict[str, Any]:
    """Record what attaching the reference did to its person's face rows (`faces.enroll_reference`'s answer)."""
    with _WRITES:
        data = metadata(state, token)
        data["face"] = face
        data["enrollment"] = "face" if face.get("result") == "added" else "reference only"
        folder = entry(state, token)
        temporary = folder / "meta.tmp"
        temporary.write_text(json.dumps(data))
        temporary.replace(folder / "meta.json")
        return data


def listing(state: pathlib.Path, person: str) -> list[dict[str, Any]]:
    rows = []
    for path in root(state).glob("*/meta.json"):
        try:
            data = metadata(state, path.parent.name)
            if data.get("person") == person:
                rows.append(data)
        except ValueError:
            continue
    return rows


def remove(state: pathlib.Path, token: str) -> None:
    with _WRITES:
        folder = entry(state, token)
        if folder.exists():
            shutil.rmtree(folder)


def rename(state: pathlib.Path, old: str, new: str) -> None:
    with _WRITES:
        for item in listing(state, old):
            item["person"] = new
            folder = entry(state, item["id"])
            temporary = folder / "meta.tmp"
            temporary.write_text(json.dumps(item))
            temporary.replace(folder / "meta.json")
