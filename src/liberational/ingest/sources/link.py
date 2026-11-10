"""The link source: a URL to a photo, video, or audio file is downloaded as is; any other page goes to yt-dlp, which
knows how to pull the media out of most video and audio sites. One-off: the Ingest tab (or `pack-ingest link`) hands
it each URL, so it has nothing to poll.

A file is one when the server says its Content-Type is image/, video/, or audio/, or, for a server that says
application/octet-stream, when its path ends in a media extension. Downloads stop at `ingest.max_mb`.
"""

import json
import re
import shutil
import subprocess
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

import requests

from liberational import config as cfg
from liberational.ingest.pipeline import safe_suffix
from liberational.ingest.sources import NotConfigured, RemoteItem, SourceState, Status

MEDIA_SUFFIXES = {
    *([".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic", ".heif", ".tif", ".tiff", ".bmp", ".avif"]),
    *([".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".3gp", ".mts"]),
    *([".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".opus", ".wma", ".aiff", ".aif"]),
}
TIMEOUT_S = 30
YT_DLP_TIMEOUT_S = 3 * 3600


class LinkError(Exception):
    """A link that couldn't be fetched, with why."""


def check_url(url: str) -> str:
    """`url` stripped, if it's http or https with a host; LinkError otherwise."""
    url = url.strip()
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise LinkError("a link starts with http:// or https://")
    return url


def file_name(url: str, disposition: str | None = None) -> str:
    """The name to give a downloaded file: Content-Disposition's filename, else the URL path's last part."""
    if disposition:
        m = re.search(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)\"?", disposition, re.IGNORECASE)
        if m:
            return Path(urllib.parse.unquote(m.group(1))).name or "download"
    return Path(urllib.parse.unquote(urllib.parse.urlsplit(url).path)).name or "download"


def is_media(content_type: str, url: str) -> bool:
    kind = content_type.split(";")[0].strip().lower()
    if kind.startswith(("image/", "video/", "audio/")):
        return True
    return (
        kind in ("", "application/octet-stream", "binary/octet-stream")
        and safe_suffix(file_name(url)) in MEDIA_SUFFIXES
    )


@dataclass
class LinkSource:
    yt_dlp: str = "yt-dlp"
    max_bytes: int = 4096 << 20
    name: str = "link"
    label: str = "Link"
    ongoing: bool = False

    @classmethod
    def from_config(cls) -> LinkSource:
        return cls(yt_dlp=str(cfg.get("ingest.yt_dlp")), max_bytes=int(cfg.get("ingest.max_mb")) << 20)

    def status(self) -> Status:
        if shutil.which(self.yt_dlp):
            return Status(True, "Direct links to files, and pages yt-dlp can read")
        return Status(True, f"Direct links to files only: {self.yt_dlp} isn't installed for pages (mise install)")

    def list_new(self, state: SourceState) -> list[RemoteItem]:
        return []  # nothing to poll: each link is handed over by hand

    def item(self, url: str) -> RemoteItem:
        url = check_url(url)
        return RemoteItem(key=url, name=file_name(url), url=url)

    def fetch(self, item: RemoteItem, into: Path) -> Path:
        return self.fetch_with_time(item, into)[0]

    def _save(self, r: requests.Response, into: Path) -> Path:
        dest = into / file_name(r.url, r.headers.get("Content-Disposition"))
        size = 0
        with dest.open("wb") as f:
            for block in r.iter_content(1 << 20):
                size += len(block)
                if size > self.max_bytes:
                    f.close()
                    dest.unlink(missing_ok=True)
                    raise LinkError(f"bigger than {self.max_bytes >> 20} MB (ingest.max_mb)")
                f.write(block)
        return dest

    def _yt_dlp(self, url: str, into: Path) -> tuple[Path, float | None]:
        """yt-dlp's download of the page's media into `into`: (its path, the page's upload time or None)."""
        program = shutil.which(self.yt_dlp)
        if not program:
            raise NotConfigured(f"that link isn't a file, and {self.yt_dlp} isn't installed to read the page")
        args = [
            program, "--no-playlist", "--no-progress", "--no-simulate", "--restrict-filenames",
            "--max-filesize", str(self.max_bytes), "-o", str(into / "%(id)s.%(ext)s"),
            "--print", "before_dl:%(.{id,title,timestamp})j", "--print", "after_move:filepath", url,
        ]  # fmt: skip
        try:
            out = subprocess.run(args, capture_output=True, text=True, timeout=YT_DLP_TIMEOUT_S)
        except subprocess.TimeoutExpired as e:
            raise LinkError("yt-dlp took too long") from e
        lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
        if out.returncode != 0 or not lines:
            why = [ln for ln in out.stderr.splitlines() if ln.strip()]
            raise LinkError(f"yt-dlp couldn't get it: {why[-1] if why else 'no reason given'}")
        path = Path(lines[-1].strip())
        if not path.is_file():
            raise LinkError("yt-dlp said it downloaded a file that isn't there")
        taken = None
        try:
            meta = json.loads(lines[0])
            taken = float(meta["timestamp"]) if meta.get("timestamp") else None
        except ValueError, KeyError, TypeError:
            pass
        return path, taken

    def fetch_with_time(self, item: RemoteItem, into: Path) -> tuple[Path, float | None]:
        """Downloads the file `item.url` names, or has yt-dlp pull the media out of the page (also when the server
        refuses a plain request, as some sites do): (its path, the page's upload time when yt-dlp found one; a
        file's own time is read from it later)."""
        if not item.url:
            raise LinkError("no link")
        refused = None
        try:
            with requests.get(item.url, stream=True, timeout=TIMEOUT_S, allow_redirects=True) as r:
                r.raise_for_status()
                if is_media(r.headers.get("Content-Type", ""), r.url):
                    return self._save(r, into), None
        except requests.RequestException as e:
            refused = e
        try:
            return self._yt_dlp(item.url, into)
        except NotConfigured:
            if refused is not None:
                raise LinkError(f"couldn't fetch the link: {refused}") from refused
            raise
