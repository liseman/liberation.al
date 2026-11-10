"""The Google Photos source: a Playwright browser profile under `<paths.ingest>/google_photos/profile` that the owner
signs in to once, by hand, in a window the Ingest tab opens on this machine (`login`). Liberational stores no
password or token of its own: the profile's cookies are the sign-in, and deleting the directory signs out.

With the profile signed in, the tab lists albums (`albums`) and an album's photos and videos, or the library's
newest (`photos`), read from the pages photos.google.com draws, headless; the owner picks what to bring in, and
`fetch` downloads each original through the same profile (the thumbnail's base URL with `=d`, or `=dv` for a video),
streaming it to disk and stopping at `ingest.max_mb`.
Reading the pages is fragile by nature: when Google changes them, `albums` and `photos` find nothing, and
`parse_tile` and `parse_album` are what to fix. Playwright is the `ingest` extra (`uv sync --inexact --extra ingest`, then
`uv run playwright install chromium`).

One-off, not polled: the profile can only be open in one browser at a time, so listing while the sign-in window is
open says so.
"""

import argparse
import datetime as dt
import os
import re
import subprocess
import sys
import tempfile
import threading
import urllib.parse
from collections.abc import Callable, Generator, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from liberational import config as cfg
from liberational.ingest.sources import NotConfigured, RemoteItem, SourceState, Status

HOME = "https://photos.google.com/"
ALBUMS = "https://photos.google.com/albums"
# Where the originals are served from; fetch refuses any other host, so the profile's cookies go nowhere else.
MEDIA_HOST = re.compile(r"(^|\.)googleusercontent\.com$")
SCROLLS = 8
_CHUNK = 1 << 20
_DOWNLOAD_TIMEOUT_S = 600
_REDIRECTS = 5
_LABEL_TIME = re.compile(r"([A-Z][a-z]{2} \d{1,2}, \d{4}, \d{1,2}:\d{2}(?::\d{2})? [AP]M)")
_open = threading.Lock()

# The links and thumbnails a photos.google.com page draws, as plain data for parse_tile and parse_album.
_TILES_JS = """() => Array.from(document.querySelectorAll('a[href*="photo/"]')).map(a => {
  const bg = a.querySelector('[style*="background-image"]');
  return {href: a.getAttribute('href') || '', label: a.getAttribute('aria-label') || '',
          thumb: bg ? bg.style.backgroundImage : ''};
})"""
_ALBUMS_JS = """() => Array.from(document.querySelectorAll('a[href*="album/"]')).map(a => ({
  href: a.getAttribute('href') || '', label: a.getAttribute('aria-label') || a.innerText || ''}))"""


def label_time(label: str) -> float | None:
    """The time in a tile's label ("Photo - Landscape - Oct 31, 2026, 4:12:33 PM"), in local time, or None."""
    text = label.replace(" ", " ").replace("\xa0", " ")
    m = _LABEL_TIME.search(text)
    if not m:
        return None
    for fmt in ("%b %d, %Y, %I:%M:%S %p", "%b %d, %Y, %I:%M %p"):
        try:
            return dt.datetime.strptime(m.group(1), fmt).timestamp()
        except ValueError:
            continue
    return None


def _thumb_base(thumb: str) -> str | None:
    """A thumbnail's base URL (before its `=w..-h..` size options) from its CSS `url("...")`, or None."""
    m = re.search(r"url\(\"?([^\")]+)\"?\)", thumb)
    if not m:
        return None
    url = m.group(1)
    host = urllib.parse.urlsplit(url).hostname or ""
    if not MEDIA_HOST.search(host):
        return None
    return url.split("=", 1)[0]


def parse_tile(href: str, label: str, thumb: str) -> RemoteItem | None:
    """One photo or video tile as an item: its id from the link, kind and time from its label, and the original's
    download URL from its thumbnail. None for a tile without all three."""
    m = re.search(r"photo/([A-Za-z0-9_-]+)", href)
    base = _thumb_base(thumb)
    if not m or not base:
        return None
    video = label.lower().startswith("video")
    pid = m.group(1)
    return RemoteItem(
        key=pid,
        name=f"{pid[:24]}.{'mp4' if video else 'jpg'}",
        taken_at=label_time(label),
        url=base + ("=dv" if video else "=d"),
        extra={"kind": "video" if video else "photo", "thumb": base + "=w256-h256-c", "label": label},
    )


def parse_album(href: str, label: str) -> dict[str, str] | None:
    """An album link as {"id", "title"}, or None for another link."""
    m = re.search(r"album/([A-Za-z0-9_-]+)", href)
    if not m:
        return None
    title = label.strip().splitlines()[0] if label.strip() else m.group(1)
    return {"id": m.group(1), "title": title}


def _unique[T](items: Sequence[T | None], key: Callable[[T], str]) -> list[T]:
    """`items` without the Nones and without repeats (scrolling reads a tile again), in order."""
    seen: set[str] = set()
    out: list[T] = []
    for it in items:
        if it is not None and key(it) not in seen:
            seen.add(key(it))
            out.append(it)
    return out


def _media_url(url: str) -> None:
    """Raise ValueError unless `url` is https on Google's media host."""
    parts = urllib.parse.urlsplit(url)
    host = parts.hostname or ""
    if parts.scheme != "https" or not MEDIA_HOST.search(host):
        raise ValueError(f"not a Google Photos download: {url!r}")


def _over_cap(max_bytes: int) -> OSError:
    return OSError(f"bigger than {max_bytes >> 20} MB (ingest.max_mb)")


def _stream_to(chunks: Iterator[bytes], dest: Path, max_bytes: int) -> None:
    """Write `chunks` to a temporary file beside `dest`, then move it into place.

    The temporary file is removed when the download passes `max_bytes` or the stream
    raises. `dest` stays as it was until the whole stream fits. Do not read an original
    with Playwright's `APIResponse.body()`: that call keeps the whole file in memory.
    """
    fd, name = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".part", dir=dest.parent)
    tmp = Path(name)
    size = 0
    try:
        with os.fdopen(fd, "wb") as handle:
            for block in chunks:
                if not block:
                    continue
                size += len(block)
                if size > max_bytes:
                    raise _over_cap(max_bytes)
                handle.write(block)
        os.replace(tmp, dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _session(ctx: Any) -> requests.Session:
    """A requests session carrying the signed-in profile's cookies and user agent."""
    session = requests.Session()
    for cookie in ctx.cookies():
        options: dict[str, str] = {"path": str(cookie.get("path") or "/")}
        domain = cookie.get("domain")
        if domain:
            options["domain"] = str(domain)
        session.cookies.set(str(cookie["name"]), str(cookie["value"]), **options)
    page = ctx.new_page()
    try:
        session.headers["User-Agent"] = str(page.evaluate("navigator.userAgent"))
    finally:
        page.close()
    return session


def _length(response: requests.Response) -> int | None:
    raw = response.headers.get("Content-Length")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _get_original(session: requests.Session, url: str, name: str) -> requests.Response:
    """GET `url`, staying on the media host across redirects. The caller closes the response."""
    current = url
    for _ in range(_REDIRECTS):
        _media_url(current)
        response = session.get(current, stream=True, timeout=_DOWNLOAD_TIMEOUT_S, allow_redirects=False)
        if response.is_redirect or response.is_permanent_redirect:
            location = response.headers.get("Location")
            response.close()
            if not location:
                raise OSError(f"Google Photos redirected {name} without a location")
            current = urllib.parse.urljoin(current, location)
            continue
        if not response.ok:
            status = response.status_code
            response.close()
            raise OSError(f"Google Photos refused {name} ({status})")
        return response
    raise OSError(f"Google Photos redirected {name} too many times")


@dataclass
class GooglePhotosSource:
    profile: Path
    max_bytes: int = 4096 << 20
    name: str = "google_photos"
    label: str = "Google Photos"
    ongoing: bool = False

    @classmethod
    def from_config(cls) -> GooglePhotosSource:
        return cls(
            profile=cfg.path("paths.ingest") / "google_photos" / "profile",
            max_bytes=int(cfg.get("ingest.max_mb")) << 20,
        )

    def signed_in_once(self) -> bool:
        """Whether the sign-in window has been opened before (the profile exists); not proof the sign-in holds."""
        return self.profile.is_dir() and any(self.profile.iterdir())

    def status(self) -> Status:
        try:
            import playwright  # noqa: F401
        except ImportError:
            return Status(
                False,
                "Google Photos needs an extra package installed on this machine before you can sign in.",
                blocked=True,
                fix="uv sync --inexact --extra ingest, then uv run playwright install chromium",
            )
        if not self.signed_in_once():
            return Status(False, "Not signed in: open the sign-in window and sign in to Google once")
        return Status(True, "Signed in through the browser profile")

    def login(self) -> subprocess.Popen[bytes]:
        """Opens the sign-in window on this machine's screen, in its own process, and returns at once. The window
        stays until it's closed."""
        self.profile.mkdir(parents=True, exist_ok=True)
        return subprocess.Popen(
            [sys.executable, "-m", "liberational.ingest.sources.google_photos", "login", str(self.profile)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    @contextmanager
    def _context(self) -> Generator[Any]:
        if not self.signed_in_once():
            raise NotConfigured("Google Photos isn't signed in yet: open the sign-in window first")
        try:
            from playwright.sync_api import Error as PlaywrightError
            from playwright.sync_api import sync_playwright
        except ImportError as e:
            raise NotConfigured("Google Photos needs an extra package installed on this machine") from e
        if not _open.acquire(timeout=120):
            raise NotConfigured("Google Photos is busy with another request")
        try:
            with sync_playwright() as p:
                try:
                    ctx = p.chromium.launch_persistent_context(str(self.profile), headless=True)
                except PlaywrightError as e:
                    raise NotConfigured(f"couldn't open the profile (is the sign-in window still open?): {e}") from e
                try:
                    yield ctx
                finally:
                    ctx.close()
        finally:
            _open.release()

    def _read(self, ctx: Any, url: str, script: str, scrolls: int = 0) -> list[dict[str, str]]:
        page = ctx.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        if "accounts.google.com" in page.url:
            raise NotConfigured("Google Photos isn't signed in: open the sign-in window and sign in")
        page.wait_for_timeout(2500)
        found: list[dict[str, str]] = list(page.evaluate(script))
        for _ in range(scrolls):
            page.mouse.wheel(0, 4000)
            page.wait_for_timeout(1200)
            found += page.evaluate(script)
        page.close()
        return found

    def albums(self) -> list[dict[str, str]]:
        with self._context() as ctx:
            raw = self._read(ctx, ALBUMS, _ALBUMS_JS)
        return _unique([parse_album(r["href"], r["label"]) for r in raw], lambda a: a["id"])

    def photos(self, album: str | None = None, scrolls: int = SCROLLS) -> list[RemoteItem]:
        if album is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", album):
            raise ValueError("not an album id")
        url = f"{HOME}album/{album}" if album else HOME
        with self._context() as ctx:
            raw = self._read(ctx, url, _TILES_JS, scrolls)
        return _unique([parse_tile(r["href"], r["label"], r["thumb"]) for r in raw], lambda i: i.key)

    def list_new(self, state: SourceState) -> list[RemoteItem]:
        return [i for i in self.photos() if not state.seen(i.key)]

    def fetch(self, item: RemoteItem, into: Path) -> Path:
        return next(iter(self.fetch_many([item], into)))[1]

    def fetch_many(self, items: Sequence[RemoteItem], into: Path) -> Iterator[tuple[RemoteItem, Path]]:
        """Downloads each item's original through the signed-in profile, streaming it to disk.

        Refuses a URL on any other host. A download bigger than `max_bytes` (`ingest.max_mb`) is
        removed. The bytes are read in chunks into a temporary file, then moved onto the destination.
        """
        for item in items:
            _media_url(item.url or "")
        with self._context() as ctx:
            session = _session(ctx)
            for item in items:
                response = _get_original(session, item.url or "", item.name)
                dest = into / item.name
                try:
                    declared = _length(response)
                    if declared is not None and declared > self.max_bytes:
                        raise _over_cap(self.max_bytes)
                    _stream_to(response.iter_content(_CHUNK), dest, self.max_bytes)
                finally:
                    response.close()
                yield item, dest


def _login_window(profile: Path) -> None:
    """The sign-in window: headed, on Google Photos, until it's closed."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(str(profile), headless=False)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(HOME)
        page.wait_for_event("close", timeout=0)  # closing the window closes its page; the profile is saved


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Google Photos sign-in window (the Ingest tab opens it)")
    parser.add_argument("command", choices=["login"])
    parser.add_argument("profile", type=Path)
    args = parser.parse_args(argv)
    _login_window(args.profile)


if __name__ == "__main__":
    main()
