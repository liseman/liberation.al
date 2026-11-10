"""The frontend's built files (dashboard/static/), shared by the dashboard and the object memory server.

Both serve them at /static/<path>. Only files inside static/ with a known type are served.

The frontend (frontend/) is built by Vite into static/app/ (`mise run frontend`; gitignored). Its pages are HTML
files there (`built_page`), and its scripts, styles, and fonts are content-hashed files under static/app/assets/, so
those can be cached for good (`cache_control`). When the build is missing, a page says how to build it instead of a
404, so nothing in Python needs Node to run.
"""

import gzip
import html
import pathlib
from functools import lru_cache

from liberational.config import CHECKOUT

STATIC = pathlib.Path(__file__).resolve().parent / "static"
APP = STATIC / "app"  # Vite's output (frontend/vite.config.ts), served at /static/app/
MANIFEST = APP / ".vite" / "manifest.json"
FRONTEND = CHECKOUT / "frontend" if CHECKOUT else None  # the sources, in a checkout only
IMMUTABLE = "public, max-age=31536000, immutable"
TYPES = {
    ".mjs": "text/javascript; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".woff2": "font/woff2",
    ".woff": "font/woff",
    ".map": "application/json; charset=utf-8",  # the build's sourcemaps
}

NOT_BUILT = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Not built yet</title></head>
<body style="font: 16px system-ui, sans-serif; max-width: 36em; margin: 3em auto; padding: 0 1em">
<h1>The dashboard isn't built yet</h1>
<p>{page} is missing from <code>src/liberational/dashboard/static/app/</code>. Build it on this machine with
<code>mise run setup</code> (or <code>mise run frontend</code>), then reload.</p>
</body></html>
"""

STALE_BANNER = (
    '<div role="status" style="position:fixed;bottom:0;left:0;right:0;z-index:99999;padding:6px 12px;'
    'font:13px system-ui,sans-serif;background:#7a5a00;color:#fff">This build is older than frontend/: run '
    "<code>mise run frontend</code> and reload.</div>"
)


def accepts_gzip(header: str) -> bool:
    """Honor explicit gzip;q=0, including when a wildcard also allows compression."""
    qualities = {}
    for part in header.lower().split(","):
        coding, *parameters = part.strip().split(";")
        quality = 1.0
        for parameter in parameters:
            key, _, value = parameter.strip().partition("=")
            if key == "q":
                try:
                    quality = float(value)
                except ValueError:
                    quality = 0.0
        qualities[coding] = quality
    return 0 < qualities.get("gzip", qualities.get("*", 0.0)) <= 1


@lru_cache(maxsize=64)
def compressed(body: bytes) -> bytes:
    """Small bounded cache for static source bytes; source changes cannot reuse stale output."""
    return gzip.compress(body, compresslevel=6, mtime=0)


def read(path: str) -> tuple[bytes, str] | None:
    """(bytes, content type) for a /static/... URL path, or None when it is not a servable file."""
    rel = path.removeprefix("/static/")
    f = (STATIC / rel).resolve()
    if rel == path or f.suffix not in TYPES or not f.is_relative_to(STATIC) or not f.is_file():
        return None
    return f.read_bytes(), TYPES[f.suffix]


def cache_control(path: str) -> str:
    """The Cache-Control for a /static/... URL path: Vite's content-hashed files under /static/app/assets/ never
    change, so they keep for a year; everything else is fetched fresh."""
    return IMMUTABLE if path.startswith("/static/app/assets/") else "no-store"


def built_page(name: str) -> tuple[str, bool]:
    """(HTML, built) for a page of the new frontend, such as "index.html" or "objects.html". When the build has
    no such page, a plain page that says how to build it, and False."""
    f = (APP / name).resolve()
    if f.suffix == ".html" and f.is_relative_to(APP) and f.is_file():
        return f.read_text(), True
    return NOT_BUILT.format(page=html.escape(name)), False


def stale(frontend: pathlib.Path | None = None, manifest: pathlib.Path | None = None) -> bool:
    """Whether a frontend source (src/, its HTML pages, config, and lock file) changed after the last build. False
    with no build or no frontend/ (a wheel has none, and an install that was never built says so in `built_page` instead)."""
    frontend, manifest = frontend or FRONTEND, manifest or MANIFEST
    if frontend is None:
        return False
    try:
        built = manifest.stat().st_mtime_ns
    except OSError:
        return False
    sources = [
        *frontend.glob("src/**/*"),
        *frontend.glob("*.html"),
        *frontend.glob("*.ts"),
        frontend / "package-lock.json",
    ]
    return any(f.is_file() and f.stat().st_mtime_ns > built for f in sources)
