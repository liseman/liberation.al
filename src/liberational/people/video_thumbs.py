"""Recent-video thumbnails for the person and pet cards, kept as files, the way face crops are (face_shots.py).

A thumbnail is one recorded frame of a visit, the full scene at most WIDTH x HEIGHT. Card listing queues a background
decode from the camera's video; only prepared files are listed and served by /people/video-thumb.jpg. It is saved as
`state/video_thumbs/<camera>-<frame ms>-<version>-r<revision>.jpg`. Every later card reads the file, with no lookup and
no decode. There's no write-time hook: when a visit closes its opening frame is long gone from the live frames.

The file name carries the segment store's deletion revision, as a crop's does: after a forget moves it, a thumbnail is
cut again from what the video still has (a deleted moment has no frame, so no thumbnail). The cache follows the
video's 30-day window by the frame's own time, not the file's, and a byte cap, whichever is reached first.
"""

from __future__ import annotations

import functools
import os
import re
import time
from pathlib import Path

KIND = "video_thumbs"
WIDTH, HEIGHT = 320, 180
VERSION = f"scene-{WIDTH}x{HEIGHT}"  # the size is part of the name: a new size is a new file, never a stale one
KEEP_DAYS = 30  # recorded video is kept 30 days: a thumbnail of a moment past it has no video either
CAP_BYTES = 256 * 1024 * 1024  # about 15 KB a thumbnail: four a card for every person and pet, many times over
CAMERA = re.compile(r"[A-Za-z0-9_.-]{1,128}")
_NAME = re.compile(rf"(?P<cam>[A-Za-z0-9_.-]+)-(?P<ms>\d+)-{re.escape(VERSION)}-r(?P<rev>[0-9]+)\.jpg")


def folder(state: Path) -> Path:
    return state / KIND


def saved(state: Path, revision: str) -> frozenset[str]:
    """Current-revision file names, indexed once per directory change or second, as face crops are."""
    directory = folder(state)
    try:
        stat = directory.stat()
    except FileNotFoundError:
        return frozenset()
    return _saved(directory, revision, stat.st_mtime_ns, stat.st_ctime_ns, int(time.monotonic()))


@functools.lru_cache(maxsize=8)
def _saved(directory: Path, revision: str, mtime: int, ctime: int, second: int) -> frozenset[str]:
    return frozenset(p.name for p in directory.glob(f"*-{VERSION}-r{revision}.jpg"))


def path(state: Path, cam: str, at: float, revision: str) -> Path:
    """Where the thumbnail of camera `cam`'s frame at `at` (seconds) is, checked under deletion revision `revision`."""
    if not CAMERA.fullmatch(cam):
        raise ValueError("bad camera name")
    return folder(state) / f"{cam}-{round(at * 1000)}-{VERSION}-r{revision}.jpg"


def others(state: Path, cam: str, at: float, revision: str) -> list[Path]:
    """The same frame's thumbnails of other revisions: from before a forget, so cut again rather than served."""
    current = path(state, cam, at, revision).name
    return [p for p in folder(state).glob(f"{cam}-{round(at * 1000)}-{VERSION}-r*.jpg") if p.name != current]


def write(state: Path, cam: str, at: float, revision: str, jpg: bytes) -> Path:
    """Saves a thumbnail atomically, so a reader never sees half of one."""
    out = path(state, cam, at, revision)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    tmp.write_bytes(jpg)
    tmp.replace(out)
    return out


def prune(state: Path, now: float | None = None, cap_bytes: int = CAP_BYTES) -> int:
    """Drops thumbnails of frames older than KEEP_DAYS, then the oldest frames until the rest are under 80% of
    `cap_bytes`, and any file this version doesn't name. Returns how many were removed."""
    now = time.time() if now is None else now
    files: list[tuple[float, int, Path]] = []
    removed = 0
    for f in folder(state).glob("*.jpg"):
        m = _NAME.fullmatch(f.name)
        try:
            size = f.stat().st_size
        except OSError:
            continue  # another process pruned it
        if m is None:  # another size or an old version: never served again
            f.unlink(missing_ok=True)
            removed += 1
            continue
        files.append((int(m["ms"]) / 1000, size, f))
    files.sort()
    total = sum(size for _, size, _ in files)
    for at, size, f in files:
        if at >= now - KEEP_DAYS * 86400 and total <= cap_bytes * 0.8:
            break
        f.unlink(missing_ok=True)
        total -= size
        removed += 1
    return removed
