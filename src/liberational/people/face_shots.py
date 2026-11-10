"""Face thumbnails: one small square JPEG per face sighting, in `state/shots/`, served as /faces/shot/<id>.jpg (#2018).

see-actions writes a sighting's crop when it writes its `face.match` row, from the frame it already has, so People's
grids read files instead of decoding video. The dashboard cuts the crop from recorded video only for a sighting
without one (older rows, or one written before this). Both cut it the same way (`square`, `encode`).

A crop's file name carries the segment store's deletion revision it was checked under. The dashboard serves a crop of
the current revision as is; after a forget moves the revision, it checks each crop's sighting once more before
serving it, so a crop of a deleted sighting is never served. The cache follows the video's retention (`KEEP_DAYS`)
and a byte cap, whichever is reached first.
"""

from __future__ import annotations

import collections
import functools
import os
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from liberational import logs

LOG = logs.get("face-shots")

PX = 160
GRID_PX = 80  # forty-CSS-pixel naming tiles and 36px avatars, including high-density displays
# Derived crops can be regenerated; salt pixel-geometry changes without replacing chosen identity photos.
VERSION = "source-geometry-2"
KEEP_DAYS = 30  # recorded video is kept 30 days (Joe, 2026-11-03): a crop of a sighting past it has no video either
CAP_BYTES = 1536 * 1024 * 1024  # about 7 KB a crop: 30 days of 2026-11-03's 4,800 sightings a day, with room
QUALITY = 82


RACY_NS = 2_000_000_000  # a directory changed this recently may share its timestamps with its next change


def saved(state: Path, kind: str = "shots", revision: str | None = None) -> frozenset[str]:
    """Directory-indexed crop IDs. Own writes invalidate it; external writes refresh within one second.

    The directory is read once per change of its timestamps, shared by every kind and revision in it (#2312). A
    timestamp can alias two quick changes, so a directory changed within RACY_NS is read again each second until it
    settles; callers also verify each returned current-revision file.
    """
    directory = folder(state, kind.removesuffix("_grid"))
    try:
        stat = directory.stat()
    except FileNotFoundError:
        return frozenset()
    racy = time.time_ns() - max(stat.st_mtime_ns, stat.st_ctime_ns) < RACY_NS
    second = int(time.monotonic()) if racy else None
    return _saved(directory, revision, _version(kind), stat.st_mtime_ns, stat.st_ctime_ns, second)


@functools.lru_cache(maxsize=8)
def _saved(
    directory: Path, revision: str | None, version: str, mtime: int, ctime: int, second: int | None
) -> frozenset[str]:
    # The same names glob("*-{version}-r{revision or *}.jpg") gives, without a Path per file.
    tail = f"-{version}-r"
    names = _names(directory, mtime, ctime, second)
    if revision is None:
        found = (n for n in names if n.endswith(".jpg") and tail in n[:-4])
    else:
        end = f"{tail}{revision}.jpg"
        found = (n for n in names if n.endswith(end))
    return frozenset(n.split(f"-{version}", 1)[0] for n in found)


@functools.lru_cache(maxsize=4)
def _names(directory: Path, mtime: int, ctime: int, second: int | None) -> tuple[str, ...]:
    """Every file name in the directory at these timestamps (the key; see saved)."""
    try:
        with os.scandir(directory) as entries:
            return tuple(entry.name for entry in entries)
    except FileNotFoundError:
        return ()


def _forget_index() -> None:
    _saved.cache_clear()
    _names.cache_clear()


CropKey = tuple[Path, str, str, str]  # state, kind, ID, deletion revision


@dataclass(frozen=True)
class CropResult:
    ready: bool
    decoded: bool = False


@dataclass(frozen=True)
class CropRetry:
    """A transient frame/RPC failure: keep pending with per-job backoff instead of marking media unavailable."""

    decoded: bool = False


class Pending:
    """Bounded, deduplicated crop jobs, started explicitly by the dashboard, never by a list read or import.

    The processor shares the dashboard's decode gate. Failed results and exceptions stay unavailable until a new
    deletion revision. None means decode-slot contention; CropRetry means a transient frame/RPC failure, retried
    with exponential backoff from one to sixty seconds. Admission is capped per kind, and the kinds take turns, so a
    flood of one kind (thousands of face crops to cut after their cache was cleared) can't keep another kind's few
    jobs (a pet's photo, a video's thumbnail) out of the queue or waiting behind it (#2579). The failure memo
    lasts for the current revision so an unresolvable history cannot continually refill the queue.
    No queue or failure state is durable; a dashboard restart rechecks requested crops.
    """

    LIMIT = 4096  # queued jobs of one kind

    def __init__(self, process: Callable[[CropKey], CropResult | CropRetry | None]) -> None:
        self.process = process
        self.condition = threading.Condition()
        self.jobs: dict[str, collections.deque[CropKey]] = {}  # each kind's queue, served in turn
        self.pending: set[CropKey] = set()
        self.queued: collections.Counter[str] = collections.Counter()  # pending jobs of each kind
        self.failed: set[CropKey] = set()
        self.retry_due: dict[CropKey, float] = {}
        self.retry_attempts: dict[CropKey, int] = {}
        self.revisions: dict[tuple[Path, str], str] = {}
        self.last_decoded = False
        self.last_busy = False
        self.stopped = threading.Event()
        self.thread: threading.Thread | None = None
        self.priority: Callable[[], None] | None = None

    def offer(self, state: Path, kind: str, sid: str, revision: str) -> bool:
        """Whether this missing crop is pending. Overflow stays pending for admission on the next list refresh."""
        key = state, kind, sid, revision
        with self.condition:
            scope = state, kind
            if self.revisions.get(scope) != revision:
                self.failed = {k for k in self.failed if k[:2] != scope}
                self.revisions[scope] = revision
            if key in self.failed:
                return False
            if key not in self.pending and self.queued[kind] < self.LIMIT:
                self.pending.add(key)
                self.queued[kind] += 1
                self.jobs.setdefault(kind, collections.deque()).append(key)
                self.condition.notify()
        if self.priority is not None and self.thread is not None and not self.thread.is_alive():
            self.start(self.priority)
        return True

    def one(self) -> bool:
        """Process one queued job; usable synchronously by tests without launching background work."""
        with self.condition:
            key = self._next()
            if key is None:
                self.last_decoded = self.last_busy = False
                return False
        try:
            result = self.process(key)
        except TimeoutError:
            result = CropRetry()
        except Exception:
            LOG.exception("background thumbnail failed")
            result = CropResult(False)
        self.last_decoded = result.decoded if result is not None else False
        self.last_busy = result is None
        with self.condition:
            current = self.revisions.get(key[:2]) == key[3]
            if isinstance(result, CropRetry) and current:
                attempt = self.retry_attempts.get(key, 0) + 1
                self.retry_attempts[key] = attempt
                self.retry_due[key] = time.monotonic() + min(60.0, 2.0 ** min(attempt - 1, 6))
                self.jobs[key[1]].append(key)
            elif result is None and current:
                self.jobs[key[1]].append(key)
            else:
                self.retry_due.pop(key, None)
                self.retry_attempts.pop(key, None)
                self.pending.discard(key)
                self.queued[key[1]] -= 1
                if isinstance(result, CropResult) and not result.ready and current:
                    self.failed.add(key)
        return True

    def _next(self) -> CropKey | None:
        """The next due job, taking the kinds in turn: the kind served last goes to the back. Holds the condition."""
        now = time.monotonic()
        for kind in list(self.jobs):
            jobs = self.jobs.pop(kind)
            self.jobs[kind] = jobs  # to the back, whether it has a due job or not
            for _ in range(len(jobs)):
                key = jobs.popleft()
                if self.retry_due.get(key, 0) <= now:
                    return key
                jobs.append(key)
        return None

    def _waiting(self) -> list[CropKey]:
        return [key for jobs in self.jobs.values() for key in jobs]

    def start(self, lower_priority: Callable[[], None]) -> None:
        """One low-priority worker, at most four real decodes per second; cheap checks drain without delay."""
        with self.condition:
            if self.thread is not None and self.thread.is_alive():
                return
            self.priority = lower_priority
            self.stopped = stopped = threading.Event()

            def run() -> None:
                try:
                    lower_priority()
                    while not stopped.is_set():
                        with self.condition:
                            self.condition.wait_for(lambda: any(self.jobs.values()) or stopped.is_set())
                            if (waiting := self._waiting()) and not stopped.is_set():
                                delay = min(self.retry_due.get(key, 0) for key in waiting) - time.monotonic()
                                if delay > 0:
                                    self.condition.wait(timeout=delay)
                                    continue
                        if stopped.is_set():
                            break
                        self.one()
                        if self.last_decoded or self.last_busy:
                            stopped.wait(0.25)
                except Exception:
                    LOG.exception("background thumbnail worker stopped")

            self.thread = threading.Thread(target=run, name="thumbnail-crops", daemon=True)
            self.thread.start()

    def stop(self) -> None:
        self.stopped.set()
        with self.condition:
            self.condition.notify_all()
        thread = self.thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)


def folder(state: Path, kind: str = "shots") -> Path:
    """The crops' folder: `shots` for faces, `pet_shots` for animals (people/pet_shots.py)."""
    return state / kind


def _version(kind: str) -> str:
    return f"{VERSION}-grid{GRID_PX}" if kind.endswith("_grid") else VERSION


def path(state: Path, sid: str, revision: str, kind: str = "shots") -> Path:
    """Where sighting sid's crop is, checked under deletion revision `revision`."""
    return folder(state, kind.removesuffix("_grid")) / f"{sid}-{_version(kind)}-r{revision}.jpg"


def others(state: Path, sid: str, revision: str, kind: str = "shots") -> list[Path]:
    """sid's crops of other revisions, keeping original and grid geometries separate for revalidation."""
    current = path(state, sid, revision, kind).name
    return [
        p for p in folder(state, kind.removesuffix("_grid")).glob(f"{sid}-{_version(kind)}-r*.jpg") if p.name != current
    ]


def promote(previous: Path, current: Path) -> bool:
    """Keep a revalidated crop under this revision, accepting another reader's completed rename."""
    try:
        previous.replace(current)
    except FileNotFoundError:
        return current.is_file()
    _forget_index()
    return True


def in_frame(box: Any) -> list[float] | None:
    """A face box (xyxy shares of the frame) clipped to the frame, or None when nothing of it is inside. A face at the
    frame's edge has a box past it (the detector's box around a half-seen head): 1,524 of 2026-11-03's 7,468 face
    rows did, and every crop of them was refused as out of range, so their shots 404ed."""
    x1, y1, x2, y2 = (min(1.0, max(0.0, float(v))) for v in box[:4])
    return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None


def square(width: int, height: int, box: tuple[float, float, float, float]) -> tuple[int, int, int] | None:
    """The square (left, top, side) in a width x height frame around a face box in its pixels, half again as wide as
    the face's longer side, kept inside the frame; None when the face is too small to show."""
    x1, y1, x2, y2 = box
    side = int(min(width, height, max(x2 - x1, y2 - y1) * 1.5))
    if side < 8:
        return None
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = int(min(max(0, cx - side / 2), width - side))
    top = int(min(max(0, cy - side / 2), height - side))
    return left, top, side


def encode(patch: Any) -> bytes | None:
    """A square BGR patch as the PX x PX JPEG served."""
    import cv2

    if patch is None or patch.size == 0:
        return None
    ok, jpg = cv2.imencode(
        ".jpg", cv2.resize(patch, (PX, PX), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, QUALITY]
    )
    return jpg.tobytes() if ok else None


def write(state: Path, sid: str, revision: str, jpg: bytes, kind: str = "shots") -> Path:
    """Saves a crop atomically, so a reader never sees half of one."""
    out = path(state, sid, revision, kind)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{out.name}.", suffix=".tmp", dir=out.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(jpg)
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
    if not kind.endswith("_grid"):
        path(state, sid, revision, kind + "_grid").unlink(missing_ok=True)
    _forget_index()
    return out


def prune(state: Path, now: float | None = None, cap_bytes: int = CAP_BYTES, kind: str = "shots") -> int:
    """Drops crops older than KEEP_DAYS, then the oldest until the rest are under 80% of `cap_bytes`. Crops removed."""
    now = time.time() if now is None else now
    files = []
    for f in folder(state, kind).glob("*.jpg"):
        try:
            st = f.stat()
        except OSError:
            continue  # another process pruned it
        files.append((st.st_mtime, st.st_size, f))
    files.sort()
    total = sum(size for _, size, _ in files)
    removed = 0
    for mtime, size, f in files:
        if mtime >= now - KEEP_DAYS * 86400 and total <= cap_bytes * 0.8:
            break
        f.unlink(missing_ok=True)
        total -= size
        removed += 1
    return removed


class Writer:
    """see-actions' side: each sighting's square patch saved as its crop under the current deletion revision, and the
    cache pruned now and then, off the frame loop."""

    PRUNE_EVERY = 500  # crops written between prunes: about 10 minutes at 2026-11-03's rate

    def __init__(self, state: Path) -> None:
        from liberational import segment_log

        self.state, self.segments = state, segment_log.root(state)
        self.written = 0

    def __call__(self, sid: str, patch: Any) -> None:
        from continuum_py import segment_revision

        jpg = encode(patch)
        if jpg is None:
            return
        try:
            write(self.state, sid, str(segment_revision(self.segments)), jpg)
        except OSError as e:  # a full disk costs the thumbnail (cut from video on request instead), not the frame loop
            LOG.warning("face thumbnail %s not saved: %s", sid, e)
            return
        self.written += 1
        if self.written % self.PRUNE_EVERY == 0:
            threading.Thread(target=prune, args=(self.state,), name="face-shots-prune", daemon=True).start()


def cut(state: Path, row: Any) -> bytes | None:
    """Sighting `row`'s crop ({cam, ts, bbox_norm}) cut from its recorded frame, or None when the frame is gone or the
    box can't be placed on it. The slow way (a video decode): for sightings see-actions wrote no crop for."""
    from liberational import vision_log
    from liberational.see.frames import source_box_to_decoded

    evidence = vision_log.at(state).frame_evidence(str(row["cam"]), float(row["ts"]))
    if evidence is None or evidence.image is None:
        return None
    frame = evidence.image
    h, w = frame.shape[:2]
    inside = in_frame(row["bbox_norm"])
    box = source_box_to_decoded(inside, evidence.metadata) if inside and evidence.metadata is not None else inside
    if box is None:
        return None  # the box can't be placed on this frame: no crop, rather than a guess
    x1, y1, x2, y2 = (float(v) for v in box)
    sq = square(w, h, (x1 * w, y1 * h, x2 * w, y2 * h))
    if sq is None:
        return None
    left, top, side = sq
    return encode(frame[top : top + side, left : left + side])


def backfill(state: Path, days: float = KEEP_DAYS, per_second: float = 8.0, now: float | None = None) -> dict[str, int]:
    """Crops for the retained sightings that have none (#2018): a camera at a time, oldest first, so neighbouring frames
    share a decoded stretch of video, at most `per_second` cuts a second. Counts of what it did."""
    from continuum_py import segment_revision

    from liberational import segment_log
    from liberational.vision import sightings

    now = time.time() if now is None else now
    revision = str(segment_revision(segment_log.root(state)))
    rows = [f for f in sightings.faces(now - days * 86400, now, state=state) if f.get("bbox_norm")]
    done = {"rows": len(rows), "had": 0, "cut": 0, "gone": 0}
    for cam in sorted({str(f["cam"]) for f in rows}):
        for f in sorted((f for f in rows if str(f["cam"]) == cam), key=lambda f: f["ts"]):
            sid = str(f["id"])
            if path(state, sid, revision).exists() or others(state, sid, revision):
                done["had"] += 1  # an older revision's crop is checked when it's asked for
                continue
            started = time.monotonic()
            jpg = cut(state, f)
            if jpg is None:
                done["gone"] += 1
            else:
                write(state, sid, revision, jpg)
                done["cut"] += 1
            time.sleep(max(0.0, 1 / per_second - (time.monotonic() - started)))
        LOG.info("face thumbnails: %s done, %s so far", cam, done)
    prune(state)
    return done


def main() -> None:
    """`python -m liberational.people.face_shots backfill [--days N] [--per-second N]`: crops for the retained
    sightings that have none, once (#2018). Run it niced; it reads recorded video at the given rate."""
    import argparse

    from liberational import config as cfg

    parser = argparse.ArgumentParser(prog="face_shots")
    parser.add_argument("command", choices=["backfill"])
    parser.add_argument("--days", type=float, default=KEEP_DAYS)
    parser.add_argument("--per-second", type=float, default=8.0)
    args = parser.parse_args()
    print(backfill(cfg.path("paths.state"), args.days, args.per_second), flush=True)


if __name__ == "__main__":
    main()
