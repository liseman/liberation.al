"""The one path every ingested file takes, whatever it came from (an upload, a link, a source's poll):

1. `add`: hash the original (SHA-256). Bytes already ingested are a duplicate: the existing item comes back and the
   new copy is dropped. Otherwise the original is kept under `<paths.ingest>/originals/` and recorded as `queued`.
2. `process`: probe it (media.py), settle its original time (the photo's EXIF or the container's creation time, else
   the time its source gave, else when it was added), convert it with ffmpeg, and register each converted file on
   Continuum (broker.py): a photo as a short clip and a video as H.264 on `<stream>.video_v1`, and audio (a video's
   sound too) as FLAC on `<stream>.audio_v1`. The converted files live under `<paths.ingest>/media/<stream>/<day>/`,
   since the broker registers them in place.

An item's stream is `ingest-<source>` unless the source names its own (each Wyze camera is `wyze-<camera>`).
"""

import datetime as dt
import hashlib
import re
import shutil
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from liberational import config as cfg
from liberational import logs
from liberational.ingest import media
from liberational.ingest.broker import BrokerRegistrar, Refused, Registrar
from liberational.ingest.store import Item, ItemGone, Store

LOG = logs.get("ingest")


class Purger(Protocol):
    def purge_item(self, item: Item) -> str | None: ...

    def purge_finished(self, purge_id: str) -> bool: ...


class AnalysisLedger(Protocol):
    def remove(self, item_id: int) -> None: ...


def slot(item_id: int) -> str:
    """An ingested item's slot in the segment store: its speech lines and voice embeddings are written there."""
    return f"ingest-{item_id}"


def forget_slot(item_id: int) -> None:
    """Forgets every row of the item's slot, every stream and all time (#1817): its lines and voice embeddings go with
    it, and the vector worker drops the embeddings from its index."""
    from continuum_py import forget_segments

    from liberational import segment_log

    forget_segments(segment_log.root(), 0, 2**63 - 1, cameras=[slot(item_id)])


def forget_video_rows(item: Item) -> None:
    """Forget this item's vision rows without removing overlapping imports on its source camera.

    Ingest passes the registered video chunk as the caption worker's frame id. Animal embeddings lack that
    column, but each video's fresh caption pipeline gives its animal tracks unique ids. Forget those embeddings
    before their groundings: if any step fails, a retry can still discover every remaining track from the rows.
    Persistent tombstones also reject a writer's later buffered appends. The item lock fences further analysis.
    """
    from continuum_py import forget_segments_where, read_segments

    from liberational import segment_log

    chunks = {str(out["chunk_id"]) for out in item.outputs if out["kind"] == "video"}
    if not chunks:
        return
    root = segment_log.root()
    cameras = [item.stream]
    # Native reads visit existing parts directly. Log.range with hour caching enabled would walk every hour
    # of this bound. Chunk identity, rather than a guessed media duration, limits which rows can be forgotten.
    groundings = read_segments(
        root,
        segment_log.ANIMAL_GROUNDINGS,
        segment_log.STREAMS[segment_log.ANIMAL_GROUNDINGS].schema,
        0,
        2**63 - 1,
        cameras,
    )
    tracks = {
        track for frame, track in zip(groundings["frame_id"], groundings["track_id"], strict=True) if frame in chunks
    }
    for track in sorted(tracks):
        forget_segments_where(
            root, "track_id", track, streams=[segment_log.ANIMAL_GROUNDING_EMBEDDINGS], cameras=cameras
        )
    streams = [segment_log.CAPTIONS, segment_log.CAMERA_OBJECTS, segment_log.ANIMAL_GROUNDINGS]
    for chunk in sorted(chunks):
        forget_segments_where(root, "frame_id", chunk, streams=streams, cameras=cameras)


class UnsafeIngestPath(ValueError):
    """Stored paths need operator repair; retrying cleanup alone cannot make them safe."""


def stream_name(text: str) -> str:
    """`text` as a stream name: lowercase letters, digits, and single dashes."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "unnamed"


def safe_suffix(name: str) -> str:
    """A file name's extension, if it's a plain one (`.jpg`), else none: the original is stored by hash."""
    suffix = Path(name).suffix.lower()
    return suffix if re.fullmatch(r"\.[a-z0-9]{1,8}", suffix) else ""


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class Incoming:
    """A file to ingest. The pipeline moves it into place unless `keep` (a file the caller still owns)."""

    path: Path
    source: str
    name: str
    key: str | None = None  # the source's own id for it, recorded so a poll never hands it over again
    taken_at: float | None = None  # when the source says it was made, used when the file itself doesn't say
    stream: str | None = None
    keep: bool = False


class Pipeline:
    """Adds and processes ingested files. `registrar` is the broker (or a fake in tests)."""

    def __init__(self, store: Store, root: Path, registrar: Registrar, now: Callable[[], float] = time.time) -> None:
        self.store, self.root, self.registrar, self.now = store, root, registrar, now

    def add(self, inc: Incoming) -> tuple[Item, bool]:
        """Records `inc` (or finds the item already holding its bytes): (item, whether it's new)."""
        sha = sha256_of(inc.path)
        item = self.store.by_hash(sha)
        if item is not None and self.store.removed(item.id):
            raise ValueError("This item is being deleted. Retry after storage cleanup finishes.")
        if item is None:
            dest = self.root / "originals" / sha[:2] / f"{sha}{safe_suffix(inc.name)}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if inc.keep:
                shutil.copyfile(inc.path, dest)
            else:
                shutil.move(inc.path, dest)
            stream = stream_name(inc.stream or f"ingest-{inc.source}")
            item, new = self.store.add(
                sha256=sha,
                source=inc.source,
                source_key=inc.key,
                name=inc.name,
                original=str(dest),
                stream=stream,
                added_ts=self.now(),
            )
            if new and inc.taken_at is not None:
                item = self.store.update(item.id, original_ts=inc.taken_at, time_from="source")
        else:
            new = False
        if not new and not inc.keep:
            inc.path.unlink(missing_ok=True)
        if inc.key is not None:
            self.store.mark_seen(inc.source, inc.key, item.id)
        return item, new

    def ingest(self, inc: Incoming) -> Item:
        """Adds `inc` and processes it when it's new; a duplicate comes back as the item it duplicates."""
        item, new = self.add(inc)
        return self.process(item.id) if new else item

    def process(self, item_id: int) -> Item:
        """Converts and registers one item, recording how it went. A failure is the item's status, never raised:
        anything unexpected too, or one bad item left `converting` would raise again first on every resume and stop
        `pack-ingest poll` for good (#3121). Only ItemGone (the item is gone, or was removed meanwhile) and a store
        failure, which every item would hit, are raised. A KeyError is a bug like any other: the item fails with it."""
        with self.store.item_lock(item_id):
            item = self.store.get(item_id)
            if item is None or self.store.removed(item_id):
                raise ItemGone(item_id)
            self.store.update(item_id, status="converting", detail="")
            try:
                return self._process(item)
            except (media.MediaError, Refused, OSError) as e:
                LOG.warning(f"item {item_id} ({item.name}) failed: {e}")
                return self.store.update(item_id, status="failed", detail=str(e))
            except ItemGone, sqlite3.Error:
                raise
            except Exception as e:
                LOG.exception(f"item {item_id} ({item.name}) failed unexpectedly")
                return self.store.update(item_id, status="failed", detail=f"{type(e).__name__}: {e}")

    def delete(self, item_id: int, ledger: AnalysisLedger, purger: Purger, *, wait_s: float = 12.0) -> dict[str, Any]:
        """Fence work, purge through the broker, and remove files only after every keeper acknowledges.

        The durable pending receipt makes a timeout/restart retryable. Ordinary item lists hide the item as
        soon as deletion is requested. Waiting for work and purge acknowledgments shares one bounded deadline.
        """
        deadline = time.monotonic() + wait_s
        self.store.request_delete(item_id)
        receipt = self.store.deletion(item_id)
        if receipt is None or not receipt["pending"]:
            return {"ok": True, "deleted": receipt is not None, "pending": False}
        while True:
            try:
                with self.store.item_lock(item_id, blocking=False):
                    return self._delete_locked(item_id, ledger, purger, deadline)
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    return {"ok": True, "deleted": True, "pending": True}
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    def _delete_locked(self, item_id: int, ledger: AnalysisLedger, purger: Purger, deadline: float) -> dict[str, Any]:
        from continuum_py import BrokerError

        item = self.store.get(item_id)
        if item is None:
            return {"ok": True, "deleted": True, "pending": False}
        paths = self._owned_paths(item)
        ledger.remove(item_id)
        forget_slot(item_id)
        forget_video_rows(item)
        receipt = self.store.deletion(item_id)
        purge_id = receipt.get("purge_id") if receipt else None
        if purge_id is None:
            purge_id = purger.purge_item(item)
            self.store.deletion_progress(item_id, purge_id=purge_id)
        while purge_id is not None:
            try:
                if purger.purge_finished(purge_id):
                    break
            except BrokerError as error:
                if not error.args or error.args[0] != "NotFound":
                    raise
                self.store.reset_purge(item_id)
                return {"ok": True, "deleted": True, "pending": True}
            if time.monotonic() >= deadline:
                self.store.deletion_progress(item_id, detail="Waiting for storage cleanup acknowledgments.")
                return {"ok": True, "deleted": True, "pending": True}
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        for path in paths:
            path.unlink(missing_ok=True)
        self.store.finish_delete(item_id)
        return {"ok": True, "deleted": True, "pending": False}

    def _owned_paths(self, item: Item) -> set[Path]:
        # Only paths owned by this pipeline: stored metadata must never turn deletion into arbitrary unlink.
        paths = {Path(item.original), *(Path(o["path"]) for o in item.outputs)}
        if item.original_ts is not None:
            day = dt.datetime.fromtimestamp(item.original_ts).strftime("%Y-%m-%d")
            base = self.root / "media" / item.stream / day / item.sha256[:16]
            paths.update((base.with_suffix(".mp4"), base.with_suffix(".flac")))
        for path in paths:
            if not path.resolve().is_relative_to(self.root.resolve()):
                raise UnsafeIngestPath("An ingest file is outside its storage directory. Manual cleanup is required.")
        return paths

    def _process(self, item: Item) -> Item:
        original = Path(item.original)
        found = media.probe(original)
        if found.creation_ts is not None:
            start, time_from = found.creation_ts, "exif" if found.kind == "photo" else "ffprobe"
        elif item.original_ts is not None and item.time_from == "source":
            start, time_from = item.original_ts, "source"
        else:
            start, time_from = item.added_ts, "upload"
        self.store.update(item.id, kind=found.kind, original_ts=start, time_from=time_from)
        day = dt.datetime.fromtimestamp(start).strftime("%Y-%m-%d")
        base = self.root / "media" / item.stream / day / item.sha256[:16]
        converted: list[dict[str, Any]] = []
        if found.kind == "photo":
            converted.append(
                {
                    "kind": "video",
                    "path": media.still_to_h264(original, base.with_suffix(".mp4")),
                    "fps": media.STILL_FPS,
                }
            )
        elif found.kind == "video":
            fps = found.fps or 0.0
            converted.append(
                {
                    "kind": "video",
                    "path": media.to_h264(original, base.with_suffix(".mp4"), found.video_codec),
                    "fps": fps,
                }
            )
        if found.kind == "audio" or found.has_audio:
            converted.append({"kind": "audio", "path": media.to_flac(original, base.with_suffix(".flac"))})
        self.store.update(item.id, status="registering")
        outputs: list[dict[str, Any]] = []
        refused: list[str] = []
        for c in converted:
            if self.store.removed(item.id):
                raise ItemGone(item.id)
            path: Path = c["path"]
            try:
                if c["kind"] == "video":
                    chunk = self.registrar.video(item.stream, path, start, float(c["fps"]))
                    stream = f"{item.stream}.video_v1"
                else:
                    chunk = self.registrar.audio(item.stream, path, start)
                    stream = f"{item.stream}.audio_v1"
            except Refused as e:
                refused.append(f"{c['kind']}: {e}")
                continue
            outputs.append({"kind": c["kind"], "stream": stream, "chunk_id": chunk, "path": str(path)})
            # Persist each successful registration before cancellation can end the next output.
            registered = {o["chunk_id"]: o for o in [*item.outputs, *outputs]}
            self.store.update(item.id, outputs=list(registered.values()))
        if self.store.removed(item.id):
            raise ItemGone(item.id)
        if not outputs:
            return self.store.update(item.id, status="failed", detail="; ".join(refused), outputs=item.outputs)
        detail = f"not registered: {'; '.join(refused)}" if refused else ""
        LOG.info(f"item {item.id} ({item.name}) registered on {', '.join(o['stream'] for o in outputs)}")
        registered = {o["chunk_id"]: o for o in [*item.outputs, *outputs]}
        return self.store.update(item.id, status="done", detail=detail, outputs=list(registered.values()))

    def resume(self) -> list[Item]:
        """Processes the items a restart cut off (still queued, converting, or registering). One removed since it
        was listed is skipped."""
        out = []
        for pending in self.store.pending():
            try:
                out.append(self.process(pending.id))
            except ItemGone:
                continue
        return out


def default(registrar: Registrar | None = None) -> Pipeline:
    """The pack's pipeline: its store in `paths.state`, its media in `paths.ingest`, registering on its broker."""
    store = Store(cfg.path("paths.state") / "ingest.sqlite3")
    return Pipeline(store, cfg.path("paths.ingest"), registrar or BrokerRegistrar())
