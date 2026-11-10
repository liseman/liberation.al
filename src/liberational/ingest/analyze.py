"""Ingested media through the same analysis as the pack's own cameras and microphones (`pack-ingest-analyze`).

The live workers follow broker rings (see/caption_worker.py, listen/worker.py), so a recording
the ingest pipeline registers on `<stream>.video_v1` / `<stream>.audio_v1` was stored but never looked at. This worker
takes each ingested item once it's `done`, decodes its converted files, and runs them through the very same pieces:

- video and photos: frames one a second (a photo: its one frame) through `CaptionPipeline` (captions and objects,
  committed by `CaptionWorker`'s own code). Faces in ingested items aren't analyzed while faces move to see-actions'
  tracked heads; the worker says so when it loads its models;
- audio, and a video's sound: v1's VAD, then the listen worker's `Transcriber` (speech gate, ASR, ReDimNet2 B3), its voice
  matcher, and an utterance on `speech.utterance` in the segment store, under the item's own slot (`ingest-<item id>`).

Every record is timed at the item's original time (pipeline.py settles it) plus its offset in the file, and names the
item's registered chunk as its parent, so it sits beside what the cameras and mics recorded then. The camera and
device names are the item's stream (`ingest-upload`, `wyze-porch`, ...). What's been analyzed is kept in
`<paths.state>/ingest-analyzed.json`. A clean restart skips ids whose pass finished. A pass that recorded an error
is tried again after a delay (30 s, then 2 min), and the third failure marks the item failed so the Ingest tab can
show why. Payload ids are a pure function of the item and the frame, so that retry is the same broker record, kept
once. Video
frames are the keyframes nearest each second, not every decoded frame. The models stay unloaded until a done item
is actually waiting, speech yields to the voice turn before it runs, and a soundtrack is read in short blocks instead of one buffer.
"""

import asyncio
import fcntl
import json
import subprocess
import threading
import time
from collections.abc import AsyncGenerator, Callable, Generator, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from liberational import logs, model_runtime, speech_log, voice_turn
from liberational.continuum import keyed_id
from liberational.ingest.pipeline import slot as item_slot

log = logs.get("ingest-analyze")
NS = 1_000_000_000
SR = 16_000
FRAME_EVERY_S = 1.0  # one frame a second through the captions' change gate
CAPTION_EVERY_S = 2.0  # the caption pipeline's change gate decides which of these the VLM sees
AUDIO_BLOCK_S = 0.1
MAX_ATTEMPTS = 3
RETRY_DELAYS_S = (30.0, 120.0)  # after the first recorded failure, then after the second
DETAIL_LIMIT = 500
LOCK_POLL_S = 0.2
VIDEO_KILL_S = 7200.0  # one long file's keyframe decode; a stuck ffmpeg is still killed


@dataclass
class Done:
    """What analyzing one item gave."""

    frames: int = 0
    captions: int = 0
    lines: int = 0
    errors: list[str] = field(default_factory=list)
    cancelled: bool = False  # the item was removed mid-pass: write nothing more, and do not record the ledger

    def public(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "captions": self.captions,
            "lines": self.lines,
            "errors": self.errors,
            "at": time.time(),
        }


def choose_keyframes(times: Sequence[float], every_s: float) -> list[int]:
    """Indexes of the keyframes to keep: the one nearest each `every_s` mark, each keyframe at most once.

    A mark is `0, every_s, 2 every_s, ...`. A keyframe belongs to the nearest mark, and that mark keeps the closest
    of its keyframes. Sparse keyframes (one every few seconds) are all kept. Keyframes denser than the marks are
    thinned to about one a mark.
    """
    if every_s <= 0:
        raise ValueError("every_s must be positive")
    best: dict[int, tuple[float, int]] = {}
    for index, raw in enumerate(times):
        t = float(raw)
        if t < 0:
            continue
        mark = int(t / every_s + 0.5)
        dist = abs(t - mark * every_s)
        current = best.get(mark)
        if current is None or dist < current[0] or (dist == current[0] and index < current[1]):
            best[mark] = (dist, index)
    return [best[mark][1] for mark in sorted(best)]


def _ffprobe(path: Path, entries: str) -> str:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        entries,
        "-of",
        "csv=p=0",
        str(path),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout, proc.stderr)
    return proc.stdout


def keyframe_times(path: Path) -> list[float]:
    """Keyframe packet times, in seconds. Packets are listed, not decoded."""
    times: list[float] = []
    for line in _ffprobe(path, "packet=pts_time,flags").splitlines():
        pts, _sep, flags = line.partition(",")
        if not pts or pts == "N/A" or "K" not in flags:
            continue
        times.append(float(pts))
    return times


def _video_size(path: Path) -> tuple[int, int]:
    """Coded size, swapped when the display matrix turns the picture 90 or 270 degrees.

    ffmpeg autorotates raw frames unless told not to, and ffprobe's width and height stay the coded size. A phone
    video that was copied without re-encoding keeps that matrix, so the upright frame is what we reshape.
    """
    lines = [
        line for line in _ffprobe(path, "stream=width,height:stream_side_data=rotation").splitlines() if line.strip()
    ]
    if not lines:
        raise RuntimeError(f"no video stream in {path.name}")
    parts = lines[0].split(",")
    width, height = int(parts[0]), int(parts[1])
    rotation = 0
    if len(parts) > 2 and parts[2] not in ("", "N/A"):
        rotation = round(float(parts[2])) % 360
    if rotation in (90, 270):
        return height, width
    return width, height


def video_frames(path: Path, every_s: float) -> Iterator[tuple[np.ndarray, float]]:
    """BGR frames nearest each `every_s` mark, from keyframes only, each with its offset in seconds.

    ffmpeg is told `-skip_frame nokey`, so the frames between keyframes are not decoded. Of the keyframes, only the
    one nearest each mark is yielded. One frame is in memory at a time.
    """
    times = keyframe_times(path)
    chosen = choose_keyframes(times, every_s)
    if not chosen:
        raise RuntimeError(f"no video keyframes in {path.name}")
    width, height = _video_size(path)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-skip_frame",
        "nokey",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-an",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.stdout is None or proc.stderr is None:
        raise RuntimeError("ffmpeg pipe was not created")
    err: list[bytes] = []

    def drain() -> None:
        assert proc.stderr is not None
        err.append(proc.stderr.read())

    drainer = threading.Thread(target=drain, daemon=True)
    drainer.start()
    timer = threading.Timer(VIDEO_KILL_S, proc.kill)
    timer.start()
    frame_n = width * height * 3
    want = set(chosen)
    last = chosen[-1]
    pending = b""
    index = 0
    try:
        while index <= last:
            while len(pending) < frame_n:
                chunk = proc.stdout.read(frame_n - len(pending))
                if not chunk:
                    break
                pending += chunk
            if len(pending) < frame_n:
                break
            raw, pending = pending[:frame_n], pending[frame_n:]
            if index in want:
                yield np.frombuffer(raw, np.uint8).reshape(height, width, 3).copy(), times[index]
            index += 1
        if index <= last:
            code = proc.wait(timeout=30)
            drainer.join(timeout=5)
            raise subprocess.CalledProcessError(
                code or 1, proc.args, stderr=b"".join(err) or b"ffmpeg ended before the last keyframe"
            )
    finally:
        timer.cancel()
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        drainer.join(timeout=5)


def audio_blocks(path: Path) -> Iterator[np.ndarray]:
    """A file's sound as 16 kHz mono float32, one `AUDIO_BLOCK_S` block at a time (ffmpeg).

    The whole soundtrack is never one array: a long video stays a few kilobytes of PCM at a time.
    """
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.stdout is None or proc.stderr is None:
        raise RuntimeError("ffmpeg pipe was not created")
    err: list[bytes] = []

    def drain() -> None:
        assert proc.stderr is not None
        err.append(proc.stderr.read())

    drainer = threading.Thread(target=drain, daemon=True)
    drainer.start()
    timer = threading.Timer(600, proc.kill)
    timer.start()
    width = 4
    want = int(SR * AUDIO_BLOCK_S) * width
    pending = b""
    try:
        while True:
            chunk = proc.stdout.read(want)
            if not chunk:
                break
            pending += chunk
            while len(pending) >= want:
                yield np.frombuffer(pending[:want], np.float32).copy()
                pending = pending[want:]
        if pending:
            n = len(pending) - (len(pending) % width)
            if n:
                yield np.frombuffer(pending[:n], np.float32).copy()
        code = proc.wait(timeout=30)
        drainer.join(timeout=5)
        if code != 0:
            raise subprocess.CalledProcessError(code, proc.args, stderr=b"".join(err))
    finally:
        timer.cancel()
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        drainer.join(timeout=5)


class Speech:
    """Utterances from a file: v1's VAD, the listen worker's transcriber and voice matcher, `speech.utterance`."""

    def __init__(
        self,
        client: Any,
        transcriber: Any,
        matcher: Any,
        vad: Callable[[], Any],
        newest_purge: Callable[[str | None], str | None] | None = None,
        segments: Path | None = None,
    ) -> None:
        self.client, self.transcriber, self.matcher, self.new_vad = client, transcriber, matcher, vad
        self.segments = segments  # the segment store (None: paths.state/segments)
        self.newest_purge = newest_purge
        self._purge: str | None = None
        self._purge_known = False

    async def _notice_purge(self) -> bool:
        """Drop the matcher's recent embeddings when the purge mark changes since the last check.

        The first mark is only remembered. A later one that differs is a purge (a person forgotten, or an ingest
        chunk deleted), and `recent` would otherwise keep matching those embeddings.
        """
        if self.matcher is None:
            return True
        try:
            if self.newest_purge is not None:
                current = self.newest_purge(self._purge)
            elif self.client is not None and callable(getattr(self.client, "list_purges", None)):
                current = self._purge
                while page := await self.client.list_purges(after=current, limit=256):
                    current = page[-1]["purge_id"]
            else:
                return True
        except Exception:
            log.warning("speech purge mark unavailable", exc_info=True)
            return False
        if self._purge_known and current != self._purge:
            forget = getattr(self.matcher, "forget", None)
            if callable(forget):
                forget()
        self._purge = current
        self._purge_known = True
        return True

    async def file(
        self,
        blocks: Iterator[np.ndarray],
        device: str,
        audio_stream: str,
        chunk: str,
        start: float,
        *,
        slot: str,
        removed: Callable[[], bool] | None = None,
    ) -> int:
        """Writes the file's utterances under `slot` (the item's own, so the item's lines are its slot's) and returns
        how many. Their ids come from the span's place in the file, so a second pass writes the same ids, and readers
        keep the first row of each."""
        writer = speech_log.Writer(slot, self.segments)
        try:
            return await self._file(blocks, device, audio_stream, chunk, start, writer, removed)
        finally:
            writer.close()
            if (close := getattr(self.matcher, "close", None)) is not None:  # its writer of this slot, if it opened one
                close(writer.slot)

    async def _file(
        self,
        blocks: Iterator[np.ndarray],
        device: str,
        audio_stream: str,
        chunk: str,
        start: float,
        writer: Any,
        removed: Callable[[], bool] | None,
    ) -> int:

        if not await self._notice_purge():
            raise RuntimeError("speech purge mark unavailable; retry")
        vad, n = self.new_vad(), 0
        spans: list[tuple[int, int, Any]] = []
        loop = asyncio.get_running_loop()
        for block in blocks:
            if removed is not None and removed():
                return n
            spans.extend(vad.feed(np.asarray(block, np.float32)))
        spans.extend(vad.feed(np.zeros(SR, np.float32)))  # a second of silence closes speech running to the end
        for index, (s0, s1, clip) in enumerate(spans):
            if removed is not None and removed():
                return n
            # On a worker thread: wait_quiet sleeps, and the event loop still has to notice SIGTERM.
            await loop.run_in_executor(None, voice_turn.wait_quiet)
            if removed is not None and removed():
                return n
            heard = await self.transcriber(np.asarray(clip, np.float32), device)
            if removed is not None and removed():
                return n
            if heard is None or not heard.text.strip():
                continue
            start_ns, end_ns = int((start + s0 / SR) * NS), int((start + s1 / SR) * NS)
            match = None
            if heard.embedding is not None:
                # A different item can be deleted during a long file. Do not keep matching its cached voice
                # until the next file begins; observe the purge before each embedding lookup.
                if not await self._notice_purge():
                    raise RuntimeError("speech purge mark unavailable; retry")
                if removed is not None and removed():
                    return n
                from liberational.listen.voices import word_count

                match = await self.matcher.match(heard.embedding, start_ns, end_ns, words=word_count(heard.text))
            if removed is not None and removed():
                return n
            line: dict[str, Any] = {
                "ts_start": start_ns / NS,
                "ts_end": end_ns / NS,
                "device": device,
                "text": heard.text,
                "lat_ms": heard.lat_ms,
                "audio": {"stream": audio_stream, "start_ns": start_ns, "end_ns": end_ns},
            }
            if match is not None:
                line |= {"speaker": match.speaker, "speaker_id": match.speaker_id, "speaker_sim": round(match.sim, 3)}
            # The span's place in this file, not the text: a different transcript must not mint a second line.
            payload_id = keyed_id(start + s0 / SR, "ingest", "speech.utterance", chunk, index)
            writer.write(speech_log.UTTERANCES, {**line, "id": payload_id}, span_ns=(start_ns, end_ns))
            writer.sync()
            if heard.embedding is not None:
                voice_id = keyed_id(start + s0 / SR, "ingest", "voice.embedding", chunk, index)
                self.matcher.learn(
                    heard.embedding,
                    match,
                    payload_id,
                    start_ns,
                    end_ns,
                    payload_id=voice_id,
                    slot=writer.slot,  # the item's own, with its lines
                )
            n += 1
        return n


class Analyzer:
    """Analyzes ingested items. `captions` is a `CaptionWorker` (its streams already registered) with a factory for its
    pipeline by camera name; `speech` makes utterances. Either can be None (that analysis is skipped)."""

    def __init__(
        self,
        *,
        captions: tuple[Any, Callable[[str], Any]] | None = None,
        speech: Speech | None = None,
        frames: Callable[[Path, float], Iterator[tuple[np.ndarray, float]]] = video_frames,
        audio: Callable[[Path], Iterator[np.ndarray]] = audio_blocks,
    ) -> None:
        self.captions, self.speech = captions, speech
        self.frames, self.audio = frames, audio
        self._removed: Callable[[], bool] | None = None

    def _halt(self) -> bool:
        return bool(self._removed is not None and self._removed())

    async def prepare_item(self, item: Any) -> bool:
        """Establish speech's purge cursor before taking the item lock. File and embedding checks still re-read
        the incremental tail inside the lock, so a purge during lock acquisition cannot leave stale voice matches.
        A failed preflight leaves the item pending without falling back to a full-history read under the lock."""
        if self.speech is None or not any(out["kind"] == "audio" for out in item.outputs):
            return True
        return await self.speech._notice_purge()

    async def item(self, item: Any, *, removed: Callable[[], bool] | None = None) -> Done:
        done = Done()
        self._removed = removed
        try:
            start = float(item.original_ts or item.added_ts)
            for out in item.outputs:
                if self._halt():
                    done.cancelled = True
                    return done
                path, chunk = Path(out["path"]), out["chunk_id"]
                try:
                    if out["kind"] == "video":
                        await self._video(item.stream, path, chunk, start, done, photo=item.kind == "photo")
                    elif out["kind"] == "audio" and self.speech is not None:
                        blocks: Iterator[np.ndarray] | None = None
                        try:
                            blocks = self.audio(path)
                            done.lines += await self.speech.file(
                                blocks,
                                item.stream,
                                out["stream"],
                                chunk,
                                start,
                                slot=item_slot(item.id),
                                removed=self._halt,
                            )
                        finally:
                            if blocks is not None and (close := getattr(blocks, "close", None)) is not None:
                                close()
                except Exception as e:  # one output failing never loses the rest
                    log.warning("item %s %s: %s", item.id, out["kind"], e, exc_info=True)
                    done.errors.append(f"{out['kind']}: {type(e).__name__}: {e}")
            if self._halt():
                done.cancelled = True
            return done
        finally:
            self._removed = None

    async def _video(self, cam: str, path: Path, chunk: str, start: float, done: Done, *, photo: bool) -> None:
        loop = asyncio.get_running_loop()
        cw, cap_pipe = None, None
        mark: dict[str, dict[str, int]] = {}
        if self.captions:
            cw = self.captions[0]
            cap_pipe = await loop.run_in_executor(cw._model, self.captions[1], cam)
            if self._halt():
                return
            # Where this camera's segment rows stand before the pass: one that doesn't finish is rewound here, so
            # its retry writes them again rather than twice. A crash mid-pass can still repeat them.
            from liberational import segment_log

            streams = (
                segment_log.CAPTIONS,
                segment_log.CAMERA_OBJECTS,
                segment_log.ANIMAL_GROUNDINGS,
                segment_log.ANIMAL_GROUNDING_EMBEDDINGS,
            )
            mark = {s: {cam: cw.segments.position(s, cam)} for s in streams}
        completed = False
        try:
            last_caption: str | None = None
            last_cap = -float("inf")
            for seq, (img, offset) in enumerate(self.frames(path, FRAME_EVERY_S)):
                if self._halt():
                    return
                ts = start + offset
                ns = int(ts * NS)
                done.frames += 1
                if cw is not None and cap_pipe is not None and ts - last_cap >= CAPTION_EVERY_S:
                    if self._halt():
                        return
                    last_cap = ts
                    before = cw.committed
                    out = await loop.run_in_executor(cw._model, cap_pipe.feed, img, ts, seq)
                    if self._halt():
                        return
                    last_caption = cw._handle(
                        chunk,
                        ns,
                        out,
                        last_caption,
                        payload_for=lambda stream, ts=ts, seq=seq: keyed_id(ts, "ingest", stream, chunk, seq),
                    )
                    done.captions += cw.committed - before
                if photo:
                    break
            completed = True
        finally:
            if cw is not None:
                if completed and not self._halt():
                    cw.segments.sync()
                else:
                    cw.segments.rewind(mark)


def _state(row: dict[str, Any] | None) -> str | None:
    """`done` for a row written before retries existed (it has no state), so those items are not analyzed again."""
    if row is None:
        return None
    return str(row.get("state") or "done")


class Ledger:
    """Which items have been analyzed, and what it gave: `<paths.state>/ingest-analyzed.json`.

    A row's `state` is `done` (finished; a missing state means the same), `retry` (try again at `next_at`), or
    `failed` (the attempt cap was hit). `remove` drops one row. `record` reloads the file under a lock first, so a
    row another process removed is not written back.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock_path = path.with_name(path.name + ".lock")
        self.done: dict[str, Any] = {}

    def _load(self) -> None:
        self.done = json.loads(self.path.read_text()) if self.path.exists() else {}

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.done, indent=1))
        tmp.replace(self.path)

    @contextmanager
    def _locked(self) -> Generator[None]:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                self._load()
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def row(self, item_id: int) -> dict[str, Any] | None:
        with self._locked():
            raw = self.done.get(str(item_id))
            return dict(raw) if isinstance(raw, dict) else None

    def attempts(self, item_id: int) -> int:
        """How many failed passes are already recorded. A `failed` row counts as none: the user sent the item back."""
        row = self.row(item_id)
        if row is None or _state(row) == "failed":
            return 0
        return int(row.get("attempts") or 0)

    def remove(self, item_id: int) -> None:
        """Drop one item's row. A missing row is fine, and calling this again is fine."""
        if not self.path.exists():
            return
        with self._locked():
            if str(item_id) not in self.done:
                return
            del self.done[str(item_id)]
            self._write()

    def record(
        self,
        item_id: int,
        result: dict[str, Any],
        *,
        state: str = "done",
        attempts: int = 1,
        next_at: float | None = None,
    ) -> None:
        row = dict(result)
        row["state"] = state
        row["attempts"] = attempts
        if next_at is not None:
            row["next_at"] = next_at
        with self._locked():
            self.done[str(item_id)] = row
            self._write()


def _due(item: Any, ledger: Ledger, now: float) -> bool:
    if item.status != "done" or not item.outputs:
        return False
    state = _state(ledger.row(item.id))
    if state is None:
        return True
    if state == "done":
        return False
    if state == "retry":
        row = ledger.row(item.id) or {}
        return now >= float(row.get("next_at") or 0)
    # Still `done` in the store: the user ran Retry, which puts a failed item back through conversion
    # and leaves this row in place. The sweep's in-lock recheck passes the store's current status,
    # because a page snapshot stays `done` after another sweep has already marked the item failed.
    return state == "failed"


def _current_item(store: Any, item: Any) -> Any:
    """The store's row when it can be reread.

    A page from `finished` stays `done` after another sweep marks the item failed. Passing that
    snapshot to `_due` looks like a user Retry. A store without `get` keeps the snapshot.
    """
    get = getattr(store, "get", None)
    if not callable(get):
        return item
    fresh = get(item.id)
    return item if fresh is None else fresh


def _removed(store: Any, item_id: int) -> bool:
    fn = getattr(store, "removed", None)
    if not callable(fn):
        return False
    return bool(fn(item_id))


def _finish(store: Any, ledger: Ledger, item: Any, result: Done, now: float) -> None:
    if not result.errors:
        ledger.record(item.id, result.public(), state="done", attempts=ledger.attempts(item.id) + 1)
        return
    attempts = ledger.attempts(item.id) + 1
    if attempts >= MAX_ATTEMPTS:
        detail = ("analysis failed: " + "; ".join(result.errors))[:DETAIL_LIMIT]
        update = getattr(store, "update", None)
        if callable(update):
            update(item.id, status="failed", detail=detail)
        ledger.record(item.id, result.public(), state="failed", attempts=attempts)
        return
    delay = RETRY_DELAYS_S[attempts - 1]
    ledger.record(item.id, result.public(), state="retry", attempts=attempts, next_at=now + delay)


def awaiting(store: Any, ledger: Ledger, limit: int = 500, now: Callable[[], float] | None = None) -> bool:
    """Whether any done item with outputs is due. Pages, so a long history still counts.

    A retry whose delay has not elapsed is not due, so an idle process stays unloaded.
    """
    clock = time.time if now is None else now
    after = 0
    while True:
        page = store.finished(after_id=after, limit=limit)
        if not page:
            return False
        if any(_due(item, ledger, clock()) for item in page):
            return True
        if len(page) < limit:
            return False
        after = page[-1].id


class _Analyzer(Protocol):
    """What a sweep runs: one pass over an item, stopping when `removed` says the item is gone."""

    async def item(self, item: Any, *, removed: Callable[[], bool] | None = None) -> Done: ...


@asynccontextmanager
async def _under_lock(store: Any, item_id: int) -> AsyncGenerator[None]:
    """Hold `store.item_lock` when the store has one. A busy lock is polled so the event loop is not blocked.

    `blocking=False` raises BlockingIOError while another process holds the item (conversion or deletion).
    """
    lock = getattr(store, "item_lock", None)
    if not callable(lock):
        yield
        return
    while True:
        try:
            held = lock(item_id, blocking=False)
            held.__enter__()
        except BlockingIOError:
            await asyncio.sleep(LOCK_POLL_S)
            continue
        try:
            yield
        finally:
            held.__exit__(None, None, None)
        return


async def sweep(
    analyzer: _Analyzer, store: Any, ledger: Ledger, limit: int = 500, now: Callable[[], float] | None = None
) -> int:
    """Analyzes every due `done` item, oldest first; returns how many.

    Pages through every done item. `store.recent` is only the newest rows, so a backlog past that window
    would otherwise sit unanalyzed forever. Optional preflight runs outside the item lock; the item lock, when
    the store has one, still covers the pass and ledger write. Due and removal are both rechecked after the
    lock is acquired. The due recheck reads the store's current status, so a sweep that waited does not skip
    a backoff, repeat a pass that just finished, or treat an item another sweep just failed as a user Retry.
    """
    clock = time.time if now is None else now
    n = 0
    after = 0
    while True:
        page = store.finished(after_id=after, limit=limit)
        if not page:
            return n
        after = page[-1].id
        for item in page:
            current = clock()
            if not _due(item, ledger, current):
                continue
            if _removed(store, item.id):
                continue
            prepare = getattr(analyzer, "prepare_item", None)
            if callable(prepare) and (await prepare(item)) is False:
                continue
            async with _under_lock(store, item.id):
                if _removed(store, item.id) or not _due(_current_item(store, item), ledger, clock()):
                    continue
                result = await analyzer.item(item, removed=lambda item_id=item.id: _removed(store, item_id))
                if result.cancelled or _removed(store, item.id):
                    log.info("item %s removed during analysis; not recording", item.id)
                    continue
                # The delay starts when the pass returns. A pass longer than the delay is not due again immediately.
                _finish(store, ledger, item, result, clock())
            log.info(
                "item %s (%s): %d frames, %d captions, %d lines%s",
                item.id,
                item.name,
                result.frames,
                result.captions,
                result.lines,
                f", errors {result.errors}" if result.errors else "",
            )
            n += 1
        if len(page) < limit:
            return n


@model_runtime.reporting("ingest-analyze")
def main(argv: Sequence[str] | None = None) -> None:
    """`pack-ingest-analyze`: analyzes ingested media like the pack's own cameras and microphones (run it with the
    broker, the vector worker and pack-vlm). `--once` analyzes what's waiting and exits."""
    import argparse
    import signal

    from liberational import config as cfg
    from liberational import continuum_link
    from liberational.ingest import pipeline as ingest_pipeline

    parser = argparse.ArgumentParser(prog="pack-ingest-analyze", description=main.__doc__)
    parser.add_argument("--once", action="store_true", help="analyze what's waiting, then exit")
    parser.add_argument("--every", type=float, default=15.0, help="seconds between sweeps")
    args = parser.parse_args(argv)
    state = Path(cfg.path("paths.state"))
    store = ingest_pipeline.default().store
    ledger = Ledger(state / "ingest-analyzed.json")

    async def open_analyzer(client: Any, search: Any) -> tuple[Analyzer, Any]:
        """Connects the caption and speech models. Called only once a done item is waiting.

        `search` stays open for the life of the matcher (the caller holds its context).
        """
        from liberational import segment_log
        from liberational.listen import gate, voices
        from liberational.listen import worker as listen
        from liberational.people import registry, transcript_view
        from liberational.see import caption_worker
        from liberational.see import vlm as v1vlm
        from liberational.see.captions import CaptionPipeline
        from liberational.speech.asr import ASR

        log.info("faces in ingested items aren't analyzed while faces move to see-actions' tracked heads (#1738)")
        backend = cfg.get("vision.backend")
        backend = backend if backend != "auto" else "mlx-server" if cfg.has_mlx() else "ollama"
        vlm = v1vlm.make_vlm(backend)
        cw = caption_worker.CaptionWorker(
            client,
            [],
            lambda cam: CaptionPipeline(cam.name, vlm),
            model=backend,
            segments=segment_log.Writers(segment_log.root(state)),
        )
        transcriber = listen.Transcriber(
            ASR, gate=gate.speech_gate(), embed=listen.SpeakerEmbedding(Path(cfg.model_path("speaker_embedding")))
        )
        try:
            matcher = voices.VoiceMatcher(
                voices.Voiceprints(
                    state / "voiceprints.json", lambda: max(registry.at(state).voiceprints(), default=0)
                ),
                voices.RegistryPeople(state),
                search,
                clones=lambda: transcript_view.clone_only_ids(state),
                segments=segment_log.root(state),
                slot="ingest",
                references=voices.ReferenceGallery(state),
            )
            speech = Speech(client, transcriber, matcher, gate.VAD, segments=segment_log.root(state))
            return (
                Analyzer(captions=(cw, lambda cam: CaptionPipeline(cam, vlm)), speech=speech),
                transcriber,
            )
        except Exception:
            transcriber.close()
            raise

    async def run() -> None:
        from continuum_py.search import Searcher

        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stop.set)
        # Idle, and on by default: do not connect or load models until an item is actually waiting.
        while not awaiting(store, ledger):
            if args.once or stop.is_set():
                log.info("analyzed 0 item(s)")
                return
            try:
                await asyncio.wait_for(stop.wait(), args.every)
            except TimeoutError:
                continue
            log.info("analyzed 0 item(s)")
            return
        log.info("a done item is waiting; loading caption and speech models")
        transcriber: Any = None
        # Closed through `connected`, and run through `run_service`: the broker going away is one WARN and an exit the
        # service manager restarts on, not a traceback.
        async with continuum_link.connected(client_name="ingest-analyze") as client:
            try:
                async with Searcher(client) as search:
                    analyzer, transcriber = await open_analyzer(client, search)
                    while True:
                        n = await sweep(analyzer, store, ledger)
                        if args.once or stop.is_set():
                            log.info("analyzed %d item(s)", n)
                            break
                        try:
                            await asyncio.wait_for(stop.wait(), args.every)
                            break
                        except TimeoutError:
                            pass
            finally:
                if transcriber is not None:
                    transcriber.close()

    continuum_link.run_service(run())
