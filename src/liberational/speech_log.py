"""The speech log: utterances, assistant replies, and the energy gate's state in the segment store,
read back as transcript lines (dicts).

Each utterance is a row on `speech.utterance`, each
reply a row on `assistant.reply`, and the listen worker's gate state rows on `speech.activity`, all in the segment
store under `paths.state/segments` (their columns and codecs are in `segment_log`). A stream has one writer
per slot (the store's camera): listen writes each microphone's lines under its device name, the voice service its
replies under its client name, and an importer under a slot of its own, since a slot's writer lock is exclusive.
Speech rows are kept until they're forgotten; there's no retention window.

Readers get each line as a dict:

- `ts_start` and `ts_end` in seconds, and `id` (the row's id, a UUIDv7).
- Utterances: `device`, `speaker`, `speaker_id`, `speaker_sim`, `text`, `lat_ms`, `password` when it was one, and
  `audio`, which names the audio stream and span to read back (`{"stream", "start_ns", "end_ns"}`).
- Replies: `speaker` (the assistant's name), `assistant: True`, `text`, `reply_to`, `route`, and `error`.

A writer that runs again over the same input (a re-analyzed ingest item, a repeated import) writes the same ids, so
readers keep the first row of each id. Rows have no broker parents: the broker refuses a parent it doesn't hold, so
nothing on the broker (an embedding) names a line as its parent either.

`SpeechLog` is the synchronous face for threaded code (the dashboard, the journal). `since` and `mark` number
the lines this process has read in the order it read them (a process-local sequence, from 1): the first call reads
every line once, and later ones only what was written since. Its broker client remains for what's still on the
broker: microphone audio, purges, and the voice turn.
"""

import asyncio
import copy
import logging
import os
import threading
import time
from collections.abc import Generator, Iterable, Mapping, Sequence
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Literal

import continuum_py
from continuum_py import Client

from liberational import config as cfg
from liberational import continuum_link, segment_log

log = logging.getLogger(__name__)

UTTERANCES = segment_log.UTTERANCES
REPLIES = segment_log.REPLIES
ACTIVITY = segment_log.SPEECH_ACTIVITY  # operational energy-gate state, not verified speech
LINES = (UTTERANCES, REPLIES)

NS = 1_000_000_000

# The keys a line may carry per stream, besides its times and id: a writer is refused any other, so nothing is
# silently dropped.
KEYS = {
    UTTERANCES: frozenset({"device", "speaker", "speaker_id", "speaker_sim", "text", "lat_ms", "password", "audio"}),
    REPLIES: frozenset({"speaker", "text", "reply_to", "route", "error", "assistant"}),
    ACTIVITY: frozenset({"device", "stream", "in_speech", "level_dbfs", "threshold_dbfs"}),
}


def to_ns(seconds: float) -> int:
    """Seconds (a float) as UTC nanoseconds. `round(t * 1e9)` is off by up to about 256 ns at today's epoch
    (1.8e18 needs more than a float's 53 bits), so the whole seconds and the fraction are converted apart."""
    whole = int(seconds // 1)
    return whole * NS + round((seconds - whole) * NS)


def socket_path() -> Path:
    """The broker's socket: `CONTINUUM_SOCKET`, or `continuum/broker.sock` under the pack's state directory."""
    env = os.environ.get("CONTINUUM_SOCKET")
    return Path(env) if env else cfg.path("paths.state") / "continuum" / "broker.sock"


def _first_of_each(lines: Iterable[dict[str, Any]], seen: set[str] | None = None) -> list[dict[str, Any]]:
    """`lines` without a second row of an id already seen (a writer that ran again over the same input)."""
    seen = set() if seen is None else seen
    out = []
    for line in lines:
        if line["id"] not in seen:
            seen.add(line["id"])
            out.append(line)
    return out


def lines(
    start: float,
    end: float,
    streams: Iterable[str] = (UTTERANCES,),
    *,
    base: Path | None = None,
    include_rejected: bool = False,
) -> list[dict[str, Any]]:
    """Every line on `streams` whose span overlaps [start, end) (seconds), merged in time order."""
    root = base if base is not None else segment_log.root()
    store = segment_log.Log(root)
    found = [line for stream in streams for line in store.range(stream, start, end)]
    found.sort(key=lambda line: (line["ts_start"], line.get("id", "")))
    found = found if not found or "id" not in found[0] else _first_of_each(found)
    from liberational.listen import revisions

    return found if include_rejected else revisions.filter_lines(root.parent, found)


async def read_range(
    start: float, end: float, streams: Iterable[str] = (UTTERANCES,), *, base: Path | None = None
) -> list[dict[str, Any]]:
    """`lines` for async code, off the event loop."""
    return await asyncio.to_thread(lines, start, end, tuple(streams), base=base)


class Writer:
    """Writes one slot's speech lines (`slot` is made a valid store name: a device name may have spaces). Each line
    is checked against its stream's keys and given an id when it has none; `write` returns the id."""

    def __init__(self, slot: str, base: Path | None = None) -> None:
        self.slot = segment_log.slot(slot)
        self._writers = segment_log.Writers(base if base is not None else segment_log.root())

    def write(self, stream: str, line: Mapping[str, Any], *, span_ns: tuple[int, int] | None = None) -> str:
        """Appends `line` on `stream`. `span_ns` gives its span in nanoseconds exactly, for a writer that has them
        (the times in `line` are then only for show)."""
        extra = set(line) - KEYS[stream] - {"ts_start", "ts_end", "id"}
        if extra:
            msg = f"{stream} lines don't carry {sorted(extra)}"
            raise ValueError(msg)
        record = dict(line)
        record["id"] = str(record.get("id") or segment_log.new_id())
        if span_ns is not None:
            record["span_ns"] = span_ns
        self._writers.append(stream, self.slot, [record])
        return record["id"]

    def sync(self) -> None:
        """Makes every line written so far durable (appends are otherwise synced every few seconds)."""
        self._writers.sync()

    def close(self) -> None:
        self._writers.close()


def activity(now: float, fresh_s: float, *, base: Path | None = None) -> dict[str, bool]:
    """The newest gate state per device stamped in [now - fresh_s, now]; a device with none there is left out (its
    state is unknown, never quiet).

    The Live tab's meters ask this four times a second per open page, and reading the window from the store read the
    hour's whole segment each time (about 20 ms of CPU a call on the Linux machine). So the store is read once per base,
    then followed: each call reads only the rows written since the last, and keeps the newest state per device."""
    root = base if base is not None else segment_log.root()
    with _activity_lock:
        known = _activity.get(root)
        if known is None:
            follower = Follower((ACTIVITY,), base=root)  # from now: what's there already comes from the read below
            known = _activity[root] = (follower, {})
            for r in lines(now - fresh_s, now + 1e-9, (ACTIVITY,), base=root):
                _note_activity(known[1], r)
        follower, newest = known
        for r, _cursor in follower.poll(0.0):
            _note_activity(newest, r)
        return {device: state for device, (at, state) in newest.items() if now - fresh_s <= at <= now + 1e-9}


_activity: dict[Path, tuple[Follower, dict[str, tuple[float, bool]]]] = {}  # per base: its follower, newest states
_activity_lock = threading.Lock()


def _note_activity(newest: dict[str, tuple[float, bool]], r: Mapping[str, Any]) -> None:
    """Keep r's state if it's its device's newest (an older state written later never replaces a newer one)."""
    device = r.get("device")
    if device and (device not in newest or r["ts_start"] >= newest[device][0]):
        newest[device] = r["ts_start"], bool(r["in_speech"])


Cursor = dict[str, dict[str, int]]  # where a follower is: the next row_seq per stream and slot


def tail(streams: Iterable[str], base: Path) -> Cursor:
    """Where each slot of `streams` ends now, without reading them: a cursor to follow from here on. A slot that
    appears later starts at its first row."""
    from continuum_py import segment_tail

    return {s: segment_tail(base, s) for s in streams}


class Follower:
    """Follows `streams` for a durable consumer, from a saved `cursor` (or, with none, from now: what's there already
    is passed over). `poll` gives each new line with the cursor just past it, so the consumer saves that cursor once
    it has handled the line and a restart takes up after the last line handled."""

    def __init__(
        self, streams: Sequence[str] = (UTTERANCES,), cursor: Cursor | None = None, *, base: Path | None = None
    ) -> None:
        from continuum_py import SegmentFollowers

        self.streams = tuple(streams)
        root = base if base is not None else segment_log.root()
        self.base = root
        schemas = {s: segment_log.STREAMS[s].schema for s in self.streams}
        start = cursor if cursor is not None else tail(self.streams, root)
        self._follow = SegmentFollowers(root, schemas, start)
        self.cursor: Cursor = {s: dict(c) for s, c in start.items()}

    def poll(self, timeout_s: float = 0.0) -> list[tuple[dict[str, Any], Cursor]]:
        """The lines written since the last poll (waiting up to `timeout_s` for some), each with the cursor past it."""
        out = []
        for stream, got in self._follow.poll(timeout_s):
            spec = segment_log.STREAMS[stream]
            for i, slot in enumerate(got["camera"]):
                line = spec.to_record({name: got[name][i] for name, _ in spec.schema}, slot)
                self.cursor.setdefault(stream, {})[slot] = int(got["row_seq"][i]) + 1
                out.append((line, {s: dict(c) for s, c in self.cursor.items()}))
        from liberational.listen import revisions

        return [(line, cursor) for line, cursor in out if revisions.filter_lines(self.base.parent, [line])]


class History:
    """Every line of `streams` this process has read, numbered from 1 in the order it read them, kept up to date by
    following the store. `poll` reads what was written since the last call."""

    def __init__(self, base: Path, streams: Sequence[str] = LINES) -> None:
        self.base, self.streams = base, tuple(streams)
        self.lines: list[dict[str, Any]] = []
        self.newest: dict[str, int] = dict.fromkeys(self.streams, 0)  # the newest line's number per stream
        self._seen: set[str] = set()
        self._follow: Any = None
        self.lock = threading.Lock()

    def poll(self) -> None:
        """Reads the lines written since the last poll. The caller holds `lock`."""
        if self._follow is None:
            from continuum_py import SegmentFollowers

            schemas = {s: segment_log.STREAMS[s].schema for s in self.streams}
            self._follow = SegmentFollowers(self.base, schemas)
        for stream, got in self._follow.poll(0.0):
            for line in _first_of_each(segment_log.records(stream, got), self._seen):
                line["seq"] = len(self.lines) + 1
                line["_stream"] = stream  # which stream it is, for since(); not a line key
                self.lines.append(line)
                self.newest[stream] = line["seq"]

    def since(self, after: int, streams: Iterable[str]) -> tuple[list[dict[str, Any]], int]:
        wanted = set(streams)
        with self.lock:
            self.poll()
            got = [line for line in self.lines[max(0, after) :] if line["_stream"] in wanted]
            return [_public(line) for line in got], len(self.lines)

    def mark(self, streams: Iterable[str]) -> int:
        with self.lock:
            self.poll()
            return max((self.newest.get(s, 0) for s in streams), default=0)


def _public(line: dict[str, Any]) -> dict[str, Any]:
    """A copy of a history line as callers get it: with its `seq`, without the stream it was filed under."""
    out = copy.deepcopy(line)
    del out["_stream"]
    return out


_shared: SpeechLog | None = None
_shared_lock = threading.Lock()


def shared() -> SpeechLog:
    """The process's one `SpeechLog`, made on first use: the journal, the timeline, history questions, and the
    summaries all read through it, so a process holds one history and one broker connection however many of them it
    runs. Tests put a fake in its place (tests/conftest.py)."""
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = SpeechLog(client_name="speech-readers")
        return _shared


_retained_lock = threading.Lock()
_retained_log: SpeechLog | None = None
_retained_mark: tuple[object, ...] | None = None
_retained_rows: list[dict[str, Any]] = []


def retained() -> list[dict[str, Any]]:
    """Every utterance kept, in time order, cached until a line is written or a purge may have taken one. People and
    clone totals need complete history. Raw rows only: each request applies a fresh identity View, so names and
    corrections never wait for new speech. Copies keep a caller's projection or annotations from changing another
    request's input."""
    global _retained_log, _retained_mark, _retained_rows
    log = shared()
    with _retained_lock:
        from liberational.listen import revisions

        state = getattr(log, "segments", segment_log.root()).parent
        mark = (*log.mark(), revisions.review_revision(state))
        if log is not _retained_log or mark != _retained_mark:
            rows = log.all_lines()
            _retained_log, _retained_mark, _retained_rows = log, mark, rows
        return copy.deepcopy(_retained_rows)


WORKER_STREAM_PREFIX = "worker."  # the broker's own worker state, such as worker.checkpoint.<hash>.parts


class SpeechLog:
    """The speech log for synchronous code. Lines are read from the segment store (`segments`, by default
    `paths.state/segments`); a broker client on a background event loop serves what's still on the broker. The
    connection is made on first use and remade after the broker restarts, and its calls block until the broker
    answers (or `timeout` seconds pass)."""

    def __init__(
        self,
        socket: Path | None = None,
        *,
        client_name: str = "speech-log",
        timeout: float = 10.0,
        segments: Path | None = None,
    ) -> None:
        self.socket = socket or socket_path()
        self.client_name = client_name
        self.timeout = timeout
        self.segments = segments if segments is not None else segment_log.root()
        self._history = History(self.segments)
        self._writer: Writer | None = None
        self._writer_lock = threading.Lock()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="speech-log", daemon=True)
        self._thread.start()
        self._client: Client | None = None
        self._lock = asyncio.Lock()
        self._purge: str | None = None  # mark()'s newest purge
        # newest_purge's scan: the last purge read, the newest one that can take data, and stream names by id (ids
        # are never reused, so the names are read again only for an id not known yet).
        self._purge_lock = asyncio.Lock()
        self._purges_read: str | None = None
        self._purges_newest: str | None = None
        self._removals_newest: str | None = None
        self._voice_newest: str | None = None  # the newest that can take speech or audio (newest_voice_purge)
        self._stream_names: dict[int, str] = {}

    def _run[T](self, coro: Any) -> T:
        future: Future[T] = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(self.timeout)

    async def _connected(self) -> Client:
        async with self._lock:
            if self._client is None:
                self._client = await Client.connect(str(self.socket), client_name=self.client_name)
            return self._client

    async def _with_client[T](self, call: Any) -> T:
        client = await self._connected()
        try:
            result = await call(client)
        except continuum_py.ConnectionClosedError:
            # The broker restarted: connect again once and retry.
            async with self._lock:
                self._client = None
            result = await call(await self._connected())
        continuum_link.OUTAGE.back()
        return result

    def range(self, start: float, end: float, streams: Iterable[str] = (UTTERANCES,)) -> list[dict[str, Any]]:
        """Every line on `streams` overlapping [start, end) (seconds), in time order."""
        return lines(start, end, streams, base=self.segments)

    def recent(
        self, seconds: float, streams: Iterable[str] = (UTTERANCES,), *, now: float | None = None
    ) -> list[dict[str, Any]]:
        """The lines of the last `seconds`."""
        end = now if now is not None else time.time()
        return self.range(end - seconds, end + 1.0, streams)

    def follow(
        self, streams: Sequence[str] = (UTTERANCES,), *, since: float | None = None, idle_s: float | None = None
    ) -> Generator[dict[str, Any] | None]:
        """Lines as they're written, from now (or those ending at `since`, seconds, or later, catching up first),
        blocking between them. With `idle_s`, a None comes after that long without a line, so the caller can do other
        work between them. Stop by breaking out of the loop."""
        from continuum_py import SegmentFollowers

        schemas = {s: segment_log.STREAMS[s].schema for s in streams}
        follower = SegmentFollowers(self.segments, schemas, tail(streams, self.segments) if since is None else None)
        seen: set[str] = set()
        wait = idle_s if idle_s is not None else 1.0
        while True:
            batches = follower.poll(wait)
            fresh = [
                line
                for stream, got in batches
                for line in _first_of_each(segment_log.records(stream, got), seen)
                if since is None or line.get("ts_end", line["ts_start"]) >= since
            ]
            from liberational.listen import revisions

            fresh = revisions.filter_lines(self.segments.parent, fresh)
            if not fresh:
                if not batches and idle_s is not None:
                    yield None
                continue
            yield from fresh

    def since(self, after_seq: int, streams: Iterable[str] = (UTTERANCES,)) -> tuple[list[dict[str, Any]], int]:
        """The lines this process read after number `after_seq` (0 for all), in the order it read them, each with its
        `seq`, and the number to pass next time. For a reader that keeps what it read and takes only what's new."""
        from liberational.listen import revisions

        rows, position = self._history.since(after_seq, streams)
        return revisions.filter_lines(self.segments.parent, rows), position

    def all_lines(self, streams: Iterable[str] = (UTTERANCES,)) -> list[dict[str, Any]]:
        """Every line kept on `streams`, in time order (from this process's history: only what's new is read)."""
        got, _ = self._history.since(0, streams)
        for line in got:
            del line["seq"]
        got.sort(key=lambda line: (line["ts_start"], line["id"]))
        from liberational.listen import revisions

        return revisions.filter_lines(self.segments.parent, got)

    def mark(self, streams: Iterable[str] = (UTTERANCES,)) -> tuple[int, str | None]:
        """Where `streams` stand: the newest line's number and the newest purge. It changes when a line is written or
        a purge may have taken one, so a cache stamped with it knows when to rebuild."""
        newest = self._history.mark(tuple(streams))
        self._purge = self.newest_purge(self._purge)
        return newest, self._purge

    def newest_purge(self, after: str | None = None) -> str | None:
        """The id of the newest purge after `after` that can take data, else `after`. A reader that keeps lines
        compares this between reads: a new purge may have taken some of them. A purge only of the broker's worker
        state (WORKER_STREAM_PREFIX: checkpoints, trimmed every few seconds) takes nothing a reader keeps, and is left
        out, so it doesn't make every cache keyed on this start over. Each call reads only purges it hasn't
        read before."""

        async def go(client: Client) -> str | None:
            async with self._purge_lock:
                cursor, newest, removal = self._purges_read, self._purges_newest, self._removals_newest
                voice = self._voice_newest
                while page := await client.list_purges(after=cursor, limit=256):
                    if page[-1]["purge_id"] == cursor:
                        break
                    if any(i not in self._stream_names for p in page for i in p.get("stream_ids") or ()):
                        self._stream_names.update(
                            {int(r["stream_id"]): str(r["name"]) for r in await client.list_streams()}
                        )
                    for purge in page:
                        if not self._worker_only(purge):
                            newest = purge["purge_id"]
                            if purge.get("kind") != "AgeOut":
                                removal = purge["purge_id"]
                            if not self._video_only(purge):
                                voice = purge["purge_id"]
                    cursor = page[-1]["purge_id"]
                self._purges_read, self._purges_newest, self._removals_newest = cursor, newest, removal
                self._voice_newest = voice
                return newest

        newest = self._run(self._with_client(go))
        return newest if newest is not None and (after is None or newest > after) else after

    def stamp(self, start: float, end: float) -> tuple[tuple[str, int, int], ...]:
        """What the store holds of the lines in [start, end), cheaply (`segment_log.Log.stamp` over the speech
        streams): it moves when a line of the window is written late (an ingested recording) or purged."""
        return segment_log.Log(self.segments).stamp(start, end, (UTTERANCES, REPLIES))

    def newest_removal(self) -> str | None:
        """As newest_purge, leaving out age-outs too: the newest purge that deleted data on purpose (a privacy purge,
        a forget), which can change what an ended day says. Age-outs only thin old media on a schedule."""
        self.newest_purge()
        return self._removals_newest

    def newest_voice_purge(self) -> str | None:
        """As newest_purge, leaving out purges only of camera video (`<camera>.video`: the nightly 30-day window, an
        approved video age-out): the newest purge that can take a line, its audio, or a voice. What voice readers key
        their kept answers on, so a video-only purge doesn't make them work everything out again."""
        self.newest_purge()
        return self._voice_newest

    def _video_only(self, purge: dict[str, Any]) -> bool:
        """A purge that names streams, all of them a camera's video. An id with no name counts as anything."""
        ids = purge.get("stream_ids") or ()
        return bool(ids) and all(self._stream_names.get(i, "").endswith(".video") for i in ids)

    def _worker_only(self, purge: dict[str, Any]) -> bool:
        """A purge that names streams, all of them the broker's worker state. An id with no name counts as data."""
        ids = purge.get("stream_ids") or ()
        return bool(ids) and all(self._stream_names.get(i, "").startswith(WORKER_STREAM_PREFIX) for i in ids)

    def audio(self, stream: str, start_ns: int, end_ns: int, channel: int = 0) -> tuple[int, Any]:
        """A microphone stream's audio over [start_ns, end_ns): (rate, float32 samples), silence where nothing was
        kept (room_audio.read)."""
        from liberational import room_audio

        return self._run(self._with_client(lambda c: room_audio.read(c, stream, start_ns, end_ns, channel)))

    def audio_spans(self, stream: str, start_ns: int, end_ns: int) -> list[tuple[int, int]]:
        """The kept stretches of a microphone stream over [start_ns, end_ns) (room_audio.spans)."""
        from liberational import room_audio

        return self._run(self._with_client(lambda c: room_audio.spans(c, stream, start_ns, end_ns)))

    def audio_streams(self) -> list[str]:
        """Every microphone stream on the broker (a PCM ring named `<device>.audio`), fresh or not, sorted."""

        async def go(client: Client) -> list[str]:
            return sorted(
                s["name"]
                for s in await client.list_streams()
                if s["name"].endswith(".audio") and str(s["shape_name"]).startswith("PCM_")
            )

        return self._run(self._with_client(go))

    def mic_freshness(self, window_s: float = 30.0) -> dict[str, int | None]:
        """Each microphone stream (a PCM ring named `<device>.audio`) and the end of its newest audio in the last
        `window_s` seconds (UTC ns; now while the ring holds records newer than its last batch), else None."""

        async def go(client: Client) -> dict[str, int | None]:
            out: dict[str, int | None] = {}
            for s in await client.list_streams():
                if not (s["name"].endswith(".audio") and str(s["shape_name"]).startswith("PCM_")):
                    continue
                now = time.time_ns()
                r = await client.query(s["name"], now - int(window_s * 1e9), now + 1_000_000_000, all=False)
                newest = max((continuum_py.decode_header(i["header"])["end_ns"] for i in r["items"]), default=None)
                out[s["name"]] = now if r.get("tail") else newest
            return out

        return self._run(self._with_client(go))

    def append(self, stream: str, line: dict[str, Any]) -> str:
        """Writes a line on `stream` under this log's client name as its slot (so one process writes a slot), synced
        before it returns; returns its id."""
        with self._writer_lock:
            if self._writer is None:
                self._writer = Writer(self.client_name, self.segments)
            line_id = self._writer.write(stream, line)
            self._writer.sync()
            return line_id

    def set_voice_turn(self, state: Literal["Idle", "Listening", "Thinking", "Speaking"]) -> None:
        """Sets the voice turn (broker system state, docs/CONTINUUM.md decision 11): the scheduler preempts GPU work
        while it's Listening or Speaking, and every session hears the change."""

        async def go(client: Client) -> None:
            client.set_system_state(state)

        self._run(self._with_client(go))

    def close(self) -> None:
        """Closes the connection and stops the background loop."""

        async def shut() -> None:
            if self._client is not None:
                await self._client.close()

        try:
            self._run(shut())
        except Exception:
            pass
        with self._writer_lock:
            if self._writer is not None:
                self._writer.close()
                self._writer = None
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
