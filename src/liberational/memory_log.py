"""The memory log: the journal's interpretations on Continuum, the way `speech_log` and
`vision_log` hold speech and vision. Layer 3 payloads with parents, where v1 kept `interpretations.sqlite3` and
`summaries/`. B1 of the plan: the streams, their codecs, and the read API that `recall`, `history_qa`, the Review
summary, and the dashboard code against; the journal worker writes them, its day, week, and month pages included
(memory/journal_pages.py).

Streams (format Text, shape name = stream name, an inline JSON body of at most MAX_BODY; layer 3 except Ask's
human question roots at layer 1 and answer/state records at layer 2):

| Stream | Body | Span | Parents | Subjects |
|---|---|---|---|---|
| `journal.entry` | `text` (with `[S<id>]` speaker tokens, rendered by the reader), `level` (the window in seconds), `speakers`, `faces`, `faces_at`, `heard_at`, `line_speakers` (finest level only), `evidence` (selected atomic source records and attribution status), `supersedes` | the window | none at the finest level (segment source IDs are evidence references); a rollup's are the child entries | the people in it |
| `person.fact` | `fact`, `kind`, `person`, `same`, `supersedes` | one hour's say on a fact (memory/facts.py folds them) | the hour entry | the person |
| `journal.page` | `level` (`day`, `week`, or `month`), `period` (its first local day, `YYYY-MM-DD`), `md` (the page, with `[S<id>]` speaker tokens), `sections` (the model's prose fields), `status` (`quiet` for a period with nothing to write), `supersedes` | the local period | a day page's are the hour entries it read; a week or month page's, the day pages | the people in them |
| `review.summary` | `level`, `text`, `status`, `topics`, `source_hash`, `evidence` (the line’s selected sources) | the stretch | the entries it read | none |
| `ask.question` | question and request context | when asked | none | none |
| `ask.state` | answer manifest, status, safe citation references | when asked | the question | none |
| `ask.answer` | bounded answer chunks or citation markers | when asked | the question and canonical source inputs | none |

A correction is a new payload whose `supersedes` names the old one (DOMAIN-MODEL rule 6): readers keep the newest
and drop what it names, so nothing is rewritten. Forgetting is the broker's: a purge takes every payload derived from
what it removes (an entry with a purged utterance among its parents goes with it), a purge by entity takes a person's
facts, and a rebuild cannot cite a purged input, since the catalog refuses a payload naming an unknown parent. Times
in the read API are seconds, as in the other logs.
"""

import asyncio
import json
import pickle
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncGenerator, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import continuum_py
from continuum_py import Client, DomainLayer, Item, PayloadHeader, Producer, read

from liberational.continuum import from_ns, to_ns
from liberational.continuum_link import BrokerLink, until_not_busy
from liberational.domain.kernel import PersonId

if TYPE_CHECKING:
    from liberational.listen.revisions import Corrections

JOURNAL = "journal.entry"
FACTS = "person.fact"
PAGES = "journal.page"
REVIEW_SUMMARY = "review.summary"
ASK_QUESTION = "ask.question"
ASK_STATE = "ask.state"
ASK_ANSWER = "ask.answer"

LAYER: DomainLayer = "Layer3Interpretation"
NS = 1_000_000_000
#: A `journal.entry` whose window had nothing to write: a marker, so the catalog says the window is done.
QUIET = "quiet"
#: A `journal.entry` whose window the journal gave up on after its retries: a marker too, rewritten by `--day`.
FAILED = "failed"
# The most bytes an inline payload holds (the broker's MAX_INLINE_BYTES): an entry over it is split, never truncated.
MAX_BODY = 64 * 1024


@dataclass(frozen=True)
class StreamDef:
    """A memory stream: which body keys its payloads carry."""

    name: str
    keys: tuple[str, ...]
    layer: DomainLayer = LAYER


STREAMS = {
    d.name: d
    for d in (
        StreamDef(
            JOURNAL,
            (
                "text",
                "search",
                "level",
                "status",
                "speakers",
                "faces",
                "faces_at",
                "heard_at",
                "line_speakers",
                "evidence",
                "supersedes",
                "transcript_revision",
            ),
        ),
        StreamDef(FACTS, ("fact", "kind", "person", "same", "supersedes", "transcript_revision")),
        StreamDef(PAGES, ("level", "period", "md", "sections", "status", "supersedes", "transcript_revision")),
        # A Review summary line: one per journal entry it was written from (its parent, its window), with the
        # hash of the entry text it came from and its topic tags; status `rejected` when the model broke the rules.
        StreamDef(REVIEW_SUMMARY, ("level", "text", "status", "source_hash", "topics", "evidence")),
        StreamDef(
            ASK_QUESTION,
            ("question", "surface", "day", "scope", "mode", "call_id", "created_at", "revisions", "transcript_review"),
            "Layer1Raw",
        ),
        StreamDef(
            ASK_STATE,
            (
                "exchange",
                "version",
                "status",
                "started_at",
                "finished_at",
                "model",
                "chunks",
                "chunk_count",
                "citations",
                "answer_unverified",
                "revisions",
                "transcript_review",
            ),
            "Layer2Decision",
        ),
        StreamDef(ASK_ANSWER, ("exchange", "version", "kind", "index", "data"), "Layer2Decision"),
    )
}

#: What a decoded record carries besides its body.
META = ("ts_start", "ts_end", "id", "parents", "subjects", "producer", "commit_seq", "position")


async def register(client: Client, stream: str) -> int:
    """Registers a memory stream (or finds it, when it's registered already) and returns its id."""
    d = STREAMS[stream]
    registered = await until_not_busy(
        lambda: client.register_stream(
            d.name, kind="derived", format="Text", shape_name=d.name, layer=d.layer, slab_class=0
        )
    )
    return registered["stream_id"]


def encode(
    stream: str,
    stream_id: int,
    record: Mapping[str, Any],
    *,
    payload_id: str | None = None,
    producer: Producer | None = None,
    parents: Sequence[str] = (),
    subjects: Sequence[str] = (),
    span_ns: tuple[int, int] | None = None,
    seq: int = 0,
) -> tuple[bytes, bytes]:
    """A record as a payload: its header and its inline JSON body. `record` has `ts_start` and `ts_end` (seconds) and
    the stream's body keys; keys it lacks are left out, None values too, and any others are refused so nothing is
    silently dropped (the decoded record's META keys are allowed, and ignored). A body over MAX_BODY is refused: the
    writer splits its window instead. The header's `seq` is `seq`, else `seq_of` the id."""
    d = STREAMS[stream]
    extra = set(record) - set(d.keys) - set(META)
    if extra:
        msg = f"{stream} records don't carry {sorted(extra)}"
        raise ValueError(msg)
    if span_ns is not None:
        start_ns, end_ns = span_ns[0], max(span_ns)
    else:
        start_ns = to_ns(float(record["ts_start"]))
        end_ns = max(start_ns, to_ns(float(record.get("ts_end", record["ts_start"]))))
    payload_id = payload_id or str(uuid.uuid7())
    header: PayloadHeader = {
        "id": payload_id,
        "stream_id": stream_id,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "layer": d.layer,
        "format": "Text",
        "shape_name": d.name,
        "parents": list(parents),
        "subjects": list(subjects),
        "seq": seq or seq_of(payload_id),
    }
    if producer is not None:
        header["producer"] = producer
    body = body_bytes(stream, record)
    if len(body) > MAX_BODY:
        msg = f"a {stream} body is at most {MAX_BODY} bytes; this one is {len(body)}"
        raise ValueError(msg)
    return continuum_py.encode_header(header), body


def body_bytes(stream: str, record: Mapping[str, Any]) -> bytes:
    """A record's body exactly as `encode` writes it (the stream's keys, None values left out, ASCII-escaped JSON),
    so a writer fitting a record to MAX_BODY measures what the broker will hold."""
    d = STREAMS[stream]
    return json.dumps({k: record[k] for k in d.keys if record.get(k) is not None}, separators=(",", ":")).encode()


def seq_of(payload_id: str) -> int:
    """The header `seq` a payload gets from its id: the catalog keys a stream's payloads by start and seq, and
    interpretations of one window start at the same instant (a correction and what it supersedes, a day's two
    summaries), so each takes a seq from the low 32 bits of its id. The same id always gets the same seq, so a repeat
    is still the same header."""
    return int.from_bytes(uuid.UUID(payload_id).bytes[-4:], "big") or 1


def decode(item: Item) -> tuple[str, dict[str, Any]] | None:
    """A payload as (its stream, its record), or None for one that isn't a memory stream's or has no inline body. The
    record carries its body keys and META: its span in seconds, id, parents, subjects, producer, and commit_seq when
    the item has one."""
    locator = item["locator"]
    if locator is None or locator["kind"] != "Inline" or "bytes" not in locator:
        return None
    header = continuum_py.decode_header(item["header"])
    stream = header["shape_name"]
    if stream not in STREAMS:
        return None
    try:
        record: dict[str, Any] = json.loads(locator["bytes"])
    except ValueError:
        return None
    if not isinstance(record, dict):
        return None
    record.update(
        ts_start=from_ns(header["start_ns"]),
        ts_end=from_ns(header["end_ns"]),
        id=header["id"],
        parents=list(header.get("parents") or []),
        subjects=list(header.get("subjects") or []),
        producer=header.get("producer"),
    )
    if "commit_seq" in item:
        record["commit_seq"] = item["commit_seq"]
    return stream, record


async def records_of(client: Client, stream: str, item: Item) -> list[dict[str, Any]]:
    """The records an item on `stream` holds: one for an inline payload (`decode`), or a flushed chunk's records,
    each with its own span and id (and the chunk's parents, subjects, and producer, since a chunk keeps one header)."""
    got = decode(item)
    if got is not None:
        return [got[1]]
    if stream not in STREAMS:
        return []
    header = continuum_py.decode_header(item["header"])
    out = []
    for rec in await read.records(client, stream, item) or []:
        try:
            record = json.loads(rec.data)
        except ValueError:
            continue
        if not isinstance(record, dict):
            continue
        record.update(
            ts_start=from_ns(rec.start_ns),
            ts_end=from_ns(rec.end_ns),
            id=rec.payload_id,
            parents=list(header.get("parents") or []),
            subjects=list(header.get("subjects") or []),
            producer=header.get("producer"),
        )
        out.append(record)
    return out


async def read_range(
    client: Client, stream: str, start: float, end: float, *, subjects: Iterable[str] = ()
) -> list[dict[str, Any]]:
    """Every record on `stream` whose span overlaps [start, end) (seconds), about any of `subjects` when given, in
    time order."""
    wanted = list(subjects)
    result = (
        await client.query(stream, to_ns(start), to_ns(end), subjects=wanted)
        if wanted
        else await client.query(stream, to_ns(start), to_ns(end))
    )
    records = [r for it in result["items"] for r in await records_of(client, stream, it)]
    records.sort(key=lambda r: (r["ts_start"], r["id"]))
    return records


async def follow(
    client: Client,
    streams: Sequence[str],
    *,
    start: Literal["Live", "Time", "Saved"] = "Live",
    since: float = 0.0,
    durable_name: str | None = None,
) -> AsyncGenerator[tuple[str, dict[str, Any]]]:
    """(stream, record) as each is committed: from now, from `since`, or after a durable name's saved position. Each
    record has its `position`; under a durable name a record's position is acknowledged when the caller asks for the
    next one (at least once). Each record is read through `client`, so it can't be a Backpressure reader; a durable
    catch-up that can pass the SDK's 4,096-event subscription queue belongs on `MemoryLog.drain` instead."""
    sub = await client.subscribe(
        list(streams), start=start, start_ns=to_ns(since), durable_name=durable_name, pinned=True
    )
    async with sub:
        async for event in sub:
            if event["type"] != "Event" or "item" not in event:
                continue
            stream = continuum_py.decode_header(event["item"]["header"])["shape_name"]
            for record in await records_of(client, stream, event["item"]):
                record["position"] = event["position"]
                yield stream, record
            if durable_name is not None:
                sub.ack(event["position"])


def current(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """`records` less the ones a later record supersedes (its `supersedes` names them), in the order given."""
    records = list(records)
    gone = {r["supersedes"] for r in records if r.get("supersedes")}
    return [r for r in records if r["id"] not in gone]


def transcript_current(
    record: Mapping[str, Any], state: str | Path | None = None, *, review: Corrections | None = None
) -> bool:
    """Whether a derived record was built after the latest owner transcript correction in its source window.
    A caller checking many records passes one `revisions.review_snapshot` as `review`, read once for them all."""
    from liberational import config as cfg
    from liberational.listen import revisions

    start = float(record.get("ts_start") or 0.0)
    end = max(float(record.get("ts_end") or start), start + 1e-6)
    if review is None:
        review = revisions.review_snapshot(Path(state) if state is not None else cfg.path("paths.state"))
    return record.get("transcript_revision", 0) == review.revision(start, end)


def reviewed(records: Iterable[dict[str, Any]], state: str | Path | None = None) -> list[dict[str, Any]]:
    """Hide stale derived evidence immediately, including while its worker is rebuilding it."""
    from liberational import config as cfg
    from liberational.listen import revisions

    records = list(records)
    if not records:
        return records
    review = revisions.review_snapshot(Path(state) if state is not None else cfg.path("paths.state"))
    if review.newest == 0:
        return records
    out = []
    for record in records:
        start = float(record.get("ts_start") or 0.0)
        end = max(float(record.get("ts_end") or start), start + 1e-6)
        affected = review.affected(start, end)
        valid = transcript_current(record, review=review) if affected else record.get("transcript_revision", 0) == 0
        if valid:
            out.append(record)
    return out


def day_bounds(day: str) -> tuple[float, float]:
    """[start, end) of a local day (`YYYY-MM-DD`), epoch seconds."""
    import datetime as dt

    from liberational.memory.history_qa import local_zone

    d = dt.date.fromisoformat(str(day))
    tz = local_zone()
    a = dt.datetime.combine(d, dt.time(), tzinfo=tz)
    b = dt.datetime.combine(d + dt.timedelta(days=1), dt.time(), tzinfo=tz)
    return a.timestamp(), b.timestamp()


RANGE_CACHE_MAX = 12  # ranges `MemoryLog.range` keeps: a day of journal is a few thousand records, a few MB
RANGE_CACHE_SPAN_S = 7 * 86400.0  # only bounded ranges are kept; "everything until now" moves with every call


class MemoryLog:
    """The memory log for synchronous code: one broker client on a background event loop (`BrokerLink`), with the
    reads the plan names. `state` is the pack state directory whose registry resolves people (merges) in the person
    filters; None reads without one. Stream registrations are cached per actual client: reconnecting must register
    again before publishing, even when the persistent catalog returns the same stream ID."""

    def __init__(
        self,
        socket: Path | None = None,
        *,
        state: str | Path | None = None,
        client_name: str = "memory-log",
        timeout: float = 10.0,
    ) -> None:
        self.link = BrokerLink(socket, client_name=client_name, timeout=timeout)
        self.state = Path(state) if state is not None else None
        self._ids: dict[str, tuple[Client, int]] = {}
        self._marks: dict[tuple[str, ...], int] = {}
        self._lock = threading.Lock()
        # `range` answers kept until a commit overlaps them or a purge takes from their stream: Review's
        # summary read each day's whole journal every minute, decoding thousands of records on this link's loop.
        # Touched only on that loop; pickled, so every caller gets records of its own to change.
        self._cached: OrderedDict[tuple[str, float, float, tuple[str, ...]], bytes] = OrderedDict()
        self._cache_seq: dict[str, int] = {}  # per stream, the newest commit the cache has accounted for
        self._cache_epoch: dict[str, int] = {}  # per stream, moved by every change: a read made across one isn't kept
        self._purge_cursor: str | None = None
        self._purges_read = False
        self._cache_lock = asyncio.Lock()

    async def _stream_id(self, client: Client, stream: str) -> int:
        cached = self._ids.get(stream)
        if cached is not None and cached[0] is client:
            return cached[1]
        # Registration belongs to a broker session, even when its catalog stream ID survives a restart.
        stream_id = await register(client, stream)
        self._ids[stream] = (client, stream_id)
        return stream_id

    # Writing

    def append(
        self,
        stream: str,
        record: Mapping[str, Any],
        *,
        parents: Sequence[str] = (),
        subjects: Sequence[str] = (),
        producer: Producer | None = None,
        payload_id: str | None = None,
        span_ns: tuple[int, int] | None = None,
    ) -> str:
        """Commits a record on `stream` (registering the stream first, once) and returns its payload id once the
        broker has it. The same id with the same header again is kept once; a parent the broker doesn't have (purged,
        or never there) is refused."""
        payload_id = payload_id or str(uuid.uuid7())

        async def go(client: Client) -> str:
            header, body = encode(
                stream,
                await self._stream_id(client, stream),
                record,
                payload_id=payload_id,
                producer=producer,
                parents=parents,
                subjects=subjects,
                span_ns=span_ns,
            )
            await client.ingest_acked(header, inline=body)
            return payload_id

        return self.link.call(go)

    def append_bytes(
        self,
        stream: str,
        shape: str,
        data: bytes,
        *,
        span: tuple[float, float],
        parents: Sequence[str] = (),
        subjects: Sequence[str] = (),
        producer: Producer | None = None,
    ) -> str:
        """Commits a RawBytes payload (an embedding) on `stream`, registered with `shape` on first use, spanning
        `span` (seconds); returns its payload id."""
        payload_id = str(uuid.uuid7())

        async def go(client: Client) -> str:
            cached = self._ids.get(stream)
            if cached is None or cached[0] is not client:
                made = await client.register_stream(
                    stream, kind="derived", format="RawBytes", shape_name=shape, layer=LAYER, slab_class=0
                )
                stream_id = made["stream_id"]
                self._ids[stream] = (client, stream_id)
            else:
                stream_id = cached[1]
            header: PayloadHeader = {
                "id": payload_id,
                "stream_id": stream_id,
                "start_ns": to_ns(span[0]),
                "end_ns": max(to_ns(span[0]), to_ns(span[1])),
                "layer": LAYER,
                "format": "RawBytes",
                "shape_name": shape,
                "parents": list(parents),
                "subjects": list(subjects),
                "seq": seq_of(payload_id),
            }
            if producer is not None:
                header["producer"] = producer
            await client.ingest_acked(continuum_py.encode_header(header), inline=data)
            return payload_id

        return self.link.call(go)

    def purge(self, ids: Iterable[str], reason: str, detail: str = "") -> int:
        """Purges the payloads `ids` and everything derived from them (the broker's cascade through parents); returns
        how many payloads were removed. `detail` is kept for good and must name no person."""
        wanted = sorted(set(ids))
        if not wanted:
            return 0
        done = self.link.call(lambda c: c.purge(ids=wanted, reason=reason, detail=detail))
        return sum(n for _, _, n in done["removed"])

    def drain(
        self,
        stream: str,
        durable: str,
        handle: Callable[[dict[str, Any]], bool],
        *,
        limit: int,
        idle_s: float = 0.5,
    ) -> int:
        """Hands `handle` the records on `stream` after the durable name's saved position, one at a time on a worker
        thread (so `handle` may read and write this log), acknowledging each one `handle` returns True for; the first
        it refuses stops the drain unacknowledged, so it comes again. Stops after `limit` records handled or when no
        record comes within `idle_s`, however long the handling takes. The broker keeps the position: a pass needs no
        cursor file. Returns how many records were handled."""

        async def go(client: Client) -> int:
            await self._stream_id(client, stream)
            # A backlog past the SDK's subscription queue blocks that connection's socket reader, so the subscription
            # gets a connection of its own, and every request (the handler's, and record reads) stays on `client`.
            # That connection makes no requests, so it holds its backlog with backpressure instead of the
            # default policy, which closes a connection whose subscription queue overflows.
            async with await Client.connect(
                str(self.link.socket),
                client_name=f"{self.link.client_name}-drain",
                subscription_overflow="backpressure",
            ) as reader:
                sub = await reader.subscribe([stream], start="Saved", durable_name=durable, pinned=True)
                handled = 0
                async with sub:
                    refused = False
                    while handled < limit and not refused:
                        try:
                            event = await asyncio.wait_for(sub.__anext__(), idle_s)
                        except TimeoutError, StopAsyncIteration:
                            break
                        if event["type"] != "Event" or "item" not in event:
                            continue
                        for record in await records_of(client, stream, event["item"]):
                            record["position"] = event["position"]
                            if not await asyncio.to_thread(handle, record):
                                refused = True  # this record comes again: nothing from here on is acknowledged
                                break
                            handled += 1
                        if not refused:
                            sub.ack(event["position"])
                    await asyncio.sleep(0.05)  # the acknowledgements go out before the subscription ends
                return handled

        # The handler may call a model for minutes: the drain waits for as long as it takes, not the link's timeout.
        return self.link.call(go, timeout=None)

    # Reading

    def parents_in(self, stream: str, start: float, end: float) -> set[str]:
        """Every parent named by a payload on `stream` overlapping [start, end): which entries have an embedding, say.
        Empty for a stream the broker doesn't have."""

        async def go(client: Client) -> set[str]:
            names = {s["name"] for s in await client.list_streams()}
            if stream not in names:
                return set()
            result = await client.query(stream, to_ns(start), to_ns(end))
            out: set[str] = set()
            for item in result["items"]:
                out |= set(continuum_py.decode_header(item["header"]).get("parents") or [])
            return out

        return self.link.call(go)

    def range(self, stream: str, start: float, end: float, *, subjects: Iterable[str] = ()) -> list[dict[str, Any]]:
        """Every record on `stream` overlapping [start, end) (seconds), about any of `subjects` when given, in time
        order, superseded ones included. Facts invalidated by an owner transcript correction are excluded; raw
        journal reads remain available to the worker that rebuilds them."""
        wanted = list(subjects)
        key = (stream, float(start), float(end), tuple(sorted(set(wanted))))
        keep = end - start <= RANGE_CACHE_SPAN_S

        async def go(client: Client) -> list[dict[str, Any]]:
            await self._stream_id(client, stream)  # a stream nobody has written yet isn't there to query
            if not keep:
                return await read_range(client, stream, start, end, subjects=wanted)
            epoch = await self._cache_check(client, stream)
            if (hit := self._cached.get(key)) is not None:
                self._cached.move_to_end(key)
                return pickle.loads(hit)
            found = await read_range(client, stream, start, end, subjects=wanted)
            if self._cache_epoch.get(stream, 0) == epoch:  # nothing changed the stream while it was read
                self._cached[key] = pickle.dumps(found, protocol=pickle.HIGHEST_PROTOCOL)
                while len(self._cached) > RANGE_CACHE_MAX:
                    self._cached.popitem(last=False)
            return found

        found = self.link.call(go)
        return reviewed(found, self.state) if stream == FACTS else found

    async def _cache_check(self, client: Client, stream: str) -> int:
        """Bring the range cache up to date with `stream`: drop each kept range that a commit since the last check
        overlaps, and all of any stream's after a purge that may have taken from it. Returns the stream's epoch, which a read
        must still see when it finishes to be kept."""
        async with self._cache_lock:
            after = self._cache_seq.get(stream)
            newest = after or 0
            changed = False
            mine = [k for k in self._cached if k[0] == stream]
            cursor = after or 0
            while True:
                page = await client.query_by_seq(stream, after_seq=cursor)
                for item in page["items"]:
                    newest = max(newest, item["commit_seq"])
                    if after is None or not mine:
                        continue  # the first look only finds where the stream stands; nothing of it is kept yet
                    header = continuum_py.decode_header(item["header"])
                    lo, hi = from_ns(header["start_ns"]), from_ns(header["end_ns"])
                    for k in mine:
                        if lo <= k[2] and hi >= k[1]:
                            self._cached.pop(k, None)
                if not page["next_after"]:
                    break
                cursor = page["next_after"]
            self._cache_seq[stream] = newest
            if after is not None and newest != after:
                changed = True
            # The purge history is read once for every stream, so each purge since the last look drops the kept ranges
            # of every stream it may have taken from, not only this one's: a later read of another stream won't see
            # that purge again. Their epochs move too, so a read of theirs in flight isn't kept.
            touched = await self._purged(client)
            hit = {name for name, (_, sid) in self._ids.items() if sid in touched}
            for k in [k for k in self._cached if k[0] in hit]:
                self._cached.pop(k, None)
            for name in hit | ({stream} if changed else set()):
                self._cache_epoch[name] = self._cache_epoch.get(name, 0) + 1
            return self._cache_epoch.get(stream, 0)

    async def _purged(self, client: Client) -> set[int]:
        """The stream ids purges since the last call may have taken from (every registered stream's, for a purge by
        ids). The first call only finds where the purges stand."""
        touched: set[int] = set()
        cursor = self._purge_cursor
        while page := await client.list_purges(after=cursor, limit=256):
            if page[-1]["purge_id"] == cursor:
                break
            if self._purges_read:
                for purge in page:
                    # A purge by ids names no streams (its cascade can reach any); one by stream or range names its
                    # streams, which for the broker's own checkpoints, every few seconds, are never ours.
                    if ids := purge.get("stream_ids"):
                        touched.update(int(i) for i in ids)
                    else:
                        touched.update(sid for _, sid in self._ids.values())
            cursor = page[-1]["purge_id"]
        self._purge_cursor, self._purges_read = cursor, True
        return touched

    def _resolve(self, pid: str) -> str:
        """`pid` followed through merges, when the state's registry is there to ask."""
        from liberational.people import registry

        if self.state is None or not registry.log_exists(self.state):
            return pid
        return registry.at(self.state).resolve(PersonId(pid)) or pid

    def entries(
        self,
        start: float,
        end: float,
        level: int | None = None,
        person: str | None = None,
        *,
        include_quiet: bool = False,
    ) -> list[dict[str, Any]]:
        """The current journal entries overlapping [start, end): at `level` (seconds) when given, about `person` (a
        person id; merges followed) when given, in time order then finest level first. A quiet window's marker is
        left out unless `include_quiet` (a failed window's marker likewise)."""
        found = reviewed(current(self.range(JOURNAL, start, end)), self.state)
        if not include_quiet:
            found = [r for r in found if not r.get("status")]  # a marker (quiet or failed) is no entry
        if level is not None:
            found = [r for r in found if r.get("level") == level]
        if person is not None:
            who = self._resolve(person)
            found = [r for r in found if who in {self._resolve(s) for s in r.get("subjects") or []}]
        return sorted(found, key=lambda r: (r["ts_start"], r.get("level") or 0, r["id"]))

    def facts(self, person: str) -> list[dict[str, Any]]:
        """Every fact record about `person` (a person id), oldest first: one per hour behind each fact, folded into
        facts by memory/facts.py (`fold`)."""
        return self.range(FACTS, 0.0, time.time() + 1.0, subjects=[person])

    def page(self, level: str, period: str) -> dict[str, Any] | None:
        """The current `journal.page` at `level` (`day`, `week`, or `month`) for the period starting on the local day
        `period` (`YYYY-MM-DD`), or None (a quiet period's marker likewise)."""
        a, _ = day_bounds(period)
        found = [
            r
            for r in reviewed(current(self.range(PAGES, a, a + 1.0)), self.state)
            if r.get("level") == level and r.get("period") == period and not r.get("status")
        ]
        return max(found, key=lambda r: str(r["id"])) if found else None  # ids are UUIDv7: time ordered

    def mark(self, streams: Iterable[str] = (JOURNAL,)) -> int:
        """Where `streams` stand: the newest commit sequence on them, for a cache to be stamped with. The first call
        per stream set scans it once; later ones read only what's new."""
        streams = tuple(streams)

        async def go(client: Client) -> int:
            for stream in streams:
                await self._stream_id(client, stream)
            newest = self._marks.get(streams, 0)
            for stream in streams:
                after = self._marks.get(streams, 0)
                while True:
                    page = await client.query_by_seq(stream, after_seq=after)
                    newest = max([newest, *(item["commit_seq"] for item in page["items"])])
                    if not page["next_after"]:
                        break
                    after = page["next_after"]
            return newest

        with self._lock:
            self._marks[streams] = self.link.call(go)
            return self._marks[streams]

    def close(self) -> None:
        """Closes the connection and stops the background loop."""
        self.link.close()


_LOGS: dict[Path, MemoryLog] = {}
_LOGS_LOCK = threading.Lock()


def at(state: str | Path | None = None) -> MemoryLog:
    """The memory log of the broker that `state` (a pack's state directory) belongs to: `continuum/broker.sock` under
    it, or `continuum_link.socket_path()` for the pack's own (and by default). One per socket, shared, made on first
    use; nothing connects until a read."""
    from liberational import config as cfg
    from liberational.people import registry

    socket = registry._socket(state)  # the broker `state` belongs to, as the registry finds it
    with _LOGS_LOCK:
        log = _LOGS.get(socket)
        if log is None:
            log = _LOGS[socket] = MemoryLog(socket, state=state if state is not None else cfg.path("paths.state"))
        return log


def close_all() -> None:
    """Closes every shared log (tests, between brokers)."""
    with _LOGS_LOCK:
        for log in _LOGS.values():
            log.close()
        _LOGS.clear()


__all__ = [
    "ASK_ANSWER",
    "ASK_QUESTION",
    "ASK_STATE",
    "FACTS",
    "FAILED",
    "JOURNAL",
    "MAX_BODY",
    "NS",
    "PAGES",
    "QUIET",
    "REVIEW_SUMMARY",
    "STREAMS",
    "MemoryLog",
    "at",
    "body_bytes",
    "close_all",
    "current",
    "day_bounds",
    "decode",
    "encode",
    "follow",
    "read_range",
    "records_of",
    "register",
    "seq_of",
]
