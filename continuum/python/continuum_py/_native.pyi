"""Type stub for continuum_py._native, the compiled part of continuum_py (crates/continuum-py)."""

from collections.abc import AsyncIterator, Awaitable, Buffer
from os import PathLike
from types import TracebackType
from typing import Any, Literal, Self

import numpy as np

from continuum_py.schema import (
    BrokerEvent,
    Capability,
    ComputeCost,
    ComputeDomain,
    CoverageDone,
    DomainLayer,
    ForgottenPage,
    H264AccessUnit,
    H264Gap,
    IngestRejected,
    Item,
    PayloadFormat,
    PayloadHeader,
    PriorityBand,
    PromiseAccepted,
    PromisePart,
    QueryResult,
    QuotedResult,
    RingLease,
    ShmLocation,
    StartAt,
    StreamLease,
    StreamRegistered,
    SubscriptionEvent,
    VoiceTurnState,
    WorkerRegistered,
)

PAGE_ALIGN: int

class ContinuumError(Exception): ...
class StaleError(ContinuumError): ...
class NotReadyError(ContinuumError): ...
class FullError(ContinuumError): ...

def encode_header(header: PayloadHeader) -> bytes:
    """A header's FlatBuffer bytes (root PayloadHeader). Raises KeyError, TypeError, or ValueError for a bad header."""

def decode_ingest_rejected(envelope: Buffer) -> IngestRejected:
    """An IngestRejected event's envelope (from Client.next_event) as a dict. Raises ValueError for anything else."""

def _shutdown() -> None:
    """Called at interpreter exit: from now on no awaitable completes (see the package's docs)."""

def decode_event(envelope: Buffer) -> BrokerEvent:
    """A pushed message's envelope (from Client.next_event) as a dict with "type" and its fields."""

def encode_vector_query(
    model: str,
    version: str,
    vector: list[float],
    k: int,
    *,
    during: tuple[int, int] | None = None,
    subjects: list[str] = ...,
    labelled: bool | None = None,
) -> bytes:
    """A vector.query payload's bytes (continuum-search); k is 1 to 40."""

def encode_text_query(
    text: str,
    k: int,
    *,
    during: tuple[int, int] | None = None,
    subjects: list[str] = ...,
    labelled: bool | None = None,
) -> bytes:
    """A text.query payload's bytes (continuum-search); k is 1 to 40."""

def decode_hits(data: Buffer) -> list[Any]:
    """A vector.search or text.search payload's hits as dicts (continuum_py.search.Hit's keys)."""

def encode_hits(hits: list[Any]) -> bytes:
    """A result payload's bytes from hit dicts: for a stand-in worker in tests."""

def stage_id(name: str) -> int:
    """A stage's id: the 32-bit FNV-1a hash of its name."""

def read_chunk_records(
    uri: str, start_ns: int, end_ns: int
) -> list[tuple[str, int, int, bytes, tuple[int, int, int] | None]] | None:
    """A chunk's live records overlapping [start_ns, end_ns), from a Chunk locator's local file: (payload_id,
    start_ns, end_ns, bytes) each, in the stream's native form. None when the file is gone. ValueError for a uri that
    isn't an absolute local path (resolve a relative locator against the data root first, as `read.records` does)."""

def v1_gop_units(path: str | PathLike[str], start_ns: int, gop: int) -> list[dict[str, Any]]:
    """GOP `gop` of the v1 recording at `path` (starting at `start_ns`), in decode order: dicts of "bytes" (Annex
    B), "pts_ns", "end_ns", "dts_ns", and "key". Raises ContinuumError for a file or GOP that can't be read."""

def decode_flac(flac: Buffer) -> tuple[int, int, bytes]:
    """FLAC bytes (a v1 archive file) as (rate, channels, samples): interleaved little-endian f32, each integer over
    2^(bits - 1) as capture's are. Raises ValueError for bytes that aren't FLAC."""

class LeaseDenied(ContinuumError):
    """No lease now; args[0] is how many milliseconds to wait before asking again."""

class Subscription:
    """A live subscription: async for event in it, ack(position) for a durable one, unsubscribe() from any task.
    async with unsubscribes on the way out."""

    @property
    def id(self) -> int: ...
    @property
    def start_position(self) -> int: ...
    @property
    def live_at_start(self) -> bool: ...
    @property
    def rings(self) -> list[RingLease]: ...
    def next(self) -> Awaitable[SubscriptionEvent | None]: ...
    def take(self, timeout_s: float, max: int = 64) -> list[SubscriptionEvent] | None:
        """Block this thread (GIL released) for the next event and every other one already waiting, up to `max`: []
        after `timeout_s` with none, None once it ends. For a thread of its own, never the event loop's."""
    def __aiter__(self) -> Self: ...
    def __anext__(self) -> Awaitable[SubscriptionEvent]: ...
    def ack(self, position: int) -> None: ...
    def unsubscribe(self) -> Awaitable[None]: ...
    def __aenter__(self) -> Awaitable[Self]: ...
    def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> Awaitable[None]: ...

class H264Window:
    """A camera's recent native H.264 (open_h264_window): async for item in it, history first and then live."""

    def next(self) -> Awaitable[H264AccessUnit | H264Gap | None]: ...
    def __aiter__(self) -> Self: ...
    def __anext__(self) -> Awaitable[H264AccessUnit | H264Gap]: ...

def open_h264_window(
    client: Client,
    live: Client,
    stream: str,
    from_ns: int,
    window_s: float = 300.0,
    now_ns: int | None = None,
    data_root: str | PathLike[str] | None = None,
) -> Awaitable[H264Window]:
    """Opens `stream`'s recent window at `from_ns`, which must be within `window_s` of now: every access unit from the
    last IDR with SPS and PPS at or before it, in decode order, then live. `live` is a connection of its own, made
    with subscription_overflow="backpressure". `data_root` is the state directory that holds `continuum/`, which
    relative chunk locators name files under (the working directory when None). Raises ValueError for a start before
    the window."""

class WorkLease:
    """A granted lease. End it with commit, abort, or release; async with releases or aborts it."""

    @property
    def id(self) -> int: ...
    @property
    def expires_at_ns(self) -> int | None: ...
    def renew(self, extend_ms: int = 30000) -> Awaitable[int]: ...
    def keep_alive(self, every_s: float, extend_ms: int = 30000) -> None: ...
    def commit(
        self,
        outputs: list[tuple[Buffer, ShmLocation | None]] = ...,
        *,
        coverage: CoverageDone | None = None,
        cost: ComputeCost | None = None,
    ) -> Awaitable[None]: ...
    def abort(self, reason: str) -> Awaitable[None]: ...
    def release(self) -> None: ...
    def __aenter__(self) -> Awaitable[Self]: ...
    def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> Awaitable[None]: ...

def decode_header(data: Buffer) -> PayloadHeader:
    """A header's bytes as a dict with every key. Raises ValueError for bytes that aren't a valid header."""

class Promise:
    """A committed promise (Client.commit_promise): its outputs, routed here by its id, so they never come from
    next_event. ContinuumError from next_part or fulfilled when it failed or the connection ended."""

    @property
    def id(self) -> str: ...
    @property
    def life(self) -> str: ...
    def next_part(self) -> Awaitable[PromisePart | None]:
        """The next part of the outputs, or None after the final part."""
    def fulfilled(self) -> Awaitable[list[Item]]:
        """Every output, gathered through the final part."""

class BrokerError(ContinuumError):
    """The broker refused a request. args are (code, message), code being the ErrorCode's name."""

class ConnectionClosedError(ConnectionError): ...

class Client:
    """A connection to the broker. Every method that talks to it returns an awaitable bound to the running event
    loop, so call them from a coroutine (``await Client.connect(...)`` inside ``async def``)."""

    @staticmethod
    def connect(
        socket: str | PathLike[str],
        client_name: str = "python",
        heartbeat_interval_s: float = 5.0,
        request_timeout_s: float = 10.0,
        subscription_overflow: str = "close",
    ) -> Awaitable[Client]: ...
    def get_chunk(self, chunk_id: str) -> Awaitable[dict[str, Any]]: ...
    @property
    def session_id(self) -> str: ...
    @property
    def broker_schema(self) -> tuple[int, int]:
        """The broker's schema version, (major, minor)."""
    @property
    def voice_turn(self) -> VoiceTurnState | None: ...
    @property
    def arena_names(self) -> list[str]: ...
    def arena(self, name: str) -> Arena: ...
    def heartbeat(
        self,
        *,
        queue_depth: int | None = None,
        models_resident: list[str] | None = None,
        ram_bytes: int = 0,
        vram_bytes: int = 0,
    ) -> Awaitable[int]: ...
    def register_stream(
        self,
        name: str,
        *,
        kind: Literal["sensor", "derived"],
        format: PayloadFormat,
        shape_name: str,
        shape_version: int = 1,
        nominal_rate_hz: float = 0.0,
        slab_class: int | None = None,
        ring_stride_bytes: int | None = None,
        ring_seconds: float = 0.0,
        anchor: tuple[str, int, int] | None = None,
        layer: DomainLayer = "Layer1Raw",
    ) -> Awaitable[StreamRegistered]: ...
    def ingest(self, header: Buffer, *, inline: bytes | None = None, shm: ShmLocation | None = None) -> None: ...
    def register_worker(
        self,
        worker_id: str,
        capabilities: list[Capability],
        *,
        version_hash: str = "",
        device_index: int | None = None,
        keeps_copies: bool = False,
    ) -> Awaitable[WorkerRegistered]: ...
    def quote(
        self,
        capability_id: int,
        *,
        stage: str,
        range: tuple[int, int],
        band: PriorityBand = "P3Background",
        compute_domain: ComputeDomain = "CPU",
        output_bytes: int = 0,
        est_duration_ms: int = 0,
        pin_parents: list[str] = ...,
    ) -> Awaitable[WorkLease]: ...
    def set_system_state(self, voice_turn: VoiceTurnState) -> None: ...
    def query(
        self,
        stream: str,
        start_ns: int,
        end_ns: int,
        *,
        all: bool = True,
        max_items: int = 0,
        band: PriorityBand = "P2Nearline",
        subjects: list[str] = ...,
        ids: list[str] = ...,
    ) -> Awaitable[QueryResult]:
        """What stream holds over [start_ns, end_ns): every item in time order, or with all=False one time query.
        truncated flags an item, inline-byte, or serialized-frame cap, separately from complete=False for gaps.
        all=True drains pages and returns truncated=False, or raises ContinuumError if a truncated page lacks a cursor.
        A single time query has no next_after continuation cursor."""

    def query_inline_records(
        self,
        stream: str,
        start_ns: int,
        end_ns: int,
        *,
        max_items: int = 0,
        band: PriorityBand = "P2Nearline",
        subjects: list[str] = ...,
        ids: list[str] = ...,
    ) -> Awaitable[list[tuple[str, int, int, bytes] | Item]]:
        """Fully drain like query(all=True), preserving order. Inline items are (id, start_ns, end_ns, bytes);
        every other item, including a None locator, stays unchanged for the ordinary reader's fallback.
        max_items bounds each page, not the total. Raises rather than returning a partial drain.
        Does not parse JSON or read slots/chunks."""

    def query_quoted(
        self,
        stream: str,
        start_ns: int,
        end_ns: int,
        *,
        band: PriorityBand = "P2Nearline",
        latency_budget_ms: int = 0,
        max_items: int = 0,
    ) -> Awaitable[QuotedResult]:
        """What stream holds over [start_ns, end_ns) (one time query, QuoteIfMissing), and "quote": the quote for
        what's missing, or None when nothing is missing or nothing missing can be made."""

    def commit_promise(self, promise_id: str, option: int = 0, band: PriorityBand = "P2Nearline") -> Awaitable[Promise]:
        """Commits a quoted promise (option 0 is the broker's pick); its outputs come to the returned Promise."""

    def cancel_promise(self, promise_id: str) -> Awaitable[PromiseAccepted]:
        """Calls a promise off: its jobs are revoked and nothing more comes."""

    def ring_reader(self, lease: StreamLease, *, from_oldest: bool = False) -> RingReader:
        """A live reader over a ring stream's ring, in an arena this client mapped."""

    def ring_writer(self, lease: StreamLease) -> RingWriter: ...
    def subscribe(
        self,
        streams: list[str],
        *,
        start: StartAt = "Live",
        start_ns: int = 0,
        durable_name: str | None = None,
        pinned: bool = False,
        subjects: list[str] = ...,
        shape_name: str | None = None,
    ) -> Awaitable[Subscription]:
        """Follows streams from now, from start_ns, or from durable_name's saved position; see Subscription."""

    def job_lease(self, job: BrokerEvent) -> WorkLease:
        """Takes on a DispatchJob (decode_event's dict): its output lease, to commit or abort. Sets job["lease"] to
        None, so a second call raises instead of making a second handle."""

    def reject_job(self, job_id: str, reason: str) -> None:
        """Turns down a DispatchJob; the broker replans its promises. One way."""

    def ack_purge(self, purge_id: str) -> None:
        """Says this worker dropped its copies of a PurgeNotice's payloads. One way."""
    def ingest_acked(
        self,
        header: Buffer,
        *,
        inline: bytes | None = None,
        dependencies: list[tuple[int, int, int, int]] = ...,
    ) -> Awaitable[int]:
        """As ingest, but returns the payload's commit sequence once the broker has it; the same header again returns
        the same sequence, and a different header under the same id raises BrokerError (Conflict). dependencies
        (minor 31): (stream_id, start_ns, end_ns, revision) stretches it was made from, revision being the stream's
        invalidation_revision (or forget_revision) read before checking what was used; refused (Conflict) if something
        on that stream was forgotten since (an age-out doesn't count, minor 33), and a later purge
        removing anything on such a stream inside the stretch takes this payload too."""
    def query_by_seq(
        self,
        stream: str,
        *,
        after_seq: int = 0,
        max_items: int = 0,
        subjects: list[str] = ...,
        ids: list[str] = ...,
        start_before: int = 0,
        max_scan: int = 0,
    ) -> Awaitable[QueryResult]:
        """One page of stream's payloads in commit order after after_seq, filtered; each item has commit_seq, and
        next_after is where the next page starts (0 when nothing remains)."""
    def list_streams(self) -> Awaitable[list[dict[str, Any]]]:
        """Every registered stream, in id order: stream_id, name, kind, layer, format, shape_name, shape_version,
        live, invalidation_revision (int | None when unsupported by the broker), and forget_revision (minor 33: the
        invalidation revision as the stream's last forget, not age-out, left it; None from an older broker)."""
    def purge(
        self,
        *,
        start_ns: int | None = None,
        end_ns: int | None = None,
        streams: list[str] = ...,
        entity: str | None = None,
        ids: list[str] = ...,
        reason: str = "purge",
        detail: str = "",
    ) -> Awaitable[dict[str, Any]]:
        """Purges exactly one of a range (on streams), an entity, or ids; the JSON detail is kept for good and must not
        name a person: {"purge_id", "removed": [(stream_id, tier, count)]}."""
    def age_out(
        self,
        ids: list[str],
        committed_before_ns: int,
        *,
        reason: str = "aged out",
        detail: str = "",
        aged_gap_ns: int = 0,
    ) -> Awaitable[dict[str, Any]]:
        """Ages out these payloads only (nothing derived from them), those committed by committed_before_ns by the
        commit clock; their spans are recorded as aged out, joined into one run where at most aged_gap_ns apart on a
        stream (minor 29; 0 keeps each payload's span): {"purge_id", "removed": [(stream_id, tier, count)]}."""
    def aged_out(self, stream: str, start_ns: int, end_ns: int, *, limit: int = 0) -> Awaitable[list[tuple[int, int]]]:
        """The spans of a stream recorded as aged out within [start_ns, end_ns)."""
    def catalog_stats(self) -> Awaitable[dict[str, Any]]:
        """Where the catalog file's bytes go, from one read snapshot (slow on a large catalog): {"page_size",
        "file_bytes", "disk_bytes", "tables": [{"name", "rows", "height", "leaf_pages", "branch_pages", "stored_bytes",
        "metadata_bytes", "fragmented_bytes"}]}, largest first. Read-only. Needs broker schema minor 30."""
    def commit_seq_at(self, at_ns: int) -> Awaitable[int]:
        """The highest commit sequence known committed at or before at_ns (0 when none is that old)."""
    def resolve_frames(self, ids: list[str]) -> Awaitable[list[dict[str, Any]]]:
        """Which chunk holds each of ids (at most 256), in the asked order: id, state ("unknown", "payload", or
        "absorbed"), stream_id, seq, chunk_id (None unless absorbed), frame_index, and start_ns. What an analysis
        receipt names as its sources. Needs broker schema minor 26."""
    def list_purges(self, *, after: str | None = None, limit: int = 0) -> Awaitable[list[dict[str, Any]]]:
        """Purge summaries, oldest first: purge_id, at_ns, kind, range, stream_ids, id_count, reason, detail, removed,
        finished, completed_ns. A summary never names a person."""
    def get_purge(self, purge_id: str) -> Awaitable[dict[str, Any]]:
        """One purge's summary plus its receipt: removed_by [(stream_id, tier, count)], removed_frames,
        partial_chunks."""
    def forget_keeper(self, name: str) -> Awaitable[int]:
        """Drops a keeper's name (a keeps_copies worker retired or renamed), so purges stop waiting for it;
        resolves to how many purges were waiting on it."""
    def remember_forgotten(self, hash: bytes) -> Awaitable[None]:
        """Records that a person was forgotten: the name hashed by the caller (32 bytes), kept in the
        catalog's FORGOTTEN, which no purge removes."""
    def save_worker_state(
        self,
        owner: str,
        contract: str,
        position: int,
        created_ns: int,
        inputs: list[tuple[str, int]],
        body: bytes,
    ) -> Awaitable[None]:
        """A durable worker's checkpoint, in place of its last; Conflict when an input's revision moved."""
    def load_worker_state(self, owner: str, contract: str) -> Awaitable[dict[str, Any] | None]:
        """{"position", "created_ns", "inputs", "body"} while its inputs' revisions hold, else None."""
    def forget_worker_state(self, owner: str, contract: str) -> Awaitable[bool]:
        """Deletes a worker's checkpoint; whether it had one."""
    def list_forgotten(self, after: bytes | None = None, limit: int = 256) -> Awaitable[ForgottenPage]:
        """Read-only ascending 32-byte hashes and exclusive next cursor; limit 1..1024, schema minor 34.
        Concurrent additions below the cursor require a new enumeration. No names or observations are returned."""
    def is_forgotten(self, hash: bytes) -> Awaitable[bool]:
        """Whether a forgotten name's hash (32 bytes) is in FORGOTTEN."""
    def purge_range(
        self, start_ns: int, end_ns: int, *, streams: list[str] = ..., reason: str = "purge"
    ) -> Awaitable[dict[str, Any]]:
        """Purges [start_ns, end_ns) on the named streams (all when empty): {"purge_id", "removed": [(stream_id,
        tier, count)]}."""
    def commit_chunk(
        self,
        header: Buffer,
        uri: str,
        length: int,
        frames: list[tuple[int, int, int, bool]],
        absorbed: list[str],
        *,
        tier: int = 1,
    ) -> Awaitable[dict[str, Any]]:
        """Commits a chunk file absorbing `absorbed` (frames are (start_ns, offset, length, cut)): {"chunk_id",
        "commit_seq", "duplicate", "hidden"}. Only a registered worker commits chunks."""
    def register_v1_flac(self, stream_id: int, path: str | PathLike[str], start_ns: int) -> Awaitable[dict[str, Any]]:
        """Registers one v1 FLAC file in place on a v1 archive stream. Idempotent; needs a registered worker."""
    def register_v1_video_index(self, cam: str, video_dir: str | PathLike[str]) -> Awaitable[list[dict[str, Any]]]:
        """Registers camera `cam`'s v1 recordings (video/index.jsonl in `video_dir`) in place on `<cam>.video_v1`.
        Idempotent; needs a registered worker."""
    def register_v1_index(self, stream_name: str, device_dir: str | PathLike[str]) -> Awaitable[list[dict[str, Any]]]:
        """Registers every file a v1 device directory's index.jsonl lists, in place: one dict per line, with
        "registered" and either the ids and span or "skipped". Idempotent; needs a registered worker."""
    def next_event(self) -> Awaitable[tuple[str, bytes] | None]: ...
    def close(self, reason: str = "closed") -> Awaitable[None]: ...
    def __aenter__(self) -> Awaitable[Self]: ...
    def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> Awaitable[None]: ...

class Arena:
    @staticmethod
    def open(name: str) -> Arena: ...
    @staticmethod
    def create(
        name: str,
        classes: list[tuple[int, int]],
        rings: list[tuple[int, int]] = ...,
        generation: int = 0,
        reclaim_requires_flush: bool = False,
    ) -> Arena: ...
    @staticmethod
    def unlink(name: str) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def size(self) -> int: ...
    @property
    def generation(self) -> int: ...
    @property
    def base_address(self) -> int: ...
    def slab_class(self, slab_class: int) -> tuple[int, int]: ...
    def used(self, slab_class: int) -> int: ...
    def held(self, slab_class: int) -> int: ...
    @property
    def class_count(self) -> int: ...
    def lease(self, slab_class: int, index: int, generation: int, ttl_ms: int = 5000) -> Lease: ...
    def write(self, slab_class: int, data: Buffer, owner: int = 0) -> tuple[int, int, int]:
        """Claims a free slot, copies data (any C-contiguous buffer) in, and publishes it; returns (index,
        generation, offset)."""
    def pin_count(self, slab_class: int, index: int, generation: int) -> int: ...
    def ring_reader(self, ring: int, from_oldest: bool = True) -> RingReader: ...

class Lease(Buffer):
    @property
    def slab_class(self) -> int: ...
    @property
    def index(self) -> int: ...
    @property
    def generation(self) -> int: ...
    @property
    def offset(self) -> int: ...
    @property
    def address(self) -> int: ...
    @property
    def nbytes(self) -> int: ...
    def __len__(self) -> int: ...
    def __buffer__(self, flags: int, /) -> memoryview: ...
    def is_valid(self) -> bool: ...
    def renew(self, ttl_ms: int) -> None: ...
    def release(self) -> bool: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None: ...
    def numpy(self, dtype: str = "uint8", shape: list[int] | None = None) -> np.ndarray[Any, Any]: ...
    def __dlpack__(
        self,
        *,
        stream: Any = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool | None = None,
    ) -> Any: ...
    def __dlpack_device__(self) -> tuple[int, int]: ...

class RingReader:
    """A reader of a ring, with its own cursor. Follow it live with ``async for seq, record, lost in reader``."""

    @property
    def stride(self) -> int: ...
    @property
    def lost(self) -> int: ...
    def wait(self, timeout_ms: int) -> bool: ...
    def read(self) -> tuple[int, bytes] | None: ...
    def read_record(self) -> tuple[int, bytes, int] | None: ...
    def _ready(self, timeout_ms: int) -> Awaitable[bool]: ...
    def next_record(self, timeout_ms: int | None = None) -> Awaitable[tuple[int, bytes, int] | None]: ...
    def __aiter__(self) -> AsyncIterator[tuple[int, bytes, int]]: ...

class RingWriter:
    """A ring stream's single writer (Client.ring_writer): push(data) appends one record, zero-padded to the stride."""

    @property
    def stride(self) -> int: ...
    def push(self, data: Buffer) -> int: ...

class SegmentWriter:
    """The one writer of a segment stream's camera. `schema` is (name, type) pairs with `ts_ns` "i64"."""

    def __init__(
        self,
        root: str | PathLike[str],
        stream: str,
        camera: str,
        schema: list[tuple[str, str]],
        resume_at: int | None = None,
    ) -> None: ...
    @property
    def next_seq(self) -> int: ...
    def append(self, columns: dict[str, Any]) -> int: ...
    def sync(self) -> int: ...
    def close(self) -> None: ...

class SegmentFollower:
    """Follows a segment stream from a cursor ({camera: next row_seq}); poll() gives what was written since."""

    def __init__(
        self,
        root: str | PathLike[str],
        stream: str,
        schema: list[tuple[str, str]],
        cursor: dict[str, int] | None = None,
        cameras: list[str] | None = None,
    ) -> None: ...
    @property
    def cursor(self) -> dict[str, int]: ...
    def poll(self, timeout_s: float = 0.0) -> dict[str, Any]: ...

class SegmentFollowers:
    """Follows several segment streams ({stream: schema}) with one cursor ({stream: {camera: next row_seq}})."""

    def __init__(
        self,
        root: str | PathLike[str],
        schemas: dict[str, list[tuple[str, str]]],
        cursor: dict[str, dict[str, int]] | None = None,
        cameras: list[str] | None = None,
    ) -> None: ...
    @property
    def cursor(self) -> dict[str, dict[str, int]]: ...
    def poll(self, timeout_s: float = 0.0) -> list[tuple[str, dict[str, Any]]]: ...

def read_segments(
    root: str | PathLike[str],
    stream: str,
    schema: list[tuple[str, str]],
    t0: int,
    t1: int,
    cameras: list[str] | None = None,
    overlapping: bool = False,
) -> dict[str, Any]:
    """Rows with ts_ns in [t0, t1) (with overlapping, spans reaching into it): the schema's columns, plus camera, row_seq, and parts (the exact part set read)."""

def forget_segments(
    root: str | PathLike[str],
    t0: int,
    t1: int,
    streams: list[str] | None = None,
    cameras: list[str] | None = None,
    kind: Literal["forget", "reset"] = "forget",
) -> dict[str, Any]:
    """Forgets rows with ts_ns in [t0, t1): gone from every read at once, and kept out of later appends."""

def forget_segments_where(
    root: str | PathLike[str],
    column: str,
    value: str,
    streams: list[str] | None = None,
    cameras: list[str] | None = None,
    kind: Literal["forget", "reset"] = "forget",
) -> dict[str, Any]:
    """Forgets every row, at any time, whose `column` holds `value`: a person or voiceprint forgotten."""

def segment_deletions(root: str | PathLike[str], after: int = 0) -> list[dict[str, Any]]:
    """The deletion feed after `after`, oldest first: revision, kind, streams, cameras, t0, t1, at_ns."""

def segment_tail(root: str | PathLike[str], stream: str, cameras: list[str] | None = None) -> dict[str, int]:
    """Where each camera of `stream` ends now: a cursor to follow from here on, without reading history."""

def compact_segments(root: str | PathLike[str]) -> int:
    """Rewrites every closed part holding more than one record batch as one; how many it rewrote."""

def segment_revision(root: str | PathLike[str]) -> int:
    """The newest deletion's revision; 0 when nothing was ever deleted."""

class H264Decoder:
    """Stateful software or CUDA/NVDEC decoder, returning host NV12. Requires the optional ffmpeg feature.
    `codec="av1"` decodes an archived chunk's AV1 temporal units, in software only."""

    def __init__(self, backend: str = "software", gpu: int = 0, codec: str = "h264") -> None: ...
    @staticmethod
    def supported() -> bool: ...
    @property
    def waiting_for_keyframe(self) -> bool: ...
    @property
    def dropped_tags(self) -> list[int]:
        """Original tags dropped in the most recent push/finish call; release their headers and mark a gap."""
    def reset(self) -> None: ...
    def push(self, data: Buffer, tag: int, presentation_ns: int | None = None) -> list[tuple[int, int, int, bytes]]: ...
    def finish(self) -> list[tuple[int, int, int, bytes]]: ...
    def push_outputs(
        self, data: Buffer, tag: int, presentation_ns: int | None = None, want_bgr: bool = False, keep: bool = True
    ) -> list[tuple[int, int, int, bytes, str, str, bytes | None]]:
        """Decode one unit; `keep` False decodes it for its references only, its picture back with empty NV12."""
    def finish_outputs(self, want_bgr: bool = False) -> list[tuple[int, int, int, bytes, str, str, bytes | None]]: ...

def h264_decoder_available() -> bool: ...
