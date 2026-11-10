"""The schema's payload types (continuum/schemas/continuum.fbs) as TypedDicts, with the .fbs field names.

Ids are canonical UUID strings, times UTC nanoseconds, and enums their schema names. encode_header and
decode_header (in continuum_py) convert a PayloadHeader to and from its FlatBuffer bytes.
"""

from typing import Literal, NotRequired, TypedDict

DomainLayer = Literal["Layer1Raw", "Layer2Decision", "Layer3Interpretation"]
StorageTier = Literal["Tier0Hot", "Tier1Warm", "Tier2Cold"]
PayloadFormat = Literal["RawBytes", "DLPackTensor", "ArrowRecordBatch", "EncodedVideo", "FlatBuffer", "Text"]


class Producer(TypedDict):
    producer_id: str
    model: str
    version: str
    version_hash: str
    prompt_id: str | None
    prompt_hash: str | None


class EvidenceRef(TypedDict):
    payload_id: str
    start_ns: int | None
    end_ns: int | None


class ComputeCost(TypedDict):
    wall_time_us: int
    queue_wait_us: int
    ram_bytes: int
    vram_bytes: int
    device: str
    items: int


class CaptureInfo(TypedDict):
    capture_mono_ns: int
    device_ts: int
    arrival_ns: int
    stride: int
    width: int
    height: int
    flip: bool
    device_seq: int
    sample_rate_hz: NotRequired[int]
    capture_epoch: NotRequired[int]


class PayloadHeader(TypedDict):
    """A payload's header. encode_header needs the keys without NotRequired; decode_header returns every key."""

    id: str
    stream_id: int
    start_ns: int
    end_ns: int
    seq: NotRequired[int]
    layer: DomainLayer
    format: PayloadFormat
    shape_name: str
    shape_version: NotRequired[int]
    producer: NotRequired[Producer | None]
    parents: NotRequired[list[str]]
    evidence: NotRequired[list[EvidenceRef]]
    source_tier: NotRequired[StorageTier]
    compute: NotRequired[ComputeCost | None]
    confidence: NotRequired[float | None]
    subjects: NotRequired[list[str]]
    capture: NotRequired[CaptureInfo | None]


class H264AccessUnit(TypedDict):
    """One access unit of a recent H.264 window (open_h264_window). A value capture didn't give is None, never a
    guess. `header` is the whole original header from Tier 0 (`tier` "tier0"); a Tier 1 chunk keeps the id, times,
    sequence, device sequence, epoch, and flip, but not the rest of the header, so it is None there."""

    type: Literal["au"]
    id: str
    start_ns: int
    end_ns: int
    seq: int
    device_seq: int | None
    capture_epoch: int | None
    width: int | None
    height: int | None
    flip: bool
    random_access: bool
    tier: Literal["tier0", "tier1"]
    header: bytes | None
    bytes: bytes


class H264Gap(TypedDict):
    """A span of a recent H.264 window with no decodable access units; the next access unit is a random access
    point."""

    type: Literal["gap"]
    from_ns: int
    to_ns: int
    why: Literal["missing", "epoch_changed", "continuity_unknown", "unreadable", "before_random_access"]


# ---- What the broker answers


class StreamLease(TypedDict):
    lease_id: int
    arena: str
    arena_generation: int
    first_slot: int
    slot_count: int
    slot_bytes: int
    ring_offset: int
    ring_records: int
    ring_index: int
    ring_data_offset: int
    ring_stride: int
    ring_generation: int


class RingLease(TypedDict):
    """A followed ring stream's ring, from a Subscription's rings: open it with Client.ring_reader(lease)."""

    stream_id: int
    lease: StreamLease


class StreamRegistered(TypedDict):
    stream_id: int
    lease: StreamLease | None


class ShmLocation(TypedDict):
    """Where a payload's bytes sit in Tier 0: a slot of an arena (Client.ingest's shm)."""

    arena: str
    arena_generation: int
    slab_class: int
    slot: int
    slot_generation: int
    offset: int
    length: int


class IngestRejected(TypedDict):
    """A refused IngestNotice, from decode_ingest_rejected."""

    stream_id: int
    payload_id: str | None
    reason: str


# ---- The worker side

ComputeDomain = Literal["CPU", "GPU", "ANE", "MediaEngine"]
PriorityBand = Literal["P0Critical", "P1Realtime", "P2Nearline", "P3Background"]
VoiceTurnState = Literal["Idle", "Listening", "Thinking", "Speaking"]
CapabilityRole = Literal["Produce", "Hydrate"]


class Requirement(TypedDict):
    stream: str
    before_ns: NotRequired[int]
    after_ns: NotRequired[int]


class Capability(TypedDict):
    """What a worker can make and needs (register_worker). produces and compute_domain are required."""

    produces: str
    compute_domain: ComputeDomain
    requires: NotRequired[list[Requirement]]
    ram_mb: NotRequired[int]
    vram_mb: NotRequired[int]
    est_ms_per_item: NotRequired[float]
    cold_start_ms: NotRequired[int]
    max_batch: NotRequired[int]
    preemptible: NotRequired[bool]
    role: NotRequired[CapabilityRole]
    output_bytes: NotRequired[int]
    outputs_per_job: NotRequired[int]
    expected_confidence: NotRequired[float | None]


class AcceptedCapability(TypedDict):
    index: int
    id: int
    runnable: bool


class WorkerRegistered(TypedDict):
    accepted: list[AcceptedCapability]
    refused_reasons: list[str]


class CoverageDone(TypedDict):
    """A commit's coverage: the stream read, the stage, and the time covered (UTC ns, half open)."""

    source_stream_id: int
    stage: str
    range: tuple[int, int]


class Locator(TypedDict):
    """Where an input's bytes are. kind is "Shm" (the ShmLocation keys), "Inline" (bytes), "Chunk" (tier, uri,
    offset, length), or "Ring" (arena, arena_generation, ring, first_seq, count: a batch of ring records)."""

    kind: Literal["Shm", "Inline", "Chunk", "Ring"]
    arena: NotRequired[str]
    arena_generation: NotRequired[int]
    slab_class: NotRequired[int]
    slot: NotRequired[int]
    slot_generation: NotRequired[int]
    bytes: NotRequired[bytes]
    tier: NotRequired[str]
    uri: NotRequired[str]
    offset: NotRequired[int]
    length: NotRequired[int]
    ring: NotRequired[int]
    first_seq: NotRequired[int]
    count: NotRequired[int]


class Item(TypedDict):
    """A dispatched job's input: its header (decode_header reads it) and where its bytes are (None: header only)."""

    header: bytes
    locator: Locator | None
    commit_seq: NotRequired[int]


class JobLease(TypedDict):
    """The output lease a DispatchJob came with (Client.job_lease takes it on)."""

    lease_id: int
    expires_at_ns: int
    slots: list[ShmLocation]


class PurgedSpan(TypedDict):
    """A stream's time range a purge removed (UTC ns, half open): a copy over it goes even when its id isn't listed."""

    stream_id: int
    range: tuple[int, int]


class BrokerEvent(TypedDict):
    """What decode_event returns; which keys are set depends on type. A Fulfilled (a part of a committed promise's
    outputs that no Promise holds) carries promise_id, items, and final_part; a PromiseFailed promise_id, reason, and
    delivered. A DispatchJob carries job_id through lease, a
    PurgeNotice purge_id, payload_ids, and spans (the client gathers a paged notice into one; ack it once), an Error (a refused one-way send) code and message, and an Undecodable "of"
    (the message's type), reason, and a job's job_id and lease_id when they could be read (None otherwise)."""

    type: str
    lease_id: NotRequired[int]
    reason: NotRequired[str]
    by_band: NotRequired[PriorityBand]
    revoked: NotRequired[bool]
    voice_turn: NotRequired[VoiceTurnState]
    changed_at_ns: NotRequired[int]
    stream_id: NotRequired[int]
    payload_id: NotRequired[str | None]
    job_id: NotRequired[str]
    promise_ids: NotRequired[list[str]]
    capability: NotRequired[str]
    capability_id: NotRequired[int]
    inputs: NotRequired[list[Item]]
    output_range: NotRequired[tuple[int, int]]
    band: NotRequired[PriorityBand]
    deadline_ns: NotRequired[int]
    lease: NotRequired[JobLease | None]
    purge_id: NotRequired[str]
    payload_ids: NotRequired[list[str]]
    spans: NotRequired[list[PurgedSpan]]
    of: NotRequired[str]
    code: NotRequired[str]
    message: NotRequired[str]
    promise_id: NotRequired[str]
    items: NotRequired[list[Item]]
    final_part: NotRequired[bool]
    delivered: NotRequired[int]


# ---- Subscriptions

StartAt = Literal["Live", "Time", "Saved"]


class SubscriptionEvent(TypedDict):
    """What a Subscription yields. type "Event" carries position and item (header bytes, and a locator when the
    subscription is pinned); "CaughtUp" carries position (the catch-up is over, live follows); "Undecodable" carries
    position and reason."""

    type: Literal["Event", "CaughtUp", "Undecodable"]
    position: int
    item: NotRequired[Item]
    reason: NotRequired[str]


# ---- Time queries


class RingTail(TypedDict):
    """A ring stream's unsealed tail (past its last sealed batch), to read from the ring directly."""

    arena: str
    ring_offset: int
    first_seq: int
    last_seq: int


class QueryResult(TypedDict):
    """What a query found, its coverage, and whether a limit cut its page short.

    truncated is independent of complete: an uncapped sparse time range can have gaps and complete=False.
    Client.query(all=True) drains all pages and returns truncated=False.
    """

    items: list[Item]
    complete: bool
    truncated: bool
    missing: list[tuple[int, int]]
    tail: RingTail | None
    next_after: int


class PlanStep(TypedDict):
    """One step of a quoted plan: a tier read (worker_id empty), or a worker running a capability."""

    read_tier: StorageTier
    worker_id: str
    produces: str
    items: int
    est_ms: float


class PlanOption(TypedDict):
    """One way to make what a query found missing."""

    steps: list[PlanStep]
    est_latency_ms: float
    cost_units: float
    exceeds_budget: bool
    expected_confidence: float | None


class PromiseQuote(TypedDict):
    """A quote for what a query found missing: commit it (Client.commit_promise) before expires_at_ns. Option 0 is the
    broker's pick."""

    promise_id: str
    status: Literal["RequiresCompute", "AwaitingCapture", "AwaitingCaptureAndCompute"]
    options: list[PlanOption]
    shares_in_flight: bool
    expires_at_ns: int


class QuotedResult(QueryResult):
    """What Client.query_quoted found: the query result, and the quote for what's missing, or None."""

    quote: PromiseQuote | None


class PromiseAccepted(TypedDict):
    """Where a promise stands after a cancel."""

    promise_id: str
    life: Literal["Armed", "Dispatching", "Cancelled", "Failed"]


class PromisePart(TypedDict):
    """One part of a promise's outputs."""

    items: list[Item]
    final_part: bool


class ForgottenPage(TypedDict):
    """Read-only opaque-hash page; next is an exclusive cursor only when more hashes remain."""

    hashes: list[bytes]
    next: bytes | None
