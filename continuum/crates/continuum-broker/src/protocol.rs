//! Building and reading `ControlEnvelope`s.
//!
//! Every message on the socket is a `ControlEnvelope` (schemas/continuum.fbs) whose `payload` is one member of the
//! `ControlPayload` union. The root type of the schema file is `PayloadHeader`, so envelopes are read with
//! [`flatbuffers::root`] rather than a generated `root_as_*` function.

/// Bounded shared-copy lifecycle request/reply codecs.
pub mod aging;
pub mod applicability;
pub mod worker_state;

use continuum_catalog::{
    ClockAnchor, PurgeKind, PurgeRecord as CatalogPurgeRecord, PurgeReport, PurgeSummary, StreamInfo, StreamKind,
};
use continuum_core::schema::continuum::{
    CancelPromise, CapabilityRole as WireRole, CaughtUp, CaughtUpArgs, CommitLease, CommitPromise, ControlEnvelope,
    ControlEnvelopeArgs, ControlPayload, DispatchJob, DispatchJobArgs, Error, ErrorArgs, ErrorCode, ForgetKeeper,
    ForgottenHash as WireForgottenHash, ForgottenRemembered, ForgottenRememberedArgs, ForgottenStatus,
    ForgottenStatusArgs, FulfillPromise, FulfillPromiseArgs, GetPurge, Heartbeat, HeartbeatArgs, IngestNotice,
    IngestRejected, IngestRejectedArgs, Ingested, IngestedArgs, Item, ItemArgs, JobRejected, KeeperForgotten,
    KeeperForgottenArgs, LeaseGrant, LeaseGrantArgs, LeasePreempted, LeasePreemptedArgs, LeaseReleased,
    LeaseReleasedArgs, ListPurges, OutputLease as WireOutputLease, OutputLeaseArgs, PayloadHeader, PlanOption,
    PlanOptionArgs, PlanStep as WirePlanStep, PlanStepArgs, PromiseAccepted, PromiseAcceptedArgs, PromiseFailed,
    PromiseFailedArgs, PromiseLife, PromiseQuote, PromiseQuoteArgs, PromiseStatus as WirePromiseStatus, Purge,
    PurgeKind as WirePurgeKind, PurgeList, PurgeListArgs, PurgeNotice, PurgeNoticeArgs, PurgeRecord, PurgeRecordArgs,
    PurgeResult, PurgeResultArgs, PurgeSummary as WirePurgeSummary, PurgeSummaryArgs, PurgedSpan, PurgedSpanArgs,
    Query as WireQuery, QueryMode as WireQueryMode, QueryResult, QueryResultArgs, QuoteRequest, RegisterStream,
    RegisterWorker, RemovedCount, RemovedCountArgs, SchemaVersion, SchemaVersionArgs, StartAt, StreamDesc,
    StreamDescArgs, StreamEvent, StreamEventArgs, StreamKind as WireKind, StreamList, StreamListArgs, StreamRegistered,
    StreamRegisteredArgs, Subscribe as WireSubscribe, Subscribed, SubscribedArgs, SystemStateChanged,
    SystemStateChangedArgs, TimeRange, Unsubscribed, UnsubscribedArgs, Uuid as WireUuid, Welcome, WelcomeArgs,
    WorkerRegistered, WorkerRegisteredArgs,
};
use continuum_core::{
    BootId, ComputeDomain, HeaderBuf, Interval, PriorityBand, StageId, StorageLocator, StorageTier, StreamId, Uuid,
    VoiceTurnState,
};
use continuum_sched::{
    CapabilityRole, ComputeCost, Dispatch, JobId, LeaseDecision, LeaseReq, Plan, PromiseId, QueryMode, Quote,
    QuoteStatus, StepKind, WorkerId,
};
use flatbuffers::{FlatBufferBuilder, UnionWIPOffset, WIPOffset};

/// The control protocol version this broker speaks (from continuum-core, which the SDKs share).
pub use continuum_core::{SCHEMA_MAJOR, SCHEMA_MINOR};

pub use continuum_wire::read_envelope;

/// Finishes `b` as an envelope around `payload` and returns its bytes.
fn finish<'b>(
    b: &'b mut FlatBufferBuilder<'static>,
    request_id: u64,
    payload_type: ControlPayload,
    payload: WIPOffset<UnionWIPOffset>,
) -> &'b [u8] {
    let env = ControlEnvelope::create(b, &ControlEnvelopeArgs { request_id, payload_type, payload: Some(payload) });
    b.finish(env, None);
    b.finished_data()
}

fn schema_version<'a>(b: &mut FlatBufferBuilder<'a>) -> WIPOffset<SchemaVersion<'a>> {
    SchemaVersion::create(b, &SchemaVersionArgs { major: SCHEMA_MAJOR, minor: SCHEMA_MINOR })
}

/// A `Welcome` for a new session, listing `arenas` in the order their fds go alongside it.
pub fn welcome(
    b: &mut FlatBufferBuilder<'static>,
    request_id: u64,
    session_id: [u8; 16],
    arenas: &[&crate::tier0::ArenaSummary],
    voice_turn: VoiceTurnState,
) -> Vec<u8> {
    use continuum_core::schema::continuum::{ArenaInfo, ArenaInfoArgs, SizeClass, SizeClassArgs};
    b.reset();
    let schema = schema_version(b);
    let infos: Vec<_> = arenas
        .iter()
        .map(|a| {
            let classes: Vec<_> = a
                .classes
                .iter()
                .map(|&(slab_bytes, count)| SizeClass::create(b, &SizeClassArgs { slab_bytes, count }))
                .collect();
            let classes = b.create_vector(&classes);
            let name = b.create_string(&a.name);
            ArenaInfo::create(
                b,
                &ArenaInfoArgs {
                    name: Some(name),
                    generation: a.generation,
                    size_bytes: a.size_bytes,
                    classes: Some(classes),
                },
            )
        })
        .collect();
    let arenas = b.create_vector::<WIPOffset<ArenaInfo>>(&infos);
    let id = WireUuid::new(&session_id);
    let w = Welcome::create(
        b,
        &WelcomeArgs { session_id: Some(&id), schema: Some(schema), arenas: Some(arenas), voice_turn },
    );
    finish(b, request_id, ControlPayload::Welcome, w.as_union_value()).to_vec()
}

/// A `Heartbeat` reply carrying the broker's clock.
pub fn heartbeat(b: &mut FlatBufferBuilder<'static>, request_id: u64, now_ns: u64) -> Vec<u8> {
    b.reset();
    let h = Heartbeat::create(b, &HeartbeatArgs { now_ns, telemetry: None });
    finish(b, request_id, ControlPayload::Heartbeat, h.as_union_value()).to_vec()
}

/// An `Error` reply.
pub fn error(b: &mut FlatBufferBuilder<'static>, request_id: u64, code: ErrorCode, message: &str) -> Vec<u8> {
    b.reset();
    let message = b.create_string(message);
    let e = Error::create(b, &ErrorArgs { code, message: Some(message) });
    finish(b, request_id, ControlPayload::Error, e.as_union_value()).to_vec()
}

/// The union member's name, for logs and error messages.
#[must_use]
pub fn payload_name(t: ControlPayload) -> &'static str {
    t.variant_name().unwrap_or("unknown")
}

/// A `StreamRegistered` reply. The standing lease comes with the data plane (continuum-shm); until then it is unset.
/// A ring a `RegisterStream` asks for: its record size and how many seconds it keeps (0 for the default).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RingAsk {
    /// Bytes per record, exactly as the device delivers them.
    pub stride: u32,
    /// How long the ring keeps its records.
    pub seconds: f32,
}

/// The ring a `RegisterStream` asks for, if any (`ring_stride_bytes` set).
///
/// # Errors
/// The ring's length isn't a time, or the stream has no rate to size it by.
pub fn ring_ask(r: &RegisterStream<'_>) -> Result<Option<RingAsk>, String> {
    if r.ring_stride_bytes() == 0 {
        return Ok(None);
    }
    let seconds = r.ring_seconds();
    if !seconds.is_finite() || seconds < 0.0 {
        return Err(format!("ring_seconds {seconds} isn't a length of time"));
    }
    if r.nominal_rate_hz().is_nan() || r.nominal_rate_hz() <= 0.0 {
        return Err("a ring stream needs nominal_rate_hz, its records per second, to size the ring".into());
    }
    Ok(Some(RingAsk { stride: r.ring_stride_bytes(), seconds }))
}

/// A ring as `StreamRegistered` announces it: the arena, its index, and its layout.
#[derive(Clone, Copy, Debug)]
pub struct RingGrant<'a> {
    /// The arena it's in.
    pub arena: &'a crate::tier0::ArenaSummary,
    /// What a `RingLocation` names.
    pub index: u16,
    /// Its layout.
    pub ring: continuum_shm::CarvedRing,
}

/// A ring's layout as a `StreamLease`.
fn ring_lease<'a>(
    b: &mut FlatBufferBuilder<'a>,
    g: RingGrant<'_>,
) -> WIPOffset<continuum_core::schema::continuum::StreamLease<'a>> {
    use continuum_core::schema::continuum::{StreamLease, StreamLeaseArgs};
    let arena = b.create_string(&g.arena.name);
    StreamLease::create(
        b,
        &StreamLeaseArgs {
            arena: Some(arena),
            arena_generation: g.arena.generation,
            ring_offset: g.ring.ctrl_offset,
            ring_records: g.ring.capacity,
            ring_index: g.index,
            ring_data_offset: g.ring.data_offset,
            ring_stride: g.ring.stride,
            ring_generation: g.ring.generation,
            ..Default::default()
        },
    )
}

/// A `StreamRegistered`, with the ring's layout in its lease for a ring stream.
pub fn stream_registered(
    b: &mut FlatBufferBuilder<'static>,
    request_id: u64,
    stream_id: u32,
    ring: Option<RingGrant<'_>>,
) -> Vec<u8> {
    b.reset();
    let lease = ring.map(|g| ring_lease(b, g));
    let r = StreamRegistered::create(b, &StreamRegisteredArgs { stream_id, lease });
    finish(b, request_id, ControlPayload::StreamRegistered, r.as_union_value()).to_vec()
}

/// The catalog's [`StreamInfo`] from a `RegisterStream`, or why it can't be one.
///
/// # Errors
///
/// A message for the client when the name is empty, the kind is unknown, or the rate isn't a finite number at least
/// 0 (0 means the stream has no nominal rate).
pub fn stream_info(r: &RegisterStream<'_>) -> Result<StreamInfo, String> {
    let name = r.name().unwrap_or("");
    if name.is_empty() {
        return Err("RegisterStream needs a name".into());
    }
    let kind = match r.kind() {
        WireKind::Sensor => StreamKind::Sensor,
        WireKind::Derived => StreamKind::Derived,
        other => return Err(format!("unknown stream kind {}", other.0)),
    };
    let rate = r.nominal_rate_hz();
    if !rate.is_finite() || rate < 0.0 {
        return Err(format!("nominal_rate_hz {rate} isn't a rate"));
    }
    let anchor = r.anchor().map(|a| ClockAnchor {
        boot: BootId::from(a.boot_id()),
        mono_ns: a.monotonic_ns(),
        utc_ns: a.utc_ns(),
    });
    Ok(StreamInfo {
        name: name.to_owned(),
        kind,
        layer: r.layer(),
        format: r.format(),
        shape_name: r.shape_name().unwrap_or("").to_owned(),
        shape_version: r.shape_version(),
        rate_hz: (rate > 0.0).then_some(f64::from(rate)),
        anchor,
    })
}

/// An `IngestRejected` for a notice the broker didn't index.
pub fn ingest_rejected(
    b: &mut FlatBufferBuilder<'static>,
    request_id: u64,
    stream_id: u32,
    payload_id: Option<[u8; 16]>,
    reason: &str,
    code: ErrorCode,
) -> Vec<u8> {
    b.reset();
    let reason = b.create_string(reason);
    let id = payload_id.map(|p| WireUuid::new(&p));
    let r = IngestRejected::create(
        b,
        &IngestRejectedArgs { stream_id, payload_id: id.as_ref(), reason: Some(reason), code },
    );
    finish(b, request_id, ControlPayload::IngestRejected, r.as_union_value()).to_vec()
}

/// A notice's header (checked as a `PayloadHeader` buffer of its own) and locator, or why they can't be used.
///
/// # Errors
///
/// A message for the client when `header_bytes` is missing or isn't a valid header, the header's stream isn't the
/// notice's, or the locator is malformed. A notice may leave the locator out (bytes that have no copy yet).
pub fn ingest(n: &IngestNotice<'_>) -> Result<(HeaderBuf, Option<StorageLocator>), String> {
    let bytes = n.header_bytes().ok_or("IngestNotice without header_bytes")?;
    let header = HeaderBuf::new(bytes.bytes().to_vec()).map_err(|e| format!("header_bytes: {e}"))?;
    if header.stream().0 != n.stream_id() {
        return Err(format!("the header is on stream {}, the notice on {}", header.stream().0, n.stream_id()));
    }
    let locator =
        n.locator().map(|l| StorageLocator::from_table(&l)).transpose().map_err(|e| format!("locator: {e}"))?;
    Ok((header, locator))
}

/// A notice's dependencies (minor 31, #1535). Only an acknowledged notice (a nonzero request id) may carry them: a
/// sender that hears nothing back couldn't learn they were refused.
///
/// # Errors
///
/// A message for the client for an empty range, or dependencies on a notice that asks for no answer.
pub fn dependencies(n: &IngestNotice<'_>, request_id: u64) -> Result<Vec<continuum_catalog::Dependency>, String> {
    let Some(deps) = n.dependencies() else { return Ok(Vec::new()) };
    if !deps.is_empty() && request_id == 0 {
        return Err("dependencies need an acknowledged notice (a nonzero request id)".into());
    }
    deps.iter()
        .map(|d| {
            let r = d.range().ok_or("a dependency without a range")?;
            let span = Interval::new(r.start_ns(), r.end_ns())
                .ok_or_else(|| format!("a dependency's range [{}, {}) is empty", r.start_ns(), r.end_ns()))?;
            Ok(continuum_catalog::Dependency { stream: StreamId(d.stream_id()), span, revision: d.revision() })
        })
        .collect()
}

/// The payload id in a notice's `header_bytes`, when they verify as a `PayloadHeader`. (The generated
/// `header_bytes_nested_flatbuffer` doesn't verify, so it would panic on bad bytes.)
#[must_use]
pub fn header_id(n: &IngestNotice<'_>) -> Option<[u8; 16]> {
    let bytes = n.header_bytes()?;
    let h = flatbuffers::root::<PayloadHeader>(bytes.bytes()).ok()?;
    h.id().map(|u| u.bytes().into())
}

/// The most bytes a purge's `reason` or `detail` may have: the record keeps them for good and every scrub copies it.
pub const MAX_PURGE_TEXT: usize = 64 * 1024;

/// What a `Purge` asks for, before stream names are resolved (the writer resolves them against the catalog).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PurgeAsk {
    /// Everything overlapping `span`, on the named streams (all streams when `streams` is empty).
    Range {
        /// The time to purge.
        span: Interval,
        /// Stream names; empty for every stream.
        streams: Vec<String>,
    },
    /// Everything about an entity.
    Entity(Uuid),
    /// These payloads (or absorbed frames), and everything derived from them.
    Ids(Vec<Uuid>),
    /// These payloads only, committed by `committed_before_ns`: an age-out.
    AgeOut {
        /// The payloads.
        ids: Vec<Uuid>,
        /// The cutoff, UTC nanoseconds, by the commit clock.
        committed_before_ns: u64,
        /// Aged-out spans this close on one stream are recorded as one (minor 29; 0 from an older client).
        aged_gap_ns: u64,
    },
}

/// A `Purge`'s selector and reason, or why it isn't one.
///
/// # Errors
///
/// A message for the client when it names none of an entity, a non-empty range, and ids, or more than one, or its
/// reason or detail is over [`MAX_PURGE_TEXT`] bytes.
pub fn purge_ask(p: &Purge<'_>) -> Result<(PurgeAsk, String, String), String> {
    let reason = p.reason().unwrap_or("").to_owned();
    let detail = p.detail().unwrap_or("").to_owned();
    if reason.len() > MAX_PURGE_TEXT || detail.len() > MAX_PURGE_TEXT {
        return Err(format!("Purge's reason and detail are kept for good: at most {MAX_PURGE_TEXT} bytes each"));
    }
    let entity = p.entity_id().map(|u| Uuid(u.bytes().into()));
    let range = p.range().map(|r| (r.start_ns(), r.end_ns())).filter(|&(a, b)| (a, b) != (0, 0));
    let ids: Vec<Uuid> = p.ids().map(|v| v.iter().map(Uuid::from).collect()).unwrap_or_default();
    let ask = match (entity, range, ids.is_empty()) {
        (Some(e), None, true) => PurgeAsk::Entity(e),
        (None, Some((a, b)), true) => {
            let span = Interval::new(a, b).ok_or_else(|| format!("Purge range [{a}, {b}) is empty"))?;
            let streams = p.streams().map(|v| v.iter().map(str::to_owned).collect()).unwrap_or_default();
            PurgeAsk::Range { span, streams }
        }
        (None, None, false) if p.age_out() => {
            PurgeAsk::AgeOut { ids, committed_before_ns: p.committed_before_ns(), aged_gap_ns: p.aged_gap_ns() }
        }
        _ if p.age_out() => return Err("an age-out Purge names ids only".into()),
        (None, None, false) => PurgeAsk::Ids(ids),
        (None, None, true) => return Err("Purge names neither an entity, a range, nor ids".into()),
        _ => return Err("Purge names more than one of an entity, a range, and ids; send one".into()),
    };
    Ok((ask, reason, detail))
}

/// A `PurgeResult`: how many payloads the purge removed, per stream and tier (a payload with copies in two tiers
/// counts once in each, and one with no copy counts under Tier 0).
/// A report's removed payloads counted per stream and tier (a payload with copies in two tiers counts once in each,
/// and one with no copy counts under Tier 0).
fn removed_counts<'a>(
    b: &mut FlatBufferBuilder<'a>,
    report: &PurgeReport,
) -> WIPOffset<flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<RemovedCount<'a>>>> {
    use std::collections::BTreeMap;
    let mut counts: BTreeMap<(u32, u8), u64> = BTreeMap::new();
    for r in &report.removed {
        if r.locations.is_empty() {
            *counts.entry((r.stream.0, 0)).or_default() += 1;
        }
        for l in &r.locations {
            *counts.entry((r.stream.0, l.tier().0)).or_default() += 1;
        }
    }
    let rows: Vec<_> = counts
        .into_iter()
        .map(|((stream_id, tier), count)| {
            RemovedCount::create(b, &RemovedCountArgs { stream_id, tier: continuum_core::StorageTier(tier), count })
        })
        .collect();
    b.create_vector(&rows)
}

/// A `PurgeResult`: how many payloads the purge removed, per stream and tier, and its record id.
pub fn purge_result(b: &mut FlatBufferBuilder<'static>, request_id: u64, report: &PurgeReport) -> Vec<u8> {
    b.reset();
    let removed = removed_counts(b, report);
    let id = WireUuid::new(&report.purge_id.0);
    let r = PurgeResult::create(b, &PurgeResultArgs { purge_id: Some(&id), removed: Some(removed) });
    finish(b, request_id, ControlPayload::PurgeResult, r.as_union_value()).to_vec()
}

/// A stage's id from its name: the 32-bit FNV-1a hash. Coverage is kept per (source stream, stage) and outlives any
/// one session, so a stage's id has to be the same every time a worker names it; a hash needs no registry, and
/// stages are few enough that a collision is not a practical worry.
#[must_use]
pub fn stage_id(name: &str) -> StageId {
    let mut h: u32 = 0x811c_9dc5;
    for b in name.bytes() {
        h ^= u32::from(b);
        h = h.wrapping_mul(0x0100_0193);
    }
    StageId(h)
}

/// One capability as a `RegisterWorker` declares it, with streams still named (the scheduler task resolves names).
#[derive(Clone, Debug, PartialEq)]
pub struct CapabilityAsk {
    /// Its index in `RegisterWorker.capabilities`.
    pub index: u32,
    /// Produce or hydrate.
    pub role: CapabilityRole,
    /// The output stream's name.
    pub produces: String,
    /// Input stream names, each with its window before and after the output's time.
    pub requires: Vec<(String, u64, u64)>,
    /// Where it computes.
    pub domain: ComputeDomain,
    /// Memory while loaded, in MB.
    pub ram_mb: u64,
    /// GPU memory while loaded, in MB.
    pub vram_mb: u64,
    /// The worker's own first guess per item, in microseconds.
    pub est_us_per_item: u64,
    /// Load time, in ms.
    pub cold_start_ms: u64,
    /// Most items per job.
    pub max_batch: u32,
    /// Tier 0 bytes per output.
    pub output_bytes: u64,
    /// Outputs per job (at least 1).
    pub outputs_per_job: u32,
    /// Whether a lease can be taken back mid-job.
    pub preemptible: bool,
    /// Typical confidence, if any.
    pub expected_confidence: Option<f32>,
}

/// A `RegisterWorker`, decoded.
#[derive(Clone, Debug, PartialEq)]
pub struct WorkerAsk {
    /// The worker's name.
    pub producer: String,
    /// Its code or weights hash.
    pub version_hash: String,
    /// Its GPU, if it names one.
    pub device_index: Option<u8>,
    /// What it can make.
    pub capabilities: Vec<CapabilityAsk>,
    /// It keeps its own copies of payloads, so purges wait for its acknowledgement.
    pub keeps_copies: bool,
}

/// A `RegisterWorker`, or why it isn't one.
///
/// # Errors
///
/// A message for the client when the worker has no name or a capability has an unknown role.
pub fn worker_ask(r: &RegisterWorker<'_>) -> Result<WorkerAsk, String> {
    let producer = r.worker_id().unwrap_or("").to_owned();
    if producer.is_empty() {
        return Err("RegisterWorker needs a worker_id".into());
    }
    let mut capabilities = Vec::new();
    for (i, c) in r.capabilities().into_iter().flatten().enumerate() {
        let role = match c.role() {
            WireRole::Produce => CapabilityRole::Produce,
            WireRole::Hydrate => CapabilityRole::Hydrate,
            other => return Err(format!("capability {i} has an unknown role {}", other.0)),
        };
        let est = c.est_ms_per_item();
        let est_us_per_item = if est.is_finite() && est > 0.0 { (f64::from(est) * 1000.0) as u64 } else { 0 };
        capabilities.push(CapabilityAsk {
            index: u32::try_from(i).unwrap_or(u32::MAX),
            role,
            produces: c.produces().unwrap_or("").to_owned(),
            requires: c
                .requires()
                .into_iter()
                .flatten()
                .map(|q| (q.stream().unwrap_or("").to_owned(), q.before_ns(), q.after_ns()))
                .collect(),
            domain: c.compute_domain(),
            ram_mb: u64::from(c.ram_mb()),
            vram_mb: u64::from(c.vram_mb()),
            est_us_per_item,
            cold_start_ms: u64::from(c.cold_start_ms()),
            max_batch: c.max_batch().max(1),
            output_bytes: c.output_bytes(),
            outputs_per_job: c.outputs_per_job().max(1),
            preemptible: c.preemptible(),
            expected_confidence: c.expected_confidence(),
        });
    }
    Ok(WorkerAsk {
        producer,
        version_hash: r.version_hash().unwrap_or("").to_owned(),
        device_index: r.device_index(),
        capabilities,
        keeps_copies: r.keeps_copies(),
    })
}

/// A `WorkerRegistered`: the accepted indexes with their capability ids and runnability, and a reason per refusal.
pub fn worker_registered(
    b: &mut FlatBufferBuilder<'static>,
    request_id: u64,
    accepted: &[(u32, u32, bool)],
    refused: &[String],
) -> Vec<u8> {
    b.reset();
    let idx: Vec<u32> = accepted.iter().map(|a| a.0).collect();
    let ids: Vec<u32> = accepted.iter().map(|a| a.1).collect();
    let run: Vec<bool> = accepted.iter().map(|a| a.2).collect();
    let reasons: Vec<_> = refused.iter().map(|r| b.create_string(r)).collect();
    let (idx, ids, run) = (b.create_vector(&idx), b.create_vector(&ids), b.create_vector(&run));
    let reasons = b.create_vector(&reasons);
    let w = WorkerRegistered::create(
        b,
        &WorkerRegisteredArgs {
            accepted: Some(idx),
            refused_reasons: Some(reasons),
            capability_ids: Some(ids),
            runnable: Some(run),
        },
    );
    finish(b, request_id, ControlPayload::WorkerRegistered, w.as_union_value()).to_vec()
}

/// A `QuoteRequest` as the scheduler's `LeaseReq`, or why it isn't one.
///
/// # Errors
///
/// A message for the client when the range is missing or empty.
pub fn lease_req(q: &QuoteRequest<'_>) -> Result<LeaseReq, String> {
    let r = q.range().ok_or("QuoteRequest needs a range")?;
    let range = Interval::new(r.start_ns(), r.end_ns()).ok_or("QuoteRequest's range is empty")?;
    Ok(LeaseReq {
        capability: continuum_sched::CapabilityId(q.capability_id()),
        band: q.band(),
        range,
        output_bytes: q.output_bytes(),
        est_duration_us: u64::from(q.est_duration_ms()) * 1000,
        pin_parents: q.pin_parents().into_iter().flatten().map(|u| Uuid(u.bytes().into())).collect(),
    })
}

fn output_lease<'a>(
    b: &mut FlatBufferBuilder<'a>,
    lease_id: u64,
    expires_at_ns: u64,
) -> flatbuffers::WIPOffset<WireOutputLease<'a>> {
    // No slots until continuum-shm joins the core: a lease so far holds compute and memory, not Tier 0 slots.
    let slots = b.create_vector::<flatbuffers::WIPOffset<continuum_core::schema::continuum::ShmLocation>>(&[]);
    WireOutputLease::create(b, &OutputLeaseArgs { lease_id, slots: Some(slots), expires_at_ns })
}

/// A `LeaseGrant` for the scheduler's decision.
pub fn lease_grant(b: &mut FlatBufferBuilder<'static>, request_id: u64, decision: &LeaseDecision) -> Vec<u8> {
    b.reset();
    let args = match decision {
        LeaseDecision::Granted(l) => {
            let lease = output_lease(b, l.lease.0, l.expires_ns);
            LeaseGrantArgs { granted: true, lease: Some(lease), retry_after_ms: 0 }
        }
        LeaseDecision::Denied { retry_after_us, .. } => LeaseGrantArgs {
            granted: false,
            lease: None,
            retry_after_ms: u32::try_from(retry_after_us.div_ceil(1000)).unwrap_or(u32::MAX),
        },
    };
    let g = LeaseGrant::create(b, &args);
    finish(b, request_id, ControlPayload::LeaseGrant, g.as_union_value()).to_vec()
}

/// The reply to a `RenewLease`: a `LeaseGrant` for the same lease with its new expiry.
pub fn lease_renewed(
    b: &mut FlatBufferBuilder<'static>,
    request_id: u64,
    lease_id: u64,
    expires_at_ns: u64,
) -> Vec<u8> {
    b.reset();
    let lease = output_lease(b, lease_id, expires_at_ns);
    let g = LeaseGrant::create(b, &LeaseGrantArgs { granted: true, lease: Some(lease), retry_after_ms: 0 });
    finish(b, request_id, ControlPayload::LeaseGrant, g.as_union_value()).to_vec()
}

/// A `LeaseReleased` answering a `CommitLease` (once its outputs and coverage are durable) or an `AbortLease`.
pub fn lease_released(
    b: &mut FlatBufferBuilder<'static>,
    request_id: u64,
    lease_id: u64,
    output_seqs: &[u64],
) -> Vec<u8> {
    b.reset();
    let output_seqs = b.create_vector(output_seqs);
    let r = LeaseReleased::create(b, &LeaseReleasedArgs { lease_id, output_seqs: Some(output_seqs) });
    finish(b, request_id, ControlPayload::LeaseReleased, r.as_union_value()).to_vec()
}

/// An `Ingested`: the answer to a numbered `IngestNotice` once its payload is committed.
pub fn ingested(b: &mut FlatBufferBuilder<'static>, request_id: u64, stream_id: u32, id: Uuid, seq: u64) -> Vec<u8> {
    b.reset();
    let id = WireUuid::new(&id.0);
    let i = Ingested::create(b, &IngestedArgs { stream_id, payload_id: Some(&id), commit_seq: seq });
    finish(b, request_id, ControlPayload::Ingested, i.as_union_value()).to_vec()
}

/// A `LeasePreempted` the broker sends on its own (request id 0).
pub fn lease_preempted(
    b: &mut FlatBufferBuilder<'static>,
    lease_id: u64,
    reason: &str,
    by_band: PriorityBand,
) -> Vec<u8> {
    b.reset();
    let reason = b.create_string(reason);
    let p = LeasePreempted::create(b, &LeasePreemptedArgs { lease_id, reason: Some(reason), by_band });
    finish(b, 0, ControlPayload::LeasePreempted, p.as_union_value()).to_vec()
}

/// A `SystemStateChanged` broadcast (request id 0).
pub fn system_state_changed(
    b: &mut FlatBufferBuilder<'static>,
    voice_turn: VoiceTurnState,
    changed_at_ns: u64,
) -> Vec<u8> {
    b.reset();
    let s = SystemStateChanged::create(b, &SystemStateChangedArgs { voice_turn, changed_at_ns });
    finish(b, 0, ControlPayload::SystemStateChanged, s.as_union_value()).to_vec()
}

/// A `CommitLease`, decoded: its lease, the outputs (headers checked), the coverage it finished, and its cost.
#[derive(Debug)]
pub struct Commit {
    /// The lease.
    pub lease: u64,
    /// The outputs, with where their bytes are.
    pub outputs: Vec<(HeaderBuf, Option<StorageLocator>)>,
    /// The (source stream, stage, span) it finished, if any.
    pub coverage: Option<(continuum_core::StreamId, StageId, Interval)>,
    /// What it cost.
    pub cost: ComputeCost,
}

/// A `CommitLease`, or why it isn't one.
///
/// # Errors
///
/// A message for the client when an output's header or locator is bad, or the coverage range is empty.
pub fn commit(c: &CommitLease<'_>) -> Result<Commit, String> {
    let mut outputs = Vec::new();
    for (i, item) in c.outputs().into_iter().flatten().enumerate() {
        let bytes = item.header_bytes().ok_or_else(|| format!("output {i} has no header_bytes"))?;
        let header = HeaderBuf::new(bytes.bytes().to_vec()).map_err(|e| format!("output {i}'s header: {e}"))?;
        let locator = item
            .locator()
            .map(|l| StorageLocator::from_table(&l))
            .transpose()
            .map_err(|e| format!("output {i}'s locator: {e}"))?;
        outputs.push((header, locator));
    }
    let coverage = match c.coverage() {
        Some(cov) => {
            let r = cov.range().ok_or("coverage without a range")?;
            let span = Interval::new(r.start_ns(), r.end_ns()).ok_or("coverage range is empty")?;
            Some((continuum_core::StreamId(cov.source_stream_id()), stage_id(cov.stage().unwrap_or("")), span))
        }
        None => None,
    };
    let none = || ComputeCost {
        wall_time_us: 0,
        queue_wait_us: 0,
        ram_bytes: 0,
        vram_bytes: 0,
        device: String::new(),
        items: 0,
    };
    let cost = c.cost().map_or_else(none, |k| ComputeCost {
        wall_time_us: k.wall_time_us(),
        queue_wait_us: k.queue_wait_us(),
        ram_bytes: k.ram_bytes(),
        vram_bytes: k.vram_bytes(),
        device: k.device().unwrap_or("").to_owned(),
        items: k.items(),
    });
    Ok(Commit { lease: c.lease_id(), outputs, coverage, cost })
}

// ---- The consumer flow: queries, quotes, promises, and the jobs that serve them ------------------------------------

/// A `Query`, decoded.
#[derive(Clone, Debug, PartialEq)]
pub struct QueryAsk {
    /// The stream wanted, by name.
    pub stream: String,
    /// The time wanted (always given for a time query; a filter for a sequence query).
    pub range: Option<Interval>,
    /// The latency the consumer can accept, in microseconds, if it gave one.
    pub budget_us: Option<u64>,
    /// The band the work would run in.
    pub band: PriorityBand,
    /// What to do about missing data.
    pub mode: QueryMode,
    /// At most this many items in the `QueryResult` (0 for the broker's default).
    pub max_items: u32,
    /// A sequence query (see `Query.by_seq` in the schema), else `None`.
    pub by_seq: Option<SeqQuery>,
}

/// The sequence half of a `Query`: commit order after a position, filtered.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeqQuery {
    /// Payloads with a commit sequence above this (0 for all).
    pub after_seq: u64,
    /// What the catalog keeps.
    pub filter: continuum_catalog::QueryFilter,
    /// Rows scanned at most (0 for the broker's default).
    pub max_scan: u32,
}

/// Decodes a `Query`.
///
/// # Errors
///
/// A message for the client when the stream name or the range is missing or the range is empty.
pub fn query_ask(q: &WireQuery<'_>) -> Result<QueryAsk, String> {
    let stream = q.stream().unwrap_or("").to_owned();
    if stream.is_empty() {
        return Err("Query needs a stream".into());
    }
    let range = match q.range() {
        Some(r) if r.start_ns() == 0 && r.end_ns() == 0 && q.by_seq() => None,
        Some(r) => Some(Interval::new(r.start_ns(), r.end_ns()).ok_or("Query's range is empty")?),
        None if q.by_seq() => None,
        None => return Err("Query needs a range".into()),
    };
    let mode = match q.mode() {
        WireQueryMode::HitOnly => QueryMode::HitOnly,
        WireQueryMode::QuoteIfMissing => QueryMode::QuoteIfMissing,
        WireQueryMode::CommitWithinBudget => QueryMode::CommitWithinBudget,
        other => return Err(format!("unknown query mode {}", other.0)),
    };
    let budget_us = (q.latency_budget_ms() > 0).then(|| u64::from(q.latency_budget_ms()) * 1000);
    let by_seq = if q.by_seq() {
        if mode != QueryMode::HitOnly {
            return Err("a sequence query makes nothing: its mode must be HitOnly".into());
        }
        let uuids = |v: Option<flatbuffers::Vector<'_, WireUuid>>| {
            v.map(|v| v.iter().map(Uuid::from).collect::<Vec<_>>()).filter(|v| !v.is_empty())
        };
        let filter = continuum_catalog::QueryFilter {
            subjects: uuids(q.subjects()),
            start_before: (q.start_before() > 0).then_some(q.start_before()),
            ids: uuids(q.ids()),
            window: None,
        };
        Some(SeqQuery { after_seq: q.after_seq(), filter, max_scan: q.max_scan() })
    } else {
        // The filters and the scan cap belong to a sequence query; a time query goes through the engine, which has no
        // use for them, so they are refused rather than silently ignored (everything in the range coming back).
        let given = [
            (q.subjects().is_some_and(|v| !v.is_empty()), "subjects"),
            (q.ids().is_some_and(|v| !v.is_empty()), "ids"),
            (q.start_before() > 0, "start_before"),
            (q.after_seq() > 0, "after_seq"),
            (q.max_scan() > 0, "max_scan"),
        ];
        if let Some((_, name)) = given.iter().find(|(set, _)| *set) {
            return Err(format!("{name} filters a sequence query: set by_seq (a time query returns the whole range)"));
        }
        None
    };
    Ok(QueryAsk { stream, range, budget_us, band: q.band(), mode, max_items: q.max_items(), by_seq })
}

/// Promise and job ids are the scheduler's `u64`s; on the wire they are UUIDs. The first eight bytes name the kind,
/// the last eight carry the id, big-endian, so an id round-trips and a promise id can't be mistaken for a job's.
const PROMISE_TAG: [u8; 8] = *b"cprm\0\0\0\0";
const JOB_TAG: [u8; 8] = *b"cjob\0\0\0\0";

fn tagged(tag: [u8; 8], id: u64) -> Uuid {
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&tag);
    bytes[8..].copy_from_slice(&id.to_be_bytes());
    Uuid(bytes)
}

fn untag(tag: [u8; 8], id: Uuid) -> Option<u64> {
    (id.0[..8] == tag).then(|| u64::from_be_bytes(id.0[8..].try_into().expect("8 bytes")))
}

/// A promise's wire id.
#[must_use]
pub fn promise_uuid(id: PromiseId) -> Uuid {
    tagged(PROMISE_TAG, id.0)
}

/// The promise a wire id names, if it's one.
#[must_use]
pub fn promise_of(id: Uuid) -> Option<PromiseId> {
    untag(PROMISE_TAG, id).map(PromiseId)
}

/// A job's wire id.
#[must_use]
pub fn job_uuid(id: JobId) -> Uuid {
    tagged(JOB_TAG, id.0)
}

/// The job a wire id names, if it's one.
#[must_use]
pub fn job_of(id: Uuid) -> Option<JobId> {
    untag(JOB_TAG, id).map(JobId)
}

/// Decodes a `CommitPromise`: the promise, the option chosen (0 is the quote's own plan), and the band.
///
/// # Errors
///
/// A message for the client when the promise id is missing or isn't a promise's.
pub fn commit_promise_ask(c: &CommitPromise<'_>) -> Result<(PromiseId, u32, PriorityBand), String> {
    let id = c.promise_id().ok_or("CommitPromise needs a promise_id")?;
    let promise = promise_of(Uuid(id.bytes().into())).ok_or("CommitPromise's promise_id is not a promise's")?;
    Ok((promise, c.option(), c.band()))
}

/// Decodes a `CancelPromise`.
///
/// # Errors
///
/// A message for the client when the promise id is missing or isn't a promise's.
pub fn cancel_promise_ask(c: &CancelPromise<'_>) -> Result<PromiseId, String> {
    let id = c.promise_id().ok_or("CancelPromise needs a promise_id")?;
    promise_of(Uuid(id.bytes().into())).ok_or_else(|| "CancelPromise's promise_id is not a promise's".to_owned())
}

/// Decodes a `JobRejected`: the job and the reason.
///
/// # Errors
///
/// A message for the client when the job id is missing or isn't a job's.
pub fn job_rejected_ask(j: &JobRejected<'_>) -> Result<(JobId, String), String> {
    let id = j.job_id().ok_or("JobRejected needs a job_id")?;
    let job = job_of(Uuid(id.bytes().into())).ok_or("JobRejected's job_id is not a job's")?;
    Ok((job, j.reason().unwrap_or("").to_owned()))
}

/// A payload as a `QueryResult`, `FulfillPromise`, or `DispatchJob` lists it: its header and where its bytes are.
#[derive(Clone, Debug, PartialEq)]
pub struct WireItem {
    /// The header, as the catalog keeps it.
    pub header: HeaderBuf,
    /// The copy to read, if it has one.
    pub locator: Option<StorageLocator>,
    /// Its commit sequence (0 when unknown).
    pub seq: u64,
}

fn put_items<'a>(
    b: &mut FlatBufferBuilder<'a>,
    items: &[WireItem],
) -> WIPOffset<flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<Item<'a>>>> {
    let rows: Vec<_> = items
        .iter()
        .map(|it| {
            let header_bytes = b.create_vector(it.header.as_bytes());
            let locator = it.locator.as_ref().map(|l| l.build_into(b));
            Item::create(b, &ItemArgs { locator, header_bytes: Some(header_bytes), commit_seq: it.seq })
        })
        .collect();
    b.create_vector(&rows)
}

fn time_range(i: Interval) -> TimeRange {
    TimeRange::new(i.start_ns(), i.end_ns())
}

/// How a `QueryResult` says where it stands: whether the items cover the whole range, where a sequence query
/// continues, and whether the page was cut short (at `max_items` or the inline byte cap) with more after it.
#[derive(Clone, Copy, Debug)]
pub struct ResultPaging {
    /// The items cover the whole range (no gaps, no unsealed tail, nothing cut).
    pub complete: bool,
    /// A sequence query's position to continue from; 0 when nothing remains.
    pub next_after: u64,
    /// The page was cut short: more may follow the last item.
    pub truncated: bool,
    /// A `PromiseQuote` follows with the same request id.
    pub quoted: bool,
}

/// A `QueryResult`: what exists of the range (pinned for reading where it's in Tier 0), whether that's all of it,
/// and what's missing.
pub fn query_result(
    b: &mut FlatBufferBuilder<'static>,
    request_id: u64,
    items: &[WireItem],
    paging: ResultPaging,
    missing: &[Interval],
    tail: Option<&Tail<'_>>,
) -> Vec<u8> {
    use continuum_core::schema::continuum::{RingTail, RingTailArgs};
    let ResultPaging { complete, next_after, truncated, quoted } = paging;
    b.reset();
    let items = put_items(b, items);
    let missing: Vec<TimeRange> = missing.iter().map(|&i| time_range(i)).collect();
    let missing = b.create_vector(&missing);
    let tail = tail.map(|t| {
        let arena = b.create_string(&t.arena.name);
        RingTail::create(
            b,
            &RingTailArgs {
                arena: Some(arena),
                ring_offset: t.ring.ctrl_offset,
                first_seq: t.first_seq,
                last_seq: t.last_seq,
                ring: t.index,
                arena_generation: t.arena.generation,
            },
        )
    });
    let r = QueryResult::create(
        b,
        &QueryResultArgs { items: Some(items), complete, missing: Some(missing), tail, next_after, truncated, quoted },
    );
    finish(b, request_id, ControlPayload::QueryResult, r.as_union_value()).to_vec()
}

/// A ring stream's unsealed tail in a `QueryResult`: records `first_seq` to `last_seq` of its ring.
#[derive(Clone, Copy, Debug)]
pub struct Tail<'a> {
    /// The arena the ring is in.
    pub arena: &'a crate::tier0::ArenaSummary,
    /// The ring's index.
    pub index: u16,
    /// Its layout.
    pub ring: continuum_shm::CarvedRing,
    /// The first record past the newest indexed batch (or the oldest still in the ring).
    pub first_seq: u64,
    /// The newest record written.
    pub last_seq: u64,
}

/// Names the broker fills into a quote: a stream's name and a worker's name, for the plan steps.
pub struct PlanNames<'n> {
    /// A stream's registered name (a hydrate step's stream can be any stream).
    pub stream: &'n dyn Fn(StreamId) -> String,
    /// A worker's name from its manifest.
    pub worker: &'n dyn Fn(WorkerId) -> String,
}

fn put_plan<'a>(
    b: &mut FlatBufferBuilder<'a>,
    plan: &Plan,
    est_latency_us: u64,
    cost_units: f64,
    exceeds_budget: bool,
    expected_confidence: Option<f32>,
    names: &PlanNames<'_>,
) -> WIPOffset<PlanOption<'a>> {
    let steps: Vec<_> = plan
        .steps
        .iter()
        .map(|s| {
            let (read_tier, worker) = match s.kind {
                StepKind::Capture => (StorageTier::Tier0Hot, None),
                StepKind::Hydrate { from } => (from, s.worker),
                StepKind::Compute => (StorageTier::Tier0Hot, s.worker),
            };
            let worker_id = b.create_string(&worker.map(|w| (names.worker)(w)).unwrap_or_default());
            let produces = b.create_string(&(names.stream)(s.stream));
            WirePlanStep::create(
                b,
                &PlanStepArgs {
                    read_tier,
                    worker_id: Some(worker_id),
                    produces: Some(produces),
                    items: u32::try_from(s.items).unwrap_or(u32::MAX),
                    est_ms: s.est_us as f32 / 1000.0,
                },
            )
        })
        .collect();
    let steps = b.create_vector(&steps);
    PlanOption::create(
        b,
        &PlanOptionArgs {
            steps: Some(steps),
            est_latency_ms: est_latency_us as f32 / 1000.0,
            cost_units: cost_units as f32,
            exceeds_budget,
            expected_confidence,
        },
    )
}

/// A `PromiseQuote` for the scheduler's quote: the quote's own plan first, then its alternatives.
pub fn promise_quote(
    b: &mut FlatBufferBuilder<'static>,
    request_id: u64,
    quote: &Quote,
    names: &PlanNames<'_>,
) -> Vec<u8> {
    b.reset();
    let mut options =
        vec![put_plan(b, &quote.plan, quote.est_latency_us, quote.cost_units, quote.exceeds_budget, None, names)];
    for a in &quote.alternatives {
        options.push(put_plan(
            b,
            &a.plan,
            a.est_latency_us,
            a.cost_units,
            a.exceeds_budget,
            a.expected_confidence,
            names,
        ));
    }
    let options = b.create_vector(&options);
    let status = match quote.status {
        QuoteStatus::RequiresCompute => WirePromiseStatus::RequiresCompute,
        QuoteStatus::AwaitingCapture => WirePromiseStatus::AwaitingCapture,
        QuoteStatus::AwaitingCaptureAndCompute => WirePromiseStatus::AwaitingCaptureAndCompute,
    };
    let promise_id = WireUuid::new(&promise_uuid(quote.promise).0);
    let q = PromiseQuote::create(
        b,
        &PromiseQuoteArgs {
            promise_id: Some(&promise_id),
            status,
            options: Some(options),
            shares_in_flight: quote.shares_in_flight,
            expires_at_ns: quote.expires_ns,
        },
    );
    finish(b, request_id, ControlPayload::PromiseQuote, q.as_union_value()).to_vec()
}

/// A `DispatchJob` for a worker (request id 0: the broker's own message), with the inputs resolved and pinned.
pub fn dispatch_job(
    b: &mut FlatBufferBuilder<'static>,
    d: &Dispatch,
    inputs: &[WireItem],
    capability: &str,
) -> Vec<u8> {
    b.reset();
    let promise_ids: Vec<WireUuid> = d.promises.iter().map(|&p| WireUuid::new(&promise_uuid(p).0)).collect();
    let promise_ids = b.create_vector(&promise_ids);
    let capability_name = b.create_string(capability);
    let inputs = put_items(b, inputs);
    let output = output_lease(b, d.output.lease.0, d.output.expires_ns);
    let job_id = WireUuid::new(&job_uuid(d.job).0);
    let output_range = time_range(d.range);
    let j = DispatchJob::create(
        b,
        &DispatchJobArgs {
            job_id: Some(&job_id),
            promise_ids: Some(promise_ids),
            capability: Some(capability_name),
            inputs: Some(inputs),
            output_range: Some(&output_range),
            band: d.band,
            deadline_ns: d.deadline_ns,
            output: Some(output),
            capability_id: d.capability.0,
        },
    );
    finish(b, 0, ControlPayload::DispatchJob, j.as_union_value()).to_vec()
}

/// A `FulfillPromise` (request id 0) with one part of the results.
pub fn fulfill_promise(
    b: &mut FlatBufferBuilder<'static>,
    promise: PromiseId,
    items: &[WireItem],
    final_part: bool,
) -> Vec<u8> {
    b.reset();
    let items = put_items(b, items);
    let promise_id = WireUuid::new(&promise_uuid(promise).0);
    let f = FulfillPromise::create(
        b,
        &FulfillPromiseArgs { promise_id: Some(&promise_id), items: Some(items), final_part },
    );
    finish(b, 0, ControlPayload::FulfillPromise, f.as_union_value()).to_vec()
}

/// The answer to a `CommitPromise` or `CancelPromise`.
pub fn promise_accepted(
    b: &mut FlatBufferBuilder<'static>,
    request_id: u64,
    promise: PromiseId,
    life: PromiseLife,
) -> Vec<u8> {
    b.reset();
    let promise_id = WireUuid::new(&promise_uuid(promise).0);
    let a = PromiseAccepted::create(b, &PromiseAcceptedArgs { promise_id: Some(&promise_id), life });
    finish(b, request_id, ControlPayload::PromiseAccepted, a.as_union_value()).to_vec()
}

/// A `PromiseFailed` (request id 0).
pub fn promise_failed(b: &mut FlatBufferBuilder<'static>, promise: PromiseId, reason: &str, delivered: u32) -> Vec<u8> {
    b.reset();
    let reason = b.create_string(reason);
    let promise_id = WireUuid::new(&promise_uuid(promise).0);
    let f =
        PromiseFailed::create(b, &PromiseFailedArgs { promise_id: Some(&promise_id), reason: Some(reason), delivered });
    finish(b, 0, ControlPayload::PromiseFailed, f.as_union_value()).to_vec()
}

#[cfg(test)]
mod purge_pages {
    use super::*;

    #[test]
    fn a_purge_list_stops_before_its_texts_pass_the_byte_cap_and_keeps_at_least_one() {
        let summary = |n: u8, detail_len: usize| PurgeSummary {
            purge_id: Uuid([n; 16]),
            at_ns: 1,
            kind: PurgeKind::Entity,
            reason: "r".into(),
            detail: "d".repeat(detail_len),
            removed: 0,
            finished: true,
            completed_ns: None,
        };
        let mut b = FlatBufferBuilder::with_capacity(1024);
        let third = MAX_PURGE_LIST_BYTES / 3 + 1;
        let three = [summary(1, third), summary(2, third), summary(3, third)];
        let bytes = purge_list(&mut b, 9, &three);
        let listed = continuum_wire::read_envelope(&bytes).expect("envelope").payload_as_purge_list().expect("list");
        assert_eq!(listed.purges().map_or(0, |p| p.len()), 2, "the third would pass the cap");
        let huge = [summary(4, MAX_PURGE_LIST_BYTES + 1)];
        let bytes = purge_list(&mut b, 10, &huge);
        let listed = continuum_wire::read_envelope(&bytes).expect("envelope").payload_as_purge_list().expect("list");
        assert_eq!(listed.purges().map_or(0, |p| p.len()), 1, "one record always goes, however large");
        assert!(!purge_list(&mut b, 11, &[]).is_empty());
    }
}

#[cfg(test)]
mod promise_ids {
    use super::*;

    #[test]
    fn promise_and_job_ids_round_trip_and_never_cross() {
        let p = promise_uuid(PromiseId(0x1234_5678_9abc_def0));
        assert_eq!(promise_of(p), Some(PromiseId(0x1234_5678_9abc_def0)));
        assert_eq!(job_of(p), None);
        let j = job_uuid(JobId(7));
        assert_eq!(job_of(j), Some(JobId(7)));
        assert_eq!(promise_of(j), None);
        assert_eq!(promise_of(Uuid([0xff; 16])), None);
    }
}

/// Where a subscription starts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SubscribeStart {
    /// What's committed from now on.
    Live,
    /// Payloads starting at or after this time, UTC nanoseconds, then live.
    Time(u64),
    /// After the position saved under the subscription's durable name (everything, when none is saved).
    Saved,
}

/// A decoded `Subscribe`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubscribeAsk {
    /// The client's number for this subscription (unique within its session).
    pub id: u64,
    /// The streams to follow, by name.
    pub streams: Vec<String>,
    /// Where to start.
    pub start: SubscribeStart,
    /// The name its position is saved under, for a durable subscription.
    pub durable_name: Option<String>,
    /// Whether events carry a locator, not only the header.
    pub pinned: bool,
    /// Only payloads with one of these subjects, when set.
    pub subjects: Option<Vec<Uuid>>,
    /// Only payloads with this shape name, when set.
    pub shape_name: Option<String>,
}

/// Decodes a `Subscribe`.
///
/// # Errors
/// What is wrong with it: no streams, an empty stream name, an unknown start, a time start without a time, or a saved
/// start without a durable name.
pub fn subscribe_ask(s: &WireSubscribe<'_>) -> Result<SubscribeAsk, String> {
    let streams: Vec<String> = s.streams().into_iter().flatten().map(str::to_owned).collect();
    if streams.is_empty() {
        return Err("Subscribe needs at least one stream".into());
    }
    if streams.iter().any(String::is_empty) {
        return Err("Subscribe names an empty stream".into());
    }
    let durable_name = s.durable_name().filter(|n| !n.is_empty()).map(str::to_owned);
    let start = match s.start() {
        StartAt::Live => SubscribeStart::Live,
        StartAt::Time if s.start_ns() == 0 => return Err("a Subscribe from a time needs start_ns".into()),
        StartAt::Time => SubscribeStart::Time(s.start_ns()),
        StartAt::Saved if durable_name.is_none() => {
            return Err("a Subscribe from a saved position needs a durable_name".into());
        }
        StartAt::Saved => SubscribeStart::Saved,
        other => return Err(format!("Subscribe has an unknown start {}", other.0)),
    };
    let subjects = s.subjects().map(|v| v.iter().map(Uuid::from).collect::<Vec<_>>()).filter(|v| !v.is_empty());
    let shape_name = s.shape_name().filter(|n| !n.is_empty()).map(str::to_owned);
    Ok(SubscribeAsk { id: s.subscription_id(), streams, start, durable_name, pinned: s.pinned(), subjects, shape_name })
}

/// A `Subscribed`: the position the subscription starts after, and whether it is live already.
pub fn subscribed(
    b: &mut FlatBufferBuilder<'static>,
    request_id: u64,
    id: u64,
    position: u64,
    live: bool,
    rings: &[(u32, RingGrant<'_>)],
) -> Vec<u8> {
    use continuum_core::schema::continuum::{StreamRing, StreamRingArgs};
    b.reset();
    let rings: Vec<_> = rings
        .iter()
        .map(|&(stream_id, g)| {
            let lease = ring_lease(b, g);
            StreamRing::create(b, &StreamRingArgs { stream_id, lease: Some(lease) })
        })
        .collect();
    let rings = (!rings.is_empty()).then(|| b.create_vector(&rings));
    let s = Subscribed::create(b, &SubscribedArgs { subscription_id: id, position, live, rings });
    finish(b, request_id, ControlPayload::Subscribed, s.as_union_value()).to_vec()
}

/// A `StreamEvent`: one payload at `position` (its commit sequence) for subscription `id`. Sent on the push path, so
/// its request id is 0.
pub fn stream_event(b: &mut FlatBufferBuilder<'static>, id: u64, position: u64, item: &WireItem) -> Vec<u8> {
    b.reset();
    let header_bytes = b.create_vector(item.header.as_bytes());
    let locator = item.locator.as_ref().map(|l| l.build_into(b));
    let item = Item::create(b, &ItemArgs { locator, header_bytes: Some(header_bytes), commit_seq: item.seq });
    let e = StreamEvent::create(b, &StreamEventArgs { subscription_id: id, position, item: Some(item) });
    finish(b, 0, ControlPayload::StreamEvent, e.as_union_value()).to_vec()
}

/// A `CaughtUp`: subscription `id` has sent everything up to `position`; what follows is live. Push path, id 0.
pub fn caught_up(b: &mut FlatBufferBuilder<'static>, id: u64, position: u64) -> Vec<u8> {
    b.reset();
    let c = CaughtUp::create(b, &CaughtUpArgs { subscription_id: id, position });
    finish(b, 0, ControlPayload::CaughtUp, c.as_union_value()).to_vec()
}

/// An `Unsubscribed`.
pub fn unsubscribed(b: &mut FlatBufferBuilder<'static>, request_id: u64, id: u64) -> Vec<u8> {
    b.reset();
    let u = Unsubscribed::create(b, &UnsubscribedArgs { subscription_id: id });
    finish(b, request_id, ControlPayload::Unsubscribed, u.as_union_value()).to_vec()
}

/// One page of a `PurgeNotice` to a worker that keeps its own copies: these payload ids, the spans removed per
/// stream (on the first page), and whether more pages follow. Push path, so its request id is 0.
pub fn purge_notice(
    b: &mut FlatBufferBuilder<'static>,
    purge_id: Uuid,
    ids: &[Uuid],
    spans: &[(StreamId, Interval)],
    more: bool,
) -> Vec<u8> {
    b.reset();
    let ids: Vec<WireUuid> = ids.iter().map(|u| WireUuid::new(&u.0)).collect();
    let ids = b.create_vector(&ids);
    let rows: Vec<_> = spans
        .iter()
        .map(|&(stream, span)| {
            let range = time_range(span);
            PurgedSpan::create(b, &PurgedSpanArgs { stream_id: stream.0, range: Some(&range) })
        })
        .collect();
    let spans = b.create_vector(&rows);
    let id = WireUuid::new(&purge_id.0);
    let n = PurgeNotice::create(
        b,
        &PurgeNoticeArgs { purge_id: Some(&id), payload_ids: Some(ids), spans: Some(spans), more },
    );
    finish(b, 0, ControlPayload::PurgeNotice, n.as_union_value()).to_vec()
}

/// What a `CommitChunk` asks: the chunk (its header, location, and frame index) and the Tier 0 payloads it absorbed,
/// in frame order.
#[derive(Clone, Debug)]
pub struct ChunkCommitAsk {
    /// The chunk, as the catalog takes it.
    pub chunk: continuum_catalog::NewChunk,
    /// The absorbed payloads, in the same order as the frames.
    pub absorbed: Vec<Uuid>,
    /// What capture attested about each frame (minor 27), or `None` when the worker sent none (version 0).
    pub provenance: Option<continuum_catalog::ChunkProvenance>,
}

/// Decode a nonabsorbing durable copy request. The writer validates the original payload and existing copies.
/// # Errors
/// Missing identity/location or a locator that is not a nonempty Tier 1 chunk range.
pub fn payload_copy_ask(
    c: &continuum_core::schema::continuum::CommitPayloadCopy<'_>,
) -> Result<(Uuid, StorageLocator), String> {
    let id = c.payload_id().map(|id| Uuid(id.0)).ok_or("CommitPayloadCopy without payload_id")?;
    let location = c.location().ok_or("CommitPayloadCopy without location")?;
    let location = StorageLocator::from_table(&location).map_err(|e| format!("location: {e}"))?;
    match &location {
        StorageLocator::Chunk { tier: StorageTier::Tier1Warm, uri, offset, length }
            if !uri.is_empty() && *length > 0 && offset.checked_add(*length).is_some() =>
        {
            Ok((id, location))
        }
        _ => Err("payload copy must be a nonempty Tier1Warm chunk range".into()),
    }
}

/// A derived payload's durable copy has been recorded without changing its commit sequence.
pub fn payload_copy_committed(b: &mut FlatBufferBuilder<'static>, request_id: u64, payload: Uuid) -> Vec<u8> {
    use continuum_core::schema::continuum::{PayloadCopyCommitted, PayloadCopyCommittedArgs};
    b.reset();
    let id = WireUuid(payload.0);
    let body = PayloadCopyCommitted::create(b, &PayloadCopyCommittedArgs { payload_id: Some(&id) });
    finish(b, request_id, ControlPayload::PayloadCopyCommitted, body.as_union_value()).to_vec()
}

fn chunk_frames(
    frames: Option<flatbuffers::Vector<'_, continuum_core::schema::continuum::ChunkFrame>>,
) -> Vec<continuum_catalog::ChunkFrame> {
    frames
        .map(|v| {
            v.iter()
                .map(|f| continuum_catalog::ChunkFrame {
                    start_ns: f.start_ns(),
                    offset: f.offset(),
                    length: f.length(),
                    cut: f.cut(),
                })
                .collect()
        })
        .unwrap_or_default()
}

/// A `CommitChunk`'s chunk and absorbed ids, or why it isn't one. The catalog checks the rest (the frames against
/// the absorbed payloads, the span, the tier).
///
/// # Errors
///
/// A message for the worker when the header bytes don't verify, there is no location, the frame and absorbed
/// counts differ, or the frame provenance has an unknown version or flag or a different count than the frames.
pub fn commit_chunk_ask(c: &continuum_core::schema::continuum::CommitChunk<'_>) -> Result<ChunkCommitAsk, String> {
    let bytes = c.header_bytes().ok_or("CommitChunk without header_bytes")?;
    let header = HeaderBuf::new(bytes.bytes().to_vec()).map_err(|e| format!("header_bytes: {e}"))?;
    let location = c.location().ok_or("CommitChunk without a location")?;
    let location = StorageLocator::from_table(&location).map_err(|e| format!("location: {e}"))?;
    let frames = chunk_frames(c.frames());
    let absorbed: Vec<Uuid> = c.absorbed().map(|v| v.iter().map(|u| Uuid(u.0)).collect()).unwrap_or_default();
    if frames.len() != absorbed.len() {
        return Err(format!("CommitChunk has {} frames and {} absorbed ids", frames.len(), absorbed.len()));
    }
    let provenance = match (c.provenance_version(), c.frame_provenance()) {
        (0, None) => None,
        (0, Some(rows)) if rows.is_empty() => None,
        (0, Some(_)) => return Err("CommitChunk has frame_provenance without a provenance_version".into()),
        (version, rows) => {
            let rows = rows
                .map(|v| {
                    v.iter()
                        .map(|f| continuum_catalog::FrameProvenance::from_wire(f.device_seq(), f.epoch(), f.flags()))
                        .collect::<Result<Vec<_>, _>>()
                })
                .transpose()
                .map_err(|e| format!("frame_provenance: {e}"))?
                .unwrap_or_default();
            let p = continuum_catalog::ChunkProvenance { version, frames: rows };
            p.validate(frames.len()).map_err(|e| format!("frame_provenance: {e}"))?;
            Some(p)
        }
    };
    let payload = continuum_catalog::NewPayload { header, location: Some(location), coverage: None };
    Ok(ChunkCommitAsk { chunk: continuum_catalog::NewChunk { payload, frames }, absorbed, provenance })
}

/// A `RewriteChunk`'s chunk id, new frame index, and new location, or why it isn't one.
///
/// # Errors
///
/// A message for the worker when the chunk id or location is missing or the location doesn't read.
pub fn rewrite_chunk_ask(
    r: &continuum_core::schema::continuum::RewriteChunk<'_>,
) -> Result<(Uuid, Vec<continuum_catalog::ChunkFrame>, StorageLocator), String> {
    let chunk = r.chunk_id().map(|u| Uuid(u.0)).ok_or("RewriteChunk without a chunk_id")?;
    let location = r.location().ok_or("RewriteChunk without a location")?;
    let location = StorageLocator::from_table(&location).map_err(|e| format!("location: {e}"))?;
    Ok((chunk, chunk_frames(r.frames()), location))
}

/// The reply to a `CommitChunk`.
pub fn chunk_committed(
    b: &mut FlatBufferBuilder<'static>,
    request_id: u64,
    chunk_id: Uuid,
    inserted: &continuum_catalog::Inserted,
) -> Vec<u8> {
    use continuum_core::schema::continuum::{ChunkCommitted, ChunkCommittedArgs};
    b.reset();
    let id = WireUuid(chunk_id.0);
    let r = ChunkCommitted::create(
        b,
        &ChunkCommittedArgs {
            chunk_id: Some(&id),
            commit_seq: inserted.seq,
            duplicate: inserted.duplicate,
            hidden: inserted.hidden,
        },
    );
    finish(b, request_id, ControlPayload::ChunkCommitted, r.as_union_value()).to_vec()
}

/// The reply to a `RewriteChunk`.
pub fn chunk_rewritten(b: &mut FlatBufferBuilder<'static>, request_id: u64, chunk_id: Uuid) -> Vec<u8> {
    use continuum_core::schema::continuum::{ChunkRewritten, ChunkRewrittenArgs};
    b.reset();
    let id = WireUuid(chunk_id.0);
    let r = ChunkRewritten::create(b, &ChunkRewrittenArgs { chunk_id: Some(&id) });
    finish(b, request_id, ControlPayload::ChunkRewritten, r.as_union_value()).to_vec()
}

/// A `ListPurges`, decoded: the purge to continue after, and the limit (0 for the broker's default).
pub fn list_purges_ask(l: &ListPurges<'_>) -> (Option<Uuid>, u32) {
    (l.after().map(|u| Uuid(u.0)).filter(|u| u.0 != [0; 16]), l.limit())
}

/// A `ForgetKeeper`, decoded: the keeper's name, `None` when missing or empty.
pub fn forget_keeper_ask(f: &ForgetKeeper<'_>) -> Option<String> {
    f.name().filter(|n| !n.is_empty()).map(str::to_owned)
}

/// A `KeeperForgotten`.
pub fn keeper_forgotten(b: &mut FlatBufferBuilder<'static>, request_id: u64, name: &str, released: u32) -> Vec<u8> {
    b.reset();
    let name = b.create_string(name);
    let k = KeeperForgotten::create(b, &KeeperForgottenArgs { name: Some(name), released });
    finish(b, request_id, ControlPayload::KeeperForgotten, k.as_union_value()).to_vec()
}

/// A forgotten name's hash off the wire.
#[must_use]
pub fn forgotten_hash(h: &WireForgottenHash) -> continuum_catalog::ForgottenHash {
    continuum_catalog::ForgottenHash(h.0)
}

/// A `ForgottenRemembered`.
pub fn forgotten_remembered(b: &mut FlatBufferBuilder<'static>, request_id: u64) -> Vec<u8> {
    b.reset();
    let f = ForgottenRemembered::create(b, &ForgottenRememberedArgs {});
    finish(b, request_id, ControlPayload::ForgottenRemembered, f.as_union_value()).to_vec()
}

/// A bounded, opaque forgotten-hash page.
pub fn forgotten_list(
    b: &mut FlatBufferBuilder<'static>,
    request_id: u64,
    page: &continuum_catalog::ForgottenPage,
) -> Vec<u8> {
    use continuum_core::schema::continuum::{ForgottenList, ForgottenListArgs};
    b.reset();
    let hashes: Vec<_> = page.hashes.iter().map(|hash| WireForgottenHash(hash.0)).collect();
    let hashes = b.create_vector(&hashes);
    let next = page.next.map(|hash| WireForgottenHash(hash.0));
    let result = ForgottenList::create(b, &ForgottenListArgs { hashes: Some(hashes), next: next.as_ref() });
    finish(b, request_id, ControlPayload::ForgottenList, result.as_union_value()).to_vec()
}

/// A `ForgottenStatus`.
pub fn forgotten_status(b: &mut FlatBufferBuilder<'static>, request_id: u64, forgotten: bool) -> Vec<u8> {
    b.reset();
    let f = ForgottenStatus::create(b, &ForgottenStatusArgs { forgotten });
    finish(b, request_id, ControlPayload::ForgottenStatus, f.as_union_value()).to_vec()
}

/// A `GetPurge`, decoded.
pub fn get_purge_ask(g: &GetPurge<'_>) -> Option<Uuid> {
    g.purge_id().map(|u| Uuid(u.0))
}

fn put_summary<'a>(b: &mut FlatBufferBuilder<'a>, s: &PurgeSummary) -> WIPOffset<WirePurgeSummary<'a>> {
    let (kind, range, streams, id_count) = match &s.kind {
        PurgeKind::Range { span, streams } => {
            (WirePurgeKind::Range, Some(time_range(*span)), streams.clone().unwrap_or_default(), 0)
        }
        PurgeKind::Entity => (WirePurgeKind::Entity, None, Vec::new(), 0),
        PurgeKind::Ids { count } => (WirePurgeKind::Ids, None, Vec::new(), u32::try_from(*count).unwrap_or(u32::MAX)),
        PurgeKind::AgeOut { count } => {
            (WirePurgeKind::AgeOut, None, Vec::new(), u32::try_from(*count).unwrap_or(u32::MAX))
        }
    };
    let stream_ids: Vec<u32> = streams.iter().map(|s| s.0).collect();
    let stream_ids = b.create_vector(&stream_ids);
    let reason = b.create_string(&s.reason);
    let detail = b.create_string(&s.detail);
    let id = WireUuid::new(&s.purge_id.0);
    WirePurgeSummary::create(
        b,
        &PurgeSummaryArgs {
            purge_id: Some(&id),
            at_ns: s.at_ns,
            kind,
            range: range.as_ref(),
            stream_ids: Some(stream_ids),
            id_count,
            reason: Some(reason),
            detail: Some(detail),
            removed: s.removed,
            finished: s.finished,
            completed_ns: s.completed_ns.unwrap_or(0),
        },
    )
}

/// Reason and detail text one `PurgeList` carries at most: a page of 1000 records with texts at [`MAX_PURGE_TEXT`]
/// would be a 128 MiB frame, so the page stops early (keeping at least one record) and the caller continues after
/// the last one it got.
pub const MAX_PURGE_LIST_BYTES: usize = 8 * 1024 * 1024;

/// A `PurgeList`: the records, as many as fit under [`MAX_PURGE_LIST_BYTES`] of text.
pub fn purge_list(b: &mut FlatBufferBuilder<'static>, request_id: u64, purges: &[PurgeSummary]) -> Vec<u8> {
    b.reset();
    let mut text = 0usize;
    let mut kept = 0usize;
    for s in purges {
        text += s.reason.len() + s.detail.len();
        if kept > 0 && text > MAX_PURGE_LIST_BYTES {
            break;
        }
        kept += 1;
    }
    let rows: Vec<_> = purges[..kept].iter().map(|s| put_summary(b, s)).collect();
    let purges = b.create_vector(&rows);
    let l = PurgeList::create(b, &PurgeListArgs { purges: Some(purges) });
    finish(b, request_id, ControlPayload::PurgeList, l.as_union_value()).to_vec()
}

/// A `PurgeRecord`: the record's summary and its report's counts.
pub fn purge_record(
    b: &mut FlatBufferBuilder<'static>,
    request_id: u64,
    purge_id: Uuid,
    record: &CatalogPurgeRecord,
) -> Vec<u8> {
    b.reset();
    let summary = PurgeSummary {
        purge_id,
        at_ns: record.at_ns,
        kind: record.kind.clone(),
        reason: record.reason.clone(),
        detail: record.detail.clone(),
        removed: record.report.as_ref().map_or(0, |r| r.removed.len() as u64),
        finished: record.report.is_some(),
        completed_ns: record.completed_ns,
    };
    let summary = put_summary(b, &summary);
    let removed = record.report.as_ref().map(|r| removed_counts(b, r));
    let r = PurgeRecord::create(
        b,
        &PurgeRecordArgs {
            summary: Some(summary),
            removed,
            removed_frames: record.report.as_ref().map_or(0, |r| r.removed_frames.len() as u64),
            partial_chunks: record.report.as_ref().map_or(0, |r| r.partial_chunks.len() as u64),
        },
    );
    finish(b, request_id, ControlPayload::PurgeRecord, r.as_union_value()).to_vec()
}

/// A listed stream: its id and info, whether a producer is connected for it now, and its invalidation and forget
/// revisions.
pub type StreamRow = (StreamId, StreamInfo, bool, Option<u64>, Option<u64>);

/// A `StreamList`: every registered stream, with whether a producer is connected for it now.
pub fn stream_list(b: &mut FlatBufferBuilder<'static>, request_id: u64, streams: &[StreamRow]) -> Vec<u8> {
    b.reset();
    let rows: Vec<_> = streams
        .iter()
        .map(|(id, info, live, revision, forget)| {
            let name = b.create_string(&info.name);
            let shape_name = b.create_string(&info.shape_name);
            let kind = match info.kind {
                StreamKind::Sensor => WireKind::Sensor,
                StreamKind::Derived => WireKind::Derived,
            };
            StreamDesc::create(
                b,
                &StreamDescArgs {
                    stream_id: id.0,
                    name: Some(name),
                    kind,
                    layer: info.layer,
                    format: info.format,
                    shape_name: Some(shape_name),
                    shape_version: info.shape_version,
                    rate_hz: info.rate_hz.unwrap_or(0.0) as f32,
                    live: *live,
                    invalidation_revision: *revision,
                    forget_revision: *forget,
                },
            )
        })
        .collect();
    let streams = b.create_vector(&rows);
    let l = StreamList::create(b, &StreamListArgs { streams: Some(streams) });
    finish(b, request_id, ControlPayload::StreamList, l.as_union_value()).to_vec()
}
