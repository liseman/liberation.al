//! The worker side (docs/CONTINUUM.md, "Worker flow: jobs and leases"): registering a worker's capabilities,
//! quoting for a lease, and the lease itself (renew, commit, abort, or give it back), plus typed forms of what the
//! broker pushes (preemptions, system state, ingest rejections).
//!
//! The order the broker expects: register the worker's output streams (`Client::register_stream`) first, then
//! `Client::register_worker`, then quote with the capability ids it returns.

use std::sync::{Arc, Weak};
use std::time::Duration;

use continuum_core::schema::continuum as fb;
use continuum_core::{HeaderBuf, StageId, StorageLocator, Uuid};
use flatbuffers::FlatBufferBuilder;
use tokio::task::AbortHandle;

use crate::{Error, Message, Shared, messages};

/// A stage's id: the 32-bit FNV-1a hash of its name, the same everywhere and across restarts (coverage is kept per
/// source stream and stage). The broker computes the same hash.
#[must_use]
pub fn stage_id(name: &str) -> StageId {
    let mut h: u32 = 0x811c_9dc5;
    for b in name.bytes() {
        h ^= u32::from(b);
        h = h.wrapping_mul(0x0100_0193);
    }
    StageId(h)
}

/// An input a capability needs: a stream, with a window around the output's time (0 and 0 for the same instant).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Requirement {
    /// The input stream's name.
    pub stream: String,
    /// How far before the output's time the input must cover, in nanoseconds.
    pub before_ns: u64,
    /// How far after.
    pub after_ns: u64,
}

/// What a worker can make and what it needs (one entry of `RegisterWorker.capabilities`).
#[derive(Clone, Debug, PartialEq)]
pub struct CapabilitySpec {
    /// The output stream's name (registered beforehand).
    pub produces: String,
    /// Its inputs.
    pub requires: Vec<Requirement>,
    /// Where it computes.
    pub compute_domain: fb::ComputeDomain,
    /// RAM it uses with its model loaded, in MB.
    pub ram_mb: u32,
    /// VRAM, in MB.
    pub vram_mb: u32,
    /// The worker's own first guess at milliseconds per item, until the broker has measured it.
    pub est_ms_per_item: f32,
    /// How long loading the model takes, in milliseconds.
    pub cold_start_ms: u32,
    /// The most items per job.
    pub max_batch: u32,
    /// Whether a lease can be taken back mid-job.
    pub preemptible: bool,
    /// Produce, or the tier worker's Hydrate.
    pub role: fb::CapabilityRole,
    /// The largest payload one job writes, in bytes.
    pub output_bytes: u64,
    /// How many payloads a job writes.
    pub outputs_per_job: u32,
    /// The model's typical confidence, if it reports one.
    pub expected_confidence: Option<f32>,
}

impl CapabilitySpec {
    /// A capability producing `produces` on `compute_domain`, with the other fields at neutral values.
    pub fn new(produces: impl Into<String>, compute_domain: fb::ComputeDomain) -> Self {
        Self {
            produces: produces.into(),
            requires: Vec::new(),
            compute_domain,
            ram_mb: 0,
            vram_mb: 0,
            est_ms_per_item: 0.0,
            cold_start_ms: 0,
            max_batch: 1,
            preemptible: true,
            role: fb::CapabilityRole::Produce,
            output_bytes: 0,
            outputs_per_job: 1,
            expected_confidence: None,
        }
    }
}

/// A worker to register (`RegisterWorker`).
#[derive(Clone, Debug, PartialEq)]
pub struct WorkerSpec {
    /// The worker and what it is, like "worker:pose".
    pub worker_id: String,
    /// A git SHA or a hash of the weights.
    pub version_hash: String,
    /// What it can do, in order; `WorkerRegistered` answers in the same order.
    pub capabilities: Vec<CapabilitySpec>,
    /// Which GPU it runs on, on a machine with several.
    pub device_index: Option<u8>,
    /// The worker keeps its own copies of payloads (chunk files, vector rows): purges send it a `PurgeNotice` and
    /// wait for its `PurgeAck`.
    pub keeps_copies: bool,
}

/// One capability the broker accepted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AcceptedCapability {
    /// Its index in `WorkerSpec::capabilities`.
    pub index: u32,
    /// The id to quote with.
    pub id: u32,
    /// Whether this machine's profile can run it at all (one that can't is never dispatched).
    pub runnable: bool,
}

/// The broker's answer to `RegisterWorker`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WorkerRegistered {
    /// The capabilities it accepted.
    pub accepted: Vec<AcceptedCapability>,
    /// Why it refused the others (each names the capability's index).
    pub refused_reasons: Vec<String>,
}

impl WorkerRegistered {
    pub(crate) fn from_fb(w: fb::WorkerRegistered<'_>) -> Self {
        let idx: Vec<u32> = w.accepted().map(|v| v.iter().collect()).unwrap_or_default();
        let ids: Vec<u32> = w.capability_ids().map(|v| v.iter().collect()).unwrap_or_default();
        let run: Vec<bool> = w.runnable().map(|v| v.iter().collect()).unwrap_or_default();
        let accepted = idx
            .iter()
            .enumerate()
            .map(|(i, &index)| AcceptedCapability {
                index,
                id: ids.get(i).copied().unwrap_or(0),
                runnable: run.get(i).copied().unwrap_or(false),
            })
            .collect();
        let refused_reasons = w.refused_reasons().map(|v| v.iter().map(str::to_owned).collect()).unwrap_or_default();
        Self { accepted, refused_reasons }
    }

    /// The id of capability `index` (in `WorkerSpec` order), if it was accepted.
    #[must_use]
    pub fn id_of(&self, index: u32) -> Option<u32> {
        self.accepted.iter().find(|a| a.index == index).map(|a| a.id)
    }
}

pub(crate) fn register_worker(request_id: u64, spec: &WorkerSpec) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let caps: Vec<_> = spec
        .capabilities
        .iter()
        .map(|c| {
            let reqs: Vec<_> = c
                .requires
                .iter()
                .map(|r| {
                    let stream = b.create_string(&r.stream);
                    fb::Requirement::create(
                        &mut b,
                        &fb::RequirementArgs { stream: Some(stream), before_ns: r.before_ns, after_ns: r.after_ns },
                    )
                })
                .collect();
            let reqs = b.create_vector(&reqs);
            let produces = b.create_string(&c.produces);
            fb::Capability::create(
                &mut b,
                &fb::CapabilityArgs {
                    produces: Some(produces),
                    requires: Some(reqs),
                    compute_domain: c.compute_domain,
                    ram_mb: c.ram_mb,
                    vram_mb: c.vram_mb,
                    est_ms_per_item: c.est_ms_per_item,
                    cold_start_ms: c.cold_start_ms,
                    max_batch: c.max_batch,
                    preemptible: c.preemptible,
                    role: c.role,
                    output_bytes: c.output_bytes,
                    outputs_per_job: c.outputs_per_job,
                    expected_confidence: c.expected_confidence,
                },
            )
        })
        .collect();
    let caps = b.create_vector(&caps);
    let worker_id = b.create_string(&spec.worker_id);
    let version_hash = b.create_string(&spec.version_hash);
    let w = fb::RegisterWorker::create(
        &mut b,
        &fb::RegisterWorkerArgs {
            worker_id: Some(worker_id),
            version_hash: Some(version_hash),
            capabilities: Some(caps),
            device_index: spec.device_index,
            keeps_copies: spec.keeps_copies,
        },
    );
    messages::finish(&mut b, request_id, fb::ControlPayload::RegisterWorker, w.as_union_value())
}

/// A lease for work the worker starts itself (`QuoteRequest`), such as a backfill over a coverage gap.
#[derive(Clone, Debug, PartialEq)]
pub struct QuoteSpec {
    /// The capability, by its id from `WorkerRegistered`.
    pub capability_id: u32,
    /// The stage's name (its id is `stage_id(stage)`).
    pub stage: String,
    /// The band to run in.
    pub band: fb::PriorityBand,
    /// Where it computes.
    pub compute_domain: fb::ComputeDomain,
    /// Output bytes wanted.
    pub output_bytes: u64,
    /// How long the work should take, in milliseconds.
    pub est_duration_ms: u32,
    /// Parent payloads to pin while the lease lasts.
    pub pin_parents: Vec<Uuid>,
    /// The time the work covers, UTC nanoseconds, half open.
    pub range: (u64, u64),
}

pub(crate) fn quote_request(request_id: u64, q: &QuoteSpec) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let stage = b.create_string(&q.stage);
    let parents: Vec<fb::Uuid> = q.pin_parents.iter().map(|&u| fb::Uuid::from(u)).collect();
    let parents = b.create_vector(&parents);
    let range = fb::TimeRange::new(q.range.0, q.range.1);
    let r = fb::QuoteRequest::create(
        &mut b,
        &fb::QuoteRequestArgs {
            stage: Some(stage),
            band: q.band,
            compute_domain: q.compute_domain,
            output_bytes: q.output_bytes,
            est_duration_ms: q.est_duration_ms,
            pin_parents: Some(parents),
            capability_id: q.capability_id,
            range: Some(&range),
        },
    );
    messages::finish(&mut b, request_id, fb::ControlPayload::QuoteRequest, r.as_union_value())
}

/// Where a lease's output may go: a slot in an arena.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputSlot {
    /// The arena's name.
    pub arena: String,
    /// The arena's generation.
    pub arena_generation: u32,
    /// The slot's size class.
    pub slab_class: u16,
    /// The slot's index.
    pub slot: u32,
    /// The slot's generation.
    pub slot_generation: u32,
    /// Its offset in the arena (16 KB aligned).
    pub offset: u64,
    /// Its length in bytes.
    pub length: u64,
}

/// The broker's answer to a quote: a lease, or how long to wait before asking again.
#[derive(Debug)]
pub enum Quote {
    /// Granted.
    Granted(WorkLease),
    /// Not now.
    Denied {
        /// Milliseconds to wait before asking again.
        retry_after_ms: u32,
    },
}

/// What a lease's commit reports for coverage: the source stream and stage it covered, and over what time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoverageDone {
    /// The stream the work read.
    pub source_stream_id: u32,
    /// The stage's name.
    pub stage: String,
    /// The time covered, UTC nanoseconds, half open.
    pub range: (u64, u64),
}

/// The measured cost of a job.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Cost {
    /// Wall time, microseconds.
    pub wall_time_us: u64,
    /// Time queued before it ran, microseconds.
    pub queue_wait_us: u64,
    /// Peak RAM, bytes.
    pub ram_bytes: u64,
    /// Peak VRAM, bytes.
    pub vram_bytes: u64,
    /// Like "CUDA:0", "MPS", or "CPU".
    pub device: String,
    /// How many items it processed, so the scheduler learns the cost per item.
    pub items: u64,
}

fn renew(request_id: u64, lease_id: u64, extend: Duration) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let extend_ms = u32::try_from(extend.as_millis()).unwrap_or(u32::MAX);
    let r = fb::RenewLease::create(&mut b, &fb::RenewLeaseArgs { lease_id, extend_ms });
    messages::finish(&mut b, request_id, fb::ControlPayload::RenewLease, r.as_union_value())
}

fn commit(
    request_id: u64,
    lease_id: u64,
    outputs: &[(HeaderBuf, Option<StorageLocator>)],
    coverage: Option<&CoverageDone>,
    cost: Option<&Cost>,
) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let items: Vec<_> = outputs
        .iter()
        .map(|(h, at)| {
            let header_bytes = b.create_vector(h.as_bytes());
            let locator = at.as_ref().map(|l| messages::locator(&mut b, l));
            fb::Item::create(&mut b, &fb::ItemArgs { locator, header_bytes: Some(header_bytes), commit_seq: 0 })
        })
        .collect();
    let items = b.create_vector(&items);
    let coverage = coverage.map(|c| {
        let stage = b.create_string(&c.stage);
        let range = fb::TimeRange::new(c.range.0, c.range.1);
        fb::CoverageDone::create(
            &mut b,
            &fb::CoverageDoneArgs { source_stream_id: c.source_stream_id, stage: Some(stage), range: Some(&range) },
        )
    });
    let cost = cost.map(|c| {
        let device = b.create_string(&c.device);
        fb::ComputeCost::create(
            &mut b,
            &fb::ComputeCostArgs {
                wall_time_us: c.wall_time_us,
                queue_wait_us: c.queue_wait_us,
                ram_bytes: c.ram_bytes,
                vram_bytes: c.vram_bytes,
                device: Some(device),
                items: c.items,
            },
        )
    });
    let c = fb::CommitLease::create(&mut b, &fb::CommitLeaseArgs { lease_id, outputs: Some(items), coverage, cost });
    messages::finish(&mut b, request_id, fb::ControlPayload::CommitLease, c.as_union_value())
}

fn abort(request_id: u64, lease_id: u64, reason: &str) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let reason = b.create_string(reason);
    let a = fb::AbortLease::create(&mut b, &fb::AbortLeaseArgs { lease_id, reason: Some(reason) });
    messages::finish(&mut b, request_id, fb::ControlPayload::AbortLease, a.as_union_value())
}

pub(crate) fn released(lease_id: u64) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let r = fb::LeaseReleased::create(&mut b, &fb::LeaseReleasedArgs { lease_id, output_seqs: None });
    messages::finish(&mut b, 0, fb::ControlPayload::LeaseReleased, r.as_union_value())
}

/// A granted lease: the worker may write its outputs and commit them. It ends with `commit`, `abort`, or `release`;
/// dropping it unended gives it back (`LeaseReleased`), so a worker that bails out never leaves it to expire.
///
/// A lease can be preempted at any time: watch `Client::next_event` for `BrokerEvent::LeasePreempted` with this
/// lease's id, stop, and drop or `release` it. Its uncommitted output is discarded.
#[derive(Debug)]
pub struct WorkLease {
    shared: Weak<Shared>,
    lease_id: u64,
    expires_at_ns: Arc<std::sync::atomic::AtomicU64>,
    slots: Vec<OutputSlot>,
    keep_alive: Option<(AbortHandle, Duration, Duration)>,
    ended: bool,
    /// The runtime the lease was granted on, so `keep_alive` can start its task from any thread (a Python or C
    /// caller has no runtime of its own).
    runtime: tokio::runtime::Handle,
}

impl WorkLease {
    pub(crate) fn from_grant(shared: &Arc<Shared>, lease: fb::OutputLease<'_>) -> Self {
        Self::from_parts(shared, lease.lease_id(), lease.expires_at_ns(), slots_of(&lease))
    }

    pub(crate) fn from_parts(shared: &Arc<Shared>, lease_id: u64, expires_at_ns: u64, slots: Vec<OutputSlot>) -> Self {
        Self {
            shared: Arc::downgrade(shared),
            lease_id,
            expires_at_ns: Arc::new(std::sync::atomic::AtomicU64::new(expires_at_ns)),
            slots,
            keep_alive: None,
            ended: false,
            runtime: shared.runtime.clone(),
        }
    }

    /// The lease's id (what a `LeasePreempted` names).
    pub fn id(&self) -> u64 {
        self.lease_id
    }

    /// When it expires, UTC nanoseconds (updated by `renew` and the keep-alive).
    pub fn expires_at_ns(&self) -> u64 {
        self.expires_at_ns.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// The output slots granted with it (empty until the broker hands out slot arenas).
    pub fn slots(&self) -> &[OutputSlot] {
        &self.slots
    }

    fn shared(&self) -> Result<Arc<Shared>, Error> {
        self.shared.upgrade().ok_or(Error::Closed)
    }

    /// A handle that renews this lease without owning it, for a wrapper that keeps the lease behind a lock and
    /// doesn't want to hold the lock across a round trip.
    pub fn renewer(&self) -> Renewer {
        Renewer { shared: self.shared.clone(), lease_id: self.lease_id, expires_at_ns: Arc::clone(&self.expires_at_ns) }
    }

    /// Renews the lease by `extend` from now and returns its new expiry, UTC nanoseconds. The broker caps `extend`
    /// at its maximum lease.
    ///
    /// # Errors
    ///
    /// As any request: `Error::Broker` with NotFound if the lease is gone (preempted or expired).
    pub async fn renew(&self, extend: Duration) -> Result<u64, Error> {
        renew_now(&*self.shared()?, self.lease_id, extend, &self.expires_at_ns).await
    }

    /// Renews the lease in the background every `every` until it ends (`extend` as in `renew`). For a long job: pick
    /// `every` well under the broker's renewal period. It doesn't stop on a preemption (the broker still renews a
    /// lease it's taking back), so watch the events for a `LeasePreempted` naming this lease and end it then; it
    /// stops on its own only when a renewal fails (the lease is gone).
    pub fn keep_alive(&mut self, every: Duration, extend: Duration) {
        if let Some((old, _, _)) = self.keep_alive.take() {
            old.abort();
        }
        let (shared, lease_id, expires) = (self.shared.clone(), self.lease_id, Arc::clone(&self.expires_at_ns));
        let task = self.runtime.spawn(async move {
            let mut tick = tokio::time::interval(every.max(Duration::from_millis(10)));
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            tick.tick().await;
            loop {
                tick.tick().await;
                let Some(shared) = shared.upgrade() else { return };
                if renew_now(&shared, lease_id, extend, &expires).await.is_err() {
                    return;
                }
            }
        });
        self.keep_alive = Some((task.abort_handle(), every, extend));
    }

    /// Marks the lease ended (so a cancelled request sends no release: the broker may already hold the commit or
    /// abort) and stops the keep-alive, returning its settings.
    fn end(&mut self) -> Option<(Duration, Duration)> {
        self.ended = true;
        self.keep_alive.take().map(|(k, every, extend)| {
            k.abort();
            (every, extend)
        })
    }

    async fn ask(&self, build: impl FnOnce(u64) -> Vec<u8>) -> Result<(), Error> {
        let reply = self.shared()?.request(build).await?;
        expect_released(&reply, self.lease_id)
    }

    /// Commits the job: its outputs (headers, and where their bytes are), the coverage it finished, and what it
    /// cost. Returns once the broker has stored the outputs durably.
    ///
    /// # Errors
    ///
    /// A `CommitError` if the broker refuses or the request fails. When the broker was only busy (Unavailable), the
    /// error hands the lease back (keep-alive still running) so the commit can be retried. Otherwise the lease is
    /// given back to the broker (a release, which it ignores for a lease it no longer holds) and is over.
    pub async fn commit(
        mut self,
        outputs: &[(HeaderBuf, Option<StorageLocator>)],
        coverage: Option<&CoverageDone>,
        cost: Option<&Cost>,
    ) -> Result<(), CommitError> {
        let renewing = self.end();
        let lease_id = self.lease_id;
        match self.ask(|id| commit(id, lease_id, outputs, coverage, cost)).await {
            Ok(()) => Ok(()),
            Err(error) if error.broker_code() == Some(fb::ErrorCode::Unavailable) => {
                self.ended = false;
                if let Some((every, extend)) = renewing {
                    self.keep_alive(every, extend);
                }
                Err(CommitError { error, lease: Some(self) })
            }
            Err(error) => {
                self.send_release();
                Err(CommitError { error, lease: None })
            }
        }
    }

    /// Gives up the job: the broker frees the lease's slots and replans the promises it served. Returns once the
    /// broker has done so.
    ///
    /// # Errors
    ///
    /// As any request; the lease is over either way (on an error it's released, which the broker ignores for a
    /// lease it no longer holds).
    pub async fn abort(mut self, reason: &str) -> Result<(), Error> {
        self.end();
        let lease_id = self.lease_id;
        let r = self.ask(|id| abort(id, lease_id, reason)).await;
        if r.is_err() {
            self.send_release();
        }
        r
    }

    /// Gives the lease back without committing (after a preemption, or when done early). One way: the broker
    /// doesn't answer.
    pub fn release(mut self) {
        self.end();
        self.send_release();
    }

    fn send_release(&self) {
        if let Some(shared) = self.shared.upgrade() {
            let _ = shared.writes.send((released(self.lease_id), None));
        }
    }
}

impl Drop for WorkLease {
    fn drop(&mut self) {
        if !self.ended {
            self.end();
            self.send_release();
        }
    }
}

/// Renews a lease it doesn't own (from `WorkLease::renewer`). Renewing a lease that already ended fails with the
/// broker's NotFound.
#[derive(Clone, Debug)]
pub struct Renewer {
    shared: Weak<Shared>,
    lease_id: u64,
    expires_at_ns: Arc<std::sync::atomic::AtomicU64>,
}

impl Renewer {
    /// As `WorkLease::renew`.
    ///
    /// # Errors
    ///
    /// As `WorkLease::renew`.
    pub async fn renew(&self, extend: Duration) -> Result<u64, Error> {
        let shared = self.shared.upgrade().ok_or(Error::Closed)?;
        renew_now(&shared, self.lease_id, extend, &self.expires_at_ns).await
    }

    /// As `WorkLease::expires_at_ns`.
    pub fn expires_at_ns(&self) -> u64 {
        self.expires_at_ns.load(std::sync::atomic::Ordering::SeqCst)
    }
}

/// A commit that didn't go through.
#[derive(Debug, thiserror::Error)]
#[error("{error}")]
pub struct CommitError {
    /// Why.
    #[source]
    pub error: Error,
    /// The lease, still held, when the broker was only busy (Unavailable): commit it again, or end it. `None`
    /// otherwise: the lease is over.
    pub lease: Option<WorkLease>,
}

impl CommitError {
    /// The broker's error code, if the broker refused.
    pub fn broker_code(&self) -> Option<fb::ErrorCode> {
        self.error.broker_code()
    }
}

impl From<CommitError> for Error {
    fn from(e: CommitError) -> Self {
        e.error
    }
}

async fn renew_now(
    shared: &Shared,
    lease_id: u64,
    extend: Duration,
    expires: &std::sync::atomic::AtomicU64,
) -> Result<u64, Error> {
    let reply = shared.request(|id| renew(id, lease_id, extend)).await?;
    let env = reply.envelope();
    let g = env.payload_as_lease_grant().ok_or_else(|| crate::unexpected("LeaseGrant", &reply))?;
    let lease = g.lease().filter(|l| g.granted() && l.lease_id() == lease_id);
    let lease = lease.ok_or_else(|| Error::Protocol(format!("a renewal of lease {lease_id} came back refused")))?;
    expires.store(lease.expires_at_ns(), std::sync::atomic::Ordering::SeqCst);
    Ok(lease.expires_at_ns())
}

fn expect_released(reply: &Message, lease_id: u64) -> Result<(), Error> {
    match reply.envelope().payload_as_lease_released() {
        Some(r) if r.lease_id() == lease_id => Ok(()),
        _ => Err(crate::unexpected("LeaseReleased", reply)),
    }
}

/// What the broker pushes on its own, decoded (`Message::event`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BrokerEvent {
    /// A lease is being taken back: stop work on it, and drop or release it.
    LeasePreempted {
        /// The lease.
        lease_id: u64,
        /// Why. One starting "revoked:" means the broker took it back without a higher band asking (it expired, say).
        reason: String,
        /// The band that took it.
        by_band: fb::PriorityBand,
    },
    /// The voice turn changed.
    SystemStateChanged {
        /// The new state.
        voice_turn: fb::VoiceTurnState,
        /// When, UTC nanoseconds.
        changed_at_ns: u64,
    },
    /// An `IngestNotice` was refused.
    IngestRejected {
        /// The notice's stream.
        stream_id: u32,
        /// The payload, if the broker could read it from the notice.
        payload_id: Option<Uuid>,
        /// Why.
        reason: String,
    },
    /// Work the broker assigned this worker: take it on with `Client::job_lease`, or turn it down with
    /// `Client::reject_job`.
    DispatchJob(Box<DispatchJob>),
    /// The broker is purging these payloads: drop any copies this worker keeps, then `Client::ack_purge` once. A
    /// large purge comes in pages; the client gathers them and delivers this once, with every page's ids.
    PurgeNotice {
        /// The purge, to acknowledge.
        purge_id: Uuid,
        /// The payloads.
        payload_ids: Vec<Uuid>,
        /// What the purge removed, per stream: a copy over any of these goes, even when its id isn't listed (a
        /// frame inside a chunk).
        spans: Vec<PurgedSpan>,
    },
    /// A part of a committed promise's outputs (`FulfillPromise`); the last part has `final_part`.
    Fulfilled {
        /// The promise.
        promise_id: Uuid,
        /// The outputs in this part.
        items: Vec<Item>,
        /// No more parts follow.
        final_part: bool,
    },
    /// A committed promise failed (`PromiseFailed`): nothing more comes for it.
    PromiseFailed {
        /// The promise.
        promise_id: Uuid,
        /// Why.
        reason: String,
        /// How many items were delivered before it failed.
        delivered: u32,
    },
    /// A message of a type this SDK decodes that it couldn't read (a bad input header, say); the `Message` has it.
    /// For a `DispatchJob`, the job and its lease are named when they could be read, so the worker can still
    /// `reject_job` it (and the broker frees the lease) instead of leaving it to expire.
    Undecodable {
        /// Its type.
        kind: fb::ControlPayload,
        /// What was wrong.
        reason: String,
        /// The job, if it was a `DispatchJob` whose id could be read.
        job_id: Option<Uuid>,
        /// Its output lease, likewise.
        lease_id: Option<u64>,
    },
    /// The broker refused something this session sent one way (a `JobRejected` for an unknown job, say).
    Error {
        /// The broker's code.
        code: fb::ErrorCode,
        /// Why.
        message: String,
    },
    /// Something else (a type this SDK doesn't decode yet); the `Message` has it.
    Other(fb::ControlPayload),
}

/// A stream's time range a purge removed (UTC ns, half open).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PurgedSpan {
    /// The stream.
    pub stream_id: u32,
    /// The range.
    pub range: (u64, u64),
}

/// The spans of a `PurgeNotice` page.
pub(crate) fn spans_of(p: &fb::PurgeNotice<'_>) -> Vec<PurgedSpan> {
    p.spans()
        .map(|v| {
            v.iter()
                .map(|s| PurgedSpan {
                    stream_id: s.stream_id(),
                    range: s.range().map_or((0, 0), |r| (r.start_ns(), r.end_ns())),
                })
                .collect()
        })
        .unwrap_or_default()
}

/// One frame of a Tier 1 chunk, as the schema's `ChunkFrame`: where its bytes are in the chunk, and whether a purge
/// cut it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkFrame {
    /// The frame's start, UTC nanoseconds.
    pub start_ns: u64,
    /// Its first byte in the chunk.
    pub offset: u64,
    /// Its length in bytes.
    pub length: u64,
    /// Cut by a purge: the place is kept, the bytes are gone.
    pub cut: bool,
}

/// What capture attested about one frame of a chunk (schema minor 27): kept in the catalog after the frame's own
/// header is absorbed, so continuity can be shown without inferring it from times.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameProvenance {
    /// The device's or driver's frame sequence.
    pub device_seq: u64,
    /// The capture session it came from; `device_seq` compares only within one nonzero epoch. Zero is unknown.
    pub epoch: u32,
    /// Stamped when the frame arrived, not by the device.
    pub stamped_at_arrival: bool,
}

/// The provenance layout this client writes and reads.
pub const PROVENANCE_VERSION: u32 = 1;

/// A chunk the tier worker closed, for [`crate::Client::commit_chunk`]: its own header and Tier 1 or 2 location,
/// and its frame index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NewChunk {
    /// The chunk's header (on the frames' stream, spanning them, no parents).
    pub header: HeaderBuf,
    /// Where the chunk file is.
    pub location: StorageLocator,
    /// The frame index, one per absorbed payload, in order.
    pub frames: Vec<ChunkFrame>,
}

/// The broker's answer to [`crate::Client::commit_chunk`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkCommitted {
    /// The chunk's id.
    pub chunk_id: Uuid,
    /// Its commit sequence: the position subscribers and the event log see.
    pub commit_seq: u64,
    /// It was committed already (a retry after a lost reply): nothing changed.
    pub duplicate: bool,
    /// It landed under a running purge, so it isn't available.
    pub hidden: bool,
}

impl ChunkCommitted {
    pub(crate) fn from_fb(r: fb::ChunkCommitted<'_>) -> Self {
        Self {
            chunk_id: Uuid(r.chunk_id().map_or([0; 16], |u| u.0)),
            commit_seq: r.commit_seq(),
            duplicate: r.duplicate(),
            hidden: r.hidden(),
        }
    }
}

/// An input of a dispatched job: its header and where its bytes are.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Item {
    /// The header, as the catalog keeps it.
    pub header: HeaderBuf,
    /// Where the bytes are; `None` for a payload that is only its header.
    pub locator: Option<StorageLocator>,
}

/// The output lease that came with a job.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JobLease {
    /// The lease.
    pub lease_id: u64,
    /// Its output slots (empty until the broker hands out slot arenas for it).
    pub slots: Vec<OutputSlot>,
    /// When it expires, UTC nanoseconds.
    pub expires_at_ns: u64,
}

/// A `DispatchJob`: work the broker assigned this worker.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DispatchJob {
    /// The job (what `reject_job` names).
    pub job_id: Uuid,
    /// The promises it serves.
    pub promise_ids: Vec<Uuid>,
    /// The stream the capability produces, by name; empty for a hydrate.
    pub capability: String,
    /// The capability's id from `WorkerRegistered`.
    pub capability_id: u32,
    /// What to read, resolved and pinned by the broker.
    pub inputs: Vec<Item>,
    /// The time to cover, UTC nanoseconds, half open.
    pub output_range: (u64, u64),
    /// Its band.
    pub band: fb::PriorityBand,
    /// When it's due, UTC nanoseconds.
    pub deadline_ns: u64,
    /// The output lease (see `Client::job_lease`).
    pub lease: Option<JobLease>,
}

fn slots_of(lease: &fb::OutputLease<'_>) -> Vec<OutputSlot> {
    lease
        .slots()
        .map(|v| {
            v.iter()
                .map(|s| OutputSlot {
                    arena: s.arena().unwrap_or_default().to_owned(),
                    arena_generation: s.arena_generation(),
                    slab_class: s.slab_class(),
                    slot: s.slot(),
                    slot_generation: s.slot_generation(),
                    offset: s.offset(),
                    length: s.length(),
                })
                .collect()
        })
        .unwrap_or_default()
}

pub(crate) fn ids(v: Option<flatbuffers::Vector<'_, fb::Uuid>>) -> Vec<Uuid> {
    v.map(|v| v.iter().map(Uuid::from).collect()).unwrap_or_default()
}

/// An `Item` from its table: the header bytes (required) and the locator (if any).
pub(crate) fn item_of(it: &fb::Item<'_>) -> Result<Item, String> {
    let bytes = it.header_bytes().ok_or("header is missing")?;
    let header = HeaderBuf::new(bytes.bytes().to_vec()).map_err(|e| format!("header: {e}"))?;
    let locator =
        it.locator().map(|l| StorageLocator::from_table(&l)).transpose().map_err(|e| format!("locator: {e}"))?;
    Ok(Item { header, locator })
}

fn dispatch_job(j: &fb::DispatchJob<'_>) -> Result<DispatchJob, String> {
    let job_id = j.job_id().map(Uuid::from).ok_or("DispatchJob without a job_id")?;
    let mut inputs = Vec::new();
    for (i, it) in j.inputs().into_iter().flatten().enumerate() {
        inputs.push(item_of(&it).map_err(|e| format!("input {i}'s {e}"))?);
    }
    let lease =
        j.output().map(|l| JobLease { lease_id: l.lease_id(), slots: slots_of(&l), expires_at_ns: l.expires_at_ns() });
    let range = j.output_range().map(|r| (r.start_ns(), r.end_ns())).ok_or("DispatchJob without an output_range")?;
    Ok(DispatchJob {
        job_id,
        promise_ids: ids(j.promise_ids()),
        capability: j.capability().unwrap_or_default().to_owned(),
        capability_id: j.capability_id(),
        inputs,
        output_range: range,
        band: j.band(),
        deadline_ns: j.deadline_ns(),
        lease,
    })
}

impl BrokerEvent {
    /// Whether this is a revocation (the broker took the lease back on its own) rather than a preemption.
    #[must_use]
    pub fn is_revocation(&self) -> bool {
        matches!(self, Self::LeasePreempted { reason, .. } if reason.starts_with("revoked:"))
    }
}

impl Message {
    /// The message as a `BrokerEvent`.
    pub fn event(&self) -> BrokerEvent {
        let env = self.envelope();
        if let Some(p) = env.payload_as_lease_preempted() {
            return BrokerEvent::LeasePreempted {
                lease_id: p.lease_id(),
                reason: p.reason().unwrap_or_default().to_owned(),
                by_band: p.by_band(),
            };
        }
        if let Some(f) = env.payload_as_fulfill_promise() {
            let promise_id = Uuid(f.promise_id().map_or([0; 16], |u| u.0));
            let items: Result<Vec<Item>, String> = f.items().into_iter().flatten().map(|it| item_of(&it)).collect();
            return match items {
                Ok(items) => BrokerEvent::Fulfilled { promise_id, items, final_part: f.final_part() },
                Err(reason) => BrokerEvent::Undecodable {
                    kind: fb::ControlPayload::FulfillPromise,
                    reason: format!("a FulfillPromise item's {reason}"),
                    job_id: None,
                    lease_id: None,
                },
            };
        }
        if let Some(f) = env.payload_as_promise_failed() {
            return BrokerEvent::PromiseFailed {
                promise_id: Uuid(f.promise_id().map_or([0; 16], |u| u.0)),
                reason: f.reason().unwrap_or_default().to_owned(),
                delivered: f.delivered(),
            };
        }
        if let Some(s) = env.payload_as_system_state_changed() {
            return BrokerEvent::SystemStateChanged { voice_turn: s.voice_turn(), changed_at_ns: s.changed_at_ns() };
        }
        if let Some(r) = env.payload_as_ingest_rejected() {
            return BrokerEvent::IngestRejected {
                stream_id: r.stream_id(),
                payload_id: r.payload_id().map(Uuid::from),
                reason: r.reason().unwrap_or_default().to_owned(),
            };
        }
        if let Some(j) = env.payload_as_dispatch_job() {
            return match dispatch_job(&j) {
                Ok(job) => BrokerEvent::DispatchJob(Box::new(job)),
                Err(reason) => BrokerEvent::Undecodable {
                    kind: env.payload_type(),
                    reason,
                    job_id: j.job_id().map(Uuid::from),
                    lease_id: j.output().map(|l| l.lease_id()),
                },
            };
        }
        if let Some(p) = env.payload_as_purge_notice() {
            return match p.purge_id() {
                Some(id) => BrokerEvent::PurgeNotice {
                    purge_id: Uuid::from(id),
                    payload_ids: ids(p.payload_ids()),
                    spans: spans_of(&p),
                },
                None => BrokerEvent::Undecodable {
                    kind: env.payload_type(),
                    reason: "PurgeNotice without a purge_id".into(),
                    job_id: None,
                    lease_id: None,
                },
            };
        }
        if let Some(e) = env.payload_as_error() {
            return BrokerEvent::Error { code: e.code(), message: e.message().unwrap_or_default().to_owned() };
        }
        BrokerEvent::Other(env.payload_type())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_ids_are_fnv_1a() {
        // The FNV-1a 32-bit test vectors.
        assert_eq!(stage_id("").0, 0x811c_9dc5);
        assert_eq!(stage_id("a").0, 0xe40c_292c);
        assert_eq!(stage_id("foobar").0, 0xbf9c_f968);
    }
}
