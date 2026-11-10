//! The broker's core, split so that nothing slow ever stalls anything fast (docs/CONTINUUM.md, the broker's no-stall
//! rule):
//!
//! - **The scheduler task** owns everything decided in memory: open sessions, which streams are live, the
//!   [`AvailabilityIndex`], and (as it is wired in) the scheduler. It never touches the catalog, so it never waits on
//!   disk. It takes commands from sessions on two channels, control and normal, and always drains control first
//!   (ingest, session open and close, system state, and the writer's results), so a burst of queries can't delay a
//!   preemption or an ingest.
//! - **The writer thread** is the only place the catalog is written. It has two queues: urgent (registrations and
//!   ingest, coalesced into batches) and background (long jobs such as a purge, run one bounded step at a time). It
//!   runs one bounded urgent batch before each background step, so neither queue can starve the other. After each
//!   commit it tells the scheduler task, which then updates the index, so the index never shows anything the catalog
//!   hasn't committed.
//! - **The reader pool** runs catalog reads (query results, pages, gaps) on several threads at once; the catalog
//!   gives each read its own snapshot, so reads never block each other or the writer.
//!
//! Sessions talk to the core through a [`CoreHandle`]. The core answers by putting encoded envelopes on the session's
//! outbound queue, which the session's writer sends; that queue also carries messages the broker sends on its own.
//! A full queue (a session that isn't reading) closes that session only. Every channel into the core is bounded and
//! sent to without waiting where waiting could stall a caller: a command the core can't take now is refused as busy.
//!
//! Liveness: a stream is live while a session that registered it is open.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use continuum_catalog::{
    Catalog, CatalogError, NewPayload, PurgeDelta, PurgeKind, PurgeProgress, PurgeRecord, PurgeReport, PurgeSelector,
    Resolved, StreamInfo, StreamKind as CatalogKind,
};
use continuum_core::schema::continuum::{ErrorCode, PromiseLife};
use continuum_core::{HeaderBuf, Interval, PriorityBand, StorageLocator, StorageTier, StreamId, Uuid, VoiceTurnState};
use continuum_sched::{
    Action, Capability, CapabilityId, CapabilityRole, ConsumerId, CoverageDone, Covered, Dispatch, Engine, IngestRef,
    JobId, LeaseDecision, LeaseId, LeaseReq, Manifest, OutputRef, Profile, PromiseId, PromiseState, QueryDecision,
    QueryReq, Quote, Refusal, Requirement, Scheduler, StreamKind, SystemState, WorkerId,
};
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, TryRecvError, TrySendError};
use flatbuffers::FlatBufferBuilder;
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, warn};

use crate::index::{AvailabilityIndex, StreamMeta};
use crate::protocol;
use crate::view::IndexView;

mod aging;
mod chunks;
mod forget;
mod payload_copies;
mod query;
mod slots;
mod subscribe;
mod upkeep;
mod worker_state;
mod writer_health;

use forget::Forget;
pub use forget::ForgetConfig;

use query::{QueryReply, ReadItem};
use subscribe::{CatchUpPage, Subscription};

/// How many commands each of the scheduler task's two channels holds.
const COMMANDS: usize = 4096;
/// How many jobs the writer's urgent queue holds before ingest is refused as busy.
const URGENT: usize = 8192;
/// How many long jobs can wait for the writer.
const BACKGROUND: usize = 64;
/// How many reads can wait for the reader pool before a read is refused as busy.
const READS: usize = 1024;
/// The most urgent jobs the writer takes in one tick, and so the most ingest notices in one catalog transaction.
///
/// A durable catalog (redb) makes every write transaction an fsync, a few milliseconds on the Mac's NVMe. So the
/// writer commits everything a tick took in one insert, one fsync per tick however many notices arrived, rather than
/// one per notice; under load a tick simply carries more.
const TICK: usize = 4096;
/// The most ingest results one event carries to the scheduler task.
const EVENT_CHUNK: usize = 256;
/// How many reader threads the pool runs.
pub const DEFAULT_READERS: usize = 4;

/// A session, as the core knows it: the id it was given at `Hello`.
pub type SessionId = uuid::Uuid;

/// A long catalog job, run on the writer thread one step at a time. Each call does one step of at most `budget`
/// rows (or items, as the job counts them) and returns whether there is more to do; the writer runs one bounded
/// urgent batch between steps, even when more urgent work keeps arriving.
///
/// The writer sizes `budget` per job so a step takes about [`WriterConfig::step_target`]: it starts at
/// [`WriterConfig::initial_budget`], halves it after a step that overran the target (logged at debug; a step
/// [`STALL_FACTOR`] times over it is warned about), and
/// doubles it after a step that took under half the target, up to [`WriterConfig::max_budget`]. Urgent batches and
/// background steps alternate while both queues have work. Purge jobs use their own cooperative work window
/// rather than the adaptive row budget, so fixed commit overhead cannot reduce them to singleton writes.
pub type BackgroundJob = Box<dyn FnMut(&dyn Catalog, usize) -> bool + Send>;

/// A long job on the writer's queue, and whether a clean stop may drop it unfinished. Only restartable upkeep that
/// never changes availability (the `FRAMES` sweep) may: the restart fence drops it rather than wait for it, so a
/// sweep that is always running can't keep every stop from saving the restart projection.
struct LongJob {
    job: BackgroundJob,
    drop_at_stop: bool,
}

impl LongJob {
    fn droppable(job: BackgroundJob) -> Self {
        Self { job, drop_at_stop: true }
    }
}

impl From<BackgroundJob> for LongJob {
    fn from(job: BackgroundJob) -> Self {
        Self { job, drop_at_stop: false }
    }
}

/// Why a clean stop skipped the restart projection. Each one means the next start rebuilds the availability index
/// from the catalog; the server logs which one it was.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckpointRefusal {
    /// Sessions were still registered with the core.
    SessionsOpen,
    /// A purge job was running.
    PurgeRunning,
    /// Reads were still outstanding on the reader pool.
    ReadsOutstanding,
    /// Ingests were still staged for the writer.
    Staging,
    /// A purge was still waiting for keepers, completing, or had unsaved keeper writes.
    ForgetUnsettled,
    /// A `FRAMES` sweep was still running after the fence.
    FrameSweep,
    /// Acknowledged subscription positions were still unsaved.
    CursorsUnsaved,
    /// A payload hid as it landed and no purge step has named it yet.
    HiddenOnLanding,
    /// A stream's kind or rate changed since the index was built.
    StreamMetadataChanged,
    /// The writer's urgent queue was full, so the fence couldn't be queued.
    WriterQueueFull,
    /// A writer callback other than a cursor save arrived while the fence was settling.
    CallbackDuringFence,
    /// Work was still queued on the writer when the fence reached it.
    WriterBusy,
    /// The catalog had pending purges.
    PendingPurges,
    /// The catalog keeps no restart stamp (restart tracking is off for it).
    NoStamp,
    /// The catalog couldn't be read for its pending purges or restart stamp.
    CatalogUnreadable,
}

impl std::fmt::Display for CheckpointRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::SessionsOpen => "sessions were still open",
            Self::PurgeRunning => "a purge was running",
            Self::ReadsOutstanding => "reads were outstanding",
            Self::Staging => "ingests were still staged",
            Self::ForgetUnsettled => "a finished purge was still waiting for keepers or completing",
            Self::FrameSweep => "a frame sweep was still running",
            Self::CursorsUnsaved => "subscription positions were unsaved",
            Self::HiddenOnLanding => "a payload hidden on landing hasn't been named by its purge",
            Self::StreamMetadataChanged => "a stream's kind or rate changed since the index was built",
            Self::WriterQueueFull => "the writer's urgent queue was full",
            Self::CallbackDuringFence => "a writer callback arrived during the fence",
            Self::WriterBusy => "work was still queued on the writer",
            Self::PendingPurges => "the catalog has pending purges",
            Self::NoStamp => "the catalog keeps no restart stamp",
            Self::CatalogUnreadable => "the catalog couldn't be read for its restart stamp",
        })
    }
}

/// What a clean-stop fence hands back: the index to encode and its catalog stamp, or why there is none.
pub type RestartCheckpoint = Result<(AvailabilityIndex, continuum_catalog::RestartStamp), CheckpointRefusal>;

/// How long the writer waits with nothing to do before making the catalog's latest commits durable.
const DURABLE_IDLE: Duration = Duration::from_secs(1);

/// A background step this many times over its target held urgent writes up long enough to warn about.
const STALL_FACTOR: u32 = 10;

/// How the writer sizes background steps.
#[derive(Clone, Copy, Debug)]
pub struct WriterConfig {
    /// How long one background step should take.
    pub step_target: Duration,
    /// The first step's budget for a new job.
    pub initial_budget: usize,
    /// The largest budget a job grows to, also the phase-operation ceiling for a purge work window.
    pub max_budget: usize,
}

impl Default for WriterConfig {
    fn default() -> Self {
        Self { step_target: Duration::from_millis(5), initial_budget: 256, max_budget: 65_536 }
    }
}

/// The budget for a job's next step, given how long its last step took. Always at least 1.
#[must_use]
pub fn next_budget(budget: usize, took: Duration, config: &WriterConfig) -> usize {
    let next = if took > config.step_target {
        budget / 2
    } else if took < config.step_target / 2 {
        budget.saturating_mul(2)
    } else {
        budget
    };
    next.clamp(1, config.max_budget.max(1))
}

/// A catalog read, run on a reader thread.
pub type ReadJob = Box<dyn FnOnce(&dyn Catalog) + Send>;

/// Counts accepted reads until their callback has been emitted; terminal shutdown refuses new reads.
struct ReadQueue {
    sender: Sender<ReadJob>,
    outstanding: Arc<AtomicUsize>,
    stopped: std::sync::atomic::AtomicBool,
}
struct PendingRead(Arc<AtomicUsize>);
impl Drop for PendingRead {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}
impl ReadQueue {
    fn new(sender: Sender<ReadJob>) -> Self {
        Self { sender, outstanding: Arc::new(AtomicUsize::new(0)), stopped: std::sync::atomic::AtomicBool::new(false) }
    }
    fn try_send(&self, job: ReadJob) -> Result<(), TrySendError<()>> {
        if self.stopped.load(Ordering::SeqCst) {
            return Err(TrySendError::Disconnected(()));
        }
        self.outstanding.fetch_add(1, Ordering::SeqCst);
        let pending = PendingRead(Arc::clone(&self.outstanding));
        let wrapped: ReadJob = Box::new(move |catalog| {
            let _pending = pending;
            job(catalog);
        });
        self.sender.try_send(wrapped).map_err(|error| match error {
            TrySendError::Full(_) => TrySendError::Full(()),
            TrySendError::Disconnected(_) => TrySendError::Disconnected(()),
        })
    }
}

/// A request to the core.
pub enum Command {
    /// Read a coherent recording applicability page.
    ListRecordingInputs {
        /// Asking session.
        session: SessionId,
        /// Request identity.
        request_id: u64,
        /// Bounded decoded request.
        ask: Box<protocol::applicability::ListRecordingAsk>,
    },
    /// `GetCatalogStats`: where the catalog file's bytes go, read off the writer.
    CatalogStats {
        /// The asking session.
        session: SessionId,
        /// The request's id, for the reply.
        request_id: u64,
    },
    /// `ListAgedOut`: a stream's aged-out spans.
    ListAgedOut {
        /// The asking session.
        session: SessionId,
        /// The request's id, for the reply.
        request_id: u64,
        /// The stream's name.
        stream: String,
        /// The window.
        window: Interval,
        /// At most this many spans.
        limit: usize,
    },
    /// `CommitSeqAt`: the commit clock.
    CommitSeqAt {
        /// The asking session.
        session: SessionId,
        /// The request's id, for the reply.
        request_id: u64,
        /// The time, UTC nanoseconds.
        at_ns: u64,
    },
    /// `ResolveFrames`: which chunk holds each of these ids.
    ResolveFrames {
        /// The asking session.
        session: SessionId,
        /// The request's id, for the reply.
        request_id: u64,
        /// At most `protocol::aging::MAX_RESOLVE_IDS`.
        ids: Vec<Uuid>,
    },
    /// `GetChunk`: a chunk's frames and copies.
    GetChunk {
        /// The asking session.
        session: SessionId,
        /// The request's id, for the reply.
        request_id: u64,
        /// The chunk.
        chunk: Uuid,
    },
    /// A session finished its handshake; `outbound` is where its replies and pushes go.
    Open {
        /// The session.
        session: SessionId,
        /// The session writer's queue.
        outbound: mpsc::Sender<Vec<u8>>,
        /// Signalled when the core drops the session (its queue filled), so the connection closes too.
        kicked: Arc<tokio::sync::Notify>,
        /// The client's process id, from its `Hello`.
        pid: u32,
        /// The turn at the same serialized point as this session opens.
        voice: oneshot::Sender<VoiceTurnState>,
    },
    /// A session ended. Its streams stop being live (unless another open session registered them too).
    Close {
        /// The session.
        session: SessionId,
    },
    /// `RegisterStream`: the core answers `StreamRegistered` or `Error`.
    RegisterStream {
        /// The session asking.
        session: SessionId,
        /// The request's id, for the reply.
        request_id: u64,
        /// The stream, as the catalog keeps it.
        info: StreamInfo,
        /// The ring it asks for, for a fixed-stride stream.
        ring: Option<protocol::RingAsk>,
    },
    /// `IngestNotice`: index the payload. Nothing is sent back unless it is refused (`IngestRejected`).
    Ingest {
        /// The session sending it.
        session: SessionId,
        /// The notice's request id, for a rejection.
        request_id: u64,
        /// The payload's header, already checked to be on the notice's stream.
        header: HeaderBuf,
        /// Where its bytes are, if anywhere yet.
        locator: Option<StorageLocator>,
        /// What it depends on, stored with it.
        dependencies: Vec<continuum_catalog::Dependency>,
    },
    /// `RegisterWorker`: the scheduler indexes the worker's capabilities; the answer is `WorkerRegistered`.
    RegisterWorker {
        /// The session registering (one worker per session).
        session: SessionId,
        /// For the reply.
        request_id: u64,
        /// What it declared.
        ask: protocol::WorkerAsk,
    },
    /// `SetSystemState`: the voice turn moved. Preempts at once, and every session hears `SystemStateChanged`.
    SetSystemState {
        /// The new voice turn.
        voice: VoiceTurnState,
    },
    /// `QuoteRequest`: a worker asks for a lease for work it starts itself. The answer is `LeaseGrant`.
    QuoteRequest {
        /// The worker's session.
        session: SessionId,
        /// For the reply.
        request_id: u64,
        /// What it asks for.
        req: LeaseReq,
    },
    /// `RenewLease`: push a lease's expiry out. The answer is a `LeaseGrant` with the new expiry.
    RenewLease {
        /// The worker's session.
        session: SessionId,
        /// For the reply.
        request_id: u64,
        /// The lease.
        lease: u64,
        /// How long from now the worker asks for, in milliseconds (0 for the default).
        extend_ms: u32,
    },
    /// `CommitLease`: index the outputs, then tell the scheduler the job is done.
    CommitLease {
        /// The worker's session.
        session: SessionId,
        /// For an error reply.
        request_id: u64,
        /// What it committed.
        commit: protocol::Commit,
    },
    /// `AbortLease`: the job failed.
    AbortLease {
        /// The worker's session.
        session: SessionId,
        /// For an error reply.
        request_id: u64,
        /// The lease.
        lease: u64,
        /// Why.
        reason: String,
    },
    /// `LeaseReleased`: the worker gave a lease back (after a preemption, or early).
    LeaseReleased {
        /// The worker's session.
        session: SessionId,
        /// The lease.
        lease: u64,
    },
    /// A worker's heartbeat telemetry.
    Telemetry {
        /// The worker's session.
        session: SessionId,
        /// What it reported.
        telemetry: continuum_sched::Telemetry,
    },
    /// `Purge`: run a purge on the writer, step by step, following each step in the index, and answer
    /// `PurgeResult` (or `Error`) when it's done.
    Purge {
        /// The session asking.
        session: SessionId,
        /// The request's id, for the reply.
        request_id: u64,
        /// What to purge.
        ask: protocol::PurgeAsk,
        /// Why, for the purge's record.
        reason: String,
        /// The caller's JSON detail, kept in the record (never naming the person).
        detail: String,
    },
    /// A registered keeper adds a durable copy of an existing derived payload, without absorption.
    CommitPayloadCopy {
        /// The keeper's session.
        session: SessionId,
        /// The request to answer.
        request_id: u64,
        /// The original payload identity.
        payload: Uuid,
        /// Its durable warm copy.
        location: StorageLocator,
    },
    /// `CommitChunk`: commit a Tier 1 chunk the tier worker closed (see `core::chunks`).
    CommitChunk {
        /// The worker's session.
        session: SessionId,
        /// The request's id, for the reply.
        request_id: u64,
        /// The chunk and what it absorbed.
        ask: protocol::ChunkCommitAsk,
    },
    /// `RewriteChunk`: record a chunk the tier worker rewrote (see `core::chunks`).
    RewriteChunk {
        /// The worker's session.
        session: SessionId,
        /// The request's id, for the reply.
        request_id: u64,
        /// The chunk.
        chunk: Uuid,
        /// Its new frame index.
        frames: Vec<continuum_catalog::ChunkFrame>,
        /// Its new location.
        location: StorageLocator,
    },
    /// Run a long job on the writer, step by step. `reply` says whether it was queued (`Busy` when the writer's
    /// background queue is full).
    Background {
        /// The job.
        job: BackgroundJob,
        /// Whether it was queued.
        reply: oneshot::Sender<Result<(), CoreError>>,
    },
    /// Run a read on the reader pool. `reply` says whether it was queued (`Busy` when the pool's queue is full).
    Read {
        /// The read.
        job: ReadJob,
        /// Whether it was queued.
        reply: oneshot::Sender<Result<(), CoreError>>,
    },
    /// Sent by the server once, before it accepts anything: what an earlier run left (its cost models) and this
    /// boot's id, both read off the scheduler thread.
    Startup {
        /// The cost models the catalog had saved.
        costs: Vec<continuum_sched::CostModelRecord>,
        /// This boot's id, for the clock samples (`None` if it couldn't be read: no samples then).
        boot: Option<continuum_core::BootId>,
    },
    /// Server-only shutdown cleanup, after admission stopped and every socket task was joined or aborted.
    CloseSessions {
        /// Answered after existing per-session cleanup has run for every remaining session.
        reply: oneshot::Sender<()>,
    },
    /// Terminal clean-stop fence: stop the writer and return a movable index only when every prerequisite is settled.
    RestartCheckpoint {
        /// The eligible index and its catalog stamp, or why shutdown must skip the projection.
        reply: oneshot::Sender<RestartCheckpoint>,
    },
    /// Answer once the writer has done every urgent job queued before this one (for tests and diagnostics: the
    /// round trip is how long an ingest waits).
    SyncWriter {
        /// Where the answer goes.
        reply: oneshot::Sender<()>,
    },
    /// The scheduler's cost models now, for tests and diagnostics.
    CostModels {
        /// Where the answer goes.
        reply: oneshot::Sender<Vec<continuum_sched::CostModelRecord>>,
    },
    /// Save the cost models now if they changed, and sample the clock, without waiting for the tick's timers (tests).
    UpkeepNow {
        /// Answered once both are queued for the writer.
        reply: oneshot::Sender<()>,
    },
    /// The streams that are live now, for tests and diagnostics.
    LiveStreams {
        /// Where the answer goes.
        reply: oneshot::Sender<HashSet<StreamId>>,
    },
    /// What the availability index says exists on a stream, for tests and diagnostics.
    Covered {
        /// The stream.
        stream: StreamId,
        /// The range asked about.
        range: Interval,
        /// Where the answer goes.
        reply: oneshot::Sender<Vec<continuum_sched::Covered>>,
    },
    /// `Query`: answered with `QueryResult`, then `PromiseQuote` when something was quoted (see `core::query`).
    Query {
        /// The consumer's session.
        session: SessionId,
        /// For the reply.
        request_id: u64,
        /// The query.
        ask: protocol::QueryAsk,
    },
    /// `CommitPromise`: run the quoted plan. Answered with `PromiseAccepted`.
    CommitPromise {
        /// The consumer's session.
        session: SessionId,
        /// For the reply.
        request_id: u64,
        /// The promise.
        promise: continuum_sched::PromiseId,
        /// The option chosen: 0 for the quote's own plan, n for its nth alternative.
        option: u32,
        /// The band to run in.
        band: PriorityBand,
    },
    /// `CancelPromise`. Answered with `PromiseAccepted`.
    CancelPromise {
        /// The consumer's session.
        session: SessionId,
        /// For the reply.
        request_id: u64,
        /// The promise.
        promise: continuum_sched::PromiseId,
    },
    /// `Subscribe`: follow streams (see `core::subscribe`). Answered with `Subscribed`.
    Subscribe {
        /// The subscriber's session.
        session: SessionId,
        /// For the reply.
        request_id: u64,
        /// The subscription.
        ask: protocol::SubscribeAsk,
    },
    /// `Unsubscribe`. Answered with `Unsubscribed`.
    Unsubscribe {
        /// The subscriber's session.
        session: SessionId,
        /// For the reply.
        request_id: u64,
        /// The subscription's id, as the client numbered it.
        id: u64,
    },
    /// `AckEvents`: a durable subscriber handled everything up to `position`. Nothing is sent back.
    AckEvents {
        /// The subscriber's session.
        session: SessionId,
        /// The subscription's id.
        id: u64,
        /// The position handled.
        position: u64,
    },
    /// `ListStreams`: every registered stream, read on the reader pool. Answered with `StreamList`.
    ListStreams {
        /// The session asking.
        session: SessionId,
        /// For the reply.
        request_id: u64,
    },
    /// `ListPurges`: the purge records, read on the reader pool. Answered with `PurgeList`.
    ListPurges {
        /// The session asking.
        session: SessionId,
        /// For the reply.
        request_id: u64,
        /// Continue after this purge; `None` from the first.
        after: Option<Uuid>,
        /// At most this many (0 for the default).
        limit: u32,
    },
    /// `GetPurge`: one purge's record, read on the reader pool. Answered with `PurgeRecord` or `NotFound`.
    GetPurge {
        /// The session asking.
        session: SessionId,
        /// For the reply.
        request_id: u64,
        /// The purge.
        purge: Uuid,
    },
    /// `PurgeAck`: a worker that keeps copies has removed what a purge named. Nothing is sent back.
    PurgeAck {
        /// The worker's session.
        session: SessionId,
        /// The purge.
        purge: Uuid,
    },
    /// `RememberForgotten`: a forgotten person's name hash into `FORGOTTEN`, through the writer. Answered with
    /// `ForgottenRemembered`.
    RememberForgotten {
        /// The session asking.
        session: SessionId,
        /// For the reply.
        request_id: u64,
        /// The hash.
        hash: continuum_catalog::ForgottenHash,
    },
    /// `SaveWorkerState`, `LoadWorkerState`, or `ForgetWorkerState`: a durable worker's checkpoint as one
    /// catalog row. Saves and deletes go through the writer, loads to the reader pool.
    WorkerState {
        /// The session asking.
        session: SessionId,
        /// For the reply.
        request_id: u64,
        /// Which, and its fields.
        ask: protocol::worker_state::Ask,
    },
    /// `ListForgotten`: a read-only bounded opaque-hash page.
    ListForgotten {
        /// The session asking.
        session: SessionId,
        /// For the reply.
        request_id: u64,
        /// Exclusive hash cursor.
        after: Option<continuum_catalog::ForgottenHash>,
        /// Page size, 1..=1024.
        limit: u32,
    },
    /// `IsForgotten`: whether a hash is in `FORGOTTEN`, read on the reader pool. Answered with `ForgottenStatus`.
    IsForgotten {
        /// The session asking.
        session: SessionId,
        /// For the reply.
        request_id: u64,
        /// The hash.
        hash: continuum_catalog::ForgottenHash,
    },
    /// `ForgetKeeper`: drop a keeper's name, so purges stop waiting for it. Answered with `KeeperForgotten`.
    ForgetKeeper {
        /// The session asking.
        session: SessionId,
        /// For the reply.
        request_id: u64,
        /// The keeper's worker name.
        name: String,
    },
    /// `JobRejected`: a worker can't take a dispatched job. Nothing is sent back; the promises are replanned or
    /// failed.
    JobRejected {
        /// The worker's session.
        session: SessionId,
        /// The job.
        job: continuum_sched::JobId,
        /// Why.
        reason: String,
    },
}

impl Command {
    /// Whether the command goes on the control channel, which the scheduler task always drains first.
    ///
    /// `SyncWriter` stays on the normal channel: the task takes it only once the control channel is empty, so it
    /// still follows every ingest sent before it, and it also follows the `AckEvents` sent before it, which a
    /// control-channel sync could overtake.
    fn is_control(&self) -> bool {
        matches!(
            self,
            Self::Open { .. }
                | Self::Close { .. }
                | Self::Ingest { .. }
                | Self::SetSystemState { .. }
                | Self::QuoteRequest { .. }
                | Self::RenewLease { .. }
                | Self::CommitLease { .. }
                | Self::AbortLease { .. }
                | Self::LeaseReleased { .. }
        )
    }
}

/// Why the core didn't take a command.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CoreError {
    /// The core has stopped.
    #[error("the broker's core has stopped")]
    Gone,
    /// The core's queue for this kind of command is full; try again shortly.
    #[error("the broker is busy; try again shortly")]
    Busy,
}

/// A cheap, cloneable way to send the core commands.
#[derive(Clone, Debug)]
pub struct CoreHandle {
    /// The voice turn only: polled ahead of everything else, writer results included, for the 5 ms preemption.
    p0: mpsc::Sender<Command>,
    control: mpsc::Sender<Command>,
    normal: mpsc::Sender<Command>,
}

impl CoreHandle {
    /// Sends a command, waiting if the core's channel for it is full. Only the session that sent it waits (and so
    /// stops reading its own socket meanwhile); the core drains its channels without ever blocking, so the wait is
    /// short, and nobody else is held up.
    ///
    /// # Errors
    ///
    /// [`CoreError::Gone`] once the core has stopped.
    pub async fn send(&self, command: Command) -> Result<(), CoreError> {
        // A session's Open shares the voice turn's channel: the client can send a voice turn as soon as Welcome
        // arrives, and one channel keeps the two in order, so the new session is open when the turn is broadcast.
        let tx = if matches!(command, Command::SetSystemState { .. } | Command::Open { .. }) {
            &self.p0
        } else if command.is_control() {
            &self.control
        } else {
            &self.normal
        };
        tx.send(command).await.map_err(|_| CoreError::Gone)
    }

    /// Queues a long job on the writer.
    ///
    /// # Errors
    ///
    /// [`CoreError::Busy`] when the writer's background queue is full, [`CoreError::Gone`] once the core stopped.
    pub async fn background(&self, job: BackgroundJob) -> Result<(), CoreError> {
        let (reply, answer) = oneshot::channel();
        self.send(Command::Background { job, reply }).await?;
        answer.await.map_err(|_| CoreError::Gone)?
    }

    /// Queues a read on the reader pool.
    ///
    /// # Errors
    ///
    /// [`CoreError::Busy`] when the pool's queue is full, [`CoreError::Gone`] once the core stopped.
    pub async fn read(&self, job: ReadJob) -> Result<(), CoreError> {
        let (reply, answer) = oneshot::channel();
        self.send(Command::Read { job, reply }).await?;
        answer.await.map_err(|_| CoreError::Gone)?
    }

    /// The scheduler's cost models now.
    ///
    /// # Errors
    ///
    /// [`CoreError::Gone`] once the core has stopped.
    pub async fn cost_models(&self) -> Result<Vec<continuum_sched::CostModelRecord>, CoreError> {
        let (reply, answer) = oneshot::channel();
        self.send(Command::CostModels { reply }).await?;
        answer.await.map_err(|_| CoreError::Gone)
    }

    /// Saves the cost models now if they changed, and samples the clock; follow it with [`CoreHandle::sync_writer`]
    /// to wait for both to be written.
    ///
    /// # Errors
    /// [`CoreError`] if the core is gone or busy.
    pub async fn upkeep_now(&self) -> Result<(), CoreError> {
        let (reply, answer) = oneshot::channel();
        self.send(Command::UpkeepNow { reply }).await?;
        answer.await.map_err(|_| CoreError::Gone)
    }

    /// The streams that are live now.
    ///
    /// # Errors
    ///
    /// [`CoreError::Gone`] once the core has stopped.
    pub async fn live_streams(&self) -> Result<HashSet<StreamId>, CoreError> {
        let (reply, answer) = oneshot::channel();
        self.send(Command::LiveStreams { reply }).await?;
        answer.await.map_err(|_| CoreError::Gone)
    }

    /// Waits until the writer has done every urgent job queued before now, and has written every durable
    /// subscription position acknowledged before now.
    ///
    /// # Errors
    ///
    /// [`CoreError::Gone`] once the core has stopped, or [`CoreError::Busy`] if the writer's urgent queue is full.
    pub async fn sync_writer(&self) -> Result<(), CoreError> {
        let (reply, answer) = oneshot::channel();
        self.send(Command::SyncWriter { reply }).await?;
        answer.await.map_err(|_| CoreError::Busy)
    }

    /// Close all remaining sessions after the server has stopped accepting and aborted/joined its tasks.
    /// Normal per-session cleanup releases producers, workers, subscriptions, pins, and keepers.
    /// # Errors
    /// The scheduler stopped before acknowledging cleanup.
    pub async fn close_sessions(&self) -> Result<(), CoreError> {
        let (reply, answer) = oneshot::channel();
        self.send(Command::CloseSessions { reply }).await?;
        answer.await.map_err(|_| CoreError::Gone)
    }

    /// Terminal shutdown barrier. No sessions may remain; the writer stops even if no checkpoint is safe.
    ///
    /// # Errors
    /// Returns `Gone` if the scheduler stopped or `Busy` if the barrier could not be queued.
    pub async fn restart_checkpoint(&self) -> Result<RestartCheckpoint, CoreError> {
        let (reply, answer) = oneshot::channel();
        self.send(Command::RestartCheckpoint { reply }).await?;
        answer.await.map_err(|_| CoreError::Busy)
    }

    /// What the availability index says exists on `stream` over `range`.
    ///
    /// # Errors
    ///
    /// [`CoreError::Gone`] once the core has stopped.
    pub async fn covered(&self, stream: StreamId, range: Interval) -> Result<Vec<continuum_sched::Covered>, CoreError> {
        let (reply, answer) = oneshot::channel();
        self.send(Command::Covered { stream, range, reply }).await?;
        answer.await.map_err(|_| CoreError::Gone)
    }
}

/// The error for a job a full or closed queue didn't take.
fn queue_error<T>(e: &TrySendError<T>) -> CoreError {
    match e {
        TrySendError::Full(_) => CoreError::Busy,
        TrySendError::Disconnected(_) => CoreError::Gone,
    }
}

/// What the writer reports back to the scheduler task after a commit.
enum Event {
    /// An aging read, answered (see `core::aging`).
    Aging {
        session: SessionId,
        request_id: u64,
        result: Result<aging::Done, (ErrorCode, String)>,
    },
    /// The catalog accepted or deferred a purge's final completion.
    PurgeCompleted {
        id: Uuid,
        result: Result<(), CatalogError>,
    },

    /// Purged slots a live lease still held, to try again on the next scan.
    SlotsToPurge(Vec<continuum_core::SlotHandle>),
    /// A purge step committed: the index drops what it doomed.
    PurgeStep {
        id: Uuid,
        delta: PurgeDelta,
    },
    /// A purge finished (its last step's delta came first, as a `PurgeStep`).
    PurgeDone {
        requester: Option<(SessionId, u64)>,
        report: PurgeReport,
        /// What kind of purge, from its record (`None` when the record couldn't be read).
        kind: Option<continuum_catalog::PurgeKind>,
        /// The keepers that acknowledged it in an earlier run, from its record.
        acked: Vec<String>,
    },
    /// A purge couldn't start or failed partway.
    PurgeFailed {
        id: Uuid,
        requester: Option<(SessionId, u64)>,
        code: ErrorCode,
        message: String,
    },
    Registered {
        session: SessionId,
        request_id: u64,
        name: String,
        ring: Option<protocol::RingAsk>,
        result: Result<(StreamId, StreamMeta), CatalogError>,
    },
    Ingested(Vec<Ingested>),
    Synced(oneshot::Sender<()>),
    RestartCheckpoint {
        stamp: Result<continuum_catalog::RestartStamp, CheckpointRefusal>,
        reply: oneshot::Sender<RestartCheckpoint>,
    },
    /// The writer couldn't save these cost models.
    CostsNotSaved(Vec<u8>),
    /// The writer finished a tick's durable cursor writes: the positions it saved, and the names it couldn't.
    CursorsWritten {
        saved: Vec<(String, u64)>,
        failed: Vec<String>,
    },
    /// A lease's outputs committed (or didn't).
    Committed {
        session: SessionId,
        request_id: u64,
        lease: u64,
        /// Each output's id, stream, and span, for the scheduler (repeats included, outputs hidden on landing not).
        outputs: Vec<(Uuid, StreamId, Interval)>,
        /// The copies the outputs added, for the index: stored now, not repeats, and not hidden on landing. Applied
        /// even when `result` is an error, since a coverage failure comes after the outputs are durable.
        copies: Vec<(StreamId, Interval, StorageTier)>,
        /// Outputs that landed under a running purge, hidden from the start.
        hidden: Vec<Uuid>,
        /// The outputs stored now (not repeats, not hidden) with their copy and commit sequence, for subscribers.
        published: Vec<(HeaderBuf, Option<StorageLocator>, u64)>,
        /// Each output's commit sequence in the outputs' order, 0 for one hidden on landing; empty when the insert
        /// failed.
        seqs: Vec<u64>,
        coverage: Option<(StreamId, continuum_core::StageId, Interval)>,
        cost: continuum_sched::ComputeCost,
        result: Result<(), (ErrorCode, String)>,
    },
    PayloadCopyCommitted {
        session: SessionId,
        request_id: u64,
        payload: Uuid,
        result: Result<Option<(StreamId, Interval)>, (ErrorCode, String)>,
    },
    /// A chunk was committed (or wasn't), with the copy it adds to the index and its frames' slots (the ones it
    /// landed without stay out of the index).
    ChunkCommitted {
        session: SessionId,
        request_id: u64,
        chunk: Uuid,
        copy: Option<(StreamId, Interval, StorageTier, Vec<Interval>)>,
        result: Result<continuum_catalog::Inserted, (ErrorCode, String)>,
    },
    /// A rewritten chunk was recorded (or wasn't). `added` is the copy it gained, when the rewrite moved it into a
    /// tier it had no copy in: its stream, span, tier, and the slots of frames it holds no bytes for.
    ChunkRewritten {
        session: SessionId,
        request_id: u64,
        chunk: Uuid,
        added: Option<(StreamId, Interval, StorageTier, Vec<Interval>)>,
        result: Result<(), (ErrorCode, String)>,
    },
    /// A query's items were read (see `core::query`).
    QueryRead {
        reply: QueryReply,
        items: Result<query::QueryItems, String>,
    },
    /// A job's inputs were read; the `DispatchJob` can go out.
    Staged {
        dispatch: Dispatch,
        inputs: Result<Vec<ReadItem>, String>,
    },
    /// A fulfilment's outputs were read; the part can go out.
    Fulfilled {
        consumer: ConsumerId,
        promise: continuum_sched::PromiseId,
        items: Result<Vec<ReadItem>, String>,
        final_part: bool,
        ordinal: usize,
    },
    /// The stream listing was read (`core::query`).
    StreamsRead {
        session: SessionId,
        request_id: u64,
        result: query::StreamListingResult,
    },
    /// The purge listing was read (`core::forget`).
    PurgesRead {
        session: SessionId,
        request_id: u64,
        result: Result<Vec<continuum_catalog::PurgeSummary>, (ErrorCode, String)>,
    },
    /// One purge's record was read.
    PurgeRecordRead {
        session: SessionId,
        request_id: u64,
        purge: Uuid,
        result: Result<Option<continuum_catalog::PurgeRecord>, (ErrorCode, String)>,
    },
    /// The writer recorded a forgotten name's hash (or didn't).
    ForgottenRemembered {
        session: SessionId,
        request_id: u64,
        result: Result<(), (ErrorCode, String)>,
    },
    /// A worker state request was answered (or refused).
    WorkerStateDone {
        session: SessionId,
        request_id: u64,
        result: Result<protocol::worker_state::Done, (ErrorCode, String)>,
    },
    /// A bounded page of forgotten hashes was read.
    ForgottenListed {
        session: SessionId,
        request_id: u64,
        result: Result<continuum_catalog::ForgottenPage, (ErrorCode, String)>,
    },
    /// Whether a hash is in `FORGOTTEN` was read.
    ForgottenRead {
        session: SessionId,
        request_id: u64,
        result: Result<bool, (ErrorCode, String)>,
    },
    /// A durable subscription's saved position was read; the subscription can start.
    CursorRead {
        session: SessionId,
        request_id: u64,
        ask: protocol::SubscribeAsk,
        position: Result<Option<u64>, String>,
    },
    /// One round of catch-up reads for a subscription (see `core::subscribe`).
    CaughtUpPage {
        session: SessionId,
        id: u64,
        /// The subscription's token, so a page read for an earlier subscription under the same id is ignored.
        token: u64,
        pages: Result<Vec<CatchUpPage>, String>,
    },
}

struct Ingested {
    session: SessionId,
    request_id: u64,
    stream: StreamId,
    id: Uuid,
    /// The payload's span and the tier of its copy, for the index (when it has a copy).
    copy: Option<(Interval, StorageTier)>,
    /// The catalog hid it as it landed (its parent is being purged): it stays out of the index.
    hidden: bool,
    /// Indexed already with identical bytes: nothing new for the index or subscribers, only the reply.
    repeat: bool,
    /// The header and copy as ingested, for subscribers.
    header: HeaderBuf,
    locator: Option<StorageLocator>,
    /// Its commit sequence (0 when it wasn't stored), the position subscribers see.
    seq: u64,
    result: Result<(), (ErrorCode, String)>,
}

/// A job for the writer's urgent queue.
enum Urgent {
    RestartCheckpoint {
        reply: oneshot::Sender<RestartCheckpoint>,
    },
    Register {
        session: SessionId,
        request_id: u64,
        info: StreamInfo,
        ring: Option<protocol::RingAsk>,
    },
    Ingest {
        session: SessionId,
        request_id: u64,
        header: HeaderBuf,
        locator: Option<StorageLocator>,
        dependencies: Vec<continuum_catalog::Dependency>,
    },
    Sync(oneshot::Sender<()>),
    Commit {
        session: SessionId,
        request_id: u64,
        commit: protocol::Commit,
    },
    /// A durable subscription's acknowledged position.
    Cursor {
        name: String,
        position: u64,
    },
    CommitPayloadCopy {
        session: SessionId,
        request_id: u64,
        payload: Uuid,
        location: StorageLocator,
        keeper: String,
    },
    /// A Tier 1 chunk to commit (see `core::chunks`).
    CommitChunk {
        session: SessionId,
        request_id: u64,
        ask: protocol::ChunkCommitAsk,
    },
    /// A rewritten chunk to record (see `core::chunks`).
    RewriteChunk {
        session: SessionId,
        request_id: u64,
        chunk: Uuid,
        frames: Vec<continuum_catalog::ChunkFrame>,
        location: StorageLocator,
    },
    /// Every copy of what a purge removed is gone: mark it complete.
    CompletePurge {
        id: Uuid,
        now_ns: u64,
    },
    /// A worker that keeps copies registered under a new name: remember it, so purges wait for it from now on.
    AddKeeper(String),
    /// A keeper's name was retired (`ForgetKeeper`).
    ForgetKeeper(String),
    /// A keeper acknowledged a purge's notice: record it, so a restart doesn't wait for it again.
    PurgeAcked {
        id: Uuid,
        name: String,
    },
    /// A forgotten person's name hash into `FORGOTTEN`, answered to the session.
    RememberForgotten {
        session: SessionId,
        request_id: u64,
        hash: continuum_catalog::ForgottenHash,
    },
    /// A worker state save or delete, answered to the session.
    WorkerState {
        session: SessionId,
        request_id: u64,
        ask: protocol::worker_state::Ask,
    },
    /// Save the scheduler's cost models (see [`crate::costs`]).
    SaveCosts(Vec<u8>),
    /// Record a `CLOCK_MAP` point.
    Clock(continuum_core::BootId, continuum_catalog::ClockPoint),
}

/// Starts the core: the scheduler task on the current runtime, the writer thread, and `readers` reader threads.
/// Everything stops once every [`CoreHandle`] is dropped.
///
/// # Panics
///
/// If the OS refuses to start a thread, or this isn't called inside a tokio runtime.
#[must_use]
#[expect(clippy::too_many_arguments, reason = "each is one of the core's parts; the server is the one caller")]
pub fn spawn(
    catalog: Arc<dyn Catalog>,
    readers: usize,
    writer_config: WriterConfig,
    index: AvailabilityIndex,
    pending_purges: Vec<Uuid>,
    known_keepers: Vec<String>,
    profile: Profile,
    tier0: Arc<crate::tier0::Tier0>,
    forget: ForgetConfig,
) -> CoreHandle {
    let (p0_tx, p0_rx) = mpsc::channel(COMMANDS);
    let (control_tx, control_rx) = mpsc::channel(COMMANDS);
    let (normal_tx, normal_rx) = mpsc::channel(COMMANDS);
    let (events_tx, events_rx) = mpsc::unbounded_channel();
    let (urgent_tx, urgent_rx) = crossbeam_channel::bounded(URGENT);
    let (background_tx, background_rx) = crossbeam_channel::bounded(BACKGROUND);
    let (read_tx, read_rx) = crossbeam_channel::bounded::<ReadJob>(READS);

    let events_tx_for_purges = events_tx.clone();
    let active_purges = Arc::new(AtomicUsize::new(0));
    let writer_health = Arc::new(writer_health::WriterHealth::default());
    let writer_active = Arc::clone(&active_purges);
    let writer_health_for_thread = Arc::clone(&writer_health);
    // Purges a previous run left pending go straight onto the writer's list of long jobs (not through the bounded
    // queue), so however many there are, none is dropped.
    let resumed: Vec<BackgroundJob> = pending_purges
        .into_iter()
        .map(|id| {
            purge_job(None, None, id, events_tx_for_purges.clone(), &active_purges, &writer_health, writer_config)
        })
        .collect();
    let writer_catalog = Arc::clone(&catalog);
    std::thread::Builder::new()
        .name("continuum-writer".into())
        .spawn(move || {
            // A broker whose writer died would refuse every write forever while looking healthy to its supervisor,
            // so a panic here is fatal: abort, and let systemd or launchd restart the broker. (A long job's own panic
            // is caught in the loop and only drops that job.)
            let run = || {
                writer(
                    &*writer_catalog,
                    &urgent_rx,
                    &background_rx,
                    &events_tx,
                    &writer_config,
                    &writer_active,
                    resumed,
                    &writer_health_for_thread,
                )
            };
            if std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)).is_err() {
                tracing::error!("the catalog writer panicked; stopping the broker so its supervisor restarts it");
                std::process::abort();
            }
        })
        .expect("starting the writer thread");
    for n in 0..readers.max(1) {
        let (catalog, jobs) = (Arc::clone(&catalog), read_rx.clone());
        std::thread::Builder::new()
            .name(format!("continuum-reader-{n}"))
            .spawn(move || {
                while let Ok(job) = jobs.recv() {
                    // A read that panics loses only itself; the thread keeps serving the pool.
                    if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job(&*catalog))).is_err() {
                        tracing::error!("a catalog read panicked");
                    }
                }
            })
            .expect("starting a reader thread");
    }

    // Purges a previous run left unfinished carry on, with nobody to answer.
    let state = Sched {
        sessions: HashMap::new(),
        producers: HashMap::new(),
        index,
        events: events_tx_for_purges,
        active_purges,
        writer_health,
        writer_config,
        urgent: urgent_tx,
        background: background_tx,
        reads: ReadQueue::new(read_tx),
        builder: FlatBufferBuilder::new(),
        engine: Engine::new(profile),
        voice: VoiceTurnState::Idle,
        workers: HashMap::new(),
        worker_sessions: HashMap::new(),
        next_worker: 0,
        leases: HashMap::new(),
        tier0,
        reader_pins: HashMap::new(),
        ring_batches: HashMap::new(),
        checkpointing: false,
        checkpoint_refusal: None,
        upkeep: upkeep::Upkeep::default(),
        slots: slots::Slots::default(),
        archive_health: ArchiveHealth::default(),
        worker_names: HashMap::new(),
        consumers: HashMap::new(),
        consumer_sessions: HashMap::new(),
        next_consumer: 0,
        pins: HashMap::new(),
        staging: HashSet::new(),
        delivered: HashMap::new(),
        fulfillments: HashMap::new(),
        cap_names: HashMap::new(),
        jobs: HashMap::new(),
        promise_owners: HashMap::new(),
        subs: HashMap::new(),
        cursors: subscribe::CursorAcks::default(),
        sub_tokens: 0,
        subs_by_stream: HashMap::new(),
        head_seq: catalog.head_seq().unwrap_or(0),
        forget: Forget::new(forget, known_keepers),
    };
    // The scheduler task gets a thread and a runtime of its own, so nothing the sessions do on the main runtime
    // (verifying large frames, say) can delay it: the 5 ms preemption depends on it never waiting for a worker.
    std::thread::Builder::new()
        .name("continuum-sched".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_time()
                .build()
                .expect("building the scheduler's runtime");
            let run = || rt.block_on(state.run(p0_rx, control_rx, normal_rx, events_rx));
            if std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)).is_err() {
                tracing::error!("the scheduler task panicked; stopping the broker so its supervisor restarts it");
                std::process::abort();
            }
        })
        .expect("starting the scheduler thread");
    CoreHandle { p0: p0_tx, control: control_tx, normal: normal_tx }
}

/// What the scheduler task keeps about an open session.
struct Session {
    outbound: mpsc::Sender<Vec<u8>>,
    /// Room in `outbound` promised to catch-up rounds still being read (`Sched::catch_up_round`).
    reserved: usize,
    kicked: Arc<tokio::sync::Notify>,
    /// Streams this session registered, which are live while it is open.
    streams: HashSet<StreamId>,
    /// The client's process id, from its `Hello`.
    pid: u32,
}

struct Sched {
    sessions: HashMap<SessionId, Session>,
    /// How many open sessions registered each stream.
    producers: HashMap<StreamId, usize>,
    index: AvailabilityIndex,
    /// For purge jobs, which report their steps to this task.
    events: mpsc::UnboundedSender<Event>,
    /// How many purges are queued or running (see [`ActivePurge`]).
    active_purges: Arc<AtomicUsize>,
    /// Memory-only writer progress, shared with the writer thread for periodic diagnostics.
    writer_health: Arc<writer_health::WriterHealth>,
    /// Work-window and operation ceiling captured consistently for new and resumed purge jobs.
    writer_config: WriterConfig,
    urgent: Sender<Urgent>,
    background: Sender<LongJob>,
    reads: ReadQueue,
    checkpointing: bool,
    /// The first reason this shutdown's fence can't publish a projection, once one is known.
    checkpoint_refusal: Option<CheckpointRefusal>,
    builder: FlatBufferBuilder<'static>,
    /// The scheduler: promises, leases, bands, and the cost model.
    engine: Engine,
    /// The current turn, captured when a session opens for its Welcome.
    voice: VoiceTurnState,
    /// The worker each session registered, and the session of each worker.
    workers: HashMap<SessionId, WorkerId>,
    worker_sessions: HashMap<WorkerId, SessionId>,
    next_worker: u64,
    /// Each live lease's worker and output stream, so a lease message from any other session, or an output on any
    /// other stream, is refused before it reaches the catalog or the engine.
    leases: HashMap<LeaseId, (WorkerId, StreamId)>,
    /// The Tier 0 arena, for the pins query results hand out.
    tier0: Arc<crate::tier0::Tier0>,
    /// The Tier 0 pins each session holds: released when the session ends, and dropped once expired.
    reader_pins: HashMap<SessionId, Vec<continuum_shm::Lease>>,
    /// Each ring's indexed batches, oldest first, so the tick can drop the ones the ring has lapped.
    ring_batches: HashMap<u16, std::collections::VecDeque<RingBatch>>,
    /// When cost models were last saved and the clock last sampled.
    upkeep: upkeep::Upkeep,
    /// The Tier 0 slots the index counts.
    slots: slots::Slots,
    /// Process-local loss and pressure diagnostics; never a persisted loss audit.
    archive_health: ArchiveHealth,
    /// Each worker's name from its manifest, for the plan steps in a quote.
    worker_names: HashMap<WorkerId, String>,
    /// The consumer each querying session is, and back.
    consumers: HashMap<SessionId, ConsumerId>,
    consumer_sessions: HashMap<ConsumerId, SessionId>,
    next_consumer: u64,
    /// What was pinned for each job's lease (released on `Unpin`).
    pins: HashMap<LeaseId, Vec<StorageLocator>>,
    /// Dispatches whose catalog snapshots have not yet become physical pins.
    staging: HashSet<LeaseId>,
    /// Items delivered so far per promise, for `PromiseFailed`.
    delivered: HashMap<continuum_sched::PromiseId, u32>,
    /// Issued parts and completed reads waiting for earlier parts of the same promise.
    fulfillments: HashMap<PromiseId, query::FulfillmentOrder>,
    /// Each accepted capability's name for `DispatchJob` (the stream it produces; empty for a hydrate).
    cap_names: HashMap<CapabilityId, String>,
    /// The worker and lease of each dispatched job, so only that worker can reject it; gone with the lease.
    jobs: HashMap<JobId, (WorkerId, LeaseId)>,
    /// The session each quoted promise belongs to, so only it may commit or cancel; gone when the promise ends.
    promise_owners: HashMap<PromiseId, SessionId>,
    /// Live subscriptions by (session, the client's subscription id), and the ones on each stream.
    subs: HashMap<(SessionId, u64), Subscription>,
    /// Durable positions acknowledged but not yet saved, at most one queued write per name.
    cursors: subscribe::CursorAcks,
    /// The last subscription token handed out (each subscription gets the next).
    sub_tokens: u64,
    subs_by_stream: HashMap<StreamId, Vec<(SessionId, u64)>>,
    /// The newest commit sequence this task has seen, the position a live subscription starts after.
    head_seq: u64,
    /// Purges waiting for the workers that keep copies (`core::forget`).
    forget: Forget,
}

/// A ring batch the index has a Tier 0 copy of.
#[derive(Clone, Copy, Debug)]
struct RingBatch {
    /// The payload the batch is.
    id: Uuid,
    /// A purge took its copy off the index already: the sweep has nothing to remove, but the batch still marks how far
    /// the ring is sealed, so its records never come back as an unindexed tail.
    purged: bool,
    first_seq: u64,
    count: u32,
    stream: StreamId,
    span: Interval,
}

#[derive(Default)]
struct ArchiveHealth {
    slot_gaps: u64,
    ring_gaps: u64,
    pressure_events: u64,
    pressure_classes: HashSet<u16>,
}

impl ArchiveHealth {
    /// Warn once on entering pressure for a class, until a later scan clears it.
    fn pressure(&mut self, class: u16, unflushed: usize) {
        if unflushed == 0 {
            self.pressure_classes.remove(&class);
        } else if self.pressure_classes.insert(class) {
            self.pressure_events = self.pressure_events.saturating_add(1);
            warn!(
                class,
                unflushed_slots = unflushed,
                tier0_reclaim_pressure_events = self.pressure_events,
                "Tier 0 archive pressure: retaining unflushed slots for Tier 1"
            );
        }
    }

    fn slot_gap(&mut self, stream: StreamId, span: Interval, reason: &'static str) {
        self.slot_gaps = self.slot_gaps.saturating_add(1);
        warn!(
            stream = stream.0,
            start_ns = span.start_ns(),
            end_ns = span.end_ns(),
            reason,
            tier0_slot_archive_gaps = self.slot_gaps,
            "a Tier 0 slot was overwritten before it reached Tier 1: a gap in the archive"
        );
    }

    fn ring_gap(&mut self, stream: StreamId, span: Interval) {
        self.ring_gaps = self.ring_gaps.saturating_add(1);
        warn!(
            stream = stream.0,
            start_ns = span.start_ns(),
            end_ns = span.end_ns(),
            tier0_ring_archive_gaps = self.ring_gaps,
            "a ring batch was overwritten before it reached Tier 1: a gap in the archive"
        );
    }
}

/// The index lets go of a batch the ring lapped. Losing it is an archive gap only for a stream owed to the archive.
fn remove_lapped_batch(index: &mut AvailabilityIndex, batch: &RingBatch, owed: bool, health: &mut ArchiveHealth) {
    if batch.purged || index.remove_copy(batch.stream, batch.span, StorageTier::Tier0Hot).is_err() || !owed {
        return;
    }
    let kept: u64 = index.covered(batch.stream, batch.span).iter().map(|copy| copy.interval.len_ns()).sum();
    if kept < batch.span.len_ns() {
        health.ring_gap(batch.stream, batch.span);
    }
}

fn log_health_line(sessions: usize, active_purges: usize, health: &writer_health::Snapshot, archive: &ArchiveHealth) {
    let purge_steps: Vec<_> = health.samples.iter().map(|p| (p.id.to_string(), p.steps)).collect();
    tracing::info!(
        sessions,
        background_queue_depth = health.queued,
        background_running = health.running,
        active_purge_jobs = active_purges,
        unfinished_purges = health.purges,
        oldest_unfinished_purge_age_s = ?health.oldest_age.map(|age| age.as_secs()),
        purge_age_unknown = health.unknown_age,
        stalled_purge_jobs = health.stalled,
        purge_steps_this_run = ?purge_steps,
        purge_steps_omitted = health.omitted,
        oldest_unfinished_by_stream = ?health
            .oldest_streams
            .iter()
            .map(|(name, age)| (name.clone(), age.as_secs()))
            .collect::<Vec<_>>(),
        tier0_slot_archive_gaps = archive.slot_gaps,
        tier0_ring_archive_gaps = archive.ring_gaps,
        tier0_reclaim_pressure_events = archive.pressure_events,
        tier0_reclaim_pressure_classes = archive.pressure_classes.len(),
        "broker health"
    );
}

/// How often the scheduler task lets time pass in the engine (quote, lease, and promise expiry).
const TICK_EVERY: Duration = Duration::from_millis(50);
/// Cheap writer health snapshots, independent of how busy the command/event queues stay.
const HEALTH_EVERY: Duration = Duration::from_secs(60);

impl Sched {
    fn log_writer_health(&self, now: Instant) {
        let health = self.writer_health.snapshot(now, self.background.len());
        log_health_line(self.sessions.len(), self.active_purges.load(Ordering::SeqCst), &health, &self.archive_health);
        let old_purges: Vec<_> = health
            .warnings
            .into_iter()
            .map(|purge| {
                (
                    purge.id.to_string(),
                    purge.age.map(|age| age.as_secs()),
                    purge.steps,
                    purge.stalled,
                    purge.elapsed,
                    purge.work,
                )
            })
            .collect();
        if !old_purges.is_empty() {
            warn!(
                ?old_purges,
                "catalog purges older than 120 seconds (id, age_s, steps, stalled, since_enqueue, catalog_step_work)"
            );
        }
    }

    async fn run(
        mut self,
        mut p0: mpsc::Receiver<Command>,
        mut control: mpsc::Receiver<Command>,
        mut normal: mpsc::Receiver<Command>,
        mut events: mpsc::UnboundedReceiver<Event>,
    ) {
        let (mut control_open, mut normal_open) = (true, true);
        let mut tick = tokio::time::interval(TICK_EVERY);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let mut next_health = Instant::now() + HEALTH_EVERY;
        while control_open || normal_open {
            let now = Instant::now();
            if now >= next_health {
                self.log_writer_health(now);
                next_health = now + HEALTH_EVERY;
            }
            tokio::select! {
                biased;
                // The voice turn first, ahead even of the writer's results, so a preemption never waits behind a
                // batch of committed ingests.
                Some(c) = p0.recv() => self.command(c),
                Some(event) = events.recv() => self.event(event),
                command = control.recv(), if control_open => match command {
                    Some(c) => self.command(c),
                    None => control_open = false,
                },
                command = normal.recv(), if normal_open => match command {
                    Some(c) => self.command(c),
                    None => normal_open = false,
                },
                _ = tick.tick(), if !self.checkpointing => {
                    // An expired pin no longer holds its slot; dropping it is a shared memory write, never I/O.
                    self.reader_pins.retain(|_, pins| {
                        pins.retain(continuum_shm::Lease::is_valid);
                        !pins.is_empty()
                    });
                    self.tick(now_ns());
                    self.subscribe_tick();
                    self.sweep_rings();
                    self.forget_tick(now_ns());
                    self.upkeep_tick();
                    self.slots_tick();
                }
            }
        }
        debug!("scheduler task stopping: every handle is gone");
    }

    /// The first reason the core isn't settled enough to publish a restart projection, or `None` when it is.
    /// `fence_settles` leaves out what the fence itself settles before the writer answers: unsaved cursor positions
    /// (it flushes them ahead of itself) and a running `FRAMES` sweep (the writer drops it). The final check, once
    /// the writer has answered, passes `false` and so checks those too.
    fn checkpoint_refusal(&self, fence_settles: bool) -> Option<CheckpointRefusal> {
        use CheckpointRefusal as R;
        if !self.sessions.is_empty() {
            Some(R::SessionsOpen)
        } else if self.active_purges.load(Ordering::SeqCst) != 0 {
            Some(R::PurgeRunning)
        } else if self.reads.outstanding.load(Ordering::SeqCst) != 0 {
            Some(R::ReadsOutstanding)
        } else if !self.staging.is_empty() {
            Some(R::Staging)
        } else if !self.forget.restart_checkpoint_clean() {
            Some(R::ForgetUnsettled)
        } else if !fence_settles && !self.upkeep.restart_checkpoint_clean() {
            Some(R::FrameSweep)
        } else if !fence_settles && !self.cursors.restart_checkpoint_clean() {
            Some(R::CursorsUnsaved)
        } else {
            self.index.restart_checkpoint_refusal()
        }
    }

    /// Keeps the first reason this shutdown can't publish a projection.
    fn refuse_checkpoint(&mut self, reason: CheckpointRefusal) {
        self.checkpoint_refusal.get_or_insert(reason);
    }

    fn command(&mut self, command: Command) {
        if self.checkpointing {
            return;
        }
        match command {
            Command::CloseSessions { reply } => {
                let sessions: Vec<_> = self.sessions.keys().copied().collect();
                for session in sessions {
                    self.close(session);
                }
                let _ = reply.send(());
            }
            Command::RestartCheckpoint { reply } => {
                // Not settled yet: dropping the reply answers Busy, and the server asks again within its budget.
                // Reads still running for clients that already left (a long query, a catch-up page) finish on the
                // reader pool; nothing new can start once the sessions are gone.
                if !self.sessions.is_empty() || self.reads.outstanding.load(Ordering::SeqCst) != 0 {
                    return;
                }
                self.checkpointing = true;
                self.reads.stopped.store(true, std::sync::atomic::Ordering::SeqCst);
                if let Some(reason) = self.checkpoint_refusal(true) {
                    self.refuse_checkpoint(reason);
                }
                // Flush any positions acknowledged since the server's preceding drain, ahead of the fence. Their
                // saves come back before the fence's answer, and the final check sees whether they all landed.
                for (name, position) in self.cursors.unqueued() {
                    if self.urgent.try_send(Urgent::Cursor { name: name.clone(), position }).is_err() {
                        let _ = reply.send(Err(CheckpointRefusal::WriterQueueFull));
                        return;
                    }
                    self.cursors.mark_sent(name, position);
                }
                if let Err(
                    TrySendError::Full(Urgent::RestartCheckpoint { reply })
                    | TrySendError::Disconnected(Urgent::RestartCheckpoint { reply }),
                ) = self.urgent.try_send(Urgent::RestartCheckpoint { reply })
                {
                    let _ = reply.send(Err(CheckpointRefusal::WriterQueueFull));
                }
            }

            Command::Open { session, outbound, kicked, pid, voice } => {
                self.sessions.insert(session, Session { outbound, reserved: 0, kicked, streams: HashSet::new(), pid });
                let _ = voice.send(self.voice);
            }
            Command::Close { session } => self.close(session),
            Command::RegisterStream { session, request_id, info, ring } => {
                if let Err(e) = self.urgent.try_send(Urgent::Register { session, request_id, info, ring }) {
                    self.refuse_busy(session, request_id, &e);
                }
            }
            Command::Ingest { session, request_id, header, locator, dependencies } => {
                self.ingest(session, request_id, header, locator, dependencies)
            }
            Command::RegisterWorker { session, request_id, ask } => self.register_worker(session, request_id, ask),
            Command::SetSystemState { voice } => {
                self.voice = voice;
                let now = now_ns();
                let actions = self.engine.set_system_state(SystemState::VoiceTurn(voice), now);
                // Preempt first: the 5 ms bound is on the preemptions, the broadcast can follow.
                self.run_actions(actions);
                let bytes = protocol::system_state_changed(&mut self.builder, voice, now);
                let everyone: Vec<SessionId> = self.sessions.keys().copied().collect();
                for s in everyone {
                    self.push(s, bytes.clone());
                }
            }
            Command::QuoteRequest { session, request_id, req } => {
                let Some(&worker) = self.workers.get(&session) else {
                    return self.reply_error(session, request_id, ErrorCode::NotAllowed, "register a worker first");
                };
                match self.engine.quote_request(worker, req, now_ns()) {
                    Ok(outcome) => {
                        if let LeaseDecision::Granted(l) = &outcome.decision {
                            self.leases.insert(l.lease, (worker, l.stream));
                        }
                        let bytes = protocol::lease_grant(&mut self.builder, request_id, &outcome.decision);
                        self.push(session, bytes);
                        self.run_actions(outcome.actions);
                    }
                    Err(e) => self.reply_error(session, request_id, ErrorCode::BadRequest, &e.to_string()),
                }
            }
            Command::RenewLease { session, request_id, lease, extend_ms } => {
                if !self.owns(session, lease) {
                    return self.reply_error(session, request_id, ErrorCode::NotAllowed, "not this session's lease");
                }
                match self.engine.renew_lease(LeaseId(lease), u64::from(extend_ms) * 1_000_000, now_ns()) {
                    Ok(expires) => {
                        let bytes = protocol::lease_renewed(&mut self.builder, request_id, lease, expires);
                        self.push(session, bytes);
                    }
                    Err(e) => self.reply_error(session, request_id, ErrorCode::NotFound, &e.to_string()),
                }
            }
            Command::CommitLease { session, request_id, commit } => {
                if !self.owns(session, commit.lease) {
                    return self.reply_error(session, request_id, ErrorCode::NotAllowed, "not this session's lease");
                }
                let stream = self.leases.get(&LeaseId(commit.lease)).map(|&(_, s)| s);
                // A hydrate lease (output stream 0) decodes whichever stream it was given into Tier 0, so its outputs
                // may be on any stream; every other lease's outputs must be on its own stream.
                let hydrate = stream == Some(StreamId(0));
                if let Some(bad) = commit.outputs.iter().find(|(h, _)| !hydrate && Some(h.stream()) != stream) {
                    let message = format!("an output is on stream {}, not the lease's", bad.0.stream().0);
                    return self.reply_error(session, request_id, ErrorCode::NotAllowed, &message);
                }
                // Coverage says "this stage has run over this source range", so its source must be one of the
                // lease's inputs (a hydrate lease's inputs mean nothing, so it's exempt).
                if let Some((source, _, _)) = commit.coverage
                    && let Some(info) = self.engine.lease_info(LeaseId(commit.lease))
                    && info.role != CapabilityRole::Hydrate
                    && !info.requires.iter().any(|r| r.stream == source)
                {
                    let message = format!("coverage cites stream {}, which isn't one of the lease's inputs", source.0);
                    return self.reply_error(session, request_id, ErrorCode::NotAllowed, &message);
                }
                // Out of the ownership map while the write is queued, so a second commit or an abort racing it is
                // refused rather than also answered as a success. A busy refusal puts it back: nothing was stored.
                let lease = LeaseId(commit.lease);
                let held = self.leases.remove(&lease);
                if let Err(e) = self.urgent.try_send(Urgent::Commit { session, request_id, commit }) {
                    if let Some(held) = held {
                        self.leases.insert(lease, held);
                    }
                    self.refuse_busy(session, request_id, &e);
                }
            }
            Command::AbortLease { session, request_id, lease, reason } => {
                if !self.owns(session, lease) {
                    return self.reply_error(session, request_id, ErrorCode::NotAllowed, "not this session's lease");
                }
                self.leases.remove(&LeaseId(lease));
                let live = self.live();
                let view = IndexView::new(&self.index, &live);
                match self.engine.abort_lease(&view, LeaseId(lease), &reason, now_ns()) {
                    Ok(actions) => {
                        let bytes = protocol::lease_released(&mut self.builder, request_id, lease, &[]);
                        self.push(session, bytes);
                        self.run_actions(actions);
                    }
                    Err(e) => self.reply_error(session, request_id, ErrorCode::NotFound, &e.to_string()),
                }
            }
            Command::LeaseReleased { session, lease } => {
                if !self.owns(session, lease) {
                    return debug!(lease, "ignoring a release of a lease this session doesn't hold");
                }
                self.leases.remove(&LeaseId(lease));
                match self.engine.release_lease(LeaseId(lease), now_ns()) {
                    Ok(actions) => self.run_actions(actions),
                    Err(e) => debug!(error = %e, "a released lease the scheduler didn't know"),
                }
            }
            Command::Telemetry { session, telemetry } => {
                if let Some(&worker) = self.workers.get(&session) {
                    self.engine.telemetry(worker, telemetry, now_ns());
                }
            }
            Command::Purge { session, request_id, ask, reason, detail } => {
                let id = Uuid(*uuid::Uuid::now_v7().as_bytes());
                let job = purge_job(
                    Some((session, request_id)),
                    Some((ask, reason, detail)),
                    id,
                    self.events.clone(),
                    &self.active_purges,
                    &self.writer_health,
                    self.writer_config,
                );
                if let Err(e) = self.background.try_send(job.into()) {
                    self.refuse_busy(session, request_id, &e);
                }
            }
            Command::Background { job, reply } => {
                let _ = reply.send(self.background.try_send(job.into()).map_err(|e| queue_error(&e)));
            }
            Command::CommitPayloadCopy { session, request_id, payload, location } => {
                self.commit_payload_copy(session, request_id, payload, location);
            }
            Command::CommitChunk { session, request_id, ask } => self.commit_chunk(session, request_id, ask),
            Command::ListRecordingInputs { session, request_id, ask } => {
                self.list_recording_inputs(session, request_id, *ask)
            }
            Command::ListAgedOut { session, request_id, stream, window, limit } => {
                self.list_aged_out(session, request_id, stream, window, limit);
            }
            Command::CommitSeqAt { session, request_id, at_ns } => self.commit_seq_at(session, request_id, at_ns),
            Command::CatalogStats { session, request_id } => self.catalog_stats(session, request_id),
            Command::GetChunk { session, request_id, chunk } => self.get_chunk(session, request_id, chunk),
            Command::ResolveFrames { session, request_id, ids } => self.resolve_frames(session, request_id, ids),
            Command::RewriteChunk { session, request_id, chunk, frames, location } => {
                self.rewrite_chunk(session, request_id, chunk, frames, location);
            }
            Command::Query { session, request_id, ask } => self.query(session, request_id, ask),
            Command::CommitPromise { session, request_id, promise, option, band } => {
                self.commit_promise(session, request_id, promise, option, band);
            }
            Command::CancelPromise { session, request_id, promise } => {
                self.cancel_promise(session, request_id, promise)
            }
            Command::Subscribe { session, request_id, ask } => self.subscribe(session, request_id, ask),
            Command::Unsubscribe { session, request_id, id } => self.unsubscribe(session, request_id, id),
            Command::AckEvents { session, id, position } => self.ack_events(session, id, position),
            Command::ListStreams { session, request_id } => self.list_streams(session, request_id),
            Command::ListPurges { session, request_id, after, limit } => {
                self.list_purges(session, request_id, after, limit);
            }
            Command::GetPurge { session, request_id, purge } => self.get_purge(session, request_id, purge),
            Command::PurgeAck { session, purge } => self.purge_acked(session, purge, now_ns()),
            Command::RememberForgotten { session, request_id, hash } => {
                self.remember_forgotten(session, request_id, hash);
            }
            Command::ListForgotten { session, request_id, after, limit } => {
                self.list_forgotten(session, request_id, after, limit)
            }
            Command::IsForgotten { session, request_id, hash } => self.is_forgotten(session, request_id, hash),
            Command::WorkerState { session, request_id, ask } => self.worker_state(session, request_id, ask),
            Command::ForgetKeeper { session, request_id, name } => {
                self.forget_keeper(session, request_id, name, now_ns());
            }
            Command::JobRejected { session, job, reason } => self.job_rejected(session, job, &reason),
            Command::Read { job, reply } => {
                let _ = reply.send(self.reads.try_send(job).map_err(|e| queue_error(&e)));
            }
            Command::Startup { costs, boot } => self.startup(&costs, boot),
            Command::CostModels { reply } => {
                let _ = reply.send(continuum_sched::Scheduler::cost_models(&self.engine));
            }
            Command::UpkeepNow { reply } => {
                self.upkeep_now();
                let _ = reply.send(());
            }
            Command::SyncWriter { reply } => {
                let urgent = &self.urgent;
                subscribe::queue_sync(&mut self.cursors, |job| urgent.try_send(job).is_ok(), reply);
            }
            Command::LiveStreams { reply } => {
                let _ = reply.send(self.producers.keys().copied().collect());
            }
            Command::Covered { stream, range, reply } => {
                let _ = reply.send(self.index.covered(stream, range));
            }
        }
    }

    /// Tells a session its request couldn't be queued.
    fn refuse_busy<T>(&mut self, session: SessionId, request_id: u64, e: &TrySendError<T>) {
        let message = match e {
            TrySendError::Full(_) => "the broker is busy; try again shortly",
            TrySendError::Disconnected(_) => "the broker's writer has stopped",
        };
        let bytes = protocol::error(&mut self.builder, request_id, ErrorCode::Unavailable, message);
        self.push(session, bytes);
    }

    /// Checks that the session produces the stream, then queues the payload for the writer.
    fn ingest(
        &mut self,
        session: SessionId,
        request_id: u64,
        header: HeaderBuf,
        locator: Option<StorageLocator>,
        dependencies: Vec<continuum_catalog::Dependency>,
    ) {
        let stream = header.stream();
        let id = header.id();
        if !self.sessions.get(&session).is_some_and(|s| s.streams.contains(&stream)) {
            let reason = format!("this session didn't register stream {}", stream.0);
            let bytes = protocol::ingest_rejected(
                &mut self.builder,
                request_id,
                stream.0,
                Some(id.0),
                &reason,
                ErrorCode::NotAllowed,
            );
            self.push(session, bytes);
            return;
        }
        if let Err(e) = self.urgent.try_send(Urgent::Ingest { session, request_id, header, locator, dependencies }) {
            let reason = match e {
                TrySendError::Full(_) => "the broker is busy; send it again shortly",
                TrySendError::Disconnected(_) => "the broker's writer has stopped",
            };
            let bytes = protocol::ingest_rejected(
                &mut self.builder,
                request_id,
                stream.0,
                Some(id.0),
                reason,
                ErrorCode::Unavailable,
            );
            self.push(session, bytes);
        }
    }

    /// A commit came back from the writer: update what lives in memory, then answer.
    fn event(&mut self, event: Event) {
        // A callback still arriving after the terminal request may enqueue follow-up work. Process it normally,
        // but never publish a projection from this shutdown: only an already settled core is eligible. The one
        // exception is a cursor save the fence queued ahead of itself: it changes no availability, and the final
        // check still requires every position to have landed.
        if self.checkpointing
            && !matches!(&event, Event::RestartCheckpoint { .. })
            && !matches!(&event, Event::CursorsWritten { failed, .. } if failed.is_empty())
        {
            self.refuse_checkpoint(CheckpointRefusal::CallbackDuringFence);
        }
        match event {
            Event::PurgeCompleted { id, result } => self.purge_completed(id, result),
            Event::PayloadCopyCommitted { session, request_id, payload, result } => {
                self.payload_copy_committed(session, request_id, payload, result);
            }
            Event::ChunkCommitted { session, request_id, chunk, copy, result } => {
                self.chunk_committed(session, request_id, chunk, copy, result);
            }
            Event::ChunkRewritten { session, request_id, chunk, added, result } => {
                self.chunk_rewritten(session, request_id, chunk, added, result);
            }
            Event::Aging { session, request_id, result } => self.aging_done(session, request_id, result),
            Event::QueryRead { reply, items } => self.query_read(reply, items),
            Event::Staged { dispatch, inputs } => self.staged(dispatch, inputs),
            Event::Fulfilled { consumer, promise, items, final_part, ordinal } => {
                self.fulfilled(consumer, promise, items, final_part, ordinal)
            }
            Event::StreamsRead { session, request_id, result } => self.streams_read(session, request_id, result),
            Event::PurgesRead { session, request_id, result } => self.purges_read(session, request_id, result),
            Event::ForgottenRemembered { session, request_id, result } => {
                self.forgotten_remembered(session, request_id, result);
            }
            Event::ForgottenListed { session, request_id, result } => {
                self.forgotten_listed(session, request_id, result)
            }
            Event::ForgottenRead { session, request_id, result } => self.forgotten_read(session, request_id, result),
            Event::WorkerStateDone { session, request_id, result } => {
                self.worker_state_done(session, request_id, result);
            }
            Event::PurgeRecordRead { session, request_id, purge, result } => {
                self.purge_record_read(session, request_id, purge, result);
            }
            Event::CursorRead { session, request_id, ask, position } => {
                self.cursor_read(session, request_id, ask, position);
            }
            Event::CaughtUpPage { session, id, token, pages } => self.caught_up_page(session, id, token, pages),
            Event::Registered { session, request_id, name, ring, result } => {
                let bytes = match result {
                    Ok((id, meta)) => {
                        // Registration already committed. Keep its metadata in the index even when a
                        // subsequent arena allocation fails, so a restart checkpoint cannot omit it.
                        self.index.register(id, meta);
                        self.index.name_stream(name.clone(), id);
                        // A ring stream gets its ring now that it has an id: shared memory only, no I/O.
                        let carved = ring.map(|a| self.tier0.ring_for(id, a.stride, meta.rate_hz, a.seconds));
                        let carved = match carved.transpose() {
                            Ok(c) => c,
                            Err(e) => {
                                let message = format!("stream {name}: {e}");
                                let bytes =
                                    protocol::error(&mut self.builder, request_id, ErrorCode::Unavailable, &message);
                                return self.push(session, bytes);
                            }
                        };
                        if let Some(s) = self.sessions.get_mut(&session)
                            && s.streams.insert(id)
                        {
                            *self.producers.entry(id).or_insert(0) += 1;
                        }
                        let grant = carved.map(|(index, ring)| protocol::RingGrant {
                            arena: self.tier0.summary(),
                            index,
                            ring,
                        });
                        protocol::stream_registered(&mut self.builder, request_id, id.0, grant)
                    }
                    Err(e) => {
                        let code = match &e {
                            CatalogError::Conflict(_) => ErrorCode::Conflict,
                            CatalogError::Invalid(_) => ErrorCode::BadRequest,
                            CatalogError::NotFound(_) => ErrorCode::NotFound,
                            _ => {
                                warn!(stream = %name, error = %e, "couldn't register a stream");
                                ErrorCode::Internal
                            }
                        };
                        protocol::error(&mut self.builder, request_id, code, &format!("stream {name}: {e}"))
                    }
                };
                self.push(session, bytes);
            }
            Event::Ingested(done) => {
                for d in done {
                    if d.hidden {
                        self.index.note_hidden_on_landing(d.id);
                    }
                    match d.result {
                        Ok(()) => {
                            if let Some((span, tier)) = d.copy {
                                if let Err(e) = self.index.add_copy(d.stream, span, tier) {
                                    warn!(
                                        stream = e.0.0,
                                        "the index doesn't know a stream the catalog just indexed into"
                                    );
                                }
                                if !d.hidden {
                                    self.track_slot(d.id, d.stream, span, d.locator.as_ref());
                                }
                                if let Some(StorageLocator::Ring { ring, first_seq, count, .. }) = &d.locator
                                    && !d.hidden
                                {
                                    let batch = RingBatch {
                                        id: d.id,
                                        purged: false,
                                        first_seq: *first_seq,
                                        count: *count,
                                        stream: d.stream,
                                        span,
                                    };
                                    self.ring_batches.entry(*ring).or_default().push_back(batch);
                                }
                                // Armed promises waiting for this range can run now.
                                let live = self.live();
                                let view = IndexView::new(&self.index, &live);
                                let ingest = IngestRef { id: d.id, stream: d.stream, range: span };
                                let actions = self.engine.on_ingest(&view, ingest, now_ns());
                                self.run_actions(actions);
                            }
                            if !d.hidden && !d.repeat {
                                self.publish(&d.header, d.locator.as_ref(), d.seq);
                            }
                            if d.request_id != 0 {
                                // Hidden on landing: never available, so no position, as LeaseReleased says it.
                                let seq = if d.hidden { 0 } else { d.seq };
                                let bytes = protocol::ingested(&mut self.builder, d.request_id, d.stream.0, d.id, seq);
                                self.push(d.session, bytes);
                            }
                        }
                        Err((code, reason)) => {
                            let bytes = protocol::ingest_rejected(
                                &mut self.builder,
                                d.request_id,
                                d.stream.0,
                                Some(d.id.0),
                                &reason,
                                code,
                            );
                            self.push(d.session, bytes);
                        }
                    }
                }
            }
            Event::SlotsToPurge(handles) => self.slots_to_purge(handles),
            Event::PurgeStep { id, delta } => {
                let mismatches = self.index.apply_purge_delta(&delta);
                if mismatches > 0 {
                    warn!(mismatches, "a purge step named copies the index didn't hold");
                }
                self.purge_slots(&delta.doomed);
                self.purge_ring_batches(&delta.doomed);
                self.purge_stepped(id, &delta);
            }
            Event::PurgeDone { requester, report, kind, acked } => {
                if let Some((session, request_id)) = requester {
                    let bytes = protocol::purge_result(&mut self.builder, request_id, &report);
                    self.push(session, bytes);
                }
                self.purge_finished(&report, kind.as_ref(), acked, now_ns());
            }
            Event::PurgeFailed { id, requester, code, message } => {
                self.purge_failed(id);
                // A client's mistake (a stream name that doesn't exist) isn't the broker's fault, so only a broker
                // failure is logged as one.
                if code == ErrorCode::Internal {
                    warn!(%message, "a purge failed");
                }
                if let Some((session, request_id)) = requester {
                    let bytes = protocol::error(&mut self.builder, request_id, code, &message);
                    self.push(session, bytes);
                }
            }
            Event::Committed {
                session,
                request_id,
                lease,
                outputs,
                copies,
                hidden,
                published,
                seqs,
                coverage,
                cost,
                result,
            } => {
                for (header, locator, seq) in &published {
                    self.publish(header, locator.as_ref(), *seq);
                    if let Some(span) = header.span() {
                        self.track_slot(header.id(), header.stream(), span, locator.as_ref());
                    }
                    // A ring batch committed as an output is swept like an ingested one once the ring laps it.
                    if let (Some(StorageLocator::Ring { ring, first_seq, count, .. }), Some(span)) =
                        (locator, header.span())
                    {
                        let batch = RingBatch {
                            id: header.id(),
                            purged: false,
                            first_seq: *first_seq,
                            count: *count,
                            stream: header.stream(),
                            span,
                        };
                        self.ring_batches.entry(*ring).or_default().push_back(batch);
                    }
                }
                // What's durable goes in the index first, whatever the result: outputs stay stored when the coverage
                // write after them fails.
                for id in hidden {
                    self.index.note_hidden_on_landing(id);
                }
                for (stream, span, tier) in copies {
                    if self.index.add_copy(stream, span, tier).is_err() {
                        warn!(stream = stream.0, "a committed output is on a stream the index doesn't know");
                    }
                }
                if let Err((code, message)) = result {
                    self.reply_error(session, request_id, code, &message);
                    self.leases.remove(&LeaseId(lease));
                    let live = self.live();
                    let view = IndexView::new(&self.index, &live);
                    if let Ok(actions) = self.engine.abort_lease(&view, LeaseId(lease), &message, now_ns()) {
                        self.run_actions(actions);
                    }
                    return;
                }
                let done = coverage.map(|(source, stage, span)| {
                    self.index.add_coverage(source, stage, span);
                    CoverageDone { source, stage, range: span }
                });
                let refs: Vec<OutputRef> =
                    outputs.iter().map(|&(id, stream, range)| OutputRef { id, stream, range }).collect();
                self.leases.remove(&LeaseId(lease));
                // The outputs and coverage are durable, so the commit succeeded, and the ack says so before anything
                // the commit sets off. A lease the scheduler already ended (it expired meanwhile) changes nothing for
                // the worker: its work is stored either way.
                let bytes = protocol::lease_released(&mut self.builder, request_id, lease, &seqs);
                self.push(session, bytes);
                match self.engine.commit_lease(LeaseId(lease), &refs, done, cost, now_ns()) {
                    Ok(actions) => self.run_actions(actions),
                    Err(e) => debug!(lease, error = %e, "a committed lease the scheduler had already ended"),
                }
            }
            Event::RestartCheckpoint { stamp, reply } => {
                let refusal =
                    self.checkpoint_refusal.or_else(|| stamp.err()).or_else(|| self.checkpoint_refusal(false)).or_else(
                        || {
                            (!self.urgent.is_empty() || !self.background.is_empty())
                                .then_some(CheckpointRefusal::WriterBusy)
                        },
                    );
                let value = match (refusal, stamp) {
                    (None, Ok(stamp)) => Ok((std::mem::take(&mut self.index), stamp)),
                    (Some(reason), _) | (None, Err(reason)) => Err(reason),
                };
                let _ = reply.send(value);
            }
            Event::Synced(reply) => {
                let _ = reply.send(());
            }
            Event::CostsNotSaved(bytes) => self.costs_not_saved(&bytes),
            Event::CursorsWritten { saved, failed } => self.cursors_written(saved, failed),
        }
    }

    /// Pins a Tier 0 locator for `session` to read (for [`crate::tier0::READ_PIN`]), held until the session ends or
    /// the pin expires. Other locators need no pin.
    ///
    /// # Errors
    /// The slot is stale or reused: the item can't be answered from Tier 0.
    pub(crate) fn pin_for_reader(&mut self, session: SessionId, locator: &StorageLocator) -> Result<(), String> {
        match locator {
            StorageLocator::Shm { .. } => {}
            // A ring batch can't be pinned: the writer never waits. It's handed out while the ring hasn't lapped it,
            // and the reader checks the records' sequence numbers as it copies them.
            StorageLocator::Ring { .. } => return self.tier0.check(locator),
            _ => return Ok(()),
        }
        let pin = self.tier0.pin(locator, crate::tier0::READ_PIN)?;
        self.reader_pins.entry(session).or_default().push(pin);
        Ok(())
    }

    /// For a time query on `stream` reaching past its ring's newest indexed batch, the ring's records written since
    /// (or, with no batch indexed, all it still holds): the live edge, not indexed yet but readable now.
    pub(super) fn ring_tail(&self, stream: StreamId, range: Interval) -> Option<query::RingTailSeqs> {
        let (index, ring) = self.tier0.ring_of(stream)?;
        let (head, capacity) = self.tier0.ring_head(index)?;
        let batches = self.ring_batches.get(&index);
        let sealed = batches.and_then(|b| b.iter().map(|b| (b.first_seq + u64::from(b.count), b.span.end_ns())).max());
        if let Some((_, end_ns)) = sealed
            && range.end_ns() <= end_ns
        {
            return None; // the query stops inside what's indexed
        }
        let first_seq = sealed.map_or(0, |(seq, _)| seq).max(head.saturating_sub(u64::from(capacity)));
        // Lazily: on an empty ring, head - 1 would underflow.
        (first_seq < head).then(|| query::RingTailSeqs { index, ring, first_seq, last_seq: head - 1 })
    }

    /// A purge took these payloads' Tier 0 copies off the index: their ring batches are marked, or a later sweep would
    /// remove the copy again, taking an overlapping surviving copy's count and logging a false gap. They stay on the
    /// list, which is also where `ring_tail` finds how far the ring is sealed.
    fn purge_ring_batches(&mut self, doomed: &[continuum_catalog::Doomed]) {
        let gone: std::collections::HashSet<Uuid> =
            doomed.iter().filter(|d| d.tiers.contains(&StorageTier::Tier0Hot)).map(|d| d.id).collect();
        if gone.is_empty() {
            return;
        }
        for b in self.ring_batches.values_mut().flat_map(|b| b.iter_mut()) {
            if gone.contains(&b.id) {
                b.purged = true;
            }
        }
    }

    /// Drops from the index every ring batch its ring has lapped (its oldest record overwritten): shared memory reads
    /// and index updates only. A lapped batch with no other copy over its span is a gap in the archive, logged.
    fn sweep_rings(&mut self) {
        for (&ring, batches) in &mut self.ring_batches {
            let Some((head, capacity)) = self.tier0.ring_head(ring) else { continue };
            let oldest = head.saturating_sub(u64::from(capacity));
            while batches.front().is_some_and(|b| b.first_seq < oldest) {
                let Some(b) = batches.pop_front() else { break };
                let owed = self.slots.is_archived(b.stream);
                remove_lapped_batch(&mut self.index, &b, owed, &mut self.archive_health);
            }
        }
        self.ring_batches.retain(|_, b| !b.is_empty());
    }

    fn close(&mut self, session: SessionId) {
        // Its pins go with it, so a client that disconnects or dies never holds a slot.
        self.reader_pins.remove(&session);
        let Some(gone) = self.sessions.remove(&session) else { return };
        // Its process's ring claims too (unless the process has another session open), so the stream's next writer
        // can take its ring even if this one died holding it.
        if !self.sessions.values().any(|s| s.pid == gone.pid) {
            self.tier0.release_rings_of_pid(gone.pid);
        }
        if let Some(worker) = self.workers.remove(&session) {
            self.worker_sessions.remove(&worker);
            self.leases.retain(|_, &mut (w, _)| w != worker);
            let live = self.live();
            let view = IndexView::new(&self.index, &live);
            let actions = self.engine.worker_gone(&view, worker, now_ns());
            self.run_actions(actions);
        }
        self.consumer_gone(session);
        self.unsubscribe_all(session);
        self.keeper_gone(session);
        for stream in gone.streams {
            if let Some(n) = self.producers.get_mut(&stream) {
                *n -= 1;
                if *n == 0 {
                    self.producers.remove(&stream);
                }
            }
        }
    }

    /// Whether `lease` is live and held by the worker this session registered.
    fn owns(&self, session: SessionId, lease: u64) -> bool {
        let worker = self.workers.get(&session);
        worker.is_some() && self.leases.get(&LeaseId(lease)).map(|&(w, _)| w) == worker.copied()
    }

    /// The live streams for the scheduler's view (no live edges yet: rings come with continuum-shm).
    fn live(&self) -> HashMap<StreamId, Option<u64>> {
        self.producers.keys().map(|&s| (s, None)).collect()
    }

    fn reply_error(&mut self, session: SessionId, request_id: u64, code: ErrorCode, message: &str) {
        let bytes = protocol::error(&mut self.builder, request_id, code, message);
        self.push(session, bytes);
    }

    /// Indexes a worker's capabilities. A capability naming a stream nobody has registered is refused (the worker
    /// registers its output streams first).
    fn register_worker(&mut self, session: SessionId, request_id: u64, ask: protocol::WorkerAsk) {
        if self.workers.contains_key(&session) {
            return self.reply_error(
                session,
                request_id,
                ErrorCode::BadRequest,
                "this session already registered a worker",
            );
        }
        // Acknowledgements of purge notices are owed by name, so two connected keepers can't share one.
        if ask.keeps_copies && self.keeper_named(&ask.producer) {
            let msg = format!("a worker named {} that keeps copies is connected already", ask.producer);
            return self.reply_error(session, request_id, ErrorCode::Conflict, &msg);
        }
        let mut accepted_index = Vec::new();
        let mut refused = Vec::new();
        let mut capabilities = Vec::new();
        for c in ask.capabilities {
            let produces = if c.role == CapabilityRole::Hydrate {
                Some(StreamId(0))
            } else {
                self.index.stream_named(&c.produces)
            };
            let Some(produces) = produces else {
                refused.push(format!("capability {}: no stream named {}", c.index, c.produces));
                continue;
            };
            let mut requires = Vec::new();
            let mut missing = None;
            for (name, before_ns, after_ns) in &c.requires {
                match self.index.stream_named(name) {
                    Some(stream) => requires.push(Requirement { stream, before_ns: *before_ns, after_ns: *after_ns }),
                    None => missing = Some(name.clone()),
                }
            }
            if let Some(name) = missing {
                refused.push(format!("capability {}: no stream named {name}", c.index));
                continue;
            }
            accepted_index.push(c.index);
            capabilities.push(Capability {
                role: c.role,
                produces,
                requires,
                domain: c.domain,
                ram_mb: c.ram_mb,
                vram_mb: c.vram_mb,
                est_us_per_item: c.est_us_per_item,
                cold_start_ms: c.cold_start_ms,
                max_batch: c.max_batch,
                output_bytes: c.output_bytes,
                outputs_per_job: c.outputs_per_job,
                preemptible: c.preemptible,
                expected_confidence: c.expected_confidence,
            });
        }
        self.next_worker += 1;
        let worker = WorkerId(self.next_worker);
        let manifest = Manifest {
            producer: ask.producer,
            version_hash: ask.version_hash,
            device_index: ask.device_index,
            capabilities,
        };
        self.worker_names.insert(worker, manifest.producer.clone());
        let names: Vec<String> = manifest.capabilities.iter().map(|c| query::capability_name(&self.index, c)).collect();
        let accepted = self.engine.register_worker(worker, manifest, now_ns());
        for (a, name) in accepted.iter().zip(names) {
            self.cap_names.insert(a.capability, name);
        }
        self.workers.insert(session, worker);
        self.worker_sessions.insert(worker, session);
        let rows: Vec<(u32, u32, bool)> =
            accepted_index.iter().zip(&accepted).map(|(&i, a)| (i, a.capability.0, a.runnable)).collect();
        let bytes = protocol::worker_registered(&mut self.builder, request_id, &rows, &refused);
        self.push(session, bytes);
        // After the reply, so the notices for purges still waiting follow it (and not when the reply closed the
        // session, which has gone from `sessions` by then).
        if ask.keeps_copies && self.sessions.contains_key(&session) {
            let name = self.worker_names[&worker].clone();
            self.keeper_registered(session, name, now_ns());
        }
    }

    /// Carries out what the scheduler decided.
    fn run_actions(&mut self, actions: Vec<Action>) {
        for action in actions {
            match action {
                Action::Preempt { worker, lease, by_band, reason } => {
                    if let Some(&session) = self.worker_sessions.get(&worker) {
                        let bytes = protocol::lease_preempted(&mut self.builder, lease.0, &reason, by_band);
                        self.push(session, bytes);
                    }
                }
                Action::RevokeLease { worker, lease, reason } => {
                    self.leases.remove(&lease);
                    // Revoked without a higher band asking (expired, say): the worker hears it as a preemption whose
                    // reason says why, so it stops either way.
                    if let Some(session) = worker.and_then(|w| self.worker_sessions.get(&w).copied()) {
                        let message = format!("revoked: {reason}");
                        let bytes =
                            protocol::lease_preempted(&mut self.builder, lease.0, &message, PriorityBand::P0Critical);
                        self.push(session, bytes);
                    }
                }
                Action::Unpin { lease } => self.unpin(lease),
                // Every exit of a lease, so the ownership map never keeps one the scheduler has dropped (a preempted
                // lease that expired sends no RevokeLease, for instance).
                Action::LeaseEnded { lease, .. } => {
                    self.leases.remove(&lease);
                    self.jobs.retain(|_, &mut (_, l)| l != lease);
                }
                // Queries and promises are wired in the next part: until then the engine has no promises,
                // so these can't come up.
                Action::Dispatch(d) => self.dispatch(d),
                Action::Fulfill { promise, consumer, outputs, final_part } => {
                    self.fulfill(promise, consumer, outputs, final_part);
                }
                Action::FailPromise { promise, consumer, reason } => self.promise_failed(promise, consumer, &reason),
            }
        }
    }

    /// Queues `bytes` for a session's writer. A session whose queue is full isn't reading: it is dropped here, and
    /// its connection is told to close (its reader still holds a sender, so dropping ours alone wouldn't end it).
    fn push(&mut self, session: SessionId, bytes: Vec<u8>) {
        let Some(s) = self.sessions.get(&session) else { return };
        match s.outbound.try_send(bytes) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(_)) => {
                warn!(
                    %session,
                    pid = s.pid,
                    "closing a session whose outbound queue is full: the client isn't reading what it was sent"
                );
                s.kicked.notify_one();
                self.close(session);
            }
            Err(mpsc::error::TrySendError::Closed(_)) => self.close(session),
        }
    }
}

/// The writer thread: one bounded urgent batch, then one step of a long job, with progress for both queues.
#[expect(clippy::too_many_arguments, reason = "the writer owns its queues, catalog, and shared diagnostics")]
fn writer(
    catalog: &dyn Catalog,
    urgent: &Receiver<Urgent>,
    background: &Receiver<LongJob>,
    events: &mpsc::UnboundedSender<Event>,
    config: &WriterConfig,
    active_purges: &AtomicUsize,
    resumed: Vec<BackgroundJob>,
    health: &writer_health::WriterHealth,
) {
    // Each waiting long job with the budget for its next step, and whether an overrun at a budget of 1 has been
    // logged for it (a job whose single row overruns would otherwise log on every step).
    let mut jobs: VecDeque<(LongJob, usize, bool)> =
        resumed.into_iter().map(|job| (LongJob::from(job), config.initial_budget, false)).collect();
    let (mut urgent_open, mut background_open) = (true, true);
    // An urgent job the writer woke up for: it joins the next tick with everything that arrived behind it, rather
    // than being committed alone.
    let mut woke_with: Vec<Urgent> = Vec::new();
    loop {
        health.set_waiting(jobs.len());
        health.set_running(false);
        // Everything urgent that is waiting now.
        let mut batch = std::mem::take(&mut woke_with);
        while urgent_open && batch.len() < TICK {
            match urgent.try_recv() {
                Ok(job) => batch.push(job),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => urgent_open = false,
            }
        }
        if let Some(position) = batch.iter().position(|job| matches!(job, Urgent::RestartCheckpoint { .. })) {
            let trailing = batch.split_off(position + 1);
            let Some(Urgent::RestartCheckpoint { reply }) = batch.pop() else { unreachable!() };
            run_urgent(catalog, batch, events, active_purges);
            // Restartable upkeep is dropped, not waited for; dropping the sweep clears its running flag before the
            // scheduler hears the answer. Anything else still queued keeps this stop from publishing.
            while let Ok(job) = background.try_recv() {
                jobs.push_back((job, config.initial_budget, false));
            }
            jobs.retain(|(job, _, _)| !job.drop_at_stop);
            let clean = trailing.is_empty()
                && jobs.is_empty()
                && urgent.is_empty()
                && active_purges.load(Ordering::SeqCst) == 0;
            run_urgent(catalog, trailing, events, active_purges);
            let stamp = if clean {
                match (catalog.pending_purges(), catalog.restart_stamp()) {
                    (Ok(purges), _) if !purges.is_empty() => Err(CheckpointRefusal::PendingPurges),
                    (Ok(_), Ok(Some(stamp))) => Ok(stamp),
                    (Ok(_), Ok(None)) => Err(CheckpointRefusal::NoStamp),
                    _ => Err(CheckpointRefusal::CatalogUnreadable),
                }
            } else {
                Err(CheckpointRefusal::WriterBusy)
            };
            let _ = events.send(Event::RestartCheckpoint { stamp, reply });
            return;
        }
        if !batch.is_empty() {
            run_urgent(catalog, batch, events, active_purges);
        }
        // Take waiting long jobs only up to the queue's size, so the backlog stays bounded and a full queue is
        // refused as busy upstream.
        while background_open && jobs.len() < BACKGROUND {
            match background.try_recv() {
                Ok(job) => jobs.push_back((job, config.initial_budget, false)),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => background_open = false,
            }
        }
        // One step of the oldest long job, then back to urgent work.
        if let Some((mut job, budget, mut warned_at_floor)) = jobs.pop_front() {
            health.set_waiting(jobs.len());
            health.set_running(true);
            let started = Instant::now();
            // A job that panics loses only itself; the writer carries on with the rest.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (job.job)(catalog, budget)));
            health.set_running(false);
            let Ok(more) = result else {
                tracing::error!("a background catalog job panicked and was dropped");
                continue;
            };
            let took = started.elapsed();
            // A step a little over its target is the budget being tuned (next_budget shrinks it), not trouble: that's
            // logged at debug. One far over it held urgent writes up, and is worth a warning.
            if took > config.step_target * STALL_FACTOR && (budget > 1 || !warned_at_floor) {
                warn!(?took, target = ?config.step_target, budget, "a background step held the writer up");
                warned_at_floor |= budget <= 1;
            } else if took > config.step_target {
                tracing::debug!(?took, target = ?config.step_target, budget, "a background step overran its target");
            }
            if more {
                jobs.push_back((job, next_budget(budget, took, config), warned_at_floor));
            }
            health.set_waiting(jobs.len());
            continue;
        }
        // Nothing to do: wait for either queue. A durable catalog leaves ordinary commits unsynced for up to
        // `DURABLE_IDLE`; a wait that long with nothing arriving makes them durable.
        let idle = match (urgent_open, background_open) {
            (false, false) => return,
            (true, true) => crossbeam_channel::select_biased! {
                recv(urgent) -> job => {
                    match job {
                        Ok(job) => woke_with.push(job),
                        Err(_) => urgent_open = false,
                    }
                    false
                },
                recv(background) -> job => {
                    match job {
                        Ok(job) => jobs.push_back((job, config.initial_budget, false)),
                        Err(_) => background_open = false,
                    }
                    false
                },
                default(DURABLE_IDLE) => true,
            },
            (true, false) => match urgent.recv_timeout(DURABLE_IDLE) {
                Ok(job) => {
                    woke_with.push(job);
                    false
                }
                Err(RecvTimeoutError::Timeout) => true,
                Err(RecvTimeoutError::Disconnected) => {
                    urgent_open = false;
                    false
                }
            },
            (false, true) => match background.recv_timeout(DURABLE_IDLE) {
                Ok(job) => {
                    jobs.push_back((job, config.initial_budget, false));
                    false
                }
                Err(RecvTimeoutError::Timeout) => true,
                Err(RecvTimeoutError::Disconnected) => {
                    background_open = false;
                    false
                }
            },
        };
        if idle && let Err(error) = catalog.flush_durable() {
            warn!(%error, "couldn't make the catalog's latest commits durable; the next commit tries again");
        }
    }
}

/// Runs one tick's urgent jobs. Every ingest in the tick goes into one catalog insert (one fsync); registrations run
/// as they come (an ingest never depends on a registration in the same tick, since the scheduler task forwards a
/// stream's ingest only after its registration has committed), and a sync commits the ingests ahead of it first.
fn run_urgent(
    catalog: &dyn Catalog,
    jobs: Vec<Urgent>,
    events: &mpsc::UnboundedSender<Event>,
    active_purges: &AtomicUsize,
) {
    let mut ingests: Vec<Queued> = Vec::new();
    // Cursors are written once per tick, the last position per name: a subscriber acking every event would otherwise
    // cost a commit each. The scheduler queues at most one per name (`subscribe::CursorAcks`) and hears back here.
    let mut cursors: Vec<(String, u64)> = Vec::new();
    for job in jobs {
        match job {
            Urgent::RestartCheckpoint { .. } => {
                unreachable!("checkpoint barriers are handled by the outer writer loop")
            }
            Urgent::Ingest { session, request_id, header, locator, dependencies } => {
                ingests.push((session, request_id, header, locator, dependencies));
            }
            Urgent::Register { session, request_id, info, ring } => {
                let name = info.name.clone();
                let meta = StreamMeta {
                    kind: match info.kind {
                        CatalogKind::Sensor => StreamKind::Sensor,
                        CatalogKind::Derived => StreamKind::Derived,
                    },
                    rate_hz: info.rate_hz.unwrap_or(0.0),
                };
                let result = catalog.register_stream(info).map(|id| (id, meta));
                let _ = events.send(Event::Registered { session, request_id, name, ring, result });
            }
            Urgent::Commit { session, request_id, commit } => {
                commit_ingests(catalog, std::mem::take(&mut ingests), events, active_purges);
                commit_outputs(catalog, session, request_id, commit, events);
            }
            Urgent::CommitPayloadCopy { session, request_id, payload, location, keeper } => {
                commit_ingests(catalog, std::mem::take(&mut ingests), events, active_purges);
                payload_copies::commit_copy_job(catalog, session, request_id, payload, location, &keeper, events);
            }
            Urgent::CommitChunk { session, request_id, ask } => {
                // The chunk names payloads that may still be in the pending ingest batch: commit those first.
                commit_ingests(catalog, std::mem::take(&mut ingests), events, active_purges);
                chunks::commit_chunk_job(catalog, session, request_id, ask, events);
            }
            Urgent::RewriteChunk { session, request_id, chunk, frames, location } => {
                chunks::rewrite_chunk_job(catalog, (session, request_id, chunk, frames, location), events);
            }
            Urgent::Sync(reply) => {
                // Everything queued before the sync is committed before it is answered, positions included.
                commit_ingests(catalog, std::mem::take(&mut ingests), events, active_purges);
                write_cursors(catalog, std::mem::take(&mut cursors), events);
                let _ = events.send(Event::Synced(reply));
            }
            Urgent::Cursor { name, position } => match cursors.iter_mut().find(|(n, _)| *n == name) {
                Some(c) => c.1 = c.1.max(position),
                None => cursors.push((name, position)),
            },
            Urgent::SaveCosts(bytes) => {
                if let Err(e) = catalog.save_cost_models(&bytes) {
                    warn!(error = %e, "couldn't save the cost models; the next save tries again");
                    let _ = events.send(Event::CostsNotSaved(bytes));
                }
            }
            Urgent::Clock(boot, point) => {
                if let Err(e) = catalog.record_clock(boot, point) {
                    warn!(error = %e, "couldn't record a clock sample");
                }
                // The commit clock: everything committed so far was committed by now (docs/CONTINUUM.md, "Aging").
                // Timed here, on the writer, after every commit the head includes: the sample was taken before it
                // was queued, possibly before work queued ahead of it committed.
                if let Err(e) = catalog.mark_commit_clock(now_ns()) {
                    warn!(error = %e, "couldn't mark the commit clock");
                }
            }
            Urgent::CompletePurge { id, now_ns } => {
                let result = catalog.complete_purge(id, now_ns);
                let _ = events.send(Event::PurgeCompleted { id, result });
            }
            Urgent::AddKeeper(name) => {
                if let Err(e) = catalog.add_keeper(&name) {
                    warn!(error = %e, name, "couldn't record a worker that keeps copies");
                }
            }
            Urgent::ForgetKeeper(name) => {
                if let Err(e) = catalog.forget_keeper(&name) {
                    warn!(error = %e, name, "couldn't drop a keeper's name");
                }
            }
            Urgent::PurgeAcked { id, name } => {
                if let Err(e) = catalog.purge_acked(id, &name) {
                    warn!(error = %e, purge = %id, name, "couldn't record a keeper's acknowledgement");
                }
            }
            Urgent::RememberForgotten { session, request_id, hash } => {
                let result = catalog.forget(hash).map_err(|e| (error_code(&e), e.to_string()));
                let _ = events.send(Event::ForgottenRemembered { session, request_id, result });
            }
            Urgent::WorkerState { session, request_id, ask } => {
                let result = worker_state::write(catalog, ask);
                let _ = events.send(Event::WorkerStateDone { session, request_id, result });
            }
        }
    }
    write_cursors(catalog, cursors, events);
    commit_ingests(catalog, ingests, events, active_purges);
}

/// Writes durable subscription positions, and tells the scheduler which landed.
fn write_cursors(catalog: &dyn Catalog, cursors: Vec<(String, u64)>, events: &mpsc::UnboundedSender<Event>) {
    if cursors.is_empty() {
        return;
    }
    let (mut saved, mut failed) = (Vec::new(), Vec::new());
    for (name, position) in cursors {
        match catalog.set_cursor(&name, &position.to_le_bytes()) {
            Ok(()) => saved.push((name, position)),
            Err(e) => {
                warn!(error = %e, name, "couldn't save a durable subscription's position; the next tick tries again");
                failed.push(name);
            }
        }
    }
    let _ = events.send(Event::CursorsWritten { saved, failed });
}

/// Commits ingested payloads in one transaction. If the batch fails, it is split in halves and each retried, so a
/// bad notice gets its own answer without taking its neighbors down, in about log2(batch) extra transactions per
/// bad notice rather than one transaction per row.
/// An ingest waiting for the writer: its session, request id, header, locator, and dependencies.
type Queued = (SessionId, u64, HeaderBuf, Option<StorageLocator>, Vec<continuum_catalog::Dependency>);

fn commit_ingests(
    catalog: &dyn Catalog,
    ingests: Vec<Queued>,
    events: &mpsc::UnboundedSender<Event>,
    active_purges: &AtomicUsize,
) {
    if ingests.is_empty() {
        return;
    }
    let describe = |(session, request_id, header, locator, _): &Queued| {
        let copy = header.span().zip(locator.as_ref().map(StorageLocator::tier));
        (*session, *request_id, header.stream(), header.id(), copy)
    };
    let meta: Vec<_> = ingests.iter().map(describe).collect();
    let kept: Vec<(HeaderBuf, Option<StorageLocator>)> =
        ingests.iter().map(|(_, _, header, locator, _)| (header.clone(), locator.clone())).collect();
    let batch: Vec<(NewPayload, Vec<continuum_catalog::Dependency>)> = ingests
        .into_iter()
        .map(|(_, _, header, location, deps)| (NewPayload { header, location, coverage: None }, deps))
        .collect();
    let results = insert_bisecting_with(catalog, batch);
    // While a purge runs, the catalog hides a payload whose parent it is forgetting as the payload lands (its doomed
    // entry comes in a later step's delta). Such a payload must not enter the index now, or it would be quoted as
    // available until that step. Insert doesn't say which it hid, so while a purge is active, ask the catalog.
    let purging = active_purges.load(Ordering::SeqCst) > 0;
    let meta: Vec<_> = meta
        .into_iter()
        .zip(&results)
        .map(|((session, request_id, stream, id, copy), result)| {
            let hidden = purging && result.is_ok() && matches!(catalog.get(id), Ok(None));
            // A repeat's copy was counted, and published, when it was stored.
            let repeat = matches!(result, Ok((_, true)));
            (session, request_id, stream, id, if hidden || repeat { None } else { copy }, hidden, repeat)
        })
        .collect();
    let done = meta
        .into_iter()
        .zip(results)
        .zip(kept)
        .map(|(((session, request_id, stream, id, copy, hidden, repeat), result), (header, locator))| Ingested {
            session,
            request_id,
            stream,
            id,
            copy,
            hidden,
            repeat,
            header,
            locator,
            seq: result.as_ref().map_or(0, |(seq, _)| *seq),
            result: result.map(|_| ()),
        })
        .collect();
    // In chunks, so the scheduler task applies a large tick's results a few hundred at a time, checking for a voice
    // turn in between.
    let mut done: Vec<Ingested> = done;
    while !done.is_empty() {
        let rest = done.split_off(done.len().min(EVENT_CHUNK));
        let _ = events.send(Event::Ingested(std::mem::replace(&mut done, rest)));
    }
}

/// Commits a lease's outputs in one transaction (all or none: a job's outputs belong together), then the coverage
/// it finished, and tells the scheduler task either way.
fn commit_outputs(
    catalog: &dyn Catalog,
    session: SessionId,
    request_id: u64,
    commit: protocol::Commit,
    events: &mpsc::UnboundedSender<Event>,
) {
    let protocol::Commit { lease, outputs, coverage, cost } = commit;
    let meta: Vec<_> =
        outputs.iter().map(|(h, l)| (h.id(), h.stream(), h.span(), l.as_ref().map(StorageLocator::tier))).collect();
    let kept: Vec<(HeaderBuf, Option<StorageLocator>)> = outputs.clone();
    let batch: Vec<NewPayload> =
        outputs.into_iter().map(|(header, location)| NewPayload { header, location, coverage: None }).collect();
    let (mut described, mut copies, mut hidden, mut published) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut seqs = Vec::new();
    let inserted = if batch.is_empty() { Ok(Vec::new()) } else { catalog.insert(batch) };
    let mut result = inserted.map(|rows| {
        for (((id, stream, span, tier), row), (header, locator)) in meta.into_iter().zip(rows).zip(kept) {
            seqs.push(if row.hidden { 0 } else { row.seq });
            if row.hidden {
                // Doomed by a running purge as it landed: never available, and no output of the job.
                hidden.push(id);
                continue;
            }
            if !row.duplicate {
                published.push((header, locator, row.seq));
            }
            let Some(span) = span else { continue };
            described.push((id, stream, span));
            // A repeat (identical bytes indexed already) is an output, but its copy was counted when it was stored.
            if !row.duplicate
                && let Some(tier) = tier
            {
                copies.push((stream, span, tier));
            }
        }
    });
    if result.is_ok()
        && let Some((source, stage, span)) = coverage
    {
        result = catalog.add_coverage(continuum_catalog::CoverageMark { source, stage, span });
    }
    let result = result.map_err(|e| (error_code(&e), format!("committing lease {lease}: {e}")));
    let _ = events.send(Event::Committed {
        session,
        request_id,
        lease,
        outputs: described,
        copies,
        hidden,
        published,
        seqs,
        coverage,
        cost,
        result,
    });
}

/// Inserts `batch` in one transaction, or, when a row is at fault, each half on its own, recursively, down to the
/// bad payloads. One result per payload, in order.
///
/// Only a row's own error (a conflict, an invalid header, an unknown stream) is worth bisecting. A storage error
/// (a full disk, an I/O failure) isn't about any row: the whole batch fails with it at once, logged once, rather than
/// costing a failing transaction and a warning per row while everyone else's ingests wait.
#[cfg(test)]
fn insert_bisecting(catalog: &dyn Catalog, batch: Vec<NewPayload>) -> Vec<Result<(u64, bool), (ErrorCode, String)>> {
    insert_bisecting_with(catalog, batch.into_iter().map(|p| (p, Vec::new())).collect())
}

/// [`insert_bisecting`] for payloads with their dependencies, which go with them into each half.
fn insert_bisecting_with(
    catalog: &dyn Catalog,
    batch: Vec<(NewPayload, Vec<continuum_catalog::Dependency>)>,
) -> Vec<Result<(u64, bool), (ErrorCode, String)>> {
    let n = batch.len();
    // `insert` takes the batch by value, so a copy is kept only for the halves a failure needs.
    let retry = if n > 1 { Some(batch.clone()) } else { None };
    let dependent = batch.iter().any(|(_, d)| !d.is_empty());
    let (payloads, deps): (Vec<NewPayload>, Vec<_>) = batch
        .into_iter()
        .map(|(p, d)| {
            let id = p.header.id();
            (p, (id, d))
        })
        .unzip();
    let done = if dependent { catalog.insert_dependent(payloads, deps) } else { catalog.insert(payloads) };
    match done {
        // The catalog takes a repeat (identical header bytes) as idempotent, and so does the broker: the existing
        // sequence comes back. A different header for the same id is the catalog's Conflict.
        Ok(done) => done.iter().map(|i| Ok((i.seq, i.duplicate))).collect(),
        Err(e @ CatalogError::Storage(_)) => {
            warn!(error = %e, rows = n, "couldn't index a batch of ingested payloads");
            vec![Err((ErrorCode::Internal, format!("the catalog failed: {e}"))); n]
        }
        Err(e) => match retry {
            None => vec![Err((error_code(&e), e.to_string()))],
            Some(mut all) => {
                let back = all.split_off(n / 2);
                let mut out = insert_bisecting_with(catalog, all);
                out.extend(insert_bisecting_with(catalog, back));
                out
            }
        },
    }
}

/// How much of a purge reason, and of its stream list, is written on one log line.
const SCOPE_REASON_CHARS: usize = 80;
const SCOPE_STREAMS_CHARS: usize = 120;

/// The scope fields on a purge start or completion line. `ids` is 0 unless the purge was by id.
struct PurgeLogScope {
    kind: &'static str,
    streams: Vec<String>,
    reason: String,
    ids: u64,
}

impl PurgeLogScope {
    fn missing() -> Self {
        Self { kind: "", streams: Vec::new(), reason: String::new(), ids: 0 }
    }

    /// The stream names as one field, cut so a wide range cannot blow the line.
    fn streams_field(&self) -> String {
        one_line(&self.streams.join(","), SCOPE_STREAMS_CHARS)
    }
}

/// One log field: no line breaks, and cut to `max` characters (an ellipsis when it was longer).
fn one_line(text: &str, max: usize) -> String {
    let flat: String = text.chars().map(|c| if c == '\n' || c == '\r' { ' ' } else { c }).collect();
    if flat.chars().count() <= max {
        return flat;
    }
    let mut out: String = flat.chars().take(max.saturating_sub(3)).collect();
    out.push_str("...");
    out
}

fn stream_names(catalog: &dyn Catalog, ids: &[StreamId]) -> Vec<String> {
    ids.iter()
        .map(|id| match catalog.stream(*id) {
            Ok(Some(info)) => info.name,
            _ => format!("stream {}", id.0),
        })
        .collect()
}

/// Kind, stream names, and a short reason from the catalog record. A range that names no streams is `"all"`.
fn purge_scope(catalog: &dyn Catalog, record: &PurgeRecord) -> PurgeLogScope {
    let (kind, streams, ids) = match &record.kind {
        PurgeKind::Range { streams: None, .. } => ("range", vec!["all".to_owned()], 0),
        PurgeKind::Range { streams: Some(ids), .. } => ("range", stream_names(catalog, ids), 0),
        PurgeKind::Entity => ("entity", Vec::new(), 0),
        PurgeKind::Ids { count } => ("ids", Vec::new(), u64::try_from(*count).unwrap_or(u64::MAX)),
        PurgeKind::AgeOut { count } => ("age-out", Vec::new(), u64::try_from(*count).unwrap_or(u64::MAX)),
    };
    PurgeLogScope { kind, streams, reason: one_line(&record.reason, SCOPE_REASON_CHARS), ids }
}

fn log_purge_started(catalog: &dyn Catalog, health: &writer_health::WriterHealth, id: Uuid) {
    let Some(record) = catalog.purge_record(id).ok().flatten() else { return };
    let scope = purge_scope(catalog, &record);
    health.note_scope(id, scope.streams.clone());
    let streams = scope.streams_field();
    tracing::info!(
        purge = %id,
        scope_kind = scope.kind,
        scope_streams = %streams,
        scope_reason = %scope.reason,
        scope_ids = scope.ids,
        "catalog purge started"
    );
}

/// A purge as a writer job. The first call starts it (resolving stream names), unless `start` is `None` because it
/// is a purge a previous run left pending; every later call fills one cooperative catalog work window, bounded by
/// the configured operation ceiling. Its committed delta goes to the scheduler task, which drops what it doomed
/// from the index before it handles anything else. Commit/fsync latency is outside the work window.
///
/// Once its rows are gone the purge is finished but not complete: the scheduler task sends `PurgeNotice` to the
/// workers that keep their own copies and completes it on their acknowledgements (`core::forget`).
fn purge_job(
    requester: Option<(SessionId, u64)>,
    start: Option<(protocol::PurgeAsk, String, String)>,
    id: Uuid,
    events: mpsc::UnboundedSender<Event>,
    active: &Arc<AtomicUsize>,
    health: &Arc<writer_health::WriterHealth>,
    config: WriterConfig,
) -> BackgroundJob {
    let queued = Instant::now();
    let resumed = start.is_none();
    let tracked = health.track(id, resumed, queued);
    let health = Arc::clone(health);
    let mut needs_age = resumed;
    let mut steps = 0u64;
    let mut catalog_work = Duration::ZERO;
    let mut start = start;
    let guard = ActivePurge::new(active);
    let mut retries = 0u32;
    Box::new(move |catalog: &dyn Catalog, _budget: usize| {
        let _running = &guard;
        let _tracked = &tracked;
        let fail = |code: ErrorCode, message: String| {
            let _ = events.send(Event::PurgeFailed { id, requester, code, message });
            false
        };
        if needs_age {
            needs_age = false;
            let began = Instant::now();
            match catalog.purge_record(id) {
                Ok(Some(record)) => {
                    if record.report.is_some() {
                        // Keeper acknowledgement/grace is not unfinished catalog work.
                        health.finish(id);
                    } else {
                        health.mark_persisted(id, record.at_ns, now_ns(), Instant::now());
                        health.note_scope(id, purge_scope(catalog, &record).streams);
                    }
                }
                Ok(None) => {}
                Err(error) => warn!(purge = %id, %error, "could not load a resumed purge's start time"),
            }
            catalog_work += began.elapsed();
        }
        if let Some((ask, reason, detail)) = start.take() {
            let began = Instant::now();
            let selector = match ask {
                protocol::PurgeAsk::Entity(e) => PurgeSelector::Entity(e),
                protocol::PurgeAsk::Ids(ids) => PurgeSelector::Ids(ids),
                protocol::PurgeAsk::AgeOut { ids, committed_before_ns, aged_gap_ns } => {
                    match catalog.commit_seq_at(committed_before_ns) {
                        Ok(committed_by_seq) => {
                            PurgeSelector::AgeOut { ids, committed_by_seq, merge_gap_ns: aged_gap_ns }
                        }
                        Err(e) => return fail(error_code(&e), format!("reading the commit clock: {e}")),
                    }
                }
                protocol::PurgeAsk::Range { span, streams } if streams.is_empty() => {
                    PurgeSelector::Range { span, streams: None }
                }
                protocol::PurgeAsk::Range { span, streams } => {
                    let mut ids = Vec::new();
                    for name in &streams {
                        match catalog.stream_id(name) {
                            Ok(Some(s)) => ids.push(s),
                            Ok(None) => return fail(ErrorCode::NotFound, format!("no stream named {name}")),
                            Err(e) => return fail(error_code(&e), format!("looking up stream {name}: {e}")),
                        }
                    }
                    PurgeSelector::Range { span, streams: Some(ids) }
                }
            };
            let at_ns = now_ns();
            if let Err(e) = catalog.start_purge(selector, &reason, &detail, id, at_ns) {
                return fail(error_code(&e), format!("starting the purge: {e}"));
            }
            catalog_work += began.elapsed();
            health.mark_persisted(id, at_ns, now_ns(), Instant::now());
            log_purge_started(catalog, &health, id);
            return true;
        }
        let began = Instant::now();
        // The generic row controller includes commit/fsync overhead and can settle at one row even when row
        // work is cheap. Purges instead fill a cooperative work window in one catalog transaction, then yield to
        // the writer's next urgent batch. Commit latency is outside this work deadline.
        let result = catalog.purge_batch(id, config.max_budget.max(1), &mut || began.elapsed() >= config.step_target);
        let took = began.elapsed();
        catalog_work += took;
        steps += 1;
        health.record_step(id, took);
        match result {
            Ok(PurgeProgress::Running { delta, .. }) => {
                retries = 0;
                let _ = events.send(Event::PurgeStep { id, delta });
                true
            }
            Ok(PurgeProgress::Removed { report, delta }) => {
                health.finish(id);
                let elapsed = queued.elapsed();
                let record = catalog.purge_record(id).ok().flatten();
                let scope =
                    record.as_ref().map(|record| purge_scope(catalog, record)).unwrap_or_else(PurgeLogScope::missing);
                let streams = scope.streams_field();
                tracing::info!(
                    purge = %id,
                    steps,
                    ?elapsed,
                    ?catalog_work,
                    writer_wait = ?elapsed.saturating_sub(catalog_work),
                    removed = report.removed.len(),
                    scope_kind = scope.kind,
                    scope_streams = %streams,
                    scope_reason = %scope.reason,
                    scope_ids = scope.ids,
                    "catalog purge finished"
                );
                // Completing is the scheduler task's call (core::forget): it waits for the workers that keep copies.
                let _ = events.send(Event::PurgeStep { id, delta });
                let kind = record.as_ref().map(|r| r.kind.clone());
                let acked = record.map(|r| r.acked).unwrap_or_default();
                let _ = events.send(Event::PurgeDone { requester, report, kind, acked });
                false
            }
            // A storage error can be passing (a full disk freed, a lock); the step is tried again on the job's next
            // turn, a few times, before the purge is given up for this run (it stays pending in the catalog and
            // carries on at the next start).
            Err(CatalogError::Storage(e)) if retries < PURGE_RETRIES => {
                retries += 1;
                warn!(error = %e, retries, "a purge step failed; trying it again");
                true
            }
            Err(e) => fail(error_code(&e), format!("purge step: {e}")),
        }
    })
}

/// How many times a purge step's storage error is retried before the purge is given up for this run.
const PURGE_RETRIES: u32 = 3;

/// UTC nanoseconds now.
fn now_ns() -> u64 {
    u64::try_from(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos())
        .unwrap_or(u64::MAX)
}

/// The error code a client gets for a catalog error: the client's mistakes keep their meaning, and only a storage
/// failure is the broker's.
fn error_code(e: &CatalogError) -> ErrorCode {
    match e {
        CatalogError::Conflict(_) => ErrorCode::Conflict,
        CatalogError::Invalid(_) => ErrorCode::BadRequest,
        CatalogError::NotFound(_) => ErrorCode::NotFound,
        _ => ErrorCode::Internal,
    }
}

/// Counts a purge as active from when its job is made until the job is dropped (done, failed, or panicked).
struct ActivePurge(Arc<AtomicUsize>);

impl ActivePurge {
    fn new(active: &Arc<AtomicUsize>) -> Self {
        active.fetch_add(1, Ordering::SeqCst);
        Self(Arc::clone(active))
    }
}

impl Drop for ActivePurge {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn restart_read_queue_counts_queued_and_running_jobs_and_releases_on_panic() {
        let (sender, receiver) = crossbeam_channel::bounded(1);
        let queue = super::ReadQueue::new(sender);
        queue.try_send(Box::new(|_| panic!("synthetic read panic"))).unwrap();
        assert_eq!(queue.outstanding.load(std::sync::atomic::Ordering::SeqCst), 1);
        let job = receiver.recv().unwrap();
        assert_eq!(queue.outstanding.load(std::sync::atomic::Ordering::SeqCst), 1);
        let catalog = continuum_catalog::MemoryCatalog::new();
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job(&catalog))).is_err());
        assert_eq!(queue.outstanding.load(std::sync::atomic::Ordering::SeqCst), 0);
        queue.stopped.store(true, std::sync::atomic::Ordering::SeqCst);
        assert!(queue.try_send(Box::new(|_| panic!("stopped read ran"))).is_err());
        assert_eq!(queue.outstanding.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[test]
    fn refused_read_does_not_leave_a_phantom_checkpoint_barrier() {
        let (sender, _receiver) = crossbeam_channel::bounded(1);
        let queue = super::ReadQueue::new(sender);
        queue.try_send(Box::new(|_| {})).unwrap();
        assert!(queue.try_send(Box::new(|_| {})).is_err());
        assert_eq!(queue.outstanding.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn restart_barrier_terminates_writer_and_reports_after_prior_sync() {
        let catalog = continuum_catalog::MemoryCatalog::new();
        let (urgent, urgent_rx) = crossbeam_channel::bounded(4);
        let (_background, background_rx) = crossbeam_channel::bounded(1);
        let (events, mut received) = tokio::sync::mpsc::unbounded_channel();
        let (sync, _synced) = tokio::sync::oneshot::channel();
        let (reply, _checkpoint) = tokio::sync::oneshot::channel();
        urgent.send(super::Urgent::Sync(sync)).unwrap();
        urgent.send(super::Urgent::RestartCheckpoint { reply }).unwrap();
        super::writer(
            &catalog,
            &urgent_rx,
            &background_rx,
            &events,
            &super::WriterConfig::default(),
            &std::sync::atomic::AtomicUsize::new(0),
            Vec::new(),
            &super::writer_health::WriterHealth::default(),
        );
        assert!(matches!(received.try_recv().unwrap(), super::Event::Synced(_)));
        assert!(matches!(
            received.try_recv().unwrap(),
            super::Event::RestartCheckpoint { stamp: Err(super::CheckpointRefusal::NoStamp), .. }
        ));
        assert!(received.try_recv().is_err());
    }

    /// Runs the writer over one queued long job and a restart fence; returns the fence's stamp and whether the job
    /// was still alive (its guard not dropped) after the writer returned.
    fn fence_over_long_job(
        drop_at_stop: bool,
    ) -> (Result<continuum_catalog::RestartStamp, super::CheckpointRefusal>, bool) {
        struct Alive(std::sync::Arc<std::sync::atomic::AtomicBool>);
        impl Drop for Alive {
            fn drop(&mut self) {
                self.0.store(false, std::sync::atomic::Ordering::SeqCst);
            }
        }
        let temp = tempfile::tempdir().unwrap();
        let catalog = continuum_catalog::RedbCatalog::open(temp.path().join("catalog.redb")).unwrap();
        // Restart tracking starts with the first stamp read, as the server's startup does.
        catalog.restart_stamp().unwrap().unwrap();
        let (urgent, urgent_rx) = crossbeam_channel::bounded(4);
        let (background, background_rx) = crossbeam_channel::bounded(1);
        let (events, mut received) = tokio::sync::mpsc::unbounded_channel();
        let alive = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        let guard = Alive(std::sync::Arc::clone(&alive));
        // Never finishes on its own, like a long frame sweep.
        let job: super::BackgroundJob = Box::new(move |_, _| {
            let _ = &guard;
            true
        });
        background.send(super::LongJob { job, drop_at_stop }).unwrap();
        let (reply, _checkpoint) = tokio::sync::oneshot::channel();
        urgent.send(super::Urgent::RestartCheckpoint { reply }).unwrap();
        super::writer(
            &catalog,
            &urgent_rx,
            &background_rx,
            &events,
            &super::WriterConfig::default(),
            &std::sync::atomic::AtomicUsize::new(0),
            Vec::new(),
            &super::writer_health::WriterHealth::default(),
        );
        let alive = alive.load(std::sync::atomic::Ordering::SeqCst);
        match received.try_recv().unwrap() {
            super::Event::RestartCheckpoint { stamp, .. } => (stamp, alive),
            _ => panic!("the fence answers with its stamp"),
        }
    }

    #[test]
    fn restart_fence_drops_restartable_upkeep_before_it_answers() {
        let (stamp, alive) = fence_over_long_job(true);
        assert!(stamp.is_ok(), "a sweep that is always running must not refuse the fence: {stamp:?}");
        assert!(!alive, "the dropped sweep clears its running flag before the scheduler hears the answer");
    }

    #[test]
    fn restart_fence_refuses_while_other_long_work_is_queued() {
        let (stamp, _) = fence_over_long_job(false);
        assert_eq!(stamp, Err(super::CheckpointRefusal::WriterBusy));
    }

    use std::io::{self, Write};
    use std::sync::Mutex;

    use tracing_subscriber::fmt::MakeWriter;

    use super::*;

    #[derive(Clone, Default)]
    struct LogBuf(Arc<Mutex<Vec<u8>>>);

    impl Write for LogBuf {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl<'a> MakeWriter<'a> for LogBuf {
        type Writer = Self;
        fn make_writer(&'a self) -> Self {
            self.clone()
        }
    }

    fn captured_logs(emit: impl FnOnce()) -> String {
        let buf = LogBuf::default();
        let sub = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .with_writer(buf.clone())
            .with_ansi(false)
            .finish();
        tracing::subscriber::with_default(sub, emit);
        String::from_utf8(buf.0.lock().unwrap().clone()).unwrap()
    }

    #[test]
    fn archive_pressure_episodes_and_losses_reach_the_health_line() {
        let mut archive = ArchiveHealth::default();
        let logs = captured_logs(|| {
            archive.pressure(0, 4);
            archive.pressure(0, 3);
            archive.pressure(1, 2);
            archive.pressure(0, 0);
            archive.pressure(0, 4);
            archive.slot_gap(StreamId(1), Interval::new(10, 20).unwrap(), "producer overwrite");
            archive.ring_gap(StreamId(2), Interval::new(20, 30).unwrap());
            let writer = writer_health::WriterHealth::default();
            log_health_line(3, 0, &writer.snapshot(Instant::now(), 0), &archive);
        });
        assert_eq!(logs.matches("Tier 0 archive pressure:").count(), 3, "one warning per class/episode");
        assert_eq!(archive.pressure_classes.len(), 2);
        let line = logs.lines().find(|line| line.contains("broker health")).unwrap();
        for field in [
            "tier0_slot_archive_gaps=1",
            "tier0_ring_archive_gaps=1",
            "tier0_reclaim_pressure_events=3",
            "tier0_reclaim_pressure_classes=2",
        ] {
            assert!(line.contains(field), "{field} missing from {line}");
        }
    }

    #[test]
    fn ring_archive_loss_counts_only_removed_uncovered_batches() {
        let stream = StreamId(1);
        let span = Interval::new(100, 200).unwrap();
        let mut health = ArchiveHealth::default();
        for (purged, archived, owed, expected) in
            [(true, false, true, 0), (false, true, true, 0), (false, false, false, 0), (false, false, true, 1)]
        {
            let mut index = AvailabilityIndex::new();
            index.register(stream, StreamMeta { kind: StreamKind::Sensor, rate_hz: 0.0 });
            index.add_copy(stream, span, StorageTier::Tier0Hot).unwrap();
            if archived {
                index.add_copy(stream, span, StorageTier::Tier1Warm).unwrap();
            }
            let batch = RingBatch { id: Uuid([1; 16]), purged, first_seq: 0, count: 10, stream, span };
            remove_lapped_batch(&mut index, &batch, owed, &mut health);
            assert_eq!(health.ring_gaps, expected);
            remove_lapped_batch(&mut index, &batch, owed, &mut health);
            assert_eq!(health.ring_gaps, expected, "a repeated observation is not another loss");
        }
    }

    fn config() -> WriterConfig {
        WriterConfig { step_target: Duration::from_millis(5), initial_budget: 256, max_budget: 4096 }
    }

    /// A MemoryCatalog that counts insert transactions.
    #[derive(Default)]
    struct Counting {
        inner: continuum_catalog::MemoryCatalog,
        inserts: std::sync::atomic::AtomicUsize,
        /// When set, every insert fails with a storage error (a full disk, say).
        disk_full: std::sync::atomic::AtomicBool,
        /// When set, every coverage write fails with a storage error.
        coverage_fails: std::sync::atomic::AtomicBool,
        /// A finite test controller keeps the next urgent write ready before releasing this one.
        cost_gate: Option<(crossbeam_channel::Sender<()>, crossbeam_channel::Receiver<()>)>,
        cost_saves: AtomicUsize,
        fulfillment_get: Option<FulfillmentGetGate>,
        cursor_gate: Option<CursorGate>,
        snapshot_swap: Option<Mutex<SnapshotSwap>>,
    }

    struct SnapshotSwap {
        current: continuum_catalog::ChunkSnapshot,
        next: continuum_catalog::ChunkSnapshot,
    }

    #[test]
    fn get_chunk_reads_one_snapshot_when_copies_change_after_get() {
        use continuum_catalog::{ChunkFrame, ChunkSnapshot, Indexed, Resolved};
        let id = Uuid([1; 16]);
        let header = HeaderBuf::build(&continuum_core::HeaderSpec::raw(id, StreamId(1), 10, 20)).unwrap();
        let old = ChunkSnapshot {
            indexed: Indexed {
                header: header.clone(),
                locations: vec![StorageLocator::Chunk {
                    tier: StorageTier::Tier1Warm,
                    uri: "/before.cnk".into(),
                    offset: 0,
                    length: 100,
                }],
                frames: Some(vec![ChunkFrame::new(10, 1, 100)]),
                commit_seq: 1,
            },
            provenance: Some(continuum_catalog::ChunkProvenance::new(vec![continuum_catalog::FrameProvenance {
                device_seq: 4,
                epoch: 9,
                stamped_at_arrival: false,
            }])),
        };
        let mut new = old.clone();
        new.indexed.locations = vec![StorageLocator::Chunk {
            tier: StorageTier::Tier2Cold,
            uri: "/after.cnk".into(),
            offset: 0,
            length: 50,
        }];
        new.provenance = None;
        let c = Counting {
            snapshot_swap: Some(Mutex::new(SnapshotSwap { current: old.clone(), next: new.clone() })),
            ..Default::default()
        };
        // Reproduce a split read: get returns A, then a rewrite swaps the catalog to B.
        let Some(Resolved::Payload(split)) = c.get(id).unwrap() else { panic!("fixture chunk missing") };
        assert_eq!(split.locations, old.indexed.locations);
        c.snapshot_swap.as_ref().unwrap().lock().unwrap().current = old.clone();
        let result = super::aging::chunk_snapshot(&c, id).unwrap_or_else(|e| panic!("{e:?}"));
        let super::aging::Done::Chunk(snapshot) = result else { panic!("not a chunk reply") };
        assert_eq!(snapshot.indexed.locations, old.indexed.locations);
        assert_eq!(
            snapshot.provenance, old.provenance,
            "the production GetChunk path cannot pair A's locator with B's provenance"
        );
    }

    /// Holds the writer inside its first durable-cursor write until the test releases it, and records every write.
    struct CursorGate {
        entered: crossbeam_channel::Sender<()>,
        release: crossbeam_channel::Receiver<()>,
        once: std::sync::atomic::AtomicBool,
        written: Arc<Mutex<Vec<(String, u64)>>>,
    }

    struct FulfillmentGetGate {
        first: Uuid,
        second: Uuid,
        first_entered: crossbeam_channel::Sender<()>,
        first_release: crossbeam_channel::Receiver<()>,
        second_finished: crossbeam_channel::Sender<()>,
        first_once: std::sync::atomic::AtomicBool,
        outcome: FulfillmentTestOutcome,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum FulfillmentTestOutcome {
        Success,
        ReadError,
        ReadPanic,
        Cancel,
        DuplicateCommit,
    }

    mod counting_impl {
        use std::sync::atomic::Ordering;

        use continuum_catalog::{
            Catalog, ChunkFrame, ClockNear, ClockPoint, CoverageMark, ForgottenHash, NewChunk, NewPayload, Page,
            PurgeProgress, PurgeRecord, PurgeReport, PurgeSelector, RangeCursor, Resolved, Result, ScrubProgress,
            StreamInfo,
        };
        use continuum_core::{BootId, Interval, StageId, StorageLocator, StorageTier, StreamId, Uuid};

        impl Catalog for super::Counting {
            fn register_stream(&self, info: StreamInfo) -> Result<StreamId> {
                self.inner.register_stream(info)
            }
            fn stream(&self, id: StreamId) -> Result<Option<StreamInfo>> {
                self.inner.stream(id)
            }
            fn streams(&self, after: Option<StreamId>, limit: usize) -> Result<Vec<(StreamId, StreamInfo)>> {
                self.inner.streams(after, limit)
            }
            fn coverage_keys(
                &self,
                after: Option<(StreamId, StageId)>,
                limit: usize,
            ) -> Result<Vec<(StreamId, StageId)>> {
                self.inner.coverage_keys(after, limit)
            }
            fn stream_id(&self, name: &str) -> Result<Option<StreamId>> {
                self.inner.stream_id(name)
            }
            fn insert(&self, payloads: Vec<NewPayload>) -> Result<Vec<continuum_catalog::Inserted>> {
                self.inserts.fetch_add(1, Ordering::SeqCst);
                if self.disk_full.load(Ordering::SeqCst) {
                    return Err(continuum_catalog::CatalogError::Storage("no space left on device".into()));
                }
                self.inner.insert(payloads)
            }
            fn get(&self, id: Uuid) -> Result<Option<Resolved>> {
                if let Some(gate) = &self.snapshot_swap {
                    let mut gate = gate.lock().unwrap();
                    let known = gate.current.indexed.clone();
                    gate.current = gate.next.clone();
                    return Ok(Some(Resolved::Payload(known)));
                }
                if let Some(gate) = &self.fulfillment_get
                    && id == gate.first
                    && !gate.first_once.swap(true, Ordering::SeqCst)
                {
                    let _ = gate.first_entered.send(());
                    gate.first_release.recv_timeout(std::time::Duration::from_secs(20)).map_err(|error| {
                        continuum_catalog::CatalogError::Storage(format!("fulfillment test gate: {error}").into())
                    })?;
                    match gate.outcome {
                        super::FulfillmentTestOutcome::ReadError => {
                            return Err(continuum_catalog::CatalogError::Storage(
                                "injected first fulfillment read failure".into(),
                            ));
                        }
                        super::FulfillmentTestOutcome::ReadPanic => panic!("injected fulfillment panic"),
                        _ => {}
                    }
                }
                let result = self.inner.get(id);
                if let Some(gate) = &self.fulfillment_get
                    && id == gate.second
                {
                    let _ = gate.second_finished.try_send(());
                }
                result
            }
            fn chunk_snapshot(&self, id: Uuid) -> Result<Option<continuum_catalog::ChunkSnapshot>> {
                if let Some(gate) = &self.snapshot_swap {
                    let mut gate = gate.lock().unwrap();
                    let known = gate.current.clone();
                    gate.current = gate.next.clone();
                    return Ok(Some(known));
                }
                self.inner.chunk_snapshot(id)
            }
            fn range(
                &self,
                stream: StreamId,
                window: Interval,
                after: Option<RangeCursor>,
                limit: usize,
            ) -> Result<Page> {
                self.inner.range(stream, window, after, limit)
            }
            fn parents(&self, id: Uuid) -> Result<Vec<Uuid>> {
                self.inner.parents(id)
            }
            fn children(&self, id: Uuid) -> Result<Vec<Uuid>> {
                self.inner.children(id)
            }
            fn descendants(&self, roots: &[Uuid], limit: usize) -> Result<Vec<Uuid>> {
                self.inner.descendants(roots, limit)
            }
            fn set_location(&self, id: Uuid, locator: StorageLocator) -> Result<()> {
                self.inner.set_location(id, locator)
            }
            fn clear_location(&self, id: Uuid, tier: StorageTier) -> Result<()> {
                self.inner.clear_location(id, tier)
            }
            fn commit_chunk(&self, chunk: NewChunk, absorbed: &[Uuid]) -> Result<continuum_catalog::Inserted> {
                self.inner.commit_chunk(chunk, absorbed)
            }
            fn head_seq(&self) -> Result<u64> {
                self.inner.head_seq()
            }
            fn query(
                &self,
                stream: StreamId,
                after_seq: u64,
                filter: &continuum_catalog::QueryFilter,
                limit: usize,
                max_scan: usize,
            ) -> Result<continuum_catalog::SeqPage> {
                self.inner.query(stream, after_seq, filter, limit, max_scan)
            }
            fn purges(&self, after: Option<Uuid>, limit: usize) -> Result<Vec<continuum_catalog::PurgeSummary>> {
                self.inner.purges(after, limit)
            }
            fn rewrite_chunk(&self, chunk: Uuid, frames: Vec<ChunkFrame>, location: StorageLocator) -> Result<()> {
                self.inner.rewrite_chunk(chunk, frames, location)
            }
            fn add_coverage(&self, mark: CoverageMark) -> Result<()> {
                if self.coverage_fails.load(Ordering::SeqCst) {
                    return Err(continuum_catalog::CatalogError::Storage("the disk failed".into()));
                }
                self.inner.add_coverage(mark)
            }
            fn coverage(&self, source: StreamId, stage: StageId, window: Interval) -> Result<Vec<Interval>> {
                self.inner.coverage(source, stage, window)
            }
            fn coverage_gaps(
                &self,
                source: StreamId,
                stage: StageId,
                window: Interval,
                max: usize,
            ) -> Result<Vec<Interval>> {
                self.inner.coverage_gaps(source, stage, window, max)
            }
            fn start_purge(
                &self,
                selector: PurgeSelector,
                reason: &str,
                detail: &str,
                purge_id: Uuid,
                now_ns: u64,
            ) -> Result<()> {
                self.inner.start_purge(selector, reason, detail, purge_id, now_ns)
            }
            fn purge_step(&self, purge_id: Uuid, budget: usize) -> Result<PurgeProgress> {
                self.inner.purge_step(purge_id, budget)
            }
            fn purge(
                &self,
                selector: PurgeSelector,
                reason: &str,
                detail: &str,
                purge_id: Uuid,
                now_ns: u64,
            ) -> Result<PurgeReport> {
                self.inner.purge(selector, reason, detail, purge_id, now_ns)
            }
            fn purge_record(&self, purge_id: Uuid) -> Result<Option<PurgeRecord>> {
                self.inner.purge_record(purge_id)
            }
            fn pending_purges(&self) -> Result<Vec<Uuid>> {
                self.inner.pending_purges()
            }
            fn complete_purge(&self, purge_id: Uuid, now_ns: u64) -> Result<()> {
                self.inner.complete_purge(purge_id, now_ns)
            }
            fn keepers(&self) -> Result<Vec<String>> {
                self.inner.keepers()
            }
            fn add_keeper(&self, name: &str) -> Result<()> {
                self.inner.add_keeper(name)
            }
            fn forget_keeper(&self, name: &str) -> Result<bool> {
                self.inner.forget_keeper(name)
            }
            fn purge_acked(&self, purge_id: Uuid, keeper: &str) -> Result<()> {
                self.inner.purge_acked(purge_id, keeper)
            }
            fn request_scrub(&self) -> Result<()> {
                self.inner.request_scrub()
            }
            fn scrub_needed(&self) -> Result<bool> {
                self.inner.scrub_needed()
            }
            fn scrub_step(&self, budget: usize) -> Result<ScrubProgress> {
                self.inner.scrub_step(budget)
            }
            fn abort_scrub(&self) -> Result<bool> {
                self.inner.abort_scrub()
            }
            fn storage(&self) -> Result<Option<continuum_catalog::CatalogStorage>> {
                self.inner.storage()
            }
            fn prune_frames(
                &self,
                before_ns: u64,
                after: Option<(Uuid, Uuid)>,
                budget: usize,
            ) -> Result<continuum_catalog::PruneProgress> {
                self.inner.prune_frames(before_ns, after, budget)
            }
            fn set_cursor(&self, name: &str, position: &[u8]) -> Result<()> {
                if let Some(gate) = &self.cursor_gate {
                    if !gate.once.swap(true, Ordering::SeqCst) {
                        let _ = gate.entered.send(());
                        let _ = gate.release.recv_timeout(std::time::Duration::from_secs(30));
                    }
                    let saved = u64::from_le_bytes(position.try_into().expect("an eight-byte position"));
                    gate.written.lock().unwrap().push((name.to_owned(), saved));
                }
                self.inner.set_cursor(name, position)
            }
            fn cursor(&self, name: &str) -> Result<Option<Vec<u8>>> {
                self.inner.cursor(name)
            }
            fn record_clock(&self, boot: BootId, point: ClockPoint) -> Result<()> {
                self.inner.record_clock(boot, point)
            }
            fn clock_near(&self, boot: BootId, mono_ns: u64) -> Result<ClockNear> {
                self.inner.clock_near(boot, mono_ns)
            }
            fn save_cost_models(&self, bytes: &[u8]) -> Result<()> {
                self.cost_saves.fetch_add(1, Ordering::SeqCst);
                if let Some((entered, release)) = &self.cost_gate {
                    entered.send(()).expect("the finite urgent controller is listening");
                    release.recv().expect("the finite urgent controller releases each write");
                }
                self.inner.save_cost_models(bytes)
            }
            fn load_cost_models(&self) -> Result<Option<Vec<u8>>> {
                self.inner.load_cost_models()
            }
            fn forget(&self, name: ForgottenHash) -> Result<()> {
                self.inner.forget(name)
            }
            fn save_worker_state(&self, state: &continuum_catalog::WorkerState) -> Result<()> {
                self.inner.save_worker_state(state)
            }
            fn load_worker_state(&self, owner: &str, contract: &str) -> Result<Option<continuum_catalog::WorkerState>> {
                self.inner.load_worker_state(owner, contract)
            }
            fn forget_worker_state(&self, owner: &str, contract: &str) -> Result<bool> {
                self.inner.forget_worker_state(owner, contract)
            }
            fn list_forgotten(
                &self,
                after: Option<ForgottenHash>,
                limit: usize,
            ) -> Result<continuum_catalog::ForgottenPage> {
                self.inner.list_forgotten(after, limit)
            }
            fn is_forgotten(&self, name: ForgottenHash) -> Result<bool> {
                self.inner.is_forgotten(name)
            }
        }
    }

    async fn receive_fulfillment_gate(
        receiver: crossbeam_channel::Receiver<()>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        tokio::task::spawn_blocking(move || receiver.recv_timeout(Duration::from_secs(20))).await??;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn native_promise_fulfilled_keeps_parts_in_action_order_when_reads_finish_out_of_order() {
        fulfillment_with_overtaking_final(FulfillmentTestOutcome::Success).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn native_promise_read_error_cancels_unfinished_capture_and_discards_later_parts() {
        fulfillment_with_overtaking_final(FulfillmentTestOutcome::ReadError).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn native_promise_fulfilled_turns_a_panicked_read_into_failure() {
        fulfillment_with_overtaking_final(FulfillmentTestOutcome::ReadPanic).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn native_promise_cancel_discards_completed_and_inflight_parts() {
        fulfillment_with_overtaking_final(FulfillmentTestOutcome::Cancel).await;
    }

    /// After a restart, a durable follower's catch-up acknowledged every event while the writer was busy. Each
    /// ack took a slot in the writer's urgent queue, the queue filled, and later acks were dropped (5,635 warnings in
    /// one second), so the saved cursor stayed behind the work done. The newest position must still reach the catalog,
    /// in a few writes rather than one per ack.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_burst_of_durable_acks_behind_a_busy_writer_still_saves_the_newest_position() {
        use continuum_client::{Client, ClientOptions, Start, StreamSpec, StreamStorage, SubscribeSpec};
        use continuum_core::schema::continuum as fb;

        let directory = tempfile::tempdir().unwrap();
        let socket = directory.path().join("broker.sock");
        let mut config = crate::server::Config::new(socket.clone(), Duration::from_secs(60));
        config.tier0 = Some(vec![continuum_sched::SlotClass { size_bytes: 16_384, count: 4 }]);
        config.tier0_ring_bytes = 4096;
        let (entered, entered_wait) = crossbeam_channel::bounded(1);
        let (release, released) = crossbeam_channel::bounded(1);
        let written = Arc::new(Mutex::new(Vec::new()));
        let catalog = Counting {
            cursor_gate: Some(CursorGate {
                entered,
                release: released,
                once: std::sync::atomic::AtomicBool::new(false),
                written: Arc::clone(&written),
            }),
            ..Counting::default()
        };
        let server = crate::server::Server::bind_with_catalog(config, Box::new(catalog)).await.unwrap();
        let (stop, stopped) = oneshot::channel();
        let server_task = tokio::spawn(server.run_until(async {
            let _ = stopped.await;
        }));
        let producer = Client::connect(&socket, ClientOptions::new("ack-producer")).await.unwrap();
        producer
            .register_stream(&StreamSpec {
                name: "capture".into(),
                kind: fb::StreamKind::Sensor,
                format: fb::PayloadFormat::RawBytes,
                shape_name: "synthetic".into(),
                shape_version: 1,
                nominal_rate_hz: 10.0,
                storage: StreamStorage::Slab { class: 0 },
                anchor: None,
                layer: fb::DomainLayer::Layer1Raw,
            })
            .await
            .unwrap();
        let follower = Client::connect(&socket, ClientOptions::new("ack-follower")).await.unwrap();
        let mut spec = SubscribeSpec::new(["capture"], Start::Saved);
        spec.durable_name = Some("follower".into());
        let sub = follower.subscribe(&spec).await.unwrap();

        sub.ack(1).unwrap();
        // The writer is now inside the write of position 1; everything after it waits on the writer.
        let gate = tokio::task::spawn_blocking(move || entered_wait.recv_timeout(Duration::from_secs(30)));
        gate.await.unwrap().expect("the writer took the first cursor write");
        let last = URGENT as u64 + 1000;
        for position in 2..=last {
            sub.ack(position).unwrap();
        }
        // The scheduler handles commands in order, so once this answers, every ack above has been handled.
        follower.list_streams().await.unwrap();
        release.send(()).unwrap();

        let mut saved = Vec::new();
        for _ in 0..2_000 {
            saved = written.lock().unwrap().clone();
            if saved.last().is_some_and(|(_, position)| *position == last) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let _ = stop.send(());
        server_task.await.unwrap();
        assert_eq!(saved.last(), Some(&("follower".to_owned(), last)), "writes: {saved:?}");
        assert!(saved.len() <= 3, "{} acks coalesce into a few writes, not {}", last, saved.len());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn rejected_duplicate_commit_preserves_the_original_pending_delivery() {
        fulfillment_with_overtaking_final(FulfillmentTestOutcome::DuplicateCommit).await;
    }

    async fn fulfillment_with_overtaking_final(outcome: FulfillmentTestOutcome) {
        use continuum_client::{Client, ClientOptions, QuerySpec, StreamSpec, StreamStorage};
        use continuum_core::HeaderSpec;
        use continuum_core::schema::continuum as fb;

        let directory = tempfile::tempdir().unwrap();
        let socket = directory.path().join("broker.sock");
        let mut config = crate::server::Config::new(socket.clone(), Duration::from_secs(60));
        config.tier0 = Some(vec![continuum_sched::SlotClass { size_bytes: 16_384, count: 4 }]);
        config.tier0_ring_bytes = 4096;
        let first = Uuid([1; 16]);
        let second = Uuid([2; 16]);
        let (first_entered, first_wait) = crossbeam_channel::bounded(1);
        let (first_release, first_released) = crossbeam_channel::bounded(1);
        let (second_finished, second_wait) = crossbeam_channel::bounded(1);
        let catalog = Counting {
            fulfillment_get: Some(FulfillmentGetGate {
                first,
                second,
                first_entered,
                first_release: first_released,
                second_finished,
                first_once: std::sync::atomic::AtomicBool::new(false),
                outcome,
            }),
            ..Counting::default()
        };
        let server = crate::server::Server::bind_with_catalog(config, Box::new(catalog)).await.unwrap();
        let core = server.core();
        let (stop, stopped) = oneshot::channel();
        let server_task = tokio::spawn(server.run_until(async {
            let _ = stopped.await;
        }));
        let (block_entered, block_wait) = crossbeam_channel::unbounded();
        let (block_release, block_released) = crossbeam_channel::unbounded();

        let setup: Result<_, Box<dyn std::error::Error + Send + Sync>> = async {
            let producer = Client::connect(&socket, ClientOptions::new("fulfillment-producer")).await?;
            let consumer = Client::connect(&socket, ClientOptions::new("fulfillment-consumer")).await?;
            let registered = producer
                .register_stream(&StreamSpec {
                    name: "capture".into(),
                    kind: fb::StreamKind::Sensor,
                    format: fb::PayloadFormat::RawBytes,
                    shape_name: "synthetic".into(),
                    shape_version: 1,
                    nominal_rate_hz: 10.0,
                    storage: StreamStorage::Slab { class: 0 },
                    anchor: None,
                    layer: fb::DomainLayer::Layer1Raw,
                })
                .await?;
            let start = now_ns() + 60_000_000_000;
            let middle = start + 100_000_000;
            let end = middle + 100_000_000;
            let mut ask = QuerySpec::hit_only("capture", (start, end));
            ask.mode = fb::QueryMode::QuoteIfMissing;
            let (_, quote) = consumer.query_quoted(&ask).await?;
            let quote = quote.ok_or("future capture did not produce a quote")?;
            let (_, promise) = consumer.commit_promise(quote.promise_id, 0, fb::PriorityBand::P2Nearline).await?;

            // Leave exactly two reader threads: the blocked first part and the overtaking final part.
            for _ in 0..DEFAULT_READERS - 2 {
                let entered = block_entered.clone();
                let release = block_released.clone();
                core.read(Box::new(move |_| {
                    let _ = entered.send(());
                    let _ = release.recv_timeout(Duration::from_secs(20));
                }))
                .await?;
                receive_fulfillment_gate(block_wait.clone()).await?;
            }
            let locator = StorageLocator::Inline { bytes: b"{}".to_vec() };
            let stream = StreamId(registered.stream_id);
            let first_header = HeaderBuf::build(&HeaderSpec::raw(first, stream, start, middle))?;
            producer.ingest_acked(&first_header, Some(&locator)).await?;
            receive_fulfillment_gate(first_wait).await?;
            // The error case leaves capture incomplete so terminal delivery failure must also cancel live work.
            let second_end = if outcome == FulfillmentTestOutcome::ReadError { end - 1 } else { end };
            let second_header = HeaderBuf::build(&HeaderSpec::raw(second, stream, middle, second_end))?;
            producer.ingest_acked(&second_header, Some(&locator)).await?;
            receive_fulfillment_gate(second_wait).await?;

            // Only the later-part worker can run this FIFO sentinel. It sends Fulfilled before taking it.
            let (sentinel, passed) = oneshot::channel();
            core.read(Box::new(move |_| {
                let _ = sentinel.send(());
            }))
            .await?;
            tokio::time::timeout(Duration::from_secs(20), passed).await??;
            // Events precede ordinary scheduler commands, so this reply proves the second event was handled.
            core.cost_models().await?;
            consumer.heartbeat(None).await?;
            let unrelated = consumer.query(&QuerySpec::hit_only("capture", (start, end))).await?;
            let query_ids = unrelated.items.iter().map(|item| item.header.id()).collect::<Vec<_>>();
            // Occupy the final reader too. Once the first get is released, only that reader can run our next
            // sentinel, proving its late event was queued before we inspect cancellation/error aftermath.
            let entered = block_entered.clone();
            let release = block_released.clone();
            core.read(Box::new(move |_| {
                let _ = entered.send(());
                let _ = release.recv_timeout(Duration::from_secs(20));
            }))
            .await?;
            receive_fulfillment_gate(block_wait.clone()).await?;
            if outcome == FulfillmentTestOutcome::DuplicateCommit {
                // The native SDK locally refuses a second handle. Inject the already-validated command to
                // exercise the broker's same-owner duplicate request while retaining the original SDK drain.
                core.send(Command::CommitPromise {
                    session: uuid::Uuid::from_bytes(consumer.session_id().0),
                    request_id: 0,
                    promise: crate::protocol::promise_of(promise.id()).ok_or("not a promise id")?,
                    option: 0,
                    band: fb::PriorityBand::P2Nearline,
                })
                .await?;
                let rejection = tokio::time::timeout(Duration::from_secs(20), consumer.next_event())
                    .await?
                    .ok_or("duplicate rejection disappeared")?;
                let envelope = rejection.envelope();
                if envelope.request_id() != 0 || envelope.payload_as_error().is_none() {
                    return Err("duplicate commit was not rejected".into());
                }
            }
            let cancelled = if outcome == FulfillmentTestOutcome::Cancel {
                Some(consumer.cancel_promise(promise.id()).await?.life)
            } else {
                None
            };
            Ok((promise, query_ids, producer, consumer, cancelled))
        }
        .await;

        // Release every finite gate even if setup failed; assertions come only after broker shutdown.
        let _ = first_release.send(());
        let late_fence: Result<(), Box<dyn std::error::Error + Send + Sync>> = async {
            if setup.is_ok() {
                let (sentinel, passed) = oneshot::channel();
                core.read(Box::new(move |_| {
                    let _ = sentinel.send(());
                }))
                .await?;
                tokio::time::timeout(Duration::from_secs(20), passed).await??;
                core.cost_models().await?;
            }
            Ok(())
        }
        .await;
        for _ in 0..DEFAULT_READERS - 1 {
            let _ = block_release.send(());
        }
        let result = match setup {
            Ok((promise, query_ids, producer, consumer, cancelled)) => {
                let delivered = tokio::time::timeout(Duration::from_secs(20), promise.fulfilled()).await;
                // A socket reply follows every earlier push, so a late completion cannot hide in transit.
                let heartbeat = consumer.heartbeat(None).await;
                let stray = {
                    let mut next = Box::pin(consumer.next_event());
                    std::future::poll_fn(|cx| {
                        std::task::Poll::Ready(std::future::Future::poll(next.as_mut(), cx).is_ready())
                    })
                    .await
                };
                let _ = producer.close("test finished").await;
                let _ = consumer.close("test finished").await;
                delivered
                    .map(|result| (result, query_ids, cancelled, stray, heartbeat))
                    .map_err(|error| format!("fulfillment drain: {error}"))
            }
            Err(error) => Err(format!("setup: {error}")),
        };
        let _ = stop.send(());
        server_task.await.unwrap();
        late_fence.unwrap();
        let (delivered, query_ids, cancelled, stray, heartbeat) = result.unwrap();
        heartbeat.unwrap();
        assert!(!stray, "ended promises must not leak late parts onto the generic push path");
        assert_eq!(query_ids, [first, second], "unrelated reads progress while first fulfillment is blocked");
        match outcome {
            FulfillmentTestOutcome::ReadError | FulfillmentTestOutcome::ReadPanic => {
                let expected = if outcome == FulfillmentTestOutcome::ReadError {
                    "injected first fulfillment read failure"
                } else {
                    "reading a promise fulfillment panicked"
                };
                assert!(
                    matches!(
                        delivered,
                        Err(continuum_client::Error::PromiseFailed { reason, delivered: 0 }) if reason.contains(expected)
                    ),
                    "the first read error must fail before any later part is delivered"
                );
            }
            FulfillmentTestOutcome::Cancel => {
                assert_eq!(cancelled, Some(fb::PromiseLife::Cancelled));
                assert!(delivered.unwrap().is_empty(), "cancellation discards buffered and late parts");
            }
            FulfillmentTestOutcome::Success | FulfillmentTestOutcome::DuplicateCommit => {
                let ids = delivered.unwrap().iter().map(|item| item.header.id()).collect::<Vec<_>>();
                assert_eq!(ids, [first, second], "native fulfilled() must not finish at the overtaking final part");
            }
        }
    }

    #[test]
    fn continuously_ready_urgent_writes_allow_round_robin_background_progress() {
        let (entered, entries) = crossbeam_channel::bounded(1);
        let (release, releases) = crossbeam_channel::bounded(1);
        let catalog = Counting { cost_gate: Some((entered, releases)), ..Counting::default() };
        let (urgent_tx, urgent_rx) = crossbeam_channel::bounded(1);
        let (background_tx, background_rx) = crossbeam_channel::bounded(2);
        let health = Arc::new(writer_health::WriterHealth::default());
        let progress = Arc::new(std::sync::Mutex::new(Vec::new()));
        // Queue both jobs and urgent work before starting the writer. Each job needs two turns.
        for name in ['A', 'B'] {
            let progress = Arc::clone(&progress);
            let ready = urgent_rx.clone();
            let health = Arc::clone(&health);
            let mut turn = 0;
            let job: BackgroundJob = Box::new(move |catalog, _budget| {
                turn += 1;
                let saves = catalog.load_cost_models().unwrap().unwrap();
                assert!(health.snapshot(Instant::now(), 0).running, "the running turn is visible");
                progress.lock().unwrap().push((name, turn, saves[0], !ready.is_empty()));
                turn < 2
            });
            background_tx.send(LongJob::from(job)).unwrap();
        }
        urgent_tx.send(Urgent::SaveCosts(vec![1])).unwrap();
        let (events, _events_rx) = mpsc::unbounded_channel();
        let active = AtomicUsize::new(0);
        let config = config();
        std::thread::scope(|scope| {
            let writer = scope
                .spawn(|| writer(&catalog, &urgent_rx, &background_rx, &events, &config, &active, Vec::new(), &health));
            for n in 1..=6u8 {
                entries.recv().unwrap();
                if n < 6 {
                    // The next urgent write is definitely ready when the current write returns.
                    urgent_tx.send(Urgent::SaveCosts(vec![n + 1])).unwrap();
                }
                release.send(()).unwrap();
            }
            // Always let the finite stream finish and the writer exit before asserting. The old writer
            // drains all six urgent writes first, so it fails rather than leaving a blocked test thread.
            drop(urgent_tx);
            drop(background_tx);
            writer.join().unwrap();
        });
        assert_eq!(catalog.cost_saves.load(std::sync::atomic::Ordering::SeqCst), 6);
        let finished = health.snapshot(Instant::now(), 0);
        assert_eq!(finished.queued, 0);
        assert!(!finished.running);
        let progress = progress.lock().unwrap();
        assert_eq!(
            progress.iter().map(|&(name, turn, _, _)| (name, turn)).collect::<Vec<_>>(),
            [('A', 1), ('B', 1), ('A', 2), ('B', 2)]
        );
        assert!(
            progress.iter().all(|&(_, _, saved, ready)| saved < 6 && ready),
            "background must progress while another urgent write is ready: {progress:?}"
        );
        assert!(
            progress.windows(2).all(|pair| pair[0].2 < pair[1].2),
            "urgent writes must progress between background turns: {progress:?}"
        );
    }

    #[test]
    fn a_trim_does_not_commit_once_per_row_at_the_scheduler_budget_floor() {
        use continuum_core::{DomainLayer, HeaderSpec, PayloadFormat};
        let temp = tempfile::tempdir().unwrap();
        let catalog = continuum_catalog::RedbCatalog::open(temp.path().join("trim.redb")).unwrap();
        let stream = catalog
            .register_stream(StreamInfo {
                name: "synthetic.live".into(),
                kind: CatalogKind::Derived,
                layer: DomainLayer::Layer1Raw,
                format: PayloadFormat::Text,
                shape_name: "synthetic.live".into(),
                shape_version: 1,
                rate_hz: None,
                anchor: None,
            })
            .unwrap();
        let identity = |n: u32| {
            let mut bytes = [0u8; 16];
            bytes[6] = 0x70;
            bytes[8] = 0x80;
            bytes[12..].copy_from_slice(&n.to_be_bytes());
            Uuid(bytes)
        };
        catalog
            .insert(
                (0..132)
                    .map(|n| NewPayload {
                        header: HeaderBuf::build(&HeaderSpec::raw(identity(n), stream, u64::from(n), u64::from(n)))
                            .unwrap(),
                        location: None,
                        coverage: None,
                    })
                    .collect(),
            )
            .unwrap();
        let health = Arc::new(writer_health::WriterHealth::default());
        let active = Arc::new(AtomicUsize::new(0));
        let (events, mut received) = mpsc::unbounded_channel();
        let mut job = purge_job(
            None,
            Some((
                protocol::PurgeAsk::Range {
                    span: Interval::new(0, 128).unwrap(),
                    streams: vec!["synthetic.live".into()],
                },
                "synthetic trim".into(),
                String::new(),
            )),
            identity(1000),
            events,
            &active,
            &health,
            WriterConfig::default(),
        );
        assert!(job(&catalog, 1));
        let mut steps = 0;
        loop {
            steps += 1;
            let more = job(&catalog, 1);
            assert!(steps < 1000, "synthetic trim must terminate");
            if !more {
                break;
            }
        }
        let mut removed = None;
        while let Ok(event) = received.try_recv() {
            if let Event::PurgeDone { report, .. } = event {
                removed = Some(report.removed.len());
            }
        }
        assert_eq!(removed, Some(128));
        for n in 0..132 {
            assert_eq!(catalog.get(identity(n)).unwrap().is_some(), n >= 128, "recent tail is retained");
        }
        eprintln!("synthetic trim: 128 rows, {steps} catalog steps at scheduler budget 1");
        assert!(steps <= 16, "fixed transaction overhead must not force singleton commits: {steps}");
    }

    #[test]
    fn purge_scope_names_the_kind_the_streams_and_a_short_reason() {
        let catalog = continuum_catalog::MemoryCatalog::new();
        let register = |name: &str| {
            catalog
                .register_stream(StreamInfo {
                    name: name.into(),
                    kind: CatalogKind::Derived,
                    layer: continuum_core::DomainLayer::Layer1Raw,
                    format: continuum_core::PayloadFormat::Text,
                    shape_name: name.into(),
                    shape_version: 1,
                    rate_hz: None,
                    anchor: None,
                })
                .unwrap()
        };
        let face = register("face.live");
        let cam = register("cam");
        let long = format!("routine trim {}", "x".repeat(90));
        catalog
            .start_purge(
                PurgeSelector::Range { span: Interval::new(0, 10).unwrap(), streams: Some(vec![face, cam]) },
                &long,
                "",
                Uuid([1; 16]),
                1,
            )
            .unwrap();
        catalog
            .start_purge(
                PurgeSelector::Range { span: Interval::new(0, 10).unwrap(), streams: None },
                "everything",
                "",
                Uuid([2; 16]),
                1,
            )
            .unwrap();
        catalog.start_purge(PurgeSelector::Entity(Uuid([9; 16])), "forget\nline", "", Uuid([3; 16]), 1).unwrap();
        catalog
            .start_purge(PurgeSelector::Ids(vec![Uuid([4; 16]), Uuid([5; 16])]), "two rows", "", Uuid([6; 16]), 1)
            .unwrap();
        let range = purge_scope(&catalog, &catalog.purge_record(Uuid([1; 16])).unwrap().unwrap());
        assert_eq!(range.kind, "range");
        assert_eq!(range.streams, ["face.live", "cam"]);
        assert!(range.reason.starts_with("routine trim "));
        assert!(range.reason.ends_with("..."));
        assert_eq!(range.reason.chars().count(), 80);
        assert!(!range.reason.contains(&"x".repeat(90)));
        assert_eq!(range.ids, 0);
        let all = purge_scope(&catalog, &catalog.purge_record(Uuid([2; 16])).unwrap().unwrap());
        assert_eq!(all.streams, ["all"]);
        let entity = purge_scope(&catalog, &catalog.purge_record(Uuid([3; 16])).unwrap().unwrap());
        assert_eq!(entity.kind, "entity");
        assert!(entity.streams.is_empty());
        assert_eq!(entity.reason, "forget line");
        let ids = purge_scope(&catalog, &catalog.purge_record(Uuid([6; 16])).unwrap().unwrap());
        assert_eq!(ids.kind, "ids");
        assert_eq!(ids.ids, 2);
    }

    #[test]
    fn a_purge_log_names_its_stream_and_reason() {
        let temp = tempfile::tempdir().unwrap();
        let catalog = continuum_catalog::RedbCatalog::open(temp.path().join("scope.redb")).unwrap();
        let stream = catalog
            .register_stream(StreamInfo {
                name: "synthetic.live".into(),
                kind: CatalogKind::Derived,
                layer: continuum_core::DomainLayer::Layer1Raw,
                format: continuum_core::PayloadFormat::Text,
                shape_name: "synthetic.live".into(),
                shape_version: 1,
                rate_hz: None,
                anchor: None,
            })
            .unwrap();
        let payload = Uuid([7; 16]);
        catalog
            .insert(vec![NewPayload {
                header: HeaderBuf::build(&continuum_core::HeaderSpec::raw(payload, stream, 0, 0)).unwrap(),
                location: None,
                coverage: None,
            }])
            .unwrap();
        let health = Arc::new(writer_health::WriterHealth::default());
        let active = Arc::new(AtomicUsize::new(0));
        let (events, _received) = mpsc::unbounded_channel();
        let mut job = purge_job(
            None,
            Some((
                protocol::PurgeAsk::Range {
                    span: Interval::new(0, 1).unwrap(),
                    streams: vec!["synthetic.live".into()],
                },
                format!("routine trim {}", "y".repeat(90)),
                String::new(),
            )),
            Uuid([8; 16]),
            events,
            &active,
            &health,
            WriterConfig::default(),
        );
        let text = captured_logs(|| {
            assert!(job(&catalog, 1), "starting a purge is not the completion");
            let during = health
                .snapshot(Instant::now(), 0)
                .oldest_streams
                .iter()
                .map(|(name, _)| name.clone())
                .collect::<Vec<_>>();
            assert_eq!(during, ["synthetic.live"], "the unfinished trim shows on the health summary");
            let mut steps = 0;
            loop {
                steps += 1;
                let more = job(&catalog, 1);
                assert!(steps < 100, "the one-row trim must finish");
                if !more {
                    break;
                }
            }
        });
        let started = text.lines().find(|line| line.contains("catalog purge started")).expect("start line");
        let finished = text.lines().find(|line| line.contains("catalog purge finished")).expect("completion line");
        for line in [started, finished] {
            assert!(line.contains("scope_kind=\"range\"") || line.contains("scope_kind=range"), "{line}");
            assert!(line.contains("synthetic.live"), "{line}");
            assert!(line.contains("routine trim "), "{line}");
            assert!(line.contains("..."), "{line}");
            assert!(!line.contains(&"y".repeat(90)), "{line}");
        }
        assert!(finished.contains("removed=1"), "{finished}");
        assert!(
            health.snapshot(Instant::now(), 0).oldest_streams.is_empty(),
            "a finished purge leaves the health summary"
        );
    }

    #[test]
    fn purge_health_counts_catalog_steps_and_finishes_before_keeper_completion() {
        use continuum_core::{DomainLayer, HeaderSpec, PayloadFormat};

        let catalog = continuum_catalog::MemoryCatalog::new();
        let stream = catalog
            .register_stream(StreamInfo {
                name: "live".into(),
                kind: CatalogKind::Derived,
                layer: DomainLayer::Layer1Raw,
                format: PayloadFormat::Text,
                shape_name: "live".into(),
                shape_version: 1,
                rate_hz: None,
                anchor: None,
            })
            .unwrap();
        let payload = Uuid([1; 16]);
        catalog
            .insert(vec![NewPayload {
                header: HeaderBuf::build(&HeaderSpec::raw(payload, stream, 10, 10)).unwrap(),
                location: Some(StorageLocator::Inline { bytes: b"{}".to_vec() }),
                coverage: None,
            }])
            .unwrap();
        let health = Arc::new(writer_health::WriterHealth::default());
        let active = Arc::new(AtomicUsize::new(0));
        let (events, _received) = mpsc::unbounded_channel();
        let id = Uuid([2; 16]);
        let mut job = purge_job(
            None,
            Some((
                protocol::PurgeAsk::Range { span: Interval::new(0, 20).unwrap(), streams: vec!["live".into()] },
                "test trim".into(),
                String::new(),
            )),
            id,
            events.clone(),
            &active,
            &health,
            WriterConfig::default(),
        );
        assert!(job(&catalog, 1), "starting a purge is not a catalog deletion step");
        assert_eq!(health.snapshot(Instant::now(), 0).samples[0].steps, 0);
        let mut done = false;
        for _ in 0..32 {
            if !job(&catalog, 1) {
                done = true;
                break;
            }
            assert!(health.snapshot(Instant::now(), 0).samples[0].steps > 0);
        }
        assert!(done, "the single-row purge finishes within a bounded number of steps");
        assert!(catalog.get(payload).unwrap().is_none());
        let record = catalog.purge_record(id).unwrap().unwrap();
        assert!(record.report.is_some());
        assert!(record.completed_ns.is_none(), "keeper completion is a separate phase");
        assert_eq!(health.snapshot(Instant::now(), 0).purges, 0);
        drop(job);
        assert_eq!(active.load(Ordering::SeqCst), 0);
        // Restarting while keeper acknowledgement is pending must not age/warn catalog work as unfinished.
        let mut resumed = purge_job(None, None, id, events, &active, &health, WriterConfig::default());
        assert!(!resumed(&catalog, 1));
        let finished = health.snapshot(Instant::now() + Duration::from_secs(121), 0);
        assert_eq!(finished.purges, 0);
        assert!(finished.warnings.is_empty());
    }

    #[test]
    fn a_storage_error_fails_the_whole_batch_once_instead_of_bisecting() {
        use continuum_core::{DomainLayer, HeaderSpec, PayloadFormat};
        let c = Counting::default();
        let info = StreamInfo {
            name: "s".into(),
            kind: CatalogKind::Sensor,
            layer: DomainLayer::Layer1Raw,
            format: PayloadFormat::RawBytes,
            shape_name: String::new(),
            shape_version: 0,
            rate_hz: None,
            anchor: None,
        };
        let s = c.register_stream(info).unwrap();
        c.disk_full.store(true, std::sync::atomic::Ordering::SeqCst);
        let batch: Vec<NewPayload> = (0..256u16)
            .map(|n| {
                let mut id = [0u8; 16];
                id[6] = 0x70;
                id[14..].copy_from_slice(&n.to_be_bytes());
                let header = HeaderBuf::build(&HeaderSpec::raw(Uuid(id), s, u64::from(n), u64::from(n))).unwrap();
                NewPayload { header, location: None, coverage: None }
            })
            .collect();
        let results = insert_bisecting(&c, batch);
        assert_eq!(results.len(), 256);
        assert!(
            results
                .iter()
                .all(|r| r.as_ref().is_err_and(|e| e.0 == ErrorCode::Internal && e.1.contains("the catalog failed")))
        );
        assert_eq!(c.inserts.load(std::sync::atomic::Ordering::SeqCst), 1, "one transaction, not 511");
    }

    #[test]
    fn outputs_stored_before_a_failed_coverage_write_still_reach_the_index() {
        let c = Counting::default();
        let info = StreamInfo {
            name: "pose".into(),
            kind: CatalogKind::Derived,
            layer: continuum_core::DomainLayer::Layer1Raw,
            format: continuum_core::PayloadFormat::RawBytes,
            shape_name: String::new(),
            shape_version: 0,
            rate_hz: None,
            anchor: None,
        };
        let s = c.register_stream(info).unwrap();
        c.coverage_fails.store(true, std::sync::atomic::Ordering::SeqCst);
        let mut id = [0u8; 16];
        id[6] = 0x70;
        id[15] = 1;
        let header = HeaderBuf::build(&continuum_core::HeaderSpec::raw(Uuid(id), s, 100, 110)).unwrap();
        let location = StorageLocator::Chunk { tier: StorageTier::Tier1Warm, uri: "c".into(), offset: 0, length: 1 };
        let commit = protocol::Commit {
            lease: 1,
            outputs: vec![(header, Some(location))],
            coverage: Some((s, continuum_core::StageId(1), Interval::new(0, 1_000).unwrap())),
            cost: continuum_sched::ComputeCost {
                wall_time_us: 0,
                queue_wait_us: 0,
                ram_bytes: 0,
                vram_bytes: 0,
                device: String::new(),
                items: 0,
            },
        };
        let (tx, mut rx) = mpsc::unbounded_channel();
        commit_outputs(&c, SessionId::nil(), 7, commit, &tx);
        let Ok(Event::Committed { copies, result, .. }) = rx.try_recv() else { panic!("a Committed event") };
        assert!(result.is_err(), "the coverage write failed, so the commit is answered with an error");
        assert_eq!(copies.len(), 1, "but the output was stored, so the index still gets its copy");
    }

    #[test]
    fn a_sync_is_answered_after_the_positions_queued_ahead_of_it_are_written() {
        use continuum_catalog::MemoryCatalog;
        let c = MemoryCatalog::new();
        let (events, mut rx) = mpsc::unbounded_channel();
        let (reply, _answer) = oneshot::channel();
        let cursor = |position| Urgent::Cursor { name: "tier".into(), position };
        run_urgent(&c, vec![cursor(7), Urgent::Sync(reply), cursor(9)], &events, &AtomicUsize::new(0));
        let mut order = Vec::new();
        while let Ok(e) = rx.try_recv() {
            match e {
                Event::CursorsWritten { saved, failed } => {
                    assert!(failed.is_empty());
                    order.push(format!("wrote {saved:?}"));
                }
                Event::Synced(_) => order.push("synced".to_owned()),
                _ => {}
            }
        }
        assert_eq!(order, [r#"wrote [("tier", 7)]"#, "synced", r#"wrote [("tier", 9)]"#]);
        assert_eq!(c.cursor("tier").unwrap(), Some(9u64.to_le_bytes().to_vec()));
    }

    #[test]
    fn a_tick_commits_all_its_ingests_in_one_insert_and_a_sync_flushes_first() {
        use continuum_core::{DomainLayer, HeaderSpec, PayloadFormat};
        let c = Counting::default();
        let info = StreamInfo {
            name: "s".into(),
            kind: CatalogKind::Sensor,
            layer: DomainLayer::Layer1Raw,
            format: PayloadFormat::RawBytes,
            shape_name: String::new(),
            shape_version: 0,
            rate_hz: None,
            anchor: None,
        };
        let s = c.register_stream(info.clone()).unwrap();
        let ingest = |n: u16| {
            let mut id = [0u8; 16];
            id[6] = 0x70;
            id[14..].copy_from_slice(&n.to_be_bytes());
            let header = HeaderBuf::build(&HeaderSpec::raw(Uuid(id), s, u64::from(n), u64::from(n))).unwrap();
            Urgent::Ingest {
                session: SessionId::nil(),
                request_id: u64::from(n),
                header,
                locator: None,
                dependencies: Vec::new(),
            }
        };
        let (events, mut rx) = mpsc::unbounded_channel();
        // 1000 ingests with a registration in the middle: one insert.
        let mut jobs: Vec<Urgent> = (0..500).map(ingest).collect();
        jobs.push(Urgent::Register {
            session: SessionId::nil(),
            request_id: 0,
            info: StreamInfo { name: "t".into(), ..info },
            ring: None,
        });
        jobs.extend((500..1000).map(ingest));
        run_urgent(&c, jobs, &events, &AtomicUsize::new(0));
        assert_eq!(c.inserts.load(std::sync::atomic::Ordering::SeqCst), 1);
        // A sync between ingests commits the ones before it first: two inserts.
        let (reply, _answer) = oneshot::channel();
        let jobs = vec![ingest(1000), ingest(1001), Urgent::Sync(reply), ingest(1002)];
        run_urgent(&c, jobs, &events, &AtomicUsize::new(0));
        assert_eq!(c.inserts.load(std::sync::atomic::Ordering::SeqCst), 3);
        let mut ingested = 0;
        while let Ok(e) = rx.try_recv() {
            if let Event::Ingested(done) = e {
                assert!(done.iter().all(|d| d.result.is_ok()));
                ingested += done.len();
            }
        }
        assert_eq!(ingested, 1003);
    }

    #[test]
    fn a_bad_payload_in_a_batch_gets_its_own_answer_and_its_neighbors_are_indexed() {
        use continuum_catalog::MemoryCatalog;
        use continuum_core::{DomainLayer, HeaderSpec, PayloadFormat};
        let c = MemoryCatalog::new();
        let info = StreamInfo {
            name: "s".into(),
            kind: CatalogKind::Sensor,
            layer: DomainLayer::Layer1Raw,
            format: PayloadFormat::RawBytes,
            shape_name: String::new(),
            shape_version: 0,
            rate_hz: None,
            anchor: None,
        };
        let s = c.register_stream(info).unwrap();
        let p = |n: u8, t: u64| {
            let mut id = [0u8; 16];
            id[6] = 0x70;
            id[15] = n;
            NewPayload {
                header: HeaderBuf::build(&HeaderSpec::raw(Uuid(id), s, t, t)).unwrap(),
                location: None,
                coverage: None,
            }
        };
        let first = c.insert(vec![p(9, 900)]).unwrap()[0].seq;
        // The fourth reuses an indexed id with a different header (a conflict); the fifth repeats an indexed payload
        // byte for byte (idempotent: its existing sequence, flagged as a repeat); the rest are new.
        let batch = vec![p(1, 10), p(2, 20), p(3, 30), p(9, 901), p(9, 900), p(5, 50), p(6, 60)];
        let results = insert_bisecting(&c, batch);
        assert_eq!(results.iter().map(Result::is_ok).collect::<Vec<_>>(), [true, true, true, false, true, true, true]);
        assert_eq!(results[4], Ok((first, true)), "the repeat answers with the sequence it already has");
        assert!(results.iter().flatten().filter(|(_, repeat)| !repeat).all(|(seq, _)| *seq > first));
        for n in [1u8, 2, 3, 5, 6] {
            let mut id = [0u8; 16];
            id[6] = 0x70;
            id[15] = n;
            assert!(c.get(Uuid(id)).unwrap().is_some(), "payload {n} was indexed");
        }
    }

    #[test]
    fn the_budget_halves_after_an_overrun_doubles_after_a_quick_step_and_stays_in_bounds() {
        let c = config();
        assert_eq!(next_budget(256, Duration::from_millis(9), &c), 128);
        assert_eq!(next_budget(256, Duration::from_millis(1), &c), 512);
        assert_eq!(next_budget(256, Duration::from_millis(4), &c), 256);
        assert_eq!(next_budget(1, Duration::from_millis(50), &c), 1, "never below 1");
        assert_eq!(next_budget(4096, Duration::from_micros(1), &c), 4096, "never above the max");
    }

    #[test]
    fn a_job_that_costs_a_fixed_time_per_row_settles_near_the_target() {
        // 20 us per row: the target of 5 ms is 250 rows, so the budget should settle between 125 and 500.
        let c = WriterConfig { initial_budget: 4096, ..config() };
        let mut budget = c.initial_budget;
        for _ in 0..20 {
            let took = Duration::from_micros(20) * u32::try_from(budget).unwrap();
            budget = next_budget(budget, took, &c);
        }
        assert!((125..=500).contains(&budget), "settled at {budget}");
    }
}
