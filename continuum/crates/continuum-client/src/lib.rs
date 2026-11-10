//! The Rust client SDK for Continuum Broker (docs/CONTINUUM.md, "SDKs").
//!
//! [`Client::connect`] opens the broker's Unix socket, sends `Hello`, and reads `Welcome` together with the arena
//! file descriptors passed alongside it (SCM_RIGHTS), mapping each arena once. After that, one task reads the socket
//! and one writes it: a request gets a fresh request id and waits for the reply that echoes it, and anything the
//! broker sends on its own (events, dispatched jobs, preemptions) goes to [`Client::next_event`]. A third task sends
//! a heartbeat every [`ClientOptions::heartbeat_interval`], so the broker doesn't close a quiet session.
//!
//! The client needs a tokio runtime: [`Client::connect`] spawns its tasks on the current one. They stop when the
//! connection closes or the [`Client`] is dropped.
//!
//! So far it covers connecting, the session (`Heartbeat`, `Goodbye`, `Error`), `RegisterStream`, and ingest (a
//! one-way `IngestNotice`, whose rejections arrive as events). The query,
//! promise, lease, and worker flows come as the broker wires them in.

pub mod aging;
mod applicability;
pub mod messages;
pub mod promise;
pub mod query;
pub mod subscribe;
pub mod worker;
pub mod worker_state;

use std::collections::HashMap;
use std::io;
use std::os::fd::{AsFd, OwnedFd};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use continuum_core::schema::continuum::{self as fb, ControlPayload, ErrorCode};
use continuum_core::{SCHEMA_MAJOR, StorageLocator, Uuid};
use continuum_shm::Arena;
use continuum_wire::{FrameError, MAX_FDS, read_envelope, read_frame_with_fds, write_frame};
use tokio::net::UnixStream;
use tokio::net::unix::{OwnedReadHalf, OwnedWriteHalf};
use tokio::sync::{mpsc, oneshot};

/// The schema minor that stores an ingest's dependencies.
pub const DEPENDENCIES_SCHEMA_MINOR: u16 = 31;

/// A payload's dependency on a stretch of a stream ([`Client::ingest_dependent`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dependency {
    /// The stream depended on.
    pub stream_id: u32,
    /// The stretch's start, UTC nanoseconds.
    pub start_ns: u64,
    /// Its end, exclusive.
    pub end_ns: u64,
    /// The stream's invalidation revision, read before the producer checked what it used.
    pub revision: u64,
}

pub use aging::{
    AGING_SCHEMA_MINOR, CATALOG_STATS_SCHEMA_MINOR, CatalogStats, ChunkInfo, FrameSource, MAX_RESOLVE_IDS,
    RESOLVE_FRAMES_SCHEMA_MINOR, ResolvedFrame, TableStorage,
};
pub use applicability::{RECORDING_SCHEMA_MINOR, RecordingCursor, RecordingInput, RecordingPage};
pub use messages::{ArenaInfo, SizeClass, StreamLease, StreamRegistered, StreamSpec, StreamStorage, Telemetry};
pub use promise::{Fulfilment, PlanOption, PlanStep, Promise, PromiseAccepted, PromiseQuote};
pub use query::{QueryResult, QuerySpec, RingTail};
pub use subscribe::{
    Start, SubEvent, SubscribeSpec, Subscription, SubscriptionHandle, SubscriptionOverflow, length_prefixed,
};
pub use worker::{
    AcceptedCapability, BrokerEvent, CapabilitySpec, ChunkCommitted, ChunkFrame, CommitError, Cost, CoverageDone,
    DispatchJob, FrameProvenance, Item, JobLease, NewChunk, OutputSlot, PROVENANCE_VERSION, PurgedSpan, Quote,
    QuoteSpec, Renewer, Requirement, WorkLease, WorkerRegistered, WorkerSpec, stage_id,
};
pub use worker_state::{WORKER_STATE_SCHEMA_MINOR, WorkerState};

/// What a [`Client::purge`] removes: one of these, as the broker's `Purge` takes exactly one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PurgeTarget {
    /// Everything over `[range.0, range.1)` on these streams (every stream when empty).
    Range {
        /// The time, UTC nanoseconds, half open.
        range: (u64, u64),
        /// The streams, by name.
        streams: Vec<String>,
    },
    /// Every payload about this entity.
    Entity(Uuid),
    /// These payloads (or absorbed frames) and everything derived from them.
    Ids(Vec<Uuid>),
    /// These payloads only, committed by `committed_before_ns`: an age-out ([`Client::age_out`]).
    AgeOut {
        /// The payloads.
        ids: Vec<Uuid>,
        /// The cutoff, UTC nanoseconds, by the commit clock.
        committed_before_ns: u64,
        /// Aged-out spans on one stream at most this far apart are recorded as one run (minor 29; an older broker
        /// ignores it and keeps each payload's span). 0 keeps each payload's span.
        aged_gap_ns: u64,
    },
}

/// The kind of a purge, as its summary says.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PurgeKind {
    /// A time range.
    Range,
    /// An entity.
    Entity,
    /// Ids.
    Ids,
    /// An age-out.
    AgeOut,
}

/// Minimum broker protocol version for opaque forgotten-hash enumeration.
pub const LIST_FORGOTTEN_SCHEMA_MINOR: u16 = 34;
/// The schema minor that commits a chunk in parts (`CommitChunk.first_frame`).
pub const CHUNK_PARTS_SCHEMA_MINOR: u16 = 35;

/// One read-only page of opaque forgotten-name hashes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForgottenPage {
    /// Strictly increasing 32-byte hashes.
    pub hashes: Vec<[u8; 32]>,
    /// Exclusive continuation cursor, or None when this page exhausted its snapshot.
    pub next: Option<[u8; 32]>,
}

/// A purge's summary from [`Client::list_purges`] (the record kept for good). It never names a person: an entity
/// purge's summary doesn't say which, and a purge by ids keeps only how many.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PurgeSummary {
    /// The purge.
    pub purge_id: Uuid,
    /// When it started, UTC nanoseconds.
    pub at_ns: u64,
    /// Its kind.
    pub kind: PurgeKind,
    /// The range, for a range purge.
    pub range: Option<(u64, u64)>,
    /// The streams a range purge named; empty for every stream.
    pub stream_ids: Vec<u32>,
    /// How many ids a purge by ids asked for.
    pub id_count: u32,
    /// Why.
    pub reason: String,
    /// The caller's JSON detail.
    pub detail: String,
    /// Payloads removed, once its steps have run (0 while they run).
    pub removed: u64,
    /// Its steps have run.
    pub finished: bool,
    /// When every copy was gone (every keeper acknowledged), UTC nanoseconds; `None` while notices are outstanding.
    pub completed_ns: Option<u64>,
}

impl PurgeSummary {
    fn from_fb(s: &fb::PurgeSummary<'_>) -> Self {
        Self {
            purge_id: Uuid(s.purge_id().map_or([0; 16], |u| u.0)),
            at_ns: s.at_ns(),
            kind: match s.kind() {
                fb::PurgeKind::Entity => PurgeKind::Entity,
                fb::PurgeKind::Ids => PurgeKind::Ids,
                fb::PurgeKind::AgeOut => PurgeKind::AgeOut,
                _ => PurgeKind::Range,
            },
            range: s.range().map(|r| (r.start_ns(), r.end_ns())),
            stream_ids: s.stream_ids().map(|v| v.iter().collect()).unwrap_or_default(),
            id_count: s.id_count(),
            reason: s.reason().unwrap_or("").to_owned(),
            detail: s.detail().unwrap_or("").to_owned(),
            removed: s.removed(),
            finished: s.finished(),
            completed_ns: (s.completed_ns() > 0).then(|| s.completed_ns()),
        }
    }
}

/// A purge's full record from [`Client::get_purge`]: its summary and its receipt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PurgeRecord {
    /// Its summary.
    pub summary: PurgeSummary,
    /// What it removed, per stream and tier.
    pub removed: Vec<(continuum_core::StreamId, continuum_core::StorageTier, u64)>,
    /// Frames cut out of chunks.
    pub removed_frames: u64,
    /// Chunks it rewrote without some of their frames.
    pub partial_chunks: u64,
}

/// What a [`Client::purge`] or [`Client::purge_range`] removed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PurgeDone {
    /// The purge's record, for its notices and acknowledgements.
    pub purge_id: Uuid,
    /// How many payloads it removed, per stream and tier.
    pub removed: Vec<(continuum_core::StreamId, continuum_core::StorageTier, u64)>,
}

/// How a client connects and keeps its session.
#[derive(Clone, Debug)]
pub struct ClientOptions {
    /// The name the broker logs the session under.
    pub client_name: String,
    /// How often to send a heartbeat. Keep it well under the broker's heartbeat timeout.
    pub heartbeat_interval: Duration,
    /// How long a request waits for its reply (and `connect` for `Welcome`).
    pub request_timeout: Duration,
    /// What to do when a subscription fills its bounded event queue. Defaults to
    /// [`SubscriptionOverflow::Close`]; a dedicated reader-only client may choose
    /// [`SubscriptionOverflow::Backpressure`], which also makes its automatic heartbeats one way.
    pub subscription_overflow: SubscriptionOverflow,
}

impl ClientOptions {
    /// Options with the defaults: a heartbeat every 5 s and a 10 s request timeout.
    pub fn new(client_name: impl Into<String>) -> Self {
        Self {
            client_name: client_name.into(),
            heartbeat_interval: Duration::from_secs(5),
            request_timeout: Duration::from_secs(10),
            subscription_overflow: SubscriptionOverflow::Close,
        }
    }
}

/// One registered stream, from [`Client::list_streams`].
#[derive(Clone, Debug, PartialEq)]
pub struct StreamDesc {
    /// The broker's id for it.
    pub id: continuum_core::StreamId,
    /// Its registered name.
    pub name: String,
    /// Sensor or derived.
    pub kind: fb::StreamKind,
    /// The domain layer of its payloads.
    pub layer: fb::DomainLayer,
    /// Its payloads' format.
    pub format: fb::PayloadFormat,
    /// Its shape name (a raw format like "NV12", or a named shape).
    pub shape_name: String,
    /// The shape's version.
    pub shape_version: u32,
    /// Its nominal rate, when it has one.
    pub rate_hz: Option<f64>,
    /// A producer session is connected for it now.
    pub live: bool,
    /// Monotone deletion revision, or None when unsupported by the broker.
    pub invalidation_revision: Option<u64>,
    /// The invalidation revision as the stream's last forget left it (minor 33): what changes only when something is
    /// forgotten, not aged. None from an older broker.
    pub forget_revision: Option<u64>,
    /// The time its payloads reach over, UTC nanoseconds `[start, end)`, from the start of its earliest to the end
    /// of the one that starts last (minor 36). None for a stream that holds nothing, and from an older broker.
    pub extent: Option<(u64, u64)>,
}

/// What went wrong.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The socket failed.
    #[error(transparent)]
    Io(#[from] io::Error),
    /// A frame couldn't be read.
    #[error(transparent)]
    Frame(#[from] FrameError),
    /// The broker answered with an `Error`.
    #[error("the broker refused the request ({code:?}): {message}")]
    Broker {
        /// The broker's error code.
        code: ErrorCode,
        /// Its message.
        message: String,
    },
    /// The broker sent something the protocol doesn't allow here.
    #[error("protocol error: {0}")]
    Protocol(String),
    /// An arena from `Welcome` couldn't be mapped, or didn't match its description.
    #[error("arena {name}: {reason}")]
    Arena {
        /// The arena's name.
        name: String,
        /// What was wrong.
        reason: String,
    },
    /// No reply within the request timeout.
    #[error("no reply within {0:?}")]
    Timeout(Duration),
    /// A committed promise failed (`PromiseFailed`): nothing more comes for it.
    #[error("the promise failed after {delivered} items: {reason}")]
    PromiseFailed {
        /// Why.
        reason: String,
        /// How many items it delivered before it failed.
        delivered: u32,
    },
    /// The connection is closed.
    #[error("the connection is closed")]
    Closed,
}

impl Error {
    /// The broker's error code, if the broker refused the request.
    pub fn broker_code(&self) -> Option<ErrorCode> {
        match self {
            Self::Broker { code, .. } => Some(*code),
            _ => None,
        }
    }
}

/// A message from the broker: its envelope's bytes, and any file descriptors that came with it (an arena announced
/// after `Welcome`, say).
#[derive(Debug)]
pub struct Message {
    bytes: Vec<u8>,
    fds: Vec<OwnedFd>,
}

impl Message {
    fn new(bytes: Vec<u8>, fds: Vec<OwnedFd>) -> Self {
        Self { bytes, fds }
    }

    /// A message from an envelope's bytes (as `as_bytes` gives them), verified.
    ///
    /// # Errors
    ///
    /// The verifier's error if the bytes aren't a `ControlEnvelope`.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, flatbuffers::InvalidFlatbuffer> {
        read_envelope(&bytes)?;
        Ok(Self::new(bytes, Vec::new()))
    }

    /// The envelope. (The bytes were verified when they arrived.)
    pub fn envelope(&self) -> fb::ControlEnvelope<'_> {
        read_envelope(&self.bytes).expect("verified on arrival")
    }

    /// Takes the file descriptors that came with the message, in order (they close when dropped).
    pub fn take_fds(&mut self) -> Vec<OwnedFd> {
        std::mem::take(&mut self.fds)
    }

    /// The payload's type.
    pub fn kind(&self) -> ControlPayload {
        self.envelope().payload_type()
    }

    /// The envelope's raw bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// An `Error` reply as an [`Error::Broker`], or `self` for anything else.
    fn into_result(self) -> Result<Self, Error> {
        match self.envelope().payload_as_error() {
            Some(e) => Err(Error::Broker { code: e.code(), message: e.message().unwrap_or_default().to_owned() }),
            None => Ok(self),
        }
    }
}

/// An arena the broker handed this client, mapped.
#[derive(Clone)]
pub struct MappedArena {
    /// How `Welcome` described it.
    pub info: ArenaInfo,
    /// The mapping.
    pub arena: Arena,
}

impl std::fmt::Debug for MappedArena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MappedArena").field("info", &self.info).finish_non_exhaustive()
    }
}

type Pending = Mutex<HashMap<u64, oneshot::Sender<Result<Message, Error>>>>;

/// A frame for the writer, and who to tell once it's written.
type Outgoing = (Vec<u8>, Option<oneshot::Sender<()>>);

pub(crate) struct Shared {
    pub(crate) writes: mpsc::UnboundedSender<Outgoing>,
    pending: Pending,
    pub(crate) next_id: AtomicU64,
    /// Each live subscription's queue, by its id (StreamEvent and CaughtUp are routed here).
    pub(crate) subscriptions: Mutex<HashMap<u64, subscribe::Route>>,
    pub(crate) next_subscription: AtomicU64,
    pub(crate) subscription_overflow: SubscriptionOverflow,
    /// PurgeNotice pages waiting for their last page, by purge id.
    purges: Mutex<HashMap<Uuid, PurgePages>>,
    /// `Client::query_quoted`'s waiters for the `PromiseQuote` that follows a quoted result, by the query's request id.
    pub(crate) quotes: Mutex<HashMap<u64, oneshot::Sender<Message>>>,
    /// Each held `Promise`'s queue for its `FulfillPromise` parts and `PromiseFailed`, by promise id, with the token
    /// of the handle that holds it (so a handle only ever removes its own route).
    pub(crate) promises: Mutex<HashMap<Uuid, (u64, mpsc::UnboundedSender<Message>)>>,
    pub(crate) next_promise_token: AtomicU64,
    timeout: Duration,
    /// Set by the reader when the connection ends, before it fails the pending requests.
    closed: AtomicBool,
    /// Updated by the reader before a state broadcast reaches any event consumer.
    voice: Mutex<fb::VoiceTurnState>,
    /// The runtime the client was connected on, for tasks started from outside it (a keep-alive from Python or C).
    pub(crate) runtime: tokio::runtime::Handle,
}

impl Shared {
    fn take_pending(&self) -> HashMap<u64, oneshot::Sender<Result<Message, Error>>> {
        std::mem::take(&mut *self.pending.lock().unwrap_or_else(std::sync::PoisonError::into_inner))
    }

    pub(crate) async fn request(&self, build: impl FnOnce(u64) -> Vec<u8>) -> Result<Message, Error> {
        self.request_checked(|id| Ok(build(id))).await
    }

    pub(crate) async fn request_checked(
        &self,
        build: impl FnOnce(u64) -> Result<Vec<u8>, Error>,
    ) -> Result<Message, Error> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let bytes = build(id)?;
        let (tx, rx) = oneshot::channel();
        self.pending.lock().unwrap_or_else(std::sync::PoisonError::into_inner).insert(id, tx);
        // If this future is dropped (a cancelled Python awaitable, a timed-out select), forget the waiter. The
        // request may already be on its way; cancelling abandons only the reply.
        let _forget = Forget { pending: &self.pending, id };
        // Checked after registering: the reader sets `closed` before it drains `pending`, so either it fails this
        // request or this sees the flag. Either way nothing waits out the timeout on a dead connection.
        if self.closed.load(Ordering::SeqCst) || self.writes.send((bytes, None)).is_err() {
            return Err(Error::Closed);
        }
        let reply = match tokio::time::timeout(self.timeout, rx).await {
            Ok(Ok(reply)) => reply,
            Ok(Err(_)) => Err(Error::Closed),
            Err(_) => Err(Error::Timeout(self.timeout)),
        };
        reply?.into_result()
    }
}

/// Removes a request's waiter when the request is done or abandoned.
struct Forget<'a> {
    pending: &'a Pending,
    id: u64,
}

impl Drop for Forget<'_> {
    fn drop(&mut self) {
        self.pending.lock().unwrap_or_else(std::sync::PoisonError::into_inner).remove(&self.id);
    }
}

/// A connection to the broker. Dropping it closes the connection (prefer [`Client::goodbye`]).
pub struct Client {
    shared: Arc<Shared>,
    /// Aborted on drop, so the socket's read half doesn't outlive the client.
    reader: tokio::task::AbortHandle,
    events: tokio::sync::Mutex<mpsc::UnboundedReceiver<Message>>,
    session_id: Uuid,
    broker_schema: (u16, u16),
    arenas: Vec<MappedArena>,
}

impl Drop for Client {
    fn drop(&mut self) {
        self.reader.abort();
    }
}

impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client").field("session_id", &self.session_id).field("arenas", &self.arenas).finish()
    }
}

impl Client {
    /// Connects to the broker at `socket`, says `Hello`, and maps the arenas `Welcome` hands over.
    ///
    /// # Errors
    ///
    /// [`Error::Broker`] if the broker refuses (`IncompatibleSchema` for a schema it doesn't speak), [`Error::Arena`]
    /// if an arena's fd is missing or doesn't match its description, [`Error::Timeout`] if no `Welcome` comes, or a
    /// socket error.
    pub async fn connect(socket: impl AsRef<Path>, options: ClientOptions) -> Result<Self, Error> {
        if options.heartbeat_interval.is_zero() || options.request_timeout.is_zero() {
            return Err(Error::Io(io::Error::new(
                io::ErrorKind::InvalidInput,
                "heartbeat_interval and request_timeout must be more than zero",
            )));
        }
        let mut stream = UnixStream::connect(socket.as_ref()).await?;
        write_frame(&mut stream, &messages::hello(0, &options.client_name, std::process::id())).await?;

        // Welcome must be read with recvmsg before anything else reads the socket, or the kernel drops its fds.
        let mut buf = Vec::new();
        let fds = tokio::time::timeout(options.request_timeout, read_frame_with_fds(&mut stream, &mut buf, MAX_FDS))
            .await
            .map_err(|_| Error::Timeout(options.request_timeout))??;
        let welcome = Message::new(buf, Vec::new());
        read_envelope(welcome.as_bytes()).map_err(|e| Error::Protocol(format!("not a ControlEnvelope: {e}")))?;
        let welcome = welcome.into_result()?;
        let env = welcome.envelope();
        let w = env
            .payload_as_welcome()
            .ok_or_else(|| Error::Protocol(format!("expected Welcome, got {:?}", env.payload_type())))?;
        let schema = w.schema().ok_or_else(|| Error::Protocol("Welcome without a schema version".into()))?;
        let broker_schema = (schema.major(), schema.minor());
        if broker_schema.0 != SCHEMA_MAJOR {
            return Err(Error::Protocol(format!(
                "the broker speaks schema {}.{}, this SDK {SCHEMA_MAJOR}",
                broker_schema.0, broker_schema.1
            )));
        }
        let session_id = w.session_id().map_or(Uuid([0; 16]), Uuid::from);
        let infos: Vec<ArenaInfo> = w.arenas().map(|v| v.iter().map(ArenaInfo::from_fb).collect()).unwrap_or_default();
        let arenas = map_arenas(infos, fds)?;
        let voice = w.voice_turn();

        let (reader, writer) = stream.into_split();
        let (writes, write_rx) = mpsc::unbounded_channel();
        let (event_tx, events) = mpsc::unbounded_channel();
        let shared = Arc::new(Shared {
            writes,
            pending: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
            subscriptions: Mutex::new(HashMap::new()),
            next_subscription: AtomicU64::new(1),
            subscription_overflow: options.subscription_overflow,
            purges: Mutex::new(HashMap::new()),
            quotes: Mutex::new(HashMap::new()),
            promises: Mutex::new(HashMap::new()),
            next_promise_token: AtomicU64::new(1),
            timeout: options.request_timeout,
            closed: AtomicBool::new(false),
            voice: Mutex::new(voice),
            runtime: tokio::runtime::Handle::current(),
        });
        let writer = tokio::spawn(write_loop(writer, write_rx, Arc::downgrade(&shared))).abort_handle();
        let reader = tokio::spawn(read_loop(reader, Arc::downgrade(&shared), event_tx, writer)).abort_handle();
        tokio::spawn(heartbeat_loop(Arc::downgrade(&shared), options.heartbeat_interval));
        Ok(Self { shared, reader, events: tokio::sync::Mutex::new(events), session_id, broker_schema, arenas })
    }

    /// How many requests are waiting for replies (for tests and diagnostics).
    pub fn pending_requests(&self) -> usize {
        self.shared.pending.lock().unwrap_or_else(std::sync::PoisonError::into_inner).len()
    }

    /// The session id the broker assigned.
    pub fn session_id(&self) -> Uuid {
        self.session_id
    }

    /// The broker's schema version, (major, minor).
    pub fn broker_schema(&self) -> (u16, u16) {
        self.broker_schema
    }

    /// Latest voice turn, including Welcome, without consuming the event queue.
    /// Returns None once the reader has observed the broker connection close.
    pub fn voice_turn(&self) -> Option<fb::VoiceTurnState> {
        if self.shared.closed.load(Ordering::SeqCst) {
            return None;
        }
        Some(*self.shared.voice.lock().unwrap_or_else(std::sync::PoisonError::into_inner))
    }

    /// The arenas the broker handed over, mapped, in `Welcome` order.
    pub fn arenas(&self) -> &[MappedArena] {
        &self.arenas
    }

    /// The mapped arena with this name.
    pub fn arena(&self, name: &str) -> Option<&Arena> {
        self.arenas.iter().find(|a| a.info.name == name).map(|a| &a.arena)
    }

    /// Sends a heartbeat (with telemetry, from a worker) and returns the broker's clock, UTC nanoseconds. The client
    /// already sends heartbeats on its own; this is for telemetry, or to check the broker is responsive.
    ///
    /// # Errors
    ///
    /// As any request.
    pub async fn heartbeat(&self, telemetry: Option<&Telemetry>) -> Result<u64, Error> {
        let reply = self.shared.request(|id| messages::heartbeat(id, now_ns(), telemetry)).await?;
        reply.envelope().payload_as_heartbeat().map(|h| h.now_ns()).ok_or_else(|| unexpected("Heartbeat", &reply))
    }

    /// Registers a stream by name and returns its id and, for a capture stream, its standing lease.
    ///
    /// # Errors
    ///
    /// As any request; the broker refuses a bad spec with [`Error::Broker`].
    pub async fn register_stream(&self, spec: &StreamSpec) -> Result<StreamRegistered, Error> {
        let reply = self.shared.request(|id| messages::register_stream(id, spec)).await?;
        reply
            .envelope()
            .payload_as_stream_registered()
            .map(StreamRegistered::from_fb)
            .ok_or_else(|| unexpected("StreamRegistered", &reply))
    }

    /// Registers this session as a worker with its capabilities. Register the output streams first; a capability
    /// naming a stream nobody registered is refused (see `WorkerRegistered::refused_reasons`).
    ///
    /// # Errors
    ///
    /// As any request.
    pub async fn register_worker(&self, spec: &WorkerSpec) -> Result<WorkerRegistered, Error> {
        let reply = self.shared.request(|id| worker::register_worker(id, spec)).await?;
        reply
            .envelope()
            .payload_as_worker_registered()
            .map(WorkerRegistered::from_fb)
            .ok_or_else(|| unexpected("WorkerRegistered", &reply))
    }

    /// Purges a time range (`Purge`), on the named streams or on every stream when `streams` is empty, and waits
    /// for the broker's `PurgeResult`: the purge's id and what it removed per stream and tier. The workers that keep
    /// copies (the tier worker's chunk files) are told after, and the purge is complete once they've acknowledged.
    ///
    /// # Errors
    ///
    /// As any request: BadRequest for an empty range, NotFound for an unknown stream name.
    pub async fn purge_range(&self, range: (u64, u64), streams: &[String], reason: &str) -> Result<PurgeDone, Error> {
        let reply = self.shared.request(|id| messages::purge_range(id, range, streams, reason)).await?;
        let r = reply.envelope().payload_as_purge_result().ok_or_else(|| unexpected("PurgeResult", &reply))?;
        Ok(PurgeDone {
            purge_id: Uuid(r.purge_id().map_or([0; 16], |u| u.0)),
            removed: r
                .removed()
                .map(|v| v.iter().map(|c| (continuum_core::StreamId(c.stream_id()), c.tier(), c.count())).collect())
                .unwrap_or_default(),
        })
    }

    /// Purges a range, an entity, or ids (`Purge`), with `reason` and a JSON `detail` kept in the record for good: the
    /// detail must not name a person (no subject id). Returns when the purge's steps have run.
    ///
    /// # Errors
    ///
    /// As any request: BadRequest for an empty range or no ids, NotFound for an unknown stream name.
    pub async fn purge(&self, target: &PurgeTarget, reason: &str, detail: &str) -> Result<PurgeDone, Error> {
        // An older broker reads an age-out as a plain purge by ids, which takes everything derived from them.
        aging::check_broker(target, self.broker_schema().1)?;
        let reply = self.shared.request(|id| messages::purge(id, target, reason, detail)).await?;
        let r = reply.envelope().payload_as_purge_result().ok_or_else(|| unexpected("PurgeResult", &reply))?;
        Ok(PurgeDone {
            purge_id: Uuid(r.purge_id().map_or([0; 16], |u| u.0)),
            removed: r
                .removed()
                .map(|v| v.iter().map(|c| (continuum_core::StreamId(c.stream_id()), c.tier(), c.count())).collect())
                .unwrap_or_default(),
        })
    }

    /// Purge summaries, oldest first, after the purge `after` (from the start when `None`), at most `limit` (0 for
    /// the broker's default).
    ///
    /// # Errors
    ///
    /// As any request.
    pub async fn list_purges(&self, after: Option<Uuid>, limit: u32) -> Result<Vec<PurgeSummary>, Error> {
        let reply = self.shared.request(|id| messages::list_purges(id, after, limit)).await?;
        let l = reply.envelope().payload_as_purge_list().ok_or_else(|| unexpected("PurgeList", &reply))?;
        Ok(l.purges().into_iter().flatten().map(|s| PurgeSummary::from_fb(&s)).collect())
    }

    /// Records that a person was forgotten: `hash` is the name, hashed by the caller (the broker never
    /// sees the name), kept in the catalog's `FORGOTTEN`, which no purge removes.
    ///
    /// # Errors
    ///
    /// As any request.
    pub async fn remember_forgotten(&self, hash: [u8; 32]) -> Result<(), Error> {
        let reply = self.shared.request(|id| messages::remember_forgotten(id, hash)).await?;
        reply.envelope().payload_as_forgotten_remembered().ok_or_else(|| unexpected("ForgottenRemembered", &reply))?;
        Ok(())
    }

    /// Read a bounded lexicographic page of forgotten hashes without names or observations.
    /// Concurrent additions below the exclusive cursor require a new enumeration.
    ///
    /// # Errors
    /// Protocol error for brokers older than schema minor 34 or limit outside 1..=1024; request errors otherwise.
    pub async fn list_forgotten(&self, after: Option<[u8; 32]>, limit: u32) -> Result<ForgottenPage, Error> {
        if self.broker_schema().1 < LIST_FORGOTTEN_SCHEMA_MINOR || !(1..=1024).contains(&limit) {
            return Err(Error::Protocol("ListForgotten requires schema minor 34 and limit 1..=1024".into()));
        }
        let reply = self.shared.request(|id| messages::list_forgotten(id, after, limit)).await?;
        let page = reply.envelope().payload_as_forgotten_list().ok_or_else(|| unexpected("ForgottenList", &reply))?;
        let hashes = page.hashes().map(|hashes| hashes.iter().map(|hash| hash.0).collect()).unwrap_or_default();
        Ok(ForgottenPage { hashes, next: page.next().map(|hash| hash.0) })
    }

    /// Whether a forgotten name's hash is in `FORGOTTEN`.
    ///
    /// # Errors
    ///
    /// As any request.
    pub async fn is_forgotten(&self, hash: [u8; 32]) -> Result<bool, Error> {
        let reply = self.shared.request(|id| messages::is_forgotten(id, hash)).await?;
        let s = reply.envelope().payload_as_forgotten_status().ok_or_else(|| unexpected("ForgottenStatus", &reply))?;
        Ok(s.forgotten())
    }

    /// Drops a keeper's name (a worker that registered with `keeps_copies` and was retired or renamed), so purges
    /// stop waiting for its acknowledgement. Returns how many purges were waiting on it.
    ///
    /// # Errors
    ///
    /// As any request.
    pub async fn forget_keeper(&self, name: &str) -> Result<u32, Error> {
        let reply = self.shared.request(|id| messages::forget_keeper(id, name)).await?;
        let k = reply.envelope().payload_as_keeper_forgotten().ok_or_else(|| unexpected("KeeperForgotten", &reply))?;
        Ok(k.released())
    }

    /// One purge's record: its summary and what it removed.
    ///
    /// # Errors
    ///
    /// As any request: NotFound for an unknown purge.
    pub async fn get_purge(&self, purge_id: Uuid) -> Result<PurgeRecord, Error> {
        let reply = self.shared.request(|id| messages::get_purge(id, purge_id)).await?;
        let r = reply.envelope().payload_as_purge_record().ok_or_else(|| unexpected("PurgeRecord", &reply))?;
        let summary = r.summary().ok_or_else(|| Error::Protocol("a PurgeRecord without its summary".into()))?;
        Ok(PurgeRecord {
            summary: PurgeSummary::from_fb(&summary),
            removed: r
                .removed()
                .map(|v| v.iter().map(|c| (continuum_core::StreamId(c.stream_id()), c.tier(), c.count())).collect())
                .unwrap_or_default(),
            removed_frames: r.removed_frames(),
            partial_chunks: r.partial_chunks(),
        })
    }

    /// Records a durable Tier 1 copy of an existing derived payload, preserving its header, parents and sequence.
    /// The copy keeper must fsync the file and its directory first; only after this returns may Tier 0 be released.
    /// Exact locator retries succeed without adding another copy.
    ///
    /// # Errors
    /// NotAllowed unless registered with `keeps_copies`; NotFound for missing/purged payloads; BadRequest for a
    /// sensor, absorbed/inline payload or non-warm chunk locator; Conflict for a different existing warm copy.
    pub async fn commit_payload_copy(&self, payload: Uuid, location: &StorageLocator) -> Result<(), Error> {
        let reply = self.shared.request(|id| messages::commit_payload_copy(id, payload, location)).await?;
        let body = reply
            .envelope()
            .payload_as_payload_copy_committed()
            .ok_or_else(|| unexpected("PayloadCopyCommitted", &reply))?;
        if body.payload_id().map(|id| Uuid(id.0)) != Some(payload) {
            return Err(unexpected("PayloadCopyCommitted for requested payload", &reply));
        }
        Ok(())
    }

    /// Commits a Tier 1 chunk the tier worker closed (`CommitChunk`): the catalog compacts the absorbed payloads'
    /// headers into it, and the broker records the copy. A chunk committed already answers `duplicate`, so a retry
    /// after a lost reply is safe.
    ///
    /// # Errors
    ///
    /// As any request: NotAllowed before `register_worker`; BadRequest when the frames don't match the absorbed
    /// payloads, the span, or the tier; Conflict when the chunk is known with other contents.
    pub async fn commit_chunk(&self, chunk: &NewChunk, absorbed: &[Uuid]) -> Result<ChunkCommitted, Error> {
        self.commit_chunk_with_provenance(chunk, absorbed, None).await
    }

    /// [`Client::commit_chunk`], with what capture attested about each frame, one per frame in order (schema minor
    /// 27). A broker below minor 27 ignores it, and the chunk's provenance stays unknown.
    ///
    /// # Errors
    ///
    /// As [`Client::commit_chunk`], and BadRequest when the provenance's count isn't the frames'.
    pub async fn commit_chunk_with_provenance(
        &self,
        chunk: &NewChunk,
        absorbed: &[Uuid],
        provenance: Option<&[FrameProvenance]>,
    ) -> Result<ChunkCommitted, Error> {
        self.commit_chunk_part(chunk, 0, absorbed, provenance).await
    }

    /// Commits a part of a chunk that is still being written (schema minor 35; docs/CONTINUUM.md, "Video chunks"):
    /// `chunk.frames` are the chunk's frames from index `first_frame` on, `absorbed` and `provenance` go with
    /// them, and `chunk`'s header and location are the chunk's own, with its new end and length. A `first_frame` of
    /// 0 commits a new chunk, as [`Client::commit_chunk_with_provenance`] does.
    ///
    /// # Errors
    ///
    /// As [`Client::commit_chunk_with_provenance`]; for a later part also NotFound when the broker doesn't know
    /// the chunk, Conflict when a running purge has hidden it, and BadRequest when `first_frame` isn't the chunk's
    /// frame count or the header isn't the chunk's own grown. A Protocol error for a later part and a broker below
    /// minor 35, which would take it for a new chunk.
    pub async fn commit_chunk_part(
        &self,
        chunk: &NewChunk,
        first_frame: u32,
        absorbed: &[Uuid],
        provenance: Option<&[FrameProvenance]>,
    ) -> Result<ChunkCommitted, Error> {
        if first_frame > 0 && self.broker_schema().1 < CHUNK_PARTS_SCHEMA_MINOR {
            return Err(Error::Protocol("a chunk part requires schema minor 35".into()));
        }
        let reply =
            self.shared.request(|id| messages::commit_chunk(id, chunk, first_frame, absorbed, provenance)).await?;
        reply
            .envelope()
            .payload_as_chunk_committed()
            .map(ChunkCommitted::from_fb)
            .ok_or_else(|| unexpected("ChunkCommitted", &reply))
    }

    /// Records a chunk the tier worker rewrote (`RewriteChunk`): after a purge cut frames from it, or on
    /// recompression. The frames are the same frames in the same places (count, starts, and cut flags).
    ///
    /// # Errors
    ///
    /// As any request: NotAllowed before `register_worker`; NotFound for an unknown chunk; BadRequest when the
    /// frames differ from the chunk's; Conflict when a running purge has hidden it.
    pub async fn rewrite_chunk(
        &self,
        chunk_id: Uuid,
        location: &StorageLocator,
        frames: &[ChunkFrame],
    ) -> Result<(), Error> {
        let reply = self.shared.request(|id| messages::rewrite_chunk(id, chunk_id, location, frames)).await?;
        reply.envelope().payload_as_chunk_rewritten().map(|_| ()).ok_or_else(|| unexpected("ChunkRewritten", &reply))
    }

    /// Asks for a lease to do work this worker starts itself (`QuoteRequest`). A granted lease must end with a commit,
    /// an abort, or a release (dropping it releases it).
    ///
    /// # Errors
    ///
    /// As any request: NotAllowed before `register_worker`, BadRequest for an empty range.
    pub async fn quote(&self, spec: &QuoteSpec) -> Result<Quote, Error> {
        let reply = self.shared.request(|id| worker::quote_request(id, spec)).await?;
        let env = reply.envelope();
        let g = env.payload_as_lease_grant().ok_or_else(|| unexpected("LeaseGrant", &reply))?;
        match g.lease() {
            Some(lease) if g.granted() => Ok(Quote::Granted(WorkLease::from_grant(&self.shared, lease))),
            _ => Ok(Quote::Denied { retry_after_ms: g.retry_after_ms() }),
        }
    }

    /// Sets the voice turn (`SetSystemState`): the broker preempts what the new state outranks, then tells every
    /// session (`SystemStateChanged`). For the voice service. One way: nothing comes back to this call.
    ///
    /// # Errors
    ///
    /// [`Error::Closed`] if the connection is closed.
    pub fn set_system_state(&self, voice_turn: fb::VoiceTurnState) -> Result<(), Error> {
        if self.shared.closed.load(Ordering::SeqCst) {
            return Err(Error::Closed);
        }
        self.shared.writes.send((messages::set_system_state(voice_turn), None)).map_err(|_| Error::Closed)
    }

    /// Takes on a dispatched job: the lease the broker granted with it, to commit or abort like a quoted one.
    /// Dropping it unended gives it back. It takes the lease out of the job, so a second call can't make a second
    /// handle (whose drop would release the lease under the first). Callable from any thread.
    ///
    /// # Errors
    ///
    /// [`Error::Closed`] if the connection is closed; [`Error::Protocol`] if the job came without an output lease
    /// or its lease was already taken.
    pub fn job_lease(&self, job: &mut DispatchJob) -> Result<WorkLease, Error> {
        let lease = job.lease.take().ok_or_else(|| {
            Error::Protocol(format!("job {} has no output lease (none came, or it was taken)", job.job_id))
        })?;
        self.lease_for(lease)
    }

    /// A `WorkLease` for a job's output lease (as `job_lease`, for a wrapper that keeps the job's parts itself).
    /// Make one per lease: dropping either of two would release it under the other.
    ///
    /// # Errors
    ///
    /// [`Error::Closed`] if the connection is closed.
    pub fn lease_for(&self, lease: JobLease) -> Result<WorkLease, Error> {
        if self.shared.closed.load(Ordering::SeqCst) {
            return Err(Error::Closed);
        }
        Ok(WorkLease::from_parts(&self.shared, lease.lease_id, lease.expires_at_ns, lease.slots))
    }

    /// Turns down a dispatched job (`JobRejected`), with why; the broker replans its promises. One way: nothing
    /// comes back.
    ///
    /// # Errors
    ///
    /// [`Error::Closed`] if the connection is closed.
    pub fn reject_job(&self, job_id: Uuid, reason: &str) -> Result<(), Error> {
        self.send_one_way(messages::job_rejected(job_id, reason))
    }

    /// Says this worker has dropped its copies of a `PurgeNotice`'s payloads (`PurgeAck`). One way.
    ///
    /// # Errors
    ///
    /// [`Error::Closed`] if the connection is closed.
    pub fn ack_purge(&self, purge_id: Uuid) -> Result<(), Error> {
        self.send_one_way(messages::purge_ack(purge_id))
    }

    fn send_one_way(&self, bytes: Vec<u8>) -> Result<(), Error> {
        if self.shared.closed.load(Ordering::SeqCst) {
            return Err(Error::Closed);
        }
        self.shared.writes.send((bytes, None)).map_err(|_| Error::Closed)
    }

    /// Tells the broker a payload is written: its header, and where its bytes are (`None` for a payload that is
    /// only its header). One way: the broker doesn't answer a notice it accepts; one it refuses comes back as an
    /// `IngestRejected` event (request id 0, naming the payload and stream) from [`Client::next_event`].
    ///
    /// `Ok` means the notice is queued for the socket, not that the broker has it: a notice queued just before the
    /// connection drops is lost with no rejection, so a producer that reconnects re-sends what it can't confirm (the
    /// catalog's insert is idempotent on the payload id). Notices aren't throttled; the broker closes a session that
    /// stops reading.
    ///
    /// # Errors
    ///
    /// [`Error::Closed`] if the connection is closed.
    pub fn ingest(
        &self,
        header: &continuum_core::HeaderBuf,
        at: Option<&continuum_core::StorageLocator>,
    ) -> Result<(), Error> {
        if self.shared.closed.load(Ordering::SeqCst) {
            return Err(Error::Closed);
        }
        self.shared.writes.send((messages::ingest(header, at), None)).map_err(|_| Error::Closed)
    }

    /// Ingests a payload and waits for its commit: the payload's commit sequence (the position subscribers and
    /// sequence queries see). A repeat with identical header bytes is idempotent and gets the existing sequence (a
    /// frame a chunk absorbed meanwhile, sent again at its start, gets the chunk's). For
    /// a client with no shared memory, `at` can be a [`continuum_core::StorageLocator::Inline`] of up to 64 KiB ([`continuum_core::MAX_INLINE_BYTES`]).
    ///
    /// # Errors
    ///
    /// [`Error::Broker`] with the rejection's code and reason: `Conflict` for a different header under the id,
    /// `BadRequest` for a bad locator, `NotAllowed` for a stream this session doesn't produce, `Unavailable` when the
    /// broker is busy (send it again), `Internal` when the catalog failed; or [`Error::Closed`], or [`Error::Timeout`].
    pub async fn ingest_acked(
        &self,
        header: &continuum_core::HeaderBuf,
        at: Option<&continuum_core::StorageLocator>,
    ) -> Result<u64, Error> {
        self.ingest_dependent(header, at, &[]).await
    }

    /// [`Client::ingest_acked`] with what the payload depends on: stretches of streams, each with the stream's
    /// invalidation revision read before the producer checked what it used. The broker stores them with the payload in
    /// one commit before it answers, and refuses the notice (Conflict) if a revision moved, so a purge in between is
    /// never missed. A later purge removing anything on a stream inside a dependency's range takes the payload too.
    ///
    /// # Errors
    ///
    /// As [`Client::ingest_acked`]; a Protocol error, sent nothing, with dependencies for a broker below
    /// [`DEPENDENCIES_SCHEMA_MINOR`] (it would ACK the payload without them).
    pub async fn ingest_dependent(
        &self,
        header: &continuum_core::HeaderBuf,
        at: Option<&continuum_core::StorageLocator>,
        dependencies: &[Dependency],
    ) -> Result<u64, Error> {
        if !dependencies.is_empty() {
            let minor = self.broker_schema().1;
            if minor < DEPENDENCIES_SCHEMA_MINOR {
                return Err(Error::Protocol(format!(
                    "the broker speaks schema minor {minor}; dependencies need {DEPENDENCIES_SCHEMA_MINOR}"
                )));
            }
        }
        let reply = self.shared.request(|id| messages::ingest_depending(id, header, at, dependencies)).await?;
        if let Some(i) = reply.envelope().payload_as_ingested() {
            return Ok(i.commit_seq());
        }
        match reply.envelope().payload_as_ingest_rejected() {
            Some(r) => Err(Error::Broker { code: r.code(), message: r.reason().unwrap_or("").to_owned() }),
            None => Err(unexpected("Ingested", &reply)),
        }
    }

    /// Every registered stream, in id order, with whether a producer is connected for it now.
    ///
    /// # Errors
    ///
    /// [`Error::Broker`] if the broker refuses, [`Error::Protocol`] for a reply it can't read, and as any request.
    pub async fn list_streams(&self) -> Result<Vec<StreamDesc>, Error> {
        let reply = self.shared.request(messages::list_streams).await?;
        let l = reply.envelope().payload_as_stream_list().ok_or_else(|| unexpected("StreamList", &reply))?;
        Ok(l.streams()
            .into_iter()
            .flatten()
            .map(|s| StreamDesc {
                id: continuum_core::StreamId(s.stream_id()),
                name: s.name().unwrap_or("").to_owned(),
                kind: s.kind(),
                layer: s.layer(),
                format: s.format(),
                shape_name: s.shape_name().unwrap_or("").to_owned(),
                shape_version: s.shape_version(),
                rate_hz: (s.rate_hz() > 0.0).then(|| f64::from(s.rate_hz())),
                live: s.live(),
                invalidation_revision: s.invalidation_revision(),
                forget_revision: s.forget_revision(),
                extent: s.extent_start_ns().zip(s.extent_end_ns()),
            })
            .collect())
    }

    /// The next message the broker sent on its own (request id 0: events, dispatched jobs, preemptions), or `None`
    /// once the connection is closed. A reply that arrives after its request timed out is dropped, not passed here.
    pub async fn next_event(&self) -> Option<Message> {
        self.events.lock().await.recv().await
    }

    /// Says `Goodbye` and closes. The broker drops the session's pins and leases.
    ///
    /// # Errors
    ///
    /// [`Error::Closed`] if the connection was already gone.
    pub async fn goodbye(self, reason: &str) -> Result<(), Error> {
        self.close(reason).await
    }

    /// Says `Goodbye` without giving up the client (for when it's shared, as the Python SDK shares it), and returns
    /// once it's written to the socket (or the request timeout passes). Requests after this fail with
    /// [`Error::Closed`].
    ///
    /// # Errors
    ///
    /// [`Error::Closed`] if the connection was already gone or closed, [`Error::Timeout`] if the write didn't finish.
    pub async fn close(&self, reason: &str) -> Result<(), Error> {
        if self.shared.closed.swap(true, Ordering::SeqCst) {
            return Err(Error::Closed);
        }
        // Closing ends the subscriptions here, rather than when the broker's close reaches a read loop that may be
        // parked on a full subscription queue.
        subscribe::end_all(&self.shared);
        let (written, done) = oneshot::channel();
        self.shared.writes.send((messages::goodbye(0, reason), Some(written))).map_err(|_| Error::Closed)?;
        match tokio::time::timeout(self.shared.timeout, done).await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(_)) => Err(Error::Closed), // the writer stopped first: the socket failed
            Err(_) => Err(Error::Timeout(self.shared.timeout)),
        }
    }
}

pub(crate) fn unexpected(want: &str, got: &Message) -> Error {
    Error::Protocol(format!("expected {want}, got {:?}", got.kind()))
}

/// Maps each arena from its fd, matched by index, and checks it against its description. Every fd is closed when
/// this returns (the mappings keep their own references), on success or failure.
fn map_arenas(infos: Vec<ArenaInfo>, fds: Vec<OwnedFd>) -> Result<Vec<MappedArena>, Error> {
    if infos.len() != fds.len() {
        return Err(Error::Protocol(format!("Welcome lists {} arenas but carried {} fds", infos.len(), fds.len())));
    }
    infos
        .into_iter()
        .zip(&fds)
        .map(|(info, fd)| {
            let bad = |reason: String| Error::Arena { name: info.name.clone(), reason };
            let arena = Arena::from_fd(fd.as_fd(), &info.name).map_err(|e| bad(e.to_string()))?;
            if arena.len() as u64 != info.size_bytes {
                return Err(bad(format!("mapped {} bytes, Welcome says {}", arena.len(), info.size_bytes)));
            }
            if arena.generation() != info.generation {
                return Err(bad(format!("generation {}, Welcome says {}", arena.generation(), info.generation)));
            }
            Ok(MappedArena { info, arena })
        })
        .collect()
}

async fn write_loop(mut writer: OwnedWriteHalf, mut frames: mpsc::UnboundedReceiver<Outgoing>, shared: Weak<Shared>) {
    while let Some((frame, written)) = frames.recv().await {
        if write_frame(&mut writer, &frame).await.is_err() {
            // The socket failed. The read loop may be parked on a full subscription queue and not see it, so end the
            // subscriptions from here (the heartbeat's next write gets here soon after a broker close).
            if let Some(shared) = shared.upgrade() {
                shared.closed.store(true, Ordering::SeqCst);
                subscribe::end_all(&shared);
            }
            break;
        }
        if let Some(written) = written {
            let _ = written.send(());
        }
    }
    // Dropping the write half shuts down the socket's write side, so the broker sees the close.
}

/// A purge's pages so far: its ids, and the spans (which come on the first page).
type PurgePages = (Vec<Uuid>, Vec<worker::PurgedSpan>);

/// Holds a `PurgeNotice` page that has more after it, and on the last page returns one notice with every page's ids
/// (and the first page's spans), so a worker sees each purge once and acknowledges it once. Anything else, and a
/// one-page notice, passes through as it came.
fn gather_purge(shared: &Shared, message: Message) -> Option<Message> {
    let env = message.envelope();
    let Some(p) = env.payload_as_purge_notice() else { return Some(message) };
    let Some(id) = p.purge_id().map(Uuid::from) else { return Some(message) };
    let mut purges = shared.purges.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if p.more() {
        let entry = purges.entry(id).or_default();
        entry.0.extend(worker::ids(p.payload_ids()));
        entry.1.extend(worker::spans_of(&p));
        return None;
    }
    let Some((mut payload_ids, mut spans)) = purges.remove(&id) else { return Some(message) };
    payload_ids.extend(worker::ids(p.payload_ids()));
    spans.extend(worker::spans_of(&p));
    Some(Message::new(messages::purge_notice(id, &payload_ids, &spans), Vec::new()))
}

async fn read_loop(
    mut reader: OwnedReadHalf,
    shared: Weak<Shared>,
    events: mpsc::UnboundedSender<Message>,
    writer: tokio::task::AbortHandle,
) {
    let mut buf = Vec::new();
    loop {
        // Every frame is read with recvmsg, so a message after Welcome that carries fds (a new arena) keeps them.
        let Ok(fds) = read_frame_with_fds(&mut reader, &mut buf, MAX_FDS).await else { break };
        let Some(shared) = shared.upgrade() else { break };
        let Ok(request_id) = read_envelope(&buf).map(|env| env.request_id()) else {
            break; // the stream can't be trusted after a bad frame
        };
        let message = Message::new(std::mem::take(&mut buf), fds);
        if request_id == 0 {
            if let Some(state) = message.envelope().payload_as_system_state_changed() {
                *shared.voice.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = state.voice_turn();
            }
            // The broker's own messages carry request id 0: a subscription's go to it, a paged PurgeNotice is held
            // until its last page, and the rest go to next_event.
            let message = match subscribe::route(&shared, message).await {
                Ok(message) => message,
                Err(id) => {
                    let _ = events.send(Message::new(messages::subscription_overflow(id), Vec::new()));
                    break;
                }
            }
            .and_then(|m| promise::route(&shared, m))
            .and_then(|m| gather_purge(&shared, m));
            if let Some(message) = message {
                let _ = events.send(message);
            }
            continue;
        }
        let waiter = shared.pending.lock().unwrap_or_else(std::sync::PoisonError::into_inner).remove(&request_id);
        if let Some(tx) = waiter {
            let _ = tx.send(Ok(message));
        } else if message.kind() == ControlPayload::PromiseQuote {
            // A quoted Query was answered by its QueryResult; the PromiseQuote that follows carries the same request
            // id: to `query_quoted` when it waits for it, else to the events.
            let waiter = shared.quotes.lock().unwrap_or_else(std::sync::PoisonError::into_inner).remove(&request_id);
            match waiter {
                Some(tx) => {
                    let _ = tx.send(message);
                }
                None => {
                    let _ = events.send(message);
                }
            }
        }
        // Otherwise it's a reply to a request that already timed out; nobody wants it.
    }
    writer.abort(); // Close the socket even when the client and its write sender remain alive.
    if let Some(shared) = shared.upgrade() {
        shared.closed.store(true, Ordering::SeqCst);
        // Ends every subscription's `next`, and every held promise's wait.
        subscribe::end_all(&shared);
        shared.promises.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clear();
        for (_, tx) in shared.take_pending() {
            let _ = tx.send(Err(Error::Closed));
        }
    }
}

async fn heartbeat_loop(shared: Weak<Shared>, every: Duration) {
    let mut tick = tokio::time::interval(every);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    tick.tick().await; // the first tick is immediate; the session was just made
    loop {
        tick.tick().await;
        let Some(shared) = shared.upgrade() else { return };
        if shared.subscription_overflow == SubscriptionOverflow::Backpressure {
            // A Backpressure reader may leave its socket unread for a while, so a numbered heartbeat's reply can sit
            // behind its subscription events. Send the keepalive one way instead: its late reply finds no waiter and
            // is dropped, nothing times out, and the cadence stays steady while the consumer is paused.
            let id = shared.next_id.fetch_add(1, Ordering::Relaxed);
            if shared.closed.load(Ordering::SeqCst)
                || shared.writes.send((messages::heartbeat(id, now_ns(), None), None)).is_err()
            {
                return;
            }
            continue;
        }
        // Under Close, a heartbeat is an ordinary numbered request; an overflow closes the connection before its reply
        // could wait behind the full queue, so this fails Closed and the loop ends.
        if matches!(shared.request(|id| messages::heartbeat(id, now_ns(), None)).await, Err(Error::Closed)) {
            return;
        }
    }
}

/// Nanoseconds since the Unix epoch, UTC.
fn now_ns() -> u64 {
    let d = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)
}
