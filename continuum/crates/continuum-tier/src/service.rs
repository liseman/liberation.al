//! The tier worker program's connection to the broker (docs/CONTINUUM.md, "Storage tiers"): it registers as a
//! worker, follows each of its streams with a durable, pinned subscription of its own, reads each payload's bytes
//! from the broker's Tier 0 arena (mapped from `Welcome`), and runs them through the [`WorkerLoop`]. Chunks, and
//! the parts of a camera's open chunk (docs/CONTINUUM.md, "Video chunks"), go to the broker with `CommitChunk`; once
//! one is committed, its slots are marked `flushed` in the arena and each subscription's saved position moves past
//! everything committed on it.
//!
//! What keeps a crash from losing anything:
//! - A slot is marked `flushed` only once its chunk or part is committed, so a slot the broker may reclaim is
//!   always in a chunk the catalog knows, or was dropped on purpose (an undecodable or quarantined payload, counted
//!   as such): nothing will ever copy that one, so holding its slot would only leak Tier 0.
//! - A subscription's saved position stops before its oldest payload whose commit hasn't happened, so after a crash
//!   the broker sends again everything that isn't safe yet.
//! - At start, [`crate::recover`] finalizes the chunks the last run left open; each is committed when its stream's
//!   first payload comes (only its records past what the catalog holds of it, asked with `GetChunk`), and the
//!   payloads already in it aren't appended again.
//! - Anything the loop can't append (a disk error) ends the run, rather than being passed over, and the service
//!   manager starts it again from the saved positions. So does the broker dropping a subscription or the connection.
//!
//! Derived streams are discovered periodically through ListStreams. Slot-backed derived originals get one
//! passthrough CNK sidecar and CommitPayloadCopy, retaining their original header and provenance. The file and
//! directory are durable before attachment; only the acknowledgement permits flushed/position advancement.
//! Startup reattaches existing sidecars by ID rather than absorbing/reinserting them. Derived purges use the
//! complete notice ID set, since unrelated originals may overlap in time. Raw interval rules stay separate.
//!
//! Each stream's subscription delivery and one-way acknowledgements use its own client with bounded backpressure. Worker
//! registration, purge/job notices, and every request/reply operation use a separate client, so a full subscription
//! queue cannot hold a chunk commit or copy attachment reply behind its own unread events. Any connection
//! ending stops the run; shutdown closes every intake client and the writer, and the next run resumes the same
//! durable stream positions.
//!
//! Ring streams (microphones) arrive as batches: each event names a run of records in the stream's ring
//! (a `RingLocation`), which the worker reads in order from a reader on the ring the subscription described, and
//! appends as one payload. A batch the ring lapped before the worker got to it is a gap, passed over (the broker
//! logs a ring gap when it laps an uncovered batch). Rings have no per-record `flushed` flag: the broker learns a
//! batch is safe from the chunk's `CommitChunk`.
//!
//! One subscription per stream, so a camera the broker doesn't know yet (not plugged in, or capture not started)
//! doesn't keep the others from being archived: it's subscribed to again every [`RESUBSCRIBE`].
//!
//! Purges: the worker registers `keeps_copies`, so a purge isn't complete until it has removed its copies and
//! answered (docs/CONTINUUM.md, "Forgetting"). On a `PurgeNotice` it first closes and commits its open chunks (so no
//! purged payload sits in a file still being written), merges the notice's spans per stream, and queues a purge job
//! (a notice sent again for a queued purge is not queued twice, and purges arriving before the last queued job has
//! begun join it, so they share one pass over the files).
//! [`Service::step`] then goes through a bounded number of chunk files per step (committed and set aside alike), so
//! subscriptions keep draining during a large purge. In each file the frames the catalog cut (those whose slot
//! overlaps a window, the rule [`continuum_catalog::frame_slots`] gives both catalogs) and those whose payload id the
//! notice names are rewritten away, keeping their places. A file wholly inside a window is deleted only when the
//! catalog doesn't know it; a chunk it knows is rewritten and gets `RewriteChunk` with its new frame index and length,
//! sent again for a chunk already cut, in case a crash came between the rewrite and the message. The catalog refusing
//! a rewrite is logged, since the file is what's left. Once every file is done it sends `PurgeAck`; any other failure
//! ends the run without acknowledging, and the broker sends the notice again. Payloads delivered after the notice
//! (they were in flight) that a recent purge covers are not archived.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use continuum_client::{
    ChunkFrame, Client, ClientOptions, Item, NewChunk, Start, SubEvent, SubscribeSpec, SubscriptionHandle,
    SubscriptionOverflow, WorkerSpec,
};
use continuum_core::schema::continuum::ErrorCode;
use continuum_core::{PayloadFormat, SlotHandle, StorageLocator, StreamId, Uuid};
use continuum_shm::Arena;
use serde::Deserialize;
use tokio::sync::mpsc;

use crate::chunk::ChunkFormat;
use crate::flush::{Closed, FlushSettings, StreamShape};
use crate::worker::{Appended, CameraFrameInfo, CommitFailure, Outputs, Payload, WorkerLoop};

/// The name the worker registers under; each stream's position is saved under `tier/<stream>`.
pub const WORKER_NAME: &str = "tier";

/// How often a stream the broker doesn't know yet is subscribed to again.
pub const RESUBSCRIBE: Duration = Duration::from_secs(5);

/// How the program runs.
#[derive(Clone, Debug)]
pub struct Settings {
    /// The broker's socket.
    pub socket: PathBuf,
    /// Where chunk files go.
    pub chunk_dir: PathBuf,
    /// The data root (the state directory that holds `continuum/`): locators of files under it are stored relative
    /// to it ([`continuum_core::chunk_path`]), and relative locators are read against it. `None` (a bare
    /// `--chunks` directory) stores absolute paths and reads relative ones against the working directory.
    pub data_root: Option<PathBuf>,
    /// Configured raw streams to follow, by name. Registered Derived streams are discovered automatically.
    pub streams: Vec<String>,
    /// How chunks are cut and synced.
    pub flush: FlushSettings,
    /// How often the loop ticks when no payload comes.
    pub tick: Duration,
}

/// The streams in a capture device manifest (`config/devices.toml`): `<name>.video` for each camera and
/// `<name>.audio` for each microphone or XVF3800 array.
///
/// # Errors
///
/// When the manifest doesn't parse.
pub fn manifest_streams(text: &str) -> Result<Vec<String>, String> {
    #[derive(Deserialize)]
    struct Manifest {
        #[serde(default)]
        device: Vec<Device>,
    }
    #[derive(Deserialize)]
    struct Device {
        name: String,
        kind: String,
    }
    let m: Manifest = toml::from_str(text).map_err(|e| e.to_string())?;
    Ok(m.device
        .into_iter()
        .filter_map(|d| match d.kind.as_str() {
            "camera" | "rtsp" => Some(format!("{}.video", d.name)),
            "microphone" | "xvf3800" => Some(format!("{}.audio", d.name)),
            _ => None,
        })
        .collect())
}

/// The chunk shape for a stream, from its first payload's header: MJPEG (and other encoded video) as it arrives, raw
/// frames compressed with LZ4, Arrow as it arrives, and a microphone's ring batches (an f32 PCM shape in a ring of
/// `ring_stride` bytes) as FLAC where the samples allow ([`crate::pcm`]).
#[must_use]
pub fn shape_of(item: &Item, ring_stride: Option<usize>) -> StreamShape {
    let h = item.header.header();
    let pcm = ring_stride.and_then(|stride| crate::pcm::PcmLayout::of(item.header.shape_name(), stride));
    let chunk_format = match h.format() {
        PayloadFormat::RawBytes if pcm.is_some() => ChunkFormat::PcmBatch,
        PayloadFormat::RawBytes => ChunkFormat::Lz4,
        PayloadFormat::ArrowRecordBatch => ChunkFormat::ArrowIpc,
        PayloadFormat::EncodedVideo if item.header.shape_name() == "H264AnnexB" => ChunkFormat::H264Au,
        _ => ChunkFormat::Passthrough,
    };
    StreamShape {
        chunk_format,
        layer: h.layer(),
        format: h.format(),
        shape_name: item.header.shape_name().to_owned(),
        shape_version: h.shape_version(),
        pcm,
        raw: raw_layout(item),
    }
}

/// A raw camera frame's layout, from its stream's shape and its capture info (width, height, and stride), or None for
/// anything else.
#[must_use]
pub fn raw_layout(item: &Item) -> Option<crate::raw::RawLayout> {
    let h = item.header.header();
    if h.format() != PayloadFormat::RawBytes {
        return None;
    }
    let c = h.capture()?;
    crate::raw::RawLayout::of(item.header.shape_name(), c.width(), c.height(), c.stride())
}

/// The client's form of a closed chunk.
#[must_use]
pub fn to_new_chunk(closed: &Closed) -> NewChunk {
    let c = closed.to_commit();
    NewChunk {
        header: c.header.clone(),
        location: c.location.clone(),
        frames: c
            .frames
            .iter()
            .map(|f| ChunkFrame { start_ns: f.start_ns, offset: f.offset, length: f.length, cut: f.cut })
            .collect(),
    }
}

/// What a payload's capture fields attest, when its producer gave a capture epoch (minor 27; zero means it didn't
/// say). A frame counts as stamped at arrival when its capture time is exactly its arrival time, which is how
/// capture marks a frame the device gave no usable time for.
fn frame_provenance(
    c: continuum_core::schema::continuum::CaptureInfo<'_>,
) -> Option<continuum_catalog::FrameProvenance> {
    (c.capture_epoch() != 0).then(|| continuum_catalog::FrameProvenance {
        device_seq: c.device_seq(),
        epoch: c.capture_epoch(),
        stamped_at_arrival: c.arrival_ns() != 0 && c.capture_mono_ns() == c.arrival_ns(),
    })
}

/// The client's form of a closed chunk's provenance.
#[must_use]
pub fn to_client_provenance(closed: &Closed) -> Option<Vec<continuum_client::FrameProvenance>> {
    closed.provenance.as_ref().map(|p| {
        p.frames
            .iter()
            .map(|f| continuum_client::FrameProvenance {
                device_seq: f.device_seq,
                epoch: f.epoch,
                stamped_at_arrival: f.stamped_at_arrival,
            })
            .collect()
    })
}

/// Whether a broker refusal can never change: the chunk itself is wrong for the catalog.
#[must_use]
pub fn is_final(e: &continuum_client::Error) -> bool {
    matches!(e.broker_code(), Some(ErrorCode::BadRequest | ErrorCode::Conflict | ErrorCode::NotFound))
}

/// One subscription's positions: what it delivered, what's waiting for a commit, and what it saved.
#[derive(Debug)]
struct Positions {
    handle: SubscriptionHandle,
    /// Positions whose payload isn't in a committed chunk yet: the saved position stays below the smallest.
    open: BTreeSet<u64>,
    /// The highest position delivered.
    seen: u64,
    acked: u64,
    /// Failed saves since the last one that worked: only the first is logged (a closed connection fails every
    /// save until the worker restarts).
    failures: u64,
}

impl Positions {
    fn ack(&mut self) {
        let safe = self.open.first().map_or(self.seen, |p| p.saturating_sub(1));
        if safe > self.acked {
            if let Err(e) = self.handle.ack(safe) {
                self.failures += 1;
                if self.failures == 1 {
                    tracing::warn!("saving a subscription's position failed: {e}; retries are not logged");
                }
                return;
            }
            if self.failures > 1 {
                tracing::info!("saving a subscription's position works again after {} failures", self.failures);
            }
            self.failures = 0;
            self.acked = safe;
        }
    }
}

/// A payload's slot and subscription position, until its chunk is committed.
#[derive(Clone, Copy, Debug)]
struct Pending {
    /// Its Tier 0 slot, to mark flushed; `None` for a ring batch (rings have no flag).
    slot: Option<SlotHandle>,
    sub: usize,
    position: u64,
}

/// The outputs that reach the broker: commits through the client, `flushed` marks in the arena, and the saved
/// positions.
struct BrokerOutputs {
    client: Arc<Client>,
    rt: tokio::runtime::Handle,
    arena: Arena,
    subs: Vec<Option<Positions>>,
    pending: BTreeMap<Uuid, Pending>,
    /// Payloads a recovered chunk's commit made safe before the broker sent them again: settled when they come.
    early: BTreeSet<Uuid>,
    /// [`Settings::data_root`], for the locators it commits.
    data_root: Option<PathBuf>,
}

impl BrokerOutputs {
    /// A payload is safe (in a committed chunk) or gone: mark its slot and release its position.
    fn settle(&mut self, id: Uuid, mark: bool) {
        if let Some(p) = self.pending.remove(&id) {
            if mark && let Some(slot) = p.slot {
                // Stale means the slot was reused already (an expired pin was evicted): nothing to mark.
                let _ = self.arena.mark_flushed(slot);
            }
            if let Some(Some(sub)) = self.subs.get_mut(p.sub) {
                sub.open.remove(&p.position);
            }
        }
    }

    fn ack_all(&mut self) {
        for sub in self.subs.iter_mut().flatten() {
            sub.ack();
        }
    }
}

impl Outputs for BrokerOutputs {
    fn flushed(&mut self, _ids: &[Uuid]) -> Result<(), String> {
        // Durable on disk, but marked only once committed (see the module doc).
        Ok(())
    }

    fn commit(&mut self, closed: &Closed) -> Result<(), CommitFailure> {
        let mut chunk = to_new_chunk(closed);
        chunk.location = stored(self.data_root.as_deref(), chunk.location);
        let provenance = to_client_provenance(closed);
        let commit = self.client.commit_chunk_part(&chunk, closed.first_frame, &closed.absorbed, provenance.as_deref());
        let done = self.rt.block_on(commit).map_err(|e| {
            if is_final(&e) { CommitFailure::Reject(e.to_string()) } else { CommitFailure::Retry(e.to_string()) }
        })?;
        if done.hidden {
            tracing::info!("chunk {} landed under a running purge", done.chunk_id);
        }
        for id in &closed.absorbed {
            if self.pending.contains_key(id) {
                self.settle(*id, true);
            } else {
                // A recovered chunk's payload, not sent again yet.
                self.early.insert(*id);
            }
        }
        self.ack_all();
        Ok(())
    }

    fn rejected(&mut self, closed: &Closed, _aside: &Path) {
        // The broker refused the chunk: its payloads aren't in Tier 1 through it. Their positions are released so
        // the stream isn't held back forever; their slots aren't marked, so the broker records any overwrite as a
        // gap.
        for id in &closed.absorbed {
            self.settle(*id, false);
        }
        self.ack_all();
    }

    fn committed_frames(&mut self, chunk: Uuid) -> Result<Option<usize>, String> {
        match self.rt.block_on(self.client.get_chunk(chunk)) {
            Ok(info) => Ok(Some(info.frames.len())),
            Err(e) if e.broker_code() == Some(ErrorCode::NotFound) => Ok(None),
            Err(e) => Err(e.to_string()),
        }
    }

    fn first_unabsorbed(&mut self, ids: &[Uuid]) -> Result<Option<usize>, String> {
        for (batch, asked) in ids.chunks(continuum_client::MAX_RESOLVE_IDS).enumerate() {
            let found = self.rt.block_on(self.client.resolve_frames(asked)).map_err(|e| e.to_string())?;
            let first = found.iter().position(|f| matches!(f.source, continuum_client::FrameSource::Payload { .. }));
            if let Some(first) = first {
                return Ok(Some(batch * continuum_client::MAX_RESOLVE_IDS + first));
            }
        }
        Ok(None)
    }

    fn settled(&mut self, ids: &[Uuid]) {
        for id in ids {
            if self.pending.contains_key(id) {
                self.settle(*id, true);
            } else {
                self.early.insert(*id);
            }
        }
        self.ack_all();
    }
}

/// Why the program stopped.
#[derive(Debug, thiserror::Error)]
pub enum ServiceError {
    /// The broker refused or went away.
    #[error("broker: {0}")]
    Broker(#[from] continuum_client::Error),
    /// The broker dropped a subscription or the connection.
    #[error("broker: {0}")]
    Lost(String),
    /// The broker described no Tier 0 arena.
    #[error("the broker described no Tier 0 arena")]
    NoArena,
    /// A chunk file problem.
    #[error(transparent)]
    Tier(#[from] crate::TierError),
}

/// Whether the program starts a new run in the same process after a run ended with `error`: a lost
/// subscription or connection, a broker that isn't answering yet, or one that described no arena. These are what a
/// cascade does to the worker (the broker restarts, and capture carves its rings after the worker subscribed), and
/// exiting for them left a failed unit and the restart delay with nothing draining Tier 0. A chunk file problem still
/// ends the program.
#[must_use]
pub fn rides_through(error: &ServiceError) -> bool {
    matches!(error, ServiceError::Lost(_) | ServiceError::Broker(_) | ServiceError::NoArena)
}

/// How long to wait before the next run: [`Backoff::FIRST`] after a run that went on for [`Backoff::SETTLED`], and
/// twice the last wait (at most [`Backoff::LONGEST`]) after one that ended sooner, so a broker that is down is tried
/// again without a busy loop, and a run lost once in a while starts again at once.
#[derive(Debug)]
pub struct Backoff {
    wait: Duration,
}

impl Backoff {
    /// The wait after a run that had settled.
    pub const FIRST: Duration = Duration::from_millis(250);
    /// The longest wait.
    pub const LONGEST: Duration = Duration::from_secs(10);
    /// A run this long had settled: the waits start over.
    pub const SETTLED: Duration = Duration::from_secs(60);

    /// No run yet.
    #[must_use]
    pub const fn new() -> Self {
        Self { wait: Duration::ZERO }
    }

    /// The wait before the next run, after one that ran for `ran`.
    pub fn after(&mut self, ran: Duration) -> Duration {
        self.wait =
            if ran >= Self::SETTLED || self.wait.is_zero() { Self::FIRST } else { (self.wait * 2).min(Self::LONGEST) };
        self.wait
    }
}

impl Default for Backoff {
    fn default() -> Self {
        Self::new()
    }
}

/// What the forwarding tasks hand the loop.
enum Incoming {
    /// A subscription's event, by the subscription's index.
    Event(usize, SubEvent),
    /// The broker said something the worker can't go on from (an error at request id 0, the connection gone).
    Lost(String),
    /// A purge to carry out on the chunk files, then acknowledge.
    Purge { purge_id: Uuid, ids: Vec<Uuid>, spans: Vec<continuum_client::PurgedSpan> },
}

/// The running program: a connected, registered worker following its streams.
pub struct Service {
    work: WorkerLoop<BrokerOutputs>,
    /// One connection per stream: only its initial Subscribe and one-way ACKs run here. All other RPCs stay
    /// on BrokerOutputs::client, including discovery and operations needed before another stream subscribes.
    readers: Vec<Arc<Client>>,
    /// Forwarders are aborted after both clients close, including a task parked on the bounded service queue.
    forwarding: Vec<tokio::task::JoinHandle<()>>,
    rt: tokio::runtime::Handle,
    events_tx: mpsc::Sender<Incoming>,
    events: mpsc::Receiver<Incoming>,
    declared: BTreeSet<StreamId>,
    /// Discovered derived streams use opaque payload copies rather than raw absorption.
    derived: BTreeSet<StreamId>,
    /// Each followed stream's name, by id, for querying its chunks during a purge.
    names: BTreeMap<StreamId, String>,
    /// A reader on each subscribed ring stream's ring.
    rings: BTreeMap<StreamId, continuum_shm::RingReader>,
    settings: Settings,
    started: Instant,
    epoch_ns: u64,
    next_resubscribe: Instant,
    last_report: Instant,
    /// Which of the counters' reports (every 10 s) is written at INFO (#59).
    report_clock: continuum_log::Periodic,
    /// Purges carried out and acknowledged.
    purges: u64,
    /// Purges being carried out, a few files a step, oldest first.
    purge_jobs: std::collections::VecDeque<PurgeJob>,
    /// Recent purges' windows and ids, so a payload delivered after the notice (it was in flight) isn't archived.
    recent_purges: Vec<(Instant, PurgeWindows, BTreeSet<Uuid>)>,
    /// The sweep of unnamed chunk files still to run (`crate::sweep`), with the paths recovery handed back.
    sweep_pending: Option<BTreeSet<PathBuf>>,
    /// The opt-in AV1 archive pass (`crate::archive`), when it's on.
    archive: Option<crate::archive::Archive>,
}

impl Drop for Service {
    fn drop(&mut self) {
        // A caller may drop the service after an error without calling shutdown. Wake and drop forwarders
        // that otherwise retain their clients forever while waiting for another broker event.
        for task in &self.forwarding {
            task.abort();
        }
    }
}

/// Files a purge goes through in one step, so subscriptions keep draining while a large purge runs.
pub const PURGE_FILES_PER_STEP: usize = 32;
/// At most how long one step spends on a purge's files, so Tier 0's events keep being read beside a large purge:
/// a frame not read before its slot is reused is lost.
pub const PURGE_STEP_BUDGET: Duration = Duration::from_millis(20);
/// At most how many queued events one step takes before it turns to purges.
pub const EVENTS_PER_STEP: usize = 512;

/// How long a purge's windows and ids keep late deliveries out of Tier 1.
pub const PURGE_MEMORY: Duration = Duration::from_secs(600);

/// A purge being carried out on the chunk files.
struct PurgeJob {
    /// The purges sharing this pass over the files: each is acknowledged once it's done.
    purge_ids: Vec<Uuid>,
    /// The pass has begun, so a purge arriving now waits for the next one.
    started: bool,
    /// The purged windows per stream, merged.
    windows: BTreeMap<StreamId, Vec<continuum_core::Interval>>,
    /// The purged payload ids: a frame in a chunk carries its id, so a purge by id reaches it without a span.
    ids: BTreeSet<Uuid>,
    files: std::collections::VecDeque<(StreamId, PathBuf)>,
    /// The chunks the catalog knows over the windows, by file: rewritten rather than deleted, and told the new
    /// frame index.
    known: BTreeMap<String, Uuid>,
    /// Known chunks still to read whole: every copy's URI and tier.
    to_snapshot: std::collections::VecDeque<Uuid>,
    /// By file URI: its copy's tier, from one snapshot.
    tiers: BTreeMap<String, continuum_core::StorageTier>,
    removed: usize,
    rewritten: usize,
    refused: usize,
}

impl PurgeJob {
    /// Takes in a purge that arrived before this pass began: one pass over the files then does both.
    fn absorb(&mut self, other: Self) {
        debug_assert!(!self.started && !other.started);
        self.purge_ids.extend(other.purge_ids);
        for (stream, windows) in other.windows {
            let mine = self.windows.entry(stream).or_default();
            mine.extend(windows);
            *mine = merge_windows(std::mem::take(mine));
        }
        self.ids.extend(other.ids);
        let have: BTreeSet<PathBuf> = self.files.iter().map(|(_, path)| path.clone()).collect();
        self.files.extend(other.files.into_iter().filter(|(_, path)| !have.contains(path)));
        self.known.extend(other.known);
        let have: BTreeSet<Uuid> = self.to_snapshot.iter().copied().collect();
        self.to_snapshot.extend(other.to_snapshot.into_iter().filter(|c| !have.contains(c)));
    }
}

/// How long each phase of a start took.
struct Phases {
    began: Instant,
    last: Instant,
    took: Vec<(&'static str, Duration)>,
}

impl Phases {
    fn new() -> Self {
        let now = Instant::now();
        Self { began: now, last: now, took: Vec::new() }
    }

    fn done(&mut self, phase: &'static str) {
        let now = Instant::now();
        self.took.push((phase, now - self.last));
        self.last = now;
    }

    /// "startup took 3.2 s: connect 0.0 s, register 3.1 s, ..." when the whole start took at least `over`.
    fn slow(&self, over: Duration) -> Option<String> {
        let total = self.last - self.began;
        (total >= over).then(|| {
            let each: Vec<String> =
                self.took.iter().map(|(phase, took)| format!("{phase} {:.1} s", took.as_secs_f64())).collect();
            format!("startup took {:.1} s: {}", total.as_secs_f64(), each.join(", "))
        })
    }
}

impl Service {
    /// The directory relative locators are read against: the data root, or the working directory without one.
    fn root(&self) -> PathBuf {
        self.settings.data_root.clone().unwrap_or_default()
    }

    /// Connects, registers, maps the arena, recovers what the last run left open, and subscribes to every stream
    /// the broker knows. Call from a thread that isn't a runtime worker (it blocks on `rt`).
    ///
    /// # Errors
    ///
    /// The broker's refusal, no arena, or a chunk directory problem.
    pub fn start(settings: Settings, rt: &tokio::runtime::Handle) -> Result<Self, ServiceError> {
        // Each phase's time, said when the start is slow: one start after a cascade took 168 s with nothing in the
        // log while capture's Tier 0 filled.
        let mut phases = Phases::new();
        let client = Arc::new(rt.block_on(Client::connect(&settings.socket, ClientOptions::new(WORKER_NAME)))?);
        phases.done("connect");
        let worker = WorkerSpec {
            worker_id: WORKER_NAME.into(),
            version_hash: env!("CARGO_PKG_VERSION").into(),
            capabilities: Vec::new(),
            device_index: None,
            // Purges wait for this worker's PurgeAck (see the module doc).
            keeps_copies: true,
        };
        rt.block_on(client.register_worker(&worker))?;
        phases.done("register");

        let arena = client.arenas().first().map(|m| m.arena.clone()).ok_or(ServiceError::NoArena)?;
        let (events_tx, events) = mpsc::channel(4_096);
        let subs = settings.streams.iter().map(|_| None).collect();
        let out = BrokerOutputs {
            client,
            rt: rt.clone(),
            arena,
            subs,
            pending: BTreeMap::new(),
            early: BTreeSet::new(),
            data_root: settings.data_root.clone(),
        };
        let began = Instant::now();
        let mut work = WorkerLoop::new(&settings.chunk_dir, settings.flush, out)?;
        phases.done("chunk directory");
        let found = crate::recover(&settings.chunk_dir)?;
        phases.done("recovery");
        if began.elapsed() >= Duration::from_secs(1) {
            // Nothing is flushed until this is done, so the encoded cameras' Tier 0 rings fill meanwhile.
            tracing::info!("startup recovery took {:.1} s", began.elapsed().as_secs_f64());
        }
        if !found.is_empty() {
            tracing::info!("recovered {} chunks the last run left open", found.len());
        }
        let recovered_paths = found.iter().map(|r| r.path.clone()).collect();
        work.recovered(found);
        let mut service = Self {
            work,
            readers: Vec::new(),
            forwarding: Vec::new(),
            rt: rt.clone(),
            events_tx,
            events,
            declared: BTreeSet::new(),
            derived: BTreeSet::new(),
            names: BTreeMap::new(),
            rings: BTreeMap::new(),
            settings,
            started: Instant::now(),
            epoch_ns: now_utc_ns(),
            next_resubscribe: Instant::now(),
            last_report: Instant::now(),
            report_clock: continuum_log::Periodic::default(),
            purges: 0,
            purge_jobs: std::collections::VecDeque::new(),
            recent_purges: Vec::new(),
            sweep_pending: Some(recovered_paths),
            archive: None,
        };
        service.watch_client(Arc::clone(&service.work.outputs().client), "writer".into());
        if let Err(error) = service.recover_copies() {
            service.close_clients("payload-copy recovery failed");
            return Err(error);
        }
        phases.done("copies");
        if let Err(error) = service.subscribe_missing() {
            service.close_clients("subscription startup failed");
            return Err(error);
        }
        phases.done("subscribe");
        if let Some(line) = phases.slow(Duration::from_secs(1)) {
            tracing::info!("{line}");
        }
        Ok(service)
    }

    fn watch_client(&mut self, client: Arc<Client>, role: String) {
        let tx = self.events_tx.clone();
        self.forwarding.push(self.rt.spawn(async move {
            loop {
                match client.next_event().await {
                    Some(message) => {
                        if let Some(error) = message.envelope().payload_as_error() {
                            let why = format!("{role}: {:?}: {}", error.code(), error.message().unwrap_or(""));
                            let _ = tx.send(Incoming::Lost(why)).await;
                            return;
                        }
                        let sent = match message.event() {
                            continuum_client::BrokerEvent::PurgeNotice { purge_id, payload_ids, spans } => {
                                tx.send(Incoming::Purge { purge_id, ids: payload_ids, spans }).await
                            }
                            _ => Ok(()),
                        };
                        if sent.is_err() {
                            return;
                        }
                    }
                    None => {
                        let _ = tx.send(Incoming::Lost(format!("the {role} connection closed"))).await;
                        return;
                    }
                }
            }
        }));
    }

    fn close_clients(&mut self, reason: &str) {
        // Commit/settle happens before this call, so reader ACKs precede its Goodbye on the write queue.
        for reader in &self.readers {
            let _ = self.rt.block_on(reader.close(reason));
        }
        let client = Arc::clone(&self.work.outputs().client);
        let _ = self.rt.block_on(client.close(reason));
        for task in self.forwarding.drain(..) {
            task.abort();
        }
    }

    /// Subscribes to each stream not yet followed; one the broker doesn't know yet is tried again later.
    fn subscribe_missing(&mut self) -> Result<(), ServiceError> {
        self.next_resubscribe = Instant::now() + RESUBSCRIBE;
        let client = Arc::clone(&self.work.outputs_mut().client);
        match self.rt.block_on(client.list_streams()) {
            Ok(streams) => {
                for stream in streams
                    .into_iter()
                    .filter(|stream| stream.kind == continuum_core::schema::continuum::StreamKind::Derived)
                {
                    self.derived.insert(stream.id);
                    self.names.insert(stream.id, stream.name.clone());
                    if !self.settings.streams.contains(&stream.name) {
                        self.settings.streams.push(stream.name);
                        self.work.outputs_mut().subs.push(None);
                    }
                }
            }
            Err(error) => tracing::warn!("discovering derived streams failed; retrying: {error}"),
        }
        for i in 0..self.settings.streams.len() {
            if self.work.outputs_mut().subs[i].is_some() {
                continue;
            }
            let name = self.settings.streams[i].clone();
            let mut spec = SubscribeSpec::new([name.clone()], Start::Saved);
            spec.durable_name = Some(format!("{WORKER_NAME}/{name}"));
            spec.pinned = true;
            // Never subscribe a second stream on an existing intake connection: its first catch-up may already
            // be applying backpressure, which would hold the next Subscribe reply behind those events.
            let mut options = ClientOptions::new(format!("{WORKER_NAME}-reader/{name}"));
            options.subscription_overflow = SubscriptionOverflow::Backpressure;
            let client = match self.rt.block_on(Client::connect(&self.settings.socket, options)) {
                Ok(client) => Arc::new(client),
                Err(error) => {
                    tracing::warn!("connecting {name}'s reader failed; retrying: {error}");
                    continue;
                }
            };
            let arena = &self.work.outputs().arena;
            if !client
                .arenas()
                .iter()
                .any(|mapped| mapped.arena.name() == arena.name() && mapped.arena.generation() == arena.generation())
            {
                let _ = self.rt.block_on(client.close("broker arena changed"));
                return Err(ServiceError::Lost("broker arena changed while opening a subscription connection".into()));
            }
            match self.rt.block_on(client.subscribe(&spec)) {
                Ok(mut sub) => {
                    self.watch_client(Arc::clone(&client), format!("reader for {name}"));
                    self.readers.push(client);
                    let start = sub.start_position();
                    for (stream, lease) in sub.rings() {
                        let out = self.work.outputs_mut();
                        let reader = lease
                            .carved_ring()
                            .and_then(|ring| out.arena.carved_ring_reader(&ring, continuum_shm::ReadFrom::Oldest).ok());
                        match reader {
                            Some(r) => {
                                self.rings.insert(*stream, r);
                            }
                            None => tracing::warn!("can't open {name}'s ring ({lease:?})"),
                        }
                    }
                    let positions = Positions {
                        handle: sub.handle(),
                        open: BTreeSet::new(),
                        seen: start,
                        acked: start,
                        failures: 0,
                    };
                    self.work.outputs_mut().subs[i] = Some(positions);
                    let tx = self.events_tx.clone();
                    tracing::info!("following {name} from position {start}");
                    self.forwarding.push(self.rt.spawn(async move {
                        while let Some(e) = sub.next().await {
                            if tx.send(Incoming::Event(i, e)).await.is_err() {
                                return;
                            }
                        }
                        let _ = tx.send(Incoming::Lost(format!("the subscription to {name} ended"))).await;
                    }));
                }
                Err(e) if e.broker_code() == Some(ErrorCode::NotFound) => {
                    let _ = self.rt.block_on(client.close("stream not registered"));
                    tracing::warn!("{name} isn't registered yet; trying again in {RESUBSCRIBE:?}");
                }
                Err(e) => {
                    let _ = self.rt.block_on(client.close("subscription failed"));
                    tracing::warn!("subscribing to {name} failed: {e}");
                }
            }
        }
        Ok(())
    }

    /// Reattach files left after fsync but before a copy acknowledgement or saved subscription position.
    fn recover_copies(&mut self) -> Result<(), ServiceError> {
        for copy in crate::copies::recover(&self.settings.chunk_dir)? {
            self.attach_copy(copy)?;
        }
        Ok(())
    }

    fn attach_copy(&mut self, copy: crate::copies::Copy) -> Result<(), ServiceError> {
        let out = self.work.outputs_mut();
        let location = stored(out.data_root.as_deref(), copy.location.clone());
        match self.rt.block_on(out.client.commit_payload_copy(copy.record.id, &location)) {
            Ok(()) => {
                self.derived.insert(copy.stream);
                // So the next start doesn't commit it again; a mark that fails only costs that.
                if let Err(error) = crate::copies::mark_committed(&copy.path) {
                    tracing::warn!("marking payload copy {} committed failed: {error}", copy.path.display());
                }
                if out.pending.contains_key(&copy.record.id) {
                    out.settle(copy.record.id, true);
                }
                // Recovery needs no per-ID cache: a hot redelivery validates/reuses the deterministic file;
                // a warm redelivery is already durable and simply advances its saved subscription position.
            }
            Err(error) if error.broker_code() == Some(ErrorCode::NotFound) => {
                // The original is missing/purged: never reinsert it. Remove the orphan durably before releasing
                // its position, including a copy that raced the purge's inventory.
                crate::copies::remove(&copy.path)?;
                out.settle(copy.record.id, false);
            }
            Err(error) => {
                // A Conflict can be another spelling of this very same file. It never authorizes deletion.
                tracing::warn!(
                    "attaching payload {} at {} failed; file preserved: {error}",
                    copy.record.id,
                    copy.path.display()
                );
                return Err(error.into());
            }
        }
        out.ack_all();
        Ok(())
    }

    fn derived_payload(&mut self, sub: usize, position: u64, item: &Item) -> Result<(), ServiceError> {
        let h = &item.header;
        // Other derived originals may share this span but have independent provenance/subjects.
        let ignored = self.recent_purges.iter().any(|(_, _, ids)| ids.contains(&h.id()));
        let out = self.work.outputs_mut();
        if let Some(Some(saved)) = out.subs.get_mut(sub) {
            saved.seen = saved.seen.max(position);
        }
        if ignored {
            out.ack_all();
            return Ok(());
        }
        let Some(StorageLocator::Shm { arena, arena_generation, slab_class, slot, slot_generation, length, .. }) =
            &item.locator
        else {
            // Inline/header-only and existing durable locations already have their proper home.
            out.ack_all();
            return Ok(());
        };
        let handle = SlotHandle { class: *slab_class, index: *slot, generation: *slot_generation };
        out.pending.insert(h.id(), Pending { slot: Some(handle), sub, position });
        if let Some(Some(saved)) = out.subs.get_mut(sub) {
            saved.open.insert(position);
        }
        let bytes = (out.arena.name() == arena && out.arena.generation() == *arena_generation)
            .then(|| read_slot(&out.arena, handle))
            .flatten();
        let Some(mut bytes) = bytes else {
            tracing::warn!("derived payload {} is gone from Tier 0", h.id());
            out.settle(h.id(), false);
            out.ack_all();
            return Ok(());
        };
        let length =
            usize::try_from(*length).map_err(|_| crate::TierError::Codec("derived slot length is too large".into()))?;
        if length > bytes.len() {
            return Err(crate::TierError::Codec("derived locator exceeds its leased payload".into()).into());
        }
        bytes.truncate(length);
        let copy = crate::copies::write(&self.settings.chunk_dir, h, &bytes)?;
        self.attach_copy(copy)
    }

    /// The loop's clock: UTC nanoseconds, moving monotonically from when the service started.
    fn now_ns(&self) -> u64 {
        self.epoch_ns + u64::try_from(self.started.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }

    /// One turn of the loop: waits up to the tick for an event, handles it, and lets time pass.
    ///
    /// # Errors
    ///
    /// A chunk file problem, a payload the loop can't append, or the broker dropping a subscription or the
    /// connection.
    pub fn step(&mut self) -> Result<(), ServiceError> {
        let (tick, events) = (self.settings.tick, &mut self.events);
        // The timer is made inside the runtime: `timeout` needs its clock when it's created.
        let next = self.rt.block_on(async { tokio::time::timeout(tick, events.recv()).await });
        if let Ok(Some(incoming)) = next {
            self.incoming(incoming)?;
            // Everything already queued too, before any purge work: a purge must never starve Tier 0's reads.
            for _ in 1..EVENTS_PER_STEP {
                let Ok(incoming) = self.events.try_recv() else { break };
                self.incoming(incoming)?;
            }
        }
        self.after_events()
    }

    fn incoming(&mut self, incoming: Incoming) -> Result<(), ServiceError> {
        match incoming {
            Incoming::Event(sub, SubEvent::Event { position, item }) => {
                self.payload(sub, position, &item)?;
            }
            Incoming::Event(sub, SubEvent::CaughtUp { position }) => {
                tracing::debug!("{} caught up at {position}", self.settings.streams[sub]);
            }
            Incoming::Event(sub, SubEvent::Undecodable { position, reason }) => {
                // It can never be read, so it can't be archived: passed over, loudly.
                tracing::warn!("{} event {position} unreadable, passed over: {reason}", self.settings.streams[sub]);
                if let Some(Some(s)) = self.work.outputs_mut().subs.get_mut(sub) {
                    s.seen = s.seen.max(position);
                }
            }
            Incoming::Lost(why) => return Err(ServiceError::Lost(why)),
            Incoming::Purge { purge_id, ids, spans } => self.begin_purge(purge_id, ids, &spans)?,
        }
        Ok(())
    }

    fn after_events(&mut self) -> Result<(), ServiceError> {
        self.work.tick(self.now_ns())?;
        self.purge_step()?;
        self.archive_step()?;
        if self.started.elapsed() >= crate::sweep::AFTER
            && let Some(skip) = self.sweep_pending.take()
        {
            self.sweep_unnamed(&skip)?;
        }
        if Instant::now() >= self.next_resubscribe {
            self.subscribe_missing()?;
        }
        if self.last_report.elapsed() >= Duration::from_secs(10) {
            self.last_report = Instant::now();
            continuum_log::periodic!(
                self.report_clock,
                "{:?}, {} pending, {} uncommitted",
                self.work.counts(),
                self.work.pending(),
                self.work.uncommitted()
            );
        }
        Ok(())
    }

    /// Runs until `stop` is set or something ends the run, then closes and commits every open chunk and says
    /// goodbye (so the last saved positions reach the broker).
    ///
    /// # Errors
    ///
    /// What ended the run, when it wasn't `stop`.
    pub fn run(&mut self, stop: &AtomicBool) -> Result<(), ServiceError> {
        let mut result = Ok(());
        while !stop.load(Ordering::SeqCst) {
            if let Err(e) = self.step() {
                result = Err(e);
                break;
            }
        }
        self.shutdown()?;
        result
    }

    /// Closes and commits every open chunk and closes all connections (the last saved positions go out before
    /// each intake connection's goodbye).
    ///
    /// # Errors
    ///
    /// A chunk file problem.
    pub fn shutdown(&mut self) -> Result<(), ServiceError> {
        if let Some(mut archive) = self.archive.take() {
            archive.stop();
        }
        let result = self.work.shutdown().map_err(ServiceError::from);
        self.close_clients("tier worker stopped");
        result
    }

    /// Turns on the AV1 archive pass (`crate::archive`). A pass that can't start (no `ffmpeg` feature) is
    /// logged and left off: recording never depends on it.
    pub fn with_archive(&mut self, settings: crate::archive::ArchiveSettings) {
        match crate::archive::Archive::start(settings) {
            Ok(archive) => {
                tracing::info!("AV1 archive on: {settings:?}");
                self.archive = Some(archive);
            }
            Err(e) => tracing::warn!("AV1 archive off: {e}"),
        }
    }

    /// What the AV1 archive pass has done this run, when it's on.
    #[must_use]
    pub fn archive_counts(&self) -> Option<&crate::archive::ArchiveCounts> {
        self.archive.as_ref().map(|a| &a.counts)
    }

    /// One turn of the AV1 archive pass: swaps in what the encoder thread finished (never during a purge, whose
    /// rewrite comes first), hands it the next chunk, and scans for more once it's idle.
    ///
    /// # Errors
    ///
    /// A chunk file problem swapping one in, or the connection failing (a refused rewrite is logged).
    fn archive_step(&mut self) -> Result<(), ServiceError> {
        let root = self.root();
        let now = self.now_ns();
        let client = Arc::clone(&self.work.outputs_mut().client);
        let Some(archive) = self.archive.as_mut() else { return Ok(()) };
        let mut rewrites = Vec::new();
        while let Some((candidate, outcome)) = archive.finished() {
            match outcome {
                #[cfg(feature = "ffmpeg")]
                crate::archive::Outcome::Ready(ready) if self.purge_jobs.is_empty() => {
                    match ready.swap(&candidate.path)? {
                        Some(rewritten) => {
                            archive.counts.converted += 1;
                            archive.counts.bytes_before += ready.before;
                            archive.counts.bytes_after += rewritten.length;
                            rewrites.push((candidate, rewritten.records, rewritten.length));
                        }
                        None => archive.counts.dropped += 1,
                    }
                }
                #[cfg(feature = "ffmpeg")]
                crate::archive::Outcome::Ready(_) => archive.counts.dropped += 1,
                crate::archive::Outcome::Kept(_) => {}
                crate::archive::Outcome::Failed(why) => {
                    tracing::warn!("AV1 archive: {}: {why}", candidate.path.display())
                }
            }
        }
        if archive.failing() {
            tracing::warn!("AV1 archive off for this run: {} encodes failed in a row", crate::archive::MAX_FAILURES);
            if let Some(mut off) = self.archive.take() {
                off.stop();
            }
            return self.send_rewrites(&client, rewrites);
        }
        archive.feed();
        // The configured camera streams, by name: a camera that's off now still has a backlog to convert.
        let cameras: Vec<String> = self.settings.streams.iter().filter(|n| n.ends_with(".video")).cloned().collect();
        if archive.idle()
            && self.purge_jobs.is_empty()
            && !cameras.is_empty()
            && let Some(range) = archive.next_range(now)
        {
            let mut items = Vec::new();
            for name in cameras {
                let spec = continuum_client::QuerySpec::hit_only(name, range);
                let found = match self.rt.block_on(client.query_all(&spec)) {
                    Ok(found) => found.items,
                    // A camera the broker doesn't know (not plugged in yet): nothing of it to convert in this slice.
                    Err(e) if e.broker_code().is_some() => continue,
                    Err(e) => return Err(e.into()),
                };
                for item in found {
                    if let Some(StorageLocator::Chunk { tier, uri, offset, length }) = item.locator {
                        // The archive works on files: the locator's own, resolved against the data root.
                        let uri = local_uri(&root, &uri)?;
                        items.push((item.header.id(), StorageLocator::Chunk { tier, uri, offset, length }));
                    }
                }
            }
            for candidate in archive.take(items) {
                let reader = crate::chunk::ChunkReader::open(&candidate.path).map_err(crate::TierError::from)?;
                let length = std::fs::metadata(&candidate.path).map_err(crate::TierError::from)?.len();
                archive.counts.resent += 1;
                rewrites.push((candidate, reader.records().to_vec(), length));
            }
            archive.feed();
        }
        if let Some(line) = archive.report() {
            tracing::info!("{line}");
        }
        self.send_rewrites(&client, rewrites)
    }

    /// Tells the catalog each swapped chunk's new frame index and length (`RewriteChunk`), keeping its tier.
    fn send_rewrites(
        &self,
        client: &Client,
        rewrites: Vec<(crate::archive::Candidate, Vec<crate::chunk::Record>, u64)>,
    ) -> Result<(), ServiceError> {
        for (candidate, records, length) in rewrites {
            let uri = locator_uri(self.settings.data_root.as_deref(), &candidate.path);
            let location = StorageLocator::Chunk { tier: candidate.tier, uri, offset: 0, length };
            match self.rt.block_on(client.rewrite_chunk(candidate.chunk, &location, &catalog_frames(&records))) {
                Ok(()) => {}
                // The chunk is gone or hidden since the scan: the file is what's left, and the next scan looks again.
                Err(e) if e.broker_code().is_some() => {
                    tracing::warn!("AV1 archive: chunk {} wasn't rewritten in the catalog: {e}", candidate.chunk);
                }
                Err(e) => return Err(e.into()),
            }
        }
        Ok(())
    }

    /// Closed chunks waiting to be committed, and payloads appended but not yet durable.
    #[must_use]
    pub fn waiting_commits(&self) -> usize {
        self.work.uncommitted() + self.work.pending()
    }

    /// Purges carried out on the chunk files and acknowledged.
    #[must_use]
    pub fn purges(&self) -> u64 {
        self.purges
    }

    /// What the loop has done.
    #[must_use]
    pub fn counts(&self) -> crate::worker::Counts {
        self.work.counts()
    }

    /// Starts a purge: closes and commits the open chunks (so no purged payload is left in a file still being
    /// written), works out what to cut, and queues the files; [`Service::step`] then goes through them a few at a
    /// time and acknowledges once the last is done.
    ///
    /// The broker sends a notice again every minute until it's acknowledged: a purge already queued isn't queued
    /// twice. A purge arriving while the last queued pass hasn't begun joins that pass, so purges that arrive faster
    /// than one pass over every file finishes share passes instead of piling up.
    ///
    /// # Errors
    ///
    /// A chunk file problem, or the broker failing a lookup.
    fn begin_purge(
        &mut self,
        purge_id: Uuid,
        ids: Vec<Uuid>,
        spans: &[continuum_client::PurgedSpan],
    ) -> Result<(), ServiceError> {
        let root = self.root();
        if self.purge_jobs.iter().any(|j| j.purge_ids.contains(&purge_id)) {
            return Ok(());
        }
        let mut by_stream: BTreeMap<StreamId, Vec<continuum_core::Interval>> = BTreeMap::new();
        for span in spans {
            if let Some(w) = continuum_core::Interval::new(span.range.0, span.range.1) {
                by_stream.entry(StreamId(span.stream_id)).or_default().push(w);
            }
        }
        let windows: BTreeMap<StreamId, Vec<continuum_core::Interval>> =
            by_stream.into_iter().map(|(s, w)| (s, merge_windows(w))).collect();
        let ids: BTreeSet<Uuid> = ids.into_iter().collect();
        // The broker names the spans it removed per stream beside the ids, so the files to look at are those
        // streams'. Only ids without spans could be anywhere. Scanning every file for an inline-only
        // stream's age-out (a ten-minute retention stream) stalled Tier 0's reads long enough to lose video.
        let anywhere = windows.is_empty() && !ids.is_empty();
        let reaches = |s: &StreamId| anywhere || windows.contains_key(s);
        if anywhere || self.work.open_streams().iter().any(reaches) {
            self.work.close_open()?;
        }
        self.recent_purges.push((Instant::now(), windows.clone(), ids.clone()));
        // The streams' names (to query their chunks), for streams no payload has named yet this run: each configured
        // stream not named yet is looked up over the windows, and named by what it returns.
        let client = Arc::clone(&self.work.outputs_mut().client);
        if windows.keys().any(|s| !self.names.contains_key(s)) {
            let all = windows.values().flatten();
            let hull =
                (all.clone().map(|w| w.start_ns()).min().unwrap_or(0), all.map(|w| w.end_ns()).max().unwrap_or(0));
            for name in self.settings.streams.clone() {
                if self.names.values().any(|n| *n == name) {
                    continue;
                }
                let spec = continuum_client::QuerySpec::hit_only(name.clone(), hull);
                if let Ok(r) = self.rt.block_on(client.query(&spec))
                    && let Some(item) = r.items.first()
                {
                    self.names.insert(item.header.stream(), name);
                }
            }
        }
        let mut known = BTreeMap::new();
        let mut located = Vec::new();
        for (stream, w) in &windows {
            let Some(name) = self.names.get(stream) else { continue };
            let hull = (w.first().map_or(0, |x| x.start_ns()), w.last().map_or(0, |x| x.end_ns()));
            for item in
                self.rt.block_on(client.query_all(&continuum_client::QuerySpec::hit_only(name.clone(), hull)))?.items
            {
                if let Some(StorageLocator::Chunk { uri, .. }) = &item.locator {
                    let uri = local_uri(&root, uri)?;
                    located.push((*stream, PathBuf::from(&uri)));
                    known.insert(uri, item.header.id());
                }
            }
        }
        // Files: those of the purged streams, or every one when ids name payloads anywhere. The chunk directory's,
        // and any file the catalog locates outside it (an Opus chunk the removed Tier 2 sweep wrote under `tier2/`):
        // forgetting reaches every copy, whatever tier its locator names.
        let mut all = chunk_files(&self.settings.chunk_dir, None)?;
        let mut have = all
            .iter()
            .map(|(_, path)| local_uri(&root, &path.to_string_lossy()))
            .collect::<Result<BTreeSet<_>, _>>()?;
        for (stream, path) in located {
            if path.is_file() && have.insert(path.to_string_lossy().into_owned()) {
                all.push((stream, path));
            }
        }
        let files: std::collections::VecDeque<(StreamId, PathBuf)> =
            all.into_iter().filter(|(s, _)| reaches(s)).collect();
        let job = PurgeJob {
            purge_ids: vec![purge_id],
            started: false,
            windows,
            ids,
            files,
            to_snapshot: known.values().copied().collect::<BTreeSet<_>>().into_iter().collect(),
            tiers: BTreeMap::new(),
            known,
            removed: 0,
            rewritten: 0,
            refused: 0,
        };
        let shared = match self.purge_jobs.back_mut().filter(|j| !j.started) {
            Some(waiting) => {
                waiting.absorb(job);
                waiting
            }
            None => {
                self.purge_jobs.push_back(job);
                self.purge_jobs.back_mut().expect("just queued")
            }
        };
        tracing::info!(
            "purge {purge_id}: {} windows on {} streams, {} ids, {} files to look at{}",
            shared.windows.values().map(Vec::len).sum::<usize>(),
            shared.windows.len(),
            shared.ids.len(),
            shared.files.len(),
            match shared.purge_ids.len() {
                1 => String::new(),
                n => format!(" (one pass for {n} purges)"),
            }
        );
        Ok(())
    }

    /// Deletes the chunk files the previous run made just before this one started that no catalog row names (see
    /// [`crate::sweep`]), once, and logs the counts. `skip` is what startup recovery handed back.
    ///
    /// # Errors
    ///
    /// An I/O error listing the chunk directory, or the connection failing.
    fn sweep_unnamed(&mut self, skip: &BTreeSet<PathBuf>) -> Result<(), ServiceError> {
        use crate::sweep;
        let root = self.root();
        let dry_run = std::env::var_os(sweep::DRY_RUN_ENV).is_some();
        let mut report = sweep::Report::default();
        let found = sweep::candidates(&self.settings.chunk_dir, self.epoch_ns, skip, &mut report.skipped)?;
        let client = Arc::clone(&self.work.outputs_mut().client);
        for (stream, (hull, files)) in sweep::by_stream(&found) {
            let Some(name) = self.names.get(&stream) else {
                report.skipped += files.len();
                continue;
            };
            let spec = continuum_client::QuerySpec::hit_only(name.clone(), hull);
            let items = self.rt.block_on(client.query_all(&spec))?.items;
            let named = sweep::named_files(
                &root,
                items.iter().filter_map(|item| match &item.locator {
                    Some(StorageLocator::Chunk { uri, .. }) => Some(uri.as_str()),
                    _ => None,
                }),
            );
            sweep::sweep(&files, &named, dry_run, &mut report);
        }
        tracing::info!(
            "sweep{}: {} chunk files from before this run looked at, {} named by no catalog row, {} deleted, \
             {} passed over",
            if dry_run { " (dry run)" } else { "" },
            report.checked,
            report.unnamed,
            report.deleted,
            report.skipped
        );
        Ok(())
    }

    /// Goes through a few of the oldest purge's files, and acknowledges it once none are left.
    ///
    /// # Errors
    ///
    /// A chunk file problem, or the connection failing (a refusal of a rewrite is logged and counted instead).
    fn purge_step(&mut self) -> Result<(), ServiceError> {
        let root = self.root();
        let data_root = self.settings.data_root.clone();
        self.recent_purges.retain(|(at, _, _)| at.elapsed() < PURGE_MEMORY);
        let Some(job) = self.purge_jobs.front_mut() else { return Ok(()) };
        job.started = true;
        let began = Instant::now();
        let client = Arc::clone(&self.work.outputs_mut().client);
        // Every copy of each known chunk, with its frame index, before any file is touched: the time query names one
        // locator per chunk, and a stored GOP file needs its chunk's original unit starts to purge exactly.
        for _ in 0..PURGE_FILES_PER_STEP {
            if began.elapsed() >= PURGE_STEP_BUDGET {
                break;
            }
            let Some(chunk) = job.to_snapshot.pop_front() else { break };
            match self.rt.block_on(client.get_chunk(chunk)) {
                Ok(info) => {
                    for location in &info.locations {
                        if let StorageLocator::Chunk { tier, uri, .. } = location {
                            let uri = local_uri(&root, uri)?;
                            job.known.insert(uri.clone(), chunk);
                            job.tiers.insert(uri, *tier);
                        }
                    }
                }
                // Gone or hidden since the query: its files fall back to the conservative rules.
                Err(e) if e.broker_code().is_some() => {}
                Err(e) => return Err(e.into()),
            }
        }
        if !job.to_snapshot.is_empty() {
            return Ok(());
        }
        for _ in 0..PURGE_FILES_PER_STEP {
            if began.elapsed() >= PURGE_STEP_BUDGET {
                break;
            }
            let Some((stream, path)) = job.files.pop_front() else { break };
            let uri = local_uri(&root, &path.to_string_lossy())?;
            let chunk = job.known.get(&uri).copied();
            let windows = job.windows.get(&stream).map_or(&[][..], Vec::as_slice);
            if crate::copies::is_copy(&path) {
                if crate::copies::purge(&path, &job.ids)? {
                    job.removed += 1;
                }
                continue;
            }
            let records = match purge_file(&path, windows, &job.ids, chunk.is_none())? {
                PurgedFile::Missing => continue,
                PurgedFile::Removed => {
                    job.removed += 1;
                    continue;
                }
                PurgedFile::Rewritten(rw) => {
                    job.rewritten += 1;
                    Some((rw.records, rw.length))
                }
                // Cut already (by this purge in a run that crashed before telling the catalog, or an earlier one):
                // the catalog is told its frame index again, which is idempotent.
                PurgedFile::Untouched { cut: true } if chunk.is_some() => {
                    let reader = crate::chunk::ChunkReader::open(&path).map_err(crate::TierError::from)?;
                    let length = std::fs::metadata(&path).map_err(crate::TierError::from)?.len();
                    Some((reader.records().to_vec(), length))
                }
                PurgedFile::Untouched { .. } => None,
            };
            if let (Some(chunk), Some((records, length))) = (chunk, records) {
                // The copy's own tier, from the snapshot: an Opus chunk under `tier2/` keeps its Tier 2 locator.
                let tier = job.tiers.get(&uri).copied().unwrap_or(continuum_core::StorageTier::Tier1Warm);
                let uri = locator_uri(data_root.as_deref(), Path::new(&uri));
                let location = StorageLocator::Chunk { tier, uri, offset: 0, length };
                let frames = catalog_frames(&records);
                match self.rt.block_on(client.rewrite_chunk(chunk, &location, &frames)) {
                    Ok(()) => {}
                    // The catalog refusing it can't change by asking again (the chunk is gone, hidden, or its cut
                    // frames differ): the file is what's left and it's purged; the catalog's own cut flags keep
                    // readers off the frames.
                    Err(e) if e.broker_code().is_some() => {
                        job.refused += 1;
                        tracing::warn!("chunk {chunk} wasn't rewritten in the catalog: {e}");
                    }
                    Err(e) => return Err(e.into()),
                }
            }
        }
        if job.files.is_empty() {
            for id in &job.purge_ids {
                client.ack_purge(*id)?;
            }
            tracing::info!(
                "purge {}{}: {} chunk files removed, {} rewritten, {} rewrites refused",
                job.purge_ids[0],
                match job.purge_ids.len() {
                    1 => String::new(),
                    n => format!(" and {} more sharing its pass", n - 1),
                },
                job.removed,
                job.rewritten,
                job.refused
            );
            self.purges += job.purge_ids.len() as u64;
            self.purge_jobs.pop_front();
        }
        Ok(())
    }

    /// Whether a delivered payload falls in a recent purge (it was in flight when the notice came).
    fn purged_lately(&self, id: Uuid, stream: StreamId, span: Option<continuum_core::Interval>) -> bool {
        self.recent_purges.iter().any(|(_, windows, ids)| {
            ids.contains(&id)
                || span.is_some_and(|s| windows.get(&stream).is_some_and(|w| w.iter().any(|x| x.overlaps(s))))
        })
    }

    /// One delivered payload: read its bytes from its slot and append it.
    fn payload(&mut self, sub: usize, position: u64, item: &Item) -> Result<(), ServiceError> {
        let h = &item.header;
        let stream = h.stream();
        if self.derived.contains(&stream) {
            return self.derived_payload(sub, position, item);
        }
        if self.work.is_quarantined(h.id()) {
            if let Some(Some(s)) = self.work.outputs_mut().subs.get_mut(sub) {
                s.seen = s.seen.max(position);
            }
            return Ok(());
        }
        if self.purged_lately(h.id(), stream, h.span()) {
            // Purged while it was on its way: not archived. Its slot is left to the broker.
            if let Some(Some(s)) = self.work.outputs_mut().subs.get_mut(sub) {
                s.seen = s.seen.max(position);
            }
            return Ok(());
        }
        let read = match &item.locator {
            Some(StorageLocator::Shm { slab_class, slot, slot_generation, .. }) => {
                let handle = SlotHandle { class: *slab_class, index: *slot, generation: *slot_generation };
                read_slot(&self.work.outputs_mut().arena, handle).map(|b| (b, Some(handle)))
            }
            Some(StorageLocator::Ring { first_seq, count, .. }) => match self.rings.get_mut(&stream) {
                Some(reader) => read_batch(reader, *first_seq, *count)?.map(|b| (b, None)),
                // Its ring was carved after this run subscribed (the producer registered it again, as capture does
                // after the broker restarts): end the run, as for a ring carved again, so the next one subscribes
                // with the ring. The batch isn't acknowledged, so it comes again then while the ring still holds it.
                None => {
                    return Err(ServiceError::Lost(format!(
                        "a ring batch on stream {} with no ring to read it from (carved after this run subscribed)",
                        stream.0
                    )));
                }
            },
            // Not in Tier 0 (a header-only payload, a chunk, a hydrated copy): nothing to flush.
            _ => {
                if let Some(Some(s)) = self.work.outputs_mut().subs.get_mut(sub) {
                    s.seen = s.seen.max(position);
                }
                return Ok(());
            }
        };
        if self.declared.insert(stream) {
            self.names.insert(stream, self.settings.streams[sub].clone());
            // Declaring also commits the chunks recovered for the stream.
            let stride = self.rings.get(&stream).map(continuum_shm::RingReader::stride);
            self.work.declare(stream, shape_of(item, stride));
        }
        if let Some(why) = self.work.recovery_error(stream) {
            return Err(ServiceError::Lost(why.to_owned()));
        }
        let Some((bytes, handle)) = read else {
            // Overwritten before the worker got to it: a gap in Tier 1, which the broker records.
            tracing::warn!("payload {} is gone from Tier 0", h.id());
            if let Some(Some(s)) = self.work.outputs_mut().subs.get_mut(sub) {
                s.seen = s.seen.max(position);
            }
            return Ok(());
        };
        let out = self.work.outputs_mut();
        out.pending.insert(h.id(), Pending { slot: handle, sub, position });
        if let Some(Some(s)) = out.subs.get_mut(sub) {
            s.open.insert(position);
            s.seen = s.seen.max(position);
        }
        let p = Payload {
            pcm_rate_hz: h.header().capture().map_or(0, |c| c.sample_rate_hz()),
            provenance: h.header().capture().and_then(frame_provenance),
            id: h.id(),
            stream,
            start_ns: h.start_ns(),
            end_ns: h.end_ns(),
            seq: h.seq(),
            bytes,
        };
        let now = self.now_ns();
        let camera = h.header().capture().filter(|c| c.width() > 0 && c.height() > 0).map(|c| {
            let stride = self.rings.get(&stream).map(continuum_shm::RingReader::stride);
            (
                shape_of(item, stride),
                CameraFrameInfo {
                    width: c.width(),
                    height: c.height(),
                    stride: c.stride(),
                    flip: c.flip(),
                    device_seq: c.device_seq(),
                },
            )
        });
        let appended = match camera {
            Some((shape, frame)) => self.work.video_payload(&p, shape, frame, now),
            None => self.work.payload(&p, now),
        };
        match appended {
            Ok(Appended::Appended) => Ok(()),
            // Dropped for good (counted as undecodable or quarantined): its position is released and its slot marked,
            // so the broker and its producer can reuse it. Unmarked, it's never reclaimed while a reader's pin is left
            // on it (even an expired one) or the archive is owed it, and the P-frames dropped after each capture
            // reconnect filled the RTSP cameras' class until a broker restart.
            Ok(Appended::Quarantined | Appended::Undecodable) => {
                let out = self.work.outputs_mut();
                out.settle(p.id, true);
                out.ack_all();
                Ok(())
            }
            // In a chunk recovered at startup; that chunk's commit settles it, or did already.
            Ok(Appended::Recovered) => {
                let out = self.work.outputs_mut();
                if out.early.remove(&p.id) {
                    out.settle(p.id, true);
                    out.ack_all();
                }
                Ok(())
            }
            // The run ends rather than pass a payload by: its position stays open, so it isn't saved past, and the
            // next run gets it again.
            Err(e) => Err(e.into()),
        }
    }
}

/// A purge's windows, merged, per stream.
type PurgeWindows = BTreeMap<StreamId, Vec<continuum_core::Interval>>;

/// What a purge did to one chunk file.
#[derive(Debug)]
pub enum PurgedFile {
    /// The file was already removed after the purge took its inventory; no bytes or catalog rewrite remain.
    Missing,
    /// Nothing more in it was purged. `cut` says whether it holds cut records (from this purge or an earlier one),
    /// so the catalog can be told its frame index again after a crash between a rewrite and its `RewriteChunk`.
    Untouched {
        /// It holds cut records.
        cut: bool,
    },
    /// It lay wholly inside a window and the catalog doesn't know it, so the file is deleted.
    Removed,
    /// The purged frames were rewritten away, keeping their places.
    Rewritten(crate::chunk::Rewritten),
}

/// Sorts and merges intervals that overlap or touch, so a purge's many spans (one a chunk, for video) become few.
#[must_use]
pub fn merge_windows(mut windows: Vec<continuum_core::Interval>) -> Vec<continuum_core::Interval> {
    windows.sort_by_key(|w| (w.start_ns(), w.end_ns()));
    let mut out: Vec<continuum_core::Interval> = Vec::new();
    for w in windows {
        if let Some(last) = out.last_mut()
            && w.start_ns() <= last.end_ns()
        {
            if w.end_ns() > last.end_ns()
                && let Some(joined) = continuum_core::Interval::new(last.start_ns(), w.end_ns())
            {
                *last = joined;
            }
            continue;
        }
        out.push(w);
    }
    out
}

/// Normalize the same local locator spellings used by the catalog without following a file alias.
/// A chunk file's records as the catalog's frame index (`RewriteChunk`), in the catalog's order.
fn catalog_frames(records: &[crate::chunk::Record]) -> Vec<ChunkFrame> {
    crate::flush::catalog_records(records)
        .into_iter()
        .map(|r| {
            let f = r.frame();
            ChunkFrame { start_ns: f.start_ns, offset: f.offset, length: f.length, cut: f.cut }
        })
        .collect()
}

/// The file a catalog locator names, as an absolute normalized path string: the key the purge, sweep, and archive
/// compare catalog rows and files by.
fn local_uri(root: &Path, uri: &str) -> Result<String, crate::TierError> {
    let path = continuum_core::chunk_path::resolve(root, uri);
    path.to_str().map(str::to_owned).ok_or_else(|| crate::TierError::Codec("chunk URI is not UTF-8".into()))
}

/// The locator a file is stored under: relative to the data root when there is one ([`continuum_core::chunk_path`]).
fn locator_uri(root: Option<&Path>, path: &Path) -> String {
    match root {
        Some(root) => continuum_core::chunk_path::to_uri(root, path),
        None => std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf()).to_string_lossy().into_owned(),
    }
}

/// `location` with a chunk's file path in its stored form ([`locator_uri`]).
fn stored(root: Option<&Path>, location: StorageLocator) -> StorageLocator {
    match location {
        StorageLocator::Chunk { tier, uri, offset, length } => {
            let uri = locator_uri(root, Path::new(uri.strip_prefix("file://").unwrap_or(&uri)));
            StorageLocator::Chunk { tier, uri, offset, length }
        }
        other => other,
    }
}

/// The chunk files in the chunk directory, committed and set aside, with the stream each belongs to (from its name):
/// only those of `stream` when given, every one otherwise (a purge by id names no stream).
///
/// # Errors
///
/// An I/O error listing the directory.
pub fn chunk_files(dir: &Path, stream: Option<StreamId>) -> Result<Vec<(StreamId, PathBuf)>, crate::TierError> {
    let mut out = Vec::new();
    if !dir.exists() {
        return Ok(out);
    }
    for path in crate::layout::chunk_paths(dir, stream)? {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let ext = path.extension().and_then(|e| e.to_str());
        let Some(id) = name.split('-').next().and_then(|n| n.parse::<u32>().ok()) else { continue };
        if matches!(ext, Some("cnk" | "rejected" | "layout-unknown")) && stream.is_none_or(|s| s.0 == id) {
            out.push((StreamId(id), path));
        }
    }
    out.sort();
    Ok(out)
}

/// Purges one chunk file by the catalog's own rules: the frames whose slot (from its start to the next frame's,
/// the last to the chunk's end) overlaps one of `windows`, and those whose payload id is in `ids`, are rewritten
/// away, keeping their places. Video GOPs select by their original units' globally ordered slots and erase the
/// whole owning reference group, even when GOP presentation hulls overlap. A file whose whole span lies inside one window is deleted instead, but only when
/// `may_delete` (the catalog doesn't know the chunk; a chunk it still has is rewritten, so its locator stays good).
///
/// # Errors
///
/// A chunk file that can't be read, removed, or rewritten. A path already removed since inventory is
/// [`PurgedFile::Missing`], not an error; permission failures and other I/O errors still propagate.
pub fn purge_file(
    path: &Path,
    windows: &[continuum_core::Interval],
    ids: &BTreeSet<Uuid>,
    may_delete: bool,
) -> Result<PurgedFile, crate::TierError> {
    match purge_present_file(path, windows, ids, may_delete) {
        Err(crate::TierError::Chunk(crate::chunk::ChunkError::Io(error)) | crate::TierError::Io(error))
            if error.kind() == std::io::ErrorKind::NotFound =>
        {
            Ok(PurgedFile::Missing)
        }
        other => other,
    }
}

fn purge_present_file(
    path: &Path,
    windows: &[continuum_core::Interval],
    ids: &BTreeSet<Uuid>,
    may_delete: bool,
) -> Result<PurgedFile, crate::TierError> {
    let reader = crate::chunk::ChunkReader::open(path)?;
    let records = reader.records().to_vec();
    let ordered = crate::flush::catalog_record_indexes(&records);
    let already = records.iter().any(|r| r.cut);
    let Some(first) = ordered.first().map(|&index| &records[index]) else {
        return Ok(PurgedFile::Untouched { cut: already });
    };
    // The chunk's span, as the flusher gave its header (crate::flush::new_chunk).
    let end = records.iter().map(crate::chunk::Record::end_exclusive).max().unwrap_or(first.start_ns.saturating_add(1));
    let Some(span) = continuum_core::Interval::new(first.start_ns, end.max(first.start_ns.saturating_add(1))) else {
        return Ok(PurgedFile::Untouched { cut: already });
    };
    if may_delete && windows.iter().any(|w| span.start_ns() >= w.start_ns() && span.end_ns() <= w.end_ns()) {
        drop(reader);
        crate::chunk::remove(path)?;
        return Ok(PurgedFile::Removed);
    }
    drop(reader);
    let frames: Vec<continuum_catalog::ChunkFrame> = ordered.iter().map(|&index| records[index].frame()).collect();
    let slots = continuum_catalog::frame_slots(span, &frames);
    let cut: Vec<usize> = ordered
        .iter()
        .enumerate()
        .filter(|(logical, physical)| {
            let record = &records[**physical];
            !record.cut
                && (ids.contains(&record.id)
                    || slots.get(*logical).is_some_and(|slot| windows.iter().any(|w| slot.overlaps(*w))))
        })
        .map(|(_, &physical)| physical)
        .collect();
    if cut.is_empty() {
        return Ok(PurgedFile::Untouched { cut: already });
    }
    Ok(PurgedFile::Rewritten(crate::chunk::rewrite_without(path, &cut)?))
}

/// A ring batch's records, in order (`count` records from `first_seq`), or `None` if the ring lapped any of them
/// first: a batch is one payload, so a partial one is a gap.
///
/// # Errors
///
/// [`ServiceError::Lost`] when the ring was carved again (the broker restarted): the run ends and subscribes anew.
fn read_batch(
    reader: &mut continuum_shm::RingReader,
    first_seq: u64,
    count: u32,
) -> Result<Option<Vec<u8>>, ServiceError> {
    let stride = reader.stride();
    let mut bytes = vec![0u8; stride * count as usize];
    reader.seek(first_seq);
    for (i, record) in bytes.chunks_exact_mut(stride).enumerate() {
        match reader.read(record) {
            Ok(Some(r)) if r.lost == 0 && r.seq == first_seq + i as u64 => {}
            Ok(_) => return Ok(None),
            Err(continuum_shm::Error::Stale) => return Err(ServiceError::Lost("a ring was carved again".into())),
            Err(e) => return Err(ServiceError::Lost(format!("reading a ring: {e}"))),
        }
    }
    Ok(Some(bytes))
}

/// A slot's bytes, copied under a lease of its own (so they can't change mid-copy), or `None` if the slot is gone.
/// A copy that outlived its lease (a stall) is tried once more under a fresh one.
fn read_slot(arena: &Arena, handle: SlotHandle) -> Option<Vec<u8>> {
    for _ in 0..2 {
        let lease = arena.lease(handle, Duration::from_secs(1)).ok()?;
        let mut bytes = vec![0; lease.len()];
        match lease.copy_to(&mut bytes) {
            Ok(()) => return Some(bytes),
            Err(continuum_shm::Error::Expired) => {}
            Err(_) => return None,
        }
    }
    None
}

pub(crate) fn now_utc_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_lost_run_or_an_absent_broker_rides_through_and_a_chunk_file_problem_ends_the_program() {
        assert!(rides_through(&ServiceError::Lost("a ring batch with no ring".into())));
        assert!(rides_through(&ServiceError::Broker(continuum_client::Error::Protocol("closed".into()))));
        assert!(rides_through(&ServiceError::NoArena));
        let io = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "chunk dir");
        assert!(!rides_through(&ServiceError::Tier(crate::TierError::Io(io))));
    }

    #[test]
    fn the_wait_between_runs_doubles_while_runs_end_quickly_and_starts_over_after_a_settled_one() {
        let quick = Duration::from_secs(1);
        let mut backoff = Backoff::new();
        let waits: Vec<u64> = (0..8).map(|_| backoff.after(quick).as_millis() as u64).collect();
        assert_eq!(waits, [250, 500, 1_000, 2_000, 4_000, 8_000, 10_000, 10_000]);
        assert_eq!(backoff.after(Backoff::SETTLED), Backoff::FIRST);
        assert_eq!(backoff.after(quick), Duration::from_millis(500));
    }

    #[test]
    fn a_slow_start_names_each_phase_and_a_quick_one_says_nothing() {
        let began = Instant::now();
        let at = |ms: u64| began + Duration::from_millis(ms);
        let phases = Phases {
            began,
            last: at(3_200),
            took: vec![("connect", Duration::from_millis(40)), ("register", Duration::from_millis(3_160))],
        };
        assert_eq!(
            phases.slow(Duration::from_secs(1)).as_deref(),
            Some("startup took 3.2 s: connect 0.0 s, register 3.2 s")
        );
        assert_eq!(phases.slow(Duration::from_secs(5)), None);
    }

    fn provenance_of(epoch: u32, capture_mono_ns: u64, arrival_ns: u64) -> Option<continuum_catalog::FrameProvenance> {
        use continuum_core::schema::continuum as fb;
        let mut b = flatbuffers::FlatBufferBuilder::new();
        let args = fb::CaptureInfoArgs {
            device_seq: 12,
            capture_epoch: epoch,
            capture_mono_ns,
            arrival_ns,
            ..Default::default()
        };
        let c = fb::CaptureInfo::create(&mut b, &args);
        b.finish(c, None);
        frame_provenance(flatbuffers::root::<fb::CaptureInfo>(b.finished_data()).unwrap())
    }

    #[test]
    fn a_header_attests_provenance_only_with_an_epoch_and_flags_an_arrival_stamp() {
        assert_eq!(provenance_of(0, 5, 9), None, "a producer before minor 27 said nothing");
        let device = continuum_catalog::FrameProvenance { device_seq: 12, epoch: 3, stamped_at_arrival: false };
        assert_eq!(provenance_of(3, 5, 9), Some(device));
        assert_eq!(
            provenance_of(3, 9, 9),
            Some(continuum_catalog::FrameProvenance { stamped_at_arrival: true, ..device })
        );
        assert_eq!(provenance_of(3, 0, 0), Some(device), "no arrival time is no arrival stamp");
    }

    #[test]
    fn full_subscription_intake_keeps_copy_rpcs_and_new_stream_subscriptions_independent() {
        use continuum_broker::server::{Config, Server};
        use continuum_catalog::{Catalog, MemoryCatalog, NewPayload, StreamInfo, StreamKind};
        use continuum_client::{StreamSpec, StreamStorage};
        use continuum_core::{DomainLayer, HeaderBuf, HeaderSpec};

        const ROWS: u32 = 9_000; // More than the service queue plus one SDK subscription queue.
        let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(3).enable_all().build().unwrap();
        let directory = tempfile::tempdir().unwrap();
        let socket = directory.path().join("broker.sock");
        let catalog = MemoryCatalog::new();
        let first_name = "derived.first";
        let second_name = "derived.second";
        let first = catalog
            .register_stream(StreamInfo {
                name: first_name.into(),
                kind: StreamKind::Derived,
                layer: DomainLayer::Layer3Interpretation,
                format: PayloadFormat::RawBytes,
                shape_name: "synthetic".into(),
                shape_version: 1,
                rate_hz: None,
                anchor: None,
            })
            .unwrap();
        let headers: Vec<_> = (0..ROWS)
            .map(|sequence| {
                let mut spec = HeaderSpec::raw(
                    Uuid(uuid::Uuid::now_v7().into_bytes()),
                    first,
                    100 + u64::from(sequence),
                    100 + u64::from(sequence),
                );
                spec.seq = sequence;
                spec.layer = DomainLayer::Layer3Interpretation;
                spec.shape_name = "synthetic".into();
                spec.shape_version = 1;
                HeaderBuf::build(&spec).unwrap()
            })
            .collect();
        let payload_id = headers[0].id();
        let copy_dir = directory.path().join("synthetic-copy");
        std::fs::create_dir_all(&copy_dir).unwrap();
        let location = crate::copies::write(&copy_dir, &headers[0], &[1]).unwrap().location;
        let first_last = catalog
            .insert(headers.into_iter().map(|header| NewPayload { header, location: None, coverage: None }).collect())
            .unwrap()
            .last()
            .unwrap()
            .seq;
        let mut config = Config::new(socket.clone(), Duration::from_secs(60));
        config.tier0 = Some(vec![continuum_sched::SlotClass { size_bytes: 16_384, count: 4 }]);
        let server = rt.block_on(Server::bind_with_catalog(config, Box::new(catalog))).unwrap();
        let core = server.core();
        let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
        let broker = rt.spawn(server.run_until(async {
            let _ = stopped.await;
        }));
        let settings = Settings {
            socket: socket.clone(),
            chunk_dir: directory.path().join("tier"),
            data_root: None,
            streams: vec![first_name.into()],
            flush: FlushSettings::default(),
            tick: Duration::from_millis(1),
        };
        let mut service = Service::start(settings.clone(), rt.handle()).unwrap();
        // Hold service intake without dropping or skipping events. The forwarder fills the bounded queue,
        // then this stream's dedicated SDK connection applies backpressure to its remaining catch-up.
        let deadline = Instant::now() + Duration::from_secs(30);
        rt.block_on(async {
            while service.events.capacity() != 0 {
                assert!(Instant::now() < deadline, "first stream never filled the bounded intake queue");
                tokio::task::yield_now().await;
            }
        });
        let writer = Arc::clone(&service.work.outputs().client);
        rt.block_on(writer.commit_payload_copy(payload_id, &location))
            .expect("copy acknowledgement must not wait for subscription intake");
        let queried = rt
            .block_on(
                writer.query_all(&continuum_client::QuerySpec::hit_only(first_name, (100, 100 + u64::from(ROWS)))),
            )
            .unwrap();
        assert_eq!(queried.items.len(), ROWS as usize);
        assert_eq!(queried.items[0].locator.as_ref(), Some(&location));

        // Discover a second stream while the first is still full. Its initial Subscribe must use a new
        // connection, rather than awaiting a reply behind the first stream's unconsumed events.
        let producer = rt.block_on(Client::connect(&socket, ClientOptions::new("second-stream-producer"))).unwrap();
        let second = StreamId(
            rt.block_on(producer.register_stream(&StreamSpec {
                name: second_name.into(),
                kind: continuum_core::schema::continuum::StreamKind::Derived,
                format: PayloadFormat::RawBytes,
                shape_name: "synthetic".into(),
                shape_version: 1,
                nominal_rate_hz: 0.0,
                storage: StreamStorage::Slab { class: 0 },
                anchor: None,
                layer: DomainLayer::Layer3Interpretation,
            }))
            .unwrap()
            .stream_id,
        );
        let mut second_last = 0;
        for sequence in 0..ROWS {
            let mut spec = HeaderSpec::raw(
                Uuid(uuid::Uuid::now_v7().into_bytes()),
                second,
                100 + u64::from(sequence),
                100 + u64::from(sequence),
            );
            spec.seq = sequence;
            spec.layer = DomainLayer::Layer3Interpretation;
            spec.shape_name = "synthetic".into();
            spec.shape_version = 1;
            let header = HeaderBuf::build(&spec).unwrap();
            if sequence + 1 == ROWS {
                second_last = rt.block_on(producer.ingest_acked(&header, None)).unwrap();
            } else {
                producer.ingest(&header, None).unwrap();
            }
        }
        service.subscribe_missing().unwrap();
        let second_index = service.settings.streams.iter().position(|name| name == second_name).unwrap();
        assert!(
            service.work.outputs().subs[second_index].is_some(),
            "the second stream must subscribe while the first remains full"
        );
        let expected = BTreeMap::from([(first_name, first_last), (second_name, second_last)]);
        while expected.iter().any(|(name, last)| {
            let index = service.settings.streams.iter().position(|stream| stream == name).unwrap();
            service.work.outputs().subs[index].as_ref().is_none_or(|position| position.acked < *last)
        }) {
            assert!(Instant::now() < deadline, "bounded catch-up did not drain both streams");
            service.step().expect("passive intake must not fail closed while writer RPCs continue independently");
        }
        // Test-only fences after draining: each heartbeat follows that connection's queued one-way ACKs,
        // then the core fence waits for their durable cursor writes before reopening the service.
        for reader in &service.readers {
            rt.block_on(reader.heartbeat(None)).unwrap();
        }
        rt.block_on(core.sync_writer()).unwrap();
        service.shutdown().unwrap();
        assert!(matches!(rt.block_on(writer.list_streams()), Err(continuum_client::Error::Closed)));
        for reader in &service.readers {
            assert!(matches!(rt.block_on(reader.heartbeat(None)), Err(continuum_client::Error::Closed)));
        }
        let mut restarted = Service::start(settings, rt.handle()).unwrap();
        let saved: BTreeMap<_, _> = restarted
            .settings
            .streams
            .iter()
            .zip(&restarted.work.outputs().subs)
            .filter_map(|(name, position)| position.as_ref().map(|position| (name.as_str(), position.acked)))
            .collect();
        assert!(saved[first_name] >= first_last, "first stream lost its durable progress");
        assert!(saved[second_name] >= second_last, "second stream lost its durable progress");
        restarted.shutdown().unwrap();
        rt.block_on(producer.close("test finished")).unwrap();
        stop.send(()).unwrap();
        rt.block_on(broker).unwrap();
    }

    #[test]
    fn the_manifest_names_its_cameras_video_streams() {
        let text = "[[device]]\nname = \"facecam\"\nkind = \"camera\"\nmatch = { usb = \"0fd9:0078\" }\n\n\
                    [[device]]\nname = \"wave3\"\nkind = \"microphone\"\nmatch = { name = \"Wave 3\" }\n";
        assert_eq!(manifest_streams(text).unwrap(), vec!["facecam.video".to_owned(), "wave3.audio".to_owned()]);
        assert!(manifest_streams("not toml [").is_err());
        assert!(manifest_streams("").unwrap().is_empty());
    }

    /// The broker sends an unacknowledged notice again every minute, and each copy used to queue another
    /// full pass over every chunk file; purges arriving while one waits now share its pass.
    #[test]
    fn a_resent_notice_is_not_queued_again_and_waiting_purges_share_one_pass() {
        use continuum_broker::server::{Config, Server};
        let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("broker.sock");
        let mut config = Config::new(socket.clone(), Duration::from_secs(30));
        config.tier0 = Some(vec![continuum_sched::SlotClass { size_bytes: 16_384, count: 4 }]);
        let server = rt.block_on(Server::bind(config)).unwrap();
        let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
        let broker = rt.spawn(server.run_until(async {
            let _ = stopped.await;
        }));
        let chunk_dir = dir.path().join("tier");
        std::fs::create_dir_all(&chunk_dir).unwrap();
        for name in ["1-a.cnk", "2-b.cnk"] {
            let stream = StreamId(name[..1].parse().unwrap());
            let hour = crate::layout::hour_dir(&chunk_dir, stream, 0);
            crate::layout::make_hour_dir(&chunk_dir, &hour).unwrap();
            std::fs::write(hour.join(name), b"never opened by this test").unwrap();
        }
        let mut service = Service::start(
            Settings {
                socket: socket.clone(),
                chunk_dir: chunk_dir.clone(),
                data_root: None,
                streams: Vec::new(),
                flush: FlushSettings::default(),
                tick: Duration::from_millis(10),
            },
            rt.handle(),
        )
        .unwrap();
        let span =
            |stream: u32, start: u64| continuum_client::PurgedSpan { stream_id: stream, range: (start, start + 10) };
        let (a, b, c) = (Uuid([0xa; 16]), Uuid([0xb; 16]), Uuid([0xc; 16]));
        service.begin_purge(a, vec![Uuid([1; 16])], &[span(1, 0)]).unwrap();
        service.begin_purge(a, vec![Uuid([1; 16])], &[span(1, 0)]).unwrap();
        assert_eq!(service.purge_jobs.len(), 1, "a resent notice must not queue a second pass");
        service.begin_purge(b, vec![Uuid([2; 16])], &[span(1, 100), span(2, 5)]).unwrap();
        assert_eq!(service.purge_jobs.len(), 1, "a purge arriving before the pass starts joins it");
        let job = &service.purge_jobs[0];
        assert_eq!(job.purge_ids, [a, b]);
        assert_eq!(job.ids, BTreeSet::from([Uuid([1; 16]), Uuid([2; 16])]));
        let windows = |s: u32| job.windows[&StreamId(s)].iter().map(|w| (w.start_ns(), w.end_ns())).collect::<Vec<_>>();
        assert_eq!((windows(1), windows(2)), (vec![(0, 10), (100, 110)], vec![(5, 15)]));
        assert_eq!(job.files.len(), 2, "each file is looked at once for both purges: {:?}", job.files);
        // Once the pass has begun, a new purge waits for the next one (files already passed never saw its ids),
        // and a resend of either is still not queued again.
        service.purge_jobs[0].started = true;
        service.begin_purge(c, vec![Uuid([3; 16])], &[span(2, 50)]).unwrap();
        service.begin_purge(b, vec![Uuid([2; 16])], &[span(1, 100), span(2, 5)]).unwrap();
        service.begin_purge(c, vec![Uuid([3; 16])], &[span(2, 50)]).unwrap();
        assert_eq!(service.purge_jobs.iter().map(|j| j.purge_ids.clone()).collect::<Vec<_>>(), [vec![a, b], vec![c]]);
        service.shutdown().unwrap();
        stop.send(()).unwrap();
        rt.block_on(broker).unwrap();
    }

    /// An age-out of an inline-only stream (one aged out every ten minutes) queued every chunk file, and the
    /// pass stalled Tier 0's reads until frames were lost. Only the purged streams' files are looked at now; ids the
    /// broker names no spans for could be anywhere, so those still reach every file.
    #[test]
    fn an_id_purge_looks_only_at_its_streams_files() {
        use continuum_broker::server::{Config, Server};
        let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("broker.sock");
        let mut config = Config::new(socket.clone(), Duration::from_secs(30));
        config.tier0 = Some(vec![continuum_sched::SlotClass { size_bytes: 16_384, count: 4 }]);
        let server = rt.block_on(Server::bind(config)).unwrap();
        let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
        let broker = rt.spawn(server.run_until(async {
            let _ = stopped.await;
        }));
        let chunk_dir = dir.path().join("tier");
        std::fs::create_dir_all(&chunk_dir).unwrap();
        for name in ["1-a.cnk", "1-b.cnk", "2-c.cnk"] {
            let stream = StreamId(name[..1].parse().unwrap());
            let hour = crate::layout::hour_dir(&chunk_dir, stream, 0);
            crate::layout::make_hour_dir(&chunk_dir, &hour).unwrap();
            std::fs::write(hour.join(name), b"never opened by this test").unwrap();
        }
        let mut service = Service::start(
            Settings {
                socket: socket.clone(),
                chunk_dir: chunk_dir.clone(),
                data_root: None,
                streams: Vec::new(),
                flush: FlushSettings::default(),
                tick: Duration::from_millis(10),
            },
            rt.handle(),
        )
        .unwrap();
        let span = |stream: u32| continuum_client::PurgedSpan { stream_id: stream, range: (0, 10) };
        let files = |service: &Service| {
            let mut names: Vec<String> = service.purge_jobs[0]
                .files
                .iter()
                .map(|(_, path)| path.file_name().unwrap().to_string_lossy().into_owned())
                .collect();
            names.sort();
            names
        };
        // An inline-only stream: the tier holds none of its files, so nothing is queued.
        service.begin_purge(Uuid([0xa; 16]), vec![Uuid([1; 16])], &[span(9)]).unwrap();
        assert_eq!(files(&service), Vec::<String>::new());
        service.purge_jobs.clear();
        service.begin_purge(Uuid([0xb; 16]), vec![Uuid([2; 16])], &[span(1)]).unwrap();
        assert_eq!(files(&service), ["1-a.cnk", "1-b.cnk"]);
        service.purge_jobs.clear();
        service.begin_purge(Uuid([0xc; 16]), vec![Uuid([3; 16])], &[]).unwrap();
        assert_eq!(files(&service), ["1-a.cnk", "1-b.cnk", "2-c.cnk"], "ids without spans could be anywhere");
        service.shutdown().unwrap();
        stop.send(()).unwrap();
        rt.block_on(broker).unwrap();
    }

    #[test]
    fn recovered_derived_copy_ids_do_not_accumulate_in_early_state() {
        recovered_copy_case(false);
    }

    #[cfg(unix)]
    #[test]
    fn recovery_conflict_via_alias_never_deletes_the_existing_warm_copy() {
        recovered_copy_case(true);
    }

    fn recovered_copy_case(alias_reopen: bool) {
        use continuum_broker::server::{Config, Server};
        use continuum_client::{StreamSpec, StreamStorage};
        use continuum_core::{HeaderBuf, HeaderSpec};
        let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("broker.sock");
        let mut config = Config::new(socket.clone(), Duration::from_secs(30));
        config.tier0 = Some(vec![continuum_sched::SlotClass { size_bytes: 16384, count: 4 }]);
        let server = rt.block_on(Server::bind(config)).unwrap();
        let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
        let running = rt.spawn(server.run_until(async {
            let _ = stopped.await;
        }));
        let producer = rt.block_on(Client::connect(&socket, ClientOptions::new("copy-recovery-test"))).unwrap();
        let stream = StreamId(
            rt.block_on(producer.register_stream(&StreamSpec {
                name: "derived.recovery".into(),
                kind: continuum_core::schema::continuum::StreamKind::Derived,
                format: PayloadFormat::RawBytes,
                shape_name: "opaque".into(),
                shape_version: 1,
                nominal_rate_hz: 0.0,
                storage: StreamStorage::Slab { class: 0 },
                anchor: None,
                layer: continuum_core::DomainLayer::Layer3Interpretation,
            }))
            .unwrap()
            .stream_id,
        );
        let chunk_dir = dir.path().join("tier");
        std::fs::create_dir_all(&chunk_dir).unwrap();
        let mut last = 0;
        for sequence in 0..3 {
            let mut spec = HeaderSpec::raw(Uuid(uuid::Uuid::now_v7().into_bytes()), stream, 100, 200);
            spec.layer = continuum_core::DomainLayer::Layer3Interpretation;
            spec.seq = sequence;
            let header = HeaderBuf::build(&spec).unwrap();
            last = rt.block_on(producer.ingest_acked(&header, None)).unwrap();
            crate::copies::write(&chunk_dir, &header, &[73; 100]).unwrap();
        }
        let mut service = Service::start(
            Settings {
                socket: socket.clone(),
                chunk_dir: chunk_dir.clone(),
                data_root: None,
                streams: vec![],
                flush: FlushSettings::default(),
                tick: Duration::from_millis(5),
            },
            rt.handle(),
        )
        .unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        while !service.work.outputs().subs.iter().flatten().any(|sub| sub.acked >= last) {
            assert!(Instant::now() < deadline, "saved derived records were not acknowledged");
            service.step().unwrap();
        }
        assert!(
            service.work.outputs().early.is_empty(),
            "already-acknowledged copy IDs must not remain cached for the archive's lifetime"
        );
        service.shutdown().unwrap();
        assert!(
            crate::copies::recover(&chunk_dir).unwrap().is_empty(),
            "acknowledged copies are not reopened and committed again at the next start"
        );
        #[cfg(unix)]
        if alias_reopen {
            let alias = dir.path().join("alias");
            std::os::unix::fs::symlink(&chunk_dir, &alias).unwrap();
            for (_, path) in chunk_files(&chunk_dir, None).unwrap() {
                // Their marks lost, so recovery attaches them again under the alias's URI.
                std::fs::File::open(&path).unwrap().set_modified(std::time::SystemTime::now()).unwrap();
            }
            let before: Vec<_> = chunk_files(&chunk_dir, None)
                .unwrap()
                .into_iter()
                .map(|(_, path)| {
                    let bytes = std::fs::read(&path).unwrap();
                    (path, bytes)
                })
                .collect();
            let reopened = Service::start(
                Settings {
                    socket: socket.clone(),
                    chunk_dir: alias,
                    data_root: None,
                    streams: vec![],
                    flush: FlushSettings::default(),
                    tick: Duration::from_millis(5),
                },
                rt.handle(),
            );
            for (path, bytes) in before {
                assert_eq!(
                    std::fs::read(&path).ok(),
                    Some(bytes),
                    "a different URI can refer to the same live warm file; Conflict does not authorize deletion"
                );
            }
            assert!(
                matches!(reopened, Err(ServiceError::Broker(error)) if error.broker_code() == Some(ErrorCode::Conflict)),
                "conflicting copy URI must remain actionable rather than be silently ignored"
            );
        }
        rt.block_on(producer.close("done")).unwrap();
        stop.send(()).unwrap();
        rt.block_on(running).unwrap();
    }
}
