//! The broker's Unix socket: binding it safely, and one session task per connection.
//!
//! A session starts with the client's `Hello`; the broker answers `Welcome` (or `Error` with `IncompatibleSchema`
//! and closes). After that the client can send `Heartbeat` (answered with the broker's clock) and `Goodbye` (the
//! broker closes). A session that sends nothing for the heartbeat timeout is closed, which is how a client that hung
//! or died without closing its socket is noticed; so is one whose replies go unread for that long (a write that
//! can't finish). `Goodbye` is accepted before `Hello` too: it closes without a reply. Every other message is
//! answered with an `Error` until the catalog, scheduler, and data plane are wired in.
//!
//! Binding: two brokers must never share one socket. The broker holds an exclusive lock on `<socket>.lock` for its
//! whole life and refuses to start when another process holds it, so two brokers started together can't both pass
//! the checks and then unlink each other's socket. With the lock held, a socket file that refuses connections is a
//! stale one from a broker that died, and it is replaced; any other connect error leaves the file alone and fails.
//! A path that exists and isn't a socket is never deleted. The socket's directory is made private (0700) when the
//! broker creates it, and the socket and lock file are 0600. The lock file stays when the broker exits (removing it
//! would reopen the race it closes).
//!
//! Sessions are capped at [`Config::max_sessions`]: a connection past the cap gets `Error` with `Unavailable` and is
//! closed, so a runaway client can't exhaust the process's file descriptors and memory. An `accept` failure (out of
//! file descriptors, say) waits [`ACCEPT_BACKOFF`] before the next try instead of spinning.

use std::ffi::OsString;
use std::fs::{File, OpenOptions, TryLockError};
use std::future::Future;
use std::io;
use std::os::unix::fs::{FileTypeExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use continuum_catalog::{Catalog, MemoryCatalog};
use continuum_core::schema::continuum::{ControlPayload, ErrorCode};
use flatbuffers::FlatBufferBuilder;
use tokio::net::unix::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc};
use tokio::task::JoinSet;
use tracing::{debug, info, warn};

use crate::core::{self, Command, CoreHandle};
use crate::protocol::{self, SCHEMA_MAJOR};
use crate::tier0::Tier0;
use continuum_wire::{FrameError, read_frame, write_frame, write_frame_with_fds};

/// The longest socket path both platforms accept: `sun_path` is 104 bytes on macOS and 108 on Linux, including the
/// terminating zero.
pub const MAX_SOCKET_PATH: usize = 103;

/// How many catalog rows the startup index load reads per page.
const INDEX_PAGE: usize = 1024;

/// How long a clean stop waits for the catalog's calls in flight before it gives up on closing the file (well inside
/// the 20 s the service manager allows a stop).
const CLOSE_WAIT: Duration = Duration::from_secs(10);

/// How long a clean stop may spend fencing, encoding, and admitting the restart projection, between the writer drain
/// and the catalog close (each of which keeps its own `CLOSE_WAIT`). Encoding and publishing ran about 2 s for 136 MB;
/// this covers a projection near its 2 GiB cap. The broker's `stop_timeout` in services.toml covers all three.
const CHECKPOINT_WAIT: Duration = Duration::from_secs(45);

/// Yield between transient shutdown-fence refusals while session-close commands settle.
const CHECKPOINT_RETRY_INTERVAL: Duration = Duration::from_millis(20);

/// Retry transient Busy refusals within one shared deadline, including time spent waiting
/// for an individual attempt; permanent errors and successful ineligibility
/// return immediately. Cancellation drops the pending attempt rather than starting another one.
async fn wait_for_restart_checkpoint<T, F, Fut>(
    deadline: tokio::time::Instant,
    mut attempt: F,
) -> Result<Result<T, core::CoreError>, tokio::time::error::Elapsed>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, core::CoreError>>,
{
    tokio::time::timeout_at(deadline, async {
        loop {
            match attempt().await {
                Err(core::CoreError::Busy) => tokio::time::sleep(CHECKPOINT_RETRY_INTERVAL).await,
                result => return result,
            }
        }
    })
    .await
}

/// How many sessions a broker serves at once by default.
pub const DEFAULT_MAX_SESSIONS: usize = 256;

/// How long the accept loop waits after `accept` fails before it tries again.
pub const ACCEPT_BACKOFF: Duration = Duration::from_millis(100);

/// How the server runs.
#[derive(Clone, Debug)]
pub struct Config {
    /// Where the Unix socket goes.
    pub socket: PathBuf,
    /// A session that sends nothing, or leaves a reply unread, for this long is closed.
    pub heartbeat_timeout: Duration,
    /// The most sessions open at once; a connection past it is refused.
    pub max_sessions: usize,
    /// How the catalog writer sizes the steps of long jobs.
    pub writer: core::WriterConfig,
    /// The machine profile the scheduler budgets for (`mac_mini_64gb` or `linux_dual_3090`).
    pub profile: String,
    /// The Tier 0 arena's slot classes; `None` takes the profile's.
    pub tier0: Option<Vec<continuum_sched::SlotClass>>,
    /// The redb catalog file; `None` keeps the catalog in memory (tests).
    pub catalog: Option<PathBuf>,
    /// The bytes reserved in the arena for rings carved at registration (PCM, IMU).
    pub tier0_ring_bytes: u64,
    /// How purges wait for the workers that keep their own copies.
    pub forget: core::ForgetConfig,
    /// When the catalog file is scrubbed into a fresh one (a file-backed catalog only).
    pub scrub: crate::scrub::ScrubConfig,
}

/// The ring region's default size: 64 MiB, room for about sixteen 10 second rings of 48 kHz audio.
pub const DEFAULT_RING_BYTES: u64 = 64 << 20;

/// The profile for this machine when none is set: the Mac's on macOS, the Linux rig's elsewhere.
#[must_use]
pub fn default_profile() -> &'static str {
    if cfg!(target_os = "macos") { "mac_mini_64gb" } else { "linux_dual_3090" }
}

impl Config {
    /// A config for `socket` with the default session cap.
    #[must_use]
    pub fn new(socket: PathBuf, heartbeat_timeout: Duration) -> Self {
        Self {
            socket,
            heartbeat_timeout,
            max_sessions: DEFAULT_MAX_SESSIONS,
            writer: core::WriterConfig::default(),
            profile: default_profile().to_owned(),
            tier0: None,
            tier0_ring_bytes: DEFAULT_RING_BYTES,
            catalog: None,
            forget: core::ForgetConfig::default(),
            scrub: crate::scrub::ScrubConfig::default(),
        }
    }
}

/// The lock file beside a socket: `<socket>.lock`.
#[must_use]
pub fn lock_path(socket: &Path) -> PathBuf {
    let mut name = OsString::from(socket.as_os_str());
    name.push(".lock");
    PathBuf::from(name)
}

/// Why the server couldn't start.
#[derive(Debug, thiserror::Error)]
pub enum BindError {
    /// Another broker holds the socket's lock, or answers on the socket.
    #[error("another broker is already running on {0}")]
    AlreadyRunning(PathBuf),
    /// Something that isn't a socket is in the way.
    #[error("{0} exists and is not a socket; refusing to replace it")]
    NotASocket(PathBuf),
    /// The path is too long for a Unix socket on macOS.
    #[error("socket path {0} is {1} bytes; the limit is {MAX_SOCKET_PATH} (pass a shorter --socket)")]
    PathTooLong(PathBuf, usize),
    /// The catalog couldn't be read at startup (the availability index loads before the socket opens).
    #[error("{0}")]
    Catalog(String),
    /// The Tier 0 arena couldn't be created.
    #[error("{0}")]
    Arena(String),
    /// Creating the directory, locking, probing, or binding failed.
    #[error("{0}: {1}")]
    Io(PathBuf, #[source] io::Error),
}

/// A bound broker socket. Dropping it removes the socket file, then releases the lock.
#[derive(Debug)]
pub struct Server {
    listener: UnixListener,
    config: Config,
    /// Held for the server's life; dropped (unlocked) after `Drop` removes the socket.
    _lock: File,
    /// The core thread's handle; the core stops when the server and every session have dropped theirs.
    core: CoreHandle,
    /// The Tier 0 arena; unlinked when the server and every session have dropped it.
    tier0: Arc<Tier0>,
    /// The catalog scrubber's thread: stopped and joined by a clean stop, and otherwise when the server drops it.
    scrubber: Option<crate::scrub::Scrubber>,
    /// The catalog the core owns, kept here so that a clean stop can close its file (`Catalog::close`).
    catalog: Closing,
}

impl Server {
    /// Binds the socket, replacing a stale one (see the module docs), with the redb catalog at `config.catalog`
    /// (opened once the lock is held, so a second broker is refused before it touches the file), or an in-memory
    /// catalog when that's `None` (tests: the broker then forgets everything when it stops).
    ///
    /// # Errors
    ///
    /// A [`BindError`] if another broker is running, the path is unusable, the catalog can't be opened, or binding
    /// fails.
    pub async fn bind(config: Config) -> Result<Self, BindError> {
        Self::bind_inner(config, None).await
    }

    /// Binds the socket like [`Server::bind`], with the core owning `catalog`.
    ///
    /// # Errors
    ///
    /// As [`Server::bind`].
    pub async fn bind_with_catalog(config: Config, catalog: Box<dyn Catalog>) -> Result<Self, BindError> {
        Self::bind_inner(config, Some(catalog)).await
    }

    async fn bind_inner(config: Config, catalog: Option<Box<dyn Catalog>>) -> Result<Self, BindError> {
        let startup_started = std::time::Instant::now();
        let path = config.socket.clone();
        let len = path.as_os_str().len();
        if len > MAX_SOCKET_PATH {
            return Err(BindError::PathTooLong(path, len));
        }
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty() && !d.exists()) {
            std::fs::create_dir_all(dir).map_err(|e| BindError::Io(dir.to_path_buf(), e))?;
            std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))
                .map_err(|e| BindError::Io(dir.to_path_buf(), e))?;
        }
        let lock_file = lock_path(&path);
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .open(&lock_file)
            .map_err(|e| BindError::Io(lock_file.clone(), e))?;
        match lock.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => return Err(BindError::AlreadyRunning(path)),
            Err(TryLockError::Error(e)) => return Err(BindError::Io(lock_file, e)),
        }
        let catalog: Arc<dyn Catalog> = match (catalog, config.catalog.clone()) {
            (Some(c), _) => Arc::from(c),
            (None, None) => Arc::new(MemoryCatalog::new()),
            (None, Some(file)) => {
                let opened = tokio::task::spawn_blocking(move || continuum_catalog::RedbCatalog::open(&file))
                    .await
                    .map_err(|e| BindError::Catalog(format!("opening the catalog stopped: {e}")))?;
                let opened = opened.map_err(|e| BindError::Catalog(format!("opening the catalog: {e}")))?;
                let phases = opened.open_timings();
                info!(took = ?phases.total, scrub_recovery = ?phases.scrub_recovery,
                    database_open = ?phases.database_open, schema_setup = ?phases.schema_setup,
                    recording_repair = ?phases.recording_repair, "opened the catalog");
                if !opened.recording_restored().is_empty() {
                    info!(
                        streams = ?opened.recording_restored(),
                        "recording history restored: these streams were marked unknown with no original left as the \
                         reason (#1669)"
                    );
                }
                for (stream, why) in opened.recording_held() {
                    info!(stream, why = %why, "recording history still unknown: the repair held this stream back (#1669)");
                }
                Arc::new(opened)
            }
        };
        // The scheduler must never see a partial index, so the whole index loads before the socket accepts anything.
        let (index, pending, keepers) = {
            let catalog = Arc::clone(&catalog);
            tokio::task::spawn_blocking(move || {
                let started = std::time::Instant::now();
                let pending_started = std::time::Instant::now();
                let pending = catalog.pending_purges()?;
                let pending_read = pending_started.elapsed();
                let cached = if pending.is_empty() {
                    match catalog.load_restart_projection(continuum_catalog::MAX_RESTART_PROJECTION_BYTES)? {
                        Ok(cached) => Some(cached),
                        Err(reason) => {
                            info!(%reason, "no usable restart projection; rebuilding the availability index");
                            None
                        }
                    }
                } else {
                    info!(
                        pending = pending.len(),
                        "pending purges; rebuilding the availability index instead of restoring its projection"
                    );
                    None
                };
                let restore_started = std::time::Instant::now();
                let restored = cached.and_then(|cached| {
                    match crate::index::AvailabilityIndex::restore_restart_projection(&cached.bytes) {
                        Ok(index) => Some((index, cached.hot_ids)),
                        Err(error) => {
                            warn!(%error, "restart projection is invalid; rebuilding the availability index");
                            None
                        }
                    }
                });
                let restore_time = restore_started.elapsed();
                let (index, stats, phases, hot_ids) = if let Some((index, hot_ids)) = restored {
                    (index, None, crate::index::LoadTimings::default(), hot_ids)
                } else {
                    let (index, stats, phases) = crate::index::load_profiled(&*catalog, INDEX_PAGE)?;
                    let hot_ids = stats.stale_tier0.clone();
                    (index, Some(stats), phases, hot_ids)
                };
                // Before the core starts, so no ingest or purge interleaves with it.
                let phase_started = std::time::Instant::now();
                let cleared = crate::index::clear_stale_tier0(&*catalog, &hot_ids)?;
                let stale_clear = phase_started.elapsed();
                let phase_started = std::time::Instant::now();
                let keepers = catalog.keepers()?;
                let keepers_read = phase_started.elapsed();
                let _ = catalog.restart_stamp()?;
                Ok::<_, continuum_catalog::CatalogError>((
                    (index, stats, phases, hot_ids.len(), restore_time),
                    cleared,
                    pending,
                    keepers,
                    (stale_clear, pending_read, keepers_read),
                    started.elapsed(),
                ))
            })
            .await
            .map_err(|e| BindError::Catalog(format!("the index load stopped: {e}")))?
            .map(|((index, stats, phases, hot_count, restore_time), cleared, pending, keepers, auxiliary, took)| {
                if let Some(stats) = stats {
                info!(
                    ?took,
                    index_load = ?phases.total, stream_reads = ?phases.stream_reads, payload_reads = ?phases.payload_reads,
                    payload_apply = ?phases.payload_apply, coverage_load = ?phases.coverage_load,
                    stale_clear = ?auxiliary.0, pending_read = ?auxiliary.1, keepers_read = ?auxiliary.2,
                    streams = stats.streams,
                    payloads = stats.payloads,
                    segments = stats.segments,
                    coverage_intervals = stats.coverage_intervals,
                    stale_tier0 = stats.stale_tier0.len(),
                    stale_tier0_cleared = cleared,
                    pending_purges = pending.len(),
                    keepers = keepers.len(),
                    "loaded the availability index"
                );
                } else {
                    let (segments, coverage_intervals) = index.size();
                    info!(?took, ?restore_time, segments, coverage_intervals,
                        stale_tier0 = hot_count, stale_tier0_cleared = cleared,
                        stale_clear = ?auxiliary.0, pending_read = ?auxiliary.1, keepers_read = ?auxiliary.2,
                        pending_purges = pending.len(), keepers = keepers.len(),
                        "restored the availability index from its clean-stop projection");
                }
                (index, pending, keepers)
            })
            .map_err(|e| BindError::Catalog(format!("loading the availability index: {e}")))?
        };
        match std::fs::symlink_metadata(&path) {
            Ok(meta) if !meta.file_type().is_socket() => return Err(BindError::NotASocket(path)),
            Ok(_) => match UnixStream::connect(&path).await {
                // Something answers without holding the lock: a broker from before the lock file, still running.
                Ok(_) => return Err(BindError::AlreadyRunning(path)),
                Err(e) if e.kind() == io::ErrorKind::ConnectionRefused => {
                    info!(path = %path.display(), "replacing a stale socket");
                    std::fs::remove_file(&path).map_err(|e| BindError::Io(path.clone(), e))?;
                }
                Err(e) => return Err(BindError::Io(path, e)),
            },
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(BindError::Io(path, e)),
        }
        let profile = continuum_sched::Profile::named(&config.profile)
            .ok_or_else(|| BindError::Catalog(format!("no machine profile named {}", config.profile)))?;
        let classes = config.tier0.clone().unwrap_or_else(|| profile.slot_classes.clone());
        let arena_started = std::time::Instant::now();
        let tier0 = Arc::new(Tier0::create(&path, &classes, config.tier0_ring_bytes).map_err(BindError::Arena)?);
        info!(took = ?arena_started.elapsed(), "initialized the Tier 0 arena");
        let listener = UnixListener::bind(&path).map_err(|e| BindError::Io(path.clone(), e))?;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))
            .map_err(|e| BindError::Io(path.clone(), e))?;
        info!(path = %path.display(), "listening");
        let catalog_for_startup = Arc::clone(&catalog);
        let core = core::spawn(
            catalog,
            core::DEFAULT_READERS,
            config.writer,
            index,
            pending,
            keepers,
            profile,
            Arc::clone(&tier0),
            config.forget,
        );
        // What an earlier run learned, and this boot's id, read off the scheduler thread (a file read, and on
        // macOS a sysctl call) and handed over before anything else reaches the core.
        let startup = {
            let catalog = Arc::clone(&catalog_for_startup);
            tokio::task::spawn_blocking(move || {
                let costs = match catalog.load_cost_models() {
                    Ok(Some(bytes)) => crate::costs::decode(&bytes).unwrap_or_else(|| {
                        warn!("the saved cost models aren't readable; starting from the manifests' estimates");
                        Vec::new()
                    }),
                    Ok(None) => Vec::new(),
                    Err(e) => {
                        warn!(error = %e, "couldn't load the saved cost models");
                        Vec::new()
                    }
                };
                let boot = crate::host::boot_id()
                    .inspect_err(|e| warn!(error = %e, "couldn't read the boot id; no clock samples"))
                    .ok();
                (costs, boot)
            })
            .await
            .unwrap_or_default()
        };
        core.send(core::Command::Startup { costs: startup.0, boot: startup.1 })
            .await
            .map_err(|e| BindError::Catalog(format!("starting the core: {e}")))?;
        let scrubber = crate::scrub::spawn(Arc::clone(&catalog_for_startup), config.scrub.clone());
        info!(took = ?startup_started.elapsed(), "completed broker startup");
        Ok(Self { listener, config, _lock: lock, core, tier0, scrubber, catalog: Closing(catalog_for_startup) })
    }

    /// The socket's path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.config.socket
    }

    /// A handle to the core, for tests and diagnostics.
    #[must_use]
    pub fn core(&self) -> CoreHandle {
        self.core.clone()
    }

    /// Accepts connections until `shutdown` completes, then closes every session.
    pub async fn run_until(mut self, shutdown: impl Future<Output = ()>) {
        let mut sessions = JoinSet::new();
        let permits = Arc::new(Semaphore::new(self.config.max_sessions));
        tokio::pin!(shutdown);
        loop {
            tokio::select! {
                () = &mut shutdown => break,
                accepted = self.listener.accept() => match accepted {
                    Ok((stream, _)) => {
                        let timeout = self.config.heartbeat_timeout;
                        match Arc::clone(&permits).try_acquire_owned() {
                            Ok(permit) => {
                                let tier0 = Arc::clone(&self.tier0);
                                sessions.spawn(run_session(stream, timeout, self.core.clone(), tier0, permit));
                            }
                            Err(_) => {
                                warn!(max = self.config.max_sessions, "refusing a connection: too many sessions");
                                sessions.spawn(refuse(stream, timeout));
                            }
                        }
                    }
                    Err(e) => {
                        warn!(error = %e, "accept failed; retrying shortly");
                        tokio::time::sleep(ACCEPT_BACKOFF).await;
                    }
                },
                Some(done) = sessions.join_next(), if !sessions.is_empty() => {
                    if let Err(e) = done
                        && e.is_panic()
                    {
                        tracing::error!(error = %e, "a session panicked");
                    }
                }
            }
        }
        info!(open = sessions.len(), "shutting down");
        sessions.shutdown().await;
        // JoinSet shutdown aborts socket tasks, so their async Session::end may never send Close.
        // Admission is stopped and every task is gone: acknowledge normal cleanup in the core
        // before draining its resulting writes or asking for a session-free terminal fence.
        let drain_deadline = tokio::time::Instant::now() + CLOSE_WAIT;
        match tokio::time::timeout_at(drain_deadline, self.core.close_sessions()).await {
            Ok(Ok(())) => {}
            Ok(Err(error)) => warn!(%error, "couldn't clean up stopped sessions before closing the catalog"),
            Err(_) => warn!("session cleanup deadline expired before closing the catalog"),
        }
        self.drain_writer(drain_deadline).await;
        self.close_catalog().await;
    }

    /// Commits what the sessions handed over before they closed. A notice the broker has read is queued for the
    /// writer, not yet in the catalog; closing the catalog under it lost the last frames a producer sent before a
    /// stop. New work can't arrive (the sessions are gone), so one sync covers everything queued.
    async fn drain_writer(&self, deadline: tokio::time::Instant) {
        loop {
            match tokio::time::timeout_at(deadline, self.core.sync_writer()).await {
                Ok(Ok(())) => return,
                Ok(Err(core::CoreError::Busy)) if tokio::time::Instant::now() < deadline => {
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
                Ok(Err(e)) => {
                    warn!(error = %e, "couldn't commit the last queued writes before closing the catalog");
                    return;
                }
                Err(_) => {
                    warn!(wait = ?CLOSE_WAIT, "the writer didn't finish its queue before closing the catalog");
                    return;
                }
            }
        }
    }

    /// Closes the catalog's file before the process exits. The core's threads are never joined, and the process
    /// exits under them, so without this the redb file was left exactly as a crash leaves it on every stop, and every
    /// start ran redb's repair over the whole file, which can keep a multi-GiB catalog from opening in time.
    /// The scrubber stops first (a scrub in progress throws its fresh file away), then the catalog waits
    /// for the writes and reads in flight and closes; anything the core tries after that fails, moments before exit.
    async fn close_catalog(&mut self) {
        let started = std::time::Instant::now();
        // Maintenance must stop before the terminal writer fence. Admit optional checkpoint work
        // within the existing close budget, reserving time for redb's ordinary clean close.
        if let Some(scrubber) = self.scrubber.take()
            && let Err(error) = tokio::task::spawn_blocking(move || scrubber.stop()).await
        {
            warn!(%error, "stopping catalog maintenance failed; skipping the restart projection");
            self.finish_catalog_close(started).await;
            return;
        }
        self.save_restart_checkpoint(tokio::time::Instant::now() + CHECKPOINT_WAIT).await;
        self.finish_catalog_close(started).await;
    }

    async fn save_restart_checkpoint(&self, deadline: tokio::time::Instant) {
        let checkpoint = match wait_for_restart_checkpoint(deadline, || self.core.restart_checkpoint()).await {
            Ok(Ok(checkpoint)) => checkpoint,
            Ok(Err(error)) => {
                warn!(%error, "couldn't fence the restart projection; the next start rebuilds");
                return;
            }
            Err(_) => {
                warn!(
                    wait = ?CHECKPOINT_WAIT,
                    "sessions or reads were still settling when the restart projection's time ran out; the next start rebuilds"
                );
                return;
            }
        };
        let (index, stamp) = match checkpoint {
            Ok(checkpoint) => checkpoint,
            Err(reason) => {
                warn!(%reason, "skipped the restart projection; the next start rebuilds");
                return;
            }
        };
        // Encoding owns only the moved index. If it exceeds the budget, its task cannot write to
        // the catalog after ordinary close starts.
        let (segments, coverage_intervals) = index.size();
        let encode_started = std::time::Instant::now();
        let bytes = match tokio::time::timeout_at(
            deadline,
            tokio::task::spawn_blocking(move || index.restart_projection_bytes()),
        )
        .await
        {
            Ok(Ok(Ok(bytes))) => bytes,
            Ok(Ok(Err(error))) => {
                warn!(%error, segments, coverage_intervals, "couldn't encode the restart projection; the next start rebuilds");
                return;
            }
            Ok(Err(error)) => {
                warn!(%error, "encoding the restart projection stopped; the next start rebuilds");
                return;
            }
            Err(_) => {
                warn!(
                    segments,
                    coverage_intervals, "restart projection encoding deadline expired; the next start rebuilds"
                );
                return;
            }
        };
        let encode = encode_started.elapsed();
        if tokio::time::Instant::now() >= deadline {
            warn!("encoding the restart projection used up its deadline; the next start rebuilds");
            return;
        }
        let catalog = Arc::clone(&self.catalog.0);
        // A redb commit cannot be cancelled safely. Once admitted, join publication before close;
        // never detach a possible catalog writer on an async timeout. The budget controls admission,
        // not a hard wall-clock limit on synchronous durable I/O.
        let publish_started = std::time::Instant::now();
        match tokio::task::spawn_blocking(move || {
            let size = bytes.len();
            catalog.save_restart_projection(stamp, &bytes).map(|saved| (saved, size))
        })
        .await
        {
            Ok(Ok((true, bytes))) => info!(
                bytes,
                segments,
                coverage_intervals,
                ?encode,
                publish = ?publish_started.elapsed(),
                "saved the clean-stop availability projection"
            ),
            Ok(Ok((false, _))) => info!("restart projection was not current; the next start rebuilds"),
            Ok(Err(error)) => warn!(%error, "couldn't save the restart projection; the next start rebuilds"),
            Err(error) => warn!(%error, "publishing the restart projection stopped; the next start rebuilds"),
        }
    }

    /// `started` is when the stop's catalog work began, for the log; the close itself always gets `CLOSE_WAIT`, so a
    /// long projection save can't leave it no time and the file needing repair at the next start.
    async fn finish_catalog_close(&self, started: std::time::Instant) {
        let catalog = Arc::clone(&self.catalog.0);
        let wait = CLOSE_WAIT;
        let closed = tokio::task::spawn_blocking(move || catalog.close(wait)).await;
        match closed {
            Ok(Ok(())) => info!(took = ?started.elapsed(), "closed the catalog"),
            Ok(Err(e)) => warn!(error = %e, "couldn't close the catalog cleanly; the next start repairs it"),
            Err(e) => warn!(error = %e, "closing the catalog stopped; the next start repairs it"),
        }
    }
}

/// The server's hold on the catalog, for closing it at a clean stop.
struct Closing(Arc<dyn Catalog>);

impl std::fmt::Debug for Closing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Closing(..)")
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.tier0.unlink();
        if let Err(e) = std::fs::remove_file(&self.config.socket)
            && e.kind() != io::ErrorKind::NotFound
        {
            warn!(error = %e, "couldn't remove the socket file");
        }
    }
}

/// Tells a connection past the session cap why it's being closed, then closes it.
async fn refuse(mut stream: UnixStream, timeout: Duration) {
    let mut b = FlatBufferBuilder::new();
    let bytes = protocol::error(&mut b, 0, ErrorCode::Unavailable, "too many sessions; try again later");
    let _ = tokio::time::timeout(timeout, write_frame(&mut stream, &bytes)).await;
}

/// Nanoseconds since the Unix epoch, UTC.
fn now_ns() -> u64 {
    let d = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)
}

/// What the session does after handling one message.
enum Next {
    Continue,
    Close,
}

/// How many replies and pushes can wait for a session's writer. A session that falls this far behind isn't reading;
/// the core drops it (see `core::Core::push`).
const OUTBOUND: usize = 1024;

/// A connection's reading side: it reads frames, answers what it can itself, and passes the rest to the core. Every
/// reply goes through `tx` to the session's writer ([`write_loop`]), which is the only place that writes the socket.
struct Session {
    rd: OwnedReadHalf,
    tx: mpsc::Sender<Vec<u8>>,
    timeout: Duration,
    core: CoreHandle,
    builder: FlatBufferBuilder<'static>,
    /// The session, as the core knows it (the handshake opened it).
    id: uuid::Uuid,
    /// Set once the core has been told the session ended, so it's told only once.
    ended: bool,
    client: String,
    /// Signalled by the core when it drops this session, so the connection closes.
    kicked: Arc<tokio::sync::Notify>,
    /// The arena that Tier 0 locators are checked against.
    tier0: Arc<Tier0>,
}

/// Runs one connection to its end: the reader and the writer together, under the session cap's `permit`.
async fn run_session(
    mut stream: UnixStream,
    timeout: Duration,
    core: CoreHandle,
    tier0: Arc<Tier0>,
    permit: OwnedSemaphorePermit,
) {
    let _permit = permit;
    // The handshake runs on the whole stream: Welcome's length prefix carries the arena fds (SCM_RIGHTS), which
    // needs the socket itself. Only after it does the stream split, so no later frame can ever carry fds.
    let Some(opened) = handshake(&mut stream, timeout, &core, &tier0).await else { return };
    let (rd, wr) = stream.into_split();
    let mut session = Session {
        rd,
        tx: opened.tx,
        timeout,
        core,
        builder: FlatBufferBuilder::new(),
        id: opened.id,
        ended: false,
        client: opened.client,
        kicked: opened.kicked,
        tier0,
    };
    let writer = write_loop(wr, opened.rx, timeout);
    tokio::pin!(writer);
    let kicked = Arc::clone(&session.kicked);
    let writer_done = {
        let reader = session.read_loop();
        tokio::pin!(reader);
        tokio::select! {
            () = &mut reader => false,
            // The writer stopped first, because a write failed or timed out: stop reading. A partial frame is lost,
            // which is fine, since the session is ending.
            () = &mut writer => true,
            // The core dropped the session because its queue filled: close now, without flushing the backlog.
            () = kicked.notified() => true,
        }
    };
    // Either way the session is over: tell the core (so it drops its sender and the session's streams stop being
    // live), then drop the reader's own sender. The writer then ends once it has sent what was queued.
    session.end().await;
    drop(session);
    if !writer_done {
        writer.await;
    }
}

/// A session that finished its handshake: the core knows it, and these are its queue's two ends.
struct Opened {
    id: uuid::Uuid,
    client: String,
    tx: mpsc::Sender<Vec<u8>>,
    rx: mpsc::Receiver<Vec<u8>>,
    kicked: Arc<tokio::sync::Notify>,
}

/// Writes one frame straight to the socket during the handshake, within the timeout. `false` if it failed.
async fn write_direct(stream: &mut UnixStream, timeout: Duration, bytes: &[u8]) -> bool {
    matches!(tokio::time::timeout(timeout, write_frame(stream, bytes)).await, Ok(Ok(())))
}

/// Reads until a good `Hello`, refusing anything else with `NotAllowed` (and `Goodbye` closes without a reply), then
/// opens the session in the core and sends `Welcome` with the arena fds on its length prefix. Every read and write
/// is bounded by the heartbeat timeout. `None` when the connection ends instead.
async fn handshake(stream: &mut UnixStream, timeout: Duration, core: &CoreHandle, tier0: &Tier0) -> Option<Opened> {
    let mut b = FlatBufferBuilder::new();
    let mut buf = Vec::new();
    loop {
        match tokio::time::timeout(timeout, read_frame(stream, &mut buf)).await {
            Err(_) => {
                info!("closing a connection that never said Hello");
                return None;
            }
            Ok(Err(FrameError::TooLarge(n))) => {
                let bytes =
                    protocol::error(&mut b, 0, ErrorCode::BadRequest, &format!("frame of {n} bytes is over the limit"));
                write_direct(stream, timeout, &bytes).await;
                return None;
            }
            Ok(Err(FrameError::Closed | FrameError::FdsTruncated(_) | FrameError::Io(_))) => return None,
            Ok(Ok(())) => {}
        }
        let env = match protocol::read_envelope(&buf) {
            Ok(env) => env,
            Err(e) => {
                let bytes = protocol::error(&mut b, 0, ErrorCode::BadRequest, &format!("not a ControlEnvelope: {e}"));
                write_direct(stream, timeout, &bytes).await;
                return None;
            }
        };
        let request_id = env.request_id();
        match env.payload_type() {
            ControlPayload::Goodbye => return None,
            ControlPayload::Hello => {
                let Some(hello) = env.payload_as_hello() else {
                    let bytes = protocol::error(&mut b, request_id, ErrorCode::BadRequest, "Hello without a body");
                    write_direct(stream, timeout, &bytes).await;
                    return None;
                };
                let major = hello.schema().map(|s| s.major());
                if major != Some(SCHEMA_MAJOR) {
                    let msg = format!("schema major {major:?}; this broker speaks {SCHEMA_MAJOR}");
                    let bytes = protocol::error(&mut b, request_id, ErrorCode::IncompatibleSchema, &msg);
                    write_direct(stream, timeout, &bytes).await;
                    return None;
                }
                let id = uuid::Uuid::now_v7();
                let client = hello.client_name().unwrap_or("").to_owned();
                info!(%client, pid = hello.pid(), session = %id, "session started");
                let (tx, rx) = mpsc::channel(OUTBOUND);
                let kicked = Arc::new(tokio::sync::Notify::new());
                // The core learns of the session before the client can send anything that needs it (Open shares the
                // voice turn's channel, so a voice turn sent right after Welcome comes after it).
                let (voice, snapshot) = tokio::sync::oneshot::channel();
                let open = Command::Open {
                    session: id,
                    outbound: tx.clone(),
                    kicked: Arc::clone(&kicked),
                    pid: hello.pid(),
                    voice,
                };
                core.send(open).await.ok()?;
                let voice = match tokio::time::timeout(timeout, snapshot).await {
                    Ok(Ok(voice)) => voice,
                    _ => {
                        let _ = core.send(Command::Close { session: id }).await;
                        return None;
                    }
                };
                // One arena, so one fd, in `Welcome.arenas` order (far under `MAX_FDS`).
                let welcome = protocol::welcome(&mut b, request_id, *id.as_bytes(), &[tier0.summary()], voice);
                let fds = [tier0.fd()];
                let sent = tokio::time::timeout(timeout, write_frame_with_fds(stream, &welcome, &fds)).await;
                if !matches!(sent, Ok(Ok(()))) {
                    let _ = core.send(Command::Close { session: id }).await;
                    return None;
                }
                return Some(Opened { id, client, tx, rx, kicked });
            }
            other => {
                let msg = format!("send Hello first, not {}", protocol::payload_name(other));
                let bytes = protocol::error(&mut b, request_id, ErrorCode::NotAllowed, &msg);
                if !write_direct(stream, timeout, &bytes).await {
                    return None;
                }
            }
        }
    }
}

/// The `Error` frame that replaces a message the broker built over [`continuum_wire::MAX_FRAME`]: the client hears
/// which request failed and why, and the session stays open. Replies that page (query results, purge and
/// stream lists) stay under the limit themselves; this is the backstop for any other builder that outgrows it.
fn oversized_reply(bytes: &[u8]) -> Vec<u8> {
    // A buffer that fails verification still gets an answer, just without a request to name.
    let (request_id, kind) = protocol::read_envelope(bytes)
        .map_or((0, "unreadable"), |env| (env.request_id(), protocol::payload_name(env.payload_type())));
    let message = format!(
        "the broker's {kind} reply is {} bytes, over the {} byte frame limit; ask for less (a smaller page or range)",
        bytes.len(),
        continuum_wire::MAX_FRAME
    );
    warn!(request_id, kind, bytes = bytes.len(), "a reply was too large for the wire; sent an error instead");
    protocol::error(&mut FlatBufferBuilder::new(), request_id, ErrorCode::Internal, &message)
}

/// Writes every queued frame, each within the heartbeat timeout. A client that sends but never reads fills the
/// socket's buffers; without the limit this write would wait forever. A frame over the wire limit is answered with
/// an `Error` for its request instead of closing the session.
async fn write_loop(mut wr: OwnedWriteHalf, mut rx: mpsc::Receiver<Vec<u8>>, timeout: Duration) {
    while let Some(mut bytes) = rx.recv().await {
        if bytes.len() > continuum_wire::MAX_FRAME {
            bytes = oversized_reply(&bytes);
        }
        match tokio::time::timeout(timeout, write_frame(&mut wr, &bytes)).await {
            Ok(Ok(())) => {}
            Ok(Err(e)) if e.kind() == io::ErrorKind::InvalidInput => {
                // Unreachable after the size check above (the replacement error is a few hundred bytes); kept so a
                // future wire refusal still ends the session loudly rather than looping.
                warn!(error = %e, bytes = bytes.len(), "a message was too large for the wire; closing the session");
                return;
            }
            Ok(Err(e)) => {
                debug!(error = %e, "session write failed");
                return;
            }
            Err(_) => {
                info!("closing a session that stopped reading its replies");
                return;
            }
        }
    }
}

impl Session {
    async fn read_loop(&mut self) {
        let mut buf = Vec::new();
        loop {
            match tokio::time::timeout(self.timeout, read_frame(&mut self.rd, &mut buf)).await {
                Err(_) => {
                    info!(client = %self.client, "closing a session that went quiet");
                    return;
                }
                Ok(Err(FrameError::Closed)) => return,
                Ok(Err(FrameError::TooLarge(n))) => {
                    let msg = format!("frame of {n} bytes is over the limit");
                    self.send_error(0, ErrorCode::BadRequest, &msg).await;
                    return;
                }
                Ok(Err(FrameError::Io(e))) => {
                    debug!(error = %e, "session read failed");
                    return;
                }
                // Plain reads never take fds, so this can't happen here; treat it as a broken stream.
                Ok(Err(FrameError::FdsTruncated(_))) => return,
                Ok(Ok(())) => {}
            }
            match self.handle(&buf).await {
                Ok(Next::Continue) => {}
                Ok(Next::Close) => return,
                Err(e) => {
                    debug!(error = %e, "session ended: {e}");
                    return;
                }
            }
        }
    }

    /// Tells the core the session is over, so its streams stop being live and it drops its sender.
    async fn end(&mut self) {
        if !std::mem::replace(&mut self.ended, true) {
            let id = self.id;
            let _ = self.core.send(Command::Close { session: id }).await;
            info!(client = %self.client, session = %id, "session ended");
        }
    }

    /// Queues a reply for the writer. Fails once the writer has stopped.
    async fn send(&mut self, bytes: Vec<u8>) -> io::Result<()> {
        self.tx.send(bytes).await.map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "the session's writer stopped"))
    }

    async fn send_error(&mut self, request_id: u64, code: ErrorCode, message: &str) {
        let bytes = protocol::error(&mut self.builder, request_id, code, message);
        if let Err(e) = self.send(bytes).await {
            debug!(error = %e, "couldn't send an error");
        }
    }

    /// Hands a command to the core. Fails (ending the session) once the core has stopped.
    async fn to_core(&self, command: Command) -> io::Result<()> {
        self.core.send(command).await.map_err(|e| io::Error::other(e.to_string()))
    }

    async fn handle(&mut self, bytes: &[u8]) -> io::Result<Next> {
        let env = match protocol::read_envelope(bytes) {
            Ok(env) => env,
            Err(e) => {
                // The stream can't be trusted after a malformed frame, so say why and close.
                self.send_error(0, ErrorCode::BadRequest, &format!("not a ControlEnvelope: {e}")).await;
                return Ok(Next::Close);
            }
        };
        let request_id = env.request_id();
        let kind = env.payload_type();

        // Goodbye is the clean disconnect: close without a reply.
        if kind == ControlPayload::Goodbye {
            return Ok(Next::Close);
        }

        match kind {
            // The handshake took the first Hello; another one is a mistake.
            ControlPayload::Hello => {
                self.send_error(request_id, ErrorCode::BadRequest, "Hello sent twice").await;
                Ok(Next::Continue)
            }
            ControlPayload::Heartbeat => {
                let bytes = protocol::heartbeat(&mut self.builder, request_id, now_ns());
                self.send(bytes).await?;
                // Heartbeat telemetry doesn't reach the scheduler yet: it names loaded models, the scheduler wants capability ids,
                // and passing an empty list would clear what the scheduler knows is loaded and charge a cold start on every quote.
                // It joins once the telemetry names capabilities.
                Ok(Next::Continue)
            }
            ControlPayload::RegisterWorker => {
                let session = self.id;
                let ask = env.payload_as_register_worker().ok_or("RegisterWorker without a body".to_owned());
                match ask.and_then(|w| protocol::worker_ask(&w)) {
                    Ok(ask) => self.to_core(Command::RegisterWorker { session, request_id, ask }).await?,
                    Err(msg) => self.send_error(request_id, ErrorCode::BadRequest, &msg).await,
                }
                Ok(Next::Continue)
            }
            ControlPayload::SetSystemState => {
                let Some(s) = env.payload_as_set_system_state() else {
                    self.send_error(request_id, ErrorCode::BadRequest, "SetSystemState without a body").await;
                    return Ok(Next::Continue);
                };
                self.to_core(Command::SetSystemState { voice: s.voice_turn() }).await?;
                Ok(Next::Continue)
            }
            ControlPayload::QuoteRequest => {
                let session = self.id;
                let req = env.payload_as_quote_request().ok_or("QuoteRequest without a body".to_owned());
                match req.and_then(|q| protocol::lease_req(&q)) {
                    Ok(req) => self.to_core(Command::QuoteRequest { session, request_id, req }).await?,
                    Err(msg) => self.send_error(request_id, ErrorCode::BadRequest, &msg).await,
                }
                Ok(Next::Continue)
            }
            ControlPayload::Query => {
                let session = self.id;
                let ask = env.payload_as_query().ok_or("Query without a body".to_owned());
                match ask.and_then(|q| protocol::query_ask(&q)) {
                    Ok(ask) => self.to_core(Command::Query { session, request_id, ask }).await?,
                    Err(msg) => self.send_error(request_id, ErrorCode::BadRequest, &msg).await,
                }
                Ok(Next::Continue)
            }
            ControlPayload::Subscribe => {
                let session = self.id;
                let ask = env.payload_as_subscribe().ok_or("Subscribe without a body".to_owned());
                match ask.and_then(|s| protocol::subscribe_ask(&s)) {
                    Ok(ask) => self.to_core(Command::Subscribe { session, request_id, ask }).await?,
                    Err(msg) => self.send_error(request_id, ErrorCode::BadRequest, &msg).await,
                }
                Ok(Next::Continue)
            }
            ControlPayload::Unsubscribe => {
                let session = self.id;
                match env.payload_as_unsubscribe() {
                    Some(u) => {
                        let id = u.subscription_id();
                        self.to_core(Command::Unsubscribe { session, request_id, id }).await?;
                    }
                    None => self.send_error(request_id, ErrorCode::BadRequest, "Unsubscribe without a body").await,
                }
                Ok(Next::Continue)
            }
            ControlPayload::AckEvents => {
                let session = self.id;
                match env.payload_as_ack_events() {
                    Some(a) => {
                        let (id, position) = (a.subscription_id(), a.position());
                        self.to_core(Command::AckEvents { session, id, position }).await?;
                    }
                    None => self.send_error(request_id, ErrorCode::BadRequest, "AckEvents without a body").await,
                }
                Ok(Next::Continue)
            }
            ControlPayload::CommitPromise => {
                let session = self.id;
                let ask = env.payload_as_commit_promise().ok_or("CommitPromise without a body".to_owned());
                match ask.and_then(|c| protocol::commit_promise_ask(&c)) {
                    Ok((promise, option, band)) => {
                        self.to_core(Command::CommitPromise { session, request_id, promise, option, band }).await?;
                    }
                    Err(msg) => self.send_error(request_id, ErrorCode::BadRequest, &msg).await,
                }
                Ok(Next::Continue)
            }
            ControlPayload::CancelPromise => {
                let session = self.id;
                let ask = env.payload_as_cancel_promise().ok_or("CancelPromise without a body".to_owned());
                match ask.and_then(|c| protocol::cancel_promise_ask(&c)) {
                    Ok(promise) => self.to_core(Command::CancelPromise { session, request_id, promise }).await?,
                    Err(msg) => self.send_error(request_id, ErrorCode::BadRequest, &msg).await,
                }
                Ok(Next::Continue)
            }
            ControlPayload::JobRejected => {
                let session = self.id;
                let ask = env.payload_as_job_rejected().ok_or("JobRejected without a body".to_owned());
                match ask.and_then(|j| protocol::job_rejected_ask(&j)) {
                    Ok((job, reason)) => self.to_core(Command::JobRejected { session, job, reason }).await?,
                    Err(msg) => self.send_error(request_id, ErrorCode::BadRequest, &msg).await,
                }
                Ok(Next::Continue)
            }
            ControlPayload::ListPurges => {
                let session = self.id;
                match env.payload_as_list_purges() {
                    Some(l) => {
                        let (after, limit) = protocol::list_purges_ask(&l);
                        self.to_core(Command::ListPurges { session, request_id, after, limit }).await?;
                    }
                    None => self.send_error(request_id, ErrorCode::BadRequest, "ListPurges without a body").await,
                }
                Ok(Next::Continue)
            }
            ControlPayload::ListStreams => {
                let session = self.id;
                self.to_core(Command::ListStreams { session, request_id }).await?;
                Ok(Next::Continue)
            }
            ControlPayload::GetPurge => {
                let session = self.id;
                match env.payload_as_get_purge().and_then(|g| protocol::get_purge_ask(&g)) {
                    Some(purge) => self.to_core(Command::GetPurge { session, request_id, purge }).await?,
                    None => self.send_error(request_id, ErrorCode::BadRequest, "GetPurge without a purge_id").await,
                }
                Ok(Next::Continue)
            }
            ControlPayload::PurgeAck => {
                let session = self.id;
                match env.payload_as_purge_ack().and_then(|a| a.purge_id().map(|u| continuum_core::Uuid(u.0))) {
                    Some(purge) => self.to_core(Command::PurgeAck { session, purge }).await?,
                    None => self.send_error(request_id, ErrorCode::BadRequest, "PurgeAck without a purge_id").await,
                }
                Ok(Next::Continue)
            }
            ControlPayload::SaveWorkerState | ControlPayload::LoadWorkerState | ControlPayload::ForgetWorkerState => {
                let session = self.id;
                let ask = match env.payload_type() {
                    ControlPayload::SaveWorkerState => {
                        env.payload_as_save_worker_state().and_then(|m| protocol::worker_state::save(&m))
                    }
                    ControlPayload::LoadWorkerState => {
                        env.payload_as_load_worker_state().and_then(|m| protocol::worker_state::load(&m))
                    }
                    _ => env.payload_as_forget_worker_state().and_then(|m| protocol::worker_state::forget(&m)),
                };
                match ask {
                    Some(ask) => self.to_core(Command::WorkerState { session, request_id, ask }).await?,
                    None => {
                        self.send_error(request_id, ErrorCode::BadRequest, "a worker state request without its owner")
                            .await
                    }
                }
                Ok(Next::Continue)
            }
            ControlPayload::RememberForgotten => {
                let session = self.id;
                match env.payload_as_remember_forgotten().and_then(|r| r.hash().map(protocol::forgotten_hash)) {
                    Some(hash) => self.to_core(Command::RememberForgotten { session, request_id, hash }).await?,
                    None => {
                        self.send_error(request_id, ErrorCode::BadRequest, "RememberForgotten without a hash").await
                    }
                }
                Ok(Next::Continue)
            }
            ControlPayload::ListForgotten => {
                match env.payload_as_list_forgotten() {
                    Some(page) => {
                        self.to_core(Command::ListForgotten {
                            session: self.id,
                            request_id,
                            after: page.after().map(protocol::forgotten_hash),
                            limit: page.limit(),
                        })
                        .await?
                    }
                    None => self.send_error(request_id, ErrorCode::BadRequest, "ListForgotten without a request").await,
                }
                Ok(Next::Continue)
            }
            ControlPayload::IsForgotten => {
                let session = self.id;
                match env.payload_as_is_forgotten().and_then(|r| r.hash().map(protocol::forgotten_hash)) {
                    Some(hash) => self.to_core(Command::IsForgotten { session, request_id, hash }).await?,
                    None => self.send_error(request_id, ErrorCode::BadRequest, "IsForgotten without a hash").await,
                }
                Ok(Next::Continue)
            }
            ControlPayload::ForgetKeeper => {
                let session = self.id;
                match env.payload_as_forget_keeper().and_then(|f| protocol::forget_keeper_ask(&f)) {
                    Some(name) => self.to_core(Command::ForgetKeeper { session, request_id, name }).await?,
                    None => self.send_error(request_id, ErrorCode::BadRequest, "ForgetKeeper without a name").await,
                }
                Ok(Next::Continue)
            }
            ControlPayload::RenewLease => {
                let session = self.id;
                let Some(r) = env.payload_as_renew_lease() else {
                    self.send_error(request_id, ErrorCode::BadRequest, "RenewLease without a body").await;
                    return Ok(Next::Continue);
                };
                let (lease, extend_ms) = (r.lease_id(), r.extend_ms());
                self.to_core(Command::RenewLease { session, request_id, lease, extend_ms }).await?;
                Ok(Next::Continue)
            }
            ControlPayload::CommitLease => {
                let session = self.id;
                let commit = env.payload_as_commit_lease().ok_or("CommitLease without a body".to_owned());
                let checked = commit.and_then(|c| protocol::commit(&c)).and_then(|c| {
                    for (i, (h, l)) in c.outputs.iter().enumerate() {
                        let checked = l.as_ref().map_or(Ok(()), |l| self.tier0.check_for(h.stream(), l));
                        checked.map_err(|e| format!("output {i}: {e}"))?;
                    }
                    Ok(c)
                });
                match checked {
                    Ok(commit) => self.to_core(Command::CommitLease { session, request_id, commit }).await?,
                    Err(msg) => self.send_error(request_id, ErrorCode::BadRequest, &msg).await,
                }
                Ok(Next::Continue)
            }
            ControlPayload::AbortLease => {
                let session = self.id;
                let (lease, reason) = env
                    .payload_as_abort_lease()
                    .map_or((0, String::new()), |a| (a.lease_id(), a.reason().unwrap_or("").to_owned()));
                self.to_core(Command::AbortLease { session, request_id, lease, reason }).await?;
                Ok(Next::Continue)
            }
            ControlPayload::LeaseReleased => {
                let session = self.id;
                let lease = env.payload_as_lease_released().map_or(0, |l| l.lease_id());
                self.to_core(Command::LeaseReleased { session, lease }).await?;
                Ok(Next::Continue)
            }
            ControlPayload::RegisterStream => {
                let session = self.id;
                let info = env.payload_as_register_stream().ok_or("RegisterStream without a body".to_owned());
                let parsed = info.and_then(|r| Ok((protocol::stream_info(&r)?, protocol::ring_ask(&r)?)));
                match parsed {
                    Ok((info, ring)) => {
                        self.to_core(Command::RegisterStream { session, request_id, info, ring }).await?;
                    }
                    Err(msg) => self.send_error(request_id, ErrorCode::BadRequest, &msg).await,
                }
                Ok(Next::Continue)
            }
            ControlPayload::IngestNotice => {
                let session = self.id;
                let Some(notice) = env.payload_as_ingest_notice() else {
                    self.send_error(request_id, ErrorCode::BadRequest, "IngestNotice without a body").await;
                    return Ok(Next::Continue);
                };
                let checked = protocol::ingest(&notice).and_then(|(header, locator)| {
                    locator.as_ref().map_or(Ok(()), |l| self.tier0.check_for(header.stream(), l))?;
                    let dependencies = protocol::dependencies(&notice, request_id)?;
                    Ok((header, locator, dependencies))
                });
                match checked {
                    Ok((header, locator, dependencies)) => {
                        self.to_core(Command::Ingest { session, request_id, header, locator, dependencies }).await?;
                    }
                    Err(reason) => {
                        // The id, if the bytes are a valid header at all; the unchecked nested accessor would panic on garbage.
                        let payload = protocol::header_id(&notice);
                        let bytes = protocol::ingest_rejected(
                            &mut self.builder,
                            request_id,
                            notice.stream_id(),
                            payload,
                            &reason,
                            ErrorCode::BadRequest,
                        );
                        self.send(bytes).await?;
                    }
                }
                Ok(Next::Continue)
            }
            ControlPayload::CommitPayloadCopy => {
                let ask = env.payload_as_commit_payload_copy().ok_or("CommitPayloadCopy without a body".to_owned());
                match ask.and_then(|c| protocol::payload_copy_ask(&c)) {
                    Ok((payload, location)) => {
                        self.to_core(Command::CommitPayloadCopy { session: self.id, request_id, payload, location })
                            .await?
                    }
                    Err(msg) => self.send_error(request_id, ErrorCode::BadRequest, &msg).await,
                }
                Ok(Next::Continue)
            }
            ControlPayload::CommitChunk => {
                let session = self.id;
                let ask = env.payload_as_commit_chunk().ok_or("CommitChunk without a body".to_owned());
                match ask.and_then(|c| protocol::commit_chunk_ask(&c)) {
                    Ok(ask) => self.to_core(Command::CommitChunk { session, request_id, ask }).await?,
                    Err(msg) => self.send_error(request_id, ErrorCode::BadRequest, &msg).await,
                }
                Ok(Next::Continue)
            }
            ControlPayload::RewriteChunk => {
                let session = self.id;
                let ask = env.payload_as_rewrite_chunk().ok_or("RewriteChunk without a body".to_owned());
                match ask.and_then(|r| protocol::rewrite_chunk_ask(&r)) {
                    Ok((chunk, frames, location)) => {
                        let command = Command::RewriteChunk { session, request_id, chunk, frames, location };
                        self.to_core(command).await?;
                    }
                    Err(msg) => self.send_error(request_id, ErrorCode::BadRequest, &msg).await,
                }
                Ok(Next::Continue)
            }
            ControlPayload::ListRecordingInputs => {
                let ask = env.payload_as_list_recording_inputs().ok_or("ListRecordingInputs without body".to_owned());
                match ask.and_then(|r| protocol::applicability::list_ask(&r)) {
                    Ok(ask) => {
                        self.to_core(Command::ListRecordingInputs { session: self.id, request_id, ask: Box::new(ask) })
                            .await?
                    }
                    Err(msg) => self.send_error(request_id, ErrorCode::BadRequest, &msg).await,
                }
                Ok(Next::Continue)
            }
            ControlPayload::ListAgedOut => {
                let session = self.id;
                let ask = env.payload_as_list_aged_out().ok_or("ListAgedOut without a body".to_owned());
                match ask.and_then(|r| protocol::aging::list_aged_out_ask(&r)) {
                    Ok((stream, window, limit)) => {
                        self.to_core(Command::ListAgedOut { session, request_id, stream, window, limit }).await?;
                    }
                    Err(msg) => self.send_error(request_id, ErrorCode::BadRequest, &msg).await,
                }
                Ok(Next::Continue)
            }
            ControlPayload::GetCatalogStats => {
                let session = self.id;
                self.to_core(Command::CatalogStats { session, request_id }).await?;
                Ok(Next::Continue)
            }
            ControlPayload::CommitSeqAt => {
                let session = self.id;
                match env.payload_as_commit_seq_at() {
                    Some(r) => self.to_core(Command::CommitSeqAt { session, request_id, at_ns: r.at_ns() }).await?,
                    None => self.send_error(request_id, ErrorCode::BadRequest, "CommitSeqAt without a body").await,
                }
                Ok(Next::Continue)
            }
            ControlPayload::GetChunk => {
                let session = self.id;
                match env.payload_as_get_chunk().and_then(|r| r.chunk_id().map(|u| continuum_core::Uuid(u.0))) {
                    Some(chunk) => self.to_core(Command::GetChunk { session, request_id, chunk }).await?,
                    None => self.send_error(request_id, ErrorCode::BadRequest, "GetChunk without a chunk_id").await,
                }
                Ok(Next::Continue)
            }
            ControlPayload::ResolveFrames => {
                let session = self.id;
                let ids: Option<Vec<continuum_core::Uuid>> = env.payload_as_resolve_frames().map(|r| {
                    r.ids().map(|v| v.iter().map(|u| continuum_core::Uuid(u.0)).collect()).unwrap_or_default()
                });
                match ids {
                    Some(ids) if ids.len() <= crate::protocol::aging::MAX_RESOLVE_IDS => {
                        self.to_core(Command::ResolveFrames { session, request_id, ids }).await?;
                    }
                    Some(ids) => {
                        let why = format!(
                            "ResolveFrames asks about {} ids; at most {}",
                            ids.len(),
                            crate::protocol::aging::MAX_RESOLVE_IDS
                        );
                        self.send_error(request_id, ErrorCode::BadRequest, &why).await;
                    }
                    None => self.send_error(request_id, ErrorCode::BadRequest, "ResolveFrames without a body").await,
                }
                Ok(Next::Continue)
            }
            ControlPayload::Purge => {
                let session = self.id;
                let ask = env.payload_as_purge().ok_or("Purge without a body".to_owned());
                match ask.and_then(|p| protocol::purge_ask(&p)) {
                    Ok((ask, reason, detail)) => {
                        self.to_core(Command::Purge { session, request_id, ask, reason, detail }).await?;
                    }
                    Err(msg) => self.send_error(request_id, ErrorCode::BadRequest, &msg).await,
                }
                Ok(Next::Continue)
            }
            ControlPayload::Error => {
                let e = env.payload_as_error();
                warn!(
                    client = %self.client,
                    code = ?e.map(|e| e.code()),
                    message = e.and_then(|e| e.message()).unwrap_or(""),
                    "client reported an error"
                );
                Ok(Next::Continue)
            }
            ControlPayload::NONE => {
                self.send_error(request_id, ErrorCode::BadRequest, "envelope has no payload").await;
                Ok(Next::Continue)
            }
            other => {
                let msg = format!("{} is not implemented yet", protocol::payload_name(other));
                self.send_error(request_id, ErrorCode::Unavailable, &msg).await;
                Ok(Next::Continue)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn restart_checkpoint_retries_busy_until_a_quiescent_fence_succeeds() {
        let mut attempts = 0;
        let result = wait_for_restart_checkpoint(tokio::time::Instant::now() + Duration::from_secs(1), || {
            attempts += 1;
            std::future::ready(if attempts < 3 { Err(core::CoreError::Busy) } else { Ok(Ok::<_, ()>(7)) })
        })
        .await
        .unwrap()
        .unwrap();
        assert_eq!(attempts, 3);
        assert_eq!(result, Ok(7));
    }

    #[tokio::test]
    async fn restart_checkpoint_always_busy_expires_without_spinning() {
        let mut attempts = 0;
        let result = wait_for_restart_checkpoint(tokio::time::Instant::now() + Duration::from_millis(45), || {
            attempts += 1;
            std::future::ready(Err::<(), _>(core::CoreError::Busy))
        })
        .await;
        assert!(result.is_err(), "a permanently busy core must respect the original deadline");
        assert!((1..=3).contains(&attempts), "retries must yield rather than spin: {attempts}");
    }

    #[tokio::test]
    async fn restart_checkpoint_permanent_failure_and_ineligible_success_are_not_retried() {
        let mut attempts = 0;
        let result = wait_for_restart_checkpoint(tokio::time::Instant::now() + Duration::from_secs(1), || {
            attempts += 1;
            std::future::ready(Err::<(), _>(core::CoreError::Gone))
        })
        .await
        .unwrap();
        assert!(matches!(result, Err(core::CoreError::Gone)));
        assert_eq!(attempts, 1);
        let result = wait_for_restart_checkpoint(tokio::time::Instant::now() + Duration::from_secs(1), || {
            attempts += 1;
            std::future::ready(Ok::<_, core::CoreError>(Err::<(), _>(core::CheckpointRefusal::FrameSweep)))
        })
        .await
        .unwrap()
        .unwrap();
        assert_eq!(result, Err(core::CheckpointRefusal::FrameSweep), "a completed but ineligible fence is final");
        assert_eq!(attempts, 2);
    }

    /// A reply the broker built over the frame limit used to close the session with nothing said. The client
    /// gets an `Error` naming its request instead, and the session keeps delivering what comes after.
    #[tokio::test]
    async fn an_oversized_reply_becomes_an_error_for_its_request_and_the_session_stays_open() {
        let (broker_side, mut client) = UnixStream::pair().unwrap();
        let (_rd, wr) = broker_side.into_split();
        let (tx, rx) = mpsc::channel(4);
        let writer = tokio::spawn(write_loop(wr, rx, Duration::from_secs(30)));
        let mut b = FlatBufferBuilder::new();
        let huge = protocol::error(&mut b, 7, ErrorCode::NotFound, &"x".repeat(continuum_wire::MAX_FRAME));
        assert!(huge.len() > continuum_wire::MAX_FRAME);
        tx.send(huge.clone()).await.unwrap();
        tx.send(protocol::heartbeat(&mut b, 8, 42)).await.unwrap();

        let mut buf = Vec::new();
        read_frame(&mut client, &mut buf).await.unwrap();
        let env = protocol::read_envelope(&buf).unwrap();
        assert_eq!((env.request_id(), env.payload_type()), (7, ControlPayload::Error));
        let error = env.payload_as_error().unwrap();
        assert_eq!(error.code(), ErrorCode::Internal);
        let message = error.message().unwrap();
        assert!(message.contains(&huge.len().to_string()), "{message}");
        assert!(message.contains("frame limit"), "{message}");

        read_frame(&mut client, &mut buf).await.unwrap();
        let env = protocol::read_envelope(&buf).unwrap();
        assert_eq!((env.request_id(), env.payload_type()), (8, ControlPayload::Heartbeat));
        drop(tx);
        writer.await.unwrap();
    }

    /// Bytes that aren't an envelope still get an answer, addressed to no request.
    #[test]
    fn an_oversized_unreadable_message_is_answered_without_a_request_id() {
        let bytes = oversized_reply(&vec![0xff; continuum_wire::MAX_FRAME + 1]);
        assert!(bytes.len() < 1024);
        let env = protocol::read_envelope(&bytes).unwrap();
        assert_eq!((env.request_id(), env.payload_type()), (0, ControlPayload::Error));
    }
}
