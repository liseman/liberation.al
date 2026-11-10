//! The catalog scrubber: decides when the catalog file is copied into a fresh one (which gives redb's freed pages
//! back to the disk and leaves no purged bytes in them), and runs that copy as small low-priority steps on a thread
//! of its own (docs/CONTINUUM.md, "Catalog" and "Forgetting").
//!
//! The copy and catch-up steps never take the catalog's writer lock: they read a snapshot of the live file and write
//! the fresh one, so ingest commits and reads go on beside them. Only the swap at the end excludes every other call,
//! for a few carried keys, one small commit, a marker file, and a rename. The thread spends at most
//! [`ScrubConfig::duty_percent`] of its time stepping and sleeps the rest, so a scrub of a large catalog is spread
//! out rather than loading the disk. A crash or stop at any point is safe: opening the catalog throws away an
//! unfinished fresh file and finishes a swap whose rename was lost (`RedbCatalog::open`).
//!
//! When a scrub starts, checked every [`ScrubConfig::check_every`]:
//! - the file's live bytes are not counted yet (a file no scrub has copied), so the first scrub counts them;
//! - a purge finished since the last scrub (`scrub_due`), at most once every [`ScrubConfig::due_gap`] (purges run
//!   every few minutes, so acting on each would mean copying the catalog without end);
//! - or the file holds at least [`ScrubConfig::min_waste_bytes`] and [`ScrubConfig::min_waste_percent`] more than
//!   its live rows need, judged by the size the last scrub's fresh file had for its live bytes.
//!
//! None starts within [`ScrubConfig::retry_gap`] of the last attempt, whatever its result. A scrub needs free disk
//! for the fresh file (about the live bytes at the last scrub's density, or the whole file before the first) plus a
//! quarter for growth and [`ScrubConfig::reserve_bytes`] left over for everything else on the disk. It is skipped,
//! with a log line, when that isn't there, and stopped (the fresh file deleted) if free space falls under the
//! reserve while it runs.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use continuum_catalog::{Catalog, CatalogStorage, ScrubProgress};
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender};
use tracing::{info, warn};

const GIB: u64 = 1 << 30;

/// When and how fast the scrubber works.
#[derive(Clone, Debug)]
pub struct ScrubConfig {
    /// Whether the scrubber runs at all.
    pub enabled: bool,
    /// The first check after startup (the index load and the workers connecting come first).
    pub first_check: Duration,
    /// How often it checks whether a scrub should start, and logs the catalog's storage line.
    pub check_every: Duration,
    /// How long one step should take; the row budget halves after a step over it and doubles after one under half.
    pub step_target: Duration,
    /// The share of wall time spent stepping, in percent: after a step that took `t`, it sleeps
    /// `t * (100 - duty) / duty`.
    pub duty_percent: u32,
    /// Below this the file is too small for the waste and first-count triggers to be worth a scrub (a due scrub
    /// still runs: on a small file it is quick).
    pub min_file_bytes: u64,
    /// The least reclaimable space that starts a scrub for waste.
    pub min_waste_bytes: u64,
    /// And the least share of the file it must be, in percent.
    pub min_waste_percent: u64,
    /// The least time between two scrubs started because a purge finished.
    pub due_gap: Duration,
    /// The least time after any attempt (swapped, skipped, stopped, or failed) before the next.
    pub retry_gap: Duration,
    /// Free disk a scrub must leave for everything else on the disk (capture, the tiers).
    pub reserve_bytes: u64,
    /// The first step's row budget.
    pub initial_budget: usize,
    /// The largest row budget a step grows to.
    pub max_budget: usize,
}

impl Default for ScrubConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            first_check: Duration::from_secs(5 * 60),
            check_every: Duration::from_secs(10 * 60),
            step_target: Duration::from_millis(50),
            duty_percent: 25,
            min_file_bytes: 256 << 20,
            min_waste_bytes: GIB,
            min_waste_percent: 25,
            due_gap: Duration::from_secs(24 * 60 * 60),
            retry_gap: Duration::from_secs(60 * 60),
            reserve_bytes: 8 * GIB,
            initial_budget: 256,
            max_budget: 65_536,
        }
    }
}

/// Why a scrub starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    /// No scrub has counted this file's live bytes yet.
    Uncounted,
    /// A purge finished since the last scrub (or one was requested), and the last was over `due_gap` ago.
    Due,
    /// The file holds this many bytes more than its live rows need.
    Waste(u64),
}

impl Reason {
    fn label(self) -> &'static str {
        match self {
            Self::Uncounted => "live bytes not counted yet",
            Self::Due => "a purge finished since the last scrub",
            Self::Waste(_) => "reclaimable space",
        }
    }
}

/// The bytes a dense copy of the catalog takes, per live byte: what the fresh file had in the last scrub with at
/// least 1 GiB live, between 1 and 4, or 1.5 before any such scrub. A small scrub's density isn't the catalog's.
#[must_use]
pub fn density(storage: &CatalogStorage) -> f64 {
    match storage.density_scrub {
        Some(last) if last.live_bytes > 0 => (last.file_after as f64 / last.live_bytes as f64).clamp(1.0, 4.0),
        _ => 1.5,
    }
}

/// How long a dense copy of the catalog would be: its live bytes at the last scrub's density, or the whole file
/// before any scrub counted them.
#[must_use]
pub fn dense_bytes(storage: &CatalogStorage) -> u64 {
    match storage.live_bytes {
        Some(live) => (live as f64 * density(storage)) as u64,
        None => storage.file_bytes,
    }
}

/// The space a scrub would give back: the file less a dense copy of it, or `None` before its live bytes are counted.
#[must_use]
pub fn reclaimable(storage: &CatalogStorage) -> Option<u64> {
    storage.live_bytes.map(|_| storage.file_bytes.saturating_sub(dense_bytes(storage)))
}

/// Whether a scrub should start now, and why. `last_attempt_ns` is the last attempt this run made (any result).
#[must_use]
pub fn reason(
    now_ns: u64,
    storage: &CatalogStorage,
    due: bool,
    last_attempt_ns: Option<u64>,
    config: &ScrubConfig,
) -> Option<Reason> {
    let since = |at: u64| Duration::from_nanos(now_ns.saturating_sub(at));
    if last_attempt_ns.is_some_and(|at| since(at) < config.retry_gap) {
        return None;
    }
    let big = storage.file_bytes >= config.min_file_bytes;
    match reclaimable(storage) {
        None if big => return Some(Reason::Uncounted),
        Some(waste)
            if big
                && waste >= config.min_waste_bytes
                && u128::from(waste) * 100 >= u128::from(storage.file_bytes) * u128::from(config.min_waste_percent) =>
        {
            return Some(Reason::Waste(waste));
        }
        _ => {}
    }
    let last_scrub = storage.last_scrub.map(|last| last.at_ns);
    (due && last_scrub.is_none_or(|at| since(at) >= config.due_gap)).then_some(Reason::Due)
}

/// The free disk a scrub needs before it starts: a dense copy, a quarter more for growth while it runs, and the
/// reserve.
#[must_use]
pub fn needed_bytes(storage: &CatalogStorage, config: &ScrubConfig) -> u64 {
    let copy = dense_bytes(storage);
    copy.saturating_add(copy / 4).saturating_add(config.reserve_bytes)
}

/// The bytes free to an unprivileged writer on the filesystem holding `path`.
#[must_use]
pub fn free_bytes(path: &Path) -> Option<u64> {
    let dir = path.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let stat = rustix::fs::statvfs(dir).ok()?;
    Some(stat.f_bavail.saturating_mul(stat.f_frsize))
}

/// How the last attempt ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// The fresh file is in place.
    Swapped {
        /// The old file's length.
        before: u64,
        /// The fresh file's length.
        after: u64,
    },
    /// Not started: too little free disk.
    NoRoom {
        /// Free bytes on the disk.
        free: u64,
        /// What it needed.
        needed: u64,
    },
    /// Stopped part way, its fresh file deleted, because free space fell under the reserve.
    OutOfRoom {
        /// Free bytes on the disk when it stopped.
        free: u64,
    },
    /// A step failed; its fresh file was deleted.
    Failed(String),
}

impl Outcome {
    fn label(&self) -> String {
        match self {
            Self::Swapped { before, after } => format!("swapped, {before} to {after} bytes"),
            Self::NoRoom { free, needed } => format!("skipped, {free} bytes free of {needed} needed"),
            Self::OutOfRoom { free } => format!("stopped, free disk fell to {free} bytes"),
            Self::Failed(e) => format!("failed: {e}"),
        }
    }
}

/// The running scrubber. Dropping it stops the thread at its next step; a scrub cut short that way leaves its
/// fresh file deleted (or, if the process exits first, for the next open to delete).
#[derive(Debug)]
pub struct Scrubber {
    stop: Option<Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Scrubber {
    /// Stops the thread and waits for it.
    pub fn stop(mut self) {
        self.stop.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for Scrubber {
    fn drop(&mut self) {
        self.stop.take();
    }
}

type FreeBytes = Box<dyn Fn(&Path) -> Option<u64> + Send>;

/// Starts the scrubber for a catalog kept in a file. `None` when it is turned off or the catalog has no file.
#[must_use]
pub fn spawn(catalog: Arc<dyn Catalog>, config: ScrubConfig) -> Option<Scrubber> {
    spawn_with(catalog, config, Box::new(free_bytes))
}

fn spawn_with(catalog: Arc<dyn Catalog>, config: ScrubConfig, free: FreeBytes) -> Option<Scrubber> {
    if !config.enabled || !matches!(catalog.storage(), Ok(Some(_))) {
        return None;
    }
    let (stop, stopped) = crossbeam_channel::bounded(0);
    // Its log lines go where the caller's do (the global subscriber in the broker, a test's own in tests).
    let dispatch = tracing::dispatcher::get_default(Clone::clone);
    let thread = std::thread::Builder::new()
        .name("continuum-scrub".into())
        .spawn(move || {
            tracing::dispatcher::with_default(&dispatch, || {
                // At normal priority, the duty cycle being the throttle: a lowered priority could keep the swap,
                // which holds the catalog alone, waiting for a CPU on a loaded machine while every commit waits.
                let mut run = Run { catalog, config, free, stopped, last: None };
                // A panic here loses only the scrubber; the catalog stays whole (open() settles a cut-short scrub).
                if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run.until_stopped())).is_err() {
                    warn!("the catalog scrubber panicked and stopped");
                    let _ = run.catalog.abort_scrub();
                }
            });
        })
        .ok()?;
    Some(Scrubber { stop: Some(stop), thread: Some(thread) })
}

fn now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
}

struct Run {
    catalog: Arc<dyn Catalog>,
    config: ScrubConfig,
    free: FreeBytes,
    stopped: Receiver<()>,
    /// When this run's last attempt ended (UTC ns), and how.
    last: Option<(u64, Outcome)>,
}

/// Why a scrub's steps ended without swapping.
enum Cut {
    Stop,
    Outcome(Outcome),
}

impl Run {
    /// Sleeps for `d`, or returns false once the scrubber is stopped.
    fn sleep(&self, d: Duration) -> bool {
        matches!(self.stopped.recv_timeout(d), Err(RecvTimeoutError::Timeout))
    }

    fn until_stopped(&mut self) {
        let mut wait = self.config.first_check;
        while self.sleep(wait) {
            wait = self.config.check_every;
            if let Err(e) = self.check() {
                warn!(error = %e, "couldn't check the catalog's storage for a scrub");
            }
        }
        // Stopped mid-scrub: delete the fresh file now rather than at the next open.
        if let Ok(true) = self.catalog.abort_scrub() {
            info!("catalog scrub stopped with the broker; its fresh file was deleted");
        }
    }

    /// One check: the storage line, then a scrub if one should start and there's room.
    fn check(&mut self) -> continuum_catalog::Result<()> {
        let Some(storage) = self.catalog.storage()? else { return Ok(()) };
        let due = self.catalog.scrub_needed()?;
        let free = (self.free)(&storage.path);
        self.log_storage(&storage, due, free);
        let Some(why) = reason(now_ns(), &storage, due, self.last.as_ref().map(|(at, _)| *at), &self.config) else {
            return Ok(());
        };
        let needed = needed_bytes(&storage, &self.config);
        let free = free.unwrap_or(0);
        if free < needed {
            warn!(
                reason = why.label(),
                free_bytes = free,
                needed_bytes = needed,
                file_bytes = storage.file_bytes,
                "catalog scrub skipped: not enough free disk for the fresh copy and the reserve"
            );
            self.last = Some((now_ns(), Outcome::NoRoom { free, needed }));
            return Ok(());
        }
        if why != Reason::Due {
            self.catalog.request_scrub()?;
        }
        info!(
            reason = why.label(),
            reclaimable_bytes = ?reclaimable(&storage),
            file_bytes = storage.file_bytes,
            live_bytes = ?storage.live_bytes,
            free_bytes = free,
            "catalog scrub started"
        );
        let began = Instant::now();
        let outcome = match self.scrub() {
            Ok((rows, active)) => {
                let after = self.catalog.storage()?;
                let after_bytes = after.as_ref().map_or(0, |s| s.file_bytes);
                info!(
                    elapsed = ?began.elapsed(),
                    stepping = ?active,
                    rows,
                    file_bytes_before = storage.file_bytes,
                    file_bytes_after = after_bytes,
                    live_bytes = ?after.as_ref().and_then(|s| s.live_bytes),
                    "catalog scrub finished"
                );
                Outcome::Swapped { before: storage.file_bytes, after: after_bytes }
            }
            Err(Cut::Stop) => return Ok(()),
            Err(Cut::Outcome(outcome)) => {
                let _ = self.catalog.abort_scrub();
                warn!(elapsed = ?began.elapsed(), result = %outcome.label(), "catalog scrub did not finish");
                outcome
            }
        };
        self.last = Some((now_ns(), outcome));
        if let Some(storage) = self.catalog.storage()? {
            self.log_storage(&storage, self.catalog.scrub_needed()?, (self.free)(&storage.path));
        }
        Ok(())
    }

    /// Steps a scrub to its swap. The rows it copied and the time spent stepping, or why it stopped.
    fn scrub(&self) -> Result<(u64, Duration), Cut> {
        let path = self.catalog.storage().ok().flatten().map(|s| s.path).unwrap_or_default();
        let mut budget = self.config.initial_budget.max(1);
        let mut active = Duration::ZERO;
        let duty = self.config.duty_percent.clamp(1, 100);
        loop {
            if let Some(free) = (self.free)(&path)
                && free < self.config.reserve_bytes
            {
                return Err(Cut::Outcome(Outcome::OutOfRoom { free }));
            }
            let started = Instant::now();
            let step = self.catalog.scrub_step(budget);
            let took = started.elapsed();
            active += took;
            match step {
                Ok(ScrubProgress::Swapped) => {
                    let rows =
                        self.catalog.storage().ok().flatten().and_then(|s| s.last_scrub).map_or(0, |last| last.rows);
                    return Ok((rows, active));
                }
                Ok(ScrubProgress::Copying { .. }) => {}
                Ok(ScrubProgress::Idle) => return Err(Cut::Outcome(Outcome::Failed("nothing was due".into()))),
                Err(e) => return Err(Cut::Outcome(Outcome::Failed(e.to_string()))),
            }
            budget = if took > self.config.step_target {
                budget / 2
            } else if took < self.config.step_target / 2 {
                budget.saturating_mul(2)
            } else {
                budget
            }
            .clamp(1, self.config.max_budget.max(1));
            if !self.sleep(took * (100 - duty) / duty) {
                return Err(Cut::Stop);
            }
        }
    }

    fn log_storage(&self, storage: &CatalogStorage, due: bool, free: Option<u64>) {
        let (last_at, last_result) = match (&self.last, storage.last_scrub) {
            // This run's last attempt, unless the catalog's record of a swap is newer.
            (Some((at, outcome)), last) if last.is_none_or(|l| l.at_ns <= *at) => (Some(*at), outcome.label()),
            (_, Some(last)) => {
                (Some(last.at_ns), Outcome::Swapped { before: last.file_before, after: last.file_after }.label())
            }
            (Some((at, outcome)), None) => (Some(*at), outcome.label()),
            (None, None) => (None, "never scrubbed".to_owned()),
        };
        let last_at = last_at.map(|ns| {
            chrono::DateTime::from_timestamp_nanos(i64::try_from(ns).unwrap_or(i64::MAX))
                .to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
        });
        info!(
            file_bytes = storage.file_bytes,
            disk_bytes = storage.disk_bytes,
            live_bytes = ?storage.live_bytes,
            reclaimable_bytes = ?reclaimable(storage),
            free_disk_bytes = ?free,
            scrub_due = due,
            scrub_running = storage.scrub_file_bytes.is_some(),
            last_scrub_at = ?last_at,
            last_scrub_result = %last_result,
            "catalog storage"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use continuum_catalog::{LastScrub, RedbCatalog};

    const NOW: u64 = 1_000_000 * 3_600 * 1_000_000_000;
    const HOUR: u64 = 3_600 * 1_000_000_000;

    /// As the catalog reports it: the last scrub stands for the density when it had 1 GiB live or more.
    fn storage(file: u64, live: Option<u64>, last: Option<LastScrub>) -> CatalogStorage {
        let density = last.filter(|l| l.live_bytes >= LastScrub::DENSITY_MIN_LIVE);
        with_density(file, live, last, density)
    }

    fn with_density(
        file: u64,
        live: Option<u64>,
        last: Option<LastScrub>,
        density: Option<LastScrub>,
    ) -> CatalogStorage {
        CatalogStorage {
            path: "/nowhere/catalog.redb".into(),
            file_bytes: file,
            disk_bytes: file,
            live_bytes: live,
            scrub_file_bytes: None,
            last_scrub: last,
            density_scrub: density,
        }
    }

    #[test]
    fn a_small_scrubs_density_doesnt_hide_a_large_catalogs_waste() {
        let c = ScrubConfig::default();
        // devbox on 2026-11-02: an 85 MB catalog scrubbed to 67 MB for 27 MB live (density 2.47), then grown to
        // 4.84 GB with 2.6 GB live. That density predicts a 6.4 GB copy, so no waste at all.
        let tiny = scrubbed(6, 67 << 20, 27 << 20);
        let grown = |density| with_density(9 * GIB, Some(4 * GIB), tiny, density);
        assert_eq!(density(&grown(None)), 1.5, "no large scrub yet: the default");
        // A larger scrub before it (1.6 bytes per live byte) is kept for the estimate.
        let earlier = scrubbed(30, 8 * GIB, 5 * GIB);
        assert_eq!(density(&grown(earlier)), 1.6);
        assert_eq!(reclaimable(&grown(earlier)), Some(9 * GIB - (4 * GIB) * 16 / 10));
        assert_eq!(reason(NOW, &grown(earlier), false, None, &c), Some(Reason::Waste(9 * GIB - (4 * GIB) * 16 / 10)));
    }

    /// A scrub `hours_ago` that left a file of `after` bytes for `live` live bytes.
    fn scrubbed(hours_ago: u64, after: u64, live: u64) -> Option<LastScrub> {
        Some(LastScrub { at_ns: NOW - hours_ago * HOUR, file_before: 0, file_after: after, live_bytes: live, rows: 1 })
    }

    #[test]
    fn when_a_scrub_starts() {
        let c = ScrubConfig::default();
        // The live catalog on devbox after 45 hours: 8 GiB, never scrubbed, purges finishing every few minutes.
        assert_eq!(reason(NOW, &storage(8 * GIB, None, None), true, None, &c), Some(Reason::Uncounted));
        // Too small to count for its own sake; a due scrub still runs.
        assert_eq!(reason(NOW, &storage(100 << 20, None, None), false, None, &c), None);
        assert_eq!(reason(NOW, &storage(100 << 20, None, None), true, None, &c), Some(Reason::Due));
        // Freshly scrubbed to 4 GiB for 2.5 GiB live (density 1.6): a due purge waits out the day.
        let fresh = scrubbed(2, 4 * GIB, 5 * GIB / 2);
        assert_eq!(reason(NOW, &storage(4 * GIB, Some(5 * GIB / 2), fresh), true, None, &c), None);
        assert_eq!(
            reason(NOW, &storage(4 * GIB, Some(5 * GIB / 2), scrubbed(25, 4 * GIB, 5 * GIB / 2)), true, None, &c),
            Some(Reason::Due)
        );
        assert_eq!(
            reason(NOW, &storage(4 * GIB, Some(5 * GIB / 2), scrubbed(25, 4 * GIB, 5 * GIB / 2)), false, None, &c),
            None
        );
        // Grown to 6 GiB with the same live bytes: 2 GiB (a third) is reclaimable, so it runs early.
        assert_eq!(
            reason(NOW, &storage(6 * GIB, Some(5 * GIB / 2), fresh), false, None, &c),
            Some(Reason::Waste(2 * GIB))
        );
        // 1.5 GiB of 12 GiB is under a quarter; 0.5 GiB of 1 GiB is under the 1 GiB floor.
        let big = scrubbed(2, 10 * GIB, 10 * GIB);
        assert_eq!(reason(NOW, &storage(23 * GIB / 2, Some(10 * GIB), big), false, None, &c), None);
        let small = scrubbed(2, GIB / 2, GIB / 2);
        assert_eq!(reason(NOW, &storage(GIB, Some(GIB / 2), small), false, None, &c), None);
        // Nothing within an hour of the last attempt, whatever it was.
        assert_eq!(reason(NOW, &storage(8 * GIB, None, None), true, Some(NOW - HOUR / 2), &c), None);
        assert_eq!(reason(NOW, &storage(8 * GIB, None, None), true, Some(NOW - 2 * HOUR), &c), Some(Reason::Uncounted));
    }

    #[test]
    fn the_room_a_scrub_needs() {
        let c = ScrubConfig::default();
        // Before the first scrub: the whole file, a quarter more, and the reserve.
        assert_eq!(needed_bytes(&storage(8 * GIB, None, None), &c), 10 * GIB + 8 * GIB);
        // After: the live bytes at the last scrub's density.
        let s = storage(6 * GIB, Some(4 * GIB), scrubbed(1, 6 * GIB, 4 * GIB));
        assert_eq!(dense_bytes(&s), 6 * GIB);
        assert_eq!(needed_bytes(&s, &c), 6 * GIB + 3 * GIB / 2 + 8 * GIB);
        // A density outside 1 to 4 is clamped.
        assert!((density(&storage(GIB, Some(GIB), scrubbed(1, 100 * GIB, GIB))) - 4.0).abs() < f64::EPSILON);
        assert!((density(&storage(GIB, Some(GIB), scrubbed(1, GIB / 2, GIB))) - 1.0).abs() < f64::EPSILON);
        assert!(free_bytes(Path::new("/tmp/x")).is_some_and(|b| b > 0));
    }

    fn payload(n: u32, stream: continuum_core::StreamId) -> continuum_catalog::NewPayload {
        let mut b = [0; 16];
        b[12..].copy_from_slice(&n.to_be_bytes());
        let spec = continuum_core::HeaderSpec::raw(continuum_core::Uuid(b), stream, u64::from(n), u64::from(n) + 1);
        continuum_catalog::NewPayload {
            header: continuum_core::HeaderBuf::build(&spec).unwrap(),
            location: None,
            coverage: None,
        }
    }

    fn info() -> continuum_catalog::StreamInfo {
        continuum_catalog::StreamInfo {
            name: "cam0".into(),
            kind: continuum_catalog::StreamKind::Sensor,
            layer: continuum_core::DomainLayer::Layer1Raw,
            format: continuum_core::PayloadFormat::RawBytes,
            shape_name: "NV12".into(),
            shape_version: 1,
            rate_hz: None,
            anchor: None,
        }
    }

    fn quick() -> ScrubConfig {
        ScrubConfig {
            first_check: Duration::from_millis(1),
            check_every: Duration::from_millis(5),
            min_file_bytes: 0,
            reserve_bytes: 0,
            initial_budget: 16,
            max_budget: 16,
            ..ScrubConfig::default()
        }
    }

    fn logs_of(emit: impl FnOnce()) -> String {
        #[derive(Clone, Default)]
        struct Buf(Arc<std::sync::Mutex<Vec<u8>>>);
        impl std::io::Write for Buf {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(b);
                Ok(b.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let buf = Buf::default();
        let out = buf.clone();
        let sub = tracing_subscriber::fmt().with_writer(move || out.clone()).with_ansi(false).finish();
        tracing::subscriber::with_default(sub, emit);
        String::from_utf8(buf.0.lock().unwrap().clone()).unwrap()
    }

    /// Waits up to 30 s for `done`.
    fn until(mut done: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(30);
        while !done() {
            assert!(Instant::now() < deadline, "timed out");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn it_scrubs_an_uncounted_file_while_writes_go_on_and_logs_the_storage_line() {
        let dir = tempfile::tempdir().unwrap();
        let catalog = Arc::new(RedbCatalog::open(dir.path().join("catalog.redb")).unwrap());
        let s = catalog.register_stream(info()).unwrap();
        catalog.insert((1..=3_000).map(|n| payload(n, s)).collect()).unwrap();
        let logs = logs_of(|| {
            let dynamic: Arc<dyn Catalog> = catalog.clone();
            let scrubber = spawn_with(dynamic, quick(), Box::new(|_| Some(u64::MAX))).unwrap();
            // Ingest goes on beside the scrub, every write landing.
            let mut n = 10_000;
            until(|| {
                catalog.insert(vec![payload(n, s)]).unwrap();
                n += 1;
                catalog.storage().unwrap().unwrap().last_scrub.is_some()
            });
            scrubber.stop();
            for k in 10_000..n {
                assert!(
                    catalog
                        .get(continuum_core::Uuid({
                            let mut b = [0; 16];
                            b[12..].copy_from_slice(&k.to_be_bytes());
                            b
                        }))
                        .unwrap()
                        .is_some()
                );
            }
        });
        assert!(logs.contains("catalog scrub started") && logs.contains("live bytes not counted yet"), "{logs}");
        assert!(logs.contains("catalog scrub finished"), "{logs}");
        let line = logs.lines().rfind(|l| l.contains("catalog storage")).unwrap();
        for field in [
            "file_bytes=",
            "live_bytes=Some(",
            "reclaimable_bytes=Some(",
            "last_scrub_at=Some(",
            "last_scrub_result=swapped",
        ] {
            assert!(line.contains(field), "{field} missing from {line}");
        }
        assert!(catalog.storage().unwrap().unwrap().live_bytes.is_some());
    }

    #[test]
    fn it_skips_without_room_and_stops_when_free_space_runs_low() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("catalog.redb");
        let catalog = Arc::new(RedbCatalog::open(&path).unwrap());
        let s = catalog.register_stream(info()).unwrap();
        catalog.insert((1..=3_000).map(|n| payload(n, s)).collect()).unwrap();
        let config = ScrubConfig { reserve_bytes: 1 << 40, retry_gap: Duration::from_secs(3_600), ..quick() };
        // Short of the reserve before it starts: skipped once, then left for the retry gap.
        let logs = logs_of(|| {
            let dynamic: Arc<dyn Catalog> = catalog.clone();
            let scrubber = spawn_with(dynamic, config.clone(), Box::new(|_| Some(1 << 30))).unwrap();
            std::thread::sleep(Duration::from_millis(100));
            scrubber.stop();
        });
        assert_eq!(logs.matches("catalog scrub skipped: not enough free disk").count(), 1, "{logs}");
        assert!(logs.contains("last_scrub_result=skipped"), "{logs}");
        assert!(!catalog.scrub_needed().unwrap(), "a skipped scrub asks for nothing");
        // Room at the start, then the disk fills: stopped, its fresh file deleted, the catalog whole.
        let checks = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let logs = logs_of(|| {
            let dynamic: Arc<dyn Catalog> = catalog.clone();
            let counter = Arc::clone(&checks);
            let free = move |_: &Path| {
                let n = counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Some(if n < 3 { u64::MAX } else { 1 << 30 })
            };
            let scrubber = spawn_with(dynamic, config, Box::new(free)).unwrap();
            until(|| checks.load(std::sync::atomic::Ordering::SeqCst) > 5);
            std::thread::sleep(Duration::from_millis(50));
            scrubber.stop();
        });
        assert!(logs.contains("catalog scrub did not finish") && logs.contains("stopped, free disk fell"), "{logs}");
        let mut scrub = path.as_os_str().to_owned();
        scrub.push(".scrub");
        assert!(!std::path::PathBuf::from(scrub).exists(), "the fresh file is gone");
        assert!(catalog.storage().unwrap().unwrap().last_scrub.is_none());
        assert!(catalog.scrub_needed().unwrap(), "still due for next time");
    }

    #[test]
    fn stopping_mid_scrub_deletes_the_fresh_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("catalog.redb");
        let catalog = Arc::new(RedbCatalog::open(&path).unwrap());
        let s = catalog.register_stream(info()).unwrap();
        catalog.insert((1..=5_000).map(|n| payload(n, s)).collect()).unwrap();
        // One row a step, a 1% duty: the scrub is still copying when it is stopped.
        let config = ScrubConfig { initial_budget: 1, max_budget: 1, duty_percent: 1, ..quick() };
        let dynamic: Arc<dyn Catalog> = catalog.clone();
        let scrubber = spawn_with(dynamic, config, Box::new(|_| Some(u64::MAX))).unwrap();
        until(|| catalog.storage().unwrap().unwrap().scrub_file_bytes.is_some());
        scrubber.stop();
        assert_eq!(catalog.storage().unwrap().unwrap().scrub_file_bytes, None);
        let mut scrub = path.as_os_str().to_owned();
        scrub.push(".scrub");
        assert!(!std::path::PathBuf::from(scrub).exists());
        drop(catalog);
        let again = RedbCatalog::open(&path).unwrap();
        assert!(
            again
                .get(continuum_core::Uuid({
                    let mut b = [0; 16];
                    b[12..].copy_from_slice(&5_000u32.to_be_bytes());
                    b
                }))
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn a_catalog_in_memory_or_a_disabled_scrubber_starts_no_thread() {
        let memory: Arc<dyn Catalog> = Arc::new(continuum_catalog::MemoryCatalog::new());
        assert!(spawn(memory, ScrubConfig::default()).is_none());
        let dir = tempfile::tempdir().unwrap();
        let file: Arc<dyn Catalog> = Arc::new(RedbCatalog::open(dir.path().join("catalog.redb")).unwrap());
        assert!(spawn(file, ScrubConfig { enabled: false, ..ScrubConfig::default() }).is_none());
    }
}
