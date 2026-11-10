//! The opt-in AV1 archive pass's scheduling (docs/STORAGE.md, "AV1 archive"): which camera chunks to
//! re-encode, when, and on what thread. The re-encoding itself is [`crate::av1`].
//!
//! Off unless `video_archive.av1` is on (the tier worker reads it from its environment, [`ArchiveSettings::from_env`]).
//! Then a camera chunk ([`ChunkFormat::H264Au`]) recorded more than `after_hours` ago becomes AV1, newest first: each
//! pass looks at the newly eligible time first, then walks the backlog back a slice at a time, down to two days
//! before the video window would age the chunk out (converting footage about to be deleted saves nothing). Turning
//! it off stops the pass; chunks already converted stay AV1, and every reader reads both.
//!
//! One chunk is encoded at a time, on a thread of its own at the lowest CPU priority (nice 19 and `SCHED_IDLE` on
//! Linux; SVT-AV1's threads inherit both), with SVT-AV1's parallelism capped (`threads`, 4 by default). So it only
//! takes CPU nothing else wants: the measured cost is about 3 to 4 cores, all day, for six cameras. The service
//! swaps the finished chunk in between its other work, never during a purge, and tells the catalog its new offsets
//! with `RewriteChunk`. A crash between the swap and that message leaves an AV1 file the catalog has the old length
//! for; the next scan finds it and sends `RewriteChunk` again.

use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use continuum_core::{StorageLocator, StorageTier, Uuid};

use crate::chunk::{ChunkFormat, format_of};

/// How much recorded time one backlog scan covers, per camera.
pub const SLICE_NS: u64 = 10 * 60 * 1_000_000_000;
/// Chunks this close to the video window's end are left for it to age out.
pub const EXPIRY_MARGIN: Duration = Duration::from_secs(2 * 24 * 3600);
/// How often the pass logs what it did.
pub const REPORT_EVERY: Duration = Duration::from_secs(10 * 60);
/// Encodes failing this many times in a row (no encoder in this FFmpeg, say) turn the pass off for the run.
pub const MAX_FAILURES: u32 = 3;
/// How far back the backlog goes when the video window is off (nothing ages out, so there's no end to measure from).
pub const NO_WINDOW_BACKLOG: Duration = Duration::from_secs(30 * 24 * 3600);
/// Scans at most this often: an empty stretch of backlog is walked a slice a second, not a slice a tick.
pub const SCAN_EVERY: Duration = Duration::from_secs(1);
/// The fewest hours a chunk waits: the recent window ([`crate::recent`]) and live readers only ever see H.264.
pub const MIN_AFTER_HOURS: u64 = 1;

/// The pass's settings, from the tier worker's environment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArchiveSettings {
    /// How long after it was recorded a chunk becomes AV1.
    pub after: Duration,
    /// The video window, when it is on: chunks within [`EXPIRY_MARGIN`] of it are left alone. Off, the backlog goes
    /// back [`NO_WINDOW_BACKLOG`].
    pub window: Option<Duration>,
    /// SVT-AV1 preset (0 to 13).
    pub preset: u8,
    /// SVT-AV1 CRF (0 to 63).
    pub crf: u8,
    /// SVT-AV1's level of parallelism.
    pub threads: u32,
}

impl ArchiveSettings {
    /// The environment the service launcher sets from `[video_archive]` and `[video_window]`
    /// (services/services.toml): `CONTINUUM_AV1_ARCHIVE` (`True` turns it on), `CONTINUUM_AV1_AFTER_HOURS`,
    /// `CONTINUUM_AV1_PRESET`, `CONTINUUM_AV1_CRF`, `CONTINUUM_AV1_THREADS`, `CONTINUUM_VIDEO_WINDOW` and
    /// `CONTINUUM_VIDEO_WINDOW_DAYS`. `None` when the pass is off; an unset number takes its default.
    ///
    /// # Errors
    ///
    /// A value that isn't a number in range, so a typo is said at start rather than ignored.
    pub fn from_env(get: impl Fn(&str) -> Option<String>) -> Result<Option<Self>, String> {
        let on = |key: &str| get(key).is_some_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "true" | "1"));
        if !on("CONTINUUM_AV1_ARCHIVE") {
            return Ok(None);
        }
        let number = |key: &str, default: u64, range: std::ops::RangeInclusive<u64>| -> Result<u64, String> {
            let Some(text) = get(key).filter(|v| !v.trim().is_empty()) else { return Ok(default) };
            let value: u64 = text.trim().parse().map_err(|_| format!("{key}={text} isn't a whole number"))?;
            if range.contains(&value) {
                Ok(value)
            } else {
                Err(format!("{key}={value} is outside {}..={}", range.start(), range.end()))
            }
        };
        let hours = number("CONTINUUM_AV1_AFTER_HOURS", 24, MIN_AFTER_HOURS..=24 * 365)?;
        let days = number("CONTINUUM_VIDEO_WINDOW_DAYS", 30, 1..=36_500)?;
        Ok(Some(Self {
            after: Duration::from_secs(hours * 3600),
            window: on("CONTINUUM_VIDEO_WINDOW").then(|| Duration::from_secs(days * 86_400)),
            preset: u8::try_from(number("CONTINUUM_AV1_PRESET", 8, 0..=13)?).unwrap_or(8),
            crf: u8::try_from(number("CONTINUUM_AV1_CRF", 52, 0..=63)?).unwrap_or(52),
            threads: u32::try_from(number("CONTINUUM_AV1_THREADS", 4, 1..=64)?).unwrap_or(4),
        }))
    }

    /// The recorded time eligible now, `[lo, hi)`: older than `after`, and not within [`EXPIRY_MARGIN`] of the video
    /// window's end (or [`NO_WINDOW_BACKLOG`] back, with the window off). Empty when the window is no longer than the
    /// wait and the margin.
    #[must_use]
    pub fn eligible(&self, now_ns: u64) -> (u64, u64) {
        let ns = |d: Duration| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX);
        let hi = now_ns.saturating_sub(ns(self.after));
        let lo = match self.window {
            Some(w) => now_ns.saturating_sub(ns(w).saturating_sub(ns(EXPIRY_MARGIN))),
            None => hi.saturating_sub(ns(NO_WINDOW_BACKLOG)),
        };
        (lo.min(hi), hi)
    }
}

/// A chunk to convert: the catalog's id for it, where it is, and the tier its locator names.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidate {
    /// The chunk payload's id, for `RewriteChunk`.
    pub chunk: Uuid,
    /// Its file.
    pub path: PathBuf,
    /// The tier its locator names, kept on the rewritten locator.
    pub tier: StorageTier,
    /// Its start, for newest-first order.
    pub start_ns: u64,
}

/// What the encoder thread made of a candidate.
#[derive(Debug)]
pub enum Outcome {
    /// Converted, ready to swap in.
    #[cfg(feature = "ffmpeg")]
    Ready(crate::av1::Ready),
    /// Left as it is, and why.
    Kept(String),
    /// The encoder failed.
    Failed(String),
}

/// What the pass has done this run.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ArchiveCounts {
    /// Chunks converted and swapped in.
    pub converted: u64,
    /// Their bytes before and after.
    pub bytes_before: u64,
    /// Their bytes after.
    pub bytes_after: u64,
    /// Chunks left as H.264, by reason.
    pub kept: BTreeMap<String, u64>,
    /// Converted chunks not swapped in: the file changed (a purge) or a purge was running.
    pub dropped: u64,
    /// `RewriteChunk` sent again for a swap a crash interrupted.
    pub resent: u64,
}

/// The pass: the queue, the scan's place, and the encoder thread.
pub struct Archive {
    settings: ArchiveSettings,
    queue: VecDeque<Candidate>,
    /// The newest recorded time scanned so far (newly eligible time is scanned from here).
    fresh_from: Option<u64>,
    /// Where the backlog scan continues down from.
    back: u64,
    backlog_done: bool,
    to_worker: mpsc::Sender<Candidate>,
    from_worker: mpsc::Receiver<(Candidate, Outcome)>,
    in_flight: bool,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    failures: u32,
    last_report: Instant,
    next_scan: Instant,
    /// What this run's pass has done.
    pub counts: ArchiveCounts,
}

impl Archive {
    /// Starts the encoder thread.
    ///
    /// # Errors
    ///
    /// Without the `ffmpeg` feature or an FFmpeg that has SVT-AV1 (no encoder), or when the thread can't be started.
    pub fn start(settings: ArchiveSettings) -> Result<Self, String> {
        if !cfg!(feature = "ffmpeg") {
            return Err("the AV1 archive needs the tier worker built with the ffmpeg feature".to_owned());
        }
        if !crate::probe::av1_encoder_available() {
            return Err("the AV1 archive needs an FFmpeg with SVT-AV1 (libsvtav1), which this one lacks".to_owned());
        }
        let (to_worker, jobs) = mpsc::channel::<Candidate>();
        let (results, from_worker) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let thread = {
            let stop = Arc::clone(&stop);
            std::thread::Builder::new()
                .name("av1-archive".to_owned())
                .spawn(move || {
                    lowest_priority();
                    for candidate in jobs {
                        if stop.load(Ordering::Relaxed) {
                            break;
                        }
                        let outcome = encode(&candidate.path, settings, &stop);
                        if results.send((candidate, outcome)).is_err() {
                            break;
                        }
                    }
                })
                .map_err(|e| format!("no AV1 archive thread: {e}"))?
        };
        Ok(Self {
            settings,
            queue: VecDeque::new(),
            fresh_from: None,
            back: 0,
            backlog_done: false,
            to_worker,
            from_worker,
            in_flight: false,
            stop,
            thread: Some(thread),
            failures: 0,
            last_report: Instant::now(),
            next_scan: Instant::now(),
            counts: ArchiveCounts::default(),
        })
    }

    /// The next finished candidate, if one is back from the encoder thread.
    pub fn finished(&mut self) -> Option<(Candidate, Outcome)> {
        let done = self.from_worker.try_recv().ok()?;
        self.in_flight = false;
        match &done.1 {
            Outcome::Failed(_) => self.failures += 1,
            Outcome::Kept(why) => {
                self.failures = 0;
                *self.counts.kept.entry(why.split(':').next().unwrap_or(why).to_owned()).or_default() += 1;
            }
            #[cfg(feature = "ffmpeg")]
            Outcome::Ready(_) => self.failures = 0,
        }
        Some(done)
    }

    /// Whether failing encodes should turn the pass off for this run.
    #[must_use]
    pub fn failing(&self) -> bool {
        self.failures >= MAX_FAILURES
    }

    /// Hands the encoder thread the next candidate when it's idle.
    pub fn feed(&mut self) {
        if self.in_flight {
            return;
        }
        if let Some(next) = self.queue.pop_front() {
            self.in_flight = self.to_worker.send(next).is_ok();
        }
    }

    /// Whether the next scan can run: the queue is empty, nothing is being encoded, and the last scan was at least
    /// [`SCAN_EVERY`] ago.
    #[must_use]
    pub fn idle(&self) -> bool {
        self.queue.is_empty() && !self.in_flight && Instant::now() >= self.next_scan
    }

    /// The recorded time the next scan covers, newest first: newly eligible time, else the backlog's next slice
    /// down, else `None` (all caught up until more time becomes eligible).
    pub fn next_range(&mut self, now_ns: u64) -> Option<(u64, u64)> {
        self.next_scan = Instant::now() + SCAN_EVERY;
        let (lo, hi) = self.settings.eligible(now_ns);
        let fresh = *self.fresh_from.get_or_insert_with(|| {
            self.back = hi;
            hi
        });
        // Newly eligible time waits until there's a minute of it (a tenth of the wait, for a short one), so the scan
        // doesn't query every camera every second.
        let batch = (u64::try_from(self.settings.after.as_nanos()).unwrap_or(u64::MAX) / 10).min(60_000_000_000);
        if hi >= fresh.saturating_add(batch.max(1)) {
            let end = hi.min(fresh.saturating_add(6 * SLICE_NS));
            self.fresh_from = Some(end);
            return Some((fresh, end));
        }
        if self.backlog_done || self.back <= lo {
            self.backlog_done = true;
            return None;
        }
        let start = self.back.saturating_sub(SLICE_NS).max(lo);
        let range = (start, self.back);
        self.back = start;
        Some(range)
    }

    /// Queues the H.264 chunks among `items` (a scan's query result: each item's id and locator, its uri resolved to the
    /// file's path) newest first, and
    /// returns the AV1 ones whose catalog locator has another length than the file (a swap a crash interrupted
    /// before its `RewriteChunk`).
    pub fn take(&mut self, items: Vec<(Uuid, StorageLocator)>) -> Vec<Candidate> {
        let mut found = Vec::new();
        let mut resend = Vec::new();
        for (chunk, locator) in items {
            let StorageLocator::Chunk { tier, uri, length, .. } = locator else { continue };
            let path = PathBuf::from(uri.strip_prefix("file://").unwrap_or(&uri));
            let candidate = Candidate { chunk, path, tier, start_ns: 0 };
            match format_of(&candidate.path) {
                Ok(ChunkFormat::H264Au) => found.push(candidate),
                Ok(ChunkFormat::Av1Au) if file_len(&candidate.path).is_some_and(|l| l != length) => {
                    resend.push(candidate);
                }
                _ => {}
            }
        }
        for c in &mut found {
            c.start_ns = start_of(&c.path);
        }
        found.sort_by_key(|c| std::cmp::Reverse(c.start_ns));
        found.dedup_by(|a, b| a.path == b.path);
        let queued: std::collections::BTreeSet<PathBuf> = self.queue.iter().map(|c| c.path.clone()).collect();
        self.queue.extend(found.into_iter().filter(|c| !queued.contains(&c.path)));
        resend
    }

    /// A log line every [`REPORT_EVERY`], when there is something to say.
    pub fn report(&mut self) -> Option<String> {
        if self.last_report.elapsed() < REPORT_EVERY {
            return None;
        }
        self.last_report = Instant::now();
        let c = &self.counts;
        let mb = |b: u64| b as f64 / 1e6;
        Some(format!(
            "AV1 archive: {} chunks converted ({:.1} MB to {:.1} MB), kept as H.264 {:?}, {} dropped for \
             a purge, {} catalog rewrites sent again, {} queued, backlog {}",
            c.converted,
            mb(c.bytes_before),
            mb(c.bytes_after),
            c.kept,
            c.dropped,
            c.resent,
            self.queue.len(),
            if self.backlog_done { "done" } else { "in progress" }
        ))
    }

    /// Stops the encoder thread (an encode in progress ends at its next frame) and waits for it.
    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.queue.clear();
        // Closing the job channel ends the thread's loop once its current encode returns.
        let (dead, _) = mpsc::channel();
        drop(std::mem::replace(&mut self.to_worker, dead));
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for Archive {
    fn drop(&mut self) {
        self.stop();
    }
}

fn file_len(path: &Path) -> Option<u64> {
    std::fs::metadata(path).ok().map(|m| m.len())
}

/// A chunk file's start, from its name (`<stream>-<created_ns>-<n>.cnk`), for ordering only.
fn start_of(path: &Path) -> u64 {
    path.file_stem()
        .and_then(|s| s.to_str())
        .and_then(|s| s.split('-').nth(1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

#[cfg(feature = "ffmpeg")]
fn encode(path: &Path, settings: ArchiveSettings, stop: &AtomicBool) -> Outcome {
    let av1 = crate::av1::Av1Settings { preset: settings.preset, crf: settings.crf, parallelism: settings.threads };
    match crate::av1::encode_chunk(path, av1, stop) {
        Ok(crate::av1::Encoded::Ready(ready)) => Outcome::Ready(ready),
        Ok(crate::av1::Encoded::Kept(why)) => Outcome::Kept(why),
        // A chunk the catalog still names that a purge or the video window removed since the scan.
        Err(crate::TierError::Chunk(crate::ChunkError::Io(e)) | crate::TierError::Io(e))
            if e.kind() == std::io::ErrorKind::NotFound =>
        {
            Outcome::Kept("gone".to_owned())
        }
        Err(crate::TierError::Chunk(e)) => Outcome::Kept(format!("unreadable: {e}")),
        Err(e) => Outcome::Failed(e.to_string()),
    }
}

#[cfg(not(feature = "ffmpeg"))]
fn encode(_path: &Path, _settings: ArchiveSettings, _stop: &AtomicBool) -> Outcome {
    Outcome::Failed("built without the ffmpeg feature".to_owned())
}

/// This thread (and the encoder threads it starts, which inherit it) at the lowest CPU priority.
fn lowest_priority() {
    #[cfg(target_os = "linux")]
    // SAFETY: plain syscalls on the calling thread with valid arguments; failure only leaves the priority as it was.
    unsafe {
        let tid = libc::syscall(libc::SYS_gettid) as libc::id_t;
        libc::setpriority(libc::PRIO_PROCESS, tid, 19);
        let param = libc::sched_param { sched_priority: 0 };
        libc::sched_setscheduler(0, libc::SCHED_IDLE, &param);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HOUR: u64 = 3_600_000_000_000;
    const DAY: u64 = 24 * HOUR;

    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: BTreeMap<String, String> = pairs.iter().map(|(k, v)| ((*k).to_owned(), (*v).to_owned())).collect();
        move |k| map.get(k).cloned()
    }

    #[test]
    fn it_is_off_unless_switched_on() {
        assert_eq!(ArchiveSettings::from_env(env(&[])), Ok(None));
        assert_eq!(ArchiveSettings::from_env(env(&[("CONTINUUM_AV1_ARCHIVE", "False")])), Ok(None));
    }

    #[test]
    fn on_it_takes_the_measured_defaults() {
        let on =
            ArchiveSettings::from_env(env(&[("CONTINUUM_AV1_ARCHIVE", "True"), ("CONTINUUM_VIDEO_WINDOW", "True")]));
        assert_eq!(
            on,
            Ok(Some(ArchiveSettings {
                after: Duration::from_secs(24 * 3600),
                window: Some(Duration::from_secs(30 * 86_400)),
                preset: 8,
                crf: 52,
                threads: 4,
            }))
        );
    }

    #[test]
    fn a_value_out_of_range_is_an_error_at_start() {
        for (key, value) in
            [("CONTINUUM_AV1_AFTER_HOURS", "0"), ("CONTINUUM_AV1_CRF", "64"), ("CONTINUUM_AV1_PRESET", "x")]
        {
            let got = ArchiveSettings::from_env(env(&[("CONTINUUM_AV1_ARCHIVE", "true"), (key, value)]));
            assert!(got.as_ref().is_err_and(|e| e.contains(key)), "{key}={value}: {got:?}");
        }
    }

    fn settings(window_days: Option<u64>) -> ArchiveSettings {
        ArchiveSettings {
            after: Duration::from_secs(24 * 3600),
            window: window_days.map(|d| Duration::from_secs(d * 86_400)),
            preset: 8,
            crf: 52,
            threads: 4,
        }
    }

    #[test]
    fn eligible_time_is_older_than_the_wait_and_short_of_the_windows_last_two_days() {
        let now = 100 * DAY;
        assert_eq!(settings(Some(30)).eligible(now), (72 * DAY, 99 * DAY));
        assert_eq!(settings(None).eligible(now), (69 * DAY, 99 * DAY), "no window: a month of backlog");
        assert_eq!(settings(Some(2)).eligible(now), (99 * DAY, 99 * DAY), "a window too short leaves nothing");
    }

    #[cfg(not(feature = "ffmpeg"))]
    #[test]
    fn starting_an_archive_without_ffmpeg_reports_the_missing_feature() {
        assert_eq!(
            Archive::start(settings(Some(30))).err().as_deref(),
            Some("the AV1 archive needs the tier worker built with the ffmpeg feature")
        );
    }

    #[cfg(feature = "ffmpeg")]
    #[test]
    fn scans_go_newest_first_then_take_newly_eligible_time_before_more_backlog() {
        if !crate::probe::av1_encoder_available() {
            eprintln!("this FFmpeg has no SVT-AV1 (the wheels' private build); skipped");
            return;
        }
        let mut a = Archive::start(settings(Some(30))).unwrap();
        let now = 100 * DAY;
        assert_eq!(a.next_range(now), Some((99 * DAY - SLICE_NS, 99 * DAY)), "the backlog's newest slice");
        assert_eq!(a.next_range(now), Some((99 * DAY - 2 * SLICE_NS, 99 * DAY - SLICE_NS)));
        let later = now + 5 * 60 * 1_000_000_000;
        assert_eq!(a.next_range(later), Some((99 * DAY, later - DAY)), "what became eligible meanwhile, first");
        assert_eq!(a.next_range(later), Some((99 * DAY - 3 * SLICE_NS, 99 * DAY - 2 * SLICE_NS)));
    }

    #[cfg(feature = "ffmpeg")]
    #[test]
    fn the_backlog_stops_two_days_before_the_window_ends() {
        if !crate::probe::av1_encoder_available() {
            eprintln!("this FFmpeg has no SVT-AV1 (the wheels' private build); skipped");
            return;
        }
        let mut a = Archive::start(settings(Some(4))).unwrap();
        let now = 100 * DAY;
        let mut ranges = Vec::new();
        while let Some(r) = a.next_range(now) {
            ranges.push(r);
        }
        assert_eq!(ranges.first().map(|r| r.1), Some(99 * DAY));
        assert_eq!(
            ranges.last().map(|r| r.0),
            Some(98 * DAY),
            "a 4-day window, less the day of waiting and the 2-day margin, leaves one day"
        );
        assert_eq!(ranges.len() as u64, DAY / SLICE_NS);
        assert_eq!(a.next_range(now), None, "caught up");
    }
}
