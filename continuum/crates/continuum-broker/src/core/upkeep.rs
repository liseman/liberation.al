//! Upkeep on the scheduler's tick: saving the cost models now and then, sampling the host clock into `CLOCK_MAP`,
//! and sweeping the `FRAMES` rows of old absorbed frames nothing names (#851). All are handed to the writer; the
//! scheduler thread only reads clocks and memory.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use tracing::{info, warn};

use continuum_catalog::ClockPoint;
use continuum_core::BootId;

use super::{Sched, Urgent};
use crate::{costs, host};

/// How often changed cost models are saved.
const SAVE_COSTS_EVERY: Duration = Duration::from_secs(300);
/// How often the clock is sampled when nothing happens.
const CLOCK_EVERY: Duration = Duration::from_secs(60);
/// A jump in the asleep time (continuous minus monotonic) past this means the machine slept: sample at once, since
/// the monotonic clock's mapping to UTC moved by the time asleep.
const WAKE_JUMP_NS: u64 = 1_000_000_000;
/// How often a sweep of the `FRAMES` rows starts (and once at startup), unless one is still running.
const PRUNE_FRAMES_EVERY: Duration = Duration::from_secs(600);
/// How long after its chunk ends an absorbed frame keeps its `FRAMES` row whatever: a child naming it as a parent
/// (ASR, a VLM, seconds late) still finds it. One per frame (#849) is about 10 million rows a day on four cameras.
const FRAME_ROWS_KEPT: Duration = Duration::from_secs(3600);

/// The upkeep's state.
#[derive(Debug, Default)]
pub(super) struct Upkeep {
    boot: Option<BootId>,
    costs: Costs,
    clock: Option<(Instant, u64)>,
    /// When the last `FRAMES` sweep was queued, and whether one is running.
    pruned_at: Option<Instant>,
    pruning: Arc<AtomicBool>,
}

impl Upkeep {
    pub(super) fn restart_checkpoint_clean(&self) -> bool {
        !self.pruning.load(Ordering::Acquire)
    }
}

/// When the cost models were last handed to the writer, and what was handed.
#[derive(Debug, Default)]
struct Costs {
    /// `None` until the server's startup restored what an earlier run saved: a save before then would write the
    /// empty set over it.
    last: Option<Instant>,
    /// What the writer has (as far as anyone knows): `None` after a failed write, so the next save writes again.
    saved: Option<Vec<u8>>,
}

impl Costs {
    fn started(&mut self, now: Instant, bytes: Vec<u8>) {
        self.handed_over(now, bytes);
    }

    /// Whether to look at the models now: after startup, once `SAVE_COSTS_EVERY` has passed (or `force`).
    fn due(&self, now: Instant, force: bool) -> bool {
        self.last.is_some_and(|at| force || now.duration_since(at) >= SAVE_COSTS_EVERY)
    }

    /// Whether `bytes` differ from what was saved.
    fn changed(&self, bytes: &[u8]) -> bool {
        self.saved.as_deref() != Some(bytes)
    }

    fn handed_over(&mut self, now: Instant, bytes: Vec<u8>) {
        (self.last, self.saved) = (Some(now), Some(bytes));
    }

    /// The writer couldn't save `bytes`: unless something newer was handed over since, the next due save writes.
    fn not_saved(&mut self, bytes: &[u8]) {
        if self.saved.as_deref() == Some(bytes) {
            self.saved = None;
        }
    }
}

impl Sched {
    /// The server's startup message: restore what an earlier run learned, and take the first clock sample.
    pub(super) fn startup(&mut self, costs: &[continuum_sched::CostModelRecord], boot: Option<BootId>) {
        continuum_sched::Scheduler::restore(&mut self.engine, costs);
        let saved = costs::encode(&self.engine.cost_models_sorted());
        self.upkeep.costs.started(Instant::now(), saved);
        self.upkeep.boot = boot;
        self.sample_clock();
    }

    /// Both upkeep jobs now, whatever their timers say.
    pub(super) fn upkeep_now(&mut self) {
        self.save_costs_if_changed(true);
        self.sample_clock();
    }

    /// The writer couldn't save these cost models.
    pub(super) fn costs_not_saved(&mut self, bytes: &[u8]) {
        self.upkeep.costs.not_saved(bytes);
    }

    pub(super) fn upkeep_tick(&mut self) {
        self.save_costs_if_changed(false);
        if !self.upkeep.pruning.load(Ordering::Acquire)
            && self.upkeep.pruned_at.is_none_or(|at| at.elapsed() >= PRUNE_FRAMES_EVERY)
        {
            self.queue_frame_sweep();
        }
        let asleep = host::continuous_ns().saturating_sub(host::monotonic_ns());
        let woke = self.upkeep.clock.is_some_and(|(_, before)| asleep.saturating_sub(before) > WAKE_JUMP_NS);
        if woke || self.upkeep.clock.is_none_or(|(at, _)| at.elapsed() >= CLOCK_EVERY) {
            self.sample_clock();
        }
    }

    fn save_costs_if_changed(&mut self, force: bool) {
        let now = Instant::now();
        if !self.upkeep.costs.due(now, force) {
            return;
        }
        let bytes = costs::encode(&self.engine.cost_models_sorted());
        if self.upkeep.costs.changed(&bytes) && self.urgent.try_send(Urgent::SaveCosts(bytes.clone())).is_err() {
            return; // the writer is full; the next tick tries again
        }
        self.upkeep.costs.handed_over(now, bytes);
    }

    fn sample_clock(&mut self) {
        let Some(boot) = self.upkeep.boot else { return };
        // A UTC reading between two monotonic ones, at their midpoint.
        let before = host::monotonic_ns();
        let utc_ns = host::utc_ns();
        let after = host::monotonic_ns();
        let point = ClockPoint { mono_ns: before + (after - before) / 2, utc_ns };
        if self.urgent.try_send(Urgent::Clock(boot, point)).is_ok() {
            let asleep = host::continuous_ns().saturating_sub(host::monotonic_ns());
            self.upkeep.clock = Some((Instant::now(), asleep));
        }
    }

    /// Queues a sweep that drops the `FRAMES` rows of frames whose chunk ended over `FRAME_ROWS_KEPT` ago and that
    /// nothing names (`Catalog::prune_frames`), a writer step at a time.
    fn queue_frame_sweep(&mut self) {
        self.upkeep.pruned_at = Some(Instant::now());
        let cutoff = super::now_ns().saturating_sub(u64::try_from(FRAME_ROWS_KEPT.as_nanos()).unwrap_or(u64::MAX));
        let job = frame_sweep(cutoff, &self.upkeep.pruning, self.writer_config);
        if self.background.try_send(super::LongJob::droppable(job)).is_err() {
            self.upkeep.pruned_at = None; // the writer is busy: the next tick tries again
        }
    }
}

trait SortedCosts {
    fn cost_models_sorted(&self) -> Vec<continuum_sched::CostModelRecord>;
}

impl SortedCosts for continuum_sched::Engine {
    /// The cost models in a stable order, so an unchanged set encodes the same and isn't saved again.
    fn cost_models_sorted(&self) -> Vec<continuum_sched::CostModelRecord> {
        let mut all = continuum_sched::Scheduler::cost_models(self);
        all.sort_by(|a, b| a.key.cmp(&b.key));
        all
    }
}

/// The sweep's job: `Catalog::prune_frames_until` a writer step at a time from where the last step stopped, logging
/// how many rows went. Like a purge, each step fills a cooperative work window (up to `max_budget` frames, stopping
/// once `step_target` has passed) instead of taking the writer's adaptive row budget: that budget counts the commit's
/// fsync, which on devbox's catalog drove it to one row a commit, so a sweep never finished (#1803). `pruning` is set
/// until the job is dropped, however it ends (done, failed, panicked, or never queued), so a second sweep never
/// starts while one runs.
fn frame_sweep(cutoff: u64, pruning: &Arc<AtomicBool>, config: super::WriterConfig) -> super::BackgroundJob {
    pruning.store(true, Ordering::Release);
    let running = Running(Arc::clone(pruning));
    let (mut after, mut pruned) = (None, 0usize);
    Box::new(move |catalog, _budget| {
        let _running = &running;
        let began = Instant::now();
        let window = &mut || began.elapsed() >= config.step_target;
        let more = match catalog.prune_frames_until(cutoff, after, config.max_budget.max(1), window) {
            Ok(step) => {
                pruned += step.pruned;
                after = step.next;
                step.next.is_some()
            }
            Err(e) => {
                warn!(error = %e, "sweeping old frame rows failed; the next sweep tries again");
                false
            }
        };
        if !more && pruned > 0 {
            info!(pruned, "dropped the rows of old absorbed frames nothing names");
        }
        more
    })
}

/// Whether a `FRAMES` sweep is running: set while the job that holds this lives.
struct Running(Arc<AtomicBool>);

impl Drop for Running {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nothing_is_saved_before_startup_restores_the_saved_models() {
        let mut c = Costs::default();
        let now = Instant::now();
        assert!(!c.due(now, false) && !c.due(now, true), "an empty set would overwrite what an earlier run learned");
        c.started(now, b"restored".to_vec());
        assert!(!c.due(now + Duration::from_secs(1), false));
        assert!(c.due(now + SAVE_COSTS_EVERY, false));
        assert!(!c.changed(b"restored"), "the restored models are what's saved");
    }

    #[test]
    fn a_failed_save_writes_again_at_the_next_due_save_even_unchanged() {
        let mut c = Costs::default();
        let now = Instant::now();
        c.started(now, b"old".to_vec());
        c.handed_over(now, b"new".to_vec());
        c.not_saved(b"new");
        assert!(c.changed(b"new"));
        // A failure report for something older than the latest hand-over changes nothing.
        c.handed_over(now, b"newer".to_vec());
        c.not_saved(b"new");
        assert!(!c.changed(b"newer"));
    }

    #[test]
    fn a_frame_sweep_steps_through_every_frame_and_clears_its_flag_when_done() {
        use continuum_catalog::{Catalog, ChunkFrame, MemoryCatalog, NewChunk, NewPayload, StreamInfo};
        use continuum_core::{HeaderBuf, HeaderSpec, StorageLocator, Uuid};

        let id = |n: u8| {
            let mut b = [0u8; 16];
            (b[6], b[8], b[15]) = (0x70, 0x80, n);
            Uuid(b)
        };
        let c = MemoryCatalog::new();
        let info = StreamInfo {
            name: "cam0".into(),
            kind: continuum_catalog::StreamKind::Sensor,
            layer: continuum_core::DomainLayer::Layer1Raw,
            format: continuum_core::PayloadFormat::RawBytes,
            shape_name: "NV12".into(),
            shape_version: 1,
            rate_hz: Some(30.0),
            anchor: None,
        };
        let s = c.register_stream(info).unwrap();
        let frame = |n: u8, t: u64| NewPayload {
            header: HeaderBuf::build(&HeaderSpec::raw(id(n), s, t, t)).unwrap(),
            location: Some(StorageLocator::Inline { bytes: vec![n] }),
            coverage: None,
        };
        c.insert((1..=3).map(|n| frame(n, u64::from(n) * 10)).collect()).unwrap();
        let chunk = NewChunk {
            payload: NewPayload {
                header: HeaderBuf::build(&HeaderSpec::raw(id(100), s, 10, 40)).unwrap(),
                location: Some(StorageLocator::Chunk {
                    tier: continuum_core::StorageTier::Tier1Warm,
                    uri: "c".into(),
                    offset: 0,
                    length: 1,
                }),
                coverage: None,
            },
            frames: (1..=3u64).map(|n| ChunkFrame::new(n * 10, n, 1)).collect(),
        };
        c.commit_chunk(chunk, &[id(1), id(2), id(3)]).unwrap();

        let pruning = Arc::new(AtomicBool::new(false));
        let mut job = frame_sweep(1_000, &pruning, super::super::WriterConfig::default());
        assert!(pruning.load(Ordering::Acquire), "set while the sweep runs");
        // The writer's adaptive budget falls to 1 when commits are slow (#1803); the sweep's work window ignores it.
        let mut steps = 1;
        while job(&c, 1) {
            steps += 1;
        }
        assert_eq!(steps, 1, "one window sweeps every frame, whatever budget the writer offers");
        assert!((1..=3).all(|n| c.get(id(n)).unwrap().is_none()), "every old frame nothing names lost its row");
        drop(job);
        assert!(!pruning.load(Ordering::Acquire), "cleared once the job is dropped");
    }
}
