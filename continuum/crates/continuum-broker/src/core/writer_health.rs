//! In-process writer diagnostics. Persisted purges survive a dropped worker as stalled entries; their age is
//! unknown after restart until saved metadata seeds it. Logging belongs to the caller, outside the state lock.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use continuum_core::Uuid;

const WARN_AGE: Duration = Duration::from_secs(120);
const SAMPLE_LIMIT: usize = 16;

#[derive(Default)]
pub(super) struct WriterHealth {
    waiting: AtomicUsize,
    running: AtomicBool,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    purges: HashMap<Uuid, Purge>,
    // Retained after finish so re-tracking an id cannot produce another warning in this process.
    warned: HashSet<Uuid>,
}

struct Purge {
    started: Instant,
    age: Option<(Duration, Instant)>,
    resumed: bool,
    persisted: bool,
    stalled: bool,
    steps: u64,
    work: Duration,
    /// Stream names this purge covers. A range that names none is `"all"`. Empty for an entity or id purge.
    streams: Vec<String>,
}

impl Purge {
    fn age(&self, now: Instant) -> Option<Duration> {
        self.age.map(|(base, at)| base.saturating_add(now.saturating_duration_since(at)))
    }
}

pub(super) struct Snapshot {
    /// Writer-local waiting jobs plus jobs still in its channel; excludes the running step.
    pub queued: usize,
    pub running: bool,
    pub purges: usize,
    pub oldest_age: Option<Duration>,
    pub unknown_age: usize,
    pub stalled: usize,
    pub samples: Vec<PurgeSample>,
    pub omitted: usize,
    /// Up to three streams, oldest unfinished purge first. Ages are known ages only.
    pub oldest_streams: Vec<(String, Duration)>,
    /// Newly due warnings. The caller logs these after snapshot releases the lock.
    pub warnings: Vec<PurgeSample>,
}

#[derive(Clone, Debug)]
pub(super) struct PurgeSample {
    pub id: Uuid,
    pub steps: u64,
    pub age: Option<Duration>,
    pub stalled: bool,
    pub elapsed: Duration,
    pub work: Duration,
}

pub(super) struct PurgeTracker {
    health: Arc<WriterHealth>,
    id: Uuid,
}

impl WriterHealth {
    pub fn set_waiting(&self, waiting: usize) {
        self.waiting.store(waiting, Ordering::Relaxed);
    }

    pub fn set_running(&self, running: bool) {
        self.running.store(running, Ordering::Relaxed);
    }

    pub fn track(self: &Arc<Self>, id: Uuid, resumed: bool, now: Instant) -> PurgeTracker {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        state.purges.entry(id).or_insert(Purge {
            started: now,
            age: (!resumed).then_some((Duration::ZERO, now)),
            resumed,
            persisted: resumed,
            stalled: false,
            steps: 0,
            work: Duration::ZERO,
            streams: Vec::new(),
        });
        PurgeTracker { health: Arc::clone(self), id }
    }

    /// Records the stream names a purge covers, for the health line. Later notes replace earlier ones.
    pub fn note_scope(&self, id: Uuid, streams: Vec<String>) {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(purge) = state.purges.get_mut(&id) {
            purge.streams = streams;
        }
    }

    pub fn mark_persisted(&self, id: Uuid, at_ns: u64, now_ns: u64, now: Instant) {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(purge) = state.purges.get_mut(&id) {
            purge.persisted = true;
            // Seed once, then use the monotonic clock even if wall time changes or metadata is reread.
            if purge.resumed && purge.age.is_none() {
                purge.age = Some((Duration::from_nanos(now_ns.saturating_sub(at_ns)), now));
            }
        }
    }

    pub fn record_step(&self, id: Uuid, took: Duration) {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(purge) = state.purges.get_mut(&id) {
            purge.steps = purge.steps.saturating_add(1);
            purge.work = purge.work.saturating_add(took);
        }
    }

    pub fn finish(&self, id: Uuid) {
        self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner).purges.remove(&id);
    }

    pub fn snapshot(&self, now: Instant, channel_len: usize) -> Snapshot {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut samples: Vec<_> = state
            .purges
            .iter()
            .map(|(&id, purge)| PurgeSample {
                id,
                steps: purge.steps,
                age: purge.age(now),
                stalled: purge.stalled,
                elapsed: now.saturating_duration_since(purge.started),
                work: purge.work,
            })
            .collect();
        // Known oldest first; unknown ages remain explicitly unknown and sort deterministically by id.
        samples.sort_by(|a, b| b.age.cmp(&a.age).then_with(|| a.id.cmp(&b.id)));
        let oldest_age = samples.first().and_then(|sample| sample.age);
        let unknown_age = samples.iter().filter(|sample| sample.age.is_none()).count();
        let stalled = samples.iter().filter(|sample| sample.stalled).count();
        let warnings = samples
            .iter()
            .filter(|sample| sample.age.is_some_and(|age| age >= WARN_AGE))
            .filter(|sample| state.warned.insert(sample.id))
            .take(SAMPLE_LIMIT)
            .cloned()
            .collect();
        let purges = samples.len();
        samples.truncate(SAMPLE_LIMIT);
        let mut by_stream: std::collections::BTreeMap<String, Duration> = std::collections::BTreeMap::new();
        for purge in state.purges.values() {
            let Some(age) = purge.age(now) else { continue };
            for name in &purge.streams {
                by_stream
                    .entry(name.clone())
                    .and_modify(|had| {
                        if age > *had {
                            *had = age;
                        }
                    })
                    .or_insert(age);
            }
        }
        let mut oldest_streams: Vec<_> = by_stream.into_iter().collect();
        oldest_streams.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        oldest_streams.truncate(3);
        Snapshot {
            queued: self.waiting.load(Ordering::Relaxed).saturating_add(channel_len),
            running: self.running.load(Ordering::Relaxed),
            purges,
            oldest_age,
            unknown_age,
            stalled,
            omitted: purges.saturating_sub(samples.len()),
            samples,
            warnings,
            oldest_streams,
        }
    }
}

impl Drop for PurgeTracker {
    fn drop(&mut self) {
        let mut state = self.health.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(purge) = state.purges.get_mut(&self.id) {
            if purge.persisted {
                purge.stalled = true;
            } else {
                state.purges.remove(&self.id);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(n: u8) -> Uuid {
        Uuid([n; 16])
    }

    #[test]
    fn warnings_start_at_120_seconds_and_are_emitted_once() {
        let health = Arc::new(WriterHealth::default());
        let now = Instant::now();
        let _track = health.track(id(1), false, now);
        assert!(health.snapshot(now + Duration::from_secs(119), 0).warnings.is_empty());
        assert_eq!(
            health
                .snapshot(now + Duration::from_secs(120), 0)
                .warnings
                .iter()
                .map(|sample| sample.id)
                .collect::<Vec<_>>(),
            [id(1)]
        );
        assert!(health.snapshot(now + Duration::from_secs(121), 0).warnings.is_empty());
        health.finish(id(1));
        let _again = health.track(id(1), false, now);
        assert!(health.snapshot(now + Duration::from_secs(121), 0).warnings.is_empty());
    }

    #[test]
    fn resumed_age_is_unknown_until_metadata_then_monotonic_even_for_future_wall_times() {
        let health = Arc::new(WriterHealth::default());
        let now = Instant::now();
        let _old = health.track(id(1), true, now);
        let unknown = health.snapshot(now + Duration::from_secs(5), 0);
        assert_eq!(unknown.unknown_age, 1);
        assert_eq!(unknown.oldest_age, None);
        assert_eq!(unknown.samples[0].age, None);
        health.mark_persisted(id(1), 1, 121_000_000_001, now);
        assert_eq!(health.snapshot(now + Duration::from_secs(2), 0).oldest_age, Some(Duration::from_secs(123)));
        health.mark_persisted(id(1), u64::MAX, 0, now + Duration::from_secs(3));
        assert_eq!(health.snapshot(now + Duration::from_secs(3), 0).oldest_age, Some(Duration::from_secs(124)));
        let _future = health.track(id(2), true, now);
        health.mark_persisted(id(2), 10, 5, now);
        let snapshot = health.snapshot(now + Duration::from_secs(4), 0);
        assert_eq!(
            snapshot.samples.iter().find(|sample| sample.id == id(2)).unwrap().age,
            Some(Duration::from_secs(4))
        );
    }

    #[test]
    fn rejected_jobs_disappear_persisted_drops_stall_and_finished_jobs_disappear() {
        let health = Arc::new(WriterHealth::default());
        let now = Instant::now();
        drop(health.track(id(1), false, now));
        assert_eq!(health.snapshot(now, 0).purges, 0);
        let tracker = health.track(id(2), false, now);
        health.mark_persisted(id(2), 0, u64::MAX, now);
        health.record_step(id(2), Duration::from_millis(3));
        drop(tracker);
        let snapshot = health.snapshot(now + Duration::from_secs(2), 0);
        assert_eq!(snapshot.stalled, 1);
        assert_eq!(snapshot.samples[0].age, Some(Duration::from_secs(2)));
        assert_eq!(snapshot.samples[0].steps, 1);
        assert_eq!(snapshot.samples[0].work, Duration::from_millis(3));
        assert_eq!(snapshot.samples[0].elapsed, Duration::from_secs(2));
        health.finish(id(2));
        let tracker = health.track(id(3), true, now);
        health.finish(id(3));
        drop(tracker);
        assert_eq!(health.snapshot(now, 0).purges, 0);
    }

    #[test]
    fn depth_excludes_running_and_samples_are_bounded_oldest_first() {
        let health = Arc::new(WriterHealth::default());
        let now = Instant::now();
        let trackers: Vec<_> =
            (0..20).map(|n| health.track(id(n), false, now + Duration::from_secs(u64::from(n)))).collect();
        health.set_waiting(3);
        health.set_running(true);
        let snapshot = health.snapshot(now + Duration::from_secs(30), 4);
        assert_eq!(snapshot.queued, 7);
        assert!(snapshot.running);
        assert_eq!((snapshot.purges, snapshot.samples.len(), snapshot.omitted), (20, 16, 4));
        assert_eq!(snapshot.samples[0].id, id(0));
        assert_eq!(snapshot.samples[15].id, id(15));
        health.set_waiting(0);
        health.set_running(false);
        assert_eq!(health.snapshot(now, 0).queued, 0);
        assert!(!health.snapshot(now, 0).running);
        drop(trackers);
    }
    #[test]
    fn poisoned_diagnostics_lock_recovers_and_warnings_include_unsampled_progress() {
        let health = Arc::new(WriterHealth::default());
        let now = Instant::now();
        let trackers: Vec<_> = (0..20).map(|n| health.track(id(n), false, now)).collect();
        health.record_step(id(19), Duration::from_millis(5));
        let _ = std::panic::catch_unwind(|| {
            let _guard = health.state.lock().unwrap();
            panic!("poison diagnostics only");
        });
        let snapshot = health.snapshot(now + Duration::from_secs(120), 0);
        assert_eq!(snapshot.samples.len(), 16);
        assert_eq!(snapshot.warnings.len(), 16, "one health pass bounds synchronous warning output");
        let next = health.snapshot(now + Duration::from_secs(120), 0);
        assert_eq!(next.warnings.len(), 4, "deferred warnings are not lost or marked as emitted");
        let warning = next.warnings.iter().find(|sample| sample.id == id(19)).unwrap();
        assert_eq!(warning.age, Some(Duration::from_secs(120)));
        assert_eq!(warning.steps, 1);
        assert_eq!(warning.work, Duration::from_millis(5));
        health.mark_persisted(id(19), 0, 0, now);
        health.record_step(id(19), Duration::from_millis(1));
        drop(trackers);
        assert_eq!(health.snapshot(now, 0).stalled, 1);
        health.finish(id(19));
        assert_eq!(health.snapshot(now, 0).purges, 0);
    }

    #[test]
    fn the_health_summary_keeps_the_three_oldest_streams() {
        let health = Arc::new(WriterHealth::default());
        let now = Instant::now();
        let seed = |n: u8, age_s: u64, name: &str| {
            let _tracker = health.track(id(n), true, now);
            health.mark_persisted(id(n), 0, age_s.saturating_mul(1_000_000_000), now);
            health.note_scope(id(n), vec![name.to_owned()]);
        };
        seed(1, 10, "newer");
        seed(2, 40, "oldest");
        seed(3, 30, "second");
        seed(4, 20, "third");
        seed(5, 50, "cam");
        // A younger purge of the same stream must not hide the older one.
        seed(6, 5, "oldest");
        let unknown = health.track(id(9), true, now);
        health.note_scope(id(9), vec!["mystery".to_owned()]);
        let names: Vec<_> =
            health.snapshot(now, 0).oldest_streams.iter().map(|(name, age)| (name.clone(), age.as_secs())).collect();
        assert_eq!(names, [("cam".to_owned(), 50), ("oldest".to_owned(), 40), ("second".to_owned(), 30)]);
        drop(unknown);
    }
}
