//! Tier 0 slots the index counts: which payload is in which slot, so the index can follow a slot that moves on (a
//! reclaim, a capture overwrite), the broker can reclaim the oldest slots when a size class runs low, and a purge can
//! zero its payloads' slots.

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use continuum_core::{Interval, SlotHandle, StorageLocator, StorageTier, StreamId, Uuid};
use continuum_shm::{Purged, Reclaimed, SlotScan, SlotState};
use tracing::{debug, warn};

use super::Sched;

/// How often the slot classes are scanned.
const SCAN_EVERY: Duration = Duration::from_secs(1);

/// A class with no free slot (and none on its way to free) gets `1 / RECLAIM` of its slots reclaimed: only when a
/// plain claim would fail, so a capture class that keeps itself full with overwrites isn't churned.
const RECLAIM: usize = 8;

/// A payload's Tier 0 copy, as the index counts it.
#[derive(Clone, Copy, Debug)]
struct SlotCopy {
    handle: SlotHandle,
    stream: StreamId,
    span: Interval,
}

/// The slots the index counts, by payload and by slot.
#[derive(Debug, Default)]
pub(super) struct Slots {
    by_payload: HashMap<Uuid, SlotCopy>,
    by_slot: HashMap<(u16, u32), Uuid>,
    scanned: Option<Instant>,
    /// Purged slots still to zero: a live lease held them, or the reader pool was full.
    purging: Vec<SlotHandle>,
    /// Streams owed to the archive: a worker that keeps copies (the tier worker) opened a durable subscription on
    /// each during this broker's run. Kept until the broker stops, so a tier worker restarting doesn't drop them.
    archived: HashSet<StreamId>,
}

/// Whether the scheduler may reclaim a slot under pressure. A flushed slot has its Tier 1 copy. An unflushed one is
/// the archive's only copy when its stream is owed to the archive and the arena's `flushed` mark means a Tier 1
/// copy, so it stays; otherwise nothing is owed, and only a reader's pin keeps it.
fn reclaimable(s: &SlotScan, owed: bool, requires_flush: bool) -> bool {
    s.state == SlotState::Ready && (s.flushed || (!(owed && requires_flush) && s.pins == 0))
}

impl Slots {
    fn forget(&mut self, id: Uuid) -> Option<SlotCopy> {
        let copy = self.by_payload.remove(&id)?;
        self.by_slot.remove(&(copy.handle.class, copy.handle.index));
        Some(copy)
    }

    /// Whether `stream` is owed to the archive (see [`Slots::archived`]).
    pub(super) fn is_archived(&self, stream: StreamId) -> bool {
        self.archived.contains(&stream)
    }

    fn remove_copy(
        &mut self,
        id: Uuid,
        index: &mut crate::index::AvailabilityIndex,
        loss_reason: Option<&'static str>,
        health: &mut super::ArchiveHealth,
    ) {
        let Some(copy) = self.forget(id) else { return };
        // A purge can have removed the copy already. It is not archive loss.
        if index.remove_copy(copy.stream, copy.span, StorageTier::Tier0Hot).is_err() {
            return;
        }
        if let Some(reason) = loss_reason {
            let kept: u64 = index.covered(copy.stream, copy.span).iter().map(|part| part.interval.len_ns()).sum();
            if kept < copy.span.len_ns() {
                health.slot_gap(copy.stream, copy.span, reason);
            }
        }
    }
}

impl Sched {
    /// A worker that keeps copies follows `streams` durably: their Tier 0 slots are owed to the archive from now on.
    pub(super) fn owe_archive(&mut self, streams: impl IntoIterator<Item = StreamId>) {
        self.slots.archived.extend(streams);
    }

    fn owed(&self, stream: StreamId) -> bool {
        self.slots.is_archived(stream)
    }

    /// Notes a payload the index just counted a Tier 0 slot copy of.
    pub(super) fn track_slot(&mut self, id: Uuid, stream: StreamId, span: Interval, locator: Option<&StorageLocator>) {
        let Some(StorageLocator::Shm { slab_class, slot, slot_generation, .. }) = locator else { return };
        let handle = SlotHandle { class: *slab_class, index: *slot, generation: *slot_generation };
        // A slot reused under a payload not yet noticed as gone: the old one's copy left with the slot.
        if let Some(old) = self.slots.by_slot.get(&(handle.class, handle.index)).copied() {
            self.drop_slot_copy(old, Some("slot reused by a producer"));
        }
        self.slots.by_payload.insert(id, SlotCopy { handle, stream, span });
        self.slots.by_slot.insert((handle.class, handle.index), id);
    }

    /// The payloads a purge step doomed: their slots stop being leasable at once, and are zeroed and freed off the
    /// scheduler thread (zeroing a 16 MB slot isn't free) once no live lease holds them. The index already dropped
    /// their copies with the step's delta.
    pub(super) fn purge_slots(&mut self, doomed: &[continuum_catalog::Doomed]) {
        let handles: Vec<SlotHandle> =
            doomed.iter().filter_map(|d| self.slots.forget(d.id)).map(|c| c.handle).collect();
        self.slots.purging.extend(handles);
        self.send_purges();
    }

    /// Slots a purge job couldn't zero yet: back on the list, for the next scan.
    pub(super) fn slots_to_purge(&mut self, handles: Vec<SlotHandle>) {
        self.slots.purging.extend(handles);
    }

    /// Hands the slots waiting to be purged to the reader pool; any it can't zero yet come back as an event.
    fn send_purges(&mut self) {
        if self.slots.purging.is_empty() {
            return;
        }
        let handles = std::mem::take(&mut self.slots.purging);
        let kept = handles.clone();
        let tier0 = std::sync::Arc::clone(&self.tier0);
        let events = self.events.clone();
        let job: super::ReadJob = Box::new(move |_| {
            let mut pending = Vec::new();
            for h in handles {
                match tier0.purge_slot(h) {
                    Ok(Purged::Zeroed) => {}
                    Ok(Purged::Pending) => pending.push(h),
                    Err(e) => debug!(error = %e, slot = h.index, "a purged payload's slot had already moved on"),
                }
            }
            if !pending.is_empty() {
                let _ = events.send(super::Event::SlotsToPurge(pending));
            }
        });
        if self.reads.try_send(job).is_err() {
            // The readers are full: keep them for the next scan.
            self.slots.purging.extend(kept);
            debug!("the readers are full; purged slots wait for the next scan");
        }
    }

    /// Once a second: the index lets go of slots that moved on, and a class running low on free slots gets its oldest
    /// published slots reclaimed, flushed ones first. An unflushed slot owed to the archive stays for the tier worker.
    pub(super) fn slots_tick(&mut self) {
        if self.slots.scanned.is_some_and(|at| at.elapsed() < SCAN_EVERY) {
            return;
        }
        self.slots.scanned = Some(Instant::now());
        self.send_purges();
        for class in 0..self.tier0.class_count() {
            let Ok(scan) = self.tier0.scan(class) else { continue };
            // Slots that moved on since the index counted them.
            for s in &scan {
                if let Some(&id) = self.slots.by_slot.get(&(class, s.handle.index))
                    && let Some(copy) = self.slots.by_payload.get(&id)
                    && (copy.handle.generation != s.handle.generation || s.state == SlotState::Free)
                {
                    self.drop_slot_copy(id, Some("slot generation or state moved on"));
                }
            }
            // Only when a plain claim would fail: nothing free, and nothing already on its way to free.
            if scan.iter().any(|s| s.state == SlotState::Free || s.state == SlotState::Reclaiming) {
                self.archive_health.pressure(class, 0);
                continue;
            }
            let mut want = (scan.len() / RECLAIM).max(1);
            let requires_flush = self.tier0.reclaim_requires_flush();
            let mut held = 0;
            // Candidates: slots the index counts (one published but not yet ingested isn't anyone's to take). An
            // unflushed slot owed to the archive is its only copy until Tier 1 commits it, pinned or not. Capture's
            // explicit overwrite policy is separate, and losses from it are counted above.
            let mut ready = Vec::new();
            for s in &scan {
                let Some(id) = self.slots.by_slot.get(&(class, s.handle.index)) else { continue };
                let owed = self.slots.by_payload.get(id).is_some_and(|c| self.owed(c.stream));
                if reclaimable(s, owed, requires_flush) {
                    ready.push(s);
                } else if s.state == SlotState::Ready && !s.flushed && owed && requires_flush {
                    held += 1;
                }
            }
            // Flushed first (a Tier 1 copy exists), oldest first within each.
            ready.sort_by_key(|s| (!s.flushed, s.seq));
            for s in ready {
                if want == 0 {
                    break;
                }
                match self.tier0.reclaim(s.handle) {
                    Ok(Reclaimed::Freed) => want -= 1,
                    Ok(Reclaimed::Evicted { pins }) => {
                        warn!(
                            slot = s.handle.index,
                            class, pins, "SLOW_CONSUMER_EVICTION: a reader's pin expired on a slot that was needed"
                        );
                        want -= 1;
                    }
                    // A flushed slot a reader still holds: it frees when the pin goes, and the scan drops its copy then.
                    Ok(Reclaimed::Pinned) => {
                        want -= 1;
                        continue;
                    }
                    Err(_) => continue, // already moved on
                }
                if let Some(&id) = self.slots.by_slot.get(&(class, s.handle.index)) {
                    self.drop_slot_copy(id, None);
                }
            }
            self.archive_health.pressure(class, if want > 0 { held } else { 0 });
        }
    }

    /// The index lets go of a payload's slot copy. Losing it is an archive gap only for a stream owed to the archive.
    fn drop_slot_copy(&mut self, id: Uuid, loss_reason: Option<&'static str>) {
        let owed = self.slots.by_payload.get(&id).is_some_and(|c| self.owed(c.stream));
        let loss_reason = loss_reason.filter(|_| owed);
        self.slots.remove_copy(id, &mut self.index, loss_reason, &mut self.archive_health);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{AvailabilityIndex, StreamMeta};
    use continuum_sched::StreamKind;

    fn indexed_slot() -> (Slots, AvailabilityIndex, Uuid, SlotCopy) {
        let id = Uuid([7; 16]);
        let copy = SlotCopy {
            handle: SlotHandle { class: 0, index: 0, generation: 1 },
            stream: StreamId(1),
            span: Interval::new(100, 200).unwrap(),
        };
        let mut slots = Slots::default();
        slots.by_payload.insert(id, copy);
        slots.by_slot.insert((0, 0), id);
        let mut index = AvailabilityIndex::new();
        index.register(copy.stream, StreamMeta { kind: StreamKind::Sensor, rate_hz: 0.0 });
        index.add_copy(copy.stream, copy.span, StorageTier::Tier0Hot).unwrap();
        (slots, index, id, copy)
    }

    #[test]
    fn a_moved_slot_counts_one_gap_and_not_an_archived_or_purged_copy() {
        let mut health = super::super::ArchiveHealth::default();
        let (mut slots, mut index, id, copy) = indexed_slot();
        slots.remove_copy(id, &mut index, Some("producer overwrite"), &mut health);
        assert_eq!(health.slot_gaps, 1);
        assert!(index.covered(copy.stream, copy.span).is_empty());
        // The next scan and a later ingest can both notice the same physical reuse.
        slots.remove_copy(id, &mut index, Some("later scan"), &mut health);
        assert_eq!(health.slot_gaps, 1);

        let (mut slots, mut index, id, copy) = indexed_slot();
        index.add_copy(copy.stream, copy.span, StorageTier::Tier1Warm).unwrap();
        slots.remove_copy(id, &mut index, Some("producer overwrite"), &mut health);
        assert_eq!(health.slot_gaps, 1, "a durable archive suppresses the loss diagnostic");
        assert_eq!(index.covered(copy.stream, copy.span)[0].tier, StorageTier::Tier1Warm);

        let (mut slots, mut index, id, copy) = indexed_slot();
        // purge_slots forgets its tracked slot after the index applies the purge delta.
        index.remove_copy(copy.stream, copy.span, StorageTier::Tier0Hot).unwrap();
        slots.forget(id);
        slots.remove_copy(id, &mut index, Some("slot zeroed by purge"), &mut health);
        assert_eq!(health.slot_gaps, 1, "intentional forgetting is not archive loss");
    }

    #[test]
    fn only_an_unflushed_slot_owed_to_a_flush_tracking_archive_is_held() {
        let slot = |state, pins, flushed| SlotScan {
            handle: SlotHandle { class: 0, index: 0, generation: 1 },
            state,
            pins,
            seq: 1,
            flushed,
        };
        // (state, pins, flushed, owed, requires_flush) -> reclaimable
        let cases = [
            (SlotState::Ready, 0, true, true, true, true),
            (SlotState::Ready, 2, true, true, true, true),
            (SlotState::Ready, 0, false, true, true, false),
            (SlotState::Ready, 2, false, true, true, false),
            // No archive obligation: the old rule, an unpinned slot may go.
            (SlotState::Ready, 0, false, false, true, true),
            (SlotState::Ready, 2, false, false, true, false),
            // An arena without reclaim_requires_flush: `flushed` promises nothing, so it guards nothing either.
            (SlotState::Ready, 0, false, true, false, true),
            (SlotState::Ready, 2, false, true, false, false),
            (SlotState::Writing, 0, true, false, false, false),
            (SlotState::Free, 0, true, false, false, false),
            (SlotState::Reclaiming, 0, true, false, false, false),
        ];
        for (state, pins, flushed, owed, requires_flush, expected) in cases {
            let s = slot(state, pins, flushed);
            assert_eq!(
                reclaimable(&s, owed, requires_flush),
                expected,
                "{state:?} pins={pins} flushed={flushed} owed={owed} requires_flush={requires_flush}"
            );
        }
    }

    #[test]
    fn a_partial_archive_does_not_hide_a_slot_gap() {
        let (mut slots, mut index, id, copy) = indexed_slot();
        let partial = Interval::new(100, 150).unwrap();
        index.add_copy(copy.stream, partial, StorageTier::Tier1Warm).unwrap();
        let mut health = super::super::ArchiveHealth::default();
        slots.remove_copy(id, &mut index, Some("producer overwrite"), &mut health);
        assert_eq!(health.slot_gaps, 1);
        assert_eq!(index.covered(copy.stream, copy.span)[0].interval, partial);
    }
}
