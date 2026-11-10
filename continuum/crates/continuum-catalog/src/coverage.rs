//! A set of disjoint time intervals, kept merged: what one stage has processed of one source stream (`COVERAGE`).
//!
//! Both catalog implementations keep coverage in this form (redb stores the same `start -> end` pairs as rows), so
//! the merging and gap rules live here once.

use std::collections::BTreeMap;

use continuum_core::Interval;

/// Disjoint, non-touching half-open intervals, as `start_ns -> end_ns`. Two intervals that touch (`[a, b)` and
/// `[b, c)`) are always stored as one, `[a, c)`, so the set has one form for the same covered time.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CoverageSet {
    spans: BTreeMap<u64, u64>,
}

impl CoverageSet {
    /// An empty set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether nothing is covered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// Every interval, in order.
    pub fn iter(&self) -> impl Iterator<Item = Interval> + '_ {
        self.spans.iter().filter_map(|(&s, &e)| Interval::new(s, e))
    }

    /// Adds `span`, merging it with every interval it overlaps or touches.
    pub fn insert(&mut self, span: Interval) {
        self.insert_within(span, 0);
    }

    /// Adds `span`, merging it with every interval it overlaps or comes within `gap_ns` of: aged-out instants a
    /// frame apart become one run (an age-out's `merge_gap_ns`).
    pub fn insert_within(&mut self, span: Interval, gap_ns: u64) {
        let (mut start, mut end) = (span.start_ns(), span.end_ns());
        if self.spans.last_key_value().is_none_or(|(_, &e)| e.saturating_add(gap_ns) < start) {
            self.spans.insert(start, end);
            return;
        }
        // The interval starting at or before `start` may reach it.
        if let Some((&s, &e)) = self.spans.range(..=start).next_back()
            && e.saturating_add(gap_ns) >= start
        {
            start = s;
            end = end.max(e);
        }
        // Every interval starting inside [start, end + gap] is absorbed; one that reaches further can bring the next
        // within the gap, so look again until none does.
        loop {
            let absorbed: Vec<(u64, u64)> =
                self.spans.range(start..=end.saturating_add(gap_ns)).map(|(&s, &e)| (s, e)).collect();
            let before = end;
            for (s, e) in absorbed {
                self.spans.remove(&s);
                end = end.max(e);
            }
            if end == before {
                break;
            }
        }
        self.spans.insert(start, end);
    }

    /// Removes `span`, splitting an interval that straddles either end.
    pub fn remove(&mut self, span: Interval) {
        let (start, end) = (span.start_ns(), span.end_ns());
        let mut keep = Vec::new();
        let hit: Vec<(u64, u64)> = self.overlapping(span).collect();
        for (s, e) in hit {
            self.spans.remove(&s);
            if s < start {
                keep.push((s, start));
            }
            if e > end {
                keep.push((end, e));
            }
        }
        self.spans.extend(keep);
    }

    /// The covered parts of `window`, clipped to it, in order.
    #[must_use]
    pub fn within(&self, window: Interval) -> Vec<Interval> {
        self.overlapping(window)
            .filter_map(|(s, e)| Interval::new(s.max(window.start_ns()), e.min(window.end_ns())))
            .collect()
    }

    /// The parts of `window` not covered, in order, at most `max` of them (the first `max`).
    #[must_use]
    pub fn gaps(&self, window: Interval, max: usize) -> Vec<Interval> {
        let mut out = Vec::new();
        let mut at = window.start_ns();
        for covered in self.within(window) {
            if out.len() == max {
                return out;
            }
            if let Some(gap) = Interval::new(at, covered.start_ns()) {
                out.push(gap);
            }
            at = covered.end_ns();
        }
        if out.len() < max
            && let Some(gap) = Interval::new(at, window.end_ns())
        {
            out.push(gap);
        }
        out
    }

    /// Stored intervals that share time with `window`, unclipped.
    fn overlapping(&self, window: Interval) -> impl Iterator<Item = (u64, u64)> + '_ {
        let before = self.spans.range(..window.start_ns()).next_back().filter(|&(_, &e)| e > window.start_ns());
        before.into_iter().chain(self.spans.range(window.start_ns()..window.end_ns())).map(|(&s, &e)| (s, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iv(s: u64, e: u64) -> Interval {
        Interval::new(s, e).unwrap()
    }

    #[test]
    fn a_frames_slot_runs_to_the_next_frame_or_the_chunks_end() {
        use crate::{ChunkFrame, frame_slots};
        let frames = [0, 33, 66, 99].map(|t| ChunkFrame::new(t, 0, 1));
        assert_eq!(frame_slots(iv(0, 99), &frames), vec![iv(0, 33), iv(33, 66), iv(66, 99), iv(99, 100)]);
        let max = [ChunkFrame::new(u64::MAX, 0, 1)];
        assert_eq!(frame_slots(iv(u64::MAX - 1, u64::MAX), &max), vec![iv(u64::MAX - 1, u64::MAX)]);
        assert_eq!(frame_slots(iv(0, 1), &[]), vec![]);
    }

    #[test]
    fn touching_intervals_merge_into_one() {
        let mut c = CoverageSet::new();
        c.insert(iv(10, 20));
        c.insert(iv(30, 40));
        c.insert(iv(20, 30));
        assert_eq!(c.iter().collect::<Vec<_>>(), vec![iv(10, 40)]);
    }

    #[test]
    fn removing_the_middle_splits_an_interval() {
        let mut c = CoverageSet::new();
        c.insert(iv(0, 100));
        c.remove(iv(40, 60));
        assert_eq!(c.iter().collect::<Vec<_>>(), vec![iv(0, 40), iv(60, 100)]);
    }

    #[test]
    fn gaps_stop_at_max() {
        let mut c = CoverageSet::new();
        c.insert(iv(10, 20));
        c.insert(iv(30, 40));
        assert_eq!(c.gaps(iv(0, 50), 2), vec![iv(0, 10), iv(20, 30)]);
        assert_eq!(c.gaps(iv(0, 50), 0), vec![]);
        assert_eq!(c.gaps(iv(12, 18), 64), vec![]);
    }
}
