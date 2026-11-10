//! An in-memory [`CatalogView`] for tests, in this crate and in the broker's.
//!
//! Set it up with what exists and what has been processed, then hand it to a scheduler. It does the interval
//! arithmetic the real catalog does (merging covered parts, resolving tiers, computing gaps), so a test states
//! facts, not answers.

use std::collections::BTreeMap;

use continuum_core::{Interval, StageId, StorageTier, StreamId};

use crate::catalog::{CatalogView, Covered, StreamKind, StreamView, ViewError};
use crate::intervals::{clip_all, merge, subtract};

/// Tiers from hottest to coldest, the order [`CatalogView::has_range`] resolves them in.
const TIERS: [StorageTier; 3] = [StorageTier::Tier0Hot, StorageTier::Tier1Warm, StorageTier::Tier2Cold];

/// A catalog that answers from what the test put in it.
#[derive(Clone, Debug, Default)]
pub struct FakeCatalog {
    streams: BTreeMap<StreamId, StreamView>,
    /// Per stream and tier, the merged set of intervals whose bytes are at that tier.
    covered: BTreeMap<(StreamId, StorageTier), Vec<Interval>>,
    processed: BTreeMap<(StreamId, StageId), Vec<Interval>>,
    /// When set, every question fails with this error, for testing how a scheduler handles a broken catalog.
    failing: Option<ViewError>,
}

impl FakeCatalog {
    /// An empty catalog.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a stream with no live edge. Registering an id again replaces it.
    pub fn add_stream(&mut self, id: StreamId, kind: StreamKind, live: bool, rate_hz: f64) -> &mut Self {
        self.streams.insert(id, StreamView { id, kind, live, rate_hz, live_edge_ns: None });
        self
    }

    /// Set whether a stream is live.
    pub fn set_live(&mut self, id: StreamId, live: bool) -> &mut Self {
        if let Some(s) = self.streams.get_mut(&id) {
            s.live = live;
        }
        self
    }

    /// Set a stream's live edge: the newest time its producer has written, sealed or not.
    pub fn set_live_edge(&mut self, id: StreamId, live_edge_ns: Option<u64>) -> &mut Self {
        if let Some(s) = self.streams.get_mut(&id) {
            s.live_edge_ns = live_edge_ns;
        }
        self
    }

    /// Record that `range` of `stream` has bytes at `tier`. A range may be recorded at several tiers (hot and
    /// flushed to warm, say); [`CatalogView::has_range`] reports each moment at the hottest tier holding it.
    pub fn cover(&mut self, stream: StreamId, range: Interval, tier: StorageTier) -> &mut Self {
        let set = self.covered.entry((stream, tier)).or_default();
        set.push(range);
        *set = merge(std::mem::take(set));
        self
    }

    /// Make every question fail with `error` (or stop failing with `None`).
    pub fn set_failing(&mut self, error: Option<ViewError>) -> &mut Self {
        self.failing = error;
        self
    }

    /// Record that `stage` has processed `range` of `source`.
    pub fn mark_processed(&mut self, source: StreamId, stage: StageId, range: Interval) -> &mut Self {
        let done = self.processed.entry((source, stage)).or_default();
        done.push(range);
        *done = merge(std::mem::take(done));
        self
    }
}

impl CatalogView for FakeCatalog {
    fn stream(&self, stream: StreamId) -> Result<Option<StreamView>, ViewError> {
        if let Some(e) = &self.failing {
            return Err(e.clone());
        }
        Ok(self.streams.get(&stream).copied())
    }

    fn has_range(&self, stream: StreamId, range: Interval) -> Result<Vec<Covered>, ViewError> {
        if let Some(e) = &self.failing {
            return Err(e.clone());
        }
        // Hottest tier first; each colder tier contributes only what no hotter tier already covers.
        let mut taken: Vec<Interval> = Vec::new();
        let mut out = Vec::new();
        for tier in TIERS {
            let Some(set) = self.covered.get(&(stream, tier)) else { continue };
            for part in clip_all(set, range) {
                for piece in subtract(part, &taken) {
                    out.push(Covered { interval: piece, tier });
                }
            }
            taken = merge(taken.into_iter().chain(clip_all(set, range)).collect());
        }
        out.sort_by_key(|c| c.interval.start_ns());
        Ok(out)
    }

    fn coverage_gaps(
        &self,
        source: StreamId,
        stage: StageId,
        window: Interval,
        max: usize,
    ) -> Result<Vec<Interval>, ViewError> {
        if let Some(e) = &self.failing {
            return Err(e.clone());
        }
        let done = self.processed.get(&(source, stage)).map_or(&[][..], Vec::as_slice);
        let mut out = subtract(window, done);
        out.truncate(max);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    fn iv(start: u64, end: u64) -> Interval {
        Interval::new(start, end).expect("test interval")
    }

    #[test]
    fn unknown_streams_have_nothing() {
        let cat = FakeCatalog::new();
        assert_eq!(cat.stream(StreamId(1)), Ok(None));
        assert_eq!(cat.has_range(StreamId(1), iv(0, 10)), Ok(Vec::new()));
        assert_eq!(cat.coverage_gaps(StreamId(1), StageId(1), iv(0, 10), 8), Ok(vec![iv(0, 10)]));
    }

    #[test]
    fn covered_parts_are_clipped_to_the_query_and_merged_per_tier() {
        let mut cat = FakeCatalog::new();
        cat.cover(StreamId(1), iv(0, 10), StorageTier::Tier1Warm)
            .cover(StreamId(1), iv(10, 20), StorageTier::Tier1Warm)
            .cover(StreamId(1), iv(30, 40), StorageTier::Tier0Hot);
        assert_eq!(
            cat.has_range(StreamId(1), iv(5, 35)),
            Ok(vec![
                Covered { interval: iv(5, 20), tier: StorageTier::Tier1Warm },
                Covered { interval: iv(30, 35), tier: StorageTier::Tier0Hot },
            ])
        );
        assert_eq!(cat.has_range(StreamId(1), iv(20, 30)), Ok(Vec::new()));
    }

    #[test]
    fn the_hottest_tier_wins_where_tiers_overlap() {
        // The common case: a range in Tier 0 that the tier worker has also flushed to Tier 1.
        let mut cat = FakeCatalog::new();
        cat.cover(StreamId(1), iv(0, 10), StorageTier::Tier1Warm).cover(StreamId(1), iv(0, 10), StorageTier::Tier0Hot);
        assert_eq!(
            cat.has_range(StreamId(1), iv(0, 10)),
            Ok(vec![Covered { interval: iv(0, 10), tier: StorageTier::Tier0Hot }])
        );

        // Interleaved: T1 [0,10), T0 [5,8), T1 [8,20) resolves to three disjoint parts.
        let mut cat = FakeCatalog::new();
        cat.cover(StreamId(2), iv(0, 10), StorageTier::Tier1Warm)
            .cover(StreamId(2), iv(5, 8), StorageTier::Tier0Hot)
            .cover(StreamId(2), iv(8, 20), StorageTier::Tier1Warm);
        assert_eq!(
            cat.has_range(StreamId(2), iv(0, 20)),
            Ok(vec![
                Covered { interval: iv(0, 5), tier: StorageTier::Tier1Warm },
                Covered { interval: iv(5, 8), tier: StorageTier::Tier0Hot },
                Covered { interval: iv(8, 20), tier: StorageTier::Tier1Warm },
            ])
        );
    }

    #[test]
    fn gaps_are_the_window_minus_what_was_processed() {
        let mut cat = FakeCatalog::new();
        let (src, stage) = (StreamId(2), StageId(7));
        cat.mark_processed(src, stage, iv(40, 50)).mark_processed(src, stage, iv(20, 25)).mark_processed(
            src,
            stage,
            iv(10, 20),
        );
        assert_eq!(cat.coverage_gaps(src, stage, iv(0, 60), 8), Ok(vec![iv(0, 10), iv(25, 40), iv(50, 60)]));
        assert_eq!(cat.coverage_gaps(src, stage, iv(0, 60), 2), Ok(vec![iv(0, 10), iv(25, 40)]));
        assert_eq!(cat.coverage_gaps(src, stage, iv(0, 60), 0), Ok(Vec::new()));
        assert_eq!(cat.coverage_gaps(src, stage, iv(12, 18), 8), Ok(Vec::new()));
        assert_eq!(cat.coverage_gaps(src, stage, iv(45, u64::MAX), 8), Ok(vec![iv(50, u64::MAX)]));
    }

    #[test]
    fn streams_can_go_live_and_back_and_have_a_live_edge() {
        let mut cat = FakeCatalog::new();
        cat.add_stream(StreamId(3), StreamKind::Sensor, true, 30.0);
        let view = |cat: &FakeCatalog| cat.stream(StreamId(3)).expect("not failing").map(|s| (s.live, s.live_edge_ns));
        assert_eq!(view(&cat), Some((true, None)));
        cat.set_live(StreamId(3), false).set_live_edge(StreamId(3), Some(1_000));
        assert_eq!(view(&cat), Some((false, Some(1_000))));
        cat.set_failing(Some(ViewError("disk".to_owned())));
        assert_eq!(cat.stream(StreamId(3)), Err(ViewError("disk".to_owned())));
        assert!(cat.has_range(StreamId(3), iv(0, 1)).is_err());
        cat.set_failing(None);
        assert!(cat.stream(StreamId(3)).is_ok());
    }

    fn tier() -> impl Strategy<Value = StorageTier> {
        prop_oneof![Just(StorageTier::Tier0Hot), Just(StorageTier::Tier1Warm), Just(StorageTier::Tier2Cold)]
    }

    fn records() -> impl Strategy<Value = Vec<(Interval, StorageTier)>> {
        prop::collection::vec(((0u64..100, 1u64..30).prop_map(|(s, l)| iv(s, s + l)), tier()), 0..10)
    }

    proptest! {
        #[test]
        fn has_range_is_disjoint_sorted_inside_the_query_and_hottest_wins(
            recs in records(), (qs, ql) in (0u64..100, 1u64..60)
        ) {
            let query = iv(qs, qs + ql);
            let mut cat = FakeCatalog::new();
            for (i, t) in &recs {
                cat.cover(StreamId(1), *i, *t);
            }
            let parts = cat.has_range(StreamId(1), query).expect("not failing");
            prop_assert!(parts.windows(2).all(|w| w[0].interval.end_ns() <= w[1].interval.start_ns()));
            for p in &parts {
                prop_assert!(query.start_ns() <= p.interval.start_ns() && p.interval.end_ns() <= query.end_ns());
            }
            for t in query.start_ns()..query.end_ns() {
                let hottest = recs.iter().filter(|(i, _)| i.contains(t)).map(|(_, tier)| tier.0).min();
                let reported = parts.iter().find(|p| p.interval.contains(t)).map(|p| p.tier.0);
                prop_assert_eq!(reported, hottest, "point {}", t);
            }
        }

        #[test]
        fn has_range_ignores_the_order_of_cover_calls(recs in records(), seed in any::<u64>()) {
            let mut a = FakeCatalog::new();
            for (i, t) in &recs {
                a.cover(StreamId(1), *i, *t);
            }
            let mut shuffled = recs.clone();
            let mut s = seed;
            for i in (1..shuffled.len()).rev() {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                let j = usize::try_from(s % (i as u64 + 1)).expect("fits");
                shuffled.swap(i, j);
            }
            let mut b = FakeCatalog::new();
            for (i, t) in &shuffled {
                b.cover(StreamId(1), *i, *t);
            }
            prop_assert_eq!(
                a.has_range(StreamId(1), iv(0, 200)).expect("not failing"),
                b.has_range(StreamId(1), iv(0, 200)).expect("not failing")
            );
        }
    }
}
