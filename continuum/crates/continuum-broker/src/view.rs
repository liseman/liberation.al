//! The scheduler's view of the catalog ([`continuum_sched::CatalogView`]), answered from memory.
//!
//! The scheduler asks its questions on its own thread, which must never wait on disk (docs/CONTINUUM-BROKER.md, the
//! broker's no-stall rule), so [`IndexView`] answers them from the broker's [`AvailabilityIndex`] and its record of
//! which streams are live, never from the catalog. Every method therefore succeeds; [`ViewError`] is for views that
//! read storage, and this one doesn't.
//!
//! [`ViewError`]: continuum_sched::ViewError

use std::collections::HashMap;

use continuum_core::{Interval, StageId, StreamId};
use continuum_sched::{CatalogView, Covered, StreamView, ViewError};

use crate::index::AvailabilityIndex;

/// The view for one scheduler call: the index, and the live streams with their live edge (the newest time the
/// producer has written, if it has reported one).
pub struct IndexView<'a> {
    index: &'a AvailabilityIndex,
    live: &'a HashMap<StreamId, Option<u64>>,
}

impl<'a> IndexView<'a> {
    /// A view over `index`, where the streams in `live` have a connected producer.
    #[must_use]
    pub fn new(index: &'a AvailabilityIndex, live: &'a HashMap<StreamId, Option<u64>>) -> Self {
        Self { index, live }
    }
}

impl CatalogView for IndexView<'_> {
    fn stream(&self, stream: StreamId) -> Result<Option<StreamView>, ViewError> {
        Ok(self.index.meta(stream).map(|meta| StreamView {
            id: stream,
            kind: meta.kind,
            live: self.live.contains_key(&stream),
            rate_hz: meta.rate_hz,
            live_edge_ns: self.live.get(&stream).copied().flatten(),
        }))
    }

    fn has_range(&self, stream: StreamId, range: Interval) -> Result<Vec<Covered>, ViewError> {
        Ok(self.index.covered(stream, range))
    }

    fn coverage_gaps(
        &self,
        source: StreamId,
        stage: StageId,
        window: Interval,
        max: usize,
    ) -> Result<Vec<Interval>, ViewError> {
        Ok(self.index.coverage_gaps(source, stage, window, max))
    }
}

#[cfg(test)]
mod tests {
    use continuum_core::StorageTier;
    use continuum_sched::StreamKind;

    use super::*;
    use crate::index::StreamMeta;

    fn iv(a: u64, b: u64) -> Interval {
        Interval::new(a, b).unwrap()
    }

    #[test]
    fn the_view_answers_from_the_index_and_the_live_set() {
        let mut index = AvailabilityIndex::new();
        let (cam, pose) = (StreamId(1), StreamId(2));
        index.register(cam, StreamMeta { kind: StreamKind::Sensor, rate_hz: 30.0 });
        index.register(pose, StreamMeta { kind: StreamKind::Derived, rate_hz: 0.0 });
        index.add_copy(cam, iv(0, 100), StorageTier::Tier1Warm).unwrap();
        index.add_copy(cam, iv(50, 100), StorageTier::Tier0Hot).unwrap();
        index.add_coverage(cam, StageId(7), iv(0, 40));
        let live = HashMap::from([(cam, Some(120))]);
        let view = IndexView::new(&index, &live);

        assert_eq!(
            view.stream(cam),
            Ok(Some(StreamView {
                id: cam,
                kind: StreamKind::Sensor,
                live: true,
                rate_hz: 30.0,
                live_edge_ns: Some(120)
            }))
        );
        assert_eq!(
            view.stream(pose),
            Ok(Some(StreamView { id: pose, kind: StreamKind::Derived, live: false, rate_hz: 0.0, live_edge_ns: None }))
        );
        assert_eq!(view.stream(StreamId(9)), Ok(None));
        assert_eq!(
            view.has_range(cam, iv(0, 200)),
            Ok(vec![
                Covered { interval: iv(0, 50), tier: StorageTier::Tier1Warm },
                Covered { interval: iv(50, 100), tier: StorageTier::Tier0Hot },
            ])
        );
        assert_eq!(view.has_range(StreamId(9), iv(0, 10)), Ok(vec![]));
        assert_eq!(view.coverage_gaps(cam, StageId(7), iv(0, 100), 10), Ok(vec![iv(40, 100)]));

        // Live without a reported edge, and live but never registered (the index decides what exists).
        let ghost = StreamId(9);
        let live = HashMap::from([(pose, None), (ghost, Some(5))]);
        let view = IndexView::new(&index, &live);
        assert_eq!(view.stream(pose).unwrap().map(|s| (s.live, s.live_edge_ns)), Some((true, None)));
        assert_eq!(view.stream(ghost), Ok(None));
    }
}
