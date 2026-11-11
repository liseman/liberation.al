//! The scheduler's view of the catalog: the few questions the planner asks, and nothing else.
//!
//! The broker implements [`CatalogView`] on top of the real catalog (`continuum-catalog`) plus its own
//! runtime state: whether a stream is live, and how far its producer has written, are matters of heartbeats and
//! ring state, not of anything stored. The scheduler never depends on the catalog crate or on shared memory, so
//! it tests against [`crate::fake::FakeCatalog`].

use std::fmt;

use continuum_core::{Interval, StageId, StorageTier, StreamId};

/// Whether a stream comes from hardware or from a worker.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StreamKind {
    /// A capture worker's stream: cameras, microphones, the IMU. Only these can be awaited into the future.
    Sensor,
    /// A worker's output. Missing ranges are made by dispatching the capability that produces it.
    Derived,
}

/// What the planner needs to know about a stream.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StreamView {
    /// The stream.
    pub id: StreamId,
    /// Sensor or derived.
    pub kind: StreamKind,
    /// Whether the stream's producer is connected and heartbeating now. A future range on a dead sensor stream is
    /// refused, not quoted.
    pub live: bool,
    /// Nominal items per second (frames, blocks, batches), for estimating how many items a range holds. 0 if the
    /// stream has no rate (events).
    pub rate_hz: f64,
}

/// A stretch of a stream that exists, and the hottest tier holding it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Covered {
    /// The covered time.
    pub interval: Interval,
    /// The hottest tier that holds these bytes now. Tier 1 and 2 cost a hydrate step in the plan.
    pub tier: StorageTier,
}

/// The catalog could not answer: an I/O or storage error underneath, never "not found" (that is `None` or an
/// empty answer). The scheduler never guesses past one: a query gets [`crate::Refusal::Catalog`], and a replan fails
/// the promises it was for, so the broker never has to check a side channel after a call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ViewError(pub String);

impl fmt::Display for ViewError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "catalog: {}", self.0)
    }
}

impl std::error::Error for ViewError {}

/// Read-only questions the planner asks the catalog. Every method is a point-in-time answer: the scheduler is
/// handed the view for one call and does not keep it, so each call that may plan takes it again.
pub trait CatalogView {
    /// The stream, or `None` for an id the broker has never registered.
    fn stream(&self, stream: StreamId) -> Result<Option<StreamView>, ViewError>;

    /// Which parts of `range` exist for `stream`, each with the hottest tier holding it, in time order and
    /// pairwise disjoint. The uncovered parts are what a quote prices. An unknown stream has no parts.
    fn has_range(&self, stream: StreamId, range: Interval) -> Result<Vec<Covered>, ViewError>;

    /// Within `window`, the parts of `source` that `stage` has not processed yet, in time order, at most `max` of
    /// them (the `COVERAGE` table, docs/CONTINUUM-BROKER.md).
    fn coverage_gaps(
        &self,
        source: StreamId,
        stage: StageId,
        window: Interval,
        max: usize,
    ) -> Result<Vec<Interval>, ViewError>;
}
