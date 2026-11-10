//! The worker flow: leases, what a worker reports when it commits, and ingest notices (docs/CONTINUUM.md, "Worker
//! flow: jobs and leases" and "Ingest").
//!
//! A dispatched job carries its output lease inside the [`crate::Dispatch`] action, so only work a worker starts
//! itself (a backfill over coverage gaps) goes through [`LeaseReq`] and [`LeaseDecision`]. Capture holds a standing
//! stream lease from registration and never asks per frame; that lease is the broker's and shared memory's, not
//! the scheduler's.

use continuum_core::{Interval, PriorityBand, StageId, StreamId, Uuid};

use crate::ids::{CapabilityId, LeaseId, WorkerId};
use crate::manifest::{CapabilityRole, Requirement};

/// A worker's `QuoteRequest` for work it starts itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeaseReq {
    /// The capability it will run (one of those the scheduler returned at registration).
    pub capability: CapabilityId,
    /// The band to run in.
    pub band: PriorityBand,
    /// The output range it will produce.
    pub range: Interval,
    /// Output bytes wanted; the scheduler picks the size class and slot count from this.
    pub output_bytes: u64,
    /// How long the worker expects to take, in microseconds.
    pub est_duration_us: u64,
    /// Parent payloads to keep pinned in Tier 0 while the lease is active.
    pub pin_parents: Vec<Uuid>,
}

/// An output lease: the slots a job may write, and when they are taken back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutputLease {
    /// The lease.
    pub lease: LeaseId,
    /// The stream the outputs belong to.
    pub stream: StreamId,
    /// The size class of the slots, an index into the profile's classes.
    pub class: u16,
    /// How many slots. The broker claims them and fills in locators.
    pub slots: u32,
    /// When the broker revokes the lease and recycles the slots unless committed or renewed, UTC nanoseconds.
    pub expires_ns: u64,
}

/// Why a lease was not granted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenyReason {
    /// The capability's memory doesn't fit the profile beside what higher or equal bands hold.
    Memory,
    /// Every slot of the compute domain is taken by higher or equal bands.
    Compute,
    /// The size class has no free slots the scheduler may hand out.
    Slots,
    /// The profile has no capacity on this capability's compute domain at all.
    NoSuchDomain,
    /// The voice turn is active and the band is P2 or P3 on the GPU or Neural Engine.
    VoiceTurn,
}

/// A lease as the scheduler holds it, for the broker to check a commit against (its owner, its capability's
/// inputs) or to learn whether a lease it remembers still exists.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeaseInfo {
    /// The worker holding it.
    pub worker: WorkerId,
    /// The capability it was granted for.
    pub capability: CapabilityId,
    /// Produce or hydrate. A hydrate's `requires` is whatever the worker declared and means nothing (the manifest
    /// says it's ignored), so a coverage-source check skips hydrate leases.
    pub role: CapabilityRole,
    /// The capability's inputs, each with its window, as declared when the lease was granted: what a commit's
    /// coverage may cite as its source. Kept on the lease, so it stays right after the worker re-registers.
    pub requires: Vec<Requirement>,
    /// The band it runs in.
    pub band: PriorityBand,
    /// When it expires, UTC nanoseconds.
    pub expires_ns: u64,
    /// Whether a `Preempt` was sent and the worker hasn't released yet.
    pub preempting: bool,
}

/// The reply to a [`LeaseReq`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeaseDecision {
    /// Granted: write into these slots, commit before they expire.
    Granted(OutputLease),
    /// Not now.
    Denied {
        /// Why.
        reason: DenyReason,
        /// How long to wait before asking again, in microseconds.
        retry_after_us: u64,
    },
}

/// One payload a job produced, as the scheduler needs it to match outputs to the promises they serve. The broker
/// resolves the rest from the catalog.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutputRef {
    /// The payload.
    pub id: Uuid,
    /// Its stream.
    pub stream: StreamId,
    /// The time it covers.
    pub range: Interval,
}

/// A payload a producer just wrote, from its `IngestNotice`. Armed promises whose range it completes fire on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IngestRef {
    /// The payload.
    pub id: Uuid,
    /// Its stream.
    pub stream: StreamId,
    /// The time it covers.
    pub range: Interval,
}

/// The coverage a committing job finished, for the scheduler's bookkeeping of self-started work. The broker writes
/// it to the `COVERAGE` table itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoverageDone {
    /// The source stream the stage processed.
    pub source: StreamId,
    /// The stage.
    pub stage: StageId,
    /// The range now done.
    pub range: Interval,
}
