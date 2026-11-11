//! What the scheduler asks the broker to do.
//!
//! Every mutating [`crate::Scheduler`] call returns a list of these, in the order to carry them out. Replies that go
//! straight back to the caller of the message (a quote, a lease decision) are return values instead, so an action
//! is always something for a party other than the caller.

use continuum_core::{Interval, PriorityBand, StreamId};

use crate::ids::{CapabilityId, ConsumerId, JobId, LeaseId, PromiseId, WorkerId};
use crate::lease::{OutputLease, OutputRef};

/// An input a dispatched job reads: a stream over a range, present in Tier 0 (or being hydrated by an earlier
/// step). The broker resolves the payload ids and locators from the catalog and hands them out; a locator in Tier 0
/// is covered only by the receiving session's reader pin, which lasts a few seconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InputRange {
    /// The input stream.
    pub stream: StreamId,
    /// The time needed, the step's range widened by the capability's window.
    pub range: Interval,
}

/// A `DispatchJob`: run one capability on one worker, under an output lease already granted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dispatch {
    /// The job.
    pub job: JobId,
    /// The worker to send it to.
    pub worker: WorkerId,
    /// The capability to run.
    pub capability: CapabilityId,
    /// The promises this job serves: at least one, several when coalesced.
    pub promises: Vec<PromiseId>,
    /// What to read.
    pub inputs: Vec<InputRange>,
    /// The output range to produce.
    pub range: Interval,
    /// Where to write, and when the lease ends.
    pub output: OutputLease,
    /// The band it runs in.
    pub band: PriorityBand,
    /// When the result is no longer wanted, UTC nanoseconds; the worker may stop at this point.
    pub deadline_ns: u64,
}

/// Why a lease left the scheduler's table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LeaseEnd {
    /// The worker released it.
    Released,
    /// The worker committed its outputs.
    Committed,
    /// The worker aborted it.
    Aborted,
    /// The worker rejected the job it was granted for.
    Rejected,
    /// It ran past `expires_ns` (an Active lease was also revoked; a Preempting one just ended).
    Expired,
    /// Its worker disconnected.
    WorkerGone,
    /// The only promise its job served ended.
    PromiseEnded,
    /// The broker couldn't claim the slots it was granted.
    ClaimFailed,
}

/// Something the broker must do.
#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    /// Send a `DispatchJob`. A hydrate is a dispatch to the tier worker's hydrate capability.
    Dispatch(Dispatch),
    /// Send `LeasePreempted`: a higher band needs what this lease holds. The worker stops at its next boundary and
    /// its uncommitted output is discarded.
    Preempt {
        /// The worker holding the lease.
        worker: WorkerId,
        /// The lease.
        lease: LeaseId,
        /// The band that took it.
        by_band: PriorityBand,
        /// Why, for the log and the message.
        reason: String,
    },
    /// Take a lease back without a higher band asking: it expired, its worker is gone, the broker couldn't claim
    /// its slots, or the only promise its job served was cancelled. Free its slots; tell the worker if it is still
    /// connected and the job may still be running.
    RevokeLease {
        /// The worker that held it, if it is still connected.
        worker: Option<WorkerId>,
        /// The lease.
        lease: LeaseId,
        /// Why.
        reason: String,
    },
    /// Send `FulfillPromise` with these outputs. A long range arrives in several parts; the last is `final_part`.
    Fulfill {
        /// The promise.
        promise: PromiseId,
        /// Who gets it.
        consumer: ConsumerId,
        /// The payloads delivered in this part. The broker attaches headers, locators, and provenance.
        outputs: Vec<OutputRef>,
        /// Whether this completes the promise.
        final_part: bool,
    },
    /// Send `PromiseFailed`.
    FailPromise {
        /// The promise.
        promise: PromiseId,
        /// Who gets it.
        consumer: ConsumerId,
        /// Why, for the message.
        reason: String,
    },
    /// The lease's pins can go. Nothing in the broker is held per lease today (a dispatched job's inputs are covered
    /// by the receiving session's reader pin, which expires on its own), so the broker only notes it.
    Unpin {
        /// The lease whose pins are released.
        lease: LeaseId,
    },
    /// The lease is gone from the scheduler, whatever the exit: forget it in any map keyed by lease (ownership).
    /// Sent exactly once per lease, after its `Unpin`, and alongside any
    /// `Preempt` or `RevokeLease` the same exit sends, so a map can never keep an entry the scheduler has dropped.
    LeaseEnded {
        /// The lease.
        lease: LeaseId,
        /// The worker that held it (connected or not).
        worker: WorkerId,
        /// Why.
        end: LeaseEnd,
    },
}
