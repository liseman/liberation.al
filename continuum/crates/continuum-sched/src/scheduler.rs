//! The [`Scheduler`] contract: the calls the broker's main loop makes, one per message or timer, each with the
//! current time passed in and returning what to do next.

use std::fmt;

use continuum_core::PriorityBand;

use crate::action::Action;
use crate::catalog::CatalogView;
use crate::cost::{ComputeCost, CostModelRecord};
use crate::ids::{CapabilityId, ConsumerId, JobId, LeaseId, PromiseId, WorkerId};
use crate::lease::{CoverageDone, IngestRef, LeaseDecision, LeaseInfo, LeaseReq, OutputRef};
use crate::manifest::{Accepted, Manifest, Telemetry};
use crate::query::{PromiseState, QueryDecision, QueryReq};
use crate::state::SystemState;

/// A reply to the caller together with the actions for everyone else.
#[derive(Clone, Debug, PartialEq)]
pub struct Outcome<T> {
    /// What to tell the caller.
    pub decision: T,
    /// What the broker must do besides (dispatches a commit started, preemptions that made room).
    pub actions: Vec<Action>,
}

/// A call that named something the scheduler doesn't have, or asked for a transition the state doesn't allow.
/// These are the broker's bugs or a client's stale message, never normal flow.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// No such promise.
    UnknownPromise(PromiseId),
    /// No such lease.
    UnknownLease(LeaseId),
    /// No such job.
    UnknownJob(JobId),
    /// No such worker, or it never registered.
    UnknownWorker(WorkerId),
    /// The promise isn't in a state that allows this: committing one that isn't `Quoted`, or cancelling one that
    /// already ended (`Fulfilled`, `Cancelled`, `Expired`, `Failed`).
    PromiseState {
        /// The promise.
        promise: PromiseId,
        /// Its state.
        state: PromiseState,
    },
    /// The quote's time to live passed before the commit.
    QuoteExpired(PromiseId),
    /// `CommitPromise` named an alternative the quote didn't offer.
    UnknownAlternative {
        /// The promise.
        promise: PromiseId,
        /// The index asked for.
        alternative: u32,
    },
    /// The lease request named a capability this worker didn't register, or one that can't run here.
    UnknownCapability {
        /// The worker.
        worker: WorkerId,
        /// The capability.
        capability: CapabilityId,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownPromise(p) => write!(f, "unknown promise {}", p.0),
            Self::UnknownLease(l) => write!(f, "unknown lease {}", l.0),
            Self::UnknownJob(j) => write!(f, "unknown job {}", j.0),
            Self::UnknownWorker(w) => write!(f, "unknown worker {}", w.0),
            Self::PromiseState { promise, state } => write!(f, "promise {} is {state:?}", promise.0),
            Self::QuoteExpired(p) => write!(f, "quote for promise {} expired", p.0),
            Self::UnknownAlternative { promise, alternative } => {
                write!(f, "promise {} offered no alternative {alternative}", promise.0)
            }
            Self::UnknownCapability { worker, capability } => {
                write!(f, "worker {} has no runnable capability {}", worker.0, capability.0)
            }
        }
    }
}

impl std::error::Error for Error {}

/// The broker's decision maker. One instance per broker, owned by the main loop task and called synchronously;
/// nothing here blocks, sleeps, or reads a clock.
///
/// Time is `now_ns`, UTC nanoseconds, passed to every call so tests can drive it. Ids for promises, leases, and jobs
/// are minted here; the broker passes them back as they come. Every call that may plan or replan takes the
/// [`CatalogView`] for that call, since the scheduler never keeps it (what exists can change between any two calls:
/// a slot is evicted, a parallel job fills part of a range). A [`crate::ViewError`] from the view is never
/// guessed past: a query is refused with [`crate::Refusal::Catalog`], and a replan fails the promises it was for
/// with [`Action::FailPromise`]; the broker has nothing to check after the call.
pub trait Scheduler {
    /// A worker registered: index its capabilities, returning one [`Accepted`] per capability in manifest order.
    fn register_worker(&mut self, worker: WorkerId, manifest: Manifest, now_ns: u64) -> Vec<Accepted>;

    /// A worker disconnected or died: revoke its leases, fail or replan the promises its jobs served, and forget
    /// its capabilities.
    fn worker_gone(&mut self, catalog: &dyn CatalogView, worker: WorkerId, now_ns: u64) -> Vec<Action>;

    /// A consumer disconnected or died: cancel its promises as [`Scheduler::cancel_promise`] would, so nothing
    /// runs for nobody and no action names it again.
    fn consumer_gone(&mut self, consumer: ConsumerId, now_ns: u64) -> Vec<Action>;

    /// A heartbeat's telemetry: queue depth, loaded models, memory in use.
    fn telemetry(&mut self, worker: WorkerId, telemetry: Telemetry, now_ns: u64);

    /// A consumer's `Query`. Answers from the catalog view, and quotes (or commits, by mode) the missing part. The
    /// actions are empty unless the mode committed.
    fn query(&mut self, catalog: &dyn CatalogView, req: QueryReq, now_ns: u64) -> Outcome<QueryDecision>;

    /// `CommitPromise`: run the quoted plan, or the alternative at this index. A future range arms instead of
    /// dispatching. Fails if the quote expired or the promise isn't `Quoted`.
    fn commit_promise(
        &mut self,
        catalog: &dyn CatalogView,
        promise: PromiseId,
        alternative: Option<u32>,
        band: PriorityBand,
        now_ns: u64,
    ) -> Result<Vec<Action>, Error>;

    /// `CancelPromise`: end a promise that hasn't ended. A quoted or armed promise is dropped. A dispatching one is
    /// detached from its jobs, which run on if another promise shares them and are revoked otherwise.
    fn cancel_promise(&mut self, promise: PromiseId, now_ns: u64) -> Result<Vec<Action>, Error>;

    /// An `IngestNotice` landed: fire armed promises whose range it completes, and start their compute steps.
    fn on_ingest(&mut self, catalog: &dyn CatalogView, ingest: IngestRef, now_ns: u64) -> Vec<Action>;

    /// A worker's `QuoteRequest` for self-started work. Granting may preempt lower bands, which come back as
    /// actions.
    fn quote_request(&mut self, worker: WorkerId, req: LeaseReq, now_ns: u64) -> Result<Outcome<LeaseDecision>, Error>;

    /// `RenewLease`: push a lease's expiry out for a long job, to `extend_ns` from now (clamped between the shortest
    /// lease and `max_lease_ns`; 0 takes the default `renew_ns`). An expiry never moves earlier, and a preempting
    /// lease isn't extended. Returns the new expiry.
    fn renew_lease(&mut self, lease: LeaseId, extend_ns: u64, now_ns: u64) -> Result<u64, Error>;

    /// `CommitLease`: the job is done. Updates the cost model from `cost`, releases the lease, and fulfills or
    /// advances the promises its outputs serve.
    fn commit_lease(
        &mut self,
        lease: LeaseId,
        outputs: &[OutputRef],
        coverage: Option<CoverageDone>,
        cost: ComputeCost,
        now_ns: u64,
    ) -> Result<Vec<Action>, Error>;

    /// `AbortLease`: the job failed. Frees the slots and fails or replans the promises it served.
    fn abort_lease(
        &mut self,
        catalog: &dyn CatalogView,
        lease: LeaseId,
        reason: &str,
        now_ns: u64,
    ) -> Result<Vec<Action>, Error>;

    /// `LeaseReleased`: the worker gave a lease back, after a preemption or early.
    fn release_lease(&mut self, lease: LeaseId, now_ns: u64) -> Result<Vec<Action>, Error>;

    /// `JobRejected`: the worker can't take a dispatched job. Replans onto another producer or fails the promises.
    fn job_rejected(
        &mut self,
        catalog: &dyn CatalogView,
        job: JobId,
        reason: &str,
        now_ns: u64,
    ) -> Result<Vec<Action>, Error>;

    /// The broker itself could not carry out a dispatch: an input range it was asked to resolve is no longer in
    /// Tier 0 (evicted between quote and commit), or the worker's connection is gone. Replans like a rejection.
    fn dispatch_failed(
        &mut self,
        catalog: &dyn CatalogView,
        job: JobId,
        reason: &str,
        now_ns: u64,
    ) -> Result<Vec<Action>, Error>;

    /// The broker could not claim the slots a lease was granted (the race between granting and claiming). The
    /// scheduler corrects its accounting and replans.
    fn claim_failed(&mut self, catalog: &dyn CatalogView, lease: LeaseId, now_ns: u64) -> Result<Vec<Action>, Error>;

    /// A system state changed. The voice turn moving to `Listening` or `Speaking` preempts P2 and P3 leases on the
    /// GPU and Neural Engine at once; the actions come back from this call, so their latency is the broker loop's.
    fn set_system_state(&mut self, state: SystemState, now_ns: u64) -> Vec<Action>;

    /// Time passed: expire quotes, leases, and armed promises whose deadline came and went.
    fn tick(&mut self, now_ns: u64) -> Vec<Action>;

    /// The learned cost models, for the broker to save in `COST_MODELS`.
    fn cost_models(&self) -> Vec<CostModelRecord>;

    /// Cost models saved by an earlier run, so quotes don't start from guesses.
    fn restore(&mut self, records: &[CostModelRecord]);

    /// A lease the scheduler still holds (its owner, its capability's inputs, its expiry), or `None` once it has
    /// ended; every end also sends [`Action::LeaseEnded`], so this is the check for a map that may have missed one.
    fn lease_info(&self, lease: LeaseId) -> Option<LeaseInfo>;

    /// Where a promise is, or `None` once it has been forgotten.
    fn promise_state(&self, promise: PromiseId) -> Option<PromiseState>;

    /// The promises a consumer has that haven't ended.
    fn promises_of(&self, consumer: ConsumerId) -> Vec<PromiseId>;
}
