//! The consumer flow: a query, what came back, and the quote and promise it may create (docs/CONTINUUM-BROKER.md,
//! "Protocol").

use std::fmt;

use continuum_core::{Interval, PriorityBand, StorageTier, StreamId};

use crate::catalog::Covered;
use crate::ids::{CapabilityId, ConsumerId, PromiseId, WorkerId};

/// What the consumer wants done about data that doesn't exist.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryMode {
    /// Return what exists and the missing intervals, and stop.
    HitOnly,
    /// Return what exists and a quote for the rest.
    QuoteIfMissing,
    /// Quote, and commit at once if the quote fits the budget. With no budget, every quote fits.
    CommitWithinBudget,
}

/// A consumer's `Query`.
#[derive(Clone, Debug, PartialEq)]
pub struct QueryReq {
    /// Who asked, so fulfillment finds them.
    pub consumer: ConsumerId,
    /// The stream wanted.
    pub stream: StreamId,
    /// The time wanted. May reach past `now` for a live sensor stream (a future range).
    pub range: Interval,
    /// The latency the consumer can accept, in microseconds, if it has one.
    pub budget_us: Option<u64>,
    /// Price with the p95 instead of the average, for a consumer that would rather be refused than late. Ignored
    /// without a budget.
    pub strict: bool,
    /// The band the work would run in if committed.
    pub band: PriorityBand,
    /// What to do about missing data.
    pub mode: QueryMode,
}

/// What a query decided. This is the `QueryResult`, with the `PromiseQuote` when there is one.
#[derive(Clone, Debug, PartialEq)]
pub enum QueryDecision {
    /// What exists, and what doesn't. Complete when `missing` is empty. The broker fetches the headers and locators
    /// for `covered` and pins them. This is every answer in [`QueryMode::HitOnly`], and the answer in the other
    /// modes when nothing is missing or when nothing missing can be made (a sensor stream's past that nobody
    /// captured), in which case `missing` says what is gone.
    Hit {
        /// What exists, in time order.
        covered: Vec<Covered>,
        /// What doesn't, in time order.
        missing: Vec<Interval>,
    },
    /// Some or none of it exists; the rest is quoted. Nothing runs until the consumer commits.
    Quoted {
        /// What exists, possibly empty.
        covered: Vec<Covered>,
        /// The quote for the rest, which names the missing intervals.
        quote: Quote,
    },
    /// The quote fit the budget and the mode was [`QueryMode::CommitWithinBudget`], so it is already committed. The
    /// actions it started come back in the call's [`crate::Outcome`].
    Committed {
        /// What exists, possibly empty.
        covered: Vec<Covered>,
        /// The quote that was committed.
        quote: Quote,
    },
    /// Nothing can be done for this query.
    Refused(Refusal),
}

/// Why a query was refused rather than answered or quoted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// No stream has this id.
    UnknownStream(StreamId),
    /// A future range on a sensor stream whose capture worker isn't connected.
    StreamNotLive(StreamId),
    /// A derived stream nobody registered a capability for, or whose inputs can't be made either.
    NoProducer(StreamId),
    /// Part of the range is in Tier 1 or 2 and no hydrate capability is registered to bring it back.
    NoHydrator,
    /// A derived stream needs sensor data from before the live edge that nobody captured, so it can't be made.
    NotCaptured(StreamId),
    /// The catalog could not answer ([`crate::ViewError`]); nothing was planned on a guess.
    Catalog(String),
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownStream(s) => write!(f, "unknown stream {}", s.0),
            Self::StreamNotLive(s) => write!(f, "stream {} is not live", s.0),
            Self::NoProducer(s) => write!(f, "nothing registered produces stream {}", s.0),
            Self::NoHydrator => f.write_str("no hydrate capability is registered"),
            Self::NotCaptured(s) => write!(f, "stream {} was not captured over part of the range", s.0),
            Self::Catalog(e) => write!(f, "catalog: {e}"),
        }
    }
}

/// Why data has to wait.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuoteStatus {
    /// The inputs exist; workers have to run.
    RequiresCompute,
    /// A sensor range in the future; it arrives when captured, with nothing to compute.
    AwaitingCapture,
    /// A derived range in the future: capture first, then compute.
    AwaitingCaptureAndCompute,
}

/// One step of a plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepKind {
    /// Wait for a live sensor stream to capture the range.
    Capture,
    /// Have the tier worker decode a Tier 1 or Tier 2 range into Tier 0.
    Hydrate {
        /// The tier it comes from.
        from: StorageTier,
    },
    /// Run a capability on inputs that are (or will be) in Tier 0.
    Compute,
}

/// A step of a plan: what it makes, who does it, and what it is expected to take.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanStep {
    /// What kind of step.
    pub kind: StepKind,
    /// The stream this step produces or brings into Tier 0.
    pub stream: StreamId,
    /// The range of it.
    pub range: Interval,
    /// The capability that runs it, for [`StepKind::Compute`] and [`StepKind::Hydrate`].
    pub capability: Option<CapabilityId>,
    /// The worker the capability is on.
    pub worker: Option<WorkerId>,
    /// How many items the step handles (frames, blocks, rows), from the range and the stream's rate.
    pub items: u64,
    /// The step's estimated time, in microseconds, by the cost formula.
    pub est_us: u64,
    /// Whether this step is work already in flight for someone else that this promise attaches to.
    pub shared: bool,
}

/// An ordered plan from inputs to the wanted stream. Steps run in order; a step's inputs are the steps before it or
/// data already present.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Plan {
    /// The steps, in execution order.
    pub steps: Vec<PlanStep>,
}

/// Another way to make the same stream (a faster or heavier producer, a cheaper input), priced alongside the plan.
#[derive(Clone, Debug, PartialEq)]
pub struct Alternative {
    /// The plan.
    pub plan: Plan,
    /// Its estimated latency, in microseconds.
    pub est_latency_us: u64,
    /// Its cost in cost units.
    pub cost_units: f64,
    /// Whether it is over the consumer's budget.
    pub exceeds_budget: bool,
    /// The confidence its producer typically reports, if known.
    pub expected_confidence: Option<f32>,
}

/// A `PromiseQuote`: what it would take to make the missing part, and the promise id to commit or cancel.
#[derive(Clone, Debug, PartialEq)]
pub struct Quote {
    /// The promise this quote created, in state [`PromiseState::Quoted`].
    pub promise: PromiseId,
    /// Why the data has to wait.
    pub status: QuoteStatus,
    /// The missing part of the range this quote covers.
    pub missing: Vec<Interval>,
    /// Missing parts nothing can make: a sensor stream's past before the live edge that nobody captured. Reported
    /// so the consumer knows the quote doesn't cover them.
    pub unavailable: Vec<Interval>,
    /// Estimated time until the last part is delivered, in microseconds.
    pub est_latency_us: u64,
    /// The plan's cost in cost units, near zero when it attaches to in-flight work.
    pub cost_units: f64,
    /// Whether `est_latency_us` is over the consumer's budget. The quote is still offered.
    pub exceeds_budget: bool,
    /// The default plan, the one committed when no alternative is chosen.
    pub plan: Plan,
    /// Other plans, chosen by index in `CommitPromise`.
    pub alternatives: Vec<Alternative>,
    /// Whether any step attaches to work already in flight for another promise (coalescing).
    pub shares_in_flight: bool,
    /// When the quote stops being committable (the quote's time to live, 2 seconds by default).
    pub expires_ns: u64,
}

/// Where a promise is in its life (the schema's `PromiseState`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PromiseState {
    /// Plan built, nothing running; expires after the quote's time to live.
    Quoted,
    /// Committed, and waiting for its time range to be captured.
    Armed,
    /// Inputs being staged and jobs running.
    Dispatching,
    /// Every part delivered.
    Fulfilled,
    /// The consumer cancelled it, or went away, before it was fulfilled.
    Cancelled,
    /// The quote was never committed in time.
    Expired,
    /// A step failed or timed out.
    Failed,
}
