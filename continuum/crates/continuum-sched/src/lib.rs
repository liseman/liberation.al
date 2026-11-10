//! The broker's decision making, as a library with no I/O (docs/CONTINUUM.md, "Scheduler" and "Request flows").
//!
//! The broker's main loop owns one [`Scheduler`] and drives it as a state machine: every message that reaches the
//! broker becomes a call with the current time passed in (`now_ns`, UTC nanoseconds like every other time in
//! Continuum), and every call returns the [`Action`]s the broker must carry out (dispatch a job, preempt a lease,
//! fulfill or fail a promise). Nothing here opens a socket, touches shared memory, or reads a clock, so the whole
//! thing runs in a test with a fake catalog and a fake clock.
//!
//! What the scheduler owns: the capability registry built from worker manifests, the planner that turns a missing
//! range into a priced plan, the promise table, the lease table with the profile's memory and compute budgets, and
//! the cost model per capability. What it borrows: a narrow read-only view of the catalog ([`CatalogView`]), which
//! the broker implements on top of the real catalog plus its own runtime state (which streams are live, and how far
//! each has been written).
//!
//! What stays in the broker: claiming shared memory slots for an output lease (the scheduler says how many of which
//! class; the broker claims them and fills in locators), resolving input payload ids and locators for a dispatched
//! job from the (stream, range) the scheduler names, pinning them per lease, writing coverage and cost models to the
//! catalog, and every message encoding.
//!
//! [`Engine`] is the implementation. [`intervals`] is the set arithmetic the planner and [`fake::FakeCatalog`]
//! share.

pub mod action;
pub mod catalog;
pub mod cost;
pub mod engine;
pub mod fake;
pub mod ids;
pub mod intervals;
pub mod lease;
pub mod manifest;
pub mod profile;
pub mod query;
pub mod scheduler;
pub mod state;

pub use action::{Action, Dispatch, InputRange, LeaseEnd};
pub use catalog::{CatalogView, Covered, StreamKind, StreamView, ViewError};
pub use cost::{CapabilityKey, ComputeCost, CostModelRecord, CostWeights};
pub use engine::{Engine, Settings};
pub use fake::FakeCatalog;
pub use ids::{CapabilityId, ConsumerId, JobId, LeaseId, PromiseId, WorkerId};
pub use lease::{CoverageDone, DenyReason, IngestRef, LeaseDecision, LeaseInfo, LeaseReq, OutputLease, OutputRef};
pub use manifest::{Accepted, Capability, CapabilityRole, Manifest, Requirement, Telemetry};
pub use profile::{Profile, SlotClass};
pub use query::{
    Alternative, Plan, PlanStep, PromiseState, QueryDecision, QueryMode, QueryReq, Quote, QuoteStatus, Refusal,
    StepKind,
};
pub use scheduler::{Error, Outcome, Scheduler};
pub use state::SystemState;
