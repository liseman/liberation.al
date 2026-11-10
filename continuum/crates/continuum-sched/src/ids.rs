//! Ids the scheduler mints or is handed.
//!
//! Streams and payloads use `continuum_core::{StreamId, Uuid}`. The ids here are local to one broker process: the
//! broker hands the scheduler a [`WorkerId`] or [`ConsumerId`] for each connected session, and the scheduler mints
//! [`PromiseId`], [`LeaseId`], and [`JobId`] from counters so every action names things consistently. None of them
//! is stored in the catalog or sent to another machine.

/// A connected worker: the broker's session id for the connection that sent `RegisterWorker`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WorkerId(pub u64);

/// A connected client that sends queries: the broker's session id for its connection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ConsumerId(pub u64);

/// A promise, from the quote that created it to its end (`promise_id` on the wire).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PromiseId(pub u64);

/// A lease on output slots, memory, and a compute domain share (`lease_id` on the wire).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LeaseId(pub u64);

/// A dispatched job (`job_id` on the wire). A job runs one capability once, under one lease.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct JobId(pub u64);

/// One capability of one worker, as the scheduler numbered it when the worker registered. The same model on two
/// workers is two capabilities; the cost model is shared through [`crate::CapabilityKey`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CapabilityId(pub u32);
