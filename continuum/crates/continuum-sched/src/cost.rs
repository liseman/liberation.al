//! The cost model: what the scheduler learns about each capability and how it prices a plan (docs/CONTINUUM-BROKER.md,
//! "Scheduler").
//!
//! The quote formula is `est_latency = storage_fetch + sum over steps of (cold_start + (queue_depth + 1) * items *
//! ema_per_item)`, with the p95 in place of the EMA under a strict budget. Cost units are compute time weighted by
//! domain plus memory held over time, so plans on different hardware compare as one number.

use std::collections::BTreeMap;

use continuum_core::{ComputeDomain, StreamId};

/// What a cost model is about: a producer (the worker's `producer_id`, like `worker:pose`) and the stream it makes.
/// Two workers running the same producer share one model; the same producer making two streams has two.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CapabilityKey {
    /// The worker's producer id from its manifest.
    pub producer: String,
    /// The output stream.
    pub output: StreamId,
}

/// The learned cost of one capability, as the broker saves it to the catalog's `COST_MODELS` table and hands it
/// back on restart. Queue depth is runtime state and is not here.
#[derive(Clone, Debug, PartialEq)]
pub struct CostModelRecord {
    /// Which capability.
    pub key: CapabilityKey,
    /// Exponential moving average of wall time per item, in microseconds (alpha 0.2).
    pub ema_us_per_item: f64,
    /// 95th percentile of wall time per item, in microseconds, for strict budgets.
    pub p95_us_per_item: f64,
    /// How many commits the model has seen. Below a few, the quote leans on the manifest's own estimate.
    pub samples: u64,
    /// RAM the capability holds when loaded, in MB, from the manifest and corrected by telemetry.
    pub ram_mb: u64,
    /// VRAM the capability holds when loaded, in MB, likewise.
    pub vram_mb: u64,
    /// How long loading the model takes, in milliseconds.
    pub cold_start_ms: u64,
}

/// What a job cost, measured, as the worker reports it in `CommitLease` and the broker passes it on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComputeCost {
    /// Wall time from the job's start to its commit, in microseconds.
    pub wall_time_us: u64,
    /// Time the job waited in the worker's queue before starting, in microseconds.
    pub queue_wait_us: u64,
    /// Peak RAM the job used, in bytes.
    pub ram_bytes: u64,
    /// Peak VRAM the job used, in bytes.
    pub vram_bytes: u64,
    /// The device it ran on, as the worker names it (`CUDA:0`, `MPS`, `ANE`, `CPU`).
    pub device: String,
    /// How many items (frames, utterances, rows) the job processed, so cost per item can be learned.
    pub items: u64,
}

/// How compute time and memory become one cost number.
#[derive(Clone, Debug, PartialEq)]
pub struct CostWeights {
    /// Cost units per microsecond of compute on each domain. A domain missing here costs nothing extra.
    pub per_domain_us: BTreeMap<ComputeDomain, f64>,
    /// Cost units per megabyte held for one second (RAM and VRAM alike).
    pub per_mb_second: f64,
}

impl Default for CostWeights {
    /// First guesses: CPU time at 1 unit per millisecond, accelerators at 4 (they are the scarce resource), media
    /// engine time at 0.5, and memory at 0.01 per MB second.
    fn default() -> Self {
        Self {
            per_domain_us: BTreeMap::from([
                (ComputeDomain::CPU, 0.001),
                (ComputeDomain::GPU, 0.004),
                (ComputeDomain::ANE, 0.004),
                (ComputeDomain::MediaEngine, 0.0005),
            ]),
            per_mb_second: 0.01,
        }
    }
}
