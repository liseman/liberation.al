//! What a worker tells the scheduler about itself: its capability manifest at registration, and its telemetry in
//! heartbeats (docs/CONTINUUM-BROKER.md).
//!
//! Stream names are resolved to [`StreamId`]s by the broker before the manifest reaches the scheduler, since the
//! broker owns stream registration.

use continuum_core::{ComputeDomain, StreamId};

use crate::ids::CapabilityId;

/// A worker's manifest: who it is and what it can produce.
#[derive(Clone, Debug, PartialEq)]
pub struct Manifest {
    /// The worker's producer id (`worker:pose`, `worker:asr`), the first half of a [`crate::CapabilityKey`].
    pub producer: String,
    /// A git SHA or a hash of the weights, so drift between builds can be traced in provenance.
    pub version_hash: String,
    /// The discrete GPU the worker runs on, as an index into the profile's `vram_mb`, so VRAM is budgeted per
    /// device. `None` where there is no discrete GPU (the Mac) or the worker uses none.
    pub device_index: Option<u8>,
    /// What it can make.
    pub capabilities: Vec<Capability>,
}

/// One input a capability needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Requirement {
    /// The input stream.
    pub stream: StreamId,
    /// How far before the output's time the input must reach, in nanoseconds (0 for the same frame).
    pub before_ns: u64,
    /// How far after the output's time the input must reach, in nanoseconds.
    pub after_ns: u64,
}

/// What kind of work a capability is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CapabilityRole {
    /// Runs a model or a transform on inputs in Tier 0 and produces `produces`.
    Produce,
    /// The tier worker's hydrate: decodes any stream's Tier 1 or Tier 2 bytes into Tier 0. `produces` and
    /// `requires` are ignored; the planner uses it for every [`crate::StepKind::Hydrate`] step.
    Hydrate,
}

/// One thing a worker can make, and what making it costs before the scheduler has measured it.
#[derive(Clone, Debug, PartialEq)]
pub struct Capability {
    /// Produce or hydrate.
    pub role: CapabilityRole,
    /// The output stream (for [`CapabilityRole::Produce`]).
    pub produces: StreamId,
    /// The input streams and the window of each around the output's time.
    pub requires: Vec<Requirement>,
    /// Where it runs.
    pub domain: ComputeDomain,
    /// RAM the loaded model holds, in MB.
    pub ram_mb: u64,
    /// VRAM the loaded model holds, in MB (0 on the Mac, where the GPU shares RAM).
    pub vram_mb: u64,
    /// The worker's own first guess at wall time per item, in microseconds, used until measured.
    pub est_us_per_item: u64,
    /// How long loading the model takes, in milliseconds.
    pub cold_start_ms: u64,
    /// Most items per job.
    pub max_batch: u32,
    /// The largest payload one job writes, in bytes. The scheduler picks the smallest size class that holds it.
    pub output_bytes: u64,
    /// How many payloads one job writes: 1 for a worker that commits one batch per job, `max_batch` for one per
    /// item. The scheduler leases that many slots.
    pub outputs_per_job: u32,
    /// Whether a lease can be taken back mid-job. A job that can't be is quoted as one unit.
    pub preemptible: bool,
    /// The confidence this producer typically reports, if it reports one, so alternatives can be compared.
    pub expected_confidence: Option<f32>,
}

/// What registration decided about one capability (`WorkerRegistered`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Accepted {
    /// The id the scheduler gave it, in manifest order.
    pub capability: CapabilityId,
    /// Whether it can ever run here. False when the profile has no capacity on its compute domain (the Neural
    /// Engine on Linux); it is still numbered so telemetry can name it, but never dispatched or granted.
    pub runnable: bool,
}

/// What a worker reports in each heartbeat.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Telemetry {
    /// Jobs the worker has taken and not yet committed or aborted.
    pub queue_depth: u32,
    /// The capabilities whose models are loaded right now. One not listed pays its cold start when dispatched.
    pub loaded: Vec<CapabilityId>,
    /// RAM in use, in MB.
    pub ram_mb: u64,
    /// VRAM in use, in MB.
    pub vram_mb: u64,
}
