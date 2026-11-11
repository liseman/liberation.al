//! The [`Scheduler`] implementation: [`Engine`].
//!
//! Everything lives in a few maps keyed by the ids the engine mints, and every call walks only what it must: an
//! ingest notice looks up the promises armed on its stream, a system state change scans the active leases, a tick
//! scans promises and leases once. Nothing allocates on a path that has nothing to do, so the broker's loop can call
//! the engine at ring rate while the catalog is busy elsewhere (docs/CONTINUUM-BROKER.md).
//!
//! The planner is here too. It turns a missing range into steps (capture, hydrate, compute) by walking the
//! capability registry from the wanted stream down to inputs that exist, prices them with the cost model, and
//! attaches to jobs already in flight for the same stream and time (coalescing). Plans run their steps in order; a
//! step dispatches when the one before it has committed, or, for a capture step, when the ingest notices have
//! covered its range.
//!
//! Known limit, to settle with the broker: a promise whose shared job committed between quote and commit recomputes
//! the range instead of reading the result from the catalog, because no action says "fulfill from the catalog" yet.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use continuum_core::{ComputeDomain, Interval, PriorityBand, StorageTier, StreamId, VoiceTurnState};

use crate::action::{Action, Dispatch, InputRange, LeaseEnd};
use crate::catalog::{CatalogView, StreamKind, StreamView};
use crate::cost::{CapabilityKey, ComputeCost, CostModelRecord};
use crate::ids::{CapabilityId, ConsumerId, JobId, LeaseId, PromiseId, WorkerId};
use crate::intervals::{clip, merge, subtract};
use crate::lease::{CoverageDone, DenyReason, IngestRef, LeaseDecision, LeaseInfo, LeaseReq, OutputLease, OutputRef};
use crate::manifest::{Accepted, Capability, CapabilityRole, Manifest, Requirement, Telemetry};
use crate::profile::Profile;
use crate::query::{
    Alternative, Plan, PlanStep, PromiseState, QueryDecision, QueryMode, QueryReq, Quote, QuoteStatus, Refusal,
    StepKind,
};
use crate::scheduler::{Error, Outcome, Scheduler};
use crate::state::SystemState;

const NS_PER_US: u64 = 1_000;
const NS_PER_S: u64 = 1_000_000_000;

/// Tunables with their defaults. Every one is a setting, not a constant.
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    /// How long a quote stays committable (2 seconds).
    pub quote_ttl_ns: u64,
    /// The shortest lease a dispatched job gets; longer when the estimate says so.
    pub min_lease_ns: u64,
    /// How much a `RenewLease` adds when it asks for no particular time.
    pub renew_ns: u64,
    /// The most a `RenewLease` may push a lease out from now (5 minutes); a longer job renews again.
    pub max_lease_ns: u64,
    /// What a denied lease request is told to wait, in microseconds.
    pub retry_after_us: u64,
    /// How many times a promise is replanned after a failure before it fails.
    pub max_replans: u32,
    /// How long an ended promise stays answerable through `promise_state`.
    pub ended_retention_ns: u64,
    /// Added to a committed promise's estimated latency (and to a capture range's end) to make its deadline, after
    /// which `tick` fails it.
    pub promise_grace_ns: u64,
    /// How many per-item samples the p95 is taken over.
    pub p95_window: usize,
    /// The alpha of the per-item EMA.
    pub ema_alpha: f64,
    /// Deepest chain of derived inputs the planner follows.
    pub max_plan_depth: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            quote_ttl_ns: 2 * NS_PER_S,
            min_lease_ns: 10 * NS_PER_S,
            renew_ns: 10 * NS_PER_S,
            max_lease_ns: 5 * 60 * NS_PER_S,
            retry_after_us: 500_000,
            max_replans: 2,
            ended_retention_ns: 60 * NS_PER_S,
            promise_grace_ns: 30 * NS_PER_S,
            p95_window: 64,
            ema_alpha: 0.2,
            max_plan_depth: 8,
        }
    }
}

/// For a live sensor stream: which missing parts can still be captured (they start at or after `edge`, the clock when
/// the query was asked), and which are gone for good (they end before it). A part straddling the edge is split.
fn split_capturable(missing: &[Interval], edge: u64) -> (Vec<Interval>, Vec<Interval>) {
    let mut capturable = Vec::new();
    let mut gone = Vec::new();
    for m in missing {
        if m.start_ns() >= edge {
            capturable.push(*m);
        } else if m.end_ns() <= edge {
            gone.push(*m);
        } else {
            gone.push(Interval::new(m.start_ns(), edge).expect("start < edge"));
            capturable.push(Interval::new(edge, m.end_ns()).expect("edge < end"));
        }
    }
    (capturable, gone)
}

struct Worker {
    device_index: Option<u8>,
    caps: Vec<CapabilityId>,
    loaded: BTreeSet<CapabilityId>,
}

struct Cap {
    worker: WorkerId,
    spec: Capability,
    key: CapabilityKey,
    runnable: bool,
    /// Active leases on this capability; its memory is charged while this is above zero.
    active: u32,
}

#[derive(Clone, Debug)]
struct CostModel {
    ema_us: f64,
    p95_us: f64,
    samples: u64,
    window: VecDeque<f64>,
    ram_mb: u64,
    vram_mb: u64,
    cold_start_ms: u64,
    queue_depth: u32,
}

impl CostModel {
    fn from_capability(spec: &Capability) -> Self {
        Self {
            ema_us: spec.est_us_per_item as f64,
            p95_us: spec.est_us_per_item as f64,
            samples: 0,
            window: VecDeque::new(),
            ram_mb: spec.ram_mb,
            vram_mb: spec.vram_mb,
            cold_start_ms: spec.cold_start_ms,
            queue_depth: 0,
        }
    }

    fn observe(&mut self, per_item_us: f64, settings: &Settings) {
        if self.samples == 0 {
            self.ema_us = per_item_us;
        } else {
            self.ema_us = settings.ema_alpha * per_item_us + (1.0 - settings.ema_alpha) * self.ema_us;
        }
        self.samples += 1;
        self.window.push_back(per_item_us);
        while self.window.len() > settings.p95_window {
            self.window.pop_front();
        }
        let mut sorted: Vec<f64> = self.window.iter().copied().collect();
        sorted.sort_by(f64::total_cmp);
        let idx = ((sorted.len() as f64) * 0.95).ceil() as usize;
        self.p95_us = sorted[idx.clamp(1, sorted.len()) - 1];
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LeaseState {
    Active,
    Preempting,
}

struct Lease {
    worker: WorkerId,
    capability: CapabilityId,
    band: PriorityBand,
    class: u16,
    slots: u32,
    domain: ComputeDomain,
    /// The GPU the worker was on when the lease was granted, so VRAM goes back to the right device.
    device: Option<u8>,
    expires_ns: u64,
    job: Option<JobId>,
    preemptible: bool,
    state: LeaseState,
    /// The capability's role and inputs as declared when the lease was granted, for `lease_info`.
    role: CapabilityRole,
    requires: Vec<Requirement>,
}

struct Job {
    lease: LeaseId,
    worker: WorkerId,
    capability: CapabilityId,
    key: CapabilityKey,
    promises: Vec<PromiseId>,
    stream: StreamId,
    range: Interval,
    expected_end_ns: u64,
}

#[derive(Clone, Debug)]
enum StepStatus {
    Pending,
    Waiting,
    Running(JobId),
    Done,
}

#[derive(Clone, Debug)]
struct StepState {
    step: PlanStep,
    status: StepStatus,
    captured: Vec<Interval>,
}

struct Promise {
    consumer: ConsumerId,
    stream: StreamId,
    band: PriorityBand,
    strict: bool,
    state: PromiseState,
    missing: Vec<Interval>,
    quote_plan: Plan,
    alternatives: Vec<Alternative>,
    est_latency_us: u64,
    expires_ns: u64,
    deadline_ns: u64,
    steps: Vec<StepState>,
    replans: u32,
    ended_ns: Option<u64>,
}

struct Budget {
    slots: Vec<u32>,
    ram_mb: u64,
    vram_mb: Vec<u64>,
    domain: BTreeMap<ComputeDomain, u32>,
}

/// What a lease needs from the budget.
#[derive(Clone, Copy)]
struct Need {
    class: u16,
    slots: u32,
    ram_mb: u64,
    vram_mb: u64,
    device: Option<u8>,
    domain: ComputeDomain,
}

/// The scheduler. One per broker; see the module docs.
pub struct Engine {
    profile: Profile,
    settings: Settings,
    next_id: u64,
    workers: BTreeMap<WorkerId, Worker>,
    caps: BTreeMap<CapabilityId, Cap>,
    producers: BTreeMap<StreamId, Vec<CapabilityId>>,
    hydrators: Vec<CapabilityId>,
    costs: BTreeMap<CapabilityKey, CostModel>,
    leases: BTreeMap<LeaseId, Lease>,
    jobs: BTreeMap<JobId, Job>,
    promises: BTreeMap<PromiseId, Promise>,
    armed: BTreeMap<StreamId, Vec<PromiseId>>,
    budget: Budget,
    voice: VoiceTurnState,
}

impl Engine {
    /// An engine for `profile` with default settings.
    #[must_use]
    pub fn new(profile: Profile) -> Self {
        Self::with_settings(profile, Settings::default())
    }

    /// An engine for `profile` with these settings.
    #[must_use]
    pub fn with_settings(profile: Profile, settings: Settings) -> Self {
        let budget = Budget {
            slots: vec![0; profile.slot_classes.len()],
            ram_mb: 0,
            vram_mb: vec![0; profile.vram_mb.len()],
            domain: BTreeMap::new(),
        };
        Self {
            profile,
            settings,
            next_id: 1,
            workers: BTreeMap::new(),
            caps: BTreeMap::new(),
            producers: BTreeMap::new(),
            hydrators: Vec::new(),
            costs: BTreeMap::new(),
            leases: BTreeMap::new(),
            jobs: BTreeMap::new(),
            promises: BTreeMap::new(),
            armed: BTreeMap::new(),
            budget,
            voice: VoiceTurnState::Idle,
        }
    }

    /// The profile in force.
    #[must_use]
    pub fn profile(&self) -> &Profile {
        &self.profile
    }

    /// The settings in force.
    #[must_use]
    pub fn settings(&self) -> &Settings {
        &self.settings
    }

    /// Leases active right now (for tests and the broker's status page).
    #[must_use]
    pub fn active_leases(&self) -> usize {
        self.leases.values().filter(|l| l.state == LeaseState::Active).count()
    }

    /// VRAM charged on each device right now, in MB (for tests and the status page).
    #[must_use]
    pub fn vram_in_use_mb(&self) -> Vec<u64> {
        self.budget.vram_mb.clone()
    }

    fn mint(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    // ----- registry -------------------------------------------------------------------------------------------

    fn runnable_here(&self, domain: ComputeDomain) -> bool {
        self.profile.domain_slots.get(&domain).is_some_and(|n| *n > 0)
    }

    fn cap_usable(&self, id: CapabilityId) -> bool {
        self.caps.get(&id).is_some_and(|c| c.runnable && self.workers.contains_key(&c.worker))
    }

    fn cost_of(&self, key: &CapabilityKey, spec: &Capability) -> CostModel {
        self.costs.get(key).cloned().unwrap_or_else(|| CostModel::from_capability(spec))
    }

    fn forget_worker_caps(&mut self, worker: WorkerId) {
        let Some(w) = self.workers.remove(&worker) else { return };
        for cap in w.caps {
            if let Some(c) = self.caps.remove(&cap) {
                if let Some(list) = self.producers.get_mut(&c.spec.produces) {
                    list.retain(|x| *x != cap);
                }
                self.hydrators.retain(|x| *x != cap);
            }
        }
    }

    // ----- budget ---------------------------------------------------------------------------------------------

    fn class_for(&self, output_bytes: u64) -> Option<u16> {
        self.profile
            .slot_classes
            .iter()
            .enumerate()
            .find(|(_, c)| c.size_bytes >= output_bytes.max(1))
            .and_then(|(i, _)| u16::try_from(i).ok())
    }

    fn need_for(&self, cap: CapabilityId, worker: WorkerId) -> Option<Need> {
        let c = self.caps.get(&cap)?;
        let class = self.class_for(c.spec.output_bytes)?;
        let charged = c.active == 0;
        Some(Need {
            class,
            slots: c.spec.outputs_per_job.max(1),
            ram_mb: if charged { c.spec.ram_mb } else { 0 },
            vram_mb: if charged { c.spec.vram_mb } else { 0 },
            device: self.workers.get(&worker).and_then(|w| w.device_index),
            domain: c.spec.domain,
        })
    }

    fn vram_slot(&self, device: Option<u8>) -> Option<usize> {
        if self.profile.vram_mb.is_empty() {
            return None;
        }
        Some(usize::from(device.unwrap_or(0)).min(self.profile.vram_mb.len() - 1))
    }

    fn deny_reason(&self, need: Need, band: PriorityBand) -> Option<DenyReason> {
        let Some(&domain_cap) = self.profile.domain_slots.get(&need.domain) else {
            return Some(DenyReason::NoSuchDomain);
        };
        if self.voice_blocks(band, need.domain) {
            return Some(DenyReason::VoiceTurn);
        }
        let class = usize::from(need.class);
        if self.budget.slots[class] + need.slots > self.profile.slot_classes[class].count {
            return Some(DenyReason::Slots);
        }
        if self.budget.ram_mb + need.ram_mb > self.profile.ram_mb {
            return Some(DenyReason::Memory);
        }
        if let Some(d) = self.vram_slot(need.device)
            && self.budget.vram_mb[d] + need.vram_mb > self.profile.vram_mb[d]
        {
            return Some(DenyReason::Memory);
        }
        if self.budget.domain.get(&need.domain).copied().unwrap_or(0) + 1 > domain_cap {
            return Some(DenyReason::Compute);
        }
        None
    }

    fn voice_blocks(&self, band: PriorityBand, domain: ComputeDomain) -> bool {
        matches!(self.voice, VoiceTurnState::Listening | VoiceTurnState::Thinking | VoiceTurnState::Speaking)
            && band.0 >= PriorityBand::P2Nearline.0
            && matches!(domain, ComputeDomain::GPU | ComputeDomain::ANE)
    }

    fn take(&mut self, need: Need) {
        self.budget.slots[usize::from(need.class)] += need.slots;
        self.budget.ram_mb += need.ram_mb;
        if let Some(d) = self.vram_slot(need.device) {
            self.budget.vram_mb[d] += need.vram_mb;
        }
        *self.budget.domain.entry(need.domain).or_insert(0) += 1;
    }

    fn give_back(&mut self, lease: &Lease) {
        let class = usize::from(lease.class);
        self.budget.slots[class] = self.budget.slots[class].saturating_sub(lease.slots);
        if let Some(n) = self.budget.domain.get_mut(&lease.domain) {
            *n = n.saturating_sub(1);
        }
        if let Some(c) = self.caps.get_mut(&lease.capability) {
            c.active = c.active.saturating_sub(1);
            if c.active == 0 {
                self.budget.ram_mb = self.budget.ram_mb.saturating_sub(c.spec.ram_mb);
                let vram = c.spec.vram_mb;
                if let Some(d) = self.vram_slot(lease.device) {
                    self.budget.vram_mb[d] = self.budget.vram_mb[d].saturating_sub(vram);
                }
            }
        }
    }

    /// What preempting `lease` would give back toward `need`: slots of the class, a slot of the domain, and the
    /// capability's memory if this is its last active lease.
    fn frees(&self, lease: &Lease, need: Need) -> (u32, u32, u64, u64) {
        let slots = if lease.class == need.class { lease.slots } else { 0 };
        let domain = u32::from(lease.domain == need.domain);
        let (ram, vram) = match self.caps.get(&lease.capability) {
            Some(c) if c.active == 1 => {
                let same_device = self.vram_slot(lease.device) == self.vram_slot(need.device);
                (c.spec.ram_mb, if same_device { c.spec.vram_mb } else { 0 })
            }
            _ => (0, 0),
        };
        (slots, domain, ram, vram)
    }

    /// Preempts lower, preemptible leases, newest first, until `need` fits for `band`. Picks only leases that give
    /// back the resource that is short, and preempts nothing when the candidates together can't close the gap.
    fn make_room(&mut self, need: Need, band: PriorityBand, reason: &str, actions: &mut Vec<Action>) -> bool {
        let Some(short) = self.deny_reason(need, band) else { return true };
        if matches!(short, DenyReason::NoSuchDomain | DenyReason::VoiceTurn) {
            return false;
        }
        let class = usize::from(need.class);
        let mut slots_short =
            (self.budget.slots[class] + need.slots).saturating_sub(self.profile.slot_classes[class].count);
        let mut ram_short = (self.budget.ram_mb + need.ram_mb).saturating_sub(self.profile.ram_mb);
        let mut vram_short = match self.vram_slot(need.device) {
            Some(d) => (self.budget.vram_mb[d] + need.vram_mb).saturating_sub(self.profile.vram_mb[d]),
            None => 0,
        };
        let domain_cap = self.profile.domain_slots.get(&need.domain).copied().unwrap_or(0);
        let mut domain_short =
            (self.budget.domain.get(&need.domain).copied().unwrap_or(0) + 1).saturating_sub(domain_cap);

        let mut candidates: Vec<(LeaseId, u8)> = self
            .leases
            .iter()
            .filter(|(_, l)| l.state == LeaseState::Active && l.preemptible && l.band.0 > band.0)
            .map(|(id, l)| (*id, l.band.0))
            .collect();
        // Lowest band first, newest first within a band.
        candidates.sort_by(|a, b| b.1.cmp(&a.1).then(b.0.cmp(&a.0)));
        let mut chosen = Vec::new();
        for (id, _) in candidates {
            if slots_short == 0 && ram_short == 0 && vram_short == 0 && domain_short == 0 {
                break;
            }
            let (s, d, r, v) = self.frees(&self.leases[&id], need);
            let helps = (slots_short > 0 && s > 0)
                || (domain_short > 0 && d > 0)
                || (ram_short > 0 && r > 0)
                || (vram_short > 0 && v > 0);
            if !helps {
                continue;
            }
            slots_short = slots_short.saturating_sub(s);
            domain_short = domain_short.saturating_sub(d);
            ram_short = ram_short.saturating_sub(r);
            vram_short = vram_short.saturating_sub(v);
            chosen.push(id);
        }
        if slots_short > 0 || ram_short > 0 || vram_short > 0 || domain_short > 0 {
            return false;
        }
        for id in chosen {
            self.preempt(id, band, reason, actions);
        }
        self.deny_reason(need, band).is_none()
    }

    fn preempt(&mut self, id: LeaseId, by_band: PriorityBand, reason: &str, actions: &mut Vec<Action>) {
        let Some(lease) = self.leases.get_mut(&id) else { return };
        if lease.state != LeaseState::Active {
            return;
        }
        lease.state = LeaseState::Preempting;
        let worker = lease.worker;
        let job = lease.job.take();
        let snapshot = Lease { requires: Vec::new(), ..*lease };
        self.give_back(&snapshot);
        actions.push(Action::Preempt { worker, lease: id, by_band, reason: reason.to_owned() });
        if let Some(job) = job {
            self.job_interrupted(job);
        }
    }

    /// A job stopped before committing: its promises' steps go back to pending, to be dispatched again.
    fn job_interrupted(&mut self, job: JobId) {
        let Some(j) = self.jobs.remove(&job) else { return };
        if let Some(c) = self.costs.get_mut(&j.key) {
            c.queue_depth = c.queue_depth.saturating_sub(1);
        }
        for p in j.promises {
            if let Some(promise) = self.promises.get_mut(&p) {
                for s in &mut promise.steps {
                    if matches!(s.status, StepStatus::Running(id) if id == job) {
                        s.status = StepStatus::Pending;
                    }
                }
            }
        }
    }

    fn grant(
        &mut self,
        worker: WorkerId,
        cap: CapabilityId,
        band: PriorityBand,
        stream: StreamId,
        lease_ns: u64,
        now_ns: u64,
    ) -> Option<(LeaseId, OutputLease)> {
        let need = self.need_for(cap, worker)?;
        let c = self.caps.get_mut(&cap)?;
        c.active += 1;
        let preemptible = c.spec.preemptible;
        let domain = c.spec.domain;
        let role = c.spec.role;
        let requires = c.spec.requires.clone();
        self.take(need);
        let id = LeaseId(self.mint());
        let expires_ns = now_ns.saturating_add(lease_ns.max(self.settings.min_lease_ns));
        let output = OutputLease { lease: id, stream, class: need.class, slots: need.slots, expires_ns };
        self.leases.insert(
            id,
            Lease {
                worker,
                capability: cap,
                band,
                class: need.class,
                slots: need.slots,
                domain,
                device: need.device,
                expires_ns,
                job: None,
                preemptible,
                state: LeaseState::Active,
                role,
                requires,
            },
        );
        Some((id, output))
    }

    /// Takes a lease off the table for `end`: gives its memory back if it was still active, then tells the broker
    /// to unpin and to forget it. Every exit goes through here, so `LeaseEnded` is sent exactly once per lease.
    fn end_lease(&mut self, id: LeaseId, end: LeaseEnd, actions: &mut Vec<Action>) -> Option<Lease> {
        let lease = self.leases.remove(&id)?;
        if lease.state == LeaseState::Active {
            self.give_back(&lease);
        }
        actions.push(Action::Unpin { lease: id });
        actions.push(Action::LeaseEnded { lease: id, worker: lease.worker, end });
        Some(lease)
    }

    // ----- planning -------------------------------------------------------------------------------------------

    fn items(view: &StreamView, range: Interval) -> u64 {
        if view.rate_hz <= 0.0 {
            return 1;
        }
        ((range.len_ns() as f64) * view.rate_hz / (NS_PER_S as f64)).ceil().max(1.0) as u64
    }

    fn step_cost_us(&self, cap: CapabilityId, items: u64, strict: bool) -> (u64, f64) {
        let Some(c) = self.caps.get(&cap) else { return (0, 0.0) };
        let model = self.cost_of(&c.key, &c.spec);
        let loaded = self.workers.get(&c.worker).is_some_and(|w| w.loaded.contains(&cap));
        let per_item = if strict { model.p95_us } else { model.ema_us };
        let cold = if loaded { 0 } else { model.cold_start_ms.saturating_mul(1_000) };
        let per_item_us = per_item.ceil() as u64;
        let est = cold.saturating_add(
            u64::from(model.queue_depth).saturating_add(1).saturating_mul(items).saturating_mul(per_item_us),
        );
        let weight = self.profile.weights.per_domain_us.get(&c.spec.domain).copied().unwrap_or(0.0);
        let compute = (items as f64) * per_item * weight;
        let memory =
            ((c.spec.ram_mb + c.spec.vram_mb) as f64) * ((est as f64) / 1e6) * self.profile.weights.per_mb_second;
        (est, compute + memory)
    }

    /// A job in flight from the same producer family that makes all of `range` of `stream`, if any.
    fn in_flight_covering(&self, stream: StreamId, range: Interval, key: Option<&CapabilityKey>) -> Option<JobId> {
        self.jobs
            .iter()
            .filter(|(_, j)| {
                j.stream == stream && j.range.start_ns() <= range.start_ns() && j.range.end_ns() >= range.end_ns()
            })
            .filter(|(_, j)| key.is_none_or(|k| j.key == *k))
            .map(|(id, _)| *id)
            .next()
    }

    /// A job in flight that makes `stream` over some of `range`, from the same producer family when `key` is given.
    fn in_flight(&self, stream: StreamId, range: Interval, key: Option<&CapabilityKey>) -> Option<(JobId, Interval)> {
        self.jobs
            .iter()
            .filter(|(_, j)| j.stream == stream && j.range.overlaps(range))
            .filter(|(_, j)| key.is_none_or(|k| j.key == *k))
            .map(|(id, j)| (*id, clip(j.range, range).expect("overlaps")))
            .max_by_key(|(_, i)| i.len_ns())
    }

    #[allow(clippy::too_many_arguments)]
    fn ensure(
        &self,
        catalog: &dyn CatalogView,
        stream: StreamId,
        range: Interval,
        now_ns: u64,
        strict: bool,
        force: Option<CapabilityId>,
        depth: u32,
        steps: &mut Vec<PlanStep>,
    ) -> Result<(), Refusal> {
        if depth > self.settings.max_plan_depth {
            return Err(Refusal::NoProducer(stream));
        }
        let view = match catalog.stream(stream) {
            Ok(Some(v)) => v,
            Ok(None) => return Err(Refusal::UnknownStream(stream)),
            Err(e) => return Err(Refusal::Catalog(e.0)),
        };
        let covered = catalog.has_range(stream, range).map_err(|e| Refusal::Catalog(e.0))?;
        for c in &covered {
            if c.tier != StorageTier::Tier0Hot {
                let hydrator =
                    self.hydrators.iter().copied().find(|h| self.cap_usable(*h)).ok_or(Refusal::NoHydrator)?;
                let items = Self::items(&view, c.interval);
                let (est, _) = self.step_cost_us(hydrator, items, strict);
                steps.push(PlanStep {
                    kind: StepKind::Hydrate { from: c.tier },
                    stream,
                    range: c.interval,
                    capability: Some(hydrator),
                    worker: self.caps.get(&hydrator).map(|h| h.worker),
                    items,
                    est_us: est,
                    shared: false,
                });
            }
        }
        let taken: Vec<Interval> = covered.iter().map(|c| c.interval).collect();
        let missing = subtract(range, &taken);
        match view.kind {
            StreamKind::Sensor => {
                if missing.is_empty() {
                    return Ok(());
                }
                if !view.live {
                    return Err(Refusal::StreamNotLive(stream));
                }
                // Up to now the bytes can still be captured; before it they are gone for good.
                let (capturable, gone) = split_capturable(&missing, now_ns);
                if !gone.is_empty() {
                    return Err(Refusal::NotCaptured(stream));
                }
                for part in capturable {
                    steps.push(PlanStep {
                        kind: StepKind::Capture,
                        stream,
                        range: part,
                        capability: None,
                        worker: None,
                        items: Self::items(&view, part),
                        est_us: part.end_ns().saturating_sub(now_ns) / NS_PER_US,
                        shared: false,
                    });
                }
            }
            StreamKind::Derived => {
                for part in missing {
                    let mut rest = vec![part];
                    if let Some((job, shared)) = self.in_flight(stream, part, None) {
                        let j = &self.jobs[&job];
                        steps.push(PlanStep {
                            kind: StepKind::Compute,
                            stream,
                            range: shared,
                            capability: Some(j.capability),
                            worker: Some(j.worker),
                            items: Self::items(&view, shared),
                            est_us: j.expected_end_ns.saturating_sub(now_ns) / NS_PER_US,
                            shared: true,
                        });
                        rest = subtract(part, &[shared]);
                    }
                    for r in rest {
                        let cap = match force {
                            Some(f) if depth == 0 => f,
                            _ => self.best_producer(stream, &view, r, strict).ok_or(Refusal::NoProducer(stream))?,
                        };
                        let spec = self.caps[&cap].spec.clone();
                        for req in &spec.requires {
                            let widened = Interval::new(
                                r.start_ns().saturating_sub(req.before_ns),
                                r.end_ns().saturating_add(req.after_ns),
                            )
                            .expect("widening keeps start < end");
                            self.ensure(catalog, req.stream, widened, now_ns, strict, None, depth + 1, steps)?;
                        }
                        let items = Self::items(&view, r);
                        let (est, _) = self.step_cost_us(cap, items, strict);
                        steps.push(PlanStep {
                            kind: StepKind::Compute,
                            stream,
                            range: r,
                            capability: Some(cap),
                            worker: Some(self.caps[&cap].worker),
                            items,
                            est_us: est,
                            shared: false,
                        });
                    }
                }
            }
        }
        Ok(())
    }

    fn producers_of(&self, stream: StreamId) -> Vec<CapabilityId> {
        self.producers
            .get(&stream)
            .map(|v| v.iter().copied().filter(|c| self.cap_usable(*c)).collect())
            .unwrap_or_default()
    }

    fn best_producer(
        &self,
        stream: StreamId,
        view: &StreamView,
        range: Interval,
        strict: bool,
    ) -> Option<CapabilityId> {
        let items = Self::items(view, range);
        self.producers_of(stream).into_iter().min_by_key(|c| self.step_cost_us(*c, items, strict).0)
    }

    fn price(&self, steps: &[PlanStep], strict: bool) -> (u64, f64) {
        let mut latency = 0u64;
        let mut units = 0.0;
        for s in steps {
            latency = latency.saturating_add(s.est_us);
            if s.shared {
                continue;
            }
            if let Some(cap) = s.capability {
                units += self.step_cost_us(cap, s.items, strict).1;
            }
        }
        (latency, units)
    }

    fn plan(
        &self,
        catalog: &dyn CatalogView,
        req: &QueryReq,
        missing: &[Interval],
        now_ns: u64,
    ) -> Result<(Plan, Vec<Alternative>), Refusal> {
        let mut steps = Vec::new();
        for m in missing {
            self.ensure(catalog, req.stream, *m, now_ns, req.strict, None, 0, &mut steps)?;
        }
        let chosen: BTreeSet<CapabilityId> =
            steps.iter().filter(|s| s.stream == req.stream && !s.shared).filter_map(|s| s.capability).collect();
        let mut alternatives = Vec::new();
        let derived = matches!(catalog.stream(req.stream), Ok(Some(v)) if v.kind == StreamKind::Derived);
        if derived {
            for cap in self.producers_of(req.stream) {
                if chosen.contains(&cap) {
                    continue;
                }
                let mut alt = Vec::new();
                let ok = missing
                    .iter()
                    .all(|m| self.ensure(catalog, req.stream, *m, now_ns, req.strict, Some(cap), 0, &mut alt).is_ok());
                if !ok {
                    continue;
                }
                let (est_latency_us, cost_units) = self.price(&alt, req.strict);
                alternatives.push(Alternative {
                    plan: Plan { steps: alt },
                    est_latency_us,
                    cost_units,
                    exceeds_budget: req.budget_us.is_some_and(|b| est_latency_us > b),
                    expected_confidence: self.caps.get(&cap).and_then(|c| c.spec.expected_confidence),
                });
            }
        }
        Ok((Plan { steps }, alternatives))
    }

    // ----- promises -------------------------------------------------------------------------------------------

    fn end_promise(&mut self, id: PromiseId, state: PromiseState, now_ns: u64) {
        let Some(p) = self.promises.get_mut(&id) else { return };
        p.state = state;
        p.ended_ns = Some(now_ns);
        for s in &mut p.steps {
            s.status = StepStatus::Done;
        }
        let armed_on: Vec<StreamId> = self.armed.iter().filter(|(_, v)| v.contains(&id)).map(|(s, _)| *s).collect();
        for s in armed_on {
            if let Some(list) = self.armed.get_mut(&s) {
                list.retain(|x| *x != id);
                if list.is_empty() {
                    self.armed.remove(&s);
                }
            }
        }
    }

    fn fail_promise(&mut self, id: PromiseId, reason: &str, now_ns: u64, actions: &mut Vec<Action>) {
        if let Some(p) = self.promises.get(&id) {
            if p.ended_ns.is_some() {
                return;
            }
            actions.push(Action::FailPromise { promise: id, consumer: p.consumer, reason: reason.to_owned() });
        }
        self.detach_jobs(id, actions);
        self.end_promise(id, PromiseState::Failed, now_ns);
    }

    /// Takes a promise off every job it shares; a job left with no promise is revoked.
    fn detach_jobs(&mut self, promise: PromiseId, actions: &mut Vec<Action>) {
        let Some(p) = self.promises.get(&promise) else { return };
        let running: Vec<JobId> =
            p.steps.iter().filter_map(|s| if let StepStatus::Running(j) = s.status { Some(j) } else { None }).collect();
        for job in running {
            let Some(j) = self.jobs.get_mut(&job) else { continue };
            j.promises.retain(|x| *x != promise);
            if j.promises.is_empty() {
                let lease = j.lease;
                let worker = j.worker;
                let key = j.key.clone();
                self.jobs.remove(&job);
                if let Some(m) = self.costs.get_mut(&key) {
                    m.queue_depth = m.queue_depth.saturating_sub(1);
                }
                self.end_lease(lease, LeaseEnd::PromiseEnded, actions);
                actions.push(Action::RevokeLease {
                    worker: Some(worker),
                    lease,
                    reason: "its only promise ended".to_owned(),
                });
            }
        }
    }

    /// Dispatches every promise step that can run now. Bounded by the number of live promises.
    fn pump(&mut self, now_ns: u64, actions: &mut Vec<Action>) {
        let ids: Vec<PromiseId> = self
            .promises
            .iter()
            .filter(|(_, p)| p.ended_ns.is_none() && p.state != PromiseState::Quoted)
            .map(|(id, _)| *id)
            .collect();
        for id in ids {
            self.advance(id, now_ns, actions);
        }
    }

    fn advance(&mut self, id: PromiseId, now_ns: u64, actions: &mut Vec<Action>) {
        let Some(p) = self.promises.get(&id) else { return };
        if p.ended_ns.is_some() {
            return;
        }
        let Some(idx) = p.steps.iter().position(|s| !matches!(s.status, StepStatus::Done)) else {
            self.end_promise(id, PromiseState::Fulfilled, now_ns);
            return;
        };
        let step = p.steps[idx].step.clone();
        let band = p.band;
        let strict = p.strict;
        let status = p.steps[idx].status.clone();
        match (step.kind, status) {
            (_, StepStatus::Running(_) | StepStatus::Done) => {}
            (StepKind::Capture, _) => {
                let p = self.promises.get_mut(&id).expect("present");
                p.state = PromiseState::Armed;
                p.steps[idx].status = StepStatus::Waiting;
                let list = self.armed.entry(step.stream).or_default();
                if !list.contains(&id) {
                    list.push(id);
                }
            }
            (_, StepStatus::Pending | StepStatus::Waiting) => {
                if step.shared {
                    // Attach to a job of the same producer family that still covers the whole range; otherwise
                    // compute it ourselves.
                    let key = step.capability.and_then(|c| self.caps.get(&c)).map(|c| c.key.clone());
                    match self.in_flight_covering(step.stream, step.range, key.as_ref()) {
                        Some(j) => {
                            self.jobs.get_mut(&j).expect("present").promises.push(id);
                            let p = self.promises.get_mut(&id).expect("present");
                            p.state = PromiseState::Dispatching;
                            p.steps[idx].status = StepStatus::Running(j);
                        }
                        None => {
                            let (est, _) =
                                step.capability.map_or((0, 0.0), |c| self.step_cost_us(c, step.items, strict));
                            let p = self.promises.get_mut(&id).expect("present");
                            p.steps[idx].step.shared = false;
                            p.steps[idx].step.est_us = est;
                            self.advance(id, now_ns, actions);
                        }
                    }
                    return;
                }
                let Some(cap) = step.capability.filter(|c| self.cap_usable(*c)) else {
                    self.replan_or_fail(id, "producer is gone", now_ns, actions);
                    return;
                };
                let worker = self.caps[&cap].worker;
                let Some(need) = self.need_for(cap, worker) else {
                    self.replan_or_fail(id, "no size class holds the output", now_ns, actions);
                    return;
                };
                if !self.make_room(need, band, "a higher band needs the resources", actions) {
                    // Queued: tried again on the next release, commit, or tick, until the promise's deadline.
                    let p = self.promises.get_mut(&id).expect("present");
                    p.state = PromiseState::Dispatching;
                    p.steps[idx].status = StepStatus::Pending;
                    return;
                }
                let est_ns = step.est_us.saturating_mul(NS_PER_US);
                let deadline_ns = self.promises[&id].deadline_ns;
                let lease_ns = est_ns.saturating_mul(3).max(deadline_ns.saturating_sub(now_ns));
                let Some((lease, output)) = self.grant(worker, cap, band, step.stream, lease_ns, now_ns) else {
                    self.replan_or_fail(id, "lease could not be granted", now_ns, actions);
                    return;
                };
                let job = JobId(self.mint());
                let key = self.caps[&cap].key.clone();
                let spec = self.caps[&cap].spec.clone();
                let m = self.costs.entry(key.clone()).or_insert_with(|| CostModel::from_capability(&spec));
                m.queue_depth += 1;
                self.leases.get_mut(&lease).expect("present").job = Some(job);
                let inputs = self.inputs_for(cap, step.stream, step.range);
                self.jobs.insert(
                    job,
                    Job {
                        lease,
                        worker,
                        capability: cap,
                        key,
                        promises: vec![id],
                        stream: step.stream,
                        range: step.range,
                        expected_end_ns: now_ns.saturating_add(est_ns),
                    },
                );
                let p = self.promises.get_mut(&id).expect("present");
                p.state = PromiseState::Dispatching;
                p.steps[idx].status = StepStatus::Running(job);
                actions.push(Action::Dispatch(Dispatch {
                    job,
                    worker,
                    capability: cap,
                    promises: vec![id],
                    inputs,
                    range: step.range,
                    output,
                    band,
                    deadline_ns,
                }));
            }
        }
    }

    fn inputs_for(&self, cap: CapabilityId, stream: StreamId, range: Interval) -> Vec<InputRange> {
        let c = &self.caps[&cap];
        if c.spec.role == CapabilityRole::Hydrate {
            return vec![InputRange { stream, range }];
        }
        c.spec
            .requires
            .iter()
            .map(|r| InputRange {
                stream: r.stream,
                range: Interval::new(
                    range.start_ns().saturating_sub(r.before_ns),
                    range.end_ns().saturating_add(r.after_ns),
                )
                .expect("start < end"),
            })
            .collect()
    }

    /// Replans a promise's unfinished steps onto other producers (never the one that just failed, unless it is the
    /// only one), or fails the promise once the replans are used up.
    fn replan_or_fail(&mut self, id: PromiseId, reason: &str, now_ns: u64, actions: &mut Vec<Action>) {
        let Some(p) = self.promises.get_mut(&id) else { return };
        if p.ended_ns.is_some() {
            return;
        }
        if p.replans >= self.settings.max_replans {
            self.fail_promise(id, reason, now_ns, actions);
            return;
        }
        p.replans += 1;
        let strict = p.strict;
        let mut pending: Vec<(usize, StepKind, StreamId, Option<CapabilityId>, u64)> = Vec::new();
        for (i, s) in p.steps.iter_mut().enumerate() {
            if matches!(s.status, StepStatus::Done | StepStatus::Running(_)) || s.step.kind == StepKind::Capture {
                continue;
            }
            s.status = StepStatus::Pending;
            s.step.shared = false;
            pending.push((i, s.step.kind, s.step.stream, s.step.capability, s.step.items));
        }
        for (i, kind, step_stream, current, items) in pending {
            let replacement = match kind {
                StepKind::Hydrate { .. } => self.hydrators.iter().copied().find(|h| self.cap_usable(*h)),
                StepKind::Compute => {
                    let others: Vec<CapabilityId> =
                        self.producers_of(step_stream).into_iter().filter(|c| Some(*c) != current).collect();
                    let pool = if others.is_empty() { self.producers_of(step_stream) } else { others };
                    pool.into_iter().min_by_key(|c| self.step_cost_us(*c, items, strict).0)
                }
                StepKind::Capture => None,
            };
            match replacement {
                Some(cap) => {
                    let (est, _) = self.step_cost_us(cap, items, strict);
                    let worker = self.caps[&cap].worker;
                    let p = self.promises.get_mut(&id).expect("present");
                    p.steps[i].step.capability = Some(cap);
                    p.steps[i].step.worker = Some(worker);
                    p.steps[i].step.est_us = est;
                }
                None => {
                    self.fail_promise(id, reason, now_ns, actions);
                    return;
                }
            }
        }
        self.advance(id, now_ns, actions);
    }

    /// A job ended without committing: every promise it served loses that step and is replanned or failed.
    fn job_failed(&mut self, job: JobId, reason: &str, now_ns: u64, actions: &mut Vec<Action>) {
        let Some(j) = self.jobs.remove(&job) else { return };
        if let Some(c) = self.costs.get_mut(&j.key) {
            c.queue_depth = c.queue_depth.saturating_sub(1);
        }
        for p in j.promises {
            if let Some(promise) = self.promises.get_mut(&p) {
                for s in &mut promise.steps {
                    if matches!(s.status, StepStatus::Running(id) if id == job) {
                        s.status = StepStatus::Pending;
                    }
                }
            }
            self.replan_or_fail(p, reason, now_ns, actions);
        }
    }

    fn deliver(&mut self, id: PromiseId, outputs: &[OutputRef], now_ns: u64, actions: &mut Vec<Action>) {
        let Some(p) = self.promises.get_mut(&id) else { return };
        if p.ended_ns.is_some() {
            return;
        }
        let mine: Vec<OutputRef> = outputs
            .iter()
            .copied()
            .filter(|o| o.stream == p.stream && p.missing.iter().any(|m| m.overlaps(o.range)))
            .collect();
        let all_done = p.steps.iter().all(|s| matches!(s.status, StepStatus::Done));
        if mine.is_empty() && !all_done {
            return;
        }
        let consumer = p.consumer;
        actions.push(Action::Fulfill { promise: id, consumer, outputs: mine, final_part: all_done });
        if all_done {
            self.end_promise(id, PromiseState::Fulfilled, now_ns);
        }
    }
}

impl Scheduler for Engine {
    fn register_worker(&mut self, worker: WorkerId, manifest: Manifest, _now_ns: u64) -> Vec<Accepted> {
        // A worker that registers again replaces its capabilities (its leases, if any, keep their old ids).
        self.forget_worker_caps(worker);
        let mut out = Vec::with_capacity(manifest.capabilities.len());
        let mut caps = Vec::with_capacity(manifest.capabilities.len());
        for spec in manifest.capabilities {
            let id = CapabilityId(u32::try_from(self.mint()).unwrap_or(u32::MAX));
            let runnable = self.runnable_here(spec.domain);
            let key = CapabilityKey { producer: manifest.producer.clone(), output: spec.produces };
            self.costs.entry(key.clone()).or_insert_with(|| CostModel::from_capability(&spec));
            match spec.role {
                CapabilityRole::Produce => self.producers.entry(spec.produces).or_default().push(id),
                CapabilityRole::Hydrate => self.hydrators.push(id),
            }
            self.caps.insert(id, Cap { worker, spec, key, runnable, active: 0 });
            caps.push(id);
            out.push(Accepted { capability: id, runnable });
        }
        self.workers.insert(worker, Worker { device_index: manifest.device_index, caps, loaded: BTreeSet::new() });
        out
    }

    fn worker_gone(&mut self, _catalog: &dyn CatalogView, worker: WorkerId, now_ns: u64) -> Vec<Action> {
        let mut actions = Vec::new();
        if !self.workers.contains_key(&worker) {
            return actions;
        }
        // End the leases while the capabilities still exist (so their memory is given back), then forget the
        // worker, then replan the jobs, so a replan can't land on the worker that just left.
        let leases: Vec<LeaseId> = self.leases.iter().filter(|(_, l)| l.worker == worker).map(|(id, _)| *id).collect();
        let mut failed_jobs = Vec::new();
        for id in leases {
            let Some(l) = self.end_lease(id, LeaseEnd::WorkerGone, &mut actions) else { continue };
            actions.push(Action::RevokeLease { worker: None, lease: id, reason: "worker gone".to_owned() });
            if let Some(job) = l.job {
                failed_jobs.push(job);
            }
        }
        self.forget_worker_caps(worker);
        for job in failed_jobs {
            self.job_failed(job, "worker gone", now_ns, &mut actions);
        }
        self.pump(now_ns, &mut actions);
        actions
    }

    fn consumer_gone(&mut self, consumer: ConsumerId, now_ns: u64) -> Vec<Action> {
        let mut actions = Vec::new();
        for id in self.promises_of(consumer) {
            if let Ok(mut a) = self.cancel_promise(id, now_ns) {
                actions.append(&mut a);
            }
        }
        actions.retain(|a| {
            !matches!(a, Action::Fulfill { consumer: c, .. } | Action::FailPromise { consumer: c, .. } if *c == consumer)
        });
        actions
    }

    /// Only `loaded` is used: whether a model is resident decides the cold start in quotes. The reported queue depth
    /// is ignored, since the engine counts dispatches and commits itself.
    fn telemetry(&mut self, worker: WorkerId, telemetry: Telemetry, _now_ns: u64) {
        if let Some(w) = self.workers.get_mut(&worker) {
            w.loaded = telemetry.loaded.iter().copied().collect();
        }
    }

    fn query(&mut self, catalog: &dyn CatalogView, req: QueryReq, now_ns: u64) -> Outcome<QueryDecision> {
        let refused = |r: Refusal| Outcome { decision: QueryDecision::Refused(r), actions: Vec::new() };
        let view = match catalog.stream(req.stream) {
            Ok(Some(v)) => v,
            Ok(None) => return refused(Refusal::UnknownStream(req.stream)),
            Err(e) => return refused(Refusal::Catalog(e.0)),
        };
        let covered = match catalog.has_range(req.stream, req.range) {
            Ok(c) => c,
            Err(e) => return refused(Refusal::Catalog(e.0)),
        };
        let taken: Vec<Interval> = covered.iter().map(|c| c.interval).collect();
        let missing = subtract(req.range, &taken);
        if missing.is_empty() || req.mode == QueryMode::HitOnly {
            return Outcome { decision: QueryDecision::Hit { covered, missing }, actions: Vec::new() };
        }
        // A sensor stream's past can't be made; only what is still ahead of the live edge can be quoted.
        let (quotable, unavailable) = if view.kind == StreamKind::Sensor {
            if !view.live {
                return refused(Refusal::StreamNotLive(req.stream));
            }
            split_capturable(&missing, now_ns)
        } else {
            (missing.clone(), Vec::new())
        };
        if quotable.is_empty() {
            return Outcome { decision: QueryDecision::Hit { covered, missing }, actions: Vec::new() };
        }
        let (plan, alternatives) = match self.plan(catalog, &req, &quotable, now_ns) {
            Ok(p) => p,
            Err(r) => return refused(r),
        };
        let (est_latency_us, cost_units) = self.price(&plan.steps, req.strict);
        let has_capture = plan.steps.iter().any(|s| s.kind == StepKind::Capture);
        let has_compute = plan.steps.iter().any(|s| s.kind != StepKind::Capture);
        let status = match (has_capture, has_compute) {
            (true, true) => QuoteStatus::AwaitingCaptureAndCompute,
            (true, false) => QuoteStatus::AwaitingCapture,
            _ => QuoteStatus::RequiresCompute,
        };
        let exceeds_budget = req.budget_us.is_some_and(|b| est_latency_us > b);
        let promise = PromiseId(self.mint());
        let expires_ns = now_ns.saturating_add(self.settings.quote_ttl_ns);
        let quote = Quote {
            promise,
            status,
            missing: quotable.clone(),
            unavailable,
            est_latency_us,
            cost_units,
            exceeds_budget,
            plan: plan.clone(),
            alternatives: alternatives.clone(),
            shares_in_flight: plan.steps.iter().any(|s| s.shared),
            expires_ns,
        };
        self.promises.insert(
            promise,
            Promise {
                consumer: req.consumer,
                stream: req.stream,
                band: req.band,
                strict: req.strict,
                state: PromiseState::Quoted,
                missing: quotable,
                quote_plan: plan,
                alternatives,
                est_latency_us,
                expires_ns,
                deadline_ns: u64::MAX,
                steps: Vec::new(),
                replans: 0,
                ended_ns: None,
            },
        );
        if req.mode == QueryMode::CommitWithinBudget && !exceeds_budget {
            let actions = self.commit_promise(catalog, promise, None, req.band, now_ns).unwrap_or_default();
            return Outcome { decision: QueryDecision::Committed { covered, quote }, actions };
        }
        Outcome { decision: QueryDecision::Quoted { covered, quote }, actions: Vec::new() }
    }

    fn commit_promise(
        &mut self,
        _catalog: &dyn CatalogView,
        promise: PromiseId,
        alternative: Option<u32>,
        band: PriorityBand,
        now_ns: u64,
    ) -> Result<Vec<Action>, Error> {
        let grace = self.settings.promise_grace_ns;
        let p = self.promises.get_mut(&promise).ok_or(Error::UnknownPromise(promise))?;
        if p.state != PromiseState::Quoted {
            return Err(Error::PromiseState { promise, state: p.state });
        }
        if now_ns > p.expires_ns {
            p.state = PromiseState::Expired;
            p.ended_ns = Some(now_ns);
            return Err(Error::QuoteExpired(promise));
        }
        let (plan, est_latency_us) = match alternative {
            None => (p.quote_plan.clone(), p.est_latency_us),
            Some(i) => p
                .alternatives
                .get(i as usize)
                .map(|a| (a.plan.clone(), a.est_latency_us))
                .ok_or(Error::UnknownAlternative { promise, alternative: i })?,
        };
        p.band = band;
        p.est_latency_us = est_latency_us;
        let capture_end =
            plan.steps.iter().filter(|s| s.kind == StepKind::Capture).map(|s| s.range.end_ns()).max().unwrap_or(now_ns);
        p.deadline_ns = capture_end
            .max(now_ns.saturating_add(est_latency_us.saturating_mul(NS_PER_US).saturating_mul(3)))
            .saturating_add(grace);
        p.steps = plan
            .steps
            .into_iter()
            .map(|step| StepState { step, status: StepStatus::Pending, captured: Vec::new() })
            .collect();
        p.state = PromiseState::Dispatching;
        let mut actions = Vec::new();
        self.advance(promise, now_ns, &mut actions);
        Ok(actions)
    }

    fn cancel_promise(&mut self, promise: PromiseId, now_ns: u64) -> Result<Vec<Action>, Error> {
        let p = self.promises.get(&promise).ok_or(Error::UnknownPromise(promise))?;
        if p.ended_ns.is_some() {
            return Err(Error::PromiseState { promise, state: p.state });
        }
        let mut actions = Vec::new();
        self.detach_jobs(promise, &mut actions);
        self.end_promise(promise, PromiseState::Cancelled, now_ns);
        self.pump(now_ns, &mut actions);
        Ok(actions)
    }

    fn on_ingest(&mut self, _catalog: &dyn CatalogView, ingest: IngestRef, now_ns: u64) -> Vec<Action> {
        let Some(list) = self.armed.get(&ingest.stream) else { return Vec::new() };
        let ids: Vec<PromiseId> = list.clone();
        let mut actions = Vec::new();
        for id in ids {
            let Some(p) = self.promises.get_mut(&id) else { continue };
            if !p.steps.iter().any(|s| s.step.kind == StepKind::Capture && s.step.range.overlaps(ingest.range)) {
                continue;
            }
            let mut completed = false;
            for s in &mut p.steps {
                if s.step.kind == StepKind::Capture
                    && s.step.stream == ingest.stream
                    && matches!(s.status, StepStatus::Waiting | StepStatus::Pending)
                    && s.step.range.overlaps(ingest.range)
                {
                    let got = clip(ingest.range, s.step.range).expect("overlaps");
                    s.captured.push(got);
                    s.captured = merge(std::mem::take(&mut s.captured));
                    if subtract(s.step.range, &s.captured).is_empty() {
                        s.status = StepStatus::Done;
                        completed = true;
                    }
                }
            }
            let capture_only = p.steps.iter().all(|s| s.step.kind == StepKind::Capture);
            if capture_only {
                let out = OutputRef { id: ingest.id, stream: ingest.stream, range: ingest.range };
                let all_done = p.steps.iter().all(|s| matches!(s.status, StepStatus::Done));
                let consumer = p.consumer;
                actions.push(Action::Fulfill { promise: id, consumer, outputs: vec![out], final_part: all_done });
                if all_done {
                    self.end_promise(id, PromiseState::Fulfilled, now_ns);
                }
            } else if completed {
                if let Some(list) = self.armed.get_mut(&ingest.stream) {
                    list.retain(|x| *x != id);
                }
                self.advance(id, now_ns, &mut actions);
            }
        }
        if self.armed.get(&ingest.stream).is_some_and(Vec::is_empty) {
            self.armed.remove(&ingest.stream);
        }
        actions
    }

    fn quote_request(&mut self, worker: WorkerId, req: LeaseReq, now_ns: u64) -> Result<Outcome<LeaseDecision>, Error> {
        if !self.workers.contains_key(&worker) {
            return Err(Error::UnknownWorker(worker));
        }
        let cap = req.capability;
        let usable = self.caps.get(&cap).is_some_and(|c| c.worker == worker && c.runnable);
        if !usable {
            return Err(Error::UnknownCapability { worker, capability: cap });
        }
        let retry = self.settings.retry_after_us;
        let denied = |reason: DenyReason, actions: Vec<Action>| {
            Ok(Outcome { decision: LeaseDecision::Denied { reason, retry_after_us: retry }, actions })
        };
        let Some(mut need) = self.need_for(cap, worker) else {
            return denied(DenyReason::Slots, Vec::new());
        };
        if let Some(class) = self.class_for(req.output_bytes) {
            need.class = class;
        }
        let mut actions = Vec::new();
        if !self.make_room(need, req.band, "a higher band needs the resources", &mut actions) {
            let reason = self.deny_reason(need, req.band).unwrap_or(DenyReason::Compute);
            return denied(reason, actions);
        }
        let stream = self.caps[&cap].spec.produces;
        let lease_ns = req.est_duration_us.saturating_mul(NS_PER_US).saturating_mul(3);
        let Some((_, output)) = self.grant(worker, cap, req.band, stream, lease_ns, now_ns) else {
            return denied(DenyReason::Slots, actions);
        };
        Ok(Outcome { decision: LeaseDecision::Granted(output), actions })
    }

    fn renew_lease(&mut self, lease: LeaseId, extend_ns: u64, now_ns: u64) -> Result<u64, Error> {
        let asked = if extend_ns == 0 { self.settings.renew_ns } else { extend_ns };
        // The default is clamped too, so a tuned `renew_ns` can't pass the cap.
        let extend =
            asked.clamp(self.settings.min_lease_ns, self.settings.max_lease_ns.max(self.settings.min_lease_ns));
        let l = self.leases.get_mut(&lease).ok_or(Error::UnknownLease(lease))?;
        // A preempting lease is on its way out, so it's never pushed later.
        if l.state == LeaseState::Active {
            l.expires_ns = now_ns.saturating_add(extend).max(l.expires_ns);
        }
        Ok(l.expires_ns)
    }

    fn commit_lease(
        &mut self,
        lease: LeaseId,
        outputs: &[OutputRef],
        _coverage: Option<CoverageDone>,
        cost: ComputeCost,
        now_ns: u64,
    ) -> Result<Vec<Action>, Error> {
        let mut actions = Vec::new();
        let l = self.end_lease(lease, LeaseEnd::Committed, &mut actions).ok_or(Error::UnknownLease(lease))?;
        // A preempted lease's job was already taken off the books; its late outputs serve nobody.
        let job = l.job.and_then(|j| self.jobs.remove(&j).map(|job| (j, job)));
        if let Some(c) = self.caps.get(&l.capability) {
            let key = c.key.clone();
            let spec = c.spec.clone();
            let per_item = (cost.wall_time_us as f64) / (cost.items.max(1) as f64);
            let settings = self.settings.clone();
            let m = self.costs.entry(key).or_insert_with(|| CostModel::from_capability(&spec));
            // A commit that reports no wall time measured nothing; learning from it would price this capability's
            // work at zero from then on.
            if cost.wall_time_us > 0 {
                m.observe(per_item, &settings);
            }
            if job.is_some() {
                m.queue_depth = m.queue_depth.saturating_sub(1);
            }
        }
        if let Some((job_id, j)) = job {
            for p in j.promises {
                if let Some(promise) = self.promises.get_mut(&p) {
                    for s in &mut promise.steps {
                        if matches!(s.status, StepStatus::Running(id) if id == job_id) {
                            s.status = StepStatus::Done;
                        }
                    }
                }
                self.deliver(p, outputs, now_ns, &mut actions);
                self.advance(p, now_ns, &mut actions);
            }
        }
        self.pump(now_ns, &mut actions);
        Ok(actions)
    }

    fn abort_lease(
        &mut self,
        _catalog: &dyn CatalogView,
        lease: LeaseId,
        reason: &str,
        now_ns: u64,
    ) -> Result<Vec<Action>, Error> {
        let mut actions = Vec::new();
        let l = self.end_lease(lease, LeaseEnd::Aborted, &mut actions).ok_or(Error::UnknownLease(lease))?;
        if let Some(job) = l.job {
            self.job_failed(job, reason, now_ns, &mut actions);
        }
        self.pump(now_ns, &mut actions);
        Ok(actions)
    }

    fn release_lease(&mut self, lease: LeaseId, now_ns: u64) -> Result<Vec<Action>, Error> {
        let mut actions = Vec::new();
        let l = self.end_lease(lease, LeaseEnd::Released, &mut actions).ok_or(Error::UnknownLease(lease))?;
        if let Some(job) = l.job {
            self.job_interrupted(job);
        }
        self.pump(now_ns, &mut actions);
        Ok(actions)
    }

    fn job_rejected(
        &mut self,
        _catalog: &dyn CatalogView,
        job: JobId,
        reason: &str,
        now_ns: u64,
    ) -> Result<Vec<Action>, Error> {
        let lease = self.jobs.get(&job).map(|j| j.lease).ok_or(Error::UnknownJob(job))?;
        let mut actions = Vec::new();
        self.end_lease(lease, LeaseEnd::Rejected, &mut actions);
        self.job_failed(job, reason, now_ns, &mut actions);
        self.pump(now_ns, &mut actions);
        Ok(actions)
    }

    fn dispatch_failed(
        &mut self,
        catalog: &dyn CatalogView,
        job: JobId,
        reason: &str,
        now_ns: u64,
    ) -> Result<Vec<Action>, Error> {
        self.job_rejected(catalog, job, reason, now_ns)
    }

    fn claim_failed(&mut self, _catalog: &dyn CatalogView, lease: LeaseId, now_ns: u64) -> Result<Vec<Action>, Error> {
        let mut actions = Vec::new();
        let l = self.end_lease(lease, LeaseEnd::ClaimFailed, &mut actions).ok_or(Error::UnknownLease(lease))?;
        if let Some(job) = l.job {
            self.job_failed(job, "slots could not be claimed", now_ns, &mut actions);
        }
        self.pump(now_ns, &mut actions);
        Ok(actions)
    }

    fn set_system_state(&mut self, state: SystemState, now_ns: u64) -> Vec<Action> {
        let SystemState::VoiceTurn(v) = state;
        let was_active =
            matches!(self.voice, VoiceTurnState::Listening | VoiceTurnState::Thinking | VoiceTurnState::Speaking);
        self.voice = v;
        let active = matches!(v, VoiceTurnState::Listening | VoiceTurnState::Thinking | VoiceTurnState::Speaking);
        let mut actions = Vec::new();
        if active {
            let victims: Vec<LeaseId> = self
                .leases
                .iter()
                .filter(|(_, l)| {
                    l.state == LeaseState::Active
                        && l.preemptible
                        && l.band.0 >= PriorityBand::P2Nearline.0
                        && matches!(l.domain, ComputeDomain::GPU | ComputeDomain::ANE)
                })
                .map(|(id, _)| *id)
                .collect();
            for id in victims {
                self.preempt(id, PriorityBand::P0Critical, "voice turn", &mut actions);
            }
        } else if was_active {
            self.pump(now_ns, &mut actions);
        }
        actions
    }

    fn tick(&mut self, now_ns: u64) -> Vec<Action> {
        let mut actions = Vec::new();
        let expired: Vec<LeaseId> =
            self.leases.iter().filter(|(_, l)| now_ns > l.expires_ns).map(|(id, _)| *id).collect();
        for id in expired {
            let Some(l) = self.leases.get(&id) else { continue };
            let worker = Some(l.worker);
            let was_active = l.state == LeaseState::Active;
            let Some(l) = self.end_lease(id, LeaseEnd::Expired, &mut actions) else { continue };
            if was_active {
                actions.push(Action::RevokeLease { worker, lease: id, reason: "lease expired".to_owned() });
            }
            if let Some(job) = l.job {
                self.job_failed(job, "lease expired", now_ns, &mut actions);
            }
        }
        let stale_quotes: Vec<PromiseId> = self
            .promises
            .iter()
            .filter(|(_, p)| p.state == PromiseState::Quoted && now_ns > p.expires_ns)
            .map(|(id, _)| *id)
            .collect();
        for id in stale_quotes {
            self.end_promise(id, PromiseState::Expired, now_ns);
        }
        let overdue: Vec<PromiseId> = self
            .promises
            .iter()
            .filter(|(_, p)| p.ended_ns.is_none() && p.state != PromiseState::Quoted && now_ns > p.deadline_ns)
            .map(|(id, _)| *id)
            .collect();
        for id in overdue {
            self.fail_promise(id, "timed out", now_ns, &mut actions);
        }
        let retention = self.settings.ended_retention_ns;
        self.promises.retain(|_, p| p.ended_ns.is_none_or(|t| now_ns.saturating_sub(t) < retention));
        self.pump(now_ns, &mut actions);
        actions
    }

    fn cost_models(&self) -> Vec<CostModelRecord> {
        self.costs
            .iter()
            .map(|(key, m)| CostModelRecord {
                key: key.clone(),
                ema_us_per_item: m.ema_us,
                p95_us_per_item: m.p95_us,
                samples: m.samples,
                ram_mb: m.ram_mb,
                vram_mb: m.vram_mb,
                cold_start_ms: m.cold_start_ms,
            })
            .collect()
    }

    fn restore(&mut self, records: &[CostModelRecord]) {
        for r in records {
            let m = self.costs.entry(r.key.clone()).or_insert_with(|| CostModel {
                ema_us: r.ema_us_per_item,
                p95_us: r.p95_us_per_item,
                samples: 0,
                window: VecDeque::new(),
                ram_mb: r.ram_mb,
                vram_mb: r.vram_mb,
                cold_start_ms: r.cold_start_ms,
                queue_depth: 0,
            });
            m.ema_us = r.ema_us_per_item;
            m.p95_us = r.p95_us_per_item;
            m.samples = r.samples;
            m.ram_mb = r.ram_mb;
            m.vram_mb = r.vram_mb;
            m.cold_start_ms = r.cold_start_ms;
            // Seed the window with the restored p95, so the first measured sample doesn't replace it outright.
            m.window.clear();
            if r.samples > 0 {
                m.window.push_back(r.p95_us_per_item);
            }
        }
    }

    fn lease_info(&self, lease: LeaseId) -> Option<LeaseInfo> {
        let l = self.leases.get(&lease)?;
        Some(LeaseInfo {
            worker: l.worker,
            capability: l.capability,
            role: l.role,
            requires: l.requires.clone(),
            band: l.band,
            expires_ns: l.expires_ns,
            preempting: l.state == LeaseState::Preempting,
        })
    }

    fn promise_state(&self, promise: PromiseId) -> Option<PromiseState> {
        self.promises.get(&promise).map(|p| p.state)
    }

    fn promises_of(&self, consumer: ConsumerId) -> Vec<PromiseId> {
        self.promises
            .iter()
            .filter(|(_, p)| p.consumer == consumer && p.ended_ns.is_none())
            .map(|(id, _)| *id)
            .collect()
    }
}
