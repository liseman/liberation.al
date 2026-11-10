//! Pins the shape of the contract the broker codes against: `Scheduler` and `CatalogView` are object safe, every
//! method is callable through `&mut dyn Scheduler` and `&dyn CatalogView`, and the types build. A stub that returns
//! nothing stands in for the implementation.

use continuum_core::{ComputeDomain, Interval, PriorityBand, StreamId, VoiceTurnState};
use continuum_sched::{
    Accepted, Action, Capability, CapabilityId, CapabilityRole, CatalogView, ComputeCost, ConsumerId, CostModelRecord,
    CoverageDone, Error, FakeCatalog, IngestRef, JobId, LeaseDecision, LeaseId, LeaseReq, Manifest, Outcome, PromiseId,
    PromiseState, QueryDecision, QueryMode, QueryReq, Refusal, Scheduler, StreamKind, SystemState, Telemetry,
    ViewError, WorkerId,
};

struct Stub;

impl Scheduler for Stub {
    fn register_worker(&mut self, _: WorkerId, manifest: Manifest, _: u64) -> Vec<Accepted> {
        manifest
            .capabilities
            .iter()
            .enumerate()
            .map(|(i, _)| Accepted { capability: CapabilityId(u32::try_from(i).expect("few")), runnable: true })
            .collect()
    }
    fn worker_gone(&mut self, _: &dyn CatalogView, _: WorkerId, _: u64) -> Vec<Action> {
        Vec::new()
    }
    fn consumer_gone(&mut self, _: ConsumerId, _: u64) -> Vec<Action> {
        Vec::new()
    }
    fn telemetry(&mut self, _: WorkerId, _: Telemetry, _: u64) {}
    fn query(&mut self, catalog: &dyn CatalogView, req: QueryReq, _: u64) -> Outcome<QueryDecision> {
        let decision = match catalog.has_range(req.stream, req.range) {
            Ok(covered) => {
                let taken: Vec<Interval> = covered.iter().map(|c| c.interval).collect();
                let missing = continuum_sched::intervals::subtract(req.range, &taken);
                QueryDecision::Hit { covered, missing }
            }
            Err(e) => QueryDecision::Refused(Refusal::Catalog(e.0)),
        };
        Outcome { decision, actions: Vec::new() }
    }
    fn commit_promise(
        &mut self,
        _: &dyn CatalogView,
        promise: PromiseId,
        _: Option<u32>,
        _: PriorityBand,
        _: u64,
    ) -> Result<Vec<Action>, Error> {
        Err(Error::UnknownPromise(promise))
    }
    fn cancel_promise(&mut self, promise: PromiseId, _: u64) -> Result<Vec<Action>, Error> {
        Err(Error::UnknownPromise(promise))
    }
    fn on_ingest(&mut self, _: &dyn CatalogView, _: IngestRef, _: u64) -> Vec<Action> {
        Vec::new()
    }
    fn quote_request(&mut self, worker: WorkerId, req: LeaseReq, _: u64) -> Result<Outcome<LeaseDecision>, Error> {
        Err(Error::UnknownCapability { worker, capability: req.capability })
    }
    fn renew_lease(&mut self, lease: LeaseId, _: u64, _: u64) -> Result<u64, Error> {
        Err(Error::UnknownLease(lease))
    }
    fn commit_lease(
        &mut self,
        lease: LeaseId,
        _: &[continuum_sched::OutputRef],
        _: Option<CoverageDone>,
        _: ComputeCost,
        _: u64,
    ) -> Result<Vec<Action>, Error> {
        Err(Error::UnknownLease(lease))
    }
    fn abort_lease(&mut self, _: &dyn CatalogView, lease: LeaseId, _: &str, _: u64) -> Result<Vec<Action>, Error> {
        Err(Error::UnknownLease(lease))
    }
    fn release_lease(&mut self, lease: LeaseId, _: u64) -> Result<Vec<Action>, Error> {
        Err(Error::UnknownLease(lease))
    }
    fn job_rejected(&mut self, _: &dyn CatalogView, job: JobId, _: &str, _: u64) -> Result<Vec<Action>, Error> {
        Err(Error::UnknownJob(job))
    }
    fn dispatch_failed(&mut self, _: &dyn CatalogView, job: JobId, _: &str, _: u64) -> Result<Vec<Action>, Error> {
        Err(Error::UnknownJob(job))
    }
    fn claim_failed(&mut self, _: &dyn CatalogView, lease: LeaseId, _: u64) -> Result<Vec<Action>, Error> {
        Err(Error::UnknownLease(lease))
    }
    fn set_system_state(&mut self, _: SystemState, _: u64) -> Vec<Action> {
        Vec::new()
    }
    fn tick(&mut self, _: u64) -> Vec<Action> {
        Vec::new()
    }
    fn cost_models(&self) -> Vec<CostModelRecord> {
        Vec::new()
    }
    fn restore(&mut self, _: &[CostModelRecord]) {}
    fn lease_info(&self, _: continuum_sched::LeaseId) -> Option<continuum_sched::LeaseInfo> {
        None
    }
    fn promise_state(&self, _: PromiseId) -> Option<PromiseState> {
        None
    }
    fn promises_of(&self, _: ConsumerId) -> Vec<PromiseId> {
        Vec::new()
    }
}

fn iv(start: u64, end: u64) -> Interval {
    Interval::new(start, end).expect("test interval")
}

#[test]
fn the_contract_is_usable_through_trait_objects() {
    let mut cat = FakeCatalog::new();
    cat.add_stream(StreamId(1), StreamKind::Sensor, true, 30.0).cover(
        StreamId(1),
        iv(0, 10),
        continuum_core::StorageTier::Tier0Hot,
    );
    let view: &dyn CatalogView = &cat;

    let mut stub = Stub;
    let sched: &mut dyn Scheduler = &mut stub;

    let accepted = sched.register_worker(
        WorkerId(1),
        Manifest {
            producer: "worker:test".to_owned(),
            version_hash: "abc".to_owned(),
            device_index: None,
            capabilities: vec![Capability {
                role: CapabilityRole::Produce,
                produces: StreamId(2),
                requires: Vec::new(),
                domain: ComputeDomain::CPU,
                ram_mb: 1,
                vram_mb: 0,
                est_us_per_item: 1_000,
                cold_start_ms: 0,
                max_batch: 1,
                output_bytes: 1_024,
                outputs_per_job: 1,
                preemptible: true,
                expected_confidence: None,
            }],
        },
        0,
    );
    assert_eq!(accepted.len(), 1);

    let out = sched.query(
        view,
        QueryReq {
            consumer: ConsumerId(1),
            stream: StreamId(1),
            range: iv(0, 20),
            budget_us: None,
            strict: false,
            band: PriorityBand::P2Nearline,
            mode: QueryMode::HitOnly,
        },
        0,
    );
    match out.decision {
        QueryDecision::Hit { covered, missing } => {
            assert_eq!(covered.len(), 1);
            assert_eq!(missing, vec![iv(10, 20)]);
        }
        other => panic!("expected Hit, got {other:?}"),
    }
    assert!(out.actions.is_empty());

    cat.set_failing(Some(ViewError("disk".to_owned())));
    let view: &dyn CatalogView = &cat;
    let out = sched.query(
        view,
        QueryReq {
            consumer: ConsumerId(1),
            stream: StreamId(1),
            range: iv(0, 20),
            budget_us: None,
            strict: false,
            band: PriorityBand::P2Nearline,
            mode: QueryMode::HitOnly,
        },
        0,
    );
    assert_eq!(out.decision, QueryDecision::Refused(Refusal::Catalog("disk".to_owned())));

    assert!(sched.set_system_state(SystemState::VoiceTurn(VoiceTurnState::Listening), 0).is_empty());
    assert_eq!(sched.cancel_promise(PromiseId(7), 0), Err(Error::UnknownPromise(PromiseId(7))));
    assert_eq!(sched.promise_state(PromiseId(7)), None);
}
