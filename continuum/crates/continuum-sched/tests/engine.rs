//! The scheduler acceptance tests, plus the paths around them, against `FakeCatalog`.

use std::time::Instant;

use continuum_core::Uuid;
use continuum_core::{ComputeDomain, Interval, PriorityBand, StorageTier, StreamId, VoiceTurnState};
use continuum_sched::{
    Action, Capability, CapabilityId, CapabilityKey, CapabilityRole, ComputeCost, ConsumerId, CostModelRecord,
    DenyReason, Engine, FakeCatalog, IngestRef, LeaseDecision, LeaseEnd, LeaseReq, Manifest, OutputRef, Profile,
    PromiseState, QueryDecision, QueryMode, QueryReq, QuoteStatus, Refusal, Requirement, Scheduler, StepKind,
    StreamKind, SystemState, Telemetry, ViewError, WorkerId,
};

const S: u64 = 1_000_000_000;
const CAM: StreamId = StreamId(1);
const CROP: StreamId = StreamId(2);
const EMBED: StreamId = StreamId(3);
const POSE: StreamId = StreamId(4);
const NOW: u64 = 1_700_000_000 * S;

fn iv(start: u64, end: u64) -> Interval {
    Interval::new(start, end).expect("test interval")
}

fn uuid(n: u8) -> Uuid {
    Uuid([n; 16])
}

fn capability(
    produces: StreamId,
    requires: &[StreamId],
    domain: ComputeDomain,
    est_us: u64,
    cold_ms: u64,
) -> Capability {
    Capability {
        role: CapabilityRole::Produce,
        produces,
        requires: requires.iter().map(|s| Requirement { stream: *s, before_ns: 0, after_ns: 0 }).collect(),
        domain,
        ram_mb: 0,
        vram_mb: 0,
        est_us_per_item: est_us,
        cold_start_ms: cold_ms,
        max_batch: 100,
        output_bytes: 1024 * 1024,
        outputs_per_job: 1,
        preemptible: true,
        expected_confidence: None,
    }
}

fn manifest(producer: &str, caps: Vec<Capability>) -> Manifest {
    Manifest { producer: producer.to_owned(), version_hash: "v".to_owned(), device_index: None, capabilities: caps }
}

fn catalog() -> FakeCatalog {
    let mut cat = FakeCatalog::new();
    cat.add_stream(CAM, StreamKind::Sensor, true, 30.0)
        .add_stream(CROP, StreamKind::Derived, true, 30.0)
        .add_stream(EMBED, StreamKind::Derived, true, 30.0)
        .add_stream(POSE, StreamKind::Derived, true, 30.0)
        .cover(CAM, iv(NOW - 10 * S, NOW), StorageTier::Tier0Hot);
    cat
}

fn query(consumer: u64, stream: StreamId, range: Interval, mode: QueryMode) -> QueryReq {
    QueryReq {
        consumer: ConsumerId(consumer),
        stream,
        range,
        budget_us: None,
        strict: false,
        band: PriorityBand::P2Nearline,
        mode,
    }
}

/// A fleet: the face pipeline (crop then embed), one with the cost models restored to known values.
fn fleet(engine: &mut Engine) -> (CapabilityId, CapabilityId) {
    let w1 = engine.register_worker(
        WorkerId(1),
        manifest("worker:crop", vec![capability(CROP, &[CAM], ComputeDomain::GPU, 1_000, 500)]),
        NOW,
    );
    let w2 = engine.register_worker(
        WorkerId(2),
        manifest("worker:embed", vec![capability(EMBED, &[CROP], ComputeDomain::GPU, 2_000, 0)]),
        NOW,
    );
    engine.restore(&[
        CostModelRecord {
            key: CapabilityKey { producer: "worker:crop".to_owned(), output: CROP },
            ema_us_per_item: 1_000.0,
            p95_us_per_item: 5_000.0,
            samples: 10,
            ram_mb: 0,
            vram_mb: 0,
            cold_start_ms: 0,
        },
        CostModelRecord {
            key: CapabilityKey { producer: "worker:embed".to_owned(), output: EMBED },
            ema_us_per_item: 2_000.0,
            p95_us_per_item: 2_000.0,
            samples: 10,
            ram_mb: 0,
            vram_mb: 0,
            cold_start_ms: 0,
        },
    ]);
    (w1[0].capability, w2[0].capability)
}

fn dispatches(actions: &[Action]) -> Vec<&continuum_sched::Dispatch> {
    actions.iter().filter_map(|a| if let Action::Dispatch(d) = a { Some(d) } else { None }).collect()
}

#[test]
fn a_voice_turn_preempts_background_gpu_leases_within_5_ms() {
    let mut engine = Engine::new(Profile::linux_dual_3090());
    let accepted = engine.register_worker(
        WorkerId(9),
        manifest("worker:pose", vec![capability(POSE, &[CAM], ComputeDomain::GPU, 3_000, 0)]),
        NOW,
    );
    let cap = accepted[0].capability;
    let req = |range: Interval| LeaseReq {
        capability: cap,
        band: PriorityBand::P3Background,
        range,
        output_bytes: 1024,
        est_duration_us: 1_000_000,
        pin_parents: Vec::new(),
    };
    let mut granted = Vec::new();
    for i in 0..3u64 {
        let out = engine.quote_request(WorkerId(9), req(iv(i * S, (i + 1) * S)), NOW).expect("worker known");
        match out.decision {
            LeaseDecision::Granted(lease) => granted.push(lease.lease),
            other => panic!("expected a grant, got {other:?}"),
        }
    }
    assert_eq!(engine.active_leases(), 3);

    let started = Instant::now();
    let actions = engine.set_system_state(SystemState::VoiceTurn(VoiceTurnState::Listening), NOW + 1);
    let elapsed = started.elapsed();

    let preempted: Vec<_> =
        actions.iter().filter(|a| matches!(a, Action::Preempt { by_band: PriorityBand::P0Critical, .. })).collect();
    assert_eq!(preempted.len(), 3, "every P3 GPU lease is preempted: {actions:?}");
    assert!(elapsed.as_millis() < 5, "preemption took {elapsed:?}");
    assert_eq!(engine.active_leases(), 0);

    // While the turn is live, P3 GPU work is refused outright.
    let out = engine.quote_request(WorkerId(9), req(iv(5 * S, 6 * S)), NOW + 2).expect("worker known");
    assert!(matches!(out.decision, LeaseDecision::Denied { reason: DenyReason::VoiceTurn, .. }), "{:?}", out.decision);

    // The workers acknowledge; the turn ends; new requests are granted again.
    for lease in granted {
        engine.release_lease(lease, NOW + 3).expect("lease known");
    }
    engine.set_system_state(SystemState::VoiceTurn(VoiceTurnState::Idle), NOW + 4);
    let out = engine.quote_request(WorkerId(9), req(iv(6 * S, 7 * S)), NOW + 5).expect("worker known");
    assert!(matches!(out.decision, LeaseDecision::Granted(_)));
}

#[test]
fn a_two_step_plan_is_priced_by_the_cost_formula() {
    let mut engine = Engine::new(Profile::linux_dual_3090());
    let (crop, embed) = fleet(&mut engine);
    let cat = catalog();
    let range = iv(NOW - S, NOW);

    let out = engine.query(&cat, query(1, EMBED, range, QueryMode::QuoteIfMissing), NOW);
    let QueryDecision::Quoted { covered, quote } = out.decision else { panic!("expected a quote: {:?}", out.decision) };
    assert!(covered.is_empty());
    assert_eq!(quote.status, QuoteStatus::RequiresCompute);
    assert_eq!(quote.missing, vec![range]);
    let kinds: Vec<_> = quote.plan.steps.iter().map(|s| (s.kind, s.stream, s.capability)).collect();
    assert_eq!(kinds, vec![(StepKind::Compute, CROP, Some(crop)), (StepKind::Compute, EMBED, Some(embed))]);
    // 30 items each; queue depth 0; no cold start (the restored models say 0): 30 * 1000 + 30 * 2000.
    assert_eq!(quote.plan.steps[0].items, 30);
    assert_eq!(quote.est_latency_us, 30 * 1_000 + 30 * 2_000);
    let weight = engine.profile().weights.per_domain_us[&ComputeDomain::GPU];
    let expected_units = 30.0 * 1_000.0 * weight + 30.0 * 2_000.0 * weight;
    assert!((quote.cost_units - expected_units).abs() < 1e-9, "{} vs {expected_units}", quote.cost_units);
    assert!(!quote.exceeds_budget);
    assert_eq!(engine.promise_state(quote.promise), Some(PromiseState::Quoted));

    // Strict pricing uses the p95 (5000 for crop) and a budget it exceeds is still quoted.
    let mut strict = query(1, EMBED, range, QueryMode::QuoteIfMissing);
    strict.strict = true;
    strict.budget_us = Some(100_000);
    let out = engine.query(&cat, strict, NOW);
    let QueryDecision::Quoted { quote, .. } = out.decision else { panic!() };
    assert_eq!(quote.est_latency_us, 30 * 5_000 + 30 * 2_000);
    assert!(quote.exceeds_budget);

    // A cold model adds its load time.
    engine.telemetry(WorkerId(2), Telemetry { queue_depth: 0, loaded: Vec::new(), ram_mb: 0, vram_mb: 0 }, NOW);
    engine.restore(&[CostModelRecord {
        key: CapabilityKey { producer: "worker:embed".to_owned(), output: EMBED },
        ema_us_per_item: 2_000.0,
        p95_us_per_item: 2_000.0,
        samples: 10,
        ram_mb: 0,
        vram_mb: 0,
        cold_start_ms: 700,
    }]);
    let out = engine.query(&cat, query(1, EMBED, range, QueryMode::QuoteIfMissing), NOW);
    let QueryDecision::Quoted { quote, .. } = out.decision else { panic!() };
    assert_eq!(quote.est_latency_us, 30 * 1_000 + 700_000 + 30 * 2_000);
}

#[test]
fn a_committed_plan_runs_its_steps_in_order_and_fulfills() {
    let mut engine = Engine::new(Profile::linux_dual_3090());
    let (crop, embed) = fleet(&mut engine);
    let cat = catalog();
    let range = iv(NOW - S, NOW);

    let out = engine.query(&cat, query(1, EMBED, range, QueryMode::QuoteIfMissing), NOW);
    let QueryDecision::Quoted { quote, .. } = out.decision else { panic!() };
    let actions = engine.commit_promise(&cat, quote.promise, None, PriorityBand::P2Nearline, NOW).expect("quoted");
    let d = dispatches(&actions);
    assert_eq!(d.len(), 1, "only the first step dispatches: {actions:?}");
    assert_eq!((d[0].capability, d[0].worker, d[0].range), (crop, WorkerId(1), range));
    assert_eq!(d[0].inputs.len(), 1);
    assert_eq!(d[0].inputs[0].stream, CAM);
    assert_eq!(engine.promise_state(quote.promise), Some(PromiseState::Dispatching));

    let crop_lease = d[0].output.lease;
    let cost = ComputeCost {
        wall_time_us: 45_000,
        queue_wait_us: 0,
        ram_bytes: 0,
        vram_bytes: 0,
        device: "CUDA:0".to_owned(),
        items: 30,
    };
    let actions = engine
        .commit_lease(
            crop_lease,
            &[OutputRef { id: uuid(1), stream: CROP, range }],
            None,
            cost.clone(),
            NOW + 50_000_000,
        )
        .expect("lease known");
    assert!(
        !actions.iter().any(|a| matches!(a, Action::Fulfill { .. })),
        "intermediate outputs are not delivered: {actions:?}"
    );
    let d = dispatches(&actions);
    assert_eq!(d.len(), 1);
    assert_eq!((d[0].capability, d[0].worker), (embed, WorkerId(2)));
    assert_eq!(d[0].inputs[0].stream, CROP);

    let actions = engine
        .commit_lease(
            d[0].output.lease,
            &[OutputRef { id: uuid(2), stream: EMBED, range }],
            None,
            cost,
            NOW + 120_000_000,
        )
        .expect("lease known");
    let fulfilled: Vec<_> = actions.iter().filter(|a| matches!(a, Action::Fulfill { promise, consumer: ConsumerId(1), final_part: true, .. } if *promise == quote.promise)).collect();
    assert_eq!(fulfilled.len(), 1, "{actions:?}");
    assert_eq!(engine.promise_state(quote.promise), Some(PromiseState::Fulfilled));
    assert_eq!(engine.active_leases(), 0);

    // The crop model learned its cost: 45 ms over 30 items is 1500 us per item.
    let crop_model = engine.cost_models().into_iter().find(|m| m.key.output == CROP).expect("model");
    assert!((crop_model.ema_us_per_item - (0.2 * 1_500.0 + 0.8 * 1_000.0)).abs() < 1e-9);
    assert_eq!(crop_model.samples, 11);
}

#[test]
fn an_overlapping_query_attaches_to_the_job_in_flight() {
    let mut engine = Engine::new(Profile::linux_dual_3090());
    fleet(&mut engine);
    let cat = catalog();
    let range = iv(NOW - S, NOW);

    let out = engine.query(&cat, query(1, CROP, range, QueryMode::QuoteIfMissing), NOW);
    let QueryDecision::Quoted { quote: first, .. } = out.decision else { panic!() };
    let actions = engine.commit_promise(&cat, first.promise, None, PriorityBand::P2Nearline, NOW).expect("quoted");
    let lease = dispatches(&actions)[0].output.lease;

    let out = engine.query(&cat, query(2, CROP, range, QueryMode::QuoteIfMissing), NOW + 10_000_000);
    let QueryDecision::Quoted { quote: second, .. } = out.decision else { panic!() };
    assert!(second.shares_in_flight);
    assert_eq!(second.cost_units, 0.0);
    assert!(second.plan.steps.iter().all(|s| s.shared));
    assert!(second.est_latency_us < first.est_latency_us);
    let actions =
        engine.commit_promise(&cat, second.promise, None, PriorityBand::P2Nearline, NOW + 10_000_000).expect("quoted");
    assert!(dispatches(&actions).is_empty(), "no second job: {actions:?}");

    let cost = ComputeCost {
        wall_time_us: 30_000,
        queue_wait_us: 0,
        ram_bytes: 0,
        vram_bytes: 0,
        device: "CUDA:0".to_owned(),
        items: 30,
    };
    let actions = engine
        .commit_lease(lease, &[OutputRef { id: uuid(1), stream: CROP, range }], None, cost, NOW + 40_000_000)
        .expect("lease known");
    let mut consumers: Vec<u64> = actions
        .iter()
        .filter_map(
            |a| if let Action::Fulfill { consumer, final_part: true, .. } = a { Some(consumer.0) } else { None },
        )
        .collect();
    consumers.sort_unstable();
    assert_eq!(consumers, vec![1, 2]);
    assert_eq!(engine.promise_state(first.promise), Some(PromiseState::Fulfilled));
    assert_eq!(engine.promise_state(second.promise), Some(PromiseState::Fulfilled));
}

#[test]
fn a_future_sensor_range_arms_and_fulfills_from_ingests() {
    let mut engine = Engine::new(Profile::linux_dual_3090());
    let cat = catalog();
    let range = iv(NOW + S, NOW + 2 * S);

    let out = engine.query(&cat, query(1, CAM, range, QueryMode::QuoteIfMissing), NOW);
    let QueryDecision::Quoted { quote, .. } = out.decision else { panic!("{:?}", out.decision) };
    assert_eq!(quote.status, QuoteStatus::AwaitingCapture);
    assert_eq!(quote.est_latency_us, 2_000_000);
    assert_eq!(quote.plan.steps.len(), 1);
    assert_eq!(quote.plan.steps[0].kind, StepKind::Capture);

    let actions = engine.commit_promise(&cat, quote.promise, None, PriorityBand::P1Realtime, NOW).expect("quoted");
    assert!(actions.is_empty());
    assert_eq!(engine.promise_state(quote.promise), Some(PromiseState::Armed));

    // A frame before the range is not ours.
    assert!(
        engine
            .on_ingest(&cat, IngestRef { id: uuid(1), stream: CAM, range: iv(NOW, NOW + S / 2) }, NOW + S / 2)
            .is_empty()
    );
    let actions = engine.on_ingest(
        &cat,
        IngestRef { id: uuid(2), stream: CAM, range: iv(NOW + S, NOW + S + S / 2) },
        NOW + S + S / 2,
    );
    assert!(
        matches!(actions.as_slice(), [Action::Fulfill { final_part: false, outputs, .. }] if outputs.len() == 1),
        "{actions:?}"
    );
    let actions = engine.on_ingest(
        &cat,
        IngestRef { id: uuid(3), stream: CAM, range: iv(NOW + S + S / 2, NOW + 2 * S) },
        NOW + 2 * S,
    );
    assert!(matches!(actions.as_slice(), [Action::Fulfill { final_part: true, .. }]), "{actions:?}");
    assert_eq!(engine.promise_state(quote.promise), Some(PromiseState::Fulfilled));

    // A dead sensor stream is refused, not quoted.
    let mut dead = catalog();
    dead.set_live(CAM, false);
    let out = engine.query(&dead, query(1, CAM, iv(NOW + 3 * S, NOW + 4 * S), QueryMode::QuoteIfMissing), NOW);
    assert_eq!(out.decision, QueryDecision::Refused(Refusal::StreamNotLive(CAM)));
}

#[test]
fn work_that_does_not_fit_the_mac_profile_is_never_granted() {
    let mut engine = Engine::new(Profile::mac_mini_64gb());
    let mut big = capability(POSE, &[CAM], ComputeDomain::GPU, 3_000, 2_000);
    big.ram_mb = 50_000;
    let accepted = engine.register_worker(WorkerId(1), manifest("worker:huge", vec![big]), NOW);
    let req = LeaseReq {
        capability: accepted[0].capability,
        band: PriorityBand::P3Background,
        range: iv(0, S),
        output_bytes: 1024,
        est_duration_us: 1_000,
        pin_parents: Vec::new(),
    };
    let out = engine.quote_request(WorkerId(1), req, NOW).expect("worker known");
    assert!(matches!(out.decision, LeaseDecision::Denied { reason: DenyReason::Memory, .. }), "{:?}", out.decision);

    // 35 GB held by a P1 job that can't be interrupted; a 12 GB background model is queued, and a query that
    // needs it is quoted with the model's load time, not refused.
    let mut engine = Engine::new(Profile::mac_mini_64gb());
    let mut llm = capability(EMBED, &[CROP], ComputeDomain::GPU, 1_000, 0);
    llm.ram_mb = 35_000;
    llm.preemptible = false;
    let llm_id = engine.register_worker(WorkerId(1), manifest("worker:llm", vec![llm]), NOW)[0].capability;
    let mut vision = capability(POSE, &[CAM], ComputeDomain::GPU, 1_000, 3_000);
    vision.ram_mb = 12_000;
    let vision_id = engine.register_worker(WorkerId(2), manifest("worker:vision", vec![vision]), NOW)[0].capability;
    let held = engine
        .quote_request(
            WorkerId(1),
            LeaseReq {
                capability: llm_id,
                band: PriorityBand::P1Realtime,
                range: iv(0, S),
                output_bytes: 1024,
                est_duration_us: 1_000,
                pin_parents: Vec::new(),
            },
            NOW,
        )
        .expect("known");
    assert!(matches!(held.decision, LeaseDecision::Granted(_)));
    let out = engine
        .quote_request(
            WorkerId(2),
            LeaseReq {
                capability: vision_id,
                band: PriorityBand::P3Background,
                range: iv(0, S),
                output_bytes: 1024,
                est_duration_us: 1_000,
                pin_parents: Vec::new(),
            },
            NOW,
        )
        .expect("known");
    assert!(matches!(out.decision, LeaseDecision::Denied { reason: DenyReason::Memory, .. }), "{:?}", out.decision);

    let cat = catalog();
    let out = engine.query(&cat, query(1, POSE, iv(NOW - S, NOW), QueryMode::QuoteIfMissing), NOW);
    let QueryDecision::Quoted { quote, .. } = out.decision else { panic!("{:?}", out.decision) };
    assert_eq!(quote.est_latency_us, 3_000_000 + 30 * 1_000);
    // Committing queues the step rather than dispatching it; it runs once the memory is released.
    let actions = engine.commit_promise(&cat, quote.promise, None, PriorityBand::P3Background, NOW).expect("quoted");
    assert!(dispatches(&actions).is_empty(), "{actions:?}");
    assert_eq!(engine.promise_state(quote.promise), Some(PromiseState::Dispatching));
    let LeaseDecision::Granted(held) = held.decision else { panic!() };
    let actions = engine.release_lease(held.lease, NOW + S).expect("known");
    assert_eq!(dispatches(&actions).len(), 1, "{actions:?}");
}

#[test]
fn quotes_expire_and_consumers_can_cancel() {
    let mut engine = Engine::new(Profile::linux_dual_3090());
    fleet(&mut engine);
    let cat = catalog();
    let range = iv(NOW - S, NOW);

    let out = engine.query(&cat, query(1, CROP, range, QueryMode::QuoteIfMissing), NOW);
    let QueryDecision::Quoted { quote, .. } = out.decision else { panic!() };
    let late = quote.expires_ns + 1;
    assert_eq!(
        engine.commit_promise(&cat, quote.promise, None, PriorityBand::P2Nearline, late),
        Err(continuum_sched::Error::QuoteExpired(quote.promise))
    );
    assert_eq!(engine.promise_state(quote.promise), Some(PromiseState::Expired));

    let out = engine.query(&cat, query(1, CROP, range, QueryMode::QuoteIfMissing), NOW);
    let QueryDecision::Quoted { quote, .. } = out.decision else { panic!() };
    assert!(engine.tick(quote.expires_ns + 1).is_empty());
    assert_eq!(engine.promise_state(quote.promise), Some(PromiseState::Expired));

    // Cancelling the only promise of a running job revokes the job's lease.
    let out = engine.query(&cat, query(1, CROP, range, QueryMode::CommitWithinBudget), NOW);
    let QueryDecision::Committed { quote, .. } = out.decision else { panic!("{:?}", out.decision) };
    assert_eq!(dispatches(&out.actions).len(), 1);
    let actions = engine.cancel_promise(quote.promise, NOW + 1).expect("live");
    assert!(actions.iter().any(|a| matches!(a, Action::RevokeLease { .. })), "{actions:?}");
    assert_eq!(engine.promise_state(quote.promise), Some(PromiseState::Cancelled));
    assert_eq!(engine.active_leases(), 0);

    // A consumer that vanishes takes its promises with it, and nothing is addressed to it afterwards.
    let out = engine.query(&cat, query(7, CROP, range, QueryMode::QuoteIfMissing), NOW);
    let QueryDecision::Quoted { quote, .. } = out.decision else { panic!() };
    assert_eq!(engine.promises_of(ConsumerId(7)), vec![quote.promise]);
    let actions = engine.consumer_gone(ConsumerId(7), NOW + 2);
    assert!(actions.iter().all(|a| !matches!(
        a,
        Action::Fulfill { consumer: ConsumerId(7), .. } | Action::FailPromise { consumer: ConsumerId(7), .. }
    )));
    assert!(engine.promises_of(ConsumerId(7)).is_empty());
}

#[test]
fn a_lost_worker_replans_onto_another_producer_or_fails() {
    let mut engine = Engine::new(Profile::linux_dual_3090());
    let fast = engine.register_worker(
        WorkerId(1),
        manifest("worker:crop-fast", vec![capability(CROP, &[CAM], ComputeDomain::GPU, 1_000, 0)]),
        NOW,
    )[0]
    .capability;
    let slow = engine.register_worker(
        WorkerId(3),
        manifest("worker:crop-slow", vec![capability(CROP, &[CAM], ComputeDomain::GPU, 9_000, 0)]),
        NOW,
    )[0]
    .capability;
    let cat = catalog();
    let range = iv(NOW - S, NOW);

    let out = engine.query(&cat, query(1, CROP, range, QueryMode::QuoteIfMissing), NOW);
    let QueryDecision::Quoted { quote, .. } = out.decision else { panic!() };
    assert_eq!(quote.plan.steps[0].capability, Some(fast));
    assert_eq!(quote.alternatives.len(), 1);
    assert_eq!(quote.alternatives[0].plan.steps[0].capability, Some(slow));
    assert!(quote.alternatives[0].est_latency_us > quote.est_latency_us);

    let actions = engine.commit_promise(&cat, quote.promise, None, PriorityBand::P2Nearline, NOW).expect("quoted");
    assert_eq!(dispatches(&actions)[0].worker, WorkerId(1));
    let actions = engine.worker_gone(&cat, WorkerId(1), NOW + 1);
    assert!(actions.iter().any(|a| matches!(a, Action::RevokeLease { .. })));
    let d = dispatches(&actions);
    assert_eq!(d.len(), 1, "replanned onto the slow producer: {actions:?}");
    assert_eq!((d[0].worker, d[0].capability), (WorkerId(3), slow));
    assert_eq!(engine.promise_state(quote.promise), Some(PromiseState::Dispatching));

    // With the last producer gone there is nothing to replan onto.
    let actions = engine.worker_gone(&cat, WorkerId(3), NOW + 2);
    assert!(actions.iter().any(|a| matches!(a, Action::FailPromise { consumer: ConsumerId(1), .. })), "{actions:?}");
    assert_eq!(engine.promise_state(quote.promise), Some(PromiseState::Failed));
}

#[test]
fn hits_hydrates_and_refusals() {
    let mut engine = Engine::new(Profile::linux_dual_3090());
    fleet(&mut engine);
    let mut cat = catalog();
    cat.cover(CAM, iv(NOW - 60 * S, NOW - 10 * S), StorageTier::Tier1Warm);

    // HitOnly says what exists and what doesn't, and never quotes.
    let out = engine.query(&cat, query(1, CAM, iv(NOW - 2 * S, NOW + S), QueryMode::HitOnly), NOW);
    let QueryDecision::Hit { covered, missing } = out.decision else { panic!("{:?}", out.decision) };
    assert_eq!(covered.len(), 1);
    assert_eq!(missing, vec![iv(NOW, NOW + S)]);

    // Warm input with no hydrator registered is refused; with one, the plan starts with a hydrate step.
    let out = engine.query(&cat, query(1, CROP, iv(NOW - 20 * S, NOW - 19 * S), QueryMode::QuoteIfMissing), NOW);
    assert_eq!(out.decision, QueryDecision::Refused(Refusal::NoHydrator));
    let mut hydrate = capability(CAM, &[], ComputeDomain::CPU, 100, 0);
    hydrate.role = CapabilityRole::Hydrate;
    let tier = engine.register_worker(WorkerId(5), manifest("worker:tier", vec![hydrate]), NOW)[0].capability;
    let out = engine.query(&cat, query(1, CROP, iv(NOW - 20 * S, NOW - 19 * S), QueryMode::QuoteIfMissing), NOW);
    let QueryDecision::Quoted { quote, .. } = out.decision else { panic!("{:?}", out.decision) };
    assert_eq!(quote.plan.steps[0].kind, StepKind::Hydrate { from: StorageTier::Tier1Warm });
    assert_eq!(quote.plan.steps[0].capability, Some(tier));
    assert_eq!(quote.plan.steps[1].kind, StepKind::Compute);
    assert_eq!(quote.est_latency_us, 30 * 100 + 30 * 1_000);

    // Unknown streams, past sensor data nobody captured, and a failing catalog.
    assert_eq!(
        engine.query(&cat, query(1, StreamId(99), iv(0, S), QueryMode::QuoteIfMissing), NOW).decision,
        QueryDecision::Refused(Refusal::UnknownStream(StreamId(99)))
    );
    let out = engine.query(&cat, query(1, CAM, iv(NOW - 100 * S, NOW - 99 * S), QueryMode::QuoteIfMissing), NOW);
    match &out.decision {
        QueryDecision::Hit { missing, .. } => assert_eq!(*missing, vec![iv(NOW - 100 * S, NOW - 99 * S)]),
        other => panic!("past sensor data is reported missing, not quoted: {other:?}"),
    }
    cat.set_failing(Some(ViewError("disk".to_owned())));
    assert_eq!(
        engine.query(&cat, query(1, CROP, iv(0, S), QueryMode::QuoteIfMissing), NOW).decision,
        QueryDecision::Refused(Refusal::Catalog("disk".to_owned()))
    );
}

#[test]
fn an_unknown_lease_or_promise_is_an_error_not_a_panic() {
    let mut engine = Engine::new(Profile::linux_dual_3090());
    let cat = catalog();
    assert!(engine.renew_lease(continuum_sched::LeaseId(42), 0, NOW).is_err());
    assert!(engine.cancel_promise(continuum_sched::PromiseId(42), NOW).is_err());
    assert!(engine.job_rejected(&cat, continuum_sched::JobId(42), "x", NOW).is_err());
    assert!(
        engine
            .quote_request(
                WorkerId(42),
                LeaseReq {
                    capability: CapabilityId(1),
                    band: PriorityBand::P3Background,
                    range: iv(0, 1),
                    output_bytes: 1,
                    est_duration_us: 1,
                    pin_parents: Vec::new()
                },
                NOW
            )
            .is_err()
    );
    assert!(engine.tick(NOW).is_empty());
    assert!(engine.on_ingest(&cat, IngestRef { id: uuid(1), stream: CAM, range: iv(0, 1) }, NOW).is_empty());
}

#[test]
fn a_quote_says_which_past_part_it_can_no_longer_get() {
    // The normal state: sealed batches a little behind now, nothing reported past them.
    let mut engine = Engine::new(Profile::linux_dual_3090());
    fleet(&mut engine);
    let mut cat = FakeCatalog::new();
    cat.add_stream(CAM, StreamKind::Sensor, true, 30.0)
        .add_stream(CROP, StreamKind::Derived, true, 30.0)
        .add_stream(EMBED, StreamKind::Derived, true, 30.0)
        .cover(CAM, iv(NOW - 10 * S, NOW), StorageTier::Tier0Hot);

    // A derived range whose sensor data exists needs no capture: one compute step over it.
    let out = engine.query(&cat, query(1, CROP, iv(NOW - S, NOW), QueryMode::QuoteIfMissing), NOW);
    let QueryDecision::Quoted { covered, quote } = out.decision else { panic!("{:?}", out.decision) };
    assert!(covered.is_empty(), "a derived stream holds no payloads of its own");
    assert_eq!(quote.status, QuoteStatus::RequiresCompute);
    let kinds: Vec<_> = quote.plan.steps.iter().map(|s| (s.kind, s.stream)).collect();
    assert_eq!(kinds, vec![(StepKind::Compute, CROP)]);
    assert_eq!(quote.plan.steps[0].range, iv(NOW - S, NOW));
    assert!(quote.unavailable.is_empty());

    // A sensor query across what is missing: only the part ahead of now is quoted.
    let out = engine.query(&cat, query(1, CAM, iv(NOW - S, NOW + S), QueryMode::QuoteIfMissing), NOW);
    let QueryDecision::Quoted { covered, quote } = out.decision else { panic!("{:?}", out.decision) };
    assert_eq!(covered.len(), 1);
    assert_eq!(covered[0].interval, iv(NOW - S, NOW));
    assert_eq!(quote.missing, vec![iv(NOW, NOW + S)]);
    assert_eq!(quote.status, QuoteStatus::AwaitingCapture);

    // A query that also reaches into a real past gap says what is gone and quotes the rest.
    let out = engine.query(&cat, query(1, CAM, iv(NOW - 100 * S, NOW + S), QueryMode::QuoteIfMissing), NOW);
    let QueryDecision::Quoted { quote, .. } = out.decision else { panic!("{:?}", out.decision) };
    assert_eq!(quote.unavailable, vec![iv(NOW - 100 * S, NOW - 10 * S)]);
    assert_eq!(quote.missing, vec![iv(NOW, NOW + S)]);

    // A derived range that needs sensor data from the gap can't be made, and says which stream.
    let out = engine.query(&cat, query(1, CROP, iv(NOW - 50 * S, NOW - 49 * S), QueryMode::QuoteIfMissing), NOW);
    assert_eq!(out.decision, QueryDecision::Refused(Refusal::NotCaptured(CAM)));
}

#[test]
fn make_room_preempts_only_what_frees_the_short_resource() {
    let mut engine = Engine::new(Profile::mac_mini_64gb());
    let mut llm = capability(EMBED, &[CROP], ComputeDomain::GPU, 1_000, 0);
    llm.ram_mb = 35_000;
    llm.preemptible = false;
    let llm_id = engine.register_worker(WorkerId(1), manifest("worker:llm", vec![llm]), NOW)[0].capability;
    let cpu_id = engine.register_worker(
        WorkerId(2),
        manifest("worker:cpu", vec![capability(POSE, &[CAM], ComputeDomain::CPU, 1_000, 0)]),
        NOW,
    )[0]
    .capability;
    let mut vision = capability(CROP, &[CAM], ComputeDomain::GPU, 1_000, 0);
    vision.ram_mb = 12_000;
    let vision_id = engine.register_worker(WorkerId(3), manifest("worker:vision", vec![vision]), NOW)[0].capability;
    let req = |cap, band, range| LeaseReq {
        capability: cap,
        band,
        range,
        output_bytes: 1024,
        est_duration_us: 1_000,
        pin_parents: Vec::new(),
    };

    assert!(matches!(
        engine
            .quote_request(WorkerId(1), req(llm_id, PriorityBand::P1Realtime, iv(0, S)), NOW)
            .expect("known")
            .decision,
        LeaseDecision::Granted(_)
    ));
    for i in 0..3u64 {
        assert!(matches!(
            engine
                .quote_request(WorkerId(2), req(cpu_id, PriorityBand::P3Background, iv(i * S, (i + 1) * S)), NOW)
                .expect("known")
                .decision,
            LeaseDecision::Granted(_)
        ));
    }
    assert_eq!(engine.active_leases(), 4);

    // Memory is short and no preemptible lease frees any: nothing is preempted, the request is denied.
    let out =
        engine.quote_request(WorkerId(3), req(vision_id, PriorityBand::P2Nearline, iv(0, S)), NOW).expect("known");
    assert!(matches!(out.decision, LeaseDecision::Denied { reason: DenyReason::Memory, .. }), "{:?}", out.decision);
    assert!(out.actions.is_empty(), "{:?}", out.actions);
    assert_eq!(engine.active_leases(), 4);

    // A P3 lease that holds 12 GB on the GPU is exactly what frees the memory: it alone is preempted.
    let mut bg = capability(POSE, &[CAM], ComputeDomain::GPU, 1_000, 0);
    bg.ram_mb = 12_000;
    let bg_id = engine.register_worker(WorkerId(4), manifest("worker:bg", vec![bg]), NOW)[0].capability;
    // Make room for it first by ending the LLM lease temporarily is not needed: 35 + 12 = 47 > 40, so grant it
    // after dropping one CPU lease has no effect either; instead give it its own headroom by a fresh engine state.
    let mut engine = Engine::new(Profile::mac_mini_64gb());
    let bg_id2 = engine.register_worker(
        WorkerId(4),
        manifest(
            "worker:bg",
            vec![{
                let mut b = capability(POSE, &[CAM], ComputeDomain::GPU, 1_000, 0);
                b.ram_mb = 30_000;
                b
            }],
        ),
        NOW,
    )[0]
    .capability;
    let _ = (bg_id, bg_id2);
    let cpu2 = engine.register_worker(
        WorkerId(2),
        manifest("worker:cpu", vec![capability(EMBED, &[CROP], ComputeDomain::CPU, 1_000, 0)]),
        NOW,
    )[0]
    .capability;
    let vis2 = engine.register_worker(
        WorkerId(3),
        manifest(
            "worker:vision",
            vec![{
                let mut v = capability(CROP, &[CAM], ComputeDomain::GPU, 1_000, 0);
                v.ram_mb = 12_000;
                v
            }],
        ),
        NOW,
    )[0]
    .capability;
    assert!(matches!(
        engine
            .quote_request(WorkerId(4), req(bg_id2, PriorityBand::P3Background, iv(0, S)), NOW)
            .expect("known")
            .decision,
        LeaseDecision::Granted(_)
    ));
    assert!(matches!(
        engine
            .quote_request(WorkerId(2), req(cpu2, PriorityBand::P3Background, iv(0, S)), NOW)
            .expect("known")
            .decision,
        LeaseDecision::Granted(_)
    ));
    let out = engine.quote_request(WorkerId(3), req(vis2, PriorityBand::P2Nearline, iv(0, S)), NOW).expect("known");
    assert!(matches!(out.decision, LeaseDecision::Granted(_)), "{:?}", out.decision);
    let preempted: Vec<_> =
        out.actions.iter().filter(|a| matches!(a, Action::Preempt { worker: WorkerId(4), .. })).collect();
    assert_eq!(preempted.len(), 1, "only the 30 GB background lease goes: {:?}", out.actions);
    assert_eq!(out.actions.len(), 1);
}

#[test]
fn a_lost_worker_gives_vram_back_to_its_own_device() {
    let mut engine = Engine::new(Profile::linux_dual_3090());
    let mut m = manifest(
        "worker:big",
        vec![{
            let mut c = capability(POSE, &[CAM], ComputeDomain::GPU, 1_000, 0);
            c.vram_mb = 20_000;
            c
        }],
    );
    m.device_index = Some(1);
    let cap = engine.register_worker(WorkerId(7), m, NOW)[0].capability;
    let out = engine
        .quote_request(
            WorkerId(7),
            LeaseReq {
                capability: cap,
                band: PriorityBand::P2Nearline,
                range: iv(0, S),
                output_bytes: 1024,
                est_duration_us: 1_000,
                pin_parents: Vec::new(),
            },
            NOW,
        )
        .expect("known");
    assert!(matches!(out.decision, LeaseDecision::Granted(_)));
    assert_eq!(engine.vram_in_use_mb(), vec![0, 20_000]);
    engine.worker_gone(&catalog(), WorkerId(7), NOW + 1);
    assert_eq!(engine.vram_in_use_mb(), vec![0, 0]);
}

#[test]
fn a_rejected_job_is_replanned_onto_the_other_producer() {
    let mut engine = Engine::new(Profile::linux_dual_3090());
    let fast = engine.register_worker(
        WorkerId(1),
        manifest("worker:crop-fast", vec![capability(CROP, &[CAM], ComputeDomain::GPU, 1_000, 0)]),
        NOW,
    )[0]
    .capability;
    let slow = engine.register_worker(
        WorkerId(3),
        manifest("worker:crop-slow", vec![capability(CROP, &[CAM], ComputeDomain::GPU, 9_000, 0)]),
        NOW,
    )[0]
    .capability;
    let cat = catalog();
    let range = iv(NOW - S, NOW);
    let out = engine.query(&cat, query(1, CROP, range, QueryMode::CommitWithinBudget), NOW);
    let QueryDecision::Committed { quote, .. } = out.decision else { panic!("{:?}", out.decision) };
    let first = dispatches(&out.actions)[0].clone();
    assert_eq!(first.capability, fast);

    let actions = engine.job_rejected(&cat, first.job, "model not loaded", NOW + 1).expect("known job");
    let d = dispatches(&actions);
    assert_eq!(d.len(), 1, "{actions:?}");
    assert_eq!((d[0].capability, d[0].worker), (slow, WorkerId(3)));

    // The second replan is the last one allowed; a third failure fails the promise.
    let actions = engine.job_rejected(&cat, d[0].job, "still not", NOW + 2).expect("known job");
    let d2 = dispatches(&actions);
    assert_eq!(d2.len(), 1, "{actions:?}");
    assert_eq!(d2[0].capability, fast, "back to the other producer, never the one that just failed");
    let actions = engine.job_rejected(&cat, d2[0].job, "no", NOW + 3).expect("known job");
    assert!(actions.iter().any(|a| matches!(a, Action::FailPromise { .. })), "{actions:?}");
    assert_eq!(engine.promise_state(quote.promise), Some(PromiseState::Failed));
    assert_eq!(engine.active_leases(), 0);
}

#[test]
fn committed_promises_time_out_and_alternatives_commit_by_index() {
    let mut engine = Engine::new(Profile::linux_dual_3090());
    let cat = catalog();
    let range = iv(NOW + S, NOW + 2 * S);
    let out = engine.query(&cat, query(1, CAM, range, QueryMode::QuoteIfMissing), NOW);
    let QueryDecision::Quoted { quote, .. } = out.decision else { panic!() };
    engine.commit_promise(&cat, quote.promise, None, PriorityBand::P1Realtime, NOW).expect("quoted");
    assert_eq!(engine.promise_state(quote.promise), Some(PromiseState::Armed));
    // The deadline is three times the estimate (2 s of waiting) plus the grace.
    let grace = engine.settings().promise_grace_ns;
    assert!(engine.tick(NOW + 6 * S + grace).is_empty());
    let actions = engine.tick(NOW + 6 * S + grace + 1);
    assert!(
        actions.iter().any(|a| matches!(a, Action::FailPromise { promise, .. } if *promise == quote.promise)),
        "{actions:?}"
    );
    assert_eq!(engine.promise_state(quote.promise), Some(PromiseState::Failed));

    let fast = engine.register_worker(
        WorkerId(1),
        manifest("worker:crop-fast", vec![capability(CROP, &[CAM], ComputeDomain::GPU, 1_000, 0)]),
        NOW,
    )[0]
    .capability;
    let slow = engine.register_worker(
        WorkerId(3),
        manifest("worker:crop-slow", vec![capability(CROP, &[CAM], ComputeDomain::GPU, 9_000, 0)]),
        NOW,
    )[0]
    .capability;
    let out = engine.query(&cat, query(1, CROP, iv(NOW - S, NOW), QueryMode::QuoteIfMissing), NOW);
    let QueryDecision::Quoted { quote, .. } = out.decision else { panic!() };
    assert_eq!(quote.plan.steps[0].capability, Some(fast));
    assert_eq!(
        engine.commit_promise(&cat, quote.promise, Some(5), PriorityBand::P2Nearline, NOW),
        Err(continuum_sched::Error::UnknownAlternative { promise: quote.promise, alternative: 5 })
    );
    let actions = engine.commit_promise(&cat, quote.promise, Some(0), PriorityBand::P2Nearline, NOW).expect("quoted");
    assert_eq!(dispatches(&actions)[0].capability, slow);
}

#[test]
fn a_preempted_lease_that_commits_late_is_not_counted_twice() {
    let mut engine = Engine::new(Profile::linux_dual_3090());
    fleet(&mut engine);
    let cat = catalog();
    let a = engine.query(&cat, query(1, CROP, iv(NOW - 2 * S, NOW - S), QueryMode::CommitWithinBudget), NOW);
    let b = engine.query(&cat, query(2, CROP, iv(NOW - S, NOW), QueryMode::CommitWithinBudget), NOW);
    let lease_a = dispatches(&a.actions)[0].output.lease;
    let _lease_b = dispatches(&b.actions)[0].output.lease;
    // Two jobs in flight: a third quote sees queue depth 2.
    let out = engine.query(&cat, query(3, CROP, iv(NOW - 3 * S, NOW - 2 * S), QueryMode::QuoteIfMissing), NOW);
    let QueryDecision::Quoted { quote, .. } = out.decision else { panic!() };
    assert_eq!(quote.est_latency_us, 3 * 30 * 1_000);

    let actions = engine.set_system_state(SystemState::VoiceTurn(VoiceTurnState::Listening), NOW + 1);
    assert_eq!(actions.iter().filter(|a| matches!(a, Action::Preempt { .. })).count(), 2);
    // The preempted jobs are off the books: a quote sees queue depth 0 (but the voice turn blocks P2 GPU work,
    // so ask as P1).
    let mut q = query(3, CROP, iv(NOW - 4 * S, NOW - 3 * S), QueryMode::QuoteIfMissing);
    q.band = PriorityBand::P1Realtime;
    let out = engine.query(&cat, q, NOW + 2);
    let QueryDecision::Quoted { quote, .. } = out.decision else { panic!("{:?}", out.decision) };
    assert_eq!(quote.est_latency_us, 30 * 1_000);

    // A late commit of a preempted lease fulfills nobody and does not touch the depth.
    let cost = ComputeCost {
        wall_time_us: 30_000,
        queue_wait_us: 0,
        ram_bytes: 0,
        vram_bytes: 0,
        device: "CUDA:0".to_owned(),
        items: 30,
    };
    let actions = engine
        .commit_lease(
            lease_a,
            &[OutputRef { id: uuid(1), stream: CROP, range: iv(NOW - 2 * S, NOW - S) }],
            None,
            cost,
            NOW + 3,
        )
        .expect("known");
    assert!(!actions.iter().any(|a| matches!(a, Action::Fulfill { .. })), "{actions:?}");
    let mut q = query(3, CROP, iv(NOW - 5 * S, NOW - 4 * S), QueryMode::QuoteIfMissing);
    q.band = PriorityBand::P1Realtime;
    let out = engine.query(&cat, q, NOW + 4);
    let QueryDecision::Quoted { quote, .. } = out.decision else { panic!() };
    assert_eq!(quote.est_latency_us, 30 * 1_000);
}

#[test]
fn a_self_started_lease_commit_lets_queued_work_run() {
    let mut engine = Engine::new(Profile::mac_mini_64gb());
    let mut llm = capability(EMBED, &[CROP], ComputeDomain::GPU, 1_000, 0);
    llm.ram_mb = 35_000;
    llm.preemptible = false;
    let llm_id = engine.register_worker(WorkerId(1), manifest("worker:llm", vec![llm]), NOW)[0].capability;
    let mut vision = capability(POSE, &[CAM], ComputeDomain::GPU, 1_000, 0);
    vision.ram_mb = 12_000;
    engine.register_worker(WorkerId(2), manifest("worker:vision", vec![vision]), NOW);
    let held = engine
        .quote_request(
            WorkerId(1),
            LeaseReq {
                capability: llm_id,
                band: PriorityBand::P1Realtime,
                range: iv(0, S),
                output_bytes: 1024,
                est_duration_us: 1_000,
                pin_parents: Vec::new(),
            },
            NOW,
        )
        .expect("known");
    let LeaseDecision::Granted(held) = held.decision else { panic!() };
    let cat = catalog();
    let out = engine.query(&cat, query(1, POSE, iv(NOW - S, NOW), QueryMode::CommitWithinBudget), NOW);
    assert!(matches!(out.decision, QueryDecision::Committed { .. }));
    assert!(dispatches(&out.actions).is_empty());
    let cost = ComputeCost {
        wall_time_us: 1_000,
        queue_wait_us: 0,
        ram_bytes: 0,
        vram_bytes: 0,
        device: "MPS".to_owned(),
        items: 1,
    };
    let actions = engine.commit_lease(held.lease, &[], None, cost, NOW + S).expect("known");
    assert_eq!(dispatches(&actions).len(), 1, "the queued step runs once the memory is back: {actions:?}");
}

#[test]
fn a_commit_that_measured_nothing_doesnt_teach_the_cost_model_zero() {
    let mut engine = Engine::new(Profile::linux_dual_3090());
    let cap = capability(POSE, &[CAM], ComputeDomain::GPU, 4_000, 0);
    let id = engine.register_worker(WorkerId(1), manifest("worker:pose", vec![cap]), NOW)[0].capability;
    let req = || LeaseReq {
        capability: id,
        band: PriorityBand::P2Nearline,
        range: iv(0, S),
        output_bytes: 0,
        est_duration_us: 1_000,
        pin_parents: Vec::new(),
    };
    let lease = |engine: &mut Engine| {
        let LeaseDecision::Granted(l) = engine.quote_request(WorkerId(1), req(), NOW).expect("known").decision else {
            panic!("granted")
        };
        l.lease
    };
    let nothing =
        ComputeCost { wall_time_us: 0, queue_wait_us: 0, ram_bytes: 0, vram_bytes: 0, device: String::new(), items: 0 };
    let l = lease(&mut engine);
    engine.commit_lease(l, &[], None, nothing.clone(), NOW + 1).expect("known");
    for m in engine.cost_models() {
        assert_eq!(m.samples, 0, "no sample taken: {m:?}");
        assert!(m.ema_us_per_item > 0.0, "the estimate isn't zeroed: {m:?}");
    }
    // A real measurement still counts.
    let measured = ComputeCost { wall_time_us: 2_000, items: 1, ..nothing };
    let l = lease(&mut engine);
    engine.commit_lease(l, &[], None, measured, NOW + 2).expect("known");
    let m = engine.cost_models().into_iter().next().expect("a model");
    assert_eq!(m.samples, 1);
    assert!((m.ema_us_per_item - 2_000.0).abs() < 1e-9, "{m:?}");
}

/// A lease on a GPU capability for the worker's own work, granted.
fn granted(
    engine: &mut Engine,
    worker: WorkerId,
    capability: CapabilityId,
    band: PriorityBand,
) -> continuum_sched::OutputLease {
    let out = engine
        .quote_request(
            worker,
            LeaseReq {
                capability,
                band,
                range: iv(0, S),
                output_bytes: 1024,
                est_duration_us: 1_000,
                pin_parents: Vec::new(),
            },
            NOW,
        )
        .expect("known worker");
    let LeaseDecision::Granted(lease) = out.decision else { panic!("{:?}", out.decision) };
    lease
}

fn ended(actions: &[Action]) -> Vec<(continuum_sched::LeaseId, WorkerId, LeaseEnd)> {
    actions
        .iter()
        .filter_map(|a| match a {
            Action::LeaseEnded { lease, worker, end } => Some((*lease, *worker, *end)),
            _ => None,
        })
        .collect()
}

#[test]
fn every_lease_exit_sends_lease_ended_once_and_lease_info_follows() {
    let mut engine = Engine::new(Profile::linux_dual_3090());
    let cap = engine.register_worker(
        WorkerId(1),
        manifest("worker:crop", vec![capability(CROP, &[CAM], ComputeDomain::GPU, 1_000, 0)]),
        NOW,
    )[0]
    .capability;

    // Before any exit: the info names the owner, the capability, and its inputs.
    let l = granted(&mut engine, WorkerId(1), cap, PriorityBand::P2Nearline);
    let info = engine.lease_info(l.lease).expect("held");
    assert_eq!(
        (info.worker, info.capability, info.band, info.expires_ns, info.preempting),
        (WorkerId(1), cap, PriorityBand::P2Nearline, l.expires_ns, false)
    );
    assert_eq!(info.requires, vec![Requirement { stream: CAM, before_ns: 0, after_ns: 0 }]);
    assert_eq!(engine.lease_info(continuum_sched::LeaseId(999)), None);
    assert_eq!(info.role, CapabilityRole::Produce);
    // The inputs stay with the lease when the worker re-registers (its capabilities are forgotten and remade).
    let cap = engine.register_worker(
        WorkerId(1),
        manifest("worker:crop", vec![capability(CROP, &[CAM], ComputeDomain::GPU, 1_000, 0)]),
        NOW,
    )[0]
    .capability;
    assert_eq!(engine.lease_info(l.lease).expect("still held").requires.len(), 1);

    // Released.
    let actions = engine.release_lease(l.lease, NOW + 1).expect("held");
    assert_eq!(ended(&actions), vec![(l.lease, WorkerId(1), LeaseEnd::Released)]);
    assert_eq!(engine.lease_info(l.lease), None);
    assert!(engine.release_lease(l.lease, NOW + 2).is_err(), "a second release is unknown");

    // Committed, with Unpin before LeaseEnded.
    let l = granted(&mut engine, WorkerId(1), cap, PriorityBand::P2Nearline);
    let actions = engine
        .commit_lease(
            l.lease,
            &[],
            None,
            ComputeCost {
                wall_time_us: 500,
                queue_wait_us: 0,
                ram_bytes: 0,
                vram_bytes: 0,
                device: "CUDA:0".to_owned(),
                items: 1,
            },
            NOW + 1,
        )
        .expect("held");
    assert_eq!(ended(&actions), vec![(l.lease, WorkerId(1), LeaseEnd::Committed)]);
    let unpin = actions.iter().position(|a| matches!(a, Action::Unpin { .. })).expect("unpin");
    let gone = actions.iter().position(|a| matches!(a, Action::LeaseEnded { .. })).expect("ended");
    assert!(unpin < gone);

    // Aborted.
    let cat = catalog();
    let l = granted(&mut engine, WorkerId(1), cap, PriorityBand::P2Nearline);
    let actions = engine.abort_lease(&cat, l.lease, "oom", NOW + 1).expect("held");
    assert_eq!(ended(&actions), vec![(l.lease, WorkerId(1), LeaseEnd::Aborted)]);

    // Expired, by the tick: the revoke and the end come together.
    let l = granted(&mut engine, WorkerId(1), cap, PriorityBand::P2Nearline);
    let actions = engine.tick(l.expires_ns + 1);
    assert_eq!(ended(&actions), vec![(l.lease, WorkerId(1), LeaseEnd::Expired)]);
    assert!(actions.iter().any(|a| matches!(a, Action::RevokeLease { lease, .. } if *lease == l.lease)));
    assert_eq!(engine.lease_info(l.lease), None);

    // A preempted lease that expires ends too (no revoke this time: the worker was already told).
    let p3 = granted(&mut engine, WorkerId(1), cap, PriorityBand::P3Background);
    let actions = engine.set_system_state(SystemState::VoiceTurn(VoiceTurnState::Listening), NOW);
    assert!(actions.iter().any(|a| matches!(a, Action::Preempt { lease, .. } if *lease == p3.lease)), "{actions:?}");
    assert!(engine.lease_info(p3.lease).expect("still held").preempting);
    let actions = engine.tick(p3.expires_ns + 1);
    assert_eq!(ended(&actions), vec![(p3.lease, WorkerId(1), LeaseEnd::Expired)]);
    assert!(!actions.iter().any(|a| matches!(a, Action::RevokeLease { lease, .. } if *lease == p3.lease)));
    assert_eq!(engine.lease_info(p3.lease), None);
    engine.set_system_state(SystemState::VoiceTurn(VoiceTurnState::Idle), NOW);

    // Worker gone: every lease of the worker ends, once each.
    let a = granted(&mut engine, WorkerId(1), cap, PriorityBand::P2Nearline);
    let b = granted(&mut engine, WorkerId(1), cap, PriorityBand::P2Nearline);
    let actions = engine.worker_gone(&cat, WorkerId(1), NOW + 1);
    let mut got = ended(&actions);
    got.sort_by_key(|(id, _, _)| *id);
    let mut want = vec![(a.lease, WorkerId(1), LeaseEnd::WorkerGone), (b.lease, WorkerId(1), LeaseEnd::WorkerGone)];
    want.sort_by_key(|(id, _, _)| *id);
    assert_eq!(got, want);
    assert_eq!(engine.active_leases(), 0);
}

#[test]
fn a_cancelled_promise_ends_its_jobs_lease_with_promise_ended() {
    let mut engine = Engine::new(Profile::linux_dual_3090());
    fleet(&mut engine);
    let cat = catalog();
    let out = engine.query(&cat, query(1, CROP, iv(NOW - S, NOW), QueryMode::CommitWithinBudget), NOW);
    let QueryDecision::Committed { quote, .. } = out.decision else { panic!("{:?}", out.decision) };
    let lease = dispatches(&out.actions)[0].output.lease;
    assert!(engine.lease_info(lease).is_some());
    let actions = engine.cancel_promise(quote.promise, NOW + 1).expect("live");
    assert_eq!(ended(&actions), vec![(lease, dispatches(&out.actions)[0].worker, LeaseEnd::PromiseEnded)]);
    assert_eq!(engine.lease_info(lease), None);
}

#[test]
fn a_renewal_takes_the_requested_time_clamped_and_never_moves_an_expiry_earlier() {
    let mut engine = Engine::new(Profile::linux_dual_3090());
    let cap = capability(POSE, &[CAM], ComputeDomain::GPU, 1_000, 0);
    let id = engine.register_worker(WorkerId(1), manifest("worker:pose", vec![cap]), NOW)[0].capability;
    let req = LeaseReq {
        capability: id,
        band: PriorityBand::P2Nearline,
        range: iv(0, S),
        output_bytes: 0,
        est_duration_us: 1_000,
        pin_parents: Vec::new(),
    };
    let LeaseDecision::Granted(l) = engine.quote_request(WorkerId(1), req, NOW).expect("known").decision else {
        panic!("granted")
    };
    let (min, max, default) = {
        let s = engine.settings();
        (s.min_lease_ns, s.max_lease_ns, s.renew_ns)
    };
    let later = NOW + S;
    assert_eq!(engine.renew_lease(l.lease, 120 * S, later).unwrap(), later + 120 * S, "as asked");
    assert_eq!(engine.renew_lease(l.lease, 0, later).unwrap(), later + 120 * S, "the default never shortens it");
    assert_eq!(engine.renew_lease(l.lease, 3_600 * S, later).unwrap(), later + max, "capped");
    let fresh = engine.quote_request(
        WorkerId(1),
        LeaseReq {
            capability: id,
            band: PriorityBand::P2Nearline,
            range: iv(S, 2 * S),
            output_bytes: 0,
            est_duration_us: 1_000,
            pin_parents: Vec::new(),
        },
        NOW,
    );
    let LeaseDecision::Granted(f) = fresh.expect("known").decision else { panic!("granted") };
    let far = NOW + 1_000 * S;
    assert_eq!(engine.renew_lease(f.lease, 1, far).unwrap(), far + min, "at least the shortest lease");
    assert_eq!(engine.renew_lease(f.lease, 0, far + 1).unwrap(), (far + min).max(far + 1 + default));
}

#[test]
fn a_default_renewal_is_capped_too() {
    let settings = continuum_sched::Settings { renew_ns: 3_600 * S, ..Default::default() };
    let mut engine = Engine::with_settings(Profile::linux_dual_3090(), settings);
    let cap = capability(POSE, &[CAM], ComputeDomain::GPU, 1_000, 0);
    let id = engine.register_worker(WorkerId(1), manifest("worker:pose", vec![cap]), NOW)[0].capability;
    let req = LeaseReq {
        capability: id,
        band: PriorityBand::P2Nearline,
        range: iv(0, S),
        output_bytes: 0,
        est_duration_us: 1_000,
        pin_parents: Vec::new(),
    };
    let LeaseDecision::Granted(l) = engine.quote_request(WorkerId(1), req, NOW).expect("known").decision else {
        panic!("granted")
    };
    let max = engine.settings().max_lease_ns;
    assert_eq!(engine.renew_lease(l.lease, 0, NOW).unwrap(), NOW + max);
}

#[test]
fn thinking_preempts_and_blocks_only_background_accelerator_leases() {
    for domain in [ComputeDomain::CPU, ComputeDomain::GPU, ComputeDomain::ANE] {
        for band in
            [PriorityBand::P0Critical, PriorityBand::P1Realtime, PriorityBand::P2Nearline, PriorityBand::P3Background]
        {
            let mut engine = Engine::new(Profile::mac_mini_64gb());
            let cap = engine.register_worker(
                WorkerId(9),
                manifest("worker:fixture", vec![capability(POSE, &[CAM], domain, 1_000, 0)]),
                NOW,
            )[0]
            .capability;
            let request = |offset| LeaseReq {
                capability: cap,
                band,
                range: iv(offset * S, (offset + 1) * S),
                output_bytes: 1024,
                est_duration_us: 1_000_000,
                pin_parents: Vec::new(),
            };
            assert!(matches!(
                engine.quote_request(WorkerId(9), request(0), NOW).unwrap().decision,
                LeaseDecision::Granted(_)
            ));
            let actions = engine.set_system_state(SystemState::VoiceTurn(VoiceTurnState::Thinking), NOW + 1);
            let protected =
                matches!(domain, ComputeDomain::GPU | ComputeDomain::ANE) && band.0 >= PriorityBand::P2Nearline.0;
            assert_eq!(
                actions.iter().filter(|a| matches!(a, Action::Preempt { .. })).count(),
                usize::from(protected),
                "{domain:?} {band:?}"
            );
            let decision = engine.quote_request(WorkerId(9), request(1), NOW + 2).unwrap().decision;
            if protected {
                assert!(
                    matches!(decision, LeaseDecision::Denied { reason: DenyReason::VoiceTurn, .. }),
                    "{domain:?} {band:?}: {decision:?}"
                );
            } else {
                assert!(matches!(decision, LeaseDecision::Granted(_)), "{domain:?} {band:?}: {decision:?}");
            }
        }
    }
}

#[test]
fn listening_thinking_speaking_preserve_protection_until_idle_resumes_work() {
    let mut engine = Engine::new(Profile::linux_dual_3090());
    fleet(&mut engine);
    let cat = catalog();
    engine.set_system_state(SystemState::VoiceTurn(VoiceTurnState::Listening), NOW);
    let out = engine.query(&cat, query(1, CROP, iv(NOW - S, NOW), QueryMode::CommitWithinBudget), NOW);
    assert!(dispatches(&out.actions).is_empty());
    for (n, turn) in [VoiceTurnState::Thinking, VoiceTurnState::Speaking].into_iter().enumerate() {
        let actions = engine.set_system_state(SystemState::VoiceTurn(turn), NOW + n as u64 + 1);
        assert!(dispatches(&actions).is_empty(), "background work resumed during {turn:?}");
    }
    let actions = engine.set_system_state(SystemState::VoiceTurn(VoiceTurnState::Idle), NOW + 4);
    assert_eq!(dispatches(&actions).len(), 1);
}

#[test]
fn leaving_thinking_without_speaking_resumes_queued_background_work() {
    let mut engine = Engine::new(Profile::linux_dual_3090());
    fleet(&mut engine);
    let cat = catalog();
    engine.set_system_state(SystemState::VoiceTurn(VoiceTurnState::Thinking), NOW);
    let out = engine.query(&cat, query(1, CROP, iv(NOW - S, NOW), QueryMode::CommitWithinBudget), NOW);
    assert!(dispatches(&out.actions).is_empty());
    let actions = engine.set_system_state(SystemState::VoiceTurn(VoiceTurnState::Idle), NOW + 1);
    assert_eq!(dispatches(&actions).len(), 1);
}
