//! The worker side against the real broker (registration, quotes, renewal, preemption), and against a stand-in for
//! what the broker doesn't send yet (a commit or an abort answered with LeaseReleased).

mod common;

use std::time::Duration;

use common::*;
use continuum_broker::server::{Config, Server};
use continuum_client::{
    BrokerEvent, CapabilitySpec, Client, ClientOptions, Error, Quote, QuoteSpec, Requirement, StreamSpec,
    StreamStorage, WorkLease, WorkerSpec,
};
use continuum_core::schema::continuum as fb;
use flatbuffers::FlatBufferBuilder;
use tokio::sync::oneshot;

struct Broker {
    _dir: tempfile::TempDir,
    socket: std::path::PathBuf,
    stop: Option<oneshot::Sender<()>>,
}

impl Drop for Broker {
    fn drop(&mut self) {
        let _ = self.stop.take().map(|s| s.send(()));
    }
}

async fn broker() -> Broker {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("b.sock");
    // A small Tier 0 arena: the profile's is gigabytes.
    let mut config = Config::new(socket.clone(), Duration::from_secs(30));
    config.tier0 = Some(vec![continuum_sched::SlotClass { size_bytes: 16_384, count: 4 }]);
    let server = Server::bind(config).await.unwrap();
    let (stop, stopped) = oneshot::channel::<()>();
    tokio::spawn(server.run_until(async {
        let _ = stopped.await;
    }));
    Broker { _dir: dir, socket, stop: Some(stop) }
}

fn stream(name: &str, kind: fb::StreamKind, layer: fb::DomainLayer) -> StreamSpec {
    StreamSpec {
        name: name.into(),
        kind,
        format: fb::PayloadFormat::RawBytes,
        shape_name: name.into(),
        shape_version: 1,
        nominal_rate_hz: 30.0,
        storage: StreamStorage::Slab { class: 0 },
        anchor: None,
        layer,
    }
}

/// A connected worker with "cam0" (sensor) and "pose" (derived, from cam0) registered; returns it and the
/// capability's id.
async fn worker(b: &Broker, domain: fb::ComputeDomain) -> (Client, u32) {
    let c = Client::connect(&b.socket, ClientOptions::new("worker-test")).await.unwrap();
    c.register_stream(&stream("cam0", fb::StreamKind::Sensor, fb::DomainLayer::Layer1Raw)).await.unwrap();
    c.register_stream(&stream("pose", fb::StreamKind::Derived, fb::DomainLayer::Layer2Decision)).await.unwrap();
    let mut cap = CapabilitySpec::new("pose", domain);
    cap.requires.push(Requirement { stream: "cam0".into(), before_ns: 0, after_ns: 0 });
    let spec = WorkerSpec {
        worker_id: "worker:pose".into(),
        version_hash: "test".into(),
        capabilities: vec![cap],
        device_index: None,
        keeps_copies: false,
    };
    let registered = c.register_worker(&spec).await.unwrap();
    assert!(registered.refused_reasons.is_empty(), "{:?}", registered.refused_reasons);
    let id = registered.id_of(0).expect("capability 0 accepted");
    assert!(registered.accepted[0].runnable, "this machine's profile runs {domain:?}");
    (c, id)
}

fn quote(capability_id: u32, band: fb::PriorityBand) -> QuoteSpec {
    quote_on(capability_id, band, fb::ComputeDomain::CPU)
}

fn quote_on(capability_id: u32, band: fb::PriorityBand, compute_domain: fb::ComputeDomain) -> QuoteSpec {
    QuoteSpec {
        capability_id,
        stage: "pose".into(),
        band,
        compute_domain,
        output_bytes: 0,
        est_duration_ms: 1_000,
        pin_parents: Vec::new(),
        range: (0, 1_000),
    }
}

async fn granted(c: &Client, capability_id: u32, band: fb::PriorityBand, domain: fb::ComputeDomain) -> WorkLease {
    match c.quote(&quote_on(capability_id, band, domain)).await.unwrap() {
        Quote::Granted(lease) => lease,
        Quote::Denied { retry_after_ms } => panic!("denied; retry after {retry_after_ms} ms"),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_worker_registers_and_a_capability_on_an_unknown_stream_is_refused() {
    let b = broker().await;
    let (_registered, _) = worker(&b, fb::ComputeDomain::CPU).await;
    // A session registers one worker, so the refused one comes from another session.
    let c = Client::connect(&b.socket, ClientOptions::new("other")).await.unwrap();
    let spec = WorkerSpec {
        worker_id: "worker:x".into(),
        version_hash: String::new(),
        capabilities: vec![CapabilitySpec::new("no-such-stream", fb::ComputeDomain::CPU)],
        device_index: None,
        keeps_copies: false,
    };
    let r = c.register_worker(&spec).await.unwrap();
    assert!(r.accepted.is_empty());
    assert!(r.refused_reasons[0].contains("no-such-stream"), "{:?}", r.refused_reasons);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn quoting_needs_a_registered_worker() {
    let b = broker().await;
    let c = Client::connect(&b.socket, ClientOptions::new("not-a-worker")).await.unwrap();
    let err = c.quote(&quote(0, fb::PriorityBand::P3Background)).await.unwrap_err();
    assert_eq!(err.broker_code(), Some(fb::ErrorCode::NotAllowed), "{err}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_granted_lease_renews_and_its_keep_alive_keeps_extending_it() {
    let b = broker().await;
    let (c, cap) = worker(&b, fb::ComputeDomain::CPU).await;
    let mut lease = granted(&c, cap, fb::PriorityBand::P3Background, fb::ComputeDomain::CPU).await;
    let first = lease.expires_at_ns();
    tokio::time::sleep(Duration::from_millis(20)).await;
    let renewed = lease.renew(Duration::from_secs(60)).await.unwrap();
    assert!(renewed > first, "the expiry moved out (by the requested time, from now)");
    assert_eq!(lease.expires_at_ns(), renewed);
    lease.keep_alive(Duration::from_millis(50), Duration::from_secs(60));
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(lease.expires_at_ns() > renewed, "the keep-alive renewed it again in the background");
    lease.release();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_voice_turn_preempts_a_background_lease_and_both_events_arrive_typed() {
    let b = broker().await;
    // A voice turn preempts background GPU work (CPU leases aren't in its way).
    let (w, cap) = worker(&b, fb::ComputeDomain::GPU).await;
    let lease = granted(&w, cap, fb::PriorityBand::P3Background, fb::ComputeDomain::GPU).await;
    let voice = Client::connect(&b.socket, ClientOptions::new("voice")).await.unwrap();
    voice.set_system_state(fb::VoiceTurnState::Listening).unwrap();

    let first = tokio::time::timeout(Duration::from_secs(5), w.next_event()).await.unwrap().unwrap().event();
    match &first {
        BrokerEvent::LeasePreempted { lease_id, by_band, .. } => {
            assert_eq!(*lease_id, lease.id());
            assert_eq!(*by_band, fb::PriorityBand::P0Critical);
        }
        other => panic!("expected the preemption first, got {other:?}"),
    }
    assert!(!first.is_revocation(), "a higher band took it, the broker didn't revoke it");
    let second = tokio::time::timeout(Duration::from_secs(5), w.next_event()).await.unwrap().unwrap().event();
    assert!(matches!(second, BrokerEvent::SystemStateChanged { voice_turn: fb::VoiceTurnState::Listening, .. }));
    lease.release(); // the worker acknowledges and stops
    let heard = tokio::time::timeout(Duration::from_secs(5), voice.next_event()).await.unwrap().unwrap().event();
    assert!(matches!(heard, BrokerEvent::SystemStateChanged { .. }), "every session hears the new state");
}

// ---- Against a stand-in: commit and abort answered with LeaseReleased, and drop giving the lease back ----

fn envelope(
    request_id: u64,
    f: impl FnOnce(
        &mut FlatBufferBuilder<'static>,
    ) -> (fb::ControlPayload, flatbuffers::WIPOffset<flatbuffers::UnionWIPOffset>),
) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let (kind, payload) = f(&mut b);
    let env = fb::ControlEnvelope::create(
        &mut b,
        &fb::ControlEnvelopeArgs { request_id, payload_type: kind, payload: Some(payload) },
    );
    b.finish(env, None);
    b.finished_data().to_vec()
}

fn grant_reply(request_id: u64, lease_id: u64) -> Vec<u8> {
    envelope(request_id, |b| {
        let slots = b.create_vector::<flatbuffers::WIPOffset<fb::ShmLocation>>(&[]);
        let lease = fb::OutputLease::create(b, &fb::OutputLeaseArgs { lease_id, slots: Some(slots), expires_at_ns: 1 });
        let g = fb::LeaseGrant::create(b, &fb::LeaseGrantArgs { granted: true, lease: Some(lease), retry_after_ms: 0 });
        (fb::ControlPayload::LeaseGrant, g.as_union_value())
    })
}

fn released_reply(request_id: u64, lease_id: u64) -> Vec<u8> {
    envelope(request_id, |b| {
        let r = fb::LeaseReleased::create(b, &fb::LeaseReleasedArgs { lease_id, output_seqs: None });
        (fb::ControlPayload::LeaseReleased, r.as_union_value())
    })
}

fn refused_reply(request_id: u64, code: fb::ErrorCode) -> Vec<u8> {
    envelope(request_id, |b| {
        let message = b.create_string("refused by the stand-in");
        let e = fb::Error::create(b, &fb::ErrorArgs { code, message: Some(message) });
        (fb::ControlPayload::Error, e.as_union_value())
    })
}

/// A stand-in that grants lease 77 to the first quote, answers a commit or abort with LeaseReleased, and reports
/// every payload type it was sent (with the lease id for a LeaseReleased).
async fn lease_stand_in(
    socket: &std::path::Path,
) -> (tokio::task::JoinHandle<()>, tokio::sync::mpsc::UnboundedReceiver<(fb::ControlPayload, u64)>) {
    lease_stand_in_refusing(socket, vec![]).await
}

/// As `lease_stand_in`, but refuses the first commits with these codes, in order.
async fn lease_stand_in_refusing(
    socket: &std::path::Path,
    commit_refusals: Vec<fb::ErrorCode>,
) -> (tokio::task::JoinHandle<()>, tokio::sync::mpsc::UnboundedReceiver<(fb::ControlPayload, u64)>) {
    let (seen, heard) = tokio::sync::mpsc::unbounded_channel();
    let mut refusals = commit_refusals.into_iter();
    let server = scripted(socket, welcome(&[]), vec![], move |mut s| async move {
        let mut buf = Vec::new();
        while continuum_wire::read_frame(&mut s, &mut buf).await.is_ok() {
            let env = continuum_wire::read_envelope(&buf).unwrap();
            let (id, kind) = (env.request_id(), env.payload_type());
            let lease = env.payload_as_lease_released().map_or(0, |r| r.lease_id());
            let _ = seen.send((kind, lease));
            let reply = match kind {
                fb::ControlPayload::QuoteRequest => Some(grant_reply(id, 77)),
                fb::ControlPayload::CommitLease => {
                    Some(refusals.next().map_or_else(|| released_reply(id, 77), |code| refused_reply(id, code)))
                }
                fb::ControlPayload::AbortLease => Some(released_reply(id, 77)),
                _ => None,
            };
            if let Some(r) = reply {
                continuum_wire::write_frame(&mut s, &r).await.unwrap();
            }
        }
    })
    .await;
    (server, heard)
}

async fn next_non_heartbeat(
    heard: &mut tokio::sync::mpsc::UnboundedReceiver<(fb::ControlPayload, u64)>,
) -> (fb::ControlPayload, u64) {
    loop {
        let got = tokio::time::timeout(Duration::from_secs(5), heard.recv()).await.unwrap().unwrap();
        if got.0 != fb::ControlPayload::Heartbeat {
            return got;
        }
    }
}

#[tokio::test]
async fn a_commit_returns_once_a_stand_in_answers_it() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("s.sock");
    let (_server, mut heard) = lease_stand_in(&socket).await;
    let c = Client::connect(&socket, ClientOptions::new("t")).await.unwrap();
    let Quote::Granted(lease) = c.quote(&quote(1, fb::PriorityBand::P2Nearline)).await.unwrap() else { panic!() };
    assert_eq!(next_non_heartbeat(&mut heard).await.0, fb::ControlPayload::QuoteRequest);
    lease.commit(&[], None, None).await.unwrap();
    assert_eq!(next_non_heartbeat(&mut heard).await.0, fb::ControlPayload::CommitLease);
    assert_eq!(c.pending_requests(), 0);
}

#[tokio::test]
async fn an_abort_returns_once_the_broker_answers_it() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("s.sock");
    let (_server, mut heard) = lease_stand_in(&socket).await;
    let c = Client::connect(&socket, ClientOptions::new("t")).await.unwrap();
    let Quote::Granted(lease) = c.quote(&quote(1, fb::PriorityBand::P2Nearline)).await.unwrap() else { panic!() };
    next_non_heartbeat(&mut heard).await;
    lease.abort("out of memory").await.unwrap();
    assert_eq!(next_non_heartbeat(&mut heard).await.0, fb::ControlPayload::AbortLease);
}

#[tokio::test]
async fn a_refused_commit_gives_the_lease_back() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("s.sock");
    let (_server, mut heard) = lease_stand_in_refusing(&socket, vec![fb::ErrorCode::NotAllowed]).await;
    let c = Client::connect(&socket, ClientOptions::new("t")).await.unwrap();
    let Quote::Granted(lease) = c.quote(&quote(1, fb::PriorityBand::P2Nearline)).await.unwrap() else { panic!() };
    next_non_heartbeat(&mut heard).await;
    let err = lease.commit(&[], None, None).await.unwrap_err();
    assert_eq!(err.broker_code(), Some(fb::ErrorCode::NotAllowed));
    assert!(err.lease.is_none(), "the lease is over");
    assert_eq!(next_non_heartbeat(&mut heard).await.0, fb::ControlPayload::CommitLease);
    assert_eq!(next_non_heartbeat(&mut heard).await, (fb::ControlPayload::LeaseReleased, 77), "and released");
}

#[tokio::test]
async fn a_commit_refused_as_busy_hands_the_lease_back_for_a_retry() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("s.sock");
    let (_server, mut heard) = lease_stand_in_refusing(&socket, vec![fb::ErrorCode::Unavailable]).await;
    let c = Client::connect(&socket, ClientOptions::new("t")).await.unwrap();
    let Quote::Granted(mut lease) = c.quote(&quote(1, fb::PriorityBand::P2Nearline)).await.unwrap() else { panic!() };
    next_non_heartbeat(&mut heard).await;
    lease.keep_alive(Duration::from_secs(60), Duration::from_secs(30));
    let err = lease.commit(&[], None, None).await.unwrap_err();
    assert_eq!(err.broker_code(), Some(fb::ErrorCode::Unavailable));
    let lease = err.lease.expect("handed back");
    assert_eq!(lease.id(), 77);
    lease.commit(&[], None, None).await.unwrap();
    assert_eq!(next_non_heartbeat(&mut heard).await.0, fb::ControlPayload::CommitLease);
    assert_eq!(next_non_heartbeat(&mut heard).await.0, fb::ControlPayload::CommitLease, "retried, no release between");
}

#[tokio::test]
async fn dropping_an_unended_lease_gives_it_back() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("s.sock");
    let (_server, mut heard) = lease_stand_in(&socket).await;
    let c = Client::connect(&socket, ClientOptions::new("t")).await.unwrap();
    let Quote::Granted(lease) = c.quote(&quote(1, fb::PriorityBand::P2Nearline)).await.unwrap() else { panic!() };
    next_non_heartbeat(&mut heard).await;
    drop(lease);
    assert_eq!(next_non_heartbeat(&mut heard).await, (fb::ControlPayload::LeaseReleased, 77));
    drop(c);
}

#[tokio::test]
async fn a_lease_outliving_its_client_ends_quietly() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("s.sock");
    let (_server, _heard) = lease_stand_in(&socket).await;
    let c = Client::connect(&socket, ClientOptions::new("t")).await.unwrap();
    let Quote::Granted(lease) = c.quote(&quote(1, fb::PriorityBand::P2Nearline)).await.unwrap() else { panic!() };
    drop(c);
    assert!(matches!(lease.renew(Duration::from_secs(1)).await, Err(Error::Closed)));
    drop(lease); // no client to tell, and no panic
}

// ---- Commit and abort against the real broker ----

fn output_on(stream: u32, n: u64) -> continuum_core::HeaderBuf {
    let mut id = [0u8; 16];
    id[6] = 0x70;
    id[8] = 0x80;
    id[8..].copy_from_slice(&(0x8000_0000_0000_0000 | n).to_be_bytes());
    continuum_core::HeaderBuf::build(&continuum_core::HeaderSpec::raw(
        continuum_core::Uuid(id),
        continuum_core::StreamId(stream),
        100,
        200,
    ))
    .unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_commit_returns_once_its_outputs_are_durable_and_an_abort_once_done() {
    let b = broker().await;
    let (c, cap) = worker(&b, fb::ComputeDomain::CPU).await;
    let pose = c
        .register_stream(&stream("pose", fb::StreamKind::Derived, fb::DomainLayer::Layer2Decision))
        .await
        .unwrap()
        .stream_id;
    let lease = granted(&c, cap, fb::PriorityBand::P2Nearline, fb::ComputeDomain::CPU).await;
    let at = continuum_core::StorageLocator::Inline { bytes: b"a pose".to_vec() };
    let cost = continuum_client::Cost { wall_time_us: 1_000, items: 1, device: "CPU".into(), ..Default::default() };
    lease.commit(&[(output_on(pose, 1), Some(at))], None, Some(&cost)).await.unwrap();

    let lease = granted(&c, cap, fb::PriorityBand::P2Nearline, fb::ComputeDomain::CPU).await;
    lease.abort("the model ran out of memory").await.unwrap();
    assert_eq!(c.pending_requests(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_commit_with_an_output_on_another_stream_is_refused() {
    let b = broker().await;
    let (c, cap) = worker(&b, fb::ComputeDomain::CPU).await;
    let cam =
        c.register_stream(&stream("cam0", fb::StreamKind::Sensor, fb::DomainLayer::Layer1Raw)).await.unwrap().stream_id;
    let lease = granted(&c, cap, fb::PriorityBand::P2Nearline, fb::ComputeDomain::CPU).await;
    let err = lease.commit(&[(output_on(cam, 2), None)], None, None).await.unwrap_err();
    assert_eq!(err.broker_code(), Some(fb::ErrorCode::NotAllowed), "{err}");
    assert!(
        err.lease.is_none(),
        "a refused commit ends the lease (a_refused_commit_gives_the_lease_back shows the release)"
    );
}
