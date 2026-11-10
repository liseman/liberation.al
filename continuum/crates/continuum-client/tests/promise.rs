//! The consumer's promises against the real broker: a quoted query, a commit, a Rust worker serving the dispatched
//! job, and the promise gathering its outputs; a cancel revoking the job; a rejected job failing the promise; and a
//! range that's all there getting no quote.

mod common;

use std::time::Duration;

use continuum_broker::server::{Config, Server};
use continuum_client::{
    BrokerEvent, CapabilitySpec, Client, ClientOptions, Error, QuerySpec, Requirement, StreamSpec, StreamStorage,
    WorkerSpec,
};
use continuum_core::schema::continuum as fb;
use continuum_core::{HeaderBuf, HeaderSpec, StorageLocator, StreamId, Uuid};
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
    let mut config = Config::new(socket.clone(), Duration::from_secs(30));
    config.tier0 = Some(vec![continuum_sched::SlotClass { size_bytes: 16_384, count: 8 }]);
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

fn v7(n: u8) -> Uuid {
    let mut id = [n; 16];
    id[6] = 0x70 | (n & 0x0f);
    id[8] = 0x80 | (n & 0x3f);
    Uuid(id)
}

/// A worker that captures cam0 and computes "pose" from it, with three cam0 frames over [100, 400) in Tier 0 slots
/// (a compute step's inputs must be in Tier 0); returns it and the pose stream's id.
async fn worker(b: &Broker) -> (Client, StreamId) {
    let c = Client::connect(&b.socket, ClientOptions::new("pose-worker")).await.unwrap();
    let cam = c.register_stream(&stream("cam0", fb::StreamKind::Sensor, fb::DomainLayer::Layer1Raw)).await.unwrap();
    let pose = c.register_stream(&stream("pose", fb::StreamKind::Derived, fb::DomainLayer::Layer2Decision)).await;
    let pose = StreamId(pose.unwrap().stream_id);
    let mut cap = CapabilitySpec::new("pose", fb::ComputeDomain::CPU);
    cap.requires.push(Requirement { stream: "cam0".into(), before_ns: 0, after_ns: 0 });
    let spec = WorkerSpec {
        worker_id: "pose-worker".into(),
        version_hash: "test".into(),
        capabilities: vec![cap],
        device_index: None,
        keeps_copies: false,
    };
    let registered = c.register_worker(&spec).await.unwrap();
    assert!(registered.accepted.first().is_some_and(|a| a.runnable), "{:?}", registered.refused_reasons);
    let info = c.arenas()[0].info.clone();
    let arena = c.arena(&info.name).unwrap();
    for (n, start) in [(1u8, 100u64), (2, 200), (3, 300)] {
        let mut slot = arena.claim(0, 1).unwrap();
        slot.as_mut_slice()[..4].copy_from_slice(b"jpeg");
        let offset = slot.offset() as u64;
        let h = slot.publish(4).unwrap();
        let at = StorageLocator::Shm {
            arena: info.name.clone(),
            arena_generation: info.generation,
            slab_class: h.class,
            slot: h.index,
            slot_generation: h.generation,
            offset,
            length: 4,
        };
        let header = HeaderBuf::build(&HeaderSpec::raw(v7(n), StreamId(cam.stream_id), start, start + 100)).unwrap();
        c.ingest_acked(&header, Some(&at)).await.unwrap();
    }
    (c, pose)
}

fn pose_query() -> QuerySpec {
    let mut spec = QuerySpec::hit_only("pose", (100, 400));
    spec.mode = fb::QueryMode::QuoteIfMissing;
    spec
}

/// The next DispatchJob the worker is sent.
async fn next_job(w: &Client) -> continuum_client::DispatchJob {
    loop {
        let m = tokio::time::timeout(Duration::from_secs(5), w.next_event()).await.unwrap().unwrap();
        if let BrokerEvent::DispatchJob(job) = m.event() {
            return *job;
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_quoted_query_commits_and_a_rust_worker_fulfils_the_promise() {
    let b = broker().await;
    let (w, pose) = worker(&b).await;
    let c = Client::connect(&b.socket, ClientOptions::new("consumer")).await.unwrap();

    let (result, quote) = c.query_quoted(&pose_query()).await.unwrap();
    assert!(result.items.is_empty() && result.quoted && result.missing == vec![(100, 400)]);
    let quote = quote.expect("pose over the frames can be made");
    assert_eq!(quote.status, fb::PromiseStatus::RequiresCompute);
    let step = &quote.options[0].steps[0];
    assert_eq!((step.worker_id.as_str(), step.produces.as_str()), ("pose-worker", "pose"));

    let (accepted, promise) = c.commit_promise(quote.promise_id, 0, fb::PriorityBand::P2Nearline).await.unwrap();
    assert_eq!((accepted.promise_id, accepted.life), (quote.promise_id, fb::PromiseLife::Dispatching));
    assert_eq!(promise.id(), quote.promise_id);

    // The worker serves the job: one pose payload over the range, from the frames it was given.
    let mut job = next_job(&w).await;
    assert!(job.promise_ids.contains(&quote.promise_id));
    assert_eq!(job.inputs.len(), 3);
    let lease = w.job_lease(&mut job).unwrap();
    let mut spec = HeaderSpec::raw(v7(77), pose, job.output_range.0, job.output_range.1);
    spec.parents = job.inputs.iter().map(|i| i.header.id()).collect();
    let out = HeaderBuf::build(&spec).unwrap();
    lease.commit(&[(out, Some(StorageLocator::Inline { bytes: b"pose".to_vec() }))], None, None).await.unwrap();

    let items = tokio::time::timeout(Duration::from_secs(5), promise.fulfilled()).await.unwrap().unwrap();
    assert_eq!(items.iter().map(|i| i.header.id()).collect::<Vec<_>>(), vec![v7(77)]);
    // Routed to the promise, so nothing about it reached the consumer's events.
    assert!(tokio::time::timeout(Duration::from_millis(200), c.next_event()).await.is_err());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancelling_a_committed_promise_revokes_its_job() {
    let b = broker().await;
    let (w, _) = worker(&b).await;
    let c = Client::connect(&b.socket, ClientOptions::new("consumer")).await.unwrap();
    let (_, quote) = c.query_quoted(&pose_query()).await.unwrap();
    let quote = quote.unwrap();
    let (_, mut promise) = c.commit_promise(quote.promise_id, 0, fb::PriorityBand::P2Nearline).await.unwrap();
    // One handle per promise: a second would take the first one's outputs.
    assert!(matches!(c.promise(quote.promise_id), Err(continuum_client::Error::Protocol(_))));
    let mut job = next_job(&w).await;
    let lease = w.job_lease(&mut job).unwrap();
    let cancelled = c.cancel_promise(quote.promise_id).await.unwrap();
    assert_eq!(cancelled.life, fb::PromiseLife::Cancelled);
    // The held handle ends rather than waiting forever for parts that won't come.
    let ended = tokio::time::timeout(Duration::from_secs(5), promise.next_part()).await.unwrap().unwrap();
    assert!(ended.is_none());
    loop {
        let m = tokio::time::timeout(Duration::from_secs(5), w.next_event()).await.unwrap().unwrap();
        let event = m.event();
        if let BrokerEvent::LeasePreempted { lease_id, .. } = event {
            assert_eq!(lease_id, lease.id());
            assert!(event.is_revocation());
            break;
        }
    }
    let gone = c.cancel_promise(quote.promise_id).await.unwrap_err();
    assert_eq!(gone.broker_code(), Some(fb::ErrorCode::NotFound));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_rejected_job_fails_the_promise() {
    let b = broker().await;
    let (w, _) = worker(&b).await;
    let c = Client::connect(&b.socket, ClientOptions::new("consumer")).await.unwrap();
    let (_, quote) = c.query_quoted(&pose_query()).await.unwrap();
    let quote = quote.unwrap();
    let (_, promise) = c.commit_promise(quote.promise_id, 0, fb::PriorityBand::P2Nearline).await.unwrap();
    // The broker replans a rejected job onto the only producer twice before it gives up.
    for _ in 0..3 {
        let job = next_job(&w).await;
        w.reject_job(job.job_id, "no model here").unwrap();
    }
    let err = tokio::time::timeout(Duration::from_secs(5), promise.fulfilled()).await.unwrap().unwrap_err();
    assert!(matches!(&err, Error::PromiseFailed { delivered: 0, reason } if reason.contains("no model")), "{err}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_range_that_is_all_there_gets_no_quote() {
    let b = broker().await;
    let (_w, _) = worker(&b).await;
    let c = Client::connect(&b.socket, ClientOptions::new("consumer")).await.unwrap();
    let mut spec = QuerySpec::hit_only("cam0", (100, 400));
    spec.mode = fb::QueryMode::QuoteIfMissing;
    let (result, quote) = c.query_quoted(&spec).await.unwrap();
    assert_eq!((result.items.len(), result.complete, result.quoted), (3, true, false));
    assert!(quote.is_none());
}
