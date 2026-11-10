//! `commit_chunk` and `rewrite_chunk` against the real broker: a registered worker commits a chunk that absorbs the
//! frames a producer ingested (a retry is a duplicate, not an error), the frames are compacted into it, and a
//! rewrite records its new location. A session that registered no worker may do neither.

use std::time::Duration;

use continuum_broker::server::{Config, Server};
use continuum_client::{
    ChunkFrame, Client, ClientOptions, Error, FrameProvenance, NewChunk, StreamSpec, StreamStorage, WorkerSpec,
};
use continuum_core::schema::continuum as fb;
use continuum_core::{HeaderBuf, HeaderSpec, PayloadFormat, StorageLocator, StorageTier, StreamId, Uuid};
use tokio::sync::oneshot;

const T0: u64 = 1_700_000_000_000_000_000;
const FRAME_NS: u64 = 33_333_333;

struct Broker {
    _dir: tempfile::TempDir,
    socket: std::path::PathBuf,
    core: continuum_broker::core::CoreHandle,
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
    let server = Server::bind(Config::new(socket.clone(), Duration::from_secs(30))).await.unwrap();
    let core = server.core();
    let (stop, stopped) = oneshot::channel::<()>();
    tokio::spawn(server.run_until(async {
        let _ = stopped.await;
    }));
    Broker { _dir: dir, socket, core, stop: Some(stop) }
}

fn uuid(n: u32) -> Uuid {
    let mut b = [0u8; 16];
    b[0] = 0x01;
    b[6] = 0x70;
    b[8] = 0x80;
    b[12..16].copy_from_slice(&n.to_be_bytes());
    Uuid(b)
}

fn frame_header(id: Uuid, stream: StreamId, start_ns: u64, seq: u32) -> HeaderBuf {
    let mut spec = HeaderSpec::raw(id, stream, start_ns, start_ns);
    spec.seq = seq;
    spec.format = PayloadFormat::EncodedVideo;
    spec.shape_name = "MJPEG".into();
    spec.shape_version = 1;
    HeaderBuf::build(&spec).unwrap()
}

/// A producer that registered "cam0" and ingested `n` frames, and a tier worker on its own session.
async fn setup(b: &Broker, n: u32) -> (Client, Client, StreamId) {
    let producer = Client::connect(&b.socket, ClientOptions::new("producer")).await.unwrap();
    let spec = StreamSpec {
        name: "cam0".into(),
        kind: fb::StreamKind::Sensor,
        format: fb::PayloadFormat::EncodedVideo,
        shape_name: "MJPEG".into(),
        shape_version: 1,
        nominal_rate_hz: 30.0,
        storage: StreamStorage::Slab { class: 0 },
        anchor: None,
        layer: fb::DomainLayer::Layer1Raw,
    };
    let cam = StreamId(producer.register_stream(&spec).await.unwrap().stream_id);
    for i in 0..n {
        producer.ingest(&frame_header(uuid(i), cam, T0 + u64::from(i) * FRAME_NS, i), None).unwrap();
    }
    // A request after the notices means the broker has read them all (one session reads in order); then the
    // writer commits them. The tier worker learns of frames from its subscription, which fires only after this.
    producer.heartbeat(None).await.unwrap();
    b.core.sync_writer().await.unwrap();
    let tier = Client::connect(&b.socket, ClientOptions::new("tier")).await.unwrap();
    let worker = WorkerSpec {
        worker_id: "tier:test".into(),
        version_hash: "test".into(),
        capabilities: vec![],
        device_index: None,
        keeps_copies: true,
    };
    tier.register_worker(&worker).await.unwrap();
    (producer, tier, cam)
}

/// A chunk absorbing frames `0..n`, 600 bytes each, in a file at `uri`.
fn chunk(cam: StreamId, n: u32, uri: &str) -> NewChunk {
    let start = T0;
    let end = T0 + u64::from(n - 1) * FRAME_NS + 1;
    let mut spec = HeaderSpec::raw(uuid(1_000), cam, start, end);
    spec.format = PayloadFormat::EncodedVideo;
    spec.shape_name = "MJPEG".into();
    spec.shape_version = 1;
    let frames = (0..n)
        .map(|i| ChunkFrame {
            start_ns: T0 + u64::from(i) * FRAME_NS,
            offset: u64::from(i) * 600,
            length: 600,
            cut: false,
        })
        .collect();
    NewChunk {
        header: HeaderBuf::build(&spec).unwrap(),
        location: StorageLocator::Chunk {
            tier: StorageTier::Tier1Warm,
            uri: uri.into(),
            offset: 0,
            length: u64::from(n) * 600,
        },
        frames,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_worker_commits_a_chunk_once_and_a_retry_is_a_duplicate() {
    let b = broker().await;
    let (_producer, tier, cam) = setup(&b, 5).await;
    let absorbed: Vec<Uuid> = (0..5).map(uuid).collect();
    let c = chunk(cam, 5, "/var/tier/cam0-1.cnk");
    let first = tier.commit_chunk(&c, &absorbed).await.unwrap();
    assert_eq!(first.chunk_id, uuid(1_000));
    assert!(first.commit_seq > 0);
    assert!(!first.duplicate && !first.hidden);
    // The broker's availability index knows the Tier 1 copy now.
    let span = continuum_core::Interval::new(T0, T0 + 4 * FRAME_NS + 1).unwrap();
    let covered = b.core.covered(cam, span).await.unwrap();
    assert_eq!(covered.len(), 1, "{covered:?}");
    assert_eq!((covered[0].interval, covered[0].tier), (span, StorageTier::Tier1Warm));
    let again = tier.commit_chunk(&c, &absorbed).await.unwrap();
    assert!(again.duplicate, "a retry after a lost reply changes nothing");
    assert_eq!(again.commit_seq, first.commit_seq);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn frame_provenance_goes_in_with_the_chunk_and_comes_back_from_get_chunk() {
    let b = broker().await;
    let (_producer, tier, cam) = setup(&b, 3).await;
    let absorbed: Vec<Uuid> = (0..3).map(uuid).collect();
    let rows: Vec<FrameProvenance> =
        (0..3).map(|i| FrameProvenance { device_seq: 100 + i, epoch: 0xc0ffee, stamped_at_arrival: i == 2 }).collect();
    // One entry short of the frames never reaches the catalog.
    let c = chunk(cam, 3, "/var/tier/cam0-1.cnk");
    let err = tier.commit_chunk_with_provenance(&c, &absorbed, Some(&rows[..2])).await.unwrap_err();
    assert_eq!(err.broker_code(), Some(fb::ErrorCode::BadRequest), "{err}");
    tier.commit_chunk_with_provenance(&c, &absorbed, Some(&rows)).await.unwrap();
    let info = tier.get_chunk(uuid(1_000)).await.unwrap();
    assert_eq!(info.provenance, Some(rows));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_chunk_committed_without_provenance_reads_back_as_unknown() {
    let b = broker().await;
    let (_producer, tier, cam) = setup(&b, 2).await;
    let absorbed: Vec<Uuid> = (0..2).map(uuid).collect();
    tier.commit_chunk(&chunk(cam, 2, "/var/tier/cam0-1.cnk"), &absorbed).await.unwrap();
    assert_eq!(tier.get_chunk(uuid(1_000)).await.unwrap().provenance, None);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_rewritten_chunk_is_recorded_and_bad_ones_are_refused() {
    let b = broker().await;
    let (_producer, tier, cam) = setup(&b, 3).await;
    let absorbed: Vec<Uuid> = (0..3).map(uuid).collect();
    let c = chunk(cam, 3, "/var/tier/cam0-1.cnk");
    tier.commit_chunk(&c, &absorbed).await.unwrap();
    let moved = StorageLocator::Chunk {
        tier: StorageTier::Tier1Warm,
        uri: "/var/tier/cam0-1.v2.cnk".into(),
        offset: 0,
        length: 1_800,
    };
    tier.rewrite_chunk(uuid(1_000), &moved, &c.frames).await.unwrap();
    // A rewrite must keep the same frames.
    let err = tier.rewrite_chunk(uuid(1_000), &moved, &c.frames[..2]).await.unwrap_err();
    assert_eq!(err.broker_code(), Some(fb::ErrorCode::BadRequest), "{err}");
    let err = tier.rewrite_chunk(uuid(9_999), &moved, &c.frames).await.unwrap_err();
    assert_eq!(err.broker_code(), Some(fb::ErrorCode::NotFound), "{err}");
    // Frames and absorbed ids that don't line up never reach the catalog.
    let err = tier.commit_chunk(&chunk(cam, 3, "/var/tier/x.cnk"), &absorbed[..2]).await.unwrap_err();
    assert_eq!(err.broker_code(), Some(fb::ErrorCode::BadRequest), "{err}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_session_that_registered_no_worker_may_not_commit_or_rewrite() {
    let b = broker().await;
    let (producer, _tier, cam) = setup(&b, 2).await;
    let absorbed: Vec<Uuid> = (0..2).map(uuid).collect();
    let c = chunk(cam, 2, "/var/tier/cam0-1.cnk");
    let err = producer.commit_chunk(&c, &absorbed).await.unwrap_err();
    assert!(matches!(err, Error::Broker { .. }), "{err}");
    assert_eq!(err.broker_code(), Some(fb::ErrorCode::NotAllowed));
    let err = producer.rewrite_chunk(uuid(1_000), &c.location, &c.frames).await.unwrap_err();
    assert_eq!(err.broker_code(), Some(fb::ErrorCode::NotAllowed));
}
