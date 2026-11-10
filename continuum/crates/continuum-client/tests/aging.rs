//! Aging against the real broker (docs/CONTINUUM-STORAGE.md): an age-out of whole payloads goes only by the commit
//! clock, and its spans read as aged out. Also frame resolution and the catalog's page stats.

use std::time::Duration;

use continuum_broker::server::{Config, Server};
use continuum_client::{
    ChunkFrame, Client, ClientOptions, FrameSource, MAX_RESOLVE_IDS, NewChunk, PurgeKind, StreamSpec, StreamStorage,
    WorkerSpec,
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

fn now_ns() -> u64 {
    u64::try_from(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()).unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_chunk_ages_out_whole_by_the_commit_clock_and_its_span_reads_as_aged() {
    let b = broker().await;
    let (_producer, tier, cam) = setup(&b, 5).await;
    let absorbed: Vec<Uuid> = (0..5).map(uuid).collect();
    let warm = StorageLocator::Chunk { tier: StorageTier::Tier1Warm, uri: "/t1/a.cnk".into(), offset: 0, length: 3000 };
    let mut c = chunk(cam, 5, "/t1/a.cnk");
    c.location = warm.clone();
    tier.commit_chunk(&c, &absorbed).await.unwrap();
    // The commit clock moves with the writer's clock sample.
    b.core.upkeep_now().await.unwrap();
    b.core.sync_writer().await.unwrap();
    let cutoff = now_ns();
    assert!(tier.commit_seq_at(cutoff).await.unwrap() > 0);
    assert_eq!(tier.commit_seq_at(1).await.unwrap(), 0);

    let info = tier.get_chunk(uuid(1_000)).await.unwrap();
    assert_eq!((info.stream, info.frames.len(), info.locations.clone()), (cam, 5, vec![warm]));
    let end = T0 + 4 * FRAME_NS + 1;
    assert!(tier.aged_out("cam0", (T0, end), 0).await.unwrap().is_empty());

    // Too early by the commit clock: nothing goes.
    tier.age_out(&[uuid(1_000)], 1, 0, "aged out", "{}").await.unwrap();
    assert!(tier.get_chunk(uuid(1_000)).await.is_ok(), "committed after the cutoff");

    // The whole chunk, aged out by the commit clock: nothing derived goes, and its record says it was an age-out.
    let done = tier.age_out(&[uuid(1_000)], cutoff, 0, "aged out", "{}").await.unwrap();
    assert_eq!(done.removed.iter().map(|r| r.2).sum::<u64>(), 1);
    let gone = tier.get_chunk(uuid(1_000)).await.unwrap_err();
    assert_eq!(gone.broker_code(), Some(fb::ErrorCode::NotFound));
    let purges = tier.list_purges(None, 0).await.unwrap();
    assert_eq!(purges.last().map(|p| p.kind), Some(PurgeKind::AgeOut));
    assert_eq!(tier.aged_out("cam0", (T0, end), 0).await.unwrap(), vec![(T0, end)]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resolving_frames_names_the_chunk_each_absorbed_frame_is_in() {
    let b = broker().await;
    let (producer, tier, cam) = setup(&b, 5).await;
    // The chunk absorbs frames 0 to 2; frames 3 and 4 stay payloads of their own.
    let absorbed: Vec<Uuid> = (0..3).map(uuid).collect();
    tier.commit_chunk(&chunk(cam, 3, "/t1/a.cnk"), &absorbed).await.unwrap();
    let chunk_seq = tier.get_chunk(uuid(1_000)).await.unwrap().seq;
    let asked = [uuid(2), uuid(0), uuid(4), uuid(999)];
    let frames = producer.resolve_frames(&asked).await.unwrap();
    let sources: Vec<_> = frames.iter().map(|f| f.source).collect();
    let FrameSource::Payload { stream, seq } = sources[2] else { panic!("frame 4 is a payload: {:?}", sources[2]) };
    assert_eq!(stream, cam);
    assert!(seq > 0);
    assert_eq!(
        sources,
        [
            FrameSource::Absorbed {
                chunk: uuid(1_000),
                stream: cam,
                seq: chunk_seq,
                frame_index: 2,
                start_ns: T0 + 2 * FRAME_NS
            },
            FrameSource::Absorbed { chunk: uuid(1_000), stream: cam, seq: chunk_seq, frame_index: 0, start_ns: T0 },
            sources[2],
            FrameSource::Unknown,
        ]
    );
    assert_eq!(frames.iter().map(|f| f.id).collect::<Vec<_>>(), asked, "one answer per id, in the asked order");
    assert!(producer.resolve_frames(&[]).await.unwrap().is_empty());
    let too_many: Vec<Uuid> = (0..=u32::try_from(MAX_RESOLVE_IDS).unwrap()).map(uuid).collect();
    assert!(matches!(producer.resolve_frames(&too_many).await, Err(continuum_client::Error::Protocol(_))));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn catalog_stats_read_the_file_catalogs_tables_through_the_broker() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("b.sock");
    let mut config = Config::new(socket.clone(), Duration::from_secs(30));
    config.catalog = Some(dir.path().join("catalog.redb"));
    let server = Server::bind(config).await.unwrap();
    let core = server.core();
    let (stop, stopped) = oneshot::channel::<()>();
    tokio::spawn(server.run_until(async {
        let _ = stopped.await;
    }));
    let b = Broker { _dir: dir, socket, core, stop: Some(stop) };
    let (producer, _, _) = setup(&b, 3).await;
    b.core.sync_writer().await.unwrap();
    let stats = producer.catalog_stats().await.unwrap();
    assert_eq!(stats.page_size, 4096);
    assert!(stats.disk_bytes > 0, "{stats:?}");
    let streams = stats.tables.iter().find(|t| t.name == "streams").expect("the streams table");
    assert_eq!(streams.rows, 1, "cam0");
    assert!(stats.tables.iter().any(|t| t.name == "payloads" && t.rows == 3), "{:?}", stats.tables);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_memory_catalog_has_no_stats_to_give() {
    let b = broker().await;
    let (producer, _, _) = setup(&b, 1).await;
    assert_eq!(producer.catalog_stats().await.unwrap(), continuum_client::CatalogStats::default());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_dependent_ingest_goes_with_a_purge_of_what_it_depends_on_and_a_stale_revision_is_refused() {
    use continuum_client::{Dependency, PurgeTarget};
    let b = broker().await;
    let (producer, _, cam) = setup(&b, 4).await;
    let spec = StreamSpec {
        name: "progress".into(),
        kind: fb::StreamKind::Derived,
        format: fb::PayloadFormat::Text,
        shape_name: "progress".into(),
        shape_version: 1,
        nominal_rate_hz: 0.0,
        storage: StreamStorage::Slab { class: 0 },
        anchor: None,
        layer: fb::DomainLayer::Layer3Interpretation,
    };
    let progress = StreamId(producer.register_stream(&spec).await.unwrap().stream_id);
    let revision = |streams: &[continuum_client::StreamDesc]| {
        streams.iter().find(|s| s.id == cam).and_then(|s| s.invalidation_revision).unwrap()
    };
    let read = revision(&producer.list_streams().await.unwrap());
    // A stretch over the first two frames (T0 and T0 + FRAME_NS).
    let dep = Dependency { stream_id: cam.0, start_ns: T0, end_ns: T0 + 2 * FRAME_NS, revision: read };
    let stretch = |n: u32| {
        let mut spec = HeaderSpec::raw(uuid(500 + n), progress, T0 + 1, T0 + 1);
        spec.seq = n;
        HeaderBuf::build(&spec).unwrap()
    };
    let at = StorageLocator::inline(b"{}".to_vec());
    let seq = producer.ingest_dependent(&stretch(0), at.as_ref(), &[dep]).await.unwrap();
    assert!(seq > 0);
    // The same notice again is a repeat; without its dependency it's a Conflict, not a weaker proof acknowledged.
    assert_eq!(producer.ingest_dependent(&stretch(0), at.as_ref(), &[dep]).await.unwrap(), seq);
    let weaker = producer.ingest_acked(&stretch(0), at.as_ref()).await.unwrap_err();
    assert_eq!(weaker.broker_code(), Some(fb::ErrorCode::Conflict));
    // A purge of the second frame takes the stretch.
    producer.purge(&PurgeTarget::Ids(vec![uuid(1)]), "privacy", "{}").await.unwrap();
    b.core.sync_writer().await.unwrap();
    let (tx, rx) = tokio::sync::oneshot::channel();
    let job: continuum_broker::core::ReadJob = Box::new(move |catalog| {
        let _ = tx.send(catalog.get(uuid(500)).unwrap().is_some());
    });
    b.core.read(job).await.unwrap();
    assert!(!rx.await.unwrap(), "the stretch went with the frame it depended on");
    // A dependency read before that purge is refused: the purge moved the revision.
    let stale = producer.ingest_dependent(&stretch(1), at.as_ref(), &[dep]).await.unwrap_err();
    assert_eq!(stale.broker_code(), Some(fb::ErrorCode::Conflict));
    let fresh = Dependency { revision: revision(&producer.list_streams().await.unwrap()), ..dep };
    assert!(producer.ingest_dependent(&stretch(1), at.as_ref(), &[fresh]).await.unwrap() > 0);
}
