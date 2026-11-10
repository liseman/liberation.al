//! Worker state against the real broker and a file catalog (#1547): a checkpoint is one row, replaced in place, kept
//! only while its inputs' revisions hold, and deleted by a purge of one of them.

use std::time::Duration;

use continuum_broker::server::{Config, Server};
use continuum_client::{Client, ClientOptions, Error, PurgeTarget, StreamSpec, StreamStorage, WorkerState};
use continuum_core::schema::continuum::{self as fb, ErrorCode};
use continuum_core::{HeaderBuf, HeaderSpec, PayloadFormat, StreamId, Uuid};
use tokio::sync::oneshot;

const T0: u64 = 1_700_000_000_000_000_000;

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
    let mut config = Config::new(socket.clone(), Duration::from_secs(30));
    config.catalog = Some(dir.path().join("catalog.redb"));
    let server = Server::bind(config).await.unwrap();
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

/// A client that registered "cam0" and ingested two frames, committed.
async fn setup(b: &Broker) -> Client {
    let client = Client::connect(&b.socket, ClientOptions::new("worker")).await.unwrap();
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
    let cam = StreamId(client.register_stream(&spec).await.unwrap().stream_id);
    for i in 0..2u32 {
        let mut h = HeaderSpec::raw(uuid(i), cam, T0 + u64::from(i), T0 + u64::from(i));
        h.seq = i;
        h.format = PayloadFormat::EncodedVideo;
        h.shape_name = "MJPEG".into();
        h.shape_version = 1;
        client.ingest(&HeaderBuf::build(&h).unwrap(), None).unwrap();
    }
    client.heartbeat(None).await.unwrap();
    b.core.sync_writer().await.unwrap();
    client
}

async fn revision(client: &Client) -> u64 {
    let streams = client.list_streams().await.unwrap();
    streams.iter().find(|s| s.name == "cam0").and_then(|s| s.invalidation_revision).unwrap()
}

fn state(revision: u64, position: u64, body: &[u8]) -> WorkerState {
    WorkerState {
        owner: "see-activity".into(),
        contract: "activity-v6".into(),
        position,
        created_ns: T0 + position,
        inputs: vec![("cam0".into(), revision)],
        body: body.to_vec(),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_checkpoint_is_one_row_replaced_in_place_and_gone_after_a_purge_of_its_input() {
    let b = broker().await;
    let client = setup(&b).await;
    let before = revision(&client).await;
    assert_eq!(client.load_worker_state("see-activity", "activity-v6").await.unwrap(), None);
    client.save_worker_state(&state(before, 1, b"first")).await.unwrap();
    let second = state(before, 2, &vec![9; 1 << 20]);
    client.save_worker_state(&second).await.unwrap();
    assert_eq!(client.load_worker_state("see-activity", "activity-v6").await.unwrap(), Some(second));

    let done = client.purge(&PurgeTarget::Ids(vec![uuid(0)]), "test", "").await.unwrap();
    drop(done);
    b.core.sync_writer().await.unwrap();
    let after = revision(&client).await;
    assert!(after > before, "the purge advanced cam0's revision");
    assert_eq!(client.load_worker_state("see-activity", "activity-v6").await.unwrap(), None);
    assert!(!client.forget_worker_state("see-activity", "activity-v6").await.unwrap(), "the purge deleted it");

    let stale = client.save_worker_state(&state(before, 3, b"read before the purge")).await;
    assert!(matches!(stale, Err(Error::Broker { code: ErrorCode::Conflict, .. })), "{stale:?}");
    client.save_worker_state(&state(after, 3, b"read after")).await.unwrap();
    assert!(client.forget_worker_state("see-activity", "activity-v6").await.unwrap());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_checkpoint_out_of_bounds_is_a_bad_request() {
    let b = broker().await;
    let client = setup(&b).await;
    let rev = revision(&client).await;
    let big = client.save_worker_state(&state(rev, 1, &vec![0; (8 << 20) + 1])).await;
    assert!(matches!(big, Err(Error::Broker { code: ErrorCode::BadRequest, .. })), "{big:?}");
    let mut none = state(rev, 1, b"x");
    none.inputs.clear();
    let none = client.save_worker_state(&none).await;
    assert!(matches!(none, Err(Error::Broker { code: ErrorCode::BadRequest, .. })), "{none:?}");
}
