//! Durable copies retain derived payload identity/provenance; raw chunk absorption stays a separate operation.

use std::path::Path;
use std::time::Duration;

use continuum_broker::server::{Config, Server};
use continuum_catalog::{Catalog, NewPayload, RedbCatalog, Resolved, StreamInfo, StreamKind};
use continuum_core::schema::continuum as fb;
use continuum_core::{DomainLayer, HeaderBuf, HeaderSpec, PayloadFormat, StorageLocator, StorageTier, StreamId, Uuid};
use continuum_wire::{read_frame, write_frame};
use flatbuffers::{FlatBufferBuilder, UnionWIPOffset, WIPOffset};
use tokio::net::UnixStream;
use tokio::sync::oneshot;

fn envelope(
    request_id: u64,
    build: impl FnOnce(&mut FlatBufferBuilder<'static>) -> (fb::ControlPayload, WIPOffset<UnionWIPOffset>),
) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let (payload_type, payload) = build(&mut b);
    let env = fb::ControlEnvelope::create(
        &mut b,
        &fb::ControlEnvelopeArgs { request_id, payload_type, payload: Some(payload) },
    );
    b.finish(env, None);
    b.finished_data().to_vec()
}

async fn request(stream: &mut UnixStream, id: u64, bytes: &[u8]) -> Vec<u8> {
    write_frame(stream, bytes).await.unwrap();
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let mut bytes = Vec::new();
            read_frame(stream, &mut bytes).await.unwrap();
            if flatbuffers::root::<fb::ControlEnvelope<'_>>(&bytes).unwrap().request_id() == id {
                return bytes;
            }
        }
    })
    .await
    .unwrap()
}

async fn connect(path: &Path) -> UnixStream {
    let mut stream = UnixStream::connect(path).await.unwrap();
    let bytes = envelope(1, |b| {
        let name = b.create_string("derived-copy-test");
        let schema = fb::SchemaVersion::create(
            b,
            &fb::SchemaVersionArgs { major: continuum_core::SCHEMA_MAJOR, minor: continuum_core::SCHEMA_MINOR },
        );
        let hello = fb::Hello::create(
            b,
            &fb::HelloArgs { client_name: Some(name), schema: Some(schema), ..Default::default() },
        );
        (fb::ControlPayload::Hello, hello.as_union_value())
    });
    let bytes = request(&mut stream, 1, &bytes).await;
    assert!(flatbuffers::root::<fb::ControlEnvelope<'_>>(&bytes).unwrap().payload_as_welcome().is_some());
    stream
}

async fn worker(stream: &mut UnixStream, keeps_copies: bool) {
    let bytes = envelope(2, |b| {
        let name = b.create_string(if keeps_copies { "copy-keeper" } else { "compute-only" });
        let worker = fb::RegisterWorker::create(
            b,
            &fb::RegisterWorkerArgs { worker_id: Some(name), keeps_copies, ..Default::default() },
        );
        (fb::ControlPayload::RegisterWorker, worker.as_union_value())
    });
    let bytes = request(stream, 2, &bytes).await;
    assert!(flatbuffers::root::<fb::ControlEnvelope<'_>>(&bytes).unwrap().payload_as_worker_registered().is_some());
}

fn copy(id: u64, payload: Uuid, location: &StorageLocator) -> Vec<u8> {
    envelope(id, |b| {
        let payload = fb::Uuid(payload.0);
        let location = location.build_into(b);
        let body = fb::CommitPayloadCopy::create(
            b,
            &fb::CommitPayloadCopyArgs { payload_id: Some(&payload), location: Some(location) },
        );
        (fb::ControlPayload::CommitPayloadCopy, body.as_union_value())
    })
}

async fn reopen(path: &Path) -> RedbCatalog {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        match RedbCatalog::open(path) {
            Ok(catalog) => return catalog,
            Err(error) => {
                assert!(tokio::time::Instant::now() < deadline, "catalog still held: {error}");
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        }
    }
}

fn error(bytes: &[u8], code: fb::ErrorCode) {
    let env = flatbuffers::root::<fb::ControlEnvelope<'_>>(bytes).unwrap();
    assert_eq!(env.payload_as_error().expect("refusal").code(), code);
}

fn info(name: &str, kind: StreamKind) -> StreamInfo {
    StreamInfo {
        name: name.into(),
        kind,
        layer: DomainLayer::Layer2Decision,
        format: PayloadFormat::RawBytes,
        shape_name: String::new(),
        shape_version: 0,
        rate_hz: None,
        anchor: None,
    }
}

fn header(id: Uuid, stream: StreamId, parent: Option<Uuid>) -> HeaderBuf {
    let mut spec = HeaderSpec::raw(id, stream, 10, 20);
    spec.parents = parent.into_iter().collect();
    HeaderBuf::build(&spec).unwrap()
}

#[tokio::test]
async fn durable_derived_copy_is_worker_only_idempotent_and_preserves_query_identity_across_restart() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("catalog.redb");
    let socket = dir.path().join("broker.sock");
    let catalog = RedbCatalog::open(&path).unwrap();
    let sensor = catalog.register_stream(info("camera", StreamKind::Sensor)).unwrap();
    let derived = catalog.register_stream(info("embedding", StreamKind::Derived)).unwrap();
    let parent = Uuid([1; 16]);
    let payload = Uuid([2; 16]);
    let inline = Uuid([3; 16]);
    let original = header(payload, derived, Some(parent));
    let inserted = catalog
        .insert(vec![
            NewPayload { header: header(parent, sensor, None), location: None, coverage: None },
            NewPayload { header: original.clone(), location: None, coverage: None },
            NewPayload {
                header: {
                    let mut spec = HeaderSpec::raw(inline, derived, 30, 40);
                    spec.parents = vec![parent];
                    HeaderBuf::build(&spec).unwrap()
                },
                location: Some(StorageLocator::Inline { bytes: vec![1] }),
                coverage: None,
            },
        ])
        .unwrap();
    let seq = inserted[1].seq;
    let location = StorageLocator::Chunk {
        tier: StorageTier::Tier1Warm,
        uri: dir.path().join("copy.cnk").to_string_lossy().into_owned(),
        offset: 0,
        length: 131072,
    };
    drop(catalog);
    for restart in 0..2 {
        let mut config = Config::new(socket.clone(), Duration::from_secs(30));
        config.tier0 = Some(vec![continuum_sched::SlotClass { size_bytes: 16_384, count: 2 }]);
        config.tier0_ring_bytes = 0;
        let server = Server::bind_with_catalog(config, Box::new(RedbCatalog::open(&path).unwrap())).await.unwrap();
        let (stop, stopped) = oneshot::channel();
        let task = tokio::spawn(server.run_until(async {
            let _ = stopped.await;
        }));
        let mut client = connect(&socket).await;
        if restart == 0 {
            error(&request(&mut client, 3, &copy(3, payload, &location)).await, fb::ErrorCode::NotAllowed);
            worker(&mut client, false).await;
            error(&request(&mut client, 4, &copy(4, payload, &location)).await, fb::ErrorCode::NotAllowed);
            drop(client);
            client = connect(&socket).await;
        }
        worker(&mut client, true).await;
        if restart == 0 {
            error(&request(&mut client, 5, &copy(5, parent, &location)).await, fb::ErrorCode::BadRequest);
            error(&request(&mut client, 6, &copy(6, inline, &location)).await, fb::ErrorCode::BadRequest);
            error(&request(&mut client, 7, &copy(7, Uuid([9; 16]), &location)).await, fb::ErrorCode::NotFound);
            let cold =
                StorageLocator::Chunk { tier: StorageTier::Tier2Cold, uri: "cold.cnk".into(), offset: 0, length: 1 };
            error(&request(&mut client, 8, &copy(8, payload, &cold)).await, fb::ErrorCode::BadRequest);
        }
        for id in [9, 10] {
            let bytes = request(&mut client, id, &copy(id, payload, &location)).await;
            let env = flatbuffers::root::<fb::ControlEnvelope<'_>>(&bytes).unwrap();
            assert_eq!(env.payload_as_payload_copy_committed().unwrap().payload_id().unwrap().0, payload.0);
        }
        let query = envelope(12, |b| {
            let stream = b.create_string("embedding");
            let range = fb::TimeRange::new(10, 20);
            let query = fb::Query::create(
                b,
                &fb::QueryArgs { stream: Some(stream), range: Some(&range), ..Default::default() },
            );
            (fb::ControlPayload::Query, query.as_union_value())
        });
        let bytes = request(&mut client, 12, &query).await;
        let env = flatbuffers::root::<fb::ControlEnvelope<'_>>(&bytes).unwrap();
        let items = env.payload_as_query_result().unwrap().items().unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items.get(0).header_bytes().unwrap().bytes(), original.as_bytes());
        assert_eq!(StorageLocator::from_table(&items.get(0).locator().unwrap()).unwrap(), location);
        let other =
            StorageLocator::Chunk { tier: StorageTier::Tier1Warm, uri: "other.cnk".into(), offset: 0, length: 1 };
        error(&request(&mut client, 11, &copy(11, payload, &other)).await, fb::ErrorCode::Conflict);
        drop(client);
        stop.send(()).unwrap();
        task.await.unwrap();
        let catalog = reopen(&path).await;
        let Some(Resolved::Payload(known)) = catalog.get(payload).unwrap() else { panic!("original disappeared") };
        assert_eq!(known.header, original);
        assert_eq!(known.commit_seq, seq);
        assert_eq!(known.locations, vec![location.clone()]);
        assert!(known.frames.is_none());
        assert_eq!(catalog.parents(payload).unwrap(), vec![parent]);
        assert_eq!(catalog.children(parent).unwrap(), vec![payload, inline]);
        assert!(catalog.keepers().unwrap().contains(&"copy-keeper".into()));
        assert_eq!(catalog.head_seq().unwrap(), 3);
    }
}
