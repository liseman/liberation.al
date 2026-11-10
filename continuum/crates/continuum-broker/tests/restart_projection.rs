//! Planned restarts restore an exact durable projection and clear old arena locators.
use std::path::Path;
use std::time::Duration;

use continuum_broker::{
    index,
    server::{Config, Server},
};
use continuum_catalog::{Catalog, ChunkFrame, NewChunk, NewPayload, RedbCatalog, StreamInfo, StreamKind};
use continuum_client::{Client, ClientOptions, Start, StreamSpec, StreamStorage, SubEvent, SubscribeSpec};
use continuum_core::schema::continuum as fb;
use continuum_core::{
    DomainLayer, HeaderBuf, HeaderSpec, Interval, PAGE_ALIGN, PayloadFormat, StorageLocator, StorageTier, StreamId,
    Uuid,
};
use continuum_sched::SlotClass;

fn id(n: u32) -> Uuid {
    let mut bytes = [0; 16];
    bytes[6] = 0x70;
    bytes[8] = 0x80;
    bytes[12..].copy_from_slice(&n.to_be_bytes());
    Uuid(bytes)
}

fn config(socket: &Path, catalog: &Path) -> Config {
    let mut config = Config::new(socket.to_owned(), Duration::from_secs(5));
    config.catalog = Some(catalog.to_owned());
    config.tier0 = Some(vec![SlotClass { size_bytes: PAGE_ALIGN as u64, count: 8 }]);
    config.tier0_ring_bytes = 64 * 1024;
    config
}

fn seed(path: &Path) -> StreamId {
    let catalog = RedbCatalog::open(path).unwrap();
    let stream = catalog
        .register_stream(StreamInfo {
            name: "synthetic.history".into(),
            kind: StreamKind::Derived,
            layer: DomainLayer::Layer1Raw,
            format: PayloadFormat::RawBytes,
            shape_name: String::new(),
            shape_version: 0,
            rate_hz: None,
            anchor: None,
        })
        .unwrap();
    catalog
        .insert(vec![NewPayload {
            header: HeaderBuf::build(&HeaderSpec::raw(id(1), stream, 10, 20)).unwrap(),
            location: Some(StorageLocator::Inline { bytes: vec![1, 2, 3] }),
            coverage: None,
        }])
        .unwrap();
    catalog.close(Duration::from_secs(1)).unwrap();
    stream
}

async fn stop(stop: tokio::sync::oneshot::Sender<()>, running: tokio::task::JoinHandle<()>) {
    stop.send(()).unwrap();
    tokio::time::timeout(Duration::from_secs(20), running).await.unwrap().unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shutdown_with_connected_clients_seals_runtime_projection_and_restart_clears_exact_hot_ids() {
    let temp = tempfile::tempdir().unwrap();
    let socket = temp.path().join("broker.sock");
    let path = temp.path().join("catalog.redb");
    let history = seed(&path);
    let server = Server::bind(config(&socket, &path)).await.unwrap();
    let core = server.core();
    let (shutdown, ended) = tokio::sync::oneshot::channel();
    let running = tokio::spawn(server.run_until(async {
        let _ = ended.await;
    }));
    let client = Client::connect(&socket, ClientOptions::new("synthetic-producer")).await.unwrap();
    let stream = StreamId(
        client
            .register_stream(&StreamSpec {
                name: "synthetic.live".into(),
                kind: fb::StreamKind::Sensor,
                format: fb::PayloadFormat::RawBytes,
                shape_name: String::new(),
                shape_version: 0,
                nominal_rate_hz: 10.0,
                storage: StreamStorage::Slab { class: 0 },
                anchor: None,
                layer: fb::DomainLayer::Layer1Raw,
            })
            .await
            .unwrap()
            .stream_id,
    );
    let mapped = &client.arenas()[0];
    let mut slot = mapped.arena.claim(0, 1).unwrap();
    slot.as_mut_slice()[0] = 7;
    let offset = slot.offset() as u64;
    let handle = slot.publish(1).unwrap();
    let hot = StorageLocator::Shm {
        arena: mapped.info.name.clone(),
        arena_generation: mapped.arena.generation(),
        slab_class: handle.class,
        slot: handle.index,
        slot_generation: handle.generation,
        offset,
        length: 1,
    };
    client
        .ingest_acked(&HeaderBuf::build(&HeaderSpec::raw(id(2), stream, 100, 101)).unwrap(), Some(&hot))
        .await
        .unwrap();
    client
        .ingest_acked(
            &HeaderBuf::build(&HeaderSpec::raw(id(3), stream, 200, 201)).unwrap(),
            Some(&StorageLocator::Inline { bytes: vec![8] }),
        )
        .await
        .unwrap();
    assert!(!core.covered(stream, Interval::new(100, 101).unwrap()).await.unwrap().is_empty());
    // Leave all 17 clients connected. Server shutdown aborts socket tasks, so their async
    // Session::end cleanup is not guaranteed to run; the explicit core cleanup must cover them.
    let mut clients = vec![client];
    for _ in 0..16 {
        clients.push(Client::connect(&socket, ClientOptions::new("synthetic-open-client")).await.unwrap());
    }
    stop(shutdown, running).await;
    drop(clients);

    let catalog = RedbCatalog::open(&path).unwrap();
    let checkpoint = catalog
        .load_restart_projection(continuum_catalog::MAX_RESTART_PROJECTION_BYTES)
        .unwrap()
        .expect("a settled real broker must publish a checkpoint");
    assert_eq!(checkpoint.hot_ids, vec![id(2)]);
    let restored = index::AvailabilityIndex::restore_restart_projection(&checkpoint.bytes).unwrap();
    let fresh = index::load(&catalog, 2).unwrap().0;
    assert_eq!(restored.snapshot(), fresh.snapshot(), "runtime checkpoint must equal an independent catalog rebuild");
    assert_eq!(restored.stream_named("synthetic.live"), Some(stream));
    assert!(
        restored.covered(stream, Interval::new(100, 101).unwrap()).is_empty(),
        "old arena availability must never survive"
    );
    catalog.close(Duration::from_secs(1)).unwrap();
    drop(catalog);

    let server = Server::bind(config(&socket, &path)).await.unwrap();
    let core = server.core();
    assert_eq!(
        core.covered(history, Interval::new(10, 20).unwrap()).await.unwrap(),
        restored.covered(history, Interval::new(10, 20).unwrap())
    );
    assert!(core.covered(stream, Interval::new(100, 101).unwrap()).await.unwrap().is_empty());
    let (shutdown, ended) = tokio::sync::oneshot::channel();
    let running = tokio::spawn(server.run_until(async {
        let _ = ended.await;
    }));
    stop(shutdown, running).await;
    let catalog = RedbCatalog::open(&path).unwrap();
    assert!(
        catalog.range(stream, Interval::new(100, 101).unwrap(), None, 4).unwrap().items[0].locations.is_empty(),
        "restore must clear the obsolete locator itself"
    );
    assert!(
        !catalog.range(stream, Interval::new(200, 201).unwrap(), None, 4).unwrap().items[0].locations.is_empty(),
        "durable copies must remain"
    );
    // An immediate second stop can legitimately refuse a checkpoint while its initial frame
    // sweep is still pending. Publish an independent fresh projection to inspect the hot-ID
    // secondary index itself; this must be empty regardless of that optional shutdown cache.
    let fresh = index::load(&catalog, 2).unwrap().0;
    let stamp = catalog.restart_stamp().unwrap().unwrap();
    assert!(catalog.save_restart_projection(stamp, &fresh.restart_projection_bytes().unwrap()).unwrap());
    assert!(
        catalog
            .load_restart_projection(continuum_catalog::MAX_RESTART_PROJECTION_BYTES)
            .unwrap()
            .unwrap()
            .hot_ids
            .is_empty()
    );
    catalog.close(Duration::from_secs(1)).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_current_but_invalid_broker_codec_falls_back_to_real_catalog_contents() {
    let temp = tempfile::tempdir().unwrap();
    let socket = temp.path().join("broker.sock");
    let path = temp.path().join("catalog.redb");
    let stream = seed(&path);
    let catalog = RedbCatalog::open(&path).unwrap();
    let stamp = catalog.restart_stamp().unwrap().unwrap();
    assert!(catalog.save_restart_projection(stamp, b"not a broker projection").unwrap());
    catalog.close(Duration::from_secs(1)).unwrap();
    drop(catalog);
    let server = Server::bind(config(&socket, &path)).await.unwrap();
    assert!(
        !server.core().covered(stream, Interval::new(10, 20).unwrap()).await.unwrap().is_empty(),
        "invalid cache cannot become an empty successful index"
    );
    let (shutdown, ended) = tokio::sync::oneshot::channel();
    let running = tokio::spawn(server.run_until(async {
        let _ = ended.await;
    }));
    stop(shutdown, running).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn failed_ring_allocation_keeps_committed_stream_metadata_in_the_checkpoint() {
    let temp = tempfile::tempdir().unwrap();
    let socket = temp.path().join("broker.sock");
    let path = temp.path().join("catalog.redb");
    seed(&path);
    let server = Server::bind(config(&socket, &path)).await.unwrap();
    let (shutdown, ended) = tokio::sync::oneshot::channel();
    let running = tokio::spawn(server.run_until(async {
        let _ = ended.await;
    }));
    let client = Client::connect(&socket, ClientOptions::new("synthetic-no-room")).await.unwrap();
    let result = client
        .register_stream(&StreamSpec {
            name: "synthetic.no-room".into(),
            kind: fb::StreamKind::Sensor,
            format: fb::PayloadFormat::RawBytes,
            shape_name: String::new(),
            shape_version: 0,
            nominal_rate_hz: 1.0,
            storage: StreamStorage::Ring { stride_bytes: 4096, seconds: 1000.0 },
            anchor: None,
            layer: fb::DomainLayer::Layer1Raw,
        })
        .await;
    let error = result.unwrap_err().to_string();
    assert!(error.contains("ring region"), "registration must fail at allocation after catalog commit: {error}");
    client.goodbye("test complete").await.unwrap();
    stop(shutdown, running).await;
    let catalog = RedbCatalog::open(&path).unwrap();
    let id = catalog
        .streams(None, 10)
        .unwrap()
        .into_iter()
        .find_map(|(id, info)| (info.name == "synthetic.no-room").then_some(id))
        .expect("the stream metadata committed before allocation failed");
    let checkpoint = catalog
        .load_restart_projection(continuum_catalog::MAX_RESTART_PROJECTION_BYTES)
        .unwrap()
        .expect("failed allocation does not undo the committed metadata");
    let restored = index::AvailabilityIndex::restore_restart_projection(&checkpoint.bytes).unwrap();
    let fresh = index::load(&catalog, 2).unwrap().0;
    assert_eq!(restored.snapshot(), fresh.snapshot(), "allocation failure must not seal an incomplete runtime index");
    assert_eq!(restored.stream_named("synthetic.no-room"), Some(id));
    catalog.close(Duration::from_secs(1)).unwrap();
}

/// Seeds `chunks` chunks of 4000 absorbed frames each on `synthetic.history`: enough `CHUNK_FRAMES` rows that a sweep
/// at a budget of one row per step is still running when the broker stops (a large catalog's sweep may not finish
/// before stop).
fn seed_absorbed_frames(path: &Path, chunks: u32) {
    const FRAMES: u32 = 4_000;
    let catalog = RedbCatalog::open(path).unwrap();
    let stream = catalog.streams(None, 10).unwrap()[0].0;
    for c in 0..chunks {
        // Ids and times clear of `seed`'s payload, one block per chunk.
        let first = 1_000 + c * (FRAMES + 1);
        let at = |n: u32| 1_000 + u64::from(first + n);
        catalog
            .insert(
                (1..=FRAMES)
                    .map(|n| NewPayload {
                        header: HeaderBuf::build(&HeaderSpec::raw(id(first + n), stream, at(n), at(n) + 1)).unwrap(),
                        location: Some(StorageLocator::Inline { bytes: vec![1] }),
                        coverage: None,
                    })
                    .collect(),
            )
            .unwrap();
        let chunk = NewChunk {
            payload: NewPayload {
                header: HeaderBuf::build(&HeaderSpec::raw(id(first), stream, at(1), at(FRAMES) + 1)).unwrap(),
                location: Some(StorageLocator::Chunk {
                    tier: StorageTier::Tier1Warm,
                    uri: format!("chunk-{c}"),
                    offset: 0,
                    length: u64::from(FRAMES),
                }),
                coverage: None,
            },
            frames: (1..=FRAMES)
                .map(|n| ChunkFrame { start_ns: at(n), offset: u64::from(n - 1), length: 1, cut: false })
                .collect(),
        };
        catalog.commit_chunk(chunk, &(1..=FRAMES).map(|n| id(first + n)).collect::<Vec<_>>()).unwrap();
    }
    catalog.close(Duration::from_secs(1)).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_frame_sweep_running_at_stop_is_dropped_and_the_projection_still_publishes() {
    let temp = tempfile::tempdir().unwrap();
    let socket = temp.path().join("broker.sock");
    let path = temp.path().join("catalog.redb");
    seed(&path);
    seed_absorbed_frames(&path, 5);
    let mut config = config(&socket, &path);
    // One frame row per writer step, as a slow sweep runs: it can't finish before this test stops the broker.
    config.writer.initial_budget = 1;
    config.writer.max_budget = 1;
    let server = Server::bind(config.clone()).await.unwrap();
    let (shutdown, ended) = tokio::sync::oneshot::channel();
    let running = tokio::spawn(server.run_until(async {
        let _ = ended.await;
    }));
    // A durable subscriber that acknowledges right before the stop: its position must be saved, and saving it must
    // not keep the projection from publishing.
    let client = Client::connect(&socket, ClientOptions::new("synthetic-durable")).await.unwrap();
    let mut spec = SubscribeSpec::new(["synthetic.history"], Start::Saved);
    spec.durable_name = Some("synthetic-durable".into());
    let mut subscription = client.subscribe(&spec).await.unwrap();
    let position = loop {
        match tokio::time::timeout(Duration::from_secs(10), subscription.next()).await.unwrap().unwrap() {
            SubEvent::Event { position, .. } => break position,
            _ => continue,
        }
    };
    subscription.ack(position).unwrap();
    // The upkeep tick queues the sweep within 50 ms of start; give it time to be well under way.
    tokio::time::sleep(Duration::from_millis(300)).await;
    stop(shutdown, running).await;
    drop(subscription);
    drop(client);

    let catalog = RedbCatalog::open(&path).unwrap();
    let checkpoint = catalog
        .load_restart_projection(continuum_catalog::MAX_RESTART_PROJECTION_BYTES)
        .unwrap()
        .expect("a frame sweep still running at stop must be dropped, not block the projection");
    let restored = index::AvailabilityIndex::restore_restart_projection(&checkpoint.bytes).unwrap();
    let fresh = index::load(&catalog, 1024).unwrap().0;
    assert_eq!(restored.snapshot(), fresh.snapshot(), "a dropped sweep must leave the projection equal to a rebuild");
    assert!(catalog.cursor("synthetic-durable").unwrap().is_some(), "the acknowledged position must be saved");
    catalog.close(Duration::from_secs(1)).unwrap();
    drop(catalog);

    // The next start restores the projection instead of rebuilding, and its own sweep starts over.
    let server = Server::bind(config).await.unwrap();
    let (shutdown, ended) = tokio::sync::oneshot::channel();
    let running = tokio::spawn(server.run_until(async {
        let _ = ended.await;
    }));
    stop(shutdown, running).await;
}

/// A stop can skip the projection with "reads were outstanding" when a read for a client that
/// had already gone was still running on the reader pool. The fence waits for it within its budget.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_read_still_running_at_stop_is_waited_for_and_the_projection_publishes() {
    let temp = tempfile::tempdir().unwrap();
    let socket = temp.path().join("broker.sock");
    let path = temp.path().join("catalog.redb");
    seed(&path);
    let server = Server::bind(config(&socket, &path)).await.unwrap();
    let core = server.core();
    let (shutdown, ended) = tokio::sync::oneshot::channel();
    let running = tokio::spawn(server.run_until(async {
        let _ = ended.await;
    }));
    let (started, began) = std::sync::mpsc::channel();
    core.read(Box::new(move |_| {
        started.send(()).unwrap();
        std::thread::sleep(Duration::from_millis(1500));
    }))
    .await
    .unwrap();
    began.recv_timeout(Duration::from_secs(5)).unwrap();
    stop(shutdown, running).await;
    let catalog = RedbCatalog::open(&path).unwrap();
    assert!(
        catalog.load_restart_projection(continuum_catalog::MAX_RESTART_PROJECTION_BYTES).unwrap().is_ok(),
        "a read finishing during the stop must not cost the projection"
    );
    catalog.close(Duration::from_secs(1)).unwrap();
}
