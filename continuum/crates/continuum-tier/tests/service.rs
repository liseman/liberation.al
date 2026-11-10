//! The tier worker program against a real broker, in-process: a producer writes frames into the broker's Tier 0
//! arena and ingests them, the service follows the stream with its durable, pinned subscription, and every frame
//! ends up in a committed Tier 1 chunk the catalog compacted it into. A second stream it follows isn't registered,
//! which doesn't hold the first back. Short and bounded: 3 seconds of 30 fps, with deadlines, not fixed sleeps.

use std::time::{Duration, Instant};

use continuum_broker::server::{Config, Server};
use continuum_catalog::{Catalog, RedbCatalog};
use continuum_client::{Client, ClientOptions, StreamSpec, StreamStorage};
use continuum_core::schema::continuum as fb;
use continuum_core::{HeaderBuf, HeaderSpec, Interval, PayloadFormat, StorageLocator, StreamId, Uuid};
use continuum_tier::service::{Service, ServiceError, Settings};
use continuum_tier::{FlushSettings, hydrate};

const FRAMES: u32 = 90;
const FRAME_NS: u64 = 33_333_333;

fn frame_bytes(i: u32) -> Vec<u8> {
    let mut v = vec![0xffu8, 0xd8];
    v.extend((0..900).map(|k| u8::try_from((k + i as usize) % 251).expect("small")));
    v
}

fn utc_now() -> u64 {
    u64::try_from(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()).unwrap()
}

#[test]
fn every_frame_a_producer_writes_into_tier_0_reaches_a_committed_chunk() {
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(3).enable_all().build().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("broker.sock");
    let catalog_path = dir.path().join("catalog.redb");
    let mut config = Config::new(socket.clone(), Duration::from_secs(30));
    config.tier0 = Some(vec![continuum_sched::SlotClass { size_bytes: 16 * 1024, count: 256 }]);
    let server =
        rt.block_on(Server::bind_with_catalog(config, Box::new(RedbCatalog::open(&catalog_path).unwrap()))).unwrap();
    let (stop_broker, stopped) = tokio::sync::oneshot::channel::<()>();
    let broker = rt.spawn(server.run_until(async {
        let _ = stopped.await;
    }));

    // The producer registers the stream before the worker subscribes to it by name.
    let producer = rt.block_on(Client::connect(&socket, ClientOptions::new("producer"))).unwrap();
    let spec = StreamSpec {
        name: "cam0.video".into(),
        kind: fb::StreamKind::Sensor,
        format: fb::PayloadFormat::EncodedVideo,
        shape_name: "MJPEG".into(),
        shape_version: 1,
        nominal_rate_hz: 30.0,
        storage: StreamStorage::Slab { class: 0 },
        anchor: None,
        layer: fb::DomainLayer::Layer1Raw,
    };
    let cam = StreamId(rt.block_on(producer.register_stream(&spec)).unwrap().stream_id);

    let chunk_dir = dir.path().join("tier");
    std::fs::create_dir_all(&chunk_dir).unwrap();
    let quarantined_id = Uuid([77; 16]);
    let quarantine = chunk_dir.join(format!("{}-0-1.layout-unknown", cam.0));
    let mut orphan =
        continuum_tier::ChunkWriter::create(&quarantine, cam, continuum_tier::ChunkFormat::Lz4, 0).unwrap();
    orphan
        .append_payload(quarantined_id, 1, 1, 0, &continuum_tier::codecs::lz4_compress(&vec![128; 64 * 48 * 2]))
        .unwrap();
    orphan.close().unwrap();
    let preserved = std::fs::read(&quarantine).unwrap();
    let old_header = HeaderBuf::build(&HeaderSpec::raw(quarantined_id, cam, 1, 1)).unwrap();
    let quarantined_position =
        rt.block_on(producer.ingest_acked(&old_header, Some(&StorageLocator::Inline { bytes: vec![1] }))).unwrap();
    let settings = Settings {
        socket: socket.clone(),
        chunk_dir: chunk_dir.clone(),
        data_root: None,
        // The second stream isn't registered: the worker follows the first and tries the second again later.
        streams: vec!["cam0.video".into(), "cam9.video".into()],
        flush: FlushSettings {
            chunk_ns: 500_000_000,
            part_ns: 500_000_000,
            audio_chunk_ns: 500_000_000,
            sync_ns: 100_000_000,
        },
        tick: Duration::from_millis(20),
    };
    let mut service = Service::start(settings, rt.handle()).expect("start");

    // The producer runs on its own thread; the service steps on this one until it has every frame, or a deadline.
    let arena = producer.arenas()[0].arena.clone();
    let t0 = utc_now();
    let feeder = {
        let producer = std::sync::Arc::new(producer);
        let p = std::sync::Arc::clone(&producer);
        let handle = std::thread::spawn(move || {
            for i in 0..FRAMES {
                let (mut slot, _) = arena.claim_overwriting(0, 1).unwrap();
                let bytes = frame_bytes(i);
                slot.as_mut_slice()[..bytes.len()].copy_from_slice(&bytes);
                let offset = slot.offset() as u64;
                let handle = slot.publish(bytes.len()).unwrap();
                let id = Uuid(uuid::Uuid::now_v7().into_bytes());
                let start = t0 + u64::from(i) * FRAME_NS;
                let mut h = HeaderSpec::raw(id, cam, start, start);
                h.seq = i;
                h.format = PayloadFormat::EncodedVideo;
                h.shape_name = "MJPEG".into();
                h.shape_version = 1;
                let locator = StorageLocator::Shm {
                    arena: arena.name().to_owned(),
                    arena_generation: arena.generation(),
                    slab_class: 0,
                    slot: handle.index,
                    slot_generation: handle.generation,
                    offset,
                    length: bytes.len() as u64,
                };
                p.ingest(&HeaderBuf::build(&h).unwrap(), Some(&locator)).unwrap();
                std::thread::sleep(Duration::from_millis(33));
            }
        });
        (producer, handle)
    };
    let deadline = Instant::now() + Duration::from_secs(30);
    while service.counts().appended < u64::from(FRAMES) {
        assert!(Instant::now() < deadline, "the worker got {:?} of {FRAMES} frames", service.counts());
        service.step().expect("step");
    }
    feeder.1.join().unwrap();
    service.shutdown().expect("shutdown");
    let counts = service.counts();
    assert_eq!((counts.appended, counts.refused, counts.rejected), (u64::from(FRAMES), 0, 0), "{counts:?}");
    assert!(counts.committed >= 1, "{counts:?}");
    let producer = std::sync::Arc::try_unwrap(feeder.0).ok().unwrap();

    rt.block_on(producer.close("done")).unwrap();
    stop_broker.send(()).unwrap();
    rt.block_on(broker).unwrap();
    let reopen = Instant::now() + Duration::from_secs(10);
    let catalog = loop {
        match RedbCatalog::open(&catalog_path) {
            Ok(c) => break c,
            Err(e) => {
                assert!(Instant::now() < reopen, "the broker never closed its catalog: {e}");
                std::thread::sleep(Duration::from_millis(20));
            }
        }
    };
    let cursor = catalog.cursor("tier/cam0.video").unwrap().unwrap();
    assert!(
        u64::from_le_bytes(cursor.try_into().unwrap()) > quarantined_position,
        "durable quarantine must not pin later acknowledgements"
    );
    assert_eq!(std::fs::read(&quarantine).unwrap(), preserved);
    let Some(continuum_catalog::Resolved::Payload(old)) = catalog.get(quarantined_id).unwrap() else {
        panic!("quarantined payload must not be absorbed into a guessed chunk");
    };
    assert!(old.frames.is_none());
    assert!(old.locations.iter().all(|location| !matches!(location, StorageLocator::Chunk { .. })));
    // Only chunks remain on the stream, and between them they hold every frame, readable from their files.
    let page = catalog.range(cam, Interval::new(t0, t0 + 10_000_000_000).unwrap(), None, 100).unwrap();
    assert!(page.items.iter().all(|i| i.frames.is_some()), "every frame compacted into a chunk");
    let mut got = Vec::new();
    for item in &page.items {
        let Some(StorageLocator::Chunk { uri, .. }) = item.locations.first() else { panic!("a Tier 1 chunk") };
        assert!(uri.starts_with(chunk_dir.to_str().unwrap()), "{uri}");
        got.extend(hydrate(std::path::Path::new(uri), t0, t0 + 10_000_000_000).unwrap().into_iter().map(|h| h.bytes));
    }
    assert_eq!(got.len(), FRAMES as usize);
    assert_eq!(got, (0..FRAMES).map(frame_bytes).collect::<Vec<_>>());
}

#[test]
fn every_ring_batch_a_microphone_writes_reaches_a_committed_chunk() {
    const BLOCK: usize = 16 + 480 * 4; // the 16-byte stamp, then 10 ms of 48 kHz mono f32
    const BATCHES: u64 = 3;
    const PER_BATCH: u32 = 100;
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(3).enable_all().build().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("broker.sock");
    let catalog_path = dir.path().join("catalog.redb");
    let mut config = Config::new(socket.clone(), Duration::from_secs(30));
    config.tier0 = Some(vec![continuum_sched::SlotClass { size_bytes: 16 * 1024, count: 16 }]);
    let server =
        rt.block_on(Server::bind_with_catalog(config, Box::new(RedbCatalog::open(&catalog_path).unwrap()))).unwrap();
    let (stop_broker, stopped) = tokio::sync::oneshot::channel::<()>();
    let broker = rt.spawn(server.run_until(async {
        let _ = stopped.await;
    }));

    let producer = rt.block_on(Client::connect(&socket, ClientOptions::new("mic"))).unwrap();
    let spec = StreamSpec {
        name: "mic.audio".into(),
        kind: fb::StreamKind::Sensor,
        format: fb::PayloadFormat::RawBytes,
        shape_name: "PCM_F32LEx1".into(),
        shape_version: 1,
        nominal_rate_hz: 100.0,
        storage: StreamStorage::Ring { stride_bytes: BLOCK as u32, seconds: 10.0 },
        anchor: None,
        layer: fb::DomainLayer::Layer1Raw,
    };
    let registered = rt.block_on(producer.register_stream(&spec)).unwrap();
    let mic = StreamId(registered.stream_id);
    let lease = registered.lease.expect("a ring lease");
    let mut writer = producer.arena(&lease.arena).unwrap().carved_ring_writer(&lease.carved_ring().unwrap()).unwrap();

    let chunk_dir = dir.path().join("tier");
    let settings = Settings {
        socket: socket.clone(),
        chunk_dir: chunk_dir.clone(),
        data_root: None,
        streams: vec!["mic.audio".into()],
        flush: FlushSettings {
            chunk_ns: 500_000_000,
            part_ns: 500_000_000,
            audio_chunk_ns: 500_000_000,
            sync_ns: 100_000_000,
        },
        tick: Duration::from_millis(20),
    };
    let mut service = Service::start(settings, rt.handle()).expect("start");

    let record = |n: u64| {
        let mut r = vec![0u8; BLOCK];
        r[..8].copy_from_slice(&n.to_le_bytes());
        r[16..].iter_mut().enumerate().for_each(|(k, b)| *b = ((k as u64 + n) % 251) as u8);
        r
    };
    let t0 = utc_now();
    let mut written = Vec::new();
    for batch in 0..BATCHES {
        let mut first = None;
        for i in 0..PER_BATCH {
            let n = batch * u64::from(PER_BATCH) + u64::from(i);
            let r = record(n);
            let seq = writer.push(&r).unwrap();
            first.get_or_insert(seq);
            written.extend_from_slice(&r);
        }
        let start = t0 + batch * 1_000_000_000;
        let mut h = HeaderSpec::raw(Uuid(uuid::Uuid::now_v7().into_bytes()), mic, start, start + 1_000_000_000);
        h.format = PayloadFormat::RawBytes;
        h.shape_name = "PCM_F32LEx1".into();
        h.shape_version = 1;
        let locator = StorageLocator::Ring {
            arena: lease.arena.clone(),
            arena_generation: lease.arena_generation,
            ring: lease.ring_index,
            first_seq: first.unwrap(),
            count: PER_BATCH,
        };
        producer.ingest(&HeaderBuf::build(&h).unwrap(), Some(&locator)).unwrap();
    }
    let deadline = Instant::now() + Duration::from_secs(30);
    while service.counts().appended < BATCHES {
        assert!(Instant::now() < deadline, "the worker got {:?} of {BATCHES} batches", service.counts());
        service.step().expect("step");
    }
    service.shutdown().expect("shutdown");
    assert_eq!(service.counts().refused, 0);

    rt.block_on(producer.close("done")).unwrap();
    stop_broker.send(()).unwrap();
    rt.block_on(broker).unwrap();
    let reopen = Instant::now() + Duration::from_secs(10);
    let catalog = loop {
        match RedbCatalog::open(&catalog_path) {
            Ok(c) => break c,
            Err(e) => {
                assert!(Instant::now() < reopen, "the broker never closed its catalog: {e}");
                std::thread::sleep(Duration::from_millis(20));
            }
        }
    };
    let page = catalog.range(mic, Interval::new(t0, t0 + 10_000_000_000).unwrap(), None, 100).unwrap();
    assert!(page.items.iter().all(|i| i.frames.is_some()), "every batch compacted into a chunk");
    let mut got = Vec::new();
    for item in &page.items {
        let Some(StorageLocator::Chunk { uri, .. }) = item.locations.first() else { panic!("a Tier 1 chunk") };
        for h in hydrate(std::path::Path::new(uri), t0, t0 + 10_000_000_000).unwrap() {
            got.extend_from_slice(&h.bytes);
        }
    }
    assert_eq!(got.len(), written.len(), "every record of every batch");
    assert!(got == written, "the records, byte for byte");
}

#[test]
fn a_ring_carved_after_the_worker_subscribed_ends_its_run_and_the_next_run_in_the_process_reads_it() {
    // After the broker restarts, the worker can subscribe to a microphone before capture registers it again: the
    // catalog knows the stream, but the new broker has carved no ring for it, so the subscription has none, and the
    // ring capture carves next was never read.
    const BLOCK: usize = 16 + 480 * 4;
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(3).enable_all().build().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("broker.sock");
    let catalog_path = dir.path().join("catalog.redb");
    let start_broker = || {
        let mut config = Config::new(socket.clone(), Duration::from_secs(30));
        config.tier0 = Some(vec![continuum_sched::SlotClass { size_bytes: 16 * 1024, count: 16 }]);
        let catalog = loop {
            // the last broker's catalog is let go as it stops
            if let Ok(c) = RedbCatalog::open(&catalog_path) {
                break c;
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        let server = rt.block_on(Server::bind_with_catalog(config, Box::new(catalog))).unwrap();
        let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
        let running = rt.spawn(server.run_until(async {
            let _ = stopped.await;
        }));
        (stop, running)
    };
    let spec = StreamSpec {
        name: "mic.audio".into(),
        kind: fb::StreamKind::Sensor,
        format: fb::PayloadFormat::RawBytes,
        shape_name: "PCM_F32LEx1".into(),
        shape_version: 1,
        nominal_rate_hz: 100.0,
        storage: StreamStorage::Ring { stride_bytes: BLOCK as u32, seconds: 10.0 },
        anchor: None,
        layer: fb::DomainLayer::Layer1Raw,
    };
    // The stream, registered with the broker before it restarted.
    let (stop_first, first_broker) = start_broker();
    let earlier = rt.block_on(Client::connect(&socket, ClientOptions::new("mic"))).unwrap();
    rt.block_on(earlier.register_stream(&spec)).unwrap();
    rt.block_on(earlier.close("restarting")).unwrap();
    stop_first.send(()).unwrap();
    rt.block_on(first_broker).unwrap();
    let (stop_broker, broker) = start_broker();

    let settings = Settings {
        socket: socket.clone(),
        chunk_dir: dir.path().join("tier"),
        data_root: None,
        streams: vec!["mic.audio".into()],
        flush: FlushSettings {
            chunk_ns: 500_000_000,
            part_ns: 500_000_000,
            audio_chunk_ns: 500_000_000,
            sync_ns: 100_000_000,
        },
        tick: Duration::from_millis(20),
    };
    let mut service = Service::start(settings.clone(), rt.handle()).expect("start");

    // Capture registers it again: a ring carved after the worker subscribed, and a batch on it.
    let producer = rt.block_on(Client::connect(&socket, ClientOptions::new("mic"))).unwrap();
    let registered = rt.block_on(producer.register_stream(&spec)).unwrap();
    let lease = registered.lease.expect("a ring lease");
    let mut writer = producer.arena(&lease.arena).unwrap().carved_ring_writer(&lease.carved_ring().unwrap()).unwrap();
    let first = writer.push(&vec![1u8; BLOCK]).unwrap();
    let t0 = utc_now();
    let mut h =
        HeaderSpec::raw(Uuid(uuid::Uuid::now_v7().into_bytes()), StreamId(registered.stream_id), t0, t0 + 10_000_000);
    h.format = PayloadFormat::RawBytes;
    h.shape_name = "PCM_F32LEx1".into();
    h.shape_version = 1;
    let locator = StorageLocator::Ring {
        arena: lease.arena.clone(),
        arena_generation: lease.arena_generation,
        ring: lease.ring_index,
        first_seq: first,
        count: 1,
    };
    producer.ingest(&HeaderBuf::build(&h).unwrap(), Some(&locator)).unwrap();

    let deadline = Instant::now() + Duration::from_secs(30);
    let ended = loop {
        assert!(Instant::now() < deadline, "the worker kept going: {:?}", service.counts());
        match service.step() {
            Ok(()) => assert_eq!(service.counts().appended, 0, "read through a ring it can't have"),
            Err(e) => break e,
        }
    };
    assert!(matches!(ended, ServiceError::Lost(_)), "{ended}");
    assert!(continuum_tier::service::rides_through(&ended), "a cascade's lost run starts again in the process");

    // The next run, in the same process as the program now does: it subscribes with the ring, and the batch
    // the lost run never acknowledged reaches a chunk.
    service.shutdown().expect("the lost run shuts down");
    drop(service);
    let mut next = Service::start(settings, rt.handle()).expect("the next run starts");
    let deadline = Instant::now() + Duration::from_secs(30);
    while next.counts().appended == 0 {
        assert!(Instant::now() < deadline, "the next run never read the batch: {:?}", next.counts());
        next.step().expect("the next run reads the ring");
    }
    assert_eq!(next.counts().appended, 1);
    next.shutdown().unwrap();

    rt.block_on(producer.close("done")).unwrap();
    stop_broker.send(()).unwrap();
    rt.block_on(broker).unwrap();
}

#[test]
fn a_purge_reaches_the_chunk_files_and_is_acknowledged() {
    const N: u32 = 60;
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(3).enable_all().build().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("broker.sock");
    let catalog_path = dir.path().join("catalog.redb");
    let mut config = Config::new(socket.clone(), Duration::from_secs(30));
    config.tier0 = Some(vec![continuum_sched::SlotClass { size_bytes: 16 * 1024, count: 256 }]);
    config.forget.keeper_grace = Duration::from_millis(200);
    let server =
        rt.block_on(Server::bind_with_catalog(config, Box::new(RedbCatalog::open(&catalog_path).unwrap()))).unwrap();
    let (stop_broker, stopped) = tokio::sync::oneshot::channel::<()>();
    let broker = rt.spawn(server.run_until(async {
        let _ = stopped.await;
    }));

    let producer = std::sync::Arc::new(rt.block_on(Client::connect(&socket, ClientOptions::new("producer"))).unwrap());
    let spec = StreamSpec {
        name: "cam0.video".into(),
        kind: fb::StreamKind::Sensor,
        format: fb::PayloadFormat::EncodedVideo,
        shape_name: "MJPEG".into(),
        shape_version: 1,
        nominal_rate_hz: 30.0,
        storage: StreamStorage::Slab { class: 0 },
        anchor: None,
        layer: fb::DomainLayer::Layer1Raw,
    };
    let cam = StreamId(rt.block_on(producer.register_stream(&spec)).unwrap().stream_id);
    let chunk_dir = dir.path().join("tier");
    let settings = Settings {
        socket: socket.clone(),
        chunk_dir: chunk_dir.clone(),
        // Relative locators (#62): the purge must find them in its known map and rewrite them relative.
        data_root: Some(dir.path().to_path_buf()),
        streams: vec!["cam0.video".into()],
        flush: FlushSettings {
            chunk_ns: 300_000_000,
            part_ns: 300_000_000,
            audio_chunk_ns: 300_000_000,
            sync_ns: 100_000_000,
        },
        tick: Duration::from_millis(20),
    };
    let mut service = Service::start(settings, rt.handle()).expect("start");

    // Two seconds of frames, all at once (their times are what matter), then the worker drains them.
    let arena = producer.arenas()[0].arena.clone();
    let t0 = utc_now();
    for i in 0..N {
        let (mut slot, _) = arena.claim_overwriting(0, 1).unwrap();
        let bytes = frame_bytes(i);
        slot.as_mut_slice()[..bytes.len()].copy_from_slice(&bytes);
        let offset = slot.offset() as u64;
        let handle = slot.publish(bytes.len()).unwrap();
        let start = t0 + u64::from(i) * FRAME_NS;
        let mut h = HeaderSpec::raw(Uuid(uuid::Uuid::now_v7().into_bytes()), cam, start, start);
        h.seq = i;
        h.format = PayloadFormat::EncodedVideo;
        h.shape_name = "MJPEG".into();
        h.shape_version = 1;
        let locator = StorageLocator::Shm {
            arena: arena.name().to_owned(),
            arena_generation: arena.generation(),
            slab_class: 0,
            slot: handle.index,
            slot_generation: handle.generation,
            offset,
            length: bytes.len() as u64,
        };
        producer.ingest(&HeaderBuf::build(&h).unwrap(), Some(&locator)).unwrap();
    }
    let deadline = Instant::now() + Duration::from_secs(30);
    while service.counts().appended < u64::from(N) {
        assert!(Instant::now() < deadline, "the worker got {:?} of {N} frames", service.counts());
        service.step().expect("step");
    }
    // Let every chunk close and commit, so the purge takes the catalog path (RewriteChunk), not the set-aside one.
    let settle = Instant::now() + Duration::from_secs(30);
    while service.counts().committed == 0 || service.waiting_commits() > 0 {
        assert!(Instant::now() < settle, "the chunks never committed: {:?}", service.counts());
        service.step().expect("step");
    }
    assert_eq!(service.counts().rejected, 0);

    // Purge frames 10 to 19 (by time), from another task, while the worker keeps stepping.
    let (from, to) = (t0 + 10 * FRAME_NS, t0 + 20 * FRAME_NS);
    let p = std::sync::Arc::clone(&producer);
    let purged = rt.spawn(async move { p.purge_range((from, to), &["cam0.video".to_owned()], "test").await });
    let deadline = Instant::now() + Duration::from_secs(30);
    while service.purges() == 0 {
        assert!(Instant::now() < deadline, "the worker never carried out the purge");
        service.step().expect("step");
    }
    let done = rt.block_on(purged).unwrap().expect("purge");
    // Frames cut from a chunk aren't payloads removed, so the result names the purge but may count nothing.
    assert_ne!(done.purge_id, Uuid([0; 16]), "{done:?}");
    service.shutdown().expect("shutdown");

    // The files hold exactly the frames outside the purge, read back byte for byte.
    let mut kept = Vec::new();
    for (_, path) in continuum_tier::service::chunk_files(&chunk_dir, Some(cam)).unwrap() {
        kept.extend(hydrate(&path, t0, t0 + 10_000_000_000).unwrap().into_iter().map(|h| h.bytes));
    }
    let expected: Vec<Vec<u8>> = (0..N).filter(|i| !(10..20).contains(i)).map(frame_bytes).collect();
    assert_eq!(kept.len(), expected.len(), "only the frames outside the purge are left");
    assert!(kept == expected);

    drop(service);
    let producer = std::sync::Arc::try_unwrap(producer).ok().unwrap();
    rt.block_on(producer.close("done")).unwrap();
    stop_broker.send(()).unwrap();
    rt.block_on(broker).unwrap();

    // The catalog agrees: frames 10 to 19 are cut in their chunks, and every frame's offset is the file's.
    let reopen = Instant::now() + Duration::from_secs(10);
    let catalog = loop {
        match RedbCatalog::open(&catalog_path) {
            Ok(c) => break c,
            Err(e) => {
                assert!(Instant::now() < reopen, "the broker never closed its catalog: {e}");
                std::thread::sleep(Duration::from_millis(20));
            }
        }
    };
    let page = catalog.range(cam, Interval::new(t0, t0 + 10_000_000_000).unwrap(), None, 100).unwrap();
    let mut cut_starts = Vec::new();
    for item in &page.items {
        let Some(StorageLocator::Chunk { uri, length, .. }) = item.locations.first() else { panic!("a chunk") };
        assert!(uri.starts_with("tier/"), "stored relative to the data root, rewrite included: {uri}");
        let path = continuum_core::chunk_path::resolve(dir.path(), uri);
        assert_eq!(*length, std::fs::metadata(&path).unwrap().len(), "the catalog has the file's length");
        let file = continuum_tier::ChunkReader::open(&path).unwrap();
        for (f, r) in item.frames.as_ref().unwrap().iter().zip(file.records()) {
            assert_eq!((f.offset, f.cut), (r.offset, r.cut), "the catalog's frame index is the file's");
            if f.cut {
                cut_starts.push(f.start_ns);
            }
        }
    }
    cut_starts.sort_unstable();
    assert_eq!(cut_starts, (10..20u64).map(|i| t0 + i * FRAME_NS).collect::<Vec<_>>());
}

#[test]
fn derived_slot_payload_gets_a_durable_copy_without_absorbing_its_header() {
    derived_copy_case(false, false);
}

#[test]
fn unrelated_derived_payload_delivered_after_an_overlapping_purge_is_archived() {
    derived_copy_case(true, false);
}

#[test]
fn derived_stream_registered_after_tier_start_is_discovered_periodically() {
    derived_copy_case(false, true);
}

fn derived_copy_case(late_after_purge: bool, register_after_start: bool) {
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("broker.sock");
    let mut config = Config::new(socket.clone(), Duration::from_secs(30));
    config.tier0 = Some(vec![continuum_sched::SlotClass { size_bytes: 128 * 1024, count: 2 }]);
    let server = rt.block_on(Server::bind(config)).unwrap();
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let running = rt.spawn(server.run_until(async {
        let _ = stopped.await;
    }));
    let chunk_dir = dir.path().join("tier");
    let start_service = || {
        Service::start(
            Settings {
                socket: socket.clone(),
                chunk_dir: chunk_dir.clone(),
                data_root: None,
                streams: vec![],
                flush: FlushSettings::default(),
                tick: Duration::from_millis(5),
            },
            rt.handle(),
        )
        .unwrap()
    };
    let before_registration = register_after_start.then(&start_service);
    let producer =
        std::sync::Arc::new(rt.block_on(Client::connect(&socket, ClientOptions::new("derived-producer"))).unwrap());
    let stream = StreamId(
        rt.block_on(producer.register_stream(&StreamSpec {
            name: "derived.large".into(),
            kind: fb::StreamKind::Derived,
            format: fb::PayloadFormat::RawBytes,
            shape_name: "application/blob".into(),
            shape_version: 1,
            nominal_rate_hz: 0.0,
            storage: StreamStorage::Slab { class: 0 },
            anchor: None,
            layer: fb::DomainLayer::Layer3Interpretation,
        }))
        .unwrap()
        .stream_id,
    );
    let at = utc_now();
    let parent = Uuid(uuid::Uuid::now_v7().into_bytes());
    let mut parent_spec = HeaderSpec::raw(parent, stream, at - 2, at - 1);
    parent_spec.layer = fb::DomainLayer::Layer3Interpretation;
    rt.block_on(
        producer
            .ingest_acked(&HeaderBuf::build(&parent_spec).unwrap(), Some(&StorageLocator::Inline { bytes: vec![1] })),
    )
    .unwrap();
    let arena = producer.arenas()[0].arena.clone();
    let bytes = vec![73; 96 * 1024];
    let (mut slot, _) = arena.claim_overwriting(0, 1).unwrap();
    slot.as_mut_slice()[..bytes.len()].copy_from_slice(&bytes);
    let offset = slot.offset() as u64;
    let handle = slot.publish(bytes.len()).unwrap();
    let id = Uuid(uuid::Uuid::now_v7().into_bytes());
    let mut spec = HeaderSpec::raw(id, stream, at, at + 10);
    spec.layer = fb::DomainLayer::Layer3Interpretation;
    spec.shape_name = "application/blob".into();
    spec.seq = 9;
    spec.parents = vec![parent];
    spec.subjects = vec![Uuid([91; 16])];
    let header = HeaderBuf::build(&spec).unwrap();
    let hot = StorageLocator::Shm {
        arena: arena.name().into(),
        arena_generation: arena.generation(),
        slab_class: 0,
        slot: handle.index,
        slot_generation: handle.generation,
        offset,
        length: bytes.len() as u64,
    };
    rt.block_on(producer.ingest_acked(&header, Some(&hot))).unwrap();
    let mut service = before_registration.unwrap_or_else(start_service);
    let query = continuum_client::QuerySpec::hit_only("derived.large", (at, at + 11));
    let deadline = Instant::now()
        + if register_after_start {
            continuum_tier::service::RESUBSCRIBE + Duration::from_secs(2)
        } else {
            Duration::from_secs(2)
        };
    loop {
        service.step().unwrap();
        if arena.scan(0).unwrap().iter().any(|slot| slot.handle == handle && slot.flushed) {
            break;
        }
        assert!(Instant::now() < deadline, "derived slot payload never received an acknowledged durable Tier 1 copy");
    }
    // Readers deliberately prefer the hottest readable location. Lose Tier 0 only after the worker's flushed
    // flag proves the durable-copy acknowledgement, then require the original header through its warm copy.
    service.shutdown().unwrap();
    let _ = arena.reclaim(handle).unwrap();
    assert!(arena.lease(handle, Duration::from_millis(10)).is_err());
    let result = rt.block_on(producer.query_all(&query)).unwrap();
    assert_eq!(result.items.len(), 1, "copy must not add a second catalog payload");
    let item = &result.items[0];
    assert_eq!(item.header.as_bytes(), header.as_bytes(), "original provenance and sequence remain intact");
    let Some(StorageLocator::Chunk { uri, .. }) = &item.locator else { panic!("durable copy") };
    let mut reader = continuum_tier::ChunkReader::open(std::path::Path::new(uri)).unwrap();
    assert_eq!(reader.format(), continuum_tier::ChunkFormat::Passthrough);
    assert_eq!(reader.records().len(), 1);
    let record = &reader.records()[0];
    assert_eq!((record.id, record.start_ns, record.end_ns, record.seq), (id, at, at + 10, 9));
    assert_eq!(reader.read(0).unwrap(), bytes);
    drop(reader);
    if late_after_purge {
        let mut service = Service::start(
            Settings {
                socket: socket.clone(),
                chunk_dir: chunk_dir.clone(),
                data_root: None,
                streams: vec![],
                flush: FlushSettings::default(),
                tick: Duration::from_millis(5),
            },
            rt.handle(),
        )
        .unwrap();
        let caller = std::sync::Arc::clone(&producer);
        let purging = rt
            .spawn(async move { caller.purge(&continuum_client::PurgeTarget::Ids(vec![parent]), "test", "{}").await });
        let deadline = Instant::now() + Duration::from_secs(3);
        while service.purges() == 0 {
            assert!(Instant::now() < deadline, "parent purge was not acknowledged");
            service.step().unwrap();
        }
        rt.block_on(purging).unwrap().unwrap();
        assert!(!std::path::Path::new(uri).exists(), "the purged descendant's copy must be removed");
        let independent_parent = Uuid(uuid::Uuid::now_v7().into_bytes());
        let mut parent_spec = HeaderSpec::raw(independent_parent, stream, at - 2, at - 1);
        parent_spec.layer = fb::DomainLayer::Layer3Interpretation;
        rt.block_on(
            producer.ingest_acked(
                &HeaderBuf::build(&parent_spec).unwrap(),
                Some(&StorageLocator::Inline { bytes: vec![2] }),
            ),
        )
        .unwrap();
        let late_bytes = vec![74; 96 * 1024];
        let (mut slot, _) = arena.claim_overwriting(0, 1).unwrap();
        slot.as_mut_slice()[..late_bytes.len()].copy_from_slice(&late_bytes);
        let offset = slot.offset() as u64;
        let late_handle = slot.publish(late_bytes.len()).unwrap();
        let late_id = Uuid(uuid::Uuid::now_v7().into_bytes());
        let mut late_spec = spec.clone();
        late_spec.id = late_id;
        late_spec.seq = 10;
        late_spec.parents = vec![independent_parent];
        late_spec.subjects = vec![Uuid([92; 16])];
        let late_header = HeaderBuf::build(&late_spec).unwrap();
        let hot = StorageLocator::Shm {
            arena: arena.name().into(),
            arena_generation: arena.generation(),
            slab_class: 0,
            slot: late_handle.index,
            slot_generation: late_handle.generation,
            offset,
            length: late_bytes.len() as u64,
        };
        rt.block_on(producer.ingest_acked(&late_header, Some(&hot))).unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            service.step().unwrap();
            if arena.scan(0).unwrap().iter().any(|slot| slot.handle == late_handle && slot.flushed) {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "unrelated late Derived payload sharing the purged span was not archived"
            );
        }
        service.shutdown().unwrap();
        let _ = arena.reclaim(late_handle).unwrap();
        let mut query = continuum_client::QuerySpec::hit_only("derived.large", (at, at + 11));
        query.ids = vec![late_id];
        let result = rt.block_on(producer.query_all(&query)).unwrap();
        assert_eq!(result.items.len(), 1);
        assert_eq!(result.items[0].header.as_bytes(), late_header.as_bytes());
        let Some(StorageLocator::Chunk { uri, .. }) = &result.items[0].locator else { panic!("late durable copy") };
        assert_eq!(continuum_tier::ChunkReader::open(std::path::Path::new(uri)).unwrap().read(0).unwrap(), late_bytes);
    }
    rt.block_on(producer.close("done")).unwrap();
    stop.send(()).unwrap();
    rt.block_on(running).unwrap();
}

/// The opt-in AV1 archive pass against a real broker: a camera's native H.264 is committed as an `H264Au`
/// chunk, the pass (with a 3 second wait instead of a day) re-encodes it, and the catalog's frame index and length
/// follow the new file, so every frame still reads back by the catalog's offsets and decodes.
#[cfg(feature = "ffmpeg")]
#[test]
fn the_av1_archive_pass_converts_a_committed_chunk_and_the_catalog_follows() {
    if !continuum_tier::probe::av1_encoder_available() {
        eprintln!("this FFmpeg has no SVT-AV1; skipped");
        return;
    }
    use continuum_tier::archive::ArchiveSettings;
    use continuum_tier::h264::{H264Encoder, H264Settings};
    use continuum_tier::{ChunkFormat, ChunkReader, hydrate_frame};

    const W: u32 = 160;
    const H: u32 = 120;
    const N: u32 = 45;
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(3).enable_all().build().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("broker.sock");
    let catalog_path = dir.path().join("catalog.redb");
    let mut config = Config::new(socket.clone(), Duration::from_secs(30));
    config.tier0 = Some(vec![continuum_sched::SlotClass { size_bytes: 64 * 1024, count: 128 }]);
    let server =
        rt.block_on(Server::bind_with_catalog(config, Box::new(RedbCatalog::open(&catalog_path).unwrap()))).unwrap();
    let (stop_broker, stopped) = tokio::sync::oneshot::channel::<()>();
    let broker = rt.spawn(server.run_until(async {
        let _ = stopped.await;
    }));
    let producer = rt.block_on(Client::connect(&socket, ClientOptions::new("producer"))).unwrap();
    let spec = StreamSpec {
        name: "cam0.video".into(),
        kind: fb::StreamKind::Sensor,
        format: fb::PayloadFormat::EncodedVideo,
        shape_name: "H264AnnexB".into(),
        shape_version: 1,
        nominal_rate_hz: 30.0,
        storage: StreamStorage::Slab { class: 0 },
        anchor: None,
        layer: fb::DomainLayer::Layer1Raw,
    };
    let cam = StreamId(rt.block_on(producer.register_stream(&spec)).unwrap().stream_id);
    let chunk_dir = dir.path().join("tier");
    let settings = Settings {
        socket: socket.clone(),
        chunk_dir: chunk_dir.clone(),
        data_root: None,
        streams: vec!["cam0.video".into()],
        flush: FlushSettings {
            chunk_ns: 60_000_000_000,
            part_ns: 60_000_000_000,
            audio_chunk_ns: 60_000_000_000,
            sync_ns: 100_000_000,
        },
        tick: Duration::from_millis(20),
    };
    let mut service = Service::start(settings, rt.handle()).expect("start");
    service.with_archive(ArchiveSettings {
        after: Duration::from_secs(3),
        window: None,
        preset: 12,
        crf: 40,
        threads: 1,
    });

    // One second of a moving gradient as a camera's own H.264, a keyframe first.
    let h264 = H264Settings { gop_frames: 30, frame_rate: 30, low_delay: true, threads: 1, ..H264Settings::default() };
    let mut encoder =
        H264Encoder::open_with(continuum_tier::h264::SOFTWARE_H264_ENCODERS, W, H, h264).expect("a software encoder");
    let t0 = utc_now();
    let mut units = Vec::new();
    for i in 0..N {
        let mut nv12 = vec![128u8; (W * H * 3 / 2) as usize];
        for (k, y) in nv12[..(W * H) as usize].iter_mut().enumerate() {
            *y = u8::try_from((k as u32 % W + 3 * i) % 200 + 20).unwrap();
        }
        let at = t0 + u64::from(i) * FRAME_NS;
        units.extend(encoder.encode(&nv12, at, at).unwrap());
    }
    units.extend(encoder.finish().unwrap());
    assert_eq!(units.len(), N as usize);
    let arena = producer.arenas()[0].arena.clone();
    for (i, unit) in units.iter().enumerate() {
        let (mut slot, _) = arena.claim_overwriting(0, 1).unwrap();
        slot.as_mut_slice()[..unit.bytes.len()].copy_from_slice(&unit.bytes);
        let offset = slot.offset() as u64;
        let handle = slot.publish(unit.bytes.len()).unwrap();
        let mut h = HeaderSpec::raw(Uuid(uuid::Uuid::now_v7().into_bytes()), cam, unit.start_ns, unit.end_ns);
        h.seq = u32::try_from(i).unwrap();
        h.format = PayloadFormat::EncodedVideo;
        h.shape_name = "H264AnnexB".into();
        h.shape_version = 1;
        let locator = StorageLocator::Shm {
            arena: arena.name().to_owned(),
            arena_generation: arena.generation(),
            slab_class: 0,
            slot: handle.index,
            slot_generation: handle.generation,
            offset,
            length: unit.bytes.len() as u64,
        };
        rt.block_on(producer.ingest_acked(&HeaderBuf::build(&h).unwrap(), Some(&locator))).unwrap();
    }
    let deadline = Instant::now() + Duration::from_secs(60);
    while service.counts().appended < u64::from(N) {
        assert!(Instant::now() < deadline, "the worker got {:?}", service.counts());
        service.step().expect("step");
    }
    // Close the chunk (a minute long otherwise) so it's committed, then let the pass find and convert it.
    service.shutdown().expect("shutdown");
    let mut service = Service::start(
        Settings {
            socket: socket.clone(),
            chunk_dir: chunk_dir.clone(),
            data_root: None,
            // cam9 isn't registered: the pass's queries for it are refused and it goes on with cam0.
            streams: vec!["cam0.video".into(), "cam9.video".into()],
            flush: FlushSettings::default(),
            tick: Duration::from_millis(20),
        },
        rt.handle(),
    )
    .expect("restart");
    service.with_archive(ArchiveSettings {
        after: Duration::from_secs(3),
        window: None,
        preset: 12,
        crf: 40,
        threads: 1,
    });
    while service.archive_counts().is_none_or(|c| c.converted == 0) {
        assert!(Instant::now() < deadline, "not converted: {:?}", service.archive_counts());
        service.step().expect("step");
    }
    service.shutdown().expect("shutdown");
    rt.block_on(producer.close("done")).unwrap();
    stop_broker.send(()).unwrap();
    rt.block_on(broker).unwrap();

    let reopen = Instant::now() + Duration::from_secs(10);
    let catalog = loop {
        match RedbCatalog::open(&catalog_path) {
            Ok(c) => break c,
            Err(e) => {
                assert!(Instant::now() < reopen, "the broker never closed its catalog: {e}");
                std::thread::sleep(Duration::from_millis(20));
            }
        }
    };
    let page = catalog.range(cam, Interval::new(t0, t0 + 10_000_000_000).unwrap(), None, 100).unwrap();
    let [item] = page.items.as_slice() else { panic!("one chunk, not {:?}", page.items.len()) };
    let Some(StorageLocator::Chunk { uri, length, .. }) = item.locations.first() else { panic!("a chunk") };
    let path = std::path::Path::new(uri);
    assert_eq!(ChunkReader::open(path).unwrap().format(), ChunkFormat::Av1Au);
    assert_eq!(*length, std::fs::metadata(path).unwrap().len(), "the catalog has the new file's length");
    let frames = item.frames.as_ref().expect("a frame index");
    assert_eq!(frames.len(), N as usize);
    for (index, frame) in frames.iter().enumerate() {
        let read = hydrate_frame(path, index).unwrap();
        assert_eq!((read.record.start_ns, read.record.offset), (frame.start_ns, frame.offset), "frame {index}");
    }
    let decoded = continuum_tier::tier2::read_frames(path, t0, t0 + 10_000_000_000).unwrap();
    assert_eq!(decoded.len(), N as usize, "every frame decodes from the AV1 chunk");
}

/// An H.264 P-frame that comes with no chunk open (a camera reconnecting mid-GOP, or a capture drop) can never
/// be archived, so the worker drops it as undecodable. Its Tier 0 slot must still become reusable. A reader that pinned
/// it and went away leaves a pin behind; the broker evicts an expired pin only on a slot it may reclaim, which an
/// unflushed one never is. Left unmarked, a class filled with them stayed full until the broker restarted, dropping
/// every camera that shares it.
#[test]
fn a_dropped_undecodable_frame_gives_its_tier_0_slot_back() {
    const SLOTS: u32 = 8;
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(3).enable_all().build().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("broker.sock");
    let catalog_path = dir.path().join("catalog.redb");
    let mut config = Config::new(socket.clone(), Duration::from_secs(30));
    config.tier0 = Some(vec![continuum_sched::SlotClass { size_bytes: 16 * 1024, count: SLOTS }]);
    let server =
        rt.block_on(Server::bind_with_catalog(config, Box::new(RedbCatalog::open(&catalog_path).unwrap()))).unwrap();
    let (stop_broker, stopped) = tokio::sync::oneshot::channel::<()>();
    let broker = rt.spawn(server.run_until(async {
        let _ = stopped.await;
    }));
    let producer = rt.block_on(Client::connect(&socket, ClientOptions::new("producer"))).unwrap();
    let spec = StreamSpec {
        name: "patio.video".into(),
        kind: fb::StreamKind::Sensor,
        format: fb::PayloadFormat::EncodedVideo,
        shape_name: "H264AnnexB".into(),
        shape_version: 1,
        nominal_rate_hz: 10.0,
        storage: StreamStorage::Slab { class: 0 },
        anchor: None,
        layer: fb::DomainLayer::Layer1Raw,
    };
    let cam = StreamId(rt.block_on(producer.register_stream(&spec)).unwrap().stream_id);
    let settings = Settings {
        socket: socket.clone(),
        chunk_dir: dir.path().join("tier"),
        data_root: None,
        streams: vec!["patio.video".into()],
        flush: FlushSettings::default(),
        tick: Duration::from_millis(20),
    };
    let mut service = Service::start(settings, rt.handle()).expect("start");

    // Fill the whole class with P-frames (NAL type 1, no IDR before them), as a reconnect mid-GOP sends.
    let arena = producer.arenas()[0].arena.clone();
    let t0 = utc_now();
    for i in 0..SLOTS {
        let bytes = [0u8, 0, 0, 1, 0x41, 0x9a, 0x02, u8::try_from(i).unwrap()];
        let mut slot = arena.claim(0, 1).expect("a free slot while filling the class");
        slot.as_mut_slice()[..bytes.len()].copy_from_slice(&bytes);
        let offset = slot.offset() as u64;
        let handle = slot.publish(bytes.len()).unwrap();
        // A reader (a shared decoder, say) pinned it, then went away without letting go.
        std::mem::forget(arena.lease(handle, Duration::from_millis(50)).unwrap());
        let start = t0 + u64::from(i) * 100_000_000;
        let mut h = HeaderSpec::raw(Uuid(uuid::Uuid::now_v7().into_bytes()), cam, start, start);
        h.seq = i;
        h.format = PayloadFormat::EncodedVideo;
        h.shape_name = "H264AnnexB".into();
        h.shape_version = 1;
        let locator = StorageLocator::Shm {
            arena: arena.name().to_owned(),
            arena_generation: arena.generation(),
            slab_class: 0,
            slot: handle.index,
            slot_generation: handle.generation,
            offset,
            length: bytes.len() as u64,
        };
        rt.block_on(producer.ingest_acked(&HeaderBuf::build(&h).unwrap(), Some(&locator))).unwrap();
    }
    let deadline = Instant::now() + Duration::from_secs(20);
    while service.counts().undecodable < u64::from(SLOTS) {
        assert!(Instant::now() < deadline, "the worker got {:?}", service.counts());
        service.step().expect("step");
    }
    assert_eq!(service.counts().appended, 0, "{:?}", service.counts());

    // The class is full of dropped frames; the broker's pressure reclaim has to hand one back for the next picture.
    let claimed = loop {
        service.step().expect("step");
        if let Ok(slot) = arena.claim(0, 2) {
            break slot;
        }
        assert!(Instant::now() < deadline, "no slot came back: {:?}", arena.scan(0).unwrap());
        std::thread::sleep(Duration::from_millis(20));
    };
    drop(claimed);
    service.shutdown().expect("shutdown");
    rt.block_on(producer.close("done")).unwrap();
    let _ = stop_broker.send(());
    rt.block_on(broker).unwrap();
}
