//! The recent H.264 window against a real broker and tier worker, in-process: a camera writes native access
//! units into Tier 0, the tier worker absorbs them into committed Tier 1 chunks, and a window opened mid-GOP yields
//! every access unit from the keyframe before its start, in decode order, across Tier 1, Tier 0, and the live edge,
//! with a missing frame reported as a gap until the next keyframe. Short and bounded, with deadlines.

use std::sync::mpsc;
use std::time::{Duration, Instant};

use continuum_broker::server::{Config, Server};
use continuum_catalog::RedbCatalog;
use continuum_client::{Client, ClientOptions, StreamSpec, StreamStorage, SubscriptionOverflow};
use continuum_core::schema::continuum as fb;
use continuum_core::{HeaderBuf, StorageLocator, StreamId, Uuid};
use continuum_tier::FlushSettings;
use continuum_tier::recent::{DEFAULT_WINDOW, Gap, GapWhy, H264Window, Item};
use continuum_tier::service::{Service, Settings};

const FRAME_NS: u64 = 33_333_333;
const GOP: u32 = 15;
const FRAMES: u32 = 70;
/// Never published: the window reports a gap from 49 to the keyframe at 60.
const LOST: u32 = 50;
/// Published before the window opens; the rest arrive live.
const BEFORE: u32 = 40;
const EPOCH: u32 = 5;

fn utc_now() -> u64 {
    u64::try_from(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()).unwrap()
}

/// A keyframe is SPS, PPS, and an IDR; the rest are P slices. Each carries its index so bytes are checkable.
fn access_unit(i: u32) -> Vec<u8> {
    let mut v = if i.is_multiple_of(GOP) {
        vec![0, 0, 0, 1, 0x67, 0x42, 0, 0, 0, 1, 0x68, 0xce, 0, 0, 1, 0x65]
    } else {
        vec![0, 0, 0, 1, 0x41]
    };
    v.extend(i.to_le_bytes());
    v.extend((0..200).map(|k| u8::try_from((k + i as usize) % 251).expect("small")));
    v
}

fn header(stream: StreamId, id: Uuid, start: u64, i: u32) -> HeaderBuf {
    let mut b = flatbuffers::FlatBufferBuilder::new();
    let shape = b.create_string("H264AnnexB");
    let cap = fb::CaptureInfo::create(
        &mut b,
        &fb::CaptureInfoArgs {
            width: 64,
            height: 48,
            device_seq: u64::from(i),
            capture_epoch: EPOCH,
            ..Default::default()
        },
    );
    let fid = fb::Uuid::from(id);
    let h = fb::PayloadHeader::create(
        &mut b,
        &fb::PayloadHeaderArgs {
            id: Some(&fid),
            stream_id: stream.0,
            start_ns: start,
            end_ns: start,
            seq: i,
            layer: fb::DomainLayer::Layer1Raw,
            format: fb::PayloadFormat::EncodedVideo,
            shape_name: Some(shape),
            source_tier: fb::StorageTier::Tier0Hot,
            capture: Some(cap),
            ..Default::default()
        },
    );
    fb::finish_payload_header_buffer(&mut b, h);
    HeaderBuf::new(b.finished_data().to_vec()).unwrap()
}

#[test]
fn a_window_opened_mid_gop_reads_both_tiers_and_follows_live_without_a_seam() {
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(3).enable_all().build().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("broker.sock");
    let mut config = Config::new(socket.clone(), Duration::from_secs(30));
    config.tier0 = Some(vec![continuum_sched::SlotClass { size_bytes: 16 * 1024, count: 512 }]);
    let catalog = RedbCatalog::open(dir.path().join("catalog.redb")).unwrap();
    let server = rt.block_on(Server::bind_with_catalog(config, Box::new(catalog))).unwrap();
    let (stop_broker, stopped) = tokio::sync::oneshot::channel::<()>();
    let broker = rt.spawn(server.run_until(async {
        let _ = stopped.await;
    }));

    let producer = rt.block_on(Client::connect(&socket, ClientOptions::new("camera"))).unwrap();
    let spec = StreamSpec {
        name: "patio.video".into(),
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
    std::fs::create_dir_all(&chunk_dir).unwrap();
    let settings = Settings {
        socket: socket.clone(),
        chunk_dir,
        // Relative locators, read back through the window's own root.
        data_root: Some(dir.path().to_path_buf()),
        streams: vec!["patio.video".into()],
        flush: FlushSettings {
            chunk_ns: 300_000_000,
            part_ns: 300_000_000,
            audio_chunk_ns: 300_000_000,
            sync_ns: 100_000_000,
        },
        tick: Duration::from_millis(20),
    };
    let mut service = Service::start(settings, rt.handle()).expect("start");

    let arena = producer.arenas()[0].arena.clone();
    let t0 = utc_now();
    let at = move |i: u32| t0 + u64::from(i) * FRAME_NS;
    let (go_on, resume) = mpsc::channel::<()>();
    let (published, before_done) = mpsc::channel::<Vec<Uuid>>();
    let feeder = std::thread::spawn(move || {
        let mut ids = Vec::new();
        for i in 0..FRAMES {
            if i == BEFORE {
                published.send(ids.clone()).unwrap();
                resume.recv().unwrap();
            }
            let id = Uuid(uuid::Uuid::now_v7().into_bytes());
            ids.push(id);
            if i == LOST {
                continue;
            }
            let bytes = access_unit(i);
            let (mut slot, _) = arena.claim_overwriting(0, 1).unwrap();
            slot.as_mut_slice()[..bytes.len()].copy_from_slice(&bytes);
            let offset = slot.offset() as u64;
            let handle = slot.publish(bytes.len()).unwrap();
            let locator = StorageLocator::Shm {
                arena: arena.name().to_owned(),
                arena_generation: arena.generation(),
                slab_class: 0,
                slot: handle.index,
                slot_generation: handle.generation,
                offset,
                length: bytes.len() as u64,
            };
            producer.ingest(&header(cam, id, at(i), i), Some(&locator)).unwrap();
            std::thread::sleep(Duration::from_millis(33));
        }
        (producer, ids)
    });

    // Until frames 0 to 39 are written and at least two chunks have reached Tier 1.
    let deadline = Instant::now() + Duration::from_secs(30);
    let early = loop {
        if let Ok(ids) = before_done.try_recv() {
            break ids;
        }
        assert!(Instant::now() < deadline, "the camera stalled");
        service.step().expect("step");
    };
    while service.counts().committed < 2 {
        assert!(Instant::now() < deadline, "the worker committed {:?}", service.counts());
        service.step().expect("step");
    }

    // The window starts mid-GOP, at frame 20: it backs up to the keyframe at 15.
    let from_ns = at(20);
    let root = dir.path().to_path_buf();
    let reader = rt.spawn(async move {
        let query = Client::connect(&socket, ClientOptions::new("reader")).await.unwrap();
        let mut options = ClientOptions::new("reader-live");
        options.subscription_overflow = SubscriptionOverflow::Backpressure;
        let live = Client::connect(&socket, options).await.unwrap();
        let mut window = H264Window::open(
            std::sync::Arc::new(query),
            std::sync::Arc::new(live),
            &root,
            "patio.video",
            from_ns,
            DEFAULT_WINDOW,
            utc_now(),
        )
        .await
        .unwrap();
        let mut items = Vec::new();
        while items.len() < 50 {
            let next = tokio::time::timeout(Duration::from_secs(20), window.next()).await;
            let Ok(Some(item)) = next else { break };
            let last = matches!(&item, Item::Au(au) if au.device_seq == Some(u64::from(FRAMES - 1)));
            items.push(item);
            if last {
                break;
            }
        }
        items
    });
    // Give the window time to subscribe before the live frames are written.
    std::thread::sleep(Duration::from_millis(300));
    go_on.send(()).unwrap();
    while !reader.is_finished() {
        assert!(Instant::now() < deadline + Duration::from_secs(30), "the window never reached the live edge");
        service.step().expect("step");
    }
    let items = rt.block_on(reader).unwrap();
    let (producer, ids) = feeder.join().unwrap();
    service.shutdown().expect("shutdown");
    assert_eq!(early.len(), BEFORE as usize);

    let mut expected: Vec<Item> = Vec::new();
    let mut tier0 = 0;
    let mut tier1 = 0;
    for item in &items {
        if let Item::Au(au) = item {
            if au.header.is_some() { tier0 += 1 } else { tier1 += 1 }
        }
    }
    for i in 15..FRAMES {
        if (LOST..60).contains(&i) {
            if i == LOST {
                expected.push(Item::Gap(Gap { from_ns: at(LOST - 1), to_ns: at(60), why: GapWhy::Missing }));
            }
            continue;
        }
        expected.push(Item::Au(continuum_tier::recent::Au {
            id: ids[i as usize],
            start_ns: at(i),
            end_ns: at(i),
            seq: i,
            device_seq: Some(u64::from(i)),
            epoch: EPOCH,
            width: 64,
            height: 48,
            flip: false,
            random_access: i.is_multiple_of(GOP),
            header: None,
            bytes: access_unit(i),
        }));
    }
    // Compare without the Tier 0 header, which only some carry; its fields are compared through the Au values.
    let stripped: Vec<Item> = items
        .into_iter()
        .map(|item| match item {
            Item::Au(mut au) => {
                au.header = None;
                Item::Au(au)
            }
            gap => gap,
        })
        .collect();
    assert_eq!(stripped, expected);
    assert!(tier1 > 0 && tier0 > 0, "the window crossed the seam: {tier1} from chunks, {tier0} from Tier 0");

    rt.block_on(producer.close("done")).unwrap();
    stop_broker.send(()).unwrap();
    rt.block_on(broker).unwrap();
}
