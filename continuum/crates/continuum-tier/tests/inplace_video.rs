//! MP4 files registered in place against a real broker, in-process, and read back by GOP: a `video/` directory
//! whose `index.jsonl` names two finished recordings of cam0, one still being written (no moov), one that's gone,
//! cam0's first listed twice, and a line for another camera. Each recording is `data/bframes.mp4` (2 s at 10
//! fps, H.264 with B-frames, a keyframe every 10 frames: see tests/mp4.rs).

use std::time::Duration;

use continuum_broker::server::{Config, Server};
use continuum_catalog::RedbCatalog;
use continuum_client::{Client, ClientOptions, QuerySpec, WorkerSpec};
use continuum_core::StorageLocator;
use continuum_tier::inplace_audio::Registration;
use continuum_tier::inplace_video::{gop_units, register_video_index};

const FIXTURE: &[u8] = include_bytes!("data/bframes.mp4");
const S: u64 = 1_000_000_000;
const MS: u64 = 1_000_000;
const T0: u64 = 1_793_300_000_250_000_000;

fn video_dir(root: &std::path::Path) -> std::path::PathBuf {
    let video = root.join("video");
    std::fs::create_dir_all(video.join("cam0/2026-10-29")).unwrap();
    std::fs::write(video.join("cam0/2026-10-29/140000.mp4"), FIXTURE).unwrap();
    std::fs::write(video.join("cam0/2026-10-29/140002.mp4"), FIXTURE).unwrap();
    let moov = FIXTURE.windows(4).position(|w| w == b"moov").unwrap() - 4;
    std::fs::write(video.join("cam0/2026-10-29/140004.mp4"), &FIXTURE[..moov]).unwrap();
    let line = |cam: &str, file: &str, start: &str| {
        format!(
            "{{\"cam\": \"{cam}\", \"file\": \"{file}\", \"start\": {start}, \"fps\": 10.0, \"w\": 64, \"h\": 48}}\n"
        )
    };
    let index = [
        line("cam0", "cam0/2026-10-29/140000.mp4", "1793299990.0"),
        line("cam1", "cam1/2026-10-29/140000.mp4", "1793300000.25"),
        line("cam0", "cam0/2026-10-29/140002.mp4", "1793300002.25"),
        line("cam0", "cam0/2026-10-29/140004.mp4", "1793300004.25"),
        line("cam0", "cam0/2026-10-29/140009.mp4", "1793300009.25"),
        line("cam0", "cam0/2026-10-29/140000.mp4", "1793300000.25"),
    ];
    std::fs::write(video.join("index.jsonl"), index.concat()).unwrap();
    video
}

#[test]
fn recordings_register_in_place_once_and_their_gops_read_back_in_decode_order() {
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("broker.sock");
    let mut config = Config::new(socket.clone(), Duration::from_secs(30));
    config.tier0 = Some(vec![continuum_sched::SlotClass { size_bytes: 4 * 1024, count: 8 }]);
    let catalog = RedbCatalog::open(dir.path().join("catalog.redb")).unwrap();
    let server = rt.block_on(Server::bind_with_catalog(config, Box::new(catalog))).unwrap();
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let broker = rt.spawn(server.run_until(async {
        let _ = stopped.await;
    }));
    let video = video_dir(dir.path());

    let client = rt.block_on(Client::connect(&socket, ClientOptions::new("video importer"))).unwrap();
    let worker = WorkerSpec {
        worker_id: "video-import".into(),
        version_hash: "test".into(),
        capabilities: vec![],
        device_index: None,
        keeps_copies: false,
    };
    rt.block_on(client.register_worker(&worker)).unwrap();
    let first = rt.block_on(register_video_index(&client, "cam0", &video)).expect("register");
    assert_eq!(first.len(), 4, "cam0's files once each, not cam1's: {first:?}");
    let spans: Vec<(u64, u64)> = first
        .iter()
        .filter_map(|(_, r)| match r {
            Registration::Registered { span, duplicate: false, .. } => Some(*span),
            _ => None,
        })
        .collect();
    assert_eq!(spans, [(T0, T0 + 2 * S), (T0 + 2 * S, T0 + 4 * S)], "the last start counts: {first:?}");
    let skipped: Vec<String> = first
        .iter()
        .filter_map(|(_, r)| if let Registration::Skipped(w) = r { Some(w.clone()) } else { None })
        .collect();
    assert!(skipped[0].contains("no moov") && skipped[1].contains("not on disk"), "{skipped:?}");

    // A second run is a repeat with the same ids.
    let second = rt.block_on(register_video_index(&client, "cam0", &video)).expect("again");
    for ((_, a), (_, b)) in first.iter().zip(&second) {
        if let (
            Registration::Registered { payload_id: p1, chunk_id: c1, .. },
            Registration::Registered { payload_id: p2, chunk_id: c2, duplicate, .. },
        ) = (a, b)
        {
            assert_eq!((p1, c1, *duplicate), (p2, c2, true));
        }
    }

    // The catalog holds the two chunks, naming their files in place, and nothing else: the GOP payloads were
    // absorbed, and the second run made no new ones.
    let found = rt.block_on(client.query_all(&QuerySpec::hit_only("cam0.video_v1", (T0, T0 + 4 * S)))).expect("query");
    let rows: Vec<_> = found.items.iter().map(|i| (i.header.seq(), i.locator.clone())).collect();
    assert_eq!(rows.len(), 2, "{rows:?}");
    let path = std::fs::canonicalize(video.join("cam0/2026-10-29/140000.mp4")).unwrap();
    assert!(found.items.iter().any(|i| matches!(&i.locator,
        Some(StorageLocator::Chunk { uri, .. }) if *uri == path.to_string_lossy())));

    // Each GOP reads back as Annex B units in decode order: the IDR first with its SPS and PPS, presentation times
    // every 100 ms from the GOP's start, and every frame decoded no later than it's shown.
    for (gop, at) in [(0, T0), (1, T0 + S)] {
        let units = gop_units(&path, T0, gop).expect("units");
        assert_eq!(units.len(), 10);
        assert!(units[0].key && units[1..].iter().all(|u| !u.key), "only the IDR is a keyframe");
        assert_eq!(&units[0].bytes[..5], &[0, 0, 0, 1, 0x67], "the SPS first");
        let mut shown: Vec<u64> = units.iter().map(|u| u.pts_ns).collect();
        assert_ne!(
            shown,
            {
                let mut s = shown.clone();
                s.sort_unstable();
                s
            },
            "B-frames: decode order isn't display order"
        );
        shown.sort_unstable();
        assert_eq!(shown, (0..10).map(|k| at + k * 100 * MS).collect::<Vec<_>>());
        assert!(units.windows(2).all(|w| w[0].dts_ns < w[1].dts_ns), "decode times rise");
        assert!(units.iter().all(|u| u.dts_ns <= u.pts_ns), "decoded no later than shown");
    }
    assert!(gop_units(&path, T0, 2).is_err(), "only two GOPs");

    rt.block_on(client.close("done")).unwrap();
    stop.send(()).unwrap();
    rt.block_on(broker).unwrap();
}

#[cfg(feature = "ffmpeg")]
#[test]
fn a_time_range_decodes_to_exactly_its_frames_in_display_order() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rec.mp4");
    std::fs::write(&path, FIXTURE).unwrap();
    let frames = continuum_tier::inplace_video::read_frames(&path, T0, T0 + 250 * MS, T0 + 1_250 * MS).expect("frames");
    let starts: Vec<u64> = frames.iter().map(|f| (f.start_ns - T0) / MS).collect();
    assert_eq!(starts, (2..=12).map(|k| k * 100).collect::<Vec<_>>(), "across the GOP boundary at 1 s");
    assert!(frames.iter().all(|f| (f.width, f.height, f.nv12.len()) == (64, 48, 64 * 48 * 3 / 2)));
}
