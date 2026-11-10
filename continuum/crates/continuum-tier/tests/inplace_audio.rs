//! FLAC files registered in place against a real broker, in-process: a directory with an
//! `index.jsonl` naming two FLAC files, one never finished (no samples in its STREAMINFO), and one that's gone.
//! The two become payloads and chunks at their exact starts, the others are skipped with reasons, a second run
//! changes nothing, and the chunk's locator names the file, which hydrates whole and decodes.

use std::time::Duration;

use continuum_broker::server::{Config, Server};
use continuum_catalog::RedbCatalog;
use continuum_client::{Client, ClientOptions, QuerySpec, WorkerSpec};
use continuum_core::StorageLocator;
use continuum_tier::inplace_audio::{Registration, decode_flac, register_audio_index};
use continuum_tier::{codecs::flac_encode, hydrate_frame};

const S: u64 = 1_000_000_000;

fn tone(n: usize, k: i32) -> Vec<i32> {
    (0..n).map(|i| (i32::try_from(i % 400).unwrap() - 200) * k).collect()
}

#[test]
fn the_archive_registers_in_place_once_and_reads_back() {
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

    // The directory to register.
    let device = dir.path().join("alsa-input-wave-3");
    std::fs::create_dir_all(device.join("2026-10-29")).unwrap();
    let a = tone(96_000, 3);
    let b = tone(48_000, 5);
    std::fs::write(device.join("2026-10-29/140000.flac"), flac_encode(&a, 1, 16, 48_000).unwrap()).unwrap();
    std::fs::write(device.join("2026-10-29/140002.flac"), flac_encode(&b, 1, 16, 48_000).unwrap()).unwrap();
    // A file still being written: its STREAMINFO's 36 bit sample count (the low nibble of byte 21, then 22 to 25)
    // is 0.
    let mut unfinished = flac_encode(&b, 1, 16, 48_000).unwrap();
    unfinished[21] &= 0xf0;
    unfinished[22..26].fill(0);
    std::fs::write(device.join("2026-10-29/140003.flac"), unfinished).unwrap();
    // A file that isn't FLAC at all (truncated to nothing useful by a full disk, say).
    std::fs::write(device.join("2026-10-29/140004.flac"), b"not flac").unwrap();
    // 140000 is listed twice (its writer restarted within the second): it counts once, at its last start.
    std::fs::write(
        device.join("index.jsonl"),
        "{\"file\": \"2026-10-29/140000.flac\", \"start\": 1793299999.5, \"sr\": 48000}\n\
         {\"file\": \"2026-10-29/140002.flac\", \"start\": 1793300002.25, \"sr\": 48000}\n\
         {\"file\": \"2026-10-29/140003.flac\", \"start\": 1793300003.25, \"sr\": 48000}\n\
         {\"file\": \"2026-10-29/140009.flac\", \"start\": 1793300009.25, \"sr\": 48000}\n\
         {\"file\": \"2026-10-29/140004.flac\", \"start\": 1793300004.25, \"sr\": 48000}\n\
         {\"file\": \"2026-10-29/140000.flac\", \"start\": 1793300000.25, \"sr\": 48000}\n",
    )
    .unwrap();

    // Only a registered worker commits chunks, so the importer registers as one.
    let client = rt.block_on(Client::connect(&socket, ClientOptions::new("audio importer"))).unwrap();
    let worker = WorkerSpec {
        worker_id: "audio-import".into(),
        version_hash: "test".into(),
        capabilities: vec![],
        device_index: None,
        keeps_copies: false,
    };
    rt.block_on(client.register_worker(&worker)).unwrap();
    let first = rt.block_on(register_audio_index(&client, "wave3.audio_v1", &device)).expect("register");
    let t0 = 1_793_300_000_250_000_000;
    let spans: Vec<_> = first
        .iter()
        .filter_map(|(_, r)| match r {
            Registration::Registered { span, duplicate: false, .. } => Some(*span),
            _ => None,
        })
        .collect();
    assert_eq!(spans, [(t0, t0 + 2 * S), (t0 + 2 * S, t0 + 3 * S)], "{first:?}");
    let skipped: Vec<_> = first
        .iter()
        .filter_map(|(_, r)| if let Registration::Skipped(why) = r { Some(why.clone()) } else { None })
        .collect();
    assert_eq!(first.len(), 5, "one entry per file: {first:?}");
    assert_eq!(skipped.len(), 3, "{first:?}");
    assert!(skipped[0].contains("no samples"), "{skipped:?}");
    assert!(skipped[1].contains("not on disk"), "{skipped:?}");
    assert!(skipped[2].contains("not readable FLAC"), "{skipped:?}");

    // A second run, through a symlink to the directory, sends the same headers: the same ids, and nothing new.
    let link = dir.path().join("link-to-device");
    std::os::unix::fs::symlink(&device, &link).unwrap();
    let second = rt.block_on(register_audio_index(&client, "wave3.audio_v1", &link)).expect("again");
    for ((_, r1), (_, r2)) in first.iter().zip(&second) {
        match (r1, r2) {
            (
                Registration::Registered { payload_id: p1, chunk_id: c1, .. },
                Registration::Registered { payload_id: p2, chunk_id: c2, duplicate, .. },
            ) => assert_eq!((p1, c1, *duplicate), (p2, c2, true)),
            (Registration::Skipped(_), Registration::Skipped(_)) => {}
            other => panic!("the runs differ: {other:?}"),
        }
    }

    // The catalog's chunk for the first file points at it in place; it hydrates whole and decodes to its samples.
    let found = rt.block_on(client.query_all(&QuerySpec::hit_only("wave3.audio_v1", (t0, t0 + 3 * S)))).expect("query");
    let uris: Vec<String> = found
        .items
        .iter()
        .filter_map(|i| match &i.locator {
            Some(StorageLocator::Chunk { uri, .. }) => Some(uri.clone()),
            _ => None,
        })
        .collect();
    let path = std::fs::canonicalize(device.join("2026-10-29/140000.flac")).unwrap();
    assert!(uris.contains(&path.to_string_lossy().into_owned()), "{uris:?}");
    let whole = hydrate_frame(&path, 0).expect("hydrate");
    assert_eq!(whole.bytes, std::fs::read(&path).unwrap());
    let (rate, channels, samples) = decode_flac(&whole.bytes).expect("decode");
    assert_eq!((rate, channels, samples.len()), (48_000, 1, a.len()));
    #[expect(clippy::cast_precision_loss, reason = "test values")]
    let want: Vec<f32> = a.iter().map(|&s| s as f32 / 32_768.0).collect();
    assert_eq!(samples, want);

    rt.block_on(client.close("done")).unwrap();
    stop.send(()).unwrap();
    rt.block_on(broker).unwrap();
}
