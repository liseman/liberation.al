//! Purging chunk files by the catalog's own rules: in a chunk, the frames whose slot overlaps a purged window, and
//! those whose payload id was purged, are rewritten away, keeping their places, so the file agrees with the frames
//! the catalog cut; a file wholly inside a window is deleted, but only when the catalog doesn't still know it.

use std::collections::BTreeSet;

use continuum_core::{Interval, StreamId, Uuid};
use continuum_tier::service::{PurgedFile, chunk_files, merge_windows, purge_file};
use continuum_tier::{ChunkFormat, ChunkReader, ChunkWriter, hydrate};

const MS: u64 = 1_000_000;
const T0: u64 = 1_700_000_000_000_000_000;

fn uuid(n: u32) -> Uuid {
    let mut b = [0u8; 16];
    b[0] = 0x01;
    b[6] = 0x70;
    b[8] = 0x80;
    b[12..16].copy_from_slice(&n.to_be_bytes());
    Uuid(b)
}

/// A chunk of ten instant frames 100 ms apart, each 10 bytes of its index.
/// The hour directory (`continuum_tier::layout`) a test's file named `<stream>-...` goes in, made.
fn hour(dir: &std::path::Path, name: &str) -> std::path::PathBuf {
    let stream = StreamId(name.split('-').next().unwrap().parse().unwrap());
    let hour = continuum_tier::layout::hour_dir(dir, stream, T0);
    continuum_tier::layout::make_hour_dir(dir, &hour).unwrap();
    hour
}

fn chunk(dir: &std::path::Path, name: &str) -> std::path::PathBuf {
    let path = hour(dir, name).join(name);
    let mut w = ChunkWriter::create(&path, StreamId(7), ChunkFormat::Passthrough, T0).unwrap();
    for i in 0..10u32 {
        let t = T0 + u64::from(i) * 100 * MS;
        w.append_payload(uuid(i), t, t, i, &[u8::try_from(i).unwrap(); 10]).unwrap();
    }
    w.close().unwrap();
    path
}

fn window(from_ms: u64, to_ms: u64) -> Interval {
    Interval::new(T0 + from_ms * MS, T0 + to_ms * MS).unwrap()
}

fn none() -> BTreeSet<Uuid> {
    BTreeSet::new()
}

#[test]
fn frames_whose_slot_overlaps_the_window_are_cut_and_keep_their_places() {
    let dir = tempfile::tempdir().unwrap();
    let path = chunk(dir.path(), "7-1-1.cnk");
    // [250, 420) ms overlaps the slots of frames 2 ([200, 300)), 3, and 4 ([400, 500)).
    let PurgedFile::Rewritten(rw) = purge_file(&path, &[window(250, 420)], &none(), true).unwrap() else {
        panic!("rewritten")
    };
    let cut: Vec<usize> = rw.records.iter().enumerate().filter(|(_, r)| r.cut).map(|(i, _)| i).collect();
    assert_eq!(cut, vec![2, 3, 4]);
    assert_eq!(rw.records.len(), 10, "every frame keeps its place");
    let reader = ChunkReader::open(&path).unwrap();
    assert_eq!(reader.records().iter().filter(|r| r.cut).count(), 3, "the file itself was rewritten");
    let left: Vec<u8> = hydrate(&path, T0, T0 + 1_000 * MS).unwrap().into_iter().map(|h| h.bytes[0]).collect();
    assert_eq!(left, vec![0, 1, 5, 6, 7, 8, 9], "the purged frames' bytes are gone");
    // The same purge again finds nothing left to cut, and says the file holds cuts.
    assert!(matches!(
        purge_file(&path, &[window(250, 420)], &none(), true).unwrap(),
        PurgedFile::Untouched { cut: true }
    ));
}

#[test]
fn a_file_wholly_inside_a_window_is_removed_only_when_the_catalog_doesnt_know_it() {
    let dir = tempfile::tempdir().unwrap();
    let path = chunk(dir.path(), "7-1-1.cnk");
    assert!(matches!(
        purge_file(&path, &[window(2_000, 3_000)], &none(), true).unwrap(),
        PurgedFile::Untouched { cut: false }
    ));
    // The catalog still knows it: every frame cut, the file kept for its locator.
    let PurgedFile::Rewritten(rw) = purge_file(&path, &[window(0, 5_000)], &none(), false).unwrap() else {
        panic!("rewritten")
    };
    assert!(rw.records.iter().all(|r| r.cut));
    assert!(path.exists());
    // The catalog doesn't (it removed the chunk, or never had it): deleted.
    let other = chunk(dir.path(), "7-2-2.cnk");
    assert!(matches!(purge_file(&other, &[window(0, 5_000)], &none(), true).unwrap(), PurgedFile::Removed));
    assert!(!other.exists());
}

#[test]
fn a_purge_by_id_cuts_those_records_without_any_window() {
    let dir = tempfile::tempdir().unwrap();
    let path = chunk(dir.path(), "7-1-1.cnk");
    let ids: BTreeSet<Uuid> = [uuid(1), uuid(7)].into_iter().collect();
    let PurgedFile::Rewritten(rw) = purge_file(&path, &[], &ids, true).unwrap() else { panic!("rewritten") };
    let cut: Vec<usize> = rw.records.iter().enumerate().filter(|(_, r)| r.cut).map(|(i, _)| i).collect();
    assert_eq!(cut, vec![1, 7]);
}

#[test]
fn windows_that_overlap_or_touch_merge() {
    let w = merge_windows(vec![window(300, 400), window(0, 100), window(100, 150), window(350, 500), window(600, 700)]);
    assert_eq!(w, vec![window(0, 150), window(300, 500), window(600, 700)]);
}

#[test]
fn a_streams_files_include_those_set_aside_and_no_one_elses() {
    let dir = tempfile::tempdir().unwrap();
    chunk(dir.path(), "7-1-1.cnk");
    chunk(dir.path(), "7-2-2.rejected");
    chunk(dir.path(), "7-3-3.layout-unknown");
    chunk(dir.path(), "17-1-1.cnk");
    std::fs::write(hour(dir.path(), "7-3-3.rewrite").join("7-3-3.rewrite"), b"leftover").unwrap();
    let names = |stream| -> Vec<String> {
        chunk_files(dir.path(), stream)
            .unwrap()
            .iter()
            .map(|(_, p)| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect()
    };
    assert_eq!(
        names(Some(StreamId(7))),
        vec!["7-1-1.cnk".to_owned(), "7-2-2.rejected".to_owned(), "7-3-3.layout-unknown".to_owned()]
    );
    assert_eq!(names(None).len(), 4, "every stream's, for a purge by id");
}

#[test]
fn a_file_retired_after_the_purge_inventory_does_not_abort_the_remaining_purge() {
    let dir = tempfile::tempdir().unwrap();
    let retired = chunk(dir.path(), "7-1-1.cnk");
    let kept = chunk(dir.path(), "7-2-2.cnk");
    let inventory = chunk_files(dir.path(), None).unwrap();
    // Something else (a rejected chunk set aside, an earlier pass) removes this file after the purge inventoried it.
    std::fs::remove_file(&retired).unwrap();
    assert!(matches!(purge_file(&retired, &[], &none(), false).unwrap(), PurgedFile::Missing));
    for (_, path) in inventory {
        assert!(purge_file(&path, &[window(250, 420)], &none(), false).is_ok());
    }
    let reader = ChunkReader::open(&kept).unwrap();
    assert_eq!(reader.records().iter().filter(|record| record.cut).count(), 3);
}

#[test]
fn other_file_errors_are_not_mistaken_for_missing_chunks() {
    let dir = tempfile::tempdir().unwrap();
    let invalid = dir.path().join("not-a-directory");
    std::fs::write(&invalid, b"not a chunk").unwrap();
    assert!(purge_file(&invalid, &[], &none(), false).is_err());
    assert!(purge_file(&invalid.join("7-1-1.cnk"), &[], &none(), false).is_err());
}

#[cfg(unix)]
#[test]
fn permission_errors_remain_visible() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let path = chunk(dir.path(), "7-1-1.cnk");
    let permissions = std::fs::metadata(&path).unwrap().permissions();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o0)).unwrap();
    let can_read = std::fs::File::open(&path).is_ok(); // privileged test runners bypass mode bits
    let result = purge_file(&path, &[], &none(), false);
    std::fs::set_permissions(&path, permissions).unwrap();
    if !can_read {
        assert!(matches!(result, Err(continuum_tier::TierError::Chunk(continuum_tier::ChunkError::Io(error)))
            if error.kind() == std::io::ErrorKind::PermissionDenied));
    }
}

#[test]
fn layout_unknown_quarantine_obeys_id_and_span_purges() {
    let dir = tempfile::tempdir().unwrap();
    let path = chunk(dir.path(), "7-1-1.layout-unknown");
    let ids = [uuid(1), uuid(7)].into_iter().collect();
    let PurgedFile::Rewritten(rewritten) = purge_file(&path, &[], &ids, true).unwrap() else {
        panic!("quarantined IDs must be rewritten")
    };
    assert!(rewritten.records[1].cut && rewritten.records[7].cut);
    let frames = hydrate(&path, T0, T0 + 1000 * MS).unwrap();
    assert_eq!(frames.len(), 8);
    assert!(frames.iter().all(|frame| frame.record.id != uuid(1) && frame.record.id != uuid(7)));
    let whole = Interval::new(T0, T0 + 1000 * MS).unwrap();
    assert!(matches!(purge_file(&path, &[whole], &BTreeSet::new(), true).unwrap(), PurgedFile::Removed));
    assert!(!path.exists());
}

#[test]
fn opus_packet_containers_keep_plain_record_slot_purge_semantics() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("7-opus.cnk");
    let mut writer = ChunkWriter::create(&path, StreamId(7), ChunkFormat::Opus, T0).unwrap();
    // Opus COP2 bodies are not video gop::Unit tables and must never be parsed as those for purge selection.
    let body = b"COP2 opaque packet container";
    writer.append(T0, T0 + MS, 1, body).unwrap();
    writer.append(T0 + MS, T0 + 2 * MS, 2, body).unwrap();
    writer.close().unwrap();
    let PurgedFile::Rewritten(rewritten) = purge_file(&path, &[window(0, 1)], &none(), false).unwrap() else {
        panic!("rewritten")
    };
    assert!(rewritten.records[0].cut);
    assert!(!rewritten.records[1].cut);
    assert_eq!(ChunkReader::open(&path).unwrap().read(1).unwrap(), body);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn broker_notice_expands_an_absorbed_frame_range_between_captures() {
    use continuum_broker::server::{Config, Server};
    use continuum_catalog::{Catalog, MemoryCatalog, NewChunk, NewPayload, StreamInfo, StreamKind};
    use continuum_client::{BrokerEvent, Client, ClientOptions, WorkerSpec};
    use continuum_core::{DomainLayer, HeaderBuf, HeaderSpec, PayloadFormat, StorageLocator, StorageTier};
    use std::time::Duration;

    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("broker.sock");
    let catalog = MemoryCatalog::new();
    let stream = catalog
        .register_stream(StreamInfo {
            name: "notice-video".into(),
            kind: StreamKind::Sensor,
            layer: DomainLayer::Layer1Raw,
            format: PayloadFormat::EncodedVideo,
            shape_name: "H264AnnexB".into(),
            shape_version: 1,
            rate_hz: Some(30.0),
            anchor: None,
        })
        .unwrap();
    let base =
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64 - 1_000_000_000;
    let starts = [base + 10 * MS, base + 20 * MS, base + 30 * MS];
    let ids = [uuid(10), uuid(20), uuid(30)];
    for (index, (&start, &id)) in starts.iter().zip(&ids).enumerate() {
        let mut header = HeaderSpec::raw(id, stream, start, start);
        header.seq = index as u32;
        catalog
            .insert(vec![NewPayload { header: HeaderBuf::build(&header).unwrap(), location: None, coverage: None }])
            .unwrap();
    }
    let chunk = uuid(100);
    catalog
        .commit_chunk(
            NewChunk {
                payload: NewPayload {
                    header: HeaderBuf::build(&HeaderSpec::raw(chunk, stream, starts[0], starts[2] + 1)).unwrap(),
                    location: Some(StorageLocator::Chunk {
                        tier: StorageTier::Tier2Cold,
                        uri: dir.path().join("source.cnk").to_string_lossy().into_owned(),
                        offset: 0,
                        length: 1,
                    }),
                    coverage: None,
                },
                frames: starts.iter().map(|&start_ns| continuum_catalog::ChunkFrame::new(start_ns, 0, 1)).collect(),
            },
            &ids,
        )
        .unwrap();
    let mut config = Config::new(socket.clone(), Duration::from_secs(30));
    config.forget.keeper_grace = Duration::ZERO;
    let server = Server::bind_with_catalog(config, Box::new(catalog)).await.unwrap();
    let (stop, stopped) = tokio::sync::oneshot::channel();
    let task = tokio::spawn(server.run_until(async {
        let _ = stopped.await;
    }));
    let client = Client::connect(&socket, ClientOptions::new("notice-test")).await.unwrap();
    client
        .register_worker(&WorkerSpec {
            worker_id: "tier:notice-test".into(),
            version_hash: "test".into(),
            capabilities: vec![],
            device_index: None,
            keeps_copies: true,
        })
        .await
        .unwrap();
    let purge = client
        .purge_range((base + 12 * MS, base + 13 * MS), &["notice-video".into()], "between captures")
        .await
        .unwrap()
        .purge_id;
    let (payloads, spans) = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if let Some(message) = client.next_event().await
                && let BrokerEvent::PurgeNotice { purge_id, payload_ids, spans } = message.event()
                && purge_id == purge
            {
                break (payload_ids, spans);
            }
        }
    })
    .await
    .unwrap();
    assert!(payloads.contains(&ids[0]));
    assert_eq!(
        spans.iter().filter(|s| s.stream_id == stream.0).map(|s| s.range).collect::<Vec<_>>(),
        [(base + 10 * MS, base + 20 * MS)],
        "keeper notice must already carry the selected original's complete slot"
    );
    assert!(client.get_chunk(chunk).await.unwrap().frames[0].cut);
    client.ack_purge(purge).unwrap();
    client.close("done").await.unwrap();
    stop.send(()).unwrap();
    task.await.unwrap();
}
