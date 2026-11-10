//! What only the redb catalog does: survive a reopen, scrub purged bytes off disk, and swap files under readers.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use continuum_catalog::*;
use continuum_core::{DomainLayer, HeaderBuf, HeaderSpec, Interval, PayloadFormat, StreamId, Uuid};

fn id(n: u32) -> Uuid {
    let mut b = [0; 16];
    b[12..].copy_from_slice(&n.to_be_bytes());
    Uuid(b)
}

fn info(name: &str) -> StreamInfo {
    StreamInfo {
        name: name.into(),
        kind: StreamKind::Sensor,
        layer: DomainLayer::Layer1Raw,
        format: PayloadFormat::RawBytes,
        shape_name: "NV12".into(),
        shape_version: 1,
        rate_hz: Some(30.0),
        anchor: None,
    }
}

fn payload(n: u32, stream: StreamId, t: u64, parents: &[Uuid], subjects: &[Uuid]) -> NewPayload {
    let mut spec = HeaderSpec::raw(id(n), stream, t, t);
    spec.parents = parents.to_vec();
    spec.subjects = subjects.to_vec();
    NewPayload { header: HeaderBuf::build(&spec).unwrap(), location: None, coverage: None }
}

/// A fresh directory for a catalog file, and the file's path in it. Keep the directory alive for the test; it is
/// removed when dropped.
fn path() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("catalog.redb");
    (dir, path)
}

fn scrub_to_the_end(c: &RedbCatalog, budget: usize) -> Duration {
    for _ in 0..100_000 {
        let t = Instant::now();
        let p = c.scrub_step(budget).unwrap();
        if p == ScrubProgress::Swapped {
            return t.elapsed();
        }
        assert_ne!(p, ScrubProgress::Idle, "a due scrub never goes idle before it swaps");
    }
    panic!("the scrub never finished");
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

/// A subject id that can't appear in the file by accident.
const SUBJECT: Uuid = Uuid(*b"forget-me-please");

#[test]
fn a_scrub_leaves_no_byte_of_a_purged_header_in_the_file() {
    let (_dir, path) = path();
    let c = RedbCatalog::open(&path).unwrap();
    let s = c.register_stream(info("cam0")).unwrap();
    let d = c.register_stream(info("face.crop")).unwrap();
    c.insert((1..=50).map(|n| payload(n, s, u64::from(n), &[], &[])).collect()).unwrap();
    c.insert((1..=50).map(|n| payload(100 + n, d, u64::from(n), &[id(n)], &[SUBJECT])).collect()).unwrap();
    c.forget(ForgottenHash([7; 32])).unwrap();
    c.flush_durable().unwrap(); // commits reach the file once durable
    assert!(contains(&std::fs::read(&path).unwrap(), &SUBJECT.0));
    c.purge(PurgeSelector::Entity(SUBJECT), "forget", "", id(9_000), 1).unwrap();
    scrub_to_the_end(&c, 16);
    let bytes = std::fs::read(&path).unwrap();
    assert!(!contains(&bytes, &SUBJECT.0), "the subject id is gone from the file");
    assert!(c.get(id(1)).unwrap().is_some() && c.get(id(101)).unwrap().is_none());
    assert!(c.is_forgotten(ForgottenHash([7; 32])).unwrap(), "FORGOTTEN survives the copy");
    drop(c);
    let again = RedbCatalog::open(&path).unwrap();
    assert!(again.get(id(50)).unwrap().is_some(), "the swapped file is the catalog after a reopen");
    assert!(!again.scrub_needed().unwrap());
}

#[test]
fn readers_and_writers_keep_going_through_a_scrub_and_its_swap() {
    let tmp = path();
    let c = Arc::new(RedbCatalog::open(&tmp.1).unwrap());
    let s = c.register_stream(info("cam0")).unwrap();
    c.insert((1..=2_000).map(|n| payload(n, s, u64::from(n) * 10, &[], &[])).collect()).unwrap();
    c.request_scrub().unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let readers: Vec<_> = (0..4)
        .map(|_| {
            let (c, stop) = (Arc::clone(&c), Arc::clone(&stop));
            std::thread::spawn(move || {
                let mut reads = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    // Every read lands on the old file or the new one, whole, and never errors.
                    let page = c.range(s, Interval::new(0, u64::MAX).unwrap(), None, 100).unwrap();
                    assert_eq!(page.items.len(), 100);
                    assert!(c.get(id(1)).unwrap().is_some());
                    reads += 1;
                }
                reads
            })
        })
        .collect();
    let writer = {
        let c = Arc::clone(&c);
        std::thread::spawn(move || {
            for n in 0..300u32 {
                c.insert(vec![payload(10_000 + n, s, 1_000_000 + u64::from(n), &[], &[])]).unwrap();
            }
        })
    };
    scrub_to_the_end(&c, 64);
    writer.join().unwrap();
    stop.store(true, Ordering::Relaxed);
    for r in readers {
        assert!(r.join().unwrap() > 0);
    }
    for n in 0..300u32 {
        assert!(c.get(id(10_000 + n)).unwrap().is_some(), "a write during the scrub was carried over: {n}");
    }
}

/// Fills a catalog with `rows` frames, then scrubs it with a writer inserting all the while. Returns how long the swap
/// held the catalog alone, the slowest insert the writer saw while the scrub ran, and how many keys the swap carried.
fn swap_under_load(rows: u32) -> (Duration, Duration, usize) {
    let tmp = path();
    let c = Arc::new(RedbCatalog::open(&tmp.1).unwrap());
    let s = c.register_stream(info("cam0")).unwrap();
    for start in (0..rows).step_by(10_000) {
        let end = (start + 10_000).min(rows);
        c.insert((start..end).map(|n| payload(n + 1, s, u64::from(n), &[], &[])).collect()).unwrap();
    }
    c.request_scrub().unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let writer = {
        let (c, stop) = (Arc::clone(&c), Arc::clone(&stop));
        std::thread::spawn(move || {
            let mut slowest = Duration::ZERO;
            let mut n = 0u32;
            while !stop.load(Ordering::Relaxed) {
                let t = Instant::now();
                c.insert(vec![payload(rows + 10 + n, s, u64::from(rows + 10 + n), &[], &[])]).unwrap();
                slowest = slowest.max(t.elapsed());
                n += 1;
            }
            slowest
        })
    };
    scrub_to_the_end(&c, 5_000);
    stop.store(true, Ordering::Relaxed);
    let slowest = writer.join().unwrap();
    assert!(c.get(id(rows)).unwrap().is_some());
    (c.last_swap_hold().unwrap(), slowest, c.last_swap_carried().unwrap())
}

/// The swap's target, asserted only on request (CONTINUUM_LATENCY_STRICT=1, on an otherwise idle machine): a shared
/// box under load can delay one durable commit past it.
const SWAP_TARGET: Duration = Duration::from_millis(10);
/// The bound a swap that did real work under the lock would break, even on a loaded box.
const SWAP_CAP: Duration = Duration::from_millis(250);

#[test]
fn the_swap_is_brief_whatever_the_catalog_holds() {
    // What makes the swap brief isn't a timing: it is that the copy and the catch-up run outside the lock, so the
    // swap carries at most SWAP_DELTA written keys, however big the catalog and however busy its writer. That is
    // asserted exactly, with a writer inserting throughout so there is always something to carry. The time it
    // takes (one durable commit and a rename, about 5 ms on a quiet NVMe disk) is asserted against a cap load can't
    // excuse, and against the 10 ms target with CONTINUUM_LATENCY_STRICT=1.
    let (hold, _, carried) = swap_under_load(10_000);
    assert!(carried <= SWAP_DELTA, "the swap carried {carried} keys under the lock, over {SWAP_DELTA}");
    assert!(hold < SWAP_CAP, "the swap held the catalog for {hold:?}");
    if std::env::var("CONTINUUM_LATENCY_STRICT").as_deref() == Ok("1") {
        assert!(hold < SWAP_TARGET, "the swap held the catalog for {hold:?}, over the {SWAP_TARGET:?} target");
    }
}

/// The measurement for the PR, at a size given by CONTINUUM_SWAP_ROWS (default 1,000,000). Not run by default:
/// `cargo test -p continuum-catalog --test redb -- --ignored --nocapture`, niced.
#[test]
#[ignore = "a measurement, run by hand"]
fn swap_hold_at_scale() {
    let rows = std::env::var("CONTINUUM_SWAP_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(1_000_000);
    let (hold, slowest, carried) = swap_under_load(rows);
    println!(
        "rows {rows}: swap held the catalog {hold:?}, carrying {carried} keys; slowest insert during the scrub {slowest:?}"
    );
}

#[test]
fn a_purge_cut_short_by_a_restart_resumes() {
    let (_dir, path) = path();
    {
        let c = RedbCatalog::open(&path).unwrap();
        let s = c.register_stream(info("cam0")).unwrap();
        let d = c.register_stream(info("derived")).unwrap();
        c.insert((1..=20).map(|n| payload(n, s, u64::from(n), &[], &[SUBJECT])).collect()).unwrap();
        c.insert((1..=20).map(|n| payload(100 + n, d, u64::from(n), &[id(n)], &[])).collect()).unwrap();
        c.start_purge(PurgeSelector::Entity(SUBJECT), "forget", "", id(9_001), 1).unwrap();
        assert!(matches!(c.purge_step(id(9_001), 3).unwrap(), PurgeProgress::Running { .. }));
    }
    let c = RedbCatalog::open(&path).unwrap();
    assert_eq!(c.pending_purges().unwrap(), vec![id(9_001)]);
    assert!(c.get(id(1)).unwrap().is_none(), "what the first step hid stays hidden");
    let report = loop {
        if let PurgeProgress::Removed { report, .. } = c.purge_step(id(9_001), 5).unwrap() {
            break report;
        }
    };
    assert_eq!(report.removed.len(), 40);
    assert!(c.get(id(120)).unwrap().is_none());
}

#[test]
fn cooperative_purge_batches_rollback_cap_resume_and_match_single_row_steps() {
    let (dir, batch_path) = path();
    let prepare = |path: &std::path::Path| {
        let c = RedbCatalog::open(path).unwrap();
        let stream = c.register_stream(info("synthetic.live")).unwrap();
        c.insert((0..132).map(|n| payload(n, stream, u64::from(n), &[], &[])).collect()).unwrap();
        c.start_purge(
            PurgeSelector::Range { span: Interval::new(0, 128).unwrap(), streams: Some(vec![stream]) },
            "synthetic trim",
            "",
            id(9000),
            0,
        )
        .unwrap();
        c
    };
    let c = prepare(&batch_path);
    assert!(matches!(c.purge_batch(id(9000), 0, &mut || false), Err(CatalogError::Invalid(_))));
    let aborted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        c.purge_batch(id(9000), 1000, &mut || panic!("abort after the first chunk"))
    }));
    assert!(aborted.is_err());
    // The aborted transaction must not have advanced the stored cursor, even after reopening.
    drop(c);
    let c = RedbCatalog::open(&batch_path).unwrap();
    match c.purge_batch(id(9000), 1000, &mut || true).unwrap() {
        PurgeProgress::Running { done, delta, .. } => {
            assert_eq!(done, 64, "an already-expired deadline still makes exactly one chunk of progress");
            assert_eq!(delta.doomed.len(), 64, "the rolled-back chunk must be delivered again");
        }
        other => panic!("unexpected progress: {other:?}"),
    }
    match c.purge_batch(id(9000), 7, &mut || false).unwrap() {
        PurgeProgress::Running { done, delta, .. } => {
            assert_eq!(done, 71, "the operation ceiling also caps a partial chunk");
            assert_eq!(delta.doomed.len(), 7);
        }
        other => panic!("unexpected progress: {other:?}"),
    }
    drop(c);
    let c = RedbCatalog::open(&batch_path).unwrap();
    let report = match c.purge_batch(id(9000), 1000, &mut || false).unwrap() {
        PurgeProgress::Removed { report, .. } => report,
        other => panic!("the remaining phase work fits in one transaction: {other:?}"),
    };
    assert_eq!(report.removed.len(), 128);
    for n in 0..132 {
        assert_eq!(c.get(id(n)).unwrap().is_some(), n >= 128);
    }
    let ordinary = prepare(&dir.path().join("ordinary.redb"));
    let mut calls = 0;
    let ordinary_report = loop {
        calls += 1;
        if let PurgeProgress::Removed { report, .. } = ordinary.purge_step(id(9000), 1).unwrap() {
            break report;
        }
        assert!(calls < 1000);
    };
    assert_eq!(calls, 386, "explicit one-row steps preserve their existing semantics");
    assert_eq!(report, ordinary_report);
}

fn warm(uri: &str) -> continuum_core::StorageLocator {
    continuum_core::StorageLocator::Chunk {
        tier: continuum_core::StorageTier::Tier1Warm,
        uri: uri.into(),
        offset: 0,
        length: 1,
    }
}

#[test]
fn a_copy_hydrated_over_a_cut_slot_while_the_purge_finishes_is_purged_too() {
    let tmp = path();
    let c = RedbCatalog::open(&tmp.1).unwrap();
    let s = c.register_stream(info("cam0")).unwrap();
    c.insert((1..=3).map(|n| payload(n, s, u64::from(n - 1) * 33, &[], &[])).collect()).unwrap();
    let mut spec = HeaderSpec::raw(id(100), s, 0, 99);
    spec.seq = 1;
    let chunk = NewChunk {
        payload: NewPayload { header: HeaderBuf::build(&spec).unwrap(), location: Some(warm("x")), coverage: None },
        frames: [0, 33, 66].map(|t| ChunkFrame::new(t, t, 1)).to_vec(),
    };
    c.commit_chunk(chunk, &[id(1), id(2), id(3)]).unwrap();
    c.start_purge(
        PurgeSelector::Range { span: Interval::new(35, 40).unwrap(), streams: Some(vec![s]) },
        "t",
        "",
        id(900),
        1,
    )
    .unwrap();
    // Step into the finishing phase, where the cut is done but the purge isn't.
    loop {
        match c.purge_step(id(900), 1).unwrap() {
            PurgeProgress::Running { phase: PurgePhase::Finishing, .. } => break,
            PurgeProgress::Running { .. } => {}
            PurgeProgress::Removed { .. } => panic!("finished before its finishing phase"),
        }
    }
    let mut copy = HeaderSpec::raw(id(200), s, 40, 40);
    copy.parents = vec![id(100)];
    c.insert(vec![NewPayload { header: HeaderBuf::build(&copy).unwrap(), location: None, coverage: None }]).unwrap();
    assert!(c.get(id(200)).unwrap().is_none(), "a copy over the cut slot is hidden as it lands");
    let report = loop {
        if let PurgeProgress::Removed { report, .. } = c.purge_step(id(900), 1).unwrap() {
            break report;
        }
    };
    assert!(report.removed.iter().any(|r| r.id == id(200)));
    assert_eq!(c.purge_record(id(900)).unwrap().unwrap().report, Some(report), "the record reads the same report");
}

#[test]
fn a_purge_that_finishes_during_a_scrub_leaves_another_scrub_due() {
    let tmp = path();
    let c = RedbCatalog::open(&tmp.1).unwrap();
    let s = c.register_stream(info("cam0")).unwrap();
    c.insert((1..=200).map(|n| payload(n, s, u64::from(n), &[], &[SUBJECT])).collect()).unwrap();
    c.request_scrub().unwrap();
    assert!(matches!(c.scrub_step(16).unwrap(), ScrubProgress::Copying { .. }));
    assert!(matches!(c.scrub_step(16).unwrap(), ScrubProgress::Copying { .. }));
    // The scrub has copied some rows; a purge now removes them, which can leave their bytes in the fresh file.
    c.purge(PurgeSelector::Entity(SUBJECT), "forget", "", id(901), 1).unwrap();
    scrub_to_the_end(&c, 16);
    assert!(c.scrub_needed().unwrap(), "the next scrub clears what the purge left in the fresh file's freed pages");
    scrub_to_the_end(&c, 16);
    assert!(!c.scrub_needed().unwrap());
    assert!(!contains(&std::fs::read(&tmp.1).unwrap(), &SUBJECT.0), "and after it the subject id is gone");
}

#[test]
fn a_marked_swap_whose_rename_was_lost_is_finished_at_open_and_an_unmarked_copy_thrown_away() {
    let (_dir, path) = path();
    let scrub = {
        let mut p = path.as_os_str().to_owned();
        p.push(".scrub");
        std::path::PathBuf::from(p)
    };
    {
        let c = RedbCatalog::open(&path).unwrap();
        let s = c.register_stream(info("cam0")).unwrap();
        c.insert(vec![payload(1, s, 1, &[], &[])]).unwrap();
        c.request_scrub().unwrap();
        scrub_to_the_end(&c, 16);
        c.insert(vec![payload(2, s, 2, &[], &[])]).unwrap();
    }
    // A power loss that undid the rename (the marker, flushed before it, survives): the swapped file is back at
    // .scrub, an older copy at the catalog's path.
    std::fs::rename(&path, &scrub).unwrap();
    let done = {
        let mut p = path.as_os_str().to_owned();
        p.push(".scrub-done");
        std::path::PathBuf::from(p)
    };
    std::fs::write(&done, b"").unwrap();
    let older = path.with_extension("older");
    {
        let c = RedbCatalog::open(&older).unwrap();
        c.register_stream(info("cam0")).unwrap();
    }
    std::fs::rename(&older, &path).unwrap();
    let c = RedbCatalog::open(&path).unwrap();
    assert!(c.get(id(2)).unwrap().is_some(), "the swapped file, with the write made after the swap, is put back");
    assert!(!scrub.exists() && !done.exists());
    drop(c);
    // A scrub cut short before its swap leaves a fresh file with no marker: thrown away, without being opened.
    std::fs::write(&scrub, b"not a whole file").unwrap();
    let c = RedbCatalog::open(&path).unwrap();
    assert!(!scrub.exists() && c.get(id(2)).unwrap().is_some());
}

#[test]
fn a_purged_inline_body_is_gone_from_the_record_and_the_file_after_a_scrub() {
    let tmp = path();
    let c = RedbCatalog::open(&tmp.1).unwrap();
    let s = c.register_stream(info("events:cam0")).unwrap();
    let secret = b"alice told me the alarm code 4711";
    let body = continuum_core::StorageLocator::inline(secret.to_vec()).unwrap();
    let spec = HeaderSpec::raw(id(1), s, 10, 10);
    c.insert(vec![NewPayload { header: HeaderBuf::build(&spec).unwrap(), location: Some(body), coverage: None }])
        .unwrap();
    c.purge(PurgeSelector::Ids(vec![id(1)]), "forget", r#"{"stream":"camera:cam0"}"#, id(900), 1).unwrap();
    scrub_to_the_end(&c, 16);
    let record = c.purge_record(id(900)).unwrap().unwrap().report.unwrap();
    assert!(!format!("{record:?}").contains("4711"));
    assert!(!contains(&std::fs::read(&tmp.1).unwrap(), secret), "the scrubbed file holds no copy of the body");
}

#[test]
fn commit_sequences_carry_on_after_a_reopen() {
    let (_dir, path) = path();
    let last = {
        let c = RedbCatalog::open(&path).unwrap();
        let s = c.register_stream(info("cam0")).unwrap();
        c.insert((1..=3).map(|n| payload(n, s, u64::from(n), &[], &[])).collect()).unwrap()[2].seq
    };
    let c = RedbCatalog::open(&path).unwrap();
    assert_eq!(c.head_seq().unwrap(), last);
    let s = c.stream_id("cam0").unwrap().unwrap();
    assert_eq!(c.insert(vec![payload(4, s, 4, &[], &[])]).unwrap()[0].seq, last + 1);
}

#[test]
fn a_catalog_file_from_before_formats_is_refused() {
    let (_dir, path) = path();
    {
        // A pre-format catalog: tables with a stream in them, and no format stamp.
        let db = redb::Database::create(&path).unwrap();
        let txn = db.begin_write().unwrap();
        {
            let def: redb::TableDefinition<&[u8], &[u8]> = redb::TableDefinition::new("streams");
            txn.open_table(def).unwrap().insert(&[0u8, 0, 0, 1][..], &b"old"[..]).unwrap();
        }
        txn.commit().unwrap();
    }
    assert!(matches!(RedbCatalog::open(&path), Err(CatalogError::Storage(_))));
}

#[test]
fn keepers_and_acknowledgements_survive_a_reopen() {
    let (_tmp, path) = path();
    {
        let c = RedbCatalog::open(&path).unwrap();
        c.add_keeper("tier-worker").unwrap();
        c.add_keeper("vector-worker").unwrap();
        let cam = c.register_stream(info("cam")).unwrap();
        c.insert(vec![payload(1, cam, 10, &[], &[])]).unwrap();
        c.start_purge(PurgeSelector::Entity(id(1)), "t", "", id(9), 1).unwrap();
        c.purge_acked(id(9), "tier-worker").unwrap();
        assert!(c.forget_keeper("vector-worker").unwrap());
    }
    let again = RedbCatalog::open(&path).unwrap();
    assert_eq!(again.keepers().unwrap(), ["tier-worker"]);
    assert_eq!(again.purge_record(id(9)).unwrap().unwrap().acked, ["tier-worker"]);
}

fn disk_format(path: &std::path::Path) -> u64 {
    use ::redb::{Database, ReadableDatabase, TableDefinition};
    const META: TableDefinition<&[u8], &[u8]> = TableDefinition::new("meta");
    let db = Database::open(path).unwrap();
    let tx = db.begin_read().unwrap();
    let value = tx.open_table(META).unwrap().get(b"format".as_slice()).unwrap().unwrap().value().to_vec();
    u64::from_be_bytes(value.try_into().unwrap())
}

/// Commits frames `first` and `first + 1` of `cam` into chunk `chunk`, with provenance or without.
fn commit_pair(c: &RedbCatalog, cam: StreamId, first: u32, chunk: u32, provenance: Option<&ChunkProvenance>) {
    let (a, b) = (u64::from(first) * 100, u64::from(first) * 100 + 33);
    c.insert(vec![payload(first, cam, a, &[], &[]), payload(first + 1, cam, b, &[], &[])]).unwrap();
    let chunk = NewChunk {
        payload: NewPayload {
            header: HeaderBuf::build(&HeaderSpec::raw(id(chunk), cam, a, b + 1)).unwrap(),
            location: Some(continuum_core::StorageLocator::Chunk {
                tier: continuum_core::StorageTier::Tier1Warm,
                uri: format!("file:///c/{chunk}"),
                offset: 0,
                length: 600,
            }),
            coverage: None,
        },
        frames: vec![ChunkFrame::new(a, 0, 300), ChunkFrame::new(b, 300, 300)],
    };
    c.commit_chunk_with_provenance(chunk, &[id(first), id(first + 1)], provenance).unwrap();
}

#[test]
fn frame_provenance_survives_reopen_and_scrub() {
    let (_dir, path) = path();
    let provenance = ChunkProvenance::new(vec![
        FrameProvenance { device_seq: 41, epoch: 0x5eed, stamped_at_arrival: false },
        FrameProvenance { device_seq: 42, epoch: 0x5eed, stamped_at_arrival: true },
    ]);
    let cam = {
        let c = RedbCatalog::open(&path).unwrap();
        let cam = c.register_stream(info("cam")).unwrap();
        commit_pair(&c, cam, 1, 100, None);
        cam
    };
    assert_eq!(disk_format(&path), 8);
    {
        let c = RedbCatalog::open(&path).unwrap();
        commit_pair(&c, cam, 3, 101, Some(&provenance));
    }
    assert_eq!(disk_format(&path), 8);
    {
        let c = RedbCatalog::open(&path).unwrap();
        assert_eq!(c.chunk_provenance(id(101)).unwrap(), Some(provenance.clone()));
        assert_eq!(c.chunk_provenance(id(100)).unwrap(), None);
        c.request_scrub().unwrap();
        scrub_to_the_end(&c, 1);
        assert_eq!(c.chunk_provenance(id(101)).unwrap(), Some(provenance.clone()), "the scrub copied it");
    }
    assert_eq!(disk_format(&path), 8);
    assert_eq!(RedbCatalog::open(&path).unwrap().chunk_provenance(id(101)).unwrap(), Some(provenance));
}

/// Writes `format` into a closed catalog file's META by hand, or removes the stamp when `None`.
fn stamp(path: &std::path::Path, format: Option<u64>) {
    use ::redb::{Database, TableDefinition};
    let db = Database::open(path).unwrap();
    let tx = db.begin_write().unwrap();
    {
        let mut meta = tx.open_table(TableDefinition::<&[u8], &[u8]>::new("meta")).unwrap();
        match format {
            Some(f) => drop(meta.insert(b"format".as_slice(), f.to_be_bytes().as_slice()).unwrap()),
            None => drop(meta.remove(b"format".as_slice()).unwrap()),
        }
    }
    tx.commit().unwrap();
}

fn refusal(path: &std::path::Path) -> String {
    match RedbCatalog::open(path) {
        Err(CatalogError::Storage(e)) => e.to_string(),
        Err(other) => panic!("expected a storage error, got {other:?}"),
        Ok(_) => panic!("the catalog opened"),
    }
}

#[test]
fn a_new_catalog_is_created_at_format_8_and_reopens() {
    let (_dir, file) = path();
    let cam = {
        let c = RedbCatalog::open(&file).unwrap();
        assert!(c.restart_stamp().unwrap().is_some(), "restart tracking starts with the file");
        c.register_stream(info("cam")).unwrap()
    };
    assert_eq!(disk_format(&file), 8);
    let c = RedbCatalog::open(&file).unwrap();
    assert_eq!(c.stream_id("cam").unwrap(), Some(cam));
    drop(c);
    assert_eq!(disk_format(&file), 8);
}

#[test]
fn a_file_stamped_with_another_format_is_refused() {
    for format in [1, 4, 6, 7] {
        let (_dir, file) = path();
        drop(RedbCatalog::open(&file).unwrap());
        stamp(&file, Some(format));
        assert_eq!(
            refusal(&file),
            format!(
                "the catalog file is format {format}; this code reads only format 8: create it again (pack-reset-catalog)"
            )
        );
        assert_eq!(disk_format(&file), format, "a refused file is left as it was");
    }
}

#[test]
fn an_unstamped_file_with_a_stream_is_refused() {
    let (_dir, file) = path();
    {
        let c = RedbCatalog::open(&file).unwrap();
        c.register_stream(info("cam")).unwrap();
    }
    stamp(&file, None);
    assert!(refusal(&file).ends_with("this code reads only format 8: create it again (pack-reset-catalog)"));
}

/// Bytes per table: what one hour of a per-frame derived stream (`animal.detection`: about 31,000 inline instants
/// 120 ms apart, about 160 bytes each) takes kept, and what it leaves once aged in `pack-catalog-age`'s batches of
/// 5,000, with `merge_gap_ns` and, when `complete`, every keeper having acknowledged each purge.
fn aged_hour(merge_gap_ns: u64, complete: bool) -> (BTreeMap<String, i64>, BTreeMap<String, i64>) {
    const ROWS: u32 = 31_000;
    const STEP_NS: u64 = 120_000_000;
    let (_dir, file) = path();
    let c = RedbCatalog::open(&file).unwrap();
    let mut stream_info = info("animal.detection");
    stream_info.kind = StreamKind::Derived;
    let s = c.register_stream(stream_info).unwrap();
    let t0 = 1_800_000_000_000_000_000u64;
    let sum = |c: &RedbCatalog| -> BTreeMap<String, i64> {
        c.table_bytes().unwrap().into_iter().map(|(name, b)| (name, i64::try_from(b).unwrap())).collect()
    };
    let empty = sum(&c);
    for start in (0..ROWS).step_by(1000) {
        let rows = (start..(start + 1000).min(ROWS))
            .map(|n| {
                let t = t0 + u64::from(n) * STEP_NS;
                NewPayload {
                    header: HeaderBuf::build(&HeaderSpec::raw(id(n + 1), s, t, t)).unwrap(),
                    location: Some(continuum_core::StorageLocator::Inline { bytes: vec![b'x'; 160] }),
                    coverage: None,
                }
            })
            .collect::<Vec<_>>();
        c.insert(rows).unwrap();
    }
    let kept = sum(&c);
    let later = t0 + 2 * 3_600_000_000_000;
    c.mark_commit_clock(later).unwrap();
    let by = c.commit_seq_at(later).unwrap();
    for (batch, start) in (0..ROWS).step_by(5000).enumerate() {
        let ids = (start..(start + 5000).min(ROWS)).map(|n| id(n + 1)).collect();
        let purge = id(1_000_000 + u32::try_from(batch).unwrap());
        let selector = PurgeSelector::AgeOut { ids, committed_by_seq: by, merge_gap_ns };
        c.purge(selector, "aged out: animal.detection", "{}", purge, 100_000).unwrap();
        if complete {
            c.complete_purge(purge, later).unwrap();
        }
    }
    let aged = sum(&c);
    let less = |m: &BTreeMap<String, i64>| -> BTreeMap<String, i64> {
        m.iter().map(|(k, v)| (k.clone(), v - empty.get(k).copied().unwrap_or(0))).filter(|(_, v)| *v != 0).collect()
    };
    (less(&kept), less(&aged))
}

/// Not a check: prints what one aged hour leaves in the catalog, as each payload's own span with the report kept
/// and as runs with the report collapsed once acknowledged. Run with `--ignored --nocapture`.
#[test]
#[ignore = "a measurement"]
fn measure_the_catalog_bytes_an_aged_hour_leaves() {
    for (label, gap, complete) in
        [("each span, report kept", 0, false), ("2 s runs, report collapsed", 2_000_000_000, true)]
    {
        let (kept, aged) = aged_hour(gap, complete);
        println!("{label}: kept {} B, aged {} B", kept.values().sum::<i64>(), aged.values().sum::<i64>());
        for (name, b) in &aged {
            println!("  {name:24} {b:>10}");
        }
    }
}

/// Bytes per table that `minutes` of one 30 fps camera leave in the catalog once its `FRAMES` rows are pruned (the
/// steady state an hour on), and the catalog writes it took: its frames ingested 2 seconds at a time, each batch
/// then committed as a chunk of its own (`chunk_s` 2, how video was stored before #49) or as a part of a chunk
/// `chunk_s` seconds long. Capture times jitter by up to 100 microseconds and access units are 300 to 3,300 bytes.
fn camera_minutes(minutes: u32, chunk_s: u32) -> (BTreeMap<String, u64>, u32) {
    const FPS: u32 = 30;
    const PART_S: u32 = 2;
    let (_dir, file) = path();
    let c = RedbCatalog::open(&file).unwrap();
    let cam = c.register_stream(info("cam0.video")).unwrap();
    let empty: BTreeMap<String, u64> = c.table_bytes().unwrap().into_iter().collect();
    let t0 = 1_800_000_000_000_000_000u64;
    let mut noise = 0x9e37_79b9_7f4a_7c15u64;
    let (mut chunk, mut start, mut held, mut at, mut writes) = (0u32, 0u64, 0u32, 0u64, 0u32);
    for part in 0..minutes * 60 / PART_S {
        if part % (chunk_s / PART_S) == 0 {
            (chunk, held, at) = (20_000_000 + part, 0, 24);
        }
        let first = part * PART_S * FPS;
        let mut tail = Vec::new();
        let mut rows = Vec::new();
        for n in first..first + PART_S * FPS {
            noise ^= noise << 13;
            noise ^= noise >> 7;
            noise ^= noise << 17;
            let t = t0 + u64::from(n) * 33_333_333 + (noise >> 40) % 100_000;
            let length = 300 + noise % 3_000;
            rows.push(payload(n + 1, cam, t, &[], &[]));
            tail.push(ChunkFrame::new(t, at + 45, length));
            at += 45 + length;
        }
        c.insert(rows).unwrap();
        if held == 0 {
            start = tail[0].start_ns;
        }
        let header = HeaderBuf::build(&HeaderSpec::raw(id(chunk), cam, start, tail[tail.len() - 1].start_ns + 1));
        let payload = NewPayload {
            header: header.unwrap(),
            location: Some(continuum_core::StorageLocator::Chunk {
                tier: continuum_core::StorageTier::Tier1Warm,
                uri: format!("/srv/pack/data/state/continuum/tier/streams/7/2027011500/7-{start}-{chunk}.cnk"),
                offset: 0,
                length: at,
            }),
            coverage: None,
        };
        let ids: Vec<Uuid> = (first..first + PART_S * FPS).map(|n| id(n + 1)).collect();
        let provenance = ChunkProvenance::new(
            (first..first + PART_S * FPS)
                .map(|n| FrameProvenance { device_seq: u64::from(n), epoch: 7, stamped_at_arrival: false })
                .collect(),
        );
        if held == 0 {
            c.commit_chunk_with_provenance(NewChunk { payload, frames: tail }, &ids, Some(&provenance)).unwrap();
        } else {
            c.extend_chunk(NewChunk { payload, frames: tail }, held, &ids, Some(&provenance)).unwrap();
        }
        held += PART_S * FPS;
        writes += 2;
    }
    let mut after = None;
    loop {
        after = c.prune_frames(u64::MAX, after, 50_000).unwrap().next;
        if after.is_none() {
            break;
        }
    }
    let left = c.table_bytes().unwrap().into_iter();
    (left.map(|(name, b)| (name.clone(), b.saturating_sub(empty.get(&name).copied().unwrap_or(0)))).collect(), writes)
}

/// Not a check: prints what an hour of one camera leaves in the catalog as two-second chunks and as one-minute chunks
/// committed in two-second parts (#49). Run with `--ignored --nocapture`.
#[test]
#[ignore = "a measurement"]
fn measure_the_catalog_bytes_a_camera_hour_leaves() {
    for (label, chunk_s) in [("2 s chunks", 2), ("60 s chunks in 2 s parts", 60)] {
        let (left, writes) = camera_minutes(60, chunk_s);
        println!("{label}: {} B in {writes} catalog writes", left.values().sum::<u64>());
        for (name, b) in left.iter().filter(|(_, b)| **b > 0) {
            println!("  {name:24} {b:>10}");
        }
    }
}

#[test]
fn a_camera_minute_as_one_chunk_in_parts_leaves_a_fraction_of_what_two_second_chunks_do() {
    let (short, _) = camera_minutes(2, 2);
    let (long, writes) = camera_minutes(2, 60);
    let (short, long): (u64, u64) = (short.values().sum(), long.values().sum());
    assert!(long * 3 < short, "one row a minute, not thirty: {long} B against {short} B");
    assert_eq!(writes, 2 * 60, "an ingest and a commit every two seconds, as before");
}

#[test]
fn an_aged_hour_leaves_runs_and_no_per_item_report_once_acknowledged() {
    let (kept, aged) = aged_hour(2_000_000_000, true);
    assert!(!aged.contains_key("purge_removed"), "the per-item report collapsed: {aged:?}");
    let left: i64 = aged.values().sum();
    assert!(left * 1000 < kept.values().sum::<i64>(), "under a thousandth of the hour stays: {left} B");
}

#[test]
fn storage_stats_name_every_table_with_its_rows_and_pages() {
    let (_dir, file) = path();
    let c = RedbCatalog::open(&file).unwrap();
    let s = c.register_stream(info("cam0")).unwrap();
    c.insert((1..=300).map(|n| payload(n, s, u64::from(n), &[], &[])).collect()).unwrap();
    let stats = c.storage_stats().unwrap().expect("a file catalog has stats");
    assert_eq!(stats.page_size, 4096);
    assert!(stats.disk_bytes > 0 && stats.file_bytes >= stats.disk_bytes / 2, "{stats:?}");
    let payloads = stats.tables.iter().find(|t| t.name == "payloads").expect("the payloads table");
    assert_eq!(payloads.rows, 300);
    assert!(payloads.leaf_pages > 0 && payloads.stored_bytes > 300 * 16);
    let pages = |t: &TableStorage| t.leaf_pages + t.branch_pages;
    assert!(stats.tables.windows(2).all(|w| pages(&w[0]) >= pages(&w[1])), "largest first");
    assert_eq!(MemoryCatalog::new().storage_stats().unwrap(), None, "no file, no stats");
}

#[test]
fn relativize_rewrites_chunk_locators_under_any_data_root_once() {
    use continuum_core::{StorageLocator, StorageTier};
    let (_dir, file) = path();
    let chunk =
        |uri: &str| StorageLocator::Chunk { tier: StorageTier::Tier1Warm, uri: uri.into(), offset: 0, length: 9 };
    let rows = [
        ("/old/checkout/data/state/continuum/tier/1/h/a.cnk", "continuum/tier/1/h/a.cnk"),
        ("file:///elsewhere/state/continuum/tier2/1/b.cnk", "continuum/tier2/1/b.cnk"),
        ("data/state/continuum/tier/1/h/c.cnk", "continuum/tier/1/h/c.cnk"),
        ("continuum/tier/1/h/d.cnk", "continuum/tier/1/h/d.cnk"),
        ("/media/v1/clip.mp4", "/media/v1/clip.mp4"),
    ];
    let stream = {
        let c = RedbCatalog::open(&file).unwrap();
        let stream = c.register_stream(info("cam.video")).unwrap();
        let payloads = rows.iter().enumerate().map(|(n, (old, _))| NewPayload {
            location: Some(chunk(old)),
            ..payload(n as u32 + 1, stream, 100 + n as u64, &[], &[])
        });
        c.insert(payloads.collect()).unwrap();
        let mut after: Option<Vec<u8>> = None;
        let (mut scanned, mut rewritten) = (0, 0);
        loop {
            // Two rows a transaction, so the run crosses batch boundaries.
            let batch = c.relativize_chunk_locators(after.as_deref(), 2).unwrap();
            scanned += batch.scanned;
            rewritten += batch.rewritten;
            match batch.next {
                Some(next) => after = Some(next),
                None => break,
            }
        }
        assert_eq!((scanned, rewritten), (5, 3));
        let again = c.relativize_chunk_locators(None, 100).unwrap();
        assert_eq!((again.scanned, again.rewritten, again.next), (5, 0, None), "a second run changes nothing");
        c.close(Duration::from_secs(5)).unwrap();
        stream
    };
    let c = RedbCatalog::open(&file).unwrap();
    for (n, (_, want)) in rows.iter().enumerate() {
        let Some(Resolved::Payload(p)) = c.get(id(n as u32 + 1)).unwrap() else { panic!("payload {n} is gone") };
        assert_eq!(p.header.stream(), stream);
        assert_eq!(p.locations, vec![chunk(want)], "row {n}, after a reopen");
    }
}

#[test]
fn the_migration_tool_rewrites_a_stopped_catalog_and_refuses_while_the_broker_holds_it() {
    use continuum_core::{StorageLocator, StorageTier};
    let (dir, file) = path();
    let old = "/old/checkout/data/state/continuum/tier/1/h/a.cnk";
    {
        let c = RedbCatalog::open(&file).unwrap();
        let stream = c.register_stream(info("mic.audio")).unwrap();
        let location = StorageLocator::Chunk { tier: StorageTier::Tier1Warm, uri: old.into(), offset: 0, length: 9 };
        c.insert(vec![NewPayload { location: Some(location), ..payload(1, stream, 100, &[], &[]) }]).unwrap();
        c.close(Duration::from_secs(5)).unwrap();
    }
    let tool = || {
        std::process::Command::new(env!("CARGO_BIN_EXE_continuum-relative-locators"))
            .arg("--catalog")
            .arg(&file)
            .output()
            .unwrap()
    };
    let lock = std::fs::File::create(dir.path().join("broker.sock.lock")).unwrap();
    lock.lock().unwrap();
    let refused = tool();
    assert!(!refused.status.success(), "a running broker's catalog is left alone");
    assert!(String::from_utf8_lossy(&refused.stderr).contains("stop it first"));
    lock.unlock().unwrap();
    let first = tool();
    assert!(first.status.success(), "{}", String::from_utf8_lossy(&first.stderr));
    assert!(
        String::from_utf8_lossy(&first.stdout).starts_with("1 location rows looked at, 1 chunk locators rewritten")
    );
    let second = tool();
    assert!(
        String::from_utf8_lossy(&second.stdout).starts_with("1 location rows looked at, 0 chunk locators rewritten")
    );
    let c = RedbCatalog::open(&file).unwrap();
    let Some(Resolved::Payload(p)) = c.get(id(1)).unwrap() else { panic!("the payload is gone") };
    assert_eq!(
        p.locations,
        vec![StorageLocator::Chunk {
            tier: StorageTier::Tier1Warm,
            uri: "continuum/tier/1/h/a.cnk".into(),
            offset: 0,
            length: 9
        }]
    );
}
