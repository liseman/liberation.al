//! The open's repair of a stream's current-era recording history, and bounding instant audio originals.
use continuum_catalog::*;
use continuum_core::{
    DomainLayer, HeaderBuf, HeaderSpec, Interval, PayloadFormat, StorageLocator, StorageTier, StreamId, Uuid,
};
use redb::{Database, TableDefinition};

fn id(n: u8) -> Uuid {
    Uuid([n; 16])
}
fn info() -> StreamInfo {
    StreamInfo {
        name: "fixture.audio".into(),
        kind: StreamKind::Sensor,
        layer: DomainLayer::Layer1Raw,
        format: PayloadFormat::RawBytes,
        shape_name: "PCM_F32LEx1".into(),
        shape_version: 1,
        rate_hz: Some(100.0),
        anchor: None,
    }
}
fn payload(n: u8, stream: StreamId, start: u64, end: u64) -> NewPayload {
    NewPayload {
        header: HeaderBuf::build(&HeaderSpec::raw(id(n), stream, start, end)).unwrap(),
        location: None,
        coverage: None,
    }
}
fn chunk(c: &dyn Catalog, stream: StreamId, n: u8, start: u64, end: u64, originals: &[Uuid]) {
    let mut own = payload(n, stream, start, end);
    own.location = Some(StorageLocator::Chunk {
        tier: StorageTier::Tier2Cold,
        uri: format!("fixture-{n}.cnk"),
        offset: 0,
        length: 100,
    });
    let frames = originals
        .iter()
        .enumerate()
        .map(|(index, raw)| {
            let Some(Resolved::Payload(original)) = c.get(*raw).unwrap() else { panic!("fixture raw input missing") };
            ChunkFrame::new(original.header.start_ns(), index as u64 * 10, 10)
        })
        .collect();
    c.commit_chunk(NewChunk { payload: own, frames }, originals).unwrap();
}
fn page(c: &dyn Catalog, stream: StreamId, start: u64, end: u64) -> RecordingPage {
    c.recording_page(stream, Interval::new(start, end).unwrap(), None, None, 128, 4096, MAX_RECORDING_BYTES).unwrap()
}
/// Removes one original's ledger row from a closed catalog file, as retention does once its source has gone.
fn drop_row(path: &std::path::Path, original: Uuid) {
    let db = Database::create(path).unwrap();
    let tx = db.begin_write().unwrap();
    {
        let mut ids = tx.open_table(TableDefinition::<&[u8], &[u8]>::new("recording_ids")).unwrap();
        let key = ids.remove(original.0.as_slice()).unwrap().unwrap().value().to_vec();
        let mut rows = tx.open_table(TableDefinition::<&[u8], &[u8]>::new("recording_rows")).unwrap();
        rows.remove(key.as_slice()).unwrap().unwrap();
    }
    tx.commit().unwrap();
}

/// A catalog whose microphone history an instant audio header marked unknown, with a chunk before it.
fn instant_marked(path: &std::path::Path) -> StreamId {
    let c = RedbCatalog::open(path).unwrap();
    let stream = c.register_stream(info()).unwrap();
    c.insert(vec![payload(1, stream, 100, 102)]).unwrap();
    chunk(&c, stream, 10, 100, 102, &[id(1)]);
    c.insert(vec![payload(2, stream, 500, 500)]).unwrap();
    assert!(!page(&c, stream, 100, 102).history_known, "the instant header marks the stream unknown");
    assert!(c.recording_restored().is_empty());
    stream
}

#[test]
fn a_stream_marked_unknown_by_an_instant_header_that_is_gone_is_restored_once_at_open() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("catalog.redb");
    let stream = instant_marked(&path);
    drop_row(&path, id(2));
    let c = RedbCatalog::open(&path).unwrap();
    assert_eq!(c.recording_restored(), [stream.0]);
    let proof = page(&c, stream, 100, 102);
    assert!(proof.history_known);
    assert_eq!(proof.inputs.iter().map(|input| input.chunk_id).collect::<Vec<_>>(), [id(10)]);
    drop(c);
    let c = RedbCatalog::open(&path).unwrap();
    assert!(c.recording_restored().is_empty(), "the repair is done once");
    assert!(page(&c, stream, 100, 102).history_known);
    // A new instant header marks it unknown again: the repair never weakens the rule itself.
    c.insert(vec![payload(3, stream, 600, 600)]).unwrap();
    assert!(!page(&c, stream, 100, 102).history_known);
}

#[test]
fn a_stream_whose_instant_header_is_still_in_its_ledger_stays_unknown() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("catalog.redb");
    let stream = instant_marked(&path);
    let c = RedbCatalog::open(&path).unwrap();
    assert!(c.recording_restored().is_empty());
    let held = c.recording_held();
    assert_eq!(held.len(), 1);
    assert_eq!(held[0].0, stream.0);
    assert!(held[0].1.contains("an instant audio header"), "{held:?}");
    assert!(!page(&c, stream, 100, 102).history_known);
}

#[test]
fn a_stream_with_an_original_whose_subjects_overflowed_stays_unknown() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("catalog.redb");
    let c = RedbCatalog::open(&path).unwrap();
    let stream = c.register_stream(info()).unwrap();
    let mut spec = HeaderSpec::raw(id(1), stream, 100, 102);
    spec.subjects = (0..129u8).map(|n| Uuid([n.wrapping_add(100); 16])).collect();
    spec.subjects.dedup();
    c.insert(vec![NewPayload { header: HeaderBuf::build(&spec).unwrap(), location: None, coverage: None }]).unwrap();
    assert!(!page(&c, stream, 100, 102).history_known);
    drop(c);
    let c = RedbCatalog::open(&path).unwrap();
    assert!(c.recording_restored().is_empty());
    assert!(!page(&c, stream, 100, 102).history_known);
}

#[test]
fn an_instant_original_flushed_into_a_chunk_still_keeps_its_stream_unknown_after_reopen() {
    // Thea's case: absorption must not drop the instant original's row while its audio still sits in a chunk.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("catalog.redb");
    let stream = instant_marked(&path);
    let c = RedbCatalog::open(&path).unwrap();
    chunk(&c, stream, 20, 500, 510, &[id(2)]);
    drop(c);
    let c = RedbCatalog::open(&path).unwrap();
    assert!(c.recording_restored().is_empty());
    assert!(!page(&c, stream, 100, 102).history_known);
}

#[test]
fn a_restore_bumps_the_revision_without_forgetting_anything() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("catalog.redb");
    let stream = instant_marked(&path);
    let c = RedbCatalog::open(&path).unwrap();
    let (revision, forget) = (c.invalidation_revision(stream).unwrap(), c.forget_revision(stream).unwrap());
    drop(c);
    drop_row(&path, id(2));
    let c = RedbCatalog::open(&path).unwrap();
    assert_eq!(c.recording_restored(), [stream.0]);
    assert!(c.invalidation_revision(stream).unwrap() > revision, "readers re-read");
    assert_eq!(c.forget_revision(stream).unwrap(), forget, "nothing was forgotten");
}

#[test]
fn no_repair_runs_while_a_range_purge_is_still_sweeping() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("catalog.redb");
    let stream = instant_marked(&path);
    let c = RedbCatalog::open(&path).unwrap();
    c.start_purge(
        PurgeSelector::Range { span: Interval::new(100, 102).unwrap(), streams: Some(vec![stream]) },
        "test",
        "synthetic",
        id(90),
        1,
    )
    .unwrap();
    drop(c);
    drop_row(&path, id(2));
    let c = RedbCatalog::open(&path).unwrap();
    assert!(c.recording_restored().is_empty());
    assert!(c.recording_held()[0].1.contains("still has to sweep it"), "{:?}", c.recording_held());
    assert!(!page(&c, stream, 100, 102).history_known);
}

#[test]
fn a_range_purge_still_sweeping_another_stream_does_not_hold_the_repair() {
    // Worker checkpoints and face.live are range-purged all the time; they never touch a microphone's history.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("catalog.redb");
    let stream = instant_marked(&path);
    let c = RedbCatalog::open(&path).unwrap();
    let other = c
        .register_stream(StreamInfo {
            name: "worker.checkpoint.fixture".into(),
            kind: StreamKind::Derived,
            format: PayloadFormat::Text,
            shape_name: "worker.checkpoint.fixture".into(),
            ..info()
        })
        .unwrap();
    c.start_purge(
        PurgeSelector::Range { span: Interval::new(100, 102).unwrap(), streams: Some(vec![other]) },
        "test",
        "synthetic",
        id(91),
        1,
    )
    .unwrap();
    drop(c);
    drop_row(&path, id(2));
    let c = RedbCatalog::open(&path).unwrap();
    assert_eq!(c.recording_restored(), [stream.0]);
    assert!(page(&c, stream, 100, 102).history_known);
}

#[test]
fn an_instant_original_bounded_by_its_own_samples_stops_holding_its_stream() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("catalog.redb");
    let stream = instant_marked(&path);
    let c = RedbCatalog::open(&path).unwrap();
    chunk(&c, stream, 20, 500, 510, &[id(2)]);
    let held = c.instant_originals().unwrap();
    assert_eq!(held, [InstantOriginal { stream, original: id(2), start_ns: 500, frame: Some((id(20), 0)) }]);
    assert!(c.bound_instant_original(id(2), 560).unwrap(), "nothing else holds the stream");
    assert!(c.instant_originals().unwrap().is_empty());
    let proof = page(&c, stream, 100, 102);
    assert!(proof.history_known);
    // The bounded original itself, and the chunk it sits in, stay uncertified.
    let around = page(&c, stream, 500, 600);
    assert!(around.history_known && around.unknown.iter().any(|span| span.start_ns() <= 500 && span.end_ns() >= 560));
    drop(c);
    let c = RedbCatalog::open(&path).unwrap();
    assert!(c.recording_held().is_empty() && page(&c, stream, 100, 102).history_known, "it survives reopen");
}

#[test]
fn only_an_instant_original_can_be_bounded_and_only_to_an_end_after_its_start() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("catalog.redb");
    let stream = instant_marked(&path);
    let c = RedbCatalog::open(&path).unwrap();
    assert!(c.bound_instant_original(id(1), 200).is_err(), "an original with its own extent");
    assert!(c.bound_instant_original(id(2), 500).is_err(), "an end at its start");
    assert!(c.bound_instant_original(id(99), 600).is_err(), "no such original");
    assert!(!page(&c, stream, 100, 102).history_known);
}
