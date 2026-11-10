//! Original microphone applicability survives normal aging, while erasure and legacy absence stay unknown.
use continuum_catalog::*;
use continuum_core::{
    BootId, DomainLayer, HeaderBuf, HeaderSpec, Interval, PayloadFormat, StorageLocator, StorageTier, StreamId, Uuid,
};

fn id(n: u16) -> Uuid {
    let mut bytes = [0; 16];
    bytes[14..].copy_from_slice(&n.to_be_bytes());
    Uuid(bytes)
}
fn iv(start: u64, end: u64) -> Interval {
    Interval::new(start, end).unwrap()
}
fn info() -> StreamInfo {
    StreamInfo {
        name: "mic.audio".into(),
        kind: StreamKind::Sensor,
        layer: DomainLayer::Layer1Raw,
        format: PayloadFormat::RawBytes,
        shape_name: "PCM_F32LEx2".into(),
        shape_version: 1,
        rate_hz: Some(50.0),
        anchor: None,
    }
}
fn payload(n: u16, stream: StreamId, start: u64, end: u64, subjects: &[Uuid]) -> NewPayload {
    let mut spec = HeaderSpec::raw(id(n), stream, start, end);
    spec.subjects = subjects.to_vec();
    NewPayload { header: HeaderBuf::build(&spec).unwrap(), location: None, coverage: None }
}
fn location(n: u16, tier: StorageTier) -> StorageLocator {
    StorageLocator::Chunk { tier, uri: format!("synthetic-{n}.cnk"), offset: 0, length: 100 }
}
fn page(c: &dyn Catalog, stream: StreamId, window: Interval) -> RecordingPage {
    c.recording_page(stream, window, None, None, 128, 4096, MAX_RECORDING_BYTES).unwrap()
}
fn chunk(
    c: &dyn Catalog,
    stream: StreamId,
    n: u16,
    start: u64,
    subjects: &[Uuid],
) -> (Uuid, Vec<ChunkFrame>, StorageLocator) {
    let raw = [n * 10 + 1, n * 10 + 2];
    c.insert(vec![
        payload(raw[0], stream, start, start + 10, subjects),
        payload(raw[1], stream, start + 20, start + 30, subjects),
    ])
    .unwrap();
    let frames = vec![ChunkFrame::new(start, 0, 10), ChunkFrame::new(start + 20, 10, 10)];
    let mut own = payload(n + 1000, stream, start, start + 30, &[]);
    let at = location(n, StorageTier::Tier1Warm);
    own.location = Some(at.clone());
    let chunk = own.header.id();
    c.commit_chunk(NewChunk { payload: own, frames: frames.clone() }, &raw.map(id)).unwrap();
    (chunk, frames, at)
}
fn both(test: impl Fn(&dyn Catalog)) {
    test(&MemoryCatalog::new());
    let dir = tempfile::tempdir().unwrap();
    test(&RedbCatalog::open(dir.path().join("catalog.redb")).unwrap());
}
fn erase(c: &dyn Catalog, selector: PurgeSelector, n: u16) {
    c.purge(selector, "test", "synthetic", id(n), 10000).unwrap();
}

#[test]
fn raw_inputs_are_unknown_until_bound_and_original_gaps_survive_an_age_out() {
    both(|c| {
        let stream = c.register_stream(info()).unwrap();
        c.insert(vec![payload(90, stream, 1, 3, &[])]).unwrap();
        assert_eq!(page(c, stream, iv(0, 5)).unknown, [iv(1, 3)]);
        let (input, _, _) = chunk(c, stream, 1, 10, &[]);
        let before = page(c, stream, iv(10, 40));
        assert!(before.history_known && before.unknown.is_empty());
        assert_eq!(before.inputs[0].spans, [iv(10, 20), iv(30, 40)]);
        assert_eq!((before.inputs[0].channels, before.inputs[0].shape_name.as_str()), (Some(2), "PCM_F32LEx2"));
        erase(
            c,
            PurgeSelector::AgeOut { ids: vec![input], committed_by_seq: c.head_seq().unwrap(), merge_gap_ns: 0 },
            9000,
        );
        assert!(c.get(input).unwrap().is_none());
        assert_eq!(page(c, stream, iv(10, 40)).inputs, before.inputs);
    });
}

#[test]
fn purged_siblings_and_all_erased_history_never_become_empty_but_unrelated_windows_remain_usable() {
    both(|c| {
        let stream = c.register_stream(info()).unwrap();
        let (first, _, _) = chunk(c, stream, 1, 10, &[]);
        let (second, _, _) = chunk(c, stream, 2, 100, &[]);
        erase(c, PurgeSelector::Ids(vec![second]), 9001);
        let unaffected = page(c, stream, iv(10, 40));
        assert_eq!(unaffected.inputs[0].chunk_id, first);
        assert!(unaffected.unknown.is_empty());
        assert_eq!(page(c, stream, iv(100, 130)).unknown, [iv(100, 110), iv(120, 130)]);
        erase(c, PurgeSelector::Ids(vec![first]), 9002);
        let empty = page(c, stream, iv(10, 130));
        assert!(empty.inputs.is_empty() && !empty.unknown.is_empty());
    });
}

#[test]
fn privacy_range_and_entity_erasure_still_reach_original_witnesses_after_safe_ageout() {
    both(|c| {
        let stream = c.register_stream(info()).unwrap();
        let (first, _, _) = chunk(c, stream, 1, 10, &[id(80)]);
        let (second, _, _) = chunk(c, stream, 2, 100, &[]);
        erase(
            c,
            PurgeSelector::AgeOut { ids: vec![first], committed_by_seq: c.head_seq().unwrap(), merge_gap_ns: 0 },
            9010,
        );
        let revision = page(c, stream, iv(0, 200)).revision;
        erase(c, PurgeSelector::Range { span: iv(15, 17), streams: Some(vec![stream]) }, 9011);
        assert_eq!(page(c, stream, iv(15, 17)).unknown, [iv(15, 17)]);
        assert!(c.recording_page(stream, iv(0, 200), None, Some(revision), 128, 4096, MAX_RECORDING_BYTES).is_err());
        let (third, _, _) = chunk(c, stream, 3, 200, &[id(81)]);
        erase(
            c,
            PurgeSelector::AgeOut { ids: vec![third], committed_by_seq: c.head_seq().unwrap(), merge_gap_ns: 0 },
            9013,
        );
        erase(c, PurgeSelector::Entity(id(81)), 9012);
        assert!(!page(c, stream, iv(200, 230)).unknown.is_empty());
        let unaffected = page(c, stream, iv(100, 130));
        assert_eq!(unaffected.inputs[0].chunk_id, second);
        assert!(unaffected.unknown.is_empty());
    });
}

#[test]
fn source_anchor_replacement_advances_both_guards_without_relabeling_historical_inputs() {
    both(|c| {
        let stream = c.register_stream(info()).unwrap();
        chunk(c, stream, 1, 10, &[]);
        let before = page(c, stream, iv(0, 100));
        let registration = c.stream_generation().unwrap();
        let mut changed = info();
        changed.anchor = Some(ClockAnchor { boot: BootId([7; 16]), mono_ns: 12, utc_ns: 100 });
        assert_eq!(c.register_stream(changed).unwrap(), stream);
        let after = page(c, stream, iv(0, 100));
        assert!(after.revision > before.revision && c.stream_generation().unwrap() > registration);
        assert_eq!((after.source_generation, after.inputs[0].source_generation), (2, 1));
        assert!(
            c.recording_page(stream, iv(0, 100), None, Some(before.revision), 128, 4096, MAX_RECORDING_BYTES).is_err()
        );
    });
}

#[test]
fn unknown_and_input_rows_share_cursor_scan_and_byte_budgets() {
    both(|c| {
        let stream = c.register_stream(info()).unwrap();
        c.insert(vec![payload(90, stream, 1, 3, &[])]).unwrap();
        let (input, _, _) = chunk(c, stream, 1, 10, &[]);
        let first = c.recording_page(stream, iv(0, 100), None, None, 1, 4096, MAX_RECORDING_BYTES).unwrap();
        assert_eq!(first.unknown, [iv(1, 3)]);
        assert!(first.next.is_some());
        let second = c
            .recording_page(stream, iv(0, 100), first.next, Some(first.revision), 1, 4096, MAX_RECORDING_BYTES)
            .unwrap();
        assert_eq!(second.inputs[0].chunk_id, input);
        assert!(second.next.is_none());
        assert_eq!(first.through_seq, c.head_seq().unwrap());
        let scan = c.recording_page(stream, iv(0, 100), None, None, 128, 1, MAX_RECORDING_BYTES).unwrap();
        assert_eq!(scan.scanned, 1);
        assert!(scan.next.is_some());
        assert!(c.recording_page(stream, iv(10, 40), None, None, 128, 4096, 128).is_err());
        assert!(c.recording_page(stream, iv(0, 100), None, None, 129, 4096, MAX_RECORDING_BYTES).is_err());
        assert!(c.recording_page(stream, iv(0, 100), None, None, 128, 4097, MAX_RECORDING_BYTES).is_err());
    });
}

#[test]
fn last_copy_loss_is_unknown_while_redundant_retirement_and_relocation_preserve_inputs() {
    both(|c| {
        let stream = c.register_stream(info()).unwrap();
        let (input, frames, source) = chunk(c, stream, 1, 10, &[]);
        let before = page(c, stream, iv(10, 40));
        let cold = location(2, StorageTier::Tier2Cold);
        c.set_location(input, cold.clone()).unwrap();
        c.clear_location(input, source.tier()).unwrap();
        assert_eq!(page(c, stream, iv(10, 40)).inputs, before.inputs);
        c.rewrite_chunk(input, frames, location(3, StorageTier::Tier2Cold)).unwrap();
        assert_eq!(page(c, stream, iv(10, 40)).inputs, before.inputs);
        c.clear_location(input, StorageTier::Tier2Cold).unwrap();
        assert!(!page(c, stream, iv(10, 40)).unknown.is_empty());
    });
}

#[test]
fn durable_witnesses_and_unknown_masks_survive_scrub_and_restart() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("catalog.redb");
    let c = RedbCatalog::open(&path).unwrap();
    let stream = c.register_stream(info()).unwrap();
    chunk(&c, stream, 1, 10, &[]);
    erase(&c, PurgeSelector::Range { span: iv(15, 17), streams: Some(vec![stream]) }, 9020);
    let before = page(&c, stream, iv(0, 200));
    c.request_scrub().unwrap();
    for _ in 0..10000 {
        if c.scrub_step(16).unwrap() == ScrubProgress::Swapped {
            break;
        }
    }
    assert!(!c.scrub_needed().unwrap());
    drop(c);
    let c = RedbCatalog::open(&path).unwrap();
    assert_eq!(page(&c, stream, iv(0, 200)), before);
}

#[test]
fn missing_recording_tables_refuse_open_instead_of_recreating_empty_history() {
    use ::redb::{Database, TableDefinition};
    for table in ["recording_meta", "recording_rows", "recording_ids", "recording_subjects", "recording_purge"] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("catalog.redb");
        let c = RedbCatalog::open(&path).unwrap();
        c.register_stream(info()).unwrap();
        drop(c);
        let db = Database::create(&path).unwrap();
        let tx = db.begin_write().unwrap();
        tx.delete_table(TableDefinition::<&[u8], &[u8]>::new(table)).unwrap();
        tx.commit().unwrap();
        drop(db);
        assert!(RedbCatalog::open(&path).is_err(), "missing {table} must not become empty history");
    }
}

#[test]
fn instant_video_target_has_original_membership_without_fabricated_duration() {
    both(|c| {
        let mut video = info();
        video.name = "cam.video".into();
        video.shape_name = "MJPEG".into();
        let stream = c.register_stream(video).unwrap();
        c.insert(vec![payload(70, stream, 100, 100, &[])]).unwrap();
        let first = c.head_seq().unwrap();
        let frames = vec![ChunkFrame::new(100, 0, 10)];
        let mut own = payload(71, stream, 100, 100, &[]);
        own.location = Some(location(71, StorageTier::Tier1Warm));
        c.commit_chunk(NewChunk { payload: own, frames }, &[id(70)]).unwrap();
        let proof = page(c, stream, iv(90, 110));
        assert!(proof.history_known && proof.unknown.is_empty());
        assert_eq!(proof.inputs[0].chunk_id, id(71));
        assert_eq!(proof.inputs[0].start_ns, 100);
        assert_eq!(proof.inputs[0].first_seq, first);
        assert!(proof.inputs[0].spans.is_empty(), "one timestamp is not an observed duration");
    });
}

#[test]
fn overbound_original_subject_attribution_never_becomes_empty_or_certified_on_reregistration() {
    both(|c| {
        let stream = c.register_stream(info()).unwrap();
        let subjects: Vec<_> = (100..229).map(id).collect();
        c.insert(vec![payload(80, stream, 10, 20, &subjects)]).unwrap();
        let proof = page(c, stream, iv(10, 20));
        assert!(!proof.history_known);
        assert_eq!(proof.unknown, [iv(10, 20)]);
        c.register_stream(info()).unwrap();
        assert!(!page(c, stream, iv(10, 20)).history_known);
    });
}

#[test]
fn original_chunk_identity_cannot_be_recreated_by_chunk_commit_after_safe_ageout() {
    both(|c| {
        let stream = c.register_stream(info()).unwrap();
        let (original, _, _) = chunk(c, stream, 1, 10, &[]);
        let before = page(c, stream, iv(10, 40)).inputs;
        erase(
            c,
            PurgeSelector::AgeOut { ids: vec![original], committed_by_seq: c.head_seq().unwrap(), merge_gap_ns: 0 },
            9100,
        );
        c.insert(vec![payload(200, stream, 100, 110, &[]), payload(201, stream, 120, 130, &[])]).unwrap();
        let mut own = payload(1001, stream, 100, 130, &[]);
        own.location = Some(location(20, StorageTier::Tier1Warm));
        let recreated = c.commit_chunk(
            NewChunk { payload: own, frames: vec![ChunkFrame::new(100, 0, 10), ChunkFrame::new(120, 10, 10)] },
            &[id(200), id(201)],
        );
        assert!(recreated.is_err(), "a new commit must not replace an immutable original witness UUID");
        assert_eq!(page(c, stream, iv(10, 40)).inputs, before);
    });
}

#[test]
fn original_chunk_identity_cannot_be_recreated_by_raw_ingest_after_safe_ageout() {
    both(|c| {
        let stream = c.register_stream(info()).unwrap();
        let (original, _, _) = chunk(c, stream, 1, 10, &[]);
        let before = page(c, stream, iv(10, 40)).inputs;
        erase(
            c,
            PurgeSelector::AgeOut { ids: vec![original], committed_by_seq: c.head_seq().unwrap(), merge_gap_ns: 0 },
            9101,
        );
        let head = c.head_seq().unwrap();
        assert!(
            c.insert(vec![payload(1001, stream, 100, 110, &[])]).is_err(),
            "raw ingestion must not reuse a historical original chunk UUID"
        );
        assert_eq!(c.head_seq().unwrap(), head, "a rejected resurrection is atomic");
        assert_eq!(page(c, stream, iv(10, 40)).inputs, before);
    });
}

#[test]
fn instant_audio_cannot_certify_neighboring_source_absence() {
    both(|c| {
        let stream = c.register_stream(info()).unwrap();
        c.insert(vec![payload(300, stream, 10, 10, &[])]).unwrap();
        let proof = page(c, stream, iv(20, 30));
        assert!(!proof.history_known, "an instant PCM header supplies no truthful source extent");
    });
}

#[test]
fn never_indexed_absorbed_input_is_rejected_instead_of_certifying_a_partial_chunk() {
    both(|c| {
        let stream = c.register_stream(info()).unwrap();
        c.insert(vec![payload(301, stream, 0, 2, &[])]).unwrap();
        let mut own = payload(1301, stream, 0, 10, &[]);
        own.location = Some(location(30, StorageTier::Tier1Warm));
        assert!(
            c.commit_chunk(
                NewChunk { payload: own, frames: vec![ChunkFrame::new(0, 0, 10), ChunkFrame::new(8, 10, 10)] },
                &[id(301), id(302)]
            )
            .is_err()
        );
        assert!(!page(c, stream, iv(0, 2)).unknown.is_empty());
    });
}

#[test]
fn overlapping_erased_original_runs_return_normalized_unknown_intervals() {
    both(|c| {
        let stream = c.register_stream(info()).unwrap();
        let (first, _, _) = chunk(c, stream, 40, 10, &[]);
        let (second, _, _) = chunk(c, stream, 41, 15, &[]);
        // Original runs interleave: [10,20), [30,40) and [15,25), [35,45).
        erase(c, PurgeSelector::Ids(vec![first, second]), 9150);
        let proof = page(c, stream, iv(0, 50));
        assert!(proof.inputs.is_empty());
        assert_eq!(proof.unknown, [iv(10, 25), iv(30, 45)]);
    });
}

#[test]
fn outside_window_certification_loss_advances_the_revision() {
    both(|c| {
        let stream = c.register_stream(info()).unwrap();
        chunk(c, stream, 50, 10, &[]);
        let proof = page(c, stream, iv(10, 40));
        c.insert(vec![payload(600, stream, 100, 100, &[])]).unwrap();
        assert!(page(c, stream, iv(10, 40)).revision > proof.revision);
    });
}

#[test]
fn late_binding_cannot_hide_original_runs_outside_the_payload_header() {
    both(|c| {
        let stream = c.register_stream(info()).unwrap();
        c.insert(vec![
            payload(610, stream, 0, 2, &[]),
            payload(611, stream, 8, 10, &[]),
            payload(612, stream, 7, 8, &[]),
        ])
        .unwrap();
        let first = c.recording_page(stream, iv(9, 10), None, None, 128, 1, MAX_RECORDING_BYTES).unwrap();
        assert!(first.next.is_some());
        let mut own = payload(1610, stream, 0, 9, &[]);
        own.location = Some(location(61, StorageTier::Tier1Warm));
        assert!(
            c.commit_chunk(
                NewChunk { payload: own, frames: vec![ChunkFrame::new(0, 0, 10), ChunkFrame::new(8, 10, 10)] },
                &[id(610), id(611)]
            )
            .is_err(),
            "short headers must not permit a late witness to move before the page cursor"
        );
        assert_eq!(page(c, stream, iv(9, 10)).unknown, [iv(9, 10)]);
    });
}

#[test]
fn native_range_hides_payload_on_first_step_before_sweeping_aged_history() {
    both(|c| {
        let stream = c.register_stream(info()).unwrap();
        let (historical, _, _) = chunk(c, stream, 70, 10, &[]);
        erase(
            c,
            PurgeSelector::AgeOut { ids: vec![historical], committed_by_seq: c.head_seq().unwrap(), merge_gap_ns: 0 },
            9170,
        );
        c.insert(vec![payload(710, stream, 100, 110, &[])]).unwrap();
        let purge = id(9171);
        c.start_purge(
            PurgeSelector::Range { span: iv(0, 120), streams: Some(vec![stream]) },
            "test",
            "synthetic",
            purge,
            0,
        )
        .unwrap();
        c.purge_step(purge, 1).unwrap();
        assert!(c.get(id(710)).unwrap().is_none(), "native payload hiding must precede historical ledger scanning");
        while matches!(c.purge_step(purge, 1).unwrap(), PurgeProgress::Running { .. }) {}
        let proof = page(c, stream, iv(10, 40));
        assert!(proof.inputs.is_empty());
        assert_eq!(proof.unknown, [iv(10, 20), iv(30, 40)]);
    });
}
