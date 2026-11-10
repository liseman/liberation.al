//! `recording_bound` against real PCM and Opus batch chunks (#1669, #1711).

use continuum_catalog::{
    Catalog, ChunkFrame, MAX_RECORDING_BYTES, NewChunk, NewPayload, RecordingPage, RedbCatalog, StreamInfo, StreamKind,
};
use continuum_core::{
    DomainLayer, HeaderBuf, HeaderSpec, Interval, PayloadFormat, StorageLocator, StorageTier, StreamId, Uuid,
};

use crate::ChunkFormat;
use crate::recording_bound::bounded_end;

const START: u64 = 1_000_000_000_000;
const ORIGINAL: Uuid = Uuid([2; 16]);
const CHUNK: Uuid = Uuid([20; 16]);

fn header(id: Uuid, stream: StreamId, start: u64, end: u64) -> HeaderBuf {
    let mut spec = HeaderSpec::raw(id, stream, start, end);
    spec.format = PayloadFormat::RawBytes;
    spec.shape_name = "PCM_F32LEx1".into();
    spec.shape_version = 1;
    HeaderBuf::build(&spec).unwrap()
}

/// 20 ring records of 160 mono frames at 16 kHz: 3200 frames, 200 ms, as raw ring bytes.
fn raw_batch() -> Vec<u8> {
    let mut raw = Vec::new();
    for block in 0..20u64 {
        raw.extend_from_slice(&(block * 10_000_000).to_le_bytes());
        raw.extend_from_slice(&(block * 160).to_le_bytes());
        for sample in 0..160u64 {
            raw.extend_from_slice(&(((block * 160 + sample) % 50) as f32 / 100.0).to_le_bytes());
        }
    }
    raw
}

const LAYOUT: crate::pcm::PcmLayout =
    crate::pcm::PcmLayout { channels: 1, stride: (crate::pcm::STAMP_BYTES + 160 * 4) as u32 };

/// The batch as Tier 1 writes it (CPB2) or as a closing microphone chunk re-encodes it (Opus).
fn batch(opus: bool) -> Vec<u8> {
    if opus {
        crate::opus::encode_batch(&raw_batch(), LAYOUT, 16_000).unwrap()
    } else {
        crate::pcm::encode_batch_at_rate(&raw_batch(), crate::pcm::PcmFormat { sample_rate_hz: 16_000, layout: LAYOUT })
            .unwrap()
            .0
    }
}

fn page(c: &dyn Catalog, stream: StreamId, start: u64, end: u64) -> RecordingPage {
    c.recording_page(stream, Interval::new(start, end).unwrap(), None, None, 128, 4096, MAX_RECORDING_BYTES).unwrap()
}

/// A microphone whose one original is an instant header, flushed into a real chunk with its 200 ms of samples. The
/// chunk's record names `record_id` as the payload it absorbed (the original, unless a test says otherwise).
fn fixture(dir: &std::path::Path) -> (RedbCatalog, StreamId) {
    fixture_with(dir, ORIGINAL, false)
}

fn fixture_with(dir: &std::path::Path, record_id: Uuid, opus: bool) -> (RedbCatalog, StreamId) {
    let c = RedbCatalog::open(dir.join("catalog.redb")).unwrap();
    let stream = c
        .register_stream(StreamInfo {
            name: "mic.audio".into(),
            kind: StreamKind::Sensor,
            layer: DomainLayer::Layer1Raw,
            format: PayloadFormat::RawBytes,
            shape_name: "PCM_F32LEx1".into(),
            shape_version: 1,
            rate_hz: Some(100.0),
            anchor: None,
        })
        .unwrap();
    c.insert(vec![NewPayload { header: header(ORIGINAL, stream, START, START), location: None, coverage: None }])
        .unwrap();
    let path = dir.join("mic.cnk");
    let format = if opus { ChunkFormat::Opus } else { ChunkFormat::PcmBatch };
    let mut writer = crate::ChunkWriter::create(&path, stream, format, START).unwrap();
    let record = writer.append_payload(record_id, START, START + 200_000_000, 0, &batch(opus)).unwrap();
    writer.close().unwrap();
    c.commit_chunk(
        NewChunk {
            payload: NewPayload {
                header: header(CHUNK, stream, START, START + 200_000_000),
                location: Some(StorageLocator::Chunk {
                    tier: if opus { StorageTier::Tier2Cold } else { StorageTier::Tier1Warm },
                    uri: path.to_string_lossy().into_owned(),
                    offset: 0,
                    length: std::fs::metadata(&path).unwrap().len(),
                }),
                coverage: None,
            },
            frames: vec![ChunkFrame::new(START, record.offset, u64::from(record.len))],
        },
        &[ORIGINAL],
    )
    .unwrap();
    (c, stream)
}

#[test]
fn an_instant_originals_own_samples_bound_its_end_and_free_its_stream() {
    let dir = tempfile::tempdir().unwrap();
    let (c, stream) = fixture(dir.path());
    assert!(!page(&c, stream, START - 10, START - 5).history_known);
    let originals = c.instant_originals().unwrap();
    let [original] = &originals[..] else { panic!("one instant original") };
    assert_eq!((original.original, original.frame), (ORIGINAL, Some((CHUNK, 0))));
    // 3200 frames at 16 kHz is 200 ms, widened by 1000 ppm (0.2 ms) and one 10 ms block.
    let end = bounded_end(&c, original, Some(100.0)).unwrap();
    assert_eq!(end, START + 210_200_000);
    assert!(c.bound_instant_original(ORIGINAL, end).unwrap());
    assert!(page(&c, stream, START - 10, START - 5).history_known);
}

#[test]
fn a_stream_rate_that_disagrees_with_the_batch_leaves_the_original_held() {
    let dir = tempfile::tempdir().unwrap();
    let (c, _) = fixture(dir.path());
    let originals = c.instant_originals().unwrap();
    let [original] = &originals[..] else { panic!("one instant original") };
    // 160-frame blocks at 16 kHz are 100 a second; a stream declaring 50 disagrees. The end is still reported.
    let why = bounded_end(&c, original, Some(50.0)).unwrap_err();
    assert!(why.contains("declares Some(50.0)") && why.contains(&format!("{}", START + 210_200_000)), "{why}");
}

/// The one instant original once the catalog has pruned its frame row, as it does an hour after the chunk ends.
fn pruned(c: &RedbCatalog) -> continuum_catalog::InstantOriginal {
    assert_eq!(c.prune_frames(u64::MAX, None, 64).unwrap().pruned, 1);
    let mut originals = c.instant_originals().unwrap();
    let original = originals.pop().expect("one instant original");
    assert_eq!((original.original, original.frame), (ORIGINAL, None));
    original
}

#[test]
fn an_original_whose_frame_row_was_pruned_is_found_by_its_start() {
    let dir = tempfile::tempdir().unwrap();
    let (c, _) = fixture(dir.path());
    let original = pruned(&c);
    assert_eq!(bounded_end(&c, &original, Some(100.0)).unwrap(), START + 210_200_000);
}

#[test]
fn an_opus_copy_bounds_the_same_end_as_the_pcm_it_was_encoded_from() {
    let dir = tempfile::tempdir().unwrap();
    let (c, _) = fixture_with(dir.path(), ORIGINAL, true);
    let original = pruned(&c);
    assert_eq!(bounded_end(&c, &original, Some(100.0)).unwrap(), START + 210_200_000);
}

#[test]
fn a_frame_at_its_start_holding_another_record_leaves_the_original_held() {
    let dir = tempfile::tempdir().unwrap();
    let (c, _) = fixture_with(dir.path(), Uuid([9; 16]), false);
    let original = pruned(&c);
    let why = bounded_end(&c, &original, Some(100.0)).unwrap_err();
    assert!(why.contains("own record"), "{why}");
}
