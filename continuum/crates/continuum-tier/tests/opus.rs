//! Small synthetic signals exercise the archive format, not throughput or live microphones.

use proptest::prelude::*;

use continuum_core::StreamId;
use continuum_tier::{ChunkFormat, ChunkWriter, hydrate, opus, pcm};

fn signal(rate: u32, channels: u16) -> (Vec<u8>, pcm::PcmLayout) {
    let frames = rate as usize / 100;
    let stride = pcm::STAMP_BYTES + frames * usize::from(channels) * 4;
    let mut batch = Vec::new();
    for block in 0..23u64 {
        // Deliberately nonconsecutive capture stamps: archive preserves them, not a reconstructed timeline.
        batch.extend_from_slice(&(1_000_000_000 + block * 10_000_003).to_le_bytes());
        batch.extend_from_slice(&(block * frames as u64 + 42).to_le_bytes());
        for frame in 0..frames {
            for channel in 0..channels {
                let t = (block as usize * frames + frame) as f32 / rate as f32;
                let frequency = if channel == 0 { 440.0 } else { 700.0 };
                let sample = (std::f32::consts::TAU * frequency * t).sin() * 0.5;
                batch.extend_from_slice(&sample.to_le_bytes());
            }
        }
    }
    (batch, pcm::PcmLayout { channels, stride: stride as u32 })
}

#[test]
fn native_rates_and_both_channels_keep_stamps_length_and_sample_alignment() {
    for rate in [8_000, 12_000, 16_000, 24_000, 48_000] {
        for channels in [1, 2] {
            let (raw, layout) = signal(rate, channels);
            let encoded = opus::encode_batch(&raw, layout, rate).unwrap();
            let decoded = opus::decode_batch(&encoded).unwrap();
            assert_eq!(decoded.len(), raw.len());
            let mut error = 0.0f64;
            let mut count = 0;
            for (before, after) in
                raw.chunks_exact(layout.stride as usize).zip(decoded.chunks_exact(layout.stride as usize))
            {
                assert_eq!(&before[..16], &after[..16]);
                for (a, b) in before[16..].chunks_exact(4).zip(after[16..].chunks_exact(4)) {
                    let a = f32::from_le_bytes(a.try_into().unwrap());
                    let b = f32::from_le_bytes(b.try_into().unwrap());
                    assert!(b.is_finite());
                    error += f64::from(a - b).powi(2);
                    count += 1;
                }
            }
            assert!(error / f64::from(count) < 0.01, "{rate}Hz {channels}channels MSE {}", error / f64::from(count));
        }
    }
}

#[test]
fn a_chunk_hydrates_independent_opus_batches_as_native_ring_records() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("audio.cnk");
    let (raw, layout) = signal(48_000, 1);
    let packed = opus::encode_batch(&raw, layout, 48_000).unwrap();
    let mut chunk = ChunkWriter::create(&file, StreamId(1), ChunkFormat::Opus, 100).unwrap();
    chunk.append(100, 230_000_100, 1, &packed).unwrap();
    chunk.append(300_000_100, 530_000_100, 2, &packed).unwrap();
    chunk.close().unwrap();
    let selected = hydrate(&file, 300_000_100, 310_000_100).unwrap();
    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0].record.seq, 2);
    assert_eq!(selected[0].bytes.len(), raw.len());
    assert_eq!(&selected[0].bytes[..16], &raw[..16]);
}

#[test]
fn unsupported_layouts_and_damaged_headers_are_explicit() {
    let (raw, layout) = signal(48_000, 1);
    assert!(!opus::supported(44_100, 1));
    assert!(!opus::supported(48_000, 6));
    assert!(opus::encode_batch(&raw, layout, 44_100).is_err());
    let encoded = opus::encode_batch(&raw, layout, 48_000).unwrap();
    for length in [0, 4, 23, 24, encoded.len() - 1] {
        assert!(opus::decode_batch(&encoded[..length]).is_err());
    }
    let mut trailing = encoded.clone();
    trailing.push(0);
    assert!(opus::decode_batch(&trailing).is_err(), "every byte of a batch is accounted for");
    let mut overlong = encoded[..40].to_vec(); // the header and the first stamp
    overlong.extend_from_slice(&[0xff; 11]);
    assert!(opus::decode_batch(&overlong).is_err());
    let mut damaged = encoded;
    damaged[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(opus::decode_batch(&damaged).is_err());
}

#[test]
fn a_pcm_chunk_becomes_opus_in_place_and_an_unknown_rate_or_cut_record_leaves_it_whole() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("1-100-1.cnk");
    let (raw, layout) = signal(12_000, 1);
    let format = pcm::PcmFormat { sample_rate_hz: 12_000, layout };
    let (known, _) = pcm::encode_batch_at_rate(&raw, format).unwrap();
    let mut chunk = ChunkWriter::create(&path, StreamId(1), ChunkFormat::PcmBatch, 100).unwrap();
    chunk.append_payload(continuum_core::Uuid([1; 16]), 100, 200, 1, &known).unwrap();
    chunk.append_payload(continuum_core::Uuid([2; 16]), 300, 400, 2, &known).unwrap();
    chunk.close().unwrap();
    let records = opus::pcm_chunk_to_opus(&path).unwrap();
    assert_eq!(
        records.iter().map(|r| (r.id, r.start_ns, r.end_ns, r.seq)).collect::<Vec<_>>(),
        vec![(continuum_core::Uuid([1; 16]), 100, 200, 1), (continuum_core::Uuid([2; 16]), 300, 400, 2)]
    );
    assert_eq!(continuum_tier::ChunkReader::open(&path).unwrap().format(), ChunkFormat::Opus);
    assert!(!dir.path().join("1-100-1.cnk.partial").exists());
    let read = hydrate(&path, 0, 1_000).unwrap();
    assert_eq!(read.len(), 2);
    assert_eq!(read[1].record.id, continuum_core::Uuid([2; 16]));
    assert_eq!(read[1].pcm, Some(format));
    assert_eq!(read[1].bytes.len(), raw.len());

    // A legacy batch has no rate, and a purge-cut record can't be carried: both leave the PCM chunk as it was.
    let (old, _) = pcm::encode_batch(&raw, layout).unwrap();
    for (name, cut) in [("2-100-1.cnk", false), ("3-100-1.cnk", true)] {
        let path = dir.path().join(name);
        let mut chunk = ChunkWriter::create(&path, StreamId(1), ChunkFormat::PcmBatch, 100).unwrap();
        chunk.append(100, 200, 1, if cut { &known } else { &old }).unwrap();
        chunk.append(300, 400, 2, &known).unwrap();
        chunk.close().unwrap();
        if cut {
            continuum_tier::chunk::rewrite_without(&path, &[0]).unwrap();
        }
        let before = std::fs::read(&path).unwrap();
        let error = opus::pcm_chunk_to_opus(&path).unwrap_err().to_string();
        assert!(error.contains(if cut { "cut" } else { "unknown source rate" }), "{error}");
        assert_eq!(std::fs::read(&path).unwrap(), before);
        assert!(!dir.path().join(format!("{name}.partial")).exists());
    }
}

fn stamps(batch: &[u8], stride: usize) -> Vec<&[u8]> {
    batch.chunks_exact(stride).map(|record| &record[..pcm::STAMP_BYTES]).collect()
}

fn header_u32(batch: &[u8], at: usize) -> usize {
    u32::from_le_bytes(batch[at..at + 4].try_into().unwrap()) as usize
}

#[test]
fn batches_in_the_original_format_still_decode_with_exact_stamps() {
    let legacy = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/cop1_16k_mono.opusbatch")).unwrap();
    assert_eq!(&legacy[..4], b"COP1", "a batch written before #1437");
    let (raw, layout) = signal(16_000, 1);
    let decoded = opus::decode_batch(&legacy).unwrap();
    assert_eq!(decoded.len(), raw.len());
    assert_eq!(stamps(&decoded, layout.stride as usize), stamps(&raw, layout.stride as usize));
    assert_eq!(opus::format_of(&legacy).unwrap(), pcm::PcmFormat { sample_rate_hz: 16_000, layout });
}

#[test]
fn a_steady_microphone_costs_two_bytes_a_record_and_at_most_two_a_packet() {
    let legacy = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/cop1_16k_mono.opusbatch")).unwrap();
    let (raw, layout) = signal(16_000, 1);
    let compact = opus::encode_batch(&raw, layout, 16_000).unwrap();
    let records = header_u32(&compact, 16);
    let packets = (records * 160 + header_u32(&compact, 20)).div_ceil(320);
    // The same audio packets; COP1 spent 16 bytes a record and 25 a packet (plus a 4-byte count) on metadata.
    let legacy_meta = 24 + records * 16 + 4 + packets * 25;
    let compact_meta_at_most = 24 + 16 + (records - 1) * 2 + packets * 2;
    assert!(
        compact.len() + legacy_meta <= legacy.len() + compact_meta_at_most,
        "{} vs {}",
        compact.len(),
        legacy.len()
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn any_stamps_round_trip_exactly(
        rate in prop::sample::select(vec![8_000u32, 16_000, 48_000]),
        stamps_in in prop::collection::vec((any::<u64>(), any::<u64>()), 1..12),
        steady in any::<bool>(),
    ) {
        let frames = rate as usize / 100;
        let stride = pcm::STAMP_BYTES + frames * 4;
        let mut batch = Vec::new();
        for (index, (time, sample)) in stamps_in.iter().enumerate() {
            // Either arbitrary stamps (wrapping, backwards, huge jumps) or near-cadence ones with small jitter.
            let (time, sample) = if steady {
                (time % 7 + index as u64 * 10_000_000, index as u64 * frames as u64 + sample % 3)
            } else {
                (*time, *sample)
            };
            batch.extend_from_slice(&time.to_le_bytes());
            batch.extend_from_slice(&sample.to_le_bytes());
            batch.extend(std::iter::repeat_n(0u8, frames * 4));
        }
        let layout = pcm::PcmLayout { channels: 1, stride: stride as u32 };
        let encoded = opus::encode_batch(&batch, layout, rate).unwrap();
        let decoded = opus::decode_batch(&encoded).unwrap();
        prop_assert_eq!(decoded.len(), batch.len());
        prop_assert_eq!(stamps(&decoded, stride), stamps(&batch, stride));
    }
}
