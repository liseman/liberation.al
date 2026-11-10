//! Microphone ring batches in Tier 1 ([`crate::ChunkFormat::PcmBatch`]).
//!
//! Capture opens every microphone as f32 (PipeWire and CoreAudio both convert), and its ring records are a 16-byte
//! stamp then one block of interleaved f32 samples. A device's native format is integer PCM (the Wave 3 gives
//! S24LE, the XVF3800 S16LE), and when PipeWire only converts the format (no stream volume other than 1.0, no
//! resampling) it divides by a power of two, so the f32 samples are exactly integers at the native depth. A batch whose every sample round-trips exactly through 16 or 24 bit integers is
//! stored as FLAC at the smallest depth that holds it; anything else (software gain, resampling, a float source) is
//! stored as LZ4 of the batch, and the worker logs the fallback. On the Wave 3 (mono, 48 kHz, 24 bit) FLAC is about
//! 231 MB an hour against 352 MB as LZ4 f32 (docs/CONTINUUM.md, the audio row).
//!
//! A record is self-describing, so one chunk can hold both kinds. [`decode_batch`] gives back the batch's bytes
//! exactly as they were in the ring, stamps included, so a reader never sees the difference.
//!
//! Layout, little-endian: the magic `CPB1`, a codec byte ([`PcmCodec`]), a reserved byte, channels (u16), the ring
//! stride (u32), the record count (u32), then for FLAC each record's 16-byte stamp in order and a FLAC stream of
//! every block's samples, interleaved; for LZ4, [`crate::codecs::lz4_compress`] of the whole batch. The FLAC stream's
//! New CPB2 records add the true source sample rate (u32) after the CPB1 header. Both versions decode losslessly.
//! The FLAC stream's own header says 48 kHz whatever the device's rate, since the ring doesn't carry it and nothing reads it: the
//! stamps place every block.

use crate::TierError;
use crate::codecs::{flac_encode, lz4_compress, lz4_decompress};

/// The magic at the front of every PCM batch record.
pub const MAGIC: [u8; 4] = *b"CPB1";

/// Bytes of stamp in front of each ring record (capture's `PCM_RECORD_HEADER`).
pub const STAMP_BYTES: usize = 16;

/// Bytes of a PCM batch record's header.
pub const HEADER_BYTES: usize = 16;

/// The most a decoded batch may be: far above any batch (capture's 1 s of 8 channels of f32 at 192 kHz is 6 MB), so
/// a damaged or hostile header can't make a reader allocate more.
pub const MAX_BATCH_BYTES: usize = 64 << 20;

/// FLAC's channel limit.
const FLAC_MAX_CHANNELS: u16 = 8;

/// The sample rate written into the FLAC stream header. The ring doesn't carry the rate and nothing reads it back:
/// the stamps place every block.
const FLAC_HEADER_RATE: usize = 48_000;

/// How a PCM batch record stores its samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PcmCodec {
    /// FLAC of the samples as 16 bit integers.
    Flac16 = 1,
    /// FLAC of the samples as 24 bit integers.
    Flac24 = 2,
    /// LZ4 of the whole batch: the fallback when the samples aren't exact integers.
    Lz4 = 3,
}

impl PcmCodec {
    fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            1 => Some(Self::Flac16),
            2 => Some(Self::Flac24),
            3 => Some(Self::Lz4),
            _ => None,
        }
    }

    /// The integer depth, for FLAC.
    #[must_use]
    pub fn bits(self) -> Option<u32> {
        match self {
            Self::Flac16 => Some(16),
            Self::Flac24 => Some(24),
            Self::Lz4 => None,
        }
    }
}

/// What a microphone stream's ring records are: enough to split a batch into stamps and samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PcmLayout {
    /// Interleaved channels.
    pub channels: u16,
    /// Bytes per ring record: the stamp, then one block.
    pub stride: u32,
}

impl PcmLayout {
    /// The layout of a stream with shape `shape_name` (capture's `PCM_F32LEx<channels>`) in a ring of `stride`
    /// bytes, or `None` for any other shape, or a stride that doesn't hold a stamp and whole frames.
    #[must_use]
    pub fn of(shape_name: &str, stride: usize) -> Option<Self> {
        let channels: u16 = shape_name.strip_prefix("PCM_F32LEx")?.parse().ok()?;
        let frame = 4 * usize::from(channels);
        let block = stride.checked_sub(STAMP_BYTES)?;
        if channels == 0 || block == 0 || !block.is_multiple_of(frame) {
            return None;
        }
        Some(Self { channels, stride: u32::try_from(stride).ok()? })
    }
}

/// A batch's exact source format. Unlike FLAC's historical placeholder, this rate comes from capture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PcmFormat {
    /// Source frames per second.
    pub sample_rate_hz: u32,
    /// Ring channels and record stride.
    pub layout: PcmLayout,
}

/// Source format stored by new CPB2 records, or None for a legacy CPB1 record with no rate.
///
/// # Errors
/// A malformed PCM header or an invalid source rate.
pub fn format_of(record: &[u8]) -> Result<Option<PcmFormat>, TierError> {
    codec_of(record)?;
    if record[..4] == MAGIC {
        return Ok(None);
    }
    let sample_rate_hz = u32::from_le_bytes(record[16..20].try_into().expect("four bytes"));
    if sample_rate_hz == 0 {
        return Err(TierError::Codec("pcm batch: zero source rate".into()));
    }
    Ok(Some(PcmFormat {
        sample_rate_hz,
        layout: PcmLayout {
            channels: u16::from_le_bytes([record[6], record[7]]),
            stride: u32::from_le_bytes(record[8..12].try_into().expect("four bytes")),
        },
    }))
}

/// How many sample frames a CPB2 batch record holds, with its format, read from its header alone: record count times
/// the frames in one ring block. None for a legacy CPB1 record, which has no rate.
///
/// # Errors
/// A malformed header, or a stride that doesn't hold a stamp and whole frames.
pub fn sample_frames(record: &[u8]) -> Result<Option<(u64, PcmFormat)>, TierError> {
    let Some(format) = format_of(record)? else { return Ok(None) };
    let frame = 4 * u64::from(format.layout.channels);
    let block = u64::from(format.layout.stride)
        .checked_sub(STAMP_BYTES as u64)
        .filter(|block| frame > 0 && *block > 0 && block % frame == 0)
        .ok_or_else(|| TierError::Codec("pcm batch: stride holds no whole frames".into()))?;
    let records = u64::from(u32::from_le_bytes(record[12..16].try_into().expect("four bytes")));
    Ok(Some((records * (block / frame), format)))
}

/// Pack a lossless Tier 1 batch with its explicit source rate. Legacy callers can use [`encode_batch`].
///
/// # Errors
/// A zero rate or an encoding error.
pub fn encode_batch_at_rate(batch: &[u8], format: PcmFormat) -> Result<(Vec<u8>, PcmCodec), TierError> {
    if format.sample_rate_hz == 0 {
        return Err(TierError::Codec("pcm batch: zero source rate".into()));
    }
    let (mut packed, codec) = encode_batch(batch, format.layout)?;
    packed[..4].copy_from_slice(b"CPB2");
    packed.splice(HEADER_BYTES..HEADER_BYTES, format.sample_rate_hz.to_le_bytes());
    Ok((packed, codec))
}

/// The smallest integer depth (16 or 24) at which every sample round-trips exactly, bit for bit, or `None`.
#[must_use]
pub fn exact_depth(samples: impl Iterator<Item = f32> + Clone) -> Option<u32> {
    [16, 24].into_iter().find(|&bits| samples.clone().all(|x| to_int(x, bits).is_some()))
}

/// `x` as a `bits` integer, if it converts back to exactly the same f32 bits.
fn to_int(x: f32, bits: u32) -> Option<i32> {
    let scale = scale(bits);
    let s = x * scale;
    if !s.is_finite() || s < -scale || s > scale - 1.0 {
        return None;
    }
    #[expect(clippy::cast_possible_truncation, reason = "s is within the bits range, checked above")]
    let i = s as i32;
    (to_f32(i, bits).to_bits() == x.to_bits()).then_some(i)
}

fn to_f32(i: i32, bits: u32) -> f32 {
    #[expect(clippy::cast_precision_loss, reason = "at most 24 significant bits, exact in an f32")]
    let f = i as f32;
    f / scale(bits)
}

fn scale(bits: u32) -> f32 {
    #[expect(clippy::cast_precision_loss, reason = "a power of two, exact")]
    let s = (1u32 << (bits - 1)) as f32;
    s
}

/// The f32 samples of a batch's records, in order, skipping each stamp.
fn samples(batch: &[u8], stride: usize) -> impl Iterator<Item = f32> + Clone + '_ {
    batch
        .chunks_exact(stride)
        .flat_map(|r| r[STAMP_BYTES..].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().expect("4 bytes"))))
}

fn header(codec: PcmCodec, layout: PcmLayout, count: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_BYTES);
    out.extend_from_slice(&MAGIC);
    out.push(codec as u8);
    out.push(0);
    out.extend_from_slice(&layout.channels.to_le_bytes());
    out.extend_from_slice(&layout.stride.to_le_bytes());
    out.extend_from_slice(&count.to_le_bytes());
    out
}

/// Packs a batch of ring records (`layout.stride` bytes each) into one Tier 1 record, as FLAC when every sample is
/// an exact integer at 16 or 24 bits, otherwise as LZ4. Returns the record and the codec it used.
///
/// # Errors
///
/// [`TierError::Codec`] when the FLAC encoder fails.
pub fn encode_batch(batch: &[u8], layout: PcmLayout) -> Result<(Vec<u8>, PcmCodec), TierError> {
    let stride = layout.stride as usize;
    let count = u32::try_from(batch.len() / stride).unwrap_or(u32::MAX);
    let whole = !batch.is_empty() && batch.len().is_multiple_of(stride);
    let depth = if whole && layout.channels <= FLAC_MAX_CHANNELS { exact_depth(samples(batch, stride)) } else { None };
    let Some(bits) = depth else { return Ok((encode_lz4(batch, layout, count), PcmCodec::Lz4)) };
    let codec = if bits == 16 { PcmCodec::Flac16 } else { PcmCodec::Flac24 };
    let ints: Vec<i32> = samples(batch, stride).map(|x| to_int(x, bits).expect("checked exact")).collect();
    let flac = flac_encode(&ints, usize::from(layout.channels), bits as usize, FLAC_HEADER_RATE)?;
    let mut out = header(codec, layout, count);
    out.reserve(count as usize * STAMP_BYTES + flac.len());
    for r in batch.chunks_exact(stride) {
        out.extend_from_slice(&r[..STAMP_BYTES]);
    }
    out.extend_from_slice(&flac);
    Ok((out, codec))
}

/// A batch as an LZ4 record: the fallback, and the form for a stream whose layout isn't known (`layout` and `count`
/// are then 0, which reading an LZ4 record never consults).
#[must_use]
pub fn encode_lz4(batch: &[u8], layout: PcmLayout, count: u32) -> Vec<u8> {
    let mut out = header(PcmCodec::Lz4, layout, count);
    out.extend_from_slice(&lz4_compress(batch));
    out
}

/// The codec of a PCM batch record, from its header.
///
/// # Errors
///
/// [`TierError::Codec`] for bytes that aren't a PCM batch record.
pub fn codec_of(record: &[u8]) -> Result<PcmCodec, TierError> {
    if record.len() < HEADER_BYTES
        || (record[..4] != MAGIC && record[..4] != *b"CPB2")
        || (record[..4] == *b"CPB2" && record.len() < HEADER_BYTES + 4)
    {
        return Err(TierError::Codec("pcm batch: no header".to_owned()));
    }
    PcmCodec::from_tag(record[4]).ok_or_else(|| TierError::Codec(format!("pcm batch: unknown codec {}", record[4])))
}

/// The inverse of [`encode_batch`]: the batch's ring records, byte for byte.
///
/// # Errors
///
/// [`TierError::Codec`] for a record that isn't a PCM batch, or whose FLAC or LZ4 doesn't decode to the records
/// its header describes.
pub fn decode_batch(record: &[u8]) -> Result<Vec<u8>, TierError> {
    let codec = codec_of(record)?;
    let u32_at = |i: usize| u32::from_le_bytes(record[i..i + 4].try_into().expect("4 bytes"));
    let channels = usize::from(u16::from_le_bytes([record[6], record[7]]));
    let stride = u32_at(8) as usize;
    let count = u32_at(12) as usize;
    let format = format_of(record)?;
    let body = &record[HEADER_BYTES + if format.is_some() { 4 } else { 0 }..];
    let Some(bits) = codec.bits() else { return lz4_decompress(body) };
    let bad = |why: String| TierError::Codec(format!("pcm batch: {why}"));
    // The header is checked before anything is allocated from it.
    let block = stride.saturating_sub(STAMP_BYTES);
    let frame = 4 * channels;
    if !(1..=usize::from(FLAC_MAX_CHANNELS)).contains(&channels) || block == 0 || !block.is_multiple_of(frame) {
        return Err(bad(format!("{channels} channels in records of {stride} bytes")));
    }
    if count == 0 || count.saturating_mul(stride) > MAX_BATCH_BYTES {
        return Err(bad(format!("{count} records of {stride} bytes")));
    }
    let block_samples = block / 4;
    let stamps_len = count.checked_mul(STAMP_BYTES).filter(|&n| n <= body.len()).ok_or_else(|| bad("short".into()))?;
    let (stamps, flac) = body.split_at(stamps_len);
    let mut reader = claxon::FlacReader::new(flac).map_err(|e| bad(format!("flac: {e}")))?;
    let info = reader.streaminfo();
    if info.channels as usize != channels || info.bits_per_sample != bits {
        return Err(bad(format!("flac has {} channels at {} bits", info.channels, info.bits_per_sample)));
    }
    let want = count * block_samples;
    let mut ints = Vec::with_capacity(want);
    for s in reader.samples() {
        if ints.len() == want {
            return Err(bad("more samples than records".into()));
        }
        ints.push(s.map_err(|e| bad(format!("flac: {e}")))?);
    }
    if ints.len() != want {
        return Err(bad(format!("{} samples for {count} records of {block_samples}", ints.len())));
    }
    let mut out = Vec::with_capacity(count * stride);
    for (stamp, block) in stamps.chunks_exact(STAMP_BYTES).zip(ints.chunks_exact(block_samples)) {
        out.extend_from_slice(stamp);
        for &i in block {
            out.extend_from_slice(&to_f32(i, bits).to_le_bytes());
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `count` records of `frames` frames of `channels` channels, each sample `f(record, index)`.
    fn batch(count: u64, frames: usize, channels: u16, f: impl Fn(u64, usize) -> f32) -> (Vec<u8>, PcmLayout) {
        let stride = STAMP_BYTES + frames * usize::from(channels) * 4;
        let mut out = Vec::new();
        for r in 0..count {
            out.extend_from_slice(&(1_000 + r * 10_000_000).to_le_bytes());
            out.extend_from_slice(&(r * frames as u64).to_le_bytes());
            for i in 0..frames * usize::from(channels) {
                out.extend_from_slice(&f(r, i).to_le_bytes());
            }
        }
        (out, PcmLayout::of(&format!("PCM_F32LEx{channels}"), stride).expect("layout"))
    }

    fn sine(r: u64, i: usize, bits: u32) -> f32 {
        #[expect(clippy::cast_precision_loss, reason = "test signal")]
        let t = (r * 480 + i as u64) as f32;
        #[expect(clippy::cast_possible_truncation, reason = "test signal")]
        let v = ((t * 0.05).sin() * 0.5 * scale(bits)) as i32;
        to_f32(v, bits)
    }

    #[test]
    fn a_batch_header_says_how_many_frames_it_holds_whatever_its_codec() {
        // 6 records of 480 frames, stereo: 2880 frames, in a FLAC batch and in an LZ4 one.
        for (raw, layout) in
            [batch(6, 480, 2, |r, i| sine(r, i, 16)), batch(6, 480, 2, |r, i| 0.1 + (r * 7 + i as u64) as f32 * 1e-7)]
        {
            let format = PcmFormat { sample_rate_hz: 44_100, layout };
            let (record, _) = encode_batch_at_rate(&raw, format).unwrap();
            assert_eq!(sample_frames(&record).unwrap(), Some((2_880, format)));
        }
        let (raw, layout) = batch(2, 120, 1, |r, i| sine(r, i, 16));
        let (legacy, _) = encode_batch(&raw, layout).unwrap();
        assert_eq!(sample_frames(&legacy).unwrap(), None, "CPB1 has no rate, so no bound");
        assert!(sample_frames(b"CPB2").is_err());
    }

    #[test]
    fn new_batches_keep_source_rate_and_old_batches_remain_unknown() {
        for gain in [1.0, 0.7] {
            let (raw, layout) = batch(3, 120, 1, |r, i| sine(r, i, 24) * gain);
            let format = PcmFormat { sample_rate_hz: 12_000, layout };
            let (old, _) = encode_batch(&raw, layout).unwrap();
            assert_eq!(format_of(&old).unwrap(), None);
            assert_eq!(decode_batch(&old).unwrap(), raw);
            let (new, _) = encode_batch_at_rate(&raw, format).unwrap();
            assert_eq!(format_of(&new).unwrap(), Some(format));
            assert_eq!(decode_batch(&new).unwrap(), raw);
            assert!(decode_batch(&new[..19]).is_err());
            let mut zero_rate = new;
            zero_rate[16..20].fill(0);
            assert!(decode_batch(&zero_rate).is_err());
        }
    }

    #[test]
    fn s16_and_s24_samples_go_to_flac_at_their_depth_and_come_back_exactly() {
        for (bits, codec) in [(16, PcmCodec::Flac16), (24, PcmCodec::Flac24)] {
            let (raw, layout) = batch(100, 480, 1, |r, i| sine(r, i, bits));
            let (packed, used) = encode_batch(&raw, layout).expect("encode");
            assert_eq!(used, codec);
            assert!(packed.len() < raw.len() / 2, "{bits} bits: {} of {}", packed.len(), raw.len());
            assert_eq!(decode_batch(&packed).expect("decode"), raw);
        }
    }

    #[test]
    fn six_channels_of_s16_round_trip() {
        let (raw, layout) = batch(50, 320, 6, |r, i| sine(r, i / 6 + i % 6, 16));
        let (packed, used) = encode_batch(&raw, layout).expect("encode");
        assert_eq!(used, PcmCodec::Flac16);
        assert_eq!(decode_batch(&packed).expect("decode"), raw);
    }

    #[test]
    fn samples_that_arent_exact_integers_fall_back_to_lz4_and_round_trip() {
        // Software gain of 0.7 leaves samples between the 24 bit steps.
        let (raw, layout) = batch(20, 480, 1, |r, i| sine(r, i, 24) * 0.7 + 1e-9);
        let (packed, used) = encode_batch(&raw, layout).expect("encode");
        assert_eq!(used, PcmCodec::Lz4);
        assert_eq!(decode_batch(&packed).expect("decode"), raw);
    }

    #[test]
    fn negative_zero_and_full_scale_are_judged_by_their_bits() {
        assert_eq!(exact_depth([0.0f32, -1.0, 0.5].into_iter()), Some(16));
        assert_eq!(exact_depth([to_f32(1, 24)].into_iter()), Some(24));
        // -0.0 would come back as +0.0, 1.0 is past the top of the range, NaN is never an integer.
        assert_eq!(exact_depth([-0.0f32].into_iter()), None);
        assert_eq!(exact_depth([1.0f32].into_iter()), None);
        assert_eq!(exact_depth([f32::NAN].into_iter()), None);
    }

    #[test]
    fn a_layout_comes_only_from_an_f32_shape_and_a_stride_of_whole_frames() {
        assert_eq!(PcmLayout::of("PCM_F32LEx2", 16 + 480 * 8), Some(PcmLayout { channels: 2, stride: 3_856 }));
        assert_eq!(PcmLayout::of("PCM_S16LEx2", 16 + 480 * 4), None);
        assert_eq!(PcmLayout::of("PCM_F32LEx2", 16 + 6), None);
        assert_eq!(PcmLayout::of("PCM_F32LEx0", 16 + 8), None);
        assert_eq!(PcmLayout::of("PCM_F32LEx1", 16), None);
    }

    #[test]
    fn a_header_asking_for_too_much_or_a_ragged_layout_is_refused_before_allocating() {
        let (raw, layout) = batch(4, 480, 1, |r, i| sine(r, i, 16));
        let (packed, _) = encode_batch(&raw, layout).expect("encode");
        let with = |at: usize, bytes: &[u8]| {
            let mut r = packed.clone();
            r[at..at + bytes.len()].copy_from_slice(bytes);
            r
        };
        for damaged in [
            with(8, &u32::MAX.to_le_bytes()),    // a 4 GB stride
            with(8, &16u32.to_le_bytes()),       // records with no samples
            with(8, &(16u32 + 6).to_le_bytes()), // a block of partial frames
            with(6, &9u16.to_le_bytes()),        // more channels than FLAC holds
            with(6, &0u16.to_le_bytes()),        // no channels
            with(12, &0u32.to_le_bytes()),       // no records
        ] {
            assert!(decode_batch(&damaged).is_err());
        }
    }

    #[test]
    fn an_lz4_record_of_unknown_layout_round_trips() {
        let raw = vec![7u8; 1_000];
        let zero = PcmLayout { channels: 0, stride: 0 };
        let packed = encode_lz4(&raw, zero, 0);
        assert_eq!(codec_of(&packed).expect("codec"), PcmCodec::Lz4);
        assert_eq!(decode_batch(&packed).expect("decode"), raw);
    }

    #[test]
    fn a_damaged_record_is_refused() {
        let (raw, layout) = batch(4, 480, 1, |r, i| sine(r, i, 16));
        let (packed, _) = encode_batch(&raw, layout).expect("encode");
        assert!(decode_batch(b"nope").is_err());
        assert!(decode_batch(&packed[..packed.len() / 2]).is_err());
        let mut wrong = packed.clone();
        wrong[12] = 9; // the count no longer matches the samples
        assert!(decode_batch(&wrong).is_err());
    }
}
