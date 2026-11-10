//! Self-contained Opus microphone batches using libopus. A microphone's PCM chunk becomes Opus as it closes
//! ([`pcm_chunk_to_opus`], called by [`crate::worker`]), before it is committed.
//!
//! Each batch starts a fresh encoder/decoder, so a range read or purge never needs earlier audio. The header
//! stores a magic, source rate (u32), channels (u16), reserved zero (u16), ring stride (u32), record count (u32),
//! and encoder lookahead in samples per channel (u32), all little-endian. Decode removes lookahead and final
//! padding, restoring the exact number and layout of ring records. Stamps are lossless; samples are lossy.
//!
//! `COP2` follows the header with the first record's 16-byte stamp, then two zigzag LEB128
//! varints per further record: its capture-time step less the previous step (the nominal block duration for the
//! second record), and its sample-index step less the block's frame count. A steady microphone costs two bytes a
//! record instead of sixteen, and any stamps at all still round-trip exactly (the arithmetic wraps). Then come one
//! LEB128 length per 20 ms packet (the count follows from the header) and the packets themselves. Packet times are
//! not stored: decoding never needed them, and they follow from the 20 ms grid.
//!
//! `COP1` (read only) instead stores every 16-byte stamp and a [`crate::gop`] table, 25 bytes per packet. At
//! 24 kbit/s that metadata was as large as the audio. Chunks written that way keep decoding unchanged.
//!
//! Only Opus's native sample rates and mono/stereo are admitted. Other layouts stay lossless PCM chunks until
//! a separately defined resampling/multistream path exists. Source rate is required, never inferred from FLAC's
//! placeholder rate or rounded timestamps. libopus API: <https://opus-codec.org/docs/opus_api-1.6.pdf>.

use std::borrow::Cow;

use ::opus::{Application, Bitrate, Channels, Decoder, Encoder};

use crate::TierError;
use crate::gop;
use crate::pcm::{MAX_BATCH_BYTES, PcmLayout, STAMP_BYTES};

const MAGIC: &[u8; 4] = b"COP2";
const MAGIC_V1: &[u8; 4] = b"COP1";
/// The most bytes a u64 LEB128 varint takes.
const MAX_VARINT: usize = 10;
const HEADER: usize = 24;
const PACKETS_PER_SECOND: usize = 50;
const MAX_PACKET_BYTES: usize = 2_552;

fn invalid(why: impl std::fmt::Display) -> TierError {
    TierError::Codec(format!("opus batch: {why}"))
}

/// Whether the first Opus archive path can preserve this rate and channel layout without resampling/downmixing.
#[must_use]
pub fn supported(rate: u32, channels: u16) -> bool {
    [8_000, 12_000, 16_000, 24_000, 48_000].contains(&rate) && (1..=2).contains(&channels)
}

fn validate(layout: PcmLayout, count: usize, rate: u32) -> Result<usize, TierError> {
    let stride = layout.stride as usize;
    let bytes = stride.checked_mul(count).ok_or_else(|| invalid("batch size overflow"))?;
    let frame = usize::from(layout.channels) * 4;
    if !supported(rate, layout.channels) || stride <= STAMP_BYTES || count == 0 || bytes > MAX_BATCH_BYTES {
        return Err(invalid("unsupported rate/layout or excessive batch"));
    }
    if !(stride - STAMP_BYTES).is_multiple_of(frame) {
        return Err(invalid("partial sample frame"));
    }
    Ok((stride - STAMP_BYTES) / frame * count)
}

/// Compress raw f32 ring records with their original stamps.
///
/// # Errors
/// Unsupported/invalid layout, nonfinite samples, or a codec error.
pub fn encode_batch(batch: &[u8], layout: PcmLayout, rate: u32) -> Result<Vec<u8>, TierError> {
    encode_at_bitrate(batch, layout, rate, 64_000 * u32::from(layout.channels))
}

/// Encode a batch at a caller-selected total bitrate. Layout and stamps follow [`encode_batch`].
///
/// # Errors
/// As [`encode_batch`], or a bitrate outside libopus's 500 to 512000 bits/s range.
fn encode_at_bitrate(batch: &[u8], layout: PcmLayout, rate: u32, bit_rate: u32) -> Result<Vec<u8>, TierError> {
    if !(500..=512_000).contains(&bit_rate) {
        return Err(invalid("bitrate must be 500 to 512000 bits/s"));
    }
    let stride = layout.stride as usize;
    if stride == 0 || !batch.len().is_multiple_of(stride) {
        return Err(invalid("partial ring record"));
    }
    let count = batch.len() / stride;
    let frames = validate(layout, count, rate)?;
    let channels = usize::from(layout.channels);
    let channel_mode = if channels == 1 { Channels::Mono } else { Channels::Stereo };
    let mut encoder = Encoder::new(rate, channel_mode, Application::Audio).map_err(invalid)?;
    encoder.set_bitrate(Bitrate::Bits(i32::try_from(bit_rate).map_err(invalid)?)).map_err(invalid)?;
    let skip = usize::try_from(encoder.get_lookahead().map_err(invalid)?).map_err(invalid)?;
    let samples: Vec<f32> = batch
        .chunks_exact(stride)
        .flat_map(|record| {
            record[STAMP_BYTES..].chunks_exact(4).map(|s| f32::from_le_bytes(s.try_into().expect("four bytes")))
        })
        .collect();
    if samples.iter().any(|s| !s.is_finite()) {
        return Err(invalid("nonfinite sample"));
    }
    let frame_frames = rate as usize / PACKETS_PER_SECOND;
    let packets = (frames + skip).div_ceil(frame_frames);
    let mut encoded = Vec::with_capacity(packets);
    for index in 0..packets {
        let at = index * frame_frames * channels;
        let mut padded = vec![0.0; frame_frames * channels];
        if at < samples.len() {
            let n = padded.len().min(samples.len() - at);
            padded[..n].copy_from_slice(&samples[at..at + n]);
        }
        encoded.push(encoder.encode_vec_float(&padded, MAX_PACKET_BYTES).map_err(invalid)?);
    }
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&layout.channels.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&layout.stride.to_le_bytes());
    out.extend_from_slice(&u32::try_from(count).map_err(invalid)?.to_le_bytes());
    out.extend_from_slice(&u32::try_from(skip).map_err(invalid)?.to_le_bytes());
    let mut cadence = Cadence::of(layout, rate);
    let mut previous: Option<Stamp> = None;
    for record in batch.chunks_exact(stride) {
        let stamp = Stamp::read(record);
        match previous {
            None => out.extend_from_slice(&record[..STAMP_BYTES]),
            Some(p) => {
                let (time, sample) = cadence.residuals(p, stamp);
                put_varint(&mut out, zigzag(time));
                put_varint(&mut out, zigzag(sample));
            }
        }
        cadence.step = previous.map_or(cadence.step, |p| stamp.time.wrapping_sub(p.time));
        previous = Some(stamp);
    }
    for packet in &encoded {
        put_varint(&mut out, packet.len() as u64);
    }
    for packet in &encoded {
        out.extend_from_slice(packet);
    }
    Ok(out)
}

/// A ring record's stamp: capture time (host monotonic ns) and the index of its first sample.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Stamp {
    time: u64,
    sample: u64,
}

impl Stamp {
    fn read(record: &[u8]) -> Self {
        let at = |i: usize| u64::from_le_bytes(record[i..i + 8].try_into().expect("eight bytes"));
        Self { time: at(0), sample: at(8) }
    }

    fn write(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.time.to_le_bytes());
        out.extend_from_slice(&self.sample.to_le_bytes());
    }
}

/// What a COP2 stamp is predicted from: the last capture-time step and the block's frame count.
#[derive(Clone, Copy)]
struct Cadence {
    step: u64,
    frames: u64,
}

impl Cadence {
    /// The nominal block duration starts the prediction; validated layouts make the division exact enough.
    fn of(layout: PcmLayout, rate: u32) -> Self {
        let frames = ((layout.stride as usize - STAMP_BYTES) / (usize::from(layout.channels) * 4)) as u64;
        Self { step: frames * 1_000_000_000 / u64::from(rate.max(1)), frames }
    }

    fn residuals(self, previous: Stamp, stamp: Stamp) -> (u64, u64) {
        let step = stamp.time.wrapping_sub(previous.time);
        (step.wrapping_sub(self.step), stamp.sample.wrapping_sub(previous.sample).wrapping_sub(self.frames))
    }

    fn next(self, previous: Stamp, time: u64, sample: u64) -> Stamp {
        Stamp {
            time: previous.time.wrapping_add(self.step.wrapping_add(time)),
            sample: previous.sample.wrapping_add(self.frames).wrapping_add(sample),
        }
    }
}

/// A wrapped difference as a signed value, so small steps either way take one byte.
fn zigzag(value: u64) -> u64 {
    let signed = value as i64;
    ((signed << 1) ^ (signed >> 63)) as u64
}

fn unzigzag(value: u64) -> u64 {
    (value >> 1) ^ (value & 1).wrapping_neg()
}

fn put_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

/// One LEB128 varint at `*at`, refusing a truncated, overlong, or overflowing one.
fn take_varint(bytes: &[u8], at: &mut usize) -> Result<u64, TierError> {
    let mut value = 0u64;
    for index in 0..MAX_VARINT {
        let byte = *bytes.get(*at).ok_or_else(|| invalid("short varint"))?;
        *at += 1;
        if index == MAX_VARINT - 1 && byte > 1 {
            return Err(invalid("varint overflows u64"));
        }
        value |= u64::from(byte & 0x7f) << (7 * index);
        if byte & 0x80 == 0 {
            if byte == 0 && index > 0 {
                return Err(invalid("overlong varint"));
            }
            return Ok(value);
        }
    }
    Err(invalid("varint overflows u64"))
}

/// The exact source format from a validated Opus batch header.
///
/// # Errors
/// A malformed or unsupported header.
pub fn format_of(record: &[u8]) -> Result<crate::pcm::PcmFormat, TierError> {
    if record.len() < HEADER || (&record[..4] != MAGIC && &record[..4] != MAGIC_V1) {
        return Err(invalid("no header"));
    }
    let sample_rate_hz = u32::from_le_bytes(record[4..8].try_into().expect("four bytes"));
    let channels = u16::from_le_bytes([record[8], record[9]]);
    let stride = u32::from_le_bytes(record[12..16].try_into().expect("four bytes"));
    let count = u32::from_le_bytes(record[16..20].try_into().expect("four bytes")) as usize;
    let layout = PcmLayout { channels, stride };
    validate(layout, count, sample_rate_hz)?;
    Ok(crate::pcm::PcmFormat { sample_rate_hz, layout })
}

/// How many sample frames an Opus batch record holds, with its format, from its header alone: the header keeps the
/// source batch's exact record count and layout, so the count matches the PCM the batch was encoded from.
///
/// # Errors
/// A malformed or unsupported header.
pub fn sample_frames(record: &[u8]) -> Result<(u64, crate::pcm::PcmFormat), TierError> {
    let format = format_of(record)?;
    let count = u32::from_le_bytes(record[16..20].try_into().expect("four bytes")) as usize;
    Ok((validate(format.layout, count, format.sample_rate_hz)? as u64, format))
}

/// Whether a record is an Opus batch (either version), by its magic.
#[must_use]
pub fn is_batch(record: &[u8]) -> bool {
    record.get(..4).is_some_and(|m| m == MAGIC || m == MAGIC_V1)
}

/// Decode one independent archived batch to f32 ring records, preserving every stamp and frame count.
///
/// # Errors
/// Corrupt/unsupported header, malformed packet sequence, or a codec error. Bounds are checked before allocation.
pub fn decode_batch(record: &[u8]) -> Result<Vec<u8>, TierError> {
    let compact = record.get(..4) == Some(MAGIC.as_slice());
    if record.len() < HEADER || !(compact || &record[..4] == MAGIC_V1) {
        return Err(invalid("no header"));
    }
    let at = |i: usize| u32::from_le_bytes(record[i..i + 4].try_into().expect("four bytes"));
    let rate = at(4);
    let channels = u16::from_le_bytes([record[8], record[9]]);
    if record[10..12] != [0, 0] {
        return Err(invalid("unknown flags"));
    }
    let layout = PcmLayout { channels, stride: at(12) };
    let count = at(16) as usize;
    let skip = at(20) as usize;
    let frames = validate(layout, count, rate)?;
    let frame_frames = rate as usize / PACKETS_PER_SECOND;
    if skip > rate as usize {
        return Err(invalid("excessive lookahead"));
    }
    let expected = (frames + skip).div_ceil(frame_frames);
    let (stamps, packets) = if compact {
        compact_parts(record, layout, rate, count, expected)?
    } else {
        legacy_parts(record, count, expected)?
    };
    let channel_mode = if channels == 1 { Channels::Mono } else { Channels::Stereo };
    let mut decoder = Decoder::new(rate, channel_mode).map_err(invalid)?;
    let n_channels = usize::from(channels);
    let mut samples = Vec::with_capacity(packets.len() * frame_frames * n_channels);
    let mut decoded = vec![0.0; frame_frames * n_channels];
    for packet in packets {
        if packet.is_empty() || packet.len() > MAX_PACKET_BYTES {
            return Err(invalid("empty or excessive packet"));
        }
        if decoder.decode_float(&packet, &mut decoded, false).map_err(invalid)? != frame_frames {
            return Err(invalid("packet is not 20 ms"));
        }
        samples.extend_from_slice(&decoded);
    }
    let begin = skip * n_channels;
    let samples = &samples[begin..begin + frames * n_channels];
    let block_samples = (layout.stride as usize - STAMP_BYTES) / 4;
    let mut out = Vec::with_capacity(count * layout.stride as usize);
    for (stamp, block) in stamps.chunks_exact(STAMP_BYTES).zip(samples.chunks_exact(block_samples)) {
        out.extend_from_slice(stamp);
        for &sample in block {
            out.extend_from_slice(&sample.to_le_bytes());
        }
    }
    Ok(out)
}

type Parts<'a> = (Vec<u8>, Vec<Cow<'a, [u8]>>);

/// COP1's raw stamps and gop-table packets, bounded before the table is parsed.
fn legacy_parts(record: &[u8], count: usize, expected: usize) -> Result<Parts<'_>, TierError> {
    let stamps_end = HEADER
        .checked_add(count * STAMP_BYTES)
        .filter(|&end| end <= record.len())
        .ok_or_else(|| invalid("short stamps"))?;
    let table = &record[stamps_end..];
    let count_bytes = table.get(..4).ok_or_else(|| invalid("no packet table"))?;
    let packet_count = u32::from_le_bytes(count_bytes.try_into().expect("four bytes")) as usize;
    if packet_count != expected || table.len() > 4 + expected * (gop::ROW + MAX_PACKET_BYTES) {
        return Err(invalid("wrong packet count or excessive packet bytes"));
    }
    let packets = gop::decode(table)?.into_iter().map(|unit| Cow::Owned(unit.bytes)).collect();
    Ok((record[HEADER..stamps_end].to_vec(), packets))
}

/// COP2's predicted stamps and length-prefixed packets; every byte must be accounted for.
fn compact_parts(
    record: &[u8],
    layout: PcmLayout,
    rate: u32,
    count: usize,
    expected: usize,
) -> Result<Parts<'_>, TierError> {
    let first = record.get(HEADER..HEADER + STAMP_BYTES).ok_or_else(|| invalid("short stamps"))?;
    let mut stamps = Vec::with_capacity(count * STAMP_BYTES);
    stamps.extend_from_slice(first);
    let mut cadence = Cadence::of(layout, rate);
    let mut previous = Stamp::read(first);
    let mut at = HEADER + STAMP_BYTES;
    for _ in 1..count {
        let time = unzigzag(take_varint(record, &mut at)?);
        let sample = unzigzag(take_varint(record, &mut at)?);
        let stamp = cadence.next(previous, time, sample);
        stamp.write(&mut stamps);
        cadence.step = stamp.time.wrapping_sub(previous.time);
        previous = stamp;
    }
    let mut lengths = Vec::with_capacity(expected.min(record.len()));
    for _ in 0..expected {
        let length = usize::try_from(take_varint(record, &mut at)?).map_err(invalid)?;
        if length == 0 || length > MAX_PACKET_BYTES {
            return Err(invalid("empty or excessive packet"));
        }
        lengths.push(length);
    }
    let mut packets = Vec::with_capacity(expected);
    for length in lengths {
        packets.push(Cow::Borrowed(record.get(at..at + length).ok_or_else(|| invalid("short packet"))?));
        at += length;
    }
    if at != record.len() {
        return Err(invalid("trailing bytes"));
    }
    Ok((stamps, packets))
}

/// Turns a finished PCM batch chunk ([`crate::ChunkFormat::PcmBatch`]) into an Opus chunk in place, one independent
/// Opus batch per PCM record, keeping each record's payload id, times, and sequence number. The Opus file is written
/// beside it as `<path>.partial`, synced, and renamed over `path`, and the directory is synced: at every moment `path`
/// is one whole, durable chunk, and a failure leaves the PCM chunk as it was. Returns the Opus chunk's records.
///
/// # Errors
/// A record without a known source rate (a legacy CPB1 batch, or a stream whose layout was unknown), a rate or layout
/// Opus can't keep ([`supported`]), a purge-cut record, a codec error, or a chunk or I/O failure.
pub fn pcm_chunk_to_opus(path: &std::path::Path) -> Result<Vec<crate::chunk::Record>, TierError> {
    use crate::chunk::{ChunkFormat, ChunkReader, ChunkWriter};
    let temp = crate::chunk::partial(path);
    let result = (|| {
        let mut reader = ChunkReader::open(path)?;
        if reader.format() != ChunkFormat::PcmBatch || !reader.finalized() {
            return Err(invalid("source is not a finished PCM batch chunk"));
        }
        let mut writer = ChunkWriter::create(&temp, reader.stream(), ChunkFormat::Opus, reader.created_ns())?;
        for (index, record) in reader.records().to_vec().iter().enumerate() {
            if record.cut {
                return Err(invalid("a record was cut by a purge"));
            }
            let packed = reader.read(index)?;
            let format = crate::pcm::format_of(&packed)?.ok_or_else(|| invalid("unknown source rate"))?;
            let raw = crate::pcm::decode_batch(&packed)?;
            let encoded = encode_batch(&raw, format.layout, format.sample_rate_hz)?;
            writer.append_payload(record.id, record.start_ns, record.end_ns, record.seq, &encoded)?;
        }
        let records = writer.close()?;
        std::fs::rename(&temp, path)?;
        // The file is Opus from here on, so its records are what must be committed whatever happens next: a failed
        // directory sync is logged, not returned (the caller would fall back to PCM records that are gone).
        if let Some(dir) = path.parent()
            && let Err(e) = std::fs::File::open(dir).and_then(|d| d.sync_all())
        {
            tracing::warn!("syncing {} after its Opus rename failed: {e}", dir.display());
        }
        Ok(records)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temp);
    }
    result
}
