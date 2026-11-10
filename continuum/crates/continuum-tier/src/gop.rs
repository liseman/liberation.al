//! Unit records: how a raw camera's H.264 and Opus audio sit in the chunk container (docs/CONTINUUM.md,
//! "Storage tiers").
//!
//! A raw camera's Tier 1 ([`ChunkFormat::H264Au`](crate::ChunkFormat::H264Au)) keeps one access unit per record, and
//! Opus ([`ChunkFormat::Opus`](crate::ChunkFormat::Opus)) keeps about a second of packets per record, in the same
//! CNK1 chunk files as everything else ([`crate::chunk`]). The record's `start_ns` and `end_ns` are the half-open
//! span its units cover ([`span`]), so the chunk's footer is the unit index, a purge cuts whole records (a tombstone
//! record), and the gateway hands the Annex B access units to WebCodecs without demuxing anything.
//!
//! A record's bytes, all integers little-endian:
//!
//! ```text
//! count u32
//! count x ( start_ns u64 | end_ns u64 | flags u8 (bit 0: keyframe) | offset u32 | len u32 )   (25 bytes each)
//! the units' bytes, concatenated; each row's offset and len are relative to the start of this section
//! ```
//!
//! Units are in decode order, each with its own presentation times (B-frames reorder; a player sorts by time). For
//! video, a keyframe unit is an Annex B access unit carrying the SPS and PPS in-band, so the run from it decodes on
//! its own. For Opus, each unit is one packet (20 ms) and none is a keyframe.

use crate::TierError;

/// The bytes of one row in a record's table.
pub const ROW: usize = 25;
const COUNT: usize = 4;

/// One unit: an access unit of video, or a packet of audio.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unit {
    /// Its presentation start time, UTC nanoseconds.
    pub start_ns: u64,
    /// Its end time; equal to `start_ns` for an instant.
    pub end_ns: u64,
    /// A keyframe: decoding can start here.
    pub key: bool,
    /// Its bytes (Annex B for video, one Opus packet for audio).
    pub bytes: Vec<u8>,
}

fn invalid(why: &str) -> TierError {
    TierError::Codec(format!("GOP record: {why}"))
}

/// The half-open span a record's units cover, `[earliest start, latest end)`, where an instant covers one
/// nanosecond: the times the record itself is written with.
///
/// # Errors
///
/// [`TierError::Codec`] for no units.
pub fn span(units: &[Unit]) -> Result<(u64, u64), TierError> {
    let start = units.iter().map(|u| u.start_ns).min().ok_or_else(|| invalid("no units"))?;
    let end = units.iter().map(|u| u.end_ns.max(u.start_ns.saturating_add(1))).max().unwrap_or(start);
    Ok((start, end))
}

/// Lays out a record's bytes.
///
/// # Errors
///
/// [`TierError::Codec`] for no units, a unit that ends before it starts, or more than the format's u32 offsets hold.
pub fn encode(units: &[Unit]) -> Result<Vec<u8>, TierError> {
    if units.is_empty() {
        return Err(invalid("no units"));
    }
    let count = u32::try_from(units.len()).map_err(|_| invalid("too many units"))?;
    let body: usize = units.iter().map(|u| u.bytes.len()).sum();
    if u32::try_from(body).is_err() {
        return Err(invalid("units larger than 4 GiB"));
    }
    let mut out = Vec::with_capacity(COUNT + units.len() * ROW + body);
    out.extend_from_slice(&count.to_le_bytes());
    let mut offset = 0u32;
    for u in units {
        if u.end_ns < u.start_ns {
            return Err(invalid("a unit ends before it starts"));
        }
        // body fits a u32, so every length and running offset does too.
        let len = u.bytes.len() as u32;
        out.extend_from_slice(&u.start_ns.to_le_bytes());
        out.extend_from_slice(&u.end_ns.to_le_bytes());
        out.push(u8::from(u.key));
        out.extend_from_slice(&offset.to_le_bytes());
        out.extend_from_slice(&len.to_le_bytes());
        offset += len;
    }
    for u in units {
        out.extend_from_slice(&u.bytes);
    }
    Ok(out)
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap_or_default())
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap_or_default())
}

/// Reads a record's units back. Every bound is checked: the bytes come from disk.
///
/// # Errors
///
/// [`TierError::Codec`] for a record that is truncated, has no units, has a unit that ends before it starts or
/// reaches past the bytes, or has units that overlap.
pub fn decode(record: &[u8]) -> Result<Vec<Unit>, TierError> {
    if record.len() < COUNT {
        return Err(invalid("shorter than its count"));
    }
    let count = u32_at(record, 0) as usize;
    if count == 0 {
        return Err(invalid("no units"));
    }
    let table_end = count.checked_mul(ROW).and_then(|t| t.checked_add(COUNT)).ok_or_else(|| invalid("bad count"))?;
    if record.len() < table_end {
        return Err(invalid("shorter than its table"));
    }
    let body = &record[table_end..];
    let mut units = Vec::with_capacity(count);
    let mut next = 0usize;
    for i in 0..count {
        let row = COUNT + i * ROW;
        let (start_ns, end_ns) = (u64_at(record, row), u64_at(record, row + 8));
        let flags = record[row + 16];
        let (offset, len) = (u32_at(record, row + 17) as usize, u32_at(record, row + 21) as usize);
        if end_ns < start_ns {
            return Err(invalid("a unit ends before it starts"));
        }
        if flags & !1 != 0 {
            return Err(invalid("unknown flags"));
        }
        // The writer lays units out in order, back to back; anything else is damage.
        if offset != next || offset.checked_add(len).is_none_or(|e| e > body.len()) {
            return Err(invalid("a unit's bytes are out of place"));
        }
        next = offset + len;
        units.push(Unit { start_ns, end_ns, key: flags & 1 == 1, bytes: body[offset..next].to_vec() });
    }
    if next != body.len() {
        return Err(invalid("trailing bytes after the last unit"));
    }
    Ok(units)
}

/// Whether an AV1 temporal unit (low-overhead OBUs, as the archive pass writes them) carries a sequence
/// header: SVT-AV1 writes one on each keyframe, so a decoder can start there.
#[must_use]
pub fn av1_key(bytes: &[u8]) -> bool {
    let mut at = 0;
    while let Some(&header) = bytes.get(at) {
        if header & 0x80 != 0 {
            return false;
        }
        let kind = (header >> 3) & 0xf;
        at += 1 + usize::from((header >> 2) & 1);
        let size = if (header >> 1) & 1 == 1 {
            let mut size = 0usize;
            for k in 0..8 {
                let Some(&b) = bytes.get(at) else { return false };
                at += 1;
                size |= usize::from(b & 0x7f) << (7 * k);
                if b & 0x80 == 0 {
                    break;
                }
            }
            size
        } else {
            bytes.len().saturating_sub(at)
        };
        if kind == 1 {
            return true;
        }
        at = at.saturating_add(size);
    }
    false
}

/// Whether an Annex-B access unit contains an actual H.264 IDR picture (NAL type 5).
/// Parameter sets alone do not make an access unit independently decodable.
#[must_use]
pub fn h264_idr(bytes: &[u8]) -> bool {
    let mut at = 0;
    while at + 3 < bytes.len() {
        let prefix = if bytes[at..].starts_with(&[0, 0, 0, 1]) {
            4
        } else if bytes[at..].starts_with(&[0, 0, 1]) {
            3
        } else {
            at += 1;
            continue;
        };
        if let Some(&nal) = bytes.get(at + prefix)
            && nal & 0x80 == 0
            && nal & 0x1f == 5
        {
            return true;
        }
        at += prefix;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn unit(start_ns: u64, end_ns: u64, key: bool, bytes: &[u8]) -> Unit {
        Unit { start_ns, end_ns, key, bytes: bytes.to_vec() }
    }

    fn units() -> impl Strategy<Value = Vec<Unit>> {
        prop::collection::vec(
            (any::<u64>(), 0u64..1_000_000, any::<bool>(), prop::collection::vec(any::<u8>(), 0..64)).prop_map(
                |(start, dur, key, bytes)| Unit { start_ns: start, end_ns: start.saturating_add(dur), key, bytes },
            ),
            1..20,
        )
    }

    proptest! {
        #[test]
        fn a_record_round_trips(units in units()) {
            prop_assert_eq!(decode(&encode(&units).unwrap()).unwrap(), units);
        }

        #[test]
        fn any_damaged_record_is_refused_or_reads_as_whole_units(units in units(), cut in 0usize..2000) {
            // A truncated record never yields units that aren't byte-for-byte the ones written.
            let bytes = encode(&units).unwrap();
            let cut = cut.min(bytes.len());
            if let Ok(back) = decode(&bytes[..cut]) {
                prop_assert_eq!(back, units);
            }
        }
    }

    #[test]
    fn the_layout_is_the_documented_one() {
        let bytes = encode(&[unit(1, 2, true, b"ab"), unit(3, 3, false, b"c")]).unwrap();
        let mut want = 2u32.to_le_bytes().to_vec();
        want.extend(1u64.to_le_bytes());
        want.extend(2u64.to_le_bytes());
        want.push(1);
        want.extend(0u32.to_le_bytes());
        want.extend(2u32.to_le_bytes());
        want.extend(3u64.to_le_bytes());
        want.extend(3u64.to_le_bytes());
        want.push(0);
        want.extend(2u32.to_le_bytes());
        want.extend(1u32.to_le_bytes());
        want.extend(b"abc");
        assert_eq!(bytes, want);
    }

    #[test]
    fn the_span_is_half_open_and_covers_out_of_order_units() {
        // Decode order with a B-frame: the second unit is shown first.
        let gop = [unit(100, 133, true, b"i"), unit(66, 100, false, b"b"), unit(133, 166, false, b"p")];
        assert_eq!(span(&gop).unwrap(), (66, 166));
        assert_eq!(span(&[unit(5, 5, true, b"i")]).unwrap(), (5, 6));
        assert!(span(&[]).is_err());
    }

    #[test]
    fn bad_records_are_refused() {
        assert!(encode(&[]).is_err());
        assert!(encode(&[unit(5, 4, true, b"x")]).is_err());
        let good = encode(&[unit(1, 2, true, b"ab")]).unwrap();
        let mut overlong = good.clone();
        overlong[COUNT + 21] = 9; // len 9 past the body
        assert!(decode(&overlong).is_err());
        let mut flags = good.clone();
        flags[COUNT + 16] = 2;
        assert!(decode(&flags).is_err());
        let mut trailing = good.clone();
        trailing.push(0);
        assert!(decode(&trailing).is_err());
        assert!(decode(&0u32.to_le_bytes()).is_err());
        let mut huge = good.clone();
        huge[..4].copy_from_slice(&u32::MAX.to_le_bytes()); // a count no table that size could follow
        assert!(decode(&huge).is_err());
        let mut far = good.clone();
        far[COUNT + 17..COUNT + 21].copy_from_slice(&u32::MAX.to_le_bytes()); // offset past the end
        assert!(decode(&far).is_err());
        assert!(decode(&[1, 0]).is_err());
    }
}
