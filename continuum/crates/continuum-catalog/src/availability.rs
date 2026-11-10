//! Single-snapshot parallel scan for reconstructing the broker's availability index at cold start.
//!
//! Walking `TIME_INDEX` stream by stream does four to five random UUID B-tree lookups per payload across `PAYLOADS`,
//! `PAYLOAD_SEQ`, `LOCATIONS`, `DOOMED`, and `CHUNKS`, while allocating and copying inline payload bytes that the
//! index never uses. Instead, this module reads one coherent `ReadTransaction`, partitions `PAYLOADS` by tightened
//! UUID key ranges, and merge-joins `PAYLOADS` with `LOCATIONS`, `DOOMED`, and `CHUNKS` sequentially across worker
//! threads without per-row heap allocation or inline-byte copies.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};
use std::time::Instant;

use continuum_core::{DecodeError, Interval, StageId, StorageTier, StreamId, Uuid};
use redb::{AccessGuard, Range, ReadOnlyTable, ReadableTable, ReadableTableMetadata};

use crate::store::R;
use crate::tables::{CHUNKS, COVERAGE, DOOMED, LOCATIONS, PAYLOADS, STREAMS, TIME_INDEX};
use crate::{
    AvailabilityCopy, AvailabilityCutChunk, AvailabilitySnapshot, CatalogError, CoverageSet, Result, StreamInfo, codec,
    frame_slots,
};

type ReadTable = ReadOnlyTable<&'static [u8], &'static [u8]>;

/// Minimum payload count before spawning worker threads; smaller catalogs scan inline on the caller's thread.
const PARALLEL_MIN_PAYLOADS: u64 = 4_096;
/// Maximum bisection depth (`2^6 = 64` tightened non-empty key shards) for dynamic work stealing.
const SHARD_DEPTH: usize = 6;
/// Cap on parallel scan threads so startup never over-subscribes huge core-count hosts.
const MAX_SCAN_WORKERS: usize = 16;

fn corrupt(what: &str) -> CatalogError {
    CatalogError::Storage(format!("catalog record is corrupt: {what}").into())
}

fn bad_header(e: DecodeError) -> CatalogError {
    CatalogError::Storage(Box::new(e))
}

/// Reads `(id, stream, span, seq)` directly from a `PayloadHeader` FlatBuffer slice without copying or verifying
/// unrelated variable-length fields (`shape_name`, `parents`, `subjects`, or `capture`).
pub(crate) fn fast_header_fields(buf: &[u8]) -> Result<(Uuid, StreamId, Option<Interval>, u32)> {
    let invalid = || corrupt("invalid payload header");
    if buf.len() < 8 {
        return Err(invalid());
    }
    let root = u32::from_le_bytes(buf.get(0..4).and_then(|s| s.try_into().ok()).ok_or_else(invalid)?) as usize;
    if root < 4 || (root & 3) != 0 || root > buf.len() - 4 {
        return Err(invalid());
    }
    let soffset = i32::from_le_bytes(buf.get(root..root + 4).and_then(|s| s.try_into().ok()).ok_or_else(invalid)?);
    let vtable = isize::try_from(soffset).ok().and_then(|s| root.checked_sub_signed(s)).ok_or_else(invalid)?;
    if (vtable & 1) != 0 {
        return Err(invalid());
    }
    let vtable_len =
        u16::from_le_bytes(buf.get(vtable..vtable + 2).and_then(|s| s.try_into().ok()).ok_or_else(invalid)?) as usize;
    let obj_len =
        u16::from_le_bytes(buf.get(vtable + 2..vtable + 4).and_then(|s| s.try_into().ok()).ok_or_else(invalid)?)
            as usize;
    if vtable_len < 4
        || !vtable_len.is_multiple_of(2)
        || obj_len < 4
        || vtable.checked_add(vtable_len).is_none_or(|end| end > buf.len())
        || root.checked_add(obj_len).is_none_or(|end| end > buf.len())
    {
        return Err(invalid());
    }

    let slot = |vt: usize| -> Result<usize> {
        if vt >= vtable_len {
            return Ok(0);
        }
        let off = u16::from_le_bytes(
            buf.get(vtable + vt..vtable + vt + 2).and_then(|s| s.try_into().ok()).ok_or_else(invalid)?,
        ) as usize;
        if off != 0 && off < 4 {
            return Err(invalid());
        }
        Ok(off)
    };

    // VT_ID = 4 ([u8; 16] struct, required by HeaderBuf::new).
    let id_off = slot(4)?;
    if id_off == 0 {
        return Err(bad_header(DecodeError::MissingId));
    }
    if id_off + 16 > obj_len {
        return Err(invalid());
    }
    let id_bytes: [u8; 16] =
        buf.get(root + id_off..root + id_off + 16).and_then(|s| s.try_into().ok()).ok_or_else(invalid)?;

    // VT_STREAM_ID = 6 (u32, default 0).
    let stream_off = slot(6)?;
    let stream = if stream_off == 0 {
        0
    } else {
        if stream_off + 4 > obj_len || ((root + stream_off) & 3) != 0 {
            return Err(invalid());
        }
        u32::from_le_bytes(
            buf.get(root + stream_off..root + stream_off + 4).and_then(|s| s.try_into().ok()).ok_or_else(invalid)?,
        )
    };

    // VT_START_NS = 8 (u64, default 0).
    let start_off = slot(8)?;
    let start_ns = if start_off == 0 {
        0
    } else {
        if start_off + 8 > obj_len || ((root + start_off) & 7) != 0 {
            return Err(invalid());
        }
        u64::from_le_bytes(
            buf.get(root + start_off..root + start_off + 8).and_then(|s| s.try_into().ok()).ok_or_else(invalid)?,
        )
    };

    // VT_END_NS = 10 (u64, default 0).
    let end_off = slot(10)?;
    let end_ns = if end_off == 0 {
        0
    } else {
        if end_off + 8 > obj_len || ((root + end_off) & 7) != 0 {
            return Err(invalid());
        }
        u64::from_le_bytes(
            buf.get(root + end_off..root + end_off + 8).and_then(|s| s.try_into().ok()).ok_or_else(invalid)?,
        )
    };

    // VT_SEQ = 12 (u32, default 0).
    let seq_off = slot(12)?;
    let seq = if seq_off == 0 {
        0
    } else {
        if seq_off + 4 > obj_len || ((root + seq_off) & 3) != 0 {
            return Err(invalid());
        }
        u32::from_le_bytes(
            buf.get(root + seq_off..root + seq_off + 4).and_then(|s| s.try_into().ok()).ok_or_else(invalid)?,
        )
    };

    if end_ns < start_ns {
        return Err(bad_header(DecodeError::EndBeforeStart));
    }
    Ok((Uuid(id_bytes), StreamId(stream), Interval::covering(start_ns, end_ns), seq))
}

enum StreamLookup {
    Dense(Vec<u32>),
    Sparse(HashMap<u32, usize>),
}

impl StreamLookup {
    fn new(streams: &[(StreamId, StreamInfo)]) -> Self {
        let max_id = streams.last().map_or(0, |(s, _)| s.0);
        if max_id <= 65_536 {
            let mut dense = vec![u32::MAX; (max_id as usize) + 1];
            for (idx, &(id, _)) in streams.iter().enumerate() {
                dense[id.0 as usize] = u32::try_from(idx).unwrap_or(u32::MAX);
            }
            Self::Dense(dense)
        } else {
            let sparse = streams.iter().enumerate().map(|(idx, &(id, _))| (id.0, idx)).collect();
            Self::Sparse(sparse)
        }
    }

    fn index_of(&self, stream: StreamId) -> Result<usize> {
        let found = match self {
            Self::Dense(dense) => dense.get(stream.0 as usize).copied().filter(|&v| v != u32::MAX).map(|v| v as usize),
            Self::Sparse(sparse) => sparse.get(&stream.0).copied(),
        };
        found.ok_or_else(|| corrupt("payload names an unregistered stream"))
    }
}

type ByteRange<'a> = Range<'a, &'static [u8], &'static [u8]>;
type ByteEntry<'a> = (AccessGuard<'a, &'static [u8]>, AccessGuard<'a, &'static [u8]>);

struct Cursor<'a> {
    range: ByteRange<'a>,
    head: Option<ByteEntry<'a>>,
}

impl<'a> Cursor<'a> {
    fn new(mut range: ByteRange<'a>) -> Result<Self> {
        let head = range.next().transpose()?;
        Ok(Self { range, head })
    }

    fn key(&self) -> Option<&[u8]> {
        self.head.as_ref().map(|(k, _)| k.value())
    }

    fn value(&self) -> &[u8] {
        self.head.as_ref().map_or(&[], |(_, v)| v.value())
    }

    fn advance(&mut self) -> Result<()> {
        self.head = self.range.next().transpose()?;
        Ok(())
    }
}

fn key_u128(raw: &[u8], label: &str) -> Result<u128> {
    let arr: [u8; 16] = raw.try_into().map_err(|_| corrupt(label))?;
    Ok(u128::from_be_bytes(arr))
}

/// Recursively bisects `[lo, hi]` (where both `lo` and `hi` are existing keys in `pay`), tightening each child bound
/// with one B-tree seek so large empty UUID gaps collapse immediately and shards stay disjoint, non-empty, and sorted.
fn collect_shards(
    pay: &ReadTable,
    lo: u128,
    hi: u128,
    depth: usize,
    out: &mut Vec<([u8; 16], [u8; 16])>,
) -> Result<()> {
    if depth == 0 || lo >= hi {
        out.push((lo.to_be_bytes(), hi.to_be_bytes()));
        return Ok(());
    }
    let mid = lo + (hi - lo) / 2;
    let (lo_b, mid_b, next_b, hi_b) = (lo.to_be_bytes(), mid.to_be_bytes(), (mid + 1).to_be_bytes(), hi.to_be_bytes());
    let left_hi = pay
        .range::<&[u8]>(lo_b.as_slice()..=mid_b.as_slice())?
        .next_back()
        .transpose()?
        .map(|(k, _)| key_u128(k.value(), "payload key"))
        .transpose()?
        .unwrap_or(lo);
    let right_lo = pay
        .range::<&[u8]>(next_b.as_slice()..=hi_b.as_slice())?
        .next()
        .transpose()?
        .map(|(k, _)| key_u128(k.value(), "payload key"))
        .transpose()?
        .unwrap_or(hi);
    collect_shards(pay, lo, left_hi, depth - 1, out)?;
    collect_shards(pay, right_lo, hi, depth - 1, out)
}

#[derive(Default)]
struct ShardOut {
    payloads: usize,
    stream_copies: Vec<Vec<AvailabilityCopy>>,
    cut_chunks: Vec<AvailabilityCutChunk>,
    stale_tier0: Vec<Uuid>,
}

struct ScanTables<'a> {
    pay: &'a ReadTable,
    loc: &'a ReadTable,
    doom: Option<&'a ReadTable>,
    chk: Option<&'a ReadTable>,
    present_ids: Option<&'a [[u8; 16]]>,
    stream_lookup: &'a StreamLookup,
    stream_count: usize,
}

fn scan_shard(tables: &ScanTables<'_>, lo: [u8; 16], hi: [u8; 16]) -> Result<ShardOut> {
    let mut out = ShardOut {
        payloads: 0,
        stream_copies: vec![Vec::new(); tables.stream_count],
        cut_chunks: Vec::new(),
        stale_tier0: Vec::new(),
    };
    let mut doomed_cursor = match tables.doom {
        Some(doom) => Some(Cursor::new(doom.range::<&[u8]>(lo.as_slice()..=hi.as_slice())?)?),
        None => None,
    };
    let mut chunks_cursor = match tables.chk {
        Some(chk) => Some(Cursor::new(chk.range::<&[u8]>(lo.as_slice()..=hi.as_slice())?)?),
        None => None,
    };
    let mut lo_loc = [0u8; 17];
    lo_loc[..16].copy_from_slice(&lo);
    let mut hi_loc = [0u8; 17];
    hi_loc[..16].copy_from_slice(&hi);
    hi_loc[16] = 0xff;
    let mut loc_cursor = Cursor::new(tables.loc.range::<&[u8]>(lo_loc.as_slice()..=hi_loc.as_slice())?)?;
    let mut present_pos = tables.present_ids.map_or(0, |ids| ids.partition_point(|x| x < &lo));

    for row in tables.pay.range::<&[u8]>(lo.as_slice()..=hi.as_slice())? {
        let (pk, pv) = row?;
        let raw_key: [u8; 16] = pk.value().try_into().map_err(|_| corrupt("payload key"))?;

        if let Some(dc) = &mut doomed_cursor {
            let mut is_doomed = false;
            while let Some(dk) = dc.key() {
                if dk.len() != 16 {
                    return Err(corrupt("doomed key"));
                }
                match dk.cmp(raw_key.as_slice()) {
                    Ordering::Less => dc.advance()?,
                    Ordering::Equal => {
                        is_doomed = true;
                        dc.advance()?;
                        break;
                    }
                    Ordering::Greater => break,
                }
            }
            if is_doomed {
                continue;
            }
        }

        if let Some(present) = tables.present_ids {
            while present_pos < present.len() && present[present_pos] < raw_key {
                present_pos += 1;
            }
            if present.get(present_pos) != Some(&raw_key) {
                continue;
            }
            present_pos += 1;
        }

        let (hdr_id, stream, span, seq) = fast_header_fields(pv.value())?;
        let stream_idx = tables.stream_lookup.index_of(stream)?;
        let Some(span) = span else {
            continue;
        };
        out.payloads += 1;

        let mut cut_slots = Vec::new();
        if let Some(cc) = &mut chunks_cursor {
            while let Some(ck) = cc.key() {
                if ck.len() != 16 {
                    return Err(corrupt("chunk key"));
                }
                match ck.cmp(raw_key.as_slice()) {
                    Ordering::Less => cc.advance()?,
                    Ordering::Equal => {
                        let frames = codec::read_frames(cc.value())?;
                        if frames.iter().any(|f| f.cut) {
                            cut_slots = frames
                                .iter()
                                .zip(frame_slots(span, &frames))
                                .filter(|(f, _)| f.cut)
                                .map(|(_, s)| s)
                                .collect();
                        }
                        cc.advance()?;
                        break;
                    }
                    Ordering::Greater => break,
                }
            }
        }

        let mut warm = 0u8;
        let mut cold = 0u8;
        while let Some(lk) = loc_cursor.key() {
            if lk.len() != 17 {
                return Err(corrupt("location key"));
            }
            match lk[..16].cmp(raw_key.as_slice()) {
                Ordering::Less => loc_cursor.advance()?,
                Ordering::Equal => {
                    match lk[16] {
                        0 => out.stale_tier0.push(hdr_id),
                        1 => warm = warm.saturating_add(1),
                        2 => cold = cold.saturating_add(1),
                        _ => return Err(bad_header(DecodeError::BadLocation)),
                    }
                    loc_cursor.advance()?;
                }
                Ordering::Greater => break,
            }
        }

        if !cut_slots.is_empty() {
            let mut tiers = Vec::with_capacity(usize::from(warm) + usize::from(cold));
            tiers.extend(std::iter::repeat_n(StorageTier::Tier1Warm, usize::from(warm)));
            tiers.extend(std::iter::repeat_n(StorageTier::Tier2Cold, usize::from(cold)));
            out.cut_chunks.push(AvailabilityCutChunk { id: hdr_id, stream, span, seq, tiers, cut_slots });
        } else if warm > 0 || cold > 0 {
            out.stream_copies[stream_idx].push(AvailabilityCopy { span, seq, warm, cold });
        }
    }
    Ok(out)
}

/// Collects every payload id present in `TIME_INDEX`, sorted by UUID. Only called in the rare case where a
/// worker-cut frame's `(stream, start_ns, seq)` key was taken by a chunk (`TIME_INDEX.len() < PAYLOADS.len()`).
fn collect_time_indexed_ids(ti: &ReadTable) -> Result<Vec<[u8; 16]>> {
    let cap = usize::try_from(ti.len()?).unwrap_or(0);
    let mut ids = Vec::with_capacity(cap);
    for row in ti.range::<&[u8]>(b"".as_slice()..)? {
        let (k, v) = row?;
        if k.value().len() != 16 {
            return Err(corrupt("time_index key"));
        }
        let id: [u8; 16] = v.value().try_into().map_err(|_| corrupt("time_index value"))?;
        ids.push(id);
    }
    ids.sort_unstable();
    Ok(ids)
}

pub(crate) fn snapshot(r: &R) -> Result<AvailabilitySnapshot> {
    let stream_start = Instant::now();
    let streams_table = r.txn.open_table(STREAMS)?;
    let mut streams = Vec::new();
    for row in streams_table.range::<&[u8]>(b"".as_slice()..)? {
        let (k, v) = row?;
        if k.value().len() != 4 {
            return Err(corrupt("stream key"));
        }
        streams.push((StreamId(codec::u32_at(k.value(), 0)?), codec::read_stream_info(v.value())?));
    }
    let stream_lookup = StreamLookup::new(&streams);
    let stream_reads = stream_start.elapsed();

    let payload_start = Instant::now();
    let pay = r.txn.open_table(PAYLOADS)?;
    let loc = r.txn.open_table(LOCATIONS)?;
    let doom = r.txn.open_table(DOOMED)?;
    let chk = r.txn.open_table(CHUNKS)?;
    let ti = r.txn.open_table(TIME_INDEX)?;

    let pay_len = pay.len()?;
    let ti_len = ti.len()?;
    let has_doomed = doom.len()? > 0;
    let has_chunks = chk.len()? > 0;

    let mut total_payloads = 0usize;
    let mut stream_copies = vec![Vec::new(); streams.len()];
    let mut cut_chunks = Vec::new();
    let mut stale_tier0 = Vec::new();

    if let (Some(first_row), Some(last_row)) = (pay.first()?, pay.last()?) {
        let first_id = key_u128(first_row.0.value(), "payload key")?;
        let last_id = key_u128(last_row.0.value(), "payload key")?;
        drop((first_row, last_row));

        let present_ids = if ti_len == pay_len { None } else { Some(collect_time_indexed_ids(&ti)?) };
        let scan_tables = ScanTables {
            pay: &pay,
            loc: &loc,
            doom: has_doomed.then_some(&doom),
            chk: has_chunks.then_some(&chk),
            present_ids: present_ids.as_deref(),
            stream_lookup: &stream_lookup,
            stream_count: streams.len(),
        };

        if pay_len < PARALLEL_MIN_PAYLOADS {
            let single = scan_shard(&scan_tables, first_id.to_be_bytes(), last_id.to_be_bytes())?;
            total_payloads = single.payloads;
            stream_copies = single.stream_copies;
            cut_chunks = single.cut_chunks;
            stale_tier0 = single.stale_tier0;
        } else {
            let mut shards = Vec::with_capacity(1 << SHARD_DEPTH);
            collect_shards(&pay, first_id, last_id, SHARD_DEPTH, &mut shards)?;
            let workers = std::thread::available_parallelism()
                .map_or(4, std::num::NonZeroUsize::get)
                .min(MAX_SCAN_WORKERS)
                .min(shards.len());
            let results: Vec<OnceLock<ShardOut>> = (0..shards.len()).map(|_| OnceLock::new()).collect();
            let next_shard = AtomicUsize::new(0);
            let abort = AtomicBool::new(false);

            std::thread::scope(|scope| -> Result<()> {
                let mut handles = Vec::with_capacity(workers);
                for _ in 0..workers {
                    let shards_ref = &shards;
                    let results_ref = &results;
                    let next_ref = &next_shard;
                    let abort_ref = &abort;
                    let tables_ref = &scan_tables;
                    handles.push(scope.spawn(move || -> Result<()> {
                        while !abort_ref.load(AtomicOrdering::Relaxed) {
                            let idx = next_ref.fetch_add(1, AtomicOrdering::Relaxed);
                            let Some(&(lo, hi)) = shards_ref.get(idx) else { break };
                            match scan_shard(tables_ref, lo, hi) {
                                Ok(shard_out) => {
                                    let _ = results_ref[idx].set(shard_out);
                                }
                                Err(err) => {
                                    abort_ref.store(true, AtomicOrdering::Relaxed);
                                    return Err(err);
                                }
                            }
                        }
                        Ok(())
                    }));
                }
                let mut first_err: Option<CatalogError> = None;
                for handle in handles {
                    match handle.join() {
                        Ok(Ok(())) => {}
                        Ok(Err(err)) => {
                            if first_err.is_none() {
                                first_err = Some(err);
                            }
                        }
                        Err(_) => {
                            if first_err.is_none() {
                                first_err = Some(CatalogError::Storage("availability scan worker panicked".into()));
                            }
                        }
                    }
                }
                if let Some(err) = first_err {
                    return Err(err);
                }
                Ok(())
            })?;

            // Pre-size per-stream copy buffers once across all shards before moving items in shard order.
            let mut per_stream_caps = vec![0usize; streams.len()];
            let mut cut_cap = 0usize;
            let mut hot_cap = 0usize;
            for slot in &results {
                if let Some(s) = slot.get() {
                    total_payloads += s.payloads;
                    cut_cap += s.cut_chunks.len();
                    hot_cap += s.stale_tier0.len();
                    for (cap, v) in per_stream_caps.iter_mut().zip(&s.stream_copies) {
                        *cap += v.len();
                    }
                }
            }
            for (dst, cap) in stream_copies.iter_mut().zip(per_stream_caps) {
                dst.reserve(cap);
            }
            cut_chunks.reserve(cut_cap);
            stale_tier0.reserve(hot_cap);
            for mut slot in results {
                let Some(s) = slot.take() else { continue };
                cut_chunks.extend(s.cut_chunks);
                stale_tier0.extend(s.stale_tier0);
                for (dst, mut src) in stream_copies.iter_mut().zip(s.stream_copies) {
                    dst.append(&mut src);
                }
            }
        }
    }
    let payload_reads = payload_start.elapsed();

    let coverage_start = Instant::now();
    let cov_table = r.txn.open_table(COVERAGE)?;
    let mut coverage: Vec<((StreamId, StageId), CoverageSet)> = Vec::new();
    for row in cov_table.range::<&[u8]>(b"".as_slice()..)? {
        let (k, v) = row?;
        let (kb, vb) = (k.value(), v.value());
        if kb.len() != 16 || vb.len() != 8 {
            return Err(corrupt("coverage row"));
        }
        let pair = (StreamId(codec::u32_at(kb, 0)?), StageId(codec::u32_at(kb, 4)?));
        let (start, end) = (codec::u64_at(kb, 8)?, codec::u64_at(vb, 0)?);
        let span = Interval::new(start, end).ok_or_else(|| corrupt("empty coverage interval"))?;
        match coverage.last_mut() {
            Some((last_pair, set)) if *last_pair == pair => set.insert(span),
            _ => {
                let mut set = CoverageSet::new();
                set.insert(span);
                coverage.push((pair, set));
            }
        }
    }
    let coverage_reads = coverage_start.elapsed();

    Ok(AvailabilitySnapshot {
        streams,
        payloads: total_payloads,
        stream_copies,
        cut_chunks,
        stale_tier0,
        coverage,
        stream_reads,
        payload_reads,
        coverage_reads,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use continuum_core::{HeaderBuf, HeaderSpec};
    use proptest::prelude::*;
    use redb::{Database, ReadableDatabase};

    #[test]
    fn fast_header_handles_default_zero_and_u64_max_boundary_fields() {
        // Default zero stream_id, start_ns, end_ns, seq (FlatBuffers omits default-valued scalar slots from vtable).
        let raw_id = Uuid([0xab; 16]);
        let minimal = HeaderBuf::build(&HeaderSpec::raw(raw_id, StreamId(0), 0, 0)).unwrap();
        assert_eq!(
            fast_header_fields(minimal.as_bytes()).unwrap(),
            (minimal.id(), minimal.stream(), minimal.span(), minimal.seq()),
        );

        // Instant at u64::MAX (`span == None`) with parents, subjects, and shape_name populated.
        let mut spec = HeaderSpec::raw(raw_id, StreamId(42), u64::MAX, u64::MAX);
        spec.seq = 9;
        spec.shape_name = "NV12".into();
        spec.shape_version = 3;
        spec.parents = vec![raw_id];
        spec.subjects = vec![raw_id];
        let boundary = HeaderBuf::build(&spec).unwrap();
        assert_eq!(boundary.span(), None);
        assert_eq!(fast_header_fields(boundary.as_bytes()).unwrap(), (raw_id, StreamId(42), None, 9));
    }

    #[test]
    fn shard_bisection_covers_bimodal_and_boundary_keys_exactly_once() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::create(dir.path().join("shards.redb")).unwrap();
        let mut expected = Vec::new();
        // Outlier at 0, tight middle cluster, and outlier at u128::MAX.
        expected.push(0u128);
        for k in 0..300u128 {
            expected.push((0x0189_0a5d_ac96_774b_u128 << 64) | (k * 17));
        }
        expected.push(u128::MAX);
        expected.sort_unstable();
        expected.dedup();

        let w = db.begin_write().unwrap();
        {
            let mut pay = w.open_table(PAYLOADS).unwrap();
            for &id in &expected {
                let bytes = id.to_be_bytes();
                pay.insert(bytes.as_slice(), b"x".as_slice()).unwrap();
            }
        }
        w.commit().unwrap();

        let r = db.begin_read().unwrap();
        let pay = r.open_table(PAYLOADS).unwrap();
        let mut shards = Vec::new();
        collect_shards(&pay, expected[0], *expected.last().unwrap(), 6, &mut shards).unwrap();
        assert!(shards.len() > 4, "cluster should split across multiple shards");
        for pair in shards.windows(2) {
            assert!(pair[0].1 < pair[1].0, "shards must be strictly disjoint and ordered");
        }
        let mut actual = Vec::new();
        for (lo, hi) in shards {
            for row in pay.range::<&[u8]>(lo.as_slice()..=hi.as_slice()).unwrap() {
                let (k, _) = row.unwrap();
                actual.push(key_u128(k.value(), "test").unwrap());
            }
        }
        assert_eq!(actual, expected);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn fast_header_matches_verified_header_buf(
            raw_id in any::<[u8; 16]>(),
            stream in any::<u32>(),
            start_ns in prop_oneof![0u64..1_000_000, (u64::MAX - 1000)..=u64::MAX, any::<u64>()],
            delta in 0u64..10_000,
            seq in any::<u32>(),
            shape in "[a-z0-9._]{0,24}",
            parents in proptest::collection::vec(any::<[u8; 16]>(), 0..4),
            subjects in proptest::collection::vec(any::<[u8; 16]>(), 0..4),
        ) {
            let end_ns = start_ns.saturating_add(delta);
            let mut spec = HeaderSpec::raw(Uuid(raw_id), StreamId(stream), start_ns, end_ns);
            spec.seq = seq;
            spec.shape_name = shape;
            spec.parents = parents.into_iter().map(Uuid).collect();
            spec.subjects = subjects.into_iter().map(Uuid).collect();
            let built = HeaderBuf::build(&spec).unwrap();
            let fast = fast_header_fields(built.as_bytes()).unwrap();
            prop_assert_eq!(fast, (built.id(), built.stream(), built.span(), built.seq()));
        }

        #[test]
        fn fast_header_never_panics_on_arbitrary_bytes(bytes in proptest::collection::vec(any::<u8>(), 0..256)) {
            let _ = fast_header_fields(&bytes);
        }
    }
}
