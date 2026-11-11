//! Single-snapshot parallel scan for reconstructing the broker's availability index at cold start.
//!
//! Walking `TIME_INDEX` stream by stream does four to five random UUID B-tree lookups per payload across `PAYLOADS`,
//! `PAYLOAD_SEQ`, `LOCATIONS`, `DOOMED`, and `CHUNKS`, while allocating and copying inline payload bytes that the
//! index never uses. Instead, this module reads one coherent `ReadTransaction`, partitions `PAYLOADS` by tightened
//! UUID key ranges, and merge-joins `PAYLOADS` with `LOCATIONS`, `DOOMED`, and `CHUNKS` sequentially across worker
//! threads without per-row heap allocation or inline-byte copies.
//!
//! Workers fold streams that have no cut chunks into [`AvailabilityRun`]s before a shard leaves the worker. The
//! caller merges those runs in shard order as each shard completes, and holds only the out-of-order prefix that is
//! still waiting. Peak raw-copy memory is one shard per worker (at most 16 of the 64 shards, about a quarter of the
//! unfolded copy vector) plus the merged runs and that waiting prefix. Cut streams stay as raw copies: removing a
//! cut does not commute. A cut-key collision no longer materializes every `TIME_INDEX` id. The scan skips payloads
//! whose time key a live cut chunk owns, and keeps that scan when the counted rows equal `TIME_INDEX.len()`. Any
//! other gap collects `TIME_INDEX` ids in parallel (serially below the parallel threshold) and filters like the
//! paged path.

use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeMap, BinaryHeap, HashMap};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::mpsc::sync_channel;
use std::time::Instant;

use continuum_core::{DecodeError, Interval, StageId, StorageTier, StreamId, Uuid};
use redb::{AccessGuard, Range, ReadOnlyTable, ReadableTable, ReadableTableMetadata};

use crate::store::R;
use crate::tables::{CHUNKS, COVERAGE, DOOMED, LOCATIONS, PAYLOADS, STREAMS, TIME_INDEX};
use crate::{
    AvailabilityCopy, AvailabilityCutChunk, AvailabilityRun, AvailabilitySnapshot, CatalogError, CoverageSet, Result,
    StreamInfo, codec, frame_slots, rate_expanded_span,
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

/// Which `PAYLOADS` rows a scan keeps.
///
/// [`Visibility::All`] is the fast path, where `TIME_INDEX` and `PAYLOADS` have the same length, so every payload
/// row is indexed. [`Visibility::SkipOwned`] drops a payload whose `(stream, start_ns, seq)` key is held by a
/// different id: the worker-cut frame a live chunk overwrote. [`Visibility::Present`] is the residual path and
/// keeps only ids that appear in `TIME_INDEX`.
enum Visibility<'a> {
    All,
    SkipOwned(&'a HashMap<(u32, u64, u32), [u8; 16]>),
    Present(&'a [[u8; 16]]),
}

#[derive(Default)]
struct ShardOut {
    payloads: usize,
    /// Rows kept from `PAYLOADS`, including doomed and spanless rows. Those rows are still indexed, so this count
    /// is comparable to `TIME_INDEX.len()`.
    indexed_rows: usize,
    stream_copies: Vec<Vec<AvailabilityCopy>>,
    cut_chunks: Vec<AvailabilityCutChunk>,
    stale_tier0: Vec<Uuid>,
}

struct ScanTables<'a> {
    pay: &'a ReadTable,
    loc: &'a ReadTable,
    doom: Option<&'a ReadTable>,
    chk: Option<&'a ReadTable>,
    visibility: Visibility<'a>,
    stream_lookup: &'a StreamLookup,
    stream_count: usize,
}

fn scan_shard(tables: &ScanTables<'_>, lo: [u8; 16], hi: [u8; 16]) -> Result<ShardOut> {
    let mut out = ShardOut {
        payloads: 0,
        indexed_rows: 0,
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
    let mut present_pos = match tables.visibility {
        Visibility::Present(ids) => ids.partition_point(|x| x < &lo),
        Visibility::All | Visibility::SkipOwned(_) => 0,
    };

    for row in tables.pay.range::<&[u8]>(lo.as_slice()..=hi.as_slice())? {
        let (pk, pv) = row?;
        let raw_key: [u8; 16] = pk.value().try_into().map_err(|_| corrupt("payload key"))?;
        let mut decoded = None;

        match tables.visibility {
            Visibility::Present(present) => {
                while present_pos < present.len() && present[present_pos] < raw_key {
                    present_pos += 1;
                }
                if present.get(present_pos) != Some(&raw_key) {
                    continue;
                }
                present_pos += 1;
                out.indexed_rows += 1;
            }
            Visibility::SkipOwned(stolen) => {
                let fields = fast_header_fields(pv.value())?;
                if let Some(span) = fields.2
                    && let Some(holder) = stolen.get(&(fields.1.0, span.start_ns(), fields.3))
                    && fields.0.0 != *holder
                {
                    continue;
                }
                out.indexed_rows += 1;
                decoded = Some(fields);
            }
            Visibility::All => {}
        }

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

        let (hdr_id, stream, span, seq) = match decoded {
            Some(fields) => fields,
            None => fast_header_fields(pv.value())?,
        };
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

/// Collects every payload id present in `TIME_INDEX`, sorted by UUID. Called whenever `TIME_INDEX.len() != PAYLOADS.len()`,
/// which occurs when a chunk overwrites a worker-cut first frame's `(stream, start_ns, seq)` key.
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
    debug_assert!(
        ids.windows(2).all(|w| w[0] < w[1]),
        "TIME_INDEX invariant violated: duplicate payload id referenced by multiple time keys",
    );
    Ok(ids)
}

/// Folds durable copies of one stream into disjoint runs.
///
/// An instant (a span of one nanosecond) on a stream with a finite `rate_hz` above zero covers one frame period
/// before the sweep, via [`rate_expanded_span`]. Overlaps sum warm and cold counts. Touching runs with equal counts
/// become one run. The output is ordered and non-overlapping, and this does not reserve one slot per copy.
#[must_use]
pub fn fold_availability_copies(rate_hz: f64, mut copies: Vec<AvailabilityCopy>) -> Vec<AvailabilityRun> {
    if copies.is_empty() {
        return Vec::new();
    }
    if !copies.is_sorted_by_key(|c| c.span.start_ns()) {
        copies.sort_unstable_by_key(|c| c.span.start_ns());
    }
    let mut runs = Vec::new();
    let mut active: BinaryHeap<Reverse<(u64, [u64; 2])>> = BinaryHeap::new();
    let mut cur = [0u64; 2];
    let mut cursor = 0u64;
    let mut have_cursor = false;
    for copy in copies {
        let add = [u64::from(copy.warm), u64::from(copy.cold)];
        if add == [0, 0] {
            continue;
        }
        let span = rate_expanded_span(rate_hz, copy.span);
        let (start, end) = (span.start_ns(), span.end_ns());
        while let Some(&Reverse((active_end, dec))) = active.peek() {
            if active_end > start {
                break;
            }
            active.pop();
            if have_cursor && active_end > cursor {
                push_counts(&mut runs, cursor, active_end, cur);
                cursor = active_end;
            }
            cur[0] -= dec[0];
            cur[1] -= dec[1];
        }
        if !have_cursor {
            cursor = start;
            have_cursor = true;
        } else if start > cursor {
            push_counts(&mut runs, cursor, start, cur);
            cursor = start;
        }
        cur[0] += add[0];
        cur[1] += add[1];
        active.push(Reverse((end, add)));
    }
    while let Some(Reverse((active_end, dec))) = active.pop() {
        if have_cursor && active_end > cursor {
            push_counts(&mut runs, cursor, active_end, cur);
            cursor = active_end;
        }
        cur[0] -= dec[0];
        cur[1] -= dec[1];
    }
    debug_assert_eq!(cur, [0, 0]);
    runs
}

fn push_counts(runs: &mut Vec<AvailabilityRun>, from: u64, to: u64, cur: [u64; 2]) {
    let warm = u32::try_from(cur[0]).unwrap_or(u32::MAX);
    let cold = u32::try_from(cur[1]).unwrap_or(u32::MAX);
    push_run(runs, from, to, warm, cold);
}

fn push_run(out: &mut Vec<AvailabilityRun>, start: u64, end: u64, warm: u32, cold: u32) {
    if start >= end || (warm == 0 && cold == 0) {
        return;
    }
    if let Some(last) = out.last_mut()
        && last.end_ns == start
        && last.warm == warm
        && last.cold == cold
    {
        last.end_ns = end;
        return;
    }
    out.push(AvailabilityRun { start_ns: start, end_ns: end, warm, cold });
}

/// Merges two folded run lists. Addition of non-cut copies commutes, so shard order does not change the result.
/// When one list ends at or before the other starts, this appends (and coalesces a shared boundary) instead of sweeping.
fn merge_runs(a: Vec<AvailabilityRun>, b: Vec<AvailabilityRun>) -> Vec<AvailabilityRun> {
    if a.is_empty() {
        return b;
    }
    if b.is_empty() {
        return a;
    }
    if a.last().is_some_and(|left| b.first().is_some_and(|right| left.end_ns <= right.start_ns)) {
        return append_runs(a, b);
    }
    if b.last().is_some_and(|left| a.first().is_some_and(|right| left.end_ns <= right.start_ns)) {
        return append_runs(b, a);
    }
    sweep_runs(&a, &b)
}

fn append_runs(mut left: Vec<AvailabilityRun>, right: Vec<AvailabilityRun>) -> Vec<AvailabilityRun> {
    let coalesce = match (left.last(), right.first()) {
        (Some(last), Some(first)) => {
            last.end_ns == first.start_ns && last.warm == first.warm && last.cold == first.cold
        }
        _ => false,
    };
    if coalesce {
        let end = right[0].end_ns;
        if let Some(last) = left.last_mut() {
            last.end_ns = end;
        }
        left.extend(right.into_iter().skip(1));
    } else {
        left.extend(right);
    }
    left
}

#[derive(Clone, Copy)]
enum SweepSide {
    A,
    B,
}

struct SweepEdge {
    at: u64,
    is_end: bool,
    warm: u32,
    cold: u32,
}

fn sweep_runs(a: &[AvailabilityRun], b: &[AvailabilityRun]) -> Vec<AvailabilityRun> {
    let mut out = Vec::new();
    let mut ai = 0usize;
    let mut bi = 0usize;
    let mut a_inside = false;
    let mut b_inside = false;
    let mut warm = 0u32;
    let mut cold = 0u32;
    let mut cursor = 0u64;
    let mut have_cursor = false;
    loop {
        let edge_a = sweep_edge(a, ai, a_inside);
        let edge_b = sweep_edge(b, bi, b_inside);
        let Some((side, edge)) = earlier_edge(edge_a, edge_b) else { break };
        if have_cursor && edge.at > cursor {
            push_run(&mut out, cursor, edge.at, warm, cold);
        }
        cursor = edge.at;
        have_cursor = true;
        if edge.is_end {
            warm = warm.saturating_sub(edge.warm);
            cold = cold.saturating_sub(edge.cold);
            match side {
                SweepSide::A => {
                    a_inside = false;
                    ai += 1;
                }
                SweepSide::B => {
                    b_inside = false;
                    bi += 1;
                }
            }
        } else {
            warm = warm.saturating_add(edge.warm);
            cold = cold.saturating_add(edge.cold);
            match side {
                SweepSide::A => a_inside = true,
                SweepSide::B => b_inside = true,
            }
        }
    }
    debug_assert_eq!((warm, cold), (0, 0));
    out
}

fn sweep_edge(runs: &[AvailabilityRun], idx: usize, inside: bool) -> Option<SweepEdge> {
    let run = runs.get(idx)?;
    if inside {
        Some(SweepEdge { at: run.end_ns, is_end: true, warm: run.warm, cold: run.cold })
    } else {
        Some(SweepEdge { at: run.start_ns, is_end: false, warm: run.warm, cold: run.cold })
    }
}

fn earlier_edge(a: Option<SweepEdge>, b: Option<SweepEdge>) -> Option<(SweepSide, SweepEdge)> {
    match (a, b) {
        (None, None) => None,
        (Some(edge), None) => Some((SweepSide::A, edge)),
        (None, Some(edge)) => Some((SweepSide::B, edge)),
        (Some(edge_a), Some(edge_b)) => {
            let a_first = edge_a.at < edge_b.at || (edge_a.at == edge_b.at && edge_a.is_end && !edge_b.is_end);
            if a_first { Some((SweepSide::A, edge_a)) } else { Some((SweepSide::B, edge_b)) }
        }
    }
}

fn time_key(stream: u32, start_ns: u64, seq: u32) -> [u8; 16] {
    let mut key = [0u8; 16];
    key[..4].copy_from_slice(&stream.to_be_bytes());
    key[4..12].copy_from_slice(&start_ns.to_be_bytes());
    key[12..].copy_from_slice(&seq.to_be_bytes());
    key
}

struct ChunkClass {
    cut_streams: Vec<bool>,
    /// `(stream, start_ns, seq)` held by a live cut chunk, mapped to that chunk's id.
    stolen: HashMap<(u32, u64, u32), [u8; 16]>,
}

/// Marks streams that have a cut chunk, and records time keys a live cut chunk currently owns.
///
/// Cut removal does not commute, so those streams keep raw copies. A key is stolen only when `TIME_INDEX` still
/// names the chunk: purging the chunk deletes that row and leaves the orphan payload for the id-list fallback.
fn classify_chunks(
    chk: &ReadTable,
    pay: &ReadTable,
    ti: &ReadTable,
    stream_lookup: &StreamLookup,
    stream_count: usize,
) -> Result<ChunkClass> {
    let mut cut_streams = vec![false; stream_count];
    let mut stolen = HashMap::new();
    for row in chk.range::<&[u8]>(b"".as_slice()..)? {
        let (key, value) = row?;
        let id_bytes: [u8; 16] = key.value().try_into().map_err(|_| corrupt("chunk key"))?;
        let frames = codec::read_frames(value.value())?;
        if !frames.iter().any(|frame| frame.cut) {
            continue;
        }
        let Some(payload) = pay.get(id_bytes.as_slice())? else {
            return Err(corrupt("cut chunk has no payload"));
        };
        let (_id, stream, span, seq) = fast_header_fields(payload.value())?;
        let Some(span) = span else { continue };
        let stream_idx = stream_lookup.index_of(stream)?;
        cut_streams[stream_idx] = true;
        let tkey = time_key(stream.0, span.start_ns(), seq);
        if let Some(holder) = ti.get(tkey.as_slice())? {
            let holder_bytes: [u8; 16] = holder.value().try_into().map_err(|_| corrupt("time_index value"))?;
            if holder_bytes == id_bytes {
                stolen.insert((stream.0, span.start_ns(), seq), id_bytes);
            }
        }
    }
    Ok(ChunkClass { cut_streams, stolen })
}

fn collect_ids_in_range(ti: &ReadTable, lo: [u8; 16], hi: [u8; 16]) -> Result<Vec<[u8; 16]>> {
    let mut ids = Vec::new();
    for row in ti.range::<&[u8]>(lo.as_slice()..=hi.as_slice())? {
        let (key, value) = row?;
        if key.value().len() != 16 {
            return Err(corrupt("time_index key"));
        }
        ids.push(value.value().try_into().map_err(|_| corrupt("time_index value"))?);
    }
    Ok(ids)
}

fn worker_count(shards: usize) -> usize {
    std::thread::available_parallelism().map_or(4, std::num::NonZeroUsize::get).min(MAX_SCAN_WORKERS).min(shards.max(1))
}

/// Parallel id collection for the residual fallback. Serial below `min_parallel`, matching the payload scan.
fn collect_time_indexed_ids_parallel(ti: &ReadTable, min_parallel: u64) -> Result<Vec<[u8; 16]>> {
    let len = ti.len()?;
    if len < min_parallel {
        return collect_time_indexed_ids(ti);
    }
    let (Some(first_row), Some(last_row)) = (ti.first()?, ti.last()?) else {
        return Ok(Vec::new());
    };
    let first_id = key_u128(first_row.0.value(), "time_index key")?;
    let last_id = key_u128(last_row.0.value(), "time_index key")?;
    drop((first_row, last_row));
    let mut shards = Vec::with_capacity(1 << SHARD_DEPTH);
    collect_shards(ti, first_id, last_id, SHARD_DEPTH, &mut shards)?;
    if shards.len() <= 1 {
        return collect_time_indexed_ids(ti);
    }
    let workers = worker_count(shards.len());
    let results: Vec<OnceLock<Vec<[u8; 16]>>> = (0..shards.len()).map(|_| OnceLock::new()).collect();
    let next_shard = AtomicUsize::new(0);
    let abort = AtomicBool::new(false);
    std::thread::scope(|scope| -> Result<()> {
        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            let shards_ref = &shards;
            let results_ref = &results;
            let next_ref = &next_shard;
            let abort_ref = &abort;
            handles.push(scope.spawn(move || -> Result<()> {
                while !abort_ref.load(AtomicOrdering::Relaxed) {
                    let idx = next_ref.fetch_add(1, AtomicOrdering::Relaxed);
                    let Some(&(lo, hi)) = shards_ref.get(idx) else { break };
                    match collect_ids_in_range(ti, lo, hi) {
                        Ok(ids) => {
                            let _ = results_ref[idx].set(ids);
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
        match first_err {
            Some(err) => Err(err),
            None => Ok(()),
        }
    })?;
    let mut ids = Vec::new();
    for slot in results {
        if let Some(part) = slot.into_inner() {
            ids.extend(part);
        }
    }
    ids.sort_unstable();
    debug_assert!(
        ids.windows(2).all(|w| w[0] < w[1]),
        "TIME_INDEX invariant violated: duplicate payload id referenced by multiple time keys",
    );
    Ok(ids)
}

struct FoldedShard {
    payloads: usize,
    indexed_rows: usize,
    stream_runs: Vec<Vec<AvailabilityRun>>,
    stream_copies: Vec<Vec<AvailabilityCopy>>,
    cut_chunks: Vec<AvailabilityCutChunk>,
    stale_tier0: Vec<Uuid>,
}

struct Accum {
    payloads: usize,
    indexed_rows: usize,
    stream_runs: Vec<Vec<AvailabilityRun>>,
    stream_copies: Vec<Vec<AvailabilityCopy>>,
    cut_chunks: Vec<AvailabilityCutChunk>,
    stale_tier0: Vec<Uuid>,
}

impl Accum {
    fn new(stream_count: usize) -> Self {
        Self {
            payloads: 0,
            indexed_rows: 0,
            stream_runs: vec![Vec::new(); stream_count],
            stream_copies: vec![Vec::new(); stream_count],
            cut_chunks: Vec::new(),
            stale_tier0: Vec::new(),
        }
    }

    fn absorb(&mut self, shard: FoldedShard) {
        self.payloads += shard.payloads;
        self.indexed_rows += shard.indexed_rows;
        for (dst, src) in self.stream_runs.iter_mut().zip(shard.stream_runs) {
            if src.is_empty() {
                continue;
            }
            if dst.is_empty() {
                *dst = src;
            } else {
                let taken = std::mem::take(dst);
                *dst = merge_runs(taken, src);
            }
        }
        for (dst, src) in self.stream_copies.iter_mut().zip(shard.stream_copies) {
            if !src.is_empty() {
                dst.extend(src);
            }
        }
        self.cut_chunks.extend(shard.cut_chunks);
        self.stale_tier0.extend(shard.stale_tier0);
    }

    fn finish(&mut self, cut_streams: &[bool]) {
        for (i, copies) in self.stream_copies.iter_mut().enumerate() {
            let cut = cut_streams.get(i).copied().unwrap_or(false);
            if cut {
                debug_assert!(self.stream_runs[i].is_empty(), "a cut stream is not folded");
                if copies.len() > 1 {
                    copies.sort_unstable_by(|left, right| {
                        (left.span.start_ns(), left.seq, left.span.end_ns(), left.warm, left.cold).cmp(&(
                            right.span.start_ns(),
                            right.seq,
                            right.span.end_ns(),
                            right.warm,
                            right.cold,
                        ))
                    });
                }
            } else {
                debug_assert!(copies.is_empty(), "a stream without cuts keeps runs, not raw copies");
            }
        }
        self.cut_chunks.sort_unstable_by_key(|chunk| chunk.id);
        if !self.stale_tier0.is_sorted() {
            self.stale_tier0.sort_unstable();
        }
    }
}

fn fold_shard(mut raw: ShardOut, cut_streams: &[bool], rates: &[f64]) -> FoldedShard {
    let mut stream_runs = vec![Vec::new(); raw.stream_copies.len()];
    for (i, copies) in raw.stream_copies.iter_mut().enumerate() {
        if cut_streams.get(i).copied().unwrap_or(false) || copies.is_empty() {
            continue;
        }
        let taken = std::mem::take(copies);
        let rate = rates.get(i).copied().unwrap_or(0.0);
        stream_runs[i] = fold_availability_copies(rate, taken);
    }
    FoldedShard {
        payloads: raw.payloads,
        indexed_rows: raw.indexed_rows,
        stream_runs,
        stream_copies: raw.stream_copies,
        cut_chunks: raw.cut_chunks,
        stale_tier0: raw.stale_tier0,
    }
}

struct ScanCtx<'a> {
    tables: ScanTables<'a>,
    cut_streams: &'a [bool],
    rates: &'a [f64],
    first_id: u128,
    last_id: u128,
    min_parallel: u64,
    pay_len: u64,
}

fn accum_from_shard(raw: ShardOut, cut_streams: &[bool], rates: &[f64]) -> Accum {
    let mut accum = Accum::new(raw.stream_copies.len());
    accum.absorb(fold_shard(raw, cut_streams, rates));
    accum.finish(cut_streams);
    accum
}

/// Peak raw-copy memory is one shard per worker (16 workers and 64 shards is about a quarter of the unfolded copy
/// vector) plus the merged runs and any out-of-order folded prefix still waiting. Raw copies are dropped inside
/// the worker for every stream that has no cut chunk.
fn scan_payloads(ctx: &ScanCtx<'_>) -> Result<Accum> {
    let lo = ctx.first_id.to_be_bytes();
    let hi = ctx.last_id.to_be_bytes();
    if ctx.pay_len < ctx.min_parallel {
        return Ok(accum_from_shard(scan_shard(&ctx.tables, lo, hi)?, ctx.cut_streams, ctx.rates));
    }
    let mut shards = Vec::with_capacity(1 << SHARD_DEPTH);
    collect_shards(ctx.tables.pay, ctx.first_id, ctx.last_id, SHARD_DEPTH, &mut shards)?;
    if shards.len() <= 1 {
        let (shard_lo, shard_hi) = shards.first().copied().unwrap_or((lo, hi));
        return Ok(accum_from_shard(scan_shard(&ctx.tables, shard_lo, shard_hi)?, ctx.cut_streams, ctx.rates));
    }
    let workers = worker_count(shards.len());
    let next_shard = AtomicUsize::new(0);
    let abort = AtomicBool::new(false);
    std::thread::scope(|scope| -> Result<Accum> {
        let (tx, rx) = sync_channel(workers);
        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            let tx = tx.clone();
            let shards_ref = &shards;
            let next_ref = &next_shard;
            let abort_ref = &abort;
            let tables_ref = &ctx.tables;
            let cut_ref = ctx.cut_streams;
            let rates_ref = ctx.rates;
            handles.push(scope.spawn(move || {
                while !abort_ref.load(AtomicOrdering::Relaxed) {
                    let idx = next_ref.fetch_add(1, AtomicOrdering::Relaxed);
                    let Some(&(shard_lo, shard_hi)) = shards_ref.get(idx) else { break };
                    match scan_shard(tables_ref, shard_lo, shard_hi) {
                        Ok(raw) => {
                            let folded = fold_shard(raw, cut_ref, rates_ref);
                            if tx.send(Ok((idx, folded))).is_err() {
                                return;
                            }
                        }
                        Err(err) => {
                            abort_ref.store(true, AtomicOrdering::Relaxed);
                            let _ = tx.send(Err(err));
                            return;
                        }
                    }
                }
            }));
        }
        drop(tx);
        let mut accum = Accum::new(ctx.tables.stream_count);
        let mut pending: BTreeMap<usize, FoldedShard> = BTreeMap::new();
        let mut next = 0usize;
        let mut first_err: Option<CatalogError> = None;
        while let Ok(msg) = rx.recv() {
            match msg {
                Ok((idx, shard)) if first_err.is_none() => {
                    pending.insert(idx, shard);
                    while let Some(ready) = pending.remove(&next) {
                        accum.absorb(ready);
                        next += 1;
                    }
                }
                Ok(_) => {}
                Err(err) => {
                    abort.store(true, AtomicOrdering::Relaxed);
                    if first_err.is_none() {
                        first_err = Some(err);
                    }
                }
            }
        }
        for handle in handles {
            if handle.join().is_err() && first_err.is_none() {
                first_err = Some(CatalogError::Storage("availability scan worker panicked".into()));
            }
        }
        if let Some(err) = first_err {
            return Err(err);
        }
        accum.finish(ctx.cut_streams);
        Ok(accum)
    })
}

fn debug_rows_match(rows: usize, ti_len: u64) {
    debug_assert_eq!(
        u64::try_from(rows).unwrap_or(u64::MAX),
        ti_len,
        "TIME_INDEX invariant violated: matched {rows} PAYLOADS rows for {ti_len} TIME_INDEX entries",
    );
}

pub(crate) fn snapshot(r: &R) -> Result<AvailabilitySnapshot> {
    snapshot_with_min_parallel(r, PARALLEL_MIN_PAYLOADS).map(|(snap, _)| snap)
}

pub(crate) fn snapshot_with_min_parallel(r: &R, min_parallel_payloads: u64) -> Result<(AvailabilitySnapshot, bool)> {
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

    // Invariant: every `TIME_INDEX` row maps to a distinct live `PAYLOADS` row, because inserts write at most one
    // `TIME_INDEX` key per payload (`redb_catalog.rs` `put`) and purges delete a `TIME_INDEX` key whenever it still
    // names the deleted payload (`redb_catalog.rs` `remove_one`). Therefore `TIME_INDEX` values form an injective
    // subset of `PAYLOADS` keys (`ti_len <= pay_len`), so `ti_len == pay_len` proves every `PAYLOADS` row is
    // time-indexed. A chunk that overwrote a worker-cut first frame's `(stream, start_ns, seq)` key leaves
    // `ti_len < pay_len`. When that chunk is still live and holds the key, the scan skips the orphan and keeps the
    // result if the counted rows equal `ti_len`. Any other gap collects `TIME_INDEX` ids and filters on them.
    debug_assert!(
        ti_len <= pay_len,
        "TIME_INDEX invariant violated: TIME_INDEX has more rows ({ti_len}) than PAYLOADS ({pay_len})",
    );

    let mut accum = Accum::new(streams.len());
    let mut used_fallback = false;

    if let (Some(first_row), Some(last_row)) = (pay.first()?, pay.last()?) {
        let first_id = key_u128(first_row.0.value(), "payload key")?;
        let last_id = key_u128(last_row.0.value(), "payload key")?;
        drop((first_row, last_row));

        let class = if has_chunks {
            classify_chunks(&chk, &pay, &ti, &stream_lookup, streams.len())?
        } else {
            ChunkClass { cut_streams: vec![false; streams.len()], stolen: HashMap::new() }
        };
        let rates: Vec<f64> = streams.iter().map(|(_, info)| info.rate_hz.unwrap_or(0.0)).collect();
        used_fallback = ti_len != pay_len;

        let run = |visibility: Visibility<'_>| -> Result<Accum> {
            scan_payloads(&ScanCtx {
                tables: ScanTables {
                    pay: &pay,
                    loc: &loc,
                    doom: has_doomed.then_some(&doom),
                    chk: has_chunks.then_some(&chk),
                    visibility,
                    stream_lookup: &stream_lookup,
                    stream_count: streams.len(),
                },
                cut_streams: &class.cut_streams,
                rates: &rates,
                first_id,
                last_id,
                min_parallel: min_parallel_payloads,
                pay_len,
            })
        };

        accum = if !used_fallback {
            run(Visibility::All)?
        } else if !class.stolen.is_empty() {
            let narrowed = run(Visibility::SkipOwned(&class.stolen))?;
            if u64::try_from(narrowed.indexed_rows).ok() == Some(ti_len) {
                narrowed
            } else {
                let ids = collect_time_indexed_ids_parallel(&ti, min_parallel_payloads)?;
                let present = run(Visibility::Present(ids.as_slice()))?;
                debug_rows_match(present.indexed_rows, ti_len);
                present
            }
        } else {
            let ids = collect_time_indexed_ids_parallel(&ti, min_parallel_payloads)?;
            let present = run(Visibility::Present(ids.as_slice()))?;
            debug_rows_match(present.indexed_rows, ti_len);
            present
        };
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

    Ok((
        AvailabilitySnapshot {
            streams,
            payloads: accum.payloads,
            stream_copies: accum.stream_copies,
            stream_runs: accum.stream_runs,
            cut_chunks: accum.cut_chunks,
            stale_tier0: accum.stale_tier0,
            coverage,
            stream_reads,
            payload_reads,
            coverage_reads,
        },
        used_fallback,
    ))
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

    #[test]
    fn snapshot_rejects_payload_on_unregistered_stream() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("unregistered_stream.redb");
        drop(crate::RedbCatalog::open(&path).unwrap());

        let db = Database::open(&path).unwrap();
        let id = Uuid([0x22; 16]);
        let hdr = HeaderBuf::build(&HeaderSpec::raw(id, StreamId(99), 10, 20)).unwrap();
        let mut ti_key = [0u8; 16];
        ti_key[0..4].copy_from_slice(&99u32.to_be_bytes());
        ti_key[4..12].copy_from_slice(&10u64.to_be_bytes());
        let w = db.begin_write().unwrap();
        {
            let mut pay = w.open_table(PAYLOADS).unwrap();
            let mut ti = w.open_table(TIME_INDEX).unwrap();
            pay.insert(id.0.as_slice(), hdr.as_bytes()).unwrap();
            ti.insert(ti_key.as_slice(), id.0.as_slice()).unwrap();
        }
        w.commit().unwrap();
        drop(db);

        let cat = crate::RedbCatalog::open(&path).unwrap();
        let err = crate::Catalog::availability_snapshot(&cat).unwrap_err();
        assert!(err.to_string().contains("unregistered stream"), "unexpected error: {err}");
    }

    #[test]
    fn snapshot_fallback_matches_across_single_and_sharded_thresholds() {
        use crate::{Catalog, ChunkFrame, NewChunk, NewPayload, RedbCatalog, StreamKind};
        use continuum_core::{DomainLayer, PayloadFormat, StorageLocator};

        let dir = tempfile::tempdir().unwrap();
        let cat = RedbCatalog::open(dir.path().join("fallback_sharded.redb")).unwrap();
        let s = cat
            .register_stream(StreamInfo {
                name: "cam".into(),
                kind: StreamKind::Sensor,
                layer: DomainLayer::Layer1Raw,
                format: PayloadFormat::RawBytes,
                shape_name: "raw".into(),
                shape_version: 1,
                rate_hz: None,
                anchor: None,
            })
            .unwrap();
        for i in 1..=12u32 {
            let mut raw = [0u8; 16];
            raw[0] = u8::try_from(i).unwrap();
            raw[15] = u8::try_from(i).unwrap();
            let spec = HeaderSpec::raw(Uuid(raw), s, u64::from(i) * 10, u64::from(i) * 10 + 5);
            cat.insert(vec![NewPayload {
                header: HeaderBuf::build(&spec).unwrap(),
                location: Some(StorageLocator::Chunk {
                    tier: StorageTier::Tier1Warm,
                    uri: "warm".into(),
                    offset: u64::from(i) * 64,
                    length: 64,
                }),
                coverage: None,
            }])
            .unwrap();
        }
        // Overwrite the first frame's `(stream, start_ns = 10, seq = 0)` TIME_INDEX key with a worker-cut chunk.
        let mut first_id = [0u8; 16];
        first_id[0] = 1;
        first_id[15] = 1;
        let chunk_id = Uuid([0x80; 16]);
        let chunk_spec = HeaderSpec::raw(chunk_id, s, 10, 25);
        cat.commit_chunk(
            NewChunk {
                payload: NewPayload {
                    header: HeaderBuf::build(&chunk_spec).unwrap(),
                    location: Some(StorageLocator::Chunk {
                        tier: StorageTier::Tier2Cold,
                        uri: "cold".into(),
                        offset: 0,
                        length: 64,
                    }),
                    coverage: None,
                },
                frames: vec![ChunkFrame { start_ns: 10, offset: 0, length: 0, cut: true }],
            },
            &[Uuid(first_id)],
        )
        .unwrap();

        let (single, single_fallback) = cat.availability_snapshot_for_test(u64::MAX).unwrap();
        let (sharded, sharded_fallback) = cat.availability_snapshot_for_test(1).unwrap();
        assert!(single_fallback && sharded_fallback);
        assert_eq!(single.payloads, 12);
        assert_eq!(sharded.payloads, 12);
        assert_eq!(single.stream_copies, sharded.stream_copies);
        assert_eq!(single.stream_runs, sharded.stream_runs);
        assert_eq!(single.cut_chunks, sharded.cut_chunks);
        assert_eq!(single.stale_tier0, sharded.stale_tier0);
    }

    #[test]
    fn fold_splits_overlaps_and_merges_equal_neighbors() {
        let copies = vec![
            AvailabilityCopy { span: Interval::new(0, 10).unwrap(), seq: 0, warm: 1, cold: 0 },
            AvailabilityCopy { span: Interval::new(10, 20).unwrap(), seq: 0, warm: 1, cold: 0 },
            AvailabilityCopy { span: Interval::new(5, 15).unwrap(), seq: 1, warm: 0, cold: 1 },
        ];
        assert_eq!(
            fold_availability_copies(0.0, copies),
            vec![
                AvailabilityRun { start_ns: 0, end_ns: 5, warm: 1, cold: 0 },
                AvailabilityRun { start_ns: 5, end_ns: 15, warm: 1, cold: 1 },
                AvailabilityRun { start_ns: 15, end_ns: 20, warm: 1, cold: 0 },
            ],
        );
    }

    #[test]
    fn fold_expands_rated_instants_before_merging() {
        let copies = vec![
            AvailabilityCopy { span: Interval::covering(10, 10).unwrap(), seq: 0, warm: 1, cold: 0 },
            AvailabilityCopy { span: Interval::covering(20, 20).unwrap(), seq: 0, warm: 1, cold: 0 },
        ];
        // 1e8 Hz is a 10 ns period, so the instants become [10, 20) and [20, 30) and then one run.
        assert_eq!(
            fold_availability_copies(1e8, copies),
            vec![AvailabilityRun { start_ns: 10, end_ns: 30, warm: 1, cold: 0 }],
        );
    }

    #[test]
    fn non_cut_runs_match_across_single_and_sharded_thresholds() {
        use crate::{Catalog, NewPayload, RedbCatalog, StreamKind};
        use continuum_core::{DomainLayer, PayloadFormat, StorageLocator};

        let dir = tempfile::tempdir().unwrap();
        let cat = RedbCatalog::open(dir.path().join("runs_sharded.redb")).unwrap();
        let rated = cat
            .register_stream(StreamInfo {
                name: "rated".into(),
                kind: StreamKind::Sensor,
                layer: DomainLayer::Layer1Raw,
                format: PayloadFormat::RawBytes,
                shape_name: "raw".into(),
                shape_version: 1,
                rate_hz: Some(1e8),
                anchor: None,
            })
            .unwrap();
        let plain = cat
            .register_stream(StreamInfo {
                name: "plain".into(),
                kind: StreamKind::Sensor,
                layer: DomainLayer::Layer1Raw,
                format: PayloadFormat::RawBytes,
                shape_name: "raw".into(),
                shape_version: 1,
                rate_hz: None,
                anchor: None,
            })
            .unwrap();
        let mut batch = Vec::new();
        for i in 0..24u32 {
            let mut rated_id = [0u8; 16];
            rated_id[0] = 1;
            rated_id[12..].copy_from_slice(&i.to_be_bytes());
            let mut plain_id = [0u8; 16];
            plain_id[0] = 2;
            plain_id[12..].copy_from_slice(&i.to_be_bytes());
            let start = u64::from(i) * 10 + 10;
            batch.push(NewPayload {
                header: HeaderBuf::build(&HeaderSpec::raw(Uuid(rated_id), rated, start, start)).unwrap(),
                location: Some(StorageLocator::Chunk {
                    tier: StorageTier::Tier1Warm,
                    uri: "warm".into(),
                    offset: u64::from(i) * 64,
                    length: 64,
                }),
                coverage: None,
            });
            batch.push(NewPayload {
                header: HeaderBuf::build(&HeaderSpec::raw(Uuid(plain_id), plain, start, start + 4)).unwrap(),
                location: Some(StorageLocator::Chunk {
                    tier: StorageTier::Tier2Cold,
                    uri: "cold".into(),
                    offset: u64::from(i) * 64,
                    length: 64,
                }),
                coverage: None,
            });
        }
        cat.insert(batch).unwrap();

        let (single, single_fallback) = cat.availability_snapshot_for_test(u64::MAX).unwrap();
        let (sharded, sharded_fallback) = cat.availability_snapshot_for_test(1).unwrap();
        assert!(!single_fallback && !sharded_fallback);
        assert_eq!(single.payloads, 48);
        assert!(single.stream_copies.iter().all(|copies| copies.is_empty()));
        assert!(sharded.stream_copies.iter().all(|copies| copies.is_empty()));
        assert_eq!(single.stream_runs, sharded.stream_runs);
        assert_eq!(single.cut_chunks, sharded.cut_chunks);
        let rated_idx = single.streams.iter().position(|(id, _)| *id == rated).unwrap();
        assert!(
            single.stream_runs[rated_idx].len() < 24,
            "rated instants should merge, got {:?}",
            single.stream_runs[rated_idx]
        );
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

    fn arb_copy() -> impl Strategy<Value = AvailabilityCopy> {
        (0u64..400, 1u64..30, 0u8..=2, 0u8..=2).prop_map(|(start, len, warm, cold)| AvailabilityCopy {
            span: Interval::new(start, start + len).expect("len is at least 1"),
            seq: 0,
            warm,
            cold,
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn merged_partials_match_folding_every_copy(
            rate in prop_oneof![Just(0.0f64), Just(1e8), Just(30.0)],
            a in proptest::collection::vec(arb_copy(), 0..8),
            b in proptest::collection::vec(arb_copy(), 0..8),
            c in proptest::collection::vec(arb_copy(), 0..8),
        ) {
            let folded_a = fold_availability_copies(rate, a.clone());
            let folded_b = fold_availability_copies(rate, b.clone());
            let folded_c = fold_availability_copies(rate, c.clone());
            let mut all = a;
            all.extend(b);
            all.extend(c);
            let want = fold_availability_copies(rate, all);
            prop_assert_eq!(merge_runs(merge_runs(folded_a.clone(), folded_b.clone()), folded_c.clone()), want.clone());
            prop_assert_eq!(merge_runs(folded_c, merge_runs(folded_b, folded_a)), want);
        }
    }
}
