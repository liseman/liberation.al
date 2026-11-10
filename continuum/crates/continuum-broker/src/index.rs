//! The broker's in-memory availability index: which parts of each stream exist and in which tiers, and which parts
//! of each source stream each stage has processed.
//!
//! The scheduler's thread asks these questions on every query and replan, and it must never wait on disk, so they
//! are answered from memory, not from the catalog (docs/CONTINUUM-BROKER.md, the no-stall rule). The index follows
//! the catalog: the broker applies each change after the catalog has committed it (an ingest, a tier move, a chunk
//! rewrite, a purge step), and loads it from the catalog at startup before accepting connections.
//!
//! Each stream keeps a [`CopySet`]: time segments with a count of copies per tier. Counting (rather than a plain set
//! of covered time) is what lets a purge remove one payload's span without removing another payload's overlap, and
//! lets a copy leave a tier while another tier still holds it. Neighboring segments with equal counts merge, so a
//! stream written contiguously stays one segment per run however many payloads it holds.

mod restart_projection;
pub use restart_projection::{MAX_RESTART_PROJECTION_BYTES, RestartProjectionError};

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashMap};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use continuum_catalog::{
    AvailabilityCopy, AvailabilityCutChunk, AvailabilitySnapshot, Catalog, CatalogError, CoverageSet, PurgeDelta,
    RangeCursor, StreamKind as CatalogKind, frame_slots,
};
use continuum_core::{Interval, StageId, StorageTier, StreamId, Uuid};
use continuum_sched::{Covered, StreamKind};

/// The number of tiers (0 hot, 1 warm, 2 cold). A tier value above the last counts as the last: the schema has
/// only these three, so a larger value can only come from a newer schema or a bad row, and treating it as the
/// slowest tier is the safe reading ([`AvailabilityIndex::add_copy`] logs it).
pub const TIERS: usize = 3;

fn tier_index(tier: StorageTier) -> usize {
    usize::from(tier.0).min(TIERS - 1)
}

fn known_tier(tier: StorageTier) -> bool {
    usize::from(tier.0) < TIERS
}

/// A segment's copy count per tier.
type Counts = [u32; TIERS];

/// Time segments, each with how many copies of it each tier holds. Segments are disjoint, never all-zero, and
/// neighbors that touch never have equal counts (they would be one segment).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CopySet {
    /// start -> (end, counts)
    segs: BTreeMap<u64, (u64, Counts)>,
}

/// A removal asked for more copies than the set held over some of the span: the index and the catalog disagree.
/// The counts are left at zero there, and the caller should log it (a reload from the catalog repairs the index).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Underflow;

/// A change named a stream the index doesn't know: the index and the catalog disagree (the catalog refuses payloads
/// on unregistered streams, so this can't follow a commit). The caller should log it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnknownStream(pub StreamId);

impl CopySet {
    /// An empty set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Every segment as (start, end, copies per tier), in order.
    #[must_use]
    pub fn segments(&self) -> Vec<Segment> {
        self.segs.iter().map(|(&s, &(e, c))| (s, e, c)).collect()
    }

    /// How many segments it holds (what it costs in memory).
    #[must_use]
    pub fn len(&self) -> usize {
        self.segs.len()
    }

    /// Whether it holds nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.segs.is_empty()
    }

    /// Makes `t` a segment boundary, splitting the segment that contains it.
    fn split_at(&mut self, t: u64) {
        let Some((&start, &(end, counts))) = self.segs.range(..t).next_back() else { return };
        if start < t && t < end {
            self.segs.insert(start, (t, counts));
            self.segs.insert(t, (end, counts));
        }
    }

    /// Merges equal-count neighbors and drops all-zero segments from `from` to `to` (and the segments touching them).
    fn normalize(&mut self, from: u64, to: u64) {
        let lo = self.segs.range(..from).next_back().map_or(from, |(&s, _)| s);
        let keys: Vec<u64> = self.segs.range(lo..=to).map(|(&s, _)| s).collect();
        let mut prev: Option<u64> = None;
        for start in keys {
            let Some(&(end, counts)) = self.segs.get(&start) else { continue };
            if counts == [0; TIERS] {
                self.segs.remove(&start);
                prev = None;
                continue;
            }
            if let Some(p) = prev
                && let Some(&(pend, pcounts)) = self.segs.get(&p)
                && pend == start
                && pcounts == counts
            {
                self.segs.remove(&start);
                self.segs.insert(p, (end, counts));
                continue;
            }
            prev = Some(start);
        }
    }

    /// Adds one copy of `span` in `tier`.
    pub fn add(&mut self, span: Interval, tier: StorageTier) {
        let (a, b, t) = (span.start_ns(), span.end_ns(), tier_index(tier));
        // Startup visits each stream in time order. A disjoint append cannot change earlier segments;
        // avoid the general split/normalize scans and their temporary vectors in this common case.
        let mut single = [0; TIERS];
        single[t] = 1;
        match self.segs.last_key_value().map(|(&start, &(end, counts))| (start, end, counts)) {
            None => {
                self.segs.insert(a, (b, single));
                return;
            }
            Some((start, end, counts)) if end <= a => {
                if end == a && counts == single {
                    self.segs.get_mut(&start).expect("the last segment is still present").0 = b;
                } else {
                    self.segs.insert(a, (b, single));
                }
                return;
            }
            _ => {}
        }
        self.split_at(a);
        self.split_at(b);
        let mut cursor = a;
        let inside: Vec<(u64, u64)> = self.segs.range(a..b).map(|(&s, &(e, _))| (s, e)).collect();
        for (s, e) in inside {
            if cursor < s {
                let mut counts = [0; TIERS];
                counts[t] = 1;
                self.segs.insert(cursor, (s, counts));
            }
            if let Some(seg) = self.segs.get_mut(&s) {
                seg.1[t] = seg.1[t].saturating_add(1);
            }
            cursor = e;
        }
        if cursor < b {
            let mut counts = [0; TIERS];
            counts[t] = 1;
            self.segs.insert(cursor, (b, counts));
        }
        self.normalize(a, b);
    }

    /// Removes one copy of `span` in `tier`.
    ///
    /// # Errors
    ///
    /// [`Underflow`] when part of the span held no copy in that tier (see [`Underflow`]); the rest is removed.
    pub fn remove(&mut self, span: Interval, tier: StorageTier) -> Result<(), Underflow> {
        let (a, b, t) = (span.start_ns(), span.end_ns(), tier_index(tier));
        self.split_at(a);
        self.split_at(b);
        let mut ok = true;
        let mut cursor = a;
        let inside: Vec<u64> = self.segs.range(a..b).map(|(&s, _)| s).collect();
        for s in inside {
            if cursor < s {
                ok = false;
            }
            if let Some(seg) = self.segs.get_mut(&s) {
                if seg.1[t] == 0 {
                    ok = false;
                } else {
                    seg.1[t] -= 1;
                }
                cursor = seg.0;
            }
        }
        if cursor < b {
            ok = false;
        }
        self.normalize(a, b);
        if ok { Ok(()) } else { Err(Underflow) }
    }

    /// The parts of `range` that exist, each with the hottest tier holding a copy, in time order, pairwise
    /// disjoint, with touching parts of the same tier merged.
    #[must_use]
    pub fn covered(&self, range: Interval) -> Vec<Covered> {
        let (a, b) = (range.start_ns(), range.end_ns());
        let first = self.segs.range(..=a).next_back().map_or(a, |(&s, _)| s);
        let mut out: Vec<Covered> = Vec::new();
        for (&s, &(e, counts)) in self.segs.range(first..b) {
            let Some(interval) = Interval::new(s.max(a), e.min(b)) else { continue };
            let Some(t) = counts.iter().position(|&n| n > 0) else { continue };
            let tier = StorageTier(u8::try_from(t).expect("fewer than 256 tiers"));
            match out.last_mut() {
                Some(last) if last.tier == tier && last.interval.end_ns() == interval.start_ns() => {
                    last.interval =
                        Interval::new(last.interval.start_ns(), interval.end_ns()).expect("extends a non-empty one");
                }
                _ => out.push(Covered { interval, tier }),
            }
        }
        out
    }
}

/// What the index keeps about a stream besides its copies.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StreamMeta {
    /// Sensor or derived.
    pub kind: StreamKind,
    /// Nominal items per second, 0 when it has none.
    pub rate_hz: f64,
}

/// What a copy over `span` holds of its stream's time. A camera frame is an instant (capture stamps it with one
/// time), yet it stands for its stream until the next frame: an instant on a stream with a nominal rate covers one
/// period, so a range of frames reads as captured and a decode over it can be planned. Anything with a
/// duration, or a stream with no rate, covers its own span.
fn frame_span(meta: &StreamMeta, span: Interval) -> Interval {
    if span.len_ns() > 1 || !(meta.rate_hz > 0.0 && meta.rate_hz.is_finite()) {
        return span;
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let period = (1e9 / meta.rate_hz).round().max(1.0) as u64;
    Interval::new(span.start_ns(), span.start_ns().saturating_add(period)).unwrap_or(span)
}

/// The availability index (see the module docs).
#[derive(Debug, Default)]
pub struct AvailabilityIndex {
    streams: HashMap<StreamId, (StreamMeta, CopySet)>,
    coverage: HashMap<(StreamId, StageId), CoverageSet>,
    /// For each chunk that has lost frames to a purge, the slots it lost. A later purge that dooms the whole chunk
    /// reports its full span; the index removes only what it still holds (the span minus these), so a chunk cut by
    /// one purge and doomed by another isn't mistaken for drift.
    cuts: HashMap<Uuid, Vec<Interval>>,
    /// Payloads the catalog hid as they landed (their parent was being purged), which the index therefore never
    /// added. A later purge step still names them; the index skips them then.
    hidden_on_landing: std::collections::HashSet<Uuid>,
    /// Streams by name, so the scheduler task can resolve a worker's capability without asking the catalog.
    names: HashMap<String, StreamId>,
    /// Existing copy spans use the metadata active when they landed. A rate/kind change cannot certify
    /// those counts as a fresh catalog load; only rebuilding clears this sticky restart refusal.
    restart_metadata_changed: bool,
}

/// `span` without the parts in `holes`, in order.
fn subtract(span: Interval, holes: &[Interval]) -> Vec<Interval> {
    let mut holes: Vec<Interval> = holes.iter().copied().filter(|h| h.overlaps(span)).collect();
    holes.sort();
    let mut out = Vec::new();
    let mut cursor = span.start_ns();
    for h in holes {
        if let Some(piece) = Interval::new(cursor, h.start_ns().min(span.end_ns())) {
            out.push(piece);
        }
        cursor = cursor.max(h.end_ns());
    }
    if let Some(piece) = Interval::new(cursor, span.end_ns()) {
        out.push(piece);
    }
    out
}

impl AvailabilityIndex {
    /// An empty index.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a stream (or updates its kind and rate), keeping any copies it already has.
    pub fn register(&mut self, stream: StreamId, meta: StreamMeta) {
        if let Some((previous, _)) = self.streams.get(&stream)
            && (previous.kind != meta.kind || previous.rate_hz.to_bits() != meta.rate_hz.to_bits())
        {
            self.restart_metadata_changed = true;
        }
        self.streams.entry(stream).and_modify(|(m, _)| *m = meta).or_insert_with(|| (meta, CopySet::new()));
    }

    /// Records a stream's name.
    pub fn name_stream(&mut self, name: String, stream: StreamId) {
        self.names.insert(name, stream);
    }

    /// Every stream the index knows, in id order.
    #[must_use]
    pub fn streams(&self) -> Vec<StreamId> {
        let mut ids: Vec<StreamId> = self.streams.keys().copied().collect();
        ids.sort_by_key(|s| s.0);
        ids
    }

    /// The stream with this name, if the index knows one.
    #[must_use]
    pub fn stream_named(&self, name: &str) -> Option<StreamId> {
        self.names.get(name).copied()
    }

    /// A stream's registered name, or `None` for one the index doesn't know.
    #[must_use]
    pub fn name_of(&self, stream: StreamId) -> Option<String> {
        self.names.iter().find_map(|(n, &s)| (s == stream).then(|| n.clone()))
    }

    /// A stream's kind and rate, or `None` for one the index doesn't know.
    #[must_use]
    pub fn meta(&self, stream: StreamId) -> Option<StreamMeta> {
        self.streams.get(&stream).map(|(m, _)| *m)
    }

    /// One copy of `span` landed in `tier` on `stream` (an ingest, a flush, a hydrate). A tier past the last is
    /// logged and counted as the last (see [`TIERS`]).
    ///
    /// # Errors
    ///
    /// [`UnknownStream`] when the index doesn't know the stream; nothing is added.
    pub fn add_copy(&mut self, stream: StreamId, span: Interval, tier: StorageTier) -> Result<(), UnknownStream> {
        if !known_tier(tier) {
            tracing::warn!(stream = stream.0, tier = tier.0, "a copy in an unknown tier counts as the coldest");
        }
        let (meta, copies) = self.streams.get_mut(&stream).ok_or(UnknownStream(stream))?;
        copies.add(frame_span(meta, span), tier);
        Ok(())
    }

    /// Adds a chunk's copy in `tier` over `span`, less the `cut` slots (frames the chunk landed without), which are
    /// recorded as its cuts.
    ///
    /// # Errors
    /// [`UnknownStream`] when `stream` isn't registered.
    pub fn add_chunk_copy(
        &mut self,
        chunk: Uuid,
        stream: StreamId,
        span: Interval,
        tier: StorageTier,
        cut: &[Interval],
    ) -> Result<(), UnknownStream> {
        self.add_copy(stream, span, tier)?;
        for &slot in cut {
            self.record_cut(chunk, slot);
            let _ = self.remove_copy(stream, slot, tier);
        }
        Ok(())
    }

    /// One copy of `span` left `tier` on `stream` (a slot reclaimed after its flush, a purge, a rewrite).
    ///
    /// # Errors
    ///
    /// [`Underflow`] when the index held no such copy over part of the span.
    pub fn remove_copy(&mut self, stream: StreamId, span: Interval, tier: StorageTier) -> Result<(), Underflow> {
        match self.streams.get_mut(&stream) {
            Some((meta, copies)) => copies.remove(frame_span(meta, span), tier),
            None => Err(Underflow),
        }
    }

    /// The parts of `range` on `stream` that exist (see [`CopySet::covered`]); nothing for an unknown stream.
    #[must_use]
    pub fn covered(&self, stream: StreamId, range: Interval) -> Vec<Covered> {
        self.streams.get(&stream).map_or_else(Vec::new, |(_, copies)| copies.covered(range))
    }

    /// `stage` finished `span` of `source`.
    pub fn add_coverage(&mut self, source: StreamId, stage: StageId, span: Interval) {
        self.coverage.entry((source, stage)).or_default().insert(span);
    }

    /// `span` of `source` no longer counts as done by `stage` (a purge took the data).
    pub fn remove_coverage(&mut self, source: StreamId, stage: StageId, span: Interval) {
        if let Some(set) = self.coverage.get_mut(&(source, stage)) {
            set.remove(span);
            if set.is_empty() {
                self.coverage.remove(&(source, stage));
            }
        }
    }

    /// Within `window`, what `stage` hasn't processed of `source`, the first `max` parts.
    #[must_use]
    pub fn coverage_gaps(&self, source: StreamId, stage: StageId, window: Interval, max: usize) -> Vec<Interval> {
        match self.coverage.get(&(source, stage)) {
            Some(set) => set.gaps(window, max),
            None => std::iter::once(window).take(max).collect(),
        }
    }

    /// How many segments and coverage intervals the index holds, for the startup log. It walks every coverage set,
    /// so don't call it on a hot path.
    #[must_use]
    pub fn size(&self) -> (usize, usize) {
        let segs = self.streams.values().map(|(_, c)| c.len()).sum();
        let cov = self.coverage.values().map(|s| s.iter().count()).sum();
        (segs, cov)
    }
}

impl AvailabilityIndex {
    /// Follows one purge step: what it doomed leaves the index (each copy in each tier it was in), cut frames leave
    /// their chunks' tiers over their slots, and uncovered spans leave the stages' coverage. Call it as soon as the
    /// step has committed, so data being forgotten is never quoted as available. Returns how many entries didn't
    /// match the index (each one is drift from the catalog, to log).
    pub fn apply_purge_delta(&mut self, delta: &PurgeDelta) -> usize {
        let mut mismatches = 0;
        // Cuts first, so a chunk cut and doomed in one delta loses each part once.
        for cut in &delta.cut_frames {
            self.record_cut(cut.chunk, cut.slot);
            for &tier in &cut.tiers {
                if self.remove_copy(cut.stream, cut.slot, tier).is_err() {
                    mismatches += 1;
                }
            }
        }
        for d in &delta.doomed {
            if self.hidden_on_landing.remove(&d.id) {
                continue;
            }
            let pieces = match self.cuts.remove(&d.id) {
                Some(cut) => subtract(d.span, &cut),
                None => vec![d.span],
            };
            for &tier in &d.tiers {
                for &piece in &pieces {
                    if self.remove_copy(d.stream, piece, tier).is_err() {
                        mismatches += 1;
                    }
                }
            }
        }
        for &(source, stage, span) in &delta.uncovered {
            self.remove_coverage(source, stage, span);
        }
        mismatches
    }

    /// Notes a payload the catalog hid as it landed, so the purge step that names it later is skipped for it.
    pub fn note_hidden_on_landing(&mut self, id: Uuid) {
        self.hidden_on_landing.insert(id);
    }

    /// Notes that `chunk` lost the frame in `slot`.
    fn record_cut(&mut self, chunk: Uuid, slot: Interval) {
        let cuts = self.cuts.entry(chunk).or_default();
        if !cuts.contains(&slot) {
            cuts.push(slot);
        }
    }

    /// Everything the index holds, in a fixed order, for comparing two indexes in tests and checks: every stream's
    /// segments with their copy counts per tier (not only the hottest tier, so a leftover copy hidden behind another
    /// one still shows), every stage's coverage, and every chunk's cut slots.
    #[must_use]
    pub fn snapshot(&self) -> IndexSnapshot {
        let mut streams: Vec<_> = self
            .streams
            .iter()
            .map(|(&s, (meta, copies))| {
                (s.0, meta.kind == StreamKind::Sensor, meta.rate_hz.to_bits(), copies.segments())
            })
            .collect();
        streams.sort_by_key(|s| s.0);
        let mut coverage: Vec<_> =
            self.coverage.iter().map(|(&(s, g), set)| ((s.0, g.0), set.iter().collect::<Vec<_>>())).collect();
        coverage.sort_by_key(|c| c.0);
        let mut cuts: Vec<_> = self
            .cuts
            .iter()
            .map(|(id, slots)| {
                let mut slots = slots.clone();
                slots.sort();
                (id.0, slots)
            })
            .collect();
        cuts.sort();
        IndexSnapshot { streams, coverage, cuts }
    }
}

/// One segment of a [`CopySet`]: start, end, and copies per tier.
pub type Segment = (u64, u64, [u32; TIERS]);

/// A comparable copy of an index's contents (see [`AvailabilityIndex::snapshot`]).
#[derive(Clone, Debug, PartialEq)]
pub struct IndexSnapshot {
    /// Per stream: id, whether it's a sensor, its rate's bits, and its segments (start, end, copies per tier).
    pub streams: Vec<(u32, bool, u64, Vec<Segment>)>,
    /// Per (source, stage): the covered intervals.
    pub coverage: Vec<((u32, u32), Vec<Interval>)>,
    /// Per chunk with cut frames: its cut slots.
    pub cuts: Vec<([u8; 16], Vec<Interval>)>,
}

/// Clears the Tier 0 copies a load found ([`LoadStats::stale_tier0`]) off the catalog, a thousand to a transaction:
/// their arena is gone, so the rows name bytes nobody can read, a query would drop their items rather than say what's
/// left, and a purge's delta would name copies the index doesn't hold. Run it before the core starts, so no ingest or
/// purge can interleave. A payload gone since or being purged is skipped. Returns how many were cleared.
///
/// # Errors
/// A storage error; the batches before it stay cleared.
pub fn clear_stale_tier0(catalog: &dyn Catalog, ids: &[Uuid]) -> Result<usize, CatalogError> {
    let mut cleared = 0;
    for batch in ids.chunks(CLEAR_BATCH) {
        cleared += catalog.clear_locations(batch, StorageTier::Tier0Hot)?;
    }
    Ok(cleared)
}

/// Payloads [`clear_stale_tier0`] clears per transaction.
const CLEAR_BATCH: usize = 1000;

/// What a load read, for the startup log.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LoadStats {
    /// Streams registered.
    pub streams: usize,
    /// Payloads read.
    pub payloads: usize,
    /// Segments the index holds after the load.
    pub segments: usize,
    /// Coverage intervals the index holds after the load.
    pub coverage_intervals: usize,
    /// Payloads whose catalog rows still name a Tier 0 copy: an earlier run's arena, gone now (every run makes a new
    /// arena generation), so the load leaves those copies out and the caller clears the rows.
    pub stale_tier0: Vec<Uuid>,
}

/// Non-overlapping startup-load phases. Kept separate from deterministic [`LoadStats`].
#[derive(Clone, Copy, Debug, Default)]
pub struct LoadTimings {
    /// Stream metadata pages.
    pub stream_reads: Duration,
    /// Payload catalog pages, including reconstruction of their locations and chunk metadata.
    pub payload_reads: Duration,
    /// Applying payload copies and cuts to the in-memory index.
    pub payload_apply: Duration,
    /// Reading and applying stage coverage.
    pub coverage_load: Duration,
    /// Entire successful load, including final counts and bookkeeping.
    pub total: Duration,
}

/// Builds the index from the catalog: from one parallel snapshot scan when the catalog supports
/// [`Catalog::availability_snapshot`], or a page at a time otherwise. The broker runs it once at startup, before it
/// accepts a connection, so the scheduler never sees a partial index.
///
/// # Errors
///
/// The first catalog error. A partial index is never returned.
pub fn load(catalog: &dyn Catalog, page: usize) -> Result<(AvailabilityIndex, LoadStats), CatalogError> {
    load_profiled(catalog, page).map(|(index, stats, _)| (index, stats))
}

/// Builds the same complete index as [`load`], with phase timing for startup diagnosis.
///
/// # Errors
/// As [`load`]; a partial index is never returned.
pub fn load_profiled(
    catalog: &dyn Catalog,
    page: usize,
) -> Result<(AvailabilityIndex, LoadStats, LoadTimings), CatalogError> {
    let load_started = Instant::now();
    match catalog.availability_snapshot()? {
        Some(snap) => Ok(load_from_snapshot(snap, load_started, PARALLEL_APPLY_MIN_COPIES)),
        None => load_paged_profiled_from(catalog, page, load_started),
    }
}

/// Builds the index through the paged [`Catalog::streams`], [`Catalog::range`], and [`Catalog::coverage`] methods,
/// bypassing [`Catalog::availability_snapshot`]. Exposed as a reference full rebuild for parity tests.
///
/// # Errors
/// As [`load`]; a partial index is never returned.
pub fn load_paged(catalog: &dyn Catalog, page: usize) -> Result<(AvailabilityIndex, LoadStats), CatalogError> {
    load_paged_profiled(catalog, page).map(|(index, stats, _)| (index, stats))
}

/// Reference paged full rebuild with phase timings (see [`load_paged`]).
///
/// # Errors
/// As [`load`]; a partial index is never returned.
pub fn load_paged_profiled(
    catalog: &dyn Catalog,
    page: usize,
) -> Result<(AvailabilityIndex, LoadStats, LoadTimings), CatalogError> {
    load_paged_profiled_from(catalog, page, Instant::now())
}

const PARALLEL_APPLY_MIN_COPIES: usize = 4_096;
const MAX_APPLY_WORKERS: usize = 16;

/// Builds a normalized [`CopySet`] from durable Tier 1 and Tier 2 copies on one stream in a single sweep, packing
/// sorted segments directly into its `BTreeMap` instead of splitting and normalizing per payload.
fn build_copy_set(meta: &StreamMeta, mut copies: Vec<AvailabilityCopy>) -> CopySet {
    if copies.is_empty() {
        return CopySet::new();
    }
    if !copies.is_sorted_by_key(|c| c.span.start_ns()) {
        copies.sort_unstable_by_key(|c| c.span.start_ns());
    }
    let mut runs: Vec<(u64, (u64, Counts))> = Vec::with_capacity(copies.len());
    let mut active: BinaryHeap<Reverse<(u64, [u64; 2])>> = BinaryHeap::with_capacity(4);
    let mut cur = [0u64; 2];
    let mut cursor = 0u64;

    let emit = |runs: &mut Vec<(u64, (u64, Counts))>, from: u64, to: u64, cur: [u64; 2]| {
        if from >= to || cur == [0, 0] {
            return;
        }
        let counts: Counts = [0, u32::try_from(cur[0]).unwrap_or(u32::MAX), u32::try_from(cur[1]).unwrap_or(u32::MAX)];
        if let Some(last) = runs.last_mut()
            && last.1.0 == from
            && last.1.1 == counts
        {
            last.1.0 = to;
            return;
        }
        runs.push((from, (to, counts)));
    };

    for copy in copies {
        let add = [u64::from(copy.warm), u64::from(copy.cold)];
        if add == [0, 0] {
            continue;
        }
        let span = frame_span(meta, copy.span);
        let (a, b) = (span.start_ns(), span.end_ns());
        while let Some(&Reverse((end, dec))) = active.peek() {
            if end > a {
                break;
            }
            active.pop();
            if end > cursor {
                emit(&mut runs, cursor, end, cur);
                cursor = end;
            }
            cur[0] -= dec[0];
            cur[1] -= dec[1];
        }
        if a > cursor {
            emit(&mut runs, cursor, a, cur);
            cursor = a;
        }
        cur[0] += add[0];
        cur[1] += add[1];
        active.push(Reverse((b, add)));
    }
    while let Some(Reverse((end, dec))) = active.pop() {
        if end > cursor {
            emit(&mut runs, cursor, end, cur);
            cursor = end;
        }
        cur[0] -= dec[0];
        cur[1] -= dec[1];
    }
    debug_assert_eq!(cur, [0, 0]);
    CopySet { segs: runs.into_iter().collect() }
}

fn build_stream_copies(
    meta: &StreamMeta,
    mut copies: Vec<AvailabilityCopy>,
    mut cuts: Vec<AvailabilityCutChunk>,
) -> CopySet {
    if cuts.is_empty() {
        return build_copy_set(meta, copies);
    }
    // Cut removal clamps at zero, so overlapping rate-expanded cut slots can depend on TIME_INDEX order
    // `(start_ns, seq)` relative to overlapping copies on the same stream. Replay in exact TIME_INDEX order.
    copies.sort_unstable_by_key(|c| (c.span.start_ns(), c.seq));
    cuts.sort_unstable_by_key(|c| (c.span.start_ns(), c.seq));
    let mut set = CopySet::new();
    let apply_copy = |set: &mut CopySet, copy: &AvailabilityCopy| {
        let span = frame_span(meta, copy.span);
        for _ in 0..copy.warm {
            set.add(span, StorageTier::Tier1Warm);
        }
        for _ in 0..copy.cold {
            set.add(span, StorageTier::Tier2Cold);
        }
    };
    let apply_cut = |set: &mut CopySet, chunk: &AvailabilityCutChunk| {
        let span = frame_span(meta, chunk.span);
        for &tier in &chunk.tiers {
            set.add(span, tier);
            for &slot in &chunk.cut_slots {
                let _ = set.remove(frame_span(meta, slot), tier);
            }
        }
    };
    let (mut ci, mut ki) = (0usize, 0usize);
    while ci < copies.len() && ki < cuts.len() {
        if (copies[ci].span.start_ns(), copies[ci].seq) <= (cuts[ki].span.start_ns(), cuts[ki].seq) {
            apply_copy(&mut set, &copies[ci]);
            ci += 1;
        } else {
            apply_cut(&mut set, &cuts[ki]);
            ki += 1;
        }
    }
    for copy in &copies[ci..] {
        apply_copy(&mut set, copy);
    }
    for chunk in &cuts[ki..] {
        apply_cut(&mut set, chunk);
    }
    set
}

/// Applies an [`AvailabilitySnapshot`] with a caller-chosen parallel apply threshold, for parity tests.
#[doc(hidden)]
#[must_use]
pub fn load_from_snapshot_for_test(
    snap: AvailabilitySnapshot,
    min_parallel_copies: usize,
) -> (AvailabilityIndex, LoadStats) {
    let (index, stats, _) = load_from_snapshot(snap, Instant::now(), min_parallel_copies);
    (index, stats)
}

fn load_from_snapshot(
    snap: AvailabilitySnapshot,
    load_started: Instant,
    min_parallel_copies: usize,
) -> (AvailabilityIndex, LoadStats, LoadTimings) {
    let mut timings = LoadTimings {
        stream_reads: snap.stream_reads,
        payload_reads: snap.payload_reads,
        payload_apply: Duration::ZERO,
        coverage_load: snap.coverage_reads,
        total: Duration::ZERO,
    };
    let mut index = AvailabilityIndex::new();
    let mut metas = Vec::with_capacity(snap.streams.len());
    for (stream, info) in snap.streams {
        let kind = match info.kind {
            CatalogKind::Sensor => StreamKind::Sensor,
            CatalogKind::Derived => StreamKind::Derived,
        };
        let meta = StreamMeta { kind, rate_hz: info.rate_hz.unwrap_or(0.0) };
        index.register(stream, meta);
        index.name_stream(info.name, stream);
        metas.push((stream, meta));
    }

    let apply_started = Instant::now();
    let mut per_stream_cuts: Vec<Vec<AvailabilityCutChunk>> = (0..metas.len()).map(|_| Vec::new()).collect();
    for chunk in snap.cut_chunks {
        for &slot in &chunk.cut_slots {
            index.record_cut(chunk.id, slot);
        }
        if !chunk.tiers.is_empty()
            && let Ok(pos) = metas.binary_search_by_key(&chunk.stream, |&(s, _)| s)
        {
            per_stream_cuts[pos].push(chunk);
        }
    }

    let total_copies: usize =
        snap.stream_copies.iter().zip(&per_stream_cuts).map(|(copies, cuts)| copies.len() + cuts.len()).sum();
    if total_copies < min_parallel_copies || metas.len() <= 1 {
        for (((stream, meta), copies), cuts) in metas.iter().zip(snap.stream_copies).zip(per_stream_cuts) {
            if !copies.is_empty() || !cuts.is_empty() {
                let set = build_stream_copies(meta, copies, cuts);
                if let Some((_, dst)) = index.streams.get_mut(stream) {
                    *dst = set;
                }
            }
        }
    } else {
        type StreamWork = (usize, StreamMeta, Vec<AvailabilityCopy>, Vec<AvailabilityCutChunk>);
        let mut order: Vec<StreamWork> = metas
            .iter()
            .enumerate()
            .zip(snap.stream_copies)
            .zip(per_stream_cuts)
            .filter(|((_, copies), cuts)| !copies.is_empty() || !cuts.is_empty())
            .map(|(((idx, &(_, meta)), copies), cuts)| (idx, meta, copies, cuts))
            .collect();
        // Schedule largest streams first so a single dense stream never starts after small ones finish.
        order.sort_unstable_by_key(|(_, _, copies, cuts)| Reverse(copies.len() + cuts.len()));
        let slots: Vec<OnceLock<(usize, CopySet)>> = (0..order.len()).map(|_| OnceLock::new()).collect();
        let inputs: Vec<std::sync::Mutex<Option<StreamWork>>> =
            order.into_iter().map(|item| std::sync::Mutex::new(Some(item))).collect();
        let workers = std::thread::available_parallelism()
            .map_or(4, std::num::NonZeroUsize::get)
            .min(MAX_APPLY_WORKERS)
            .min(inputs.len().max(1));
        let next = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..workers {
                let inputs_ref = &inputs;
                let slots_ref = &slots;
                let next_ref = &next;
                scope.spawn(move || {
                    loop {
                        let i = next_ref.fetch_add(1, Ordering::Relaxed);
                        let Some(cell) = inputs_ref.get(i) else { break };
                        let Some((stream_idx, meta, copies, cuts)) =
                            cell.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take()
                        else {
                            continue;
                        };
                        let set = build_stream_copies(&meta, copies, cuts);
                        let _ = slots_ref[i].set((stream_idx, set));
                    }
                });
            }
        });
        for mut slot in slots {
            if let Some((stream_idx, set)) = slot.take() {
                let stream = metas[stream_idx].0;
                if let Some((_, dst)) = index.streams.get_mut(&stream) {
                    *dst = set;
                }
            }
        }
    }
    timings.payload_apply = apply_started.elapsed();

    let cov_apply_started = Instant::now();
    for (pair, set) in snap.coverage {
        if !set.is_empty() {
            index.coverage.insert(pair, set);
        }
    }
    timings.coverage_load += cov_apply_started.elapsed();

    let mut stale_tier0 = snap.stale_tier0;
    if !stale_tier0.is_sorted() {
        stale_tier0.sort_unstable();
    }
    let (segments, coverage_intervals) = index.size();
    let stats = LoadStats { streams: metas.len(), payloads: snap.payloads, segments, coverage_intervals, stale_tier0 };
    timings.total = load_started.elapsed();
    (index, stats, timings)
}

fn load_paged_profiled_from(
    catalog: &dyn Catalog,
    page: usize,
    load_started: Instant,
) -> Result<(AvailabilityIndex, LoadStats, LoadTimings), CatalogError> {
    let mut timings = LoadTimings::default();
    let page = page.max(1);
    let everything = Interval::new(0, u64::MAX).expect("non-empty");
    let mut index = AvailabilityIndex::new();
    let mut stats = LoadStats::default();
    let mut after_stream = None;
    loop {
        let started = Instant::now();
        let streams = catalog.streams(after_stream, page)?;
        timings.stream_reads += started.elapsed();
        let Some(&(last, _)) = streams.last() else { break };
        after_stream = Some(last);
        for (stream, info) in streams {
            stats.streams += 1;
            let kind = match info.kind {
                CatalogKind::Sensor => StreamKind::Sensor,
                CatalogKind::Derived => StreamKind::Derived,
            };
            index.register(stream, StreamMeta { kind, rate_hz: info.rate_hz.unwrap_or(0.0) });
            index.name_stream(info.name.clone(), stream);
            let mut after: Option<RangeCursor> = None;
            loop {
                let started = Instant::now();
                let got = catalog.range(stream, everything, after, page)?;
                timings.payload_reads += started.elapsed();
                let started = Instant::now();
                for item in &got.items {
                    stats.payloads += 1;
                    let Some(span) = item.header.span() else { continue };
                    // A cut frame's slot isn't readable from the chunk any more.
                    let cut: Vec<Interval> = item.frames.as_ref().map_or_else(Vec::new, |frames| {
                        frames.iter().zip(frame_slots(span, frames)).filter(|(f, _)| f.cut).map(|(_, s)| s).collect()
                    });
                    for slot in &cut {
                        index.record_cut(item.header.id(), *slot);
                    }
                    for l in &item.locations {
                        let tier = l.tier();
                        if tier == StorageTier::Tier0Hot {
                            stats.stale_tier0.push(item.header.id());
                            continue;
                        }
                        index.add_copy(stream, span, tier).expect("registered above");
                        for &slot in &cut {
                            let _ = index.remove_copy(stream, slot, tier);
                        }
                    }
                }
                timings.payload_apply += started.elapsed();
                match got.next {
                    Some(next) => after = Some(next),
                    None => break,
                }
            }
        }
    }
    let coverage_started = Instant::now();
    let mut after_key = None;
    loop {
        let keys = catalog.coverage_keys(after_key, page)?;
        let Some(&last) = keys.last() else { break };
        after_key = Some(last);
        for (source, stage) in keys {
            for span in catalog.coverage(source, stage, everything)? {
                index.add_coverage(source, stage, span);
            }
        }
    }
    timings.coverage_load = coverage_started.elapsed();
    if !stats.stale_tier0.is_sorted() {
        stats.stale_tier0.sort_unstable();
    }
    (stats.segments, stats.coverage_intervals) = index.size();
    timings.total = load_started.elapsed();
    Ok((index, stats, timings))
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    fn iv(a: u64, b: u64) -> Interval {
        Interval::new(a, b).unwrap()
    }

    fn tier(t: usize) -> StorageTier {
        StorageTier(u8::try_from(t).unwrap())
    }

    /// A frame captured at `t`, as capture stamps it: an instant.
    fn instant(t: u64) -> Interval {
        Interval::covering(t, t).unwrap()
    }

    #[test]
    fn camera_frames_cover_their_stream_until_the_next_frame() {
        let mut idx = AvailabilityIndex::new();
        let (cam, plain) = (StreamId(1), StreamId(2));
        idx.register(cam, StreamMeta { kind: StreamKind::Sensor, rate_hz: 30.0 });
        idx.register(plain, StreamMeta { kind: StreamKind::Sensor, rate_hz: 0.0 });
        for i in 0..3 {
            idx.add_copy(cam, instant(i * 33_333_333), StorageTier::Tier0Hot).unwrap();
            idx.add_copy(plain, instant(i * 33_333_333), StorageTier::Tier0Hot).unwrap();
        }
        let range = iv(0, 99_999_999); // three periods of 33,333,333 ns
        let covered: Vec<Interval> = idx.covered(cam, range).iter().map(|c| c.interval).collect();
        assert_eq!(covered, vec![range], "three 30 fps frames cover their 0.1 s with no gaps");
        assert_eq!(idx.covered(plain, range).len(), 3, "with no rate, an instant is its own nanosecond");
        // A frame removed (purged, or reclaimed with no other copy) takes its period with it.
        idx.remove_copy(cam, instant(33_333_333), StorageTier::Tier0Hot).unwrap();
        let covered: Vec<Interval> = idx.covered(cam, range).iter().map(|c| c.interval).collect();
        assert_eq!(covered, vec![iv(0, 33_333_333), iv(66_666_666, 99_999_999)]);
    }

    #[test]
    fn a_payload_with_a_duration_covers_exactly_its_span() {
        let mut idx = AvailabilityIndex::new();
        let mic = StreamId(3);
        idx.register(mic, StreamMeta { kind: StreamKind::Sensor, rate_hz: 30.0 });
        idx.add_copy(mic, iv(0, 10), StorageTier::Tier0Hot).unwrap();
        let covered: Vec<Interval> = idx.covered(mic, iv(0, 1_000)).iter().map(|c| c.interval).collect();
        assert_eq!(covered, vec![iv(0, 10)]);
    }

    #[test]
    fn contiguous_copies_merge_into_one_segment() {
        let mut s = CopySet::new();
        for n in 0..100u64 {
            s.add(iv(n * 10, n * 10 + 10), StorageTier::Tier0Hot);
        }
        assert_eq!(s.len(), 1);
        assert_eq!(s.covered(iv(0, 2000)), [Covered { interval: iv(0, 1000), tier: StorageTier::Tier0Hot }]);
    }

    #[test]
    fn removing_one_copy_keeps_an_overlapping_one() {
        let mut s = CopySet::new();
        s.add(iv(0, 20), StorageTier::Tier0Hot);
        s.add(iv(10, 30), StorageTier::Tier0Hot);
        s.remove(iv(0, 20), StorageTier::Tier0Hot).unwrap();
        assert_eq!(s.covered(iv(0, 40)), [Covered { interval: iv(10, 30), tier: StorageTier::Tier0Hot }]);
    }

    #[test]
    fn a_copy_moving_down_a_tier_shows_the_hottest_while_both_exist() {
        let mut s = CopySet::new();
        s.add(iv(0, 10), StorageTier::Tier0Hot);
        s.add(iv(0, 10), StorageTier::Tier1Warm);
        assert_eq!(s.covered(iv(0, 10))[0].tier, StorageTier::Tier0Hot);
        s.remove(iv(0, 10), StorageTier::Tier0Hot).unwrap();
        assert_eq!(s.covered(iv(0, 10)), [Covered { interval: iv(0, 10), tier: StorageTier::Tier1Warm }]);
        s.remove(iv(0, 10), StorageTier::Tier1Warm).unwrap();
        assert!(s.is_empty());
    }

    #[test]
    fn removing_what_was_never_added_is_an_underflow() {
        let mut s = CopySet::new();
        s.add(iv(0, 10), StorageTier::Tier0Hot);
        assert_eq!(s.remove(iv(5, 15), StorageTier::Tier0Hot), Err(Underflow));
        assert!(s.covered(iv(0, 20)).iter().all(|c| c.interval.end_ns() <= 5), "the held part was removed");
        assert_eq!(s.remove(iv(0, 5), StorageTier::Tier1Warm), Err(Underflow));
    }

    #[test]
    fn coverage_gaps_cover_a_window_with_no_coverage_at_all() {
        let mut idx = AvailabilityIndex::new();
        let (s, stage) = (StreamId(1), StageId(2));
        assert_eq!(idx.coverage_gaps(s, stage, iv(0, 100), 5), [iv(0, 100)]);
        assert!(idx.coverage_gaps(s, stage, iv(0, 100), 0).is_empty());
        idx.add_coverage(s, stage, iv(20, 30));
        assert_eq!(idx.coverage_gaps(s, stage, iv(0, 100), 5), [iv(0, 20), iv(30, 100)]);
        idx.remove_coverage(s, stage, iv(20, 30));
        assert_eq!(idx.coverage_gaps(s, stage, iv(0, 100), 5), [iv(0, 100)]);
        assert_eq!(idx.size(), (0, 0));
    }

    #[test]
    fn a_query_starting_or_ending_inside_a_segment_is_clipped() {
        let mut s = CopySet::new();
        s.add(iv(0, 100), StorageTier::Tier1Warm);
        assert_eq!(s.covered(iv(50, 80)), [Covered { interval: iv(50, 80), tier: StorageTier::Tier1Warm }]);
        assert_eq!(s.covered(iv(100, 200)), []);
        assert_eq!(s.covered(iv(99, 200)), [Covered { interval: iv(99, 100), tier: StorageTier::Tier1Warm }]);
    }

    #[test]
    fn spans_reaching_u64_max_work() {
        let mut s = CopySet::new();
        s.add(iv(u64::MAX - 10, u64::MAX), StorageTier::Tier0Hot);
        s.add(iv(0, u64::MAX), StorageTier::Tier0Hot);
        assert_eq!(s.len(), 2);
        assert_eq!(s.covered(iv(0, u64::MAX)), [Covered { interval: iv(0, u64::MAX), tier: StorageTier::Tier0Hot }]);
        s.remove(iv(u64::MAX - 10, u64::MAX), StorageTier::Tier0Hot).unwrap();
        assert_eq!(s.len(), 1);
        s.remove(iv(0, u64::MAX), StorageTier::Tier0Hot).unwrap();
        assert!(s.is_empty());
    }

    #[test]
    fn an_underflow_keeps_other_tiers_and_removes_what_was_held() {
        let mut s = CopySet::new();
        s.add(iv(0, 10), StorageTier::Tier0Hot);
        s.add(iv(0, 10), StorageTier::Tier1Warm);
        s.remove(iv(0, 10), StorageTier::Tier0Hot).unwrap();
        assert_eq!(s.remove(iv(0, 10), StorageTier::Tier0Hot), Err(Underflow));
        assert_eq!(s.covered(iv(0, 10)), [Covered { interval: iv(0, 10), tier: StorageTier::Tier1Warm }]);
        let mut s = CopySet::new();
        s.add(iv(0, 10), StorageTier::Tier0Hot);
        s.add(iv(20, 30), StorageTier::Tier0Hot);
        assert_eq!(s.remove(iv(0, 30), StorageTier::Tier0Hot), Err(Underflow), "the gap at 10..20 held nothing");
        assert!(s.is_empty());
    }

    #[test]
    fn a_tier_past_the_last_counts_as_the_coldest() {
        let mut s = CopySet::new();
        s.add(iv(0, 10), StorageTier(7));
        assert_eq!(s.covered(iv(0, 10))[0].tier, StorageTier::Tier2Cold);
        assert_eq!(s.remove(iv(0, 10), StorageTier(200)), Ok(()));
    }

    #[test]
    fn neighbors_merge_only_once_they_touch() {
        let mut s = CopySet::new();
        s.add(iv(0, 5), StorageTier::Tier0Hot);
        s.add(iv(10, 20), StorageTier::Tier0Hot);
        assert_eq!(s.len(), 2);
        s.add(iv(5, 10), StorageTier::Tier0Hot);
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn changes_on_an_unknown_stream_are_refused_and_re_registering_keeps_copies() {
        let mut idx = AvailabilityIndex::new();
        let s = StreamId(4);
        assert_eq!(idx.add_copy(s, iv(0, 10), StorageTier::Tier0Hot), Err(UnknownStream(s)));
        assert_eq!(idx.remove_copy(s, iv(0, 10), StorageTier::Tier0Hot), Err(Underflow));
        idx.register(s, StreamMeta { kind: StreamKind::Sensor, rate_hz: 30.0 });
        idx.add_copy(s, iv(0, 10), StorageTier::Tier0Hot).unwrap();
        idx.register(s, StreamMeta { kind: StreamKind::Sensor, rate_hz: 10.0 });
        assert_eq!(idx.meta(s).unwrap().rate_hz, 10.0);
        assert_eq!(idx.covered(s, iv(0, 10)).len(), 1, "a rate change keeps the copies");
    }

    #[derive(Clone, Debug)]
    enum Op {
        Add(u64, u64, usize),
        /// Remove any span, whether it is held or not.
        Remove(u64, u64, usize),
        /// Remove the copy added by the n-th still-listed add (modulo how many there are).
        RemoveAdded(usize),
    }

    fn op() -> impl Strategy<Value = Op> {
        prop_oneof![
            (0u64..60, 1u64..20, 0usize..TIERS).prop_map(|(a, len, t)| Op::Add(a, a + len, t)),
            (0u64..60, 1u64..20, 0usize..TIERS).prop_map(|(a, len, t)| Op::Remove(a, a + len, t)),
            (0usize..64).prop_map(Op::RemoveAdded),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]

        /// The append path and reverse-order general path must represent the same multiset of copies,
        /// including touching spans, duplicates, tier changes, and later removals near either clock boundary.
        #[test]
        fn ordered_appends_match_reverse_order_and_later_removals(
            pieces in proptest::collection::vec((0u64..4, 1u64..6, 0usize..(TIERS + 2), 1u32..4), 0..64),
            base in prop_oneof![Just(0u64), Just(u64::MAX - 1024)],
        ) {
            let mut cursor = base;
            let mut copies = Vec::new();
            for (gap, len, t, count) in pieces {
                cursor += gap;
                let span = iv(cursor, cursor + len);
                cursor += len;
                for _ in 0..count {
                    copies.push((span, tier(t)));
                }
            }
            let mut forward = CopySet::new();
            let mut reverse = CopySet::new();
            for &(span, t) in &copies {
                forward.add(span, t);
            }
            for &(span, t) in copies.iter().rev() {
                reverse.add(span, t);
            }
            prop_assert_eq!(forward.segments(), reverse.segments());
            for &(span, t) in copies.iter().step_by(2) {
                prop_assert_eq!(forward.remove(span, t), Ok(()));
                prop_assert_eq!(reverse.remove(span, t), Ok(()));
            }
            prop_assert_eq!(forward.segments(), reverse.segments());
        }

        /// Against a per-nanosecond count of copies per tier: any adds and any removes (held or not) give the
        /// right Underflow answer, a covered query over any range gives each nanosecond the hottest tier with a
        /// copy, the answer is clipped, ordered, disjoint, and merged, and the segments stay normalized.
        #[test]
        fn the_set_matches_a_count_per_nanosecond(
            ops in proptest::collection::vec(op(), 0..40),
            qa in 0u64..90,
            ql in 1u64..40,
        ) {
            let mut set = CopySet::new();
            let mut counts = vec![[0u32; TIERS]; 140];
            let mut added: Vec<(u64, u64, usize)> = Vec::new();
            let remove = |set: &mut CopySet, counts: &mut Vec<Counts>, a: u64, b: u64, t: usize| {
                let held = (a..b).all(|n| counts[usize::try_from(n).unwrap()][t] > 0);
                let got = set.remove(iv(a, b), tier(t));
                for n in a..b {
                    let c = &mut counts[usize::try_from(n).unwrap()][t];
                    *c = c.saturating_sub(1);
                }
                (got.is_ok(), held)
            };
            for op in ops {
                match op {
                    Op::Add(a, b, t) => {
                        set.add(iv(a, b), tier(t));
                        for n in a..b {
                            counts[usize::try_from(n).unwrap()][t] += 1;
                        }
                        added.push((a, b, t));
                    }
                    Op::Remove(a, b, t) => {
                        let (ok, held) = remove(&mut set, &mut counts, a, b, t);
                        prop_assert_eq!(ok, held);
                    }
                    Op::RemoveAdded(n) if !added.is_empty() => {
                        let (a, b, t) = added.remove(n % added.len());
                        let (ok, held) = remove(&mut set, &mut counts, a, b, t);
                        prop_assert_eq!(ok, held);
                    }
                    Op::RemoveAdded(_) => {}
                }
            }
            let q = iv(qa, qa + ql);
            let got = set.covered(q);
            for c in &got {
                prop_assert!(c.interval.start_ns() >= q.start_ns() && c.interval.end_ns() <= q.end_ns());
            }
            for w in got.windows(2) {
                prop_assert!(w[0].interval.end_ns() <= w[1].interval.start_ns(), "ordered and disjoint");
                prop_assert!(
                    !(w[0].interval.end_ns() == w[1].interval.start_ns() && w[0].tier == w[1].tier),
                    "touching parts of one tier are merged"
                );
            }
            for n in q.start_ns()..q.end_ns() {
                let want = counts[usize::try_from(n).unwrap()].iter().position(|&c| c > 0);
                let have = got.iter().find(|c| c.interval.contains(n)).map(|c| usize::from(c.tier.0));
                prop_assert_eq!(have, want, "at {}", n);
            }
            let segs: Vec<(u64, u64, Counts)> = set.segs.iter().map(|(&s, &(e, c))| (s, e, c)).collect();
            for &(s, e, c) in &segs {
                prop_assert!(s < e);
                prop_assert!(c != [0; TIERS]);
            }
            for w in segs.windows(2) {
                prop_assert!(w[0].1 <= w[1].0);
                prop_assert!(!(w[0].1 == w[1].0 && w[0].2 == w[1].2));
            }
            if counts.iter().all(|c| *c == [0; TIERS]) {
                prop_assert!(set.is_empty());
            }
        }

        /// Bulk sweep-line CopySet construction must equal incremental `AvailabilityIndex::add_copy` across
        /// arbitrary unordered, overlapping, touching, rated-instant, and `u64::MAX`-adjacent spans.
        #[test]
        fn bulk_copy_set_matches_incremental_adds(
            rate_hz in prop_oneof![Just(0.0), Just(30.0), Just(60.0)],
            base in prop_oneof![Just(0u64), Just(u64::MAX - 256)],
            items in proptest::collection::vec((0u64..120, 1u64..40, 0u8..=1, 0u8..=1), 0..80),
        ) {
            let meta = StreamMeta { kind: StreamKind::Sensor, rate_hz };
            let stream = StreamId(1);
            let mut incremental = AvailabilityIndex::new();
            incremental.register(stream, meta);
            let mut bulk_items = Vec::with_capacity(items.len());
            for (offset, len, warm, cold) in items {
                let start = base.saturating_add(offset).min(u64::MAX - 1);
                let end = start.saturating_add(len).max(start + 1);
                let span = iv(start, end);
                if warm > 0 {
                    incremental.add_copy(stream, span, StorageTier::Tier1Warm).unwrap();
                }
                if cold > 0 {
                    incremental.add_copy(stream, span, StorageTier::Tier2Cold).unwrap();
                }
                bulk_items.push(AvailabilityCopy { span, seq: 0, warm, cold });
            }
            let bulk = build_copy_set(&meta, bulk_items);
            prop_assert_eq!(bulk.segments(), incremental.streams.get(&stream).unwrap().1.segments());
        }
    }
}
