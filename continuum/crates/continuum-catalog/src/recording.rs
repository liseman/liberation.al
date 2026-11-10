//! Durable original native-media inputs, independent of current physical copies and analysis outcomes.
//! Provisional raw inputs and erased history remain unknown. Aging never manufactures completion receipts.

use crate::codec::{self, In, Out};
use crate::{CatalogError, CoverageSet, Result, StreamInfo, StreamKind};
use continuum_core::{HeaderBuf, Interval, StreamId, Uuid};
use std::collections::{BTreeMap, HashMap};

/// Maximum returned inputs and unknown intervals per page.
pub const MAX_RECORDING_ITEMS: usize = 128;
/// Maximum examined recording rows per page.
pub const MAX_RECORDING_SCAN: usize = 4096;
/// Maximum canonical response bytes per page.
pub const MAX_RECORDING_BYTES: usize = 1024 * 1024;

/// One original native-media chunk. Audio runs are actual durations; video may have no duration evidence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordingInput {
    /// Original chunk UUID to which native analysis receipts bind.
    pub chunk_id: Uuid,
    /// Immutable original temporal key, including eligibility-only instant video.
    pub start_ns: u64,
    /// Minimum original raw ingress sequence, never the later chunk commit.
    pub first_seq: u64,
    /// Normalized actual audio durations; gaps are preserved. Video eligibility may use an empty vector.
    pub spans: Vec<Interval>,
    /// Source generation at capture; historical generations remain valid.
    pub source_generation: u64,
    /// Immutable native PCM channels, unknown for unsupported source shapes.
    pub channels: Option<u16>,
    /// Immutable source shape at capture.
    pub shape_name: String,
    /// Immutable source shape version at capture.
    pub shape_version: u32,
}
/// Exclusive time-index position, shared by inputs, provisional rows, and unknown rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct RecordingCursor {
    /// First original span start.
    pub start_ns: u64,
    /// Stable original input identity.
    pub id: Uuid,
}
/// One coherent bounded recording-applicability snapshot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordingPage {
    /// Original durable inputs intersecting the query window.
    pub inputs: Vec<RecordingInput>,
    /// Unknown/unassigned/erased source intervals clipped to the query window.
    pub unknown: Vec<Interval>,
    /// Continue after this examined row; None means the scan finished.
    pub next: Option<RecordingCursor>,
    /// Current certified-era completeness. Pre-era uncertainty is reported separately and never authorizes a target.
    pub history_known: bool,
    /// First catalog sequence in the certified recording era.
    pub era_start_seq: u64,
    /// Pre-era known or unknown history may overlap; diagnostic only for current-era targets.
    pub legacy_overlap: bool,
    /// Invalidation revision from this snapshot.
    pub revision: u64,
    /// Catalog head watermark from this snapshot, including non-source commits.
    pub through_seq: u64,
    /// Current source generation; does not replace historical witness generations.
    pub source_generation: u64,
    /// Rows examined, including rows that did not intersect the query window.
    pub scanned: usize,
}

pub(crate) fn invalid(message: &str) -> CatalogError {
    CatalogError::Invalid(format!("recording applicability: {message}"))
}
pub(crate) fn native(info: &StreamInfo) -> bool {
    info.kind == StreamKind::Sensor && (info.name.ends_with(".audio") || info.name.ends_with(".video"))
}
#[derive(Clone, Default)]
pub(crate) struct Meta {
    pub known: bool,
    pub current_known: bool,
    pub generation: u64,
    pub longest: u64,
}
impl Meta {
    pub fn encode(&self) -> Vec<u8> {
        Out::new()
            .u8(u8::from(self.known))
            .u8(u8::from(self.current_known))
            .u64(self.generation)
            .u64(self.longest)
            .done()
    }
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let mut i = In::new(bytes);
        let known = match i.u8()? {
            0 => false,
            1 => true,
            _ => return Err(invalid("corrupt history flag")),
        };
        let current_known = match i.u8()? {
            0 => false,
            1 => true,
            _ => return Err(invalid("corrupt era flag")),
        };
        let value = Self { known, current_known, generation: i.u64()?, longest: i.u64()? };
        if !i.is_empty() {
            return Err(invalid("trailing metadata"));
        }
        Ok(value)
    }
}
#[derive(Clone)]
pub(crate) struct Row {
    pub input: RecordingInput,
    pub end_ns: u64,
    pub bound: bool,
    pub unknown: bool,
    pub scope_only: bool,
    pub subjects: Vec<Uuid>,
    pub subjects_complete: bool,
}
impl Row {
    pub fn raw(header: &HeaderBuf, info: &StreamInfo, generation: u64, first_seq: u64) -> Result<Self> {
        let span = header.span().ok_or_else(|| invalid("unrepresentable source interval"))?;
        let channels = (info.shape_version == 1 && info.format == continuum_core::PayloadFormat::RawBytes)
            .then(|| info.shape_name.strip_prefix("PCM_F32LEx")?.parse::<u16>().ok().filter(|n| *n > 0))
            .flatten();
        let subjects = header.subjects();
        let subjects_complete = subjects.len() <= 128;
        Ok(Self {
            end_ns: span.end_ns(),
            subjects: if subjects_complete { subjects } else { vec![] },
            subjects_complete,
            scope_only: false,
            input: RecordingInput {
                chunk_id: header.id(),
                start_ns: header.start_ns(),
                first_seq,
                spans: if info.name.ends_with(".audio") && header.end_ns() > header.start_ns() {
                    vec![span]
                } else {
                    vec![]
                },
                source_generation: generation,
                channels,
                shape_name: info.shape_name.clone(),
                shape_version: info.shape_version,
            },
            bound: false,
            unknown: !subjects_complete || (info.name.ends_with(".audio") && header.end_ns() <= header.start_ns()),
        })
    }
    pub fn cursor(&self) -> RecordingCursor {
        RecordingCursor { start_ns: self.input.start_ns, id: self.input.chunk_id }
    }
    pub fn reach(&self) -> u64 {
        self.end_ns.saturating_sub(self.input.start_ns)
    }
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Out::new()
            .u8(u8::from(self.bound))
            .u8(u8::from(self.unknown))
            .u8(u8::from(self.scope_only))
            .u8(u8::from(self.subjects_complete))
            .u32(self.subjects.len() as u32)
            .raw(&self.subjects.iter().flat_map(|s| s.0).collect::<Vec<_>>())
            .id(self.input.chunk_id)
            .u64(self.input.start_ns)
            .u64(self.end_ns)
            .u64(self.input.first_seq)
            .u64(self.input.source_generation)
            .u32(u32::from(self.input.channels.unwrap_or(0)))
            .str(&self.input.shape_name)
            .u32(self.input.shape_version)
            .u32(self.input.spans.len() as u32);
        for span in &self.input.spans {
            out = codec::put_interval(out, *span);
        }
        out.done()
    }
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() > MAX_RECORDING_BYTES {
            return Err(invalid("oversized stored witness"));
        }
        let mut i = In::new(bytes);
        let bound = i.u8()?;
        let unknown = i.u8()?;
        let scope_only = i.u8()?;
        let subjects_complete = i.u8()?;
        let subject_count = i.u32()? as usize;
        if subject_count > 128 {
            return Err(invalid("invalid subject count"));
        }
        let subjects = (0..subject_count).map(|_| i.id()).collect::<Result<Vec<_>>>()?;
        if bound > 1
            || unknown > 1
            || scope_only > 1
            || subjects_complete > 1
            || (subjects_complete == 0 && unknown == 0)
            || (scope_only != 0 && (unknown == 0 || bound != 0))
        {
            return Err(invalid("corrupt witness flags"));
        }
        let chunk_id = i.id()?;
        let start_ns = i.u64()?;
        let end_ns = i.u64()?;
        let first_seq = i.u64()?;
        if end_ns <= start_ns {
            return Err(invalid("invalid eligibility envelope"));
        }
        let source_generation = i.u64()?;
        let channels = i.u32()?;
        if channels > u32::from(u16::MAX) {
            return Err(invalid("corrupt channel count"));
        }
        let shape_name = i.string()?;
        let shape_version = i.u32()?;
        let count = i.u32()? as usize;
        if count > MAX_RECORDING_SCAN {
            return Err(invalid("invalid source run count"));
        }
        let mut spans = Vec::<Interval>::with_capacity(count);
        for _ in 0..count {
            let span = codec::interval(&mut i)?;
            if spans.last().is_some_and(|last| last.end_ns() >= span.start_ns()) {
                return Err(invalid("unnormalized source runs"));
            }
            spans.push(span);
        }
        if !i.is_empty() {
            return Err(invalid("trailing witness bytes"));
        }
        Ok(Self {
            end_ns,
            scope_only: scope_only != 0,
            subjects_complete: subjects_complete != 0,
            subjects,
            input: RecordingInput {
                chunk_id,
                start_ns,
                first_seq,
                source_generation,
                channels: (channels != 0).then_some(channels as u16),
                shape_name,
                shape_version,
                spans,
            },
            bound: bound != 0,
            unknown: unknown != 0,
        })
    }
}
/// Bind only complete, same-generation original raw witnesses. Missing/cut inputs stay explicitly unknown.
pub(crate) fn bind(
    header: &HeaderBuf,
    info: &StreamInfo,
    generation: u64,
    first_seq: u64,
    raw: &[Option<Row>],
    clean: bool,
) -> Result<Row> {
    let mut row = Row::raw(header, info, generation, first_seq)?;
    // Guarded publication scans payload header spans. A bound witness must not
    // extend beyond that searchable envelope, or late binding could evade CAS.
    if raw.iter().flatten().any(|source| source.input.start_ns < header.start_ns() || source.end_ns > row.end_ns) {
        return Err(invalid("chunk header does not cover its original recording inputs"));
    }
    row.bound = true;
    row.unknown |= !clean || raw.is_empty() || raw.iter().any(Option::is_none);
    row.input.first_seq = raw.iter().flatten().map(|r| r.input.first_seq).min().unwrap_or(0);
    let mut spans = CoverageSet::new();
    if let Some(first) = raw.iter().flatten().next() {
        row.input.source_generation = first.input.source_generation;
        row.input.channels = first.input.channels;
        row.input.shape_name.clone_from(&first.input.shape_name);
        row.input.shape_version = first.input.shape_version;
    }
    for source in raw.iter().flatten() {
        row.end_ns = row.end_ns.max(source.end_ns);
        row.input.start_ns = row.input.start_ns.min(source.input.start_ns);
        row.unknown |= source.unknown
            || source.bound
            || source.input.source_generation != row.input.source_generation
            || source.input.channels != row.input.channels
            || source.input.shape_name != row.input.shape_name
            || source.input.shape_version != row.input.shape_version;
        for span in &source.input.spans {
            spans.insert(*span);
        }
    }
    let mut subjects: std::collections::BTreeSet<_> = row.subjects.iter().copied().collect();
    for source in raw.iter().flatten() {
        row.subjects_complete &= source.subjects_complete;
        subjects.extend(source.subjects.iter().copied());
        if subjects.len() > 128 {
            row.subjects_complete = false;
            break;
        }
    }
    if row.subjects_complete {
        row.subjects = subjects.into_iter().collect();
    } else {
        row.subjects.clear();
        row.unknown = true;
    }
    if row.unknown {
        for span in &row.input.spans {
            spans.insert(*span);
        }
    }
    if !spans.is_empty() {
        row.input.spans = spans.iter().collect();
    }
    Ok(row)
}

pub(crate) struct PageBuilder {
    pub page: RecordingPage,
    pub more: bool,
    window: Interval,
    after: Option<RecordingCursor>,
    last: Option<RecordingCursor>,
    max_items: usize,
    max_scan: usize,
    max_bytes: usize,
    bytes: usize,
}
impl PageBuilder {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        meta: &Meta,
        revision: u64,
        through_seq: u64,
        era_start_seq: u64,
        window: Interval,
        after: Option<RecordingCursor>,
        max_items: usize,
        max_scan: usize,
        max_bytes: usize,
    ) -> Result<Self> {
        if max_items == 0
            || max_items > MAX_RECORDING_ITEMS
            || max_scan == 0
            || max_scan > MAX_RECORDING_SCAN
            || !(128..=MAX_RECORDING_BYTES).contains(&max_bytes)
        {
            return Err(invalid("invalid page budget"));
        }
        if revision == u64::MAX {
            return Err(invalid("invalidation revision unavailable"));
        }
        Ok(Self {
            page: RecordingPage {
                inputs: vec![],
                unknown: vec![],
                next: None,
                history_known: meta.current_known && era_start_seq != 0,
                era_start_seq,
                legacy_overlap: !meta.known,
                revision,
                through_seq,
                source_generation: meta.generation,
                scanned: 0,
            },
            more: false,
            window,
            after,
            last: after,
            max_items,
            max_scan,
            max_bytes,
            bytes: 128,
        })
    }
    pub fn visit(&mut self, row: Row) -> Result<bool> {
        let cursor = row.cursor();
        if self.after.is_some_and(|after| cursor <= after) {
            return Ok(true);
        }
        if self.page.scanned == self.max_scan {
            self.more = true;
            return Ok(false);
        }
        self.page.scanned += 1;
        let envelope =
            Interval::new(row.input.start_ns, row.end_ns).ok_or_else(|| invalid("invalid source envelope"))?;
        let footprints: Vec<_> =
            if row.scope_only || row.input.spans.is_empty() { vec![envelope] } else { row.input.spans.clone() };
        let clipped: Vec<_> = footprints
            .iter()
            .filter_map(|s| {
                Interval::new(s.start_ns().max(self.window.start_ns()), s.end_ns().min(self.window.end_ns()))
            })
            .collect();
        if !clipped.is_empty() && row.input.first_seq < self.page.era_start_seq {
            self.page.legacy_overlap = true;
            self.last = Some(cursor);
            return Ok(true);
        }
        if !clipped.is_empty() {
            let unknown = row.unknown || !row.bound;
            let items = if unknown { clipped.len() } else { 1 };
            let bytes = if unknown { clipped.len() * 16 } else { row.encode().len() };
            if self.page.inputs.len() + self.page.unknown.len() + items > self.max_items
                || self.bytes + bytes > self.max_bytes
            {
                if self.page.inputs.is_empty() && self.page.unknown.is_empty() {
                    return Err(invalid("one witness exceeds page budget"));
                }
                self.more = true;
                return Ok(false);
            }
            self.bytes += bytes;
            if unknown {
                self.page.unknown.extend(clipped);
            } else {
                self.page.inputs.push(row.input);
            }
        }
        self.last = Some(cursor);
        Ok(true)
    }
    pub fn finish(mut self) -> RecordingPage {
        // Independent original inputs can overlap in UTC. The wire contract exposes
        // one normalized union, while scan/item/byte accounting remains conservative.
        let mut unknown = CoverageSet::new();
        for span in self.page.unknown.drain(..) {
            unknown.insert(span);
        }
        self.page.unknown = unknown.iter().collect();
        if self.more {
            self.page.next = self.last;
        }
        self.page
    }
}

#[derive(Clone, Default)]
pub(crate) struct MemoryLedger {
    pub meta: HashMap<u32, Meta>,
    pub rows: BTreeMap<(u32, RecordingCursor), Row>,
    pub ids: HashMap<Uuid, (u32, RecordingCursor)>,
}
impl MemoryLedger {
    pub fn put(&mut self, stream: StreamId, row: Row) {
        let key = (stream.0, row.cursor());
        if row.scope_only {
            if self.rows.get(&key).is_some_and(|r| !r.scope_only) {
                let meta = self.meta.entry(stream.0).or_default();
                meta.known = false;
                meta.current_known = false;
                return;
            }
        } else if let Some(old) = self.ids.insert(row.input.chunk_id, key) {
            self.rows.remove(&old);
        }
        let meta = self.meta.entry(stream.0).or_default();
        meta.longest = meta.longest.max(row.reach());
        if !row.subjects_complete {
            meta.known = false;
            meta.current_known = false;
        }
        self.rows.insert(key, row);
    }
    pub fn get(&self, id: Uuid) -> Option<Row> {
        self.ids.get(&id).and_then(|key| self.rows.get(key)).cloned()
    }
    pub fn remove(&mut self, id: Uuid) {
        if let Some(key) = self.ids.remove(&id) {
            self.rows.remove(&key);
        }
    }
    pub fn unknown(&mut self, id: Uuid) {
        if let Some(key) = self.ids.get(&id)
            && let Some(row) = self.rows.get_mut(key)
        {
            row.unknown = true;
        }
    }
}
