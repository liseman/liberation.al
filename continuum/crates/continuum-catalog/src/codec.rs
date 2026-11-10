//! Byte encodings for the redb catalog: fixed-width big-endian keys (so byte order is key order) and simple
//! length-prefixed values for the catalog's own records.

use continuum_core::{BootId, DomainLayer, Interval, PayloadFormat, StorageLocator, StorageTier, StreamId, Uuid};

use crate::types::*;
use crate::{CatalogError, Result};

/// Builds a value or key.
#[derive(Default)]
pub(crate) struct Out(pub(crate) Vec<u8>);

impl Out {
    pub(crate) fn new() -> Self {
        Self(Vec::new())
    }
    pub(crate) fn u8(mut self, v: u8) -> Self {
        self.0.push(v);
        self
    }
    pub(crate) fn u32(mut self, v: u32) -> Self {
        self.0.extend_from_slice(&v.to_be_bytes());
        self
    }
    pub(crate) fn u64(mut self, v: u64) -> Self {
        self.0.extend_from_slice(&v.to_be_bytes());
        self
    }
    pub(crate) fn id(mut self, v: Uuid) -> Self {
        self.0.extend_from_slice(&v.0);
        self
    }
    pub(crate) fn raw(mut self, v: &[u8]) -> Self {
        self.0.extend_from_slice(v);
        self
    }
    /// A LEB128 varint (values only: it breaks key order).
    pub(crate) fn var(mut self, mut v: u64) -> Self {
        while v >= 0x80 {
            self.0.push((v & 0x7f) as u8 | 0x80);
            v >>= 7;
        }
        self.0.push(v as u8);
        self
    }
    /// A signed difference as a zigzag varint.
    pub(crate) fn zigzag(self, v: i64) -> Self {
        self.var(((v << 1) ^ (v >> 63)).cast_unsigned())
    }
    /// A length-prefixed byte string (values only: it breaks key order).
    pub(crate) fn bytes(self, v: &[u8]) -> Self {
        let len = u32::try_from(v.len()).unwrap_or(u32::MAX);
        self.u32(len).raw(v)
    }
    pub(crate) fn str(self, v: &str) -> Self {
        self.bytes(v.as_bytes())
    }
    pub(crate) fn opt_u64(self, v: Option<u64>) -> Self {
        match v {
            Some(v) => self.u8(1).u64(v),
            None => self.u8(0),
        }
    }
    pub(crate) fn done(self) -> Vec<u8> {
        self.0
    }
}

/// Reads what [`Out`] wrote.
pub(crate) struct In<'a>(&'a [u8]);

fn corrupt(what: &str) -> CatalogError {
    CatalogError::Storage(format!("catalog record is corrupt: {what}").into())
}

impl<'a> In<'a> {
    pub(crate) fn new(b: &'a [u8]) -> Self {
        Self(b)
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.0.len() < n {
            return Err(corrupt("too short"));
        }
        let (head, rest) = self.0.split_at(n);
        self.0 = rest;
        Ok(head)
    }
    pub(crate) fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    pub(crate) fn var(&mut self) -> Result<u64> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let b = self.u8()?;
            if shift == 63 && b > 1 {
                break;
            }
            v |= u64::from(b & 0x7f) << shift;
            if b & 0x80 == 0 {
                return Ok(v);
            }
        }
        Err(corrupt("a varint over 64 bits"))
    }
    pub(crate) fn zigzag(&mut self) -> Result<i64> {
        let v = self.var()?;
        Ok((v >> 1).cast_signed() ^ -((v & 1).cast_signed()))
    }
    pub(crate) fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_be_bytes(self.take(4)?.try_into().map_err(|_| corrupt("u32"))?))
    }
    pub(crate) fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_be_bytes(self.take(8)?.try_into().map_err(|_| corrupt("u64"))?))
    }
    pub(crate) fn id(&mut self) -> Result<Uuid> {
        Ok(Uuid(self.take(16)?.try_into().map_err(|_| corrupt("id"))?))
    }
    pub(crate) fn bytes(&mut self) -> Result<&'a [u8]> {
        let n = self.u32()? as usize;
        self.take(n)
    }
    pub(crate) fn string(&mut self) -> Result<String> {
        String::from_utf8(self.bytes()?.to_vec()).map_err(|_| corrupt("utf-8"))
    }
    pub(crate) fn opt_u64(&mut self) -> Result<Option<u64>> {
        Ok(if self.u8()? == 1 { Some(self.u64()?) } else { None })
    }
    pub(crate) fn rest(&self) -> &'a [u8] {
        self.0
    }
    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

pub(crate) fn interval(i: &mut In<'_>) -> Result<Interval> {
    let (s, e) = (i.u64()?, i.u64()?);
    Interval::new(s, e).ok_or_else(|| corrupt("empty interval"))
}

pub(crate) fn put_interval(o: Out, v: Interval) -> Out {
    o.u64(v.start_ns()).u64(v.end_ns())
}

pub(crate) fn ids(b: &[u8]) -> Result<Vec<Uuid>> {
    if !b.len().is_multiple_of(16) {
        return Err(corrupt("id list"));
    }
    Ok(b.chunks_exact(16).map(|c| Uuid(c.try_into().unwrap_or([0; 16]))).collect())
}

pub(crate) fn id_at(b: &[u8], offset: usize) -> Result<Uuid> {
    b.get(offset..offset + 16).and_then(|s| s.try_into().ok()).map(Uuid).ok_or_else(|| corrupt("id"))
}

pub(crate) fn u32_at(b: &[u8], offset: usize) -> Result<u32> {
    b.get(offset..offset + 4).and_then(|s| s.try_into().ok()).map(u32::from_be_bytes).ok_or_else(|| corrupt("u32"))
}

pub(crate) fn u64_at(b: &[u8], offset: usize) -> Result<u64> {
    b.get(offset..offset + 8).and_then(|s| s.try_into().ok()).map(u64::from_be_bytes).ok_or_else(|| corrupt("u64"))
}

// ---- records ----

pub(crate) fn stream_info(v: &StreamInfo) -> Vec<u8> {
    let o = Out::new()
        .str(&v.name)
        .u8(match v.kind {
            StreamKind::Sensor => 0,
            StreamKind::Derived => 1,
        })
        .u8(v.layer.0)
        .u8(v.format.0)
        .str(&v.shape_name)
        .u32(v.shape_version)
        .opt_u64(v.rate_hz.map(f64::to_bits));
    match v.anchor {
        Some(a) => o.u8(1).raw(&a.boot.0).u64(a.mono_ns).u64(a.utc_ns).done(),
        None => o.u8(0).done(),
    }
}

pub(crate) fn read_stream_info(b: &[u8]) -> Result<StreamInfo> {
    let mut i = In::new(b);
    let name = i.string()?;
    let kind = if i.u8()? == 0 { StreamKind::Sensor } else { StreamKind::Derived };
    let layer = DomainLayer(i.u8()?);
    let format = PayloadFormat(i.u8()?);
    let shape_name = i.string()?;
    let shape_version = i.u32()?;
    let rate_hz = i.opt_u64()?.map(f64::from_bits);
    let anchor = if i.u8()? == 1 {
        let boot = BootId(i.id()?.0);
        Some(ClockAnchor { boot, mono_ns: i.u64()?, utc_ns: i.u64()? })
    } else {
        None
    };
    Ok(StreamInfo { name, kind, layer, format, shape_name, shape_version, rate_hz, anchor })
}

/// A frame index stored as its columns, as they are.
const FRAMES_PLAIN: u8 = 0;
/// A frame index whose columns are zstd-compressed: the columns' length as a varint, then the frame.
const FRAMES_ZSTD: u8 = 1;
/// The most a frame index's columns may take, compressed or not: far past any chunk the catalog accepts.
const MAX_FRAME_COLUMNS: usize = 16 << 20;

/// A chunk's frame index, delta-coded in columns (docs/CONTINUUM.md, "Video chunks"): the frame count; each frame's
/// start as the change in its spacing from the previous frame's; each record's offset as the change in its distance
/// past the previous record's end (a fixed record head apart: all zeros); each length as a varint; and the indexes
/// of the cut frames. All differences wrap, so any index round-trips. The columns are then zstd-compressed when
/// that is smaller, which it is for anything but a handful of frames: a camera's frame takes about 4 bytes (most
/// of it the jitter in its capture times), against 25 as fixed-width integers.
pub(crate) fn frames(v: &[ChunkFrame]) -> Vec<u8> {
    let mut times = Out::new().var(v.len() as u64);
    let (mut gaps, mut lengths) = (Out::new(), Out::new());
    let (mut start, mut spacing, mut end, mut gap) = (0u64, 0u64, 0u64, 0u64);
    for f in v {
        let step = f.start_ns.wrapping_sub(start);
        let here = f.offset.wrapping_sub(end);
        times = times.zigzag(step.wrapping_sub(spacing).cast_signed());
        gaps = gaps.zigzag(here.wrapping_sub(gap).cast_signed());
        lengths = lengths.var(f.length);
        (start, spacing, end, gap) = (f.start_ns, step, f.offset.wrapping_add(f.length), here);
    }
    let cut = crate::cut_indexes(v);
    let mut columns = times.raw(&gaps.done()).raw(&lengths.done()).var(cut.len() as u64);
    let mut last = 0;
    for index in cut {
        columns = columns.var(u64::from(index - last));
        last = index;
    }
    let columns = columns.done();
    match zstd::bulk::compress(&columns, 3) {
        Ok(packed) if packed.len() + 5 < columns.len() => {
            Out::new().u8(FRAMES_ZSTD).var(columns.len() as u64).raw(&packed).done()
        }
        _ => Out::new().u8(FRAMES_PLAIN).raw(&columns).done(),
    }
}

pub(crate) fn read_frames(b: &[u8]) -> Result<Vec<ChunkFrame>> {
    let mut i = In::new(b);
    let unpacked;
    let mut i = match i.u8()? {
        FRAMES_PLAIN => i,
        FRAMES_ZSTD => {
            let len = usize::try_from(i.var()?).ok().filter(|&n| n <= MAX_FRAME_COLUMNS);
            let len = len.ok_or_else(|| corrupt("a frame index too large"))?;
            unpacked =
                zstd::bulk::decompress(i.rest(), len).map_err(|_| corrupt("a frame index that doesn't unpack"))?;
            if unpacked.len() != len {
                return Err(corrupt("a frame index of another length"));
            }
            In::new(&unpacked)
        }
        _ => return Err(corrupt("an unknown frame index layout")),
    };
    // Every frame takes at least three bytes of columns, so a count past that is damage, not a size to allocate.
    let count = usize::try_from(i.var()?).ok().filter(|&n| n <= i.rest().len() / 3);
    let count = count.ok_or_else(|| corrupt("a frame count past the index"))?;
    let mut out = Vec::with_capacity(count);
    let (mut start, mut spacing) = (0u64, 0u64);
    for _ in 0..count {
        spacing = spacing.wrapping_add(i.zigzag()?.cast_unsigned());
        start = start.wrapping_add(spacing);
        out.push(ChunkFrame { start_ns: start, offset: 0, length: 0, cut: false });
    }
    let gaps = (0..count).map(|_| i.zigzag()).collect::<Result<Vec<_>>>()?;
    let (mut end, mut gap) = (0u64, 0u64);
    for (f, change) in out.iter_mut().zip(gaps) {
        gap = gap.wrapping_add(change.cast_unsigned());
        f.offset = end.wrapping_add(gap);
        f.length = i.var()?;
        end = f.offset.wrapping_add(f.length);
    }
    let mut index = 0usize;
    for _ in 0..i.var()? {
        index = usize::try_from(i.var()?).ok().and_then(|step| index.checked_add(step)).unwrap_or(usize::MAX);
        out.get_mut(index).ok_or_else(|| corrupt("a cut frame past the index"))?.cut = true;
    }
    if !i.is_empty() {
        return Err(corrupt("bytes after the frame index"));
    }
    Ok(out)
}

pub(crate) fn locator(b: &[u8]) -> Result<StorageLocator> {
    StorageLocator::decode(b).map_err(|e| CatalogError::Storage(Box::new(e)))
}

fn put_locators(mut o: Out, v: &[StorageLocator]) -> Out {
    o = o.u32(u32::try_from(v.len()).unwrap_or(u32::MAX));
    for l in v {
        o = o.bytes(&l.encode());
    }
    o
}

fn get_locators(i: &mut In<'_>) -> Result<Vec<StorageLocator>> {
    let n = i.u32()?;
    (0..n).map(|_| locator(i.bytes()?)).collect()
}

pub(crate) fn removed(v: &RemovedPayload) -> Vec<u8> {
    put_locators(Out::new().id(v.id).u32(v.stream.0), &v.locations).done()
}

pub(crate) fn read_removed(b: &[u8]) -> Result<RemovedPayload> {
    let mut i = In::new(b);
    Ok(RemovedPayload { id: i.id()?, stream: StreamId(i.u32()?), locations: get_locators(&mut i)? })
}

fn put_kind(o: Out, k: &PurgeKind) -> Out {
    match k {
        PurgeKind::Entity => o.u8(0),
        PurgeKind::Ids { count } => o.u8(2).u64(*count as u64),
        PurgeKind::AgeOut { count } => o.u8(3).u64(*count as u64),
        PurgeKind::Range { span, streams } => {
            let o = put_interval(o.u8(1), *span);
            match streams {
                None => o.u8(0),
                Some(list) => {
                    let mut o = o.u8(1).u32(u32::try_from(list.len()).unwrap_or(u32::MAX));
                    for s in list {
                        o = o.u32(s.0);
                    }
                    o
                }
            }
        }
    }
}

fn get_kind(i: &mut In<'_>) -> Result<PurgeKind> {
    match i.u8()? {
        0 => return Ok(PurgeKind::Entity),
        2 => return Ok(PurgeKind::Ids { count: usize::try_from(i.u64()?).unwrap_or(usize::MAX) }),
        3 => return Ok(PurgeKind::AgeOut { count: usize::try_from(i.u64()?).unwrap_or(usize::MAX) }),
        _ => {}
    }
    let span = interval(i)?;
    let streams = if i.u8()? == 1 {
        let n = i.u32()?;
        Some((0..n).map(|_| i.u32().map(StreamId)).collect::<Result<_>>()?)
    } else {
        None
    };
    Ok(PurgeKind::Range { span, streams })
}

/// A purge's selector as its job keeps it (the entity id included), so a restart can resume it.
pub(crate) fn selector(v: &PurgeSelector) -> Vec<u8> {
    match v {
        PurgeSelector::Entity(e) => Out::new().u8(0).id(*e).done(),
        PurgeSelector::Ids(ids) => {
            let mut o = Out::new().u8(2).u32(u32::try_from(ids.len()).unwrap_or(u32::MAX));
            for id in ids {
                o = o.id(*id);
            }
            o.done()
        }
        PurgeSelector::AgeOut { ids, committed_by_seq, merge_gap_ns } => {
            let mut o = Out::new()
                .u8(3)
                .u64(*committed_by_seq)
                .u64(*merge_gap_ns)
                .u32(u32::try_from(ids.len()).unwrap_or(u32::MAX));
            for id in ids {
                o = o.id(*id);
            }
            o.done()
        }
        PurgeSelector::Range { span, streams } => {
            put_kind(Out::new().u8(1), &PurgeKind::Range { span: *span, streams: streams.clone() }).done()
        }
    }
}

pub(crate) fn read_selector(b: &[u8]) -> Result<PurgeSelector> {
    let mut i = In::new(b);
    match b.first() {
        Some(0) => {
            i.u8()?;
            return Ok(PurgeSelector::Entity(i.id()?));
        }
        Some(2) => {
            i.u8()?;
            let n = i.u32()?;
            return Ok(PurgeSelector::Ids((0..n).map(|_| i.id()).collect::<Result<_>>()?));
        }
        Some(3) => {
            i.u8()?;
            let committed_by_seq = i.u64()?;
            let merge_gap_ns = i.u64()?;
            let n = i.u32()?;
            let ids = (0..n).map(|_| i.id()).collect::<Result<_>>()?;
            return Ok(PurgeSelector::AgeOut { ids, committed_by_seq, merge_gap_ns });
        }
        _ => {}
    }
    i.u8()?;
    match get_kind(&mut i)? {
        PurgeKind::Range { span, streams } => Ok(PurgeSelector::Range { span, streams }),
        _ => Err(corrupt("selector")),
    }
}

pub(crate) fn report(r: &PurgeReport) -> Vec<u8> {
    let mut o = Out::new().id(r.purge_id).u32(u32::try_from(r.removed.len()).unwrap_or(u32::MAX));
    for x in &r.removed {
        o = o.bytes(&removed(x));
    }
    o = o.u32(u32::try_from(r.removed_frames.len()).unwrap_or(u32::MAX));
    for f in &r.removed_frames {
        o = o.id(*f);
    }
    o = o.u32(u32::try_from(r.partial_chunks.len()).unwrap_or(u32::MAX));
    for p in &r.partial_chunks {
        o = o.id(p.chunk).u32(p.stream.0).u32(u32::try_from(p.frames.len()).unwrap_or(u32::MAX));
        for f in &p.frames {
            o = o.u32(*f);
        }
        o = put_locators(o, &p.locations);
    }
    o.done()
}

pub(crate) fn read_report(b: &[u8]) -> Result<PurgeReport> {
    let mut i = In::new(b);
    let purge_id = i.id()?;
    let n = i.u32()?;
    let removed = (0..n).map(|_| read_removed(i.bytes()?)).collect::<Result<_>>()?;
    let n = i.u32()?;
    let removed_frames = (0..n).map(|_| i.id()).collect::<Result<_>>()?;
    let n = i.u32()?;
    let mut partial_chunks = Vec::new();
    for _ in 0..n {
        let chunk = i.id()?;
        let stream = StreamId(i.u32()?);
        let m = i.u32()?;
        let frames = (0..m).map(|_| i.u32()).collect::<Result<_>>()?;
        partial_chunks.push(PartialChunk { chunk, stream, frames, locations: get_locators(&mut i)? });
    }
    Ok(PurgeReport { purge_id, removed, removed_frames, partial_chunks })
}

pub(crate) fn record(r: &PurgeRecord) -> Vec<u8> {
    let o = put_kind(Out::new().u64(r.at_ns), &r.kind).str(&r.reason).str(&r.detail).opt_u64(r.completed_ns);
    let o = match &r.report {
        Some(rep) => o.u8(1).bytes(&report(rep)),
        None => o.u8(0),
    };
    let mut o = o.u32(u32::try_from(r.acked.len()).unwrap_or(u32::MAX));
    for name in &r.acked {
        o = o.str(name);
    }
    o.done()
}

pub(crate) fn read_record(b: &[u8]) -> Result<PurgeRecord> {
    let mut i = In::new(b);
    let at_ns = i.u64()?;
    let kind = get_kind(&mut i)?;
    let reason = i.string()?;
    let detail = i.string()?;
    let completed_ns = i.opt_u64()?;
    let report = if i.u8()? == 1 { Some(read_report(i.bytes()?)?) } else { None };
    let mut acked = Vec::new();
    for _ in 0..i.u32()? {
        acked.push(i.string()?);
    }
    if !i.is_empty() {
        return Err(corrupt("trailing purge record bytes"));
    }
    Ok(PurgeRecord { at_ns, kind, reason, detail, report, completed_ns, acked })
}

pub(crate) fn cut_frame(c: &CutFrame) -> Vec<u8> {
    let mut o = put_interval(Out::new().id(c.chunk).u32(c.index).u32(c.stream.0), c.slot)
        .u32(u32::try_from(c.tiers.len()).unwrap_or(0));
    for t in &c.tiers {
        o = o.u8(t.0);
    }
    o.done()
}

pub(crate) fn read_cut_frame(b: &[u8]) -> Result<CutFrame> {
    let mut i = In::new(b);
    let (chunk, index, stream) = (i.id()?, i.u32()?, StreamId(i.u32()?));
    let slot = interval(&mut i)?;
    let n = i.u32()?;
    let tiers = (0..n).map(|_| i.u8().map(StorageTier)).collect::<Result<_>>()?;
    Ok(CutFrame { chunk, index, stream, slot, tiers })
}

pub(crate) fn doomed(d: &Doomed) -> Vec<u8> {
    let mut o =
        put_interval(Out::new().id(d.id).u32(d.stream.0), d.span).u32(u32::try_from(d.tiers.len()).unwrap_or(0));
    for t in &d.tiers {
        o = o.u8(t.0);
    }
    o.u8(u8::from(d.late)).done()
}

pub(crate) fn read_doomed(b: &[u8]) -> Result<Doomed> {
    let mut i = In::new(b);
    let id = i.id()?;
    let stream = StreamId(i.u32()?);
    let span = interval(&mut i)?;
    let n = i.u32()?;
    let tiers = (0..n).map(|_| i.u8().map(StorageTier)).collect::<Result<_>>()?;
    Ok(Doomed { id, stream, span, tiers, late: i.u8()? == 1 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::strategy::Strategy as _;

    #[test]
    fn records_round_trip() {
        let info = StreamInfo {
            name: "cam0".into(),
            kind: StreamKind::Derived,
            layer: DomainLayer::Layer2Decision,
            format: PayloadFormat::Text,
            shape_name: "x".into(),
            shape_version: 3,
            rate_hz: Some(29.97),
            anchor: Some(ClockAnchor { boot: BootId([4; 16]), mono_ns: 5, utc_ns: 6 }),
        };
        assert_eq!(read_stream_info(&stream_info(&info)).unwrap(), info);
        let fr = vec![ChunkFrame::new(1, 2, 3), ChunkFrame { start_ns: 9, offset: 8, length: 7, cut: true }];
        assert_eq!(read_frames(&frames(&fr)).unwrap(), fr);
        let loc = StorageLocator::Chunk { tier: StorageTier::Tier1Warm, uri: "u".into(), offset: 1, length: 2 };
        let rep = PurgeReport {
            purge_id: Uuid([1; 16]),
            removed: vec![RemovedPayload { id: Uuid([2; 16]), stream: StreamId(3), locations: vec![loc.clone()] }],
            removed_frames: vec![Uuid([5; 16])],
            partial_chunks: vec![PartialChunk {
                chunk: Uuid([6; 16]),
                stream: StreamId(1),
                frames: vec![0, 4],
                locations: vec![loc],
            }],
        };
        let rec = PurgeRecord {
            at_ns: 7,
            kind: PurgeKind::Range { span: Interval::new(1, 2).unwrap(), streams: Some(vec![StreamId(1)]) },
            reason: "why".into(),
            detail: r#"{"stream":"camera:cam0"}"#.into(),
            report: Some(rep),
            completed_ns: Some(9),
            acked: vec!["tier-worker".into(), "vector-worker".into()],
        };
        assert_eq!(read_record(&record(&rec)).unwrap(), rec);
        let sel = PurgeSelector::Entity(Uuid([8; 16]));
        assert_eq!(read_selector(&selector(&sel)).unwrap(), sel);
        let ids = PurgeSelector::Ids(vec![Uuid([1; 16]), Uuid([2; 16])]);
        assert_eq!(read_selector(&selector(&ids)).unwrap(), ids);
        let range = PurgeSelector::Range { span: Interval::new(3, 4).unwrap(), streams: None };
        assert_eq!(read_selector(&selector(&range)).unwrap(), range);
        let age = PurgeSelector::AgeOut { ids: vec![Uuid([3; 16])], committed_by_seq: 41, merge_gap_ns: 0 };
        assert_eq!(read_selector(&selector(&age)).unwrap(), age);
        let runs =
            PurgeSelector::AgeOut { ids: vec![Uuid([3; 16])], committed_by_seq: 41, merge_gap_ns: 2_000_000_000 };
        assert_eq!(read_selector(&selector(&runs)).unwrap(), runs);
        let aged = PurgeRecord { kind: PurgeKind::AgeOut { count: 5 }, ..rec.clone() };
        assert_eq!(read_record(&record(&aged)).unwrap(), aged);
        let by_ids = PurgeRecord { kind: PurgeKind::Ids { count: 2 }, report: None, completed_ns: None, ..rec };
        assert_eq!(read_record(&record(&by_ids)).unwrap(), by_ids);
    }

    #[test]
    fn a_cameras_frame_index_is_a_few_bytes_a_frame() {
        // A minute at 30 fps with capture times that jitter by up to 40 microseconds, records one 45-byte head apart,
        // and access units of a few kilobytes: what a camera's chunk holds.
        let mut at = 24 + 45;
        let mut noise = 0x9e37_79b9_7f4a_7c15u64;
        let fr: Vec<ChunkFrame> = (0..1800u64)
            .map(|n| {
                noise ^= noise << 13;
                noise ^= noise >> 7;
                noise ^= noise << 17;
                let length = 300 + noise % 3_000;
                let start_ns = 1_794_151_800_000_000_000 + n * 33_333_333 + (noise >> 40) % 40_000;
                let f = ChunkFrame { start_ns, offset: at, length, cut: n == 9 };
                at += length + 45;
                f
            })
            .collect();
        let bytes = frames(&fr);
        assert_eq!(read_frames(&bytes).unwrap(), fr);
        assert_eq!(bytes[0], FRAMES_ZSTD);
        assert!(bytes.len() < 5 * fr.len(), "{} bytes for {} frames (25 each fixed-width)", bytes.len(), fr.len());
        // A few frames aren't worth a compressor's frame.
        assert_eq!(frames(&fr[..3])[0], FRAMES_PLAIN);
        assert_eq!(read_frames(&frames(&fr[..3])).unwrap(), fr[..3]);
        assert_eq!(read_frames(&frames(&[])).unwrap(), []);
    }

    #[test]
    fn a_damaged_frame_index_is_refused() {
        let two = [ChunkFrame::new(5, 69, 300), ChunkFrame { cut: true, ..ChunkFrame::new(9, 414, 2) }];
        let bytes = frames(&two);
        assert_eq!(read_frames(&bytes).unwrap(), two);
        assert!(read_frames(&bytes[..bytes.len() - 1]).is_err(), "truncated");
        assert!(read_frames(&[bytes.as_slice(), &[0]].concat()).is_err(), "trailing bytes");
        let mut layout = bytes.clone();
        layout[0] = 9;
        assert!(read_frames(&layout).is_err(), "an unknown layout");
        let mut count = bytes.clone();
        count[1] = 0x7f;
        assert!(read_frames(&count).is_err(), "a count past the index");
        let mut cut = bytes.clone();
        *cut.last_mut().unwrap() = 5;
        assert!(read_frames(&cut).is_err(), "a cut frame past the index");
        assert!(read_frames(&[FRAMES_ZSTD, 4, 1, 2, 3]).is_err(), "not a zstd frame");
        assert!(In::new(&[0xff; 11]).var().is_err(), "a varint over 64 bits");
    }

    proptest::proptest! {
        #[test]
        fn any_frame_index_round_trips(
            fr in proptest::collection::vec(
                (proptest::num::u64::ANY, proptest::num::u64::ANY, proptest::num::u64::ANY, proptest::bool::ANY)
                    .prop_map(|(start_ns, offset, length, cut)| ChunkFrame { start_ns, offset, length, cut }),
                0..200,
            )
        ) {
            proptest::prop_assert_eq!(read_frames(&frames(&fr)).unwrap(), fr);
        }

        #[test]
        fn any_signed_difference_round_trips(v in proptest::num::i64::ANY) {
            let bytes = Out::new().zigzag(v).done();
            proptest::prop_assert_eq!(In::new(&bytes).zigzag().unwrap(), v);
        }
    }
}
