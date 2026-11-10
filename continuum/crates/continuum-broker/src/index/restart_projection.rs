//! Versioned, bounded restart projection. This is a cache codec, not a validity certificate: its caller must
//! atomically bind the bytes to the catalog identity/content epoch and quiesce pending writes and purges.
//! FNV-1a detects accidental corruption; it does not authenticate untrusted data.

use std::collections::{BTreeMap, HashSet};

use super::{AvailabilityIndex, CopySet, Counts, StreamMeta, TIERS};
use continuum_catalog::CoverageSet;
use continuum_core::{Interval, StageId, StreamId, Uuid};
use continuum_sched::StreamKind;

const MAGIC: &[u8; 8] = b"CNTAVP01";
// Bump this when frame expansion, tier counts, or any persisted projection semantics change.
const VERSION: u32 = 1;
const HEADER: usize = 28;
pub use continuum_catalog::MAX_RESTART_PROJECTION_BYTES;
const MAX_STREAMS: usize = 65_536;
const MAX_NAME: usize = 4096;
// The Linux machine had 8.1M segments and grows about 6.5M a day; 8M refused every stop.
const MAX_ROWS: usize = 64_000_000;

/// A restart cache is unusable. Callers must rebuild from the catalog, never substitute an empty index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum RestartProjectionError {
    /// Hidden-on-landing bookkeeping belongs to an unfinished purge, not a clean restart cache.
    #[error("availability projection has hidden payloads")]
    HiddenPayloads,
    /// Historical copy spans cannot be reconstructed under changed stream metadata without a catalog reload.
    #[error("availability projection requires a rebuild after stream metadata changed")]
    MetadataChanged,
    /// Encoded bytes, strings, streams, or interval counts exceed the fixed codec budget.
    #[error("availability projection exceeds its bounds")]
    Bounds,
    /// Corrupt, unsupported, noncanonical, or incomplete state.
    #[error("invalid availability projection: {0}")]
    Invalid(&'static str),
}

type Result<T> = std::result::Result<T, RestartProjectionError>;
fn invalid(reason: &'static str) -> RestartProjectionError {
    RestartProjectionError::Invalid(reason)
}

fn checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |state, byte| (state ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3))
}

struct Writer {
    bytes: Vec<u8>,
    rows: usize,
}
impl Writer {
    fn put(&mut self, bytes: &[u8]) -> Result<()> {
        if self.bytes.len().checked_add(bytes.len()).is_none_or(|len| len > MAX_RESTART_PROJECTION_BYTES) {
            return Err(RestartProjectionError::Bounds);
        }
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }
    fn u32(&mut self, value: usize) -> Result<()> {
        self.put(&u32::try_from(value).map_err(|_| RestartProjectionError::Bounds)?.to_le_bytes())
    }
    fn row(&mut self) -> Result<()> {
        self.rows += 1;
        if self.rows > MAX_ROWS {
            return Err(RestartProjectionError::Bounds);
        }
        Ok(())
    }
    fn interval(&mut self, span: Interval) -> Result<()> {
        self.row()?;
        self.put(&span.start_ns().to_le_bytes())?;
        self.put(&span.end_ns().to_le_bytes())
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
    rows: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self.at.checked_add(len).ok_or(RestartProjectionError::Bounds)?;
        let part = self.bytes.get(self.at..end).ok_or_else(|| invalid("truncated"))?;
        self.at = end;
        Ok(part)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().map_err(|_| invalid("integer"))?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().map_err(|_| invalid("integer"))?))
    }
    fn count(&mut self, max: usize, min_bytes: usize) -> Result<usize> {
        let count = self.u32()? as usize;
        if count > max || count.checked_mul(min_bytes).is_none_or(|n| n > self.bytes.len() - self.at) {
            return Err(RestartProjectionError::Bounds);
        }
        Ok(count)
    }
    fn interval(&mut self) -> Result<Interval> {
        self.rows += 1;
        if self.rows > MAX_ROWS {
            return Err(RestartProjectionError::Bounds);
        }
        Interval::new(self.u64()?, self.u64()?).ok_or_else(|| invalid("interval"))
    }
}

fn rate(bits: u64) -> Result<f64> {
    let value = f64::from_bits(bits);
    if !value.is_finite() || value < 0.0 {
        return Err(invalid("rate"));
    }
    Ok(value)
}

fn durable_segments(copies: &CopySet, each: &mut impl FnMut(Interval, Counts) -> Result<()>) -> Result<()> {
    let mut pending: Option<(u64, u64, Counts)> = None;
    for (&start, &(end, mut counts)) in &copies.segs {
        counts[0] = 0;
        if counts == [0; TIERS] {
            continue;
        }
        if let Some((a, b, previous)) = pending {
            if b == start && previous == counts {
                pending = Some((a, end, counts));
                continue;
            }
            each(Interval::new(a, b).ok_or_else(|| invalid("interval"))?, previous)?;
        }
        pending = Some((start, end, counts));
    }
    if let Some((a, b, counts)) = pending {
        each(Interval::new(a, b).ok_or_else(|| invalid("interval"))?, counts)?;
    }
    Ok(())
}

impl AvailabilityIndex {
    /// Whether hidden-on-landing bookkeeping and historical metadata permit a clean checkpoint. The core must also
    /// establish its own write/purge/barrier fence; size and structure are validated during encoding.
    #[must_use]
    pub fn restart_checkpoint_clean(&self) -> bool {
        self.restart_checkpoint_refusal().is_none()
    }

    /// Why hidden-on-landing bookkeeping or historical metadata refuses a checkpoint, or `None` when it doesn't.
    #[must_use]
    pub fn restart_checkpoint_refusal(&self) -> Option<crate::core::CheckpointRefusal> {
        if !self.hidden_on_landing.is_empty() {
            Some(crate::core::CheckpointRefusal::HiddenOnLanding)
        } else if self.restart_metadata_changed {
            Some(crate::core::CheckpointRefusal::StreamMetadataChanged)
        } else {
            None
        }
    }

    /// Encode durable restart availability, excluding all old-arena Tier 0 counts. Names, exact rate bits,
    /// durable multiplicities, stage coverage, and individual cut slots are retained. No I/O is performed.
    ///
    /// The caller must verify catalog ownership of the cut bookkeeping and a matching content epoch, and must
    /// refuse pending purges, outstanding writes, and physical barriers before publishing these bytes.
    ///
    /// # Errors
    /// Hidden payloads, changed historical metadata, nonfinite/negative rates, missing/ambiguous names, or codec bounds.
    /// No partial cache returns.
    pub fn restart_projection_bytes(&self) -> Result<Vec<u8>> {
        if !self.hidden_on_landing.is_empty() {
            return Err(RestartProjectionError::HiddenPayloads);
        }
        if self.restart_metadata_changed {
            return Err(RestartProjectionError::MetadataChanged);
        }
        if self.streams.len() > MAX_STREAMS || self.names.len() != self.streams.len() {
            return Err(invalid("stream names"));
        }
        let mut names = BTreeMap::new();
        for (name, &id) in &self.names {
            if name.is_empty()
                || name.len() > MAX_NAME
                || !self.streams.contains_key(&id)
                || names.insert(id, name).is_some()
            {
                return Err(invalid("stream names"));
            }
        }
        let mut out = Writer { bytes: vec![0; HEADER], rows: 0 };
        out.u32(self.streams.len())?;
        let mut streams: Vec<_> = self.streams.iter().collect();
        streams.sort_by_key(|(id, _)| id.0);
        for (&id, (meta, copies)) in streams {
            out.put(&id.0.to_le_bytes())?;
            out.put(&[u8::from(meta.kind == StreamKind::Derived)])?;
            rate(meta.rate_hz.to_bits())?;
            out.put(&meta.rate_hz.to_bits().to_le_bytes())?;
            let name = names.get(&id).ok_or_else(|| invalid("missing stream name"))?;
            out.u32(name.len())?;
            out.put(name.as_bytes())?;
            let count_at = out.bytes.len();
            out.u32(0)?;
            let mut count = 0_u32;
            durable_segments(copies, &mut |span, counts| {
                out.interval(span)?;
                out.put(&counts[1].to_le_bytes())?;
                out.put(&counts[2].to_le_bytes())?;
                count += 1;
                Ok(())
            })?;
            out.bytes[count_at..count_at + 4].copy_from_slice(&count.to_le_bytes());
        }
        let mut coverage: Vec<_> = self.coverage.iter().collect();
        coverage.sort_by_key(|((source, stage), _)| (source.0, stage.0));
        out.u32(coverage.len())?;
        for (&(source, stage), set) in coverage {
            if !self.streams.contains_key(&source) || set.is_empty() {
                return Err(invalid("coverage source"));
            }
            out.put(&source.0.to_le_bytes())?;
            out.put(&stage.0.to_le_bytes())?;
            out.u32(set.iter().count())?;
            for span in set.iter() {
                out.interval(span)?;
            }
        }
        let mut chunks: Vec<_> = self.cuts.iter().collect();
        chunks.sort_by_key(|(id, _)| id.0);
        out.u32(chunks.len())?;
        for (&id, slots) in chunks {
            if id == Uuid([0; 16]) || slots.is_empty() {
                return Err(invalid("cut chunk"));
            }
            let mut slots = slots.clone();
            slots.sort();
            if slots.windows(2).any(|pair| pair[0].end_ns() > pair[1].start_ns()) {
                return Err(invalid("cut ordering"));
            }
            out.put(&id.0)?;
            out.u32(slots.len())?;
            for span in slots {
                out.interval(span)?;
            }
        }
        let body_len = (out.bytes.len() - HEADER) as u64;
        let digest = checksum(&out.bytes[HEADER..]);
        out.bytes[..8].copy_from_slice(MAGIC);
        out.bytes[8..12].copy_from_slice(&VERSION.to_le_bytes());
        out.bytes[12..20].copy_from_slice(&body_len.to_le_bytes());
        out.bytes[20..28].copy_from_slice(&digest.to_le_bytes());
        Ok(out.bytes)
    }

    /// Restore an entire validated durable projection. This performs no catalog reads and does not prove that
    /// the cache belongs to the current catalog: its caller must first validate the external identity/epoch seal.
    ///
    /// # Errors
    /// Bad version/checksum, size/count bounds, missing/duplicate keys, invalid rates/intervals, noncanonical
    /// ordering or normalization, trailing bytes. Never returns a partial index.
    pub fn restore_restart_projection(bytes: &[u8]) -> Result<Self> {
        if bytes.len() > MAX_RESTART_PROJECTION_BYTES {
            return Err(RestartProjectionError::Bounds);
        }
        let mut envelope = Reader { bytes, at: 0, rows: 0 };
        if envelope.take(8)? != MAGIC {
            return Err(invalid("magic"));
        }
        if envelope.u32()? != VERSION {
            return Err(invalid("version"));
        }
        let size = usize::try_from(envelope.u64()?).map_err(|_| RestartProjectionError::Bounds)?;
        let digest = envelope.u64()?;
        if size != bytes.len() - HEADER || checksum(&bytes[HEADER..]) != digest {
            return Err(invalid("length or checksum"));
        }
        let mut input = Reader { bytes: &bytes[HEADER..], at: 0, rows: 0 };
        let mut index = Self::new();
        let mut previous = None;
        let mut names = HashSet::new();
        let count = input.count(MAX_STREAMS, 21)?;
        for _ in 0..count {
            let id = StreamId(input.u32()?);
            if previous.is_some_and(|last| last >= id.0) {
                return Err(invalid("stream ordering"));
            }
            previous = Some(id.0);
            let kind = match input.take(1)?[0] {
                0 => StreamKind::Sensor,
                1 => StreamKind::Derived,
                _ => return Err(invalid("stream kind")),
            };
            let meta = StreamMeta { kind, rate_hz: rate(input.u64()?)? };
            let len = input.count(MAX_NAME, 1)?;
            let name = std::str::from_utf8(input.take(len)?).map_err(|_| invalid("name UTF8"))?.to_owned();
            if name.is_empty() || !names.insert(name.clone()) {
                return Err(invalid("stream name"));
            }
            let mut copies = CopySet::new();
            let mut last: Option<(u64, Counts)> = None;
            let segments = input.count(MAX_ROWS, 24)?;
            for _ in 0..segments {
                let span = input.interval()?;
                let counts = [0, input.u32()?, input.u32()?];
                if counts == [0; TIERS]
                    || last.is_some_and(|(end, prior)| {
                        end > span.start_ns() || (end == span.start_ns() && counts == prior)
                    })
                {
                    return Err(invalid("segment normalization"));
                }
                copies.segs.insert(span.start_ns(), (span.end_ns(), counts));
                last = Some((span.end_ns(), counts));
            }
            index.streams.insert(id, (meta, copies));
            index.names.insert(name, id);
        }
        let count = input.count(MAX_ROWS, 12)?;
        let mut previous = None;
        for _ in 0..count {
            let source = StreamId(input.u32()?);
            let stage = StageId(input.u32()?);
            let key = (source.0, stage.0);
            if !index.streams.contains_key(&source) || previous.is_some_and(|last| last >= key) {
                return Err(invalid("coverage keys"));
            }
            previous = Some(key);
            let count = input.count(MAX_ROWS, 16)?;
            if count == 0 {
                return Err(invalid("empty coverage"));
            }
            let mut set = CoverageSet::new();
            let mut end = None;
            for _ in 0..count {
                let span = input.interval()?;
                if end.is_some_and(|last| last >= span.start_ns()) {
                    return Err(invalid("coverage normalization"));
                }
                set.insert(span);
                end = Some(span.end_ns());
            }
            index.coverage.insert((source, stage), set);
        }
        let count = input.count(MAX_ROWS, 20)?;
        let mut previous = None;
        for _ in 0..count {
            let id = Uuid(input.take(16)?.try_into().map_err(|_| invalid("UUID"))?);
            if id == Uuid([0; 16]) || previous.is_some_and(|last| last >= id) {
                return Err(invalid("cut keys"));
            }
            previous = Some(id);
            let count = input.count(MAX_ROWS, 16)?;
            if count == 0 {
                return Err(invalid("empty cuts"));
            }
            let mut slots = Vec::with_capacity(count);
            let mut end = None;
            for _ in 0..count {
                let span = input.interval()?;
                if end.is_some_and(|last| last > span.start_ns()) {
                    return Err(invalid("cut ordering"));
                }
                slots.push(span);
                end = Some(span.end_ns());
            }
            index.cuts.insert(id, slots);
        }
        if input.at != input.bytes.len() {
            return Err(invalid("trailing bytes"));
        }
        Ok(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use continuum_core::StorageTier;
    use proptest::prelude::*;

    fn iv(a: u64, b: u64) -> Interval {
        Interval::new(a, b).unwrap()
    }
    fn register(index: &mut AvailabilityIndex, id: u32, name: &str, rate_hz: f64) {
        index.register(StreamId(id), StreamMeta { kind: StreamKind::Sensor, rate_hz });
        index.name_stream(name.into(), StreamId(id));
    }
    fn fixture() -> AvailabilityIndex {
        let mut index = AvailabilityIndex::new();
        register(&mut index, 1, "camera.video", 30.0);
        register(&mut index, 2, "derived", -0.0);
        for (a, b, tier) in [(0, 10, 1), (0, 10, 1), (10, 20, 1), (0, 20, 2), (5, 15, 0), (30, 40, 0)] {
            index.add_copy(StreamId(1), iv(a, b), StorageTier(tier)).unwrap();
        }
        index.add_coverage(StreamId(1), StageId(7), iv(0, 10));
        index.add_coverage(StreamId(1), StageId(7), iv(20, 30));
        index.record_cut(Uuid([1; 16]), iv(12, 15));
        index.record_cut(Uuid([1; 16]), iv(10, 12));
        index
    }
    fn reseal(bytes: &mut [u8]) {
        let digest = checksum(&bytes[HEADER..]);
        bytes[20..28].copy_from_slice(&digest.to_le_bytes());
    }
    #[test]
    fn restart_codec_preserves_names_rate_bits_multiplicity_coverage_and_individual_cuts() {
        let original = fixture();
        let bytes = original.restart_projection_bytes().unwrap();
        let mut restored = AvailabilityIndex::restore_restart_projection(&bytes).unwrap();
        assert_eq!(restored.name_of(StreamId(1)).as_deref(), Some("camera.video"));
        assert_eq!(restored.stream_named("derived"), Some(StreamId(2)));
        assert_eq!(restored.meta(StreamId(2)).unwrap().rate_hz.to_bits(), (-0.0_f64).to_bits());
        assert_eq!(restored.streams[&StreamId(1)].1.segments(), [(0, 10, [0, 2, 1]), (10, 20, [0, 1, 1])]);
        assert_eq!(restored.cuts[&Uuid([1; 16])], [iv(10, 12), iv(12, 15)]);
        assert_eq!(restored.coverage_gaps(StreamId(1), StageId(7), iv(0, 30), 8), [iv(10, 20)]);
        restored.remove_copy(StreamId(1), iv(0, 10), StorageTier::Tier1Warm).unwrap();
        assert_eq!(restored.streams[&StreamId(1)].1.segments(), [(0, 20, [0, 1, 1])]);
    }
    #[test]
    fn stripping_hot_counts_merges_durable_neighbors_and_drops_hot_only_segments() {
        let mut index = AvailabilityIndex::new();
        register(&mut index, 1, "x", 0.0);
        index.add_copy(StreamId(1), iv(0, 30), StorageTier::Tier1Warm).unwrap();
        index.add_copy(StreamId(1), iv(10, 20), StorageTier::Tier0Hot).unwrap();
        index.add_copy(StreamId(1), iv(30, 40), StorageTier::Tier0Hot).unwrap();
        let back = AvailabilityIndex::restore_restart_projection(&index.restart_projection_bytes().unwrap()).unwrap();
        assert_eq!(back.streams[&StreamId(1)].1.segments(), [(0, 30, [0, 1, 0])]);
        assert_eq!(index.streams[&StreamId(1)].1.len(), 4, "encoding never mutates live counts");
    }
    #[test]
    fn empty_and_maximum_endpoint_projections_are_canonical() {
        let empty = AvailabilityIndex::new().restart_projection_bytes().unwrap();
        assert_eq!(
            AvailabilityIndex::restore_restart_projection(&empty).unwrap().snapshot(),
            AvailabilityIndex::new().snapshot()
        );
        let mut index = AvailabilityIndex::new();
        register(&mut index, 1, "max", 0.0);
        index.add_copy(StreamId(1), iv(u64::MAX - 1, u64::MAX), StorageTier::Tier2Cold).unwrap();
        let back = AvailabilityIndex::restore_restart_projection(&index.restart_projection_bytes().unwrap()).unwrap();
        assert_eq!(back.snapshot(), index.snapshot());
    }
    #[test]
    fn ephemeral_hidden_state_and_incomplete_names_cannot_be_published() {
        let mut index = fixture();
        index.note_hidden_on_landing(Uuid([9; 16]));
        assert!(!index.restart_checkpoint_clean());
        assert_eq!(index.restart_projection_bytes(), Err(RestartProjectionError::HiddenPayloads));
        let mut index = fixture();
        index.names.clear();
        assert!(index.restart_projection_bytes().is_err());
        let mut index = fixture();
        index.streams.get_mut(&StreamId(1)).unwrap().0.rate_hz = f64::NAN;
        assert!(index.restart_projection_bytes().is_err());
    }
    #[test]
    fn truncation_checksum_future_version_and_trailing_data_fail_closed() {
        let bytes = fixture().restart_projection_bytes().unwrap();
        for end in 0..bytes.len() {
            assert!(AvailabilityIndex::restore_restart_projection(&bytes[..end]).is_err());
        }
        let mut bad = bytes.clone();
        bad[8..12].copy_from_slice(&2_u32.to_le_bytes());
        assert!(AvailabilityIndex::restore_restart_projection(&bad).is_err());
        let mut bad = bytes.clone();
        bad[HEADER + 10] ^= 1;
        assert!(AvailabilityIndex::restore_restart_projection(&bad).is_err());
        let mut bad = bytes.clone();
        bad.extend_from_slice(&[0]);
        let len = (bad.len() - HEADER) as u64;
        bad[12..20].copy_from_slice(&len.to_le_bytes());
        reseal(&mut bad);
        assert!(AvailabilityIndex::restore_restart_projection(&bad).is_err());
        let mut bad = bytes;
        bad[HEADER..HEADER + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        reseal(&mut bad);
        assert!(AvailabilityIndex::restore_restart_projection(&bad).is_err());
    }
    #[test]
    fn checksum_valid_malformed_rates_counts_ordering_and_names_are_refused() {
        let mut index = AvailabilityIndex::new();
        register(&mut index, 1, "x", 30.0);
        index.add_copy(StreamId(1), iv(10, 20), StorageTier::Tier1Warm).unwrap();
        let bytes = index.restart_projection_bytes().unwrap();
        // Body: count4,id4,kind1,rate8,name-length4,name1,segment-count4,span16,counts8.
        for (offset, replacement) in [
            (9, f64::INFINITY.to_bits().to_le_bytes().to_vec()),
            (26, 20_u64.to_le_bytes().to_vec()),
            (42, vec![0; 8]),
            (17, 0_u32.to_le_bytes().to_vec()),
        ] {
            let mut bad = bytes.clone();
            bad[HEADER + offset..HEADER + offset + replacement.len()].copy_from_slice(&replacement);
            reseal(&mut bad);
            assert!(AvailabilityIndex::restore_restart_projection(&bad).is_err(), "offset {offset}");
        }
    }

    #[test]
    fn existing_stream_kind_change_requires_a_full_rebuild_before_restart_checkpoint() {
        let mut index = AvailabilityIndex::new();
        register(&mut index, 1, "source", 10.0);
        index.add_copy(StreamId(1), iv(100, 200), StorageTier::Tier1Warm).unwrap();
        index.register(StreamId(1), StreamMeta { kind: StreamKind::Derived, rate_hz: 10.0 });
        assert!(
            !index.restart_checkpoint_clean(),
            "changing an existing stream kind must invalidate restart projection eligibility"
        );
        assert!(index.restart_projection_bytes().is_err());
    }

    #[test]
    fn checksum_valid_duplicate_stream_keys_names_and_unmerged_segments_are_refused() {
        let mut index = AvailabilityIndex::new();
        register(&mut index, 1, "x", 0.0);
        register(&mut index, 2, "y", 0.0);
        let bytes = index.restart_projection_bytes().unwrap();
        // Each empty stream has 22 bytes (id4,kind1,rate8,name-length4,name1,segment-count4).
        let mut duplicate = bytes.clone();
        duplicate[HEADER + 26..HEADER + 30].copy_from_slice(&1_u32.to_le_bytes());
        reseal(&mut duplicate);
        assert!(AvailabilityIndex::restore_restart_projection(&duplicate).is_err());
        let mut duplicate = bytes;
        duplicate[HEADER + 43] = b'x';
        reseal(&mut duplicate);
        assert!(AvailabilityIndex::restore_restart_projection(&duplicate).is_err());
        let mut index = AvailabilityIndex::new();
        register(&mut index, 1, "x", 0.0);
        index.add_copy(StreamId(1), iv(0, 10), StorageTier::Tier1Warm).unwrap();
        index.add_copy(StreamId(1), iv(20, 30), StorageTier::Tier1Warm).unwrap();
        let mut bytes = index.restart_projection_bytes().unwrap();
        bytes[HEADER + 50..HEADER + 58].copy_from_slice(&10_u64.to_le_bytes());
        reseal(&mut bytes);
        assert!(
            AvailabilityIndex::restore_restart_projection(&bytes).is_err(),
            "equal touching segments must already have merged"
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]
        #[test]
        fn restart_roundtrip_matches_durable_only_multiset(
            rows in proptest::collection::vec((0u64..100,1u64..20,0u8..5),0..80),
        ) {
            let mut live=AvailabilityIndex::new();let mut expected=AvailabilityIndex::new();
            register(&mut live,1,"camera",25.0);register(&mut expected,1,"camera",25.0);
            for (start,len,tier) in rows {
                live.add_copy(StreamId(1),iv(start,start+len),StorageTier(tier)).unwrap();
                if tier!=0 {expected.add_copy(StreamId(1),iv(start,start+len),StorageTier(tier)).unwrap();}
            }
            let bytes=live.restart_projection_bytes().unwrap();let restored=AvailabilityIndex::restore_restart_projection(&bytes).unwrap();
            prop_assert_eq!(restored.snapshot(),expected.snapshot());
            prop_assert_eq!(restored.restart_projection_bytes().unwrap(),bytes);
        }
        #[test]
        fn arbitrary_bytes_are_rejected_or_roundtrip_canonically(bytes in proptest::collection::vec(any::<u8>(),0..2048)) {
            if let Ok(index)=AvailabilityIndex::restore_restart_projection(&bytes) {prop_assert_eq!(index.restart_projection_bytes().unwrap(),bytes);}
        }
    }
}

#[cfg(test)]
mod catalog_parity_tests {
    use super::*;
    use continuum_catalog::{
        Catalog, ChunkFrame, CoverageMark, NewChunk, NewPayload, PurgeProgress, PurgeSelector, RedbCatalog, StreamInfo,
        StreamKind as CatalogKind,
    };
    use continuum_core::{DomainLayer, HeaderBuf, HeaderSpec, PayloadFormat, StorageLocator, StorageTier};

    fn iv(a: u64, b: u64) -> Interval {
        Interval::new(a, b).unwrap()
    }
    fn header(id: Uuid, stream: StreamId, start: u64, end: u64) -> HeaderBuf {
        let mut spec = HeaderSpec::raw(id, stream, start, end);
        // The chunk and an independent coverage payload share start zero. Catalog time keys include seq.
        spec.seq = u32::from(id.0[0]);
        HeaderBuf::build(&spec).unwrap()
    }
    fn location(tier: StorageTier) -> StorageLocator {
        StorageLocator::Chunk { tier, uri: format!("synthetic-{}", tier.0), offset: 0, length: 100 }
    }
    fn parity(catalog: &dyn Catalog, live: &AvailabilityIndex) {
        let fresh = crate::index::load(catalog, 1).unwrap().0;
        let restored =
            AvailabilityIndex::restore_restart_projection(&live.restart_projection_bytes().unwrap()).unwrap();
        assert_eq!(restored.snapshot(), fresh.snapshot());
        assert_eq!(restored.names, fresh.names);
        assert_eq!(restored.restart_projection_bytes().unwrap(), fresh.restart_projection_bytes().unwrap());
    }
    fn purge(catalog: &dyn Catalog, live: &mut AvailabilityIndex, selector: PurgeSelector, id: Uuid) {
        catalog.start_purge(selector, "synthetic", "", id, 1).unwrap();
        loop {
            match catalog.purge_step(id, 2).unwrap() {
                PurgeProgress::Running { delta, .. } => {
                    assert_eq!(live.apply_purge_delta(&delta), 0);
                }
                PurgeProgress::Removed { delta, .. } => {
                    assert_eq!(live.apply_purge_delta(&delta), 0);
                    break;
                }
            }
        }
    }
    #[test]
    fn redb_rate_change_cannot_checkpoint_old_instant_expansion_under_new_metadata() {
        let temp = tempfile::tempdir().unwrap();
        let catalog = RedbCatalog::open(temp.path().join("rate.redb")).unwrap();
        let mut info = StreamInfo {
            name: "camera.video".into(),
            kind: CatalogKind::Sensor,
            layer: DomainLayer::Layer1Raw,
            format: PayloadFormat::RawBytes,
            shape_name: "".into(),
            shape_version: 0,
            rate_hz: Some(10.0),
            anchor: None,
        };
        let stream = catalog.register_stream(info.clone()).unwrap();
        catalog
            .insert(vec![NewPayload {
                header: header(Uuid([11; 16]), stream, 100, 100),
                location: Some(StorageLocator::Inline { bytes: vec![1] }),
                coverage: None,
            }])
            .unwrap();
        let mut live = crate::index::load(&catalog, 1).unwrap().0;
        assert_eq!(live.covered(stream, iv(0, 200_000_000))[0].interval, iv(100, 100_000_100));
        assert_eq!(catalog.register_stream(info.clone()).unwrap(), stream);
        live.register(stream, StreamMeta { kind: StreamKind::Sensor, rate_hz: 10.0 });
        assert!(live.restart_checkpoint_clean(), "unchanged producer re-registration must remain eligible");
        parity(&catalog, &live);
        info.rate_hz = Some(20.0);
        assert_eq!(catalog.register_stream(info).unwrap(), stream);
        live.register(stream, StreamMeta { kind: StreamKind::Sensor, rate_hz: 20.0 });
        let fresh = crate::index::load(&catalog, 1).unwrap().0;
        assert_eq!(fresh.covered(stream, iv(0, 200_000_000))[0].interval, iv(100, 50_000_100));
        assert_ne!(
            live.snapshot(),
            fresh.snapshot(),
            "runtime register intentionally does not retime historical counts"
        );
        assert!(
            !live.restart_checkpoint_clean(),
            "old counts under a new rate cannot be persisted as fresh-start availability"
        );
        assert!(live.restart_projection_bytes().is_err());
        catalog.close(std::time::Duration::from_secs(5)).unwrap();
    }

    #[test]
    fn redb_restart_projection_matches_fresh_load_after_tier_moves_coverage_and_completed_purges() {
        let temp = tempfile::tempdir().unwrap();
        let catalog = RedbCatalog::open(temp.path().join("synthetic.redb")).unwrap();
        let cam = catalog
            .register_stream(StreamInfo {
                name: "camera.video".into(),
                kind: CatalogKind::Sensor,
                layer: DomainLayer::Layer1Raw,
                format: PayloadFormat::RawBytes,
                shape_name: "".into(),
                shape_version: 0,
                rate_hz: None,
                anchor: None,
            })
            .unwrap();
        let chunk = Uuid([1; 16]);
        let slots = [iv(10, 20)];
        let frames = vec![
            ChunkFrame::new(0, 0, 10),
            ChunkFrame { start_ns: 10, offset: 0, length: 0, cut: true },
            ChunkFrame::new(20, 20, 10),
        ];
        let absorbed = [Uuid([11; 16]), Uuid([12; 16]), Uuid([13; 16])];
        catalog
            .insert(
                absorbed
                    .iter()
                    .enumerate()
                    .map(|(i, &id)| NewPayload {
                        header: header(id, cam, i as u64 * 10, (i as u64 + 1) * 10),
                        location: None,
                        coverage: None,
                    })
                    .collect(),
            )
            .unwrap();
        catalog
            .commit_chunk(
                NewChunk {
                    payload: NewPayload {
                        header: header(chunk, cam, 0, 30),
                        location: Some(location(StorageTier::Tier1Warm)),
                        coverage: None,
                    },
                    frames: frames.clone(),
                },
                &absorbed,
            )
            .unwrap();
        let mut live = crate::index::load(&catalog, 1).unwrap().0;
        parity(&catalog, &live);
        catalog.rewrite_chunk(chunk, frames, location(StorageTier::Tier2Cold)).unwrap();
        live.add_chunk_copy(chunk, cam, iv(0, 30), StorageTier::Tier2Cold, &slots).unwrap();
        parity(&catalog, &live);
        catalog.clear_location(chunk, StorageTier::Tier1Warm).unwrap();
        live.remove_copy(cam, iv(0, 10), StorageTier::Tier1Warm).unwrap();
        live.remove_copy(cam, iv(20, 30), StorageTier::Tier1Warm).unwrap();
        parity(&catalog, &live);
        catalog
            .insert(vec![NewPayload {
                header: header(Uuid([2; 16]), cam, 0, 30),
                location: Some(StorageLocator::Inline { bytes: vec![1] }),
                coverage: Some(CoverageMark { source: cam, stage: StageId(7), span: iv(0, 30) }),
            }])
            .unwrap();
        live.add_copy(cam, iv(0, 30), StorageTier::Tier1Warm).unwrap();
        live.add_coverage(cam, StageId(7), iv(0, 30));
        parity(&catalog, &live);
        catalog
            .insert(vec![NewPayload {
                header: header(Uuid([3; 16]), cam, 100, 100),
                location: Some(StorageLocator::Shm {
                    arena: "synthetic".into(),
                    arena_generation: 1,
                    slab_class: 0,
                    slot: 1,
                    slot_generation: 1,
                    offset: 0,
                    length: 1,
                }),
                coverage: None,
            }])
            .unwrap();
        live.add_copy(cam, Interval::covering(100, 100).unwrap(), StorageTier::Tier0Hot).unwrap();
        parity(&catalog, &live);
        purge(&catalog, &mut live, PurgeSelector::Range { streams: Some(vec![cam]), span: iv(20, 21) }, Uuid([4; 16]));
        parity(&catalog, &live);
        purge(&catalog, &mut live, PurgeSelector::Ids(vec![chunk]), Uuid([5; 16]));
        assert!(!live.cuts.contains_key(&chunk), "a fully doomed chunk must not leave orphan cut bookkeeping");
        parity(&catalog, &live);
        catalog.close(std::time::Duration::from_secs(5)).unwrap();
    }
}
