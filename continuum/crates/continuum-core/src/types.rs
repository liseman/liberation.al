//! Owned types shared by the catalog, the scheduler, and the broker: stage and boot ids, time intervals, a validated
//! payload header buffer, and storage locators.

use std::fmt;

use crate::schema::continuum as fb;
use crate::{DomainLayer, PayloadFormat, StorageTier, StreamId, Uuid};

/// A processing stage (a worker's capability over a source stream), as coverage is kept per (source stream, stage).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StageId(pub u32);

/// A host's boot id: Linux's `/proc/sys/kernel/random/boot_id`, or the macOS boot session UUID. A random UUID, not a
/// UUIDv7. The monotonic clock restarts at every boot, so a monotonic time means nothing without it (decision 9).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BootId(pub [u8; 16]);

impl From<BootId> for fb::Uuid {
    fn from(id: BootId) -> Self {
        Self::new(&id.0)
    }
}

impl From<&fb::Uuid> for BootId {
    fn from(id: &fb::Uuid) -> Self {
        Self(id.bytes().into())
    }
}

impl fmt::Display for BootId {
    /// The hyphenated lowercase form, as the OS prints it.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Uuid(self.0).fmt(f)
    }
}

/// A half-open span of UTC nanoseconds, `[start_ns, end_ns)`, never empty.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Interval {
    start_ns: u64,
    end_ns: u64,
}

impl Interval {
    /// `[start_ns, end_ns)`, or `None` when it would be empty (`end_ns <= start_ns`).
    #[must_use]
    pub const fn new(start_ns: u64, end_ns: u64) -> Option<Self> {
        if end_ns > start_ns { Some(Self { start_ns, end_ns }) } else { None }
    }

    /// The first nanosecond in the span.
    #[must_use]
    pub const fn start_ns(self) -> u64 {
        self.start_ns
    }

    /// The first nanosecond after the span.
    #[must_use]
    pub const fn end_ns(self) -> u64 {
        self.end_ns
    }

    /// Its length in nanoseconds (at least 1).
    #[must_use]
    pub const fn len_ns(self) -> u64 {
        self.end_ns - self.start_ns
    }

    /// Whether `t_ns` falls inside it.
    #[must_use]
    pub const fn contains(self, t_ns: u64) -> bool {
        self.start_ns <= t_ns && t_ns < self.end_ns
    }

    /// Whether the two share at least one nanosecond.
    #[must_use]
    pub const fn overlaps(self, other: Self) -> bool {
        self.start_ns < other.end_ns && other.start_ns < self.end_ns
    }

    /// The span a payload covers, from its header's `start_ns` and `end_ns`: `[start_ns, end_ns)`, or for an instant
    /// (`end_ns == start_ns`) the one nanosecond `[start_ns, start_ns + 1)`. `None` when `end_ns < start_ns`, or for
    /// an instant at `u64::MAX`, which has no next nanosecond.
    #[must_use]
    pub const fn covering(start_ns: u64, end_ns: u64) -> Option<Self> {
        if end_ns == start_ns {
            match start_ns.checked_add(1) {
                Some(end) => Some(Self { start_ns, end_ns: end }),
                None => None,
            }
        } else {
            Self::new(start_ns, end_ns)
        }
    }
}

/// Why bytes aren't a usable header or locator.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DecodeError {
    /// Not a valid FlatBuffer of the expected table.
    Invalid(flatbuffers::InvalidFlatbuffer),
    /// A payload header without an id.
    MissingId,
    /// A header whose `end_ns` is before its `start_ns`.
    EndBeforeStart,
    /// A locator without a location, with a location that doesn't match its tier (shared memory is Tier 0 only,
    /// chunks are Tier 1 or 2 only), without its arena name or chunk uri, or a shared memory offset that isn't a
    /// multiple of 16,384.
    BadLocation,
    /// An inline location with more than [`MAX_INLINE_BYTES`] bytes.
    InlineTooLarge,
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(e) => write!(f, "invalid FlatBuffer: {e}"),
            Self::MissingId => f.write_str("payload header has no id"),
            Self::EndBeforeStart => f.write_str("payload header ends before it starts"),
            Self::BadLocation => f.write_str("storage locator is incomplete or doesn't match its tier"),
            Self::InlineTooLarge => {
                write!(f, "inline bytes are over {MAX_INLINE_BYTES} bytes: put the payload in shared memory or a chunk")
            }
        }
    }
}

impl std::error::Error for DecodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Invalid(e) => Some(e),
            _ => None,
        }
    }
}

/// A `PayloadHeader` as owned FlatBuffer bytes, checked once when it's made: a valid buffer, with an id, and ending
/// no earlier than it starts. Headers cross between the broker's parts in this form.
///
/// Equality and hashing compare the bytes, so two encodings of the same fields differ; compare [`HeaderBuf::id`] to
/// ask whether two are the same payload.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct HeaderBuf(Vec<u8>);

impl HeaderBuf {
    /// Checks `bytes` and keeps them.
    ///
    /// # Errors
    /// [`DecodeError`] when they aren't a valid header with an id and `end_ns >= start_ns`.
    pub fn new(bytes: Vec<u8>) -> Result<Self, DecodeError> {
        let h = fb::root_as_payload_header(&bytes).map_err(DecodeError::Invalid)?;
        if h.id().is_none() {
            return Err(DecodeError::MissingId);
        }
        if h.end_ns() < h.start_ns() {
            return Err(DecodeError::EndBeforeStart);
        }
        Ok(Self(bytes))
    }

    /// Builds a header from the fields most callers set (see [`HeaderSpec`]).
    ///
    /// # Errors
    /// [`DecodeError::EndBeforeStart`] when `spec.end_ns < spec.start_ns`.
    pub fn build(spec: &HeaderSpec) -> Result<Self, DecodeError> {
        let mut b = flatbuffers::FlatBufferBuilder::new();
        let shape = b.create_string(&spec.shape_name);
        let ids = |ids: &[Uuid]| ids.iter().map(|&u| fb::Uuid::from(u)).collect::<Vec<_>>();
        let parents = b.create_vector(&ids(&spec.parents));
        let subjects = b.create_vector(&ids(&spec.subjects));
        let id = fb::Uuid::from(spec.id);
        let header = fb::PayloadHeader::create(
            &mut b,
            &fb::PayloadHeaderArgs {
                id: Some(&id),
                stream_id: spec.stream.0,
                start_ns: spec.start_ns,
                end_ns: spec.end_ns,
                seq: spec.seq,
                layer: spec.layer,
                format: spec.format,
                shape_name: Some(shape),
                shape_version: spec.shape_version,
                parents: Some(parents),
                subjects: Some(subjects),
                source_tier: spec.source_tier,
                ..Default::default()
            },
        );
        fb::finish_payload_header_buffer(&mut b, header);
        Self::new(b.finished_data().to_vec())
    }

    /// The header, read in place.
    #[must_use]
    pub fn header(&self) -> fb::PayloadHeader<'_> {
        // SAFETY: the bytes were verified as a PayloadHeader in `new`, and a HeaderBuf never changes them.
        unsafe { flatbuffers::root_unchecked::<fb::PayloadHeader<'_>>(&self.0) }
    }

    /// The FlatBuffer bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// The bytes, giving up the buffer.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.0
    }

    /// The payload's id.
    #[must_use]
    pub fn id(&self) -> Uuid {
        Uuid::from(self.header().id().expect("HeaderBuf::new checked the id is there"))
    }

    /// The stream it belongs to.
    #[must_use]
    pub fn stream(&self) -> StreamId {
        StreamId(self.header().stream_id())
    }

    /// The first nanosecond it covers.
    #[must_use]
    pub fn start_ns(&self) -> u64 {
        self.header().start_ns()
    }

    /// Where its span ends (equal to `start_ns` for an instant; see [`Interval::covering`]).
    #[must_use]
    pub fn end_ns(&self) -> u64 {
        self.header().end_ns()
    }

    /// The span it covers, as [`Interval::covering`] reads `start_ns` and `end_ns`. `None` only for an instant at
    /// `u64::MAX`.
    #[must_use]
    pub fn span(&self) -> Option<Interval> {
        Interval::covering(self.start_ns(), self.end_ns())
    }

    /// Its order among the stream's payloads with the same start.
    #[must_use]
    pub fn seq(&self) -> u32 {
        self.header().seq()
    }

    /// The payloads it was computed from.
    #[must_use]
    pub fn parents(&self) -> Vec<Uuid> {
        self.header().parents().map(|v| v.iter().map(Uuid::from).collect()).unwrap_or_default()
    }

    /// The people or things it's about.
    #[must_use]
    pub fn subjects(&self) -> Vec<Uuid> {
        self.header().subjects().map(|v| v.iter().map(Uuid::from).collect()).unwrap_or_default()
    }

    /// Its shape's name (a raw format like "NV12", or a named shape); empty when it has none.
    #[must_use]
    pub fn shape_name(&self) -> &str {
        self.header().shape_name().unwrap_or("")
    }
}

impl fmt::Debug for HeaderBuf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HeaderBuf")
            .field("id", &self.id().to_string())
            .field("stream", &self.stream())
            .field("start_ns", &self.start_ns())
            .field("end_ns", &self.end_ns())
            .field("seq", &self.seq())
            .finish_non_exhaustive()
    }
}

/// The header fields [`HeaderBuf::build`] sets; the rest (producer, evidence, compute, confidence, capture) stay
/// unset. Producers that need them build the FlatBuffer themselves and check it with [`HeaderBuf::new`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeaderSpec {
    /// The payload's id.
    pub id: Uuid,
    /// Its stream.
    pub stream: StreamId,
    /// The time it covers, UTC nanoseconds.
    pub start_ns: u64,
    /// Where the span ends: exclusive, or equal to `start_ns` for an instant (see [`Interval::covering`]).
    pub end_ns: u64,
    /// Order among the stream's payloads with the same start.
    pub seq: u32,
    /// Its domain layer.
    pub layer: DomainLayer,
    /// How its bytes are encoded.
    pub format: PayloadFormat,
    /// A raw format like "NV12", or a named shape.
    pub shape_name: String,
    /// The shape's version.
    pub shape_version: u32,
    /// The payloads it was computed from.
    pub parents: Vec<Uuid>,
    /// The people or things it's about.
    pub subjects: Vec<Uuid>,
    /// The coldest tier any input was read from.
    pub source_tier: StorageTier,
}

impl HeaderSpec {
    /// A raw layer 1 payload on `stream` over `[start_ns, end_ns]`, with no parents or subjects.
    #[must_use]
    pub fn raw(id: Uuid, stream: StreamId, start_ns: u64, end_ns: u64) -> Self {
        Self {
            id,
            stream,
            start_ns,
            end_ns,
            seq: 0,
            layer: DomainLayer::Layer1Raw,
            format: PayloadFormat::RawBytes,
            shape_name: String::new(),
            shape_version: 0,
            parents: Vec::new(),
            subjects: Vec::new(),
            source_tier: StorageTier::Tier0Hot,
        }
    }
}

/// Where a payload's bytes are now: an owned form of the schema's `StorageLocator`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum StorageLocator {
    /// Tier 0: a slot in a shared memory arena.
    Shm {
        /// The arena's name.
        arena: String,
        /// The arena's generation (a recreated arena gets a new one).
        arena_generation: u32,
        /// The slot's size class.
        slab_class: u16,
        /// The slot's index in its class.
        slot: u32,
        /// The slot's generation.
        slot_generation: u32,
        /// From the arena's start; a multiple of 16,384.
        offset: u64,
        /// Bytes used in the slot.
        length: u64,
    },
    /// The bytes themselves, kept in the catalog: for small payloads (an event's JSON), at most
    /// [`MAX_INLINE_BYTES`]. It reports `Tier1Warm`, since the catalog file is durable and on NVMe; it never moves
    /// between tiers, and a purge removes it with the payload's row.
    Inline {
        /// The payload's bytes.
        bytes: Vec<u8>,
    },
    /// Tier 0: a batch of records in a fixed-stride ring, by sequence number (records `first_seq` to
    /// `first_seq + count - 1`). Readable until the ring laps it.
    Ring {
        /// The arena's name.
        arena: String,
        /// The arena's generation.
        arena_generation: u32,
        /// The ring's index, from its stream's registration.
        ring: u16,
        /// The batch's first record.
        first_seq: u64,
        /// How many records.
        count: u32,
    },
    /// Tier 1 or 2: a byte range in a chunk file or archive object.
    Chunk {
        /// `Tier1Warm` or `Tier2Cold`.
        tier: StorageTier,
        /// The chunk's file or object.
        uri: String,
        /// Where the bytes start in it.
        offset: u64,
        /// How many bytes.
        length: u64,
    },
}

/// The most bytes a [`StorageLocator::Inline`] holds; a larger payload goes in shared memory.
/// 64 KiB: text events are far smaller (a spoken reply is a few hundred bytes), and the catalog stores values this
/// size well. A query result caps its inline bytes so a page of them still fits one wire frame.
pub const MAX_INLINE_BYTES: usize = 64 * 1024;

impl StorageLocator {
    /// An inline locator holding `bytes`, or `None` when they're more than [`MAX_INLINE_BYTES`].
    #[must_use]
    pub fn inline(bytes: Vec<u8>) -> Option<Self> {
        (bytes.len() <= MAX_INLINE_BYTES).then_some(Self::Inline { bytes })
    }

    /// The tier its bytes are in.
    #[must_use]
    pub fn tier(&self) -> StorageTier {
        match self {
            Self::Shm { .. } | Self::Ring { .. } => StorageTier::Tier0Hot,
            Self::Inline { .. } => StorageTier::Tier1Warm,
            Self::Chunk { tier, .. } => *tier,
        }
    }

    /// The schema's FlatBuffer form, as a buffer whose root is a `StorageLocator`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut b = flatbuffers::FlatBufferBuilder::new();
        let root = self.build_into(&mut b);
        b.finish(root, None);
        b.finished_data().to_vec()
    }

    /// Writes the locator into a buffer being built (an `Item` in a `QueryResult`, say) and returns its offset.
    pub fn build_into<'a>(
        &self,
        b: &mut flatbuffers::FlatBufferBuilder<'a>,
    ) -> flatbuffers::WIPOffset<fb::StorageLocator<'a>> {
        let (location_type, location) = match self {
            Self::Shm { arena, arena_generation, slab_class, slot, slot_generation, offset, length } => {
                let arena = b.create_string(arena);
                let shm = fb::ShmLocation::create(
                    b,
                    &fb::ShmLocationArgs {
                        arena: Some(arena),
                        arena_generation: *arena_generation,
                        slab_class: *slab_class,
                        slot: *slot,
                        slot_generation: *slot_generation,
                        offset: *offset,
                        length: *length,
                    },
                );
                (fb::Location::ShmLocation, shm.as_union_value())
            }
            Self::Ring { arena, arena_generation, ring, first_seq, count } => {
                let arena = b.create_string(arena);
                let r = fb::RingLocation::create(
                    b,
                    &fb::RingLocationArgs {
                        arena: Some(arena),
                        arena_generation: *arena_generation,
                        ring: *ring,
                        first_seq: *first_seq,
                        count: *count,
                    },
                );
                (fb::Location::RingLocation, r.as_union_value())
            }
            Self::Inline { bytes } => {
                let bytes = b.create_vector(bytes);
                let inline = fb::InlineLocation::create(b, &fb::InlineLocationArgs { bytes: Some(bytes) });
                (fb::Location::InlineLocation, inline.as_union_value())
            }
            Self::Chunk { uri, offset, length, .. } => {
                let uri = b.create_string(uri);
                let chunk = fb::ChunkLocation::create(
                    b,
                    &fb::ChunkLocationArgs { uri: Some(uri), offset: *offset, length: *length },
                );
                (fb::Location::ChunkLocation, chunk.as_union_value())
            }
        };
        fb::StorageLocator::create(
            b,
            &fb::StorageLocatorArgs { tier: self.tier(), location_type, location: Some(location) },
        )
    }

    /// Reads what [`StorageLocator::encode`] writes.
    ///
    /// # Errors
    /// [`DecodeError`] when the bytes aren't a valid locator, or its location doesn't match its tier.
    pub fn decode(bytes: &[u8]) -> Result<Self, DecodeError> {
        let l = flatbuffers::root::<fb::StorageLocator<'_>>(bytes).map_err(DecodeError::Invalid)?;
        Self::from_table(&l)
    }

    /// The owned form of a locator read from a FlatBuffer.
    ///
    /// # Errors
    /// [`DecodeError::BadLocation`] when it has no location, or one that doesn't match its tier;
    /// [`DecodeError::InlineTooLarge`] for inline bytes over [`MAX_INLINE_BYTES`].
    pub fn from_table(l: &fb::StorageLocator<'_>) -> Result<Self, DecodeError> {
        if let Some(i) = l.location_as_inline_location() {
            let bytes = i.bytes().map(|v| v.bytes().to_vec()).unwrap_or_default();
            if bytes.len() > MAX_INLINE_BYTES {
                return Err(DecodeError::InlineTooLarge);
            }
            return if l.tier() == StorageTier::Tier1Warm {
                Ok(Self::Inline { bytes })
            } else {
                Err(DecodeError::BadLocation)
            };
        }
        if let Some(r) = l.location_as_ring_location() {
            return match r.arena() {
                Some(arena) if l.tier() == StorageTier::Tier0Hot && r.count() > 0 => Ok(Self::Ring {
                    arena: arena.to_owned(),
                    arena_generation: r.arena_generation(),
                    ring: r.ring(),
                    first_seq: r.first_seq(),
                    count: r.count(),
                }),
                _ => Err(DecodeError::BadLocation),
            };
        }
        match (l.tier(), l.location_as_shm_location(), l.location_as_chunk_location()) {
            (StorageTier::Tier0Hot, Some(s), None)
                if s.arena().is_some() && s.offset().is_multiple_of(crate::PAGE_ALIGN as u64) =>
            {
                Ok(Self::Shm {
                    arena: s.arena().unwrap_or_default().to_owned(),
                    arena_generation: s.arena_generation(),
                    slab_class: s.slab_class(),
                    slot: s.slot(),
                    slot_generation: s.slot_generation(),
                    offset: s.offset(),
                    length: s.length(),
                })
            }
            (tier @ (StorageTier::Tier1Warm | StorageTier::Tier2Cold), None, Some(c)) if c.uri().is_some() => {
                Ok(Self::Chunk {
                    tier,
                    uri: c.uri().unwrap_or_default().to_owned(),
                    offset: c.offset(),
                    length: c.length(),
                })
            }
            _ => Err(DecodeError::BadLocation),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_interval_is_half_open_and_never_empty() {
        assert_eq!(Interval::new(5, 5), None);
        assert_eq!(Interval::new(6, 5), None);
        let i = Interval::new(5, 8).unwrap();
        assert_eq!((i.start_ns(), i.end_ns(), i.len_ns()), (5, 8, 3));
        assert!(i.contains(5) && i.contains(7) && !i.contains(8) && !i.contains(4));
        assert!(i.overlaps(Interval::new(7, 9).unwrap()));
        assert!(!i.overlaps(Interval::new(8, 9).unwrap()), "touching ends don't overlap");
        assert!(!i.overlaps(Interval::new(1, 5).unwrap()));
    }

    #[test]
    fn covering_gives_an_instant_its_one_nanosecond() {
        assert_eq!(Interval::covering(7, 7), Interval::new(7, 8));
        assert_eq!(Interval::covering(7, 9), Interval::new(7, 9));
        assert_eq!(Interval::covering(9, 7), None);
        assert_eq!(Interval::covering(u64::MAX, u64::MAX), None);
        let instant = HeaderBuf::build(&HeaderSpec::raw(Uuid([1; 16]), StreamId(1), 40, 40)).unwrap();
        assert_eq!(instant.span(), Interval::new(40, 41));
    }

    #[test]
    fn overlaps_is_symmetric_and_means_a_shared_nanosecond() {
        let all: Vec<Interval> = (0..6u64).flat_map(|a| (a + 1..7).filter_map(move |b| Interval::new(a, b))).collect();
        for &x in &all {
            for &y in &all {
                let shared = (x.start_ns()..x.end_ns()).any(|t| y.contains(t));
                assert_eq!(x.overlaps(y), shared, "{x:?} {y:?}");
                assert_eq!(x.overlaps(y), y.overlaps(x));
            }
        }
    }

    #[test]
    fn a_built_header_reads_back_its_fields() {
        let mut spec = HeaderSpec::raw(Uuid([9; 16]), StreamId(4), 100, 133);
        spec.seq = 2;
        spec.parents = vec![Uuid([1; 16]), Uuid([2; 16])];
        spec.subjects = vec![Uuid([3; 16])];
        spec.shape_name = "NV12".into();
        let h = HeaderBuf::build(&spec).unwrap();
        assert_eq!((h.id(), h.stream(), h.start_ns(), h.end_ns(), h.seq()), (Uuid([9; 16]), StreamId(4), 100, 133, 2));
        assert_eq!(h.parents(), spec.parents);
        assert_eq!(h.subjects(), spec.subjects);
        assert_eq!(h.header().shape_name(), Some("NV12"));
        assert_eq!(HeaderBuf::new(h.as_bytes().to_vec()).unwrap(), h);
    }

    #[test]
    fn a_header_without_an_id_or_ending_before_it_starts_is_refused() {
        let mut b = flatbuffers::FlatBufferBuilder::new();
        let header = fb::PayloadHeader::create(&mut b, &fb::PayloadHeaderArgs { stream_id: 1, ..Default::default() });
        fb::finish_payload_header_buffer(&mut b, header);
        assert_eq!(HeaderBuf::new(b.finished_data().to_vec()), Err(DecodeError::MissingId));
        let backwards = HeaderSpec::raw(Uuid([1; 16]), StreamId(1), 10, 9);
        assert_eq!(HeaderBuf::build(&backwards), Err(DecodeError::EndBeforeStart));
        assert!(matches!(HeaderBuf::new(vec![1, 2, 3]), Err(DecodeError::Invalid(_))));
        let good = HeaderBuf::build(&HeaderSpec::raw(Uuid([1; 16]), StreamId(1), 1, 2)).unwrap().into_bytes();
        assert!(HeaderBuf::new(good[..good.len() / 2].to_vec()).is_err(), "a truncated header");
    }

    #[test]
    fn an_inline_locator_round_trips_up_to_its_cap() {
        let small = StorageLocator::inline(b"{\"type\":\"seen\"}".to_vec()).unwrap();
        assert_eq!(StorageLocator::decode(&small.encode()).unwrap(), small);
        assert_eq!(small.tier(), StorageTier::Tier1Warm);
        assert!(StorageLocator::inline(vec![0; MAX_INLINE_BYTES]).is_some());
        assert!(StorageLocator::inline(vec![0; MAX_INLINE_BYTES + 1]).is_none());
        let too_big = StorageLocator::Inline { bytes: vec![0; MAX_INLINE_BYTES + 1] };
        assert_eq!(StorageLocator::decode(&too_big.encode()), Err(DecodeError::InlineTooLarge));
    }

    #[test]
    fn locators_round_trip_and_know_their_tier() {
        let shm = StorageLocator::Shm {
            arena: "cam0".into(),
            arena_generation: 3,
            slab_class: 2,
            slot: 17,
            slot_generation: 9,
            offset: 5 * 16_384,
            length: 1234,
        };
        let chunk = StorageLocator::Chunk {
            tier: StorageTier::Tier2Cold,
            uri: "file:///c/1.mkv".into(),
            offset: 64,
            length: 9,
        };
        assert_eq!(StorageLocator::decode(&shm.encode()).unwrap(), shm);
        assert_eq!(StorageLocator::decode(&chunk.encode()).unwrap(), chunk);
        assert_eq!((shm.tier(), chunk.tier()), (StorageTier::Tier0Hot, StorageTier::Tier2Cold));
        let ring = StorageLocator::Ring { arena: "a".into(), arena_generation: 4, ring: 2, first_seq: 900, count: 100 };
        assert_eq!(StorageLocator::decode(&ring.encode()).unwrap(), ring);
        assert_eq!(ring.tier(), StorageTier::Tier0Hot);
        // An empty batch isn't a batch.
        let empty = StorageLocator::Ring { arena: "a".into(), arena_generation: 4, ring: 2, first_seq: 900, count: 0 };
        assert_eq!(StorageLocator::decode(&empty.encode()), Err(DecodeError::BadLocation));
    }

    fn shm_at(offset: u64) -> StorageLocator {
        StorageLocator::Shm {
            arena: "a".into(),
            arena_generation: 0,
            slab_class: 0,
            slot: 0,
            slot_generation: 0,
            offset,
            length: 1,
        }
    }

    #[test]
    fn a_locator_that_breaks_its_tier_or_alignment_is_refused() {
        let chunk_in_tier_0 =
            StorageLocator::Chunk { tier: StorageTier::Tier0Hot, uri: "x".into(), offset: 0, length: 1 };
        assert_eq!(StorageLocator::decode(&chunk_in_tier_0.encode()), Err(DecodeError::BadLocation));
        assert_eq!(StorageLocator::decode(&shm_at(4096).encode()), Err(DecodeError::BadLocation));
        let mut b = flatbuffers::FlatBufferBuilder::new();
        let shm = fb::ShmLocation::create(&mut b, &fb::ShmLocationArgs { arena: None, ..Default::default() });
        let root = fb::StorageLocator::create(
            &mut b,
            &fb::StorageLocatorArgs {
                tier: StorageTier::Tier1Warm,
                location_type: fb::Location::ShmLocation,
                location: Some(shm.as_union_value()),
            },
        );
        b.finish(root, None);
        assert_eq!(StorageLocator::decode(b.finished_data()), Err(DecodeError::BadLocation), "shm in tier 1");
        let mut b = flatbuffers::FlatBufferBuilder::new();
        let root = fb::StorageLocator::create(&mut b, &fb::StorageLocatorArgs::default());
        b.finish(root, None);
        assert_eq!(StorageLocator::decode(b.finished_data()), Err(DecodeError::BadLocation), "no location");
    }
}
