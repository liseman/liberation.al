//! The aging messages (docs/CONTINUUM.md, "Aging"): `ListAgedOut` and `CommitSeqAt`, plus `GetChunk`,
//! `GetCatalogStats`, and `ResolveFrames` (source identity for analysis receipts, #1394), read into owned asks and
//! answered from what the catalog said. An age-out of whole payloads is a `Purge` with
//! `age_out` set (see [`super::purge_ask`]).

use continuum_catalog::Resolved;
use continuum_core::schema::continuum as fb;
use continuum_core::{Interval, StreamId, Uuid};
use flatbuffers::FlatBufferBuilder;

use super::{finish, time_range};

/// A `ListAgedOut`'s default and largest page.
pub const DEFAULT_AGED_LIMIT: usize = 1000;
/// See [`DEFAULT_AGED_LIMIT`].
pub const MAX_AGED_LIMIT: usize = 10_000;
/// The most ids one `ResolveFrames` may ask about.
pub const MAX_RESOLVE_IDS: usize = 256;

/// What one id a `ResolveFrames` asked about resolved to: only the identity, never the chunk's frame index, so a
/// full answer stays small.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameIdentity {
    /// The asked id.
    pub id: Uuid,
    /// The schema's state.
    pub state: fb::FrameState,
    /// The payload's stream, or its chunk's.
    pub stream: StreamId,
    /// The payload's commit sequence, or its chunk's.
    pub seq: u64,
    /// The chunk holding it, when absorbed.
    pub chunk: Option<Uuid>,
    /// Its index in the chunk.
    pub frame_index: u32,
    /// Its start from the chunk's frame index, UTC nanoseconds.
    pub start_ns: u64,
}

impl FrameIdentity {
    /// What `catalog.get(id)` said, `None` for unknown.
    #[must_use]
    pub fn of(id: Uuid, resolved: Option<&Resolved>) -> Self {
        let unknown = Self {
            id,
            state: fb::FrameState::Unknown,
            stream: StreamId(0),
            seq: 0,
            chunk: None,
            frame_index: 0,
            start_ns: 0,
        };
        match resolved {
            None => unknown,
            Some(Resolved::Payload(p)) => {
                Self { state: fb::FrameState::Payload, stream: p.header.stream(), seq: p.commit_seq, ..unknown }
            }
            Some(Resolved::AbsorbedFrame { chunk, frame_index }) => Self {
                state: fb::FrameState::Absorbed,
                stream: chunk.header.stream(),
                seq: chunk.commit_seq,
                chunk: Some(chunk.header.id()),
                frame_index: *frame_index,
                start_ns: chunk.frames.as_ref().and_then(|f| f.get(*frame_index as usize)).map_or(0, |f| f.start_ns),
                ..unknown
            },
        }
    }
}

/// A `ListAgedOut`, read: the stream's name, the window, and the page size.
///
/// # Errors
/// A message for a `BadRequest`: no stream or range, or an empty range.
pub fn list_aged_out_ask(r: &fb::ListAgedOut<'_>) -> Result<(String, Interval, usize), String> {
    let stream = r.stream().filter(|s| !s.is_empty()).ok_or("ListAgedOut without a stream")?.to_owned();
    let range = r.range().ok_or("ListAgedOut without a range")?;
    let window = Interval::new(range.start_ns(), range.end_ns())
        .ok_or_else(|| format!("ListAgedOut range [{}, {}) is empty", range.start_ns(), range.end_ns()))?;
    let limit = match r.limit() {
        0 => DEFAULT_AGED_LIMIT,
        n => usize::try_from(n).unwrap_or(MAX_AGED_LIMIT).min(MAX_AGED_LIMIT),
    };
    Ok((stream, window, limit))
}

/// The reply to a `ListAgedOut`.
pub fn aged_out_list(b: &mut FlatBufferBuilder<'static>, request_id: u64, spans: &[Interval]) -> Vec<u8> {
    b.reset();
    let spans: Vec<fb::TimeRange> = spans.iter().map(|&s| time_range(s)).collect();
    let spans = b.create_vector(&spans);
    let r = fb::AgedOutList::create(b, &fb::AgedOutListArgs { spans: Some(spans) });
    finish(b, request_id, fb::ControlPayload::AgedOutList, r.as_union_value()).to_vec()
}

/// The reply to a `GetCatalogStats`.
pub fn catalog_stats(
    b: &mut FlatBufferBuilder<'static>,
    request_id: u64,
    stats: &continuum_catalog::StorageStats,
) -> Vec<u8> {
    b.reset();
    let tables: Vec<_> = stats
        .tables
        .iter()
        .map(|t| {
            let name = b.create_string(&t.name);
            fb::TableStorage::create(
                b,
                &fb::TableStorageArgs {
                    name: Some(name),
                    rows: t.rows,
                    height: t.height,
                    leaf_pages: t.leaf_pages,
                    branch_pages: t.branch_pages,
                    stored_bytes: t.stored_bytes,
                    metadata_bytes: t.metadata_bytes,
                    fragmented_bytes: t.fragmented_bytes,
                },
            )
        })
        .collect();
    let tables = b.create_vector(&tables);
    let r = fb::CatalogStats::create(
        b,
        &fb::CatalogStatsArgs {
            page_size: stats.page_size,
            file_bytes: stats.file_bytes,
            disk_bytes: stats.disk_bytes,
            tables: Some(tables),
        },
    );
    finish(b, request_id, fb::ControlPayload::CatalogStats, r.as_union_value()).to_vec()
}

/// The reply to a `CommitSeqAt`.
pub fn commit_seq(b: &mut FlatBufferBuilder<'static>, request_id: u64, seq: u64) -> Vec<u8> {
    b.reset();
    let r = fb::CommitSeq::create(b, &fb::CommitSeqArgs { seq });
    finish(b, request_id, fb::ControlPayload::CommitSeq, r.as_union_value()).to_vec()
}

/// The reply to a `GetChunk`: the chunk's stream, commit sequence, span, frame index, every copy, and provenance.
pub fn chunk_info(
    b: &mut FlatBufferBuilder<'static>,
    request_id: u64,
    snapshot: &continuum_catalog::ChunkSnapshot,
) -> Vec<u8> {
    b.reset();
    let chunk = &snapshot.indexed;
    let header = &chunk.header;
    let id = fb::Uuid(header.id().0);
    let range = fb::TimeRange::new(header.start_ns(), header.end_ns());
    let frames: Vec<fb::ChunkFrame> =
        chunk.frames.iter().flatten().map(|f| fb::ChunkFrame::new(f.start_ns, f.offset, f.length, f.cut)).collect();
    let frames = b.create_vector(&frames);
    let locations: Vec<_> = chunk.locations.iter().map(|l| l.build_into(b)).collect();
    let locations = b.create_vector(&locations);
    let (frame_provenance, provenance_version) = match &snapshot.provenance {
        Some(p) => {
            let rows: Vec<fb::FrameProvenance> =
                p.frames.iter().map(|f| fb::FrameProvenance::new(f.device_seq, f.epoch, f.flags())).collect();
            (Some(b.create_vector(&rows)), p.version)
        }
        None => (None, 0),
    };
    let r = fb::ChunkInfo::create(
        b,
        &fb::ChunkInfoArgs {
            chunk_id: Some(&id),
            stream_id: header.stream().0,
            seq: chunk.commit_seq,
            range: Some(&range),
            frames: Some(frames),
            locations: Some(locations),
            frame_provenance,
            provenance_version,
            ..Default::default()
        },
    );
    finish(b, request_id, fb::ControlPayload::ChunkInfo, r.as_union_value()).to_vec()
}

/// The reply to a `ResolveFrames`: one entry per asked id, in the asked order.
pub fn frames_resolved(b: &mut FlatBufferBuilder<'static>, request_id: u64, frames: &[FrameIdentity]) -> Vec<u8> {
    b.reset();
    let entries: Vec<_> = frames
        .iter()
        .map(|f| {
            let id = fb::Uuid(f.id.0);
            let chunk = f.chunk.map(|c| fb::Uuid(c.0));
            fb::ResolvedFrame::create(
                b,
                &fb::ResolvedFrameArgs {
                    id: Some(&id),
                    state: f.state,
                    stream_id: f.stream.0,
                    seq: f.seq,
                    chunk_id: chunk.as_ref(),
                    frame_index: f.frame_index,
                    start_ns: f.start_ns,
                },
            )
        })
        .collect();
    let entries = b.create_vector(&entries);
    let r = fb::FramesResolved::create(b, &fb::FramesResolvedArgs { frames: Some(entries) });
    finish(b, request_id, fb::ControlPayload::FramesResolved, r.as_union_value()).to_vec()
}
