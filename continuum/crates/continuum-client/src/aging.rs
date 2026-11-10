//! Aging (docs/CONTINUUM.md, "Aging"): age out whole payloads ([`Client::age_out`]), and the reads the video
//! window makes: the commit clock and a stream's aged-out spans. Also a chunk's frames and copies
//! ([`Client::get_chunk`]) and the catalog's page stats. These need broker schema minor 23 or later.
//! [`Client::resolve_frames`], the source identity an analysis receipt names (#1394), needs 26.

use continuum_core::schema::continuum as fb;
use continuum_core::{StorageLocator, StreamId, Uuid};
use flatbuffers::FlatBufferBuilder;

use crate::messages::finish;
use crate::worker::ChunkFrame;
use crate::{Client, Error, PurgeDone, PurgeTarget, unexpected};

/// The broker schema minor that has the aging messages.
pub const AGING_SCHEMA_MINOR: u16 = 23;
/// The broker schema minor that has [`Client::resolve_frames`].
pub const RESOLVE_FRAMES_SCHEMA_MINOR: u16 = 26;
/// The schema minor that answers `GetCatalogStats`.
pub const CATALOG_STATS_SCHEMA_MINOR: u16 = 30;

/// One table's share of the catalog file, from [`Client::catalog_stats`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TableStorage {
    /// The table.
    pub name: String,
    /// Its entries.
    pub rows: u64,
    /// The B-tree's height.
    pub height: u32,
    /// Leaf pages.
    pub leaf_pages: u64,
    /// Branch pages.
    pub branch_pages: u64,
    /// Key and value bytes.
    pub stored_bytes: u64,
    /// Page headers and per-entry offsets.
    pub metadata_bytes: u64,
    /// Unused space inside its pages.
    pub fragmented_bytes: u64,
}

/// Where the catalog file's bytes go, from one read snapshot ([`Client::catalog_stats`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CatalogStats {
    /// The page size.
    pub page_size: u64,
    /// The file's length, including space reserved but never written (sparse on disk).
    pub file_bytes: u64,
    /// The blocks the file holds on disk.
    pub disk_bytes: u64,
    /// Every table, largest first.
    pub tables: Vec<TableStorage>,
}
/// The most ids one [`Client::resolve_frames`] may ask about.
pub const MAX_RESOLVE_IDS: usize = 256;

/// Where an id the catalog was asked about is, from [`Client::resolve_frames`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameSource {
    /// Never committed, forgotten, or its frame row pruned (an hour after its chunk ended, for a frame nothing was
    /// derived from).
    Unknown,
    /// A payload with its own header: not absorbed into a chunk (yet), or not a raw frame.
    Payload {
        /// Its stream.
        stream: StreamId,
        /// Its commit sequence.
        seq: u64,
    },
    /// A raw frame inside a chunk.
    Absorbed {
        /// The chunk.
        chunk: Uuid,
        /// The chunk's stream.
        stream: StreamId,
        /// The chunk's commit sequence.
        seq: u64,
        /// The frame's index in the chunk.
        frame_index: u32,
        /// The frame's start, UTC nanoseconds, from the chunk's frame index.
        start_ns: u64,
    },
}

/// One id [`Client::resolve_frames`] asked about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolvedFrame {
    /// The asked id.
    pub id: Uuid,
    /// Where it is.
    pub source: FrameSource,
}

/// A chunk's frames and copies, from [`Client::get_chunk`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChunkInfo {
    /// The chunk.
    pub chunk_id: Uuid,
    /// Its stream.
    pub stream: StreamId,
    /// Its commit sequence.
    pub seq: u64,
    /// Its span, UTC nanoseconds.
    pub span: (u64, u64),
    /// Its frame index, cut flags included.
    pub frames: Vec<ChunkFrame>,
    /// Every copy, Tier 0 first.
    pub locations: Vec<StorageLocator>,
    /// What capture attested about each frame, in frame order: unknown (`None`) on brokers below minor 27, for
    /// chunks committed without it, and for a provenance version this client doesn't read.
    pub provenance: Option<Vec<crate::FrameProvenance>>,
}

impl Client {
    /// Ages out these payloads (a `Purge` with `age_out`): only they go, none of what was derived from them, and
    /// only those committed by `committed_before_ns` (by the commit clock, whatever their own time). Their spans are
    /// recorded as aged out, joined into one run wherever they're at most `aged_gap_ns` apart on a stream (0: each
    /// payload's own span). `reason` and `detail` are kept for good, as for any purge; the per-item report is kept
    /// until every keeper acknowledges it, then only the spans and the count stay.
    ///
    /// # Errors
    ///
    /// As [`Client::purge`].
    pub async fn age_out(
        &self,
        ids: &[Uuid],
        committed_before_ns: u64,
        aged_gap_ns: u64,
        reason: &str,
        detail: &str,
    ) -> Result<PurgeDone, Error> {
        let target = PurgeTarget::AgeOut { ids: ids.to_vec(), committed_before_ns, aged_gap_ns };
        self.purge(&target, reason, detail).await
    }

    /// The highest commit sequence known committed at or before `at_ns` (0 when no mark is that old).
    ///
    /// # Errors
    ///
    /// As any request.
    pub async fn commit_seq_at(&self, at_ns: u64) -> Result<u64, Error> {
        let reply = self
            .shared
            .request(|id| {
                let mut b = FlatBufferBuilder::new();
                let r = fb::CommitSeqAt::create(&mut b, &fb::CommitSeqAtArgs { at_ns });
                finish(&mut b, id, fb::ControlPayload::CommitSeqAt, r.as_union_value())
            })
            .await?;
        let r = reply.envelope().payload_as_commit_seq().ok_or_else(|| unexpected("CommitSeq", &reply))?;
        Ok(r.seq())
    }

    /// A stream's aged-out spans within `range`, at most `limit` (0 for the broker's default).
    ///
    /// # Errors
    ///
    /// As any request: NotFound for an unknown stream, BadRequest for an empty range.
    pub async fn aged_out(&self, stream: &str, range: (u64, u64), limit: u32) -> Result<Vec<(u64, u64)>, Error> {
        let reply = self
            .shared
            .request(|id| {
                let mut b = FlatBufferBuilder::new();
                let stream = b.create_string(stream);
                let range = fb::TimeRange::new(range.0, range.1);
                let r = fb::ListAgedOut::create(
                    &mut b,
                    &fb::ListAgedOutArgs { stream: Some(stream), range: Some(&range), limit },
                );
                finish(&mut b, id, fb::ControlPayload::ListAgedOut, r.as_union_value())
            })
            .await?;
        let r = reply.envelope().payload_as_aged_out_list().ok_or_else(|| unexpected("AgedOutList", &reply))?;
        Ok(r.spans().map(|v| v.iter().map(|s| (s.start_ns(), s.end_ns())).collect()).unwrap_or_default())
    }

    /// Where the catalog file's bytes go, from one read snapshot on the broker's readers (it walks every table's
    /// pages, so on a large catalog it takes a while): each table's pages and bytes, the file's apparent size, and its
    /// blocks on disk. Read-only. Empty for a catalog that keeps no file.
    ///
    /// # Errors
    ///
    /// As any request; a Protocol error, sent nothing, for a broker below [`CATALOG_STATS_SCHEMA_MINOR`].
    pub async fn catalog_stats(&self) -> Result<CatalogStats, Error> {
        let minor = self.broker_schema().1;
        if minor < CATALOG_STATS_SCHEMA_MINOR {
            return Err(Error::Protocol(format!(
                "the broker speaks schema minor {minor}; catalog stats need {CATALOG_STATS_SCHEMA_MINOR}"
            )));
        }
        let reply = self
            .shared
            .request(|id| {
                let mut b = FlatBufferBuilder::new();
                let r = fb::GetCatalogStats::create(&mut b, &fb::GetCatalogStatsArgs {});
                finish(&mut b, id, fb::ControlPayload::GetCatalogStats, r.as_union_value())
            })
            .await?;
        let r = reply.envelope().payload_as_catalog_stats().ok_or_else(|| unexpected("CatalogStats", &reply))?;
        let tables = r
            .tables()
            .map(|v| {
                v.iter()
                    .map(|t| TableStorage {
                        name: t.name().unwrap_or_default().to_owned(),
                        rows: t.rows(),
                        height: t.height(),
                        leaf_pages: t.leaf_pages(),
                        branch_pages: t.branch_pages(),
                        stored_bytes: t.stored_bytes(),
                        metadata_bytes: t.metadata_bytes(),
                        fragmented_bytes: t.fragmented_bytes(),
                    })
                    .collect()
            })
            .unwrap_or_default();
        Ok(CatalogStats { page_size: r.page_size(), file_bytes: r.file_bytes(), disk_bytes: r.disk_bytes(), tables })
    }

    /// Which chunk holds each of `ids` (at most [`MAX_RESOLVE_IDS`]), one entry per id in the asked order: so a
    /// producer's analysis receipt names the recordings it actually evaluated, not whatever shares their time.
    ///
    /// # Errors
    ///
    /// As any request; a Protocol error, sent nothing, for more than [`MAX_RESOLVE_IDS`] ids or a broker below
    /// [`RESOLVE_FRAMES_SCHEMA_MINOR`].
    pub async fn resolve_frames(&self, ids: &[Uuid]) -> Result<Vec<ResolvedFrame>, Error> {
        let minor = self.broker_schema().1;
        if minor < RESOLVE_FRAMES_SCHEMA_MINOR {
            return Err(Error::Protocol(format!(
                "the broker speaks schema minor {minor}; resolving frames needs {RESOLVE_FRAMES_SCHEMA_MINOR}"
            )));
        }
        if ids.len() > MAX_RESOLVE_IDS {
            return Err(Error::Protocol(format!("{} ids to resolve; at most {MAX_RESOLVE_IDS}", ids.len())));
        }
        let reply = self
            .shared
            .request(|id| {
                let mut b = FlatBufferBuilder::new();
                let asked: Vec<fb::Uuid> = ids.iter().map(|u| fb::Uuid(u.0)).collect();
                let asked = b.create_vector(&asked);
                let r = fb::ResolveFrames::create(&mut b, &fb::ResolveFramesArgs { ids: Some(asked) });
                finish(&mut b, id, fb::ControlPayload::ResolveFrames, r.as_union_value())
            })
            .await?;
        let r = reply.envelope().payload_as_frames_resolved().ok_or_else(|| unexpected("FramesResolved", &reply))?;
        let frames: Vec<ResolvedFrame> = r
            .frames()
            .into_iter()
            .flatten()
            .map(|f| {
                let uuid = |u: Option<&fb::Uuid>| Uuid(u.map_or([0; 16], |u| u.0));
                let stream = StreamId(f.stream_id());
                let source = match f.state() {
                    fb::FrameState::Payload => FrameSource::Payload { stream, seq: f.seq() },
                    fb::FrameState::Absorbed => FrameSource::Absorbed {
                        chunk: uuid(f.chunk_id()),
                        stream,
                        seq: f.seq(),
                        frame_index: f.frame_index(),
                        start_ns: f.start_ns(),
                    },
                    _ => FrameSource::Unknown,
                };
                ResolvedFrame { id: uuid(f.id()), source }
            })
            .collect();
        if frames.len() != ids.len() || frames.iter().zip(ids).any(|(f, id)| f.id != *id) {
            return Err(Error::Protocol("FramesResolved doesn't answer the asked ids in order".into()));
        }
        Ok(frames)
    }

    /// A chunk's frame index, every copy, and its frame provenance.
    ///
    /// # Errors
    ///
    /// As any request: NotFound for an unknown chunk, BadRequest for a payload that isn't a chunk.
    pub async fn get_chunk(&self, chunk: Uuid) -> Result<ChunkInfo, Error> {
        let reply = self
            .shared
            .request(|id| {
                let mut b = FlatBufferBuilder::new();
                let chunk = fb::Uuid(chunk.0);
                let r = fb::GetChunk::create(&mut b, &fb::GetChunkArgs { chunk_id: Some(&chunk) });
                finish(&mut b, id, fb::ControlPayload::GetChunk, r.as_union_value())
            })
            .await?;
        let r = reply.envelope().payload_as_chunk_info().ok_or_else(|| unexpected("ChunkInfo", &reply))?;
        let mut locations = Vec::new();
        for l in r.locations().into_iter().flatten() {
            locations
                .push(StorageLocator::from_table(&l).map_err(|e| Error::Protocol(format!("ChunkInfo location: {e}")))?);
        }
        Ok(ChunkInfo {
            chunk_id: Uuid(r.chunk_id().map_or([0; 16], |u| u.0)),
            stream: StreamId(r.stream_id()),
            seq: r.seq(),
            span: r.range().map_or((0, 0), |s| (s.start_ns(), s.end_ns())),
            frames: r
                .frames()
                .map(|v| {
                    v.iter()
                        .map(|f| ChunkFrame {
                            start_ns: f.start_ns(),
                            offset: f.offset(),
                            length: f.length(),
                            cut: f.cut(),
                        })
                        .collect()
                })
                .unwrap_or_default(),
            locations,
            provenance: decode_provenance(r.provenance_version(), r.frame_provenance())?,
        })
    }
}

fn decode_provenance(
    version: u32,
    rows: Option<flatbuffers::Vector<'_, fb::FrameProvenance>>,
) -> Result<Option<Vec<crate::FrameProvenance>>, Error> {
    if version != crate::PROVENANCE_VERSION {
        return Ok(None);
    }
    rows.into_iter()
        .flatten()
        .map(|f| {
            if f.flags() & !1 != 0 {
                return Err(Error::Protocol(format!("ChunkInfo frame provenance flags {:#x}", f.flags())));
            }
            Ok(crate::FrameProvenance {
                device_seq: f.device_seq(),
                epoch: f.epoch(),
                stamped_at_arrival: f.flags() & 1 != 0,
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

/// Refuses an age-out to a broker below [`AGING_SCHEMA_MINOR`], which would read it as a plain purge by ids and
/// take everything derived from them.
pub(crate) fn check_broker(target: &PurgeTarget, broker_minor: u16) -> Result<(), Error> {
    if matches!(target, PurgeTarget::AgeOut { .. }) && broker_minor < AGING_SCHEMA_MINOR {
        return Err(Error::Protocol(format!(
            "the broker speaks schema minor {broker_minor}; aging needs {AGING_SCHEMA_MINOR}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_age_out_is_refused_by_a_broker_before_minor_23_and_other_purges_are_not() {
        let age = PurgeTarget::AgeOut { ids: vec![Uuid([1; 16])], committed_before_ns: 5, aged_gap_ns: 0 };
        assert!(matches!(check_broker(&age, 22), Err(Error::Protocol(_))));
        assert!(check_broker(&age, 23).is_ok());
        assert!(check_broker(&PurgeTarget::Ids(vec![Uuid([1; 16])]), 22).is_ok());
    }

    #[test]
    fn an_age_outs_gap_reaches_the_broker_and_an_older_clients_reads_as_none() {
        let read = |target: &PurgeTarget| {
            let bytes = crate::messages::purge(7, target, "aged", "{}");
            let env = flatbuffers::root::<fb::ControlEnvelope>(&bytes).unwrap();
            continuum_broker::protocol::purge_ask(&env.payload_as_purge().unwrap()).unwrap().0
        };
        let runs = PurgeTarget::AgeOut { ids: vec![Uuid([1; 16])], committed_before_ns: 5, aged_gap_ns: 2_000_000_000 };
        assert!(matches!(
            read(&runs),
            continuum_broker::protocol::PurgeAsk::AgeOut { committed_before_ns: 5, aged_gap_ns: 2_000_000_000, .. }
        ));
        let exact = PurgeTarget::AgeOut { ids: vec![Uuid([1; 16])], committed_before_ns: 5, aged_gap_ns: 0 };
        assert!(matches!(read(&exact), continuum_broker::protocol::PurgeAsk::AgeOut { aged_gap_ns: 0, .. }));
    }

    #[test]
    fn dependencies_reach_the_broker_only_on_an_acknowledged_notice_with_real_ranges() {
        let header = continuum_core::HeaderBuf::build(&continuum_core::HeaderSpec::raw(
            Uuid([7; 16]),
            continuum_core::StreamId(3),
            10,
            10,
        ))
        .unwrap();
        let read = |request_id: u64, deps: &[crate::Dependency]| {
            let bytes = crate::messages::ingest_depending(request_id, &header, None, deps);
            let env = flatbuffers::root::<fb::ControlEnvelope>(&bytes).unwrap();
            continuum_broker::protocol::dependencies(&env.payload_as_ingest_notice().unwrap(), request_id)
        };
        let dep = crate::Dependency { stream_id: 2, start_ns: 0, end_ns: 25, revision: 4 };
        let got = read(9, &[dep]).unwrap();
        assert_eq!((got[0].stream.0, got[0].span.start_ns(), got[0].span.end_ns(), got[0].revision), (2, 0, 25, 4));
        assert!(read(0, &[dep]).is_err(), "an unanswered notice couldn't learn they were refused");
        assert!(read(9, &[crate::Dependency { end_ns: 0, ..dep }]).is_err(), "an empty range");
        assert!(read(0, &[]).unwrap().is_empty(), "no dependencies: the plain notice");
    }
}
