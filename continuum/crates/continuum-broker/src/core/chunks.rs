//! `CommitChunk` and `RewriteChunk`: the tier worker's chunks reaching the catalog (docs/CONTINUUM.md,
//! "Storage tiers"). The worker never holds the catalog: it closes a chunk, and the broker commits it, which compacts
//! the absorbed Tier 0 payloads' headers into the chunk, and records the Tier 1 copy in the availability index. After
//! a purge cut frames from a chunk, the worker rewrites the file and the broker records its new frame index and
//! location.
//!
//! Only a session that registered a worker may send either: a chunk names payloads it absorbed, so a stray client
//! must not be able to compact (or relocate) someone else's frames. The catalog writes go through the writer's urgent
//! queue, never on this task; pending ingests are committed first, so every absorbed payload is indexed by the time
//! the chunk names it. A committed chunk isn't published to subscribers: its frames were, as each was ingested, and
//! the tier worker follows those streams itself.

use super::*;

impl Sched {
    pub(super) fn commit_chunk(&mut self, session: SessionId, request_id: u64, ask: protocol::ChunkCommitAsk) {
        if !self.workers.contains_key(&session) {
            return self.reply_error(
                session,
                request_id,
                ErrorCode::NotAllowed,
                "only a registered worker commits chunks",
            );
        }
        if let Err(e) = self.urgent.try_send(Urgent::CommitChunk { session, request_id, ask }) {
            self.refuse_busy(session, request_id, &e);
        }
    }

    pub(super) fn rewrite_chunk(
        &mut self,
        session: SessionId,
        request_id: u64,
        chunk: Uuid,
        frames: Vec<continuum_catalog::ChunkFrame>,
        location: StorageLocator,
    ) {
        if !self.workers.contains_key(&session) {
            return self.reply_error(
                session,
                request_id,
                ErrorCode::NotAllowed,
                "only a registered worker rewrites chunks",
            );
        }
        if let Err(e) = self.urgent.try_send(Urgent::RewriteChunk { session, request_id, chunk, frames, location }) {
            self.refuse_busy(session, request_id, &e);
        }
    }

    /// The writer committed a chunk (or didn't): record its copy and answer.
    pub(super) fn chunk_committed(
        &mut self,
        session: SessionId,
        request_id: u64,
        chunk: Uuid,
        copy: Option<(StreamId, Interval, StorageTier, Vec<Interval>)>,
        result: Result<continuum_catalog::Inserted, (ErrorCode, String)>,
    ) {
        match result {
            Ok(inserted) => {
                // A repeat added nothing; a chunk that landed under a running purge isn't available; a slot the
                // chunk landed without (its frame cut by a purge, or about to be) isn't either.
                if !inserted.duplicate
                    && !inserted.hidden
                    && let Some((stream, span, tier, slots)) = copy
                {
                    let cut: Vec<Interval> =
                        inserted.cut.iter().filter_map(|&i| slots.get(i as usize).copied()).collect();
                    if self.index.add_chunk_copy(chunk, stream, span, tier, &cut).is_err() {
                        warn!(chunk = %chunk, stream = stream.0, "a committed chunk's stream isn't in the index");
                    }
                }
                let bytes = protocol::chunk_committed(&mut self.builder, request_id, chunk, &inserted);
                self.push(session, bytes);
            }
            Err((code, message)) => self.reply_error(session, request_id, code, &message),
        }
    }

    /// The writer recorded a rewritten chunk (or didn't): record the copy it gained, if it moved tiers, and answer.
    pub(super) fn chunk_rewritten(
        &mut self,
        session: SessionId,
        request_id: u64,
        chunk: Uuid,
        added: Option<(StreamId, Interval, StorageTier, Vec<Interval>)>,
        result: Result<(), (ErrorCode, String)>,
    ) {
        match result {
            Ok(()) => {
                if let Some((stream, span, tier, cut)) = added
                    && self.index.add_chunk_copy(chunk, stream, span, tier, &cut).is_err()
                {
                    warn!(chunk = %chunk, stream = stream.0, "a rewritten chunk's stream isn't in the index");
                }
                let bytes = protocol::chunk_rewritten(&mut self.builder, request_id, chunk);
                self.push(session, bytes);
            }
            Err((code, message)) => self.reply_error(session, request_id, code, &message),
        }
    }
}

/// The error code a catalog refusal goes back as.
fn code_of(e: &CatalogError) -> ErrorCode {
    match e {
        CatalogError::NotFound(_) => ErrorCode::NotFound,
        CatalogError::Conflict(_) => ErrorCode::Conflict,
        CatalogError::Invalid(_) => ErrorCode::BadRequest,
        _ => ErrorCode::Internal,
    }
}

/// On the writer: commits a chunk, or a part of one still being written, and reports it to the scheduler task. A
/// chunk the catalog has already, with the same header, is answered as a duplicate (with its commit sequence), so
/// the worker can retry after a lost reply; the catalog answers a repeated part the same way.
pub(super) fn commit_chunk_job(
    catalog: &dyn Catalog,
    session: SessionId,
    request_id: u64,
    ask: protocol::ChunkCommitAsk,
    events: &mpsc::UnboundedSender<Event>,
) {
    let header = &ask.chunk.payload.header;
    let chunk = header.id();
    let span = Interval::new(header.start_ns(), header.end_ns().max(header.start_ns().saturating_add(1)));
    let tier = ask.chunk.payload.location.as_ref().map(StorageLocator::tier);
    let known = catalog.get(chunk);
    let (copy, result) = if ask.first_frame > 0 {
        // The index gains only what the part adds: the chunk's span from its old end, and the part's cut slots.
        let old_end = match &known {
            Ok(Some(continuum_catalog::Resolved::Payload(k))) => k.header.end_ns().max(k.header.start_ns() + 1),
            _ => header.start_ns(),
        };
        let slots = span.map(|s| continuum_catalog::frame_slots(s, &ask.chunk.frames)).unwrap_or_default();
        let added = Interval::new(old_end.min(header.end_ns()), header.end_ns());
        let copy = added.zip(tier).map(|(added, tier)| (header.stream(), added, tier, slots));
        let first_frame = ask.first_frame;
        let result = catalog
            .extend_chunk(ask.chunk, first_frame, &ask.absorbed, ask.provenance.as_ref())
            .map(|mut inserted| {
                // The catalog names cut frames by their place in the whole chunk; the slots here are the part's.
                inserted.cut = inserted.cut.iter().filter_map(|i| i.checked_sub(first_frame)).collect();
                inserted
            })
            .map_err(|e| (code_of(&e), e.to_string()));
        (copy, result)
    } else {
        let slots = span.map(|s| continuum_catalog::frame_slots(s, &ask.chunk.frames)).unwrap_or_default();
        let copy = span.zip(tier).map(|(span, tier)| (header.stream(), span, tier, slots));
        let result = match known {
            // A retry after a lost reply: the first commit compacted the frames, so the catalog would refuse them
            // now. The same chunk is a duplicate; another chunk under its id is a conflict.
            Ok(Some(continuum_catalog::Resolved::Payload(known))) if known.header.as_bytes() == header.as_bytes() => {
                Ok(continuum_catalog::Inserted {
                    seq: known.commit_seq,
                    hidden: false,
                    duplicate: true,
                    cut: Vec::new(),
                })
            }
            Ok(Some(_)) => Err((ErrorCode::Conflict, format!("{chunk} is known already with other contents"))),
            Ok(None) => catalog
                .commit_chunk_with_provenance(ask.chunk, &ask.absorbed, ask.provenance.as_ref())
                .map_err(|e| (code_of(&e), e.to_string())),
            Err(e) => Err((code_of(&e), e.to_string())),
        };
        (copy, result)
    };
    let _ = events.send(Event::ChunkCommitted { session, request_id, chunk, copy, result });
}

/// On the writer: records a rewritten chunk and reports it to the scheduler task.
pub(super) fn rewrite_chunk_job(
    catalog: &dyn Catalog,
    job: (SessionId, u64, Uuid, Vec<continuum_catalog::ChunkFrame>, StorageLocator),
    events: &mpsc::UnboundedSender<Event>,
) {
    let (session, request_id, chunk, frames, location) = job;
    let tier = location.tier();
    // A rewrite into a tier the chunk had no copy in (a recompression) gives it a copy there, for the index.
    let added = match catalog.get(chunk) {
        Ok(Some(continuum_catalog::Resolved::Payload(known))) if !known.locations.iter().any(|l| l.tier() == tier) => {
            let header = &known.header;
            Interval::new(header.start_ns(), header.end_ns().max(header.start_ns().saturating_add(1))).map(|span| {
                let slots = continuum_catalog::frame_slots(span, &frames);
                let cut = frames.iter().zip(slots).filter(|(f, _)| f.cut).map(|(_, s)| s).collect();
                (header.stream(), span, tier, cut)
            })
        }
        _ => None,
    };
    let result = catalog.rewrite_chunk(chunk, frames, location).map_err(|e| (code_of(&e), e.to_string()));
    let _ = events.send(Event::ChunkRewritten { session, request_id, chunk, added, result });
}
