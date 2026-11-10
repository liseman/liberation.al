//! Aging (docs/CONTINUUM-STORAGE.md): the reads the video window makes (`ListAgedOut`, `CommitSeqAt`), and the
//! catalog reads that share its reply path: `GetChunk`, `GetCatalogStats`, recording applicability pages, and
//! `ResolveFrames`, the source identity an analysis receipt names. An age-out of whole payloads is a `Purge`
//! (see `core::forget`).

use super::*;

/// What an aging request came back with, from the writer or a reader.
pub(super) enum Done {
    /// A `ListAgedOut`.
    Spans(Vec<Interval>),
    /// A `CommitSeqAt`.
    Seq(u64),
    /// A `GetChunk`: the chunk, its copies, and its provenance.
    Chunk(Box<continuum_catalog::ChunkSnapshot>),
    /// One coherent original recording applicability page.
    Recording(continuum_catalog::RecordingPage, StreamId, usize),
    /// A `ResolveFrames`, in the asked order.
    Resolved(Vec<protocol::aging::FrameIdentity>),
    /// A `GetCatalogStats` (empty for a catalog that keeps no file).
    Stats(Box<continuum_catalog::StorageStats>),
}

impl Sched {
    pub(super) fn list_recording_inputs(
        &mut self,
        session: SessionId,
        request_id: u64,
        ask: protocol::applicability::ListRecordingAsk,
    ) {
        let Some(id) = self.index.stream_named(&ask.stream) else {
            return self.reply_error(session, request_id, ErrorCode::NotFound, "recording source is not registered");
        };
        if id != ask.stream_id {
            return self.reply_error(session, request_id, ErrorCode::Conflict, "recording source identity changed");
        }
        self.aging_read(session, request_id, move |catalog| {
            catalog
                .recording_page(
                    id,
                    ask.window,
                    ask.after,
                    ask.expected_revision,
                    ask.max_items,
                    ask.max_scan,
                    ask.max_bytes,
                )
                .map(|page| Done::Recording(page, id, ask.max_bytes))
        });
    }

    pub(super) fn list_aged_out(
        &mut self,
        session: SessionId,
        request_id: u64,
        stream: String,
        window: Interval,
        limit: usize,
    ) {
        let Some(id) = self.index.stream_named(&stream) else {
            return self.reply_error(session, request_id, ErrorCode::NotFound, &format!("no stream named {stream}"));
        };
        self.aging_read(session, request_id, move |catalog| catalog.aged_out(id, window, limit).map(Done::Spans));
    }

    /// Walks every table's pages in one read snapshot, on the readers, so the writer never waits for it.
    pub(super) fn catalog_stats(&mut self, session: SessionId, request_id: u64) {
        self.aging_read(session, request_id, |catalog| {
            Ok(Done::Stats(Box::new(catalog.storage_stats()?.unwrap_or_default())))
        });
    }

    pub(super) fn commit_seq_at(&mut self, session: SessionId, request_id: u64, at_ns: u64) {
        self.aging_read(session, request_id, move |catalog| catalog.commit_seq_at(at_ns).map(Done::Seq));
    }

    pub(super) fn get_chunk(&mut self, session: SessionId, request_id: u64, chunk: Uuid) {
        let events = self.events.clone();
        let job: ReadJob = Box::new(move |catalog| {
            let result = chunk_snapshot(catalog, chunk);
            let _ = events.send(Event::Aging { session, request_id, result });
        });
        if let Err(why) = self.read(job) {
            self.reply_error(session, request_id, ErrorCode::Unavailable, why);
        }
    }

    /// A `ResolveFrames`, on a reader: each id's identity, from the catalog's frame rows.
    pub(super) fn resolve_frames(&mut self, session: SessionId, request_id: u64, ids: Vec<Uuid>) {
        self.aging_read(session, request_id, move |catalog| {
            let mut out = Vec::with_capacity(ids.len());
            for id in ids {
                out.push(protocol::aging::FrameIdentity::of(id, catalog.get(id)?.as_ref()));
            }
            Ok(Done::Resolved(out))
        });
    }

    fn aging_read(
        &mut self,
        session: SessionId,
        request_id: u64,
        f: impl FnOnce(&dyn Catalog) -> continuum_catalog::Result<Done> + Send + 'static,
    ) {
        let events = self.events.clone();
        let job: ReadJob = Box::new(move |catalog| {
            let result = f(catalog).map_err(|e| (error_code(&e), e.to_string()));
            let _ = events.send(Event::Aging { session, request_id, result });
        });
        if let Err(why) = self.read(job) {
            self.reply_error(session, request_id, ErrorCode::Unavailable, why);
        }
    }

    /// An aging request's answer.
    pub(super) fn aging_done(
        &mut self,
        session: SessionId,
        request_id: u64,
        result: Result<Done, (ErrorCode, String)>,
    ) {
        let bytes = match result {
            Ok(Done::Spans(spans)) => protocol::aging::aged_out_list(&mut self.builder, request_id, &spans),
            Ok(Done::Seq(seq)) => protocol::aging::commit_seq(&mut self.builder, request_id, seq),
            Ok(Done::Stats(stats)) => protocol::aging::catalog_stats(&mut self.builder, request_id, &stats),
            Ok(Done::Chunk(snapshot)) => protocol::aging::chunk_info(&mut self.builder, request_id, &snapshot),
            Ok(Done::Recording(page, stream, budget)) => {
                match protocol::applicability::page(&mut self.builder, request_id, stream, &page, budget) {
                    Ok(bytes) => bytes,
                    Err(message) => return self.reply_error(session, request_id, ErrorCode::BadRequest, &message),
                }
            }
            Ok(Done::Resolved(frames)) => protocol::aging::frames_resolved(&mut self.builder, request_id, &frames),
            Err((code, message)) => return self.reply_error(session, request_id, code, &message),
        };
        self.push(session, bytes);
    }
}

/// GetChunk reads the chunk, its copies, and its provenance from one catalog snapshot.
pub(super) fn chunk_snapshot(catalog: &dyn Catalog, chunk: Uuid) -> Result<Done, (ErrorCode, String)> {
    match catalog.chunk_snapshot(chunk) {
        Ok(Some(snapshot)) => Ok(Done::Chunk(Box::new(snapshot))),
        Ok(None) => Err((ErrorCode::NotFound, format!("no chunk {chunk}"))),
        Err(e) => Err((error_code(&e), e.to_string())),
    }
}
