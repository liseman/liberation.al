//! The vector worker's loop: dispatched searches in, result payloads out, purge notices acknowledged (#546;
//! docs/CONTINUUM.md, "Vector worker").
//!
//! The worker registers two capabilities, [`VECTOR_SEARCH`] and [`TEXT_SEARCH`]. The broker dispatches a job whose
//! one input is a query payload ([`VECTOR_QUERY`] or [`TEXT_QUERY`]); [`SearchWorker::handle`] answers it from the
//! store and commits one result payload on the capability's output stream, with the query as its parent, over the
//! query's own time span. A job it can't answer (the wrong number of inputs, the wrong shape, bytes it can't read, a
//! query the store refuses) is turned down with the reason, and the broker replans.
//!
//! A purge notice names payload ids: the worker removes their rows completely, then acknowledges, so the broker
//! finishes the purge only once this copy is gone. A purge that fails is not acknowledged; the broker keeps it open.
//! The notice's spans don't matter here: they name copies inside chunk files (a frame), and every row this worker
//! keeps is a whole payload, named by its id. The worker registers with `keeps_copies` ([`worker_spec`]) so the
//! broker waits for its ack.
//!
//! Embeddings come in by a durable subscription ([`SearchWorker::index`]): each pinned, inline event whose shape is
//! `embedding/<model>/<version>` is filed, then acknowledged, so a restarted worker resumes after the last one filed.
//! The store drops a row for anything already purged ([`VectorStore::purge_payloads`]), since a purge can arrive
//! while an embedding it names still waits here.

use std::sync::Arc;

use continuum_client::{
    BrokerEvent, CapabilitySpec, Client, CoverageDone, DispatchJob, Item, Requirement, StreamSpec, StreamStorage,
    SubEvent, Subscription, WorkerSpec,
};
use continuum_core::schema::continuum as fb;
use continuum_core::{DomainLayer, HeaderBuf, HeaderSpec, PayloadFormat, StorageLocator, StorageTier, StreamId, Uuid};

use crate::jobs::{
    EMBEDDING_SHAPE_PREFIX, TEXT_QUERY, TEXT_SEARCH, VECTOR_QUERY, VECTOR_SEARCH, answer_text_query,
    answer_vector_query, decode_hits, index_embedding, index_text, purge,
};
use crate::{Key, VectorStore};

/// The durable name the embedding subscription saves its position under.
pub const INDEX_SUBSCRIPTION: &str = "vector-worker.embeddings";

/// What the vector worker registers: its two search capabilities (on CPU, one query per job), and that it keeps
/// copies, so purges wait for it.
#[must_use]
pub fn worker_spec(version_hash: &str) -> WorkerSpec {
    // Each takes its query payload as its one input: the broker builds a job's inputs from `requires`.
    let search = |produces: &str, query: &str| CapabilitySpec {
        requires: vec![Requirement { stream: query.into(), before_ns: 0, after_ns: 0 }],
        est_ms_per_item: 20.0,
        ..CapabilitySpec::new(produces, fb::ComputeDomain::CPU)
    };
    WorkerSpec {
        worker_id: "worker:vector".into(),
        version_hash: version_hash.into(),
        capabilities: vec![search(VECTOR_SEARCH, VECTOR_QUERY), search(TEXT_SEARCH, TEXT_QUERY)],
        device_index: None,
        keeps_copies: true,
    }
}

/// The version of the result payload's shape ([`crate::jobs::encode_hits`]).
pub const RESULT_SHAPE_VERSION: u32 = 1;

/// What went wrong with one event.
#[derive(Debug, thiserror::Error)]
pub enum WorkerError {
    /// A job this worker can't answer; it was turned down with this reason.
    #[error("job {job}: {reason}")]
    Rejected {
        /// The job.
        job: Uuid,
        /// Why.
        reason: String,
    },
    /// The store failed on a purge; the notice was not acknowledged.
    #[error("purge {purge}: {source}")]
    Purge {
        /// The purge.
        purge: Uuid,
        /// The store's error.
        source: crate::Error,
    },
    /// The store failed to file an embedding; the event was not acknowledged.
    #[error("indexing at position {position}: {source}")]
    Index {
        /// The event's position.
        position: u64,
        /// The store's error.
        source: crate::Error,
    },
    /// The broker connection failed.
    #[error("broker: {0}")]
    Client(String),
}

/// How the query and result streams are registered (`vector.query`, `vector.search`, `text.query`, `text.search`): a
/// consumer registers a query stream the same way before committing to it. Queries are layer 1, results layer 2.
#[must_use]
pub fn search_stream(name: &str) -> StreamSpec {
    let layer = if name == VECTOR_QUERY || name == TEXT_QUERY {
        fb::DomainLayer::Layer1Raw
    } else {
        fb::DomainLayer::Layer2Decision
    };
    StreamSpec {
        name: name.into(),
        kind: fb::StreamKind::Derived,
        format: fb::PayloadFormat::ArrowRecordBatch,
        shape_name: name.into(),
        shape_version: RESULT_SHAPE_VERSION,
        nominal_rate_hz: 0.0,
        storage: StreamStorage::Slab { class: 0 },
        anchor: None,
        layer,
    }
}

/// Registers the query and result streams, then the worker ([`worker_spec`]), and returns the streams' ids. Registering a
/// stream again by the same name returns the same id, so a restarted worker (a new session) calls this every time; a
/// session registers once.
///
/// # Errors
///
/// [`WorkerError::Client`] when the broker refuses a stream or either capability.
pub async fn register(client: &Client, version_hash: &str) -> Result<SearchStreams, WorkerError> {
    let failed = |e: continuum_client::Error| WorkerError::Client(e.to_string());
    // The query streams the capabilities require must exist too. Registering one doesn't take it over: any session
    // that registers a stream may commit to it, so consumers register them as well and get the same ids.
    for query in [VECTOR_QUERY, TEXT_QUERY] {
        client.register_stream(&search_stream(query)).await.map_err(failed)?;
    }
    let vector = client.register_stream(&search_stream(VECTOR_SEARCH)).await.map_err(failed)?;
    let text = client.register_stream(&search_stream(TEXT_SEARCH)).await.map_err(failed)?;
    let (vector, text) = (vector.stream_id, text.stream_id);
    let registered = client.register_worker(&worker_spec(version_hash)).await.map_err(failed)?;
    if registered.accepted.len() != 2 {
        return Err(WorkerError::Client(format!("the broker refused: {}", registered.refused_reasons.join("; "))));
    }
    Ok(SearchStreams { vector: StreamId(vector), text: StreamId(text) })
}

/// The output streams the two capabilities write, as the broker registered them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchStreams {
    /// Where [`VECTOR_SEARCH`] results go.
    pub vector: StreamId,
    /// Where [`TEXT_SEARCH`] results go.
    pub text: StreamId,
}

/// The vector worker, on a connected client.
pub struct SearchWorker {
    client: Arc<Client>,
    store: VectorStore,
    streams: SearchStreams,
}

impl SearchWorker {
    /// A worker answering from `store` and writing to `streams`.
    pub fn new(client: Arc<Client>, store: VectorStore, streams: SearchStreams) -> Self {
        Self { client, store, streams }
    }

    /// Its registered RPC connection; subscription readers are separate and supplied to the intake methods.
    #[must_use]
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// Its store.
    #[must_use]
    pub fn store(&self) -> &VectorStore {
        &self.store
    }

    /// Handles events until the connection closes. An event that fails is logged by the caller's `on_error` and the
    /// loop goes on: one bad job never stops the worker.
    pub async fn run(&self, mut on_error: impl FnMut(WorkerError)) {
        while let Some(message) = self.client.next_event().await {
            if let Err(e) = self.handle(message.event()).await {
                on_error(e);
            }
        }
    }

    /// Handles one event: answers a dispatched search, or carries out a purge notice. Other events are ignored.
    ///
    /// # Errors
    ///
    /// [`WorkerError::Rejected`] for a job it turned down, [`WorkerError::Purge`] for a purge the store failed, and
    /// [`WorkerError::Client`] when the broker connection failed.
    pub async fn handle(&self, event: BrokerEvent) -> Result<(), WorkerError> {
        match event {
            BrokerEvent::DispatchJob(mut job) => self.answer(&mut job).await,
            BrokerEvent::PurgeNotice { purge_id, payload_ids, .. } => {
                let ids: Vec<uuid::Uuid> = payload_ids.iter().map(|u| uuid::Uuid::from_bytes(u.0)).collect();
                purge(&self.store, &ids).await.map_err(|source| WorkerError::Purge { purge: purge_id, source })?;
                self.client.ack_purge(purge_id).map_err(|e| WorkerError::Client(e.to_string()))
            }
            _ => Ok(()),
        }
    }

    /// Answers queries committed outside a dispatched job, until the subscription ends: the interim path while the
    /// SDKs have no consumer promise calls (docs/CONTINUUM.md, "Vector worker"). Subscribe live and pinned to
    /// [`VECTOR_QUERY`] and [`TEXT_QUERY`]. Each query is answered with one result payload on the matching result
    /// stream, its parent the query, committed by this worker; the consumer waits for the result whose parent is its
    /// query. A query that can't be answered (not inline, bad bytes, a space the store refuses) is passed to
    /// `failed`, and the consumer's wait times out. Nothing is acknowledged: a query nobody answered while the worker
    /// was down has already timed out.
    ///
    /// # Errors
    ///
    /// [`WorkerError::Client`] when the broker connection closes.
    pub async fn serve(&self, sub: &mut Subscription, mut failed: impl FnMut(u64, String)) -> Result<(), WorkerError> {
        while let Some(event) = sub.next().await {
            let SubEvent::Event { position, item } = event else {
                if let SubEvent::Undecodable { position, reason } = event {
                    failed(position, reason);
                }
                continue;
            };
            let (shape, stream) = match item.header.shape_name() {
                VECTOR_QUERY => (VECTOR_SEARCH, self.streams.vector),
                TEXT_QUERY => (TEXT_SEARCH, self.streams.text),
                other => {
                    failed(position, format!("shape {other:?} isn't a query"));
                    continue;
                }
            };
            let Some(StorageLocator::Inline { bytes }) = &item.locator else {
                failed(position, "a query must be committed inline".into());
                continue;
            };
            let answered = if shape == VECTOR_SEARCH {
                answer_vector_query(&self.store, bytes).await
            } else {
                answer_text_query(&self.store, bytes).await
            };
            let result = match answered {
                Ok(result) => result,
                Err(e) => {
                    failed(position, e.to_string());
                    continue;
                }
            };
            let header = match result_header(&item.header, stream, shape, item.locator.as_ref(), &result) {
                Ok(header) => header,
                Err(e) => {
                    failed(position, e.to_string());
                    continue;
                }
            };
            let Some(at) = StorageLocator::inline(result) else {
                failed(position, "the result is larger than an inline payload".into());
                continue;
            };
            // One refused or failed commit (the broker busy, say) costs that query only; a closed connection ends it.
            match self.client.ingest_acked(&header, Some(&at)).await {
                Ok(_) => {}
                Err(continuum_client::Error::Closed) => {
                    return Err(WorkerError::Client("the broker connection closed".into()));
                }
                Err(e) => failed(position, format!("committing the answer: {e}")),
            }
        }
        Ok(())
    }

    /// Files the embeddings a subscription delivers until it ends. Subscribe pinned, durable under
    /// [`INDEX_SUBSCRIPTION`] from `Start::Saved`, to the embedding streams. An event that isn't a usable embedding
    /// (another shape, not inline, a vector of the wrong form), or one that can't be decoded, is passed to `skipped`
    /// and acknowledged, so it never blocks the ones after it.
    ///
    /// # Errors
    ///
    /// [`WorkerError::Index`] when the store fails: the event isn't acknowledged, so a restart files it again.
    /// [`WorkerError::Client`] when an acknowledgement can't be sent.
    pub async fn index(&self, sub: &mut Subscription, skipped: impl FnMut(u64, String)) -> Result<(), WorkerError> {
        self.index_from(sub, 0, |_| {}, skipped).await
    }

    /// A reader reconnect retains successfully handled positions even if its final acknowledgement was lost.
    pub(crate) async fn index_from(
        &self,
        sub: &mut Subscription,
        after: u64,
        mut handled: impl FnMut(u64),
        mut skipped: impl FnMut(u64, String),
    ) -> Result<(), WorkerError> {
        while let Some(event) = sub.next().await {
            let (position, item) = match event {
                SubEvent::Event { position, .. } | SubEvent::Undecodable { position, .. }
                    if after > 0 && position <= after =>
                {
                    continue;
                }
                SubEvent::Event { position, item } => (position, item),
                SubEvent::Undecodable { position, reason } => {
                    skipped(position, reason);
                    handled(position);
                    sub.ack(position).map_err(|e| WorkerError::Client(e.to_string()))?;
                    continue;
                }
                SubEvent::CaughtUp { .. } => continue,
            };
            let shape = item.header.shape_name();
            if !shape.starts_with(EMBEDDING_SHAPE_PREFIX) {
                skipped(position, format!("shape {shape:?} isn't an embedding"));
            } else {
                // Embeddings are committed inline (a few KB at most), so their bytes live in the catalog for as long as
                // the payload does: never a Tier 0 slot that may be reused before a lagging intake reads it, nor a chunk.
                match &item.locator {
                    Some(StorageLocator::Inline { bytes }) => {
                        let h = &item.header;
                        let key = Key {
                            payload_id: uuid::Uuid::from_bytes(h.id().0),
                            entity_id: h.subjects().first().map(|s| uuid::Uuid::from_bytes(s.0)),
                            parent_id: h.parents().first().map(|p| uuid::Uuid::from_bytes(p.0)),
                            start_ns: h.start_ns(),
                            end_ns: h.end_ns(),
                        };
                        match index_embedding(&self.store, key, shape, bytes).await {
                            Ok(()) => {}
                            // A bad vector or shape name is the payload's fault; a LanceDB failure is the store's.
                            Err(crate::Error::Invalid(reason)) => skipped(position, reason),
                            Err(source) => return Err(WorkerError::Index { position, source }),
                        }
                    }
                    _ => skipped(position, "an embedding must be committed inline".into()),
                }
            }
            handled(position);
            sub.ack(position).map_err(|e| WorkerError::Client(e.to_string()))?;
        }
        Ok(())
    }

    /// Files the searchable text of each interpretation a subscription delivers (`journal.entry`: an inline JSON body
    /// with a `text` field and optional rendered `search`) for full-text search, acknowledging each event once filed.
    /// One row per subject, so a search
    /// filtered by person finds every entry they're in (a row has one entity); an entry about nobody is one row with no
    /// entity. A quiet window's marker, or an entry without text, files nothing. Returns when the subscription ends.
    ///
    /// # Errors
    ///
    /// [`WorkerError::Index`] when the store fails, before that event is acknowledged.
    pub async fn index_texts(
        &self,
        sub: &mut Subscription,
        skipped: impl FnMut(u64, String),
    ) -> Result<(), WorkerError> {
        self.index_texts_from(sub, 0, |_| {}, skipped).await
    }

    /// Text intake uses the same in-process checkpoint as embedding intake, without changing store semantics.
    pub(crate) async fn index_texts_from(
        &self,
        sub: &mut Subscription,
        after: u64,
        mut handled: impl FnMut(u64),
        mut skipped: impl FnMut(u64, String),
    ) -> Result<(), WorkerError> {
        while let Some(event) = sub.next().await {
            let (position, item) = match event {
                SubEvent::Event { position, .. } | SubEvent::Undecodable { position, .. }
                    if after > 0 && position <= after =>
                {
                    continue;
                }
                SubEvent::Event { position, item } => (position, item),
                SubEvent::Undecodable { position, reason } => {
                    skipped(position, reason);
                    handled(position);
                    sub.ack(position).map_err(|e| WorkerError::Client(e.to_string()))?;
                    continue;
                }
                SubEvent::CaughtUp { .. } => continue,
            };
            match &item.locator {
                Some(StorageLocator::Inline { bytes }) => match text_of(bytes) {
                    Ok(Some(text)) => {
                        let h = &item.header;
                        let key = |entity: Option<uuid::Uuid>| Key {
                            payload_id: uuid::Uuid::from_bytes(h.id().0),
                            entity_id: entity,
                            parent_id: h.parents().first().map(|p| uuid::Uuid::from_bytes(p.0)),
                            start_ns: h.start_ns(),
                            end_ns: h.end_ns(),
                        };
                        let subjects: Vec<Option<uuid::Uuid>> =
                            h.subjects().iter().map(|s| Some(uuid::Uuid::from_bytes(s.0))).collect();
                        let entities = if subjects.is_empty() { vec![None] } else { subjects };
                        for entity in entities {
                            match index_text(&self.store, key(entity), &text).await {
                                Ok(()) => {}
                                Err(crate::Error::Invalid(reason)) => skipped(position, reason),
                                Err(source) => return Err(WorkerError::Index { position, source }),
                            }
                        }
                    }
                    Ok(None) => {}
                    Err(reason) => skipped(position, reason),
                },
                _ => skipped(position, "an interpretation's text must be committed inline".into()),
            }
            handled(position);
            sub.ack(position).map_err(|e| WorkerError::Client(e.to_string()))?;
        }
        Ok(())
    }

    async fn answer(&self, job: &mut DispatchJob) -> Result<(), WorkerError> {
        let job_id = job.job_id;
        let reject = |reason: String| {
            // The broker replans either way; a closed connection is reported by the next request.
            let _ = self.client.reject_job(job_id, &reason);
            WorkerError::Rejected { job: job_id, reason }
        };
        let (want, stream) = match job.capability.as_str() {
            VECTOR_SEARCH => (VECTOR_QUERY, self.streams.vector),
            TEXT_SEARCH => (TEXT_QUERY, self.streams.text),
            other => return Err(reject(format!("this worker doesn't make {other:?}"))),
        };
        let [query] = job.inputs.as_slice() else {
            return Err(reject(format!("a search takes one query, not {}", job.inputs.len())));
        };
        // Owned, since taking the job's lease below borrows the job mutably.
        let query = query.clone();
        if query.header.shape_name() != want {
            return Err(reject(format!("{} answers {want}, not {:?}", job.capability, query.header.shape_name())));
        }
        let bytes = self.bytes_of(&query).map_err(reject)?;
        let answered = if want == VECTOR_QUERY {
            answer_vector_query(&self.store, &bytes).await
        } else {
            answer_text_query(&self.store, &bytes).await
        };
        let result = answered.map_err(|e| reject(e.to_string()))?;
        let lease = self.client.job_lease(job).map_err(|e| reject(e.to_string()))?;
        let header = result_header(&query.header, stream, &job.capability, query.locator.as_ref(), &result)
            .map_err(|e| reject(e.to_string()))?;
        // MAX_K keeps every result inline (continuum-search's tests check a full result fits); the broker hands out
        // no output slots yet.
        let at = StorageLocator::inline(result)
            .ok_or_else(|| reject("the result is larger than an inline payload".into()))?;
        let coverage = CoverageDone {
            source_stream_id: query.header.stream().0,
            stage: job.capability.clone(),
            range: job.output_range,
        };
        lease
            .commit(&[(header, Some(at))], Some(&coverage), None)
            .await
            .map_err(|e| WorkerError::Client(format!("committing job {}: {e}", job.job_id)))
    }

    /// An input's bytes: inline, or copied out of the shared memory slot the broker pinned for the job.
    fn bytes_of(&self, item: &Item) -> Result<Vec<u8>, String> {
        match &item.locator {
            Some(StorageLocator::Inline { bytes }) => Ok(bytes.clone()),
            Some(StorageLocator::Shm { arena, arena_generation, offset, length, .. }) => {
                let mapped = self.client.arena(arena).ok_or_else(|| format!("arena {arena} isn't mapped"))?;
                if mapped.generation() != *arena_generation {
                    return Err(format!("arena {arena} is generation {}, not {arena_generation}", mapped.generation()));
                }
                let range = span_in(*offset, *length, mapped.len())
                    .ok_or_else(|| format!("slot at {offset}+{length} is outside arena {arena}"))?;
                // SAFETY: the range is inside the mapping (checked above), and the broker pins the slot for the job.
                let slice = unsafe { std::slice::from_raw_parts(mapped.base_ptr().add(range.start), range.len()) };
                Ok(slice.to_vec())
            }
            Some(StorageLocator::Chunk { .. } | StorageLocator::Ring { .. }) => {
                Err("a payload in a chunk file or a ring batch isn't supported; it should be inline or in a slot"
                    .into())
            }
            None => Err("the query payload has no bytes".into()),
        }
    }
}

/// `offset..offset + length` when it lies inside `len` bytes.
fn span_in(offset: u64, length: u64, len: usize) -> Option<std::ops::Range<usize>> {
    let start = usize::try_from(offset).ok()?;
    let end = start.checked_add(usize::try_from(length).ok()?)?;
    (end <= len).then_some(start..end)
}

/// A result's header: a fresh id on `stream`, over the query's span, with the query as its parent (how a consumer
/// finds its answer). Its subjects are the query's and every hit's entity: a result names those people, so forgetting
/// any of them (a purge by entity) takes the result too. The hits' payloads aren't parents: the broker refuses a
/// parent it doesn't know, so a hit purged a moment before would fail the whole answer, and a hit's payload id names
/// no one.
fn result_header(
    query: &HeaderBuf,
    stream: StreamId,
    shape: &str,
    from: Option<&StorageLocator>,
    result: &[u8],
) -> Result<HeaderBuf, String> {
    let hits = decode_hits(result).map_err(|e| e.to_string())?;
    let mut subjects = query.subjects();
    for h in &hits {
        if let Some(entity) = h.key.entity_id.map(|e| Uuid(e.into_bytes()))
            && !subjects.contains(&entity)
        {
            subjects.push(entity);
        }
    }
    HeaderBuf::build(&HeaderSpec {
        id: Uuid(uuid::Uuid::now_v7().into_bytes()),
        stream,
        start_ns: query.start_ns(),
        end_ns: query.end_ns(),
        // The query's own seq: two queries at one instant (two faces in a frame) differ by seq, and so do their
        // answers, each on its query stream's own result stream.
        seq: query.seq(),
        layer: DomainLayer::Layer2Decision,
        format: PayloadFormat::ArrowRecordBatch,
        shape_name: shape.to_owned(),
        shape_version: RESULT_SHAPE_VERSION,
        parents: vec![query.id()],
        subjects,
        source_tier: from.map_or(StorageTier::Tier0Hot, StorageLocator::tier),
    })
    .map_err(|e| e.to_string())
}

/// The text to file from an interpretation's JSON body: its nonblank `search` field (the journal renders speaker
/// tokens as names there), falling back to `text` when `search` is absent or blank. Quiet markers and entries without
/// written text are never filed, even if they carry a search field.
///
/// # Errors
///
/// A body that isn't a JSON object, a `text` that isn't a string, or a present `search` that isn't a string when
/// there is written text. Bodies without a rendered search field omit it.
pub fn text_of(body: &[u8]) -> Result<Option<String>, String> {
    let value: serde_json::Value = serde_json::from_slice(body).map_err(|e| format!("not JSON: {e}"))?;
    let obj = value.as_object().ok_or("not a JSON object")?;
    if obj.get("status").and_then(serde_json::Value::as_str) == Some("quiet") {
        return Ok(None);
    }
    let text = match obj.get("text") {
        None | Some(serde_json::Value::Null) => return Ok(None),
        Some(serde_json::Value::String(t)) if t.trim().is_empty() => return Ok(None),
        Some(serde_json::Value::String(t)) => t,
        Some(_) => return Err("its text isn't a string".into()),
    };
    let search = match obj.get("search") {
        None => text,
        Some(serde_json::Value::String(s)) if s.trim().is_empty() => text,
        Some(serde_json::Value::String(s)) => s,
        Some(_) => return Err("its search isn't a string".into()),
    };
    Ok(Some(search.clone()))
}

#[cfg(test)]
mod text_tests {
    use super::text_of;

    #[test]
    fn rendered_names_are_indexed_instead_of_stored_speaker_tokens() {
        assert_eq!(
            text_of(br#"{"text":"[S12] planted tomatoes.","search":"Dana planted tomatoes.","level":30}"#),
            Ok(Some("Dana planted tomatoes.".into()))
        );
    }

    #[test]
    fn absent_or_blank_search_keeps_text_but_invalid_search_is_rejected() {
        for body in [
            br#"{"text":"[S12] planted tomatoes."}"#.as_slice(),
            br#"{"text":"[S12] planted tomatoes.","search":""}"#,
            br#"{"text":"[S12] planted tomatoes.","search":"  \n "}"#,
        ] {
            assert_eq!(text_of(body), Ok(Some("[S12] planted tomatoes.".into())));
        }
        for search in ["null", "3", "true", "[]", "{}"] {
            let body = format!(r#"{{"text":"[S12] planted tomatoes.","search":{search}}}"#);
            assert!(text_of(body.as_bytes()).is_err(), "invalid search field: {search}");
        }
    }

    #[test]
    fn search_never_creates_an_entry_without_valid_written_text() {
        for body in [
            br#"{"status":"quiet","text":"Old words","search":"Dana"}"#.as_slice(),
            br#"{"search":"Dana"}"#,
            br#"{"text":null,"search":"Dana"}"#,
            br#"{"text":" \n ","search":"Dana"}"#,
            br#"{"text":"","search":3}"#,
        ] {
            assert_eq!(text_of(body), Ok(None));
        }
        assert!(text_of(br#"{"text":3,"search":"Dana"}"#).is_err());
    }

    #[test]
    fn an_entrys_text_is_filed_and_a_quiet_marker_or_a_bad_body_is_not() {
        assert_eq!(
            text_of(br#"{"text": "Dana planted tomatoes.", "level": 30}"#),
            Ok(Some("Dana planted tomatoes.".into()))
        );
        assert_eq!(text_of(br#"{"status": "quiet", "level": 30}"#), Ok(None));
        assert_eq!(text_of(br#"{"text": "  "}"#), Ok(None));
        assert_eq!(text_of(br#"{"level": 30}"#), Ok(None));
        assert!(text_of(br#"{"text": 3}"#).is_err());
        assert!(text_of(b"[1]").is_err());
        assert!(text_of(b"not json").is_err());
    }
}
