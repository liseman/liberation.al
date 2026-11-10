//! What the vector worker does with a job, apart from talking to the broker (docs/CONTINUUM.md, "Vector
//! worker").
//!
//! A search is a derived stream. A consumer commits a query payload, `vector.query` ([`VectorQuery`]) or `text.query`
//! ([`TextQuery`]), and asks for `vector.search` or `text.search` at that instant; the broker dispatches a job whose
//! input is the query payload, and the worker answers with a result payload ([`encode_hits`]) whose parent is the
//! query. Query and result payloads are Arrow IPC streams (one record batch); coalescing and provenance come from the
//! broker like any other derived stream.
//!
//! Embeddings come in as payloads other workers commit: format `RawBytes`, the vector as little-endian f32, and the
//! shape `embedding/<model>/<version>` ([`space_of_shape`]), which names the embedding space; the header's first
//! subject, if any, is the entity. [`index_embedding`] files one as a row keyed by its payload id.
//!
//! A purge notice names payload ids; [`purge`] removes their rows completely (see [`VectorStore::purge_payloads`])
//! before the worker acks.

pub use continuum_search::{
    EMBEDDING_SHAPE_PREFIX, MAX_K, TEXT_QUERY, TEXT_SEARCH, TextQuery, VECTOR_QUERY, VECTOR_SEARCH, VectorQuery,
    check_k, decode_hits, embedding_vector, encode_hits,
};
use uuid::Uuid;

use crate::{EmbeddingRow, EmbeddingSpace, Error, Key, Result, TextRow, VectorStore};

impl From<continuum_search::Error> for Error {
    fn from(e: continuum_search::Error) -> Self {
        Error::Invalid(e.0)
    }
}

fn invalid(what: impl std::fmt::Display) -> Error {
    Error::Invalid(what.to_string())
}

/// The embedding space a vector query searches: its model and version, and its vector's length.
#[must_use]
pub fn space_of(q: &VectorQuery) -> EmbeddingSpace {
    EmbeddingSpace { model: q.model.clone(), version: q.version.clone(), dim: q.vector.len() }
}

/// The embedding space an embedding payload's shape names (`embedding/<model>/<version>`), for a vector of `dim`.
///
/// # Errors
///
/// [`Error::Invalid`] for another shape, or a model or version the store can't name a table for.
pub fn space_of_shape(shape: &str, dim: usize) -> Result<EmbeddingSpace> {
    let rest =
        shape.strip_prefix(EMBEDDING_SHAPE_PREFIX).ok_or_else(|| invalid(format!("not an embedding: {shape}")))?;
    // The version is after the last slash: a model name may have one of its own (`user/model:tag`), a version never.
    let (model, version) = rest.rsplit_once('/').ok_or_else(|| invalid(format!("no version in {shape}")))?;
    let space = EmbeddingSpace { model: model.to_owned(), version: version.to_owned(), dim };
    space.table_name()?;
    Ok(space)
}

/// Answers a `vector.query` payload with a `vector.search` result payload.
///
/// # Errors
///
/// [`Error::Invalid`] for a bad query; LanceDB errors as they come.
pub async fn answer_vector_query(store: &VectorStore, query: &[u8]) -> Result<Vec<u8>> {
    let q = VectorQuery::decode(query)?;
    check_k(q.k)?;
    let hits = store.search(&space_of(&q), &q.vector, q.k as usize, &q.filter).await?;
    Ok(encode_hits(&hits)?)
}

/// Answers a `text.query` payload with a `text.search` result payload.
///
/// # Errors
///
/// [`Error::Invalid`] for a bad query; LanceDB errors as they come.
pub async fn answer_text_query(store: &VectorStore, query: &[u8]) -> Result<Vec<u8>> {
    let q = TextQuery::decode(query)?;
    check_k(q.k)?;
    let hits = store.text_search(&q.text, q.k as usize, &q.filter).await?;
    Ok(encode_hits(&hits)?)
}

/// Files an embedding payload: its vector, in the space its shape names, keyed by its payload id, entity, and time.
///
/// # Errors
///
/// [`Error::Invalid`] for a payload that isn't an embedding; LanceDB errors as they come.
pub async fn index_embedding(store: &VectorStore, key: Key, shape: &str, bytes: &[u8]) -> Result<()> {
    let vector = embedding_vector(bytes)?;
    let space = space_of_shape(shape, vector.len())?;
    store.insert_embeddings(&space, &[EmbeddingRow { key, vector }]).await
}

/// Files an interpretation's text for full-text search.
///
/// # Errors
///
/// LanceDB errors as they come.
pub async fn index_text(store: &VectorStore, key: Key, text: &str) -> Result<()> {
    store.insert_texts(&[TextRow { key, text: text.to_owned() }]).await
}

/// Handles a purge notice: removes every row from these payloads, completely. The worker acks only after this
/// returns `Ok`; on an error it doesn't ack, and the broker sends the notice again.
///
/// # Errors
///
/// LanceDB errors as they come.
pub async fn purge(store: &VectorStore, payload_ids: &[Uuid]) -> Result<u64> {
    store.purge_payloads(payload_ids).await
}
