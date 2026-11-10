//! Continuum's search payloads, shared by the vector worker (`continuum-vector`) and the SDKs (`continuum_py`), so
//! both sides read and write the same bytes (docs/CONTINUUM-STORAGE.md). Arrow only, no LanceDB.
//!
//! A consumer commits a `vector.query` ([`VectorQuery`]) or `text.query` ([`TextQuery`]) payload, one row of Arrow
//! IPC, and the vector worker answers with one `vector.search` or `text.search` payload ([`encode_hits`]) whose
//! parent is the query. An embedding payload is `RawBytes`: the vector as little-endian f32 ([`embedding_vector`]).

use std::sync::Arc;

use arrow_array::builder::{Float32Builder, ListBuilder, StringBuilder};
use arrow_array::{
    Array, BooleanArray, FixedSizeBinaryArray, Float32Array, ListArray, RecordBatch, StringArray, UInt32Array,
    UInt64Array,
};
use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::StreamWriter;
use arrow_schema::{DataType, Field, Schema};
use uuid::Uuid;

/// A payload that can't be encoded or read, and why.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct Error(pub String);

/// This crate's result.
pub type Result<T> = std::result::Result<T, Error>;

/// The fields every row has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Key {
    /// The payload the row was made from.
    pub payload_id: Uuid,
    /// The person or thing it is about, if one.
    pub entity_id: Option<Uuid>,
    /// The payload's first parent, if it has one (for a voiceprint, the utterance), so a consumer can tell two rows
    /// made from the same source apart from two sources.
    pub parent_id: Option<Uuid>,
    /// The time it covers, UTC nanoseconds, half-open; `end_ns == start_ns` for an instant (decision 9).
    pub start_ns: u64,
    /// See `start_ns`.
    pub end_ns: u64,
}

/// Which rows a search considers. Unset fields don't filter.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Filter {
    /// Rows whose time overlaps `[start_ns, end_ns)`.
    pub during: Option<(u64, u64)>,
    /// Rows about any of these entities.
    pub subjects: Vec<Uuid>,
    /// Only rows with an entity (`Some(true)`: enrolled faces, named voices), or only rows without one
    /// (`Some(false)`: unknowns to cluster).
    pub labelled: Option<bool>,
}

/// One search result.
#[derive(Clone, Debug, PartialEq)]
pub struct Hit {
    /// The row's payload, entity, and time.
    pub key: Key,
    /// Distance for a vector search (smaller is nearer), relevance for a text search (larger is better).
    pub score: f32,
}

/// The query stream for vector search, and the derived stream that answers it.
pub const VECTOR_QUERY: &str = "vector.query";
/// The most hits one query asks for: what kept a result under the broker's old 4 KiB inline limit (about 72 bytes a
/// hit and 700 of schema). The limit is 64 KiB now (continuum_core::MAX_INLINE_BYTES), so this can rise to several
/// hundred.
pub const MAX_K: u32 = 40;

/// Refuses a `k` of 0 (nothing to answer) or more than [`MAX_K`].
pub fn check_k(k: u32) -> Result<()> {
    if (1..=MAX_K).contains(&k) {
        Ok(())
    } else {
        Err(invalid(format!("k is {k}; a query asks for 1 to {MAX_K} hits")))
    }
}

/// The derived stream answering [`VECTOR_QUERY`].
pub const VECTOR_SEARCH: &str = "vector.search";
/// The query stream for full-text search over interpretation text.
pub const TEXT_QUERY: &str = "text.query";
/// The derived stream answering [`TEXT_QUERY`].
pub const TEXT_SEARCH: &str = "text.search";
/// The shape of an embedding payload is this, then `<model>/<version>`.
pub const EMBEDDING_SHAPE_PREFIX: &str = "embedding/";

fn invalid(what: impl std::fmt::Display) -> Error {
    Error(what.to_string())
}

fn to_ipc(batch: &RecordBatch) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut w = StreamWriter::try_new(&mut out, &batch.schema()).map_err(invalid)?;
    w.write(batch).map_err(invalid)?;
    w.finish().map_err(invalid)?;
    drop(w);
    Ok(out)
}

/// The one record batch an IPC payload holds.
fn from_ipc(bytes: &[u8]) -> Result<RecordBatch> {
    let mut reader = StreamReader::try_new(bytes, None).map_err(invalid)?;
    let batch = reader.next().ok_or_else(|| invalid("an empty Arrow payload"))?.map_err(invalid)?;
    if reader.next().is_some() {
        return Err(invalid("an Arrow payload with more than one batch"));
    }
    Ok(batch)
}

fn column<'a, T: 'static>(b: &'a RecordBatch, name: &str) -> Result<&'a T> {
    b.column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<T>())
        .ok_or_else(|| invalid(format!("payload has no usable {name} column")))
}

/// The time range and subjects of a query, as columns, and back.
fn filter_fields() -> Vec<Field> {
    let item = Arc::new(Field::new("item", DataType::Utf8, true));
    vec![
        Field::new("from_ns", DataType::UInt64, true),
        Field::new("to_ns", DataType::UInt64, true),
        Field::new("subjects", DataType::List(item), false),
        Field::new("labelled", DataType::Boolean, true),
    ]
}

fn filter_columns(f: &Filter) -> Vec<Arc<dyn Array>> {
    let mut subjects = ListBuilder::new(StringBuilder::new());
    for s in &f.subjects {
        subjects.values().append_value(s.hyphenated().to_string());
    }
    subjects.append(true);
    vec![
        Arc::new(UInt64Array::from(vec![f.during.map(|d| d.0)])),
        Arc::new(UInt64Array::from(vec![f.during.map(|d| d.1)])),
        Arc::new(subjects.finish()),
        Arc::new(BooleanArray::from(vec![f.labelled])),
    ]
}

fn read_filter(b: &RecordBatch) -> Result<Filter> {
    let (from, to): (&UInt64Array, &UInt64Array) = (column(b, "from_ns")?, column(b, "to_ns")?);
    let during = match (from.is_null(0), to.is_null(0)) {
        (true, true) => None,
        (false, false) => Some((from.value(0), to.value(0))),
        _ => return Err(invalid("a query's time range needs both ends or neither")),
    };
    let list: &ListArray = column(b, "subjects")?;
    let values = list.value(0);
    let names = values.as_any().downcast_ref::<StringArray>().ok_or_else(|| invalid("subjects aren't text"))?;
    let subjects = (0..names.len())
        .map(|i| Uuid::parse_str(names.value(i)).map_err(|e| invalid(format!("bad subject {}: {e}", names.value(i)))))
        .collect::<Result<Vec<_>>>()?;
    // A query from before the labelled filter has no such column: no filter.
    let labelled = b
        .column_by_name("labelled")
        .and_then(|c| c.as_any().downcast_ref::<BooleanArray>())
        .and_then(|l| (!l.is_null(0)).then(|| l.value(0)));
    Ok(Filter { during, subjects, labelled })
}

fn one_row(b: &RecordBatch) -> Result<()> {
    if b.num_rows() == 1 { Ok(()) } else { Err(invalid(format!("a query is one row, not {}", b.num_rows()))) }
}

/// A `vector.query` payload: the `k` nearest embeddings to `vector` in one embedding space.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorQuery {
    /// The embedding model.
    pub model: String,
    /// Its version.
    pub version: String,
    /// How many results.
    pub k: u32,
    /// The query vector (its length is the space's size).
    pub vector: Vec<f32>,
    /// Which rows to consider.
    pub filter: Filter,
}

/// Refuses a time range that's empty or backwards, before it's sent rather than by the worker.
fn check_during(f: &Filter) -> Result<()> {
    match f.during {
        Some((a, b)) if b <= a => Err(invalid(format!("an empty or backwards time range: [{a}, {b})"))),
        _ => Ok(()),
    }
}

impl VectorQuery {
    /// The payload's bytes.
    ///
    /// # Errors
    ///
    /// [`Error`] for an empty vector, a value that isn't finite, an empty or backwards time range, or a batch Arrow
    /// refuses.
    pub fn encode(&self) -> Result<Vec<u8>> {
        if self.vector.is_empty() || self.vector.iter().any(|x| !x.is_finite()) {
            return Err(invalid("a query vector needs at least one value, every one finite"));
        }
        check_during(&self.filter)?;
        let item = Arc::new(Field::new("item", DataType::Float32, true));
        let mut fields = vec![
            Field::new("model", DataType::Utf8, false),
            Field::new("version", DataType::Utf8, false),
            Field::new("k", DataType::UInt32, false),
            Field::new("vector", DataType::List(item), false),
        ];
        fields.extend(filter_fields());
        let mut vector = ListBuilder::new(Float32Builder::new());
        vector.values().append_slice(&self.vector);
        vector.append(true);
        let mut columns: Vec<Arc<dyn Array>> = vec![
            Arc::new(StringArray::from(vec![self.model.as_str()])),
            Arc::new(StringArray::from(vec![self.version.as_str()])),
            Arc::new(UInt32Array::from(vec![self.k])),
            Arc::new(vector.finish()),
        ];
        columns.extend(filter_columns(&self.filter));
        to_ipc(&RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).map_err(invalid)?)
    }

    /// Reads a `vector.query` payload.
    ///
    /// # Errors
    ///
    /// [`Error`] for bytes that aren't one query row with every column.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let b = from_ipc(bytes)?;
        one_row(&b)?;
        let (model, version): (&StringArray, &StringArray) = (column(&b, "model")?, column(&b, "version")?);
        let k: &UInt32Array = column(&b, "k")?;
        let list: &ListArray = column(&b, "vector")?;
        let values = list.value(0);
        let floats = values.as_any().downcast_ref::<Float32Array>().ok_or_else(|| invalid("vector isn't f32"))?;
        Ok(Self {
            model: model.value(0).to_owned(),
            version: version.value(0).to_owned(),
            k: k.value(0),
            vector: floats.values().to_vec(),
            filter: read_filter(&b)?,
        })
    }
}

/// A `text.query` payload: the `k` interpretation texts most relevant to `text`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextQuery {
    /// The search text.
    pub text: String,
    /// How many results.
    pub k: u32,
    /// Which rows to consider.
    pub filter: Filter,
}

impl TextQuery {
    /// The payload's bytes.
    ///
    /// # Errors
    ///
    /// [`Error`] if Arrow refuses the batch.
    pub fn encode(&self) -> Result<Vec<u8>> {
        check_during(&self.filter)?;
        let mut fields = vec![Field::new("text", DataType::Utf8, false), Field::new("k", DataType::UInt32, false)];
        fields.extend(filter_fields());
        let mut columns: Vec<Arc<dyn Array>> =
            vec![Arc::new(StringArray::from(vec![self.text.as_str()])), Arc::new(UInt32Array::from(vec![self.k]))];
        columns.extend(filter_columns(&self.filter));
        to_ipc(&RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).map_err(invalid)?)
    }

    /// Reads a `text.query` payload.
    ///
    /// # Errors
    ///
    /// [`Error`] for bytes that aren't one query row with every column.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let b = from_ipc(bytes)?;
        one_row(&b)?;
        let text: &StringArray = column(&b, "text")?;
        let k: &UInt32Array = column(&b, "k")?;
        Ok(Self { text: text.value(0).to_owned(), k: k.value(0), filter: read_filter(&b)? })
    }
}

/// A search result payload (`vector.search` or `text.search`): one row per hit, best first.
///
/// # Errors
///
/// [`Error`] if Arrow refuses the batch.
pub fn encode_hits(hits: &[Hit]) -> Result<Vec<u8>> {
    // Ids as 16 bytes, not strings, so a full result (MAX_K hits) stays an inline payload.
    let id = || DataType::FixedSizeBinary(16);
    let schema = Schema::new(vec![
        Field::new("payload_id", id(), false),
        Field::new("entity_id", id(), true),
        Field::new("parent_id", id(), true),
        Field::new("start_ns", DataType::UInt64, false),
        Field::new("end_ns", DataType::UInt64, false),
        Field::new("score", DataType::Float32, false),
    ]);
    let ids = |f: &dyn Fn(&Hit) -> Option<Uuid>| {
        FixedSizeBinaryArray::try_from_sparse_iter_with_size(hits.iter().map(|h| f(h).map(Uuid::into_bytes)), 16)
            .map_err(invalid)
    };
    let columns: Vec<Arc<dyn Array>> = vec![
        Arc::new(ids(&|h| Some(h.key.payload_id))?),
        Arc::new(ids(&|h| h.key.entity_id)?),
        Arc::new(ids(&|h| h.key.parent_id)?),
        Arc::new(UInt64Array::from_iter_values(hits.iter().map(|h| h.key.start_ns))),
        Arc::new(UInt64Array::from_iter_values(hits.iter().map(|h| h.key.end_ns))),
        Arc::new(Float32Array::from_iter_values(hits.iter().map(|h| h.score))),
    ];
    to_ipc(&RecordBatch::try_new(Arc::new(schema), columns).map_err(invalid)?)
}

/// Reads a search result payload.
///
/// # Errors
///
/// [`Error`] for bytes that aren't a result batch.
pub fn decode_hits(bytes: &[u8]) -> Result<Vec<Hit>> {
    let b = from_ipc(bytes)?;
    let (pids, eids, parents): (&FixedSizeBinaryArray, &FixedSizeBinaryArray, &FixedSizeBinaryArray) =
        (column(&b, "payload_id")?, column(&b, "entity_id")?, column(&b, "parent_id")?);
    let (starts, ends): (&UInt64Array, &UInt64Array) = (column(&b, "start_ns")?, column(&b, "end_ns")?);
    let scores: &Float32Array = column(&b, "score")?;
    let id = |a: &FixedSizeBinaryArray, i: usize| -> Result<Option<Uuid>> {
        if a.is_null(i) {
            return Ok(None);
        }
        Uuid::from_slice(a.value(i)).map(Some).map_err(|e| invalid(format!("bad id: {e}")))
    };
    (0..b.num_rows())
        .map(|i| {
            Ok(Hit {
                key: Key {
                    payload_id: id(pids, i)?.ok_or_else(|| invalid("a hit without a payload id"))?,
                    entity_id: id(eids, i)?,
                    parent_id: id(parents, i)?,
                    start_ns: starts.value(i),
                    end_ns: ends.value(i),
                },
                score: scores.value(i),
            })
        })
        .collect()
}

/// An embedding payload's vector: little-endian f32, every value finite.
///
/// # Errors
///
/// [`Error`] for an empty payload, a length that isn't a whole number of floats, or a value that isn't
/// finite.
pub fn embedding_vector(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
        return Err(invalid(format!("an embedding of {} bytes isn't whole f32s", bytes.len())));
    }
    let v: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    if v.iter().any(|x| !x.is_finite()) {
        return Err(invalid("an embedding value isn't finite"));
    }
    Ok(v)
}
