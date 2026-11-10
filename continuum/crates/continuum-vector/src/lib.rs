//! LanceDB storage for Continuum's vector worker (docs/CONTINUUM.md decisions 4 and 5).
//!
//! One table per embedding space (a model and its version), and one `texts` table with a full-text index over
//! interpretation text. Every row is keyed by the payload id it came from, with an optional entity id (the person or
//! thing it is about) and the time it covers in UTC nanoseconds. Rows never carry LanceDB row ids out: `_rowid`
//! changes when a table is compacted, so purges delete by a predicate on payload id or entity id.
//!
//! Ids are stored as canonical UUID strings, the form the Python SDK passes (the wire and the catalog keep their 16
//! bytes; decision 8): a string column keeps the delete predicates plain. Every id reaches a predicate through
//! [`Uuid`], whose hyphenated form is only hex digits and hyphens, so a predicate can't be given anything else.
//!
//! Forgetting is complete: LanceDB's delete only marks rows, the bytes stay in the data files, in older table
//! versions, and in the full-text index until those are rewritten, and the delete's own version records its
//! predicate, which names the purged ids. So [`VectorStore::purge_payloads`] and [`VectorStore::purge_entity`] delete,
//! then, in every table whether or not anything matched (so a purge retried after a crash still finishes), compact,
//! rebuild the full-text index, commit a fresh version that names nothing (a restore of the current one), and prune
//! every older version, before they return.
//!
//! Writes (inserts, purges, and compaction) take one lock, and reads share it: a purge's prune deletes every file no
//! retained version references, so an insert whose data file isn't committed yet would lose it, and a read could
//! land on a version being pruned.

pub mod jobs;
pub mod segments;
pub mod service;
pub mod worker;

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use futures::TryStreamExt;
use lance_index::scalar::FullTextSearchQuery;
use lancedb::arrow::arrow_array::{Array, FixedSizeListArray, Float32Array, RecordBatch, StringArray, UInt64Array};
use lancedb::arrow::arrow_schema::{DataType, Field, Schema, SchemaRef};
use lancedb::connection::Connection;
use lancedb::index::Index;
use lancedb::index::scalar::FtsIndexBuilder;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::table::{CompactionOptions, OptimizeAction};
use lancedb::{DistanceType, Table};
use uuid::Uuid;

/// The table that holds interpretation text for full-text search.
pub const TEXTS: &str = "texts";
/// The prefix of every embedding space's table name.
pub const EMBEDDINGS_PREFIX: &str = "emb__";
/// Ids per `IN (...)` list in a delete predicate.
const IDS_PER_PREDICATE: usize = 500;

/// What can go wrong.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// LanceDB refused or failed.
    #[error("lancedb: {0}")]
    Lance(#[from] lancedb::Error),
    /// An embedding space's name, a row, or a query is unusable.
    #[error("{0}")]
    Invalid(String),
}

/// This crate's result.
pub type Result<T> = std::result::Result<T, Error>;

/// An embedding space: vectors from one model and version, which only compare with each other.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmbeddingSpace {
    /// The model's name, like `arcface` (letters, digits, `_`, `-`, and `.`, with no `__`).
    pub model: String,
    /// The model's version, like `r100` (the same characters).
    pub version: String,
    /// How many floats each vector has.
    pub dim: usize,
}

impl EmbeddingSpace {
    /// The space's table name. Refuses a name that would need escaping, rather than changing it (two names could
    /// otherwise land on one table).
    pub fn table_name(&self) -> Result<String> {
        let (model, version) = (table_part(&self.model), table_part(&self.version));
        if model.is_empty() || version.is_empty() || model.contains("__") || version.contains("__") {
            return Err(Error::Invalid(format!("unusable embedding space name: {:?} {:?}", self.model, self.version)));
        }
        if self.dim == 0 || i32::try_from(self.dim).is_err() {
            return Err(Error::Invalid(format!("unusable vector size: {}", self.dim)));
        }
        Ok(format!("{EMBEDDINGS_PREFIX}{model}__{version}"))
    }

    fn dim_i32(&self) -> i32 {
        // table_name() refuses a dim that doesn't fit, and every caller checks it first.
        i32::try_from(self.dim).unwrap_or(i32::MAX)
    }
}

/// A model or version name as a table name's part: ASCII letters, digits, `_`, and `.` as they are, and every other
/// byte as `-` and two hex digits (`-` itself too), so any name works (an Ollama model is `qwen3-embedding:4b`, or
/// `user/model:tag`) and two names never share a table. Names of only those characters (`arcface`, `buffalo_l`,
/// `campp`) keep their tables.
fn table_part(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for b in name.bytes() {
        if b.is_ascii_alphanumeric() || b == b'_' || b == b'.' {
            out.push(char::from(b));
        } else {
            out.push_str(&format!("-{b:02x}"));
        }
    }
    out
}

pub use continuum_search::{Filter, Hit, Key};

/// One embedding.
#[derive(Clone, Debug, PartialEq)]
pub struct EmbeddingRow {
    /// Its payload, entity, and time.
    pub key: Key,
    /// The vector, `dim` floats long.
    pub vector: Vec<f32>,
}

/// One interpretation's text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextRow {
    /// Its payload, entity, and time.
    pub key: Key,
    /// The text to search.
    pub text: String,
}

/// What this process has purged: an insert of one of these payloads, or of a row about one of these entities, is
/// dropped. A purge can reach the worker while an embedding it names is still waiting to be filed (the intake runs
/// behind the subscription), and filing it afterwards would bring back what was forgotten. In memory is enough: after
/// a restart the catalog no longer has the payload, so nothing replays it.
#[derive(Default)]
struct Forgotten {
    payloads: HashSet<Uuid>,
    entities: HashSet<Uuid>,
    parents: HashSet<Uuid>,
}

impl Forgotten {
    fn keeps(&self, k: &Key) -> bool {
        !self.payloads.contains(&k.payload_id)
            && k.entity_id.is_none_or(|e| !self.entities.contains(&e))
            && k.parent_id.is_none_or(|p| !self.parents.contains(&p))
    }
}

/// The vector worker's tables in one directory.
pub struct VectorStore {
    db: Connection,
    /// Writes hold it exclusively and reads share it; it also holds what's been purged, so a purge and an insert of a
    /// payload it names can't interleave.
    lock: tokio::sync::RwLock<Forgotten>,
}

fn key_fields() -> Vec<Field> {
    vec![
        Field::new("payload_id", DataType::Utf8, false),
        Field::new("entity_id", DataType::Utf8, true),
        Field::new("parent_id", DataType::Utf8, true),
        Field::new("start_ns", DataType::UInt64, false),
        Field::new("end_ns", DataType::UInt64, false),
    ]
}

fn item() -> Arc<Field> {
    Arc::new(Field::new("item", DataType::Float32, true))
}

fn embedding_schema(space: &EmbeddingSpace) -> SchemaRef {
    let mut fields = key_fields();
    fields.push(Field::new("vector", DataType::FixedSizeList(item(), space.dim_i32()), false));
    Arc::new(Schema::new(fields))
}

fn text_schema() -> SchemaRef {
    let mut fields = key_fields();
    fields.push(Field::new("text", DataType::Utf8, false));
    Arc::new(Schema::new(fields))
}

fn check_key(k: &Key) -> Result<()> {
    if k.end_ns < k.start_ns {
        return Err(Error::Invalid(format!("row for {} ends before it starts", k.payload_id)));
    }
    Ok(())
}

fn key_columns(keys: &[Key]) -> Vec<Arc<dyn Array>> {
    vec![
        Arc::new(StringArray::from_iter_values(keys.iter().map(|k| k.payload_id.hyphenated().to_string()))),
        Arc::new(StringArray::from_iter(keys.iter().map(|k| k.entity_id.map(|e| e.hyphenated().to_string())))),
        Arc::new(StringArray::from_iter(keys.iter().map(|k| k.parent_id.map(|e| e.hyphenated().to_string())))),
        Arc::new(UInt64Array::from_iter_values(keys.iter().map(|k| k.start_ns))),
        Arc::new(UInt64Array::from_iter_values(keys.iter().map(|k| k.end_ns))),
    ]
}

fn quoted(ids: &[Uuid]) -> String {
    ids.iter().map(|u| format!("'{}'", u.hyphenated())).collect::<Vec<_>>().join(", ")
}

/// The SQL predicate for a filter, or None for no filter. A row covers `[start_ns, max(end_ns, start_ns + 1))`, so
/// it overlaps `[a, b)` when it starts before `b` and covers past `a`.
fn predicate(f: &Filter) -> Option<String> {
    let mut parts = Vec::new();
    if let Some((a, b)) = f.during {
        parts.push(format!("(start_ns < {b} AND (end_ns > {a} OR start_ns >= {a}))"));
    }
    if !f.subjects.is_empty() {
        parts.push(format!("entity_id IN ({})", quoted(&f.subjects)));
    }
    match f.labelled {
        Some(true) => parts.push("entity_id IS NOT NULL".into()),
        Some(false) => parts.push("entity_id IS NULL".into()),
        None => {}
    }
    (!parts.is_empty()).then(|| parts.join(" AND "))
}

fn check_filter(f: &Filter) -> Result<()> {
    match f.during {
        Some((a, b)) if b <= a => Err(Error::Invalid(format!("an empty or backwards time range: [{a}, {b})"))),
        _ => Ok(()),
    }
}

fn column<'a, T: 'static>(b: &'a RecordBatch, name: &str) -> Result<&'a T> {
    b.column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<T>())
        .ok_or_else(|| Error::Invalid(format!("search result has no usable {name} column")))
}

fn parse_id(s: &str) -> Result<Uuid> {
    Uuid::parse_str(s).map_err(|e| Error::Invalid(format!("bad stored id {s}: {e}")))
}

fn hits(batches: &[RecordBatch], score_column: &str) -> Result<Vec<Hit>> {
    let mut out = Vec::new();
    for b in batches {
        let pids: &StringArray = column(b, "payload_id")?;
        let eids: &StringArray = column(b, "entity_id")?;
        let parents: &StringArray = column(b, "parent_id")?;
        let starts: &UInt64Array = column(b, "start_ns")?;
        let ends: &UInt64Array = column(b, "end_ns")?;
        let scores: &Float32Array = column(b, score_column)?;
        for i in 0..b.num_rows() {
            out.push(Hit {
                key: Key {
                    payload_id: parse_id(pids.value(i))?,
                    entity_id: if eids.is_null(i) { None } else { Some(parse_id(eids.value(i))?) },
                    parent_id: if parents.is_null(i) { None } else { Some(parse_id(parents.value(i))?) },
                    start_ns: starts.value(i),
                    end_ns: ends.value(i),
                },
                score: scores.value(i),
            });
        }
    }
    Ok(out)
}

impl VectorStore {
    /// Open (or create) the tables in `dir`.
    pub async fn open(dir: &Path) -> Result<Self> {
        let uri = dir.to_str().ok_or_else(|| Error::Invalid(format!("not a UTF-8 path: {}", dir.display())))?;
        Ok(Self { db: lancedb::connect(uri).execute().await?, lock: tokio::sync::RwLock::new(Forgotten::default()) })
    }

    async fn table(&self, name: &str) -> Result<Option<Table>> {
        if self.db.table_names().execute().await?.iter().any(|n| n == name) {
            Ok(Some(self.db.open_table(name).execute().await?))
        } else {
            Ok(None)
        }
    }

    async fn table_or_create(&self, name: &str, schema: SchemaRef) -> Result<Table> {
        match self.table(name).await? {
            Some(t) => {
                if t.schema().await? != schema {
                    return Err(Error::Invalid(format!("table {name} has another schema (a different vector size?)")));
                }
                Ok(t)
            }
            None => Ok(self.db.create_empty_table(name, schema).execute().await?),
        }
    }

    /// Add embeddings to their space's table, creating it on first use.
    pub async fn insert_embeddings(&self, space: &EmbeddingSpace, rows: &[EmbeddingRow]) -> Result<()> {
        let name = space.table_name()?;
        if rows.is_empty() {
            return Ok(());
        }
        for r in rows {
            check_key(&r.key)?;
            if r.vector.len() != space.dim {
                return Err(Error::Invalid(format!("vector of {} floats in a space of {}", r.vector.len(), space.dim)));
            }
        }
        let schema = embedding_schema(space);
        let forgotten = self.lock.write().await;
        let rows: Vec<&EmbeddingRow> = rows.iter().filter(|r| forgotten.keeps(&r.key)).collect();
        if rows.is_empty() {
            return Ok(());
        }
        let table = self.table_or_create(&name, schema.clone()).await?;
        let mut columns = key_columns(&rows.iter().map(|r| r.key).collect::<Vec<_>>());
        let values = Float32Array::from_iter_values(rows.iter().flat_map(|r| r.vector.iter().copied()));
        let vectors = FixedSizeListArray::try_new(item(), space.dim_i32(), Arc::new(values), None)
            .map_err(|e| Error::Invalid(e.to_string()))?;
        columns.push(Arc::new(vectors));
        let batch = RecordBatch::try_new(schema, columns).map_err(|e| Error::Invalid(e.to_string()))?;
        table.add(batch).execute().await?;
        Ok(())
    }

    /// The `k` nearest embeddings to `query` in a space, by cosine distance, among the rows the filter allows.
    pub async fn search(&self, space: &EmbeddingSpace, query: &[f32], k: usize, filter: &Filter) -> Result<Vec<Hit>> {
        let name = space.table_name()?;
        if query.len() != space.dim {
            return Err(Error::Invalid(format!("query of {} floats in a space of {}", query.len(), space.dim)));
        }
        check_filter(filter)?;
        let _read = self.lock.read().await;
        let Some(table) = self.table(&name).await? else {
            return Ok(Vec::new());
        };
        let mut q = table.query().nearest_to(query)?.distance_type(DistanceType::Cosine).limit(k);
        if let Some(p) = predicate(filter) {
            q = q.only_if(p);
        }
        let batches: Vec<RecordBatch> = q.execute().await?.try_collect().await?;
        hits(&batches, "_distance")
    }

    /// Add interpretation text, creating the table and its full-text index on first use. Later rows are searched
    /// before they're indexed (LanceDB searches unindexed fragments directly); `compact` folds them into the index.
    pub async fn insert_texts(&self, rows: &[TextRow]) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        for r in rows {
            check_key(&r.key)?;
        }
        let schema = text_schema();
        let forgotten = self.lock.write().await;
        let rows: Vec<&TextRow> = rows.iter().filter(|r| forgotten.keeps(&r.key)).collect();
        if rows.is_empty() {
            return Ok(());
        }
        let table = self.table_or_create(TEXTS, schema.clone()).await?;
        let mut columns = key_columns(&rows.iter().map(|r| r.key).collect::<Vec<_>>());
        columns.push(Arc::new(StringArray::from_iter_values(rows.iter().map(|r| r.text.as_str()))));
        let batch = RecordBatch::try_new(schema, columns).map_err(|e| Error::Invalid(e.to_string()))?;
        table.add(batch).execute().await?;
        if table.list_indices().await?.is_empty() {
            Self::index_texts(&table).await?;
        }
        Ok(())
    }

    async fn index_texts(table: &Table) -> Result<()> {
        table.create_index(&["text"], Index::FTS(FtsIndexBuilder::default())).replace(true).execute().await?;
        Ok(())
    }

    /// The `k` most relevant texts for `query` (full-text search), among the rows the filter allows.
    pub async fn text_search(&self, query: &str, k: usize, filter: &Filter) -> Result<Vec<Hit>> {
        check_filter(filter)?;
        let _read = self.lock.read().await;
        let Some(table) = self.table(TEXTS).await? else {
            return Ok(Vec::new());
        };
        let mut q = table.query().full_text_search(FullTextSearchQuery::new(query.to_owned())).limit(k);
        if let Some(p) = predicate(filter) {
            q = q.only_if(p);
        }
        let batches: Vec<RecordBatch> = q.execute().await?.try_collect().await?;
        hits(&batches, "_score")
    }

    async fn tables(&self) -> Result<Vec<Table>> {
        let mut out = Vec::new();
        for name in self.db.table_names().execute().await? {
            if name == TEXTS || name.starts_with(EMBEDDINGS_PREFIX) {
                out.push(self.db.open_table(name).execute().await?);
            }
        }
        Ok(out)
    }

    /// Deletes the rows `predicates` match, under the write lock the caller holds (`_held`).
    async fn purge_where(&self, _held: &Forgotten, predicates: &[String]) -> Result<u64> {
        let mut removed = 0;
        for table in self.tables().await? {
            for p in predicates {
                removed += table.delete(p.as_str()).await?.num_deleted_rows;
            }
            // Always, not only when rows went: a retry after a crash between the delete and here deletes nothing.
            Self::rewrite(&table).await?;
        }
        Ok(removed)
    }

    /// Make deleted rows unrecoverable: rewrite the data files without them, rebuild the text index, commit a version
    /// that names nothing (a delete's own version records its predicate, and pruning never removes the newest), and
    /// drop every older version of the table.
    async fn rewrite(table: &Table) -> Result<()> {
        let options = CompactionOptions { materialize_deletions_threshold: 0.0, ..Default::default() };
        table.optimize(OptimizeAction::Compact { options, remap_options: None }).await?;
        if table.name() == TEXTS && table.count_rows(None).await? > 0 {
            Self::index_texts(table).await?;
        }
        table.checkout(table.version().await?).await?;
        table.restore().await?;
        Self::prune(table).await
    }

    /// Drop every version but the newest, and every file only they referenced. Only under the write lock: it also
    /// deletes a data file an uncommitted insert has written.
    async fn prune(table: &Table) -> Result<()> {
        table
            .optimize(OptimizeAction::Prune {
                older_than: Some(chrono::TimeDelta::zero()),
                delete_unverified: Some(true),
                error_if_tagged_old_versions: Some(false),
            })
            .await?;
        Ok(())
    }

    /// Remove every row made from these payloads, in every table, completely. Returns how many rows went.
    pub async fn purge_payloads(&self, ids: &[Uuid]) -> Result<u64> {
        let predicates: Vec<String> =
            ids.chunks(IDS_PER_PREDICATE).map(|c| format!("payload_id IN ({})", quoted(c))).collect();
        let mut forgotten = self.lock.write().await;
        forgotten.payloads.extend(ids.iter().copied());
        self.purge_where(&forgotten, &predicates).await
    }

    /// Remove every row about this entity, in every table, completely. Returns how many rows went.
    pub async fn purge_entity(&self, entity: Uuid) -> Result<u64> {
        let mut forgotten = self.lock.write().await;
        forgotten.entities.insert(entity);
        self.purge_where(&forgotten, &[format!("entity_id = '{}'", entity.hyphenated())]).await
    }

    /// Remove every row whose parent is `parent` (a segment voice row's voiceprint), in every table, completely.
    /// Returns how many rows went.
    pub async fn purge_parent(&self, parent: Uuid) -> Result<u64> {
        let mut forgotten = self.lock.write().await;
        forgotten.parents.insert(parent);
        self.purge_where(&forgotten, &[format!("parent_id = '{}'", parent.hyphenated())]).await
    }

    /// The payload ids of a space's rows that start in `[t0, t1)`.
    pub async fn payloads_between(&self, space: &EmbeddingSpace, t0: u64, t1: u64) -> Result<Vec<Uuid>> {
        let name = space.table_name()?;
        let _read = self.lock.read().await;
        let Some(table) = self.table(&name).await? else {
            return Ok(Vec::new());
        };
        let q = table
            .query()
            .only_if(format!("start_ns >= {t0} AND start_ns < {t1}"))
            .select(lancedb::query::Select::columns(&["payload_id"]));
        let batches: Vec<RecordBatch> = q.execute().await?.try_collect().await?;
        let mut out = Vec::new();
        for b in &batches {
            for id in column::<StringArray>(b, "payload_id")?.iter().flatten() {
                out.push(parse_id(id)?);
            }
        }
        Ok(out)
    }

    /// Merge small files, bring the indexes up to date, and drop older versions, in every table (P3 work for the
    /// worker loop). The text index is rebuilt whole, so there is one, not a growing set of incremental deltas.
    pub async fn compact(&self) -> Result<()> {
        let _write = self.lock.write().await;
        for table in self.tables().await? {
            if table.name() == TEXTS {
                table
                    .optimize(OptimizeAction::Compact { options: CompactionOptions::default(), remap_options: None })
                    .await?;
                if table.count_rows(None).await? > 0 {
                    Self::index_texts(&table).await?;
                }
            } else {
                table.optimize(OptimizeAction::All).await?;
            }
            Self::prune(&table).await?;
        }
        Ok(())
    }

    /// How many rows in all tables came from this payload.
    pub async fn rows_for(&self, payload_id: Uuid) -> Result<usize> {
        let _read = self.lock.read().await;
        let mut n = 0;
        for table in self.tables().await? {
            n += table.count_rows(Some(format!("payload_id = '{}'", payload_id.hyphenated()))).await?;
        }
        Ok(n)
    }
}
