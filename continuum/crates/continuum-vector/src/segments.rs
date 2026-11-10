//! Embeddings followed from the segment store, one row per filed embedding under `paths.state/segments`:
//!
//! - `voice.embedding` (listen, ingest, and the dashboard's enroll): `id`, `ts_ns` and `end_ns`, `voiceprint` (the
//!   voiceprint's subject), `person` (the person it names, when it names one), `utterance` (its line, when there is
//!   one), and `vector` (192 float32s, ReDimNet2 B3). Keyed on the person, else the voiceprint, with the voiceprint as parent.
//! - `face.embedding` (see-actions): the same with `group` (the face group's subject) for `voiceprint`, and 512
//!   float32s (AdaFace IR50 WebFace4M RGB). Keyed on the person, else the group, with the group as parent.
//!
//! The entity is what a match names; the parent lets a forget of a voiceprint or a group find every row of it, whoever
//! it names.
//!
//! The intake keeps its place in `segments.<stream>.json` in the store's directory: the store's deletion revision it
//! has applied, and the next `row_seq` per slot (the store's camera). Filing comes before the place is saved, so a
//! crash between them files a batch again (a duplicate row of a payload already there, harmless to a nearest search).
//! Each forget covering the stream is applied once, after the rows before it were filed:
//!
//! - by an entity column (`person`, `voiceprint`, or `group`): every row of that entity, and by the parent column every
//!   row with that parent;
//! - by `id`: that payload's rows;
//! - by a range (a slot, such as an ingested item's, over some time): the rows filed for that time that the store no
//!   longer holds. A legacy row the broker's stream filed in that time goes too; the broker's purges keep covering
//!   the rest of those.
//!
//! A store reset (the revision going back, or a reset deletion) starts the place over and keeps what was filed: the
//! evidence is gone, what was learned from it stands.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Duration;

use arrow_array::{Array, FixedSizeBinaryArray, FixedSizeListArray, Float32Array, Int64Array, RecordBatch};
use continuum_segments::{Deletion, DeletionKind, Follower};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::jobs::space_of_shape;
use crate::{EmbeddingRow, EmbeddingSpace, Error, Key, Result, VectorStore};

/// An embedding stream in the segment store: the shape (model and version) its vectors are, which id columns name a
/// row's entity (the first one set), and which names its parent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SegmentStream {
    /// The stream's name.
    pub name: String,
    /// Its vectors' shape name (`embedding/<model>/<version>`), which names their space.
    pub shape: String,
    /// The id columns a row's entity is, in order: the first that's set (a person, else a voiceprint or face group).
    pub entity: Vec<String>,
    /// The id column that's a row's parent, so forgetting by it finds every row of it whoever it names.
    pub parent: Option<String>,
}

impl SegmentStream {
    fn new(name: &str, shape: &str, entity: &[&str], parent: Option<&str>) -> Self {
        Self {
            name: name.into(),
            shape: shape.into(),
            entity: entity.iter().map(|c| (*c).to_owned()).collect(),
            parent: parent.map(str::to_owned),
        }
    }
}

/// The segment streams followed when none are named: listen's voice embeddings (keyed on the person, else the
/// voiceprint) and see-actions' face embeddings (keyed on the person, else the face group).
#[must_use]
pub fn default_streams() -> Vec<SegmentStream> {
    vec![
        SegmentStream::new(
            "voice.embedding",
            "embedding/redimnet2-b3/vox2-lm-v1",
            &["person", "voiceprint"],
            Some("voiceprint"),
        ),
        SegmentStream::new(
            "face.embedding",
            "embedding/adaface/ir50_webface4m_rgb",
            &["person", "group"],
            Some("group"),
        ),
    ]
}

/// Where an intake is.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Place {
    /// The newest deletion it has applied.
    pub revision: u64,
    /// The next `row_seq` per slot.
    pub cursor: BTreeMap<String, u64>,
}

fn place_path(store: &Path, stream: &str) -> PathBuf {
    store.join(format!("segments.{stream}.json"))
}

/// The saved place, or None when there's none (never followed).
///
/// # Errors
///
/// [`Error::Invalid`] for a file that isn't a place, or one that can't be read.
pub fn load_place(store: &Path, stream: &str) -> Result<Option<Place>> {
    let path = place_path(store, stream);
    match std::fs::read_to_string(&path) {
        Ok(text) => {
            serde_json::from_str(&text).map(Some).map_err(|e| Error::Invalid(format!("{}: {e}", path.display())))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(Error::Invalid(format!("{}: {e}", path.display()))),
    }
}

/// Saves the place (written aside and synced, then renamed over the old one, so a power cut leaves the old place or
/// the new one, never an empty file that [`load_place`] refuses).
///
/// # Errors
///
/// [`Error::Invalid`] for a write that failed.
pub fn save_place(store: &Path, stream: &str, place: &Place) -> Result<()> {
    use std::io::Write as _;
    let path = place_path(store, stream);
    let aside = path.with_extension("json.tmp");
    let text = serde_json::to_string(place).map_err(|e| Error::Invalid(e.to_string()))?;
    std::fs::File::create(&aside)
        .and_then(|mut f| f.write_all(text.as_bytes()).and_then(|()| f.sync_data()))
        .and_then(|()| std::fs::rename(&aside, &path))
        .map_err(|e| Error::Invalid(format!("{}: {e}", path.display())))
}

fn segments(e: continuum_segments::Error) -> Error {
    Error::Invalid(format!("segment store: {e}"))
}

fn ids(batch: &RecordBatch, name: &str) -> Result<Vec<Option<Uuid>>> {
    let col = batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<FixedSizeBinaryArray>())
        .ok_or_else(|| Error::Invalid(format!("no id column {name}")))?;
    Ok((0..col.len()).map(|i| (!col.is_null(i)).then(|| Uuid::from_slice(col.value(i)).ok()).flatten()).collect())
}

fn times(batch: &RecordBatch, name: &str) -> Result<Vec<u64>> {
    let col = batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
        .ok_or_else(|| Error::Invalid(format!("no time column {name}")))?;
    Ok(col.values().iter().map(|&t| u64::try_from(t).unwrap_or(0)).collect())
}

/// A batch's rows as embedding rows of `space`, keyed as `stream` says. A row without an id or a vector of the
/// space's size is left out and counted in the second value.
///
/// # Errors
///
/// [`Error::Invalid`] for a batch without the columns.
pub fn rows_of(
    batch: &RecordBatch,
    space: &EmbeddingSpace,
    stream: &SegmentStream,
) -> Result<(Vec<EmbeddingRow>, usize)> {
    let payload = ids(batch, "id")?;
    let entities = stream.entity.iter().map(|c| ids(batch, c)).collect::<Result<Vec<_>>>()?;
    let parents = stream.parent.as_deref().map(|c| ids(batch, c)).transpose()?;
    let (start, end) = (times(batch, "ts_ns")?, times(batch, "end_ns")?);
    let vectors = batch
        .column_by_name("vector")
        .and_then(|c| c.as_any().downcast_ref::<FixedSizeListArray>())
        .ok_or_else(|| Error::Invalid("no vector column".into()))?;
    let mut out = Vec::with_capacity(batch.num_rows());
    let mut skipped = 0;
    for i in 0..batch.num_rows() {
        let values = vectors.value(i);
        let floats = values.as_any().downcast_ref::<Float32Array>();
        match (payload[i], floats) {
            (Some(payload_id), Some(f)) if f.len() == space.dim && f.null_count() == 0 => {
                out.push(EmbeddingRow {
                    key: Key {
                        payload_id,
                        entity_id: entities.iter().find_map(|e| e[i]),
                        parent_id: parents.as_ref().and_then(|p| p[i]),
                        start_ns: start[i],
                        end_ns: end[i].max(start[i]),
                    },
                    vector: f.values().to_vec(),
                });
            }
            _ => skipped += 1,
        }
    }
    Ok((out, skipped))
}

/// Applies one forget to the index: see the module's list. Returns how many rows went.
///
/// # Errors
///
/// The store's or the segment store's errors.
pub async fn apply(
    store: &VectorStore,
    root: &Path,
    stream: &SegmentStream,
    space: &EmbeddingSpace,
    d: &Deletion,
) -> Result<u64> {
    if let (Some(column), Some(value)) = (d.column.as_deref(), d.value.as_deref()) {
        let Ok(id) = Uuid::parse_str(value) else {
            return Ok(0); // a str column's value: no index row is keyed on one
        };
        if column == "id" {
            return store.purge_payloads(&[id]).await;
        }
        let mut gone = 0;
        if stream.parent.as_deref() == Some(column) {
            gone += store.purge_parent(id).await?;
        }
        if stream.entity.iter().any(|c| c == column) {
            gone += store.purge_entity(id).await?;
        }
        return Ok(gone);
    }
    let stream = stream.name.as_str();
    // A range: what the index has for that time, less what the store still holds then (every slot: an index row
    // doesn't know its slot).
    let filed = store.payloads_between(space, d.t0, d.t1).await?;
    if filed.is_empty() {
        return Ok(0);
    }
    let (root, stream, t0, t1) = (root.to_path_buf(), stream.to_owned(), d.t0, d.t1);
    let read = tokio::task::spawn_blocking(move || continuum_segments::read(&root, &stream, None, t0, t1))
        .await
        .map_err(|e| Error::Invalid(e.to_string()))?
        .map_err(segments)?;
    let mut kept = HashSet::new();
    for (_slot, batch) in &read.batches {
        kept.extend(ids(batch, "id")?.into_iter().flatten());
    }
    let gone: Vec<Uuid> = filed.into_iter().filter(|id| !kept.contains(id)).collect();
    if gone.is_empty() { Ok(0) } else { store.purge_payloads(&gone).await }
}

/// Follows `stream` in the segment store at `root`, filing its rows into `store` (whose directory is `dir`) and
/// applying its forgets, until an error. It waits up to `wait` for rows each round.
///
/// # Errors
///
/// The store's or the segment store's errors; the caller restarts it.
pub async fn follow(
    store: &VectorStore,
    dir: PathBuf,
    root: PathBuf,
    stream: SegmentStream,
    wait: Duration,
) -> Result<()> {
    let probe = space_of_shape(&stream.shape, 1)?;
    let mut place = match load_place(&dir, &stream.name)? {
        Some(p) => p,
        // First run: everything the store holds is filed, and the deletions before now are already in what it holds.
        None => Place { revision: revision(&root).await?, cursor: BTreeMap::new() },
    };
    let mut follower = Follower::new(&root, &stream.name, None, place.cursor.clone());
    loop {
        let now = revision(&root).await?;
        if now < place.revision {
            // The store was replaced (a reset): start the place over.
            place = Place { revision: now, cursor: BTreeMap::new() };
            follower = Follower::new(&root, &stream.name, None, BTreeMap::new());
        } else if now > place.revision {
            let r = root.clone();
            let after = place.revision;
            let new = tokio::task::spawn_blocking(move || continuum_segments::deletions(&r, after))
                .await
                .map_err(|e| Error::Invalid(e.to_string()))?
                .map_err(segments)?;
            for d in new.iter().filter(|d| d.streams.is_empty() || d.streams.iter().any(|s| s == &stream.name)) {
                match d.kind {
                    DeletionKind::Forget => {
                        // A space's table is named by its model and version, so the probe's size doesn't matter here.
                        let gone = apply(store, &root, &stream, &probe, d).await?;
                        eprintln!("continuum-vector-worker: {}: forget {}: {gone} rows", stream.name, d.revision);
                    }
                    DeletionKind::Reset => {
                        place.cursor.clear();
                        follower = Follower::new(&root, &stream.name, None, BTreeMap::new());
                    }
                }
            }
            place.revision = now;
            save_place(&dir, &stream.name, &place)?;
        }
        let (back, got) = tokio::task::spawn_blocking(move || {
            let got = follower.wait(wait);
            (follower, got)
        })
        .await
        .map_err(|e| Error::Invalid(e.to_string()))?;
        follower = back;
        let got = got.map_err(segments)?;
        if got.is_empty() {
            continue;
        }
        for (slot, batch) in &got {
            let dim = vector_dim(batch).unwrap_or(probe.dim);
            let space = space_of_shape(&stream.shape, dim)?;
            let (rows, skipped) = rows_of(batch, &space, &stream)?;
            if skipped > 0 {
                eprintln!("continuum-vector-worker: {} {slot}: skipped {skipped} rows (no id or vector)", stream.name);
            }
            store.insert_embeddings(&space, &rows).await?;
        }
        place.cursor = follower.cursor().clone();
        save_place(&dir, &stream.name, &place)?;
    }
}

async fn revision(root: &Path) -> Result<u64> {
    let r = root.to_path_buf();
    tokio::task::spawn_blocking(move || continuum_segments::revision(&r))
        .await
        .map_err(|e| Error::Invalid(e.to_string()))?
        .map_err(segments)
}

fn vector_dim(batch: &RecordBatch) -> Option<usize> {
    let vectors = batch.column_by_name("vector")?.as_any().downcast_ref::<FixedSizeListArray>()?;
    usize::try_from(vectors.value_length()).ok()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, FixedSizeBinaryArray, FixedSizeListArray, Float32Array, Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema, SchemaRef};
    use continuum_segments::{DeletionKind, Writer, forget, forget_where};

    use super::*;

    const STREAM: &str = "voice.embedding";
    const T: i64 = 1_800_000_000_000_000_000;
    const DIM: i32 = 4;

    fn schema() -> SchemaRef {
        let item = Arc::new(Field::new("item", DataType::Float32, true));
        Arc::new(Schema::new(vec![
            Field::new("ts_ns", DataType::Int64, false),
            Field::new("end_ns", DataType::Int64, false),
            Field::new("id", DataType::FixedSizeBinary(16), false),
            Field::new("voiceprint", DataType::FixedSizeBinary(16), false),
            Field::new("person", DataType::FixedSizeBinary(16), true),
            Field::new("utterance", DataType::FixedSizeBinary(16), true),
            Field::new("vector", DataType::FixedSizeList(item, DIM), false),
        ]))
    }

    struct Row {
        at: i64,
        id: Uuid,
        print: Uuid,
        person: Option<Uuid>,
    }

    fn batch(rows: &[Row]) -> RecordBatch {
        let ids = |f: &dyn Fn(&Row) -> Option<Uuid>| -> ArrayRef {
            Arc::new(
                FixedSizeBinaryArray::try_from_sparse_iter_with_size(
                    rows.iter().map(|r| f(r).map(|u| *u.as_bytes())),
                    16,
                )
                .unwrap(),
            )
        };
        let values = Float32Array::from_iter_values(rows.iter().flat_map(|r| [r.at as f32, 1.0, 0.0, 0.0]));
        let item = Arc::new(Field::new("item", DataType::Float32, true));
        let vectors = FixedSizeListArray::try_new(item, DIM, Arc::new(values), None).unwrap();
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.at))),
                Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.at + 1_000_000_000))),
                ids(&|r| Some(r.id)),
                ids(&|r| Some(r.print)),
                ids(&|r| r.person),
                ids(&|_| None),
                Arc::new(vectors),
            ],
        )
        .unwrap()
    }

    fn write(root: &Path, slot: &str, rows: &[Row]) {
        let mut w = Writer::open(root, STREAM, slot, schema()).unwrap();
        w.append(&batch(rows)).unwrap();
        w.close().unwrap();
    }

    fn row(at: i64, print: Uuid, person: Option<Uuid>) -> Row {
        Row { at, id: Uuid::now_v7(), print, person }
    }

    async fn eventually(store: &VectorStore, id: Uuid, want: usize) {
        for _ in 0..200 {
            if store.rows_for(id).await.unwrap() == want {
                return;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        panic!("{id} never had {want} rows (has {})", store.rows_for(id).await.unwrap());
    }

    #[test]
    fn a_row_is_keyed_on_its_person_else_its_voiceprint_with_the_voiceprint_as_parent() {
        let (print, person) = (Uuid::now_v7(), Uuid::now_v7());
        let (named, unnamed) = (row(T, print, Some(person)), row(T + 5, print, None));
        let space = space_of_shape("embedding/redimnet2-b3/vox2-lm-v1", 4).unwrap();
        let voice = default_streams().remove(0);
        let (rows, skipped) = rows_of(&batch(&[named, unnamed]), &space, &voice).unwrap();
        assert_eq!(skipped, 0);
        assert_eq!(
            rows.iter().map(|r| (r.key.entity_id, r.key.parent_id)).collect::<Vec<_>>(),
            [(Some(person), Some(print)), (Some(print), Some(print))]
        );
        assert_eq!((rows[0].key.start_ns, rows[0].key.end_ns), (T as u64, T as u64 + 1_000_000_000));
        assert_eq!(rows[0].vector, [T as f32, 1.0, 0.0, 0.0]);
        let other = space_of_shape("embedding/redimnet2-b3/vox2-lm-v1", 8).unwrap();
        assert_eq!(
            rows_of(&batch(&[row(T, print, None)]), &other, &voice).unwrap().1,
            1,
            "a vector of another size is skipped"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn face_intake_is_searchable_in_adaface_space_without_leaking_into_arcface() {
        let dir = tempfile::tempdir().unwrap();
        let (root, vectors) = (dir.path().join("segments"), dir.path().join("vector"));
        std::fs::create_dir_all(&vectors).unwrap();
        let (group, person) = (Uuid::now_v7(), Uuid::now_v7());
        let face = row(T, group, Some(person));
        let id = face.id;
        let base = batch(&[face]);
        let mut fields: Vec<Field> = base.schema().fields().iter().map(|f| f.as_ref().clone()).collect();
        fields[3] = Field::new("group", DataType::FixedSizeBinary(16), false);
        let item = Arc::new(Field::new("item", DataType::Float32, true));
        fields[6] = Field::new("vector", DataType::FixedSizeList(Arc::clone(&item), 512), false);
        let mut query = vec![0.0_f32; 512];
        query[0] = 1.0;
        let mut columns = base.columns().to_vec();
        columns[6] = Arc::new(
            FixedSizeListArray::try_new(item, 512, Arc::new(Float32Array::from(query.clone())), None).unwrap(),
        );
        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(Arc::clone(&schema), columns).unwrap();
        let mut writer = Writer::open(&root, "face.embedding", "camera", schema).unwrap();
        writer.append(&batch).unwrap();
        writer.close().unwrap();

        let store = Arc::new(VectorStore::open(&vectors).await.unwrap());
        let stream = default_streams().into_iter().find(|s| s.name == "face.embedding").unwrap();
        let task_store = Arc::clone(&store);
        let intake =
            tokio::spawn(async move { follow(&task_store, vectors, root, stream, Duration::from_millis(50)).await });
        eventually(&store, id, 1).await;
        let current = space_of_shape("embedding/adaface/ir50_webface4m_rgb", 512).unwrap();
        let hits = store.search(&current, &query, 1, &crate::Filter::default()).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(
            (hits[0].key.payload_id, hits[0].key.entity_id, hits[0].key.parent_id),
            (id, Some(person), Some(group))
        );
        let old = space_of_shape("embedding/arcface/buffalo_l", 512).unwrap();
        assert!(store.search(&old, &query, 1, &crate::Filter::default()).await.unwrap().is_empty());
        intake.abort();
        let _ = intake.await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_intake_files_rows_applies_each_forget_and_resumes_where_it_was() {
        let dir = tempfile::tempdir().unwrap();
        let (root, vectors) = (dir.path().join("segments"), dir.path().join("vector"));
        std::fs::create_dir_all(&vectors).unwrap();
        let (p1, p2, p3, dana) = (Uuid::now_v7(), Uuid::now_v7(), Uuid::now_v7(), Uuid::now_v7());
        let hers = row(T, p1, Some(dana));
        let unnamed = row(T + 10, p2, None);
        let ingested = row(T - 3_600_000_000_000, p3, None); // an imported clip, from an hour before
        let kept = row(T + 20, p3, None);
        let (hers_id, unnamed_id, ingested_id, kept_id) = (hers.id, unnamed.id, ingested.id, kept.id);
        write(&root, "mic", &[hers, unnamed, kept]);
        write(&root, "ingest-7", &[ingested]);

        let store = Arc::new(VectorStore::open(&vectors).await.unwrap());
        let stream = default_streams().remove(0);
        let run = |store: Arc<VectorStore>| {
            let (vectors, root, stream) = (vectors.clone(), root.clone(), stream.clone());
            tokio::spawn(async move { follow(&store, vectors, root, stream, Duration::from_millis(50)).await })
        };
        let intake = run(Arc::clone(&store));
        for id in [hers_id, unnamed_id, ingested_id, kept_id] {
            eventually(&store, id, 1).await;
        }

        forget_where(&root, Some(&[STREAM]), None, "person", &dana.to_string(), DeletionKind::Forget).unwrap();
        eventually(&store, hers_id, 0).await;
        forget_where(&root, Some(&[STREAM]), None, "voiceprint", &p2.to_string(), DeletionKind::Forget).unwrap();
        eventually(&store, unnamed_id, 0).await;
        forget(&root, Some(&[STREAM]), Some(&["ingest-7"]), 0, u64::MAX, DeletionKind::Forget).unwrap();
        eventually(&store, ingested_id, 0).await;
        assert_eq!(store.rows_for(kept_id).await.unwrap(), 1, "a forget takes only its own rows");

        // A restart takes up where it was: nothing filed again, nothing forgotten back.
        intake.abort();
        let _ = intake.await;
        let later = row(T + 30, p3, None);
        let later_id = later.id;
        write(&root, "mic", &[later]);
        let intake = run(Arc::clone(&store));
        eventually(&store, later_id, 1).await;
        for (id, want) in [(kept_id, 1), (hers_id, 0), (unnamed_id, 0), (ingested_id, 0)] {
            assert_eq!(store.rows_for(id).await.unwrap(), want, "{id}");
        }
        intake.abort();
    }
}
