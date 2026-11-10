//! The vector store against a temporary directory: inserts, searches with filters, compaction, and purges that
//! leave nothing behind.

use std::path::Path;

use continuum_vector::{EmbeddingRow, EmbeddingSpace, Filter, Key, TextRow, VectorStore};
use uuid::Uuid;

fn space() -> EmbeddingSpace {
    EmbeddingSpace { model: "arcface".into(), version: "r100".into(), dim: 3 }
}

fn id(n: u128) -> Uuid {
    // A fixed UUIDv7 per n: the version and variant bits over n.
    Uuid::from_u128((n & !(0xF << 76) & !(0x3 << 62)) | (0x7 << 76) | (0x2 << 62))
}

fn key(payload: u128, entity: Option<u128>, start_ns: u64, end_ns: u64) -> Key {
    Key { payload_id: id(payload), entity_id: entity.map(id), parent_id: None, start_ns, end_ns }
}

fn emb(payload: u128, entity: Option<u128>, at: u64, vector: [f32; 3]) -> EmbeddingRow {
    EmbeddingRow { key: key(payload, entity, at, at), vector: vector.to_vec() }
}

async fn store(dir: &Path) -> VectorStore {
    VectorStore::open(dir).await.expect("open")
}

/// Every file under dir, read whole.
fn files(dir: &Path) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).expect("read_dir") {
        let path = entry.expect("entry").path();
        if path.is_dir() {
            out.extend(files(&path));
        } else {
            out.push(std::fs::read(&path).expect("read"));
        }
    }
    out
}

fn on_disk(dir: &Path, needle: &str) -> bool {
    files(dir).iter().any(|f| f.windows(needle.len()).any(|w| w == needle.as_bytes()))
}

async fn nearest(s: &VectorStore, query: [f32; 3], filter: &Filter) -> Vec<Uuid> {
    s.search(&space(), &query, 10, filter).await.expect("search").into_iter().map(|h| h.key.payload_id).collect()
}

#[tokio::test]
async fn search_finds_the_nearest_embedding_first() {
    let dir = tempfile::tempdir().unwrap();
    let s = store(dir.path()).await;
    s.insert_embeddings(&space(), &[emb(1, None, 10, [1.0, 0.0, 0.0]), emb(2, None, 20, [0.0, 1.0, 0.0])])
        .await
        .unwrap();
    assert_eq!(nearest(&s, [0.9, 0.1, 0.0], &Filter::default()).await, vec![id(1), id(2)]);
    assert_eq!(nearest(&s, [0.1, 0.9, 0.0], &Filter::default()).await, vec![id(2), id(1)]);
}

#[tokio::test]
async fn a_time_filter_is_half_open_and_an_instant_covers_its_nanosecond() {
    let dir = tempfile::tempdir().unwrap();
    let s = store(dir.path()).await;
    let rows = vec![
        emb(1, None, 99, [1.0, 0.0, 0.0]),  // an instant just before the window
        emb(2, None, 100, [1.0, 0.0, 0.0]), // an instant at its start: in
        emb(3, None, 199, [1.0, 0.0, 0.0]), // an instant at its last nanosecond: in
        emb(4, None, 200, [1.0, 0.0, 0.0]), // an instant at its end: out
        EmbeddingRow { key: key(5, None, 50, 100), vector: vec![1.0, 0.0, 0.0] }, // a span ending at its start: out
        EmbeddingRow { key: key(6, None, 50, 101), vector: vec![1.0, 0.0, 0.0] }, // a span reaching into it: in
    ];
    s.insert_embeddings(&space(), &rows).await.unwrap();
    let mut got =
        nearest(&s, [1.0, 0.0, 0.0], &Filter { during: Some((100, 200)), subjects: vec![], labelled: None }).await;
    got.sort();
    assert_eq!(got, vec![id(2), id(3), id(6)]);
}

#[tokio::test]
async fn a_subject_filter_keeps_rows_about_those_entities() {
    let dir = tempfile::tempdir().unwrap();
    let s = store(dir.path()).await;
    let rows = [
        emb(1, Some(100), 10, [1.0, 0.0, 0.0]),
        emb(2, Some(200), 10, [1.0, 0.0, 0.0]),
        emb(3, None, 10, [1.0, 0.0, 0.0]),
    ];
    s.insert_embeddings(&space(), &rows).await.unwrap();
    let only = Filter { during: None, subjects: vec![id(200)], labelled: None };
    assert_eq!(nearest(&s, [1.0, 0.0, 0.0], &only).await, vec![id(2)]);
}

#[tokio::test]
async fn rows_are_still_found_by_payload_id_after_compaction() {
    let dir = tempfile::tempdir().unwrap();
    let s = store(dir.path()).await;
    for n in 1..=5 {
        // One insert per row, so there are files to merge.
        s.insert_embeddings(&space(), &[emb(n, None, n as u64, [n as f32, 1.0, 0.0])]).await.unwrap();
    }
    s.insert_texts(&[TextRow { key: key(9, None, 1, 1), text: "the kettle boiled".into() }]).await.unwrap();
    s.compact().await.unwrap();
    for n in 1..=5 {
        assert_eq!(s.rows_for(id(n)).await.unwrap(), 1);
    }
    assert_eq!(nearest(&s, [3.0, 1.0, 0.0], &Filter::default()).await[0], id(3));
    let found = s.text_search("kettle", 10, &Filter::default()).await.unwrap();
    assert_eq!(found.iter().map(|h| h.key.payload_id).collect::<Vec<_>>(), vec![id(9)]);
}

#[tokio::test]
async fn text_search_finds_text_added_after_the_first_index_and_honors_filters() {
    let dir = tempfile::tempdir().unwrap();
    let s = store(dir.path()).await;
    s.insert_texts(&[TextRow { key: key(1, None, 10, 10), text: "walked the dog in the park".into() }]).await.unwrap();
    s.insert_texts(&[TextRow { key: key(2, None, 500, 500), text: "another walk with the dog".into() }]).await.unwrap();
    let all = s.text_search("dog", 10, &Filter::default()).await.unwrap();
    let mut ids: Vec<Uuid> = all.iter().map(|h| h.key.payload_id).collect();
    ids.sort();
    assert_eq!(ids, vec![id(1), id(2)]);
    let late = s
        .text_search("dog", 10, &Filter { during: Some((100, 1000)), subjects: vec![], labelled: None })
        .await
        .unwrap();
    assert_eq!(late.iter().map(|h| h.key.payload_id).collect::<Vec<_>>(), vec![id(2)]);
}

#[tokio::test]
async fn a_purge_by_payload_removes_only_those_rows_from_every_table() {
    let dir = tempfile::tempdir().unwrap();
    let s = store(dir.path()).await;
    s.insert_embeddings(&space(), &[emb(1, None, 1, [1.0, 0.0, 0.0]), emb(2, None, 2, [0.0, 1.0, 0.0])]).await.unwrap();
    s.insert_texts(&[TextRow { key: key(1, None, 1, 1), text: "derived from payload one".into() }]).await.unwrap();
    assert_eq!(s.purge_payloads(&[id(1)]).await.unwrap(), 2);
    assert_eq!(s.rows_for(id(1)).await.unwrap(), 0);
    assert_eq!(nearest(&s, [1.0, 0.0, 0.0], &Filter::default()).await, vec![id(2)]);
    assert!(s.text_search("payload", 10, &Filter::default()).await.unwrap().is_empty());
}

#[tokio::test]
async fn a_purge_by_entity_removes_its_rows_in_every_space_and_the_texts() {
    let dir = tempfile::tempdir().unwrap();
    let s = store(dir.path()).await;
    let voices = EmbeddingSpace { model: "campplus".into(), version: "1".into(), dim: 2 };
    s.insert_embeddings(&space(), &[emb(1, Some(7), 1, [1.0, 0.0, 0.0]), emb(2, Some(8), 1, [1.0, 0.0, 0.0])])
        .await
        .unwrap();
    s.insert_embeddings(&voices, &[EmbeddingRow { key: key(3, Some(7), 1, 1), vector: vec![1.0, 0.0] }]).await.unwrap();
    s.insert_texts(&[TextRow { key: key(4, Some(7), 1, 1), text: "about entity seven".into() }]).await.unwrap();
    assert_eq!(s.purge_entity(id(7)).await.unwrap(), 3);
    for gone in [1, 3, 4] {
        assert_eq!(s.rows_for(id(gone)).await.unwrap(), 0, "payload {gone}");
    }
    assert_eq!(s.rows_for(id(2)).await.unwrap(), 1);
}

#[tokio::test]
async fn a_purge_leaves_no_copy_in_any_file_or_older_version() {
    let dir = tempfile::tempdir().unwrap();
    let s = store(dir.path()).await;
    let marker = "zebracrossingmarker";
    let doomed = id(1);
    s.insert_texts(&[TextRow { key: key(1, None, 1, 1), text: format!("a note naming {marker}") }]).await.unwrap();
    s.insert_texts(&[TextRow { key: key(2, None, 2, 2), text: "a note that stays".into() }]).await.unwrap();
    s.insert_embeddings(&space(), &[emb(1, None, 1, [1.0, 0.0, 0.0]), emb(2, None, 2, [0.0, 1.0, 0.0])]).await.unwrap();
    // The marker and the id are on disk before (so the check below can't pass because of an encoding that hides
    // them), and nowhere after: not in a data file, an older version, or the text index.
    assert!(on_disk(dir.path(), marker));
    assert!(on_disk(dir.path(), &doomed.hyphenated().to_string()));
    s.purge_payloads(&[doomed]).await.unwrap();
    assert!(!on_disk(dir.path(), marker));
    assert!(!on_disk(dir.path(), &doomed.hyphenated().to_string()));
    assert_eq!(s.rows_for(id(2)).await.unwrap(), 2);
    assert_eq!(s.text_search("stays", 10, &Filter::default()).await.unwrap().len(), 1);
}

#[tokio::test]
async fn a_purge_of_nothing_removes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let s = store(dir.path()).await;
    // (Purging payload 5, not 1: an id already purged is never filed again.)
    assert_eq!(s.purge_payloads(&[id(5)]).await.unwrap(), 0);
    s.insert_embeddings(&space(), &[emb(1, None, 1, [1.0, 0.0, 0.0])]).await.unwrap();
    assert_eq!(s.purge_payloads(&[]).await.unwrap(), 0);
    assert_eq!(s.purge_entity(id(99)).await.unwrap(), 0);
    assert_eq!(s.rows_for(id(1)).await.unwrap(), 1);
}

#[tokio::test]
async fn an_ollama_named_space_files_and_answers_under_the_same_name() {
    let dir = tempfile::tempdir().unwrap();
    let s = store(dir.path()).await;
    let qwen = EmbeddingSpace { model: "qwen3-embedding:4b".into(), version: "df5bd2e3c74c".into(), dim: 3 };
    s.insert_embeddings(&qwen, &[emb(1, None, 1, [1.0, 0.0, 0.0])]).await.unwrap();
    let hits = s.search(&qwen, &[0.9, 0.1, 0.0], 5, &Filter::default()).await.unwrap();
    assert_eq!(hits.len(), 1);
    // A name the old rule refused is safe now: it's encoded, never put in a query as it is.
    let odd = EmbeddingSpace { model: "arc face".into(), version: "1'; drop".into(), dim: 3 };
    s.insert_embeddings(&odd, &[emb(2, None, 1, [0.0, 1.0, 0.0])]).await.unwrap();
    assert_eq!(s.search(&odd, &[0.0, 1.0, 0.0], 5, &Filter::default()).await.unwrap().len(), 1);
    assert_eq!(s.search(&qwen, &[0.9, 0.1, 0.0], 5, &Filter::default()).await.unwrap().len(), 1, "its own table");
}

#[tokio::test]
async fn bad_input_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let s = store(dir.path()).await;
    // Any name is safe as a table name (encoded); only an empty one, or one with the separator `__`, is refused.
    for (model, version) in [("arc__face", "1"), ("", "1"), ("arcface", "")] {
        let bad = EmbeddingSpace { model: model.into(), version: version.into(), dim: 3 };
        assert!(s.insert_embeddings(&bad, &[emb(1, None, 1, [1.0, 0.0, 0.0])]).await.is_err(), "{model} {version}");
    }
    let wrong_size = EmbeddingRow { key: key(1, None, 1, 1), vector: vec![1.0, 0.0] };
    assert!(s.insert_embeddings(&space(), &[wrong_size]).await.is_err());
    let backwards = EmbeddingRow { key: key(1, None, 5, 4), vector: vec![1.0, 0.0, 0.0] };
    assert!(s.insert_embeddings(&space(), &[backwards]).await.is_err());
    s.insert_embeddings(&space(), &[emb(1, None, 1, [1.0, 0.0, 0.0])]).await.unwrap();
    let same_name_other_size = EmbeddingSpace { dim: 4, ..space() };
    let four = EmbeddingRow { key: key(2, None, 1, 1), vector: vec![1.0, 0.0, 0.0, 0.0] };
    assert!(s.insert_embeddings(&same_name_other_size, &[four]).await.is_err());
    assert!(s.search(&space(), &[1.0, 0.0], 5, &Filter::default()).await.is_err());
}

/// Two payloads, one insert each (as the worker writes), so each row is a fragment of its own: purging one drops its
/// fragment whole and leaves a single fragment, which compaction has nothing to do with.
async fn two_fragments(dir: &Path) -> VectorStore {
    let s = store(dir).await;
    s.insert_embeddings(&space(), &[emb(1, Some(51), 1, [1.0, 0.0, 1.0])]).await.unwrap();
    s.insert_embeddings(&space(), &[emb(2, Some(52), 2, [2.0, 0.0, 1.0])]).await.unwrap();
    s
}

#[tokio::test]
async fn a_purge_of_a_whole_fragment_leaves_its_payload_id_in_no_manifest_or_transaction() {
    let dir = tempfile::tempdir().unwrap();
    let s = two_fragments(dir.path()).await;
    let payload = id(1).hyphenated().to_string();
    assert!(on_disk(dir.path(), &payload));
    assert_eq!(s.purge_payloads(&[id(1)]).await.unwrap(), 1);
    assert!(!on_disk(dir.path(), &payload), "the purged payload id is still in a file");
    assert_eq!(s.rows_for(id(2)).await.unwrap(), 1);
}

#[tokio::test]
async fn a_purge_of_a_whole_fragment_leaves_its_entity_id_in_no_manifest_or_transaction() {
    let dir = tempfile::tempdir().unwrap();
    let s = two_fragments(dir.path()).await;
    let entity = id(51).hyphenated().to_string();
    assert!(on_disk(dir.path(), &entity));
    assert_eq!(s.purge_entity(id(51)).await.unwrap(), 1);
    assert!(!on_disk(dir.path(), &entity), "the purged entity id is still in a file");
    assert_eq!(s.rows_for(id(2)).await.unwrap(), 1);
}

#[tokio::test]
async fn a_purge_retried_after_its_delete_landed_still_leaves_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let s = store(dir.path()).await;
    let marker = "giraffecrossingmarker";
    let rows = [
        TextRow { key: key(1, None, 1, 1), text: format!("a note naming {marker}") },
        TextRow { key: key(2, None, 2, 2), text: "a note that stays".into() },
    ];
    s.insert_texts(&rows).await.unwrap();
    // The worker died after its delete committed and before the rewrite: only the delete happened.
    let raw = lancedb::connect(dir.path().to_str().unwrap()).execute().await.unwrap();
    let texts = raw.open_table(continuum_vector::TEXTS).execute().await.unwrap();
    texts.delete(format!("payload_id = '{}'", id(1).hyphenated()).as_str()).await.unwrap();
    assert!(on_disk(dir.path(), marker));
    // The broker retries the purge: nothing is left to delete, and the bytes must still go.
    assert_eq!(s.purge_payloads(&[id(1)]).await.unwrap(), 0);
    assert!(!on_disk(dir.path(), marker));
}

#[tokio::test]
async fn an_empty_or_backwards_time_range_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let s = store(dir.path()).await;
    s.insert_embeddings(&space(), &[emb(1, None, 150, [1.0, 0.0, 0.0])]).await.unwrap();
    s.insert_texts(&[TextRow { key: key(2, None, 150, 150), text: "kettle".into() }]).await.unwrap();
    for during in [(100, 100), (200, 100)] {
        let f = Filter { during: Some(during), subjects: vec![], labelled: None };
        assert!(s.search(&space(), &[1.0, 0.0, 0.0], 5, &f).await.is_err(), "{during:?}");
        assert!(s.text_search("kettle", 5, &f).await.is_err(), "{during:?}");
    }
}

#[tokio::test]
async fn inserts_and_compaction_keep_one_text_index_and_one_version() {
    let dir = tempfile::tempdir().unwrap();
    let s = store(dir.path()).await;
    for n in 1..=5 {
        s.insert_texts(&[TextRow { key: key(n, None, n as u64, n as u64), text: format!("entry number {n}") }])
            .await
            .unwrap();
    }
    // Rows added after the index are searchable before any compaction.
    assert_eq!(s.text_search("entry", 10, &Filter::default()).await.unwrap().len(), 5);
    s.compact().await.unwrap();
    let texts = dir.path().join("texts.lance");
    assert_eq!(std::fs::read_dir(texts.join("_indices")).unwrap().count(), 1);
    let manifests = std::fs::read_dir(texts.join("_versions"))
        .unwrap()
        .filter(|e| e.as_ref().unwrap().file_name().to_string_lossy().ends_with(".manifest"))
        .count();
    assert_eq!(manifests, 1); // beside it, Lance keeps only latest_version_hint.json
    assert_eq!(s.text_search("entry", 10, &Filter::default()).await.unwrap().len(), 5);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn inserts_running_alongside_purges_all_stay_readable() {
    let dir = tempfile::tempdir().unwrap();
    let s = std::sync::Arc::new(store(dir.path()).await);
    s.insert_embeddings(&space(), &[emb(1000, None, 1, [1.0, 0.0, 0.0])]).await.unwrap();
    let writer = {
        let s = s.clone();
        tokio::spawn(async move {
            for n in 1..=20 {
                s.insert_embeddings(&space(), &[emb(n, None, n as u64, [n as f32, 1.0, 0.0])]).await.unwrap();
            }
        })
    };
    for n in 2000..2010 {
        s.purge_payloads(&[id(n)]).await.unwrap();
    }
    writer.await.unwrap();
    for n in 1..=20 {
        assert_eq!(s.rows_for(id(n)).await.unwrap(), 1, "row {n}");
    }
    assert_eq!(nearest(&s, [5.0, 1.0, 0.0], &Filter::default()).await.len(), 10);
}

#[tokio::test]
async fn a_row_for_something_already_purged_is_never_filed() {
    // A purge can arrive while an embedding it names is still waiting to be filed: filing it afterwards would bring
    // back what was forgotten.
    let dir = tempfile::tempdir().unwrap();
    let s = store(dir.path()).await;
    s.purge_payloads(&[id(1)]).await.unwrap();
    s.purge_entity(id(70)).await.unwrap();
    s.insert_embeddings(&space(), &[emb(1, None, 1, [1.0, 0.0, 0.0]), emb(2, None, 2, [0.0, 1.0, 0.0])]).await.unwrap();
    s.insert_embeddings(&space(), &[emb(3, Some(70), 3, [0.0, 0.0, 1.0])]).await.unwrap();
    s.insert_texts(&[TextRow { key: key(1, None, 1, 1), text: "derived from one".into() }]).await.unwrap();
    assert_eq!(s.rows_for(id(1)).await.unwrap(), 0, "the purged payload");
    assert_eq!(s.rows_for(id(3)).await.unwrap(), 0, "a row about the purged entity");
    assert_eq!(s.rows_for(id(2)).await.unwrap(), 1, "everything else still files");
}

#[tokio::test]
async fn the_labelled_filter_keeps_rows_with_or_without_an_entity() {
    let dir = tempfile::tempdir().unwrap();
    let s = store(dir.path()).await;
    s.insert_embeddings(&space(), &[emb(1, Some(70), 1, [1.0, 0.0, 0.0]), emb(2, None, 2, [0.9, 0.1, 0.0])])
        .await
        .unwrap();
    let ids = async |f: Filter| {
        s.search(&space(), &[1.0, 0.0, 0.0], 10, &f).await.unwrap().iter().map(|h| h.key.payload_id).collect::<Vec<_>>()
    };
    assert_eq!(ids(Filter { labelled: Some(true), ..Filter::default() }).await, vec![id(1)], "enrolled only");
    assert_eq!(ids(Filter { labelled: Some(false), ..Filter::default() }).await, vec![id(2)], "unknowns only");
    assert_eq!(ids(Filter::default()).await, vec![id(1), id(2)]);
}
