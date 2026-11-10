//! The vector worker's job logic against a temporary store: query and result payloads, answering searches,
//! filing embeddings and text, and purges.

use continuum_vector::EmbeddingSpace;
use continuum_vector::jobs::{
    MAX_K, TextQuery, VectorQuery, answer_text_query, answer_vector_query, decode_hits, embedding_vector, encode_hits,
    index_embedding, index_text, purge, space_of_shape,
};
use continuum_vector::{Filter, Hit, Key, VectorStore};
use uuid::Uuid;

fn id(n: u128) -> Uuid {
    Uuid::from_u128((n & !(0xF << 76) & !(0x3 << 62)) | (0x7 << 76) | (0x2 << 62))
}

fn key(payload: u128, entity: Option<u128>, at: u64) -> Key {
    Key { payload_id: id(payload), entity_id: entity.map(id), parent_id: None, start_ns: at, end_ns: at }
}

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

const SHAPE: &str = "embedding/arcface/r100";

#[test]
fn a_vector_query_round_trips_with_and_without_a_filter() {
    for filter in [Filter::default(), Filter { during: Some((10, 20)), subjects: vec![id(7), id(8)], labelled: None }] {
        let q =
            VectorQuery { model: "arcface".into(), version: "r100".into(), k: 5, vector: vec![0.5, -1.0, 2.0], filter };
        assert_eq!(VectorQuery::decode(&q.encode().unwrap()).unwrap(), q);
    }
}

#[test]
fn a_text_query_round_trips() {
    let q = TextQuery {
        text: "the kettle".into(),
        k: 3,
        filter: Filter { during: Some((1, 2)), subjects: vec![id(9)], labelled: Some(false) },
    };
    assert_eq!(TextQuery::decode(&q.encode().unwrap()).unwrap(), q);
}

#[test]
fn hits_round_trip_including_none() {
    let hits = vec![
        Hit { key: Key { parent_id: Some(id(5)), ..key(1, Some(7), 10) }, score: 0.25 },
        Hit { key: Key { payload_id: id(2), entity_id: None, parent_id: None, start_ns: 20, end_ns: 30 }, score: 0.5 },
    ];
    assert_eq!(decode_hits(&encode_hits(&hits).unwrap()).unwrap(), hits);
    assert_eq!(decode_hits(&encode_hits(&[]).unwrap()).unwrap(), vec![]);
}

#[test]
fn bad_payloads_are_refused() {
    assert!(VectorQuery::decode(b"not arrow").is_err());
    assert!(TextQuery::decode(&encode_hits(&[]).unwrap()).is_err()); // a result isn't a query
    assert!(embedding_vector(&[]).is_err());
    assert!(embedding_vector(&[0, 0, 0]).is_err());
    assert!(embedding_vector(&f32s(&[1.0, f32::NAN])).is_err());
    assert!(space_of_shape("caption/qwen/1", 3).is_err());
    assert!(space_of_shape("embedding/arcface", 3).is_err());
    assert_eq!(space_of_shape(SHAPE, 3).unwrap().table_name().unwrap(), "emb__arcface__r100");
}

#[test]
fn an_ollama_model_name_is_a_space_and_its_slash_isnt_the_version() {
    // The journal's embeddings: the Ollama model, colon and all, and the weights' digest.
    let journal = space_of_shape("embedding/qwen3-embedding:4b/df5bd2e3c74c", 1024).unwrap();
    assert_eq!((journal.model.as_str(), journal.version.as_str()), ("qwen3-embedding:4b", "df5bd2e3c74c"));
    assert_eq!(journal.table_name().unwrap(), "emb__qwen3-2dembedding-3a4b__df5bd2e3c74c");
    // A namespaced model keeps its slash; the version is after the last one.
    let spaced = space_of_shape("embedding/user/model:tag/abc123", 3).unwrap();
    assert_eq!((spaced.model.as_str(), spaced.version.as_str()), ("user/model:tag", "abc123"));
    // Names that look alike keep tables of their own.
    let name = |model: &str| EmbeddingSpace { model: model.into(), version: "1".into(), dim: 3 }.table_name().unwrap();
    let names = ["a:b", "a-3ab", "a-b", "a b", "a/b", "a.b", "a_b"].map(name);
    let unique: std::collections::BTreeSet<_> = names.iter().collect();
    assert_eq!(unique.len(), names.len(), "{names:?}");
}

#[tokio::test]
async fn embeddings_are_filed_and_a_query_payload_is_answered() {
    let dir = tempfile::tempdir().unwrap();
    let store = VectorStore::open(dir.path()).await.unwrap();
    index_embedding(&store, key(1, Some(7), 100), SHAPE, &f32s(&[1.0, 0.0, 0.0])).await.unwrap();
    index_embedding(&store, key(2, Some(8), 200), SHAPE, &f32s(&[0.0, 1.0, 0.0])).await.unwrap();
    let ask = |vector: Vec<f32>, filter: Filter| {
        VectorQuery { model: "arcface".into(), version: "r100".into(), k: 5, vector, filter }.encode().unwrap()
    };
    let hits =
        decode_hits(&answer_vector_query(&store, &ask(vec![0.9, 0.1, 0.0], Filter::default())).await.unwrap()).unwrap();
    assert_eq!(hits.iter().map(|h| h.key.payload_id).collect::<Vec<_>>(), vec![id(1), id(2)]);
    assert_eq!(hits[0].key, key(1, Some(7), 100));
    // The query's filter reaches the store.
    let only = Filter { during: None, subjects: vec![id(8)], labelled: None };
    let hits = decode_hits(&answer_vector_query(&store, &ask(vec![0.9, 0.1, 0.0], only)).await.unwrap()).unwrap();
    assert_eq!(hits.iter().map(|h| h.key.payload_id).collect::<Vec<_>>(), vec![id(2)]);
    // A vector of another size against this model's table is refused, never answered with nonsense.
    assert!(answer_vector_query(&store, &ask(vec![1.0, 0.0], Filter::default())).await.is_err());
    // k is 1 to MAX_K.
    let with_k = |k: u32| {
        VectorQuery {
            model: "arcface".into(),
            version: "r100".into(),
            k,
            vector: vec![1.0, 0.0, 0.0],
            filter: Filter::default(),
        }
        .encode()
        .unwrap()
    };
    assert_eq!(decode_hits(&answer_vector_query(&store, &with_k(MAX_K)).await.unwrap()).unwrap().len(), 2);
    assert!(answer_vector_query(&store, &with_k(0)).await.is_err());
    assert!(answer_vector_query(&store, &with_k(MAX_K + 1)).await.is_err());
}

#[tokio::test]
async fn text_is_filed_and_a_text_query_payload_is_answered() {
    let dir = tempfile::tempdir().unwrap();
    let store = VectorStore::open(dir.path()).await.unwrap();
    index_text(&store, key(3, None, 50), "Dana boiled the kettle").await.unwrap();
    index_text(&store, key(4, None, 60), "a walk in the park").await.unwrap();
    let q = TextQuery { text: "kettle".into(), k: 5, filter: Filter::default() }.encode().unwrap();
    let hits = decode_hits(&answer_text_query(&store, &q).await.unwrap()).unwrap();
    assert_eq!(hits.iter().map(|h| h.key.payload_id).collect::<Vec<_>>(), vec![id(3)]);
}

#[tokio::test]
async fn a_purge_notice_removes_its_payloads_rows() {
    let dir = tempfile::tempdir().unwrap();
    let store = VectorStore::open(dir.path()).await.unwrap();
    index_embedding(&store, key(1, None, 1), SHAPE, &f32s(&[1.0, 0.0, 0.0])).await.unwrap();
    index_text(&store, key(1, None, 1), "derived from payload one").await.unwrap();
    index_embedding(&store, key(2, None, 2), SHAPE, &f32s(&[0.0, 1.0, 0.0])).await.unwrap();
    assert_eq!(purge(&store, &[id(1)]).await.unwrap(), 2);
    assert_eq!(store.rows_for(id(1)).await.unwrap(), 0);
    assert_eq!(store.rows_for(id(2)).await.unwrap(), 1);
}

#[test]
fn a_full_result_stays_an_inline_payload() {
    // Every id set, so the largest a result of MAX_K hits can be.
    let hits: Vec<Hit> = (0..u128::from(MAX_K))
        .map(|i| Hit { key: Key { parent_id: Some(id(i + 500)), ..key(i, Some(i + 1000), 10) }, score: 0.5 })
        .collect();
    assert!(encode_hits(&hits).unwrap().len() <= continuum_core::MAX_INLINE_BYTES);
}
