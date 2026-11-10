//! The vector worker service against a real broker: it waits for an embedding stream that isn't registered yet, files
//! what's committed to it once it is, answers a live query with it, and stops cleanly when asked; and it files the
//! journal's entry text and its prefixed embedding streams.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use continuum_broker::server::{Config, Server};
use continuum_client::{Client, ClientOptions, Start, StreamSpec, StreamStorage, SubEvent, SubscribeSpec};
use continuum_core::schema::continuum as fb;
use continuum_core::{HeaderBuf, HeaderSpec, PayloadFormat, StorageLocator, StreamId, Uuid};
use continuum_vector::Filter;
use continuum_vector::jobs::{
    TEXT_QUERY, TEXT_SEARCH, TextQuery, VECTOR_QUERY, VECTOR_SEARCH, VectorQuery, decode_hits,
};
use continuum_vector::service::{Settings, run};
use continuum_vector::worker::search_stream;

fn v7(n: u8) -> Uuid {
    let mut id = [n; 16];
    id[6] = 0x70 | (n & 0x0f);
    id[8] = 0x80 | (n & 0x3f);
    Uuid(id)
}

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Asks the worker for the nearest embedding to `vector` through the live path, or `None` when the answer has no
/// hits (the embedding isn't filed yet).
async fn nearest(c: &Client, queries: StreamId, at: u64, vector: Vec<f32>) -> Vec<continuum_vector::Hit> {
    let mut follow = SubscribeSpec::new([VECTOR_SEARCH], Start::Live);
    follow.pinned = true;
    let mut results = c.subscribe(&follow).await.unwrap();
    let id = Uuid(uuid::Uuid::now_v7().into_bytes());
    let mut h = HeaderSpec::raw(id, queries, at, at);
    h.shape_name = VECTOR_QUERY.into();
    h.format = PayloadFormat::ArrowRecordBatch;
    let q =
        VectorQuery { model: "arcface".into(), version: "buffalo_l".into(), k: 5, vector, filter: Filter::default() };
    let at = StorageLocator::Inline { bytes: q.encode().unwrap() };
    c.ingest_acked(&HeaderBuf::build(&h).unwrap(), Some(&at)).await.unwrap();
    loop {
        // A query committed before the worker's live query subscription started is never answered (it registers the
        // search streams first): that's "not filed yet" to the caller, which asks again.
        let Ok(event) = tokio::time::timeout(Duration::from_secs(1), results.next()).await else {
            results.unsubscribe().await.unwrap();
            return Vec::new();
        };
        let event = event.unwrap();
        if let SubEvent::Event { item, .. } = event
            && item.header.parents() == vec![id]
        {
            let Some(StorageLocator::Inline { bytes }) = item.locator else { panic!("an inline result") };
            results.unsubscribe().await.unwrap();
            return decode_hits(&bytes).unwrap();
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_service_waits_for_an_embedding_stream_files_it_and_answers_from_it() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("b.sock");
    let server = Server::bind(Config::new(socket.clone(), Duration::from_secs(30))).await.unwrap();
    let (stop_broker, stopped) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(server.run_until(async {
        let _ = stopped.await;
    }));

    // The service starts first: face.embedding isn't registered yet.
    let mut settings = Settings::new(socket.clone(), dir.path().join("vector"));
    settings.embeddings = vec!["face.embedding".into()];
    settings.wait_for_streams = Duration::from_millis(50);
    let stop = Arc::new(AtomicBool::new(false));
    let service = tokio::spawn(run(settings, Arc::clone(&stop)));

    // Then the face worker registers its stream and commits an embedding about person 30.
    let faces = Client::connect(&socket, ClientOptions::new("faces")).await.unwrap();
    let spec = StreamSpec {
        name: "face.embedding".into(),
        kind: fb::StreamKind::Derived,
        format: fb::PayloadFormat::RawBytes,
        shape_name: "embedding/arcface/buffalo_l".into(),
        shape_version: 1,
        nominal_rate_hz: 0.0,
        storage: StreamStorage::Slab { class: 0 },
        anchor: None,
        layer: fb::DomainLayer::Layer2Decision,
    };
    let stream = StreamId(faces.register_stream(&spec).await.unwrap().stream_id);
    let mut h = HeaderSpec::raw(v7(1), stream, 100, 100);
    h.shape_name = "embedding/arcface/buffalo_l".into();
    h.subjects = vec![v7(30)];
    let at = StorageLocator::Inline { bytes: f32s(&[1.0, 0.0, 0.0]) };
    faces.ingest_acked(&HeaderBuf::build(&h).unwrap(), Some(&at)).await.unwrap();

    // A consumer's live query finds it once the service has filed it (the service registered the query stream).
    let consumer = Client::connect(&socket, ClientOptions::new("consumer")).await.unwrap();
    let mut queries = None;
    for _ in 0..250 {
        if let Ok(r) = consumer.register_stream(&search_stream(VECTOR_QUERY)).await
            && consumer.list_streams().await.unwrap().iter().any(|s| s.name == VECTOR_SEARCH)
        {
            queries = Some(StreamId(r.stream_id));
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let queries = queries.expect("the service registered the search streams");
    let mut hits = Vec::new();
    for n in 0..250u64 {
        hits = nearest(&consumer, queries, 1_000 + n, vec![0.9, 0.1, 0.0]).await;
        if !hits.is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let [hit] = hits.as_slice() else { panic!("one hit, not {hits:?}") };
    assert_eq!(hit.key.payload_id, uuid::Uuid::from_bytes(v7(1).0));
    assert_eq!(hit.key.entity_id, Some(uuid::Uuid::from_bytes(v7(30).0)));

    // Asked to stop, it returns cleanly.
    stop.store(true, Ordering::SeqCst);
    let stopped = tokio::time::timeout(Duration::from_secs(10), service).await.unwrap().unwrap();
    assert!(stopped.is_ok(), "{stopped:?}");
    let _ = stop_broker.send(());
}

/// Commits a query payload of `shape` on `queries` at `at` and returns the hits of the result that answers it, from
/// the live `results` stream.
async fn ask(
    c: &Client,
    queries: StreamId,
    at: u64,
    shape: &str,
    bytes: Vec<u8>,
    results_stream: &str,
) -> Vec<continuum_vector::Hit> {
    let mut follow = SubscribeSpec::new([results_stream], Start::Live);
    follow.pinned = true;
    let mut results = c.subscribe(&follow).await.unwrap();
    let id = Uuid(uuid::Uuid::now_v7().into_bytes());
    let mut h = HeaderSpec::raw(id, queries, at, at);
    h.shape_name = shape.into();
    h.format = PayloadFormat::ArrowRecordBatch;
    c.ingest_acked(&HeaderBuf::build(&h).unwrap(), Some(&StorageLocator::Inline { bytes })).await.unwrap();
    loop {
        // A query committed before the worker's live query subscription started is never answered (it registers the
        // search streams first): that's "not filed yet" to the caller, which asks again.
        let Ok(event) = tokio::time::timeout(Duration::from_secs(1), results.next()).await else {
            results.unsubscribe().await.unwrap();
            return Vec::new();
        };
        let event = event.unwrap();
        if let SubEvent::Event { item, .. } = event
            && item.header.parents() == vec![id]
        {
            let Some(StorageLocator::Inline { bytes }) = item.locator else { panic!("an inline result") };
            results.unsubscribe().await.unwrap();
            return decode_hits(&bytes).unwrap();
        }
    }
}

/// Registers a query stream once the service has registered its results stream.
async fn query_stream(c: &Client, query: &str, results: &str) -> StreamId {
    for _ in 0..250 {
        if let Ok(r) = c.register_stream(&search_stream(query)).await
            && c.list_streams().await.unwrap().iter().any(|s| s.name == results)
        {
            return StreamId(r.stream_id);
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("the service never registered {results}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_service_files_journal_entry_text_and_the_journals_embedding_streams_by_prefix() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("b.sock");
    let server = Server::bind(Config::new(socket.clone(), Duration::from_secs(30))).await.unwrap();
    let (stop_broker, stopped) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(server.run_until(async {
        let _ = stopped.await;
    }));
    let mut settings = Settings::new(socket.clone(), dir.path().join("vector"));
    settings.embeddings = Vec::new(); // only the journal's, which the defaults add
    settings.wait_for_streams = Duration::from_millis(50);
    let stop = Arc::new(AtomicBool::new(false));
    let service = tokio::spawn(run(settings, Arc::clone(&stop)));

    // The journal worker registers its entries and, later, an embedding stream for its model.
    let journal = Client::connect(&socket, ClientOptions::new("journal")).await.unwrap();
    let stream = |name: &str, format, shape: &str| StreamSpec {
        name: name.into(),
        kind: fb::StreamKind::Derived,
        format,
        shape_name: shape.into(),
        shape_version: 1,
        nominal_rate_hz: 0.0,
        storage: StreamStorage::Slab { class: 0 },
        anchor: None,
        layer: fb::DomainLayer::Layer3Interpretation,
    };
    let entries = journal.register_stream(&stream("journal.entry", fb::PayloadFormat::Text, "journal.entry")).await;
    let entries = StreamId(entries.unwrap().stream_id);
    let entry = |n: u8, at: u64, body: &str, subjects: Vec<Uuid>| {
        let mut h = HeaderSpec::raw(v7(n), entries, at, at + 30);
        h.shape_name = "journal.entry".into();
        h.format = PayloadFormat::Text;
        h.subjects = subjects;
        (HeaderBuf::build(&h).unwrap(), StorageLocator::Inline { bytes: body.as_bytes().to_vec() })
    };
    for (h, at) in [
        entry(1, 100, r#"{"text": "Dana planted tomatoes by the fence.", "level": 30}"#, vec![v7(30), v7(31)]),
        entry(2, 200, r#"{"text": "Alice fixed the tomato cage.", "level": 30}"#, vec![v7(32)]),
        entry(3, 300, r#"{"status": "quiet", "level": 30}"#, vec![]),
    ] {
        journal.ingest_acked(&h, Some(&at)).await.unwrap();
    }
    let digest = "embedding/qwen3/abc123def456";
    let model_spec = stream("journal.embedding.abc123def456", fb::PayloadFormat::RawBytes, digest);
    let model = StreamId(journal.register_stream(&model_spec).await.unwrap().stream_id);
    let mut h = HeaderSpec::raw(v7(4), model, 100, 130);
    h.shape_name = digest.into();
    h.parents = vec![v7(1)];
    h.subjects = vec![v7(30)];
    let at = StorageLocator::Inline { bytes: f32s(&[0.0, 1.0, 0.0]) };
    journal.ingest_acked(&HeaderBuf::build(&h).unwrap(), Some(&at)).await.unwrap();

    let consumer = Client::connect(&socket, ClientOptions::new("consumer")).await.unwrap();
    let texts = query_stream(&consumer, TEXT_QUERY, TEXT_SEARCH).await;
    let vectors = query_stream(&consumer, VECTOR_QUERY, VECTOR_SEARCH).await;
    let text = |words: &str, subjects: Vec<Uuid>| TextQuery {
        text: words.into(),
        k: 10,
        filter: Filter {
            subjects: subjects.into_iter().map(|s| uuid::Uuid::from_bytes(s.0)).collect(),
            ..Filter::default()
        },
    };
    // Every entry's text is filed, one row per person in it: a search filtered to Dana's second id finds hers only.
    let mut hits = Vec::new();
    for n in 0..250u64 {
        hits =
            ask(&consumer, texts, 1_000 + n, TEXT_QUERY, text("tomatoes", vec![v7(31)]).encode().unwrap(), TEXT_SEARCH)
                .await;
        if !hits.is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let [hit] = hits.as_slice() else { panic!("Dana's entry alone, not {hits:?}") };
    assert_eq!(
        (hit.key.payload_id, hit.key.entity_id),
        (uuid::Uuid::from_bytes(v7(1).0), Some(uuid::Uuid::from_bytes(v7(31).0)))
    );
    let all = ask(&consumer, texts, 2_000, TEXT_QUERY, text("tomato", vec![]).encode().unwrap(), TEXT_SEARCH).await;
    let mut ids: Vec<_> = all.iter().map(|h| h.key.payload_id).collect();
    ids.sort();
    ids.dedup();
    assert_eq!(ids.len(), 2, "both entries with text, and nothing for the quiet marker: {all:?}");

    // The model's embedding stream, registered after the service started, was found by its prefix and filed.
    let q = VectorQuery {
        model: "qwen3".into(),
        version: "abc123def456".into(),
        k: 5,
        vector: vec![0.0, 0.9, 0.1],
        filter: Filter::default(),
    };
    let mut near = Vec::new();
    for n in 0..250u64 {
        near = ask(&consumer, vectors, 3_000 + n, VECTOR_QUERY, q.encode().unwrap(), VECTOR_SEARCH).await;
        if !near.is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let [hit] = near.as_slice() else { panic!("the entry's embedding, not {near:?}") };
    assert_eq!(hit.key.parent_id, Some(uuid::Uuid::from_bytes(v7(1).0)), "its parent is the entry");

    stop.store(true, Ordering::SeqCst);
    let stopped = tokio::time::timeout(Duration::from_secs(10), service).await.unwrap().unwrap();
    assert!(stopped.is_ok(), "{stopped:?}");
    let _ = stop_broker.send(());
}
