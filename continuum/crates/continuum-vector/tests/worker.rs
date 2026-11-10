//! The vector worker's loop against a stand-in broker: a dispatched search is answered with one result payload on
//! the capability's stream (the query its parent, over the query's span), a query in shared memory is read from its
//! slot, a job it can't answer is turned down, and a purge notice removes the rows before it's acknowledged; and,
//! against the real broker, embeddings come in by a durable subscription that resumes after the last one filed.

use std::os::fd::{AsFd, OwnedFd};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use continuum_broker::server::{Config, Server};
use continuum_client::{Client, ClientOptions, Start, StreamSpec, StreamStorage, SubscribeSpec};
use continuum_core::schema::continuum as fb;
use continuum_core::{
    HeaderBuf, HeaderSpec, PayloadFormat, SCHEMA_MAJOR, SCHEMA_MINOR, StorageLocator, StreamId, Uuid,
};
use continuum_shm::{Arena, ArenaConfig, SlabClass};
use continuum_vector::jobs::{
    TEXT_SEARCH, VECTOR_QUERY, VECTOR_SEARCH, VectorQuery, decode_hits, index_embedding, index_text,
};
use continuum_vector::worker::{
    INDEX_SUBSCRIPTION, SearchStreams, SearchWorker, WorkerError, register, search_stream, text_of,
};
use continuum_vector::{Filter, Key, VectorStore};
use flatbuffers::FlatBufferBuilder;
use tokio::net::UnixListener;
use tokio::sync::mpsc;

fn v7(n: u8) -> Uuid {
    let mut id = [n; 16];
    id[6] = 0x70 | (n & 0x0f);
    id[8] = 0x80 | (n & 0x3f);
    Uuid(id)
}

fn uid(n: u8) -> uuid::Uuid {
    uuid::Uuid::from_bytes(v7(n).0)
}

const STREAMS: SearchStreams = SearchStreams { vector: StreamId(40), text: StreamId(41) };
const QUERY_STREAM: StreamId = StreamId(12);

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// A query payload's header: payload `n`, on the query stream over [1000, 1500), about subject 30.
fn query_header(n: u8, shape: &str) -> HeaderBuf {
    let mut spec = HeaderSpec::raw(v7(n), QUERY_STREAM, 1_000, 1_500);
    spec.shape_name = shape.to_owned();
    spec.format = PayloadFormat::ArrowRecordBatch;
    spec.subjects = vec![v7(30)];
    HeaderBuf::build(&spec).unwrap()
}

fn ask(vector: Vec<f32>) -> Vec<u8> {
    VectorQuery { model: "arcface".into(), version: "r100".into(), k: 5, vector, filter: Filter::default() }
        .encode()
        .unwrap()
}

fn envelope(
    f: impl FnOnce(
        &mut FlatBufferBuilder<'static>,
    ) -> (fb::ControlPayload, flatbuffers::WIPOffset<flatbuffers::UnionWIPOffset>),
    request_id: u64,
) -> Vec<u8> {
    let mut b = FlatBufferBuilder::new();
    let (kind, payload) = f(&mut b);
    let env = fb::ControlEnvelope::create(
        &mut b,
        &fb::ControlEnvelopeArgs { request_id, payload_type: kind, payload: Some(payload) },
    );
    b.finish(env, None);
    b.finished_data().to_vec()
}

/// A `DispatchJob` (job `job`, output lease 91) for `capability` with these inputs.
fn dispatch(job: u8, capability: &str, inputs: &[(HeaderBuf, Option<StorageLocator>)]) -> Vec<u8> {
    envelope(
        |b| {
            let id = fb::Uuid::new(&v7(job).0);
            let promises = b.create_vector(&[fb::Uuid::new(&v7(2).0)]);
            let capability = b.create_string(capability);
            let items: Vec<_> = inputs
                .iter()
                .map(|(h, l)| {
                    let h = b.create_vector(h.as_bytes());
                    let l = l.as_ref().map(|l| l.build_into(b));
                    fb::Item::create(b, &fb::ItemArgs { locator: l, header_bytes: Some(h), commit_seq: 0 })
                })
                .collect();
            let items = b.create_vector(&items);
            let output = fb::OutputLease::create(
                b,
                &fb::OutputLeaseArgs { lease_id: 91, slots: None, expires_at_ns: u64::MAX / 2 },
            );
            let range = fb::TimeRange::new(1_000, 1_500);
            let j = fb::DispatchJob::create(
                b,
                &fb::DispatchJobArgs {
                    job_id: Some(&id),
                    promise_ids: Some(promises),
                    capability: Some(capability),
                    inputs: Some(items),
                    output_range: Some(&range),
                    band: fb::PriorityBand::P1Realtime,
                    deadline_ns: u64::MAX / 2,
                    output: Some(output),
                    capability_id: 1,
                },
            );
            (fb::ControlPayload::DispatchJob, j.as_union_value())
        },
        0,
    )
}

fn purge_notice(purge: u8, ids: &[u8]) -> Vec<u8> {
    envelope(
        |b| {
            let purge = fb::Uuid::new(&v7(purge).0);
            let ids: Vec<_> = ids.iter().map(|&n| fb::Uuid::new(&v7(n).0)).collect();
            let ids = b.create_vector(&ids);
            let p = fb::PurgeNotice::create(
                b,
                &fb::PurgeNoticeArgs { purge_id: Some(&purge), payload_ids: Some(ids), more: false, spans: None },
            );
            (fb::ControlPayload::PurgeNotice, p.as_union_value())
        },
        0,
    )
}

/// A `Welcome` handing over `arena` (or none).
fn welcome(arena: Option<&Arena>) -> Vec<u8> {
    envelope(
        |b| {
            let infos: Vec<_> = arena
                .iter()
                .map(|a| {
                    let name = b.create_string(a.name());
                    let class = fb::SizeClass::create(b, &fb::SizeClassArgs { slab_bytes: 64 * 1024, count: 2 });
                    let classes = b.create_vector(&[class]);
                    fb::ArenaInfo::create(
                        b,
                        &fb::ArenaInfoArgs {
                            name: Some(name),
                            generation: a.generation(),
                            size_bytes: a.len() as u64,
                            classes: Some(classes),
                        },
                    )
                })
                .collect();
            let infos = b.create_vector(&infos);
            let schema =
                fb::SchemaVersion::create(b, &fb::SchemaVersionArgs { major: SCHEMA_MAJOR, minor: SCHEMA_MINOR });
            let id = fb::Uuid::new(&[7; 16]);
            let w = fb::Welcome::create(
                b,
                &fb::WelcomeArgs {
                    session_id: Some(&id),
                    schema: Some(schema),
                    arenas: Some(infos),
                    ..Default::default()
                },
            );
            (fb::ControlPayload::Welcome, w.as_union_value())
        },
        0,
    )
}

/// What the stand-in heard, other than heartbeats.
#[derive(Debug)]
enum Heard {
    Commit { outputs: Vec<(HeaderBuf, StorageLocator)>, stage_range: Option<(u32, (u64, u64))> },
    Rejected(Uuid, String),
    PurgeAck(Uuid),
}

/// A stand-in broker: it hands over `arena`'s fd, pushes `pushes`, answers a commit with `LeaseReleased`, and reports
/// what it hears.
async fn stand(
    socket: &std::path::Path,
    arena: Option<&Arena>,
    pushes: Vec<Vec<u8>>,
) -> mpsc::UnboundedReceiver<Heard> {
    let listener = UnixListener::bind(socket).unwrap();
    let hello = welcome(arena);
    let fds: Vec<OwnedFd> = arena.iter().map(|a| dup(a)).collect();
    let (tx, rx) = mpsc::unbounded_channel();
    tokio::spawn(async move {
        let (mut s, _) = listener.accept().await.unwrap();
        let mut buf = Vec::new();
        continuum_wire::read_frame(&mut s, &mut buf).await.unwrap();
        let borrowed: Vec<_> = fds.iter().map(AsFd::as_fd).collect();
        continuum_wire::write_frame_with_fds(&mut s, &hello, &borrowed).await.unwrap();
        for p in &pushes {
            continuum_wire::write_frame(&mut s, p).await.unwrap();
        }
        while continuum_wire::read_frame(&mut s, &mut buf).await.is_ok() {
            let env = continuum_wire::read_envelope(&buf).unwrap();
            let heard = if let Some(c) = env.payload_as_commit_lease() {
                let outputs = c
                    .outputs()
                    .unwrap()
                    .iter()
                    .map(|i| {
                        let h = HeaderBuf::new(i.header_bytes().unwrap().bytes().to_vec()).unwrap();
                        (h, StorageLocator::from_table(&i.locator().unwrap()).unwrap())
                    })
                    .collect();
                let stage_range = c.coverage().map(|d| {
                    let r = d.range().unwrap();
                    (d.source_stream_id(), (r.start_ns(), r.end_ns()))
                });
                let lease = c.lease_id();
                let r = envelope(
                    |b| {
                        let r =
                            fb::LeaseReleased::create(b, &fb::LeaseReleasedArgs { lease_id: lease, output_seqs: None });
                        (fb::ControlPayload::LeaseReleased, r.as_union_value())
                    },
                    env.request_id(),
                );
                continuum_wire::write_frame(&mut s, &r).await.unwrap();
                Heard::Commit { outputs, stage_range }
            } else if let Some(j) = env.payload_as_job_rejected() {
                Heard::Rejected(Uuid::from(j.job_id().unwrap()), j.reason().unwrap_or_default().to_owned())
            } else if let Some(a) = env.payload_as_purge_ack() {
                Heard::PurgeAck(Uuid::from(a.purge_id().unwrap()))
            } else {
                continue;
            };
            let _ = tx.send(heard);
        }
    });
    rx
}

fn dup(a: &Arena) -> OwnedFd {
    let name = std::ffi::CString::new(a.name()).unwrap();
    // SAFETY: a valid NUL-terminated name.
    let raw = unsafe { libc::shm_open(name.as_ptr(), libc::O_RDWR, 0) };
    assert!(raw >= 0);
    // SAFETY: a fresh fd this test owns.
    unsafe { std::os::fd::FromRawFd::from_raw_fd(raw) }
}

/// An arena unlinked when the test ends.
struct TestArena(Arena);

impl Drop for TestArena {
    fn drop(&mut self) {
        Arena::unlink(self.0.name()).ok();
    }
}

fn arena() -> TestArena {
    static N: AtomicU32 = AtomicU32::new(0);
    let name = format!("/cnt.v.{}.{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed));
    let config =
        ArenaConfig { classes: vec![SlabClass { slot_size: 64 * 1024, slot_count: 2 }], ..ArenaConfig::default() };
    TestArena(Arena::create(&name, &config).unwrap())
}

/// A store with two faces: payload 1 near [1, 0, 0] and payload 2 near [0, 1, 0].
async fn store(dir: &std::path::Path) -> VectorStore {
    let s = VectorStore::open(dir).await.unwrap();
    // Payload 1 is about entity 40, so a result that hits it names entity 40 too.
    let key = |n: u8, at: u64| Key {
        payload_id: uid(n),
        entity_id: (n == 1).then(|| uid(40)),
        parent_id: None,
        start_ns: at,
        end_ns: at,
    };
    index_embedding(&s, key(1, 100), "embedding/arcface/r100", &f32s(&[1.0, 0.0, 0.0])).await.unwrap();
    index_embedding(&s, key(2, 200), "embedding/arcface/r100", &f32s(&[0.0, 1.0, 0.0])).await.unwrap();
    s
}

/// Connects, handles `events` events, and returns the client's worker and what each handle returned.
async fn work(socket: &std::path::Path, store: VectorStore, events: usize) -> Vec<Result<(), WorkerError>> {
    let client = Arc::new(Client::connect(socket, ClientOptions::new("vector")).await.unwrap());
    let worker = SearchWorker::new(client.clone(), store, STREAMS);
    let mut results = Vec::new();
    for _ in 0..events {
        let m = tokio::time::timeout(Duration::from_secs(5), client.next_event()).await.unwrap().unwrap();
        results.push(worker.handle(m.event()).await);
    }
    results
}

async fn next(rx: &mut mpsc::UnboundedReceiver<Heard>) -> Heard {
    tokio::time::timeout(Duration::from_secs(5), rx.recv()).await.unwrap().unwrap()
}

#[tokio::test]
async fn a_vector_search_is_answered_with_one_result_whose_parent_is_the_query() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("s.sock");
    let query = query_header(9, VECTOR_QUERY);
    let job = dispatch(1, VECTOR_SEARCH, &[(query, Some(StorageLocator::Inline { bytes: ask(vec![0.9, 0.1, 0.0]) }))]);
    let mut heard = stand(&socket, None, vec![job]).await;
    let results = work(&socket, store(&dir.path().join("db")).await, 1).await;
    assert!(results[0].is_ok(), "{results:?}");
    let Heard::Commit { outputs, stage_range } = next(&mut heard).await else { panic!("expected a commit") };
    assert_eq!(stage_range, Some((QUERY_STREAM.0, (1_000, 1_500))));
    let [(header, StorageLocator::Inline { bytes })] = outputs.as_slice() else { panic!("one inline result") };
    assert_eq!(header.stream(), STREAMS.vector);
    assert_eq!(header.shape_name(), VECTOR_SEARCH);
    assert_eq!((header.start_ns(), header.end_ns()), (1_000, 1_500));
    assert_eq!(header.parents(), vec![v7(9)]);
    // The query's subject, then the hits' entities: forgetting entity 40 reaches this result.
    assert_eq!(header.subjects(), vec![v7(30), v7(40)]);
    assert_ne!(header.id(), v7(9));
    let hits = decode_hits(bytes).unwrap();
    assert_eq!(hits.iter().map(|h| h.key.payload_id).collect::<Vec<_>>(), vec![uid(1), uid(2)]);
}

#[tokio::test]
async fn journal_full_text_search_finds_rendered_names_from_tokenized_entries() {
    let dir = tempfile::tempdir().unwrap();
    let store = VectorStore::open(&dir.path().join("db")).await.unwrap();
    let key = Key { payload_id: uid(3), entity_id: Some(uid(30)), parent_id: None, start_ns: 100, end_ns: 150 };
    let text = text_of(br#"{"text":"[S12] planted tomatoes.","search":"Dana planted tomatoes.","level":30}"#)
        .unwrap()
        .unwrap();
    index_text(&store, key, &text).await.unwrap();
    let filter = Filter { during: Some((100, 200)), subjects: vec![uid(30)], labelled: None };
    let hits = store.text_search("Dana", 10, &filter).await.unwrap();
    assert_eq!(hits.iter().map(|hit| hit.key.payload_id).collect::<Vec<_>>(), vec![uid(3)]);
    assert!(store.text_search("S12", 10, &filter).await.unwrap().is_empty());
}

#[tokio::test]
async fn a_text_search_writes_to_the_text_stream() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("s.sock");
    let s = VectorStore::open(&dir.path().join("db")).await.unwrap();
    let key = Key { payload_id: uid(3), entity_id: None, parent_id: None, start_ns: 5, end_ns: 5 };
    index_text(&s, key, "Dana boiled the kettle").await.unwrap();
    let q = continuum_vector::jobs::TextQuery { text: "kettle".into(), k: 3, filter: Filter::default() };
    let query = query_header(9, continuum_vector::jobs::TEXT_QUERY);
    let job = dispatch(1, TEXT_SEARCH, &[(query, Some(StorageLocator::Inline { bytes: q.encode().unwrap() }))]);
    let mut heard = stand(&socket, None, vec![job]).await;
    assert!(work(&socket, s, 1).await[0].is_ok());
    let Heard::Commit { outputs, .. } = next(&mut heard).await else { panic!("expected a commit") };
    assert_eq!(outputs[0].0.stream(), STREAMS.text);
    let StorageLocator::Inline { bytes } = &outputs[0].1 else { panic!("inline") };
    assert_eq!(decode_hits(bytes).unwrap().iter().map(|h| h.key.payload_id).collect::<Vec<_>>(), vec![uid(3)]);
}

#[tokio::test]
async fn a_query_in_shared_memory_is_read_from_its_slot() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("s.sock");
    let a = arena();
    let bytes = ask(vec![0.0, 0.9, 0.1]);
    let mut slot = a.0.claim(0, 1).unwrap();
    slot.as_mut_slice()[..bytes.len()].copy_from_slice(&bytes);
    let offset = slot.offset() as u64;
    let handle = slot.publish(bytes.len()).unwrap();
    let at = StorageLocator::Shm {
        arena: a.0.name().to_owned(),
        arena_generation: a.0.generation(),
        slab_class: 0,
        slot: handle.index,
        slot_generation: handle.generation,
        offset,
        length: bytes.len() as u64,
    };
    let job = dispatch(1, VECTOR_SEARCH, &[(query_header(9, VECTOR_QUERY), Some(at))]);
    let mut heard = stand(&socket, Some(&a.0), vec![job]).await;
    assert!(work(&socket, store(&dir.path().join("db")).await, 1).await[0].is_ok());
    let Heard::Commit { outputs, .. } = next(&mut heard).await else { panic!("expected a commit") };
    let StorageLocator::Inline { bytes } = &outputs[0].1 else { panic!("inline") };
    // The vector came from the slot: [0, 0.9, 0.1] is nearest payload 2.
    assert_eq!(decode_hits(bytes).unwrap()[0].key.payload_id, uid(2));
}

#[tokio::test]
async fn a_job_it_cant_answer_is_turned_down_with_the_reason() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("s.sock");
    let good = || Some(StorageLocator::Inline { bytes: ask(vec![1.0, 0.0, 0.0]) });
    let jobs = vec![
        dispatch(11, "pose", &[(query_header(9, VECTOR_QUERY), good())]),
        dispatch(12, VECTOR_SEARCH, &[]),
        dispatch(13, VECTOR_SEARCH, &[(query_header(9, "caption"), good())]),
        dispatch(14, VECTOR_SEARCH, &[(query_header(9, VECTOR_QUERY), None)]),
        dispatch(
            15,
            VECTOR_SEARCH,
            &[(query_header(9, VECTOR_QUERY), Some(StorageLocator::Inline { bytes: b"x".to_vec() }))],
        ),
        // A query in a space of another size is refused by the store, never answered with nonsense.
        dispatch(
            16,
            VECTOR_SEARCH,
            &[(query_header(9, VECTOR_QUERY), Some(StorageLocator::Inline { bytes: ask(vec![1.0]) }))],
        ),
    ];
    let mut heard = stand(&socket, None, jobs).await;
    let results = work(&socket, store(&dir.path().join("db")).await, 6).await;
    for (n, (r, want)) in
        results.iter().zip(["pose", "one query", "caption", "no bytes", "schema", "dimension"]).enumerate()
    {
        let Err(WorkerError::Rejected { job, reason }) = r else { panic!("job {n}: {r:?}") };
        assert_eq!(*job, v7(11 + n as u8));
        assert!(reason.contains(want), "{reason}");
        let Heard::Rejected(id, said) = next(&mut heard).await else { panic!("expected a rejection") };
        assert_eq!((id, said.as_str()), (*job, reason.as_str()));
    }
}

#[tokio::test]
async fn a_purge_notice_removes_the_rows_then_acks() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("s.sock");
    let s = store(&dir.path().join("db")).await;
    let mut heard = stand(&socket, None, vec![purge_notice(50, &[1, 77])]).await;
    assert!(work(&socket, s, 1).await[0].is_ok());
    let probe = VectorStore::open(&dir.path().join("db")).await.unwrap();
    let Heard::PurgeAck(id) = next(&mut heard).await else { panic!("expected an ack") };
    assert_eq!(id, v7(50));
    assert_eq!(probe.rows_for(uid(1)).await.unwrap(), 0);
    assert_eq!(probe.rows_for(uid(2)).await.unwrap(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn embeddings_are_filed_from_a_durable_subscription_that_resumes_after_them() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("b.sock");
    let server = Server::bind(Config::new(socket.clone(), Duration::from_secs(30))).await.unwrap();
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(server.run_until(async {
        let _ = stopped.await;
    }));
    let producer = Client::connect(&socket, ClientOptions::new("faces")).await.unwrap();
    let spec = StreamSpec {
        name: "face.embedding".into(),
        kind: fb::StreamKind::Derived,
        format: fb::PayloadFormat::RawBytes,
        shape_name: "embedding/arcface/r100".into(),
        shape_version: 1,
        nominal_rate_hz: 1.0,
        storage: StreamStorage::Slab { class: 0 },
        anchor: None,
        layer: fb::DomainLayer::Layer2Decision,
    };
    let stream = StreamId(producer.register_stream(&spec).await.unwrap().stream_id);
    // The utterances the embeddings come from: their parents, which the broker must know first.
    let utterances = StreamSpec { name: "speech.utterance".into(), shape_name: "utterance".into(), ..spec.clone() };
    let utterances = StreamId(producer.register_stream(&utterances).await.unwrap().stream_id);
    for n in [60, 61] {
        let h = HeaderBuf::build(&HeaderSpec::raw(v7(n), utterances, u64::from(n), u64::from(n) + 1)).unwrap();
        producer.ingest_acked(&h, Some(&StorageLocator::Inline { bytes: b"hi".to_vec() })).await.unwrap();
    }
    let ingest = |n: u8, shape: &str, bytes: Vec<u8>| {
        let mut h = HeaderSpec::raw(v7(n), stream, u64::from(n) * 100, u64::from(n) * 100 + 50);
        h.shape_name = shape.to_owned();
        h.subjects = vec![v7(30)];
        h.parents = vec![v7(60), v7(61)];
        let header = HeaderBuf::build(&h).unwrap();
        let producer = &producer;
        async move { producer.ingest_acked(&header, Some(&StorageLocator::Inline { bytes })).await.unwrap() }
    };
    ingest(1, "embedding/arcface/r100", f32s(&[1.0, 0.0, 0.0])).await;
    ingest(2, "frame", f32s(&[1.0, 0.0, 0.0])).await; // not an embedding
    ingest(3, "embedding/arcface/r100", vec![1, 2, 3]).await; // not whole floats
    // Its commit sequence is its position in a subscription.
    let last = ingest(4, "embedding/arcface/r100", f32s(&[0.0, 1.0, 0.0])).await;

    // A first run registers and goes; the next (a new session) gets the same streams.
    let first = Client::connect(&socket, ClientOptions::new("vector, first run")).await.unwrap();
    let streams = register(&first, "test").await.unwrap();
    assert_ne!(streams.vector, streams.text);
    first.goodbye("restarting").await.unwrap();
    let client = Arc::new(Client::connect(&socket, ClientOptions::new("vector")).await.unwrap());
    let mut again = Err(WorkerError::Client(String::new()));
    for _ in 0..250 {
        // The broker drops the first session's registration when it sees the goodbye.
        again = register(&client, "test").await;
        if again.is_ok() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(again.unwrap(), streams);
    let db = dir.path().join("db");
    let worker = SearchWorker::new(client.clone(), VectorStore::open(&db).await.unwrap(), streams);
    let mut durable = SubscribeSpec::new(["face.embedding"], Start::Saved);
    durable.durable_name = Some(INDEX_SUBSCRIPTION.into());
    durable.pinned = true;
    let mut sub = client.subscribe(&durable).await.unwrap();
    let probe = VectorStore::open(&db).await.unwrap();
    let mut skipped = Vec::new();
    {
        let indexing = worker.index(&mut sub, |position, why| skipped.push((position, why)));
        tokio::pin!(indexing);
        let filed = async {
            for _ in 0..250 {
                if probe.rows_for(uid(4)).await.unwrap() == 1 {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            panic!("payload 4 was never filed");
        };
        tokio::select! {
            r = &mut indexing => panic!("indexing ended early: {r:?}"),
            () = filed => {}
        }
    }
    assert_eq!(probe.rows_for(uid(1)).await.unwrap(), 1);
    assert_eq!(probe.rows_for(uid(2)).await.unwrap(), 0);
    assert_eq!(probe.rows_for(uid(3)).await.unwrap(), 0);
    assert_eq!(skipped.len(), 2, "{skipped:?}");
    assert!(skipped[0].1.contains("frame") && !skipped[1].1.is_empty(), "{skipped:?}");
    // The entity is the header's first subject: a filter on it finds payload 1.
    let q = VectorQuery {
        model: "arcface".into(),
        version: "r100".into(),
        k: 5,
        vector: vec![1.0, 0.0, 0.0],
        filter: Filter { during: None, subjects: vec![uid(30)], labelled: None },
    };
    let hits =
        decode_hits(&continuum_vector::jobs::answer_vector_query(&probe, &q.encode().unwrap()).await.unwrap()).unwrap();
    assert_eq!(hits[0].key.payload_id, uid(1));
    assert_eq!(hits[0].key.parent_id, Some(uid(60)), "hits carry the payload's first parent");
    sub.unsubscribe().await.unwrap();

    // Everything was acknowledged, the skipped ones too: a restart starts after payload 4. The cursor is saved
    // through the broker's writer, so it may take a moment to show.
    let mut resumed = 0;
    for _ in 0..250 {
        let s = client.subscribe(&durable).await.unwrap();
        resumed = s.start_position();
        s.unsubscribe().await.unwrap();
        if resumed == last {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(resumed, last, "resumes after payload 4");
    let _ = stop.send(());
}

/// A real broker on a socket in `dir`; dropping the sender stops it.
async fn real_broker(dir: &std::path::Path) -> (std::path::PathBuf, tokio::sync::oneshot::Sender<()>) {
    let socket = dir.join("b.sock");
    let server = Server::bind(Config::new(socket.clone(), Duration::from_secs(30))).await.unwrap();
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(server.run_until(async {
        let _ = stopped.await;
    }));
    (socket, stop)
}

/// A consumer of searches through the interim live path: it registers the query stream, follows the results, and
/// commits a query, then waits for the result whose parent is its query.
struct Consumer {
    client: Client,
    queries: StreamId,
    results: continuum_client::Subscription,
}

impl Consumer {
    async fn connect(socket: &std::path::Path) -> Self {
        let client = Client::connect(socket, ClientOptions::new("consumer")).await.unwrap();
        let queries = StreamId(client.register_stream(&search_stream(VECTOR_QUERY)).await.unwrap().stream_id);
        let mut follow = SubscribeSpec::new([VECTOR_SEARCH], Start::Live);
        follow.pinned = true;
        let results = client.subscribe(&follow).await.unwrap();
        Self { client, queries, results }
    }

    async fn nearest(&mut self, n: u64, seq: u32, vector: Vec<f32>, k: u32) -> Vec<continuum_vector::Hit> {
        let id = Uuid(uuid::Uuid::now_v7().into_bytes());
        let mut h = HeaderSpec::raw(id, self.queries, n, n);
        h.seq = seq;
        h.shape_name = VECTOR_QUERY.into();
        h.format = PayloadFormat::ArrowRecordBatch;
        let q = VectorQuery { model: "arcface".into(), version: "r100".into(), k, vector, filter: Filter::default() };
        let at = StorageLocator::Inline { bytes: q.encode().unwrap() };
        self.client.ingest_acked(&HeaderBuf::build(&h).unwrap(), Some(&at)).await.unwrap();
        loop {
            let event = tokio::time::timeout(Duration::from_secs(5), self.results.next()).await.unwrap().unwrap();
            if let continuum_client::SubEvent::Event { item, .. } = event
                && item.header.parents() == vec![id]
            {
                let Some(StorageLocator::Inline { bytes }) = item.locator else { panic!("an inline result") };
                return decode_hits(&bytes).unwrap();
            }
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_query_committed_live_is_answered_by_the_worker() {
    let dir = tempfile::tempdir().unwrap();
    let (socket, stop) = real_broker(dir.path()).await;
    let client = Arc::new(Client::connect(&socket, ClientOptions::new("vector")).await.unwrap());
    let streams = register(&client, "test").await.unwrap();
    let worker = Arc::new(SearchWorker::new(client.clone(), store(&dir.path().join("db")).await, streams));
    let mut live = SubscribeSpec::new([VECTOR_QUERY, continuum_vector::jobs::TEXT_QUERY], Start::Live);
    live.pinned = true;
    let mut queries = client.subscribe(&live).await.unwrap();
    let serving = tokio::spawn({
        let worker = worker.clone();
        async move { worker.serve(&mut queries, |p, why| panic!("query at {p} failed: {why}")).await }
    });
    let mut consumer = Consumer::connect(&socket).await;
    let hits = consumer.nearest(1_000, 0, vec![0.0, 0.9, 0.1], 5).await;
    assert_eq!(hits.iter().map(|h| h.key.payload_id).collect::<Vec<_>>(), vec![uid(2), uid(1)]);
    // A second query at the same instant (two faces in one frame) differs by seq, and so must its answer: it's
    // answered too, and the worker goes on serving.
    let hits = consumer.nearest(1_000, 1, vec![0.9, 0.1, 0.0], 5).await;
    assert_eq!(hits[0].key.payload_id, uid(1));
    let hits = consumer.nearest(2_000, 0, vec![0.0, 0.9, 0.1], 1).await;
    assert_eq!(hits[0].key.payload_id, uid(2));
    serving.abort();
    let _ = stop.send(());
}

/// The p95 of a live query's round trip (commit the query, the worker answers, the result arrives) with 50,000
/// 192-float rows, as the audio port asked (under 50 ms). Not part of the suite: run it by name with `--ignored`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "a one-off measurement: about a minute"]
async fn the_live_round_trip_p95_at_50k_rows() {
    const ROWS: usize = 50_000;
    const DIM: usize = 192;
    let dir = tempfile::tempdir().unwrap();
    let (socket, stop) = real_broker(dir.path()).await;
    let client = Arc::new(Client::connect(&socket, ClientOptions::new("vector")).await.unwrap());
    let streams = register(&client, "test").await.unwrap();
    let store = VectorStore::open(&dir.path().join("db")).await.unwrap();
    let space = continuum_vector::EmbeddingSpace { model: "arcface".into(), version: "r100".into(), dim: DIM };
    // A fixed pseudo-random spread of vectors (an LCG, so the run is repeatable).
    let mut seed = 7u64;
    let mut next = move || {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    };
    for chunk in 0..(ROWS / 5_000) {
        let rows: Vec<continuum_vector::EmbeddingRow> = (0..5_000)
            .map(|i| {
                let n = (chunk * 5_000 + i) as u128;
                continuum_vector::EmbeddingRow {
                    key: Key {
                        payload_id: uuid::Uuid::from_u128(n + 1),
                        entity_id: None,
                        parent_id: None,
                        start_ns: 0,
                        end_ns: 0,
                    },
                    vector: (0..DIM).map(|_| next()).collect(),
                }
            })
            .collect();
        store.insert_embeddings(&space, &rows).await.unwrap();
    }
    let worker = Arc::new(SearchWorker::new(client.clone(), store, streams));
    let mut live = SubscribeSpec::new([VECTOR_QUERY], Start::Live);
    live.pinned = true;
    let mut queries = client.subscribe(&live).await.unwrap();
    let serving = tokio::spawn({
        let worker = worker.clone();
        async move { worker.serve(&mut queries, |p, why| panic!("query at {p} failed: {why}")).await }
    });
    let mut consumer = Consumer::connect(&socket).await;
    let mut took = Vec::new();
    for n in 0..220u64 {
        let vector: Vec<f32> = (0..DIM).map(|_| next()).collect();
        let t = std::time::Instant::now();
        let hits = consumer.nearest(10_000 + n, 0, vector, 20).await;
        if n >= 20 {
            took.push(t.elapsed()); // the first 20 warm up
        }
        assert_eq!(hits.len(), 20);
    }
    took.sort();
    let p = |q: f64| took[((took.len() as f64 * q) as usize).min(took.len() - 1)];
    eprintln!("live round trip over {ROWS} rows of {DIM}: p50 {:?}, p95 {:?}, max {:?}", p(0.5), p(0.95), p(1.0));
    assert!(p(0.95) < Duration::from_millis(50), "p95 {:?}", p(0.95));
    serving.abort();
    let _ = stop.send(());
}
