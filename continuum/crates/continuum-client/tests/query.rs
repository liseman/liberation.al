//! Time queries against the real broker: what a stream holds over a range, with inline bytes, and paging past a
//! per-result limit without losing or repeating items, including several that start together at a page edge.

use std::time::Duration;

use continuum_broker::server::{Config, Server};
use continuum_client::{Client, ClientOptions, QuerySpec, Start, StreamSpec, StreamStorage, SubEvent, SubscribeSpec};
use continuum_core::schema::continuum as fb;
use continuum_core::{HeaderBuf, HeaderSpec, StorageLocator, StreamId, Uuid};
use tokio::sync::oneshot;

struct Broker {
    _dir: tempfile::TempDir,
    socket: std::path::PathBuf,
    stop: Option<oneshot::Sender<()>>,
}

impl Drop for Broker {
    fn drop(&mut self) {
        let _ = self.stop.take().map(|s| s.send(()));
    }
}

async fn broker() -> Broker {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("b.sock");
    let server = Server::bind(Config::new(socket.clone(), Duration::from_secs(30))).await.unwrap();
    let (stop, stopped) = oneshot::channel::<()>();
    tokio::spawn(server.run_until(async {
        let _ = stopped.await;
    }));
    Broker { _dir: dir, socket, stop: Some(stop) }
}

fn id_of(n: u8) -> Uuid {
    let mut id = [0u8; 16];
    id[6] = 0x70;
    id[8] = 0x80;
    id[15] = n;
    Uuid(id)
}

/// A producer with "events" (derived, text) registered and payloads `ns` ingested at the given starts, each 10 ns
/// long, inline; returns once they're committed (a live subscription has seen every one).
async fn with_events(b: &Broker, payloads: &[(u8, u64)]) -> Client {
    let p = Client::connect(&b.socket, ClientOptions::new("producer")).await.unwrap();
    let spec = StreamSpec {
        name: "events".into(),
        kind: fb::StreamKind::Derived,
        format: fb::PayloadFormat::Text,
        shape_name: "event".into(),
        shape_version: 1,
        nominal_rate_hz: 0.0,
        storage: StreamStorage::Slab { class: 0 },
        anchor: None,
        layer: fb::DomainLayer::Layer2Decision,
    };
    let stream = p.register_stream(&spec).await.unwrap().stream_id;
    let mut live = p.subscribe(&SubscribeSpec::new(["events"], Start::Live)).await.unwrap();
    for &(n, start) in payloads {
        // Payloads are keyed by stream, start, and seq: ones that start together need their own seq.
        let mut h = HeaderSpec::raw(id_of(n), StreamId(stream), start, start + 10);
        h.seq = u32::from(n);
        let header = HeaderBuf::build(&h).unwrap();
        p.ingest(&header, Some(&StorageLocator::Inline { bytes: vec![n] })).unwrap();
    }
    for _ in payloads {
        let got = tokio::time::timeout(Duration::from_secs(5), live.next()).await.unwrap().unwrap();
        assert!(matches!(got, SubEvent::Event { .. }), "{got:?}");
    }
    p
}

fn numbers(items: &[continuum_client::Item]) -> Vec<u8> {
    items.iter().map(|it| it.header.id().0[15]).collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_query_returns_what_the_range_holds_with_inline_bytes() {
    let b = broker().await;
    let c = with_events(&b, &[(1, 100), (2, 200), (3, 300), (4, 400)]).await;
    let r = tokio::time::timeout(Duration::from_secs(10), c.query(&QuerySpec::hit_only("events", (150, 350))))
        .await
        .expect("answered")
        .unwrap();
    assert_eq!(numbers(&r.items), vec![2, 3]);
    assert_eq!(r.items[0].locator, Some(StorageLocator::Inline { bytes: vec![2] }));
    assert!(r.tail.is_none(), "not a ring stream");

    let err = c.query(&QuerySpec::hit_only("nope", (0, 1))).await.unwrap_err();
    assert!(err.broker_code().is_some(), "an unknown stream is refused: {err}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn query_all_pages_past_the_limit_without_losing_or_repeating_items() {
    let b = broker().await;
    // Three payloads share a start, so a page of two cuts through them.
    let c = with_events(&b, &[(1, 100), (2, 200), (3, 200), (4, 200), (5, 300), (6, 400)]).await;
    let mut spec = QuerySpec::hit_only("events", (0, 1_000));
    spec.max_items = 2;
    let all = tokio::time::timeout(Duration::from_secs(10), c.query_all(&spec)).await.expect("query_all ends").unwrap();
    assert_eq!(numbers(&all.items), vec![1, 2, 3, 4, 5, 6], "every item once, in time order");
    // A range that keeps only some: the cursor still walks the whole stream and stops.
    let mut some = QuerySpec::hit_only("events", (150, 350));
    some.max_items = 1;
    let r = tokio::time::timeout(Duration::from_secs(10), c.query_all(&some)).await.expect("query_all ends").unwrap();
    assert_eq!(numbers(&r.items), vec![2, 3, 4, 5]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_forgotten_names_hash_round_trips_through_the_client() {
    let b = broker().await;
    let c = Client::connect(&b.socket, ClientOptions::new("people")).await.unwrap();
    assert!(!c.is_forgotten([9; 32]).await.unwrap());
    c.remember_forgotten([9; 32]).await.unwrap();
    assert!(c.is_forgotten([9; 32]).await.unwrap());
    assert!(!c.is_forgotten([10; 32]).await.unwrap());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn forgotten_hash_enumeration_is_bounded_and_exclusive() {
    let b = broker().await;
    let c = Client::connect(&b.socket, ClientOptions::new("forgotten-export")).await.unwrap();
    assert!(c.list_forgotten(None, 2).await.unwrap().hashes.is_empty());
    for hash in [[255; 32], [0; 32], [2; 32], [1; 32]] {
        c.remember_forgotten(hash).await.unwrap();
    }
    let first = c.list_forgotten(None, 2).await.unwrap();
    assert_eq!(first.hashes, vec![[0; 32], [1; 32]]);
    assert_eq!(first.next, Some([1; 32]));
    let last = c.list_forgotten(first.next, 2).await.unwrap();
    assert_eq!(last.hashes, vec![[2; 32], [255; 32]]);
    assert_eq!(last.next, None);
    assert!(c.list_forgotten(last.hashes.last().copied(), 2).await.unwrap().hashes.is_empty());
    assert!(c.list_forgotten(None, 0).await.is_err());
    assert!(c.list_forgotten(None, 1025).await.is_err());
}
