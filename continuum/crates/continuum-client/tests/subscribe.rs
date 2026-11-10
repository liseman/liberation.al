//! Subscriptions against the real broker: a live subscription gets each commit in order, a time start catches up and
//! then goes live (with `CaughtUp` between), a durable one resumes after its acknowledged position, the filters keep
//! only what they match, events never reach `next_event`, and a dropped subscription stops its events.

mod common;

use std::time::Duration;

use continuum_broker::server::{Config, Server};
use continuum_client::{
    Client, ClientOptions, Start, StreamSpec, StreamStorage, SubEvent, SubscribeSpec, Subscription,
};
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

/// A producer with "cam0" registered; returns it and the stream's id.
async fn producer(b: &Broker) -> (Client, u32) {
    let c = Client::connect(&b.socket, ClientOptions::new("producer")).await.unwrap();
    let spec = StreamSpec {
        name: "cam0".into(),
        kind: fb::StreamKind::Sensor,
        format: fb::PayloadFormat::RawBytes,
        shape_name: "frame".into(),
        shape_version: 1,
        nominal_rate_hz: 30.0,
        storage: StreamStorage::Slab { class: 0 },
        anchor: None,
        layer: fb::DomainLayer::Layer1Raw,
    };
    let id = c.register_stream(&spec).await.unwrap().stream_id;
    (c, id)
}

fn id_of(n: u8) -> Uuid {
    let mut id = [0u8; 16];
    id[6] = 0x70;
    id[8] = 0x80;
    id[15] = n;
    Uuid(id)
}

fn subject(n: u8) -> Uuid {
    let mut id = [0u8; 16];
    id[6] = 0x70;
    id[8] = 0x80;
    id[14] = 0xEE;
    id[15] = n;
    Uuid(id)
}

/// Ingests payload `n` (inline, spanning `start` for 100 ns) with these subjects and this shape.
fn ingest(p: &Client, stream: u32, n: u8, start: u64, subjects: &[u8], shape: &str) {
    let mut spec = HeaderSpec::raw(id_of(n), StreamId(stream), start, start + 100);
    spec.subjects = subjects.iter().map(|&s| subject(s)).collect();
    spec.shape_name = shape.to_owned();
    let header = HeaderBuf::build(&spec).unwrap();
    p.ingest(&header, Some(&StorageLocator::Inline { bytes: vec![n] })).unwrap();
}

async fn next(s: &mut Subscription) -> SubEvent {
    tokio::time::timeout(Duration::from_secs(5), s.next()).await.expect("an event in time").expect("still open")
}

/// The next `n` events: (payload number, position).
async fn events(s: &mut Subscription, n: usize) -> Vec<(u8, u64)> {
    let mut got = Vec::new();
    while got.len() < n {
        match next(s).await {
            SubEvent::Event { position, item } => got.push((item.header.id().0[15], position)),
            other => panic!("expected an event, got {other:?}"),
        }
    }
    got
}

/// The next `n` events, skipping a `CaughtUp` (a catch-up over an empty history ends at once, so it can come before
/// or between them).
async fn events_past_catch_up(s: &mut Subscription, n: usize) -> Vec<(u8, u64)> {
    let mut got = Vec::new();
    while got.len() < n {
        match next(s).await {
            SubEvent::Event { position, item } => got.push((item.header.id().0[15], position)),
            SubEvent::CaughtUp { .. } => {}
            other @ SubEvent::Undecodable { .. } => panic!("expected an event, got {other:?}"),
        }
    }
    got
}

/// Nothing more arrives within a short wait.
async fn quiet(s: &mut Subscription) {
    let got = tokio::time::timeout(Duration::from_millis(200), s.next()).await;
    assert!(got.is_err(), "expected nothing, got {got:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_live_subscription_gets_each_commit_in_order_and_nothing_reaches_next_event() {
    let b = broker().await;
    let (p, cam) = producer(&b).await;
    let c = Client::connect(&b.socket, ClientOptions::new("consumer")).await.unwrap();
    let mut s = c.subscribe(&SubscribeSpec::new(["cam0"], Start::Live)).await.unwrap();
    assert!(s.live_at_start());
    for (n, start) in [(1u8, 100u64), (2, 200), (3, 300)] {
        ingest(&p, cam, n, start, &[], "frame");
    }
    let got = events(&mut s, 3).await;
    assert_eq!(got.iter().map(|g| g.0).collect::<Vec<_>>(), vec![1, 2, 3]);
    assert!(got.windows(2).all(|w| w[0].1 < w[1].1), "positions increase: {got:?}");
    let pending = tokio::time::timeout(Duration::from_millis(100), c.next_event()).await;
    assert!(pending.is_err(), "subscription events are routed to the subscription, not next_event");
    s.unsubscribe().await.unwrap();
    assert_eq!(c.pending_requests(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_time_start_catches_up_then_goes_live() {
    let b = broker().await;
    let (p, cam) = producer(&b).await;
    let c = Client::connect(&b.socket, ClientOptions::new("consumer")).await.unwrap();
    // A live subscription first, to know the three are committed before the time start reads history.
    let mut live = c.subscribe(&SubscribeSpec::new(["cam0"], Start::Live)).await.unwrap();
    for (n, start) in [(1u8, 100u64), (2, 200), (3, 300)] {
        ingest(&p, cam, n, start, &[], "frame");
    }
    events(&mut live, 3).await;
    drop(live);

    // From 250: payload 2 (200 to 300) reaches into it, payload 1 doesn't.
    let mut s = c.subscribe(&SubscribeSpec::new(["cam0"], Start::Time(250))).await.unwrap();
    assert!(!s.live_at_start());
    let got = events(&mut s, 2).await;
    assert_eq!(got.iter().map(|g| g.0).collect::<Vec<_>>(), vec![2, 3]);
    let SubEvent::CaughtUp { position } = next(&mut s).await else { panic!("expected CaughtUp") };
    assert!(position >= got[1].1);
    ingest(&p, cam, 4, 400, &[], "frame");
    let after = events(&mut s, 1).await;
    assert_eq!(after[0].0, 4);
    assert!(after[0].1 > position, "live events follow the catch-up");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_durable_subscription_resumes_after_its_acknowledged_position() {
    let b = broker().await;
    let (p, cam) = producer(&b).await;
    let c = Client::connect(&b.socket, ClientOptions::new("consumer")).await.unwrap();
    let mut durable = SubscribeSpec::new(["cam0"], Start::Saved);
    durable.durable_name = Some("journal".into());
    let mut s = c.subscribe(&durable).await.unwrap();
    assert_eq!(s.start_position(), 0, "nothing saved yet");
    for (n, start) in [(1u8, 100u64), (2, 200), (3, 300)] {
        ingest(&p, cam, n, start, &[], "frame");
    }
    let got = events_past_catch_up(&mut s, 3).await;
    // Acknowledged through the second one, from a separate handle (as a reading task and an acking task would).
    s.handle().ack(got[1].1).unwrap();
    s.unsubscribe().await.unwrap();

    // The cursor is saved through the writer, so a restarted consumer may need a moment to see it.
    let d = Client::connect(&b.socket, ClientOptions::new("restarted")).await.unwrap();
    let mut resumed = None;
    for _ in 0..250 {
        let s = d.subscribe(&durable).await.unwrap();
        if s.start_position() == got[1].1 {
            resumed = Some(s);
            break;
        }
        s.unsubscribe().await.unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let mut s = resumed.expect("the acknowledged position was saved");
    assert_eq!(events(&mut s, 1).await[0].0, 3, "resumes after the acknowledged event");
    assert!(matches!(next(&mut s).await, SubEvent::CaughtUp { .. }));

    // A Saved start without a durable name is refused before anything is sent.
    let err = d.subscribe(&SubscribeSpec::new(["cam0"], Start::Saved)).await.unwrap_err();
    assert!(matches!(err, continuum_client::Error::Protocol(_)), "{err}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn filters_keep_only_their_subjects_and_shape_and_pinned_events_carry_a_locator() {
    let b = broker().await;
    let (p, cam) = producer(&b).await;
    let c = Client::connect(&b.socket, ClientOptions::new("consumer")).await.unwrap();
    let mut about = SubscribeSpec::new(["cam0"], Start::Live);
    about.subjects = vec![subject(7)];
    about.pinned = true;
    let mut about = c.subscribe(&about).await.unwrap();
    let mut shaped = SubscribeSpec::new(["cam0"], Start::Live);
    shaped.shape_name = Some("pose".into());
    let mut shaped = c.subscribe(&shaped).await.unwrap();

    ingest(&p, cam, 1, 100, &[8], "frame");
    ingest(&p, cam, 2, 200, &[7, 8], "frame");
    ingest(&p, cam, 3, 300, &[], "pose");
    let SubEvent::Event { item, .. } = next(&mut about).await else { panic!("expected an event") };
    assert_eq!(item.header.id(), id_of(2), "only the payload about subject 7");
    assert_eq!(item.locator, Some(StorageLocator::Inline { bytes: vec![2] }), "pinned: the locator comes too");
    let SubEvent::Event { item, .. } = next(&mut shaped).await else { panic!("expected an event") };
    assert_eq!(item.header.id(), id_of(3), "only the pose");
    assert_eq!(item.locator, None, "not pinned: the header only");
    quiet(&mut about).await;
    quiet(&mut shaped).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropping_one_subscription_leaves_another_running_and_an_unknown_stream_is_refused() {
    let b = broker().await;
    let (p, cam) = producer(&b).await;
    let c = Client::connect(&b.socket, ClientOptions::new("consumer")).await.unwrap();
    let dropped = c.subscribe(&SubscribeSpec::new(["cam0"], Start::Live)).await.unwrap();
    let mut kept = c.subscribe(&SubscribeSpec::new(["cam0"], Start::Live)).await.unwrap();
    assert_ne!(dropped.id(), kept.id());
    drop(dropped);
    ingest(&p, cam, 1, 100, &[], "frame");
    assert_eq!(events(&mut kept, 1).await[0].0, 1, "the other subscription carries on");

    let err = c.subscribe(&SubscribeSpec::new(["nope"], Start::Live)).await.unwrap_err();
    assert_eq!(err.broker_code(), Some(fb::ErrorCode::NotFound), "{err}");
    // The refused subscription's id isn't left routed: the client still works.
    c.heartbeat(None).await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_subscription_ends_when_the_connection_does() {
    let b = broker().await;
    let c = Client::connect(&b.socket, ClientOptions::new("consumer")).await.unwrap();
    producer(&b).await;
    let mut s = c.subscribe(&SubscribeSpec::new(["cam0"], Start::Live)).await.unwrap();
    drop(b);
    let end = tokio::time::timeout(Duration::from_secs(5), s.next()).await.expect("ends in time");
    assert_eq!(end, None);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_handle_unsubscribes_while_a_reader_waits_and_the_reader_sees_the_end() {
    let b = broker().await;
    let (p, cam) = producer(&b).await;
    let c = Client::connect(&b.socket, ClientOptions::new("consumer")).await.unwrap();
    let mut s = c.subscribe(&SubscribeSpec::new(["cam0"], Start::Live)).await.unwrap();
    let handle = s.handle();
    let reader = tokio::spawn(async move { s.next().await });
    tokio::time::sleep(Duration::from_millis(50)).await;
    handle.unsubscribe().await.unwrap();
    let end = tokio::time::timeout(Duration::from_secs(5), reader).await.expect("the reader ends").unwrap();
    assert_eq!(end, None);
    handle.unsubscribe().await.unwrap(); // already ended: nothing to do
    ingest(&p, cam, 1, 100, &[], "frame");
    c.heartbeat(None).await.unwrap();
    assert_eq!(c.pending_requests(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn closing_the_client_ends_every_subscription() {
    let b = broker().await;
    producer(&b).await;
    let c = Client::connect(&b.socket, ClientOptions::new("consumer")).await.unwrap();
    let mut first = c.subscribe(&SubscribeSpec::new(["cam0"], Start::Live)).await.unwrap();
    let mut second = c.subscribe(&SubscribeSpec::new(["cam0"], Start::Live)).await.unwrap();
    c.close("done").await.unwrap();
    for s in [&mut first, &mut second] {
        let end = tokio::time::timeout(Duration::from_secs(5), s.next()).await.expect("ends in time");
        assert_eq!(end, None);
    }
}

/// A stand-in broker that never answers Subscribe, and reports the subscription id of each Unsubscribe it hears.
#[tokio::test]
async fn a_subscribe_dropped_before_its_answer_is_unsubscribed() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("s.sock");
    let (heard_tx, mut heard) = tokio::sync::mpsc::unbounded_channel();
    let listener = tokio::net::UnixListener::bind(&socket).unwrap();
    tokio::spawn(async move {
        let (mut s, _) = listener.accept().await.unwrap();
        let mut buf = Vec::new();
        continuum_wire::read_frame(&mut s, &mut buf).await.unwrap();
        continuum_wire::write_frame(&mut s, &common::welcome(&[])).await.unwrap();
        while continuum_wire::read_frame(&mut s, &mut buf).await.is_ok() {
            let env = continuum_wire::read_envelope(&buf).unwrap();
            if let Some(u) = env.payload_as_unsubscribe() {
                let _ = heard_tx.send(u.subscription_id());
            }
        }
    });
    let c = Client::connect(&socket, ClientOptions::new("t")).await.unwrap();
    let spec = SubscribeSpec::new(["cam0"], Start::Live);
    let pending = tokio::time::timeout(Duration::from_millis(100), c.subscribe(&spec)).await;
    assert!(pending.is_err(), "the stand-in never answers");
    let id = tokio::time::timeout(Duration::from_secs(5), heard.recv()).await.unwrap().unwrap();
    assert_eq!(id, 1, "the dropped subscribe told the broker to drop subscription 1");
}

/// A catch-up goes at the subscriber's pace: two subscriptions catching up over a backlog far larger than the
/// client's buffers and the broker's outbound queue together, left unread for a while, still get every event and
/// keep the connection. Before rounds were paced by the queue's room, the broker sent rounds faster than they were
/// read, filled the queue, and closed the session (the vector worker's crash loop behind a busy face.embedding).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_long_catch_up_left_unread_for_a_while_keeps_the_connection_and_every_event() {
    const N: u16 = 6000;
    let b = broker().await;
    let (p, cam) = producer(&b).await;
    let mut options = ClientOptions::new("dedicated-reader");
    options.subscription_overflow = continuum_client::SubscriptionOverflow::Backpressure;
    let c = Client::connect(&b.socket, options).await.unwrap();
    // Every payload acknowledged, so the whole backlog is committed before anyone subscribes (a one-way ingest can
    // be turned away while the broker is busy). Many at once, from tasks sharing the producer.
    let p = std::sync::Arc::new(p);
    let mut ingests = tokio::task::JoinSet::new();
    for t in 0..24u16 {
        let p = std::sync::Arc::clone(&p);
        ingests.spawn(async move {
            for n in (t..N).step_by(24) {
                let mut id = id_of(0).0;
                id[13..15].copy_from_slice(&n.to_be_bytes());
                let start = 1_000 + u64::from(n) * 1_000;
                let header = HeaderBuf::build(&HeaderSpec::raw(Uuid(id), StreamId(cam), start, start + 100)).unwrap();
                p.ingest_acked(&header, Some(&StorageLocator::Inline { bytes: vec![1] })).await.unwrap();
            }
        });
    }
    while let Some(done) = ingests.join_next().await {
        done.unwrap();
    }

    // A mixed-use reader fails closed at its bound without saving the undelivered tail as handled.
    let mixed = Client::connect(&b.socket, ClientOptions::new("mixed-overflow")).await.unwrap();
    let mut durable = SubscribeSpec::new(["cam0"], Start::Saved);
    durable.durable_name = Some("overflow-replay".into());
    let mut failed = mixed.subscribe(&durable).await.unwrap();
    let diagnostic = tokio::time::timeout(Duration::from_secs(10), mixed.next_event()).await.unwrap().unwrap();
    assert!(diagnostic.envelope().payload_as_error().unwrap().message().unwrap().contains("bounded queue"));
    let mut admitted = 0;
    while let Some(event) = failed.next().await {
        assert!(matches!(event, SubEvent::Event { .. }));
        admitted += 1;
    }
    assert_eq!(admitted, 4096, "only the bounded ordered prefix was admitted");
    assert!(matches!(failed.ack(1), Err(continuum_client::Error::Closed)));
    drop(failed);
    drop(mixed);

    let mut subs = Vec::new();
    // The first reader resumes the failed durable session. No overflow event advanced its cursor.
    let resumed = c.subscribe(&durable).await.unwrap();
    assert_eq!(resumed.start_position(), 0);
    subs.push(resumed);
    let mut second_options = ClientOptions::new("second-dedicated-reader");
    second_options.subscription_overflow = continuum_client::SubscriptionOverflow::Backpressure;
    let second = Client::connect(&b.socket, second_options).await.unwrap();
    subs.push(second.subscribe(&SubscribeSpec::new(["cam0"], Start::Time(1))).await.unwrap());
    tokio::time::sleep(Duration::from_secs(1)).await; // nobody reads: the broker must wait, not overflow
    // Both read at once, as a follower of two streams does (a client stops reading its socket while any one
    // subscription's buffer is full).
    async fn drain(s: &mut Subscription) -> usize {
        let (mut got, mut last) = (0usize, 0u64);
        loop {
            match next(s).await {
                SubEvent::Event { position, .. } => {
                    assert!(position > last, "positions increase");
                    last = position;
                    got += 1;
                }
                SubEvent::CaughtUp { .. } => return got,
                other @ SubEvent::Undecodable { .. } => panic!("expected an event, got {other:?}"),
            }
        }
    }
    let [a, b2] = &mut subs[..] else { unreachable!() };
    let (x, y) = tokio::join!(drain(a), drain(b2));
    assert_eq!((x, y), (usize::from(N), usize::from(N)), "every event of the backlog, once each");
    // Still connected: a new commit reaches both.
    let header = HeaderBuf::build(&HeaderSpec::raw(id_of(9), StreamId(cam), 9_000_000, 9_000_100)).unwrap();
    p.ingest(&header, Some(&StorageLocator::Inline { bytes: vec![1] })).unwrap();
    for s in &mut subs {
        assert!(matches!(next(s).await, SubEvent::Event { .. }));
    }
}
