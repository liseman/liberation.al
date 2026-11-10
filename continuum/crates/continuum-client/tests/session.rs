//! The client against the real broker (sessions, heartbeats, errors), and against a stand-in broker for what the
//! real one doesn't send yet (arenas and their fds in `Welcome`).

use std::os::fd::AsFd;
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

mod common;

use common::*;
use continuum_broker::server::{Config, Server};
use continuum_client::{Client, ClientOptions, Error, StreamSpec, StreamStorage};
use continuum_core::schema::continuum as fb;
use continuum_core::{SCHEMA_MAJOR, SlotHandle};
use continuum_shm::Arena;
use tokio::sync::oneshot;

/// A real broker on a socket in a temp dir, shut down when dropped.
struct Broker {
    _dir: tempfile::TempDir,
    socket: PathBuf,
    stop: Option<oneshot::Sender<()>>,
}

impl Drop for Broker {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
    }
}

async fn broker(heartbeat_timeout: Duration) -> Broker {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("b.sock");
    let mut config = Config::new(socket.clone(), heartbeat_timeout);
    // A small Tier 0 arena, so each test doesn't map the profile's.
    config.tier0 = Some(vec![continuum_sched::SlotClass { size_bytes: 16_384, count: 4 }]);
    let server = Server::bind(config).await.unwrap();
    let (stop, stopped) = oneshot::channel::<()>();
    tokio::spawn(server.run_until(async {
        let _ = stopped.await;
    }));
    Broker { _dir: dir, socket, stop: Some(stop) }
}

fn options() -> ClientOptions {
    ClientOptions::new("client-test")
}

fn now_ns() -> u64 {
    u64::try_from(SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()).unwrap()
}

#[tokio::test]
async fn a_client_gets_a_session_from_the_broker() {
    let b = broker(Duration::from_secs(5)).await;
    let c = Client::connect(&b.socket, options()).await.unwrap();
    assert_ne!(c.session_id().0, [0; 16]);
    assert_eq!(c.session_id().unix_ms() / 1000, now_ns() / 1_000_000_000, "a UUIDv7 minted just now");
    assert_eq!(c.broker_schema().0, SCHEMA_MAJOR);
    // The broker's Tier 0 arena, mapped from the fd that came with Welcome.
    let [a] = c.arenas() else { panic!("one arena: {:?}", c.arenas()) };
    assert!(a.info.name.starts_with("/cnt."), "{}", a.info.name);
    assert_ne!(a.info.generation, 0);
    assert_eq!(a.arena.len() as u64, a.info.size_bytes);
    assert_eq!(a.arena.class(0).unwrap().slot_count, 4);
}

#[tokio::test]
async fn voice_state_is_seeded_on_connect_and_tracks_broadcasts_without_draining_events() {
    let b = broker(Duration::from_secs(5)).await;
    let voice = Client::connect(&b.socket, options()).await.unwrap();
    assert_eq!(voice.voice_turn(), Some(fb::VoiceTurnState::Idle));
    voice.set_system_state(fb::VoiceTurnState::Thinking).unwrap();
    tokio::time::timeout(Duration::from_secs(5), async {
        while voice.voice_turn() != Some(fb::VoiceTurnState::Thinking) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let background = Client::connect(&b.socket, options()).await.unwrap();
    assert_eq!(background.voice_turn(), Some(fb::VoiceTurnState::Thinking));
    for state in [fb::VoiceTurnState::Speaking, fb::VoiceTurnState::Idle] {
        voice.set_system_state(state).unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            while background.voice_turn() != Some(state) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }
    // Tracking must leave events available to clients that also need the stream.
    let event = tokio::time::timeout(Duration::from_secs(5), background.next_event()).await.unwrap().unwrap();
    assert_eq!(event.envelope().payload_as_system_state_changed().unwrap().voice_turn(), fb::VoiceTurnState::Speaking);
    let event = tokio::time::timeout(Duration::from_secs(5), background.next_event()).await.unwrap().unwrap();
    assert_eq!(event.envelope().payload_as_system_state_changed().unwrap().voice_turn(), fb::VoiceTurnState::Idle);
}

#[tokio::test]
async fn a_heartbeat_returns_the_brokers_clock() {
    let b = broker(Duration::from_secs(5)).await;
    let c = Client::connect(&b.socket, options()).await.unwrap();
    let before = now_ns();
    let broker_now = c.heartbeat(None).await.unwrap();
    assert!(broker_now >= before && broker_now <= now_ns());
}

#[tokio::test]
async fn background_heartbeats_keep_a_quiet_session_open() {
    let b = broker(Duration::from_secs(1)).await;
    let opts = ClientOptions { heartbeat_interval: Duration::from_millis(100), ..options() };
    let c = Client::connect(&b.socket, opts).await.unwrap();
    tokio::time::sleep(Duration::from_millis(2500)).await; // past two broker timeouts with nothing from the caller
    c.heartbeat(None).await.unwrap();
}

#[tokio::test]
async fn without_heartbeats_the_broker_closes_the_session() {
    let b = broker(Duration::from_millis(300)).await;
    let opts = ClientOptions { heartbeat_interval: Duration::from_secs(60), ..options() };
    let c = Client::connect(&b.socket, opts).await.unwrap();
    tokio::time::sleep(Duration::from_millis(1500)).await; // five broker timeouts
    assert!(matches!(c.heartbeat(None).await, Err(Error::Closed)), "the broker closed the quiet session");
    assert_eq!(c.voice_turn(), None, "a disconnected broker must not leave a stale busy turn");
}

#[tokio::test]
async fn concurrent_requests_each_get_their_own_reply() {
    let b = broker(Duration::from_secs(5)).await;
    let c = std::sync::Arc::new(Client::connect(&b.socket, options()).await.unwrap());
    let tasks: Vec<_> = (0..50)
        .map(|_| {
            let c = c.clone();
            tokio::spawn(async move { c.heartbeat(None).await })
        })
        .collect();
    for t in tasks {
        t.await.unwrap().unwrap();
    }
}

fn camera(name: &str, rate: f32) -> StreamSpec {
    StreamSpec {
        name: name.into(),
        kind: fb::StreamKind::Sensor,
        format: fb::PayloadFormat::RawBytes,
        shape_name: "NV12".into(),
        shape_version: 1,
        nominal_rate_hz: rate,
        storage: StreamStorage::Slab { class: 2 },
        anchor: None,
        layer: fb::DomainLayer::Layer1Raw,
    }
}

#[tokio::test]
async fn registering_a_stream_returns_its_id_and_the_same_id_again() {
    let b = broker(Duration::from_secs(5)).await;
    let c = Client::connect(&b.socket, options()).await.unwrap();
    let first = c.register_stream(&camera("cam0.video", 30.0)).await.unwrap();
    let other = c.register_stream(&camera("cam1.video", 30.0)).await.unwrap();
    assert_ne!(first.stream_id, other.stream_id);
    let again = c.register_stream(&camera("cam0.video", 30.0)).await.unwrap();
    assert_eq!(again.stream_id, first.stream_id, "the same name and info is the same stream");
}

#[tokio::test]
async fn a_request_the_broker_refuses_is_a_broker_error_and_the_session_survives() {
    let b = broker(Duration::from_secs(5)).await;
    let c = Client::connect(&b.socket, options()).await.unwrap();
    c.register_stream(&camera("cam0.video", 30.0)).await.unwrap();
    // Identity is the kind, layer, format, and shape; the rate may change.
    let jpeg = StreamSpec { shape_name: "MJPEG".into(), ..camera("cam0.video", 30.0) };
    let err = c.register_stream(&jpeg).await.unwrap_err();
    assert_eq!(err.broker_code(), Some(fb::ErrorCode::Conflict), "same name, a different shape: {err}");
    c.register_stream(&camera("cam0.video", 60.0)).await.unwrap();
    let empty = c.register_stream(&camera("", 30.0)).await.unwrap_err();
    assert_eq!(empty.broker_code(), Some(fb::ErrorCode::BadRequest), "{empty}");
    c.heartbeat(None).await.unwrap();
}

#[tokio::test]
async fn after_close_requests_fail_at_once() {
    let b = broker(Duration::from_secs(5)).await;
    let c = Client::connect(&b.socket, options()).await.unwrap();
    c.close("done").await.unwrap();
    let started = std::time::Instant::now();
    assert!(matches!(c.heartbeat(None).await, Err(Error::Closed)));
    assert!(started.elapsed() < Duration::from_secs(1), "failed at once, not after the request timeout");
    assert!(matches!(c.close("again").await, Err(Error::Closed)));
}

#[tokio::test]
async fn connecting_to_nothing_is_an_io_error() {
    let dir = tempfile::tempdir().unwrap();
    let err = Client::connect(dir.path().join("none.sock"), options()).await.unwrap_err();
    assert!(matches!(err, Error::Io(_)), "{err}");
}

#[tokio::test]
async fn arenas_from_welcome_are_mapped_and_read_in_place() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("s.sock");
    let shm = make_arena(3);
    let mut slot = shm.0.claim(0, 1).unwrap();
    slot.as_mut_slice()[..5].copy_from_slice(b"frame");
    let handle: SlotHandle = slot.publish(5).unwrap();
    let reply = welcome(&[("hot", 3, shm.0.len() as u64)]);
    let server = stand_in(&socket, reply, vec![dup(&shm.0)]).await;

    let c = Client::connect(&socket, options()).await.unwrap();
    assert_eq!(c.arenas().len(), 1);
    assert_eq!(c.arenas()[0].info.name, "hot");
    let arena = c.arena("hot").unwrap();
    let lease = arena.lease(handle, Duration::from_secs(5)).unwrap();
    let mut out = [0u8; 5];
    lease.copy_to(&mut out).unwrap();
    assert_eq!(&out, b"frame");
    assert_eq!(shm.0.pin_count(handle).unwrap(), 1, "the client's pin shows in the broker's mapping");
    drop(lease);
    assert_eq!(shm.0.pin_count(handle).unwrap(), 0);
    drop(c);
    server.await.unwrap();
}

#[tokio::test]
async fn a_welcome_whose_fds_dont_match_its_arenas_fails_the_connect() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("s.sock");
    let shm = make_arena(1);
    let reply = welcome(&[("a", 1, shm.0.len() as u64), ("b", 1, shm.0.len() as u64)]);
    let _server = stand_in(&socket, reply, vec![dup(&shm.0)]).await;
    let err = Client::connect(&socket, options()).await.unwrap_err();
    assert!(matches!(&err, Error::Protocol(m) if m.contains("2 arenas but carried 1 fds")), "{err}");
}

#[tokio::test]
async fn an_arena_that_doesnt_match_its_description_fails_the_connect() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("s.sock");
    let shm = make_arena(1);
    let reply = welcome(&[("hot", 2, shm.0.len() as u64)]); // the arena's own generation is 1
    let _server = stand_in(&socket, reply, vec![dup(&shm.0)]).await;
    let err = Client::connect(&socket, options()).await.unwrap_err();
    assert!(matches!(&err, Error::Arena { name, reason } if name == "hot" && reason.contains("generation")), "{err}");
}

#[tokio::test]
async fn an_incompatible_schema_refusal_is_reported_with_its_code() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("s.sock");
    let reply = error_reply(fb::ErrorCode::IncompatibleSchema, "schema major 9; this broker speaks 0");
    let _server = stand_in(&socket, reply, vec![]).await;
    let err = Client::connect(&socket, options()).await.unwrap_err();
    assert_eq!(err.broker_code(), Some(fb::ErrorCode::IncompatibleSchema), "{err}");
}

fn ok_welcome() -> Vec<u8> {
    welcome(&[])
}

fn socket_in(dir: &tempfile::TempDir) -> PathBuf {
    dir.path().join("s.sock")
}

#[tokio::test]
async fn replies_out_of_order_reach_their_own_requests() {
    let dir = tempfile::tempdir().unwrap();
    let socket = socket_in(&dir);
    // Read five heartbeats, then answer them newest first, each with its own queue depth as the "clock".
    let server = scripted(&socket, ok_welcome(), vec![], |mut s| async move {
        let mut got = Vec::new();
        while got.len() < 5 {
            let (id, kind, depth, _) = read_request(&mut s).await.unwrap();
            if kind == fb::ControlPayload::Heartbeat && depth > 0 {
                got.push((id, depth));
            }
        }
        for (id, depth) in got.into_iter().rev() {
            continuum_wire::write_frame(&mut s, &heartbeat_reply(id, u64::from(depth))).await.unwrap();
        }
        while read_request(&mut s).await.is_some() {}
    })
    .await;
    let c = std::sync::Arc::new(Client::connect(&socket, options()).await.unwrap());
    let tasks: Vec<_> = (1..=5u32)
        .map(|depth| {
            let c = c.clone();
            tokio::spawn(async move {
                let t = continuum_client::Telemetry { queue_depth: depth, ..Default::default() };
                (depth, c.heartbeat(Some(&t)).await.unwrap())
            })
        })
        .collect();
    for t in tasks {
        let (depth, answer) = t.await.unwrap();
        assert_eq!(answer, u64::from(depth), "each request got the reply to it, not another's");
    }
    drop(c);
    server.await.unwrap();
}

#[tokio::test]
async fn a_reply_after_its_request_timed_out_is_dropped_not_passed_on_as_an_event() {
    let dir = tempfile::tempdir().unwrap();
    let socket = socket_in(&dir);
    let server = scripted(&socket, ok_welcome(), vec![], |mut s| async move {
        let (id, _, _, _) = read_request(&mut s).await.unwrap();
        tokio::time::sleep(Duration::from_millis(400)).await; // past the client's 200 ms timeout
        continuum_wire::write_frame(&mut s, &heartbeat_reply(id, 1)).await.unwrap();
        continuum_wire::write_frame(&mut s, &heartbeat_reply(0, 42)).await.unwrap(); // a real event
        while read_request(&mut s).await.is_some() {}
    })
    .await;
    let opts = ClientOptions { request_timeout: Duration::from_millis(200), ..options() };
    let c = Client::connect(&socket, opts).await.unwrap();
    assert!(matches!(c.heartbeat(None).await, Err(Error::Timeout(_))));
    let event = tokio::time::timeout(Duration::from_secs(5), c.next_event()).await.unwrap().unwrap();
    let clock = event.envelope().payload_as_heartbeat().unwrap().now_ns();
    assert_eq!(clock, 42, "the first event is the broker's own message, not the late reply");
    drop(c);
    server.await.unwrap();
}

#[tokio::test]
async fn broker_messages_are_events_with_their_fds_and_end_when_the_broker_closes() {
    let dir = tempfile::tempdir().unwrap();
    let socket = socket_in(&dir);
    let shm = make_arena(1);
    let fd = dup(&shm.0);
    let server = scripted(&socket, ok_welcome(), vec![], |mut s| async move {
        // An announcement after Welcome, carrying an fd (as a new arena would).
        continuum_wire::write_frame_with_fds(&mut s, &heartbeat_reply(0, 7), &[fd.as_fd()]).await.unwrap();
        drop(s); // then the broker goes away
    })
    .await;
    let c = Client::connect(&socket, options()).await.unwrap();
    let mut event = c.next_event().await.unwrap();
    assert_eq!(event.kind(), fb::ControlPayload::Heartbeat);
    let fds = event.take_fds();
    assert_eq!(fds.len(), 1, "the fd came with the message, not lost to a plain read");
    let mapped = Arena::from_fd(fds[0].as_fd(), "announced").unwrap();
    assert_eq!(mapped.len(), shm.0.len());
    assert!(c.next_event().await.is_none(), "no more events once the broker closed");
    server.await.unwrap();
}

#[tokio::test]
async fn goodbye_reaches_the_broker_before_it_returns() {
    let dir = tempfile::tempdir().unwrap();
    let socket = socket_in(&dir);
    let (said, heard) = oneshot::channel();
    let server = scripted(&socket, ok_welcome(), vec![], |mut s| async move {
        while let Some((_, kind, _, reason)) = read_request(&mut s).await {
            if kind == fb::ControlPayload::Goodbye {
                let _ = said.send(reason);
                return;
            }
        }
    })
    .await;
    let c = Client::connect(&socket, options()).await.unwrap();
    c.goodbye("all done").await.unwrap();
    assert_eq!(heard.await.unwrap(), "all done");
    server.await.unwrap();
}

#[tokio::test]
async fn a_welcome_without_a_schema_version_fails_the_connect() {
    let dir = tempfile::tempdir().unwrap();
    let socket = socket_in(&dir);
    let _server = stand_in(&socket, welcome_without_schema(), vec![]).await;
    let err = Client::connect(&socket, options()).await.unwrap_err();
    assert!(matches!(&err, Error::Protocol(m) if m.contains("schema")), "{err}");
}

#[tokio::test]
async fn a_zero_heartbeat_interval_is_refused_up_front() {
    let opts = ClientOptions { heartbeat_interval: Duration::ZERO, ..options() };
    let err = Client::connect("/nonexistent/never.sock", opts).await.unwrap_err();
    assert!(matches!(&err, Error::Io(e) if e.kind() == std::io::ErrorKind::InvalidInput), "{err}");
}

#[tokio::test]
async fn an_abandoned_request_leaves_no_waiter_and_the_client_works_on() {
    let dir = tempfile::tempdir().unwrap();
    let socket = socket_in(&dir);
    // Read the first heartbeat and never answer it; answer the next one.
    let server = scripted(&socket, ok_welcome(), vec![], |mut s| async move {
        let _ignored = read_request(&mut s).await.unwrap();
        let (id, _, _, _) = read_request(&mut s).await.unwrap();
        continuum_wire::write_frame(&mut s, &heartbeat_reply(id, 5)).await.unwrap();
        while read_request(&mut s).await.is_some() {}
    })
    .await;
    let c = Client::connect(&socket, options()).await.unwrap();
    let abandoned = tokio::time::timeout(Duration::from_millis(100), c.heartbeat(None)).await;
    assert!(abandoned.is_err(), "the first request was cut off by the caller");
    assert_eq!(c.heartbeat(None).await.unwrap(), 5);
    assert_eq!(c.pending_requests(), 0, "the abandoned request's waiter is gone");
    drop(c);
    server.await.unwrap();
}

// ---- Ingest ----

/// A UUIDv7 from a counter: unique within the test run, with the version and variant bits set.
fn v7(n: u64) -> continuum_core::Uuid {
    let mut b = [0u8; 16];
    b[..6].copy_from_slice(&(now_ns() / 1_000_000).to_be_bytes()[2..]);
    b[8..].copy_from_slice(&n.to_be_bytes());
    b[6] = 0x70;
    b[8] = 0x80 | (b[8] & 0x3f);
    continuum_core::Uuid(b)
}

fn header(n: u64, stream: u32) -> continuum_core::HeaderBuf {
    let start = now_ns();
    continuum_core::HeaderBuf::build(&continuum_core::HeaderSpec::raw(
        v7(n),
        continuum_core::StreamId(stream),
        start,
        start,
    ))
    .unwrap()
}

/// The next event within `wait`, or None.
async fn event_within(c: &Client, wait: Duration) -> Option<continuum_client::Message> {
    tokio::time::timeout(wait, c.next_event()).await.ok().flatten()
}

#[tokio::test]
async fn an_ingest_on_a_registered_stream_is_accepted_without_a_reply() {
    let b = broker(Duration::from_secs(5)).await;
    let c = Client::connect(&b.socket, options()).await.unwrap();
    let stream = c.register_stream(&camera("cam0.video", 30.0)).await.unwrap().stream_id;
    c.ingest(&header(1, stream), None).unwrap();
    let inline = continuum_core::StorageLocator::Inline { bytes: b"a small payload".to_vec() };
    c.ingest(&header(2, stream), Some(&inline)).unwrap();
    // A sentinel the broker's core refuses (an unregistered stream), sent after the good ones: the core takes notices
    // in order, so if it had refused either good one, that rejection would come first. A refusal from the catalog
    // writer comes later and asynchronously, so after the sentinel the test also waits a while for any other event.
    let sentinel = header(3, 4243);
    c.ingest(&sentinel, None).unwrap();
    let first = event_within(&c, Duration::from_secs(5)).await.expect("the sentinel's rejection");
    let rejected = first.envelope().payload_as_ingest_rejected().expect("an IngestRejected");
    let id = rejected.payload_id().map(continuum_core::Uuid::from);
    assert_eq!(id, Some(sentinel.id()), "the core refused neither good notice");
    assert!(event_within(&c, Duration::from_millis(500)).await.is_none(), "and the writer refused neither");
}

#[tokio::test]
async fn an_ingest_the_broker_refuses_comes_back_as_an_event() {
    let b = broker(Duration::from_secs(5)).await;
    let c = Client::connect(&b.socket, options()).await.unwrap();
    let h = header(4, 4242); // a stream this session never registered
    c.ingest(&h, None).unwrap();
    let event = event_within(&c, Duration::from_secs(5)).await.expect("a rejection");
    assert_eq!(event.kind(), fb::ControlPayload::IngestRejected);
    let envelope = event.envelope();
    assert_eq!(envelope.request_id(), 0);
    let r = envelope.payload_as_ingest_rejected().unwrap();
    assert_eq!(r.stream_id(), 4242);
    assert_eq!(r.payload_id().map(continuum_core::Uuid::from), Some(h.id()));
    assert!(r.reason().unwrap_or_default().contains("didn't register"), "{:?}", r.reason());
    c.heartbeat(None).await.unwrap(); // the session goes on
}

#[tokio::test]
async fn ingest_after_close_fails_at_once() {
    let b = broker(Duration::from_secs(5)).await;
    let c = Client::connect(&b.socket, options()).await.unwrap();
    c.close("done").await.unwrap();
    assert!(matches!(c.ingest(&header(5, 1), None), Err(Error::Closed)));
}

#[tokio::test]
async fn stream_revisions_preserve_unsupported_zero_and_full_width_values() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("streams.sock");
    let peer = scripted(&socket, welcome(&[]), vec![], |mut stream| async move {
        let (request_id, kind, _, _) = read_request(&mut stream).await.unwrap();
        assert_eq!(kind, fb::ControlPayload::ListStreams);
        let mut b = flatbuffers::FlatBufferBuilder::new();
        let rows: Vec<_> = [None, Some(0), Some(u64::MAX)]
            .into_iter()
            .map(|revision| {
                // None omits the appended field, exactly as an older broker's table does.
                fb::StreamDesc::create(
                    &mut b,
                    &fb::StreamDescArgs { invalidation_revision: revision, ..Default::default() },
                )
            })
            .collect();
        let rows = b.create_vector(&rows);
        let list = fb::StreamList::create(&mut b, &fb::StreamListArgs { streams: Some(rows) });
        let env = fb::ControlEnvelope::create(
            &mut b,
            &fb::ControlEnvelopeArgs {
                request_id,
                payload_type: fb::ControlPayload::StreamList,
                payload: Some(list.as_union_value()),
            },
        );
        b.finish(env, None);
        continuum_wire::write_frame_with_fds(&mut stream, b.finished_data(), &[]).await.unwrap();
        let _ = read_request(&mut stream).await;
    })
    .await;
    let c = Client::connect(&socket, options()).await.unwrap();
    let rows = c.list_streams().await.unwrap();
    assert_eq!(rows.iter().map(|s| s.invalidation_revision).collect::<Vec<_>>(), [None, Some(0), Some(u64::MAX)]);
    c.close("test complete").await.unwrap();
    peer.await.unwrap();
}
